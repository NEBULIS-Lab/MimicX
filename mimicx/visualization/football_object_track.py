from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from mimicx.visualization.racket_calibration import quaternion_matrix_wxyz


SCHEMA = "mimicx.football-object-track.v1"


@dataclass(frozen=True)
class FootballTrackFrame:
    frame: int
    relative_lateral_person_heights: float
    height_person_heights: float
    detected: bool = True
    confidence: float = 1.0


@dataclass(frozen=True)
class FootballObjectTrack:
    source_video: str
    source_fps: float
    source_frame_count: int
    replay_fps: float
    replay_frame_count: int
    person_height_m: float
    ball_radius_m: float
    activity: str
    frames: tuple[FootballTrackFrame, ...]


def relative_ball_measurement(
    *,
    ball_center_xy: tuple[float, float],
    person_box_xyxy: tuple[float, float, float, float],
) -> tuple[float, float]:
    left, top, right, bottom = (float(value) for value in person_box_xyxy)
    person_height = bottom - top
    if person_height <= 0:
        raise ValueError("person box must have positive height")
    person_center_x = 0.5 * (left + right)
    ball_x, ball_y = (float(value) for value in ball_center_xy)
    return (ball_x - person_center_x) / person_height, (bottom - ball_y) / person_height


def fill_missing_measurements(
    lateral: np.ndarray,
    height: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(lateral, dtype=np.float64), np.asarray(height, dtype=np.float64)]
    if arrays[0].shape != arrays[1].shape or arrays[0].ndim != 1:
        raise ValueError("track measurements must be matching one-dimensional arrays")
    output = []
    indices = np.arange(len(arrays[0]), dtype=np.float64)
    for values in arrays:
        valid = np.isfinite(values)
        if not valid.any():
            raise ValueError("at least one detected measurement is required")
        output.append(np.interp(indices, indices[valid], values[valid]))
    return output[0], output[1]


def load_football_track(path: str | Path) -> FootballObjectTrack:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported football track schema: {payload.get('schema')}")
    frames = tuple(
        FootballTrackFrame(
            frame=int(row["frame"]),
            relative_lateral_person_heights=float(row["relative_lateral_person_heights"]),
            height_person_heights=float(row["height_person_heights"]),
            detected=bool(row.get("detected", True)),
            confidence=float(row.get("confidence", 1.0)),
        )
        for row in payload["frames"]
    )
    source_count = int(payload["source_frame_count"])
    if len(frames) != source_count or [row.frame for row in frames] != list(range(source_count)):
        raise ValueError("football track frames must be contiguous and cover the source video")
    return FootballObjectTrack(
        source_video=str(payload["source_video"]),
        source_fps=float(payload["source_fps"]),
        source_frame_count=source_count,
        replay_fps=float(payload["replay_fps"]),
        replay_frame_count=int(payload["replay_frame_count"]),
        person_height_m=float(payload.get("person_height_m", 1.75)),
        ball_radius_m=float(payload.get("ball_radius_m", 0.11)),
        activity=str(payload.get("activity", "")),
        frames=frames,
    )


def replay_step_to_video_frame(replay_index: int, *, replay_count: int, video_count: int) -> int:
    if replay_count < 2 or video_count < 2:
        raise ValueError("replay_count and video_count must be at least two")
    if replay_index < 0 or replay_index >= replay_count:
        raise IndexError("replay index outside track horizon")
    return int(round(replay_index * (video_count - 1) / (replay_count - 1)))


def tracked_ball_world_position(
    track: FootballObjectTrack,
    *,
    replay_index: int,
    pelvis_pose: np.ndarray,
    surface_z: float,
) -> np.ndarray:
    pose = np.asarray(pelvis_pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError("pelvis_pose must contain xyz and wxyz quaternion")
    source_index = replay_step_to_video_frame(
        replay_index,
        replay_count=track.replay_frame_count,
        video_count=track.source_frame_count,
    )
    frame = track.frames[source_index]
    rotation = quaternion_matrix_wxyz(pose[3:])
    forward = rotation @ np.asarray([1.0, 0.0, 0.0])
    lateral = rotation @ np.asarray([0.0, 1.0, 0.0])
    forward[2] = 0.0
    lateral[2] = 0.0
    forward /= np.linalg.norm(forward)
    lateral /= np.linalg.norm(lateral)
    position = pose[:3].copy() + 0.35 * forward
    position += frame.relative_lateral_person_heights * track.person_height_m * lateral
    position[2] = surface_z + track.ball_radius_m + max(
        0.0, frame.height_person_heights * track.person_height_m
    )
    return position
