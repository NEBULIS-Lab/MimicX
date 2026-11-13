"""Read-only CPU audit of motion horizons, not a physical success scorer.

Reference NPZ: positive scalar fps/frame_rate and at least one position array
(joint_pos/dof_pos [T,J], body_pos_w/body_pos [T,B,3], root_pos [T,3]).
All recognized trajectories must be finite and agree on T >= 2. Duration is
(T-1)/fps, as in the existing common scorer, not T/fps or video duration.

Rollout NPZ (optional): row-synchronized timestamps in seconds, zero-based
reference_frame (or reference_frames), reset/failure event flags, valid_mask,
and/or native_failure_frame (a *sample index*, not a reference frame). Keys
can be explicitly mapped by the caller. timestamp_kind must be declared as
wall (measured monotonic/epoch seconds) or simulation (actual simulator seconds),
in the NPZ or API. Either clock can verify continuous source progress. A
simulation-clock audit does not require execution at wall-clock playback speed;
real_time_playback_verified remains unknown without an audited wall clock.
Never synthesize timestamps or frame IDs from row numbers, fps, or poses.
Missing evidence is reported as unknown. A reset/failure sample and its entire
tail are censored, matching native common scoring; flags at initialization
must therefore be excluded upstream. valid_mask must be prefix-only.

Consecutive held frame IDs are allowed (e.g. oversampled telemetry) but add no
reference coverage. Backtracking, skipped frames and excessive sample gaps
break continuity. No interpolation, sorting, time warping or episode stitching
is performed. Clock tolerances default to the existing common protocol.
Stage boundaries are explicit inclusive source-frame endpoints; adjacent
stages share one boundary frame and must partition the scoring window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mimicx.datasets.motion_study import sha256, timing_check
from mimicx.experiments.motion_study_scoring import common_protocol
from mimicx.experiments.sonic_study import clocks


SCHEMA = "mimicx.horizon-audit.v1"
_POSITIONS = ("joint_pos", "dof_pos", "body_pos_w", "body_pos", "root_pos")
_SHAPES = {
    "joint_pos": (2, None), "dof_pos": (2, None),
    "joint_vel": (2, None), "dof_vel": (2, None),
    "body_pos_w": (3, 3), "body_pos": (3, 3),
    "body_quat_w": (3, 4), "body_quat": (3, 4),
    "body_lin_vel_w": (3, 3), "body_lin_vel": (3, 3),
    "body_ang_vel_w": (3, 3), "body_ang_vel": (3, 3),
    "root_pos": (2, 3), "root_rot": (2, 4),
}
_DEFAULT_PROTOCOL = common_protocol()


@dataclass(frozen=True)
class Stage:
    name: str
    start_frame: int
    end_frame_inclusive: int


def _numeric(value, label):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite numeric data")
    return array


def _positive(value, label):
    array = _numeric(value, label)
    if array.size != 1 or float(array.item()) <= 0:
        raise ValueError(f"{label} must be a positive scalar")
    return float(array.item())


def _integer(value, label):
    array = _numeric(value, label)
    if array.ndim != 0 or float(array) != int(array):
        raise ValueError(f"{label} must be an integer scalar")
    return int(array)


def _flags(value, label):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "biuf" or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{label} must be a one-dimensional boolean/0/1 vector")
    return array.astype(bool)


def _archive(path, label):
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"{label} must be an NPZ archive, not a standalone array")
    return archive


def _reference(path):
    with _archive(path, "reference") as archive:
        rate_keys = [key for key in ("fps", "frame_rate") if key in archive]
        if not rate_keys:
            raise ValueError("reference requires explicit fps or frame_rate")
        rates = [_positive(archive[key], key) for key in rate_keys]
        if any(not np.isclose(rate, rates[0], atol=1e-9, rtol=0) for rate in rates):
            raise ValueError("conflicting fps/frame_rate")
        if not any(key in archive for key in _POSITIONS):
            raise ValueError("reference requires a supported position trajectory")
        shapes = {}
        frames = None
        for key, (ndim, width) in _SHAPES.items():
            if key not in archive:
                continue
            array = _numeric(archive[key], key)
            if (array.ndim != ndim or any(size == 0 for size in array.shape)
                    or (width is not None and array.shape[-1] != width)):
                raise ValueError(f"invalid reference shape: {key}")
            if frames is not None and len(array) != frames:
                raise ValueError("reference trajectory frame counts disagree")
            frames = len(array)
            shapes[key] = list(array.shape)
        for left, right in (("joint_pos", "dof_pos"), ("body_pos_w", "body_pos")):
            if left in archive and right in archive and not np.array_equal(archive[left], archive[right]):
                raise ValueError(f"conflicting reference aliases: {left}/{right}")
        if frames < 2:
            raise ValueError("reference requires at least two frames")
        if "timestamps" in archive:
            ts = clocks(_numeric(archive["timestamps"], "reference timestamps"), "reference")
            if ts.shape != (frames,) or not np.allclose(
                    ts - ts[0], np.arange(frames) / rates[0], atol=1e-6, rtol=0):
                raise ValueError("reference timestamps disagree with fps/frame_rate")
    duration = timing_check(frames, rates[0], frames, rates[0])["source_duration"]
    if not np.isfinite(duration):
        raise ValueError("reference duration must be finite")
    return dict(frames=frames, fps=rates[0], duration_seconds=duration,
                duration_convention="inclusive endpoints: (frames - 1) / fps",
                trajectory_shapes=shapes, sha256=sha256(Path(path)))


def _rollout(path, *, timestamp_key, timestamp_kind, reference_frame_key, reset_key, failure_key):
    keys = [timestamp_key, reference_frame_key, reset_key, failure_key, "valid_mask", "native_failure_frame"]
    if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("rollout key mappings must be distinct nonempty strings")
    with _archive(path, "rollout") as archive:
        stored_kind = None
        if "timestamp_kind" in archive:
            kind = np.asarray(archive["timestamp_kind"])
            if kind.ndim != 0 or kind.dtype.kind != "U":
                raise ValueError("timestamp_kind must be a scalar Unicode string")
            stored_kind = str(kind.item())
        if timestamp_kind is not None and stored_kind is not None and timestamp_kind != stored_kind:
            raise ValueError("timestamp_kind contradicts rollout metadata")
        kind = timestamp_kind if timestamp_kind is not None else stored_kind
        if kind is not None and kind not in ("wall", "simulation"):
            raise ValueError("timestamp_kind must be wall or simulation")
        result = {"timestamp_kind": kind}
        vectors = {}
        if timestamp_key in archive:
            if kind is None:
                raise ValueError("declare timestamp_kind; timestamps do not identify simulation vs wall clock")
            vectors["timestamps"] = clocks(_numeric(archive[timestamp_key], timestamp_key), "rollout")
            with np.errstate(over="ignore"):
                if not np.isfinite(vectors["timestamps"] - vectors["timestamps"][0]).all():
                    raise ValueError("rollout elapsed time must be finite")
        if reference_frame_key in archive:
            array = _numeric(archive[reference_frame_key], reference_frame_key)
            if array.ndim != 1 or np.any(array != np.floor(array)):
                raise ValueError("reference_frame must be a one-dimensional integer vector")
            vectors["reference_frame"] = array
        if reference_frame_key == "reference_frame" and "reference_frames" in archive:
            other = _numeric(archive["reference_frames"], "reference_frames")
            if "reference_frame" in vectors and not np.array_equal(vectors["reference_frame"], other):
                raise ValueError("conflicting reference_frame/reference_frames")
            if other.ndim != 1 or np.any(other != np.floor(other)):
                raise ValueError("reference_frames must be a one-dimensional integer vector")
            vectors["reference_frame"] = other
            if reference_frame_key not in archive:
                reference_frame_key = "reference_frames"
        for dest, key in (("reset", reset_key), ("failure", failure_key), ("valid_mask", "valid_mask")):
            if key in archive:
                vectors[dest] = _flags(archive[key], key)
        sizes = {len(value) for value in vectors.values()}
        if len(sizes) != 1 or next(iter(sizes), 0) < 2:
            raise ValueError("rollout requires matching vectors with at least two samples")
        count = sizes.pop()
        if "valid_mask" in vectors and np.any(np.diff(vectors["valid_mask"].astype(int)) > 0):
            raise ValueError("valid_mask must be prefix-only; no postreset reentry")
        if "native_failure_frame" in archive:
            index = _integer(archive["native_failure_frame"], "native_failure_frame")
            if not 0 <= index < count:
                raise ValueError("native_failure_frame must be a rollout sample index in range")
            native = np.arange(count) == index
            vectors["failure"] = vectors.get("failure", np.zeros(count, dtype=bool)) | native
    result.update(vectors)
    result["samples"] = count
    result["sha256"] = sha256(Path(path))
    result["keys"] = dict(timestamps=timestamp_key, reference_frame=reference_frame_key,
                          reset=reset_key, failure=failure_key)
    return result


def _window(start, end, fps):
    return dict(start_frame=start, end_frame_inclusive=end, frames=end - start + 1,
                start_seconds=start / fps, end_seconds=end / fps,
                duration_seconds=(end - start) / fps)


def _stages(stages, start, end):
    stages = list(stages or [])
    names = set()
    previous = start
    normalized = []
    for stage in stages:
        if not isinstance(stage, Stage) or not isinstance(stage.name, str) or not stage.name.strip():
            raise ValueError("stages must be Stage objects with nonempty names")
        left = _integer(stage.start_frame, "stage start_frame")
        right = _integer(stage.end_frame_inclusive, "stage end_frame_inclusive")
        if stage.name in names or left != previous or not left < right <= end:
            raise ValueError("stages must have unique names and partition the window with shared endpoints")
        names.add(stage.name)
        previous = right
        normalized.append(Stage(str(stage.name), left, right))
    if stages and previous != end:
        raise ValueError("stages must partition the entire scoring window")
    return normalized


def _segments(frames, timestamps, valid_count, gap):
    if not valid_count:
        return []
    delta = np.diff(frames[:valid_count])
    broken = (delta < 0) | (delta > 1)
    if timestamps is not None:
        broken |= np.diff(timestamps[:valid_count]) > gap + 1e-9
    bounds = [0, *(np.flatnonzero(broken) + 1).tolist(), valid_count]
    return list(zip(bounds[:-1], bounds[1:]))


def _coverage(frames, timestamps, segments, valid_count, start, end, fps, tolerance, ratio_bounds):
    inside = (frames >= start) & (frames <= end)
    raw = np.flatnonzero(inside)
    first = int(raw[0]) if raw.size else None
    spans = []
    for low, high in segments:
        # IDs are monotone within each segment, so clipping cannot join gaps.
        ids = frames[low:high]
        left = low + int(np.searchsorted(ids, start, side="left"))
        right = low + int(np.searchsorted(ids, end, side="right"))
        if left >= right:
            continue
        # Trailing holds/recording after first reaching an endpoint add no horizon.
        right = left + int(np.searchsorted(frames[left:right], frames[right - 1], side="left")) + 1
        a, b = int(frames[left]), int(frames[right - 1])
        span = dict(start_sample=left, end_sample_inclusive=right - 1,
                    start_frame=a, end_frame_inclusive=b, unique_frames=b - a + 1,
                    reference_seconds=(b - a) / fps, elapsed_seconds=None,
                    duration_ratio=None, max_clock_error_seconds=None, clock_matches_reference=None)
        if timestamps is not None:
            elapsed = timestamps[left:right] - timestamps[left]
            nominal = (frames[left:right] - a) / fps
            error = float(np.max(np.abs(elapsed - nominal)))
            with np.errstate(over="ignore"):
                ratio = float(elapsed[-1] / nominal[-1]) if nominal[-1] else None
            if ratio is not None and not np.isfinite(ratio):
                raise ValueError("rollout/reference duration ratio must be finite")
            span.update(elapsed_seconds=float(elapsed[-1]), duration_ratio=ratio,
                        max_clock_error_seconds=error,
                        clock_matches_reference=bool(ratio is not None and error <= tolerance + 1e-9
                            and ratio_bounds[0] - 1e-9 <= ratio <= ratio_bounds[1] + 1e-9))
        spans.append(span)
    prefix = next((span for span in spans if span["start_sample"] == first
                   and span["start_frame"] == start), None)
    longest = max(spans, key=lambda span: span["unique_frames"], default=None)
    requested = end - start + 1
    prefix_count = prefix["unique_frames"] if prefix else 0
    return dict(observed_unique_frames=int(np.unique(frames[inside]).size),
                valid_unique_frames=int(np.unique(frames[:valid_count][inside[:valid_count]]).size),
                requested_frames=requested, continuous_segment_count=len(spans),
                any_segment_complete=bool(longest and longest["unique_frames"] == requested),
                prefix_unique_frames=prefix_count, prefix_fraction=prefix_count / requested,
                continuous_window_complete=prefix_count == requested,
                prefix=prefix, longest_segment=longest)


def audit_motion_horizon(
    reference: str | Path,
    rollout: str | Path | None = None,
    *,
    scoring_window: tuple[int, int] | None = None,
    stages: Sequence[Stage] | None = None,
    minimum_duration_s: float | None = None,
    timestamp_kind: str | None = None,
    timestamp_key: str = "timestamps",
    reference_frame_key: str = "reference_frame",
    reset_key: str = "reset",
    failure_key: str = "failure",
    clock_tolerance_s: float = _DEFAULT_PROTOCOL["clock_tolerance_s"],
    state_gap_s: float = _DEFAULT_PROTOCOL["state_gap_s"],
    duration_ratio_bounds: tuple[float, float] = (.97, 1.03),
) -> dict:
    """Return JSON-safe evidence, with no machine paths or success inference.

    ``horizon_verified`` checks the complete continuous scoring prefix on the
    declared actual simulation OR wall clock (``clock_basis``), with explicit
    reset AND failure telemetry and the requested minimum duration. Simulator
    progress need not run at wall speed. ``real_time_playback_verified`` applies
    the same audit only when wall timestamps were selected; otherwise it is
    None, not a missing prerequisite for simulation-clock horizon verification.
    Results are False for measured violations, None for missing evidence. Even
    True never certifies tracking quality or task success. Neither timestamps
    nor reference frame IDs are synthesized from fps, row indices or poses.
    Stage results include ``covered_in_scoring_prefix`` to distinguish a true
    ordered multi-stage execution from stages seen across separate segments.
    """
    source = _reference(reference)
    fps, total = source["fps"], source["frames"]
    if scoring_window is None:
        start, end = 0, total - 1
    else:
        if len(scoring_window) != 2:
            raise ValueError("scoring_window requires two inclusive source-frame endpoints")
        start, end = [_integer(value, "scoring_window endpoint") for value in scoring_window]
        if not 0 <= start < end < total:
            raise ValueError("scoring_window outside source or empty")
    stages = _stages(stages, start, end)
    tolerance = _positive(clock_tolerance_s, "clock_tolerance_s")
    gap = _positive(state_gap_s, "state_gap_s")
    bounds = _numeric(duration_ratio_bounds, "duration_ratio_bounds")
    if bounds.shape != (2,) or not 0 < bounds[0] <= 1 <= bounds[1]:
        raise ValueError("duration_ratio_bounds must be positive ordered bounds containing 1")
    minimum = None if minimum_duration_s is None else _positive(minimum_duration_s, "minimum_duration_s")
    window = _window(start, end, fps)
    meets_minimum = minimum is None or window["duration_seconds"] + 1e-9 >= minimum
    result = dict(schema=SCHEMA, source=source, scoring_window=window,
                  minimum_duration_seconds=minimum, scoring_meets_minimum=meets_minimum,
                  protocol=dict(clock_tolerance_s=tolerance, state_gap_s=gap,
                                duration_ratio_bounds=bounds.tolist(),
                                censoring="first reset/failure/invalid sample and entire tail"),
                  stages=[dict(name=s.name, **_window(s.start_frame, s.end_frame_inclusive, fps),
                               coverage=None, covered_in_scoring_prefix=None) for s in stages],
                  rollout=None, clock_basis=None, horizon_verified=None,
                  multi_stage_horizon_verified=None, real_time_playback_verified=None,
                  success=None, success_basis="Requires a separate physical/task scorer; recording is not success",
                  missing_evidence=[])
    if not meets_minimum:
        result["horizon_verified"] = False
    if rollout is None:
        result["missing_evidence"] = ["actual rollout"]
        if len(stages) >= 2:
            result["multi_stage_horizon_verified"] = result["horizon_verified"]
        return result
    data = _rollout(rollout, timestamp_key=timestamp_key, timestamp_kind=timestamp_kind,
                    reference_frame_key=reference_frame_key, reset_key=reset_key, failure_key=failure_key)
    count = data["samples"]
    times, frames = data.get("timestamps"), data.get("reference_frame")
    result["clock_basis"] = data["timestamp_kind"] if times is not None else None
    if frames is not None:
        if np.any((frames < 0) | (frames >= total)):
            raise ValueError("reference_frame outside source; no modulo or clamping allowed")
        frames = frames.astype(np.int64)
    bad = np.zeros(count, dtype=bool)
    for key in ("reset", "failure"):
        if key in data:
            bad |= data[key]
    if "valid_mask" in data:
        bad |= ~data["valid_mask"]
    invalid = np.flatnonzero(bad)
    valid_count = int(invalid[0]) if invalid.size else count
    telemetry = dict(samples=count, sha256=data["sha256"], keys=data["keys"],
                     timestamp_kind=data["timestamp_kind"],
                     elapsed_seconds=float(times[-1] - times[0]) if times is not None else None,
                     wall_elapsed_seconds=float(times[-1] - times[0])
                         if times is not None and data["timestamp_kind"] == "wall" else None,
                     reset_flagged_samples=int(data["reset"].sum()) if "reset" in data else None,
                     failure_flagged_samples=int(data["failure"].sum()) if "failure" in data else None,
                     evidence_present={key: key in data for key in ("timestamps", "reference_frame", "reset", "failure", "valid_mask")},
                     first_censored_sample=valid_count if invalid.size else None,
                     valid_prefix_samples=valid_count, source_coverage=None, scoring_coverage=None,
                     repeated_frame_steps=None, backward_steps=None, wrap_steps=None,
                     skipped_frame_steps=None, revisited_frame_samples=None,
                     sample_gap_count=int(np.count_nonzero(np.diff(times) > gap + 1e-9))
                         if times is not None else None)
    result["rollout"] = telemetry
    missing = result["missing_evidence"]
    if times is None:
        missing.append("timestamps in seconds")
    for key in ("reference_frame", "reset", "failure"):
        if key not in data:
            missing.append(key)
    if frames is not None:
        delta = np.diff(frames)
        repeated = int(np.count_nonzero(delta == 0))
        telemetry.update(repeated_frame_steps=repeated, backward_steps=int(np.count_nonzero(delta < 0)),
                         wrap_steps=int(np.count_nonzero((frames[:-1] == total - 1) & (frames[1:] == 0))),
                         skipped_frame_steps=int(np.count_nonzero(delta > 1)),
                         revisited_frame_samples=int(count - np.unique(frames).size - repeated))
        segments = _segments(frames, times, valid_count, gap)

        def coverage(a, b):
            return _coverage(frames, times, segments, valid_count, a, b, fps, tolerance, bounds)

        telemetry["source_coverage"] = coverage(0, total - 1)
        scored = telemetry["scoring_coverage"] = coverage(start, end)
        prefix = scored["prefix"]
        for stage in result["stages"]:
            stage["coverage"] = coverage(stage["start_frame"], stage["end_frame_inclusive"])
            stage["covered_in_scoring_prefix"] = bool(prefix and prefix["end_frame_inclusive"] >= stage["end_frame_inclusive"])
        if not scored["continuous_window_complete"] or (prefix and prefix["clock_matches_reference"] is False):
            result["horizon_verified"] = False
        elif not missing and meets_minimum:
            result["horizon_verified"] = True
    if len(stages) >= 2:
        result["multi_stage_horizon_verified"] = result["horizon_verified"]
    if result["clock_basis"] == "wall":
        result["real_time_playback_verified"] = result["horizon_verified"]
    return result
