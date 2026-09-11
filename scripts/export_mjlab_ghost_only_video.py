#!/usr/bin/env python3
"""Export a ghost/reference-only video for an MjLab tracking motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--motion-file", required=True, type=Path)
    parser.add_argument("--video-output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", default=1, type=int)
    parser.add_argument("--steps", default=None, type=int)
    parser.add_argument("--video-fps", default=None, type=float)
    parser.add_argument("--video-width", default=1280, type=int)
    parser.add_argument("--video-height", default=720, type=int)
    parser.add_argument("--metrics-output", default=None, type=Path)
    parser.add_argument("--keep-robot-visible", action="store_true")
    parser.add_argument(
        "--no-reference-visual-ground-align",
        dest="reference_visual_ground_align",
        action="store_false",
        default=True,
        help=(
            "Disable display-only ground alignment for the transparent reference "
            "ghost. Training/reference motion data are never modified."
        ),
    )
    return parser.parse_args()


def _to_uint8_frame(frame):
    import numpy as np
    import torch

    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype == np.uint8:
        return frame
    frame = np.asarray(frame)
    if frame.max(initial=0.0) <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _load_motion_info(motion_file: Path) -> tuple[int, float]:
    import numpy as np

    data = np.load(motion_file)
    if "joint_pos" not in data.files:
        raise KeyError(f"{motion_file} does not contain joint_pos")
    frame_count = int(data["joint_pos"].shape[0])
    fps = float(data["fps"][0]) if "fps" in data.files else 50.0
    return frame_count, fps


def motion_command_cfg_types() -> tuple[type, ...]:
    from mjlab.tasks.tracking.mdp import MotionCommandCfg as UpstreamMotionCommandCfg

    command_types: list[type] = [UpstreamMotionCommandCfg]
    try:
        from src.tasks.tracking.mdp import MotionCommandCfg as NativeMotionCommandCfg
    except ImportError:
        pass
    else:
        command_types.append(NativeMotionCommandCfg)
    return tuple(dict.fromkeys(command_types))


def _hide_robot_geoms(env) -> int:
    import mujoco

    model = env.unwrapped.sim.mj_model
    robot = env.unwrapped.scene["robot"]
    body_ids = set()
    robot_prefix = getattr(robot, "name", "robot")
    for body_name in robot.body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{robot_prefix}/{body_name}",
            )
        if body_id >= 0:
            body_ids.add(body_id)

    hidden = 0
    for geom_id, body_id in enumerate(model.geom_bodyid):
        if int(body_id) in body_ids:
            model.geom_rgba[geom_id, 3] = 0.0
            hidden += 1
    return hidden


def _sync_hidden_robot_to_reference(command) -> None:
    import torch

    env_ids = torch.arange(command.num_envs, device=command.device)
    root_state = torch.cat(
        [
            command.body_pos_w[:, 0],
            command.body_quat_w[:, 0],
            command.body_lin_vel_w[:, 0],
            command.body_ang_vel_w[:, 0],
        ],
        dim=-1,
    )
    command.robot.write_joint_state_to_sim(
        command.joint_pos.clone(),
        command.joint_vel.clone(),
        env_ids=env_ids,
    )
    command.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    command.robot.clear_state(env_ids=env_ids)


def main() -> int:
    args = parse_args()
    import torch
    import os

    if "MIMICX_DEVICE_ID" in os.environ:
        args.device = f"cuda:{int(os.environ['MIMICX_DEVICE_ID'])}"
    motion_file = args.motion_file.expanduser().resolve()
    if not motion_file.exists():
        raise FileNotFoundError(motion_file)

    motion_frames, motion_fps = _load_motion_info(motion_file)
    steps = args.steps if args.steps is not None else motion_frames
    video_fps = args.video_fps if args.video_fps is not None else motion_fps

    video_output = args.video_output.expanduser().resolve()
    metrics_output = args.metrics_output.expanduser().resolve() if args.metrics_output else None
    video_output.parent.mkdir(parents=True, exist_ok=True)
    if metrics_output is not None:
        metrics_output.parent.mkdir(parents=True, exist_ok=True)

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.viewer.width = args.video_width
    env_cfg.viewer.height = args.video_height
    env_cfg.terminations = {}

    motion_cmd = env_cfg.commands["motion"]
    if not isinstance(motion_cmd, motion_command_cfg_types()):
        raise TypeError(f"Task {args.task} does not expose a MotionCommand")
    motion_cmd.motion_file = str(motion_file)
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.debug_vis = True
    motion_cmd.viz.mode = "ghost"
    motion_cmd.viz.ground_align = args.reference_visual_ground_align

    print(f"task={args.task}")
    print(f"motion_file={motion_file}")
    print(f"motion_frames={motion_frames}")
    print(f"steps={steps}")
    print(f"video_output={video_output}")
    print(f"video_fps={video_fps}")
    print(f"viewer_size={args.video_width}x{args.video_height}")
    print(f"device={args.device}")
    print(f"reference_visual_ground_align={args.reference_visual_ground_align}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.get_observations()

    command = env.unwrapped.command_manager.get_term("motion")
    command.time_steps[:] = 0
    hidden_geom_count = 0
    if not args.keep_robot_visible:
        hidden_geom_count = _hide_robot_geoms(env)
    print(f"hidden_robot_geom_count={hidden_geom_count}")

    zero_actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    frames = []
    for index in range(steps):
        _sync_hidden_robot_to_reference(command)
        frames.append(_to_uint8_frame(env.unwrapped.render()))
        if index + 1 < steps:
            env.step(zero_actions)
        if index in {0, steps - 1}:
            print(f"frame={index + 1}/{steps} time_step={int(command.time_steps[0].detach().cpu())}")

    env.close()

    import mediapy as media

    media.write_video(str(video_output), frames, fps=video_fps)
    if metrics_output is not None:
        metrics = {
            "task": args.task,
            "motion_file": str(motion_file),
            "video_output": str(video_output),
            "motion_frames": motion_frames,
            "exported_frames": steps,
            "video_fps": video_fps,
            "video_width": args.video_width,
            "video_height": args.video_height,
            "num_envs": args.num_envs,
            "device": args.device,
            "hidden_robot_geom_count": hidden_geom_count,
            "ghost_only": not args.keep_robot_visible,
            "reference_visual_ground_align": args.reference_visual_ground_align,
            "ghost_ground_z_offset": getattr(command, "_ghost_ground_z_offset", None),
        }
        metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"metrics_output={metrics_output}")
    print("ghost_video_export_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
