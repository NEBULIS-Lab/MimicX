#!/usr/bin/env python3
"""Headless Unitree RL Mjlab tracking rollout with optional visual export."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_rgba(value: str) -> tuple[float, float, float, float]:
    try:
        rgba = tuple(float(component.strip()) for component in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("RGBA must contain four numeric values") from error
    if len(rgba) != 4 or any(component < 0.0 or component > 1.0 for component in rgba):
        raise argparse.ArgumentTypeError("RGBA must contain four values in [0, 1]")
    return rgba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Unitree-G1-Tracking-No-State-Estimation")
    parser.add_argument("--motion-file", required=True, type=Path)
    parser.add_argument("--num-envs", default=4, type=int)
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-terminations", action="store_true")
    parser.add_argument("--video-output", default=None, type=Path)
    parser.add_argument("--video-fps", default=50.0, type=float)
    parser.add_argument("--video-width", default=None, type=int)
    parser.add_argument("--video-height", default=None, type=int)
    parser.add_argument("--render-every", default=1, type=int)
    parser.add_argument("--screenshot-dir", default=None, type=Path)
    parser.add_argument("--screenshot-every", default=0, type=int)
    parser.add_argument("--metrics-output", default=None, type=Path)
    parser.add_argument("--step-metrics-output", default=None, type=Path)
    parser.add_argument(
        "--replay-output",
        default=None,
        type=Path,
        help="Optional directory for exact robot/reference states used by offline publication rendering.",
    )
    parser.add_argument(
        "--hide-reference-visual",
        action="store_true",
        help=(
            "Render the rollout without the motion command reference/ghost debug "
            "visualization. This affects visual export only."
        ),
    )
    parser.add_argument(
        "--ghost-color",
        type=_parse_rgba,
        default=None,
        metavar="R,G,B,A",
        help="Override the transparent reference robot RGBA for visual export.",
    )
    parser.add_argument(
        "--no-reference-visual-ground-align",
        dest="reference_visual_ground_align",
        action="store_false",
        default=True,
        help=(
            "Disable display-only ground alignment for the transparent reference "
            "ghost. This never changes the motion command used by rewards/training."
        ),
    )
    parser.add_argument(
        "--disable-joint-init-noise",
        action="store_true",
        help="Diagnostic only: set MotionCommand joint_position_range to (0, 0).",
    )
    parser.add_argument(
        "--feet-only-ee-termination",
        action="store_true",
        help=(
            "Diagnostic only: keep ee_body_pos termination for ankle bodies only, "
            "while leaving rewards and reference commands unchanged."
        ),
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        type=Path,
        help="Optional RSL-RL checkpoint. If omitted, a zero-action policy is used.",
    )
    parser.add_argument(
        "--reference-joint-policy",
        action="store_true",
        help=(
            "Diagnostic only: drive the action target directly toward the motion "
            "reference joint positions instead of using a learned policy."
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


def _metric_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name)


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


def main() -> int:
    args = parse_args()
    import torch

    if "MIMICX_DEVICE_ID" in os.environ:
        args.device = f"cuda:{int(os.environ['MIMICX_DEVICE_ID'])}"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    motion_file = args.motion_file.expanduser().resolve()
    if not motion_file.exists():
        raise FileNotFoundError(motion_file)
    if args.render_every < 1:
        raise ValueError("--render-every must be >= 1")

    video_output = args.video_output.expanduser().resolve() if args.video_output else None
    screenshot_dir = args.screenshot_dir.expanduser().resolve() if args.screenshot_dir else None
    metrics_output = args.metrics_output.expanduser().resolve() if args.metrics_output else None
    step_metrics_output = (
        args.step_metrics_output.expanduser().resolve()
        if args.step_metrics_output
        else None
    )
    replay_output = args.replay_output.expanduser().resolve() if args.replay_output else None
    if video_output is not None:
        video_output.parent.mkdir(parents=True, exist_ok=True)
    if screenshot_dir is not None:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
    if metrics_output is not None:
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
    if step_metrics_output is not None:
        step_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    if replay_output is not None:
        replay_output.mkdir(parents=True, exist_ok=True)

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    if hasattr(agent_cfg, "seed"):
        agent_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    if args.video_width is not None:
        env_cfg.viewer.width = args.video_width
    if args.video_height is not None:
        env_cfg.viewer.height = args.video_height
    if args.disable_terminations:
        env_cfg.terminations = {}

    motion_cmd = env_cfg.commands["motion"]
    if not isinstance(motion_cmd, motion_command_cfg_types()):
        raise TypeError(f"Task {args.task} does not expose a motion command")
    motion_cmd.motion_file = str(motion_file)
    if args.hide_reference_visual:
        motion_cmd.debug_vis = False
    else:
        motion_cmd.viz.ground_align = args.reference_visual_ground_align
        if args.ghost_color is not None:
            motion_cmd.viz.ghost_color = args.ghost_color
    if args.disable_joint_init_noise:
        motion_cmd.joint_position_range = (0.0, 0.0)
    if args.feet_only_ee_termination and "ee_body_pos" in env_cfg.terminations:
        env_cfg.terminations["ee_body_pos"].params["body_names"] = (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        )

    print(f"task={args.task}")
    print(f"motion_file={motion_cmd.motion_file}")
    print(f"num_envs={env_cfg.scene.num_envs}")
    print(f"device={args.device}")
    print(f"seed={args.seed}")
    print(f"disable_joint_init_noise={args.disable_joint_init_noise}")
    print(f"feet_only_ee_termination={args.feet_only_ee_termination}")
    print(f"hide_reference_visual={args.hide_reference_visual}")
    print(f"ghost_color={None if args.hide_reference_visual else motion_cmd.viz.ghost_color}")
    print(f"reference_visual_ground_align={not args.hide_reference_visual and args.reference_visual_ground_align}")
    print(f"render={bool(video_output or screenshot_dir)}")
    if args.video_width or args.video_height:
        print(f"viewer_size={env_cfg.viewer.width}x{env_cfg.viewer.height}")

    render_mode = "rgb_array" if video_output or screenshot_dir else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=render_mode)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    obs = env.get_observations()
    print(f"obs_groups={list(obs.keys())}")
    print(f"action_space={env.unwrapped.action_space.shape}")

    policy = None
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    if args.reference_joint_policy:
        if args.checkpoint_file is not None:
            print("policy=reference_joint checkpoint_ignored=True")
        else:
            print("policy=reference_joint")
    elif args.checkpoint_file is not None:
        checkpoint_file = args.checkpoint_file.expanduser().resolve()
        if not checkpoint_file.exists():
            raise FileNotFoundError(checkpoint_file)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        runner.load(
            str(checkpoint_file),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
        print(f"policy=checkpoint checkpoint_file={checkpoint_file}")
    else:
        print("policy=zero")

    zero_actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    last_rew = None
    last_done = None
    frames = []
    reward_trace = []
    done_trace = []
    screenshot_steps = {0, max(0, args.steps // 2 - 1), args.steps - 1}
    if args.screenshot_every > 0:
        screenshot_steps.update(range(0, args.steps, args.screenshot_every))
    step_rows = []
    replay_rows = {
        "robot_root_pose": [],
        "robot_joint_positions": [],
        "robot_body_pose": [],
        "reference_root_pose": [],
        "reference_joint_positions": [],
        "reference_body_pose": [],
    }
    command = None
    ee_body_names: tuple[str, ...] = ()
    if "motion" in env.unwrapped.command_manager.active_terms:
        command = env.unwrapped.command_manager.get_term("motion")
    if command is not None and "ee_body_pos" in env.unwrapped.termination_manager.active_terms:
        try:
            ee_cfg = env.unwrapped.termination_manager.get_term_cfg("ee_body_pos")
            ee_body_names = tuple(ee_cfg.params.get("body_names", ()))
        except Exception:
            ee_body_names = ()
    for index in range(args.steps):
        if args.reference_joint_policy:
            if command is None:
                raise RuntimeError("reference joint policy requires a motion command")
            actions = (command.joint_pos - action_term.offset) / action_term.scale
            actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            actions = policy(obs) if policy is not None else zero_actions
        obs, rew, done, _ = env.step(actions)
        last_rew = rew
        last_done = done
        rew_mean = float(rew.mean().detach().cpu())
        done_any = bool(done.any().detach().cpu())
        reward_trace.append(rew_mean)
        done_trace.append(done_any)
        row = {
            "step": index + 1,
            "rew_mean": rew_mean,
            "done_any": done_any,
        }
        term_dones = getattr(env.unwrapped.termination_manager, "_term_dones", {})
        for name, value in sorted(term_dones.items()):
            try:
                row[f"termination_{name}"] = bool(value.any().detach().cpu())
            except Exception:
                pass
        if command is not None:
            for key, value in sorted(command.metrics.items()):
                try:
                    row[key] = float(value.mean().detach().cpu())
                except Exception:
                    pass
            try:
                row["ref_anchor_z"] = float(command.anchor_pos_w[:, 2].mean().detach().cpu())
                row["robot_anchor_z"] = float(command.robot_anchor_pos_w[:, 2].mean().detach().cpu())
                for axis, axis_name in enumerate(("x", "y", "z")):
                    row[f"ref_anchor_{axis_name}"] = float(
                        command.anchor_pos_w[:, axis].mean().detach().cpu()
                    )
                    row[f"robot_anchor_{axis_name}"] = float(
                        command.robot_anchor_pos_w[:, axis].mean().detach().cpu()
                    )
            except Exception:
                pass
            try:
                body_pos_errors = []
                body_z_errors = []
                for body_index, body_name in enumerate(command.cfg.body_names):
                    ref_body_pos = command.body_pos_relative_w[:, body_index]
                    robot_body_pos = command.robot_body_pos_w[:, body_index]
                    pos_error = torch.linalg.norm(ref_body_pos - robot_body_pos, dim=-1)
                    z_error = torch.abs(ref_body_pos[:, 2] - robot_body_pos[:, 2])
                    safe_name = _metric_name(body_name)
                    row[f"body_pos_error_{safe_name}"] = float(
                        pos_error.mean().detach().cpu()
                    )
                    row[f"body_z_error_{safe_name}"] = float(
                        z_error.mean().detach().cpu()
                    )
                    body_pos_errors.append(row[f"body_pos_error_{safe_name}"])
                    body_z_errors.append(row[f"body_z_error_{safe_name}"])
                row["body_pos_error_max"] = max(body_pos_errors)
                row["body_z_error_max"] = max(body_z_errors)
            except Exception:
                pass
            if ee_body_names:
                try:
                    ee_errors = {}
                    for body_name in ee_body_names:
                        body_index = command.cfg.body_names.index(body_name)
                        z_error = torch.abs(
                            command.body_pos_relative_w[:, body_index, 2]
                            - command.robot_body_pos_w[:, body_index, 2]
                        )
                        key = f"ee_z_error_{body_name}"
                        row[key] = float(z_error.mean().detach().cpu())
                        ee_errors[key] = row[key]
                    row["ee_z_error_max"] = max(ee_errors.values())
                except Exception:
                    pass
        step_rows.append(row)
        if replay_output is not None:
            if command is None:
                raise RuntimeError("exact replay export requires a motion command")
            robot = env.unwrapped.scene["robot"]
            robot_body_pose = robot.data.body_link_pose_w
            reference_body_pose = torch.cat(
                [command.body_pos_relative_w, command.body_quat_relative_w], dim=-1
            )
            replay_rows["robot_root_pose"].append(
                robot.data.root_link_pose_w[0].detach().cpu().numpy().copy()
            )
            replay_rows["robot_joint_positions"].append(
                command.robot_joint_pos[0].detach().cpu().numpy().copy()
            )
            replay_rows["robot_body_pose"].append(
                robot_body_pose[0].detach().cpu().numpy().copy()
            )
            replay_rows["reference_root_pose"].append(
                reference_body_pose[0, 0].detach().cpu().numpy().copy()
            )
            replay_rows["reference_joint_positions"].append(
                command.joint_pos[0].detach().cpu().numpy().copy()
            )
            replay_rows["reference_body_pose"].append(
                reference_body_pose[0].detach().cpu().numpy().copy()
            )
        if render_mode is not None and (index % args.render_every == 0 or index in screenshot_steps):
            frame = _to_uint8_frame(env.unwrapped.render())
            if video_output is not None:
                frames.append(frame)
            if screenshot_dir is not None and index in screenshot_steps:
                import imageio.v2 as imageio

                imageio.imwrite(screenshot_dir / f"step_{index + 1:06d}.png", frame)
        if index in {0, args.steps - 1}:
            print(
                f"step={index + 1} "
                f"rew_mean={rew_mean:.6f} "
                f"done_any={done_any}"
            )

    if replay_output is not None:
        from mimicx.visualization.replay_package import write_replay_package

        robot = env.unwrapped.scene["robot"]
        step_dt = float(env.unwrapped.step_dt)
        replay_arrays = {
            "timestamps": np.arange(1, len(step_rows) + 1, dtype=np.float64) * step_dt,
            **{
                name: np.stack(values).astype(np.float32, copy=False)
                for name, values in replay_rows.items()
            },
        }
        replay_manifest = write_replay_package(
            replay_output,
            arrays=replay_arrays,
            metadata={
                "task": args.task,
                "motion_file": str(motion_file),
                "checkpoint": (
                    str(args.checkpoint_file.expanduser().resolve())
                    if args.checkpoint_file
                    else None
                ),
                "seed": args.seed,
                "device": args.device,
                "fps": 1.0 / step_dt,
                "timestep_seconds": step_dt,
                "reference_definition": "reward-space body pose after policy-anchor yaw/xy alignment",
            },
            body_names=list(robot.body_names),
            joint_names=list(robot.joint_names),
            reference_body_names=list(command.cfg.body_names),
        )
        print(f"replay_output={replay_output} frames={replay_manifest['frames']}")
    env.close()
    assert last_rew is not None
    assert last_done is not None
    final_rew_mean = float(last_rew.mean().detach().cpu())
    final_done_any = bool(last_done.any().detach().cpu())
    if video_output is not None:
        if not frames:
            raise RuntimeError("No rendered frames were captured for video output")
        import mediapy as media

        media.write_video(str(video_output), frames, fps=args.video_fps)
        print(f"video_output={video_output}")
    if screenshot_dir is not None:
        print(f"screenshot_dir={screenshot_dir}")
    if metrics_output is not None:
        metrics = {
            "task": args.task,
            "motion_file": str(motion_file),
            "checkpoint_file": str(args.checkpoint_file.expanduser().resolve()) if args.checkpoint_file else None,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "device": args.device,
            "disable_terminations": args.disable_terminations,
            "hide_reference_visual": args.hide_reference_visual,
            "ghost_color": (
                None
                if args.hide_reference_visual
                else list(motion_cmd.viz.ghost_color)
            ),
            "reference_visual_ground_align": (
                not args.hide_reference_visual and args.reference_visual_ground_align
            ),
            "ghost_ground_z_offset": (
                getattr(command, "_ghost_ground_z_offset", None) if command is not None else None
            ),
            "video_output": str(video_output) if video_output else None,
            "screenshot_dir": str(screenshot_dir) if screenshot_dir else None,
            "final_rew_mean": final_rew_mean,
            "final_done_any": final_done_any,
            "reward_mean_min": min(reward_trace),
            "reward_mean_max": max(reward_trace),
            "reward_mean_avg": sum(reward_trace) / len(reward_trace),
            "done_any_count": sum(done_trace),
        }
        metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"metrics_output={metrics_output}")
    if step_metrics_output is not None:
        import csv

        fieldnames = sorted({key for row in step_rows for key in row.keys()})
        with step_metrics_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(step_rows)
        print(f"step_metrics_output={step_metrics_output}")
    print(
        f"rollout_ok steps={args.steps} "
        f"final_rew_mean={final_rew_mean:.6f} "
        f"final_done_any={final_done_any}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
