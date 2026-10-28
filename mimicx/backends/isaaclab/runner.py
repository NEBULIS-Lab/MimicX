"""Local-file training and finite physical rollout, with lazy simulator startup."""
import argparse
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import traceback

import numpy as np

from mimicx.adapters.isaac_motion import validate_motion
from mimicx.backends.isaaclab.configuration import policy_observations, reapply_learning_rate
from mimicx.refinement.manifest import hash_file
from mimicx.runtime.backends import isaac_rollout_outputs


TASKS = {"MimicX-Tracking-Flat-G1-v0", "MimicX-Tracking-Flat-G1-Strict-v0",
         "BeyondMimic-Tracking-Flat-G1-v0", "BeyondMimic-Tracking-Flat-G1-Strict-v0"}


def runner_config(*, rsl_version, seed, device, learning_rate):
    major = int(rsl_version.split(".")[0])
    cfg = {"seed": seed, "device": device, "num_steps_per_env": 24,
           "save_interval": 25, "logger": "tensorboard",
           "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0,
                         "use_clipped_value_loss": True, "clip_param": 0.2,
                         "entropy_coef": 0.005, "num_learning_epochs": 5,
                         "num_mini_batches": 4, "learning_rate": learning_rate,
                         "schedule": "adaptive", "gamma": 0.99, "lam": 0.95,
                         "desired_kl": 0.01, "max_grad_norm": 1.0,
                         "rnd_cfg": None, "symmetry_cfg": None}}
    if major == 5:
        cfg.update(obs_groups={"actor": ["policy"], "critic": ["critic"]},
                   actor={"class_name": "MLPModel", "hidden_dims": [512, 256, 128],
                          "activation": "elu", "obs_normalization": True,
                          "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0}},
                   critic={"class_name": "MLPModel", "hidden_dims": [512, 256, 128],
                           "activation": "elu", "obs_normalization": True})
    elif major == 2:
        cfg.update(empirical_normalization=True,
                   policy={"class_name": "ActorCritic", "init_noise_std": 1.0,
                           "actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128],
                           "activation": "elu"})
    else:
        raise ValueError(f"Unaudited RSL-RL version {rsl_version}; supported API families are 2.x and 5.x")
    return cfg


def parser(mode):
    p = argparse.ArgumentParser(description=f"MimicX Isaac Lab local-file {mode}")
    p.add_argument("--task", default="MimicX-Tracking-Flat-G1-Strict-v0" if mode == "rollout" else "MimicX-Tracking-Flat-G1-v0")
    p.add_argument("--motion-file", type=Path, required=True)
    p.add_argument("--checkpoint-file", type=Path, required=mode == "rollout")
    p.add_argument("--num-envs", type=int, default=1 if mode == "rollout" else 1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--disable-joint-init-noise", action="store_true")
    if mode == "train":
        p.add_argument("--iterations", type=int, default=200)
        p.add_argument("--learning-rate", type=float, default=1e-3)
        p.add_argument("--output-root", type=Path, required=True)
        p.add_argument("--run-name", required=True)
        p.add_argument("--patch-file", type=Path, default=os.environ.get("MIMICX_AUTOREFINE_PATCH_FILE"))
    else:
        p.add_argument("--steps", type=int, required=True)
        p.add_argument("--metrics-output", type=Path, required=True)
        p.add_argument("--step-metrics-output", type=Path, required=True)
        p.add_argument("--video-output", type=Path)
    return p


def main(mode, argv=None):
    args = parser(mode).parse_args(argv)
    if args.task not in TASKS:
        raise ValueError(f"Use a MimicX task from {sorted(TASKS)}")
    if mode == "train" and args.task.endswith("Strict-v0"):
        raise ValueError("Strict task is reserved for evaluation")
    if mode == "train" and args.patch_file and args.task.startswith("BeyondMimic-"):
        raise ValueError("BeyondMimic baseline cannot use a refinement patch")
    if args.num_envs <= 0:
        raise ValueError("num-envs must be positive")
    args.motion_file = args.motion_file.resolve(strict=True)
    if args.checkpoint_file:
        args.checkpoint_file = args.checkpoint_file.resolve(strict=True)
    with np.load(args.motion_file, allow_pickle=False) as data:
        frames, fps = validate_motion(data)
        if "velocity_reference" in data and str(data["velocity_reference"].item()) != "link_origin":
            raise ValueError("Motion body linear velocities must be at link origins, not COMs")
        for key, expected in (("quaternion_order", "wxyz"), ("units", "SI"), ("angular_velocity_frame", "world")):
            if key not in data or str(data[key].item()) != expected:
                raise ValueError(f"Motion requires explicit {key}={expected}; use convert_motion.py")
    if mode == "rollout" and (args.num_envs != 1 or not 0 < args.steps < frames):
        raise ValueError("Strict rollout requires one environment and 0 < steps < motion frames")
    if mode == "rollout":
        outputs = isaac_rollout_outputs(args.metrics_output, args.step_metrics_output,
                                        protected_paths=(args.motion_file, args.checkpoint_file))
        args.metrics_output, args.step_metrics_output = outputs[:2]
    if mode == "train":
        if args.iterations <= 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("iterations and learning-rate must be positive")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
            raise ValueError("run-name must be a single safe path component")
        log_dir = args.output_root.resolve() / args.run_name
        log_dir.mkdir(parents=True, exist_ok=False)
    else:
        log_dir = None

    from isaaclab.app import AppLauncher
    launcher = AppLauncher({"headless": args.headless, "device": args.device,
                            "enable_cameras": mode == "rollout" and args.video_output is not None})
    app = launcher.app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        # Git is optional for RSL's best-effort repository logging. Source pins
        # are recorded independently; do not require a copied host executable.
        os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
        from rsl_rl.runners import OnPolicyRunner
        from mimicx.backends.isaaclab.tasks import MimicXEnvCfg, configure, register_tasks

        register_tasks()
        patch = None
        if mode == "train" and args.patch_file:
            patch = json.loads(Path(args.patch_file).read_text())
        cfg = MimicXEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.seed
        cfg.sim.device = args.device
        if not np.isclose(fps, 1 / (cfg.decimation * cfg.sim.dt), atol=1e-5):
            raise ValueError("Motion timing does not match environment control frequency")
        configure(cfg, motion_file=args.motion_file, strict=mode == "rollout",
                  patch=patch, frames=frames, fps=fps)
        if args.disable_joint_init_noise:
            cfg.commands.motion.joint_position_range = (0.0, 0.0)
        env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if mode == "rollout" and args.video_output else None)
        robot = env.unwrapped.scene["robot"]
        contact = env.unwrapped.scene.sensors["contact_forces"]
        contact_names = list(contact.body_names)
        if set(contact_names) != set(robot.body_names) or len(contact_names) != len(robot.body_names):
            raise ValueError("Contact sensor must cover each full-articulation body exactly once")
        with np.load(args.motion_file, allow_pickle=False) as data:
            validate_motion(data, joint_names=robot.joint_names, body_names=robot.body_names, fps=fps)
        if mode == "rollout" and args.video_output:
            env = gym.wrappers.RecordVideo(env, video_folder=str(args.video_output.resolve()),
                                         step_trigger=lambda step: step == 0, video_length=args.steps,
                                         disable_logger=True)
        env = RslRlVecEnvWrapper(env)
        # The wrapper resets and forwards FK, but upstream MotionCommand only
        # initializes relative-body targets on its first post-step update.
        env.unwrapped.command_manager.get_term("motion").refresh_relative()
        config = runner_config(rsl_version=version("rsl-rl-lib"), seed=args.seed,
                               device=args.device, learning_rate=getattr(args, "learning_rate", 2e-6))
        runner = OnPolicyRunner(env, deepcopy(config), log_dir=str(log_dir) if log_dir else None, device=args.device)
        if args.checkpoint_file:
            runner.load(str(args.checkpoint_file))
        metadata = {"task": args.task, "motion_sha256": hash_file(args.motion_file),
                    "checkpoint_sha256": hash_file(args.checkpoint_file) if args.checkpoint_file else None,
                    "rsl_rl_version": version("rsl-rl-lib"), "seed": args.seed,
                    "fps": fps, "joint_names": list(robot.joint_names), "body_names": list(robot.body_names),
                    "contact_body_names": contact_names,
                    "reference_body_names": list(cfg.commands.motion.body_names),
                    "runner_config": config}
        source_manifest = Path(__file__).resolve().parents[3] / "dependencies" / "isaaclab-sources.json"
        if source_manifest.is_file():
            sources = json.loads(source_manifest.read_text())
            metadata["source_manifest_sha256"] = hash_file(source_manifest)
            metadata["declared_source_identities"] = {
                key: sources[key] for key in ("isaac_sim", "isaac_lab", "whole_body_tracking", "rsl_rl")}
        if mode == "train":
            from isaaclab.utils.io import dump_yaml
            reapply_learning_rate(runner, args.learning_rate)
            dump_yaml(str(log_dir / "params" / "env.yaml"), cfg)
            dump_yaml(str(log_dir / "params" / "agent.yaml"), config)
            (log_dir / "mimicx_run.json").write_text(json.dumps({**metadata, "patch": patch}, indent=2) + "\n")
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
            checkpoint = log_dir / f"model_{runner.current_learning_iteration}.pt"
            runner.save(str(checkpoint))
            # Verify saved policy + observation normalizers, not merely file existence.
            with torch.inference_mode():
                observation = policy_observations(env.get_observations())
                before = runner.get_inference_policy(device=args.device)(observation).clone()
                runner.load(str(checkpoint))
                after = runner.get_inference_policy(device=args.device)(observation)
                if not torch.allclose(before, after, rtol=1e-6, atol=1e-7):
                    raise RuntimeError("Saved checkpoint failed identical-observation action reload check")
            completion = json.dumps({
                "checkpoint_sha256": hash_file(checkpoint), "passed": True,
                "max_abs_action_error": float((before - after).abs().max().item())}, indent=2) + "\n"
            (log_dir / "reload_check.json").write_text(completion)
            (log_dir.parent / f"{args.run_name}.completed.json").write_text(completion)
            print(f"train_ok checkpoint={checkpoint}", flush=True)
        else:
            from mimicx.backends.isaaclab.telemetry import RolloutTelemetry
            telemetry = RolloutTelemetry()
            raw = env.unwrapped
            # Isaac auto-resets done environments in step(). Capture physical
            # state at termination computation, not from the reset observation.
            compute = raw.termination_manager.compute
            def capture_compute():
                result = compute()
                telemetry.capture(raw)
                return result
            raw.termination_manager.compute = capture_compute
            policy = runner.get_inference_policy(device=args.device)
            obs = policy_observations(env.get_observations())
            with torch.inference_mode():
                for _ in range(args.steps):
                    if not app.is_running():
                        raise RuntimeError("Simulator stopped before requested rollout horizon")
                    actions = policy(obs)
                    obs, reward, _, _ = env.step(actions)
                    telemetry.rows[-1]["rew_mean"] = float(reward.mean().item())
                    telemetry.states[-1]["actions"] = actions[0].detach().cpu().numpy().copy()
            telemetry.write(args.metrics_output, args.step_metrics_output, metadata=metadata)
            print(f"rollout_ok steps={args.steps}", flush=True)
    except BaseException:
        # Some Kit shutdown paths exit the process before Python can render a
        # pending traceback. Emit it while the app is still alive.
        traceback.print_exc(file=sys.stderr)
        print(f"isaac_{mode}_failed", file=sys.stderr, flush=True)
        raise
    finally:
        if env is not None:
            env.close()
        app.close()
    return 0
