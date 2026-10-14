"""Replay, evaluation, and frozen-actor integration for Parkour actor+MPC."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .predictive import Planner, PlannerConfig, orientation_error, world_penetration
from .servo import ActionAffine, actuator_names, make_environment_config, save_json

VALIDATION_SEEDS = (28002, 28003, 28004, 29001, 29002, 29003)
CHANNELS = ("pelvis", "torso", "joints", "world_body", "hip_knee", "anchor")


def completion_report(series, valid):
    """Original 269-step, seven-check motion score, separate from survival."""
    series = np.asarray(series, dtype=float)
    if (series.ndim != 2 or series.shape[1] != 6 or not 0 <= valid <= 269
            or len(series) <= valid or not np.isfinite(series[:valid + 1]).all()):
        raise ValueError("Invalid completion metrics or valid-step count")
    values = series[:valid + 1]
    tail = values[225:]
    terminal = values[265:270] if valid == 269 else None
    result = dict(horizon_complete=valid == 269, valid_steps=int(valid),
                  pelvis_tail_mean=float(tail[:, 0].mean()) if len(tail) else None,
                  pelvis_tail_p95=float(np.quantile(tail[:, 0], .95)) if len(tail) else None)
    for i, label in enumerate(CHANNELS):
        result[label + "_terminal"] = float(np.quantile(terminal[:, i], .95)) if terminal is not None else None
    limits = dict(pelvis_terminal=.6, torso_terminal=.6, joints_terminal=.45,
                  world_body_terminal=.22, hip_knee_terminal=.5, anchor_terminal=.20,
                  pelvis_tail_p95=.85)
    checks = {k: result[k] is not None and result[k] <= threshold for k, threshold in limits.items()}
    result.update(checks=checks, thresholds=limits,
                  motion_complete=result["horizon_complete"] and all(checks.values()),
                  terminal_pass=result["horizon_complete"] and all(v for k, v in checks.items() if k != "pelvis_tail_p95"))
    padded = np.tile([np.pi, np.pi, 2., 1., 2., .7], (270, 1))
    padded[:len(values)] = values
    norm = padded / np.array([1., 1., .6, .3, .6, .2])
    costs = (norm ** 2 * np.array([.35, .15, .20, .1, .1, .1])).sum(-1)
    result["objective"] = float(.2 * costs[195:225].mean() + .5 * costs[225:].mean() + .3 * costs[260:].mean())
    return result


def evaluate_six_resets(evaluator, seeds=VALIDATION_SEEDS):
    """Invoke a fresh episode per seed; the callable must return its real report."""
    seeds = list(seeds)
    if len(seeds) != 6 or len(set(seeds)) != 6:
        raise ValueError("Evaluation requires six distinct reset seeds")
    rows = []
    for seed in seeds:
        row = evaluator(seed)
        if row.get("seed") != seed:
            raise ValueError("Evaluator returned the wrong reset seed")
        rows.append(row)
    completion = [row["completion"] for row in rows]
    return dict(seeds=seeds, count=6, training_updates=0,
                horizon_complete=sum(bool(c["horizon_complete"]) for c in completion),
                terminal_checks_pass=sum(bool(c["horizon_complete"] and all(v for k, v in c["checks"].items()
                                                                           if k != "pelvis_tail_p95")) for c in completion),
                all_motion_checks_pass=sum(bool(c["horizon_complete"] and all(c["checks"].values())) for c in completion),
                results=rows)


class StateAudit:
    """CPU FK/contact audit; writes only its private MjData scratch state."""

    def __init__(self, model, names, motion):
        import mujoco

        self.mj, self.model, self.motion = mujoco, model, motion
        self.data = mujoco.MjData(model)
        joints, bodies = names["joint_names"], names["body_names"]
        self.qids = [model.jnt_qposadr[model.joint("robot/" + n).id] for n in joints]
        self.vids = [model.jnt_dofadr[model.joint("robot/" + n).id] for n in joints]
        self.legs = [i for i, n in enumerate(joints) if "hip_pitch" in n or "knee" in n]
        tracked = names.get("reference_body_names", bodies)
        self.ref_ids = [bodies.index(n) for n in tracked]
        self.body_ids = [model.body("robot/" + n).id for n in tracked]
        self.torso = model.body("robot/torso_link").id
        self.torso_ref = bodies.index("torso_link")
        feet = ("left_ankle_roll_link", "right_ankle_roll_link")
        self.feet = [model.body("robot/" + n).id for n in feet]
        self.feet_ref = [bodies.index(n) for n in feet]

    def measure(self, frame, qpos):
        self.data.qpos[:] = qpos
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_collision(self.model, self.data)
        pos, quat = self.motion["body_pos_w"][frame], self.motion["body_quat_w"][frame]
        joint_error = qpos[self.qids] - self.motion["joint_pos"][frame]
        torso_quat = self.data.xquat[self.torso]
        values = np.array([
            orientation_error(qpos[3:7], quat[0]),
            orientation_error(torso_quat, quat[self.torso_ref]),
            np.sqrt(np.mean(joint_error ** 2)),
            np.sqrt(np.mean(np.sum((self.data.xpos[self.body_ids] - pos[self.ref_ids]) ** 2, -1))),
            np.sqrt(np.mean(joint_error[self.legs] ** 2)),
            np.linalg.norm(self.data.xpos[self.torso] - pos[self.torso_ref]),
        ])
        # These are native termination thresholds, not the stricter MPC cost guards.
        failures = []
        if abs(self.data.xpos[self.torso, 2] - pos[self.torso_ref, 2]) > .35:
            failures.append("anchor_pos")
        gravity = abs(2 * np.sum(torso_quat[1:3] ** 2) - 2 * np.sum(quat[self.torso_ref, 1:3] ** 2))
        if gravity > .8:
            failures.append("anchor_ori")
        if np.max(np.abs(self.data.xpos[self.feet, 2] - pos[self.feet_ref, 2])) > .25:
            failures.append("ee_body_pos")
        return values, world_penetration(self.model, self.data), failures


def _contact_report(depth):
    depth = np.asarray(depth)
    tail = depth[190:]
    return dict(sample_hz=50, max_sampled_penetration_m=float(depth.max(initial=0.)),
                tail_peak_m=float(tail.max()) if len(tail) else None,
                tail_p95_m=float(np.quantile(tail, .95)) if len(tail) else None,
                tail_frames_above_2cm=int(np.sum(tail > .02)))


def replay_states(model, traces, names, motion, *, index=2):
    """Reconstruct saved poses exactly for inspection, not a dynamics claim."""
    audit = StateAudit(model, names, motion)
    root, joints, active = traces["root_pose"], traces["joint_pos"], traces["active"]
    if root.ndim != 3 or not 0 <= index < root.shape[1]:
        raise ValueError("Invalid recorded-state controller index")
    mask = np.asarray(active[:, index], dtype=bool)
    if not len(mask) or not mask[0] or np.any(np.diff(mask.astype(int)) > 0):
        raise ValueError("Active mask must be one contiguous true-start episode")
    valid = int(mask.sum()) - 1
    qpos = np.tile(model.qpos0, (valid + 1, 1))
    qpos[:, :7], qpos[:, audit.qids] = root[:valid + 1, index], joints[:valid + 1, index]
    if not np.isfinite(qpos).all():
        raise ValueError("Nonfinite recorded poses")
    metrics, depth = [], []
    for frame, q in enumerate(qpos):
        values, penetration, _ = audit.measure(frame, q)
        metrics.append(values)
        depth.append(penetration)
    report = dict(execution="kinematic_state_replay", physical_execution_verified=False,
                  training_updates=0, completion=completion_report(metrics, valid), contacts=_contact_report(depth))
    return dict(qpos=qpos, completion_metrics=np.asarray(metrics), penetration=np.asarray(depth), report=report)


def initial_state(model, motion, names, traces=None, *, index=2):
    """True-start reset from reference velocities and optionally recorded pose.

    Old traces have no root velocities. Only frame zero is supported: native
    play reset has zero velocity perturbation and copies reference velocities.
    Arbitrary continuation states must supply actual qpos AND qvel separately.
    """
    audit = StateAudit(model, names, motion)
    qpos, qvel = model.qpos0.copy(), np.zeros(model.nv)
    if traces is None:
        qpos[:3], qpos[3:7] = motion["body_pos_w"][0, 0], motion["body_quat_w"][0, 0]
        qpos[audit.qids] = motion["joint_pos"][0]
        qvel[audit.vids] = motion["joint_vel"][0]
    else:
        qpos[:7] = traces["root_pose"][0, index]
        qpos[audit.qids] = traces["joint_pos"][0, index]
        qvel[audit.vids] = traces["joint_vel"][0, index]
    qvel[:3] = motion["body_lin_vel_w"][0, 0]
    qvel[3:6] = Rotation.from_quat(qpos[[4, 5, 6, 3]]).inv().apply(motion["body_ang_vel_w"][0, 0])
    return qpos, qvel


def run_cpu(model, motion, names, affine, actions, qpos, qvel, *, planner=None, steps=269):
    """Physical servo execution or recorded-prefix + live CPU MPC, with no resets."""
    import mujoco

    if not 1 <= steps <= 269 or not np.isclose(model.opt.timestep * 4, .02):
        raise ValueError("CPU evaluation expects 1..269 steps at 50 Hz with four physics substeps")
    if planner is not None and (planner.config.total_steps != 269 or planner.config.substeps != 4
                                or not np.isclose(planner.config.control_dt, .02)):
        raise ValueError("Planner clock must match the 269-step, 50 Hz evaluator")
    actions = np.asarray(actions)
    if actions.shape != (269, len(affine.names)) or not np.isfinite(actions).all():
        raise ValueError("Expected a full 269-frame normalized control sequence")
    control_names = actuator_names(model)
    data = mujoco.MjData(model)
    if np.shape(qpos) != (model.nq,) or np.shape(qvel) != (model.nv,) or not np.isfinite(np.r_[qpos, qvel]).all():
        raise ValueError("Invalid initial state")
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_forward(model, data)
    audit = StateAudit(model, names, motion)
    poses, velocities, metrics, depths, controls, forces = [data.qpos.copy()], [data.qvel.copy()], [], [], [], []
    values, depth, _ = audit.measure(0, data.qpos)
    metrics.append(values)
    depths.append(depth)
    valid, failures = 0, []
    for frame in range(steps):
        action = (planner.step(frame, data.qpos, data.qvel) if planner is not None and frame >= planner.config.start
                  else actions[frame])
        data.ctrl[:] = affine.to_controls(action, control_names)
        mujoco.mj_step(model, data, nstep=4)
        if not np.isfinite(np.r_[data.qpos, data.qvel]).all() or not np.isclose(data.time, (frame + 1) * .02):
            failures = ["simulation_diverged"]
            break
        poses.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        controls.append(np.asarray(action).copy())
        forces.append(data.actuator_force.copy())
        # Native termination precedes reference advancement; reporting follows it.
        _, _, failures = audit.measure(frame, data.qpos)
        values, depth, _ = audit.measure(frame + 1, data.qpos)
        metrics.append(values)
        depths.append(depth)
        if failures:
            break
        valid = frame + 1
    control_array = np.asarray(controls).reshape(-1, len(affine.names))
    quality = dict(peak_abs_action=float(np.abs(control_array).max(initial=0.)),
                   action_delta_rms=float(np.sqrt(np.mean(np.diff(control_array, axis=0) ** 2))) if len(controls) > 1 else 0.,
                   peak_actuator_force=float(np.abs(forces).max(initial=0.)))
    report = dict(execution="recorded_prefix_cpu_mpc" if planner else "physical_servo_replay",
                  backend="mujoco_cpu", frozen_actor_executed=False, training_updates=0,
                  pose_writes_after_reset=0, root_forces=False, termination_channels=failures,
                  completion=completion_report(metrics, valid), contacts=_contact_report(depths[:valid + 1]),
                  action_quality=quality, original_gpu_rollout_equivalence_verified=False)
    result = dict(qpos=np.asarray(poses), qvel=np.asarray(velocities), actions=control_array,
                  actuator_force=np.asarray(forces), completion_metrics=np.asarray(metrics),
                  penetration=np.asarray(depths), report=report)
    if planner:
        result["planner"] = planner.report()
    return result


def evaluate_actor(scene, motion_path, checkpoint, names, nominal_actions, output, *, seed=29003,
                   config=None, device="cuda:0", ppo_only=False):
    """Optional native-runtime entry point, separate from the CPU replay CLI.

    The caller owns device allocation. This function creates a fresh environment
    and planner, loads only the frozen actor, and stops at the first termination.
    """
    from dataclasses import asdict
    import torch
    from .environment import install_initial_reference_refresh

    config = config or PlannerConfig()
    if config.total_steps != 269 or config.substeps != 4 or not np.isclose(config.control_dt, .02):
        raise ValueError("Planner clock must match the 269-step, 50 Hz evaluator")
    original, _ = install_initial_reference_refresh()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

    cfg = make_environment_config(scene, motion_path)
    agent = unitree_g1_tracking_ppo_runner_cfg()
    cfg.seed = agent.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = planner = None
    try:
        env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=device), clip_actions=agent.clip_actions)
        base = env.unwrapped
        action = base.action_manager.get_term("joint_pos")
        robot = base.scene["robot"]
        if list(robot.joint_names) != names["joint_names"] or list(robot.body_names) != names["body_names"]:
            raise ValueError("Robot names differ from the recorded reference")
        if list(base.command_manager.get_term("motion").cfg.body_names) != names["reference_body_names"]:
            raise ValueError("Tracked body order differs from recorded evaluation")
        def cpu(value):
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        def row(value):
            value = cpu(value)
            return value[0] if value.ndim == 2 else value
        affine = ActionAffine(list(action.target_names), row(action.scale), row(action.offset))
        with np.load(motion_path, allow_pickle=False) as archive:
            motion = dict(archive)
        model = base.sim.mj_model
        if not ppo_only:
            planner = Planner(model, motion, names, affine, nominal_actions, config)
        runner = MjlabOnPolicyRunner(env, asdict(agent), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        audit = StateAudit(model, names, motion)
        obs = env.get_observations()
        poses = [cpu(base.sim.data.qpos[0]).copy()]
        velocities = [cpu(base.sim.data.qvel[0]).copy()]
        metrics, depths, actions, forces, terms = [], [], [], [], []
        values, depth, _ = audit.measure(0, poses[0])
        metrics.append(values)
        depths.append(depth)
        valid = 0
        with torch.inference_mode():
            for frame in range(config.total_steps):
                target = policy(obs)
                if planner is not None:
                    target = planner.callback(frame, target, base, None, action, [True], index=0)
                actions.append(cpu(target[0]).copy())
                obs, _, done, _ = env.step(target)
                if bool(done[0]):
                    terms = [k for k, v in base.termination_manager._term_dones.items() if bool(v[0])]
                    break  # Auto-reset state is never captured or counted as valid.
                poses.append(cpu(base.sim.data.qpos[0]).copy())
                velocities.append(cpu(base.sim.data.qvel[0]).copy())
                forces.append(cpu(robot.data.actuator_force[0]).copy())
                values, depth, _ = audit.measure(frame + 1, poses[-1])
                metrics.append(values)
                depths.append(depth)
                valid = frame + 1
        report = dict(seed=seed, execution="frozen_ppo" if ppo_only else "frozen_actor_plus_mpc",
                      training_updates=0, pose_writes_after_reset=0, root_forces=False,
                      completion=completion_report(metrics, valid), contacts=_contact_report(depths),
                      termination_channels=terms,
                      action_quality=dict(peak_abs_action=float(np.abs(actions).max(initial=0.)),
                        action_delta_rms=float(np.sqrt(np.mean(np.diff(actions, axis=0) ** 2))) if len(actions) > 1 else 0.,
                        peak_actuator_force=float(np.abs(forces).max(initial=0.))))
        write_result(output, dict(qpos=np.asarray(poses), qvel=np.asarray(velocities),
                     completion_metrics=np.asarray(metrics), actions=np.asarray(actions),
                     actuator_force=np.asarray(forces), report=report,
                     **({"planner": planner.report()} if planner else {})))
        return report
    finally:
        if planner is not None:
            planner.close()
        if env is not None:
            env.close()
        import mjlab.envs
        mjlab.envs.ManagerBasedRlEnv = original


def write_result(output, result):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "traces.npz", **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    save_json(output / "report.json", result["report"])
    if "planner" in result:
        save_json(output / "planner.json", result["planner"])


def read_validation_report(path, index=2):
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return payload[index]
    if "screen" in payload:
        return payload["screen"][index]
    if "reports" in payload:
        return payload["reports"][index]
    return payload
