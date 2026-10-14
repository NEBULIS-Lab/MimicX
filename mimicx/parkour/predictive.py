"""State-feedback Parkour MPC: native MuJoCo rollout and differential evolution.

The frozen actor is an external approach controller. This module never trains,
writes the live pose, changes contact physics, or applies root forces.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

from .servo import actuator_names


GROUPS = (
    {"left_hip_pitch_joint": 1, "right_hip_pitch_joint": 1},
    {"left_hip_pitch_joint": 1, "right_hip_pitch_joint": -1},
    {"left_knee_joint": 1, "right_knee_joint": 1},
    {"left_knee_joint": 1, "right_knee_joint": -1},
    {"waist_yaw_joint": 1}, {"waist_pitch_joint": 1}, {"waist_roll_joint": 1},
    {"left_shoulder_pitch_joint": 1, "right_shoulder_pitch_joint": 1},
    {"left_shoulder_roll_joint": 1, "right_shoulder_roll_joint": -1},
    {"left_elbow_joint": 1, "right_elbow_joint": 1},
    {"left_hip_roll_joint": 1, "right_hip_roll_joint": -1},
    {"left_ankle_pitch_joint": 1, "right_ankle_pitch_joint": 1},
    {"left_hip_yaw_joint": 1, "right_hip_yaw_joint": 1},
    {"left_hip_yaw_joint": 1, "right_hip_yaw_joint": -1},
    {"left_shoulder_pitch_joint": 1, "right_shoulder_pitch_joint": -1},
)
LIMITS = np.array([2.8, 1.2, 2.5, 1.2, 1.8, .7, .5, 1.5, .6, 1.5, .5, .8, .9, .9, 1.2])


def orientation_error(a, b):
    """Geodesic angle in radians, for scalar-first (wxyz) quaternions."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    for q in (a, b):
        if q.shape[-1:] != (4,) or not np.isfinite(q).all() or np.any(np.linalg.norm(q, axis=-1) < 1e-12):
            raise ValueError("Invalid quaternion")
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0., 1.))


def interpolate(knots, length):
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 3 or knots.shape[1] < 2 or length < 1:
        raise ValueError("Expected batched knots and positive length")
    points = np.linspace(0, knots.shape[1] - 1, length)
    lower = np.minimum(points.astype(int), knots.shape[1] - 2)
    alpha = (points - lower)[None, :, None]
    return knots[:, lower] * (1 - alpha) + knots[:, lower + 1] * alpha


def group_matrix(names):
    return np.array([[group.get(name, 0.) for name in names] for group in GROUPS])


def penetration_penalty(depth, tolerance=.01, weight=5000.):
    if tolerance < 0 or weight < 0:
        raise ValueError("Contact tolerance and weight must be nonnegative")
    excess = np.maximum(0., np.asarray(depth) - tolerance) ** 2
    return weight * (excess.max(axis=-1) + .1 * excess.mean(axis=-1))


def world_penetration(model, data):
    contacts = data.contact
    if not len(contacts):
        return 0.
    world = (model.geom_bodyid[contacts.geom1] == 0) | (model.geom_bodyid[contacts.geom2] == 0)
    return float(np.maximum(0., -contacts.dist[world]).max(initial=0.))


@dataclass(frozen=True)
class PlannerConfig:
    start: int = 190
    total_steps: int = 269
    horizon: int = 45
    replan_every: int = 3
    population: int = 96
    generations: int = 24
    threads: int = 6
    optimizer_seed: int = 28001
    control_dt: float = .02
    substeps: int = 4
    momentum: bool = True
    nominal: str = "recorded"
    contact_weight: float = 0.
    penetration_tolerance: float = .01

    def __post_init__(self):
        for key in ("start", "total_steps", "horizon", "replan_every", "population", "generations", "threads", "substeps", "optimizer_seed"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if (not 0 <= self.start < self.total_steps or self.population < 5 or self.generations < 0
                or min(self.horizon, self.replan_every, self.threads, self.substeps) < 1
                or self.replan_every > self.horizon or self.optimizer_seed < 0):
            raise ValueError("Invalid planner horizon or search budget")
        if (not np.isfinite([self.control_dt, self.contact_weight, self.penetration_tolerance]).all()
                or self.control_dt <= 0 or min(self.contact_weight, self.penetration_tolerance) < 0):
            raise ValueError("Invalid timing/contact settings")
        if self.nominal not in ("recorded", "reference"):
            raise ValueError("Nominal must be recorded or reference")

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


def recover_nominal(actions, history, affine, control_names, *, start=190):
    """Invert executed MPC residuals to recover the missing recorded prior tail.

    Prefix entries are retained but never used by a planner starting at `start`.
    The recovered prior inherits float32 recording precision, not byte identity.
    """
    controls = affine.to_controls(actions, control_names)
    replans = history["replans"]
    if not replans or replans[0]["frame"] != start:
        raise ValueError("Planner history must cover the entire controlled tail")
    previous = start - 1
    for i, row in enumerate(replans):
        frame, length = row["frame"], row["horizon"]
        end = replans[i + 1]["frame"] if i + 1 < len(replans) else len(controls)
        if not previous < frame < end <= min(frame + length, len(controls)):
            raise ValueError("Invalid or incomplete replan intervals")
        coefficients = np.asarray(row["coefficients"], dtype=float).reshape(1, 3, len(GROUPS))
        if not np.isfinite(coefficients).all():
            raise ValueError("Invalid planner coefficients")
        corrections = interpolate(coefficients, length)[0] @ group_matrix(control_names)
        controls[frame:end] -= corrections[:end - frame]
        previous = frame
    return affine.to_actions(controls, control_names)


class Planner:
    """One planner per episode. `step` returns normalized joint actions only."""

    def __init__(self, model, motion, names, affine, nominal_actions=None, config=None):
        import mujoco as mj
        from mujoco.rollout import Rollout

        self.mj, self.model = mj, model
        self.config = config or PlannerConfig()
        cfg = self.config
        self.affine, self.motion = affine, {k: np.asarray(v) for k, v in motion.items()}
        self.actuator_names = actuator_names(model)
        affine._order(self.actuator_names)
        if model.nq != model.nv + 1 or model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE:
            raise ValueError("Expected one floating-base robot")
        if not np.isclose(model.opt.timestep * cfg.substeps, cfg.control_dt):
            raise ValueError("Model timestep and planner decimation disagree")
        joints, bodies = names["joint_names"], names["body_names"]
        if len(set(joints)) != len(joints) or len(set(bodies)) != len(bodies) or bodies[0] != "pelvis":
            raise ValueError("Names must be unique, with pelvis first")
        for key, shape in (("joint_pos", (len(joints),)), ("joint_vel", (len(joints),)),
                           ("body_pos_w", (len(bodies), 3)), ("body_quat_w", (len(bodies), 4)),
                           ("body_ang_vel_w", (len(bodies), 3))):
            value = self.motion[key]
            if value.shape[1:] != shape or len(value) <= cfg.total_steps or not np.isfinite(value).all():
                raise ValueError(f"Invalid reference {key}; use full body_names, not tracked subset")
        orientation_error(self.motion["body_quat_w"], self.motion["body_quat_w"])
        self.qpos_indices = np.array([model.jnt_qposadr[model.joint("robot/" + n).id] for n in joints])
        self.leg_indices = [i for i, n in enumerate(joints) if "hip_pitch" in n or "knee" in n]
        if not self.leg_indices:
            raise ValueError("Reference is missing hip/knee joints")
        self.ctrl_indices = np.array([joints.index(n) for n in self.actuator_names])
        self.body_ids = [model.body("robot/" + n).id for n in bodies]
        feet = ("left_ankle_roll_link", "right_ankle_roll_link")
        self.foot_ids = [model.body("robot/" + n).id for n in feet]
        self.foot_ref_ids = [bodies.index(n) for n in feet]
        self.torso_id = model.body("robot/torso_link").id
        self.torso_ref_id = bodies.index("torso_link")
        self.groups, self.limits = group_matrix(self.actuator_names), LIMITS.copy()
        self.width = len(self.groups)
        self.old_controls = None
        if cfg.nominal == "recorded":
            if nominal_actions is None or np.shape(nominal_actions) != (cfg.total_steps, len(affine.names)):
                raise ValueError("Recorded nominal requires a complete named action sequence")
            self.old_controls = affine.to_controls(nominal_actions, self.actuator_names)
        self.fk_data = mj.MjData(model)
        self.data = [mj.MjData(model) for _ in range(cfg.threads)]
        self.pool = Rollout(nthread=cfg.threads)
        self.rng = np.random.default_rng(cfg.optimizer_seed)
        self.plan = self.plan_start = self.last_prediction = self.last_frame = None
        self.history, self.prediction_errors = [], []

    def close(self):
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def report(self):
        return dict(config=asdict(self.config), replans=self.history,
                    prediction_errors=self.prediction_errors, training_updates=0)

    def _cost(self, vectors, initial, nominal, reference, frames):
        cfg, motion = self.config, self.motion
        length = len(frames)
        vectors = np.asarray(vectors).T
        corrections = interpolate(vectors.reshape(-1, 3, self.width), length) @ self.groups
        controls = nominal[None] + corrections
        states, _ = self.pool.rollout(self.model, self.data, initial,
                                     control=np.repeat(controls, cfg.substeps, axis=1))
        states = states[:, cfg.substeps - 1::cfg.substeps]
        positions = states[:, :, 1:1 + self.model.nq]
        # Diverged MuJoCo rollouts repeat a state; penalize those, not just NaNs.
        expected_time = initial[0] + np.arange(1, length + 1) * cfg.control_dt
        valid = np.isfinite(states).all(axis=(1, 2)) & np.all(np.isclose(states[:, :, 0], expected_time), axis=1)
        costs = np.full(len(vectors), 1e6)
        if not valid.any():
            return costs
        positions, states, corrections = positions[valid], states[valid], corrections[valid]
        angle = orientation_error(positions[:, :, 3:7], motion["body_quat_w"][frames, 0])
        displacement = np.linalg.norm(positions[:, :, :3] - motion["body_pos_w"][frames, 0], axis=-1)
        joint_ref = motion["joint_pos"][frames]
        joint_error = ((positions[:, :, self.qpos_indices] - joint_ref) ** 2).mean(-1)
        legs = self.leg_indices
        leg_error = ((positions[:, :, self.qpos_indices[legs]] - joint_ref[:, legs]) ** 2).mean(-1)
        foot_z = np.empty((*positions.shape[:2], 2))
        torso_pose = np.empty((*positions.shape[:2], 7))
        body_error = np.empty(positions.shape[:2])
        depth = np.zeros(positions.shape[:2])
        for c in range(len(positions)):
            for t, frame in enumerate(frames):
                self.fk_data.qpos[:] = positions[c, t]
                self.mj.mj_kinematics(self.model, self.fk_data)
                foot_z[c, t] = self.fk_data.xpos[self.foot_ids, 2]
                torso_pose[c, t] = np.r_[self.fk_data.xpos[self.torso_id], self.fk_data.xquat[self.torso_id]]
                body_error[c, t] = np.mean(np.sum((self.fk_data.xpos[self.body_ids] - motion["body_pos_w"][frame]) ** 2, -1))
                if cfg.contact_weight:
                    self.mj.mj_collision(self.model, self.fk_data)
                    depth[c, t] = world_penetration(self.model, self.fk_data)
        feet_error = np.abs(foot_z - motion["body_pos_w"][frames][:, self.foot_ref_ids, 2]).max(-1)
        torso_reference = motion["body_quat_w"][frames, self.torso_ref_id]
        torso_angle = orientation_error(torso_pose[:, :, 3:], torso_reference)
        height_error = np.abs(torso_pose[:, :, 2] - motion["body_pos_w"][frames, self.torso_ref_id, 2])
        gravity_error = np.abs(2 * np.sum(torso_pose[:, :, 4:6] ** 2, -1) - 2 * np.sum(torso_reference[:, 1:3] ** 2, -1))
        violation = (np.maximum(0., feet_error - .22) ** 2 + np.maximum(0., height_error - .32) ** 2
                     + np.maximum(0., gravity_error - .75) ** 2)
        per_step = (3 * angle ** 2 + 15 * displacement ** 2 + 1.5 * joint_error + 3 * leg_error
                    + torso_angle ** 2 + 8 * body_error + 500 * violation)
        if cfg.momentum:
            omega = states[:, :, 1 + self.model.nq + 3:1 + self.model.nq + 6]
            quat = positions[:, :, 3:7].reshape(-1, 4)
            world = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).apply(omega.reshape(-1, 3)).reshape(omega.shape)
            per_step += .08 * np.sum((world - motion["body_ang_vel_w"][frames, 0]) ** 2, -1)
        values = .65 * per_step.mean(-1) + .35 * per_step[:, -min(10, length):].mean(-1)
        values += 10000 * violation.max(-1) + .008 * (corrections ** 2).mean(axis=(1, 2))
        if cfg.contact_weight:
            values += penetration_penalty(depth, cfg.penetration_tolerance, cfg.contact_weight)
        costs[valid] = np.nan_to_num(values, nan=1e6, posinf=1e6, neginf=1e6)
        return costs

    def step(self, frame, qpos, qvel):
        cfg = self.config
        if not cfg.start <= frame < cfg.total_steps:
            raise ValueError("Frame is outside the MPC interval")
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError("Use a fresh planner for each reset; frames must increase")
        qpos, qvel = np.asarray(qpos, dtype=float), np.asarray(qvel, dtype=float)
        if qpos.shape != (self.model.nq,) or qvel.shape != (self.model.nv,) or not np.isfinite(np.r_[qpos, qvel]).all():
            raise ValueError("Invalid observed physics state")
        orientation_error(qpos[3:7], qpos[3:7])
        if self.last_prediction is not None and frame == self.last_frame + 1:
            self.prediction_errors.append(dict(frame=frame,
                qpos_max=float(np.max(np.abs(qpos - self.last_prediction[1:1 + self.model.nq]))),
                root_position=float(np.linalg.norm(qpos[:3] - self.last_prediction[1:4])),
                root_angle=float(orientation_error(qpos[3:7], self.last_prediction[4:8]))))
        initial = np.r_[frame * cfg.control_dt, qpos, qvel]
        if self.plan is None or frame - self.plan_start >= cfg.replan_every:
            length = min(cfg.horizon, cfg.total_steps - frame)
            frames = np.arange(frame + 1, frame + 1 + length)
            reference = self.motion["joint_pos"][frames][:, self.ctrl_indices]
            if cfg.nominal == "recorded":
                nominal = self.old_controls[frames - 1]
            else:
                lead = -self.model.actuator_biasprm[:, 2] / self.model.actuator_gainprm[:, 0]
                nominal = reference + self.motion["joint_vel"][frames][:, self.ctrl_indices] * lead
            warm = np.zeros((3, self.width))
            if self.plan is not None:
                remaining = self.plan[frame - self.plan_start:]
                if len(remaining):
                    residual = remaining - nominal[:len(remaining)]
                    samples = residual[np.linspace(0, len(residual) - 1, 3).astype(int)]
                    warm = np.linalg.lstsq(self.groups.T, samples.T, rcond=None)[0].T
            limits = np.tile(self.limits, 3)
            init = warm.ravel()[None] + self.rng.normal(0, .25, (cfg.population, len(limits)))
            init[0], init[1] = warm.ravel(), 0.
            desired = (reference - nominal)[np.linspace(0, length - 1, 3).astype(int)]
            init[2] = np.linalg.lstsq(self.groups.T, desired.T, rcond=None)[0].T.ravel()
            init = np.clip(init, -limits, limits)
            evaluated = 0

            def objective(vectors):
                nonlocal evaluated
                evaluated += vectors.shape[-1]
                return self._cost(vectors, initial, nominal, reference, frames)

            started = time.monotonic()
            # Generator-valued seed works on both SciPy 1.10 and the native 1.17.
            result = differential_evolution(objective, list(zip(-limits, limits)), init=init,
                seed=self.rng, maxiter=cfg.generations, mutation=(.25, .65), recombination=.8,
                vectorized=True, updating="deferred", polish=False, tol=0., atol=0.)
            if result.fun >= 1e6:
                raise RuntimeError("All MPC candidates diverged")
            self.plan = nominal + interpolate(result.x.reshape(1, 3, self.width), length)[0] @ self.groups
            self.plan_start = frame
            self.history.append(dict(frame=frame, horizon=length, cost=float(result.fun), rollouts=evaluated,
                seconds=time.monotonic() - started, coefficients=result.x.tolist()))
        control = self.plan[frame - self.plan_start]
        predicted, _ = self.pool.rollout(self.model, self.data, initial,
            control=np.repeat(control[None, None], cfg.substeps, axis=1))
        self.last_prediction, self.last_frame = predicted[0, -1].copy(), frame
        return self.affine.to_actions(control, self.actuator_names)

    def callback(self, frame, actions, base, command, action, active, *, index=-1):
        """Adapter for the original evaluator; touches only the selected action row."""
        if frame < self.config.start or not bool(active[index]):
            return actions
        import torch

        def array(value):
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

        def row(value):
            value = array(value)
            return value[index if len(value) > 1 else 0] if value.ndim == 2 else value

        if (list(action.target_names) != self.affine.names
                or not np.allclose(row(action.scale), self.affine.scale)
                or not np.allclose(row(action.offset), self.affine.offset)):
            raise ValueError("Runtime action transform differs from planner metadata")
        target = self.step(frame, array(base.sim.data.qpos[index]), array(base.sim.data.qvel[index]))
        result = actions.clone()
        result[index] = torch.as_tensor(target, device=actions.device, dtype=actions.dtype)
        return result
