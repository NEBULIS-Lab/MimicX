"""Recorded robot states with a dynamic ball, not a trained ball-control policy.

MuJoCo is imported only when constructing BallReplay. Auditing and initial-state
utilities require NumPy only. No viewer, renderer, policy or GPU is initialized.
"""
from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from .baseline_recording import named_joint_addresses, validate_times


@dataclass(frozen=True)
class BallConfig:
    dt: float = 1 / 1200
    mass: float = .057
    radius: float = .0335
    damping: float = .18
    friction: float = .35
    stiffness: float = 1_000_000.
    gravity: tuple[float, float, float] = (0., 0., -9.81)
    floor_z: float | None = 0.
    robot_armature: float = 1_000_000.

    def __post_init__(self):
        positive = [self.dt, self.mass, self.radius, self.damping, self.stiffness, self.robot_armature]
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("Ball timestep, mass, radius, damping, stiffness and armature must be positive")
        if not np.isfinite(self.friction) or self.friction < 0:
            raise ValueError("Friction must be nonnegative")
        if np.shape(self.gravity) != (3,) or not np.isfinite(self.gravity).all():
            raise ValueError("Gravity must be a finite three-vector")
        if self.floor_z is not None and not np.isfinite(self.floor_z):
            raise ValueError("Floor height must be finite")


def _numbers(values):
    return " ".join(f"{x:.17g}" for x in np.asarray(values).ravel())


def ballistic_initial(position, target, *, flight_time, gravity=(0, 0, -9.81)):
    position, target, gravity = (np.asarray(v, dtype=float) for v in (position, target, gravity))
    if (any(v.shape != (3,) or not np.isfinite(v).all() for v in (position, target, gravity))
            or not np.isfinite(flight_time) or flight_time <= 0):
        raise ValueError("Finite 3D points and positive flight time are required")
    velocity = (target - position - .5 * gravity * flight_time ** 2) / flight_time
    return np.r_[position, velocity]


def group_contacts(contacts, *, dt, max_gap=.04, min_normal_force=.05):
    if not np.isfinite([dt, max_gap, min_normal_force]).all() or dt <= 0 or min(max_gap, min_normal_force) < 0:
        raise ValueError("Invalid contact grouping thresholds")
    groups = []
    for event in sorted(contacts, key=lambda event: event["time"]):
        if event["normal_force_n"] < min_normal_force:
            continue
        previous = next((g for g in reversed(groups) if g["geom"] == event["geom"]), None)
        if previous is None or event["time"] - previous["end"] > max_gap:
            previous = dict(geom=event["geom"], start=event["time"], end=event["time"],
                            normal_impulse_ns=0., impulse_world_ns=[0., 0., 0.])
            groups.append(previous)
        step_dt = float(event.get("dt", dt))
        previous["end"] = event["time"]
        previous["normal_impulse_ns"] += float(event["normal_force_n"]) * step_dt
        previous["impulse_world_ns"] = (np.asarray(previous["impulse_world_ns"]) +
                                        np.asarray(event["force_world_n"]) * step_dt).tolist()
    return groups


def audit_replay(out, *, mass, gravity=(0, 0, -9.81), tolerance=1e-6):
    """Reject non-Euler teleports, unsupported impulses and inconsistent contacts."""
    if not np.isfinite([mass, tolerance]).all() or min(mass, tolerance) <= 0:
        raise ValueError("Mass and tolerance must be positive")
    times = validate_times(out["time"], "ball clock")
    dt = np.diff(times)[:, None]
    n = len(times)
    shapes = dict(position=(n, 3), velocity=(n, 3), quaternion_wxyz=(n, 4),
                  constraint_force=(n - 1, 3), contact_count=(n - 1,))
    arrays = {k: np.asarray(out[k]) for k in shapes}
    if any(arrays[k].shape != shape or not np.isfinite(arrays[k]).all() for k, shape in shapes.items()):
        raise ValueError("Invalid replay arrays")
    p, v, q, force, count = (arrays[k] for k in shapes)
    gravity = np.asarray(gravity, dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("Invalid gravity")
    if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Ball quaternions are not unit wxyz")
    if np.any(count < 0) or np.any(count != np.floor(count)):
        raise ValueError("Contact counts must be nonnegative integers")
    integration = float(np.max(np.abs(np.diff(p, axis=0) - dt * v[1:])))
    delta = np.diff(v, axis=0)
    residual = float(np.linalg.norm(delta - dt * (gravity + force / mass), axis=1).max())
    unexplained = int(np.sum((np.linalg.norm(delta - dt * gravity, axis=1) > tolerance) & (count == 0)))
    if integration > tolerance or residual > tolerance or unexplained:
        raise ValueError("Ball replay has a teleport, force imbalance or unsupported velocity change")
    contacts = out.get("contacts", [])
    event_force = np.zeros_like(force)
    event_count = np.zeros(n - 1, dtype=int)
    for event in contacts:
        step = int(event["step"])
        normal = np.asarray(event["normal_world"], dtype=float)
        world = np.asarray(event["force_world_n"], dtype=float)
        if (step != event["step"] or not 0 <= step < n - 1
                or normal.shape != (3,) or world.shape != (3,)
                or not np.isfinite(np.r_[normal, world, event["time"], event["normal_force_n"]]).all()
                or abs(event["time"] - times[step]) > 1e-10
                or not np.isclose(np.linalg.norm(normal), 1, atol=1e-7)
                or event["normal_force_n"] < 0
                or abs(np.dot(normal, world) - event["normal_force_n"]) > tolerance):
            raise ValueError("Invalid contact time, normal or force")
        event_dt = float(event.get("dt", dt[step, 0]))
        if not np.isfinite(event_dt) or abs(event_dt - dt[step, 0]) > 1e-10:
            raise ValueError("Contact duration does not match the integration clock")
        event_force[step] += world
        event_count[step] += 1
    if not np.array_equal(event_count, count) or not np.allclose(event_force, force, atol=tolerance, rtol=1e-7):
        raise ValueError("Saved contact events do not explain constraint forces")
    return dict(integration_residual_m=integration, force_balance_residual_max_m_s=residual,
                unexplained_velocity_change_substeps=unexplained, contact_substeps=int(np.sum(count > 0)),
                max_speed_m_s=float(np.linalg.norm(v, axis=1).max()),
                scope="Recorded robot states and a dynamic ball; not a trained ball policy")


class BallReplay:
    """Prescribe one free-root robot and integrate one free ball in a supplied MJCF.

    Existing enabled collision geoms are retained; visual-only geoms stay disabled.
    Include racket/foot collision geometry in the caller's model. Assets can be
    supplied as a MuJoCo VFS mapping, without copying them into this package.
    """

    def __init__(self, xml, timestamps, root_pose, joint_positions, joint_names, *, config=None, assets=None):
        import mujoco
        from .source_camera import resample

        self.config = config or BallConfig()
        self.timestamps = validate_times(timestamps, "robot clock").copy()
        self.root_pose = np.asarray(root_pose, dtype=float).copy()
        self.joint_positions = np.asarray(joint_positions, dtype=float).copy()
        self.joint_names = list(joint_names)
        if self.joint_positions.shape != (len(self.timestamps), len(self.joint_names)):
            raise ValueError("Joint array does not match named joints")
        resample(self.timestamps, self.root_pose, self.joint_positions, self.timestamps, max_endpoint_hold=0)
        root = ET.fromstring(xml)
        for tag in ("actuator", "sensor", "contact", "equality", "keyframe", "tendon"):
            for child in root.findall(tag):
                root.remove(child)
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        c = self.config
        option.attrib.update(timestep=str(c.dt), integrator="Euler", gravity=_numbers(c.gravity),
                             iterations="60", tolerance="1e-10", solver="Newton")
        world = root.find("worldbody")
        if world is None:
            raise ValueError("MJCF requires a worldbody")
        if c.floor_z is not None:
            ET.SubElement(world, "geom", name="replay_floor", type="plane", size="100 100 .1",
                          pos=_numbers([0, 0, c.floor_z]))
        body = ET.SubElement(world, "body", name="replay_dynamic_ball", pos="0 0 1")
        ET.SubElement(body, "freejoint", name="replay_ball_free")
        ET.SubElement(body, "geom", name="replay_ball", type="sphere", size=str(c.radius), mass=str(c.mass),
                      contype="2", conaffinity="1", condim="3", priority="2")
        self.model = m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=assets)
        self.ball_geom = m.geom("replay_ball").id
        self.ball_body = m.body("replay_dynamic_ball").id
        self.joint_qpos, self.joint_qvel = named_joint_addresses(m, joint_names)
        if (m.nq != 14 + len(joint_names) or m.nv != 12 + len(joint_names)
                or m.jnt_type[0] != 0 or m.jnt_qposadr[m.joint("replay_ball_free").id] != m.nq - 7):
            raise ValueError("Expected exactly one free-root robot with scalar joints and the appended ball")
        # Prescribed robot inertial response is suppressed, then its exact state is
        # restored after every integration step. The ball is never reset mid-flight.
        m.dof_armature[:-6] = c.robot_armature
        m.dof_armature[-6:] = 0
        m.dof_damping[:] = 0
        m.dof_frictionloss[:] = 0
        m.body_gravcomp[:] = 0
        m.jnt_limited[:] = False
        for i in range(m.ngeom):
            if i == self.ball_geom or m.geom_contype[i] or m.geom_conaffinity[i]:
                m.geom_contype[i], m.geom_conaffinity[i] = ((2, 1) if i == self.ball_geom else (1, 2))
                m.geom_condim[i] = 3
                m.geom_friction[i] = [c.friction, .003, .0005]
                m.geom_solref[i] = [-c.stiffness, -2000 * c.damping]
                m.geom_solimp[i] = [.999, .999, .001, .5, 2]
                m.geom_margin[i] = m.geom_gap[i] = 0
        self.data = mujoco.MjData(m)
        self.robot_qpos = self._poses(self.timestamps)

    def _poses(self, query):
        from .source_camera import resample

        packed = resample(self.timestamps, self.root_pose, self.joint_positions, query, max_endpoint_hold=0)
        qpos = np.tile(self.model.qpos0[:-7], (len(query), 1))
        qpos[:, :7] = packed[:, :7]
        qpos[:, self.joint_qpos] = packed[:, 7:]
        return qpos

    def simulate(self, initial, *, start=None, end=None):
        import mujoco

        c, m, d = self.config, self.model, self.data
        initial = np.asarray(initial, dtype=float)
        start = float(self.timestamps[0] if start is None else start)
        end = float(self.timestamps[-1] if end is None else end)
        if initial.shape not in ((6,), (9,)) or not np.isfinite(initial).all():
            raise ValueError("Initial ball state must contain XYZ, linear velocity and optional angular velocity")
        if not self.timestamps[0] <= start < end <= self.timestamps[-1]:
            raise ValueError("Ball interval must be covered by the recorded robot")
        times = start + np.arange(int(np.ceil((end - start) / c.dt))) * c.dt
        times = np.r_[times[times < end - 1e-12], end]
        if len(times) < 2:
            raise ValueError("Ball interval is too short")
        qpos = self._poses(times)
        velocities = np.zeros((len(times) - 1, m.nv))
        before, after = m.qpos0.copy(), m.qpos0.copy()
        for i, dt in enumerate(np.diff(times)):
            before[:-7], after[:-7] = qpos[i], qpos[i + 1]
            mujoco.mj_differentiatePos(m, velocities[i], dt, before, after)
        mujoco.mj_resetData(m, d)
        d.qpos[:-7], d.qvel[:-6] = qpos[0], velocities[0, :-6]
        d.qpos[-7:] = np.r_[initial[:3], 1, 0, 0, 0]
        d.qvel[-6:] = np.r_[initial[3:6], initial[6:] if len(initial) == 9 else np.zeros(3)]
        d.time = start
        mujoco.mj_forward(m, d)
        initial_contacts = [dict(distance_m=float(contact.dist)) for contact in d.contact
                            if self.ball_geom in contact.geom]
        positions, speed, rotations = [d.qpos[-7:-4].copy()], [d.qvel[-6:-3].copy()], [d.qpos[-4:].copy()]
        counts, forces, events = [], [], []
        contact_force = np.zeros(6)
        try:
            for i, dt in enumerate(np.diff(times)):
                m.opt.timestep = dt
                d.qpos[:-7], d.qvel[:-6], d.time = qpos[i], velocities[i, :-6], times[i]
                mujoco.mj_step(m, d)
                if abs(d.time - times[i + 1]) > 1e-10:
                    raise RuntimeError("Simulator reset during ball replay")
                count = 0
                for j, contact in enumerate(d.contact):
                    if self.ball_geom not in contact.geom:
                        continue
                    mujoco.mj_contactForce(m, d, j, contact_force)
                    if np.linalg.norm(contact_force[:3]) <= 1e-12:
                        continue
                    sign = 1 if contact.geom[1] == self.ball_geom else -1
                    frame = contact.frame.reshape(3, 3)
                    other = int(contact.geom[0] if sign == 1 else contact.geom[1])
                    events.append(dict(step=i, time=float(times[i]), dt=float(dt),
                                       geom=m.geom(other).name or f"geom_{other}",
                                       normal_force_n=float(contact_force[0]),
                                       normal_world=(sign * frame[0]).tolist(),
                                       force_world_n=(sign * frame.T @ contact_force[:3]).tolist(),
                                       distance_m=float(contact.dist), position=contact.pos.tolist()))
                    count += 1
                counts.append(count)
                forces.append(d.qfrc_constraint[-6:-3].copy())
                positions.append(d.qpos[-7:-4].copy())
                speed.append(d.qvel[-6:-3].copy())
                rotations.append(d.qpos[-4:].copy())
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    raise RuntimeError("Nonfinite physics state")
                d.qpos[:-7], d.qvel[:-6] = qpos[i + 1], velocities[min(i + 1, len(velocities) - 1), :-6]
        finally:
            m.opt.timestep = c.dt
        return dict(time=times, position=np.asarray(positions), velocity=np.asarray(speed),
                    quaternion_wxyz=np.asarray(rotations), constraint_force=np.asarray(forces),
                    contact_count=np.asarray(counts), contacts=events, initial_contacts=initial_contacts)


def fit_initial_conditions(engine, observed_times, observed_positions, *, lower, upper,
                           rounds=6, population=64, seed=0, start=None, end=None):
    """Bounded CEM fit of six initial-state values, never per-frame ball forcing.

    Observations are world XYZ in the unchanged robot coordinate system. Missing
    observation rows may be NaN. The objective is position RMSE, not policy return.
    """
    times = validate_times(observed_times, "observation clock")
    observed = np.asarray(observed_positions, dtype=float)
    low, high = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if observed.shape != (len(times), 3) or not np.isfinite(observed).all(axis=1).any():
        raise ValueError("Need at least one finite observed XYZ row")
    if (low.shape != (6,) or high.shape != (6,) or not np.isfinite([low, high]).all() or np.any(low > high)
            or int(rounds) != rounds or int(population) != population or rounds < 1 or population < 2):
        raise ValueError("Invalid initial-state bounds or search budget")
    begin = engine.timestamps[0] if start is None else start
    finish = engine.timestamps[-1] if end is None else end
    if times[0] < begin or times[-1] > finish:
        raise ValueError("Observations are outside the simulation interval")
    valid = np.isfinite(observed).all(axis=1)
    rng = np.random.default_rng(seed)
    mean, std = (low + high) / 2, (high - low) / 4
    minimum = std * .07
    best = None
    for _ in range(rounds):
        samples = np.clip(rng.normal(mean, std, (population, 6)), low, high)
        samples[0] = mean
        if best is not None:
            samples[1] = best[1]
        rows = []
        for initial in samples:
            out = engine.simulate(initial, start=start, end=end)
            predicted = np.column_stack([np.interp(times[valid], out["time"], out["position"][:, j]) for j in range(3)])
            error = float(np.mean(np.sum((predicted - observed[valid]) ** 2, axis=1)))
            rows.append((error, initial.copy()))
        rows.sort(key=lambda row: row[0])
        if best is None or rows[0][0] < best[0]:
            best = rows[0]
        elite = np.asarray([row[1] for row in rows[:max(2, population // 6)]])
        mean = .3 * mean + .7 * elite.mean(axis=0)
        std = np.maximum(minimum, .3 * std + .7 * elite.std(axis=0))
    return dict(initial=best[1].tolist(), position_rmse_m=float(np.sqrt(best[0])),
                trials=rounds * population, seed=seed, scope="Initial-state fit only; not policy training")
