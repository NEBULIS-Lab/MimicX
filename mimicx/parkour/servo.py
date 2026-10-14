"""Named position-servo controls and CPU-only model preparation."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@dataclass
class ActionAffine:
    """Action-space order and affine transform, independent of actuator order."""

    names: list[str]
    scale: object
    offset: object

    def __post_init__(self):
        self.names = list(self.names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Action names must be nonempty and unique")
        self.scale = np.broadcast_to(self.scale, (len(self.names),)).astype(float).copy()
        self.offset = np.broadcast_to(self.offset, (len(self.names),)).astype(float).copy()
        if not np.isfinite([self.scale, self.offset]).all() or np.any(self.scale == 0):
            raise ValueError("Action affine parameters must be finite with nonzero scale")

    def _order(self, actuator_names):
        if len(actuator_names) != len(self.names) or set(actuator_names) != set(self.names):
            raise ValueError("Actuator and action names must be a one-to-one mapping")
        return [self.names.index(n) for n in actuator_names]

    def to_controls(self, actions, actuator_names):
        values = np.asarray(actions, dtype=float)
        if values.shape[-1:] != (len(self.names),) or not np.isfinite(values).all():
            raise ValueError("Invalid action shape or nonfinite actions")
        return (values * self.scale + self.offset)[..., self._order(actuator_names)]

    def to_actions(self, controls, actuator_names):
        order = self._order(actuator_names)
        values = np.asarray(controls, dtype=float)
        if values.shape[-1:] != (len(order),) or not np.isfinite(values).all():
            raise ValueError("Invalid control shape or nonfinite controls")
        return (values[..., np.argsort(order)] - self.offset) / self.scale

    def as_dict(self):
        return dict(names=self.names, scale=self.scale.tolist(), offset=self.offset.tolist())


def load_actions(path, *, index=None, horizon=None, sha256=None):
    if sha256 is not None and digest(path) != sha256:
        raise ValueError("Action archive checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        values = archive["actions"].copy()
    if values.ndim == 3:
        if index is None or not 0 <= index < values.shape[1]:
            raise ValueError("An explicit valid controller index is required")
        values = values[:, index]
    elif index is not None:
        raise ValueError("A controller index is only valid for a batched archive")
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite actions with shape (frames, joints)")
    if horizon is not None and len(values) != horizon:
        raise ValueError("Action horizon does not match the requested horizon")
    return values


def load_model(*, model_xml=None, model_mjb=None):
    if (model_xml is None) == (model_mjb is None):
        raise ValueError("Supply exactly one of --model-xml or --model-mjb")
    import mujoco

    model = (mujoco.MjModel.from_xml_path(str(model_xml)) if model_xml is not None
             else mujoco.MjModel.from_binary_path(str(model_mjb)))
    return model


def actuator_names(model):
    import mujoco

    if model.na or np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
        raise ValueError("Planner requires stateless joint position actuators")
    ids = model.actuator_trnid[:, 0]
    if np.any(model.jnt_type[ids] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Only scalar hinge-joint actuators are supported; no root actuation")
    if (np.any(model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
            or np.any(model.actuator_biastype != mujoco.mjtBias.mjBIAS_AFFINE)
            or np.any(model.actuator_gainprm[:, 0] <= 0)
            or not np.allclose(model.actuator_biasprm[:, 1], -model.actuator_gainprm[:, 0])
            or not np.allclose(model.actuator_gear[:, 0], 1)):
        raise ValueError("Expected native unit-gear position servos, not torque actuators")
    return [model.joint(int(j)).name.removeprefix("robot/") for j in ids]


def make_environment_config(scene, motion, horizon=269):
    """The original play configuration; does not create a simulator/device."""
    from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_mimicx_curriculum_env_cfg
    from .environment import SceneBuilder

    cfg = unitree_g1_mimicx_curriculum_env_cfg(play=True)
    cfg.scene.terrain = None
    cfg.scene.spec_fn = SceneBuilder(str(Path(scene).resolve()))
    cfg.scene.extent = 12.
    cfg.scene.num_envs = 1
    cfg.sim.nconmax, cfg.sim.njmax = 192, 800
    cfg.episode_length_s = max(20., horizon / 50. + 2.)
    cmd = cfg.commands["motion"]
    cmd.motion_file = str(Path(motion).resolve())
    cmd.viz.ground_align = False
    cmd.sampling_mode = "start"
    cmd.joint_position_range = (-.01, .01)
    return cfg


def prepare_model(scene_path, motion_path, output):
    """Compile the backend's real robot and scene on CPU, without env initialization.

    Third-party robot assets must already be installed under their own license.
    Only local MJB and metadata are written; no meshes or weights are copied.
    """
    import mujoco
    from mjlab.scene import Scene
    from mjlab.utils.lab_api.string import resolve_matching_names_values

    cfg = make_environment_config(scene_path, motion_path)
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    robot = scene["robot"]
    action = cfg.actions["joint_pos"]
    _, names = robot.find_joints_by_actuator_names(action.actuator_names)

    def resolve(value, default):
        if not isinstance(value, dict):
            return np.full(len(names), value, dtype=float)
        result = np.full(len(names), default, dtype=float)
        ids, _, values = resolve_matching_names_values(value, names)
        result[ids] = values
        return result

    if action.clip is not None:
        raise ValueError("This release expects unclipped native joint targets")
    offset = (resolve(robot.cfg.init_state.joint_pos, 0.) if action.use_default_offset
              else resolve(action.offset, 0.))
    affine = ActionAffine(names, resolve(action.scale, 1.), offset)
    affine._order(actuator_names(model))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mujoco.mj_saveModel(model, str(output / "model.mjb"))
    save_json(output / "affine.json", affine.as_dict())
    save_json(output / "names.json", dict(joint_names=list(robot.joint_names),
              body_names=list(robot.body_names), reference_body_names=list(cfg.commands["motion"].body_names)))
    save_json(output / "model_manifest.json", dict(mujoco_version=mujoco.__version__,
              model_sha256=digest(output / "model.mjb"), scene_sha256=digest(scene_path),
              motion_sha256=digest(motion_path), timestep=model.opt.timestep, decimation=cfg.decimation,
              nq=model.nq, nv=model.nv, nu=model.nu, training_updates=0))
    return model
