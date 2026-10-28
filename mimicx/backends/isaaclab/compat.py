"""Narrow process-local bridges for audited upstream API renames.

Call after AppLauncher, before importing whole_body_tracking (whose package
initializer eagerly imports every task). No upstream files are changed.
"""
import re


def prepare_upstream_import(*, noise_module=None):
    if noise_module is None:
        import isaaclab.utils.noise as noise_module

    if hasattr(noise_module, "AdditiveUniformNoiseCfg"):
        return ()
    replacement = getattr(noise_module, "UniformNoiseCfg", None)
    if replacement is None:
        raise ImportError("Isaac noise API has neither AdditiveUniformNoiseCfg nor UniformNoiseCfg")
    # Lab3 UniformNoiseCfg retains n_min/n_max and operation='add' by default.
    noise_module.AdditiveUniformNoiseCfg = replacement
    return ("AdditiveUniformNoiseCfg=UniformNoiseCfg",)


def torch_view(value):
    """Lab2 tensors pass through; Lab3 ProxyArray exposes a zero-copy view."""
    return getattr(value, "torch", value)


def canonical_to_native_quaternion(value, *, xyzw):
    return value[..., [1, 2, 3, 0]] if xyzw else value


def native_to_canonical_quaternion(value, *, xyzw):
    return value[..., [3, 0, 1, 2]] if xyzw else value


def contact_paths_by_env(paths, num_envs):
    if not paths or len(set(paths)) != len(paths) or num_envs < 1:
        raise ValueError("Contact body paths must be unique and nonempty")
    concrete = []
    for env_id in range(num_envs):
        for path in paths:
            item, replacements = re.subn(r"/env_(?:\.\*|\*)/", f"/env_{env_id}/", path)
            if replacements != 1:
                raise ValueError("Full-body contacts require the standard env_<index> namespace")
            concrete.append(item)
    return concrete


class TorchDataView:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        # Reference velocities are at link origins, not centers of mass.
        if name in {"body_lin_vel_w", "body_ang_vel_w"}:
            link_name = name.replace("body_", "body_link_", 1)
            if hasattr(self._data, link_name):
                return torch_view(getattr(self._data, link_name))
        return torch_view(getattr(self._data, name))


class TorchArticulationView:
    """Only upstream MotionCommand sees this view; scene assets stay native."""
    def __init__(self, asset):
        self._asset = asset
        self.data = TorchDataView(asset.data)

    def __getattr__(self, name):
        return getattr(self._asset, name)

    def write_root_state_to_sim(self, state, *, env_ids=None):
        # The NPZ stores root link-origin velocity; the legacy root-state API
        # instead interprets its last six values as COM velocity.
        if not hasattr(self._asset.data.joint_pos, "torch"):
            self._asset.write_root_link_state_to_sim(state, env_ids=env_ids)
            return
        import torch
        from isaaclab.utils.math import quat_apply

        ids = slice(None) if env_ids is None else env_ids
        offset_b = torch_view(self._asset.data.body_com_pos_b)[ids, 0]
        offset_w = quat_apply(state[:, 3:7], offset_b)
        velocity = state[:, 7:13].clone()
        velocity[:, :3] += torch.linalg.cross(velocity[:, 3:6], offset_w, dim=-1)
        self._asset.write_root_link_pose_to_sim_index(root_pose=state[:, :7], env_ids=env_ids)
        # beta2's link-velocity setter submits the link buffer to the PhysX
        # COM-velocity API. Supply explicitly translated COM velocity instead.
        self._asset.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
