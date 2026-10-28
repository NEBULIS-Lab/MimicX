"""Full-body contact reporting for Lab3's nested URDF link hierarchy."""
import re

from isaaclab.sensors.contact_sensor import BaseContactSensor
from isaaclab.sim.utils.queries import resolve_matching_prims_from_source
from isaaclab_physx.physics import PhysxManager
from isaaclab_physx.sensors.contact_sensor.contact_sensor import ContactSensor
from mimicx.backends.isaaclab.compat import contact_paths_by_env


def spawn_with_full_contact_reports(prim_path, cfg, *args, **kwargs):
    from pxr import PhysxSchema, Usd, UsdPhysics
    from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

    prim = G1_CYLINDER_CFG.spawn.func(prim_path, cfg, *args, **kwargs)
    # Lab's nested-schema helper stops descending once it applies to a rigid
    # parent. URDF link trees need reporters on its rigid descendants as well.
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(child).CreateThresholdAttr(0.0)
    return prim


class FullBodyContactSensor(ContactSensor):
    def _initialize_impl(self):
        from pxr import Usd, UsdPhysics

        BaseContactSensor._initialize_impl(self)
        if self.cfg.filter_prim_paths_expr or self.cfg.track_contact_points or self.cfg.track_friction_forces:
            raise ValueError("FullBodyContactSensor supports unfiltered net forces only")
        self._physics_sim_view = PhysxManager.get_physics_sim_view()
        parent, leaf = self.cfg.prim_path.rsplit("/", 1)
        matches = resolve_matching_prims_from_source(parent)
        if len(matches) != 1:
            raise ValueError("Contact sensor requires exactly one robot source subtree")
        root, destination = matches[0]
        source = root.GetPath().pathString
        paths = []
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and re.fullmatch(leaf, prim.GetName()):
                if "PhysxContactReportAPI" not in prim.GetAppliedSchemas():
                    raise RuntimeError(f"Missing contact reporter on {prim.GetPath()}")
                paths.append(destination + prim.GetPath().pathString[len(source):])
        if not paths:
            raise RuntimeError("No rigid links found for full-body contact sensing")
        # Explicit env-major paths avoid flattening nested links into fake siblings.
        concrete = contact_paths_by_env(paths, self._num_envs)
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(concrete)
        self._contact_view = self._physics_sim_view.create_rigid_contact_view(
            concrete, filter_patterns=[],
            max_contact_data_count=self.cfg.max_contact_data_count_per_prim * len(concrete))
        if list(self._body_physx_view.prim_paths) != concrete:
            raise RuntimeError("PhysX contact view did not preserve explicit env-major body ordering")
        self._num_sensors = len(paths)
        self._create_buffers()
