"""Terminal-safe rollout metrics and compact state export (no Isaac imports)."""
import csv
import json
from pathlib import Path
import numpy as np
from mimicx.backends.isaaclab.compat import torch_view, native_to_canonical_quaternion
from mimicx.refinement.manifest import hash_file
from mimicx.refinement.state import atomic_write_json
from mimicx.runtime.backends import isaac_rollout_outputs


SCHEMA = "mimicx.isaac-rollout.v2"


class RolloutTelemetry:
    def __init__(self):
        self.rows = []
        self.states = []
        self.episode_id = 0
        self.episode_step = 0

    def capture(self, env):
        """Called immediately after termination computation, BEFORE auto reset."""
        command = env.command_manager.get_term("motion")
        as_array = lambda x: torch_view(x).detach().cpu().numpy().copy()
        ref = as_array(command.body_pos_w)[0]
        robot = as_array(command.robot_body_pos_w)[0]
        relative = as_array(command.body_pos_relative_w)[0]
        delta = relative - robot
        root_index = command.cfg.body_names.index("pelvis")
        manager = env.termination_manager
        scalar = lambda value: bool(torch_view(value)[0].item())
        done = scalar(manager.dones)
        terminated = scalar(getattr(manager, "terminated", manager.dones))
        timeouts = getattr(manager, "time_outs", None)
        truncated = scalar(timeouts) if timeouts is not None else False
        dt = float(getattr(env, "step_dt", 0.02))
        self.episode_step += 1
        row = {"step": len(self.rows) + 1,
               "episode_id": self.episode_id, "episode_step": self.episode_step,
               "sample_time_s": (len(self.rows) + 1) * dt,
               "episode_time_s": self.episode_step * dt,
               "command_time_s": int(command.time_steps[0].item()) * dt,
               "motion_frame": int(command.time_steps[0].item()),
               "done_any": done, "terminated_any": terminated, "truncated_any": truncated,
               "error_anchor_pos": float(np.linalg.norm(as_array(command.anchor_pos_w - command.robot_anchor_pos_w)[0])),
               "error_body_pos": float(np.linalg.norm(delta, axis=-1).mean()),
               "world_body_pos_error": float(np.linalg.norm(ref - robot, axis=-1).mean()),
               "world_root_pos_error": float(np.linalg.norm(ref[root_index] - robot[root_index])),
               "body_pos_error_max": float(np.linalg.norm(delta, axis=-1).max()),
               "body_z_error_max": float(np.abs(delta[:, 2]).max()),
               "error_joint_pos": float(np.linalg.norm(as_array(command.joint_pos - command.robot_joint_pos)[0]))}
        for index, name in enumerate(command.cfg.body_names):
            row[f"body_pos_error_{name}"] = float(np.linalg.norm(delta[index]))
            row[f"body_z_error_{name}"] = float(abs(delta[index, 2]))
        for name in env.termination_manager.active_terms:
            row[f"termination_{name}"] = bool(env.termination_manager.get_term(name)[0].item())
        self.rows.append(row)
        asset = env.scene["robot"]
        self.states.append({"joint_pos": as_array(asset.data.joint_pos)[0],
                            "joint_vel": as_array(asset.data.joint_vel)[0],
                            "body_pos_w": as_array(asset.data.body_pos_w)[0] - as_array(env.scene.env_origins)[0],
                            "body_quat_w": as_array(asset.data.body_quat_w)[0],
                            "reference_body_pos_w": ref - as_array(env.scene.env_origins)[0]})
        if hasattr(asset.data.body_quat_w, "torch"):
            self.states[-1]["body_quat_w"] = native_to_canonical_quaternion(
                self.states[-1]["body_quat_w"], xyzw=True)
        for key in ("body_lin_vel_w", "body_ang_vel_w"):
            link_key = key.replace("body_", "body_link_", 1)
            self.states[-1][key] = as_array(getattr(asset.data, link_key, getattr(asset.data, key)))[0]
        sensors = getattr(env.scene, "sensors", {})
        if "contact_forces" in sensors:
            self.states[-1]["contact_forces_w"] = as_array(sensors["contact_forces"].data.net_forces_w)[0]
        if done:
            self.episode_id += 1
            self.episode_step = 0

    def write(self, metrics_path, steps_path, *, metadata):
        if not self.rows:
            raise ValueError("No physical rollout samples collected")
        metrics_path, steps_path, states_path, completion_path = isaac_rollout_outputs(metrics_path, steps_path)
        completion_path.unlink(missing_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        steps_path.parent.mkdir(parents=True, exist_ok=True)
        if len(self.states) != len(self.rows):
            raise ValueError("Each rollout row requires one captured pre-reset state")
        failures = [r["step"] for r in self.rows if r["done_any"]]
        metrics = {key: float(np.mean([r[key] for r in self.rows]))
                   for key in self.rows[0] if key.startswith(("error_", "world_", "body_"))}
        payload = {"schema": SCHEMA, "backend": "isaaclab", "steps": len(self.rows),
                   "done_any_count": len(failures), "first_done_step": failures[0] if failures else None,
                   "terminated_count": sum(r["terminated_any"] for r in self.rows),
                   "truncated_count": sum(r["truncated_any"] for r in self.rows),
                   "reward_mean_avg": float(np.mean([r["rew_mean"] for r in self.rows])),
                   "reference_clock": "post-physics state versus action-command frame, before reference advance",
                   "metrics": metrics}
        if payload.keys() & metadata.keys():
            raise ValueError("Metadata cannot override rollout results")
        payload.update(metadata)
        # Validate JSON before touching existing artifacts or committing success.
        json.dumps(payload, allow_nan=False)
        arrays = {key: np.stack([state[key] for state in self.states]) for key in self.states[0]}
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("Rollout states/actions must be finite")
        for key in ("episode_id", "episode_step", "motion_frame", "sample_time_s", "episode_time_s", "command_time_s"):
            arrays[key] = np.asarray([row[key] for row in self.rows])
        arrays["quaternion_order"] = np.asarray("wxyz")
        arrays["velocity_reference"] = np.asarray("link_origin")
        for key in ("joint_names", "body_names", "contact_body_names", "reference_body_names", "fps"):
            if key in metadata:
                arrays[key] = np.asarray(metadata[key])
        with steps_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        np.savez_compressed(states_path, **arrays)
        atomic_write_json(metrics_path, payload)
        # Kit can exit 0 during shutdown even after an exception. This marker
        # commits the complete JSON/CSV/state set, not just the metrics prefix.
        atomic_write_json(completion_path, {
            "schema": SCHEMA, "steps": len(self.rows),
            "artifact_sha256": {path.name: hash_file(path) for path in (metrics_path, steps_path, states_path)},
            "motion_sha256": metadata.get("motion_sha256"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        })
        return payload
