# SPDX-License-Identifier: Apache-2.0
"""MimicX's corrected, headless SONIC release and continuous-recording helpers.

Adapted from MimicX's corrected sonic_release/sonic_recording helpers and the
GEAR-SONIC viewer's startup-band release behavior. Upstream robot/simulator:
https://github.com/NVlabs/GR00T-WholeBodyControl (Apache-2.0).
No upstream policy, weights or robot assets are included in this module.
"""
import json
import math
from pathlib import Path
import time
from types import MethodType

import numpy as np


def release_if_requested(sim_env, request_path):
    if not request_path:
        return False
    request = Path(request_path)
    if not request.is_file():
        return False
    ack = Path(str(request) + ".ack.json")
    band = getattr(sim_env, "elastic_band", None)
    enabled = bool(band is not None and band.enable)
    if ack.exists():
        residual = band is not None and np.any(sim_env.mj_data.xfrc_applied[sim_env.band_attached_link])
        if enabled or residual:
            raise ValueError("stale release acknowledgement while support is active")
        return False
    if band is not None:
        band.enable = False
        sim_env.mj_data.xfrc_applied[sim_env.band_attached_link] = 0
    record = dict(previously_enabled=enabled, enabled=False,
                  simulation_time=float(sim_env.mj_data.time), monotonic_time=time.monotonic())
    temporary = ack.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(ack)
    return True


def install_no_reset_fall_monitor(sim_env):
    def check_fall(env):
        env.fall = bool(env.mj_data.qpos[2] < .2)
    sim_env.check_fall = MethodType(check_fall, sim_env)


def require_forward_time(previous, current):
    if not math.isfinite(previous) or not math.isfinite(current) or current <= previous:
        raise ValueError(f"simulator clock reset/stalled: {previous} -> {current}")


class DdsResetEcho:
    """Simulator-only application-level receipt proof using the SDK's header echo.

    G1Deploy stores LowState before echoing mode_machine in LowCmd. The simulated
    bridge does not use that header to compute torques. Restore the original
    value and prove unchanged motor commands before permitting CONTROL.
    """
    def __init__(self, bridge, *, monotonic_time, timeout_s=.25):
        self.bridge = bridge
        self.original = int(bridge.low_state.mode_machine)
        self.marker = self.original ^ 1
        self.started = monotonic_time
        self.timeout_s = timeout_s
        self.stage = "marker"
        self.commands = self.command_signature()
        self.events = dict(original_header=self.original, marker_header=self.marker,
                           reset_publish_monotonic=monotonic_time)
        bridge.low_state.mode_machine = self.marker

    def command_signature(self):
        return np.asarray([[getattr(m, f) for f in ("q", "dq", "kp", "kd", "tau")]
                           for m in self.bridge.low_cmd.motor_cmd[:29]], dtype=float)

    def observe(self, *, monotonic_time):
        if monotonic_time - self.started > self.timeout_s:
            self.bridge.low_state.mode_machine = self.original
            raise ValueError("DDS post-reset application receipt/restore timeout")
        header = int(self.bridge.low_cmd.mode_machine)
        if self.stage == "marker" and header == self.marker:
            self.events["marker_received_monotonic"] = monotonic_time
            self.bridge.low_state.mode_machine = self.original
            self.stage = "restore"
        elif self.stage == "restore" and header == self.original:
            if not np.array_equal(self.commands, self.command_signature()):
                raise ValueError("motor command changed before CONTROL during reset receipt handshake")
            self.events.update(restored_received_monotonic=monotonic_time,
                               original_header_restored=True, motor_commands_unchanged=True)
            self.stage = "done"
            return dict(self.events)
        return None


class StartupPublications:
    """Capture actual source/wire values without changing publication or clocks."""
    def __init__(self, bridge, physical_state, *, joint_names, capture_seconds=.5):
        self.rows = {"lowstate": [], "torso": []}
        self.reset = None
        self.current = None
        self.sequence = 0
        self.capture_seconds = capture_seconds
        self.joint_names = list(joint_names)
        original = bridge.PublishLowState

        def publish(obs):
            active = self.reset is None or time.monotonic() - self.reset["monotonic_time"] <= capture_seconds
            if active:
                self.sequence += 1
                self.current = dict(sequence=self.sequence, source_time=float(obs["time"]),
                                    source_monotonic=time.monotonic(), **physical_state())
            try:
                return original(obs)
            finally:
                self.current = None
        bridge.PublishLowState = publish
        for topic, publisher in (("lowstate", bridge.low_state_puber), ("torso", bridge.torso_imu_puber)):
            write = publisher.Write

            def record(sample, *args, _topic=topic, _write=write, **kwargs):
                source = self.current
                if source is None:
                    return _write(sample, *args, **kwargs)
                row = {k: np.asarray(v).copy() if isinstance(v, np.ndarray) else v for k, v in source.items()}
                wire = lambda v: np.asarray(v, dtype=np.float32).astype(float)
                if _topic == "lowstate":
                    row.update(tick=int(sample.tick), mode_machine=int(sample.mode_machine),
                               wire_motor_count=len(sample.motor_state),
                               q=wire([m.q for m in sample.motor_state[:29]]),
                               dq=wire([m.dq for m in sample.motor_state[:29]]),
                               quat=wire(sample.imu_state.quaternion))
                else:
                    row.update(quat=wire(sample.quaternion), gyro=wire(sample.gyroscope),
                               accel=wire(sample.accelerometer))
                row["write_start"] = time.monotonic()
                result = _write(sample, *args, **kwargs)
                row.update(write_end=time.monotonic(), write_success=result is not False)
                self.rows[_topic].append(row)
                return result
            publisher.Write = record

    def start(self, initialization):
        if self.reset is not None:
            raise ValueError("publication evidence cannot reset during an episode")
        self.reset = {k: initialization[k] for k in ("monotonic_time", "simulation_time")}

    def save(self, folder):
        folder = Path(folder)
        arrays = dict(joint_names=np.asarray(self.joint_names), motor_indices=np.arange(len(self.joint_names)))
        for topic, rows in self.rows.items():
            if rows:
                arrays.update({f"{topic}_{k}": np.asarray([r[k] for r in rows]) for k in rows[0]})
        np.savez_compressed(folder / "startup_publications.npz", **arrays)
        (folder / "startup_publications.json").write_text(json.dumps(dict(
            schema="mimicx.sonic-startup-publications.v1", reset=self.reset,
            capture_seconds=self.capture_seconds, actual_float32_payload=True,
            clocks_modified=False, physics_steps_added=0), indent=2) + "\n")


def instrument_startup_cpp(source):
    """Optional logging-only patch for the released deploy source; never edits it."""
    read = "      auto imu_data = imu_torso_buffer_.GetDataWithTime();"
    call = "        state_logger_->LogFullState(base_quat,"
    end = "                                    ros_timestamp);\n      }\n      return true;\n    }"
    if any(source.count(token) != 1 for token in (read, call, end)):
        raise ValueError("unsupported deploy source: startup logger anchors differ")
    source = "#include <fstream>\n#include <iomanip>\n#include <cstdlib>\n" + source
    source = source.replace(read, read + "\n      const auto mimicx_control_read = std::chrono::steady_clock::now();")
    source = source.replace(call, "        auto mimicx_state_index = state_logger_->LogFullState(base_quat,")
    extra = r'''
        if (mimicx_state_index == 0) {
          const char* path = std::getenv("MIMICX_STARTUP_STATE_LOG");
          if (path) {
            std::ofstream out(path);
            out << "index,control_read_monotonic,lowstate_received_monotonic,torso_received_monotonic,lowstate_tick,mode_machine";
            for (int i=0;i<3;i++) out << ",torso_gyro_" << i;
            for (int i=0;i<3;i++) out << ",torso_accel_" << i;
            out << "\n" << std::fixed << std::setprecision(9) << mimicx_state_index << ","
                << std::chrono::duration<double>(mimicx_control_read.time_since_epoch()).count() << ","
                << std::chrono::duration<double>(low_state_data.timestamp.time_since_epoch()).count() << ","
                << std::chrono::duration<double>(imu_data.timestamp.time_since_epoch()).count() << ","
                << ls->tick() << "," << unsigned(ls->mode_machine());
            for (int i=0;i<3;i++) out << "," << imu_torso->gyroscope()[i];
            for (int i=0;i<3;i++) out << "," << imu_torso->accelerometer()[i];
            out << "\n";
          }
        }
'''
    return source.replace(end, "                                    ros_timestamp);" + extra + "      }\n      return true;\n    }")
