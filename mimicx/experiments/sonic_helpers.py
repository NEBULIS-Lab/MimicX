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
