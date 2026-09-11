from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from pathlib import Path


def load_rollout_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "rollout_mjlab_motion_smoke.py"
    spec = importlib.util.spec_from_file_location("mimicx_rollout_seed_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ghost_export_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "export_mjlab_ghost_only_video.py"
    spec = importlib.util.spec_from_file_location("mimicx_ghost_export_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rollout_parser_accepts_explicit_seed(monkeypatch) -> None:
    module = load_rollout_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["rollout", "--motion-file", "motion.npz", "--seed", "1234"],
    )

    args = module.parse_args()

    assert args.seed == 1234


def test_rollout_parser_accepts_explicit_ghost_rgba(monkeypatch) -> None:
    module = load_rollout_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollout",
            "--motion-file",
            "motion.npz",
            "--ghost-color",
            "0.831,0.357,0.298,0.55",
        ],
    )

    args = module.parse_args()

    assert args.ghost_color == (0.831, 0.357, 0.298, 0.55)


def test_motion_command_types_include_native_overlay(monkeypatch) -> None:
    module = load_rollout_module()

    class UpstreamMotionCommandCfg:
        pass

    class NativeMotionCommandCfg:
        pass

    upstream = ModuleType("mjlab.tasks.tracking.mdp")
    upstream.MotionCommandCfg = UpstreamMotionCommandCfg
    native = ModuleType("src.tasks.tracking.mdp")
    native.MotionCommandCfg = NativeMotionCommandCfg
    monkeypatch.setitem(sys.modules, "mjlab.tasks.tracking.mdp", upstream)
    monkeypatch.setitem(sys.modules, "src.tasks.tracking.mdp", native)

    command_types = module.motion_command_cfg_types()

    assert UpstreamMotionCommandCfg in command_types
    assert NativeMotionCommandCfg in command_types
    assert isinstance(NativeMotionCommandCfg(), command_types)


def test_ghost_export_motion_command_types_include_native_overlay(monkeypatch) -> None:
    module = load_ghost_export_module()

    class UpstreamMotionCommandCfg:
        pass

    class NativeMotionCommandCfg:
        pass

    upstream = ModuleType("mjlab.tasks.tracking.mdp")
    upstream.MotionCommandCfg = UpstreamMotionCommandCfg
    native = ModuleType("src.tasks.tracking.mdp")
    native.MotionCommandCfg = NativeMotionCommandCfg
    monkeypatch.setitem(sys.modules, "mjlab.tasks.tracking.mdp", upstream)
    monkeypatch.setitem(sys.modules, "src.tasks.tracking.mdp", native)

    command_types = module.motion_command_cfg_types()

    assert UpstreamMotionCommandCfg in command_types
    assert NativeMotionCommandCfg in command_types
