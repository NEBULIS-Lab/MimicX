"""Portable controller contracts; no simulator or actor dependency required."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def module(name):
    return importlib.import_module("mimicx.parkour." + name)


def test_quaternion_error_is_sign_invariant_and_uses_wxyz():
    error = module("predictive").orientation_error
    assert error([2, 0, 0, 0], [-1, 0, 0, 0]) == 0
    assert error([1, 0, 0, 0], [0, 1, 0, 0]) == pytest.approx(np.pi)
    with pytest.raises(ValueError, match="quaternion"):
        error([0, 0, 0, 0], [1, 0, 0, 0])


def test_knots_include_endpoints_and_single_frame():
    interpolate = module("predictive").interpolate
    knots = np.array([[[0.], [2.], [0.]]])
    np.testing.assert_allclose(interpolate(knots, 5)[0, :, 0], [0, 1, 2, 1, 0])
    np.testing.assert_allclose(interpolate(knots, 1), [[[0.]]])


def test_affine_maps_action_order_to_actuator_order_without_clipping():
    affine = module("servo").ActionAffine(["knee", "hip"], [2., .5], [.1, -.2])
    target = affine.to_controls([3., 4.], ["hip", "knee"])
    np.testing.assert_allclose(target, [1.8, 6.1])
    np.testing.assert_allclose(affine.to_actions(target, ["hip", "knee"]), [3, 4])
    with pytest.raises(ValueError):
        affine.to_controls([3., 4.], ["hip", "hip"])
    with pytest.raises(ValueError):
        module("servo").ActionAffine(["knee"], 0., 0.)


def test_action_archive_requires_explicit_valid_controller_and_hash(tmp_path):
    servo = module("servo")
    path = tmp_path / "traces.npz"
    np.savez(path, actions=np.arange(24.).reshape(4, 3, 2))
    np.testing.assert_array_equal(servo.load_actions(path, index=2, horizon=4),
                                  [[4, 5], [10, 11], [16, 17], [22, 23]])
    with pytest.raises(ValueError, match="index"):
        servo.load_actions(path)
    with pytest.raises(ValueError, match="checksum"):
        servo.load_actions(path, index=2, sha256="0" * 64)
    with pytest.raises(ValueError, match="horizon"):
        servo.load_actions(path, index=2, horizon=5)


def test_contact_penalty_preserves_brief_violation():
    penalty = module("predictive").penetration_penalty
    np.testing.assert_allclose(penalty([[0, .01, .03], [0, 0, 0]], .01, 5000),
                               [2 + 2 / 30, 0])


def test_nominal_recovery_subtracts_only_executed_replan_segment():
    p = module("predictive")
    affine = module("servo").ActionAffine(["left_knee_joint", "right_knee_joint"], 2., 1.)
    coeff = np.zeros((3, 15))
    coeff[:, 2] = [0, 1, 2]
    history = {"replans": [dict(frame=1, horizon=3, coefficients=coeff.ravel().tolist()),
                            dict(frame=3, horizon=1, coefficients=np.zeros(45).tolist())]}
    actions = np.array([[7, 8], [2, 3], [2.5, 3.5], [2, 3]])
    recovered = p.recover_nominal(actions, history, affine, affine.names, start=1)
    np.testing.assert_allclose(recovered, [[7, 8], [2, 3], [2, 3], [2, 3]])


def test_completion_keeps_horizon_terminal_and_motion_separate():
    report = module("hybrid").completion_report
    values = np.zeros((270, 6))
    good = report(values, 269)
    assert good["motion_complete"] and good["terminal_pass"]
    values[225:260, 0] = 1.
    result = report(values, 269)
    assert result["horizon_complete"] and result["terminal_pass"]
    assert not result["motion_complete"]
    failed = report(np.zeros((270, 6)), 220)
    assert not failed["terminal_pass"] and failed["objective"] > good["objective"]
    with pytest.raises(ValueError):
        report(np.zeros((10, 6)), 269)


def test_six_reset_hook_uses_fresh_calls_and_does_not_claim_survival_as_motion():
    hybrid = module("hybrid")
    def evaluate(seed):
        metrics = np.zeros((270, 6))
        metrics[225:, 0] = 1. if seed == 29003 else 0.
        return {"seed": seed, "completion": hybrid.completion_report(metrics, 269)}
    result = hybrid.evaluate_six_resets(evaluate)
    assert result["seeds"] == [28002, 28003, 28004, 29001, 29002, 29003]
    assert result["horizon_complete"] == 6
    assert result["all_motion_checks_pass"] == 5
    assert result["training_updates"] == 0
    with pytest.raises(ValueError, match="six distinct"):
        hybrid.evaluate_six_resets(evaluate, [1] * 6)


def test_planner_config_rejects_typos_and_invalid_budget():
    config = module("predictive").PlannerConfig
    with pytest.raises((TypeError, ValueError)):
        config.from_dict({"horizn": 45})
    with pytest.raises(ValueError):
        config(population=4)
    with pytest.raises(ValueError):
        config(replan_every=0)


def test_cli_help_does_not_load_gpu_runtime():
    script = Path(__file__).resolve().parents[1] / "scripts/parkour/hybrid.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "prepare-model" in result.stdout and "actor-mpc" in result.stdout


def test_portable_model_build_help():
    script = Path(__file__).resolve().parents[1] / "scripts/parkour/hybrid.py"
    result = subprocess.run([sys.executable, str(script), "prepare-model", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert all(flag in result.stdout for flag in ("--scene", "--motion", "--output"))


def test_report_cli_recomputes_six_reports(tmp_path):
    hybrid = module("hybrid")
    for seed in hybrid.VALIDATION_SEEDS:
        row = {"seed": seed, "completion": hybrid.completion_report(np.zeros((270, 6)), 269)}
        (tmp_path / f"seed{seed}.json").write_text(json.dumps([row]))
    script = Path(__file__).resolve().parents[1] / "scripts/parkour/hybrid.py"
    result = subprocess.run([sys.executable, str(script), "summarize", "--reports", str(tmp_path),
                             "--index", "0"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["all_motion_checks_pass"] == 6
