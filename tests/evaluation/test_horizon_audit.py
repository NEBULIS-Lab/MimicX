"""CPU-only real-clock and ordered long-horizon evidence contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mimicx.evaluation.horizon_audit import Stage, audit_motion_horizon


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/experiments/audit_motion_horizon.py"


def reference(tmp_path, frames=301, **changes):
    path = tmp_path / "reference.npz"
    data = dict(fps=np.array([50.]), joint_pos=np.zeros((frames, 2)), body_pos_w=np.zeros((frames, 3, 3)))
    data.update(changes)
    np.savez(path, **data)
    return path


def rollout(tmp_path, frames, **changes):
    path = tmp_path / "rollout.npz"
    ids = np.asarray(frames)
    data = dict(timestamps=100. + np.arange(len(ids)) / 50., reference_frame=ids,
                reset=np.zeros(len(ids), dtype=bool), failure=np.zeros(len(ids), dtype=bool))
    data.update(changes)
    np.savez(path, **data)
    return path


def audit(ref, trace=None, **kwargs):
    return audit_motion_horizon(ref, trace, timestamp_kind="wall", **kwargs)


def test_true_long_multistage_clock_not_just_full_video(tmp_path):
    frames = 6001
    ref = reference(tmp_path, frames)
    trace = rollout(tmp_path, np.arange(frames))
    stages = [Stage("approach", 0, 1500), Stage("traverse", 1500, 4500), Stage("finish", 4500, 6000)]
    result = audit(ref, trace, stages=stages, minimum_duration_s=120)
    assert result["source"]["duration_seconds"] == 120
    assert result["horizon_verified"] is True
    assert result["multi_stage_horizon_verified"] is True
    assert result["clock_basis"] == "wall"
    assert result["real_time_playback_verified"] is True
    assert all(stage["covered_in_scoring_prefix"] for stage in result["stages"])
    assert result["rollout"]["scoring_coverage"]["prefix"]["elapsed_seconds"] == pytest.approx(120)
    assert result["success"] is None
    assert str(tmp_path) not in json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("rate_key,position_key,shape", [
    ("fps", "joint_pos", (101, 2)), ("frame_rate", "dof_pos", (101, 3)),
    ("frame_rate", "body_pos_w", (101, 4, 3)), ("fps", "body_pos", (101, 4, 3)),
    ("fps", "root_pos", (101, 3)),
])
def test_reference_formats_and_no_rollout_unknown(tmp_path, rate_key, position_key, shape):
    ref = tmp_path / "motion.npz"
    np.savez(ref, **{rate_key: 50., position_key: np.zeros(shape)})
    result = audit_motion_horizon(ref)
    assert result["source"]["frames"] == 101
    assert result["source"]["duration_seconds"] == 2
    assert result["horizon_verified"] is None
    assert result["clock_basis"] is None
    assert result["real_time_playback_verified"] is None
    assert result["rollout"] is None
    assert result["success"] is None


def test_source_scoring_and_recording_durations_are_distinct(tmp_path):
    ref = reference(tmp_path, 6001)
    trace = rollout(tmp_path, np.arange(1001))
    result = audit(ref, trace, scoring_window=(0, 1000), minimum_duration_s=60)
    assert result["source"]["duration_seconds"] == 120
    assert result["scoring_window"]["duration_seconds"] == 20
    assert result["rollout"]["wall_elapsed_seconds"] == pytest.approx(20)
    assert result["rollout"]["scoring_coverage"]["continuous_window_complete"] is True
    assert result["rollout"]["source_coverage"]["continuous_window_complete"] is False
    assert result["horizon_verified"] is False


def test_offset_scoring_window_and_shared_stage_endpoints(tmp_path):
    ref = reference(tmp_path)
    trace = rollout(tmp_path, np.arange(301))
    result = audit(ref, trace, scoring_window=(50, 250), stages=[Stage("a", 50, 150), Stage("b", 150, 250)])
    assert result["scoring_window"]["frames"] == 201
    assert result["horizon_verified"] is True
    assert result["rollout"]["scoring_coverage"]["prefix"]["start_sample"] == 50
    assert [s["duration_seconds"] for s in result["stages"]] == [2., 2.]


def test_repeated_short_clip_does_not_become_long_horizon(tmp_path):
    ref = reference(tmp_path, 101)
    trace = rollout(tmp_path, np.tile(np.arange(101), 60))
    result = audit(ref, trace, minimum_duration_s=60)
    assert result["rollout"]["elapsed_seconds"] > 120
    assert result["source"]["duration_seconds"] == 2
    assert result["rollout"]["wrap_steps"] == 59
    assert result["rollout"]["revisited_frame_samples"] == 59 * 101
    assert result["rollout"]["scoring_coverage"]["longest_segment"]["reference_seconds"] == 2
    assert result["horizon_verified"] is False


@pytest.mark.parametrize("event", ["reset", "failure", "valid_mask", "native_failure_frame"])
def test_full_recording_after_event_is_censored(tmp_path, event):
    ref = reference(tmp_path)
    if event == "native_failure_frame":
        extra = {event: 150}
    else:
        flags = np.arange(301) >= 150 if event == "valid_mask" else np.arange(301) == 150
        extra = {event: ~flags if event == "valid_mask" else flags}
    trace = rollout(tmp_path, np.arange(301), **extra)
    result = audit(ref, trace, stages=[Stage("a", 0, 100), Stage("b", 100, 300)])
    coverage = result["rollout"]["scoring_coverage"]
    assert coverage["observed_unique_frames"] == 301
    assert coverage["valid_unique_frames"] == 150
    assert coverage["prefix_unique_frames"] == 150
    assert result["rollout"]["first_censored_sample"] == 150
    assert [s["covered_in_scoring_prefix"] for s in result["stages"]] == [True, False]
    assert result["horizon_verified"] is False


def test_stages_seen_across_backtracking_are_not_a_continuous_task(tmp_path):
    ref = reference(tmp_path)
    ids = np.r_[np.arange(151), np.arange(100, 301)]
    result = audit(ref, rollout(tmp_path, ids), stages=[Stage("a", 0, 150), Stage("b", 150, 300)])
    coverage = result["rollout"]["scoring_coverage"]
    assert coverage["observed_unique_frames"] == 301
    assert coverage["continuous_segment_count"] == 2
    assert coverage["prefix_unique_frames"] == 151
    assert result["stages"][1]["coverage"]["any_segment_complete"] is True
    assert result["stages"][1]["coverage"]["continuous_window_complete"] is False
    assert result["stages"][1]["covered_in_scoring_prefix"] is False
    assert result["multi_stage_horizon_verified"] is False


def test_held_frames_add_time_but_not_coverage(tmp_path):
    ref = reference(tmp_path)
    ids = np.r_[np.arange(101), np.full(100, 100), np.arange(101, 301)]
    result = audit(ref, rollout(tmp_path, ids))
    coverage = result["rollout"]["scoring_coverage"]
    assert result["rollout"]["repeated_frame_steps"] == 100
    assert coverage["prefix_unique_frames"] == 301
    assert coverage["prefix"]["max_clock_error_seconds"] == pytest.approx(2)
    assert result["horizon_verified"] is False


def test_oversampling_and_trailing_video_do_not_inflate_horizon(tmp_path):
    ref = reference(tmp_path)
    ids = np.repeat(np.arange(301), 2)
    trace = rollout(tmp_path, ids, timestamps=10 + np.arange(len(ids)) / 100.)
    result = audit(ref, trace)
    assert result["horizon_verified"] is True
    assert result["rollout"]["scoring_coverage"]["prefix"]["reference_seconds"] == 6
    ids = np.r_[np.arange(301), np.full(1000, 300)]
    result = audit(ref, rollout(tmp_path, ids))
    assert result["horizon_verified"] is True
    assert result["rollout"]["wall_elapsed_seconds"] == pytest.approx(26)
    assert result["rollout"]["scoring_coverage"]["prefix"]["elapsed_seconds"] == pytest.approx(6)


@pytest.mark.parametrize("clock_basis", ["wall", "simulation"])
@pytest.mark.parametrize("kind", ["speedup", "slowdown", "gap", "skip", "late_start", "drift"])
def test_discontinuous_or_wrong_clock_execution_is_not_verified(tmp_path, kind, clock_basis):
    ref = reference(tmp_path)
    ids = np.arange(301)
    times = np.arange(301) / 50.
    if kind == "speedup":
        times *= .5
    elif kind == "slowdown":
        times *= 1.04
    elif kind == "gap":
        times[150:] += .2
    elif kind == "skip":
        ids, times = np.delete(ids, 150), np.delete(times, 150)
    elif kind == "late_start":
        ids, times = ids[1:], times[1:]
    else:
        times[1:-1] += .04 * np.sin(np.linspace(0, np.pi, 299))
    result = audit_motion_horizon(ref, rollout(tmp_path, ids, timestamps=times), timestamp_kind=clock_basis)
    assert result["horizon_verified"] is False
    assert result["clock_basis"] == clock_basis
    assert result["real_time_playback_verified"] is (False if clock_basis == "wall" else None)
    if kind == "gap":
        assert result["rollout"]["sample_gap_count"] == 1
    if kind == "skip":
        assert result["rollout"]["skipped_frame_steps"] == 1


@pytest.mark.parametrize("clock_basis", ["wall", "simulation"])
@pytest.mark.parametrize("missing", ["timestamps", "reference_frame", "reset", "failure"])
def test_partial_metadata_never_silently_guesses(tmp_path, missing, clock_basis):
    ref = reference(tmp_path)
    data = dict(timestamps=np.arange(301) / 50, reference_frame=np.arange(301), reset=np.zeros(301), failure=np.zeros(301))
    del data[missing]
    path = tmp_path / "partial.npz"
    np.savez(path, **data)
    result = audit_motion_horizon(ref, path, timestamp_kind=clock_basis)
    assert result["horizon_verified"] is None
    assert result["missing_evidence"]
    assert result["success"] is None


def test_simulation_clock_is_not_a_wall_clock_and_key_mapping_is_explicit(tmp_path):
    ref = reference(tmp_path)
    trace = rollout(tmp_path, np.arange(301))
    result = audit_motion_horizon(ref, trace, timestamp_kind="simulation")
    assert result["horizon_verified"] is True
    assert result["clock_basis"] == "simulation"
    assert result["real_time_playback_verified"] is None
    assert result["missing_evidence"] == []
    assert result["rollout"]["wall_elapsed_seconds"] is None
    path = tmp_path / "mapped.npz"
    np.savez(path, monotonic_time=20 + np.arange(301) / 50, reference_frames=np.arange(301),
             restarted=np.zeros(301), fall_detected=np.zeros(301), timestamp_kind="wall")
    result = audit_motion_horizon(ref, path, timestamp_key="monotonic_time", reset_key="restarted", failure_key="fall_detected")
    assert result["horizon_verified"] is True
    assert result["real_time_playback_verified"] is True
    assert result["rollout"]["keys"]["reference_frame"] == "reference_frames"
    with pytest.raises(ValueError, match="contradicts"):
        audit_motion_horizon(ref, path, timestamp_kind="simulation")


def test_matching_simulation_horizon_does_not_require_wall_speed(tmp_path):
    ref = reference(tmp_path, 3001)
    trace = rollout(tmp_path, np.arange(3001), time=np.arange(3001) / 50.,
                    monotonic_time=10. + np.arange(3001) / 500.)
    stages = [Stage("approach", 0, 1000), Stage("traverse", 1000, 2000), Stage("finish", 2000, 3000)]
    simulation = audit_motion_horizon(ref, trace, timestamp_key="time", timestamp_kind="simulation",
                                      stages=stages, minimum_duration_s=60)
    assert simulation["clock_basis"] == "simulation"
    assert simulation["horizon_verified"] is True
    assert simulation["multi_stage_horizon_verified"] is True
    assert simulation["real_time_playback_verified"] is None
    assert simulation["missing_evidence"] == []
    assert simulation["rollout"]["elapsed_seconds"] == pytest.approx(60)
    assert simulation["success"] is None
    # The same samples span only six wall seconds, not sixty-second playback.
    wall = audit_motion_horizon(ref, trace, timestamp_key="monotonic_time", timestamp_kind="wall",
                                stages=stages, minimum_duration_s=60)
    assert wall["horizon_verified"] is False
    assert wall["real_time_playback_verified"] is False
    assert wall["rollout"]["wall_elapsed_seconds"] == pytest.approx(6)


@pytest.mark.parametrize("changes,match", [
    ({"fps": 0}, "positive"), ({"fps": float("nan")}, "finite"),
    ({"frame_rate": 30}, "conflicting"), ({"joint_pos": np.zeros((300, 2))}, "counts"),
    ({"body_pos_w": np.zeros((301, 3))}, "shape"),
    ({"joint_pos": np.full((301, 2), np.nan)}, "finite"),
    ({"dof_pos": np.ones((301, 2))}, "aliases"),
    ({"timestamps": np.arange(301) / 30.}, "disagree"),
])
def test_invalid_reference_rejected(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        audit(reference(tmp_path, **changes))


@pytest.mark.parametrize("changes,match", [
    ({"timestamps": np.zeros(301)}, "increasing"),
    ({"timestamps": np.r_[np.arange(150), np.arange(151)]}, "increasing"),
    ({"timestamps": np.full(301, np.inf)}, "finite"),
    ({"timestamps": np.arange(300)}, "matching"),
    ({"reference_frame": np.arange(301) + .5}, "integer"),
    ({"reference_frame": np.arange(301) + 1}, "outside"),
    ({"reference_frames": np.zeros(301)}, "conflicting"),
    ({"reset": np.full(301, "False")}, "boolean"),
    ({"failure": np.full(301, 2)}, "boolean"),
    ({"valid_mask": np.arange(301) % 2}, "prefix-only"),
    ({"native_failure_frame": -1}, "sample index"),
])
def test_invalid_rollout_rejected(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        audit(reference(tmp_path), rollout(tmp_path, np.arange(301), **changes))


def test_ambiguous_timestamps_rejected(tmp_path):
    with pytest.raises(ValueError, match="declare timestamp_kind"):
        audit_motion_horizon(reference(tmp_path), rollout(tmp_path, np.arange(301)))


def test_all_censored_and_failure_after_window_end(tmp_path):
    ref = reference(tmp_path)
    trace = rollout(tmp_path, np.arange(301), failure=np.arange(301) == 0)
    result = audit(ref, trace)
    assert result["rollout"]["scoring_coverage"]["prefix"] is None
    assert result["horizon_verified"] is False
    trace = rollout(tmp_path, np.arange(301), reset=np.arange(301) == 201)
    result = audit(ref, trace, scoring_window=(0, 200))
    assert result["horizon_verified"] is True
    assert result["rollout"]["source_coverage"]["continuous_window_complete"] is False


def test_valid_mask_without_termination_telemetry_is_not_failure_free_proof(tmp_path):
    ref = reference(tmp_path)
    path = tmp_path / "native.npz"
    np.savez(path, timestamps=np.arange(301) / 50, reference_frame=np.arange(301), valid_mask=np.ones(301))
    result = audit(ref, path)
    assert result["horizon_verified"] is None
    assert result["rollout"]["scoring_coverage"]["continuous_window_complete"] is True
    assert set(result["missing_evidence"]) == {"reset", "failure"}


def test_custom_clock_tolerances_and_non_50hz_reference(tmp_path):
    ref = reference(tmp_path, 101, fps=10.)
    trace = rollout(tmp_path, np.arange(101), timestamps=np.arange(101) / 10.)
    assert audit(ref, trace)["horizon_verified"] is False
    result = audit(ref, trace, state_gap_s=.11)
    assert result["horizon_verified"] is True
    assert result["source"]["duration_seconds"] == 10


@pytest.mark.parametrize("kwargs", [
    {"scoring_window": (0, 301)}, {"scoring_window": (1, 1)},
    {"scoring_window": (0, 2.5)}, {"scoring_window": (False, 3)},
    {"clock_tolerance_s": 0}, {"state_gap_s": float("nan")},
    {"duration_ratio_bounds": (1.1, 1.2)}, {"minimum_duration_s": -1},
])
def test_invalid_window_or_protocol_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        audit(reference(tmp_path), **kwargs)


@pytest.mark.parametrize("payload,match", [
    ({"joint_pos": np.zeros((3, 2))}, "fps or frame_rate"),
    ({"fps": 50, "qpos": np.zeros((3, 2))}, "position trajectory"),
    ({"fps": 50, "joint_pos": np.zeros((1, 2))}, "at least two"),
])
def test_missing_reference_contract_is_not_inferred(tmp_path, payload, match):
    ref = tmp_path / "unsupported.npz"
    np.savez(ref, **payload)
    with pytest.raises(ValueError, match=match):
        audit(ref)


def test_agreeing_aliases_and_reference_timestamps(tmp_path):
    ref = reference(tmp_path, frame_rate=np.array([50.]), dof_pos=np.zeros((301, 2)),
                    timestamps=3 + np.arange(301) / 50)
    assert audit(ref)["source"]["duration_seconds"] == 6


def test_numpy_stage_endpoints_are_normalized_to_json_safe_integers(tmp_path):
    result = audit(reference(tmp_path), stages=[Stage("a", np.int64(0), np.int64(300))])
    assert type(result["stages"][0]["frames"]) is int
    json.dumps(result, allow_nan=False)


def test_standalone_array_is_not_an_npz_reference(tmp_path):
    path = tmp_path / "array.npy"
    np.save(path, np.zeros((301, 2)))
    with pytest.raises(ValueError, match="NPZ archive"):
        audit(path)


@pytest.mark.parametrize("stages", [
    [Stage("a", 0, 100), Stage("b", 101, 300)],
    [Stage("a", 0, 100), Stage("b", 99, 300)],
    [Stage("a", 0, 100), Stage("a", 100, 300)],
    [Stage("a", 1, 300)], [Stage("a", 0, 299)], [Stage("a", 0, 0)],
])
def test_stage_gaps_overlaps_duplicates_or_incomplete_partition_rejected(tmp_path, stages):
    with pytest.raises(ValueError, match="partition"):
        audit(reference(tmp_path), stages=stages)


def run_cli(*args, cwd=ROOT):
    return subprocess.run([sys.executable, "-B", str(CLI), *map(str, args)], cwd=cwd,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=""),
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("clock_basis", ["wall", "simulation"])
def test_cli_portable_json_and_verification_exit_codes(tmp_path, clock_basis):
    ref = reference(tmp_path, 6001)
    trace = rollout(tmp_path, np.arange(6001))
    stages = tmp_path / "stages.json"
    stages.write_text(json.dumps([dict(name="a", start_frame=0, end_frame_inclusive=3000),
                                 dict(name="b", start_frame=3000, end_frame_inclusive=6000)]))
    proc = run_cli(ref.name, "--rollout", trace.name, "--timestamp-kind", clock_basis, "--minimum-duration-s", 60,
                   "--stages-json", stages.name, "--require-verified", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["multi_stage_horizon_verified"] is True
    assert report["clock_basis"] == clock_basis
    assert report["real_time_playback_verified"] is (True if clock_basis == "wall" else None)
    assert str(tmp_path) not in proc.stdout
    proc = run_cli(ref, "--require-verified")
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["horizon_verified"] is None
    proc = run_cli(ref, "--rollout", trace)
    assert proc.returncode == 2
    assert "timestamp_kind" in proc.stderr


def test_cli_help_output_and_no_overwrite(tmp_path):
    proc = run_cli("--help", cwd=tmp_path)
    assert proc.returncode == 0
    assert "--scoring-window" in proc.stdout
    output = tmp_path / "audit.json"
    ref = reference(tmp_path)
    proc = run_cli(ref, "--output", output)
    assert proc.returncode == 0, proc.stderr
    original = output.read_bytes()
    proc = run_cli(ref, "--output", output)
    assert proc.returncode == 2
    assert output.read_bytes() == original


def test_import_does_not_load_gpu_or_simulator_packages():
    code = ("import sys; import mimicx.evaluation.horizon_audit; "
            "assert not any(k.split('.')[0] in {'torch', 'mujoco', 'isaaclab', 'jax'} for k in sys.modules)")
    proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
