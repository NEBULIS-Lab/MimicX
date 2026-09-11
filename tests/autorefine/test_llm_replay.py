from __future__ import annotations

import json
from pathlib import Path

import pytest

from mimicx.refinement.llm_replay import evaluate_replay, write_replay_report


def history_record() -> dict:
    return {
        "record_id": "tennis_history",
        "task_id": "tennis",
        "context": {
            "task_id": "tennis",
            "horizon": 800,
            "done_count": 3,
            "first_failure_step": 400,
            "failure_window": [360, 440],
            "dominant_bodies": ["left_wrist_yaw_link", "pelvis"],
            "dominant_channels": ["body_pos", "velocity"],
            "recommended_families": ["arm_wrist_precision", "root_core_guard"],
            "top_metrics": [],
        },
    }


def valid_response() -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "id": "wrist_root_window",
                    "sampling": {"mode": "window", "start": 370, "end": 430, "ratio": 0.55},
                    "reward_overrides": {
                        "motion_global_root_pos": {"weight": 5.0, "std": 0.14}
                    },
                    "body_pos_rewards": [
                        {
                            "name": "wrist_precision",
                            "body_names": ["left_wrist_yaw_link"],
                            "weight": 2.0,
                            "std": 0.1,
                        },
                        {
                            "name": "core_pos",
                            "body_names": ["pelvis", "torso_link"],
                            "weight": 2.5,
                            "std": 0.11,
                        },
                    ],
                    "body_z_rewards": [],
                    "body_lin_vel_rewards": [],
                    "body_ang_vel_rewards": [],
                }
            ]
        }
    )


def test_replay_scores_schema_failure_family_body_and_window_coverage() -> None:
    score = evaluate_replay(history_record(), valid_response())

    assert score["schema_valid"] is True
    assert score["candidate_count"] == 1
    assert score["family_recall"] == pytest.approx(1.0)
    assert score["body_recall"] == pytest.approx(1.0)
    assert score["window_overlap"] == pytest.approx(1.0)
    assert score["structural_diversity"] == pytest.approx(1.0)


def test_replay_records_validation_error_without_crashing_batch() -> None:
    invalid = json.dumps({"candidates": [{"id": "bad", "verification": {"repeats": 1}}]})

    score = evaluate_replay(history_record(), invalid)

    assert score["schema_valid"] is False
    assert "verification" in score["validation_error"] or "Missing" in score["validation_error"]
    assert score["candidate_count"] == 0


def test_write_replay_report_keeps_raw_responses_and_aggregate_metrics(tmp_path: Path) -> None:
    records = [history_record()]
    responses = {"tennis_history": valid_response()}

    summary = write_replay_report(records, responses, tmp_path)

    assert summary["records"] == 1
    assert summary["schema_valid_rate"] == pytest.approx(1.0)
    assert summary["cross_task_structural_diversity"] == pytest.approx(1.0)
    assert (tmp_path / "responses" / "tennis_history.txt").read_text() == valid_response()
    response_manifest = json.loads((tmp_path / "response_manifest.json").read_text())
    assert len(response_manifest["responses"][0]["sha256"]) == 64
    assert len((tmp_path / "replay_scores.jsonl").read_text().splitlines()) == 1
    assert "Family recall" in (tmp_path / "REPORT.md").read_text()
