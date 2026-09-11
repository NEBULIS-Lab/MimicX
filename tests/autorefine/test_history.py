from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mimicx.refinement.history import build_history_dataset, write_history_dataset


def _fixture_config(tmp_path: Path) -> Path:
    failure = tmp_path / "failure.json"
    failure.write_text(
        json.dumps(
            {
                "summary": {"steps": 100, "done_any_count": 2, "first_done_step": 44},
                "failure_windows": [
                    {
                        "window": [30, 60],
                        "dominant_bodies": ["left_ankle_roll_link"],
                        "dominant_channels": ["body_z"],
                        "recommended_families": ["lower_body_contact_z"],
                        "top_metrics": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (patch_dir / "candidate.json").write_text(
        json.dumps({"id": "candidate", "sampling": {"mode": "start"}}),
        encoding="utf-8",
    )
    outcome = tmp_path / "outcome.json"
    outcome.write_text(
        json.dumps(
            {
                "best": {"name": "candidate", "reward_mean_avg": 0.4, "done_any_count": 0},
                "baseline": {"name": "base", "reward_mean_avg": 0.3, "done_any_count": 2},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "history.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema": "mimicx.autorefine-history-sources.v1",
                "sources": [
                    {
                        "id": "football_round1",
                        "task_id": "football1",
                        "evidence_scope": "candidate_level",
                        "failure_report": "failure.json",
                        "candidate_patch_glob": "patches/*.json",
                        "outcome_file": "outcome.json",
                        "selected_selector": "best",
                        "baseline_selector": "baseline",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config


def test_build_history_dataset_keeps_context_candidates_outcome_and_hashes(tmp_path: Path) -> None:
    records = build_history_dataset(_fixture_config(tmp_path))

    assert len(records) == 1
    record = records[0]
    assert record["record_id"] == "football_round1"
    assert record["task_id"] == "football1"
    assert record["context"]["failure_window"] == [30, 60]
    assert record["context"]["recommended_families"] == ["lower_body_contact_z"]
    assert record["candidates"][0]["patch"]["id"] == "candidate"
    assert record["outcome"]["selected"]["reward_mean_avg"] == 0.4
    assert record["outcome"]["baseline"]["reward_mean_avg"] == 0.3
    assert record["outcome"]["delta"]["reward_mean_avg"] == pytest.approx(0.1)
    assert record["outcome"]["delta"]["done_any_count"] == -2
    assert len(record["provenance"]["failure_report"]["sha256"]) == 64
    assert len(record["provenance"]["candidate_patches"][0]["sha256"]) == 64


def test_write_history_dataset_emits_jsonl_manifest_and_summary(tmp_path: Path) -> None:
    output = tmp_path / "output"

    status = write_history_dataset(_fixture_config(tmp_path), output)

    assert status["records"] == 1
    assert status["tasks"] == ["football1"]
    assert len((output / "history.jsonl").read_text().splitlines()) == 1
    assert json.loads((output / "manifest.json").read_text())["records"] == 1
    assert "candidate_level" in (output / "SUMMARY.md").read_text()


def test_history_builder_rejects_missing_provenance_source(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    payload = yaml.safe_load(config.read_text())
    payload["sources"][0]["failure_report"] = "missing.json"
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing.json"):
        build_history_dataset(config)
