"""Offline evaluation of constrained language-guided AutoRefine proposals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from mimicx.refinement.proposals import (
    ALLOWED_BODIES,
    ProposalContext,
    generate_candidates,
)


def _context(payload: Mapping[str, Any]) -> ProposalContext:
    return ProposalContext(
        task_id=str(payload["task_id"]),
        horizon=int(payload["horizon"]),
        done_count=int(payload.get("done_count", 0)),
        first_failure_step=(
            int(payload["first_failure_step"])
            if payload.get("first_failure_step") is not None
            else None
        ),
        failure_window=tuple(int(value) for value in payload["failure_window"]),
        dominant_bodies=tuple(str(value) for value in payload.get("dominant_bodies", [])),
        dominant_channels=tuple(str(value) for value in payload.get("dominant_channels", [])),
        recommended_families=tuple(
            str(value) for value in payload.get("recommended_families", [])
        ),
        top_metrics=tuple(payload.get("top_metrics", [])),
    )


def _proposed_families(patch: Mapping[str, Any]) -> set[str]:
    terms = [
        term
        for family in (
            "body_pos_rewards",
            "body_z_rewards",
            "body_lin_vel_rewards",
            "body_ang_vel_rewards",
        )
        for term in patch[family]
    ]
    bodies = {body for term in terms for body in term["body_names"]}
    overrides = set(patch["reward_overrides"])
    families = {"general_body_tracking"} if terms or overrides else set()
    if any("wrist" in body or "elbow" in body or "shoulder" in body for body in bodies):
        families.add("arm_wrist_precision")
    if any("ankle" in body or "knee" in body or "hip" in body for body in bodies):
        families.add("lower_body_contact_z")
    if patch["body_z_rewards"]:
        families.add("z_clearance_contact")
    if bodies.intersection({"pelvis", "torso_link"}) or any("root" in name for name in overrides):
        families.add("root_core_guard")
    if patch["body_ang_vel_rewards"] or "action_rate_l2" in overrides:
        families.add("dynamics_smoothness")
    return families


def _patch_bodies(patch: Mapping[str, Any]) -> set[str]:
    return {
        body
        for family in (
            "body_pos_rewards",
            "body_z_rewards",
            "body_lin_vel_rewards",
            "body_ang_vel_rewards",
        )
        for term in patch[family]
        for body in term["body_names"]
    }


def _mean_optional(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def evaluate_replay(record: Mapping[str, Any], response: str) -> dict[str, Any]:
    context = _context(record["context"])
    score: dict[str, Any] = {
        "record_id": str(record["record_id"]),
        "task_id": str(record["task_id"]),
        "schema_valid": False,
        "validation_error": None,
        "candidate_count": 0,
        "family_recall": 0.0,
        "body_recall": None,
        "window_overlap": 0.0,
        "structural_diversity": 0.0,
        "proposal_signatures": [],
    }
    try:
        patches = generate_candidates(
            context,
            backend="llm",
            llm_generate=lambda _prompt: response,
            candidate_budget=3,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        score["validation_error"] = str(error)
        return score

    score["schema_valid"] = True
    score["candidate_count"] = len(patches)
    expected_families = set(context.recommended_families)
    proposed_families = set().union(*(_proposed_families(patch) for patch in patches))
    score["family_recall"] = (
        len(expected_families & proposed_families) / len(expected_families)
        if expected_families
        else 1.0
    )
    expected_bodies = set(context.dominant_bodies) & ALLOWED_BODIES
    proposed_bodies = set().union(*(_patch_bodies(patch) for patch in patches))
    score["body_recall"] = (
        len(expected_bodies & proposed_bodies) / len(expected_bodies)
        if expected_bodies
        else None
    )
    failure_start, failure_end = context.failure_window
    overlaps = []
    for patch in patches:
        sampling = patch["sampling"]
        if sampling["mode"] == "start":
            overlaps.append(1.0)
        else:
            overlaps.append(
                1.0
                if min(int(sampling["end"]), failure_end)
                > max(int(sampling["start"]), failure_start)
                else 0.0
            )
    score["window_overlap"] = _mean_optional(overlaps) or 0.0
    structures = {
        json.dumps({key: value for key, value in patch.items() if key != "id"}, sort_keys=True)
        for patch in patches
    }
    score["structural_diversity"] = len(structures) / len(patches) if patches else 0.0
    signatures = []
    for patch in patches:
        normalized = {key: value for key, value in patch.items() if key != "id"}
        sampling = dict(normalized["sampling"])
        sampling.pop("start", None)
        sampling.pop("end", None)
        normalized["sampling"] = sampling
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        signatures.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    score["proposal_signatures"] = signatures
    return score


def write_replay_report(
    records: Sequence[Mapping[str, Any]],
    responses: Mapping[str, str],
    output_dir: Path,
    *,
    model_path: str | None = None,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    response_dir = output / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    response_records = []
    for record in records:
        record_id = str(record["record_id"])
        if record_id not in responses:
            raise KeyError(f"Missing replay response for {record_id}")
        response = responses[record_id]
        response_path = response_dir / f"{record_id}.txt"
        response_path.write_text(response, encoding="utf-8")
        response_records.append(
            {
                "record_id": record_id,
                "path": str(response_path),
                "sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            }
        )
        scores.append(evaluate_replay(record, response))
    with (output / "replay_scores.jsonl").open("w", encoding="utf-8") as handle:
        for score in scores:
            handle.write(json.dumps(score, sort_keys=True) + "\n")
    signatures = [signature for score in scores for signature in score["proposal_signatures"]]
    model_config_sha256 = None
    if model_path:
        config_path = Path(model_path) / "config.json"
        if config_path.is_file():
            model_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    response_manifest = {
        "model_path": model_path,
        "model_config_sha256": model_config_sha256,
        "responses": response_records,
    }
    (output / "response_manifest.json").write_text(
        json.dumps(response_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "records": len(scores),
        "model_path": model_path,
        "model_config_sha256": model_config_sha256,
        "schema_valid_rate": fmean(score["schema_valid"] for score in scores),
        "family_recall_mean": _mean_optional([score["family_recall"] for score in scores]),
        "body_recall_mean": _mean_optional([score["body_recall"] for score in scores]),
        "window_overlap_mean": _mean_optional([score["window_overlap"] for score in scores]),
        "structural_diversity_mean": _mean_optional(
            [score["structural_diversity"] for score in scores]
        ),
        "cross_task_structural_diversity": (
            len(set(signatures)) / len(signatures) if signatures else 0.0
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Language-Guided AutoRefine Offline Replay",
        "",
        "This replay evaluates proposal validity and failure-conditioned coverage. "
        "It does not measure policy execution quality; strict rollout gates remain authoritative.",
        "",
        "| Task | Schema | Family recall | Body recall | Window overlap | Diversity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for score in scores:
        body = "-" if score["body_recall"] is None else f"{score['body_recall']:.3f}"
        lines.append(
            f"| {score['task_id']} | {int(score['schema_valid'])} | "
            f"{score['family_recall']:.3f} | {body} | {score['window_overlap']:.3f} | "
            f"{score['structural_diversity']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- Schema-valid rate: **{summary['schema_valid_rate']:.3f}**",
            f"- Family recall: **{summary['family_recall_mean']:.3f}**",
            f"- Window overlap: **{summary['window_overlap_mean']:.3f}**",
            "- Cross-task structural diversity: "
            f"**{summary['cross_task_structural_diversity']:.3f}**",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
