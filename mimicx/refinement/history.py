"""Build provenance-preserving AutoRefine history records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from mimicx.refinement.manifest import hash_file
from mimicx.refinement.proposals import ProposalContext


HISTORY_SOURCE_SCHEMA = "mimicx.autorefine-history-sources.v1"
HISTORY_RECORD_SCHEMA = "mimicx.autorefine-history.v1"


def _load_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _resolve(path_value: Any, root: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"History provenance source does not exist: {path}")
    return path


def _select(payload: Mapping[str, Any], selector: str | None) -> Any:
    if not selector:
        return None
    value: Any = payload
    for part in selector.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Selector {selector!r} is absent from outcome source")
        value = value[part]
    return value


def _numeric_delta(selected: Any, baseline: Any) -> dict[str, float]:
    if not isinstance(selected, Mapping) or not isinstance(baseline, Mapping):
        return {}
    delta: dict[str, float] = {}
    for key in sorted(set(selected) & set(baseline)):
        left = selected[key]
        right = baseline[key]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta[key] = float(left) - float(right)
    return delta


def _provenance(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hash_file(path)}


def _patch_paths(entry: Mapping[str, Any], root: Path) -> list[Path]:
    paths: list[Path] = []
    if entry.get("candidate_patch"):
        paths.append(_resolve(entry["candidate_patch"], root))
    if entry.get("candidate_patch_glob"):
        pattern = str(entry["candidate_patch_glob"])
        absolute_pattern = Path(pattern).expanduser()
        if absolute_pattern.is_absolute():
            search_root = absolute_pattern.parent
            matches = sorted(search_root.glob(absolute_pattern.name))
        else:
            matches = sorted(root.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Candidate patch glob matched no files: {root / pattern}")
        paths.extend(path.resolve() for path in matches if path.is_file())
    if not paths:
        raise ValueError(f"History source {entry.get('id')} has no candidate patches")
    return list(dict.fromkeys(paths))


def build_history_dataset(config_path: Path) -> list[dict[str, Any]]:
    source = config_path.expanduser().resolve()
    config = _load_yaml(source)
    if config.get("schema") != HISTORY_SOURCE_SCHEMA:
        raise ValueError(f"Unsupported history source schema: {config.get('schema')!r}")
    root = source.parent
    records: list[dict[str, Any]] = []
    for entry in config.get("sources", []):
        if not isinstance(entry, Mapping):
            raise ValueError("Each history source must be an object")
        task_id = str(entry["task_id"])
        failure_path = _resolve(entry["failure_report"], root)
        failure_report = json.loads(failure_path.read_text(encoding="utf-8"))
        context = ProposalContext.from_failure_report(failure_report, task_id=task_id)
        patch_paths = _patch_paths(entry, root)
        candidates = [
            {
                "patch": json.loads(path.read_text(encoding="utf-8")),
                "source": _provenance(path),
            }
            for path in patch_paths
        ]
        outcome_path = _resolve(entry["outcome_file"], root)
        outcome_payload = json.loads(outcome_path.read_text(encoding="utf-8"))
        selected = _select(outcome_payload, str(entry.get("selected_selector", "best")))
        baseline = _select(outcome_payload, entry.get("baseline_selector"))
        records.append(
            {
                "schema": HISTORY_RECORD_SCHEMA,
                "record_id": str(entry["id"]),
                "task_id": task_id,
                "evidence_scope": str(entry.get("evidence_scope", "candidate_level")),
                "context": context.prompt_payload(),
                "candidates": candidates,
                "outcome": {
                    "selected": selected,
                    "baseline": baseline,
                    "delta": _numeric_delta(selected, baseline),
                },
                "provenance": {
                    "failure_report": _provenance(failure_path),
                    "candidate_patches": [_provenance(path) for path in patch_paths],
                    "outcome_file": _provenance(outcome_path),
                },
            }
        )
    if not records:
        raise ValueError(f"No history sources configured in {source}")
    return records


def write_history_dataset(config_path: Path, output_dir: Path) -> dict[str, Any]:
    records = build_history_dataset(config_path)
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "history.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    tasks = sorted({record["task_id"] for record in records})
    scopes: dict[str, int] = {}
    for record in records:
        scope = record["evidence_scope"]
        scopes[scope] = scopes.get(scope, 0) + 1
    status = {
        "schema": HISTORY_RECORD_SCHEMA,
        "source_config": _provenance(config_path.expanduser().resolve()),
        "records": len(records),
        "tasks": tasks,
        "evidence_scopes": scopes,
    }
    (output / "manifest.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# AutoRefine History Dataset",
        "",
        f"- Records: **{len(records)}**",
        f"- Tasks: **{', '.join(tasks)}**",
        "",
        "| Record | Task | Evidence scope | Candidates | Selected |",
        "|---|---|---|---:|---|",
    ]
    for record in records:
        selected = record["outcome"]["selected"]
        selected_name = "-"
        if isinstance(selected, Mapping):
            selected_name = str(
                selected.get("name", selected.get("label", selected.get("id", "recorded")))
            )
        lines.append(
            f"| {record['record_id']} | {record['task_id']} | {record['evidence_scope']} | "
            f"{len(record['candidates'])} | {selected_name} |"
        )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status
