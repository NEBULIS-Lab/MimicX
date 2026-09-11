"""Repeated strict-rollout aggregation and incumbent-protected selection."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from mimicx.refinement.manifest import GuardSpec, VerificationSpec


@dataclass(frozen=True)
class RolloutRecord:
    source: Path
    seed: int
    horizon: int
    done_count: int
    first_failure_step: int
    reward: float | None
    metrics: Mapping[str, float | None]


@dataclass(frozen=True)
class RepeatAggregate:
    candidate_id: str
    repeats: int
    zero_termination_repeats: int
    worst_first_failure_step: int
    total_terminations: int
    median_reward: float | None
    median_metrics: Mapping[str, float | None]
    records: tuple[RolloutRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for item in payload["records"]:
            item["source"] = str(item["source"])
        return payload


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    selected_id: str
    previous_id: str
    reasons: tuple[str, ...]
    ranked_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _step_csv_path(metrics_path: Path) -> Path | None:
    paths = [
        metrics_path.with_name(f"{metrics_path.stem}_steps.csv"),
        metrics_path.with_name(metrics_path.name.replace(".json", "_steps.csv")),
    ]
    for path in paths:
        if path.exists():
            return path
    return None


def _step_csv_summary(path: Path | None, metric_keys: Sequence[str]) -> tuple[int | None, dict[str, float]]:
    if path is None:
        return None, {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    first_failure: int | None = None
    for row in rows:
        if _as_bool(row.get("done_any")) or any(
            key.startswith("termination_") and _as_bool(value)
            for key, value in row.items()
        ):
            first_failure = int(row["step"])
            break
    last_values: dict[str, float] = {}
    for key in metric_keys:
        for row in reversed(rows):
            value = _as_float(row.get(key))
            if value is not None:
                last_values[key] = value
                break
    return first_failure, last_values


def load_rollout_metrics(
    path: Path,
    *,
    horizon: int,
    metric_keys: Sequence[str],
    seed: int,
) -> RolloutRecord:
    source = path.expanduser().resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Rollout metrics must be an object: {source}")
    done_count = int(data.get("done_any_count", data.get("done_count", 0)) or 0)
    csv_failure, csv_metrics = _step_csv_summary(_step_csv_path(source), metric_keys)
    first_failure = data.get("first_done_step", data.get("first_failure_step", csv_failure))
    if done_count == 0:
        first_failure = horizon + 1
    elif first_failure is None:
        first_failure = min(int(data.get("steps", horizon)), horizon)

    nested_metrics = data.get("metrics", {})
    if not isinstance(nested_metrics, Mapping):
        nested_metrics = {}
    metrics: dict[str, float | None] = {}
    for key in metric_keys:
        value = _as_float(data.get(key))
        if value is None:
            value = _as_float(nested_metrics.get(key))
        if value is None:
            value = csv_metrics.get(key)
        metrics[key] = value
    reward = _as_float(data.get("reward_mean_avg"))
    if reward is None:
        reward = _as_float(data.get("final_rew_mean"))
    return RolloutRecord(
        source=source,
        seed=seed,
        horizon=horizon,
        done_count=done_count,
        first_failure_step=int(first_failure),
        reward=reward,
        metrics=metrics,
    )


def _median_optional(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(median(present)) if present else None


def aggregate_repeats(candidate_id: str, records: Sequence[RolloutRecord]) -> RepeatAggregate:
    if not records:
        raise ValueError(f"No rollout records for {candidate_id}")
    metric_keys = sorted({key for record in records for key in record.metrics})
    median_metrics = {
        key: _median_optional([record.metrics.get(key) for record in records])
        for key in metric_keys
    }
    return RepeatAggregate(
        candidate_id=candidate_id,
        repeats=len(records),
        zero_termination_repeats=sum(record.done_count == 0 for record in records),
        worst_first_failure_step=min(record.first_failure_step for record in records),
        total_terminations=sum(record.done_count for record in records),
        median_reward=_median_optional([record.reward for record in records]),
        median_metrics=median_metrics,
        records=tuple(records),
    )


def _metric_rank(value: float | None, direction: str) -> float:
    if value is None:
        return float("-inf")
    return value if direction == "higher" else -value


def _rank_key(aggregate: RepeatAggregate, spec: VerificationSpec) -> tuple[float, ...]:
    key: list[float] = [
        float(aggregate.zero_termination_repeats),
        float(aggregate.worst_first_failure_step),
        float(-aggregate.total_terminations),
    ]
    for metric in spec.metric_keys:
        guard = spec.guards.get(metric, GuardSpec(direction="lower"))
        key.append(_metric_rank(aggregate.median_metrics.get(metric), guard.direction))
    key.append(aggregate.median_reward if aggregate.median_reward is not None else float("-inf"))
    return tuple(key)


def _guard_failure(
    incumbent: RepeatAggregate,
    candidate: RepeatAggregate,
    metric: str,
    guard: GuardSpec,
) -> str | None:
    base = incumbent.median_metrics.get(metric)
    proposed = candidate.median_metrics.get(metric)
    if base is None or proposed is None:
        return f"{metric} guard failed: missing incumbent or candidate metric"
    tolerance = abs(base) * guard.relative_tolerance + guard.absolute_tolerance
    if guard.direction == "lower" and proposed > base + tolerance:
        return f"{metric} guard failed: {proposed:.6g} > {base + tolerance:.6g}"
    if guard.direction == "higher" and proposed < base - tolerance:
        return f"{metric} guard failed: {proposed:.6g} < {base - tolerance:.6g}"
    return None


def decide_acceptance(
    incumbent: RepeatAggregate,
    candidates: Sequence[RepeatAggregate],
    spec: VerificationSpec,
) -> GateDecision:
    ranked = sorted(candidates, key=lambda item: _rank_key(item, spec), reverse=True)
    reasons: list[str] = []
    incumbent_key = _rank_key(incumbent, spec)
    for candidate in ranked:
        if candidate.repeats < spec.repeats:
            reasons.append(
                f"{candidate.candidate_id} requires {spec.repeats} repeats; found {candidate.repeats}"
            )
            continue
        if _rank_key(candidate, spec) <= incumbent_key:
            reasons.append(f"{candidate.candidate_id} does not dominate incumbent execution")
            continue
        failures = [
            failure
            for metric, guard in spec.guards.items()
            if (failure := _guard_failure(incumbent, candidate, metric, guard)) is not None
        ]
        if failures:
            reasons.extend(f"{candidate.candidate_id}: {failure}" for failure in failures)
            continue
        reasons.append(f"{candidate.candidate_id} dominates incumbent and passes all guards")
        return GateDecision(
            accepted=True,
            selected_id=candidate.candidate_id,
            previous_id=incumbent.candidate_id,
            reasons=tuple(reasons),
            ranked_ids=tuple(item.candidate_id for item in ranked),
        )
    if not reasons:
        reasons.append("no candidate metrics were provided")
    return GateDecision(
        accepted=False,
        selected_id=incumbent.candidate_id,
        previous_id=incumbent.candidate_id,
        reasons=tuple(reasons),
        ranked_ids=tuple(item.candidate_id for item in ranked),
    )
