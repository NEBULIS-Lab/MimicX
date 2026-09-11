from __future__ import annotations

import json
from pathlib import Path

from mimicx.refinement.gate import (
    RolloutRecord,
    aggregate_repeats,
    decide_acceptance,
    load_rollout_metrics,
)
from mimicx.refinement.manifest import GuardSpec, VerificationSpec


def spec(relative_tolerance: float = 0.1) -> VerificationSpec:
    return VerificationSpec(
        repeats=3,
        seeds=(11, 22, 33),
        metric_keys=("body_pos_error_max", "ee_z_error_max"),
        guards={
            "body_pos_error_max": GuardSpec(
                direction="lower", relative_tolerance=relative_tolerance
            )
        },
    )


def record(
    seed: int,
    *,
    horizon: int = 100,
    done_count: int = 0,
    first_failure_step: int | None = None,
    body: float = 0.20,
    ee_z: float = 0.10,
    reward: float = 1.0,
) -> RolloutRecord:
    return RolloutRecord(
        source=Path(f"seed_{seed}.json"),
        seed=seed,
        horizon=horizon,
        done_count=done_count,
        first_failure_step=first_failure_step or horizon + 1,
        reward=reward,
        metrics={"body_pos_error_max": body, "ee_z_error_max": ee_z},
    )


def test_load_rollout_metrics_uses_first_done_from_step_csv(tmp_path: Path) -> None:
    metrics = tmp_path / "rollout.json"
    metrics.write_text(
        json.dumps(
            {
                "steps": 100,
                "done_any_count": 1,
                "reward_mean_avg": 0.8,
                "body_pos_error_max": 0.24,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "rollout_steps.csv").write_text(
        "step,done_any,ee_z_error_max\n1,false,0.10\n63,true,0.14\n",
        encoding="utf-8",
    )

    loaded = load_rollout_metrics(
        metrics, horizon=100, metric_keys=("body_pos_error_max", "ee_z_error_max"), seed=7
    )

    assert loaded.first_failure_step == 63
    assert loaded.done_count == 1
    assert loaded.metrics["body_pos_error_max"] == 0.24
    assert loaded.metrics["ee_z_error_max"] == 0.14


def test_lucky_single_completion_cannot_replace_consistent_incumbent() -> None:
    incumbent = aggregate_repeats(
        "incumbent", [record(seed) for seed in (11, 22, 33)]
    )
    lucky = aggregate_repeats(
        "lucky",
        [
            record(11, reward=2.0),
            record(22, done_count=1, first_failure_step=55, reward=2.0),
            record(33, done_count=1, first_failure_step=60, reward=2.0),
        ],
    )

    decision = decide_acceptance(incumbent, [lucky], spec())

    assert decision.accepted is False
    assert decision.selected_id == "incumbent"
    assert "does not dominate incumbent execution" in decision.reasons[0]


def test_better_worst_case_failure_frontier_is_accepted() -> None:
    incumbent = aggregate_repeats(
        "incumbent",
        [
            record(11, done_count=1, first_failure_step=70),
            record(22, done_count=1, first_failure_step=75),
            record(33, done_count=1, first_failure_step=80),
        ],
    )
    robust = aggregate_repeats(
        "robust",
        [
            record(11, done_count=1, first_failure_step=86, body=0.21),
            record(22, done_count=1, first_failure_step=90, body=0.21),
            record(33, done_count=1, first_failure_step=88, body=0.21),
        ],
    )

    decision = decide_acceptance(incumbent, [robust], spec())

    assert decision.accepted is True
    assert decision.selected_id == "robust"
    assert decision.previous_id == "incumbent"


def test_reward_only_gain_does_not_override_worse_tracking() -> None:
    incumbent = aggregate_repeats(
        "incumbent", [record(seed, reward=1.0) for seed in (11, 22, 33)]
    )
    reward_only = aggregate_repeats(
        "reward-only",
        [record(seed, body=0.21, ee_z=0.12, reward=3.0) for seed in (11, 22, 33)],
    )

    decision = decide_acceptance(incumbent, [reward_only], spec())

    assert decision.accepted is False
    assert decision.selected_id == "incumbent"


def test_guard_regression_forces_rollback_despite_better_frontier() -> None:
    incumbent = aggregate_repeats(
        "incumbent",
        [
            record(11, done_count=1, first_failure_step=70),
            record(22, done_count=1, first_failure_step=72),
            record(33, done_count=1, first_failure_step=74),
        ],
    )
    regressed = aggregate_repeats(
        "regressed",
        [
            record(11, done_count=1, first_failure_step=90, body=0.31),
            record(22, done_count=1, first_failure_step=92, body=0.30),
            record(33, done_count=1, first_failure_step=94, body=0.32),
        ],
    )

    decision = decide_acceptance(incumbent, [regressed], spec())

    assert decision.accepted is False
    assert decision.selected_id == "incumbent"
    assert any("body_pos_error_max guard failed" in reason for reason in decision.reasons)


def test_candidate_with_too_few_repeats_is_not_eligible() -> None:
    incumbent = aggregate_repeats(
        "incumbent", [record(seed) for seed in (11, 22, 33)]
    )
    incomplete = aggregate_repeats("incomplete", [record(11, reward=5.0)])

    decision = decide_acceptance(incumbent, [incomplete], spec())

    assert decision.accepted is False
    assert any("requires 3 repeats" in reason for reason in decision.reasons)
