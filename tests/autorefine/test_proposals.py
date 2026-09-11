from __future__ import annotations

import json

import pytest

from mimicx.refinement.proposals import (
    ProposalContext,
    build_llm_prompt,
    generate_candidates,
    validate_patch,
)


def failure_report() -> dict:
    return {
        "summary": {"steps": 800, "done_any_count": 4, "first_done_step": 136},
        "failure_windows": [
            {
                "window": [96, 176],
                "dominant_bodies": ["left_wrist_yaw_link", "pelvis"],
                "dominant_channels": ["body_pos", "velocity"],
                "recommended_families": ["arm_wrist_precision", "root_core_guard"],
                "top_metrics": [{"metric": "body_pos_error_max", "avg": 0.33}],
            }
        ],
    }


def test_heuristic_backend_preserves_three_established_candidates() -> None:
    context = ProposalContext.from_failure_report(failure_report(), task_id="tennis")

    patches = generate_candidates(context, backend="heuristic", candidate_budget=3)

    assert [patch["id"] for patch in patches] == [
        "window_local_global",
        "window_strong_replay",
        "full_start_consolidate",
    ]
    assert patches[0]["sampling"] == {
        "mode": "window",
        "start": 96,
        "end": 176,
        "ratio": 0.45,
    }
    assert patches[1]["body_ang_vel_rewards"]
    assert patches[0]["body_pos_rewards"][1]["name"] == "arm_pos"


def test_llm_backend_accepts_only_schema_valid_patch_json() -> None:
    context = ProposalContext.from_failure_report(failure_report(), task_id="tennis")
    response = json.dumps(
        {
            "candidates": [
                {
                    "id": "llm_wrist_window",
                    "sampling": {"mode": "window", "start": 110, "end": 170, "ratio": 0.55},
                    "reward_overrides": {
                        "motion_global_root_pos": {"weight": 5.0, "std": 0.15}
                    },
                    "body_pos_rewards": [
                        {
                            "name": "wrist_precision",
                            "body_names": ["left_wrist_yaw_link"],
                            "weight": 2.0,
                            "std": 0.10,
                        }
                    ],
                    "body_z_rewards": [],
                    "body_lin_vel_rewards": [],
                    "body_ang_vel_rewards": [],
                }
            ]
        }
    )

    patches = generate_candidates(
        context,
        backend="llm",
        llm_generate=lambda _prompt: response,
        candidate_budget=2,
    )

    assert len(patches) == 1
    assert patches[0]["id"] == "llm_wrist_window"


def test_llm_cannot_change_verification_or_acceptance_policy() -> None:
    context = ProposalContext.from_failure_report(failure_report(), task_id="tennis")
    invalid = {
        "id": "unsafe",
        "sampling": {"mode": "start"},
        "reward_overrides": {},
        "body_pos_rewards": [],
        "body_z_rewards": [],
        "body_lin_vel_rewards": [],
        "body_ang_vel_rewards": [],
        "verification": {"repeats": 1},
    }

    with pytest.raises(ValueError, match="verification"):
        validate_patch(invalid, context=context)


def test_hybrid_reserves_budget_for_failure_conditioned_llm_candidate() -> None:
    context = ProposalContext.from_failure_report(failure_report(), task_id="tennis")
    response = json.dumps(
        {
            "candidates": [
                {
                    "id": "llm_specific",
                    "sampling": {"mode": "window", "start": 100, "end": 160, "ratio": 0.5},
                    "reward_overrides": {},
                    "body_pos_rewards": [],
                    "body_z_rewards": [],
                    "body_lin_vel_rewards": [],
                    "body_ang_vel_rewards": [],
                }
            ]
        }
    )

    patches = generate_candidates(
        context,
        backend="hybrid",
        llm_generate=lambda _prompt: response,
        candidate_budget=3,
    )

    assert [patch["id"] for patch in patches] == [
        "window_local_global",
        "window_strong_replay",
        "llm_specific",
    ]


def test_llm_prompt_contains_compact_array_schema_and_budget() -> None:
    context = ProposalContext.from_failure_report(failure_report(), task_id="tennis")

    prompt = build_llm_prompt(context, candidate_budget=1)

    assert '"candidate_budget": 1' in prompt
    assert '"body_pos_rewards": [' in prompt
    assert '"body_z_rewards": []' in prompt
    assert "at most three body-specific reward terms" in prompt
    assert "compact single-line JSON" in prompt


def test_llm_backend_rejects_more_than_three_body_reward_terms() -> None:
    context = ProposalContext.from_failure_report(failure_report(), task_id="tennis")
    candidate = {
        "id": "over_budget_terms",
        "sampling": {"mode": "window", "start": 100, "end": 160, "ratio": 0.5},
        "reward_overrides": {},
        "body_pos_rewards": [
            {
                "name": f"term_{index}",
                "body_names": ["pelvis"],
                "weight": 1.0,
                "std": 0.1,
            }
            for index in range(4)
        ],
        "body_z_rewards": [],
        "body_lin_vel_rewards": [],
        "body_ang_vel_rewards": [],
    }
    response = {"candidates": [candidate]}

    with pytest.raises(ValueError, match="at most 3"):
        generate_candidates(
            context,
            backend="llm",
            llm_generate=lambda _prompt: response,
            candidate_budget=1,
        )
