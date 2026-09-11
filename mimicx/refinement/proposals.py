"""Constrained candidate proposal backends for policy-aware AutoRefine."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


CORE_BODIES = ("pelvis", "torso_link")
ARM_BODIES = (
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
LOWER_BODIES = (
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
)
ALLOWED_BODIES = frozenset(CORE_BODIES + ARM_BODIES + LOWER_BODIES)
PATCH_KEYS = frozenset(
    {
        "id",
        "sampling",
        "reward_overrides",
        "body_pos_rewards",
        "body_z_rewards",
        "body_lin_vel_rewards",
        "body_ang_vel_rewards",
    }
)
REWARD_OVERRIDE_KEYS = frozenset(
    {
        "motion_global_root_pos",
        "motion_global_root_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "action_rate_l2",
    }
)


@dataclass(frozen=True)
class ProposalContext:
    task_id: str
    horizon: int
    done_count: int
    first_failure_step: int | None
    failure_window: tuple[int, int]
    dominant_bodies: tuple[str, ...]
    dominant_channels: tuple[str, ...]
    recommended_families: tuple[str, ...]
    top_metrics: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_failure_report(
        cls,
        report: Mapping[str, Any],
        *,
        task_id: str = "unknown",
    ) -> "ProposalContext":
        windows = report.get("failure_windows", [])
        if not windows:
            raise ValueError("Failure report has no failure_windows")
        failure = windows[0]
        window = failure.get("window")
        if not isinstance(window, Sequence) or len(window) != 2:
            raise ValueError("Failure window must contain start and end")
        summary = report.get("summary", {})
        return cls(
            task_id=task_id,
            horizon=int(summary.get("steps", max(int(window[1]), 1))),
            done_count=int(summary.get("done_any_count", 0) or 0),
            first_failure_step=(
                int(summary["first_done_step"])
                if summary.get("first_done_step") is not None
                else None
            ),
            failure_window=(int(window[0]), int(window[1])),
            dominant_bodies=tuple(str(value) for value in failure.get("dominant_bodies", [])),
            dominant_channels=tuple(str(value) for value in failure.get("dominant_channels", [])),
            recommended_families=tuple(
                str(value) for value in failure.get("recommended_families", [])
            ),
            top_metrics=tuple(
                item for item in failure.get("top_metrics", []) if isinstance(item, Mapping)
            ),
        )

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "horizon": self.horizon,
            "done_count": self.done_count,
            "first_failure_step": self.first_failure_step,
            "failure_window": list(self.failure_window),
            "dominant_bodies": list(self.dominant_bodies),
            "dominant_channels": list(self.dominant_channels),
            "recommended_families": list(self.recommended_families),
            "top_metrics": list(self.top_metrics),
        }


def _unique(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _valid_bodies(bodies: Sequence[str]) -> list[str]:
    return [body for body in _unique(bodies) if body in ALLOWED_BODIES]


def candidate_patch(
    candidate_id: str,
    sampling: Mapping[str, Any],
    families: Sequence[str],
    bodies: Sequence[str],
) -> dict[str, Any]:
    body_focus = _valid_bodies(bodies)
    lower_focus = [body for body in body_focus if any(x in body for x in ("hip", "knee", "ankle"))]
    arm_focus = [body for body in body_focus if any(x in body for x in ("shoulder", "elbow", "wrist"))]
    patch: dict[str, Any] = {
        "id": candidate_id,
        "sampling": dict(sampling),
        "reward_overrides": {
            "motion_global_root_pos": {"weight": 5.2, "std": 0.145},
            "motion_global_root_ori": {"weight": 1.25, "std": 0.28},
            "motion_body_pos": {"std": 0.24},
            "motion_body_ori": {"std": 0.32},
            "action_rate_l2": {"weight": -0.065},
        },
        "body_pos_rewards": [
            {"name": "core_pos", "body_names": list(CORE_BODIES), "weight": 2.5, "std": 0.11}
        ],
        "body_z_rewards": [],
        "body_lin_vel_rewards": [
            {"name": "torso_lin_vel", "body_names": ["torso_link"], "weight": 0.55, "std": 0.65}
        ],
        "body_ang_vel_rewards": [],
    }
    if "lower_body_contact_z" in families or "z_clearance_contact" in families or lower_focus:
        lower_names = lower_focus or list(LOWER_BODIES)
        patch["body_pos_rewards"].append(
            {"name": "lower_body_pos", "body_names": lower_names, "weight": 2.3, "std": 0.11}
        )
        patch["body_z_rewards"].append(
            {"name": "lower_body_z", "body_names": lower_names, "weight": 1.8, "std": 0.052}
        )
        patch["body_lin_vel_rewards"].append(
            {"name": "lower_body_lin_vel", "body_names": lower_names, "weight": 0.8, "std": 0.60}
        )
    if "arm_wrist_precision" in families or arm_focus:
        arm_names = arm_focus or list(ARM_BODIES)
        patch["body_pos_rewards"].append(
            {"name": "arm_pos", "body_names": arm_names, "weight": 1.8, "std": 0.13}
        )
        patch["body_lin_vel_rewards"].append(
            {"name": "arm_lin_vel", "body_names": arm_names, "weight": 0.55, "std": 0.75}
        )
    if "dynamics_smoothness" in families:
        patch["body_ang_vel_rewards"].append(
            {"name": "core_ang_vel", "body_names": list(CORE_BODIES), "weight": 0.35, "std": 2.1}
        )
    return patch


def heuristic_candidates(context: ProposalContext) -> list[dict[str, Any]]:
    start, end = context.failure_window
    families = list(context.recommended_families)
    bodies = list(context.dominant_bodies)
    return [
        candidate_patch(
            "window_local_global",
            {"mode": "window", "start": start, "end": end, "ratio": 0.45},
            families,
            bodies,
        ),
        candidate_patch(
            "window_strong_replay",
            {"mode": "window", "start": start, "end": end, "ratio": 0.65},
            families + ["dynamics_smoothness"],
            bodies,
        ),
        candidate_patch(
            "full_start_consolidate",
            {"mode": "start"},
            _unique(families + ["root_core_guard", "dynamics_smoothness"]),
            bodies,
        ),
    ]


def _bounded_number(value: Any, *, name: str, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math_is_finite(parsed) or not low <= parsed <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return parsed


def math_is_finite(value: float) -> bool:
    return value not in (float("inf"), float("-inf")) and value == value


def validate_patch(patch: Mapping[str, Any], *, context: ProposalContext) -> dict[str, Any]:
    extra = set(patch) - PATCH_KEYS
    missing = PATCH_KEYS - set(patch)
    if extra:
        raise ValueError(f"Forbidden proposal keys: {', '.join(sorted(extra))}")
    if missing:
        raise ValueError(f"Missing proposal keys: {', '.join(sorted(missing))}")
    candidate_id = str(patch["id"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]{2,63}", candidate_id):
        raise ValueError("id must be a lowercase snake-case identifier")

    sampling = patch["sampling"]
    if not isinstance(sampling, Mapping):
        raise ValueError("sampling must be an object")
    mode = sampling.get("mode")
    if mode not in {"start", "window"}:
        raise ValueError("sampling.mode must be start or window")
    if mode == "window":
        start = int(sampling.get("start", -1))
        end = int(sampling.get("end", -1))
        if not 0 <= start < end <= context.horizon:
            raise ValueError("sampling window is outside the rollout horizon")
        failure_start, failure_end = context.failure_window
        if end < failure_start or start > failure_end:
            raise ValueError("sampling window must overlap the diagnosed failure window")
        _bounded_number(sampling.get("ratio"), name="sampling.ratio", low=0.2, high=0.8)

    overrides = patch["reward_overrides"]
    if not isinstance(overrides, Mapping):
        raise ValueError("reward_overrides must be an object")
    unknown_overrides = set(overrides) - REWARD_OVERRIDE_KEYS
    if unknown_overrides:
        raise ValueError(f"Unknown reward overrides: {', '.join(sorted(unknown_overrides))}")
    for reward_name, parameters in overrides.items():
        if not isinstance(parameters, Mapping) or set(parameters) - {"weight", "std"}:
            raise ValueError(f"Invalid parameters for {reward_name}")
        if "weight" in parameters:
            low, high = (-1.0, 0.0) if reward_name == "action_rate_l2" else (0.0, 10.0)
            _bounded_number(parameters["weight"], name=f"{reward_name}.weight", low=low, high=high)
        if "std" in parameters:
            _bounded_number(parameters["std"], name=f"{reward_name}.std", low=0.01, high=5.0)

    for family in (
        "body_pos_rewards",
        "body_z_rewards",
        "body_lin_vel_rewards",
        "body_ang_vel_rewards",
    ):
        terms = patch[family]
        if not isinstance(terms, list):
            raise ValueError(f"{family} must be a list")
        for term in terms:
            if not isinstance(term, Mapping) or set(term) != {"name", "body_names", "weight", "std"}:
                raise ValueError(f"Invalid reward term in {family}")
            bodies = term["body_names"]
            if not isinstance(bodies, list) or not bodies or any(body not in ALLOWED_BODIES for body in bodies):
                raise ValueError(f"Invalid body_names in {family}")
            _bounded_number(term["weight"], name=f"{family}.weight", low=0.0, high=10.0)
            _bounded_number(term["std"], name=f"{family}.std", low=0.01, high=5.0)
    return json.loads(json.dumps(patch))


def build_llm_prompt(context: ProposalContext, *, candidate_budget: int) -> str:
    focus_body = next(
        (body for body in context.dominant_bodies if body in ALLOWED_BODIES),
        "pelvis",
    )
    start, end = context.failure_window
    example = {
        "candidates": [
            {
                "id": "failure_focused_candidate",
                "sampling": {
                    "mode": "window",
                    "start": start,
                    "end": end,
                    "ratio": 0.5,
                },
                "reward_overrides": {
                    "motion_global_root_pos": {"weight": 5.0, "std": 0.15}
                },
                "body_pos_rewards": [
                    {
                        "name": "focus_pos",
                        "body_names": [focus_body],
                        "weight": 2.0,
                        "std": 0.12,
                    }
                ],
                "body_z_rewards": [],
                "body_lin_vel_rewards": [],
                "body_ang_vel_rewards": [],
            }
        ]
    }
    contract = {
        "output_example": example,
        "required_patch_keys": sorted(PATCH_KEYS),
        "allowed_bodies": sorted(ALLOWED_BODIES),
        "allowed_reward_overrides": sorted(REWARD_OVERRIDE_KEYS),
        "candidate_budget": candidate_budget,
        "constraints": [
            "Do not emit verification, acceptance, guard, evaluation, checkpoint, or command fields.",
            "Sampling windows must overlap the diagnosed failure window.",
            "Every body_*_rewards value must be a JSON array of reward-term objects, never a body-to-weight map.",
            "Use at most three body-specific reward terms in total and do not enumerate every body.",
            "Return compact single-line JSON only, with no markdown or explanation.",
        ],
    }
    return (
        "You propose bounded training patches for humanoid motion tracking. "
        "Rollout evaluation and acceptance are immutable and handled outside the model.\n"
        f"CONTEXT={json.dumps(context.prompt_payload(), sort_keys=True)}\n"
        f"CONTRACT={json.dumps(contract, sort_keys=True)}"
    )


def _parse_llm_response(response: str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return response
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].lstrip()
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("LLM response must be a JSON object")
    return payload


def llm_candidates(
    context: ProposalContext,
    *,
    llm_generate: Callable[[str], str | Mapping[str, Any]],
    candidate_budget: int,
) -> list[dict[str, Any]]:
    payload = _parse_llm_response(llm_generate(build_llm_prompt(context, candidate_budget=candidate_budget)))
    if set(payload) != {"candidates"} or not isinstance(payload["candidates"], list):
        raise ValueError("LLM response must contain only a candidates list")
    patches = []
    for candidate in payload["candidates"][:candidate_budget]:
        patch = validate_patch(candidate, context=context)
        term_count = sum(
            len(patch[family])
            for family in (
                "body_pos_rewards",
                "body_z_rewards",
                "body_lin_vel_rewards",
                "body_ang_vel_rewards",
            )
        )
        if term_count > 3:
            raise ValueError("LLM proposals may contain at most 3 body reward terms")
        patches.append(patch)
    return patches


def generate_candidates(
    context: ProposalContext,
    *,
    backend: str = "heuristic",
    llm_generate: Callable[[str], str | Mapping[str, Any]] | None = None,
    candidate_budget: int = 3,
) -> list[dict[str, Any]]:
    if candidate_budget < 1:
        raise ValueError("candidate_budget must be positive")
    heuristic = [validate_patch(patch, context=context) for patch in heuristic_candidates(context)]
    if backend == "heuristic":
        return heuristic[:candidate_budget]
    if backend not in {"llm", "hybrid"}:
        raise ValueError(f"Unsupported proposal backend: {backend}")
    if llm_generate is None:
        raise ValueError(f"{backend} backend requires llm_generate")
    proposed = llm_candidates(
        context,
        llm_generate=llm_generate,
        candidate_budget=candidate_budget,
    )
    if backend == "llm":
        return proposed
    heuristic_slots = max(candidate_budget - len(proposed), 0)
    combined = heuristic[:heuristic_slots] + proposed
    seen: set[str] = set()
    unique_patches = []
    for patch in combined:
        if patch["id"] in seen:
            continue
        seen.add(patch["id"])
        unique_patches.append(patch)
    return unique_patches[:candidate_budget]
