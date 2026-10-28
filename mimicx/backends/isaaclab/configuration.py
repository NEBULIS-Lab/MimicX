"""CPU-testable translation of bounded MimicX patches to Isaac reward terms."""
from mimicx.refinement.proposals import ProposalContext, validate_patch


ALIASES = {"motion_global_root_pos": "motion_global_anchor_pos",
           "motion_global_root_ori": "motion_global_anchor_ori"}


def apply_patch(cfg, patch, *, frames, reward_factory, functions):
    context = ProposalContext("isaac", frames - 1, 0, None, (0, frames - 1), (), (), (), ())
    patch = validate_patch(patch, context=context)
    cfg.commands.motion.mimicx_sampling = patch["sampling"]
    for key, values in patch["reward_overrides"].items():
        term = getattr(cfg.rewards, ALIASES.get(key, key))
        if "weight" in values:
            term.weight = float(values["weight"])
        if "std" in values:
            term.params["std"] = float(values["std"])
    for family, function in functions.items():
        for index, term in enumerate(patch[family]):
            missing = set(term["body_names"]) - set(cfg.commands.motion.body_names)
            if missing:
                raise ValueError(f"Untracked reward bodies: {sorted(missing)}")
            # Use generated attribute names, never untrusted proposal identifiers.
            setattr(cfg.rewards, f"mimicx_{family}_{index}", reward_factory(
                func=function, weight=float(term["weight"]),
                params={"command_name": "motion", "std": float(term["std"]),
                        "body_names": list(term["body_names"])}))


def reapply_learning_rate(runner, learning_rate):
    """Optimizer state loading restores old LR; explicit continuation LR wins."""
    runner.alg.learning_rate = learning_rate
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = learning_rate


def policy_observations(result):
    # IsaacLab2 wrapper returns (obs, extras); Lab3/RSL5 returns TensorDict.
    return result[0] if isinstance(result, tuple) else result
