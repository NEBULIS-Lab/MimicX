"""Prospective startup-recovery selection for the motion study, without reward.

The rollout exporter reads state AFTER ``env.step``; MjLab 1.2 auto-resets
inside that call. The first done row and every later row are therefore excluded
from tracking evidence, including when the incumbent fails on its first step.

Every candidate repeat must have positive valid evidence and must preserve that
seed's valid horizon. For an incumbent with no valid evidence, bound the whole
candidate pre-failure prefix using STARTUP_PRECISION_BOUNDS. The position bounds
and rotation mean reuse the existing recovery precision limits; the additional
0.35-rad rotation p95 limits sustained orientation excursions. These fixed
defaults are selection rules, not tuned per task or candidate.

Any positive prefix, including a partial recovery, can qualify for selection;
full completion remains a separate evaluation outcome. Seeds with shared valid
evidence retain the original relative tracking guard. With no startup failures,
the existing accept rule is reused after the per-seed non-regression check.
"""

import statistics
from types import MappingProxyType

from mimicx.policy_recovery.metrics import _values, accept as recovery_accept, summarize


GATE_VERSION = 'motion_study_startup_recovery_v1'
STARTUP_PRECISION_BOUNDS = MappingProxyType({
    'world_anchor_mean': .08,
    'world_anchor_p95': .15,
    'aligned_body_max_mean': .10,
    'aligned_body_max_p95': .20,
    'anchor_rotation_mean': .20,
    'anchor_rotation_p95': .35,
})


def accept(old_rows, new_rows, horizon):
    """Return the existing acceptance dictionary plus auditable startup guards.

    Repeat order must be the same matched seed order in both arguments. Missing
    common-prefix evidence is represented by None, never by terminal values or
    invented zero error. Inputs retain the existing full-rollout validation.
    """
    if len(old_rows) != len(new_rows) or len(old_rows) < 3:
        raise ValueError('Gate requires matched repeated evaluations')
    old = [summarize(rows, horizon) for rows in old_rows]
    new = [summarize(rows, horizon) for rows in new_rows]
    startup = [i for i, result in enumerate(old) if result['valid_steps'] == 0]
    regressions = [i for i, (a, b) in enumerate(zip(old, new))
                   if b['valid_steps'] < a['valid_steps']]
    no_evidence = [i for i, result in enumerate(new) if result['valid_steps'] == 0]
    audit = dict(gate_version=GATE_VERSION, uses_training_reward=False,
                 startup_seed_indices=startup,
                 startup_precision_bounds=dict(STARTUP_PRECISION_BOUNDS),
                 startup_checks=[], no_seed_frontier_regression=not regressions,
                 regressed_seed_indices=regressions, no_evidence_seed_indices=no_evidence)
    result = dict(accepted=False, reason='', old=old, new=new, **audit)
    if regressions:
        result['reason'] = 'Per-seed horizon regression'
        return result
    if no_evidence:
        result['reason'] = 'No valid candidate evidence'
        return result
    if not startup:
        return {**recovery_accept(old_rows, new_rows, horizon), **audit}

    prefix_old, prefix_new = [], []
    tracking_guard = True
    for i, (a, b, sa, sb) in enumerate(zip(old_rows, new_rows, old, new)):
        n = min(sa['valid_steps'], sb['valid_steps'])
        if n == 0:
            prefix_old.append(None)
            prefix_new.append(None)
            measured = {key: sb[key] for key in STARTUP_PRECISION_BOUNDS}
            violations = [key for key, limit in STARTUP_PRECISION_BOUNDS.items()
                          if measured[key] > limit]
            result['startup_checks'].append(dict(seed_index=i, valid_steps=sb['valid_steps'],
                measured=measured, violated_bounds=violations, passed=not violations))
            continue
        means_old = tuple(statistics.mean(_values(row)[j] for row in a[:n]) for j in range(3))
        means_new = tuple(statistics.mean(_values(row)[j] for row in b[:n]) for j in range(3))
        prefix_old.append(means_old)
        prefix_new.append(means_new)
        tracking_guard &= all(means_new[j] <= means_old[j] * 1.20 + (.015 if j < 2 else .03)
                              for j in range(3))

    startup_guard = all(check['passed'] for check in result['startup_checks'])
    # All new prefixes are positive and none regresses, so the old minimum of
    # zero strictly improves. No incomparable startup error enters a mean score.
    result.update(accepted=tracking_guard and startup_guard,
                  reason=('Tracking guard' if not tracking_guard else
                          'Startup precision guard' if not startup_guard else 'Accepted'),
                  common_prefix_old=prefix_old, common_prefix_new=prefix_new)
    return result
