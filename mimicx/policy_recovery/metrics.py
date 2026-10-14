"""CPU-only execution gates; never score post-reset states as continuation."""
import math
import statistics


def _done(row):
    value = str(row['done_any']).lower()
    if value not in ('true', 'false', '1', '0'):
        raise ValueError(f'Invalid done flag: {value}')
    return value in ('true', '1')


def _values(row):
    root = math.sqrt(sum((float(row[f'robot_anchor_{a}']) -
                         float(row[f'ref_anchor_{a}'])) ** 2 for a in 'xyz'))
    values = (root, float(row['body_pos_error_max']), float(row['error_anchor_rot']))
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Non-finite or negative tracking error')
    return values


def _percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] * (hi - x) + values[hi] * (x - lo) if hi != lo else values[lo]


def summarize(rows, horizon):
    if horizon <= 0 or len(rows) != horizon:
        raise ValueError('Evaluation must contain exactly the requested horizon')
    if [int(r['step']) for r in rows] != list(range(1, horizon + 1)):
        raise ValueError('Non-contiguous evaluation steps')
    values = [_values(r) for r in rows]
    failures = [i for i, r in enumerate(rows) if _done(r)]
    valid = failures[0] if failures else horizon
    result = dict(complete=not failures, valid_steps=valid, terminations=len(failures),
                  horizon=horizon, first_failure_step=valid + 1 if failures else None)
    for j, key in enumerate(('world_anchor', 'aligned_body_max', 'anchor_rotation')):
        series = [v[j] for v in values[:valid]]
        result[key + '_mean'] = statistics.mean(series) if series else None
        result[key + '_p95'] = _percentile(series, .95)
    return result


def qualifies(results):
    """Operational video-quality gate, in addition to strict termination checks."""
    limits = dict(world_anchor_mean=.08, world_anchor_p95=.15,
                  aligned_body_max_mean=.10, aligned_body_max_p95=.20,
                  anchor_rotation_mean=.20)
    return len(results) >= 3 and all(
        r['complete'] and all(r[k] is not None and r[k] <= limit
                             for k, limit in limits.items()) for r in results)


def accept(old_rows, new_rows, horizon):
    if len(old_rows) != len(new_rows) or len(old_rows) < 3:
        raise ValueError('Gate requires matched repeated evaluations')
    old = [summarize(r, horizon) for r in old_rows]
    new = [summarize(r, horizon) for r in new_rows]
    def rank(items):
        return (sum(r['complete'] for r in items), min(r['valid_steps'] for r in items),
                statistics.mean(r['valid_steps'] for r in items))
    prefix_old, prefix_new = [], []
    for a, b, sa, sb in zip(old_rows, new_rows, old, new):
        n = min(sa['valid_steps'], sb['valid_steps'])
        if n == 0:
            return dict(accepted=False, reason='No valid common prefix', old=old, new=new)
        prefix_old.append(tuple(statistics.mean(_values(r)[j] for r in a[:n]) for j in range(3)))
        prefix_new.append(tuple(statistics.mean(_values(r)[j] for r in b[:n]) for j in range(3)))
    # Small absolute tolerance avoids punishing tiny numerical fluctuations.
    guarded = all(b[j] <= a[j] * 1.20 + (.015 if j < 2 else .03)
                  for a, b in zip(prefix_old, prefix_new) for j in range(3))
    old_score = statistics.mean(sum(v) for v in prefix_old)
    new_score = statistics.mean(sum(v) for v in prefix_new)
    improved = rank(new) > rank(old) or (rank(new) == rank(old) and new_score < .98 * old_score)
    return dict(accepted=guarded and improved, reason=('Accepted' if guarded and improved else
                'Tracking guard' if not guarded else 'No execution improvement'), old=old, new=new,
                common_prefix_old=prefix_old, common_prefix_new=prefix_new)


def failure_window(repeats, horizon):
    summaries = [summarize(r, horizon) for r in repeats]
    failures = [r['first_failure_step'] for r in summaries if not r['complete']]
    if failures:
        center = min(failures)
    else:
        center = max(range(horizon), key=lambda i: statistics.mean(
            sum(_values(r[i])[:2]) for r in repeats))
    radius = max(10, min(90, horizon // 8))
    return max(0, center - radius), min(horizon, center + radius)
