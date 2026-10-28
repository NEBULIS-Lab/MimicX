"""Fixed native state capture for the existing cross-controller physical scorer."""


def common_protocol():
    return dict(root_height_m=.25, relative_ee_height_m=.25, root_orientation_rad=1.,
                persistence_frames=5, common_prefix_seconds=1., clock_tolerance_s=.03,
                state_gap_s=.045, critical_bodies=['left_ankle_roll_link', 'right_ankle_roll_link',
                                                'left_wrist_yaw_link', 'right_wrist_yaw_link'])


def capture_class(base, contract, samples):
    """Observe link states before frame one and after each real native step."""
    import numpy as np

    class Capture(base):
        def _study_sample(self):
            robot = self.scene['robot']
            joints = [list(robot.joint_names).index(name) for name in contract['joint_names']]
            bodies = [list(robot.body_names).index(name) for name in contract['body_names']]
            data = robot.data
            arrays = dict(joint_pos=data.joint_pos[:, joints], joint_vel=data.joint_vel[:, joints],
                          body_pos=data.body_link_pos_w[:, bodies], body_quat=data.body_link_quat_w[:, bodies],
                          body_lin_vel=data.body_link_lin_vel_w[:, bodies], body_ang_vel=data.body_link_ang_vel_w[:, bodies])
            return {key: value[0].detach().cpu().numpy().copy() for key, value in arrays.items()}

        def step(self, action):
            if not np.isclose(self.step_dt, .02, atol=1e-9, rtol=0):
                raise ValueError('Native physical control clock is not 50 Hz')
            if not samples:
                samples.append((self._study_sample(), False))
            result = super().step(action)
            if len(result) != 5:
                raise ValueError('Unsupported native step API; terminated/truncated flags required')
            done = bool((result[2] | result[3]).any().detach().cpu().item())
            samples.append((self._study_sample(), done))
            return result

    return Capture


def align_rotation_csv(output, actual, target, body_names, valid, anchor_body):
    """Keep native raw telemetry, but align gate rotation to the CSV frame clock."""
    import csv
    import shutil
    from .sonic_study import quaternion_error
    from .motion_study import digest, save_json
    path = output / 'steps.csv'
    raw = output / 'steps.native_raw.csv'
    if raw.exists():
        raise FileExistsError(raw)
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), list(reader.fieldnames or [])
    if [int(row['step']) for row in rows] != list(range(1, len(valid))):
        raise ValueError('Native rotation telemetry does not match the inclusive capture clock')
    if 'error_anchor_rot' not in fields:
        raise ValueError('Native rotation telemetry is missing')
    index = body_names.index(anchor_body)
    angles = quaternion_error(actual['body_quat'][:, index], target['body_quat'][:, index])
    cached_key = 'error_anchor_rot_native_cached'
    aligned_key = 'rotation_clock_aligned'
    if cached_key in fields or aligned_key in fields:
        raise ValueError('Native rotation telemetry was already transformed')
    deltas = []
    for frame, row in enumerate(rows, start=1):
        row[cached_key] = row['error_anchor_rot']
        row[aligned_key] = bool(valid[frame])
        if valid[frame]:
            deltas.append(abs(float(row['error_anchor_rot']) - float(angles[frame])))
            row['error_anchor_rot'] = float(angles[frame])
    shutil.copy2(path, raw)
    temporary = path.with_suffix('.pending')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + [cached_key, aligned_key])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    save_json(output / 'rotation_clock.json', dict(anchor_body=anchor_body,
        formula='Existing quaternion_error on captured anchor and immutable reference at the same frame',
        control_rows=len(rows), aligned_valid_rows=len(deltas), first_reference_frame=1,
        maximum_cached_vs_aligned_delta_rad=max(deltas) if deltas else None,
        raw_steps_sha256=digest(raw), aligned_steps_sha256=digest(path),
        invalid_reset_tail='Preserved native cached values; excluded from gate evidence'))


def score_capture(campaign, task, samples, output, *, align_rotation=False, anchor_body='torso_link'):
    import numpy as np
    from mimicx.experiments.sonic_study import common_metrics
    from .motion_study import save_json
    contract = campaign['reference_contract']
    n = task['frames']
    if len(samples) != n:
        raise ValueError('Native capture does not cover inclusive phase-zero through final frame')
    actual = {key: np.stack([s[0][key] for s in samples]) for key in samples[0][0]}
    failures = [i for i, (_, done) in enumerate(samples) if done]
    first = failures[0] if failures else None
    valid = np.arange(n) < (first if first is not None else n)
    fields = dict(joint_pos='joint_pos', joint_vel='joint_vel', body_pos='body_pos_w', body_quat='body_quat_w',
                  body_lin_vel='body_lin_vel_w', body_ang_vel='body_ang_vel_w')
    with np.load(task['motion_path'], allow_pickle=False) as reference:
        target = {key: reference[source].copy() for key, source in fields.items()}
        for key in ('joint_names', 'body_names'):
            if key in reference and list(reference[key]) != contract[key]:
                raise ValueError('Native reference named-order contract changed')
    metrics = common_metrics(actual, target, model_xml=campaign['runtime']['common_model_xml'],
        joint_names=contract['joint_names'], body_names=contract['body_names'],
        protocol=campaign['runtime']['common_protocol'], reference_frames=np.arange(n),
        native_failure_frame=first, valid_mask=valid)
    steps = metrics.pop('_steps')
    metrics['native_termination'] = 'MjLab terminated|truncated with automatic reset; first reset and all later states censored'
    metrics['native_termination_source'] = metrics['native_termination']
    metrics['native_failure_frame'] = first
    metrics['scoring_body_names'] = contract['body_names']
    np.savez_compressed(output / 'native_actual.npz', **actual, valid_mask=valid, timestamps=np.arange(n) / 50.)
    np.savez_compressed(output / 'common_steps.npz', **steps)
    save_json(output / 'common_metrics.json', metrics)
    if align_rotation:
        align_rotation_csv(output, actual, target, contract['body_names'], valid, anchor_body)
    return metrics
