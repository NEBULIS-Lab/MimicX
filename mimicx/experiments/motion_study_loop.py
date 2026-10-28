"""Matched candidate budgets, native evaluation, and existing AutoRefine gates."""
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys
import time

from mimicx.policy_recovery.metrics import failure_window, qualifies, summarize
from .motion_study_gate import accept
from mimicx.refinement.proposals import ALLOWED_BODIES, ProposalContext, generate_candidates, validate_patch
from . import motion_study as study


def horizon(task):
    # Start at frame zero, advance through the final frame, never wrap the motion.
    return task['frames'] - 1


def validation_protocol(raw=None):
    p = dict(raw or {})
    p.setdefault('selection_seeds', [1001, 1002, 1003])
    p.setdefault('selection_noise', .005)
    p.setdefault('nominal_noise', .005)
    p.setdefault('perturbed_noise', .01)
    p.setdefault('holdout', {
        'development': {'nominal': [2001, 2002, 2003], 'perturbed': [3001, 3002, 3003]},
        'confirmation': {'nominal': list(range(2001, 2011)), 'perturbed': list(range(3001, 3011))}})
    if len(p['selection_seeds']) < 3:
        raise ValueError('Selection requires at least three repeated validation seeds')
    for split, count in (('development', 3), ('confirmation', 10)):
        groups = [p['selection_seeds'], p['holdout'][split]['nominal'], p['holdout'][split]['perturbed']]
        if any(len(g) < count for g in groups[1:]):
            raise ValueError('Insufficient holdout repeats')
        flat = sum(groups, [])
        if any(type(s) is not int or s < 0 for s in flat) or len(flat) != len(set(flat)):
            raise ValueError('Validation and holdout seeds must be unique and disjoint')
    for key in ('selection_noise', 'nominal_noise', 'perturbed_noise'):
        if not 0 <= p[key] <= .1:
            raise ValueError('Invalid evaluation joint initialization noise')
    return p


def static_patches(task):
    n = horizon(task)
    context = ProposalContext(task['task_id'], n, 0, None, (0, n), tuple(sorted(ALLOWED_BODIES)),
                              (), ('root_core_guard', 'lower_body_contact_z', 'arm_wrist_precision'), ())
    return generate_candidates(context, candidate_budget=2)


def untried_patch(context, tried):
    def signature(patch):
        return json.dumps({key: value for key, value in patch.items() if key != 'id'}, sort_keys=True, separators=(',', ':'))
    seen = {signature(patch) for patch in tried}
    for patch in generate_candidates(context, candidate_budget=3):
        if signature(patch) not in seen:
            return patch
    raise ValueError('No untried bounded proposal remains; do not repeat an identical candidate')


def apply_patch(cfg, patch, task):
    if patch is None:
        return
    n = horizon(task)
    context = ProposalContext(task['task_id'], n, 0, None, (0, n), (), (), (), ())
    patch = validate_patch(patch, context=context)
    from src.tasks.tracking.config.g1.env_cfgs import _apply_mimicx_autorefine_patch
    missing = set(patch['reward_overrides']) - set(cfg.rewards)
    if missing:
        raise ValueError(f'Patch references missing native rewards: {sorted(missing)}')
    original_motion = cfg.commands['motion'].motion_file
    _apply_mimicx_autorefine_patch(cfg, patch, play=False)
    if cfg.commands['motion'].motion_file != original_motion:
        raise RuntimeError('Training patch changed immutable reference')


def run_rollout_cli(script):
    try:
        runpy.run_path(str(script), run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise


def evaluate_native(campaign, job, args):
    """Register one common native play contract and execute the real rollout CLI."""
    if args.output.exists():
        raise FileExistsError(args.output)
    task = next(t for t in campaign['tasks'] if t['task_id'] == job['task_id'])
    study.native_configs(campaign['runtime'], task, job['seed'])
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, register_mjlab_task
    from mimicx.parkour.environment import install_initial_reference_refresh
    from mimicx.policy_recovery.worker import evaluation_argv
    cfg = load_env_cfg('Unitree-G1-Tracking', play=True)
    cfg.episode_length_s = horizon(task) / 50. + 2.
    cmd = cfg.commands['motion']
    cmd.motion_file = task['motion_path']
    cmd.sampling_mode = 'start'
    cmd.joint_position_range = (-args.noise, args.noise)
    cfg.viewer.max_extra_envs = 0
    name = 'MimicX-Motion-Study-Common-Evaluation'
    register_mjlab_task(name, cfg, cfg, load_rl_cfg('Unitree-G1-Tracking'))
    install_initial_reference_refresh()
    args.device = campaign['runtime']['device']
    args.video, args.no_ghost = False, True
    script = Path(campaign['runtime'].get('rollout_script', study.ROOT / 'scripts/rollout_mjlab_motion_smoke.py'))
    argv = evaluation_argv(script, name, {'motion': task['motion_path'], 'horizon': horizon(task)}, args, kind='recovery')
    args.output.mkdir(parents=True)
    previous = sys.argv
    import mjlab.envs as native_envs
    original_env = native_envs.ManagerBasedRlEnv
    samples = []
    try:
        if campaign.get('reference_contract'):
            from .motion_study_scoring import capture_class
            native_envs.ManagerBasedRlEnv = capture_class(original_env, campaign['reference_contract'], samples)
        sys.argv = argv
        run_rollout_cli(script)
    finally:
        native_envs.ManagerBasedRlEnv = original_env
        sys.argv = previous
    if campaign.get('reference_contract'):
        from .motion_study_scoring import score_capture
        score_capture(campaign, task, samples, args.output, align_rotation=True, anchor_body=cmd.anchor_body_name)


def load_rows(path, n):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    summarize(rows, n)
    return rows


def verify_replay(folder, n):
    import numpy as np
    manifest = study.read(folder / 'manifest.json')
    if manifest['frames'] != n or manifest['joint_count'] != 29 or manifest['fps'] != 50.:
        raise ValueError('Replay does not match the full-motion 29-DoF / 50 Hz contract')
    required = {'timestamps', 'robot_root_pose', 'robot_joint_positions', 'robot_body_pose',
                'reference_root_pose', 'reference_joint_positions', 'reference_body_pose'}
    if set(manifest['arrays']) != required:
        raise ValueError('Replay is missing required state arrays')
    for key, record in manifest['arrays'].items():
        path = folder / record['filename']
        if path.parent != folder or study.digest(path) != record['sha256']:
            raise ValueError('Replay state artifact changed')
        data = np.load(path, allow_pickle=False)
        if data.shape[0] != n or not np.isfinite(data).all():
            raise ValueError('Invalid replay state array')
        if key == 'timestamps' and not np.allclose(data, np.arange(1, n + 1) / 50., atol=1e-7, rtol=0):
            raise ValueError('Replay clock is not the fixed 50 Hz clock')


def checked_process(command, runtime, folder, label):
    folder.mkdir(parents=True, exist_ok=True)
    attempt_path = folder / 'attempts' / f'{time.time_ns()}.json'
    started = time.time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    attempt = dict(command=command, started=started, status='running', kind=label,
                   accounting_scope='nested; included in enclosing workflow attempt total', log=str(folder / 'run.log'))
    study.save_json(attempt_path, attempt)
    try:
        with (folder / 'run.log').open('a') as log:
            result = subprocess.run(command, cwd=runtime['backend_root'], stdout=log,
                                    stderr=subprocess.STDOUT, check=False,
                                    env={**os.environ, 'GIT_PYTHON_REFRESH': os.environ.get('GIT_PYTHON_REFRESH', 'quiet')})
        attempt['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'{label} failed ({result.returncode}); see {folder / "run.log"}')
        attempt['status'] = 'completed'
    except BaseException as exc:
        attempt.update(status='failed', error=repr(exc))
        raise
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall = time.time() - started
        attempt.update(wall_seconds=wall, finished=time.time(),
                       child_cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                       assigned_gpu_wall_seconds=wall if runtime['device'] != 'cpu' and label == 'evaluation' else 0,
                       measured_gpu_compute_seconds=None)
        study.save_json(attempt_path, attempt)


def evaluate(campaign_path, campaign, job, checkpoint, folder, seeds, noise):
    runtime = campaign['runtime']
    task = next(t for t in campaign['tasks'] if t['task_id'] == job['task_id'])
    n = horizon(task)
    repeats = []
    for seed in seeds:
        run = folder / f'seed{seed}'
        output = run / 'rollout'
        signature = dict(campaign_sha256=study.digest(campaign_path), checkpoint_sha256=study.digest(checkpoint),
                         motion_sha256=task['sha256'], seed=seed, noise=noise, horizon=n)
        study.immutable_json(run / 'contract.json', signature)
        if (run / 'done.json').exists():
            receipt = study.verified_receipt(run)
            if receipt['signature'] != signature:
                raise ValueError('Evaluation receipt mismatch')
        else:
            if output.exists():
                output.rename(run / f'incomplete_rollout_{time.time_ns()}')
            command = [runtime['python'], str(study.ENTRY), '_evaluate', '--campaign', str(campaign_path),
                       '--job', job['id'], '--checkpoint', str(checkpoint), '--output', str(output),
                       '--seed', str(seed), '--noise', str(noise)]
            checked_process(command, runtime, run, 'evaluation')
            load_rows(output / 'steps.csv', n)
            metrics = study.read(output / 'metrics.json')
            if metrics.get('steps') != n:
                raise ValueError('Evaluation metrics horizon mismatch')
            verify_replay(output / 'replay', n)
            if campaign.get('reference_contract'):
                common = study.read(output / 'common_metrics.json')
                if common['frames'] != task['frames']:
                    raise ValueError('Common physical score is missing full-motion frames')
            files = [p for p in output.rglob('*') if p.is_file()]
            study.save_json(run / 'done.json', dict(signature=signature, status='evaluated',
                artifacts={str(p.relative_to(run)): study.digest(p) for p in files}))
        repeats.append(load_rows(output / 'steps.csv', n))
    common_scores = [study.read(folder / f'seed{seed}/rollout/common_metrics.json') for seed in seeds
                     if (folder / f'seed{seed}/rollout/common_metrics.json').exists()]
    study.save_json(folder / 'summary.json', dict(checkpoint=str(checkpoint), sha256=study.digest(checkpoint),
        seeds=seeds, noise=noise, horizon=n, results=[summarize(r, n) for r in repeats], qualified=qualifies(
            [summarize(r, n) for r in repeats]), common_metrics=common_scores))
    return repeats


def validate_checkpoint(campaign_path, job_id, checkpoint, folder):
    """Development-only diagnostics, never a continuation selection opportunity."""
    campaign_path, checkpoint, folder = map(lambda p: Path(p).resolve(), (campaign_path, checkpoint, folder))
    campaign = study.read(campaign_path)
    study.verify_campaign(campaign)
    study.check_execution_environment(campaign['runtime']['device'])
    job = next(j for j in campaign['jobs'] if j['id'] == job_id)
    task = next(t for t in campaign['tasks'] if t['task_id'] == job['task_id'])
    if not job['executable'] or job['method'] != 'warmstart' or task['split'] != 'development':
        raise ValueError('Diagnostic checkpoint validation is restricted to eligible development warmstarts')
    if folder == campaign_path.parent or folder.is_relative_to(campaign_path.parent / 'jobs'):
        raise ValueError('Diagnostic output must be separate from primary job/selection records')
    signature = dict(campaign_sha256=study.digest(campaign_path), checkpoint_sha256=study.digest(checkpoint),
                     job=job, purpose='diagnostic_only_not_selected')
    with study.duplicate_guard(folder):
        study.immutable_json(folder / 'contract.json', signature)
        if (folder / 'done.json').exists():
            receipt = study.verified_receipt(folder)
            if receipt['signature'] != signature:
                raise ValueError('Diagnostic receipt mismatch')
            return receipt
        started, before = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN)
        attempt_path = folder / 'attempts' / f'{time.time_ns()}.json'
        attempt = dict(status='running', started=started, command=sys.argv,
                       accounting_scope='diagnostic workflow total; not an arm training/selection slot')
        study.save_json(attempt_path, attempt)
        try:
            protocol = campaign['runtime']['validation_protocol']
            evaluate(campaign_path, campaign, job, checkpoint, folder / 'evaluation',
                     protocol['selection_seeds'], protocol['selection_noise'])
            files = [p for p in (folder / 'evaluation').rglob('*') if p.is_file() and p.name != 'run.log'
                     and 'attempts' not in p.parts and not any(part.startswith('incomplete_rollout_') for part in p.parts)]
            receipt = dict(signature=signature, status='diagnostic_evaluated_not_selected',
                artifacts={str(p.relative_to(folder)): study.digest(p) for p in files})
            study.save_json(folder / 'done.json', receipt)
            attempt['status'] = 'completed'
            return receipt
        except BaseException as exc:
            attempt.update(status='failed', error=repr(exc))
            raise
        finally:
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            wall = time.time() - started
            attempt.update(finished=time.time(), wall_seconds=wall,
                child_cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                assigned_gpu_wall_seconds=wall if campaign['runtime']['device'] != 'cpu' else 0,
                measured_gpu_compute_seconds=None)
            study.save_json(attempt_path, attempt)


def diagnose(campaign, task, repeats, folder):
    """Existing miner sees only pre-reset evidence from the worst validation repeat."""
    n = horizon(task)
    index = min(range(len(repeats)), key=lambda i: (summarize(repeats[i], n)['valid_steps'], i))
    rows = repeats[index]
    valid = summarize(rows, n)['valid_steps']
    # A done step may already contain reset observations. Do not mine that state.
    evidence = [{key: value for key, value in row.items()
                 if not key.startswith(('ref_anchor_', 'robot_anchor_'))} for row in rows[:valid]]
    if not evidence:
        window = tuple(failure_window(repeats, n))
        report = dict(source={'kind': 'native first-step failure flags; no pre-reset body evidence'},
            summary=dict(steps=n, done_any_count=sum(not summarize(r, n)['complete'] for r in repeats), first_done_step=1),
            failure_windows=[dict(window=window, dominant_bodies=[], dominant_channels=[],
                                  recommended_families=['general_body_tracking'], top_metrics=[])])
        study.immutable_json(folder / 'failure_report.json', report)
        context = ProposalContext.from_failure_report(report, task_id=task['task_id'])
        study.immutable_json(folder / 'context.json', context.prompt_payload())
        return context
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / 'diagnosis_steps.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=evidence[0].keys())
        writer.writeheader()
        writer.writerows(evidence)
    study.save_json(folder / 'diagnosis_metrics.json', {'done_any_count': sum(not summarize(r, n)['complete'] for r in repeats)})
    report_path = folder / 'failure_report.json'
    command = [campaign['runtime']['python'], str(study.ROOT / 'scripts/autorefine/mine_failures.py'),
               '--metrics-json', str(folder / 'diagnosis_metrics.json'), '--step-csv', str(csv_path),
               '--output-json', str(report_path), '--output-md', str(folder / 'failure_report.md')]
    checked_process(command, campaign['runtime'], folder, 'diagnosis')
    context = ProposalContext.from_failure_report(study.read(report_path), task_id=task['task_id'])
    # Use the existing repeated-rollout window rule, retaining miner body/channel diagnosis.
    failures = [summarize(r, n)['first_failure_step'] for r in repeats if not summarize(r, n)['complete']]
    context = replace(context, horizon=n, failure_window=tuple(failure_window(repeats, n)),
                      first_failure_step=min(failures) if failures else None)
    study.save_json(folder / 'context.json', context.prompt_payload())
    return context


def execute_arm(campaign_path, campaign, job, folder):
    task = next(t for t in campaign['tasks'] if t['task_id'] == job['task_id'])
    n = horizon(task)
    protocol = campaign['runtime']['validation_protocol']
    warm_job = next(j for j in campaign['jobs'] if j['id'] == job['dependencies'][0])
    warm = folder.parent / 'warmstart/final.pt'
    incumbent = warm
    incumbent_rows = evaluate(campaign_path, campaign, warm_job, warm,
        folder.parent / 'shared_validation/warmstart', protocol['selection_seeds'], protocol['selection_noise'])
    incumbent_common = study.read(folder.parent / 'shared_validation/warmstart/summary.json')['common_metrics']
    history, slots, stage_receipts = [], [], []
    if job['method'] == 'beyond_mimic':
        training = study.train_native(campaign, job, folder)
        stages = [(folder, training)]
    else:
        stages = None
    for stage_index, declaration in enumerate(job['stages']):
        if stages is None:
            stage_folder = folder / declaration['name']
            parent = warm if job['method'] == 'static_search' else incumbent
            spec_path = stage_folder / 'stage.json'
            if spec_path.exists():
                spec = study.read(spec_path)
                if spec['parent'] != str(parent) or spec['parent_sha256'] != study.digest(parent):
                    raise ValueError('Stage protected-parent contract changed')
            else:
                patch = declaration.get('patch')
                if job['method'] == 'mimicx':
                    context = diagnose(campaign, task, incumbent_rows, stage_folder / 'diagnosis')
                    tried = [study.read(folder / old['name'] / 'stage.json')['patch']
                             for old in job['stages'][:stage_index]]
                    patch = untried_patch(context, tried)
                    study.immutable_json(stage_folder / 'proposal_history.json', dict(tried=tried, selected=patch,
                        rule='First structurally untried existing bounded proposal; IDs excluded from duplicate signature'))
                spec = dict(updates=1500, parent=str(parent), parent_sha256=study.digest(parent), patch=patch)
                study.immutable_json(spec_path, spec)
            training = study.train_native(campaign, job, stage_folder, stage=spec)
            current = stage_folder
        else:
            current, training = stages[0]
        stage_receipts.append(dict(folder=str(current.relative_to(folder)), updates=training['completed_updates'],
                                   transitions=training['transitions']))
        for name in training['selection_checkpoints']:
            candidate = current / name
            slot = len(slots)
            rows = evaluate(campaign_path, campaign, job, candidate, folder / 'validation' / f'slot{slot:02d}',
                            protocol['selection_seeds'], protocol['selection_noise'])
            gate = accept(incumbent_rows, rows, n)
            gate.update(slot=slot, stage=stage_index, candidate=str(candidate), candidate_sha256=study.digest(candidate),
                        incumbent=str(incumbent), incumbent_sha256=study.digest(incumbent))
            study.immutable_json(folder / 'gates' / f'slot{slot:02d}.json', gate)
            history.append(gate)
            slots.append(str(candidate.relative_to(folder)))
            if gate['accepted']:
                incumbent, incumbent_rows = candidate, rows
                incumbent_common = study.read(folder / 'validation' / f'slot{slot:02d}' / 'summary.json')['common_metrics']
    if len(slots) != 6 or sum(s['updates'] for s in stage_receipts) != 3000:
        raise RuntimeError('Arm did not consume exactly 3000 updates and six validation opportunities')
    selection = dict(checkpoint=str(incumbent), sha256=study.digest(incumbent), history=history,
                     seeds=protocol['selection_seeds'], criterion='motion_study_gate.accept', slots=6)
    study.immutable_json(folder / 'selection.json', selection)
    held = {}
    for kind in ('nominal', 'perturbed'):
        rows = evaluate(campaign_path, campaign, job, incumbent, folder / 'holdout' / kind,
            protocol['holdout'][task['split']][kind], protocol[kind + '_noise'])
        held[kind] = [summarize(r, n) for r in rows]
    common_held = {kind: study.read(folder / 'holdout' / kind / 'summary.json')['common_metrics']
                   for kind in ('nominal', 'perturbed')}
    native_qualified = qualifies([summarize(r, n) for r in incumbent_rows]) and all(qualifies(v) for v in held.values())
    common_qualified = bool(campaign.get('reference_contract')) and len(incumbent_common) == len(protocol['selection_seeds'])
    common_qualified = common_qualified and all(m['complete_motion'] for m in incumbent_common)
    common_qualified = common_qualified and all(len(common_held[k]) == len(protocol['holdout'][task['split']][k])
                                               and all(m['complete_motion'] for m in common_held[k]) for k in held)
    report = dict(status='completed', selected_checkpoint=str(incumbent), selected_sha256=study.digest(incumbent),
                  selection=[summarize(r, n) for r in incumbent_rows], holdout=held,
                  qualified=native_qualified and common_qualified, native_qualified=native_qualified,
                  common_physical_qualified=common_qualified, common_selection_metrics=incumbent_common,
                  common_protocol=campaign['runtime']['common_protocol'],
                  common_protocol_sha256=campaign['common_protocol_sha256'],
                  common_physical_metrics=common_held,
                  metrics_contract='Existing native gate; separate common physical scorer against original world reference',
                  completed_updates=3000, transitions=sum(s['transitions'] for s in stage_receipts),
                  stages=stage_receipts, selected_warmstart=incumbent == warm)
    study.immutable_json(folder / 'report.json', report)
    if job['method'] != 'beyond_mimic':
        import shutil
        study.atomic_save(lambda path, infos=None: shutil.copy2(incumbent, path), folder / 'final.pt')
        study.immutable_json(folder / 'effective_training.json', dict(campaign=campaign, job=job, stages=stage_receipts))
    # Preserve the full budget ledger even when the protected warmstart wins.
    files = [p for p in folder.rglob('*') if p.is_file() and p != folder / 'training.json'
             and p.name not in ('.run.lock', 'done.json', 'run.log', 'contract.json') and 'attempts' not in p.parts
             and not any(part.startswith('incomplete_rollout_') for part in p.parts)]
    summary = dict(status='completed', completed_updates=3000, transitions=report['transitions'],
                   selection_checkpoints=slots, artifacts=[str(p.relative_to(folder)) for p in files],
                   stages=stage_receipts, selected_checkpoint=str(incumbent), selected_sha256=study.digest(incumbent),
                   qualified=report['qualified'], attempted_transitions_exact=None)
    study.save_json(folder / 'training.json', summary)
    return summary
