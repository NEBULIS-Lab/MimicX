"""Prospective reduced confirmation freeze; no proposal or gate implementation."""
import copy
import hashlib
import json
from pathlib import Path


CONFIRMATION_STRATEGY = 'joint_adaptive_precision_confirmation_v1'
PROTOCOL_VERSION = 'mimicx.confirmation-12x2-3plus3.v1'
SCHEMA = 'mimicx.confirmation-freeze.v1'
TRAINING_SEEDS = [101, 202]
SELECTION_SEEDS = [1001, 1002, 1003]
HOLDOUT_SEEDS = {'nominal': [21001, 21002, 21003], 'perturbed': [22001, 22002, 22003]}
BUDGET = dict(shared_warmstart=3000, per_arm=3000, selection_slots=6, selection_interval=500)
CONTRACT_FIELDS = ('confirmation_contract', 'confirmation_contract_sha256')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def confirmation_validation_protocol(raw):
    """Version only the reduced evaluation route; require parent-supplied holdout."""
    p = copy.deepcopy(raw or {})
    p.setdefault('protocol_version', PROTOCOL_VERSION)
    p.setdefault('selection_seeds', SELECTION_SEEDS.copy())
    for key, value in (('selection_noise', .005), ('nominal_noise', .005), ('perturbed_noise', .01)):
        p.setdefault(key, value)
        if type(p[key]) not in (int, float) or p[key] != value:
            raise ValueError(f'Confirmation requires unchanged {key}={value}')
    if p['protocol_version'] != PROTOCOL_VERSION:
        raise ValueError('Confirmation protocol version mismatch')
    if _json(p['selection_seeds']) != _json(SELECTION_SEEDS):
        raise ValueError('Confirmation selection seeds must be 1001/1002/1003')
    if _json(p.get('holdout', {}).get('confirmation')) != _json(HOLDOUT_SEEDS):
        raise ValueError('Confirmation holdout must be the approved disjoint 3+3 seeds')
    for split, groups in p['holdout'].items():
        if split not in ('development', 'confirmation') or set(groups) != {'nominal', 'perturbed'}:
            raise ValueError('Unknown confirmation protocol split or holdout group')
        seeds = p['selection_seeds'] + groups['nominal'] + groups['perturbed']
        if (any(type(s) is not int or s < 0 for s in seeds)
                or len(seeds) != len(set(seeds))
                or any(len(groups[k]) < 3 for k in ('nominal', 'perturbed'))):
            raise ValueError('Validation and holdout seeds must be unique and disjoint')
    return p


def _record(campaign):
    """Validate approved scope, then bind the complete normalized execution plan."""
    from . import motion_study as study
    from .motion_study_scoring import common_protocol

    runtime, tasks = campaign['runtime'], campaign['tasks']
    if runtime.get('refinement_strategy') != CONFIRMATION_STRATEGY:
        raise ValueError('Confirmation requires its explicit frozen strategy')
    if (len(tasks) != 12 or len({t['task_id'] for t in tasks}) != 12
            or any(t['split'] != 'confirmation' or t.get('training_eligible') is not True for t in tasks)):
        raise ValueError('Confirmation requires exactly 12 unique eligible confirmation tasks')
    if _json(runtime.get('seeds', {}).get('confirmation')) != _json(TRAINING_SEEDS):
        raise ValueError('Confirmation training seeds must be 101/202')
    if runtime.get('backends', ['mjlab']) != ['mjlab']:
        raise ValueError('Reduced confirmation supports exactly the mjlab backend')
    protocol = confirmation_validation_protocol(runtime['validation_protocol'])
    if protocol != runtime['validation_protocol']:
        raise ValueError('Confirmation validation protocol must be normalized before freezing')
    if _json(campaign['budget']) != _json(BUDGET):
        raise ValueError('Confirmation budget must remain 3000/3000 updates and six slots')
    if _json(campaign['jobs']) != _json(study.make_jobs(tasks, runtime)):
        raise ValueError('Confirmation jobs differ from the approved budgets, seeds or stages')
    canonical = campaign.get('reference_contract')
    if not isinstance(canonical, dict):
        raise ValueError('Confirmation requires canonical reference capture')
    for key, count in (('joint_names', 29), ('body_names', 30)):
        names = canonical.get(key, [])
        if (not isinstance(names, list) or len(names) != count
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != count):
            raise ValueError(f'Confirmation requires full canonical {key}')
    if study.digest(runtime['common_model_xml']) != canonical.get('xml_sha256'):
        raise ValueError('Confirmation canonical reference XML changed')
    physical = runtime['common_protocol']
    if (any(physical.get(k) != v for k, v in common_protocol().items())
            or _sha(physical) != campaign['common_protocol_sha256']):
        raise ValueError('Confirmation common physical protocol changed')
    # Hash bytes, never AST dumps: 3.11 and 3.12 must agree for identical files.
    if not campaign['sources'] or campaign['sources'] != study.source_identity(runtime):
        raise ValueError('Confirmation frozen source identity changed')
    payload = copy.deepcopy(campaign)
    for key in CONTRACT_FIELDS:
        payload['runtime'].pop(key, None)
    return dict(schema=SCHEMA, strategy=CONFIRMATION_STRATEGY,
                algorithm=study.V6_STRATEGY, gate=study.V4_STRATEGY,
                task_ids=[t['task_id'] for t in tasks], training_seeds=TRAINING_SEEDS.copy(),
                validation_protocol=protocol, budget=copy.deepcopy(BUDGET),
                reference_contract=copy.deepcopy(canonical),
                common_protocol_sha256=campaign['common_protocol_sha256'],
                manifest_sha256=campaign['manifest_sha256'], sources=dict(campaign['sources']),
                campaign_sha256=_sha(payload))


def build_confirmation_contract(manifest, runtime_path, output=None):
    """CPU-only parent operation, once final deployment paths and code are fixed."""
    from . import motion_study as study
    campaign = study._build_campaign(manifest, runtime_path)
    record = _record(campaign)
    if output is not None:
        study.immutable_json(Path(output).expanduser().resolve(), record)
    return record


def verify_confirmation_contract(campaign):
    """Require an external, hash-pinned immutable record and its exact payload."""
    from . import motion_study as study
    runtime = campaign['runtime']
    if not all(runtime.get(k) for k in CONTRACT_FIELDS):
        raise ValueError('Confirmation requires confirmation_contract and confirmation_contract_sha256')
    path = Path(runtime['confirmation_contract'])
    if not path.is_absolute():
        raise ValueError('Confirmation contract path must be normalized')
    if study.digest(path) != runtime['confirmation_contract_sha256']:
        raise ValueError('Confirmation contract file hash mismatch')
    if _json(study.read(path)) != _json(_record(campaign)):
        raise ValueError('Confirmation frozen contract does not match campaign')
