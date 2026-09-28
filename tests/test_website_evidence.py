import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

SITE = Path(__file__).resolve().parents[1] / 'docs'


def test_themed_figures_and_sources():
    folder = SITE / 'assets/media/evidence'
    records = json.loads((folder/'manifest.json').read_text())
    assert len(records) == 10
    for row in records:
        path = folder / row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        doc = ET.fromstring(path.read_text())
        assert len(doc.findall('.//{http://www.w3.org/2000/svg}text')) > 5
        for name in row['sources']:
            assert (SITE/'assets/results'/name).is_file()


def test_current_evidence_preserves_cohorts():
    results = SITE/'assets/results'
    direct = list(csv.DictReader((results/'direct_policy_comparison.csv').open()))
    assert len(direct) == 42
    tennis = [r for r in direct if r['task']=='tennis']
    assert len(tennis)==12
    assert len({r['reference_sha256'] for r in tennis})==1
    assert {r['frames'] for r in tennis}=={'518'}
    assert all(not r['source'].startswith('/') for r in direct)
    collision = list(csv.DictReader((results/'collision_scene_results.csv').open()))
    assert len(collision)==5
    assert sum(int(r['fixed_completion']) for r in collision)==9
    assert sum(int(r['refined_completion']) for r in collision)==12
    assert (results/'archive/sonic_comparison.csv').is_file()
