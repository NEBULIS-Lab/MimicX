import hashlib
import json
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / 'docs'


def test_five_scene_tasks_include_sources_and_recorded_comparisons():
    html = (SITE / 'index.html').read_text()
    for task in ('track', 'stairs', 'forest', 'bench', 'parkour'):
        row = re.search(rf'<article[^>]*data-policy-task="{task}".*?</article>', html, re.S)
        assert row, task
        assert row[0].count('<video ') == 3
        assert 'data-sync="independent"' in row[0]
        for method in ('input', 'fixed', 'ours'):
            assert f'data-method="{method}"' in row[0]
        assert 'Robot only' in row[0]
    assert '5.38 s' in html and 'locally retimed' in html
    assert 'historical' in html.lower()
    js = (SITE / 'assets/js/recordings.js').read_text()
    assert "row.element.dataset.sync === 'independent'" in js


def test_terrain_media_matches_frozen_selected_policies():
    directory = SITE / 'assets/media/terrain-release'
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert len(manifest['tasks']) == 5
    assert manifest['updates_paper_metrics'] is False
    for task in manifest['tasks']:
        assert task['holdout_completed'] == task['holdout_count'] == 3
        assert len(task['checkpoint_sha256']) == 64
        for media in task['media']:
            path = directory / media['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == media['sha256']
            if path.suffix == '.mp4':
                assert media['full_decode_passed']
                assert media['duration_seconds'] > 0
    parkour = next(t for t in manifest['tasks'] if t['case'] == 'parkour')
    assert parkour['horizon'] == 269
    assert parkour['policy_duration_seconds'] == 5.38
    assert parkour['original_speed_completed'] == 0
    assert parkour['tail_candidates_accepted'] == 0
    serialized = json.dumps(manifest)
    assert '/data/' not in serialized and '/home/' not in serialized


def test_terrain_media_does_not_crop_scene_or_force_source_timing():
    css = (SITE / 'assets/css/recordings.css').read_text()
    assert '.terrain-row .recording-frame video' in css
    assert 'object-fit: contain' in css
