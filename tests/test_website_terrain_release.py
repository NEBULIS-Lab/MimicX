import hashlib
import json
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / 'docs'


def test_five_scene_tasks_pair_input_with_mimicx_and_label_parkour_mpc():
    html = (SITE / 'index.html').read_text()
    scenes = html.split('<div class="rx-scene-recordings">', 1)[1].split('<section id="overview"', 1)[0]
    assert 'Fixed Reference' not in scenes
    assert 'class="terrain-grid"' in scenes
    for task in ('track', 'stairs', 'forest', 'bench', 'parkour'):
        row = re.search(rf'<article[^>]*data-policy-task="{task}".*?</article>', html, re.S)
        assert row, task
        assert row[0].count('<video ') == 2
        assert 'data-sync="independent"' in row[0]
        assert re.findall(r'<figure data-method="([^"]+)"', row[0]) == ['input', 'ours']
        assert f'src="assets/media/terrain-release/{task}-input.mp4"' in row[0]
        if task == 'parkour':
            assert 'src="assets/media/terrain-release/parkour-mpc.mp4"' in row[0]
            assert 'poster="assets/media/terrain-release/parkour-mpc.jpg"' in row[0]
            assert 'aria-label="MimicX + MPC"' in row[0]
            assert 'robot joint control' in row[0]
            assert 'diagnostic control demonstration' in row[0]
            assert 'parkour-robot.mp4' not in row[0]
        else:
            assert f'src="assets/media/terrain-release/{task}-ours.mp4"' in row[0]
            assert 'aria-label="MimicX"' in row[0]
            assert 'Robot only' in row[0]
    assert '5.38 s' in html and 'locally retimed' in html
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


def test_terrain_pairs_form_four_desktop_videos_with_uncropped_16_9_frames():
    css = (SITE / 'assets/css/recordings.css').read_text()
    # Two task columns, each with two videos, become one task on mobile.
    assert '.terrain-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));' in css
    assert '.terrain-row .recording-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); width: 100%; }' in css
    assert '.terrain-row .recording-frame { aspect-ratio: 16 / 9; }' in css
    assert '.terrain-row .recording-grid figure:first-child { grid-column: auto; width: 100%; }' in css
    mobile = css.split('@media (max-width: 900px)', 1)[1].split('@media', 1)[0]
    assert '.terrain-grid { grid-template-columns: minmax(0, 1fr); }' in mobile
    assert '#research .terrain-row .recording-frame video { object-fit: contain; }' in css
    html = (SITE / 'index.html').read_text()
    for task in ('track', 'stairs', 'forest', 'bench', 'parkour'):
        row = re.search(rf'<article[^>]*data-policy-task="{task}".*?</article>', html, re.S)[0]
        execution = 'parkour-mpc' if task == 'parkour' else f'{task}-ours'
        for stem in (f'{task}-input', execution):
            assert f'href="assets/media/terrain-release/{stem}.mp4" data-viewer="video"' in row
        assert f'{task}-fixed.mp4' not in row
