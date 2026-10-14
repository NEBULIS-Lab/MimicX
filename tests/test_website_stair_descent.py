"""The sixth scene retains an independently verified input/execution pair."""
import hashlib
import json
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / 'docs'


def test_only_descent_input_uses_full_window_crop():
    html = (SITE / 'index.html').read_text()
    row = re.search(r'<article[^>]*data-policy-task="descent".*?</article>', html, re.S).group()
    input_figure = re.search(r'<figure data-method="input".*?</figure>', row, re.S).group()
    assert 'data-crop="descent"' in input_figure
    assert html.count('data-crop="descent"') == 1
    css = (SITE / 'assets/css/recordings.css').read_text()
    assert '.recording-frame[data-crop="descent"] video { object-fit: cover;' in css


def test_six_scene_tasks_fill_three_desktop_rows():
    html = (SITE / 'index.html').read_text()
    rows = re.findall(r'<article class="recording-row terrain-row".*?</article>', html, re.S)
    assert len(rows) == 6
    assert 'data-policy-task="descent"' in rows[-1]
    assert 'data-policy-task="parkour"' in rows[-2]
    assert 'MimicX + MPC' in rows[-2]
    row = rows[-1]
    assert row.count('<video ') == 2
    assert 'data-sync="independent"' in row
    assert re.findall(r'<figure data-method="([^"]+)"', row) == ['input', 'ours']
    assert 'Fixed Reference' not in row
    for role in ('input', 'ours', 'robot'):
        assert f'href="assets/media/terrain-release/descent-{role}.mp4" data-viewer="video"' in row
    for role in ('input', 'ours'):
        assert f'poster="assets/media/terrain-release/descent-{role}.jpg"' in row
    assert 'Six human inputs' in html


def test_descent_artifacts_are_complete_and_provenanced():
    media = SITE / 'assets/media/terrain-release'
    manifest = json.loads((media / 'descent-manifest.json').read_text())
    assert manifest['updates_paper_metrics'] is False
    assert manifest['updates_published_weights'] is False
    assert manifest['pose_modification'] is False
    assert manifest['root_alignment'] is False
    assert manifest['camera']['static_camera']
    assert manifest['selection']['accepted']
    assert manifest['selection']['candidate_frontiers'] == [240] * 3
    assert [x['seed'] for x in manifest['holdouts']] == [4001, 5002, 6003]
    assert all(x['complete'] and x['quality_passed'] for x in manifest['holdouts'])
    assert manifest['input_source']['retimed'] is False
    assert manifest['input_source']['source_frames'] == [0, 145]
    for item in manifest['media']:
        path = media / item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        if path.suffix == '.mp4':
            assert item['full_decode_passed']
            if item['role'] != 'input':
                assert (item['frames'], item['fps']) == (240, 50)
                assert (item['width'], item['height']) == (1920, 1080)
    raw = json.dumps(manifest)
    for local_detail in ('/data/', '/home/', 'SLURM', 'shuaijun', 'ningxinsu'):
        assert local_detail not in raw
