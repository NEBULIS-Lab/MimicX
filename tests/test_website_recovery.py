import hashlib
import json
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "docs"


def test_all_tasks_show_input_baseline_and_latest_policy_without_selection():
    html = (SITE / "index.html").read_text()
    for task in ("tennis", "football", "dance", "kungfu"):
        row = re.search(rf'<article[^>]*data-policy-task="{task}".*?</article>', html, re.S)
        assert row, task
        assert row[0].count("<video ") == 5
        for media in (f"recovery/{task}-input.mp4", f"recovery/{task}-policy.mp4"):
            assert media in row[0]
        for method in ("fixed", "beyond", "sonic", "ours"):
            assert f'source-camera/{task}-{method}-ghost.mp4' in row[0]
        ours = re.search(r'<figure data-method="ours">.*?</figure>', row[0], re.S)
        assert ours, task
        for attribute, suffix in (("src", "mp4"), ("href", "mp4"), ("poster", "jpg")):
            assert f'{attribute}="assets/media/source-camera/{task}-ours-ghost.{suffix}"' in ours[0]
        assert f"{task}-policy.mp4" not in ours[0]
        assert ">Robot only</a>" in row[0]
        assert row[0].count('aria-label="MimicX"') == 1
        assert row[0].count('aria-label="Fixed Reference"') == 1
    assert 'id="baseline-select"' not in html
    assert 'role="tabpanel"' not in html
    assert 'data-comparison-task=' not in html
    assert 'Same-reference paper comparisons' not in html


def test_square_recordings_and_scene_derivatives():
    css = (SITE / "assets/css/recordings.css").read_text()
    assert 'repeat(5, minmax(0, 1fr))' in css
    assert 'aspect-ratio: 1' in css
    assert 'object-fit: cover' in css
    assert 'object-position: center bottom' in css
    html = (SITE / "index.html").read_text()
    assert 'data-crop="top-only"' in html
    rows = json.loads((SITE / 'assets/media/recording-grid/manifest.json').read_text())
    for row in rows:
        path = SITE / 'assets/media/recording-grid' / row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        if row['file'].startswith('scene-') and row['file'].endswith('.mp4'):
            assert row['width'] == row['height']
            assert row['crop']['top'] > 0
            assert row['full_decode_passed']


def test_release_manifest_and_autoplay_lifecycle():
    rows = json.loads((SITE / "assets/media/recovery/manifest.json").read_text())
    videos = [r for r in rows if r["file"].endswith(".mp4")]
    assert len(videos) == 20
    for row in rows:
        path = SITE / "assets/media/recovery" / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert "/" not in row["source"]
    for row in videos:
        assert row["full_decode_passed"]
        assert row["duration_seconds"] > 0
    js = (SITE / "assets/js/recordings.js").read_text()
    for guard in ("IntersectionObserver", "visibilitychange", "prefers-reduced-motion", "mimicx-media-open", "userPaused", "dialog.open"):
        assert guard in js
    assert "pointerenter" not in js


def test_source_camera_release_matches_each_input_clock():
    rows = json.loads((SITE / 'assets/media/source-camera/manifest.json').read_text())
    assert len(rows) == 16
    for task in ('tennis', 'football', 'dance', 'kungfu'):
        clips = [r for r in rows if r['task'] == task]
        assert {r['method'] for r in clips} == {'fixed', 'beyond', 'sonic', 'ours'}
        assert len({r['source_calibration_sha256'] for r in clips}) == 1
        assert len({r['ghost_color'] for r in clips}) == 4
        for clip in clips:
            assert clip['schema'] == 'mimicx.source-camera-replay.v2'
            assert clip['display_revision'] == 'stable-camera-robot-first-v1'
            assert clip['ghost_alpha'] == .23
            assert clip['protected_pixel_max_delta'] == 0
            if task != 'kungfu':
                assert clip['camera_stability']['new_step_max_cm'] == 0
            assert clip['full_decode_passed']
            assert clip['decoded_frames'] == clip['source_frames']
            assert clip['encoded_fps'] == clip['source_fps']
            assert clip['retime_factor'] == 1
            assert max(clip['endpoint_hold_seconds']) <= .04
            path = SITE / 'assets/media/source-camera' / clip['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == clip['sha256']
            assert '/data/' not in json.dumps(clip)
    js = (SITE / 'assets/js/recordings.js').read_text()
    assert 'syncRow' in js
    assert "'seeking'" in js
