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
        assert row[0].count("<video ") == 3
        for media in (f"recovery/{task}-input.mp4", f"{task}-fixed.mp4", f"recovery/{task}-policy.mp4", f"recovery/{task}-ghost.mp4"):
            assert media in row[0]
    assert 'id="baseline-select"' not in html
    assert 'role="tabpanel"' not in html
    assert html.count('data-comparison-task=') == 2
    for task in ("tennis", "football"):
        for method in ("fixed", "beyond", "sonic", "ours"):
            assert f'cases/direct-{task}-{method}.mp4' in html


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
