"""Published motion evidence is additive and agrees with its downloadable data."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "docs"


def test_motion_summary_matches_all_tasks_and_denominators():
    directory = ROOT / "assets/results/motion-study"
    rows = list(csv.DictReader((directory / "all_tasks.csv").open()))
    summary = json.loads((directory / "summary.json").read_text())
    assert len(rows) == 96
    assert len({r["task_id"] for r in rows}) == 12
    for method, expected in summary["completion"].items():
        group = [r for r in rows if r["method"] == method]
        assert sum(int(r["complete"]) for r in group) == expected["complete"]
        assert sum(int(r["count"]) for r in group) == expected["count"]
    source = (ROOT / "index.html").read_text()
    table = source.split('<tbody id="motion-completion-rows">')[1].split('</tbody>')[0]
    assert table.strip() == (directory / "completion-rows.html").read_text().strip()
    assert table.count('<tr>') == 12


def test_new_videos_have_previews_originals_and_posters():
    from scripts.prepare_website_previews import fast_start, metadata
    for task in ("broad-jump", "heel-toe"):
        directory = ROOT / "assets/media/motion-study"
        original, preview = [directory / f'{task}{suffix}.mp4' for suffix in ('', '-preview')]
        assert (directory / f'{task}.jpg').is_file()
        assert fast_start(preview)
        a, b = metadata(original), metadata(preview)
        assert abs(a['duration'] - b['duration']) < .025
        assert a['fps'] == b['fps'] == 50
        assert b['size'] == (960, 600)
