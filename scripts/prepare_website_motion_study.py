"""Publish a portable, read-only website view of the completed motion study."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg


METHODS = {"beyond_mimic": "BeyondMimic / MjLab", "static_search": "Static Search",
           "mimicx": "MimicX", "sonic": "Released SONIC"}


def prepare(delivery, site):
    result_dir = delivery / "results/four-method-delivery-v1"
    rows = list(csv.DictReader((result_dir / "all_tasks.csv").open()))
    assert len(rows) == 96
    summary = {}
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        assert len(group) == 24
        summary[method] = dict(complete=sum(int(r["complete"]) for r in group),
                               count=sum(int(r["count"]) for r in group))
    results = site / "assets/results/motion-study"
    results.mkdir(parents=True, exist_ok=True)
    for name in ("all_tasks.csv", "native_phase_costs.csv", "four_methods.md"):
        (results / name).write_text((result_dir / name).read_text())
    media = site / "assets/media/motion-study"
    media.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for task, short in (("bones_broad_jump", "broad-jump"), ("bones_heel_toe", "heel-toe")):
        source = delivery / f"media/review-flat/CONFIRMATION_20261110__{task}__FOUR_METHOD_COMPARISON.mp4"
        original = media / f"{short}.mp4"
        shutil.copyfile(source, original)
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source),
                        "-vf", "scale=960:600", "-c:v", "libx264", "-preset", "medium",
                        "-crf", "25", "-maxrate", "800k", "-bufsize", "1600k", "-an",
                        "-movflags", "+faststart", str(media / f"{short}-preview.mp4")], check=True)
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-ss", "0.5",
                        "-i", str(source), "-frames:v", "1", "-q:v", "2",
                        str(media / f"{short}.jpg")], check=True)
    task_ids = list(dict.fromkeys(r["task_id"] for r in rows))
    task_rows = []
    for task in task_ids:
        group = [r for r in rows if r["task_id"] == task]
        values = []
        for method in METHODS:
            entries = [r for r in group if r["method"] == method]
            values.append(f'{sum(int(r["complete"]) for r in entries)}/{sum(int(r["count"]) for r in entries)}')
        task_rows.append('<tr><th scope="row">' + html.escape(group[0]["label"]) + '</th>' +
                         ''.join(f'<td>{v}</td>' for v in values) + '</tr>')
    (results / "completion-rows.html").write_text('\n'.join(task_rows) + '\n')
    page = site / "index.html"
    source = page.read_text()
    marker = '<tbody id="motion-completion-rows">'
    start = source.index(marker) + len(marker)
    end = source.index('</tbody>', start)
    page.write_text(source[:start] + '\n' + '\n'.join(task_rows) + '\n' + source[end:])
    (results / "summary.json").write_text(json.dumps(dict(
        motions=12, training_seeds=2, conditions=["nominal", "perturbed"],
        native_repeats_per_seed_condition=3, sonic_repeats_per_condition=3,
        completion=summary, media="Fixed-clock representative recordings; statistics use all evaluations."
    ), indent=2) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.delivery, args.site)
