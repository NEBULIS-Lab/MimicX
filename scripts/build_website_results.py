#!/usr/bin/env python3
"""Render website-owned CSV evidence into static HTML; no browser fetch needed."""

import csv
from html import escape
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"tennis": "Tennis Swing", "football1": "Football Juggling",
         "dance2": "Dance Sequence", "kongfu1": "Kung Fu Sequence"}
METHODS = {"m0_open_loop": "Fixed Reference", "m3_full_mimicx": "MimicX"}


def build(page, results):
    with (results / "core_method_task_summary.csv").open(newline="") as handle:
        data = {(row["task_id"], row["method_id"]): row for row in csv.DictReader(handle)}
    rows = []
    for task, label in TASKS.items():
        for method, name in METHODS.items():
            row = data[task, method]
            cls = ' class="ours-row"' if method == "m3_full_mimicx" else ""
            rows.append(f'<tr{cls}><th scope="row">{escape(label)}</th><td>{name}</td>'
                        f'<td>{float(row["strict_success_rate"]):.1%}</td>'
                        f'<td>{float(row["worst_first_failure_step_mean"]):.1f}</td>'
                        f'<td>{float(row["body_pos_error_mean"]):.3f} m</td></tr>')
    html = page.read_text()
    html, count = re.subn(r'(<tbody id="core-results">).*?(</tbody>)',
                         lambda m: m[1] + "\n" + "\n".join(rows) + "\n" + m[2], html, flags=re.S)
    assert count == 1, "Expected exactly one results table"
    links = [f'<a href="assets/results/{escape(path.name)}">{escape(path.stem.replace("_", " "))} (CSV)</a>'
             for path in sorted(results.glob("*.csv"))]
    html, count = re.subn(r'(<div id="result-downloads">).*?(</div>)',
                         lambda m: m[1] + "\n" + "\n".join(links) + "\n" + m[2], html, flags=re.S)
    assert count == 1, "Expected exactly one downloads block"
    page.write_text(html)


if __name__ == "__main__":
    build(ROOT / "docs/index.html", ROOT / "docs/assets/results")
