"""Generate live, explicitly partial paper assets from matrix aggregates."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


TASK_LABELS = {
    "tennis": "Tennis",
    "football1": "Football1",
    "dance2": "Dance2",
    "kongfu1": "Kongfu1",
}
METHOD_LABELS = {
    "m0_open_loop": "M0",
    "m1_policy_window": "M1",
    "m2_task_hierarchy": "M2",
    "m3_full_mimicx": "M3",
}
METHOD_COLORS = {
    "m0_open_loop": "#6B7280",
    "m1_policy_window": "#2F6B9A",
    "m2_task_hierarchy": "#3A7D5D",
    "m3_full_mimicx": "#C46A2D",
}


@dataclass(frozen=True)
class PaperAssetReport:
    total_jobs: int
    completed_jobs: int
    pending_jobs: int
    generated_files: int
    output_dir: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric(value: str | None, *, digits: int, percent: bool = False) -> str:
    if value is None or value == "":
        return "--"
    number = float(value)
    if percent:
        number *= 100.0
    return f"{number:.{digits}f}"


def _write_latex(rows: list[dict[str, str]], output: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Task & Method & Seeds & Strict success (\%) & Reward & Body error \\",
        r"\midrule",
    ]
    for row in rows:
        completed = int(row["completed_seeds"])
        expected = int(row["expected_seeds"])
        seed_label = f"{completed}/{expected}"
        if 0 < completed < expected:
            seed_label += "*"
        lines.append(
            " & ".join(
                [
                    TASK_LABELS.get(row["task_id"], row["task_id"]),
                    METHOD_LABELS.get(row["method_id"], row["method_id"]),
                    seed_label,
                    _metric(row.get("strict_success_rate"), digits=1, percent=True),
                    _metric(row.get("reward_mean"), digits=4),
                    _metric(row.get("body_pos_error_mean"), digits=4),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% * Partial live aggregate; replace after all declared seeds complete.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    plotted = []
    for row in rows:
        completed = int(row["completed_seeds"])
        if completed == 0:
            continue
        expected = int(row["expected_seeds"])
        plotted.append(
            {
                "task_id": row["task_id"],
                "method_id": row["method_id"],
                "method_label": METHOD_LABELS.get(row["method_id"], row["method_id"]),
                "completed_seeds": str(completed),
                "expected_seeds": str(expected),
                "is_partial": str(completed < expected).lower(),
                "strict_success_rate": row["strict_success_rate"],
            }
        )
    return plotted


def _write_plot_data(rows: list[dict[str, str]], output: Path) -> None:
    fields = [
        "task_id",
        "method_id",
        "method_label",
        "completed_seeds",
        "expected_seeds",
        "is_partial",
        "strict_success_rate",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_svg(rows: list[dict[str, str]], output: Path) -> None:
    width, height = 720, 400
    left, right, top, bottom = 150, 36, 30, 58
    chart_width = width - left - right
    usable_height = height - top - bottom
    spacing = usable_height / max(len(rows), 1)
    bar_height = min(34.0, spacing * 0.62)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#1F2937">',
    ]
    for tick in range(0, 101, 20):
        x = left + chart_width * tick / 100.0
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" '
            'stroke="#D1D5DB" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" '
            f'font-size="13">{tick}</text>'
        )
    for index, row in enumerate(rows):
        y = top + spacing * (index + 0.5)
        rate = float(row["strict_success_rate"])
        bar_width = chart_width * rate
        color = METHOD_COLORS.get(row["method_id"], "#4B5563")
        partial = row["is_partial"] == "true"
        seed_label = f'{row["completed_seeds"]}/{row["expected_seeds"]}'
        if partial:
            seed_label += "*"
        label = f'{TASK_LABELS.get(row["task_id"], row["task_id"])} {row["method_label"]}'
        parts.append(
            f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="14">{label}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y-bar_height/2:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{color}" opacity="{0.72 if partial else 1.0}"/>'
        )
        parts.append(
            f'<text x="{min(left+bar_width+8, width-right-44):.1f}" y="{y+5:.1f}" '
            f'font-size="13">{rate*100:.1f} ({seed_label})</text>'
        )
    parts.extend(
        [
            f'<text x="{left+chart_width/2:.1f}" y="{height-10}" text-anchor="middle" '
            'font-size="14">Strict success (%)</text>',
            '</g>',
            '</svg>',
        ]
    )
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_live_paper_assets(
    summary_csv: Path,
    status_json: Path,
    output_dir: Path,
) -> PaperAssetReport:
    rows = _read_rows(summary_csv)
    status = json.loads(status_json.read_text(encoding="utf-8"))
    completed_from_rows = sum(int(row["completed_seeds"]) for row in rows)
    if completed_from_rows != int(status["completed_jobs"]):
        raise ValueError(
            "completed job mismatch: "
            f"summary={completed_from_rows}, status={status['completed_jobs']}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    latex_path = output_dir / "main_results_live.tex"
    plot_data_path = output_dir / "strict_success_plot_data.csv"
    svg_path = output_dir / "strict_success_live.svg"
    manifest_path = output_dir / "asset_manifest.json"
    _write_latex(rows, latex_path)
    plotted = _plot_rows(rows)
    _write_plot_data(plotted, plot_data_path)
    _write_svg(plotted, svg_path)

    report = PaperAssetReport(
        total_jobs=int(status["total_jobs"]),
        completed_jobs=int(status["completed_jobs"]),
        pending_jobs=int(status["pending_jobs"]),
        generated_files=4,
        output_dir=str(output_dir.resolve()),
    )
    manifest_path.write_text(
        json.dumps(
            {
                **asdict(report),
                "inputs": {
                    str(summary_csv.resolve()): _sha256(summary_csv),
                    str(status_json.resolve()): _sha256(status_json),
                },
                "outputs": [latex_path.name, plot_data_path.name, svg_path.name],
                "partial_rows": sum(row["is_partial"] == "true" for row in plotted),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
