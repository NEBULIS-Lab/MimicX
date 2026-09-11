"""Checkpoint mapping and curve summaries for policy convergence evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MilestoneCheckpoint:
    percent: int
    step: int
    path: Path


def _step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"checkpoint name lacks integer step: {path}") from exc


def select_milestone_checkpoints(
    checkpoints: Sequence[Path], milestones: Sequence[int]
) -> list[MilestoneCheckpoint]:
    if not checkpoints:
        raise ValueError("checkpoint list is empty")
    steps = [_step(path) for path in checkpoints]
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("checkpoint steps must be strictly increasing")
    base = steps[0] - 1
    final = steps[-1]
    selected = []
    for percent in milestones:
        if not 0 < int(percent) <= 100:
            raise ValueError(f"invalid milestone: {percent}")
        target = base + (final - base) * int(percent) / 100.0
        index = min(range(len(steps)), key=lambda item: (abs(steps[item] - target), steps[item]))
        selected.append(MilestoneCheckpoint(int(percent), steps[index], checkpoints[index]))
    if len({item.step for item in selected}) != len(selected):
        raise ValueError("milestones collapse onto duplicate checkpoints")
    return selected


def constant_milestone_checkpoints(
    checkpoint: Path, milestones: Sequence[int]
) -> list[MilestoneCheckpoint]:
    """Represent an incumbent-protected rollback with no accepted policy update."""
    step = _step(checkpoint)
    return [MilestoneCheckpoint(int(percent), step, checkpoint) for percent in milestones]


def normalized_auc(x_percent: Sequence[float], values: Sequence[float]) -> float:
    if len(x_percent) != len(values) or len(values) < 2:
        raise ValueError("AUC requires matching arrays with at least two points")
    x = np.asarray(x_percent, dtype=np.float64) / 100.0
    y = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
        raise ValueError("AUC inputs must be finite with increasing milestones")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def threshold_milestone(
    x_percent: Sequence[int], values: Sequence[float], threshold: float
) -> int | None:
    for percent, value in zip(x_percent, values):
        if value >= threshold:
            return int(percent)
    return None
