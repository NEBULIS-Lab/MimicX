"""Checkpoint discovery helpers shared by AutoRefine launchers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def select_highest_step_checkpoint(paths: Iterable[Path]) -> Path:
    candidates = list(paths)
    if not candidates:
        raise ValueError("Cannot select a checkpoint from an empty collection")
    return max(
        candidates,
        key=lambda path: (checkpoint_step(path), path.stat().st_mtime_ns, path.name),
    )
