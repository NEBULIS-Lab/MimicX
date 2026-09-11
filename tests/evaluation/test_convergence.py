from pathlib import Path

import pytest

from mimicx.evaluation.convergence import (
    constant_milestone_checkpoints,
    normalized_auc,
    select_milestone_checkpoints,
    threshold_milestone,
)


def test_selects_nearest_actual_checkpoints() -> None:
    paths = [Path(f"model_{step}.pt") for step in (1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1248)]
    selected = select_milestone_checkpoints(paths, [20, 40, 60, 80, 100])
    assert [item.step for item in selected] == [1050, 1100, 1150, 1200, 1248]
    assert [item.percent for item in selected] == [20, 40, 60, 80, 100]


def test_normalized_auc_and_threshold() -> None:
    assert normalized_auc([20, 40, 60, 80, 100], [0, 0.25, 0.5, 0.75, 1.0]) == pytest.approx(0.5)
    assert threshold_milestone([20, 40, 60], [0.2, 0.91, 1.0], 0.9) == 40
    assert threshold_milestone([20, 40], [0.1, 0.2], 0.9) is None


def test_rejects_nonmonotonic_checkpoint_steps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        select_milestone_checkpoints([Path("model_10.pt"), Path("model_5.pt")], [50])


def test_protected_rollback_has_explicit_constant_trajectory() -> None:
    points = constant_milestone_checkpoints(Path("model_2318.pt"), [20, 40, 60, 80, 100])
    assert [point.percent for point in points] == [20, 40, 60, 80, 100]
    assert {point.step for point in points} == {2318}
