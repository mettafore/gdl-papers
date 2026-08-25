"""Loss, metric, and benchmark contracts for the MACE reproduction."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

import evaluate
import train


def test_reference_table_matches_approved_target() -> None:
    assert evaluate.REFERENCE_RMSE_MEV == {
        "300K": {"energy": 3.0, "forces": 8.8},
        "600K": {"energy": 9.7, "forces": 21.8},
        "1200K": {"energy": 29.8, "forces": 62.0},
        "dihedral": {"energy": 7.8, "forces": 16.5},
    }


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="TODO(luv): energy-force loss convention",
)
def test_energy_force_loss_uses_published_weights() -> None:
    predicted_energy = torch.tensor([1.0, 3.0])
    target_energy = torch.tensor([1.0, 1.0])
    predicted_forces = torch.tensor([[1.0, 0.0, 0.0]])
    target_forces = torch.zeros((1, 3))
    loss = train.energy_force_loss(
        predicted_energy,
        predicted_forces,
        target_energy,
        target_forces,
        energy_weight=1.0,
        force_weight=1000.0,
    )
    expected_energy_mse = 2.0
    expected_force_mse = 1.0 / 3.0
    assert loss.item() == pytest.approx(
        expected_energy_mse + 1000.0 * expected_force_mse
    )


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="TODO(luv): component RMSE"
)
def test_rmse_uses_components_not_vector_norms() -> None:
    prediction = torch.tensor([[3.0, 0.0, 0.0]])
    target = torch.zeros((1, 3))
    assert evaluate.root_mean_square_error(prediction, target) == pytest.approx(3**0.5)
