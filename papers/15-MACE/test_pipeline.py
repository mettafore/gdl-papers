"""Loss, metric, and benchmark contracts for the MACE reproduction."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

import evaluate
import train

import data


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
    raises=NotImplementedError, strict=True, reason="TODO(luv): 3BPA data scaling"
)
def test_data_statistics_use_residual_energy_per_atom_and_force_components() -> None:
    first = data.AtomicGraph(
        atomic_numbers=torch.tensor([1, 6]),
        node_attrs=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        positions=torch.zeros((2, 3), dtype=torch.float64),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        shifts=torch.zeros((2, 3), dtype=torch.float64),
        energy=torch.tensor(34.0, dtype=torch.float64),
        forces=torch.ones((2, 3), dtype=torch.float64),
    )
    second = data.AtomicGraph(
        atomic_numbers=torch.tensor([8]),
        node_attrs=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64),
        positions=torch.zeros((1, 3), dtype=torch.float64),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        shifts=torch.empty((0, 3), dtype=torch.float64),
        energy=torch.tensor(43.0, dtype=torch.float64),
        forces=torch.full((1, 3), 3.0, dtype=torch.float64),
    )

    mean, rms = train.compute_mean_rms_energy_forces(
        [first, second],
        atomic_energies=torch.tensor([10.0, 20.0, 30.0, 40.0]),
    )

    assert mean == pytest.approx(2.5)
    assert rms == pytest.approx((11.0 / 3.0) ** 0.5)


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="TODO(luv): neighbour normalization"
)
def test_average_num_neighbors_uses_receiver_counts() -> None:
    graphs = [
        data.AtomicGraph(
            positions=torch.zeros((3, 3)),
            edge_index=torch.tensor([[0, 1, 2], [1, 1, 0]], dtype=torch.long),
        ),
        data.AtomicGraph(
            positions=torch.zeros((2, 3)),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        ),
    ]

    assert train.compute_average_num_neighbors(graphs) == pytest.approx(1.25)


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="TODO(luv): component RMSE"
)
def test_rmse_uses_components_not_vector_norms() -> None:
    prediction = torch.tensor([[3.0, 0.0, 0.0]])
    target = torch.zeros((1, 3))
    assert evaluate.root_mean_square_error(prediction, target) == pytest.approx(3**0.5)
