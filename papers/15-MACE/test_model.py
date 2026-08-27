"""Small contracts for the learning-owned MACE tensor machinery."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

import model


def test_published_model_defaults_distinguish_output_and_internal_ell() -> None:
    config = model.MACEConfig()
    assert config.max_ell == 2
    assert config.spherical_harmonics_max_ell == 3
    assert config.correlation == 3
    assert config.num_interactions == 2


def test_edge_geometry_uses_paper_sender_receiver_direction() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=torch.float64)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    vectors, lengths = model.edge_vectors_and_lengths(positions, edge_index)

    assert torch.equal(
        vectors,
        torch.tensor([[2.0, 1.0, 0.0], [-2.0, -1.0, 0.0]], dtype=torch.float64),
    )
    assert torch.allclose(lengths, torch.full((2, 1), 5.0**0.5, dtype=torch.float64))


def test_force_loss_backpropagates_through_force_derivative() -> None:
    scale = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    _, lengths = model.edge_vectors_and_lengths(positions, edge_index)
    energy = scale * lengths.square().sum()

    forces = model.forces_from_energy(energy.reshape(1), positions, training=True)
    forces.square().sum().backward()

    assert scale.grad is not None
    assert torch.isfinite(scale.grad)
    assert scale.grad.abs() > 0


def test_position_independent_energy_has_zero_forces() -> None:
    scale = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    positions = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)

    forces = model.forces_from_energy(scale.reshape(1), positions, training=True)

    assert torch.equal(forces, torch.zeros_like(positions))


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="TODO(luv): first tensor-product route exercise",
)
def test_route_enumeration_1_times_2() -> None:
    routes = model.enumerate_coupling_routes((1, 2), output_max_ell=3, correlation=2)
    assert {route.output_ell for route in routes} == {1, 2, 3}


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="TODO(luv): radial basis"
)
def test_radial_basis_shape_is_finite_and_zero_at_cutoff() -> None:
    basis = model.BesselRadialBasis(num_basis=8, cutoff=5.0, power=5)
    distances = torch.tensor([[0.0], [1.25], [5.0]], dtype=torch.float64)
    output = basis(distances)
    assert output.shape == (3, 8)
    assert torch.isfinite(output).all()
    assert torch.equal(output[-1], torch.zeros(8, dtype=torch.float64))


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="TODO(luv): scalar energy and autograd force readout",
)
def test_model_returns_graph_energies_and_atom_forces() -> None:
    mace = model.MACE(model.MACEConfig(channels=4))
    atomic_numbers = torch.tensor([1, 6, 8])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long
    )
    graph_index = torch.zeros(3, dtype=torch.long)
    energy, forces = mace(atomic_numbers, positions, edge_index, graph_index)
    assert energy.shape == (1,)
    assert forces.shape == (3, 3)
    assert energy.dtype == torch.float64
    assert forces.dtype == torch.float64
