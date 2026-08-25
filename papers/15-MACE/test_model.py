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
