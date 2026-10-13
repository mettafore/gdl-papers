"""Small contracts for the learning-owned MACE tensor machinery."""

import sys
from pathlib import Path

import pytest
import torch
from e3nn import o3

sys.path.insert(0, str(Path(__file__).parent / "src"))

import data
import model


def test_published_model_defaults_distinguish_output_and_internal_ell() -> None:
    config = model.MACEConfig()
    assert config.max_ell == 2
    assert config.spherical_harmonics_max_ell == 3
    assert config.correlation == 3
    assert config.num_interactions == 2


@pytest.mark.parametrize(
    "override",
    [
        {"num_interactions": 1},
        {"correlation": 2},
        {"max_ell": 1},
        {"spherical_harmonics_max_ell": 2},
    ],
)
def test_config_rejects_architectures_the_model_does_not_implement(
    override: dict,
) -> None:
    with pytest.raises(ValueError):
        model.MACEConfig(**override)


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


def test_route_enumeration_1_times_2() -> None:
    routes = model.enumerate_coupling_routes((1, 2), output_max_ell=3, correlation=2)
    assert {route.output_ell for route in routes} == {1, 2, 3}


def test_route_enumeration_nu3_retains_each_intermediate_path() -> None:
    routes = model.enumerate_coupling_routes((1, 1, 1), output_max_ell=3, correlation=3)

    assert {(route.intermediate_ells, route.output_ell) for route in routes} == {
        ((0,), 1),
        ((1,), 0),
        ((1,), 1),
        ((1,), 2),
        ((2,), 1),
        ((2,), 2),
        ((2,), 3),
    }


def test_radial_basis_shape_is_finite_and_zero_at_cutoff() -> None:
    basis = model.BesselRadialBasis(num_basis=8, cutoff=5.0, power=5)
    distances = torch.tensor([[0.0], [1.25], [5.0]], dtype=torch.float64)
    output = basis(distances)
    assert output.shape == (3, 8)
    assert torch.isfinite(output).all()
    assert torch.equal(output[-1], torch.zeros(8, dtype=torch.float64))


def test_atomic_basis_uses_configured_number_of_radial_features() -> None:
    config = model.MACEConfig(
        channels=2,
        num_radial_basis=3,
        radial_mlp_widths=(4, 8),
    )
    atomic_basis = model.AtomicBasis(config, avg_num_neighbors=2.0)
    node_features = torch.randn(2, config.channels)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_vectors = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    edge_radial = torch.randn(2, config.num_radial_basis)

    output = atomic_basis(
        node_features,
        torch.eye(config.num_elements)[:2],
        edge_index,
        edge_vectors,
        edge_radial,
    )

    assert output.shape == (2, config.channels * 16)


def test_u_matrix_real_builds_nonzero_nu3_paths() -> None:
    """Eq. 10 U contains one fixed path axis for each legal CG route."""
    coupling_irreps = o3.Irreps("0e + 1o")

    u_matrix = model.u_matrix_real(
        coupling_irreps,
        o3.Irrep("0e"),
        correlation=3,
        dtype=torch.float64,
    )

    assert u_matrix.shape[:4] == (1, 4, 4, 4)
    assert u_matrix.shape[-1] > 0
    assert torch.isfinite(u_matrix).all()
    assert torch.count_nonzero(u_matrix) > 0


def test_model_matches_pinned_author_energy_and_force_fixture() -> None:
    """Reference: ACEsuit/mace b3e3b2c, all trainable weights fixed to 0.2."""
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        config = model.MACEConfig(channels=2, radial_mlp_widths=(64, 64, 64, 8))
        mace = model.MACE(
            config,
            avg_num_neighbours=2.0,
            atomic_energies=torch.tensor([0.1, 0.2, 0.3, 0.4]),
            mean_energy_shift=-0.2,
            rms_forces=1.3,
        ).double()
    finally:
        torch.set_default_dtype(previous_dtype)
    with torch.no_grad():
        for parameter in mace.parameters():
            parameter.fill_(0.2)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.1],
            [0.1, 1.1, 0.3],
            [0.0, 0.0, 0.0],
            [1.2, 0.3, 0.2],
        ],
        dtype=torch.float64,
    )
    attrs = torch.eye(4, dtype=torch.float64)[[0, 1, 3, 2, 0]]
    edges = torch.tensor([[0, 1, 0, 2, 1, 2, 3, 4], [1, 0, 2, 0, 2, 1, 4, 3]])
    energy, forces = mace(attrs, positions, edges, torch.tensor([0, 0, 0, 1, 1]))
    torch.testing.assert_close(
        energy,
        torch.tensor(
            [0.10000123909869263, 1.6880516429118586e-08], dtype=torch.float64
        ),
        atol=1e-12,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        forces,
        torch.tensor(
            [
                [
                    -2.0430371803798855e-05,
                    -7.070092104246348e-06,
                    -2.844301204093874e-06,
                ],
                [1.487926163618328e-05, 9.305627001321285e-06, 3.1875878816595703e-06],
                [
                    5.551110167615576e-06,
                    -2.235534897074936e-06,
                    -3.4328667756569685e-07,
                ],
                [5.480161432909597e-06, 1.3700403582273992e-06, 9.133602388182662e-07],
                [
                    -5.480161432909597e-06,
                    -1.3700403582273992e-06,
                    -9.133602388182662e-07,
                ],
            ],
            dtype=torch.float64,
        ),
        atol=1e-12,
        rtol=1e-10,
    )
    energy.sum().backward()
    disconnected = [
        name
        for name, parameter in mace.named_parameters()
        if parameter.grad is None or not parameter.grad.count_nonzero()
    ]
    assert disconnected == []


def test_first_atomic_basis_normalizes_by_training_neighbor_average() -> None:
    config = model.MACEConfig(channels=2, radial_mlp_widths=(4, 8))
    first = model.AtomicBasis(config, avg_num_neighbors=1.0).double()
    second = model.AtomicBasis(config, avg_num_neighbors=2.0).double()
    second.load_state_dict(first.state_dict())
    features = torch.ones(2, 2, dtype=torch.float64)
    attrs = torch.eye(4, dtype=torch.float64)[:2]
    edges = torch.tensor([[0, 1], [1, 0]])
    vectors = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    radial = torch.ones(2, 8, dtype=torch.float64)
    output = first(features, attrs, edges, vectors, radial)
    assert output.count_nonzero()
    torch.testing.assert_close(
        second(features, attrs, edges, vectors, radial), output / 2
    )


def test_first_atomic_basis_uses_receiver_species_without_changing_sender_features() -> (
    None
):
    torch.manual_seed(123)
    config = model.MACEConfig(channels=2, radial_mlp_widths=(4, 8))
    first = model.AtomicBasis(config, avg_num_neighbors=2.0).double()
    features = torch.ones(2, 2, dtype=torch.float64)
    attrs = torch.eye(4, dtype=torch.float64)[:2]
    edges = torch.tensor([[0], [1]])
    vectors = torch.tensor([[1.0, 0.2, 0.1]], dtype=torch.float64)
    radial = torch.ones(1, 8, dtype=torch.float64)
    output = first(features, attrs, edges, vectors, radial)
    other_attrs = attrs.clone()
    other_attrs[1] = torch.eye(4, dtype=torch.float64)[3]
    other = first(features, other_attrs, edges, vectors, radial)
    assert not torch.allclose(output[1], other[1])
    torch.testing.assert_close(output[0], other[0])


def test_symmetric_contraction_returns_packed_hidden_features() -> None:
    """Eq. 10--11 contractions return the requested 0e/1o/2e blocks."""
    config = model.MACEConfig(channels=2)
    contraction = model.SymmetricContraction(config)
    atomic_basis = torch.randn(3, 2 * (1 + 3 + 5 + 7))
    node_attrs = torch.eye(config.num_elements)[:3]

    output = contraction(atomic_basis, node_attrs)

    assert output.shape == (3, o3.Irreps(config.hidden_irreps).dim)
    assert torch.isfinite(output).all()


def test_interaction_block_aggregates_edges_and_preserves_gradients() -> None:
    """Eq. 8 returns packed atomic features with a differentiable edge path."""
    config = model.MACEConfig(channels=2, radial_mlp_widths=(4, 4, 4, 8))
    interaction = model.InteractionBlock(config, avg_num_neighbors=2.0)
    node_features = torch.randn(
        3,
        o3.Irreps(config.hidden_irreps).dim,
        requires_grad=True,
    )
    node_attrs = torch.eye(config.num_elements)[:3]
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
        dtype=torch.long,
    )
    edge_vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ]
    )
    edge_radial = torch.randn(6, config.num_radial_basis, requires_grad=True)

    atomic_basis, skip = interaction(
        node_features,
        node_attrs,
        edge_index,
        edge_vectors,
        edge_radial,
    )

    assert atomic_basis.shape == (
        3,
        config.channels * sum(2 * ell + 1 for ell in range(4)),
    )
    assert skip.shape == (3, o3.Irreps(config.hidden_irreps).dim)
    assert torch.isfinite(atomic_basis).all()
    assert torch.isfinite(skip).all()

    (atomic_basis.square().sum() + skip.square().sum()).backward()
    assert node_features.grad is not None
    assert torch.isfinite(node_features.grad).all()
    assert edge_radial.grad is not None
    assert torch.isfinite(edge_radial.grad).all()


def test_model_returns_graph_energies_and_atom_forces() -> None:
    config = model.MACEConfig(channels=4, radial_mlp_widths=(8, 8, 8, 16))
    mace = model.MACE(config, avg_num_neighbours=2.0).double()
    node_attrs = torch.eye(config.num_elements, dtype=torch.float64)[[0, 1, 3]]
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long
    )
    graph_index = torch.zeros(3, dtype=torch.long)
    energy, forces = mace(node_attrs, positions, edge_index, graph_index)
    assert energy.shape == (1,)
    assert forces.shape == (3, 3)
    assert energy.dtype == torch.float64
    assert forces.dtype == torch.float64


def test_model_applies_energy_baseline_shift_and_force_rms_scale() -> None:
    config = model.MACEConfig(channels=4, radial_mlp_widths=(8, 8, 8, 16))
    torch.manual_seed(17)
    unscaled_model = model.MACE(config, avg_num_neighbours=2.0).double().eval()
    torch.manual_seed(17)
    scaled_model = (
        model.MACE(
            config,
            avg_num_neighbours=2.0,
            atomic_energies=torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64),
            mean_energy_shift=2.0,
            rms_forces=3.0,
        )
        .double()
        .eval()
    )

    node_attrs = torch.eye(config.num_elements, dtype=torch.float64)[[0, 1, 3]]
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long
    )
    graph_index = torch.zeros(3, dtype=torch.long)

    unscaled_energy, unscaled_forces = unscaled_model(
        node_attrs, positions.clone(), edge_index, graph_index
    )
    scaled_energy, scaled_forces = scaled_model(
        node_attrs, positions.clone(), edge_index, graph_index
    )

    # Baselines are H + C + O = 10 + 20 + 40; shift is 2 per atom.
    torch.testing.assert_close(scaled_energy, 3.0 * unscaled_energy + 76.0)
    torch.testing.assert_close(scaled_forces, 3.0 * unscaled_forces)


def test_symmetric_contraction_preserves_packed_irrep_equivariance() -> None:
    """Unpacking A must preserve each channel's angular components."""
    torch.manual_seed(42)
    config = model.MACEConfig(channels=4)
    contraction = model.SymmetricContraction(config).double()
    atomic_basis = torch.randn(2, 64, dtype=torch.float64)
    node_attrs = torch.eye(4, dtype=torch.float64)[:2]
    rotation = o3.rand_matrix(dtype=torch.float64)
    input_rotation = contraction.input_irreps.D_from_matrix(rotation)
    output_rotation = contraction.output_irreps.D_from_matrix(rotation)

    output = contraction(atomic_basis, node_attrs)
    rotated_output = contraction(atomic_basis @ input_rotation.T, node_attrs)

    torch.testing.assert_close(
        rotated_output, output @ output_rotation.T, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("reflection", [False, True])
def test_model_energy_and_forces_respect_orthogonal_transformations(
    reflection: bool,
) -> None:
    """A rotated/reflected molecule has the same energy and transformed forces."""
    torch.manual_seed(42)
    config = model.MACEConfig(channels=4, radial_mlp_widths=(8, 8, 8, 16))
    mace = model.MACE(config, avg_num_neighbours=2.0).double().eval()
    node_attrs = torch.eye(4, dtype=torch.float64)[[0, 1, 3]]
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [0.1, 1.1, 0.3]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
    graph_index = torch.zeros(3, dtype=torch.long)
    transform = o3.rand_matrix(dtype=torch.float64)
    if reflection:
        transform = -transform

    energy, forces = mace(node_attrs, positions.clone(), edges, graph_index)
    transformed_energy, transformed_forces = mace(
        node_attrs, positions @ transform.T, edges, graph_index
    )

    torch.testing.assert_close(transformed_energy, energy, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        transformed_forces, forces @ transform.T, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("block_kind", ["first", "second"])
def test_atomic_features_vanish_when_an_edge_reaches_cutoff(block_kind: str) -> None:
    """A zero cutoff radial input must not leave a bias-driven edge message."""
    torch.manual_seed(123)
    config = model.MACEConfig(channels=4, radial_mlp_widths=(8, 8, 8, 16))
    vectors = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float64)
    lengths = torch.tensor([[5.0]], dtype=torch.float64)
    edge_radial = model.BesselRadialBasis()(lengths)
    edges = torch.tensor([[0], [1]])
    if block_kind == "first":
        first = model.AtomicBasis(config, avg_num_neighbors=2.0).double()
        node_features = torch.randn(2, config.channels, dtype=torch.float64)
        atomic_basis = first(
            node_features,
            torch.eye(4, dtype=torch.float64)[:2],
            edges,
            vectors,
            edge_radial,
        )
    else:
        second = model.InteractionBlock(config, avg_num_neighbors=2.0).double()
        node_features = torch.randn(
            2, o3.Irreps(config.hidden_irreps).dim, dtype=torch.float64
        )
        node_attrs = torch.eye(4, dtype=torch.float64)[:2]
        atomic_basis, _ = second(node_features, node_attrs, edges, vectors, edge_radial)

    torch.testing.assert_close(
        atomic_basis, torch.zeros_like(atomic_basis), atol=0, rtol=0
    )


def test_model_energy_and_forces_are_continuous_when_an_edge_leaves_cutoff() -> None:
    """Rebuilding topology across 5 angstrom must not introduce an energy jump."""
    torch.manual_seed(17)
    config = model.MACEConfig(channels=4, radial_mlp_widths=(8, 8, 8, 16))
    mace = model.MACE(config, avg_num_neighbours=2.0).double().eval()
    predictions = []
    edge_counts = []
    for distance in (5.0 - 1e-7, 5.0 + 1e-7):
        record = data.AtomicConfiguration(
            atomic_numbers=torch.tensor([1, 6, 8]),
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.3, 0.5, 0.0], [distance, 0.0, 0.0]],
                dtype=torch.float64,
            ),
            energy=torch.tensor(0.0, dtype=torch.float64),
            forces=torch.zeros(3, 3, dtype=torch.float64),
        )
        graph = data.build_neighbor_graph(record, cutoff=config.cutoff)
        edge_counts.append(graph.edge_index.shape[1])
        predictions.append(
            mace(
                graph.node_attrs,
                graph.positions,
                graph.edge_index,
                torch.zeros(3, dtype=torch.long),
            )
        )

    assert edge_counts == [6, 4]
    for inside, outside in zip(predictions[0], predictions[1], strict=True):
        torch.testing.assert_close(inside, outside, atol=1e-6, rtol=1e-6)
