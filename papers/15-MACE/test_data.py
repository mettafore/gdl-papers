"""Executable contracts for 3BPA splitting and graph construction."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

import data


def test_validate_data_dir_reports_all_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train_300K.xyz"):
        data.validate_data_dir(tmp_path)


def test_split_is_deterministic_and_has_paper_sizes() -> None:
    frames = list(range(500))
    train_a, valid_a = data.split_train_valid(frames, seed=123)
    train_b, valid_b = data.split_train_valid(frames, seed=123)
    assert train_a == train_b
    assert valid_a == valid_b
    assert len(train_a) == 450
    assert len(valid_a) == 50
    assert set(train_a).isdisjoint(valid_a)


def test_split_matches_paper_era_prefix_train_suffix_valid() -> None:
    train, valid = data.split_train_valid(list(range(10)), valid_size=2, seed=123)
    assert train == [7, 4, 0, 2, 1, 5, 6, 8]
    assert valid == [3, 9]


def test_neighbor_graph_has_both_directions_and_no_self_edges() -> None:
    configuration = data.AtomicConfiguration(
        atomic_numbers=torch.tensor([1, 6, 8]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        energy=torch.tensor(-1.0, dtype=torch.float64),
        forces=torch.zeros((3, 3), dtype=torch.float64),
    )
    graph = data.build_neighbor_graph(configuration, cutoff=1.5)
    edges = set(map(tuple, graph.edge_index.T.tolist()))
    assert edges == {(0, 1), (1, 0)}
    assert graph.shifts.shape == (2, 3)
    assert torch.equal(graph.shifts, torch.zeros((2, 3), dtype=torch.float64))
    assert torch.equal(
        graph.node_attrs,
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
    )


def test_neighbor_vectors_are_translation_invariant() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]], dtype=torch.float64)
    base = data.AtomicConfiguration(
        torch.tensor([1, 6]),
        positions,
        torch.tensor(0.0, dtype=torch.float64),
        torch.zeros((2, 3), dtype=torch.float64),
    )
    shifted = data.AtomicConfiguration(
        base.atomic_numbers,
        positions + torch.tensor([4.0, -2.0, 7.0]),
        base.energy,
        base.forces,
    )
    graph = data.build_neighbor_graph(base, cutoff=2.0)
    shifted_graph = data.build_neighbor_graph(shifted, cutoff=2.0)
    assert torch.equal(graph.edge_index, shifted_graph.edge_index)
    assert torch.equal(graph.shifts, shifted_graph.shifts)


def test_data_loader_offsets_edges_and_assigns_graph_membership() -> None:
    first = data.AtomicConfiguration(
        torch.tensor([1, 6]),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor(-1.0, dtype=torch.float64),
        torch.zeros((2, 3), dtype=torch.float64),
    )
    second = data.AtomicConfiguration(
        torch.tensor([7, 8]),
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64),
        torch.tensor(-2.0, dtype=torch.float64),
        torch.zeros((2, 3), dtype=torch.float64),
    )

    batch = next(
        iter(
            data.get_data_loader(
                [
                    data.build_neighbor_graph(first, cutoff=1.5),
                    data.build_neighbor_graph(second, cutoff=1.5),
                ],
                batch_size=2,
                shuffle=False,
            )
        )
    )

    assert set(map(tuple, batch.edge_index.T.tolist())) == {
        (0, 1),
        (1, 0),
        (2, 3),
        (3, 2),
    }
    assert batch.batch.tolist() == [0, 0, 1, 1]
    assert batch.ptr.tolist() == [0, 2, 4]
    assert batch.energy.tolist() == [-1.0, -2.0]
    assert batch.node_attrs.shape == (4, 4)


def test_isolated_atom_energies_follow_fixed_h_c_n_o_order(tmp_path: Path) -> None:
    path = tmp_path / "iso_atoms.xyz"
    path.write_text(
        """1
Properties=species:S:1:pos:R:3 energy=-20.0 pbc=\"F F F\"
C 0.0 0.0 0.0
1
Properties=species:S:1:pos:R:3 energy=-10.0 pbc=\"F F F\"
H 0.0 0.0 0.0
1
Properties=species:S:1:pos:R:3 energy=-30.0 pbc=\"F F F\"
N 0.0 0.0 0.0
1
Properties=species:S:1:pos:R:3 energy=-40.0 pbc=\"F F F\"
O 0.0 0.0 0.0
""",
        encoding="utf-8",
    )

    assert torch.equal(
        data.load_isolated_atom_energies(path),
        torch.tensor([-10.0, -20.0, -30.0, -40.0], dtype=torch.float64),
    )
