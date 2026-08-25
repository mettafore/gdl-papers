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


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="TODO(luv): deterministic split"
)
def test_split_is_deterministic_and_has_paper_sizes() -> None:
    frames = list(range(500))
    train_a, valid_a = data.split_train_valid(frames, seed=123)
    train_b, valid_b = data.split_train_valid(frames, seed=123)
    assert train_a == train_b
    assert valid_a == valid_b
    assert len(train_a) == 450
    assert len(valid_a) == 50
    assert set(train_a).isdisjoint(valid_a)


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="TODO(luv): neighbour graph"
)
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
    assert graph.edge_vectors.shape == (2, 3)
    assert torch.allclose(graph.edge_vectors[0], -graph.edge_vectors[1])


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="TODO(luv): relative geometry must ignore translation",
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
    assert torch.allclose(graph.edge_vectors, shifted_graph.edge_vectors)
