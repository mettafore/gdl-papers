"""3BPA records and graph construction contracts.

This module intentionally contains no completed data pipeline. Implement one
`TODO(luv)` at a time and use the strict-xfail tests as the next specification.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from torch import Tensor

T = TypeVar("T")

TRAIN_FILE = "train_300K.xyz"
TEST_FILES = {
    "300K": ("test_300K.xyz",),
    "600K": ("test_600K.xyz",),
    "1200K": ("test_1200K.xyz",),
    "dihedral": ("dih_beta120.xyz", "dih_beta150.xyz", "dih_beta180.xyz"),
}


@dataclass(frozen=True)
class AtomicConfiguration:
    """One molecule before neighbour-list construction.

    Shapes: atomic_numbers `[N]`, positions `[N, 3]`, energy scalar, forces
    `[N, 3]`.
    """

    atomic_numbers: Tensor
    positions: Tensor
    energy: Tensor
    forces: Tensor


@dataclass(frozen=True)
class AtomicGraph:
    """One directed, non-periodic molecular graph.

    `edge_index[0]` contains senders and `edge_index[1]` receivers. Relative
    vectors point from receiver to sender and therefore have shape `[E, 3]`.
    """

    atomic_numbers: Tensor
    positions: Tensor
    edge_index: Tensor
    edge_vectors: Tensor
    graph_index: Tensor
    energy: Tensor
    forces: Tensor


def load_extxyz(path: Path) -> list[AtomicConfiguration]:
    """Read all configurations from one 3BPA extxyz file using ASE.

    TODO(luv): inspect one ASE ``Atoms`` object before deciding which info/array
    keys contain energy and forces; validate shapes and convert to float64.
    """
    raise NotImplementedError


def split_train_valid(
    configurations: list[T], *, valid_size: int = 50, seed: int = 123
) -> tuple[list[T], list[T]]:
    """Return the deterministic 450/50 split used for the 500 training frames.

    TODO(luv): permute indices with a local seeded generator. Do not mutate the
    input list and do not let test configurations enter either result.
    """
    raise NotImplementedError


def build_neighbor_graph(
    configuration: AtomicConfiguration, *, cutoff: float = 5.0
) -> AtomicGraph:
    """Build directed edges for every distinct atom pair inside the cutoff.

    TODO(luv): exclude self edges; include both directions; derive every edge
    vector from positions so translation cancels; keep sender/receiver ordering
    consistent with the docstring above.
    """
    raise NotImplementedError


def collate_graphs(graphs: list[AtomicGraph]) -> AtomicGraph:
    """Batch graphs while offsetting edge indices and assigning graph indices.

    TODO(luv): concatenate atom/edge fields, offset each graph's local indices,
    and stack one scalar energy per graph.
    """
    raise NotImplementedError


def validate_data_dir(data_dir: Path) -> None:
    """Fail early if an expected official 3BPA subset is absent."""
    required = {TRAIN_FILE}
    required.update(name for names in TEST_FILES.values() for name in names)
    missing = sorted(name for name in required if not (data_dir / name).is_file())
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"missing 3BPA files in {data_dir}: {joined}")
