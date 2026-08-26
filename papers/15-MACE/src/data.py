"""3BPA records and author-faithful graph construction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeVar

import ase.neighborlist
import numpy as np
import torch
from ase.io import read
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

T = TypeVar("T")

TRAIN_FILE = "train_300K.xyz"
ISOLATED_ATOMS_FILE = "iso_atoms.xyz"
ATOMIC_NUMBERS = (1, 6, 7, 8)  # H, C, N, O
TEST_FILES = {
    "300K": ("test_300K.xyz",),
    "600K": ("test_600K.xyz",),
    "1200K": ("test_1200K.xyz",),
    "dihedral": (
        "test_dih_beta120.xyz",
        "test_dih_beta150.xyz",
        "test_dih_beta180.xyz",
    ),
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


class AtomicGraph(Data):
    """One 3BPA graph containing topology but no derived edge geometry.

    ``edge_index[0]`` contains senders and ``edge_index[1]`` receivers. PyG
    creates the batched ``batch`` and ``ptr`` arrays at collation time.
    """

    @property
    def node_attrs(self) -> Tensor:
        """Return MACE node attributes despite PyG 2.8's method-name collision."""
        return self["node_attrs"]

    def __init__(
        self,
        *,
        atomic_numbers: Tensor | None = None,
        node_attrs: Tensor | None = None,
        positions: Tensor | None = None,
        edge_index: Tensor | None = None,
        shifts: Tensor | None = None,
        energy: Tensor | None = None,
        forces: Tensor | None = None,
    ) -> None:
        super().__init__()
        if atomic_numbers is not None:
            self.atomic_numbers = atomic_numbers
        if node_attrs is not None:
            self["node_attrs"] = node_attrs
        if positions is not None:
            self.positions = positions
            self.num_nodes = positions.shape[0]
        if edge_index is not None:
            self.edge_index = edge_index
        if shifts is not None:
            self.shifts = shifts
        if energy is not None:
            self.energy = energy
        if forces is not None:
            self.forces = forces


def load_extxyz(path: Path) -> list[AtomicConfiguration]:
    """Read all configurations from one 3BPA extxyz file in float64."""
    frames = read(path, index=":")
    configurations = []
    for atoms in frames:
        configurations.append(
            AtomicConfiguration(
                atomic_numbers=torch.as_tensor(
                    atoms.get_atomic_numbers(), dtype=torch.long
                ),
                positions=torch.as_tensor(atoms.get_positions(), dtype=torch.float64),
                energy=torch.tensor(atoms.get_potential_energy(), dtype=torch.float64),
                forces=torch.as_tensor(atoms.get_forces(), dtype=torch.float64),
            )
        )
    return configurations


def load_isolated_atom_energies(path: Path) -> Tensor:
    """Return isolated-atom energies in fixed H, C, N, O channel order."""
    frames = read(path, index=":")
    energies_by_atomic_number: dict[int, float] = {}
    for atoms in frames:
        if len(atoms) != 1:
            raise ValueError(f"expected one atom per configuration in {path}")
        atomic_number = int(atoms.get_atomic_numbers()[0])
        energies_by_atomic_number[atomic_number] = float(atoms.get_potential_energy())

    missing = [z for z in ATOMIC_NUMBERS if z not in energies_by_atomic_number]
    if missing:
        raise ValueError(f"missing isolated-atom energies for atomic numbers {missing}")
    return torch.tensor(
        [energies_by_atomic_number[z] for z in ATOMIC_NUMBERS],
        dtype=torch.float64,
    )


def split_train_valid(
    configurations: list[T], *, valid_size: int = 50, seed: int = 123
) -> tuple[list[T], list[T]]:
    """Return the paper-era shuffled-prefix train, shuffled-suffix valid split."""
    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(len(configurations))
    train_size = len(configurations) - valid_size
    train = [configurations[i] for i in indices[:train_size]]
    valid = [configurations[i] for i in indices[train_size:]]
    return train, valid


def build_neighbor_graph(
    configuration: AtomicConfiguration, *, cutoff: float = 5.0
) -> AtomicGraph:
    """Build directed topology for every distinct atom pair inside the cutoff.

    Edge vectors are deliberately absent: the model derives them from its live,
    differentiable positions on every forward pass.
    """
    cell = np.identity(3, dtype=float)
    sender, receiver, unit_shifts = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        pbc=(False, False, False),
        cell=cell,
        positions=configuration.positions.detach().cpu().numpy(),
        cutoff=cutoff,
        self_interaction=True,
        use_scaled_positions=False,
    )
    keep = ~((sender == receiver) & np.all(unit_shifts == 0, axis=1))
    edge_index = torch.as_tensor(
        np.stack((sender[keep], receiver[keep])), dtype=torch.long
    )
    shifts = torch.as_tensor(
        unit_shifts[keep] @ cell, dtype=configuration.positions.dtype
    )

    try:
        element_indices = [
            ATOMIC_NUMBERS.index(int(z)) for z in configuration.atomic_numbers
        ]
    except ValueError as exc:
        raise ValueError("3BPA supports only H, C, N, and O") from exc
    node_attrs = F.one_hot(
        torch.tensor(element_indices, dtype=torch.long),
        num_classes=len(ATOMIC_NUMBERS),
    ).to(dtype=configuration.positions.dtype)

    return AtomicGraph(
        atomic_numbers=configuration.atomic_numbers,
        node_attrs=node_attrs,
        positions=configuration.positions,
        edge_index=edge_index,
        shifts=shifts,
        energy=configuration.energy,
        forces=configuration.forces,
    )


def get_data_loader(
    dataset: Sequence[AtomicGraph],
    *,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Batch graphs with PyG's index-offset and graph-membership handling."""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def validate_data_dir(data_dir: Path) -> None:
    """Fail early if an expected official 3BPA subset is absent."""
    required = {TRAIN_FILE, ISOLATED_ATOMS_FILE}
    required.update(name for names in TEST_FILES.values() for name in names)
    missing = sorted(name for name in required if not (data_dir / name).is_file())
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"missing 3BPA files in {data_dir}: {joined}")
