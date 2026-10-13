"""3BPA records and author-faithful graph construction."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

TRAIN_FILE = "train_300K.xyz"
ISOLATED_ATOMS_FILE = "iso_atoms.xyz"
ATOMIC_NUMBERS = (1, 6, 7, 8)  # H, C, N, O
ATOMIC_NUMBER_TO_INDEX = {x: idx for idx, x in enumerate(ATOMIC_NUMBERS)}
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
    iso_atoms = read(path, index=":")
    energy_dict: dict[int, Tensor] = {}
    for frame_index, atoms in enumerate(iso_atoms):
        if len(atoms) != 1:
            raise ValueError(
                f"isolated-atom frame {frame_index} contains "
                f"{len(atoms)} atoms; expected 1"
            )
        atomic_number = int(atoms.get_atomic_numbers()[0])
        if atomic_number not in ATOMIC_NUMBER_TO_INDEX:
            raise ValueError(
                f"unsupported isolated-atom atomic number {atomic_number} "
                f"in frame {frame_index}"
            )
        if atomic_number in energy_dict:
            raise ValueError(
                f"duplicate isolated-atom record for atomic number {atomic_number}"
            )
        energy_dict[atomic_number] = torch.tensor(
            atoms.get_potential_energy(), dtype=torch.float64
        )

    missing = [z for z in ATOMIC_NUMBERS if z not in energy_dict]
    if missing:
        raise ValueError(f"missing isolated-atom records for atomic numbers {missing}")
    return torch.stack([energy_dict[z] for z in ATOMIC_NUMBERS])


def split_train_valid[T](
    configurations: list[T], *, valid_size: int = 50, seed: int = 123
) -> tuple[list[T], list[T]]:
    """Return the paper-era shuffled-prefix train, shuffled-suffix valid split."""
    if not 0 < valid_size < len(configurations):
        raise ValueError(
            "valid_size must be positive and smaller than the number of configurations"
        )
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
    senders: list[int] = []
    receivers: list[int] = []
    for sender in range(configuration.positions.shape[0]):
        for receiver in range(configuration.positions.shape[0]):
            if sender == receiver:
                continue
            distance = torch.linalg.vector_norm(
                configuration.positions[receiver] - configuration.positions[sender]
            )
            if distance < cutoff:
                senders.append(sender)
                receivers.append(receiver)

    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    node_attrs = torch.tensor(
        [ATOMIC_NUMBER_TO_INDEX[x.item()] for x in configuration.atomic_numbers]
    )
    node_attrs = torch.nn.functional.one_hot(
        node_attrs, num_classes=len(ATOMIC_NUMBERS)
    ).to(torch.float64)
    return AtomicGraph(
        atomic_numbers=configuration.atomic_numbers,
        positions=configuration.positions,
        edge_index=edge_index,
        energy=configuration.energy,
        forces=configuration.forces,
        node_attrs=node_attrs,
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
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


def validate_data_dir(data_dir: Path) -> None:
    """Fail early if an expected official 3BPA subset is absent."""
    required = {TRAIN_FILE, ISOLATED_ATOMS_FILE}
    required.update(name for names in TEST_FILES.values() for name in names)
    missing = sorted(name for name in required if not (data_dir / name).is_file())
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"missing 3BPA files in {data_dir}: {joined}")
