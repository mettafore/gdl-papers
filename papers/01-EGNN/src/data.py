"""Prepare QM9 molecules and reproducible dataset splits for EGNN.

Atom features and complete directed graphs follow the author’s QM9 setup.
Training and evaluation share the same seeded split. Target normalization uses
only training-set statistics (mean and mean absolute deviation).

Paper: Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# torch_geometric is optional-at-import so the file is inspectable without the
# dep installed. The actual functions require it.
try:
    from torch_geometric.datasets import QM9
    from torch_geometric.loader import DataLoader as PyGDataLoader

    _HAS_PYG = True
except ImportError:  # pragma: no cover - dep not installed yet
    QM9 = None
    PyGDataLoader = None
    _HAS_PYG = False


# QM9 target columns in torch_geometric's `data.y` (shape [1, 19]).
# Reference table — VERIFY these indices against the PyG QM9 docs before trusting.
# Only the properties EGNN benchmarks (Table 3) are listed.
QM9_TARGETS: dict[str, int] = {
    "mu": 0,  # dipole moment (D)
    "alpha": 1,  # isotropic polarizability (a0^3)
    "homo": 2,  # HOMO (eV)
    "lumo": 3,  # LUMO (eV)
    "gap": 4,  # HOMO-LUMO gap (eV)
    "U0": 7,  # internal energy at 0K (eV)
    "Cv": 11,  # heat capacity (cal/mol/K)
    "R2": 5,  # electronic spatial extent (bohr^2)
    "ZPVE": 6,  # zero-point vibrational energy (eV)
    "U": 8,  # internal energy at 298K (eV)
    "H": 9,  # enthalpy at 298K (eV)
    "G": 10,  # free energy at 298K (eV)
}

# Standard QM9 split sizes (Satorras et al.).
N_TRAIN = 100_000
N_VAL = 18_000

# Charge-power node features (reference repo's QM9 config).
# PyG's default data.x is 11-dim: [one-hot(H,C,N,O,F), Z, aromatic, sp, sp2, sp3, num_Hs].
# The reference instead builds 15-dim features from atomic number alone:
#   one_hot(z) ⊗ (z / charge_scale)^p  for p = 0, 1, 2   →  5 × 3 = 15 dims
# where charge_scale = 9 (fluorine, the max Z in QM9). See vgsatorras/egnn
# qm9/data/utils.py `preprocess_input` (charge_power=2).
CHARGE_SCALE = 9.0
CHARGE_POWER = 2


def charge_power_features(z: torch.Tensor) -> torch.Tensor:
    """Map atomic numbers to the reference repo's 15-dim node features.

    Args:
        z: (num_nodes,) int tensor of atomic numbers (PyG QM9's `data.z`;
           values in {1, 6, 7, 8, 9} = H, C, N, O, F).

    Returns:
        (num_nodes, 15) float tensor: for each node, the outer product of its
        5-dim one-hot with [(z/9)^0, (z/9)^1, (z/9)^2], flattened.

    """
    lookup = torch.full((10,), -1, dtype=torch.long)
    lookup[torch.tensor([1, 6, 7, 8, 9])] = torch.arange(5)
    ids = lookup[z]
    one_hot = torch.nn.functional.one_hot(ids, num_classes=5).float()
    powers = (z.float() / CHARGE_SCALE).unsqueeze(1) ** torch.arange(CHARGE_POWER + 1)
    combined = one_hot[:, :, None] * powers[:, None, :]
    return combined.reshape(z.size(0), -1)


def fully_connected_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a fully-connected `edge_index` (all ordered pairs i != j).

    THE BUG THIS FIXES: PyG QM9's default `edge_index` is molecular BONDS
    (sparse — methane gets 8 edges), but EGNN operates on distances between
    ALL atom pairs (reference uses get_adj_matrix = fully connected). Training
    on bonds only made the model blind to most of the geometry — capped MAE
    at ~2x the paper. Reference has NO self-loops and NO edge features.

    Args:
        num_nodes: N atoms in one molecule.

    Returns:
        (2, N*(N-1)) long tensor: row 0 = receiving node i, row 1 = sending node j, every ordered
        pair with source != dest. Methane (N=5) -> 20 edges, not 8.

    """
    idx = torch.arange(num_nodes)
    row = idx.repeat_interleave(num_nodes)
    col = idx.repeat(num_nodes)
    mask = row != col
    return torch.stack([row[mask], col[mask]], dim=0)


def _to_charge_power(data):
    data.x = charge_power_features(data.z)
    # Reference EGNN runs QM9 on complete graphs (get_adj_matrix), not the
    # molecular bonds PyG ships in edge_index. Rebuild per molecule here,
    # before batching, so PyG's collate offsets it correctly. Drop the
    # bond-typed edge_attr — reference uses in_edge_nf=0.
    data.edge_index = fully_connected_edge_index(data.z.shape[0])
    data.edge_attr = None
    return data


@dataclass
class NormStats:
    """Train-set target normalization: subtract mean, divide by MAD."""

    mean: float
    mad: float

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        y_norm = (y - self.mean) / self.mad
        return y_norm

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        y_denorm = y * self.mad + self.mean
        return y_denorm


def _require_pyg() -> None:
    if not _HAS_PYG:
        raise ImportError(
            "torch_geometric is required for QM9. Install it with: uv add torch_geometric"
        )


def load_split(
    target: str = "gap",
    root: str = "data/qm9",
    seed: int = 42,
    batch_size: int = 96,
    num_workers: int = 0,
):
    """Return (train_loader, val_loader, test_loader, norm, dims).

    dims: dict with `in_node_dim` (atom feature width) and `out_dim` (=1), so the
    model never hardcodes shapes. norm: NormStats fit on the TRAIN split.
    """
    _require_pyg()
    if target not in QM9_TARGETS:
        raise KeyError(f"unknown target {target!r}; choose from {sorted(QM9_TARGETS)}")
    col = QM9_TARGETS[target]

    dataset = QM9(root=root, transform=_to_charge_power)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=gen)
    train_idx = perm[:N_TRAIN]
    val_idx = perm[N_TRAIN : N_TRAIN + N_VAL]
    test_idx = perm[N_TRAIN + N_VAL :]
    train_data = dataset[train_idx]
    train_y = torch.cat([i.y[:, col] for i in train_data], dim=0)
    mean = train_y.mean().item()
    mad = (train_y - mean).abs().mean().item()
    normalizer = NormStats(mean=mean, mad=mad)
    train_loader = PyGDataLoader(
        dataset[train_idx],
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = PyGDataLoader(
        dataset[val_idx],
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = PyGDataLoader(
        dataset[test_idx],
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dims = {"in_node_dim": dataset[0].x.shape[1], "out_dim": 1, "target_col": col}
    return train_loader, val_loader, test_loader, normalizer, dims
