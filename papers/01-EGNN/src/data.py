"""QM9 data + the shared, deterministic split for EGNN.  [SCAFFOLD — you fill bodies]

Contract to match (same as template `data.py`):
  ONE seeded split, imported by both train.py and evaluate.py so they always
  agree on train/val/test — no leakage. Target normalization stats come from the
  TRAIN split only and are returned, so train/evaluate normalize identically.

Backend: torch_geometric.datasets.QM9 (small molecules, 3D coords, 19 targets).
NOTE: torch_geometric is not yet a project dependency. Add it before running:
    uv add torch_geometric

Paper: Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021.
Standard split: 100K train / 18K val / 13K test.

------------------------------------------------------------------------------
WHAT YOU NEED TO BUILD (in rough order):
  1. NormStats.normalize / denormalize        — the (y-mean)/mad transform
  2. load_split:
       a. load QM9(root)
       b. deterministic permutation w/ an EXPLICIT generator (not global RNG)
       c. slice into 100K / 18K / 13K
       d. compute NormStats on TRAIN targets only  (think: which column? why train-only?)
       e. wrap each split in a PyG DataLoader
       f. return (train, val, test, norm, dims)

Questions worth answering as you go (this is the point of the exercise):
  - Why must the split use an explicit generator, not torch.manual_seed?
  - Why is normalization fit on train only — what leaks if you use the full set?
  - QM9 y has 19 columns. Which index is your target? Verify against PyG docs.
  - Where should edge construction live — here, or in the model? Why?
------------------------------------------------------------------------------
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
# Only the properties EGNN benchmarks (Table 1) are listed.
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

    dims: dict with `in_node_dim` (atom one-hot width) and `out_dim` (=1), so the
    model never hardcodes shapes. norm: NormStats fit on the TRAIN split.
    """
    _require_pyg()
    if target not in QM9_TARGETS:
        raise KeyError(f"unknown target {target!r}; choose from {sorted(QM9_TARGETS)}")
    col = QM9_TARGETS[target]

    # TODO a: load QM9
    dataset = QM9(root=root)
    # TODO b: deterministic perm with an EXPLICIT torch.Generator().manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=gen)
    # TODO c: slice perm -> train_idx / val_idx / test_idx (use N_TRAIN, N_VAL)
    train_idx = perm[:N_TRAIN]
    val_idx = perm[N_TRAIN : N_TRAIN + N_VAL]
    test_idx = perm[N_TRAIN + N_VAL :]
    # TODO d: fit NormStats on train targets at column `col` (train ONLY)
    train_data = dataset[train_idx]
    train_y = torch.cat([i.y[:, col] for i in train_data], dim=0)
    mean = train_y.mean().item()
    mad = (train_y - mean).abs().mean().item()
    normalizer = NormStats(mean=mean, mad=mad)
    # TODO e: wrap each split in PyGDataLoader (shuffle train, not val/test)
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
    # TODO f: build dims
    dims = {"in_node_dim": dataset.num_features, "out_dim": 1, "target_col": col}
    return train_loader, val_loader, test_loader, normalizer, dims
