"""QM9 -> dense padded NumPy arrays, converted once (no per-step transforms).

Reuses ../src/data.py for the dataset, target columns, split sizes and the
15-dim charge-power features, so the split and normalization match PyTorch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data  # noqa: E402  (the PyTorch data module)

N_MAX = 29  # largest QM9 molecule

# Row z = 15-dim charge-power features of atomic number z; row 0 = padding.
_ELEMENTS = torch.tensor([1, 6, 7, 8, 9])
FEATURE_TABLE = np.zeros((10, 15), np.float32)
FEATURE_TABLE[_ELEMENTS.numpy()] = data.charge_power_features(_ELEMENTS).numpy()


def pad(zs: list[np.ndarray], poss: list[np.ndarray], n_max: int = N_MAX) -> dict:
    """Pad molecules to n_max atoms -> {"z": (M, n_max) int8, "pos": (M, n_max, 3)}."""
    z = np.zeros((len(zs), n_max), np.int8)
    pos = np.zeros((len(zs), n_max, 3), np.float32)
    for i, (zi, pi) in enumerate(zip(zs, poss, strict=True)):
        z[i, : len(zi)] = zi
        pos[i, : len(zi)] = pi
    return {"z": z, "pos": pos}


def load_qm9(
    root: str, target: str = "gap", seed: int = 42
) -> tuple[dict, data.NormStats]:
    """Return ({"train"|"val"|"test": {"z","pos","y"}}, NormStats on train)."""
    col = data.QM9_TARGETS[target]
    ds = data.QM9(root=root)  # no transform: features/edges are built here, once
    z_all = ds._data.z.numpy()
    pos_all = ds._data.pos.numpy()
    y_all = ds._data.y[:, col].numpy()
    starts = ds.slices["z"].numpy()

    # Same split as data.load_split (seeded randperm over the whole dataset).
    perm = torch.randperm(
        len(ds), generator=torch.Generator().manual_seed(seed)
    ).numpy()
    idx = {
        "train": perm[: data.N_TRAIN],
        "val": perm[data.N_TRAIN : data.N_TRAIN + data.N_VAL],
        "test": perm[data.N_TRAIN + data.N_VAL :],
    }

    splits = {}
    for name, ids in idx.items():
        zs = [z_all[starts[i] : starts[i + 1]] for i in ids]
        poss = [pos_all[starts[i] : starts[i + 1]] for i in ids]
        splits[name] = pad(zs, poss) | {"y": y_all[ids].astype(np.float32)}

    y = torch.from_numpy(splits["train"]["y"])  # float32, as in data.load_split
    mean = y.mean().item()
    norm = data.NormStats(mean=mean, mad=(y - mean).abs().mean().item())
    return splits, norm
