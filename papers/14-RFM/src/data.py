"""Fire dataset loading and deterministic splits on the unit sphere."""

from pathlib import Path

import pandas as pd
import torch


def latlon_degrees_to_cartesian(latlon: torch.Tensor) -> torch.Tensor:
    """Convert latitude/longitude degrees from shape ``(n, 2)`` to ``(n, 3)``.

    The output must be float32 Cartesian points on the unit sphere. Input rows
    are ordered as latitude then longitude.
    """
    if latlon.ndim != 2:
        raise ValueError("latlon dataset should be 2-dimensional")
    if latlon.shape[1] != 2:
        raise ValueError("latlon records must have 2 columns.")
    if (latlon[:, 0] > 90).any() or (latlon[:, 0] < -90).any():
        raise ValueError("Latitudes must be between -90 and 90 degrees.")
    if (latlon[:, 1] > 180).any() or (latlon[:, 1] < -180).any():
        raise ValueError("Longitudes must be between -180 and 180 degrees.")
    if ~torch.isfinite(latlon).all():
        raise ValueError("latlon must contain finite values.")
    if latlon.dtype != torch.float32:
        latlon = latlon.to(torch.float32)
    z = torch.sin(torch.pi / 180 * latlon[:, 0]).reshape(-1, 1)
    theta = torch.pi / 180 * latlon[:, 1]
    r = torch.sqrt(1 - torch.square(z))
    x = r * torch.cos(theta).reshape(-1, 1)
    y = r * torch.sin(theta).reshape(-1, 1)
    return torch.concat([x, y, z], dim=1)


def load_fire_csv(path: str | Path) -> torch.Tensor:
    """Load a headered Fire CSV and return Cartesian points of shape ``(n, 3)``."""
    data = pd.read_csv(path)
    latlon = torch.tensor(data.values)
    return latlon_degrees_to_cartesian(latlon)


def split_points(
    points: torch.Tensor, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic train, validation, and test tensors (80/10/10)."""
    generator = torch.Generator().manual_seed(seed)
    n = len(points)
    indices = torch.randperm(n, generator=generator)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train, val, test = torch.split(points[indices], [n_train, n_val, n_test])
    return train, val, test
