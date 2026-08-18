"""Fire dataset loading and deterministic splits on the unit sphere."""

import math
from pathlib import Path

import pandas as pd
import torch


def cartesian_to_latlon_degrees(points: torch.Tensor) -> torch.Tensor:
    """Convert Cartesian points from shape ``(n, 3)`` to latitude/longitude.

    Rows are normalized before conversion so small numerical drift from ODE
    integration does not distort the geographic coordinates.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if points.shape[0] == 0:
        raise ValueError("points must contain at least one row")
    if not torch.isfinite(points).all():
        raise ValueError("points must contain finite values")

    norms = torch.linalg.vector_norm(points, dim=1, keepdim=True)
    if torch.any(norms == 0):
        raise ValueError("points must not contain the zero vector")
    unit_points = points / norms
    latitude = torch.rad2deg(torch.asin(unit_points[:, 2].clamp(-1.0, 1.0)))
    longitude = torch.rad2deg(torch.atan2(unit_points[:, 1], unit_points[:, 0]))
    return torch.stack((latitude, longitude), dim=1)


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
    points: torch.Tensor,
    seed: int = 0,
    *,
    train_pct: float = 80.0,
    val_pct: float = 10.0,
    test_pct: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic train, validation, and test tensors by percentage.

    The three percentages must be finite, non-negative, and sum to 100. Any
    fractional row left after rounding train and validation counts is assigned
    to the test split, preserving every input row exactly once.
    """
    percentages = (train_pct, val_pct, test_pct)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in percentages
    ):
        raise ValueError("split percentages must be finite numbers")
    if any(value < 0 for value in percentages):
        raise ValueError("split percentages must be non-negative")
    if not math.isclose(sum(percentages), 100.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split percentages must sum to 100")

    generator = torch.Generator().manual_seed(seed)
    n = len(points)
    indices = torch.randperm(n, generator=generator)
    n_train = int(train_pct / 100.0 * n)
    n_val = int(val_pct / 100.0 * n)
    n_test = n - n_train - n_val
    train, val, test = torch.split(points[indices], [n_train, n_val, n_test])
    return train, val, test
