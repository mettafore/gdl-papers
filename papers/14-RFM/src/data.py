"""Fire dataset loading and deterministic splits on the unit sphere."""

from pathlib import Path

import torch


def latlon_degrees_to_cartesian(latlon: torch.Tensor) -> torch.Tensor:
    """Convert latitude/longitude degrees from shape ``(n, 2)`` to ``(n, 3)``.

    The output must be float32 Cartesian points on the unit sphere. Input rows
    are ordered as latitude then longitude.
    """
    # TODO(luv):
    # - Reject malformed or non-finite coordinates with a clear ValueError.
    # - Enforce the documented latitude and longitude domains.
    # - Convert each valid row into the agreed Cartesian coordinate convention.
    raise NotImplementedError


def load_fire_csv(path: str | Path) -> torch.Tensor:
    """Load a headered Fire CSV and return Cartesian points of shape ``(n, 3)``."""
    # TODO(luv):
    # - Read the two documented coordinate columns without hiding malformed rows.
    # - Return the validated Cartesian representation through the public converter.
    raise NotImplementedError


def split_points(
    points: torch.Tensor, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic train, validation, and test tensors (80/10/10)."""
    # TODO(luv):
    # - Validate the point tensor before splitting.
    # - Create one seed-controlled partition with every row used exactly once.
    # - Return partitions in train, validation, test order.
    raise NotImplementedError
