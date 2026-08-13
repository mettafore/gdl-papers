"""Scaffold for a time-conditioned vector field on the unit sphere."""

import torch
from torch import nn


class TimeConditionedVectorField(nn.Module):
    """Return tangent vectors on ``S^2`` from points and batch-wise times.

    Public contract: ``x`` has shape ``(n, 3)``, ``t`` has shape ``(n,)``, and
    the returned vector field has shape ``(n, 3)`` with each row tangent at
    the corresponding point in ``x``.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim

    @staticmethod
    def _validate_inputs(x: torch.Tensor, t: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("x must have shape (n, 3)")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must have shape (n,) matching x")
        if not torch.isfinite(x).all() or not torch.isfinite(t).all():
            raise ValueError("x and t must contain finite values")

    def _ambient_field(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Placeholder for the unconstrained ambient ``R^3`` vector field."""
        # TODO(luv): use time conditioning so the MLP represents a field at t.
        # TODO(luv): produce the ambient output in R^3 from x and conditioned t.
        raise NotImplementedError

    @staticmethod
    def _project_to_tangent(x: torch.Tensor, ambient: torch.Tensor) -> torch.Tensor:
        """Placeholder for projection from ambient space to ``T_x S^2``."""
        # TODO(luv): project the ambient output onto the tangent space at x.
        raise NotImplementedError

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the time-conditioned tangent vector field."""
        self._validate_inputs(x, t)
        ambient = self._ambient_field(x, t)
        # TODO(luv): return the tangent projection of the ambient network output.
        return self._project_to_tangent(x, ambient)
