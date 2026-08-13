"""Time-conditioned tangent vector field on the unit sphere."""

import torch
from torch import nn


class TimeConditionedVectorField(nn.Module):
    """Return tangent vectors on ``S^2`` from points and batch-wise times.

    Public contract: ``x`` has shape ``(n, 3)``, ``t`` has shape ``(n,)``, and
    the returned vector field has shape ``(n, 3)`` with each row tangent at
    the corresponding point in ``x``.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 10) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.linear = torch.nn.Linear(4, self.hidden_dim)
        self.non_lin = torch.nn.SiLU()
        self.hidden_layers = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim), torch.nn.SiLU()
                )
                for _ in range(n_layers)
            ]
        )
        self.out = torch.nn.Linear(hidden_dim, 3)

    @staticmethod
    def _validate_inputs(x: torch.Tensor, t: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("x must have shape (n, 3)")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must have shape (n,) matching x")
        if not torch.isfinite(x).all() or not torch.isfinite(t).all():
            raise ValueError("x and t must contain finite values")

    def _ambient_field(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the unconstrained ambient ``R^3`` vector field."""
        xt_concat = torch.concat([x, t.reshape(-1, 1)], dim=1)
        ut = self.linear(xt_concat)
        ut = self.non_lin(ut)
        ut = self.hidden_layers(ut)
        return self.out(ut)

    @staticmethod
    def _project_to_tangent(x: torch.Tensor, ambient: torch.Tensor) -> torch.Tensor:
        """Project an ambient vector onto the tangent space at ``x``."""
        dot_prod = (x * ambient).sum(dim=1, keepdim=True)
        u_par = dot_prod * x
        u_perp = ambient - u_par
        return u_perp

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the time-conditioned tangent vector field."""
        self._validate_inputs(x, t)
        ambient = self._ambient_field(x, t)
        return self._project_to_tangent(x, ambient)
