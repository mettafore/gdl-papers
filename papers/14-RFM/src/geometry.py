"""Sphere geometry contracts for the RFM reproduction.

The notebook ``nbs/spherical.ipynb`` is the executable derivation and oracle.
This module is intentionally scaffold-only: the correctness-critical math is
implemented by the learner and checked separately.
"""

import torch


def _check_point_batch(x: torch.Tensor, name: str) -> None:
    """Validate the non-mathematical part of a batched S2 point contract."""
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} must contain finite values")


def _check_point_pair(x: torch.Tensor, y: torch.Tensor) -> None:
    """Validate matching batched point shapes."""
    _check_point_batch(x, "x")
    _check_point_batch(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")


def sample_sphere(n: int) -> torch.Tensor:
    """Return ``n`` approximately uniform points on the unit sphere.

    Contract: return a float tensor of shape ``(n, 3)`` whose rows have unit
    Euclidean norm.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    # Sampling z from -1 to 1
    z = 2 * torch.rand(n, 1) - 1
    # Getting radius of circle at coord z
    r = torch.sqrt(1 - torch.square(z))
    # Getting theta to get x and y
    theta = 2 * torch.pi * torch.rand(n, 1) - 1
    x = (r * torch.cos(theta)).reshape(-1, 1)
    y = (r * torch.sin(theta)).reshape(-1, 1)
    return torch.concat([x, y, z], dim=1)


def exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Map a tangent vector ``v`` at ``x`` onto ``S^2``.

    Contract: ``x`` and ``v`` have shape ``(n, 3)``; return shape ``(n, 3)``.
    For nonzero ``v``, use the great-circle rotation in the plane spanned by
    ``x`` and ``v``. The zero-vector case leaves ``x`` unchanged.
    """
    _check_point_pair(x, v)

    v_normalized = torch.nn.functional.normalize(v, dim=-1)
    v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    exp_x = torch.cos(v_norm) * x + torch.sin(v_norm) * v_normalized
    return exp_x


def log_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the shortest tangent vector at ``x`` pointing to ``y``.

    Contract: ``x`` and ``y`` have shape ``(n, 3)``; return shape ``(n, 3)``.
    The output is tangent at ``x`` and has norm equal to the spherical distance
    between each pair. Generic pairs exclude identical and antipodal points.
    """
    _check_point_pair(x, y)

    dot_prod = (x * y).sum(dim=1, keepdim=True)
    theta = torch.arccos(torch.clamp(dot_prod, -1, 1))
    y_parallel = dot_prod * x
    y_perp = y - y_parallel
    v_normalized = torch.nn.functional.normalize(y_perp, dim=-1)
    v = theta * v_normalized
    return v


def premetric_d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return row-wise spherical premetric distances with shape ``(n,)``.

    Target equation: d(x, y) = arccos(clip(x dot y, -1, 1)).
    """
    _check_point_pair(x, y)

    return torch.arccos(torch.clamp((x * y).sum(dim=1), -1, 1))


def grad_d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the gradient of ``d(x, y)`` with respect to ``x``.

    Contract: return a tangent tensor of shape ``(n, 3)``. For generic pairs,
    each row has unit norm and points opposite to ``log_map(x, y)``.
    """
    _check_point_pair(x, y)

    v = log_map(x, y)
    v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return -v / v_norm


def kappa(t: torch.Tensor) -> torch.Tensor:
    """Return the RFM schedule ``kappa(t)`` for ``t`` in ``[0, 1]``."""
    if t.ndim != 1:
        raise ValueError("t must have shape (n,)")
    if not torch.isfinite(t).all() or not torch.all((0 <= t) & (t <= 1)):
        raise ValueError("t must be finite and lie in [0, 1]")
    return 1 - t


def dlog_kappa_dt(t: torch.Tensor) -> torch.Tensor:
    """Return the time derivative of ``log(kappa(t))``."""
    if t.ndim != 1:
        raise ValueError("t must have shape (n,)")
    if not torch.isfinite(t).all() or not torch.all((0 <= t) & (t < 1)):
        raise ValueError("t must be finite and lie in [0, 1)")
    return -torch.reciprocal(1 - t)


def conditional_vf(x: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return the conditional velocity field from RFM Eq. 13.

    Contract: ``x`` and ``x1`` are ``(n, 3)``; ``t`` is ``(n,)``; return is
    tangent at ``x`` with shape ``(n, 3)``.
    """
    _check_point_pair(x, x1)
    if t.ndim != 1 or t.shape[0] != x.shape[0]:
        raise ValueError("t must have shape (n,) matching x")
    if not torch.isfinite(t).all() or not torch.all((0 <= t) & (t < 1)):
        raise ValueError("t must be finite and lie in [0, 1)")
    logk_deriv = dlog_kappa_dt(t).reshape(-1, 1)
    d = premetric_d(x, x1).reshape(-1, 1)
    grad = grad_d(x, x1)
    grad_norm_sq = torch.square(torch.linalg.vector_norm(grad, dim=1, keepdim=True))
    return logk_deriv * d * grad * torch.reciprocal(grad_norm_sq)


def geodesic_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return the closed-form sphere path from ``x0`` to ``x1``.

    Contract: ``x0`` and ``x1`` are ``(n, 3)``; ``t`` is ``(n,)``; return is
    ``(n, 3)`` and satisfies Eq. 12 for ``kappa(t)=1-t``.
    """
    _check_point_pair(x0, x1)
    if t.ndim != 1 or t.shape[0] != x0.shape[0]:
        raise ValueError("t must have shape (n,) matching x0")
    if not torch.isfinite(t).all() or not torch.all((0 <= t) & (t <= 1)):
        raise ValueError("t must be finite and lie in [0, 1]")

    return exp_map(x0, t.reshape(-1, 1) * log_map(x0, x1))
