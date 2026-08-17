"""Sampling and exact likelihood evaluation for the RFM reproduction."""

import math
from collections.abc import Callable
from typing import TypeAlias

import torch
from torch import nn
from torch.func import jacrev, vmap
from torchdiffeq import odeint

VelocityField: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _check_points(points: torch.Tensor, name: str) -> None:
    """Validate a batch of Cartesian unit-sphere-shaped state vectors."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if points.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    if not torch.isfinite(points).all():
        raise ValueError(f"{name} must contain finite values")


def _check_time_interval(t0: float, t1: float, steps: int) -> None:
    """Validate the scalar integration interval and step count."""
    if not all(torch.isfinite(torch.tensor(value)) for value in (t0, t1)):
        raise ValueError("t0 and t1 must be finite")
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")


@torch.no_grad()
def integrate_ode(
    velocity_field: VelocityField,
    initial_state: torch.Tensor,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    steps: int = 100,
) -> torch.Tensor:
    """Integrate an RFM velocity field without constructing an autograd graph.

    The solver must preserve the leading batch shape and return a tensor with
    the same shape as ``initial_state``.  The ``no_grad`` decorator is part of
    this contract: sampling must not backpropagate through ODE integration.
    """
    if not callable(velocity_field):
        raise TypeError("velocity_field must be callable")
    _check_points(initial_state, "initial_state")
    _check_time_interval(t0, t1, steps)

    state = torch.clone(initial_state)
    dt = (t1 - t0) / steps
    current_time = torch.tensor(t0).to(device=state.device, dtype=state.dtype)
    for _ in range(steps):
        velocity = velocity_field(state, current_time)
        state += dt * velocity
        current_time += dt
    return state


@torch.no_grad()
def sample(
    model: nn.Module,
    initial_state: torch.Tensor,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    steps: int = 100,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Sample from the learned flow starting from ``initial_state``.

    Sampling is explicitly outside autograd.  The model is moved to the
    requested device, wrapped as a velocity field, and passed to
    :func:`integrate_ode`.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    _check_points(initial_state, "initial_state")
    _check_time_interval(t0, t1, steps)
    try:
        torch.device("cpu" if device is None else device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid device: {device!r}") from exc

    with torch.no_grad():
        initial_state = initial_state.to(device=device)
        model = model.to(device=device)

        def velocity_model(x, t):
            t_series = t.repeat(x.shape[0])
            return model(x, t_series)

        return integrate_ode(
            velocity_field=velocity_model,
            initial_state=initial_state,
            t0=t0,
            t1=t1,
            steps=steps,
        )


def model_one_point(
    model: nn.Module,
    x_single: torch.Tensor,
    t_single: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a batched velocity model for one spatial point and time."""
    return model(
        x_single.unsqueeze(0),
        t_single.unsqueeze(0),
    ).reshape(-1)


def div_fn(
    u: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a scalar divergence function from a vector-valued field."""
    J = jacrev(u)
    return lambda x: torch.trace(J(x))


def divergence(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Return one spatial-divergence value per point.

    Args:
        model: Learned time-conditioned velocity field.
        x: Sphere points with shape ``(n, 3)``.
        t: Per-point times with shape ``(n,)``.

    Returns:
        A tensor with shape ``(n,)`` containing the Jacobian trace for each
        point.
    """

    def model_helper(x_single, t_single):
        def u(z):
            return model_one_point(model, z, t_single)

        J = jacrev(u)(x_single)
        return torch.trace(J)

    return vmap(model_helper)(x, t)


def negative_log_likelihood(
    model: nn.Module,
    samples: torch.Tensor,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-7,
) -> float:
    """Compute exact test NLL with the continuous change-of-variables formula.

    The base distribution is the uniform density on the unit sphere, and the
    divergence accumulator supplies the change-of-variables correction. The
    augmented state is integrated backward from data time ``1`` to base time
    ``0`` using the paper's adaptive ``dopri5`` solver.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    _check_points(samples, "samples")
    if not math.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be a positive finite number")
    if not math.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be a positive finite number")

    accumulator_at_data = torch.zeros_like(samples[:, :1])
    state_at_data = torch.cat((samples, accumulator_at_data), dim=1)

    def augmented_dynamics(t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return d/dt of the augmented [position, log-density] state."""
        x = state[:, :3]
        times = t.expand(x.shape[0])
        velocity = model(x, times)
        spatial_divergence = divergence(model, x, times).unsqueeze(1)

        return torch.concat([velocity, -spatial_divergence], dim=1)

    integration_times = samples.new_tensor([1.0, 0.0])
    trajectory = odeint(
        augmented_dynamics,
        state_at_data,
        integration_times,
        rtol=rtol,
        atol=atol,
        method="dopri5",
        options={"min_step": 1e-5},
    )

    state_at_base = trajectory[-1]
    log_density_correction = state_at_base[:, 3]
    uniform_log_density = -samples.new_tensor(4.0 * math.pi).log()

    return -float(uniform_log_density - log_density_correction.mean())
