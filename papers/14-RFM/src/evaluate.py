"""Sampling and evaluation harness contracts for the RFM reproduction.

The ODE solver, sampling implementation, distribution metric, and likelihood
calculation are intentionally left for the learner to implement.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import nn

VelocityField: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Document the values returned by :func:`evaluate`."""

    samples: torch.Tensor
    metric: float | None
    nll: float


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

    # TODO(luv): implement the chosen fixed-step ODE solver and evaluate the
    # velocity field at each time while retaining the (n, 3) state shape.
    raise NotImplementedError


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

    Sampling is explicitly outside autograd.  The future implementation will
    resolve ``device``, wrap ``model`` as a velocity field, and delegate to
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
        # TODO(luv): construct the model-backed velocity field and call
        # integrate_ode; do not re-enable autograd in this sampling path.
        raise NotImplementedError


def compute_metric(samples: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute the chosen sample-vs-reference metric.

    The metric contract is a finite scalar; its definition belongs to the RFM
    experiment rather than this harness.
    """
    _check_points(samples, "samples")
    _check_points(reference, "reference")

    # TODO(luv): implement the selected distribution metric and return a
    # finite scalar for comparison with the uniform-sphere baseline.
    raise NotImplementedError


def negative_log_likelihood(model: nn.Module, samples: torch.Tensor) -> float:
    """Estimate the model's sample likelihood as a finite scalar NLL."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    _check_points(samples, "samples")

    # TODO(luv): implement the RFM likelihood/NLL calculation for the learned
    # flow, including any required trace or divergence estimate.
    raise NotImplementedError


def evaluate(
    model: nn.Module,
    initial_state: torch.Tensor,
    reference: torch.Tensor,
    *,
    metric: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
    t0: float = 0.0,
    t1: float = 1.0,
    steps: int = 100,
    device: str | torch.device | None = None,
) -> EvaluationResult:
    """Sample and score a learned flow against reference sphere points.

    Args:
        model: Learned velocity-field model.
        initial_state: Base samples with shape ``(n, 3)``.
        reference: Held-out reference points with shape ``(n, 3)``.
        metric: Optional metric callable; ``compute_metric`` is the default
            contract when this is ``None``.
        t0: Initial ODE time.
        t1: Final ODE time.
        steps: Number of solver steps reserved for ODE integration.
        device: Device on which sampling will run.

    Returns:
        An :class:`EvaluationResult` containing generated samples, the chosen
        metric, and the model NLL.

    Raises:
        NotImplementedError: Until sampling, metric, and NLL bodies are
        implemented by the learner.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    _check_points(initial_state, "initial_state")
    _check_points(reference, "reference")
    if metric is not None and not callable(metric):
        raise TypeError("metric must be callable or None")
    _check_time_interval(t0, t1, steps)

    # TODO(luv): call sample(), compute the selected metric, and compute NLL;
    # return all three values in the documented EvaluationResult structure.
    raise NotImplementedError
