"""RCFM loss contract for the RFM reproduction."""

import torch


def _check_velocity_batch(value: torch.Tensor, name: str) -> None:
    """Validate the non-mathematical part of a velocity batch contract."""
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")


def rcfm_loss(
    predicted_velocity: torch.Tensor, conditional_target: torch.Tensor
) -> torch.Tensor:
    """Return the scalar RCFM loss for predicted and conditional velocities.

    Contract: both inputs are finite tensors of shape ``(n, 3)`` with matching
    batch sizes; return one scalar tensor.
    """
    _check_velocity_batch(predicted_velocity, "predicted_velocity")
    _check_velocity_batch(conditional_target, "conditional_target")
    if predicted_velocity.shape != conditional_target.shape:
        raise ValueError(
            "predicted_velocity and conditional_target must have the same shape"
        )

    # TODO(luv): implement the RCFM loss reduction from paper Eq. 10.
    raise NotImplementedError
