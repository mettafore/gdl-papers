"""Stateless metric functions + a registry.

All metrics take (output, target) and return a float, so a paper can select one
by name from config: ``METRICS[cfg["metric"]](output, target)``.
"""

import torch


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Classification accuracy. logits: (N, C), target: (N,) class indices."""
    preds = logits.argmax(dim=-1)
    return (preds == target).float().mean().item()


def mae(output: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error (regression)."""
    return (output - target).abs().mean().item()


def rmse(output: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error (regression)."""
    return ((output - target) ** 2).mean().sqrt().item()


METRICS = {
    "accuracy": accuracy,
    "mae": mae,
    "rmse": rmse,
}
