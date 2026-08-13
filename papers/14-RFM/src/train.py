"""Lightweight training harness contracts for the RFM reproduction.

This module deliberately contains no training loop.  The learner will fill in
the optimization body after the data and geometry contracts are established.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import nn

DataSource: TypeAlias = torch.Tensor | Iterable[object]
LossCallable: TypeAlias = Callable[..., torch.Tensor]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Document the values returned by :func:`train`.

    ``train_loss`` and ``validation_loss`` hold one scalar per completed
    epoch.  The model is returned so a caller can continue with paper-specific
    checkpointing once the training body is implemented.
    """

    model: nn.Module
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    epochs: int
    seed: int | None
    device: torch.device


def _check_data_source(data: DataSource, name: str) -> None:
    """Check a tensor or loader-like iterable without consuming an iterable."""
    if isinstance(data, torch.Tensor):
        if data.ndim == 0 or data.shape[0] == 0:
            raise ValueError(f"{name} tensor must have a non-empty leading dimension")
        if not torch.isfinite(data).all():
            raise ValueError(f"{name} tensor must contain finite values")
        return
    if not isinstance(data, Iterable):
        raise TypeError(f"{name} must be a torch.Tensor or an iterable loader")


def _check_options(
    model: nn.Module,
    loss_fn: LossCallable,
    optimizer_settings: Mapping[str, object] | None,
    epochs: int,
    seed: int | None,
    device: str | torch.device | None,
) -> torch.device:
    """Validate options shared by the future training implementation."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    if optimizer_settings is not None and not isinstance(optimizer_settings, Mapping):
        raise TypeError("optimizer_settings must be a mapping or None")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative")

    try:
        return torch.device("cpu" if device is None else device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid device: {device!r}") from exc


def train(
    train_data: DataSource,
    validation_data: DataSource,
    model: nn.Module,
    loss_fn: LossCallable,
    *,
    optimizer_settings: Mapping[str, object] | None = None,
    epochs: int = 1,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> TrainingResult:
    """Train ``model`` against tensor or loader-like RFM data.

    Args:
        train_data: Training points as a tensor or an iterable data loader.
        validation_data: Validation points as a tensor or an iterable loader.
        model: The velocity-field model to optimize.
        loss_fn: Callable used to produce a scalar training loss.
        optimizer_settings: Keyword settings reserved for the future
            optimizer construction, for example ``{"lr": 1e-3}``.
        epochs: Number of optimization epochs to run.
        seed: Optional non-negative seed for reproducibility.
        device: Device on which the future implementation will train.

    Returns:
        A :class:`TrainingResult` containing the model, per-epoch losses, and
        the resolved run options.

    Raises:
        TypeError: If a data source, model, loss, or option has the wrong type.
        ValueError: If an input or option violates its contract.
        NotImplementedError: Until the learner implements the training loop.
    """
    _check_data_source(train_data, "train_data")
    _check_data_source(validation_data, "validation_data")
    _resolved_device = _check_options(
        model, loss_fn, optimizer_settings, epochs, seed, device
    )

    # TODO(luv): seed the run, move the model and batches to the resolved
    # device, and implement the train/validation optimization loop.
    raise NotImplementedError
