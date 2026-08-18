"""Lightweight training harness for the RFM reproduction."""

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypedDict

import geometry as geom
import loss as l
import model as m
import torch
from torch import nn

import data as d
from gdl import log_metrics, new_run_dir, save_checkpoint, set_seed

DataSource: TypeAlias = torch.Tensor
LossCallable: TypeAlias = Callable[..., torch.Tensor]
PAPER_DIR = Path(__file__).resolve().parent.parent
METRIC = "nll"


class AdamSettings(TypedDict, total=False):
    """Supported keyword settings for the Adam optimizer."""

    lr: float | torch.Tensor
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    foreach: bool | None
    maximize: bool
    capturable: bool
    differentiable: bool
    fused: bool | None


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
    """Check one tensor data source."""
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if data.ndim == 0 or data.shape[0] == 0:
        raise ValueError(f"{name} tensor must have a non-empty leading dimension")
    if not torch.isfinite(data).all():
        raise ValueError(f"{name} tensor must contain finite values")


def _check_options(
    model: nn.Module,
    loss_fn: LossCallable,
    optimizer_settings: AdamSettings | None,
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
    optimizer_settings: AdamSettings | None = None,
    epochs: int = 1,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> TrainingResult:
    """Train ``model`` against tensor RFM data.

    Args:
        train_data: Training points as a tensor.
        validation_data: Validation points as a tensor.
        model: The velocity-field model to optimize.
        loss_fn: Callable used to produce a scalar training loss.
        optimizer_settings: Keyword settings reserved for the future
            optimizer construction, for example ``{"lr": 1e-3}``.
        epochs: Number of optimization epochs to run.
        seed: Optional non-negative seed for reproducibility.
        device: Device on which the implementation will train.

    Returns:
        A :class:`TrainingResult` containing the model, per-epoch losses, and
        the resolved run options.

    Raises:
        TypeError: If a data source, model, loss, or option has the wrong type.
        ValueError: If an input or option violates its contract.
    """
    _check_data_source(train_data, "train_data")
    _check_data_source(validation_data, "validation_data")
    _resolved_device = _check_options(
        model, loss_fn, optimizer_settings, epochs, seed, device
    )
    if seed is not None:
        torch.manual_seed(seed)
    x1 = train_data.to(_resolved_device)
    val = validation_data.to(_resolved_device)
    n_val = len(val)
    if optimizer_settings is None:
        optimizer_settings = dict()
    model = model.to(_resolved_device)
    optim = torch.optim.Adam(model.parameters(), **optimizer_settings)
    n = len(train_data)
    losses = []
    val_losses = []
    for _ in range(epochs):
        x0 = geom.sample_sphere(n).to(_resolved_device)
        t = 1e-3 + (1 - 2e-3) * torch.rand(n, device=_resolved_device)
        xt = geom.geodesic_path(x0, x1, t)
        cvf = geom.conditional_vf(xt, x1, t)
        ut = model(xt, t)
        loss = loss_fn(ut, cvf)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        with torch.no_grad():
            # Validation Losses
            x0_val = geom.sample_sphere(n_val).to(_resolved_device)
            t_val = 1e-3 + (1 - 2e-3) * torch.rand(n_val, device=_resolved_device)
            xt_val = geom.geodesic_path(x0_val, val, t_val)
            cvf_val = geom.conditional_vf(xt_val, val, t_val)
            ut_val = model(xt_val, t_val)
            val_loss = loss_fn(ut_val, cvf_val)
            val_losses.append(val_loss.item())

    assert len(losses) == epochs
    assert len(val_losses) == epochs

    return TrainingResult(
        model=model,
        device=_resolved_device,
        train_loss=tuple(losses),
        validation_loss=tuple(val_losses),
        epochs=epochs,
        seed=seed,
    )


def train_run(
    data_path: str | Path,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    n_layers: int = 10,
    seed: int = 42,
    train_pct: float = 80.0,
    val_pct: float = 10.0,
    test_pct: float = 10.0,
    device: str | torch.device = "cpu",
    runs_base: str = "runs",
    paper_dir: str | Path = PAPER_DIR,
) -> tuple[str, Path, TrainingResult]:
    """Train Fire RFM and persist its reproducible run artifacts."""
    resolved_data_path = Path(data_path).resolve()
    points = d.load_fire_csv(resolved_data_path)
    train_data, validation_data, test_data = d.split_points(
        points,
        seed,
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
    )
    if any(
        len(partition) == 0 for partition in (train_data, validation_data, test_data)
    ):
        raise ValueError(
            "split must produce non-empty train, validation, and test data"
        )

    set_seed(seed)
    model = m.TimeConditionedVectorField(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )
    config = {
        "data": {
            "dataset": "fire",
            "path": str(resolved_data_path),
            "split": [train_pct, val_pct, test_pct],
        },
        "model": {"hidden_dim": hidden_dim, "n_layers": n_layers},
        "hyperparams": {"lr": lr, "epochs": epochs},
        "seed": seed,
        "metric": METRIC,
    }
    run_id, run_directory = new_run_dir(paper_dir, config, base=runs_base)

    result = train(
        train_data=train_data,
        validation_data=validation_data,
        model=model,
        loss_fn=l.rcfm_loss,
        optimizer_settings={"lr": lr},
        epochs=epochs,
        seed=seed,
        device=device,
    )
    for epoch, (train_loss, validation_loss) in enumerate(
        zip(result.train_loss, result.validation_loss)
    ):
        log_metrics(
            {
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            },
            step=epoch,
            run_dir=run_directory,
        )
    save_checkpoint(result.model, run_directory)
    return run_id, run_directory, result


def main():
    """Train Fire RFM from CLI and print the persisted run ID."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--data-path", type=Path, default=PAPER_DIR / "data" / "fire.csv"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-pct", type=float, default=80.0)
    parser.add_argument("--val-pct", type=float, default=10.0)
    parser.add_argument("--test-pct", type=float, default=10.0)
    parser.add_argument("--runs-base", type=str, default="runs")
    args = parser.parse_args()

    run_id, _, _ = train_run(
        data_path=args.data_path,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        seed=args.seed,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        device=args.device,
        runs_base=args.runs_base,
    )
    print(f"run_id: {run_id}")


if __name__ == "__main__":
    main()
