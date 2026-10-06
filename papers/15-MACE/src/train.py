"""Training contract for the 3BPA MACE reproduction."""

import argparse
import json
import math
import os
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from model import MACE, MACEConfig
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel
from torch_geometric.data import Data

import data
from gdl import (
    load_run_config,
    log_metrics,
    new_run_dir,
    set_seed,
)
from gdl.run import run_dir as get_run_dir


@dataclass(frozen=True)
class TrainConfig:
    """Published optimization defaults plus run identity."""

    learning_rate: float = 0.01
    batch_size: int = 5
    energy_weight: float = 1.0
    force_weight: float = 1000.0
    scheduler_patience: int = 50
    scheduler_factor: float = 0.8
    ema_decay: float = 0.99
    weight_decay: float = 5e-7
    amsgrad: bool = True
    drop_last: bool = True
    seed: int = 123
    max_epochs: int = 500
    max_train_batches: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("batch_size and max_epochs must be positive")
        if self.max_train_batches is not None and self.max_train_batches < 1:
            raise ValueError("max_train_batches must be positive when set")


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Use CUDA when available by default; this float64 model supports CPU/CUDA."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        resolved_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'") from error

    if resolved_device.type == "mps":
        raise ValueError("MPS is not supported because this model uses float64")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in this PyTorch environment")
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    return resolved_device


def get_module_device(module: nn.Module) -> torch.device:
    """Find a module's device, including buffer-only modules used in evaluation."""
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _capture_rng_state() -> dict[str, Any]:
    """Capture global RNGs so the next shuffled epoch can continue reproducibly."""
    numpy_name, numpy_keys, numpy_position, has_gauss, cached_gaussian = (
        np.random.get_state()
    )
    numpy_keys_array = np.asarray(numpy_keys, dtype=np.uint32)
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_name,
            torch.from_numpy(numpy_keys_array.astype(np.int64)),
            numpy_position,
            has_gauss,
            cached_gaussian,
        ),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state after model and optimizer construction on resume."""
    random.setstate(state["python"])
    numpy_name, numpy_keys, numpy_position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state(
        (
            numpy_name,
            numpy_keys.cpu().numpy().astype(np.uint32, copy=False),
            numpy_position,
            has_gauss,
            cached_gaussian,
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_training_state(
    *,
    path: Path,
    model: nn.Module,
    ema_model: AveragedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_epochs: int,
    best_validation_loss: float,
    best_epoch: int,
    best_ema_state: dict[str, Tensor],
) -> None:
    """Atomically save all state required to continue after an epoch boundary."""
    state = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "completed_epochs": completed_epochs,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "best_ema_model": best_ema_state,
        "rng": _capture_rng_state(),
    }
    _atomic_torch_save(state, path)


def _atomic_torch_save(state: object, path: Path) -> None:
    """Replace a checkpoint atomically so interruption leaves the prior file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_training_state(
    *,
    path: Path,
    model: nn.Module,
    ema_model: AveragedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, float, int, dict[str, Tensor]]:
    """Restore training state and return completed epochs and best-score data."""
    if not path.is_file():
        raise FileNotFoundError(f"no resumable training state at {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    ema_model.load_state_dict(state["ema_model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    _restore_rng_state(state["rng"])
    return (
        int(state["completed_epochs"]),
        float(state["best_validation_loss"]),
        int(state["best_epoch"]),
        state["best_ema_model"],
    )


def _reconcile_run_files(
    run_path: Path, completed_epochs: int, best_ema_state: dict[str, Tensor]
) -> None:
    """Restore derived files to the epoch recorded in training_state.pt."""
    metrics_path = run_path / "metrics.jsonl"
    if metrics_path.exists():
        original_contents = metrics_path.read_text()
        kept_lines = []
        for line in original_contents.splitlines(keepends=True):
            if not line.endswith("\n"):
                break  # An interrupted append may leave an incomplete final line.
            if int(json.loads(line)["step"]) < completed_epochs:
                kept_lines.append(line)
        contents = "".join(kept_lines)
        if contents != original_contents:
            temporary_path = metrics_path.with_suffix(".jsonl.tmp")
            temporary_path.write_text(contents)
            os.replace(temporary_path, metrics_path)
    _atomic_torch_save(best_ema_state, run_path / "checkpoint.pt")


def _validate_resume_config(
    saved_config: dict,
    current_config: dict,
) -> None:
    """Reject resuming a run with changed data, model, or training settings."""
    for section in ("data", "model"):
        current_section = json.loads(json.dumps(current_config[section]))
        if saved_config[section] != current_section:
            raise ValueError(f"cannot resume: saved {section} config differs")

    saved_training = dict(saved_config["hyperparams"])
    current_training = dict(current_config["hyperparams"])
    saved_training.pop("max_epochs")
    current_training.pop("max_epochs")
    if saved_training != current_training:
        raise ValueError("cannot resume: training hyperparameters differ")


def compute_mean_rms_energy_forces(
    graphs: Sequence[Data], atomic_energies: Tensor
) -> tuple[float, float]:
    """Compute the paper's residual-energy shift and force-component RMS scale."""
    graph_dft_energies = torch.tensor([graph.energy for graph in graphs])
    atomic_number_dict = {x: idx for idx, x in enumerate([1, 6, 7, 8])}
    graph_iso_energies = torch.tensor(
        [
            sum(
                list(
                    map(
                        lambda x: atomic_energies[atomic_number_dict.get(int(x))],
                        graph.atomic_numbers,
                    )
                )
            )
            for graph in graphs
        ]
    )
    n_atoms = torch.tensor([len(graph.atomic_numbers) for graph in graphs])

    all_force_components = torch.cat([graph.forces.reshape(-1) for graph in graphs])

    rmse_forces = torch.sqrt((all_force_components**2).mean())

    residual_energy_shift = ((graph_dft_energies - graph_iso_energies) / n_atoms).mean()
    return residual_energy_shift.item(), rmse_forces.item()


def compute_average_num_neighbors(graphs: Sequence[Data]) -> float:
    """Return the paper-era mean receiver degree over the training graphs."""
    receiver_counts = [
        torch.unique(graph.edge_index[1], return_counts=True)[1] for graph in graphs
    ]
    return torch.cat(receiver_counts).to(dtype=torch.get_default_dtype()).mean().item()


def energy_force_loss(
    predicted_energy: Tensor,
    predicted_forces: Tensor,
    target_energy: Tensor,
    target_forces: Tensor,
    *,
    energy_weight: float = 1.0,
    force_weight: float = 1000.0,
) -> Tensor:
    """Return energy MSE over graphs plus force MSE over all XYZ components."""
    return (
        energy_weight * ((predicted_energy - target_energy) ** 2).mean()
        + (force_weight * (predicted_forces - target_forces) ** 2).mean()
    )


def make_ema_model(model: nn.Module, *, decay: float) -> AveragedModel:
    """Create the paper's EMA copy of the live model parameters."""

    def ema_average(average: Tensor, current: Tensor, _: int) -> Tensor:
        return decay * average + (1.0 - decay) * current

    return AveragedModel(
        model,
        avg_fn=ema_average,
        device=get_module_device(model),
    )


def compute_validation_loss(
    model: nn.Module,
    valid_loader: Iterable[Data],
    *,
    energy_weight: float,
    force_weight: float,
) -> float:
    """Evaluate energy/force loss for the provided model, including EMA copies."""
    model.eval()
    validation_losses = []
    device = get_module_device(model)

    for batch in valid_loader:
        batch = batch.to(device)
        predicted_energy, predicted_forces = model(
            batch.node_attrs.double(),
            batch.positions,
            batch.edge_index,
            batch.batch,
        )
        loss = energy_force_loss(
            predicted_energy,
            predicted_forces,
            batch.energy,
            batch.forces,
            energy_weight=energy_weight,
            force_weight=force_weight,
        )
        validation_losses.append(loss.detach())

    return torch.stack(validation_losses).mean().item()


def train(
    *,
    data_dir: Path,
    train_config: TrainConfig = TrainConfig(),
    model_config: MACEConfig = MACEConfig(),
    paper_dir: Path | None = None,
    device: str | torch.device | None = None,
    resume_run_id: str | None = None,
    on_epoch_checkpoint: Callable[[], None] | None = None,
) -> tuple[str, Path]:
    """Train one seed and write config, JSONL metrics, and the best EMA checkpoint."""
    data_dir = data_dir.resolve()
    device = resolve_device(device)
    set_seed(train_config.seed)
    configurations = data.load_extxyz(data_dir / data.TRAIN_FILE)
    train_configurations, valid_configurations = data.split_train_valid(
        configurations, seed=train_config.seed
    )

    train_graphs = [
        data.build_neighbor_graph(record) for record in train_configurations
    ]

    valid_graphs = [
        data.build_neighbor_graph(record) for record in valid_configurations
    ]

    train_loader = data.get_data_loader(
        dataset=train_graphs,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=train_config.drop_last,
    )

    valid_loader = data.get_data_loader(
        dataset=valid_graphs,
        batch_size=train_config.batch_size,
        shuffle=False,
    )

    atomic_energies = data.load_isolated_atom_energies(
        data_dir / data.ISOLATED_ATOMS_FILE
    )

    mean_energy_shift, rms_forces = compute_mean_rms_energy_forces(
        train_graphs, atomic_energies
    )

    avg_num_neighbors = compute_average_num_neighbors(train_graphs)

    # Save the settings and training-data statistics needed to rebuild this run.
    run_config = {
        "data": {
            "dataset": "3BPA",
            "data_dir": str(data_dir),
            "atomic_energies": atomic_energies.tolist(),
            "mean_energy_shift": mean_energy_shift,
            "rms_forces": rms_forces,
            "avg_num_neighbors": avg_num_neighbors,
        },
        "model": asdict(model_config),
        "hyperparams": asdict(train_config),
        "seed": train_config.seed,
        "metric": "rmse",
        "runtime": {"device": str(device)},
    }
    if paper_dir is None:
        paper_dir = Path(__file__).resolve().parent.parent
    if resume_run_id is None:
        run_id, run_path = new_run_dir(paper_dir, run_config)
    else:
        run_id = resume_run_id
        run_path = get_run_dir(paper_dir, run_id)
        saved_config = load_run_config(paper_dir, run_id)
        _validate_resume_config(saved_config, run_config)
    print(f"run_id: {run_id}", flush=True)

    model = MACE(
        model_config,
        avg_num_neighbours=avg_num_neighbors,
        atomic_energies=atomic_energies,
        mean_energy_shift=mean_energy_shift,
        rms_forces=rms_forces,
    ).double()
    model = model.to(device)

    decay_parameters = []
    other_parameters = []

    for name, parameter in model.named_parameters():
        if ".symmetric_contraction.contraction" in name:
            decay_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    optimizer = torch.optim.Adam(
        [
            {
                "params": decay_parameters,
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": other_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=train_config.learning_rate,
        amsgrad=train_config.amsgrad,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=train_config.scheduler_factor,
        patience=train_config.scheduler_patience,
    )

    ema_model = make_ema_model(model, decay=train_config.ema_decay)
    best_validation_loss = float("inf")
    best_epoch = -1
    best_ema_state: dict[str, Tensor] = {}
    start_epoch = 0
    if resume_run_id is not None:
        start_epoch, best_validation_loss, best_epoch, best_ema_state = (
            _load_training_state(
                path=run_path / "training_state.pt",
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        )
        if start_epoch > train_config.max_epochs:
            raise ValueError(
                f"run already completed {start_epoch} epochs, exceeding requested "
                f"limit {train_config.max_epochs}"
            )
        _reconcile_run_files(run_path, start_epoch, best_ema_state)

    for epoch in range(start_epoch, train_config.max_epochs):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if (
                train_config.max_train_batches is not None
                and batch_index >= train_config.max_train_batches
            ):
                break
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            predicted_energy, predicted_forces = model(
                batch.node_attrs.double(),
                batch.positions,
                batch.edge_index,
                batch.batch,
            )

            loss = energy_force_loss(
                predicted_energy,
                predicted_forces,
                batch.energy,
                batch.forces,
                energy_weight=train_config.energy_weight,
                force_weight=train_config.force_weight,
            )

            loss.backward()
            optimizer.step()
            ema_model.update_parameters(model)

        validation_loss = compute_validation_loss(
            ema_model,
            valid_loader,
            energy_weight=train_config.energy_weight,
            force_weight=train_config.force_weight,
        )
        if not math.isfinite(validation_loss):
            raise RuntimeError(f"non-finite validation loss at epoch {epoch}")
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_ema_state = {
                name: value.detach().cpu().clone()
                for name, value in ema_model.module.state_dict().items()
            }
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(validation_loss)
        log_metrics(
            metrics={
                "validation_loss": validation_loss,
                "learning_rate": lr,
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
            },
            step=epoch,
            run_dir=run_path,
        )
        _save_training_state(
            path=run_path / "training_state.pt",
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            completed_epochs=epoch + 1,
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
            best_ema_state=best_ema_state,
        )
        if improved:
            _atomic_torch_save(best_ema_state, run_path / "checkpoint.pt")
        if on_epoch_checkpoint is not None:
            on_epoch_checkpoint()
    return run_id, run_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume-run-id")
    args = parser.parse_args()
    train_config = TrainConfig(
        seed=args.seed,
        max_epochs=args.epochs,
        max_train_batches=args.max_train_batches,
    )
    run_id, _ = train(
        data_dir=args.data,
        train_config=train_config,
        device=args.device,
        resume_run_id=args.resume_run_id,
    )


if __name__ == "__main__":
    main()
