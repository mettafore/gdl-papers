"""Training contract for the 3BPA MACE reproduction."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from model import MACEConfig
from torch import Tensor
from torch_geometric.data import Data


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


def compute_mean_rms_energy_forces(
    graphs: Sequence[Data], atomic_energies: Tensor
) -> tuple[float, float]:
    """Compute the paper's residual-energy shift and force-component RMS scale."""
    # TODO(luv): subtract the atomic baseline, average residual energy per atom,
    # and compute RMS over force components rather than force-vector magnitudes.
    raise NotImplementedError


def compute_average_num_neighbors(graphs: Sequence[Data]) -> float:
    """Return the paper-era mean receiver degree over the training graphs."""
    # TODO(luv): aggregate receiver degrees over training graphs and return the
    # scalar message-normalization constant.
    raise NotImplementedError


def energy_force_loss(
    predicted_energy: Tensor,
    predicted_forces: Tensor,
    target_energy: Tensor,
    target_forces: Tensor,
    *,
    energy_weight: float = 1.0,
    force_weight: float = 1000.0,
) -> Tensor:
    """Return the paper's weighted sum of energy and force MSE.

    TODO(luv): make the averaging convention explicit and test it on literal
    tensors before connecting it to the training loop.
    """
    raise NotImplementedError


def train(
    *,
    data_dir: Path,
    train_config: TrainConfig = TrainConfig(),
    model_config: MACEConfig = MACEConfig(),
    paper_dir: Path | None = None,
) -> tuple[str, Path]:
    """Train one seed and write config, JSONL metrics, and checkpoint.

    TODO(luv): reuse only the paper-agnostic `gdl` run/checkpoint helpers. Keep
    dataset, loss, optimizer, scheduler, EMA, and model logic in this paper.
    """
    _ = asdict(train_config), asdict(model_config), data_dir, paper_dir
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    train_config = TrainConfig(seed=args.seed)
    train(data_dir=args.data, train_config=train_config)


if __name__ == "__main__":
    main()
