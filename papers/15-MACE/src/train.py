"""Training contract for the 3BPA MACE reproduction."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from model import MACEConfig
from torch import Tensor


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
    seed: int = 123


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
