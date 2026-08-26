"""Training contract for the 3BPA MACE reproduction."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
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
    residual_energy_per_atom = []
    force_components = []
    for graph in graphs:
        reference = atomic_energies.to(
            dtype=graph.node_attrs.dtype, device=graph.node_attrs.device
        )
        graph_reference_energy = (graph.node_attrs @ reference).sum()
        residual_energy_per_atom.append(
            (graph.energy - graph_reference_energy) / graph.num_nodes
        )
        force_components.append(graph.forces.reshape(-1))

    mean = torch.stack(residual_energy_per_atom).mean()
    rms = torch.cat(force_components).square().mean().sqrt()
    return float(mean), float(rms)


def compute_average_num_neighbors(graphs: Sequence[Data]) -> float:
    """Return the paper-era mean receiver degree over the training graphs."""
    receiver_counts = []
    for graph in graphs:
        receiver = graph.edge_index[1]
        _, counts = torch.unique(receiver, return_counts=True)
        receiver_counts.append(counts)
    return float(torch.cat(receiver_counts).to(dtype=torch.float64).mean())


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
