"""Evaluation contract for the four 3BPA test conditions."""

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
from model import MACE, MACEConfig
from torch import Tensor, nn
from torch_geometric.data import Data
from train import (
    get_module_device,
    resolve_device,
)

import data
from gdl import load_checkpoint, load_run_config, run_dir

REFERENCE_RMSE_MEV = {
    "300K": {"energy": 3.0, "forces": 8.8},
    "600K": {"energy": 9.7, "forces": 21.8},
    "1200K": {"energy": 29.8, "forces": 62.0},
    "dihedral": {"energy": 7.8, "forces": 16.5},
}


def root_mean_square_error(prediction: Tensor, target: Tensor) -> float:
    """Compute one scalar RMSE; force tensors are reduced over XYZ components."""
    return (prediction - target).square().mean().sqrt().item()


def _evaluate_graphs(
    model: nn.Module,
    graphs: Sequence[Data],
    *,
    batch_size: int,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Return energy and force-component RMSEs in meV-based units."""
    if not graphs:
        raise ValueError("cannot evaluate an empty test set")

    loader = data.get_data_loader(graphs, batch_size=batch_size, shuffle=False)
    predicted_energies = []
    target_energies = []
    predicted_forces = []
    target_forces = []
    device = get_module_device(model) if device is None else resolve_device(device)

    model.eval()
    # Forces are energy derivatives, so evaluation must keep autograd enabled
    # for positions even though we do not build a training graph.
    with torch.enable_grad():
        for batch in loader:
            batch = batch.to(device)
            energy, forces = model(
                batch.node_attrs.double(),
                batch.positions,
                batch.edge_index,
                batch.batch,
            )
            predicted_energies.append(energy.detach().cpu())
            target_energies.append(batch.energy.detach().cpu())
            predicted_forces.append(forces.detach().cpu())
            target_forces.append(batch.forces.detach().cpu())

    energy_rmse = root_mean_square_error(
        torch.cat(predicted_energies), torch.cat(target_energies)
    )
    force_rmse = root_mean_square_error(
        torch.cat(predicted_forces), torch.cat(target_forces)
    )
    # Dataset labels and model predictions are in eV and eV/angstrom.
    return {"energy": 1000.0 * energy_rmse, "forces": 1000.0 * force_rmse}


def evaluate(
    run_id: str,
    *,
    data_dir: Path,
    paper_dir: Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate one checkpoint on 300K/600K/1200K/pooled-dihedral subsets.

    Returns energy RMSE in meV and force-component RMSE in meV/angstrom.
    """
    torch.set_default_dtype(torch.float64)
    data.validate_data_dir(data_dir)
    device = resolve_device(device)
    if paper_dir is None:
        paper_dir = Path(__file__).resolve().parent.parent

    config = load_run_config(paper_dir, run_id)
    data_config = config["data"]
    model_values = dict(config["model"])
    model_values["radial_mlp_widths"] = tuple(model_values["radial_mlp_widths"])
    model_config = MACEConfig(**model_values)

    model = MACE(
        model_config,
        avg_num_neighbours=data_config["avg_num_neighbors"],
        atomic_energies=torch.tensor(
            data_config["atomic_energies"], dtype=torch.float64
        ),
        mean_energy_shift=data_config["mean_energy_shift"],
        rms_forces=data_config["rms_forces"],
    )
    model = model.double().to(device)
    load_checkpoint(model, run_dir(paper_dir, run_id))

    batch_size = config["hyperparams"]["batch_size"]
    results = {}
    for test_name, filenames in data.TEST_FILES.items():
        configurations = [
            configuration
            for filename in filenames
            for configuration in data.load_extxyz(data_dir / filename)
        ]
        graphs = [
            data.build_neighbor_graph(item, cutoff=model_config.cutoff)
            for item in configurations
        ]
        results[test_name] = _evaluate_graphs(
            model,
            graphs,
            batch_size=batch_size,
            device=device,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    results = evaluate(args.run, data_dir=args.data, device=args.device)
    print(
        json.dumps(
            {"rmse_mev": results, "paper_reference_mev": REFERENCE_RMSE_MEV}, indent=2
        )
    )


if __name__ == "__main__":
    main()
