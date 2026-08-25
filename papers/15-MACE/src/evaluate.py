"""Evaluation contract for the four 3BPA test conditions."""

import argparse
from pathlib import Path

from torch import Tensor

REFERENCE_RMSE_MEV = {
    "300K": {"energy": 3.0, "forces": 8.8},
    "600K": {"energy": 9.7, "forces": 21.8},
    "1200K": {"energy": 29.8, "forces": 62.0},
    "dihedral": {"energy": 7.8, "forces": 16.5},
}


def root_mean_square_error(prediction: Tensor, target: Tensor) -> float:
    """Compute one scalar RMSE using an explicit component-wise convention.

    TODO(luv): verify energy and force unit conversion separately; force RMSE is
    over Cartesian components, not norms per atom.
    """
    raise NotImplementedError


def evaluate(
    run_id: str, *, data_dir: Path, paper_dir: Path | None = None
) -> dict[str, dict[str, float]]:
    """Evaluate one checkpoint on 300K/600K/1200K/pooled-dihedral subsets.

    TODO(luv): reconstruct the model from config.json, load the checkpoint, and
    return energy/force RMSE in meV and meV/angstrom for each subset.
    """
    _ = run_id, data_dir, paper_dir
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run, data_dir=args.data)


if __name__ == "__main__":
    main()
