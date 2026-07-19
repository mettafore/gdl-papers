"""Evaluate a saved run.

Run by path:  uv run python papers/00-template/src/evaluate.py --run <run_id>
Reads the run's config.json, rebuilds the model, loads the checkpoint, and
reports the metric on the SAME held-out split training used.
"""

import argparse
from pathlib import Path

import torch

import data
from model import MLP
from gdl import load_run_config, load_checkpoint, METRICS, run_dir

PAPER_DIR = Path(__file__).parent.parent  # runs/ lives at the paper root, not in src/


def evaluate(run_id, runs_base="runs", paper_dir=PAPER_DIR):
    """Score a saved run on its held-out split. Returns (metric_name, score)."""
    cfg = load_run_config(paper_dir, run_id, base=runs_base)

    model = MLP(
        in_dim=cfg["data"]["in_dim"],
        hidden=cfg["model"]["hidden"],
        out_dim=cfg["data"]["out_dim"],
    )
    rdir = run_dir(paper_dir, run_id, base=runs_base)
    load_checkpoint(model, rdir)
    model.eval()

    _, _, X_test, y_test, _, _ = data.load_split(
        seed=cfg["seed"], test_size=cfg["data"]["test_size"]
    )

    metric_fn = METRICS[cfg["metric"]]
    with torch.no_grad():
        score = metric_fn(model(X_test), y_test)
    return cfg["metric"], score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run_id under runs/")
    p.add_argument("--runs-base", type=str, default="runs")
    args = p.parse_args()

    metric_name, score = evaluate(args.run, runs_base=args.runs_base)
    print(f"{metric_name}: {score:.4f}")


if __name__ == "__main__":
    main()
