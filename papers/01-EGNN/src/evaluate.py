"""Evaluate a saved EGNN run on QM9's held-out test split.

Run by path:  uv run python papers/01-EGNN/src/evaluate.py --run <run_id>
Reads the run's config.json, rebuilds the model + the same seeded split,
loads the checkpoint, and reports MAE in the SAME units QM9 stores the
target in (see UNIT_NOTE below for how that compares to the paper's table).
"""

import argparse
from pathlib import Path

import torch

import data
from gdl import METRICS, load_checkpoint, load_run_config, run_dir
from model import EGNN
from train import raw_target_for

PAPER_DIR = Path(__file__).parent

# QM9 stores homo/lumo/gap/U0 in eV; the paper's Table 1 reports them in meV
# (eV * 1000). mu/alpha/Cv are already in the paper's units (D, a0^3,
# cal/mol/K) — no conversion needed for those.
EV_TO_MEV_TARGETS = {"homo", "lumo", "gap", "U0"}


def evaluate(run_id, runs_base="runs", paper_dir=PAPER_DIR, data_root="data/qm9"):
    """Score a saved run on its held-out test split. Returns (metric_name, score)."""
    cfg = load_run_config(paper_dir, run_id, base=runs_base)

    egnn = EGNN(
        in_node_nf=cfg["data"]["dims"]["in_node_dim"],
        hidden_nf=cfg["model"]["hidden_nf"],
        n_layers=cfg["model"]["n_layers"],
        attention=cfg["model"]["attention"],
    )
    rdir = run_dir(paper_dir, run_id, base=runs_base)
    load_checkpoint(egnn, rdir)
    egnn.eval()

    _, _, test_loader, norm, dims = data.load_split(
        target=cfg["data"]["target"],
        seed=cfg["seed"],
        root=data_root,
    )

    metric_fn = METRICS[cfg["metric"]]
    total_score = 0.0
    total_n = 0
    with torch.no_grad():
        for b in test_loader:
            out = egnn(b.x, b.pos, b.edge_index, batch=b.batch)
            pred = norm.denormalize(out)
            target = raw_target_for(b, dims["target_col"])
            total_score += metric_fn(pred, target) * b.num_graphs
            total_n += b.num_graphs
    return cfg["metric"], total_score / total_n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run_id under runs/")
    p.add_argument("--runs-base", type=str, default="runs")
    p.add_argument("--data-root", type=str, default="data/qm9")
    args = p.parse_args()

    cfg = load_run_config(PAPER_DIR, args.run, base=args.runs_base)
    target = cfg["data"]["target"]
    metric_name, score = evaluate(
        args.run,
        runs_base=args.runs_base,
        data_root=args.data_root,
    )
    print(f"{metric_name} ({target}): {score:.4f}")
    if target in EV_TO_MEV_TARGETS:
        print(
            f"{metric_name} ({target}, meV, matches paper Table 1 units): {score * 1000:.2f}"
        )


if __name__ == "__main__":
    main()
