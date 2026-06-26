"""Train the template MLP on Iris.

Run by path:  uv run python papers/00-template/train.py [--lr ... --epochs ...]
Creates a run folder first (records the full recipe), trains, logs per epoch,
saves a checkpoint, and prints the run_id to pass to evaluate.py.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

import data
from model import MLP
from gdl import set_seed, log_metrics, new_run_dir, save_checkpoint

PAPER_DIR = Path(__file__).parent
TEST_SIZE = 0.2
METRIC = "accuracy"


def train(lr=1e-2, epochs=100, hidden=16, seed=42, runs_base="runs",
          paper_dir=PAPER_DIR):
    """Train and persist a run. Returns (run_id, run_dir, losses)."""
    set_seed(seed)

    X_train, y_train, _, _, in_dim, out_dim = data.load_split(
        seed=seed, test_size=TEST_SIZE
    )

    config = {
        "data": {"dataset": "iris", "in_dim": in_dim, "out_dim": out_dim,
                 "test_size": TEST_SIZE},
        "model": {"hidden": hidden},
        "hyperparams": {"lr": lr, "epochs": epochs},
        "seed": seed,
        "metric": METRIC,
    }
    run_id, run_dir = new_run_dir(paper_dir, config, base=runs_base)

    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    losses = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if epoch % 10 == 0 or epoch == epochs - 1:
            log_metrics({"loss": loss.item()}, step=epoch, run_dir=run_dir)

    save_checkpoint(model, run_dir)
    return run_id, run_dir, losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-base", type=str, default="runs",
                   help="base dir for run folders (tests redirect to tmp)")
    args = p.parse_args()

    run_id, _, _ = train(lr=args.lr, epochs=args.epochs, hidden=args.hidden,
                         seed=args.seed, runs_base=args.runs_base)
    print(f"run_id: {run_id}")


if __name__ == "__main__":
    main()
