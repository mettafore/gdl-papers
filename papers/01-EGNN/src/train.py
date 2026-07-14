"""Train EGNN on QM9.  [SCAFFOLD — you fill bodies]

Contract to match (same shape as papers/00-template/train.py):
  build config -> new_run_dir -> build model from data.dims -> train loop ->
  log_metrics periodically -> save_checkpoint -> return (run_id, run_dir, ...).

Paper: Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021.
Training recipe (CLAUDE.md): Adam lr=5e-4, ReduceLROnPlateau(patience=10,
factor=0.5, min_lr=1e-7), L1 (MAE) loss on NORMALIZED targets, batch 96,
early stopping patience ~50.

------------------------------------------------------------------------------
IMPORTANT — read before writing the loop:
  EGNN.forward(h, x, edge_index, edge_attr) currently sums EVERY node passed
  in into ONE scalar (single-graph assumption). A PyG DataLoader batch merges
  several molecules into one big disconnected graph plus a `batch` index
  vector (data.batch) that says which molecule each node belongs to.
  Question worth answering before you loop: does EGNN.forward need to change
  to pool per-molecule (using `batch`), or can you sidestep it for now
  (e.g. batch_size=1, or loop per-graph inside the step)? Don't silently
  ignore this — a naive sum over a multi-molecule batch is wrong.

WHAT YOU NEED TO BUILD (in rough order):
  1. train(...) function:
       a. set_seed(seed)
       b. data.load_split(...) -> train_loader, val_loader, test_loader, norm, dims
       c. build config dict (data/model/hyperparams/seed/metric)
       d. new_run_dir(paper_dir, config) -> run_id, run_dir
       e. build EGNN from dims (in_node_nf=dims["in_node_dim"], hidden_nf=...)
       f. optimizer = Adam(lr=5e-4); scheduler = ReduceLROnPlateau(...)
       g. loss_fn = L1Loss; train on norm.normalize(target column)
       h. per-epoch: train step(s) -> val loss -> scheduler.step(val_loss) ->
          log_metrics -> track best val loss -> save_checkpoint on improvement
       i. optional: early stopping on val loss plateau
  2. main(): argparse CLI mirroring the template (--lr/--epochs/--seed/...),
     prints run_id.

Questions worth answering as you go:
  - Why step the scheduler on VAL loss, not train loss?
  - What happens to the metric (MAE) if you evaluate in normalized space vs
    denormalized (original units)? Which one matches the paper's Table 1 units?
  - Where does `gdl.metrics.mae` fit vs the L1 training loss?
------------------------------------------------------------------------------
"""

from pathlib import Path

import torch
from torch import nn

import data
from model import EGNN
from gdl import set_seed, log_metrics, new_run_dir, save_checkpoint

PAPER_DIR = Path(__file__).parent
METRIC = "mae"


def train(target="gap", lr=5e-4, epochs=1000, hidden_nf=128, n_layers=7,
          seed=42, batch_size=96, runs_base="runs", paper_dir=PAPER_DIR):
    """Train and persist a run. Returns (run_id, run_dir, ...)."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
