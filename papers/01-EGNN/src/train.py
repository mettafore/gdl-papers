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

import argparse
from pathlib import Path

import torch
from torch import nn

import data
from model import EGNN
from gdl import set_seed, log_metrics, new_run_dir, save_checkpoint

PAPER_DIR = Path(__file__).parent
METRIC = "mae"


def target_for(batch, norm, col):
    """Normalized target column for one PyG batch, shaped (num_graphs, 1)."""
    return norm.normalize(batch.y[:, col : col + 1])

def raw_target_for(batch, col):
  return batch.y[:, col:col+1]


def train(target="gap", lr=5e-4, epochs=1000, hidden_nf=128, n_layers=7,
          seed=42, batch_size=96, runs_base="runs", paper_dir=PAPER_DIR,
          data_root="data/qm9"):
    """Train and persist a run. Returns (run_id, run_dir, ...).

    # TODO (rough order — see module docstring for the batching gotcha):
    #   a. seed everything (there's a helper imported for this — not torch's
    #      own seeding function)
    #   b. get the three loaders + norm stats + dims from this file's sibling
    #      data module, using this function's own target/seed/batch_size args
    #   c. assemble a config dict describing this run (data/model/hyperparams/
    #      seed/metric) — look at the template's train.py for the shape
    #   d. create the run folder from that config (helper imported already)
    #   e. build the model from `dims` (don't hardcode the node-feature width —
    #      read it off what load_split returned)
    #   f. optimizer + scheduler, per the recipe in this paper's CLAUDE.md
    #      (Adam, ReduceLROnPlateau — exact hyperparams are documented there)
    #   g. loss: L1, computed against the NORMALIZED target column (the norm
    #      object from load_split does this) — and don't forget the model's
    #      forward now needs a `batch` argument; a PyG batch object carries
    #      exactly that under one of its attributes
    #   h. per epoch: train pass, then a no-grad val pass, step the scheduler
    #      on val loss (not train — think about why), log metrics, checkpoint
    #      whenever val loss improves
    #   i. optional: stop early if val loss stalls for a while
    #   return whatever a caller needs to locate + inspect this run
    """
    set_seed(seed)
    train_loader, val_loader, test_loader, norm, dims = data.load_split(
                target=target,
                seed=seed,
                batch_size=batch_size,
                root=data_root,
                )
    config = {
      "data":{"target": target,"dims": dims},
      "model": {"hidden_nf": hidden_nf,
      "n_layers": n_layers},
      "hyperparams": {"lr": lr, "epochs": epochs},
      "seed": seed,
      "metric":METRIC
      
    }

    run_id, run_dir = new_run_dir(paper_dir, config, runs_base)

    egnn = EGNN(
      in_node_nf=dims["in_node_dim"],
      hidden_nf=hidden_nf,
      n_layers=n_layers
                )
    optimizer = torch.optim.Adam(egnn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optimizer, 
                  patience=10,
                  factor=0.5,
                  min_lr=1e-7)
    loss = nn.L1Loss()

    best_val = float("inf")
    for epoch in range(epochs):
      egnn.train()
      for b in train_loader:
        optimizer.zero_grad()
        out = egnn(b.x, b.pos, b.edge_index, batch=b.batch)
        target = target_for(b, norm, dims["target_col"])
        loss_value = loss(out, target)
        loss_value.backward()
        optimizer.step()
    
      egnn.eval()
      val_losses = []
      with torch.no_grad():
        for b in val_loader:
          out = egnn(b.x, b.pos, b.edge_index, batch=b.batch)
          pred = norm.denormalize(out)
          target = raw_target_for(b, dims["target_col"])
          val_losses.append(loss(pred, target).item())
      val_loss = sum(val_losses) / len(val_losses)
      scheduler.step(val_loss)
      log_metrics({"val_loss":val_loss}, epoch, run_dir)
      if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(egnn, run_dir)
    return run_id, run_dir, best_val


def main():
    """CLI entrypoint. Mirrors papers/00-template/train.py's main().

    # TODO:
    #   - argparse: --target, --lr, --epochs, --hidden-nf, --n-layers, --seed,
    #     --batch-size, --runs-base (see template main() for the pattern)
    #   - call train(...) with parsed args
    #   - print(f"run_id: {run_id}")
    """
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--target", type=str, default="gap")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--hidden-nf", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--runs-base", type=str, default="runs")

    args = p.parse_args()
    run_id, run_dir, best_val = train(
      target= args.target,
      lr = args.lr,
      epochs = args.epochs,
      hidden_nf = args.hidden_nf,
      n_layers = args.n_layers,
      seed = args.seed,
      batch_size = args.batch_size,
      runs_base = args.runs_base,
      paper_dir = PAPER_DIR
    )

    print(f"run_id: {run_id}")

    

if __name__ == "__main__":
    main()
