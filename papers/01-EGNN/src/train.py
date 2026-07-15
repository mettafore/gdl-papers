"""Train EGNN on QM9.  [SCAFFOLD — you fill bodies]

Contract to match (same shape as papers/00-template/src/train.py):
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
from model import EGNN
from torch import nn

import data
from gdl import log_metrics, new_run_dir, save_checkpoint, set_seed

PAPER_DIR = Path(__file__).parent
METRIC = "mae"


def target_for(batch, norm, col):
    """Normalized target column for one PyG batch, shaped (num_graphs, 1)."""
    return norm.normalize(batch.y[:, col : col + 1])


def raw_target_for(batch, col):
    return batch.y[:, col : col + 1]


def train(
    target="gap",
    lr=1e-3,
    epochs=1000,
    hidden_nf=128,
    n_layers=7,
    seed=42,
    batch_size=96,
    runs_base="runs",
    paper_dir=PAPER_DIR,
    data_root="data/qm9",
    num_workers=0,
    weight_decay=1e-16,
    early_stop_patience=50,
    attention=True,
):
    """Train and persist a run. Returns (run_id, run_dir, best_val).

    Recipe matches vgsatorras/egnn's main_qm9.py: Adam(lr=1e-3,
    weight_decay=1e-16) + CosineAnnealingLR over the full `epochs` schedule
    (not ReduceLROnPlateau — that halves LR on every 10-epoch stall, which
    on noisy val loss can collapse LR toward its floor early and freeze
    training; cosine annealing decays smoothly regardless of per-epoch
    noise). L1 loss on the normalized target. Checkpoints on every val_loss
    improvement; stops early after `early_stop_patience` epochs without one.
    """
    set_seed(seed)
    train_loader, val_loader, test_loader, norm, dims = data.load_split(
        target=target,
        seed=seed,
        batch_size=batch_size,
        root=data_root,
        num_workers=num_workers,
    )
    config = {
        "data": {"target": target, "dims": dims},
        "model": {
            "hidden_nf": hidden_nf,
            "n_layers": n_layers,
            "attention": attention,
        },
        "hyperparams": {
            "lr": lr,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "early_stop_patience": early_stop_patience,
        },
        "seed": seed,
        "metric": METRIC,
    }

    run_id, run_dir = new_run_dir(paper_dir, config, runs_base)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    egnn = EGNN(
        in_node_nf=dims["in_node_dim"],
        hidden_nf=hidden_nf,
        n_layers=n_layers,
        attention=attention,
    ).to(device)
    optimizer = torch.optim.Adam(egnn.parameters(), lr=lr, weight_decay=weight_decay)
    # CosineAnnealingLR (not ReduceLROnPlateau): matches the reference repo's
    # recipe. ReduceLROnPlateau halves LR on every 10-epoch stall, which on a
    # noisy val loss can collapse LR toward its floor within ~200 epochs and
    # freeze training — that's what happened on the first real run (flat
    # val_loss from epoch ~200 on). Cosine annealing decays smoothly across
    # the full schedule regardless of per-epoch noise.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss = nn.L1Loss()

    best_val = float("inf")
    epochs_since_improve = 0
    for epoch in range(epochs):
        egnn.train()
        for b in train_loader:
            b = b.to(device)
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
                b = b.to(device)
                out = egnn(b.x, b.pos, b.edge_index, batch=b.batch)
                pred = norm.denormalize(out)
                target = raw_target_for(b, dims["target_col"])
                val_losses.append(loss(pred, target).item())
        val_loss = sum(val_losses) / len(val_losses)
        # CosineAnnealingLR steps on a schedule, not on val_loss (no arg).
        scheduler.step()
        log_metrics({"val_loss": val_loss}, epoch, run_dir)
        if val_loss < best_val:
            best_val = val_loss
            epochs_since_improve = 0
            save_checkpoint(egnn, run_dir)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= early_stop_patience:
                print(
                    f"early stop: no val_loss improvement in {early_stop_patience} epochs"
                )
                break
    return run_id, run_dir, best_val


def main():
    """CLI entrypoint. Mirrors papers/00-template/src/train.py's main().

    # TODO:
    #   - argparse: --target, --lr, --epochs, --hidden-nf, --n-layers, --seed,
    #     --batch-size, --runs-base (see template main() for the pattern)
    #   - call train(...) with parsed args
    #   - print(f"run_id: {run_id}")
    """
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target", type=str, default="gap")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--hidden-nf", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--runs-base", type=str, default="runs")
    p.add_argument(
        "--attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = p.parse_args()
    run_id, run_dir, best_val = train(
        target=args.target,
        lr=args.lr,
        epochs=args.epochs,
        hidden_nf=args.hidden_nf,
        n_layers=args.n_layers,
        seed=args.seed,
        batch_size=args.batch_size,
        runs_base=args.runs_base,
        paper_dir=PAPER_DIR,
        attention=args.attention,
    )

    print(f"run_id: {run_id}")


if __name__ == "__main__":
    main()
