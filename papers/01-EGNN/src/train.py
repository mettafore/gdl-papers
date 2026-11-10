"""Train EGNN on QM9 and save configuration, metrics, and checkpoints.

Uses Adam with cosine learning-rate annealing and normalized-target L1 loss.
Validation MAE is measured in original target units; the best validation model
is saved, with early stopping after the configured patience.

Paper: Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021.
"""

import argparse
import os
import time
from pathlib import Path

import data
import torch
from model import EGNN
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from gdl import load_checkpoint, log_metrics, new_run_dir, save_checkpoint, set_seed

PAPER_DIR = Path(__file__).parent.parent
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
    resume_from=None,
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
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ:
        torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
        init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    set_seed(seed)
    train_loader, val_loader, test_loader, norm, dims = data.load_split(
        target=target,
        seed=seed,
        batch_size=batch_size,
        root=data_root,
        num_workers=num_workers,
    )
    config = {
        "data": {"target": target, "root": str(data_root), "dims": dims},
        "model": {
            "hidden_nf": hidden_nf,
            "n_layers": n_layers,
            "attention": attention,
        },
        "hyperparams": {
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "weight_decay": weight_decay,
            "early_stop_patience": early_stop_patience,
        },
        "seed": seed,
        "metric": METRIC,
        "resume_from": str(resume_from) if resume_from is not None else None,
    }

    if rank == 0:
        run_id, run_dir = new_run_dir(paper_dir, config, runs_base)
    else:
        run_id, run_dir = None, None

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    egnn = EGNN(
        in_node_nf=dims["in_node_dim"],
        hidden_nf=hidden_nf,
        n_layers=n_layers,
        attention=attention,
    ).to(device)
    # Warm start: load weights from a prior run's checkpoint before training.
    # Used to continue the best checkpoint at a lower LR (test whether the
    # high-LR plateau breaks once steps shrink). Weights only — optimizer
    # (Adam) moment buffers are NOT restored, so expect a small transient
    # bump for a few epochs. config records resume_from for provenance.
    if resume_from is not None:
        load_checkpoint(egnn, resume_from)
    for name, param in egnn.named_parameters():
        # if param is coord_mlp, freeze gradients (no training on coordinates)
        if "coord_mlp" in name:
            param.requires_grad = False
    model = egnn
    if "RANK" in os.environ:
        egnn = DDP(egnn, device_ids=[local_rank] if torch.cuda.is_available() else None)
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
        t0 = time.perf_counter()
        if "RANK" in os.environ:
            train_loader.sampler.set_epoch(epoch)
        stop = False
        egnn.train()
        for b in train_loader:
            b = b.to(device)
            optimizer.zero_grad()
            out = egnn(b.x, b.pos, b.edge_index, batch=b.batch)
            target = target_for(b, norm, dims["target_col"])
            loss_value = loss(out, target)
            loss_value.backward()
            optimizer.step()
        current_lr = optimizer.param_groups[0]["lr"]
        # CosineAnnealingLR steps on a schedule, not on val_loss (no arg).
        scheduler.step()
        if rank == 0:
            model.eval()
            val_loss_sum = 0.0
            val_num_graphs = 0
            with torch.no_grad():
                for b in val_loader:
                    b = b.to(device)
                    out = model(b.x, b.pos, b.edge_index, batch=b.batch)
                    pred = norm.denormalize(out)
                    target = raw_target_for(b, dims["target_col"])
                    val_loss_sum += loss(pred, target).item() * b.num_graphs
                    val_num_graphs += b.num_graphs
            val_loss = val_loss_sum / val_num_graphs
            # Log the LR *before* stepping, so it's the rate this epoch actually
            # trained at (lets us see whether val drops as cosine anneals it down).
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            log_metrics(
                {
                    "val_loss": val_loss,
                    "lr": current_lr,
                    "time": time.perf_counter() - t0,
                },
                epoch,
                run_dir,
            )
            if val_loss < best_val:
                best_val = val_loss
                epochs_since_improve = 0
                if "RANK" in os.environ:
                    save_checkpoint(egnn.module, run_dir)
                else:
                    save_checkpoint(egnn, run_dir)
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= early_stop_patience:
                    print(
                        f"early stop: no val_loss improvement in {early_stop_patience} epochs"
                    )
                    stop = True
        if "RANK" in os.environ:
            flag = torch.tensor(int(stop)).to(device)
            torch.distributed.broadcast(flag, src=0)
            stop = bool(flag.item())
        if stop:
            break
    if "RANK" in os.environ:
        destroy_process_group()
    return run_id, run_dir, best_val


def main():
    """CLI entrypoint. Mirrors papers/00-template/src/train.py's main()."""
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
    if run_id is not None:
        print(f"run_id: {run_id}")


if __name__ == "__main__":
    main()
