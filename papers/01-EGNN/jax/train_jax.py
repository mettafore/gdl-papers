"""Train the JAX EGNN on QM9 — one jit-compiled XLA program per train step.

Recipe matches ../src/train.py: Adam(lr=1e-3, coupled L2 weight_decay=1e-16),
cosine LR stepped per epoch, L1 on the normalized target, batch 96.
Data parallel over all local devices: the batch axis is sharded, params and
the (small) dataset are replicated, XLA inserts the gradient all-reduce.

    uv run python papers/01-EGNN/jax/train_jax.py --epochs 2 --subset 2000
"""

from __future__ import annotations

import argparse
import math
import time

import data_jax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from egnn_jax import egnn_forward, init_params
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def make_optimizer(lr: float, weight_decay: float, epochs: int, steps_per_epoch: int):
    # torch.optim.Adam(weight_decay=wd) adds wd*param to the grad *before* Adam
    # (coupled L2, not AdamW), so decay goes first in the chain.
    def schedule(step):  # CosineAnnealingLR(T_max=epochs), stepped once per epoch
        epoch = step // steps_per_epoch
        return lr * 0.5 * (1 + jnp.cos(jnp.pi * epoch / epochs))

    return optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_adam(),
        optax.scale_by_learning_rate(schedule),
    )


def make_steps(optimizer, mad: float):
    table = jnp.asarray(data_jax.FEATURE_TABLE)

    def predict(params, split, idx):
        z = split["z"][idx]
        mask = (z > 0).astype(jnp.float32)
        return egnn_forward(params, table[z], split["pos"][idx], mask)

    def loss_fn(params, split, idx, gmask):
        err = jnp.abs(predict(params, split, idx) - split["y"][idx]) * gmask
        return jnp.sum(err) / jnp.sum(gmask)

    def train_step(params, opt_state, split, idx, gmask):
        loss, grads = jax.value_and_grad(loss_fn)(params, split, idx, gmask)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def eval_step(params, split, idx, gmask):
        # Sum of |error| in raw target units (normalized error * MAD).
        err = jnp.abs(predict(params, split, idx) - split["y"][idx]) * mad
        return jnp.sum(err * gmask)

    train_step = jax.jit(train_step, donate_argnums=(0, 1))
    return train_step, eval_step


def batches(n: int, batch_size: int, rng: np.random.Generator | None):
    """Yield (idx, gmask) of fixed length batch_size; last batch padded+masked."""
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for s in range(0, n, batch_size):
        chunk = order[s : s + batch_size]
        idx = np.zeros(batch_size, np.int32)
        idx[: len(chunk)] = chunk
        gmask = np.zeros(batch_size, np.float32)
        gmask[: len(chunk)] = 1
        yield idx, gmask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-16)
    p.add_argument("--target", type=str, default="gap")
    p.add_argument("--hidden-nf", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="data/qm9")
    p.add_argument("--subset", type=int, default=0, help="use first N train/val mols")
    args = p.parse_args()

    devices = jax.devices()
    if args.batch_size % len(devices):
        raise SystemExit(f"--batch-size must be divisible by {len(devices)} devices")
    mesh = Mesh(np.array(devices), ("d",))
    shard, repl = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    print(f"devices: {devices}")

    t0 = time.perf_counter()
    splits, norm = data_jax.load_qm9(args.data_root, args.target, args.seed)
    if args.subset:
        for k in ("train", "val"):
            splits[k] = {a: v[: args.subset] for a, v in splits[k].items()}
    for s in splits.values():
        s["y"] = ((s["y"] - norm.mean) / norm.mad).astype(np.float32)
    train, val = (jax.device_put(splits[k], repl) for k in ("train", "val"))
    n_train, n_val = len(splits["train"]["y"]), len(splits["val"]["y"])
    print(f"data: {n_train} train / {n_val} val in {time.perf_counter() - t0:.1f}s")

    steps_per_epoch = math.ceil(n_train / args.batch_size)
    optimizer = make_optimizer(args.lr, args.weight_decay, args.epochs, steps_per_epoch)
    params = init_params(jax.random.key(args.seed), 15, args.hidden_nf, args.n_layers)
    params = jax.device_put(params, repl)
    opt_state = jax.device_put(optimizer.init(params), repl)
    train_step, eval_step = make_steps(optimizer, norm.mad)

    def put(idx, gmask):
        return jax.device_put(idx, shard), jax.device_put(gmask, shard)

    # Compile ahead of time so epoch times are pure execution.
    t0 = time.perf_counter()
    ex = put(*next(batches(n_train, args.batch_size, None)))
    train_step = train_step.lower(params, opt_state, train, *ex).compile()
    eval_step = eval_step.lower(params, val, *ex).compile()
    print(f"compile: {time.perf_counter() - t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        t0 = time.perf_counter()
        loss_sum = jnp.zeros(())
        for b in batches(n_train, args.batch_size, rng):
            params, opt_state, loss = train_step(params, opt_state, train, *put(*b))
            loss_sum = loss_sum + loss  # stays on device; no per-step sync
        loss_sum.block_until_ready()
        t_train = time.perf_counter() - t0
        err_sum = sum(
            (
                eval_step(params, val, *put(*b))
                for b in batches(n_val, args.batch_size, None)
            ),
            start=jnp.zeros(()),
        )
        val_mae = float(err_sum) / n_val  # == PyTorch train.py's val_loss (raw units)
        print(
            f"epoch {epoch:4d}  train_l1_norm {float(loss_sum) / steps_per_epoch:.4f}  "
            f"val_mae {val_mae:.5f}  val_l1_norm {val_mae / norm.mad:.4f}  "
            f"lr {lr:.2e}  train_time {t_train:.2f}s  "
            f"epoch_time {time.perf_counter() - t0:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
