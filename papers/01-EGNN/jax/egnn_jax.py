"""EGNN for QM9 in plain JAX — a port of ../src/model.py.

Layout: every molecule is padded to `n_max` atoms with a node mask, and edges
are the constant fully connected template over those `n_max` slots (all i != j).
The single-molecule forward is `vmap`ped over the batch, so a batch is just
dense arrays of shape (B, n_max, ...) — fixed shapes, one XLA compile.

Params are a plain dict pytree mirroring the torch state_dict; the 7 EGCL
layers are stacked on a leading axis and run with `lax.scan`. `coord_mlp` is
omitted: the torch model freezes it and never calls it on QM9.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

Params = dict


def _linear_init(key: jax.Array, fan_in: int, fan_out: int) -> dict:
    # torch nn.Linear default: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for W and b.
    bound = 1.0 / math.sqrt(fan_in)
    kw, kb = jax.random.split(key)
    return {
        "w": jax.random.uniform(kw, (fan_in, fan_out), minval=-bound, maxval=bound),
        "b": jax.random.uniform(kb, (fan_out,), minval=-bound, maxval=bound),
    }


def _dense(p: dict, x: jax.Array) -> jax.Array:
    return x @ p["w"] + p["b"]


def init_params(
    key: jax.Array, in_node_nf: int = 15, hidden_nf: int = 128, n_layers: int = 7
) -> Params:
    h = hidden_nf
    k_emb, k_layers, k_dec = jax.random.split(key, 3)

    def layer(k: jax.Array) -> dict:
        ks = jax.random.split(k, 5)
        return {
            "edge1": _linear_init(ks[0], 2 * h + 1, h),
            "edge2": _linear_init(ks[1], h, h),
            "att": _linear_init(ks[2], h, 1),
            "node1": _linear_init(ks[3], 2 * h, h),
            "node2": _linear_init(ks[4], h, h),
        }

    layers = [layer(k) for k in jax.random.split(k_layers, n_layers)]
    kd = jax.random.split(k_dec, 4)
    return {
        "emb": _linear_init(k_emb, in_node_nf, h),
        "layers": jax.tree.map(lambda *xs: jnp.stack(xs), *layers),
        "node_dec1": _linear_init(kd[0], h, h),
        "node_dec2": _linear_init(kd[1], h, h),
        "graph_dec1": _linear_init(kd[2], h, h),
        "graph_dec2": _linear_init(kd[3], h, 1),
    }


def params_from_torch(state_dict: dict) -> Params:
    """Convert a torch `EGNN(attention=True)` state_dict (torch W is (out, in))."""
    sd = {k: np.asarray(v.detach().cpu().numpy()) for k, v in state_dict.items()}

    def lin(prefix: str) -> dict:
        return {
            "w": jnp.asarray(sd[f"{prefix}.weight"].T),
            "b": jnp.asarray(sd[f"{prefix}.bias"]),
        }

    n_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers."))
    names = {
        "edge1": "edge_mlp.0",
        "edge2": "edge_mlp.2",
        "att": "att_mlp.0",
        "node1": "node_mlp.0",
        "node2": "node_mlp.2",
    }
    layers = [
        {ours: lin(f"layers.{i}.{theirs}") for ours, theirs in names.items()}
        for i in range(n_layers)
    ]
    return {
        "emb": lin("emb"),
        "layers": jax.tree.map(lambda *xs: jnp.stack(xs), *layers),
        "node_dec1": lin("node_dec.0"),
        "node_dec2": lin("node_dec.2"),
        "graph_dec1": lin("graph_dec.0"),
        "graph_dec2": lin("graph_dec.2"),
    }


def _edge_template(n_max: int) -> tuple[np.ndarray, np.ndarray]:
    # Same ordering as data.fully_connected_edge_index: row = receiver i.
    idx = np.arange(n_max)
    row, col = np.repeat(idx, n_max), np.tile(idx, n_max)
    keep = row != col
    return row[keep], col[keep]


def _molecule(
    params: Params, x: jax.Array, pos: jax.Array, mask: jax.Array
) -> jax.Array:
    """One padded molecule: x (n_max, F), pos (n_max, 3), mask (n_max,) -> scalar."""
    n_max = x.shape[0]
    row, col = _edge_template(n_max)
    edge_mask = (mask[row] * mask[col])[:, None]
    radial = jnp.sum((pos[row] - pos[col]) ** 2, axis=1, keepdims=True)

    def egcl(h: jax.Array, p: dict) -> tuple[jax.Array, None]:
        m = jnp.concatenate([h[row], h[col], radial], axis=1)
        m = jax.nn.silu(_dense(p["edge2"], jax.nn.silu(_dense(p["edge1"], m))))
        m = jax.nn.sigmoid(_dense(p["att"], m)) * m * edge_mask
        agg = jax.ops.segment_sum(m, row, num_segments=n_max)
        upd = _dense(
            p["node2"], jax.nn.silu(_dense(p["node1"], jnp.concatenate([h, agg], 1)))
        )
        return h + upd, None

    h = _dense(params["emb"], x)
    h, _ = jax.lax.scan(egcl, h, params["layers"])
    out = _dense(params["node_dec2"], jax.nn.silu(_dense(params["node_dec1"], h)))
    pooled = jnp.sum(out * mask[:, None], axis=0)
    g = _dense(params["graph_dec2"], jax.nn.silu(_dense(params["graph_dec1"], pooled)))
    return g[0]


def egnn_forward(
    params: Params, x: jax.Array, pos: jax.Array, mask: jax.Array
) -> jax.Array:
    """Batched: x (B, n_max, F), pos (B, n_max, 3), mask (B, n_max) -> (B,)."""
    return jax.vmap(_molecule, in_axes=(None, 0, 0, 0))(params, x, pos, mask)
