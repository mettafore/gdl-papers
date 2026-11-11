"""JAX EGNN port: invariance, padding, torch parity, multi-device equivalence."""

import os
import sys
from pathlib import Path

# Two fake CPU devices so the data-parallel path is exercised on a laptop.
# Must be set before jax is imported.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import data  # noqa: E402
import data_jax  # noqa: E402
from egnn_jax import egnn_forward, init_params, params_from_torch  # noqa: E402
from model import EGNN  # noqa: E402

rng = np.random.default_rng(0)


def _molecules(sizes):
    zs = [rng.choice([1, 6, 7, 8, 9], size=n).astype(np.int8) for n in sizes]
    poss = [rng.normal(scale=1.5, size=(n, 3)).astype(np.float32) for n in sizes]
    return zs, poss


def _inputs(zs, poss, n_max=data_jax.N_MAX):
    b = data_jax.pad(zs, poss, n_max)
    x = jnp.asarray(data_jax.FEATURE_TABLE[b["z"]])
    return x, jnp.asarray(b["pos"]), jnp.asarray((b["z"] > 0).astype(np.float32))


PARAMS = init_params(jax.random.key(0))
SIZES = [5, 18, 29, 3]


def test_rotation_translation_invariance():
    zs, poss = _molecules(SIZES)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    moved = [(p @ q.T + rng.normal(size=3) * 5).astype(np.float32) for p in poss]
    a = egnn_forward(PARAMS, *_inputs(zs, poss))
    b = egnn_forward(PARAMS, *_inputs(zs, moved))
    np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-5)


def test_padding_does_not_change_predictions():
    zs, poss = _molecules(SIZES)
    a = egnn_forward(PARAMS, *_inputs(zs, poss))
    # More pad slots per molecule, plus an all-padding molecule at the end.
    x, pos, mask = _inputs(zs, poss, n_max=35)
    pos = pos.at[:, 30:].set(7.0)  # junk in padded slots must not leak
    x, pos, mask = (jnp.concatenate([t, jnp.zeros_like(t[:1])]) for t in (x, pos, mask))
    b = egnn_forward(PARAMS, x, pos, mask)
    np.testing.assert_allclose(a, b[: len(SIZES)], atol=1e-5, rtol=1e-5)


def test_parity_with_torch_egnn():
    torch.manual_seed(0)
    model = EGNN(in_node_nf=15, hidden_nf=128, n_layers=7, attention=True).eval()
    zs, poss = _molecules(SIZES)
    h, x, edges, batch, offset = [], [], [], [], 0
    for g, (z, p) in enumerate(zip(zs, poss, strict=True)):
        h.append(data.charge_power_features(torch.from_numpy(z).long()))
        x.append(torch.from_numpy(p))
        edges.append(data.fully_connected_edge_index(len(z)) + offset)
        batch.append(torch.full((len(z),), g))
        offset += len(z)
    with torch.no_grad():
        ref = model(
            torch.cat(h), torch.cat(x), torch.cat(edges, 1), batch=torch.cat(batch)
        )
    out = egnn_forward(params_from_torch(model.state_dict()), *_inputs(zs, poss))
    np.testing.assert_allclose(out, ref.numpy()[:, 0], atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(jax.device_count() < 2, reason="needs 2 devices")
def test_two_device_step_matches_single_device():
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from train_jax import make_optimizer, make_steps

    zs, poss = _molecules([4, 9, 12, 20, 29, 7, 3, 16])
    split = data_jax.pad(zs, poss) | {"y": rng.normal(size=8).astype(np.float32)}
    idx, gmask = np.arange(8, dtype=np.int32), np.array([1] * 7 + [0], np.float32)
    opt = make_optimizer(1e-3, 1e-16, epochs=10, steps_per_epoch=5)

    results = []
    for devs in (jax.devices()[:1], jax.devices()[:2]):
        mesh = Mesh(np.array(devs), ("d",))
        shard, repl = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
        params = jax.device_put(jax.tree.map(jnp.copy, PARAMS), repl)  # donated
        train_step, _ = make_steps(opt, mad=1.0)
        new, _, loss = train_step(
            params,
            jax.device_put(opt.init(params), repl),
            jax.device_put(split, repl),
            jax.device_put(idx, shard),
            jax.device_put(gmask, shard),
        )
        results.append((float(loss), new))
    assert results[0][0] == pytest.approx(results[1][0], rel=1e-5)
    # Adam's first step is ~lr*sign(g): reduction-order noise in tiny grads shows
    # up as ~1e-6 param differences, hence atol 1e-5 rather than exact equality.
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=1e-5),
        results[0][1],
        results[1][1],
    )
