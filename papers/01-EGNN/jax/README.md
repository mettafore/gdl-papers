# EGNN on QM9 — JAX track

A separate speed experiment, not part of the PyTorch reproduction (`../src` is
unchanged). The PyTorch step is launch-bound: about 986 small CUDA kernels from
Python per step, and the GPU is busy only 25-40% of the time on a Kaggle 2xT4.
Here the whole train step (forward, loss, backward, Adam) is a single `jax.jit`
XLA program, and per-molecule preprocessing runs once at startup.

## Files

| File | What |
|------|------|
| `egnn_jax.py` | Model in plain JAX (dict params, `vmap` over molecules, `lax.scan` over the 7 layers) and `params_from_torch` |
| `data_jax.py` | QM9 converted once into padded NumPy arrays. Reuses `../src/data.py` for the dataset, the split sizes, the targets and the charge-power features |
| `train_jax.py` | CLI: optax Adam, cosine LR, jitted train and eval steps, data-parallel over local devices |
| `test_egnn_jax.py` | Invariance, padding, torch-parity and 1-vs-2-device tests |

## Run

```bash
uv run python papers/01-EGNN/jax/train_jax.py --epochs 3 --batch-size 96 \
    --data-root data/qm9            # add --subset 2000 for a quick CPU run
uv run pytest papers/01-EGNN/jax/   # fakes 2 CPU devices for the DP test
```

Each epoch prints `train_l1_norm` (the training loss), `val_mae` in raw target
units, `val_l1_norm` (= `val_mae / MAD`), `lr`, and `train_time`/`epoch_time`.
`val_mae` is the same quantity `../src/train.py` logs as `val_loss`.
Compilation is done ahead of time and reported separately, so epoch times only
measure execution.

## Design choices

- **Framework:** plain JAX + optax, with no Flax. The params dict mirrors the
  torch `state_dict`, so copying weights is just a transpose.
- **Padding:** every molecule is padded to `n_max = 29` atoms (the largest in
  QM9), with a node mask. Edges are one constant fully connected template of
  29·28 = 812 directed pairs. Messages are masked before
  `jax.ops.segment_sum(num_segments=29)`, node outputs are masked before sum
  pooling, and padded graphs in the last batch are masked out of the loss. A
  batch is `(B, 29, ...)` dense arrays, so there is one fixed shape and one
  compile.
- **Cost of that choice:** real molecules average 18.0 atoms, so on average only
  **38.9 %** of the 812 edge slots are real. That means about 2.6× wasted edge
  FLOPs, traded for zero recompiles and trivial sharding. If the GPU becomes
  compute-bound, packing molecules into flat node/edge budgets (jraph-style)
  would recover it.
- **Data on device:** the whole split (`z` int8, `pos`, normalized `y`) is
  placed on the device once. Each step only sends `(B,)` indices and a graph
  mask, and the gather plus feature lookup happen inside the jitted step.
- **Multi-GPU:** a 1-D `Mesh` over `jax.devices()`. Index and mask are sharded
  on the batch axis, while params, optimizer state and dataset are replicated.
  XLA inserts the gradient all-reduce, and one device works with no code
  change. `--batch-size` must be divisible by the device count.
- **Optimizer:** `add_decayed_weights → scale_by_adam → lr`, which is torch
  `Adam(weight_decay=1e-16)` (coupled L2, not AdamW). LR is
  `CosineAnnealingLR(T_max=epochs)`, held constant within an epoch as torch
  steps it per epoch.
- **Init:** torch `nn.Linear` default, U(±1/√fan_in) for weights and biases.
- **Differences from PyTorch:** `coord_mlp` is omitted, because torch freezes it
  and never calls it on QM9. There is no checkpointing, JSONL logging, early
  stopping or resume yet, since this track is for timing. The shuffle order
  differs from torch's DataLoader (NumPy RNG), while the split itself is
  identical.

## Verified (Mac, CPU, jax 0.5.3)

- Rotating and translating positions leaves predictions unchanged (atol 1e-4).
- Repadding to `n_max=35`, with junk positions in the pad slots and an extra
  all-padding molecule, leaves predictions unchanged (atol 1e-5).
- **Parity:** weights copied from a torch `EGNN(attention=True)` give the same
  predictions on a 4-molecule batch (atol 1e-4).
- One train step on 2 fake CPU devices matches 1 device: loss rel 1e-5,
  params atol 1e-5. The compiled 2-device train step contains only `all-reduce`
  collectives (gradient and loss sums) and no `all-gather`, so the batch axis
  stays split through the gather, scan and `segment_sum`.
- Real QM9 (one-off script, not a test): `load_qm9` reproduces
  `data.load_split` exactly. Mean and MAD are identical, and z, pos and y match
  for the first val molecules. Copying the torch weights and running a real
  8-molecule val batch gives max |diff| 1.8e-7.
- 3 epochs on a 2k-molecule subset: train L1 1.106 → 0.847 → 0.809, val MAE
  0.940 → 0.872 → 0.838 eV. Each epoch took about 25 s on CPU (21 steps of 96).

Not verified: real GPU speed, real 2-GPU scaling, and full-dataset convergence.

## Kaggle (2xT4) cell

Push this branch as `jax-egnn` first. Internet must be on, because QM9 is
downloaded by PyG.

The cell runs batch 96 and batch 256 on both GPUs, plus batch 96 on one GPU so
the scaling can be compared.

```python
import os, subprocess, sys
BRANCH, REPO = "jax-egnn", "/kaggle/working/gdl-papers"

def run(cmd, env=None):  # stream child output into the cell, line by line
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})})
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise RuntimeError(f"failed: {cmd}")

py = sys.executable
if not os.path.isdir(REPO):
    run(["git", "clone", "--depth", "1", "-b", BRANCH, "https://github.com/mettafore/gdl-papers.git", REPO])
# Keep Kaggle's CUDA jax if it already sees GPUs; install only what is missing.
probe = "import jax; print(any(d.platform == 'gpu' for d in jax.devices()))"
if subprocess.run([py, "-c", probe], capture_output=True, text=True).stdout.strip() != "True":
    run([py, "-m", "pip", "install", "-q", "-U", "jax[cuda12]"])
for mod in ("optax", "torch_geometric"):
    if subprocess.run([py, "-c", f"import {mod}"]).returncode:
        run([py, "-m", "pip", "install", "-q", mod])
run([py, "-c", "import jax, optax; print(jax.__version__, optax.__version__, jax.devices())"])

env = {"PYTHONPATH": f"{REPO}/src:{REPO}/papers/01-EGNN/src"}
script = f"{REPO}/papers/01-EGNN/jax/train_jax.py"
for bs, gpus in [(96, "0,1"), (256, "0,1"), (96, "0")]:
    print(f"\n===== batch {bs}, GPUs {gpus} =====", flush=True)
    run([py, script, "--epochs", "3", "--batch-size", str(bs),
         "--data-root", "/kaggle/working/data/qm9"], {**env, "CUDA_VISIBLE_DEVICES": gpus})
```
