# 01-EGNN — E(n) Equivariant Graph Neural Networks

Reproduction of Satorras et al., *E(n) Equivariant Graph Neural Networks*
(ICML 2021) for QM9 molecular-property prediction. The model is invariant to
translations, rotations, and reflections of molecular coordinates when
predicting a scalar property.

## Files

- `src/model.py` — EGNN model and equivariant graph-convolution layers
- `src/data.py` — QM9 loading, deterministic split, features, and complete graphs
- `src/train.py` — training CLI and run-artifact creation
- `src/evaluate.py` — checkpoint evaluation on the held-out test split
- `modal_train.py` — persistent Modal GPU training wrapper
- `modal_profile.py` — Modal data-loading and compute profiling
- `test_*.py` — model, data, training, and evaluation tests
- `notes.md` — reading notes and experiment log

## Data

- Dataset: [QM9](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html), loaded through PyTorch Geometric and cached at `data/qm9`.
- Task: one scalar molecular property; `gap` (HOMO-LUMO gap) is the default.
- Split: deterministic seeded permutation, with 100,000 training and 18,000
  validation molecules; the remainder is the test split.
- Input features: 15-dimensional charge-power features derived from atomic
  number, matching the reference configuration.
- Graphs: fully connected directed graphs without self-loops. QM9's supplied
  bond graph is deliberately replaced because EGNN uses all atom-pair distances.
- Normalization: target mean and mean absolute deviation are fit on the training
  split only.

QM9-backed tests are intentionally opt-in:

```bash
uv run pytest papers/01-EGNN/ -m needs_data
```

## Hyperparameters

- `--target` (default `gap`) — QM9 property to predict
- `--lr` (default `1e-3`) — Adam learning rate
- `--epochs` (default `1000`) — maximum epochs
- `--hidden-nf` (default `128`) — hidden feature width
- `--n-layers` (default `7`) — EGCL layers
- `--batch-size` (default `96`) — molecules per batch
- `--seed` (default `42`) — RNG seed and split seed
- `--[no-]attention` (default enabled) — sigmoid message gate in each EGCL

Training uses normalized-target L1 loss, cosine learning-rate annealing, and
early stopping after 50 epochs without validation improvement. Every run stores
its complete recipe in `runs/<run_id>/config.json`.

## Run

For a short local smoke run (the first invocation downloads QM9):

```bash
uv run python papers/01-EGNN/src/train.py --epochs 1 --hidden-nf 8 --n-layers 2
```

For the full default recipe, use a GPU. The Modal wrapper persists QM9 and run
artifacts in its `egnn-qm9` volume:

```bash
uv run modal run papers/01-EGNN/modal_train.py --detach
```

Both commands print a `run_id`. Evaluate it on the same held-out split:

```bash
uv run python papers/01-EGNN/src/evaluate.py --run <run_id>
```

Local runs live in `papers/01-EGNN/runs/<run_id>/`. Modal runs live in the
Modal volume at `/vol/runs/<run_id>/`.

## Results

The paper's QM9 benchmark is Table 3. For `gap`, it reports an MAE of **48 meV**.

| Target | Paper MAE | Reproduction test MAE | Run |
|---|---:|---:|---|
| `gap` | 48 meV | 48.50 meV | `20260716-082300`, epoch 181 |

The reported reproduction result uses fully connected graphs, charge-power
features, attention, and a lower-learning-rate warm start from the best prior
checkpoint. See [`docs/progress.md`](../../docs/progress.md) for the experiment
history and planned target sweep.
