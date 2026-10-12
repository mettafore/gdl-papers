# 15-MACE — higher-order equivariant message passing

Learning-first reproduction of the MACE `L=2` result on 3BPA. The scientific
target is the MACE row in Table 2 of Batatia et al. (NeurIPS 2022), not the
current production MACE package.

The paper-local training and evaluation path is implemented and covered by a
short end-to-end smoke test. Matching the published three-seed results remains
an experimental target, not a claimed result. `enumerate_coupling_routes` in
`model.py` is a pedagogical SO(3) exercise, not part of the final MACE forward
path; the model uses parity-aware e3nn tensor products and contraction tables.

## Files

- `src/data.py` — extxyz records, exact 450/50 split, neighbour topology, PyG batching
- `src/model.py` — radial/angular features, coupling routes, Eqs. 8–11, MACE
- `src/train.py` — energy/force loss and the training driver
- `src/evaluate.py` — per-subset energy and force RMSE
- `test_data.py` — split and graph contracts
- `test_model.py` — route, cutoff, shape, and equivariance contracts
- `test_pipeline.py` — loss and published benchmark contract

## Data

Download the official 3BPA supplementary archive and place its extracted
`dataset_3BPA/` directory outside git, then pass that directory with `--data`.
The relevant extxyz files are:

- `train_300K.xyz` — 500 configurations; seeded split into 450 train / 50 valid
- `iso_atoms.xyz` — isolated H/C/N/O energies used for the atomic baseline
- `test_300K.xyz` — in-distribution test
- `test_600K.xyz` and `test_1200K.xyz` — temperature extrapolation tests
- `test_dih_beta120.xyz`, `test_dih_beta150.xyz`, `test_dih_beta180.xyz` — pooled dihedral test

Each configuration has 27 atoms (H, C, N, O), one total energy, and one force
vector per atom. Positions are in angstrom, energies in eV, and forces in
eV/angstrom. The model predicts a scalar total energy and obtains forces as
`-dE/dR`.

Use ASE to construct the non-periodic 5-angstrom neighbour topology. Use PyG to
batch graphs, offset their atom indices, and assign each atom to a graph. Your
data pipeline should store positions and sender/receiver indices—but never
precomputed edge vectors.

The model must derive each vector during `forward` as
`positions[receiver] - positions[sender]`, after enabling gradients on
positions. Forces are then `-dE/dR`; force-loss training requires
`create_graph=True` so that gradient information reaches the model parameters.

## Hyperparameters

Published `L=2` recipe:

- two interaction layers; correlation order `nu=3`
- output irreps through `L=2`; internal spherical harmonics through `l_max=3`
- 256 channels; 5 angstrom cutoff
- eight Bessel radial functions; polynomial envelope `p=5`
- radial MLP `[64, 64, 64, 1024]` with SiLU
- hidden irreps `256x0e + 256x1o + 256x2e`
- first interaction non-residual; second interaction residual
- message aggregation normalized by the training-set average neighbour count
- first readout linear; final readout MLP with hidden width `16`
- energy/force loss weights `1 / 1000`
- AMSGrad Adam, learning rate `0.01`, batch size `5`
- plateau scheduler: patience `50`, factor `0.8`
- EMA `0.99`; weight decay `5e-7` on Eq. 10/11 weights
- per-atom mean energy shift and RMS-force scale; float64

The shift is the mean residual energy per atom after subtracting the isolated
H/C/N/O baseline. The scale is the RMS over individual force components.

Small-width or one-layer settings are debugging stages only. They are not the
benchmark model.

This reproduction is deliberately bounded to non-periodic 3BPA. It does not
implement the original repository's crystal/PBC generality or modern MACE
foundation-model machinery.

## Run

Train one seed, then evaluate its best-validation EMA checkpoint:

```bash
uv run python papers/15-MACE/src/train.py --data /path/to/dataset_3BPA
uv run python papers/15-MACE/src/evaluate.py --run <run_id> \
  --data /path/to/dataset_3BPA
uv run pytest papers/15-MACE/
```

By default, both commands use CUDA when PyTorch detects it and otherwise use
CPU. Select a device explicitly with `--device cuda` or `--device cpu`. This
float64 reproduction does not support Apple's MPS backend.

Run the bounded one-batch CUDA smoke on Modal with:

```bash
uv run modal run papers/15-MACE/modal_smoke.py
```

It trains one batch, validates, reloads the checkpoint, and evaluates one
configuration. It is an execution check, not a benchmark result. Modal requires
a payment method before it will allocate the T4 GPU.

For full training with persistent, resumable checkpoints, use the separate
Modal launcher. Start a run with the desired epoch limit, or resume an existing
run to a higher total epoch limit:

```bash
uv run modal run --detach papers/15-MACE/modal_train.py --epochs 3 --seed 123
uv run modal run --detach papers/15-MACE/modal_train.py --epochs 500 --seed 123 \
  --resume-run-id <run_id>
```

It stores run folders in the named Modal Volume `gdl-papers-mace-runs` and
commits each completed epoch. `training_state.pt` atomically records the latest
live model, latest EMA, best-validation EMA, optimizer, scheduler, completed
epoch, and RNG states. The evaluation `checkpoint.pt` is regenerated from that
snapshot on resume. If a job stops, rerun the same command with
`--resume-run-id <run_id>` from the log. Modal limits one function call to 24
hours; the resumed job retries automatically from its most recent completed
epoch after a timeout or interruption. Updates made during the current epoch
may need to be repeated.

To evaluate a Modal run locally, download its run folder first:

```bash
uv run modal volume get gdl-papers-mace-runs runs/<run_id> papers/15-MACE/runs/<run_id>
uv run python papers/15-MACE/src/evaluate.py --run <run_id> \
  --data papers/15-MACE/data/dataset_3BPA
```

Training prints a `run_id`. Use that ID with `--run`. The training command also
accepts `--seed` and `--epochs`; `--max-train-batches` limits training batches
per epoch for smoke/debug runs only. The default 500-epoch cap is a configurable
run setting, not a separately verified published stopping rule. Runs write
`config.json`, `metrics.jsonl`, the best-validation `checkpoint.pt`, and the
latest resumable `training_state.pt` under `runs/<run_id>/`.
For local interrupted runs, add `--resume-run-id <run_id>` to the training command.

The 2026-09-28 correction unpacks angular blocks before contraction and uses
the author's bias-free radial networks. Checkpoints from the previous biased
radial implementation are incompatible and must be retrained. Regression
tests now cover rotation/reflection equivariance and energy/force continuity
when an edge crosses the cutoff.

## Results

Report energy RMSE in meV and force-component RMSE in meV/angstrom. Run three
seeds and compare each three-seed mean with the published mean:

| Test set | Energy RMSE | Force RMSE |
|---|---:|---:|
| 300 K | 3.0 +/- 0.2 | 8.8 +/- 0.3 |
| 600 K | 9.7 +/- 0.5 | 21.8 +/- 0.6 |
| 1200 K | 29.8 +/- 1.0 | 62.0 +/- 0.7 |
| Dihedral slices | 7.8 +/- 0.6 | 16.5 +/- 1.7 |

Completion means all eight reproduced means are within 10% of the corresponding
published mean. This reproduction does not claim the paper's baseline
comparisons, latency, ablations, or production/foundation-model features.

## Implementation walkthrough

1. Follow `data.py`: extxyz records, split, neighbour edges, and batching.
2. Follow `model.py`: Eq. 9 atomic basis, Eq. 8 interaction, Eqs. 10–12
   contraction, two readouts, and energy-derived forces.
3. Follow `train.py`: training statistics, weighted loss, EMA, validation, and
   best-checkpoint selection.
4. Follow `evaluate.py`: reload the checkpoint and report each test condition.
5. Run the full three-seed benchmark and compare all eight RMSE means above.
