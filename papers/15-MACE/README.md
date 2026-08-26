# 15-MACE — higher-order equivariant message passing

Learning-first reproduction of the MACE `L=2` result on 3BPA. The scientific
target is the MACE row in Table 2 of Batatia et al. (NeurIPS 2022), not the
current production MACE package.

The source files are deliberately incomplete. `TODO(luv)` marks the
learning-owned implementation; strict `xfail` tests turn into failures as soon
as a stub is replaced by an incorrect implementation.

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

ASE reads extxyz and constructs the non-periodic 5-angstrom neighbour topology.
PyG batches graphs, offsets their atom indices, and assigns each atom to a graph.
The data pipeline stores positions, sender/receiver indices, and zero image
shifts—but never precomputed edge vectors.

The model must derive each vector during `forward` as
`positions[receiver] - positions[sender] + shifts`, after enabling gradients on
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

The commands below are contracts until their learning-owned stubs are filled:

```bash
uv run python papers/15-MACE/src/train.py --data /path/to/dataset_3BPA
uv run python papers/15-MACE/src/evaluate.py --run <run_id> \
  --data /path/to/dataset_3BPA
uv run pytest papers/15-MACE/
```

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

## Learning order

1. Make `test_route_enumeration_1_times_2` pass.
2. Make the radial basis and cutoff test pass.
3. Build Eq. 8 atomic features and one Eq. 11 message block.
4. Pass rotation, translation, and permutation tests on the tiny graph.
5. Add the second interaction, scalar readout, and autograd forces.
6. Make the weighted loss and tiny overfit/round-trip tests pass.
7. Train reduced debug models before launching the three-seed paper recipe.
