# RFM Fire Experiment Design

## Goal

Build one focused Riemannian Flow Matching experiment on the Fire dataset from
Chen and Lipman (ICLR 2024). The experiment should demonstrate that a learned
time-dependent tangent vector field transports a simple base distribution on
the unit sphere toward the observed fire-location distribution.

This is a three-day, learning-first reproduction. It preserves the paper's
geometry and objective while reducing the authors' width-2048, 10-layer,
batch-8192 setup to something practical on an M1 Mac. Matching the paper's
`-1.86 +/- 0.11` test NLL is a reference target, not the completion gate.

## Scope

The implementation covers only the Fire dataset on `S^2`. Bunny meshes,
spectral premetrics, Hydra, Lightning, multi-dataset abstractions, and general
manifold interfaces are excluded.

Exploration and reproducible training are separate:

- `papers/14-RFM/nbs/fire_eda.ipynb` explores the raw Fire data and its sphere
  representation.
- Python files under `papers/14-RFM/src/` implement the reproducible experiment.
- `papers/14-RFM/test_scaffold.py` provides red/green contracts while Luv fills
  the learning-critical bodies.

## Files and responsibilities

- `README.md`: files, data provenance, hyperparameters, commands, completion
  criteria, and the paper benchmark.
- `nbs/fire_eda.ipynb`: dataset shape, missing values, latitude/longitude ranges,
  split counts, unit-sphere conversion check, and a geographic scatter plot.
- `src/data.py`: load the official `fire.csv`, validate it, convert latitude and
  longitude to Cartesian unit vectors, and create deterministic 80/10/10 splits.
- `src/geometry.py`: move the already validated sphere primitives from
  `spherical.ipynb` into importable functions without changing their mathematics.
- `src/model.py`: a time-conditioned ambient MLP whose output is projected into
  `T_x S^2`.
- `src/train.py`: sample `x0`, `x1`, and `t`; construct `x_t` and the conditional
  velocity target; compute Eq. 10; optimize; and write the standard run artifacts.
- `src/evaluate.py`: load a run, integrate the learned field outside the training
  autograd graph, verify samples remain on `S^2`, and report evaluation outputs.
- `test_scaffold.py`: data, geometry, model, loss, split, and train/evaluate
  contracts.

## Data flow

The official `fire.csv` stores latitude/longitude locations. Loading converts
each row into a three-coordinate unit vector. A seeded split produces 80% train,
10% validation, and 10% test partitions, matching the paper's stated protocol.

For each training batch:

1. Draw target points `x1` from the Fire training split.
2. Draw base points `x0` from the uniform distribution on `S^2`.
3. Draw times `t` uniformly from `[0, 1)`.
4. Construct the closed-form geodesic point `x_t`.
5. Construct the Eq. 13 conditional tangent velocity.
6. Regress the model's tangent output against that target using Eq. 10.

## Contracts and failure handling

- Raw data must have two finite coordinate columns and valid latitude/longitude
  ranges; invalid rows fail with a clear error rather than being silently fixed.
- Cartesian points, geodesic points, and generated samples must have shape
  `(n, 3)` and unit norm within a documented numerical tolerance.
- Per-record scalar quantities use shape `(n, 1)` to avoid accidental `(n, n)`
  broadcasting.
- The conditional target and model output must be tangent at the current point.
- The conditional field excludes `t = 1`, where the log-schedule derivative is
  singular.
- Evaluation reconstructs the model from `config.json` and uses the same seeded
  data split as training.

## Tests

The initial scaffold tests intentionally fail until Luv implements each TODO.
They check:

- deterministic 80/10/10 splits with no overlap;
- latitude/longitude conversion returns `(n, 3)` unit vectors;
- sphere primitives retain the notebook invariants;
- model output has shape `(n, 3)` and is tangent at `x`;
- one RCFM loss call returns a finite scalar and supports backpropagation;
- a short training smoke test lowers loss and writes standard run artifacts;
- evaluation loads a run and generated samples remain on the sphere.

## Completion criteria

The Fire experiment is complete when one reproducible run:

- trains without geometry or broadcasting failures;
- produces finite validation and test evaluation values;
- generates points that remain on `S^2`;
- visibly recovers the concentrated geographic structure of the Fire data;
- performs better than the uniform-sphere baseline under the chosen evaluation;
- records settings, seed, plots, and results for later blog use.

Exact reproduction of the paper's five-seed NLL is explicitly out of scope for
the three-day build.

## Provenance and deliberate deviations

The structure follows `papers/00-template`. Geometry and training behavior are
checked against `facebookresearch/riemannian-fm`, especially
`configs/experiment/fire.yaml`, `manifm/datasets.py`, `manifm/model/arch.py`, and
`manifm/model_pl.py`.

Deliberate deviations from the authors' experiment are a smaller MLP, smaller
batch size, CPU/MPS execution, one primary seed, plain PyTorch instead of
Lightning/Hydra, and no required EMA for the first successful run.
