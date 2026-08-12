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

Exploration and reproducible training are separate. The first scaffold is
deliberately limited to the data slice:

- `papers/14-RFM/nbs/fire_eda.ipynb` explores the raw Fire data and its sphere
  representation.
- `papers/14-RFM/src/data.py` begins the reproducible pipeline.
- `papers/14-RFM/test_data.py` provides AI-authored red/green contracts while
  Luv fills the learning-critical data bodies.

Geometry, model, training, and evaluation files are scaffolded only after the
data slice passes. This avoids empty-file clutter and premature interfaces.

## Files and responsibilities

- `README.md`: files, data provenance, hyperparameters, commands, completion
  criteria, and the paper benchmark.
- `nbs/fire_eda.ipynb`: dataset shape, missing values, latitude/longitude ranges,
  split counts, unit-sphere conversion check, and a geographic scatter plot.
- `src/data.py`: load the official `fire.csv`, validate it, convert latitude and
  longitude to Cartesian unit vectors, and create deterministic 80/10/10 splits.
- Later slices add `geometry.py`, `model.py`, `train.py`, `evaluate.py`, and their
  tests one at a time after the preceding contract passes.

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

## Test ownership

AI owns the test harness, invariants, edge cases, tolerances, execution, and
failure triage. Luv owns correctness-determining implementation bodies,
including coordinate conversion, geometry, tangent projection, the RCFM loss,
and training logic. Tests may state expected behavior directly but must not
encode the implementation method.

The first data-scaffold tests intentionally fail until Luv implements each TODO.
They check:

- deterministic 80/10/10 splits with no overlap;
- latitude/longitude conversion returns `(n, 3)` unit vectors;
- known landmark coordinates map to the expected sphere axes;
- malformed, non-finite, or out-of-range rows fail clearly.

Later test slices cover sphere primitives, tangent model output, RCFM loss,
training artifacts, and evaluation samples.

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
