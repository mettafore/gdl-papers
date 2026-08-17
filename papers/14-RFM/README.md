# 14-RFM — Flow Matching on General Geometries

Learning-first reproduction of Chen and Lipman (ICLR 2024). The current slice
fits the **Fire** location distribution on the unit sphere `S^2`.

## Files

- `nbs/spherical.ipynb` — validated closed-form sphere geometry and Eq. 13.
- `nbs/fire_eda.ipynb` — Fire dataset exploration scaffold.
- `src/data.py` — Fire loading, coordinate conversion, and seeded splits.
- `src/geometry.py` — closed-form sphere geometry and conditional flow.
- `src/model.py` — time-conditioned tangent velocity field.
- `src/loss.py` — RCFM regression objective.
- `src/train.py` — training CLI and canonical run persistence.
- `src/evaluate.py` — sampling API and exact held-out NLL CLI.
- `test_*.py` — AI-owned contracts for data, geometry, model, loss, training,
  sampling, and evaluation.

## Data

- Dataset: Fire locations compiled by Mathieu and Nickel (2020) from EOSDIS.
- Paper size: 12,809 total points.
- Source: the authors' [`data.zip`](https://rtqichen.com/manifold_data/data.zip).
- Expected file: `fire.csv`, with header and rows ordered as latitude, longitude
  in degrees.
- Split: deterministic 80% train, 10% validation, 10% test.

Data files are local inputs and must not be committed.

## Run

Run from the repository root with the CLI defaults:

```bash
uv run python papers/14-RFM/src/train.py
```

The defaults are:

- `--epochs 3`
- `--lr 0.001`
- `--hidden-dim 64`
- `--n-layers 10`
- `--device cpu`
- `--data-path papers/14-RFM/data/fire.csv`
- `--seed 42`
- `--runs-base runs`

Training prints a `run_id` and writes
`papers/14-RFM/runs/<run_id>/{config.json,metrics.jsonl,checkpoint.pt}`. Evaluate
that exact run with:

```bash
uv run python papers/14-RFM/src/evaluate.py --run <run_id>
```

Evaluation recreates the seeded 80/10/10 split and reports exact NLL on the
held-out test points. It uses batches of 64 by default because exact Jacobian
traces are memory intensive. Override this with `--batch-size`; use
`--data-path` if the CSV has moved since training.

Suggested first Fire run:

```bash
uv run python papers/14-RFM/src/train.py \
  --epochs 10 \
  --lr 0.001 \
  --device cpu \
  --seed 42
```

Full paper tests:

```bash
uv run pytest papers/14-RFM/ -q
```

Open `papers/14-RFM/nbs/fire_eda.ipynb` for exploration after downloading the
CSV.

## Hyperparameters

The training CLI accepts `--epochs`, `--lr`, `--hidden-dim`, `--n-layers`,
`--device`, `--data-path`, `--seed`, and `--runs-base`. Evaluation accepts
`--run`, `--batch-size` (default 64), `--device` (default CPU), `--rtol` and
`--atol` (both default `1e-7`), `--data-path`, and `--runs-base`.

## Completion

The three-day Fire experiment is complete when one reproducible run generates
unit-sphere samples that visibly recover the Fire distribution and outperform
the uniform-sphere baseline under the chosen evaluation.

Paper reference: RFM with geodesics reports test NLL **-1.86 +/- 0.11** over five
runs. Matching that number is not the completion gate for the M1 reproduction.
