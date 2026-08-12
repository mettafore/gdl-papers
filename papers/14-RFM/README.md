# 14-RFM — Flow Matching on General Geometries

Learning-first reproduction of Chen and Lipman (ICLR 2024). The current slice
fits the **Fire** location distribution on the unit sphere `S^2`.

## Files

- `nbs/spherical.ipynb` — validated closed-form sphere geometry and Eq. 13.
- `nbs/fire_eda.ipynb` — Fire dataset exploration scaffold.
- `src/data.py` — Fire loading, coordinate conversion, and seeded splits.
- `test_data.py` — AI-owned data contracts; initially red by design.

Geometry, model, training, and evaluation modules are added only after the data
contracts pass.

## Data

- Dataset: Fire locations compiled by Mathieu and Nickel (2020) from EOSDIS.
- Paper size: 12,809 total points.
- Source: the authors' [`data.zip`](https://rtqichen.com/manifold_data/data.zip).
- Expected file: `fire.csv`, with header and rows ordered as latitude, longitude
  in degrees.
- Split: deterministic 80% train, 10% validation, 10% test.

Data files are local inputs and must not be committed.

## Run

Focused data tests:

```bash
uv run pytest papers/14-RFM/test_data.py -q
```

Open `papers/14-RFM/nbs/fire_eda.ipynb` for exploration after downloading the
CSV. Training and evaluation commands will be added with those slices.

## Completion

The three-day Fire experiment is complete when one reproducible run generates
unit-sphere samples that visibly recover the Fire distribution and outperform
the uniform-sphere baseline under the chosen evaluation.

Paper reference: RFM with geodesics reports test NLL **-1.86 +/- 0.11** over five
runs. Matching that number is not the completion gate for the M1 reproduction.
