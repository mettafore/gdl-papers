# 00-Template — the scaffold template

A trivial MLP trained on Iris. Not real science — it exists to prove the
train → evaluate loop and to be the **canonical shape every paper copies**. To
start a new paper: copy this folder, swap `model.py` (and `data.py`), keep the
same file layout and this README structure.

## Files
- `model.py`     — the model (the swappable slot); here a small MLP
- `data.py`      — dataset loading + the shared seeded split
- `train.py`     — training entry point (CLI hyperparams)
- `evaluate.py`  — load a run, print the metric
- `test_scaffold.py` — smoke + unit + split-determinism tests

## Data
- dataset: Iris (scikit-learn, bundled — no download)
- task:    3-class classification, 4 features, 150 samples
- split:   train/test 80/20, `random_state=42`, stratified (deterministic;
           train + evaluate share one seeded split in `data.py` — no leakage)
- override: `--data <name>` — forward-looking convention, not implemented here

## Hyperparameters
- `--lr`      (default 1e-2)   learning rate
- `--epochs`  (default 100)    training epochs
- `--hidden`  (default 16)     hidden width
- `--seed`    (default 42)     RNG seed

## Run
- train:    `uv run python papers/00-template/train.py`
- evaluate: `uv run python papers/00-template/evaluate.py --run <run_id>`

`train.py` prints a `run_id` and writes `runs/<run_id>/` (config.json,
metrics.jsonl, checkpoint.pt). Pass that id to `evaluate.py`.

## Results
- expected: test **accuracy ≈ 1.0** (Iris is easy; this just shows the loop works).
  For real papers this section names the published benchmark to match —
  the number an autoresearch agent reads as the target to beat.
