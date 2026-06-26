---
description: Train a paper (defaults or hyperparameter overrides) via the run-paper skill
---

Use the `run-paper` skill to TRAIN a paper in this repo.

Arguments: `$ARGUMENTS` — the paper folder name followed by optional colloquial
hyperparameter overrides, e.g. `00-template` or `00-template lr 1e-3, 50 epochs`.

Steps:
1. Read `papers/<paper>/README.md` — the `## Run` (train command) and
   `## Hyperparameters` (valid CLI args) sections.
2. Map any overrides to CLI args (only args listed in `## Hyperparameters`):
   "lr 1e-3, 50 epochs" → `--lr 1e-3 --epochs 50`.
3. Run `uv run python papers/<paper>/train.py [args]`.
4. Report the final loss and the printed `run_id`.
