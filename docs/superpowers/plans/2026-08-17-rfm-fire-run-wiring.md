# RFM Fire Run Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist reproducible RFM Fire training runs and evaluate their held-out exact NLL through the standard paper CLI contract.

**Architecture:** Keep the existing tensor-only `train()` and all geometry, loss, divergence, and Eq. 29-31 code unchanged. Add paper-local orchestration around it: training creates a canonical run folder and artifacts; evaluation reconstructs the model and deterministic split from `config.json`, then averages per-batch NLL with sample-count weighting.

**Tech Stack:** Python 3.12, PyTorch, torchdiffeq, pandas, pytest, shared `gdl` run/checkpoint/logging helpers.

## Global Constraints

- Work only under `papers/14-RFM/` except this plan document.
- Preserve the existing learning-core implementations in `train()`, geometry, model, loss, divergence, and `negative_log_likelihood()`.
- Use `uv run` for all checks.
- Do not run long training, commit datasets/checkpoints/runs, or push.

---

### Task 1: Persist a Fire training run

**Files:**
- Modify: `papers/14-RFM/test_train.py`
- Modify: `papers/14-RFM/src/train.py`

**Interfaces:**
- Consumes: `train(...) -> TrainingResult`, `data.load_fire_csv`, `data.split_points`, and shared `gdl` run helpers.
- Produces: `train_run(...) -> tuple[str, Path, TrainingResult]` plus a CLI that prints `run_id`.

- [x] Add a failing temp-directory test asserting `config.json`, one metrics row per epoch containing both losses, and `checkpoint.pt`.
- [x] Run the focused persistence test and verify it fails because `train_run` does not exist.
- [x] Add a thin `train_run` orchestration wrapper that creates the run before training, records dataset/model/hyperparameters/seed/metric, calls the unchanged `train()`, logs returned losses, and saves the returned model.
- [x] Make `main()` call `train_run`, expose model dimensions and `--runs-base`, and print the resulting run ID.
- [x] Run `papers/14-RFM/test_train.py` and verify it passes.

### Task 2: Evaluate a saved run in weighted batches

**Files:**
- Modify: `papers/14-RFM/test_evaluate.py`
- Modify: `papers/14-RFM/src/evaluate.py`

**Interfaces:**
- Consumes: a run ID, saved config/checkpoint, deterministic Fire split, and unchanged `negative_log_likelihood(model, samples, rtol, atol)`.
- Produces: `evaluate_run(...) -> tuple[str, float]` and `--run` evaluation CLI output.

- [x] Add a failing round-trip wiring test using a temp Fire CSV and small model checkpoint.
- [x] Add a failing unequal-batch test whose expected score proves the final mean is weighted by batch size rather than batch count.
- [x] Run the focused tests and verify they fail because `evaluate_run` does not exist.
- [x] Add config loading, model reconstruction, checkpoint loading, deterministic test split recreation, device movement, validated batch iteration, and sample-count-weighted NLL aggregation.
- [x] Add a `--run` CLI with batch size, device, tolerances, runs base, and optional data-path override; retain `sample()` as the separate sampling API.
- [x] Run `papers/14-RFM/test_evaluate.py` and verify it passes.

### Task 3: Document and verify the workflow

**Files:**
- Modify: `papers/14-RFM/README.md`

- [x] Document exact default training and `--run` evaluation commands, defaults, run artifacts, the held-out NLL target, and data-path override.
- [x] Run Ruff format/check and mypy on all touched Python files.
- [x] Run `uv run pytest papers/14-RFM/` and verify the complete paper suite passes.
- [x] Review the final diff for accidental learning-core changes, datasets, checkpoints, or run artifacts.
- [x] Commit the verified wiring changes without pushing.
