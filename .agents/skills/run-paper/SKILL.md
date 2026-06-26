---
name: run-paper
description: >
  Run a paper's training or evaluation in the gdl-papers repo. Use when the user
  says "train <paper>", "evaluate <paper>", "run <paper> with lr/epochs/...", or
  asks to sweep a hyperparameter. Reads the paper's README ## Run + ## Hyperparameters,
  builds the command, forwards any overrides to CLI args, runs it, and reports the
  result. NOT for explaining repo layout (that's gdl-repo-map).
---

# run-paper

Generic runner for any paper under `papers/NN-name/`. The paper owns its commands
and knobs (in its README + `train.py`); this skill just reads them and runs.

## Steps

1. Resolve the paper folder: `papers/<paper>/` (e.g. `00-template`).
2. Read `papers/<paper>/README.md` — the `## Run` section gives the literal
   train/evaluate commands; `## Hyperparameters` lists the tunable CLI args.
3. **Train** ("train `<paper>`" [+ overrides]):
   - Start from the README's train command.
   - Map colloquial overrides to CLI args: "lr 1e-4, 50 epochs" → `--lr 1e-4 --epochs 50`.
     Only use args listed in `## Hyperparameters`.
   - Run via `uv run python papers/<paper>/train.py [args]`.
   - Report the final loss and the printed `run_id`.
4. **Evaluate** ("evaluate `<paper>` run `<id>`"):
   - Run `uv run python papers/<paper>/evaluate.py --run <id>`.
   - Report the metric.

## Rules
- Always use `uv run` (never bare python).
- Never invent hyperparameters not in the paper's `## Hyperparameters`.
- Don't hardcode any paper's schema here — read it from the README each time.
