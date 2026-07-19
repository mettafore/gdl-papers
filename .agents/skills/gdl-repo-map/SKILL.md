---
name: gdl-repo-map
description: >
  Explains how the gdl-papers repo is organized. Use when asked where code should
  go (src/gdl vs papers/), how to add a new paper, what the run-folder or
  config.json layout is, or how train/evaluate/skills fit together. A map for
  navigating and extending the repo — NOT for running training (that's run-paper).
---

# gdl-papers — repo map

## Layout
- `src/gdl/` — shared, paper-agnostic helpers (`seed`, `metrics` + `METRICS`
  registry, `run_log`, `run`, `checkpoint`). Installed package → `from gdl import ...`.
  Pulled BY papers; never calls into them.
- `papers/NN-name/` — one self-contained paper each: code in `src/`
  (`model.py`, `data.py`, `train.py`, `evaluate.py`), tests (`test_*.py`) and
  `README.md` at the paper root.
- `papers/NN-name/runs/<run_id>/` — per run: `config.json`, `metrics.jsonl`,
  `checkpoint.pt` (gitignored). `evaluate.py --run <id>` reads it.

## The one rule
Papers are independent. `src/gdl/` grows only when 2+ papers prove a helper is
shared. Paper-specific logic (data, training step, loss, metric) stays in the paper.
Each `evaluate.py` is the driver — it pulls `gdl` helpers; there is no central
dispatcher.

## config.json (train ↔ evaluate contract)
`data` = facts the dataset dictates (in_dim, out_dim, dataset, test_size);
`model` = what you chose (hidden); plus `hyperparams`, `seed`, `metric`.
evaluate rebuilds the model from `data`+`model`, scores via `METRICS[metric]`,
splits via `load_split(seed)`.

## Adding a paper
Copy `papers/00-template/`, swap the model, follow its README contract + the
config.json schema. Don't invent a new layout.

## Read more
- Full scope + contracts → `docs/spec.md`
- Operating rules → `AGENTS.md`
- External library/API docs → `docs/reference-index.md` (or the context7 MCP)
- Running train/eval → the `run-paper` skill
