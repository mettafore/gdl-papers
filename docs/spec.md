# Spec — Training Scaffold

## Intent

A reusable train → evaluate scaffold so every paper folder runs the same way. HW1
uses a trivial model on a real-but-tiny dataset (Iris) to prove the loop works;
real models (EGNN, …) reuse the same scaffold later. Deliverable is the harness,
not the science.

Guiding principle that drives every decision below: **a run records everything
needed to reproduce and evaluate it** — config, metrics, weights, seeds — in one
folder.

## Setup (packaging)

`src/gdl/` is an **installed package** so `from gdl import ...` works from any script,
test, or notebook (and for the autoresearch agent). Add to `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/gdl"]
```

Then: `uv add scikit-learn` · `uv add --dev pytest` · `uv sync` (installs `gdl`
editable — edits take effect live).

## Acceptance criteria

1. Training runs in seconds, produces a **decreasing loss**, and writes a run folder
   `papers/00-template/runs/<run_id>/` containing `config.json`, `metrics.jsonl`,
   `checkpoint.pt`.
2. Evaluation, given a `<run_id>`, rebuilds the model from that run's `config.json`,
   loads its checkpoint, and reports the metric (accuracy) on the **same** held-out
   split training used.
3. The test suite passes: smoke (train runs + loss drops), unit (metrics correct),
   split (train/test identical across train & evaluate).
4. Shared helpers live in `src/gdl/`, import-clean (`from gdl import ...`):
   `set_seed`, the metric registry (`accuracy`, `mae`, `rmse`), `log_metrics`,
   `save_checkpoint` / `load_checkpoint`, `load_run_config`.
5. Nothing in `src/gdl/` references the template or any specific paper.
6. `papers/00-template/README.md` follows the full README contract (below):
   `## Files`, `## Data`, `## Hyperparameters`, `## Run`, `## Results`.
7. `train.py` takes hyperparams as CLI args with defaults; e.g.
   `... train.py --lr 1e-3 --epochs 50` runs and respects them. The `run-paper`
   skill forwards colloquial overrides to these args.
8. The split is deterministic and shared: both `train.py` and `evaluate.py` use one
   seeded `data.load_split(seed)` with an **explicit** `random_state` — no leakage.
9. The metric is config-driven: `config.json` names the metric; `evaluate.py` looks
   it up in the registry (simple papers) — `accuracy` is not hardcoded.
10. Both skills live at `.agents/skills/<name>/SKILL.md` (+ `.claude/skills/` symlinks)
    and activate on the right triggers: `run-paper` (procedural) on "train/evaluate
    <paper>"; `gdl-repo-map` (knowledge) on "where does code go / how do I add a paper /
    run-folder layout" — a discovery surface, not a dump.
11. Root `README.md` explains what the repo is, a quickstart (incl. a **hyperparameter
    override** train example), how to run via the skills (natural language),
    the layout, and where rules / skills / MCP / docs live. It also points out that
    **each paper has its own README** (what the paper is, how to run, expected results) —
    a self-contained learning resource for students new to that paper/area.
12. `docs/reference-index.md` is a curated doc index with two parts:
    (a) **External docs** (static bookmarks) — PyTorch, scikit-learn, uv, pytest,
    GDL proto-book; prefer context7 MCP for live API lookups.
    (b) **Living context — keep current** — the in-repo docs that must not rot, each
    with its update trigger: `docs/progress.md` (**always** when paper work starts,
    finishes, or pauses), `docs/spec.md` (per feature), `AGENTS.md` (when commands/
    paths change), `references/papers.md` (tier or repo path changes), per-paper
    `README.md` (results/run instructions), per-paper `notes.md` (experiments),
    the skills (when conventions change), root `README.md` (links to progress.md).

## Scaffold detail

```
src/gdl/                       # shared, paper-agnostic — pulled BY papers, never calls them
  __init__.py
  seed.py        set_seed(seed) -> None                  # random, numpy, torch
  metrics.py     accuracy / mae / rmse — all (output, target) -> float; METRICS = {name: fn}
  run_log.py     log_metrics(metrics: dict, step, run_dir) -> None
                 # dual sink: prints to console + appends a line to run_dir/metrics.jsonl.
  run.py         new_run_dir(paper_dir, config, base="runs") -> (run_id, run_dir)
                 # run_id = UTC timestamp "YYYYMMDD-HHMMSS"; makes <base>/<id>/, writes config.json.
                 # base is a param so tests redirect to a tmp dir (no litter).
                 load_run_config(paper_dir, run_id) -> dict
  checkpoint.py  save_checkpoint(model, run_dir) ; load_checkpoint(model, run_dir)
papers/00-template/
  src/model.py   MLP(in_dim, hidden, out_dim) — small nn.Module
  src/data.py    load_split(seed=42) -> (X_train, y_train, X_test, y_test, in_dim, out_dim).
                 X float32, y long (CE). ONE shared seeded split (explicit random_state)
                 imported by both train.py and evaluate.py — identical split, no leakage.
                 data REPORTS its own in_dim/out_dim (train never hardcodes shapes).
                 Iris-only here; --data is a forward-looking README convention, not implemented.
  src/train.py   argparse(--lr,--epochs,--hidden,--seed) -> set_seed -> load_split ->
                 new_run_dir FIRST (writes config.json) -> build MLP -> Adam -> CE loss ->
                 log_metrics(..., run_dir) per epoch -> save_checkpoint. Prints run_id at end.
  src/evaluate.py  --run <id> -> load_run_config -> rebuild MLP from cfg -> load_checkpoint ->
                 load_split(cfg.seed) -> metric = METRICS[cfg.metric] -> print.
  test_scaffold.py   smoke + unit + split-determinism tests
  README.md      the canonical README contract (below)
  runs/          per-run folders (gitignored): <run_id>/{config.json,metrics.jsonl,checkpoint.pt}
```

### `config.json` schema (the train ↔ evaluate contract)

train writes it; evaluate reads it. Grouped by origin — `data` = facts the dataset
dictates, `model` = what you chose:

```json
{
  "data":  {"dataset": "iris", "in_dim": 4, "out_dim": 3, "test_size": 0.2},
  "model": {"hidden": 16},
  "hyperparams": {"lr": 0.01, "epochs": 100},
  "seed": 42,
  "metric": "accuracy"
}
```

evaluate rebuilds via `MLP(in_dim=cfg["data"]["in_dim"], hidden=cfg["model"]["hidden"],
out_dim=cfg["data"]["out_dim"])`, scores via `METRICS[cfg["metric"]]`, splits via
`load_split(cfg["seed"])`. `in_dim`/`out_dim` live under `data` (the dataset dictates
them); `hidden` under `model` (you chose it).

### evaluate is per-paper, not a dispatcher

Each paper owns its `evaluate.py` (symmetric with `train.py`). Control flows
**paper → `gdl` helpers** (the paper pulls shared utilities); there is **no general
evaluate that calls into papers**. Uniform interface (`--run <id>`, prints a metric),
custom internals:

- **Simple papers** (Iris, EGNN) — metric via the registry: `METRICS[cfg["metric"]]`.
- **Complex papers** (MACE = energy+forces, RFM = generative) — compute their own
  metric inline in their `evaluate.py`; still declare + record the metric name/value.
  Do not force them through the registry.

## Per-paper README contract + `run-paper` skill

Each paper folder differs (architecture, harness, eval scheme), so the *procedure*
is not hardcoded in shared code. Each paper documents itself; one skill reads that
doc and runs it.

- **README contract** — every `papers/NN-name/README.md` follows this full structure,
  in this order. `papers/00-template/README.md` is the canonical example papers copy:

  ```markdown
  # 00-Template — <one-line: what this paper is>
  (real papers: arXiv link + citation)

  <2–3 sentences: what the model does, what task it solves>

  ## Files
  - src/model.py     — the model (the swappable slot)
  - src/data.py      — dataset loading + the shared seeded split
  - src/train.py     — training entry point (CLI hyperparams)
  - src/evaluate.py  — load a run, print the metric

  ## Data
  - dataset: Iris (sklearn, bundled — no download)
  - task:    3-class classification, 4 features, 150 samples
  - split:   train/test 80/20, random_state=42 (deterministic; train + evaluate
             share one seeded split in data.py — no leakage)
  - override: --data <name>  (when a paper supports alternatives)

  ## Hyperparameters
  - --lr      (default 1e-2)   learning rate
  - --epochs  (default 100)    training epochs
  - --hidden  (default 16)     hidden width
  - --seed    (default 42)     RNG seed

  ## Run
  - train:    uv run python papers/00-template/train.py
  - evaluate: uv run python papers/00-template/evaluate.py --run <run_id>

  ## Results
  - expected metric (e.g. test accuracy ~0.9X) — what "working" looks like.
    For real papers: the published benchmark to match. Autoresearch reads this
    as the number to beat.
  ```

  The `## Run`, `## Hyperparameters`, and `## Data` sections are the machine-readable
  surfaces the `run-paper` skill and autonomous agents parse.

- **`run-paper` skill** — generic orchestration with override pass-through. Lives at
  `.agents/skills/run-paper/SKILL.md` (tool-neutral, open-standard format), with a
  `.claude/skills/run-paper` → `../../.agents/skills/run-paper` symlink so Claude Code
  discovers it. Invoked by **natural language** (no slash commands). Two operations:
  - "train `<paper>` [overrides]" → read the paper's README `## Run`, build the train
    command, **forward colloquial overrides to CLI args** ("lr 1e-4, 50 epochs" →
    `--lr 1e-4 --epochs 50`), execute, report loss + run folder.
  - "evaluate `<paper>` run `<id>`" → execute the evaluate command, report the metric.

  The skill forwards overrides **without hardcoding any paper's schema** — the paper
  owns the knobs (defaults in `train.py`, documented in its README); the skill passes
  through whatever the caller says. Decoupled, yet a researcher *or* an autoresearch
  agent can sweep hyperparameters and data over the same entry point. The skill holds
  the *pattern*; the README + `train.py` hold the *specifics*. Adding a paper = write
  its README + train/eval scripts; the skill already works. No per-paper skills
  (avoids a skill graveyard).

- **Autoresearch-compatible by construction** — a config-driven metric + persisted run
  folders + a parameterized `train.py` entry + git commit per improvement means an
  autonomous agent (read code → tweak param → short run → measure from `metrics.jsonl`
  / `config.json` → commit or rollback) drives the same harness a human uses.

- **`gdl-repo-map` skill** (knowledge) — a discovery surface explaining repo layout,
  the `src/` vs `papers/` rule, the run-folder/config schema, and how to add a paper.
  Points to `docs/spec.md` / `AGENTS.md` for detail, and to `docs/reference-index.md`
  for external library/API docs — rather than duplicating any of them. Lives at
  `.agents/skills/gdl-repo-map/SKILL.md` (+ `.claude/skills/` symlink). Its
  `description` names the triggers ("where does code go / how do I add a paper / run
  folder layout") and the boundary (NOT for running training — that's `run-paper`).

## Constraints

- Python 3.12, **uv** only (`uv run`, `uv add`). No bare `python`, no pip/conda.
- Deps: torch (pinned) + `scikit-learn` (Iris) + `pytest` (dev). Nothing heavier.
- `gdl` is an installed package (see Setup) so `from gdl import ...` works anywhere.
- Keep `src/gdl/` thin and paper-agnostic — it is pulled by papers, never calls them.
- `src/gdl/` grows only when duplication across 2+ papers proves a helper shared.

## Non-goals

- No EGNN / real model, no QM9 / large datasets, no benchmark numbers (HW2+).
- No graphs / pooling / equivariance — trivial MLP only.
- No W&B / MLflow wired now — console + JSONL only (W&B slot is pluggable, added later).
- No custom dashboard or chatbot app.
- AGENTS.md — handled separately. (Both skills — `run-paper` procedural + `gdl-repo-map`
  knowledge — the per-paper README contract, the root `README.md`, and
  `docs/reference-index.md` ARE in scope.)

## Verification

- `uv run python papers/00-template/train.py` → loss drops + a `runs/<id>/` folder
  appears with config.json, metrics.jsonl, checkpoint.pt.
- `uv run python papers/00-template/evaluate.py --run <id>` → prints accuracy on the
  held-out split.
- `uv run pytest papers/00-template/` → green (incl. the split-determinism test).
- Say "train 00-template" in a fresh session → the `run-paper` skill fires, runs
  training, reports the run_id (verify the `.claude/skills/` symlink resolves; if not,
  fall back to placing the skill directly under `.claude/skills/`).
- Say "evaluate 00-template run <id>" → the skill fires, runs evaluation, reports the metric.
