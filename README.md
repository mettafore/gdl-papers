# Saved the plan in the current GDL tracker: [gdl_goals.md (line 11)](/Users/luvsuneja/Documents/wiki/Career/gdl_goals.md:11).
The three hours are allocated:
- 0:00–0:10 — rerun Section 12 CG checks
- 0:10–0:30 — Section 13: \(\nu,\eta_3\) coupling paths
- 0:30–0:50 — Section 14: radial basis/cutoff
- 0:50–1:20 — Section 15 / Eq. 8
- 1:20–1:40 — Section 16 / Eq. 9
- 1:40–2:10 — Section 17 / Eq. 10
- 2:10–2:30 — Section 18 / Eq. 11
- 2:30–2:40 — shape ledger
- 2:40–2:50 — symmetry checks
- 2:50–3:00 — recap and save
Hand derivations and .py implementation are explicitly scheduled after Pali.


gdl-papers

PyTorch reproductions of geometric deep learning papers. Each paper is implemented,
trained, evaluated against its published benchmark, and written up. One
self-contained paper per folder, sharing a small train/evaluate scaffold.

## Quickstart

```bash
uv sync                                               # install deps + the gdl package

# train with defaults
uv run python papers/00-template/src/train.py             # prints a run_id

# train with hyperparameter overrides
uv run python papers/00-template/src/train.py --lr 1e-3 --epochs 50 --hidden 32

# evaluate a specific run
uv run python papers/00-template/src/evaluate.py --run <run_id>

# tests
uv run pytest papers/00-template/
```

## Running via skills (conversational)

In Claude Code, drive any paper in natural language — the `run-paper` skill maps it
to the commands above:

- "train 00-template" — train with defaults
- "train 00-template with lr 1e-3, 50 epochs" — overrides forwarded to CLI args
- "evaluate 00-template run <run_id>" — evaluate a run

Ask `gdl-repo-map` things like "where does shared code go?" or "how do I add a paper?"

## Commit checks

Enable the versioned Git hook once per clone:

```bash
git config core.hooksPath .githooks
```

It blocks commits unless Ruff formatting and linting, per-paper type checks, and
the lightweight test suites pass. Dataset-backed tests remain opt-in.

## Layout

- `src/gdl/` — shared, paper-agnostic helpers (seed, metrics, logging, run, checkpoint)
- `papers/NN-name/` — one paper each (model, data, train, evaluate, README, runs/)
- `papers/00-template/` — the runnable template; copy it to start a new paper

## Learning from this repo

Each paper folder has its own README — what the paper does, how to run it, and the
expected results — so it works as a self-contained learning resource. New to
geometric deep learning? Start at `papers/00-template/` (the simplest), then read
each paper's README to follow the progression.

## AI operating surface

Built with **Claude Code**.

- **Rules** → [AGENTS.md](AGENTS.md)
- **Skills** → `.agents/skills/` — `run-paper` (run train/eval), `gdl-repo-map` (repo map)
- **MCP** → context7 (live library docs)
- **Docs** → [docs/spec.md](docs/spec.md) (scope + contracts), [docs/reference-index.md](docs/reference-index.md)

## Papers

**Status:** [docs/progress.md](docs/progress.md) — canonical tracker, update it when you
work on a paper.

Full list and tiers → [references/papers.md](references/papers.md).
