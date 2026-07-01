# AGENTS.md — Operating Rules for gdl-papers

PyTorch reproductions of geometric deep learning papers — one self-contained
paper per folder, each trained and benchmarked against its published numbers.
Human overview in [README.md](README.md); current scope in [docs/spec.md](docs/spec.md).

## Environment

- Python **3.12**, managed with **uv** (`uv.lock` is the source of truth).
- Sync deps: `uv sync`
- Run anything: `uv run python ...` (never call a bare `python` — it bypasses the venv).

## Commands

> Intended state — filled in as the scaffold lands. Update when a command changes.

| Task | Command |
|------|---------|
| Install / sync deps | `uv sync` |
| Add a dependency | `uv add <pkg>` (`--dev` for test/tooling) |
| Run training (template) | `uv run python papers/00-template/train.py` (prints a `run_id`) |
| Run evaluation (template) | `uv run python papers/00-template/evaluate.py --run <run_id>` |
| Run tests | `uv run pytest papers/00-template/` |
| Run a paper via skill | say "train <paper> [overrides]" / "evaluate <paper> run <id>" (the `run-paper` skill) |

> Run scripts **by path**, not `-m` — folder names like `00-template` aren't valid
> Python module names.

## Key paths

- `src/gdl/` — **shared, paper-agnostic** toolbox: `seed`, metric registry
  (`metrics`), `run_log` (console + JSONL), `run` (run folders + config),
  `checkpoint`. Pulled by papers; never calls them. Installed package — see Setup
  in `docs/spec.md` (`from gdl import ...`).
- `papers/NN-name/` — one self-contained paper each. Owns its `model.py`,
  `data.py`, `train.py`, `evaluate.py`, `README.md`. Each `evaluate.py` is the
  driver (no central dispatcher).
- `papers/NN-name/runs/<run_id>/` — per run: `config.json` (full recipe),
  `metrics.jsonl`, `checkpoint.pt`. Gitignored. `evaluate.py --run <id>` reads it.
- `papers/00-template/` — reference scaffold (trivial MLP on Iris). Copy to start a
  new paper; its `README.md` is the canonical contract (`## Files/Data/Hyperparameters/Run/Results`).
- `.agents/skills/run-paper/` — the procedural skill (canonical, tool-neutral);
  `.claude/skills/run-paper` is a symlink to it.
- `docs/` — durable truth: `spec.md` (scope), `reference-index.md` (doc links).
- `references/papers.md` — the paper list with arXiv IDs and tiers.
- `course/` — AI-assisted-coding coursework. Gitignored, **not part of the research artifact.**

## The one architectural rule

**Papers are independent. `src/gdl/` only grows when duplication across 2+ papers
proves a helper is shared.** Do not hoist paper-specific logic (data loading,
training steps, losses) into `src/`. A thin `src/` is a healthy `src/`.

When in doubt: *Can this function run without knowing which paper called it?*
Yes → `src/gdl/`. No → the paper's folder.

## Adding a paper

Copy `papers/00-template/` (the living, runnable template), swap the model, and follow
its README contract + the `config.json` schema in [docs/spec.md](docs/spec.md). Don't
invent a new layout — 00-template is the canonical shape every paper copies.

## Tools

- **MCP: context7** — use for live PyTorch / torch_geometric / library docs.
  Prefer it over training-data recall for API questions.

## Always / Ask first / Never

**Always**
- Run `uv run pytest papers/<paper>/` before committing scaffold or training changes.
- Use `uv run` for execution and `uv add` for dependencies.
- Match existing patterns in the paper folder you're editing.
- Commit at the end of every agent turn that touched files: run `uv run ruff format <files touched this turn>` and `uv run ruff check <same files>`, fix what it flags — except lint errors inside a `NotImplementedError` TODO stub (expected until the user implements it, leave those) — then commit.

**Ask first**
- Adding a new dependency.
- Changing the `src/` vs `papers/` boundary, or extracting shared code.
- Starting a new paper folder.
- Long training runs (full dataset, many epochs).

**Never**
- Commit datasets, checkpoints, or W&B runs (not always caught by `.gitignore`).
- Hoist paper-specific code into `src/gdl/`.
- Push or open PRs unless explicitly asked.
- Write a paper's implementation code (model/data/train bodies) for the user. The
  point of this repo is the user learning the papers by implementing them. When
  asked to "scaffold", give a skeleton only — signatures, return-shape contract,
  docstrings, TODO steps, guiding questions — bodies left `NotImplementedError`.
  Write real implementations only when explicitly asked; offer review *after* they
  implement, not before.
