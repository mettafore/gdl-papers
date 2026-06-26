# Decisions

Durable record of the non-obvious architectural calls — so they don't get
re-litigated, and the next agent/human knows *why*, not just *what*. Append as new
decisions arise (EGNN/MACE will add some).

---

## 1. The repo is the homework; the model is trivial on purpose
Use the real `gdl-papers` project for the AI-assisted-coding course, not a throwaway
app. The HW1 deliverable is the **harness + operating surface**, so the model in the
slot is a trivial MLP on Iris.
**Why:** lessons learned on a throwaway evaporate; the context surfaces (AGENTS.md,
skills) only earn their keep on a real domain. The model isn't where the learning is —
the scaffold and (later) the novel extension are.
**Rejected:** a todo/bookmarks app (learning wouldn't transfer); building a real model
in HW1 (ML difficulty would swamp the surface lesson — deferred to HW2).

## 2. Per-paper `train.py` / `evaluate.py`, not a shared training script
Each paper owns its training and evaluation entry points. The shared layer (`src/gdl`)
is *pulled by* papers, never calls into them. No central dispatcher.
**Why:** the training *step* varies fundamentally across papers — EGNN regresses a
scalar, MACE needs gradients of energy w.r.t. coordinates (forces), RFM is a generative
flow-matching loss. A "model fits into one fixed loop" design breaks on the 2nd paper.
**Rejected:** a general `train.py` with a swappable model slot (would need `if paper ==
...` branches or a config zoo — the platform trap).

## 3. `src/gdl/` stays thin and grows only on proven duplication
Shared helpers are added only when 2+ papers demonstrably need the same thing. Today:
`seed`, `metrics` (+ registry), `run_log`, `run`, `checkpoint`.
**Why:** abstracting ahead of evidence means abstracting against imagined papers; those
abstractions get torn up when the real paper arrives.
**Test:** *Can this run without knowing which paper called it?* Yes → `src/gdl`. No →
the paper folder.

## 4. A run records everything needed to reproduce + evaluate it
Each run is a folder `papers/NN/runs/<run_id>/` holding `config.json`, `metrics.jsonl`,
`checkpoint.pt`. `evaluate.py --run <id>` reads `config.json` to rebuild the model.
**Why:** one principle resolves three problems at once — checkpoint can't rebuild the
model (config carries the dims), analytics (metrics persist), and autoresearch (an agent
reads the metric to decide commit/rollback). The run folder is the atomic unit.
**Rejected:** bare `state_dict` checkpoints (lose the rebuild recipe); a separate
metrics DB or `data.json` (premature fragmentation).

## 5. `config.json` grouped by origin
`data` = facts the dataset dictates (`in_dim`, `out_dim`, `dataset`, `test_size`);
`model` = what you chose (`hidden`); plus `hyperparams`, `seed`, `metric`.
**Why:** `in_dim`/`out_dim` aren't choices — the dataset dictates them — so they live
under `data`, and `data.py` *reports* them (train never hardcodes shapes).
**Rejected:** putting all three model dims under `model` (groups a fact with choices).

## 6. Deterministic split via explicit `random_state`, shared by train + evaluate
One `data.load_split(seed)` with `random_state=seed` passed *explicitly* to
`train_test_split`, imported by both scripts.
**Why:** global `set_seed` is **not** enough — under global RNG the split outcome depends
on call order, which differs between train.py and evaluate.py → different splits →
leakage. An explicit `random_state` is order-independent.
**Rejected:** relying on `set_seed` alone (silent train/test leakage — the worst kind of
bug, inflates accuracy with no crash).

## 7. Config-driven metric via a registry
`METRICS = {"accuracy", "mae", "rmse"}`; `config.json` names the metric; simple papers do
`METRICS[cfg["metric"]]`. Complex papers (MACE energy+forces, RFM generative) compute
their own metric inline but still record its name/value.
**Why:** keeps `evaluate.py` paper-agnostic (no hardcoded metric); the registry is the
menu, not dead code.

## 8. Logging substrate = console + JSONL; everything else deferred
`log_metrics` prints to console and appends to `run_dir/metrics.jsonl`. Backend is
pluggable (W&B can be added behind the same call later).
**Why:** the persisted JSONL is the source of truth; viewing is a swappable front-end
(agent plots it, or W&B later, or a Quarto page). Persisted data means deferring the
viewer loses nothing.
**Rejected:** wiring W&B now (premature; only pays off at sweep scale or for public
sharing); a custom dashboard or chatbot app (yak-shave — the conversational front-end
already exists via Claude Code + the run-paper skill).

## 9. Skills invoked by natural language, not slash commands
`run-paper` and `gdl-repo-map` activate on natural language ("train 00-template",
"how do I add a paper?"). No `/train` `/eval` slash commands.
**Why:** slash commands are a separate, redundant mechanism (`.claude/commands/`) that
over-complicates; the skill already does the job conversationally.
**Rejected:** `/train` `/eval` slash commands delegating to the skill (built then
removed — unnecessary indirection).

## 10. Skills are tool-neutral, with a Claude symlink
Canonical `SKILL.md` lives in `.agents/skills/<name>/`; `.claude/skills/<name>` is a
symlink so Claude Code discovers it.
**Why:** portability (open-standard format, not Claude-locked) without losing Claude
auto-discovery. One source of truth.

## 11. `gdl` is an installed package; pin `numpy<2`; `torch.load(weights_only=True)`
`src/gdl` is installed (hatchling, src layout) so `from gdl import ...` works everywhere.
`numpy<2` because torch 2.2.2 is built against NumPy 1.x. `weights_only=True` on load.
**Why:** clean imports for scripts/tests/notebooks/autoresearch; NumPy 2 breaks the
torch↔numpy bridge; `weights_only=True` silences the FutureWarning and is safe (payload
is always a state_dict).

---

*Append new decisions as they happen — this is a living doc, not a one-time artifact.*
