# Progress

Canonical status for every paper in this repo. Update this file when you start,
finish, or pause work on a paper — not the root README status table (that should
only link here).

Per-paper experiment logs and reading notes stay in `papers/NN-name/notes.md`.
Implementation context and stage checklists stay in `papers/NN-name/CLAUDE.md`
(or the paper README once the scaffold lands).

**Last updated:** 2026-07-14 (EGNN model + train.py done, TDD'd, first CLI run verified)

---

## Summary

| # | Paper | Tier | Status | Last touched |
|---|-------|------|--------|--------------|
| 00 | Template (Iris MLP) | scaffold | ✅ done | — |
| 01 | EGNN | implement | 🟡 in progress | 2026-07-14 |
| 02 | Hitchhiker's Guide | study | 📄 PDF only | — |
| 03 | MPNN | study | 📄 PDF only | — |
| 05 | SchNet | study | 📄 PDF + sources | — |
| 06 | TFN | study | 📄 PDF + sources | — |
| 07 | SE(3)-Transformer | study | 📄 PDF + sources | — |
| 08 | EquiFormer | study | 📄 PDF + sources | — |
| 09 | NequIP | study | 📄 PDF + sources | — |
| 10 | ManifoldFormer | breadth | 📄 PDF only | — |

Tier definitions → [`references/papers.md`](../references/papers.md).

**Legend:** ✅ done · 🟡 in progress · ⬜ not started · 📄 materials only

---

## Active: 01-EGNN

**Goal:** QM9 property prediction; match paper MAE within ~5–10%.

**Overall:** data pipeline, model, and training loop all done and TDD'd.
No `evaluate.py` yet — that's the only thing blocking a real benchmark run.

### Data (`papers/01-EGNN/src/data.py`)

| Item | Status |
|------|--------|
| `NormStats` (mean / MAD) | ✅ |
| QM9 load + deterministic split (100K / 18K / 13K) | ✅ |
| DataLoaders | ✅ |
| Edge construction | ✅ (QM9 default fully-connected `edge_index`, no cutoff needed at this size) |
| `dims["target_col"]` (which QM9 y-column this run trains on) | ✅ |

### Model (`papers/01-EGNN/src/model.py`)

| Item | Status |
|------|--------|
| `EGCL` (edge/coord/node MLPs, forward) | ✅ |
| `unsorted_segment_sum` helper | ✅ |
| `EGNN` (embed → 7× EGCL → node_dec → pool → graph_dec) | ✅ |
| Batch-aware pooling (`batch` arg, multi-molecule DataLoader batches) | ✅ (bug caught + fixed after initial "done" only handled single-graph) |

### Train / eval / benchmark

| Item | Status |
|------|--------|
| `train.py` (`train()` + `main()` CLI) | ✅ verified via real CLI run |
| `evaluate.py` | ⬜ not started |
| Tests (22 passing: model + data + train helpers) | ✅ |
| First training run (smoke test, 1 epoch) | ✅ |
| Full training run (1000 epochs, real target) | ⬜ |
| Benchmark vs paper Table 1 | ⬜ (blocked on `evaluate.py`) |

### Next step

Build `evaluate.py`: load a checkpoint by `run_id`, run the held-out test
split, report denormalized MAE per target, compare against paper's Table 1.
Then a real (non-smoke) training run on `gap` to sanity-check the numbers
before running the full benchmark sweep across all 7 targets.

---

## Done: 00-template

Runnable Iris MLP scaffold — proves train → evaluate → run folder loop.
See [`papers/00-template/README.md`](../papers/00-template/README.md).

---

## Not started (implement tier)

| Paper | Repo path | Notes |
|-------|-----------|-------|
| MACE | — | folder not created |
| Riemannian Flow Matching | — | folder not created |

---

## Study / breadth (materials collected)

PDFs and LaTeX sources live under `papers/NN-name/`. No reproduction code yet.
Reading order suggestion: SchNet → TFN → SE(3)-Transformer → EquiFormer → NequIP,
after EGNN lands.
