# Progress

Canonical status for every paper in this repo. Update this file when you start,
finish, or pause work on a paper — not the root README status table (that should
only link here).

Per-paper experiment logs and reading notes stay in `papers/NN-name/notes.md`.
Implementation context and stage checklists stay in `papers/NN-name/CLAUDE.md`
(or the paper README once the scaffold lands).

**Last updated:** 2026-07-14 (evaluate.py done; first real GPU benchmark run — gap MAE 93.01 meV vs paper's 48.39 meV, ~1.9x off target)

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

**Overall:** full pipeline done (data, model, train, evaluate), TDD'd, and
run end-to-end on real Modal GPU hardware. First real benchmark result is
in — not yet within target, see below.

**First real benchmark (2026-07-14):** `gap`, run `20260714-151153`, 261
epochs (of 1000 — plateaued, no early stopping implemented so it just kept
running until manually stopped). **MAE 93.01 meV vs paper's 48.39 meV — ~1.9x
off**, outside the ~5-10% target. Val loss froze essentially flat from
~epoch 200 on (ReduceLROnPlateau likely decayed LR to a near-zero floor).
Needs investigation: possibly needs the full 1000 epochs / different LR
schedule tuning, possibly a real gap vs the reference implementation
(attention mechanism not implemented, coordinate updates skipped per QM9
convention — worth double-checking against vgsatorras/egnn's qm9 config
for anything else missed).

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
| `evaluate.py` | ✅ (loads checkpoint, denormalizes, reports MAE incl. meV conversion for paper-comparable units) |
| Modal GPU wrapper (`modal_train.py`) | ✅ (had to fix: train.py never used the GPU at all initially, ~5x speedup once fixed; +num_workers ~1.35x on top) |
| Tests (23 passing: model + data + train + evaluate) | ✅ |
| First training run (smoke test, 1 epoch) | ✅ |
| Real training run (gap, GPU) | ✅ 261/1000 epochs, plateaued — see benchmark result above |
| Benchmark vs paper Table 1 | 🟡 first data point in, not within target yet (93.01 vs 48.39 meV) |

### Next step

Investigate the MAE gap: early stopping was never implemented (skipped as
"optional" in the original scaffold) — the run coasted at a frozen val loss
for 60+ epochs before being manually stopped instead of the scheduler/early
stop catching it. Options to try: implement early stopping properly, tune
LR schedule, diff remaining hyperparams against vgsatorras/egnn's qm9
config, or just let a run go the full 1000 epochs to rule out
under-training. Then re-run the benchmark and, if `gap` lands in range,
sweep the other 6 targets (mu, alpha, homo, lumo, U0, Cv).

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
