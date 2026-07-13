# Progress

Canonical status for every paper in this repo. Update this file when you start,
finish, or pause work on a paper — not the root README status table (that should
only link here).

Per-paper experiment logs and reading notes stay in `papers/NN-name/notes.md`.
Implementation context and stage checklists stay in `papers/NN-name/CLAUDE.md`
(or the paper README once the scaffold lands).

**Last updated:** 2026-07-13 (unsorted_segment_sum done, tested vs reference)

---

## Summary

| # | Paper | Tier | Status | Last touched |
|---|-------|------|--------|--------------|
| 00 | Template (Iris MLP) | scaffold | ✅ done | — |
| 01 | EGNN | implement | 🟡 in progress | 2026-07-10 |
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

**Overall:** data pipeline mostly done; model partially done; no train/eval scaffold yet.

### Data (`papers/01-EGNN/src/data.py`)

| Item | Status |
|------|--------|
| `NormStats` (mean / MAD) | ✅ |
| QM9 load + deterministic split (100K / 18K / 13K) | ✅ |
| DataLoaders | ✅ |
| Edge construction | ⬜ (decide: here vs model) |

### Model (`papers/01-EGNN/src/model.py`)

| Item | Status |
|------|--------|
| `EGCL.__init__` (edge/coord/node MLPs) | ✅ |
| `EGCL._coord2radial` | ✅ |
| `EGCL.edge_model` | ✅ |
| `unsorted_segment_sum` helper | ✅ |
| `EGCL.node_model` | ✅ |
| `EGCL.forward` | ⬜ |
| `EGNN` (embed → 7× EGCL → head) | ⬜ |

### Train / eval / benchmark

| Item | Status |
|------|--------|
| `train.py` / `evaluate.py` (00-template shape) | ⬜ |
| Tests | ⬜ |
| First training run | ⬜ |
| Benchmark vs paper Table 1 | ⬜ |

### Next step

Write `unsorted_segment_sum` → `node_model` → `EGCL.forward`. Detail in
[`papers/01-EGNN/notes.md`](../papers/01-EGNN/notes.md) (model.py section, 2026-07-10).

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
