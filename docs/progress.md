# Progress

Canonical status for every paper in this repo. Update this file when you start,
finish, or pause work on a paper — not the root README status table (that should
only link here).

Per-paper experiment logs and reading notes stay in `papers/NN-name/notes.md`.
Implementation context and stage checklists stay in `papers/NN-name/CLAUDE.md`
(or the paper README once the scaffold lands).

**Last updated:** 2026-07-15 (added attention gate + charge-power node features to match paper's Table-1 config; overnight recipe-fix run died at epoch 11 — not detached; benchmark still pending)

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
~epoch 200 on (ReduceLROnPlateau decayed LR to a near-zero floor). This run
also predates both the recipe fix AND the two config changes below.

**Overnight recipe-fix run (2026-07-14, run `20260714-190424`): DIED at
epoch 11** of 1000 — not a training failure, the `modal run` was not
detached and the process was killed (laptop sleep). Trajectory over those 11
epochs was healthy and descending (val_loss 0.294 → 0.194 normalized), no
plateau. No benchmark number from it. Re-launch must use `modal run
--detach`.

**Config now matches paper Table 1 (2026-07-15):** implemented the two
remaining deviations that a paper-faithful reproduction needs — see below.

**Two launch/config bugs found & fixed (2026-07-15):**
- `modal_train.py` hardcoded `lr=5e-4` in the local_entrypoint, overriding
  `train()`'s recipe default of 1e-3. So **every earlier Modal run trained
  at 5e-4, not the reference 1e-3** — the recipe fix never actually reached
  the GPU. Fixed to 1e-3.
- `.remote()` blocks the client 12h; any SIGTERM to it cancels the remote
  call. Killed three launch attempts (laptop sleep, manual kill, background
  reap). Switched to `.spawn()` + `modal run --detach` — dispatch
  server-side, client exits immediately, job runs independently.

**Full paper-faithful run LIVE (2026-07-15 ~14:49, app `ap-DQXDO61eUW5nqda...`):**
`gap`, T4, attention + charge-power + lr 1e-3 + CosineAnnealingLR, spawned
detached, confirmed training (val_loss descending from ~0.33). First run
with the correct LR *and* the full config. This is the benchmark that
decides Table-1 match.

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
| Attention (sigmoid message gate, eq. 8 — reference Table-1 default) | ✅ (2026-07-15, TDD; `EGNN(attention=True)` threads to every EGCL) |
| Charge-power node features (15-dim: one-hot ⊗ (z/9)^{0,1,2}) | ✅ (2026-07-15, TDD; replaces PyG's default 11-dim `x`, matches reference) |

### Train / eval / benchmark

| Item | Status |
|------|--------|
| `train.py` (`train()` + `main()` CLI) | ✅ verified via real CLI run |
| `evaluate.py` | ✅ (loads checkpoint, denormalizes, reports MAE incl. meV conversion for paper-comparable units) |
| Modal GPU wrapper (`modal_train.py`) | ✅ (had to fix: train.py never used the GPU at all initially, ~5x speedup once fixed; +num_workers ~1.35x on top) |
| Tests (33 passing: model + data + train + evaluate, incl. attention + charge-power) | ✅ |
| First training run (smoke test, 1 epoch) | ✅ |
| Real training run (gap, GPU) | ✅ 261/1000 epochs, plateaued — see benchmark result above |
| Benchmark vs paper Table 1 | 🟡 first data point in, not within target yet (93.01 vs 48.39 meV) |

### Next step

Launch the full paper-faithful config on Modal, **detached** (`modal run
--detach modal_train.py`), T4, batch 96 — recipe + attention + charge-power
all in. This is the run that decides whether we match Table 1's 48.39 meV on
`gap`. If it lands in range, sweep the other 6 targets (mu, alpha, homo,
lumo, U0, Cv). If still off, remaining suspects are narrow (coordinate
updates are intentionally skipped per QM9 convention, matching reference).

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
