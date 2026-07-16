# Progress

Canonical status for every paper in this repo. Update this file when you start,
finish, or pause work on a paper — not the root README status table (that should
only link here).

Per-paper experiment logs and reading notes stay in `papers/NN-name/notes.md`.
Implementation context and stage checklists stay in `papers/NN-name/CLAUDE.md`
(or the paper README once the scaffold lands).

**Last updated:** 2026-07-16 (found + fixed the real bug: QM9 edges were molecular bonds, not fully-connected — test MAE dropped 93→63.77 meV; resume-at-lower-LR run in progress, val already at ~49 meV, closing in on paper's 48.39)

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

**First full-config run diverged (2026-07-15, `ap-DQXDO61eUW5nqda...`):**
descended cleanly to val_loss 0.11 (~110 meV) by epoch 80, then a single
epoch (81→82) exploded to 1.08 and froze — early stop fired at epoch 130.
Diffed our EGCL/EGNN against `vgsatorras/egnn` line-by-line (optimizer,
scheduler, attention gate, charge-power formula, node_attr flag — all
matched). Reference also has no gradient clipping, so that wasn't it either.

**Root cause found (2026-07-15): QM9 edges were molecular BONDS, not
fully-connected.** PyG's default `edge_index` gives methane 8 edges (its
bonds); reference EGNN (`get_adj_matrix`) uses the complete graph — every
atom pair, N*(N-1) edges. EGNN's entire mechanism is pairwise distances, so
training on bonds-only made the model blind to most of the geometry. This
was the real driver of the ~2x MAE gap (and very likely the divergence
too — confirmed fixed, see below). Added `fully_connected_edge_index()` in
`data.py`, wired into the QM9 transform. See git history for the fix
commit; `docs/reference-index.md` / this file's edge-construction row
updated to stop claiming "fully-connected" when it wasn't.

**Rerun with fully-connected edges (2026-07-15 evening, run
`20260715-191930`): no divergence, best val 0.0640 (64.0 meV) at epoch
124**, early-stopped at 174 (patience 50, LR still ~93% of max — a
high-LR plateau, not real convergence). **Evaluated on the real held-out
test split (not val): 63.77 meV** — confirms val_loss was a reliable
proxy here. 63.77 / 48.39 = 1.32x off, inside striking distance for the
first time.

**Resume-at-lower-LR (2026-07-16, run off `ap-wdyESp2VrgLk5hJrFYc3WD`):**
loaded the epoch-124 checkpoint (weights only, Adam state not preserved)
and continued at lr=1e-4 (down from 1e-3), 300-epoch cosine anneal to 0,
via new `train(resume_from=...)` + per-epoch LR logging in metrics.jsonl.
**One epoch: val_loss 0.0640 → 0.0535 (53.5 meV).** Confirms the plateau
was a high-LR noise floor, not a real local minimum — dropping the LR
immediately unstuck it. By epoch 15: **val ~49.4 meV**, still descending,
285 epochs of anneal left. This is currently the best-tracking run; get
its test-set MAE via `evaluate.py` once it settles, don't trust val alone
(see the 63.77-vs-64.0 check above for why that's usually fine here, but
verify per the current run before reporting a final number).

### Data (`papers/01-EGNN/src/data.py`)

| Item | Status |
|------|--------|
| `NormStats` (mean / MAD) | ✅ |
| QM9 load + deterministic split (100K / 18K / 13K) | ✅ |
| DataLoaders | ✅ |
| Edge construction | ✅ FIXED 2026-07-15: build fully-connected `edge_index` per molecule (all i≠j). PyG QM9's default `edge_index` is molecular BONDS (sparse), NOT fully connected — earlier claim here was wrong. Reference EGNN uses complete graphs; bonds-only made the model blind to most pairwise geometry (~2x MAE gap). |
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

Let the resume-at-lower-LR run (off `20260715-191930`, currently ~49 meV
val at epoch 15) finish or plateau, then run `evaluate.py` on its checkpoint
for the real test-set MAE (don't report val_loss as final — see above).
If test MAE lands at/under ~50.8-53.2 meV (5-10% of paper's 48.39), `gap`
is matched — sweep the other 6 targets next (mu, alpha, homo, lumo, U0,
Cv), each its own from-scratch faithful run (fully-connected edges +
attention + charge-power + lr 1e-3 cosine over 1000 epochs, no early
stop — resume-at-lower-LR was a diagnostic/shortcut for `gap` specifically,
not yet established as the standard recipe for every target).

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
