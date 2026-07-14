# EGNN — Reading Notes & Experiment Log

**Paper:** E(n) Equivariant Graph Neural Networks (Satorras et al., ICML 2021)

---

## Key Takeaways

- [ ] Read paper end-to-end
- [ ] Understand why scalar-weighted coordinate updates preserve equivariance
- [ ] Compare to TFN/SE(3)-Transformer approach (spherical harmonics vs. this)
- [ ] Understand the connection to normalizing flows (Section 5 in paper)

## Questions

-

## Experiment Log

| Date | Experiment | Target | Result (MAE) | Notes |
|------|-----------|--------|--------------|-------|
| | | | | |

## Observations

-

# EGNN — What, How, Next

## What problem did they solve?
GNNs that operate on 3D data (molecules, particles) had no way to respect physical symmetries — rotate a molecule, get a different prediction. Previous equivariant methods were computationally expensive and locked to 3D.

## How did they solve it?
Two simple changes to standard message passing: feed squared distance $\|x_i - x_j\|^2$ into the edge operation (makes messages invariant), and update coordinates using a weighted sum of relative differences $(x_i - x_j)$ scaled by invariant scalars (makes coordinates equivariant). No spherical harmonics. Works in any dimension.

## Where to go next?
The distance metric is Euclidean. Many real systems are anisotropic — the geometry depends on direction, not just distance. Replacing $\|x_i - x_j\|^2$ with a Finsler distance is an open research direction with almost no existing work. This is the natural next paper after reproducing EGNN.

## model.py progress (as of 2026-07-10)

Working through `src/model.py` in learn-mode (`?`-triggered, see `.agents/skills/learn-mode/`).

**Done:**
- `EGCL.__init__` — edge_mlp, coord_mlp, node_mlp defined (input_nf == hidden_nf for every layer, confirmed against original repo `vgsatorras/egnn`, embedding happens once in EGNN before any EGCL layer runs)
- `EGCL._coord2radial(self, edge_index, coord)` — done, `keepdim=True` on the sum (radial must be `[num_edges, 1]` to match edge_mlp's concat)
- `EGCL.edge_model(self, h_row, h_col, radial, edge_attr)` — done, concat + edge_mlp

**In progress:**
- `EGCL.node_model(h, edge_index, edge_feat)` — logic worked out, not yet written to file:
  ```
  row, col = edge_index
  agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0))  # row = target node i
  combined = torch.cat([h, agg], dim=1)
  out = self.node_mlp(combined)
  return h + out   # residual: MLP predicts a correction, not the new value outright
  ```
- `unsorted_segment_sum` helper — decided to hand-roll (not PyG scatter) for the learning value; look at `torch.Tensor.scatter_add_` signature, needs a zero-init result tensor shaped `[num_segments, feat_dim]`.

**Done:** `unsorted_segment_sum`, `EGCL` (edge/node model + forward), full `EGNN`
(`__init__`: emb + 7×EGCL ModuleList + node_dec/graph_dec; `forward`: embed →
layers → node_dec → batch-aware pool → graph_dec, takes a `batch` arg so a
multi-molecule DataLoader batch pools per-graph, not into one blended scalar).
`data.py` (`load_split`: QM9 load, seeded split, NormStats fit on train,
`dims["target_col"]`). `train.py` (`train()` + `main()`: full epoch loop —
train pass, val pass denormalized to original units, scheduler.step on val
loss, checkpoint on improvement). 22/22 tests pass. Verified with a real CLI
run (`uv run python papers/01-EGNN/src/train.py --epochs 1 ...`) end to end.

**Bugs caught + fixed via TDD (not just manual check):**
- `EGNN.forward` originally summed every node across a whole call into one
  scalar — broke on real multi-molecule batches (see reference repo's
  `qm9/models.py` for the pooling convention this was compared against).
- `train.py`'s loss compared model output `(N,1)` against QM9's full 19-column
  `y` `(N,19)` — L1Loss silently broadcasts instead of erroring, so it would've
  trained on the wrong signal with no crash. Fixed via `dims["target_col"]` +
  `target_for`/`raw_target_for` helpers, tests written before the fix.

**Next session (Stage 5 — evaluate.py):** build `evaluate.py` — load a
checkpoint by `run_id`, run test split, report denormalized MAE per target,
compare against paper's Table 1 numbers (see CLAUDE.md benchmark targets).
