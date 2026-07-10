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

**Still TODO:**
- `EGCL.forward(h, x, edge_index, edge_attr)` — ties coord2radial → edge_model → node_model together
- `EGNN.__init__` — embedding in/out linear layers + stack of 7 EGCL (use `nn.ModuleList`, not plain Python list, so params register)
- `EGNN.forward`

**Next session:** write `unsorted_segment_sum`, then `node_model`, then `EGCL.forward`.
