# EGNN — E(n) Equivariant Graph Neural Networks

**Paper:** Satorras, Hoogeboom, Welling. "E(n) Equivariant Graph Neural Networks." ICML 2021.
**arXiv:** https://arxiv.org/abs/2102.09844
**Reference code:** https://github.com/vgsatorras/egnn

---

## What This Paper Does

EGNN is a graph neural network that is **equivariant to E(n)** — the group of rotations, reflections, and translations in n-dimensional Euclidean space. Unlike prior work (TFN, SE(3)-Transformers), it achieves this **without spherical harmonics or expensive irreducible representations**. The key insight: operate on relative positions and distances, update coordinates using equivariant vector operations.

---

## Key Equations

The EGNN layer updates node embeddings `h`, coordinates `x`, and (optionally) edge attributes `a` as follows:

### Message computation
```
m_ij = φ_e(h_i^l, h_j^l, ||x_i^l - x_j^l||^2, a_ij)
```

### Coordinate update (the equivariant part)
```
x_i^(l+1) = x_i^l + C * Σ_j (x_i^l - x_j^l) · φ_x(m_ij)
```

where `φ_x` outputs a **scalar** weight per edge. Because `(x_i - x_j)` is equivariant to translations and rotations, and it's scaled by a scalar, the whole update is equivariant. `C = 1/(M-1)` normalizes by number of neighbors.

### Node embedding update
```
m_i = Σ_j m_ij
h_i^(l+1) = φ_h(h_i^l, m_i)
```

### All φ functions are MLPs:
- `φ_e`: edge MLP (input: concat of h_i, h_j, d_ij^2, a_ij → hidden → output)
- `φ_x`: coordinate MLP (input: m_ij → hidden → 1 scalar, often with tanh clamp)
- `φ_h`: node MLP (input: concat of h_i, m_i → hidden → output)

---

## What Equivariance Means Here

- **Invariant** outputs (like energy predictions) don't change when you rotate/translate the molecule
- **Equivariant** outputs (like coordinate updates) transform *with* the rotation/translation
- EGNN coordinates are **E(n) equivariant**: if you rotate all input coords by R, all output coords rotate by R too
- EGNN node features are **E(n) invariant**: they only depend on distances, not absolute positions
- This is achieved without any group representation theory — just by construction

**Test:** After training, rotate all input coordinates by a random rotation matrix. The predicted property (energy/force) should be identical (up to numerical precision).

---

## Target: QM9 Dataset

QM9 contains ~130K small molecules with up to 9 heavy atoms (C, N, O, F) plus hydrogens. Each molecule has 3D coordinates and 12 quantum chemical properties.

### Key benchmark targets (from Table 1 in paper)

| Property | Unit | EGNN (paper) | SchNet | DimeNet++ |
|----------|------|-------------|--------|-----------|
| mu (dipole moment) | D | 0.029 | 0.033 | 0.030 |
| alpha (polarizability) | a₀³ | 0.071 | 0.235 | 0.044 |
| HOMO | meV | 29.86 | 41.0 | 24.6 |
| LUMO | meV | 25.37 | 34.0 | 19.5 |
| gap (HOMO-LUMO) | meV | 48.39 | 63.0 | 32.6 |
| U0 | meV | 11.0 | 14.0 | 6.32 |
| Cv (heat capacity) | cal/mol·K | 0.031 | 0.033 | 0.023 |

**Our target:** Match EGNN paper numbers within 5-10%.

### QM9 data details
- Train/val/test split: 100K / 18K / 13K (standard split)
- Node features: 15-dim charge-power = one-hot(H,C,N,O,F) ⊗ (z/9)^{0,1,2} (matches reference; replaces PyG's default 11-dim x, dropping aromatic/hybridization/num_Hs annotations)
- Edge construction: fully connected within cutoff (typically ~5Å) or k-nearest neighbors
- Coordinates: 3D positions in Angstroms

---

## Architecture Details

### Hyperparameters (from paper/reference code)
- Hidden dim: 128
- Number of EGNN layers: 7
- Coordinate update: yes (with `tanh` clamp on φ_x output)
- Edge features: squared distances
- Normalization: `coords_agg = "mean"` (divide coordinate update by number of neighbors)
- Attention: optional (sigmoid gate on messages) — improves some targets
- Residual connections on `h`
- SiLU (Swish) activation

### Training
- Optimizer: Adam, lr=5e-4 with ReduceLROnPlateau (patience 10, factor 0.5, min_lr 1e-7)
- Batch size: 96
- Epochs: ~1000 (with early stopping, patience ~50)
- Loss: L1 (MAE) on normalized targets
- Target normalization: subtract mean, divide by std (computed on training set)

---

## Implementation Plan

### Stage 1: Data Pipeline
- [x] Load QM9 via `torch_geometric.datasets.QM9`
- [x] Implement standard train/val/test split (100K/18K/13K, seeded)
- [x] Fully connected edges (QM9 molecules are small; PyG's default `edge_index`)
- [x] Normalize targets (NormStats: mean/MAD, fit on train only)
- [x] DataLoader with batching

### Stage 2: EGNN Layer
- [x] Implement single EGNN layer as `nn.Module`
  - Edge MLP: `φ_e(h_i, h_j, d²_ij) → m_ij`
  - Coord MLP: `φ_x(m_ij) → scalar` (with tanh)
  - Node MLP: `φ_h(h_i, Σ m_ij) → h_i'`
- [ ] Coordinate update with mean aggregation (skipped for QM9 — positions static)
- [x] Residual connection on node features
- [x] Unit test: verify equivariance (rotate inputs, check outputs transform correctly)

### Stage 3: Full Model
- [x] Stack 7 EGNN layers
- [x] Input embedding: one-hot atomic number → hidden dim
- [x] Output head: node-level MLP → graph-level sum pooling → prediction
- [x] Batch-aware pooling (PyG `batch` index; multi-molecule batches, not just single-graph)
- [x] Attention mechanism (sigmoid gate on messages, eq. 8 — reference Table-1 default, `attention=True`)

### Stage 4: Training Loop
- [x] Training script with Adam + ReduceLROnPlateau
- [x] Logging (this repo's own JSONL `log_metrics`, not wandb/tensorboard)
- [x] Checkpointing best model (on val-loss improvement)
- [ ] Evaluation on test set (belongs in `evaluate.py`, not yet built)

### Stage 5: Benchmarking
- [ ] Run on mu, alpha, HOMO, LUMO, gap, U0, Cv
- [ ] Compare MAE against paper numbers
- [ ] Ablation: with/without coordinate updates, with/without attention

---

## Common Pitfalls

1. **Forgetting to detach coordinates** from the computation graph if you don't want to backprop through coordinate updates (for property prediction, you typically DO want gradients through coords)
2. **Edge index construction**: QM9 molecules are small, so fully connected is fine. For larger molecules, use a cutoff.
3. **Target normalization**: Must normalize targets to zero mean, unit variance. Without this, training is unstable.
4. **Numerical equivariance test**: Use `torch.allclose` with reasonable `atol` (~1e-5). Exact equality won't hold due to floating point.
5. **Batch handling**: `torch_geometric` batches graphs by concatenating — make sure sum pooling respects batch boundaries (use `scatter` or `global_add_pool`).

---

## File Layout

```
papers/01-EGNN/
├── CLAUDE.md          # This file
├── notes.md           # Reading notes and experiment log
└── src/
    ├── model.py       # EGNN layer + full model
    ├── data.py        # QM9 data loading and preprocessing
    ├── train.py       # Training loop
    ├── evaluate.py    # Evaluation and benchmarking
    └── utils.py       # Equivariance tests, helpers
```
