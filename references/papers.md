# GDL Papers Reference

## Implement (full reproduction + benchmark match)

### EGNN — Equivariant Graph Neural Networks
- **arXiv:** 2102.09844
- **Authors:** Satorras, Hoogeboom, Welling (ICML 2021)
- **Why:** Entry point to equivariant message passing. No spherical harmonics — displacement vectors + invariant scalars. QM9 benchmark.
- **Repo path:** `papers/01-EGNN/`

### MACE — Higher Order Equivariant Message Passing
- **arXiv:** 2206.07697
- **Authors:** Batatia, Kovács, Simm, Ortner, Csányi (NeurIPS 2022)
- **Why:** NequIP's successor. SOTA molecular force fields. Most practically deployed equivariant GNN.
- **Repo path:** `papers/` (not yet created)

### Riemannian Flow Matching
- **arXiv:** 2302.03660
- **Authors:** Chen & Lipman (ICLR 2024)
- **Why:** Direct preprint ancestor. Extension target: swap Riemannian metric for Finsler metric.
- **Repo path:** `papers/` (not yet created)

---

## Study (Pass 1 & 2)

### Hitchhiker's Guide to Geometric GNNs
- **arXiv:** 2312.07511
- **Authors:** Duval, Bronstein et al. (2024)
- **Why:** Survey — maps the entire equivariant GNN landscape.
- **Repo path:** `papers/02-Hitchhikers/`

### MPNN — Neural Message Passing for Quantum Chemistry
- **arXiv:** 1704.01212
- **Authors:** Gilmer et al. (ICML 2017)
- **Why:** Grandfather of all molecular GNNs. Establishes message passing framework.
- **Repo path:** `papers/03-MPNN/`

### SchNet — A Continuous-Filter CNN for Modeling Quantum Interactions
- **arXiv:** 1706.08566
- **Authors:** Schütt et al. (NeurIPS 2018)
- **Why:** First continuous radial filters from distances. Direct ancestor of EGNN's invariance trick.
- **Repo path:** `papers/05-SchNet/`

### TFN — Tensor Field Networks
- **arXiv:** 1802.08219
- **Authors:** Thomas et al. (NeurIPS 2018)
- **Why:** Introduces irreps machinery — missing rung between SchNet and SE(3)-Transformers.
- **Repo path:** `papers/06-TFN/`

### SE(3)-Transformers
- **arXiv:** 2006.10503
- **Authors:** Fuchs et al. (NeurIPS 2020)
- **Why:** Full spherical harmonics + irreps + attention. The expensive version of what EGNN simplified.
- **Repo path:** `papers/07-SE3-Transformer/`

### Equiformer
- **arXiv:** 2206.11990
- **Authors:** Liao & Smidt (ICLR 2023)
- **Why:** SOTA equivariant attention. Vanilla Transformer block with every op swapped for equivariant counterpart.
- **Repo path:** `papers/08-EquiFormer/`

### NequIP — E(3)-Equivariant Graph Neural Networks for Data-Efficient Atomistic Potentials
- **arXiv:** 2101.03164
- **Authors:** Batzner et al. (Nat. Commun. 2022)
- **Why:** First equivariant interatomic potential. Predecessor to MACE.
- **Repo path:** `papers/09-Nequip/`

### DDPM — Denoising Diffusion Probabilistic Models
- **arXiv:** 2006.11239
- **Authors:** Ho, Jain, Abbeel (NeurIPS 2020)
- **Why:** Foundation for diffusion-based generative models.

### Score-Based SDEs
- **arXiv:** 2011.13456
- **Authors:** Song et al. (ICLR 2021)
- **Why:** Unified framework for score-based generative modeling via SDEs.

### Flow Matching
- **arXiv:** 2210.02747
- **Authors:** Lipman, Chen, Ben-Hamu, Nickel (ICLR 2023)
- **Why:** Simulation-free training of continuous normalizing flows. Precursor to Riemannian Flow Matching.

---

## Breadth (read once for awareness)

### ManifoldFormer
- **arXiv:** 2511.16828
- **Authors:** Fu et al. (2025)
- **Why:** Riemannian VAE + geodesic attention + neural ODEs on EEG. Consciousness breadth slot.
- **Repo path:** `papers/10-ManifoldFormer/`
