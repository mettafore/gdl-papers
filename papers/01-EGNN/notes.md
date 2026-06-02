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
