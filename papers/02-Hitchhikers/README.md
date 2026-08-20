# A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems

Gold-standard pedagogical reference for the geometric foundations used throughout
this repository.

- **Paper:** *A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems*
- **Authors:** Alexandre Duval, Simon V. Mathis, Chaitanya K. Joshi, Victor
  Schmidt, Santiago Miret, Fragkiskos D. Malliaros, Taco Cohen, Pietro Liò,
  Yoshua Bengio, and Michael Bronstein
- **arXiv:** [2312.07511](https://arxiv.org/abs/2312.07511), v2
- **Local copy:** `Hitchhikers-Guide-Geometric-GNNs.pdf`
- **SHA-256:** `d0fde41d9310b91e33d280a45fb151a265e38e464b266b47ff8f048ea0b5ebb4`

## Why this is kept here

This is a reference-only folder, not an implementation scaffold. The paper is
especially useful because it introduces equivariant features from familiar
objects before introducing representation-theory language.

For the MACE learning notebook, the key route is:

1. Section 5.1 — Cartesian tensors and their rotation laws.
2. Section 5.2 — decomposition of a rank-2 tensor into irreducible pieces.
3. Section 5.3 — spherical tensors, angular orders, and equivariant operations.

The central concrete bridge is the nine-component rank-2 decomposition

```text
9 = 1 + 3 + 5
```

corresponding to trace, antisymmetric, and symmetric-traceless pieces. Only
after that example do we need the names `l = 0`, `l = 1`, and `l = 2`.

## Related learning notes

- Obsidian reading plan:
  `/Users/luvsuneja/Documents/wiki/GeometricDeepLearning/Papers/Hitchhikers_what_how_next.md`
- Obsidian foundations note:
  `/Users/luvsuneja/Documents/wiki/GeometricDeepLearning/Mind Map/Foundations/Tensors and Representation Theory.md`
- Claude export containing the conversation **Hitchhiker doubts**:
  `/Users/luvsuneja/Downloads/data-be266601-4df9-4866-a6cf-12b29fff00eb-1785934127-f8bb9e7a-batch-0000/conversations.json`

The raw chat is provenance, not canonical exposition. Transfer only insights
that survive verification against the paper or a primary mathematical source.
