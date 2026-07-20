# Papers

One folder per paper. Reading copies (PDFs) and arXiv source trees are **not in
git** — run `./papers/fetch-papers.sh` after cloning to download them.

| Folder | Paper | arXiv | Tier |
|--------|-------|-------|------|
| `00-template/` | Template — Iris MLP scaffold | — | scaffold |
| `01-EGNN/` | E(n) Equivariant Graph Neural Networks (Satorras et al., 2021) | [2102.09844](https://arxiv.org/abs/2102.09844) | implement |
| `02-Hitchhikers/` | Hitchhiker's Guide to Geometric GNNs (Duval et al., 2024) | [2312.07511](https://arxiv.org/abs/2312.07511) | study |
| `03-MPNN/` | Neural Message Passing for Quantum Chemistry (Gilmer et al., 2017) | [1704.01212](https://arxiv.org/abs/1704.01212) | study |
| `05-SchNet/` | SchNet — Continuous-Filter CNN (Schütt et al., 2017) | [1706.08566](https://arxiv.org/abs/1706.08566) | study |
| `06-TFN/` | Tensor Field Networks (Thomas et al., 2018) | [1802.08219](https://arxiv.org/abs/1802.08219) | study |
| `07-SE3-Transformer/` | SE(3)-Transformer (Fuchs et al., 2020) | [2006.10503](https://arxiv.org/abs/2006.10503) | study |
| `08-EquiFormer/` | EquiFormer (Liao & Smidt, 2022) | [2206.11990](https://arxiv.org/abs/2206.11990) | study |
| `09-Nequip/` | NequIP (Batzner et al., 2021) | [2101.03164](https://arxiv.org/abs/2101.03164) | study |
| `10-ManifoldFormer/` | ManifoldFormer | [2511.16828](https://arxiv.org/abs/2511.16828) | breadth |

- Full annotations (authors, why each paper, planned papers not yet in a
  folder) → [`references/papers.md`](../references/papers.md)
- Per-paper status → [`docs/progress.md`](../docs/progress.md)
- Folder layout contract → [`00-template/README.md`](00-template/README.md)

When adding a paper: create the folder from `00-template/`, add a row here, and
add its arXiv ID to `fetch-papers.sh`'s mapping.
