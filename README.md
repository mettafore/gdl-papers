# GDL Papers

Reproducing key papers in Geometric Deep Learning from scratch in PyTorch.

## Goal

Build working implementations of foundational GDL papers, benchmark them against published results on standard datasets (QM9, MD17, etc.), and document what we learn along the way.

## Repo Structure

```
gdl-papers/
├── README.md
├── papers/
│   └── 01-EGNN/
│       ├── CLAUDE.md      # Full context for Claude Code sessions
│       ├── notes.md       # Reading notes, observations, benchmark results
│       └── src/           # Implementation code
```

Each paper gets a numbered folder. Inside:

- **CLAUDE.md** — everything Claude Code needs to help implement the paper: key equations, architecture details, target benchmarks, implementation plan.
- **notes.md** — your reading notes, questions, experiment logs, and results.
- **src/** — the actual PyTorch implementation.

## Papers

| # | Paper | Authors | Venue | Status |
|---|-------|---------|-------|--------|
| 01 | [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844) | Satorras, Hoogeboom, Welling | ICML 2021 | In Progress |

## Setup

```bash
# Create a conda env (recommended)
conda create -n gdl python=3.10
conda activate gdl
pip install torch torch-geometric

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install torch torch-geometric
```

## Convention

- One paper per folder, numbered in reading order
- Implementations should be self-contained within each paper's `src/`
- Target: match or come within 5% of published benchmark numbers
