# Reference Index

Curated docs for working in this repo. Prefer the **context7** MCP for live API
lookups; the links below are the canonical entry points.

## External docs (static bookmarks)

### Framework
- PyTorch — https://pytorch.org/docs/stable/index.html
  (`nn.Module`, `optim`, `save`/`load`, tensors)

### Data
- scikit-learn datasets — https://scikit-learn.org/stable/datasets/toy_dataset.html
  (`load_iris`, `train_test_split`)

### Tooling
- uv — https://docs.astral.sh/uv/  (env, deps, `uv run`, packaging)
- pytest — https://docs.pytest.org/  (test discovery, `tmp_path`)

### Domain (GDL)
- Geometric Deep Learning proto-book (Bronstein et al.) — https://arxiv.org/abs/2104.13478
- Paper list + tiers → [`references/papers.md`](../references/papers.md)

### Reference implementations (original author code)
- EGNN (Satorras et al.) — https://github.com/vgsatorras/egnn

## Living context — keep current

In-repo docs that must not rot. Update each when its trigger fires:

| Doc | Update when |
|-----|-------------|
| `docs/progress.md` | **always** — start, finish, or pause work on any paper |
| `docs/spec.md` | starting or changing a feature |
| `AGENTS.md` | commands or key paths change |
| `references/papers.md` | a paper's tier or repo path changes |
| `papers/NN/README.md` | results or run instructions change |
| `papers/NN/notes.md` | reading notes or experiment log entries |
| `.agents/skills/*` | a convention changes |
| `README.md` (root) | quickstart or layout changes (status → link to progress.md) |
