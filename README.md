# gdl-papers

PyTorch reproductions of geometric deep learning papers. Each paper is implemented,
trained, evaluated against its published benchmark, and written up. One
self-contained paper per folder, sharing a small train/evaluate scaffold.

## Results

Each reproduction is compared with the number the paper reports. Lower is better.

| Paper | Task | Metric | Paper | This repo |
|---|---|---|---:|---:|
| [EGNN](papers/01-EGNN/) | QM9 HOMO–LUMO gap | MAE (meV) | 48 | **48.50** |
| [Riemannian Flow Matching](papers/14-RFM/) | Fire locations on S² | Test NLL (uniform-sphere baseline: +2.531) | — | **−0.700** |
| [MACE](papers/15-MACE/) | 3BPA, 300 K | Energy RMSE (meV) | 3.0 | **3.99** |
| | | Force RMSE (meV/Å) | 8.8 | **11.56** |

**MACE, all 3BPA test sets** (epoch-360 checkpoint, single seed; paper values are three-seed means):

| Test set | Energy RMSE (meV) | Paper | Force RMSE (meV/Å) | Paper |
|---|---:|---:|---:|---:|
| 300 K | 3.99 | 3.0 | 11.56 | 8.8 |
| 600 K | 13.28 | 9.7 | 28.96 | 21.8 |
| 1200 K | 44.51 | 29.8 | 96.01 | 62.0 |
| Dihedral slices | 17.57 | 7.8 | 21.15 | 16.5 |

MACE is still in progress: more seeds and an energy-calibration check on the dihedral set are next.

## Quickstart

```bash
uv sync                                               # install deps + the gdl package

# train with defaults
uv run python papers/00-template/src/train.py             # prints a run_id

# train with hyperparameter overrides
uv run python papers/00-template/src/train.py --lr 1e-3 --epochs 50 --hidden 32

# evaluate a specific run
uv run python papers/00-template/src/evaluate.py --run <run_id>

# tests
uv run pytest papers/00-template/
```

## Running via skills (conversational)

In Claude Code, drive any paper in natural language — the `run-paper` skill maps it
to the commands above:

- "train 00-template" — train with defaults
- "train 00-template with lr 1e-3, 50 epochs" — overrides forwarded to CLI args
- "evaluate 00-template run <run_id>" — evaluate a run

Ask `gdl-repo-map` things like "where does shared code go?" or "how do I add a paper?"

## Commit checks

Enable the versioned Git hook once per clone:

```bash
git config core.hooksPath .githooks
```

It blocks commits that add a file over 1 MB or contain a secret, and commits
where Ruff formatting and linting, per-paper type checks, or the lightweight test
suites fail. Dataset-backed tests remain opt-in.

The secret scan needs [gitleaks](https://github.com/gitleaks/gitleaks) on your
`PATH` (`brew install gitleaks`); without it the hook warns and skips that step.
CI runs the same checks, and its gitleaks scan covers the full history.

## License

Original repository code and documentation use the [MIT license](LICENSE).
Third-party source, research papers, and datasets retain their own terms; see
[Third-party notices](THIRD_PARTY_NOTICES.md).

## Automated PR review

- Claude reviews new or updated pull requests. Comment-triggered reviews require
  `@claude` on a pull request; ordinary issue comments do not start a review.
- Configure `CLAUDE_CODE_OAUTH_TOKEN` in GitHub repository secrets using the
  token produced by `claude setup-token`.
- The workflow grants the CLI access to PR viewing, diffs, and comments. It
  reports common authentication, usage-limit, or model-access failures with
  fixed diagnostic messages. Raw provider responses are not printed.
- Failed reviews remain failed. Local diagnostic tests do not prove that the
  configured Claude account can complete a live review. Setup details:
  [official action documentation](https://github.com/anthropics/claude-code-action/blob/main/docs/setup.md).

## Layout

- `src/gdl/` — shared, paper-agnostic helpers (seed, metrics, logging, run, checkpoint)
- `papers/NN-name/` — one paper each (model, data, train, evaluate, README, runs/)
- `papers/00-template/` — the runnable template; copy it to start a new paper

## Learning from this repo

Each paper folder has its own README — what the paper does, how to run it, and the
expected results — so it works as a self-contained learning resource. New to
geometric deep learning? Start at `papers/00-template/` (the simplest), then read
each paper's README to follow the progression.

## AI operating surface

Built with **Claude Code**.

- **Rules** → [AGENTS.md](AGENTS.md)
- **Skills** → `.agents/skills/` — `run-paper` (run train/eval), `gdl-repo-map` (repo map)
- **MCP** → context7 (live library docs)
- **Docs** → [docs/spec.md](docs/spec.md) (scope + contracts), [docs/reference-index.md](docs/reference-index.md)

## Papers

**Status:** [docs/progress.md](docs/progress.md) — canonical tracker, update it when you
work on a paper.

Full list and tiers → [references/papers.md](references/papers.md).
