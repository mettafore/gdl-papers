# RFM Fire Data Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the minimum Fire-data scaffold so Luv can explore the official CSV and implement a tested latitude/longitude-to-`S^2` data pipeline.

**Architecture:** Keep exploratory work in one notebook and reproducible data behavior in one Python module. AI owns direct contract tests; all correctness-determining data bodies remain `TODO(luv)` stubs that initially raise `NotImplementedError`.

**Tech Stack:** Python 3.12, PyTorch, Jupyter, pytest, uv; no new dependencies.

## Global Constraints

- Fire is the only dataset in this slice.
- Follow the `papers/00-template` paper-local layout.
- Use the official author convention: CSV columns are latitude then longitude in degrees; Cartesian output is `(x, y, z)` on the unit sphere.
- Use deterministic 80/10/10 splits with seed `0` by default.
- AI writes and runs tests; Luv writes coordinate conversion, loading, validation, and split logic.
- Do not scaffold geometry, model, training, evaluation, Bunny, Hydra, Lightning, EMA, or ODE code yet.
- Do not add dependencies.
- Keep `papers/14-RFM/nbs/eq13_path_velocity_test.md` untracked and untouched.

---

### Task 1: Fire experiment landing page and EDA shell

**Files:**
- Create: `papers/14-RFM/README.md`
- Create: `papers/14-RFM/nbs/fire_eda.ipynb`

**Interfaces:**
- Consumes: official `fire.csv` path supplied by the user after download.
- Produces: a documented Fire-only scope and an executable notebook outline with no analysis implementation.

- [ ] **Step 1: Create the README contract**

Document only: purpose, current files, official-data provenance, expected CSV convention, commands, three-day completion criteria, and the paper reference NLL `-1.86 +/- 0.11`.

- [ ] **Step 2: Create the EDA notebook outline**

Create Markdown headings and empty code cells for:

1. imports and data path;
2. raw shape and first rows;
3. missing/non-finite values;
4. latitude/longitude ranges;
5. geographic scatter;
6. Cartesian conversion;
7. unit-norm check;
8. seeded split counts.

- [ ] **Step 3: Validate notebook structure**

Run: `jq empty papers/14-RFM/nbs/fire_eda.ipynb`

Expected: exit code `0`.

- [ ] **Step 4: Commit the documentation scaffold**

Stage only the README and EDA notebook. Commit message:

```text
docs(rfm): scaffold fire data exploration
```

### Task 2: Tested data contracts

**Files:**
- Create: `papers/14-RFM/src/data.py`
- Create: `papers/14-RFM/test_data.py`

**Interfaces:**
- Produces: `latlon_degrees_to_cartesian(latlon: torch.Tensor) -> torch.Tensor`.
- Produces: `load_fire_csv(path: str | Path) -> torch.Tensor`, returning Cartesian float32 points with shape `(n, 3)`.
- Produces: `split_points(points: torch.Tensor, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`, ordered train, validation, test.

- [ ] **Step 1: Write the exact failing tests**

AI-authored tests must cover:

- `(n, 2)` latitude/longitude input returns `(n, 3)` float32 output;
- every Cartesian row has unit norm;
- `(lat, lon) = (0, 0)` maps to the positive x-axis;
- `(lat, lon) = (90, 0)` maps to the positive z-axis;
- invalid input shape, NaN/Inf, latitude outside `[-90, 90]`, and longitude outside `[-180, 180]` raise `ValueError`;
- a temporary CSV with header plus valid rows loads into the same Cartesian values as direct conversion;
- a 101-row tensor splits into 81 train, 10 validation, and 10 test rows;
- the same seed reproduces identical splits and a different seed changes at least one partition;
- the three partitions contain every original row exactly once.

- [ ] **Step 2: Create learning-owned stubs**

Add imports, typed signatures, docstrings, input/output contracts, and
`TODO(luv)` intent bullets. Each body raises `NotImplementedError`; no formula,
library call sequence, split indices, or implementation hint is included.

- [ ] **Step 3: Run the focused tests and verify red state**

Run:

```text
uv run pytest papers/14-RFM/test_data.py -q
```

Expected: tests fail specifically because the three functions raise
`NotImplementedError`; collection and imports succeed.

- [ ] **Step 4: Run static checks on Python scaffold files**

Run:

```text
uv run ruff format papers/14-RFM/src/data.py papers/14-RFM/test_data.py
uv run ruff check papers/14-RFM/src/data.py papers/14-RFM/test_data.py
uv run mypy papers/14-RFM/src/data.py papers/14-RFM/test_data.py
```

Expected: pass, except a type/lint diagnostic caused solely by an intentional
`NotImplementedError` stub may remain under the repository rules.

- [ ] **Step 5: Review the scaffold diff**

Confirm that tests state behavior without embedding the conversion or split
implementation, and confirm no unrelated or scratch file is staged.

- [ ] **Step 6: Commit the red scaffold**

Stage only `src/data.py` and `test_data.py`. Commit message:

```text
test(rfm): scaffold fire data contracts
```

### Task 3: Learning handoff

**Files:**
- Modify: `docs/progress.md`
- Modify: `/Users/luvsuneja/Documents/wiki/GeometricDeepLearning/Papers/RFM.md`

**Interfaces:**
- Consumes: the committed red data scaffold.
- Produces: one unambiguous next action for Luv.

- [ ] **Step 1: Update repository progress**

Record Fire as the selected sphere dataset and state that the data scaffold is
red by design.

- [ ] **Step 2: Update the Obsidian handoff**

Record the Fire-only decision, AI test ownership, and the first implementation
contract: make `latlon_degrees_to_cartesian` pass its focused tests.

- [ ] **Step 3: Verify the final state**

Run:

```text
git status --short
git log -3 --oneline
```

Expected: only the pre-existing scratch Markdown remains untracked; the plan,
design revision, documentation scaffold, and red test scaffold are committed.
