# RFM Adaptive NLL Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide all adaptive exact-likelihood plumbing while leaving the two pedagogically central change-of-variables expressions to Luv.

**Architecture:** `negative_log_likelihood` constructs an augmented state and delegates backward integration to `torchdiffeq.odeint`. A nested RHS exposes the learner-owned derivative expression; a second learner-owned expression converts the solved accumulator into NLL.

**Tech Stack:** Python 3.12, PyTorch, torch.func, torchdiffeq, pytest, uv.

## Global Constraints

- Keep all implementation under `papers/14-RFM/`.
- Keep the ODE solve outside the training autograd graph, while permitting Jacobian computation inside the divergence function.
- Leave only the augmented derivative and final NLL expression learner-owned.

---

### Task 1: Adaptive NLL scaffold

**Files:**
- Modify: `papers/14-RFM/src/evaluate.py`
- Modify: `papers/14-RFM/test_evaluate.py`
- Test: `papers/14-RFM/test_evaluate.py`

**Interfaces:**
- Consumes: `divergence(model, x, t) -> Tensor[n]` and `torchdiffeq.odeint`.
- Produces: `negative_log_likelihood(model, samples, *, rtol, atol) -> float` after Luv completes two expressions.

- [ ] **Step 1: Preserve the existing likelihood contract tests**

Keep strict expected failures for the uniform zero-field baseline and the
constant-divergence sign check while learner markers remain.

- [ ] **Step 2: Add solver plumbing**

Import `odeint`; allocate the zero accumulator; concatenate the `(n, 4)` state;
define the nested RHS with state splitting and batched times; call `odeint` on
`[1, 0]` with `dopri5`, `rtol`, `atol`, and `min_step=1e-5`; extract the final
position and accumulator; calculate the uniform base log density.

- [ ] **Step 3: Preserve two learner-owned expressions**

The RHS raises `NotImplementedError` where velocity and signed divergence must
be concatenated. The final calculation raises `NotImplementedError` where base
log density and the correction must become mean NLL. Both locations carry a
short equation-oriented `TODO(luv)` comment.

- [ ] **Step 4: Verify the scaffold**

Run `uv run pytest papers/14-RFM/test_evaluate.py -q`; expected result is nine
passing tests and two strict expected failures. Run Ruff formatting/checking,
mypy, and the complete `papers/14-RFM` suite before committing.
