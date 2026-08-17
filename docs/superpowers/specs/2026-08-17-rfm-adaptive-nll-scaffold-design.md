# RFM Adaptive NLL Scaffold Design

## Goal

Replace the fixed-step likelihood scaffold with paper-aligned adaptive ODE
plumbing while preserving the two central change-of-variables decisions for
Luv to implement.

## Boundary

The scaffold owns input validation, the augmented `(n, 4)` state allocation,
time broadcasting, the backward `[1, 0]` interval, the `torchdiffeq.odeint`
call with `dopri5`, extraction of the final state, and tensor-to-float return
plumbing.

Luv owns exactly two correctness-determining expressions:

1. The augmented derivative joining the learned velocity with the correctly
   signed divergence accumulator derivative.
2. The final change-of-variables expression joining the uniform sphere base
   log density and accumulated correction into mean NLL.

Both expressions remain marked `TODO(luv)` and raise `NotImplementedError`
until completed.

## Tests

AI-owned tests specify two invariants:

- A zero vector field has NLL `log(4*pi)`.
- A field with constant divergence `-0.5` has NLL `log(4*pi) - 0.5` under the
  backward solve.

The tests remain strict expected failures only while the two learner-owned
expressions are incomplete. All unrelated evaluation tests must pass.
