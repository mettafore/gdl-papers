# RFM Stage 1 Scaffold Design

## Scope

Create the standard paper-folder code shape inside `papers/14-RFM/` without a
notebook and without implementing any Riemannian Flow Matching mathematics.
Existing paper materials in the folder remain untouched.

## Files

- `README.md`: a minimal paper overview, current status, file map, and the
  Volcano benchmark target of test NLL `-7.93 +/- 1.67`.
- `src/data.py`: signatures and contracts for sphere sampling and later dataset
  loading; correctness-determining bodies remain `TODO(luv)` stubs.
- `src/model.py`: placeholder module for the Stage 3 tangent-projected MLP.
- `src/train.py`: placeholder training entry point.
- `src/evaluate.py`: placeholder evaluation entry point.
- `test_scaffold.py`: Stage 1 assertion boilerplate for `sample_sphere(n)`,
  checking shape `(n, 3)` and unit norms. It is expected to fail with
  `NotImplementedError` until luv fills the sampling body.

## Authorship Boundary

The scaffold may provide imports, signatures, docstrings, return-shape
contracts, TODO steps, and assertions. It must not implement sphere sampling,
exp/log maps, the premetric or its gradient, the conditional vector field, the
RCFM loss, or any other correctness-determining mathematical line.

## Verification and Stop Point

Run formatting, linting, and type checks on Python scaffold files. Run only the
Stage 1 test and confirm it reaches the intentional `NotImplementedError` stub.
Stop before adding the exp/log-map stage or any training functionality.
