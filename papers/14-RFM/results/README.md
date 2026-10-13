# RFM result artifacts

- [Local evaluation](local-evaluation-20261013.json): CPU evaluation on
  2026-10-13 of checkpoint `20260818-103835`, with test NLL −0.7004 across all
  1,282 held-out points and exit code 0. The record includes the seeded split,
  solver settings, evaluator revision, and checkpoint/data hashes. The CLI
  reports four decimal places; no additional precision is claimed.
- [Fire comparison](fire-comparison.png): qualitative visualization from the
  same saved run, with 12,809 observed locations and 2,000 generated samples.
  The observed panel includes training data and is not a held-out metric.

Training and evaluation are separate jobs. The reported NLL comes from the
local held-out evaluation, not from the training loss or the sample plot.
