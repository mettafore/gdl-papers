# MACE experiment notes

## 2026-10-18: seed 123 interim holdout evaluation

Evaluated the latest EMA checkpoint for run `20261009-125019` after 30 of 500
target epochs. Epoch 30 was also the best validation epoch. The evaluation used
the published 256-channel, two-interaction, correlation-3, `L=2` configuration
and covered all 12,993 configurations in the four holdout conditions. Energy
RMSE is in meV; force-component RMSE is in meV/angstrom.

| Holdout | Configurations | This run: energy | Paper: energy | This run: forces | Paper: forces |
|---|---:|---:|---:|---:|---:|
| 300 K | 1,669 | 21.62 | 3.0 ± 0.2 | 27.30 | 8.8 ± 0.3 |
| 600 K | 2,138 | 33.65 | 9.7 ± 0.5 | 52.99 | 21.8 ± 0.6 |
| 1200 K | 2,139 | 79.49 | 29.8 ± 1.0 | 130.27 | 62.0 ± 0.7 |
| Dihedral slices | 7,047 | 34.05 | 7.8 ± 0.6 | 34.75 | 16.5 ± 1.7 |

The paper values are the three-run means and standard deviations from Table 2 of
[Batatia et al.](https://arxiv.org/html/2206.07697v2). This is one early
checkpoint from one seed, so it does not establish the three-seed benchmark.
All eight errors are above the paper means. The saved full-precision result and
model/run metadata are in the ignored run folder at
`runs/20261009-125019/mace-holdout-epoch-30.json`.

Evaluation ran in float64 on an NVIDIA A100 and took about 28.9 minutes. The
separate seed-123 training task remained active when checked after evaluation.
