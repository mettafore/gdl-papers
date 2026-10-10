# MACE experiment notes

## 2026-10-11: stopped seed 123 and evaluated the best checkpoint

Stopped training app `ap-Mq3FdH8F4u2eF40ccXy5hs` at the learner's request to
preserve the remaining $2.99 credit. Modal confirmed zero running training
tasks at 01:36:53 Dubai time. Run `20261009-125019` completed 367 epochs,
representing 33,030 optimizer updates at persisted epoch boundaries. Its best
validation EMA checkpoint is from epoch 360 (`best_epoch=359` in the zero-based
training state), with weighted validation loss `0.13177227648722206`.

Evaluated that fixed best checkpoint in float64 on an A100-40GB using the
existing paper evaluator. All 12,993 configurations in the four holdout
conditions were evaluated. Energy RMSE is in meV per configuration;
force-component RMSE is in meV/angstrom. The checkpoint's tensors matched the
recorded best EMA state before evaluation. Evaluation took 29.7 minutes.

| Holdout | Configurations | This run: energy | Paper: energy | Energy gap | This run: forces | Paper: forces | Force gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 300 K | 1,669 | 3.99 | 3.0 | +33.1% | 11.56 | 8.8 | +31.4% |
| 600 K | 2,138 | 13.28 | 9.7 | +36.9% | 28.96 | 21.8 | +32.9% |
| 1200 K | 2,139 | 44.51 | 29.8 | +49.3% | 96.01 | 62.0 | +54.9% |
| Dihedral slices | 7,047 | 17.57 | 7.8 | +125.3% | 21.15 | 16.5 | +28.2% |

The paper values are the MACE L=2 three-seed means from Table 2, recorded in
the [paper README](README.md#results). All eight errors exceed the paper
means; none meet the repository's 10% target. The largest gap is dihedral
energy RMSE, about 2.25 times the paper mean. This is a single-seed result
after an intentionally stopped run and does not establish benchmark reproduction.
All four conditions improved over the earlier epoch-30 evaluation.

The [full-precision results](holdout-seed123-epoch360.json) include sample
counts, checkpoint hash, source and test-file hashes, model configuration,
completed epochs, and percentage differences. The original evaluation result
also remains in the local run folder and the persistent Modal Volume
`gdl-papers-mace-runs` as `runs/20261009-125019/mace-holdout-best-epoch-360.json`.
Both `checkpoint.pt` and `training_state.pt` remain on that volume and were
downloaded to the local run folder. Training remains stopped; resuming paid
training requires a new learner request.

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
