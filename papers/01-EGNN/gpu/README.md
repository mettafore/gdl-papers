# EGNN on multiple GPUs — experiment record

Lab notebook for the GPU training primer (steps 3–5 and the speed ladder).
EGNN, QM9 target `gap`, Kaggle "GPU T4 x2" (2× Tesla T4, 4 CPUs).
Code: `../src/train.py` flags (`--batch-size`, `--amp`, `--profile`, `--num-workers`).

## Where the evidence is

- `runs/<run_id>/` (paper folder, **not in git**): full run, with checkpoint and profiler trace (`tb/`).
- `gpu/results/<run_id>/` (**in git**): `config.json` + `metrics.jsonl` copies.
- `gpu/results/kaggle-logs/jobN.log` (**in git**): the full console log of each Kaggle job, with profiler tables.

`config.json` does not yet record the number of GPUs, `--amp` or `--profile`.
For the runs below, those columns come from the order of the commands in the Kaggle cell.

## Run index

Epoch time = epoch 3 (epoch 1 includes warm-up). It includes validation, which runs on GPU 0 only.
Val loss = normalized L1 after the last epoch.

| Run ID | Job | GPUs | Batch/GPU | fp16 | Workers | LR | Profile | Epochs | s/epoch | Val loss |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261110-103604 | 1 | 1 | 96 | no | 0 | 1e-3 | no | 3 | 78.4 | 0.2225 |
| 20261110-104137 | 1 | 2 | 96 | no | 0 | 1e-3 | no | 3 | 50.2 | 0.2507 |
| 20261110-104539 | 1 | 2 | 48 | no | 0 | 1e-3 | no | 3 | 60.0 | 0.2267 |
| 20261110-113611 | 2 | 1 | 96 | yes | 0 | 1e-3 | no | 3 | 79.4 | 0.2138 |
| 20261110-114154 | 2 | 2 | 96 | yes | 0 | 1e-3 | no | 3 | 49.8 | 0.2524 |
| 20261110-114558 | 2 | 2 | 96 | no | 0 | 1.4e-3 | no | 3 | 49.0 | 0.2461 |
| 20261110-114954 | 2 | 1 | 96 | no | 0 | 1e-3 | yes | 1 | (113.2, ep 1) | — |
| 20261110-115256 | 2 | 2 | 96 | no | 0 | 1e-3 | yes | 1 | (77.0, ep 1) | — |
| 20261110-115526 | 2 | 2 | 96 | yes | 0 | 1e-3 | yes | 1 | (80.7, ep 1) | — |
| 20261110-125921 | 3 | 1 | 96 | no | 4 | 1e-3 | no | 3 | 59.1 | 0.2215 |
| 20261110-130322 | 3 | 1 | 96 | yes | 4 | 1e-3 | no | 3 | 59.2 | 0.2224 |
| 20261110-130728 | 3 | 2 | 96 | no | 2 | 1e-3 | no | 3 | 54.4 | 0.2514 |
| 20261110-131124 | 3 | 2 | 96 | yes | 2 | 1e-3 | no | 3 | 58.9 | 0.2527 |
| 20261110-131533 | 3 | 2 | 96 | yes | 2 | 1e-3 | yes | 1 | (65.7, ep 1) | — |

Job 3 also ran `src/equivariance_fp16.py` (random-init model): fp32 error 6.0e-7,
fp16 error 7.3e-4, fp16 rounding error 8.1e-4. Rotation adds no error above fp16 rounding.

## Findings so far

1. 2 GPUs at 96/GPU: 78.4 → 50.2 s/epoch (1.56×). At 48/GPU: 1.31×.
2. Throughput is not convergence: 2 GPUs at 96/GPU make half the optimizer steps per epoch,
   so val loss after 3 epochs is worse (0.251 vs 0.222). 48/GPU matches 1 GPU (0.227).
3. fp16 gives no speedup in any setup.
4. Profiler (run 20261110-131533, one step): the GPU is busy only 24–39% of each step.
   The CPU sends ~986 small jobs (kernels) per step and takes ~100 ms; the GPU calculates for ~29 ms.
   Some steps also wait 20–50 ms for data. The NCCL all-reduce takes ~0.5 ms per step.
5. Cause: the step is launch-bound (Python sends many small kernels) and the per-molecule
   `transform` in `data.py` runs on the CPU every epoch.

## Speed ladder (2 GPUs)

Each rung adds one change on top of the previous rung. Write the prediction before the run.

| Rung | Change | Prediction | s/epoch | GPU busy | Kernels/step | Val loss (3 ep) | Run ID |
|---|---|---|---|---|---|---|---|
| 0 | Baseline: 96/GPU, fp32, 0 workers | — | 50.2 | ~30%* | ~986* | 0.2507 | 20261110-104137 |
| 1 | + batch 256/GPU | | | | | | |
| 2 | + precompute data | | | | | | |
| 3 | + `torch.compile` | | | | | | |
| 4 | + padding + CUDA Graphs | | | | | | |

\* Measured on the profiled run with fp16 and 2 workers (20261110-131533), not on the exact rung 0 setup.
