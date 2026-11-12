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
| 20261111-110827 | 4 | 1 | 256 | no | 0 | 1.6e-3 | no | 3 | 73.6 | 0.2637 |
| 20261111-111351 | 4 | 2 | 256 | no | 0 | 2.3e-3 | no | 3 | 48.0 | 0.3510 |
| 20261111-111745 | 4 | 2 | 256 | no | 0 | 2.3e-3 | yes | 1 | (86.5, ep 1) | — |

| 20261112-071358 | 6 | 2 | 256 | no | 0 | 2.3e-3 | no | 3 | 22.2 | 0.3145 |
| 20261112-071711 | 6 | 2 | 256 | yes | 0 | 2.3e-3 | no | 3 | 13.5 | 0.3139 |
| 20261112-071957 | 6 | 1 | 256 | no | 0 | 1.6e-3 | no | 3 | 42.7 | 0.2858 |
| 20261112-072346 | 6 | 2 | 256 | no | 0 | 2.3e-3 | yes | 1 | (31.6, ep 1) | — |
| 20261112-072552 | 6 | 2 | 256 | no | 0 | 2.3e-3 | yes | 1 | (38.9, ep 1) | — |

From job 4 on, `config.json` has a `hardware` section (world_size, gpu, amp, profile).
Job 6 runs all use `--precompute`; all except 20261112-072346 use `--compile`.

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

## Full ladder, one job (job 7, 12 Nov) — the clean comparison

All rungs in one Kaggle job (same machine, same code `cfa9cd0`, so rung 3A — no `.item()`
sync — is in every rung). QM9 downloaded once with one process first (fixes the DDP download
race). s/epoch = epoch 3. Profiles: 1-epoch runs, mean of recorded steps 2–3, rank 0.
This Kaggle machine was faster than jobs 1–6 (rung 0: 43.1 vs 50.2 s), so compare within
this table, not across jobs.

| Rung | Flags (2 GPUs) | s/epoch | Step | Data wait | GPU busy | Kernels/step | Val (3 ep) | Run ID |
|---|---|---|---|---|---|---|---|---|
| 0 | batch 96, LR 1e-3 | 43.1 | 101 ms | 61 ms | 50% | ~670 | 0.2675 | 20261112-094206 |
| 1 | batch 256, LR 2.3e-3 | 34.7 | 189 ms | 150 ms | 59% | ~700 | 0.3094 | 20261112-094524 |
| 2 | + `--precompute` | 28.9 | 121 ms | 17 ms | **97%** | ~670 | 0.3268 | 20261112-094841 |
| 3 | + `--compile` | 24.9 | 92 ms | 18 ms | 96% | ~400 | 0.3616 | 20261112-095135 |
| 3 + fp16 | + `--amp` | **14.2** | 57 ms | 16 ms | **64%** | ~590 | 0.3212 | 20261112-095456 |

1 GPU: rung 0 = 63.4 s (20261112-095725), rung 3 + fp16 = 23.9 s (20261112-100205).
- **Total: 63.4 s (1 GPU, rung 0) → 14.2 s (2 GPUs, rung 3 + fp16) = 4.5×.** 2 GPUs only: 43.1 → 14.2 = 3.0×.
- 2-GPU scaling: 1.47× at rung 0, 1.68× at rung 3 + fp16.
- fp16 halves the GPU time per step (88 → 37 ms), so the GPU is busy only 64% again: the
  fixed CPU part (~16 ms data/collate per step + launches) is now a bigger share.
  **The bottleneck moved back to the CPU side.** That is the case where rung 4 (GPU-resident
  packed batches + CUDA Graphs) would pay off.
- Profile run IDs: 20261112-100445 (R0), -100646 (R1), -100912 (R2), -101120 (R3), -101337 (R3+fp16).
- Val loss after 3 epochs differs by recipe (batch/LR) and is noisy at 3 epochs; it is a
  sanity check, not a convergence result.

## Speed ladder (2 GPUs), first pass (jobs 1–6)

Each rung adds one change on top of the previous rung. Write the prediction before the run.

| Rung | Change | Prediction | s/epoch | GPU busy | Kernels/step | Val loss (3 ep) | Run ID |
|---|---|---|---|---|---|---|---|
| 0 | Baseline: 96/GPU, fp32, 0 workers | — | 50.2 | ~30%* | ~986* | 0.2507 | 20261110-104137 |
| 1 | + batch 256/GPU, LR 2.3e-3 | — | 48.0 | 32–41% | ~620 | 0.3510 | 20261111-111351 |
| 2 | + precompute data (+ 3A: no `.item()` sync) | — | ~31 (ep 1, profiled)† | 97% | ~670 | — | 20261112-072346 |
| 3 | + `torch.compile` | — | **22.2** | 95–96% | ~430 | 0.3145 | 20261112-071358 |
| 3 + fp16 | + `--amp` | — | **13.5** | — | — | 0.3139 | 20261112-071711 |
| 4 | + packing + CUDA Graphs | not built: GPU already ≥ 95% busy | | | | | |

† The plain rung 2 run (3 epochs) crashed before training: on the fresh Kaggle machine both
DDP ranks downloaded and processed QM9 at the same time (job 6 log). Fix for next time: download
the dataset once (one plain `python` call, or rank 0 first + barrier) before `torchrun`.
The 31 s is the 1-epoch profiled run, which includes profiler overhead.

Rung 3 on **1 GPU** (20261112-071957, batch 256, LR 1.6e-3): 42.7 s/epoch → 2-GPU scaling
42.7 → 22.2 = **1.92×** (was 1.56× at rung 0).

Profiles after rung 2 and 3 (2 GPUs, rank 0, 3 steps):

| | Step | Data wait | GPU busy | Kernels | Forward CPU | Backward CPU |
|---|---|---|---|---|---|---|
| Rung 1 | ~360 ms | ~280 ms | 32–41% | ~620 | ~48 ms | ~30 ms |
| Rung 2 | ~122 ms | ~15 ms | **97%** | ~650 | ~11 ms | ~23 ms |
| Rung 3 | ~90 ms | ~16 ms | **95–96%** | ~430 | ~11 ms | ~20 ms |

- Rung 2 removed the data bottleneck: the GPU went from ~35% to 97% busy.
- Rung 3 (compile) fused kernels (~650 → ~430) and cut the step from ~122 to ~90 ms.
- Now GPU-bound, so fp16 finally helps: 22.2 → 13.5 s (1.64×). At rungs 0–1 it gave 0%.
- **Total: 50.2 → 13.5 s/epoch on 2×T4 (3.7×); 78.4 s (1 GPU, rung 0) → 13.5 s = 5.8×.**
- Rung 4 dropped: it removes CPU launch overhead, and the GPU is already ≥ 95% busy.

\* Measured on the profiled run with fp16 and 2 workers (20261110-131533), not on the exact rung 0 setup.

Rung 1 profile (20261111-111745, 3 steps): step ≈ 360 ms, of which **≈ 280 ms is waiting for the
DataLoader**. Forward + backward CPU ≈ 75 ms, GPU busy ≈ 120–150 ms. The bigger batch cut the
launch cost per molecule, but the per-molecule data work (`transform` + collate, 0 workers) now
dominates. The bigger batch only gave 50.2 → 48.0 s.

## JAX track (branch `jax-egnn`, job 5, log `results/kaggle-logs/job5-jax.log`)

Plain JAX + optax, whole train step under `jax.jit`, data on the GPU, every molecule padded to
29 atoms / 812 edges (≈ 39% of edge slots real). fp32. Train time per epoch (validation excluded):

| GPUs | Batch/GPU-total | Train s/epoch | Val MAE after 3 ep |
|---|---|---|---|
| 1 | 96 | 145.9 | 0.2211 |
| 2 | 96 | 73.4 | 0.2196 |
| 2 | 256 | 73.5 | 0.2648 |

- Scaling 1 → 2 GPUs: 145.9 → 73.4 s = **1.99×**. Almost perfect.
- Batch 96 → 256: no change. The step is **GPU-bound**: time follows the number of
  (padded) molecules, not the number of steps.
- Slower than PyTorch per GPU because of the padding waste (~2.6× edge work) and fp32.
- Val MAE at 1 GPU, batch 96 matches PyTorch (0.2211 vs 0.2225).
