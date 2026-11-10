# DDP for EGNN: what changed and why

How `src/train.py` and `src/data.py` were changed to train on several processes
with PyTorch DistributedDataParallel (DDP). Written so I can redo this on another
model (MACE next) without re-learning it.

Branch `ddp-egnn`, first commit `d8120ab` (10 Nov 2026). Step 2 of the GPU
training primer in my wiki.

## TL;DR

`torchrun --nproc_per_node=N` runs N copies of the script at once. Each copy
trains on its own slice of the data, and DDP averages their gradients, so the
weights stay identical. Every DDP line is guarded by `if "RANK" in os.environ:` (the `coord_mlp` freeze is not, but it's a no-op in plain mode, since those weights never get gradients):
`torchrun` sets `RANK`, plain `python` doesn't, so plain `python` still works.

**Run:**

```bash
uv run python papers/01-EGNN/src/train.py                       # 1 process, as before
GLOO_SOCKET_IFNAME=lo0 .venv/bin/torchrun --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29500 papers/01-EGNN/src/train.py   # 2 processes
```

The `GLOO_SOCKET_IFNAME` / `--master_addr` part is Mac-only. Without it, the
processes hang trying to find each other.

| # | Step | Code | Why |
|---|---|---|---|
| 1 | Connect the copies | `rank = int(os.environ.get("RANK", 0))`<br>`if "RANK" in os.environ:`<br>`    init_process_group(backend="gloo")`<br>…<br>`    destroy_process_group()  # before return` | Copies can exchange gradients. Plain `python` = rank 0. `gloo` = CPU, `nccl` = GPU. |
| 2 | Wrap the model, after the checkpoint load | `for name, p in egnn.named_parameters():`<br>`    if "coord_mlp" in name: p.requires_grad = False`<br>`model = egnn`<br>`if "RANK" in os.environ: egnn = DDP(egnn)` | DDP averages gradients after `backward()`. Unused `coord_mlp` would crash DDP. `model` = plain model for validation. |
| 3a | Split the data (`data.py`) | `sampler = DistributedSampler(train_data, shuffle=True) if "RANK" in os.environ else None`<br>`PyGDataLoader(..., sampler=sampler, shuffle=sampler is None)` | Each copy gets its own slice, 0 overlap. Train loader only. |
| 3b | Reshuffle each epoch | `if "RANK" in os.environ:`<br>`    train_loader.sampler.set_epoch(epoch)` | Otherwise every epoch repeats the same order. |
| 4 | Files on rank 0 only | `if rank == 0: run_id, run_dir = new_run_dir(...)`<br>`else: run_id, run_dir = None, None`<br>`save_checkpoint(egnn.module if "RANK" in os.environ else egnn, run_dir)` | No duplicate folders or logs. `egnn.module` avoids the `module.` key prefix. |
| 5a | LR schedule on every rank | `scheduler.step()  # outside if rank == 0` | Otherwise ranks get different LRs and weights drift. |
| 5b | Validate on rank 0 | `if rank == 0:`<br>`    model.eval(); ...; log_metrics(...)`<br>`    if epochs_since_improve >= early_stop_patience: stop = True` | No repeated work. |
| 5c | Everyone stops together | `flag = torch.tensor(int(stop))`<br>`torch.distributed.broadcast(flag, src=0)`<br>`stop = bool(flag.item())`<br>`if stop: break` | Every rank calls it. Rank 1 gets rank 0's decision. Without it, rank 1 hangs. |

Code in the table is shortened. The real code (and the sections below) uses
plain `if` blocks.

**Verified (Mac CPU):** both ranks early-stop together, 1 run folder, checkpoint
loads into a plain `EGNN`, and after real training the ranks' weights differ by
0.0.

## Details

Full code per step, plus what the table leaves out.

### 1. Connect the copies

```python
rank = int(os.environ.get("RANK", 0))
if "RANK" in os.environ:
    init_process_group(backend="gloo")
...
if "RANK" in os.environ:
    destroy_process_group()   # before return
```

- `nccl` needs NVIDIA GPUs. Use it in step 3.

### 2. Wrap the model

```python
for name, param in egnn.named_parameters():
    if "coord_mlp" in name:
        param.requires_grad = False
model = egnn                  # plain model, for validation
if "RANK" in os.environ:
    egnn = DDP(egnn)
```

- Wrap **after** the checkpoint load: DDP copies rank 0's weights to all ranks when created.
- `coord_mlp` is unused on QM9 (no coordinate update). DDP waits for every parameter's gradient → crash on batch 2.
- Freeze, not `find_unused_parameters=True`: that flag searches the graph every step and hides real bugs.
- Select by name. `param.shape[0] == 3` matched nothing.
- Validate with `model`: the wrapper's forward can sync buffers (e.g. BatchNorm stats) across ranks, which hangs if only rank 0 calls it. EGNN has no buffers today; the plain model is safe regardless.

### 3. Split the data

```python
# data.py
sampler: DistributedSampler | None = None
if "RANK" in os.environ:
    sampler = DistributedSampler(train_data, shuffle=True)
train_loader = PyGDataLoader(
    dataset[train_idx], sampler=sampler, shuffle=sampler is None,
    batch_size=batch_size, num_workers=num_workers,
)

# train.py, top of each epoch
if "RANK" in os.environ:
    train_loader.sampler.set_epoch(epoch)
```

- How: same shuffle on every rank (sampler seed 0 + epoch), pad to equal length, keep `indices[rank::world_size]`.
- Padding keeps batch counts equal. Otherwise one rank waits forever.
- Deleting `shuffle=True` instead silently trained plain `python` in fixed order.
- Train loader only: sampler indices reach 100k, val has 18k.
- Trap: the editor auto-imported `matplotlib.dates.set_epoch`. No error, no effect.
- Sampler seed defaults to 0 and ignores `set_seed()`: under `torchrun`, `--seed` changes init weights, not data order.

### 4. Files on rank 0 only

```python
if rank == 0:
    run_id, run_dir = new_run_dir(paper_dir, config, runs_base)
else:
    run_id, run_dir = None, None
...
if "RANK" in os.environ:
    save_checkpoint(egnn.module, run_dir)
else:
    save_checkpoint(egnn, run_dir)
```

- `None` so the variables exist on other ranks. `main()` prints `run_id` only if not `None`.
- Wrapper keys start with `module.`, which `evaluate.py` can't load. Plain `python` has no `.module`, hence the branch.

### 5. Validate on rank 0, stop together

```python
stop = False                  # top of each epoch, every rank
...
current_lr = optimizer.param_groups[0]["lr"]
scheduler.step()              # every rank
if rank == 0:
    ...                       # validate with `model`, log, checkpoint
    # on patience: stop = True
if "RANK" in os.environ:
    flag = torch.tensor(int(stop))
    torch.distributed.broadcast(flag, src=0)
    stop = bool(flag.item())
if stop:
    break
```

- `scheduler.step()` inside `if rank == 0` → only rank 0's LR decays → weights drift, no error.
- `broadcast` is a collective: every rank calls it, same point, every epoch.
- It sends tensors, and overwrites `flag`, not `stop`. Hence `int(...)` in, `bool(flag.item())` out.
- Other ranks return `(None, None, inf)` from `train()`: only rank 0 knows the run and `best_val`.
- If rank 0 crashes, `torchrun` kills the other ranks, so a crash does not leave rank 1 hanging in `broadcast`.
- Simpler alternative: validate on every rank. Same weights and val set give the same `val_loss` on CPU, so they stop together. On GPU, tiny float differences could split the decision; broadcast is safer.

## Verified (Mac CPU)

| Check | Result |
|---|---|
| Tiny data, `lr=0`, patience 1, plain and `torchrun` ×2 | Both stop at epoch 1, exit 0, 1 run folder, metrics written once |
| Checkpoint | Loads into plain `EGNN`, no `module.` keys |
| Data split | 50k / 50k, 0 overlap |
| 2 epochs at `lr=1e-3` | Weights moved 7.5e-3; rank difference **0.0** |
| Full data, 1 epoch | Plain: 261 s, val MAE 0.368. `torchrun` ×2: 326 s, val MAE 0.461. Both exit 0, 1 folder, 1 checkpoint |

Why `torchrun` was slower and worse after 1 epoch:
- Slower: `torchrun` sets `OMP_NUM_THREADS=1`, so 2 ranks get 2 threads, while plain `python` uses every core. Mac runs test correctness, not speed.
- Worse: same LR, but the effective batch is 192, not 96, so the epoch takes half as many steps (521 vs 1042). Expected; that's why LR scales up with N.

## Errors hit

| Error | Fix |
|---|---|
| `environment variable RANK expected, but not set` | Guard `init_process_group` |
| `SyntaxError: expected 'else' after 'if' expression` | Normal `if` block, not `call() if cond` |
| `Expected to have finished reduction...` | Freeze unused params (`coord_mlp`) |
| `too many values to unpack` | `named_parameters()`, not `parameters()` |
| `NameError: name 'os' is not defined` | Import `os` in `data.py` |
| Hang, `socket.cpp` IPv6 warnings | Mac loopback settings |

## For step 3 (GPUs)

- Backend: `nccl` if `torch.cuda.is_available()` else `gloo`.
- `nccl` only sends GPU tensors: create the stop flag with `device=device`, or `broadcast` fails.
- Device: `cuda:{LOCAL_RANK}` + `torch.cuda.set_device` per rank.
- Prepare QM9 before launch, or rank 0 first behind a barrier.
- Effective batch = batch × N. Adam: LR ×√N with warmup. Throughput ≠ convergence.
