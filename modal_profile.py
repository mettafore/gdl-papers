"""Profile where per-epoch time goes: data loading (CPU collate) vs
compute (GPU forward+backward+step). Diagnostic only — doesn't touch
train.py or the live training run.

Usage:
    uv run modal run modal_profile.py
"""

import modal

app = modal.App("gdl-papers-egnn-profile")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["dev"])
    .add_local_file("pyproject.toml", remote_path="/root/gdl-papers/pyproject.toml", copy=True)
    .add_local_file("README.md", remote_path="/root/gdl-papers/README.md", copy=True)
    .add_local_dir("src/gdl", remote_path="/root/gdl-papers/src/gdl", copy=True)
    .add_local_dir(
        "papers/01-EGNN/src",
        remote_path="/root/gdl-papers/papers/01-EGNN/src",
        copy=True,
    )
    .run_commands("cd /root/gdl-papers && uv pip install --no-deps -e .")
)

volume = modal.Volume.from_name("egnn-qm9", create_if_missing=True)
VOLUME_PATH = "/vol"


@app.function(image=image, gpu="T4", volumes={VOLUME_PATH: volume}, timeout=600)
def profile(n_batches=50):
    import sys
    import time

    import torch

    sys.path.insert(0, "/root/gdl-papers/papers/01-EGNN/src")
    import data
    from model import EGNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, _, _, norm, dims = data.load_split(
        target="gap", batch_size=96, root=f"{VOLUME_PATH}/data/qm9",
    )
    egnn = EGNN(in_node_nf=dims["in_node_dim"], hidden_nf=128, n_layers=7).to(device)
    optimizer = torch.optim.Adam(egnn.parameters(), lr=5e-4)
    loss_fn = torch.nn.L1Loss()

    from train import target_for

    data_time = 0.0
    compute_time = 0.0

    it = iter(train_loader)
    for i in range(n_batches):
        t0 = time.perf_counter()
        b = next(it)
        b = b.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        data_time += t1 - t0

        optimizer.zero_grad()
        out = egnn(b.x, b.pos, b.edge_index, batch=b.batch)
        target = target_for(b, norm, dims["target_col"])
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        compute_time += t2 - t1

    total = data_time + compute_time
    print(f"batches profiled: {n_batches}")
    print(f"data loading:  {data_time:.2f}s  ({100*data_time/total:.1f}%)")
    print(f"compute:       {compute_time:.2f}s  ({100*compute_time/total:.1f}%)")
    print(f"avg per batch: {total/n_batches*1000:.1f}ms  "
          f"(data {data_time/n_batches*1000:.1f}ms, compute {compute_time/n_batches*1000:.1f}ms)")

    batches_per_epoch = len(train_loader)
    print(f"batches per epoch: {batches_per_epoch}")
    print(f"estimated time per epoch: {total/n_batches*batches_per_epoch:.1f}s")


@app.local_entrypoint()
def main(n_batches: int = 50):
    profile.remote(n_batches=n_batches)
