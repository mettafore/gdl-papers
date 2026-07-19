"""Run papers/01-EGNN/src/train.py on a Modal GPU container.

Usage:
    uv run modal run modal_train.py --epochs 2 --hidden-nf 8 --n-layers 2   # cheap smoke test
    uv run modal run modal_train.py --epochs 1000                          # real run (defaults)

QM9 data and run checkpoints persist in a Modal Volume across invocations,
so the ~130K-molecule download only happens once.
"""

import modal

app = modal.App("gdl-papers-egnn")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Reads pyproject.toml + uv.lock and installs the exact locked versions
    # (uv sync --frozen) — avoids hand-duplicating deps here and having them
    # silently drift from the real lockfile. Installs third-party deps only,
    # not this project's own `gdl` package (`--no-install-project`), hence
    # the separate pip install -e below. `groups=["dev"]` because Modal's
    # image builder wants its own `modal` package resolvable in the locked
    # environment, and it lives in the dev dependency group here.
    .uv_sync(groups=["dev"])
    # Copy only what's actually needed for the editable install + this paper
    # — not the whole repo (add_local_dir(".", ...) also pulled in .git,
    # .venv, other papers, and any cached QM9 data, and broke once because a
    # background git fetch mutated .git/FETCH_HEAD mid-snapshot).
    .add_local_file(
        "pyproject.toml", remote_path="/root/gdl-papers/pyproject.toml", copy=True
    )
    # pyproject.toml declares readme = "README.md" — hatchling's build
    # requires the file to actually exist, even though we don't need its
    # contents for anything here.
    .add_local_file("README.md", remote_path="/root/gdl-papers/README.md", copy=True)
    .add_local_dir("src/gdl", remote_path="/root/gdl-papers/src/gdl", copy=True)
    .add_local_dir(
        "papers/01-EGNN/src",
        remote_path="/root/gdl-papers/papers/01-EGNN/src",
        copy=True,
    )
    # uv_sync()'s venv has no pip installed (uv doesn't need it) — use
    # `uv pip install` instead of `python -m pip install`.
    .run_commands("cd /root/gdl-papers && uv pip install --no-deps -e .")
)

volume = modal.Volume.from_name("egnn-qm9", create_if_missing=True)
VOLUME_PATH = "/vol"


@app.function(
    image=image, gpu="T4", cpu=4, volumes={VOLUME_PATH: volume}, timeout=60 * 60 * 12
)
def run_train(
    target="gap",
    lr=5e-4,
    epochs=1000,
    hidden_nf=128,
    n_layers=7,
    seed=42,
    batch_size=96,
):
    import sys

    sys.path.insert(0, "/root/gdl-papers/papers/01-EGNN/src")
    from train import train

    run_id, run_dir, best_val = train(
        target=target,
        lr=lr,
        epochs=epochs,
        hidden_nf=hidden_nf,
        n_layers=n_layers,
        seed=seed,
        batch_size=batch_size,
        runs_base=f"{VOLUME_PATH}/runs",
        # new_run_dir() does Path(paper_dir) / runs_base / run_id — pathlib
        # discards the left side when the right side is absolute, so any
        # real path here works; runs_base being absolute is what actually
        # controls where the run folder lands.
        paper_dir="/root/gdl-papers/papers/01-EGNN",
        # QM9 lives on the Volume too, so it's downloaded once and reused —
        # PyG's QM9 dataset checks for already-processed files at `root`
        # before downloading, so repeat runs just load from disk.
        data_root=f"{VOLUME_PATH}/data/qm9",
        # profiled at ~1.35x speedup on a 4-cpu container (modal_profile.py)
        num_workers=4,
    )
    volume.commit()
    return run_id, run_dir, best_val


@app.local_entrypoint()
def main(
    target: str = "gap",
    lr: float = 5e-4,
    epochs: int = 1000,
    hidden_nf: int = 128,
    n_layers: int = 7,
    seed: int = 42,
    batch_size: int = 96,
):
    run_id, run_dir, best_val = run_train.remote(
        target=target,
        lr=lr,
        epochs=epochs,
        hidden_nf=hidden_nf,
        n_layers=n_layers,
        seed=seed,
        batch_size=batch_size,
    )
    print(f"run_id: {run_id}")
    print(f"run_dir (in Modal volume): {run_dir}")
    print(f"best_val: {best_val}")
