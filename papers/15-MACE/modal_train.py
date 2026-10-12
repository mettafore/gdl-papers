"""Run resumable full 3BPA MACE training on Modal's persistent T4 volume.

Fresh run:
    uv run modal run --detach papers/15-MACE/modal_train.py --epochs 3 --seed 123

Continue to 500 epochs, including automatic retries after interruption:
    uv run modal run --detach papers/15-MACE/modal_train.py --epochs 500 --seed 123 \
        --resume-run-id <run_id>
"""

import json
from pathlib import Path

import modal

REMOTE_ROOT = "/root/gdl-papers"
PAPER_SRC = f"{REMOTE_ROOT}/papers/15-MACE/src"
DATA_DIR = f"{REMOTE_ROOT}/papers/15-MACE/data/dataset_3BPA"
RUNS_ROOT = "/mace-persistent"

app = modal.App("gdl-papers-mace-training")
runs_volume = modal.Volume.from_name("gdl-papers-mace-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["dev"])
    .add_local_file(
        "pyproject.toml", remote_path=f"{REMOTE_ROOT}/pyproject.toml", copy=True
    )
    .add_local_file("README.md", remote_path=f"{REMOTE_ROOT}/README.md", copy=True)
    .add_local_dir("src/gdl", remote_path=f"{REMOTE_ROOT}/src/gdl", copy=True)
    .add_local_dir("papers/15-MACE/src", remote_path=PAPER_SRC, copy=True)
    .add_local_dir(
        "papers/15-MACE/data/dataset_3BPA",
        remote_path=DATA_DIR,
        copy=True,
    )
    .run_commands(f"cd {REMOTE_ROOT} && uv pip install --no-deps -e .")
)


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    timeout=24 * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
    single_use_containers=True,
    volumes={RUNS_ROOT: runs_volume},
)
def run_training(
    *, epochs: int, seed: int, resume_run_id: str | None
) -> dict[str, float | int | str]:
    """Train or resume one seed, persisting each epoch to the named Volume."""
    import sys

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal allocated no CUDA device")

    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    sys.path.insert(0, PAPER_SRC)

    from train import TrainConfig, train

    run_id, run_path = train(
        data_dir=Path(DATA_DIR),
        paper_dir=Path(RUNS_ROOT),
        train_config=TrainConfig(max_epochs=epochs, seed=seed),
        device="cuda",
        resume_run_id=resume_run_id,
        on_epoch_checkpoint=runs_volume.commit,
    )
    runs_volume.commit()

    with (run_path / "metrics.jsonl").open() as metrics_file:
        metrics = [json.loads(line) for line in metrics_file]
    latest = metrics[-1]
    return {
        "run_id": run_id,
        "gpu": torch.cuda.get_device_name(),
        "epochs_completed": int(latest["step"]) + 1,
        "best_epoch": int(latest["best_epoch"]),
        "best_validation_loss": float(latest["best_validation_loss"]),
    }


@app.local_entrypoint()
def main(
    epochs: int = 500,
    seed: int = 123,
    resume_run_id: str | None = None,
) -> None:
    result = run_training.spawn(
        epochs=epochs,
        seed=seed,
        resume_run_id=resume_run_id,
    ).get()
    print(json.dumps(result, indent=2, sort_keys=True))
