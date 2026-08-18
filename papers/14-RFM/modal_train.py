"""Run an author-scale Fire calibration on a Modal GPU."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, cast

import modal

REMOTE_ROOT = "/root/rfm"

if modal.is_local():
    repo_root = Path(__file__).resolve().parents[2]
    paper_dir = Path(__file__).resolve().parent
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "torch==2.2.2",
            "numpy<2",
            "pandas>=2.3.3",
        )
        .add_local_dir(
            str(repo_root / "src" / "gdl"),
            f"{REMOTE_ROOT}/gdl",
        )
        .add_local_dir(
            str(paper_dir / "src"),
            f"{REMOTE_ROOT}/paper_src",
        )
        .add_local_file(
            str(paper_dir / "data" / "fire.csv"),
            f"{REMOTE_ROOT}/data/fire.csv",
        )
    )
else:
    image = modal.Image.debian_slim(python_version="3.12")

app = modal.App("rfm-fire-calibration")


@app.function(image=image)
def run_calibration(
    *,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    n_layers: int,
    lr: float,
    weight_decay: float,
    seed: int,
    train_pct: float,
    val_pct: float,
    test_pct: float,
    paper_dir: str,
) -> dict[str, float | int | str]:
    """Train the Fire model remotely and return timing and loss metrics."""
    sys.path.insert(0, f"{REMOTE_ROOT}/paper_src")
    sys.path.insert(0, REMOTE_ROOT)

    from train import train_run

    import data as d

    paper_data = cast(Any, d)

    data_path = Path(f"{REMOTE_ROOT}/data/fire.csv")
    points = paper_data.load_fire_csv(data_path)
    train_data, _, _ = paper_data.split_points(
        points,
        seed=seed,
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
    )

    steps_per_epoch = math.ceil(len(train_data) / batch_size)
    expected_steps = epochs * steps_per_epoch

    start = time.perf_counter()
    run_id, _, result = train_run(
        data_path=data_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        seed=seed,
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        device="cuda",
        paper_dir=paper_dir,
    )
    elapsed = time.perf_counter() - start

    return {
        "run_id": run_id,
        "steps": expected_steps,
        "seconds": elapsed,
        "steps_per_second": expected_steps / elapsed,
        "seconds_per_step": elapsed / expected_steps,
        "final_train_loss": result.train_loss[-1],
        "final_validation_loss": result.validation_loss[-1],
    }


@app.local_entrypoint()
def main(
    gpu: str = "A100-40GB",
    cpu: float = 8.0,
    memory: int = 32768,
    timeout: int = 3600,
    epochs: int = 500,
    batch_size: int = 8192,
    hidden_dim: int = 2048,
    n_layers: int = 10,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    seed: int = 42,
    train_pct: float = 80.0,
    val_pct: float = 10.0,
    test_pct: float = 10.0,
) -> None:
    """Parse CLI options, provision Modal resources, and run calibration."""
    remote_function = run_calibration.with_options(
        gpu=gpu,
        cpu=cpu,
        memory=memory,
        timeout=timeout,
    )
    result = remote_function.remote(
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        paper_dir="/tmp/rfm-calibration",
    )
    print(result)
