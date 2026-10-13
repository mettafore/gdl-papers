"""Run an author-scale Fire calibration on a Modal GPU."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

import modal

REMOTE_ROOT = "/root/rfm"
RESULTS_ROOT = "/root/rfm-results"

if modal.is_local():
    repo_root = Path(__file__).resolve().parents[2]
    paper_dir = Path(__file__).resolve().parent
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "torch==2.14.0",
            "numpy<2",
            "pandas>=2.3.3",
            "torchdiffeq>=0.2.5",
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
results_volume = modal.Volume.from_name(
    "rfm-fire-results",
    create_if_missing=True,
)


@app.function(
    image=image,
    volumes={RESULTS_ROOT: results_volume},
)
def run_calibration(
    *,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    n_layers: int,
    lr: float,
    weight_decay: float,
    eval_batch_size: int,
    rtol: float,
    atol: float,
    seed: int,
    train_pct: float,
    val_pct: float,
    test_pct: float,
    paper_dir: str,
) -> dict[str, float | int | str]:
    """Train the Fire model and return timing, loss, and test NLL metrics."""
    sys.path.insert(0, f"{REMOTE_ROOT}/paper_src")
    sys.path.insert(0, REMOTE_ROOT)

    import data as d
    from evaluate import evaluate_run
    from train import train_run

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
    metric_name, test_nll = evaluate_run(
        run_id,
        paper_dir=paper_dir,
        data_path=data_path,
        batch_size=eval_batch_size,
        device="cuda",
        rtol=rtol,
        atol=atol,
    )

    summary = {
        "run_id": run_id,
        "steps": expected_steps,
        "seconds": elapsed,
        "steps_per_second": expected_steps / elapsed,
        "seconds_per_step": elapsed / expected_steps,
        "final_train_loss": result.train_loss[-1],
        "final_validation_loss": result.validation_loss[-1],
        metric_name: test_nll,
    }
    summary_path = Path(paper_dir) / "latest_result.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    results_volume.commit()
    return summary


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
    eval_batch_size: int = 64,
    rtol: float = 1e-7,
    atol: float = 1e-7,
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
    function_call = remote_function.spawn(
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        lr=lr,
        weight_decay=weight_decay,
        eval_batch_size=eval_batch_size,
        rtol=rtol,
        atol=atol,
        seed=seed,
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        paper_dir=f"{RESULTS_ROOT}/calibration",
    )
    print(f"function_call_id: {function_call.object_id}")
