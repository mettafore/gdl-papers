"""Run one real-data MACE training batch and checkpoint reload on Modal CUDA.

From the repository root:
    uv run modal run papers/15-MACE/modal_smoke.py
"""

import json
from pathlib import Path

import modal

REMOTE_ROOT = "/root/gdl-papers"
PAPER_SRC = f"{REMOTE_ROOT}/papers/15-MACE/src"
DATA_DIR = f"{REMOTE_ROOT}/papers/15-MACE/data/dataset_3BPA"
SMOKE_RUNS = "/tmp/mace-modal-smoke"

app = modal.App("gdl-papers-mace-smoke")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["dev"])
    .add_local_file(
        "pyproject.toml", remote_path=f"{REMOTE_ROOT}/pyproject.toml", copy=True
    )
    .add_local_file("README.md", remote_path=f"{REMOTE_ROOT}/README.md", copy=True)
    .add_local_dir("src/gdl", remote_path=f"{REMOTE_ROOT}/src/gdl", copy=True)
    .add_local_dir(
        "papers/15-MACE/src",
        remote_path=PAPER_SRC,
        copy=True,
    )
    .add_local_dir(
        "papers/15-MACE/data/dataset_3BPA",
        remote_path=DATA_DIR,
        copy=True,
    )
    .run_commands(f"cd {REMOTE_ROOT} && uv pip install --no-deps -e .")
)


@app.function(image=image, gpu="T4", cpu=4, timeout=60 * 30)
def run_smoke() -> dict[str, float | int | str | bool]:
    """Train one batch, validate, reload the checkpoint, and evaluate one frame."""
    import sys

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal allocated no CUDA device")

    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    sys.path.insert(0, PAPER_SRC)

    from evaluate import _evaluate_graphs
    from model import MACE, MACEConfig
    from train import TrainConfig, train

    import data
    from gdl import load_checkpoint, load_run_config, run_dir

    data_path = Path(DATA_DIR)
    output_dir = Path(SMOKE_RUNS)
    run_id, written_run_dir = train(
        data_dir=data_path,
        paper_dir=output_dir,
        train_config=TrainConfig(
            batch_size=5,
            max_epochs=1,
            max_train_batches=1,
            seed=123,
        ),
        device="cuda",
    )

    saved_config = load_run_config(output_dir, run_id)
    data_config = saved_config["data"]
    model_values = dict(saved_config["model"])
    model_values["radial_mlp_widths"] = tuple(model_values["radial_mlp_widths"])
    model = (
        MACE(
            MACEConfig(**model_values),
            avg_num_neighbours=data_config["avg_num_neighbors"],
            atomic_energies=torch.tensor(
                data_config["atomic_energies"], dtype=torch.float64
            ),
            mean_energy_shift=data_config["mean_energy_shift"],
            rms_forces=data_config["rms_forces"],
        )
        .double()
        .to("cuda")
    )
    load_checkpoint(model, run_dir(output_dir, run_id))

    one_frame = data.load_extxyz(data_path / data.TRAIN_FILE)[:1]
    one_graph = [data.build_neighbor_graph(one_frame[0])]
    single_graph_metrics = _evaluate_graphs(
        model,
        one_graph,
        batch_size=1,
        device="cuda",
    )

    with (written_run_dir / "metrics.jsonl").open() as metrics_file:
        final_epoch = json.loads(metrics_file.readlines()[-1])

    return {
        "device": str(next(model.parameters()).device),
        "gpu_name": torch.cuda.get_device_name(),
        "train_batches": 1,
        "validation_loss": final_epoch["validation_loss"],
        "checkpoint_reloaded": True,
        "single_graph_energy_rmse_mev": single_graph_metrics["energy"],
        "single_graph_force_rmse_mev_per_angstrom": single_graph_metrics["forces"],
    }


@app.local_entrypoint()
def main() -> None:
    result = run_smoke.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
