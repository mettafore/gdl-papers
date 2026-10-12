"""Crash-resume contract for MACE training."""

import json
import sys
from functools import partial
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

import model
import train

import data


def test_retry_before_first_checkpoint_keeps_run_id(
    tmp_path: Path, monkeypatch
) -> None:
    records = _synthetic_records()
    monkeypatch.setattr(data, "load_extxyz", lambda _: records)
    monkeypatch.setattr(
        data,
        "split_train_valid",
        lambda configurations, *, seed: (configurations[:10], configurations[10:]),
    )
    monkeypatch.setattr(data, "load_isolated_atom_energies", lambda _: torch.zeros(4))
    run = partial(
        train.train,
        data_dir=tmp_path,
        paper_dir=tmp_path,
        device="cpu",
        train_config=train.TrainConfig(max_epochs=1, batch_size=5),
        model_config=model.MACEConfig(
            channels=2,
            radial_mlp_widths=(4, 4, 4, 8),
            readout_hidden_width=2,
        ),
        fresh_run_id="preempted-run",
    )
    with monkeypatch.context() as patch:

        def interrupt(*args, **kwargs):
            raise RuntimeError("simulated preemption")

        patch.setattr(model.MACE, "forward", interrupt)
        with pytest.raises(RuntimeError, match="simulated preemption"):
            run()
    assert not (tmp_path / "runs/preempted-run/training_state.pt").exists()
    run_id, run_path = run()
    assert run_id == "preempted-run"
    assert list((tmp_path / "runs").iterdir()) == [run_path]
    assert (
        torch.load(run_path / "training_state.pt", weights_only=True)[
            "completed_epochs"
        ]
        == 1
    )


def _synthetic_records() -> list[data.AtomicConfiguration]:
    return [
        data.AtomicConfiguration(
            atomic_numbers=torch.tensor([1, 6, 8], dtype=torch.long),
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0 + index * 0.02, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float64,
            ),
            energy=torch.tensor(-20.0 + index * 0.1, dtype=torch.float64),
            forces=torch.full((3, 3), 0.2, dtype=torch.float64),
        )
        for index in range(11)
    ]


def test_resume_matches_uninterrupted_epoch_boundary_run(
    tmp_path: Path, monkeypatch
) -> None:
    records = _synthetic_records()
    monkeypatch.setattr(data, "load_extxyz", lambda _: records)
    monkeypatch.setattr(
        data,
        "split_train_valid",
        lambda configurations, *, seed: (configurations[:10], configurations[10:]),
    )
    monkeypatch.setattr(
        data,
        "load_isolated_atom_energies",
        lambda _: torch.zeros(4, dtype=torch.float64),
    )

    small_model = model.MACEConfig(
        channels=2,
        radial_mlp_widths=(4, 4, 4, 8),
        readout_hidden_width=2,
    )
    full_id, full_run = train.train(
        data_dir=tmp_path,
        paper_dir=tmp_path / "uninterrupted",
        device="cpu",
        train_config=train.TrainConfig(max_epochs=2, batch_size=5),
        model_config=small_model,
    )
    staged_id, staged_run = train.train(
        data_dir=tmp_path,
        paper_dir=tmp_path / "resumed",
        device="cpu",
        train_config=train.TrainConfig(max_epochs=1, batch_size=5),
        model_config=small_model,
        fresh_run_id="stable-modal-run",
    )
    assert staged_id == "stable-modal-run"
    staged_state = torch.load(staged_run / "training_state.pt", weights_only=True)
    assert staged_state["best_ema_model"]
    torch.save({"interrupted": torch.tensor(1)}, staged_run / "checkpoint.pt")
    with (staged_run / "metrics.jsonl").open("a") as metrics_file:
        metrics_file.write(json.dumps({"step": 1, "validation_loss": -1.0}) + "\n")

    # Resume at the saved limit first: repair files without running another epoch.
    train.train(
        data_dir=tmp_path,
        paper_dir=tmp_path / "resumed",
        device="cpu",
        train_config=train.TrainConfig(max_epochs=1, batch_size=5),
        model_config=small_model,
        resume_run_id=staged_id,
    )
    repaired_checkpoint = torch.load(staged_run / "checkpoint.pt", weights_only=True)
    assert all(
        torch.equal(repaired_checkpoint[key], value)
        for key, value in staged_state["best_ema_model"].items()
    )
    assert [
        json.loads(line)["step"]
        for line in (staged_run / "metrics.jsonl").read_text().splitlines()
    ] == [0]

    durable_epoch_callbacks = []
    resumed_id, resumed_run = train.train(
        data_dir=tmp_path,
        paper_dir=tmp_path / "resumed",
        device="cpu",
        train_config=train.TrainConfig(max_epochs=2, batch_size=5),
        model_config=small_model,
        fresh_run_id=staged_id,
        on_epoch_checkpoint=lambda: durable_epoch_callbacks.append(True),
    )

    full_state = torch.load(full_run / "training_state.pt", weights_only=True)
    resumed_state = torch.load(resumed_run / "training_state.pt", weights_only=True)
    assert resumed_id == staged_id
    assert full_id != resumed_id
    assert (staged_run / "checkpoint.pt").is_file()
    assert resumed_state["completed_epochs"] == 2
    assert full_state["completed_epochs"] == 2
    assert durable_epoch_callbacks == [True]
    for state_name in ("model", "ema_model", "best_ema_model"):
        assert full_state[state_name].keys() == resumed_state[state_name].keys()
        assert all(
            torch.equal(full_state[state_name][key], resumed_state[state_name][key])
            for key in full_state[state_name]
        )
    assert [
        json.loads(line)["step"]
        for line in (resumed_run / "metrics.jsonl").read_text().splitlines()
    ] == [0, 1]
    best_checkpoint = torch.load(resumed_run / "checkpoint.pt", weights_only=True)
    assert all(
        torch.equal(best_checkpoint[key], value)
        for key, value in resumed_state["best_ema_model"].items()
    )
