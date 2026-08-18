"""Tests for the lightweight RFM training harness."""

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

import train as train_module
from geometry import sample_sphere
from loss import rcfm_loss
from model import TimeConditionedVectorField
from train import train


class _GradRecordingModel(TimeConditionedVectorField):
    """Record whether each forward pass runs with autograd enabled."""

    def __init__(self) -> None:
        super().__init__(hidden_dim=8, n_layers=1)
        self.grad_enabled: list[bool] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.grad_enabled.append(torch.is_grad_enabled())
        return super().forward(x, t)


def test_train_rejects_iterable_data_source_with_clear_error() -> None:
    model = TimeConditionedVectorField(hidden_dim=8, n_layers=1)

    with pytest.raises(TypeError, match="torch.Tensor"):
        train(
            [torch.zeros(3)],  # type: ignore[arg-type]
            sample_sphere(4),
            model,
            rcfm_loss,
        )


@pytest.mark.parametrize(
    "bad_data",
    [
        torch.empty((0, 3)),
        torch.tensor([[float("nan"), 0.0, 0.0]]),
    ],
)
def test_train_rejects_empty_or_nonfinite_data(bad_data: torch.Tensor) -> None:
    model = TimeConditionedVectorField(hidden_dim=8, n_layers=1)

    with pytest.raises(ValueError):
        train(bad_data, sample_sphere(4), model, rcfm_loss)


@pytest.mark.parametrize("epochs", [0, -1, True])
def test_train_rejects_invalid_epoch_count(epochs: int) -> None:
    model = TimeConditionedVectorField(hidden_dim=8, n_layers=1)

    with pytest.raises(ValueError, match="positive integer"):
        train(sample_sphere(4), sample_sphere(4), model, rcfm_loss, epochs=epochs)


def test_train_returns_finite_loss_per_epoch() -> None:
    train_data = sample_sphere(8)
    validation_data = sample_sphere(4)
    model = TimeConditionedVectorField(hidden_dim=8, n_layers=1)

    result = train(
        train_data,
        validation_data,
        model,
        rcfm_loss,
        epochs=3,
        seed=7,
        device="cpu",
    )

    assert result.epochs == 3
    assert len(result.train_loss) == 3
    assert len(result.validation_loss) == 3
    assert all(isinstance(value, float) for value in result.train_loss)
    assert all(isinstance(value, float) for value in result.validation_loss)
    assert all(math.isfinite(value) for value in result.train_loss)
    assert all(math.isfinite(value) for value in result.validation_loss)


def test_train_updates_model_parameters() -> None:
    model = TimeConditionedVectorField(hidden_dim=8, n_layers=1)
    parameters_before = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )

    result = train(
        sample_sphere(8),
        sample_sphere(4),
        model,
        rcfm_loss,
        optimizer_settings={"lr": 1e-2},
        epochs=1,
        seed=7,
        device="cpu",
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, result.model.parameters())
    )


def test_train_keeps_returned_model_on_resolved_device() -> None:
    result = train(
        sample_sphere(8),
        sample_sphere(4),
        TimeConditionedVectorField(hidden_dim=8, n_layers=1),
        rcfm_loss,
        epochs=1,
        seed=7,
        device="cpu",
    )

    assert result.device == torch.device("cpu")
    assert all(
        parameter.device == result.device for parameter in result.model.parameters()
    )


def test_validation_forward_disables_gradients() -> None:
    model = _GradRecordingModel()

    train(
        sample_sphere(8),
        sample_sphere(4),
        model,
        rcfm_loss,
        epochs=1,
        seed=7,
        device="cpu",
    )

    assert model.grad_enabled == [True, False]


def test_train_is_reproducible_when_model_initialization_is_repeated() -> None:
    train_data = sample_sphere(8)
    validation_data = sample_sphere(4)

    torch.manual_seed(11)
    first_model = TimeConditionedVectorField(hidden_dim=8, n_layers=1)
    first = train(
        train_data,
        validation_data,
        first_model,
        rcfm_loss,
        epochs=2,
        seed=42,
        device="cpu",
    )

    torch.manual_seed(11)
    second_model = TimeConditionedVectorField(hidden_dim=8, n_layers=1)
    second = train(
        train_data,
        validation_data,
        second_model,
        rcfm_loss,
        epochs=2,
        seed=42,
        device="cpu",
    )

    assert first.train_loss == second.train_loss
    assert first.validation_loss == second.validation_loss


def test_train_keeps_losses_finite_near_endpoint_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_data = torch.tensor([[0.0, 1.0, 0.0]]).repeat(4, 1)
    base_points = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)

    def near_endpoint_rand(
        size: int, *, device: torch.device | None = None
    ) -> torch.Tensor:
        return torch.full((size,), 0.9999, device=device)

    monkeypatch.setattr(train_module.geom, "sample_sphere", lambda n: base_points[:n])
    monkeypatch.setattr(train_module.torch, "rand", near_endpoint_rand)

    result = train(
        train_data,
        train_data,
        TimeConditionedVectorField(hidden_dim=8, n_layers=1),
        rcfm_loss,
        epochs=1,
        seed=7,
        device="cpu",
    )

    assert all(math.isfinite(value) for value in result.train_loss)
    assert all(math.isfinite(value) for value in result.validation_loss)


def test_train_run_persists_config_metrics_and_checkpoint(tmp_path: Path) -> None:
    data_path = tmp_path / "fire.csv"
    data_path.write_text(
        "LATITUDE,LONGITUDE\n"
        "0,0\n10,20\n-10,-20\n20,40\n-20,-40\n"
        "30,60\n-30,-60\n40,80\n-40,-80\n50,100\n"
    )

    run_id, run_dir, result = train_module.train_run(
        data_path=data_path,
        epochs=2,
        lr=1e-3,
        hidden_dim=8,
        n_layers=1,
        seed=7,
        train_pct=70,
        val_pct=20,
        test_pct=10,
        device="cpu",
        runs_base="runs",
        paper_dir=tmp_path,
    )

    assert run_dir == tmp_path / "runs" / run_id
    config = json.loads((run_dir / "config.json").read_text())
    assert config["data"] == {
        "dataset": "fire",
        "path": str(data_path.resolve()),
        "split": [70, 20, 10],
    }
    assert config["model"] == {"hidden_dim": 8, "n_layers": 1}
    assert config["hyperparams"] == {"lr": 1e-3, "epochs": 2}
    assert config["seed"] == 7
    assert config["metric"] == "nll"

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics == [
        {
            "step": epoch,
            "train_loss": result.train_loss[epoch],
            "validation_loss": result.validation_loss[epoch],
        }
        for epoch in range(2)
    ]
    assert (run_dir / "checkpoint.pt").is_file()


def test_train_run_rejects_an_empty_partition_before_creating_a_run(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "fire.csv"
    data_path.write_text(
        "LATITUDE,LONGITUDE\n"
        "0,0\n10,20\n-10,-20\n20,40\n-20,-40\n"
        "30,60\n-30,-60\n40,80\n-40,-80\n50,100\n"
    )

    with pytest.raises(ValueError, match="non-empty"):
        train_module.train_run(
            data_path=data_path,
            train_pct=100,
            val_pct=0,
            test_pct=0,
            runs_base="runs",
            paper_dir=tmp_path,
        )

    assert not (tmp_path / "runs").exists()
