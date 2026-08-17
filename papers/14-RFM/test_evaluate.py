"""Tests for the RFM sampling and ODE integration contracts."""

import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent / "src"))

import evaluate as evaluate_module
from evaluate import (
    divergence,
    integrate_ode,
    negative_log_likelihood,
    sample,
)
from model import TimeConditionedVectorField

import data as data_module
from gdl import new_run_dir, save_checkpoint


def _initial_state() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )


def test_integrate_ode_matches_constant_velocity_with_fixed_steps() -> None:
    initial_state = _initial_state()

    def constant_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    result = integrate_ode(constant_velocity, initial_state, steps=4)

    assert torch.allclose(result, initial_state.detach() + 1.0)
    assert result.shape == initial_state.shape
    assert not result.requires_grad


def test_integrate_ode_queries_velocity_at_multiple_times() -> None:
    times: list[float] = []

    def recording_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        times.append(float(t))
        return torch.zeros_like(x)

    integrate_ode(recording_velocity, _initial_state(), steps=4)

    assert len(times) == 4
    assert times == pytest.approx([0.0, 0.25, 0.5, 0.75])


@pytest.mark.parametrize(
    ("t0", "t1", "steps"),
    [(1.0, 1.0, 4), (1.0, 0.0, 4), (0.0, 1.0, 0), (0.0, 1.0, True)],
)
def test_integrate_ode_rejects_invalid_solver_options(
    t0: float, t1: float, steps: int
) -> None:
    with pytest.raises(ValueError):
        integrate_ode(
            lambda x, t: torch.zeros_like(x),
            _initial_state(),
            t0=t0,
            t1=t1,
            steps=steps,
        )


class _RecordingConstantModel(nn.Module):
    """Return a constant velocity while recording queried times."""

    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.ones(()))
        self.times: list[float] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert t.shape == (x.shape[0],)
        self.times.append(float(t[0]))
        return torch.ones_like(x) * self.parameter


class _DiagonalLinearField(nn.Module):
    """A field whose divergence can be calculated by hand."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return torch.stack(
            (2.0 * x[:, 0], -3.0 * x[:, 1], 0.5 * x[:, 2]),
            dim=1,
        )


class _ZeroField(nn.Module):
    """A stationary flow with no change-of-variables correction."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return torch.zeros_like(x)


def test_divergence_returns_per_point_jacobian_trace() -> None:
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]],
        requires_grad=True,
    )
    t = torch.tensor([0.2, 0.8])

    result = divergence(_DiagonalLinearField(), x, t)

    assert result.shape == (2,)
    assert torch.allclose(result, torch.tensor([-0.5, -0.5]))


def test_sample_integrates_model_without_gradients() -> None:
    model = _RecordingConstantModel()
    initial_state = _initial_state()

    result = sample(model, initial_state, steps=4, device="cpu")

    assert torch.allclose(result, initial_state.detach() + 1.0)
    assert result.shape == initial_state.shape
    assert not result.requires_grad
    assert model.times == pytest.approx([0.0, 0.25, 0.5, 0.75])


def test_sample_adapts_scalar_solver_time_to_model_batch_time() -> None:
    model = TimeConditionedVectorField(hidden_dim=8, n_layers=1)

    result = sample(model, _initial_state().detach(), steps=2, device="cpu")

    assert result.shape == (2, 3)
    assert torch.isfinite(result).all()


def test_zero_field_nll_matches_uniform_sphere_baseline() -> None:
    result = negative_log_likelihood(
        _ZeroField(),
        _initial_state().detach(),
        rtol=1e-7,
        atol=1e-7,
    )

    assert isinstance(result, float)
    assert math.isclose(result, math.log(4.0 * math.pi), rel_tol=1e-5)


def test_nll_applies_reverse_flow_divergence_correction() -> None:
    result = negative_log_likelihood(
        _DiagonalLinearField(),
        _initial_state().detach(),
        rtol=1e-7,
        atol=1e-7,
    )

    assert math.isclose(
        result,
        math.log(4.0 * math.pi) - 0.5,
        rel_tol=1e-5,
    )


def test_evaluate_run_rebuilds_checkpoint_and_weights_unequal_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "fire.csv"
    rows = [
        f"{latitude},{longitude}"
        for latitude, longitude in zip(range(-15, 15), range(-150, 150, 10))
    ]
    data_path.write_text("LATITUDE,LONGITUDE\n" + "\n".join(rows) + "\n")
    config = {
        "data": {
            "dataset": "fire",
            "path": str(data_path.resolve()),
            "split": [0.8, 0.1, 0.1],
        },
        "model": {"hidden_dim": 8, "n_layers": 1},
        "hyperparams": {"lr": 1e-3, "epochs": 1},
        "seed": 7,
        "metric": "nll",
    }
    run_id, run_dir = new_run_dir(tmp_path, config, base="runs")
    saved_model = TimeConditionedVectorField(hidden_dim=8, n_layers=1)
    with torch.no_grad():
        for parameter in saved_model.parameters():
            parameter.fill_(0.25)
    save_checkpoint(saved_model, run_dir)

    observed_batches: list[torch.Tensor] = []

    def batch_mean_x(
        model: nn.Module,
        samples: torch.Tensor,
        *,
        rtol: float,
        atol: float,
    ) -> float:
        del rtol, atol
        assert all(
            torch.allclose(parameter, torch.full_like(parameter, 0.25))
            for parameter in model.parameters()
        )
        observed_batches.append(samples.detach().cpu())
        return float(samples[:, 0].mean())

    monkeypatch.setattr(evaluate_module, "negative_log_likelihood", batch_mean_x)

    metric, score = evaluate_module.evaluate_run(
        run_id,
        runs_base="runs",
        paper_dir=tmp_path,
        batch_size=2,
        device="cpu",
    )

    expected_test = data_module.split_points(
        data_module.load_fire_csv(data_path), seed=7
    )[2]
    assert metric == "nll"
    assert [len(batch) for batch in observed_batches] == [2, 1]
    assert torch.equal(torch.cat(observed_batches), expected_test)
    assert score == pytest.approx(float(expected_test[:, 0].mean()))


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_evaluate_run_rejects_invalid_batch_size(
    batch_size: int,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        evaluate_module.evaluate_run(
            "unused",
            paper_dir=tmp_path,
            batch_size=batch_size,
        )
