"""Tests for the RFM sampling and ODE integration contracts."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluate import integrate_ode, sample
from model import TimeConditionedVectorField


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
