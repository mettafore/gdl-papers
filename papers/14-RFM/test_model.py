"""AI-owned contracts for the time-conditioned S2 vector-field scaffold."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.func import jacrev

sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import TimeConditionedVectorField


def _sphere_batch(n: int = 4) -> torch.Tensor:
    raw = torch.arange(1, n * 3 + 1, dtype=torch.float32).reshape(n, 3)
    return raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True)


def test_model_construction_and_forward_contract() -> None:
    model = TimeConditionedVectorField(hidden_dim=16)

    assert isinstance(model, nn.Module)
    output = model(_sphere_batch(), torch.zeros(4))

    assert output.shape == (4, 3)


def test_model_output_is_tangent_to_each_input_point() -> None:
    x = _sphere_batch()
    t = torch.linspace(0.0, 1.0, len(x))
    output = TimeConditionedVectorField(hidden_dim=16)(x, t)

    radial_component = (output * x).sum(dim=-1)
    assert torch.allclose(
        radial_component, torch.zeros_like(radial_component), atol=1e-6
    )


def test_model_is_invariant_to_radial_input_scaling() -> None:
    torch.manual_seed(0)
    model = TimeConditionedVectorField(hidden_dim=16)
    x = _sphere_batch()
    t = torch.linspace(0.0, 1.0, len(x))

    expected = model(x, t)
    radially_scaled = model(3.0 * x, t)

    assert torch.allclose(radially_scaled, expected, atol=1e-6)


def test_model_jacobian_has_zero_radial_derivative() -> None:
    torch.manual_seed(0)
    model = TimeConditionedVectorField(hidden_dim=16)
    x = _sphere_batch(n=1).squeeze(0)
    t = torch.tensor(0.5)

    def field(point: torch.Tensor) -> torch.Tensor:
        return model(point.unsqueeze(0), t.unsqueeze(0)).squeeze(0)

    jacobian = jacrev(field)(x)

    assert torch.allclose(jacobian @ x, torch.zeros_like(x), atol=1e-6)


@pytest.mark.parametrize(
    ("x", "t"),
    [
        (torch.zeros(3), torch.zeros(3)),
        (torch.zeros(2, 2), torch.zeros(2)),
        (torch.zeros(2, 3), torch.zeros(1)),
    ],
)
def test_model_rejects_invalid_input_shapes(x: torch.Tensor, t: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        TimeConditionedVectorField()(x, t)
