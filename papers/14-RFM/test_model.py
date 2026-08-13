"""AI-owned contracts for the time-conditioned S2 vector-field scaffold."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import TimeConditionedVectorField


def _sphere_batch(n: int = 4) -> torch.Tensor:
    raw = torch.arange(1, n * 3 + 1, dtype=torch.float32).reshape(n, 3)
    return raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True)


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="ambient field and tangent projection are not implemented yet",
)
def test_model_construction_and_forward_contract() -> None:
    model = TimeConditionedVectorField(hidden_dim=16)

    assert isinstance(model, nn.Module)
    output = model(_sphere_batch(), torch.zeros(4))

    assert output.shape == (4, 3)


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="ambient field and tangent projection are not implemented yet",
)
def test_model_output_is_tangent_to_each_input_point() -> None:
    x = _sphere_batch()
    t = torch.linspace(0.0, 1.0, len(x))
    output = TimeConditionedVectorField(hidden_dim=16)(x, t)

    radial_component = (output * x).sum(dim=-1)
    assert torch.allclose(radial_component, torch.zeros_like(radial_component))


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
