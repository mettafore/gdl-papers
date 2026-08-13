"""AI-owned contract tests for the RCFM loss scaffold."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loss import rcfm_loss


def _placeholder_batch(n: int) -> torch.Tensor:
    """Return a placeholder velocity or conditional-target batch."""
    return torch.zeros((n, 3), dtype=torch.float32)


def test_rcfm_loss_rejects_mismatched_shapes() -> None:
    predicted_velocity = _placeholder_batch(4)
    conditional_target = _placeholder_batch(3)

    with pytest.raises(ValueError, match="same shape"):
        rcfm_loss(predicted_velocity, conditional_target)


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="RCFM loss body is not implemented yet",
)
def test_rcfm_loss_returns_one_scalar_tensor() -> None:
    predicted_velocity = _placeholder_batch(4)
    conditional_target = _placeholder_batch(4)

    loss = rcfm_loss(predicted_velocity, conditional_target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="RCFM loss body is not implemented yet",
)
def test_rcfm_loss_accepts_predicted_velocity_and_conditional_target() -> None:
    predicted_velocity = _placeholder_batch(4)
    conditional_target = torch.ones((4, 3), dtype=torch.float32)

    loss = rcfm_loss(predicted_velocity, conditional_target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
