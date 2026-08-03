import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch_geometric")

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from train import raw_target_for, target_for

from data import NormStats


def _fake_batch(y):
    return types.SimpleNamespace(y=y)


def test_target_for_selects_only_the_training_column():
    # 2 molecules, 19 QM9 columns each; column values equal the column index
    # so picking the wrong column is easy to detect.
    y = torch.arange(19).float().unsqueeze(0).repeat(2, 1)
    norm = NormStats(mean=0.0, mad=1.0)  # identity normalize

    target = target_for(_fake_batch(y), norm, col=4)

    assert torch.equal(target, torch.tensor([[4.0], [4.0]]))


def test_target_for_keeps_column_dim_for_loss_shape():
    y = torch.arange(19).float().unsqueeze(0).repeat(3, 1)
    norm = NormStats(mean=0.0, mad=1.0)

    target = target_for(_fake_batch(y), norm, col=0)

    assert target.shape == (3, 1)


def test_raw_target_for_selects_column_without_normalizing():
    # values chosen so a normalize-then-return would fail this check
    y = torch.arange(19).float().unsqueeze(0).repeat(2, 1)

    target = raw_target_for(_fake_batch(y), col=4)

    assert torch.equal(target, torch.tensor([[4.0], [4.0]]))


def test_raw_target_for_keeps_column_dim():
    y = torch.arange(19).float().unsqueeze(0).repeat(3, 1)

    target = raw_target_for(_fake_batch(y), col=0)

    assert target.shape == (3, 1)
