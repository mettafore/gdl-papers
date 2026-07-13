"""Scaffold tests for EGNN model pieces, filled in alongside src/model.py."""

import torch

from src.model import unsorted_segment_sum


def test_unsorted_segment_sum_known_values():
    data = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    segment_ids = torch.tensor([0, 1, 0])
    out = unsorted_segment_sum(data, segment_ids, num_segments=2)
    expected = torch.tensor([[4.0, 4.0], [2.0, 2.0]])
    assert torch.equal(out, expected)


def test_unsorted_segment_sum_shape():
    data = torch.randn(7, 5)
    segment_ids = torch.randint(0, 3, (7,))
    out = unsorted_segment_sum(data, segment_ids, num_segments=3)
    assert out.shape == (3, 5)


def test_unsorted_segment_sum_empty_segment_is_zero():
    data = torch.tensor([[1.0], [2.0]])
    segment_ids = torch.tensor([0, 0])
    out = unsorted_segment_sum(data, segment_ids, num_segments=3)
    assert torch.equal(out[2], torch.zeros(1))


def test_unsorted_segment_sum_dtype_device_match_data():
    data = torch.randn(4, 2, dtype=torch.float64)
    segment_ids = torch.tensor([0, 0, 1, 1])
    out = unsorted_segment_sum(data, segment_ids, num_segments=2)
    assert out.dtype == data.dtype
    assert out.device == data.device
