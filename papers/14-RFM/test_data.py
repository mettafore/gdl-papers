"""Contracts for the Fire data pipeline."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import latlon_degrees_to_cartesian, load_fire_csv, split_points


def test_conversion_reports_shape_dtype_and_unit_norm() -> None:
    latlon = torch.tensor(
        [[12.5, -45.0], [-33.25, 151.2], [0.0, 0.0]], dtype=torch.float64
    )

    points = latlon_degrees_to_cartesian(latlon)

    assert points.shape == (3, 3)
    assert points.dtype == torch.float32
    assert torch.allclose(torch.linalg.vector_norm(points, dim=-1), torch.ones(3))


@pytest.mark.parametrize(
    ("latlon", "expected"),
    [
        ([0.0, 0.0], [1.0, 0.0, 0.0]),
        ([90.0, 0.0], [0.0, 0.0, 1.0]),
    ],
)
def test_conversion_known_landmarks(latlon: list[float], expected: list[float]) -> None:
    points = latlon_degrees_to_cartesian(torch.tensor([latlon]))
    assert torch.allclose(points[0], torch.tensor(expected), atol=1e-6)


@pytest.mark.parametrize(
    "latlon",
    [
        torch.zeros(3),
        torch.zeros(2, 3),
        torch.tensor([[float("nan"), 0.0]]),
        torch.tensor([[0.0, float("inf")]]),
        torch.tensor([[90.01, 0.0]]),
        torch.tensor([[-90.01, 0.0]]),
        torch.tensor([[0.0, 180.01]]),
        torch.tensor([[0.0, -180.01]]),
    ],
)
def test_conversion_rejects_invalid_coordinates(latlon: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        latlon_degrees_to_cartesian(latlon)


def test_load_fire_csv_matches_direct_conversion(tmp_path: Path) -> None:
    csv_path = tmp_path / "fire.csv"
    csv_path.write_text("latitude,longitude\n0,0\n90,0\n", encoding="utf-8")
    expected = latlon_degrees_to_cartesian(torch.tensor([[0.0, 0.0], [90.0, 0.0]]))

    points = load_fire_csv(csv_path)

    assert torch.equal(points, expected)


def _unique_points(n: int = 101) -> torch.Tensor:
    return torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)


def test_split_sizes_follow_80_10_10_protocol() -> None:
    train, validation, test = split_points(_unique_points(), seed=0)
    assert (len(train), len(validation), len(test)) == (81, 10, 10)


def test_split_is_seeded_and_deterministic() -> None:
    points = _unique_points()
    first = split_points(points, seed=7)
    repeated = split_points(points, seed=7)
    changed = split_points(points, seed=8)

    assert all(torch.equal(a, b) for a, b in zip(first, repeated, strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(first, changed, strict=True))


def test_split_uses_every_row_exactly_once() -> None:
    points = _unique_points()
    partitions = split_points(points, seed=0)
    observed = torch.cat(partitions, dim=0)

    expected_rows = {tuple(row.tolist()) for row in points}
    observed_rows = {tuple(row.tolist()) for row in observed}
    assert observed_rows == expected_rows
    assert len(observed) == len(observed_rows)
