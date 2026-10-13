"""Tests for the closed-form sphere geometry used by RFM."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from geometry import conditional_vf, geodesic_path, grad_d, log_map, sample_sphere


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_coincident_points_have_zero_log_and_conditional_velocity(dtype) -> None:
    points = torch.nn.functional.normalize(
        torch.tensor([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]], dtype=dtype), dim=-1
    )
    time = torch.tensor([0.25, 0.98], dtype=dtype)
    for result in (
        log_map(points, points),
        grad_d(points, points),
        conditional_vf(points, points, time),
    ):
        torch.testing.assert_close(result, torch.zeros_like(points), atol=0, rtol=0)


@pytest.mark.parametrize("operation", [log_map, grad_d, geodesic_path, conditional_vf])
def test_antipodal_points_report_nonunique_geodesic(operation) -> None:
    point = torch.tensor([[1.0, 0.0, 0.0]])
    args = (point, -point)
    if operation in (geodesic_path, conditional_vf):
        args += (torch.tensor([0.5]),)
    with pytest.raises(ValueError, match="antipodal"):
        operation(*args)


def test_near_antipodal_points_reach_endpoint_and_stay_finite() -> None:
    start = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    angle = torch.tensor(torch.pi - 1e-8, dtype=torch.float64)
    end = torch.stack([angle.cos(), angle.sin(), angle.new_zeros(())]).reshape(1, 3)
    torch.testing.assert_close(
        geodesic_path(start, end, torch.ones(1, dtype=torch.float64)),
        end,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.isfinite(
        conditional_vf(start, end, torch.tensor([0.5], dtype=torch.float64))
    ).all()


def test_coincident_conditional_velocity_has_tangent_derivative() -> None:
    point = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    time = torch.tensor([0.5], dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(
        lambda x: conditional_vf(x, point, time), point
    ).reshape(3, 3)
    torch.testing.assert_close(
        jacobian, torch.diag(torch.tensor([0.0, -2.0, -2.0], dtype=torch.float64))
    )


def test_sub_epsilon_angle_preserves_logarithmic_displacement() -> None:
    angle = torch.tensor(1e-14, dtype=torch.float64)
    start = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    end = torch.stack([angle.cos(), angle.sin(), angle.new_zeros(())]).reshape(1, 3)
    expected = torch.tensor([[0.0, 1e-14, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(log_map(start, end), expected, atol=1e-28, rtol=1e-12)


def test_sample_sphere_respects_requested_device() -> None:
    points = sample_sphere(4, device="cpu")

    assert points.device == torch.device("cpu")


def _batch_geodesics(
    x0: torch.Tensor, x1: torch.Tensor, times: torch.Tensor
) -> torch.Tensor:
    """Return paths with shape ``(n_times, n_points, 3)``."""
    paths = [geodesic_path(x0, x1, t.expand(x0.shape[0])) for t in times]
    return torch.stack(paths)


def test_conditional_vf_matches_path_velocity() -> None:
    """Eq. 13 agrees with the centered derivative of the path."""
    torch.manual_seed(0)
    x0 = sample_sphere(16)
    x1 = sample_sphere(16)
    eps = 1e-3
    tolerance = 1e-3
    times = torch.tensor([0.25, 0.5, 0.75])

    paths = _batch_geodesics(x0, x1, times)
    paths_plus = _batch_geodesics(x0, x1, times + eps)
    paths_minus = _batch_geodesics(x0, x1, times - eps)
    finite_velocity = (paths_plus - paths_minus) / (2 * eps)

    conditional_velocity = torch.stack(
        [
            conditional_vf(paths[i], x1, times[i].expand(x0.shape[0]))
            for i in range(len(times))
        ]
    )

    error = torch.linalg.vector_norm(finite_velocity - conditional_velocity, dim=-1)
    assert torch.all(error <= tolerance)


def test_conditional_vf_stays_finite_for_small_angles() -> None:
    """The field remains finite when the path is close to its endpoint."""
    angle = torch.tensor([0.01])
    x0 = torch.cat(
        [torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)]
    ).reshape(1, 3)
    x1 = torch.tensor([[1.0, 0.0, 0.0]])
    t = torch.tensor([0.98])

    xt = geodesic_path(x0, x1, t)
    field = conditional_vf(xt, x1, t)

    assert torch.isfinite(field).all()


def test_euclidean_conditional_field_is_straight_velocity() -> None:
    """Eq. 13 reduces to ``x1 - x0`` under the flat Euclidean metric."""
    torch.manual_seed(1)
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    times = torch.tensor([0.25, 0.5, 0.75])

    paths = torch.stack([(1 - t) * x0 + t * x1 for t in times])
    residual = paths - x1.unsqueeze(0)
    distance = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
    gradient = residual / distance
    log_schedule_derivative = -1 / (1 - times).reshape(-1, 1, 1)
    conditional_velocity = log_schedule_derivative * distance * gradient

    expected_velocity = (x1 - x0).unsqueeze(0)
    assert torch.allclose(
        conditional_velocity,
        expected_velocity.expand_as(conditional_velocity),
        atol=1e-6,
    )
