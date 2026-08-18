"""Tests for comparing observed and generated Fire locations."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import TimeConditionedVectorField
from plot_samples import generate_samples, load_trained_model, save_comparison_plot

from gdl import new_run_dir, save_checkpoint


class _RotationVelocityModel(nn.Module):
    """Return a nonzero tangent rotation around the sphere's z-axis."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return torch.stack((-x[:, 1], x[:, 0], torch.zeros_like(x[:, 2])), dim=1)


def test_load_trained_model_reconstructs_saved_run(tmp_path: Path) -> None:
    config = {
        "model": {"hidden_dim": 8, "n_layers": 1},
        "metric": "nll",
    }
    run_id, run_directory = new_run_dir(tmp_path, config)
    expected = TimeConditionedVectorField(hidden_dim=8, n_layers=1)
    with torch.no_grad():
        for parameter in expected.parameters():
            parameter.fill_(0.25)
    save_checkpoint(expected, run_directory)

    loaded = load_trained_model(run_id, paper_dir=tmp_path, device="cpu")

    assert not loaded.training
    assert all(
        torch.equal(actual, wanted)
        for actual, wanted in zip(
            loaded.parameters(), expected.parameters(), strict=True
        )
    )


def test_generate_samples_returns_unit_sphere_batch() -> None:
    generated = generate_samples(
        _RotationVelocityModel(),
        n_samples=7,
        batch_size=3,
        steps=4,
        device="cpu",
        seed=11,
    )

    assert generated.shape == (7, 3)
    assert torch.allclose(
        torch.linalg.vector_norm(generated, dim=1),
        torch.ones(7),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("n_samples", "batch_size"),
    [(1.5, 2), (2, 1.5), (True, 2), (2, True)],
)
def test_generate_samples_rejects_noninteger_sizes(
    n_samples: int,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        generate_samples(
            _RotationVelocityModel(),
            n_samples=n_samples,
            batch_size=batch_size,
            steps=2,
            device="cpu",
            seed=11,
        )


def test_save_comparison_plot_writes_nonempty_png(tmp_path: Path) -> None:
    observed = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    generated = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    output_path = tmp_path / "comparison.png"

    save_comparison_plot(observed, generated, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
