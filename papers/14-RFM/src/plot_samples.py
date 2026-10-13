"""Plot observed Fire locations beside samples from a trained RFM run."""

import argparse
from pathlib import Path

import data as d
import evaluate
import geometry
import matplotlib
import model as m
import torch
from torch import nn

from gdl import load_checkpoint, load_run_config, run_dir

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parent.parent


def load_trained_model(
    run_id: str,
    *,
    paper_dir: str | Path,
    runs_base: str = "runs",
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Reconstruct a trained vector field from one persisted run."""
    config = load_run_config(paper_dir, run_id, base=runs_base)
    model_config = config["model"]
    trained_model = m.TimeConditionedVectorField(
        hidden_dim=int(model_config["hidden_dim"]),
        n_layers=int(model_config["n_layers"]),
    )
    load_checkpoint(
        trained_model,
        run_dir(paper_dir, run_id, base=runs_base),
    )
    trained_model = trained_model.to(torch.device(device))
    trained_model.eval()
    return trained_model


def generate_samples(
    model: nn.Module,
    *,
    n_samples: int,
    batch_size: int,
    steps: int,
    device: str | torch.device,
    seed: int,
) -> torch.Tensor:
    """Generate model samples in bounded batches and return them on CPU."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")

    resolved_device = torch.device(device)
    torch.manual_seed(seed)
    model = model.to(resolved_device)
    model.eval()

    generated_batches = []
    for start in range(0, n_samples, batch_size):
        current_batch_size = min(batch_size, n_samples - start)
        initial_state = geometry.sample_sphere(
            current_batch_size,
            device=resolved_device,
        )
        generated = evaluate.sample(
            model,
            initial_state,
            steps=steps,
            device=resolved_device,
        )
        generated_batches.append(generated.cpu())
    return torch.cat(generated_batches, dim=0)


def save_comparison_plot(
    observed: torch.Tensor,
    generated: torch.Tensor,
    output_path: str | Path,
) -> None:
    """Save side-by-side longitude/latitude scatter plots."""
    observed_latlon = d.cartesian_to_latlon_degrees(observed).detach().cpu()
    generated_latlon = d.cartesian_to_latlon_degrees(generated).detach().cpu()

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    panels = (
        (axes[0], observed_latlon, f"Observed Fire data (n={len(observed_latlon):,})"),
        (
            axes[1],
            generated_latlon,
            f"RFM generated samples (n={len(generated_latlon):,})",
        ),
    )
    for axis, latlon, title in panels:
        axis.scatter(latlon[:, 1], latlon[:, 0], s=5, alpha=0.45)
        axis.set_title(title)
        axis.set_xlabel("Longitude (degrees)")
        axis.set_xlim(-180, 180)
        axis.set_ylim(-90, 90)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Latitude (degrees)")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def main() -> None:
    """Generate samples from a saved run and write the comparison plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run_id under runs/")
    parser.add_argument("--runs-base", default="runs")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PAPER_DIR / "data" / "fire.csv",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    observed = d.load_fire_csv(args.data_path)
    trained_model = load_trained_model(
        args.run,
        paper_dir=PAPER_DIR,
        runs_base=args.runs_base,
        device=args.device,
    )
    generated = generate_samples(
        trained_model,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
    )
    output_path = args.output or (
        run_dir(PAPER_DIR, args.run, base=args.runs_base) / "fire-comparison.png"
    )
    save_comparison_plot(observed, generated, output_path)
    print(f"saved comparison plot: {output_path}")


if __name__ == "__main__":
    main()
