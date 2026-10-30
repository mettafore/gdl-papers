"""Plot the validation-loss learning curve of a MACE run.

Usage: uv run python papers/15-MACE/results/plot_learning_curve.py runs/<run_id>/metrics.jsonl
Writes results/learning-curve.png next to this script.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main(metrics_path: str) -> None:
    rows = [
        json.loads(line)
        for line in Path(metrics_path).read_text().splitlines()
        if line.strip()
    ]
    epochs = [
        r["step"] + 1 for r in rows
    ]  # metrics log 0-indexed steps; README uses 1-indexed epochs
    loss = [r["validation_loss"] for r in rows]
    best = rows[-1]["best_epoch"] + 1

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(epochs, loss, lw=1.6, color="#185FA5", label="Validation loss")
    ax.axvline(
        best, color="#993C1D", ls="--", lw=1, label=f"Best checkpoint (epoch {best})"
    )
    ax.axvspan(
        epochs[-1],
        500,
        color="0.85",
        alpha=0.6,
        label="Planned epochs not run (compute)",
    )
    ax.set_yscale("log")
    ax.set_xlim(0, 500)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (energy + force, log scale)")
    ax.set_title("MACE on 3BPA, seed 123: still improving when stopped at epoch 367")
    ax.text(
        0.98,
        0.62,
        "Learning rate stayed at 0.01:\nthe plateau scheduler never fired",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=9,
        color="0.25",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = Path(__file__).with_name("learning-curve.png")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
