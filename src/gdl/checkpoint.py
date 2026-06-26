"""Checkpoint I/O — bare model state_dict into a run folder.

The run's config.json carries the recipe to rebuild the model; the checkpoint
carries only the learned weights.
"""

from pathlib import Path

import torch


def save_checkpoint(model, run_dir) -> None:
    torch.save(model.state_dict(), Path(run_dir) / "checkpoint.pt")


def load_checkpoint(model, run_dir) -> None:
    """Load weights into an already-built model (built from config.json)."""
    state = torch.load(Path(run_dir) / "checkpoint.pt", weights_only=True)
    model.load_state_dict(state)
