"""Trivial model — the swappable slot. A small MLP for Iris classification.

Not the science; just a real nn.Module to prove the train/evaluate loop. Real
papers (EGNN, MACE, ...) drop their own model.py into the same slot.
"""

from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
