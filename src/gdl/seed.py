"""Global RNG seeding for reproducible training.

Note: this seeds the *global* RNG. Deterministic data splits should still pass
an explicit seed to the split itself (see papers/00-template/data.py) — global
seeding alone does not fix split order-dependence across separate scripts.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
