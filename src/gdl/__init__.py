"""gdl — shared, paper-agnostic helpers for the gdl-papers scaffold.

Pulled BY papers; never calls into them. See docs/spec.md.
"""

from gdl.seed import set_seed
from gdl.metrics import accuracy, mae, rmse, METRICS
from gdl.run_log import log_metrics
from gdl.run import run_dir, new_run_dir, load_run_config
from gdl.checkpoint import save_checkpoint, load_checkpoint

__all__ = [
    "set_seed",
    "accuracy",
    "mae",
    "rmse",
    "METRICS",
    "log_metrics",
    "run_dir",
    "new_run_dir",
    "load_run_config",
    "save_checkpoint",
    "load_checkpoint",
]
