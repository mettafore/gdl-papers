"""gdl — shared, paper-agnostic helpers for the gdl-papers scaffold.

Pulled BY papers; never calls into them. See docs/spec.md.
"""

from gdl.checkpoint import load_checkpoint, save_checkpoint
from gdl.metrics import METRICS, accuracy, mae, rmse
from gdl.run import load_run_config, new_run_dir, run_dir
from gdl.run_log import log_metrics
from gdl.seed import set_seed

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
