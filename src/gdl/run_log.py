"""Metric logging — dual sink: console + per-run JSONL.

Backend-agnostic by design. A richer backend (e.g. W&B) can be added behind
this same call later without changing any paper's train.py.
"""

import json
from pathlib import Path


def log_metrics(metrics: dict, step: int, run_dir) -> None:
    """Print metrics to console and append one JSON line to run_dir/metrics.jsonl."""
    record = {"step": step, **metrics}

    pretty = "  ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in record.items()
    )
    print(pretty)

    path = Path(run_dir) / "metrics.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
