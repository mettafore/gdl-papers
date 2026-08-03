"""Run folders — the atomic unit of provenance.

Each run lives in ``<paper_dir>/<base>/<run_id>/`` and holds config.json,
metrics.jsonl, and checkpoint.pt. A run records everything needed to reproduce
and evaluate it.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _provenance() -> dict:
    """Best-effort code identity for a run: git commit, dirty flag, uv.lock hash.

    Makes any benchmark number traceable to the exact code that produced it.
    Every field degrades to None outside a git checkout (e.g. a Modal
    container) rather than failing the run.
    """
    commit, dirty, lock_sha = None, None, None
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        lock = Path(root) / "uv.lock"
        if lock.exists():
            lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()[:16]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return {"git_commit": commit, "git_dirty": dirty, "uv_lock_sha256": lock_sha}


def run_dir(paper_dir, run_id: str, base: str = "runs") -> Path:
    """The canonical path to a run folder. One definition; all callers use it."""
    return Path(paper_dir) / base / run_id


def new_run_dir(paper_dir, config: dict, base: str = "runs"):
    """Create a fresh run folder and write config.json.

    Returns (run_id, run_dir). ``base`` is a parameter so tests can redirect runs
    to a tmp dir instead of littering the paper's runs/.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id, suffix = stamp, 0
    while run_dir(paper_dir, run_id, base).exists():
        suffix += 1
        run_id = f"{stamp}-{suffix}"  # avoid collision on same-second runs
    rdir = run_dir(paper_dir, run_id, base)
    rdir.mkdir(parents=True, exist_ok=True)
    with (rdir / "config.json").open("w") as f:
        json.dump({**config, "provenance": _provenance()}, f, indent=2)
    return run_id, rdir


def load_run_config(paper_dir, run_id: str, base: str = "runs") -> dict:
    """Read config.json for a given run."""
    path = run_dir(paper_dir, run_id, base) / "config.json"
    with path.open() as f:
        return json.load(f)
