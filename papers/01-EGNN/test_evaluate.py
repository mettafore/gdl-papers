import sys
from pathlib import Path

import pytest

pytest.importorskip("torch_geometric")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from train import train
from evaluate import evaluate


def test_evaluate_scores_a_trained_run(tmp_path):
    run_id, _, _ = train(
        target="gap", epochs=1, hidden_nf=8, n_layers=2, batch_size=32,
        runs_base=str(tmp_path),
    )

    metric_name, score = evaluate(run_id, runs_base=str(tmp_path))

    assert metric_name == "mae"
    assert isinstance(score, float)
    assert score > 0
