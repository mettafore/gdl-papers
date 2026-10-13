import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch_geometric")

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

import evaluate as evaluate_module
from data import NormStats
from evaluate import evaluate
from train import train


@pytest.mark.parametrize(
    ("saved_root", "override", "expected"),
    [
        ("saved/qm9", None, "saved/qm9"),
        ("saved/qm9", "moved/qm9", "moved/qm9"),
        (None, None, "data/qm9"),
    ],
)
def test_evaluation_replays_saved_data_root_unless_overridden(
    monkeypatch, saved_root, override, expected
):
    config = {
        "data": {"target": "gap", "dims": {"in_node_dim": 1}},
        "model": {"hidden_nf": 4, "n_layers": 1, "attention": False},
        "seed": 123,
        "metric": "mae",
    }
    if saved_root is not None:
        config["data"]["root"] = saved_root
    roots = []

    class ConstantModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            return torch.tensor([[2.0]])

    batch = SimpleNamespace(
        x=torch.ones((1, 1)),
        pos=torch.zeros((1, 3)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        batch=torch.zeros(1, dtype=torch.long),
        y=torch.tensor([[2.0]]),
        num_graphs=1,
    )

    def load_split(*, root, **kwargs):
        roots.append(root)
        return None, None, [batch], NormStats(0.0, 1.0), {"target_col": 0}

    monkeypatch.setattr(evaluate_module, "load_run_config", lambda *a, **k: config)
    monkeypatch.setattr(evaluate_module, "load_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(evaluate_module, "EGNN", ConstantModel)
    monkeypatch.setattr(evaluate_module.data, "load_split", load_split)

    kwargs = {} if override is None else {"data_root": override}
    assert evaluate_module.evaluate("fixture", **kwargs) == ("mae", 0.0)
    assert roots == [expected]


@pytest.mark.needs_data
def test_evaluate_scores_a_trained_run(tmp_path):
    run_id, _, _ = train(
        target="gap",
        epochs=1,
        hidden_nf=8,
        n_layers=2,
        batch_size=32,
        runs_base=str(tmp_path),
    )

    metric_name, score = evaluate(run_id, runs_base=str(tmp_path))

    assert metric_name == "mae"
    assert isinstance(score, float)
    assert score > 0
