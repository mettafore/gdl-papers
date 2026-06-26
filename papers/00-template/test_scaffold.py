"""Scaffold tests: smoke (train runs + loss drops), unit (metrics), split
(train/test deterministic and identical across train & evaluate)."""

import pytest
import torch

import data
import train as train_mod
import evaluate as eval_mod
from gdl import accuracy, mae, rmse


# --- unit: metrics ---

def test_accuracy_perfect_and_partial():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])  # argmax: 0, 2
    assert accuracy(logits, torch.tensor([0, 2])) == 1.0
    assert accuracy(logits, torch.tensor([0, 0])) == 0.5


def test_mae_rmse_known_values():
    out = torch.tensor([1.0, 2.0, 3.0])
    tgt = torch.tensor([1.0, 2.0, 5.0])  # errors: 0, 0, 2
    assert mae(out, tgt) == pytest.approx(2.0 / 3.0, rel=1e-5)
    assert rmse(out, tgt) == pytest.approx((4.0 / 3.0) ** 0.5, rel=1e-5)


# --- split: determinism + no leakage ---

def test_split_deterministic():
    a = data.load_split(seed=42)
    b = data.load_split(seed=42)
    for ta, tb in zip(a[:4], b[:4]):
        assert torch.equal(ta, tb)


def test_split_reports_shapes():
    _, _, _, _, in_dim, out_dim = data.load_split(seed=42)
    assert in_dim == 4 and out_dim == 3


def test_train_eval_use_same_split():
    # evaluate must score on the exact test split train held out.
    Xtr, ytr, Xte, yte, _, _ = data.load_split(seed=7)
    Xtr2, ytr2, Xte2, yte2, _, _ = data.load_split(seed=7)
    assert torch.equal(Xte, Xte2) and torch.equal(yte, yte2)
    # train and test are disjoint in size (no overlap by construction of split)
    assert Xtr.shape[0] + Xte.shape[0] == 150


# --- smoke: train runs + loss drops, artifacts written ---

def test_train_smoke_loss_drops(tmp_path):
    run_id, run_dir, losses = train_mod.train(
        epochs=50, seed=42, runs_base="runs", paper_dir=tmp_path
    )
    assert losses[-1] < losses[0]
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoint.pt").exists()


def test_train_then_evaluate_roundtrip(tmp_path):
    run_id, _, _ = train_mod.train(
        epochs=100, seed=42, runs_base="runs", paper_dir=tmp_path
    )
    metric, score = eval_mod.evaluate(run_id, runs_base="runs", paper_dir=tmp_path)
    assert metric == "accuracy"
    assert 0.0 <= score <= 1.0
    assert score > 0.8  # Iris MLP should learn easily
