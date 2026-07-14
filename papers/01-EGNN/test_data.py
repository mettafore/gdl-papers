import pytest

pytest.importorskip("torch_geometric")

from src.data import load_split, QM9_TARGETS


def test_load_split_dims_include_target_col():
    _, _, _, _, dims = load_split(target="gap", batch_size=4)
    assert dims["target_col"] == QM9_TARGETS["gap"]


def test_load_split_dims_target_col_tracks_target():
    _, _, _, _, dims = load_split(target="mu", batch_size=4)
    assert dims["target_col"] == QM9_TARGETS["mu"]
