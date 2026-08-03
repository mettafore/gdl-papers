import pytest

pytest.importorskip("torch_geometric")

from src.data import QM9_TARGETS, load_split


@pytest.mark.needs_data
def test_load_split_dims_include_target_col():
    _, _, _, _, dims = load_split(target="gap", batch_size=4)
    assert dims["target_col"] == QM9_TARGETS["gap"]


@pytest.mark.needs_data
def test_load_split_dims_target_col_tracks_target():
    _, _, _, _, dims = load_split(target="mu", batch_size=4)
    assert dims["target_col"] == QM9_TARGETS["mu"]
