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


# --- charge-power node features (reference Table-1 config) ---

import torch

from src.data import charge_power_features


def test_charge_power_shape():
    z = torch.tensor([1, 6, 7, 8, 9])  # H C N O F
    out = charge_power_features(z)
    assert out.shape == (5, 15)


def test_charge_power_p0_slice_is_one_hot():
    # (z/9)^0 == 1, so those 5 columns must reproduce the plain one-hot:
    # exactly one 1.0 per row, in a distinct column per element.
    z = torch.tensor([1, 6, 7, 8, 9])
    out = charge_power_features(z).reshape(5, 5, 3)
    p0 = out[:, :, 0]
    assert torch.equal(p0, torch.eye(5))


def test_charge_power_known_values_hydrogen():
    # H: z=1 -> nonzero block is [(1/9)^0, (1/9)^1, (1/9)^2]; all else 0.
    z = torch.tensor([1])
    out = charge_power_features(z).reshape(1, 5, 3)
    expected = torch.tensor([1.0, 1 / 9, (1 / 9) ** 2])
    assert torch.allclose(out[0, 0], expected)
    assert torch.count_nonzero(out) == 3


def test_charge_power_fluorine_powers_all_one():
    # F: z=9 -> (9/9)^p = 1 for every p.
    z = torch.tensor([9])
    out = charge_power_features(z).reshape(1, 5, 3)
    assert torch.allclose(out[0, 4], torch.ones(3))


def test_load_split_reports_15_dim_nodes():
    # Wiring test: once the transform is applied, dims must reflect the REAL
    # batch width (15), not dataset.num_features' stale 11.
    train_loader, _, _, _, dims = load_split(target="gap", batch_size=4)
    assert dims["in_node_dim"] == 15
    batch = next(iter(train_loader))
    assert batch.x.shape[1] == 15


# --- fully-connected edges (reference uses complete graphs, not bonds) ---

from src.data import fully_connected_edge_index


def test_fc_edge_count_methane():
    # N=5 -> 5*4 = 20 ordered pairs, no self-loops.
    ei = fully_connected_edge_index(5)
    assert ei.shape == (2, 20)


def test_fc_no_self_loops():
    ei = fully_connected_edge_index(6)
    assert (ei[0] != ei[1]).all()


def test_fc_covers_all_pairs():
    # every ordered pair i!=j present exactly once.
    n = 4
    ei = fully_connected_edge_index(n)
    pairs = {(int(s), int(d)) for s, d in zip(ei[0], ei[1])}
    expected = {(i, j) for i in range(n) for j in range(n) if i != j}
    assert pairs == expected
    assert ei.shape[1] == len(expected)


def test_fc_dtype_long():
    ei = fully_connected_edge_index(3)
    assert ei.dtype == torch.long


def test_load_split_batches_are_fully_connected():
    # Wiring: after the transform, a single-molecule's edge count must be
    # N*(N-1), not the bond count. Uses the first val molecule (no shuffle).
    _, val_loader, _, _, _ = load_split(target="gap", batch_size=1)
    batch = next(iter(val_loader))
    n = batch.x.shape[0]
    assert batch.edge_index.shape[1] == n * (n - 1)
