"""Scaffold tests for EGNN model pieces, filled in alongside src/model.py."""

import torch

from src.model import unsorted_segment_sum, EGCL


def test_unsorted_segment_sum_known_values():
    data = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    segment_ids = torch.tensor([0, 1, 0])
    out = unsorted_segment_sum(data, segment_ids, num_segments=2)
    expected = torch.tensor([[4.0, 4.0], [2.0, 2.0]])
    assert torch.equal(out, expected)


def test_unsorted_segment_sum_shape():
    data = torch.randn(7, 5)
    segment_ids = torch.randint(0, 3, (7,))
    out = unsorted_segment_sum(data, segment_ids, num_segments=3)
    assert out.shape == (3, 5)


def test_unsorted_segment_sum_empty_segment_is_zero():
    data = torch.tensor([[1.0], [2.0]])
    segment_ids = torch.tensor([0, 0])
    out = unsorted_segment_sum(data, segment_ids, num_segments=3)
    assert torch.equal(out[2], torch.zeros(1))


def test_unsorted_segment_sum_dtype_device_match_data():
    data = torch.randn(4, 2, dtype=torch.float64)
    segment_ids = torch.tensor([0, 0, 1, 1])
    out = unsorted_segment_sum(data, segment_ids, num_segments=2)
    assert out.dtype == data.dtype
    assert out.device == data.device


# --- edge_model ---

def test_edge_model_output_shape():
    egcl = EGCL(input_nf=4, hidden_nf=4, edges_in_d=0)
    num_edges = 5
    h_row = torch.randn(num_edges, 4)
    h_col = torch.randn(num_edges, 4)
    radial = torch.randn(num_edges, 1)
    edge_attr = torch.zeros(num_edges, 0)
    out = egcl.edge_model(h_row, h_col, radial, edge_attr)
    assert out.shape == (num_edges, 4)


def test_edge_model_with_edge_attr():
    egcl = EGCL(input_nf=4, hidden_nf=6, edges_in_d=2)
    num_edges = 3
    h_row = torch.randn(num_edges, 4)
    h_col = torch.randn(num_edges, 4)
    radial = torch.randn(num_edges, 1)
    edge_attr = torch.randn(num_edges, 2)
    out = egcl.edge_model(h_row, h_col, radial, edge_attr)
    assert out.shape == (num_edges, 6)


# --- node_model ---

def _zeroed_egcl(hidden_nf=4):
    """EGCL with node_mlp weights/biases zeroed so node_mlp(x) == 0 for any x,
    isolating the residual/aggregation wiring in node_model from φ_h's output."""
    egcl = EGCL(input_nf=hidden_nf, hidden_nf=hidden_nf, edges_in_d=0)
    with torch.no_grad():
        for p in egcl.node_mlp.parameters():
            p.zero_()
    return egcl


def test_node_model_shape():
    egcl = _zeroed_egcl(hidden_nf=4)
    h = torch.randn(3, 4)
    edge_index = (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0]))
    edge_feat = torch.randn(3, 4)
    out = egcl.node_model(h, edge_index, edge_feat)
    assert out.shape == h.shape


def test_node_model_residual_with_zeroed_mlp():
    # node_mlp(x) == 0 for all x here, so node_model must reduce to pure residual: h + 0 == h.
    egcl = _zeroed_egcl(hidden_nf=4)
    h = torch.randn(3, 4)
    edge_index = (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0]))
    edge_feat = torch.randn(3, 4)
    out = egcl.node_model(h, edge_index, edge_feat)
    assert torch.allclose(out, h)


def test_node_model_more_edges_than_nodes():
    # num_edges (6) != num_nodes (4): output must have one row per node, not
    # per edge. Guards against sizing num_segments off the edge count.
    egcl = _zeroed_egcl(hidden_nf=3)
    h = torch.randn(4, 3)
    row = torch.tensor([0, 0, 1, 1, 2, 3])
    col = torch.tensor([1, 2, 0, 3, 0, 1])
    edge_feat = torch.randn(6, 3)
    out = egcl.node_model(h, (row, col), edge_feat)
    assert out.shape == (4, 3)


def test_node_model_matches_manual_aggregate_and_mlp():
    # Cross-check node_model's full output (real, non-zeroed node_mlp) against
    # manually computing agg via the already-tested unsorted_segment_sum and
    # feeding [h, agg] through egcl.node_mlp directly. Pins down aggregation
    # index (row = destination) and the residual add, without prescribing
    # the internal order of operations inside node_model.
    egcl = EGCL(input_nf=3, hidden_nf=3, edges_in_d=0)
    h = torch.randn(4, 3)
    row = torch.tensor([0, 0, 2, 3])
    col = torch.tensor([1, 2, 0, 1])
    edge_feat = torch.randn(4, 3)

    with torch.no_grad():
        agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0))
        expected = h + egcl.node_mlp(torch.cat([h, agg], dim=1))
        out = egcl.node_model(h, (row, col), edge_feat)

    assert torch.allclose(out, expected)
