"""Scaffold tests for EGNN model pieces, filled in alongside src/model.py."""

import torch
from src.model import EGCL, EGNN, unsorted_segment_sum


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


def test_forward_output_shape():
    egcl = EGCL(input_nf=4, hidden_nf=4, edges_in_d=0)
    h = torch.randn(5, 4)
    x = torch.randn(5, 3)
    edge_index = (torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4]))
    out = egcl.forward(h, x, edge_index)
    assert out.shape == (5, 4)


def test_forward_matches_manual_chain():
    # forward must equal node_model(h, ei, edge_model(h[row], h[col], radial, attr)).
    egcl = EGCL(input_nf=3, hidden_nf=3, edges_in_d=0)
    h = torch.randn(4, 3)
    x = torch.randn(4, 3)
    row = torch.tensor([0, 1, 2, 3])
    col = torch.tensor([1, 2, 3, 0])
    with torch.no_grad():
        radial = egcl._coord2radial((row, col), x)
        empty_attr = torch.zeros(row.size(0), 0)
        edge_feat = egcl.edge_model(h[row], h[col], radial, empty_attr)
        expected = egcl.node_model(h, (row, col), edge_feat)
        out = egcl.forward(h, x, (row, col))
    assert torch.allclose(out, expected)


def test_forward_is_e3_invariant():
    # EGNN's core property: node features depend only on distances, so rotating
    # AND translating input coords must leave the output h unchanged.
    torch.manual_seed(0)
    egcl = EGCL(input_nf=4, hidden_nf=4, edges_in_d=0)
    h = torch.randn(6, 4)
    x = torch.randn(6, 3)
    edge_index = (torch.tensor([0, 1, 2, 3, 4, 5]), torch.tensor([1, 2, 3, 4, 5, 0]))

    # random rotation (Q from QR) + translation
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    t = torch.randn(3)
    x_transformed = x @ Q.T + t

    with torch.no_grad():
        out = egcl.forward(h, x, edge_index)
        out_t = egcl.forward(h, x_transformed, edge_index)
    assert torch.allclose(out, out_t, atol=1e-5)


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


# --- EGNN (full model) ---


def _small_egnn():
    return EGNN(in_node_nf=5, hidden_nf=8, n_layers=3, edges_in_d=0)


def _mol(num_nodes=6):
    torch.manual_seed(0)
    h = torch.randn(num_nodes, 5)
    x = torch.randn(num_nodes, 3)
    # ring graph
    row = torch.arange(num_nodes)
    col = (row + 1) % num_nodes
    return h, x, (row, col)


def test_egnn_output_is_scalar():
    model = _small_egnn()
    h, x, ei = _mol()
    out = model(h, x, ei)
    assert out.numel() == 1


def test_egnn_layers_registered_as_params():
    # ModuleList (not a plain list) means EGCL params show up in model.parameters().
    # A plain list would silently drop them and the layers would never train.
    model = _small_egnn()
    names = [n for n, _ in model.named_parameters()]
    assert any("layers" in n for n in names)
    # 3 layers each with edge_mlp + node_mlp params -> comfortably > 10 tensors
    assert len(names) > 10


def test_egnn_is_e3_invariant():
    # Whole-model property prediction must be unchanged under rotation+translation.
    model = _small_egnn()
    h, x, ei = _mol()
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    t = torch.randn(3)
    with torch.no_grad():
        out = model(h, x, ei)
        out_t = model(h, x @ Q.T + t, ei)
    assert torch.allclose(out, out_t, atol=1e-4)


# --- attention (sigmoid message gate; reference Table-1 config) ---

def test_attention_off_has_no_att_mlp():
    egcl = EGCL(input_nf=4, hidden_nf=4, attention=False)
    assert not hasattr(egcl, "att_mlp") or egcl.att_mlp is None


def test_attention_gate_shrinks_messages():
    # Gate is sigmoid -> strictly in (0,1) -> every gated message must have
    # strictly smaller magnitude than the ungated one from the same edge_mlp.
    torch.manual_seed(0)
    egcl = EGCL(input_nf=4, hidden_nf=4, attention=True)
    num_edges = 6
    h_row, h_col = torch.randn(num_edges, 4), torch.randn(num_edges, 4)
    radial = torch.randn(num_edges, 1)
    edge_attr = torch.zeros(num_edges, 0)
    with torch.no_grad():
        gated = egcl.edge_model(h_row, h_col, radial, edge_attr)
        egcl.attention = False
        ungated = egcl.edge_model(h_row, h_col, radial, edge_attr)
    assert gated.shape == ungated.shape
    assert (gated.abs() <= ungated.abs()).all()
    assert not torch.allclose(gated, ungated)


def test_attention_params_registered_and_trainable():
    egcl = EGCL(input_nf=4, hidden_nf=4, attention=True)
    att_params = [n for n, _ in egcl.named_parameters() if "att" in n]
    assert len(att_params) >= 2  # weight + bias of the gate's Linear


def test_egnn_threads_attention_to_layers():
    model = EGNN(in_node_nf=5, hidden_nf=8, n_layers=3, attention=True)
    assert all(layer.attention for layer in model.layers)


def test_egnn_with_attention_still_e3_invariant():
    model = EGNN(in_node_nf=5, hidden_nf=8, n_layers=2, attention=True)
    h = torch.randn(4, 5)
    x = torch.randn(4, 3)
    row = torch.tensor([0, 1, 2, 3, 0, 2])
    col = torch.tensor([1, 0, 3, 2, 2, 0])
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    t = torch.randn(3)
    with torch.no_grad():
        out = model(h, x, (row, col))
        out_t = model(h, x @ Q.T + t, (row, col))
    assert torch.allclose(out, out_t, atol=1e-4)
