"""EGNN model for QM9 property prediction. [SCAFFOLD]

Paper: Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021.
QM9 config: 7 EGCL layers, 128 hidden features, Swish activation.
"""

import torch
import torch.nn as nn


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Sum rows of `data` grouped by `segment_ids` (scatter-add).

    Used to aggregate per-edge messages m_ij into per-node messages m_i:
    m_i = Σ_j m_ij, where j ranges over edges with row==i.

    Args:
        data: (num_edges, feat_dim) — per-edge values (e.g. edge_model output).
        segment_ids: (num_edges,) — destination node index per edge (the `row`
            from edge_index / _coord2radial).
        num_segments: number of nodes (output has this many rows).

    Returns:
        (num_segments, feat_dim) tensor, row i = sum of data[e] for all edges e
        where segment_ids[e] == i.
    # TODO:
    #   - allocate output tensor of zeros, shape (num_segments, data.size(1)),
    #     same dtype/device as data
    #   - expand segment_ids to match data's shape so it can index every column
    #     (hint: unsqueeze + expand_as)
    
    #   - use Tensor.scatter_add_ along dim 0
    #   - (equivalent alt: torch_scatter.scatter or torch_geometric.utils.scatter)
    """
    output = torch.zeros(num_segments, data.size(1), dtype=data.dtype, device=data.device)
    segment_id_exp = torch.unsqueeze(segment_ids, 1).expand_as(data)
    return output.scatter_add_(0, segment_id_exp, data)

class EGCL(nn.Module):
    """Equivariant Graph Convolutional Layer (equations 3-6).

    For QM9, coordinate update (eq 4) is skipped since positions are static.
    """

    # TODO: __init__
    def __init__(self, input_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU()):
        super().__init__()
        self.input_nf = input_nf
        self.hidden_nf = hidden_nf
        self.edges_in_d = edges_in_d
        self.act_fn = act_fn

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.input_nf + self.edges_in_d + 1, self.hidden_nf),
            nn.SiLU(),
            nn.Linear(self.hidden_nf, self.hidden_nf),
            nn.SiLU(),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            nn.SiLU(),
            nn.Linear(self.hidden_nf, 1),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(self.hidden_nf + self.input_nf, self.input_nf),
            nn.SiLU(),
            nn.Linear(self.input_nf, self.input_nf),
        )

    #   - edge_mlp (φ_e): [h_i, h_j, ||dist||^2, edge_attr] → hidden
    #   - coord_mlp (φ_x): hidden → 1 (optional for QM9)
    #   - node_mlp (φ_h): [h, agg] → hidden, with residual
    #   - coord2radial helper: compute squared distances
    def _coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True)
        return radial

    def edge_model(self, h_row, h_col, radial, edge_attr):
        """Message computation (eq. 3): m_ij = φ_e(h_i, h_j, ||x_i - x_j||², a_ij).

        Args:
            h_row: (num_edges, input_nf) — embedding of destination node per edge (h[row]).
            h_col: (num_edges, input_nf) — embedding of source node per edge (h[col]).
            radial: (num_edges, 1) — squared distance per edge (from _coord2radial).
            edge_attr: (num_edges, edges_in_d) — edge features a_ij (empty if edges_in_d==0).

        Returns:
            (num_edges, hidden_nf) tensor — per-edge messages m_ij.
        """
        combined_input = torch.cat([h_row, h_col, radial, edge_attr], dim=1)
        return self.edge_mlp(combined_input)

    def node_model(self, h, edge_index, edge_feat):
        """Node update (eq. 5-6): aggregate incoming messages, apply φ_h with residual.

        Args:
            h: (num_nodes, hidden_nf) — current node embeddings.
            edge_index: (2, num_edges) — (row, col) = (dest, src) per edge.
            edge_feat: (num_edges, hidden_nf) — per-edge messages m_ij (edge_model output).

        Returns:
            (num_nodes, hidden_nf) tensor — updated node embeddings h_i^(l+1).
        # TODO:
        #   - split edge_index into row, col
        #   - aggregate edge_feat per destination node (row) via unsorted_segment_sum,
        #     num_segments = h.size(0)
        #   - concat [h, agg] along dim=1, pass through self.node_mlp
        #   - return residual: h + node_mlp output (MLP predicts a correction)
        """
        raise NotImplementedError

    # TODO: forward(h, x, edge_index, edge_attr)


class EGNN(nn.Module):
    """Full EGNN: embedding → EGCL layers → output head.

    QM9: 7 layers, 128 features, sum pooling.
    """

    # TODO: __init__
    #   - embedding: Linear(in_node_dim, hidden)
    #   - 7 EGCL layers
    #   - node_dec: Linear → Swish → Linear
    #   - graph_dec: Linear → Swish → Linear → 1

    # TODO: forward(h, x, edge_index, edge_attr)
    #   - embed h
    #   - pass through EGCL layers
    #   - node_dec → sum pooling → graph_dec → scalar
