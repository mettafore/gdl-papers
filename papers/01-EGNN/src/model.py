"""EGNN model for QM9 property prediction. [SCAFFOLD]

Paper: Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021.
QM9 config: 7 EGCL layers, 128 hidden features, Swish activation.
"""

import torch
import torch.nn as nn

# TODO: helper unsorted_segment_sum (or use PyG scatter)


class EGCL(nn.Module):
    """Equivariant Graph Convolutional Layer (equations 3-6).

    For QM9, coordinate update (eq 4) is skipped since positions are static.
    """
    # TODO: __init__
    #   - edge_mlp (φ_e): [h_i, h_j, ||dist||^2, edge_attr] → hidden
    #   - coord_mlp (φ_x): hidden → 1 (optional for QM9)
    #   - node_mlp (φ_h): [h, agg] → hidden, with residual
    #   - coord2radial helper: compute squared distances
    
    # TODO: edge_model(h_row, h_col, radial, edge_attr)
    # TODO: node_model(h, edge_index, edge_feat)
    # TODO: forward(h, x, edge_index, edge_attr)


class EGNN(nn.Module):
    """Full EGNN: embedding → EGCL layers → output head.

    QM9: 7 layers, 128 features, sum pooling.
    """
    def __init__(self, input_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU()):
        super().__init__()
        self.input_nf = input_nf
        self.hidden_nf = hidden_nf
        self.edges_in_d = edges_in_d
        self.act_fn = act_fn

        self.edge_mlp = nn.Sequential(
            nn.Linear(2*self.input_nf + self.edges_in_d + 1, self.hidden_nf),
            nn.SiLU(),
            nn.Linear(self.hidden_nf, hidden_nf),
            nn.SiLU()
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            nn.SiLU(),
            nn.Linear(self.hidden_nf, 1)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(self.hidden_nf + self.input_nf, self.hidden_nf),
            nn.SiLU(),
            nn.Linear(self.hidden_nf, hidden_nf)
        )

    # TODO: __init__
    #   - embedding: Linear(in_node_dim, hidden)
    #   - 7 EGCL layers
    #   - node_dec: Linear → Swish → Linear
    #   - graph_dec: Linear → Swish → Linear → 1
    
    # TODO: forward(h, x, edge_index, edge_attr)
    #   - embed h
    #   - pass through EGCL layers
    #   - node_dec → sum pooling → graph_dec → scalar
