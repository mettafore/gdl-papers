import torch
from egnn_clean import EGNN, get_edges_batch


def random_rotation():
    Q, _ = torch.linalg.qr(torch.randn(3,3))
    return Q

def test_equivariance():
    batch_size=2
    n_nodes = 5
    n_feat = 4
    x_dim = 3

    h = torch.randn(batch_size * n_nodes, n_feat)
    x = torch.randn(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    model = EGNN(
        in_node_nf=n_feat,
        hidden_nf=32,
        out_node_nf=n_feat,
        in_edge_nf=1
    )
    model.eval()

    with torch.no_grad():
        h_out, x_out = model(
            h,x,edges, edge_attr
        )
    
    Q = random_rotation()
    g = torch.randn(1, x_dim)

    x_t = x @ Q.T + g

    with torch.no_grad():
        h_out2, x_out2 = model(
            h,
            x_t,
            edges,
            edge_attr
        )
    
    x_out_t = x_out @ Q.T + g
    assert torch.allclose(h_out, h_out2, atol=1e-5)
    assert torch.allclose(x_out2, x_out_t, atol=1e-5)

if __name__=="__main__":
    test_equivariance()
    print("All equivariance tests passed")
