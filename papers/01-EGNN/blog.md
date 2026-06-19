# What the EGNN Paper Doesn't Tell You

**Paper:** E(n) Equivariant Graph Neural Networks — Satorras et al., ICML 2021

---

I spent a week doing a deep Pass 3 read of EGNN — not just understanding the paper, but verifying every equivariance guarantee line by line in the actual implementation. Here's what I learned that the paper doesn't tell you.

---

## The Paper in 30 Seconds

Standard GNNs operating on 3D data (molecules, particles) have no way to respect physical symmetries. Rotate a molecule and you get a different prediction — which is wrong, the physics doesn't change.

Previous equivariant methods (TFN, SE(3)-Transformer) solved this using spherical harmonics — mathematically powerful but computationally expensive and locked to 3D.

EGNN solves it with two simple ideas:

1. Feed $\|x_i - x_j\|^2$ (squared distance, rotation-invariant) into the edge MLP instead of the raw position vectors
2. Update coordinates using $(x_i - x_j)$ scaled by invariant scalars — equivariant by construction

The 4 core equations:

$$m_{ij} = \phi_e(h_i^l, h_j^l, \|x_i^l - x_j^l\|^2, a_{ij}) \quad \text{(edge update)}$$

$$x_i^{l+1} = x_i^l + C\sum_{j \neq i}(x_i^l - x_j^l)\phi_x(m_{ij}) \quad \text{(coordinate update)}$$

$$m_i = \sum_{j \neq i} m_{ij} \quad \text{(aggregation)}$$

$$h_i^{l+1} = \phi_h(h_i^l, m_i) \quad \text{(node update)}$$

Clean. Simple. Works in any dimension n, not just 3D.

---

## The Gap Between Equations and Implementation

Here's what I didn't expect: the equations look like 4 lines of math. The implementation has to actively enforce every guarantee the math assumes. Three choices in the code that are invisible in the paper:

### 1. `bias=False` on the coordinate MLP

```python
layer = nn.Linear(hidden_nf, 1, bias=False)
```

The final layer of $\phi_x$ has no bias term. Why?

If $\phi_x$ had a bias, it would output `scalar + constant`. That constant would multiply $(x_i - x_j)$ and sum over all neighbors — adding a fixed translation to every node's position. Translation equivariance silently broken.

**Silent failure:** add `bias=True`, train the model, loss goes down — but translation equivariance is gone.

### 2. Squared distance goes into the MLP — the direction vector does not

```python
# coord2radial
coord_diff = coord[row] - coord[col]           # (x_i - x_j) — direction vector
radial = torch.sum(coord_diff**2, 1).unsqueeze(1)  # ||x_i - x_j||² — invariant scalar

# edge_model
out = torch.cat([source, target, radial], dim=1)  # radial goes in, not coord_diff
```

`radial` ($\|x_i - x_j\|^2$) is rotation-invariant — safe to feed into an MLP. `coord_diff` ($(x_i - x_j)$) is a geometric vector — it rotates with the data. Feeding it into an MLP would mean treating each spatial component as an independent number, destroying rotation equivariance.

The rule: **geometric vectors stay outside MLPs, only used for multiplication by invariant scalars**.

**Silent failure:** replace `radial` with `coord_diff` as MLP input — model trains fine, rotation equivariance gone.

### 3. `.detach()` on the normalization denominator

```python
norm = torch.sqrt(radial).detach() + self.epsilon
coord_diff = coord_diff / norm
```

When the optional `normalize` flag is on, `coord_diff` gets divided by its norm to produce a unit vector. The `.detach()` cuts the gradient path through `norm`.

Without it, the gradient of $\sqrt{x}$ near zero blows up when two nodes get very close — $\frac{d}{dx}\sqrt{x} = \frac{1}{2\sqrt{x}} \to \infty$ as $x \to 0$. Not an equivariance issue — a training stability issue. An accepted tradeoff: you lose a small gradient contribution, you gain numerical stability.

---

## Verifying the Guarantees

I wrote an explicit equivariance test — something the original repo doesn't include:

```python
def test_equivariance():
    h = torch.randn(batch_size * n_nodes, n_feat)
    x = torch.randn(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)
    model = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=n_feat, in_edge_nf=1)
    model.eval()

    # Forward pass on original input
    with torch.no_grad():
        h_out, x_out = model(h, x, edges, edge_attr)

    # Rotate and translate input coordinates
    Q, _ = torch.linalg.qr(torch.randn(3, 3))  # random rotation
    g = torch.randn(1, 3)                        # random translation
    x_transformed = x @ Q.T + g

    # Forward pass on transformed input
    with torch.no_grad():
        h_out2, x_out2 = model(h, x_transformed, edges, edge_attr)

    # Check equivariance: f(Qx + g) = Q*f(x) + g
    x_out_transformed = x_out @ Q.T + g
    assert torch.allclose(h_out, h_out2, atol=1e-5)   # h is invariant
    assert torch.allclose(x_out2, x_out_transformed, atol=1e-5)  # x is equivariant
```

Both assertions pass. The guarantees hold.

---

## The N-body Velocity Extension (Eq. 7)

For the N-body experiment, the paper extends the coordinate update to incorporate velocity:

$$v_i^{l+1} = \phi_v(h_i^l) \cdot v_i^{init} + C\sum_{j \neq i}(x_i^l - x_j^l)\phi_x(m_{ij})$$
$$x_i^{l+1} = x_i^l + v_i^{l+1}$$

Two subtle things:

1. It uses $v_i^{init}$ — the **original** input velocity, not a recursively updated one. Every layer scales the same initial velocity. This avoids error accumulation across layers.

2. Node features $h$ are initialized with **speed** $\|v_i\|$ (invariant scalar), not the velocity vector. Feeding raw velocity vectors as node features would break rotation equivariance — same rule as above.

---

## One Insight

The equations tell you **what** to build. The implementation tells you **how to make it actually work**.

Every design choice that looks like a detail — `bias=False`, detached norms, speed not velocity as node features — is load-bearing. Remove any one of them and equivariance breaks silently. The model still trains. The loss still goes down. You'd never know.

This is what makes equivariant networks hard: they fail quietly.

---

## Where Next

EGNN uses Euclidean distance $\|x_i - x_j\|^2$ as its geometric invariant. This assumes the space is isotropic — geometry depends only on distance, not direction.

Many real systems are anisotropic. Replacing the Euclidean metric with a Finsler metric — where distance depends on direction — is an open research direction with almost no existing work. That's the natural next question after EGNN.
