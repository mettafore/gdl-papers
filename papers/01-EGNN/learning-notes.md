# EGNN learning notes

## Overall code structure

- **`data.py`** prepares molecules, graph edges, input features, and prediction targets.

- **`train.py`** runs training: feeds batches from the data loader into the model, computes the loss, and updates parameters after each training batch.

- **`model.py`** defines the neural network architecture and how it computes predictions.

- **`evaluate.py`** loads a saved checkpoint and measures prediction error on the test set, so we can compare the result with the paper's benchmark.

## EGNN class notes

Constructor inputs:

- **`in_node_nf`** — number of input features per atom, before they are converted to a hidden representation by the embedding layer.
- **`hidden_nf`** — width of each node's hidden representation after embedding and throughout the EGCL layers.
- **`n_layers=7`** — number of EGCL layers.
- **`edges_in_d=0`** — no additional edge attributes $a_{ij}$. Squared distances still enter the edge-message computation separately.
- **`act_fn=nn.SiLU()`** — activation function used by the node and graph decoders.
- **`attention=False`** — optional scalar gating of edge messages is disabled by default. Nodes still exchange messages; the training configuration can enable the gate.


### Architecture defined in `__init__`

- **Embedding:** a linear layer mapping `in_node_nf` to `hidden_nf`.
- **EGCL stack:** `n_layers` EGCL layers, seven by default.
- **Node decoder:** Linear → activation → Linear, with `hidden_nf` inputs and outputs. What is its role?
- **Graph decoder:** Linear → activation → Linear, ending with one output for the scalar target. How does its role differ from the node decoder?

### `forward` inputs

- **`h`** — input atom features. I expected a different symbol for the features before embedding; the code reuses `h` after embedding.
- **`x`** — 3D positions of the atoms.
- **`edge_index`** — sender/receiver connections between atoms.
- **`edge_attr=None`** — optional edge-feature values.
- **`batch`** — one molecule index per atom/node. For two molecules with three and two atoms, `batch = [0, 0, 0, 1, 1]`. Pooling uses these indices to sum atom representations separately for each molecule. If omitted (`None`), all nodes are treated as one molecule. I initially mistook this for the batch size.


### Forward pass

If `batch` is `None`, every atom gets molecule index `0`.

Input atom features → embedding → `n_layers` EGCL layers → node decoder → sum per molecule using `batch` → graph decoder → one scalar prediction per molecule.

The node decoder transforms each atom's vector, keeping its width at `hidden_nf`. Pooling sums these vectors separately for each molecule. The graph decoder then acts on each pooled molecule vector and produces its scalar prediction.

### `unsorted_segment_sum`

Adds rows belonging to the same segment. They do not need to be adjacent.

```text
data         = [[1, 2], [3, 4], [5, 6]]
segment_ids  = [0, 1, 0]
num_segments = 2
output       = [[6, 8], [3, 4]]
```

Line 38 was the confusing part: `unsqueeze` turns `[0, 1, 0]` into a column, then `expand_as(data)` expands it to `[[0, 0], [1, 1], [0, 0]]`.

My understanding: we expand the segment indices to match the columns of `data`, so each value has a destination row. The same column adds to the same column: column 0 stays in column 0, and column 1 stays in column 1.

## Data preparation before EGCL

### `charge_power_features`

The input `z` holds the atomic number of each atom. The function turns those numbers into node features for the model.

- It starts with a length-10 lookup tensor filled with `-1`. It assigns the five elements in QM9—atomic numbers `1, 6, 7, 8, 9`—the IDs `0, 1, 2, 3, 4`. Looking up each value in `z` gives its element ID.
- It converts each ID to a five-entry one-hot vector.
- For each atom, it calculates three values: `1`, `z/9`, and `(z/9)²`.
- It combines the one-hot vector with those values. Only the group for that atom's element is nonzero; that group contains the three powers.
- The intermediate shape is `(N, 5, 3)`. Flattening the last two dimensions gives `(N, 15)`: **N atoms, each with 5 possible element types × 3 power features**. I initially wrote `N*15`, but the atom dimension stays separate.

### `fully_connected_edge_index`

The input is the number of atoms, `N`. The function forms every ordered pair of atom IDs, like two nested `for` loops. `repeat_interleave` supplies the outer-loop index, and `repeat` supplies the inner-loop index. Pairing the resulting `row` and `col` entries gives `N × N` pairs.

The mask removes pairs where an atom connects to itself (`row == col`). Stacking the remaining indices gives an `edge_index` of shape `(2, N(N−1))`. Both `(i, j)` and `(j, i)` remain, so every atom can exchange messages with every other atom—even if they are not chemically bonded.
