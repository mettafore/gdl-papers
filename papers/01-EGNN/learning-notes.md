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

#### Shape trace

Example: two molecules with 3 and 5 atoms, 15 input features per atom, and hidden width 128. There are 8 atoms total, with `batch = [0, 0, 0, 1, 1, 1, 1, 1]`.

| Stage | Shape | What each row represents |
|---|---|---|
| Input `h` | `(8, 15)` | One atom's input features |
| `self.emb(h)` | `(8, 128)` | One atom's embedded features |
| After all seven EGCL layers | `(8, 128)` | One atom's updated features |
| `self.node_dec(h)` | `(8, 128)` | One atom's decoded features |
| Sum using `batch` | `(2, 128)` | One pooled molecule vector |
| `self.graph_dec(out)` | `(2, 1)` | One scalar prediction per molecule |

My understanding: embedding changes the feature width; EGCL layers and the node decoder preserve the shape. Pooling changes the row count from atoms to molecules. The graph decoder changes the feature width to 1 while preserving the two molecule rows. I initially answered `(3, 1)` at the final step, then corrected it to `(2, 1)` by tracking that row count.

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

### `_to_charge_power`

`data` is a PyTorch Geometric object for one molecule. This function prepares it for EGNN: it replaces `data.x` with the charge-power atom features, replaces `data.edge_index` with the complete directed graph, sets `data.edge_attr` to `None`, and returns the object. The molecule's positions (`pos`) and target (`y`) remain available.

### `load_split`

This function loads QM9 with `_to_charge_power` as a transform, so each molecule gets the prepared atom features and complete graph when accessed. It checks the requested target, uses a seeded generator to make a reproducible random split, and separates the molecules into training, validation, and test sets.

It computes the training target's mean and mean absolute deviation (MAD), then returns a `NormStats` object for normalization. The three PyG data loaders serve the splits; only the training loader shuffles. `num_workers` controls data-loading worker processes. `dims` records the input **feature width** (15), scalar output width (1), and selected target column—not the number of atoms. Molecules can have different numbers of atoms.

I initially thought `batch_size` counted atoms. Each QM9 dataset item is one molecule, so `batch_size=2` selects two molecules. If they have 3 and 5 atoms, the model sees 8 atoms, with molecule membership recorded as `batch = [0, 0, 0, 1, 1, 1, 1, 1]`. EGCL processes their atoms and edges; pooling later separates the two molecule predictions.

## EGCL class notes

### Constructor and feature widths

`input_nf` means the feature width entering this EGCL layer. These node features have already passed through EGNN's embedding. In the default configuration, both `input_nf` and `hidden_nf` are 128; the original atom-feature width is 15. This naming initially confused me.

`edges_in_d` is the width of the extra edge attributes $a_{ij}$. The constructor stores its arguments on `self`. When attention is enabled, it builds an MLP mapping each edge message to one scalar multiplier through Linear → Sigmoid. We will trace where that multiplier is applied later.

### Edge MLP — equation (3)

$$
m_{ij}=\phi_e\left(h_i,h_j,\|x_i-x_j\|^2,a_{ij}\right).
$$

The first linear layer takes `2 * input_nf + edges_in_d + 1` inputs:

- `2 * input_nf`: the feature vectors of the two endpoint nodes, $h_i$ and $h_j$.
- `edges_in_d`: the extra edge attributes $a_{ij}$.
- `1`: the squared coordinate-difference norm $\|x_i-x_j\|^2$, one scalar per edge.

The edge MLP follows Linear → SiLU → Linear → SiLU. Both linear layers output `hidden_nf` features. Its result is the edge message $m_{ij}$, with shape `(num_edges, hidden_nf)`.

### Node MLP — equations (5) and (6)

The aggregated message is $m_i=\sum_{j\in\mathcal N(i)}m_{ij}$. In this complete graph without self-edges, the neighbors are all other atoms ($j\ne i$). Summing preserves the message width, `hidden_nf`.

The node MLP implements the learned transformation in $\phi_h(h_i,m_i)$. Its input combines the current node features (`input_nf`) with the aggregated message (`hidden_nf`). It follows Linear → SiLU → Linear, mapping `input_nf + hidden_nf` to `input_nf`, then `input_nf` to `input_nf`. With `residual=True`, `node_model` adds the MLP output to the incoming node features to form the updated features.

### Coordinate MLP — part of equation (4)

This MLP represents $\phi_x$, one component of the coordinate-update equation. It takes the edge message $m_{ij}$ and follows Linear → SiLU → Linear: `hidden_nf` → `hidden_nf` → **1**. The output $\phi_x(m_{ij})$ is a scalar weight that would multiply the relative coordinate vector $x_i-x_j$.

I initially expected a coordinate-update step later, but the QM9 forward path keeps coordinates fixed. `coord_mlp` is defined here but never called.

### `_coord2radial` — squared distance per edge

`edge_index` has shape `(2, E)`: two rows, with one column per edge. I initially imagined `(E, 2)`. `row, col = edge_index` unpacks the receiver-index row and sender-index row.

`coord[row]` and `coord[col]` gather the coordinates of the two endpoint nodes for every edge, each with shape `(E, 3)`. Their difference is one relative vector per edge. Squaring its components and summing along `dim=1` gives $(x_i-x_j)^2+(y_i-y_j)^2+(z_i-z_j)^2$, the squared-distance term in equation (3). There is no square root.

`keepdim=True` keeps the summed coordinate dimension as a singleton, so the output is `(E, 1)`. My understanding: the three coordinate components collapse to one value, while the dimension remains present.

Dimension check for two molecules with 3 and 5 atoms: each molecule has all directed edges except self-edges, giving `E = 3 × 2 + 5 × 4 = 26`. I initially counted unordered pairs using combinations, then corrected this to permutations because both edge directions are present.

| Tensor | Shape |
|---|---|
| `edge_index` | `(2, 26)` |
| `row` and `col` | Each `(26,)` |
| `coord[row]` and `coord[col]` | Each `(26, 3)` |
| `coord_diff` | `(26, 3)` |
| `radial` | `(26, 1)` |

### `edge_model` — equation (3) and attention gating

I initially associated this function with equation (4), then recognized that it computes equation (3). It concatenates the two endpoint feature vectors, $h_i$ and $h_j$, the squared distance, and any extra edge attributes $a_{ij}$. Passing that combined input through `edge_mlp` produces the edge message.

I thought `out` was a scalar. Correction: it is a message vector of width `hidden_nf` for each edge, with shape `(E, hidden_nf)`. If attention is enabled, `att_mlp(out)` produces one scalar gate per edge, shaped `(E, 1)`. Multiplication broadcasts that gate across the components of the corresponding message vector.

My intuition is that the gate represents how seriously to take that edge's message. Sigmoid keeps it between 0 and 1. This is learned message gating, rather than transformer self-attention. I still want to trace how the gated messages affect the node update.

Dimension check with `h` shaped `(8, 128)` and 26 edges:

| Tensor | Shape |
|---|---|
| `h_row = h[row]` and `h_col = h[col]` | Each `(26, 128)` |
| Empty `edge_attr` | `(26, 0)` |
| `combined_input` | `(26, 257)` — `128 + 128 + 1 + 0` features per edge |
| `out = edge_mlp(combined_input)` | `(26, 128)` |
| `attn_score = att_mlp(out)` | `(26, 1)` |
| Gated message `attn_score * out` | `(26, 128)` |

My understanding: gathering changes the row count from nodes to edges. Concatenation combines the endpoint features and distance along the feature dimension. The MLP produces a vector per edge; the scalar gate broadcasts across that vector's 128 features.

### `node_model` — aggregation, node update, and residual

The function gets the receiver indices from `edge_index`. `unsorted_segment_sum` adds all edge messages with the same receiver index, giving one aggregated message vector per node. It concatenates each node's current features with that node's aggregated messages and passes the result through `node_mlp`.

With `residual=True`, it returns `h + node_mlp(concat_input)`: the current features plus the vector produced by the MLP. Otherwise, it returns the MLP output directly.

I initially thought `h` was ordered by receivers in the edge list. Correction: `h[i]` is the feature vector for node index `i` in the current graph or batch. Gathering `h[row]` produces edge-ordered receiver features, including repeats. Aggregation puts messages back into node order, so `agg[i]` and `h[i]` belong to the same node.

| Stage | Shape in our two-molecule example |
|---|---|
| Incoming edge messages `edge_feat` | `(26, 128)` |
| Receiver indices `row` | `(26,)` |
| Current node features `h` | `(8, 128)` |
| `agg = unsorted_segment_sum(edge_feat, row, len(h))` | `(8, 128)` |
| `concat_input = torch.concat([h, agg], dim=1)` | `(8, 256)` |
| `node_mlp(concat_input)` | `(8, 128)` |
| Residual sum or direct MLP output | `(8, 128)` |

My understanding: fixing the receiver index while summing reduces edge rows to node rows. Concatenation doubles the feature width here; the MLP brings it back to 128. The residual addition works because both vectors have matching shapes and refer to the same nodes.

### Empty edge attributes in `EGCL.forward`

When `edge_attr` is `None`, the forward method creates an empty tensor shaped `(E, 0)`, or `(26, 0)` in our example. This lets it concatenate tensors with matching edge rows while adding no extra feature columns. A `(26, 1)` column of zeros would add a feature and make the combined width 258 instead of the 257 expected by `edge_mlp`.

### `EGCL.forward` — putting the layer together

1. Unpack `edge_index` into the receiver and sender indices, `row` and `col`.
2. Compute `radial`, the squared distance between the endpoints of each edge.
3. If `edge_attr` is `None`, create an empty `(E, 0)` tensor for concatenation. `len(row)` is the number of edges, not their geometric length.
4. Pass the receiver features, sender features, squared distances, and extra edge attributes through `edge_model`. This computes the edge messages $m_{ij}=\phi_e(\ldots)$, with optional attention gating.
5. Return `node_model`: sum the edge messages for each receiver to form $m_i$, concatenate each aggregate with its node's current features, and apply the node MLP. If the residual is enabled, add the MLP output to the current node features.

| Stage | Shape for 8 atoms and 26 edges |
|---|---|
| Input node features `h` | `(8, 128)` |
| Coordinates `x` | `(8, 3)` |
| `edge_index` | `(2, 26)` |
| `row`, `col` | Each `(26,)` |
| `radial` | `(26, 1)` |
| Empty edge attributes | `(26, 0)` |
| Gathered `h[row]`, `h[col]` | Each `(26, 128)` |
| Edge messages `edge_feat` | `(26, 128)` |
| Aggregated messages inside `node_model` | `(8, 128)` |
| Node MLP input | `(8, 256)` |
| Returned node features | `(8, 128)` |

My understanding: there is no coordinate update in this forward pass. Coordinates supply the squared pairwise distances used in the messages. The layer updates node features, which the full EGNN later pools to predict a scalar molecular property.

## Training notes

### `train` arguments

- **`target="gap"`** — molecular property to predict; the default is the HOMO–LUMO gap.
- **`lr=1e-3`**, **`epochs=1000`** — learning rate and maximum number of epochs.
- **`hidden_nf=128`**, **`n_layers=7`** — hidden feature width and number of EGCL layers.
- **`seed=42`** — random seed.
- **`batch_size=96`** — number of molecules per batch. I initially thought atoms; the total atom count varies with the selected molecules.
- **`runs_base="runs"`**, **`paper_dir=PAPER_DIR`**, **`data_root="data/qm9"`** — run-artifact location, paper directory, and dataset location.
- **`num_workers=0`** — number of data-loading worker processes; zero loads in the main process.
- **`weight_decay=1e-16`** — regularization that discourages large parameter values. This value is effectively negligible; it does not guarantee against exploding gradients.
- **`early_stop_patience=50`** — stop after 50 consecutive epochs without an improvement in validation loss.
- **`attention=True`** — enable the learned scalar gates on edge messages.
- **`resume_from=None`** — optional checkpoint to load. It restores model weights only; the optimizer and scheduler start fresh.

### Setup before the training loop

We set the random seed, then get the training, validation, and test loaders, normalization statistics (`norm`), and dimensions (`dims`) from `data.load_split`. We already went through this in `data.py`.

The config dictionary records the data settings, model settings, hyperparameters, seed, metric, and checkpoint source. `target="gap"` selects the HOMO–LUMO gap; `root` is the QM9 dataset folder. `dims` contains the atom feature width (15), output width (1), and target column (4 for `gap`). These dimensions describe features and targets, not the number of atoms.

`new_run_dir` creates a run folder and saves `config.json` there. The run ID comes from the UTC date and time, with a suffix if needed to avoid a collision.

We choose CUDA if available, otherwise CPU, construct the EGNN using the input feature width and model settings, and move it to that device. If a checkpoint was supplied, we load its weights before creating Adam with the model parameters, learning rate, and weight decay.

I initially thought the scheduler lowered the learning rate when loss stopped improving. Correction: `CosineAnnealingLR` follows a smooth cosine schedule based on epoch progress, toward its default minimum of zero. It does not react to validation loss. `nn.L1Loss()` creates the callable mean absolute error loss. We initialize `best_val` to infinity and the epochs-without-improvement counter to zero.

### Training and validation loop

For each epoch, we set the EGNN to training mode and take one batch at a time from the training loader. We move the batch to the device, clear the gradients, get the model output, and obtain the normalized ground-truth target using `target_for`. We calculate the loss, backpropagate with `loss_value.backward()`, and update the weights with `optimizer.step()`.

My correction: `train()` sets training behavior for relevant layers; it does not itself update parameters. The optimizer step performs the update.

For validation, we switch to evaluation mode and use `torch.no_grad()` to disable gradient tracking. We run each validation batch through the model. The dataset still stores raw targets; training normalizes them when `target_for` is called:

```text
normalized target = (raw target - training mean) / training MAD
```

The model learns to predict on this normalized scale. `norm.denormalize(out)` reverses the transformation:

```text
prediction in original units = out * training MAD + training mean
```

`raw_target_for` retrieves the unnormalized ground truth. Validation therefore compares predictions and targets in their original units: eV for `gap`.

`b.num_graphs` is the number of molecules in the batch, not the number of atoms. Since `L1Loss` returns a batch mean, multiplying it by `b.num_graphs` gives that batch's total absolute error. We accumulate this error and the molecule count across batches, then divide to get the validation MAE. This correctly weights a smaller final batch too.

We record the learning rate used for the current epoch, step the scheduler for the next epoch, and log the validation MAE and current learning rate. If validation improves, we save the checkpoint and reset the patience counter. Otherwise, we increment the counter and stop when it reaches `early_stop_patience` (`>=`, not strictly greater). Finally, the function returns the run ID, run directory, and best validation MAE.

### `main` — command-line entry point

We define the command-line arguments and their defaults with `add_argument`, then read the supplied arguments using `parse_args()`. We pass those values into `train`, which returns `run_id`, `run_dir`, and `best_val`. Only the run ID is printed.

The final `if __name__ == "__main__":` calls `main()` when we run this file directly. Importing the file does not automatically start training.

## Evaluation notes

### `evaluate` — score a saved run

After the imports, we define the paper directory and a set of energy target names called `EV_TO_MEV_TARGETS`. These targets are stored in eV by QM9, while the paper reports their errors in meV.

We pass the run ID, runs base, paper directory, and data root into `evaluate`. It loads the run's configuration dictionary from `runs/<run_id>/config.json`, reconstructs the EGNN with the saved dimensions and model settings, loads its checkpoint, and switches to evaluation mode.

We recreate the data split using the saved target and seed. This gives us the same held-out test molecules and training-set normalization statistics. The metric comes from the saved config: `"metric": "mae"` selects mean absolute error from the metric registry.

We initialize the total error and molecule count to zero. Inside `torch.no_grad()`, which disables gradient tracking, we loop through the test batches, get the model output, denormalize the predictions, and retrieve the raw targets. We add each batch's mean absolute error multiplied by its molecule count (`b.num_graphs`) to the total error, and accumulate the molecule count. Finally, we return the metric name and total error divided by the total number of molecules.

### Evaluation `main`

We define and parse the command-line arguments, load the config, and get the target from it. Calling `evaluate` returns the metric name and score. We print the metric name, target, and score in the dataset's units. If the target belongs to the `EV_TO_MEV_TARGETS` set, we also print the score multiplied by 1,000, in meV, for comparison with the paper.

### eV, meV, and the training scale

I initially thought eV was the smaller unit. Correction: milli means one-thousandth, so meV is smaller, like millimetres compared with metres:

```text
1 eV = 1000 meV
0.03 eV = 30 meV
```

For `gap`, the dataset stores targets in eV, but training uses dimensionless normalized targets: `(target - training mean) / training MAD`. Validation and testing reverse the normalization to measure error in eV; the extra reporting step converts that error to meV.

We could store the values in meV throughout and get the same normalized training targets. Multiplying the target, mean, and MAD by 1,000 cancels out in the normalization formula. Keeping the dataset's native units and converting only for reporting puts the conversion in one explicit place.

### Saved-model evaluation and wrap-up

We finished the walkthrough of `data.py`, `model.py`, `train.py`, and `evaluate.py`, then evaluated the saved checkpoint from run `20260716-082300` on its held-out test split:

```bash
UV_CACHE_DIR=/private/tmp/gdl-papers-uv-cache uv run --no-sync python papers/01-EGNN/src/evaluate.py --run 20260716-082300
```

The evaluation returned a `gap` test MAE of **0.0485 eV = 48.50 meV**. [Table 3 of the paper](https://proceedings.mlr.press/v139/satorras21a/satorras21a.pdf#page=8) reports **48 meV** for EGNN on the same property. Our result is 0.50 meV higher, about 1%: a close result on our seeded split. This is one checkpoint and one target; it does not establish reproduction of the full benchmark or identical dataset partitions.

This wraps up our EGNN code walkthrough and saved-model evaluation. The explanations and dimension checks are collected here for the later writeup.

## Blog idea

Working title: **Understanding EGNN by Following the Tensors**.

The story is working from the equations through the implementation to a saved model with a `gap` test MAE of 48.50 meV, close to the paper's 48 meV.

Possible article flow:

1. The idea: predict a molecular property independently of rotations and translations of the molecule.
2. Turn molecules into tensors: charge-power features, fully connected edges, batching, and concrete shapes.
3. Follow one message-passing layer from equation to code to shapes. Use the 8-atom, 26-edge example as the centerpiece.
4. Explain the surprise: the QM9 model keeps coordinates fixed, using distances to supply the geometry.
5. Cover normalization, validation, saved checkpoints, and the evaluated result, including the dataset-split caveat.

Keep a few misunderstandings as moments where things clicked: edge ordering versus node ordering, scalar attention gates versus vector messages, and eV versus meV. Aim for one focused article explaining the model, linking to the repository for the full training details.
