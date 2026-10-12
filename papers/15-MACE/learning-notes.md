# MACE learning notes

## ASE basics for loading 3BPA

ASE stands for **Atomic Simulation Environment**. Its `Atoms` object holds
the elements, positions, simulation cell, boundary flags, and attached results
for an atomic structure.

### Read a frame

```python
from ase import Atoms
from ase.io import read

frame = read(DATA_PATH, index=0, format="extxyz")
all_frames = read(DATA_PATH, index=":", format="extxyz")
```

`extxyz` means **extended XYZ**. It stores element symbols and XYZ positions,
with extra metadata and per-atom properties such as forces. Multiple frames
can appear consecutively in one file.

`index=0` selects the first frame. `index=":"` selects all frames and returns
a list. In this training file, `len(all_frames)` is 500, while `len(frame)`
is 27: 500 geometries of the same 27-atom molecule. Positions, energy, and
forces vary between snapshots; the atoms and their ordering stay fixed.

### Inspect an `Atoms` object

| Expression | Meaning |
|---|---|
| `frame.symbols` | Element symbols in stored atom order; this sequence does not specify bonds. |
| `frame.numbers` | Atomic numbers: H = 1, C = 6, N = 7, O = 8. |
| `frame.positions` | Position array `(27, 3)`, one XYZ row per atom, in angstroms. |
| `frame.cell` | Simulation-cell vectors; this file uses a 50-angstrom cube. |
| `frame.pbc` | Periodic boundary flags; `[False, False, False]` means no periodic copies. |
| `frame.arrays.keys()` | Names of per-atom arrays; here, `numbers` and `positions`. |
| `frame.info.keys()` | Names of extra frame metadata; here, `dihedrals`. |

`frame.info["dihedrals"]` contains three selected torsion angles in degrees.
Each describes a four-atom group: A-B-C defines one plane, B-C-D another,
and the torsion measures their relative twist around B-C. These are three
values for the snapshot, rather than one value per atom. Selected groups
can share atoms while measuring different central bonds.

### Retrieve energy and forces

```python
energy = frame.get_potential_energy()
forces = frame.get_forces()
```

In this loaded dataset, these methods retrieve stored labels through an
attached single-point calculator. They do not run a new simulation.

| Quantity | Shape | Units |
|---|---|---|
| Energy of one frame | Scalar | eV |
| Forces of one frame | `(27, 3)` | eV/angstrom |
| Energies of all frames | `(500,)` | eV |
| Force magnitudes of all frames | `(500, 27)` | eV/angstrom |

### Force magnitude: the useful correction

I initially treated the mean of squared force components as the force norm.
The norm is the geometric length of one atom's force vector:

$$
\|\mathbf F_a\| = \sqrt{F_{ax}^2 + F_{ay}^2 + F_{az}^2}.
$$

For `(3, 4, 0)`, the number of components is 3 and the magnitude is 5.
The mean squared component is `25/3`; it has squared force units.
There are 27 separate force magnitudes per snapshot. A correct array shape
does not establish that the entries contain the intended quantity.

## `load_extxyz`: from ASE frames to tensor configurations

Code: [data.py](src/data.py), `AtomicConfiguration` and `load_extxyz`.

This is similar to the notebook: I read every frame from the path, create an
empty list, loop through the frames, and convert their atomic numbers,
positions, energy, and forces into tensors. Each iteration creates one
`AtomicConfiguration` instance and appends it to the list. The function
returns all 500 records for the training file.

I wasn't sure why we needed configurations. Each record keeps one snapshot's
four tensors together, so its geometry stays paired with the correct energy
and force labels. Atomic numbers and positions are model inputs; energy and
forces are training targets. The class is already defined above the loader.

| Field in one configuration | Shape | Dtype |
|---|---|---|
| `atomic_numbers` | `(27,)` | `torch.long` |
| `positions` | `(27, 3)` | `torch.float64` |
| `energy` | Scalar, `()` | `torch.float64` |
| `forces` | `(27, 3)` | `torch.float64` |

Atomic numbers are discrete element identifiers, so they use an integer
dtype. Positions, energy, and forces use floating-point values; this
reproduction uses 64-bit floats.

My hunch about device transfer was right: the loader converts the records
to tensors, and training later moves each graph batch onto the selected
device with `batch.to(device)` in [train.py](src/train.py). Between these
steps, the data pipeline splits configurations, builds neighbour graphs,
and batches them. Dtype describes how a tensor stores its numbers; device
describes where the tensor lives, such as CPU or GPU.

## `AtomicConfiguration` and `AtomicGraph`

Code: [data.py](src/data.py), the two classes above `load_extxyz`.

`AtomicConfiguration` feels like a C++ `struct`: it groups atomic numbers,
positions, energy, and forces into one record. `@dataclass` generates its
constructor. `frozen=True` blocks normal reassignment of fields and adding
attributes after construction. The tensors inside the record can still
change in place. The `Tensor` annotations describe expected types; Python
does not automatically enforce those types or the documented shapes.

`AtomicGraph` represents one frame as a PyG graph. It inherits from `Data`,
initializes that graph container with `super().__init__()`, and stores the
tensors supplied by its caller. The constructor sets `num_nodes` from
`positions.shape[0]`, which is the number of atoms. It also stores the
connections in `edge_index`: row 0 contains senders, row 1 receivers.

I initially associated `node_attrs` with charge or the raw atomic number Z.
Here it contains a one-hot encoding of the element, in H/C/N/O order:
H is `[1, 0, 0, 0]`, C is `[0, 1, 0, 0]`, N is `[0, 0, 1, 0]`, and
O is `[0, 0, 0, 1]`. Atomic numbers and node attributes encode element
identity in different forms. Electrical charge would be a separate quantity.

`@property` makes `graph.node_attrs` call the getter without parentheses.
The getter returns the tensor held in `self["node_attrs"]`. This class
defines the getter; it does not define a property setter.

### Shape check: one graph

For the practice example with **27 atoms, four element types, and 146
directed edges**:

| Tensor | Shape | What each entry describes |
|---|---|---|
| `atomic_numbers` | `(27,)` | One atomic number per atom |
| `node_attrs` | `(27, 4)` | One four-component element encoding per atom |
| `positions` | `(27, 3)` | One XYZ position per atom |
| `edge_index` | `(2, 146)` | One sender/receiver pair per directed edge |
| `energy` | Scalar, `()` | Total energy of this molecular snapshot |
| `forces` | `(27, 3)` | One XYZ force vector per atom |

I initially gave positions only the atom axis; each atom also needs its
three coordinates. I also corrected my energy answer: energy belongs to
the entire molecular snapshot, so it is one scalar. The 27 atoms each have
their own positions and forces, while the graph has one total energy.

## `load_isolated_atom_energies`: element energy references

Code: [data.py](src/data.py), `load_isolated_atom_energies`.

I read all the frames from `iso_atoms.xyz` and create an energy dictionary
mapping an integer atomic number to a scalar energy tensor. The loop visits
one frame at a time. The `len(atoms) != 1` check verifies that the frame
contains exactly one isolated atom; the loop itself does not enforce that.
`atoms.get_atomic_numbers()[0]` selects that frame's sole atomic number.

The loader rejects unsupported elements and duplicate records, then stores
each atom's potential energy as a float64 tensor under its atomic-number key.
After reading, it checks that every required element has a reference energy.
Missing entries raise an error.

The dictionary lets me retrieve energies in a fixed order even when the
file uses another order. This file begins with C, then H. The final
`torch.stack` follows `ATOMIC_NUMBERS = (1, 6, 7, 8)` and returns
`[energy_H, energy_C, energy_N, energy_O]`: a float64 tensor of shape `(4,)`.
This is also the H/C/N/O order used by the graph's one-hot element channels.
These energies supply the model's isolated-atom energy baseline.

## `split_train_valid`: a seeded train/validation split

Code: [data.py](src/data.py), `split_train_valid`.

I pass a list of configurations, a validation size, and a seed. The test
sets come from separate files, so this function splits the training
collection into training and validation records.

The function creates a NumPy generator with `np.random.default_rng(seed=seed)`.
I initially described this as `np.random.seed(seed)`, but that older call
sets global RNG state and returns `None`. `default_rng` returns the generator
used here. Its `permutation` creates a shuffled ordering of every configuration
index, without repetitions. The same input length and seed reproduce the split.

The training size is **total configurations minus validation size**:
`500 - 50 = 450`. The first 450 shuffled indices select training records,
and the remaining 50 select validation records. The list comprehensions
return the records themselves, keeping each snapshot's geometry and labels
together. The function returns the two lists as `(train, valid)`.

### What `T` means

`T = TypeVar("T")` defines a type variable: a placeholder for whatever kind
of item the input list contains. `list[T]` as input and
`tuple[list[T], list[T]]` as output describe splitting one list into two
lists of the same item type. Here, the items are `AtomicConfiguration`
objects. The same function could split integers or other records.
This annotation helps type checkers; it does not perform the split or
convert the items.

## `build_neighbor_graph`: selecting directed edges

Code: [data.py](src/data.py), `build_neighbor_graph`.

The input is one `AtomicConfiguration` and a distance cutoff, defaulting to
5 angstroms. I create empty sender and receiver lists. The outer loop visits
every atom as a sender, and the inner loop visits every atom as a receiver.
Both use `configuration.positions.shape[0]`, the number of atoms.

When sender and receiver are equal, I skip the pair because we exclude
self-edges. Otherwise, I subtract their positions and take the vector norm
to obtain their Euclidean distance. If that distance is strictly less than
the cutoff, I append the sender and receiver to their respective lists.
Entries at the same list index stay matched as one directed edge.

The loops visit ordered pairs. If atoms 3 and 7 are within the cutoff,
they create both `3 -> 7` and `7 -> 3`. Connections depend on distance,
rather than a list of chemical bonds. A pair exactly at the cutoff is
excluded by the strict comparison.

Finally, `torch.tensor([senders, receivers], dtype=torch.long)` forms
`edge_index` with shape `(2, E)`: row 0 contains senders and row 1 contains
receivers. These distances select the topology. The graph stores the
indices, and the model later derives edge vectors and lengths from its
live positions during the forward pass.

### Element indices, one-hot attributes, and the returned graph

After creating `edge_index`, I build a tensor containing one element index
per atom. `ATOMIC_NUMBER_TO_INDEX` maps H/C/N/O atomic numbers `1/6/7/8`
to indices `0/1/2/3`. I initially said the indices might run from 0 to 4;
four element types require indices from 0 to 3.

The loop preserves the atoms' existing order; it does not sort their atomic
numbers. For example, `[6, 1, 8, 7]` becomes `[1, 0, 3, 2]`. Each
`x.item()` extracts the scalar tensor's Python value for the dictionary lookup.
At this point, `node_attrs` has shape `(27,)`.

One-hot encoding expands those indices into four H/C/N/O channels per atom,
giving shape `(27, 4)`. `.to(torch.float64)` sets the encoding's dtype.
The final `AtomicGraph` bundles atomic numbers, positions, `edge_index`,
energy, forces, and these one-hot node attributes. The original atomic
numbers remain available alongside their encoded representation.

## `get_data_loader`: batching atomic graphs

I pass the dataset, batch size, shuffle flag, and `drop_last` flag into the
loader. The dataset is a `Sequence[AtomicGraph]`: an ordered, indexed
collection such as a list or tuple of graphs. Batch size sets how many
configurations go into each batch, and shuffle controls their ordering.

I initially called this a native PyTorch loader. The import comes from
`torch_geometric.loader`: PyTorch Geometric builds on PyTorch's loader and
handles graph batching. It concatenates atom tensors, offsets edge indices,
and creates a `batch` tensor recording which graph each atom belongs to.
The resulting graph contains disconnected molecular graphs.

For two 27-atom graphs, positions and forces have shape `(54, 3)`, node
attributes have shape `(54, 4)`, and energies have shape `(2,)`. The
`batch` tensor has shape `(54,)`, with 27 zeros followed by 27 ones.
An edge `3 -> 7` in the second graph becomes `30 -> 34` in the batch.

`drop_last=True` drops the final batch only when it is incomplete. With
500 graphs and batch size 32, the loader produces 15 full batches and a
final batch of 20; that final batch is dropped when the flag is true.

## `validate_data_dir`: checking the expected dataset files

I create a `required` set containing the training and isolated-atom
filenames, then add every test filename. `TEST_FILES` is the dictionary:
it maps test categories such as `"300K"` and `"dihedral"` to tuples of
filenames. In the nested iteration, `names` is one tuple and `name` is
one filename inside it.

I initially described `missing` as names outside `required`. It actually
checks every name already in `required` and collects those whose expected
path is not a file. `(data_dir / name)` joins the directory and filename;
`.is_file()` checks that path. `sorted` gives the missing names a stable,
readable order.

If any files are missing, `", ".join(missing)` creates a comma-separated
list and the function raises `FileNotFoundError` with that list in its
message. Raising the exception stops normal execution; an uncaught
exception displays the message in its traceback. This check validates
file presence. It does not inspect file contents.

## `MACEConfig`: the model settings

Code: [model.py](src/model.py), `MACEConfig`.

`MACEConfig` is a frozen dataclass holding the architecture settings.
`num_elements=4` refers to the supported species H, C, N, and O.
`channels=256` gives 256 learned feature channels for each angular type.
I initially associated channels directly with the radial network's output
width; channels also organize the model's atomic features.

I initially mixed up interaction count and correlation order.
`num_interactions=2` means two successive message-passing stages.
`correlation=3` means products of up to three neighborhood-feature factors.
In the first stage, these include up to four-body structure: the central
atom and three neighbors. Repeated neighbor indices also contribute
lower-body-order terms. Later stages use features that already contain
neighborhood information, so their dependence on positions can involve
more atoms.

| Setting | Meaning |
|---|---|
| `max_ell=2` | Retained hidden features use angular orders 0, 1, and 2. |
| `spherical_harmonics_max_ell=3` | Directional edge features use spherical harmonics through order 3. |
| `cutoff=5.0` | The neighborhood radius is 5 angstroms. |
| `num_radial_basis=8` | Each edge distance becomes eight radial basis values, which feed the radial MLP. |
| `envelope_power=5` | A dimensionless parameter controlling the smooth cutoff polynomial's shape. |
| `radial_mlp_widths=(64, 64, 64, 1024)` | The first radial MLP has three hidden layers of width 64 and an output width of 1024. |
| `readout_hidden_width=16` | The final nonlinear energy readout uses 16 hidden scalar features. |

The first radial MLP follows `8 -> 64 -> 64 -> 64 -> 1024`.
Its 1024 outputs are distance-dependent weights for 256 channels across
four angular orders, 0 through 3. The second interaction's radial MLP
uses the same three hidden widths but determines its output width from
the allowed tensor-product routes.

My intuition about the envelope gradually reducing contributions near
the cutoff was right. The envelope makes radial features zero at and
beyond 5 angstroms. I initially interpreted `envelope_power=5` as another
distance setting; the radius comes from `cutoff`, while the power controls
the polynomial shape.

The readouts produce energy contributions per atom. The model combines
the two stages' contributions and sums atoms to obtain molecular energy.
Forces come from differentiating that energy with respect to positions.

### `__post_init__`: checking values after construction

I guessed that `__post_init__` happens after initialization. `@dataclass`
generates `__init__` from the declared fields, so the constructor is already
provided even though I do not see its body in the source. `MACEConfig()`
uses the field defaults; `MACEConfig(channels=128)` overrides one default.
The generated constructor assigns the fields, then calls `__post_init__`
to check their values.

The checks require positive element and channel counts, exactly two
interactions and correlation order three, output angular order two and
spherical-harmonic order three, and positive cutoff and radial-basis count.
An unsupported value raises `ValueError`. Defaults do not prevent callers
from supplying other values; these checks enforce the supported settings.
For example, `MACEConfig(num_interactions=3)` is rejected.

My description of `num_interactions` as the number of layers is right
when I mean interaction or message-passing layers. Each stage gathers
neighbor information and processes atomic features through several
operations. Two interaction stages therefore contain more than two
linear neural-network layers. Correlation order separately controls the
feature-product order within each stage.

### `hidden_irreps`: describing the hidden features

`hidden_irreps` is a custom property of `MACEConfig`, accessed as
`config.hidden_irreps` without parentheses. It returns a string describing
the feature types; it does not print or return the feature tensor itself.
The model passes this description to `o3.Irreps` to configure its layers.
“Irreps” means irreducible representations: basic types of features with
specified transformation rules.

With 256 channels, the description is
`"256x0e + 256x1o + 256x2e"`. I initially treated 0, 1, and 2 as channel
counts. They are angular orders; 256 is the number of copies of each type.

| Block | Feature type | Components per copy | Total components |
|---|---|---|---|
| `256x0e` | Even scalar, angular order 0 | 1 | 256 |
| `256x1o` | Odd vector, angular order 1 | 3 | 768 |
| `256x2e` | Even order-2 feature | 5 | 1280 |

The `+` combines these feature blocks; it does not add their values
numerically. Together they give 2304 components per atom.

### Even and odd: what happens when space is inverted?

Spatial inversion sends every position `(x, y, z)` to `(-x, -y, -z)`.
An **even** feature stays unchanged under that operation; an **odd**
feature changes sign. These labels describe transformation behavior,
rather than whether a value is positive or negative.

A distance or molecular energy is even. A displacement or force vector
is odd. Products such as `xy`, `xz`, and `x² - y²` are even because both
coordinate signs reverse and cancel. Such quadratic expressions help
illustrate order-2 features. A `2e` feature can still change under an
ordinary rotation: “even” only specifies its response to inversion.

My sine/cosine comparison captures the same idea:
`cos(-x) = cos(x)` and `sin(-x) = -sin(x)`. Here, however, we invert
the entire atomic geometry, and the features need not be sine or cosine.

### Why the sequence starts with `0e` and alternates

Ordinary spherical harmonics obey
`Y_l^m(-r̂) = (-1)^l Y_l^m(r̂)`. Therefore their natural parity is
even for even angular order and odd for odd angular order:
`0e, 1o, 2e, 3o, ...`.

The intuition is that order 0 is constant, order 1 is linear in direction
coordinates, and higher orders have the corresponding polynomial degree.
Reversing every coordinate gives one minus sign per factor: an even number
of factors cancels, while an odd number leaves a minus sign.

This configuration chooses `0e + 1o + 2e`, following that natural-parity
pattern. Angular order and parity are independent labels in general:
`0o` is a valid odd scalar, and `1e` is a valid even axial vector, such as
a cross product of two displacement vectors. We choose the letter from
the feature's transformation rule; alternation is not compulsory for
every possible hidden representation.

## `edge_vectors_and_lengths`: geometry for every edge

Code: [model.py](src/model.py), `edge_vectors_and_lengths`.

The atomic graph supplies `positions` with shape `(N, 3)` and
`edge_index` with shape `(2, E)`. Transposing `edge_index` gives `(E, 2)`,
so each loop iteration extracts one sender and receiver pair.

For each edge, I subtract the sender's position from the receiver's:

\[
\mathbf v = \mathbf R_{\mathrm{receiver}} - \mathbf R_{\mathrm{sender}},
\qquad
d = \lVert\mathbf v\rVert_2 = \sqrt{v_x^2 + v_y^2 + v_z^2}.
\]

I initially described the vector as pointing from receiver to sender.
Destination minus origin points from **sender to receiver**. For a sender
at `(1, 0, 0)` and receiver at `(4, 0, 0)`, the vector is `(3, 0, 0)`
and its length is `3`.

The function keeps two lists: `edge_vectors` collects the three-component
displacements, and `edge_distances` collects their scalar L2 norms.
`torch.stack` combines the tensors in each list into one tensor.

| Output | Shape | Contents |
|---|---|---|
| `edge_vectors_tensor` | `(E, 3)` | One displacement vector per edge. |
| `edge_distances_tensor` | `(E, 1)` | One distance per edge. |

Stacking the scalar distances first gives `(E,)`; `reshape(-1, 1)` adds
the single column, with PyTorch inferring the number of rows.
The function returns the vectors first and distances second.

These operations preserve the differentiation path from edge geometry
back to `positions`. The model can therefore use this geometry to predict
energy and differentiate that energy with respect to positions for forces.

## `forces_from_energy`: differentiating energy into forces

Code: [model.py](src/model.py), `forces_from_energy`.

My starting point was right: each atom's force is the negative gradient
of energy with respect to that atom's position.

\[
\mathbf F_i = -\frac{\partial E}{\partial\mathbf R_i}
= -\left(\frac{\partial E}{\partial x_i},
\frac{\partial E}{\partial y_i},
\frac{\partial E}{\partial z_i}\right).
\]

The function calls `torch.autograd.grad(energy.sum(), positions,
create_graph=training, allow_unused=True)`. It returns a tuple with one
gradient per input tensor; `(gradient,)` unpacks the gradient for
`positions`. The gradient and returned forces both have shape `(N, 3)`.

### Why sum the energies in a batch?

The model has already summed atomic energy contributions within each
configuration. Here, `energy` has shape `(B,)`, with one total energy per
configuration. `energy.sum()` adds those totals into one scalar for
differentiation.

Configurations remain independent because the batched graph has no edges
between them. For configurations A and B:

\[
\frac{\partial(E_A(\mathbf R_A)+E_B(\mathbf R_B))}
{\partial\mathbf R_{A,i}}
= \frac{\partial E_A}{\partial\mathbf R_{A,i}},
\]

because B's energy does not depend on A's positions. One differentiation
call therefore computes the separate force vectors for every atom.

Overlapping numerical coordinates across configurations do not connect
them. Their positions occupy separate entries in the tensor and their
atoms have separate indices. For example, with `E_A = x_A²`, `E_B = x_B²`,
and both coordinates equal to 2, differentiating their sum returns
`(2*x_A, 2*x_B) = (4, 4)`. Each derivative still belongs to its own atom.

### Training through the force calculation

`create_graph=training` records the derivative calculation during
training. A loss on predicted forces can then differentiate through
those forces to adjust the model's weights. During evaluation, setting
it to false still computes forces without building that additional
derivative graph. Positions must already participate in a differentiable
energy calculation; the model enables their gradients before its forward
calculations.

`allow_unused=True` permits an input tensor with no computation path to
the output. If the energy calculation never used `positions`, its returned
gradient can be `None`, and the function supplies `torch.zeros_like(positions)`.
Those zeros have the same shape, dtype, and device as positions. This
fallback does not refer to the first training iteration. A connected
calculation with a zero derivative normally returns a zero tensor.
The option also does not enable gradients on an output that lacks them.

### Summing force errors for weight updates

Each atom retains one three-component force vector. A simple force
mean-squared-error objective combines the errors into a scalar:

\[
L_F = \frac{1}{3N}\sum_{i=1}^{N}\sum_{\alpha\in\{x,y,z\}}
\left(F^{\mathrm{pred}}_{i,\alpha}
- F^{\mathrm{target}}_{i,\alpha}\right)^2.
\]

Here, `i` visits every atom in the batch, and `alpha` visits the x, y,
and z force components for that atom. The inner sum adds three squared
component errors; the outer sum adds the results across atoms. For one
27-atom configuration, the mean divides 81 squared errors by 81.
This illustrates component-wise MSE; the training recipe specifies its
actual loss weights and normalization.

| Quantity | Shape |
|---|---|
| Predicted or target forces | `(N, 3)` |
| Squared component errors | `(N, 3)` |
| Mean force loss | Scalar |

Summing squared errors produces the loss while preserving the individual
force predictions. Summing raw force errors would let opposite errors
cancel. Backpropagation differentiates the loss through predicted forces
and energy to obtain weight gradients; the optimizer uses those gradients
to update the weights. A combined objective can also include an energy
loss, `L = lambda_E * L_E + lambda_F * L_F`.

## `CouplingRoute`: recording an angular combination

Code: [model.py](src/model.py), `CouplingRoute`.

I recognized this as a dataclass storing input, output, and intermediate
angular orders. It describes one allowed tensor-product route for
combining angular features.

| Field | Meaning |
|---|---|
| `input_ells` | Tuple of angular orders of the input factors. |
| `output_ell` | Angular order of the final result. |
| `intermediate_ells` | Tuple of angular orders produced between the inputs and final result. |

For example, `CouplingRoute(input_ells=(1, 1, 1), output_ell=1,
intermediate_ells=(2,))` describes combining the first two order-1
features into order 2, then combining that intermediate feature with
the third order-1 input to produce order 1. A two-factor product needs
no intermediate combination, so `intermediate_ells` is empty.

`frozen=True` prevents reassignment of the fields after construction.
The stored values describe a route; other code performs the feature
calculations.

My connection to atomic-basis construction was close: angular coupling
appears when building products of atomic-basis features, which relates
to the product basis. In this source, `CouplingRoute` supports
`enumerate_coupling_routes`, a pedagogical SO(3) helper outside the final
model execution path. The model uses `u_matrix_real` for its actual
O(3) coupling calculations, including parity. This route description
records angular orders only.

## `MACE.forward`: connecting the complete model

Code: [model.py](src/model.py), `MACE.forward`.

I traced the inputs as one-hot node attributes, positions, edge indices,
and graph indices. `graph_index` assigns each atom to its configuration.
The model enables position gradients so its predicted energy can produce
forces by differentiation.

The element embedding turns one-hot species into learned node features.
Edge geometry supplies displacements and distances, and the radial basis
turns each distance into eight values. The first atomic basis produces
`A0`, which the first product-basis block processes into `h1`. Its `None`
argument means there is no skip contribution in this first update.

The first readout produces `E1`. The second interaction uses `h1` to
produce `A1` and the skip contribution `sc`. The second product-basis
block combines these into `h2`, and the second readout produces `E2`.
Both readouts produce one scalar contribution per atom.

| Quantity | Shape with the default configuration |
|---|---|
| `node_attrs` | `(N, 4)` |
| Initial node embedding | `(N, 256)` |
| `edge_radial` | `(E, 8)` |
| `A0`, `A1` | `(N, 4096)` |
| `h1` | `(N, 2304)` |
| `h2`, `sc` in the final stage | `(N, 256)` |
| `E1`, `E2`, restored per-atom energy | `(N, 1)` |
| Configuration energies `E_g` | `(B,)` |
| Forces `F_g` | `(N, 3)` |

Here `N` counts atoms in the batch, `E` counts edges, and `B` counts
configurations. We study the blocks in this execution order, opening their
helpers when needed. The pedagogical route enumerator is optional.

### Restoring the energy reference and scale

The exact expression is:

```python
E = self.rms_forces * (E1 + E2) + self.mean_energy_shift + atomic_baseline
```

Only the learned readout sum is multiplied by `rms_forces`.
The model predicts a deviation around a typical reference energy:
isolated-atom baseline plus typical residual correction plus scaled
learned deviation.

`node_attrs @ atomic_energies.unsqueeze(-1)` selects the isolated-atom
reference for each atom's species. A carbon one-hot row selects the
carbon reference. This gives `(N, 1)`; the sum over atoms happens later.

`mean_energy_shift` is the training-set average residual energy per atom.
For each training configuration, subtract its isolated-atom energy sum
from its reference energy and divide by its atom count, then average
across configurations. The forward calculation adds this mean back.
When the learned readouts are zero, the prediction is the baseline plus
this typical correction.

For a made-up single-atom example, take baseline `-100`, mean shift `-2`,
scale `4`, `E1=0.25`, and `E2=0.50`. The restored energy is
`4 * (0.25 + 0.50) - 2 - 100 = -99`. The network supplies a deviation
of `+3` from the reference-plus-mean value `-102`.

### What `rms_forces` measures

RMS means root mean square. The training statistics collect all three
reference force components for every training atom, square the values,
average them, then take the square root. For `M` training atoms:

\[
s_F = \sqrt{\frac{1}{3M}\sum_{i=1}^{M}
\left(F_{i,x}^2+F_{i,y}^2+F_{i,z}^2\right)}.
\]

My earlier two-component illustration was misleading: every atomic force
has three components. For one atom with force `(3, 4, 0)`, component RMS
is `sqrt((9 + 16 + 0) / 3)`, approximately `2.89`; its vector magnitude
is `sqrt(9 + 16 + 0) = 5`. The dataset RMS averages all training components.

This is a fixed scale computed from reference forces, not a prediction
error. The mean shift and RMS are calculated in
[train.py](src/train.py), `compute_mean_rms_energy_forces`.
Because the baseline and shift do not depend on positions, only the
scaled learned correction contributes to forces:
`F = -rms_forces * gradient_positions((E1 + E2).sum())`.
Training adjusts the weights producing the readout values.

### Summing atoms into configurations

`E.new_zeros((graph_index.max() + 1, E.shape[1]))` creates a `(B, 1)`
accumulator with the per-atom energy tensor's dtype and device. Graph
indices run from `0` to `B-1`.

`E_g.index_add_(0, graph_index, E)` adds each atom's energy to its own
configuration's row. For atom energies `[10, 20, 30, 40]` and graph IDs
`[0, 0, 1, 1]`, the two configuration totals are `[30, 70]`.
`squeeze(-1)` removes the one-column axis, changing `(B, 1)` into `(B,)`.
Finally, `forces_from_energy` produces `(N, 3)` forces, and the model
returns configuration energies and atomic forces.

## `MACE.__init__`: creating the model's blocks

Code: [model.py](src/model.py), `MACE.__init__`.

The constructor receives `MACEConfig`, the average neighbor count,
isolated-atom energies, mean residual-energy shift, and force RMS scale.
It saves the configuration and neighbor average, then creates the modules
that `forward` uses.

My guess about the neighbor average was right: `train.py` calculates it
from incoming-edge counts in the training graphs. The second interaction
uses this fixed value to normalize its summed neighbor messages.

`atomic_energies` contains one isolated-atom reference per supported
species, loaded from the isolated-atoms extxyz file. If none are supplied,
the constructor defaults to a zero vector of length `num_elements`.
It checks the vector's shape and requires a positive `rms_forces`.

The three tensors `atomic_energies`, `mean_energy_shift`, and `rms_forces`
are registered as float64 buffers. Buffers belong to the model's saved
state and move with its device. The optimizer does not treat them as
learnable parameters. This supports checkpointing and device movement,
beyond simply making the values easy to retrieve.

### Species embedding and radial-module setup

`o3.Linear("4x0e", "256x0e")` maps one-hot element
identity into learned scalar features using e3nn's initialization and
normalization conventions. I guessed 128 channels; the default
is 256, so `(N, 4)` becomes `(N, 256)`. Each one-hot row selects the learned
weight-matrix column for its species. Initially, atoms of the same species
receive the same embedding; the later blocks incorporate geometry.
This supplies the `W_(k,z_j)` species factor in equation 9.

`BesselRadialBasis(...)` creates a module instance with eight basis
functions, cutoff `5.0` angstroms, and envelope power `5`. The power is
dimensionless and controls the cutoff polynomial's shape. Its later
forward call maps edge distances `(E, 1)` into radial features `(E, 8)`.

### Mapping the remaining blocks to the paper

| Block | Equations | Role |
|---|---|---|
| `first_atomic_basis` | 9 | Build the first atomic basis from scalar species features and edge geometry. |
| `first_product_basis` | 10-12 | Couple products, combine their route contributions, and update hidden features. |
| `first_readout` | 13 | Produce a linear energy contribution per atom. |
| `second_interaction_block` | 8 and the skip term for 12 | Build a new atomic basis from the previous hidden features and prepare `sc`. |
| `second_product_basis` | 10-12 | Apply the product-basis update and add the supplied skip contribution. |
| `second_readout` | 13 | Produce the final nonlinear energy contribution per atom. |

I initially associated the second interaction with equation 10 and the
product basis only with equation 11. The interaction builds `A`; the
product block handles coupled products `B`, message `m`, and updated `h`.

### The nonlinear final readout

The second readout is an `nn.Sequential` containing:

```text
256x0e          -> equivariant linear -> 16x0e
                -> normalized SiLU   -> 16x0e
                -> equivariant linear -> 1x0e
```

The nonlinearity sits between the two linear layers. `16x0e` means 16 even
scalar features; `1x0e` means one even scalar energy contribution per atom.
The e3nn activation applies approximately `1.67918 * SiLU(x)` to these
even scalars. The fixed factor normalizes the expected squared output for
standard-normal inputs. The next linear layer can absorb the factor, so
it adds no expressive power, but it affects initialization and optimization.
We retain it to match the author's convention.

`readout_hidden_width=16` is a hyperparameter specified in the paper's
training setup. The paper gives no particular justification for choosing
16. Other hidden scalar widths would preserve symmetry while changing
capacity and parameter count. We use 16 to match the published recipe.
It is independent of the 16 spherical-harmonic components through order
3, whose count comes from `1 + 3 + 5 + 7`.

## `BesselRadialBasis`: eight smooth distance features

Code: [model.py](src/model.py), `BesselRadialBasis`.

The constructor stores `num_basis=8`, `cutoff=5.0` angstroms, and
`power=5`. The forward method receives edge distances `(E, 1)` and
returns `(E, 8)`: one row per edge and one column per basis function.

`n` contains the basis indices `[1, 2, ..., 8]` and matches the distances'
device and dtype. `zero_distances = (distances == 0).reshape(-1)` creates
an `(E,)` boolean mask without changing the distance values. I initially
mixed this comparison up with `zeros_like`, which creates numeric zeros.

`torch.zeros_like(distances).repeat(1, num_basis)` creates an `(E, 8)`
output array to fill. `tensor_2` stores the constant 2 with the matching
device and dtype. The zero-distance mask selects entire edge rows;
the trailing `:` selects all eight feature columns.

### Nonzero distances and the zero-distance limit

For a nonzero distance, each basis index gives:

\[
b_n(r)=\sqrt{\frac{2}{r_c}}\,
\frac{\sin(n\pi r/r_c)}{r}.
\]

My initial verbal formula misplaced the factors. The square root covers
`2 / cutoff`; the sine is divided by the edge distance `r`.
For zero distances, direct evaluation would give `0/0`. The separate
branch uses the finite limit:

\[
\lim_{r\to0}\frac{\sin(ar)}{r}=a,
\qquad
b_n(0)=\sqrt{\frac{2}{r_c}}\,\frac{n\pi}{r_c}.
\]

My interpretation of the zero-distance branch as a limit was correct.
Each selected row receives eight values, corresponding to `n=1,...,8`.

### The polynomial cutoff envelope

`r_over_cutoff = distances / cutoff` forms the dimensionless ratio
`x = r / r_c`. With `p=5`, the envelope inside the cutoff is:

\[
f(x)=1-21x^5+35x^6-15x^7.
\]

It starts at 1 for zero distance and smoothly decreases to 0 at the
cutoff. Its first and second derivatives vanish there, supporting smooth
energy and force behavior. The strict comparison `distances < cutoff`
makes the envelope zero **at and beyond** the cutoff.

The return expression multiplies the Bessel features by the envelope.
An `(E, 1)` envelope broadcasts across the eight columns of `(E, 8)`,
applying one cutoff factor to every basis value for the same edge.
Graph construction separately excludes edges outside the cutoff;
this module explicitly enforces zero radial features there as well.

## `AtomicBasis`: distance, direction, and sender features

Code: [model.py](src/model.py), `AtomicBasis.forward`.

The constructor stores the configuration and creates a scalar radial MLP:
`8 -> 64 -> 64 -> 64 -> 1024`. SiLU acts after the hidden layers; the
output layer is linear. These outputs become `(E, 256, 4)`: one learned
distance weight for each edge, channel, and angular order.
The code first reshapes to `(E, 4, 256)` and transposes the last two axes.
This assigns consecutive MLP outputs to channels within an angular order,
matching the author's ordering. Directly reshaping to `(E, 256, 4)` is also
valid when used consistently, but transferring trained weights between the
two conventions requires permuting the final MLP outputs.

The spherical harmonics have shape `(E, 16)`. `split_sizes = [1, 3, 5, 7]`
specifies block lengths, so splitting produces `(E, 1)`, `(E, 3)`,
`(E, 5)`, and `(E, 7)` for orders 0 through 3. These are not split indices.

For a chosen angular order, the edge message multiplies three factors:

| Factor | Shape | Meaning |
| --- | --- | --- |
| Selected radial weights | `(E, 256, 1)` | Learned distance weights for this order |
| Spherical-harmonic block | `(E, 1, 2*l + 1)` | Direction components for this order |
| Gathered sender features | `(E, 256, 1)` | Learned scalar species embeddings of the sending atoms |

Here the sender features come from the initial node embedding followed by
`sender_linear`, gathered
using `edge_index[0]`. They are the neighbour's features, rather than the
receiving atom's features. The later interaction block handles hidden
features. There is no additional existing `edge_message` to multiply.

Broadcasting gives `(E, 256, 2*l + 1)`. Each of the 256 channels scales
the same angular block with its own radial weight and sender embedding.
For that channel and order, the radial weight is shared across all
`2*l + 1` angular components.

This first, scalar-input atomic basis implements the special case in
Equation 9 of the more general construction in Equation 8:

\[
A_{i,k\ell m} = \sum_{j\in\mathcal N(i)}
R_{k\ell}(r_{ji})\,Y_{\ell m}(\hat{\mathbf r}_{ji})\,W_{k,z_j}.
\]

The three factors encode distance, direction, and the sender's element.
The sum gathers neighbours into each receiving atom's environment.

### Accumulating edges into atoms

For each order, the zero accumulator has shape `(N, 256, 2*l + 1)`.
`index_add_(0, receiver_index, edge_message)` adds along the atom axis:
each edge's complete message goes into its receiver's row. If two edges
have the same receiver, their messages add in that row. This changes
the leading axis from `E` edges to `N` atoms.

Each accumulated block is flattened from dimension 1 before it is
appended to `agg_messages`:

| Angular order | Accumulated shape | Flattened shape |
| --- | --- | --- |
| 0 | `(N, 256, 1)` | `(N, 256)` |
| 1 | `(N, 256, 3)` | `(N, 768)` |
| 2 | `(N, 256, 5)` | `(N, 1280)` |
| 3 | `(N, 256, 7)` | `(N, 1792)` |

Concatenating these already flattened blocks along dimension 1 produces
`(N, 4096)`, because `256 * (1 + 3 + 5 + 7) = 4096`. No further flattening
is needed. Packing preserves the order: angular block, channel, then
angular component within each channel.

Before returning A, the block applies `atomic_basis_projection`, divides by
the training-set average neighbour count, and applies `receiver_selector`.
The projection mixes the 256 channels separately within each angular order,
corresponding to the channel-mixing factor inside Equation 10. The receiver
selector uses `(N, 4)` element one-hots to select a learned 256-by-256 channel
matrix for each angular order and central species. It shares that matrix
across the magnetic components and preserves `(N, 4096)` throughout.
It does not combine features from different atoms or angular orders.
This additional species-conditioned transform comes from the author code.
The separate sender linear can be absorbed into the initial embedding;
we retain the author's parameterization after review.

## `ProductBasisBlock`: higher-order messages and hidden-feature updates

Code: [model.py](src/model.py), `ProductBasisBlock`.

The constructor receives a `MACEConfig` object and optional `output_irreps`. It creates
`SymmetricContraction` for Equations 10–11 and an equivariant linear layer
for the first term of Equation 12. `o3.Irreps(config.hidden_irreps)` declares
the layout `256x0e + 256x1o + 256x2e`, containing
`256 * (1 + 3 + 5) = 2304` components per atom.
The first stage uses that full output layout. The final stage explicitly
requests `256x0e`, so its contraction, message linear, and skip output all
have `(N, 256)` shape. Both stages still receive the full `(N, 4096)` A.

The forward inputs are:

| Input | Meaning | Shape |
| --- | --- | --- |
| `node_features` | Atomic basis A from the atomic-basis or interaction block | `(N, 4096)` |
| `sc` | Transformed previous hidden features for the skip connection | `(N, 256)` in the final stage; `None` in the first |
| `node_attrs` | Element one-hot encodings | `(N, 4)` |

### From atomic basis A to message m

Equation 10 constructs a product of exactly `nu` atomic-basis factors at
each correlation order: one, two, or three factors here. Fixed angular
coupling coefficients combine their components into the desired output
type `(L, M)`. Different valid coupling paths produce different B features.
The paper expression also includes learned channel mixing before the products;
the atomic-basis projection performs that operation upstream of contraction.

Equation 11 takes a learned weighted sum over paths within each order,
then sums the orders to obtain m:

\[
m_{i,kLM} = \sum_{\nu=1}^{3}\sum_{\eta_\nu}
W_{z_i kL,\eta_\nu}\,B_{i,\eta_\nu kLM}.
\]

`nu` is the correlation order, `eta_nu` is a coupling path, and `xi` counts
the factors inside Equation 10's product. These sums retain the atom,
channel, and output angular components. The learned route weights depend
on the receiving atom's element, selected using `node_attrs`.

The local contraction implementation evaluates the coupled products and
route weights together. `SymmetricContraction` packs its scalar, vector,
and order-2 output blocks into message m with shape `(N, 2304)` in the first
stage. The final stage constructs only the scalar output `(N, 256)`.

### Equation 12 and the skip connection

`message_linear(product_features)` mixes compatible message channels while
preserving their angular transformation rules and configured output layout.
Adding `sc`, when present, supplies Equation 12's second term:

```text
updated hidden features = message_linear(m) + sc
```

The previous product-basis block already produced the previous hidden
features. In the second stage, `InteractionBlock` uses those features to
construct both the new atomic basis A1 and an element-conditioned skip
connection. `ProductBasisBlock.forward` receives those tensors and adds
the skip after transforming the message. The first stage passes `sc=None`.

Equations 8–9 construct atomic basis A; Equation 12 produces the updated
hidden features. `__init__` creates the layers and weights once, while
`forward` passes each batch's actual feature tensors and skip connection.

## `SymmetricContraction.__init__`: one contraction per output type

Code: [model.py](src/model.py), `SymmetricContraction.__init__`.

`ProductBasisBlock.__init__` creates this module using the configuration
object. `self.config = config` stores a reference to the same object.

The input irreps describe the atomic basis A. There are four angular
orders, each with 256 channels. The output irreps describe the message m
and match the hidden-feature layout:

| Layout | Irreps | Components per atom |
| --- | --- | --- |
| Input A | `256x0e + 256x1o + 256x2e + 256x3o` | `256 * (1 + 3 + 5 + 7) = 4096` |
| First-stage output m | `256x0e + 256x1o + 256x2e` | `256 * (1 + 3 + 5) = 2304` |
| Final-stage output m | `256x0e` | 256 |

Irreps specify the transformation rules as well as the feature sizes.

The list comprehension loops over the three output entries, creating one
`Contraction` for each. `mul_irrep.ir` extracts its angular type and parity:
`0e`, `1o`, or `2e`. Each contraction receives the full input layout,
its assigned output type, the maximum correlation order, and the number
of chemical element types. The contraction derives the 256-channel count
from the input layout.

My prediction is that each contraction fixes its output type and builds
the valid coupling paths from the input features to that output. Fixed
angular coefficients specify each path; learned weights determine how
much it contributes. Paths are constructed separately for each supported
correlation order.

`config.correlation = 3` sets the maximum order, so the calculation includes
orders 1, 2, and 3. `self.correlation` stores that maximum.
`num_elements = 4` counts distinct chemical species: H, C, N, O.

`nn.ModuleList` stores the three contractions as registered submodules.
PyTorch can track their parameters and move them with the parent model.
The final stage creates only the scalar contraction because its explicit
output irreps contain only `0e`; the input angular orders remain unchanged.
The forward method explicitly calls them; the list itself does not run
them as a sequence.

## `Contraction.forward`: the first order-3 contraction

The mixed atomic basis is unpacked as `x` with shape `(N, 256, 16)`.
`N` counts atoms across the batch. Its 16 angular components concatenate
the order-0, order-1, order-2, and order-3 blocks.

```python
out = torch.einsum(
    "mijkp,epc,nck,ne->ncmij",
    u_max, self.weights_max, x, node_attrs,
)
```

| Tensor | Indices | Meaning |
| --- | --- | --- |
| `u_max` | `m i j k p` | Output component, three input angular slots, coupling path |
| `weights_max` | `e p c` | Receiver element type, coupling path, channel |
| `x` | `n c k` | Atom, channel, input angular component |
| `node_attrs` | `n e` | Atom and element one-hot |

Here `m` ranges over the components of one chosen output irrep: 1 for
`0e`, 3 for `1o`, or 5 for `2e`. Each input angular slot `i`, `j`, and `k`
ranges over 16 components. These indices do not identify different atoms.

The operation evaluates

\[
\mathrm{out}_{ncmij}
=\sum_{e,p,k}U_{mijkp}W_{epc}X_{nck}z_{ne}.
\]

The one-hot selects the central atom's element weights. Summing over
`e`, `p`, and `k` leaves `(N, 256, output_dim, 16, 16)`. This is an
intermediate order-3 contribution with only one atomic-basis factor consumed.
The remaining slots `i` and `j` will receive two more factors. It is not
yet the completed message from Equation 11. The learned path weights here
are Equation 11's weights; the channel mixing inside Equation 10 already
occurred in the upstream atomic-basis projection.

### Species selection and the paper's notation

For a fixed atom `n`, summing over `e` selects its element's weights;
it does not loop over or aggregate atoms of that element. With element
order `[H, C, N, O]`, a carbon atom has `node_attrs[n] = [0, 1, 0, 0]`, so

\[
\sum_e W_{epc}z_{ne}=W_{\mathrm C,pc}.
\]

The one-hot zeros the other element contributions, not other atom rows.
The repeated `n` aligns the atomic basis and element identity for the same
atom. It remains in the einsum output, so it is not summed. The summed
indices `e`, `p`, and `k` are absent from the output.

Equation 11 writes species selection directly as a weight subscript
`W_{z_i kL, eta_nu}`. The code expands this lookup into the one-hot sum
above. The paper's atom `i`, channel `k`, and path `eta_nu` become the
code's `n`, `c`, and `p`. Equation 10 constructs coupled products;
Equation 11 weights their paths by central species. This einsum evaluates
parts of both together, while leaving two atomic-basis factors for later.

### Completing orders 3, 2, and 1 by nested contraction

The order-2 coefficient tensor sums over element and path indices:

```python
c_tensor = torch.einsum("mijp,epc,ne->ncmij", u_2, weights_2, node_attrs)
out = out + c_tensor
out = torch.einsum("ncmij,ncj->ncmi", out, x)
```

Before multiplying by `x`, the terms are `U3 W3 X[k] + U2 W2`, with
species weights selected by the one-hot. Consuming angular slot `j`
produces `U3 W3 X[k] X[j] + U2 W2 X[j]`. The shape becomes
`(N, 256, output_dim, 16)`. Atom `n`, channel `c`, output component `m`,
and input slot `i` remain. Slot `j` is summed, not a channel index.

The learner predicted the remaining steps: prepare `U1 W1`, add it to
`out`, and multiply by the final `X[i]` factor. The code does exactly this:

```python
c_tensor = torch.einsum("mip,epc,ne->ncmi", u_1, weights_1, node_attrs)
out = out + c_tensor
out = torch.einsum("ncmi,nci->ncm", out, x)
return out.flatten(start_dim=1)
```

The completed sum contains three, two, and one atomic-basis factors:
`U3 W3 X[k] X[j] X[i] + U2 W2 X[j] X[i] + U1 W1 X[i]`.
Its shape is `(N, 256, output_dim)`, flattened into the output irrep block.

The nested evaluation shares remaining contractions across orders.
A scalar analogy is `a3*x**3 + a2*x**2 + a1*x`, evaluated as
`((a3*x + a2)*x + a1)*x`. Computing each term separately without sharing
powers requires six multiplications and two additions. Sharing powers
requires five multiplications and two additions. The nested form requires
three multiplications and two additions. These counts explain the saving
but do not measure the tensor implementation: its costs depend on angular
dimensions, coupling paths, and the einsum contraction plan. They do not
establish a twofold speedup for the full model.

### Who calls the contractions

`ProductBasisBlock.forward` calls `self.symmetric_contraction(node_features,
node_attrs)`, where `node_features` contains the atomic basis A.
`SymmetricContraction.forward` passes that same A and the element one-hots
to each registered `Contraction`, then concatenates their outputs.

Each `Contraction` fixes one output angular type and evaluates all its valid
coupling paths at orders 1, 2, and 3. The U3, U2, and U1 nested contractions
happen inside that module's `forward` method. `SymmetricContraction` wraps
these modules across requested output types, rather than assigning a separate
module to every path.

For the first stage, the `0e`, `1o`, and `2e` modules return `(N, 256)`,
`(N, 768)`, and `(N, 1280)`. Concatenation produces `(N, 2304)`.
The final stage requests only `0e` and returns `(N, 256)`.

## `InteractionBlock.forward`: hidden features into the next atomic basis

The second interaction receives `h1` as `node_features`, with shape
`(N, 2304)`, and element one-hots with shape `(N, 4)`.

```python
sc = self.skip_tp(node_features, node_attrs)
mixed_node_features = self.sender_linear(node_features)
sender_index = edge_index[0]
receiver_index = edge_index[1]
sender_features = mixed_node_features[sender_index]
```

The skip tensor product prepares Equation 12's element-dependent residual.
In this final stage, `sc` contains only scalar channels, with shape `(N, 256)`.
The sender linear prepares Equation 8's hidden-feature mixing, retaining
`0e + 1o + 2e` and shape `(N, 2304)`. Gathering by sender index produces
`(E, 2304)`. These are sending atoms' features, not central receiver features.

The skip cannot replace this sender input without changing the architecture:
it has discarded vector and order-2 features and uses a different,
element-dependent transform. The sender branch constructs A1 through
Equation 8; Equations 10–11 then form the product message, and Equation 12
combines the transformed message with `sc` to produce h2.

Edge harmonics encode directions with shape `(E, 16)`. `edge_irreps`
specifies their angular types; `edge_vectors` supplies the directions.
The radial MLP receives the eight distance-basis values per edge and emits
weights for legal hidden-feature/harmonic coupling routes.
Unlike the first scalar-input atomic basis, this interaction needs CG
coupling because its sender features include non-scalar angular blocks.
The detailed route instructions and tensor-product setup remain to be studied.

### Tensor-product instructions and intermediate route blocks

`tensor_product_instructions` receives the mixed hidden-feature specification,
the edge-harmonic specification, and the target atomic-basis types. It loops
over irrep blocks, not feature values. The product `input_irrep * edge_irrep`
enumerates angular and parity types permitted by e3nn. The target specification
filters those types. Every accepted coupling gets a separate output block.

For example, `0e x 0e -> 0e`, `1o x 1o -> 0e`, and `2e x 2e -> 0e`
are distinct routes with the same output type. Each keeps its own 256-channel
block and radial weights until the later projection combines matching types.

An instruction `(input_index, edge_index, output_index, "uvu", True)` connects
a block in the hidden specification and a block in the harmonic specification
to one block in the intermediate output specification. Its output index is a
block position, not angular momentum L. `"uvu"` aligns output channels with
the hidden-input channels; `True` enables externally supplied radial weights.

The helper builds `o3.Irreps(irreps_out_list)` and sorts its blocks by type.
The permutation maps old block positions to new positions. For old types
`[2e, 0e, 1o]`, the sorted types are `[0e, 1o, 2e]` and the mapping is
`(2, 0, 1)`. Remapping each instruction's output index preserves its connection.
Repeated output types remain distinct blocks after sorting.

`InteractionBlock.__init__` receives the helper's sorted specification as
`self.intermediate_irreps`, then passes it as the output specification to
`o3.TensorProduct`. The first two constructor specifications describe its two
feature inputs. In forward, the third argument is the external radial weights.
`internal_weights=False` means the module stores no learned route weights;
`shared_weights=False` allows the radial MLP to supply different weights per edge.

For the default configuration, the 17 intermediate blocks are:

| Sorted indices | Output type | Number of blocks |
| --- | --- | --- |
| 0–2 | `256x0e` | 3 |
| 3–7 | `256x1o` | 5 |
| 8–12 | `256x2e` | 5 |
| 13–16 | `256x3o` | 4 |

The edge tensor product returns `(E, 18176)` packed route features.
The later `atomic_basis_projection` mixes repeated blocks of matching type into
`256x0e + 256x1o + 256x2e + 256x3o`, producing the 4096-component A layout
after receiver aggregation. The target layout filters permitted types; it does
not require the intermediate layout to contain only one block of each type.

### Radial weights, receiver aggregation, and projection

The interaction radial MLP is an ordinary scalar network with widths
`8 -> 64 -> 64 -> 64 -> 4352`. Its hidden layers use SiLU and its final
layer is linear. The last size is `edge_tensor_product.weight_numel`:
17 routes times 256 channel weights. Each edge gets its own distance-dependent
weights. These are unrestricted scaling weights, not probabilities.

The tensor product receives sender features `(E, 2304)`, harmonics `(E, 16)`,
and radial weights `(E, 4352)`, producing route features `(E, 18176)`.
`index_add_(0, receiver_index, edge_features)` sums them into receiver rows
of a zero accumulator `(N, 18176)`. N counts atoms across the batch.
The feature count includes angular components:
`18176 = 256 * (3*1 + 5*3 + 5*5 + 4*7)`.
It is not simply channels times the number of routes.

The block divides the sum by the fixed training-set average neighbour count,
then applies the learned equivariant projection:

```python
self.atomic_basis_projection = o3.Linear(
    self.intermediate_irreps, self.atomic_basis_irreps,
)
atomic_basis = self.atomic_basis_projection(aggregated_features)
return atomic_basis, sc
```

The target is `256x0e + 256x1o + 256x2e + 256x3o`, giving packed A1
with shape `(N, 4096)`. It can later be unpacked and arranged as `(N, 256, 16)`.
For the vector part, five routes supply 1280 channels with three components
each. One learned channel map reduces `(N, 1280, 3)` to `(N, 256, 3)`:

\[
A_{n,c,m}=\sum_a W_{ca}F_{n,a,m}.
\]

The same channel matrix acts on all three components; W has no component
index m. A general unrestricted 3840-to-768 linear layer would not enforce
this equivariance. Each angular type has its own channel map. There is no
activation in this projection. The returned skip is added later in Equation 12.

This completes the main model execution walkthrough. The construction of the
fixed coupling tensors inside `u_matrix_real` remains a separate deeper topic.

## Reviewed training and evaluation conventions

Training and evaluation set PyTorch's default dtype to float64 before
constructing MACE directly. This creates fixed CG tensors at full precision;
converting float32 tensors afterward cannot recover lost precision.

Graph construction receives `model_config.cutoff` in training, validation,
and evaluation. The default is 5 angstroms. Evaluation reads it from the
saved run configuration.

Adam applies weight decay to both atomic-basis projections and all product
block parameters. Embedding, sender linears, receiver selectors, and readouts
receive zero decay. In this Adam setup, decay adds `lambda * weight` to the
loss gradient, discouraging large weights.

After `loss.backward()`, gradient clipping limits the combined parameter
gradient norm to 10 before `optimizer.step()`. It scales all gradients together
when the norm exceeds the limit; `None` disables it. This limits gradients,
not Adam's final update magnitude or the predicted atomic forces.

EMA retains a separate average of model weights. Its effective decay is
`min(decay, (1 + updates) / (10 + updates))`, giving current weights more
influence early in training. The counter starts at 1 so the first update
blends initial and updated weights. The optimizer updates the live model;
validation and the best saved model use the averaged weights.

Run metadata records implementation version 2 and the author revision.
The version label does not reject checkpoints. Configuration and weight-shape
compatibility checks still apply when resuming or loading weights.

## `train.py`: entry point, graph preparation, and loaders

Running the script calls `main()` through the `__name__ == "__main__"`
guard. `main` parses the command-line arguments, creates `TrainConfig`,
and calls `train` with the data path, device choice, and optional resume ID.

`train` sets the default dtype to float64, resolves the filesystem data path,
selects CPU or CUDA, and sets the random seed. Path resolution does not
disable logging. It loads configurations from the training file and splits
them into training and validation sets; the test holdout files remain separate.

Each configuration becomes an atomic graph using `model_config.cutoff`.
The graph carries node attributes, positions, edge indices, energy, and force
targets. Graph construction uses distances to choose edges; the model later
recomputes differentiable edge vectors and distances from positions.

The PyTorch Geometric loaders combine configurations into disconnected batched
graphs. Both training and validation use `train_config.batch_size`, normally
five configurations. Training shuffles and uses the configured `drop_last`;
validation does not shuffle and retains the final incomplete batch. Validation
does not load only one configuration at a time.

### Training-set reference and normalization statistics

The isolated-atom loader provides element reference energies used to sum
the atomic baseline for each configuration. `compute_mean_rms_energy_forces`
calculates the mean residual energy per atom after baseline subtraction and
the RMS of training force components. The model adds the mean energy shift
and uses the force RMS to scale learned energy contributions.
`compute_average_num_neighbors` supplies the fixed normalization for neighbour
sums. These statistics come from the training graphs, not the validation or
test sets. The next step is tracing the statistics helper's exact calculation.

### `compute_mean_rms_energy_forces`: reference-energy lookup

The helper takes training graphs and isolated atomic energies.
`graph_dft_energies` collects one dataset reference energy per configuration.
The dictionary `{1: 0, 6: 1, 7: 2, 8: 3}` maps atomic numbers for H, C, N,
and O to indices in the isolated-energy tensor. Here it performs energy
lookup, not one-hot encoding: carbon's atomic number 6 selects index 1.
The following expression sums the selected references over each graph's atoms
to produce one atomic-baseline energy per configuration.

The nested `map` looks up one isolated reference energy per atom, and `sum`
adds those values within each graph. The outer comprehension repeats this
for every training graph. `graph_iso_energies` therefore contains one baseline
energy per configuration, in the same graph order as `graph_dft_energies`:
\(E_{\mathrm{iso},g}=\sum_{a\in g}E_{\mathrm{iso}}(Z_a)\).
This is the isolated-reference sum, not the interacting configuration's energy.

`n_atoms` has shape `(G,)`, with one atom count for each of the G graphs.
Each force tensor has shape `(N_g, 3)`. `reshape(-1)` flattens its individual
x, y, and z components into `(3*N_g,)`; concatenating the flattened tensors
produces `(3*total_atoms,)` across the training graphs. These are force
components, not vector magnitudes or one summed force per configuration.

The force statistic squares each component, averages those squares, and takes
the square root. It is RMS, not a second squaring and not a prediction-error
RMSE despite the local variable name `rmse_forces`.

\[
F_{\mathrm{RMS}}=\sqrt{\operatorname{mean}(F_\alpha^2)},\qquad
\mu_E=\operatorname{mean}_g\left(
\frac{E_{\mathrm{DFT},g}-E_{\mathrm{iso},g}}{N_g}\right).
\]

DFT means density functional theory. The dataset provides the interacting
configuration's reference energy. The isolated baseline sums separate atomic
reference energies; it is not another energy of the interacting molecule.
The code divides the whole energy difference by the atom count, then averages
over training graphs. `.item()` converts each scalar tensor to a Python number.

Calculated from `train_300K.xyz` with seed 123 and the default 450/50 split:

| Training statistic | Value |
| --- | --- |
| Mean residual energy shift | -4.873118444762043 eV/atom |
| Force-component RMS | 0.9461023467607689 eV/angstrom |

### `compute_average_num_neighbors`: incoming edges per receiver

`graph.edge_index[1]` contains the receiver atom index for every directed edge.
`torch.unique(..., return_counts=True)` returns unique receiver indices and
their occurrence counts; `[1]` selects the counts. For receiver indices
`[0, 0, 1, 1, 1, 2]`, the counts are `[2, 3, 1]`.

The helper concatenates these counts across training graphs, converts them
to the default floating dtype, averages them, and returns a Python number.
This is an average over receiver atoms, not an equally weighted average of
graph-level averages. Atoms with zero incoming neighbours do not appear in
`torch.unique` and are omitted by this implementation.

### `run_config`: recording the training recipe

This dictionary records existing settings and training statistics; constructing
it does not change them. The data section stores the dataset path, isolated
atomic energies, mean per-atom residual energy, force-component RMS, and average
incoming-neighbour count. The force statistic is square, mean, then square root;
the neighbour statistic averages counts per receiver, not the number of unique
receivers in each graph.

`asdict(model_config)` and `asdict(train_config)` convert dataclasses into
dictionaries. They do not retrieve remote configuration. The remaining fields
record seed, metric, device, implementation version, and the author-code commit
used for comparison. `author_revision` is provenance, not authorization.

### Run directory, model construction, and optimizer

The run-directory block creates a new run and saves its configuration, or
finds an existing run and checks its saved settings before resuming.
If no paper directory is supplied, it derives the paper root from the script path.

The code creates MACE using the model configuration and computed training
statistics, converts it to double precision, and moves it to the selected
device. `make_optimizer(model, train_config)` then creates Adam with the
parameter groups reviewed earlier. These steps reuse the model constructor
and optimizer behavior we already studied.

### Validation-loss learning-rate scheduler

`ReduceLROnPlateau` uses `mode="min"` to monitor validation loss, supplied
later by `scheduler.step(validation_loss)`. If the loss shows no sufficient
improvement for the configured patience, the scheduler multiplies the learning
rate by its factor. With factor 0.8, a rate of 0.01 becomes 0.008.
Patience is measured in validation checks, which here occur once per epoch,
not once per batch. The default patience is 50.

### EMA revisited: smoothing live weights

EMA keeps a smoothed model alongside the live model that Adam updates.
For decay 0.99, the update is
\(w_{\mathrm{EMA,new}}=0.99w_{\mathrm{EMA,old}}+0.01w_{\mathrm{live}}\).
If an averaged weight is 2 and the live weight is 3, the new average is 2.01.
High decay suppresses fluctuations from noisy batch updates, but responds more
slowly to genuine changes. Decay 0.99 has a characteristic memory of roughly
100 updates, with older contributions fading rather than being abruptly dropped.

Warmup computes `effective_decay = min(decay, (1 + updates)/(10 + updates))`
inside `ema_average`. The returned blend uses that value on the old average and
`1 - effective_decay` on the current weights. At update counts 1, 10, 100,
and 1000, effective decay is approximately 0.182, 0.55, 0.918, and 0.99.
Early averaging follows the rapidly changing live model more closely.
`ema_model.update_parameters(model)` applies this after each optimizer step.

EMA warmup and the learning-rate scheduler both adjust behavior over time,
but control different operations. The scheduler changes Adam's learning rate
based on validation progress; EMA warmup changes the averaged model's tracking
rate based on update count. It does not change the optimizer's live-weight update.

### Resuming an interrupted training run

When a resume ID is supplied, `_load_training_state` restores the live model,
EMA model, optimizer, scheduler, and random-number-generator states. It also
returns the next epoch and best-validation tracking values. This continues
from the last saved epoch; it does not recover unsaved progress within an epoch.
Loading only weights would lose Adam's accumulated state and the scheduler's
history. The following guard rejects an epoch limit below the completed count,
and run-file reconciliation aligns logs and the best checkpoint with saved state.

### Epoch and batch loops

`range(start_epoch, max_epochs)` starts at the next epoch to run and excludes
the upper limit. `model.train()` selects training mode. The inner loop visits
the shuffled training batches with a zero-based batch index. If an optional
`max_train_batches` is set, the loop stops when `batch_index >=` that limit;
for a limit of 2 it processes indices 0 and 1. This is a diagnostic cap, not
a limit on the number of configurations in each batch.

`batch.to(device)` moves the graph batch to the selected device.
`optimizer.zero_grad(set_to_none=True)` clears gradients from the previous
update by setting parameter gradients to None, preventing accumulation across
ordinary batch updates. The next step runs the model on that batch.

### Prediction, loss, and weight updates

Calling MACE on node attributes, positions, edge indices, and graph assignments
returns predicted energies `(B,)` and forces `(N, 3)`. The loss combines energy
and force errors with default weights 1 and 1000. `loss.backward()` calculates
gradients with respect to model parameters. Optional gradient clipping limits
their combined norm, then `optimizer.step()` updates the live weights.
`ema_model.update_parameters(model)` blends those updated weights into the EMA.

### Validation and best-model tracking

After the training batches, validation evaluates the EMA model. The code
rejects a non-finite validation loss and compares a finite result against the
best loss so far. On improvement, it records the loss and epoch and copies
the EMA state dictionary, including weights and buffers, to independent CPU
tensors using `detach().cpu().clone()`.

The code reads the current learning rate for logging before calling
`scheduler.step(validation_loss)`. This advances the existing scheduler and
may reduce the learning rate; it does not restart the scheduler.

### Logging, checkpoint files, and the epoch callback

`log_metrics` writes the epoch metrics to the console and `metrics.jsonl`
in the run directory. It does not involve Tailscale.
`training_state.pt` is saved every epoch with the state needed for resuming.
`checkpoint.pt` is updated only when validation improves and contains the
best EMA model state. Training returns the run ID and run path when its loop ends.

`on_epoch_checkpoint` is an optional function called after the epoch's files
are saved. Local training normally supplies None. The Modal launcher supplies
`runs_volume.commit`, which persists run files to its storage volume after
each saved epoch. The callback does not perform the model checkpoint save itself.

### `compute_validation_loss`: evaluate the EMA model

The helper sets evaluation mode, finds the supplied model's device, and moves
each validation batch to that device. It predicts energies and forces, computes
the same energy/force loss, and appends `loss.detach()` to a list. Detaching
stores each value without retaining its gradient graph. Stacking and averaging
the batch losses produces the returned Python number.

Evaluation mode does not disable autograd. Position derivatives are still
needed to obtain forces from energy, but validation does not call `backward`
or update parameters. This helper averages batch losses equally; it does not
weight that final average by batch size if the last batch is smaller.

### `energy_force_loss`: where the weighted errors are used

The helper returns energy MSE over configurations plus force-component MSE
weighted by the configured coefficients, normally 1 and 1000. It combines
squared prediction errors, not raw energies and forces. Training differentiates
this loss to update parameters. Validation uses the same loss on the EMA model
for best-checkpoint selection and learning-rate scheduling.

## `evaluate.py`: rebuild the saved model and visit test conditions

The command-line entry point accepts a run ID, data path, and device choice,
then calls `evaluate`. That function sets float64, validates the data directory,
selects the device, and loads the saved run configuration. It reconstructs
MACEConfig, including restoring the radial-width list to a tuple.

MACE is built using the saved architecture and training statistics, converted
to double precision, and moved to the device. Loading the checkpoint restores
the best EMA state. Evaluation reuses the training normalization values rather
than recalculating them on test data.

The code reads the saved batch size and creates a results dictionary. It visits
the 300 K, 600 K, 1200 K, and pooled-dihedral test conditions, not cross-validation
folds. For each condition it loads configurations from its files, builds graphs
using the saved cutoff, and calls `_evaluate_graphs`. The dictionary stores test
energy and force RMSE under the condition name, not validation loss.

### `_evaluate_graphs`: collect predictions and targets

The helper creates an unshuffled loader for test graphs and four lists for
predicted and target energies and forces. It sets evaluation mode and enables
autograd for position derivatives. `torch.enable_grad()` does not start training:
there is no loss backpropagation or optimizer update. Each batch moves to the
device and produces energies and forces. Predictions and targets are detached,
moved to CPU, and appended to the lists. The next step concatenates batches
and calculates RMSE across the whole test condition.

### Final test RMSE and units

The helper concatenates predictions and targets across batches, subtracts
them, squares the errors, averages, and takes the square root. Energy RMSE
averages over configuration energies; force RMSE averages over all atom/XYZ
components. Multiplying by 1000 converts energy from eV to meV and force
from eV/angstrom to meV/angstrom. The force unit is not milliamperes.
These returned values measure prediction errors on each test condition.

The evaluation entry point prints the measured RMSE dictionary alongside
the paper reference values. This completes the main data, model, training,
and evaluation execution walkthrough. Fixed CG construction and detailed
checkpoint/storage helper internals remain optional deeper topics.

## `u_matrix_real`: constructing fixed coupling tensors

The inputs specify the angular types to couple (`0e + 1o + 2e + 3o`),
one desired output irrep, correlation order, coefficient normalization, and
numerical dtype. Normalization is a scaling convention, not a relativistic
setting. The resulting U tensors are fixed coupling coefficients, not learned
weights.

The helper checks that correlation is positive and uses the default dtype
when none is supplied. `(coupling_irreps,) * correlation` repeats the input
specification once for each input: three copies for order 3. It does not
multiply feature values. `_wigner_nj` constructs a list of paths, each stored
as `(output_irrep, coupling_tensor)`. The following code selects the requested
output type and stacks its path tensors into the final U tensor.

### `_wigner_nj`: sufficient understanding for the model walkthrough

`_wigner_nj` is a local recursive helper. It calls e3nn's `o3.wigner_3j`
to construct fixed angular coupling coefficients. It returns a list of
`(output_irrep, coupling_tensor)` pairs, with one entry per legal coupling
path. Several paths can reach the same output irrep.

For correlation 1, identity slices select the input components belonging to
each output irrep. For higher orders, the helper couples the previous result
with the rightmost input. Each input has the ordered component layout
`0e + 1o + 2e + 3o`, with 1 + 3 + 5 + 7 = 16 components. The coefficient
tensors use that full layout, with zeros outside the components used by a path.

For example, `1o × 1o → 0e` starts with coefficients shaped `[1, 3, 3]`.
The helper inserts them into the full input layout, producing `[1, 16, 16]`.
This constructs coefficients; it does not yet multiply atomic feature values.
`u_matrix_real` selects paths for the requested output and stacks them along
the last axis. `Contraction` later combines U, atomic features, and learned
weights.

The recursion, reshaping, and padding are optional implementation detail for
this walkthrough. Understanding their role is sufficient to follow MACE;
revisit the indexing when modifying or debugging coupling construction.
The main data, model, training, and evaluation walkthrough is covered.

### Identity slices and the full input layout

For `0e + 1o + 2e + 3o`, the correlation-1 call slices a 16-by-16
identity. Each slice keeps all 16 input columns:

| Irrep | Python row slice | Slice shape |
| --- | --- | --- |
| `0e` | `0:1` | `[1, 16]` |
| `1o` | `1:4` | `[3, 16]` |
| `2e` | `4:9` | `[5, 16]` |
| `3o` | `9:16` | `[7, 16]` |

`left_irrep * right_irrep` lists allowed angular orders from
`abs(l_left - l_right)` through `l_left + l_right`, inclusive. Output
parity is the product of the two input parities. The CG call receives
angular orders `.l`, with axes `[output, left, right]`.

e3nn normalizes its Wigner-3j tensor to unit Frobenius norm. The
`component` convention multiplies by `sqrt(irrep_out.dim)`; the `norm`
convention multiplies by `sqrt(left_irrep.dim * right_irrep.dim)`.
For scalar-vector coupling, the raw matching entries are `1/sqrt(3)`;
component scaling turns them into 1 in the checked e3nn real basis.

### Correlation 2: scalar and vector into a vector

Take `0e` from input 1 and `1o` from input 2, producing `1o`.
`left_u` is the first identity row, shaped `[1, 16]`. The normalized
CG tensor is shaped `[3, 1, 3]`, with `cg[:, 0, :]` equal to the
3-by-3 identity.

```python
coupled_u = torch.einsum("jk,ijl->ikl", left_u.flatten(1), cg)
```

This means `coupled_u[i, k, l] = sum_j left_u[j, k] * cg[i, j, l]`.
The intermediate left component `j` is summed away; the original left
input axis `k` remains. The result is `[3, 16, 3]`, with ones at
`[0, 0, 0]`, `[1, 0, 1]`, and `[2, 0, 2]`, and zeros elsewhere.

The reshape keeps `[3, 16, 3]`. Padding then allocates `[3, 16, 16]`:
`right_irrep.dim` is the selected vector block's size 3, while
`irrepss_right.dim` is the full right input's size 16. After the scalar
block, `right_start = 1`, so this vector block uses `slice(1, 4)`.
`padded_u[..., right_slice] = coupled_u` selects the last axis and
inserts the vector block into the full input layout. The resulting
nonzero entries are `[0, 0, 1]`, `[1, 0, 2]`, and `[2, 0, 3]`.

### Correlation 3: reuse the combined first two inputs

The recursive call first combines inputs 1 and 2. Its completed
`padded_u` becomes `left_u` in the outer call. For the route above,
the outer call therefore receives `left_irrep = 1o` and
`left_u.shape = [3, 16, 16]`, describing both original inputs.

Now couple this intermediate `1o` with the `1o` block from input 3
and select output `0e`. The new CG tensor has shape `[1, 3, 3]`, with
`cg[0, :, :]` equal to the identity divided by `sqrt(3)`. Component
normalization leaves it unchanged because the output dimension is 1.

| Step | Shape |
| --- | --- |
| `left_u` | `[3, 16, 16]` |
| `left_u.flatten(1)` | `[3, 256]` |
| New `cg` | `[1, 3, 3]` |
| Einsum result | `[1, 256, 3]` |
| Reshape into the original input axes | `[1, 16, 16, 3]` |
| Pad the full third input | `[1, 16, 16, 16]` |

Before padding, the only nonzero entries are
`coupled_u[0, 0, 1, 0]`, `coupled_u[0, 0, 2, 1]`, and
`coupled_u[0, 0, 3, 2]`, each `1/sqrt(3)` (about 0.57735).
Their axes are `[output, input 1, input 2, selected input-3 block]`.
Padding shifts input 3's vector components into its full slots 1–3.

The identity-slice example makes the contraction look like placing
coefficients into slots. In general, `left_u` already contains combined
coefficients, so the sum over `j` multiplies and adds them to the new CG
coefficients. Reshaping restores the original input axes afterward.

### Keeping paths separate, then selecting an output

Each entry in `paths` is `(irrep_out, padded_u)`. Different routes to
the same output remain separate tensors. Their route information is
represented by the coefficients; this local helper does not retain
separate ancestry labels. `Contraction` later learns a weight per path,
receiver element type, and feature channel; the U tensors stay fixed.

The multiplicity loop builds one path per copy of the current right
irrep, placing coefficients into each copy's own slice. The current
base case assumes one copy per angular term, as used by the
channel-stripped `coupling_irreps`. The pinned author's helper explicitly
expands repeated left copies in its base case.

`u_matrix_real` filters to one requested output irrep before stacking.
Thus each remaining tensor has the same output dimension and full input
dimensions. `torch.stack(u_paths, dim=-1)` creates a new path axis of
size `len(u_paths)`; it does not merge an existing axis. Correlation 3
returns `[out_dim, 16, 16, 16, num_paths]`. The empty-path fallback uses
one zero placeholder on the path axis to preserve this shape contract.

The intermediate model builds separate contractions for `0e`, `1o`,
and `2e`; the final stage keeps only scalars. Each output-specific
construction reruns the full path enumeration before filtering, for
each correlation order. These fixed tensors are constructed during
initialization and stored as buffers, then reused in forward passes.
The repeated setup cost was not timed in this session.

### References

- [ASE overview](https://ase-lib.org/)
- [Atoms object](https://docs.ase-lib.org/ase/atoms.html)
- [Reading files](https://docs.ase-lib.org/ase/io/io.html)
- [Extended XYZ format](https://docs.ase-lib.org/ase/io/formatoptions.html#extxyz)
- Practice: [EDA_practice.ipynb](EDA_practice.ipynb)
- [MACE architecture, Section 4](https://arxiv.org/html/2206.07697v2#S4)
- [MACE training settings, Appendix A.5.1](https://arxiv.org/html/2206.07697v2#A1.SS5.SSS1)
- [e3nn irreducible representations](https://docs.e3nn.org/en/stable/api/o3/o3_irreps.html)
- [e3nn Wigner-3j coefficients](https://docs.e3nn.org/en/stable/api/o3/o3_wigner.html#e3nn.o3.wigner_3j)
- [Pinned MACE author coupling helper](https://github.com/ACEsuit/mace/blob/b3e3b2c6c9ac89dbc375905740e15e1e5b1ad721/mace/tools/cg.py)
- [e3nn spherical harmonics](https://docs.e3nn.org/en/stable/api/o3/o3_sh.html)
- [PyTorch ModuleList](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ModuleList.html)
- [PyTorch automatic gradients](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html)
