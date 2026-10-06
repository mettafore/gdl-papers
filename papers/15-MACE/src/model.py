"""Learning-owned MACE model contracts for Eqs. 8--11."""

from dataclasses import dataclass

import torch
from e3nn import o3
from e3nn.nn import Activation, FullyConnectedNet
from e3nn.o3 import spherical_harmonics
from torch import Tensor, nn

IMPLEMENTATION_VERSION = 2
AUTHOR_REVISION = "b3e3b2c6c9ac89dbc375905740e15e1e5b1ad721"


@dataclass(frozen=True)
class MACEConfig:
    """Published 3BPA `L=2` architecture defaults."""

    num_elements: int = 4
    channels: int = 256
    num_interactions: int = 2
    correlation: int = 3
    max_ell: int = 2
    spherical_harmonics_max_ell: int = 3
    cutoff: float = 5.0
    num_radial_basis: int = 8
    envelope_power: int = 5
    radial_mlp_widths: tuple[int, ...] = (64, 64, 64, 1024)
    readout_hidden_width: int = 16

    def __post_init__(self) -> None:
        if self.num_elements <= 0 or self.channels <= 0:
            raise ValueError("num_elements and channels must be positive")
        if self.num_interactions != 2 or self.correlation != 3:
            raise ValueError("this reproduction implements two interactions and nu=3")
        if self.max_ell != 2 or self.spherical_harmonics_max_ell != 3:
            raise ValueError("this reproduction implements output L=2 and internal l=3")
        if self.cutoff <= 0 or self.num_radial_basis <= 0:
            raise ValueError("cutoff and num_radial_basis must be positive")

    @property
    def hidden_irreps(self) -> str:
        """Published even/odd O(3) feature types through ``L=2``."""
        return f"{self.channels}x0e + {self.channels}x1o + {self.channels}x2e"


def edge_vectors_and_lengths(
    positions: Tensor, edge_index: Tensor
) -> tuple[Tensor, Tensor]:
    """Derive live sender-to-receiver geometry from differentiable positions."""
    edge_vectors = []
    edge_distances = []
    for sender, receiver in edge_index.T:
        edge_vector = positions[receiver] - positions[sender]
        edge_distance = torch.linalg.vector_norm(edge_vector, 2)
        edge_vectors.append(edge_vector)
        edge_distances.append(edge_distance)
    edge_vectors_tensor = torch.stack(edge_vectors)
    edge_distances_tensor = torch.stack(edge_distances).reshape(-1, 1)

    return edge_vectors_tensor, edge_distances_tensor


def forces_from_energy(energy: Tensor, positions: Tensor, *, training: bool) -> Tensor:
    """Return ``-dE/dR``, retaining higher derivatives only during training."""
    (gradient,) = torch.autograd.grad(
        energy.sum(), positions, create_graph=training, allow_unused=True
    )
    if gradient is None:
        return torch.zeros_like(positions)
    return -gradient


@dataclass(frozen=True)
class CouplingRoute:
    """One allowed irreducible tensor-product path.

    ``intermediate_ells`` records the angular orders produced between the
    input factors and the final output. It is empty for a two-factor product.
    """

    input_ells: tuple[int, ...]
    output_ell: int
    intermediate_ells: tuple[int, ...] = ()


def enumerate_coupling_routes(
    input_ells: tuple[int, ...], *, output_max_ell: int, correlation: int
) -> tuple[CouplingRoute, ...]:
    """Pedagogical SO(3) route enumeration, outside the final model path.

    The model uses ``u_matrix_real`` for full O(3) coupling and parity.
    """

    def recursive_cpl_routes(input_route, intermed_ell, l2, pos, intermed_ells):
        """Recursive helper"""
        allowed_ell = range(abs(intermed_ell - l2), intermed_ell + l2 + 1)
        intermed_ells.append(intermed_ell)
        if correlation == pos + 2:
            for output_ell in allowed_ell:
                if output_ell <= output_max_ell:
                    coupling_routes.append(
                        CouplingRoute(
                            input_ells=tuple(input_route),
                            output_ell=output_ell,
                            intermediate_ells=tuple(intermed_ells[1:]),
                        )
                    )
            return coupling_routes
        else:
            pos += 1
            for ell in allowed_ell:
                recursive_cpl_routes(
                    input_route, ell, input_route[pos + 1], pos, intermed_ells
                )
                intermed_ells.pop()

        return coupling_routes

    coupling_routes: list[CouplingRoute] = []
    intermed_ells: list[int] = []
    coupling_routes = coupling_routes + recursive_cpl_routes(
        input_ells,
        input_ells[0],
        input_ells[1],
        pos=0,
        intermed_ells=intermed_ells,
    )
    return tuple(coupling_routes)


class BesselRadialBasis(nn.Module):
    """Eight radial functions multiplied by the polynomial cutoff envelope."""

    def __init__(self, num_basis: int = 8, cutoff: float = 5.0, power: int = 5):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.power = power

    def forward(self, distances: Tensor) -> Tensor:
        """Map distances `[E, 1]` to cutoff radial features `[E, num_basis]`."""
        n = torch.tensor([x for x in range(1, self.num_basis + 1)]).to(
            device=distances.device, dtype=distances.dtype
        )
        zero_distances = (distances == 0).reshape(-1)
        bessel_fn = torch.zeros_like(distances).repeat(1, self.num_basis)
        tensor_2 = torch.tensor(2).to(device=distances.device, dtype=distances.dtype)
        bessel_fn[zero_distances, :] = (
            torch.sqrt(tensor_2 / self.cutoff) * n * torch.pi / self.cutoff
        )
        bessel_fn[~zero_distances, :] = (
            torch.sqrt(tensor_2 / self.cutoff)
            * torch.sin(n * torch.pi * distances[~zero_distances, :] / self.cutoff)
            / distances[~zero_distances, :]
        )
        r_over_cutoff = distances / self.cutoff
        p = self.power
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * r_over_cutoff.pow(p)
            + p * (p + 2.0) * r_over_cutoff.pow(p + 1)
            - (p * (p + 1.0) / 2.0) * r_over_cutoff.pow(p + 2)
        )
        envelope = envelope * (distances < self.cutoff)
        return bessel_fn * envelope


class AtomicBasis(nn.Module):
    """Construct first-stage Eq. 9 features with the reference's transforms."""

    def __init__(self, config: MACEConfig, *, avg_num_neighbors: float):
        super().__init__()
        self.config = config
        if avg_num_neighbors <= 0:
            raise ValueError("avg_num_neighbors must be positive")
        self.avg_num_neighbors = avg_num_neighbors
        self.input_irreps = o3.Irreps(f"{config.channels}x0e")
        self.atomic_basis_irreps = o3.Irreps(
            [
                (config.channels, ir)
                for _, ir in o3.Irreps.spherical_harmonics(
                    config.spherical_harmonics_max_ell
                )
            ]
        )
        self.sender_linear = o3.Linear(self.input_irreps, self.input_irreps)
        self.atomic_basis_projection = o3.Linear(
            self.atomic_basis_irreps, self.atomic_basis_irreps
        )
        self.receiver_selector = o3.FullyConnectedTensorProduct(
            self.atomic_basis_irreps,
            o3.Irreps(f"{config.num_elements}x0e"),
            self.atomic_basis_irreps,
        )
        # Match the author's zero-preserving, variance-normalized radial MLP.
        # No biases: when the cutoff makes all radial features zero, the
        # emitted edge weights are zero too, even after training.
        self.radial_mlp = FullyConnectedNet(
            [config.num_radial_basis, *config.radial_mlp_widths],
            torch.nn.functional.silu,
        )

    def forward(
        self,
        node_features: Tensor,
        node_attrs: Tensor,
        edge_index: Tensor,
        edge_vectors: Tensor,
        edge_radial: Tensor,
    ) -> Tensor:
        """Return edge-conditioned atomic basis features.

        ``edge_radial`` has shape ``[E, num_radial_basis]`` and is computed
        once from edge lengths before entering this block.
        """
        node_features = self.sender_linear(node_features)
        sender_features = node_features[edge_index[0]].unsqueeze(-1)
        receiver_index = edge_index[1]
        ells = list(range(self.config.spherical_harmonics_max_ell + 1))
        normalize = True
        normalization = "component"
        spherical_tensors = spherical_harmonics(
            l=ells, x=edge_vectors, normalize=normalize, normalization=normalization
        )
        # The reference tensor product orders weights by ell, then channel.
        radial_weight = (
            self.radial_mlp(edge_radial)
            .reshape(
                -1,
                self.config.spherical_harmonics_max_ell + 1,
                self.config.channels,
            )
            .transpose(1, 2)
        )

        # A degree ell has 2 * ell + 1 magnetic components m. Split the packed
        # [E, 16] spherical-harmonic tensor into its ell = 0, 1, 2, 3 blocks.
        split_sizes = [2 * ell + 1 for ell in ells]
        split_sperical_tensors = torch.split(spherical_tensors, split_sizes, dim=1)

        # Keep one receiver-aggregated message block per ell before packing them.
        agg_messages = list()
        for ell in ells:
            # Select this ell's distance-dependent weight and add a trailing
            # broadcasting axis: [E, channels] -> [E, channels, 1].
            radial_weight_mul = radial_weight[:, :, ell].unsqueeze(2)

            # Add a channel broadcasting axis: [E, 2 * ell + 1]
            # -> [E, 1, 2 * ell + 1].
            spherical_tensor_mul = split_sperical_tensors[ell].unsqueeze(1)

            # For each directed edge j -> i, combine R_{k ell}(r_ji),
            # Y_{ell m}(rhat_ji), and the sender's element embedding W_{k z_j}:
            # [E, 256, 1] * [E, 1, 2 * ell + 1] * [E, 256, 1]
            # -> edge message [E, 256, 2 * ell + 1].
            edge_message = radial_weight_mul * spherical_tensor_mul * sender_features

            # Start one zero message per atom. Sum every complete edge message
            # j -> i into the row of its receiver i.
            cumulative_message = torch.zeros(
                len(node_features), self.config.channels, split_sizes[ell]
            ).to(device=node_features.device, dtype=node_features.dtype)

            # Equivalent to: cumulative_message[receiver_index[edge]] +=
            # edge_message[edge], for every edge. Result: [N, 256, 2 * ell + 1].
            cumulative_message = cumulative_message.index_add_(
                0, receiver_index, edge_message
            )

            # Preserve the atom axis N and pack channel k with magnetic index m:
            # [N, 256, 2 * ell + 1] -> [N, 256 * (2 * ell + 1)].
            agg_messages.append(cumulative_message.flatten(start_dim=1))

        # Join the ell = 0, 1, 2, 3 blocks into the packed atomic basis [N, 4096].
        atomic_basis = torch.concat(agg_messages, dim=1)
        atomic_basis = self.atomic_basis_projection(atomic_basis)
        atomic_basis = atomic_basis / self.avg_num_neighbors
        return self.receiver_selector(atomic_basis, node_attrs)


def _wigner_nj(
    irrepss: tuple[o3.Irreps, ...],
    *,
    normalization: str,
    dtype: torch.dtype,
) -> list[tuple[o3.Irrep, Tensor]]:
    """Return every legal output irrep and its fixed real U-path tensor."""
    if len(irrepss) == 1:
        (irreps,) = irrepss
        identity = torch.eye(irreps.dim, dtype=dtype)

        return [
            (irrep, identity[irrep_slice])
            for (_, irrep), irrep_slice in zip(irreps, irreps.slices())
        ]
    elif len(irrepss) > 1:
        *irrepss_left, irrepss_right = irrepss
        left_paths = _wigner_nj(
            irrepss=tuple(irrepss_left), normalization=normalization, dtype=dtype
        )
        paths = []
        for left_irrep, left_u in left_paths:
            right_start = 0
            for multiplicity, right_irrep in irrepss_right:
                for irrep_out in left_irrep * right_irrep:
                    cg = o3.wigner_3j(
                        irrep_out.l,
                        left_irrep.l,
                        right_irrep.l,
                        dtype=dtype,
                    )
                    if normalization == "component":
                        cg *= irrep_out.dim**0.5
                    elif normalization == "norm":
                        cg *= (left_irrep.dim * right_irrep.dim) ** 0.5
                    coupled_u = torch.einsum(
                        "jk,ijl->ikl",
                        left_u.flatten(1),
                        cg,
                    )
                    coupled_u = coupled_u.reshape(
                        irrep_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        right_irrep.dim,
                    )
                    for copy in range(multiplicity):
                        padded_u = torch.zeros(
                            irrep_out.dim,
                            *(irrep.dim for irrep in irrepss_left),
                            irrepss_right.dim,
                            dtype=dtype,
                        )
                        right_slice = slice(
                            right_start + copy * right_irrep.dim,
                            right_start + (copy + 1) * right_irrep.dim,
                        )
                        padded_u[..., right_slice] = coupled_u
                        paths.append((irrep_out, padded_u))
                right_start += multiplicity * right_irrep.dim
    return sorted(paths, key=lambda path: path[0])


def u_matrix_real(
    coupling_irreps: o3.Irreps,
    irrep_out: o3.Irrep,
    correlation: int,
    *,
    normalization: str = "component",
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Build fixed real CG paths for one Eq. 10 output irrep.

    Returns ``[m_out, in_dim, ..., in_dim, n_paths]``, with one ``in_dim``
    axis per correlation factor.
    """
    if correlation < 1:
        raise ValueError("correlation must be at least 1")
    if dtype is None:
        dtype = torch.get_default_dtype()
    irrepss = (coupling_irreps,) * correlation
    paths = _wigner_nj(
        irrepss=irrepss,
        normalization=normalization,
        dtype=dtype,
    )
    u_paths = [u for (output_irrep, u) in paths if output_irrep == irrep_out]

    if not u_paths:
        return torch.zeros(
            irrep_out.dim, *([coupling_irreps.dim] * correlation), 1, dtype=dtype
        )
    return torch.stack(u_paths, dim=-1)


class Contraction(nn.Module):
    """Build one Eq. 10--11 output-irrep block.

    ``input_irreps`` carries all learned feature channels; ``coupling_irreps``
    strips that multiplicity so the fixed U tensors describe angular types only.
    """

    def __init__(
        self,
        *,
        input_irreps: o3.Irreps,
        irrep_out: o3.Irrep,
        correlation: int,
        num_elements: int,
    ) -> None:
        super().__init__()
        # Atomic basis A is packed as [N, 4096] and viewed here as [N, 256, 16].
        # num_features is the learned channel count k: 256, not N * 4096.
        self.num_features = input_irreps.count("0e")

        # Strip the 256 channel multiplicity: fixed U tensors couple only the
        # angular types 0e + 1o + 2e + 3o, whose packed dimension is 16.
        self.coupling_irreps = o3.Irreps([mul_irrep.ir for mul_irrep in input_irreps])

        # This Contraction constructs one output type: 0e, 1o, or 2e.
        self.irrep_out = irrep_out

        # Maximum correlation order nu. This bounded reproduction fixes nu = 3.
        self.correlation = correlation
        if self.correlation != 3:
            raise ValueError(
                "This learning implementation currently supports correlation=3 only."
            )

        # Receiver element types (H, C, N, O): Eq. 11 learns separate weights
        # for each central atom type z_i.
        self.num_elements = num_elements

        # Eq. 10: construct fixed generalized-CG/U tensors for each order.
        for nu in range(1, self.correlation + 1):
            # Shape: [out_dim, 16, ..., 16, num_paths], with nu copies of 16.
            # Each 16 selects an angular (ell, m) component from one A factor.
            u_tensor = u_matrix_real(
                coupling_irreps=self.coupling_irreps,
                irrep_out=self.irrep_out,
                correlation=nu,
            )

            # Fixed, non-trainable buffers move with the model but receive no gradients.
            self.register_buffer(f"U_matrix_{nu}", u_tensor)

        # Learned Eq. 11 path weights for lower orders, retained in nu=2, nu=1 order.
        self.weights = nn.ParameterList()

        # Start with the maximum correlation order, nu=3.
        u_max = getattr(self, f"U_matrix_{self.correlation}")
        # The last U axis enumerates its legal generalized-CG routes eta_nu.
        num_paths = u_max.size(-1)

        # Eq. 11 weights for nu=3: [4 receiver elements, num_paths, 256 channels].
        self.weights_max = nn.Parameter(
            torch.randn(
                self.num_elements,
                num_paths,
                self.num_features,
            )
            # Keep a sum over many paths from starting at an overly large scale.
            / num_paths
        )

        # Add separate Eq. 11 weights for nu=2 then nu=1; forward evaluates
        # the same descending recurrence after the maximum-order contraction.
        for nu in range(self.correlation - 1, 0, -1):
            # Read this order's U tensor to learn one weight per legal path.
            u_tensor = getattr(self, f"U_matrix_{nu}")
            num_paths = u_tensor.size(-1)
            self.weights.append(
                nn.Parameter(
                    torch.randn(self.num_elements, num_paths, self.num_features)
                    / num_paths
                )
            )

    def forward(self, atomic_basis: Tensor, node_attrs: Tensor) -> Tensor:
        """Return this output irrep's `[N, channels * (2 * ell + 1)]` block."""
        # e3nn packs A by ell first, then by channel and m within each ell.
        # Follow the author's reshape_irreps: unpack each ell separately, then
        # concatenate its m components along the angular axis. A whole-tensor
        # reshape would mix different channels and angular types.
        blocks = torch.split(
            atomic_basis,
            [self.num_features * ir.dim for _, ir in self.coupling_irreps],
            dim=-1,
        )
        x = torch.cat(
            [
                block.reshape(atomic_basis.size(0), self.num_features, ir.dim)
                for block, (_, ir) in zip(blocks, self.coupling_irreps)
            ],
            dim=-1,
        )

        # Retrieve U_3: fixed Eq. 10 coupling paths for maximum correlation nu=3.
        # Shape: [out_dim, 16, 16, 16, num_paths].
        u_max = getattr(self, f"U_matrix_{self.correlation}")
        # Eq. 11 weights: [4 receiver elements, num_paths, 256 channels].
        # x: [N, 256, 16]; node_attrs: [N, 4].
        # Code indices i, j, k are three packed angular slots (size 16), not
        # atom indices. The paper's channel k is called c in this einsum.
        # Output shape: [N, 256, out_dim, 16, 16]; k is contracted first.
        out = torch.einsum(
            "mijkp,epc,nck,ne->ncmij",
            u_max,
            self.weights_max,
            x,
            node_attrs,
        )
        # Build the fixed nu=2 stencil: [out_dim, 16, 16, num_paths].
        u_2 = getattr(self, "U_matrix_2")
        # Eq. 11 nu=2 weights: [4 receiver elements, num_paths, 256 channels].
        weights_2 = self.weights[0]

        # This is the nu=2 coefficient stencil, not yet its full A_i product.
        # Shape: [N, 256, out_dim, 16, 16].
        c_tensor = torch.einsum(
            "mijp,epc,ne->ncmij",
            u_2,
            weights_2,
            node_attrs,
        )
        # Add terms with the same remaining angular axes i,j:
        # U3 W3 A[k] + U2 W2.
        out = out + c_tensor
        # Consume the second A factor, indexed by j:
        # U3 W3 A[k] A[j] + U2 W2 A[j].
        out = torch.einsum(
            "ncmij,ncj->ncmi",
            out,
            x,
        )
        # Retrieve the fixed nu=1 coupling stencil and its Eq. 11 weights.
        u_1 = getattr(self, "U_matrix_1")
        weights_1 = self.weights[1]

        # U1 W1 has one remaining angular axis i: [N, 256, out_dim, 16].
        c_tensor = torch.einsum(
            "mip,epc,ne->ncmi",
            u_1,
            weights_1,
            node_attrs,
        )
        # Add the nu=1 term:
        # U3 W3 A[k] A[j] + U2 W2 A[j] + U1 W1.
        out = out + c_tensor
        # Consume the final A factor, indexed by i. This avoids materializing
        # the full raw outer product [N, 256, 16, 16, 16].
        out = torch.einsum(
            "ncmi,nci->ncm",
            out,
            x,
        )

        # Final shape: [N, 256, out_dim]. Pack channel c and output component m
        # into [N, 256 * out_dim] for this one output irrep.
        return out.flatten(start_dim=1)


class SymmetricContraction(nn.Module):
    """Form Eq. 10 higher-order equivariant product features."""

    def __init__(self, config: MACEConfig, *, output_irreps: o3.Irreps | None = None):
        super().__init__()
        self.config = config
        # this declares the layout of atomic basis A
        self.input_irreps = o3.Irreps(
            f"{config.channels}x0e + {config.channels}x1o + "
            f"{config.channels}x2e + {config.channels}x3o"
        )
        # Output Irreps is the output specification of m in Eq 11
        self.output_irreps = (
            o3.Irreps(config.hidden_irreps) if output_irreps is None else output_irreps
        )

        self.contractions = nn.ModuleList(
            [
                Contraction(
                    # input spec of A
                    input_irreps=self.input_irreps,
                    # the irrep type for the particular output irrep
                    irrep_out=mul_irrep.ir,
                    # Correlation nu
                    correlation=config.correlation,
                    # Number of types of elements
                    num_elements=config.num_elements,
                )
                for mul_irrep in self.output_irreps
            ]
        )

        self.correlation = config.correlation

    def forward(self, atomic_basis: Tensor, node_attrs: Tensor) -> Tensor:
        """Return the Eq. 11 message m with shape `[N, output_irreps.dim]`.

        Each Contraction internally constructs Eq. 10 higher-order paths and
        combines them with receiver-element weights; this method packs its
        0e, 1o, and 2e output-irrep blocks into one message tensor.
        """
        # One packed message block per output irrep: 0e, 1o, and 2e.
        outputs = [
            contraction(atomic_basis, node_attrs) for contraction in self.contractions
        ]
        # Concatenate channels/components: [N, 256] + [N, 768] + [N, 1280].
        return torch.cat(outputs, dim=1)


class ProductBasisBlock(nn.Module):
    """Apply the Eq. 10--12 node-wise product-basis update."""

    def __init__(
        self, config: MACEConfig, *, output_irreps: o3.Irreps | None = None
    ) -> None:
        super().__init__()
        # Eq. 10--11: build higher-order B paths and combine them into message m.
        self.symmetric_contraction = SymmetricContraction(
            config, output_irreps=output_irreps
        )
        # The first update keeps 0e + 1o + 2e; the final update keeps only 0e.
        self.hidden_irreps = self.symmetric_contraction.output_irreps
        # Eq. 12 first term: equivariantly mix channels within each output L.
        self.message_linear = o3.Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.hidden_irreps,
        )

    def forward(
        self,
        node_features: Tensor,
        sc: Tensor | None,
        node_attrs: Tensor,
    ) -> Tensor:
        """Apply Eq. 10--12 and optionally add the interaction skip."""
        # Eq. 10--11: take atomic basis A [N, 4096] and receiver element
        # one-hots [N, 4], returning the configured message irreps.
        product_features = self.symmetric_contraction(
            node_features,
            node_attrs,
        )

        # Eq. 12 first term: mix compatible channels of m without mixing L or M.
        updated_features = self.message_linear(product_features)

        if sc is not None:
            # Eq. 12 second term: add the receiver-element-conditioned skip of
            # the previous hidden features, so this block learns an update to h.
            updated_features = updated_features + sc

        return updated_features


def tensor_product_instructions(
    irreps_1: o3.Irreps,
    irreps_2: o3.Irreps,
    target_irreps: o3.Irreps,
) -> tuple[o3.Irreps, list[tuple[int, int, int, str, bool]]]:
    """List the legal Eq. 8 tensor-product connections."""
    # Each accepted coupling route gets its own output block. Different routes
    # can produce the same irrep type, so duplicate types remain separate here.
    irreps_out_list: list[tuple[int, o3.Irrep]] = []
    instructions = []
    # Visit each hidden-state irrep block, then each spherical-harmonic block.
    for input_index, (multiplicity, input_irrep) in enumerate(irreps_1):
        # The index identifies the spherical-harmonic block in irreps_2.
        for edge_index, (_, edge_irrep) in enumerate(irreps_2):
            # This iterates over allowed output irrep types, not feature values.
            for output_irrep in input_irrep * edge_irrep:
                # Keep only couplings whose output type is requested for A.
                if output_irrep in target_irreps:
                    # The current list length is the next output-block index.
                    output_index = len(irreps_out_list)
                    # Preserve the hidden-channel multiplicity for this route.
                    irreps_out_list.append((multiplicity, output_irrep))
                    # Record (hidden block, harmonic block, output block, mode,
                    # weighted). "uvu" keeps output channel u aligned with
                    # hidden channel u; v indexes/sums the second input copies.
                    # True enables route weights, supplied per edge by radial_mlp.
                    instructions.append(
                        (
                            input_index,
                            edge_index,
                            output_index,
                            "uvu",
                            True,
                        )
                    )
    # Build the intermediate representation; repeated irrep types from
    # different routes are distinct blocks and are not collapsed here.
    irreps_out = o3.Irreps(irreps_out_list)
    # Sort blocks into e3nn's canonical irrep order (including parity).
    # permutation maps each old output-block index to its new sorted index.
    irreps_out, permutation, _ = irreps_out.sort()
    # Remap each instruction's output index to match the sorted block order.
    instructions = [
        (
            input_index,
            edge_index,
            permutation[output_index],
            mode,
            trainable,
        )
        for input_index, edge_index, output_index, mode, trainable in instructions
    ]
    # Keep instructions grouped by their remapped output block.
    return irreps_out, sorted(instructions, key=lambda instruction: instruction[2])


class InteractionBlock(nn.Module):
    """Build Eq. 8 neighbour-aggregated atomic-basis features."""

    def __init__(
        self,
        config: MACEConfig,
        avg_num_neighbors: float,
        *,
        output_irreps: o3.Irreps | None = None,
    ) -> None:
        super().__init__()

        if avg_num_neighbors <= 0:
            raise ValueError("avg_num_neighbors must be positive")

        # Training-set average used to normalize summed neighbour messages.
        self.avg_num_neighbors = avg_num_neighbors

        # Hidden state h: 256 channels for each angular order ell = 0, 1, 2.
        self.hidden_irreps = o3.Irreps(config.hidden_irreps)
        self.output_irreps = (
            self.hidden_irreps if output_irreps is None else output_irreps
        )

        # One-hot element identity z (H, C, N, O), represented as four scalars.
        self.node_attr_irreps = o3.Irreps(f"{config.num_elements}x0e")

        # Eq. 12 residual term: condition the previous hidden state on the
        # receiver atom's element, while preserving its irreps.
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.hidden_irreps,
            self.node_attr_irreps,
            self.output_irreps,
        )

        # Eq. 8 channel mixing W: mix hidden channels within each ell while
        # preserving angular order and magnetic components.
        self.sender_linear = o3.Linear(self.hidden_irreps, self.hidden_irreps)

        # Spherical-harmonic irreps for edge directions, ell = 0, 1, 2, 3.
        self.edge_irreps = o3.Irreps.spherical_harmonics(
            config.spherical_harmonics_max_ell
        )

        # Eq. 8 target A: 256 channels for each harmonic irrep, ell = 0..3.
        # Their component counts are 1 + 3 + 5 + 7, so the packed width is 4096.
        self.atomic_basis_irreps = o3.Irreps(
            [(config.channels, irrep) for _, irrep in self.edge_irreps]
        )

        # Enumerate legal equivariant couplings h_(ell2) x Y_(ell1) -> A_(ell3).
        # The hidden state has non-scalar types, so Eq. 8 needs CG coupling
        # rather than the direct scalar-feature multiplication used in Eq. 9.
        self.intermediate_irreps, instructions = tensor_product_instructions(
            self.hidden_irreps,
            self.edge_irreps,
            self.atomic_basis_irreps,
        )
        # Apply the fixed CG couplings to hidden features and edge harmonics.
        # Route weights are supplied externally by radial_mlp in forward;
        # they are not stored inside this TensorProduct and may differ by edge.
        self.edge_tensor_product = o3.TensorProduct(
            self.hidden_irreps,
            self.edge_irreps,
            self.intermediate_irreps,
            instructions=instructions,
            internal_weights=False,
            shared_weights=False,
        )

        # The author's bias-free radial MLP preserves the cutoff's zero input.
        # Emit one external weight per channel and legal tensor-product route.
        self.radial_mlp = FullyConnectedNet(
            [
                config.num_radial_basis,
                *config.radial_mlp_widths[:-1],
                self.edge_tensor_product.weight_numel,
            ],
            torch.nn.functional.silu,
        )

        # Combine route-specific copies of matching irreps into the requested
        # A layout (256 channels each of 0e, 1o, 2e, and 3o; packed width 4096).
        self.atomic_basis_projection = o3.Linear(
            self.intermediate_irreps,
            self.atomic_basis_irreps,
        )

    def forward(
        self,
        node_features: Tensor,
        node_attrs: Tensor,
        edge_index: Tensor,
        edge_vectors: Tensor,
        edge_radial: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build Eq. 8 atomic-basis features and the Eq. 12 skip contribution."""
        # Eq. 12 second term: prepare the element-conditioned skip from each
        # atom's current hidden features; ProductBasisBlock adds it to the
        # neighbor-message update.
        sc = self.skip_tp(node_features, node_attrs)

        # Eq. 8 channel mixing: mix hidden channels within each irrep; the
        # weights are shared across m, preserving equivariance.
        mixed_node_features = self.sender_linear(node_features)

        # Each edge column stores its sender and receiver atom indices.
        sender_index = edge_index[0]
        receiver_index = edge_index[1]

        # Gather the sender state h_j for every edge j -> i; senders may repeat.
        sender_features = mixed_node_features[sender_index]

        # Compute edge-direction harmonics for ell = 0..3. normalize=True
        # uses unit directions; normalization sets the harmonic scaling convention.
        edge_harmonics = spherical_harmonics(
            l=self.edge_irreps,
            x=edge_vectors,
            normalize=True,
            normalization="component",
        )
        # edge_radial has shape [E, 8]: Bessel-expanded, cutoff-weighted distances.
        radial_weights = self.radial_mlp(edge_radial)
        # Per edge, the radial MLP supplies one weight per hidden channel per
        # route (256 here); the tensor product applies them with fixed CG couplings.
        edge_features = self.edge_tensor_product(
            sender_features,
            edge_harmonics,
            radial_weights,
        )

        # Allocate one message accumulator row per atom.
        num_nodes = node_features.shape[0]
        aggregated_features = edge_features.new_zeros(
            num_nodes,
            self.intermediate_irreps.dim,
        )
        # Sum each edge message into its receiver atom's row.
        aggregated_features.index_add_(
            0,
            receiver_index,
            edge_features,
        )
        # Normalize the summed messages by the training-set average neighbor count.
        aggregated_features = aggregated_features / self.avg_num_neighbors
        # Pack the intermediate route outputs into the requested atomic-basis irreps.
        atomic_basis = self.atomic_basis_projection(aggregated_features)
        # Return A_i and the skip contribution for the later Eq. 12 update.
        return atomic_basis, sc


class MACE(nn.Module):
    """Two-interaction scalar-energy model with forces from autograd."""

    def __init__(
        self,
        config: MACEConfig,
        *,
        avg_num_neighbours: float,
        atomic_energies: Tensor | None = None,
        mean_energy_shift: float = 0.0,
        rms_forces: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.avg_num_neighbours = avg_num_neighbours

        if atomic_energies is None:
            atomic_energies = torch.zeros(config.num_elements, dtype=torch.float64)
        if atomic_energies.shape != (config.num_elements,):
            raise ValueError("atomic_energies must have one value per element")
        if rms_forces <= 0.0:
            raise ValueError("rms_forces must be positive")

        # Fixed training-set energy references/scales; buffers travel with the
        # checkpoint but are not trainable parameters.
        self.register_buffer("atomic_energies", atomic_energies.to(torch.float64))
        self.register_buffer(
            "mean_energy_shift", torch.tensor(mean_energy_shift, dtype=torch.float64)
        )
        self.register_buffer(
            "rms_forces", torch.tensor(rms_forces, dtype=torch.float64)
        )

        # creating embedding from one hot to num channels
        self.node_embedding = o3.Linear(
            o3.Irreps(f"{config.num_elements}x0e"),
            o3.Irreps(f"{config.channels}x0e"),
        )

        self.radial_basis = BesselRadialBasis(
            num_basis=config.num_radial_basis,
            cutoff=config.cutoff,
            power=config.envelope_power,
        )
        # Creating clas which encapsulates first layer (Eq 9) of MACE
        # With inputs edge widths and atomic embeddings
        self.first_atomic_basis = AtomicBasis(
            config, avg_num_neighbors=self.avg_num_neighbours
        )

        self.first_product_basis = ProductBasisBlock(config)

        self.first_readout = o3.Linear(
            irreps_in=o3.Irreps(config.hidden_irreps),
            irreps_out=o3.Irreps("1x0e"),
        )

        # The reference's final update keeps only scalars for the energy readout.
        final_irreps = o3.Irreps(f"{config.channels}x0e")
        self.second_interaction_block = InteractionBlock(
            config,
            avg_num_neighbors=self.avg_num_neighbours,
            output_irreps=final_irreps,
        )

        self.second_product_basis = ProductBasisBlock(
            config, output_irreps=final_irreps
        )

        self.second_readout = nn.Sequential()

        self.second_readout.append(
            o3.Linear(
                irreps_in=final_irreps,
                irreps_out=o3.Irreps(f"{config.readout_hidden_width}x0e"),
            )
        )

        self.second_readout.append(
            Activation(
                o3.Irreps(f"{config.readout_hidden_width}x0e"),
                [torch.nn.functional.silu],
            )
        )

        self.second_readout.append(
            o3.Linear(
                irreps_in=o3.Irreps(f"{config.readout_hidden_width}x0e"),
                irreps_out=o3.Irreps("1x0e"),
            )
        )

    def forward(
        self,
        node_attrs: Tensor,
        positions: Tensor,
        edge_index: Tensor,
        graph_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return total energies `[B]` and forces `[N, 3]`."""
        # h0 -> first atomic basis -> A0 -> first product basis -> h1
        #                                                       |-> first readout -> E1
        #                                                       `-> second interaction
        #                                                           -> (A1, skip)
        #                                                           -> second product basis -> h2
        #                                                                                 `-> second readout -> E2
        # Add E1 and E2 per atom, sum atoms by graph for total energy, then get forces from energy.
        positions.requires_grad_(True)
        node_features = self.node_embedding(node_attrs)
        edge_vectors, edge_distances = edge_vectors_and_lengths(positions, edge_index)

        edge_radial = self.radial_basis(edge_distances)

        A0 = self.first_atomic_basis(
            node_features,
            node_attrs,
            edge_index,
            edge_vectors,
            edge_radial,
        )

        h1 = self.first_product_basis(
            A0,
            None,
            node_attrs,
        )

        E1 = self.first_readout(h1)

        A1, sc = self.second_interaction_block(
            h1,
            node_attrs,
            edge_index,
            edge_vectors,
            edge_radial,
        )

        h2 = self.second_product_basis(A1, sc, node_attrs)

        E2 = self.second_readout(h2)

        # Convert learned per-atom residuals back to physical energies:
        # isolated-atom baseline + dataset mean residual + force-RMS-scaled model.
        atomic_baseline = node_attrs @ self.atomic_energies.unsqueeze(-1)
        E = self.rms_forces * (E1 + E2) + self.mean_energy_shift + atomic_baseline

        E_g = E.new_zeros((graph_index.max() + 1, E.shape[1]))

        E_g.index_add_(0, graph_index, E)

        E_g = E_g.squeeze(-1)

        F_g = forces_from_energy(E_g, positions, training=self.training)
        return E_g, F_g
