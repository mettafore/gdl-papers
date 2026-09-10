"""Learning-owned MACE model contracts for Eqs. 8--11."""

from dataclasses import dataclass

import torch
from e3nn import o3
from e3nn.o3 import spherical_harmonics
from torch import Tensor, nn


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
        if self.num_interactions <= 0 or self.correlation <= 0:
            raise ValueError("num_interactions and correlation must be positive")
        if self.max_ell < 0 or self.spherical_harmonics_max_ell < self.max_ell:
            raise ValueError("internal spherical-harmonic ell must cover output ell")
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
    # TODO(luv): derive vectors from the positions passed into the model using
    # the sender/receiver convention in the data contract; return lengths [E, 1].
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
    # TODO(luv): differentiate total energy with respect to positions; preserve
    # second derivatives during training and give unused-position energies zero
    # physical forces.
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
    """Enumerate allowed angular-momentum paths up to the requested order.

    TODO(luv): apply the triangle rule one coupling at a time, retain the route
    history (not only the final ell), enforce the requested output range, and
    then add parity when moving from SO(3) intuition to full O(3) irreps.
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
        """Map distances `[E, 1]` to cutoff radial features `[E, num_basis]`.

        TODO(luv): implement the paper's Bessel basis and polynomial envelope;
        handle the removable singularity at zero without masking real errors.
        """
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
    """Construct Eq. 8 neighbour features from radial and angular factors."""

    def __init__(self, config: MACEConfig):
        super().__init__()
        self.config = config
        self.radial_mlp = nn.Sequential()
        self.radial_mlp.append(nn.Linear(8, config.radial_mlp_widths[0]))
        for idx in range(len(config.radial_mlp_widths) - 1):
            self.radial_mlp.append(nn.SiLU())
            self.radial_mlp.append(
                nn.Linear(
                    config.radial_mlp_widths[idx], config.radial_mlp_widths[idx + 1]
                )
            )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_vectors: Tensor,
        edge_radial: Tensor,
    ) -> Tensor:
        """Return edge-conditioned atomic basis features.

        ``edge_radial`` has shape ``[E, num_radial_basis]`` and is computed
        once from edge lengths before entering this block.

        TODO(luv): combine sender features, learned weights of ``edge_radial``,
        and e3nn real spherical harmonics of ``edge_vectors`` with component
        normalization.
        """
        sender_features = node_features[edge_index[0]].unsqueeze(-1)
        receiver_index = edge_index[1]
        ells = list(range(self.config.spherical_harmonics_max_ell + 1))
        normalize = True
        normalization = "component"
        spherical_tensors = spherical_harmonics(
            l=ells, x=edge_vectors, normalize=normalize, normalization=normalization
        )
        radial_weight = self.radial_mlp(edge_radial).reshape(
            -1,
            self.config.channels,
            self.config.spherical_harmonics_max_ell + 1,
        )

        split_sizes = [2 * ell + 1 for ell in ells]
        split_sperical_tensors = torch.split(spherical_tensors, split_sizes, dim=1)
        agg_messages = list()
        for ell in ells:
            radial_weight_mul = radial_weight[:, :, ell].unsqueeze(2)
            spherical_tensor_mul = split_sperical_tensors[ell].unsqueeze(1)
            edge_message = radial_weight_mul * spherical_tensor_mul * sender_features
            cumulative_message = torch.zeros(
                len(node_features), self.config.channels, split_sizes[ell]
            ).to(device=node_features.device, dtype=node_features.dtype)
            cumulative_message = cumulative_message.index_add_(
                0, receiver_index, edge_message
            )
            agg_messages.append(cumulative_message.flatten(start_dim=1))
        return torch.concat(agg_messages, dim=1)


def _wigner_nj(
    irrepss: tuple[o3.Irreps, ...],
    *,
    normalization: str,
    dtype: torch.dtype,
) -> list[tuple[o3.Irrep, Tensor]]:
    """Return every legal output irrep and its fixed real U-path tensor.

    TODO(luv):
    1. For one factor, return one identity selector per packed irrep block.
    2. Otherwise recursively couple the left factors, then attach each block
       of the final factor using the fixed real Wigner-3j tensor.
    3. Preserve one full packed-input axis per input factor and sort by output
       irrep so paths of the same type remain contiguous.
    """
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
    """Build one Eq. 10 output-irrep block.

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
        self.num_features = input_irreps.count((0, 1))
        self.coupling_irreps = o3.Irreps([mul_irrep.ir for mul_irrep in input_irreps])
        self.irrep_out = irrep_out
        self.correlation = correlation
        if self.correlation != 3:
            raise ValueError(
                "This learning implementation currently supports correlation=3 only."
            )
        self.num_elements = num_elements

        for nu in range(1, self.correlation + 1):
            u_tensor = u_matrix_real(
                coupling_irreps=self.coupling_irreps,
                irrep_out=self.irrep_out,
                correlation=nu,
            )
            self.register_buffer(f"U_matrix_{nu}", u_tensor)

        self.weights = nn.ParameterList()
        u_max = getattr(self, f"U_matrix_{self.correlation}")
        num_paths = u_max.size(-1)

        self.weights_max = nn.Parameter(
            torch.randn(
                self.num_elements,
                num_paths,
                self.num_features,
            )
            / num_paths
        )

    def forward(self, atomic_basis: Tensor, node_attrs: Tensor) -> Tensor:
        """Return this output irrep's `[N, channels * (2 * ell + 1)]` block.

        TODO(luv): contract live atomic-basis features with the fixed U paths
        and learned element-conditioned weights.
        """
        raise NotImplementedError


class SymmetricContraction(nn.Module):
    """Form Eq. 10 higher-order equivariant product features."""

    def __init__(self, config: MACEConfig):
        super().__init__()
        self.config = config
        self.input_irreps = o3.Irreps(
            f"{config.channels}x0e + {config.channels}x1o + "
            f"{config.channels}x2e + {config.channels}x3o"
        )
        self.output_irreps = o3.Irreps(config.hidden_irreps)
        # TODO(luv): follow the author implementation: make one `Contraction`
        # per requested output irrep. Each contraction uses fixed U matrices
        # (higher-order real CG coefficients), not manually staged
        # `o3.TensorProduct` modules or exposed intermediate irreps.

    def forward(self, atomic_basis: Tensor, node_attrs: Tensor) -> Tensor:
        """Return the Eq. 10 product basis with shape `[N, output_irreps.dim]`.

        TODO(luv): apply each author-style per-output-irrep contraction to
        `atomic_basis` and element one-hot `node_attrs`, then concatenate the
        output blocks. `U_matrix_real` supplies fixed geometry; learned
        element/channel weights remain separate.
        """
        raise NotImplementedError


class InteractionBlock(nn.Module):
    """Build Eq. 8 neighbour-aggregated equivariant messages."""

    def forward(
        self,
        node_features: Tensor,
        node_attrs: Tensor,
        edge_index: Tensor,
        edge_vectors: Tensor,
    ) -> Tensor:
        """Return one updated feature block per node.

        TODO(luv): assemble Eq. 8's edge message and neighbour aggregation,
        divide by the training-set average neighbour count, and apply the
        interaction-specific residual path. Eq. 10 is applied afterward by
        `SymmetricContraction`, not inside this block.
        """
        raise NotImplementedError


class MACE(nn.Module):
    """Two-interaction scalar-energy model with forces from autograd."""

    def __init__(self, config: MACEConfig):
        super().__init__()
        self.config = config
        # TODO(luv): construct embeddings, two interaction/product blocks, and
        # the linear then one-hidden-layer scalar readouts.

    def forward(
        self,
        atomic_numbers: Tensor,
        positions: Tensor,
        edge_index: Tensor,
        graph_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return total energies `[B]` and forces `[N, 3]`.

        TODO(luv): enable position gradients, derive edge vectors and lengths
        here (never in the data loader), sum invariant atomic readouts by graph,
        and obtain forces with :func:`forces_from_energy`.
        """
        raise NotImplementedError


def random_rotation(
    *, dtype: torch.dtype = torch.float64, device: torch.device | None = None
) -> Tensor:
    """Return a random proper 3D rotation for equivariance tests.

    TODO(luv): move the notebook's tested construction here only when the model
    symmetry test needs it.
    """
    raise NotImplementedError
