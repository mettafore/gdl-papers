"""Learning-owned MACE model contracts for Eqs. 8--11."""

from dataclasses import dataclass

import torch
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
    """One allowed irreducible tensor-product path."""

    input_ells: tuple[int, ...]
    output_ell: int


def enumerate_coupling_routes(
    input_ells: tuple[int, ...], *, output_max_ell: int, correlation: int
) -> tuple[CouplingRoute, ...]:
    """Enumerate allowed angular-momentum paths up to the requested order.

    TODO(luv): apply the triangle rule one coupling at a time, retain the route
    history (not only the final ell), enforce the requested output range, and
    then add parity when moving from SO(3) intuition to full O(3) irreps.
    """
    raise NotImplementedError


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
        raise NotImplementedError


class AtomicBasis(nn.Module):
    """Construct Eq. 8 neighbour features from radial and angular factors."""

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_vectors: Tensor,
    ) -> Tensor:
        """Return edge-conditioned atomic basis features.

        TODO(luv): combine sender features, learned radial weights, and e3nn real
        spherical harmonics with component normalization.
        """
        raise NotImplementedError


class SymmetricContraction(nn.Module):
    """Form Eq. 10 higher-order equivariant product features."""

    def forward(self, atomic_basis: Tensor, node_attrs: Tensor) -> Tensor:
        """Contract allowed coupling routes through correlation order `nu=3`.

        TODO(luv): use the enumerated routes and CG coefficients; keep channel
        mixing shared across every `m` within the same irrep.
        """
        raise NotImplementedError


class InteractionBlock(nn.Module):
    """Aggregate Eq. 11 messages and update node states equivariantly."""

    def forward(
        self,
        node_features: Tensor,
        node_attrs: Tensor,
        edge_index: Tensor,
        edge_vectors: Tensor,
    ) -> Tensor:
        """Return one updated feature block per node.

        TODO(luv): assemble Eq. 8 -> Eq. 10 -> neighbour aggregation, divide by
        the training-set average neighbour count, and apply the interaction-
        specific residual path described in the supplement.
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
