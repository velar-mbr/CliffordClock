# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: N815
# `temperature_uK` is the WP4/CONVENTIONS.md-mandated name at the sampler
# API boundary (docs/CONVENTIONS.md section 10: "temperature uK at sampler
# APIs"); pep8-naming's N815 (mixedCase class-scope variable) would
# otherwise flag the embedded capital K on this dataclass field.
"""Ensemble samplers (Maxwell-Boltzmann Monte-Carlo, Hermite-Gaussian quadrature).

Produces *where the atoms are*: classical thermal trajectories
(transportable / ion-trap regime, `cliffordclock.ensemble.classical`) or
static quadrature node sets for quantum motional states in optical
lattices (`cliffordclock.ensemble.lattice`). The rotor path integrator
consumes the output as position arrays ``(M, T, 3)`` or static node
sets ``(M, 3)`` with weights.

`EnsembleSampler` below is a thin façade over `cliffordclock.ensemble.classical`
and `cliffordclock.ensemble.lattice`; the underlying functions are also
directly importable for callers that want explicit control (e.g. explicit
PRNG key threading without going through the façade).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax

from cliffordclock.ensemble import classical, lattice
from cliffordclock.ensemble.species import Species, get_species
from cliffordclock.ensemble.traps import HarmonicTrap, Trap

__all__ = [
    "EnsembleSampler",
    "HarmonicTrap",
    "Trap",
    "Species",
    "get_species",
]


@dataclass(frozen=True)
class EnsembleSampler:
    """Unified front door over the classical and quantum-lattice ensemble regimes.

    Parameters
    ----------
    species : Species
        Atomic species (see `cliffordclock.ensemble.species.get_species`).
    trap_type : Trap
        Trap model instance (e.g. `HarmonicTrap`). Sprint 1 provides
        exactly one trap model; the parameter is named ``trap_type`` to
        match the public API contract, even though it currently accepts
        a concrete trap *instance* rather than a type tag.
    temperature_uK : float
        Ensemble temperature, microkelvin.
    """

    species: Species
    trap_type: Trap
    temperature_uK: float

    def generate_classical_ensemble(
        self, key: jax.Array, num_particles: int
    ) -> tuple[jax.Array, jax.Array]:
        """Sample a classical Maxwell-Boltzmann ensemble.

        Delegates to `cliffordclock.ensemble.classical.sample_maxwell_boltzmann`.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key. Required: this sampler uses explicit PRNG key
            threading with no global RNG state, which is why this
            parameter is present even though it does not appear in the
            abbreviated method signature some callers may expect.
        num_particles : int
            Number of particles to sample.

        Returns
        -------
        positions : jax.Array
            Shape ``(num_particles, 3)``, meters, dtype float64.
        velocities : jax.Array
            Shape ``(num_particles, 3)``, m/s, dtype float64.
        """
        return classical.sample_maxwell_boltzmann(
            key, self.species, self.temperature_uK, num_particles, self.trap_type
        )

    def generate_lattice_quadrature_nodes(
        self, motional_n: tuple[int, int, int], n_quad: int
    ) -> tuple[jax.Array, jax.Array]:
        """Generate Hermite-Gauss quadrature nodes/weights for a motional eigenstate.

        Delegates to `cliffordclock.ensemble.lattice.hermite_gaussian_nodes`.
        Requires `self.trap_type` to be a `HarmonicTrap` (the only trap
        model for which quantum motional eigenstates are defined in
        Sprint 1).

        Parameters
        ----------
        motional_n : tuple[int, int, int]
            Motional quantum numbers ``(nx, ny, nz)``.
        n_quad : int
            Number of 1D Gauss-Hermite quadrature points per axis.
            Required: this parameter controls quadrature accuracy and is
            not optional, even though it does not appear in the
            abbreviated method signature some callers may expect.

        Returns
        -------
        nodes : jax.Array
            Shape ``(n_quad**3, 3)``, meters, dtype float64.
        weights : jax.Array
            Shape ``(n_quad**3,)``, dimensionless, dtype float64.
        """
        if not isinstance(self.trap_type, HarmonicTrap):
            raise TypeError(
                "generate_lattice_quadrature_nodes requires a HarmonicTrap "
                f"(got {type(self.trap_type).__name__}); Sprint 1 defines quantum "
                "motional eigenstates only for the harmonic trap."
            )
        return lattice.hermite_gaussian_nodes(self.species, self.trap_type, motional_n, n_quad)
