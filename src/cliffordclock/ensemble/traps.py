# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: N803
# `temperature_uK` is the CONVENTIONS.md-mandated parameter name at the
# sampler API boundary (docs/CONVENTIONS.md section 10: "temperature uK at
# sampler APIs"); pep8-naming's N803 (lowercase-argument-name) would
# otherwise flag the embedded capital K.
"""Trap potential models for the ensemble sampler.

Sprint 1 implements exactly one trap model, `HarmonicTrap`. Real
(anharmonic, CAD-field-derived) trap potentials are out of scope for now.
To keep `ensemble.classical` and `ensemble.lattice` from hard-coding
harmonicity, they are written against the `Trap` structural protocol below
rather than against `HarmonicTrap` directly, so a future trap model can be
substituted without changing sampler signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from cliffordclock import constants


@runtime_checkable
class Trap(Protocol):
    """Structural interface every trap model must satisfy.

    A trap model owns the force law and the classical thermal-equilibrium
    position distribution needed by `cliffordclock.ensemble.classical`. It
    is deliberately minimal: `cliffordclock.ensemble.lattice` (quantum
    motional states) is intrinsically a harmonic-oscillator construction
    and therefore type-hints `HarmonicTrap` directly rather than this
    protocol.
    """

    def acceleration(self, positions: jax.Array) -> jax.Array:
        """Acceleration at `positions`.

        Parameters
        ----------
        positions : jax.Array
            Shape ``(M, 3)``, meters.

        Returns
        -------
        jax.Array
            Shape ``(M, 3)``, m/s^2.
        """
        ...

    def potential_energy(self, mass_kg: float, positions: jax.Array) -> jax.Array:
        """Potential energy at `positions` for a particle of mass `mass_kg`.

        Parameters
        ----------
        mass_kg : float
            Particle mass, kilograms.
        positions : jax.Array
            Shape ``(M, 3)``, meters.

        Returns
        -------
        jax.Array
            Shape ``(M,)``, joules.
        """
        ...

    def sample_thermal_positions(
        self, key: jax.Array, mass_kg: float, temperature_uK: float, num: int
    ) -> jax.Array:
        """Sample positions from the trap's classical thermal-equilibrium distribution.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        mass_kg : float
            Particle mass, kilograms.
        temperature_uK : float
            Ensemble temperature, microkelvin.
        num : int
            Number of samples to draw.

        Returns
        -------
        jax.Array
            Shape ``(num, 3)``, meters, dtype float64.
        """
        ...


@dataclass(frozen=True)
class HarmonicTrap:
    """A 3D harmonic (parabolic) trap.

    ``V(r) = (1/2) m * sum_k omega_k^2 (r_k - center_k)^2``

    This is the only trap model in Sprint 1 (WP4 scope: "Minimal harmonic
    trap abstraction (MVP)"). Real trap potentials (anharmonic,
    CAD-field-derived) arrive with the field-importer integration in a
    later sprint.

    Parameters
    ----------
    omega_xyz : tuple[float, float, float]
        Angular trap frequencies along x, y, z, rad/s.
    center : tuple[float, float, float]
        Trap center, meters. Defaults to the origin.
    """

    omega_xyz: tuple[float, float, float]
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def acceleration(self, positions: jax.Array) -> jax.Array:
        """``a = -omega^2 (r - center)``, m/s^2, shape ``(M, 3)``.

        Mass-independent: `HarmonicTrap` is parameterized by angular
        frequency (rather than a spring constant), so Newton's second law
        ``a = F/m = -(m omega^2 * displacement) / m`` cancels the mass.
        This is why `cliffordclock.ensemble.classical.propagate_verlet`
        does not take a mass argument.
        """
        omega = jnp.asarray(self.omega_xyz, dtype=jnp.float64)
        center = jnp.asarray(self.center, dtype=jnp.float64)
        positions = jnp.asarray(positions, dtype=jnp.float64)
        return -(omega**2) * (positions - center)

    def force(self, mass_kg: float, positions: jax.Array) -> jax.Array:
        """``F = m * a``, newtons, shape ``(M, 3)``."""
        return mass_kg * self.acceleration(positions)

    def potential_energy(self, mass_kg: float, positions: jax.Array) -> jax.Array:
        """``V = (1/2) m sum_k omega_k^2 (r_k - center_k)^2``, joules, shape ``(M,)``."""
        omega = jnp.asarray(self.omega_xyz, dtype=jnp.float64)
        center = jnp.asarray(self.center, dtype=jnp.float64)
        positions = jnp.asarray(positions, dtype=jnp.float64)
        displacement = positions - center
        return 0.5 * mass_kg * jnp.sum((omega * displacement) ** 2, axis=-1)

    def kinetic_energy(self, mass_kg: float, velocities: jax.Array) -> jax.Array:
        """``T = (1/2) m |v|^2``, joules, shape ``(M,)``."""
        velocities = jnp.asarray(velocities, dtype=jnp.float64)
        return 0.5 * mass_kg * jnp.sum(velocities**2, axis=-1)

    def energy(self, mass_kg: float, positions: jax.Array, velocities: jax.Array) -> jax.Array:
        """Total mechanical energy ``T + V``, joules, shape ``(M,)``.

        The analytical energy referenced in WP4 scope item 2, used by the
        Verlet energy-conservation test.
        """
        return self.kinetic_energy(mass_kg, velocities) + self.potential_energy(mass_kg, positions)

    def thermal_position_variance(self, mass_kg: float, temperature_uK: float) -> jax.Array:
        """Equipartition position variance per axis, ``k_B T / (m omega_k^2)``, m^2.

        Returns
        -------
        jax.Array
            Shape ``(3,)``.
        """
        temperature_k = temperature_uK * 1e-6
        omega = jnp.asarray(self.omega_xyz, dtype=jnp.float64)
        return constants.BOLTZMANN_K * temperature_k / (mass_kg * omega**2)

    def sample_thermal_positions(
        self, key: jax.Array, mass_kg: float, temperature_uK: float, num: int
    ) -> jax.Array:
        """Sample ``num`` positions from the trap's Gaussian thermal equilibrium.

        Each axis is an independent Gaussian centered on `center` with
        variance ``k_B T / (m omega_k^2)`` (equipartition for a harmonic
        trap).

        Returns
        -------
        jax.Array
            Shape ``(num, 3)``, meters, dtype float64.
        """
        variance = self.thermal_position_variance(mass_kg, temperature_uK)
        sigma = jnp.sqrt(variance)
        center = jnp.asarray(self.center, dtype=jnp.float64)
        noise = jax.random.normal(key, shape=(num, 3), dtype=jnp.float64)
        return center + sigma * noise
