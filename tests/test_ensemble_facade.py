# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: N803
"""EnsembleSampler facade integration tests."""

import math

import jax
import numpy as np
import pytest

from cliffordclock.ensemble import EnsembleSampler, HarmonicTrap
from cliffordclock.ensemble.classical import sample_maxwell_boltzmann
from cliffordclock.ensemble.lattice import hermite_gaussian_nodes
from cliffordclock.ensemble.species import get_species


def test_facade_generate_classical_ensemble_matches_module_function() -> None:
    species = get_species("Sr87")
    trap = HarmonicTrap(omega_xyz=(1e5, 1e5, 1e5))
    sampler = EnsembleSampler(species=species, trap_type=trap, temperature_uK=2.0)

    key = jax.random.PRNGKey(11)
    positions, velocities = sampler.generate_classical_ensemble(key, 50)
    expected_positions, expected_velocities = sample_maxwell_boltzmann(key, species, 2.0, 50, trap)

    np.testing.assert_array_equal(positions, expected_positions)
    np.testing.assert_array_equal(velocities, expected_velocities)


def test_facade_generate_lattice_quadrature_nodes_matches_module_function() -> None:
    species = get_species("Yb171")
    trap = HarmonicTrap(omega_xyz=(2 * math.pi * 4e5, 2 * math.pi * 4e5, 2 * math.pi * 4e5))
    sampler = EnsembleSampler(species=species, trap_type=trap, temperature_uK=1.0)

    nodes, weights = sampler.generate_lattice_quadrature_nodes((0, 1, 2), 10)
    expected_nodes, expected_weights = hermite_gaussian_nodes(species, trap, (0, 1, 2), 10)

    np.testing.assert_array_equal(nodes, expected_nodes)
    np.testing.assert_array_equal(weights, expected_weights)


class _NotAHarmonicTrap:
    """A minimal `Trap`-protocol implementer that is not a `HarmonicTrap`."""

    def acceleration(self, positions: jax.Array) -> jax.Array:
        return positions

    def potential_energy(self, mass_kg: float, positions: jax.Array) -> jax.Array:
        return positions[..., 0]

    def sample_thermal_positions(
        self, key: jax.Array, mass_kg: float, temperature_uK: float, num: int
    ) -> jax.Array:
        return jax.random.normal(key, shape=(num, 3))


def test_facade_lattice_requires_harmonic_trap() -> None:
    species = get_species("Al27+")
    sampler = EnsembleSampler(species=species, trap_type=_NotAHarmonicTrap(), temperature_uK=1.0)
    with pytest.raises(TypeError):
        sampler.generate_lattice_quadrature_nodes((0, 0, 0), 5)
