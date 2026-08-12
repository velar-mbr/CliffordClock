# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: N806
"""HarmonicTrap unit tests: force law, energy, thermal variance."""

import jax.numpy as jnp
import numpy as np

from cliffordclock import constants
from cliffordclock.ensemble.traps import HarmonicTrap, Trap


def test_harmonic_trap_satisfies_trap_protocol() -> None:
    trap = HarmonicTrap(omega_xyz=(1.0, 2.0, 3.0))
    assert isinstance(trap, Trap)


def test_acceleration_is_mass_independent_and_matches_force_law() -> None:
    trap = HarmonicTrap(omega_xyz=(1e6, 2e6, 3e6), center=(1e-7, -2e-7, 0.0))
    positions = jnp.array([[2e-7, -1e-7, 5e-8], [0.0, 0.0, 0.0]])
    acc = trap.acceleration(positions)
    omega = jnp.array(trap.omega_xyz)
    center = jnp.array(trap.center)
    expected = -(omega**2) * (positions - center)
    np.testing.assert_allclose(acc, expected, rtol=1e-14, atol=0)


def test_force_equals_mass_times_acceleration() -> None:
    trap = HarmonicTrap(omega_xyz=(1e6, 2e6, 3e6))
    positions = jnp.array([[1e-7, 2e-7, -3e-7]])
    mass_kg = 1.45e-25
    np.testing.assert_allclose(
        trap.force(mass_kg, positions), mass_kg * trap.acceleration(positions), rtol=1e-14, atol=0
    )


def test_potential_energy_matches_analytical_harmonic_formula() -> None:
    trap = HarmonicTrap(omega_xyz=(1e6, 2e6, 3e6))
    positions = jnp.array([[1e-7, -2e-7, 3e-7]])
    mass_kg = 1.45e-25
    expected = (
        0.5 * mass_kg * sum((w * x) ** 2 for w, x in zip(trap.omega_xyz, positions[0], strict=True))
    )
    energy = trap.potential_energy(mass_kg, positions)
    np.testing.assert_allclose(float(energy[0]), float(expected), rtol=1e-12, atol=0)


def test_kinetic_plus_potential_equals_energy() -> None:
    trap = HarmonicTrap(omega_xyz=(1e6, 2e6, 3e6))
    positions = jnp.array([[1e-7, -2e-7, 3e-7]])
    velocities = jnp.array([[10.0, -5.0, 2.0]])
    mass_kg = 1.45e-25
    expected = trap.kinetic_energy(mass_kg, velocities) + trap.potential_energy(mass_kg, positions)
    np.testing.assert_allclose(
        trap.energy(mass_kg, positions, velocities), expected, rtol=1e-14, atol=0
    )


def test_thermal_position_variance_matches_equipartition() -> None:
    trap = HarmonicTrap(omega_xyz=(1e6, 1e6, 1e6))
    mass_kg = 1.45e-25
    temperature_uK = 1.0
    variance = trap.thermal_position_variance(mass_kg, temperature_uK)
    expected = constants.BOLTZMANN_K * 1e-6 / (mass_kg * 1e6**2)
    np.testing.assert_allclose(variance, jnp.full(3, expected), rtol=1e-12, atol=0)


def test_outputs_are_float64() -> None:
    trap = HarmonicTrap(omega_xyz=(1e6, 2e6, 3e6))
    positions = jnp.zeros((4, 3))
    velocities = jnp.zeros((4, 3))
    assert trap.acceleration(positions).dtype == jnp.float64
    assert trap.potential_energy(1e-25, positions).dtype == jnp.float64
    assert trap.energy(1e-25, positions, velocities).dtype == jnp.float64
    assert trap.thermal_position_variance(1e-25, 1.0).dtype == jnp.float64
