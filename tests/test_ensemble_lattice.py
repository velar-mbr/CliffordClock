# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: N806
"""Hermite-Gauss quadrature tests: exactness, thermal occupation.

WP4 test contract item 4, plus shapes/dtype (item 6).
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock import constants
from cliffordclock.ensemble.lattice import hermite_gaussian_nodes, thermal_occupation
from cliffordclock.ensemble.species import get_species
from cliffordclock.ensemble.traps import HarmonicTrap

SR87 = get_species("Sr87")
N_QUAD = 12  # 2*N_QUAD-1-2*n_max=17 >= tested polynomial degrees (up to 4) for n_max=3


@pytest.fixture
def trap() -> HarmonicTrap:
    return HarmonicTrap(omega_xyz=(2 * math.pi * 5e5, 2 * math.pi * 6e5, 2 * math.pi * 7e5))


def _x0(trap: HarmonicTrap) -> np.ndarray:
    """Per-axis harmonic-oscillator length scale sqrt(hbar / (m omega))."""
    omega = np.asarray(trap.omega_xyz)
    return np.asarray(np.sqrt(constants.HBAR / (SR87.mass_kg * omega)))


@pytest.mark.parametrize("motional_n", [(0, 0, 0), (1, 2, 3)])
def test_quadrature_reproduces_analytical_second_moment(
    trap: HarmonicTrap, motional_n: tuple[int, int, int]
) -> None:
    """<x_k^2>_n = x0_k^2 (n_k + 1/2), exact to 1e-12 relative."""
    nodes, weights = hermite_gaussian_nodes(SR87, trap, motional_n, N_QUAD)
    x0 = _x0(trap)

    assert float(jnp.sum(weights)) == pytest.approx(1.0, rel=1e-12, abs=0)

    for axis in range(3):
        second_moment = float(jnp.sum(weights * nodes[:, axis] ** 2))
        expected = x0[axis] ** 2 * (motional_n[axis] + 0.5)
        assert second_moment == pytest.approx(expected, rel=1e-12, abs=0)


@pytest.mark.parametrize("motional_n", [(0, 0, 0), (1, 2, 3)])
def test_quadrature_reproduces_analytical_fourth_moment(
    trap: HarmonicTrap, motional_n: tuple[int, int, int]
) -> None:
    """<x_k^4>_n = x0_k^4 (6 n_k^2 + 6 n_k + 3) / 4, exact to 1e-12 relative.

    Standard ladder-operator result for a quantum harmonic oscillator Fock
    state: <n|(a+a^dagger)^4|n> = 6n^2+6n+3.
    """
    nodes, weights = hermite_gaussian_nodes(SR87, trap, motional_n, N_QUAD)
    x0 = _x0(trap)

    for axis in range(3):
        n = motional_n[axis]
        fourth_moment = float(jnp.sum(weights * nodes[:, axis] ** 4))
        expected = x0[axis] ** 4 * (6 * n**2 + 6 * n + 3) / 4
        assert fourth_moment == pytest.approx(expected, rel=1e-12, abs=0)


def test_quadrature_cross_axis_moments_vanish_by_symmetry(trap: HarmonicTrap) -> None:
    """<x y>_n = 0 exactly (odd-parity integrand factorizes per axis)."""
    nodes, weights = hermite_gaussian_nodes(SR87, trap, (1, 2, 3), N_QUAD)
    cross = float(jnp.sum(weights * nodes[:, 0] * nodes[:, 1]))
    x0 = _x0(trap)
    scale = x0[0] * x0[1]
    assert abs(cross) < 1e-12 * scale


def test_quadrature_shapes_and_dtype(trap: HarmonicTrap) -> None:
    nodes, weights = hermite_gaussian_nodes(SR87, trap, (0, 0, 0), N_QUAD)
    assert nodes.shape == (N_QUAD**3, 3)
    assert weights.shape == (N_QUAD**3,)
    assert nodes.dtype == jnp.float64
    assert weights.dtype == jnp.float64


def test_quadrature_nodes_centered_on_trap_center() -> None:
    center = (1e-6, -2e-6, 3e-6)
    trap = HarmonicTrap(omega_xyz=(2 * math.pi * 5e5,) * 3, center=center)
    nodes, weights = hermite_gaussian_nodes(SR87, trap, (0, 0, 0), N_QUAD)
    mean_position = jnp.sum(weights[:, None] * nodes, axis=0)
    np.testing.assert_allclose(mean_position, jnp.asarray(center), atol=1e-20)


def test_thermal_occupation_matches_bose_einstein_formula(trap: HarmonicTrap) -> None:
    temperature_uK = 5.0
    n_bar = thermal_occupation(temperature_uK, trap)
    temperature_k = temperature_uK * 1e-6
    omega = np.asarray(trap.omega_xyz)
    expected = 1.0 / (np.exp(constants.HBAR * omega / (constants.BOLTZMANN_K * temperature_k)) - 1)
    np.testing.assert_allclose(n_bar, expected, rtol=1e-10, atol=0)
    assert n_bar.dtype == jnp.float64
    assert n_bar.shape == (3,)


def test_thermal_occupation_decreases_with_lower_temperature(trap: HarmonicTrap) -> None:
    n_hot = thermal_occupation(10.0, trap)
    n_cold = thermal_occupation(0.1, trap)
    assert bool(jnp.all(n_cold < n_hot))
