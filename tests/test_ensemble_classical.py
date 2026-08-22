# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classical ensemble tests: MB moments, reproducibility, Verlet conservation.

WP4 test contract items 1, 2, 3, 6.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock import constants
from cliffordclock.ensemble.classical import (
    _verlet_trajectory,
    propagate_verlet,
    sample_maxwell_boltzmann,
)
from cliffordclock.ensemble.species import get_species
from cliffordclock.ensemble.traps import HarmonicTrap

SR87 = get_species("Sr87")
N_MB = 100_000
TEMPERATURE_UK = 1.0


@pytest.fixture
def mb_trap() -> HarmonicTrap:
    return HarmonicTrap(omega_xyz=(2 * math.pi * 1e5, 2 * math.pi * 1.3e5, 2 * math.pi * 0.8e5))


def _single_particle_state() -> tuple[jax.Array, jax.Array]:
    positions = jnp.array([[1e-7, -5e-8, 2e-7]])
    velocities = jnp.array([[0.05, -0.02, 0.03]])
    return positions, velocities


# --- Maxwell-Boltzmann moments (test contract item 1) ---


def test_mb_mean_kinetic_energy_matches_equipartition(mb_trap: HarmonicTrap) -> None:
    """Mean KE per axis = (1/2) k_B T within 3-sigma statistical tolerance."""
    key = jax.random.PRNGKey(1)
    _, velocities = sample_maxwell_boltzmann(key, SR87, TEMPERATURE_UK, N_MB, mb_trap)

    temperature_k = TEMPERATURE_UK * 1e-6
    expected_ke = 0.5 * constants.BOLTZMANN_K * temperature_k
    # For v ~ Normal(0, kT/m): Var(0.5 m v^2) = 0.5 (kT)^2 exactly.
    stderr = math.sqrt(0.5) * constants.BOLTZMANN_K * temperature_k / math.sqrt(N_MB)

    ke_per_axis = 0.5 * SR87.mass_kg * jnp.mean(velocities**2, axis=0)
    for axis in range(3):
        assert abs(float(ke_per_axis[axis]) - expected_ke) < 3 * stderr


def test_mb_velocity_components_uncorrelated(mb_trap: HarmonicTrap) -> None:
    """Velocity components must show no pairwise correlation."""
    key = jax.random.PRNGKey(2)
    _, velocities = sample_maxwell_boltzmann(key, SR87, TEMPERATURE_UK, N_MB, mb_trap)
    corr = np.corrcoef(np.asarray(velocities), rowvar=False)
    threshold = 3 / math.sqrt(N_MB)  # 3-sigma bound for independent variables
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(corr[i, j]) < threshold


def test_mb_positions_match_trap_thermal_variance(mb_trap: HarmonicTrap) -> None:
    """Position variance per axis = k_B T / (m omega^2) within 3-sigma."""
    key = jax.random.PRNGKey(3)
    positions, _ = sample_maxwell_boltzmann(key, SR87, TEMPERATURE_UK, N_MB, mb_trap)
    expected_variance = mb_trap.thermal_position_variance(SR87.mass_kg, TEMPERATURE_UK)

    sample_variance = jnp.var(positions, axis=0, ddof=1)
    # Var(sample variance) ~= 2 sigma^4 / (N - 1) for a Gaussian sample.
    stderr = jnp.sqrt(2 * expected_variance**2 / (N_MB - 1))
    for axis in range(3):
        diff = abs(float(sample_variance[axis]) - float(expected_variance[axis]))
        assert diff < 3 * float(stderr[axis])


# --- Reproducibility (test contract item 2) ---


def test_mb_reproducible_with_same_key(mb_trap: HarmonicTrap) -> None:
    """Same PRNG key -> identical samples."""
    key = jax.random.PRNGKey(42)
    pos1, vel1 = sample_maxwell_boltzmann(key, SR87, TEMPERATURE_UK, 1000, mb_trap)
    pos2, vel2 = sample_maxwell_boltzmann(key, SR87, TEMPERATURE_UK, 1000, mb_trap)
    np.testing.assert_array_equal(pos1, pos2)
    np.testing.assert_array_equal(vel1, vel2)


def test_mb_different_keys_are_statistically_independent(mb_trap: HarmonicTrap) -> None:
    """Different PRNG keys -> statistically independent samples."""
    key_a, key_b = jax.random.split(jax.random.PRNGKey(7))
    pos_a, vel_a = sample_maxwell_boltzmann(key_a, SR87, TEMPERATURE_UK, N_MB, mb_trap)
    pos_b, vel_b = sample_maxwell_boltzmann(key_b, SR87, TEMPERATURE_UK, N_MB, mb_trap)

    assert not np.allclose(pos_a, pos_b)
    corr = np.corrcoef(np.asarray(vel_a[:, 0]), np.asarray(vel_b[:, 0]))[0, 1]
    assert abs(corr) < 3 / math.sqrt(N_MB)


def test_mb_outputs_are_float64_with_documented_shape(mb_trap: HarmonicTrap) -> None:
    key = jax.random.PRNGKey(9)
    positions, velocities = sample_maxwell_boltzmann(key, SR87, TEMPERATURE_UK, 10, mb_trap)
    assert positions.dtype == jnp.float64
    assert velocities.dtype == jnp.float64
    assert positions.shape == (10, 3)
    assert velocities.shape == (10, 3)


# --- Verlet integrator (test contract item 3) ---


def _shadow_energy(
    trap: HarmonicTrap, mass_kg: float, dt: float, pos_traj: jax.Array, vel_traj: jax.Array
) -> jax.Array:
    """The exact discrete invariant of velocity-Verlet applied to a harmonic oscillator.

    AMBIGUITY (see the WP4 builder report): at dt = 2 pi / (100 omega), the
    literal continuum energy trap.energy() = (1/2)m v^2 + (1/2)m omega^2 x^2
    is *not* a quantity any velocity-Verlet implementation holds to 1e-8
    relative accuracy pointwise -- it has an intrinsic, purely periodic
    O((omega dt)^2) ~ 1e-3 relative bounded oscillation (verified
    numerically), independent of implementation correctness, because
    velocity-Verlet applied to a linear oscillator is an exact symplectic
    *rotation* at a slightly shifted discrete frequency Omega != omega, with
    cos(Omega dt) = 1 - (omega dt)^2 / 2. Naive endpoint or windowed-average
    comparisons of the continuum energy cannot reach 1e-8 within a 1e5-step
    budget (confirmed numerically: residual decays only like a Dirichlet
    kernel, ~1e-8 would need ~1e7 steps).

    What velocity-Verlet *does* exactly conserve (to floating-point
    roundoff, not just approximately) for a linear oscillator is a nearby
    "shadow" quadratic form -- the standard backward-error-analysis
    invariant for symplectic integrators (Hairer, Lubich & Wanner,
    "Geometric Numerical Integration", chapter IX). Diagonalizing the
    one-step map's symplectic invariance condition M^T A M = A for the
    velocity-Verlet map on (x, v) gives, per axis:

        H_shadow = (1/2) m [ omega^2 x^2 + v^2 / (1 - (omega dt / 2)^2) ]

    which collapses to the continuum energy as dt -> 0 and is conserved by
    construction (verified numerically to ~1e-14 relative over 1e5 steps
    at this dt). This -- not the raw bounded-oscillating continuum energy
    -- is what "relative energy drift < 1e-8" in the WP4 test contract can
    actually mean for a correct implementation at the specified dt; see
    also `test_verlet_energy_oscillation_is_bounded_not_secular` below,
    which checks the literal continuum energy for the qualitative
    (boundedness) signature instead of a literal 1e-8 pointwise bound.
    """
    omega = jnp.asarray(trap.omega_xyz, dtype=jnp.float64)
    correction = 1.0 - (omega * dt / 2.0) ** 2
    return 0.5 * mass_kg * jnp.sum(omega**2 * pos_traj**2 + vel_traj**2 / correction, axis=-1)


@pytest.mark.slow
def test_verlet_conserves_the_exact_discrete_invariant() -> None:
    """Velocity-Verlet exactly conserves its shadow Hamiltonian to ~roundoff.

    Harmonic trap, 1e5 steps at dt = 2 pi / (100 omega) (WP4 test contract
    item 3). See `_shadow_energy` for why the shadow invariant, rather than
    the raw continuum KE+PE, is the quantity asserted against the
    1e-8 relative tolerance.
    """
    trap = HarmonicTrap(omega_xyz=(1e6, 1e6, 1e6))
    mass_kg = SR87.mass_kg
    omega = 1e6
    dt = 2 * math.pi / (100 * omega)
    num_steps = 100_000

    positions, velocities = _single_particle_state()
    pos_traj, vel_traj = _verlet_trajectory(trap, positions, velocities, dt, num_steps)

    shadow_energy = _shadow_energy(trap, mass_kg, dt, pos_traj[0], vel_traj[0])
    relative_drift = abs(float(shadow_energy[-1] - shadow_energy[0])) / abs(float(shadow_energy[0]))
    assert relative_drift < 1e-8


@pytest.mark.slow
def test_verlet_energy_oscillation_is_bounded_not_secular() -> None:
    """The literal continuum energy's oscillation band does not grow with run length.

    Symplectic integrators exhibit a bounded energy oscillation that does
    not grow over time, unlike non-symplectic integrators (e.g. explicit
    Euler), whose energy error grows secularly (unboundedly) with the
    number of steps. This is the qualitative content of "no drift" for the
    literal trap.energy() (KE + PE), complementing the quantitative
    shadow-invariant check above.
    """
    trap = HarmonicTrap(omega_xyz=(1e6, 1e6, 1e6))
    mass_kg = SR87.mass_kg
    omega = 1e6
    dt = 2 * math.pi / (100 * omega)

    positions, velocities = _single_particle_state()

    def oscillation_band(num_steps: int) -> float:
        pos_traj, vel_traj = _verlet_trajectory(trap, positions, velocities, dt, num_steps)
        energy = trap.energy(mass_kg, pos_traj[0], vel_traj[0])
        return float((jnp.max(energy) - jnp.min(energy)) / jnp.mean(energy))

    band_short = oscillation_band(100)  # 1 nominal period
    band_long = oscillation_band(100_000)  # 1000 nominal periods

    # Both bands are set by the same O((omega dt)^2) mechanism; a
    # non-symplectic (drifting) integrator would instead show band_long
    # growing roughly linearly with the number of periods (~1000x here).
    assert band_long < 10 * band_short


def test_verlet_is_time_reversible() -> None:
    """Forward num_steps then backward (v -> -v) num_steps returns to the start."""
    trap = HarmonicTrap(omega_xyz=(1e6, 1.3e6, 0.7e6))
    omega_max = 1.3e6
    dt = 2 * math.pi / (100 * omega_max)
    num_steps = 10_000

    positions0, velocities0 = _single_particle_state()
    pos_traj, vel_traj = _verlet_trajectory(trap, positions0, velocities0, dt, num_steps)
    pos_final, vel_final = pos_traj[:, -1, :], vel_traj[:, -1, :]

    pos_back, _ = _verlet_trajectory(trap, pos_final, -vel_final, dt, num_steps)
    pos_return = pos_back[:, -1, :]

    rel_error = jnp.linalg.norm(pos_return - positions0) / jnp.linalg.norm(positions0)
    assert float(rel_error) < 1e-9


def test_propagate_verlet_shape_dtype_and_initial_state() -> None:
    trap = HarmonicTrap(omega_xyz=(1e6, 1e6, 1e6))
    positions, velocities = _single_particle_state()
    positions = jnp.tile(positions, (5, 1))
    velocities = jnp.tile(velocities, (5, 1))
    dt = 2 * math.pi / (100 * 1e6)
    num_steps = 50
    traj = propagate_verlet(trap, positions, velocities, dt, num_steps)
    assert traj.shape == (5, num_steps + 1, 3)
    assert traj.dtype == jnp.float64
    np.testing.assert_array_equal(traj[:, 0, :], positions)


def test_propagate_verlet_is_jit_safe() -> None:
    trap = HarmonicTrap(omega_xyz=(1e6, 1e6, 1e6))
    positions, velocities = _single_particle_state()

    jitted = jax.jit(propagate_verlet, static_argnames=("trap", "num_steps"))
    dt = 2 * math.pi / (100 * 1e6)
    traj_jit = jitted(trap, positions, velocities, dt, 20)
    traj_eager = propagate_verlet(trap, positions, velocities, dt, 20)
    np.testing.assert_allclose(traj_jit, traj_eager, rtol=1e-14, atol=0)
