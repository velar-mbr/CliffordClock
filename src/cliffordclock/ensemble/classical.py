# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: N803
# `temperature_uK` is the CONVENTIONS.md-mandated parameter name at the
# sampler API boundary (docs/CONVENTIONS.md section 10: "temperature uK at
# sampler APIs"); pep8-naming's N803 (lowercase-argument-name) would
# otherwise flag the embedded capital K.
"""Classical thermal ensembles: Maxwell-Boltzmann sampling and Verlet propagation.

Used for the transportable / ion-trap regime, where atoms are treated as
classical point particles in thermal equilibrium with a trap.

Units at the API boundary follow ``docs/CONVENTIONS.md`` section 10: SI in,
SI out (positions in meters, velocities in m/s, time in seconds,
temperature in microkelvin), fp64 throughout.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from cliffordclock import constants
from cliffordclock.ensemble.species import Species
from cliffordclock.ensemble.traps import HarmonicTrap, Trap


def sample_maxwell_boltzmann(
    key: jax.Array,
    species: Species,
    temperature_uK: float,
    num: int,
    trap: Trap,
    *,
    squeezing_r: tuple[float, float, float] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Sample a classical thermal ensemble: positions and Maxwell-Boltzmann velocities.

    Positions are drawn from `trap`'s classical thermal-equilibrium spatial
    distribution (`trap.sample_thermal_positions`); velocities are drawn
    independently, componentwise, from the Maxwell-Boltzmann distribution
    ``v_k ~ Normal(0, k_B T / m)``. The PRNG key is split explicitly (no
    global RNG state), so repeated calls with the same `key` are exactly
    reproducible and calls with different keys are statistically
    independent.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    species : Species
        Atomic species (supplies mass).
    temperature_uK : float
        Ensemble temperature, microkelvin.
    num : int
        Number of particles to sample.
    trap : Trap
        Trap model supplying the thermal position distribution.
    squeezing_r : tuple[float, float, float], optional
        Per-axis squeezing parameter `r` (WP31, CONVENTIONS.md section 8's
        E39 squeezed-motional-state input), `None` by default -- today's
        unsqueezed thermal sampling, reproduced bitwise (this keyword is
        not even read in that case). When given, this trap-frame phase
        space's POSITION quadrature variance is scaled by ``exp(-2r)``
        and the VELOCITY quadrature variance by ``exp(+2r)`` per axis (the
        engine's convention: `r > 0` squeezes position and antisqueezes
        velocity, the standard optics/motional-squeezing sign choice,
        chosen so the position variance -- the quantity most directly tied
        to a field-gradient-driven dephasing spread -- shrinks for
        positive `r`; the product of the two quadrature standard
        deviations, and hence the Heisenberg-limited phase-space area, is
        preserved exactly since ``exp(-r)*exp(+r) = 1``). Requires
        `trap` to be a :class:`~cliffordclock.ensemble.traps.HarmonicTrap`
        (the only trap model this MVP's thermal-position sampling
        supports; see :meth:`~cliffordclock.ensemble.traps.HarmonicTrap.thermal_position_variance`).

    Returns
    -------
    positions : jax.Array
        Shape ``(num, 3)``, meters, dtype float64.
    velocities : jax.Array
        Shape ``(num, 3)``, m/s, dtype float64.

    Raises
    ------
    TypeError
        `squeezing_r` is given but `trap` is not a `HarmonicTrap`.
    """
    key_pos, key_vel = jax.random.split(key)
    temperature_k = temperature_uK * 1e-6

    if squeezing_r is None:
        positions = trap.sample_thermal_positions(key_pos, species.mass_kg, temperature_uK, num)
        sigma = jnp.sqrt(constants.BOLTZMANN_K * temperature_k / species.mass_kg)
        noise = jax.random.normal(key_vel, shape=(num, 3), dtype=jnp.float64)
        velocities = sigma * noise
    else:
        if not isinstance(trap, HarmonicTrap):
            raise TypeError(
                "sample_maxwell_boltzmann: squeezing_r requires trap to be a "
                f"HarmonicTrap (the only trap model whose thermal position variance "
                f"this MVP can rescale); got {type(trap).__name__}"
            )
        r = jnp.asarray(squeezing_r, dtype=jnp.float64)
        position_variance = trap.thermal_position_variance(species.mass_kg, temperature_uK)
        sigma_pos = jnp.sqrt(position_variance) * jnp.exp(-r)
        center = jnp.asarray(trap.center, dtype=jnp.float64)
        noise_pos = jax.random.normal(key_pos, shape=(num, 3), dtype=jnp.float64)
        positions = center + sigma_pos * noise_pos

        sigma_v0 = jnp.sqrt(constants.BOLTZMANN_K * temperature_k / species.mass_kg)
        sigma_v = sigma_v0 * jnp.exp(r)
        noise_vel = jax.random.normal(key_vel, shape=(num, 3), dtype=jnp.float64)
        velocities = sigma_v * noise_vel

    return jnp.asarray(positions, dtype=jnp.float64), jnp.asarray(velocities, dtype=jnp.float64)


def _verlet_trajectory(
    trap: Trap,
    positions: jax.Array,
    velocities: jax.Array,
    dt: float,
    num_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """Internal: velocity-Verlet trajectory carrying both positions and velocities.

    `propagate_verlet` (the public API) returns only the position
    trajectory, per its WP4-specified contract. This helper additionally
    returns the velocity trajectory, needed internally (and by
    ``tests/test_ensemble_classical.py``) for energy-conservation and
    time-reversibility checks, which require the velocity state alongside
    position.

    Returns
    -------
    position_trajectory : jax.Array
        Shape ``(M, num_steps + 1, 3)``, meters; index 0 is the initial state.
    velocity_trajectory : jax.Array
        Shape ``(M, num_steps + 1, 3)``, m/s; index 0 is the initial state.
    """
    positions0 = jnp.asarray(positions, dtype=jnp.float64)
    velocities0 = jnp.asarray(velocities, dtype=jnp.float64)

    def step(
        carry: tuple[jax.Array, jax.Array], _: None
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        pos, vel = carry
        acc = trap.acceleration(pos)
        pos_new = pos + vel * dt + 0.5 * acc * dt**2
        acc_new = trap.acceleration(pos_new)
        vel_new = vel + 0.5 * (acc + acc_new) * dt
        return (pos_new, vel_new), (pos_new, vel_new)

    _, (pos_hist, vel_hist) = lax.scan(step, (positions0, velocities0), xs=None, length=num_steps)

    # pos_hist/vel_hist: (num_steps, M, 3) -> (M, num_steps, 3), then prepend t=0.
    pos_hist = jnp.moveaxis(pos_hist, 0, 1)
    vel_hist = jnp.moveaxis(vel_hist, 0, 1)
    pos_traj = jnp.concatenate([positions0[:, None, :], pos_hist], axis=1)
    vel_traj = jnp.concatenate([velocities0[:, None, :], vel_hist], axis=1)
    return pos_traj, vel_traj


def propagate_verlet(
    trap: Trap,
    positions: jax.Array,
    velocities: jax.Array,
    dt: float,
    num_steps: int,
) -> jax.Array:
    """Propagate a classical ensemble under `trap` with velocity-Verlet integration.

    Velocity-Verlet is symplectic (bounded energy oscillation, no secular
    drift) and time-reversible. Implemented as a `jax.lax.scan` (jit-safe,
    no Python-level loop). `dt` is given in seconds at this API boundary
    (see ``docs/CONVENTIONS.md`` section 10); no internal
    non-dimensionalization is applied here; see the WP4 builder report
    for why (Compton-unit non-dimensionalization, CONVENTIONS.md E9, is
    specific to the rotor/phase integrator's need to avoid accumulating
    ~1e20 rad of absolute phase, and does not apply to trap-timescale
    classical dynamics).

    Parameters
    ----------
    trap : Trap
        Trap model supplying the acceleration field.
    positions : jax.Array
        Initial positions, shape ``(M, 3)``, meters.
    velocities : jax.Array
        Initial velocities, shape ``(M, 3)``, m/s.
    dt : float
        Integration time step, seconds.
    num_steps : int
        Number of Verlet steps to take.

    Returns
    -------
    jax.Array
        Position trajectory, shape ``(M, num_steps + 1, 3)``, meters,
        dtype float64. Index 0 along axis 1 is the initial state.
    """
    pos_traj, _ = _verlet_trajectory(trap, positions, velocities, dt, num_steps)
    return pos_traj
