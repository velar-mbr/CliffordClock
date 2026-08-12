# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for `cliffordclock.integrator.fastpath.secular_average_shift` (E30).

Includes WP8 test contract item 3, **Tier B(ii) == Tier B(i)**: the
secular average times `T` must match large-step direct integration
(`cliffordclock.integrator.worldline.integrate_worldline` at
`select_dtau`'s dtau) on the same harmonic (V4) case, to a stated
tolerance with a documented bound rationale (below).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.constants import TAU_COMPTON
from cliffordclock.ensemble.traps import HarmonicTrap
from cliffordclock.fields.synthetic import as_field_fn, constant_gradient_field
from cliffordclock.integrator import fastpath
from cliffordclock.integrator.omega import scalar_rate_perturbation
from cliffordclock.integrator.worldline import integrate_worldline


def _e14a_rate_fn(field_fn, mu: jnp.ndarray) -> fastpath.RateFn:  # noqa: ANN001
    """See `tests/test_fastpath_lattice.py`'s identical helper."""

    def rate_fn(pos: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        delta_e, _grad = field_fn(pos)
        return scalar_rate_perturbation(delta_e, v, mu)

    return rate_fn


# Same V4 setup as `tests/test_fastpath_select_dtau.py` (target-regime mu,
# small oscillation amplitude -- see that module's docstring finding for
# why: keeps the rotor-diagnostic path inside exp_bivector's convergence
# radius, irrelevant here since E30 never touches the rotor, but the small
# amplitude also keeps this test's *own* numerics clean).
_OMEGA = 2.0e5
_CENTER = np.array([0.01, -0.02, 0.03])
_GRAD = np.array([[1.0e3, 2.0e3, 0.0], [0.0, -1.0e3, 5.0e2], [3.0e2, 0.0, -2.0e3]])
_MU = 1.0e-33 * np.array([1.0, 2.0, -3.0])
_SCALE = 1.0e-5
_DELTA_R0 = _SCALE * np.array([1.0e-3, -0.7e-3, 1.2e-3])
_V0 = _SCALE * np.array([1.0e-2, -0.8e-2, 0.6e-2])
_R0 = _CENTER + _DELTA_R0

_TRAP = HarmonicTrap(omega_xyz=(_OMEGA, _OMEGA, _OMEGA), center=tuple(_CENTER))
_FIELD_FN = as_field_fn(*constant_gradient_field(jnp.zeros(3), jnp.asarray(_GRAD)))
_RATE_FN = _e14a_rate_fn(_FIELD_FN, jnp.asarray(_MU))
_T_ORBIT_S = 2.0 * np.pi / _OMEGA


def _direct_trajectory(n_steps: int, dtau: float) -> np.ndarray:
    dt_s = dtau * TAU_COMPTON
    t = np.arange(n_steps + 1, dtype=np.float64) * dt_s
    return (
        _CENTER
        + _DELTA_R0 * np.cos(_OMEGA * t)[:, None]
        + (_V0 / _OMEGA) * np.sin(_OMEGA * t)[:, None]
    )


# ---------------------------------------------------------------------------
# Validity-bound / shape tests.
# ---------------------------------------------------------------------------


def test_secular_average_shift_rejects_anisotropic_trap() -> None:
    trap = HarmonicTrap(omega_xyz=(2.0e5, 3.0e5, 2.0e5))
    with pytest.raises(ValueError, match="isotropic"):
        fastpath.secular_average_shift(_RATE_FN, trap, jnp.asarray(_R0), jnp.asarray(_V0), 1.0)


def test_secular_average_shift_rejects_nonpositive_time() -> None:
    with pytest.raises(ValueError, match="t_interrogation_s"):
        fastpath.secular_average_shift(_RATE_FN, _TRAP, jnp.asarray(_R0), jnp.asarray(_V0), 0.0)


def test_secular_average_shift_result_has_no_rotor_state() -> None:
    result = fastpath.secular_average_shift(
        _RATE_FN, _TRAP, jnp.asarray(_R0), jnp.asarray(_V0), 5.0 * _T_ORBIT_S
    )
    assert float(result.norm_error) == 0.0
    assert float(result.max_norm_drift) == 0.0
    assert float(result.phase_rotor) == float(result.phase)
    np.testing.assert_allclose(float(result.t_orbit_s), _T_ORBIT_S, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# WP8 test contract item 3: Tier B(ii) == Tier B(i).
# ---------------------------------------------------------------------------


def test_tier_b_ii_equals_tier_b_i_exact_period_multiple() -> None:
    """`T` an exact integer multiple of `T_orb`: both tiers evaluate the
    *same* underlying midpoint-quadrature/Kahan-sum of `rate_fn` along
    (numerically) the same periodic trajectory -- Tier B(ii) once per
    orbit then scaled by the period count, Tier B(i) directly over the
    whole span -- so for a static field and an exactly periodic orbit
    (E30's own validity bound), the two must agree to within floating-
    point summation-order noise, not any physical approximation. Bound
    rationale: `rtol=1e-9` is ~1e6x looser than the measured
    (implementation-specific) ~1e-16 relative discrepancy, giving ample
    margin for reordering/platform differences while still being a
    meaningful (not vacuous) check.
    """
    n_periods = 5
    points_per_period = 100
    t_total_s = n_periods * _T_ORBIT_S

    secular = fastpath.secular_average_shift(
        _RATE_FN,
        _TRAP,
        jnp.asarray(_R0),
        jnp.asarray(_V0),
        t_total_s,
        points_per_period=points_per_period,
    )

    dtau = fastpath.select_dtau(_TRAP, points_per_period)
    n_steps_direct = n_periods * points_per_period
    traj = _direct_trajectory(n_steps_direct, dtau)
    direct = integrate_worldline(
        _FIELD_FN, jnp.asarray(traj), dtau, jnp.asarray(_MU), renorm_every=1
    )

    np.testing.assert_allclose(float(secular.phase), float(direct.phase), rtol=1e-9, atol=0)
    assert int(secular.n_steps) == points_per_period  # one orbit's worth of sub-steps.
    # Non-vacuous.
    assert abs(float(direct.phase)) > 1e-6


def test_secular_partial_orbit_matches_e30_epsilon_bound() -> None:
    """`T` *not* an exact multiple of `T_orb` (the realistic case): the
    discrepancy between Tier B(ii) and Tier B(i) must not exceed E30's own
    documented remainder bound, ``|ε| <= T̃_orb * max_t|δω̃ - ⟨δω̃⟩_orb|``
    (`SecularResult.epsilon_bound_phase`) -- this is the actual physics
    content of E30's approximation (not just floating-point noise, unlike
    the exact-multiple case above).
    """
    points_per_period = 100
    n_periods = 4.37  # deliberately not an integer.
    t_total_s = n_periods * _T_ORBIT_S

    secular = fastpath.secular_average_shift(
        _RATE_FN,
        _TRAP,
        jnp.asarray(_R0),
        jnp.asarray(_V0),
        t_total_s,
        points_per_period=points_per_period,
    )

    dtau = fastpath.select_dtau(_TRAP, points_per_period)
    n_steps_direct = round(n_periods * points_per_period)
    traj = _direct_trajectory(n_steps_direct, dtau)
    direct = integrate_worldline(
        _FIELD_FN, jnp.asarray(traj), dtau, jnp.asarray(_MU), renorm_every=1
    )

    diff = abs(float(secular.phase) - float(direct.phase))
    eps_bound = float(secular.epsilon_bound_phase)
    assert diff <= eps_bound, (
        f"|secular - direct| = {diff!r} exceeds E30's own remainder bound {eps_bound!r}"
    )
    # The bound should be a meaningful (not wildly loose) estimate: within
    # 2 orders of magnitude of the actual discrepancy, and itself small
    # relative to the total phase.
    assert diff > 0.0
    assert eps_bound < 100.0 * diff
    assert eps_bound < 0.1 * abs(float(direct.phase))


def test_secular_average_shift_ensemble_matches_per_atom_loop() -> None:
    """`secular_average_shift_ensemble` (vmap) matches a Python loop over
    `secular_average_shift`.
    """
    rng = np.random.default_rng(0)
    m = 6
    positions = _CENTER[None, :] + _SCALE * rng.uniform(-1e-3, 1e-3, size=(m, 3))
    velocities = _SCALE * rng.uniform(-1e-2, 1e-2, size=(m, 3))
    t_total_s = 5.0 * _T_ORBIT_S

    batched = fastpath.secular_average_shift_ensemble(
        _RATE_FN, _TRAP, jnp.asarray(positions), jnp.asarray(velocities), t_total_s
    )
    loop_phases = [
        float(
            fastpath.secular_average_shift(
                _RATE_FN, _TRAP, jnp.asarray(positions[i]), jnp.asarray(velocities[i]), t_total_s
            ).phase
        )
        for i in range(m)
    ]

    np.testing.assert_allclose(
        np.asarray(batched.phase), np.asarray(loop_phases), rtol=1e-13, atol=0
    )
    assert batched.phase.shape == (m,)
    assert batched.r_final.shape == (m, 16)
