# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for `cliffordclock.integrator.fastpath.select_dtau` (E31) and the
WP8 large-dτ̃ accuracy study (test contract item 2, CONVENTIONS.md V4).

The V4 closed form (CONVENTIONS.md section 9) is reimplemented here
independently (plain NumPy, not imported from
`cliffordclock.integrator.fastpath`), matching the same "not a shared
kernel" testing philosophy `tests/reference_impl.py` uses for Case C:

    <P(r(t))-1>_orb = P(center)-1   (exact: a constant-gradient field is
        affine in r, and a full-period average of the sinusoidal orbit
        r(t) = center + dr0*cos(w t) + (v0/w)*sin(w t) is exactly `center`)
    <v^2>_orb = (1/2)*(w^2*|dr0|^2 + |v0|^2)   (exact, SHM virial identity)
    <delta_omega~>_orb = (P(center)-1) - <v^2>_orb/(2 c^2)   (V4: pivot +
        second-order-Doppler time average)
    Delta_Phi_closed = <delta_omega~>_orb * T_tilde

**Finding (WP8 builder report):** at E31's large auto-selected dtau, a
*single* `exp_bivector` call is accurate (per-step norm error ~1e-13-1e-14,
verified directly below), but the DEFAULT `renorm_every=1000` cadence
(tuned in WP3 for Compton-scale `dtau~1`, where the per-step floor is far
smaller) lets that per-step floor accumulate past `1e-12` over hundreds of
large-dtau steps. This is a `renorm_every` *cadence* question, not a
correctness bug (E20/E31's own claim -- norm preservation is step-size
independent -- is about the *exact* exponential map, not this fixed-order
Taylor implementation's error floor at any given step count): tightening
`renorm_every` (exposed exactly for this purpose, see
`cliffordclock.integrator.worldline.DEFAULT_RENORM_EVERY`'s docstring)
restores `<1e-12` easily. `docs/timescales.md` documents this. The primary
scalar phase (E21/E22, what `mean_fractional_shift`/the CLI actually
report) is entirely unaffected by this either way -- it is accumulated
directly from `omega[..., IDX_E12] * dtau`, never through `exp_bivector`
at all (see `cliffordclock.integrator.worldline`'s module docstring).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.constants import ELECTRON_MASS, SPEED_OF_LIGHT, TAU_COMPTON
from cliffordclock.ensemble.traps import HarmonicTrap
from cliffordclock.fields.synthetic import as_field_fn, constant_gradient_field
from cliffordclock.integrator import fastpath
from cliffordclock.integrator.worldline import integrate_worldline

_M_E_C2 = ELECTRON_MASS * SPEED_OF_LIGHT**2

# ---------------------------------------------------------------------------
# select_dtau (E31) unit tests.
# ---------------------------------------------------------------------------


def test_select_dtau_uses_slowest_axis() -> None:
    """`T_orb = 2*pi/min(omega_xyz)` -- the longest (most conservative) period."""
    trap = HarmonicTrap(omega_xyz=(2.0e5, 3.0e5, 1.0e5))
    dtau = fastpath.select_dtau(trap, points_per_period=100)
    expected_period_s = 2.0 * np.pi / 1.0e5
    expected_dtau = (expected_period_s / TAU_COMPTON) / 100
    np.testing.assert_allclose(dtau, expected_dtau, rtol=1e-14, atol=0)


def test_select_dtau_default_points_per_period_is_100() -> None:
    trap = HarmonicTrap(omega_xyz=(2.0e5, 2.0e5, 2.0e5))
    default = fastpath.select_dtau(trap)
    explicit = fastpath.select_dtau(trap, points_per_period=100)
    assert default == explicit
    assert fastpath.DEFAULT_POINTS_PER_PERIOD == 100


def test_select_dtau_rejects_invalid_points_per_period() -> None:
    trap = HarmonicTrap(omega_xyz=(2.0e5, 2.0e5, 2.0e5))
    with pytest.raises(ValueError, match="points_per_period"):
        fastpath.select_dtau(trap, points_per_period=0)


def test_select_dtau_rejects_nonpositive_omega() -> None:
    trap = HarmonicTrap(omega_xyz=(2.0e5, 0.0, 2.0e5))
    with pytest.raises(ValueError, match="omega_xyz"):
        fastpath.select_dtau(trap)


def test_select_dtau_scales_inversely_with_points_per_period() -> None:
    trap = HarmonicTrap(omega_xyz=(2.0e5, 2.0e5, 2.0e5))
    dtau_100 = fastpath.select_dtau(trap, 100)
    dtau_200 = fastpath.select_dtau(trap, 200)
    np.testing.assert_allclose(dtau_200, dtau_100 / 2.0, rtol=1e-14, atol=0)


# ---------------------------------------------------------------------------
# V4 closed-form setup, shared by the accuracy-study tests below.
# ---------------------------------------------------------------------------

_OMEGA = 2.0e5  # rad/s, isotropic trap.
_CENTER = np.array([0.01, -0.02, 0.03])
_GRAD = np.array([[1.0e3, 2.0e3, 0.0], [0.0, -1.0e3, 5.0e2], [3.0e2, 0.0, -2.0e3]])
_E0 = np.zeros(3)
# Sized so P(center) - 1 ~ 3.8e-18 -- the target clock-shift regime this
# project cares about (CONVENTIONS.md section 3), and small enough to keep
# the per-step rotor generator well inside exp_bivector's (fixed 12-term,
# 10-halving) Taylor convergence radius at E31's large dtau (see module
# docstring finding; a much larger mu, e.g. WP3/WP6's ~1e-25 convention,
# drives the per-step angle into the thousands of radians and makes
# exp_bivector's scaled-Taylor-series implementation return NaN --
# unrelated to phase accuracy, since the scalar E21/E22 pipeline never
# calls exp_bivector, but it does make the *rotor* diagnostics (E20/E24)
# meaningless, so this test avoids that regime for the same physical
# reason a real clock analysis would: target-regime shifts, not an
# arbitrarily large synthetic one).
_MU = 1.0e-33 * np.array([1.0, 2.0, -3.0])
# Oscillation amplitude/velocity small relative to the (dominant, and
# discretization-error-free by periodicity) center-offset pivot term, so
# the *closed-form* comparison isolates the E19 midpoint scheme's genuine
# O(dtau^2) discretization error rather than being swamped by the
# oscillating contribution's own (much smaller, but non-vanishing) share.
_SCALE = 1.0e-5
_DELTA_R0 = _SCALE * np.array([1.0e-3, -0.7e-3, 1.2e-3])
_V0 = _SCALE * np.array([1.0e-2, -0.8e-2, 0.6e-2])

_TRAP = HarmonicTrap(omega_xyz=(_OMEGA, _OMEGA, _OMEGA), center=tuple(_CENTER))
_FIELD_FN = as_field_fn(*constant_gradient_field(jnp.asarray(_E0), jnp.asarray(_GRAD)))

_P_MINUS_1_CENTER = float(np.dot(_E0 + _CENTER @ _GRAD, _MU) / _M_E_C2)
_MEAN_V2 = 0.5 * (_OMEGA**2 * np.dot(_DELTA_R0, _DELTA_R0) + np.dot(_V0, _V0))
_MEAN_RATE = _P_MINUS_1_CENTER - _MEAN_V2 / (2.0 * SPEED_OF_LIGHT**2)  # V4 closed form.

#: Tightened renorm cadence for large-dtau runs (see module docstring
#: finding); does not affect the scalar phase, only the rotor-norm
#: diagnostics (E20).
_LARGE_DTAU_RENORM_EVERY = 1


def _sinusoidal_trajectory(n_steps: int, dtau: float) -> np.ndarray:
    """The exact SHM orbit (V4), sampled at `n_steps + 1` points spaced `dtau` apart."""
    dt_s = dtau * TAU_COMPTON
    t = np.arange(n_steps + 1, dtype=np.float64) * dt_s
    return (
        _CENTER
        + _DELTA_R0 * np.cos(_OMEGA * t)[:, None]
        + (_V0 / _OMEGA) * np.sin(_OMEGA * t)[:, None]
    )


def _closed_form_phase(n_steps: int, dtau: float) -> float:
    t_tilde = n_steps * dtau
    return _MEAN_RATE * t_tilde


# ---------------------------------------------------------------------------
# WP8 test contract item 2: large-dtau accuracy vs the V4 closed form.
# ---------------------------------------------------------------------------


def test_v4_closed_form_accuracy_at_select_dtau_default() -> None:
    """`select_dtau`'s default (100 pts/period) over 5 trap periods: phase
    matches the V4 closed form to rtol <= 1e-8, and (with the tightened
    `renorm_every` from this module's docstring finding) rotor-norm drift
    stays < 1e-12 -- both halves of the WP8 test contract's item 2.
    """
    points_per_period = 100
    n_periods = 5
    dtau = fastpath.select_dtau(_TRAP, points_per_period)
    n_steps = n_periods * points_per_period
    assert n_periods >= 3

    traj = _sinusoidal_trajectory(n_steps, dtau)
    result = integrate_worldline(
        _FIELD_FN,
        jnp.asarray(traj),
        dtau,
        jnp.asarray(_MU),
        renorm_every=_LARGE_DTAU_RENORM_EVERY,
    )

    phase_closed = _closed_form_phase(n_steps, dtau)
    rel_err = abs(float(result.phase) - phase_closed) / abs(phase_closed)
    assert rel_err <= 1e-8, f"phase rel_err {rel_err!r} exceeds the 1e-8 WP8 bound"
    assert float(result.max_norm_drift) < 1e-12, (
        f"rotor-norm drift {float(result.max_norm_drift)!r} exceeds 1e-12 (E20) even with "
        f"renorm_every={_LARGE_DTAU_RENORM_EVERY}"
    )
    # Non-vacuous: the closed-form phase itself is not accidentally ~0.
    assert abs(phase_closed) > 1e-3


def test_v4_error_scales_order_two_with_dtau() -> None:
    """WP8 test contract item 2: confirm the expected order-2 scaling
    region (E19's design order) as `dtau` shrinks (`points_per_period`
    grows), fixing the physical span at exactly 3 trap periods.
    """
    points_per_period_values = [25, 50, 100, 200, 400]
    n_periods = 3
    dtaus = []
    errors = []
    for ppp in points_per_period_values:
        dtau = fastpath.select_dtau(_TRAP, ppp)
        n_steps = n_periods * ppp
        traj = _sinusoidal_trajectory(n_steps, dtau)
        result = integrate_worldline(
            _FIELD_FN,
            jnp.asarray(traj),
            dtau,
            jnp.asarray(_MU),
            renorm_every=_LARGE_DTAU_RENORM_EVERY,
        )
        phase_closed = _closed_form_phase(n_steps, dtau)
        dtaus.append(dtau)
        errors.append(abs(float(result.phase) - phase_closed))

    errors_arr = np.array(errors)
    assert np.all(errors_arr > 0), "errors are exactly zero; test is not measuring anything"
    slope, _intercept = np.polyfit(np.log(dtaus), np.log(errors_arr), 1)
    assert 1.8 < slope < 2.2, f"measured convergence order {slope}, expected ~2 (E19)"
