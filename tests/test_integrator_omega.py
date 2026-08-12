# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for cliffordclock.integrator.omega (CONVENTIONS.md E14a, E16, E18, E21).

Covers the pivot/spin-connection/omega construction in isolation from the
stepper/worldline machinery.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cliffordclock.cl13 import (
    IDX_E01,
    IDX_E02,
    IDX_E03,
    IDX_E1,
    IDX_E12,
    IDX_SCALAR,
)
from cliffordclock.constants import ELECTRON_MASS, LAMBDA_BAR_COMPTON, SPEED_OF_LIGHT
from cliffordclock.integrator.omega import (
    build_omega,
    pivot,
    pivot_perturbation,
    scalar_rate_perturbation,
    spin_connection,
)

_M_E_C2 = ELECTRON_MASS * SPEED_OF_LIGHT**2


def test_pivot_matches_e14a_formula() -> None:
    """P(r) = 1 + delta_E . mu / (m_e c^2) (E14a), directly."""
    delta_e = jnp.array([1.0, 2.0, 3.0])
    mu = jnp.array([4.0, 5.0, 6.0])
    expected = 1.0 + (1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0) / _M_E_C2
    assert pivot(delta_e, mu).dtype == jnp.float64
    # rtol=..., atol=0: a rel-only pytest.approx comparison silently
    # falls back to a default abs of 1e-12, which would mask precision
    # loss for any tiny-magnitude expected value (MAJOR 3 audit).
    np.testing.assert_allclose(float(pivot(delta_e, mu)), expected, rtol=1e-14, atol=0)


def test_pivot_batched() -> None:
    """pivot batches over a leading axis."""
    delta_e = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    mu = jnp.array([1.0, 1.0, 1.0])
    p = pivot(delta_e, mu)
    assert p.shape == (3,)
    # delta_E=0 -> P=1 exactly; rtol/atol=0 explicit (MAJOR 3 audit).
    np.testing.assert_allclose(float(p[2]), 1.0, rtol=1e-14, atol=0)


def test_spin_connection_matches_independent_autodiff() -> None:
    """omega_0k = d_k ln P(r) (E16), cross-checked against an independent
    jax.jacfwd of a hand-built P(r) closure for a constant-gradient field.

    This is a non-circular check: `spin_connection` computes d_k(P) via the
    analytic E14a chain rule on a *pre-supplied* grad_delta_e tensor, while
    this test differentiates a *separate* closure built directly from
    `pivot`, through autodiff, for a field whose gradient is known exactly.
    """
    grad = jnp.array([[1e5, 2e5, 0.0], [0.0, -3e5, 1e5], [4e5, 0.0, -1e5]])
    e0 = jnp.array([10.0, -5.0, 2.0])
    mu = jnp.array([1e-25, -2e-25, 3e-25])

    def delta_e_fn(pos: jnp.ndarray) -> jnp.ndarray:
        return e0 + pos @ grad

    def ln_p_fn(pos: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(pivot(delta_e_fn(pos), mu))

    pos0 = jnp.array([0.1, -0.2, 0.05])
    expected = jax.jacfwd(ln_p_fn)(pos0)  # (3,), d(ln P)/dr_k via autodiff.

    delta_e0 = delta_e_fn(pos0)
    grad_at_pos0 = grad  # constant-gradient field: grad_delta_e is position-independent.
    got = spin_connection(delta_e0, grad_at_pos0, mu)

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-10, atol=0)


def test_scalar_rate_perturbation_matches_e21_at_rest() -> None:
    """delta_omega_tilde = P(r) - 1 when v = 0 (E21 static-node regime)."""
    delta_e = jnp.array([100.0, 0.0, 0.0])
    mu = jnp.array([1e-28, 0.0, 0.0])
    v = jnp.zeros(3)
    # E10 precision discipline (see pivot_perturbation's docstring):
    # `pivot(...) - 1.0` re-introduces exactly the catastrophic-cancellation
    # failure mode this module is built to avoid -- `expected` here is
    # ~1.22e-13, so forming `pivot()`'s `1.0 + x` sum and then subtracting
    # 1.0 back off only recovers ~3-4 significant digits, nowhere near
    # tight enough for a MAJOR-3-audited rtol=1e-14 comparison (this was
    # previously masked by pytest.approx's default abs=1e-12 floor).
    # `pivot_perturbation` is the precision-safe way to get `P - 1` directly.
    expected = pivot_perturbation(delta_e, mu)
    got = scalar_rate_perturbation(delta_e, v, mu)
    np.testing.assert_allclose(float(got), float(expected), rtol=1e-14, atol=0)


def test_scalar_rate_perturbation_includes_kinematic_factor() -> None:
    """With delta_E = 0 (P = 1 exactly), delta_omega_tilde = sqrt(1-v^2/c^2) - 1 (E21/E15)."""
    delta_e = jnp.zeros(3)
    mu = jnp.array([1.0, 0.0, 0.0])
    v = jnp.array([0.1 * SPEED_OF_LIGHT, 0.0, 0.0])
    expected = jnp.sqrt(1.0 - 0.1**2) - 1.0
    got = scalar_rate_perturbation(delta_e, v, mu)
    # MAJOR 3 audit: explicit atol=0 so the intended rtol is what governs.
    np.testing.assert_allclose(float(got), float(expected), rtol=1e-13, atol=0)


def test_scalar_rate_perturbation_kinematic_term_survives_cold_atom_regime() -> None:
    """BLOCKER 1 regression: the kinematic term must not catastrophically
    cancel to 0.0 at realistic cold-atom velocities (E21/E10).

    ``kinematic = sqrt(1 - x) - 1`` (with ``x = v^2/c^2``) computed
    literally underflows to exactly 0.0 once ``x`` drops below the
    float64 relative epsilon (~2.2e-16), i.e. for any ``v/c`` below
    ~1e-8 -- squarely inside the realistic regime (Sr @ 1 uK has
    ``v/c ~ 3.3e-11``). The stable ``-x / (1 + sqrt(1 - x))`` form must
    recover the correct nonzero O(x) Taylor value instead.
    """
    delta_e = jnp.zeros(3)  # P = 1 exactly, isolating the kinematic term.
    mu = jnp.array([1.0, 0.0, 0.0])
    v_over_c = 3.3e-11  # Sr @ 1 uK.
    v = jnp.array([v_over_c * SPEED_OF_LIGHT, 0.0, 0.0])

    x = v_over_c**2
    taylor = -x / 2.0 - x**2 / 8.0

    got = float(scalar_rate_perturbation(delta_e, v, mu))
    assert got != 0.0, "kinematic term collapsed to zero (catastrophic cancellation regression)"
    np.testing.assert_allclose(got, taylor, rtol=1e-10, atol=0)


def test_scalar_rate_perturbation_kinematic_term_matches_taylor_at_1e6_v_over_c() -> None:
    """BLOCKER 1 regression, second point: v/c = 1e-6 (thermal-atom scale),
    matching the -x/2 - x^2/8 Taylor value to 1e-12 relative.
    """
    delta_e = jnp.zeros(3)
    mu = jnp.array([1.0, 0.0, 0.0])
    v_over_c = 1e-6
    v = jnp.array([v_over_c * SPEED_OF_LIGHT, 0.0, 0.0])

    x = v_over_c**2
    taylor = -x / 2.0 - x**2 / 8.0

    got = float(scalar_rate_perturbation(delta_e, v, mu))
    assert got != 0.0
    np.testing.assert_allclose(got, taylor, rtol=1e-12, atol=0)


def test_build_omega_is_bivector_only() -> None:
    """Omega has exactly the 4 documented nonzero components (E18); everything
    else (scalar, vector, e13, e23, trivector, pseudoscalar grades) is zero.
    """
    delta_e = jnp.array([1.0, 2.0, 3.0])
    grad_delta_e = jnp.array([[1e5, 2e5, 0.0], [0.0, -3e5, 1e5], [4e5, 0.0, -1e5]])
    v = jnp.array([1.0, 2.0, 3.0])
    mu = jnp.array([1e-25, -2e-25, 3e-25])

    omega = build_omega(delta_e, grad_delta_e, v, mu)
    assert omega.shape == (16,)

    allowed = {IDX_E12, IDX_E01, IDX_E02, IDX_E03}
    for idx in range(16):
        if idx not in allowed:
            assert float(omega[idx]) == 0.0, f"unexpected nonzero component at index {idx}"
    # And the e_1 (vector) component in particular must be exactly zero,
    # i.e. Omega is not accidentally leaking grade-1 content.
    assert float(omega[IDX_E1]) == 0.0
    assert float(omega[IDX_SCALAR]) == 0.0


def test_build_omega_zero_when_fully_unperturbed() -> None:
    """delta_E = 0, grad = 0, v = 0 => Omega = 0 exactly (no absolute-phase leakage)."""
    delta_e = jnp.zeros(3)
    grad_delta_e = jnp.zeros((3, 3))
    v = jnp.zeros(3)
    mu = jnp.array([1e-25, -2e-25, 3e-25])
    omega = build_omega(delta_e, grad_delta_e, v, mu)
    np.testing.assert_array_equal(np.asarray(omega), np.zeros(16))


def test_boost_term_requires_nonzero_velocity() -> None:
    """A nonzero gradient alone (v=0) gives zero omega_boost (E18: boost scales as v/c)."""
    delta_e = jnp.array([1.0, 0.0, 0.0])
    grad_delta_e = jnp.array([[1e8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mu = jnp.array([1e-20, 0.0, 0.0])

    omega_static = build_omega(delta_e, grad_delta_e, jnp.zeros(3), mu)
    assert float(omega_static[IDX_E01]) == 0.0
    assert float(omega_static[IDX_E02]) == 0.0
    assert float(omega_static[IDX_E03]) == 0.0

    omega_moving = build_omega(delta_e, grad_delta_e, jnp.array([1e3, 0.0, 0.0]), mu)
    assert float(omega_moving[IDX_E01]) != 0.0


def test_boost_coefficient_matches_hand_derivation() -> None:
    """Directly check the omega_boost[e01] coefficient against a hand-computed value.

    omega_0x = (mu . grad_delta_e[x, :]) / (P * m_e c^2); omega_tilde_0x =
    lambda_bar_C * omega_0x; boost_coeff_x = (v_x/c) * omega_tilde_0x; and
    Omega[IDX_E01] = -boost_coeff_x (since e_1 ^ e_0 = -e_01, see build_omega
    docstring).
    """
    delta_e = jnp.zeros(3)  # P = 1 exactly.
    grad_delta_e = jnp.zeros((3, 3)).at[0, 0].set(1e10)
    mu = jnp.array([1e-20, 0.0, 0.0])
    v = jnp.array([0.05 * SPEED_OF_LIGHT, 0.0, 0.0])

    omega_0x = (mu[0] * grad_delta_e[0, 0]) / _M_E_C2
    omega_tilde_0x = LAMBDA_BAR_COMPTON * omega_0x
    boost_coeff_x = (v[0] / SPEED_OF_LIGHT) * omega_tilde_0x
    expected_e01 = -boost_coeff_x

    omega = build_omega(delta_e, grad_delta_e, v, mu)
    # MAJOR 3 audit: explicit atol=0 so the intended rtol is what governs.
    np.testing.assert_allclose(float(omega[IDX_E01]), float(expected_e01), rtol=1e-12, atol=0)


def test_dtype_is_float64() -> None:
    delta_e = jnp.array([1.0, 2.0, 3.0])
    grad_delta_e = jnp.zeros((3, 3))
    v = jnp.zeros(3)
    mu = jnp.array([1.0, 0.0, 0.0])
    assert pivot(delta_e, mu).dtype == jnp.float64
    assert spin_connection(delta_e, grad_delta_e, mu).dtype == jnp.float64
    assert scalar_rate_perturbation(delta_e, v, mu).dtype == jnp.float64
    assert build_omega(delta_e, grad_delta_e, v, mu).dtype == jnp.float64
