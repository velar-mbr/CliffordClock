# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for cliffordclock.integrator.stepper (CONVENTIONS.md E17, E19, E21, E24).

Covers: single-step norm preservation, the E24 rotor<->scalar cross-check
(first-order agreement and second-order divergence scaling), and the WP3
test-contract item 3 convergence-order measurement (V3).

AMBIGUITY (documented, not silent -- see builder report): CONVENTIONS.md
V3 states "Constant Omega rotor: R(tau) = exp(-1/2 Omega tau) R(0) exactly
-- used for integrator convergence-order measurement." A literally
*constant* Omega (unchanging across the whole run) makes the exponential
midpoint stepper's composition mathematically EXACT for any step count
(the matrix-exponential semigroup property exp(A/N)^N = exp(A) holds
regardless of N), so its error floor is pure float64 rounding, not a
clean h^2 power law -- unusable for a slope-fit convergence measurement.
`test_convergence_order_matches_design_order` below instead uses a
generalization that nests literal-constant-Omega as its zero-frequency
limit: Omega(tau) = f(tau) * e_12 for a FIXED bivector direction (e_12,
i.e. B_hat_C) and a time-varying magnitude f(tau) = A sin(omega tau).
Because the direction never changes, Omega(tau1) and Omega(tau2) commute
for all tau1, tau2, so the exact solution is still closed-form,
R(tau) = exp(-1/2 * integral_0^tau f(s) ds * e_12) R(0), while the
midpoint-rule discretization now has genuine, measurable O(dtau^2) global
error (since f is not affine in tau, midpoint quadrature does not
integrate it exactly).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.cl13 import IDX_SCALAR, rotor_norm_sq
from cliffordclock.constants import SPEED_OF_LIGHT
from cliffordclock.integrator.stepper import rotor_step

_IDENTITY = jnp.zeros(16, dtype=jnp.float64).at[IDX_SCALAR].set(1.0)


def test_rotor_step_preserves_norm() -> None:
    """A single step keeps |<R R~>_0 - 1| tiny (E20), for a generic Omega."""
    delta_e = jnp.array([10.0, -5.0, 2.0])
    grad_delta_e = jnp.array([[1e5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1e5]])
    v = jnp.array([1e4, 0.0, 0.0])
    mu = jnp.array([1e-25, 2e-25, -1e-25])
    r_next, _dphase = rotor_step(_IDENTITY, delta_e, grad_delta_e, v, mu, dtau=1.0)
    assert abs(float(rotor_norm_sq(r_next)) - 1.0) < 1e-13


def test_rotor_step_dtype_float64() -> None:
    delta_e = jnp.array([1.0, 0.0, 0.0])
    grad_delta_e = jnp.zeros((3, 3))
    v = jnp.zeros(3)
    mu = jnp.array([1.0, 0.0, 0.0])
    r_next, dphase = rotor_step(_IDENTITY, delta_e, grad_delta_e, v, mu, dtau=1.0)
    assert r_next.dtype == jnp.float64
    assert dphase.scalar.dtype == jnp.float64
    assert dphase.rotor.dtype == jnp.float64


def test_rotor_plane_angle_matches_scalar_phase_for_pure_e12_rotation() -> None:
    """With no boost term (grad_delta_e = 0), the rotor-extracted phase (E24)
    equals the scalar E21 phase increment to near machine precision -- this
    is the E24 "equality at first order" acceptance criterion in its
    cleanest form (zero boost, not just small boost).
    """
    delta_e = jnp.array([500.0, -200.0, 100.0])
    grad_delta_e = jnp.zeros((3, 3))
    v = jnp.zeros(3)
    mu = jnp.array([1e-27, 3e-27, -2e-27])
    dtau = 0.5

    _r_next, dphase = rotor_step(_IDENTITY, delta_e, grad_delta_e, v, mu, dtau)
    assert pytest.approx(float(dphase.scalar), rel=0, abs=1e-14) == float(dphase.rotor)


def test_e24_second_order_divergence_scales_quadratically_with_boost() -> None:
    """WP3 orchestrator requirement: exercise omega_boost with artificially
    exaggerated parameters (realistic v/c * lambda_bar_C * grad(ln P) is
    astronomically small) and confirm the E24 rotor-vs-scalar discrepancy
    scales as O(omega_boost^2) -- the E24 acceptance criterion's permitted
    *second-order* divergence, not a bug.
    """
    mu = jnp.array([1e-20, 0.0, 0.0])
    grad_delta_e = jnp.zeros((3, 3)).at[0, 0].set(1e18)
    # A large, velocity-INDEPENDENT rotation term (Omega_e12 driven by the
    # field pivot, not by v^2/c^2 kinematic time dilation) so the only
    # thing varying with `s` below is the boost term itself -- isolating
    # the O(omega_boost^2) scaling from the (also quadratic-in-v) E21
    # kinematic factor, which would otherwise confound the measurement.
    delta_e = jnp.array([1e7, 0.0, 0.0])
    dtau = 0.05

    scales = [1.0, 2.0, 4.0, 8.0]
    discrepancies = []
    for s in scales:
        v = jnp.array([0.02 * s * SPEED_OF_LIGHT, 0.0, 0.0])
        _r_next, dphase = rotor_step(_IDENTITY, delta_e, grad_delta_e, v, mu, dtau)
        discrepancies.append(abs(float(dphase.rotor) - float(dphase.scalar)))

    assert discrepancies[0] > 1e-13, "boost machinery not actually exercised (discrepancy ~0)"

    ratios = [discrepancies[i + 1] / discrepancies[i] for i in range(len(discrepancies) - 1)]
    # Doubling the boost-driving velocity should ~quadruple the discrepancy
    # (O(omega_boost^2)); allow a generous band since this is a leading-order
    # asymptotic scaling, not an exact quadratic.
    for ratio in ratios:
        assert 3.5 < ratio < 4.6, f"expected ~4x scaling per doubling, got {ratio}"


def test_e24_first_order_agreement_at_realistic_boost_scale() -> None:
    """At the realistic v/c ~ 1e-6 (thermal-atom-scale) regime, even a large
    synthetic gradient produces a rotor<->scalar discrepancy far below any
    physically meaningful floor (1e-18) -- i.e. first-order equality holds
    in practice, per E24.
    """
    mu = jnp.array([1e-20, 0.0, 0.0])
    grad_delta_e = jnp.zeros((3, 3)).at[0, 0].set(1e18)
    delta_e = jnp.zeros(3)
    v = jnp.array([1e-6 * SPEED_OF_LIGHT, 0.0, 0.0])
    dtau = 0.05

    _r_next, dphase = rotor_step(_IDENTITY, delta_e, grad_delta_e, v, mu, dtau)
    discrepancy = abs(float(dphase.rotor) - float(dphase.scalar))
    assert discrepancy < 1e-18


def test_convergence_order_matches_design_order() -> None:
    """WP3 test contract item 3: halving dtau reduces error by the design
    order (exponential midpoint = order 2, E19), fit slope within +/-0.2.

    See module docstring AMBIGUITY note for why this uses a fixed-plane,
    time-varying-magnitude Omega(tau) = A sin(omega tau) * e_12 rather than
    a literally constant Omega.
    """
    amplitude = 1e-3
    ang_freq = 0.7
    t_total = 3.0
    mu = jnp.array([1.0, 0.0, 0.0])
    grad_delta_e = jnp.zeros((3, 3))
    v = jnp.zeros(3)

    from cliffordclock.constants import ELECTRON_MASS

    m_e_c2 = ELECTRON_MASS * SPEED_OF_LIGHT**2

    def rate(tau: jnp.ndarray) -> jnp.ndarray:
        """Target instantaneous rotation_coeff f(tau) = A sin(omega tau)."""
        return amplitude * jnp.sin(ang_freq * tau)

    def exact_phase(tau: jnp.ndarray) -> jnp.ndarray:
        """integral_0^tau A sin(omega s) ds = (A/omega)(1 - cos(omega tau))."""
        return (amplitude / ang_freq) * (1.0 - jnp.cos(ang_freq * tau))

    def run(n_steps: int) -> float:
        dtau = t_total / n_steps

        def body(r: jnp.ndarray, k: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            tau_mid = (k.astype(jnp.float64) + 0.5) * dtau
            target = rate(tau_mid)
            delta_e = jnp.array([target * m_e_c2 / mu[0], 0.0, 0.0])
            r_next, dphase = rotor_step(r, delta_e, grad_delta_e, v, mu, dtau)
            return r_next, dphase.rotor

        _r_final, dphases = jax.lax.scan(body, _IDENTITY, jnp.arange(n_steps))
        return float(jnp.sum(dphases))

    step_counts = [16, 32, 64, 128, 256]
    dtaus = np.array([t_total / n for n in step_counts])
    exact = float(exact_phase(jnp.asarray(t_total)))
    errors = np.array([abs(run(n) - exact) for n in step_counts])

    assert np.all(errors > 0), "errors are exactly zero; test is not measuring anything"
    slope, _intercept = np.polyfit(np.log(dtaus), np.log(errors), 1)
    assert 1.8 < slope < 2.2, f"measured convergence order {slope}, expected ~2"
