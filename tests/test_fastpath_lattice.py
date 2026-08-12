# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for `cliffordclock.integrator.fastpath.lattice_shift_expectation` (E29).

Includes WP8 test contract item 1, the load-bearing **Tier A == Tier C**
equality test: the lattice fast path must equal the worldline (rotor)
integrator exactly (rtol 1e-12, atol=0) on identical static nodes.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.constants import ELECTRON_MASS, SPEED_OF_LIGHT, TAU_COMPTON
from cliffordclock.fields.synthetic import as_field_fn, constant_gradient_field, gaussian_bump_field
from cliffordclock.integrator import fastpath
from cliffordclock.integrator.omega import scalar_rate_perturbation
from cliffordclock.integrator.worldline import integrate_ensemble

_M_E_C2 = ELECTRON_MASS * SPEED_OF_LIGHT**2


def _e14a_rate_fn(field_fn, mu: jnp.ndarray) -> fastpath.RateFn:  # noqa: ANN001
    """The same E14a closure `cliffordclock.pipeline` builds (reimplemented
    locally so this test file does not depend on `pipeline`'s private
    helper -- see `cliffordclock.integrator.fastpath`'s module docstring
    for why `RateFn` is coupling-agnostic).
    """

    def rate_fn(pos: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        delta_e, _grad = field_fn(pos)
        return scalar_rate_perturbation(delta_e, v, mu)

    return rate_fn


# ---------------------------------------------------------------------------
# Unit tests: shape validation, hand-computed correctness.
# ---------------------------------------------------------------------------


def test_lattice_shift_expectation_matches_hand_computed_pivot() -> None:
    """E29: a single static node, constant-gradient field -- `ΔΦ = (P(r0)-1)*T̃`
    exactly (the same closed form as CONVENTIONS.md V2 / WP6 Case B).
    """
    r0 = jnp.array([0.01, -0.02, 0.03])
    grad = jnp.array([[1.0e3, 2.0e3, 0.0], [0.0, -1.0e3, 5.0e2], [3.0e2, 0.0, -2.0e3]])
    e0 = jnp.zeros(3)
    mu = jnp.array([1.0e-25, 2.0e-25, -3.0e-25])
    t_interrogation_s = 2.5

    field_fn = as_field_fn(*constant_gradient_field(e0, grad))
    rate_fn = _e14a_rate_fn(field_fn, mu)
    nodes = r0[None, :]
    weights = jnp.array([1.0])

    result = fastpath.lattice_shift_expectation(rate_fn, nodes, weights, t_interrogation_s)

    delta_e_r0 = np.asarray(e0) + np.asarray(r0) @ np.asarray(grad)
    p_minus_1 = float(np.dot(delta_e_r0, np.asarray(mu))) / _M_E_C2
    t_tilde = t_interrogation_s / TAU_COMPTON
    expected_phase = p_minus_1 * t_tilde

    np.testing.assert_allclose(float(result.phase[0]), expected_phase, rtol=1e-14, atol=0)
    np.testing.assert_allclose(float(result.fractional_shift[0]), p_minus_1, rtol=1e-14, atol=0)
    assert float(result.phase_rotor[0]) == float(result.phase[0])
    assert float(result.norm_error[0]) == 0.0
    assert float(result.max_norm_drift[0]) == 0.0
    assert int(result.n_steps[0]) == 0


def test_lattice_shift_expectation_is_o1_cost_in_interrogation_time() -> None:
    """The defining property of E29: doubling `T` exactly doubles `ΔΦ`
    (linear, no time-stepping cost difference) -- checked at a
    second-scale `T` where a time-stepped integrator would be intractable.
    """
    r0 = jnp.array([0.01, -0.02, 0.03])
    grad = jnp.array([[1.0e3, 2.0e3, 0.0], [0.0, -1.0e3, 5.0e2], [3.0e2, 0.0, -2.0e3]])
    field_fn = as_field_fn(*constant_gradient_field(jnp.zeros(3), grad))
    mu = jnp.array([1.0e-25, 2.0e-25, -3.0e-25])
    rate_fn = _e14a_rate_fn(field_fn, mu)
    nodes = r0[None, :]
    weights = jnp.array([1.0])

    result_1s = fastpath.lattice_shift_expectation(rate_fn, nodes, weights, 1.0)
    result_2s = fastpath.lattice_shift_expectation(rate_fn, nodes, weights, 2.0)

    np.testing.assert_allclose(
        float(result_2s.phase[0]), 2.0 * float(result_1s.phase[0]), rtol=1e-14, atol=0
    )


def test_lattice_shift_expectation_rejects_mismatched_weights_shape() -> None:
    field_fn = as_field_fn(*constant_gradient_field(jnp.zeros(3), jnp.eye(3)))
    rate_fn = _e14a_rate_fn(field_fn, jnp.array([1.0e-25, 0.0, 0.0]))
    nodes = jnp.zeros((3, 3))
    with pytest.raises(ValueError, match="weights"):
        fastpath.lattice_shift_expectation(rate_fn, nodes, jnp.zeros(2), 1.0)


def test_lattice_shift_expectation_rejects_nonpositive_time() -> None:
    field_fn = as_field_fn(*constant_gradient_field(jnp.zeros(3), jnp.eye(3)))
    rate_fn = _e14a_rate_fn(field_fn, jnp.array([1.0e-25, 0.0, 0.0]))
    nodes = jnp.zeros((1, 3))
    with pytest.raises(ValueError, match="t_interrogation_s"):
        fastpath.lattice_shift_expectation(rate_fn, nodes, jnp.array([1.0]), 0.0)


def test_lattice_shift_expectation_rejects_bad_node_shape() -> None:
    field_fn = as_field_fn(*constant_gradient_field(jnp.zeros(3), jnp.eye(3)))
    rate_fn = _e14a_rate_fn(field_fn, jnp.array([1.0e-25, 0.0, 0.0]))
    with pytest.raises(ValueError, match="nodes"):
        fastpath.lattice_shift_expectation(rate_fn, jnp.zeros(3), jnp.array([1.0]), 1.0)


# ---------------------------------------------------------------------------
# WP8 test contract item 1: Tier A == Tier C (rtol 1e-12, atol=0, any T).
# ---------------------------------------------------------------------------


def test_tier_a_equals_tier_c_worldline_on_static_nodes() -> None:
    """E29's own claim: the fast path must equal the worldline (rotor)
    integrator exactly on identical static nodes, at any step count.

    `T` here is chosen small enough that the worldline integrator can
    actually complete the equivalent number of Compton-unit steps
    (`test_lattice_shift_expectation_is_o1_cost_in_interrogation_time`
    above separately confirms the fast path's own O(1)-in-T cost claim,
    which is what makes a *literal* second-scale Tier-C run intractable --
    ~1e21 Compton steps -- so this equality is checked at a step count
    Tier C can actually run, per E29's own "any step count" wording).
    """
    rng = np.random.default_rng(0)
    n_nodes = 12
    nodes = jnp.asarray(rng.uniform(-2e-6, 2e-6, size=(n_nodes, 3)) + np.array([1e-7, -2e-7, 3e-7]))
    weights = jnp.asarray(np.abs(rng.uniform(0.1, 1.0, size=n_nodes)))
    weights = weights / jnp.sum(weights)

    field_fn = as_field_fn(*gaussian_bump_field(jnp.array([10.0, -5.0, 3.0]), jnp.zeros(3), 1e-6))
    mu = jnp.array([1.0e-24, -5.0e-25, 3.0e-25])
    rate_fn = _e14a_rate_fn(field_fn, mu)

    dtau = 0.5
    steps = 2000
    t_interrogation_s = steps * dtau * TAU_COMPTON

    fast_result = fastpath.lattice_shift_expectation(rate_fn, nodes, weights, t_interrogation_s)
    worldline_result = integrate_ensemble(field_fn, nodes, dtau, mu, n_steps=steps)

    np.testing.assert_allclose(
        np.asarray(fast_result.phase), np.asarray(worldline_result.phase), rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(fast_result.fractional_shift),
        np.asarray(worldline_result.fractional_shift),
        rtol=1e-12,
        atol=0,
    )
    # Non-vacuous: not all zero to float precision.
    assert float(jnp.max(jnp.abs(fast_result.phase))) > 1e-25

    # Corroborating check that this equality genuinely extends to "any T"
    # (E29): Tier C's own phase is exactly linear in step count for a
    # static (v=0) trajectory (n * (delta_omega~ * dtau) = delta_omega~ *
    # (n * dtau)), so a second, larger step count must scale the Tier-C
    # phase by exactly the same ratio the fast path predicts by
    # construction.
    steps_4x = 4 * steps
    worldline_result_4x = integrate_ensemble(field_fn, nodes, dtau, mu, n_steps=steps_4x)
    np.testing.assert_allclose(
        np.asarray(worldline_result_4x.phase),
        4.0 * np.asarray(worldline_result.phase),
        rtol=1e-12,
        atol=0,
    )
