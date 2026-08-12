# SPDX-License-Identifier: AGPL-3.0-or-later
"""Large-angle hardening tests for `exp_bivector` (E6 + 2*pi range reduction).

A WP8 review measured the original fixed-order scaled-Taylor kernel silently
degrading for large compact (rotation-like) generator angles: rotor-norm
error ``1.9e-7`` at ``theta ~ 1e3``, finite garbage (``~1e112``) at ``5e3``,
NaN from ``1e4``. These tests pin the hardened behavior (exact invariant-
split range reduction, see ``_reduce_compact_angle`` in
``cliffordclock.cl13.ops``) in exactly those formerly-silent regimes, plus
the gradient/jit/vmap safety properties the reduction must not break. The
small-angle regime (``theta <= pi``, bitwise-unchanged path) stays covered
by the WP1 oracle tests in ``test_cl13_oracle.py``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.cl13 import basis, ops


def _bivector_from_components(components: dict[int, float]) -> jnp.ndarray:
    value = np.zeros(16, dtype=np.float64)
    for idx, val in components.items():
        value[idx] = val
    return jnp.asarray(value)


# --- Formerly-silent regimes: large finite garbage (1e3, 5e3) and NaN onset
# (1e4). 500 pins the last formerly-good magnitude; 1e6 pins the documented
# far end of the accuracy contract. Angles land away from cos/sin zeros, so
# rtol with atol=0 is meaningful on every nonzero component. ---


@pytest.mark.parametrize("theta", [500.0, 1000.0, 5000.0, 10000.0, -5000.0])
def test_exp_bivector_large_spacelike_angle_matches_closed_form(theta: float) -> None:
    """B = theta e_12: exp(B) = cos(theta) + sin(theta) e_12 for any theta."""
    bivector = _bivector_from_components({basis.IDX_E12: theta})
    result = np.asarray(ops.exp_bivector(bivector))

    nonzero = np.array([result[basis.IDX_SCALAR], result[basis.IDX_E12]])
    expected = np.array([math.cos(theta), math.sin(theta)])
    np.testing.assert_allclose(nonzero, expected, rtol=1e-11, atol=0)

    # Components outside the {1, e_12} subalgebra never receive a nonzero
    # einsum contribution and must be exactly zero, not merely small.
    outside = np.delete(result, [basis.IDX_SCALAR, basis.IDX_E12])
    np.testing.assert_array_equal(outside, np.zeros_like(outside))

    norm_sq = float(ops.rotor_norm_sq(jnp.asarray(result)))
    assert abs(norm_sq - 1.0) < 1e-12


def test_exp_bivector_far_contract_boundary_1e6() -> None:
    """theta = 1e6: documented contract bound (error ~ theta * 1e-16 < 1e-9)."""
    theta = 1.0e6
    bivector = _bivector_from_components({basis.IDX_E12: theta})
    result = np.asarray(ops.exp_bivector(bivector))

    nonzero = np.array([result[basis.IDX_SCALAR], result[basis.IDX_E12]])
    expected = np.array([math.cos(theta), math.sin(theta)])
    np.testing.assert_allclose(nonzero, expected, rtol=1e-9, atol=0)

    norm_sq = float(ops.rotor_norm_sq(jnp.asarray(result)))
    assert abs(norm_sq - 1.0) < 1e-9


def test_exp_bivector_large_commuting_boost_plus_rotation() -> None:
    """B = a e_01 + t e_23 (disjoint planes commute): exact product closed form.

    Only the compact e_23 angle must be reduced; the boost factor must pass
    through untouched. exp(B) = (cosh a + sinh a e_01)(cos t + sin t e_23),
    whose four nonzero components (e_01 e_23 = e_0123) are all O(1) or
    larger at these parameters, so rtol with atol=0 is meaningful.
    """
    a, t = 2.0, 5000.0
    bivector = _bivector_from_components({basis.IDX_E01: a, basis.IDX_E23: t})
    result = np.asarray(ops.exp_bivector(bivector))

    indices = [basis.IDX_SCALAR, basis.IDX_E01, basis.IDX_E23, basis.IDX_E0123]
    expected = np.array(
        [
            math.cosh(a) * math.cos(t),
            math.sinh(a) * math.cos(t),
            math.cosh(a) * math.sin(t),
            math.sinh(a) * math.sin(t),
        ]
    )
    np.testing.assert_allclose(result[indices], expected, rtol=1e-11, atol=0)

    outside = np.delete(result, indices)
    np.testing.assert_array_equal(outside, np.zeros_like(outside))


def test_exp_bivector_periodicity_in_compact_angle() -> None:
    """exp((theta + 2 pi k) e_12) == exp(theta e_12) for large k (reduction is exact)."""
    theta = 0.37
    # k >= 100 only: the k=1 case (theta ~ 6.65 rad) already passed on the
    # pre-reduction kernel, so it carries no regression-protection weight.
    for k in [100, 1000, 10000]:
        shifted = _bivector_from_components({basis.IDX_E12: theta + 2.0 * math.pi * k})
        base = _bivector_from_components({basis.IDX_E12: theta})
        np.testing.assert_allclose(
            np.asarray(ops.exp_bivector(shifted)),
            np.asarray(ops.exp_bivector(base)),
            rtol=1e-11,
            atol=1e-11,
        )


def test_exp_bivector_large_null_bivector_stays_exact() -> None:
    """B null (B^2 = 0, e.g. t(e_01 + e_12)): exp(B) = 1 + B; reduction stays inactive."""
    t = 5000.0
    bivector = _bivector_from_components({basis.IDX_E01: t, basis.IDX_E12: t})
    b_sq = np.asarray(ops.geometric_product(bivector, bivector))
    np.testing.assert_array_equal(b_sq, np.zeros_like(b_sq))

    result = np.asarray(ops.exp_bivector(bivector))
    expected = np.zeros(16, dtype=np.float64)
    expected[basis.IDX_SCALAR] = 1.0
    expected[basis.IDX_E01] = t
    expected[basis.IDX_E12] = t
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


def test_exp_bivector_large_pure_boost_is_untouched_by_reduction() -> None:
    """B = a e_01 (non-compact): no reduction; exp = cosh a + sinh a e_01."""
    a = 300.0
    bivector = _bivector_from_components({basis.IDX_E01: a})
    result = np.asarray(ops.exp_bivector(bivector))
    nonzero = np.array([result[basis.IDX_SCALAR], result[basis.IDX_E01]])
    expected = np.array([math.cosh(a), math.sinh(a)])
    np.testing.assert_allclose(nonzero, expected, rtol=1e-11, atol=0)


# --- Safety properties the reduction must not break ---


def _scalar_of_exp(bivector: jnp.ndarray) -> jnp.ndarray:
    return ops.exp_bivector(bivector)[..., basis.IDX_SCALAR]


@pytest.mark.parametrize(
    "components",
    [
        {},  # zero bivector: q = 0 hits both sqrt clamps
        {basis.IDX_E01: 5.0},  # pure boost: theta^2 = 0 hits the second clamp
        {basis.IDX_E01: 1.0, basis.IDX_E12: 1.0},  # null: q = 0
        {basis.IDX_E12: 1000.0},  # active reduction (k > 0)
        {basis.IDX_E01: 2.0, basis.IDX_E23: 700.0},  # mixed, active reduction
        {basis.IDX_E12: 46000.0},  # WP8-review NaN magnitude: old kernel's grad was all-NaN here
    ],
    ids=["zero", "pure-boost", "null", "large-rotation", "mixed-large", "wp8-nan-regime"],
)
def test_exp_bivector_gradients_are_finite(components: dict[int, float]) -> None:
    """No 0*inf NaN gradients at the reduction's clamped singular points."""
    bivector = _bivector_from_components(components)
    grad = np.asarray(jax.grad(_scalar_of_exp)(bivector))
    assert np.all(np.isfinite(grad))


@pytest.mark.parametrize("theta", [1000.0, 1.0e4, 1.0e5])
def test_exp_bivector_gradient_matches_analytic_derivative_in_reduced_regime(
    theta: float,
) -> None:
    """d/dtheta <exp(theta e_12)>_0 = -sin(theta), checked where k > 0.

    The theta >= 1e4 cases pin the magnitude where the pre-reduction kernel's
    backward pass returned an all-NaN gradient (WP8-review regime): they
    regression-test the gradient-path repair, not just forward accuracy.
    Tolerance scales with the theta*1e-16 error-growth model documented in
    exp_bivector's accuracy contract.
    """
    bivector = _bivector_from_components({basis.IDX_E12: theta})
    grad = float(np.asarray(jax.grad(_scalar_of_exp)(bivector))[basis.IDX_E12])
    np.testing.assert_allclose(grad, -math.sin(theta), rtol=max(1e-9, theta * 1e-12), atol=0)


def test_exp_bivector_jit_vmap_batch_spanning_all_regimes() -> None:
    """One jitted vmap batch mixing every regime matches per-row evaluation."""
    rows = [
        {},
        {basis.IDX_E01: 2.0},
        {basis.IDX_E12: 5000.0},
        {basis.IDX_E01: 3.0, basis.IDX_E23: 700.0},
    ]
    batch = jnp.stack([_bivector_from_components(c) for c in rows])
    batched = np.asarray(jax.jit(jax.vmap(ops.exp_bivector))(batch))
    assert np.all(np.isfinite(batched))
    assert batched.dtype == np.float64
    for i, components in enumerate(rows):
        single = np.asarray(ops.exp_bivector(_bivector_from_components(components)))
        np.testing.assert_allclose(batched[i], single, rtol=1e-12, atol=1e-12)
