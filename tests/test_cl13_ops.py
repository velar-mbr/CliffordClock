# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cl(1,3) operation property tests (WP1 test contract items 2, 3, 4, 5, 8, 9).

Property-based tests use `hypothesis` with bounded float64 strategies and a
fixed (derandomized) seed profile, per WP1's test contract. Oracle
cross-checks against the `clifford` package live in `test_cl13_oracle.py`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from cliffordclock.cl13 import basis, ops

# Fixed seed profile: `derandomize=True` makes hypothesis pick a seed derived
# deterministically from the test itself, so runs are reproducible without
# hardcoding a magic number (WP1: "a fixed seed profile").
_PROPERTY_SETTINGS = settings(
    max_examples=500,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

# Bounded float strategy: values kept modest so cubic products (three
# geometric products chained, as in the associativity test) stay well within
# float64's dynamic range with headroom for the 1e-12 relative tolerance.
_FLOAT = st.floats(
    min_value=-4.0,
    max_value=4.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)
_MULTIVECTOR = arrays(dtype=np.float64, shape=(16,), elements=_FLOAT)
_SCALAR = st.floats(
    min_value=-4.0,
    max_value=4.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)

_BATCH_N = 5
_BATCHED_MULTIVECTOR = arrays(dtype=np.float64, shape=(_BATCH_N, 16), elements=_FLOAT)


def _assert_allclose_1e12(actual: jnp.ndarray, expected: jnp.ndarray) -> None:
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=1e-12)


# --- Test contract item 2: associativity ---


@_PROPERTY_SETTINGS
@given(a=_MULTIVECTOR, b=_MULTIVECTOR, c=_MULTIVECTOR)
def test_geometric_product_associative(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    """(AB)C == A(BC) to 1e-12 relative (E3)."""
    lhs = ops.geometric_product(ops.geometric_product(a, b), c)
    rhs = ops.geometric_product(a, ops.geometric_product(b, c))
    _assert_allclose_1e12(lhs, rhs)


# --- Test contract item 3: distributivity & linearity ---


@_PROPERTY_SETTINGS
@given(a=_MULTIVECTOR, b=_MULTIVECTOR, c=_MULTIVECTOR)
def test_geometric_product_distributes_over_addition(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> None:
    """(A + B)C == AC + BC and A(B + C) == AB + AC (E3, bilinearity)."""
    left_distribute = ops.geometric_product(jnp.asarray(a) + jnp.asarray(b), c)
    left_expected = ops.geometric_product(a, c) + ops.geometric_product(b, c)
    _assert_allclose_1e12(left_distribute, left_expected)

    right_distribute = ops.geometric_product(a, jnp.asarray(b) + jnp.asarray(c))
    right_expected = ops.geometric_product(a, b) + ops.geometric_product(a, c)
    _assert_allclose_1e12(right_distribute, right_expected)


@_PROPERTY_SETTINGS
@given(a=_MULTIVECTOR, b=_MULTIVECTOR, scalar=_SCALAR)
def test_geometric_product_linear_in_scalar(a: np.ndarray, b: np.ndarray, scalar: float) -> None:
    """(alpha A)B == alpha (AB) == A(alpha B) (E3, bilinearity)."""
    scaled_left = ops.geometric_product(scalar * jnp.asarray(a), b)
    scaled_right = ops.geometric_product(a, scalar * jnp.asarray(b))
    scaled_product = scalar * ops.geometric_product(a, b)
    _assert_allclose_1e12(scaled_left, scaled_product)
    _assert_allclose_1e12(scaled_right, scaled_product)


# --- Test contract item 4: reverse ---


@_PROPERTY_SETTINGS
@given(a=_MULTIVECTOR, b=_MULTIVECTOR)
def test_reverse_of_product_reverses_order(a: np.ndarray, b: np.ndarray) -> None:
    """reverse(AB) == reverse(B) reverse(A) (E4)."""
    lhs = ops.reverse(ops.geometric_product(a, b))
    rhs = ops.geometric_product(ops.reverse(b), ops.reverse(a))
    _assert_allclose_1e12(lhs, rhs)


@_PROPERTY_SETTINGS
@given(a=_MULTIVECTOR)
def test_reverse_is_an_involution(a: np.ndarray) -> None:
    """reverse(reverse(A)) == A exactly (E4: sign-flip matrix squares to identity)."""
    twice = ops.reverse(ops.reverse(a))
    np.testing.assert_array_equal(np.asarray(twice), a)


# --- Test contract item 5: grade completeness ---


@_PROPERTY_SETTINGS
@given(a=_MULTIVECTOR)
def test_grade_projections_sum_to_input(a: np.ndarray) -> None:
    """Summing the five grade projections reconstructs the input exactly (E2)."""
    projections = ops.grade_project(a)
    assert len(projections) == 5
    total = sum(projections)
    np.testing.assert_array_equal(np.asarray(total), a)


@_PROPERTY_SETTINGS
@given(a=_MULTIVECTOR)
def test_grade_projections_have_disjoint_support(a: np.ndarray) -> None:
    """Each grade projection is nonzero only within its own 16-component slice."""
    projections = ops.grade_project(a)
    for projection, sl in zip(projections, basis.GRADE_SLICES, strict=True):
        outside = np.delete(np.asarray(projection), np.arange(16)[sl])
        np.testing.assert_array_equal(outside, np.zeros_like(outside))
        expected_inside = np.asarray(a)[sl]
        np.testing.assert_array_equal(np.asarray(projection)[sl], expected_inside)


# --- Test contract item 8: batching ---


@_PROPERTY_SETTINGS
@given(a=_BATCHED_MULTIVECTOR, b=_BATCHED_MULTIVECTOR)
def test_geometric_product_batches_match_python_loop(a: np.ndarray, b: np.ndarray) -> None:
    batched = ops.geometric_product(a, b)
    looped = jnp.stack([ops.geometric_product(a[i], b[i]) for i in range(_BATCH_N)])
    _assert_allclose_1e12(batched, looped)


@_PROPERTY_SETTINGS
@given(a=_BATCHED_MULTIVECTOR)
def test_reverse_batches_match_python_loop(a: np.ndarray) -> None:
    batched = ops.reverse(a)
    looped = jnp.stack([ops.reverse(a[i]) for i in range(_BATCH_N)])
    _assert_allclose_1e12(batched, looped)


@_PROPERTY_SETTINGS
@given(a=_BATCHED_MULTIVECTOR)
def test_exp_bivector_batches_match_python_loop(a: np.ndarray) -> None:
    bivector_only = np.array(a, copy=True)
    bivector_only[..., : basis.BIVECTOR_SLICE.start] = 0.0
    bivector_only[..., basis.BIVECTOR_SLICE.stop :] = 0.0
    bivector_only *= 0.1  # keep small so the fixed-order Taylor series is accurate

    batched = ops.exp_bivector(jnp.asarray(bivector_only))
    looped = jnp.stack([ops.exp_bivector(jnp.asarray(bivector_only[i])) for i in range(_BATCH_N)])
    _assert_allclose_1e12(batched, looped)


# --- Test contract item 9: dtype ---


@pytest.mark.parametrize(
    "op_name,call",
    [
        ("geometric_product", lambda a, b: ops.geometric_product(a, b)),
        ("reverse", lambda a, b: ops.reverse(a)),
        ("grade", lambda a, b: ops.grade(a, 2)),
        ("scalar_part", lambda a, b: ops.scalar_part(a)),
        ("bivector_part", lambda a, b: ops.bivector_part(a)),
        ("rotor_norm_sq", lambda a, b: ops.rotor_norm_sq(a)),
        ("exp_bivector", lambda a, b: ops.exp_bivector(a)),
        ("commutator", lambda a, b: ops.commutator(a, b)),
    ],
)
def test_outputs_are_float64(op_name: str, call: object) -> None:
    """Every op returns float64 output, even given float32/python-list input."""
    a_list = [float(x) for x in range(16)]
    b_list = [float(x) * 0.5 for x in range(16)]
    result = call(a_list, b_list)  # type: ignore[operator]
    assert jnp.asarray(result).dtype == jnp.float64, op_name


def test_grade_project_outputs_are_float64() -> None:
    a_list = [float(x) for x in range(16)]
    for projection in ops.grade_project(a_list):  # type: ignore[arg-type]
        assert projection.dtype == jnp.float64


def test_normalize_rotor_output_is_float64() -> None:
    identity = jnp.zeros(16, dtype=jnp.float64).at[basis.IDX_SCALAR].set(1.0)
    normalized = ops.normalize_rotor(identity)
    assert normalized.dtype == jnp.float64


# --- Additional sanity checks (not in the numbered test contract, but cheap) ---


@_PROPERTY_SETTINGS
@given(a=_MULTIVECTOR, b=_MULTIVECTOR)
def test_commutator_is_antisymmetric(a: np.ndarray, b: np.ndarray) -> None:
    lhs = ops.commutator(a, b)
    rhs = -ops.commutator(b, a)
    _assert_allclose_1e12(lhs, rhs)


def test_normalize_rotor_gives_unit_norm_for_identity() -> None:
    identity = jnp.zeros(16, dtype=jnp.float64).at[basis.IDX_SCALAR].set(1.0)
    normalized = ops.normalize_rotor(identity)
    assert abs(float(ops.rotor_norm_sq(normalized)) - 1.0) < 1e-12
