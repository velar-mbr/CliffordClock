# SPDX-License-Identifier: AGPL-3.0-or-later
"""Oracle cross-checks against the `clifford` PyPI package (WP1 test contract items 6, 7).

`clifford` is a general Cl(p,q) library; we configure its `Layout` with the
same signature as E1 (Cl(1,3), (+,-,-,-)) and build an explicit index
translation table between its blade ordering and this package's E2
ordering (WP1 orchestrator instructions), rather than assuming the two
orderings coincide.
"""

from __future__ import annotations

import math

import clifford as cf
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from cliffordclock.cl13 import basis, ops

# --- Oracle Layout: Cl(1,3), signature (+, -, -, -) matching E1 ---
#
# `clifford.Cl(p, q)` builds a Layout with 1-indexed vectors e_1..e_{p+q}
# and signature [+1]*p + [-1]*q. Cl(1, 3) therefore gives e_1**2 = +1 and
# e_2**2 = e_3**2 = e_4**2 = -1, matching E1 (e_0**2 = +1, e_k**2 = -1 for
# k = 1, 2, 3) under the correspondence "clifford e_k <-> our e_{k-1}".
_LAYOUT, _ = cf.Cl(1, 3)
assert tuple(int(s) for s in _LAYOUT.sig) == (1, -1, -1, -1), (
    "clifford Layout signature must match E1 (+, -, -, -)"
)

# `_LAYOUT.bladeTupList` gives clifford's internal per-component blade
# ordering as 1-indexed vector-id tuples. Shifting every id down by 1 maps
# clifford's vector labels onto ours (clifford e_k <-> our e_{k-1}); the
# translation table below is built from that explicitly, not assumed.
_CLIFFORD_BLADE_TUPLES: list[tuple[int, ...]] = [
    tuple(vector_id - 1 for vector_id in blade_tuple) for blade_tuple in _LAYOUT.bladeTupList
]

#: TRANSLATION[i] = our E2 component index corresponding to clifford's
#: internal component index i. Built via `basis.blade_index`, so it is
#: correct regardless of whether clifford's internal ordering happens to
#: coincide with ours (it does, but this table does not assume that).
TRANSLATION: np.ndarray = np.array(
    [basis.blade_index(blade_tuple) for blade_tuple in _CLIFFORD_BLADE_TUPLES],
    dtype=np.int64,
)


def to_clifford(mv: jnp.ndarray) -> cf.MultiVector:
    """Convert one (16,) multivector array (E2 order) to a clifford MultiVector."""
    mv_arr = np.asarray(mv, dtype=np.float64)
    clifford_value = mv_arr[TRANSLATION]
    return _LAYOUT.MultiVector(value=clifford_value)


def from_clifford(cmv: cf.MultiVector) -> np.ndarray:
    """Convert a clifford MultiVector to one (16,) multivector array (E2 order)."""
    clifford_value = np.asarray(cmv.value, dtype=np.float64)
    out = np.zeros(16, dtype=np.float64)
    out[TRANSLATION] = clifford_value
    return out


def test_translation_table_is_a_bijection() -> None:
    assert sorted(TRANSLATION.tolist()) == list(range(16))


def test_translation_round_trips_basis_blades() -> None:
    for idx in range(16):
        probe = np.zeros(16, dtype=np.float64)
        probe[idx] = 1.0
        round_tripped = from_clifford(to_clifford(probe))
        np.testing.assert_array_equal(round_tripped, probe)


# --- Test contract item 6: oracle cross-check (>= 1000 random pairs) ---

_ORACLE_SETTINGS = settings(
    max_examples=1000,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

_FLOAT = st.floats(
    min_value=-4.0,
    max_value=4.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)
_MULTIVECTOR = arrays(dtype=np.float64, shape=(16,), elements=_FLOAT)


@_ORACLE_SETTINGS
@given(a=_MULTIVECTOR, b=_MULTIVECTOR)
def test_geometric_product_matches_clifford_oracle(a: np.ndarray, b: np.ndarray) -> None:
    ours = np.asarray(ops.geometric_product(a, b))
    oracle = from_clifford(to_clifford(a) * to_clifford(b))
    np.testing.assert_allclose(ours, oracle, rtol=1e-12, atol=1e-12)


@_ORACLE_SETTINGS
@given(a=_MULTIVECTOR)
def test_reverse_matches_clifford_oracle(a: np.ndarray) -> None:
    ours = np.asarray(ops.reverse(a))
    oracle = from_clifford(~to_clifford(a))
    np.testing.assert_allclose(ours, oracle, rtol=1e-12, atol=1e-12)


@_ORACLE_SETTINGS
@given(a=_MULTIVECTOR)
def test_grade_projections_match_clifford_oracle(a: np.ndarray) -> None:
    projections = ops.grade_project(a)
    a_clifford = to_clifford(a)
    for k, projection in enumerate(projections):
        ours = np.asarray(projection)
        oracle = from_clifford(a_clifford(k))
        np.testing.assert_allclose(ours, oracle, rtol=1e-12, atol=1e-12)


# --- Test contract item 7: rotor exponential ---


def _bivector_from_components(components: dict[int, float]) -> jnp.ndarray:
    value = np.zeros(16, dtype=np.float64)
    for idx, val in components.items():
        value[idx] = val
    return jnp.asarray(value)


def test_exp_bivector_matches_closed_form_for_spacelike_rotation_plane() -> None:
    """B = theta * e_12 is spacelike (e_12^2 = -1): exp(B) = cos(theta) + sin(theta) e_12."""
    theta = 0.37
    bivector = _bivector_from_components({basis.IDX_E12: theta})
    result = np.asarray(ops.exp_bivector(bivector))

    expected = np.zeros(16, dtype=np.float64)
    expected[basis.IDX_SCALAR] = math.cos(theta)
    expected[basis.IDX_E12] = math.sin(theta)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    norm_sq = float(ops.rotor_norm_sq(jnp.asarray(result)))
    assert abs(norm_sq - 1.0) < 1e-12


def test_exp_bivector_matches_closed_form_for_timelike_boost_plane() -> None:
    """B = theta * e_01 is timelike (e_01^2 = +1): exp(B) = cosh(theta) + sinh(theta) e_01."""
    theta = 0.11
    bivector = _bivector_from_components({basis.IDX_E01: theta})
    result = np.asarray(ops.exp_bivector(bivector))

    expected = np.zeros(16, dtype=np.float64)
    expected[basis.IDX_SCALAR] = math.cosh(theta)
    expected[basis.IDX_E01] = math.sinh(theta)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    norm_sq = float(ops.rotor_norm_sq(jnp.asarray(result)))
    assert abs(norm_sq - 1.0) < 1e-12


_SMALL_BIVECTOR_FLOAT = st.floats(
    min_value=-0.2,
    max_value=0.2,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)
_SMALL_BIVECTOR_COMPONENTS = arrays(dtype=np.float64, shape=(6,), elements=_SMALL_BIVECTOR_FLOAT)


def _pack_bivector(components: np.ndarray) -> jnp.ndarray:
    value = np.zeros(16, dtype=np.float64)
    value[basis.BIVECTOR_SLICE] = components
    return jnp.asarray(value)


@_ORACLE_SETTINGS
@given(components=_SMALL_BIVECTOR_COMPONENTS)
def test_exp_bivector_matches_clifford_oracle_for_small_random_bivectors(
    components: np.ndarray,
) -> None:
    bivector = _pack_bivector(components)
    ours = np.asarray(ops.exp_bivector(bivector))
    oracle = from_clifford(to_clifford(bivector).exp())
    np.testing.assert_allclose(ours, oracle, rtol=1e-12, atol=1e-12)

    norm_sq = float(ops.rotor_norm_sq(jnp.asarray(ours)))
    assert abs(norm_sq - 1.0) < 1e-12


def test_exp_bivector_of_zero_is_identity() -> None:
    zero = jnp.zeros(16, dtype=jnp.float64)
    result = np.asarray(ops.exp_bivector(zero))
    expected = np.zeros(16, dtype=np.float64)
    expected[basis.IDX_SCALAR] = 1.0
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("theta", [0.001, 0.05, 0.2, -0.3])
def test_normalize_rotor_is_idempotent_after_exp(theta: float) -> None:
    bivector = _bivector_from_components({basis.IDX_E12: theta, basis.IDX_E01: theta * 0.5})
    rotor = ops.exp_bivector(bivector)
    normalized = ops.normalize_rotor(rotor)
    assert abs(float(ops.rotor_norm_sq(normalized)) - 1.0) < 1e-12
