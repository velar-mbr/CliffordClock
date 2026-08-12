# SPDX-License-Identifier: AGPL-3.0-or-later
"""Structure tensor tests (WP1 test contract item 1; docs/CONVENTIONS.md E1-E3).

Checks the geometric-product structure tensor directly: shape, entry
range, determinism, and the exact metric relation
``e_mu e_nu + e_nu e_mu == 2 eta_mu_nu`` for all basis vector pairs.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.cl13 import basis, ops
from cliffordclock.cl13.structure import STRUCTURE_TENSOR, build_structure_tensor

_VECTOR_INDEX = {0: basis.IDX_E0, 1: basis.IDX_E1, 2: basis.IDX_E2, 3: basis.IDX_E3}


def _basis_vector(mu: int) -> jnp.ndarray:
    value = np.zeros(16, dtype=np.float64)
    value[_VECTOR_INDEX[mu]] = 1.0
    return jnp.asarray(value)


def test_structure_tensor_shape_and_dtype() -> None:
    assert STRUCTURE_TENSOR.shape == (16, 16, 16)
    assert STRUCTURE_TENSOR.dtype == np.float64


def test_structure_tensor_entries_in_minus_one_zero_plus_one() -> None:
    assert set(np.unique(STRUCTURE_TENSOR).tolist()) <= {-1.0, 0.0, 1.0}


def test_structure_tensor_is_deterministic() -> None:
    """Rebuilding the tensor from scratch reproduces the frozen constant exactly."""
    rebuilt = build_structure_tensor()
    np.testing.assert_array_equal(rebuilt, STRUCTURE_TENSOR)


def test_structure_tensor_is_frozen() -> None:
    with pytest.raises(ValueError):
        STRUCTURE_TENSOR[0, 0, 0] = 5.0


@pytest.mark.parametrize("mu,nu", list(itertools.product(range(4), range(4))))
def test_metric_relation_exact(mu: int, nu: int) -> None:
    """(E1) e_mu e_nu + e_nu e_mu = 2 eta_mu_nu exactly, for every basis vector pair."""
    e_mu = _basis_vector(mu)
    e_nu = _basis_vector(nu)
    lhs = ops.geometric_product(e_mu, e_nu) + ops.geometric_product(e_nu, e_mu)

    expected = np.zeros(16, dtype=np.float64)
    if mu == nu:
        expected[basis.IDX_SCALAR] = 2.0 * basis.METRIC[mu]
    # else: eta_mu_nu = 0 for mu != nu (E1), so expected stays all-zero.

    np.testing.assert_array_equal(np.asarray(lhs), expected)
