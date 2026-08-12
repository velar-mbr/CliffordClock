# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cl(1,3) geometric-product structure tensor (E3).

Builds the dense ``(16, 16, 16)`` structure tensor ``T`` such that
``(A B)_k = sum_{i,j} T[i, j, k] A_i B_j`` reproduces the Cl(1,3) geometric
product for multivectors ``A``, ``B`` given in the E2 component ordering.

``T`` is generated programmatically from first principles (E1 signature +
E2 ordering): each entry ``T[i, j, k]`` is derived from the geometric
product of basis blades ``i`` and ``j``, computed by anticommuting vector
factors past each other (a sign flip per transposition, since
``e_mu e_nu = -e_nu e_mu`` for ``mu != nu``) and contracting equal adjacent
factors through the metric (E1). This is *not* a hand-written or imported
lookup table; the 4096-entry tensor is derived at import time from the
16-blade list and the four-entry metric dict in :mod:`cliffordclock.cl13.basis`.

By design, downstream code (``ops.py``) must evaluate all geometric
products via ``jnp.einsum`` against this dense tensor. Integer pair-table
/ lookup-table product kernels or any "optimized sparse index" formulation
are deliberately excluded from this repository.
"""

from __future__ import annotations

import numpy as np

from cliffordclock.cl13.basis import BLADES, METRIC, N_BLADES, blade_index


def _multiply_blades(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    """Multiply two basis blades via anticommutation and metric contraction (E1).

    Concatenates the vector-index factors of ``a`` and ``b``, then repeatedly
    (1) swaps adjacent out-of-order factors, flipping the running sign
    (anticommutation: ``e_mu e_nu = -e_nu e_mu`` for ``mu != nu``, E1), and
    (2) contracts adjacent equal factors via the metric (``e_mu e_mu =
    METRIC[mu]``, E1), removing both, until the remaining factors are sorted
    with no duplicates -- i.e. a valid basis blade in the E2 sense.

    Parameters
    ----------
    a, b : tuple[int, ...]
        Sorted vector-index tuples for the two input blades.

    Returns
    -------
    sign : int
        Overall sign (+1 or -1) accumulated from transpositions and metric
        contractions.
    result : tuple[int, ...]
        The sorted vector-index tuple of the resulting blade.
    """
    factors = list(a) + list(b)
    sign = 1
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(factors) - 1:
            if factors[i] == factors[i + 1]:
                sign *= int(METRIC[factors[i]])
                del factors[i : i + 2]
                changed = True
                # Do not advance i: the new neighbours at i must be rechecked.
            elif factors[i] > factors[i + 1]:
                factors[i], factors[i + 1] = factors[i + 1], factors[i]
                sign *= -1
                changed = True
                i += 1
            else:
                i += 1
    return sign, tuple(factors)


def build_structure_tensor() -> np.ndarray:
    """Build the ``(16, 16, 16)`` Cl(1,3) geometric-product structure tensor (E3).

    Returns
    -------
    numpy.ndarray
        Shape ``(16, 16, 16)``, dtype float64, entries in ``{-1, 0, +1}``.
        ``(A B)_k = sum_{i,j} T[i, j, k] A_i B_j`` for multivectors ``A``,
        ``B`` in the E2 component ordering (E3).
    """
    tensor = np.zeros((N_BLADES, N_BLADES, N_BLADES), dtype=np.float64)
    for i, blade_i in enumerate(BLADES):
        for j, blade_j in enumerate(BLADES):
            sign, result_blade = _multiply_blades(blade_i, blade_j)
            k = blade_index(result_blade)
            tensor[i, j, k] = sign
    return tensor


#: The frozen (16, 16, 16) structure tensor, generated once at import time.
STRUCTURE_TENSOR: np.ndarray = build_structure_tensor()
STRUCTURE_TENSOR.flags.writeable = False
