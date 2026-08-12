# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cl(1,3) basis ordering, signature, and named component indices (E1, E2).

Every multivector in this package is a plain array of shape ``(..., 16)``,
float64, whose 16 trailing components are the blades listed in
``docs/CONVENTIONS.md`` E2, in that exact order. This module is the single
place that ordering is defined; every other module in ``cl13`` (and any
downstream code) must index into the 16-component axis using the constants
and slices defined here rather than bare integer literals.
"""

from __future__ import annotations

#: (E1) Metric signature eta = diag(+1, -1, -1, -1): e_0^2 = +1 (timelike),
#: e_k^2 = -1 for k in {1, 2, 3} (spacelike), e_mu . e_nu = 0 for mu != nu.
#: Keyed by vector index (0..3).
METRIC: dict[int, float] = {0: 1.0, 1: -1.0, 2: -1.0, 3: -1.0}

#: (E2) Basis ordering for the 16-component multivector array. Each entry is
#: a blade e_{mu1 mu2 ...} represented as the sorted tuple of vector indices
#: (mu1 < mu2 < ...), matching the CONVENTIONS.md table exactly:
#:   0      : 1                                    (scalar, grade 0)
#:   1-4    : e_0, e_1, e_2, e_3                    (vectors, grade 1)
#:   5-10   : e_01, e_02, e_03, e_12, e_13, e_23     (bivectors, grade 2)
#:   11-14  : e_012, e_013, e_023, e_123             (trivectors, grade 3)
#:   15     : e_0123                                 (pseudoscalar, grade 4)
BLADES: tuple[tuple[int, ...], ...] = (
    (),
    (0,),
    (1,),
    (2,),
    (3,),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
    (0, 1, 2, 3),
)

#: Number of basis blades in Cl(1,3): 2^4 = 16.
N_BLADES: int = len(BLADES)

_BLADE_TO_INDEX: dict[tuple[int, ...], int] = {blade: idx for idx, blade in enumerate(BLADES)}


def blade_index(blade: tuple[int, ...]) -> int:
    """Look up the component index (E2) of a blade given as a sorted index tuple.

    Parameters
    ----------
    blade : tuple[int, ...]
        A blade's vector indices in strictly increasing order, e.g. ``(1, 3)``
        for ``e_13``. Must be one of the 16 blades in :data:`BLADES`.

    Returns
    -------
    int
        The blade's position (0-15) in the E2 component ordering.
    """
    return _BLADE_TO_INDEX[blade]


#: Grade (number of vector factors) of each of the 16 blades, in E2 order.
GRADE: tuple[int, ...] = tuple(len(blade) for blade in BLADES)

#: (E4) Reverse sign per component, in E2 order: reversing a grade-g blade
#: (reversing the order of its g vector factors) picks up sign
#: (-1)^(g(g-1)/2), i.e. (+, +, -, -, +) for grades (0, 1, 2, 3, 4).
REVERSE_SIGN: tuple[int, ...] = tuple((-1) ** (g * (g - 1) // 2) for g in GRADE)

# --- Named component indices (E2) ---

IDX_SCALAR = 0

IDX_E0 = 1
IDX_E1 = 2
IDX_E2 = 3
IDX_E3 = 4

IDX_E01 = 5
IDX_E02 = 6
IDX_E03 = 7
IDX_E12 = 8
IDX_E13 = 9
IDX_E23 = 10

IDX_E012 = 11
IDX_E013 = 12
IDX_E023 = 13
IDX_E123 = 14

IDX_E0123 = 15

# --- Grade slices into the 16-component trailing axis (E2) ---

#: Grade 0 (scalar): 1 component.
SCALAR_SLICE = slice(0, 1)
#: Grade 1 (vectors e_0..e_3): 4 components.
VECTOR_SLICE = slice(1, 5)
#: Grade 2 (bivectors): 6 components.
BIVECTOR_SLICE = slice(5, 11)
#: Grade 3 (trivectors): 4 components.
TRIVECTOR_SLICE = slice(11, 15)
#: Grade 4 (pseudoscalar): 1 component.
PSEUDOSCALAR_SLICE = slice(15, 16)

#: Grade slices indexed by grade number (0-4), for programmatic grade selection.
GRADE_SLICES: tuple[slice, slice, slice, slice, slice] = (
    SCALAR_SLICE,
    VECTOR_SLICE,
    BIVECTOR_SLICE,
    TRIVECTOR_SLICE,
    PSEUDOSCALAR_SLICE,
)
