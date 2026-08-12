# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cl(1,3) multivector operations (E3-E6).

A multivector is a plain ``jax.numpy`` array of shape ``(..., 16)``, dtype
float64, whose trailing axis is ordered per E2 (see
:mod:`cliffordclock.cl13.basis`). All functions here are pure, batched over
arbitrary leading (``...``) axes, dimensionless, and safe under `jax.jit`,
`jax.vmap`, and `jax.grad`: no data-dependent Python control flow occurs on
traced values (grade selectors branch only on static Python ints; the
Taylor/squaring loop in :func:`exp_bivector` unrolls over static Python
ints at trace time).

No custom multivector class: this is plain arrays plus pure functions, so
the hot path (:func:`geometric_product`) is a single `jnp.einsum` against
the precomputed structure tensor (see ``structure.py``) rather than an
integer pair-table / lookup-table product kernel.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import lax

from cliffordclock.cl13.basis import (
    BIVECTOR_SLICE,
    GRADE_SLICES,
    IDX_E0123,
    IDX_SCALAR,
    N_BLADES,
    REVERSE_SIGN,
)
from cliffordclock.cl13.structure import STRUCTURE_TENSOR

#: Geometric-product structure tensor (E3) as a JAX float64 array.
_STRUCTURE_TENSOR_JAX = jnp.asarray(STRUCTURE_TENSOR, dtype=jnp.float64)

#: Per-component reverse sign (E4), as a JAX float64 array, shape (16,).
_REVERSE_SIGN_JAX = jnp.asarray(REVERSE_SIGN, dtype=jnp.float64)

#: Per-grade 0/1 masks (E2 grade slices), shape (16,) each, used by `grade`.
_GRADE_MASKS: tuple[jnp.ndarray, ...] = tuple(
    jnp.zeros((N_BLADES,), dtype=jnp.float64).at[sl].set(1.0) for sl in GRADE_SLICES
)

#: Number of dyadic halvings applied before the Taylor series in
#: `exp_bivector`, and undone by repeated squaring afterwards (scaled Taylor
#: series with squaring, E6). Fixed (not data-dependent) for jit/grad safety.
_EXP_SCALING_HALVINGS = 10

_TWO_PI = 2.0 * math.pi

#: Unit pseudoscalar e_0123 as a (16,) multivector, used by the range
#: reduction in `exp_bivector` (it maps a bivector B to the bivector I B in
#: the orthogonal-complement plane; I commutes with all even elements and
#: I^2 = -1 in Cl(1,3)).
_PSEUDOSCALAR = jnp.zeros((N_BLADES,), dtype=jnp.float64).at[IDX_E0123].set(1.0)


def _reduce_compact_angle(b: jnp.ndarray) -> jnp.ndarray:
    """Range-reduce the compact (rotation) angle of a bivector modulo 2*pi.

    Any bivector ``B`` in Cl(1,3) squares to a scalar plus a pseudoscalar,
    ``B^2 = s + p I`` with ``I = e_0123`` (its grade-2 part is the vanishing
    self-commutator). Writing ``r = sqrt(s^2 + p^2)``, `B` splits into two
    commuting invariant blades ``B = B_boost + B_rot`` with
    ``B_boost^2 = a^2 = (r + s)/2 >= 0`` (non-compact, boost-like) and
    ``B_rot^2 = -theta^2``, ``theta^2 = (r - s)/2 >= 0`` (compact,
    rotation-like), where ``B_rot = (theta^2 B + (p/2) I B) / r``. Because
    ``exp`` of the compact blade is 2*pi-periodic and the blades commute,

        ``exp(B) = exp(B - 2*pi*k * B_rot/theta)``  for any integer `k`,

    exactly. This function returns that reduced bivector with
    ``k = round(theta / 2*pi)``, bringing the compact angle into
    ``[-pi, pi]`` where the fixed-order scaled Taylor series in
    :func:`exp_bivector` is accurate. `k` is wrapped in
    `jax.lax.stop_gradient` (locally constant), so gradients of the reduced
    exponential equal gradients of the original.

    For ``theta <= pi`` the correction term is exactly zero (``k = 0``
    multiplies it), so small-angle inputs pass through bitwise-unchanged.
    The intermediate `sqrt` arguments are clamped to 1 via `jnp.where`
    wherever the true value could reach 0 (`B` zero, null, or a pure
    boost); in every such case ``theta <= 1 < pi`` forces ``k = 0``, so the
    clamps never alter a result but do keep reverse-mode gradients free of
    ``0 * inf`` NaNs at those points. Everything is a fixed sequence of
    array ops: safe under `jax.jit`, `jax.vmap`, and `jax.grad`.

    Only meaningful for (near-)bivector input: the invariant split above
    relies on ``B^2`` having no grade-2 part. See the accuracy contract in
    :func:`exp_bivector`.
    """
    b_sq = geometric_product(b, b)
    s = b_sq[..., IDX_SCALAR]
    p = b_sq[..., IDX_E0123]
    q = s * s + p * p
    # If q <= 1 then theta <= 1 < pi and k = 0 below; clamp keeps sqrt smooth.
    r = jnp.sqrt(jnp.where(q > 1.0, q, 1.0))
    theta_sq = 0.5 * (r - s)
    # Same reasoning: theta_sq <= 1 implies k = 0; clamp only guards grads.
    theta = jnp.sqrt(jnp.where(theta_sq > 1.0, theta_sq, 1.0))
    k = lax.stop_gradient(jnp.round(theta / _TWO_PI))
    coefficient = _TWO_PI * k / (theta * r)
    i_b = geometric_product(_PSEUDOSCALAR, b)
    correction = coefficient[..., None] * (
        (theta * theta)[..., None] * b + 0.5 * p[..., None] * i_b
    )
    return b - correction


def geometric_product(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Cl(1,3) geometric product ``A B`` (E3).

    Evaluated as a single `jnp.einsum` contraction against the precomputed
    dense structure tensor `T` (see :mod:`cliffordclock.cl13.structure`):
    ``(A B)_k = sum_{i,j} T[i, j, k] A_i B_j``. No lookup-table or
    pair-table kernel is used, by design.

    Parameters
    ----------
    a, b : jax.Array
        Multivectors, shape ``(..., 16)``, float64. Leading axes broadcast
        against each other as in ordinary array broadcasting.

    Returns
    -------
    jax.Array
        The geometric product ``A B``, shape ``(..., 16)``, float64,
        dimensionless.
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    return jnp.einsum("...i,ijk,...j->...k", a, _STRUCTURE_TENSOR_JAX, b)


def reverse(a: jnp.ndarray) -> jnp.ndarray:
    """Reverse of a multivector, ``A~`` (E4).

    Flips the sign of each grade-``g`` component by ``(-1)^(g(g-1)/2)``,
    i.e. grades ``(0, 1, 2, 3, 4)`` pick up signs ``(+, +, -, -, +)``.

    Parameters
    ----------
    a : jax.Array
        Multivector, shape ``(..., 16)``, float64.

    Returns
    -------
    jax.Array
        ``A~``, shape ``(..., 16)``, float64, dimensionless.
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    return a * _REVERSE_SIGN_JAX


def grade(a: jnp.ndarray, k: int) -> jnp.ndarray:
    """Grade-``k`` projection of a multivector (E2 grade structure).

    Parameters
    ----------
    a : jax.Array
        Multivector, shape ``(..., 16)``, float64.
    k : int
        Grade to select, one of ``0, 1, 2, 3, 4``. Must be a static Python
        int (not a traced value): it selects which of the five precomputed
        0/1 masks to apply, so this function is jit-safe only when `k` is
        static (e.g. closed over, or passed via `functools.partial` /
        `static_argnums` under `jax.jit`).

    Returns
    -------
    jax.Array
        `a` with all components outside grade `k` zeroed, shape
        ``(..., 16)``, float64, dimensionless.
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    return a * _GRADE_MASKS[k]


def grade_project(
    a: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project a multivector onto each of its five grades (E2 grade structure).

    Parameters
    ----------
    a : jax.Array
        Multivector, shape ``(..., 16)``, float64.

    Returns
    -------
    tuple of 5 jax.Array
        ``(grade0, grade1, grade2, grade3, grade4)``, each shape
        ``(..., 16)``, float64, dimensionless, each `a` masked to that
        single grade. The five results are disjoint-support and sum
        exactly to `a` (grade completeness, WP1 test contract item 5).
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    return tuple(a * mask for mask in _GRADE_MASKS)  # type: ignore[return-value]


def scalar_part(a: jnp.ndarray) -> jnp.ndarray:
    """Scalar (grade-0) component of a multivector, as a bare number.

    Convenience selector distinct from ``grade(a, 0)``: this returns just
    the scalar value (last-axis size 1 squeezed away), not a 16-component
    multivector with the other 15 components zeroed.

    Parameters
    ----------
    a : jax.Array
        Multivector, shape ``(..., 16)``, float64.

    Returns
    -------
    jax.Array
        Shape ``(...,)``, float64, dimensionless.
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    return a[..., IDX_SCALAR]


def bivector_part(a: jnp.ndarray) -> jnp.ndarray:
    """Bivector (grade-2) components of a multivector, packed.

    Convenience selector distinct from ``grade(a, 2)``: this returns just
    the 6 bivector components (E2 order: e01, e02, e03, e12, e13, e23), not
    a 16-component multivector with the other 10 components zeroed.

    Parameters
    ----------
    a : jax.Array
        Multivector, shape ``(..., 16)``, float64.

    Returns
    -------
    jax.Array
        Shape ``(..., 6)``, float64, dimensionless.
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    return a[..., BIVECTOR_SLICE]


def rotor_norm_sq(a: jnp.ndarray) -> jnp.ndarray:
    """Rotor norm squared, ``<A A~>_0`` (E5).

    Parameters
    ----------
    a : jax.Array
        Multivector, shape ``(..., 16)``, float64. For a rotor `R`, `R R~`
        is a scalar (its non-scalar grades vanish); this function only
        extracts the scalar part, so it is well-defined for any
        multivector, not only rotors.

    Returns
    -------
    jax.Array
        Shape ``(...,)``, float64, dimensionless.
    """
    return scalar_part(geometric_product(a, reverse(a)))


def normalize_rotor(r: jnp.ndarray) -> jnp.ndarray:
    """Normalize a rotor to unit norm, ``R / sqrt(<R R~>_0)`` (E5).

    Assumes ``rotor_norm_sq(r) > 0`` (true for every rotor produced by
    :func:`exp_bivector` on a physically valid bivector in this package's
    use cases -- E5/E6). No branch or clamp is applied for non-positive
    norms; callers passing a degenerate or non-rotor multivector will get
    `nan`/`inf` back rather than a silently "safe" fallback value.

    Parameters
    ----------
    r : jax.Array
        Multivector (intended to be a rotor), shape ``(..., 16)``, float64.

    Returns
    -------
    jax.Array
        Shape ``(..., 16)``, float64, dimensionless, with
        ``rotor_norm_sq`` equal to 1 (to numerical precision).
    """
    r = jnp.asarray(r, dtype=jnp.float64)
    norm = jnp.sqrt(rotor_norm_sq(r))
    return r / norm[..., None]


def exp_bivector(b: jnp.ndarray, order: int = 12) -> jnp.ndarray:
    """Exponential of a bivector, ``exp(B) = sum_n B^n / n!`` (E6).

    Evaluated as a *scaled* Taylor series with squaring (E6), after an
    exact 2*pi range reduction of the compact (rotation-like) invariant
    component (:func:`_reduce_compact_angle`). The reduced bivector `B'`
    satisfies ``exp(B') = exp(B)`` identically and has compact angle in
    ``[-pi, pi]``; it is then halved `_EXP_SCALING_HALVINGS` times (so the
    series is evaluated at ``B' / 2**_EXP_SCALING_HALVINGS``, where a fixed
    `order`-term truncation is far more accurate), and the result is
    repeatedly squared (`exp(x) = (exp(x/2))^2`, applied via
    :func:`geometric_product`) to recover `exp(B)`. The Taylor order, the
    number of halvings, and the reduction are all fixed sequences of array
    ops on static Python ints -- safe under `jax.jit`, `jax.vmap`, and
    `jax.grad` (no data-dependent Python branching).

    For a bivector `B` in this package's use (timelike or spacelike, per
    the WP1 test contract), `exp(B)` is a rotor: ``R R~ = 1`` to numerical
    precision (E6).

    Accuracy contract (measured, pure-``e_12`` generator of angle
    ``theta``; rotor-norm error ``|<R R~>_0 - 1|``):

    ======================  ==================================
    ``|theta|``             norm error
    ======================  ==================================
    ``<= pi``               ``< 1e-12`` (reduction inactive;
                            output bitwise-identical to the
                            unreduced kernel)
    ``<= 1e4``              ``< 1e-12``
    ``<= 1e6``              ``< 1e-9``
    ======================  ==================================

    Error grows roughly as ``theta * 1e-16`` (cancellation in
    ``B - 2*pi*k*B_rot/theta``). Without the reduction, the fixed-order
    series degraded silently: ``1.9e-7`` at ``theta ~ 1e3``, finite
    garbage (``~1e112``) at ``5e3``, NaN from ``1e4``. Boost-like
    (non-compact) magnitudes are not reduced -- ``exp`` of a boost
    genuinely grows as ``exp(a)`` and overflows float64 to ``inf`` for
    ``a >~ 710``.

    Parameters
    ----------
    b : jax.Array
        Bivector, shape ``(..., 16)``, float64. Only grade-2 components are
        expected to be nonzero; this function does not enforce that. The
        range reduction relies on ``B^2`` having only scalar and
        pseudoscalar parts, which holds for every bivector but not for
        arbitrary multivectors: non-bivector input is supported (and
        matches the plain Taylor series) only while the computed compact
        angle stays ``<= pi``, i.e. small inputs. The E6 rotor guarantee
        (``R R~ = 1``) likewise only holds for true bivectors.
    order : int, default 12
        Number of Taylor terms (beyond the constant term) to sum, evaluated
        at the scaled-down bivector. Must be a static Python int.

    Returns
    -------
    jax.Array
        ``exp(B)``, shape ``(..., 16)``, float64, dimensionless.
    """
    b = jnp.asarray(b, dtype=jnp.float64)
    b = _reduce_compact_angle(b)
    scale = float(2**_EXP_SCALING_HALVINGS)
    b_scaled = b / scale

    identity = jnp.zeros_like(b_scaled).at[..., IDX_SCALAR].set(1.0)
    acc = identity
    term = identity
    for n in range(1, order + 1):
        term = geometric_product(term, b_scaled) / n
        acc = acc + term

    for _ in range(_EXP_SCALING_HALVINGS):
        acc = geometric_product(acc, acc)
    return acc


def commutator(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Commutator product, ``½(A B − B A)``.

    Not itself a numbered equation in ``docs/CONVENTIONS.md``; built
    directly from the geometric product (E3), which is the only algebraic
    primitive this package defines.

    Parameters
    ----------
    a, b : jax.Array
        Multivectors, shape ``(..., 16)``, float64.

    Returns
    -------
    jax.Array
        Shape ``(..., 16)``, float64, dimensionless.
    """
    return 0.5 * (geometric_product(a, b) - geometric_product(b, a))
