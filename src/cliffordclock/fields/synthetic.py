# SPDX-License-Identifier: AGPL-3.0-or-later
"""Closed-form synthetic E-fields with hand-derived exact gradients.

Used across the project's test suite (WP2 and downstream WPs) as ground
truth: unlike ``smoother.FieldSmoother``, whose ``∇E`` comes from
autodiff of a fitted interpolant, every gradient here is an independent,
hand-derived closed form, so tests that compare a smoother's autodiff
gradient against these are not circular.

Each field factory returns ``(e_fn, grad_fn)``:

- ``e_fn(pos)``: ``(N, 3)`` positions (m) -> ``(N, 3)`` field (V/m).
- ``grad_fn(pos)``: ``(N, 3)`` positions (m) -> ``(N, 3, 3)`` gradient
  tensor, ``grad_E[n, i, j] = ∂_i E_j`` (E13), V/m².

:func:`sample_on_grid` builds a :class:`~cliffordclock.fields.io.FieldGrid`
by evaluating an ``e_fn`` on a regular grid, for feeding into
``FieldSmoother.fit``. :func:`as_field_fn` adapts an ``(e_fn, grad_fn)``
pair to the single combined ``pos -> (E, grad_E)`` callable convention used
by :meth:`~cliffordclock.fields.smoother.FieldSmoother.evaluate` and the
rotor integrator's :data:`~cliffordclock.integrator.worldline.FieldFn`
(WP6 interface note; see :mod:`cliffordclock.pipeline` for its use).
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from cliffordclock.fields.io import FieldGrid

#: A closed-form field: (N, 3) m -> (N, 3) V/m.
FieldFn = Callable[[jnp.ndarray], jnp.ndarray]
#: A closed-form gradient tensor: (N, 3) m -> (N, 3, 3) V/m^2, grad[..., i, j] = d_i E_j.
GradFn = Callable[[jnp.ndarray], jnp.ndarray]


def uniform_field(e0: jnp.ndarray) -> tuple[FieldFn, GradFn]:
    """Spatially constant field ``E(r) = e0``.

    Parameters
    ----------
    e0 : array-like, shape (3,)
        The (constant) field vector, V/m.

    Returns
    -------
    (e_fn, grad_fn)
        ``grad_fn`` is identically zero (E13: ``∇E = 0`` for a uniform
        field).
    """
    e0_arr = jnp.asarray(e0, dtype=jnp.float64)

    def e_fn(pos: jnp.ndarray) -> jnp.ndarray:
        pos = jnp.asarray(pos, dtype=jnp.float64)
        return jnp.broadcast_to(e0_arr, pos.shape)

    def grad_fn(pos: jnp.ndarray) -> jnp.ndarray:
        pos = jnp.asarray(pos, dtype=jnp.float64)
        return jnp.zeros((pos.shape[0], 3, 3), dtype=jnp.float64)

    return e_fn, grad_fn


def constant_gradient_field(e0: jnp.ndarray, grad: jnp.ndarray) -> tuple[FieldFn, GradFn]:
    """Degree-1 field ``E(r) = e0 + r @ grad`` with a fixed gradient tensor.

    Parameters
    ----------
    e0 : array-like, shape (3,)
        Field value at the origin, V/m.
    grad : array-like, shape (3, 3)
        Constant gradient tensor, ``grad[i, j] = ∂_i E_j`` (E13), V/m².

    Returns
    -------
    (e_fn, grad_fn)
        ``grad_fn`` returns ``grad`` broadcast to every query point
        (E13: exact for a degree-1 field).
    """
    e0_arr = jnp.asarray(e0, dtype=jnp.float64)
    grad_arr = jnp.asarray(grad, dtype=jnp.float64)

    def e_fn(pos: jnp.ndarray) -> jnp.ndarray:
        pos = jnp.asarray(pos, dtype=jnp.float64)
        return e0_arr[None, :] + pos @ grad_arr

    def grad_fn(pos: jnp.ndarray) -> jnp.ndarray:
        pos = jnp.asarray(pos, dtype=jnp.float64)
        return jnp.broadcast_to(grad_arr, (pos.shape[0], 3, 3))

    return e_fn, grad_fn


def quadrupole_field(k: float) -> tuple[FieldFn, GradFn]:
    """Spherical quadrupole field ``E(r) = k · (x, y, −2z)``.

    A traceless linear (degree-1) field, used as the WP2 round-trip test
    field. Its gradient tensor is the constant diagonal
    ``diag(k, k, −2k)`` (E13); this is a special case of
    :func:`constant_gradient_field` with ``e0 = 0``.

    Parameters
    ----------
    k : float
        Quadrupole strength, V/m² (so ``E`` has units V/m given ``r`` in m).

    Returns
    -------
    (e_fn, grad_fn)
    """
    grad_arr = jnp.diag(jnp.asarray([k, k, -2.0 * k], dtype=jnp.float64))
    return constant_gradient_field(jnp.zeros(3, dtype=jnp.float64), grad_arr)


def gaussian_bump_field(
    amplitude: jnp.ndarray, center: jnp.ndarray, width: float
) -> tuple[FieldFn, GradFn]:
    """Smooth, spatially localized (Gaussian) field perturbation.

    ``E(r) = amplitude · exp(−‖r − center‖² / (2 width²))``: a fixed
    direction vector ``amplitude`` modulated by an isotropic Gaussian
    envelope centered at ``center``. Used to synthesize a nonlinear
    residual (after subtracting a linear baseline) that genuinely
    exercises the RBF smoother, unlike the purely linear
    :func:`quadrupole_field`/:func:`constant_gradient_field`.

    Parameters
    ----------
    amplitude : array-like, shape (3,)
        Peak field vector at ``r = center``, V/m.
    center : array-like, shape (3,)
        Bump center, m.
    width : float
        Gaussian standard deviation (envelope width), m.

    Returns
    -------
    (e_fn, grad_fn)
        ``grad_fn`` is the hand-derived exact gradient
        ``∂_i E_j(r) = amplitude_j · (−(r_i − center_i) / width²) ·
        exp(−‖r − center‖² / (2 width²))``.
    """
    amp_arr = jnp.asarray(amplitude, dtype=jnp.float64)
    center_arr = jnp.asarray(center, dtype=jnp.float64)
    width = float(width)

    def _envelope(pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        offset = pos - center_arr[None, :]  # (N, 3)
        sq_dist = jnp.sum(offset * offset, axis=-1)  # (N,)
        return offset, jnp.exp(-sq_dist / (2.0 * width**2))  # (N, 3), (N,)

    def e_fn(pos: jnp.ndarray) -> jnp.ndarray:
        pos = jnp.asarray(pos, dtype=jnp.float64)
        _offset, envelope = _envelope(pos)
        return amp_arr[None, :] * envelope[:, None]

    def grad_fn(pos: jnp.ndarray) -> jnp.ndarray:
        pos = jnp.asarray(pos, dtype=jnp.float64)
        offset, envelope = _envelope(pos)
        # d/dr_i exp(-|r-c|^2 / 2w^2) = -(r_i - c_i) / w^2 * exp(...)
        d_envelope_dr = -(offset / width**2) * envelope[:, None]  # (N, 3), index i
        # grad[n, i, j] = amplitude_j * d_envelope_dr[n, i]
        return amp_arr[None, None, :] * d_envelope_dr[:, :, None]

    return e_fn, grad_fn


def as_field_fn(
    e_fn: FieldFn, grad_fn: GradFn
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Adapt an ``(e_fn, grad_fn)`` pair to the combined ``pos -> (E, grad_E)`` convention.

    WP6 interface note (from the WP3 review): every factory in this module
    returns two separate callables, each batched over a leading ``(N, 3)``
    axis. Two other parts of the codebase use one *combined* callable
    instead --
    :meth:`~cliffordclock.fields.smoother.FieldSmoother.evaluate` and the
    rotor integrator's :data:`~cliffordclock.integrator.worldline.FieldFn`,
    which additionally calls the callable with a single un-batched
    ``(3,)`` position at every integration step, not a batch. This adapter
    bridges both mismatches at once -- two callables -> one, and dual-mode
    ``(3,)``-or-``(N, 3)`` input -> matching-rank output -- so call sites
    standardize on the combined convention instead of writing their own
    ad-hoc per-call-site lambda. See :mod:`cliffordclock.pipeline` for the
    one place this is used to build an integrator ``FieldFn`` from a
    synthetic field factory.

    Parameters
    ----------
    e_fn : FieldFn
        A synthetic field callable, e.g. from :func:`quadrupole_field`.
    grad_fn : GradFn
        The matching gradient callable.

    Returns
    -------
    Callable[[pos], (E, grad_E)]
        ``pos``: shape ``(3,)`` or ``(N, 3)``, meters. Returns ``E`` (V/m)
        and ``grad_E`` (V/m², E13: ``grad_E[..., i, j] = ∂_i E_j``), each
        unbatched (``(3,)``, ``(3, 3)``) when `pos` is unbatched, batched
        (``(N, 3)``, ``(N, 3, 3)``) when `pos` is batched -- matching
        :meth:`~cliffordclock.fields.smoother.FieldSmoother.evaluate`'s
        dual-mode contract exactly.
    """

    def field_fn(pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        pos = jnp.asarray(pos, dtype=jnp.float64)
        single = pos.ndim == 1
        pos2d = pos[None, :] if single else pos
        e = e_fn(pos2d)
        grad_e = grad_fn(pos2d)
        if single:
            return e[0], grad_e[0]
        return e, grad_e

    return field_fn


def sample_on_grid(
    e_fn: FieldFn,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    n_per_axis: int,
) -> FieldGrid:
    """Evaluate ``e_fn`` on a regular ``n_per_axis**3`` grid to build a :class:`FieldGrid`.

    Parameters
    ----------
    e_fn : FieldFn
        A closed-form field, e.g. from :func:`quadrupole_field`.
    bounds : tuple of 3 (low, high) pairs
        Axis-aligned bounding box ``((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))``, m.
    n_per_axis : int
        Number of samples per axis (grid has ``n_per_axis**3`` points total).

    Returns
    -------
    FieldGrid
        ``regular=True`` grid with ``e_fn`` sampled at every grid point.
    """
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = bounds
    x_axis = np.linspace(x_lo, x_hi, n_per_axis)
    y_axis = np.linspace(y_lo, y_hi, n_per_axis)
    z_axis = np.linspace(z_lo, z_hi, n_per_axis)
    axes = (x_axis, y_axis, z_axis)
    x_grid, y_grid, z_grid = np.meshgrid(*axes, indexing="ij")
    points = np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=-1)
    points = np.ascontiguousarray(points, dtype=np.float64)
    values = np.asarray(e_fn(jnp.asarray(points)), dtype=np.float64)
    return FieldGrid(
        points=points,
        values=values,
        regular=True,
        axes=axes,
        shape=(n_per_axis, n_per_axis, n_per_axis),
        metadata={"source": "synthetic"},
    )
