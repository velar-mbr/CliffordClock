# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background/perturbation field decomposition (CONVENTIONS.md E11).

(E11) ``E_total(r) = E_0(r) + δE(r)``: ``E_0`` is a low-order analytical
baseline (uniform + linear terms: a degree-1 vector polynomial in each
component) fitted to the data by ordinary least squares, and ``δE`` is the
residual left for the RBF smoother (``smoother.py``) to handle.

Rationale: the downstream physics (CONVENTIONS.md §7) evaluates fractional
frequency shifts at the ~1e-18 level from ``δE·μ`` terms. If the smoother
had to represent the field's full dynamic range, including its dominant
uniform/linear part (already exactly representable in closed form), its
residuals and their gradients would carry the same dynamic range, and the
fit conditioning (and thus the 1e-18-level arithmetic that consumes
``∇E``) would degrade. Subtracting the exact analytical baseline first
keeps ``δE`` small and well-conditioned for the smoother to fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cliffordclock.fields.io import FieldGrid


@dataclass(frozen=True, eq=False)
class Baseline:
    """Degree-1 vector-polynomial baseline field ``E_0(r)`` (E11).

    ``E_0(r) = offset + r @ grad``, i.e. component ``j``:
    ``E_0_j(r) = offset[j] + Σ_i r[i] * grad[i, j]``.

    Attributes
    ----------
    offset : jax.Array, shape (3,)
        Field value at the origin, V/m.
    grad : jax.Array, shape (3, 3)
        Constant gradient tensor, ``grad[i, j] = ∂_i E_j`` (E13), V/m².
        Exact and position-independent by construction (degree-1
        polynomial), so ``gradient(pos)`` below is exact closed form, not
        an approximation.
    """

    offset: jnp.ndarray
    grad: jnp.ndarray

    def evaluate(self, pos: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the baseline field.

        Parameters
        ----------
        pos : jax.Array, shape (N, 3)
            Query positions, meters.

        Returns
        -------
        jax.Array, shape (N, 3)
            ``E_0(pos)``, V/m.
        """
        pos = jnp.asarray(pos, dtype=jnp.float64)
        return self.offset[None, :] + pos @ self.grad

    def gradient(self, pos: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the (constant) baseline gradient tensor.

        Parameters
        ----------
        pos : jax.Array, shape (N, 3)
            Query positions, meters (only used for broadcasting shape;
            the baseline gradient does not depend on position).

        Returns
        -------
        jax.Array, shape (N, 3, 3)
            ``grad_E[n, i, j] = ∂_i E_j`` (E13), V/m², identical for every
            ``n`` since the baseline is degree-1.
        """
        pos = jnp.asarray(pos, dtype=jnp.float64)
        n = pos.shape[0]
        return jnp.broadcast_to(self.grad, (n, 3, 3))


def fit_baseline(grid: FieldGrid) -> Baseline:
    """Fit the degree-1 vector-polynomial baseline ``E_0`` (E11) by least squares.

    Parameters
    ----------
    grid : FieldGrid
        Ingested field data; ``grid.points`` (N, 3) m, ``grid.values``
        (N, 3) V/m.

    Returns
    -------
    Baseline
        Fitted baseline with exact closed-form ``evaluate``/``gradient``.
    """
    points = grid.points
    values = grid.values
    design = np.hstack([np.ones((points.shape[0], 1)), points])  # (N, 4): [1, x, y, z]
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(design, values, rcond=None)  # (4, 3)
    offset = np.ascontiguousarray(coeffs[0, :])
    grad = np.ascontiguousarray(coeffs[1:, :])
    return Baseline(offset=jnp.asarray(offset), grad=jnp.asarray(grad))


def residual(grid: FieldGrid, baseline: Baseline) -> NDArray[np.float64]:
    """Compute the residual ``δE = E_total − E_0`` at the grid points (E11).

    Parameters
    ----------
    grid : FieldGrid
        Ingested field data.
    baseline : Baseline
        Fitted baseline, typically from :func:`fit_baseline`.

    Returns
    -------
    NDArray[np.float64], shape (N, 3)
        ``δE`` at ``grid.points``, V/m. This is what ``smoother.py`` fits
        the RBF interpolant to.
    """
    e0 = np.asarray(baseline.evaluate(jnp.asarray(grid.points)), dtype=np.float64)
    delta_e: NDArray[np.float64] = grid.values - e0
    return delta_e
