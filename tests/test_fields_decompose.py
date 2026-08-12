# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for cliffordclock.fields.decompose (E11 baseline/residual split).

WP2 test contract item 3 ("baseline exactness"): for a purely
uniform+linear field, the baseline fit alone reproduces field and
gradient to machine precision, and the residual delta_E is ~0.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from cliffordclock.fields.decompose import fit_baseline, residual
from cliffordclock.fields.synthetic import constant_gradient_field, sample_on_grid, uniform_field


def test_baseline_exact_for_uniform_linear_field() -> None:
    """E11: baseline reproduces a degree-1 field to machine precision; residual ~0."""
    e0 = np.array([1.0e4, -5.0e3, 2.0e3])
    grad = np.array([[500.0, 12.0, -3.0], [7.0, -300.0, 4.0], [-2.0, 6.0, -200.0]])
    e_fn, grad_fn = constant_gradient_field(e0, grad)
    grid = sample_on_grid(e_fn, ((-1e-3, 1e-3),) * 3, 9)

    baseline = fit_baseline(grid)

    # Fitted coefficients match the true offset/gradient to (near-)machine precision.
    np.testing.assert_allclose(np.asarray(baseline.offset), e0, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(np.asarray(baseline.grad), grad, rtol=1e-8, atol=1e-6)

    # baseline.evaluate/gradient reproduce the field and its gradient everywhere.
    rng = np.random.default_rng(1)
    test_pts = rng.uniform(-0.9e-3, 0.9e-3, size=(200, 3))
    test_pts_j = jnp.asarray(test_pts)

    e_pred = np.asarray(baseline.evaluate(test_pts_j))
    e_true = np.asarray(e_fn(test_pts_j))
    np.testing.assert_allclose(e_pred, e_true, rtol=1e-8, atol=1e-6)

    grad_pred = np.asarray(baseline.gradient(test_pts_j))
    grad_true = np.asarray(grad_fn(test_pts_j))
    np.testing.assert_allclose(grad_pred, grad_true, rtol=1e-8, atol=1e-6)

    # The residual left for the smoother is ~0 (machine-precision noise only).
    delta_e = residual(grid, baseline)
    assert np.max(np.abs(delta_e)) < 1e-6  # V/m, vs field values of order 1e3-1e4 V/m


def test_baseline_exact_for_uniform_field() -> None:
    """Degenerate case of E11: a purely uniform field has zero gradient and zero residual."""
    e0 = np.array([3.0, -7.0, 42.0])
    e_fn, _grad_fn = uniform_field(e0)
    grid = sample_on_grid(e_fn, ((-1.0, 1.0),) * 3, 5)

    baseline = fit_baseline(grid)

    np.testing.assert_allclose(np.asarray(baseline.offset), e0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(baseline.grad), np.zeros((3, 3)), atol=1e-10)

    delta_e = residual(grid, baseline)
    assert np.max(np.abs(delta_e)) < 1e-10
