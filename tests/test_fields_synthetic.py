# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for cliffordclock.fields.synthetic closed-form test fields.

Each field's hand-derived ``grad_fn`` is cross-checked against
``jax.jacfwd`` of its own ``e_fn`` at several points: an independent
sanity check on the hand derivation itself (catches sign/index-order
mistakes in the closed forms) that does not depend on
``smoother.FieldSmoother`` at all, so it stays meaningful as an oracle for
the smoother tests in ``test_fields_smoother.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.fields.synthetic import (
    constant_gradient_field,
    gaussian_bump_field,
    quadrupole_field,
    sample_on_grid,
    uniform_field,
)


def _assert_grad_matches_autodiff(
    e_fn: Callable[[jnp.ndarray], jnp.ndarray],
    grad_fn: Callable[[jnp.ndarray], jnp.ndarray],
    points: np.ndarray,
) -> None:
    """grad_fn(p) must equal jacfwd of a single-point wrapper around e_fn, for each p."""

    def e_single(p: jnp.ndarray) -> jnp.ndarray:
        return e_fn(p[None, :])[0]

    jac_fn = jax.jacfwd(e_single)  # jac[a, b] = d E_a / d x_b
    for p in points:
        jac = np.asarray(jac_fn(jnp.asarray(p)))
        expected_grad = jac.T  # grad[i, j] = d E_j / d x_i (E13)
        actual_grad = np.asarray(grad_fn(jnp.asarray(p)[None, :]))[0]
        np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-9, atol=1e-9)


@pytest.fixture
def sample_points() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(-1e-3, 1e-3, size=(20, 3))


def test_uniform_field_matches_autodiff(sample_points: np.ndarray) -> None:
    e0 = np.array([1.0, -2.0, 3.0])
    e_fn, grad_fn = uniform_field(e0)
    _assert_grad_matches_autodiff(e_fn, grad_fn, sample_points)

    values = np.asarray(e_fn(jnp.asarray(sample_points)))
    # rtol=0, atol=0: a uniform field is a pure broadcast of e0 -- exact.
    np.testing.assert_allclose(values, np.broadcast_to(e0, values.shape), rtol=0, atol=0)


def test_constant_gradient_field_matches_autodiff(sample_points: np.ndarray) -> None:
    e0 = np.array([10.0, -20.0, 30.0])
    grad = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    e_fn, grad_fn = constant_gradient_field(e0, grad)
    _assert_grad_matches_autodiff(e_fn, grad_fn, sample_points)

    grad_vals = np.asarray(grad_fn(jnp.asarray(sample_points)))
    # rtol=0, atol=0: the gradient of a constant-gradient field is a pure
    # broadcast of `grad` -- exact.
    np.testing.assert_allclose(grad_vals, np.broadcast_to(grad, grad_vals.shape), rtol=0, atol=0)


def test_quadrupole_field_matches_autodiff_and_formula(sample_points: np.ndarray) -> None:
    k = 1234.5
    e_fn, grad_fn = quadrupole_field(k)
    _assert_grad_matches_autodiff(e_fn, grad_fn, sample_points)

    values = np.asarray(e_fn(jnp.asarray(sample_points)))
    expected = k * np.stack(
        [sample_points[:, 0], sample_points[:, 1], -2.0 * sample_points[:, 2]], axis=-1
    )
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=0)

    grad_vals = np.asarray(grad_fn(jnp.asarray(sample_points)))
    expected_grad = np.diag([k, k, -2.0 * k])
    # rtol=0, atol=0: the quadrupole gradient is the constant diag(k, k, -2k)
    # (exact zeros off-diagonal), constructed rather than differentiated.
    np.testing.assert_allclose(
        grad_vals, np.broadcast_to(expected_grad, grad_vals.shape), rtol=0, atol=0
    )


def test_gaussian_bump_field_matches_autodiff(sample_points: np.ndarray) -> None:
    amp = np.array([5.0, 0.0, -3.0])
    center = np.array([0.1e-3, -0.2e-3, 0.05e-3])
    width = 0.3e-3
    e_fn, grad_fn = gaussian_bump_field(amp, center, width)
    _assert_grad_matches_autodiff(e_fn, grad_fn, sample_points)

    # Peak value at the center is exactly the amplitude: exp(0) == 1.0 and
    # amp * 1.0 are exact in IEEE double (including the amp[1] == 0.0
    # component, where a rel bound alone would be meaningless).
    peak = np.asarray(e_fn(jnp.asarray(center)[None, :]))[0]
    np.testing.assert_allclose(peak, amp, rtol=0, atol=0)


def test_sample_on_grid_shape_and_values() -> None:
    k = 100.0
    e_fn, _grad_fn = quadrupole_field(k)
    grid = sample_on_grid(e_fn, ((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)), 4)

    assert grid.regular is True
    assert grid.shape == (4, 4, 4)
    assert grid.points.shape == (64, 3)
    assert grid.values.shape == (64, 3)

    expected_values = np.asarray(e_fn(jnp.asarray(grid.points)))
    # rtol=0, atol=0: sample_on_grid must store exactly what e_fn returns
    # at the stored points.
    np.testing.assert_allclose(grid.values, expected_values, rtol=0, atol=0)
