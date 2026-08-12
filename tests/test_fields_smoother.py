# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for cliffordclock.fields.smoother.FieldSmoother (WP2 core deliverable).

Covers WP2 test contract items 1 (round trip), 2 (grad/finite-difference
consistency), 4 (smoothness across former grid cells), 5 (noise
robustness), and 7 (jit/vmap compatibility, float64), plus the guard
rails from the WP2 spec (fit-size cap, out-of-bounds warning).
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.fields.io import FieldGrid, NearDuplicatePointsWarning
from cliffordclock.fields.smoother import (
    MAX_FIT_POINTS,
    FieldSmoother,
    IllConditionedFitWarning,
    OutOfBoundsWarning,
    chunked_apply,
)
from cliffordclock.fields.synthetic import (
    constant_gradient_field,
    gaussian_bump_field,
    quadrupole_field,
    sample_on_grid,
)

# Quadrupole strength for the round-trip fixture: chosen so gradients are
# O(1e6) V/m^2, a physically plausible-order scale for the absolute
# gradient tolerance (1e-6 * typical gradient magnitude) to be meaningful.
_QUADRUPOLE_K = 1.0e6


@pytest.fixture(scope="module")
def quadrupole_smoother() -> tuple[FieldSmoother, jnp.ndarray, jnp.ndarray]:
    """Fit contract item 1's fixture once and share it across tests 1, 2, 7."""
    e_fn, grad_fn = quadrupole_field(_QUADRUPOLE_K)
    grid = sample_on_grid(e_fn, ((-1e-3, 1e-3),) * 3, 21)
    smoother = FieldSmoother.fit(grid, method="auto", smoothing=0.0)
    return smoother, e_fn, grad_fn


def test_round_trip_quadrupole_field(
    quadrupole_smoother: tuple[FieldSmoother, jnp.ndarray, jnp.ndarray],
) -> None:
    """Item 1: fit a 21^3 grid, evaluate off-grid.

    Field rel err < 1e-8, grad abs err < 1e-6 * typical gradient magnitude.
    """
    smoother, e_fn, grad_fn = quadrupole_smoother

    rng = np.random.default_rng(0)
    test_pts = rng.uniform(-0.9e-3, 0.9e-3, size=(500, 3))
    test_pts_j = jnp.asarray(test_pts)

    e_pred, grad_pred = smoother.evaluate(test_pts_j)
    e_true = np.asarray(e_fn(test_pts_j))
    grad_true = np.asarray(grad_fn(test_pts_j))

    rel_field_err = np.max(
        np.abs(np.asarray(e_pred) - e_true) / (np.abs(e_true) + 1e-30 * np.max(np.abs(e_true)))
    )
    typical_grad = np.max(np.abs(grad_true))
    grad_abs_err = np.max(np.abs(np.asarray(grad_pred) - grad_true))

    # Measured (fixed seed=0, k=1e6, 21^3 grid, 500 pts): rel field err ~5.8e-15,
    # grad abs err ~3.5e-9 vs typical grad 2e6 (required bound 1e-6*2e6 = 2.0).
    assert rel_field_err < 1e-8
    assert grad_abs_err < 1e-6 * typical_grad


def test_grad_matches_central_finite_difference(
    quadrupole_smoother: tuple[FieldSmoother, jnp.ndarray, jnp.ndarray],
) -> None:
    """Item 2: grad_E equals central finite differences of evaluate's E output to 1e-6 relative."""
    smoother, _e_fn, _grad_fn = quadrupole_smoother

    rng = np.random.default_rng(1)
    test_pts = rng.uniform(-0.9e-3, 0.9e-3, size=(20, 3))
    h = 1e-7  # m, small vs. the 1e-3 m domain but not so small fp64 noise dominates

    def fd_gradient(p: np.ndarray) -> np.ndarray:
        rows = []
        for i in range(3):
            step = np.zeros(3)
            step[i] = h
            e_plus, _ = smoother.evaluate(jnp.asarray(p + step)[None, :])
            e_minus, _ = smoother.evaluate(jnp.asarray(p - step)[None, :])
            rows.append((np.asarray(e_plus)[0] - np.asarray(e_minus)[0]) / (2 * h))
        return np.stack(rows, axis=0)  # (3, 3): [i, j] = d E_j / d x_i

    max_rel_err = 0.0
    for p in test_pts:
        _e, grad_auto = smoother.evaluate(jnp.asarray(p)[None, :])
        grad_auto = np.asarray(grad_auto)[0]
        grad_fd = fd_gradient(p)
        rel_err = np.max(np.abs(grad_auto - grad_fd)) / np.max(np.abs(grad_fd))
        max_rel_err = max(max_rel_err, rel_err)

    # Measured (seed=1, 20 points, h=1e-7): max rel err ~3e-14.
    assert max_rel_err < 1e-6


def test_grad_finite_and_correct_at_exact_grid_node_query(
    quadrupole_smoother: tuple[FieldSmoother, jnp.ndarray, jnp.ndarray],
) -> None:
    """Regression guard: grad_E at a query point coinciding exactly with an RBF center.

    Found during WP21 (quadrupole shift, the first consumer of grad_E):
    the TPS kernel used to be evaluated as ``phi(sqrt(r2))``, and
    ``d(sqrt)/d(r2)`` is infinite at ``r2 = 0``, so ``jax.jacfwd``
    produced an all-NaN gradient tensor whenever a query point landed
    exactly on one of the fit grid's nodes (E itself stayed finite --
    the kernel's ``log(0)`` mask only guarded the primal value). The
    E14b DC-Stark path never reads grad_E, which is why this stayed
    invisible through WP2-WP20. The kernel now works in the squared
    distance directly (``phi = 0.5 r2 ln r2``, no sqrt) with a
    double-``where`` guard, making the coincident center's gradient
    contribution exactly 0 -- the analytic limit, since
    ``grad phi = (2 ln r + 1)(p - c) -> 0`` as ``p -> c``.

    Pins both finiteness and correctness (against the fixture's analytic
    gradient, same tolerance as the off-grid round-trip test) at exact
    fit-node queries, batched and single-point.
    """
    smoother, e_fn, grad_fn = quadrupole_smoother

    centers = np.asarray(smoother.centers)
    # A spread of exact node positions: corners, center of the grid, and
    # a few arbitrary interior nodes.
    idx = [0, centers.shape[0] // 2, centers.shape[0] - 1, 137, 4242]
    pts = jnp.asarray(centers[idx])

    e, grad = smoother.evaluate(pts)
    assert np.all(np.isfinite(np.asarray(e)))
    assert np.all(np.isfinite(np.asarray(grad)))

    e_true = np.asarray(e_fn(pts))
    grad_true = np.asarray(grad_fn(pts))
    assert np.max(np.abs(np.asarray(e) - e_true)) < 1e-8 * np.max(np.abs(e_true))
    assert np.max(np.abs(np.asarray(grad) - grad_true)) < 1e-6 * np.max(np.abs(grad_true))

    # Single-point (unbatched) path hits the same kernel; guard it too.
    _e1, grad1 = smoother.evaluate(jnp.asarray(centers[0]))
    assert np.all(np.isfinite(np.asarray(grad1)))


def test_grad_orientation_nonsymmetric_gradient() -> None:
    """FINDING 1 regression guard: grad_E's axis orientation, with a non-symmetric G.

    Every other round-trip fixture in this file (the quadrupole
    diag(k, k, -2k) and the constant-gradient diag(...) used in the
    smoothness/noise tests) has a *symmetric* true gradient tensor. That
    means a bug in the smoother.py:226-230 swapaxes -- which converts
    jax.jacfwd's (output, input) axis order to the E13 convention
    grad_E[i, j] = d_i E_j -- would go completely undetected there: for a
    symmetric G, G == G.T, so "grad_pred matches G" and "grad_pred matches
    G.T" are the same assertion. Use a genuinely non-symmetric G instead,
    and assert both that grad_pred matches G (correct orientation) and
    that it does *not* match G.T (the orientation a transposed-axes
    regression would produce), so a swapaxes regression is actually
    caught.
    """
    g_true = np.array(
        [
            [500.0, 12.0, -3.0],
            [7.0, -300.0, 4.0],
            [-2.0, 6.0, -200.0],
        ]
    )
    e0 = np.array([1.0e3, -2.0e3, 5.0e2])
    e_fn, grad_fn = constant_gradient_field(e0, g_true)
    grid = sample_on_grid(e_fn, ((-1e-3, 1e-3),) * 3, 9)
    smoother = FieldSmoother.fit(grid, smoothing=0.0)

    rng = np.random.default_rng(42)
    test_pts = rng.uniform(-0.9e-3, 0.9e-3, size=(50, 3))
    test_pts_j = jnp.asarray(test_pts)
    _e_pred, grad_pred = smoother.evaluate(test_pts_j)
    grad_pred = np.asarray(grad_pred)
    grad_true = np.asarray(grad_fn(test_pts_j))  # g_true broadcast to (50, 3, 3)

    typical_grad = np.max(np.abs(grad_true))
    grad_abs_err = np.max(np.abs(grad_pred - grad_true))
    assert grad_abs_err < 1e-6 * typical_grad

    # Orientation trap: g_true is deliberately non-symmetric (g_true[0, 1] =
    # 12 != 7 = g_true[1, 0], etc.), so g_true.T is a genuinely different
    # matrix. A transposed grad_E would satisfy neither the tight-tolerance
    # check above against a symmetric fixture nor be flagged by comparing
    # against G alone here -- this explicit "far from G.T" assertion is what
    # would actually fail if the swapaxes at smoother.py:226-230 regressed.
    grad_transposed_err = np.max(np.abs(grad_pred - np.swapaxes(grad_true, -1, -2)))
    assert grad_transposed_err > 1.0  # G vs G.T differ by O(10) V/m^2, way above fit noise


def test_smoothness_no_discontinuities_across_grid_cells() -> None:
    """Item 4: E and grad_E along a line crossing several former cells show no jumps.

    Uses a linear-plus-Gaussian-bump field (not the purely-linear
    quadrupole fixture) so the RBF residual is nontrivial and former grid
    cell boundaries are actually exercised.
    """
    e0 = np.array([1.0e4, -5.0e3, 2.0e3])
    grad = np.array([[500.0, 0.0, 0.0], [0.0, -300.0, 0.0], [0.0, 0.0, -200.0]])
    e_lin, _grad_lin = constant_gradient_field(e0, grad)
    amp = np.array([2.0e4, 0.0, -1.0e4])
    center = np.array([0.0, 0.0, 0.0])
    width = 3.0e-4
    e_bump, _grad_bump = gaussian_bump_field(amp, center, width)

    def e_fn(pos: jnp.ndarray) -> jnp.ndarray:
        return e_lin(pos) + e_bump(pos)

    grid = sample_on_grid(e_fn, ((-1e-3, 1e-3),) * 3, 15)  # dx ~1.43e-4 m
    smoother = FieldSmoother.fit(grid, smoothing=0.0)

    n_line = 10_000
    t = np.linspace(-0.9e-3, 0.9e-3, n_line)  # crosses ~13 former grid cells
    line_pts = np.stack([t, np.full_like(t, 0.05e-3), np.full_like(t, -0.07e-3)], axis=-1)
    dx = t[1] - t[0]

    e_line, grad_line = smoother.evaluate(jnp.asarray(line_pts))
    e_line = np.asarray(e_line)
    grad_line = np.asarray(grad_line)

    max_step = np.max(np.abs(np.diff(e_line, axis=0)))
    local_lipschitz = np.max(np.abs(grad_line))  # max |dE_j/dx_i| observed along the line
    safety_factor = 3.0  # margin above the exact mean-value-theorem bound

    # Measured (n=15 grid, dx~1.43e-4 m, line dx~1.8e-7 m): max_step / (lipschitz*dx) ~ 1.0000.
    assert max_step < safety_factor * local_lipschitz * dx


def test_noise_robustness_smoothing_degrades_gracefully() -> None:
    """Item 5: with 1e-6 relative noise, smoothing>0 keeps gradient error bounded and graceful.

    Uses a purely linear field (baseline captures it exactly; true
    residual is exactly zero) so any nonzero fitted RBF coefficient is
    attributable entirely to fitting the injected noise -- an isolated,
    controlled measurement of noise sensitivity.
    """
    e0 = np.array([1.0e4, -5.0e3, 2.0e3])
    grad = np.array([[500.0, 0.0, 0.0], [0.0, -300.0, 0.0], [0.0, 0.0, -200.0]])
    e_fn, grad_fn = constant_gradient_field(e0, grad)
    grid = sample_on_grid(e_fn, ((-1e-3, 1e-3),) * 3, 15)

    rng = np.random.default_rng(2)
    test_pts = rng.uniform(-0.8e-3, 0.8e-3, size=(300, 3))
    test_pts_j = jnp.asarray(test_pts)
    grad_true = np.asarray(grad_fn(test_pts_j))
    typical_grad = np.max(np.abs(grad_true))  # 500 V/m^2

    noise_scale = 1e-6 * np.mean(np.abs(grid.values))
    noise = rng.normal(scale=noise_scale, size=grid.values.shape)
    noisy_grid = replace(grid, values=grid.values + noise)

    smoother_unregularized = FieldSmoother.fit(noisy_grid, smoothing=0.0)
    _, grad_unreg = smoother_unregularized.evaluate(test_pts_j)
    err_unregularized = np.max(np.abs(np.asarray(grad_unreg) - grad_true))

    smoothing_value = 1e-3
    smoother_regularized = FieldSmoother.fit(noisy_grid, smoothing=smoothing_value)
    _, grad_reg = smoother_regularized.evaluate(test_pts_j)
    err_regularized = np.max(np.abs(np.asarray(grad_reg) - grad_true))

    # Measured (seed=2, 15^3 grid, 1e-6 relative noise, smoothing=1e-3):
    # unregularized grad err ~192 V/m^2 (~38% of typical_grad -- exact
    # interpolation of noise devastates the gradient); regularized grad
    # err ~0.55 V/m^2 (~0.11% of typical_grad), a ~350x improvement.
    assert err_regularized < err_unregularized
    assert err_regularized < 1e-2 * typical_grad  # graceful: well under 1% of the signal


def test_evaluate_jit_and_vmap_return_float64() -> None:
    """Item 7: evaluate works under jax.jit and jax.vmap and returns float64."""
    e_fn, _grad_fn = quadrupole_field(1.0e3)
    grid = sample_on_grid(e_fn, ((-1.0, 1.0),) * 3, 5)
    smoother = FieldSmoother.fit(grid, smoothing=0.0)

    pos = jnp.asarray(np.random.default_rng(3).uniform(-0.9, 0.9, size=(10, 3)))

    e_jit, grad_jit = jax.jit(smoother.evaluate)(pos)
    assert e_jit.dtype == jnp.float64
    assert grad_jit.dtype == jnp.float64
    e_plain, grad_plain = smoother.evaluate(pos)
    np.testing.assert_allclose(np.asarray(e_jit), np.asarray(e_plain), rtol=1e-12, atol=0)
    # jit fusion reorders the gradient reduction, so entries that pass
    # through ~0 (the quadrupole's off-diagonals) differ by ~1e-27 in
    # absolute terms; bound that at 1e-12 of the gradient scale instead of
    # a meaningless rel on near-zero entries.
    grad_scale = float(np.max(np.abs(np.asarray(grad_plain))))
    np.testing.assert_allclose(
        np.asarray(grad_jit), np.asarray(grad_plain), rtol=1e-12, atol=1e-12 * grad_scale
    )

    def single_point_field(p: jnp.ndarray) -> jnp.ndarray:
        e, _grad = smoother.evaluate(p)
        return e

    e_vmapped = jax.vmap(single_point_field)(pos)
    assert e_vmapped.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(e_vmapped), np.asarray(e_plain), rtol=1e-12, atol=0)


def test_single_point_input_returns_unbatched_shapes() -> None:
    """evaluate accepts a single (3,) position and returns (3,)/(3,3), not batched."""
    e_fn, _grad_fn = quadrupole_field(1.0)
    grid = sample_on_grid(e_fn, ((-1.0, 1.0),) * 3, 5)
    smoother = FieldSmoother.fit(grid, smoothing=0.0)

    e, grad_e = smoother.evaluate(jnp.array([0.1, -0.2, 0.3]))
    assert e.shape == (3,)
    assert grad_e.shape == (3, 3)


def test_fit_rejects_unknown_method() -> None:
    e_fn, _grad_fn = quadrupole_field(1.0)
    grid = sample_on_grid(e_fn, ((-1.0, 1.0),) * 3, 3)
    with pytest.raises(ValueError, match="unknown method"):
        FieldSmoother.fit(grid, method="tensor-bspline")


def test_fit_rejects_negative_smoothing() -> None:
    e_fn, _grad_fn = quadrupole_field(1.0)
    grid = sample_on_grid(e_fn, ((-1.0, 1.0),) * 3, 3)
    with pytest.raises(ValueError, match="smoothing"):
        FieldSmoother.fit(grid, smoothing=-1.0)


def test_fit_rejects_oversized_point_set() -> None:
    """Guard rail: fit size is capped (~20k points, O(N^3) fit cost documented)."""
    n = MAX_FIT_POINTS + 1
    dummy_points = np.zeros((n, 3), dtype=np.float64)
    dummy_values = np.zeros((n, 3), dtype=np.float64)
    oversized_grid = FieldGrid(
        points=dummy_points,
        values=dummy_values,
        regular=False,
        axes=None,
        shape=None,
        metadata={},
    )
    with pytest.raises(ValueError, match=str(MAX_FIT_POINTS)):
        FieldSmoother.fit(oversized_grid)


def test_fit_warns_on_near_duplicate_points_and_ill_conditioning() -> None:
    """FINDING 2 (a, b): reproduces the reviewer's ill-conditioning scenario.

    A point ~1e-12 m from an existing fit point (far below the
    1e-9-times-domain-diagonal near-duplicate threshold, on this ~3.5e-3 m
    domain) makes the RBF kernel matrix's condition number blow up
    (measured ~1e17 for this fixture, matching the reviewer's report of
    cond(K)~4e16 for a similar near-duplicate setup). Both structured
    warnings fire: NearDuplicatePointsWarning (io.py's shared point-cloud
    check) and IllConditionedFitWarning (smoother.py's new dgecon-based
    rcond check) -- previously neither existed and np.linalg.solve
    silently returned huge garbage coefficients.
    """
    e_fn, _grad_fn = quadrupole_field(1.0e6)
    grid = sample_on_grid(e_fn, ((-1e-3, 1e-3),) * 3, 6)
    points = np.array(grid.points, copy=True)
    values = np.array(grid.values, copy=True)

    near_dup_point = points[0].copy()
    near_dup_point[0] += 1e-12  # nanometer-scale offset on a millimeter-scale domain
    points = np.vstack([points, near_dup_point[None, :]])
    values = np.vstack([values, values[0][None, :]])  # same value: field varies smoothly here

    dup_grid = FieldGrid(
        points=points, values=values, regular=False, axes=None, shape=None, metadata={}
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FieldSmoother.fit(dup_grid, smoothing=0.0)

    categories = {w.category for w in caught}
    assert NearDuplicatePointsWarning in categories
    assert IllConditionedFitWarning in categories


def test_fit_no_ill_conditioning_warnings_on_clean_grid() -> None:
    """FINDING 2: a normally-spaced fit grid triggers neither new warning.

    Also covers finding 2(c)'s "quadrupole round-trip fixture still passes
    with the new solve path" requirement together with
    ``test_round_trip_quadrupole_field`` above (same fixture, same
    tolerances, now exercising ``scipy.linalg.lu_factor``/``lu_solve``
    instead of ``np.linalg.solve``).
    """
    e_fn, _grad_fn = quadrupole_field(1.0e6)
    grid = sample_on_grid(e_fn, ((-1e-3, 1e-3),) * 3, 9)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NearDuplicatePointsWarning)
        warnings.simplefilter("error", IllConditionedFitWarning)
        FieldSmoother.fit(grid, smoothing=0.0)


def test_evaluate_outside_bounding_box_warns() -> None:
    """Guard rail: evaluating outside the fit bounding box raises OutOfBoundsWarning."""
    e_fn, _grad_fn = quadrupole_field(1.0)
    grid = sample_on_grid(e_fn, ((-1.0, 1.0),) * 3, 5)
    smoother = FieldSmoother.fit(grid, smoothing=0.0)

    with pytest.warns(OutOfBoundsWarning):
        smoother.evaluate(jnp.array([[10.0, 10.0, 10.0]]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        smoother.evaluate(jnp.array([[0.1, 0.1, 0.1]]))  # inside bbox: no warning


# ---------------------------------------------------------------------------
# WP19: chunked evaluation (cliffordclock.fields.smoother.chunked_apply /
# FieldSmoother.evaluate_chunked). The trajectory-memory guard's
# smoother-evaluation term (cliffordclock.pipeline
# ._TRAJECTORY_MEMORY_FACTOR_SMOOTHER) exists because FieldSmoother.evaluate's
# (N, K, 3) intermediate is unbounded in N; chunking is the actual memory
# fix (the guard remains an advisory pre-flight estimate/selector, WP19).
# ---------------------------------------------------------------------------


def test_evaluate_chunked_matches_unchunked_exactly(
    quadrupole_smoother: tuple[FieldSmoother, jnp.ndarray, jnp.ndarray],
) -> None:
    """`evaluate_chunked` matches `evaluate` bitwise for chunk_size >= 2,
    and to <= 1 ulp for chunk_size == 1.

    Each query point's ``(E, grad_E)`` depends only on that point and the
    fixed fit data (`FieldSmoother._field_at_point` has no cross-row
    reduction -- every point's RBF sum is independent of every other
    query point), and `evaluate_chunked` calls the exact same
    `_evaluate_2d` core `evaluate` itself calls, on a sub-array, so in
    principle no floating-point operation's *order* changes with chunk
    boundaries. Measured: `chunk_size >= 2` is bitwise exact
    (`np.array_equal`) for every size tried, including a chunk_size at
    least the full query-point count (the `chunked_apply` fast-path
    branch, no looping at all). `chunk_size == 1` measures a <= 1 ulp
    (`2.22e-16` relative, exactly `float64`'s machine epsilon) difference
    on `E` for this fixture -- not a bug in this project's code: `E` is
    computed via `jax.vmap(self._field_at_point)`, and XLA's lowering of
    a `vmap` batch of size exactly 1 takes a measurably different (still
    correctly-rounded) instruction sequence than a batch of size >= 2 on
    this backend, a documented JAX/XLA implementation detail outside this
    project's control -- not something `chunked_apply`/`evaluate_chunked`
    can avoid short of special-casing batch-of-1 to bypass `vmap`
    entirely, which is not worth the complexity for a <=1-ulp effect.
    `grad_E` (via `jax.jacfwd`) measured exactly 0 diff even at
    chunk_size=1 for this fixture.
    """
    smoother, _e_fn, _grad_fn = quadrupole_smoother
    rng = np.random.default_rng(1)
    query = jnp.asarray(rng.uniform(-0.9e-3, 0.9e-3, size=(777, 3)))

    e_ref, grad_ref = smoother.evaluate(query)

    for chunk_size in (2, 7, 100, 1000, 10_000):
        e_chunked, grad_chunked = smoother.evaluate_chunked(query, chunk_size=chunk_size)
        assert np.array_equal(np.asarray(e_ref), np.asarray(e_chunked)), chunk_size
        assert np.array_equal(np.asarray(grad_ref), np.asarray(grad_chunked)), chunk_size

    # chunk_size=1: <= 1 ulp, not bitwise (see docstring) -- atol=0, a tight
    # rtol a few ulps above the measured 2.22e-16 (this project's tolerance
    # doctrine: assert_allclose(atol=0) for a relative bound).
    e_chunk1, grad_chunk1 = smoother.evaluate_chunked(query, chunk_size=1)
    np.testing.assert_allclose(np.asarray(e_ref), np.asarray(e_chunk1), rtol=4e-16, atol=0)
    np.testing.assert_allclose(np.asarray(grad_ref), np.asarray(grad_chunk1), rtol=4e-16, atol=0)

    single_pos = query[3]
    e_ref_single, grad_ref_single = smoother.evaluate(single_pos)
    e_chunked_single, grad_chunked_single = smoother.evaluate_chunked(single_pos, chunk_size=10)
    assert np.array_equal(np.asarray(e_ref_single), np.asarray(e_chunked_single))
    assert np.array_equal(np.asarray(grad_ref_single), np.asarray(grad_chunked_single))


def test_chunked_apply_matches_direct_call_for_multi_arg_fn() -> None:
    """`chunked_apply` generalizes to functions taking multiple co-indexed
    arrays (the ``rate_fn(pos, v) -> delta_omega`` signature
    `cliffordclock.pipeline`'s streaming accumulators chunk, not just
    `FieldSmoother.evaluate`'s single-array ``pos -> (E, grad_E)``).
    """
    rng = np.random.default_rng(2)
    pos = jnp.asarray(rng.uniform(-1, 1, size=(53, 3)))
    v = jnp.asarray(rng.uniform(-1, 1, size=(53, 3)))

    def rate_fn(p: jnp.ndarray, vel: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(p * p, axis=-1) + jnp.sum(vel, axis=-1)

    direct = rate_fn(pos, v)
    chunked = chunked_apply(rate_fn, pos, v, chunk_size=8)
    assert np.array_equal(np.asarray(direct), np.asarray(chunked))


def test_chunked_apply_rejects_bad_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        chunked_apply(lambda x: x, jnp.zeros((5, 3)), chunk_size=0)


def test_chunked_apply_rejects_no_arrays() -> None:
    with pytest.raises(ValueError, match="at least one array"):
        chunked_apply(lambda: None, chunk_size=10)
