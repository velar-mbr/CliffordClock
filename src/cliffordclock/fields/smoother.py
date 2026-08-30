# SPDX-License-Identifier: AGPL-3.0-or-later
"""C² field smoother: analytical baseline + thin-plate-spline RBF residual.

Implements the full decomposition of CONVENTIONS.md E11-E13:

- ``E_0`` (the degree-1 baseline, ``decompose.py``) captures the uniform
  and linear part of the field exactly.
- ``δE_smooth(r) = Σ_j γ_j φ(‖r − c_j‖)`` (E12), a thin-plate-spline RBF
  with kernel ``φ(r) = r² ln r``, captures the residual left after
  subtracting the baseline. Fitted per vector component, jointly by a
  linear solve, on the CSV grid points as RBF centers ``c_j``.
- ``FieldSmoother.evaluate`` is the sum of the two, ``E = E_0 + δE_smooth``.

Fit vs. evaluate split (binding): fitting the RBF coefficients (baseline
least squares, RBF linear solve) is done with plain NumPy; SciPy/NumPy
may be used freely there. Evaluation is a pure JAX function of (fitted
coefficients, centers, query position), so ``jax.jacfwd`` differentiates
*through the fit result* to get ``∇E``: this is what guarantees ``E`` and
``∇E`` are exactly consistent (the gradient is not a separately-fitted or
finite-differenced quantity) and C^∞ in the interpolant, and what lets the
rotor path integrator call ``evaluate`` under ``jit``/``vmap``/``grad``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]

from cliffordclock.fields.decompose import Baseline, fit_baseline, residual
from cliffordclock.fields.io import FieldGrid, check_near_duplicate_points

#: MVP cap on the number of RBF centers (= fit points). The fit solves a
#: dense (N, N) linear system, an O(N^3) operation; beyond a few times
#: this size that becomes impractical on a single machine. Larger inputs
#: need either subsampling by the caller or the tensor B-spline method
#: documented as future work (not implemented — see WP2 non-goals).
MAX_FIT_POINTS = 20_000

#: Default query-chunk size for :func:`chunked_apply`/:meth:`FieldSmoother.evaluate_chunked`
#: (WP19, streaming/chunked evaluation). ``FieldSmoother.evaluate``'s peak
#: intermediate is ``O(N * K)`` (`N` query points, `K` RBF fit centers,
#: :func:`FieldSmoother._field_at_point`'s ``diffs``/``phi`` arrays,
#: further multiplied by `jax.jacfwd`'s forward-mode tangent bookkeeping
#: for `grad_E`) -- unbounded in `N` for a single call. Chunking bounds
#: peak memory to ``chunk_size * K * 3 * 8 * factor`` bytes, independent
#: of `N`. 4096 is a round number comfortably larger than every ensemble
#: size (`M`) this project ships (hundreds, occasionally low thousands),
#: so a single per-step streaming call (WP19,
#: ``cliffordclock.pipeline``'s streaming accumulators) almost always
#: resolves to exactly one chunk (no behavioral difference from the
#: unchunked path, just the loop-of-one overhead) while still bounding
#: the pathological case of an unusually large `N` in one call (e.g. a
#: batched whole-trajectory evaluation, or a future ensemble size far
#: beyond what this project has measured).
DEFAULT_CHUNK_SIZE = 4096

_SUPPORTED_METHODS = ("auto", "rbf")

#: Threshold on the RBF kernel system's estimated reciprocal condition
#: number (LAPACK ``dgecon``'s 1-norm rcond) below which :meth:`FieldSmoother.fit`
#: emits :class:`IllConditionedFitWarning`. ``rcond`` is roughly
#: ``1 / cond(system)``; fp64 has ~15-16 decimal digits, so a system with
#: ``cond > ~1e12`` (``rcond < 1e-12``) has already lost most of that
#: precision in the solve.
_ILL_CONDITIONED_RCOND_THRESHOLD = 1e-12


class OutOfBoundsWarning(UserWarning):
    """Raised when :meth:`FieldSmoother.evaluate` is queried outside the fit bounding box.

    The RBF/baseline sum is defined everywhere, but it is unconstrained by
    data outside the convex hull of the fit points, so extrapolated values
    should not be trusted the way interior values are.
    """


class IllConditionedFitWarning(UserWarning):
    """Warned when :meth:`FieldSmoother.fit`'s RBF kernel matrix is severely ill-conditioned.

    Unlike an exactly singular matrix (which raises a LAPACK factorization
    warning that :meth:`FieldSmoother.fit` catches internally and handles
    by falling back to a least-squares solve), a merely ill-conditioned
    matrix factorizes and solves "successfully" while returning
    coefficients dominated by numerical noise -- silently, with no
    exception. This warning is ``FieldSmoother.fit``'s way of surfacing
    that failure mode. It is almost always caused by near-duplicate fit
    points (see :class:`~cliffordclock.fields.io.NearDuplicatePointsWarning`);
    the fix is to clean up the input points or increase ``smoothing``.
    """


def chunked_apply(fn: Any, *arrays: jnp.ndarray, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Any:
    """Apply `fn` over `arrays`' shared leading axis in fixed-size chunks (WP19).

    Bounds peak memory to whatever `fn` allocates for one `chunk_size`-row
    slice, independent of the arrays' full leading-axis length `N` --
    the fix for the memory-blowup failure mode documented in
    ``docs/timescales.md`` ("Safety net: the trajectory-memory guard",
    smoother-evaluation term): :meth:`FieldSmoother.evaluate` (and any
    ``rate_fn`` built on it, e.g.
    :func:`cliffordclock.pipeline._make_stark_rate_fn`) has an internal
    ``(N, K, 3)`` intermediate that is *not* bounded by `N` on its own.

    A plain Python loop over chunks (not `jax.lax.map`): each chunk is
    dispatched as its own XLA call, so only one chunk's intermediates are
    ever live at once -- `lax.map` by default still traces a single fused
    computation whose *un-fused* per-iteration buffers are not guaranteed
    to be freed between iterations the way separate dispatched calls are
    (and, more simply, the choice is builder's-choice per the WP19 plan;
    a Python loop is the simpler, more directly verifiable option and
    is what :func:`cliffordclock.pipeline`'s streaming accumulators call
    from inside an *already-scanned* per-step body, where `fn` is called
    with a small, static `N` = ensemble size in the common case anyway).

    Parameters
    ----------
    fn : Callable[..., Any]
        A function taking one or more arrays (all sharing the same
        leading-axis length `N`) and returning an array or pytree of
        arrays, each with leading axis `N` too (e.g.
        :meth:`FieldSmoother.evaluate`, returning ``(E, grad_E)``, or a
        ``rate_fn`` of signature ``(pos, v) -> delta_omega``).
    *arrays : jax.Array
        One or more arrays, each shape ``(N, ...)``, chunked identically
        (the same row range from every array is passed to `fn` together).
    chunk_size : int, default DEFAULT_CHUNK_SIZE
        Rows per chunk. Must be positive.

    Returns
    -------
    Any
        `fn`'s output pytree, with every leaf concatenated back along
        axis 0 to length `N` -- identical *shape* to calling
        ``fn(*arrays)`` directly, and (for `fn` = :meth:`FieldSmoother.evaluate`/
        :meth:`FieldSmoother._evaluate_2d`, or a `rate_fn` built on it)
        numerically identical **for `chunk_size >= 2`** too: each query
        point's evaluation is independent of every other query point (no
        reduction mixes rows), so which chunk of size >= 2 a row falls in
        cannot change its floating-point result. `chunk_size == 1` is a
        documented exception (<= 1 ulp, a `jax.vmap`-batch-of-1 XLA
        lowering detail, not a bug) -- see
        :meth:`FieldSmoother.evaluate_chunked`'s docstring and
        ``tests/test_fields_smoother.py::test_evaluate_chunked_matches_unchunked_exactly``.

    Raises
    ------
    ValueError
        `chunk_size` is not positive, or `arrays` is empty.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if not arrays:
        raise ValueError("chunked_apply requires at least one array argument")
    n = arrays[0].shape[0]
    if n <= chunk_size:
        # Common case (WP19 streaming accumulators call this once per step
        # with N = ensemble size, almost always <= chunk_size): no looping,
        # no behavioral difference from calling `fn` directly.
        return fn(*arrays)
    outputs = [
        fn(*(a[start : start + chunk_size] for a in arrays)) for start in range(0, n, chunk_size)
    ]
    return jax.tree_util.tree_map(lambda *leaves: jnp.concatenate(leaves, axis=0), *outputs)


def _tps_kernel_numpy(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Thin-plate-spline kernel ``φ(r) = r² ln r`` (E12), NumPy fit-side version.

    ``φ(0) := 0`` (the analytic limit); ``r_safe`` avoids ``log(0)``
    without changing the result, since the ``r**2`` factor is already zero
    there.
    """
    r_safe = np.where(r > 0, r, 1.0)
    return r**2 * np.log(r_safe)


def _tps_kernel_sq_jax(r2: jnp.ndarray) -> jnp.ndarray:
    """Thin-plate-spline kernel ``φ(r) = r² ln r`` (E12) from the *squared* distance ``r²``.

    JAX evaluate-side version, taking ``r²`` (not ``r``) so the removable
    singularity at a coincident center can be masked *before* the
    ``sqrt``. This matters for the gradient, not the value:
    ``d(sqrt)/d(r²)`` is infinite at ``r² = 0``, so the previous
    formulation (``r = sqrt(r²)`` in the caller, ``log(0)`` masked here)
    made ``jax.jacfwd`` produce an all-NaN gradient (``0 · ∞`` in the
    tangent chain) whenever a query point landed exactly on an RBF
    center, even though the primal ``φ`` value was finite.

    The double-``where`` guard handles the singularity in both the primal
    and the tangent: the inner ``where`` feeds ``sqrt`` (and ``log``) a
    harmless 1.0 on the coincident-center branch, the outer ``where``
    selects the analytic limit ``φ(0) = 0`` there. The coincident
    center's gradient contribution comes out exactly 0, which is the true
    limit: ``∇φ = (2 ln r + 1)(p − c) → 0`` as ``p → c``. For
    ``r² > 0`` the op sequence (sqrt, square, log, multiply) is unchanged
    from the pre-guard implementation, so off-node values are
    bitwise-identical to it (pinned by the byte-identical shipped-example
    snapshot tests in ``tests/test_bbr_pipeline.py``).
    """
    r2_safe = jnp.where(r2 > 0, r2, 1.0)
    r = jnp.sqrt(r2_safe)
    return jnp.where(r2 > 0, r**2 * jnp.log(r), 0.0)


@dataclass(frozen=True, eq=False)
class FieldSmoother:
    """C² field model: baseline (E11) plus thin-plate-spline RBF residual (E12).

    Construct with :meth:`fit`; do not call the dataclass constructor
    directly.

    Attributes
    ----------
    baseline : Baseline
        Fitted degree-1 baseline ``E_0``.
    centers : jax.Array, shape (K, 3)
        RBF centers ``c_j`` (the fit grid's points), meters.
    gamma : jax.Array, shape (K, 3)
        Fitted RBF coefficients, one 3-vector per center.
    method : str
        Resolved fit method (currently always ``"rbf"``).
    smoothing : float
        Tikhonov regularization used in the fit (0.0 = exact interpolation).
    bbox_min, bbox_max : NDArray[np.float64], shape (3,)
        Bounding box of the fit points, meters; used by :meth:`evaluate`
        to raise :class:`OutOfBoundsWarning` on extrapolation queries.
    """

    baseline: Baseline
    centers: jnp.ndarray
    gamma: jnp.ndarray
    method: str
    smoothing: float
    bbox_min: NDArray[np.float64]
    bbox_max: NDArray[np.float64]

    @classmethod
    def fit(cls, grid: FieldGrid, method: str = "auto", smoothing: float = 0.0) -> FieldSmoother:
        """Fit baseline + RBF smoother to a field grid.

        Parameters
        ----------
        grid : FieldGrid
            Ingested field data (``load_field_csv`` or ``synthetic.py``).
        method : str, default "auto"
            ``"rbf"`` (thin-plate-spline RBF on the baseline residual) or
            ``"auto"`` (currently always resolves to ``"rbf"``; tensor
            B-splines are documented future work, not implemented here).
        smoothing : float, default 0.0
            Tikhonov regularization added to the RBF linear system's
            diagonal. ``0.0`` fits the residual exactly at the data
            points; ``> 0`` trades fit accuracy for robustness to noisy
            input (see the noise-robustness test in
            ``tests/test_fields_smoother.py``).

        Returns
        -------
        FieldSmoother

        Raises
        ------
        ValueError
            Unknown ``method``, negative ``smoothing``, more fit points
            than :data:`MAX_FIT_POINTS`, or near-duplicate fit points with
            disagreeing values (see
            :func:`~cliffordclock.fields.io.check_near_duplicate_points`).

        Warns
        -----
        NearDuplicatePointsWarning
            Near-duplicate fit points were found (see
            :func:`~cliffordclock.fields.io.check_near_duplicate_points`).
        IllConditionedFitWarning
            The RBF kernel matrix's estimated reciprocal condition number
            is below :data:`_ILL_CONDITIONED_RCOND_THRESHOLD`.
        """
        if method not in _SUPPORTED_METHODS:
            raise ValueError(f"unknown method {method!r}; supported: {_SUPPORTED_METHODS}")
        if smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing}")

        n_points = grid.points.shape[0]
        if n_points > MAX_FIT_POINTS:
            raise ValueError(
                f"fit set has {n_points} points, exceeding the MVP cap of "
                f"{MAX_FIT_POINTS} (RBF fit solves a dense (N, N) linear system, "
                "O(N^3) cost); downsample the input or wait for a future "
                "out-of-core/tensor-B-spline method (not implemented in WP2)"
            )

        resolved_method = "rbf"  # "auto" -> "rbf"; see method docstring.

        baseline = fit_baseline(grid)
        delta_e = residual(grid, baseline)  # (N, 3), NumPy, fit-side

        centers = np.ascontiguousarray(grid.points)
        check_near_duplicate_points(centers, grid.values, context="FieldSmoother.fit")
        dists = cdist(centers, centers)  # (N, N)
        phi = _tps_kernel_numpy(dists)

        system = phi + smoothing * np.eye(n_points) if smoothing > 0 else phi

        # LU-factorize once and reuse the factorization both for the solve
        # and for a cheap O(N^2) 1-norm reciprocal-condition-number estimate
        # (LAPACK dgecon) -- this is *not* redundant with np.linalg.solve's
        # exact-singularity check: solve/lstsq are only "identical" (to
        # working precision) on well-conditioned full-rank systems. On a
        # near-singular but not exactly singular system (e.g. two fit points
        # a few nanometers apart on a millimeter-scale domain -- cond ~1e17,
        # no LinAlgError), solve silently returns coefficients with
        # magnitude many orders larger than the data, while lstsq's SVD-based
        # minimum-norm solution stays bounded. dgecon's rcond estimate is
        # what lets us warn about that regime instead of silently returning
        # garbage.
        with warnings.catch_warnings(record=True) as _caught_lu_warnings:
            warnings.simplefilter("always", scipy.linalg.LinAlgWarning)
            lu, piv = scipy.linalg.lu_factor(system)
        exactly_singular = any(
            issubclass(w.category, scipy.linalg.LinAlgWarning) for w in _caught_lu_warnings
        )
        if exactly_singular:
            # lu_factor warns (rather than raising) on an exact zero pivot;
            # that warning is our signal to fall back to the slower but
            # rank-deficiency-robust minimum-norm least-squares solve, same
            # as the previous np.linalg.solve -> LinAlgError -> lstsq path.
            gamma, _residuals, _rank, _sv = np.linalg.lstsq(system, delta_e, rcond=None)
        else:
            anorm = np.linalg.norm(system, ord=1)
            rcond_estimate, _gecon_info = scipy.linalg.lapack.dgecon(lu, anorm)
            if rcond_estimate < _ILL_CONDITIONED_RCOND_THRESHOLD:
                warnings.warn(
                    "FieldSmoother.fit: RBF kernel matrix is ill-conditioned "
                    f"(estimated rcond={rcond_estimate:.3e}, below the "
                    f"{_ILL_CONDITIONED_RCOND_THRESHOLD:.0e} threshold); fitted "
                    "coefficients may be dominated by numerical noise rather than "
                    "the actual field data. This is usually caused by near-duplicate "
                    "fit points -- clean up the input, or increase `smoothing`.",
                    IllConditionedFitWarning,
                    stacklevel=2,
                )
            gamma = scipy.linalg.lu_solve((lu, piv), delta_e)

        return cls(
            baseline=baseline,
            centers=jnp.asarray(centers),
            gamma=jnp.asarray(gamma),
            method=resolved_method,
            smoothing=float(smoothing),
            bbox_min=centers.min(axis=0),
            bbox_max=centers.max(axis=0),
        )

    def _field_at_point(self, p: jnp.ndarray) -> jnp.ndarray:
        """``E(p) = E_0(p) + δE_smooth(p)`` for a single position ``p``, shape (3,)."""
        diffs = p[None, :] - self.centers  # (K, 3)
        # Squared distance only -- no sqrt, so jacfwd stays finite when p
        # coincides with a center (see _tps_kernel_sq_jax).
        r2 = jnp.sum(diffs * diffs, axis=-1)  # (K,)
        phi = _tps_kernel_sq_jax(r2)  # (K,)
        delta_e = phi @ self.gamma  # (3,)
        e0 = self.baseline.offset + p @ self.baseline.grad  # (3,)
        return e0 + delta_e

    def _evaluate_2d(self, pos2d: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Shared ``(N, 3) -> (E, grad_E)`` core of :meth:`evaluate`/:meth:`evaluate_chunked`.

        Factored out (WP19) so :meth:`evaluate_chunked` can call this
        *exact* same per-chunk computation `evaluate` itself uses --
        guaranteeing the chunked and unchunked paths are numerically
        identical (see :meth:`evaluate_chunked`'s docstring), rather than
        a separately-written chunked evaluator that happens to compute
        the same thing.
        """
        e = jax.vmap(self._field_at_point)(pos2d)  # (N, 3)
        # jac[n, a, b] = d(E_a)/d(x_b); E13 wants grad_E[n, i, j] = d(E_j)/d(x_i),
        # i.e. the transpose of jacfwd's (output, input) axis order.
        jac = jax.vmap(jax.jacfwd(self._field_at_point))(pos2d)  # (N, 3, 3)
        grad_e = jnp.swapaxes(jac, 1, 2)
        return e, grad_e

    def evaluate(self, pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate the field and its gradient tensor at query positions.

        Pure JAX: works under ``jax.jit`` and ``jax.vmap``, and composes
        with ``jax.grad``/``jax.jacfwd`` of *callers* (WP3's rotor
        integrator differentiates through this).

        No internal bound on `N`: peak memory scales with `N * K` (`K`
        = number of fit centers, ``self.centers.shape[0]``) with no
        chunking -- the fast default for the common case where `N * K`
        is small. See :meth:`evaluate_chunked` for a memory-bounded
        equivalent when `N` may be large (WP19).

        Parameters
        ----------
        pos : jax.Array, shape (N, 3) or (3,)
            Query positions, meters.

        Returns
        -------
        E : jax.Array, shape (N, 3) or (3,)
            Field vectors, V/m.
        grad_E : jax.Array, shape (N, 3, 3) or (3, 3)
            ``grad_E[..., i, j] = ∂_i E_j`` (E13), V/m². Computed by
            ``jax.jacfwd`` of the same evaluator used for ``E``, so ``E``
            and ``grad_E`` are exactly consistent by construction.
        """
        pos_in = jnp.asarray(pos, dtype=jnp.float64)
        single = pos_in.ndim == 1
        pos2d = pos_in[None, :] if single else pos_in

        self._warn_if_outside_bounds(pos2d)

        e, grad_e = self._evaluate_2d(pos2d)

        if single:
            return e[0], grad_e[0]
        return e, grad_e

    def evaluate_chunked(
        self, pos: jnp.ndarray, chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Memory-bounded equivalent of :meth:`evaluate` (WP19).

        Identical contract (same signature convention, same return
        shapes) to :meth:`evaluate`, but evaluates `pos` in fixed-size
        row chunks via :func:`chunked_apply` instead of in one call --
        peak memory ``O(chunk_size * K)`` instead of ``O(N * K)``,
        independent of `N` (see ``docs/timescales.md``, "Safety net: the
        trajectory-memory guard").

        **Numerically identical to `evaluate` for `chunk_size >= 2`
        (bitwise, not just close); <= 1 ulp for `chunk_size == 1`.**
        Every chunk calls the exact same :meth:`_evaluate_2d` core
        `evaluate` itself calls, and each query point's ``(E, grad_E)``
        depends only on that point and the fixed fit data
        (`self.centers`/`self.gamma`) -- never on any other query point
        (no cross-row reduction in :meth:`_field_at_point`) -- so in
        principle which chunk a row lands in cannot change its
        floating-point result. Measured
        (``tests/test_fields_smoother.py::test_evaluate_chunked_matches_unchunked_exactly``):
        bitwise-exact for every `chunk_size >= 2` tried; `chunk_size == 1`
        measures a <= 1 ulp (`2.22e-16` relative) difference on `E`,
        traced to `jax.vmap`'s lowering of a batch of size exactly 1
        taking a measurably different (still correctly-rounded)
        instruction sequence on this backend than a batch of size >= 2 --
        a JAX/XLA implementation detail, not a bug in this function; see
        that test's docstring for the full measurement.

        Parameters
        ----------
        pos : jax.Array, shape (N, 3) or (3,)
            Query positions, meters.
        chunk_size : int, default DEFAULT_CHUNK_SIZE
            Rows evaluated per chunk. Must be positive.

        Returns
        -------
        E : jax.Array, shape (N, 3) or (3,)
            Field vectors, V/m.
        grad_E : jax.Array, shape (N, 3, 3) or (3, 3)
            Gradient tensor, V/m² (E13).
        """
        pos_in = jnp.asarray(pos, dtype=jnp.float64)
        single = pos_in.ndim == 1
        pos2d = pos_in[None, :] if single else pos_in

        self._warn_if_outside_bounds(pos2d)

        e, grad_e = chunked_apply(self._evaluate_2d, pos2d, chunk_size=chunk_size)

        if single:
            return e[0], grad_e[0]
        return e, grad_e

    def _warn_if_outside_bounds(self, pos2d: jnp.ndarray) -> None:
        """Best-effort :class:`OutOfBoundsWarning` on extrapolation queries.

        Only checkable on concrete (non-traced) positions: under
        ``jax.jit``/``jax.vmap`` tracing, ``pos2d`` is an abstract tracer
        with no concrete value to compare against the bounding box, so the
        check is silently skipped there (``evaluate`` must still work
        under those transforms per the WP2 test contract; it just loses
        this diagnostic while traced).
        """
        try:
            pos_np = np.asarray(pos2d)
        except Exception:  # noqa: BLE001 - deliberately broad: any tracer-conversion error
            return
        if np.any(pos_np < self.bbox_min) or np.any(pos_np > self.bbox_max):
            warnings.warn(
                "FieldSmoother.evaluate: query position(s) outside the fit data's "
                f"bounding box [{self.bbox_min.tolist()}, {self.bbox_max.tolist()}]; "
                "these are extrapolated, not interpolated, values.",
                OutOfBoundsWarning,
                stacklevel=3,
            )
