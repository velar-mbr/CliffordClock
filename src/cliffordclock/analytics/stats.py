# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ensemble metrology statistics (CONVENTIONS.md E22-E23, E25-E28; WP5 scope item 1).

Consumes per-atom accumulated perturbation phases ``ΔΦ_i`` (E22,
dimensionless, Compton-scaled) produced by
:mod:`cliffordclock.integrator.worldline` (WP3, ``WorldlineResult.phase`` /
``EnsembleResult.phase``), with optional per-atom weights (uniform for
classical Monte-Carlo ensembles, quadrature weights for lattice motional
nodes, WP4), and turns them into the numbers metrologists consume: the
weighted mean fractional shift, its standard error, the inhomogeneous
dephasing time T2*, the ensemble coherence function, and the spectral line
profile.

jax/numpy boundary (WP5 orchestrator instruction 7): callers may pass
``jax.Array`` phases straight from WP3's result types. This module converts
every input to ``numpy`` float64/complex128 explicitly at each function's
entry point and computes purely in ``numpy`` from there -- these are
terminal summary statistics (means, variances, an FFT), not quantities any
downstream code needs to differentiate through, so there is no
differentiability requirement to preserve by staying in ``jax``, and
``numpy``'s scalar reductions (plus ``math.fsum`` below) give the more
directly auditable precision guarantees this module's numerics discipline
depends on.

Numerical-precision discipline (E10, CONVENTIONS.md section 3; WP5
orchestrator instruction 5, "doctrine after two shipped bugs"): phase
values are small numbers (typically far below 1) multiplied by large
ensemble counts in the reductions below. Every mean/variance in this module
is computed by **two-pass** summation (mean first, then a separate pass
over deviations from that mean) using ``math.fsum`` (exact, correctly
rounded summation of the input floats -- Shewchuk's algorithm) for each
pass. Nothing here ever computes ``E[x**2] - E[x]**2``: that one-pass
form cancels catastrophically once the ensemble's phase *spread* is many
orders of magnitude below its *mean* (WP5 test contract item 3;
``tests/test_analytics_stats.py::test_cancellation_guard_naive_would_fail``
demonstrates the failure mode directly and measures the margin).

Future work (WP5 non-goals): Allan deviation / overlapping Allan variance
is deliberately not implemented in Sprint 1 -- the statistics here
characterize a *single* interrogation's ensemble spread, not shot-to-shot
stability over a measurement campaign. It is the natural next statistic
when multi-shot simulation lands (post-MVP).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cliffordclock.constants import TAU_COMPTON

__all__ = [
    "WeightedPhaseStats",
    "weighted_phase_stats",
    "mean_fractional_shift",
    "shift_std_error",
    "dephasing_time_t2star",
    "coherence_function",
    "line_profile",
]


def _as_phi(phi: ArrayLike) -> NDArray[np.float64]:
    """Coerce `phi` to a 1-D float64 numpy array, validating shape."""
    x = np.asarray(phi, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"phi must be 1-D, shape (M,); got shape {x.shape}")
    if x.shape[0] < 1:
        raise ValueError("phi must contain at least one atom")
    return x


def _normalized_weights(weights: ArrayLike | None, m: int) -> NDArray[np.float64]:
    """Return normalized (sum-to-1) float64 weights, shape ``(m,)``.

    Uniform ``1/m`` when `weights` is ``None`` (classical Monte-Carlo
    convention, E23). Non-uniform `weights` (e.g. lattice quadrature
    weights, WP4) are renormalized defensively so callers do not need to
    guarantee exact-sum-to-1 inputs.
    """
    if weights is None:
        return np.full(m, 1.0 / m, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (m,):
        raise ValueError(f"weights must have shape ({m},), matching phi; got {w.shape}")
    total = math.fsum(w.tolist())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return w / total


def _weighted_mean(x: NDArray[np.float64], w: NDArray[np.float64]) -> float:
    """Weighted mean ``Σ w_i x_i``, exact (correctly-rounded) summation."""
    return math.fsum((w * x).tolist())


class WeightedPhaseStats(NamedTuple):
    """Weighted mean and variance of an ensemble's accumulated phase (E25).

    Attributes
    ----------
    mean : float
        Weighted mean ``ΔΦ̄ = Σ w_i ΔΦ_i``, dimensionless.
    variance : float
        Weighted, bias-corrected sample variance ``σ_Φ²`` (E25's weighted
        generalization for quadrature-node weights; the "reliability
        weights" estimator ``Σw_i(ΔΦ_i-ΔΦ̄)² / (1 - Σw_i²)``, which reduces
        exactly to E25's stated unweighted form
        ``Σ(ΔΦ_i-ΔΦ̄)²/(M-1)`` when ``w_i = 1/M``. **[INTERPRETATION]**:
        CONVENTIONS.md E25 states the unweighted formula and says only
        "weighted generalization" without pinning the exact weighted
        estimator; this is the standard reliability-weights unbiased
        estimator.
    sum_weights_sq : float
        ``Σ w_i²`` -- the weighted-ensemble analogue of ``1/M``, i.e. the
        reciprocal effective sample size; used by `shift_std_error`.
    """

    mean: float
    variance: float
    sum_weights_sq: float


def weighted_phase_stats(phi: ArrayLike, weights: ArrayLike | None = None) -> WeightedPhaseStats:
    """Weighted mean and variance of accumulated phases ``ΔΦ_i`` (E25).

    Two-pass: the mean is computed first, then the variance from
    deviations against that fixed mean -- never ``E[x²] - E[x]²`` (see
    module docstring). Both passes use ``math.fsum`` (exact summation of
    the input floats), so accuracy is limited only by each individual
    floating-point term's own representation error, not by summation
    order or cancellation in a final subtraction.

    Parameters
    ----------
    phi : array_like, shape (M,)
        Accumulated perturbation phases ``ΔΦ_i`` (E22), dimensionless.
        May be a `jax.Array` (converted to numpy float64 internally).
    weights : array_like, shape (M,), optional
        Ensemble weights (E23); uniform ``1/M`` if omitted. Renormalized
        to sum to 1 internally.

    Returns
    -------
    WeightedPhaseStats

    Raises
    ------
    ValueError
        If `phi` has fewer than 2 elements, or the effective sample size
        implied by `weights` is below 2 (``Σw_i² >= 1``): the variance
        estimator is undefined in that regime.
    """
    x = _as_phi(phi)
    m = x.shape[0]
    w = _normalized_weights(weights, m)

    mean = _weighted_mean(x, w)
    sum_w_sq = math.fsum((w * w).tolist())
    denom = 1.0 - sum_w_sq
    if m < 2 or denom <= 0.0:
        raise ValueError(
            "weighted variance is undefined for fewer than ~2 effective atoms "
            f"(M={m}, sum(w^2)={sum_w_sq:.6e})"
        )
    sq_dev = w * (x - mean) ** 2
    s = math.fsum(sq_dev.tolist())
    variance = s / denom
    return WeightedPhaseStats(mean=mean, variance=variance, sum_weights_sq=sum_w_sq)


def mean_fractional_shift(
    phi: ArrayLike,
    t_interrogation_s: float,
    weights: ArrayLike | None = None,
) -> float:
    """Weighted ensemble mean fractional frequency shift ``⟨Δν/ν₀⟩`` (E23).

    ``⟨Δν/ν₀⟩ = Σ_i w_i (ΔΦ_i / T̃)`` with ``T̃ = T_interrogation / τ_c``
    (E9). Note ``ν₀`` itself does not appear in this formula -- E23's
    fractional-shift form is already normalized by the clock frequency via
    the phase-accumulation pipeline (E21-E22) -- which is why
    species/``ν₀`` is not a parameter here.

    Parameters
    ----------
    phi : array_like, shape (M,)
        Accumulated perturbation phases ``ΔΦ_i`` (E22), dimensionless.
    t_interrogation_s : float
        Interrogation time ``T``, seconds (SI at the API boundary,
        CONVENTIONS.md section 10; converted to Compton units ``T̃``
        internally via E9).
    weights : array_like, shape (M,), optional
        Ensemble weights; uniform ``1/M`` if omitted.

    Returns
    -------
    float
        ``⟨Δν/ν₀⟩``, dimensionless.
    """
    x = _as_phi(phi)
    w = _normalized_weights(weights, x.shape[0])
    if t_interrogation_s <= 0.0:
        raise ValueError("t_interrogation_s must be positive")
    mean_phi = _weighted_mean(x, w)
    t_tilde = t_interrogation_s / TAU_COMPTON
    return mean_phi / t_tilde


def shift_std_error(
    phi: ArrayLike,
    t_interrogation_s: float,
    weights: ArrayLike | None = None,
) -> float:
    """Weighted standard error of the mean fractional shift (E23 & E25).

    ``SEM(Δν/ν₀) = sqrt(σ_Φ² · Σw_i²) / T̃`` -- the standard
    delta-method standard error of a weighted mean under a common-variance
    assumption (``Var(Σw_i x_i) = σ² Σw_i²``), scaled from phase to
    fractional-shift units by E23's linear ``ΔΦ_i -> ΔΦ_i/T̃`` map.
    **[INTERPRETATION]**: CONVENTIONS.md defines the *point estimate*
    (E23) and the *variance* (E25) but not a weighted standard-error
    formula explicitly; this is the standard weighted-mean SEM (reduces to
    the familiar ``σ/sqrt(M)`` for uniform weights).

    Parameters
    ----------
    phi : array_like, shape (M,)
        Accumulated perturbation phases ``ΔΦ_i`` (E22), dimensionless.
    t_interrogation_s : float
        Interrogation time ``T``, seconds.
    weights : array_like, shape (M,), optional
        Ensemble weights; uniform ``1/M`` if omitted.

    Returns
    -------
    float
        Standard error of ``⟨Δν/ν₀⟩``, dimensionless.
    """
    if t_interrogation_s <= 0.0:
        raise ValueError("t_interrogation_s must be positive")
    stats = weighted_phase_stats(phi, weights)
    sem_phi = math.sqrt(stats.variance * stats.sum_weights_sq)
    t_tilde = t_interrogation_s / TAU_COMPTON
    return sem_phi / t_tilde


def dephasing_time_t2star(
    phi: ArrayLike,
    t_interrogation_s: float,
    weights: ArrayLike | None = None,
) -> float:
    """Inhomogeneous dephasing time ``T2*`` (E27).

    ``T2* = sqrt(2) · T_interrogation / σ_Φ``, seconds (G0 item 4,
    CONVENTIONS.md section 8: standard inhomogeneous dephasing, Gaussian
    frequency-offset spread). ``σ_Φ`` is the weighted phase standard
    deviation from `weighted_phase_stats` (E25).

    ``σ_Φ = 0`` (every atom accumulated the *identical* phase -- e.g. a
    lattice ensemble in a spatially uniform field) returns ``math.inf``:
    E27's limit as ``σ_Φ -> 0+`` is ``+inf``, and it is the physically
    correct answer -- zero inhomogeneous spread means no inhomogeneous
    dephasing, so the statistic is *defined* here (unlike the M=1 case,
    where E25's variance estimator itself does not exist and
    `weighted_phase_stats` raises). Only an *exactly* zero float64
    variance takes this branch -- no "numerically negligible" tolerance
    is applied, per this project's tolerance discipline: a tiny nonzero
    ``σ_Φ`` yields the huge-but-finite T2* E27 dictates. (Pedantic
    boundary note: "exactly zero" means the computed float64 variance;
    phase deviations below ~1.5e-162 underflow to zero when squared, so
    formally-distinct phases at that magnitude would also land here --
    ~140 orders of magnitude below any physical ΔΦ this pipeline
    produces, documented for completeness only.) Report serialization
    writes the resulting non-finite value as JSON ``null`` (see
    `cliffordclock.analytics.report.write_json` and
    ``docs/report-schema.md``).

    Parameters
    ----------
    phi : array_like, shape (M,)
        Accumulated perturbation phases ``ΔΦ_i`` (E22), dimensionless.
    t_interrogation_s : float
        Interrogation time ``T``, seconds.
    weights : array_like, shape (M,), optional
        Ensemble weights; uniform ``1/M`` if omitted.

    Returns
    -------
    float
        ``T2*``, seconds; ``math.inf`` when the ensemble phase variance
        is exactly zero.
    """
    if t_interrogation_s <= 0.0:
        raise ValueError("t_interrogation_s must be positive")
    stats = weighted_phase_stats(phi, weights)
    sigma_phi = math.sqrt(stats.variance)
    if sigma_phi == 0.0:
        return math.inf
    return math.sqrt(2.0) * t_interrogation_s / sigma_phi


def coherence_function(
    phi_final: ArrayLike,
    t_interrogation_s: float,
    t_grid_s: ArrayLike,
    weights: ArrayLike | None = None,
    phase_history: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Ensemble coherence function ``C(t) = ⟨exp(i ΔΦ_i(t))⟩`` (E26).

    Two input modes:

    - **Exact** (`phase_history` given): `phase_history[i, k]` is the
      running accumulated phase of atom `i` at grid time `t_grid_s[k]`
      (the same quantity as E22's integrand, partially integrated to
      ``t_grid_s[k]``). Exact for any time-dependent per-atom rate
      ``δω̃_i(t)``.
    - **Reconstructed** (`phase_history` omitted): `phi_final` (the E22
      *final* phase) is linearly interpolated in time,
      ``ΔΦ_i(t) = ΔΦ_i · (t / T_interrogation)``. This is **exact** only
      when each atom's instantaneous rate ``δω̃_i(t)`` is constant over
      the interrogation window (E22 integrand time-independent); for a
      time-varying rate (e.g. a classical atom oscillating through a
      field gradient) it captures the correct *mean* rate and hence the
      correct T2*/mean-shift-consistent endpoint, but not the detailed
      time-domain shape of ``C(t)`` (no curvature/modulation from the
      time dependence) -- document this when using the reconstructed path
      for line-shape work.

    .. note::
       WP3's committed `EnsembleResult`
       (:mod:`cliffordclock.integrator.worldline`) exposes only the
       *final* scalar `phase` per atom, not a per-step history array (its
       ``lax.scan`` discards per-step outputs) -- so in the current
       pipeline, `phase_history` is always ``None`` in practice; the
       parameter exists for callers that reconstruct a history themselves
       or for a future WP3 revision that exposes one.

    Parameters
    ----------
    phi_final : array_like, shape (M,)
        Final accumulated perturbation phases ``ΔΦ_i`` (E22),
        dimensionless. Used for the reconstructed path; also validated
        for ensemble size ``M`` against `weights` and `phase_history`
        even when `phase_history` is supplied.
    t_interrogation_s : float
        Interrogation time ``T``, seconds. Used for the reconstructed
        path's linear time map; unused (but still validated for
        ``M``-consistency via `phi_final`) when `phase_history` is given.
    t_grid_s : array_like, shape (T,)
        Time grid, seconds, at which to evaluate ``C(t)``.
    weights : array_like, shape (M,), optional
        Ensemble weights; uniform ``1/M`` if omitted.
    phase_history : array_like, shape (M, T), optional
        Exact per-atom running phase at each `t_grid_s` sample,
        dimensionless. See "Exact" mode above.

    Returns
    -------
    numpy.ndarray, shape (T,), complex128
        ``C(t)`` sampled at `t_grid_s`.
    """
    phi = _as_phi(phi_final)
    m = phi.shape[0]
    w = _normalized_weights(weights, m)
    t_grid = np.asarray(t_grid_s, dtype=np.float64)
    if t_grid.ndim != 1:
        raise ValueError(f"t_grid_s must be 1-D, shape (T,); got shape {t_grid.shape}")

    if phase_history is not None:
        history = np.asarray(phase_history, dtype=np.float64)
        if history.shape != (m, t_grid.shape[0]):
            raise ValueError(
                f"phase_history must have shape (M, T) = ({m}, {t_grid.shape[0]}); "
                f"got {history.shape}"
            )
        phase_it = history
    else:
        if t_interrogation_s <= 0.0:
            raise ValueError("t_interrogation_s must be positive")
        phase_it = phi[:, None] * (t_grid[None, :] / t_interrogation_s)  # (M, T)

    phases = np.exp(1j * phase_it)  # (M, T) complex128, |phases| == 1 exactly
    coherence: NDArray[np.complex128] = np.sum(w[:, None] * phases, axis=0)
    return coherence


def line_profile(
    coherence: ArrayLike, dt_s: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Spectral line profile: Fourier transform of the coherence function (E28).

    Frequency-axis convention (test-pinned, WP5 spec): `coherence` is
    treated as uniformly sampled at spacing `dt_s` seconds; the profile is
    ``numpy.fft.fft`` (forward transform, kernel ``exp(-2πi k n/N)``),
    shifted to a centered, monotonically increasing frequency axis via
    ``numpy.fft.fftshift``. With this convention a coherence component
    ``exp(i·2π·f0·t)`` (a real, positive frequency offset `f0` from the
    clock's nominal frequency) produces a peak at ``+f0`` in the returned
    `frequency_offsets_hz` axis -- verified directly in
    ``tests/test_analytics_stats.py``. Amplitude is normalized by the
    number of samples (``|FFT| / N``), so a single unit-modulus coherence
    component yields peak amplitude ``O(1)``.

    Parameters
    ----------
    coherence : array_like, shape (T,), complex
        ``C(t)`` sampled uniformly at spacing `dt_s`, e.g. from
        `coherence_function`.
    dt_s : float
        Sample spacing of `coherence`, seconds.

    Returns
    -------
    frequency_offsets_hz : numpy.ndarray, shape (T,), float64
        Frequency offsets from the clock's nominal frequency, hertz,
        monotonically increasing (`numpy.fft.fftshift` of
        `numpy.fft.fftfreq`).
    amplitude : numpy.ndarray, shape (T,), float64
        Normalized spectral amplitude ``|FFT(C)| / T``, dimensionless.
    """
    c = np.asarray(coherence, dtype=np.complex128)
    if c.ndim != 1:
        raise ValueError(f"coherence must be 1-D, shape (T,); got shape {c.shape}")
    if c.shape[0] < 2:
        raise ValueError("coherence must have at least 2 samples")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    n = c.shape[0]
    spectrum = np.fft.fftshift(np.fft.fft(c))
    frequency_offsets_hz = np.fft.fftshift(np.fft.fftfreq(n, d=dt_s))
    amplitude = np.abs(spectrum) / n
    return frequency_offsets_hz.astype(np.float64), amplitude.astype(np.float64)
