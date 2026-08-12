# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for cliffordclock.analytics.stats (CONVENTIONS.md E22-E23, E25-E28).

Covers the metrology-analytics test contract (mean shift, SEM, T2*, and
the phase-variance cancellation guard).

Item 3 (the cancellation guard) needs an independent, non-`stats.py`
reference at up to ~1e6 samples; `mpmath` is not installed in this
project's venv (and is not a project dependency -- WP5 orchestrator
instruction 5 permits `math.fsum`/`decimal` as the fallback), so the
reference below is built with the stdlib `decimal` module at high
precision, a genuinely independent code path from both the naive
one-pass formula under test and `stats.py`'s own `math.fsum`-based
two-pass implementation.
"""

from __future__ import annotations

import decimal
import math
import statistics

import numpy as np
import pytest
from numpy.polynomial import hermite as np_hermite

from cliffordclock.analytics.stats import (
    coherence_function,
    dephasing_time_t2star,
    line_profile,
    mean_fractional_shift,
    shift_std_error,
    weighted_phase_stats,
)
from cliffordclock.constants import TAU_COMPTON

# ---------------------------------------------------------------------------
# Test contract item 1: known-distribution T2*.
# ---------------------------------------------------------------------------


def test_dephasing_time_t2star_matches_closed_form_on_gaussian_ensemble():
    """T2* = sqrt(2) * T / sigma_Phi (E27) on a synthetic Gaussian phase
    ensemble, checked against an independent reference (`statistics.variance`,
    a stdlib implementation distinct from `stats.py`'s `math.fsum`-based
    two-pass) to 1e-10 relative -- i.e. this pins the *formula*
    (sqrt(2)*T/sigma), not statistical convergence of the sample to some
    population parameter.
    """
    rng = np.random.default_rng(20260808)
    m = 200_000
    mean_phi = 3.7e-18  # realistic 1e-18-scale phase, not a "convenient" O(1) number
    sigma0 = 4.0e-20
    phi = mean_phi + sigma0 * rng.standard_normal(m)
    t_interrogation_s = 1.0

    # Independent reference: stdlib `statistics.variance` (ddof=1), a
    # different implementation from stats.py's math.fsum two-pass.
    sigma_phi_ref = math.sqrt(statistics.variance(phi.tolist()))
    t2_star_ref = math.sqrt(2.0) * t_interrogation_s / sigma_phi_ref

    t2_star = dephasing_time_t2star(phi, t_interrogation_s)
    np.testing.assert_allclose(t2_star, t2_star_ref, rtol=1e-10, atol=0)


def test_dephasing_time_t2star_gauss_hermite_exact_second_moment():
    """A second, fully closed-form (non-statistical) T2* check: phases are
    Gauss-Hermite quadrature nodes/weights representing a Gaussian
    population exactly in its first two moments (any n_quad >= 2 node GH
    quadrature integrates a degree-2 polynomial, i.e. the variance,
    exactly) -- so the expected `weighted_phase_stats` variance is
    knowable in closed form to double-precision roundoff, not sampling
    noise, and non-uniform ("quadrature-node form") weights are exercised
    per the WP5 spec's explicit mention of that case.
    """
    n_quad = 12
    xi, wi = np_hermite.hermgauss(n_quad)
    weights = wi / math.fsum(wi.tolist())  # normalize GH weights to sum to 1
    sigma0 = 5.0e-19
    mean_phi = 8.2e-18
    phi = mean_phi + sigma0 * math.sqrt(2.0) * xi  # GH node placement for N(mean_phi, sigma0^2)

    # Closed form: Gauss-Hermite of degree 2*n_quad-1 >= 3 exactly integrates
    # x^2, so the *population* (biased, sum-of-weights=1) variance recovered
    # by quadrature equals sigma0^2 exactly (up to fp roundoff). stats.py's
    # weighted variance applies the reliability-weights bias correction
    # S / (1 - sum(w^2)); reproduce that exactly here for the expected value.
    sum_w_sq = math.fsum((weights * weights).tolist())
    expected_variance = sigma0**2 / (1.0 - sum_w_sq)
    t_interrogation_s = 0.5
    expected_t2_star = math.sqrt(2.0) * t_interrogation_s / math.sqrt(expected_variance)

    stats = weighted_phase_stats(phi, weights)
    np.testing.assert_allclose(stats.variance, expected_variance, rtol=1e-10, atol=0)

    t2_star = dephasing_time_t2star(phi, t_interrogation_s, weights)
    np.testing.assert_allclose(t2_star, expected_t2_star, rtol=1e-10, atol=0)


def test_dephasing_time_t2star_zero_variance_returns_inf():
    """WP9 regression (latent WP5 edge case): an ensemble whose atoms all
    accumulated the *identical* phase has exactly zero weighted phase
    variance (E25), so E27's ``sigma_Phi -> 0+`` limit applies:
    `dephasing_time_t2star` must return ``+inf`` (no inhomogeneous
    dephasing), not raise ``ZeroDivisionError`` (the pre-fix behavior --
    it propagated as an unhandled traceback out of
    `cliffordclock.pipeline.run_pipeline_full` for any lattice run in a
    spatially uniform field with more than one quadrature node).

    M=4 (a power of two) with uniform weights makes the identical
    *nonzero* phase case hit variance == 0.0 exactly (``1/4`` and the
    ``0.25 * phi`` products are exact in binary floating point, and
    `math.fsum` sums them exactly), so this exercises the zero-*spread*
    case, not just the zero-*phase* one.
    """
    t_interrogation_s = 1.0
    phi = np.full(4, 3.7e-18)

    t2_star = dephasing_time_t2star(phi, t_interrogation_s)
    assert math.isinf(t2_star)
    assert t2_star > 0.0

    # Sanity: this is the sigma_Phi == 0 branch, not luck.
    assert weighted_phase_stats(phi).variance == 0.0


def test_dephasing_time_t2star_zero_phase_quadrature_weights_returns_inf():
    """The exact shape of the pipeline reproduction: non-uniform
    (Gauss-Hermite lattice quadrature, WP4) weights and every node's phase
    *exactly zero* (a uniform field orthogonal to the coupling ``mu``
    gives ``P - 1 == 0`` identically). Zero phases have exactly zero
    weighted mean and variance for *any* weight vector, so T2* is
    ``+inf`` here too; `shift_std_error` at zero variance is exactly
    ``0.0`` (well-defined -- zero spread has zero standard error), and
    must stay that way rather than becoming NaN/raising.
    """
    n_quad = 4
    _, wi = np_hermite.hermgauss(n_quad)
    weights = wi / math.fsum(wi.tolist())
    phi = np.zeros(n_quad)
    t_interrogation_s = 1.0

    t2_star = dephasing_time_t2star(phi, t_interrogation_s, weights)
    assert math.isinf(t2_star)
    assert t2_star > 0.0

    sem = shift_std_error(phi, t_interrogation_s, weights)
    assert sem == 0.0


# ---------------------------------------------------------------------------
# Test contract item 2: shift correctness with non-uniform weights.
# ---------------------------------------------------------------------------


def test_mean_fractional_shift_exact_with_quadrature_weights():
    """`mean_fractional_shift` (E23) against a hand-computable weighted mean,
    using non-uniform Gauss-Hermite quadrature-node-form weights (WP5 spec:
    "weights of quadrature-node form (non-uniform) handled correctly"), to
    1e-15 relative.
    """
    n_quad = 8
    xi, wi = np_hermite.hermgauss(n_quad)
    weights = wi / math.fsum(wi.tolist())
    # Realistic 1e-18-scale phases, deliberately not symmetric so the
    # weighted mean isn't trivially zero.
    phi = 1.5e-18 + 2.0e-19 * xi + 3.0e-20 * xi**3

    t_interrogation_s = 2.0
    t_tilde = t_interrogation_s / TAU_COMPTON
    expected_mean_phi = math.fsum((weights * phi).tolist())
    expected_shift = expected_mean_phi / t_tilde

    shift = mean_fractional_shift(phi, t_interrogation_s, weights)
    np.testing.assert_allclose(shift, expected_shift, rtol=1e-15, atol=0)


def test_mean_fractional_shift_uniform_weights_matches_plain_average():
    """Sanity check: omitting `weights` matches explicit uniform 1/M weights."""
    rng = np.random.default_rng(7)
    phi = 2.0e-18 + 1.0e-20 * rng.standard_normal(5_000)
    t_interrogation_s = 1.0
    uniform = np.full(phi.shape[0], 1.0 / phi.shape[0])
    np.testing.assert_allclose(
        mean_fractional_shift(phi, t_interrogation_s),
        mean_fractional_shift(phi, t_interrogation_s, uniform),
        rtol=1e-12,
        atol=0,
    )


def test_shift_std_error_shrinks_with_ensemble_size():
    """Basic sanity/regression check: SEM ~ 1/sqrt(M) for uniform weights."""
    rng = np.random.default_rng(99)
    sigma0 = 1.0e-19
    t_interrogation_s = 1.0

    small = 1.0e-18 + sigma0 * rng.standard_normal(400)
    large = 1.0e-18 + sigma0 * rng.standard_normal(40_000)

    sem_small = shift_std_error(small, t_interrogation_s)
    sem_large = shift_std_error(large, t_interrogation_s)
    # 100x more atoms -> ~10x smaller SEM; allow generous statistical slack.
    assert sem_large < sem_small / 5.0


# ---------------------------------------------------------------------------
# Test contract item 3: cancellation guard.
# ---------------------------------------------------------------------------


def _decimal_reference_variance(x: np.ndarray, prec: int = 50) -> float:
    """Independent high-precision (uniform-weight, ddof=1) reference variance,
    computed with `decimal` at `prec` significant digits -- a code path
    sharing no arithmetic routines with either the naive one-pass formula or
    `stats.py`'s `math.fsum`-based implementation.
    """
    decimal.getcontext().prec = prec
    m = x.shape[0]
    total = decimal.Decimal(0)
    for v in x.tolist():
        total += decimal.Decimal(v)
    mean = total / m
    sq_total = decimal.Decimal(0)
    for v in x.tolist():
        dv = decimal.Decimal(v) - mean
        sq_total += dv * dv
    return float(sq_total / (m - 1))


def _naive_variance(x: np.ndarray) -> float:
    """The textbook one-pass `E[x^2] - E[x]^2` formula (unbiased, M/(M-1)
    correction) -- exactly what E25/CONVENTIONS.md section 8 and WP5
    orchestrator instruction 5 forbid. Deliberately *not* imported from
    `stats.py` (which never implements this): reimplemented here only to
    demonstrate the failure mode it exhibits.
    """
    m = x.shape[0]
    mean = np.sum(x) / m
    mean_sq = np.sum(x * x) / m
    return float((mean_sq - mean**2) * m / (m - 1))


def test_cancellation_guard_matches_decimal_reference_at_spec_magnitudes():
    """WP5 spec's literal cancellation-guard magnitudes (phases 1e-18 +/-
    1e-21, M=1e6): `weighted_phase_stats`'s two-pass variance matches an
    independent `decimal` reference to >= 6 significant figures (spec
    requirement). See `test_cancellation_guard_naive_would_fail` below for
    why these exact magnitudes do *not* by themselves discriminate a naive
    implementation (measured there) -- this test still exercises the spec's
    literal numbers end to end.
    """
    rng = np.random.default_rng(1)
    m = 1_000_000
    mu = 1e-18
    sigma = 1e-21
    phi = (mu + sigma * rng.standard_normal(m)).astype(np.float64)

    reference_variance = _decimal_reference_variance(phi)
    stats = weighted_phase_stats(phi)

    np.testing.assert_allclose(stats.variance, reference_variance, rtol=1e-6, atol=0)


def test_cancellation_guard_naive_would_fail():
    """Genuine discrimination test (WP5 orchestrator instruction 6): at
    phases with a realistic 1e-18-scale mean but a spread six orders of
    magnitude below the mean (mu=1e-18, sigma=1e-24, M=1e6 -- still
    "phases ~1e-18-scale" per instruction 5, just with a smaller inter-atom
    spread than the WP5 spec's illustrative 1e-21; see the WP5 builder
    report AMBIGUITY note for the measured reasoning), the naive
    `E[x^2]-E[x]^2` formula loses essentially all significant figures,
    while `weighted_phase_stats`'s two-pass `math.fsum` implementation
    stays accurate to >= 6 significant figures against an independent
    `decimal` reference.
    """
    rng = np.random.default_rng(2)
    m = 1_000_000
    mu = 1e-18
    sigma = 1e-24
    phi = (mu + sigma * rng.standard_normal(m)).astype(np.float64)

    reference_variance = _decimal_reference_variance(phi)

    naive_variance = _naive_variance(phi)
    naive_rel_err = abs(naive_variance - reference_variance) / reference_variance
    # The naive formula must fail a 6-sig-fig bar (rel err > 1e-6); assert a
    # comfortable margin below that bar so this isn't a coin flip.
    assert naive_rel_err > 1e-5, (
        f"naive E[x^2]-E[x]^2 unexpectedly accurate (rel_err={naive_rel_err:.3e}); "
        "cancellation-guard test does not discriminate at these magnitudes"
    )

    stats = weighted_phase_stats(phi)
    twopass_rel_err = abs(stats.variance - reference_variance) / reference_variance
    assert twopass_rel_err < 1e-6, f"two-pass variance rel_err={twopass_rel_err:.3e} >= 1e-6"

    # The two-pass implementation must be meaningfully (not just nominally)
    # better than naive: require at least a 100x margin.
    assert naive_rel_err > 100.0 * max(twopass_rel_err, 1e-300)


# ---------------------------------------------------------------------------
# Test contract item 4: line profile peak positions.
# ---------------------------------------------------------------------------


def test_line_profile_single_component_peak_at_predicted_offset():
    """Linear-in-time phases (a pure per-atom frequency offset) -> the FFT
    line profile peaks at the predicted Hz offset within one frequency bin.
    """
    n = 4096
    t_interrogation_s = 1.0
    dt_s = t_interrogation_s / (n - 1)
    t_grid_s = np.arange(n) * dt_s

    f0_hz = 137.0
    # Constant-rate atoms: phase(t) = 2*pi*f0*t exactly, so phi_final (at
    # t=T) = 2*pi*f0*T reproduces this exactly through the linear
    # reconstruction path.
    phi_final = np.full(500, 2.0 * math.pi * f0_hz * t_interrogation_s)

    coherence = coherence_function(phi_final, t_interrogation_s, t_grid_s)
    freqs_hz, amplitude = line_profile(coherence, dt_s)

    bin_width_hz = freqs_hz[1] - freqs_hz[0]
    peak_freq = freqs_hz[np.argmax(amplitude)]
    assert abs(peak_freq - f0_hz) <= bin_width_hz


def test_line_profile_two_component_resolves_both_peaks():
    """A two-component ensemble (two distinct constant frequency offsets)
    yields two resolved peaks at the correct positions.
    """
    n = 8192
    t_interrogation_s = 1.0
    dt_s = t_interrogation_s / (n - 1)
    t_grid_s = np.arange(n) * dt_s

    f1_hz, f2_hz = 50.0, -300.0
    phi_final = np.concatenate(
        [
            np.full(300, 2.0 * math.pi * f1_hz * t_interrogation_s),
            np.full(300, 2.0 * math.pi * f2_hz * t_interrogation_s),
        ]
    )

    coherence = coherence_function(phi_final, t_interrogation_s, t_grid_s)
    freqs_hz, amplitude = line_profile(coherence, dt_s)

    bin_width_hz = freqs_hz[1] - freqs_hz[0]
    # Two resolved peaks: find the top two local maxima by amplitude.
    order = np.argsort(amplitude)[::-1]
    top_two_freqs = sorted(freqs_hz[order[:2]].tolist())
    expected = sorted([f1_hz, f2_hz])

    assert abs(top_two_freqs[0] - expected[0]) <= bin_width_hz
    assert abs(top_two_freqs[1] - expected[1]) <= bin_width_hz


def test_coherence_function_exact_history_path():
    """`phase_history` (exact per-step running phase) is honored directly,
    bypassing the linear reconstruction -- verified against a hand-built
    two-atom, two-time-point history with a known closed-form C(t).
    """
    t_grid_s = np.array([0.0, 1.0])
    phi_final = np.array([1.23, 4.56])  # unused by the exact path
    # Atom 0: phase 0 -> pi/2; atom 1: phase 0 -> pi.
    history = np.array([[0.0, math.pi / 2.0], [0.0, math.pi]])
    weights = np.array([0.5, 0.5])

    coherence = coherence_function(phi_final, 1.0, t_grid_s, weights=weights, phase_history=history)

    expected_t0 = 0.5 * np.exp(1j * 0.0) + 0.5 * np.exp(1j * 0.0)
    expected_t1 = 0.5 * np.exp(1j * math.pi / 2.0) + 0.5 * np.exp(1j * math.pi)
    np.testing.assert_allclose(coherence, np.array([expected_t0, expected_t1]), rtol=0, atol=1e-14)


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------


def test_weighted_phase_stats_rejects_single_atom():
    with pytest.raises(ValueError, match="undefined"):
        weighted_phase_stats(np.array([1.0e-18]))


def test_weighted_phase_stats_rejects_mismatched_weights_shape():
    with pytest.raises(ValueError, match="shape"):
        weighted_phase_stats(np.array([1.0e-18, 2.0e-18, 3.0e-18]), weights=np.array([0.5, 0.5]))


# ---------------------------------------------------------------------------
# WP6 addition: M=1 (single-particle) boundary through the public API, and
# dtype assertions (WP5 review coverage gap; see the WP6 builder report).
# ---------------------------------------------------------------------------


def test_mean_fractional_shift_works_at_m_equals_1():
    """`mean_fractional_shift` (E23) needs no variance estimate, so unlike
    `weighted_phase_stats` it must work at M=1: the "mean" of a single
    value is that value itself.
    """
    phi = np.array([3.0e-18])
    t_interrogation_s = 2.0
    expected = phi[0] / (t_interrogation_s / TAU_COMPTON)
    shift = mean_fractional_shift(phi, t_interrogation_s)
    np.testing.assert_allclose(shift, expected, rtol=1e-15, atol=0)


def test_shift_std_error_rejects_single_atom():
    """`shift_std_error` (E25) delegates to `weighted_phase_stats`'s
    variance estimator, which is undefined for M=1 -- must raise cleanly,
    not return a nonsensical (e.g. zero or NaN-silent) value.
    """
    with pytest.raises(ValueError, match="undefined"):
        shift_std_error(np.array([1.0e-18]), 1.0)


def test_dephasing_time_t2star_rejects_single_atom():
    """`dephasing_time_t2star` (E27) is likewise undefined at M=1 (needs sigma_Phi)."""
    with pytest.raises(ValueError, match="undefined"):
        dephasing_time_t2star(np.array([1.0e-18]), 1.0)


def test_coherence_function_dtype_complex128():
    t_grid_s = np.array([0.0, 0.5, 1.0])
    phi_final = np.array([1.0e-18, 2.0e-18, -3.0e-18])
    coherence = coherence_function(phi_final, 1.0, t_grid_s)
    assert coherence.dtype == np.complex128


def test_line_profile_dtype_float64():
    t_grid_s = np.linspace(0.0, 1.0, 64)
    phi_final = np.full(10, 1.0e-18)
    coherence = coherence_function(phi_final, 1.0, t_grid_s)
    freqs_hz, amplitude = line_profile(coherence, t_grid_s[1] - t_grid_s[0])
    assert freqs_hz.dtype == np.float64
    assert amplitude.dtype == np.float64
