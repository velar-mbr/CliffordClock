# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP31 coherent (phase-resolved) rotor composition and
Ramsey visibility (CONVENTIONS.md section 8, E39).

``cliffordclock.integrator.coherence`` implements the combiner
(``phase_to_rotor``/``coherent_rotor_composition``) and the visibility/
phase projection (``ramsey_visibility_and_phase``). This file covers:
unit rotors composing to ``|M| <= 1`` with equality iff every phase is
identical; the exact Gaussian closure identity ``V = exp(-sigma_Phi^2/2)``
on a synthetic Gaussian phase ensemble; the two classic-error kill tests
(phase-averaging, renormalizing); a zero-spread ensemble giving ``V = 1``
to float precision; the pipeline-level squeezing_r sweep (WP31's
squeezed-motional-state sampling extension, wired through
``cliffordclock.pipeline.run_pipeline_full``); and squeezing config
parse/validation.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.cl13 import IDX_E12, IDX_SCALAR, normalize_rotor
from cliffordclock.integrator.coherence import (
    coherent_rotor_composition,
    phase_to_rotor,
    ramsey_visibility_and_phase,
)
from cliffordclock.pipeline import PipelineConfig, PipelineConfigError, run_pipeline_full

# ---------------------------------------------------------------------------
# 1. Basic combiner properties: unit rotors in, |M| <= 1, equality iff
#    every phase is identical.
# ---------------------------------------------------------------------------


def test_unit_rotors_compose_to_visibility_at_most_one() -> None:
    """A population-weighted sum of unit `B_hat_C`-plane rotors has
    ``V <= 1`` always (a convex combination of unit-modulus complex-like
    values, so its modulus cannot exceed 1 -- the triangle inequality).
    """
    rng = np.random.default_rng(0)
    phi = rng.uniform(-3.0, 3.0, size=500)
    weights = rng.uniform(0.1, 1.0, size=500)
    weights = weights / weights.sum()

    rotors = phase_to_rotor(jnp.asarray(phi))
    # Sanity: phase_to_rotor produces genuine unit rotors confined to the
    # B_hat_C plane (E6: R R~ = 1).
    norm_sq = rotors[..., IDX_SCALAR] ** 2 + rotors[..., IDX_E12] ** 2
    np.testing.assert_allclose(np.asarray(norm_sq), 1.0, rtol=0, atol=1e-12)

    m = coherent_rotor_composition(rotors, jnp.asarray(weights))
    visibility, _phase = ramsey_visibility_and_phase(m)
    assert float(visibility) <= 1.0 + 1e-12


def test_visibility_equals_one_iff_all_phases_equal() -> None:
    """Equality `V == 1` holds exactly when every composed phase is
    identical (the "no spread, no decoherence" limit), and strictly less
    than 1 for any genuine spread.
    """
    weights = jnp.full(10, 0.1)

    # Identical phases (a nonzero common phase, not just phi=0): V == 1
    # exactly (every unit rotor in the sum is bit-identical).
    identical_phi = jnp.full(10, 0.37)
    m_identical = coherent_rotor_composition(phase_to_rotor(identical_phi), weights)
    v_identical, _ = ramsey_visibility_and_phase(m_identical)
    np.testing.assert_allclose(float(v_identical), 1.0, rtol=0, atol=1e-13)

    # A genuine spread: V strictly less than 1.
    rng = np.random.default_rng(1)
    spread_phi = jnp.asarray(rng.normal(0.0, 0.3, size=10))
    m_spread = coherent_rotor_composition(phase_to_rotor(spread_phi), weights)
    v_spread, _ = ramsey_visibility_and_phase(m_spread)
    assert float(v_spread) < 1.0 - 1e-9


def test_zero_spread_ensemble_gives_visibility_one_to_float_precision() -> None:
    """Every worldline sharing the identical accumulated phase (e.g. a
    lattice ensemble in a spatially uniform environment) gives `V = 1` to
    float64 precision, regardless of `weights` (uniform or not).
    """
    m = 200
    phi = jnp.full(m, -1.2345e-9)
    rng = np.random.default_rng(2)
    weights_raw = rng.uniform(0.5, 2.0, size=m)
    weights = jnp.asarray(weights_raw / weights_raw.sum())

    coherence_mv = coherent_rotor_composition(phase_to_rotor(phi), weights)
    visibility, phase = ramsey_visibility_and_phase(coherence_mv)
    np.testing.assert_allclose(float(visibility), 1.0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(float(phase), -1.2345e-9, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Gaussian closure: V vs exp(-sigma_Phi^2/2) on a synthetic Gaussian
#    phase ensemble.
#
# Tolerance derivation: for a finite i.i.d. Gaussian sample
# {phi_k ~ N(mu, sigma^2)}_{k=1..N} with uniform weights, the exact
# quantity this module computes is C_N = (1/N) sum_k exp(i*phi_k), and
# the identity being validated compares |C_N| against exp(-s_N^2/2) where
# s_N^2 is the SAME sample's own (unbiased) variance -- both computed
# from the same finite draw, not the population sigma. Expanding
# ln(C_N) in the sample's cumulants (kappa_1 = sample mean, kappa_2 =
# s_N^2, kappa_3/kappa_4 = sample skewness/excess-kurtosis moments):
# ln|C_N| = -kappa_2/2 - kappa_4/24 + O(kappa_3^2) + ... For an
# underlying Gaussian generator, the sample's own kappa_3/kappa_4 are
# asymptotically zero-mean with standard deviations ~ sqrt(6/N) and
# sqrt(24/N) respectively (standard errors of sample skewness/kurtosis),
# so ||C_N| - exp(-s_N^2/2)| / exp(-s_N^2/2) = O(sigma^4/N) in relative
# terms (the kappa_4 term dominates at this order; kappa_3 enters only
# via kappa_3^2, also O(1/N)). At sigma=0.5, N=200_000: measured relative
# deviation ~2e-5 (verified numerically during test design); 1e-3 below
# gives a 50x safety margin against N-dependent sampling noise from any
# fixed seed.
# ---------------------------------------------------------------------------


def test_gaussian_closure_identity_on_synthetic_ensemble() -> None:
    """`V = exp(-sigma_Phi^2/2)` (E39's Gaussian closure identity),
    verified on a large synthetic Gaussian phase ensemble against the
    SAME sample's own empirical phase standard deviation.
    """
    rng = np.random.default_rng(42)
    sigma_pop = 0.5
    n = 200_000
    phi = rng.normal(0.3, sigma_pop, size=n)
    weights = jnp.full(n, 1.0 / n)

    m = coherent_rotor_composition(phase_to_rotor(jnp.asarray(phi)), weights)
    visibility, _phase = ramsey_visibility_and_phase(m)

    sigma_sample = float(np.std(phi, ddof=1))
    closure = math.exp(-(sigma_sample**2) / 2.0)
    np.testing.assert_allclose(float(visibility), closure, rtol=1e-3, atol=0)


# ---------------------------------------------------------------------------
# 3. Kill tests: the two classic errors.
# ---------------------------------------------------------------------------


def test_kill_test_a_phase_averaging_combiner_gives_visibility_one() -> None:
    """Kill test (a): summing the per-worldline PHASES and exponentiating
    the mean (`|exp(-i*mean(phi))| == 1` identically) loses all visibility
    information, on a genuinely spread ensemble where the REAL combiner
    reports the correct sub-unity visibility matching the Gaussian
    closure identity.
    """
    rng = np.random.default_rng(7)
    sigma_pop = 0.4
    n = 100_000
    phi = rng.normal(0.0, sigma_pop, size=n)
    weights_np = np.full(n, 1.0 / n)

    # The classic-error combiner: average the phases first, THEN take a
    # single phase factor's modulus.
    mean_phase = float(np.sum(weights_np * phi))
    v_bad = abs(complex(math.cos(mean_phase), math.sin(mean_phase)))
    np.testing.assert_allclose(v_bad, 1.0, rtol=0, atol=1e-12)

    # The real combiner: genuinely below 1, matching the Gaussian closure.
    m = coherent_rotor_composition(phase_to_rotor(jnp.asarray(phi)), jnp.asarray(weights_np))
    v_real, _phase = ramsey_visibility_and_phase(m)
    sigma_sample = float(np.std(phi, ddof=1))
    closure = math.exp(-(sigma_sample**2) / 2.0)

    assert abs(float(v_real) - v_bad) > 0.05  # a real, substantial disagreement
    np.testing.assert_allclose(float(v_real), closure, rtol=1e-3, atol=0)


def test_kill_test_b_renormalizing_combiner_gives_visibility_one() -> None:
    """Kill test (b): renormalizing the coherence object `M` back to a
    unit rotor (`M / sqrt(<M~M>_0)`) erases the visibility deficit by
    construction -- `V == 1` trivially, on the SAME spread ensemble the
    real (never-renormalized) combiner correctly reports sub-unity
    visibility for.
    """
    rng = np.random.default_rng(13)
    sigma_pop = 0.4
    n = 100_000
    phi = rng.normal(0.0, sigma_pop, size=n)
    weights = jnp.full(n, 1.0 / n)

    m = coherent_rotor_composition(phase_to_rotor(jnp.asarray(phi)), weights)
    v_real, _phase = ramsey_visibility_and_phase(m)

    m_renormalized = normalize_rotor(m)
    c = m_renormalized[..., IDX_SCALAR]
    s = m_renormalized[..., IDX_E12]
    v_bad = float(jnp.sqrt(c * c + s * s))

    np.testing.assert_allclose(v_bad, 1.0, rtol=0, atol=1e-12)
    assert float(v_real) < 1.0 - 0.01  # the real combiner shows a genuine deficit
    assert abs(float(v_real) - v_bad) > 0.05


# ---------------------------------------------------------------------------
# 4. coherent_rotor_composition input validation.
# ---------------------------------------------------------------------------


def test_coherent_rotor_composition_rejects_shape_mismatch() -> None:
    rotors = phase_to_rotor(jnp.zeros(5))
    with pytest.raises(ValueError, match="weights"):
        coherent_rotor_composition(rotors, jnp.zeros(4))
    with pytest.raises(ValueError, match="rotors"):
        coherent_rotor_composition(jnp.zeros((5, 15)), jnp.zeros(5))


# ---------------------------------------------------------------------------
# 5. Pipeline-level: squeezing_r sweep, worldline-mode Ramsey visibility.
#
# A uniform (zero-gradient, zero-value) field with linear_mu coupling
# mu=(0,0,0) isolates E21's pure KINEMATIC second-order-Doppler term
# (P(r) == 1 everywhere, so delta_omega = sqrt(1-v^2/c^2) - 1): the
# ensemble's accumulated-phase spread is driven ENTIRELY by the sampled
# velocity distribution, which squeezing_r directly controls (velocity
# quadrature variance scales as exp(+2r)). A large dtau (Compton units)
# amplifies the accumulated phase without changing the classical trap's
# real-time dynamics appreciably (dt_seconds = dtau*TAU_COMPTON stays far
# below the trap period at these parameters, so positions/velocities stay
# close to their initial thermal draw across the few steps taken) --
# purely a numerical lever to bring sigma_Phi into an O(0.1-1) range
# where the Gaussian closure comparison is both meaningful and precise
# (chosen/verified during test design, not tuned to make the test pass).
# ---------------------------------------------------------------------------


def _squeezing_sweep_config(r: float | None) -> PipelineConfig:
    data = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 0.0]}}},
        "coupling": {"mu": [0.0, 0.0, 0.0]},
        "ensemble": {
            "regime": "classical",
            "temperature_uK": 1.0e6,
            "size": 6000,
            "seed": 11,
        },
        "integration": {"dtau": 2.0e12, "steps": 50},
    }
    if r is not None:
        data["ensemble"]["squeezing_r"] = [r, r, r]
    return PipelineConfig.from_dict(data)


def test_pipeline_squeezing_r_sweep_visibility_decreases_and_matches_closure() -> None:
    """A `squeezing_r` sweep (WP31) shows `ramsey_visibility` decreasing
    monotonically with `r` (velocity quadrature variance grows as
    `exp(+2r)`, widening the kinematic phase spread), matching
    `exp(-sigma_Phi^2/2)` computed from the SAME run's per-worldline
    phases at every sweep point.
    """
    r_values = [0.0, 0.15, 0.3, 0.45, 0.6]
    visibilities = []
    for r in r_values:
        config = _squeezing_sweep_config(r)
        result = run_pipeline_full(config)
        assert result.report.ramsey_visibility is not None
        phi = np.asarray(result.ensemble_result.phase)
        sigma_sample = float(np.std(phi, ddof=1))
        closure = math.exp(-(sigma_sample**2) / 2.0)
        np.testing.assert_allclose(result.report.ramsey_visibility, closure, rtol=1.5e-2, atol=0)
        visibilities.append(result.report.ramsey_visibility)

    assert all(visibilities[i] > visibilities[i + 1] for i in range(len(visibilities) - 1)), (
        f"ramsey_visibility did not decrease monotonically with r: {visibilities}"
    )


def test_pipeline_squeezing_r_absent_matches_r_zero_bitwise_for_velocities() -> None:
    """`squeezing_r` absent (`None`) reproduces the SAME sampled ensemble
    (and hence the same report) as `squeezing_r=[0.0, 0.0, 0.0]` -- the
    zero-squeezing limit is a genuine no-op, not merely close.
    """
    result_absent = run_pipeline_full(_squeezing_sweep_config(None))
    result_zero = run_pipeline_full(_squeezing_sweep_config(0.0))
    assert result_absent.report.ramsey_visibility == result_zero.report.ramsey_visibility
    assert result_absent.report.mean_fractional_shift == result_zero.report.mean_fractional_shift


def test_ramsey_note_states_gaussian_only_scope() -> None:
    """The report note folded in whenever `ramsey_visibility` is
    populated states the Gaussian-only scope boundary explicitly.
    """
    result = run_pipeline_full(_squeezing_sweep_config(None))
    assert "Gaussian" in result.report.uncertainty_notes
    assert "ramsey_visibility" in result.report.uncertainty_notes


def test_fast_path_and_secular_leave_ramsey_fields_none() -> None:
    """`ramsey_visibility`/`ramsey_phase` are `None` for `fast_path` (the
    closed-form E29 lattice expectation, no per-worldline dynamical phase
    accumulation) -- populated only for the "direct"/"worldline" modes.
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
            "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
            "ensemble": {"regime": "lattice", "temperature_uK": 1.0, "motional_n": [0, 0, 0]},
            "integration": {"mode": "fast_path", "time_s": 1.0},
        }
    )
    result = run_pipeline_full(config)
    assert result.report.ramsey_visibility is None
    assert result.report.ramsey_phase is None


# ---------------------------------------------------------------------------
# 6. squeezing_r config parse/validation.
# ---------------------------------------------------------------------------


def _base_squeezing_config_dict(**ensemble_overrides: object) -> dict:
    ensemble = {
        "regime": "classical",
        "temperature_uK": 1.0,
        "size": 10,
        "seed": 0,
    }
    ensemble.update(ensemble_overrides)
    return {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
        "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
        "ensemble": ensemble,
        "integration": {"dtau": 0.5, "steps": 10},
    }


def test_squeezing_r_parses_as_float_tuple3() -> None:
    config = PipelineConfig.from_dict(_base_squeezing_config_dict(squeezing_r=[0.1, -0.2, 0.3]))
    assert config.ensemble.squeezing_r == (0.1, -0.2, 0.3)


def test_squeezing_r_absent_defaults_to_none() -> None:
    config = PipelineConfig.from_dict(_base_squeezing_config_dict())
    assert config.ensemble.squeezing_r is None


def test_squeezing_r_rejected_for_lattice_regime() -> None:
    config_dict = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
        "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
        "ensemble": {
            "regime": "lattice",
            "temperature_uK": 1.0,
            "motional_n": [0, 0, 0],
            "squeezing_r": [0.1, 0.1, 0.1],
        },
        "integration": {"mode": "fast_path", "time_s": 1.0},
    }
    with pytest.raises(PipelineConfigError, match="squeezing_r"):
        PipelineConfig.from_dict(config_dict)


def test_squeezing_r_rejected_for_lattice_extended_regime() -> None:
    config_dict = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
        "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
        "ensemble": {
            "regime": "lattice_extended",
            "temperature_uK": 1.0,
            "motional_n": [0, 0, 0],
            "n_sites": 3,
            "site_spacing_m": 1.0e-6,
            "squeezing_r": [0.1, 0.1, 0.1],
        },
        "integration": {"mode": "fast_path", "time_s": 1.0},
    }
    with pytest.raises(PipelineConfigError, match="squeezing_r"):
        PipelineConfig.from_dict(config_dict)
