# SPDX-License-Identifier: AGPL-3.0-or-later
"""Known-answer validation (KA1-4; see ``docs/validation.md``).

``tests/test_e2e.py`` validated the engine against **our own** closed
forms. This suite delivers the trust-building moment a metrology-lab
user needs instead: the tool reproduces numbers **the community already
knows**, at the coefficient level, via the physical E14b DC-Stark
coupling and the fast-path architecture (no toy numbers, no femtosecond
windows -- every case here runs at a realistic interrogation time).

- **KA1/KA2** (Sr87/Yb171 uniform-field DC-Stark): the full pipeline
  (species registry -> synthetic field -> lattice fast path -> report)
  reproduces the textbook formula ``Delta_nu/nu0 = -(Delta_alpha/2)|E|^2/(h
  nu0)`` every clock paper in this space uses, to rtol 1e-10.
- **KA3** (gradient line shift + broadening): a linear-gradient field over
  a ground-state lattice motional distribution; pipeline mean shift AND
  phase spread (Var) match an independent Gaussian-moment perturbation-
  theory hand calculation (``tests/reference_impl.py``), to rtol 1e-8.
- **KA4** (second-order Doppler): a classical thermal ensemble, secular
  mode, zero field (isolating the kinematic term); mean shift matches
  ``-<v^2>/2c^2`` from equipartition, within a documented statistical
  tolerance (multiples of the pipeline's own reported SEM).

Tolerance doctrine (this project's binding reviewer checklist): every
comparison below is ``numpy.testing.assert_allclose`` with an
explicit, justified ``rtol`` and ``atol=0`` (never a bare ``pytest.approx``);
every expected value is derived in this file (or in
``tests/reference_impl.py``) from the literature formula/species data, not
copied from a prior pipeline run. If a case ever misses its target, that is
a finding to report -- inputs are never adjusted to force a pass (WP9
non-goals).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import reference_impl
from cliffordclock.constants import BOLTZMANN_K, PLANCK_H, SPEED_OF_LIGHT
from cliffordclock.ensemble.species import get_species
from cliffordclock.pipeline import PipelineConfig, run_pipeline_full

# ---------------------------------------------------------------------------
# KA1/KA2 -- uniform-field DC-Stark shift, Sr87 and Yb171 (CONVENTIONS.md
# E14b): "the textbook Delta_nu = -(1/2) Delta_alpha |E|^2 / h" formula
# every optical-lattice-clock DC-Stark paper (e.g. Middelmann et al. 2012,
# Sherman et al. 2012 -- the same papers the species registry's Delta_alpha
# values are cited to) states directly.
#
# A single static lattice node (n_quad=1, motional_n=(0,0,0)) is used
# rather than a wider quadrature: a *uniform* field gives every quadrature
# node an identical shift by construction, so a wider quadrature would hit
# a latent divide-by-zero in cliffordclock.analytics.stats.dephasing_time_t2star
# (Var=0 ensembles are undefined for T2*, CONVENTIONS.md E27 -- a pre-
# existing pipeline edge case found during this WP, out of its file scope
# to fix; flagged in the builder report, not in analytics/ here). This
# matches CONVENTIONS.md V2's own single-static-atom validation pattern and
# still exercises the full pipeline: species registry -> synthetic field ->
# lattice ensemble -> E29 fast path -> coupling-provenance-carrying report.
# ---------------------------------------------------------------------------

#: Lab-typical stray DC field magnitude (V/m) for KA1/KA2's quotable number
#: (WP9 spec: "document the resulting Delta_nu/nu0 at a lab-typical stray
#: field, e.g. 100 V/m").
_KA1_KA2_FIELD_V_PER_M = 100.0

_KA1_KA2_TRAP_OMEGA = (2.0e5, 2.0e5, 2.0e5)
_KA1_KA2_INTERROGATION_TIME_S = 1.0  # a genuine 1 s Ramsey-style interrogation


def _ka_uniform_field_config(species_name: str, field_v_per_m: float) -> PipelineConfig:
    return PipelineConfig.from_dict(
        {
            "species": species_name,
            "trap": {"omega_xyz": list(_KA1_KA2_TRAP_OMEGA)},
            "field": {
                "synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, field_v_per_m]}}
            },
            "coupling": {"type": "stark_dc"},
            "ensemble": {
                "regime": "lattice",
                "temperature_uK": 1.0,
                "motional_n": [0, 0, 0],
                "n_quad": 1,
            },
            "integration": {"time_s": _KA1_KA2_INTERROGATION_TIME_S},
        }
    )


def _textbook_dc_stark_shift(species_name: str, field_v_per_m: float) -> float:
    """``Delta_nu/nu0 = -(Delta_alpha/2)|E|^2/(h nu0)`` (CONVENTIONS.md E14b),
    computed directly from the species registry's cited Delta_alpha -- the
    literal textbook formula, independent of which internal helper
    (``pivot_perturbation_stark``, ``resolve_stark_coefficient_hz_per_v2_m2``)
    the pipeline itself uses to arrive at the same number.
    """
    species = get_species(species_name)
    assert species.delta_alpha_dc_si is not None  # Sr87/Yb171 are populated (WP7)
    e_sq = field_v_per_m**2
    return -(species.delta_alpha_dc_si / 2.0) * e_sq / (PLANCK_H * species.clock_frequency_hz)


def test_ka1_sr87_uniform_field_dc_stark_matches_textbook_formula() -> None:
    """KA1: Sr87, 100 V/m stray field, 1 s interrogation.

    ``Delta_alpha = 4.07873e-39 C^2 m^2 J^-1`` (Middelmann et al., PRL 109,
    263004 (2012), cited in ``cliffordclock.ensemble.species.SR87``).
    Textbook formula per CONVENTIONS.md E14b:
    ``Delta_nu/nu0 = -(Delta_alpha/2)|E|^2/(h nu0)``.
    """
    config = _ka_uniform_field_config("Sr87", _KA1_KA2_FIELD_V_PER_M)
    result = run_pipeline_full(config)

    expected = _textbook_dc_stark_shift("Sr87", _KA1_KA2_FIELD_V_PER_M)
    species = get_species("Sr87")
    delta_nu_hz = expected * species.clock_frequency_hz

    np.testing.assert_allclose(result.report.mean_fractional_shift, expected, rtol=1e-10, atol=0)
    assert "coupling=stark_dc" in result.report.uncertainty_notes
    assert "Middelmann" in result.report.uncertainty_notes

    # Non-vacuous and quotable (WP9 definition of done): a nonzero shift at
    # a lab-realistic field, documented here for the G5 human review.
    assert abs(expected) > 1e-20
    print(
        f"KA1 Sr87 @ {_KA1_KA2_FIELD_V_PER_M:.0f} V/m, "
        f"T={_KA1_KA2_INTERROGATION_TIME_S:.0f}s: "
        f"Delta_nu/nu0 = {expected:+.6e}, Delta_nu = {delta_nu_hz:+.6e} Hz"
    )


def test_ka2_yb171_uniform_field_dc_stark_matches_textbook_formula() -> None:
    """KA2: Yb171, 100 V/m stray field, 1 s interrogation.

    ``Delta_alpha = 2.40269e-39 C^2 m^2 J^-1`` (Sherman et al., PRL 108,
    153002 (2012), cited in ``cliffordclock.ensemble.species.YB171``).
    Same textbook formula as KA1 (CONVENTIONS.md E14b).
    """
    config = _ka_uniform_field_config("Yb171", _KA1_KA2_FIELD_V_PER_M)
    result = run_pipeline_full(config)

    expected = _textbook_dc_stark_shift("Yb171", _KA1_KA2_FIELD_V_PER_M)
    species = get_species("Yb171")
    delta_nu_hz = expected * species.clock_frequency_hz

    np.testing.assert_allclose(result.report.mean_fractional_shift, expected, rtol=1e-10, atol=0)
    assert "coupling=stark_dc" in result.report.uncertainty_notes
    assert "Sherman" in result.report.uncertainty_notes

    assert abs(expected) > 1e-20
    print(
        f"KA2 Yb171 @ {_KA1_KA2_FIELD_V_PER_M:.0f} V/m, "
        f"T={_KA1_KA2_INTERROGATION_TIME_S:.0f}s: "
        f"Delta_nu/nu0 = {expected:+.6e}, Delta_nu = {delta_nu_hz:+.6e} Hz"
    )


# ---------------------------------------------------------------------------
# KA3 -- gradient line shift + broadening: linear-gradient field over a
# ground-state lattice motional distribution, cross-checked against
# reference_impl.stark_shift_mean_and_variance (Gaussian-moment
# perturbation theory, see that function's derivation comment).
# ---------------------------------------------------------------------------

_KA3_SPECIES = "Yb171"
_KA3_TRAP_OMEGA = (2.0e5, 2.0e5, 2.0e5)
_KA3_E0 = (30.0, -20.0, 10.0)  # V/m, a modest bias field
_KA3_GRAD = (  # V/m^2, a modest gradient (E13 convention: grad[i, j] = d_i E_j)
    (5.0e2, -2.0e2, 0.0),
    (1.0e2, 3.0e2, -1.0e2),
    (0.0, 2.0e2, -6.0e2),
)
_KA3_N_QUAD = 6  # exact for polynomials up to degree 2*6-1=11 >> the required degree 4
_KA3_INTERROGATION_TIME_S = 1.0


def test_ka3_gradient_field_mean_and_variance_match_gaussian_moment_reference() -> None:
    """KA3: Yb171, linear-gradient field, ground-state lattice ensemble.

    The E14b pivot ``P(r) - 1 = k_S|E(r)|^2/nu0`` is a *quadratic*
    polynomial in position for a field linear in position (``E(r) = e0 + r
    @ grad``, CONVENTIONS.md V2's constant-gradient field). The ground
    motional state's position density is an exact multivariate Gaussian
    (unlike excited states), so both ``<P-1>_psi`` and ``Var(P-1)_psi`` are
    known in closed form via Gaussian-moment perturbation theory --
    ``reference_impl.stark_shift_mean_and_variance`` derives this
    independently (see its module-level derivation comment).

    Gauss-Hermite quadrature at ``n_quad=6`` is exact for polynomials up
    to degree ``2*6-1=11`` per axis (``cliffordclock.ensemble.lattice``),
    comfortably covering the degree-4 polynomial ``(P-1)^2`` the pipeline's
    weighted variance needs -- so the pipeline's quadrature-based mean/
    variance are themselves *exact* (to float64 precision) realizations of
    the same continuum Gaussian-moment expectations the reference computes
    by a completely different route (closed-form linear algebra vs.
    discrete quadrature over the fast path's per-node results): a genuine,
    non-tautological cross-check.

    **Raw (uncorrected) weighted variance, not `report.shift_std_error`.**
    The pipeline's ``weighted_phase_stats``/``shift_std_error`` apply a
    "reliability weights" *sample*-variance bias correction
    (``1/(1-sum(w_i^2))``, ``cliffordclock.analytics.stats``) appropriate
    for treating quadrature nodes as if they were a finite random sample.
    That correction has no counterpart in the reference's *exact* quantum-
    expectation ``Var(P-1)_psi`` (a continuum quantity, no finite-sample
    correction applies), so this test compares the *raw* weighted variance
    ``sum_q w_q (shift_q - mean)^2`` computed directly from the pipeline's
    per-node ``fractional_shift``/`weights` -- exactly what quadrature
    computes when it is exact for the required polynomial degree -- rather
    than the bias-corrected ``report.shift_std_error``/`t2_star_s`.
    """
    config = PipelineConfig.from_dict(
        {
            "species": _KA3_SPECIES,
            "trap": {"omega_xyz": list(_KA3_TRAP_OMEGA)},
            "field": {
                "synthetic": {
                    "kind": "constant_gradient",
                    "params": {"e0": list(_KA3_E0), "grad": [list(row) for row in _KA3_GRAD]},
                }
            },
            "coupling": {"type": "stark_dc"},
            "ensemble": {
                "regime": "lattice",
                "temperature_uK": 1.0,
                "motional_n": [0, 0, 0],
                "n_quad": _KA3_N_QUAD,
            },
            "integration": {"time_s": _KA3_INTERROGATION_TIME_S},
        }
    )
    result = run_pipeline_full(config)
    assert result.report.ensemble_size == _KA3_N_QUAD**3

    species = get_species(_KA3_SPECIES)
    k_s = species.resolve_stark_coefficient_hz_per_v2_m2()
    expected_mean, expected_var = reference_impl.stark_shift_mean_and_variance(
        np.asarray(_KA3_E0, dtype=np.float64),
        np.asarray(_KA3_GRAD, dtype=np.float64),
        np.asarray(_KA3_TRAP_OMEGA, dtype=np.float64),
        species.mass_kg,
        k_s,
        species.clock_frequency_hz,
    )

    shift = np.asarray(result.ensemble_result.fractional_shift, dtype=np.float64)
    weights = np.asarray(result.weights, dtype=np.float64)
    # Already normalized to 1 by hermite_gaussian_nodes; renormalized defensively.
    weights = weights / np.sum(weights)
    pipeline_mean = float(np.sum(weights * shift))
    pipeline_var = float(np.sum(weights * (shift - pipeline_mean) ** 2))

    np.testing.assert_allclose(pipeline_mean, expected_mean, rtol=1e-8, atol=0)
    np.testing.assert_allclose(pipeline_var, expected_var, rtol=1e-8, atol=0)
    # Non-vacuous: both the mean and the gradient-driven spread must be
    # genuinely nonzero, not trivially "agreeing" at zero.
    assert abs(expected_mean) > 1e-20
    assert expected_var > 0.0

    print(
        f"KA3 {_KA3_SPECIES} gradient field: <Delta_nu/nu0>_psi = {expected_mean:+.6e}, "
        f"sqrt(Var)_psi = {math.sqrt(expected_var):.6e}"
    )


# ---------------------------------------------------------------------------
# KA4 -- second-order Doppler shift: classical thermal ensemble, secular
# mode (E30, carries the full delta_omega~ including the kinematic term),
# zero field (isolates the kinematic term: the E14b Stark contribution is
# then identically zero for *any* species/k_S, since it is proportional to
# |E|^2 = 0).
#
# Expected value derivation (equipartition, correct per-axis accounting).
# In a 3D isotropic harmonic trap, an atom's velocity has 3 independent
# Cartesian components v_x, v_y, v_z. Equipartition assigns each quadratic
# kinetic-energy degree of freedom a mean energy of (1/2) k_B T:
#
#   <(1/2) m v_k^2> = (1/2) k_B T   for k in {x, y, z}
#   => <v_k^2> = k_B T / m          (matches the Maxwell-Boltzmann sampler's
#                                     per-axis variance, sigma^2 = k_B T/m,
#                                     cliffordclock.ensemble.classical)
#   => <v^2> = <v_x^2> + <v_y^2> + <v_z^2> = 3 k_B T / m
#
# The E21 kinematic term is -<v^2>/(2c^2) (the sqrt(1-v^2/c^2)-1 expansion
# at v/c << 1), so:
#
#   <Delta_nu/nu0>_kinematic = -<v^2>/(2c^2) = -3 k_B T / (2 m c^2)
#
# This is the ensemble MEAN measured at t=0 (equipartition, instantaneous).
# The pipeline's secular mode (E30) instead reports the ORBIT-TIME-AVERAGE
# of v(t)^2 over one full trap period, per atom -- not each atom's initial
# v^2. By the virial theorem for a harmonic oscillator, the time-averaged
# kinetic energy over one period equals exactly half the atom's *total*
# mechanical energy (KE + PE), for *any* single atom's specific total
# energy: <KE>_time = E_total/2. Averaged over the ensemble (positions AND
# velocities both drawn from the correct thermal Boltzmann distributions,
# cliffordclock.ensemble.classical/traps), <E_total> = <KE0> + <PE0> =
# (3/2)k_B T + (3/2)k_B T = 3 k_B T (each of the 3 position and 3 velocity
# quadratic degrees of freedom contributing (1/2)k_B T), so the ensemble
# mean of the *time-averaged* kinetic energy is <E_total>/2 = (3/2)k_B T --
# identical to the naive *instantaneous* equipartition value. So the
# equipartition formula above is exactly what the secular-mode ensemble
# mean converges to, not merely an approximation to it.
# ---------------------------------------------------------------------------

_KA4_SPECIES = "Sr87"
_KA4_TRAP_OMEGA = (2.0e5, 2.0e5, 2.0e5)
_KA4_TEMPERATURE_UK = 5.0
_KA4_ENSEMBLE_SIZE = 5000
_KA4_SEED = 42
_KA4_INTERROGATION_TIME_S = 1.0
#: Statistical tolerance multiplier on the pipeline's own reported SEM
#: (standard error of the weighted mean, CONVENTIONS.md E23/E25) -- a
#: documented multi-sigma bound (Monte Carlo ensemble, not an exact
#: closed-form case like KA1-3), not a number picked to make this pass.
_KA4_SEM_MULTIPLE = 5.0


def test_ka4_second_order_doppler_matches_equipartition() -> None:
    """KA4: Sr87 classical ensemble, secular mode, zero field.

    See the module-level derivation comment above for the equipartition
    result ``-<v^2>/(2c^2) = -3 k_B T / (2 m c^2)`` and why the pipeline's
    secular-mode (orbit-time-averaged) ensemble mean converges to exactly
    that value, not merely approximates it.
    """
    species = get_species(_KA4_SPECIES)
    temperature_k = _KA4_TEMPERATURE_UK * 1.0e-6
    expected = -3.0 * BOLTZMANN_K * temperature_k / (2.0 * species.mass_kg * SPEED_OF_LIGHT**2)

    config = PipelineConfig.from_dict(
        {
            "species": _KA4_SPECIES,
            "trap": {"omega_xyz": list(_KA4_TRAP_OMEGA)},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 0.0]}}},
            "coupling": {"type": "stark_dc"},
            "ensemble": {
                "regime": "classical",
                "temperature_uK": _KA4_TEMPERATURE_UK,
                "size": _KA4_ENSEMBLE_SIZE,
                "seed": _KA4_SEED,
            },
            "integration": {"mode": "secular", "time_s": _KA4_INTERROGATION_TIME_S},
        }
    )
    result = run_pipeline_full(config)
    assert result.report.ensemble_type == "classical_secular_average"
    assert result.report.ensemble_size == _KA4_ENSEMBLE_SIZE

    measured = result.report.mean_fractional_shift
    sem = result.report.shift_std_error
    assert np.isfinite(sem) and sem > 0.0

    tolerance = _KA4_SEM_MULTIPLE * sem
    diff = abs(measured - expected)
    assert diff < tolerance, (
        f"KA4 measured mean shift {measured!r} disagrees with the equipartition "
        f"prediction {expected!r} by {diff!r}, exceeding {_KA4_SEM_MULTIPLE}x the "
        f"pipeline's reported SEM ({sem!r})"
    )
    # Non-vacuous: the tolerance must be tight relative to the signal itself,
    # not so loose it passes trivially regardless of what the pipeline computes.
    assert tolerance < 0.1 * abs(expected)

    print(
        f"KA4 {_KA4_SPECIES} @ {_KA4_TEMPERATURE_UK:.1f} uK (M={_KA4_ENSEMBLE_SIZE}): "
        f"measured -<v^2>/2c^2 = {measured:+.6e} +/- {sem:.2e} (SEM), "
        f"equipartition = {expected:+.6e} "
        f"(diff = {diff / sem:.3f} SEM)"
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "-s"])
