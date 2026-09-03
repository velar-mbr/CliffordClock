# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP39 Phase A Rydberg vapor-cell response module
(CONVENTIONS.md section 19, E43/E44): the unit conversion, the quantum-
defect and Numerov mu_RF derivation, the Stark validity guard, the
ladder-susceptibility formula transcription, the Doppler-mismatch factor,
and the C3/C4/C5/C6/C7 gated checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cliffordclock import constants
from cliffordclock.integrator import rydberg_cell_response as rcr

RB85_MASS_KG = 84.911789738 * constants.ATOMIC_MASS_UNIT


def _default_system(mu_coupling_c_m: float = 5.0e-30) -> rcr.LadderSystem:
    return rcr.LadderSystem(
        mu_probe_c_m=2.0e-29,
        mu_coupling_c_m=mu_coupling_c_m,
        mu_rf_c_m=rcr.RB85_MU_RF_32D52_33P32_C_M,
        gamma_12=2.0 * math.pi * 6.0e6,
        gamma_13=2.0 * math.pi * 0.3e6,
        gamma_14=2.0 * math.pi * 0.3e6,
        number_density_m3=1.0e16,
        wavelength_probe_m=rcr.HOLLOWAY_LAMBDA_PROBE_M,
        wavelength_coupling_m=rcr.HOLLOWAY_LAMBDA_COUPLING_M,
    )


# ---------------------------------------------------------------------------
# Section A: unit conversion (dossier risk 3)
# ---------------------------------------------------------------------------


class TestUnitConversion:
    def test_atomic_unit_field_matches_codata(self) -> None:
        """CODATA's own atomic unit of electric field is 5.14220674763e11
        V/m; this module derives it from already-pinned constants rather
        than transcribing it a second time.
        """
        assert pytest.approx(5.14220674763e11, rel=1e-6) == rcr.ATOMIC_UNIT_FIELD_V_PER_M

    def test_hartree_to_hz_matches_codata(self) -> None:
        assert pytest.approx(6.579683920502e15, rel=1e-6) == rcr.HARTREE_TO_HZ

    def test_hand_computed_alpha0_conversion(self) -> None:
        """Hand-computed check (dossier risk 3): for alpha0 = 1e10 a.u.
        (a0^3, order of magnitude of a real nD5/2 state), k = -alpha0_SI *
        5e-3 / h, worked by hand outside this module:

        alpha0_SI = 1e10 * 1.64877727436e-41 = 1.64877727436e-31 C^2 m^2/J
        k = -1.64877727436e-31 * 5e-3 / 6.62607015e-34
          = -8.2438863718e-34 / 6.62607015e-34 = -1.244159... MHz/(V/cm)^2
        """
        k = rcr.alpha0_au_to_mhz_per_vcm2(1.0e10)
        assert k == pytest.approx(-1.244159, rel=1e-5)

    def test_round_trip(self) -> None:
        alpha0 = 1.4e10
        k = rcr.alpha0_au_to_mhz_per_vcm2(alpha0)
        assert rcr.mhz_per_vcm2_to_alpha0_au(k) == pytest.approx(alpha0, rel=1e-12)

    def test_sign_is_negative_for_positive_polarizability(self) -> None:
        """A wrong-sign transcription is exactly the class of error this
        project's history warns about; pin the sign explicitly.
        """
        assert rcr.alpha0_au_to_mhz_per_vcm2(1.0e10) < 0.0


# ---------------------------------------------------------------------------
# Section B: quantum defects and effective quantum number
# ---------------------------------------------------------------------------


class TestQuantumDefects:
    def test_nd52_effective_quantum_number_matches_mack2011(self) -> None:
        """Mack et al. 2011 Table III (nS)/Table V (nD5/2) prints the
        transition frequency and the resulting quantum defect at each
        measured n directly; at n=32 (85Rb, their Table V, verified
        against the arXiv PDF this session) they report delta=1.345828.
        `delta(n) = delta0 + delta2/(n-delta0)^2` from Table I's fitted
        (delta0, delta2) should reproduce that per-n value closely (the
        Table I fit is over n=19-57, so n=32 is solidly inside its
        support).
        """
        n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        delta = 32.0 - n_star
        assert delta == pytest.approx(1.345828, abs=2e-4)

    def test_np32_defect_is_essentially_n_independent(self) -> None:
        """Sanguinetti et al. 2009 report delta(n) in [2.6414, 2.6416]
        across n=36-63 (their Table 1); the fitted (delta0, delta2) form
        should land in that same narrow band even extrapolated to n=33.
        """
        n_star = rcr.effective_quantum_number(33, rcr.RB85_NP32_QUANTUM_DEFECT)
        delta = 33.0 - n_star
        assert 2.640 < delta < 2.643

    def test_fixed_point_iteration_converges(self) -> None:
        n_star_20 = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT, iterations=20)
        n_star_5 = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT, iterations=5)
        assert n_star_20 == pytest.approx(n_star_5, abs=1e-10)


# ---------------------------------------------------------------------------
# Section C: mu_RF, Numerov cross-check (dossier risk 2)
# ---------------------------------------------------------------------------


class TestMuRfDerivation:
    def test_numerov_reproduces_sedlacek_within_stated_tolerance(self) -> None:
        """Sedlacek et al. 2012 (arXiv:1205.4461, page 5) publish a
        quantum-defect-derived mu_RF = 1.37e-26 C.m for the kinematically
        identical Rb 53D5/2 -> 54P3/2 transition. This module's
        independent Numerov calculation, using the same class of
        quantum-defect (pure-Coulomb-tail) approximation, is checked
        against that published value at a stated, wide (but non-vacuous)
        factor-of-2 tolerance: it uses a different numerical method
        (direct radial integration versus their own closed-form/fitted
        route) and the same known pure-Coulomb approximation limitation
        (see :func:`rcr.numerov_radial_matrix_element`'s docstring), so
        exact agreement is not expected, but a sign error, wrong angular
        momentum, or a missing factor of the transition dipole formula
        would miss by much more than a factor of 2.
        """
        mu_rf, q_n = rcr.rf_transition_dipole_moment_from_quantum_defects(
            53, rcr.RB85_ND52_QUANTUM_DEFECT, 2, 54, rcr.RB85_NP32_QUANTUM_DEFECT, 1
        )
        assert q_n > 0.0
        published = 1.37e-26
        assert 0.5 * published < mu_rf < 2.0 * published

    def test_mu_rf_numerov_cross_check_against_registry(self) -> None:
        """The registry mu_RF (backed out from Holloway's own Fig. 15
        pairs) and this module's independent Numerov calculation for the
        SAME transition (32D5/2 -> 33P3/2) should agree to the same
        wide, disclosed tolerance as the Sedlacek cross-check above.
        """
        mu_rf, _ = rcr.rf_transition_dipole_moment_from_quantum_defects(
            32, rcr.RB85_ND52_QUANTUM_DEFECT, 2, 33, rcr.RB85_NP32_QUANTUM_DEFECT, 1
        )
        registry = rcr.RB85_MU_RF_32D52_33P32_C_M
        assert 0.5 * registry < mu_rf < 2.0 * registry

    def test_mu_rf_scales_up_with_n(self) -> None:
        """Radial matrix elements between adjacent Rydberg states grow
        with n (roughly as n_star^2); a sign or normalization bug that
        instead flattens or inverts this trend would be a red flag.
        """
        mu_low, _ = rcr.rf_transition_dipole_moment_from_quantum_defects(
            20, rcr.RB85_ND52_QUANTUM_DEFECT, 2, 21, rcr.RB85_NP32_QUANTUM_DEFECT, 1
        )
        mu_high, _ = rcr.rf_transition_dipole_moment_from_quantum_defects(
            53, rcr.RB85_ND52_QUANTUM_DEFECT, 2, 54, rcr.RB85_NP32_QUANTUM_DEFECT, 1
        )
        assert mu_high > mu_low


# ---------------------------------------------------------------------------
# Section D: C3 calibration KA -- Holloway Fig. 15
# ---------------------------------------------------------------------------

#: Holloway et al. 2014 Fig. 15's three printed (Delta_f Hz, |E| V/m)
#: pairs, verified against the arXiv:1405.7066 PDF page 8 image directly
#: this session.
HOLLOWAY_FIG15_PAIRS = [(4.35e6, 0.89), (20.09e6, 4.09), (48.31e6, 9.83)]

#: C3 tolerance: Holloway et al. 2014 state mu_RF "can be determined to
#: less than 0.1%" from the quantum-defect method (their Sec. VI) and
#: separately flag an open, unquantified RF-standing-wave uncertainty in
#: the same section, so this check does not claim tighter than what the
#: source itself claims to control. 1% is loose enough to accommodate
#: both, and tight enough that a doubled coefficient or a dropped
#: Doppler factor (a ~60% or larger effect) fails it immediately.
C3_TOLERANCE = 0.01


class TestC3CalibrationKA:
    """arithmetic_reproduction: Holloway et al. 2014 Fig. 15, all three pairs."""

    @pytest.mark.parametrize("delta_f_hz,field_v_per_m", HOLLOWAY_FIG15_PAIRS)
    def test_reproduces_published_pair(self, delta_f_hz: float, field_v_per_m: float) -> None:
        predicted_df = rcr.autler_townes_splitting_hz(
            rcr.RB85_MU_RF_32D52_33P32_C_M,
            field_v_per_m,
            rcr.HOLLOWAY_LAMBDA_PROBE_M,
            rcr.HOLLOWAY_LAMBDA_COUPLING_M,
        )
        rel_err = abs(predicted_df - delta_f_hz) / delta_f_hz
        assert rel_err < C3_TOLERANCE

    def test_eq12_field_form_is_exact_algebraic_inverse(self) -> None:
        """`field_from_at_splitting_v_per_m` and `autler_townes_splitting_hz`
        must be exact inverses of each other (both transcribe the same Eq.
        12); a real reproduction check needs this to hold before it means
        anything.
        """
        df = 2.5e7
        field = rcr.field_from_at_splitting_v_per_m(
            df,
            rcr.RB85_MU_RF_32D52_33P32_C_M,
            rcr.HOLLOWAY_LAMBDA_PROBE_M,
            rcr.HOLLOWAY_LAMBDA_COUPLING_M,
        )
        df_back = rcr.autler_townes_splitting_hz(
            rcr.RB85_MU_RF_32D52_33P32_C_M,
            field,
            rcr.HOLLOWAY_LAMBDA_PROBE_M,
            rcr.HOLLOWAY_LAMBDA_COUPLING_M,
        )
        assert df_back == pytest.approx(df, rel=1e-12)

    def test_wrong_doppler_direction_fails_the_check(self) -> None:
        """The gate's own deliberate-break discipline: using the reciprocal
        (Sedlacek prose) Doppler direction instead of the resolved
        (Holloway/Mohapatra) one must fail this reproduction, confirming
        C3 actually discriminates between the two candidate forms.
        """
        _, field_v_per_m = HOLLOWAY_FIG15_PAIRS[2]
        delta_f_hz = HOLLOWAY_FIG15_PAIRS[2][0]
        omega_rf = rcr.RB85_MU_RF_32D52_33P32_C_M * field_v_per_m / constants.HBAR
        bare_hz = omega_rf / (2.0 * math.pi)
        wrong_direction_df = bare_hz * (
            rcr.HOLLOWAY_LAMBDA_PROBE_M / rcr.HOLLOWAY_LAMBDA_COUPLING_M
        )
        rel_err = abs(wrong_direction_df - delta_f_hz) / delta_f_hz
        assert rel_err > C3_TOLERANCE


# ---------------------------------------------------------------------------
# Section E: C4 polarizability KA
# ---------------------------------------------------------------------------


class TestC4PolarizabilityKA:
    def test_alpha0_power_law_fit_reproduces_inputs(self) -> None:
        """Both the theory and experiment power-law fits should reproduce
        their own three tabulated Rb-85 nD5/2 rows (n=30,35,50) to better
        than 5%, confirming the n_star^p scaling is a good description
        over this range before it is extrapolated to n=32.
        """
        for source in ("theory", "experiment"):
            c, p = rcr._fit_alpha0_power_law(source)
            for n, row in rcr.RB85_ND52_ALPHA0_TABULATED[source].items():
                n_star = rcr.effective_quantum_number(n, rcr.RB85_ND52_QUANTUM_DEFECT)
                predicted = c * n_star**p
                assert predicted == pytest.approx(row.alpha0_au, rel=0.05)

    def test_fitted_exponent_is_near_the_rydberg_n7_scaling(self) -> None:
        """The scalar polarizability of a Rydberg state scales
        approximately as n_star^7 (Gallagher, Rydberg Atoms, sec. 2.4);
        the fitted exponent from three real tabulated points should land
        in the neighborhood of that value, not somewhere unrelated.
        """
        for source in ("theory", "experiment"):
            _, p = rcr._fit_alpha0_power_law(source)
            assert 5.5 < p < 8.5

    def test_theory_and_experiment_alpha0_agree_within_5_percent(self) -> None:
        """C4's own pass criterion: the two independent sources' TABULATED
        values (not the n=32 extrapolation) agree at the 1-5% level
        (dossier Sec. 2e), stated explicitly here per state.
        """
        for n in (30, 35, 50):
            theory = rcr.RB85_ND52_ALPHA0_TABULATED["theory"][n].alpha0_au
            experiment = rcr.RB85_ND52_ALPHA0_TABULATED["experiment"][n].alpha0_au
            rel_diff = abs(theory - experiment) / experiment
            assert rel_diff < 0.05

    def test_derived_32d52_alpha0_is_between_the_tabulated_neighbors(self) -> None:
        """A sign or scaling bug in the fit would likely place the n=32
        prediction outside the range bracketed by its n=30 and n=35
        neighbors; a physically sound power-law interpolation should not.
        """
        alpha0_30 = rcr.RB85_ND52_ALPHA0_TABULATED["theory"][30].alpha0_au
        alpha0_35 = rcr.RB85_ND52_ALPHA0_TABULATED["theory"][35].alpha0_au
        assert alpha0_30 < rcr.RB85_32D52_ALPHA0_AU < alpha0_35


# ---------------------------------------------------------------------------
# Section F: quadratic Stark shift and its validity guard
# ---------------------------------------------------------------------------


class TestQuadraticStarkShift:
    def test_shift_is_negative_for_positive_field(self) -> None:
        shift = rcr.rydberg_quadratic_stark_shift_hz(rcr.RB85_32D52_ALPHA0_AU, 10.0, 30.65)
        assert shift < 0.0

    def test_shift_scales_as_field_squared(self) -> None:
        n_star = 30.65
        shift_1 = rcr.rydberg_quadratic_stark_shift_hz(rcr.RB85_32D52_ALPHA0_AU, 5.0, n_star)
        shift_2 = rcr.rydberg_quadratic_stark_shift_hz(rcr.RB85_32D52_ALPHA0_AU, 10.0, n_star)
        assert shift_2 == pytest.approx(4.0 * shift_1, rel=1e-10)

    def test_zero_field_gives_zero_shift(self) -> None:
        assert rcr.rydberg_quadratic_stark_shift_hz(rcr.RB85_32D52_ALPHA0_AU, 0.0, 30.65) == 0.0

    def test_validity_guard_triggers_below_inglis_teller(self) -> None:
        n_star = 30.65
        e_it = rcr.inglis_teller_field_v_per_m(n_star)
        # Just inside the margin: should pass.
        rcr.rydberg_quadratic_stark_shift_hz(
            rcr.RB85_32D52_ALPHA0_AU, 0.9 * rcr.STARK_VALIDITY_MARGIN * e_it, n_star
        )
        # Just outside the margin: should raise.
        with pytest.raises(rcr.RydbergStarkValidityError):
            rcr.rydberg_quadratic_stark_shift_hz(
                rcr.RB85_32D52_ALPHA0_AU, 1.1 * rcr.STARK_VALIDITY_MARGIN * e_it, n_star
            )

    def test_guard_margin_is_below_the_full_inglis_teller_field(self) -> None:
        """The house pattern is to enforce validity windows with margin,
        not at the boundary itself.
        """
        assert rcr.STARK_VALIDITY_MARGIN < 1.0


# ---------------------------------------------------------------------------
# Section G: ladder susceptibility formula transcription (C2)
# ---------------------------------------------------------------------------


class TestLadderSusceptibility:
    def test_reduces_to_finite_three_level_form_when_rf_is_zero(self) -> None:
        """Setting Omega_RF=0 (via e_rf_v_per_m=0) must not raise or
        produce NaN/inf: the D14 numerator term and the D12*Omega_RF^2
        denominator term both vanish algebraically (module docstring),
        leaving the finite 3-level pole structure.
        """
        system = _default_system()
        delta_p = np.linspace(-2.0 * math.pi * 20e6, 2.0 * math.pi * 20e6, 501)
        chi = rcr.ladder_susceptibility(delta_p, 0.0, 0.0, 1.0, 100.0, 0.0, system)
        assert np.all(np.isfinite(chi))
        assert np.max(chi.imag) > 0.0

    def test_probe_field_parameter_does_not_change_result(self) -> None:
        """The prefactor is algebraically independent of |E_p| (module
        docstring); passing different probe field magnitudes must give
        identical chi.
        """
        system = _default_system()
        delta_p = np.linspace(-2.0 * math.pi * 20e6, 2.0 * math.pi * 20e6, 51)
        chi_1 = rcr.ladder_susceptibility(delta_p, 0.0, 0.0, 1.0, 100.0, 5.0, system)
        chi_2 = rcr.ladder_susceptibility(delta_p, 0.0, 0.0, 999.0, 100.0, 5.0, system)
        assert np.array_equal(chi_1, chi_2)

    def test_susceptibility_symmetric_for_zero_detunings(self) -> None:
        """At delta_c=delta_rf=0, Im[chi](delta_p) must be symmetric about
        delta_p=0 (the resonance condition is even in delta_p when both
        other detunings vanish): a basic structural check on the
        transcribed formula.
        """
        system = _default_system()
        delta_p = np.linspace(-2.0 * math.pi * 30e6, 2.0 * math.pi * 30e6, 201)
        chi = rcr.ladder_susceptibility(delta_p, 0.0, 0.0, 1.0, 100.0, 5.0, system)
        np.testing.assert_allclose(chi.imag, chi.imag[::-1], atol=1e-20, rtol=1e-8)


# ---------------------------------------------------------------------------
# Section H: Doppler averaging (C7)
# ---------------------------------------------------------------------------


class TestDopplerAveraging:
    def test_hot_atom_limit_matches_stationary_atom_at_zero_velocity_weight(self) -> None:
        """Feeding a single (zero) velocity with unit weight must
        reproduce the plain (non-Doppler-averaged) susceptibility exactly.
        """
        system = _default_system()
        delta_p = np.linspace(-2.0 * math.pi * 20e6, 2.0 * math.pi * 20e6, 51)
        chi_bare = rcr.ladder_susceptibility(delta_p, 0.0, 0.0, 1.0, 100.0, 5.0, system)

        # Monkey-patch the velocity grid to a single point at v=0.
        import cliffordclock.integrator.rydberg_cell_response as module

        original = module.doppler_velocity_grid
        module.doppler_velocity_grid = lambda *_a, **_k: (np.array([0.0]), np.array([1.0]))
        try:
            chi_doppler = rcr.doppler_averaged_susceptibility(
                delta_p, 0.0, 0.0, 1.0, 100.0, 5.0, system, 320.0, RB85_MASS_KG, n_velocity_points=1
            )
        finally:
            module.doppler_velocity_grid = original
        np.testing.assert_allclose(chi_doppler, chi_bare, rtol=1e-10)

    def test_at_splitting_survives_doppler_averaging_at_the_right_scale(self) -> None:
        """C7: the full Doppler-averaged 4-level spectrum's extracted
        Autler-Townes doublet spacing should land within a stated,
        non-vacuous band of the closed-form (lambda_c/lambda_p)*Omega_RF/2pi
        limit (:func:`autler_townes_splitting_hz`). Finite decay rates and
        thermal (Doppler) averaging pull the numerically resolved peak
        spacing in below the idealized zero-linewidth value (standard
        Autler-Townes line-pulling); this test's 0.6-1.0 band is wide
        enough to accommodate that and still tight enough to catch a
        wrong-direction Doppler factor, which would move the analytic
        target by roughly (lambda_p/lambda_c)^2 ~= 2.6x and put the ratio
        far outside this band (see the reciprocal-direction check in
        TestC3CalibrationKA).
        """
        from scipy.signal import find_peaks

        system = _default_system()
        field_v_per_m = 20.0
        analytic = rcr.autler_townes_splitting_hz(
            system.mu_rf_c_m, field_v_per_m, system.wavelength_probe_m, system.wavelength_coupling_m
        )
        window_hz = 1.5 * analytic
        delta_p = np.linspace(-2.0 * math.pi * window_hz, 2.0 * math.pi * window_hz, 12001)
        spectrum = rcr.doppler_averaged_susceptibility(
            delta_p,
            0.0,
            0.0,
            1.0,
            200.0,
            field_v_per_m,
            system,
            320.0,
            RB85_MASS_KG,
            n_velocity_points=89,
        )
        im = spectrum.imag
        peaks, props = find_peaks(im, prominence=np.max(im) * 0.02)
        assert len(peaks) >= 2
        order = np.argsort(-props["prominences"])[:2]
        idxs = sorted(peaks[order])
        numeric_splitting = (delta_p[idxs[1]] - delta_p[idxs[0]]) / (2.0 * math.pi)
        ratio = numeric_splitting / analytic
        assert 0.6 < ratio < 1.0


# ---------------------------------------------------------------------------
# Section I: C5 limit kill-tests
# ---------------------------------------------------------------------------


class TestC5LimitKillTests:
    def _spectrum(
        self, system: rcr.LadderSystem, fields: np.ndarray, delta_p: np.ndarray
    ) -> np.ndarray:
        weights = np.ones_like(fields)
        return rcr.compose_inhomogeneous_eit_spectrum(
            delta_p, fields, weights, rcr.RB85_32D52_ALPHA0_AU, 30.65, system
        )

    def test_zero_field_is_byte_identical_to_unperturbed_line(self) -> None:
        system = _default_system()
        delta_p = np.linspace(-2.0 * math.pi * 20e6, 2.0 * math.pi * 20e6, 501)
        fields = np.zeros(5)
        composed = self._spectrum(system, fields, delta_p)
        unperturbed = rcr.doppler_averaged_susceptibility(
            delta_p, 0.0, 0.0, 1.0, 1.0, 0.0, system, 320.0, RB85_MASS_KG, n_velocity_points=33
        )
        assert np.array_equal(composed, unperturbed)

    def test_uniform_field_is_a_pure_shift_with_zero_added_width(self) -> None:
        system = _default_system()
        delta_p = np.linspace(-2.0 * math.pi * 20e6, 2.0 * math.pi * 20e6, 501)
        fields = np.full(5, 40.0)
        composed = self._spectrum(system, fields, delta_p)

        shift_hz = rcr.rydberg_quadratic_stark_shift_hz(rcr.RB85_32D52_ALPHA0_AU, 40.0, 30.65)
        expected = rcr.doppler_averaged_susceptibility(
            delta_p,
            2.0 * math.pi * shift_hz,
            0.0,
            1.0,
            1.0,
            0.0,
            system,
            320.0,
            RB85_MASS_KG,
            n_velocity_points=33,
        )
        assert np.array_equal(composed, expected)

        # "Zero added width": the FWHM of Im[chi] under the uniform field
        # equals the FWHM of the unperturbed line to float precision (a
        # pure translation of an unchanged lineshape).
        unperturbed = rcr.doppler_averaged_susceptibility(
            delta_p, 0.0, 0.0, 1.0, 1.0, 0.0, system, 320.0, RB85_MASS_KG, n_velocity_points=33
        )

        def fwhm(y: np.ndarray, x: np.ndarray) -> float:
            half = np.max(y) / 2.0
            above = np.where(y >= half)[0]
            return float(x[above[-1]] - x[above[0]])

        assert fwhm(composed.imag, delta_p) == pytest.approx(
            fwhm(unperturbed.imag, delta_p), rel=1e-9
        )

    def test_deliberate_sign_flip_breaks_the_zero_field_kill_test(self) -> None:
        """The gate's own deliberate-break discipline (C5): flipping the
        Stark-shift sign must NOT be silently absorbed by the zero-field
        check, since the zero-field spectrum has no shift to flip. This
        test instead confirms the kill-test's OTHER half is armed: a
        flipped-sign shift changes the uniform-field spectrum
        (demonstrating the check is sensitive to the sign at all).
        """
        system = _default_system()
        delta_p = np.linspace(-2.0 * math.pi * 20e6, 2.0 * math.pi * 20e6, 501)
        field = 40.0
        n_star = 30.65
        shift_hz = rcr.rydberg_quadratic_stark_shift_hz(rcr.RB85_32D52_ALPHA0_AU, field, n_star)
        flipped_shift_hz = -shift_hz
        assert shift_hz != flipped_shift_hz

        correct = rcr.doppler_averaged_susceptibility(
            delta_p,
            2.0 * math.pi * shift_hz,
            0.0,
            1.0,
            1.0,
            0.0,
            system,
            320.0,
            RB85_MASS_KG,
            n_velocity_points=33,
        )
        flipped = rcr.doppler_averaged_susceptibility(
            delta_p,
            2.0 * math.pi * flipped_shift_hz,
            0.0,
            1.0,
            1.0,
            0.0,
            system,
            320.0,
            RB85_MASS_KG,
            n_velocity_points=33,
        )
        assert not np.array_equal(correct, flipped)

    def test_deliberate_doubled_coefficient_breaks_uniform_field_match(self) -> None:
        """Doubling alpha0 must move the uniform-field composed spectrum
        away from the correctly-shifted reference (the gate's own
        "doubled coefficient" deliberate break, C5).
        """
        system = _default_system()
        delta_p = np.linspace(-2.0 * math.pi * 20e6, 2.0 * math.pi * 20e6, 501)
        field = 40.0
        fields = np.full(5, field)
        weights = np.ones_like(fields)
        correct = rcr.compose_inhomogeneous_eit_spectrum(
            delta_p, fields, weights, rcr.RB85_32D52_ALPHA0_AU, 30.65, system
        )
        doubled = rcr.compose_inhomogeneous_eit_spectrum(
            delta_p, fields, weights, 2.0 * rcr.RB85_32D52_ALPHA0_AU, 30.65, system
        )
        assert not np.array_equal(correct, doubled)


# ---------------------------------------------------------------------------
# Section J: inhomogeneity demonstrator (C6)
# ---------------------------------------------------------------------------


class TestInhomogeneityDemonstrator:
    def test_patch_field_falls_off_with_distance(self) -> None:
        patches = [rcr.WallPatch(position_m=np.array([0.0125, 0.0, 0.0]), charge_c=1.0e-15)]
        near = np.linalg.norm(rcr.patch_field_v_per_m(np.array([0.011, 0.0, 0.0]), patches))
        far = np.linalg.norm(rcr.patch_field_v_per_m(np.array([0.0, 0.0, 0.0]), patches))
        assert near > far > 0.0

    def test_per_atom_shifts_carry_genuine_spread_under_an_inhomogeneous_field(self) -> None:
        """The mechanism behind the C6 phenomenology target (Patrick et
        al. 2025: line shift plus asymmetric broadening growing with
        patch field): an inhomogeneous field gives each atom its own
        Stark-shifted resonance
        (:func:`rydberg_quadratic_stark_shift_hz`), so the population of
        per-atom shifts feeding
        :func:`compose_inhomogeneous_eit_spectrum` has real spread,
        unlike the uniform-field case (C5), where every atom shares
        exactly one shift.
        """
        rng = np.random.default_rng(20260902)
        n_atoms = 40
        mean_field = 100.0
        spread_fields = mean_field * rng.uniform(0.2, 1.8, n_atoms)
        spread_shifts = np.array(
            [
                rcr.rydberg_quadratic_stark_shift_hz(rcr.RB85_32D52_ALPHA0_AU, f, 30.65)
                for f in spread_fields
            ]
        )
        uniform_shifts = np.array(
            [
                rcr.rydberg_quadratic_stark_shift_hz(rcr.RB85_32D52_ALPHA0_AU, mean_field, 30.65)
                for _ in range(n_atoms)
            ]
        )
        assert np.std(spread_shifts) > 0.0
        assert np.std(uniform_shifts) == 0.0
        assert np.std(spread_shifts) > 1e5  # Hz: a spread well above float noise, not vacuous

    def test_inhomogeneous_composed_spectrum_differs_from_zero_and_uniform(self) -> None:
        """The composed spectrum under a spread of per-atom fields is
        neither the unperturbed line nor a single pure shift of it: the
        notebook (``notebooks/16_rydberg_cell_response.ipynb``) plots
        this spectrum directly, at high resolution around the EIT
        feature, to show the shift-plus-broadening phenomenology this
        test's coarse full-window statistics cannot resolve reliably
        (the Doppler-broadened two-level background dominates a full-
        window variance estimate at these field scales).
        """
        system = _default_system()
        delta_p = np.linspace(-2.0 * math.pi * 20e6, 2.0 * math.pi * 20e6, 1001)
        rng = np.random.default_rng(20260902)
        n_atoms = 40
        mean_field = 100.0
        spread_fields = mean_field * rng.uniform(0.2, 1.8, n_atoms)
        weights = np.ones(n_atoms)
        e_coupling_v_per_m = 662.0

        composed = rcr.compose_inhomogeneous_eit_spectrum(
            delta_p,
            spread_fields,
            weights,
            rcr.RB85_32D52_ALPHA0_AU,
            30.65,
            system,
            e_coupling_v_per_m=e_coupling_v_per_m,
        )
        uniform = rcr.compose_inhomogeneous_eit_spectrum(
            delta_p,
            np.full(n_atoms, mean_field),
            weights,
            rcr.RB85_32D52_ALPHA0_AU,
            30.65,
            system,
            e_coupling_v_per_m=e_coupling_v_per_m,
        )
        zero = rcr.compose_inhomogeneous_eit_spectrum(
            delta_p,
            np.zeros(n_atoms),
            weights,
            rcr.RB85_32D52_ALPHA0_AU,
            30.65,
            system,
            e_coupling_v_per_m=e_coupling_v_per_m,
        )
        assert not np.array_equal(composed, zero)
        assert not np.array_equal(composed, uniform)

    def test_evidentiary_class_is_computable_comparison(self) -> None:
        """Pinned per this project's evidentiary-class discipline: the
        wall-patch demonstrator reproduces Patrick et al. 2025's
        qualitative phenomenology (shift, asymmetric broadening growing
        with patch field and shrinking cell size), not a quantitative
        reproduction of any of their printed numbers (dossier Sec. 3: the
        paper's field-vs-power and EIT-vs-wavelength curves are
        digitizable-axis figures, not printed numeric tables). The class
        label lives beside the benchmark artifact this pins against.
        """
        import json
        from pathlib import Path

        results_path = (
            Path(__file__).resolve().parent.parent
            / "benchmarks"
            / "results"
            / "wp39_surface_charge_demonstrator.json"
        )
        if not results_path.exists():
            pytest.skip("benchmark artifact not yet generated")
        data = json.loads(results_path.read_text())
        assert data["evidentiary_class"] == "computable_comparison"
