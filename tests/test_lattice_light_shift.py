# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP36 Phase 1 lattice-light-shift models (CONVENTIONS.md
section 17, E40/E41): both models' formulas, the BO+WKB numerics'
convergence guards and harmonic-limit consistency, and the benchmark
cases' regression pins.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

import cliffordclock.integrator.lattice_light_shift as lls
from cliffordclock.constants import SPEED_OF_LIGHT
from cliffordclock.ensemble.species import get_species

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import run_lattice_light_shift as lls_benchmark  # noqa: E402

# ---------------------------------------------------------------------------
# Model A: harmonic/operational model
# ---------------------------------------------------------------------------


class TestReductionFactors:
    def test_ushijima_and_jila_agree_to_leading_order(self) -> None:
        """The linear (Ushijima) and reciprocal (JILA) reduction factors
        agree to leading order in `j*kB*Tr/(u*E_R) << 1` (CONVENTIONS.md
        E40): `(1+x)^-1 ~= 1-x` for small `x`.
        """
        e_r = lls.recoil_energy_j(813e-9, 87 * 1.66053906892e-27)
        u = 500.0  # deep, so kB*Tr/(u*E_R) is small even at a few uK
        tr_k = 1e-6
        linear = lls.ushijima_reduction_factor(u, 1.0, tr_k, e_r)
        reciprocal = lls.jila_reduction_factor(u, 1.0, tr_k, e_r)
        assert linear == pytest.approx(reciprocal, abs=2e-3)

    def test_reduction_factors_differ_at_higher_order(self) -> None:
        """The two forms are NOT interchangeable away from the small-`x`
        limit: at a shallow, hot condition they diverge measurably.
        """
        e_r = lls.recoil_energy_j(813e-9, 87 * 1.66053906892e-27)
        u = 5.0
        tr_k = 5e-6
        linear = lls.ushijima_reduction_factor(u, 1.0, tr_k, e_r)
        reciprocal = lls.jila_reduction_factor(u, 1.0, tr_k, e_r)
        assert abs(linear - reciprocal) > 1e-3

    def test_reduction_factor_rejects_nonpositive_u(self) -> None:
        with pytest.raises(ValueError):
            lls.ushijima_reduction_factor(0.0, 1.0, 1e-6, 1e-30)
        with pytest.raises(ValueError):
            lls.jila_reduction_factor(-1.0, 1.0, 1e-6, 1e-30)


class TestHarmonicLightShift:
    def test_rejects_nonpositive_u(self) -> None:
        with pytest.raises(ValueError):
            lls.harmonic_light_shift_hz(0.0, 0.0, 0.0, lls.USHIJIMA_2018_SR87)

    def test_rejects_negative_n_z(self) -> None:
        with pytest.raises(ValueError):
            lls.harmonic_light_shift_hz(10.0, 0.0, -1.0, lls.USHIJIMA_2018_SR87)

    def test_requires_thermal_args_when_reduction_requested(self) -> None:
        with pytest.raises(ValueError):
            lls.harmonic_light_shift_hz(
                10.0, 0.0, 0.0, lls.USHIJIMA_2018_SR87, reduction_form="ushijima_linear"
            )

    def test_zero_beta_and_m1e2_leaves_only_e1_term(self) -> None:
        """A sanity check on Eq. 1's algebra: with `alpha~qm=beta~=0`,
        only the E1-slope terms survive, and `nu_LS` is then exactly
        proportional to `delta_L` (the entire M1E2/hyperpolarizability
        physics is gone).
        """
        coeffs = lls.HarmonicLatticeCoefficients(
            e1_slope_per_hz=2e-11,
            e1_slope_per_hz_uncertainty=0.0,
            m1e2_hz=0.0,
            m1e2_hz_uncertainty=0.0,
            hyperpolarizability_hz=0.0,
            hyperpolarizability_hz_uncertainty=0.0,
        )
        u, n_z = 50.0, 0.0
        shift_1 = lls.harmonic_light_shift_hz(u, 1.0e6, n_z, coeffs)
        shift_2 = lls.harmonic_light_shift_hz(u, 2.0e6, n_z, coeffs)
        # Linear in delta_L when beta~=alpha~qm=0: shift_2 == 2*shift_1.
        assert shift_2 == pytest.approx(2.0 * shift_1, rel=1e-10, abs=0)

    def test_uncertainty_zero_when_all_coefficient_uncertainties_zero(self) -> None:
        coeffs = lls.HarmonicLatticeCoefficients(
            e1_slope_per_hz=2e-11,
            e1_slope_per_hz_uncertainty=0.0,
            m1e2_hz=-1e-3,
            m1e2_hz_uncertainty=0.0,
            hyperpolarizability_hz=-1e-7,
            hyperpolarizability_hz_uncertainty=0.0,
        )
        unc = lls.harmonic_light_shift_uncertainty_hz(50.0, 1e6, 0.0, coeffs)
        assert unc == 0.0

    def test_uncertainty_propagation_matches_finite_difference_by_hand(self) -> None:
        """Cross-check `harmonic_light_shift_uncertainty_hz` against an
        independently hand-rolled finite-difference propagation (not the
        same code path, catching a copy-paste bug in the engine's own
        internal finite-difference logic).
        """
        coeffs = lls.USHIJIMA_2018_SR87
        u, detuning, n_z = 72.0, 5.3e6, 0.0
        engine_unc = lls.harmonic_light_shift_uncertainty_hz(u, detuning, n_z, coeffs)

        def replace(field: str, value: float) -> lls.HarmonicLatticeCoefficients:
            return lls._replace_field(coeffs, field, value)

        contributions = []
        for field, sigma in (
            ("e1_slope_per_hz", coeffs.e1_slope_per_hz_uncertainty),
            ("m1e2_hz", coeffs.m1e2_hz_uncertainty),
            ("hyperpolarizability_hz", coeffs.hyperpolarizability_hz_uncertainty),
        ):
            value = getattr(coeffs, field)
            step = value * 1e-6
            plus = lls.harmonic_light_shift_hz(u, detuning, n_z, replace(field, value + step))
            minus = lls.harmonic_light_shift_hz(u, detuning, n_z, replace(field, value - step))
            partial = (plus - minus) / (2 * step)
            contributions.append((partial * sigma) ** 2)
        hand_unc = math.sqrt(sum(contributions))
        assert engine_unc == pytest.approx(hand_unc, rel=1e-6, abs=0)


class TestOperationalPointSolve:
    def test_reproduces_ushijima_2018_operational_point(self) -> None:
        """Target 1 (CONVENTIONS.md E40): solving Ushijima et al. 2018's
        own Eq. 1 for `n_z=0` reproduces their published
        `u_op=72(2) E_R` at `delta_L_op=5.3(2) MHz`."""
        result = lls.solve_harmonic_operational_point(lls.USHIJIMA_2018_SR87, n_z=0.0)
        assert result.u_op == pytest.approx(72.0, abs=2.0)
        assert result.detuning_hz_op == pytest.approx(5.3e6, abs=0.2e6)
        assert abs(result.residual_shift_hz) < 1.0
        assert abs(result.residual_slope_hz) < 1.0

    def test_rejects_invalid_bracket(self) -> None:
        with pytest.raises(ValueError):
            lls.solve_harmonic_operational_point(lls.USHIJIMA_2018_SR87, u_bracket=(-1.0, 10.0))


# ---------------------------------------------------------------------------
# Model B: BO+WKB numerics
# ---------------------------------------------------------------------------


class TestAxialFiniteDifferenceSolver:
    def test_harmonic_potential_matches_analytic_qho_spectrum(self) -> None:
        """The finite-difference solver, fed the pure harmonic potential
        `v(x)=depth*x^2`, must recover the exact 1D quantum-harmonic-
        oscillator spectrum `E_n/E_R = 2*sqrt(depth)*(n+1/2)`
        (CONVENTIONS.md E41's harmonic-limit consistency check on the
        solver itself, independent of the Eq. 4/Eq. 11 algebraic check).
        """
        depth = 50.0
        energies = lls.axial_energies_er(depth, 3, potential="harmonic")
        expected = np.array([2.0 * math.sqrt(depth) * (n + 0.5) - depth for n in range(3)])
        # axial_energies_er's "harmonic" potential is v=-depth+depth*x^2,
        # so E_n = -depth + 2*sqrt(depth)*(n+1/2).
        np.testing.assert_allclose(energies, expected, atol=1e-3)

    def test_cos2_ground_state_deeper_than_harmonic_approximation(self) -> None:
        """The true `cos^2` site potential's ground state is DEEPER than
        the harmonic (small-angle) approximation at the same nominal
        depth: `cos^2(x) ~= 1 - x^2 + x^4/3` near the well bottom is an
        ATTRACTIVE quartic correction, the physically correct direction
        (CONVENTIONS.md E41's harmonic-limit note)."""
        depth = 50.0
        e_cos2 = lls.axial_energies_er(depth, 1, potential="cos2")[0]
        e_harmonic = lls.axial_energies_er(depth, 1, potential="harmonic")[0]
        assert e_cos2 < e_harmonic

    def test_convergence_guard_raises_when_max_grid_too_small(self) -> None:
        """Kill test: an unreasonably tight tolerance combined with a
        `max_grid_n` too small to ever reach it must raise
        `LatticeLightShiftConvergenceError`, never silently return an
        unconverged result."""
        with pytest.raises(lls.LatticeLightShiftConvergenceError):
            lls.axial_energies_er(50.0, 2, tol_er=1e-14, grid_n0=17, max_grid_n=33)

    def test_zero_or_negative_local_depth_gives_zero_energies(self) -> None:
        energies = lls.axial_energies_er(0.0, 3)
        assert np.all(energies == 0.0)


class TestDensityOfStatesConsistency:
    """CONVENTIONS.md E41's central self-check, mirroring Beloy et al.
    2020's own Section VI: the general Eq. 11 density of states must
    reduce to the closed-form harmonic Eq. 4, both algebraically (feeding
    the closed-form turning radius into Eq. 11 directly) and numerically
    (running the full finite-difference/turning-radius machinery with
    `potential="harmonic"`).
    """

    def _site(self) -> lls.SitePotential:
        mass = 171 * 1.66053906892e-27
        return lls.make_site_potential(
            depth_er=50.0, waist_m=50e-6, wavelength_m=759e-9, mass_kg=mass
        )

    def test_numeric_bo_wkb_matches_closed_form_in_harmonic_limit(self) -> None:
        site = self._site()
        er = site.recoil_energy_j_value
        for n_z in (0, 1):
            e0 = lls.axial_band_energy_er(site, n_z, 0.0, potential="harmonic") * er
            for frac in (0.2, 0.5, 0.9):
                energy_j = e0 * (1.0 - frac)
                closed = lls.harmonic_density_of_states_closed_form(site, n_z, energy_j)
                numeric = lls.bo_wkb_density_of_states(site, n_z, energy_j, potential="harmonic")
                assert numeric == pytest.approx(closed, rel=2e-3, abs=0)

    def test_true_potential_density_of_states_is_nonnegative_and_finite(self) -> None:
        site = self._site()
        er = site.recoil_energy_j_value
        e0 = lls.axial_band_energy_er(site, 0, 0.0) * er
        g = lls.bo_wkb_density_of_states(site, 0, 0.5 * e0)
        assert g >= 0.0
        assert math.isfinite(g)

    def test_density_of_states_zero_outside_band(self) -> None:
        site = self._site()
        er = site.recoil_energy_j_value
        assert lls.bo_wkb_density_of_states(site, 0, 1.0) == 0.0  # E > 0
        e0 = lls.axial_band_energy_er(site, 0, 0.0) * er
        assert lls.bo_wkb_density_of_states(site, 0, e0 * 2.0) == 0.0  # below band bottom


class TestThermalShapeFactors:
    """CONVENTIONS.md E41's thermally-averaged X/Y/Z factors, validated
    against Bothwell et al. 2025's own published BO+WKB column
    (Appendix A Table I)."""

    def test_nominal_limit_at_zero_temperature(self) -> None:
        """At `Tr=0` the radial ensemble collapses to a delta function at
        `rho=0`; `X`/`Z` should be close to 1 and `Y` close to 0 for a
        deep, cold ground-state band (Beloy's own stated "nominal"
        values)."""
        mass = 171 * 1.66053906892e-27
        site = lls.make_site_potential(
            depth_er=100.0, waist_m=50e-6, wavelength_m=759e-9, mass_kg=mass
        )
        factors = lls.axial_thermal_factors(site, 0, 0.0)
        assert factors.x_nz > 0.9
        assert factors.y_nz < 0.1
        assert factors.z_nz > 0.8

    def test_reproduces_bothwell_2025_table1_bowkb_column(self) -> None:
        """Target 3a (CONVENTIONS.md E41): reproduces two of Bothwell et
        al. 2025's own four published `(u0, Tr)` rows to better than 1%
        relative error (the full four-row sweep is exercised by
        `benchmarks/run_lattice_light_shift.py`; this test pins two rows
        directly as a standing regression)."""
        yb = get_species("Yb171")
        wavelength_m = SPEED_OF_LIGHT / 394_798_266.9e6
        rows = [
            (56.8, 650e-9, 0.785, 0.0608, 0.645),
            (112.2, 720e-9, 0.879, 0.0454, 0.781),
        ]
        for u0, tr_k, x_pub, y_pub, z_pub in rows:
            site = lls.make_site_potential(
                depth_er=u0, waist_m=50e-6, wavelength_m=wavelength_m, mass_kg=yb.mass_kg
            )
            factors = lls.axial_thermal_factors(site, 0, tr_k)
            assert factors.x_nz == pytest.approx(x_pub, rel=1e-2, abs=0)
            assert factors.y_nz == pytest.approx(y_pub, rel=1e-2, abs=0)
            assert factors.z_nz == pytest.approx(z_pub, rel=1e-2, abs=0)

    def test_x_y_z_independent_of_waist(self) -> None:
        """`X`/`Y`/`Z` depend only on `(n_z, u0, Tr)`, not on the lattice
        waist: `kappa` cancels exactly out of the defining ratio
        (CONVENTIONS.md E41's derivation note)."""
        mass = 171 * 1.66053906892e-27
        site_a = lls.make_site_potential(
            depth_er=56.8, waist_m=50e-6, wavelength_m=759e-9, mass_kg=mass
        )
        site_b = lls.make_site_potential(
            depth_er=56.8, waist_m=25e-6, wavelength_m=759e-9, mass_kg=mass
        )
        f_a = lls.axial_thermal_factors(site_a, 0, 650e-9)
        f_b = lls.axial_thermal_factors(site_b, 0, 650e-9)
        assert f_a.x_nz == pytest.approx(f_b.x_nz, rel=1e-6, abs=0)
        assert f_a.y_nz == pytest.approx(f_b.y_nz, rel=1e-6, abs=0)
        assert f_a.z_nz == pytest.approx(f_b.z_nz, rel=1e-6, abs=0)

    def test_rejects_negative_n_z(self) -> None:
        mass = 171 * 1.66053906892e-27
        site = lls.make_site_potential(
            depth_er=50.0, waist_m=50e-6, wavelength_m=759e-9, mass_kg=mass
        )
        with pytest.raises(ValueError):
            lls.axial_thermal_factors(site, -1, 1e-6)


class TestBoWkbFractionalLightShift:
    def test_requires_matching_depth(self) -> None:
        mass = 171 * 1.66053906892e-27
        site = lls.make_site_potential(
            depth_er=50.0, waist_m=50e-6, wavelength_m=759e-9, mass_kg=mass
        )
        with pytest.raises(ValueError):
            lls.bo_wkb_fractional_light_shift(
                0, 60.0, 0.0, 6e-7, lls.BOTHWELL_2025_YB171_BOWKB, site
            )

    def test_returns_finite_shift(self) -> None:
        mass = 171 * 1.66053906892e-27
        site = lls.make_site_potential(
            depth_er=50.0, waist_m=50e-6, wavelength_m=759e-9, mass_kg=mass
        )
        shift, factors = lls.bo_wkb_fractional_light_shift(
            0, 50.0, 0.0, 6e-7, lls.BOTHWELL_2025_YB171_BOWKB, site
        )
        assert math.isfinite(shift)
        assert 0.0 <= factors.x_nz <= 1.0
        assert 0.0 <= factors.y_nz <= 1.0
        assert 0.0 <= factors.z_nz <= 1.0


# ---------------------------------------------------------------------------
# Benchmark regression pins (run_lattice_light_shift.py)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBenchmarkRegressions:
    """Pins the full benchmark report's verdicts and headline numbers.
    Marked slow: the density-of-states contrast case's `scipy.integrate.quad`
    sweep (six radial temperatures, each requiring the convergence-guarded
    axial finite-difference solver at many quadrature-internal energy
    points) measured at ~28 s locally
    (`pytest tests/test_lattice_light_shift.py -m slow --durations=25`);
    `test_build_report_is_json_serializable` reruns the same case inside
    `build_report()` and measured ~30 s. Both are many times this file's
    fast-lane budget (its non-slow tests together run in ~3.5 s) and a
    several-x multiple of a shared CI runner's expected slowdown, so both
    stay out of the fast shard per the CI doctrine even though neither
    alone threatens the 15-minute fast-shard budget.
    """

    def test_target1_verdict_met(self) -> None:
        case = lls_benchmark.run_ushijima_operational_point_case()
        assert case.u_op_within_published_uncertainty
        assert case.detuning_op_within_published_uncertainty

    def test_target2_verdict_met(self) -> None:
        case = lls_benchmark.run_aeppli_lattice_line_item_case()
        assert case.kpi_verdict == "MET"

    def test_target3a_verdict_met(self) -> None:
        case = lls_benchmark.run_bothwell_table1_reproduction_case()
        assert case.kpi_verdict == "MET"
        assert case.worst_relative_error < 0.01

    def test_target3b_is_computable_comparison_not_arithmetic_reproduction(self) -> None:
        case = lls_benchmark.run_bothwell_headline_comparison_case()
        assert case.case_class == "computable_comparison"
        assert math.isfinite(case.model_difference_fractional)

    def test_density_of_states_contrast_ratio_grows_with_temperature(self) -> None:
        """The dossier's qualitative claim, checked as a monotonic trend:
        the BO+WKB/harmonic cumulative-state-count ratio should not
        decrease as radial temperature rises (E41's density-of-states
        contrast case)."""
        case = lls_benchmark.run_density_of_states_contrast_case()
        ratios = [row["ratio_cos2_over_harmonic"] for row in case.rows]
        assert all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:], strict=False))
        assert ratios[-1] > ratios[0]

    def test_build_report_is_json_serializable(self) -> None:
        import json

        report = lls_benchmark.build_report()
        json.dumps(report)  # raises on a non-serializable value
        assert report["target1_ushijima_operational_point"]["u_op_within_published_uncertainty"]
