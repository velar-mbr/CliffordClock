#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figure 8 (Al+ motional-Doppler progression): six E38 case variants
against Marshall et al.'s published secular-motion band, plus every quoted
number in the paper's E38/E39 validation subsections.

Calls the real ``benchmarks/run_motional_al_ion.py`` case builders (WP30
single-mass, WP31 mass-ratio participation, WP32 measured-spectrum-
reconstructed participation, WP33 participation-times-leading-order-
intrinsic-micromotion-enhancement, WP34 participation-times-exact-Floquet
enhancement, WP35 the coupled two-ion Floquet constrained fit), the same
case objects ``notebooks/13_trapped_ion_quantum_motion.ipynb`` sections 3-4
run live, so this script and that notebook can never silently disagree.

**Binding classification label: every one of the six cases below is an
`arithmetic_reproduction`** (``benchmarks/run_motional_al_ion.py``'s own
``case_class``, never ``reproducibility``/``blind_prediction``): Marshall et
al.'s published mode frequencies and n_bar values, and their own published
per-mode/total secular-motion rows, are what each case reproduces from an
independently published standard formula, not an independent measurement
this project supplied. None of the six join this paper's two-
reproducibility-case validation headline (Sec. sec:validation); each carries
its own class label wherever it is quoted.

Outputs
-------
- ``figures/fig8_motional_progression.pdf``: bar chart of the four totals
  against the published band.
- ``generated/motional_values.tex``: every quoted number in the paper's E38
  per-mode-progression and E39 subsections, as ``\\newcommand`` macros.
"""

from __future__ import annotations

import math

import common  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import run_motional_al_ion as motional  # noqa: E402  (benchmarks/, real WP30-33 code)


def _fmt_sci(x: float, sig: int = 4) -> str:
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10.0**exponent)
    return f"{mantissa:.{sig - 1}f}\\times10^{{{exponent}}}"


def _fmt_1e19(x: float, sig: int = 4) -> str:
    """Format a fractional shift in units of 1e-19 (Marshall's own convention,
    e.g. ``-114.6``), matching the way this project's own docs quote these
    numbers."""
    return f"{x / 1.0e-19:.{sig}f}"


def _fmt_compact_1e19(nominal: float, unc: float, sig: int = 1) -> str:
    """Format ``nominal +/- unc`` (both fractional shifts) as a single compact
    ``-115.1(2.7)`` macro in units of 1e-19, Marshall et al.'s own convention
    for this quantity (their Table I: ``-114.6(3.8)``)."""
    n19 = nominal / 1.0e-19
    u19 = unc / 1.0e-19
    return f"{n19:.{sig}f}({u19:.{sig}f})"


def _sigma_from_published(total: float, unc: float, published) -> float:
    """Combined-band sigma distance, the same statistic
    ``notebooks/13_trapped_ion_quantum_motion.ipynb``'s own progression table
    uses: ``abs(predicted - published) / sqrt(sigma_published^2 + sigma_predicted^2)``."""
    sigma_published = (published.hi - published.lo) / 2.0
    return abs(total - published.nominal) / math.sqrt(sigma_published**2 + unc**2)


def main() -> None:
    common.reset_tex_macro_file("motional_values.tex")

    al_case = motional.run_motional_al_ion_arithmetic_reproduction_case()
    participation_case = motional.run_motional_al_ion_participation_variant_case()
    radial_case = motional.run_motional_al_ion_radial_reconstructed_case()
    enhanced_case = motional.run_motional_al_ion_intrinsic_micromotion_enhanced_case()
    brewer_check = motional.run_wp33_brewer_consistency_check()
    exact_case = motional.run_motional_al_ion_exact_intrinsic_micromotion_enhanced_case()
    diagnostic_case = motional.run_motional_al_ion_coupled_floquet_case()
    constrained_case = motional.run_motional_al_ion_constrained_floquet_case()
    constrained_brewer_check = motional.run_wp35_constrained_brewer_consistency_check()

    for case, label in (
        (al_case, "WP30"),
        (participation_case, "WP31"),
        (radial_case, "WP32"),
        (enhanced_case, "WP33"),
        (exact_case, "WP34"),
        (diagnostic_case, "WP35 diagnostic"),
        (constrained_case, "WP35 constrained"),
    ):
        assert case.case_class == "arithmetic_reproduction", (
            f"unexpected case_class for {label} -- the classification-labeling prose this "
            "script's macros feed assumes every WP30-35 case is an arithmetic reproduction"
        )

    published = motional.loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT

    progression = [
        (
            "single mass",
            al_case.predicted_shift_nominal,
            al_case.predicted_uncertainty_fractional,
        ),
        (
            "participation\n(mass-ratio form)",
            participation_case.predicted_total_nominal,
            participation_case.predicted_total_uncertainty_fractional,
        ),
        (
            "participation\n(spectrum-reconstructed)",
            radial_case.predicted_total_nominal,
            radial_case.predicted_total_uncertainty_fractional,
        ),
        (
            "participation\n$\\times$ leading-order\nenhancement",
            enhanced_case.predicted_total_nominal,
            enhanced_case.predicted_total_uncertainty_fractional,
        ),
        (
            "participation\n$\\times$ exact Floquet\nenhancement",
            exact_case.predicted_total_nominal,
            exact_case.predicted_total_uncertainty_combined_fractional,
        ),
        (
            "coupled\nconstrained fit",
            constrained_case.predicted_total_nominal,
            constrained_case.predicted_total_uncertainty_combined_fractional,
        ),
    ]

    # --- Figure 8: the six-variant progression against the published band. ---
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x_pos = np.arange(len(progression))
    totals_1e19 = np.array([p[1] for p in progression]) / 1.0e-19
    uncs_1e19 = np.array([p[2] for p in progression]) / 1.0e-19
    labels = [p[0] for p in progression]

    ax.axhspan(
        published.lo / 1.0e-19,
        published.hi / 1.0e-19,
        color=common.COLOR_REFERENCE,
        alpha=0.18,
        lw=0,
        label="Marshall et al. published band",
        zorder=1,
    )
    ax.axhline(published.nominal / 1.0e-19, color=common.COLOR_REFERENCE, lw=0.9, ls="--", zorder=2)
    ax.bar(
        x_pos,
        totals_1e19,
        yerr=uncs_1e19,
        color=common.COLOR_ENGINE,
        width=0.55,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=7.0)
    ax.set_ylabel(r"$(P-1)_\mathrm{motional}$ ($\times10^{-19}$)")
    ax.set_title("Al$^+$ secular-motion total: the six-variant progression")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(common.FIGURES_DIR / "fig8_motional_progression.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Generated \input macros (never hand-typed in main.tex). ------------
    common.write_tex_macro(
        "AlIonPublishedNominal", _fmt_1e19(published.nominal), "motional_values.tex"
    )
    common.write_tex_macro(
        "AlIonPublishedUnc",
        _fmt_1e19((published.hi - published.lo) / 2.0),
        "motional_values.tex",
    )

    sigma_wp30 = _sigma_from_published(
        al_case.predicted_shift_nominal, al_case.predicted_uncertainty_fractional, published
    )
    common.write_tex_macro(
        "AlIonWpThirtyTotal", _fmt_1e19(al_case.predicted_shift_nominal), "motional_values.tex"
    )
    common.write_tex_macro(
        "AlIonWpThirtyUnc",
        _fmt_1e19(al_case.predicted_uncertainty_fractional),
        "motional_values.tex",
    )
    common.write_tex_macro("AlIonWpThirtySigma", f"{sigma_wp30:.2f}", "motional_values.tex")
    common.write_tex_macro("AlIonWpThirtyVerdict", al_case.kpi_verdict, "motional_values.tex")
    common.write_tex_macro(
        "AlIonWpThirtyCompact",
        _fmt_compact_1e19(
            al_case.predicted_shift_nominal, al_case.predicted_uncertainty_fractional, sig=2
        ),
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonPublishedCompact",
        _fmt_compact_1e19(published.nominal, (published.hi - published.lo) / 2.0, sig=1),
        "motional_values.tex",
    )

    sigma_wp31 = _sigma_from_published(
        participation_case.predicted_total_nominal,
        participation_case.predicted_total_uncertainty_fractional,
        published,
    )
    common.write_tex_macro(
        "AlIonWpThirtyOneTotal",
        _fmt_1e19(participation_case.predicted_total_nominal),
        "motional_values.tex",
    )
    common.write_tex_macro("AlIonWpThirtyOneSigma", f"{sigma_wp31:.2f}", "motional_values.tex")
    common.write_tex_macro(
        "AlIonWpThirtyOneVerdict", participation_case.total_kpi_verdict, "motional_values.tex"
    )

    sigma_wp32 = _sigma_from_published(
        radial_case.predicted_total_nominal,
        radial_case.predicted_total_uncertainty_fractional,
        published,
    )
    common.write_tex_macro(
        "AlIonWpThirtyTwoTotal",
        _fmt_1e19(radial_case.predicted_total_nominal),
        "motional_values.tex",
    )
    common.write_tex_macro("AlIonWpThirtyTwoSigma", f"{sigma_wp32:.2f}", "motional_values.tex")
    common.write_tex_macro(
        "AlIonWpThirtyTwoVerdict", radial_case.total_kpi_verdict, "motional_values.tex"
    )

    sigma_wp33 = _sigma_from_published(
        enhanced_case.predicted_total_nominal,
        enhanced_case.predicted_total_uncertainty_fractional,
        published,
    )
    common.write_tex_macro(
        "AlIonWpThirtyThreeTotal",
        _fmt_sci(enhanced_case.predicted_total_nominal),
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonWpThirtyThreeUnc",
        _fmt_sci(enhanced_case.predicted_total_uncertainty_fractional),
        "motional_values.tex",
    )
    common.write_tex_macro("AlIonWpThirtyThreeSigma", f"{sigma_wp33:.2f}", "motional_values.tex")
    common.write_tex_macro(
        "AlIonWpThirtyThreeVerdict", enhanced_case.total_kpi_verdict, "motional_values.tex"
    )

    # Per-mode radial ratios (predicted/published), WP32 (participation only)
    # and WP33 (participation x enhancement).
    radial_by_name = {m.name: m for m in radial_case.per_mode if not m.is_axial}
    enhanced_by_name = {m.name: m for m in enhanced_case.per_mode if not m.is_axial}
    for name, macro_suffix in (
        ("x_com", "Xcom"),
        ("x_str", "Xstr"),
        ("y_com", "Ycom"),
        ("y_str", "Ystr"),
    ):
        common.write_tex_macro(
            f"AlIonParticipationOnly{macro_suffix}",
            f"{radial_by_name[name].ratio_predicted_over_published:.2f}",
            "motional_values.tex",
        )
        common.write_tex_macro(
            f"AlIonFinal{macro_suffix}",
            f"{enhanced_by_name[name].ratio_predicted_over_published:.2f}",
            "motional_values.tex",
        )

    # Mathieu parameters and enhancement factors (Marshall trap).
    cm = enhanced_case.clock_mathieu
    common.write_tex_macro("AlIonMathieuQ", f"{cm['mathieu_q']:.4f}", "motional_values.tex")
    common.write_tex_macro("AlIonMathieuAx", f"{cm['mathieu_a_x']:.5f}", "motional_values.tex")
    common.write_tex_macro("AlIonMathieuAy", f"{cm['mathieu_a_y']:.5f}", "motional_values.tex")
    common.write_tex_macro(
        "AlIonEnhancementX", f"{enhanced_case.enhancement_x:.3f}", "motional_values.tex"
    )
    common.write_tex_macro(
        "AlIonEnhancementY", f"{enhanced_case.enhancement_y:.3f}", "motional_values.tex"
    )
    common.write_tex_macro(
        "AlIonPartnerDevX",
        f"{enhanced_case.partner_x_relative_deviation * 100.0:.2f}",
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonPartnerDevY",
        f"{enhanced_case.partner_y_relative_deviation * 100.0:.2f}",
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonRfDriveMhz",
        f"{motional.loaders.MARSHALL_AL_ION_RF_DRIVE_FREQUENCY_HZ / 1.0e6:.2f}",
        "motional_values.tex",
    )

    # Brewer et al. independent-trap consistency check.
    common.write_tex_macro(
        "AlIonBrewerRfDriveMhz",
        f"{motional.loaders.BREWER_AL_ION_RF_DRIVE_FREQUENCY_HZ / 1.0e6:.2f}",
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonBrewerPartnerDevX",
        f"{brewer_check.partner_x_relative_deviation * 100.0:.2f}",
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonBrewerPartnerDevY",
        f"{brewer_check.partner_y_relative_deviation * 100.0:.2f}",
        "motional_values.tex",
    )
    brewer_ratios = (
        brewer_check.per_mode_ratio_x_com,
        brewer_check.per_mode_ratio_x_str,
        brewer_check.per_mode_ratio_y_com,
        brewer_check.per_mode_ratio_y_str,
    )
    common.write_tex_macro("AlIonBrewerRatioLo", f"{min(brewer_ratios):.2f}", "motional_values.tex")
    common.write_tex_macro("AlIonBrewerRatioHi", f"{max(brewer_ratios):.2f}", "motional_values.tex")

    # --- WP34: numerically exact Floquet enhancement (Marshall trap). --------
    sigma_wp34 = _sigma_from_published(
        exact_case.predicted_total_nominal,
        exact_case.predicted_total_uncertainty_combined_fractional,
        published,
    )
    common.write_tex_macro(
        "AlIonWpThirtyFourTotal",
        _fmt_sci(exact_case.predicted_total_nominal),
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonWpThirtyFourUnc",
        _fmt_sci(exact_case.predicted_total_uncertainty_combined_fractional),
        "motional_values.tex",
    )
    common.write_tex_macro("AlIonWpThirtyFourSigma", f"{sigma_wp34:.2f}", "motional_values.tex")
    common.write_tex_macro(
        "AlIonWpThirtyFourVerdict", exact_case.total_kpi_verdict, "motional_values.tex"
    )
    common.write_tex_macro(
        "AlIonWpThirtyFourEnhancementX", f"{exact_case.enhancement_x:.3f}", "motional_values.tex"
    )
    common.write_tex_macro(
        "AlIonWpThirtyFourEnhancementY", f"{exact_case.enhancement_y:.3f}", "motional_values.tex"
    )
    enhancement_x_rel_pct = (
        (exact_case.enhancement_x - exact_case.enhancement_x_leading_order)
        / exact_case.enhancement_x_leading_order
        * 100.0
    )
    enhancement_y_rel_pct = (
        (exact_case.enhancement_y - exact_case.enhancement_y_leading_order)
        / exact_case.enhancement_y_leading_order
        * 100.0
    )
    common.write_tex_macro(
        "AlIonWpThirtyFourEnhancementXRelPct", f"{enhancement_x_rel_pct:.2f}", "motional_values.tex"
    )
    common.write_tex_macro(
        "AlIonWpThirtyFourEnhancementYRelPct", f"{enhancement_y_rel_pct:.2f}", "motional_values.tex"
    )
    common.write_tex_macro(
        "AlIonWpThirtyFourPartnerDevX",
        f"{exact_case.partner_x_relative_deviation * 100.0:.2f}",
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonWpThirtyFourPartnerDevY",
        f"{exact_case.partner_y_relative_deviation * 100.0:.2f}",
        "motional_values.tex",
    )
    exact_by_name = {m.name: m for m in exact_case.per_mode if not m.is_axial}
    common.write_tex_macro(
        "AlIonWpThirtyFourFinalXcom",
        f"{exact_by_name['x_com'].ratio_predicted_over_published:.2f}",
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonWpThirtyFourFinalXstr",
        f"{exact_by_name['x_str'].ratio_predicted_over_published:.2f}",
        "motional_values.tex",
    )

    # --- WP35: the coupled two-ion Floquet solve. -----------------------------
    # Diagnostic per-axis variant (kept as a labeled diagnostic, G17 gate):
    # its own total is the model-structure component's other endpoint, and
    # its own limit checks (c -> 0 against WP34, RF -> 0 against WP32) tie
    # the coupled solve to its predecessors.
    sigma_wp35_diagnostic = _sigma_from_published(
        diagnostic_case.predicted_total_nominal,
        diagnostic_case.predicted_total_uncertainty_combined_fractional,
        published,
    )
    common.write_tex_macro(
        "AlIonWpThirtyFiveDiagnosticTotal",
        _fmt_sci(diagnostic_case.predicted_total_nominal),
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonWpThirtyFiveDiagnosticSigma", f"{sigma_wp35_diagnostic:.2f}", "motional_values.tex"
    )
    common.write_tex_macro(
        "AlIonWpThirtyFiveDiagnosticVerdict",
        diagnostic_case.total_kpi_verdict,
        "motional_values.tex",
    )

    # The constrained fit: this WP's own headline result (G17 gate fix loop).
    fit = constrained_case.constrained_fit
    common.write_tex_macro("AlIonWpThirtyFiveQ", f"{fit['mathieu_q']:.6f}", "motional_values.tex")
    common.write_tex_macro("AlIonWpThirtyFiveAlpha", f"{fit['alpha']:.6f}", "motional_values.tex")
    common.write_tex_macro(
        "AlIonWpThirtyFiveAx", f"{fit['mathieu_a_x']:.5f}", "motional_values.tex"
    )
    common.write_tex_macro(
        "AlIonWpThirtyFiveAy", f"{fit['mathieu_a_y']:.5f}", "motional_values.tex"
    )

    _measured_hz = {
        "x_com": motional.loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[2][1] * 1.0e6,
        "x_str": motional.loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[3][1] * 1.0e6,
        "y_com": motional.loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[4][1] * 1.0e6,
        "y_str": motional.loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[5][1] * 1.0e6,
    }
    for mode_key, residual_hz, macro_suffix in (
        ("x_com", fit["residual_x_com_hz"], "Xcom"),
        ("x_str", fit["residual_x_str_hz"], "Xstr"),
        ("y_com", fit["residual_y_com_hz"], "Ycom"),
        ("y_str", fit["residual_y_str_hz"], "Ystr"),
    ):
        common.write_tex_macro(
            f"AlIonWpThirtyFiveResid{macro_suffix}Hz", f"{residual_hz:.1f}", "motional_values.tex"
        )
        common.write_tex_macro(
            f"AlIonWpThirtyFiveResid{macro_suffix}Pct",
            f"{residual_hz / _measured_hz[mode_key] * 100.0:.3f}",
            "motional_values.tex",
        )

    sigma_wp35 = _sigma_from_published(
        constrained_case.predicted_total_nominal,
        constrained_case.predicted_total_uncertainty_combined_fractional,
        published,
    )
    common.write_tex_macro(
        "AlIonWpThirtyFiveTotal",
        _fmt_sci(constrained_case.predicted_total_nominal),
        "motional_values.tex",
    )
    common.write_tex_macro("AlIonWpThirtyFiveSigma", f"{sigma_wp35:.2f}", "motional_values.tex")
    common.write_tex_macro(
        "AlIonWpThirtyFiveVerdict", constrained_case.total_kpi_verdict, "motional_values.tex"
    )

    # The three-component uncertainty budget (G17 item 3).
    common.write_tex_macro(
        "AlIonWpThirtyFiveUncThermometry",
        _fmt_sci(constrained_case.predicted_total_uncertainty_nbar_fractional),
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonWpThirtyFiveUncRounding",
        _fmt_sci(constrained_case.predicted_total_uncertainty_rounding_fractional),
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonWpThirtyFiveUncModelStructure",
        _fmt_sci(constrained_case.predicted_total_uncertainty_model_structure_fractional),
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonWpThirtyFiveUncCombined",
        _fmt_sci(constrained_case.predicted_total_uncertainty_combined_fractional),
        "motional_values.tex",
    )

    # The over-determination partner check (G17 item 2: both traps).
    common.write_tex_macro(
        "AlIonWpThirtyFivePartnerDevX",
        f"{constrained_case.partner_x_relative_deviation * 100.0:.2f}",
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonWpThirtyFivePartnerDevY",
        f"{constrained_case.partner_y_relative_deviation * 100.0:.2f}",
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonWpThirtyFiveBrewerPartnerDevX",
        f"{constrained_brewer_check.partner_x_relative_deviation * 100.0:.2f}",
        "motional_values.tex",
    )
    common.write_tex_macro(
        "AlIonWpThirtyFiveBrewerPartnerDevY",
        f"{constrained_brewer_check.partner_y_relative_deviation * 100.0:.2f}",
        "motional_values.tex",
    )

    # Radial per-mode ratios (constrained fit, Marshall trap).
    constrained_by_name = {m.name: m for m in constrained_case.per_mode if not m.is_axial}
    for name, macro_suffix in (
        ("x_com", "Xcom"),
        ("x_str", "Xstr"),
        ("y_com", "Ycom"),
        ("y_str", "Ystr"),
    ):
        common.write_tex_macro(
            f"AlIonWpThirtyFiveRatio{macro_suffix}",
            f"{constrained_by_name[name].ratio_predicted_over_published:.2f}",
            "motional_values.tex",
        )

    print(
        f"WP30 (single-mass): {al_case.predicted_shift_nominal / 1e-19:.2f}e-19, "
        f"sigma={sigma_wp30:.2f}, verdict={al_case.kpi_verdict}"
    )
    print(
        f"WP31 (participation, mass-ratio): "
        f"{participation_case.predicted_total_nominal / 1e-19:.2f}e-19, sigma={sigma_wp31:.2f}, "
        f"verdict={participation_case.total_kpi_verdict}"
    )
    print(
        f"WP32 (participation, reconstructed): "
        f"{radial_case.predicted_total_nominal / 1e-19:.2f}e-19, sigma={sigma_wp32:.2f}, "
        f"verdict={radial_case.total_kpi_verdict}"
    )
    print(
        f"WP33 (participation x enhancement): "
        f"{enhanced_case.predicted_total_nominal:.4e}, sigma={sigma_wp33:.2f}, "
        f"verdict={enhanced_case.total_kpi_verdict}"
    )
    print(f"Wrote {common.FIGURES_DIR / 'fig8_motional_progression.pdf'}")
    print(f"Wrote {common.GENERATED_DIR / 'motional_values.tex'}")


if __name__ == "__main__":
    main()
