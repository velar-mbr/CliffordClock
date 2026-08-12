#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figure 7 (Bothwell per-site frequency map): site map, WLS slope, published bands.

Re-runs the REAL ``ensemble.regime='lattice_extended'`` pipeline at the
exact configuration ``benchmarks/run_bothwell_redshift.py`` pins (its own
exported geometry/gravity constants, so the two cannot drift apart
silently; a cross-assertion below fails loudly if they ever do), and
plots the per-site fractional-frequency map (``PipelineResult.site_map``,
the Bothwell observable, CONVENTIONS.md section 15) together with the
pipeline's own weighted-least-squares slope fit and Bothwell et al.'s two
published corrected slope bands (Nature 602, 420 (2022)), mapped into the
engine's own physical-height sign convention so all three slopes are
directly comparable on one axis.

**Binding classification label (G9 sign-off B4, ratified; rides wherever
the case appears): "reproducibility", with the INVERTED-NPL caveat: the
g/c^2 arithmetic is textbook and the authors computed it themselves
trivially; what this case validates is the extended-sample MACHINERY
(per-site geometry, Gaussian-envelope weighting, map assembly) producing
the right measured-map slope end-to-end, with zero adjustable inputs. It
does not change the blind-prediction count.**

Every macro below is computed at run time: the g/c^2 magnitude is
recomputed from ``cliffordclock.constants`` (the G9 gate caught a
hand-transcribed ``1.0912e-16/m``, one digit off, so this script never
carries that value as a literal), the slope/sigma/verdict numbers come
from the real benchmark case object, and the site map comes from the real
pipeline run.

Outputs
-------
- ``figures/fig7_bothwell_sitemap.pdf``: two-panel figure (per-site map +
  fit + published bands; occupation envelope below).
- ``generated/bothwell_values.tex``: every quoted number in the paper's
  gravitational-redshift and Bothwell-case prose, as ``\\newcommand``
  macros.
"""

from __future__ import annotations

import math

import common  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import run_bothwell_redshift as bothwell  # noqa: E402  (benchmarks/, real WP22 code)

from cliffordclock.constants import SPEED_OF_LIGHT, STANDARD_GRAVITY  # noqa: E402
from cliffordclock.pipeline import (  # noqa: E402
    GRAVITY_EXTENT_WARN_M,
    PipelineConfig,
    run_pipeline_full,
)

#: WGS-84 mean Earth radius, metres (documented input to the ~76 m
#: uniform-g validity bound below, CONVENTIONS.md section 15 A3; the
#: bound itself is COMPUTED here, never transcribed).
_EARTH_RADIUS_M = 6.371e6

#: The 1e-19 fractional floor at which the uniform-g bound is evaluated
#: (CONVENTIONS.md section 15 A3: epsilon = (g/(c^2*R_E)) * Delta_h^2).
_UNIFORM_G_EPSILON = 1.0e-19

#: Bothwell's own analysis-window half width in envelope sigmas (their
#: stated "two +/-1.5-sigma regions"; the same 1.5 the benchmark module's
#: windowed cross-check fit and its ENVELOPE_SIGMA_M inference both pin).
_ANALYSIS_HALF_WIDTH_SIGMA = 1.5


def _fmt_sci(x: float, sig: int = 4) -> str:
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10.0**exponent)
    return f"{mantissa:.{sig - 1}f}\\times10^{{{exponent}}}"


def _site_map_pipeline_run() -> object:
    """Run the real lattice_extended pipeline at the benchmark module's own
    pinned configuration (same dict as
    ``run_bothwell_redshift.run_bothwell_redshift_case``, built from that
    module's exported constants) and return ``PipelineResult.site_map``.
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 0.0]}}},
            "coupling": {"type": "stark_dc"},
            "ensemble": {
                "regime": "lattice_extended",
                "temperature_uK": 1.0,
                "motional_n": [0, 0, 0],
                "n_quad": 1,
                "n_sites": bothwell.N_SITES,
                "site_spacing_m": bothwell.SITE_SPACING_M,
                "site_axis": [0.0, 0.0, 1.0],
                "site_envelope": "gaussian",
                "site_envelope_sigma_m": bothwell.ENVELOPE_SIGMA_M,
            },
            "integration": {"mode": "fast_path", "time_s": 1.0},
            "environment": {
                "gravity": {
                    "g_m_s2": bothwell.BOTHWELL_SURVEYED_G_M_S2,
                    "up_axis": [0.0, 0.0, 1.0],
                    "reference_height_m": 0.0,
                }
            },
        }
    )
    result = run_pipeline_full(config)
    site_map = result.site_map
    assert site_map is not None, "lattice_extended run unexpectedly produced no site_map"
    return site_map


def main() -> None:
    common.reset_tex_macro_file("bothwell_values.tex")

    # --- The real benchmark case (slopes, sigma distances, verdicts). -------
    case = bothwell.run_bothwell_redshift_case()
    assert case.case_class == "reproducibility", (
        "unexpected case_class; the classification-labeling prose this script's macros "
        "feed assumes the G9-ratified reproducibility class"
    )

    # --- The real pipeline's per-site map at the identical configuration. ---
    site_map = _site_map_pipeline_run()
    assert math.isclose(
        site_map.slope_per_m,
        case.predicted_slope_engine_convention_per_m,
        rel_tol=1e-12,
    ), (
        "this figure's pipeline re-run and the benchmark case disagree on the fitted "
        "slope; the two configurations have drifted apart, investigate before "
        "regenerating the paper"
    )

    offsets_mm = np.array([s.offset_m for s in site_map.sites]) * 1.0e3
    shifts_1e19 = np.array([s.mean_fractional_shift for s in site_map.sites]) / 1.0e-19
    weights = np.array([s.weight for s in site_map.sites])

    # Fit line and published bands, all in the engine's own physical-height
    # convention (a higher clock runs faster, positive slope): the published
    # per-mm slopes are negated back out of Bothwell's own coordinate
    # convention (the case module's documented coordinate-sign mapping).
    slope_per_mm_engine = site_map.slope_per_m / 1.0e3  # fractional per mm
    intercept = site_map.intercept
    x_line = np.linspace(offsets_mm.min(), offsets_mm.max(), 2)
    fit_1e19 = (intercept + slope_per_mm_engine * x_line) / 1.0e-19

    method_a = case.measured_slope_method_a
    method_b = case.measured_slope_method_b

    def _band_1e19(measured: dict) -> tuple[np.ndarray, np.ndarray]:
        lo_engine = -measured["hi"]  # sign map: engine convention
        hi_engine = -measured["lo"]
        y_lo = (intercept + lo_engine * x_line) / 1.0e-19
        y_hi = (intercept + hi_engine * x_line) / 1.0e-19
        return y_lo, y_hi

    band_a_lo, band_a_hi = _band_1e19(method_a)
    band_b_lo, band_b_hi = _band_1e19(method_b)

    fig, (ax_map, ax_env) = plt.subplots(
        2,
        1,
        figsize=(5.4, 3.9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.08},
    )

    ax_map.fill_between(
        x_line,
        band_a_lo,
        band_a_hi,
        color=common.COLOR_REFERENCE,
        alpha=0.18,
        lw=0,
        label="Bothwell method A corrected band",
        zorder=1,
    )
    ax_map.fill_between(
        x_line,
        band_b_lo,
        band_b_hi,
        color=common.COLOR_NEUTRAL,
        alpha=0.22,
        lw=0,
        label="Bothwell method B corrected band",
        zorder=1,
    )
    ax_map.plot(
        offsets_mm,
        shifts_1e19,
        ".",
        color=common.COLOR_ENGINE,
        ms=1.6,
        label=f"engine per-site map ({len(site_map.sites)} sites)",
        zorder=3,
        rasterized=True,
    )
    ax_map.plot(
        x_line,
        fit_1e19,
        "-",
        color="black",
        lw=1.1,
        label="weighted-least-squares fit",
        zorder=4,
    )
    ax_map.annotate(
        f"{case.sigma_distance_method_a:.2f}$\\sigma$ (A), "
        f"{case.sigma_distance_method_b:.2f}$\\sigma$ (B)",
        xy=(0.03, 0.86),
        xycoords="axes fraction",
        fontsize=8,
    )
    ax_map.set_ylabel(r"$\Delta\nu/\nu_0$ per site ($\times10^{-19}$)")
    ax_map.set_title("Bothwell 2022 mm-scale redshift: per-site map vs. published slopes")
    ax_map.legend(fontsize=6.5, loc="lower right")
    ax_map.grid(True, alpha=0.25)

    ax_env.plot(offsets_mm, weights, "-", color=common.COLOR_ENGINE, lw=1.0)
    half_window_mm = _ANALYSIS_HALF_WIDTH_SIGMA * bothwell.ENVELOPE_SIGMA_M * 1.0e3
    for x in (-half_window_mm, half_window_mm):
        ax_env.axvline(x, color=common.COLOR_NEUTRAL, lw=0.8, ls=":")
    ax_env.annotate(
        rf"$\pm{_ANALYSIS_HALF_WIDTH_SIGMA}\sigma$ analysis window",
        xy=(0.03, 0.72),
        xycoords="axes fraction",
        fontsize=7,
        color=common.COLOR_NEUTRAL,
    )
    ax_env.set_xlabel("site offset along the lattice axis (mm)")
    ax_env.set_ylabel("site weight")
    ax_env.grid(True, alpha=0.25)

    fig.savefig(common.FIGURES_DIR / "fig7_bothwell_sitemap.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Generated \input macros (never hand-typed in main.tex). ------------
    # E36 magnitude, recomputed from the pinned constants at build time (G9
    # sign-off A1's "computed never transcribed" discipline).
    g_over_c2_std = STANDARD_GRAVITY / SPEED_OF_LIGHT**2
    g_over_c2_local = bothwell.BOTHWELL_SURVEYED_G_M_S2 / SPEED_OF_LIGHT**2
    uniform_g_bound_m = math.sqrt(
        _UNIFORM_G_EPSILON * SPEED_OF_LIGHT**2 * _EARTH_RADIUS_M / STANDARD_GRAVITY
    )

    common.write_tex_macro(
        "GravPivotPerMetre", _fmt_sci(g_over_c2_std, sig=8), "bothwell_values.tex"
    )
    common.write_tex_macro(
        "GravStdSlopePerMm", _fmt_sci(g_over_c2_std * 1.0e-3, sig=5), "bothwell_values.tex"
    )
    common.write_tex_macro(
        "BothwellLocalSlopePerMm",
        _fmt_sci(g_over_c2_local * 1.0e-3, sig=5),
        "bothwell_values.tex",
    )
    common.write_tex_macro("GravExtentWarnM", f"{GRAVITY_EXTENT_WARN_M:.0f}", "bothwell_values.tex")
    common.write_tex_macro("GravUniformBoundM", f"{uniform_g_bound_m:.0f}", "bothwell_values.tex")
    common.write_tex_macro("BothwellNSites", f"{case.n_sites}", "bothwell_values.tex")
    common.write_tex_macro(
        "BothwellWindowSigma", f"{_ANALYSIS_HALF_WIDTH_SIGMA:g}", "bothwell_values.tex"
    )
    common.write_tex_macro(
        "BothwellSiteSpacingNm", f"{case.site_spacing_m * 1.0e9:.1f}", "bothwell_values.tex"
    )
    common.write_tex_macro(
        "BothwellEnvelopeSigmaUm", f"{case.envelope_sigma_m * 1.0e6:.1f}", "bothwell_values.tex"
    )
    common.write_tex_macro("BothwellSurveyedG", f"{case.g_m_s2:.3f}", "bothwell_values.tex")
    common.write_tex_macro(
        "BothwellPredictedSlopePerMm",
        _fmt_sci(case.predicted_slope_per_mm, sig=5),
        "bothwell_values.tex",
    )
    common.write_tex_macro(
        "BothwellMeasuredANominal", _fmt_sci(method_a["nominal"], sig=2), "bothwell_values.tex"
    )
    common.write_tex_macro(
        "BothwellMeasuredAUnc",
        _fmt_sci((method_a["hi"] - method_a["lo"]) / 2.0, sig=2),
        "bothwell_values.tex",
    )
    common.write_tex_macro(
        "BothwellMeasuredBNominal", _fmt_sci(method_b["nominal"], sig=3), "bothwell_values.tex"
    )
    common.write_tex_macro(
        "BothwellMeasuredBUnc",
        _fmt_sci((method_b["hi"] - method_b["lo"]) / 2.0, sig=2),
        "bothwell_values.tex",
    )
    common.write_tex_macro(
        "BothwellSigmaA", f"{case.sigma_distance_method_a:.2f}", "bothwell_values.tex"
    )
    common.write_tex_macro(
        "BothwellSigmaB", f"{case.sigma_distance_method_b:.2f}", "bothwell_values.tex"
    )
    common.write_tex_macro("BothwellVerdictA", case.kpi_verdict_method_a, "bothwell_values.tex")
    common.write_tex_macro("BothwellVerdictB", case.kpi_verdict_method_b, "bothwell_values.tex")

    print(
        f"Figure 7: {case.n_sites} sites, predicted slope {case.predicted_slope_per_mm:.4e}/mm, "
        f"sigma distances A={case.sigma_distance_method_a:.2f} "
        f"B={case.sigma_distance_method_b:.2f}, "
        f"verdicts A={case.kpi_verdict_method_a} B={case.kpi_verdict_method_b}"
    )
    print(f"g/c^2 (standard, computed) = {g_over_c2_std:.7e} /m")
    print(f"Wrote {common.FIGURES_DIR / 'fig7_bothwell_sitemap.pdf'}")
    print(f"Wrote {common.GENERATED_DIR / 'bothwell_values.tex'}")


if __name__ == "__main__":
    main()
