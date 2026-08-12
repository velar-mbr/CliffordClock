#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generated \\newcommand macros for the JILA BBR arithmetic-reproduction case.

Calls the real ``benchmarks/run_bbr_jila_arithmetic_reproduction.py``
(WP20 addendum), which runs the actual engine functions
(``cliffordclock.integrator.omega.bbr_pivot_perturbation``/
``bbr_pivot_uncertainty``) against the pinned Sr87 BBR registry
coefficients at JILA's own published operating temperature and compares
the result against JILA's own published BBR row (Aeppli et al.,
arXiv:2403.10664v2, PRL 133, 023401 (2024), Table I "BBR"). This is
recomputed from the engine here rather than only read back from the
committed ``benchmarks/results/wp20_bbr_arithmetic_reproduction.json``,
so the paper can never silently drift from what
``tests/test_bbr_pivot.py`` and the benchmark script itself verify -- the
same discipline every other ``paper/figures/*.py`` script follows.

**Binding classification label (G7 sign-off B5, ratified;
the project's theory sign-off record (G7)): "arithmetic reproduction of a
published standard-formula evaluation" -- explicitly weaker than this
paper's NPL reproducibility case (Sec. sec:npl): JILA's own BBR row is
itself computed from their own measured temperature and literature
coefficients, not an independently measured shift, and the registry's
dynamic-term coefficients are themselves anchored to Aeppli et al.'s own
measurement, so close agreement here is expected almost by construction.
This case is NOT counted toward this paper's validation headline (one
reproducibility case, zero blind-prediction cases, Sec. sec:validation);
main.tex's prose must say so explicitly wherever this case is discussed.**

Outputs
-------
- ``generated/bbr_jila_values.tex``: every quoted number in the paper's
  JILA BBR-row subsection, as ``\\newcommand`` macros.
"""

from __future__ import annotations

import common  # noqa: E402
import numpy as np
import run_bbr_jila_arithmetic_reproduction as jila_bbr  # noqa: E402  (benchmarks/, real WP20 code)


def _fmt_sci(x: float, sig: int = 4) -> str:
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10.0**exponent)
    return f"{mantissa:.{sig - 1}f}\\times10^{{{exponent}}}"


def main() -> None:
    common.reset_tex_macro_file("bbr_jila_values.tex")

    case = jila_bbr.run_jila_bbr_arithmetic_reproduction_case()
    assert case.case_class == "arithmetic_reproduction", (
        "unexpected case_class -- the classification-labeling machinery this script "
        "quotes assumes the WP20 arithmetic-reproduction class"
    )

    common.write_tex_macro("JilaBbrTemperature", f"{case.temperature_k:.3f}", "bbr_jila_values.tex")
    common.write_tex_macro(
        "JilaBbrTemperatureUnc",
        f"{case.temperature_uncertainty_k:.3f}",
        "bbr_jila_values.tex",
    )
    common.write_tex_macro(
        "JilaBbrPredictedNominal",
        _fmt_sci(case.predicted_shift_nominal, sig=6),
        "bbr_jila_values.tex",
    )
    common.write_tex_macro(
        "JilaBbrPredictedUnc",
        _fmt_sci(case.predicted_combined_uncertainty_fractional),
        "bbr_jila_values.tex",
    )
    common.write_tex_macro(
        "JilaBbrPredictedLo",
        _fmt_sci(case.predicted_combined_band_lo, sig=6),
        "bbr_jila_values.tex",
    )
    common.write_tex_macro(
        "JilaBbrPredictedHi",
        _fmt_sci(case.predicted_combined_band_hi, sig=6),
        "bbr_jila_values.tex",
    )
    common.write_tex_macro(
        "JilaBbrPublishedNominal",
        _fmt_sci(case.published_shift_nominal, sig=6),
        "bbr_jila_values.tex",
    )
    common.write_tex_macro(
        "JilaBbrPublishedLo", _fmt_sci(case.published_shift_lo, sig=6), "bbr_jila_values.tex"
    )
    common.write_tex_macro(
        "JilaBbrPublishedHi", _fmt_sci(case.published_shift_hi, sig=6), "bbr_jila_values.tex"
    )
    common.write_tex_macro(
        "JilaBbrResidual", _fmt_sci(case.residual_fractional), "bbr_jila_values.tex"
    )
    common.write_tex_macro(
        "JilaBbrBandsOverlap",
        "overlap" if case.bands_overlap else "do not overlap",
        "bbr_jila_values.tex",
    )
    common.write_tex_macro("JilaBbrVerdict", case.kpi_verdict, "bbr_jila_values.tex")
    common.write_tex_macro("JilaBbrCaseClass", "arithmetic reproduction", "bbr_jila_values.tex")

    print(
        f"JILA BBR arithmetic-reproduction case: predicted {case.predicted_shift_nominal:.6e} "
        f"+/- {case.predicted_combined_uncertainty_fractional:.3e}, published "
        f"{case.published_shift_nominal:.6e}, residual {case.residual_fractional:+.3e}, "
        f"verdict={case.kpi_verdict}, case_class={case.case_class}"
    )
    print(f"Wrote {common.GENERATED_DIR / 'bbr_jila_values.tex'}")


if __name__ == "__main__":
    main()
