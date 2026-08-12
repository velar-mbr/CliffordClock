#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figure 2 (NPL reproducibility case): predicted band vs. published band.

Calls the real ``benchmarks/run_benchmarks.run_npl_reproducibility_case``
(WP10), which runs the actual ``coupling.type: stark_dc`` pipeline three
times (at NPL's residual field's asymmetric-uncertainty low/nominal/high
bounds, Bowden et al., "Rydberg Electrometry for Optical Lattice Clocks",
arXiv:1706.01944 / PRA 96, 023419 (2017)) and compares the resulting band
against NPL's own published DC-Stark shift band -- **a reproducibility
check, not a blind prediction** (see the module docstring of
``benchmarks/run_benchmarks.py`` and ``benchmarks/MAPPING.md`` for the
binding classification-labeling rationale this figure and its caption must not
contradict).

Outputs
-------
- ``figures/fig2_npl_band.pdf``: predicted band vs. published band,
  drawn as asymmetric error bars around each nominal value.
- ``generated/npl_values.tex``: every quoted number in the paper's NPL
  subsection, as ``\\newcommand`` macros.
"""

from __future__ import annotations

import common  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import run_benchmarks  # noqa: E402  (benchmarks/run_benchmarks.py, real WP10 code)


def _fmt_sci(x: float, sig: int = 4) -> str:
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10.0**exponent)
    return f"{mantissa:.{sig - 1}f}\\times10^{{{exponent}}}"


def main() -> None:
    common.reset_tex_macro_file("npl_values.tex")

    case = run_benchmarks.run_npl_reproducibility_case()

    # --- Figure: two vertical bands (predicted, published), 1e-20 units. ---
    scale = 1.0e-20
    fig, ax = plt.subplots(figsize=(4.2, 4.0))

    labels = ["This engine\n(predicted)", "NPL 2017\n(published)"]
    nominals = [case.predicted_shift_nominal / scale, case.published_shift_nominal / scale]
    lo_err = [
        (case.predicted_shift_nominal - case.predicted_shift_lo) / scale,
        (case.published_shift_nominal - case.published_shift_lo) / scale,
    ]
    hi_err = [
        (case.predicted_shift_hi - case.predicted_shift_nominal) / scale,
        (case.published_shift_hi - case.published_shift_nominal) / scale,
    ]
    colors = [common.COLOR_ENGINE, common.COLOR_REFERENCE]
    x_pos = [0, 1]

    for xi, nom, lo, hi, color, _label in zip(
        x_pos, nominals, lo_err, hi_err, colors, labels, strict=True
    ):
        ax.errorbar(
            xi,
            nom,
            yerr=[[lo], [hi]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.5,
            capsize=6,
            markersize=7,
        )
    ax.axhspan(
        case.published_shift_lo / scale,
        case.published_shift_hi / scale,
        color=common.COLOR_REFERENCE,
        alpha=0.12,
        zorder=0,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(r"$\Delta\nu/\nu_0$ ($\times 10^{-20}$)")
    ax.set_title("Sr-87 DC-Stark reproducibility case\n(NPL residual field, Bowden et al. 2017)")
    ax.axhline(0.0, color=common.COLOR_NEUTRAL, lw=0.6, ls=":")
    fig.tight_layout()
    fig.savefig(common.FIGURES_DIR / "fig2_npl_band.pdf")
    plt.close(fig)

    # --- Generated \input macros. ---
    common.write_tex_macro("NplFieldLo", f"{case.field_lo_v_per_m:.3f}", "npl_values.tex")
    common.write_tex_macro("NplFieldNominal", f"{case.field_nominal_v_per_m:.2f}", "npl_values.tex")
    common.write_tex_macro("NplFieldHi", f"{case.field_hi_v_per_m:.3f}", "npl_values.tex")
    common.write_tex_macro("NplPredictedLo", _fmt_sci(case.predicted_shift_lo), "npl_values.tex")
    common.write_tex_macro(
        "NplPredictedNominal", _fmt_sci(case.predicted_shift_nominal), "npl_values.tex"
    )
    common.write_tex_macro("NplPredictedHi", _fmt_sci(case.predicted_shift_hi), "npl_values.tex")
    common.write_tex_macro("NplPublishedLo", _fmt_sci(case.published_shift_lo), "npl_values.tex")
    common.write_tex_macro(
        "NplPublishedNominal", _fmt_sci(case.published_shift_nominal), "npl_values.tex"
    )
    common.write_tex_macro("NplPublishedHi", _fmt_sci(case.published_shift_hi), "npl_values.tex")
    common.write_tex_macro("NplVerdict", case.kpi_verdict, "npl_values.tex")
    common.write_tex_macro(
        "NplBandsOverlap", "overlap" if case.bands_overlap else "do not overlap", "npl_values.tex"
    )
    common.write_tex_macro("NplCaseClass", case.case_class, "npl_values.tex")

    print(
        f"Figure 2: predicted band [{case.predicted_shift_lo:.3e}, "
        f"{case.predicted_shift_hi:.3e}], published band "
        f"[{case.published_shift_lo:.3e}, {case.published_shift_hi:.3e}], "
        f"verdict={case.kpi_verdict}, case_class={case.case_class}"
    )
    print(f"Wrote {common.FIGURES_DIR / 'fig2_npl_band.pdf'}")
    print(f"Wrote {common.GENERATED_DIR / 'npl_values.tex'}")


if __name__ == "__main__":
    main()
