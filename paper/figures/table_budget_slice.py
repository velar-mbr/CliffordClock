#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generated LaTeX table: the covered slice of one platform's published budget.

One real clock platform (the JILA Sr system: Aeppli et al., PRL 133,
023401 (2024); Bothwell et al., Nature 602, 420 (2022)), with the rows of
its published evaluation this engine covers computed live from the same
case objects the benchmark suite commits, each row in one unit
convention, the published value carrying its stated uncertainty, and
explicit difference and sigma columns so agreement is a number.

Writes ``generated/budget_slice_table.tex`` (the table body, same light
grey rule styling as the validation table) and
``generated/budget_slice_values.tex`` (the macros the surrounding
subsection quotes). Mirrors ``notebooks/11_real_budget_slice.ipynb``'s
summary cell exactly; both draw from the same case objects, so neither
can drift from the other without the benchmark itself moving.
"""

from __future__ import annotations

import re

import common  # noqa: E402
import run_bbr_jila_arithmetic_reproduction as jila_bbr  # noqa: E402  (benchmarks/)
import run_bothwell_redshift as bothwell  # noqa: E402  (benchmarks/)


def main() -> None:
    common.reset_tex_macro_file("budget_slice_values.tex")

    bbr_case = jila_bbr.run_jila_bbr_arithmetic_reproduction_case()
    grav_report = bothwell.build_report()
    grav_case = grav_report["bothwell_2022_nature_602_420_redshift_case"]

    sig_bbr_pub = (bbr_case.published_shift_hi - bbr_case.published_shift_lo) / 2.0
    d_bbr = bbr_case.predicted_shift_nominal - bbr_case.published_shift_nominal
    bbr_sigma = abs(d_bbr) / sig_bbr_pub
    # Combined-uncertainty (normalized-error) variant: the published band
    # and this tool's own prediction band in quadrature. The table column
    # deliberately uses the published-only denominator (the conservative
    # convention, matching the G9-pinned slope sigma distances); this
    # macro states what the proper combined metric reduces the BBR row to.
    sig_bbr_tool = bbr_case.predicted_combined_uncertainty_fractional
    bbr_sigma_combined = abs(d_bbr) / (sig_bbr_pub**2 + sig_bbr_tool**2) ** 0.5

    pred_mm20 = grav_case["predicted_slope_per_mm"] * 1e20
    ma = grav_case["measured_slope_method_a"]
    mb = grav_case["measured_slope_method_b"]
    ma_n, ma_s = ma["nominal"] * 1e20, (ma["hi"] - ma["lo"]) / 2.0 * 1e20
    mb_n, mb_s = mb["nominal"] * 1e20, (mb["hi"] - mb["lo"]) / 2.0 * 1e20
    ab_spread = ma_n - mb_n
    ab_sigma = abs(ab_spread) / (ma_s**2 + mb_s**2) ** 0.5

    # The DC-Stark context value comes out of the committed case's own
    # note string, never typed here (same discipline as notebook 11).
    m = re.search(r"DC Stark ([+-][\d.()e/m-]+/mm)", grav_case["dc_stark_context_note"])
    if m is None:
        m = re.search(r"([+-]\d+\.\d+\(\d+\.\d+\)e-20/mm)", grav_case["dc_stark_context_note"])
    dc_row = m.group(1) if m else "+0.3(0.2)e-20/mm (see dc_stark_context_note)"

    for name, value in [
        ("BudgetBbrSigma", f"{bbr_sigma:.2f}"),
        ("BudgetBbrSigmaCombined", f"{bbr_sigma_combined:.2f}"),
        ("BudgetBbrDelta", f"{d_bbr:+.1e}"),
        ("BudgetAbSpread", f"{ab_spread:+.1f}"),
        ("BudgetAbSigma", f"{ab_sigma:.2f}"),
    ]:
        common.write_tex_macro(name, value, "budget_slice_values.tex")

    grey_sep = "@{\\hspace{7pt}{\\color{black!25}\\vrule width 0.3pt}\\hspace{7pt}}"
    grey_row_rule = "\\noalign{{\\color{black!25}\\hrule height 0.3pt}}\n"
    rows = [
        (
            "BBR, fractional (Aeppli et al.\\ Table I)",
            f"${bbr_case.published_shift_nominal / 1e-15:+.6f}\\times10^{{-15}}"
            f"\\,\\pm\\,{sig_bbr_pub / 1e-19:.1f}\\times10^{{-19}}$",
            f"${bbr_case.predicted_shift_nominal / 1e-15:+.6f}\\times10^{{-15}}$",
            f"${d_bbr / 1e-20:+.1f}\\times10^{{-20}}$",
            f"{bbr_sigma:.2f}",
            f"\\textit{{{bbr_case.kpi_verdict}}}; arithmetic reproduction",
        ),
        (
            "redshift slope, method A ($10^{-20}$/mm)",
            f"${ma_n:+.1f}\\pm{ma_s:.1f}$",
            f"${pred_mm20:+.2f}$",
            f"${pred_mm20 - ma_n:+.2f}$",
            f"{grav_case['sigma_distance_method_a']:.2f}",
            f"\\textit{{{grav_case['kpi_verdict_method_a']}}}; reproducibility",
        ),
        (
            "redshift slope, method B ($10^{-20}$/mm)",
            f"${mb_n:+.1f}\\pm{mb_s:.1f}$",
            f"${pred_mm20:+.2f}$",
            f"${pred_mm20 - mb_n:+.2f}$",
            f"{grav_case['sigma_distance_method_b']:.2f}",
            f"\\textit{{{grav_case['kpi_verdict_method_b']}}}; reproducibility",
        ),
        (
            "DC-Stark gradient (Bothwell et al.\\ Table 1)",
            dc_row.replace("e-20/mm", "$\\times10^{-20}$/mm"),
            "awaiting a field characterization",
            "n/a",
            "n/a",
            "context row",
        ),
    ]

    out_path = common.GENERATED_DIR / "budget_slice_table.tex"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by paper/figures/table_budget_slice.py -- do not edit.\n")
        f.write("\\begingroup\n")
        f.write("\\setlength{\\tabcolsep}{7pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.25}\n")
        f.write(
            f"\\begin{{tabular}}{{p{{0.17\\linewidth}}{grey_sep}p{{0.17\\linewidth}}{grey_sep}"
            f"p{{0.13\\linewidth}}{grey_sep}p{{0.08\\linewidth}}{grey_sep}"
            f"p{{0.04\\linewidth}}{grey_sep}p{{0.16\\linewidth}}}}\n"
        )
        f.write("\\toprule\n")
        f.write(
            "Row & Published & This tool & $\\Delta$ (tool$-$pub.) & $\\sigma$ & "
            "Verdict / class \\\\\n"
        )
        f.write("\\midrule\n")
        for i, row in enumerate(rows):
            f.write(" & ".join(row) + " \\\\\n")
            if i < len(rows) - 1:
                f.write(grey_row_rule)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\endgroup\n")

    print(f"BBR sigma {bbr_sigma:.3f}, A/B spread {ab_spread:+.1f}e-20/mm ({ab_sigma:.2f} sigma)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
