#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generated LaTeX validation table: case | reference | computed | agreement.

Runs the *real* pipeline for every row (V1-V4 closed-form self-consistency
cases, KA1-KA4 literature known-answer cases, the E24 rotor/scalar
cross-check, the NPL and Bothwell reproducibility cases, and the Roos
cross-vintage quadrupole-slope case) and writes
``generated/validation_table.tex`` -- no number in this table is
hand-copied from ``docs/validation.md``/``benchmarks/RESULTS.md``; it is
recomputed here from the same configs those documents describe (V1/V2/V3
reuse ``tests/test_e2e.py``'s exact Case A/B/C parameters; KA1-KA4 reuse
``tests/test_known_answers.py``'s exact configs; the NPL case calls
``benchmarks/run_benchmarks.run_npl_reproducibility_case`` directly).

If a future code change moves any of these numbers, this script's output
(and hence the paper's table) moves with it automatically -- the paper
can never silently drift from what the test suite verifies.
"""

from __future__ import annotations

import math

import common  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np
import run_benchmarks  # noqa: E402  (benchmarks/run_benchmarks.py)

import reference_impl  # noqa: E402  (tests/reference_impl.py, independent NumPy reference)
from cliffordclock.constants import (  # noqa: E402
    BOLTZMANN_K,
    ELECTRON_MASS,
    PLANCK_H,
    SPEED_OF_LIGHT,
)
from cliffordclock.ensemble.species import get_species  # noqa: E402
from cliffordclock.pipeline import PipelineConfig, run_pipeline_full  # noqa: E402

_M_E_C2 = ELECTRON_MASS * SPEED_OF_LIGHT**2


def _sci_bare(x: float, sig: int = 3) -> str:
    """Scientific notation without surrounding ``$`` (for embedding inside a larger math span)."""
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10.0**exponent)
    return f"{mantissa:.{sig - 1}f}\\times10^{{{exponent}}}"


def _sci(x: float, sig: int = 3) -> str:
    return f"${_sci_bare(x, sig)}$"


class Row:
    def __init__(
        self, case: str, reference: str, computed: str, agreement: str, note: str = ""
    ) -> None:
        self.case = case
        self.reference = reference
        self.computed = computed
        self.agreement = agreement
        self.note = note


def _row_v1() -> Row:
    """V1: uniform field -- rotor-rate shift beyond the scalar baseline < 1e-19."""
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1000.0]}}},
            "coupling": {"mu": [0.0, 0.0, 1.0e-25]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 30, "seed": 0},
            "integration": {"dtau": 0.5, "steps": 200},
        }
    )
    result = run_pipeline_full(config)
    beyond_baseline = float(
        jnp.max(jnp.abs(result.ensemble_result.phase_rotor - result.ensemble_result.phase))
    )
    return Row(
        "V1: uniform field, rotor vs.\\ scalar",
        "closed form (CONVENTIONS.md \\S9): $0$",
        _sci(beyond_baseline),
        "PASS" if beyond_baseline < 1e-19 else "FAIL",
    )


def _row_v2() -> Row:
    """V2: constant gradient, single static lattice node, closed form."""
    r0 = (0.01, -0.02, 0.03)
    grad = [[1.0e3, 2.0e3, 0.0], [0.0, -1.0e3, 5.0e2], [3.0e2, 0.0, -2.0e3]]
    mu = (1.0e-25, 2.0e-25, -3.0e-25)
    config = PipelineConfig.from_dict(
        {
            "species": "Yb171",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": list(r0)},
            "field": {
                "synthetic": {
                    "kind": "constant_gradient",
                    "params": {"e0": [0.0, 0.0, 0.0], "grad": grad},
                }
            },
            "coupling": {"mu": list(mu)},
            "ensemble": {
                "regime": "lattice",
                "temperature_uK": 1.0,
                "motional_n": [0, 0, 0],
                "n_quad": 1,
            },
            "integration": {"dtau": 1.0, "steps": 1000},
        }
    )
    result = run_pipeline_full(config)
    grad_arr = np.asarray(grad, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    r0_arr = np.asarray(r0, dtype=np.float64)
    expected = float(np.dot(r0_arr @ grad_arr, mu_arr)) / _M_E_C2
    rel_err = abs(result.report.mean_fractional_shift - expected) / abs(expected)
    verdict = " (PASS, rtol $10^{-12}$)" if rel_err < 1e-12 else " (FAIL)"
    return Row(
        "V2: constant gradient, static atom",
        _sci(expected),
        _sci(result.report.mean_fractional_shift),
        f"rel.\\ err.\\ {_sci(rel_err)}" + verdict,
    )


def _row_v3_e24() -> tuple[Row, Row]:
    """V3/E24: quadrupole field, M=100 classical ensemble; scalar path and rotor path."""
    k = 2.0e6
    mu = (1.0e-24, -5.0e-25, 3.0e-25)
    dtau = 0.5
    steps = 60
    m = 100
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [3.0e5, 3.0e5, 4.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "quadrupole", "params": {"k": k}}},
            "coupling": {"mu": list(mu)},
            "ensemble": {"regime": "classical", "temperature_uK": 2.0, "size": m, "seed": 7},
            "integration": {"dtau": dtau, "steps": steps},
        }
    )
    result = run_pipeline_full(config)
    trajectories = np.asarray(result.trajectories)
    mu_arr = np.asarray(mu, dtype=np.float64)
    t_tilde = steps * dtau

    reference_phases = np.array(
        [
            reference_impl.accumulate_phase_midpoint(trajectories[i], dtau, k, mu_arr)
            for i in range(m)
        ]
    )
    reference_mean_shift = reference_impl.mean_fractional_shift(reference_phases, t_tilde)
    scalar_rel_err = abs(result.report.mean_fractional_shift - reference_mean_shift) / abs(
        reference_mean_shift
    )
    row_scalar = Row(
        "V3: quadrupole field, $M{=}100$ (scalar)",
        "independent NumPy reference",
        _sci(result.report.mean_fractional_shift),
        f"rel.\\ err.\\ {_sci(scalar_rel_err)}"
        + (" (PASS, rtol $10^{-10}$)" if scalar_rel_err < 1e-10 else " (FAIL)"),
    )

    per_atom_bounds = np.array(
        [
            reference_impl.e24_second_order_bound_phase(trajectories[i], dtau, k, mu_arr)
            for i in range(m)
        ]
    )
    physical_bound_shift = float(np.mean(per_atom_bounds)) / t_tilde
    tolerance = 10.0 * physical_bound_shift + 1e-9 * abs(reference_mean_shift)
    rotor_mean_shift = float(jnp.mean(result.ensemble_result.phase_rotor)) / t_tilde
    rotor_diff = abs(rotor_mean_shift - reference_mean_shift)
    row_e24 = Row(
        "E24: rotor/scalar cross-check",
        "independent NumPy reference",
        _sci(rotor_mean_shift),
        "PASS" if rotor_diff < tolerance else "FAIL",
        note="first-order equality required; $O(\\omega_\\mathrm{boost}^2)$ divergence permitted",
    )
    return row_scalar, row_e24


def _row_v4() -> Row:
    """V4: harmonic trap, linear-gradient field -- order-2 accuracy (fig. 3)."""
    macro_path = common.GENERATED_DIR / "step_size_values.tex"
    if not macro_path.exists():
        raise RuntimeError(
            "step_size_values.tex missing -- run fig3_step_size_accuracy.py before "
            "table_validation.py (see paper/figures/make_figures.py ordering)"
        )
    macros = {}
    for line in macro_path.read_text(encoding="utf-8").splitlines():
        # \newcommand{\Name}{value}
        name = line.split("{")[1].rstrip("}").lstrip("\\")
        value = line.split("{", 2)[2].rsplit("}", 1)[0]
        macros[name] = value
    return Row(
        "V4: harmonic trap, linear gradient",
        "closed form (SHM + 2nd-order Doppler)",
        f"order {macros['StepSizeOrder']} (design: 2)",
        f"rel.\\ err.\\ {_sci(float(macros['StepSizeDefaultError']))} at "
        f"$N_\\mathrm{{res}}{{=}}{macros['StepSizeNRes']}$",
    )


def _row_ka1() -> Row:
    return _ka_uniform_field_row("Sr87", "Middelmann et al.\\ 2012")


def _row_ka2() -> Row:
    return _ka_uniform_field_row("Yb171", "Sherman et al.\\ 2012")


def _ka_uniform_field_row(species_name: str, cite_short: str) -> Row:
    field_v_per_m = 100.0
    config = PipelineConfig.from_dict(
        {
            "species": species_name,
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
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
            "integration": {"time_s": 1.0},
        }
    )
    result = run_pipeline_full(config)
    species = get_species(species_name)
    expected = (
        -(species.delta_alpha_dc_si / 2.0)
        * field_v_per_m**2
        / (PLANCK_H * species.clock_frequency_hz)
    )
    rel_err = (
        0.0
        if expected == result.report.mean_fractional_shift
        else abs(result.report.mean_fractional_shift - expected) / abs(expected)
    )
    label = "KA1" if species_name == "Sr87" else "KA2"
    return Row(
        f"{label}: {species_name} uniform-field DC Stark",
        f"textbook formula, {cite_short}: {_sci(expected)}",
        _sci(result.report.mean_fractional_shift),
        "PASS (exact to fp64)" if rel_err < 1e-10 else f"rel.\\ err.\\ {_sci(rel_err)}",
    )


def _row_ka3() -> Row:
    species_name = "Yb171"
    trap_omega = (2.0e5, 2.0e5, 2.0e5)
    e0 = (30.0, -20.0, 10.0)
    grad = (
        (5.0e2, -2.0e2, 0.0),
        (1.0e2, 3.0e2, -1.0e2),
        (0.0, 2.0e2, -6.0e2),
    )
    n_quad = 6
    config = PipelineConfig.from_dict(
        {
            "species": species_name,
            "trap": {"omega_xyz": list(trap_omega)},
            "field": {
                "synthetic": {
                    "kind": "constant_gradient",
                    "params": {"e0": list(e0), "grad": [list(row) for row in grad]},
                }
            },
            "coupling": {"type": "stark_dc"},
            "ensemble": {
                "regime": "lattice",
                "temperature_uK": 1.0,
                "motional_n": [0, 0, 0],
                "n_quad": n_quad,
            },
            "integration": {"time_s": 1.0},
        }
    )
    result = run_pipeline_full(config)
    species = get_species(species_name)
    k_s = species.resolve_stark_coefficient_hz_per_v2_m2()
    expected_mean, expected_var = reference_impl.stark_shift_mean_and_variance(
        np.asarray(e0, dtype=np.float64),
        np.asarray(grad, dtype=np.float64),
        np.asarray(trap_omega, dtype=np.float64),
        species.mass_kg,
        k_s,
        species.clock_frequency_hz,
    )
    shift = np.asarray(result.ensemble_result.fractional_shift, dtype=np.float64)
    weights = np.asarray(result.weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    pipeline_mean = float(np.sum(weights * shift))
    pipeline_var = float(np.sum(weights * (shift - pipeline_mean) ** 2))
    mean_rel_err = abs(pipeline_mean - expected_mean) / abs(expected_mean)
    var_rel_err = abs(pipeline_var - expected_var) / abs(expected_var)
    return Row(
        "KA3: Yb171 gradient shift + spread",
        f"Gaussian-moment reference: mean {_sci(expected_mean)}, "
        f"$\\sqrt{{\\mathrm{{Var}}}}$ {_sci(math.sqrt(expected_var))}",
        f"mean {_sci(pipeline_mean)}, $\\sqrt{{\\mathrm{{Var}}}}$ {_sci(math.sqrt(pipeline_var))}",
        f"rel.\\ err.\\ mean {_sci(mean_rel_err)}, var {_sci(var_rel_err)} (rtol $10^{{-8}}$)",
    )


def _row_ka4() -> Row:
    species_name = "Sr87"
    temperature_uk = 5.0
    ensemble_size = 5000
    seed = 42
    species = get_species(species_name)
    temperature_k = temperature_uk * 1.0e-6
    expected = -3.0 * BOLTZMANN_K * temperature_k / (2.0 * species.mass_kg * SPEED_OF_LIGHT**2)

    config = PipelineConfig.from_dict(
        {
            "species": species_name,
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 0.0]}}},
            "coupling": {"type": "stark_dc"},
            "ensemble": {
                "regime": "classical",
                "temperature_uK": temperature_uk,
                "size": ensemble_size,
                "seed": seed,
            },
            "integration": {"mode": "secular", "time_s": 1.0},
        }
    )
    result = run_pipeline_full(config)
    measured = result.report.mean_fractional_shift
    sem = result.report.shift_std_error
    n_sigma = abs(measured - expected) / sem
    pass_note = " (PASS, bound $5\\sigma$)" if n_sigma < 5.0 else " (FAIL)"
    verdict = f"{n_sigma:.2f}$\\sigma$" + pass_note
    return Row(
        "KA4: Sr87 second-order Doppler",
        f"equipartition $-3k_BT/2mc^2$: {_sci(expected)}",
        f"{_sci(measured)} $\\pm$ {_sci(sem)} (SEM)",
        verdict,
    )


def _row_npl() -> Row:
    case = run_benchmarks.run_npl_reproducibility_case()
    published = f"[{_sci_bare(case.published_shift_lo)}, {_sci_bare(case.published_shift_hi)}]"
    predicted = f"[{_sci_bare(case.predicted_shift_lo)}, {_sci_bare(case.predicted_shift_hi)}]"
    return Row(
        "NPL: Sr87 reproducibility case",
        f"Bowden et al.\\ 2017: ${published}$",
        f"${predicted}$",
        f"bands {'overlap' if case.bands_overlap else 'disjoint'} "
        f"(\\textit{{{case.kpi_verdict}}}; {case.case_class}, not blind prediction)",
    )


def _row_bothwell() -> Row:
    """Bothwell 2022 mm-scale redshift case (the second reproducibility case;
    real benchmarks/run_bothwell_redshift.py case, G9 sign-off B4 label)."""
    import run_bothwell_redshift as bothwell  # noqa: PLC0415 (benchmarks/, heavy import)

    case = bothwell.run_bothwell_redshift_case()
    a = case.measured_slope_method_a
    b = case.measured_slope_method_b
    return Row(
        "Bothwell: Sr87 mm-scale redshift",
        f"Bothwell et al.\\ 2022: ${_sci_bare(a['nominal'], 2)}$, "
        f"${_sci_bare(b['nominal'], 3)}$ per mm",
        f"${_sci_bare(case.predicted_slope_per_mm, 5)}$ per mm",
        f"{case.sigma_distance_method_a:.2f}$\\sigma$ / "
        f"{case.sigma_distance_method_b:.2f}$\\sigma$ "
        f"(\\textit{{{case.kpi_verdict_method_a}}}; {case.case_class}, not blind prediction)",
    )


def _row_roos() -> Row:
    """Roos 2006 two-ion quadrupole slope, cross-vintage headline variant
    (real benchmarks/run_roos_quadrupole_slope.py case, G8 sign-off B4 label;
    NOT MET is the expected verdict: it recovers the literature's own Theta
    theory-vs-measurement tension, and the case never joins the paper's
    reproducibility/blind-prediction headline counts)."""
    import run_roos_quadrupole_slope as roos  # noqa: PLC0415 (benchmarks/, heavy import)

    case = roos.run_roos_quadrupole_slope_case()
    cv = case.cross_vintage
    return Row(
        "Roos: Ca$^+$ two-ion quadrupole slope",
        f"Roos et al.\\ 2006: {cv.measured_slope_hz_mm2_per_v:.3f} Hz\\,mm$^2$/V",
        f"{cv.predicted_slope_hz_mm2_per_v:.4f} Hz\\,mm$^2$/V (Itano-2006 theory $\\Theta$)",
        f"{cv.residual_fractional * 100.0:+.2f}\\% (\\textit{{{cv.kpi_verdict}}}, expected: "
        "known $\\Theta$ tension; cross-vintage comparison)",
    )


def main() -> None:
    rows: list[Row] = []
    rows.append(_row_v1())
    rows.append(_row_v2())
    row_v3, row_e24 = _row_v3_e24()
    rows.append(row_v3)
    rows.append(_row_v4())
    rows.append(row_e24)
    rows.append(_row_ka1())
    rows.append(_row_ka2())
    rows.append(_row_ka3())
    rows.append(_row_ka4())
    rows.append(_row_npl())
    rows.append(_row_bothwell())
    rows.append(_row_roos())

    out_path = common.GENERATED_DIR / "validation_table.tex"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by paper/figures/table_validation.py -- do not edit.\n")
        # Readability (owner review, 2026-08-12): wider inter-column
        # spacing, a little vertical breathing room per row, and light
        # grey rules between columns and rows. colortbl clashes with
        # revtex4-2's tabular internals ("Extra \\or"), so the grey
        # rules use package-free primitives: a colored \\vrule in the
        # !{...} column separator and a colored \\hrule in \\noalign
        # between rows. Structural top/header/bottom rules stay black
        # booktabs rules.
        grey_sep = "@{\\hspace{8pt}{\\color{black!25}\\vrule width 0.3pt}\\hspace{8pt}}"
        grey_row_rule = "\\noalign{{\\color{black!25}\\hrule height 0.3pt}}\n"
        f.write("\\begingroup\n")
        f.write("\\setlength{\\tabcolsep}{9pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.25}\n")
        f.write(
            f"\\begin{{tabular}}{{p{{0.19\\linewidth}}{grey_sep}p{{0.26\\linewidth}}{grey_sep}"
            f"p{{0.22\\linewidth}}{grey_sep}p{{0.21\\linewidth}}}}\n"
        )
        f.write("\\toprule\n")
        f.write("Case & Reference & Computed & Agreement \\\\\n")
        f.write("\\midrule\n")
        for i, row in enumerate(rows):
            f.write(f"{row.case} & {row.reference} & {row.computed} & {row.agreement} \\\\\n")
            if i < len(rows) - 1:
                f.write(grey_row_rule)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\endgroup\n")

    for row in rows:
        print(f"{row.case}: computed={row.computed} agreement={row.agreement}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
