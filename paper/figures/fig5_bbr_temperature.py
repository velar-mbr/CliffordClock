#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figure 5 (BBR pivot vs. temperature): engine curve, published-dataset overlay, residual.

Plots the real engine's Sr-87 blackbody-radiation pivot
``(P-1)_BBR(T)`` (:func:`cliffordclock.integrator.omega.bbr_pivot_perturbation`,
CONVENTIONS.md E32) across its full 50-350~K validity window, with a
coefficient-uncertainty band from :func:`bbr_pivot_uncertainty`, overlaid
with the Lisdat, Dörscher, Nosske, Sterr open dataset's own tabulated
total BBR shift (``benchmarks/data/lisdat2021_dataset/BBR_shift.dat``,
PTB Open Access Repository, DOI 10.7795/720.20210928, CC BY-ND 4.0),
converted to a fractional shift via the *engine's own* registry clock
frequency (``Species.clock_frequency_hz``), never a hand-typed
``nu_0``. JILA's published operating point (Aeppli et al.
arXiv:2403.10664, T = 293.282 K) is marked directly from the real
``benchmarks/run_bbr_jila_arithmetic_reproduction.py`` case (WP20), the
same module :mod:`bbr_jila_values` already calls.

**Binding labeling note (do not weaken):** the registry's Sr-87 dynamic
BBR coefficients are the PTB-2025 rescaling of *this dataset's own* fit
shape onto the more precise Aeppli et al. 2024 anchor
(Lisdat et al., PRR 3, L042036 (2021); Aeppli et al., PRL 133, 023401
(2024)). This means the
engine curve and the dataset points are expected to agree in *shape* by
construction, and the residual between them is an expected,
non-vanishing offset from the Lisdat-2021-to-Aeppli-2024 anchor
revision, not a bug and not something this script tunes away. The
residual panel plots exactly this comparison; whatever structure it
shows is reported as computed, not adjusted to look like agreement.

Residual decomposition at 300 K (independent-review reconciliation,
2026-08-11): the −2.670 mHz total-shift offset splits into −2.543 mHz
from the dynamic-term anchor revision (Aeppli −153.06 mHz vs. this
dataset's −150.51 mHz) plus −0.127 mHz from an internal rounding quirk
in the published ``BBR_shift.dat`` itself (its tabulated total column
minus its dynamic column differs from the Middelmann static value,
which both Lisdat's paper and this registry use, by that amount at
300 K). Neither piece is a CliffordClock defect; both are properties
of the published inputs, stated here so nobody re-derives this.

Outputs
-------
- ``figures/fig5_bbr_temperature.pdf``: two-panel figure (engine curve +
  dataset overlay + JILA point; residual panel below).
- ``generated/bbr_temperature_values.tex``: every quoted number in the
  paper's discussion of this figure, as ``\\newcommand`` macros.
"""

from __future__ import annotations

import common  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import run_bbr_jila_arithmetic_reproduction as jila_bbr  # noqa: E402  (benchmarks/, real WP20 code)

from cliffordclock.ensemble.species import get_species  # noqa: E402
from cliffordclock.integrator.omega import (  # noqa: E402
    bbr_pivot_perturbation,
    bbr_pivot_uncertainty,
)

_DATASET_PATH = common.REPO_ROOT / "benchmarks" / "data" / "lisdat2021_dataset" / "BBR_shift.dat"
_T_MIN_K = 50.0
_T_MAX_K = 350.0
_T_GRID_POINTS = 301


def _fmt_sci(x: float, sig: int = 4) -> str:
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10.0**exponent)
    return f"{mantissa:.{sig - 1}f}\\times10^{{{exponent}}}"


def _load_dataset_bbr_shift() -> np.ndarray:
    """Load ``BBR_shift.dat`` verbatim (5 leading columns; the 6th, residuals, is ragged)."""
    return np.genfromtxt(
        _DATASET_PATH,
        skip_header=2,
        delimiter="\t",
        usecols=(0, 1, 2, 3, 4),
        dtype=float,
    )


def main() -> None:
    common.reset_tex_macro_file("bbr_temperature_values.tex")

    species = get_species("Sr87")
    nu_0 = species.clock_frequency_hz  # engine registry value, never hand-typed.

    # --- Engine curve across the full validity window. ----------------------
    t_grid = np.linspace(_T_MIN_K, _T_MAX_K, _T_GRID_POINTS)
    engine_vals = np.array([bbr_pivot_perturbation(float(t), species) for t in t_grid])
    engine_sigma = np.array([bbr_pivot_uncertainty(float(t), species)[0] for t in t_grid])

    # --- Published-dataset overlay, restricted to the engine's window. ------
    dataset = _load_dataset_bbr_shift()
    t_dataset = dataset[:, 0]
    total_shift_hz = dataset[:, 1]
    unc_stat_1e19 = dataset[:, 3]
    unc_dyn_1e19 = dataset[:, 4]
    window = (t_dataset >= _T_MIN_K) & (t_dataset <= _T_MAX_K)
    t_dataset_w = t_dataset[window]
    dataset_frac_w = total_shift_hz[window] / nu_0
    dataset_unc_1e19_w = np.hypot(unc_stat_1e19[window], unc_dyn_1e19[window])

    # --- JILA's published operating point (real WP20 case, not hand-typed). -
    jila_case = jila_bbr.run_jila_bbr_arithmetic_reproduction_case()

    # --- Residual: engine minus dataset, at the dataset's own T points. -----
    engine_at_dataset_t = np.array([bbr_pivot_perturbation(float(t), species) for t in t_dataset_w])
    residual_frac = engine_at_dataset_t - dataset_frac_w
    residual_1e19 = residual_frac / 1.0e-19

    idx_300 = int(np.argmin(np.abs(t_dataset_w - 300.0)))
    offset_300k_1e19 = float(residual_1e19[idx_300])
    max_abs_residual_1e19 = float(np.max(np.abs(residual_1e19)))
    max_abs_residual_t = float(t_dataset_w[int(np.argmax(np.abs(residual_1e19)))])

    # --- Figure: engine curve + dataset overlay + JILA point, with residual panel. ---
    fig, (ax_main, ax_res) = plt.subplots(
        2,
        1,
        figsize=(5.4, 3.7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1.0], "hspace": 0.08},
    )

    ax_main.plot(
        t_grid,
        engine_vals,
        "-",
        color=common.COLOR_ENGINE,
        lw=1.6,
        label=r"engine $(P-1)_{\rm BBR}(T)$, Sr-87",
        zorder=3,
    )
    ax_main.fill_between(
        t_grid,
        engine_vals - engine_sigma,
        engine_vals + engine_sigma,
        color=common.COLOR_ENGINE,
        alpha=0.22,
        lw=0,
        label="coefficient-uncertainty band",
        zorder=2,
    )
    ax_main.plot(
        t_dataset_w,
        dataset_frac_w,
        "o",
        color=common.COLOR_REFERENCE,
        ms=3.5,
        label="Lisdat et al.\\ 2021 dataset (total shift)",
        zorder=4,
    )
    ax_main.plot(
        [jila_case.temperature_k],
        [jila_case.published_shift_nominal],
        "*",
        color="black",
        ms=13,
        label="Aeppli et al. operating point (published)",
        zorder=5,
    )
    ax_main.set_ylabel(r"$(P-1)_{\rm BBR}$ (fractional)")
    ax_main.set_title("Sr-87 blackbody-radiation pivot vs.\\ temperature")
    ax_main.legend(fontsize=7, loc="lower left")
    ax_main.grid(True, alpha=0.25)

    ax_res.axhline(0.0, color=common.COLOR_NEUTRAL, lw=0.7, ls=":")
    ax_res.errorbar(
        t_dataset_w,
        residual_1e19,
        yerr=dataset_unc_1e19_w,
        fmt="o",
        color=common.COLOR_ENGINE,
        ecolor=common.COLOR_NEUTRAL,
        ms=3.5,
        elinewidth=1.0,
        capsize=2,
        label="engine $-$ dataset (error bar: dataset's own stated uncertainty)",
    )
    ax_res.set_xlabel("Temperature (K)")
    ax_res.set_ylabel(r"residual ($\times10^{-19}$)")
    ax_res.legend(fontsize=6.5, loc="lower left")
    ax_res.grid(True, alpha=0.25)

    fig.savefig(common.FIGURES_DIR / "fig5_bbr_temperature.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Generated \\input macros. --------------------------------------------
    common.write_tex_macro("BbrTempWindowLo", f"{_T_MIN_K:.0f}", "bbr_temperature_values.tex")
    common.write_tex_macro("BbrTempWindowHi", f"{_T_MAX_K:.0f}", "bbr_temperature_values.tex")
    common.write_tex_macro(
        "BbrOffsetAtThreeHundredK", f"{offset_300k_1e19:.1f}", "bbr_temperature_values.tex"
    )
    common.write_tex_macro(
        "BbrMaxAbsResidual", f"{max_abs_residual_1e19:.1f}", "bbr_temperature_values.tex"
    )
    common.write_tex_macro(
        "BbrMaxAbsResidualT", f"{max_abs_residual_t:.0f}", "bbr_temperature_values.tex"
    )

    print(
        f"Figure 5: offset at 300 K = {offset_300k_1e19:.2f}e-19 fractional "
        f"({offset_300k_1e19 * 1.0e-19 * nu_0 * 1.0e3:.3f} mHz absolute), "
        f"max |residual| over {_T_MIN_K:.0f}-{_T_MAX_K:.0f} K = {max_abs_residual_1e19:.2f}e-19 "
        f"at T={max_abs_residual_t:.0f} K"
    )
    print(f"Wrote {common.FIGURES_DIR / 'fig5_bbr_temperature.pdf'}")
    print(f"Wrote {common.GENERATED_DIR / 'bbr_temperature_values.tex'}")


if __name__ == "__main__":
    main()
