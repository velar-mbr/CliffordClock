#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generated \\newcommand macros for the E37 multi-surface thermal-environment
validation subsections (PTB position-resolved scan; Bothwell thesis
six-surface exchange-factor build).

Calls the real ``cliffordclock.integrator.omega`` E37 functions
(``bbr_environment_pivot_perturbation``/``bbr_pivot_perturbation``) at the
exact published inputs ``notebooks/12_thermal_environment.ipynb`` sections 6
and 7 already exercise, so this script never re-derives a number the
notebook has not already computed the same way -- it only re-runs the same
engine calls at build time so ``main.tex`` can ``\\input`` the result instead
of transcribing it.

**Two published external comparisons this script does NOT reproduce from the
engine (both from Nosske et al., arXiv:2507.14030): their own closed-form
model prediction and their own measured differential shift.** These are
literal citations, kept as named Python constants below (mirroring how
``fig2_npl_band.py``/``fig7_bothwell_sitemap.py`` carry NPL's and Bothwell's
own published bands as constants, never as free-floating numbers typed
directly into ``main.tex``), so the paper's prose macro for "the published
model" or "the published measurement" traces to one place in this repository.

Outputs
-------
- ``generated/thermal_environment_values.tex``: every quoted number in the
  paper's E37 validation-anchor prose, as ``\\newcommand`` macros.
"""

from __future__ import annotations

import math

import common  # noqa: E402
import numpy as np  # noqa: E402

from cliffordclock.ensemble.species import get_species  # noqa: E402
from cliffordclock.integrator.omega import (  # noqa: E402
    RadiationSurface,
    bbr_environment_pivot_perturbation,
    bbr_environment_pivot_uncertainty,
    bbr_pivot_perturbation,
)

# --- Section 3/7 anchor: Nosske et al., arXiv:2507.14030, published shield -----
# geometry, verbatim from the paper (p. 4, 6-8), the same numbers
# notebooks/12_thermal_environment.ipynb section 7 uses.
_R1_MM = 0.484
_R2_MM = 0.484
_SHIELD_LENGTH_MM = 20.0
_SHIELD_EPS_IN = 0.926
_T_SHIELD_K = -50.1 + 273.15
_T_OUT_K = 21.0 + 273.15

#: Nosske et al.'s own published position-scan comparison numbers (their
#: model prediction and their own measurement), quoted directly from the
#: paper (p. 8): both are literal external citations, not engine output.
NOSSKE_MODEL_NOMINAL_FRACTIONAL = -3.32e-15
NOSSKE_MODEL_UNC_FRACTIONAL = 0.07e-15
NOSSKE_MEASURED_NOMINAL_FRACTIONAL = -3.33e-15
NOSSKE_MEASURED_UNC_FRACTIONAL = 0.03e-15


def _nosske_omega_over_4pi(z_mm: float, r1_mm: float = _R1_MM, r2_mm: float = _R2_MM) -> float:
    """Nosske et al. Eq. 2: fractional solid angle under which the atoms see
    the shield's two end apertures, as a function of axial position ``z_mm``
    relative to the shield's own center. Verbatim from the paper; the same
    closed form ``notebooks/12_thermal_environment.ipynb`` section 7 runs.
    """
    term1 = 0.5 * (1.0 - math.sin(math.atan((z_mm + _SHIELD_LENGTH_MM / 2.0) / r1_mm)))
    term2 = 0.5 * (1.0 + math.sin(math.atan((z_mm - _SHIELD_LENGTH_MM / 2.0) / r2_mm)))
    return term1 + term2


# --- Section 6 anchor: Bothwell PhD thesis, Table 2.7 "Vacuum chamber -------
# generalized exchange factors" (p. 40), transcribed at the table's own
# published 3-sig-fig precision; the oven nozzle's 575 C operating
# temperature (Fig. 2.3 caption, p. 17) is excluded below since it sits far
# outside the engine's validated 50-350 K BBR fit window.
BOTHWELL_EXCHANGE_FACTORS_RAW: dict[str, float] = {
    "heated_sapphire_window": 8.81e-4,
    "viewport_2_75in_extended_flange": 1.47e-2,
    "viewport_2_75in_direct_flange": 1.18e-1,
    "viewport_6in": 5.39e-1,
    "glass_cell": 1.08e-3,
    "metal_chamber_tubing_slower": 3.26e-1,
    "oven_nozzle": 2.15e-5,
}
_T_SETPOINT_K = 22.0 + 273.15  # thesis Section 2.3.1, p. 20: every setpoint is 22 C
_T_OVEN_NOZZLE_K = 575.0 + 273.15  # thesis Fig. 2.3 caption, p. 17
#: Thesis Table 2.1, p. 22: the net correction from ray-traced-model
#: temperature to the directly measured (in-vacuum TFPRT probe) atom
#: temperature, and its own stated combined uncertainty.
_THESIS_ATOM_TEMPERATURE_CORRECTION_K = 0.0191


def _fmt_sci(x: float, sig: int = 4) -> str:
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10.0**exponent)
    return f"{mantissa:.{sig - 1}f}\\times10^{{{exponent}}}"


def _fmt_compact(nominal: float, unc: float, sig: int = 3) -> str:
    """Format ``nominal +/- unc`` as a single compact parenthetical-uncertainty
    macro (e.g. ``-3.32(7)\\times10^{-15}``), the convention the cited source
    paper itself uses for this quantity."""
    exponent = int(np.floor(np.log10(abs(nominal))))
    mantissa = nominal / (10.0**exponent)
    unc_scaled = unc / (10.0**exponent)
    unc_last_digit = round(unc_scaled * 10.0 ** (sig - 1))
    return f"{mantissa:.{sig - 1}f}({unc_last_digit}){{\\times}}10^{{{exponent}}}"


def main() -> None:
    common.reset_tex_macro_file("thermal_environment_values.tex")
    sr87 = get_species("Sr87")

    # --- PTB position-resolved scan (section 7 of notebook 12). -------------
    w_center = _nosske_omega_over_4pi(0.0)
    shift_center = bbr_environment_pivot_perturbation(
        (
            RadiationSurface(name="aperture", weight=w_center, temperature_k=_T_OUT_K),
            RadiationSurface(
                name="shield",
                weight=1.0 - w_center,
                temperature_k=_T_SHIELD_K,
                emissivity=_SHIELD_EPS_IN,
            ),
        ),
        sr87,
    )
    # Far outside the shield, Omega(z)/4pi -> 1 exactly (Eq. 2's z -> +/-infinity
    # limit): the aperture's effective weight saturates at 1 and the
    # environment reduces to the pure T_out single-temperature shift, the
    # same single-surface reduction the section-3 formula check pins.
    shift_outside = bbr_pivot_perturbation(_T_OUT_K, sr87)
    engine_differential = shift_outside - shift_center

    diff_vs_model = abs(engine_differential - NOSSKE_MODEL_NOMINAL_FRACTIONAL)
    diff_vs_measured = abs(engine_differential - NOSSKE_MEASURED_NOMINAL_FRACTIONAL)

    common.write_tex_macro(
        "PtbPositionEngineDifferential",
        _fmt_sci(engine_differential),
        "thermal_environment_values.tex",
    )
    common.write_tex_macro(
        "PtbPositionModelCompact",
        _fmt_compact(NOSSKE_MODEL_NOMINAL_FRACTIONAL, NOSSKE_MODEL_UNC_FRACTIONAL),
        "thermal_environment_values.tex",
    )
    common.write_tex_macro(
        "PtbPositionMeasuredCompact",
        _fmt_compact(NOSSKE_MEASURED_NOMINAL_FRACTIONAL, NOSSKE_MEASURED_UNC_FRACTIONAL),
        "thermal_environment_values.tex",
    )
    common.write_tex_macro(
        "PtbPositionDiffModel", _fmt_sci(diff_vs_model), "thermal_environment_values.tex"
    )
    common.write_tex_macro(
        "PtbPositionDiffMeasured", _fmt_sci(diff_vs_measured), "thermal_environment_values.tex"
    )

    # --- Bothwell thesis six-surface exchange-factor build (section 6). -----
    raw_sum_seven = math.fsum(BOTHWELL_EXCHANGE_FACTORS_RAW.values())
    table_controlled = {
        name: value
        for name, value in BOTHWELL_EXCHANGE_FACTORS_RAW.items()
        if name != "oven_nozzle"
    }
    raw_sum_six = math.fsum(table_controlled.values())
    norm_six = {name: value / raw_sum_six for name, value in table_controlled.items()}
    sensor_unc_k = {name: (0.013 if name == "viewport_6in" else 0.050) for name in table_controlled}
    thesis_surfaces = tuple(
        RadiationSurface(
            name=name,
            weight=norm_six[name],
            temperature_k=_T_SETPOINT_K,
            temperature_uncertainty_k=sensor_unc_k[name],
        )
        for name in table_controlled
    )
    six_surface_shift = bbr_environment_pivot_perturbation(thesis_surfaces, sr87)
    six_surface_sigma_indep, _ = bbr_environment_pivot_uncertainty(
        thesis_surfaces, sr87, correlated=False
    )

    thesis_corrected_t_k = _T_SETPOINT_K + _THESIS_ATOM_TEMPERATURE_CORRECTION_K
    thesis_bookkeeping_shift = bbr_pivot_perturbation(thesis_corrected_t_k, sr87)
    bothwell_agreement = abs(six_surface_shift - thesis_bookkeeping_shift)

    # The excluded nozzle's own physically citable omission bound: its
    # static T^4 term alone, at its real 848.15 K operating temperature
    # (the dynamic terms are NOT extrapolated that far past the engine's
    # validated 50-350 K window; doing so is exactly the invalid move E32's
    # own validity check exists to block).
    nozzle_weight = BOTHWELL_EXCHANGE_FACTORS_RAW["oven_nozzle"] / raw_sum_seven
    coeffs = sr87.resolve_bbr_coefficients()
    nozzle_t_ratio = _T_OVEN_NOZZLE_K / 300.0
    nozzle_static_bound = (
        coeffs.nu_stat_300k_hz * nozzle_weight * nozzle_t_ratio**4 / sr87.clock_frequency_hz
    )

    common.write_tex_macro(
        "BothwellExchangeEngineShift", _fmt_sci(six_surface_shift), "thermal_environment_values.tex"
    )
    common.write_tex_macro(
        "BothwellExchangeThesisShift",
        _fmt_sci(thesis_bookkeeping_shift),
        "thermal_environment_values.tex",
    )
    common.write_tex_macro(
        "BothwellExchangeAgreement", _fmt_sci(bothwell_agreement), "thermal_environment_values.tex"
    )
    common.write_tex_macro(
        "BothwellExchangeUncertainty",
        _fmt_sci(six_surface_sigma_indep),
        "thermal_environment_values.tex",
    )
    common.write_tex_macro(
        "BothwellNozzleStaticBound",
        _fmt_sci(nozzle_static_bound),
        "thermal_environment_values.tex",
    )

    print(
        f"PTB position scan: engine {engine_differential:.4e}, published model "
        f"{NOSSKE_MODEL_NOMINAL_FRACTIONAL:.3e}, published measurement "
        f"{NOSSKE_MEASURED_NOMINAL_FRACTIONAL:.3e}, |diff| vs model {diff_vs_model:.2e}, "
        f"|diff| vs measurement {diff_vs_measured:.2e}"
    )
    print(
        f"Bothwell six-surface build: engine {six_surface_shift:.6e}, thesis bookkeeping "
        f"{thesis_bookkeeping_shift:.6e}, agreement {bothwell_agreement:.3e}, nozzle static "
        f"bound {nozzle_static_bound:+.3e}"
    )
    print(f"Wrote {common.GENERATED_DIR / 'thermal_environment_values.tex'}")


if __name__ == "__main__":
    main()
