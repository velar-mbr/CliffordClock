#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generated \\newcommand macros for the Roos two-ion quadrupole-slope case.

Calls the real ``benchmarks/run_roos_quadrupole_slope.py`` case builder,
which drives the actual engine quadrupole functions
(``cliffordclock.integrator.omega.quadrupole_shift_joules``/
``quadrupole_mj_factor``) and the Ca+:D5/2 species-registry entry against
Roos et al.'s measured two-ion Fig. 4a slope (Nature 443, 316 (2006),
quant-ph/0701215v1). Both labeled variants are surfaced:

- **Cross-vintage comparison (headline, G8 sign-off B4):** Itano's
  independent theory Theta (Phys. Rev. A 73, 022510 (2006)) through the
  engine's own factor chain; ``NOT MET`` is the EXPECTED verdict (it
  recovers the literature's own Theta theory-vs-measurement tension).
- **Arithmetic reproduction (secondary, circular by construction):**
  Roos's own extracted Theta predicting their own fit's slope.

main.tex's prose must keep the two variants' labels distinct and must
not count either toward the paper's reproducibility/blind-prediction
headline (G8 sign-off B4; ``kpi_summary_impact`` in the case record).

The micromotion-boundary percentages below are pinned published values
(Dube et al., PRL 95, 033001 (2005), via docs/CONVENTIONS.md section 14's
primary-text extraction and docs/roadmap.md's owner-approved framing:
the measured m_J-dependent angle-scan budget in a real Sr+ trap is
roughly 95% micromotion-driven tensor Stark against roughly 5%
quadrupole), emitted as macros here so the paper's prose never carries
them as hand-typed literals.

Outputs
-------
- ``generated/roos_values.tex``: every quoted number in the paper's
  quadrupole-shift and Roos-case prose, as ``\\newcommand`` macros.
"""

from __future__ import annotations

import math

import common  # noqa: E402
import run_roos_quadrupole_slope as roos  # noqa: E402  (benchmarks/, real WP21 benchmark code)

#: Published micromotion-vs-quadrupole split of the m_J-dependent budget
#: in a real Sr+ trap (Dube et al., PRL 95, 033001 (2005), their own
#: model of their measured angle scan; transcribed in docs/CONVENTIONS.md
#: section 14 and docs/roadmap.md). Pinned here, with this citation, so
#: the paper quotes it through a macro.
_MICROMOTION_TENSOR_STARK_PCT = 95
_QUADRUPOLE_PCT = 5


def main() -> None:
    common.reset_tex_macro_file("roos_values.tex")

    case = roos.run_roos_quadrupole_slope_case()
    cv = case.cross_vintage
    ar = case.arithmetic_reproduction
    assert cv.case_class == "cross_vintage_comparison", (
        "unexpected headline case_class; the paper's labeling prose assumes the "
        "G8-ratified cross-vintage class"
    )
    assert ar.case_class == "arithmetic_reproduction", (
        "unexpected secondary case_class; the paper's labeling prose assumes the "
        "G8-ratified arithmetic-reproduction class"
    )
    assert math.isclose(case.structural_two_ion_enhancement_ratio, 24.0 / 5.0, rel_tol=1e-12), (
        "the engine-derived two-ion enhancement ratio no longer equals 24/5; "
        "investigate before regenerating the paper"
    )

    f = "roos_values.tex"

    def _compact(value: float, unc: float, decimals: int) -> str:
        """Standard parenthetical last-digit notation, e.g. 2.975(2)."""
        return f"{value:.{decimals}f}({round(unc * 10**decimals)})"

    measured_unc = (cv.measured_slope_hi - cv.measured_slope_lo) / 2.0
    common.write_tex_macro("RoosMeasuredSlope", f"{cv.measured_slope_hz_mm2_per_v:.3f}", f)
    common.write_tex_macro(
        "RoosMeasuredSlopeCompact",
        _compact(cv.measured_slope_hz_mm2_per_v, measured_unc, 3),
        f,
    )
    common.write_tex_macro("RoosThetaTheory", f"{cv.theta_au:.3f}", f)
    common.write_tex_macro("RoosThetaMeasured", f"{ar.theta_au:.2f}", f)
    common.write_tex_macro(
        "RoosThetaMeasuredCompact", _compact(ar.theta_au, ar.theta_au_uncertainty, 2), f
    )
    common.write_tex_macro("RoosPredictedSlope", f"{cv.predicted_slope_hz_mm2_per_v:.4f}", f)
    common.write_tex_macro("RoosResidualPct", f"{cv.residual_fractional * 100.0:+.2f}", f)
    common.write_tex_macro("RoosCrossVintageVerdict", cv.kpi_verdict, f)
    common.write_tex_macro("RoosArithPredictedSlope", f"{ar.predicted_slope_hz_mm2_per_v:.4f}", f)
    common.write_tex_macro("RoosArithResidualPct", f"{ar.residual_fractional * 100.0:+.2f}", f)
    common.write_tex_macro("RoosArithVerdict", ar.kpi_verdict, f)
    common.write_tex_macro(
        "RoosArithBandsOverlap", "overlap" if ar.bands_overlap else "do not overlap", f
    )
    common.write_tex_macro(
        "RoosEnhancementRatio", f"{case.structural_two_ion_enhancement_ratio:.1f}", f
    )
    common.write_tex_macro("IonMicromotionTensorPct", f"{_MICROMOTION_TENSOR_STARK_PCT}", f)
    common.write_tex_macro("IonQuadrupolePct", f"{_QUADRUPOLE_PCT}", f)

    print(
        f"Roos case: cross-vintage predicted {cv.predicted_slope_hz_mm2_per_v:.6f} "
        f"(residual {cv.residual_fractional * 100.0:+.2f}%, verdict {cv.kpi_verdict}); "
        f"arithmetic-reproduction predicted {ar.predicted_slope_hz_mm2_per_v:.6f} "
        f"(residual {ar.residual_fractional * 100.0:+.2f}%, verdict {ar.kpi_verdict}); "
        f"two-ion enhancement {case.structural_two_ion_enhancement_ratio}"
    )
    print(f"Wrote {common.GENERATED_DIR / 'roos_values.tex'}")


if __name__ == "__main__":
    main()
