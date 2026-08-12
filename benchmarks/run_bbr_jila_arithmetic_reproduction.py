# SPDX-License-Identifier: AGPL-3.0-or-later
"""WP20 benchmark case: JILA BBR-row **arithmetic reproduction**
(WP20 design item 5, gate edit 8;
the project's theory sign-off record (G7) B5).

This script runs the *real* engine functions
(:func:`cliffordclock.integrator.omega.bbr_pivot_perturbation`/
:func:`bbr_pivot_uncertainty`) against the pinned species-registry Sr87
BBR coefficients (:data:`cliffordclock.ensemble.species.SR87.bbr_coefficients`
-- the PTB-2025 rescaled dynamic polynomial, ``{6: -0.13216, 8: -0.01231,
10: -0.00858}`` Hz) at JILA's own published operating temperature
(``T = 293.282(4) K``, ``benchmarks/loaders.JILA_BBR_TEMPERATURE_K``), and
compares the result against JILA's own published BBR row
(``benchmarks/loaders.JILA_BBR_PUBLISHED_SHIFT``, arXiv:2403.10664v2 Table
I: ``-4.84172(73)e-15``).

**Binding classification label (G7 sign-off B5, ratified): "arithmetic
reproduction of a published standard-formula evaluation."** This is
explicitly a WEAKER class than ``benchmarks/run_benchmarks.py``'s NPL
``case_class = "reproducibility"`` case:

- The NPL case reconstructs a shift from a field that was measured
  **independently of the clock transition** (Rydberg-EIT spectroscopy on
  a different atomic state).
- This case reconstructs a shift from JILA's own **computed** row: their
  own measured temperature run through the same standard BBR formula
  with their own coefficients. Nothing about the shift itself was
  independently measured the way NPL's field was.
- The registry's dynamic polynomial is itself the PTB-2025 rescaling of
  Lisdat's fit shape, anchored (renormalized) to JILA's own
  ``-153.06(33) mHz`` dynamic-term value (arXiv:2507.14030) -- so close
  agreement here is **expected almost by construction**, not a
  surprising independent success. What this case demonstrates is the
  engine's arithmetic and provenance chain end-to-end (registry
  coefficients -> ``bbr_pivot_perturbation``/``bbr_pivot_uncertainty`` ->
  a number that lands inside JILA's own published band) -- NOT
  independent BBR physics validation.
- Any ``1e-19``-class number this case reports is **"arithmetic-
  reproduction fidelity," never "BBR accuracy"** (G7 sign-off A4#2c/B5c).

``case_class`` in this script's output is always the literal string
``"arithmetic_reproduction"`` -- structurally distinct from
``run_benchmarks.py``'s ``"reproducibility"`` and ``"blind_prediction"``
categories, and deliberately kept in a **separate** script/report/summary
rather than folded into ``benchmarks/results/wp10_results.json``'s
``kpi_summary`` counts, so the WP10 pinned totals (14 rows considered,
1 reproducibility case) never change as a side effect of this WP20
addition.

Run this yourself: ``python benchmarks/run_bbr_jila_arithmetic_reproduction.py``
(from the repo root, with ``.venv`` active). Regenerates
``benchmarks/results/wp20_bbr_arithmetic_reproduction.json`` and
``benchmarks/results/wp20_bbr_arithmetic_reproduction.md``.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow running as `python benchmarks/run_bbr_jila_arithmetic_reproduction.py`
# (no package install needed -- benchmarks/ is deliberately not part of the
# installed package, see benchmarks/SOURCES.md's packaging note), and mirror
# how `benchmarks/run_benchmarks.py` imports its sibling `loaders` module.
_BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import loaders  # noqa: E402
import run_benchmarks  # noqa: E402 -- reuses the already-tested `_bands_overlap`

from cliffordclock.ensemble.species import get_species  # noqa: E402
from cliffordclock.integrator.omega import (  # noqa: E402
    bbr_pivot_perturbation,
    bbr_pivot_uncertainty,
)

_RESULTS_DIR = _BENCHMARKS_DIR / "results"

#: The exact, binding classification label (G7 sign-off B5) -- callers/docs/tests
#: should compare against this constant rather than re-typing the string,
#: so a future edit to the wording is a single-source change.
CASE_LABEL = (
    "arithmetic reproduction of a published standard-formula evaluation "
    "(arithmetic-reproduction fidelity, NOT BBR accuracy; explicitly weaker "
    "than a reproducibility case; JILA's own BBR row is itself computed, "
    "not an independently measured shift)"
)

_REGISTRY_COEFFICIENT_CITATION = (
    "cliffordclock.ensemble.species.SR87.bbr_coefficients: static "
    "nu_stat_300k_hz=-2.13023(6) Hz (T. Middelmann, S. Falke, C. Lisdat, "
    "U. Sterr, Phys. Rev. Lett. 109, 263004 (2012)); dynamic "
    "dyn_coeffs_hz={6: -0.13216, 8: -0.01231, 10: -0.00858} Hz, the "
    "PTB-2025 rescaled polynomial (arXiv:2507.14030) -- Lisdat et al. PR "
    "Research 3, L042036 (2021)'s fit SHAPE rescaled to Aeppli et al. "
    "arXiv:2403.10664 (2024)'s own -153.06(33) mHz dynamic-term ANCHOR at "
    "300 K. See the project's theory sign-off record (G7) B2 for the "
    "shape-vs-anchor reasoning the G7 gate ratified."
)


@dataclass(frozen=True)
class JilaBbrArithmeticReproductionCase:
    """The WP20 JILA-BBR-row arithmetic-reproduction case (see module
    docstring for the full method and the binding classification-labeling
    rationale). Every numeric field is produced by the real engine
    functions (`bbr_pivot_perturbation`, `bbr_pivot_uncertainty`) called
    with the pinned species-registry Sr87 coefficients -- no hand
    arithmetic feeds any field below.

    Attributes
    ----------
    case_class : str
        Always the literal string ``"arithmetic_reproduction"`` --
        distinct from `run_benchmarks.py`'s ``"reproducibility"`` and
        ``"blind_prediction"`` categories (see module docstring).
    case_label : str
        Always :data:`CASE_LABEL`, verbatim.
    species_name : str
        Always ``"Sr87"`` (the only species JILA's row applies to).
    temperature_k, temperature_uncertainty_k : float
        JILA's published operating temperature and its 1-sigma
        uncertainty (`loaders.JILA_BBR_TEMPERATURE_K`).
    predicted_shift_nominal : float
        ``bbr_pivot_perturbation(temperature_k, Sr87)`` -- the engine's
        BBR pivot term at JILA's nominal temperature, using ONLY the
        pinned registry polynomial (not the dossier's single-T^6 quick
        estimate).
    predicted_shift_temperature_band_lo, _hi : float
        `bbr_pivot_perturbation` evaluated at ``temperature_k -
        temperature_uncertainty_k`` / ``+ temperature_uncertainty_k`` --
        the temperature-uncertainty-only band (G7 sign-off A4#3), each a
        direct, independent engine call (not a derivative estimate).
    predicted_coefficient_uncertainty_fractional : float
        `bbr_pivot_uncertainty(temperature_k, Sr87)` with no
        `temperature_uncertainty_k` argument -- the registry
        coefficient-uncertainty-only band half-width (G7 sign-off A4#2),
        "arithmetic-reproduction fidelity," never "BBR accuracy."
    predicted_combined_uncertainty_fractional : float
        `bbr_pivot_uncertainty(temperature_k, Sr87, temperature_uncertainty_k)`
        -- coefficient and temperature uncertainty combined in quadrature
        by the engine itself. This is the band used for the overlap/KPI
        test below (the most complete of the three bands).
    predicted_combined_band_lo, _hi : float
        ``predicted_shift_nominal +/- predicted_combined_uncertainty_fractional``.
    published_shift_nominal, _lo, _hi : float
        JILA's own published BBR row (`loaders.JILA_BBR_PUBLISHED_SHIFT`).
    residual_fractional : float
        ``predicted_shift_nominal - published_shift_nominal`` (point
        estimate; NOT scaled by any tolerance).
    bands_overlap : bool
        Whether ``[predicted_combined_band_lo, predicted_combined_band_hi]``
        and ``[published_shift_lo, published_shift_hi]`` overlap, per
        `run_benchmarks._bands_overlap`'s precise closed-interval
        definition (reused, not re-implemented).
    kpi_verdict : str
        ``"MET"`` if `bands_overlap` else ``"NOT MET"`` -- never
        ``"PASS"``/``"FAIL"``, this project's reserved vocabulary.
    temperature_citation, published_shift_citation, registry_coefficient_citation : str
        Exact source citations for every input.
    """

    case_class: str
    case_label: str
    species_name: str
    temperature_k: float
    temperature_uncertainty_k: float
    predicted_shift_nominal: float
    predicted_shift_temperature_band_lo: float
    predicted_shift_temperature_band_hi: float
    predicted_coefficient_uncertainty_fractional: float
    predicted_combined_uncertainty_fractional: float
    predicted_combined_band_lo: float
    predicted_combined_band_hi: float
    published_shift_nominal: float
    published_shift_lo: float
    published_shift_hi: float
    residual_fractional: float
    bands_overlap: bool
    kpi_verdict: str
    temperature_citation: str
    published_shift_citation: str
    registry_coefficient_citation: str


def run_jila_bbr_arithmetic_reproduction_case() -> JilaBbrArithmeticReproductionCase:
    """Build the WP20 JILA-BBR-row arithmetic-reproduction case.

    Method (mirrors `run_benchmarks.run_npl_reproducibility_case`'s
    discipline -- real engine calls, no algebraic shortcut standing in
    for one):

    1. Resolve `Sr87` from the species registry (the pinned PTB-2025
       rescaled dynamic polynomial + Middelmann static term).
    2. Call `bbr_pivot_perturbation` at JILA's nominal temperature
       (``293.282 K``) and independently at ``T +/- 0.004 K`` (the
       published temperature uncertainty) -- three direct engine calls,
       giving the temperature band.
    3. Call `bbr_pivot_uncertainty` twice: once without a temperature
       uncertainty (the coefficient-uncertainty-only band), once with
       ``temperature_uncertainty_k=0.004`` (the combined band used for
       the overlap/KPI test).
    4. Compare the combined band against JILA's own published band
       (`loaders.JILA_BBR_PUBLISHED_SHIFT`) via
       `run_benchmarks._bands_overlap`.

    Returns
    -------
    JilaBbrArithmeticReproductionCase
        `kpi_verdict` is `"MET"` if the two bands overlap, else
        `"NOT MET"`.

    Raises
    ------
    AssertionError
        If `loaders.JILA_BBR_TEMPERATURE_K`'s band is not symmetric
        about its nominal value (a malformed-fixture guard -- the
        published `+/-0.004 K` uncertainty is symmetric; an asymmetric
        band here would indicate a transcription error, not a real
        physical asymmetry), or if `bbr_pivot_uncertainty` unexpectedly
        fails to include the temperature-uncertainty contribution.
    """
    species = get_species("Sr87")
    temperature = loaders.JILA_BBR_TEMPERATURE_K
    t_nominal = temperature.nominal
    t_unc_lo = t_nominal - temperature.lo
    t_unc_hi = temperature.hi - t_nominal
    assert math.isclose(t_unc_lo, t_unc_hi, rel_tol=1e-12), (
        f"JILA_BBR_TEMPERATURE_K band is not symmetric ({t_unc_lo=!r} != {t_unc_hi=!r}) "
        "-- check benchmarks/loaders.py for a transcription error"
    )
    t_unc = t_unc_hi

    predicted_nominal = bbr_pivot_perturbation(t_nominal, species)
    predicted_t_lo = bbr_pivot_perturbation(temperature.lo, species)
    predicted_t_hi = bbr_pivot_perturbation(temperature.hi, species)

    sigma_coeff_frac, _coeff_only = bbr_pivot_uncertainty(t_nominal, species)
    sigma_combined_frac, t_included = bbr_pivot_uncertainty(t_nominal, species, t_unc)
    assert t_included, (
        "bbr_pivot_uncertainty did not include the temperature-uncertainty "
        "contribution despite temperature_uncertainty_k being passed"
    )

    published = loaders.JILA_BBR_PUBLISHED_SHIFT
    residual = predicted_nominal - published.nominal

    pred_band_lo = predicted_nominal - sigma_combined_frac
    pred_band_hi = predicted_nominal + sigma_combined_frac
    overlap = run_benchmarks._bands_overlap(  # noqa: SLF001 -- reusing the tested helper
        pred_band_lo, pred_band_hi, published.lo, published.hi
    )

    return JilaBbrArithmeticReproductionCase(
        case_class="arithmetic_reproduction",
        case_label=CASE_LABEL,
        species_name="Sr87",
        temperature_k=t_nominal,
        temperature_uncertainty_k=t_unc,
        predicted_shift_nominal=predicted_nominal,
        predicted_shift_temperature_band_lo=min(predicted_t_lo, predicted_t_hi),
        predicted_shift_temperature_band_hi=max(predicted_t_lo, predicted_t_hi),
        predicted_coefficient_uncertainty_fractional=sigma_coeff_frac,
        predicted_combined_uncertainty_fractional=sigma_combined_frac,
        predicted_combined_band_lo=pred_band_lo,
        predicted_combined_band_hi=pred_band_hi,
        published_shift_nominal=published.nominal,
        published_shift_lo=published.lo,
        published_shift_hi=published.hi,
        residual_fractional=residual,
        bands_overlap=overlap,
        kpi_verdict="MET" if overlap else "NOT MET",
        temperature_citation=temperature.citation,
        published_shift_citation=published.citation,
        registry_coefficient_citation=_REGISTRY_COEFFICIENT_CITATION,
    )


def build_report() -> dict[str, Any]:
    """Build the full WP20 BBR arithmetic-reproduction report as a
    JSON-serializable dict.

    Returns
    -------
    dict[str, Any]
        Metadata plus the single case (see
        :func:`run_jila_bbr_arithmetic_reproduction_case`). Deliberately
        NOT merged into `run_benchmarks.build_report`'s WP10 report or
        `kpi_summary` -- see module docstring.
    """
    case = run_jila_bbr_arithmetic_reproduction_case()
    return {
        "wp20_bbr_benchmark_schema": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "case_label": CASE_LABEL,
        "case_class": case.case_class,
        "jila_2403_10664_bbr_arithmetic_reproduction_case": asdict(case),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the WP20 BBR arithmetic-reproduction case as a markdown
    summary, mirroring `run_benchmarks.render_markdown_table`'s style.

    Parameters
    ----------
    report : dict[str, Any]
        A report dict as returned by :func:`build_report`.

    Returns
    -------
    str
        A markdown document suitable for embedding or diffing against
        `benchmarks/RESULTS.md`.
    """
    case = report["jila_2403_10664_bbr_arithmetic_reproduction_case"]
    lines = [
        "# WP20 BBR benchmark case (generated)",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Arithmetic-reproduction case: JILA arXiv:2403.10664 Table I 'BBR' row",
        "",
        f"**Classification label (binding, G7 sign-off B5): {CASE_LABEL}**",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Species | {case['species_name']} |",
        (
            f"| Temperature T | {case['temperature_k']:.3f} +/- "
            f"{case['temperature_uncertainty_k']:.3f} K |"
        ),
        f"| Predicted (P-1)_BBR (nominal T) | {case['predicted_shift_nominal']:+.6e} |",
        (
            "| Predicted, temperature band (T +/- 0.004 K) | "
            f"[{case['predicted_shift_temperature_band_lo']:+.6e}, "
            f"{case['predicted_shift_temperature_band_hi']:+.6e}] |"
        ),
        (
            "| Predicted, coefficient-uncertainty band | "
            f"+/-{case['predicted_coefficient_uncertainty_fractional']:.3e} |"
        ),
        (
            "| Predicted, combined (coefficient+T) band | "
            f"[{case['predicted_combined_band_lo']:+.6e}, "
            f"{case['predicted_combined_band_hi']:+.6e}] |"
        ),
        f"| Published (JILA Table I 'BBR') | {case['published_shift_nominal']:+.6e} |",
        (
            "| Published band | "
            f"[{case['published_shift_lo']:+.6e}, {case['published_shift_hi']:+.6e}] |"
        ),
        f"| Residual (predicted - published) | {case['residual_fractional']:+.3e} |",
        f"| Bands overlap | {case['bands_overlap']} |",
        f"| **kpi_verdict** | **{case['kpi_verdict']}** |",
        "",
        "This is NOT counted toward `benchmarks/results/wp10_results.json`'s "
        "`kpi_summary` (reproducibility/blind-prediction/not-applicable) "
        "totals; it is a structurally distinct, weaker class "
        '(`case_class = "arithmetic_reproduction"`), tracked in this '
        "separate report. See `benchmarks/RESULTS.md` for the full "
        "write-up and why agreement here is expected almost by "
        "construction.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the WP20 BBR benchmark case and write
    `benchmarks/results/wp20_bbr_arithmetic_reproduction.json` and a
    generated markdown summary alongside it."""
    report = build_report()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / "wp20_bbr_arithmetic_reproduction.json"
    md_path = _RESULTS_DIR / "wp20_bbr_arithmetic_reproduction.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
