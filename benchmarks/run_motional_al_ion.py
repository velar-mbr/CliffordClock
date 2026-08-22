# SPDX-License-Identifier: AGPL-3.0-or-later
"""WP30 benchmark case: Al+ secular-motion time-dilation **arithmetic
reproduction** (CONVENTIONS.md section 16, E38).

This script runs the *real* engine functions
(:func:`cliffordclock.integrator.omega.motional_pivot_perturbation`/
:func:`motional_pivot_uncertainty`) against a published trapped-ion clock's
own secular-motion inputs (per-mode frequency and mean vibrational
occupation `n_bar` from sideband thermometry) and compares the result
against that same paper's own published secular-motion systematic-budget
row.

**SOURCES (primary-source provenance, fetched directly this session).**

1. **arXiv:2504.13071** (Marshall, Rodriguez Castillo, Arthur-Dworschack,
   Aeppli, Kim, Lee, Warfield, Hinrichs, Nardelli, Fortier, Ye, Leibrandt,
   Hume, "High-Stability Single-Ion Clock with 5.5e-19 Systematic
   Uncertainty," v2, 2025). Byline as listed on the arXiv abstract page.
   Preprint as fetched; no journal acceptance was confirmed at fetch
   time, so it is cited as an arXiv preprint, not a peer-reviewed final
   published number.

   - Species/transition: single trapped 27Al+ ion, quantum-logic
     spectroscopy via a co-trapped 25Mg+ ion, 1S0<->3P0 clock transition.
   - **Mode frequencies + n_bar (Supplemental Material Table S2, "Secular
     motion parameters for the 25Mg+-27Al+ crystal"):** six normal modes
     (axial COM/STR, X COM/STR, Y COM/STR) of the two-ion crystal, each
     with a "Frequency [MHz]" and a "Measured n_bar" (with the table's own
     1-sigma uncertainty). Quoted verbatim in
     `benchmarks/loaders.MARSHALL_AL_ION_MODES_CITATION`; the raw
     `(name, frequency_MHz, n_bar, n_bar_uncertainty)` tuples are
     `benchmarks/loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR`.
   - **Resulting secular-motion budget row (Table I, "Secular motion"
     row): "-114.6  3.8" (units 1e-19).** Main text: "we add them in
     quadrature to get a total secular motion shift of Delta_nu/nu =
     -(114.6 +/- 3.8) x 10^-19." Quoted verbatim in
     `benchmarks/loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT`.
   - This is the ONE paper located (of the two named in this WP's design
     brief; see item 2 below) that publishes BOTH halves needed for an
     independent arithmetic reproduction (per-mode inputs AND the
     resulting budget row) in a form directly usable by this project's
     E38 formula, with no additional model machinery required.

2. **arXiv:1902.07694** (Brewer, Chen, Hankin, Clements, Chou, Wineland,
   Hume, Leibrandt, "27Al+ Quantum-Logic Clock with a Systematic
   Uncertainty below 10^-18," Phys. Rev. Lett. 123, 033201 (2019)),
   already cited in this project's `cliffordclock.ensemble.species`
   registry (`ION_MICROMOTION_NOTES["Al27+"]`: "secular motion
   -17.3(2.9)e-19"). Also publishes per-mode inputs (Supplemental Material
   Table S2) and a resulting secular-motion row (Table I: "-17.3  2.9",
   units 1e-19), but its per-mode occupation input is a 95%-confidence-
   interval BOUND on `n_bar_0` (not a symmetric point estimate; "The
   zero-point energy is not included," per its own table footnote),
   combined with a per-mode heating rate `n_bar_dot` and the 150 ms
   interrogation time via its own Eq. 3 (`Delta_nu/nu = sum_p
   (Delta_nu_p/nu)*[1/2 + n_bar_p,0 + (1/2)*n_bar_dot_p*t_i]`), not the
   static `n_bar_i + 1/2` form E38 implements. Reproducing Brewer's
   -17.3(2.9)e-19 row would require re-implementing that time-dependent
   heating-rate model, not just E38's formula, so Marshall 2025 (item 1)
   is used for this case instead, as the cleaner, more directly applicable
   published set. Brewer 2019's own numbers are NOT independently
   re-verified here (this project already carries them, unverified beyond
   the registry citation, in `ION_MICROMOTION_NOTES["Al27+"]`).

**A genuine physics caveat, flagged prominently, not silently absorbed
(corrected per the project's G11 gate record, `plan/reviews/G11-e38-motional-time-dilation.md`,
section A3).** Marshall 2025's six modes are TWO-ION (27Al+/25Mg+) crystal
normal modes. The physically complete per-mode evaluation partitions each
mode's zero-point and thermal motion between the two ions according to
their own normal-mode amplitudes (an ion-and-eigenvector-dependent
participation factor, one per ion per mode), a quantity E38's formula does
not consume: `motional_pivot_perturbation` takes one `species` (hence one
mass) for every mode, with no per-mode amplitude-partition input. This is
a genuine, documented scope boundary of this MVP tier (analogous to E37's
own single-enclosure-vs-multi-reflector scope boundary), not an oversight
discovered here. (Table S2 separately lists a per-mode "Geometric factor
kappa": per the paper's own Eq. 1, that is a Doppler-cooling-laser
geometry factor used only in Marshall's own cooling-limit calculation, has
no role in the secular-motion time-dilation row, and is not the missing
quantity described here.)

As a direct consequence, the engine's naive per-mode contributions
(`-(hbar*omega_i)/(2*m_Al*c^2)*(n_bar_i+1/2)`) are NOT Marshall's own
per-mode "Frequency shift per quantum" values: comparing them mode by mode
shows differences of up to several-fold (the project's G11 gate record
tabulates all six ratios, ranging from about 0.44x to 3.1x). Despite this
large per-mode disagreement, summing E38's naive per-mode formula over the
COMPLETE six-mode set, using the Al+ ion's own registry mass throughout
(never Mg+'s, and never a kappa-weighted effective mass), reproduces
Marshall's published `-114.6(3.8) x 10^-19` row inside both the predicted
and published uncertainty bands (see the case result below). The project's
G11 gate record derives a genuine orthogonality identity over the two-ion
normal-mode basis (each ion's squared eigenvector components sum to 1
across a complete mode set, true for any masses or spring constants) that
is qualitatively consistent with why a naive single-mass sum over a
COMPLETE mode set can land near the correct total despite large per-mode
errors; the same record shows this identity alone does not certify the
level of agreement observed here, since it would additionally require the
per-mode weight `hbar*omega_i*(n_bar_i+1/2)` to be uniform across modes,
which it is not. The mechanism behind the total-level agreement is
therefore reported as an open empirical observation, not an established
identity, and is not needed to support this case's classification below.

The open item is a full two-mass normal-mode treatment of the E38 formula,
taking each ion's own per-mode amplitude (participation) vector as an
explicit input so the per-mode arithmetic itself becomes exact, not only
the six-mode total; this belongs to the same future work package as the
RF/micromotion dynamics treatment CONVENTIONS.md section 16 already flags
as out of scope for this tier.

**Classification: `arithmetic_reproduction`, NOT a `reproducibility`/
`blind_prediction` case (mirrors `run_bbr_jila_arithmetic_reproduction.py`'s
own binding classification discipline).** Marshall et al. computed their
own `-114.6(3.8)e-19` row from the same underlying second-order-Doppler
physics E38 implements, applied per-ion with the full two-mass normal-mode
weighting this project's single-species-mass formula does not reproduce at
the per-mode level (caveat above). This case validates E38's formula and
unit chain (the hbar/2pi/mass/c^2 arithmetic, correctly wired end to end)
on the published mode list against the published TOTAL, not an
independent physical prediction and not a per-mode reproduction of
Marshall's own values. Close total agreement here demonstrates the engine
computes its formula correctly and lands inside Marshall's own uncertainty
band; it does not demonstrate that this project's simplified single-
species-mass model matches Marshall's true per-ion motional physics mode
by mode.

``case_class`` in this script's output is always the literal string
``"arithmetic_reproduction"``, kept in a separate script/report,
deliberately NOT folded into ``benchmarks/results/wp10_results.json``'s
``kpi_summary`` counts, mirroring the WP20 BBR arithmetic-reproduction
case's own isolation.

Run this yourself: ``python benchmarks/run_motional_al_ion.py`` (from the
repo root, with ``.venv`` active). Regenerates
``benchmarks/results/wp30_motional_al_ion_arithmetic_reproduction.json``
and the accompanying ``.md`` summary.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow running as `python benchmarks/run_motional_al_ion.py` (no package
# install needed: benchmarks/ is deliberately not part of the installed
# package, see benchmarks/SOURCES.md's packaging note).
_BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import loaders  # noqa: E402
import run_benchmarks  # noqa: E402 (reuses the already-tested `_bands_overlap`)

from cliffordclock.ensemble.species import get_species  # noqa: E402
from cliffordclock.integrator.omega import (  # noqa: E402
    MotionalMode,
    motional_mean_squared_velocity_m2_s2,
    motional_pivot_perturbation,
    motional_pivot_uncertainty,
)

_RESULTS_DIR = _BENCHMARKS_DIR / "results"

#: The exact, binding classification label (mirrors
#: `run_bbr_jila_arithmetic_reproduction.CASE_LABEL`'s role); callers/
#: docs/tests should compare against this constant instead of re-typing
#: the string.
CASE_LABEL = (
    "arithmetic reproduction of a published standard-formula evaluation "
    "(arithmetic-reproduction fidelity, validates the engine's E38 "
    "implementation and unit chain, NOT an independent motional-Doppler "
    "physics prediction; Marshall et al.'s own secular-motion row is "
    "itself computed from their own measured inputs through the same "
    "standard formula)"
)

#: The two-mass-normal-mode scope-boundary caveat, verbatim, folded into the
#: case's own record so a reader of the JSON/markdown output sees it
#: without having to open this module's docstring. Corrected per the
#: project's G11 gate record (`plan/reviews/G11-e38-motional-time-dilation.md`,
#: section A3): the missing physics is each ion's own per-mode normal-mode
#: amplitude, NOT Marshall's published "geometric factor kappa" (a Doppler-
#: cooling-laser geometry factor from their Eq. 1, unrelated to the
#: secular-motion time-dilation row).
GEOMETRIC_FACTOR_CAVEAT = (
    "SCOPE CAVEAT (corrected per the project's G11 gate record, section A3): "
    "Marshall et al.'s six modes are TWO-ION (27Al+/25Mg+) crystal normal modes. "
    "The physically complete per-mode evaluation partitions each mode's zero-point "
    "and thermal motion between the two ions by their own normal-mode amplitudes, "
    "a quantity this project's E38 formula does not consume (one species/mass for "
    "every mode, CONVENTIONS.md section 16): a documented scope boundary, not an "
    "oversight. As a result the engine's per-mode contributions differ from "
    "Marshall's own per-mode values by up to several-fold, while summing over the "
    "complete six-mode set reproduces their published TOTAL inside both "
    "uncertainty bands. The G11 gate record derives a genuine orthogonality "
    "identity over the two-ion normal-mode basis that is qualitatively "
    "consistent with this total-level agreement despite the per-mode "
    "differences, but that record also shows the identity alone does not "
    "certify the observed precision: the mechanism is reported as an open "
    "empirical observation, not a proven identity. The open item is a full "
    "two-mass normal-mode treatment (per-ion amplitude vectors as explicit "
    "input), belonging to the same future package as the RF/micromotion "
    "dynamics treatment already flagged out of scope for this tier."
)


@dataclass(frozen=True)
class MotionalAlIonArithmeticReproductionCase:
    """The WP30 Al+ secular-motion arithmetic-reproduction case (see module
    docstring for the full method, sources, and the geometric-factor
    caveat). Every numeric field is produced by the real engine functions
    (`motional_pivot_perturbation`, `motional_pivot_uncertainty`,
    `motional_mean_squared_velocity_m2_s2`) called with Marshall et al.'s
    own published per-mode inputs; no hand arithmetic feeds any field
    below.

    Attributes
    ----------
    case_class : str
        Always the literal string ``"arithmetic_reproduction"``.
    case_label : str
        Always :data:`CASE_LABEL`, verbatim.
    species_name : str
        Always ``"Al27+"`` (this project's registry mass for the target
        ion; NEVER 25Mg+'s mass or a kappa-weighted effective mass; see
        the geometric-factor caveat).
    n_modes : int
        Always ``6`` (Marshall et al.'s Table S2 mode count).
    mean_squared_velocity_m2_s2 : float
        ``<v^2>`` (`motional_mean_squared_velocity_m2_s2`), the sum over
        all six modes' `(hbar*omega_i/m)*(n_bar_i+1/2)` contributions.
    predicted_shift_nominal : float
        ``motional_pivot_perturbation`` at Marshall's own six
        `(frequency, n_bar)` pairs and the Al27+ registry mass.
    predicted_uncertainty_fractional : float
        ``motional_pivot_uncertainty`` propagated from Marshall's own
        per-mode `n_bar` uncertainties (frequency uncertainties are not
        published in Table S2, so `frequency_uncertainty_hz` is left at
        its `0.0` default for every mode).
    predicted_band_lo, _hi : float
        ``predicted_shift_nominal +/- predicted_uncertainty_fractional``.
    published_shift_nominal, _lo, _hi : float
        Marshall et al.'s own published secular-motion row
        (`loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT`).
    residual_fractional : float
        ``predicted_shift_nominal - published_shift_nominal`` (point
        estimate; NOT scaled by any tolerance).
    bands_overlap : bool
        Whether ``[predicted_band_lo, predicted_band_hi]`` and
        ``[published_shift_lo, published_shift_hi]`` overlap, per
        `run_benchmarks._bands_overlap`'s precise closed-interval
        definition (reused, not re-implemented).
    kpi_verdict : str
        ``"MET"`` if `bands_overlap` else ``"NOT MET"``; never
        ``"PASS"``/``"FAIL"``, this project's reserved vocabulary.
    modes_citation, published_shift_citation, geometric_factor_caveat : str
        Exact source citations, and the scope-boundary caveat, verbatim.
    """

    case_class: str
    case_label: str
    species_name: str
    n_modes: int
    mean_squared_velocity_m2_s2: float
    predicted_shift_nominal: float
    predicted_uncertainty_fractional: float
    predicted_band_lo: float
    predicted_band_hi: float
    published_shift_nominal: float
    published_shift_lo: float
    published_shift_hi: float
    residual_fractional: float
    bands_overlap: bool
    kpi_verdict: str
    modes_citation: str
    published_shift_citation: str
    geometric_factor_caveat: str


def run_motional_al_ion_arithmetic_reproduction_case() -> MotionalAlIonArithmeticReproductionCase:
    """Build the WP30 Al+ secular-motion arithmetic-reproduction case.

    Method (mirrors
    `run_bbr_jila_arithmetic_reproduction.run_jila_bbr_arithmetic_reproduction_case`'s
    discipline: real engine calls, no algebraic shortcut standing in for
    one):

    1. Resolve `Al27+` from the species registry (its own registry mass,
       `mass_kg`, never Mg+'s or a kappa-weighted effective mass).
    2. Build six `MotionalMode` entries directly from
       `loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR` (converting MHz to Hz;
       `motional_pivot_perturbation` performs the ``omega = 2*pi*f``
       conversion internally).
    3. Call `motional_pivot_perturbation`/`motional_mean_squared_velocity_m2_s2`
       once each (real engine calls).
    4. Call `motional_pivot_uncertainty` once, propagating each mode's own
       published `n_bar` uncertainty (no `frequency_uncertainty_hz` is
       published in Table S2, so it stays at its `0.0` default).
    5. Compare the resulting band against Marshall et al.'s own published
       band (`loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT`) via
       `run_benchmarks._bands_overlap`.

    Returns
    -------
    MotionalAlIonArithmeticReproductionCase
        `kpi_verdict` is `"MET"` if the two bands overlap, else
        `"NOT MET"`.
    """
    species = get_species("Al27+")
    modes = tuple(
        MotionalMode(
            name=name,
            frequency_hz=frequency_mhz * 1.0e6,
            n_bar=n_bar,
            n_bar_uncertainty=n_bar_uncertainty,
        )
        for name, frequency_mhz, n_bar, n_bar_uncertainty in loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR
    )

    mean_v2 = motional_mean_squared_velocity_m2_s2(modes, species)
    predicted_nominal = motional_pivot_perturbation(modes, species)
    predicted_sigma = motional_pivot_uncertainty(modes, species)

    published = loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT
    residual = predicted_nominal - published.nominal

    pred_band_lo = predicted_nominal - predicted_sigma
    pred_band_hi = predicted_nominal + predicted_sigma
    overlap = run_benchmarks._bands_overlap(  # noqa: SLF001 (reusing the tested helper)
        pred_band_lo, pred_band_hi, published.lo, published.hi
    )

    return MotionalAlIonArithmeticReproductionCase(
        case_class="arithmetic_reproduction",
        case_label=CASE_LABEL,
        species_name="Al27+",
        n_modes=len(modes),
        mean_squared_velocity_m2_s2=mean_v2,
        predicted_shift_nominal=predicted_nominal,
        predicted_uncertainty_fractional=predicted_sigma,
        predicted_band_lo=pred_band_lo,
        predicted_band_hi=pred_band_hi,
        published_shift_nominal=published.nominal,
        published_shift_lo=published.lo,
        published_shift_hi=published.hi,
        residual_fractional=residual,
        bands_overlap=overlap,
        kpi_verdict="MET" if overlap else "NOT MET",
        modes_citation=loaders.MARSHALL_AL_ION_MODES_CITATION,
        published_shift_citation=published.citation,
        geometric_factor_caveat=GEOMETRIC_FACTOR_CAVEAT,
    )


def build_report() -> dict[str, Any]:
    """Build the full WP30 motional-Al-ion benchmark report as a
    JSON-serializable dict.

    Returns
    -------
    dict[str, Any]
        Metadata plus the single case (see
        :func:`run_motional_al_ion_arithmetic_reproduction_case`).
        Deliberately NOT merged into `run_benchmarks.build_report`'s WP10
        report or `kpi_summary`; see module docstring.
    """
    case = run_motional_al_ion_arithmetic_reproduction_case()
    return {
        "wp30_motional_al_ion_benchmark_schema": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "case_label": CASE_LABEL,
        "case_class": case.case_class,
        "marshall_2504_13071_secular_motion_arithmetic_reproduction_case": asdict(case),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the WP30 motional-Al-ion benchmark case as a markdown
    summary, mirroring `run_bbr_jila_arithmetic_reproduction.render_markdown`'s
    style.

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
    case = report["marshall_2504_13071_secular_motion_arithmetic_reproduction_case"]
    lines = [
        "# WP30 motional Al+ ion benchmark case (generated)",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Arithmetic-reproduction case: Marshall et al. arXiv:2504.13071 Table I "
        '"Secular motion" row',
        "",
        f"**Classification label (binding): {CASE_LABEL}**",
        "",
        f"**{GEOMETRIC_FACTOR_CAVEAT}**",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Species | {case['species_name']} |",
        f"| Number of modes | {case['n_modes']} |",
        f"| <v^2> (m^2/s^2) | {case['mean_squared_velocity_m2_s2']:.6e} |",
        f"| Predicted (P-1)_motional | {case['predicted_shift_nominal']:+.6e} |",
        f"| Predicted uncertainty (1-sigma) | +/-{case['predicted_uncertainty_fractional']:.3e} |",
        (
            "| Predicted band | "
            f"[{case['predicted_band_lo']:+.6e}, {case['predicted_band_hi']:+.6e}] |"
        ),
        (
            '| Published (Marshall Table I "Secular motion") | '
            f"{case['published_shift_nominal']:+.6e} |"
        ),
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
        "separate report. See this script's module docstring for the full "
        "SOURCES provenance, the Brewer 2019 alternative-source discussion, "
        "and the two-mass normal-mode scope caveat.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the WP30 motional-Al-ion benchmark case and write
    `benchmarks/results/wp30_motional_al_ion_arithmetic_reproduction.json`
    and a generated markdown summary alongside it."""
    report = build_report()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / "wp30_motional_al_ion_arithmetic_reproduction.json"
    md_path = _RESULTS_DIR / "wp30_motional_al_ion_arithmetic_reproduction.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
