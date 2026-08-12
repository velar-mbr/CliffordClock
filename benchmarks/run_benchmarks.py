# SPDX-License-Identifier: AGPL-3.0-or-later
"""WP10 benchmark runner: comparison, reported as found, against the authorized
public dataset sources (``benchmarks/SOURCES.md``: arXiv:2403.10664,
data.nist.gov DOI 10.18434/M32206, the PRL 133,023401 Supplemental
Material follow-up, and -- second follow-up, 2026-08-10 -- arXiv:1706.01944
(NPL Rydberg electrometry) and Metrologia 63, 025002 (2026) (USTC Sr1)).

This script is a **script**, not a unit test (WP10 section 4: "the full
benchmark run itself is a script ... not a unit test"). It:

1. Loads the JILA arXiv:2403.10664 Table I systematic-shift budget
   (``benchmarks/fixtures/jila_2403_10664_table1.csv``) and classifies
   every row as in/out of this engine's current physics scope
   (CONVENTIONS.md E14b scalar DC Stark + E21 second-order Doppler only).
2. For the one in-scope row (DC Stark), documents -- rather than fakes --
   why no independent predicted-vs-published residual is possible (see
   ``benchmarks/MAPPING.md``): the paper reports only the resulting
   fractional shift, never the residual stray-field magnitude (V/m) that
   produced it, and there is no second, independent route to that field
   from the paper's other numbers. Solving for the field that would
   reproduce their number is not a benchmark -- it is exactly the "tuned
   parameter" the WP10 labeling discipline forbids, so this script never
   does it and reports as much.
3. Runs a genuine, *non-tuned* illustrative sweep of the pipeline's
   ``coupling.type: stark_dc`` DC-Stark shift (Sr87, lattice fast path,
   1 s interrogation -- the same machinery KA1 in ``tests/test_known_answers.py``
   validates) over round-number stray-field magnitudes, so the JILA
   number has physical context (a scale on the same axis) without ever
   masquerading as a residual against it. The swept field values are
   fixed constants chosen before this script ever computed anything
   (1/5/10/20/50/100 V/m), never adjusted afterward.
4. Loads the NIST DOI 10.18434/M32206 phase-vs-time CSVs and classifies
   them: this is Allan-deviation/phase-instability data for an optical-
   to-microwave frequency-division scheme, not a systematic-shift/field-
   gradient measurement -- there is no pipeline config this data maps to
   at all (see ``benchmarks/MAPPING.md``).
5. **Runs the NPL reproducibility case** (:func:`run_npl_reproducibility_case`):
   the one authorized source that publishes an independent field
   magnitude for the DC-Stark systematic. Runs the real
   ``coupling.type: stark_dc`` pipeline at the field's asymmetric
   uncertainty bounds and compares the resulting predicted band against
   NPL's own published shift band -- labeled a
   *reproducibility* check (this engine + PTB's Delta-alpha reconstructs
   what NPL themselves already computed from the same two published
   ingredients), never a "blind prediction" or "validation against an
   independent measurement of the shift" (see ``benchmarks/MAPPING.md``
   for why that framing would be a misrepresentation).
6. Classifies the USTC Metrologia 63,025002 DC-Stark budget constraint
   (:func:`classify_ustc_dc_stark`) -- same structural class as the JILA
   row: in scope, but no independent field magnitude published in *this*
   paper (see ``benchmarks/MAPPING.md`` for the ref-[30] follow-up note).
7. Emits ``benchmarks/results/wp10_results.json`` (machine-readable) and
   prints/writes a markdown summary table.

No parameter here is fitted, tuned, or selected to reduce a residual
(WP10 section 3, binding). Every published number traces to
``benchmarks/SOURCES.md``/``benchmarks/MAPPING.md`` with an exact
table/section/page citation.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow running as `python benchmarks/run_benchmarks.py` (no package install
# needed -- benchmarks/ is deliberately not part of the installed package,
# see the module docstring and benchmarks/SOURCES.md).
_BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import loaders  # noqa: E402

from cliffordclock.ensemble.species import get_species  # noqa: E402
from cliffordclock.pipeline import PipelineConfig, run_pipeline_full  # noqa: E402

_FIXTURES_DIR = _BENCHMARKS_DIR / "fixtures"
_RESULTS_DIR = _BENCHMARKS_DIR / "results"

#: Illustrative stray-field sweep magnitudes (V/m), fixed before any
#: comparison is computed -- see module docstring item 3. Chosen as
#: round numbers spanning "well-shielded lab apparatus" (a few V/m) to
#: "unshielded stray patch field" (the WP11 realistic-worked-example
#: regime, `docs/CONVENTIONS.md`/`examples/realistic_lattice_sr87.yaml`).
DC_STARK_CONTEXT_SWEEP_V_PER_M: tuple[float, ...] = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0)

_TRAP_OMEGA_XYZ = (2.0e5, 2.0e5, 2.0e5)  # rad/s, matches tests/test_known_answers.py KA1/KA2
_INTERROGATION_TIME_S = 1.0


@dataclass(frozen=True)
class JilaCaseVerdict:
    """One "budget-only" row's classification for the WP10 benchmark table
    -- a published systematic-shift/constraint value with no independent
    input this engine could forward-predict from. Used for every JILA
    arXiv:2403.10664 Table I row *and* (see :func:`classify_ustc_dc_stark`)
    the USTC Metrologia 63,025002 DC-Stark constraint, which is the same
    structural class (in scope, but no independent field magnitude
    published) -- kept under this name for continuity with the original
    WP10 pass rather than renamed mid-project.
    """

    shift_name: str
    published_shift_fractional: float
    published_uncertainty_fractional: float
    uncertainty_is_upper_bound: bool
    in_engine_scope: bool
    comparable: bool
    kpi_verdict: str  # "N/A" for every budget-only row (never "PASS"/"FAIL")
    reason: str
    citation: str


@dataclass(frozen=True)
class DcStarkSweepPoint:
    """One point of the illustrative (non-comparison) DC-Stark field sweep."""

    field_v_per_m: float
    predicted_fractional_shift: float
    predicted_sem: float | None


@dataclass(frozen=True)
class NistCaseVerdict:
    """A NIST M32206 data file's classification for the WP10 benchmark table."""

    source_file: str
    excerpt_n_samples: int
    full_dataset_n_samples: int
    phase_units: str
    comparable: bool
    kpi_verdict: str
    reason: str
    citation: str


@dataclass(frozen=True)
class NplReproducibilityCase:
    """The WP10 follow-up (2026-08-10) "NPL reproducibility case": this
    engine's ``coupling.type: stark_dc`` pipeline reconstructs NPL's
    published DC-Stark shift from NPL's published residual field and
    PTB's (Middelmann et al.) published differential polarizability --
    the exact same two ingredients NPL themselves combined.

    **Classification labeling (binding, WP10 follow-up instruction):** this is a
    ``case_class = "reproducibility"`` case, categorically distinct from
    a ``"blind_prediction"`` case (still zero of those -- see
    `benchmarks/MAPPING.md`). It demonstrates end-to-end pipeline
    correctness against two independently published inputs/outputs with
    no fitting -- it does NOT demonstrate that this engine predicts a
    clock shift NPL had not already computed themselves from the same
    inputs. `kpi_verdict` is `"MET"`/`"NOT MET"` (never `"PASS"`/`"FAIL"`,
    reserved vocabulary this project never uses for any WP10 case) --
    "MET" means the predicted band and NPL's published band overlap
    (:func:`_bands_overlap`), not that every digit matches.
    """

    case_class: str  # always "reproducibility"
    field_lo_v_per_m: float
    field_nominal_v_per_m: float
    field_hi_v_per_m: float
    field_citation: str
    predicted_shift_lo: float  # from field_hi_v_per_m (largest |E| -> most negative shift)
    predicted_shift_nominal: float
    predicted_shift_hi: float  # from field_lo_v_per_m (smallest |E| -> least negative shift)
    published_shift_lo: float
    published_shift_nominal: float
    published_shift_hi: float
    published_shift_citation: str
    bands_overlap: bool
    kpi_verdict: str  # "MET" or "NOT MET"
    uncertainty_propagation_method: str
    species_name: str
    delta_alpha_dc_si: float
    delta_alpha_citation: str


_JILA_CITATION = (
    'Aeppli, Kim, Warfield, Safronova, Ye, "A clock with 8x10^-19 systematic '
    'uncertainty", arXiv:2403.10664v2 (2024), Table I "Fractional frequency '
    'shifts and uncertainties for the JILA 1D Sr optical lattice clock"'
)

_JILA_DC_STARK_NOT_COMPARABLE_REASON = (
    "In engine physics scope (CONVENTIONS.md E14b scalar DC Stark) but not "
    "an independent forward-comparable case: the paper's main text reports "
    "only the resulting fractional shift (-9.8+/-0.7e-20) and never the "
    "residual stray-field magnitude (V/m) that produced it. The PRL 133, "
    "023401 Supplemental Material was also checked (owner-authorized "
    "follow-up, 2026-08-10): it could not be retrieved directly (paywalled; "
    "no credential/paywall bypass attempted), but a direct cross-check "
    "against the already-fetched arXiv v2 text (same LaTeX sources) shows "
    "its five sections cover five other systematics (lifetime/BBR, "
    "temperature, first-order Zeeman, background gas) and the DC-Stark "
    "paragraph cites no supplemental reference at all -- so it would not "
    "have added a field value either. Building a pipeline config would "
    "require guessing a field and then either (a) reporting a residual "
    "that is meaningless because the input was never independently known, "
    "or (b) solving for the field that reproduces their number, which is "
    "exactly the 'tuned parameter' WP10's labeling discipline forbids (section 3: "
    "'No parameter may be fitted, tuned, or selected to reduce residuals'). "
    "See benchmarks/MAPPING.md and benchmarks/SOURCES.md section 3 for the "
    "full reasoning and the illustrative (explicitly non-comparison) field "
    "sweep this script runs instead."
)

_NIST_NOT_COMPARABLE_REASON = (
    "This file is a phase-vs-time record (optical-to-microwave frequency "
    "division instability, Allan-deviation-type data per the dataset's own "
    "description) from a completely different measurement category than "
    "this engine computes: static/quasi-static systematic frequency SHIFTS "
    "(CONVENTIONS.md E14b DC Stark, E21 second-order Doppler), not short- "
    "term phase/frequency INSTABILITY of a specific down-conversion scheme. "
    "There is no field-gradient, temperature, or trap parameter in this "
    "data to map to a pipeline config, and computing an Allan deviation "
    "from it to compare against would require new analytics functionality "
    "outside WP10's scope (non-goal: no new physics/analytics modules). "
    "See benchmarks/MAPPING.md."
)

_NIST_CITATION = (
    "T. Nakamura et al., \"Data for 'Coherent Optical Clock Down-Conversion "
    "for Microwave Frequencies with 10-18 Instability'\", "
    "data.nist.gov DOI 10.18434/M32206 (2020), referencing arXiv:2003.02923"
)

#: Full published file row counts (see benchmarks/SOURCES.md) -- the
#: shipped fixtures are the first 20 rows of each file, since a
#: not-comparable dataset does not warrant committing the full ~2.2 MB
#: files (see benchmarks/SOURCES.md's "what was fetched but not
#: committed" note; `benchmarks/fetch_data.py` retrieves the full files
#: on demand).
_NIST_FULL_DATASET_N_SAMPLES = 44002

_USTC_CITATION = (
    'Jia et al., "Improved systematic evaluation of a strontium optical '
    'clock with uncertainty below 1e-18", Metrologia 63, 025002 (2026), '
    "DOI 10.1088/1681-7575/ae449e, CC BY 4.0, Table 3 + Sec. 3.5 "
    '"Residual DC Stark shift"'
)

_USTC_DC_STARK_NOT_COMPARABLE_REASON = (
    "In engine physics scope (CONVENTIONS.md E14b scalar DC Stark) but not "
    "an independent forward-comparable case -- same structural class as "
    "the JILA arXiv:2403.10664 DC Stark row (benchmarks/MAPPING.md): the "
    "paper constrains the TOTAL shift (0.0(0.1)e-19, Table 3 / Sec. 3.5) "
    "via geometry/shielding-factor reasoning (viewport distances 142 mm "
    "vs 237 mm, an 8x geometric factor, a 3x FE-simulated shielding "
    "factor) applied to a PRIOR measurement -- a y-component shift of "
    "1.4(5.2)e-21 -- cited to their ref [30] (Li J et al 2024 Metrologia "
    "61 015006). That prior paper, per this paper's own description, is "
    "the one that actually characterized an applied external field -- but "
    "it is NOT among this project's authorized sources and was NOT "
    "fetched (flagged in benchmarks/MAPPING.md as the next authorization "
    "candidate). No independent field magnitude (V/m) is published in "
    "THIS paper for the DC-Stark constraint itself."
)


def classify_ustc_dc_stark() -> JilaCaseVerdict:
    """Classify the USTC Metrologia 63,025002 DC-Stark budget constraint.

    Returns
    -------
    JilaCaseVerdict
        Always `comparable=False` -- see `_USTC_DC_STARK_NOT_COMPARABLE_REASON`.
    """
    entry = loaders.USTC_DC_STARK_CONSTRAINT
    return JilaCaseVerdict(
        shift_name=entry.shift_name,
        published_shift_fractional=entry.shift_fractional,
        published_uncertainty_fractional=entry.uncertainty_fractional,
        uncertainty_is_upper_bound=entry.uncertainty_is_upper_bound,
        in_engine_scope=entry.in_engine_scope,
        comparable=False,
        kpi_verdict="N/A",
        reason=_USTC_DC_STARK_NOT_COMPARABLE_REASON,
        citation=_USTC_CITATION,
    )


def _bands_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    """Interval-overlap test for two closed bands ``[lo1, hi1]``/``[lo2, hi2]``.

    Two closed intervals on the real line overlap iff each interval's low
    bound is at or below the other interval's high bound. This is the
    precise, documented definition behind every "predicted band
    overlaps/contains the published value" claim this module makes -- no
    Gaussian/symmetric assumption, works for any two asymmetric bands.

    Parameters
    ----------
    lo1, hi1 : float
        First interval's low/high bounds (``lo1 <= hi1``).
    lo2, hi2 : float
        Second interval's low/high bounds (``lo2 <= hi2``).

    Returns
    -------
    bool
        True iff the two closed intervals share at least one point.
    """
    assert lo1 <= hi1, f"malformed interval: lo1={lo1!r} > hi1={hi1!r}"
    assert lo2 <= hi2, f"malformed interval: lo2={lo2!r} > hi2={hi2!r}"
    return lo1 <= hi2 and lo2 <= hi1


def _run_npl_reproducibility_case_impl(*, integration: dict[str, Any]) -> NplReproducibilityCase:
    """Shared implementation behind :func:`run_npl_reproducibility_case`
    (scalar/fast-path, the original WP10 follow-up case) and
    :func:`run_npl_reproducibility_case_rotor` (WP16: the same case
    re-run through the true Cl(1,3) rotor path). See
    :func:`run_npl_reproducibility_case`'s docstring for the method (the
    `integration` parameter is the only thing that differs between the
    two callers: it selects `integration.mode` -- `"auto"` resolves to
    `fast_path` for `ensemble.regime: lattice`, the scalar case's mode;
    `{"mode": "worldline", ...}` selects the rotor case's mode -- both
    valid for `ensemble.regime: lattice` per
    `cliffordclock.pipeline.VALID_INTEGRATION_MODES_BY_REGIME`).

    Parameters
    ----------
    integration : dict[str, Any]
        Forwarded verbatim as the config's ``integration:`` section.

    Returns
    -------
    NplReproducibilityCase
        `kpi_verdict` is `"MET"` if the two bands overlap, else
        `"NOT MET"` -- never `"PASS"`/`"FAIL"`.
    """
    field = loaders.NPL_RESIDUAL_FIELD_V_PER_M
    e_lo = field.combined_lo
    e_hi = field.combined_hi

    def _shift_at(e_field: float) -> float:
        config = PipelineConfig.from_dict(
            {
                "species": "Sr87",
                "trap": {"omega_xyz": list(_TRAP_OMEGA_XYZ)},
                "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, e_field]}}},
                "coupling": {"type": "stark_dc"},
                "ensemble": {
                    "regime": "lattice",
                    "temperature_uK": 1.0,
                    "motional_n": [0, 0, 0],
                    "n_quad": 1,
                },
                "integration": integration,
            }
        )
        return run_pipeline_full(config).report.mean_fractional_shift

    predicted_shift_hi = _shift_at(e_lo)  # smallest |E| -> least-negative shift
    predicted_shift_nominal = _shift_at(field.nominal)
    predicted_shift_lo = _shift_at(e_hi)  # largest |E| -> most-negative shift
    assert predicted_shift_lo <= predicted_shift_nominal <= predicted_shift_hi, (
        "NPL reproducibility case: predicted band is not monotonic in field "
        f"magnitude ({predicted_shift_lo=!r}, {predicted_shift_nominal=!r}, "
        f"{predicted_shift_hi=!r}) -- investigate before trusting this case"
    )

    published = loaders.NPL_PUBLISHED_SHIFT
    overlap = _bands_overlap(predicted_shift_lo, predicted_shift_hi, published.lo, published.hi)

    sr87 = get_species("Sr87")
    return NplReproducibilityCase(
        case_class="reproducibility",
        field_lo_v_per_m=e_lo,
        field_nominal_v_per_m=field.nominal,
        field_hi_v_per_m=e_hi,
        field_citation=field.citation,
        predicted_shift_lo=predicted_shift_lo,
        predicted_shift_nominal=predicted_shift_nominal,
        predicted_shift_hi=predicted_shift_hi,
        published_shift_lo=published.lo,
        published_shift_nominal=published.nominal,
        published_shift_hi=published.hi,
        published_shift_citation=published.citation,
        bands_overlap=overlap,
        kpi_verdict="MET" if overlap else "NOT MET",
        uncertainty_propagation_method=(
            "Field: stat and sys uncertainties combined in quadrature "
            "separately on each side (sqrt(stat**2+sys**2)), never "
            "symmetrized between sides -- see "
            "loaders.AsymmetricMeasurement.combined_lo/combined_hi. "
            "Shift band: three independent real pipeline runs at "
            "(E_lo, E_nominal, E_hi), not an algebraic error-propagation "
            "formula -- the E14b quadratic form's monotonicity in |E| is "
            "asserted, not assumed, by checking predicted_shift_lo <= "
            "predicted_shift_nominal <= predicted_shift_hi at runtime."
        ),
        species_name="Sr87",
        delta_alpha_dc_si=sr87.delta_alpha_dc_si or float("nan"),
        delta_alpha_citation=(
            "T. Middelmann, S. Falke, C. Lisdat, U. Sterr, Phys. Rev. Lett. "
            "109, 263004 (2012) -- cliffordclock.ensemble.species.SR87's "
            "own citation, confirmed to be the same source NPL cites as "
            "arXiv:1706.01944's reference [3] (see "
            "benchmarks/SOURCES.md section 4)."
        ),
    )


def run_npl_reproducibility_case() -> NplReproducibilityCase:
    """Build the NPL reproducibility case (WP10 follow-up, 2026-08-10).

    Method (documented per the follow-up's binding instruction: "no
    Gaussian pretence on asymmetric errors"):

    1. Take NPL's published residual field (`loaders.NPL_RESIDUAL_FIELD_V_PER_M`,
       ``1.52 V/m`` with independent, asymmetric statistical and
       systematic uncertainties).
    2. Combine the statistical and systematic contributions **on each
       side separately** in quadrature (`AsymmetricMeasurement.combined_lo`/
       `combined_hi`) -- this never symmetrizes the low and high sides
       against each other, so the resulting ``(E_lo, E_nominal, E_hi)``
       triple stays asymmetric exactly like the source measurement.
    3. Run the real pipeline (``species: Sr87``, ``coupling.type:
       stark_dc`` resolved from the species registry's Middelmann-sourced
       `Delta_alpha` -- the same citation NPL used -- lattice fast path,
       ``n_quad=1`` uniform field, 1 s interrogation) at each of
       ``E_lo``, ``E_nominal``, ``E_hi`` -- three real, independent
       pipeline calls, not an algebraic shortcut.
    4. Because the E14b shift is `-(k_S)|E|^2/nu0` (monotonically more
       negative with `|E|` for `k_S < 0`, true for Sr87), `E_hi` maps to
       the most-negative predicted shift and `E_lo` to the least-negative
       -- so the predicted band's own asymmetry is *inherited* from the
       field's asymmetry through the pipeline, not assumed.
    5. Compare the resulting ``[predicted_shift_lo, predicted_shift_hi]``
       band against NPL's own published band
       (`loaders.NPL_PUBLISHED_SHIFT`, already given as absolute bounds
       in the paper) via :func:`_bands_overlap`.

    This is the ``integration.mode: fast_path`` (E29 exact quadrature
    expectation) case -- the scalar accumulation path, unchanged since the
    original WP10 follow-up. See :func:`run_npl_reproducibility_case_rotor`
    for the WP16 rotor-path re-run of the identical case.

    Returns
    -------
    NplReproducibilityCase
        `kpi_verdict` is `"MET"` if the two bands overlap, else
        `"NOT MET"` -- never `"PASS"`/`"FAIL"`.
    """
    return _run_npl_reproducibility_case_impl(integration={"time_s": _INTERROGATION_TIME_S})


#: Explicit ``integration.dtau`` (Compton units, E9) for the rotor-path
#: NPL re-run. This case's field is spatially uniform and its single
#: quadrature node (``n_quad=1``) is static (``v=0`` always) -- Omega is
#: therefore *exactly* constant across the entire run (same field, same
#: gradient, same zero velocity, every step), so the rotor result is
#: mathematically independent of step count (`dtau`/`steps`) for any
#: choice at all (the same "static nodes, any step count" exactness E29
#: claims, and `_stark_rotor_ensemble`'s docstring notes). Left on
#: `integration.dtau`'s auto-selection (E31: `dtau <= T_orb/N_res` at this
#: benchmark's `_TRAP_OMEGA_XYZ = 2e5 rad/s`), this case would auto-select
#: ~7.8e6 steps and take minutes; an explicit, much coarser `dtau` is
#: exactly as numerically correct here (verified: identical
#: `predicted_shift_*` to the auto-selected-dtau value at every digit
#: checked) and ~250x faster (`~7.8e3` steps instead). The pre-flight
#: generator-angle guard (`cliffordclock.pipeline.MAX_PER_STEP_ROTOR_ANGLE_RAD`)
#: still applies to this explicit value and passes comfortably at this
#: case's tiny field-driven rate (order 1e-20).
_NPL_ROTOR_DTAU = 1e17


def run_npl_reproducibility_case_rotor() -> NplReproducibilityCase:
    """The NPL reproducibility case (see :func:`run_npl_reproducibility_case`
    for the full method), re-run through the true Cl(1,3) rotor path
    (WP16, CONVENTIONS.md E16-E18 instantiated for the E14b Stark pivot:
    `cliffordclock.integrator.omega.build_omega_stark`, wired into the
    pipeline as `cliffordclock.pipeline._stark_rotor_ensemble` for
    ``integration.mode: worldline``) instead of the scalar fast path.

    Every input (NPL's published residual field, the species-registry
    Middelmann `Delta_alpha`, NPL's published shift band) and every step
    of the method is identical to :func:`run_npl_reproducibility_case` --
    the *only* difference is ``integration: {mode: "worldline", time_s:
    ..., dtau: _NPL_ROTOR_DTAU}`` instead of the fast-path default, which
    is what routes these three pipeline calls through the rotor
    accumulator instead of the E29 scalar quadrature expectation (the
    explicit `dtau` is a performance choice only -- see
    :data:`_NPL_ROTOR_DTAU`'s docstring for why it changes no digit of the
    result). Both `ensemble.regime: lattice` modes are exact for static
    quadrature nodes (E29's own claim, and `_stark_rotor_ensemble`'s
    docstring), so this is expected to -- and, per
    `tests/test_benchmarks_loaders.py`, does -- reproduce the same
    `kpi_verdict` as the scalar case: this function exists to make that
    an *executed*, standing check of WP16's rotor<->scalar equivalence at
    a real, previously-published, three-independent-pipeline-call case,
    not just the synthetic unit tests in
    `tests/test_integrator_stark_rotor.py`.

    Returns
    -------
    NplReproducibilityCase
        `kpi_verdict` is `"MET"` if the two bands overlap, else
        `"NOT MET"` -- never `"PASS"`/`"FAIL"`.
    """
    return _run_npl_reproducibility_case_impl(
        integration={
            "mode": "worldline",
            "time_s": _INTERROGATION_TIME_S,
            "dtau": _NPL_ROTOR_DTAU,
        }
    )


def classify_jila_table1(entries: list[loaders.SystematicShiftEntry]) -> list[JilaCaseVerdict]:
    """Classify every JILA Table I row: in/out of scope, comparable or not.

    Parameters
    ----------
    entries : list[loaders.SystematicShiftEntry]
        Parsed rows from :func:`loaders.load_jila_table1`.

    Returns
    -------
    list[JilaCaseVerdict]
        One verdict per row, in file order. No row is ever marked
        ``comparable=True`` by this function (see module docstring item 2
        for the DC Stark row's specific reasoning) -- this is the WP10
        finding, not a placeholder.
    """
    verdicts = []
    for entry in entries:
        if entry.shift_name == "Total Shift":
            reason = (
                "Sum of all rows above; not a single physical mechanism, so no "
                "per-effect scope/comparison classification applies."
            )
        elif not entry.in_engine_scope:
            reason = entry.scope_note
        else:
            # Only "DC Stark" is in_engine_scope=True in the shipped fixture.
            reason = _JILA_DC_STARK_NOT_COMPARABLE_REASON
        verdicts.append(
            JilaCaseVerdict(
                shift_name=entry.shift_name,
                published_shift_fractional=entry.shift_fractional,
                published_uncertainty_fractional=entry.uncertainty_fractional,
                uncertainty_is_upper_bound=entry.uncertainty_is_upper_bound,
                in_engine_scope=entry.in_engine_scope,
                comparable=False,
                kpi_verdict="N/A",
                reason=reason,
                citation=_JILA_CITATION + f", row {entry.shift_name!r}",
            )
        )
    return verdicts


def run_dc_stark_context_sweep() -> list[DcStarkSweepPoint]:
    """Run the real pipeline (Sr87, ``coupling.type: stark_dc``, lattice
    fast path) at each fixed field in `DC_STARK_CONTEXT_SWEEP_V_PER_M`.

    This exercises the same E14b/E29 machinery `tests/test_known_answers.py`
    KA1 validates, at round-number field magnitudes fixed before any
    comparison exists -- it is explicit physical *context* for where
    JILA's actively-nulled residual (-9.8e-20) sits on the same axis, not
    a benchmark case (no residual against a published value is computed
    here; see :func:`classify_jila_table1`).

    Returns
    -------
    list[DcStarkSweepPoint]
        One point per swept field magnitude, in
        `DC_STARK_CONTEXT_SWEEP_V_PER_M` order.
    """
    points = []
    for field in DC_STARK_CONTEXT_SWEEP_V_PER_M:
        config = PipelineConfig.from_dict(
            {
                "species": "Sr87",
                "trap": {"omega_xyz": list(_TRAP_OMEGA_XYZ)},
                "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, field]}}},
                "coupling": {"type": "stark_dc"},
                "ensemble": {
                    "regime": "lattice",
                    "temperature_uK": 1.0,
                    "motional_n": [0, 0, 0],
                    "n_quad": 1,
                },
                "integration": {"time_s": _INTERROGATION_TIME_S},
            }
        )
        result = run_pipeline_full(config)
        sem = result.report.shift_std_error
        points.append(
            DcStarkSweepPoint(
                field_v_per_m=field,
                predicted_fractional_shift=result.report.mean_fractional_shift,
                predicted_sem=None if sem != sem else sem,  # NaN -> None (single-node ensemble)
            )
        )
    return points


def classify_nist_series(series: loaders.NistPhaseSeries) -> NistCaseVerdict:
    """Classify a parsed NIST M32206 phase series (always not-comparable).

    Parameters
    ----------
    series : loaders.NistPhaseSeries
        A file parsed by :func:`loaders.load_nist_phase_csv`.

    Returns
    -------
    NistCaseVerdict
        Always `comparable=False` -- see `_NIST_NOT_COMPARABLE_REASON`.
    """
    return NistCaseVerdict(
        source_file=series.source_file,
        excerpt_n_samples=int(series.time_s.shape[0]),
        full_dataset_n_samples=_NIST_FULL_DATASET_N_SAMPLES,
        phase_units=series.phase_units,
        comparable=False,
        kpi_verdict="N/A",
        reason=_NIST_NOT_COMPARABLE_REASON,
        citation=_NIST_CITATION,
    )


def build_report(
    *, jila_fixture: Path, nist_yb_fixture: Path, nist_10ghz_fixture: Path
) -> dict[str, Any]:
    """Build the full WP10 benchmark report as a JSON-serializable dict.

    Parameters
    ----------
    jila_fixture : Path
        Path to the JILA Table I CSV (fixture excerpt or, if ever
        expanded, a full transcription -- same format either way).
    nist_yb_fixture : Path
        Path to a NIST Yb-clock-phase CSV (fixture excerpt or full file).
    nist_10ghz_fixture : Path
        Path to a NIST 10 GHz-phase CSV (fixture excerpt or full file).

    Returns
    -------
    dict[str, Any]
        The full report: metadata, JILA case verdicts, the illustrative
        DC-Stark sweep, NIST case verdicts, the NPL reproducibility case
        (plus its WP16 rotor-path re-run, informational only -- see
        :func:`run_npl_reproducibility_case_rotor`), the USTC budget-only
        row, and a KPI summary distinguishing "reproducibility",
        "blind_prediction", and "not_applicable" rows (WP10 follow-up,
        2026-08-10 -- see the module docstring). The rotor re-run does
        **not** add to `kpi_summary`'s row counts (it re-checks the same
        case's verdict via a different accumulator, not a second
        independent case).
    """
    jila_entries = loaders.load_jila_table1(jila_fixture)
    jila_verdicts = classify_jila_table1(jila_entries)
    sweep = run_dc_stark_context_sweep()
    nist_verdicts = [
        classify_nist_series(loaders.load_nist_phase_csv(nist_yb_fixture, phase_units="rad")),
        classify_nist_series(loaders.load_nist_phase_csv(nist_10ghz_fixture, phase_units="mrad")),
    ]
    ustc_verdict = classify_ustc_dc_stark()
    npl_case = run_npl_reproducibility_case()
    npl_case_rotor = run_npl_reproducibility_case_rotor()

    sr87 = get_species("Sr87")
    not_applicable_rows = list(jila_verdicts) + list(nist_verdicts) + [ustc_verdict]
    assert all(not c.comparable for c in not_applicable_rows)  # WP10 finding, pinned

    reproducibility_cases = [npl_case]
    reproducibility_met = sum(1 for c in reproducibility_cases if c.kpi_verdict == "MET")
    blind_prediction_cases: list[Any] = []  # still empty -- see benchmarks/MAPPING.md

    total_rows_considered = (
        len(not_applicable_rows) + len(reproducibility_cases) + len(blind_prediction_cases)
    )

    return {
        "wp10_report_schema": "2.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "kpi_target": "residual between predicted and published experimental shifts < 1e-18",
        "kpi_summary": {
            "total_rows_considered": total_rows_considered,
            "not_applicable_rows": len(not_applicable_rows),
            "reproducibility_cases_total": len(reproducibility_cases),
            "reproducibility_cases_met": reproducibility_met,
            "blind_prediction_cases_total": len(blind_prediction_cases),
            "blind_prediction_cases_met": 0,
            "headline_finding": (
                f"{reproducibility_met} of {len(reproducibility_cases)} "
                "reproducibility case(s) met (NPL arXiv:1706.01944: this "
                "engine's coupling.type=stark_dc pipeline, given NPL's "
                "published residual field and PTB's published "
                "Delta_alpha, reconstructs NPL's own published DC-Stark "
                "shift band; NOT a blind prediction, since NPL combined "
                "the same two ingredients themselves; see "
                "benchmarks/MAPPING.md for why this label is binding). "
                f"{len(blind_prediction_cases)} blind-prediction cases "
                f"(still none available from any authorized source). "
                f"{len(not_applicable_rows)} rows remain not-applicable: "
                "JILA's Table I DC-Stark row and the USTC Metrologia "
                "63,025002 DC-Stark constraint both publish only a "
                "resulting shift/bound with no independent field input; "
                "every other row (BBR, Zeeman, density, lattice light, "
                "background gas, etc., both papers) is physics outside "
                "this engine's scope; the NIST M32206 dataset measures a "
                "different physical quantity entirely. See "
                "benchmarks/RESULTS.md for the full gap analysis."
            ),
        },
        "jila_2403_10664_table1": [asdict(v) for v in jila_verdicts],
        "dc_stark_context_sweep_species_sr87": {
            "note": (
                "Illustrative only -- not a benchmark/residual case. Fixed "
                "field magnitudes (never tuned), run through the real "
                "coupling.type=stark_dc lattice fast-path pipeline."
            ),
            "clock_frequency_hz": sr87.clock_frequency_hz,
            "delta_alpha_dc_si": sr87.delta_alpha_dc_si,
            "points": [asdict(p) for p in sweep],
        },
        "nist_m32206": [asdict(v) for v in nist_verdicts],
        "npl_1706_01944_reproducibility_case": asdict(npl_case),
        "npl_1706_01944_reproducibility_case_rotor_crosscheck": {
            "note": (
                "WP16: the identical reproducibility case re-run through the "
                "true Cl(1,3) rotor path (integration.mode=worldline) instead "
                "of the E29 scalar fast path -- a standing check that the "
                "rotor and scalar coupling.type=stark_dc accumulators agree, "
                "not a second, independently-counted KPI row (excluded from "
                "kpi_summary's totals; see run_npl_reproducibility_case_rotor "
                "and cliffordclock.pipeline._stark_rotor_ensemble)."
            ),
            "case": asdict(npl_case_rotor),
            "bands_overlap_and_verdict_match_scalar_case": (
                npl_case_rotor.bands_overlap == npl_case.bands_overlap
                and npl_case_rotor.kpi_verdict == npl_case.kpi_verdict
            ),
        },
        "ustc_metrologia_63_025002": [asdict(ustc_verdict)],
    }


def render_markdown_table(report: dict[str, Any]) -> str:
    """Render the NPL reproducibility case + JILA/NIST/USTC verdicts as a
    markdown summary.

    Parameters
    ----------
    report : dict[str, Any]
        A report dict as returned by :func:`build_report`.

    Returns
    -------
    str
        A markdown document (reproducibility case + not-applicable-rows
        table + short summary), suitable for embedding or diffing against
        `benchmarks/RESULTS.md`.
    """
    npl = report["npl_1706_01944_reproducibility_case"]
    npl_rotor = report["npl_1706_01944_reproducibility_case_rotor_crosscheck"]
    lines = [
        "# WP10 benchmark summary (generated)",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Reproducibility case: NPL arXiv:1706.01944 (Sr87 DC Stark)",
        "",
        "Zero-free-parameter reproducibility, NOT a blind prediction (see "
        "benchmarks/MAPPING.md): this engine's coupling.type=stark_dc "
        "pipeline, given NPL's published residual field and this "
        "project's Middelmann-sourced species-registry Delta_alpha (the "
        "same source NPL cites), against NPL's own published shift band.",
        "",
        "| Quantity | Low | Nominal | High |",
        "|---|---|---|---|",
        (
            f"| Field (V/m) | {npl['field_lo_v_per_m']:.4f} | "
            f"{npl['field_nominal_v_per_m']:.4f} | {npl['field_hi_v_per_m']:.4f} |"
        ),
        (
            f"| Predicted Δν/ν₀ | {npl['predicted_shift_lo']:+.4e} | "
            f"{npl['predicted_shift_nominal']:+.4e} | {npl['predicted_shift_hi']:+.4e} |"
        ),
        (
            f"| Published Δν/ν₀ (NPL) | {npl['published_shift_lo']:+.4e} | "
            f"{npl['published_shift_nominal']:+.4e} | {npl['published_shift_hi']:+.4e} |"
        ),
        "",
        f"**Bands overlap: {npl['bands_overlap']}; kpi_verdict: {npl['kpi_verdict']}**",
        "",
        (
            "**WP16 rotor-path re-run (informational, not a second KPI row):** "
            f"kpi_verdict={npl_rotor['case']['kpi_verdict']} "
            "(same-verdict-as-scalar-case="
            f"{npl_rotor['bands_overlap_and_verdict_match_scalar_case']}"
            f"); integration.mode=worldline true Cl(1,3) rotor "
            "(`cliffordclock.pipeline._stark_rotor_ensemble`) instead of "
            "the E29 scalar fast path used above."
        ),
        "",
        "| Source | Effect / file | Published value | In scope | Comparable | KPI verdict |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["jila_2403_10664_table1"]:
        published = (
            f"{row['published_shift_fractional']:+.3e} ± "
            f"{row['published_uncertainty_fractional']:.1e}"
        )
        lines.append(
            f"| JILA 2403.10664 Table I | {row['shift_name']} | {published} | "
            f"{row['in_engine_scope']} | {row['comparable']} | {row['kpi_verdict']} |"
        )
    for row in report["ustc_metrologia_63_025002"]:
        published = (
            f"{row['published_shift_fractional']:+.3e} ± "
            f"{row['published_uncertainty_fractional']:.1e}"
        )
        lines.append(
            f"| USTC Metrologia 63,025002 | {row['shift_name']} | {published} | "
            f"{row['in_engine_scope']} | {row['comparable']} | {row['kpi_verdict']} |"
        )
    for row in report["nist_m32206"]:
        lines.append(
            f"| NIST M32206 | {row['source_file']} "
            f"({row['excerpt_n_samples']}/{row['full_dataset_n_samples']} samples, excerpt/full) | "
            f"n/a (phase time series) | False | {row['comparable']} | {row['kpi_verdict']} |"
        )
    summary = report["kpi_summary"]
    lines += [
        "",
        f"**KPI summary:** {summary['reproducibility_cases_met']}/"
        f"{summary['reproducibility_cases_total']} reproducibility case(s) met, "
        f"{summary['blind_prediction_cases_met']}/{summary['blind_prediction_cases_total']} "
        f"blind-prediction case(s) met, {summary['not_applicable_rows']} rows "
        f"not-applicable (of {summary['total_rows_considered']} rows considered).",
        "",
        summary["headline_finding"],
        "",
        "## Illustrative DC-Stark field-magnitude context (Sr87, not a benchmark case)",
        "",
        "| Field (V/m) | Predicted Δν/ν₀ |",
        "|---|---|",
    ]
    for point in report["dc_stark_context_sweep_species_sr87"]["points"]:
        lines.append(
            f"| {point['field_v_per_m']:.1f} | {point['predicted_fractional_shift']:+.6e} |"
        )
    lines.append("")
    lines.append(
        "For reference (not a residual): JILA's actively-nulled published "
        "residual is -9.8e-20, which the table above shows sits between the "
        "1 and 5 V/m rows, i.e. at a scale consistent with a few-V/m "
        "residual field, well below the ~19 V/m unshielded-patch-field "
        "example in `examples/realistic_lattice_sr87.yaml` (WP11)."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the WP10 benchmark and write `benchmarks/results/wp10_results.json`
    and a generated markdown summary alongside it."""
    report = build_report(
        jila_fixture=_FIXTURES_DIR / "jila_2403_10664_table1.csv",
        nist_yb_fixture=_FIXTURES_DIR / "nist_m32206_yb_clock_phase_excerpt.csv",
        nist_10ghz_fixture=_FIXTURES_DIR / "nist_m32206_10ghz_phase_excerpt.csv",
    )
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / "wp10_results.json"
    md_path = _RESULTS_DIR / "wp10_results_table.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_table(report), encoding="utf-8")
    print(render_markdown_table(report))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
