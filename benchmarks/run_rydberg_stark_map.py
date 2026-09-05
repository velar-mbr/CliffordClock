# SPDX-License-Identifier: AGPL-3.0-or-later
"""WP40 Phase B benchmark cases: full Rydberg Stark maps beyond the
quadratic regime (CONVENTIONS.md section 20).

**Cases.**

1. **C3 quadratic-crossover consistency** (:func:`run_c3_crossover_case`):
   for each of the four WP39 registry states (30, 32, 35, 50 nD5/2), the
   map's own low-field curvature (mj-averaged over ``mj=1/2,3/2,5/2`` to
   cancel the tensor term, Section F of the module) is compared against
   Phase A's own registry ``alpha0`` (tabulated for n=30/35/50, derived
   via the registry's own power-law fit for n=32). Kill-tested (sign
   flip, doubled coefficient). Classification: `arithmetic_reproduction`
   (reproducing this project's own already-gated registry number by an
   independent computational path).
2. **C4 ARC cross-validation** (:func:`run_c4_arc_validation_case`):
   the same four states' full field-swept eigenvalue curves against
   ``benchmarks/fixtures/wp40_arc_stark_map_reference.json`` (ARC
   v.3.10.2, commit 4b4573e965222e798ac59636ad7a8b3457262835, this
   project's own tracking code applied to ARC's own Hamiltonian
   matrices; see ``benchmarks/generate_wp40_arc_reference.py``'s own
   docstring for the full provenance and why that isolates the
   Hamiltonian-construction comparison). Two tiers, per the dossier's
   own instruction not to set one flat tolerance across the whole field
   range: a tight, gated low-field tolerance (below half the Inglis-
   Teller estimate, where both this module's own and ARC's tracked
   curves reliably follow the same physical branch), and a reported
   (not strictly gated) full-range comparison plus a crossover-location
   check, since two independently-built near-degenerate Hamiltonians can
   legitimately track through a shared crossing onto swapped branches
   without either being wrong (Section 4 below explains this in detail).
   Classification: `independent_implementation_reproduction` for the
   low-field tier.
3. **C5 published anchor, three-part** (:func:`run_c5_published_anchor_case`):
   (a) low-field reduction to the already-validated Phase A/Holloway
   calibration; (b) O'Sullivan & Stoicheff 1985's printed nS crossing-
   field fit as a same-family method check (computed via this module's
   own nS-basis map, not the registry's nD5/2 state); (c) whether
   Grimmel et al. 2015's supplementary data is machine-readable enough
   for a quantitative comparison (dossier Sec. 2c's own recommended
   first step). Evidentiary classes set per what each part's data
   quality supports (dossier Sec. 2c).
4. **C6 basis-truncation convergence** (:func:`run_c6_convergence_case`):
   :func:`rydberg_stark_map.convergence_sweep` for 50D5/2 (the dossier's
   own flagged load-bearing risk state, crossover ~6.3 V/cm, an order of
   magnitude below ARC's own stated l_max~20 rule-of-thumb's tested
   regime) and, for comparison, 32D5/2. A second sweep
   (:func:`run_c6_crossover_stability_case`) reports 50D5/2's own
   FIRST-CROSSOVER field, the quantity the validity guard actually uses,
   across the same basis-size growth (up to (7,24)) with its own stated
   tolerance.

Run this yourself: ``python benchmarks/run_rydberg_stark_map.py`` (from
the repo root, with ``.venv`` active; no ARC needed for this script
itself, only for regenerating the C4 fixture,
``benchmarks/generate_wp40_arc_reference.py``). Regenerates
``benchmarks/results/wp40_rydberg_stark_map.json`` and its markdown
summary. Runtime: several minutes (dozens of full 451-state Stark-map
diagonalizations across a field grid).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

from cliffordclock.integrator import rydberg_cell_response as rcr
from cliffordclock.integrator import rydberg_stark_map as rsm

_BENCHMARKS_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _BENCHMARKS_DIR / "results"
_FIXTURES_DIR = _BENCHMARKS_DIR / "fixtures"
_ARC_FIXTURE_PATH = _FIXTURES_DIR / "wp40_arc_stark_map_reference.json"

FULL_DELTA_N = 5
FULL_L_MAX = 20
N_FIELD_POINTS = 60
FIELD_RANGE_IT_MULTIPLE = 2.2

# ---------------------------------------------------------------------------
# C3: quadratic-crossover consistency (mj-averaged alpha0 vs. E43 registry)
# ---------------------------------------------------------------------------

#: Registry alpha0 comes from the map's own low-field fit vs. Phase A's
#: already-cross-validated (theory vs. experiment, ~1-5% agreement)
#: registry value; a wider tolerance than that internal agreement is
#: appropriate here since this compares two DIFFERENT computational
#: methods (basis diagonalization vs. tabulated/fitted published
#: numbers), not two sources of the same underlying quantity. Observed
#: worst case across all four states (this session): 4.7% (n=50); 15%
#: leaves a real margin above that, not a vacuous ceiling.
C3_TOLERANCE_RELATIVE = 0.15


@dataclass
class C3Row:
    n: int
    n_star: float
    map_alpha0_au: float
    registry_alpha0_au: float
    registry_source: str
    relative_error: float
    within_tolerance: bool


@dataclass
class C3CrossoverCase:
    case_class: str
    tolerance_relative: float
    rows: list[dict[str, Any]]
    worst_relative_error: float
    kpi_verdict: str
    sign_flip_kill_test_armed: bool
    doubled_coefficient_kill_test_armed: bool


def _registry_alpha0(n: int) -> tuple[float, str]:
    if n == 32:
        return rcr.RB85_32D52_ALPHA0_AU, "derived (power-law fit through n=30/35/50, Phase A)"
    theory = rcr.RB85_ND52_ALPHA0_TABULATED["theory"][n].alpha0_au
    experiment = rcr.RB85_ND52_ALPHA0_TABULATED["experiment"][n].alpha0_au
    return 0.5 * (theory + experiment), "mean of Yerokhin theory/experiment, Phase A Table IV"


def run_c3_crossover_case() -> C3CrossoverCase:
    rows = []
    for n0 in rsm.REGISTRY_N_VALUES:
        n_star = rcr.effective_quantum_number(n0, rcr.RB85_ND52_QUANTUM_DEFECT)
        it_field = rcr.inglis_teller_field_v_per_m(n_star)
        fields = np.linspace(0.0, FIELD_RANGE_IT_MULTIPLE * it_field, N_FIELD_POINTS)
        map_alpha0, _ = rsm.scalar_polarizability_from_map(
            n0,
            fields,
            delta_n=FULL_DELTA_N,
            l_max=FULL_L_MAX,
            max_field_v_per_m=0.15 * it_field,
        )
        registry_alpha0, source = _registry_alpha0(n0)
        rel_err = abs(map_alpha0 - registry_alpha0) / abs(registry_alpha0)
        rows.append(
            C3Row(
                n=n0,
                n_star=n_star,
                map_alpha0_au=map_alpha0,
                registry_alpha0_au=registry_alpha0,
                registry_source=source,
                relative_error=rel_err,
                within_tolerance=rel_err < C3_TOLERANCE_RELATIVE,
            )
        )

    worst = max(r.relative_error for r in rows)

    # Kill tests, formula-level: a sign-flipped or doubled map alpha0 must
    # miss the SAME registry value by far more than the stated tolerance.
    first = rows[0]
    sign_flip_rel = abs(-first.map_alpha0_au - first.registry_alpha0_au) / abs(
        first.registry_alpha0_au
    )
    doubled_rel = abs(2.0 * first.map_alpha0_au - first.registry_alpha0_au) / abs(
        first.registry_alpha0_au
    )

    return C3CrossoverCase(
        case_class="arithmetic_reproduction",
        tolerance_relative=C3_TOLERANCE_RELATIVE,
        rows=[asdict(r) for r in rows],
        worst_relative_error=worst,
        kpi_verdict="MET"
        if worst < C3_TOLERANCE_RELATIVE and all(r.within_tolerance for r in rows)
        else "NOT MET",
        sign_flip_kill_test_armed=sign_flip_rel > C3_TOLERANCE_RELATIVE,
        doubled_coefficient_kill_test_armed=doubled_rel > C3_TOLERANCE_RELATIVE,
    )


# ---------------------------------------------------------------------------
# C4: ARC cross-validation
# ---------------------------------------------------------------------------

#: Low-field tier tolerance: this module's own map vs. ARC's own
#: Hamiltonian (same tracking code both sides), field <= 0.5x the
#: Inglis-Teller estimate. Observed worst case across all four states
#: (this session): 2.05% (n=50); 5% leaves real margin. This is the
#: dossier's own "tighter near zero field" tier.
C4_LOW_FIELD_TOLERANCE_RELATIVE = 0.05

#: Field fraction (of the Inglis-Teller estimate) below which the C4
#: low-field tier applies.
C4_LOW_FIELD_IT_FRACTION = 0.5


@dataclass
class C4Row:
    n: int
    basis_size_arc: int
    basis_size_mine: int
    target_index_arc: int
    target_index_mine: int
    low_field_worst_relative_error: float
    low_field_n_points: int
    my_crossover_field_v_per_m: float | None
    arc_first_low_overlap_field_v_per_m: float | None
    my_min_overlap: float
    arc_min_overlap: float


@dataclass
class C4ArcValidationCase:
    case_class: str
    arc_version: str
    arc_commit: str
    arc_license: str
    tolerance_relative_low_field: float
    low_field_it_fraction: float
    rows: list[dict[str, Any]]
    worst_low_field_relative_error: float
    kpi_verdict: str
    full_range_note: str


def run_c4_arc_validation_case() -> C4ArcValidationCase | None:
    if not _ARC_FIXTURE_PATH.exists():
        return None
    fixture = json.loads(_ARC_FIXTURE_PATH.read_text(encoding="utf-8"))

    rows = []
    for n0_str, state in fixture["states"].items():
        n0 = int(n0_str)
        fields = np.array(state["field_v_per_m"])
        arc_hz = np.array(state["tracked_energy_hz"])
        arc_overlaps = np.array(state["step_overlaps"])
        it_field = state["inglis_teller_field_v_per_m"]

        hamiltonian = rsm.stark_hamiltonian(n0, 2, 2.5, 0.5, delta_n=FULL_DELTA_N, l_max=FULL_L_MAX)
        result = rsm.diagonalize_stark_map(hamiltonian, fields)
        my_hz = result.tracked_energy_hz

        my_shift = my_hz - my_hz[0]
        arc_shift = arc_hz - arc_hz[0]
        low_mask = fields <= C4_LOW_FIELD_IT_FRACTION * it_field
        edge_scale = float(np.max(np.abs(arc_shift[low_mask]))) if np.any(low_mask) else 1.0
        edge_scale = max(edge_scale, 1.0)  # guard against a degenerate all-zero low-field window
        low_field_worst = float(
            np.max(np.abs(my_shift[low_mask] - arc_shift[low_mask])) / edge_scale
        )

        my_cross = rsm.first_crossover_field_v_per_m(result)
        arc_cross_idx = np.argmax(arc_overlaps < 0.9) if np.any(arc_overlaps < 0.9) else None
        arc_cross = float(fields[arc_cross_idx]) if arc_cross_idx is not None else None

        rows.append(
            C4Row(
                n=n0,
                basis_size_arc=state["basis_size"],
                basis_size_mine=hamiltonian.h0.shape[0],
                target_index_arc=state["target_index"],
                target_index_mine=hamiltonian.target_index,
                low_field_worst_relative_error=low_field_worst,
                low_field_n_points=int(np.sum(low_mask)),
                my_crossover_field_v_per_m=my_cross,
                arc_first_low_overlap_field_v_per_m=arc_cross,
                my_min_overlap=result.min_overlap,
                arc_min_overlap=state["min_overlap"],
            )
        )

    worst = max(r.low_field_worst_relative_error for r in rows)
    provenance = fixture["provenance"]
    return C4ArcValidationCase(
        case_class="independent_implementation_reproduction",
        arc_version=provenance["installed_version"],
        arc_commit=provenance["commit"],
        arc_license=provenance["license"],
        tolerance_relative_low_field=C4_LOW_FIELD_TOLERANCE_RELATIVE,
        low_field_it_fraction=C4_LOW_FIELD_IT_FRACTION,
        rows=[asdict(r) for r in rows],
        worst_low_field_relative_error=worst,
        kpi_verdict="MET" if worst < C4_LOW_FIELD_TOLERANCE_RELATIVE else "NOT MET",
        full_range_note=(
            "Beyond the low-field tier, this module's own and ARC's own tracked "
            "curves are built from two INDEPENDENTLY constructed Hamiltonians "
            "(different radial-integral method, different quantum defects for "
            "some l, no fine-structure term in this module's model potential); "
            "near an avoided crossing the two can legitimately settle onto "
            "SWAPPED branches (both individually well-tracked, i.e. locally high "
            "step-overlap, but globally divergent past that point) without "
            "either side's tracking algorithm being wrong; verified directly: "
            "restricting the comparison to points where BOTH curves report a "
            "high step-to-step overlap (>0.95) does not by itself bring the "
            "worst-case full-range error down, confirming this is a branch-"
            "identity effect near a crossing, not a resolution or tracking "
            "bug. This is reported, not gated by a numeric tolerance, per the "
            "dossier's own instruction to loosen (not eliminate) the check near "
            "a crossing; the crossover-location fields above are the closest "
            "thing to a quantitative beyond-crossing check this comparison "
            "supports."
        ),
    )


# ---------------------------------------------------------------------------
# C5: published anchor, three-part
# ---------------------------------------------------------------------------


@dataclass
class C5LowFieldEndpointPart:
    evidentiary_class: str
    citation: str
    rows: list[dict[str, Any]]
    worst_relative_error: float
    tolerance_relative: float
    kpi_verdict: str


@dataclass
class C5OSullivanStoicheffPart:
    evidentiary_class: str
    citation: str
    n_tested: int
    n_star: float
    printed_fit_crossing_field_v_per_m: float
    map_detected_crossing_field_v_per_m: float | None
    relative_error: float | None
    tolerance_relative: float
    kpi_verdict: str


@dataclass
class C5GrimmelSupplementaryDataPart:
    evidentiary_class: str
    citation: str
    supplementary_data_url: str
    fetch_attempted: bool
    fetch_succeeded: bool
    note: str


@dataclass
class C5PublishedAnchorCase:
    low_field_endpoint: dict[str, Any]
    ns_crossing_method_check: dict[str, Any]
    grimmel_supplementary_data: dict[str, Any]
    overall_note: str


def _run_c5_low_field_endpoint() -> C5LowFieldEndpointPart:
    """Holloway et al. 2014 Fig. 15's three printed (splitting, field)
    pairs (already Phase A's own C3 anchor), reproduced here from the
    FULL map's own low-field eigenvalue shift instead of the closed-form
    quadratic formula: the map should reduce to the same answer well
    inside the quadratic window (dossier Sec. 2c, item 1).

    Averaged over ``mj = 1/2, 3/2, 5/2`` (Section F's own scalar-
    polarizability construction), not evaluated at a single ``mj``: a
    single ``mj`` sublevel's own shift includes the ``alpha2`` tensor
    term (Yerokhin et al. 2016 Eq. 7), which for 32D5/2 at ``mj=1/2`` is
    large enough to flip the sign relative to the pure-scalar ``alpha0``
    E43 evaluates -- verified directly this session (the single-``mj``
    shift and the E43 closed form disagreed by both sign and ~5x
    magnitude at every one of Holloway's three fields, a large,
    field-independent ratio that is the signature of comparing two
    genuinely different physical quantities, not a numerical error). The
    mj-average cancels that tensor term exactly (Section F's own
    docstring), leaving the same scalar quantity E43 computes.
    """
    # HOLLOWAY_FIG15_PAIRS lives in the sibling benchmark module (Phase A's own
    # already-cited transcription of Fig. 15), reused here rather than
    # re-transcribed a second time.
    from run_rydberg_cell_response import HOLLOWAY_FIG15_PAIRS

    n_star_32d52 = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
    mj_values = (0.5, 1.5, 2.5)
    hamiltonians = {
        mj: rsm.stark_hamiltonian(32, 2, 2.5, mj, delta_n=FULL_DELTA_N, l_max=FULL_L_MAX)
        for mj in mj_values
    }
    rows = []
    for _delta_f_hz, field_v_per_m in HOLLOWAY_FIG15_PAIRS:
        shifts_hz = []
        for mj in mj_values:
            result = rsm.diagonalize_stark_map(hamiltonians[mj], np.array([0.0, field_v_per_m]))
            shifts_hz.append(result.tracked_energy_hz[1] - result.tracked_energy_hz[0])
        map_shift_hz = float(np.mean(shifts_hz))
        # This mj-averaged map shift is the RYDBERG-LEVEL (32D5/2) SCALAR
        # quadratic shift alone, the same quantity rydberg_quadratic_stark_shift_hz
        # computes; compare against that closed-form E43 evaluation at the
        # same field (not against Holloway's own Delta_f, which also carries
        # the RF Rabi/Doppler-mismatch machinery E44 adds on top).
        closed_form_shift_hz = rcr.rydberg_quadratic_stark_shift_hz(
            rcr.RB85_32D52_ALPHA0_AU, field_v_per_m, n_star_32d52
        )
        rel_err = abs(map_shift_hz - closed_form_shift_hz) / abs(closed_form_shift_hz)
        rows.append(
            {
                "field_v_per_m": field_v_per_m,
                "map_shift_hz": map_shift_hz,
                "closed_form_e43_shift_hz": closed_form_shift_hz,
                "relative_error": rel_err,
            }
        )
    worst = max(r["relative_error"] for r in rows)
    tolerance = 0.10
    return C5LowFieldEndpointPart(
        evidentiary_class="arithmetic_reproduction",
        citation=(
            "Holloway et al., IEEE TAP 62, 6169 (2014) [arXiv:1405.7066], Fig. 15 "
            "field values, reused from the Phase A/C3 anchor; the map is checked "
            "against Phase A's own closed-form E43 evaluation at the same fields, "
            "not against Holloway's own Delta_f (a different, E44-level quantity)"
        ),
        rows=rows,
        worst_relative_error=worst,
        tolerance_relative=tolerance,
        kpi_verdict="MET" if worst < tolerance else "NOT MET",
    )


def _run_c5_ns_crossing_method_check() -> C5OSullivanStoicheffPart:
    """O'Sullivan & Stoicheff 1985's printed Rb-85 nS crossing-field fit,
    ``E_crossing(V/cm) = 4.638e8/n*^5 + 1.528e10/n*^7``, checked against
    THIS module's own map, built for the nS1/2 series (not the registry's
    nD5/2), at a convenient n (dossier Sec. 2c, item 2: "the method,"
    same Hamiltonian machinery, different l).
    """
    n_test = 40
    defect_s = rsm.quantum_defect_for(0, 0.5)
    n_star = rcr.effective_quantum_number(n_test, defect_s)
    printed_field_v_per_cm = 4.638e8 / n_star**5 + 1.528e10 / n_star**7
    printed_field_v_per_m = printed_field_v_per_cm * 100.0

    field_grid = np.linspace(0.0, 2.0 * printed_field_v_per_m, 80)
    result = rsm.stark_map_registry_state(
        n_test, field_grid, l0=0, j0=0.5, mj=0.5, delta_n=FULL_DELTA_N, l_max=FULL_L_MAX
    )
    detected = rsm.first_crossover_field_v_per_m(result)
    rel_err = (
        abs(detected - printed_field_v_per_m) / printed_field_v_per_m
        if detected is not None
        else None
    )
    tolerance = 0.25
    return C5OSullivanStoicheffPart(
        evidentiary_class="arithmetic_reproduction",
        citation="O'Sullivan & Stoicheff, Phys. Rev. A 31, 2718 (1985), printed nS crossing fit",
        n_tested=n_test,
        n_star=n_star,
        printed_fit_crossing_field_v_per_m=printed_field_v_per_m,
        map_detected_crossing_field_v_per_m=detected,
        relative_error=rel_err,
        tolerance_relative=tolerance,
        kpi_verdict=(
            "MET" if (rel_err is not None and rel_err < tolerance) else "NOT MET (see note)"
        ),
    )


def _run_c5_grimmel_supplementary_data() -> C5GrimmelSupplementaryDataPart:
    """dossier Sec. 2c's own recommended first step: try the printed
    supplementary-data URL before digitizing Figs. 4-6 by hand.

    This is a live network fetch (`urlopen`, 10s timeout), the only one
    in this benchmark: publisher content, layout, and availability can
    all change after this session, so a future run of this exact script
    can legitimately report a different HTTP status or content type for
    the SAME URL than the value committed in this file's own results.
    The `except` branches below degrade gracefully either way: no
    quantitative Grimmel comparison is included, reported as such, and
    nothing here raises, so a network failure never blocks the rest of
    C5 or the benchmark run.
    """
    url = "https://stacks.iop.org/njp/17/053005/mmedia"
    fetch_attempted = True
    fetch_succeeded = False
    note = ""
    try:
        with urlopen(url, timeout=10) as response:  # noqa: S310
            fetch_succeeded = 200 <= response.status < 300
            note = f"HTTP {response.status}; content-type={response.headers.get('Content-Type')}"
    except URLError as exc:
        note = f"fetch failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        note = f"fetch failed: {exc}"

    if not fetch_succeeded:
        note += (
            ". No quantitative Grimmel et al. 2015 comparison is included in this "
            "benchmark: their own Figs. 4-6 (the actual Stark-map data) are "
            "grayscale spectrogram images, and their supplementary-data link did "
            "not yield a fetchable machine-readable file this session. Per this "
            "project's standing digitization caution, no digitized-plot "
            "comparison is substituted silently; the 85Rb/87Rb isotope mismatch "
            "(dossier risk 3) would also need correcting for any quantitative use."
        )

    return C5GrimmelSupplementaryDataPart(
        evidentiary_class="not_attempted (no machine-readable data found)",
        citation="Grimmel et al., New J. Phys. 17, 053005 (2015) [arXiv:1503.08953]",
        supplementary_data_url=url,
        fetch_attempted=fetch_attempted,
        fetch_succeeded=fetch_succeeded,
        note=note,
    )


def run_c5_published_anchor_case() -> C5PublishedAnchorCase:
    return C5PublishedAnchorCase(
        low_field_endpoint=asdict(_run_c5_low_field_endpoint()),
        ns_crossing_method_check=asdict(_run_c5_ns_crossing_method_check()),
        grimmel_supplementary_data=asdict(_run_c5_grimmel_supplementary_data()),
        overall_note=(
            "No single source combines the registry species (85Rb), the "
            "registry l (D5/2), and coverage through an avoided crossing with "
            "printed (non-digitized) numbers (dossier Sec. 2c/6, item 2); the "
            "three parts above are the fullest composite this project can "
            "support, each labeled with its own evidentiary class, none "
            "presented as a full through-the-crossing validation on its own."
        ),
    )


# ---------------------------------------------------------------------------
# C6: basis-truncation convergence
# ---------------------------------------------------------------------------


@dataclass
class C6StateResult:
    n: int
    it_field_v_per_m: float
    rows: list[dict[str, Any]]
    converged: bool


@dataclass
class C6CrossoverStabilityRow:
    delta_n: int
    l_max: int
    basis_size: int
    crossover_field_v_per_m: float | None
    relative_shift_from_largest: float | None


@dataclass
class C6CrossoverStabilityResult:
    n: int
    it_field_v_per_m: float
    tolerance_relative: float
    rows: list[dict[str, Any]]
    stable: bool


@dataclass
class C6ConvergenceCase:
    case_class: str
    convergence_threshold_relative: float
    states: list[dict[str, Any]]
    crossover_stability: dict[str, Any]
    kpi_verdict: str


#: Convergence criterion: the largest basis's own nearest smaller-basis
#: neighbor must be within this relative shift (a non-vacuous, stated
#: bound, not merely "the sweep ran").
C6_CONVERGENCE_THRESHOLD_RELATIVE = 0.10

#: Same criterion, applied to the FIRST-CROSSOVER field itself, the
#: quantity `stark_validity_field_v_per_m` actually guards (the low-field
#: curvature `convergence_sweep` checks above is a different quantity).
C6_CROSSOVER_STABILITY_THRESHOLD_RELATIVE = 0.05

#: `convergence_sweep`'s own default basis-size sequence, used here in
#: full (including its largest (7,24) member, which the C6 states loop
#: above omits since it is not needed for low-field curvature
#: convergence at this state).
C6_CROSSOVER_STABILITY_BASIS_SIZES: list[tuple[int, int]] = [
    (2, 6),
    (3, 10),
    (5, 14),
    (5, 20),
    (7, 24),
]


def run_c6_crossover_stability_case(n0: int = 50) -> C6CrossoverStabilityResult:
    """Sweep `(delta_n, l_max)` basis sizes and report the FIRST-CROSSOVER
    field itself for 50D5/2 (the dossier's own flagged load-bearing risk
    state), the quantity `stark_validity_field_v_per_m` actually guards
    and a quantity `convergence_sweep`'s own low-field-curvature check
    above does not directly test. Uses the same field grid convention as
    `stark_validity_field_v_per_m`'s own default (0 to 2.2x the
    Inglis-Teller estimate, 60 points).
    """
    n_star = rcr.effective_quantum_number(n0, rcr.RB85_ND52_QUANTUM_DEFECT)
    it_field = rcr.inglis_teller_field_v_per_m(n_star)
    fields = np.linspace(0.0, 2.2 * it_field, 60)

    basis_sizes: list[int] = []
    crossovers: list[float | None] = []
    for delta_n, l_max in C6_CROSSOVER_STABILITY_BASIS_SIZES:
        hamiltonian = rsm.stark_hamiltonian(n0, 2, 2.5, 0.5, delta_n=delta_n, l_max=l_max)
        result = rsm.diagonalize_stark_map(hamiltonian, fields)
        basis_sizes.append(hamiltonian.h0.shape[0])
        crossovers.append(rsm.first_crossover_field_v_per_m(result))

    reference = crossovers[-1]
    rows = [
        C6CrossoverStabilityRow(
            delta_n=delta_n,
            l_max=l_max,
            basis_size=basis_size,
            crossover_field_v_per_m=crossover,
            relative_shift_from_largest=(
                None
                if crossover is None or reference is None
                else abs(crossover - reference) / abs(reference)
            ),
        )
        for (delta_n, l_max), basis_size, crossover in zip(
            C6_CROSSOVER_STABILITY_BASIS_SIZES, basis_sizes, crossovers, strict=True
        )
    ]

    # Stable if the SECOND-largest basis already agrees with the largest
    # to within tolerance, the same criterion the C6 states loop above
    # applies to the low-field curvature.
    second_largest_shift = rows[-2].relative_shift_from_largest
    stable = second_largest_shift is not None and second_largest_shift < (
        C6_CROSSOVER_STABILITY_THRESHOLD_RELATIVE
    )

    return C6CrossoverStabilityResult(
        n=n0,
        it_field_v_per_m=it_field,
        tolerance_relative=C6_CROSSOVER_STABILITY_THRESHOLD_RELATIVE,
        rows=[asdict(r) for r in rows],
        stable=stable,
    )


def run_c6_convergence_case() -> C6ConvergenceCase:
    states_out = []
    for n0 in (50, 32):
        n_star = rcr.effective_quantum_number(n0, rcr.RB85_ND52_QUANTUM_DEFECT)
        it_field = rcr.inglis_teller_field_v_per_m(n_star)
        fields = np.linspace(0.0, 1.5 * it_field, 30)
        rows = rsm.convergence_sweep(
            n0,
            fields,
            basis_sizes=[(2, 6), (3, 10), (5, 14), (5, 20)],
            max_field_v_per_m=0.15 * it_field,
        )
        # converged if the SECOND-largest basis is already within the threshold
        # of the largest (not just the largest vs. itself, which is trivial).
        converged = (
            len(rows) >= 2
            and rows[-2].relative_shift_from_largest < C6_CONVERGENCE_THRESHOLD_RELATIVE
        )
        states_out.append(
            C6StateResult(
                n=n0,
                it_field_v_per_m=it_field,
                rows=[asdict(r) for r in rows],
                converged=converged,
            )
        )

    all_converged = all(s.converged for s in states_out)
    crossover_stability = run_c6_crossover_stability_case()
    return C6ConvergenceCase(
        case_class="convergence_study",
        convergence_threshold_relative=C6_CONVERGENCE_THRESHOLD_RELATIVE,
        states=[asdict(s) for s in states_out],
        crossover_stability=asdict(crossover_stability),
        kpi_verdict="MET" if (all_converged and crossover_stability.stable) else "NOT MET",
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report() -> dict[str, Any]:
    c3 = run_c3_crossover_case()
    c4 = run_c4_arc_validation_case()
    c5 = run_c5_published_anchor_case()
    c6 = run_c6_convergence_case()
    return {
        "wp40_rydberg_stark_map_benchmark_schema": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "c3_quadratic_crossover_consistency": asdict(c3),
        "c4_arc_cross_validation": asdict(c4) if c4 is not None else None,
        "c5_published_anchor": asdict(c5),
        "c6_basis_truncation_convergence": asdict(c6),
    }


def _render_ns_crossing_line(part: dict[str, Any]) -> str:
    detected = part["map_detected_crossing_field_v_per_m"]
    detected_str = "n/a" if detected is None else f"{detected / 100:.2f} V/cm"
    printed = part["printed_fit_crossing_field_v_per_m"] / 100
    return (
        f"Classification: {part['evidentiary_class']}. "
        f"n={part['n_tested']}, printed field={printed:.2f} V/cm, "
        f"map-detected field={detected_str}"
    )


def render_markdown(report: dict[str, Any]) -> str:
    c3 = report["c3_quadratic_crossover_consistency"]
    c4 = report["c4_arc_cross_validation"]
    c5 = report["c5_published_anchor"]
    c6 = report["c6_basis_truncation_convergence"]

    lines = [
        "# WP40 Rydberg Stark-map benchmark cases (generated)",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "Validates the full Stark-map module (CONVENTIONS.md section 20) "
        "against Phase A's own registry (C3), an independent open-source "
        "implementation (C4, ARC), published literature (C5), and its own "
        "basis-truncation convergence (C6).",
        "",
        "## C3: quadratic-crossover consistency vs. the E43 registry",
        "",
        f"**Classification: {c3['case_class']}**",
        "",
        "| n | n* | map alpha0 (a.u.) | registry alpha0 (a.u.) | Relative error |",
        "|---|---|---|---|---|",
        *(
            f"| {r['n']} | {r['n_star']:.3f} | {r['map_alpha0_au']:.4e} | "
            f"{r['registry_alpha0_au']:.4e} | {r['relative_error']:.2%} |"
            for r in c3["rows"]
        ),
        "",
        f"Tolerance: {c3['tolerance_relative']:.0%}. Worst relative error: "
        f"{c3['worst_relative_error']:.2%}. **kpi_verdict: {c3['kpi_verdict']}**",
        "",
        (
            f"Kill tests armed: sign-flip={c3['sign_flip_kill_test_armed']}, "
            f"doubled-coefficient={c3['doubled_coefficient_kill_test_armed']}."
        ),
        "",
    ]

    if c4 is None:
        lines += [
            "## C4: ARC cross-validation",
            "",
            "SKIPPED: `benchmarks/fixtures/wp40_arc_stark_map_reference.json` not found. "
            "Run `python benchmarks/generate_wp40_arc_reference.py` (ARC installed) first.",
            "",
        ]
    else:

        def _vcm(field_v_per_m: float | None) -> str:
            return "n/a" if field_v_per_m is None else f"{field_v_per_m / 100:.2f}"

        lines += [
            "## C4: ARC cross-validation",
            "",
            f"**Classification: {c4['case_class']}**. ARC {c4['arc_version']}, "
            f"commit {c4['arc_commit']}, {c4['arc_license']}.",
            "",
            (
                "| n | Basis (mine/ARC) | Target idx (mine/ARC) | Low-field worst rel. err | "
                "My crossover (V/cm) | ARC low-overlap field (V/cm) |"
            ),
            "|---|---|---|---|---|---|",
            *(
                f"| {r['n']} | {r['basis_size_mine']}/{r['basis_size_arc']} | "
                f"{r['target_index_mine']}/{r['target_index_arc']} | "
                f"{r['low_field_worst_relative_error']:.3%} | "
                f"{_vcm(r['my_crossover_field_v_per_m'])} | "
                f"{_vcm(r['arc_first_low_overlap_field_v_per_m'])} |"
                for r in c4["rows"]
            ),
            "",
            f"Low-field tolerance ({c4['low_field_it_fraction']:.0%} of the Inglis-Teller "
            f"estimate): {c4['tolerance_relative_low_field']:.0%}. Worst observed: "
            f"{c4['worst_low_field_relative_error']:.3%}. **kpi_verdict: {c4['kpi_verdict']}**",
            "",
            c4["full_range_note"],
            "",
        ]

    lines += [
        "## C5: published anchor, three-part",
        "",
        c5["overall_note"],
        "",
        "### (a) Holloway et al. 2014 Fig. 15 field-endpoint reduction",
        "",
        f"Classification: {c5['low_field_endpoint']['evidentiary_class']}. "
        f"Worst relative error: {c5['low_field_endpoint']['worst_relative_error']:.3%}. "
        f"**kpi_verdict: {c5['low_field_endpoint']['kpi_verdict']}**",
        "",
        "### (b) O'Sullivan & Stoicheff 1985 nS crossing-field method check",
        "",
        _render_ns_crossing_line(c5["ns_crossing_method_check"]),
        f"**kpi_verdict: {c5['ns_crossing_method_check']['kpi_verdict']}**",
        "",
        "### (c) Grimmel et al. 2015 supplementary data availability",
        "",
        f"Classification: {c5['grimmel_supplementary_data']['evidentiary_class']}.",
        c5["grimmel_supplementary_data"]["note"],
        "",
        "## C6: basis-truncation convergence",
        "",
        f"**Classification: {c6['case_class']}**. Convergence threshold: "
        f"{c6['convergence_threshold_relative']:.0%}. **kpi_verdict: {c6['kpi_verdict']}**",
        "",
    ]
    for state in c6["states"]:
        lines.append(
            f"### n={state['n']} (IT estimate: {state['it_field_v_per_m'] / 100:.2f} V/cm, "
            f"converged={state['converged']})"
        )
        lines.append("")
        lines.append(
            "| delta_n | l_max | Basis size | alpha0 (a.u.) | Relative shift from largest |"
        )
        lines.append("|---|---|---|---|---|")
        for row in state["rows"]:
            lines.append(
                f"| {row['delta_n']} | {row['l_max']} | {row['basis_size']} | "
                f"{row['alpha0_au']:.4e} | {row['relative_shift_from_largest']:.3%} |"
            )
        lines.append("")

    cs = c6["crossover_stability"]
    lines.append(
        f"### Crossover-field stability across basis size (n={cs['n']}, IT estimate: "
        f"{cs['it_field_v_per_m'] / 100:.2f} V/cm, stable={cs['stable']})"
    )
    lines.append("")
    lines.append(
        "Reports the FIRST-CROSSOVER field itself, the quantity "
        "`stark_validity_field_v_per_m` guards. The table above checks a "
        "different quantity, the low-field curvature."
    )
    lines.append("")
    lines.append(
        "| delta_n | l_max | Basis size | First-crossover field (V/cm) | "
        "Relative shift from largest |"
    )
    lines.append("|---|---|---|---|---|")
    for row in cs["rows"]:
        crossover = row["crossover_field_v_per_m"]
        crossover_str = "n/a" if crossover is None else f"{crossover / 100:.4f}"
        shift = row["relative_shift_from_largest"]
        shift_str = "n/a" if shift is None else f"{shift:.3%}"
        lines.append(
            f"| {row['delta_n']} | {row['l_max']} | {row['basis_size']} | "
            f"{crossover_str} | {shift_str} |"
        )
    lines.append("")
    lines.append(f"Tolerance: {cs['tolerance_relative']:.0%}. **stable: {cs['stable']}**")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / "wp40_rydberg_stark_map.json"
    md_path = _RESULTS_DIR / "wp40_rydberg_stark_map.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
