# SPDX-License-Identifier: AGPL-3.0-or-later
"""WP36 Phase 1 benchmark cases: lattice light shift, both community models
against their defining papers (CONVENTIONS.md section 17, E40/E41).

Mirrors `run_motional_al_ion.py`'s structure: dataclasses per case, real
engine calls only (no hand arithmetic standing in for the engine), a JSON
report plus a markdown summary written to `benchmarks/results/`.

**Four cases.**

1. **Target 1** (:func:`run_ushijima_operational_point_case`): Ushijima et
   al. 2018's own operational point, `u_op = 72(2) E_R` at
   `delta_L_op = 5.3(2) MHz`, recomputed by solving their Eq. 1 directly
   (`lattice_light_shift.solve_harmonic_operational_point`). Classification:
   `arithmetic_reproduction` (a closed-form solve against the paper's own
   published formula and coefficients, zero fitted parameters).

2. **Target 2** (:func:`run_aeppli_lattice_line_item_case`): Aeppli et al.
   2024 Table I's lattice-light-shift budget row, `-0.1(3.2)e-19` at
   `15.06(17) E_R`, `10.5 MHz` detuning, `Tr ~= 120 nK` (Kim et al. 2023's
   own operating-point radial temperature, the same paper Aeppli et al.
   2024 states it reuses "identical atomic coefficients" from).
   Classification: `arithmetic_reproduction`.

3. **Target 3** (two parts, both needed for a truthful account):

   a. :func:`run_bothwell_table1_reproduction_case`: Bothwell et al.
      2025's own Appendix A Table I, the harmonic-vs-BO+WKB `X`/`Y`/`Z`
      trap-depth-reduction-factor comparison at four published
      `(u0, Tr)` points for Yb-171. This module's BO+WKB machinery
      (:func:`lattice_light_shift.axial_thermal_factors`) reproduces the
      BO+WKB column at all four points to the paper's own stated
      precision: the strongest direct numerical validation of this
      module's Model B implementation available from any paper in this
      work package. Classification: `arithmetic_reproduction`.
   b. :func:`run_bothwell_headline_comparison_case`: the paper's headline
      Table III number, `alpha~M1E2/h = -1.41(9)e-18` (harmonic) vs.
      `-1.45(8)e-18` (BO+WKB). These are OUTPUTS of Bothwell et al.'s own
      nonlinear fit against their raw (unpublished) scan data, not
      arithmetic functions of any published input. Reproducing them
      would mean re-running their fit, which this module's inputs cannot
      do. What IS computable from published inputs: both models,
      evaluated at Bothwell's own stated operating conditions
      (`u0 < 140 E_R`, `Tr ~= 600 nK`, `n_z = 0`) using EACH model's own
      published coefficient column, quantifying the resulting fractional
      shift difference. Classification: `computable_comparison` (a
      distinct label from `arithmetic_reproduction`: the coefficients
      themselves are not independently rederived here, only the
      light-shift EVALUATION at stated conditions is).

4. **Density-of-states contrast** (:func:`run_density_of_states_contrast_case`):
   both models' axial-band degeneracy (Beloy et al. 2020 Eqs. 4/11) at
   matched conditions, and how the two curves diverge as radial
   temperature rises: the dossier's "harmonic grows linearly, BO+WKB
   grows near-exponentially" claim, computed here directly from the
   papers' own equations.

Run this yourself: ``python benchmarks/run_lattice_light_shift.py`` (from
the repo root, with ``.venv`` active). Regenerates
``benchmarks/results/wp36_lattice_light_shift.json`` and the accompanying
``.md`` summary.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import run_benchmarks  # noqa: E402 (reuses the already-tested `_bands_overlap`)

import cliffordclock.integrator.lattice_light_shift as lls  # noqa: E402
from cliffordclock.constants import SPEED_OF_LIGHT  # noqa: E402
from cliffordclock.ensemble.species import get_species  # noqa: E402

_RESULTS_DIR = _BENCHMARKS_DIR / "results"

#: The Sr-87 magic-lattice wavelength implied by Kim et al. 2023's own
#: measured E1 magic frequency, `nu_813 = 368554825.9(4) MHz` (their
#: Table, transcribed directly from the typeset PDF, arXiv:2210.16374).
#: Used for Target 2's `E_R` (this project's `Sr87` registry mass, never
#: hand-typed).
KIM_2023_MAGIC_FREQUENCY_HZ = 368_554_825.9e6

#: Yb-171's magic-lattice wavelength implied by Bothwell et al. 2025's own
#: measured E1 magic frequency, harmonic-basis column,
#: `nu_E1 = 394798266.9(26) MHz` (their Table III, transcribed directly
#: from the typeset PDF, arXiv:2409.10782). Used for Target 3's `E_R`.
BOTHWELL_2025_MAGIC_FREQUENCY_HZ = 394_798_266.9e6

#: This project's convention for what an "arithmetic reproduction" label
#: means, restated here (mirrors `run_bbr_jila_arithmetic_reproduction.CASE_LABEL`'s
#: role) for Targets 1, 2, and 3a.
ARITHMETIC_REPRODUCTION_LABEL = (
    "arithmetic reproduction of a published standard-formula evaluation "
    "(arithmetic-reproduction fidelity: the engine evaluates the paper's "
    "own published equation and coefficients with zero fitted parameters "
    "and is compared against that same paper's own published result)"
)

#: The distinct label for Target 3b: the coefficients being compared are
#: fit OUTPUTS from the source paper's own raw data, not independently
#: reproduced here; only each model's light-shift EVALUATION at the
#: paper's own stated operating conditions is computed.
COMPUTABLE_COMPARISON_LABEL = (
    "computable comparison, NOT an arithmetic reproduction of the paper's "
    "own fitted coefficient values: alpha~M1E2 in Bothwell et al. 2025's "
    "Table III is an output of their own nonlinear fit against raw "
    "(unpublished) scan data; reproducing it would require running that "
    "same fit against their raw data, which this module does not have. "
    "What is computed here is each model's own light-shift prediction at "
    "the paper's stated operating conditions, using that same paper's own "
    "published coefficient columns, and the resulting fractional-shift "
    "difference between the two models: the part of Target 3 that "
    "published inputs alone can settle."
)


# ---------------------------------------------------------------------------
# Target 1: Ushijima et al. 2018 operational point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UshijimaOperationalPointCase:
    """Target 1: solving Ushijima et al. 2018's own Eq. 1 for the
    operational point, compared against their published `u_op`/`delta_L_op`.
    """

    case_class: str
    case_label: str
    species_name: str
    u_op_predicted: float
    detuning_hz_op_predicted: float
    u_op_published: float
    u_op_published_uncertainty: float
    detuning_hz_op_published: float
    detuning_hz_op_published_uncertainty: float
    u_op_within_published_uncertainty: bool
    detuning_op_within_published_uncertainty: bool
    residual_shift_hz: float
    residual_slope_hz: float
    citation: str


def run_ushijima_operational_point_case() -> UshijimaOperationalPointCase:
    """Solve Ushijima et al. 2018's operational-point conditions from their
    own Eq. 1 and Table I coefficients
    (:func:`lattice_light_shift.solve_harmonic_operational_point`), and
    compare against their published `u_op = 72(2)`, `delta_L_op = 5.3(2) MHz`.
    """
    result = lls.solve_harmonic_operational_point(lls.USHIJIMA_2018_SR87, n_z=0.0)
    u_op_pub, u_op_unc = 72.0, 2.0
    detuning_pub_hz, detuning_unc_hz = 5.3e6, 0.2e6
    u_within = abs(result.u_op - u_op_pub) <= u_op_unc
    detuning_within = abs(result.detuning_hz_op - detuning_pub_hz) <= detuning_unc_hz
    return UshijimaOperationalPointCase(
        case_class="arithmetic_reproduction",
        case_label=ARITHMETIC_REPRODUCTION_LABEL,
        species_name="Sr87",
        u_op_predicted=result.u_op,
        detuning_hz_op_predicted=result.detuning_hz_op,
        u_op_published=u_op_pub,
        u_op_published_uncertainty=u_op_unc,
        detuning_hz_op_published=detuning_pub_hz,
        detuning_hz_op_published_uncertainty=detuning_unc_hz,
        u_op_within_published_uncertainty=u_within,
        detuning_op_within_published_uncertainty=detuning_within,
        residual_shift_hz=result.residual_shift_hz,
        residual_slope_hz=result.residual_slope_hz,
        citation="Ushijima, Takamoto, Katori, PRL 121, 263202 (2018), main text and Eqs. 14-15",
    )


# ---------------------------------------------------------------------------
# Target 2: Aeppli et al. 2024 lattice-light-shift budget line
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AeppliLatticeLineItemCase:
    """Target 2: Aeppli et al. 2024 Table I's `Lattice Light` row,
    recomputed from Kim et al. 2023's coefficients ("identical atomic
    coefficients as in Ref. [19]", Aeppli et al.'s own words) via the
    JILA-lineage reciprocal reduction factor.
    """

    case_class: str
    case_label: str
    species_name: str
    u0: float
    detuning_hz: float
    radial_temperature_k: float
    predicted_shift_fractional: float
    predicted_uncertainty_fractional: float
    predicted_band_lo: float
    predicted_band_hi: float
    published_shift_fractional: float
    published_uncertainty_fractional: float
    published_band_lo: float
    published_band_hi: float
    bands_overlap: bool
    kpi_verdict: str
    citation: str


def run_aeppli_lattice_line_item_case() -> AeppliLatticeLineItemCase:
    """Recompute Aeppli et al. 2024 Table I's `Lattice Light` row
    (`-0.1(3.2)e-19` at `15.06(17) E_R`, `10.5 MHz` detuning) from Kim et
    al. 2023's own coefficients (`KIM_2023_SR87`) and the JILA reciprocal
    reduction factor (:func:`lattice_light_shift.jila_reduction_factor`),
    with `Tr ~= 120 nK` (Kim et al. 2023's own stated radial temperature at
    their `15 E_R` operating point).
    """
    sr = get_species("Sr87")
    wavelength_m = SPEED_OF_LIGHT / KIM_2023_MAGIC_FREQUENCY_HZ
    e_r = lls.recoil_energy_j(wavelength_m, sr.mass_kg)
    u0 = 15.06
    detuning_hz = 10.5e6
    tr_k = 120e-9

    shift_hz = lls.harmonic_light_shift_hz(
        u0,
        detuning_hz,
        0.0,
        lls.KIM_2023_SR87,
        reduction_form="jila_reciprocal",
        radial_temperature_k=tr_k,
        recoil_energy_j_value=e_r,
    )
    unc_hz = lls.harmonic_light_shift_uncertainty_hz(
        u0,
        detuning_hz,
        0.0,
        lls.KIM_2023_SR87,
        reduction_form="jila_reciprocal",
        radial_temperature_k=tr_k,
        recoil_energy_j_value=e_r,
    )
    predicted = shift_hz / sr.clock_frequency_hz
    predicted_unc = unc_hz / sr.clock_frequency_hz
    published, published_unc = -0.1e-19, 3.2e-19

    bands_overlap = run_benchmarks._bands_overlap(
        predicted - predicted_unc,
        predicted + predicted_unc,
        published - published_unc,
        published + published_unc,
    )
    return AeppliLatticeLineItemCase(
        case_class="arithmetic_reproduction",
        case_label=ARITHMETIC_REPRODUCTION_LABEL,
        species_name="Sr87",
        u0=u0,
        detuning_hz=detuning_hz,
        radial_temperature_k=tr_k,
        predicted_shift_fractional=predicted,
        predicted_uncertainty_fractional=predicted_unc,
        predicted_band_lo=predicted - predicted_unc,
        predicted_band_hi=predicted + predicted_unc,
        published_shift_fractional=published,
        published_uncertainty_fractional=published_unc,
        published_band_lo=published - published_unc,
        published_band_hi=published + published_unc,
        bands_overlap=bands_overlap,
        kpi_verdict="MET" if bands_overlap else "NOT MET",
        citation=(
            "Aeppli, Kim, Warfield, Safronova, Ye, PRL 133, 023401 (2024), Table I; "
            "coefficients and Tr from Kim, Aeppli, Bothwell, Ye, PRL 130, 113203 (2023)"
        ),
    )


# ---------------------------------------------------------------------------
# Target 3a: Bothwell et al. 2025 Table I (X/Y/Z reduction factors)
# ---------------------------------------------------------------------------


#: Bothwell et al. 2025's Appendix A Table I, transcribed directly from
#: the typeset PDF (arXiv:2409.10782): four `(u0, Tr_nK)` points, each
#: with the paper's own harmonic-basis and BO+WKB `X`/`Y`/`Z` values for
#: `n_z = 0`.
BOTHWELL_2025_TABLE1_ROWS: tuple[dict[str, float], ...] = (
    {
        "u0": 56.8,
        "tr_nk": 650.0,
        "x_harmonic": 0.832,
        "x_bowkb": 0.785,
        "y_harmonic": 0.0627,
        "y_bowkb": 0.0608,
        "z_harmonic": 0.708,
        "z_bowkb": 0.645,
    },
    {
        "u0": 66.4,
        "tr_nk": 550.0,
        "x_harmonic": 0.863,
        "x_bowkb": 0.838,
        "y_harmonic": 0.0589,
        "y_bowkb": 0.0580,
        "z_harmonic": 0.756,
        "z_bowkb": 0.719,
    },
    {
        "u0": 86.2,
        "tr_nk": 600.0,
        "x_harmonic": 0.881,
        "x_bowkb": 0.864,
        "y_harmonic": 0.0520,
        "y_bowkb": 0.0515,
        "z_harmonic": 0.786,
        "z_bowkb": 0.759,
    },
    {
        "u0": 112.2,
        "tr_nk": 720.0,
        "x_harmonic": 0.892,
        "x_bowkb": 0.879,
        "y_harmonic": 0.0457,
        "y_bowkb": 0.0454,
        "z_harmonic": 0.804,
        "z_bowkb": 0.781,
    },
)


@dataclass(frozen=True)
class BothwellTable1Row:
    """One row's predicted-vs-published `X`/`Y`/`Z` comparison."""

    u0: float
    tr_nk: float
    x_predicted: float
    x_published_bowkb: float
    y_predicted: float
    y_published_bowkb: float
    z_predicted: float
    z_published_bowkb: float
    max_abs_relative_error: float


@dataclass(frozen=True)
class BothwellTable1ReproductionCase:
    """Target 3a: this module's BO+WKB `X`/`Y`/`Z` machinery
    (:func:`lattice_light_shift.axial_thermal_factors`) against Bothwell
    et al. 2025's own published BO+WKB column, all four table rows.
    """

    case_class: str
    case_label: str
    species_name: str
    n_z: int
    rows: tuple[dict[str, float], ...]
    worst_relative_error: float
    tolerance_relative: float
    kpi_verdict: str
    citation: str


def run_bothwell_table1_reproduction_case(
    tolerance_relative: float = 0.01,
) -> BothwellTable1ReproductionCase:
    """Reproduce Bothwell et al. 2025's Appendix A Table I BO+WKB column
    at all four published `(u0, Tr)` rows, `n_z = 0`, using
    :func:`lattice_light_shift.axial_thermal_factors` end to end (real
    finite-difference axial solve, real WKB turning-radius density of
    states, real thermal averaging, no shortcut).

    `tolerance_relative`, default `1%`: the table publishes `X`/`Y`/`Z` to
    3-4 significant figures, so this tolerance is generous relative to the
    paper's own rounding; verdict is `MET` only if every one of the twelve
    published BO+WKB values (4 rows x 3 factors) is reproduced within it.
    """
    yb = get_species("Yb171")
    wavelength_m = SPEED_OF_LIGHT / BOTHWELL_2025_MAGIC_FREQUENCY_HZ
    rows: list[BothwellTable1Row] = []
    for row in BOTHWELL_2025_TABLE1_ROWS:
        site = lls.make_site_potential(
            depth_er=row["u0"], waist_m=50e-6, wavelength_m=wavelength_m, mass_kg=yb.mass_kg
        )
        factors = lls.axial_thermal_factors(site, 0, row["tr_nk"] * 1e-9)
        errors = [
            abs(factors.x_nz - row["x_bowkb"]) / row["x_bowkb"],
            abs(factors.y_nz - row["y_bowkb"]) / row["y_bowkb"],
            abs(factors.z_nz - row["z_bowkb"]) / row["z_bowkb"],
        ]
        rows.append(
            BothwellTable1Row(
                u0=row["u0"],
                tr_nk=row["tr_nk"],
                x_predicted=factors.x_nz,
                x_published_bowkb=row["x_bowkb"],
                y_predicted=factors.y_nz,
                y_published_bowkb=row["y_bowkb"],
                z_predicted=factors.z_nz,
                z_published_bowkb=row["z_bowkb"],
                max_abs_relative_error=max(errors),
            )
        )
    worst = max(r.max_abs_relative_error for r in rows)
    return BothwellTable1ReproductionCase(
        case_class="arithmetic_reproduction",
        case_label=ARITHMETIC_REPRODUCTION_LABEL,
        species_name="Yb171",
        n_z=0,
        rows=tuple(asdict(r) for r in rows),
        worst_relative_error=worst,
        tolerance_relative=tolerance_relative,
        kpi_verdict="MET" if worst <= tolerance_relative else "NOT MET",
        citation=(
            "Bothwell, Hunt, Siegel, Hassan, Grogan, Kobayashi, Gibble, Porsev, "
            "Safronova, Brown, Beloy, Ludlow, PRL 134, 033201 (2025), Appendix A Table I"
        ),
    )


# ---------------------------------------------------------------------------
# Target 3b: Bothwell et al. 2025 headline coefficient, computable comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BothwellHeadlineComparisonCase:
    """Target 3b: both models evaluated at Bothwell et al. 2025's own
    stated operating conditions, using each model's own published
    coefficient column, with the resulting fractional-shift difference
    reported. See :data:`COMPUTABLE_COMPARISON_LABEL` for what this case
    does and does not settle.
    """

    case_class: str
    case_label: str
    species_name: str
    u0: float
    detuning_hz: float
    radial_temperature_k: float
    n_z: int
    harmonic_shift_fractional: float
    bowkb_shift_fractional: float
    model_difference_fractional: float
    published_harmonic_m1e2_hz: float
    published_bowkb_m1e2_hz: float
    published_m1e2_relative_difference: float
    citation: str


def run_bothwell_headline_comparison_case() -> BothwellHeadlineComparisonCase:
    """Evaluate Model A (harmonic) and Model B (BO+WKB) at Bothwell et al.
    2025's own stated Yb-171 operating conditions (`u0 = 100 E_R`,
    comfortably inside their stated `< 140 E_R` range; `delta_L = 0`, the
    on-magic-frequency case, since the paper does not tabulate a single
    representative detuning; `Tr = 600 nK`, their stated standing-wave
    radial temperature; `n_z = 0`, the dominant band), using each model's
    own published Table III coefficient column
    (`BOTHWELL_2025_YB171_HARMONIC`/`_BOWKB`), and reports the resulting
    fractional-shift difference between the two models at this one
    operating point, the computable half of Target 3 (see
    :data:`COMPUTABLE_COMPARISON_LABEL`).
    """
    yb = get_species("Yb171")
    wavelength_m = SPEED_OF_LIGHT / BOTHWELL_2025_MAGIC_FREQUENCY_HZ
    u0 = 100.0
    detuning_hz = 0.0
    tr_k = 600e-9

    harmonic_shift = lls.harmonic_light_shift_hz(
        u0,
        detuning_hz,
        0.0,
        lls.BOTHWELL_2025_YB171_HARMONIC,
        reduction_form="jila_reciprocal",
        radial_temperature_k=tr_k,
        recoil_energy_j_value=lls.recoil_energy_j(wavelength_m, yb.mass_kg),
    )
    site = lls.make_site_potential(
        depth_er=u0, waist_m=50e-6, wavelength_m=wavelength_m, mass_kg=yb.mass_kg
    )
    bowkb_shift, _factors = lls.bo_wkb_fractional_light_shift(
        0, u0, detuning_hz, tr_k, lls.BOTHWELL_2025_YB171_BOWKB, site
    )
    m1e2_h = lls.BOTHWELL_2025_YB171_HARMONIC.m1e2_hz
    m1e2_b = lls.BOTHWELL_2025_YB171_BOWKB.m1e2_hz
    return BothwellHeadlineComparisonCase(
        case_class="computable_comparison",
        case_label=COMPUTABLE_COMPARISON_LABEL,
        species_name="Yb171",
        u0=u0,
        detuning_hz=detuning_hz,
        radial_temperature_k=tr_k,
        n_z=0,
        harmonic_shift_fractional=harmonic_shift,
        bowkb_shift_fractional=bowkb_shift,
        model_difference_fractional=bowkb_shift - harmonic_shift,
        published_harmonic_m1e2_hz=m1e2_h,
        published_bowkb_m1e2_hz=m1e2_b,
        published_m1e2_relative_difference=(m1e2_b - m1e2_h) / m1e2_h,
        citation=(
            "Bothwell, Hunt, Siegel, Hassan, Grogan, Kobayashi, Gibble, Porsev, "
            "Safronova, Brown, Beloy, Ludlow, PRL 134, 033201 (2025), Table III "
            "and main text (u0 < 140 E_R, Tr ~= 600 nK)"
        ),
    )


# ---------------------------------------------------------------------------
# Density-of-states contrast
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DensityOfStatesContrastRow:
    """One radial-temperature sample's cumulative axial-band-0 state
    count (Beloy et al. 2020 Eqs. 4/11 integrated from the band bottom up
    to one thermal quantum above it) for both models, and their ratio.
    """

    radial_temperature_k: float
    cumulative_states_cos2: float
    cumulative_states_harmonic: float
    ratio_cos2_over_harmonic: float


@dataclass(frozen=True)
class DensityOfStatesContrastCase:
    """Both models' axial-band-0 degeneracy at matched `(u0, mass,
    wavelength)`, as a function of radial temperature: the cumulative
    number of radial states (Eqs. 4/11 integrated from the band bottom to
    `U_0(0) + kB*Tr`) for the true site potential (BO+WKB, Eq. 11) versus
    the harmonic closed form (Eq. 4). The dossier's qualitative claim
    (harmonic grows linearly in the thermal-excitation scale, BO+WKB
    grows faster) is checked here directly, as a computed ratio between
    the two models' own equations.
    """

    species_name: str
    u0: float
    n_z: int
    rows: tuple[dict[str, float], ...]


def run_density_of_states_contrast_case() -> DensityOfStatesContrastCase:
    """Compute the cumulative axial-band-0 (`n_z=0`) density of states for
    both models at Yb-171, `u0 = 100 E_R`, over a range of radial
    temperatures, and report the BO+WKB/harmonic ratio at each.
    """
    from scipy.integrate import quad

    from cliffordclock.constants import BOLTZMANN_K

    yb = get_species("Yb171")
    wavelength_m = SPEED_OF_LIGHT / BOTHWELL_2025_MAGIC_FREQUENCY_HZ
    u0 = 100.0
    site = lls.make_site_potential(
        depth_er=u0, waist_m=50e-6, wavelength_m=wavelength_m, mass_kg=yb.mass_kg
    )
    e_r = site.recoil_energy_j_value
    e0_cos2 = lls.axial_band_energy_er(site, 0, 0.0, potential="cos2") * e_r
    e0_harmonic = lls.axial_band_energy_er(site, 0, 0.0, potential="harmonic") * e_r

    def cumulative(e_top_j: float, potential: str, e0_j: float) -> float:
        value, _ = quad(
            lambda e: lls.bo_wkb_density_of_states(site, 0, e, potential=potential),
            e0_j,
            e_top_j,
            limit=100,
        )
        return value

    radial_temperatures_k = [50e-9, 100e-9, 200e-9, 400e-9, 800e-9, 1600e-9]
    rows = []
    for tr_k in radial_temperatures_k:
        kt = BOLTZMANN_K * tr_k
        n_cos2 = cumulative(min(e0_cos2 + kt, 0.0), "cos2", e0_cos2)
        n_harmonic = cumulative(min(e0_harmonic + kt, 0.0), "harmonic", e0_harmonic)
        rows.append(
            DensityOfStatesContrastRow(
                radial_temperature_k=tr_k,
                cumulative_states_cos2=n_cos2,
                cumulative_states_harmonic=n_harmonic,
                ratio_cos2_over_harmonic=(n_cos2 / n_harmonic if n_harmonic > 0 else float("nan")),
            )
        )
    return DensityOfStatesContrastCase(
        species_name="Yb171",
        u0=u0,
        n_z=0,
        rows=tuple(asdict(r) for r in rows),
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report() -> dict[str, Any]:
    """Build the full WP36 Phase 1 lattice-light-shift benchmark report as
    a JSON-serializable dict."""
    target1 = run_ushijima_operational_point_case()
    target2 = run_aeppli_lattice_line_item_case()
    target3a = run_bothwell_table1_reproduction_case()
    target3b = run_bothwell_headline_comparison_case()
    dos_contrast = run_density_of_states_contrast_case()
    return {
        "wp36_lattice_light_shift_benchmark_schema": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "target1_ushijima_operational_point": asdict(target1),
        "target2_aeppli_lattice_line_item": asdict(target2),
        "target3a_bothwell_table1_reproduction": asdict(target3a),
        "target3b_bothwell_headline_comparison": asdict(target3b),
        "density_of_states_contrast": asdict(dos_contrast),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the WP36 lattice-light-shift benchmark report as markdown."""
    t1 = report["target1_ushijima_operational_point"]
    t2 = report["target2_aeppli_lattice_line_item"]
    t3a = report["target3a_bothwell_table1_reproduction"]
    t3b = report["target3b_bothwell_headline_comparison"]
    dos = report["density_of_states_contrast"]

    lines = [
        "# WP36 lattice light shift benchmark cases (generated)",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "This report validates both community lattice-light-shift models "
        "(the Katori-lineage harmonic/operational model, and the NIST "
        "Born-Oppenheimer+WKB model) against their defining papers, before "
        "either is wired into the pipeline (a later phase).",
        "",
        "## Target 1: Ushijima et al. 2018 operational point",
        "",
        f"**Classification: {t1['case_class']}**",
        "",
        "| Quantity | Predicted | Published |",
        "|---|---|---|",
        (
            f"| u_op | {t1['u_op_predicted']:.3f} | "
            f"{t1['u_op_published']}({t1['u_op_published_uncertainty']:.0f}) |"
        ),
        (
            "| delta_L_op (MHz) | "
            f"{t1['detuning_hz_op_predicted'] / 1e6:.3f} | "
            f"{t1['detuning_hz_op_published'] / 1e6}"
            f"({t1['detuning_hz_op_published_uncertainty'] / 1e6:.1f}) |"
        ),
        f"| u_op within published uncertainty | {t1['u_op_within_published_uncertainty']} | |",
        (
            "| detuning_op within published uncertainty | "
            f"{t1['detuning_op_within_published_uncertainty']} | |"
        ),
        "",
        f"Source: {t1['citation']}",
        "",
        "## Target 2: Aeppli et al. 2024 lattice-light-shift budget line",
        "",
        f"**Classification: {t2['case_class']}**",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| u0 | {t2['u0']} |",
        f"| detuning (MHz) | {t2['detuning_hz'] / 1e6} |",
        f"| Tr (nK) | {t2['radial_temperature_k'] * 1e9:.0f} |",
        f"| Predicted shift (1e-19) | {t2['predicted_shift_fractional'] * 1e19:+.3f} |",
        (
            "| Predicted uncertainty (1e-19) | "
            f"+/-{t2['predicted_uncertainty_fractional'] * 1e19:.3f} |"
        ),
        f"| Published shift (1e-19) | {t2['published_shift_fractional'] * 1e19:+.3f} |",
        (
            "| Published uncertainty (1e-19) | "
            f"+/-{t2['published_uncertainty_fractional'] * 1e19:.3f} |"
        ),
        f"| Bands overlap | {t2['bands_overlap']} |",
        f"| **kpi_verdict** | **{t2['kpi_verdict']}** |",
        "",
        f"Source: {t2['citation']}",
        "",
        "## Target 3a: Bothwell et al. 2025 Table I, X/Y/Z reproduction",
        "",
        f"**Classification: {t3a['case_class']}**",
        "",
        (
            "| u0 (E_R) | Tr (nK) | X pred | X pub (BO+WKB) | Y pred | Y pub | "
            "Z pred | Z pub | max rel. err |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
        *(
            f"| {r['u0']} | {r['tr_nk']:.0f} | {r['x_predicted']:.4f} | "
            f"{r['x_published_bowkb']:.4f} | {r['y_predicted']:.4f} | "
            f"{r['y_published_bowkb']:.4f} | {r['z_predicted']:.4f} | "
            f"{r['z_published_bowkb']:.4f} | {r['max_abs_relative_error']:.2e} |"
            for r in t3a["rows"]
        ),
        "",
        f"Worst relative error across all 4 rows: {t3a['worst_relative_error']:.2e} "
        f"(tolerance {t3a['tolerance_relative']:.0%}). **kpi_verdict: {t3a['kpi_verdict']}**",
        "",
        f"Source: {t3a['citation']}",
        "",
        "## Target 3b: Bothwell et al. 2025 headline coefficient, computable comparison",
        "",
        f"**Classification: {t3b['case_class']}**",
        "",
        f"{COMPUTABLE_COMPARISON_LABEL}",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| u0 | {t3b['u0']} |",
        f"| Tr (nK) | {t3b['radial_temperature_k'] * 1e9:.0f} |",
        f"| Harmonic-model shift (fractional) | {t3b['harmonic_shift_fractional']:+.3e} |",
        f"| BO+WKB-model shift (fractional) | {t3b['bowkb_shift_fractional']:+.3e} |",
        f"| Model difference (fractional) | {t3b['model_difference_fractional']:+.3e} |",
        f"| Published alpha~M1E2/h, harmonic (Hz) | {t3b['published_harmonic_m1e2_hz']:+.3e} |",
        f"| Published alpha~M1E2/h, BO+WKB (Hz) | {t3b['published_bowkb_m1e2_hz']:+.3e} |",
        (
            "| Published relative difference between the two models' "
            f"alpha~M1E2 | {t3b['published_m1e2_relative_difference']:+.3%} |"
        ),
        "",
        f"Source: {t3b['citation']}",
        "",
        "## Density-of-states contrast",
        "",
        f"Species {dos['species_name']}, u0={dos['u0']}, n_z={dos['n_z']}. Cumulative "
        "number of radial states from the band bottom up to one radial-temperature "
        "thermal quantum above it, both models.",
        "",
        "| Tr (nK) | Cumulative states (cos2/BO+WKB) | Cumulative states (harmonic) | Ratio |",
        "|---|---|---|---|",
        *(
            f"| {r['radial_temperature_k'] * 1e9:.0f} | {r['cumulative_states_cos2']:.4e} | "
            f"{r['cumulative_states_harmonic']:.4e} | {r['ratio_cos2_over_harmonic']:.4f} |"
            for r in dos["rows"]
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run every WP36 Phase 1 case and write
    `benchmarks/results/wp36_lattice_light_shift.json` and its markdown
    summary."""
    report = build_report()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / "wp36_lattice_light_shift.json"
    md_path = _RESULTS_DIR / "wp36_lattice_light_shift.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
