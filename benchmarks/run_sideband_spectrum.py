# SPDX-License-Identifier: AGPL-3.0-or-later
"""WP38 Deliverable 2: sideband-spectrum forward model
(`cliffordclock.integrator.sideband_spectrum_jax`) cross-validated
against `large-lattice-model` (github.com/inrim/large-lattice-model,
MIT license, (c) 2021-2024 Marco Pizzocaro, INRIM), an INDEPENDENT
OPEN-SOURCE ORACLE.

**Provenance.** `large-lattice-model` is a real, public, third-party
implementation of Beloy et al. 2020's Born-Oppenheimer+WKB motional
model. It solves the axial eigenproblem with exact Mathieu-function
characteristic values, a different numerical method from this project's
own finite-difference solver, and finds the turning radius with
`scipy.optimize.brentq`. It is cited as reference [50] in Goti et al.
2025's own Fig. 4/Fig. 7 sideband fits (this project's private research
dossier, Phase 2 addendum). This project's own comparison values were
generated once, offline, by running `large-lattice-model`'s own code
against its own dedicated virtual environment, never CliffordClock's
own `.venv`: see
`benchmarks/fixtures/wp38_inrim_large_lattice_model_reference.json`'s
own `provenance` field for the exact commit hash, license, and
generation method. This script reads only the numeric JSON fixture that
generation run produced, the same "published-table-as-fixture" pattern
this project already uses for paper Table values
(`benchmarks/fixtures/jila_2403_10664_table1.csv`). Here that pattern
is applied to an independent implementation's OUTPUT, one step removed
from a paper's own published number.

**A new evidentiary class.** This project's established
`arithmetic_reproduction` label means "reproduces a PAPER's own
published number." Comparing against `large-lattice-model`'s output is
a different kind of evidence: `independent_implementation_reproduction`,
reproducing an independent CODE implementation's own output at matched
inputs, with a different numerical method on each side. A second
working codebase reaching the same answer is a form of corroboration a
single paper's own printed digits cannot supply on its own, and this
script labels every case with its own class throughout, so a reader
never confuses a case here with the paper-Table reproduction cases
`run_lattice_light_shift.py` reports.

**Three comparison tiers, from tightest to loosest, and why.**

1. :func:`run_band_energy_reproduction_case` (`independent_implementation_reproduction`):
   the axial Born-Oppenheimer eigenvalue `U_nz(0)/E_R` (Beloy et al. 2020
   Eq. 5, the band bottom), computed by this project's ALREADY G18-GATED
   finite-difference solver
   (`cliffordclock.integrator.lattice_light_shift.axial_energies_er`,
   run at ITS OWN converged tolerance) against `large-lattice-model`'s
   EXACT Mathieu-characteristic-value formula. No Lorentzian, no
   linewidth, no amplitude convention anywhere in this comparison: pure
   eigenvalue vs. pure eigenvalue, the tightest possible check.
2. :func:`run_condon_detuning_reproduction_case` (`independent_implementation_reproduction`):
   the Franck-Condon transition detuning `delta_nu(E)` (Goti et al. 2025
   Eq. 5), computed by THIS work package's own module
   (`sideband_spectrum_jax.condon_detuning_hz`, at its own
   spectrum-scale resolution, `AXIAL_GRID_N_SPECTRUM`) against
   `large-lattice-model`'s `DeltaU(R(E,D,nz),D,nz,1)`. Looser than tier
   1: it exercises this module's OWN lower-resolution, table-
   interpolated numerical route, while tier 1 exercises the tighter
   reference solver.
3. :func:`run_sideband_shape_comparison_case` (`computable_comparison`):
   the FULL normalized sideband shape
   (`sideband_spectrum_jax.bowkb_sideband_shape`) against
   `large-lattice-model`'s own `sidebands()` output. This is the
   weakest tier. The two sides use GENUINELY DIFFERENT lineshape
   conventions that this script bridges but cannot eliminate (see
   :data:`CONVENTION_BRIDGES` below), so this tier is labeled
   `computable_comparison`, the class this project reserves for a
   comparison bridged across documented convention differences: a
   Lorentzian peak-height factor of `2` (`large-lattice-model`'s own
   `lorentzian()` peaks at `0.5`, this project's at `1`), a
   State-dependent (Rabi-weighted) linewidth on `large-lattice-model`'s
   side versus this project's own fixed `linewidth_hz`, and a fixed
   `E_max=0` cutoff on `large-lattice-model`'s side versus this
   project's own exact target-band-boundedness mask (module docstring's
   "Integration domain" section). What tier 3 CAN and does check,
   bridged for those differences: the sideband's own peak-detuning
   position (a convention-independent physical quantity) and a
   convention-normalized shape correlation.

Run this yourself: ``python benchmarks/run_sideband_spectrum.py`` (from
the repo root, with ``.venv`` active). Regenerates
``benchmarks/results/wp38_sideband_spectrum.json`` and the accompanying
``.md`` summary. Reads
``benchmarks/fixtures/wp38_inrim_large_lattice_model_reference.json``
(committed). Regenerating that fixture itself requires
`large-lattice-model` installed in a SEPARATE environment from this
project's own `.venv`. This script's own module docstring above and
that fixture's own `provenance` field together carry the exact
regeneration recipe, including its one non-default argument:
`large_lattice_model.sidebands.sidebands`'s own `fac` parameter
(default `10`, "controlling the number of lorentzian functions used to
calculate the sideband shape," that function's own docstring) is called
at `fac=20` for this fixture's `sideband_spectra` rows, for a smoother
reference curve. Every other value in this fixture (`band_bottoms`,
`condon_detunings`) comes from `large-lattice-model`'s own `U`/`R`/`DeltaU`
at every argument left at that library's own default.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import cliffordclock.integrator.lattice_light_shift as lls
from cliffordclock.constants import SPEED_OF_LIGHT
from cliffordclock.ensemble.species import get_species
from cliffordclock.integrator import sideband_spectrum_jax as ssj
from cliffordclock.integrator.lattice_light_shift_jax import make_site_potential_jax

_BENCHMARKS_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _BENCHMARKS_DIR / "results"
_FIXTURE_PATH = _BENCHMARKS_DIR / "fixtures" / "wp38_inrim_large_lattice_model_reference.json"

#: Bothwell et al. 2025's own measured Yb-171 E1 magic frequency, reused
#: from `run_lattice_light_shift.py`'s own Target 3 constant (this
#: project's own species/wavelength convention); close to but not
#: identical to large-lattice-model's own default `394798.267e9` Hz
#: (relative difference `~1.7e-9`, negligible against every tolerance
#: this script checks).
YB171_MAGIC_FREQUENCY_HZ = 394_798_266.9e6

WAIST_M = 50e-6
PROBE_WAVELENGTH_M = 578e-9

INDEPENDENT_IMPLEMENTATION_LABEL = (
    "independent_implementation_reproduction: reproduces the output of "
    "large-lattice-model (github.com/inrim/large-lattice-model, MIT "
    "license, INRIM), a real third-party open-source implementation of "
    "Beloy et al. 2020's Born-Oppenheimer+WKB model. That implementation "
    "solves the axial eigenproblem with exact Mathieu-function "
    "characteristic values; this project's own solver uses a "
    "finite-difference method. This project's `arithmetic_reproduction` "
    "class is reserved "
    "for a PAPER's own published number; this class checks two "
    "independent CODE implementations against each other."
)

COMPUTABLE_COMPARISON_LABEL = (
    "computable_comparison: the full sideband lineshape compared against "
    "large-lattice-model's own `sidebands()` output at matched (D, Tz, "
    "Tr), bridging documented convention differences (Lorentzian peak "
    "height, state-dependent vs. fixed linewidth, integration-domain "
    "cutoff: see this script's module docstring's tier-3 description). "
    "The two convention-independent quantities checked directly are peak "
    "position and shape correlation."
)


def _load_fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _yb171_mass_kg() -> float:
    return get_species("Yb171").mass_kg


def _yb171_wavelength_m() -> float:
    return SPEED_OF_LIGHT / YB171_MAGIC_FREQUENCY_HZ


# ---------------------------------------------------------------------------
# Tier 1: band-bottom eigenvalue reproduction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BandEnergyRow:
    depth_er: float
    n_z: int
    predicted_er: float
    inrim_er: float
    relative_error: float


@dataclass(frozen=True)
class BandEnergyReproductionCase:
    case_class: str
    case_label: str
    rows: list[dict[str, Any]]
    worst_relative_error: float
    tolerance_relative: float
    kpi_verdict: str
    citation: str


def run_band_energy_reproduction_case(
    fixture: dict[str, Any], *, tolerance_relative: float = 1e-4
) -> BandEnergyReproductionCase:
    """Tier 1 (module docstring): `lattice_light_shift.axial_energies_er`
    (this project's OWN already-G18-gated, converged finite-difference
    solver) vs. large-lattice-model's exact Mathieu-characteristic-value
    `U(0, D, nz)`, at every `(D, nz)` row the fixture carries.
    """
    rows: list[BandEnergyRow] = []
    depths = sorted({row["D_er"] for row in fixture["band_bottoms"]})
    for depth in depths:
        depth_rows = [r for r in fixture["band_bottoms"] if r["D_er"] == depth]
        n_states = max(r["nz"] for r in depth_rows) + 1
        predicted = lls.axial_energies_er(depth, n_states)
        for r in sorted(depth_rows, key=lambda r: r["nz"]):
            nz = r["nz"]
            pred = float(predicted[nz])
            pub = r["u_nz_0_er"]
            rel_err = abs(pred - pub) / abs(pub)
            rows.append(
                BandEnergyRow(
                    depth_er=depth, n_z=nz, predicted_er=pred, inrim_er=pub, relative_error=rel_err
                )
            )
    worst = max(r.relative_error for r in rows)
    verdict = "PASS" if worst <= tolerance_relative else "FAIL"
    return BandEnergyReproductionCase(
        case_class="independent_implementation_reproduction",
        case_label=INDEPENDENT_IMPLEMENTATION_LABEL,
        rows=[asdict(r) for r in rows],
        worst_relative_error=worst,
        tolerance_relative=tolerance_relative,
        kpi_verdict=verdict,
        citation=(
            "large-lattice-model, github.com/inrim/large-lattice-model, MIT, "
            "M. Pizzocaro, commit " + fixture["provenance"]["commit"]
        ),
    )


# ---------------------------------------------------------------------------
# Tier 2: Condon-point (Franck-Condon) detuning reproduction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CondonDetuningRow:
    depth_er: float
    n_z: int
    energy_er: float
    predicted_hz: float
    inrim_hz: float
    absolute_error_hz: float
    relative_error: float
    near_band_top: bool


@dataclass(frozen=True)
class CondonDetuningReproductionCase:
    case_class: str
    case_label: str
    rows: list[dict[str, Any]]
    worst_relative_error: float
    tolerance_relative: float
    kpi_verdict: str
    citation: str


def run_condon_detuning_reproduction_case(
    fixture: dict[str, Any], *, tolerance_relative: float = 2e-2
) -> CondonDetuningReproductionCase:
    """Tier 2: `sideband_spectrum_jax.condon_detuning_hz` (Goti et al.
    2025 Eq. 5, this WORK PACKAGE's own table-interpolated numerical
    route) vs. large-lattice-model's `DeltaU(R(E,D,nz),D,nz,1)*Er/h`, at
    the fixture's own `(D, E)` grid, band `nz=0` only (the fixture's own
    coverage). The looser tolerance than tier 1 (`2%` vs. `0.01%`)
    reflects this module's OWN spectrum-scale resolution choice
    (`AXIAL_GRID_N_SPECTRUM=321`, `RHO_TABLE_N=129`, linear
    interpolation). See
    `tests/test_sideband_spectrum_jax.py::TestOfflineConvergenceStudy`
    for the same comparison run at the tighter reference resolution,
    where the agreement tightens to tier 1's own level.

    **A known, stated resolution limit near the very top of the band.**
    This case excludes points near the band top from its tolerance
    check and reports them separately, with a `near_band_top` flag. A
    single looser tolerance covering those points too would hide the
    reason for the difference. Within about
    `5 E_R` of the band edge (`E -> 0`, `n_r -> infinity`, the classical
    turning radius growing without bound as the local trap depth
    vanishes), this module's finite (`RHO_TABLE_N=129`-point) radial
    table and linear interpolation lose accuracy fast: at
    `E/E_bottom ~ 1e-6` (the fixture's own closest sampled point to the
    band top), this module's predicted detuning collapses toward `0`
    while large-lattice-model's own root-find (bracketed to a fixed
    `rho <= 2.7/kappa`, its own `R()` docstring) reports a persistent,
    non-vanishing detuning. Both routes are doing something numerically
    delicate at that limit: this module's table runs out of resolution,
    and large-lattice-model's own bracket has a stated fixed radius
    ceiling. This benchmark reports both numbers at every near-band-top
    point and leaves which one is more accurate there as an open
    question. Points within `5 E_R` of the band top carry the
    `near_band_top` flag in every row below.
    """
    mass_kg = _yb171_mass_kg()
    wavelength_m = _yb171_wavelength_m()
    rows: list[CondonDetuningRow] = []
    depths = sorted({row["D_er"] for row in fixture["condon_detunings"]})
    for depth in depths:
        site = make_site_potential_jax(depth, WAIST_M, wavelength_m, mass_kg)
        table0 = ssj.build_band_energy_table(site, 0)
        table1 = ssj.build_band_energy_table(site, 1)
        depth_rows = [r for r in fixture["condon_detunings"] if r["D_er"] == depth]
        for r in depth_rows:
            e_er = r["E_er"]
            pred_hz = float(
                ssj.condon_detuning_hz(table0, table1, np.asarray(e_er), site.recoil_energy_j_value)
            )
            pub_hz = r["blue_detuning_hz"]
            abs_err = abs(pred_hz - pub_hz)
            rel_err = abs_err / abs(pub_hz) if abs(pub_hz) > 1.0 else abs_err / 1.0
            rows.append(
                CondonDetuningRow(
                    depth_er=depth,
                    n_z=0,
                    energy_er=e_er,
                    predicted_hz=pred_hz,
                    inrim_hz=pub_hz,
                    absolute_error_hz=abs_err,
                    relative_error=rel_err,
                    near_band_top=abs(e_er) < 5.0,
                )
            )
    # Exclude points within 5 E_R of the band top (docstring's "known,
    # stated resolution limit" section): both this module's table
    # interpolation and large-lattice-model's own fixed-radius bracket
    # are numerically delicate there, and this benchmark does not
    # adjudicate which side is more accurate at that specific limit.
    interior = [r for r in rows if not r.near_band_top]
    worst = max(r.relative_error for r in interior)
    verdict = "PASS" if worst <= tolerance_relative else "FAIL"
    return CondonDetuningReproductionCase(
        case_class="independent_implementation_reproduction",
        case_label=INDEPENDENT_IMPLEMENTATION_LABEL,
        rows=[asdict(r) for r in rows],
        worst_relative_error=worst,
        tolerance_relative=tolerance_relative,
        kpi_verdict=verdict,
        citation=(
            "large-lattice-model, github.com/inrim/large-lattice-model, MIT, "
            "M. Pizzocaro, commit " + fixture["provenance"]["commit"]
        ),
    )


# ---------------------------------------------------------------------------
# Tier 3: full sideband-shape comparison, convention-bridged
# ---------------------------------------------------------------------------

#: Every convention difference this script must bridge to compare
#: `bowkb_sideband_shape` against large-lattice-model's `sidebands()`,
#: with WHERE each side defines its own convention.
CONVENTION_BRIDGES: dict[str, str] = {
    "lorentzian_peak_height": (
        "large-lattice-model's own lorentzian(x,x0,w) (latticemodel.py) "
        "peaks at 0.5 (den=1+(x-x0)^2/w^2, returns 0.5/den); this "
        "project's harmonic_sideband_shape/bowkb_sideband_shape (Blatt "
        "et al. 2009 App. A1's own Lorentzian form) peak at 1. Bridged by "
        "comparing peak-normalized SHAPE, canceling the height "
        "difference before any comparison runs."
    ),
    "linewidth_convention": (
        "large-lattice-model scales each (nz, rc) state's own Lorentzian "
        "width by its normalized Rabi frequency (rabi_ho, latticemodel.py, "
        "Wineland and Itano 1979 Eq. 31), so `wc` is a per-state-scaled "
        "half-width; this project's linewidth_hz is a single FIXED "
        "half-width shared by every (n_z, E) term (module docstring's own "
        "stated Blatt et al. 2009 approximation, 'given by the carrier "
        "Rabi frequency'). Bridged by using large-lattice-model's own "
        "wc=2000 Hz directly as this project's linewidth_hz, an "
        "approximate match between two different conventions."
    ),
    "integration_domain_e_max": (
        "large-lattice-model's own sidebands() fixes E_max=0.0 for both "
        "blue and red (sidebands.py); this project's bowkb_sideband_shape "
        "masks by the target-band-boundedness condition directly (module "
        "docstring's 'Integration domain' section). Both converge to the "
        "same physical boundary for a deep trap; bridged by comparing "
        "shapes at moderate depth (D=80-100 E_R), where the two "
        "conditions land close together."
    ),
    "depth_definition": (
        "Both sides define D/u0 identically: peak trap depth in units of "
        "the recoil energy E_R = h^2/(2*m*lambda^2) (large-lattice-model's "
        "settings.py; this project's recoil_energy_j_jax). No bridging "
        "needed."
    ),
    "temperature_convention": (
        "Both sides take Tz/Tr as ordinary kelvin, two independent "
        "temperatures for the longitudinal and radial degrees of freedom "
        "(Beloy et al. 2020's own two-temperature ansatz, Goti et al. "
        "2025 Eq. 9). No bridging needed."
    ),
}


@dataclass(frozen=True)
class SidebandShapeRow:
    depth_er: float
    temperature_z_k: float
    temperature_r_k: float
    sideband: str
    inrim_peak_hz: float
    predicted_peak_hz: float
    peak_position_difference_hz: float
    shape_correlation: float


@dataclass(frozen=True)
class SidebandShapeComparisonCase:
    case_class: str
    case_label: str
    convention_bridges: dict[str, str]
    rows: list[dict[str, Any]]
    worst_peak_position_difference_hz: float
    min_shape_correlation: float
    citation: str


def run_sideband_shape_comparison_case(fixture: dict[str, Any]) -> SidebandShapeComparisonCase:
    """Tier 3 (module docstring): full-shape comparison, bridged per
    :data:`CONVENTION_BRIDGES`.
    """
    mass_kg = _yb171_mass_kg()
    wavelength_m = _yb171_wavelength_m()
    rows: list[SidebandShapeRow] = []
    for entry in fixture["sideband_spectra"]:
        depth = entry["D_er"]
        tz, tr = entry["Tz_k"], entry["Tr_k"]
        wc = entry["wc_hz"]
        for sideband, sign, freq_key in (("blue", 1, "blue_freq_hz"), ("red", -1, "red_freq_hz")):
            freq_hz = np.asarray(entry[freq_key])
            inrim_vals = np.asarray(entry[sideband])
            predicted = np.asarray(
                ssj.bowkb_sideband_shape(
                    freq_hz, sign, depth, WAIST_M, wavelength_m, mass_kg, tz, tr, wc
                )
            )
            inrim_peak_hz = float(freq_hz[np.argmax(inrim_vals)])
            predicted_peak_hz = float(freq_hz[np.argmax(predicted)])
            # Shape correlation: both curves peak-normalized to their own
            # max before correlating, so the (bridged-away) overall
            # amplitude convention cannot influence the reported number.
            inrim_norm = inrim_vals / np.max(inrim_vals) if np.max(inrim_vals) > 0 else inrim_vals
            pred_norm = predicted / np.max(predicted) if np.max(predicted) > 0 else predicted
            if np.std(inrim_norm) > 0 and np.std(pred_norm) > 0:
                corr = float(np.corrcoef(inrim_norm, pred_norm)[0, 1])
            else:
                corr = float("nan")
            rows.append(
                SidebandShapeRow(
                    depth_er=depth,
                    temperature_z_k=tz,
                    temperature_r_k=tr,
                    sideband=sideband,
                    inrim_peak_hz=inrim_peak_hz,
                    predicted_peak_hz=predicted_peak_hz,
                    peak_position_difference_hz=abs(predicted_peak_hz - inrim_peak_hz),
                    shape_correlation=corr,
                )
            )
    worst_peak = max(r.peak_position_difference_hz for r in rows)
    min_corr = min(r.shape_correlation for r in rows)
    return SidebandShapeComparisonCase(
        case_class="computable_comparison",
        case_label=COMPUTABLE_COMPARISON_LABEL,
        convention_bridges=CONVENTION_BRIDGES,
        rows=[asdict(r) for r in rows],
        worst_peak_position_difference_hz=worst_peak,
        min_shape_correlation=min_corr,
        citation=(
            "large-lattice-model, github.com/inrim/large-lattice-model, MIT, "
            "M. Pizzocaro, commit " + fixture["provenance"]["commit"]
        ),
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report() -> dict[str, Any]:
    fixture = _load_fixture()
    tier1 = run_band_energy_reproduction_case(fixture)
    tier2 = run_condon_detuning_reproduction_case(fixture)
    tier3 = run_sideband_shape_comparison_case(fixture)
    return {
        "wp38_sideband_spectrum_cross_validation_schema": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "fixture_provenance": fixture["provenance"],
        "tier1_band_energy_reproduction": asdict(tier1),
        "tier2_condon_detuning_reproduction": asdict(tier2),
        "tier3_sideband_shape_comparison": asdict(tier3),
    }


def render_markdown(report: dict[str, Any]) -> str:
    t1 = report["tier1_band_energy_reproduction"]
    t2 = report["tier2_condon_detuning_reproduction"]
    t3 = report["tier3_sideband_shape_comparison"]
    lines = [
        "# WP38 Deliverable 2: sideband-spectrum cross-validation against large-lattice-model",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "Independent oracle: large-lattice-model "
        "(github.com/inrim/large-lattice-model), MIT license, INRIM, "
        f"commit `{report['fixture_provenance']['commit']}`. No code from "
        "that repository enters CliffordClock; only its numeric output "
        "(a fixture JSON, generated in a separate environment) is compared "
        "here.",
        "",
        "## Tier 1: axial band-bottom eigenvalue reproduction",
        "",
        f"**Classification: {t1['case_class']}**",
        "",
        f"Worst relative error: {t1['worst_relative_error']:.2e} "
        f"(tolerance {t1['tolerance_relative']:.0%}). **kpi_verdict: {t1['kpi_verdict']}**",
        "",
        "| D (E_R) | n_z | CliffordClock (E_R) | INRIM Mathieu (E_R) | rel. err |",
        "|---|---|---|---|---|",
        *(
            f"| {r['depth_er']} | {r['n_z']} | {r['predicted_er']:.6f} | "
            f"{r['inrim_er']:.6f} | {r['relative_error']:.2e} |"
            for r in t1["rows"]
        ),
        "",
        "## Tier 2: Franck-Condon detuning reproduction",
        "",
        f"**Classification: {t2['case_class']}**",
        "",
        f"Worst relative error, excluding points within 5 E_R of the band "
        f"top (see this case's own docstring for why): "
        f"{t2['worst_relative_error']:.2e} (tolerance {t2['tolerance_relative']:.0%}). "
        f"**kpi_verdict: {t2['kpi_verdict']}**. "
        f"{sum(1 for r in t2['rows'] if r['near_band_top'])} of "
        f"{len(t2['rows'])} points excluded as near-band-top.",
        "",
        "## Tier 3: full sideband-shape comparison (convention-bridged)",
        "",
        f"**Classification: {t3['case_class']}**",
        "",
        "Convention bridges:",
        "",
        *(f"- **{k}**: {v}" for k, v in t3["convention_bridges"].items()),
        "",
        "| D (E_R) | Tz (uK) | Tr (uK) | sideband | INRIM peak (Hz) | "
        "predicted peak (Hz) | diff (Hz) | shape corr. |",
        "|---|---|---|---|---|---|---|---|",
        *(
            f"| {r['depth_er']} | {r['temperature_z_k'] * 1e6:.2f} | "
            f"{r['temperature_r_k'] * 1e6:.2f} | {r['sideband']} | "
            f"{r['inrim_peak_hz']:.0f} | {r['predicted_peak_hz']:.0f} | "
            f"{r['peak_position_difference_hz']:.0f} | {r['shape_correlation']:.4f} |"
            for r in t3["rows"]
        ),
        "",
        f"Worst peak-position difference: {t3['worst_peak_position_difference_hz']:.0f} Hz. "
        f"Minimum shape correlation: {t3['min_shape_correlation']:.4f}.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / "wp38_sideband_spectrum.json"
    md_path = _RESULTS_DIR / "wp38_sideband_spectrum.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
