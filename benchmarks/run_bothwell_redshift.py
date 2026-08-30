# SPDX-License-Identifier: AGPL-3.0-or-later
"""WP22 Part 3 benchmark case: Bothwell et al. 2022 mm-scale gravitational
redshift (WP22 Part 3;
the project's theory sign-off record (G9) Part B;
Bothwell et al., Nature 602, 420 (2022), arXiv:2109.12238).

Bothwell et al. (Nature 602, 420 (2022); preprint arXiv:2109.12238, read
cover to cover, dossier date 2026-08-11) measured the gravitational
redshift gradient across a millimetre-scale ``87``Sr optical-lattice
sample -- the first single-apparatus measurement of General Relativity's
gravitational time dilation at this length scale. This script configures
the REAL ``ensemble.regime: lattice_extended`` pipeline (WP22 Part 2) to
their sample geometry and lets the actual per-site machinery (Gaussian
occupation envelope, per-site Hermite-Gauss motional quadrature, the E36
gravitational-redshift pivot term) predict the per-site slope end to end
-- **not a shortcut ``g/c^2`` arithmetic formula**: the physics of that
formula is textbook (any reader can compute it on a calculator), so the
point of this case is exercising the full extended-sample pipeline
(geometry -> per-site pivot evaluation -> weighted-least-squares slope
fit) against Bothwell's own published, independently measured map slope,
with zero adjustable inputs.

**Binding classification label (G9 sign-off B4, ratified): "reproducibility,
with the inverted-NPL caveat."** `benchmarks/run_benchmarks.py`'s NPL case
established this project's ``case_class = "reproducibility"`` vocabulary:
this engine + a published field + a published polarizability reconstructs
what the *authors themselves already computed* from the same inputs -- not
a `"blind_prediction"`. The Bothwell case is structurally the SAME class,
but with the caveat INVERTED relative to NPL's: the ``g/c^2`` arithmetic
here is textbook and Bothwell computed it themselves trivially (unlike
NPL's differential-polarizability reconstruction, which took real domain
expertise); what this case demonstrates is NOT the arithmetic but the
extended-sample MACHINERY -- the per-site geometry, Gaussian-envelope
weighting, and map assembly -- producing the right measured-map slope
end-to-end. **This caveat is restated wherever the case appears** (this
module docstring, :data:`CASE_LABEL`, the dataclass docstring, and the
rendered markdown) per the G9 sign-off's explicit instruction ("make sure
the caveat rides wherever the case appears, not just once").

**Coordinate-sign mapping (G9 sign-off A1, required, gate edit 2).**
Bothwell's own z-axis puts LOWER physical positions at LARGER coordinate,
so their published redshift gradient is NEGATIVE (Table 1: "Known
Redshift -10.9e-20/mm"), while this engine's ``(P-1)_grav`` increases with
physical height (CONVENTIONS.md section 15 E36's sign convention: a HIGHER
clock runs FASTER). This script computes the slope in the engine's own
physical-height convention (:data:`BothwellRedshiftCase.predicted_slope_engine_convention_per_m`,
positive for `up_axis = +z`) and then NEGATES it for the comparison against
Bothwell's own published sign convention
(:data:`BothwellRedshiftCase.predicted_slope_per_mm`) -- so the sign
agreement with their published gradient is deliberate, stated explicitly
here, not coincidental.

**Geometry (dossier section "Sample geometry", B3 ratified with the
INFERRED flag).** ``~100,000`` :sup:`87`\\ Sr atoms, magic-wavelength
(813 nm) lattice, Gaussian atom-number envelope along the lattice axis,
sample spanning ``~1-1.3 mm``, imaging resolution 6.04 um/pixel, analysis
restricted to +/-1.5-sigma regions of ~100 pixels each. The dossier does
not state sigma directly; **this script derives it (flagged INFERRED)**
from "two +/-1.5-sigma regions, ~100 pixels each" at 6.04 um/pixel:
``sigma = (100 * 6.04um) / 1.5 ~= 402.7 um`` (:data:`ENVELOPE_SIGMA_M`).
Site spacing is likewise **INFERRED** from the 813 nm magic wavelength's
lambda/2 spacing, 406.5 nm (dossier B3, "813/2" check). This script's
computational site grid spans +/-3-sigma (:data:`N_SIGMA_HALF_SPAN`,
comfortably covering the +/-1.5-sigma analysis region with margin) at the
REAL 406.5 nm site spacing -- ~5900 sites, each a single-node (motional
ground-state-only) evaluation, since the dossier gives no per-site trap
frequency and the mm-scale gravitational gradient this case targets does
not depend on motional averaging within a site (E29's own scope: static,
v=0 nodes).

**Isolated gravitational gradient (dossier "What the published slope
already has removed").** Bothwell's published corrected slope has
per-pixel density and second-order-Zeeman corrections applied before the
linear fit, PLUS budget-level corrections for BBR (0(0.3)e-20/mm),
lattice light (-0.5(0.1)e-20/mm), DC Stark (+0.3(0.2)e-20/mm), and pixel
calibration (0(0.8)e-20/mm) (their Table 1) -- so their corrected number
targets the ISOLATED gravitational gradient E36 predicts, with no other
systematic mixed in. This script's own field is configured to exactly
zero (``field.synthetic.kind="uniform"``, ``e0=[0,0,0]``) for the same
reason: it isolates E36, matching what the comparison target actually is.
Their own DC-Stark gradient row is in-scope physics for this engine and
worth a narrative line (dossier), not a term this case models.

**Comparison targets (G9 sign-off B2, ratified).** Both corrected
measurements: method A (14-dataset campaign) -9.8(2.3)e-20/mm, method B
(synchronous two-region) -1.28(27)e-19/mm (:data:`MEASURED_SLOPE_METHOD_A`/
:data:`MEASURED_SLOPE_METHOD_B`). Slope-level comparison only -- no
per-slice dataset is available (dossier "Data availability", a hard
negative: no deposited per-pixel map found anywhere accessible).

**Reference gravity (G9 sign-off B1, ratified-with-edit).** Bothwell's own
USGS-surveyed LOCAL value, ``g = 9.796 m/s^2`` (van Westrum, NOAA Tech.
Memo NOS NGS-77 (2019)), is pinned as this case's input
(:data:`BOTHWELL_SURVEYED_G_M_S2`) -- the physically correct choice at the
1e-19 level (CONVENTIONS.md section 15); the engine's OWN default
(`cliffordclock.constants.STANDARD_GRAVITY`) stays standard gravity, a
placeholder, for every other (non-Bothwell) run.

Run this yourself: ``python benchmarks/run_bothwell_redshift.py`` (from
the repo root, with ``.venv`` active). Regenerates
``benchmarks/results/bothwell_redshift.json`` and
``benchmarks/results/bothwell_redshift.md``. Deliberately NOT merged into
``benchmarks/run_benchmarks.py``'s WP10 report or
``benchmarks/results/wp10_results.json``'s frozen ``kpi_summary`` counts
(mirrors ``benchmarks/run_bbr_jila_arithmetic_reproduction.py``'s
precedent of a separate script/report) -- see the WP22 builder report for
the headline-count question this raises (this case IS ``case_class =
"reproducibility"``, unlike the BBR case's weaker
``"arithmetic_reproduction"`` class, so whether/how it should join the
WP10 headline ``reproducibility`` count is a wording decision left to the
coordinator/owner, not decided by this script).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow running as `python benchmarks/run_bothwell_redshift.py` (no package
# install needed -- benchmarks/ is deliberately not part of the installed
# package); mirrors how `benchmarks/run_bbr_jila_arithmetic_reproduction.py`
# imports its sibling `loaders`/`run_benchmarks` modules.
_BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import numpy as np  # noqa: E402
from loaders import PublishedBand  # noqa: E402
from run_benchmarks import _bands_overlap  # noqa: E402 -- reuses the already-tested helper

from cliffordclock.constants import SPEED_OF_LIGHT, STANDARD_GRAVITY  # noqa: E402
from cliffordclock.pipeline import PipelineConfig, run_pipeline_full  # noqa: E402

_RESULTS_DIR = _BENCHMARKS_DIR / "results"

#: Standard-gravity g/c^2, computed here purely for a printed reference
#: line in `main()` -- never used as this case's actual reference gravity
#: (see BOTHWELL_SURVEYED_G_M_S2, the pinned Bothwell-specific input).
STANDARD_GRAVITY_OVER_C2 = STANDARD_GRAVITY / SPEED_OF_LIGHT**2

#: The exact, binding classification label (G9 sign-off B4) -- callers/docs/
#: tests should compare against this constant rather than re-typing the
#: string, so a future wording edit is a single-source change.
CASE_LABEL = (
    "reproducibility, with the INVERTED-NPL caveat: the g/c^2 arithmetic is "
    "textbook and the authors computed it themselves trivially (unlike NPL's "
    "differential-polarizability reconstruction); what this case validates is "
    "the extended-sample MACHINERY (per-site geometry, Gaussian-envelope "
    "weighting, map assembly) producing the right measured-map slope "
    "end-to-end against a published measured map, with zero adjustable "
    "inputs; it does not change the blind-prediction count."
)

# ---------------------------------------------------------------------------
# Geometry (dossier "Sample geometry"; B3 ratified with the INFERRED flag).
# ---------------------------------------------------------------------------

#: Magic wavelength for 87-Sr's clock transition lattice (dossier: "813 nm
#: shallow large-waist lattice").
MAGIC_WAVELENGTH_M = 813e-9

#: Lattice site spacing, INFERRED from lambda/2 (dossier B3: "site spacing
#: ~406.5 nm, INFERRED from lambda/2"; 813/2 = 406.5, the dossier's own
#: cross-check). Discretization-only (B3: "affects site discretization
#: only, not the slope") -- this project does not derive it from a
#: primary-text statement of the spacing itself.
SITE_SPACING_M = MAGIC_WAVELENGTH_M / 2.0

#: Imaging resolution, i.e. the pixel size the published map/analysis is
#: binned at (dossier: "imaging resolution 6.04 um/pixel ~= 15 lattice
#: sites"; 6.04e-6 / 406.5e-9 ~= 14.9, the dossier's own cross-check).
PIXEL_SIZE_M = 6.04e-6

#: Pixels per +/-1.5-sigma analysis region (dossier: "two-region method:
#: two +/-1.5sigma regions, ~100 pixels each").
PIXELS_PER_HALF_ANALYSIS_REGION = 100

#: Gaussian envelope standard deviation, INFERRED (the dossier states the
#: analysis window in pixels, not sigma directly): a +/-1.5-sigma region of
#: ~100 pixels at 6.04 um/pixel implies
#: ``sigma = (100 * 6.04um) / 1.5``. Flagged INFERRED throughout this
#: script's output, per the same discipline as SITE_SPACING_M (B3).
ENVELOPE_SIGMA_M = (PIXELS_PER_HALF_ANALYSIS_REGION * PIXEL_SIZE_M) / 1.5

#: This script's computational site grid covers +/-N_SIGMA_HALF_SPAN sigma
#: of the Gaussian envelope -- comfortably beyond the +/-1.5-sigma analysis
#: window (>99.7% of the envelope's mass) while keeping the site count
#: (~5900 at the real 406.5 nm spacing) a fast, single vectorized
#: `fast_path` evaluation. Not itself a claim about the physical sample's
#: total extent (the dossier's own "~1-1.3 mm span" is a separate,
#: consistent-order-of-magnitude number: 2*3*402.7um ~= 2.4mm, versus
#: dossier "~1-1.3mm" -- both describe "most of the atom-number envelope,"
#: not identical cutoffs, and the fit itself (see module docstring) is
#: exactly window-independent for this field-free, purely-linear-in-height
#: case).
N_SIGMA_HALF_SPAN = 3.0


def _n_sites_covering(half_span_m: float, spacing_m: float) -> int:
    """Smallest ODD site count whose extent covers +/- `half_span_m` at
    `spacing_m` center-to-center spacing (odd: symmetric about offset 0,
    matching `extended_lattice_nodes`'s own site-centering convention)."""
    half_sites = math.ceil(half_span_m / spacing_m)
    return 2 * half_sites + 1


#: Computational site count (see N_SIGMA_HALF_SPAN's docstring).
N_SITES = _n_sites_covering(N_SIGMA_HALF_SPAN * ENVELOPE_SIGMA_M, SITE_SPACING_M)

# ---------------------------------------------------------------------------
# Reference gravity (G9 sign-off B1) and comparison targets (B2).
# ---------------------------------------------------------------------------

#: Bothwell's own USGS-surveyed LOCAL gravitational acceleration (dossier:
#: "Known Redshift ... computed from their USGS-surveyed LOCAL gravity
#: a = 9.796 m/s^2 (Methods, 'Known Redshift'; ref: van Westrum, NOAA Tech.
#: Memo NOS NGS-77 (2019))"). Pinned as this case's reference input (G9
#: sign-off B1); the engine's own default (STANDARD_GRAVITY) is a
#: placeholder, unaffected by this case-specific override.
BOTHWELL_SURVEYED_G_M_S2 = 9.796
_BOTHWELL_G_CITATION = (
    "van Westrum, D., NOAA Technical Memorandum NOS NGS-77 (2019), as cited "
    "in Bothwell et al., Nature 602, 420 (2022) / arXiv:2109.12238 Methods "
    "'Known Redshift': USGS-surveyed local g = 9.796 m/s^2 at the JILA "
    "Boulder, CO site."
)

#: Method A: 14-dataset campaign (10 days), corrected gradient (dossier:
#: "Measured, method A (14-dataset campaign, 10 days): ... after
#: budget-level corrections -9.8(2.3)e-20/mm (the headline corrected
#: gradient)"). Units: fractional frequency shift per millimetre.
MEASURED_SLOPE_METHOD_A = PublishedBand(
    nominal=-9.8e-20,
    lo=-9.8e-20 - 2.3e-20,
    hi=-9.8e-20 + 2.3e-20,
    units="fractional/mm",
    citation=(
        "Bothwell et al., Nature 602, 420 (2022) / arXiv:2109.12238, Table 1 "
        "'Known Redshift' row context + main text 'method A' corrected "
        "gradient: -9.8(2.3)e-20/mm (14-dataset campaign, 10 days; raw "
        "-1.00(12)e-19/mm before budget-level corrections)."
    ),
)

#: Method B: synchronous two-region (92 h), corrected gradient (dossier:
#: "Measured, method B (synchronous two-region, 92 h): raw -1.30(18)e-19/mm;
#: corrected -1.28(27)e-19/mm").
MEASURED_SLOPE_METHOD_B = PublishedBand(
    nominal=-1.28e-19,
    lo=-1.28e-19 - 0.27e-19,
    hi=-1.28e-19 + 0.27e-19,
    units="fractional/mm",
    citation=(
        "Bothwell et al., Nature 602, 420 (2022) / arXiv:2109.12238, main "
        "text 'method B' corrected gradient: -1.28(27)e-19/mm (synchronous "
        "two-region, 92 h; raw -1.30(18)e-19/mm before budget-level "
        "corrections)."
    ),
)

#: What Bothwell's published corrected slope already has removed (dossier
#: "What the published slope already has removed"), so the comparison
#: target is the ISOLATED gravitational gradient E36 predicts -- narrative
#: only, not applied as a correction by this script (this script's own
#: field is exactly zero, isolating E36 the same way).
_ISOLATION_NOTE = (
    "The published corrected slopes already have per-pixel density and "
    "second-order-Zeeman corrections applied before the linear fit, plus "
    "budget-level corrections (their Table 1, units 1e-20/mm): BBR 0(0.3), "
    "lattice light -0.5(0.1), DC Stark +0.3(0.2), pixel calibration 0(0.8), "
    "so the corrected slope targets the ISOLATED gravitational gradient "
    "E36 predicts. This script's own field is configured to exactly zero "
    "(field.synthetic.kind='uniform', e0=[0,0,0]) for the same reason: no "
    "DC-Stark/BBR/quadrupole term is active, isolating E36 to match."
)


@dataclass(frozen=True)
class BothwellRedshiftCase:
    """The WP22 Part 3 Bothwell 2022 mm-scale redshift case (see module
    docstring for the full method, the coordinate-sign mapping, and the
    binding classification-labeling rationale -- REPEATED here per the G9
    sign-off's "caveat rides wherever the case appears" instruction).

    **Classification (G9 sign-off B4): "reproducibility", with the
    INVERTED-NPL caveat -- the g/c^2 arithmetic is textbook (Bothwell
    computed it themselves); what is validated is the extended-sample
    MACHINERY producing their measured-map slope end-to-end, with zero
    adjustable inputs.** Does not change the blind-prediction count.

    Every numeric field below is produced by the REAL
    `ensemble.regime='lattice_extended'` pipeline
    (`cliffordclock.pipeline.run_pipeline_full`) -- no hand arithmetic
    feeds `predicted_slope_engine_convention_per_m`/`predicted_slope_per_mm`.

    Attributes
    ----------
    case_class : str
        Always the literal string ``"reproducibility"``.
    case_label : str
        Always :data:`CASE_LABEL`, verbatim.
    n_sites, site_spacing_m, envelope_sigma_m : int, float, float
        The configured site geometry (see module docstring; `site_spacing_m`
        and `envelope_sigma_m` are both flagged INFERRED there).
    g_m_s2, g_citation : float, str
        Bothwell's surveyed local gravity (:data:`BOTHWELL_SURVEYED_G_M_S2`)
        and its citation.
    predicted_slope_engine_convention_per_m : float
        The real pipeline's `LatticeExtendedSiteMap.slope_per_m` -- this
        engine's OWN physical-height sign convention (positive: a clock
        higher along `up_axis = +z` runs faster), meters.
    predicted_slope_per_mm : float
        `predicted_slope_engine_convention_per_m` NEGATED and converted to
        per-millimetre, per the coordinate-sign mapping (module docstring):
        Bothwell's z-axis puts lower positions at larger coordinate, so
        their published gradient sign is the OPPOSITE of this engine's own
        height convention. This is the value actually compared against
        `measured_slope_method_a`/`_b`.
    predicted_slope_windowed_per_mm : float
        A cross-check: the SAME slope, but fit only over sites within
        +/-1.5-sigma of the envelope center (Bothwell's own analysis
        window) rather than the full +/-N_SIGMA_HALF_SPAN-sigma
        computational grid -- computed independently from
        `LatticeExtendedSiteMap.sites`' raw per-site data, not reusing
        `predicted_slope_per_mm`. Expected to agree with it to numerical
        precision (the model is exactly linear in height for this
        field-free, gravity-only case, so the fit is window-independent --
        this field is the check that confirms that directly, rather than
        leaving it assumed).
    measured_slope_method_a, measured_slope_method_b : PublishedBand
        Bothwell's two independent corrected measurements (G9 sign-off B2).
    bands_overlap_method_a, bands_overlap_method_b : bool
        Whether `predicted_slope_per_mm` (a point value -- this is a
        deterministic pipeline evaluation, not a fit to noisy data, so it
        carries no uncertainty of its own at this project's scope) falls
        within each measurement's own published band.
    sigma_distance_method_a, sigma_distance_method_b : float
        ``|predicted - measured.nominal| / measured.uncertainty`` for each
        method (G9 sign-off B2: "0.48-sigma and 0.70-sigma respectively").
    kpi_verdict_method_a, kpi_verdict_method_b : str
        ``"MET"``/``"NOT MET"`` (never ``"PASS"``/``"FAIL"``).
    isolation_note : str
        :data:`_ISOLATION_NOTE`, verbatim.
    dc_stark_context_note : str
        Bothwell's own DC-Stark gradient row context (dossier: "in-scope
        physics for us and worth a narrative line, no more").
    """

    case_class: str
    case_label: str
    n_sites: int
    site_spacing_m: float
    site_spacing_inferred: bool
    envelope_sigma_m: float
    envelope_sigma_inferred: bool
    g_m_s2: float
    g_citation: str
    predicted_slope_engine_convention_per_m: float
    predicted_slope_per_mm: float
    predicted_slope_windowed_per_mm: float
    measured_slope_method_a: dict[str, Any]
    measured_slope_method_b: dict[str, Any]
    bands_overlap_method_a: bool
    bands_overlap_method_b: bool
    sigma_distance_method_a: float
    sigma_distance_method_b: float
    kpi_verdict_method_a: str
    kpi_verdict_method_b: str
    isolation_note: str
    dc_stark_context_note: str


_DC_STARK_CONTEXT_NOTE = (
    "Bothwell's own DC-Stark gradient row, +0.3(0.2)e-20/mm (their Table 1 "
    "budget-level corrections), is in-scope physics for this engine "
    "(CONVENTIONS.md E14b) but is a separate systematic this case does not "
    "model (the comparison target already has it corrected out, see "
    "isolation_note); it enters this report as a narrative cross-reference only."
)


def _windowed_slope_per_mm(
    offsets_m: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    half_sigma: float,
    sigma_m: float,
) -> float:
    """Independent weighted-least-squares fit restricted to
    ``|offset| <= half_sigma * sigma_m`` (Bothwell's own +/-1.5-sigma
    analysis window), computed from raw per-site arrays -- deliberately
    NOT calling `cliffordclock.pipeline._weighted_linear_fit` (an
    independent formula, `numpy.polyfit`, so this is a genuine cross-check
    of the pipeline's own fit, not the same code path re-invoked).

    Returns
    -------
    float
        The windowed fit's slope, converted to per-millimetre and sign-
        mapped into Bothwell's own coordinate convention (negated), same
        as `predicted_slope_per_mm`.
    """
    mask = np.abs(offsets_m) <= half_sigma * sigma_m
    coeffs = np.polyfit(offsets_m[mask], shifts[mask], deg=1, w=weights[mask])
    slope_per_m = float(coeffs[0])
    return -slope_per_m / 1000.0


def run_bothwell_redshift_case() -> BothwellRedshiftCase:
    """Build the WP22 Part 3 Bothwell 2022 case.

    Method:

    1. Configure `ensemble.regime='lattice_extended'` (WP22 Part 2) to the
       dossier's geometry (:data:`N_SITES`/:data:`SITE_SPACING_M`/
       :data:`ENVELOPE_SIGMA_M`), `coupling.type='stark_dc'` with a
       spatially uniform, exactly-ZERO field (isolating E36), and
       `environment.gravity.g_m_s2 = BOTHWELL_SURVEYED_G_M_S2` (G9
       sign-off B1).
    2. Run the REAL pipeline (`cliffordclock.pipeline.run_pipeline_full`)
       and read `LatticeExtendedSiteMap.slope_per_m` -- the actual
       per-site machinery's weighted-least-squares fit, not a shortcut
       formula.
    3. Apply the coordinate-sign mapping (module docstring) to get
       `predicted_slope_per_mm`, and independently cross-check it with a
       +/-1.5-sigma-windowed fit (:func:`_windowed_slope_per_mm`).
    4. Compare against both corrected measurements (G9 sign-off B2) via
       `run_benchmarks._bands_overlap`, treating the (deterministic,
       zero-adjustable-input) prediction as a degenerate point band.

    Returns
    -------
    BothwellRedshiftCase
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            # trap.omega_xyz is irrelevant to the result: motional_n=(0,0,0)
            # with n_quad=1 puts every site's single Hermite-Gauss node
            # exactly at its own site center regardless of trap frequency
            # (the dossier gives no per-site trap frequency for this
            # mm-scale free-space-imaged sample; E29's fast path is exact
            # for static v=0 nodes regardless).
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 0.0]}}},
            "coupling": {"type": "stark_dc"},
            "ensemble": {
                "regime": "lattice_extended",
                "temperature_uK": 1.0,
                "motional_n": [0, 0, 0],
                "n_quad": 1,
                "n_sites": N_SITES,
                "site_spacing_m": SITE_SPACING_M,
                "site_axis": [0.0, 0.0, 1.0],
                "site_envelope": "gaussian",
                "site_envelope_sigma_m": ENVELOPE_SIGMA_M,
            },
            "integration": {"mode": "fast_path", "time_s": 1.0},
            "environment": {
                "gravity": {
                    "g_m_s2": BOTHWELL_SURVEYED_G_M_S2,
                    "up_axis": [0.0, 0.0, 1.0],
                    "reference_height_m": 0.0,
                }
            },
        }
    )
    result = run_pipeline_full(config)
    site_map = result.site_map
    assert site_map is not None, "lattice_extended run unexpectedly produced no site_map"
    assert len(site_map.sites) == N_SITES

    engine_slope_per_m = site_map.slope_per_m
    predicted_per_mm = -engine_slope_per_m / 1000.0  # coordinate-sign mapping, module docstring

    offsets_m = np.array([site.offset_m for site in site_map.sites])
    shifts = np.array([site.mean_fractional_shift for site in site_map.sites])
    weights = np.array([site.weight for site in site_map.sites])
    windowed_per_mm = _windowed_slope_per_mm(offsets_m, shifts, weights, 1.5, ENVELOPE_SIGMA_M)
    assert math.isclose(predicted_per_mm, windowed_per_mm, rel_tol=1e-6), (
        "the full-range and +/-1.5-sigma-windowed fits disagree beyond floating-"
        f"point precision ({predicted_per_mm!r} vs {windowed_per_mm!r}) -- "
        "the field-free gravity-only model should be exactly window-independent "
        "(module docstring); investigate before trusting this case"
    )

    def _compare(measured: PublishedBand) -> tuple[bool, float, str]:
        overlap = _bands_overlap(predicted_per_mm, predicted_per_mm, measured.lo, measured.hi)
        uncertainty = (measured.hi - measured.lo) / 2.0
        sigma_distance = abs(predicted_per_mm - measured.nominal) / uncertainty
        return overlap, sigma_distance, "MET" if overlap else "NOT MET"

    overlap_a, sigma_a, verdict_a = _compare(MEASURED_SLOPE_METHOD_A)
    overlap_b, sigma_b, verdict_b = _compare(MEASURED_SLOPE_METHOD_B)

    return BothwellRedshiftCase(
        case_class="reproducibility",
        case_label=CASE_LABEL,
        n_sites=N_SITES,
        site_spacing_m=SITE_SPACING_M,
        site_spacing_inferred=True,
        envelope_sigma_m=ENVELOPE_SIGMA_M,
        envelope_sigma_inferred=True,
        g_m_s2=BOTHWELL_SURVEYED_G_M_S2,
        g_citation=_BOTHWELL_G_CITATION,
        predicted_slope_engine_convention_per_m=engine_slope_per_m,
        predicted_slope_per_mm=predicted_per_mm,
        predicted_slope_windowed_per_mm=windowed_per_mm,
        measured_slope_method_a=asdict(MEASURED_SLOPE_METHOD_A),
        measured_slope_method_b=asdict(MEASURED_SLOPE_METHOD_B),
        bands_overlap_method_a=overlap_a,
        bands_overlap_method_b=overlap_b,
        sigma_distance_method_a=sigma_a,
        sigma_distance_method_b=sigma_b,
        kpi_verdict_method_a=verdict_a,
        kpi_verdict_method_b=verdict_b,
        isolation_note=_ISOLATION_NOTE,
        dc_stark_context_note=_DC_STARK_CONTEXT_NOTE,
    )


def build_report() -> dict[str, Any]:
    """Build the full WP22 Part 3 Bothwell benchmark report as a
    JSON-serializable dict.

    Returns
    -------
    dict[str, Any]
        Metadata plus the single case (see
        :func:`run_bothwell_redshift_case`). Deliberately NOT merged into
        `run_benchmarks.py`'s WP10 report or `wp10_results.json`'s
        `kpi_summary` (see module docstring's headline-count note).
    """
    case = run_bothwell_redshift_case()
    return {
        "wp22_bothwell_benchmark_schema": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "case_label": CASE_LABEL,
        "case_class": case.case_class,
        "bothwell_2022_nature_602_420_redshift_case": asdict(case),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the WP22 Part 3 Bothwell case as a markdown summary,
    mirroring `run_benchmarks.render_markdown_table`'s style.

    Parameters
    ----------
    report : dict[str, Any]
        A report dict as returned by :func:`build_report`.

    Returns
    -------
    str
        A markdown document suitable for embedding or diffing against
        `benchmarks/RESULTS.md` (once the coordinator integrates
        `benchmarks/_wp22_md_sections_STAGING.md`).
    """
    case = report["bothwell_2022_nature_602_420_redshift_case"]
    lines = [
        "# WP22 Bothwell mm-scale redshift benchmark case (generated)",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Reproducibility case: Bothwell et al., Nature 602, 420 (2022) / arXiv:2109.12238",
        "",
        f"**Classification label (binding, G9 sign-off B4): {CASE_LABEL}**",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Sites (computational grid) | {case['n_sites']} |",
        f"| Site spacing (INFERRED, lambda/2) | {case['site_spacing_m']:.4e} m |",
        f"| Envelope sigma (INFERRED) | {case['envelope_sigma_m']:.4e} m |",
        f"| Reference g (Bothwell surveyed, B1) | {case['g_m_s2']:.4g} m/s^2 |",
        (
            "| Predicted slope (engine's own height convention) | "
            f"{case['predicted_slope_engine_convention_per_m']:+.6e} /m |"
        ),
        (
            "| Predicted slope (Bothwell's coordinate convention, "
            f"sign-mapped) | {case['predicted_slope_per_mm']:+.4e} /mm |"
        ),
        (
            "| Predicted slope (+/-1.5-sigma-windowed cross-check) | "
            f"{case['predicted_slope_windowed_per_mm']:+.4e} /mm |"
        ),
        (
            "| Measured, method A (14-dataset campaign) | "
            f"{case['measured_slope_method_a']['nominal']:+.2e} "
            f"[{case['measured_slope_method_a']['lo']:+.2e}, "
            f"{case['measured_slope_method_a']['hi']:+.2e}] /mm |"
        ),
        (
            "| Measured, method B (synchronous two-region) | "
            f"{case['measured_slope_method_b']['nominal']:+.2e} "
            f"[{case['measured_slope_method_b']['lo']:+.2e}, "
            f"{case['measured_slope_method_b']['hi']:+.2e}] /mm |"
        ),
        f"| Sigma distance, method A | {case['sigma_distance_method_a']:.2f} sigma |",
        f"| Sigma distance, method B | {case['sigma_distance_method_b']:.2f} sigma |",
        f"| **kpi_verdict, method A** | **{case['kpi_verdict_method_a']}** |",
        f"| **kpi_verdict, method B** | **{case['kpi_verdict_method_b']}** |",
        "",
        case["isolation_note"],
        "",
        case["dc_stark_context_note"],
        "",
        "Not counted toward `benchmarks/results/wp10_results.json`'s "
        "`kpi_summary` totals; this is a separate script/report (mirrors "
        "`benchmarks/run_bbr_jila_arithmetic_reproduction.py`'s precedent), "
        "see the module docstring's headline-count note.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the WP22 Part 3 Bothwell benchmark case and write
    `benchmarks/results/bothwell_redshift.json` and a generated markdown
    summary alongside it."""
    report = build_report()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / "bothwell_redshift.json"
    md_path = _RESULTS_DIR / "bothwell_redshift.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"(g/c^2 at standard gravity, for reference: {STANDARD_GRAVITY_OVER_C2!r} /m)")


if __name__ == "__main__":
    main()
