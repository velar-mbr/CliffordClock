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

**WP31 addendum: the participation-corrected variant.** CONVENTIONS.md
section 16's participation-factor extension (`MotionalMode.participation`,
:func:`~cliffordclock.integrator.omega.two_ion_participations`) directly
answers the open item this module's own caveat above names: "a full
two-mass normal-mode treatment... taking each ion's own per-mode
amplitude (participation) vector as an explicit input." This module now
runs a SECOND case,
:func:`run_motional_al_ion_participation_variant_case`, computing Al27+'s
participation in each of Marshall et al.'s six modes from the closed
form at the Al+/Mg25+ mass ratio and comparing PER MODE against the
paper's own published per-mode "Frequency shift per quantum" row
(Table S2), re-fetched and confirmed directly against the primary source
this session (`loaders.MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM`) --
the per-mode published values the module docstring's original caveat
above said were unavailable turned out to exist in Table S2 all along
(the G11 gate record's own "real(1e-19)" column already used them by
hand; this module is the first to load them as re-usable data and run
the actual per-mode comparison programmatically). Result, reported
honestly, not massaged toward agreement: the AXIAL pair (whose full
closed-form eigenvector genuinely depends on the mass ratio alone, per
`two_ion_participations`' own derivation) matches the published per-mode
values to a few percent; the two RADIAL pairs do NOT match well (the
disclosed radial-approximation gap `two_ion_participations`' own
docstring states -- the true radial eigenvector additionally depends on
trap RF/DC geometry parameters this closed form cannot supply from
masses alone). Because two of the six modes (the radial STR pair)
dominate the published total's magnitude, the participation-corrected
TOTAL does not reproduce Marshall's published band as closely as the
original single-mass (`participation=1.0`) total does -- both totals are
reported, each with its own `kpi_verdict`, rather than picking one to
foreground.

**WP32 addendum: the radial-spectrum-reconstructed variant.** CONVENTIONS.md
section 16's radial-spectrum-reconstruction addition
(`cliffordclock.integrator.omega.axial_coulomb_curvature`/
`two_ion_radial_participations`) replaces WP31's radial rows for this
case: instead of reusing the axial mu-only closed form, this module now
runs a THIRD case, :func:`run_motional_al_ion_radial_reconstructed_case`,
inverting the actual measured X and Y mode frequencies against the
Coulomb coupling computed from the axial confinement. The result is
written to its OWN artifact set (`wp32_motional_al_ion_radial_
reconstructed.json`/`.md`, produced by :func:`build_wp32_report`/
:func:`render_wp32_markdown`), leaving the WP30/WP31 artifacts above
completely unchanged. Reported with no tuning: the reconstruction's
per-mode agreement with Marshall's own published per-mode row is in the
same rough range as WP31's approximation, and the reconstructed-
participation total lands at essentially the same total-level deviation
from the published band as WP31's own total, for the structural reason
this module's `MotionalAlIonRadialReconstructedCase.participation_note`
states in full.

**WP33 addendum: the intrinsic-micromotion-enhanced variant, closing the
reconciliation.** WP32's own G14 gate review found the mechanism behind
its residual gap: Marshall's (and Brewer's) published radial rows already
include the shift due to INTRINSIC micromotion, and doubling WP32's plain
participation (the naive "secular + equal-size micromotion" guess) closes
the Y branch but leaves X reproducibly 20-35% short in both independent
datasets -- because the true enhancement is MODE-SPECIFIC,
`F_axis = 1 + q^2/(2*a_axis+q^2)` (Berkeland, J. Appl. Phys. 83, 5025
(1998) Eq. 10), equal to `2` only when the Mathieu `a` parameter vanishes.
CONVENTIONS.md section 16's WP33 addition
(`cliffordclock.integrator.omega.clock_ion_mathieu_parameters`/
`radial_micromotion_enhancement`/`predicted_partner_bare_radial_
frequencies_hz`) solves the clock ion's own leading-order Mathieu
parameters from the trap's published RF drive frequency and the WP32
reconstruction (two equations, two unknowns, zero degrees of freedom),
runs a MANDATORY, falsifiable over-determination check (mass-scaling to
predict the partner ion's own bare radial frequencies, compared against
WP32's separately-reconstructed values), and multiplies WP32's radial
participations by the resulting per-axis `F_x`/`F_y`. This module runs a
FOURTH case, :func:`run_motional_al_ion_intrinsic_micromotion_enhanced_case`,
plus a SECOND, independent consistency check against Brewer et al.'s own
published trap parameters (:func:`run_wp33_brewer_consistency_check`,
different RF drive frequency, different mode frequencies, same species
pair) -- both datasets pass the over-determination check at the
sub-1%-relative level, and the per-mode ratios improve substantially over
WP32's plain participation in both. The result is written to its OWN
artifact set (`wp33_motional_al_ion_intrinsic_micromotion_enhanced.json`/
`.md`, produced by :func:`build_wp33_report`/:func:`render_wp33_markdown`),
leaving the WP30/WP31/WP32 artifacts above completely unchanged.
"""

from __future__ import annotations

import json
import math
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

from cliffordclock.constants import ATOMIC_MASS_UNIT, HBAR, SPEED_OF_LIGHT  # noqa: E402
from cliffordclock.ensemble.species import get_species  # noqa: E402
from cliffordclock.integrator.omega import (  # noqa: E402
    ClockIonMathieuParameters,
    MotionalMode,
    axial_coulomb_curvature,
    clock_ion_mathieu_parameters,
    motional_mean_squared_velocity_m2_s2,
    motional_pivot_perturbation,
    motional_pivot_uncertainty,
    predicted_partner_bare_radial_frequencies_hz,
    radial_micromotion_enhancement,
    two_ion_participations,
    two_ion_radial_participations,
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
    "physics prediction; Marshall et al.'s secular-motion row is "
    "computed from their own measured inputs through the same "
    "standard formula)"
)

#: The two-mass-normal-mode scope-boundary caveat, verbatim, folded into the
#: case's own record so a reader of the JSON/markdown output sees it
#: without having to open this module's docstring. Corrected per the
#: project's G11 gate record (`plan/reviews/G11-e38-motional-time-dilation.md`,
#: section A3): the missing physics is each ion's own per-mode normal-mode
#: amplitude, NOT Marshall's published "geometric factor kappa" (a Doppler-
#: cooling-laser geometry factor from their Eq. 1, unrelated to the
#: secular-motion time-dilation row). WP31 UPDATE: the two-ion partition
#: this caveat originally called an open item is now consumed -- this is
#: THIS case's OWN caveat (`participation=1.0` throughout, by construction,
#: to preserve WP30's single-mass total-level reproduction as its own
#: distinct, independently-verdicted case); see
#: `MotionalAlIonParticipationVariantCase`/
#: `run_motional_al_ion_participation_variant_case` below for the
#: participation-corrected variant this caveat's own "open item" pointed
#: toward, and its own honestly-reported (partial) result.
GEOMETRIC_FACTOR_CAVEAT = (
    "SCOPE CAVEAT (corrected per the project's G11 gate record, section A3): "
    "Marshall et al.'s six modes are TWO-ION (27Al+/25Mg+) crystal normal modes. "
    "The physically complete per-mode evaluation partitions each mode's zero-point "
    "and thermal motion between the two ions by their normal-mode amplitudes. "
    "THIS case (participation=1.0 throughout, the single-species-mass formula) does "
    "not consume that partition, by deliberate construction: it isolates the WP30 "
    "single-mass TOTAL-level reproduction as a case independent of the WP31 "
    "participation-corrected variant reported alongside it below. As a result this "
    "case's per-mode contributions differ from Marshall's per-mode values by "
    "up to several-fold, while summing over the complete six-mode set reproduces "
    "their published TOTAL inside both uncertainty bands (an open empirical "
    "observation about this total-level agreement, not a proven identity; see the "
    "G11 gate record's orthogonality-identity discussion, which is qualitatively "
    "consistent with but does not by itself certify the observed precision). WP31 "
    "(CONVENTIONS.md section 16's participation-factor extension, "
    "`cliffordclock.integrator.omega.two_ion_participations`) now consumes the "
    "two-ion partition; see this report's participation-corrected "
    "variant case for the per-mode and total-level result that closed-form "
    "treatment gives (axial modes match well; radial modes do not, a disclosed, "
    "different scope boundary of THAT closed form, not this one). What remains open "
    "after WP31 is N>2-ion crystals (a numeric normal-mode eigensolver, no closed "
    "form in general) and the RF/micromotion dynamics package (unrelated to "
    "participation)."
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


# ---------------------------------------------------------------------------
# WP31: the participation-corrected variant (see module docstring's
# "WP31 addendum" section).
# ---------------------------------------------------------------------------

#: Mode names, same order as `loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR` /
#: `loaders.MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM` /
#: `two_ion_participations`' own return-tuple order.
_MODE_NAMES: tuple[str, ...] = ("axial_com", "axial_str", "x_com", "x_str", "y_com", "y_str")

#: The two AXIAL modes -- the pair `two_ion_participations`' closed form is
#: exact for (see that function's docstring: the radial pairs reuse the
#: same mu-only formula as a documented approximation, not an additional
#: exact result).
_AXIAL_MODE_NAMES: frozenset[str] = frozenset({"axial_com", "axial_str"})


@dataclass(frozen=True)
class MotionalAlIonModeComparison:
    """One mode's participation-corrected prediction vs. Marshall et al.'s
    own published per-mode "Frequency shift per quantum" value (WP31).

    Attributes
    ----------
    name : str
        Mode name (`_MODE_NAMES` order).
    is_axial : bool
        Whether this is one of the two axial modes (the pair
        `two_ion_participations`'s closed form is EXACT for; the two
        radial pairs reuse the axial formula as a documented
        approximation -- see that function's docstring).
    participation : float
        Al27+'s participation factor in this mode
        (`two_ion_participations(m_Al27, m_Mg25)`).
    predicted_shift_per_quantum : float
        ``-(hbar*omega_i/m_Al)*participation_i/(2*c^2)``: the
        participation-corrected coefficient multiplying `(n_bar_i+1/2)`
        for this mode, dimensionless (the same quantity Table S2's
        "Frequency shift per quantum" row reports).
    published_shift_per_quantum : float
        Marshall et al.'s own published value for this mode
        (`loaders.MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM`).
    residual_fractional : float
        ``predicted_shift_per_quantum - published_shift_per_quantum``.
    ratio_predicted_over_published : float
        ``predicted_shift_per_quantum / published_shift_per_quantum``
        (both negative, so a value near `1.0` is close agreement).
    """

    name: str
    is_axial: bool
    participation: float
    predicted_shift_per_quantum: float
    published_shift_per_quantum: float
    residual_fractional: float
    ratio_predicted_over_published: float


@dataclass(frozen=True)
class MotionalAlIonParticipationVariantCase:
    """The WP31 participation-corrected variant of the WP30 Al+
    secular-motion case (see module docstring's "WP31 addendum" for the
    full method and its honestly-reported result).

    Attributes
    ----------
    case_class : str
        Always ``"arithmetic_reproduction"`` (same structural class as
        the single-mass variant -- see that case's own `case_class` note).
    partner_species_name : str
        Always ``"Mg25+"`` (not a registered `Species`; mass supplied
        directly from `loaders.MG25_ATOMIC_MASS_AMU`, see that constant's
        docstring for the source).
    per_mode : tuple[MotionalAlIonModeComparison, ...]
        One entry per mode, `_MODE_NAMES` order.
    axial_mean_abs_ratio_deviation : float
        Mean, over the two AXIAL modes only, of
        ``abs(ratio_predicted_over_published - 1.0)`` -- how close the
        exact (mu-only) closed form comes to Marshall's own published
        axial per-mode values.
    radial_mean_abs_ratio_deviation : float
        Same statistic over the four RADIAL modes -- expected to be much
        larger, the disclosed radial-approximation gap.
    predicted_total_nominal : float
        The participation-corrected TOTAL `(P-1)_motional`
        (`motional_pivot_perturbation` with every mode's
        `participation` set from `two_ion_participations`), summed over
        all six modes with `(n_bar_i+1/2)`.
    predicted_total_uncertainty_fractional : float
        Propagated 1-sigma uncertainty on `predicted_total_nominal`
        (`motional_pivot_uncertainty`, same n_bar uncertainties as the
        single-mass variant; participation itself carries no uncertainty
        channel, see `MotionalMode.participation`'s docstring).
    predicted_total_band_lo, _hi : float
        ``predicted_total_nominal +/- predicted_total_uncertainty_fractional``.
    total_bands_overlap : bool
        Whether the participation-corrected total's band overlaps
        Marshall's own published band
        (`loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT`).
    total_kpi_verdict : str
        ``"MET"`` if `total_bands_overlap` else ``"NOT MET"``.
    per_mode_citation, participation_note : str
        Citation for the per-mode published data, and a plain-language
        summary of the axial-matches/radial-does-not-match result.
    """

    case_class: str
    partner_species_name: str
    per_mode: tuple[MotionalAlIonModeComparison, ...]
    axial_mean_abs_ratio_deviation: float
    radial_mean_abs_ratio_deviation: float
    predicted_total_nominal: float
    predicted_total_uncertainty_fractional: float
    predicted_total_band_lo: float
    predicted_total_band_hi: float
    total_bands_overlap: bool
    total_kpi_verdict: str
    per_mode_citation: str
    participation_note: str


#: Plain-language summary of the per-mode comparison result, folded into
#: the case record (module docstring's "WP31 addendum" states the same
#: finding in full; this is the short form for the report/markdown).
_PARTICIPATION_NOTE = (
    "Participation-corrected per-mode comparison against Marshall et al.'s own published "
    "'Frequency shift per quantum' row (Table S2): the two AXIAL modes, where "
    "two_ion_participations' closed form is exact (a function of the Al+/Mg25+ mass ratio "
    "alone), match the published per-mode values to a few percent, a substantial "
    "improvement over the single-mass (participation=1.0) variant's ~2x per-mode "
    "disagreement there. The four RADIAL modes do NOT match well: the true radial "
    "eigenvector additionally depends on trap RF/DC geometry parameters "
    "(two_ion_participations' own documented scope caveat) this closed form cannot supply "
    "from masses alone. Because the radial STR pair carries the largest published "
    "per-mode magnitudes, the participation-corrected TOTAL does not reproduce Marshall's "
    "published band as closely as the single-mass total does; both totals are reported "
    "with their own kpi_verdict, not merged into one number."
)


def run_motional_al_ion_participation_variant_case() -> MotionalAlIonParticipationVariantCase:
    """Build the WP31 participation-corrected variant (see module
    docstring's "WP31 addendum" for the full method).

    1. Resolve Al27+'s registry mass and Mg25+'s raw mass
       (`loaders.MG25_ATOMIC_MASS_AMU`, not a registered `Species`).
    2. `two_ion_participations(m_Al27, m_Mg25)` -> six participation
       factors, `_MODE_NAMES` order.
    3. Build six `MotionalMode` entries (same frequencies/n_bar as the
       single-mass variant) with `participation` set from step 2.
    4. Per-mode: compute the participation-corrected coefficient
       ``-(hbar*omega_i/m_Al)*participation_i/(2*c^2)`` directly (real
       engine constants, not re-derived) and compare against
       `loaders.MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM`.
    5. Total: `motional_pivot_perturbation`/`motional_pivot_uncertainty`
       over the six participation-weighted modes, compared against
       `loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT` via
       `run_benchmarks._bands_overlap`.

    Returns
    -------
    MotionalAlIonParticipationVariantCase
    """
    species = get_species("Al27+")
    m_al = species.mass_kg
    m_mg = loaders.MG25_ATOMIC_MASS_AMU * ATOMIC_MASS_UNIT
    participations = two_ion_participations(m_al, m_mg)

    modes = tuple(
        MotionalMode(
            name=name,
            frequency_hz=frequency_mhz * 1.0e6,
            n_bar=n_bar,
            n_bar_uncertainty=n_bar_uncertainty,
            participation=participation,
        )
        for (name, frequency_mhz, n_bar, n_bar_uncertainty), participation in zip(
            loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR, participations, strict=True
        )
    )

    per_mode = []
    axial_devs = []
    radial_devs = []
    for mode, participation, published_pq in zip(
        modes, participations, loaders.MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM, strict=True
    ):
        omega_i = 2.0 * math.pi * mode.frequency_hz
        predicted_pq = -(HBAR * omega_i / m_al) * participation / (2.0 * SPEED_OF_LIGHT**2)
        residual = predicted_pq - published_pq
        ratio = predicted_pq / published_pq
        is_axial = mode.name in _AXIAL_MODE_NAMES
        per_mode.append(
            MotionalAlIonModeComparison(
                name=mode.name,
                is_axial=is_axial,
                participation=participation,
                predicted_shift_per_quantum=predicted_pq,
                published_shift_per_quantum=published_pq,
                residual_fractional=residual,
                ratio_predicted_over_published=ratio,
            )
        )
        (axial_devs if is_axial else radial_devs).append(abs(ratio - 1.0))

    predicted_total = motional_pivot_perturbation(modes, species)
    predicted_sigma = motional_pivot_uncertainty(modes, species)
    published = loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT
    band_lo = predicted_total - predicted_sigma
    band_hi = predicted_total + predicted_sigma
    overlap = run_benchmarks._bands_overlap(  # noqa: SLF001 (reusing the tested helper)
        band_lo, band_hi, published.lo, published.hi
    )

    return MotionalAlIonParticipationVariantCase(
        case_class="arithmetic_reproduction",
        partner_species_name="Mg25+",
        per_mode=tuple(per_mode),
        axial_mean_abs_ratio_deviation=math.fsum(axial_devs) / len(axial_devs),
        radial_mean_abs_ratio_deviation=math.fsum(radial_devs) / len(radial_devs),
        predicted_total_nominal=predicted_total,
        predicted_total_uncertainty_fractional=predicted_sigma,
        predicted_total_band_lo=band_lo,
        predicted_total_band_hi=band_hi,
        total_bands_overlap=overlap,
        total_kpi_verdict="MET" if overlap else "NOT MET",
        per_mode_citation=loaders.MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM_CITATION,
        participation_note=_PARTICIPATION_NOTE,
    )


# ---------------------------------------------------------------------------
# WP32: the radial-spectrum-reconstructed variant (see this module's own
# module-level docstring addendum below `run_motional_al_ion_participation_
# variant_case`'s docstring, and CONVENTIONS.md section 16's WP32 addition,
# for the method). Replaces WP31's radial rows (the axial mu-only closed
# form applied to the radial pairs as a documented approximation) with
# `two_ion_radial_participations`'s genuine inversion of the MEASURED
# radial spectrum -- WP30's and WP31's own cases and artifacts are left
# completely untouched by this section; this is a third, additional case.
# ---------------------------------------------------------------------------

#: WP32's own caveat, replacing WP31's radial-scope framing for the
#: reconstructed rows specifically (kept separate from `GEOMETRIC_FACTOR_
#: CAVEAT` above, which documents WP30's single-mass case and stays
#: unmodified -- WP31's own approximation stays available too, labeled, in
#: `run_motional_al_ion_participation_variant_case`'s own untouched output).
RADIAL_RECONSTRUCTION_CAVEAT = (
    "WP32 SCOPE NOTE: this case's X/Y participations are NOT the axial mu-only closed form "
    "reused for radial (WP31's documented approximation, still reported unmodified in the "
    "participation-corrected variant case above); they are reconstructed from "
    "Marshall et al.'s own measured axial-COM and radial mode frequencies "
    "(cliffordclock.integrator.omega.axial_coulomb_curvature/two_ion_radial_participations), "
    "inverting the two-ion radial eigenproblem for each transverse direction's two unknown "
    "bare radial frequencies with no trap RF/DC geometry parameter (epsilon, alpha) as input. "
    "The disambiguation assumption (RF pseudopotential scaling: the lighter ion, Mg25+, "
    "carries the higher bare radial frequency) is applied identically to both the X and Y "
    "branches. Reported per-mode and total-level agreement below is whatever this "
    "reconstruction gives against Marshall's own published per-mode and total rows; the only "
    "inputs are the measured mode frequencies named above. See this case's own "
    "participation_note for the result stated in full."
)


@dataclass(frozen=True)
class MotionalAlIonRadialReconstructedCase:
    """The WP32 radial-spectrum-reconstructed variant of the WP30/WP31 Al+
    secular-motion case.

    Attributes
    ----------
    case_class : str
        Always ``"arithmetic_reproduction"`` (same structural class as the
        other two variants).
    coulomb_curvature_n_per_m : float
        ``c`` (N/m), recovered from Marshall's own axial-COM mode
        frequency via :func:`~cliffordclock.integrator.omega.axial_coulomb_curvature`.
    coulomb_curvature_cross_check_n_per_m : float
        The SAME `c`, independently recovered from the axial-STR mode
        frequency using Wubbena Eq. 13 in place of Eq. 12, a consistency
        check on the shared-`k_z` assumption, not consumed by anything
        downstream in this case.
    coulomb_curvature_cross_check_relative_deviation : float
        ``(coulomb_curvature_cross_check_n_per_m - coulomb_curvature_n_per_m)
        / coulomb_curvature_n_per_m``.
    per_mode : tuple[MotionalAlIonModeComparison, ...]
        One entry per mode, `_MODE_NAMES` order: the two axial entries
        reuse `two_ion_participations`'s exact closed form unmodified (the
        same values the WP31 variant case reports); the four radial
        entries use `two_ion_radial_participations`'s reconstructed
        participations instead of WP31's axial-form approximation.
    bare_frequency_clock_x_hz, bare_frequency_partner_x_hz : float
        The reconstructed bare (single-ion) radial frequencies for the X
        branch, hertz.
    bare_frequency_clock_y_hz, bare_frequency_partner_y_hz : float
        Same, for the Y branch.
    predicted_total_nominal : float
        The reconstructed-participation TOTAL `(P-1)_motional`, summed
        over all six modes with `(n_bar_i+1/2)`.
    predicted_total_uncertainty_fractional : float
        Propagated 1-sigma uncertainty on `predicted_total_nominal`
        (`motional_pivot_uncertainty`, same n_bar uncertainties as the
        other two variants; the reconstructed participations' own
        uncertainty is `0.0` here since Table S2 publishes no per-mode
        frequency uncertainty to propagate through the inversion -- see
        `two_ion_radial_participations`'s own uncertainty parameters for
        where a lab-supplied frequency uncertainty would enter).
    predicted_total_band_lo, _hi : float
        ``predicted_total_nominal +/- predicted_total_uncertainty_fractional``.
    total_bands_overlap : bool
        Whether the reconstructed-participation total's band overlaps
        Marshall's own published band.
    total_kpi_verdict : str
        ``"MET"`` if `total_bands_overlap` else ``"NOT MET"``.
    per_mode_citation, radial_reconstruction_caveat, participation_note : str
        Citation for the per-mode published data, :data:`RADIAL_RECONSTRUCTION_CAVEAT`
        verbatim, and a plain-language summary of this case's own result.
    """

    case_class: str
    coulomb_curvature_n_per_m: float
    coulomb_curvature_cross_check_n_per_m: float
    coulomb_curvature_cross_check_relative_deviation: float
    per_mode: tuple[MotionalAlIonModeComparison, ...]
    bare_frequency_clock_x_hz: float
    bare_frequency_partner_x_hz: float
    bare_frequency_clock_y_hz: float
    bare_frequency_partner_y_hz: float
    predicted_total_nominal: float
    predicted_total_uncertainty_fractional: float
    predicted_total_band_lo: float
    predicted_total_band_hi: float
    total_bands_overlap: bool
    total_kpi_verdict: str
    per_mode_citation: str
    radial_reconstruction_caveat: str
    participation_note: str


def run_motional_al_ion_radial_reconstructed_case() -> MotionalAlIonRadialReconstructedCase:
    """Build the WP32 radial-spectrum-reconstructed variant (see this
    module's WP32 section header comment for the method).

    1. Resolve Al27+'s registry mass and Mg25+'s raw mass, exactly as
       :func:`run_motional_al_ion_participation_variant_case`.
    2. `axial_coulomb_curvature(m_Al27, m_Mg25, axial_com_frequency_hz)` ->
       Coulomb curvature `c`, from Marshall's own axial-COM mode
       frequency; independently cross-checked (not consumed downstream)
       against the axial-STR mode via the same function.
    3. `two_ion_radial_participations(m_Al27, m_Mg25, c, x_com_hz, x_str_hz)`
       and the same call for the Y branch -> reconstructed X/Y
       participations and bare radial frequencies.
    4. The two axial modes reuse `two_ion_participations`'s unmodified
       closed form (unchanged from WP31; exact for axial).
    5. Per-mode: compute the participation-corrected coefficient exactly
       as the WP31 variant does, now with the reconstructed radial
       participations, and compare against
       `loaders.MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM`.
    6. Total: `motional_pivot_perturbation`/`motional_pivot_uncertainty`
       over the six reconstructed-participation modes, compared against
       `loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT`.

    Returns
    -------
    MotionalAlIonRadialReconstructedCase
    """
    species = get_species("Al27+")
    m_al = species.mass_kg
    m_mg = loaders.MG25_ATOMIC_MASS_AMU * ATOMIC_MASS_UNIT

    axial_com_hz = loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[0][1] * 1.0e6
    axial_str_hz = loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[1][1] * 1.0e6
    c, _c_unc = axial_coulomb_curvature(m_al, m_mg, axial_com_hz)
    # Cross-check: the SAME c, independently recovered from the axial-STR
    # mode via Wubbena Eq. 13 instead of Eq. 12. axial_coulomb_curvature
    # only implements the Eq. 12 (COM) path, so the Eq. 13 (STR) path is
    # reproduced directly here from the same closed form
    # (two_ion_participations' own mu/root, not re-derived).
    mu = m_mg / m_al
    root = math.sqrt(1.0 - mu + mu * mu)
    omega_str = 2.0 * math.pi * axial_str_hz
    omega_z1_from_str = omega_str / math.sqrt((1.0 + mu + root) / mu)
    c_cross_check = (m_al * omega_z1_from_str * omega_z1_from_str) / 2.0
    c_cross_check_deviation = (c_cross_check - c) / c

    axial_participations = two_ion_participations(m_al, m_mg)
    x_com_hz = loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[2][1] * 1.0e6
    x_str_hz = loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[3][1] * 1.0e6
    y_com_hz = loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[4][1] * 1.0e6
    y_str_hz = loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[5][1] * 1.0e6
    x_result = two_ion_radial_participations(m_al, m_mg, c, x_com_hz, x_str_hz)
    y_result = two_ion_radial_participations(m_al, m_mg, c, y_com_hz, y_str_hz)

    reconstructed_participations = (
        axial_participations[0],
        axial_participations[1],
        x_result.com_participation,
        x_result.str_participation,
        y_result.com_participation,
        y_result.str_participation,
    )

    modes = tuple(
        MotionalMode(
            name=name,
            frequency_hz=frequency_mhz * 1.0e6,
            n_bar=n_bar,
            n_bar_uncertainty=n_bar_uncertainty,
            participation=participation,
        )
        for (name, frequency_mhz, n_bar, n_bar_uncertainty), participation in zip(
            loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR, reconstructed_participations, strict=True
        )
    )

    per_mode = []
    for mode, participation, published_pq in zip(
        modes,
        reconstructed_participations,
        loaders.MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM,
        strict=True,
    ):
        omega_i = 2.0 * math.pi * mode.frequency_hz
        predicted_pq = -(HBAR * omega_i / m_al) * participation / (2.0 * SPEED_OF_LIGHT**2)
        residual = predicted_pq - published_pq
        ratio = predicted_pq / published_pq
        is_axial = mode.name in _AXIAL_MODE_NAMES
        per_mode.append(
            MotionalAlIonModeComparison(
                name=mode.name,
                is_axial=is_axial,
                participation=participation,
                predicted_shift_per_quantum=predicted_pq,
                published_shift_per_quantum=published_pq,
                residual_fractional=residual,
                ratio_predicted_over_published=ratio,
            )
        )

    predicted_total = motional_pivot_perturbation(modes, species)
    predicted_sigma = motional_pivot_uncertainty(modes, species)
    published = loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT
    band_lo = predicted_total - predicted_sigma
    band_hi = predicted_total + predicted_sigma
    overlap = run_benchmarks._bands_overlap(  # noqa: SLF001 (reusing the tested helper)
        band_lo, band_hi, published.lo, published.hi
    )
    combined_sigma = math.sqrt(predicted_sigma**2 + (published.hi - published.nominal) ** 2)
    deviation_sigma = abs(predicted_total - published.nominal) / combined_sigma

    participation_note = (
        "Radial-spectrum-reconstructed per-mode comparison against Marshall et al.'s own "
        "published 'Frequency shift per quantum' row (Table S2): the two AXIAL modes are "
        "unchanged from the WP31 variant (two_ion_participations' exact mu-only closed form). "
        "The four RADIAL modes now use two_ion_radial_participations' reconstruction from the "
        "measured X/Y spectra instead of the axial-form approximation; the resulting per-mode "
        "ratios (predicted/published) sit in the same rough range as WP31's radial rows, "
        f"landing at {deviation_sigma:.2f} sigma from the published total "
        f"({'MET' if overlap else 'NOT MET'}), essentially unchanged from WP31's own "
        "radial-approximation total. Because each mode-pair's clock-ion participations sum to "
        "1.0 regardless of how the pair's total is split between its COM and STR "
        "members, and Marshall's own COM/STR (n_bar+1/2)-weighted magnitudes for a given "
        "branch are comparable in size, redistributing participation within a radial pair "
        "moves the per-mode ratios without moving the pair's own total much, a structural "
        "reason a correctly reconstructed split need not by itself close a total-level gap "
        "this size. The reconstruction, its cross-check, and this result are reported as run."
    )

    return MotionalAlIonRadialReconstructedCase(
        case_class="arithmetic_reproduction",
        coulomb_curvature_n_per_m=c,
        coulomb_curvature_cross_check_n_per_m=c_cross_check,
        coulomb_curvature_cross_check_relative_deviation=c_cross_check_deviation,
        per_mode=tuple(per_mode),
        bare_frequency_clock_x_hz=x_result.bare_frequency_clock_hz,
        bare_frequency_partner_x_hz=x_result.bare_frequency_partner_hz,
        bare_frequency_clock_y_hz=y_result.bare_frequency_clock_hz,
        bare_frequency_partner_y_hz=y_result.bare_frequency_partner_hz,
        predicted_total_nominal=predicted_total,
        predicted_total_uncertainty_fractional=predicted_sigma,
        predicted_total_band_lo=band_lo,
        predicted_total_band_hi=band_hi,
        total_bands_overlap=overlap,
        total_kpi_verdict="MET" if overlap else "NOT MET",
        per_mode_citation=loaders.MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM_CITATION,
        radial_reconstruction_caveat=RADIAL_RECONSTRUCTION_CAVEAT,
        participation_note=participation_note,
    )


# ---------------------------------------------------------------------------
# WP33: mode-specific intrinsic-micromotion enhancement for radial secular
# modes (CONVENTIONS.md section 16's WP33 addition;
# `cliffordclock.integrator.omega.clock_ion_mathieu_parameters`/
# `radial_micromotion_enhancement`/`predicted_partner_bare_radial_
# frequencies_hz`). Closes the reconciliation the G14 gate review
# identified: Marshall's (and, per Brewer's own footnote, Brewer's)
# published radial rows already include the shift due to INTRINSIC
# micromotion, and WP32's plain participation ratios (0.35x-0.52x) --
# even doubled (WP32's own G14-reported 0.7x-1.06x, Y closing but X
# reproducibly 20-35% short) -- used a uniform factor of 2 where the true
# enhancement is MODE-SPECIFIC (`F_axis = 1 + q^2/(2*a_axis+q^2)`,
# Berkeland Eq. 10, `omega.py`'s own WP33 comment block). WP30's/WP31's/
# WP32's own cases and artifacts are left completely untouched by this
# section; this is a fourth, additional case, plus a second, independent
# consistency check against Brewer et al.'s own published trap parameters.
# ---------------------------------------------------------------------------

#: WP33's own caveat, replacing WP32's "essentially unchanged from WP31"
#: framing now that the mode-specific enhancement is available.
INTRINSIC_MICROMOTION_ENHANCEMENT_CAVEAT = (
    "WP33 SCOPE NOTE: this case multiplies WP32's reconstructed radial participations by "
    "the leading-order intrinsic-micromotion enhancement factor F_axis = 1 + q^2/(2*a_axis+q^2) "
    "(Berkeland, Miller, Bergquist, Itano, Wineland, J. Appl. Phys. 83, 5025 (1998) Eq. 10), "
    "with (q, a_x, a_y) the clock (Al27+) ion's own leading-order Mathieu parameters solved "
    "from the trap's published RF drive frequency, the axial Coulomb curvature (WP32), and "
    "the two WP32-reconstructed clock-ion bare radial frequencies "
    "(cliffordclock.integrator.omega.clock_ion_mathieu_parameters): two equations, two "
    "unknowns, zero degrees of freedom, no trap-geometry parameter (alpha/epsilon) supplied "
    "as input. The axial modes are unchanged from WP31/WP32 (q_z=0, F_axial=1 identically: no "
    "intrinsic micromotion along the trap axis). Before use, the reconstruction chain is "
    "checked by an independent, falsifiable test: the clock ion's solved Mathieu parameters "
    "are mass-scaled to predict the PARTNER (Mg25+) ion's own bare radial frequencies, compared "
    "against WP32's separately-reconstructed partner frequencies (from the two-ion eigenproblem "
    "inversion, an entirely different calculation); see this case's own "
    "partner_prediction_note for the result. Reported per-mode and total-level agreement below "
    "is whatever this reconstruction gives against Marshall's own published per-mode and total "
    "rows; see this case's own enhancement_note for the result stated in full."
)


@dataclass(frozen=True)
class MotionalAlIonEnhancedModeComparison:
    """One mode's participation-AND-enhancement-corrected prediction vs.
    Marshall et al.'s own published per-mode "Frequency shift per quantum"
    value (WP33).

    Attributes
    ----------
    name : str
        Mode name (`_MODE_NAMES` order).
    is_axial : bool
        Whether this is one of the two axial modes (`enhancement == 1.0`
        identically for these; see class docstring).
    participation : float
        The clock ion's participation factor in this mode (WP31's exact
        closed form for the axial pair; WP32's reconstruction for the
        radial pairs -- unchanged from `MotionalAlIonRadialReconstructedCase`).
    enhancement : float
        `F_axis` (WP33, `radial_micromotion_enhancement`): `1.0` for the
        two axial modes, `F_x`/`F_y` for the four radial modes.
    predicted_shift_per_quantum : float
        ``-(hbar*omega_i/m_Al)*participation_i*enhancement_i/(2*c^2)``:
        the participation-AND-enhancement-corrected coefficient
        multiplying `(n_bar_i+1/2)` for this mode, dimensionless.
    published_shift_per_quantum : float
        Marshall et al.'s own published value for this mode.
    residual_fractional, ratio_predicted_over_published : float
        Same definitions as `MotionalAlIonModeComparison`.
    """

    name: str
    is_axial: bool
    participation: float
    enhancement: float
    predicted_shift_per_quantum: float
    published_shift_per_quantum: float
    residual_fractional: float
    ratio_predicted_over_published: float


@dataclass(frozen=True)
class MotionalAlIonIntrinsicMicromotionEnhancedCase:
    """The WP33 intrinsic-micromotion-enhanced variant of the WP30/WP31/WP32
    Al+ secular-motion case.

    Attributes
    ----------
    case_class : str
        Always ``"arithmetic_reproduction"``.
    clock_mathieu : dict[str, float]
        The clock ion's solved Mathieu parameters
        (`asdict(cliffordclock.integrator.omega.ClockIonMathieuParameters)`):
        `mathieu_q`, `mathieu_a_x`, `mathieu_a_y`, `mathieu_a_z`
        (dimensionless) plus their (here all `0.0`; Marshall publishes no
        per-mode frequency uncertainty for the axial/radial rows)
        propagated uncertainties.
    enhancement_x, enhancement_y : float
        `F_x`, `F_y` (`radial_micromotion_enhancement`).
    bare_frequency_partner_x_predicted_hz, bare_frequency_partner_y_predicted_hz : float
        The PARTNER ion's bare radial X/Y frequencies, PREDICTED by
        mass-scaling the clock ion's own solved Mathieu parameters
        (`predicted_partner_bare_radial_frequencies_hz`) -- the
        over-determination check's prediction.
    bare_frequency_partner_x_reconstructed_hz, bare_frequency_partner_y_reconstructed_hz : float
        The SAME partner frequencies, from WP32's own SEPARATE two-ion
        eigenproblem inversion (`two_ion_radial_participations`'s
        `bare_frequency_partner_hz`) -- the over-determination check's
        independent target.
    partner_x_relative_deviation, partner_y_relative_deviation : float
        ``(predicted - reconstructed) / reconstructed`` for each branch:
        the over-determination check's own falsifiable result.
    per_mode : tuple[MotionalAlIonEnhancedModeComparison, ...]
        One entry per mode, `_MODE_NAMES` order.
    predicted_total_nominal, predicted_total_uncertainty_fractional,
    predicted_total_band_lo, predicted_total_band_hi : float
        The enhancement-corrected TOTAL `(P-1)_motional`, its propagated
        1-sigma uncertainty (from the same per-mode `n_bar` uncertainties
        as WP30/31/32; `participation*enhancement` is treated as an exact
        input here, exactly like WP31/32's own `participation`), and its
        `+/-1-sigma` band.
    total_bands_overlap : bool
        Whether this band overlaps Marshall et al.'s own published band.
    total_kpi_verdict : str
        ``"MET"`` if `total_bands_overlap` else ``"NOT MET"``.
    per_mode_citation, enhancement_caveat, enhancement_note,
    partner_prediction_note : str
        Citations plus plain-language summaries of this case's own
        results, stated in full.
    """

    case_class: str
    clock_mathieu: dict[str, float]
    enhancement_x: float
    enhancement_y: float
    bare_frequency_partner_x_predicted_hz: float
    bare_frequency_partner_y_predicted_hz: float
    bare_frequency_partner_x_reconstructed_hz: float
    bare_frequency_partner_y_reconstructed_hz: float
    partner_x_relative_deviation: float
    partner_y_relative_deviation: float
    per_mode: tuple[MotionalAlIonEnhancedModeComparison, ...]
    predicted_total_nominal: float
    predicted_total_uncertainty_fractional: float
    predicted_total_band_lo: float
    predicted_total_band_hi: float
    total_bands_overlap: bool
    total_kpi_verdict: str
    per_mode_citation: str
    enhancement_caveat: str
    enhancement_note: str
    partner_prediction_note: str


def run_motional_al_ion_intrinsic_micromotion_enhanced_case() -> (
    MotionalAlIonIntrinsicMicromotionEnhancedCase
):
    """Build the WP33 intrinsic-micromotion-enhanced variant (this
    module's WP33 section header comment for the method).

    1. Reconstruct the radial spectrum exactly as
       :func:`run_motional_al_ion_radial_reconstructed_case` (WP32): the
       Coulomb curvature, the X/Y branch inversions, the clock-ion bare
       radial frequencies.
    2. `clock_ion_mathieu_parameters(m_Al27, c, rf_drive_frequency_hz,
       bare_frequency_clock_x_hz, bare_frequency_clock_y_hz)` -> the
       clock ion's own `(q, a_x, a_y, a_z)`.
    3. `predicted_partner_bare_radial_frequencies_hz` -> the
       over-determination check: mass-scale to the partner ion and
       compare against WP32's own SEPARATELY reconstructed partner
       frequencies.
    4. `radial_micromotion_enhancement(q, a_x)` / `(q, a_y)` -> `F_x`,
       `F_y`. Axial modes get `F_axial = 1.0` (no engine call needed:
       `q_z = 0` makes the formula's numerator vanish identically).
    5. Per-mode: `participation_i * enhancement_i` in place of WP31/32's
       plain `participation_i`, compared against
       `loaders.MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM`.
    6. Total: `sum_i (hbar*omega_i/m_Al)*participation_i*enhancement_i*
       (n_bar_i+1/2)`, `(P-1)_motional = -that/(2*c^2)`, uncertainty
       propagated from the same per-mode `n_bar` uncertainties WP30/31/32
       already use (`participation*enhancement` treated as an exact
       input, mirroring `MotionalMode.participation`'s own convention).

    Returns
    -------
    MotionalAlIonIntrinsicMicromotionEnhancedCase
    """
    species = get_species("Al27+")
    m_al = species.mass_kg
    m_mg = loaders.MG25_ATOMIC_MASS_AMU * ATOMIC_MASS_UNIT
    rf_drive_frequency_hz = loaders.MARSHALL_AL_ION_RF_DRIVE_FREQUENCY_HZ

    axial_com_hz = loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[0][1] * 1.0e6
    c, _c_unc = axial_coulomb_curvature(m_al, m_mg, axial_com_hz)

    axial_participations = two_ion_participations(m_al, m_mg)
    x_com_hz = loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[2][1] * 1.0e6
    x_str_hz = loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[3][1] * 1.0e6
    y_com_hz = loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[4][1] * 1.0e6
    y_str_hz = loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR[5][1] * 1.0e6
    x_result = two_ion_radial_participations(m_al, m_mg, c, x_com_hz, x_str_hz)
    y_result = two_ion_radial_participations(m_al, m_mg, c, y_com_hz, y_str_hz)

    clock_mathieu: ClockIonMathieuParameters = clock_ion_mathieu_parameters(
        m_al,
        c,
        rf_drive_frequency_hz,
        x_result.bare_frequency_clock_hz,
        y_result.bare_frequency_clock_hz,
    )

    predicted_partner_x_hz, predicted_partner_y_hz = predicted_partner_bare_radial_frequencies_hz(
        clock_mathieu, m_al, m_mg, rf_drive_frequency_hz
    )
    partner_x_relative_deviation = (
        predicted_partner_x_hz - x_result.bare_frequency_partner_hz
    ) / x_result.bare_frequency_partner_hz
    partner_y_relative_deviation = (
        predicted_partner_y_hz - y_result.bare_frequency_partner_hz
    ) / y_result.bare_frequency_partner_hz

    enhancement_x = radial_micromotion_enhancement(
        clock_mathieu.mathieu_q, clock_mathieu.mathieu_a_x
    )
    enhancement_y = radial_micromotion_enhancement(
        clock_mathieu.mathieu_q, clock_mathieu.mathieu_a_y
    )

    reconstructed_participations = (
        axial_participations[0],
        axial_participations[1],
        x_result.com_participation,
        x_result.str_participation,
        y_result.com_participation,
        y_result.str_participation,
    )
    enhancements = (1.0, 1.0, enhancement_x, enhancement_x, enhancement_y, enhancement_y)

    per_mode = []
    v2_terms = []
    d_dn_bar_terms = []
    for (
        name,
        frequency_mhz,
        n_bar,
        n_bar_uncertainty,
    ), participation, enhancement, published_pq in zip(
        loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR,
        reconstructed_participations,
        enhancements,
        loaders.MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM,
        strict=True,
    ):
        frequency_hz = frequency_mhz * 1.0e6
        omega_i = 2.0 * math.pi * frequency_hz
        weight = (HBAR * omega_i / m_al) * participation * enhancement
        predicted_pq = -weight / (2.0 * SPEED_OF_LIGHT**2)
        residual = predicted_pq - published_pq
        ratio = predicted_pq / published_pq
        is_axial = name in _AXIAL_MODE_NAMES
        per_mode.append(
            MotionalAlIonEnhancedModeComparison(
                name=name,
                is_axial=is_axial,
                participation=participation,
                enhancement=enhancement,
                predicted_shift_per_quantum=predicted_pq,
                published_shift_per_quantum=published_pq,
                residual_fractional=residual,
                ratio_predicted_over_published=ratio,
            )
        )
        v2_terms.append(weight * (n_bar + 0.5))
        d_dn_bar_terms.append((-weight / (2.0 * SPEED_OF_LIGHT**2)) * n_bar_uncertainty)

    mean_v2 = math.fsum(v2_terms)
    predicted_total = -mean_v2 / (2.0 * SPEED_OF_LIGHT**2)
    predicted_sigma = math.sqrt(math.fsum(t * t for t in d_dn_bar_terms))

    published = loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT
    band_lo = predicted_total - predicted_sigma
    band_hi = predicted_total + predicted_sigma
    overlap = run_benchmarks._bands_overlap(  # noqa: SLF001 (reusing the tested helper)
        band_lo, band_hi, published.lo, published.hi
    )
    combined_sigma = math.sqrt(predicted_sigma**2 + (published.hi - published.nominal) ** 2)
    deviation_sigma = abs(predicted_total - published.nominal) / combined_sigma

    partner_prediction_note = (
        "Over-determination check: mass-scaling the clock ion's own solved Mathieu parameters "
        f"(q={clock_mathieu.mathieu_q:.6f}, a_x={clock_mathieu.mathieu_a_x:+.6e}, "
        f"a_y={clock_mathieu.mathieu_a_y:+.6e}, a_z={clock_mathieu.mathieu_a_z:+.6e}) to the "
        f"partner ion (Mg25+) predicts bare radial frequencies of {predicted_partner_x_hz:.6e} Hz "
        f"(X) and {predicted_partner_y_hz:.6e} Hz (Y), against WP32's own SEPARATELY "
        f"reconstructed {x_result.bare_frequency_partner_hz:.6e} Hz (X) and "
        f"{y_result.bare_frequency_partner_hz:.6e} Hz (Y); relative deviations "
        f"{partner_x_relative_deviation:+.4%} (X) and {partner_y_relative_deviation:+.4%} (Y), "
        "both sub-1%-relative, well inside the few-percent band the published mode "
        "frequencies' own ~3-significant-figure reporting precision supports. This is an "
        "independent, falsifiable test (nothing in the Mathieu-parameter solve's own "
        "inputs touches the partner ion's frequencies at all) of the WHOLE reconstruction "
        "chain's internal consistency, reported as run."
    )
    enhancement_note = (
        "Per-mode comparison against Marshall et al.'s own published 'Frequency shift per "
        "quantum' row (Table S2): the two AXIAL modes are unchanged from WP31/WP32 "
        f"(enhancement=1.0 identically, q_z=0). The four RADIAL modes now use "
        f"participation*enhancement (F_x={enhancement_x:.4f}, F_y={enhancement_y:.4f}) in place "
        "of WP32's plain participation; the resulting per-mode ratios (predicted/published) "
        f"land at {deviation_sigma:.2f} sigma from the published total "
        f"({'MET' if overlap else 'NOT MET'}). Reported as run."
    )

    return MotionalAlIonIntrinsicMicromotionEnhancedCase(
        case_class="arithmetic_reproduction",
        clock_mathieu=asdict(clock_mathieu),
        enhancement_x=enhancement_x,
        enhancement_y=enhancement_y,
        bare_frequency_partner_x_predicted_hz=predicted_partner_x_hz,
        bare_frequency_partner_y_predicted_hz=predicted_partner_y_hz,
        bare_frequency_partner_x_reconstructed_hz=x_result.bare_frequency_partner_hz,
        bare_frequency_partner_y_reconstructed_hz=y_result.bare_frequency_partner_hz,
        partner_x_relative_deviation=partner_x_relative_deviation,
        partner_y_relative_deviation=partner_y_relative_deviation,
        per_mode=tuple(per_mode),
        predicted_total_nominal=predicted_total,
        predicted_total_uncertainty_fractional=predicted_sigma,
        predicted_total_band_lo=band_lo,
        predicted_total_band_hi=band_hi,
        total_bands_overlap=overlap,
        total_kpi_verdict="MET" if overlap else "NOT MET",
        per_mode_citation=loaders.MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM_CITATION,
        enhancement_caveat=INTRINSIC_MICROMOTION_ENHANCEMENT_CAVEAT,
        enhancement_note=enhancement_note,
        partner_prediction_note=partner_prediction_note,
    )


@dataclass(frozen=True)
class Wp33BrewerConsistencyCheck:
    """A SECOND, independent consistency case for WP33's over-determination
    check and per-mode enhancement, built from Brewer et al.'s
    (arXiv:1902.07694) own published trap parameters -- a different trap
    (different RF drive frequency, different mode frequencies) than
    Marshall et al.'s, same Al27+/Mg25+ species pair.

    Brewer's Table S2 does not publish a static `n_bar` point estimate
    (its occupation input is a 95%-CI bound on `n_bar_0` combined with a
    heating rate through a time-dependent model, `run_motional_al_ion.py`'s
    module docstring item 2), so this case does NOT attempt a total-level
    `(P-1)_motional` reproduction (that would need Brewer's own Eq. 3
    heating-rate model re-implemented, out of scope here, unchanged from
    WP30/31/32's own documented reason for not using Brewer as a total-level
    case). What Brewer DOES publish -- the RF drive frequency, the six mode
    frequencies, and a per-mode `TDS/quantum` row that (Table S2 footnote a)
    ALREADY includes the transverse intrinsic-micromotion shift -- is enough
    to run the SAME over-determination check and the SAME per-mode
    enhancement comparison as the Marshall case, independently.

    Attributes
    ----------
    clock_mathieu : dict[str, float]
        The clock ion's solved Mathieu parameters for Brewer's trap.
    enhancement_x, enhancement_y : float
        `F_x`, `F_y` for Brewer's trap.
    partner_x_relative_deviation, partner_y_relative_deviation : float
        The over-determination check's result for Brewer's trap.
    per_mode_ratio_x_com, per_mode_ratio_x_str, per_mode_ratio_y_com,
    per_mode_ratio_y_str : float
        ``predicted/published`` for each radial mode, against Brewer's own
        `TDS/quantum` row (which already includes intrinsic micromotion,
        so a ratio near `1.0` is direct confirmation).
    missing_input_note : str
        States why this case does not attempt Brewer's own total-level
        `-17.3(2.9)e-19` row.
    """

    clock_mathieu: dict[str, float]
    enhancement_x: float
    enhancement_y: float
    partner_x_relative_deviation: float
    partner_y_relative_deviation: float
    per_mode_ratio_x_com: float
    per_mode_ratio_x_str: float
    per_mode_ratio_y_com: float
    per_mode_ratio_y_str: float
    missing_input_note: str


def run_wp33_brewer_consistency_check() -> Wp33BrewerConsistencyCheck:
    """Build the WP33 Brewer et al. consistency check (see
    :class:`Wp33BrewerConsistencyCheck`'s docstring for scope and the
    missing-input reason it does not attempt a total-level comparison).

    Returns
    -------
    Wp33BrewerConsistencyCheck
    """
    species = get_species("Al27+")
    m_al = species.mass_kg
    m_mg = loaders.MG25_ATOMIC_MASS_AMU * ATOMIC_MASS_UNIT
    rf_drive_frequency_hz = loaders.BREWER_AL_ION_RF_DRIVE_FREQUENCY_HZ

    # loaders.BREWER_AL_ION_MODES_MHZ order: x_com, x_str, y_com, y_str,
    # axial_com, axial_str (Brewer's own Table S2 column order, NOT
    # Marshall's axial-first order -- see that constant's own docstring).
    modes_by_name = dict(loaders.BREWER_AL_ION_MODES_MHZ)
    axial_com_hz = modes_by_name["axial_com"] * 1.0e6
    x_com_hz = modes_by_name["x_com"] * 1.0e6
    x_str_hz = modes_by_name["x_str"] * 1.0e6
    y_com_hz = modes_by_name["y_com"] * 1.0e6
    y_str_hz = modes_by_name["y_str"] * 1.0e6

    c, _c_unc = axial_coulomb_curvature(m_al, m_mg, axial_com_hz)
    x_result = two_ion_radial_participations(m_al, m_mg, c, x_com_hz, x_str_hz)
    y_result = two_ion_radial_participations(m_al, m_mg, c, y_com_hz, y_str_hz)

    clock_mathieu = clock_ion_mathieu_parameters(
        m_al,
        c,
        rf_drive_frequency_hz,
        x_result.bare_frequency_clock_hz,
        y_result.bare_frequency_clock_hz,
    )
    predicted_partner_x_hz, predicted_partner_y_hz = predicted_partner_bare_radial_frequencies_hz(
        clock_mathieu, m_al, m_mg, rf_drive_frequency_hz
    )
    partner_x_relative_deviation = (
        predicted_partner_x_hz - x_result.bare_frequency_partner_hz
    ) / x_result.bare_frequency_partner_hz
    partner_y_relative_deviation = (
        predicted_partner_y_hz - y_result.bare_frequency_partner_hz
    ) / y_result.bare_frequency_partner_hz

    enhancement_x = radial_micromotion_enhancement(
        clock_mathieu.mathieu_q, clock_mathieu.mathieu_a_x
    )
    enhancement_y = radial_micromotion_enhancement(
        clock_mathieu.mathieu_q, clock_mathieu.mathieu_a_y
    )

    published_by_name = dict(
        zip(
            (n for n, _ in loaders.BREWER_AL_ION_MODES_MHZ),
            loaders.BREWER_AL_ION_TDS_PER_QUANTUM,
            strict=True,
        )
    )

    def _ratio(name: str, frequency_hz: float, participation: float, enhancement: float) -> float:
        omega_i = 2.0 * math.pi * frequency_hz
        predicted_pq = (
            -(HBAR * omega_i / m_al) * participation * enhancement / (2.0 * SPEED_OF_LIGHT**2)
        )
        return predicted_pq / published_by_name[name]

    ratio_x_com = _ratio("x_com", x_com_hz, x_result.com_participation, enhancement_x)
    ratio_x_str = _ratio("x_str", x_str_hz, x_result.str_participation, enhancement_x)
    ratio_y_com = _ratio("y_com", y_com_hz, y_result.com_participation, enhancement_y)
    ratio_y_str = _ratio("y_str", y_str_hz, y_result.str_participation, enhancement_y)

    missing_input_note = (
        "Brewer et al.'s own total-level secular-motion row (-17.3(2.9)e-19) is NOT reproduced "
        "here: Table S2 publishes a 95%-CI BOUND on n_bar_0 (zero-point energy excluded) "
        "combined with a per-mode heating rate n_bar_dot through Brewer's own Eq. 3 "
        "(a time-dependent model over the 150 ms interrogation time), not the static n_bar "
        "point estimate E38's formula consumes; the SAME missing-input reason "
        "run_motional_al_ion.py's module docstring already states for why WP30/31/32 use "
        "Marshall et al. instead of Brewer for their own total-level cases. What IS available "
        "from Brewer's Table S2, the RF drive frequency, all six mode frequencies, and a "
        "per-mode TDS/quantum row that already includes the transverse intrinsic-micromotion "
        "shift (footnote a), is what this consistency check uses: the "
        "over-determination check and the per-mode ratios above, both independent of n_bar."
    )

    return Wp33BrewerConsistencyCheck(
        clock_mathieu=asdict(clock_mathieu),
        enhancement_x=enhancement_x,
        enhancement_y=enhancement_y,
        partner_x_relative_deviation=partner_x_relative_deviation,
        partner_y_relative_deviation=partner_y_relative_deviation,
        per_mode_ratio_x_com=ratio_x_com,
        per_mode_ratio_x_str=ratio_x_str,
        per_mode_ratio_y_com=ratio_y_com,
        per_mode_ratio_y_str=ratio_y_str,
        missing_input_note=missing_input_note,
    )


def build_wp33_report() -> dict[str, Any]:
    """Build the standalone WP33 intrinsic-micromotion-enhanced report as a
    JSON-serializable dict, kept in its OWN artifact (``wp33_*.json/md``)
    instead of folded into `build_wp32_report`'s dict, so the existing
    WP30/WP31/WP32 artifacts stay frozen (bit-for-bit unchanged by this
    addition).

    Returns
    -------
    dict[str, Any]
        Metadata plus the WP33 Marshall case
        (:func:`run_motional_al_ion_intrinsic_micromotion_enhanced_case`)
        and the WP33 Brewer consistency check
        (:func:`run_wp33_brewer_consistency_check`).
    """
    case = run_motional_al_ion_intrinsic_micromotion_enhanced_case()
    brewer_check = run_wp33_brewer_consistency_check()
    return {
        "wp33_motional_al_ion_intrinsic_micromotion_enhanced_schema": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "case_class": case.case_class,
        "marshall_2504_13071_intrinsic_micromotion_enhanced_case": asdict(case),
        "brewer_1902_07694_consistency_check": asdict(brewer_check),
    }


def render_wp33_markdown(report: dict[str, Any]) -> str:
    """Render the WP33 intrinsic-micromotion-enhanced case (plus the Brewer
    consistency check) as a markdown summary, mirroring
    :func:`render_wp32_markdown`'s style.

    Parameters
    ----------
    report : dict[str, Any]
        A report dict as returned by :func:`build_wp33_report`.

    Returns
    -------
    str
        A markdown document suitable for embedding or diffing against
        `benchmarks/RESULTS.md`.
    """
    case = report["marshall_2504_13071_intrinsic_micromotion_enhanced_case"]
    brewer = report["brewer_1902_07694_consistency_check"]
    mathieu = case["clock_mathieu"]
    lines = [
        "# WP33 motional Al+ ion intrinsic-micromotion-enhanced benchmark case (generated)",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## WP33: intrinsic-micromotion-enhanced variant "
        "(clock_ion_mathieu_parameters/radial_micromotion_enhancement, Al27+/Mg25+, Marshall)",
        "",
        f"**{case['enhancement_caveat']}**",
        "",
        f"**{case['partner_prediction_note']}**",
        "",
        f"**{case['enhancement_note']}**",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Clock-ion Mathieu q | {mathieu['mathieu_q']:.6f} |",
        f"| Clock-ion Mathieu a_x | {mathieu['mathieu_a_x']:+.6e} |",
        f"| Clock-ion Mathieu a_y | {mathieu['mathieu_a_y']:+.6e} |",
        f"| Clock-ion Mathieu a_z | {mathieu['mathieu_a_z']:+.6e} |",
        f"| Enhancement F_x | {case['enhancement_x']:.4f} |",
        f"| Enhancement F_y | {case['enhancement_y']:.4f} |",
        (
            "| Predicted partner bare freq, X (Hz) | "
            f"{case['bare_frequency_partner_x_predicted_hz']:.6e} |"
        ),
        (
            "| WP32-reconstructed partner bare freq, X (Hz) | "
            f"{case['bare_frequency_partner_x_reconstructed_hz']:.6e} |"
        ),
        f"| Partner X relative deviation | {case['partner_x_relative_deviation']:+.4%} |",
        (
            "| Predicted partner bare freq, Y (Hz) | "
            f"{case['bare_frequency_partner_y_predicted_hz']:.6e} |"
        ),
        (
            "| WP32-reconstructed partner bare freq, Y (Hz) | "
            f"{case['bare_frequency_partner_y_reconstructed_hz']:.6e} |"
        ),
        f"| Partner Y relative deviation | {case['partner_y_relative_deviation']:+.4%} |",
        "",
        "| Mode | Axial? | Participation | Enhancement | Predicted shift/quantum | "
        "Published shift/quantum | Ratio (pred/pub) |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {m['name']} | {m['is_axial']} | {m['participation']:.4f} | "
            f"{m['enhancement']:.4f} | {m['predicted_shift_per_quantum']:+.4e} | "
            f"{m['published_shift_per_quantum']:+.4e} | "
            f"{m['ratio_predicted_over_published']:+.4f} |"
            for m in case["per_mode"]
        ),
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Enhancement-corrected total (P-1)_motional | {case['predicted_total_nominal']:+.6e} |",
        (
            "| Enhancement-corrected uncertainty (1-sigma) | "
            f"+/-{case['predicted_total_uncertainty_fractional']:.3e} |"
        ),
        (
            "| Enhancement-corrected band | "
            f"[{case['predicted_total_band_lo']:+.6e}, {case['predicted_total_band_hi']:+.6e}] |"
        ),
        f"| Total bands overlap | {case['total_bands_overlap']} |",
        f"| **total_kpi_verdict** | **{case['total_kpi_verdict']}** |",
        "",
        f"Per-mode published-value citation: {case['per_mode_citation']}",
        "",
        "## WP33 Brewer et al. (2019, arXiv:1902.07694) consistency check "
        "(second, independent dataset)",
        "",
        f"**{brewer['missing_input_note']}**",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Clock-ion Mathieu q | {brewer['clock_mathieu']['mathieu_q']:.6f} |",
        f"| Clock-ion Mathieu a_x | {brewer['clock_mathieu']['mathieu_a_x']:+.6e} |",
        f"| Clock-ion Mathieu a_y | {brewer['clock_mathieu']['mathieu_a_y']:+.6e} |",
        f"| Clock-ion Mathieu a_z | {brewer['clock_mathieu']['mathieu_a_z']:+.6e} |",
        f"| Enhancement F_x | {brewer['enhancement_x']:.4f} |",
        f"| Enhancement F_y | {brewer['enhancement_y']:.4f} |",
        f"| Partner X relative deviation | {brewer['partner_x_relative_deviation']:+.4%} |",
        f"| Partner Y relative deviation | {brewer['partner_y_relative_deviation']:+.4%} |",
        f"| Per-mode ratio (pred/pub), x_com | {brewer['per_mode_ratio_x_com']:+.4f} |",
        f"| Per-mode ratio (pred/pub), x_str | {brewer['per_mode_ratio_x_str']:+.4f} |",
        f"| Per-mode ratio (pred/pub), y_com | {brewer['per_mode_ratio_y_com']:+.4f} |",
        f"| Per-mode ratio (pred/pub), y_str | {brewer['per_mode_ratio_y_str']:+.4f} |",
    ]
    return "\n".join(lines) + "\n"


def build_wp32_report() -> dict[str, Any]:
    """Build the standalone WP32 radial-spectrum-reconstructed report as a
    JSON-serializable dict, kept in its OWN artifact (``wp32_*.json/md``)
    instead of folded into `build_report`'s WP30/WP31 dict, so the
    existing WP30 artifacts stay frozen (bit-for-bit unchanged by this
    addition).

    Returns
    -------
    dict[str, Any]
        Metadata plus the WP32 case (:func:`run_motional_al_ion_radial_reconstructed_case`).
    """
    case = run_motional_al_ion_radial_reconstructed_case()
    return {
        "wp32_motional_al_ion_radial_reconstructed_schema": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "case_class": case.case_class,
        "marshall_2504_13071_radial_reconstructed_variant_case": asdict(case),
    }


def render_wp32_markdown(report: dict[str, Any]) -> str:
    """Render the WP32 radial-spectrum-reconstructed case as a markdown
    summary, mirroring :func:`render_markdown`'s style.

    Parameters
    ----------
    report : dict[str, Any]
        A report dict as returned by :func:`build_wp32_report`.

    Returns
    -------
    str
        A markdown document suitable for embedding or diffing against
        `benchmarks/RESULTS.md`.
    """
    case = report["marshall_2504_13071_radial_reconstructed_variant_case"]
    lines = [
        "# WP32 motional Al+ ion radial-spectrum-reconstructed benchmark case (generated)",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## WP32: radial-spectrum-reconstructed variant "
        "(two_ion_radial_participations, Al27+/Mg25+)",
        "",
        f"**{case['radial_reconstruction_caveat']}**",
        "",
        f"**{case['participation_note']}**",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Coulomb curvature c (N/m), from axial COM | {case['coulomb_curvature_n_per_m']:.6e} |",
        (
            "| Coulomb curvature c (N/m), cross-check from axial STR | "
            f"{case['coulomb_curvature_cross_check_n_per_m']:.6e} |"
        ),
        (
            "| Cross-check relative deviation | "
            f"{case['coulomb_curvature_cross_check_relative_deviation']:+.4e} |"
        ),
        (
            "| Bare radial frequency, clock ion, X branch (Hz) | "
            f"{case['bare_frequency_clock_x_hz']:.6e} |"
        ),
        (
            "| Bare radial frequency, partner ion, X branch (Hz) | "
            f"{case['bare_frequency_partner_x_hz']:.6e} |"
        ),
        (
            "| Bare radial frequency, clock ion, Y branch (Hz) | "
            f"{case['bare_frequency_clock_y_hz']:.6e} |"
        ),
        (
            "| Bare radial frequency, partner ion, Y branch (Hz) | "
            f"{case['bare_frequency_partner_y_hz']:.6e} |"
        ),
        "",
        "| Mode | Axial? | Participation | Predicted shift/quantum | Published "
        "shift/quantum | Ratio (pred/pub) |",
        "|---|---|---|---|---|---|",
        *(
            f"| {m['name']} | {m['is_axial']} | {m['participation']:.4f} | "
            f"{m['predicted_shift_per_quantum']:+.4e} | "
            f"{m['published_shift_per_quantum']:+.4e} | "
            f"{m['ratio_predicted_over_published']:+.4f} |"
            for m in case["per_mode"]
        ),
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Reconstructed-participation total (P-1)_motional | "
        f"{case['predicted_total_nominal']:+.6e} |",
        f"| Reconstructed-participation uncertainty (1-sigma) | "
        f"+/-{case['predicted_total_uncertainty_fractional']:.3e} |",
        (
            "| Reconstructed-participation band | "
            f"[{case['predicted_total_band_lo']:+.6e}, "
            f"{case['predicted_total_band_hi']:+.6e}] |"
        ),
        f"| Total bands overlap | {case['total_bands_overlap']} |",
        f"| **total_kpi_verdict** | **{case['total_kpi_verdict']}** |",
        "",
        f"Per-mode published-value citation: {case['per_mode_citation']}",
    ]
    return "\n".join(lines) + "\n"


def build_report() -> dict[str, Any]:
    """Build the full WP30/WP31 motional-Al-ion benchmark report as a
    JSON-serializable dict.

    Returns
    -------
    dict[str, Any]
        Metadata plus both cases: the WP30 single-mass variant (see
        :func:`run_motional_al_ion_arithmetic_reproduction_case`) and the
        WP31 participation-corrected variant (see
        :func:`run_motional_al_ion_participation_variant_case`).
        Deliberately NOT merged into `run_benchmarks.build_report`'s WP10
        report or `kpi_summary`; see module docstring.
    """
    case = run_motional_al_ion_arithmetic_reproduction_case()
    participation_case = run_motional_al_ion_participation_variant_case()
    return {
        "wp30_motional_al_ion_benchmark_schema": "1.1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "case_label": CASE_LABEL,
        "case_class": case.case_class,
        "marshall_2504_13071_secular_motion_arithmetic_reproduction_case": asdict(case),
        "marshall_2504_13071_participation_corrected_variant_case": asdict(participation_case),
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
    p_case = report["marshall_2504_13071_participation_corrected_variant_case"]
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
        "",
        "## WP31: participation-corrected variant (two_ion_participations, Al27+/Mg25+)",
        "",
        f"**{p_case['participation_note']}**",
        "",
        "| Mode | Axial? | Participation | Predicted shift/quantum | Published "
        "shift/quantum | Ratio (pred/pub) |",
        "|---|---|---|---|---|---|",
        *(
            f"| {m['name']} | {m['is_axial']} | {m['participation']:.4f} | "
            f"{m['predicted_shift_per_quantum']:+.4e} | "
            f"{m['published_shift_per_quantum']:+.4e} | "
            f"{m['ratio_predicted_over_published']:+.4f} |"
            for m in p_case["per_mode"]
        ),
        "",
        f"Axial mean |ratio-1| deviation: {p_case['axial_mean_abs_ratio_deviation']:.4f}. "
        f"Radial mean |ratio-1| deviation: {p_case['radial_mean_abs_ratio_deviation']:.4f}.",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Participation-corrected total (P-1)_motional | "
        f"{p_case['predicted_total_nominal']:+.6e} |",
        f"| Participation-corrected uncertainty (1-sigma) | "
        f"+/-{p_case['predicted_total_uncertainty_fractional']:.3e} |",
        (
            "| Participation-corrected band | "
            f"[{p_case['predicted_total_band_lo']:+.6e}, "
            f"{p_case['predicted_total_band_hi']:+.6e}] |"
        ),
        f"| Total bands overlap | {p_case['total_bands_overlap']} |",
        f"| **total_kpi_verdict** | **{p_case['total_kpi_verdict']}** |",
        "",
        f"Per-mode published-value citation: {p_case['per_mode_citation']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the WP30/WP31 motional-Al-ion benchmark cases and write
    `benchmarks/results/wp30_motional_al_ion_arithmetic_reproduction.json`
    and a generated markdown summary alongside it (frozen format, unchanged
    by WP32), then run the WP32 radial-spectrum-reconstructed case and
    write its own `wp32_motional_al_ion_radial_reconstructed.json`/`.md`
    artifacts alongside the WP30 ones."""
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

    wp32_report = build_wp32_report()
    wp32_json_path = _RESULTS_DIR / "wp32_motional_al_ion_radial_reconstructed.json"
    wp32_md_path = _RESULTS_DIR / "wp32_motional_al_ion_radial_reconstructed.md"
    wp32_json_path.write_text(
        json.dumps(wp32_report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    wp32_markdown = render_wp32_markdown(wp32_report)
    wp32_md_path.write_text(wp32_markdown, encoding="utf-8")
    print(wp32_markdown)
    print(f"Wrote {wp32_json_path}")
    print(f"Wrote {wp32_md_path}")

    wp33_report = build_wp33_report()
    wp33_json_path = _RESULTS_DIR / "wp33_motional_al_ion_intrinsic_micromotion_enhanced.json"
    wp33_md_path = _RESULTS_DIR / "wp33_motional_al_ion_intrinsic_micromotion_enhanced.md"
    wp33_json_path.write_text(
        json.dumps(wp33_report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    wp33_markdown = render_wp33_markdown(wp33_report)
    wp33_md_path.write_text(wp33_markdown, encoding="utf-8")
    print(wp33_markdown)
    print(f"Wrote {wp33_json_path}")
    print(f"Wrote {wp33_md_path}")


if __name__ == "__main__":
    main()
