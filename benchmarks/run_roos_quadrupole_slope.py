# SPDX-License-Identifier: AGPL-3.0-or-later
"""Roos-slope ion benchmark case: Ca+:D5/2 two-ion quadrupole-shift slope
(Roos et al., Nature 443, 316 (2006), quant-ph/0701215v1, Eq. 1/Fig. 4a,
the primary-text extraction; the project's theory sign-off
record (G8) B4's labeling ruling, applied here to the
Roos case exactly as it was ratified for the analogous Barwood case).

**What this case checks.** Roos et al. (quant-ph/0701215v1, published
Nature 443, 316 (2006)) measured a two-ion entangled state's quadrupole-
shift *parity-oscillation frequency* as a function of an applied,
mechanically (omega_z) calibrated axial electric-field gradient, and fit a
straight line to it (Fig. 4a): slope ``a = 2.975(2) Hz*mm^2/V`` (beta=0,
dE_z/dz = 12-48 V/mm^2), offset ``Delta_0/(2*pi) = -2.4(1) Hz``. This
script predicts that slope two ways from this engine's real quadrupole-
shift functions (``cliffordclock.integrator.omega.quadrupole_shift_joules``/
``quadrupole_mj_factor``) and the species registry's Ca+:D5/2 quadrupole
moment (``cliffordclock.ensemble.species.QUADRUPOLE_MOMENTS["Ca+:D5/2"]``),
and reports both predictions against Roos's own published slope.

**Binding classification (G8 sign-off B4, ratified for the Barwood case;
applied identically here per the builder brief's instruction to follow
B4's ruling): lead with the CROSS-VINTAGE comparison, compute both
variants, label them distinctly.**

- **Cross-vintage comparison (headline, ``case_class =
  "cross_vintage_comparison"``):** predict Roos's slope from Itano's
  INDEPENDENT theory value, ``Theta_theory(Ca+:D5/2) = 1.917 ea0^2``
  (Itano, Phys. Rev. A 73, 022510 (2006) -- a different vintage/method
  from Roos's own 2006 measurement: an ab-initio calculation, not a fit to
  this same dataset). This is a genuine comparison: neither Theta_theory
  nor its provenance comes from the Fig. 4a fit being predicted.
- **Arithmetic reproduction (weaker, explicitly labeled,
  ``case_class = "arithmetic_reproduction"``):** predict Roos's slope from
  Roos's OWN extracted ``Theta = 1.83(1) ea0^2`` (the very quantity their
  own paper derives BY INVERTING their own Fig. 4a fit through
  ``Theta = (5/12)*h*a``). Using Roos's own Theta to predict Roos's own
  slope is circular by construction -- not a genuine external check, only
  a closed-loop arithmetic/factor-consistency check of this engine's own
  chain against their published conversion relation.
- **Even Roos's own gradient is not independent-of-a-trap-model.** Per
  G8 B4's second nuance (documented for completeness, not re-litigated
  here): Roos's ``dE_z/dz`` is calibrated from the ion's own measured
  axial secular frequency ``omega_z`` via ``dE_z/dz = -m*omega_z^2/e``
  (dossier section 6) -- a direct mechanical measurement, but still a
  trap-model-derived gradient, not an FEA-independent field
  reconstruction. This case is therefore not a fully independent-gradient
  validation either, the same caveat B4 required for Barwood.
- **This case is neither ``"reproducibility"`` nor ``"blind_prediction"``**
  (``benchmarks/run_benchmarks.py``'s taxonomy) and does NOT increment
  either of those headline counts (``benchmarks/results/wp10_results.json``
  stays at 1 reproducibility case / 0 blind-prediction cases). It is
  tracked in this separate script/report, exactly like
  ``benchmarks/run_bbr_jila_arithmetic_reproduction.py``'s WP20 case.
  ``"cross_vintage_comparison"`` is a NEW taxonomy label (this is the
  first case needing it): weaker than ``"blind_prediction"`` (Roos's own
  applied gradient still traces to their own trap-model calibration, and
  their own fit is what produced the ``a`` this compares against -- not an
  independent field/shift pair this engine's inputs never touched), but
  distinct from ``"arithmetic_reproduction"`` (the theory Theta is
  genuinely independent of the fit being predicted, unlike the
  arithmetic-reproduction variant's circular Theta). See
  ``benchmarks/RESULTS.md``'s "Classification taxonomy" section for the
  added definition.

---

**THE PHYSICS CHAIN (derived independently below; every numeric literal
cited to Roos et al., Nature 443, 316 (2006), quant-ph/0701215v1,
Eq. 1/Fig. 4a; every computed
value below is produced by REAL calls to
``cliffordclock.integrator.omega.quadrupole_shift_joules``/
``quadrupole_mj_factor`` -- no hand arithmetic feeds any reported number).**

**1. Single-ion shift, general (J, m_J), at beta=0.** CONVENTIONS.md E34's
coordinate-free form (already implemented by
``quadrupole_shift_joules``/``quadrupole_mj_factor``, itself Roos Eq. 1
transcribed verbatim -- see CONVENTIONS.md section 14):

    Delta_E_Q(m_J) = (Theta_SI/2) * f(m_J) * (dE_z/dz),
    f(m_J) = [J(J+1) - 3*m_J^2] / [J(2J-1)]   (beta=0: n_hat^T.G.n_hat = dE_z/dz)

For J=5/2 (Ca+:D5/2): ``f(5/2)=-1``, ``f(3/2)=+0.2``, ``f(1/2)=+0.8`` --
each an evaluation of the SAME ``quadrupole_mj_factor(j=2.5, m_j=...)``
this project's engine already ships and CONVENTIONS.md section 14 already
derives; not re-derived here, only exercised.

**2. The two-ion entangled state's parity-oscillation frequency.** Psi_1 =
(|-5/2>|+3/2> + |-1/2>|-1/2>)/sqrt(2) (dossier section 6) is a coherent
superposition of two two-ion PRODUCT states. Each product state's total
quadrupole energy is the sum of its two ions' single-particle shifts (the
quadrupole Hamiltonian is diagonal in each ion's own m_J -- no cross-ion
coupling term enters at this order). The parity-oscillation angular
frequency observed between the two components is the energy DIFFERENCE
between the two product states, divided by hbar:

    hbar*Delta_1 = [Delta_E_Q(-5/2) + Delta_E_Q(+3/2)] - [Delta_E_Q(-1/2) + Delta_E_Q(-1/2)]

evaluated PER ION at whatever gradient each ion actually sees. Per Roos
p.5-6 (dossier section 6): "the presence of a second ion doubles the
electric field gradient at the location of the other ion" -- by the
trap's symmetry, BOTH ions therefore see ``2*(dE_z/dz)_applied``, not the
bare applied gradient. Substituting the single-ion formula (step 1) at
this doubled gradient into the bracket above and using
``f(5/2)+f(3/2)-f(1/2)-f(1/2) = -1+0.2-0.8-0.8 = -2.4`` (the dossier's
stated check; reproduced here via REAL ``quadrupole_mj_factor`` calls, not
copied) gives

    hbar*Delta_1 = (Theta_SI/2) * [2*(dE_z/dz)_applied] * (-2.4)
                 = -2.4 * Theta_SI * (dE_z/dz)_applied

**Structural pin: the 24/5 two-ion enhancement.** A single |-5/2> ion's
shift at the SAME (undoubled) applied gradient is
``Delta_E_Q(-5/2) = (Theta_SI/2)*(-1)*(dE_z/dz)_applied``. The ratio

    |hbar*Delta_1| / |Delta_E_Q(-5/2)| = |-2.4*Theta_SI*g| / |-0.5*Theta_SI*g| = 4.8 = 24/5

exactly reproduces Roos's stated "24/5 enhancement relative to a single
|-5/2> ion" (dossier section 6) -- a real, engine-derived structural check
(:func:`structural_two_ion_enhancement_ratio` below), not asserted.

**3. Slope prediction and the (5/12)*h*a closure.** Converting the angular
frequency to an ordinary (Hz) parity-oscillation frequency,
``Delta_nu_1 = hbar*Delta_1 / h_planck`` (``h_planck`` the REAL
``cliffordclock.constants.PLANCK_H``), and dividing by the applied
gradient gives the predicted slope:

    a_pred = |Delta_nu_1 / (dE_z/dz)_applied| = 2.4 * Theta_SI / h_planck   [Hz per (V/m^2)]

(the absolute value per the builder brief -- see "Sign, deliberately not
addressed" below). Converting to Roos's own units, ``Hz*mm^2/V`` (1
V/mm^2 = 1e6 V/m^2, so a numeric slope value in Hz/(V/mm^2) is 1e6x the
same physical slope's numeric value in Hz/(V/m^2)):

    a_pred [Hz*mm^2/V] = 2.4 * Theta_SI / h_planck * 1e6

Solving this same relation for Theta reproduces Roos's own stated
conversion EXACTLY: ``Theta_SI = (a_pred[Hz/(V/m^2)] * h_planck) / 2.4``,
and ``1/2.4 = 5/12`` -- i.e. ``Theta = (5/12)*h*a``, Roos's own stated
relation (dossier section 6), confirming this engine's factor chain
carries the SAME leading numerical coefficient Roos's own paper uses (a
genuine, non-trivial factor-consistency check: this project's coefficient
chain was derived independently above, not copied from Roos's ``5/12``).
:func:`roos_theta_au_roundtrip` below performs this round trip
numerically (Theta -> a_pred -> Theta_check) and confirms
``Theta_check == Theta_in`` to float64 precision for ANY input Theta --
this is the "closed-loop arithmetic check" the builder brief specifies:
it validates this engine's ``2.4`` factor against Roos's published
``5/12`` INDEPENDENTLY of any external rounding (the round trip uses only
this engine's own two directions of the same linear relation, never
comparing against Roos's rounded ``Theta=1.83`` or ``a=2.975`` figures).

**Sign, deliberately not addressed.** ``docs/CONVENTIONS.md`` section 14's
own "AMBIGUITY" note explicitly reserves the full two-ion Fig. 3a/4a
ABSOLUTE sign triple (which requires the trap's own electrode-polarity
convention, not derivable from Eq. 1 alone) for "the separate, later
Roos/Barwood benchmark WP" -- i.e. this script. Per the builder brief's
own derivation sketch, ``a_pred`` is defined here as an absolute value,
matching how Roos report a positive slope; this script does not attempt
to independently re-derive which physical gradient direction Roos call
positive. This is a real, standing open point (not silently resolved),
flagged again in this module's test suite and in the builder's final
report -- it does not affect the magnitude comparison this case makes.

**4. The offset does not enter this comparison.** ``Delta_0/(2*pi) =
-2.4(1) Hz`` (dossier section 6) is Fig. 4a's fitted Y-INTERCEPT -- the
shift at ZERO applied gradient. It is explicitly NOT part of the
``Theta``-dependent SLOPE this case predicts: of the ``-2.4 Hz``, Roos
attribute ``-2.9 Hz`` to second-order Zeeman shift at their 2.9 G bias
field (a magnetic-field mechanism this engine has no model for --
``docs/CONVENTIONS.md`` has no Zeeman term anywhere) and the remainder to
a residual stray quadrupole field (a DIFFERENT, uncontrolled gradient this
case's applied-gradient sweep does not represent). Composing an offset
into this case's linear ``a_pred`` prediction would require BOTH a Zeeman
model this engine does not have AND an independently characterized stray
field this paper does not publish -- exactly the same "no independent
field input" gap ``benchmarks/MAPPING.md`` documents for the JILA/USTC
DC-Stark rows. ``ROOS_FIT_OFFSET_HZ`` (``benchmarks/loaders.py``) is
carried in this case's report purely for documentation/citation
completeness, never combined with ``a_pred``.

Run this yourself: ``python benchmarks/run_roos_quadrupole_slope.py``
(from the repo root, with ``.venv`` active). Regenerates
``benchmarks/results/roos_quadrupole_slope.json`` and
``benchmarks/results/roos_quadrupole_slope.md``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp

# Allow running as `python benchmarks/run_roos_quadrupole_slope.py` (no
# package install needed -- benchmarks/ is deliberately not part of the
# installed package, see benchmarks/SOURCES.md's packaging note), and
# mirror how `benchmarks/run_bbr_jila_arithmetic_reproduction.py` imports
# its sibling `loaders`/`run_benchmarks` modules.
_BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import loaders  # noqa: E402
import run_benchmarks  # noqa: E402 -- reuses the already-tested `_bands_overlap`

from cliffordclock.constants import PLANCK_H  # noqa: E402
from cliffordclock.ensemble.species import EA0_SQUARED_SI, QUADRUPOLE_MOMENTS  # noqa: E402
from cliffordclock.integrator.omega import quadrupole_shift_joules  # noqa: E402

_RESULTS_DIR = _BENCHMARKS_DIR / "results"

#: The Ca+:D5/2 total angular momentum (dossier section 6, registry key
#: "Ca+:D5/2"). Pulled from the registry entry's own `j` field at call
#: time, not hand-typed as a bare literal, wherever the registry object is
#: in scope; this module-level constant exists only for the handful of
#: helper-function signatures below that need a bare float.
_J_D5_2 = QUADRUPOLE_MOMENTS["Ca+:D5/2"].j

#: V/mm^2 -> V/m^2 (Roos's own gradient unit; 1 mm^2 = 1e-6 m^2, so 1
#: V/mm^2 = 1e6 V/m^2). Used to convert this case's V/mm^2-denominated
#: applied-gradient inputs into the SI (V/m^2) units
#: `quadrupole_shift_joules` requires (CONVENTIONS.md E13).
_VMM2_TO_VM2 = 1.0e6

#: A physically representative applied gradient inside Roos's own measured
#: range (Fig. 4a: dE_z/dz = 12-48 V/mm^2, dossier section 6), used as the
#: single evaluation point for every slope computation below. Because
#: `Delta_E_Q` is EXACTLY linear in the gradient (no offset term in E34),
#: the slope `a_pred` computed at this one point equals the slope at any
#: other point -- `test_roos_benchmark.py` pins this linearity directly
#: (two independent gradients, same recovered slope to float precision)
#: rather than merely assuming it.
CANONICAL_APPLIED_GRADIENT_VMM2 = 30.0

#: Binding classification label for the cross-vintage (headline) variant
#: -- see module docstring for the full labeling rationale (G8 sign-off
#: B4, applied to the Roos case).
CROSS_VINTAGE_CASE_LABEL = (
    "cross-vintage comparison: Roos et al.'s measured Fig. 4a slope against "
    "an INDEPENDENT theory Theta (Itano, Phys. Rev. A 73, 022510 (2006)), "
    "not against Roos's own extracted Theta; weaker than a blind "
    "prediction (Roos's own applied gradient is itself trap-model "
    "calibrated, and their own fit produced the slope being predicted), "
    "but a genuine external comparison, distinct from arithmetic "
    "reproduction (G8 sign-off B4)"
)

#: Binding classification label for the arithmetic-reproduction (weaker,
#: secondary) variant.
ARITHMETIC_REPRODUCTION_CASE_LABEL = (
    "arithmetic reproduction: Roos et al.'s own extracted Theta, inverted "
    "from their own Fig. 4a fit via Theta = (5/12)*h*a, used to predict "
    "that SAME fit's slope; circular by construction, a closed-loop "
    "factor-consistency check of this engine's chain against Roos's own "
    "published conversion relation, never an independent validation (G8 "
    "sign-off B4)"
)

_DOSSIER_CITATION = (
    "Roos et al., Nature 443, 316 (2006), quant-ph/0701215v1, Eq. 1 + "
    "Fig. 4a + p.5-6/p.9 text; benchmarks/SOURCES.md section 7 "
    "(owner-supplied primary text, provenance note)."
)


def _traceless_axial_gradient_tensor(dez_dz_v_per_m2: float) -> jnp.ndarray:
    """A traceless, axially symmetric (eps=0) gradient tensor with
    ``dE_z/dz = dez_dz_v_per_m2`` along the lab-frame z axis:
    ``diag(-g/2, -g/2, g)`` -- already traceless (E34's requirement), so
    `quadrupole_shift_joules`'s internal `traceless_symmetric_gradient`
    call is a no-op here and the z-axis quantization contraction below
    returns exactly `dez_dz_v_per_m2` (CONVENTIONS.md section 14, beta=0
    special case).
    """
    g = float(dez_dz_v_per_m2)
    return jnp.array(
        [
            [-g / 2.0, 0.0, 0.0],
            [0.0, -g / 2.0, 0.0],
            [0.0, 0.0, g],
        ]
    )


#: Lab-frame quantization axis for every evaluation below: the z axis,
#: i.e. beta=0 in CONVENTIONS.md E34's angular notation (Roos's Fig. 4a
#: data is explicitly taken at beta=0, dossier section 6).
_Z_AXIS = jnp.array([0.0, 0.0, 1.0])


def _single_ion_shift_joules(theta_au: float, dez_dz_v_per_m2: float, m_j: float) -> float:
    """``Delta_E_Q(m_J)`` (joules) for one Ca+:D5/2 ion at the given
    (undoubled) axial gradient, beta=0 -- a direct call to the real engine
    function `quadrupole_shift_joules`, no hand arithmetic.
    """
    grad = _traceless_axial_gradient_tensor(dez_dz_v_per_m2)
    return float(quadrupole_shift_joules(grad, _Z_AXIS, theta_au, _J_D5_2, m_j))


def two_ion_hbar_delta1_joules(theta_au: float, applied_gradient_vmm2: float) -> float:
    """``hbar*Delta_1`` (joules): the two-ion Psi_1 state's parity-
    oscillation ENERGY splitting, at the given APPLIED gradient (Roos's
    own Fig. 4a x-axis quantity, ``Hz*mm^2/V``'s denominator).

    Per the module docstring's step 2: each ion sees the DOUBLED gradient
    (``2 * applied_gradient``, dossier section 6's "the presence of a
    second ion doubles the electric field gradient at the location of the
    other ion"); this function calls `quadrupole_shift_joules` four times
    (once per ion per product-state component:
    ``m_J in {-5/2, +3/2, -1/2, -1/2}``) at that doubled gradient and
    differences the two product-state energies -- no shortcut formula.

    Parameters
    ----------
    theta_au : float
        Ca+:D5/2's electric-quadrupole moment, atomic units.
    applied_gradient_vmm2 : float
        The APPLIED (not doubled) axial gradient, ``V/mm^2`` (Roos's own
        Fig. 4a units).

    Returns
    -------
    float
        ``hbar*Delta_1``, joules (negative for positive `theta_au`, since
        the bracket sums to ``-2.4``, CONVENTIONS.md E34's sign
        convention -- see module docstring's "Sign, deliberately not
        addressed" note).
    """
    doubled_gradient_v_per_m2 = 2.0 * applied_gradient_vmm2 * _VMM2_TO_VM2
    e_component_1 = _single_ion_shift_joules(
        theta_au, doubled_gradient_v_per_m2, -2.5
    ) + _single_ion_shift_joules(theta_au, doubled_gradient_v_per_m2, 1.5)
    e_component_2 = _single_ion_shift_joules(
        theta_au, doubled_gradient_v_per_m2, -0.5
    ) + _single_ion_shift_joules(theta_au, doubled_gradient_v_per_m2, -0.5)
    return e_component_1 - e_component_2


def structural_two_ion_enhancement_ratio(
    applied_gradient_vmm2: float = CANONICAL_APPLIED_GRADIENT_VMM2,
) -> float:
    """The two-ion Psi_1 enhancement ratio relative to a single ``|-5/2>``
    ion's shift at the SAME applied (undoubled) gradient -- structural pin
    (module docstring step 2), sign-convention-independent (any nonzero
    `theta_au` cancels in the ratio, so `theta_au=1.0` is used).

    Returns
    -------
    float
        ``hbar*Delta_1 / Delta_E_Q(-5/2)`` -- must equal ``+24/5 = +4.8``
        to float precision (POSITIVE: both the two-ion splitting and the
        single-ion reference shift carry the same overall sign, so their
        ratio is positive even though the underlying bracket sum
        ``f(5/2)+f(3/2)-f(1/2)-f(1/2) = -2.4 = -12/5`` that produces it is
        negative -- the dossier's own "24/5 ... in magnitude" phrasing
        already flags this as a magnitude, not a signed quantity; module
        docstring step 2 shows the ``|...|/|...|`` form explicitly).
        Recovered here from REAL `quadrupole_shift_joules` calls, not
        asserted.
    """
    hbar_delta1 = two_ion_hbar_delta1_joules(1.0, applied_gradient_vmm2)
    single_ion = _single_ion_shift_joules(1.0, applied_gradient_vmm2 * _VMM2_TO_VM2, -2.5)
    return hbar_delta1 / single_ion


def predicted_slope_hz_mm2_per_v(
    theta_au: float, applied_gradient_vmm2: float = CANONICAL_APPLIED_GRADIENT_VMM2
) -> float:
    """``a_pred`` (module docstring step 3): the predicted Fig. 4a slope
    magnitude, ``Hz*mm^2/V``, for the given Theta.

    ``|Delta_nu_1 / applied_gradient|``, with ``Delta_nu_1 =
    two_ion_hbar_delta1_joules(...) / PLANCK_H`` (the real engine constant,
    not a hand-typed ``6.626e-34``) -- an ordinary-frequency (Hz)
    parity-oscillation rate, per the module docstring's "their `Delta_nu`
    an angular-frequency shift" transcription note
    (`docs/CONVENTIONS.md` section 14): `hbar*Delta_1` is an ENERGY, so
    dividing by `h` (not `hbar`) gives the ordinary-Hz beat frequency Roos
    report in `Hz*mm^2/V`.
    """
    hbar_delta1 = two_ion_hbar_delta1_joules(theta_au, applied_gradient_vmm2)
    delta_nu1_hz = hbar_delta1 / PLANCK_H
    return abs(delta_nu1_hz / applied_gradient_vmm2)


def roos_theta_au_roundtrip(
    theta_au: float, applied_gradient_vmm2: float = CANONICAL_APPLIED_GRADIENT_VMM2
) -> float:
    """The closed-loop arithmetic check (module docstring step 3):
    ``theta_au -> a_pred -> Theta_check``, via Roos's own stated
    ``Theta = (5/12)*h*a`` relation (dossier section 6), independently of
    any comparison against Roos's published, rounded numbers.

    Returns
    -------
    float
        ``Theta_check_au`` -- must equal ``abs(theta_au)`` to float64
        precision for ANY input (an identity of this engine's own linear
        chain, per the module docstring's derivation: ``(5/12)*2.4 = 1``
        exactly). The absolute value, not `theta_au` itself, because
        `predicted_slope_hz_mm2_per_v` (module docstring's "Sign,
        deliberately not addressed" note) returns a magnitude, so any sign
        on the input `theta_au` is lost before this function's own
        ``(5/12)*h*a`` step runs.
    """
    a_pred_hz_mm2_per_v = predicted_slope_hz_mm2_per_v(theta_au, applied_gradient_vmm2)
    a_pred_hz_per_vm2 = a_pred_hz_mm2_per_v / _VMM2_TO_VM2
    theta_check_si = (5.0 / 12.0) * PLANCK_H * a_pred_hz_per_vm2
    return theta_check_si / EA0_SQUARED_SI


@dataclass(frozen=True)
class QuadrupoleSlopeVariant:
    """One Theta-input variant of the Roos two-ion slope prediction (see
    module docstring: `cross_vintage` uses Itano's independent theory
    Theta; `arithmetic_reproduction` uses Roos's own extracted Theta).

    Attributes
    ----------
    variant_label : str
        Always `CROSS_VINTAGE_CASE_LABEL` or
        `ARITHMETIC_REPRODUCTION_CASE_LABEL`, verbatim.
    case_class : str
        `"cross_vintage_comparison"` or `"arithmetic_reproduction"`.
    theta_au, theta_au_uncertainty : float, float | None
        The input Theta and its 1-sigma uncertainty (`None` for the
        cross-vintage variant: Itano's theory value carries no published
        uncertainty, dossier section 6 -- stated, not silently treated as
        zero).
    theta_source : str
        Citation for `theta_au`.
    predicted_slope_hz_mm2_per_v : float
        `a_pred`, this variant's Theta run through
        :func:`predicted_slope_hz_mm2_per_v`.
    predicted_slope_band_lo, _hi : float
        `a_pred` evaluated at `theta_au -/+ theta_au_uncertainty` (two
        more direct engine calls, not a derivative estimate) -- degenerate
        (`lo == hi == predicted_slope_hz_mm2_per_v`) when
        `theta_au_uncertainty` is `None`.
    measured_slope_hz_mm2_per_v, measured_slope_lo, _hi : float
        Roos's own published Fig. 4a slope band
        (`loaders.ROOS_MEASURED_SLOPE_HZ_MM2_PER_V`).
    residual_hz_mm2_per_v, residual_fractional : float
        `predicted_slope_hz_mm2_per_v - measured_slope_hz_mm2_per_v`,
        absolute and relative-to-measured.
    bands_overlap : bool
        Whether `[predicted_slope_band_lo, _hi]` and
        `[measured_slope_lo, _hi]` overlap
        (`run_benchmarks._bands_overlap`, reused).
    kpi_verdict : str
        `"MET"` if `bands_overlap` else `"NOT MET"` (never `"PASS"`/
        `"FAIL"`, this project's reserved vocabulary) -- for the
        cross-vintage variant, `"NOT MET"` is the EXPECTED outcome (module
        docstring: this recovers the known Theta theory-vs-measurement
        tension, not an engine defect).
    verdict_note : str | None
        Self-contextualization carried IN the record itself (review
        requirement: a consumer reading the JSON alone must not see a
        bare miss with no signal it is expected). Populated for the
        cross-vintage variant; `None` where the bare verdict is
        self-explanatory.
    """

    variant_label: str
    case_class: str
    theta_au: float
    theta_au_uncertainty: float | None
    theta_source: str
    predicted_slope_hz_mm2_per_v: float
    predicted_slope_band_lo: float
    predicted_slope_band_hi: float
    measured_slope_hz_mm2_per_v: float
    measured_slope_lo: float
    measured_slope_hi: float
    residual_hz_mm2_per_v: float
    residual_fractional: float
    bands_overlap: bool
    kpi_verdict: str
    verdict_note: str | None


@dataclass(frozen=True)
class RoosQuadrupoleSlopeCase:
    """The full Roos-slope benchmark case: both Theta variants plus the
    structural 24/5 pin and the offset's (non-)role.

    Attributes
    ----------
    structural_two_ion_enhancement_ratio : float
        :func:`structural_two_ion_enhancement_ratio`'s output -- must
        equal `+24/5` (positive) to float precision (module docstring
        step 2).
    cross_vintage, arithmetic_reproduction : QuadrupoleSlopeVariant
        The two labeled variants (module docstring; cross-vintage is the
        headline presentation per G8 sign-off B4).
    theta_roundtrip_check_au : float
        :func:`roos_theta_au_roundtrip` applied to the arithmetic-
        reproduction variant's `theta_au` (1.83) -- must equal 1.83 to
        float64 precision (the closed-loop factor-consistency check,
        module docstring step 3).
    offset_hz, offset_uncertainty_hz : float
        Roos's Fig. 4a fitted offset (`loaders.ROOS_FIT_OFFSET_HZ`) --
        carried for citation completeness ONLY, never combined with
        either variant's slope (module docstring step 4 explains why).
    kpi_summary_impact : str
        A fixed, binding statement that this case leaves
        `benchmarks/results/wp10_results.json`'s `reproducibility_cases_total`
        (1) and `blind_prediction_cases_total` (0) unchanged.
    dossier_citation : str
        `_DOSSIER_CITATION`, verbatim.
    """

    structural_two_ion_enhancement_ratio: float
    cross_vintage: QuadrupoleSlopeVariant
    arithmetic_reproduction: QuadrupoleSlopeVariant
    theta_roundtrip_check_au: float
    offset_hz: float
    offset_uncertainty_hz: float
    kpi_summary_impact: str
    dossier_citation: str


def _build_variant(
    *,
    variant_label: str,
    case_class: str,
    theta_au: float,
    theta_au_uncertainty: float | None,
    theta_source: str,
    verdict_note_if_not_met: str | None = None,
) -> QuadrupoleSlopeVariant:
    """Build one `QuadrupoleSlopeVariant`, calling the real engine chain
    (`predicted_slope_hz_mm2_per_v`, itself built entirely on
    `quadrupole_shift_joules`) for the nominal value and, if an
    uncertainty is given, for the +/- band -- three (or one) direct calls,
    never a derivative estimate.
    """
    nominal = predicted_slope_hz_mm2_per_v(theta_au)
    if theta_au_uncertainty is None:
        band_lo = band_hi = nominal
    else:
        at_lo = predicted_slope_hz_mm2_per_v(theta_au - theta_au_uncertainty)
        at_hi = predicted_slope_hz_mm2_per_v(theta_au + theta_au_uncertainty)
        band_lo, band_hi = min(at_lo, at_hi), max(at_lo, at_hi)

    measured = loaders.ROOS_MEASURED_SLOPE_HZ_MM2_PER_V
    residual = nominal - measured.nominal
    overlap = run_benchmarks._bands_overlap(  # noqa: SLF001 -- reusing the tested helper
        band_lo, band_hi, measured.lo, measured.hi
    )

    return QuadrupoleSlopeVariant(
        variant_label=variant_label,
        case_class=case_class,
        theta_au=theta_au,
        theta_au_uncertainty=theta_au_uncertainty,
        theta_source=theta_source,
        predicted_slope_hz_mm2_per_v=nominal,
        predicted_slope_band_lo=band_lo,
        predicted_slope_band_hi=band_hi,
        measured_slope_hz_mm2_per_v=measured.nominal,
        measured_slope_lo=measured.lo,
        measured_slope_hi=measured.hi,
        residual_hz_mm2_per_v=residual,
        residual_fractional=residual / measured.nominal,
        bands_overlap=overlap,
        kpi_verdict="MET" if overlap else "NOT MET",
        verdict_note=None if overlap else verdict_note_if_not_met,
    )


def run_roos_quadrupole_slope_case() -> RoosQuadrupoleSlopeCase:
    """Build the full Roos-slope benchmark case (see module docstring for
    the complete method).

    Returns
    -------
    RoosQuadrupoleSlopeCase
    """
    moment = QUADRUPOLE_MOMENTS["Ca+:D5/2"]
    assert moment.theory_theta_au is not None  # narrows Optional for mypy
    assert moment.theory_source is not None

    cross_vintage = _build_variant(
        variant_label=CROSS_VINTAGE_CASE_LABEL,
        case_class="cross_vintage_comparison",
        theta_au=moment.theory_theta_au,
        theta_au_uncertainty=None,
        theta_source=moment.theory_source,
        verdict_note_if_not_met=(
            "NOT MET is the EXPECTED outcome for this variant: the "
            "residual recovers the literature-known ~4.7% Theta "
            "theory-vs-measurement tension (Itano theory 1.917 ea0^2 vs "
            "Roos's measured 1.83(1) ea0^2), validating the engine's "
            "factor chain while isolating the coefficient. It is not an "
            "engine defect. See the case write-up in "
            "benchmarks/RESULTS.md and the module docstring."
        ),
    )
    arithmetic_reproduction = _build_variant(
        variant_label=ARITHMETIC_REPRODUCTION_CASE_LABEL,
        case_class="arithmetic_reproduction",
        theta_au=moment.theta_au,
        theta_au_uncertainty=moment.theta_au_uncertainty,
        theta_source=moment.source,
    )
    roundtrip = roos_theta_au_roundtrip(moment.theta_au)
    offset = loaders.ROOS_FIT_OFFSET_HZ

    return RoosQuadrupoleSlopeCase(
        structural_two_ion_enhancement_ratio=structural_two_ion_enhancement_ratio(),
        cross_vintage=cross_vintage,
        arithmetic_reproduction=arithmetic_reproduction,
        theta_roundtrip_check_au=roundtrip,
        offset_hz=offset.nominal,
        offset_uncertainty_hz=offset.hi - offset.nominal,
        kpi_summary_impact=(
            "Does not increment benchmarks/results/wp10_results.json's "
            "reproducibility_cases_total (stays 1) or "
            "blind_prediction_cases_total (stays 0); tracked separately, "
            "same pattern as benchmarks/run_bbr_jila_arithmetic_reproduction.py."
        ),
        dossier_citation=_DOSSIER_CITATION,
    )


def build_report() -> dict[str, Any]:
    """Build the full Roos-slope benchmark report as a JSON-serializable dict."""
    case = run_roos_quadrupole_slope_case()
    return {
        "roos_quadrupole_slope_schema": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "cross_vintage_case_label": CROSS_VINTAGE_CASE_LABEL,
        "arithmetic_reproduction_case_label": ARITHMETIC_REPRODUCTION_CASE_LABEL,
        "roos_quadrupole_slope_case": asdict(case),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the Roos-slope benchmark case as a markdown summary,
    mirroring `run_bbr_jila_arithmetic_reproduction.render_markdown`'s style.
    """
    case = report["roos_quadrupole_slope_case"]
    cv = case["cross_vintage"]
    ar = case["arithmetic_reproduction"]
    lines = [
        "# Roos-slope ion benchmark case (generated)",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Structural pin: two-ion enhancement",
        "",
        (
            f"`hbar*Delta_1 / Delta_E_Q(-5/2)` = "
            f"{case['structural_two_ion_enhancement_ratio']:+.6f} "
            "(must equal +24/5 = +4.8)"
        ),
        "",
        "## Headline: cross-vintage comparison (independent theory Theta)",
        "",
        f"**Classification label (binding, G8 sign-off B4): {CROSS_VINTAGE_CASE_LABEL}**",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Theta (Itano theory) | {cv['theta_au']} ea0^2 (no published uncertainty) |",
        f"| Predicted slope \\|a_pred\\| | {cv['predicted_slope_hz_mm2_per_v']:.6f} Hz*mm^2/V |",
        f"| Measured slope (Roos Fig. 4a) | {cv['measured_slope_hz_mm2_per_v']} +/- "
        f"{cv['measured_slope_hi'] - cv['measured_slope_hz_mm2_per_v']:.3f} Hz*mm^2/V |",
        f"| Residual (predicted - measured) | {cv['residual_hz_mm2_per_v']:+.6f} Hz*mm^2/V "
        f"({cv['residual_fractional']:+.4%}) |",
        f"| Bands overlap | {cv['bands_overlap']} |",
        f"| **kpi_verdict** | **{cv['kpi_verdict']}** (expected: recovers the known "
        "Theta theory-vs-measurement tension, not an engine defect) |",
        "",
        "## Secondary: arithmetic reproduction (Roos's own Theta, circular)",
        "",
        f"**Classification label (binding, G8 sign-off B4): {ARITHMETIC_REPRODUCTION_CASE_LABEL}**",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Theta (Roos's own extraction) | {ar['theta_au']} +/- "
        f"{ar['theta_au_uncertainty']} ea0^2 |",
        f"| Predicted slope \\|a_pred\\| | {ar['predicted_slope_hz_mm2_per_v']:.6f} Hz*mm^2/V |",
        f"| Predicted slope band | [{ar['predicted_slope_band_lo']:.6f}, "
        f"{ar['predicted_slope_band_hi']:.6f}] Hz*mm^2/V |",
        f"| Measured slope (Roos Fig. 4a) | {ar['measured_slope_hz_mm2_per_v']} +/- "
        f"{ar['measured_slope_hi'] - ar['measured_slope_hz_mm2_per_v']:.3f} Hz*mm^2/V |",
        f"| Residual (predicted - measured) | {ar['residual_hz_mm2_per_v']:+.6f} Hz*mm^2/V "
        f"({ar['residual_fractional']:+.4%}) |",
        f"| Bands overlap | {ar['bands_overlap']} |",
        f"| **kpi_verdict** | **{ar['kpi_verdict']}** |",
        f"| Theta round-trip check ((5/12)*h*a_pred) | {case['theta_roundtrip_check_au']:.6f} "
        f"(input {ar['theta_au']:.6f}; equality pinned to 1e-9 absolute in "
        "`tests/test_roos_benchmark.py`) |",
        "",
        "## Offset (documentation only, not part of the slope comparison)",
        "",
        f"Delta_0/(2*pi) = {case['offset_hz']:.1f} +/- {case['offset_uncertainty_hz']:.1f} Hz "
        "at zero applied gradient; not combined with either slope prediction above "
        "(module docstring step 4: partly second-order Zeeman, a mechanism this engine "
        "does not model, partly an uncharacterized residual stray field).",
        "",
        "## KPI-summary impact",
        "",
        case["kpi_summary_impact"],
        "",
        f"Source: {case['dossier_citation']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the Roos-slope benchmark case and write
    `benchmarks/results/roos_quadrupole_slope.json` and a generated
    markdown summary alongside it."""
    report = build_report()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / "roos_quadrupole_slope.json"
    md_path = _RESULTS_DIR / "roos_quadrupole_slope.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
