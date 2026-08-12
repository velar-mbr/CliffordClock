# SPDX-License-Identifier: AGPL-3.0-or-later
"""Atomic species registry for the ensemble sampler.

Four species are supported: ``"Sr87"``, ``"Yb171"``, ``"Al27+"``, and
``"In115+"``. Values are cited to CODATA/NIST/BIPM published data in each
registry entry below; a pin test in ``tests/test_ensemble_species.py``
guards against silent transcription drift.

The registry also carries optional differential DC-Stark polarizability
data (CONVENTIONS.md E14b): `Species.delta_alpha_dc_si` and/or
`Species.stark_coefficient_hz_per_v2_m2`, plus the a.u.->SI conversion
constant `ALPHA_AU_TO_SI` and the `StarkCoefficients` override type, used
by `cliffordclock.integrator.omega.pivot_perturbation_stark`. Pin tests
for this data live in `tests/test_stark_species.py`.

WP20 scope (CONVENTIONS.md E32, blackbody-radiation shift): the registry
additionally carries optional per-species BBR static/dynamic coefficients,
`Species.bbr_coefficients` (`BbrCoefficients`), consumed by
`cliffordclock.integrator.omega.bbr_pivot_perturbation`/
`bbr_pivot_uncertainty`. Provenance: the Sr-87 dynamic polynomial is the
PTB-2025 rescaling (Nosske et al., arXiv:2507.14030) of the Lisdat et
al., PRR 3, L042036 (2021) fit shape onto the Aeppli et al., PRL 133,
023401 (2024) anchor; the static coefficient is Middelmann et al.,
Phys. Rev. Lett. 109, 263004 (2012); each entry below carries its own
citation, per the project's G7 theory sign-off record. Pin tests live in
`tests/test_bbr_species.py`.

WP21 scope (CONVENTIONS.md E34/E35, ion-clock support, Tiers 1+2): two
J=0->J=0 ion-clock species are added (`In115+`, and `Al27+` above now
carries `delta_alpha_dc_si`), plus a separate D/F-state electric-
quadrupole-moment registry, `QUADRUPOLE_MOMENTS` (`QuadrupoleMoment`),
consumed by `cliffordclock.integrator.omega.quadrupole_pivot_perturbation`.
No `BbrCoefficients` are populated for either ion species: the dossier
gives only a single BBR data point per species (Al27+: one measured shift
at one temperature; In115+: one theory fractional value at 300 K), not
the independent static/dynamic split `BbrCoefficients` requires -- per
the G8 gate's explicit instruction, that split is NOT invented here (see
each species' docstring). Provenance: Roos et al., Nature 443, 316
(2006), quant-ph/0701215v1 (Eq. 1/Fig. 4a, the Ca+ measured quadrupole
moment); Itano, Phys. Rev. A 73, 022510 (2006) (the Ca+ theory value);
the per-entry citations below for every other coefficient; and the
project's G8 theory sign-off record.
Pin tests live in `tests/test_ion_species.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cliffordclock import constants

#: Atomic-unit -> SI conversion factor for static electric dipole
#: polarizability (CONVENTIONS.md E14b): ``alpha[SI, C^2 m^2 J^-1] =
#: alpha[a.u.] * ALPHA_AU_TO_SI``. Exact definition: ``4 pi eps0 a0^3``
#: (Gaussian/a.u. polarizability expressed in SI units of C^2 m^2 J^-1,
#: equivalently C.m^2.V^-1 since J = C.V).
#:
#: CODATA-derived value: with CODATA 2022 ``eps0 = 8.8541878128e-12 F/m``
#: (vacuum electric permittivity) and ``a0 = 5.29177210544e-11 m`` (Bohr
#: radius, CODATA 2022), ``4*pi*eps0*a0**3 = 1.64877727436e-41`` (the
#: CODATA "atomic unit of electric polarizability").
#:
#: Pinned at full precision (independently reviewed and confirmed, see
#: CONVENTIONS.md section 11): literature polarizabilities carry ~6-7
#: significant figures, so this conversion constant must not be the
#: precision-limiting term. History: an early transcription of this
#: constant carried a digit-swap (``...772``); caught and corrected before
#: release, and independently recomputed from CODATA to confirm the fix.
ALPHA_AU_TO_SI = 1.64877727436e-41

#: Atomic-unit -> SI conversion factor for electric quadrupole moment
#: (CONVENTIONS.md E34, WP21 Tier 2, G8 sign-off gate edit 2): ``Theta_SI
#: [C m^2] = Theta[a.u.] * EA0_SQUARED_SI``, i.e. ``e * a0^2`` -- NOT an
#: additional factor of `e` on top (the "double-e" trap the gate edit
#: names, the direct analog of `ALPHA_AU_TO_SI`'s digit-swap lesson: see
#: CONVENTIONS.md section 14's unit-pin note). Computed, not hand-
#: transcribed, from `cliffordclock.constants.ELEMENTARY_CHARGE` and
#: `cliffordclock.constants.BOHR_RADIUS` (both already CODATA 2022,
#: exact-by-definition for `ELEMENTARY_CHARGE`) -- deriving it from the
#: same two pinned constants this module already trusts for
#: `ALPHA_AU_TO_SI` avoids a second independent transcription surface for
#: essentially the same physical combination (`e * a0^n`). Numerically
#: ``4.4865515185255e-40 C m^2``, matching the G8 sign-off's quoted
#: ``4.4866e-40`` (their rounding) and ``4.486551e-40`` (their fuller
#: precision, computed from a slightly older CODATA a0 -- 5.29177210903e-11
#: vs this module's CODATA-2022 5.29177210544e-11 -- differing only in the
#: 9th significant figure, far below any tabulated Theta's precision).
#: `tests/test_ion_species.py` pins this value and independently verifies
#: it applies `e` exactly once (dimensional round-trip: `[C m^2] . [V/m^2]
#: = J`, CONVENTIONS.md E34).
EA0_SQUARED_SI = constants.ELEMENTARY_CHARGE * constants.BOHR_RADIUS**2

#: Reference temperature T0 for the BBR fractional-shift polynomial (E32),
#: kelvin. Middelmann et al. PRL 109, 263004 (2012) and every downstream
#: paper (Lisdat 2021, Aeppli 2024, Hassan 2025) normalize their static/
#: dynamic split at 300 K; dossier Sec.1.
BBR_REFERENCE_TEMPERATURE_K = 300.0

#: Hard validity window for `environment.radiation_temperature_K` (WP20
#: Gate G7 edit 5): both species' registry fits are published over
#: 50-350 K (Lisdat 2021 Eq. 6-7 fit range; dossier Sec.2/Sec.7 registry
#: recommendation; Yb's cryo end at 77 K, within this window, is
#: separately anchored by Hassan 2025). Outside this range the pipeline
#: raises `PipelineConfigError` rather than silently extrapolating a fit
#: past its support (G7 sign-off B4: "silently extrapolating a fit past
#: its support is exactly how wrong clock corrections get made").
BBR_VALIDITY_MIN_K = 50.0
BBR_VALIDITY_MAX_K = 350.0

#: Upper edge of the PTB<->JILA 1e-19-class cross-verification band (G7
#: sign-off B4 "RATIFY-WITH-EDIT"): T <= this value carries full 1e-19-
#: class confidence; `BBR_VALIDITY_MIN_K` <= T <= `BBR_VALIDITY_MAX_K` but
#: above this value is in-fit-range (the polynomial is still evaluated)
#: but beyond the PTB<->JILA cross-verification statement -- the pipeline
#: report carries an explicit note in that band rather than silently
#: reusing the same confidence claim (dossier Sec.7; arXiv:2507.14030
#: "agree ... within 1e-19 for T <= 300 K").
BBR_CROSS_VERIFIED_MAX_K = 300.0


@dataclass(frozen=True)
class BbrCoefficients:
    """Per-species blackbody-radiation shift coefficients (CONVENTIONS.md E32).

    ``(P-1)_BBR = [nu_stat_300k_hz*(T/T0)^4 + sum_n dyn_coeffs_hz[n]*(T/T0)^n]
    / nu_0``, ``T0 = BBR_REFERENCE_TEMPERATURE_K``. See
    `cliffordclock.integrator.omega.bbr_pivot_perturbation` for the formula
    implementation and `bbr_pivot_uncertainty` for the coefficient/
    temperature uncertainty propagation (G7 sign-off A4#2-3).

    Attributes
    ----------
    nu_stat_300k_hz : float
        Static (T^4) BBR shift at T0, hertz (negative: the clock runs
        slow in the thermal bath, same sign convention as E14b/Delta_alpha
        > 0).
    nu_stat_300k_uncertainty_hz : float
        1-sigma uncertainty on `nu_stat_300k_hz`, hertz.
    dyn_coeffs_hz : dict[int, float]
        Dynamic-term fit coefficients, keyed by the power of ``(T/T0)``
        they multiply (e.g. ``{6: ..., 8: ..., 10: ...}``). These are FITS
        to the exact Planck-weighted integral (Lisdat et al. PR Research 3,
        L042036 (2021) Eq. 6-7), not a truncated Taylor series -- Lisdat's
        Appendix A proves the naive Taylor series in 1/y has zero
        convergence radius for Sr's dominant transition (dossier Sec.1-2).
    dyn_anchor_uncertainty_hz : float
        Dominant (anchor) uncertainty on the *summed* dynamic term at T0,
        hertz -- not decomposed per coefficient (the published
        uncertainties are for the fitted/anchored sum, not independent
        per-term covariances). Used by `bbr_pivot_uncertainty` scaled by
        the leading dynamic power (T/T0)^6.
    validity_min_k, validity_max_k : float
        Per-species validity window, kelvin. Both registry entries below
        use `BBR_VALIDITY_MIN_K`/`BBR_VALIDITY_MAX_K` (50-350 K); stored
        per-entry (rather than only the module-level constants) so a
        future species with a narrower published fit range can differ.
    cross_verified_max_k : float
        Per-species upper edge of the 1e-19-class cross-verification band
        (G7 sign-off B4); see `BBR_CROSS_VERIFIED_MAX_K`.
    """

    nu_stat_300k_hz: float
    nu_stat_300k_uncertainty_hz: float
    dyn_coeffs_hz: dict[int, float]
    dyn_anchor_uncertainty_hz: float
    validity_min_k: float = BBR_VALIDITY_MIN_K
    validity_max_k: float = BBR_VALIDITY_MAX_K
    cross_verified_max_k: float = BBR_CROSS_VERIFIED_MAX_K


@dataclass(frozen=True)
class QuadrupoleMoment:
    """Electric-quadrupole moment Theta(J) for one D/F-state ion-clock
    level (CONVENTIONS.md E34, WP21 Tier 2).

    A *state* registry, independent of `Species`: none of the D/F-state
    ions below (Ca+, Sr+, Ba+, Yb+) are registered as full `Species`
    entries here (that would require also pinning their own clock-
    transition frequencies, which is not in the WP21 Tier-2 dossier scope
    -- `cliffordclock.pipeline.QuadrupoleConfig` takes `nu_0_hz` as an
    explicit input instead, mirroring `StarkCoefficients`'s "explicit
    override, no fabricated registry entry" pattern for data this project
    does not independently verify).

    Attributes
    ----------
    j : float
        Total electronic angular momentum J of the state (e.g. 2.5 for a
        D5/2 state). `j < 1` (J=0 or J=1/2) states carry no quadrupole
        coupling (CONVENTIONS.md E34's immunity note) --
        `cliffordclock.integrator.omega.quadrupole_mj_factor` raises for
        such a `j`.
    theta_au : float
        Theta(J), atomic units (= e*a0^2, `EA0_SQUARED_SI`). SIGN carries
        physical meaning (CONVENTIONS.md E34's sign-discipline note): the
        D-state entries below are positive; Yb+ F7/2 is negative and
        serves as the registry's sign anchor.
    theta_au_uncertainty : float
        1-sigma uncertainty on `theta_au`, same units.
    source : str
        Citation for the tabulated (measured, or best available) value.
    verification : str
        One of `"primary"`, `"secondary"`, `"secondary-paywalled"` (G8
        sign-off gate edit 7: "Theta table pins carry their secondary/
        paywalled flags in the registry docstrings" -- carried as a field,
        not only prose, so callers/reports can act on it programmatically).
    theory_theta_au, theory_source : float | None, str | None
        An independent theoretical value and its citation, for reviewer
        cross-reference only -- never used as the registry value (mirrors
        `Species`'s docstring convention for the Sr87/Yb171 Delta_alpha
        theory cross-checks).
    """

    j: float
    theta_au: float
    theta_au_uncertainty: float
    source: str
    verification: str
    theory_theta_au: float | None = None
    theory_source: str | None = None


#: D/F-state electric-quadrupole-moment registry (CONVENTIONS.md E34, WP21
#: Tier 2), keyed `"<ion>:<state>"`. Provenance: Roos et al.,
#: Nature 443, 316 (2006), quant-ph/0701215v1, Eq. 1/Fig. 4a (the Ca+
#: measured value, primary-text upgrade); Itano, Phys. Rev. A 73, 022510
#: (2006) (the Ca+ theory value); the per-entry citations below for every
#: other value; G8 sign-off B3 "RATIFY (honor status
#: flags)".
QUADRUPOLE_MOMENTS: dict[str, QuadrupoleMoment] = {
    # Ca+ 3d 2D5/2: measured 1.83(1) ea0^2, Roos, Chwalla, Kim, Riebe,
    # Blatt, "Designer atoms for quantum metrology," quant-ph/0701215v1
    # (published Nature 443, 316 (2006)). PRIMARY VERIFIED 2026-08-11 (owner
    # -supplied preprint, full read in this project's internal review): the
    # earlier draft's
    # "Nature paywalled, secondary via >=2 citing papers" label is
    # superseded -- this is now a direct primary-text read. Uncertainty
    # dominated by the paper's 3-degree angle (beta) determination.
    # Theory 1.917 ea0^2, Itano, Phys. Rev. A 73, 022510 (2006) (citation
    # corrected from the preprint id physics/0512250 to its published PRA
    # form per dossier section 3) and Sur et al., Phys. Rev. Lett. 96,
    # 193001 (2006) -- cross-reference only, not the registry value.
    "Ca+:D5/2": QuadrupoleMoment(
        j=2.5,
        theta_au=1.83,
        theta_au_uncertainty=0.01,
        source="Roos et al., quant-ph/0701215v1 (Nature 443, 316 (2006)), Eq. 1 fit",
        verification="primary",
        theory_theta_au=1.917,
        theory_source="Itano, Phys. Rev. A 73, 022510 (2006)",
    ),
    # Sr+ 4d 2D5/2: measured 2.973(+26/-33) ea0^2, Shaniv, Akerman, Ozeri,
    # Phys. Rev. Lett. 116, 140801 (2016) -- supersedes Barwood et al. 2004's
    # 2.6(3) (dossier section 3); the registry pins the newer vintage per
    # G8 sign-off B3. Theory 2.94(7) ea0^2, Sur et al., Phys. Rev. Lett. 96,
    # 193001 (2006) -- cross-reference only. `theta_au_uncertainty` uses the
    # larger (asymmetric) 0.033 bound conservatively (the source's
    # +0.026/-0.033).
    "Sr+:D5/2": QuadrupoleMoment(
        j=2.5,
        theta_au=2.973,
        theta_au_uncertainty=0.033,
        source="Shaniv, Akerman, Ozeri, Phys. Rev. Lett. 116, 140801 (2016)",
        verification="secondary",
        theory_theta_au=2.94,
        theory_source="Sur et al., Phys. Rev. Lett. 96, 193001 (2006)",
    ),
    # Ba+ 5d 2D5/2: measured 3.229(89) ea0^2, Barrett group, Phys. Rev. A
    # 99, 022515 (2019). SECONDARY (paywalled primary; not independently
    # read by this builder -- G8 sign-off B3: "owner-fetch follow-up").
    # Theory 3.379 ea0^2, Itano (see Ca+ entry's citation) -- cross-
    # reference only.
    "Ba+:D5/2": QuadrupoleMoment(
        j=2.5,
        theta_au=3.229,
        theta_au_uncertainty=0.089,
        source="Barrett group, Phys. Rev. A 99, 022515 (2019)",
        verification="secondary-paywalled",
        theory_theta_au=3.379,
        theory_source="Itano, Phys. Rev. A 73, 022510 (2006)",
    ),
    # Yb+ 5d 2D3/2 (J=3/2, not 5/2 -- the E2-clock D-state, distinct from
    # the F7/2 E3-clock state below): measured 2.08(11) ea0^2, Schneider,
    # Peik, Tamm, Phys. Rev. Lett. 94, 230801 (2005). SECONDARY (paywalled
    # primary; not independently read). Theory 2.174 ea0^2, Itano (see Ca+
    # entry) -- cross-reference only.
    "Yb+:D3/2": QuadrupoleMoment(
        j=1.5,
        theta_au=2.08,
        theta_au_uncertainty=0.11,
        source="Schneider, Peik, Tamm, Phys. Rev. Lett. 94, 230801 (2005)",
        verification="secondary-paywalled",
        theory_theta_au=2.174,
        theory_source="Itano, Phys. Rev. A 73, 022510 (2006)",
    ),
    # Yb+ 4f^13 6s^2 2F7/2 (J=7/2, the E3-clock upper state): measured
    # -0.041(5) ea0^2, Huntemann et al., Phys. Rev. Lett. 108, 090801
    # (2012). PRIMARY. NEGATIVE Theta -- this is the registry's sign
    # anchor for the E34 A1 regression (dossier section 3: "~50-80x
    # smaller than D states; why the E3 budget's quadrupole row is
    # 0(0.3)e-18"): a positive-Theta D-state and this state must produce
    # opposite-sign shifts under an otherwise identical (gradient, m_J,
    # axis), tests/test_quadrupole_pivot.py.
    "Yb+:F7/2": QuadrupoleMoment(
        j=3.5,
        theta_au=-0.041,
        theta_au_uncertainty=0.005,
        source="Huntemann et al., Phys. Rev. Lett. 108, 090801 (2012)",
        verification="primary",
    ),
}


def get_quadrupole_moment(state: str) -> QuadrupoleMoment:
    """Look up a `QuadrupoleMoment` by registry key (e.g. ``"Ca+:D5/2"``).

    Parameters
    ----------
    state : str
        Registry key; see `QUADRUPOLE_MOMENTS` for the valid set.

    Returns
    -------
    QuadrupoleMoment

    Raises
    ------
    KeyError
        If `state` is not a registered key; the error message lists the
        valid keys.
    """
    try:
        return QUADRUPOLE_MOMENTS[state]
    except KeyError:
        valid = ", ".join(sorted(QUADRUPOLE_MOMENTS))
        raise KeyError(f"Unknown quadrupole state {state!r}; valid keys are: {valid}") from None


#: Micromotion-boundary report note (CONVENTIONS.md E34/E35 scope note;
#: G8 sign-off gate edits 5/6, "shipping requirement... test-pinned on
#: every ion-species report"). Keyed by `Species.name`; every entry states
#: the SHARED CAUSE explicitly (one stray field, two pathways, only the
#: smaller one modeled) per the gate's required edit. Al27+/In115+ are
#: J=0 -> J=0 clocks, so the note is at its STRONGEST wording (no
#: first-order quadrupole term exists at all for these species -- the
#: tool characterizes essentially none of their field-sensitive budget),
#: per gate edit 5's "strongest for J=0 ions" requirement.
ION_MICROMOTION_NOTES: dict[str, str] = {
    "Al27+": (
        "Micromotion boundary (out of scope): the same stray DC field this tool's DC-Stark/"
        "quadrupole terms model also displaces the ion from the RF null, producing excess "
        "micromotion (time-dilation + AC-Stark from the trap RF) -- a separate, unmodeled "
        "pathway sourced by the SAME field. Al27+ is J=0 -> J=0: no first-order quadrupole "
        "term exists at all, so this tool characterizes essentially none of the ion's "
        "field-sensitive systematic budget, which is dominated by excess micromotion "
        "-45.8(5.9)e-19 + secular motion -17.3(2.9)e-19 (Brewer et al., Phys. Rev. Lett. "
        "123, 033201 (2019), PRIMARY). A DC-Stark number alone is not a stray-field budget "
        "(canonical treatment: Berkeland, Miller, Bergquist, Itano, Wineland, J. Appl. Phys. "
        "83, 5025 (1998); mechanism restated in Ludlow et al., Rev. Mod. Phys. 87, 637 "
        "(2015))."
    ),
    "In115+": (
        "Micromotion boundary (out of scope): the same stray DC field this tool's DC-Stark/"
        "quadrupole terms model also displaces the ion from the RF null, producing excess "
        "micromotion (time-dilation + AC-Stark from the trap RF) -- a separate, unmodeled "
        "pathway sourced by the SAME field. In115+ is J=0 -> J=0: no first-order quadrupole "
        "term exists at all, so this tool characterizes essentially none of the ion's "
        "field-sensitive systematic budget (canonical treatment: Berkeland, Miller, "
        "Bergquist, Itano, Wineland, J. Appl. Phys. 83, 5025 (1998); mechanism restated in "
        "Ludlow et al., Rev. Mod. Phys. 87, 637 (2015); representative multi-ion-clock "
        "micromotion/quadrupole comparison, Huntemann et al., Phys. Rev. Lett. 108, 090801 "
        "(2012): quadrupole 0(0.3)e-18 (averaged away) vs. residual motion -3.7(2.1)e-18, "
        "~7x larger)."
    ),
}

#: Hyperfine-mediated electric-quadrupole (E2) budget-line note
#: (CONVENTIONS.md E34's Tier-1 caveat; G8 sign-off gate edit 4/A4:
#: "a report budget LINE-ITEM, not a footnote; never state
#: 'J=0 quadrupole-immune' without the I!=0 qualifier"). J=0 states carry
#: no FIRST-ORDER quadrupole coupling (E34's immunity note) but nuclear
#: spin I!=0 mixes in a small SECOND-order hyperfine-mediated effective
#: quadrupole moment (Beloy, Leibrandt, Itano, Phys. Rev. A 95, 043405
#: (2017)) -- above the 1e-19 floor for the affected species, so it must
#: be an explicit line-item, not silently implied away by "J=0 immune"
#: language.
ION_HYPERFINE_E2_BUDGET_NOTES: dict[str, str] = {
    "Al27+": (
        "Hyperfine-mediated E2 budget line (Beloy, Leibrandt, Itano, Phys. Rev. A 95, 043405 "
        "(2017)): J=0 has no first-order quadrupole moment, but nuclear spin I=5/2 mixes in a "
        "small second-order hyperfine-mediated effective quadrupole shift. Not computed by "
        "this tool (out of E34 scope: E34 is the first-order fine-structure quadrupole for "
        "J>=1 upper states). Dossier: 'Al27+'s nearly cancels' -- not independently quantified "
        "here; do not read this species' shift as exactly quadrupole-immune."
    ),
    "In115+": (
        "Hyperfine-mediated E2 budget line (Beloy, Leibrandt, Itano, Phys. Rev. A 95, 043405 "
        "(2017)): J=0 has no first-order quadrupole moment, but nuclear spin I=9/2 mixes in a "
        "small second-order hyperfine-mediated effective quadrupole shift. Not computed by "
        "this tool (out of E34 scope: E34 is the first-order fine-structure quadrupole for "
        "J>=1 upper states); a representative In+/Yb+ Coulomb-crystal evaluation budgets "
        "-1.4(3)e-19 for it (arXiv:2402.16807, 2024), above the 1e-19 floor -- do not read "
        "this species' shift as exactly quadrupole-immune."
    ),
}


@dataclass(frozen=True)
class Species:
    """An atomic (or ionic) species relevant to optical-clock ensembles.

    Attributes
    ----------
    name : str
        Registry key, e.g. ``"Sr87"``.
    mass_kg : float
        Atomic (or ionic) mass, kilograms.
    clock_frequency_hz : float
        Clock transition frequency, hertz.
    clock_wavelength_m : float
        Clock transition vacuum wavelength, meters (``c / clock_frequency_hz``).
    charge_state : int
        Net charge in units of the elementary charge (0 for a neutral atom,
        +1 for a singly charged positive ion).
    delta_alpha_dc_si : float | None
        Differential static scalar polarizability of the clock transition,
        ``Delta alpha = alpha(excited) - alpha(ground)`` (CONVENTIONS.md
        E14b), units C^2 m^2 J^-1 (equivalently C.m^2.V^-1). `None` if no
        published scalar Delta-alpha value is populated for this species
        (Sprint 1/WP7 scope: J=0 -> J=0 lattice clocks only; ion clocks
        need tensor/quadrupole terms, out of MVP scope per E14b's scope
        note -- see `Al27+`). Use `Species.resolve_stark_coefficient_hz_per_v2_m2`
        rather than reading this field directly when only the Stark
        coefficient is needed, since it also covers the
        `stark_coefficient_hz_per_v2_m2`-only case.
    stark_coefficient_hz_per_v2_m2 : float | None
        Equivalent Stark coefficient ``k_S`` such that
        ``P - 1 = k_S |E|^2 / nu_0`` (CONVENTIONS.md E14b "equivalent
        per-species input"), units Hz.m^2.V^-2. For the registry entries
        below this is *derived* from `delta_alpha_dc_si` as
        ``k_S = -Delta_alpha / (2h)`` (never independently transcribed,
        to avoid a second transcription-error surface for the same
        physical quantity); `None` under the same condition as
        `delta_alpha_dc_si`.
    bbr_coefficients : BbrCoefficients | None
        Blackbody-radiation shift coefficients (CONVENTIONS.md E32, WP20).
        `None` if no published BBR fit is populated for this species (the
        `Al27+` case: no ion-clock BBR evaluation is in scope here, mirroring
        `delta_alpha_dc_si`'s `None` for the same species). Use
        `Species.resolve_bbr_coefficients` rather than reading this field
        directly for a clear error message.
    """

    name: str
    mass_kg: float
    clock_frequency_hz: float
    clock_wavelength_m: float
    charge_state: int
    delta_alpha_dc_si: float | None = None
    stark_coefficient_hz_per_v2_m2: float | None = None
    bbr_coefficients: BbrCoefficients | None = None

    def resolve_bbr_coefficients(self) -> BbrCoefficients:
        """Return this species' `BbrCoefficients` (CONVENTIONS.md E32), or raise clearly.

        Raises
        ------
        ValueError
            If `bbr_coefficients` is not populated (the `Al27+` case in
            this registry: no BBR fit is in scope for ion clocks here).
        """
        if self.bbr_coefficients is not None:
            return self.bbr_coefficients
        raise ValueError(
            f"Species {self.name!r} has no BBR shift data (CONVENTIONS.md E32): "
            "bbr_coefficients is not populated in the registry. The WP20 registry "
            "covers Sr87 and Yb171 only (J=0 -> J=0 lattice clocks with a "
            "published static/dynamic BBR fit); ion-clock BBR evaluation is out "
            "of MVP scope."
        )

    def resolve_stark_coefficient_hz_per_v2_m2(self) -> float:
        """Return ``k_S`` (Hz.m^2.V^-2, CONVENTIONS.md E14b), or raise clearly.

        Prefers `stark_coefficient_hz_per_v2_m2` if populated; otherwise
        derives it from `delta_alpha_dc_si` via ``k_S = -Delta_alpha /
        (2h)``. Raises `ValueError` if neither field is populated (the
        `Al27+` case in Sprint 1/WP7: ion-clock differential static
        polarizability needs tensor/quadrupole terms, out of MVP scope
        per CONVENTIONS.md E14b's scope note).

        Returns
        -------
        float
            ``k_S``, Hz.m^2.V^-2.

        Raises
        ------
        ValueError
            If this species has no DC-Stark polarizability data.
        """
        if self.stark_coefficient_hz_per_v2_m2 is not None:
            return self.stark_coefficient_hz_per_v2_m2
        if self.delta_alpha_dc_si is not None:
            return -self.delta_alpha_dc_si / (2.0 * constants.PLANCK_H)
        raise ValueError(
            f"Species {self.name!r} has no DC-Stark polarizability data "
            "(CONVENTIONS.md E14b): neither delta_alpha_dc_si nor "
            "stark_coefficient_hz_per_v2_m2 is populated. Sprint 1/WP7 "
            "scope covers J=0 -> J=0 lattice clocks (Sr87, Yb171) only; "
            "ion-clock differential static polarizability needs "
            "tensor/electric-quadrupole terms, explicitly out of MVP "
            "scope (E14b scope note). Supply an explicit "
            "cliffordclock.ensemble.species.StarkCoefficients override "
            "instead of the species registry if you have a value for "
            f"{self.name!r} from another source."
        )


@dataclass(frozen=True)
class StarkCoefficients:
    """Explicit DC-Stark coupling data, bypassing the species registry.

    Lets a caller supply `delta_alpha_dc_si` / `stark_coefficient_hz_per_v2_m2`
    and the clock frequency directly (CONVENTIONS.md E14b), e.g. for a
    species not in the registry (including `Al27+`, until a scalar
    Delta-alpha value or a tensor-polarizability treatment lands), or a
    newer measurement than the pinned registry entry. Mirrors
    `Species`'s two-field representation and
    `resolve_stark_coefficient_hz_per_v2_m2` API so
    `cliffordclock.integrator.omega.pivot_perturbation_stark` can accept
    either interchangeably.

    Parameters
    ----------
    clock_frequency_hz : float
        Clock transition frequency, hertz (E14b's ``nu_0``; the pivot
        denominator is ``h nu_0``, not ``m_e c^2``).
    delta_alpha_dc_si : float | None
        Differential static scalar polarizability, C^2 m^2 J^-1.
    stark_coefficient_hz_per_v2_m2 : float | None
        Equivalent Stark coefficient ``k_S``, Hz.m^2.V^-2.

    Raises
    ------
    ValueError
        If neither coefficient field is given, or if both are given but
        mutually inconsistent (``k_S != -Delta_alpha / (2h)``, checked to
        a loose 1e-6 relative tolerance -- this is a sanity check on
        caller-supplied data, not a precision-critical computation).
    """

    clock_frequency_hz: float
    delta_alpha_dc_si: float | None = None
    stark_coefficient_hz_per_v2_m2: float | None = None

    def __post_init__(self) -> None:
        if self.delta_alpha_dc_si is None and self.stark_coefficient_hz_per_v2_m2 is None:
            raise ValueError(
                "StarkCoefficients requires at least one of delta_alpha_dc_si "
                "or stark_coefficient_hz_per_v2_m2 (CONVENTIONS.md E14b)."
            )
        if self.delta_alpha_dc_si is not None and self.stark_coefficient_hz_per_v2_m2 is not None:
            implied = -self.delta_alpha_dc_si / (2.0 * constants.PLANCK_H)
            if not math.isclose(implied, self.stark_coefficient_hz_per_v2_m2, rel_tol=1e-6):
                raise ValueError(
                    "StarkCoefficients: delta_alpha_dc_si and "
                    "stark_coefficient_hz_per_v2_m2 are mutually inconsistent "
                    f"(k_S implied by delta_alpha_dc_si = {implied!r}, given "
                    f"stark_coefficient_hz_per_v2_m2 = "
                    f"{self.stark_coefficient_hz_per_v2_m2!r}); CONVENTIONS.md "
                    "E14b: k_S = -Delta_alpha / (2h)."
                )

    def resolve_stark_coefficient_hz_per_v2_m2(self) -> float:
        """Return ``k_S`` (Hz.m^2.V^-2, CONVENTIONS.md E14b).

        Mirrors `Species.resolve_stark_coefficient_hz_per_v2_m2`;
        `__post_init__` already guarantees at least one field is
        populated, so this never raises.
        """
        if self.stark_coefficient_hz_per_v2_m2 is not None:
            return self.stark_coefficient_hz_per_v2_m2
        assert self.delta_alpha_dc_si is not None  # guaranteed by __post_init__
        return -self.delta_alpha_dc_si / (2.0 * constants.PLANCK_H)


def _species_from_amu_and_frequency(
    name: str,
    mass_amu: float,
    clock_frequency_hz: float,
    charge_state: int,
    delta_alpha_dc_si: float | None = None,
    bbr_coefficients: BbrCoefficients | None = None,
) -> Species:
    """Build a `Species`, deriving mass in kg, wavelength, and (if
    `delta_alpha_dc_si` is given) the Stark coefficient from the inputs.
    """
    stark_coefficient_hz_per_v2_m2 = (
        None if delta_alpha_dc_si is None else -delta_alpha_dc_si / (2.0 * constants.PLANCK_H)
    )
    return Species(
        name=name,
        mass_kg=mass_amu * constants.ATOMIC_MASS_UNIT,
        clock_frequency_hz=clock_frequency_hz,
        clock_wavelength_m=constants.SPEED_OF_LIGHT / clock_frequency_hz,
        charge_state=charge_state,
        delta_alpha_dc_si=delta_alpha_dc_si,
        stark_coefficient_hz_per_v2_m2=stark_coefficient_hz_per_v2_m2,
        bbr_coefficients=bbr_coefficients,
    )


#: Sr-87 (strontium-87), neutral-atom optical lattice clock species.
#:
#: Mass: 86.908 877 497(50) u. Source: M. Wang, W.J. Huang, F.G. Kondev,
#: G. Audi, S. Naimi, "The AME 2020 atomic mass evaluation (II)", Chinese
#: Physics C 45, 030003 (2021).
#:
#: Clock transition: 5s^2 1S0 -> 5s5p 3P0 (F=9/2 -> F=9/2), frequency
#: 429 228 004 229 873.4 Hz. Source: BIPM "List of standard frequencies"
#: (secondary representation of the second based on the 87Sr transition,
#: mise en pratique for the definition of the second, CCTF), consistent
#: with S.L. Campbell et al., "A Fermi-degenerate three-dimensional optical
#: lattice clock", Science 358, 90 (2017) and T.L. Nicholson et al.,
#: "Systematic evaluation of an atomic clock at 2e-18 total uncertainty",
#: Nat. Commun. 6, 6896 (2015).
#:
#: Differential static (dc) scalar polarizability (CONVENTIONS.md E14b):
#: Delta_alpha = 4.07873(11) x 1e-39 C^2 m^2 J^-1 (measured on 88Sr; the
#: paper's Eq. 4 isotope-shift calculation shows the correction for 87Sr
#: is a fractional change of -1.2e-6, negligible at this precision --
#: applied here as-is). Primary source: T. Middelmann, S. Falke, C.
#: Lisdat, U. Sterr, "High Accuracy Correction of Blackbody Radiation
#: Shift in an Optical Lattice Clock", Phys. Rev. Lett. 109, 263004
#: (2012), arXiv:1208.2848, Eq. (1) sign convention
#: (Delta-nu_BBR = -(1/2h) Delta_alpha <E^2>) and main text
#: "Delta_alpha = 4.07873(11) x 10^-39 Cm^2/V". Independent second
#: (theoretical) source, cited in the same paper as a cross-check:
#: S.G. Porsev and coauthors, Delta_alpha = 4.305(59) x 1e-39 Cm^2/V
#: (Middelmann et al. cite this as "more than 3-sigma from our value,
#: still within typical calculated/measured deviation" -- included here
#: for reviewer cross-reference, not used as the registry value).
#:
#: Blackbody-radiation shift coefficients (CONVENTIONS.md E32, WP20 Gate
#: G7 sign-off B2 "RATIFY", registry recommendation dossier Sec.7):
#: static Delta-nu_stat(300 K) = -2.13023(6) Hz, UNCHANGED across the
#: Lisdat/Aeppli/PTB revisions (only the dynamic term moved) -- primary
#: source T. Middelmann et al., PRL 109, 263004 (2012), same paper/value
#: as `delta_alpha_dc_si` above. Uncertainty: the "(6)" is the last
#: quoted digit, i.e. +/-0.00006 Hz (G7 sign-off A4#2 correction to the
#: theory brief's arithmetic, which misread it as +/-6 mHz -> 1.4e-17;
#: the correct fractional contribution is 0.00006/429.228e12 ~ 1.4e-19,
#: consistent with Lisdat 2021's stated *total* atomic-response
#: uncertainty of 1e-18 at 300 K, which a static-term-alone 1.4e-17 would
#: contradict).
#:
#: Dynamic term: the PTB-2025 rescaled polynomial (arXiv:2507.14030,
#: Kim/Aeppli/Ye group -- the Lisdat-paper authors' own follow-up),
#: {6: -0.13216, 8: -0.01231, 10: -0.00858} Hz (sum at 300 K:
#: -153.05 mHz), stated by that paper to "agree with the full calculation
#: in [Aeppli 2024] within 1e-19 for T <= 300 K". **Shape-vs-anchor
#: reasoning** (G7 sign-off B2 required edit): the temperature *shape* of
#: the dynamic term comes from the atomic transition spectrum (Lisdat et
#: al., PR Research 3, L042036 (2021), Eq. 6-7 fit to the exact
#: Planck-weighted integral -- fixed, transition-spectrum-determined
#: physics), while the overall *scale* is set by the (5s4d)^3D1 state's
#: radiative lifetime, which Aeppli et al. (arXiv:2403.10664, remeasured
#: tau = 2.156(5) us) pinned more precisely than Lisdat's original input
#: -- so rescaling Lisdat's published *shape* to the Aeppli-anchored
#: *scale* (exactly what arXiv:2507.14030 does) is the physically correct
#: operation, not an ad hoc reconciliation, and it is the community's own
#: published choice (dossier Sec.7: JILA 2024, PTB 2025, USTC 2026 all
#: follow this lineage; nobody still uses the pre-2021 -148.7(7) mHz
#: value). Uncertainty: dominated by the Aeppli anchor, +/-0.33 mHz at
#: 300 K (dossier Sec.2) -> ~7.7e-19 fractional -- see
#: `BbrCoefficients.dyn_anchor_uncertainty_hz`'s docstring for how this
#: combines with the static uncertainty in
#: `cliffordclock.integrator.omega.bbr_pivot_uncertainty`.
#:
#: Validity window 50-350 K (Lisdat's fit range); the T <= 300 K subrange
#: carries the 1e-19-class PTB<->JILA cross-verification (see
#: `BBR_CROSS_VERIFIED_MAX_K`; G7 sign-off B4).
SR87 = _species_from_amu_and_frequency(
    name="Sr87",
    mass_amu=86.908877497,
    clock_frequency_hz=429_228_004_229_873.4,
    charge_state=0,
    delta_alpha_dc_si=4.07873e-39,
    bbr_coefficients=BbrCoefficients(
        nu_stat_300k_hz=-2.13023,
        nu_stat_300k_uncertainty_hz=0.00006,
        dyn_coeffs_hz={6: -0.13216, 8: -0.01231, 10: -0.00858},
        dyn_anchor_uncertainty_hz=0.00033,
    ),
)

#: Yb-171 (ytterbium-171), neutral-atom optical lattice clock species.
#:
#: Mass: 170.936 330 2(2) u. Source: M. Wang et al., "The AME 2020 atomic
#: mass evaluation (II)", Chinese Physics C 45, 030003 (2021).
#:
#: Clock transition: 6s^2 1S0 -> 6s6p 3P0 (F=1/2 -> F=1/2), frequency
#: 518 295 836 590 863.6 Hz. Source: BIPM "List of standard frequencies"
#: (secondary representation of the second based on the 171Yb transition),
#: consistent with W.F. McGrew et al., "Atomic clock performance enabling
#: geodesy below the centimetre level", Nature 564, 87 (2018).
#:
#: Differential static (dc) scalar polarizability (CONVENTIONS.md E14b):
#: Delta_alpha = 2.40269(5) x 1e-39 C^2 m^2 J^-1 (= 145.726(3) a.u.,
#: round-trips through ALPHA_AU_TO_SI to the same value, see
#: tests/test_stark_species.py). Primary source: J.A. Sherman, N.D.
#: Lemke, N. Hinkley, M. Pizzocaro, R.W. Fox, A.D. Ludlow, C.W. Oates,
#: "High-Accuracy Measurement of Atomic Polarizability in an Optical
#: Lattice Clock", Phys. Rev. Lett. 108, 153002 (2012), arXiv:1112.2766,
#: abstract and Table II "this work" row: "alpha_clock = 36.2612(7)
#: kHz(kV/cm)^-2" = "2.40269(5) x 1e-39 [C m^2/V]" (their alpha_clock is
#: this module's Delta_alpha; note the paper's own Eq. (2) pulls a
#: separate -1/2 prefactor out in front, so their native
#: kHz(kV/cm)^-2 column is *not* directly this module's
#: stark_coefficient_hz_per_v2_m2 -- see that field's docstring).
#: Independent second source, from the same paper's Table II comparison
#: with theory: S.G. Porsev, A. Derevianko, Phys. Rev. A 74, 020502
#: (2006), Delta_alpha = 2.56(26) x 1e-39 Cm^2/V (a theoretical value,
#: consistent with the experimental value within combined uncertainty;
#: included for reviewer cross-reference, not used as the registry
#: value).
#:
#: Blackbody-radiation shift coefficients (CONVENTIONS.md E32, WP20 Gate
#: G7 sign-off B3 "RATIFY", registry recommendation dossier Sec.7):
#: static Delta-nu_stat(300 K) = -1.2545(10) Hz and dynamic
#: nu_dyn,6 = -22.17(34) mHz -- both literature-direct, K. Hassan et al.,
#: arXiv:2506.05304 (2025) (weighted mean including a direct cryogenic
#: 77 K measurement; consistent with the Sherman-derived -1.2549 Hz
#: computed independently from `delta_alpha_dc_si` above via the same
#: Middelmann-convention <E^2> closure, dossier Sec.3). This measurement
#: is explicitly *not* subject to Sr's Taylor-divergence problem: Hassan
#: et al. state "this measurement is unaffected by the Taylor series
#: expansion issues identified in [Lisdat 2021], as the expansion for Yb
#: does not introduce the theory error observed in Sr."
#:
#: nu_dyn,8 = -0.744(20) mHz is derived (not independently transcribed)
#: from Beloy et al., PRL 113, 260801 (2014)'s eta_2 = 0.000593(16)
#: parameter via ``nu_dyn,8 = nu_stat_300k_hz * eta_2 = -1.2545 x
#: 0.000593 = -0.744 mHz``, propagated uncertainty
#: ``0.0010/0.000593... ~ 2.7% -> +/-0.020 mHz`` (independently verified
#: arithmetic, G7 sign-off B3: "arithmetic and propagated uncertainty
#: both correct"). The eta_1 -> nu_dyn,6 identification is cross-checked
#: the same way against Hassan's *directly measured* nu_dyn,6:
#: ``nu_stat_300k_hz * eta_1 = -1.2545 x 0.01745 = -21.9 mHz`` vs
#: Hassan's -22.17(34) mHz -- agreement confirms the eta_1/eta_2 -> T^6/T^8
#: order mapping is pinned by physics (both orders follow the same
#: expansion), not an unverified dataset-index inference (G7 sign-off B3).
#:
#: **Truncation bound (G7 sign-off B3 required edit, gate item 7):** the
#: omitted T^10 term is bounded via the coefficient *ratio*
#: ``nu_dyn,8 / nu_dyn,6 = 0.744 / 22.17 = 0.034`` (the per-order
#: suppression), not the Lisdat open dataset's order-index estimate --
#: Yb's eta-series is monotone-convergent (Hassan; Lisdat's own Yb
#: convergence file, `Approximation_G(n)_Yb.dat`, 3.6e-2 -> 2.2e-3 ->
#: 2.2e-4), so the next term is suppressed at least as fast:
#: ``|nu_dyn,10| <~ 0.034 x 0.744 mHz ~ 0.025 mHz ~ 5e-20`` fractional --
#: safely below the 1e-19 floor, and NOT included as a registry
#: coefficient (a documented, bounded omission, per E33's neglected-terms
#: discipline).
#:
#: Validity window 50-350 K (same window as Sr87, dossier Sec.7's
#: recommendation, cryo end anchored by Hassan's 77 K operation).
YB171 = _species_from_amu_and_frequency(
    name="Yb171",
    mass_amu=170.9363302,
    clock_frequency_hz=518_295_836_590_863.6,
    charge_state=0,
    delta_alpha_dc_si=2.40269e-39,
    bbr_coefficients=BbrCoefficients(
        nu_stat_300k_hz=-1.2545,
        nu_stat_300k_uncertainty_hz=0.0010,
        dyn_coeffs_hz={6: -0.02217, 8: -0.000744},
        dyn_anchor_uncertainty_hz=0.00034,
    ),
)

#: Al-27+ (aluminum-27, singly ionized), trapped-ion quantum-logic clock
#: species.
#:
#: Mass: 26.981 538 53(11) u (Al-27 is mono-isotopic; ionization removes
#: one electron mass, negligible at this precision). Source: M. Wang et
#: al., "The AME 2020 atomic mass evaluation (II)", Chinese Physics C 45,
#: 030003 (2021).
#:
#: Clock transition: 3s^2 1S0 -> 3s3p 3P0 (267.4 nm), frequency
#: 1 121 015 393 207 857.4 Hz. Source: S.M. Brewer et al., "27Al+
#: Quantum-Logic Clock with a Systematic Uncertainty below 10^-18",
#: Phys. Rev. Lett. 123, 033201 (2019), PRIMARY TEXT; BIPM "List of
#: standard frequencies" secondary representation, consistent value.
#:
#: WP21 Tier 1 (CONVENTIONS.md E14b, ion-clock scalar Stark coupling; this
#: is a J=0 -> J=0 transition, so E14b's scalar-Delta_alpha treatment
#: applies as-is -- no tensor/quadrupole term needed for THIS species,
#: unlike the D/F-state ions QUADRUPOLE_MOMENTS registers):
#: Delta_alpha(0) = 0.416(14) a.u. -- Wei et al., Phys. Rev. Lett. 133,
#: 033001 (2024), co-trapped-ion polarizability scale vs Ca+. SECONDARY
#: (2 corroborating snippets, not primary text; G8 sign-off B1
#: "RATIFY-WITH-EDIT" -- adopted as the registry pin). Consistent within
#: ~1sigma (0.17sigma) with the PRIMARY-TEXT fallback: Delta_alpha(0) =
#: 7.02(95)e-42 J m^2/V^2 = 0.426(58) a.u. -- Brewer et al. 2019 (above),
#: PRIMARY TEXT (ar5iv), whose BBR row is -30.5(4.2)e-19 at 294.8(2.7) K.
#: The famous near-cancellation: Delta_alpha sits ~2 orders below typical
#: ionic polarizabilities (the 3P0 state's contributions nearly cancel
#: 1S0's), so this choice barely moves any downstream fractional shift.
#: `delta_alpha_dc_si` below is the SECONDARY (Wei 2024) value, converted
#: via `ALPHA_AU_TO_SI`; see `tests/test_ion_species.py` for the Brewer
#: cross-check and the round-trip pin.
#:
#: No `bbr_coefficients` populated (WP21, G8 sign-off): the dossier gives
#: only Brewer 2019's single measured BBR shift at one temperature
#: (-30.5(4.2)e-19 at 294.8(2.7) K), not the independent static-T^4/
#: dynamic-polynomial split `BbrCoefficients` requires -- inventing that
#: split from one data point (rather than the Planck-weighted-integral fit
#: Sr87/Yb171 draw on) is explicitly not done here (WP21 instruction:
#: "leave bbr_coefficients unpopulated and document why (do NOT invent a
#: split)"). `resolve_bbr_coefficients` raises a clear `ValueError`.
#:
#: See `ION_MICROMOTION_NOTES["Al27+"]` (the dominant, unmodeled
#: field-sensitive systematic -- micromotion/secular motion, Brewer 2019)
#: and `ION_HYPERFINE_E2_BUDGET_NOTES["Al27+"]` (I=5/2 hyperfine-mediated
#: E2, Beloy et al. 2017) for the two documented budget-line exclusions
#: every report using this species carries
#: (`cliffordclock.pipeline.run_pipeline_full`).
AL27_PLUS = _species_from_amu_and_frequency(
    name="Al27+",
    mass_amu=26.98153853,
    clock_frequency_hz=1_121_015_393_207_857.4,
    charge_state=1,
    delta_alpha_dc_si=0.416 * ALPHA_AU_TO_SI,
)

#: In-115+ (indium-115, singly ionized), trapped-ion quantum-logic clock
#: species.
#:
#: Mass: 114.903 878 776(12) u. Source: NIST Atomic Weights and Isotopic
#: Compositions (CIAAW-derived; consistent with the AME2020 evaluation
#: this module's other species masses cite, M. Wang et al., Chinese
#: Physics C 45, 030003 (2021)).
#:
#: Clock transition: 5s^2 1S0 -> 5s5p 3P0 (236.5 nm), frequency
#: 1 267 402 452 901 049.9(6.9) Hz. Source: Ohtsubo, Li, Matsubara, Ido,
#: Hayasaka, "Frequency measurement of the clock transition of an indium
#: ion sympathetically-cooled in a linear trap", Opt. Express 25, 11725
#: (2017), arXiv:1703.02717, PRIMARY TEXT (ar5iv): "The frequency is
#: determined to be 1 267 402 452 901 049.9 (6.9) Hz by averaging 36
#: measurements using an optical frequency comb..." (the species'
#: Delta_alpha and BBR-fraction values below are from the independent
#: PRIMARY-TEXT theory source, not this frequency measurement -- see
#: below).
#:
#: WP21 Tier 1 (CONVENTIONS.md E14b; J=0 -> J=0, scalar Stark applies
#: as-is): Delta_alpha(0) theory = 2.01 a.u. -- Safronova, Porsev,
#: Safronova, Phys. Rev. Lett. 107, 143006 (2011) Table IV, PRIMARY TEXT
#: (ar5iv). G8 sign-off B2 "RATIFY": "pin the Safronova 2011 theory value
#: ... and correctly do not pin the 2024 Coulomb-crystal paper's
#: un-cross-checked '3.3(3) J/(V/m)^2' (single source)". Labeled
#: theory-derived (no independent experimental Delta_alpha measurement is
#: in the dossier), so its BBR-relevant uncertainty is understood as
#: theory-limited, not measurement-limited.
#:
#: No `bbr_coefficients` populated (WP21, G8 sign-off), same reasoning as
#: Al27+: the dossier's only BBR datum is Safronova 2011's single
#: fractional-shift value at 300 K (1.36(10)e-17), not an independent
#: static/dynamic-polynomial split -- not invented here.
#:
#: See `ION_MICROMOTION_NOTES["In115+"]` and
#: `ION_HYPERFINE_E2_BUDGET_NOTES["In115+"]` (I=9/2 hyperfine-mediated E2,
#: budgeted at -1.4(3)e-19 in a representative 2024 evaluation) for the
#: two documented budget-line exclusions every report using this species
#: carries.
IN115_PLUS = _species_from_amu_and_frequency(
    name="In115+",
    mass_amu=114.903878776,
    clock_frequency_hz=1_267_402_452_901_049.9,
    charge_state=1,
    delta_alpha_dc_si=2.01 * ALPHA_AU_TO_SI,
)

_REGISTRY: dict[str, Species] = {
    "Sr87": SR87,
    "Yb171": YB171,
    "Al27+": AL27_PLUS,
    "In115+": IN115_PLUS,
}


def get_species(name: str) -> Species:
    """Look up a `Species` by registry name.

    Parameters
    ----------
    name : str
        One of ``"Sr87"``, ``"Yb171"``, ``"Al27+"``.

    Returns
    -------
    Species
        The registry entry for `name`.

    Raises
    ------
    KeyError
        If `name` is not a registered species; the error message lists the
        valid names.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        valid = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown species {name!r}; valid names are: {valid}") from None
