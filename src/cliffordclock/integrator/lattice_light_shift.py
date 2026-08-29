# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lattice light shift models (WP36 Phase 1, CONVENTIONS.md section 17,
E40/E41): the Katori-lineage harmonic/operational model and the NIST
Born-Oppenheimer+WKB (BO+WKB) model, implemented as pure functions.

**Scope boundary, stated up front.** This module is functions and
benchmarks only. Neither model is wired into `cliffordclock.pipeline`'s
config surface (no `coupling.type`/`environment.*` entry consumes these
functions): that wiring, and the side-by-side comparison notebook, are
later phases. This module exists so both community models can be
validated against their defining papers on their own terms before any
pipeline integration decision is made.

**Model A (E40): the Katori-lineage harmonic/operational model.**
Ushijima, Takamoto, Katori, PRL 121, 263202 (2018), arXiv:1812.11815,
their Eq. 1 (the light shift as a harmonic-oscillator perturbative series
in the reduced trap depth ``u``, exact through hyperpolarizability order)
and Eq. 2 (the radial-thermal reduction factor). Coefficients are
per-species E1-polarizability-slope/M1+E2-polarizability/hyperpolarizability
triples, each carrying its own paper citation
(:class:`HarmonicLatticeCoefficients`). Two forms of the radial-thermal
reduction factor are implemented and clearly distinguished
(:func:`ushijima_reduction_factor`, :func:`jila_reduction_factor`): the two
are algebraically different formulas from different papers in the same
lineage, not interchangeable notations for the same thing (see each
function's docstring).

**Model B (E41): the NIST Born-Oppenheimer+WKB model.** Beloy, McGrew,
Zhang, Nicolodi, Fasano, Hassan, Brown, Ludlow, PRA 101, 053416 (2020),
arXiv:2004.06224. The Born-Oppenheimer axial eigenproblem at fixed
radial coordinate (their Eq. 5) is solved by numerical diagonalization,
converged to a stated tolerance (see the module's Numerics note below); the
resulting radial potential per axial band is quantized by WKB by inverting
the classical turning radius against energy (their Eqs. 8, 9, 11); the
density of states follows from the squared turning radius (Eq. 11), and is
checked against the closed-form harmonic-oscillator limit (Eq. 4) both
algebraically (:func:`harmonic_density_of_states_closed_form` vs.
:func:`bo_wkb_density_of_states` with ``potential="harmonic"``) and via the
same finite-difference axial solver used for the true site potential
(:func:`axial_energies_er` with ``potential="harmonic"`` reproducing the
exact 1D quantum-harmonic-oscillator spectrum). The thermally-averaged
light shift uses the ensemble-averaged trap-depth-reduction factors
``X``/``Y``/``Z`` (Beloy's Eqs. 16-21) evaluated through Bothwell, Hunt,
Siegel, Hassan, Grogan, Kobayashi, Gibble, Porsev, Safronova, Brown, Beloy,
Ludlow, PRL 134, 033201 (2025), arXiv:2409.10782, Eq. 6 (the practical
form applying Beloy's factors with the same per-species coefficients
Model A tabulates, verified against that paper's typeset PDF, section
"Appendix A: Born-Oppenheimer + WKB approximation").

**Numerics.** The axial Born-Oppenheimer eigenproblem is solved by
finite-difference diagonalization on the dimensionless domain
``x = k*z in [-pi/2, pi/2]`` (one lattice site, Dirichlet boundary
conditions, the domain over which Beloy's own Eq. 13 integrates,
justified by the deep-lattice/negligible-tunneling assumption both papers
make). Every function that depends on this discretization carries an
explicit convergence guard (:class:`LatticeLightShiftConvergenceError`):
grid resolution is doubled until the quantity of interest stabilizes
within a stated tolerance. If that tolerance is never reached, the guard
raises with the residual named in the message; it never returns an
unconverged number silently.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import eigh_tridiagonal  # type: ignore[import-untyped]
from scipy.optimize import brentq  # type: ignore[import-untyped]

from cliffordclock.constants import BOLTZMANN_K, HBAR, PLANCK_H

__all__ = [
    "LatticeLightShiftConvergenceError",
    "HarmonicLatticeCoefficients",
    "recoil_energy_j",
    "ushijima_reduction_factor",
    "jila_reduction_factor",
    "harmonic_light_shift_hz",
    "harmonic_light_shift_uncertainty_hz",
    "HarmonicOperationalPoint",
    "solve_harmonic_operational_point",
    "SitePotential",
    "make_site_potential",
    "axial_energies_er",
    "harmonic_density_of_states_closed_form",
    "axial_band_energy_er",
    "turning_radius_m",
    "bo_wkb_density_of_states",
    "ThermalShapeFactors",
    "axial_thermal_factors",
    "bo_wkb_fractional_light_shift",
]


class LatticeLightShiftConvergenceError(RuntimeError):
    """Raised when a grid-resolution convergence guard fails to reach its
    stated tolerance within its stated maximum resolution. Never silently
    returned as a best-effort number: every numeric routine in this module
    either converges to its stated tolerance or raises this, naming the
    quantity, the tolerance, the resolution reached, and the residual
    change at that resolution.
    """


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def recoil_energy_j(wavelength_m: float, mass_kg: float) -> float:
    """Lattice photon recoil energy ``E_R`` (joules).

    ``E_R = h^2 / (2*m*lambda^2)``, algebraically identical to Beloy et
    al. 2020's own ``E_R = hbar^2 k^2 / (2m)`` (their section II, just
    below Eq. 1, with ``k = 2*pi/lambda``) and to Ushijima et al. 2018's
    ``E_R = (h*nu_L/c)^2/(2m)`` (their Eq. 1 context) and Bothwell et al.
    2025's ``E_R = (h*nu_L)^2/(2*m*c^2)`` (their Eq. 1 context): all three
    are the same recoil energy written in wavelength, angular-wavenumber,
    and frequency form respectively (`nu_L = c/lambda`).

    Parameters
    ----------
    wavelength_m : float
        Lattice laser wavelength, meters. Must be `> 0`.
    mass_kg : float
        Atomic mass, kilograms. Must be `> 0`.

    Returns
    -------
    float
        `E_R`, joules.
    """
    if wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be > 0, got {wavelength_m}")
    if mass_kg <= 0:
        raise ValueError(f"mass_kg must be > 0, got {mass_kg}")
    return PLANCK_H**2 / (2.0 * mass_kg * wavelength_m**2)


# ---------------------------------------------------------------------------
# Model A: Katori-lineage harmonic/operational model (E40)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarmonicLatticeCoefficients:
    """Per-species Model A coefficients (Ushijima et al. 2018 Eq. 1's
    ``(1/h)*d(alpha~^E1)/d(nu)``, ``alpha~^qm/h``, ``beta~/h``), each
    tabulated with its own 1-sigma uncertainty and source citation
    (CONVENTIONS.md section 17, E40). Every field name matches the
    quantity's own symbol so a reader can cross-check against a paper's
    table directly.

    Attributes
    ----------
    e1_slope_per_hz : float
        ``(1/h)*d(alpha~^E1)/d(nu)``, the E1-polarizability-difference
        slope with respect to lattice detuning, per hertz (dimensionless
        numerically, e.g. Ushijima's ``1.735e-11``). Multiplying by a
        detuning in Hz gives a dimensionless fraction of `u`'s coefficient
        in Eq. 1.
    e1_slope_per_hz_uncertainty : float
        1-sigma uncertainty on `e1_slope_per_hz`. Must be `>= 0`.
    m1e2_hz : float
        ``alpha~^qm/h`` (or, in Bothwell et al. 2025's notation,
        ``alpha~^M1E2``), the combined M1+E2 polarizability difference,
        hertz (Ushijima: negative mHz-scale; Kim/Aeppli/Bothwell use the
        same symbol at a different numeric value for their own species).
    m1e2_hz_uncertainty : float
        1-sigma uncertainty on `m1e2_hz`, hertz. Must be `>= 0`.
    hyperpolarizability_hz : float
        ``beta~/h``, the hyperpolarizability, hertz.
    hyperpolarizability_hz_uncertainty : float
        1-sigma uncertainty on `hyperpolarizability_hz`, hertz. Must be
        `>= 0`.
    magic_frequency_hz : float | None
        The E1 magic frequency ``nu^E1`` this coefficient set is anchored
        to, hertz, if published (optional: not needed to evaluate Eq. 1,
        only to relate an absolute lattice frequency to a detuning
        `delta_L`). Default `None`.
    magic_frequency_hz_uncertainty : float
        1-sigma uncertainty on `magic_frequency_hz`, hertz. Default `0.0`.
    citation : str
        The source publication, verbatim, for every numeric field above.
    """

    e1_slope_per_hz: float
    e1_slope_per_hz_uncertainty: float
    m1e2_hz: float
    m1e2_hz_uncertainty: float
    hyperpolarizability_hz: float
    hyperpolarizability_hz_uncertainty: float
    magic_frequency_hz: float | None = None
    magic_frequency_hz_uncertainty: float = 0.0
    citation: str = ""


#: Ushijima, Takamoto, Katori, PRL 121, 263202 (2018), Table I, transcribed
#: directly from the typeset PDF (arXiv:1812.11815): "(1/h)d(alpha~E1)/dnu
#: 1.735(13)e-11", "alpha~qm/h -0.962(40) mHz", "beta~/h -0.461(14) uHz",
#: "nu^E1 368554465.1(1.0) MHz". Species: Sr-87, RIKEN. This is Target 1's
#: coefficient set (`benchmarks/run_lattice_light_shift.py`).
USHIJIMA_2018_SR87 = HarmonicLatticeCoefficients(
    e1_slope_per_hz=1.735e-11,
    e1_slope_per_hz_uncertainty=0.013e-11,
    m1e2_hz=-0.962e-3,
    m1e2_hz_uncertainty=0.040e-3,
    hyperpolarizability_hz=-0.461e-6,
    hyperpolarizability_hz_uncertainty=0.014e-6,
    magic_frequency_hz=368_554_465.1e6,
    magic_frequency_hz_uncertainty=1.0e6,
    citation="Ushijima, Takamoto, Katori, PRL 121, 263202 (2018), Table I",
)

#: Kim, Aeppli, Bothwell, Ye, PRL 130, 113203 (2023), transcribed directly
#: from the typeset PDF (arXiv:2210.16374): "(1/h)d(alpha~E1)/dnu
#: 1.859(5)e-11", "alpha~qm/h -1.24(5) mHz", "beta~/h -0.51(4) uHz".
#: Species: Sr-87, JILA. Uses the RECIPROCAL radial-thermal reduction
#: factor (:func:`jila_reduction_factor`), not Ushijima's linear form; see
#: that function's docstring. This is the coefficient set Aeppli et al.
#: 2024 reuses verbatim ("identical atomic coefficients as in Ref. [19]",
#: their Ref. 19 being this paper) for Target 2.
KIM_2023_SR87 = HarmonicLatticeCoefficients(
    e1_slope_per_hz=1.859e-11,
    e1_slope_per_hz_uncertainty=0.005e-11,
    m1e2_hz=-1.24e-3,
    m1e2_hz_uncertainty=0.05e-3,
    hyperpolarizability_hz=-0.51e-6,
    hyperpolarizability_hz_uncertainty=0.04e-6,
    citation="Kim, Aeppli, Bothwell, Ye, PRL 130, 113203 (2023), main text",
)

#: Bothwell, Hunt, Siegel, Hassan, Grogan, Kobayashi, Gibble, Porsev,
#: Safronova, Brown, Beloy, Ludlow, PRL 134, 033201 (2025), Table III,
#: HARMONIC-BASIS column, transcribed directly from the typeset PDF
#: (arXiv:2409.10782): "d(alpha~E1)/dnu 4.21(10)e-20/MHz", "beta~
#: -1.7(4)e-21", "alpha~M1E2 -1.41(9)e-18". Species: Yb-171, NIST. Unlike
#: `USHIJIMA_2018_SR87`/`KIM_2023_SR87`, these coefficients are ALREADY
#: normalized by the clock frequency (Bothwell's own Eq. 1 gives
#: `delta_nu_LS/nu_c` directly, not `h*nu_LS`): `e1_slope_per_hz` here is
#: `d(alpha~E1)/dnu` in units of per-hertz-of-detuning directly
#: (`4.21e-20/MHz = 4.21e-26/Hz`), and `m1e2_hz`/`hyperpolarizability_hz`
#: are DIMENSIONLESS fractional coefficients despite the field names
#: (`_hz` names are kept for API uniformity with the other two entries).
#: `harmonic_light_shift_hz`'s formula is a pure coefficient-algebra
#: evaluator with no unit assumption of its own, so it accepts this
#: constant directly and returns the fractional `nu_LS/nu_c` (matching
#: Bothwell's own Eq. 1), never `nu_LS` in hertz: the RESULT'S UNIT
#: tracks whichever coefficient convention is passed in, and callers must
#: track which one they used (Bothwell's Table III main text states this
#: normalization explicitly: "we have divided the clock shift (delta_nu_LS)
#: by the clock frequency (nu_c)"). `bo_wkb_fractional_light_shift` (Eq. 6)
#: requires this SAME Bothwell convention, never the Ushijima/Kim `.../h`
#: hertz convention; see its own docstring for the consequence of mixing
#: the two.
BOTHWELL_2025_YB171_HARMONIC = HarmonicLatticeCoefficients(
    e1_slope_per_hz=4.21e-20 / 1.0e6,
    e1_slope_per_hz_uncertainty=0.10e-20 / 1.0e6,
    m1e2_hz=-1.41e-18,
    m1e2_hz_uncertainty=0.09e-18,
    hyperpolarizability_hz=-1.7e-21,
    hyperpolarizability_hz_uncertainty=0.4e-21,
    magic_frequency_hz=394_798_266.9e6,
    magic_frequency_hz_uncertainty=0.26e6,
    citation=(
        "Bothwell, Hunt, Siegel, Hassan, Grogan, Kobayashi, Gibble, Porsev, "
        "Safronova, Brown, Beloy, Ludlow, PRL 134, 033201 (2025), Table III, "
        "harmonic-basis column"
    ),
)

#: Same table, BO+WKB column: "d(alpha~E1)/dnu 4.31(9)e-20/MHz", "beta~
#: -2.0(6)e-21", "alpha~M1E2 -1.45(8)e-18". Same unit contract as
#: `BOTHWELL_2025_YB171_HARMONIC` above.
BOTHWELL_2025_YB171_BOWKB = HarmonicLatticeCoefficients(
    e1_slope_per_hz=4.31e-20 / 1.0e6,
    e1_slope_per_hz_uncertainty=0.09e-20 / 1.0e6,
    m1e2_hz=-1.45e-18,
    m1e2_hz_uncertainty=0.08e-18,
    hyperpolarizability_hz=-2.0e-21,
    hyperpolarizability_hz_uncertainty=0.6e-21,
    magic_frequency_hz=394_798_266.3e6,
    magic_frequency_hz_uncertainty=0.30e6,
    citation=(
        "Bothwell, Hunt, Siegel, Hassan, Grogan, Kobayashi, Gibble, Porsev, "
        "Safronova, Brown, Beloy, Ludlow, PRL 134, 033201 (2025), Table III, "
        "BO+WKB column"
    ),
)


def ushijima_reduction_factor(
    u: float, j: float, radial_temperature_k: float, recoil_energy_j_value: float
) -> float:
    """Ushijima et al. 2018 Eq. 2's radial-thermal reduction factor,
    LINEAR form (CONVENTIONS.md E40): ``zeta_j(u) ~= 1 - j*kB*Tr/(u*E_R)``.

    Ushijima's own Eq. 2 defines this via a thermal average of the
    Boltzmann-weighted radial intensity profile to the ``j``-th power,
    approximated (their stated ``~=``, valid for `0.95 < zeta_1(u) < 0.99`
    over their own `150 < u < 1150` range) by the linear form above. This
    is the form used for **Target 1** (Ushijima's own operational point).

    **Not the same formula as** :func:`jila_reduction_factor`. Kim et al.
    2023 (their own text, "use of an effective depth, `uj = (1 +
    j*kB*Tr/(u0*ER))^-1 * uj0`") and Bothwell et al. 2025 (their Eq. 1
    context, the identical reciprocal form) use the EXACT reciprocal of
    the linear approximation here, not this formula: to leading order in
    `j*kB*Tr/(u*E_R)` the two agree, but they are algebraically different
    expressions and must not be interchanged when reproducing a specific
    paper's numbers (:func:`jila_reduction_factor`'s docstring states the
    reciprocal form explicitly).

    Parameters
    ----------
    u : float
        Reduced (peak) trap depth, `u = U/E_R`, dimensionless. Must be
        `> 0`.
    j : float
        The power of `u` this reduction factor multiplies (`0.5`, `1`,
        `1.5`, or `2` for Eq. 1's four terms). Must be `> 0`.
    radial_temperature_k : float
        Radial temperature `Tr`, kelvin. Must be `>= 0`.
    recoil_energy_j_value : float
        `E_R`, joules (:func:`recoil_energy_j`). Must be `> 0`.

    Returns
    -------
    float
        `zeta_j(u)`, dimensionless.
    """
    if u <= 0:
        raise ValueError(f"u must be > 0, got {u}")
    if j <= 0:
        raise ValueError(f"j must be > 0, got {j}")
    if radial_temperature_k < 0:
        raise ValueError(f"radial_temperature_k must be >= 0, got {radial_temperature_k}")
    if recoil_energy_j_value <= 0:
        raise ValueError(f"recoil_energy_j_value must be > 0, got {recoil_energy_j_value}")
    return 1.0 - j * BOLTZMANN_K * radial_temperature_k / (u * recoil_energy_j_value)


def jila_reduction_factor(
    u: float, j: float, radial_temperature_k: float, recoil_energy_j_value: float
) -> float:
    """The JILA-lineage RECIPROCAL radial-thermal reduction factor
    (CONVENTIONS.md E40): ``zeta_j(u) = (1 + j*kB*Tr/(u*E_R))^-1``.

    Kim, Aeppli, Bothwell, Ye, PRL 130, 113203 (2023) (main text: "use of
    an effective depth, `uj = (1 + j*kB*Tr/(u0*ER))^-1 * uj0`") and
    Bothwell et al. 2025 (their Eq. 1 context, the same form) use this
    exact reciprocal in place of Ushijima et al. 2018's linear
    approximation (:func:`ushijima_reduction_factor`). The two forms agree
    to leading order in `j*kB*Tr/(u*E_R) << 1` (`(1+x)^-1 ~= 1-x`) but
    differ at the next order; this function is the one required to
    reproduce **Target 2** (Aeppli et al. 2024) and the harmonic side of
    **Target 3** (Bothwell et al. 2025), both of which explicitly adopt
    the Kim et al. 2023 form.

    Parameters, returns, and raises: identical contract to
    :func:`ushijima_reduction_factor`.
    """
    if u <= 0:
        raise ValueError(f"u must be > 0, got {u}")
    if j <= 0:
        raise ValueError(f"j must be > 0, got {j}")
    if radial_temperature_k < 0:
        raise ValueError(f"radial_temperature_k must be >= 0, got {radial_temperature_k}")
    if recoil_energy_j_value <= 0:
        raise ValueError(f"recoil_energy_j_value must be > 0, got {recoil_energy_j_value}")
    return 1.0 / (1.0 + j * BOLTZMANN_K * radial_temperature_k / (u * recoil_energy_j_value))


ReductionForm = Literal["none", "ushijima_linear", "jila_reciprocal"]


def harmonic_light_shift_hz(
    u: float,
    detuning_hz: float,
    n_z: float,
    coeffs: HarmonicLatticeCoefficients,
    *,
    reduction_form: ReductionForm = "none",
    radial_temperature_k: float | None = None,
    recoil_energy_j_value: float | None = None,
) -> float:
    """Model A's light shift, ``nu_LS(u, delta_L, n_z)`` (CONVENTIONS.md
    E40, Ushijima et al. 2018 Eq. 1, transcribed verbatim from the typeset
    PDF):

        h*nu_LS(u, delta_L, n_z) ~=
            [d(alpha~E1)/dnu * delta_L - alpha~qm] * (n_z+1/2) * u^(1/2)
          - [d(alpha~E1)/dnu * delta_L + (3/2)*beta~*(n_z^2+n_z+1/2)] * u
          + 2*beta~*(n_z+1/2) * u^(3/2)
          - beta~ * u^2

    with every coefficient already the paper's own ``/h`` form (hertz), so
    this function returns `nu_LS` directly in hertz, not `h*nu_LS` in
    joules (matching how `coeffs`' fields are tabulated). `u` enters at
    four different powers (`1/2`, `1`, `3/2`, `2`); when
    `reduction_form != "none"`, EACH power's own `u^j` is replaced by
    `zeta_j(u) * u^j` (Ushijima Eq. 2's radial-thermal folding, applied
    term-by-term, per the paper's own "effective intensity" convention),
    using either :func:`ushijima_reduction_factor` or
    :func:`jila_reduction_factor` per `reduction_form`.

    Parameters
    ----------
    u : float
        Reduced (peak) trap depth, dimensionless. Must be `> 0`.
    detuning_hz : float
        `delta_L`, the lattice-laser detuning from the E1 magic frequency,
        hertz. May be any sign.
    n_z : float
        Axial vibrational quantum number. Must be `>= 0`. Physical calls
        always pass a non-negative integer value; the parameter is typed
        `float` so finite-difference uncertainty-propagation callers can
        pass it straight through the same call signature.
    coeffs : HarmonicLatticeCoefficients
        The species' E1-slope/M1+E2/hyperpolarizability coefficients.
    reduction_form : {"none", "ushijima_linear", "jila_reciprocal"}, default "none"
        Which radial-thermal reduction factor to fold in, if any. `"none"`
        (the default) evaluates Eq. 1 exactly as published, no radial
        averaging (the bare per-`(u, delta_L, n_z)` formula; this is what
        **Target 1**'s operational-point solve uses, matching Ushijima's
        own derivation of `u^op`/`delta_L^op` directly from Eq. 1).
    radial_temperature_k : float | None
        Radial temperature `Tr`, kelvin. Required (and must be `>= 0`)
        when `reduction_form != "none"`; ignored otherwise.
    recoil_energy_j_value : float | None
        `E_R`, joules. Required (and must be `> 0`) when
        `reduction_form != "none"`; ignored otherwise.

    Returns
    -------
    float
        `nu_LS`, hertz.

    Raises
    ------
    ValueError
        `u <= 0`, `n_z < 0`, or a required radial-averaging argument is
        missing/invalid.
    """
    if u <= 0:
        raise ValueError(f"u must be > 0, got {u}")
    if n_z < 0:
        raise ValueError(f"n_z must be >= 0, got {n_z}")
    if reduction_form != "none":
        if radial_temperature_k is None or recoil_energy_j_value is None:
            raise ValueError(
                "radial_temperature_k and recoil_energy_j_value are required when "
                f"reduction_form={reduction_form!r}"
            )
        if radial_temperature_k < 0:
            raise ValueError(f"radial_temperature_k must be >= 0, got {radial_temperature_k}")
        if recoil_energy_j_value <= 0:
            raise ValueError(f"recoil_energy_j_value must be > 0, got {recoil_energy_j_value}")
        reduction_fn = (
            ushijima_reduction_factor
            if reduction_form == "ushijima_linear"
            else jila_reduction_factor
        )

        def u_pow(j: float) -> float:
            zeta = reduction_fn(u, j, radial_temperature_k, recoil_energy_j_value)
            return float(zeta * u**j)
    else:

        def u_pow(j: float) -> float:
            return float(u**j)

    e1_slope = coeffs.e1_slope_per_hz
    m1e2 = coeffs.m1e2_hz
    beta = coeffs.hyperpolarizability_hz

    term1 = (e1_slope * detuning_hz - m1e2) * (n_z + 0.5) * u_pow(0.5)
    term2 = -(e1_slope * detuning_hz + 1.5 * beta * (n_z**2 + n_z + 0.5)) * u_pow(1.0)
    term3 = 2.0 * beta * (n_z + 0.5) * u_pow(1.5)
    term4 = -beta * u_pow(2.0)
    return term1 + term2 + term3 + term4


def harmonic_light_shift_uncertainty_hz(
    u: float,
    detuning_hz: float,
    n_z: float,
    coeffs: HarmonicLatticeCoefficients,
    *,
    reduction_form: ReductionForm = "none",
    radial_temperature_k: float | None = None,
    recoil_energy_j_value: float | None = None,
) -> float:
    """Propagated 1-sigma uncertainty on :func:`harmonic_light_shift_hz`
    from `coeffs`' own coefficient uncertainties (CONVENTIONS.md E40,
    "arithmetic-reproduction fidelity, not an independent physics-accuracy
    claim", the same framing E32's own uncertainty note already states for
    its BBR registry coefficients).

    Central finite-difference partial derivatives with respect to each of
    `coeffs.e1_slope_per_hz`, `coeffs.m1e2_hz`, `coeffs.hyperpolarizability_hz`
    (relative step `1e-6`, or an absolute floor of `1e-30` for an
    exactly-zero coefficient), each multiplied by that coefficient's own
    1-sigma uncertainty and combined in quadrature via `math.fsum`
    (E10-style compensated summation, mirroring
    `cliffordclock.integrator.omega.bbr_pivot_uncertainty`'s pattern).
    `magic_frequency_hz`'s uncertainty is not propagated here: it does not
    enter :func:`harmonic_light_shift_hz`'s formula directly (only
    `detuning_hz`, an independent caller input, does).

    Parameters, returns: same as :func:`harmonic_light_shift_hz`.

    Returns
    -------
    float
        1-sigma uncertainty on `nu_LS`, hertz.
    """

    def evaluate(perturbed: HarmonicLatticeCoefficients) -> float:
        return harmonic_light_shift_hz(
            u,
            detuning_hz,
            n_z,
            perturbed,
            reduction_form=reduction_form,
            radial_temperature_k=radial_temperature_k,
            recoil_energy_j_value=recoil_energy_j_value,
        )

    contributions: list[float] = []
    for field_name, sigma in (
        ("e1_slope_per_hz", coeffs.e1_slope_per_hz_uncertainty),
        ("m1e2_hz", coeffs.m1e2_hz_uncertainty),
        ("hyperpolarizability_hz", coeffs.hyperpolarizability_hz_uncertainty),
    ):
        if sigma == 0.0:
            continue
        value = getattr(coeffs, field_name)
        step = max(abs(value) * 1e-6, 1e-30)
        plus = evaluate(_replace_field(coeffs, field_name, value + step))
        minus = evaluate(_replace_field(coeffs, field_name, value - step))
        partial = (plus - minus) / (2.0 * step)
        contributions.append((partial * sigma) ** 2)
    return math.sqrt(math.fsum(contributions)) if contributions else 0.0


def _replace_field(
    coeffs: HarmonicLatticeCoefficients, field_name: str, value: float
) -> HarmonicLatticeCoefficients:
    """Return a copy of `coeffs` with one field replaced (internal helper
    for the finite-difference uncertainty propagation above)."""
    return dataclasses.replace(coeffs, **{field_name: value})  # type: ignore[arg-type]


@dataclass(frozen=True)
class HarmonicOperationalPoint:
    """Result of :func:`solve_harmonic_operational_point`.

    Attributes
    ----------
    u_op : float
        The solved reduced operational trap depth.
    detuning_hz_op : float
        The solved operational detuning, hertz.
    residual_shift_hz : float
        `nu_LS(u_op, detuning_hz_op, n_z)`, hertz. Close to `0` to solver
        tolerance, the defining condition.
    residual_slope_hz : float
        `d(nu_LS)/du` at `(u_op, detuning_hz_op, n_z)`, hertz. Also close
        to `0` to solver tolerance, the second defining condition.
    """

    u_op: float
    detuning_hz_op: float
    residual_shift_hz: float
    residual_slope_hz: float


def solve_harmonic_operational_point(
    coeffs: HarmonicLatticeCoefficients,
    n_z: float = 0.0,
    *,
    u_bracket: tuple[float, float] = (2.0, 200.0),
    du: float = 1e-4,
    xtol: float = 1e-10,
) -> HarmonicOperationalPoint:
    """Solve Ushijima et al. 2018's own operational-point conditions,
    ``nu_LS(u_op, delta_L_op, n_z) = 0`` AND
    ``d(nu_LS)/du|_(u_op, delta_L_op, n_z) = 0`` simultaneously
    (CONVENTIONS.md E40; **Target 1**'s reproduction method).

    Eq. 1 is exactly LINEAR in `delta_L` at fixed `u`: writing
    `nu_LS(u, delta_L, n_z) = A(u)*delta_L + B(u)` (both obtained from two
    calls to :func:`harmonic_light_shift_hz` at `delta_L in {0, 1}`, the
    same engine formula every other call site uses:
    `B(u) = nu_LS(u, 0, n_z)`, `A(u) = nu_LS(u, 1,
    n_z) - B(u)`), the first condition gives `delta_L(u) = -B(u)/A(u)` for
    any `u`; substituting into the total-derivative form of the second
    condition and clearing `A(u)` from the denominator gives a single
    equation in `u` alone,

        g(u) = B'(u)*A(u) - A'(u)*B(u) = 0 ,

    with `A'`/`B'` central finite differences (step `du`, default `1e-4`
    in `u`, far below the smooth polynomial-in-`u^(1/2)` curvature scale
    of Eq. 1). `g(u)` is solved by :func:`scipy.optimize.brentq` bracketed
    at `u_bracket`'s endpoints (default `(2, 200)`, safely spanning
    Ushijima's own published `u_op = 72(2)`); the resulting `u_op` then
    gives `delta_L_op = -B(u_op)/A(u_op)` directly.

    Parameters
    ----------
    coeffs : HarmonicLatticeCoefficients
        The species' coefficients (Eq. 1 is evaluated with
        `reduction_form="none"`, matching Ushijima's own derivation of
        `u^op`/`delta_L^op`, their Eqs. 12-15, from the bare Eq. 1 with no
        radial-thermal folding).
    n_z : float, default 0.0
        The axial state the operational point is solved for. Ushijima's
        own `u_op = 72(2)`/`delta_L^op = 5.3(2) MHz` is the `n_z = 0`
        solution (their Eqs. 14-15 explicitly set `n = 0`).
    u_bracket : tuple[float, float], default (2.0, 200.0)
        Bracket for the `brentq` root search on `g(u)`. Must bracket a
        sign change; raises `ValueError` (propagated from `brentq`) if
        not.
    du : float, default 1e-4
        Finite-difference step for `A'(u)`/`B'(u)`.
    xtol : float, default 1e-10
        `brentq`'s absolute `u` tolerance.

    Returns
    -------
    HarmonicOperationalPoint
    """
    if u_bracket[0] <= 0 or u_bracket[1] <= u_bracket[0]:
        raise ValueError(
            f"u_bracket must be an increasing pair of positive floats, got {u_bracket}"
        )

    def a_of_u(u: float) -> float:
        b = harmonic_light_shift_hz(u, 0.0, n_z, coeffs)
        one = harmonic_light_shift_hz(u, 1.0, n_z, coeffs)
        return one - b

    def b_of_u(u: float) -> float:
        return harmonic_light_shift_hz(u, 0.0, n_z, coeffs)

    def g(u: float) -> float:
        a_plus, a_minus = a_of_u(u + du), a_of_u(u - du)
        b_plus, b_minus = b_of_u(u + du), b_of_u(u - du)
        a_prime = (a_plus - a_minus) / (2.0 * du)
        b_prime = (b_plus - b_minus) / (2.0 * du)
        return b_prime * a_of_u(u) - a_prime * b_of_u(u)

    u_op = brentq(g, u_bracket[0], u_bracket[1], xtol=xtol)
    a_val, b_val = a_of_u(u_op), b_of_u(u_op)
    detuning_op = -b_val / a_val
    residual_shift = harmonic_light_shift_hz(u_op, detuning_op, n_z, coeffs)
    a_plus, a_minus = a_of_u(u_op + du), a_of_u(u_op - du)
    b_plus, b_minus = b_of_u(u_op + du), b_of_u(u_op - du)
    slope_plus = a_plus * detuning_op + b_plus
    slope_minus = a_minus * detuning_op + b_minus
    residual_slope = (slope_plus - slope_minus) / (2.0 * du)
    return HarmonicOperationalPoint(
        u_op=u_op,
        detuning_hz_op=detuning_op,
        residual_shift_hz=residual_shift,
        residual_slope_hz=residual_slope,
    )


# ---------------------------------------------------------------------------
# Model B: NIST Born-Oppenheimer + WKB model (E41)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SitePotential:
    """A 1D optical lattice site's confining potential (CONVENTIONS.md
    E41, Beloy et al. 2020 Eq. 1): ``U(rho,z) = -D*exp(-kappa^2*rho^2)*
    cos^2(k*z)``, with ``D = (E0/2)^2 * alpha_E1`` the peak depth,
    ``kappa = sqrt(2)/w`` (``w`` the lattice beam's 1/e^2 intensity
    radius), and ``k = 2*pi/lambda``.

    Attributes
    ----------
    depth_er : float
        Peak trap depth `D` in units of the recoil energy `E_R`
        (`u0` in Beloy's/Bothwell's notation). Must be `> 0`.
    waist_m : float
        Lattice beam 1/e^2 intensity radius `w`, meters. Must be `> 0`.
    wavelength_m : float
        Lattice laser wavelength `lambda`, meters. Must be `> 0`.
    mass_kg : float
        Atomic mass, kilograms. Must be `> 0`.
    recoil_energy_j_value : float
        `E_R` (:func:`recoil_energy_j`), joules; cached here so call sites
        reuse it directly.
    kappa_per_m : float
        `kappa = sqrt(2)/w`, per meter; cached here so call sites reuse it
        directly.
    k_per_m : float
        `k = 2*pi/lambda`, per meter; cached here so call sites reuse it
        directly.
    """

    depth_er: float
    waist_m: float
    wavelength_m: float
    mass_kg: float
    recoil_energy_j_value: float
    kappa_per_m: float
    k_per_m: float


def make_site_potential(
    depth_er: float, waist_m: float, wavelength_m: float, mass_kg: float
) -> SitePotential:
    """Build a :class:`SitePotential` from its physical inputs, computing
    `recoil_energy_j_value`/`kappa_per_m`/`k_per_m` via
    :func:`recoil_energy_j` and Beloy et al. 2020's own definitions
    (``kappa = sqrt(2)/w``, ``k = 2*pi/lambda``, both stated directly below
    their Eq. 1).

    **Species trap.** `mass_kg`/`wavelength_m` set this site's `E_R`, which
    controls the thermal weighting (`kB*Tr/E_R`) every downstream `X`/`Y`/`Z`
    evaluation uses (:func:`axial_thermal_factors`). Reusing a `(u0, Tr)`
    pair from a published table with the wrong species' mass/wavelength
    evaluates a different physical trap depth than the table intends, even
    though `waist_m` itself never affects the result: see
    :func:`axial_thermal_factors`'s own docstring for the measured size of
    this effect.
    """
    if depth_er <= 0:
        raise ValueError(f"depth_er must be > 0, got {depth_er}")
    if waist_m <= 0:
        raise ValueError(f"waist_m must be > 0, got {waist_m}")
    e_r = recoil_energy_j(wavelength_m, mass_kg)
    return SitePotential(
        depth_er=depth_er,
        waist_m=waist_m,
        wavelength_m=wavelength_m,
        mass_kg=mass_kg,
        recoil_energy_j_value=e_r,
        kappa_per_m=math.sqrt(2.0) / waist_m,
        k_per_m=2.0 * math.pi / wavelength_m,
    )


def _local_depth_er(site: SitePotential, rho_m: float) -> float:
    """Local (radius-``rho``) peak axial depth in `E_R` units,
    `D(rho)/E_R = depth_er * exp(-kappa^2*rho^2)`: because Beloy's Eq. 1
    potential factors exactly as `-D*exp(-kappa^2*rho^2)*cos^2(kz)`, the
    axial eigenproblem at any fixed `rho` is IDENTICAL in form to the
    on-axis (`rho=0`) problem with a rescaled depth: the key
    simplification this module's finite-difference solver exploits. One
    1D solver, called once per `rho` grid point, stands in for a genuine
    2D solve.
    """
    return site.depth_er * math.exp(-(site.kappa_per_m**2) * rho_m**2)


def _axial_fd_solve(
    depth_local_er: float, n_states: int, grid_n: int, potential: Literal["cos2", "harmonic"]
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """One finite-difference axial eigensolve at a fixed local depth
    (internal; carries no convergence guard of its own, callers own that).

    Domain `x = k*z in [-pi/2, pi/2]` (one lattice site), Dirichlet
    boundary conditions (`psi=0` at the domain edges, where the true
    `cos^2` potential reaches its zero/maximum, `Beloy`'s own Eq. 13
    integration domain), `grid_n` interior grid points, uniform spacing.
    Energies and the Hamiltonian are both in units of `E_R` (the kinetic
    operator `-d^2/dx^2` has coefficient exactly `1` in these units, since
    `-hbar^2/(2m) d^2/dz^2 = -hbar^2/(2m) * k^2 * d^2/dx^2 = -E_R
    d^2/dx^2`).

    `potential="cos2"`: `v(x) = -depth_local_er * cos(x)^2` (the true site
    potential, Eq. 1). `potential="harmonic"`: `v(x) = -depth_local_er +
    depth_local_er*x^2` (the small-`x` Taylor expansion of the `cos2` form,
    `Beloy`'s Eq. 2's axial part; used only for this module's own
    harmonic-limit consistency tests, never for a physical BO+WKB
    evaluation).

    Returns
    -------
    x_grid : np.ndarray, shape (grid_n,)
    dx : float
    energies_er : np.ndarray, shape (n_states,)
        Ascending, in `E_R` units.
    eigvecs : np.ndarray, shape (grid_n, n_states)
        Columns are eigenvectors, L2-normalized in the discrete sense
        (`sum(v**2) == 1`), equivalent to `integral(|psi(x)|^2 dx) == 1`.
    """
    dx = math.pi / (grid_n + 1)
    x_grid = -math.pi / 2.0 + dx * np.arange(1, grid_n + 1)
    v: np.ndarray
    if potential == "cos2":
        v = np.asarray(-depth_local_er * np.cos(x_grid) ** 2, dtype=np.float64)
    elif potential == "harmonic":
        v = np.asarray(-depth_local_er + depth_local_er * x_grid**2, dtype=np.float64)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown potential {potential!r}")
    diag = (2.0 / dx**2) + v
    offdiag = np.full(grid_n - 1, -1.0 / dx**2)
    n_states = min(n_states, grid_n)
    energies, eigvecs = eigh_tridiagonal(diag, offdiag, select="i", select_range=(0, n_states - 1))
    return x_grid, dx, energies, eigvecs


#: Default axial finite-difference grid ladder for the convergence guard:
#: starting resolution and hard ceiling. Chosen so a typical (`depth_er`
#: 15-150, `n_z` 0-3) solve converges to `AXIAL_ENERGY_TOL_ER` within 1-2
#: doublings (a few milliseconds), while `AXIAL_GRID_N_MAX` still bounds
#: worst-case runtime for a benchmark sweep (`tests/test_lattice_light_shift.py`
#: measures and pins this).
AXIAL_GRID_N0 = 161
AXIAL_GRID_N_MAX = 40961
AXIAL_ENERGY_TOL_ER = 1e-5


def axial_energies_er(
    depth_local_er: float,
    n_states: int,
    *,
    potential: Literal["cos2", "harmonic"] = "cos2",
    tol_er: float = AXIAL_ENERGY_TOL_ER,
    grid_n0: int = AXIAL_GRID_N0,
    max_grid_n: int = AXIAL_GRID_N_MAX,
) -> np.ndarray:
    """Convergence-guarded axial band energies `U_nz(rho)/E_R` at one
    local depth (CONVENTIONS.md E41, Beloy et al. 2020 Eq. 5's
    Born-Oppenheimer axial eigenproblem, solved by finite-difference
    diagonalization: see :func:`_axial_fd_solve`).

    Grid resolution starts at `grid_n0` and doubles (`grid_n -> 2*grid_n+1`)
    until every one of the `n_states` lowest eigenvalues changes by less
    than `tol_er` (in `E_R` units) between successive resolutions, or
    `max_grid_n` is reached, at which point
    :class:`LatticeLightShiftConvergenceError` is raised naming the
    residual. A state whose energy is `>= 0` (unbound at this depth, e.g.
    `n_z` too high for a shallow local depth) is clamped to exactly `0.0`
    (Beloy's own convention for `G_nz`, Eq. 4/11's text: "understood to be
    zero" beyond the bound-state range), and clamped values are treated as
    already converged (no oscillation to chase).

    Parameters
    ----------
    depth_local_er : float
        Local peak depth `D(rho)/E_R` (:func:`_local_depth_er`). If
        `<= 0`, returns an all-zero array immediately (no bound states
        possible; not an error, since :func:`_local_depth_er` legitimately
        reaches `~0` far from the lattice site center).
    n_states : int
        Number of lowest axial bands to return (`n_z = 0, ..., n_states-1`).
        Must be `>= 1`.
    potential : {"cos2", "harmonic"}, default "cos2"
        Passed to :func:`_axial_fd_solve`.
    tol_er : float, default 1e-5
        Convergence tolerance, `E_R` units.
    grid_n0 : int, default 161
    max_grid_n : int, default 40961

    Returns
    -------
    np.ndarray, shape (n_states,)
        Ascending axial band energies, `E_R` units, each `<= 0`.

    Raises
    ------
    ValueError
        `n_states < 1`.
    LatticeLightShiftConvergenceError
        Convergence not reached by `max_grid_n`.
    """
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")
    if depth_local_er <= 0:
        return np.zeros(n_states)

    grid_n = grid_n0
    prev: np.ndarray | None = None
    while True:
        n_avail = min(n_states, grid_n)
        _, _, energies, _ = _axial_fd_solve(depth_local_er, n_avail, grid_n, potential)
        if n_avail < n_states:
            energies = np.concatenate([energies, np.zeros(n_states - n_avail)])
        energies = np.where(energies >= 0.0, 0.0, energies)
        if prev is not None:
            diffs = np.abs(energies - prev)
            if np.all(diffs < tol_er):
                return energies
        if grid_n >= max_grid_n:
            worst = float(np.max(diffs)) if prev is not None else float("nan")
            raise LatticeLightShiftConvergenceError(
                "axial_energies_er failed to converge: grid_n reached max_grid_n="
                f"{max_grid_n} with depth_local_er={depth_local_er}, n_states={n_states}, "
                f"worst residual change {worst:.3e} E_R exceeds tol_er={tol_er:.3e}"
            )
        prev = energies
        grid_n = min(2 * grid_n + 1, max_grid_n)


def _axial_energies_er_at_rho(
    site: SitePotential,
    n_states: int,
    rho_m: float,
    potential: Literal["cos2", "harmonic"],
    *,
    tol_er: float,
    grid_n0: int,
    max_grid_n: int,
) -> np.ndarray:
    """`rho`-dependence dispatch shared by :func:`axial_band_energy_er`,
    :func:`turning_radius_m`, and :func:`bo_wkb_density_of_states`
    (internal).

    `potential="cos2"`: the true site potential factors exactly as
    `-D*exp(-kappa^2*rho^2)*cos^2(kz)` (Eq. 1), so the axial curvature
    itself is reduced by the radial Gaussian factor at each `rho`
    (:func:`_local_depth_er`).

    `potential="harmonic"`: Beloy's Eq. 2 harmonic potential,
    `U_HO(rho,z)/E_R = -depth_er*(1-kappa^2*rho^2) + k^2*z^2*depth_er`, is
    ADDITIVE in `rho` and `z`. The axial curvature (`depth_er*k^2`) stays
    fixed at the peak value for every `rho`, with no Gaussian reduction;
    only a rigid, `n_z`-independent offset `depth_er*kappa^2*rho^2` shifts
    with `rho`. This is Beloy's own derived closed form
    (transcribed in :func:`harmonic_density_of_states_closed_form`'s
    module-level context, "`Unz(rho) -> -D + D*kappa^2*rho^2 +
    2*sqrt(D*ER)*(nz+1/2)`"): the axial spectrum at the PEAK depth
    (`site.depth_er`, computed once, `rho`-independent) plus the additive
    offset `site.depth_er*(kappa*rho)^2`.
    """
    if potential == "cos2":
        depth_local = _local_depth_er(site, rho_m)
        return axial_energies_er(
            depth_local,
            n_states,
            potential="cos2",
            tol_er=tol_er,
            grid_n0=grid_n0,
            max_grid_n=max_grid_n,
        )
    base = axial_energies_er(
        site.depth_er,
        n_states,
        potential="harmonic",
        tol_er=tol_er,
        grid_n0=grid_n0,
        max_grid_n=max_grid_n,
    )
    offset = site.depth_er * (site.kappa_per_m * rho_m) ** 2
    return base + offset


def axial_band_energy_er(
    site: SitePotential,
    n_z: int,
    rho_m: float,
    *,
    potential: Literal["cos2", "harmonic"] = "cos2",
    tol_er: float = AXIAL_ENERGY_TOL_ER,
    grid_n0: int = AXIAL_GRID_N0,
    max_grid_n: int = AXIAL_GRID_N_MAX,
) -> float:
    """`U_nz(rho)`, the `n_z`-th axial band energy at radius `rho` (`E_R`
    units) (CONVENTIONS.md E41).

    `potential="cos2"` (default): the true site potential (Eq. 1).
    `potential="harmonic"`: Beloy's Eq. 2 harmonic approximation, used
    ONLY for this module's own harmonic-limit consistency tests (never for
    a physical BO+WKB evaluation); see
    :func:`_axial_energies_er_at_rho` for the rho-dependence this
    distinction implies.
    """
    if n_z < 0:
        raise ValueError(f"n_z must be >= 0, got {n_z}")
    energies = _axial_energies_er_at_rho(
        site, n_z + 1, rho_m, potential, tol_er=tol_er, grid_n0=grid_n0, max_grid_n=max_grid_n
    )
    return float(energies[n_z])


def harmonic_density_of_states_closed_form(site: SitePotential, n_z: int, energy_j: float) -> float:
    """The harmonic-oscillator density of states `G^HO_nz(E)`
    (CONVENTIONS.md E41, Beloy et al. 2020 Eq. 4, transcribed verbatim
    from the typeset PDF):

        G^HO_nz(E) = (kappa/k)^(-2) / (4*D*E_R)
                     * [E + D - 2*sqrt(D*E_R)*(n_z + 1/2)]

    with `D` the site's peak depth (joules), `E_R` the recoil energy
    (joules), `E` the argument (joules, `E < 0`). Only non-negative
    bracket values are physical; this function clamps a negative bracket
    to `0.0` (Beloy's own stated convention: "`G^HO_nz(E)` understood to
    be zero when the right-hand-side... returns negative values").

    Used as the closed-form reference this module's numeric BO+WKB
    density of states (:func:`bo_wkb_density_of_states`, Eq. 11) must
    reduce to, both algebraically (feeding the closed-form harmonic
    turning radius directly into Eq. 11's formula) and numerically
    (running the same finite-difference axial solver with
    `potential="harmonic"`), per Beloy's own Section VI consistency check.

    Parameters
    ----------
    site : SitePotential
    n_z : int
        Must be `>= 0`.
    energy_j : float
        `E`, joules (typically `<= 0`).

    Returns
    -------
    float
        `G^HO_nz(E)`, states per joule, `>= 0`.
    """
    if n_z < 0:
        raise ValueError(f"n_z must be >= 0, got {n_z}")
    d_joules = site.depth_er * site.recoil_energy_j_value
    e_r = site.recoil_energy_j_value
    kappa_over_k_sq = (site.kappa_per_m / site.k_per_m) ** 2
    bracket = energy_j + d_joules - 2.0 * math.sqrt(d_joules * e_r) * (n_z + 0.5)
    if bracket <= 0.0:
        return 0.0
    return bracket / (kappa_over_k_sq * 4.0 * d_joules * e_r)


def turning_radius_m(
    site: SitePotential,
    n_z: int,
    energy_er: float,
    *,
    potential: Literal["cos2", "harmonic"] = "cos2",
    rho_search_max_m: float | None = None,
    tol_er: float = AXIAL_ENERGY_TOL_ER,
) -> float:
    """`R_nz(E)`, the classical turning radius (Beloy et al. 2020's
    notation): the radius at which the `n_z`-th axial band energy
    `U_nz(rho)` equals the given `energy_er` (CONVENTIONS.md E41; `Rnz` is
    the inverse function of `Unz`, `Unz(Rnz(E)) = E`, stated directly
    below their Eq. 10).

    Solved by bracketing (`rho=0` where `U_nz(0)` is the band's most
    negative energy, expanding outward geometrically until
    `axial_band_energy_er(site, n_z, rho) >= energy_er`, i.e. the band
    closes) then :func:`scipy.optimize.brentq`.

    Parameters
    ----------
    site : SitePotential
    n_z : int
    energy_er : float
        Target energy, `E_R` units. Must satisfy
        `axial_band_energy_er(site, n_z, 0.0) <= energy_er <= 0`, i.e. lie
        within the band's actual range (raises `ValueError` naming both
        bounds otherwise).
    rho_search_max_m : float | None, default None
        Upper bound for the outward bracket search, meters. Defaults to
        `10 * site.waist_m` (comfortably beyond where any bound axial
        state persists for realistic depths).
    tol_er : float, default 1e-5
        Passed through to the underlying axial solves.

    Returns
    -------
    float
        `R_nz(E)`, meters, `>= 0`.
    """
    rho_max = rho_search_max_m if rho_search_max_m is not None else 10.0 * site.waist_m

    def u_nz_er(rho_m: float) -> float:
        return axial_band_energy_er(site, n_z, rho_m, potential=potential, tol_er=tol_er)

    e0 = u_nz_er(0.0)
    if not (e0 - 1e-9 <= energy_er <= 1e-9):
        raise ValueError(
            f"energy_er={energy_er} outside band range [{e0}, 0] for n_z={n_z} "
            f"(site.depth_er={site.depth_er})"
        )
    if energy_er >= -1e-12:
        # E == 0 (top of band): the turning radius is where the band closes.
        rho = rho_max / 2.0
        step = rho_max / 4.0
        for _ in range(200):
            if u_nz_er(rho) >= 0.0:
                rho -= step
            else:
                rho += step
            step /= 2.0
            if step < 1e-12 * site.waist_m:
                break
        return max(rho, 0.0)

    def f(rho_m: float) -> float:
        return u_nz_er(rho_m) - energy_er

    lo, hi = 0.0, rho_max
    if f(0.0) > 0.0:
        # Numerical edge case: even rho=0 already exceeds the target energy
        # (energy_er extremely close to e0); the turning radius is ~0.
        return 0.0
    f_hi = f(hi)
    expansions = 0
    while f_hi < 0.0 and expansions < 60:
        hi *= 1.5
        f_hi = f(hi)
        expansions += 1
    if f_hi < 0.0:
        raise LatticeLightShiftConvergenceError(
            f"turning_radius_m failed to bracket a root within rho <= {hi} m "
            f"for n_z={n_z}, energy_er={energy_er}"
        )
    return float(brentq(f, lo, hi, xtol=1e-12 * site.waist_m))


def bo_wkb_density_of_states(
    site: SitePotential,
    n_z: int,
    energy_j: float,
    *,
    potential: Literal["cos2", "harmonic"] = "cos2",
    tol_er: float = AXIAL_ENERGY_TOL_ER,
) -> float:
    """The general BO+WKB density of states `G_nz(E)` (CONVENTIONS.md
    E41, Beloy et al. 2020 Eq. 11, transcribed verbatim from the typeset
    PDF):

        G_nz(E) = (1/4) * (2*m/hbar^2) * [R_nz(E)]^2

    with `R_nz(E)` the classical turning radius (:func:`turning_radius_m`)
    and `m` the atomic mass. Zero outside the band's range (`E < U_nz(0)`
    or `E > 0`).

    `potential="harmonic"` runs this same Eq.-11 machinery against Beloy's
    Eq. 2 harmonic approximation
    (:func:`_axial_energies_er_at_rho`'s harmonic branch); this is the
    numeric half of this module's Eq.-4/Eq.-11 consistency check
    (`tests/test_lattice_light_shift.py`): the result must equal
    :func:`harmonic_density_of_states_closed_form` (Eq. 4) to numerical
    precision, mirroring Beloy et al. 2020's own Section VI derivation.

    Parameters
    ----------
    site : SitePotential
    n_z : int
    energy_j : float
        `E`, joules.
    potential : {"cos2", "harmonic"}, default "cos2"
    tol_er : float, default 1e-5

    Returns
    -------
    float
        `G_nz(E)`, states per joule, `>= 0`.
    """
    e_r_j = site.recoil_energy_j_value
    energy_er = energy_j / e_r_j
    e0 = axial_band_energy_er(site, n_z, 0.0, potential=potential, tol_er=tol_er)
    if energy_er < e0 or energy_er > 0.0:
        return 0.0
    r_nz = turning_radius_m(site, n_z, energy_er, potential=potential, tol_er=tol_er)
    return 0.25 * (2.0 * site.mass_kg / HBAR**2) * r_nz**2


# ---------------------------------------------------------------------------
# Thermal shape factors (X, Y, Z) and the BO+WKB light shift
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalShapeFactors:
    """Result of :func:`axial_thermal_factors`: the ensemble-averaged
    trap-depth-reduction factors `X(n_z, u0, Tr)`, `Y(n_z, u0, Tr)`,
    `Z(n_z, u0, Tr)` (Beloy et al. 2020 Eqs. 19-21; Aeppli et al. 2024's
    and Bothwell et al. 2025's own notation for the same quantities).

    Attributes
    ----------
    x_nz, y_nz, z_nz : float
        The three factors, each restricted to `[0, 1]` (Beloy's own stated
        range). `X`/`Z -> 1` and `Y -> 0` in the deep-and-cold limit (an
        atom localized at the site center).
    rho_max_m : float
        `R_nz(0)`, the band's turning radius at `E=0`, meters (the
        integration domain's upper bound).
    n_rho_points : int
        Radial grid points used at convergence.
    axial_grid_n : int
        Axial finite-difference grid resolution used at convergence.
    """

    x_nz: float
    y_nz: float
    z_nz: float
    rho_max_m: float
    n_rho_points: int
    axial_grid_n: int


def _shape_integrand_at_rho(
    site: SitePotential, n_z: int, rho_m: float, grid_n: int
) -> tuple[float, float, float, float] | None:
    """One `rho` sample's `(x_nz(rho), y_nz(rho), z_nz(rho), U_nz(rho)/E_R)`
    (Beloy et al. 2020's Eq. 13 dimensionless shape factors, evaluated
    from the axial eigenvector at this `rho`), or `None` if the band is
    not bound at this `rho` (local depth too shallow for `n_z`).
    """
    depth_local = _local_depth_er(site, rho_m)
    if depth_local <= 0.0:
        return None
    _, _, energies, eigvecs = _axial_fd_solve(depth_local, n_z + 1, grid_n, "cos2")
    if n_z >= len(energies) or energies[n_z] >= 0.0:
        return None
    dx = math.pi / (grid_n + 1)
    x_grid = -math.pi / 2.0 + dx * np.arange(1, grid_n + 1)
    v = eigvecs[:, n_z]
    v2 = v * v
    cos2 = np.cos(x_grid) ** 2
    sin2 = 1.0 - cos2
    cos4 = cos2 * cos2
    kappa2_rho2 = (site.kappa_per_m * rho_m) ** 2
    x_nz = math.exp(-kappa2_rho2) * float(np.sum(v2 * cos2))
    y_nz = math.exp(-kappa2_rho2) * float(np.sum(v2 * sin2))
    z_nz = math.exp(-2.0 * kappa2_rho2) * float(np.sum(v2 * cos4))
    return x_nz, y_nz, z_nz, float(energies[n_z])


#: Convergence-guard ladder for :func:`axial_thermal_factors`: starting
#: radial-grid point count and axial finite-difference grid, doubled until
#: the resulting `(X, Y, Z)` triple stabilizes within
#: `THERMAL_FACTOR_TOL` or `MAX_RHO_POINTS`/`AXIAL_GRID_N_MAX` is reached.
RHO_POINTS0 = 41
MAX_RHO_POINTS = 641
THERMAL_FACTOR_TOL = 1e-4


def axial_thermal_factors(
    site: SitePotential,
    n_z: int,
    radial_temperature_k: float,
    *,
    rho_points0: int = RHO_POINTS0,
    max_rho_points: int = MAX_RHO_POINTS,
    axial_grid_n: int = AXIAL_GRID_N0,
    tol: float = THERMAL_FACTOR_TOL,
) -> ThermalShapeFactors:
    """Ensemble-averaged (Boltzmann, radial temperature `Tr`, fixed axial
    band `n_z`) trap-depth-reduction factors `X`, `Y`, `Z` (CONVENTIONS.md
    E41, Beloy et al. 2020 Eq. 21, transcribed verbatim from the typeset
    PDF, specialized to the isolated-`n_z` form the text gives):

        X_nz = integral_0^Rnz(0) [ xnz(rho)*rho*(exp(-Unz(rho)/kB*Tr) - 1) ] drho
               / integral_0^Rnz(0) [ rho*(exp(-Unz(rho)/kB*Tr) - 1) ] drho

    (analogous forms for `Y`, `Z` with `y_nz(rho)`, `z_nz(rho)` in the
    numerator, same denominator).

    **Numerical stability (the reason for this function's own explicit
    convergence guard, beyond the axial solver's own).** `Unz(rho)` is
    negative and can be tens of `E_R` in magnitude at `rho=0`; evaluating
    `exp(-Unz(rho)/kB*Tr)` directly overflows float64 whenever
    `|Unz(0)|/(kB*Tr)` exceeds about 700 (routine for realistic deep,
    cold lattices: `kB*Tr` a few `E_R` and `|Unz(0)|` tens of `E_R` already
    gives exponents order 10-1000). This function instead factors out the
    common, dominant exponential `exp(-Unz(0)/kB*Tr)` (algebraically exact,
    since it appears identically in both the numerator and denominator
    integrals and cancels in the ratio, so this function never computes it
    explicitly) and integrates the SHIFTED, bounded integrand
    `exp(-(Unz(rho)-Unz(0))/kB*Tr) - exp(Unz(0)/kB*Tr)`, which is
    `<= 1` everywhere on the integration domain by construction (`Unz(rho)
    >= Unz(0)`, and `exp(Unz(0)/kB*Tr) <= 1` since `Unz(0) <= 0`). The
    ratio of the two shifted integrals equals `X_nz`/`Y_nz`/`Z_nz` exactly:
    this is a reformulation with the same value as the original ratio,
    computed on numbers float64 can represent.

    **Species trap.** `X`/`Y`/`Z` cancel `site.mass_kg` and `site.waist_m`
    exactly out of their defining ratio, so neither affects the result; the
    species' own recoil energy `E_R` (:func:`make_site_potential`) still
    enters through the `kB*Tr/E_R` thermal-weighting ratio above. Reusing a
    published `(u0, Tr)` pair with the wrong species' `E_R` silently
    evaluates a different physical trap depth than intended: an earlier
    build of this module's own Target 3a case used Sr-87's `E_R` against
    a Yb-171 table, landing 5-15% off the published BO+WKB column and
    closer to the published harmonic column instead.

    Radial integration is trapezoidal on a `rho` grid from `0` to `Rnz(0)`
    (:func:`turning_radius_m` at `E=0`); the grid point count and the
    shared axial finite-difference resolution are both doubled (with the
    same "run at two resolutions, compare" discipline as
    :func:`axial_energies_er`) until `X`/`Y`/`Z` each change by less than
    `tol` between successive resolutions, or `max_rho_points`/
    `AXIAL_GRID_N_MAX` is reached, at which point
    :class:`LatticeLightShiftConvergenceError` is raised.

    Parameters
    ----------
    site : SitePotential
    n_z : int
        Must be `>= 0`.
    radial_temperature_k : float
        `Tr`, kelvin. Must be `> 0` (a `Tr=0` ensemble is a single point,
        `rho=0`, handled as the exact `X=Z=1, Y=0` limit without running
        the integrator).
    rho_points0 : int, default 41
    max_rho_points : int, default 641
    axial_grid_n : int, default 161
        Starting axial finite-difference resolution (also doubled).
    tol : float, default 1e-4
        Convergence tolerance on each of `X`, `Y`, `Z`.

    Returns
    -------
    ThermalShapeFactors

    Raises
    ------
    ValueError
        `n_z < 0` or `radial_temperature_k < 0`.
    LatticeLightShiftConvergenceError
        Convergence not reached by `max_rho_points`/`AXIAL_GRID_N_MAX`, or
        the band is not bound at `rho=0` for the given `site`/`n_z`.
    """
    if n_z < 0:
        raise ValueError(f"n_z must be >= 0, got {n_z}")
    if radial_temperature_k < 0:
        raise ValueError(f"radial_temperature_k must be >= 0, got {radial_temperature_k}")

    u0_sample = _shape_integrand_at_rho(site, n_z, 0.0, axial_grid_n)
    if u0_sample is None:
        raise LatticeLightShiftConvergenceError(
            f"axial band n_z={n_z} is not bound at rho=0 for site.depth_er={site.depth_er}"
        )
    if radial_temperature_k == 0.0:
        x0, y0, z0, _ = u0_sample
        return ThermalShapeFactors(
            x_nz=x0, y_nz=y0, z_nz=z0, rho_max_m=0.0, n_rho_points=1, axial_grid_n=axial_grid_n
        )

    kt = BOLTZMANN_K * radial_temperature_k

    def compute(n_rho: int, grid_n: int) -> tuple[float, float, float, float]:
        rho_max = turning_radius_m(site, n_z, 0.0)
        rhos = np.linspace(0.0, rho_max, n_rho)
        x_arr = np.zeros(n_rho)
        y_arr = np.zeros(n_rho)
        z_arr = np.zeros(n_rho)
        u_arr = np.zeros(n_rho)
        u0_er = None
        for i, rho in enumerate(rhos):
            sample = _shape_integrand_at_rho(site, n_z, float(rho), grid_n)
            if sample is None:
                # Beyond the band's support (can happen at the very last
                # point due to floating-point roundoff at rho_max): the
                # band has zero weight there, matching Beloy's convention.
                x_arr[i] = x_arr[i - 1] if i > 0 else 1.0
                y_arr[i] = y_arr[i - 1] if i > 0 else 0.0
                z_arr[i] = z_arr[i - 1] if i > 0 else 1.0
                u_arr[i] = 0.0
                continue
            x_arr[i], y_arr[i], z_arr[i], u_er = sample
            u_arr[i] = u_er
            if i == 0:
                u0_er = u_er
        assert u0_er is not None
        u0_j = u0_er * site.recoil_energy_j_value
        shifted = np.exp(-(u_arr * site.recoil_energy_j_value - u0_j) / kt) - math.exp(u0_j / kt)
        weight = rhos * shifted
        denom = np.trapezoid(weight, rhos)
        num_x = np.trapezoid(x_arr * weight, rhos)
        num_y = np.trapezoid(y_arr * weight, rhos)
        num_z = np.trapezoid(z_arr * weight, rhos)
        return float(num_x / denom), float(num_y / denom), float(num_z / denom), rho_max

    n_rho, grid_n = rho_points0, axial_grid_n
    prev = compute(n_rho, grid_n)
    while True:
        n_rho_next = min(2 * n_rho - 1, max_rho_points)
        grid_n_next = min(2 * grid_n + 1, AXIAL_GRID_N_MAX)
        cur = compute(n_rho_next, grid_n_next)
        residual = max(abs(cur[0] - prev[0]), abs(cur[1] - prev[1]), abs(cur[2] - prev[2]))
        if residual < tol:
            return ThermalShapeFactors(
                x_nz=cur[0],
                y_nz=cur[1],
                z_nz=cur[2],
                rho_max_m=cur[3],
                n_rho_points=n_rho_next,
                axial_grid_n=grid_n_next,
            )
        if n_rho_next == n_rho and grid_n_next == grid_n:
            raise LatticeLightShiftConvergenceError(
                "axial_thermal_factors failed to converge: reached max_rho_points="
                f"{max_rho_points}/AXIAL_GRID_N_MAX={AXIAL_GRID_N_MAX} with residual "
                f"{residual:.3e} exceeding tol={tol:.3e}"
            )
        n_rho, grid_n, prev = n_rho_next, grid_n_next, cur


def bo_wkb_fractional_light_shift(
    n_z: int,
    u0: float,
    detuning_hz: float,
    radial_temperature_k: float,
    coeffs: HarmonicLatticeCoefficients,
    site: SitePotential,
) -> tuple[float, ThermalShapeFactors]:
    """Model B's fractional light shift `delta_nu_LS/nu_c` (CONVENTIONS.md
    E41, Bothwell et al. 2025 Eq. 6, transcribed verbatim from the typeset
    PDF's Appendix A, specialized to a single dominant band `W_nz=1`):

        delta_nu_LS/nu_c ~= -[ (d(alpha~E1)/dnu)*delta_L*X(n_z,u0,Tr)*u0
                              + alpha~M1E2*Y(n_z,u0,Tr)*u0
                              + beta~*Z(n_z,u0,Tr)*u0^2 ]

    with `X`/`Y`/`Z` from :func:`axial_thermal_factors` (Beloy's Eqs.
    19-21, the same factors Bothwell's Eq. 6 cites Beloy for). **Unit
    contract, stated explicitly because it differs from
    :func:`harmonic_light_shift_hz`'s**: `coeffs` here must be in
    Bothwell's OWN normalization (`BOTHWELL_2025_YB171_HARMONIC`/
    `_BOWKB`'s convention: `e1_slope_per_hz` already per-hertz-of-`nu_c`,
    `m1e2_hz`/`hyperpolarizability_hz` already dimensionless fractions),
    NOT the Ushijima/Kim `.../h` hertz convention
    `harmonic_light_shift_hz` expects. Passing a
    `USHIJIMA_2018_SR87`/`KIM_2023_SR87`-style coefficient set here would
    silently produce a result in the wrong units by a factor of `nu_c`;
    this function does not (cannot, in general, without a species-specific
    `nu_c`) detect that mismatch, so callers must use the matching
    constant.

    Parameters
    ----------
    n_z : int
    u0 : float
        Peak reduced trap depth. Also used as `site.depth_er`'s expected
        value; `site` must already be built with this `depth_er` (not
        re-derived here, to keep this function a pure evaluator over an
        already-constructed `SitePotential`).
    detuning_hz : float
        `delta_L`, hertz.
    radial_temperature_k : float
        `Tr`, kelvin.
    coeffs : HarmonicLatticeCoefficients
        Bothwell-convention coefficients (see unit-contract note above).
    site : SitePotential
        Must have `site.depth_er == u0` (raises `ValueError` otherwise).

    Returns
    -------
    tuple[float, ThermalShapeFactors]
        `(delta_nu_LS/nu_c, the ThermalShapeFactors used)`.
    """
    if abs(site.depth_er - u0) > 1e-9 * max(1.0, abs(u0)):
        raise ValueError(f"site.depth_er ({site.depth_er}) must equal u0 ({u0})")
    factors = axial_thermal_factors(site, n_z, radial_temperature_k)
    shift = -(
        coeffs.e1_slope_per_hz * detuning_hz * factors.x_nz * u0
        + coeffs.m1e2_hz * factors.y_nz * u0
        + coeffs.hyperpolarizability_hz * factors.z_nz * u0**2
    )
    return shift, factors
