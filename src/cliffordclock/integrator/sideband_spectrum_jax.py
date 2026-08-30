# SPDX-License-Identifier: AGPL-3.0-or-later
"""WP38: the sideband-spectrum forward model (CONVENTIONS.md section 17
addendum, E42), differentiable end to end on JAX. This module computes
the clock-transition excitation spectrum of a 1D lattice clock, carrier
plus red/blue axial sidebands, as a function of the physical trap and
motional-state parameters (peak depth `u0`, radial temperature `Tr`,
longitudinal temperature `Tz`, free-space Rabi frequency, probe
detuning), building on the already-gated BO+WKB core
(`cliffordclock.integrator.lattice_light_shift_jax`, E41).

**Two lineshape paths, both implemented, each its own labeled function.**
Blatt, Thomsen, Campbell, Ludlow, Swallows, Martin, Boyd, Ye, "Rabi
spectroscopy and excitation inhomogeneity in a one-dimensional optical
lattice clock," PRA 80, 052703 (2009), arXiv:0906.1419 (every equation
number below transcribed directly from the typeset PDF, all 12 pages
read page-image by page-image, not from an ar5iv summary), derives the
field's standard sideband lineshape from a PERTURBED HARMONIC OSCILLATOR
model of the lattice site: harmonic radial and axial confinement plus a
first-order quartic/coupling perturbation (their Eq. 2). This is the
**harmonic path** (:func:`harmonic_carrier_excitation_probability`,
:func:`harmonic_sideband_shape`, :func:`harmonic_full_spectrum`): the
VALIDATION ANCHOR, since it is the field's sixteen-year incumbent
methodology, confirmed still current by a 2025 primary source (Goti et
al. 2025's own introduction, quoted in this project's private research
dossier: "The most common technique in 1D optical lattice clocks is
sideband spectroscopy [Blatt et al. 2009]").

Goti, Petrucciani, Condio, Levi, Calonico, Pizzocaro, "Atomic thermometry
in optical lattice clocks," arXiv:2508.08164 (v2, 2 Sept 2025), INRIM,
builds the sideband lineshape directly on Beloy et al. 2020's
Born-Oppenheimer+WKB (BO+WKB) motional spectrum (their Section II.B,
Eqs. 5-9, transcribed below, all 8 relevant pages read directly from the
typeset PDF), replacing the harmonic model's quantized radial motion
with the true `cos^2` site potential's WKB density of states and
locating each transition at its classical Franck-Condon point, a WKB
construction that stands in for direct diagonalization. This is the
**BO+WKB path**
(:func:`bowkb_sideband_shape`, :func:`bowkb_full_spectrum`): the
CAPABILITY, since Goti et al. 2025's own abstract reports "discrepancies
up to a factor of two in extracted temperatures" between the two models,
and no prior publication (this project's own research dossier's
sweep, corroborated by Beloy et al. 2020's own stated conclusion) fits a
BO+WKB-class sideband lineshape to spectroscopy by GRADIENT-based
optimization; INRIM's own `large-lattice-model` (Deliverable 2 below)
fits it by conventional finite-difference-Jacobian least squares, a real
but non-differentiable prior art.

**Where the two paths agree and where they structurally part ways.**
Both papers use the IDENTICAL formula for the harmonic-oscillator
perturbed energy spectrum (Blatt et al. 2009 Eq. 3; Goti et al. 2025
Eq. 1, restated with `h` factored out) and the identical longitudinal
energy-gap formula (Blatt Eq. 8; Goti Eq. 2): this module implements that
SHARED formula once
(:func:`blue_sideband_detuning_hz`/:func:`red_sideband_detuning_hz`,
:func:`longitudinal_energy_hz`) and both papers' own sideband-population
weighting is the harmonic path's own (Blatt Eqs. 19-20 for the carrier;
Goti Eq. 4, itself an explicit generalization of Blatt's Appendix A, for
the sideband). Where the two papers diverge is the BO+WKB model's own
reason for existing: Goti et al. 2025's Eqs. 5-9 replace the
harmonic model's `(n_x, n_y)`-quantized radial sum with a WKB density of
states over a continuous radial energy `E`, using the SAME `G_nz(E)`
object Beloy et al. 2020's Eq. 11 (already implemented and G18/G19-gated
in this project as
:func:`~cliffordclock.integrator.lattice_light_shift_jax.bo_wkb_density_of_states_jax`)
defines, and locates the transition frequency at the Franck-Condon
(classical turning-point) detuning. This replaces the harmonic path's
quantized `(n_r, l)` sideband position with a continuous energy
integral. No carrier formula is given by either the BO+WKB
paper (Beloy et al. 2020) or Goti et al. 2025: this module's carrier
component (:func:`harmonic_carrier_excitation_probability`) is shared
by both full-spectrum functions
(:func:`harmonic_full_spectrum`/:func:`bowkb_full_spectrum`), since
neither paper proposes a distinct BO+WKB carrier treatment and the
Lamb-Dicke-regime carrier is dominated by the ground axial band, where
the harmonic and true `cos^2` potentials still agree closely (this
project's own G18-gated harmonic-limit consistency check).

**Why the BO+WKB sideband path needs its own numerical route, distinct
from `lattice_light_shift_jax`'s `turning_radius_m_jax`.** A single
sideband-spectrum evaluation needs the Franck-Condon detuning
(Goti Eq. 5) at MANY radial energies `E` per axial band (a fixed
quadrature over the band's bound-energy range, `N_E_QUAD` per band), for
several bands (`N_Z_MAX_BOWKB + 1` of them, plus each band's
`+1`/`-1` neighbor). `turning_radius_m_jax`'s own bisection root-find
(`BISECTION_ITERS = 60`) evaluates the axial eigenproblem ONCE PER
BISECTION STEP, so reusing it directly here would mean roughly
`N_Z_MAX_BOWKB * N_E_QUAD * BISECTION_ITERS` dense
`AXIAL_GRID_N_JAX = 1281` eigendecompositions per spectrum call. That
is the right cost for the G18/G19 gate's 1e-19-level light-shift target,
evaluated once at a handful of published operating points, and the
wrong cost for a spectrum evaluated (and differentiated, and fit) at
many detunings and many candidate `(u0, Tr)` pairs. This module instead
precomputes each needed band's `U_nz(rho)/E_R` on a SMALL, FIXED radial
grid (`RHO_TABLE_N` points, one batched `jax.vmap` over `jax.numpy.linalg.eigh`
per band, at a resolution `AXIAL_GRID_N_SPECTRUM` chosen for this
module's own accuracy target, smaller than `AXIAL_GRID_N_JAX`) and finds
the turning radius / Franck-Condon point by linear interpolation
(`jax.numpy.interp`) against that table, monotonic by construction
(`U_nz(rho)` increases from the band bottom to `0` as `rho` grows). This
turns `N_Z_MAX_BOWKB * N_E_QUAD * BISECTION_ITERS` dense eigensolves
into `(N_Z_MAX_BOWKB + 2) * RHO_TABLE_N` of them (a factor of order
`BISECTION_ITERS * N_E_QUAD / RHO_TABLE_N`, roughly two orders of
magnitude, fewer), all batchable as one `jax.vmap` call per band.
`jax.numpy.interp` is a standard, differentiable linear interpolant: its
gradient with respect to the table's own VALUES (which is how `u0`'s
dependence enters, since the table is rebuilt from `u0` on every trace)
flows through the ordinary linear-interpolation weights; no
`jax.lax.custom_root`/implicit-function-theorem machinery is needed
here, unlike `turning_radius_m_jax`'s own bisection-based root-find,
because the table itself is already a smooth (piecewise-linear) function
of `u0` with no root-finding step in the loop.

**Chosen resolution, and its verified error bound (offline; see
`tests/test_sideband_spectrum_jax.py::TestOfflineConvergenceStudy`,
mirroring G19's own documented pattern for
`lattice_light_shift_jax.py`).** `AXIAL_GRID_N_SPECTRUM = 321` and
`RHO_TABLE_N = 129` are verified, at Bothwell et al. 2025's own four
Table I Yb-171 operating points (the same points G18/G19 validate
against), by comparing this module's table-interpolated Franck-Condon
detuning directly against `lattice_light_shift_jax.turning_radius_m_jax`'s
own bisection-based turning radius (`AXIAL_GRID_N_JAX = 1281`,
`BISECTION_ITERS = 60`) at a spot-check grid of energies within each
band: worst-case relative disagreement is reported in the offline
study's own docstring. This module's resolution is chosen for
SPECTRUM-SCALE evaluation, fast enough to appear inside a
`scipy.optimize.minimize` inner loop (Deliverable 3). The G18/G19
gate's own 1e-19 light-shift precision target needs
`lattice_light_shift_jax`'s own tighter, adaptive resolution; a caller
needing that precision should call that module directly.

**Static shapes for jit; float64 throughout** (inherited from
`cliffordclock`'s own package-level `jax_enable_x64` configuration, the
same dependency `lattice_light_shift_jax`'s own module docstring
states). Every quantum-number sum and energy quadrature in this module
is truncated at a fixed, static Python-int bound
(`N_Z_MAX_HARMONIC`/`N_R_MAX_HARMONIC` for the harmonic path,
`N_Z_MAX_BOWKB`/`N_E_QUAD` for the BO+WKB path), chosen generously
against the Boltzmann suppression at realistic operating temperatures
and verified offline (see the convergence-study tests). The bound stays
fixed across every call, the same static-shape discipline
`lattice_light_shift_jax`'s own module docstring establishes for
`jax.jit`/`jax.grad` compatibility.

**Amplitude convention.** Neither paper fixes one. Blatt et al. 2009's
own sideband cross section (their Eq. 10/11, Appendix A) is stated
"proportional to" (`\\propto`), a shape with a scale left open. Their
own Fig. 2 fits the carrier and each sideband as separate curves, each
its own independently fitted amplitude. This module follows that same
practice: :func:`harmonic_sideband_shape`/:func:`bowkb_sideband_shape`
each return a POPULATION-NORMALIZED (quantum-number weights summing to
`1`) convex combination of unit-height Lorentzians, so the returned
shape is bounded in `[0, 1]` and its own peak height is set entirely by
how sharply peaked the population/lineshape combination is (deep, cold
traps concentrate population near the sideband edge and peak closer to
`1`; hot, shallow traps spread it out and peak lower), with NO free
amplitude built in.
:func:`harmonic_full_spectrum`/:func:`bowkb_full_spectrum` take explicit
`blue_amplitude`/`red_amplitude` scale arguments, multiplying this
shape and mirroring Blatt et al. 2009's own per-feature fitting
practice. Every call site names its own amplitude value directly.

**Scope boundary.** This module is a forward model, a cross-validation
target (Deliverable 2, `benchmarks/run_sideband_spectrum.py`), and a
fitting demonstration (Deliverable 3, `benchmarks/run_sideband_fit.py`).
It carries no `cliffordclock.pipeline` config surface, the same
phase-boundary discipline E40/E41 already establish. No spontaneous-
emission/probe-decoherence physics (Blatt et al. 2009's own stated
regime, "we can
neglect any decoherence rates in the system," their Section V, valid for
their <1 s pulse times against a ~1 s lattice lifetime); no probe
misalignment radial sideband contribution (Blatt's own `eta_x` Lamb-Dicke
parameter is carried through the carrier for completeness but this
module's sideband functions assume an aligned probe, `Delta_theta = 0`,
matching the dominant term in Blatt's own Eq. 16 discussion); no
density-shift/collisional physics (Blatt et al. 2009 Section VI, Eqs.
24-33, a separate physical effect the same paper covers alongside the
sideband lineshape).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cliffordclock.constants import BOLTZMANN_K, HBAR, PLANCK_H
from cliffordclock.integrator.lattice_light_shift_jax import (
    SitePotentialJax,
    _axial_band_energy_er_at_rho,
    _axial_grid,
    make_site_potential_jax,
    recoil_energy_j_jax,
)

__all__ = [
    "AXIAL_GRID_N_SPECTRUM",
    "RHO_TABLE_N",
    "N_Z_MAX_HARMONIC",
    "N_R_MAX_HARMONIC",
    "N_X_MAX_CARRIER",
    "N_Z_MAX_BOWKB",
    "N_E_QUAD",
    "blatt_trap_frequencies_hz",
    "longitudinal_energy_hz",
    "blue_sideband_detuning_hz",
    "red_sideband_detuning_hz",
    "goti_e00_hz",
    "laguerre_values",
    "carrier_rabi_frequency_hz",
    "harmonic_carrier_excitation_probability",
    "harmonic_sideband_shape",
    "harmonic_full_spectrum",
    "BandEnergyTable",
    "build_band_energy_table",
    "condon_point_m",
    "condon_detuning_hz",
    "bowkb_density_of_states_from_table",
    "bowkb_sideband_shape",
    "bowkb_full_spectrum",
]

# ---------------------------------------------------------------------------
# Static resolution choices (offline-validated; see module docstring and
# tests/test_sideband_spectrum_jax.py::TestOfflineConvergenceStudy)
# ---------------------------------------------------------------------------

#: Axial finite-difference grid for THIS module's own band-energy tables,
#: smaller than `lattice_light_shift_jax.AXIAL_GRID_N_JAX` (1281): a
#: spectrum evaluation needs many band-energy samples per call, a
#: different resolution trade-off from a single 1e-19-precision
#: light-shift evaluation. See module docstring.
AXIAL_GRID_N_SPECTRUM = 321

#: Radial grid points per band-energy table (:func:`build_band_energy_table`).
RHO_TABLE_N = 129

#: Static quantum-number cutoffs for the harmonic path's `(n_z, n_r)` sum
#: (:func:`harmonic_sideband_shape`) and the carrier's `(n_x, n_z)` sum
#: (:func:`harmonic_carrier_excitation_probability`). Chosen so the
#: Boltzmann weight at the cutoff is negligible (`< 1e-6` of the peak
#: weight) across this module's own offline-verified temperature/depth
#: domain (`tests/test_sideband_spectrum_jax.py::TestOfflineConvergenceStudy`).
N_Z_MAX_HARMONIC = 8
N_R_MAX_HARMONIC = 400
N_X_MAX_CARRIER = 40

#: Static axial-band cutoff and per-band energy-quadrature point count for
#: the BO+WKB path (:func:`bowkb_sideband_shape`).
N_Z_MAX_BOWKB = 5
N_E_QUAD = 96

#: Outward radial bracket multiple for each band-energy table, matching
#: `lattice_light_shift_jax.DEFAULT_RHO_BRACKET_WAIST_MULTIPLE`.
RHO_BRACKET_WAIST_MULTIPLE = 10.0


# ---------------------------------------------------------------------------
# Shared harmonic-oscillator motional spectrum (Blatt et al. 2009 Eqs. 3-8;
# Goti et al. 2025 Eqs. 1-2, the identical formula restated)
# ---------------------------------------------------------------------------


def blatt_trap_frequencies_hz(
    u0: jnp.ndarray, waist_m: jnp.ndarray, wavelength_m: jnp.ndarray, mass_kg: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Longitudinal and radial harmonic trap frequencies, and the recoil
    frequency (Blatt et al. 2009 Eqs. 4-5, transcribed verbatim from the
    typeset PDF):

        nu_z = 2*nu_rec*sqrt(U0/(h*nu_rec))     (Eq. 4)
        nu_r = sqrt(U0/(m*pi^2*w0^2))            (Eq. 5)

    with `U0 = u0*E_R` the peak trap depth (joules) and
    `nu_rec = E_R/h` the recoil frequency. `U0/(h*nu_rec) = U0/E_R = u0`,
    the reduced depth, so `nu_z = 2*nu_rec*sqrt(u0)` directly.

    Parameters
    ----------
    u0 : jax.Array
        Reduced (peak) trap depth, dimensionless.
    waist_m, wavelength_m, mass_kg : jax.Array
        Lattice beam waist, lattice wavelength, atomic mass (site
        geometry/species, same fields as
        :class:`~cliffordclock.integrator.lattice_light_shift_jax.SitePotentialJax`).

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        `(nu_z, nu_r, nu_rec)`, all hertz.
    """
    e_r = recoil_energy_j_jax(wavelength_m, mass_kg)
    nu_rec = e_r / PLANCK_H
    u0_j = u0 * e_r
    nu_z = 2.0 * nu_rec * jnp.sqrt(u0)
    nu_r = jnp.sqrt(u0_j / (mass_kg * jnp.pi**2 * waist_m**2))
    return nu_z, nu_r, nu_rec


def longitudinal_energy_hz(n_z: jnp.ndarray, nu_z: jnp.ndarray, nu_rec: jnp.ndarray) -> jnp.ndarray:
    """The longitudinal-only (radial quantum numbers at `0`) motional
    energy `E_{0,0,n_z}/h` (Blatt et al. 2009 Eq. 3 with `n_x = n_y = 0`;
    algebraically the same term Goti et al. 2025's Eq. 1 calls `E_{00n_z}`
    when their own `n_x = n_y = 0`, up to the additive, `n_z`-independent
    `nu_r` constant, which is dropped here since it cancels in every
    Boltzmann-weight RATIO this module uses):

        E_{0,0,n_z}/h = nu_z*(n_z + 1/2) - (nu_rec/2)*(n_z^2 + n_z + 1/2)

    Parameters
    ----------
    n_z : jax.Array or int
        Longitudinal quantum number.
    nu_z, nu_rec : jax.Array
        From :func:`blatt_trap_frequencies_hz`.

    Returns
    -------
    jax.Array
        Hertz.
    """
    return nu_z * (n_z + 0.5) - 0.5 * nu_rec * (n_z**2 + n_z + 0.5)


def blue_sideband_detuning_hz(
    n_z: jnp.ndarray, n_r: jnp.ndarray, nu_z: jnp.ndarray, nu_r: jnp.ndarray, nu_rec: jnp.ndarray
) -> jnp.ndarray:
    """The blue-sideband (`n_z -> n_z+1`) longitudinal energy gap
    `gamma(n_z)` (Blatt et al. 2009 Eq. 8; Goti et al. 2025 Eq. 2, the
    identical formula, `n_r = n_x + n_y`, transcribed verbatim):

        gamma(n_z) = nu_z - nu_rec*(n_z+1) - nu_rec*(nu_r/nu_z)*(n_r+1)

    Parameters
    ----------
    n_z, n_r : jax.Array or int
        Longitudinal and combined-radial (`n_x + n_y`) quantum numbers.
    nu_z, nu_r, nu_rec : jax.Array
        From :func:`blatt_trap_frequencies_hz`.

    Returns
    -------
    jax.Array
        Hertz.
    """
    return nu_z - nu_rec * (n_z + 1.0) - nu_rec * (nu_r / nu_z) * (n_r + 1.0)


def red_sideband_detuning_hz(
    n_z: jnp.ndarray, n_r: jnp.ndarray, nu_z: jnp.ndarray, nu_r: jnp.ndarray, nu_rec: jnp.ndarray
) -> jnp.ndarray:
    """The red-sideband (`n_z -> n_z-1`) longitudinal energy gap, derived
    directly from Blatt et al. 2009 Eq. 3 the same way Eq. 8 is (the
    energy difference `E_{n_z-1} - E_{n_z}`, evaluated at `n_x=n_y=0`,
    which is exactly the negative of :func:`blue_sideband_detuning_hz`
    evaluated one band lower, `-gamma(n_z-1)`):

        delta_red(n_z) = -[nu_z - nu_rec*n_z - nu_rec*(nu_r/nu_z)*(n_r+1)]

    Not itself a separately numbered equation in either paper (both
    papers state the blue sideband explicitly and note the red sideband
    follows "by symmetry," Blatt et al. 2009's own text just above their
    Eq. 12); this is that symmetry made algebraically explicit.

    Parameters, returns: same contract as
    :func:`blue_sideband_detuning_hz`.
    """
    return -(nu_z - nu_rec * n_z - nu_rec * (nu_r / nu_z) * (n_r + 1.0))


def goti_e00_hz(
    n_z: jnp.ndarray, nu_z: jnp.ndarray, nu_r: jnp.ndarray, nu_rec: jnp.ndarray
) -> jnp.ndarray:
    """`E_{0,0,n_z}/h` including the radial-coupling term at `n_x=n_y=0`
    (Goti et al. 2025 Eq. 1 evaluated at `n_x=n_y=0`, transcribed
    verbatim, the SAME formula :func:`longitudinal_energy_hz` implements
    but keeping the `n_z`-dependent coupling term Eq. 1 carries even at
    zero radial excitation):

        E_{0,0,n_z}/h = nu_z*(n_z+1/2) + nu_r
                        - (nu_rec/2)*(n_z^2+n_z+1/2)
                        - nu_rec*(nu_r/nu_z)*(n_z+1/2)

    This is the quantity Goti et al. 2025's own Eq. 4 (the harmonic-path
    sideband population weight, :func:`harmonic_sideband_unnormalized_weight`)
    uses in its `exp[-E_{00n_z}/(k_B T_z)]` factor, distinct from
    :func:`longitudinal_energy_hz` (which Blatt et al. 2009's own
    Appendix A Eq. A5 uses for the OUTER band-average, with the coupling
    term already absorbed into the per-band sideband shape via
    :func:`blue_sideband_detuning_hz`/:func:`red_sideband_detuning_hz`
    instead). The additive `nu_r` constant is `n_z`-independent and
    cancels in every population RATIO this module forms; kept here only
    for a verbatim transcription of Eq. 1.

    Parameters, returns: same contract as :func:`longitudinal_energy_hz`.
    """
    return (
        nu_z * (n_z + 0.5)
        + nu_r
        - 0.5 * nu_rec * (n_z**2 + n_z + 0.5)
        - nu_rec * (nu_r / nu_z) * (n_z + 0.5)
    )


# ---------------------------------------------------------------------------
# Harmonic path, carrier (Blatt et al. 2009 Eqs. 13-20)
# ---------------------------------------------------------------------------


def laguerre_values(x: jnp.ndarray, n_max: int) -> jnp.ndarray:
    """`[L_0(x), L_1(x), ..., L_{n_max}(x)]`, the (physicists') Laguerre
    polynomials Blatt et al. 2009 Eq. 13's carrier Rabi-frequency formula
    needs, via the standard three-term recurrence (not separately numbered
    in either paper; a standard special-function identity),
    `L_0(x)=1`, `L_1(x)=1-x`,
    `(n+1)*L_{n+1}(x) = (2n+1-x)*L_n(x) - n*L_{n-1}(x)`, evaluated by a
    fixed-length `jax.lax.scan` so the result stays differentiable with
    respect to `x` end to end (no special-function library call: `jax`
    ships no built-in generalized Laguerre polynomial).

    Parameters
    ----------
    x : jax.Array, scalar
    n_max : int
        Static. Highest order returned.

    Returns
    -------
    jax.Array, shape (n_max+1,)
    """
    l0 = jnp.ones((), dtype=jnp.float64)
    l1 = 1.0 - x

    def body(
        carry: tuple[jnp.ndarray, jnp.ndarray], n: jnp.ndarray
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        l_nm1, l_n = carry
        n_f = n.astype(jnp.float64)
        l_np1 = ((2.0 * n_f + 1.0 - x) * l_n - n_f * l_nm1) / (n_f + 1.0)
        return (l_n, l_np1), l_np1

    if n_max == 0:
        return l0[None]
    if n_max == 1:
        return jnp.stack([l0, l1])
    ns = jnp.arange(1, n_max, dtype=jnp.int64)
    _, rest = jax.lax.scan(body, (l0, l1), ns)
    return jnp.concatenate([l0[None], l1[None], rest])


def carrier_rabi_frequency_hz(
    n_x: jnp.ndarray,
    n_z: jnp.ndarray,
    rabi0_hz: jnp.ndarray,
    nu_z: jnp.ndarray,
    nu_r: jnp.ndarray,
    probe_wavelength_m: jnp.ndarray,
    mass_kg: jnp.ndarray,
    misalignment_rad: jnp.ndarray,
    n_x_max: int,
    n_z_max: int,
) -> jnp.ndarray:
    """The carrier Rabi frequency `Omega_{n_x,n_z}` for a harmonically
    trapped particle in motional state `|n_x, n_z>`, probed with a
    (possibly misaligned) traveling-wave beam (Blatt et al. 2009 Eqs.
    14-16, transcribed verbatim):

        Omega_{n_x,n_z} ~= Omega_0 * e^(-eta_z^2/2) * e^(-eta_x^2/2)
                            * L_{n_x}(eta_x^2) * L_{n_z}(eta_z^2)     (Eq. 14)
        eta_z = (1/lambda_p) * sqrt(h/(2*m*nu_z))                     (Eq. 15)
        eta_x = (Delta_theta/lambda_p) * sqrt(h/(2*m*nu_r))           (Eq. 16)

    with `Delta_theta` the probe misalignment angle from the lattice
    axis. At `Delta_theta = 0`, `eta_x = 0` and `L_{n_x}(0) = 1`
    identically for every `n_x` (the generalized Laguerre polynomial's
    own value at its argument's origin), so `Omega_{n_x,n_z}` reduces to
    Eq. 13's pure single-axis form, independent of `n_x`: a perfectly
    aligned probe cannot distinguish transverse motional states on the
    carrier, matching Blatt et al. 2009's own stated physical picture
    (their Section V, "the broad transverse profile can be neglected").

    Parameters
    ----------
    n_x, n_z : jax.Array or int
        Motional quantum numbers.
    rabi0_hz : jax.Array
        `Omega_0`, the free-space Rabi frequency, hertz.
    nu_z, nu_r : jax.Array
        From :func:`blatt_trap_frequencies_hz`.
    probe_wavelength_m : jax.Array
        `lambda_p`, the clock-probe wavelength, meters.
    mass_kg : jax.Array
    misalignment_rad : jax.Array
        `Delta_theta`, radians.
    n_x_max, n_z_max : int
        Static. Highest Laguerre order each argument needs (must be
        `>= the largest n_x/n_z this call is evaluated at`; the caller is
        responsible for a consistent cutoff, matching
        :data:`N_X_MAX_CARRIER`/:data:`N_Z_MAX_HARMONIC`).

    Returns
    -------
    jax.Array
        `Omega_{n_x,n_z}`, hertz.
    """
    eta_z2 = (1.0 / probe_wavelength_m) ** 2 * (PLANCK_H / (2.0 * mass_kg * nu_z))
    eta_x2 = (misalignment_rad / probe_wavelength_m) ** 2 * (PLANCK_H / (2.0 * mass_kg * nu_r))
    l_x = laguerre_values(eta_x2, n_x_max)[n_x] if n_x_max > 0 else jnp.ones(())
    l_z = laguerre_values(eta_z2, n_z_max)[n_z]
    return rabi0_hz * jnp.exp(-0.5 * eta_z2) * jnp.exp(-0.5 * eta_x2) * l_x * l_z


def harmonic_carrier_excitation_probability(
    delta_hz: jnp.ndarray,
    t_s: jnp.ndarray,
    u0: jnp.ndarray,
    waist_m: jnp.ndarray,
    wavelength_m: jnp.ndarray,
    mass_kg: jnp.ndarray,
    probe_wavelength_m: jnp.ndarray,
    rabi0_hz: jnp.ndarray,
    temperature_z_k: jnp.ndarray,
    temperature_r_k: jnp.ndarray,
    *,
    misalignment_rad: jnp.ndarray | float = 0.0,
    n_x_max: int = N_X_MAX_CARRIER,
    n_z_max: int = N_Z_MAX_HARMONIC,
) -> jnp.ndarray:
    """The thermally-averaged carrier excitation probability `P_e(delta,
    t)` (Blatt et al. 2009 Eqs. 17-20, transcribed verbatim):

        p_e(n,delta,t) = [Omega_n^2 / (Omega_n^2+delta^2)]
                          * sin^2[pi*t*sqrt(Omega_n^2+delta^2)]          (Eq. 17)
        P_e(delta,t) = sum_{n_x,n_z} q_{n_x}(T_r)*q_{n_z}(T_z)*p_e(n,delta,t)   (Eq. 18)
        q_{n_x} = (1-z_r)*z_r^{n_x},  z_r = exp[-h*nu_r/(k_B*T_r)]        (Eq. 19)
        q_{n_z} = (1-z_z)*z_z^{n_z},  z_z = exp[-h*nu_z/(k_B*T_z)]        (Eq. 20)

    Summed over `n_x in [0, n_x_max]`, `n_z in [0, n_z_max]` (static
    truncation of the geometric Boltzmann series; the dropped tail's
    weight is `z_r^{n_x_max+1}`/`z_z^{n_z_max+1}`, verified negligible at
    this module's own offline-checked operating domain, see the module
    docstring). `n_y` does not appear separately: Blatt et al. 2009's own
    Eq. 13/14 carrier Rabi frequency has no `n_y`-dependence at
    `Delta_theta` confined to the `x`-axis (their own stated
    convention), so the `n_y` sum is already absorbed into normalization
    and contributes no additional structure.

    Parameters
    ----------
    delta_hz : jax.Array
        Probe detuning from the carrier, hertz. Any shape (broadcasts).
    t_s : jax.Array
        Probe pulse time, seconds.
    u0, waist_m, wavelength_m, mass_kg : jax.Array
        Site/species parameters (:func:`blatt_trap_frequencies_hz`).
    probe_wavelength_m : jax.Array
        Clock-probe wavelength, meters.
    rabi0_hz : jax.Array
        Free-space Rabi frequency, hertz.
    temperature_z_k, temperature_r_k : jax.Array
        `T_z`, `T_r`, kelvin.
    misalignment_rad : jax.Array or float, default 0.0
        `Delta_theta`, radians.
    n_x_max, n_z_max : int, default :data:`N_X_MAX_CARRIER`/:data:`N_Z_MAX_HARMONIC`
        Static truncation.

    Returns
    -------
    jax.Array
        `P_e(delta, t)`, same shape as `delta_hz`, bounded `[0, 1]`.
    """
    nu_z, nu_r, _ = blatt_trap_frequencies_hz(u0, waist_m, wavelength_m, mass_kg)
    z_r = jnp.exp(-PLANCK_H * nu_r / (BOLTZMANN_K * temperature_r_k))
    z_z = jnp.exp(-PLANCK_H * nu_z / (BOLTZMANN_K * temperature_z_k))

    n_x_grid = jnp.arange(n_x_max + 1, dtype=jnp.int64)
    n_z_grid = jnp.arange(n_z_max + 1, dtype=jnp.int64)
    q_x = (1.0 - z_r) * z_r**n_x_grid
    q_z = (1.0 - z_z) * z_z**n_z_grid

    delta = jnp.asarray(delta_hz)
    delta_ndim = delta.ndim

    def per_state(n_x: jnp.ndarray, n_z: jnp.ndarray) -> jnp.ndarray:
        omega = carrier_rabi_frequency_hz(
            n_x,
            n_z,
            rabi0_hz,
            nu_z,
            nu_r,
            probe_wavelength_m,
            mass_kg,
            jnp.asarray(misalignment_rad),
            n_x_max,
            n_z_max,
        )
        omega2 = omega**2
        denom = omega2 + delta**2
        p_e = (omega2 / denom) * jnp.sin(jnp.pi * t_s * jnp.sqrt(denom)) ** 2
        return p_e

    # p_grid: shape (n_x_max+1, n_z_max+1, *delta.shape).
    p_grid = jax.vmap(lambda n_x: jax.vmap(lambda n_z: per_state(n_x, n_z))(n_z_grid))(n_x_grid)
    q_z_b = q_z.reshape((1, n_z_max + 1) + (1,) * delta_ndim)
    q_x_b = q_x.reshape((n_x_max + 1, 1) + (1,) * delta_ndim)
    weighted = p_grid * q_z_b * q_x_b
    return jnp.sum(weighted, axis=(0, 1))


# ---------------------------------------------------------------------------
# Harmonic path, sideband (Blatt et al. 2009 Eq. 8, App. A1-A2; Goti et al.
# 2025 Eqs. 2, 4)
# ---------------------------------------------------------------------------


def harmonic_sideband_shape(
    delta_hz: jnp.ndarray,
    sign: int,
    u0: jnp.ndarray,
    waist_m: jnp.ndarray,
    wavelength_m: jnp.ndarray,
    mass_kg: jnp.ndarray,
    temperature_z_k: jnp.ndarray,
    temperature_r_k: jnp.ndarray,
    linewidth_hz: jnp.ndarray,
    *,
    n_z_max: int = N_Z_MAX_HARMONIC,
    n_r_max: int = N_R_MAX_HARMONIC,
) -> jnp.ndarray:
    """The harmonic-path (Blatt-faithful) sideband shape, a
    population-normalized sum of power-broadened Lorentzians (Blatt et
    al. 2009 Appendix A Eqs. A1-A2, generalized here to the FULL
    `(n_z, n_r)` sum. Eqs. A3-A5 go on to reduce this to the single
    dominant `n_r` term at the shallow sideband edge; this function
    keeps the full sum, since a differentiable forward model useful for
    fitting an entire sideband needs the full lineshape. Goti et al.
    2025 Eq. 4 supplies the population weight, the same reference this
    module cites for :func:`goti_e00_hz`):

        shape(delta) = sum_{n_z,n_r} w(n_z,n_r) / (1 + [(delta-detuning(n_z,n_r))/gamma]^2)

    with `detuning(n_z,n_r)` from :func:`blue_sideband_detuning_hz`/
    :func:`red_sideband_detuning_hz`, `gamma` the power-broadened base
    linewidth (`linewidth_hz`, Blatt et al. 2009's own stated
    approximation, "given by the carrier Rabi frequency"), and

        w(n_z,n_r) proportional-to (n_r+1) * exp[-h*nu_r*(n_r+1)/(k_B*T_r)]
                                   * exp[-goti_e00_hz(n_z)/(k_B*T_z)]      (Goti Eq. 4)

    normalized here so `sum_{n_z,n_r} w(n_z,n_r) = 1` (a genuine
    population distribution over the truncated `(n_z, n_r)` grid), which
    makes `shape` a convex combination of unit-height Lorentzians and
    therefore bounded `[0, 1]` (module docstring's "amplitude
    convention" section). This is the general Eq. A1/A2 form, the full
    quantum-number sum. Eqs. A3-A5 go on to reduce that sum further, to
    the shallow-sideband-edge slope alone; this function keeps the full
    sum, since a differentiable forward model useful for fitting an
    entire sideband needs the full lineshape.

    **Numerical stability**, the same shifted-exponential reformulation
    `lattice_light_shift`'s own `axial_thermal_factors` uses: `w`'s two
    exponentials are evaluated relative to their own maxima
    (`n_r=0`/`n_z=0`) before normalizing, so no intermediate value
    overflows float64 even for deep, cold traps where the bare
    Boltzmann factors would.

    Parameters
    ----------
    delta_hz : jax.Array
        Detuning from the carrier, hertz. Any shape.
    sign : int
        `+1` for the blue sideband, `-1` for the red. Static (selects
        which detuning formula is used; not itself a `jax.Array`).
    u0, waist_m, wavelength_m, mass_kg : jax.Array
    temperature_z_k, temperature_r_k : jax.Array
        `T_z`, `T_r`, kelvin.
    linewidth_hz : jax.Array
        `gamma`, the power-broadened base linewidth, hertz.
    n_z_max, n_r_max : int, default :data:`N_Z_MAX_HARMONIC`/:data:`N_R_MAX_HARMONIC`
        Static truncation.

    Returns
    -------
    jax.Array
        Bounded `[0, 1]`, same shape as `delta_hz`.

    Raises
    ------
    ValueError
        `sign not in (1, -1)`.
    """
    if sign not in (1, -1):
        raise ValueError(f"sign must be 1 (blue) or -1 (red), got {sign}")
    nu_z, nu_r, nu_rec = blatt_trap_frequencies_hz(u0, waist_m, wavelength_m, mass_kg)

    n_z_grid = jnp.arange(n_z_max + 1, dtype=jnp.float64)
    n_r_grid = jnp.arange(n_r_max + 1, dtype=jnp.float64)

    e00 = goti_e00_hz(n_z_grid, nu_z, nu_r, nu_rec)
    log_w_z = -(e00 - e00[0]) / temperature_z_k * (PLANCK_H / BOLTZMANN_K)
    # log_w_z[0] == 0 by construction; monotonically <= 0 for a well-formed
    # spectrum where deeper (n_z=0) is favored.

    log_w_r = -nu_r * (n_r_grid + 1.0) / temperature_r_k * (PLANCK_H / BOLTZMANN_K)
    log_w_r = log_w_r - log_w_r[0]
    degeneracy = n_r_grid + 1.0

    w_z = jnp.exp(log_w_z)
    if sign == -1:
        # Blatt et al. 2009, text immediately following their Eq. 12: "There
        # is no contribution from the longitudinal ground state to the red
        # sideband" (an atom in n_z=0 has no n_z=-1 state to transition to).
        # Excluded here by zeroing that band's own weight before summing:
        # this keeps `n_z_grid` shared with the blue-sideband call, one
        # simple shared code path for both signs.
        w_z = w_z.at[0].set(0.0)
    w_r = degeneracy * jnp.exp(log_w_r)
    w = w_z[:, None] * w_r[None, :]
    total = jnp.sum(w)
    # Guards the degenerate case (e.g. n_z_max=0 for the red sideband,
    # where Blatt's own ground-state exclusion above zeros every term):
    # returns a well-defined all-zero shape there, avoiding the 0/0 NaN
    # a bare division would otherwise produce.
    w = jnp.where(total > 0.0, w / jnp.where(total > 0.0, total, 1.0), 0.0)

    detuning_fn = blue_sideband_detuning_hz if sign == 1 else red_sideband_detuning_hz
    detuning = detuning_fn(n_z_grid[:, None], n_r_grid[None, :], nu_z, nu_r, nu_rec)

    delta = jnp.asarray(delta_hz)
    diff = delta[..., None, None] - detuning[None, ...]
    lorentzian = 1.0 / (1.0 + (diff / linewidth_hz) ** 2)
    return jnp.sum(lorentzian * w[None, ...], axis=(-2, -1))


def harmonic_full_spectrum(
    delta_hz: jnp.ndarray,
    t_s: jnp.ndarray,
    u0: jnp.ndarray,
    waist_m: jnp.ndarray,
    wavelength_m: jnp.ndarray,
    mass_kg: jnp.ndarray,
    probe_wavelength_m: jnp.ndarray,
    rabi0_hz: jnp.ndarray,
    temperature_z_k: jnp.ndarray,
    temperature_r_k: jnp.ndarray,
    blue_amplitude: jnp.ndarray,
    red_amplitude: jnp.ndarray,
    sideband_linewidth_hz: jnp.ndarray,
    *,
    misalignment_rad: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """The full harmonic-path (Blatt-faithful) excitation spectrum: the
    carrier (:func:`harmonic_carrier_excitation_probability`, Eqs.
    17-20) plus the blue and red sideband shapes
    (:func:`harmonic_sideband_shape`, App. A1-A2/Goti Eq. 4), each
    scaled by its own amplitude (module docstring's "amplitude
    convention" section, matching Blatt et al. 2009's own Fig. 2
    independently-fitted-feature practice):

        P(delta) = P_e^carrier(delta,t) + blue_amplitude*shape_blue(delta)
                                          + red_amplitude*shape_red(delta)

    Parameters
    ----------
    delta_hz, t_s : jax.Array
    u0, waist_m, wavelength_m, mass_kg, probe_wavelength_m, rabi0_hz : jax.Array
    temperature_z_k, temperature_r_k : jax.Array
    blue_amplitude, red_amplitude : jax.Array
        Peak-scale amplitudes for each sideband feature, dimensionless.
    sideband_linewidth_hz : jax.Array
        `gamma`, shared by both sidebands.
    misalignment_rad : jax.Array or float, default 0.0

    Returns
    -------
    jax.Array
        Same shape as `delta_hz`.
    """
    carrier = harmonic_carrier_excitation_probability(
        delta_hz,
        t_s,
        u0,
        waist_m,
        wavelength_m,
        mass_kg,
        probe_wavelength_m,
        rabi0_hz,
        temperature_z_k,
        temperature_r_k,
        misalignment_rad=misalignment_rad,
    )
    blue = harmonic_sideband_shape(
        delta_hz,
        1,
        u0,
        waist_m,
        wavelength_m,
        mass_kg,
        temperature_z_k,
        temperature_r_k,
        sideband_linewidth_hz,
    )
    red = harmonic_sideband_shape(
        delta_hz,
        -1,
        u0,
        waist_m,
        wavelength_m,
        mass_kg,
        temperature_z_k,
        temperature_r_k,
        sideband_linewidth_hz,
    )
    return carrier + blue_amplitude * blue + red_amplitude * red


# ---------------------------------------------------------------------------
# BO+WKB path (Goti et al. 2025 Eqs. 5-9, built on the G18/G19-gated BO+WKB
# density-of-states core)
# ---------------------------------------------------------------------------


class BandEnergyTable(NamedTuple):
    """One axial band's `U_{n_z}(rho)/E_R` sampled on a fixed radial grid
    (:func:`build_band_energy_table`), the table
    :func:`condon_point_m`/:func:`condon_detuning_hz` interpolate against.

    Attributes
    ----------
    rho_grid_m : jax.Array, shape (RHO_TABLE_N,)
        Radial sample points, `0` to `rho_max_m`.
    energy_er : jax.Array, shape (RHO_TABLE_N,)
        `U_{n_z}(rho)/E_R` at each `rho_grid_m` point, monotonically
        non-decreasing from the band bottom (`rho=0`) to `0`
        (:func:`~cliffordclock.integrator.lattice_light_shift_jax._axial_band_energy_er_at_rho`'s
        own clamped convention).
    rho_max_m : jax.Array, scalar
        The table's own outer bracket (`RHO_BRACKET_WAIST_MULTIPLE * waist_m`,
        NOT the true turning radius at `E=0`: the table already reaches
        `energy_er=0` well before this bound for any physically bound
        band, since the clamp holds the energy at exactly `0` beyond the
        true crossing).
    """

    rho_grid_m: jnp.ndarray
    energy_er: jnp.ndarray
    rho_max_m: jnp.ndarray


def build_band_energy_table(
    site: SitePotentialJax,
    n_z: int,
    *,
    axial_grid_n: int = AXIAL_GRID_N_SPECTRUM,
    rho_table_n: int = RHO_TABLE_N,
    rho_bracket_waist_multiple: float = RHO_BRACKET_WAIST_MULTIPLE,
) -> BandEnergyTable:
    """Build one :class:`BandEnergyTable` for axial band `n_z`: a single
    `jax.vmap` over `jax.numpy.linalg.eigh` calls, one per radial grid
    point, batched into one XLA op (module docstring's "why the BO+WKB
    sideband path needs its own numerical route" section derives why
    this replaces `lattice_light_shift_jax.turning_radius_m_jax`'s own
    per-energy bisection for THIS module's use case).

    Parameters
    ----------
    site : SitePotentialJax
    n_z : int
        Static.
    axial_grid_n : int, default :data:`AXIAL_GRID_N_SPECTRUM`
        Static axial finite-difference resolution.
    rho_table_n : int, default :data:`RHO_TABLE_N`
        Static radial grid point count.
    rho_bracket_waist_multiple : float, default :data:`RHO_BRACKET_WAIST_MULTIPLE`

    Returns
    -------
    BandEnergyTable
    """
    x_grid, dx = _axial_grid(axial_grid_n)
    rho_max = rho_bracket_waist_multiple * site.waist_m
    rho_grid = jnp.linspace(0.0, rho_max, rho_table_n, dtype=jnp.float64)

    def sample(rho: jnp.ndarray) -> jnp.ndarray:
        return _axial_band_energy_er_at_rho(site.depth_er, site.kappa_per_m, n_z, rho, x_grid, dx)

    energy_er = jax.vmap(sample)(rho_grid)
    return BandEnergyTable(rho_grid_m=rho_grid, energy_er=energy_er, rho_max_m=rho_max)


def condon_point_m(table: BandEnergyTable, energy_er: jnp.ndarray) -> jnp.ndarray:
    """The classical turning radius `R_{n_z}(E)` (Beloy et al. 2020's
    notation, `Unz(Rnz(E))=E`; the same object Goti et al. 2025's Eq. 5
    calls the Condon point `r_c`), found by linear interpolation against
    `table` (:func:`build_band_energy_table`'s output for band `n_z`).
    The module docstring's "why the BO+WKB sideband path needs its own
    numerical route" section derives why this interpolation replaces
    `lattice_light_shift_jax.turning_radius_m_jax`'s own bisection for
    this module's use case. `table.energy_er` is monotonically
    non-decreasing in `rho`, so `jax.numpy.interp` (which requires an
    increasing `xp`) applies directly with the table's own arrays as
    `(xp, fp) = (energy_er, rho_grid_m)`.

    Parameters
    ----------
    table : BandEnergyTable
    energy_er : jax.Array
        `E`, `E_R` units. Should lie within `[table.energy_er[0], 0]`;
        outside that range `jax.numpy.interp` clips to the table's own
        endpoint (no error raised, matching this module's static-shape,
        validate-outside-the-jitted-core discipline).

    Returns
    -------
    jax.Array
        `R_{n_z}(E)`, meters.
    """
    return jnp.interp(energy_er, table.energy_er, table.rho_grid_m)


def condon_detuning_hz(
    table_nz: BandEnergyTable,
    table_target: BandEnergyTable,
    energy_er: jnp.ndarray,
    recoil_energy_j_value: jnp.ndarray,
) -> jnp.ndarray:
    """The Franck-Condon transition detuning `delta_nu` at energy `E`
    within band `n_z` (Goti et al. 2025 Eq. 5, transcribed verbatim):

        delta_nu = [U_{n_z'}(r_c) - U_{n_z}(r_c)] / h

    with `r_c = R_{n_z}(E)` (:func:`condon_point_m`) and `n_z'` the
    target band (`n_z+1` for the blue sideband, `n_z-1` for the red;
    `table_target` is that band's own :class:`BandEnergyTable`). Since
    `U_{n_z}(r_c) = E` by `r_c`'s own definition, this reduces to
    `[U_{n_z'}(r_c) - E]/h`, evaluated here by interpolating
    `table_target` at the SAME `r_c` (:func:`condon_point_m` again, this
    time reading the target table's `rho -> energy` direction directly
    via `jax.numpy.interp` on `(table_target.rho_grid_m,
    table_target.energy_er)`).

    Parameters
    ----------
    table_nz, table_target : BandEnergyTable
        The starting band's own table and the target band's table (same
        `site`, different `n_z`).
    energy_er : jax.Array
        `E`, `E_R` units, within band `n_z`'s range.
    recoil_energy_j_value : jax.Array
        `E_R`, joules (converts `E` to joules for the `/h` division).

    Returns
    -------
    jax.Array
        Hertz.
    """
    r_c = condon_point_m(table_nz, energy_er)
    target_energy_er = jnp.interp(r_c, table_target.rho_grid_m, table_target.energy_er)
    energy_j = energy_er * recoil_energy_j_value
    target_energy_j = target_energy_er * recoil_energy_j_value
    return (target_energy_j - energy_j) / PLANCK_H


def bowkb_density_of_states_from_table(
    table: BandEnergyTable, energy_er: jnp.ndarray, mass_kg: jnp.ndarray
) -> jnp.ndarray:
    """`G_{n_z}(E) = (m/(2*hbar^2)) * [R_{n_z}(E)]^2` (Beloy et al. 2020
    Eq. 11; Goti et al. 2025 Eq. 8's own `G_{n_z}(E) = (m/(2*hbar^2))*[R_{n_z}(E)]^2`,
    the algebraically identical form), evaluated from `table` via
    :func:`condon_point_m`, the module's own table-interpolated turning
    radius (module docstring). `lattice_light_shift_jax.bo_wkb_density_of_states_jax`
    computes the same physical quantity through its own bisection-based
    turning radius.

    Parameters
    ----------
    table : BandEnergyTable
    energy_er : jax.Array
        `E_R` units.
    mass_kg : jax.Array

    Returns
    -------
    jax.Array
        States per joule, `>= 0`.
    """
    r_nz = condon_point_m(table, energy_er)
    return (mass_kg / (2.0 * HBAR**2)) * r_nz**2


def bowkb_sideband_shape(
    delta_hz: jnp.ndarray,
    sign: int,
    u0: jnp.ndarray,
    waist_m: jnp.ndarray,
    wavelength_m: jnp.ndarray,
    mass_kg: jnp.ndarray,
    temperature_z_k: jnp.ndarray,
    temperature_r_k: jnp.ndarray,
    linewidth_hz: jnp.ndarray,
    *,
    n_z_max: int = N_Z_MAX_BOWKB,
    n_e_quad: int = N_E_QUAD,
    axial_grid_n: int = AXIAL_GRID_N_SPECTRUM,
    rho_table_n: int = RHO_TABLE_N,
) -> jnp.ndarray:
    """The BO+WKB-path sideband shape (Goti et al. 2025 Eqs. 6-9,
    transcribed verbatim): a sum over axial bands `n_z` of an ENERGY
    integral (WKB-continuous in the radial direction, replacing the
    harmonic path's discrete `n_r` sum) of the density of states times
    the two-temperature Boltzmann population, Lorentzian-weighted at
    each energy's own Franck-Condon detuning:

        sigma(delta) proportional-to sum_{n_z} integral_E
            G_{n_z}(E) * p_{n_z}(E) / (1 + [(delta-delta_nu(E))/gamma]^2) dE     (Eq. 8)

        p_{n_z}(E) proportional-to exp[-(E-U_{n_z}(0))/(k_B*T_r)]
                                   * exp[-U_{n_z}(0)/(k_B*T_z)]                   (Eq. 9)

    with `G_{n_z}(E)` from :func:`bowkb_density_of_states_from_table`
    (Eq. 8's own density-of-states factor) and `delta_nu(E)` from
    :func:`condon_detuning_hz` (Eq. 5). Normalized (like
    :func:`harmonic_sideband_shape`) so the total `(n_z, E)` weight sums
    to `1`, making this function's output a convex combination of
    unit-height Lorentzians, bounded `[0, 1]` (module docstring's
    "amplitude convention").

    **Integration domain, and the refinement over Eq. 8's own stated
    approximation.** Goti et al. 2025's own Eq. 8 integrates `E` from
    `E_min = U_{n_z}(0)` to `E_max = -h*delta` (blue) or `E_max = 0`
    (red), an approximation their own text calls "valid for a deep
    vertical lattice where we can neglect tunnelling and the
    Wannier-Stark ladder," because the TRUE physical requirement is that
    the target band's own Condon-point energy stay bound
    (`U_{n_z'}(r_c) <= 0`), which `E_max = -h*delta` only approximates.
    This function evaluates that TRUE condition directly (it already has
    `U_{n_z'}(r_c)` in hand from :func:`condon_detuning_hz`'s own
    computation, at no extra table-building cost): the fixed `n_e_quad`-point
    quadrature spans the FULL band range `[U_{n_z}(0), 0)` (a small
    margin below `0` avoids the degenerate top-of-band point, where the
    table's own clamped value repeats and the local density of states
    formally diverges), and any energy whose target-band Condon-point
    energy is unbound (`> 0`) is masked to zero weight via
    `jax.numpy.where`. This directly evaluates Eq. 8's own stated
    physical boundary condition. Eq. 8 itself approximates that
    condition through a `-h*delta` proxy; the direct condition used
    here is `delta`-independent because this function is evaluated at a
    fixed quadrature grid shared across all query `delta_hz` values,
    masked once per `(n_z, E)` pair and reused for the whole spectrum,
    keeping the grid static for `jax.jit`.

    Parameters
    ----------
    delta_hz : jax.Array
    sign : int
        `+1` blue, `-1` red. Static.
    u0, waist_m, wavelength_m, mass_kg : jax.Array
    temperature_z_k, temperature_r_k : jax.Array
    linewidth_hz : jax.Array
    n_z_max : int, default :data:`N_Z_MAX_BOWKB`
    n_e_quad : int, default :data:`N_E_QUAD`
    axial_grid_n : int, default :data:`AXIAL_GRID_N_SPECTRUM`
    rho_table_n : int, default :data:`RHO_TABLE_N`

    Returns
    -------
    jax.Array
        Bounded `[0, 1]`, same shape as `delta_hz`.

    Raises
    ------
    ValueError
        `sign not in (1, -1)`.
    """
    if sign not in (1, -1):
        raise ValueError(f"sign must be 1 (blue) or -1 (red), got {sign}")
    site = make_site_potential_jax(u0, waist_m, wavelength_m, mass_kg)

    # Bands 0..n_z_max+1 cover every table this function can need: blue
    # sidebands read target n_z+1 (up to n_z_max+1), red sidebands read
    # target n_z-1 (down to 0).
    tables = {
        nz: build_band_energy_table(site, nz, axial_grid_n=axial_grid_n, rho_table_n=rho_table_n)
        for nz in range(n_z_max + 2)
    }

    # Blatt et al. 2009, text following their Eq. 12: "There is no
    # contribution from the longitudinal ground state to the red
    # sideband" (n_z=0 has no n_z=-1 target band). Excluded here by
    # starting the starting-band range at 1 for the red sideband, the
    # same physical exclusion :func:`harmonic_sideband_shape` applies.
    nz_values = list(range(n_z_max + 1)) if sign == 1 else list(range(1, n_z_max + 1))

    delta = jnp.asarray(delta_hz)
    all_lorentzians = {}
    all_weights = {}
    for nz in nz_values:
        target_nz = nz + sign
        table_nz = tables[nz]
        table_target = tables[target_nz]
        e_bottom = table_nz.energy_er[0]
        # A 2%-of-band-depth margin below E=0. An earlier version of this
        # function used a tiny 1e-6 margin; the cross-validation benchmark
        # (`benchmarks/run_sideband_spectrum.py`'s
        # tier 2, `run_condon_detuning_reproduction_case`) found this
        # module's table-interpolated turning radius loses accuracy fast
        # within about 5 E_R of the band top (the classical turning
        # radius grows without bound as the local depth vanishes, and a
        # finite, linearly-spaced radial table cannot represent that
        # divergence); the independent-oracle comparison there measured
        # this module's own predicted detuning collapsing toward 0 well
        # before large-lattice-model's own root-find does, at the
        # closest-to-edge point either side was checked at. This margin
        # keeps this function's own quadrature grid out of that
        # known-unreliable region.
        eps = jnp.maximum(0.02 * jnp.abs(e_bottom), 1e-3)
        e_grid = jnp.linspace(e_bottom, -eps, n_e_quad, dtype=jnp.float64)

        g_nz = bowkb_density_of_states_from_table(table_nz, e_grid, mass_kg)
        e_bottom_j = e_bottom * site.recoil_energy_j_value
        e_grid_j = e_grid * site.recoil_energy_j_value
        # Eq. 9's WITHIN-band radial factor, exp[-(E-Unz(0))/(kB*Tr)],
        # bounded in (0, 1] since e_grid_j >= e_bottom_j by construction:
        # no overflow risk. Eq. 9's remaining, band-level factor,
        # exp[-Unz(0)/(kB*Tz)], is NOT applied here (it would overflow for
        # a deep band evaluated in isolation); it is instead applied
        # once per band, relative to the n_z=0 band's own bottom, via
        # `band_scale` below, algebraically the same product with the
        # same numerical-stability reformulation
        # `axial_thermal_factors`/`harmonic_sideband_shape` both use.
        p_nz = jnp.exp(-(e_grid_j - e_bottom_j) / (BOLTZMANN_K * temperature_r_k))

        delta_nu = condon_detuning_hz(table_nz, table_target, e_grid, site.recoil_energy_j_value)
        target_energy_er = jnp.interp(
            condon_point_m(table_nz, e_grid), table_target.rho_grid_m, table_target.energy_er
        )
        bound_mask = target_energy_er <= 0.0

        weight = jnp.where(bound_mask, g_nz * p_nz, 0.0)
        diff = delta[..., None] - delta_nu[None, :]
        lorentzian = 1.0 / (1.0 + (diff / linewidth_hz) ** 2)
        all_lorentzians[nz] = lorentzian
        all_weights[nz] = weight

    # Stabilize the cross-band weighting the same way
    # `harmonic_sideband_shape` does: work in log-space relative to the
    # n_z=0 band's own bottom (band 0 always exists and anchors the scale
    # even when the red sideband's own sum excludes it as a STARTING
    # band above) before combining bands, then normalize once over the
    # full (n_z, E) grid.
    band0_bottom_j = tables[0].energy_er[0] * site.recoil_energy_j_value

    total = jnp.zeros_like(delta)
    total_weight = jnp.zeros(())
    for nz in nz_values:
        band_bottom_j = tables[nz].energy_er[0] * site.recoil_energy_j_value
        band_scale = jnp.exp(-(band_bottom_j - band0_bottom_j) / (BOLTZMANN_K * temperature_z_k))
        w = all_weights[nz] * band_scale
        total = total + jnp.sum(all_lorentzians[nz] * w[None, :], axis=-1)
        total_weight = total_weight + jnp.sum(w)
    return total / total_weight


def bowkb_full_spectrum(
    delta_hz: jnp.ndarray,
    t_s: jnp.ndarray,
    u0: jnp.ndarray,
    waist_m: jnp.ndarray,
    wavelength_m: jnp.ndarray,
    mass_kg: jnp.ndarray,
    probe_wavelength_m: jnp.ndarray,
    rabi0_hz: jnp.ndarray,
    temperature_z_k: jnp.ndarray,
    temperature_r_k: jnp.ndarray,
    blue_amplitude: jnp.ndarray,
    red_amplitude: jnp.ndarray,
    sideband_linewidth_hz: jnp.ndarray,
    *,
    misalignment_rad: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """The full BO+WKB-path excitation spectrum: the SAME carrier
    component as :func:`harmonic_full_spectrum`
    (:func:`harmonic_carrier_excitation_probability`; module docstring's
    "no carrier formula is given by either the BO+WKB paper... or Goti et
    al. 2025" note) plus the BO+WKB blue/red sideband shapes
    (:func:`bowkb_sideband_shape`, Eqs. 6-9), each scaled by its own
    amplitude:

        P(delta) = P_e^carrier(delta,t) + blue_amplitude*shape_blue^BOWKB(delta)
                                          + red_amplitude*shape_red^BOWKB(delta)

    Parameters, returns: identical contract to
    :func:`harmonic_full_spectrum`.
    """
    carrier = harmonic_carrier_excitation_probability(
        delta_hz,
        t_s,
        u0,
        waist_m,
        wavelength_m,
        mass_kg,
        probe_wavelength_m,
        rabi0_hz,
        temperature_z_k,
        temperature_r_k,
        misalignment_rad=misalignment_rad,
    )
    blue = bowkb_sideband_shape(
        delta_hz,
        1,
        u0,
        waist_m,
        wavelength_m,
        mass_kg,
        temperature_z_k,
        temperature_r_k,
        sideband_linewidth_hz,
    )
    red = bowkb_sideband_shape(
        delta_hz,
        -1,
        u0,
        waist_m,
        wavelength_m,
        mass_kg,
        temperature_z_k,
        temperature_r_k,
        sideband_linewidth_hz,
    )
    return carrier + blue_amplitude * blue + red_amplitude * red
