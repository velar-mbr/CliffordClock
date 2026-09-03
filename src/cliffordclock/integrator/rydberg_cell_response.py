# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rydberg vapor-cell response: quadratic Stark shift and EIT/Autler-Townes
observable (CONVENTIONS.md section 19, E43/E44, WP39 Phase A).

Scope boundary, matching the pattern set by
:mod:`cliffordclock.integrator.lattice_light_shift`: this module is
functions and a benchmark only, plain numpy, not wired into
:mod:`cliffordclock.pipeline`'s config surface. The regime is the
quadratic (isolated-state) Stark shift of a single Rb-85 Rydberg state,
the three-level ladder EIT susceptibility with a fourth, RF-coupled level
for Autler-Townes splitting, and the composition of many atoms' shifts
across an inhomogeneous field into one observed line profile. Full
Stark-map diagonalization beyond the quadratic regime, tensor
polarizability, and differentiability are later work.

Species and states: Rb-85, the 5S1/2-5P3/2-32D5/2-33P3/2 ladder, matching
Holloway, Gordon, Jefferts, Schwarzkopf, Anderson, Miller, Thaicharoen,
Raithel, "Broadband Rydberg Atom-Based Electric-Field Probe for
SI-Traceable, Self-Calibrated Measurements," IEEE Trans. Antennas Propag.
62, 6169 (2014), arXiv:1405.7066 -- the anchor this module's provenance
follows throughout, chosen over the Sedlacek et al. 2012 candidate
because Fig. 15 prints three (splitting, field) pairs directly with no
plot digitization, dossier Sec. 1. Every equation number cited below was
read from the arXiv PDF text directly this session (verbatim, not a
citing paper's summary); page/figure locations are given so a reviewer
can repeat the check.

CONVENTIONS.md section 19 carries the full derivation of the
Doppler-mismatch factor this module implements, the O'Sullivan-Stoicheff
to Yerokhin unit conversion, and every coefficient's source. This
docstring gives the equation-by-equation map from source to code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cliffordclock import constants
from cliffordclock.ensemble.species import ALPHA_AU_TO_SI

# ---------------------------------------------------------------------------
# Section A: atomic-unit <-> lab-unit conversions (E43, dossier risk 3)
# ---------------------------------------------------------------------------

#: Atomic unit of electric field strength, e / (4 pi eps0 a0^2), V/m.
#: Derived from the same CODATA constants already pinned in
#: :mod:`cliffordclock.constants` (the house pattern this project follows
#: for every derived unit constant: one transcription surface, not two).
#: Numerically 5.14220674763e11 V/m, matching the CODATA "atomic unit of
#: electric field" to the precision those constants carry.
ATOMIC_UNIT_FIELD_V_PER_M = constants.ELEMENTARY_CHARGE / (
    4.0 * math.pi * 8.8541878128e-12 * constants.BOHR_RADIUS**2
)


def _hartree_to_hz() -> float:
    """Hartree energy, expressed as a frequency via ``E_h / h``.

    The Hartree is ``alpha_fs^2 * m_e * c^2`` (alpha_fs the fine-structure
    constant); computed here from the Rydberg energy relation
    ``E_h = 2 * Ry = hbar^2 / (m_e * a0^2)`` so it draws only on constants
    already pinned in :mod:`cliffordclock.constants`, the same
    single-transcription-surface discipline as ``ATOMIC_UNIT_FIELD_V_PER_M``
    above. Numerically ``E_h/h = 6.579683920502e15 Hz`` (CODATA "Hartree
    energy in Hz"), reproduced by this formula to float64 precision.
    """
    hartree_j = constants.HBAR**2 / (constants.ELECTRON_MASS * constants.BOHR_RADIUS**2)
    return hartree_j / constants.PLANCK_H


#: Hartree energy expressed as a frequency, E_h / h, hertz. Exported for
#: any caller working directly in atomic energy units; this module's own
#: conversions (:func:`alpha0_au_to_mhz_per_vcm2`,
#: :func:`rydberg_quadratic_stark_shift_hz`) route through
#: :data:`cliffordclock.ensemble.species.ALPHA_AU_TO_SI` instead, so they
#: do not use this constant directly.
HARTREE_TO_HZ = _hartree_to_hz()


def alpha0_au_to_mhz_per_vcm2(alpha0_au: float) -> float:
    """Convert a scalar polarizability from atomic units (a0^3) to the
    O'Sullivan-Stoicheff frequency-shift-per-field-squared convention,
    MHz/(V/cm)^2.

    Both conventions share the same physical sign: ``Delta_E = -(1/2) *
    alpha0 * E^2`` (Yerokhin et al. 2016 Eq. 5, dossier Sec. 2d), so
    ``Delta_f[MHz] = k * E[V/cm]^2`` with ``k <= 0`` for a positive
    polarizability. Derivation, shown so a reviewer can check every
    factor:

    1. ``alpha0_SI = alpha0_au * ALPHA_AU_TO_SI`` (C^2 m^2 J^-1), reusing
       :data:`cliffordclock.ensemble.species.ALPHA_AU_TO_SI`
       (``4*pi*eps0*a0^3``) rather than re-deriving the same constant a
       second time (CONVENTIONS.md section 14's precedent for this
       exact discipline).
    2. ``Delta_E[J] = -(1/2) * alpha0_SI * E[V/m]^2``.
    3. ``Delta_f[Hz] = Delta_E[J] / h``.
    4. ``E[V/m] = 100 * E[V/cm]``, so ``E[V/m]^2 = 1e4 * E[V/cm]^2``.
    5. ``Delta_f[MHz] = Delta_f[Hz] / 1e6``.

    Combining steps 2-5: ``k = -alpha0_SI * 1e4 / (2 * h * 1e6) =
    -alpha0_SI * 5e-3 / h``.
    """
    alpha0_si = alpha0_au * ALPHA_AU_TO_SI
    return -alpha0_si * 5.0e-3 / constants.PLANCK_H


def mhz_per_vcm2_to_alpha0_au(k_mhz_per_vcm2: float) -> float:
    """Inverse of :func:`alpha0_au_to_mhz_per_vcm2`."""
    alpha0_si = -k_mhz_per_vcm2 * constants.PLANCK_H / 5.0e-3
    return alpha0_si / ALPHA_AU_TO_SI


# ---------------------------------------------------------------------------
# Section B: Rydberg-Ritz quantum defects (E44, mu_RF derivation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantumDefect:
    """Modified Rydberg-Ritz parameters, ``delta(n) = delta0 + delta2/(n -
    delta0)^2`` (Mack et al. 2011 Eq. 2, arXiv:1103.6221, page 3).

    ``n_star = n - delta(n)`` is solved by fixed-point iteration in
    :func:`effective_quantum_number` (the ``(n - delta0)`` on the
    right-hand side is itself only an initial guess; converged in under
    10 iterations for every state this module uses).
    """

    delta0: float
    delta2: float
    citation: str


#: Rb-85 nD5/2 quantum defect, read directly from Mack, Karlewski,
#: Hattermann, Hoeckh, Jessen, Cano, Fortagh, "Measurement of absolute
#: transition frequencies of 87Rb to nS and nD Rydberg states by means of
#: electromagnetically induced transparency," Phys. Rev. A 83, 052515
#: (2011), arXiv:1103.6221, Table I, the Rb-85 column, which the paper
#: attributes to Li, Mourachko, Noel, Gallagher, "Millimeter-wave
#: spectroscopy of cold Rb Rydberg atoms in a magneto-optical trap:
#: Quantum defects of the ns, np, and nd series," Phys. Rev. A 67, 052502
#: (2003) (Mack et al.'s own ref. [11]). Li et al. 2003 has no arXiv
#: preprint and is paywalled; this project could not verify its printed
#: table against the primary PDF directly, so the number is taken from
#: Mack et al. 2011's own reproduction of it (verified against that
#: paper's arXiv PDF text this session, page 4, Table I: "nD5/2 delta0
#: 1.346 465 7(3)  delta2 -0.596 0(2)" in the 85Rb column) rather than
#: from memory or a search-engine summary.
RB85_ND52_QUANTUM_DEFECT = QuantumDefect(
    delta0=1.3464657,
    delta2=-0.5960,
    citation=(
        "Li, Mourachko, Noel, Gallagher, PRA 67, 052502 (2003), "
        "as reproduced (85Rb column) in Mack et al., PRA 83, 052515 (2011) "
        "[arXiv:1103.6221], Table I"
    ),
)

#: Rb-85 nP3/2 quantum defect. Li et al. 2003 also covers the np series
#: (its own title), but with no accessible text this project used an
#: independent, later, higher-precision measurement instead:
#: Sanguinetti, Majeed, Jones, Varcoe, "Precision measurements of
#: quantum defects in the nP3/2 Rydberg States of 85Rb," J. Phys. B 42,
#: 165004 (2009), arXiv:0905.0571 (n = 36-63), Table 3, "Method 3" column
#: (their own preferred fit, a direct nonlinear fit of the extended
#: Rydberg-Ritz formula to the measured energies rather than a two-step
#: fit through a separately-fitted ionization energy): delta0 =
#: 2.64157(2), delta2 = 0.304(4), verified against that paper's arXiv PDF
#: text this session. The fitted defect is essentially n-independent
#: across their measured range (2.6414-2.6416 for delta(n) itself, Table
#: 1), so extrapolating the fit four principal quantum numbers below
#: their n=36 floor to n=33 (this module's calibration state) carries
#: little additional risk.
RB85_NP32_QUANTUM_DEFECT = QuantumDefect(
    delta0=2.64157,
    delta2=0.304,
    citation=(
        "Sanguinetti, Majeed, Jones, Varcoe, J. Phys. B 42, 165004 (2009) "
        "[arXiv:0905.0571], Table 3, Method 3"
    ),
)


def effective_quantum_number(n: int, defect: QuantumDefect, *, iterations: int = 20) -> float:
    """Solve ``n_star = n - delta0 - delta2/n_star^2`` by fixed-point
    iteration (Mack et al. 2011 Eq. 2 inverted). Converges to float64
    precision in well under ``iterations=20`` for every Rb Rydberg state
    this module evaluates (delta2/n_star^2 is a percent-level correction
    to delta0 for n >~ 20).
    """
    n_star = float(n) - defect.delta0
    for _ in range(iterations):
        delta = defect.delta0 + defect.delta2 / n_star**2
        n_star = float(n) - delta
    return n_star


# ---------------------------------------------------------------------------
# Section C: Numerov radial matrix element (mu_RF derivation, dossier risk 2)
# ---------------------------------------------------------------------------


def _turning_points(n_star: float, l_orbital: int) -> tuple[float, float]:
    """Classical inner/outer turning radii (atomic units) for angular
    momentum ``l_orbital`` at energy ``E = -1/(2 n_star^2)``, roots of
    ``r^2 - 2 n_star^2 r + l(l+1) n_star^2 = 0``.
    """
    a, b, c = 1.0, -2.0 * n_star**2, l_orbital * (l_orbital + 1) * n_star**2
    disc = b**2 - 4.0 * a * c
    inner = (-b - math.sqrt(disc)) / (2.0 * a)
    outer = (-b + math.sqrt(disc)) / (2.0 * a)
    return inner, outer


def _numerov_outward(
    n_star: float, l_orbital: int, r_min: float, r_stop: float, n_points: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Numerov-integrate ``u''(r) = g(r) u(r)`` outward from ``r_min`` to
    ``r_stop`` on a uniform grid, ``g(r) = l(l+1)/r^2 - 2/r - 2E`` for the
    quantum-defect (pure-Coulomb-tail) radial equation, atomic units.

    The boundary condition ``u(r) ~ r^(l+1)`` as ``r -> 0`` seeds the
    outward march; this is the numerically stable direction for a
    Rydberg bound state (the classically-allowed region is oscillatory
    with no growing/decaying split, and stopping ``r_stop`` only a
    little beyond the outer classical turning point, per
    :func:`_turning_points`, avoids the outer forbidden-zone tail where a
    naive outward march would eventually pick up the divergent solution).
    """
    r = np.linspace(r_min, r_stop, n_points)
    h = r[1] - r[0]
    energy = -1.0 / (2.0 * n_star**2)
    g = l_orbital * (l_orbital + 1) / r**2 - 2.0 / r - 2.0 * energy
    u = np.zeros(n_points)
    u[0] = r_min ** (l_orbital + 1)
    u[1] = r[1] ** (l_orbital + 1)
    for i in range(1, n_points - 1):
        u[i + 1] = (
            2.0 * (1.0 - 5.0 * h**2 * g[i] / 12.0) * u[i]
            - (1.0 + h**2 * g[i - 1] / 12.0) * u[i - 1]
        ) / (1.0 + h**2 * g[i + 1] / 12.0)
    return r, u, h


def numerov_radial_matrix_element(
    n_star_1: float,
    l1: int,
    n_star_2: float,
    l2: int,
    *,
    r_min: float = 1.0e-3,
    outer_margin: float = 1.2,
    n_points: int = 200_000,
) -> float:
    """Radial dipole matrix element ``<n1 l1| r |n2 l2>`` (units of a0)
    between two quantum-defect (pure-Coulomb-tail) Rydberg wavefunctions,
    by direct Numerov integration of the radial Schrodinger equation
    (atomic units) rather than a tabulated or fitted plot.

    This is the quantity Holloway et al. 2014 call ``Qn = R/a0``, "the
    normalized radial part of the dipole moment" (their Fig. 7 caption,
    arXiv:1405.7066 page 7), which they read off a log-log plot rather
    than compute directly; this function computes it instead, so it is
    verified against a print statement, and against the plot's own
    z-axis, that this module cannot read here.

    Approximation and its known limitation: both radial
    wavefunctions use the pure-Coulomb tail set by the state's effective
    quantum number ``n_star`` (:func:`effective_quantum_number`), not the
    true short-range atomic potential. Rb D and P states carry quantum
    defects of order 1-2.6 (real core penetration), so this
    approximation's accuracy is bounded, not exact. Cross-checked in
    ``tests/test_rydberg_cell_response.py`` against Sedlacek et al.
    2012's independently published, quantum-defect-derived value for the
    kinematically identical Rb 53D5/2 -> 54P3/2 transition (mu_RF =
    1.37e-26 C.m, arXiv:1205.4461 page 5): this function's prediction for
    that transition agrees with the published value to within a factor
    of 2, the tolerance this module states and defends (a wrong sign,
    wrong l, or order-of-magnitude coding error would miss by much more
    than that; a residual pure-Coulomb approximation error of this size,
    for states with quantum defects of order unity, does not). The
    registry value this module actually carries for 32D5/2 -> 33P3/2
    (:data:`RB85_MU_RF_32D52_33P32`) comes from a tighter, independent
    route instead: backed out self-consistently from Holloway et al.
    2014's own three published (splitting, field) calibration pairs
    (dossier Sec. 1, item 5), which agree with each other to within
    0.6%. This function's result for that same transition is reported
    alongside it as a disclosed, wider-tolerance cross-check, not
    substituted as the registry value.
    """
    _, outer_1 = _turning_points(n_star_1, l1)
    _, outer_2 = _turning_points(n_star_2, l2)
    r_stop = outer_margin * max(outer_1, outer_2)

    r1, u1, h1 = _numerov_outward(n_star_1, l1, r_min, r_stop, n_points)
    r2, u2, h2 = _numerov_outward(n_star_2, l2, r_min, r_stop, n_points)

    norm1 = np.sum(u1**2) * h1
    norm2 = np.sum(u2**2) * h2
    u1n = u1 / math.sqrt(norm1)
    u2n = u2 / math.sqrt(norm2)

    # The overall sign of a radial matrix element between two different
    # bound states depends on an arbitrary wavefunction phase convention
    # (here fixed by seeding u ~ r^(l+1) > 0 at r_min for both states) and
    # on how many radial nodes separate them, not on any physical
    # observable; only the magnitude enters the Rabi-frequency formula
    # this feeds (:func:`rf_transition_dipole_moment_from_quantum_defects`).
    return float(abs(np.sum(u1n * r1 * u2n) * h1))


def rf_transition_dipole_moment_from_quantum_defects(
    n1: int,
    defect1: QuantumDefect,
    l1: int,
    n2: int,
    defect2: QuantumDefect,
    l2: int,
) -> tuple[float, float]:
    """RF transition dipole moment ``mu_RF`` (C.m), Holloway et al. 2014
    Eq. 11 (verified, arXiv:1405.7066 page 7): ``wp_RF = 0.49 * e * a0 *
    Qn``, with ``Qn`` computed here by :func:`numerov_radial_matrix_element`
    in place of the paper's own Fig. 7 plot. Returns ``(mu_RF_C_m, Qn)``.
    """
    n_star_1 = effective_quantum_number(n1, defect1)
    n_star_2 = effective_quantum_number(n2, defect2)
    q_n = numerov_radial_matrix_element(n_star_1, l1, n_star_2, l2)
    mu_rf = 0.49 * constants.ELEMENTARY_CHARGE * constants.BOHR_RADIUS * q_n
    return mu_rf, q_n


# ---------------------------------------------------------------------------
# Section D: E43 registry -- scalar polarizabilities and mu_RF
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StarkPolarizability:
    """A published scalar polarizability ``alpha0`` (a0^3, ``Delta_E =
    -(1/2) alpha0 E^2`` convention, Yerokhin et al. 2016 Eq. 5) for one
    Rb-85 nD5/2 state, from one named source.
    """

    n: int
    alpha0_au: float
    alpha0_uncertainty_au: float | None
    citation: str


#: Rb-85 nD5/2 scalar polarizabilities, n = 30, 35, 50, BOTH independent
#: sources (measured and theoretical), cross-tabulated against each other
#: in Yerokhin, Buhmann, Fritzsche, Surzhykov, "Model-potential approach
#: to the calculation of dipole polarizabilities of alkali-metal atoms,"
#: Phys. Rev. A 94, 032503 (2016), arXiv:1608.04515, Table IV (dossier
#: Sec. 2e; the dossier's own pre-task read this table directly from the
#: arXiv PDF, pages 9-10). ``"theory"`` is Yerokhin et al.'s own
#: Dirac-Fock + core-polarization (DFCP) calculation; ``"experiment"`` is
#: O'Sullivan and Stoicheff, Phys. Rev. A 31, 2718 (1985) and Phys. Rev.
#: A 33, 1640 (1986) (jointly cited, refs. [32],[33], for every nD5/2 row
#: in Yerokhin's table). n=32, the calibration state this module needs
#: for the Stark term (:data:`RB85_32D52_ALPHA0_AU`), is not itself a
#: tabulated row in either source; the closest three that are (30, 35,
#: 50) fix a power-law fit instead (Section E below).
RB85_ND52_ALPHA0_TABULATED: dict[str, dict[int, StarkPolarizability]] = {
    "theory": {
        30: StarkPolarizability(
            30, 0.909e10, None, "Yerokhin et al., PRA 94, 032503 (2016) Table IV, DFCP"
        ),
        35: StarkPolarizability(
            35, 2.58e10, None, "Yerokhin et al., PRA 94, 032503 (2016) Table IV, DFCP"
        ),
        50: StarkPolarizability(
            50, 2.88e11, None, "Yerokhin et al., PRA 94, 032503 (2016) Table IV, DFCP"
        ),
    },
    "experiment": {
        30: StarkPolarizability(
            30,
            0.936e10,
            None,
            (
                "O'Sullivan & Stoicheff, PRA 31/33 (1985/1986), as tabulated "
                "in Yerokhin et al. 2016 Table IV"
            ),
        ),
        35: StarkPolarizability(
            35,
            2.53e10,
            0.08e10,
            (
                "O'Sullivan & Stoicheff, PRA 31/33 (1985/1986), as tabulated "
                "in Yerokhin et al. 2016 Table IV"
            ),
        ),
        50: StarkPolarizability(
            50,
            2.89e11,
            0.16e11,
            (
                "O'Sullivan & Stoicheff, PRA 31/33 (1985/1986), as tabulated "
                "in Yerokhin et al. 2016 Table IV"
            ),
        ),
    },
}


def _fit_alpha0_power_law(source: str) -> tuple[float, float]:
    """Fit ``alpha0(n_star) = C * n_star^p`` in log-log space to the three
    tabulated Rb-85 nD5/2 points from ``source`` ("theory" or
    "experiment"), the physically motivated Rydberg scaling for a
    diagonal scalar polarizability (Gallagher, "Rydberg Atoms," Cambridge
    Univ. Press, 1994, sec. 2.4; the same n*-power scaling the dossier
    names for the Inglis-Teller field estimate, Sec. 4). Returns
    ``(C, p)``.
    """
    rows = RB85_ND52_ALPHA0_TABULATED[source]
    n_stars = np.array([effective_quantum_number(n, RB85_ND52_QUANTUM_DEFECT) for n in rows])
    alphas = np.array([rows[n].alpha0_au for n in rows])
    p, log_c = np.polyfit(np.log(n_stars), np.log(alphas), 1)
    return float(math.exp(log_c)), float(p)


def derive_rb85_32d52_alpha0_au() -> float:
    """Derivation-based ``alpha0(32D5/2)`` for Rb-85, atomic units (a0^3).

    32D5/2 is Holloway et al. 2014's calibration state (Fig. 15) but is
    not itself one of the tabulated Yerokhin/O'Sullivan-Stoicheff rows
    (dossier Sec. 2e, risk 3). This function fits the power law
    ``alpha0(n_star) = C * n_star^p`` (:func:`_fit_alpha0_power_law`)
    separately to the theory and the experiment tabulated rows (n = 30,
    35, 50) and averages the two fits' predictions at
    ``n_star(32D5/2)``. Both fits reproduce their own three input rows to
    better than 4% (checked in
    ``tests/test_rydberg_cell_response.py::test_alpha0_power_law_fit_reproduces_inputs``)
    and land within 1% of each other at n=32, so the theory/experiment
    disagreement is not the dominant uncertainty here. This is a fitted,
    derived number, not a value printed in any source; every caller and
    docstring says so.
    """
    predictions = []
    n_star_32 = effective_quantum_number(32, RB85_ND52_QUANTUM_DEFECT)
    for source in ("theory", "experiment"):
        c, p = _fit_alpha0_power_law(source)
        predictions.append(c * n_star_32**p)
    return float(np.mean(predictions))


#: Derivation-based scalar polarizability of Rb-85 32D5/2, atomic units
#: (a0^3). See :func:`derive_rb85_32d52_alpha0_au` for the method; this
#: is NOT a value printed in Yerokhin et al. 2016 or O'Sullivan and
#: Stoicheff -- it is a power-law fit through their tabulated neighbors.
RB85_32D52_ALPHA0_AU = derive_rb85_32d52_alpha0_au()

#: mu_RF for the 32D5/2 -> 33P3/2 transition (Rb-85, 68.64 GHz), C.m.
#: Backed out self-consistently from Holloway et al. 2014's own three
#: published Fig. 15 (splitting, field) pairs via their Eq. 12
#: (:func:`field_from_at_splitting_v_per_m`, solved for mu_RF instead of
#: E): dossier Sec. 1, item 5, the "practical shortcut" the dossier
#: recommends over digitizing the paper's Fig. 7 plot. The three pairs
#: give 5.2452e-27, 5.2713e-27, and 5.2741e-27 C.m (mean used here),
#: agreeing with each other to within 0.6% -- itself a non-trivial
#: consistency check, since the three pairs are independent measurements
#: at three different field strengths. Cross-checked, at a wider,
#: disclosed tolerance, against an independent quantum-defect-based
#: Numerov calculation
#: (:func:`rf_transition_dipole_moment_from_quantum_defects`); see that
#: function's docstring and
#: ``tests/test_rydberg_cell_response.py::test_mu_rf_numerov_cross_check``
#: for the comparison.
RB85_MU_RF_32D52_33P32_C_M = 5.2635e-27

#: Wavelengths for the Holloway et al. 2014 68.64 GHz / 32D5/2-33P3/2
#: calibration (verified against the arXiv:1405.7066 PDF text this
#: session, page 8: "The blue laser is tuned to ~481.75 nm to couple
#: states 5P3/2 and 32D5/2"; the probe wavelength 780.24 nm is stated on
#: page 3 for the same 85Rb ladder).
HOLLOWAY_LAMBDA_PROBE_M = 780.24e-9
HOLLOWAY_LAMBDA_COUPLING_M = 481.75e-9

#: Rb-85 atomic mass, kg (CODATA/AME2020 atomic mass 84.911789738 u,
#: converted via :data:`cliffordclock.constants.ATOMIC_MASS_UNIT`). Used
#: as the default Doppler-averaging mass throughout this module; every
#: state and citation above is specifically for the 85Rb isotope
#: (Holloway et al. 2014's own anchor species), not natural-abundance Rb.
RB85_MASS_KG = 84.911789738 * constants.ATOMIC_MASS_UNIT


# ---------------------------------------------------------------------------
# Section E: E43 quadratic Stark shift, with validity guard
# ---------------------------------------------------------------------------


class RydbergStarkValidityError(ValueError):
    """Raised when a requested field exceeds this module's guarded
    quadratic-Stark validity window (Section E, CONVENTIONS.md section 19).
    """


def inglis_teller_field_v_per_m(n_star: float) -> float:
    """Order-of-magnitude Inglis-Teller field estimate, ``E_IT ~
    1/(3 n_star^5)`` atomic units (Gallagher, Rydberg Atoms, Cambridge
    Univ. Press, 1994; attributed to this text by both anchor papers'
    own quantum-defect references, dossier Sec. 4), converted to V/m via
    :data:`ATOMIC_UNIT_FIELD_V_PER_M`.

    This is an order-of-magnitude guard, not a fitted, published
    coefficient for the Rb nD5/2 series specifically (the dossier could
    not obtain O'Sullivan and Stoicheff's own nD avoided-crossing fit,
    Phys. Rev. A 33, 1640 (1986), body text; only their nS fit, dossier
    Sec. 4). :func:`rydberg_quadratic_stark_shift_hz` triggers well below
    this estimate rather than at it, the house pattern for validity
    windows enforced with margin.
    """
    e_it_au = 1.0 / (3.0 * n_star**5)
    return e_it_au * ATOMIC_UNIT_FIELD_V_PER_M


#: Fraction of the Inglis-Teller field at which
#: :func:`rydberg_quadratic_stark_shift_hz` raises
#: :class:`RydbergStarkValidityError`. The Inglis-Teller estimate is
#: itself an order-of-magnitude guard (see
#: :func:`inglis_teller_field_v_per_m`), so the guard triggers well
#: inside it rather than at it.
STARK_VALIDITY_MARGIN = 1.0 / 3.0


def rydberg_quadratic_stark_shift_hz(
    alpha0_au: float,
    field_v_per_m: float,
    n_star: float,
    *,
    margin: float = STARK_VALIDITY_MARGIN,
) -> float:
    """Quadratic Stark shift, ``Delta_f = -(1/2) * alpha0 * E^2 / h``
    (Yerokhin et al. 2016 Eq. 5 convention, dossier Sec. 2d), in Hz.

    Raises :class:`RydbergStarkValidityError` if ``field_v_per_m``
    exceeds ``margin * E_IT(n_star)`` (:func:`inglis_teller_field_v_per_m`),
    the guard against silently evaluating the isolated-state (quadratic)
    formula past the field where Stark-map diagonalization is actually
    required (CONVENTIONS.md section 19; the house pattern of enforcing
    validity windows with margin, e.g. section 13's BBR temperature
    guard).
    """
    guard = margin * inglis_teller_field_v_per_m(n_star)
    if field_v_per_m > guard:
        raise RydbergStarkValidityError(
            f"field {field_v_per_m:.3e} V/m exceeds the guarded quadratic-Stark "
            f"validity window ({margin:g} of the Inglis-Teller estimate "
            f"{inglis_teller_field_v_per_m(n_star):.3e} V/m for n_star={n_star:.2f}); "
            "Stark-map diagonalization is required above this field, out of scope "
            "for this module (CONVENTIONS.md section 19)."
        )
    alpha0_si = alpha0_au * ALPHA_AU_TO_SI
    shift_j = -0.5 * alpha0_si * field_v_per_m**2
    return shift_j / constants.PLANCK_H


# ---------------------------------------------------------------------------
# Section F: E44 four-level ladder susceptibility (Holloway Eqs 1-4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LadderSystem:
    """The four-level ladder's fixed parameters (Holloway et al. 2014
    Fig. 3, arXiv:1405.7066 page 3): probe |1>-|2>, coupling |2>-|3>, RF
    |3>-|4>.
    """

    mu_probe_c_m: float
    mu_coupling_c_m: float
    mu_rf_c_m: float
    gamma_12: float
    gamma_13: float
    gamma_14: float
    number_density_m3: float
    wavelength_probe_m: float
    wavelength_coupling_m: float


def ladder_susceptibility(
    delta_p: NDArray[np.float64] | float,
    delta_c: float,
    delta_rf: float,
    e_probe_v_per_m: float,
    e_coupling_v_per_m: float,
    e_rf_v_per_m: float,
    system: LadderSystem,
) -> NDArray[np.complex128]:
    """Susceptibility ``chi`` for the probe transition, Holloway et al.
    2014 Eqs. (1)-(4) (verified, arXiv:1405.7066 page 3):

        chi = [j N |mu_p| Omega_p / (eps0 |E_p|)] *
              [(Omega_RF)^2 + 4 D13 D14] /
              [D12 (Omega_RF)^2 + D14 (Omega_c)^2 + 4 D12 D13 D14]

    with ``Omega_x = |E_x| mu_x / hbar`` (Eq. 4) and ``D_1i = gamma_1i -
    j*Delta_{1i}`` (Eq. 3, this project's own generalization of the
    paper's ``D_1i = gamma_1i - j*Delta_p`` to nonzero coupling/RF
    detuning: the paper itself sets ``Delta_c = Delta_RF = 0`` for its
    own data and states the general form would carry those detunings
    too, citing its ref. [21], which this project did not chase down;
    the natural ladder generalization used here is ``D12`` carrying
    ``Delta_p`` alone, ``D13`` carrying the two-photon detuning
    ``Delta_p + Delta_c``, and ``D14`` carrying the three-photon
    detuning ``Delta_p + Delta_c + Delta_RF``, each ``D_1i`` summing the
    detunings of the states between |1> and |i>). Reduces to the
    standard 3-level ladder EIT susceptibility when ``Omega_RF = 0``
    (the ``D14`` terms drop from the numerator and the ``D12
    (Omega_RF)^2`` term vanishes from the denominator, leaving ``chi
    -prop- D13^-1`` for the two-photon-resonant pole, the same pole
    structure as Fleischhauer, Imamoglu, Marangos, Rev. Mod. Phys. 77,
    633 (2005) Eq. 13's lambda-type susceptibility and this project's
    own :func:`three_level_reduction_matches_rmp_pole` test).

    Uses the algebraically simplified prefactor ``N mu_p^2 / (hbar
    eps0)`` (since ``mu_p Omega_p / |E_p| = mu_p^2 / hbar`` by Eq. 4)
    rather than passing ``|E_p|`` through unused; ``e_probe_v_per_m`` is
    still accepted, for API symmetry with the coupling and RF fields and
    because the probe is treated as weak/non-perturbative in this
    ladder, and is otherwise ignored.
    """
    del e_probe_v_per_m  # cancels algebraically (see docstring); kept for API symmetry.
    delta_p = np.asarray(delta_p, dtype=np.float64)
    omega_c = e_coupling_v_per_m * system.mu_coupling_c_m / constants.HBAR
    omega_rf = e_rf_v_per_m * system.mu_rf_c_m / constants.HBAR

    d12 = system.gamma_12 - 1j * delta_p
    d13 = system.gamma_13 - 1j * (delta_p + delta_c)
    d14 = system.gamma_14 - 1j * (delta_p + delta_c + delta_rf)

    numerator = omega_rf**2 + 4.0 * d13 * d14
    denominator = d12 * omega_rf**2 + d14 * omega_c**2 + 4.0 * d12 * d13 * d14

    eps0 = 8.8541878128e-12
    prefactor = 1j * system.number_density_m3 * system.mu_probe_c_m**2 / (constants.HBAR * eps0)
    return np.asarray(prefactor * numerator / denominator, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Section G: Doppler averaging (E44, C7; Mohapatra et al. 2007 Eq. 1 structure)
# ---------------------------------------------------------------------------


def doppler_velocity_grid(
    temperature_k: float, mass_kg: float, n_points: int = 65
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gauss-Hermite quadrature nodes/weights for the 1D Maxwell-Boltzmann
    velocity distribution along the optical axis, ``sigma_v = sqrt(kB T /
    m)``. Deterministic by construction (no random sampling): a fixed
    quadrature rule, exact for polynomials up to degree ``2*n_points-1``
    against a Gaussian weight, is the right tool for integrating a smooth
    function against a Gaussian and needs no RNG or seed.
    """
    sigma_v = math.sqrt(constants.BOLTZMANN_K * temperature_k / mass_kg)
    nodes, weights = np.polynomial.hermite.hermgauss(n_points)
    velocities = math.sqrt(2.0) * sigma_v * nodes
    normalized_weights = weights / math.sqrt(math.pi)
    return velocities, normalized_weights


def doppler_averaged_susceptibility(
    delta_p: NDArray[np.float64],
    delta_c: float,
    delta_rf: float,
    e_probe_v_per_m: float,
    e_coupling_v_per_m: float,
    e_rf_v_per_m: float,
    system: LadderSystem,
    temperature_k: float,
    mass_kg: float,
    *,
    n_velocity_points: int = 65,
) -> NDArray[np.complex128]:
    """Doppler-averaged ladder susceptibility: :func:`ladder_susceptibility`
    evaluated at velocity-shifted detunings and averaged over a thermal
    (Maxwell-Boltzmann) velocity distribution, the structure Mohapatra,
    Jackson, Adams, PRL 98, 113003 (2007) Eq. (1) states for the 3-level
    case (velocity-resolved susceptibility integrated over ``N(v) dv``,
    arXiv:quant-ph/0612200 page 2), extended here to the 4-level (RF)
    case.

    Counter-propagating probe (wavevector ``+k_p``) and coupling
    (wavevector ``-k_c``, opposite direction) beams (Holloway et al. 2014
    Fig. 2(b); Mohapatra et al. 2007's own geometry) Doppler-shift the
    single-photon and two-photon detunings for an atom moving at ``v``
    along the beam axis as ``Delta_p -> Delta_p - k_p v`` and
    ``Delta_p + Delta_c -> Delta_p + Delta_c - (k_p - k_c) v`` (CONVENTIONS.md
    section 19's derivation). The RF leg's own Doppler shift is dropped:
    at 68.64 GHz the RF wavelength is centimeters, four orders of
    magnitude longer than the optical legs, the same approximation
    Sedlacek et al. 2012 state explicitly for their own RF leg
    (arXiv:1205.4461 page 8: "because the wavelength of the RF field is
    large, the Doppler effect on the 53D5/2->54P3/2 transition can be
    neglected").
    """
    velocities, weights = doppler_velocity_grid(temperature_k, mass_kg, n_velocity_points)
    k_p = 2.0 * math.pi / system.wavelength_probe_m
    k_c = 2.0 * math.pi / system.wavelength_coupling_m

    delta_p = np.asarray(delta_p, dtype=np.float64)
    total = np.zeros(delta_p.shape, dtype=np.complex128)
    for v, w in zip(velocities, weights, strict=True):
        shifted_delta_p = delta_p - k_p * v
        shifted_delta_c = delta_c + k_c * v  # (Delta_p+Delta_c) picks up -(k_p-k_c)v net.
        total += w * ladder_susceptibility(
            shifted_delta_p,
            shifted_delta_c,
            delta_rf,
            e_probe_v_per_m,
            e_coupling_v_per_m,
            e_rf_v_per_m,
            system,
        )
    return total


# ---------------------------------------------------------------------------
# Section H: Autler-Townes splitting, Doppler-mismatch factor (E44, C2/C3)
# ---------------------------------------------------------------------------


def autler_townes_splitting_hz(
    mu_rf_c_m: float, field_v_per_m: float, lambda_probe_m: float, lambda_coupling_m: float
) -> float:
    """Observed Autler-Townes splitting in the probe-transmission
    spectrum, ``Delta_f = (lambda_c / lambda_p) * Omega_RF / (2 pi)``
    (CONVENTIONS.md section 19's resolved Doppler-mismatch direction;
    Holloway et al. 2014 arXiv:1405.7066 page 8, "states are scaled by
    lambda_c/lambda_p [Mohapatra et al. 2007]"; equivalent to inverting
    their Eq. 12, verified below).
    """
    omega_rf = mu_rf_c_m * field_v_per_m / constants.HBAR
    bare_hz = omega_rf / (2.0 * math.pi)
    return bare_hz * (lambda_coupling_m / lambda_probe_m)


def field_from_at_splitting_v_per_m(
    delta_f_hz: float, mu_rf_c_m: float, lambda_probe_m: float, lambda_coupling_m: float
) -> float:
    """Holloway et al. 2014 Eq. (12) (verified, arXiv:1405.7066 page 8):

        |E_RF| = 2 pi (hbar / mu_RF) (lambda_p / lambda_c) Delta_f

    The exact algebraic inverse of :func:`autler_townes_splitting_hz`.
    """
    return (
        2.0
        * math.pi
        * (constants.HBAR / mu_rf_c_m)
        * (lambda_probe_m / lambda_coupling_m)
        * delta_f_hz
    )


def mu_rf_from_at_splitting_c_m(
    delta_f_hz: float, field_v_per_m: float, lambda_probe_m: float, lambda_coupling_m: float
) -> float:
    """Solve Holloway et al. 2014 Eq. (12) for ``mu_RF`` instead of
    ``E_RF``, given a published ``(Delta_f, E)`` pair. The "practical
    shortcut" the dossier recommends (Sec. 1, item 5) in place of
    digitizing Fig. 7.
    """
    return (
        2.0
        * math.pi
        * constants.HBAR
        * (lambda_probe_m / lambda_coupling_m)
        * delta_f_hz
        / field_v_per_m
    )


# ---------------------------------------------------------------------------
# Section I: per-atom composition over an inhomogeneous field (C5, C6)
# ---------------------------------------------------------------------------


def compose_inhomogeneous_eit_spectrum(
    delta_p: NDArray[np.float64],
    atom_field_magnitudes_v_per_m: NDArray[np.float64],
    atom_weights: NDArray[np.float64],
    alpha0_au: float,
    n_star: float,
    system: LadderSystem,
    *,
    delta_c: float = 0.0,
    delta_rf: float = 0.0,
    e_probe_v_per_m: float = 1.0,
    e_coupling_v_per_m: float = 1.0,
    e_rf_v_per_m: float = 0.0,
    temperature_k: float = 320.0,
    mass_kg: float = RB85_MASS_KG,
    n_velocity_points: int = 33,
) -> NDArray[np.complex128]:
    """Compose one observed line profile from many atoms, each shifted by
    its own local field's quadratic Stark shift on the Rydberg (coupling
    upper) level.

    A per-atom Stark shift ``delta_f_atom`` (Hz,
    :func:`rydberg_quadratic_stark_shift_hz`) on the Rydberg level shifts
    that atom's own two-photon (coupling) resonance by the same amount:
    ``delta_c_atom = delta_c + 2*pi*delta_f_atom``. The composed
    spectrum is the population-weighted sum of each atom's Doppler-
    averaged susceptibility (:func:`doppler_averaged_susceptibility`) at
    its own shifted ``delta_c_atom``, normalized so uniform weights over
    identical atoms reduce to a plain average.

    Two structural limits, each with a dedicated test in
    ``tests/test_rydberg_cell_response.py``:

    - Zero field everywhere: every atom's shift is exactly 0.0, so every
      term in the sum is byte-identical to the single-atom, unperturbed
      spectrum, and the composed spectrum equals it exactly (checked at
      the byte level, not just numerically close).
    - A uniform nonzero field: every atom shares the same shift, so the
      composed spectrum equals the single-atom spectrum evaluated at one
      shifted ``delta_c`` -- a pure translation of the same lineshape,
      with zero added width, and again checked directly against that
      single-atom evaluation.
    """
    weights = np.asarray(atom_weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    fields = np.asarray(atom_field_magnitudes_v_per_m, dtype=np.float64)
    delta_p = np.asarray(delta_p, dtype=np.float64)

    def _single_atom_spectrum(delta_c_atom: float) -> NDArray[np.complex128]:
        return doppler_averaged_susceptibility(
            delta_p,
            delta_c_atom,
            delta_rf,
            e_probe_v_per_m,
            e_coupling_v_per_m,
            e_rf_v_per_m,
            system,
            temperature_k,
            mass_kg,
            n_velocity_points=n_velocity_points,
        )

    # Every atom sharing the same field (the zero-field case included) is
    # evaluated once, not summed term by term: floating-point addition is
    # not associative, so a weighted sum of N identical terms is not
    # guaranteed bit-identical to one direct evaluation, and the C5 limit
    # checks require byte-identical artifacts, not merely close ones.
    if np.all(fields == fields[0]):
        shift_hz = (
            0.0
            if fields[0] == 0.0
            else rydberg_quadratic_stark_shift_hz(alpha0_au, fields[0], n_star)
        )
        return _single_atom_spectrum(delta_c + 2.0 * math.pi * shift_hz)

    total = np.zeros(delta_p.shape, dtype=np.complex128)
    for field_mag, weight in zip(fields, weights, strict=True):
        shift_hz = (
            0.0
            if field_mag == 0.0
            else rydberg_quadratic_stark_shift_hz(alpha0_au, field_mag, n_star)
        )
        total += weight * _single_atom_spectrum(delta_c + 2.0 * math.pi * shift_hz)
    return total


# ---------------------------------------------------------------------------
# Section J: wall-patch closed-form field model (C6 demonstrator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WallPatch:
    """A point-charge model of one photoionized surface-charge patch on
    the vapor cell wall (Patrick et al. 2025, arXiv:2502.07018, Sec.
    III/Fig. 5: visible light photoionizes a condensed alkali layer on
    the glass, leaving a positive surface charge). Position in meters,
    charge in coulombs.
    """

    position_m: NDArray[np.float64]
    charge_c: float


#: Vacuum permittivity, F/m (CODATA 2022; this module's own local
#: constant since :mod:`cliffordclock.constants` does not carry it).
VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12


def patch_field_v_per_m(
    position_m: NDArray[np.float64], patches: list[WallPatch]
) -> NDArray[np.float64]:
    """Closed-form electrostatic field at ``position_m`` from a
    superposition of point-charge wall patches, ``E = sum_i q_i (r -
    r_i) / (4 pi eps0 |r - r_i|^3)``. Zero-dependency Coulomb
    superposition, chosen over a finite-difference Laplace solve: this
    codebase's field-import machinery (:mod:`cliffordclock.fields.io`)
    consumes exported field maps rather than solving electrostatics
    itself (CONVENTIONS.md's standing "consume field exports" posture,
    also stated in the WP39 plan), and no finite-difference electrode
    solver exists in this codebase to reuse (the WP17 work referenced in
    this module's build plan is COMSOL-export ingestion, not a Laplace
    solver). A superposition of point (or, per :func:`disk_patch_field_v_per_m`,
    finite-disk) charges gives the qualitative phenomenology this
    demonstrator needs (field concentrated near a wall patch, falling
    off with distance) with an exact closed form and no numerical solve.
    """
    position_m = np.asarray(position_m, dtype=np.float64)
    field = np.zeros(3, dtype=np.float64)
    for patch in patches:
        r_vec = position_m - patch.position_m
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1e-9:
            continue
        field += patch.charge_c * r_vec / (4.0 * math.pi * VACUUM_PERMITTIVITY_F_PER_M * r_mag**3)
    return field


def cylindrical_cell_atom_positions(
    radius_m: float, length_m: float, n_atoms: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Seeded, deterministic (given ``rng``) sample of atom positions
    filling a cylindrical vapor cell (Patrick et al. 2025's own cell
    geometry, arXiv:2502.07018: 78 mm length, 25 mm diameter -- this
    function takes radius/length as parameters rather than hardcoding
    that geometry, so the benchmark and notebook can use it directly and
    a caller can explore other cell sizes).
    """
    z = rng.uniform(-length_m / 2.0, length_m / 2.0, n_atoms)
    r = radius_m * np.sqrt(rng.uniform(0.0, 1.0, n_atoms))
    theta = rng.uniform(0.0, 2.0 * math.pi, n_atoms)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=-1)
