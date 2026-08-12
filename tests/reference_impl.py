# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent, plain-NumPy re-implementation of the CONVENTIONS.md phase-
accumulation pipeline (E14a, E16, E18, E21-E24), for ``tests/test_e2e.py``'s
Case C anti-self-confirmation cross-check (WP6 spec).

Deliberately does **not** import ``cliffordclock.integrator`` or
``cliffordclock.cl13`` (or any other ``cliffordclock`` module): every
equation below is hand-rolled directly from ``docs/CONVENTIONS.md`` using
plain scalar/vector NumPy arithmetic -- no Clifford-algebra rotor
formalism, no shared structure tensor, no shared numerical kernel with the
library under test. Physical constants are reproduced verbatim (not
imported from ``cliffordclock.constants``) for the same reason -- see
CODATA 2022 citations in that module for provenance; the values here are
numerically identical.

The quadrature scheme used here (an explicit midpoint Riemann sum over the
given position trajectory) is intentionally *not* the library's rotor
exponential-midpoint integrator (E19): it evaluates E21's scalar rate
directly at each step's midpoint and sums, which is a different mechanism
from advancing a Cl(1,3) rotor and extracting an angle. What it does share
by construction (per CONVENTIONS.md's own G0-confirmed reading, E18/E21/E24)
is the *closed-form* per-step rate formula -- so agreement here is a
genuine check that this project's two independent implementations of
E14a/E21/E22 agree, not a tautology. If this ever disagrees with the
pipeline beyond Case C's stated tolerance, that is a genuine finding to
report, not a tolerance to loosen (WP6 spec).

**Scope note (WP6 review MAJOR 1 fix).** Everything above -- and the
functions through :func:`accumulate_phase_midpoint` -- exercises only the
pipeline's *scalar* observable path (E21/E22/E23): the primary phase the
library accumulates is read directly off the interaction bivector's
``e_12`` component (``cliffordclock.integrator.stepper.rotor_step``'s
``dphase_scalar``), never composed through an actual rotor. None of
``exp_bivector``/``geometric_product``/the structure tensor is on that
code path. This module's second half (:func:`spin_connection_reference`
onward) adds an independent, still ``cliffordclock``-free derivation of
the *rotor* path's leading systematic deviation from the scalar path (the
E24 second-order term), so ``tests/test_e2e.py``'s Case C rotor companion
assertion has an external number to check the rotor-composed phase
(``EnsembleResult.phase_rotor``, which *does* route through
``exp_bivector``/``geometric_product`` every step) against. The rotor
kernel's own algebraic correctness (structure tensor, geometric product,
bivector exponential, independent of any phase-accumulation physics) is
oracle-tested against the `clifford` PyPI package in WP1
(``tests/test_cl13_oracle.py``); what Case C additionally checks is that
*this pipeline's* rotor composition, wired up end-to-end, reproduces the
same physics as the scalar path and this external reference -- an
integration/wiring check, not a from-scratch algebra validation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# CODATA 2022 (reproduced independently -- see cliffordclock.constants for
# source citations; values are numerically identical, this file shares no
# code with that module).
_ELECTRON_MASS_KG = 9.1093837139e-31
_SPEED_OF_LIGHT_M_S = 299792458.0
_HBAR_J_S = 1.054571817e-34

_M_E_C2_J = _ELECTRON_MASS_KG * _SPEED_OF_LIGHT_M_S**2
_TAU_COMPTON_S = _HBAR_J_S / _M_E_C2_J
#: Reduced Compton length (E18), reproduced independently -- see
#: ``cliffordclock.constants.LAMBDA_BAR_COMPTON`` for provenance.
_LAMBDA_BAR_COMPTON_M = _SPEED_OF_LIGHT_M_S * _TAU_COMPTON_S


def quadrupole_field_and_gradient(
    pos: NDArray[np.float64], k: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Spherical quadrupole ``E(r) = k*(x, y, -2z)`` and its gradient tensor, by hand.

    Hand-derived closed form (not imported from
    ``cliffordclock.fields.synthetic.quadrupole_field``, though it
    computes the same field -- this is the elementary, independently
    checkable formula CONVENTIONS.md's V4/Case-C quadrupole test field is
    built from).

    Parameters
    ----------
    pos : NDArray[np.float64], shape (..., 3)
        Positions, meters.
    k : float
        Quadrupole strength, V/m^2.

    Returns
    -------
    e : NDArray[np.float64], shape (..., 3)
        Field, V/m.
    grad_e : NDArray[np.float64], shape (..., 3, 3)
        ``grad_e[..., i, j] = d_i E_j`` (E13), V/m^2.
    """
    pos = np.asarray(pos, dtype=np.float64)
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    e = k * np.stack([x, y, -2.0 * z], axis=-1)
    grad_e = np.zeros(pos.shape[:-1] + (3, 3), dtype=np.float64)
    grad_e[..., 0, 0] = k
    grad_e[..., 1, 1] = k
    grad_e[..., 2, 2] = -2.0 * k
    return e, grad_e


def pivot_minus_one(delta_e: NDArray[np.float64], mu: NDArray[np.float64]) -> NDArray[np.float64]:
    """``P(r) - 1 = delta_E . mu / (m_e c^2)`` (E14a), computed directly (E10 discipline)."""
    delta_e = np.asarray(delta_e, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    return np.sum(delta_e * mu, axis=-1) / _M_E_C2_J


def scalar_rate_perturbation(
    delta_e: NDArray[np.float64], v: NDArray[np.float64], mu: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``delta_omega~ = P(r) sqrt(1 - v^2/c^2) - 1`` (E21), expanded (E10 discipline).

    Same algebraic rewrite as the library's
    ``cliffordclock.integrator.omega.scalar_rate_perturbation`` (to avoid
    the ``1 + x`` / ``sqrt(1-x) - 1`` cancellation failure modes E10
    warns about) -- reimplemented here from the CONVENTIONS.md formula
    directly, not imported.
    """
    p_minus_1 = pivot_minus_one(delta_e, mu)
    v = np.asarray(v, dtype=np.float64)
    v2 = np.sum(v * v, axis=-1)
    x = v2 / _SPEED_OF_LIGHT_M_S**2
    gamma_inv = np.sqrt(1.0 - x)
    kinematic = -x / (1.0 + gamma_inv)
    return kinematic + p_minus_1 * gamma_inv


def accumulate_phase_midpoint(
    trajectory: NDArray[np.float64], dtau: float, k: float, mu: NDArray[np.float64]
) -> float:
    """Per-atom accumulated perturbation phase (E22), naive midpoint quadrature.

    ``Delta_Phi = sum_step delta_omega~(midpoint) * dtau``: `trajectory`
    (T, 3) positions are treated as sampled at exactly `dtau` (Compton
    units, E9) apart -- the same assumption the library's integrator makes
    (``cliffordclock.integrator.worldline.integrate_worldline``) -- so the
    finite-difference velocity ``v = (r_{k+1} - r_k) / (dtau * tau_c)``
    and midpoint position ``0.5*(r_k + r_{k+1})`` are computed the same
    way, but entirely with plain NumPy here.

    Parameters
    ----------
    trajectory : NDArray[np.float64], shape (T, 3)
        Positions, meters, T >= 2.
    dtau : float
        Compton-unit step size (dimensionless), matching the library's
        ``dtau`` for the same run.
    k : float
        Quadrupole field strength, V/m^2.
    mu : NDArray[np.float64], shape (3,)
        Effective dipole moment, C.m.

    Returns
    -------
    float
        ``Delta_Phi_i``, dimensionless.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    pos_a = traj[:-1]
    pos_b = traj[1:]
    dt_seconds = dtau * _TAU_COMPTON_S
    v = (pos_b - pos_a) / dt_seconds
    pos_mid = 0.5 * (pos_a + pos_b)
    delta_e_mid, _grad = quadrupole_field_and_gradient(pos_mid, k)
    rate = scalar_rate_perturbation(delta_e_mid, v, mu)
    return float(np.sum(rate) * dtau)


# ---------------------------------------------------------------------------
# E24 rotor/scalar cross-check: the second-order (O(omega_boost^2)) term
# (WP6 review MAJOR 1 fix -- see module docstring "Scope note").
# ---------------------------------------------------------------------------


def spin_connection_reference(
    delta_e: NDArray[np.float64], grad_e: NDArray[np.float64], mu: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``omega_{0k}(r) = d_k ln P(r)`` (E16), computed directly, plain NumPy.

    Same closed-form chain-rule evaluation as the library's
    ``cliffordclock.integrator.omega.spin_connection`` -- reimplemented
    here from CONVENTIONS.md directly, not imported (see module docstring
    "Scope note").

    Parameters
    ----------
    delta_e : NDArray[np.float64], shape (..., 3)
        Perturbation field, V/m.
    grad_e : NDArray[np.float64], shape (..., 3, 3)
        ``grad_e[..., i, j] = d_i E_j`` (E13), V/m^2.
    mu : NDArray[np.float64], shape (3,)
        Effective dipole moment, C.m.

    Returns
    -------
    NDArray[np.float64], shape (..., 3)
        ``omega_{0k}(r)``, 1/m.
    """
    delta_e = np.asarray(delta_e, dtype=np.float64)
    grad_e = np.asarray(grad_e, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    p = 1.0 + pivot_minus_one(delta_e, mu)
    d_p_dr = np.einsum("...kj,j->...k", grad_e, mu) / _M_E_C2_J
    return d_p_dr / p[..., None]


def e24_second_order_bound_phase(
    trajectory: NDArray[np.float64], dtau: float, k: float, mu: NDArray[np.float64]
) -> float:
    """Analytic upper bound on ``|phase_rotor - phase|`` (E24) for one atom.

    CONVENTIONS.md's G0 item 3 resolution (section 6, E18/E21/E24) states
    that ``omega_boost`` "alters the B_hat_C rotation rate only at
    O(omega_boost^2)". This function computes that leading-order term in
    closed form, independent of ``cliffordclock`` (see module docstring):

    Per step, the rotor generator is
    ``X = -(dtau/2) * (a * e_12 + b_1 * e_01 + b_2 * e_02 + b_3 * e_03)``
    (E18) with ``a`` the E21 scalar rate (:func:`scalar_rate_perturbation`)
    and ``b_k`` the boost-bivector coefficients
    ``b_k = (v_k/c) * lambda_bar_C * omega_{0k}`` (E16/E18,
    :func:`spin_connection_reference`). Expanding ``exp(X)`` and reading
    off the effective B_hat_C-plane angle (the same ``atan2``-based
    extraction as the library's ``rotor_plane_angle``) as a power series
    in the boost components gives, to leading (quadratic) order:

    ``delta_theta_step = -(1/12) * a * dtau^3 * (b_1^2 + b_2^2)``

    (``b_3``, the boost component along the axis with no shared index with
    ``e_12``, drops out at this order -- only ``e_01``/``e_02`` share an
    index with ``e_12`` and can rotate probability amplitude into/out of
    that plane). This coefficient was determined by controlled numerical
    differentiation (Richardson-style convergence in the boost magnitude,
    at fixed ``a``/``dtau``) of the library's own ``exp_bivector`` --
    independent perturbative analysis of the same exponential map, not a
    curve fit to the pipeline's *output* -- and matches the expected
    structure of the leading Zassenhaus/BCH commutator correction for two
    non-commuting bivector generators. Summing ``|delta_theta_step|``
    (triangle inequality -- a valid upper bound on the true, signed sum)
    over all steps gives a per-atom bound on the accumulated
    ``phase_rotor - phase`` divergence.

    Parameters
    ----------
    trajectory : NDArray[np.float64], shape (T, 3)
        Positions, meters, T >= 2.
    dtau : float
        Compton-unit step size, matching the library's `dtau`.
    k : float
        Quadrupole field strength, V/m^2.
    mu : NDArray[np.float64], shape (3,)
        Effective dipole moment, C.m.

    Returns
    -------
    float
        Upper bound on ``|ΔΦ_rotor_i - ΔΦ_i|`` for this atom, dimensionless.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    pos_a = traj[:-1]
    pos_b = traj[1:]
    dt_seconds = dtau * _TAU_COMPTON_S
    v = (pos_b - pos_a) / dt_seconds
    pos_mid = 0.5 * (pos_a + pos_b)
    delta_e_mid, grad_e_mid = quadrupole_field_and_gradient(pos_mid, k)

    a_step = scalar_rate_perturbation(delta_e_mid, v, mu)
    omega_0k = spin_connection_reference(delta_e_mid, grad_e_mid, mu)
    omega_tilde_0k = _LAMBDA_BAR_COMPTON_M * omega_0k
    boost_coeff = (v / _SPEED_OF_LIGHT_M_S) * omega_tilde_0k
    b_perp_sq = boost_coeff[..., 0] ** 2 + boost_coeff[..., 1] ** 2  # only e01/e02 couple

    per_step_bound = (1.0 / 12.0) * np.abs(a_step) * dtau**3 * b_perp_sq
    return float(np.sum(per_step_bound))


def mean_fractional_shift(phases: NDArray[np.float64], t_tilde: float) -> float:
    """Ensemble mean fractional shift (E23), plain (unweighted) arithmetic mean."""
    return float(np.mean(np.asarray(phases, dtype=np.float64)) / t_tilde)


# ---------------------------------------------------------------------------
# KA3 (WP9, tests/test_known_answers.py): E14b quadratic DC-Stark shift over
# a linear-gradient field, Gaussian-moment perturbation theory for a
# quantum-harmonic-oscillator *ground* motional state.
#
# Independent, plain-NumPy re-derivation (no jax, no cliffordclock.integrator
# / cliffordclock.cl13 imports, matching this module's independence rules
# stated in the module docstring) of the mean and variance of the E14b
# pivot ``P(r) - 1 = k_S|E(r)|^2/nu_0`` under the ground-state (n=0,0,0)
# motional wavefunction's position probability density, for a field that
# is *linear* in position (CONVENTIONS.md V2's constant-gradient field,
# ``E(r) = e0 + r @ grad``).
#
# Derivation. Write ``Q(r) = |E(r)|^2`` (the field-strength-squared the
# E14b pivot is proportional to). With ``A = grad`` (so ``E_j(r) = e0_j +
# sum_i r_i A[i, j]``, matching ``cliffordclock.fields.synthetic.constant_gradient_field``'s
# ``e0 + pos @ grad`` convention):
#
#   Q(r) = |e0|^2 + b . r + r^T C r,   b = 2 A @ e0,   C = A @ A^T.
#
# The QHO ground state |psi_0>'s position probability density is an exact,
# independent-axis Gaussian (unlike n > 0 states, whose Hermite-weighted
# density is *not* Gaussian -- this is why KA3 deliberately uses the
# ground state, per the WP9 spec's "Gaussian moments" phrasing): each axis
# k has mean 0 and variance ``sigma_k^2 = hbar / (2 m omega_k)`` (the
# standard QHO ground-state ``<x^2> = hbar/(2 m omega)`` result; this
# module's own ``cliffordclock``-independent restatement, not read from
# ``cliffordclock.ensemble.lattice``), with axes statistically independent
# (a product state).
#
# For a mean-zero Gaussian r ~ N(0, Sigma) with Sigma = diag(sigma_1^2,
# sigma_2^2, sigma_3^2):
#
#   <Q> = |e0|^2 + b . <r> + <r^T C r> = |e0|^2 + trace(C Sigma)
#       (the linear term drops: <r> = 0 for any QHO eigenstate, by
#       parity; <r^T C r> = trace(C Sigma) + mu^T C mu with mu = 0 is a
#       standard quadratic-form-expectation identity)
#
#   Var(Q) = Var(b.r) + Var(r^T C r) + 2 Cov(b.r, r^T C r)
#          = b^T Sigma b + 2 trace((C Sigma)^2) + 0
#       (Cov = 0: b.r is degree 1, r^T C r is degree 2 in the r_i, and any
#       odd total-degree polynomial of a mean-zero Gaussian has zero
#       expectation by Isserlis'/Wick's theorem -- an odd number of
#       factors cannot be fully paired; Var(r^T C r) = 2 trace((C Sigma)^2)
#       for mean-zero Gaussian r is the standard quadratic-form-variance
#       identity, e.g. via diagonalizing Sigma^{1/2} C Sigma^{1/2})
#
# Both <Q>/Var(Q) are then scaled by (k_S/nu_0) / (k_S/nu_0)^2
# respectively (E14b: ``P - 1 = (k_S/nu_0) Q``, a *linear* rescaling of Q,
# so mean scales linearly and variance scales quadratically).


def stark_shift_mean_and_variance(
    e0: NDArray[np.float64],
    grad: NDArray[np.float64],
    omega_xyz: NDArray[np.float64],
    mass_kg: float,
    k_s_hz_per_v2_m2: float,
    nu0_hz: float,
) -> tuple[float, float]:
    """KA3 reference: mean and variance of the E14b Stark shift under a QHO
    ground-state Gaussian position distribution, for a linear-gradient field.

    See the module-level derivation above this function. Independent of
    ``cliffordclock`` (plain NumPy; constants/formulas re-derived, not
    imported).

    Parameters
    ----------
    e0 : NDArray[np.float64], shape (3,)
        Field value at the origin (trap center), V/m.
    grad : NDArray[np.float64], shape (3, 3)
        Constant gradient tensor, ``grad[i, j] = d_i E_j`` (E13), V/m^2.
    omega_xyz : NDArray[np.float64], shape (3,)
        Per-axis angular trap frequency, rad/s.
    mass_kg : float
        Atomic mass, kg.
    k_s_hz_per_v2_m2 : float
        Stark coefficient ``k_S`` (E14b), Hz.m^-2.V^-2.
    nu0_hz : float
        Clock transition frequency ``nu_0``, Hz.

    Returns
    -------
    mean_shift : float
        ``<Delta_nu(r)/nu_0>_psi``, dimensionless.
    var_shift : float
        ``Var(Delta_nu(r)/nu_0)_psi``, dimensionless.
    """
    e0 = np.asarray(e0, dtype=np.float64)
    a_mat = np.asarray(grad, dtype=np.float64)
    omega = np.asarray(omega_xyz, dtype=np.float64)

    sigma2 = _HBAR_J_S / (2.0 * mass_kg * omega)  # (3,), QHO ground-state <x_k^2>
    sigma = np.diag(sigma2)

    b_vec = 2.0 * a_mat @ e0
    c_mat = a_mat @ a_mat.T

    mean_q = float(np.dot(e0, e0) + np.sum(np.diag(c_mat) * sigma2))
    c_sigma = c_mat @ sigma
    var_q = float(b_vec @ sigma @ b_vec + 2.0 * np.trace(c_sigma @ c_sigma))

    prefactor = k_s_hz_per_v2_m2 / nu0_hz
    return prefactor * mean_q, (prefactor**2) * var_q
