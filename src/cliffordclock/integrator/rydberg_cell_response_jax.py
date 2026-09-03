# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differentiable JAX port of the quadratic-path Rydberg vapor-cell chain
(WP41, CONVENTIONS.md section 21 addendum, E45): the same physics
`cliffordclock.integrator.rydberg_cell_response` implements for the
quadratic (isolated-state) Stark regime, reimplemented so that
`jax.grad`/`jax.jit` work end to end. This module adds no new physics
claim for the Stark/EIT chain itself: every formula below is the SAME
equation the reference module cites (Yerokhin et al. 2016 Eq. 5 for the
quadratic Stark shift; Holloway et al. 2014 Eqs. 1-4 for the ladder
susceptibility; Mohapatra et al. 2007 Eq. 1's structure for Doppler
averaging), evaluated by `jax.numpy` in place of `numpy`. The one new
piece is a differentiable field model for the field-reconstruction
demonstrator (`cell_field_magnitude_v_per_m_jax`, documented in its own
section below): this module's own construction, built for this work
package, its design choices stated in full where it is defined.

**Why the quadratic path only.** The research pre-task's own
differentiability-risk assessment (this project's private WP40/WP41
planning note) found that `jax.numpy.linalg.eigh`'s reverse-mode gradient
carries a `(lambda_i - lambda_j)^-1` term that is ill-conditioned near an
avoided crossing, with no published precedent for a differentiable
Rydberg Stark-map eigensolve to lean on. The quadratic Stark shift
(`rydberg_quadratic_stark_shift_hz_jax`) has no eigensolve in it at all:
it is a closed-form polynomial in the field, so its `jax.grad` is exact
and unconditioned everywhere, with no crossing-adjacent numerical risk to
manage. Every deliverable this module supports (agreement with the
reference implementation, gradient validation against finite differences,
jit determinism, a memory bound, and the gradient-based field-
reconstruction demonstrator) is built entirely on this path, inside the
same guarded validity window the reference module enforces
(`cliffordclock.integrator.rydberg_cell_response.rydberg_quadratic_stark_shift_hz`'s
own `RydbergStarkValidityError` guard). A full Stark-map eigensolve
differentiable path is out of scope for this module.

**No clamp, no `jnp.where`, no branch anywhere in this module's physics
chain.** This is a structural fact worth stating plainly. WP37's own
differentiable lattice-light-shift module hit a `NaN`-gradient bug from
this class of site: a hard clamp created a flat region whose subgradient
divided a root-find's implicit-function-theorem tangent by zero. Every
function below is polynomial or a ratio of functions whose denominator
is bounded away from zero by construction:
- `rydberg_quadratic_stark_shift_hz_jax` is `-(1/2) alpha0 E^2 / h`, a
  degree-2 polynomial in `E` with no branch.
- `ladder_susceptibility_jax`'s denominator is a sum of products of
  `D_1i = gamma_1i - j*Delta_i` terms; every `gamma_1i` is a positive
  physical decay rate, so every `D_1i` has a strictly positive real part
  and the denominator is bounded away from `0` for any finite detuning.
  `tests/test_rydberg_cell_response_jax.py::TestNaNSweep` checks this
  argument empirically, with a direct `jax.grad` sweep across extreme
  drive and detuning inputs.
- `cell_field_magnitude_v_per_m_jax`'s one potential singularity (a
  patch-model term with a `1/distance^2`-family point-source falloff) is
  regularized by a fixed additive softening length in the denominator
  (`patch_softening_m^2`, added directly to the squared distance, with
  no `jnp.where` comparison anywhere in the formula), so the denominator
  is bounded below by `patch_softening_m^2 > 0` for every input,
  including a position exactly at the patch location. This is a smooth
  (Plummer-style) softening: the function's gradient is finite and
  continuous everywhere, including at the softened core, with no branch
  in the computation.

**Float64, required.** As with every other JAX module in this package,
`cliffordclock`'s own `__init__.py` enables `jax_enable_x64` as an
import-time side effect; this module relies on that already having run
(`cliffordclock.integrator.lattice_light_shift_jax`'s own module
docstring states the identical dependency).

**Scope boundary.** No pipeline wiring: this module is functions only,
matching `cliffordclock.integrator.rydberg_cell_response`'s own stated
scope boundary and the E40/E41/E42 JAX modules' precedent. See
`tests/test_rydberg_cell_response_jax.py` for the agreement, gradient,
determinism, and memory-bound checks, and
`benchmarks/run_rydberg_field_reconstruction.py` for the gradient-based
field-reconstruction demonstrator.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from cliffordclock.constants import BOLTZMANN_K, HBAR, PLANCK_H
from cliffordclock.ensemble.species import ALPHA_AU_TO_SI
from cliffordclock.integrator.rydberg_cell_response import RB85_MASS_KG

__all__ = [
    "VACUUM_PERMITTIVITY_F_PER_M",
    "LadderSystemJax",
    "rydberg_quadratic_stark_shift_hz_jax",
    "ladder_susceptibility_jax",
    "doppler_averaged_susceptibility_jax",
    "compose_inhomogeneous_eit_spectrum_jax",
    "cell_field_magnitude_v_per_m_jax",
    "rb85_field_reconstruction_forward_model_jax",
]

#: Vacuum permittivity, F/m (CODATA 2022), matching the local constant the
#: reference module's own `ladder_susceptibility` defines inline (that
#: module's own comment: `cliffordclock.constants` does not carry it).
VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12


# ---------------------------------------------------------------------------
# Section A: the ladder system and its susceptibility (E44 port)
# ---------------------------------------------------------------------------


class LadderSystemJax(NamedTuple):
    """JAX-pytree counterpart to
    :class:`cliffordclock.integrator.rydberg_cell_response.LadderSystem`: a
    plain `NamedTuple`, so `jax.jit`/`jax.grad` can pass it through and,
    if a caller builds it from traced leaves, differentiate with respect
    to its fields directly. Same fields, same physical meaning, as the
    reference dataclass.
    """

    mu_probe_c_m: jnp.ndarray
    mu_coupling_c_m: jnp.ndarray
    mu_rf_c_m: jnp.ndarray
    gamma_12: jnp.ndarray
    gamma_13: jnp.ndarray
    gamma_14: jnp.ndarray
    number_density_m3: jnp.ndarray
    wavelength_probe_m: jnp.ndarray
    wavelength_coupling_m: jnp.ndarray


def ladder_susceptibility_jax(
    delta_p: jnp.ndarray,
    delta_c: jnp.ndarray | float,
    delta_rf: jnp.ndarray | float,
    e_probe_v_per_m: jnp.ndarray | float,
    e_coupling_v_per_m: jnp.ndarray | float,
    e_rf_v_per_m: jnp.ndarray | float,
    system: LadderSystemJax,
) -> jnp.ndarray:
    """Susceptibility `chi` for the probe transition, a direct `jax.numpy`
    port of
    :func:`cliffordclock.integrator.rydberg_cell_response.ladder_susceptibility`
    (Holloway et al. 2014 Eqs. 1-4, CONVENTIONS.md section 19's own
    equation-by-equation citation). Same formula, same sign convention,
    no numerical difference from the reference implementation: this is
    pure complex algebra, no eigenproblem and no root-find, so both
    implementations evaluate the identical expression.

    Every argument broadcasts under ordinary `jax.numpy` rules, so this
    function accepts scalars for a single-atom evaluation
    (:func:`doppler_averaged_susceptibility_jax`) or higher-rank arrays
    for the batched per-atom/per-velocity evaluation
    (:func:`compose_inhomogeneous_eit_spectrum_jax`), with no separate
    code path for either case.

    `e_probe_v_per_m` cancels algebraically out of this formula (the
    reference function's own docstring shows the cancellation); it is
    still accepted, for signature parity with the reference function and
    because a caller may want to pass it symbolically for API symmetry
    with `e_coupling_v_per_m`/`e_rf_v_per_m`.
    """
    del e_probe_v_per_m  # cancels algebraically; kept for API symmetry (see docstring).
    delta_p = jnp.asarray(delta_p)
    omega_c = e_coupling_v_per_m * system.mu_coupling_c_m / HBAR
    omega_rf = e_rf_v_per_m * system.mu_rf_c_m / HBAR

    d12 = system.gamma_12 - 1j * delta_p
    d13 = system.gamma_13 - 1j * (delta_p + delta_c)
    d14 = system.gamma_14 - 1j * (delta_p + delta_c + delta_rf)

    numerator = omega_rf**2 + 4.0 * d13 * d14
    denominator = d12 * omega_rf**2 + d14 * omega_c**2 + 4.0 * d12 * d13 * d14

    prefactor = (
        1j
        * system.number_density_m3
        * system.mu_probe_c_m**2
        / (HBAR * VACUUM_PERMITTIVITY_F_PER_M)
    )
    return prefactor * numerator / denominator


def _gauss_hermite_nodes_weights(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed Gauss-Hermite quadrature nodes/normalized weights, computed
    once with plain `numpy` (matching
    :func:`cliffordclock.integrator.rydberg_cell_response.doppler_velocity_grid`'s
    own quadrature rule). `n_points` is a static Python `int` (the array
    shape it produces is fixed at trace time), so calling this from
    inside a `jax.jit`-traced function is safe: it runs once, at trace
    time, and produces the same fixed-shape constant array every time for
    a given `n_points`.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n_points)
    return nodes, weights / math.sqrt(math.pi)


def doppler_averaged_susceptibility_jax(
    delta_p: jnp.ndarray,
    delta_c: jnp.ndarray | float,
    delta_rf: jnp.ndarray | float,
    e_probe_v_per_m: jnp.ndarray | float,
    e_coupling_v_per_m: jnp.ndarray | float,
    e_rf_v_per_m: jnp.ndarray | float,
    system: LadderSystemJax,
    temperature_k: jnp.ndarray | float,
    mass_kg: jnp.ndarray | float,
    *,
    n_velocity_points: int = 65,
) -> jnp.ndarray:
    """Single-atom Doppler-averaged ladder susceptibility, the
    differentiable counterpart to
    :func:`cliffordclock.integrator.rydberg_cell_response.doppler_averaged_susceptibility`.
    Same fixed Gauss-Hermite quadrature the reference function uses:
    `n_velocity_points` sets a STATIC array shape, fixed at trace time,
    so this function is `jax.jit`-clean at any fixed `n_velocity_points`.
    The sum over that fixed array runs via `jnp.sum`; the reference
    function sums the same fixed-length quadrature arrays through a plain
    Python `for` loop.

    `delta_p` is a 1-D array of probe detunings (rad/s, angular, matching
    the reference module's own convention throughout); `delta_c`,
    `delta_rf`, `temperature_k`, `mass_kg`, and the three field arguments
    are scalars. `temperature_k` and `mass_kg` are differentiable: the
    quadrature NODES are fixed constants (computed by
    :func:`_gauss_hermite_nodes_weights`, plain `numpy`, no
    `temperature_k` dependence), but the velocities the nodes are scaled
    to (`sqrt(2)*sigma_v*node`, `sigma_v = sqrt(kB*T/m)`) depend on
    `temperature_k`/`mass_kg` through ordinary `jax.numpy` arithmetic, so
    `jax.grad` with respect to either flows through that scaling the same
    way it flows through any other closed-form input.
    """
    nodes, weights = _gauss_hermite_nodes_weights(n_velocity_points)
    nodes_j = jnp.asarray(nodes)
    weights_j = jnp.asarray(weights)

    sigma_v = jnp.sqrt(BOLTZMANN_K * temperature_k / mass_kg)
    velocities = jnp.sqrt(2.0) * sigma_v * nodes_j

    k_p = 2.0 * jnp.pi / system.wavelength_probe_m
    k_c = 2.0 * jnp.pi / system.wavelength_coupling_m

    delta_p = jnp.asarray(delta_p)
    shifted_delta_p = delta_p[None, :] - k_p * velocities[:, None]
    shifted_delta_c = delta_c + k_c * velocities[:, None]

    chi = ladder_susceptibility_jax(
        shifted_delta_p,
        shifted_delta_c,
        delta_rf,
        e_probe_v_per_m,
        e_coupling_v_per_m,
        e_rf_v_per_m,
        system,
    )
    return jnp.sum(weights_j[:, None] * chi, axis=0)


# ---------------------------------------------------------------------------
# Section B: quadratic Stark shift and per-atom composition (E43/E44 port)
# ---------------------------------------------------------------------------


def rydberg_quadratic_stark_shift_hz_jax(
    alpha0_au: jnp.ndarray, field_v_per_m: jnp.ndarray
) -> jnp.ndarray:
    """Quadratic Stark shift, `Delta_f = -(1/2) * alpha0 * E^2 / h`
    (Yerokhin et al. 2016 Eq. 5, the same formula and sign convention as
    :func:`cliffordclock.integrator.rydberg_cell_response.rydberg_quadratic_stark_shift_hz`),
    in Hz. A degree-2 polynomial in `field_v_per_m`: `jax.grad` of this
    function is exact and unconditioned for any finite input, with no
    branch or clamp anywhere in it.

    **No validity guard inside this function**, unlike the reference
    function (which raises
    :class:`~cliffordclock.integrator.rydberg_cell_response.RydbergStarkValidityError`
    above a guarded fraction of the Inglis-Teller field). A `jax.jit`
    traced function cannot raise on a traced value's concrete magnitude
    (matching this package's established "validate outside the jitted
    core" discipline, stated explicitly in
    `cliffordclock.integrator.lattice_light_shift_jax`'s own module
    docstring). A caller who needs the guard enforced calls
    `cliffordclock.integrator.rydberg_cell_response.rydberg_quadratic_stark_shift_hz`
    (or its exported
    `inglis_teller_field_v_per_m`/`STARK_VALIDITY_MARGIN`) eagerly, in
    plain Python, before ever tracing this function; every caller in this
    package that generates synthetic data or sets optimizer bounds with
    this function does so (`benchmarks/run_rydberg_field_reconstruction.py`).
    """
    alpha0_si = alpha0_au * ALPHA_AU_TO_SI
    shift_j = -0.5 * alpha0_si * field_v_per_m**2
    return shift_j / PLANCK_H


def compose_inhomogeneous_eit_spectrum_jax(
    delta_p: jnp.ndarray,
    atom_field_magnitudes_v_per_m: jnp.ndarray,
    atom_weights: jnp.ndarray,
    alpha0_au: jnp.ndarray,
    system: LadderSystemJax,
    *,
    delta_c: jnp.ndarray | float = 0.0,
    delta_rf: jnp.ndarray | float = 0.0,
    e_probe_v_per_m: jnp.ndarray | float = 1.0,
    e_coupling_v_per_m: jnp.ndarray | float = 1.0,
    e_rf_v_per_m: jnp.ndarray | float = 0.0,
    temperature_k: jnp.ndarray | float = 320.0,
    mass_kg: jnp.ndarray | float = RB85_MASS_KG,
    n_velocity_points: int = 33,
) -> jnp.ndarray:
    """Compose one observed line profile from many atoms, each shifted by
    its own local field's quadratic Stark shift on the Rydberg level. The
    differentiable, fixed-shape counterpart to
    :func:`cliffordclock.integrator.rydberg_cell_response.compose_inhomogeneous_eit_spectrum`,
    with the same per-atom mechanism: `delta_c_atom = delta_c +
    2*pi*Delta_f_atom` (:func:`rydberg_quadratic_stark_shift_hz_jax`), the
    composed spectrum a weight-normalized sum of each atom's Doppler-
    averaged susceptibility at its own shifted `delta_c_atom`.

    **Numerical route, and why it differs from the reference's.** The
    reference function branches on whether every atom shares the same
    field (`if np.all(fields == fields[0])`), a data-dependent Python
    branch `jax.jit` cannot trace, and its own docstring explains why:
    floating-point addition is not associative, so summing `N` identical
    terms is not bit-identical to evaluating one term directly, and the
    reference's own C5 kill-tests need that bit-identical guarantee. This
    function always takes the general weighted-sum path, for every input
    including a uniform field: `jax.jit` needs one fixed computational
    graph. This module's own C1 agreement check
    (`tests/test_rydberg_cell_response_jax.py`) validates this function
    against the reference's general path at ordinary floating-point
    tolerance; the reference's own byte-exact structural checks are a
    separate, stricter guarantee that module's own docstring documents
    for its own general-vs-uniform-path branch specifically.

    This function evaluates every `(atom, velocity)` pair in one batched
    `(n_atoms, n_velocity_points, len(delta_p))` array via `jax.numpy`
    broadcasting. The reference module evaluates the same pairs through a
    Python `for` loop over atoms (itself wrapping a `for` loop over
    velocities): both loops have a STATIC length fixed at trace time
    (`atom_field_magnitudes_v_per_m.shape[0]` and `n_velocity_points`),
    so either a Python loop or direct broadcasting would be
    `jax.jit`-clean. Broadcasting is used here for two reasons: it
    compiles to a single vectorized XLA computation, where a Python loop
    would compile to `n_atoms` unrolled copies of the same graph, and
    :func:`rb85_field_reconstruction_forward_model_jax`'s own memory-bound
    check (`tests/test_rydberg_cell_response_jax.py::TestMemoryBound`)
    measured this route comfortably inside its bound at the atom counts
    the field-reconstruction demonstrator uses.

    `alpha0_au` and every keyword argument other than the atom arrays are
    differentiable scalars. `n_star` (the reference function's fourth
    positional argument, needed there only to call the reference's own
    guarded Stark-shift function) is not a parameter here:
    :func:`rydberg_quadratic_stark_shift_hz_jax` carries no validity
    guard (see that function's own docstring), so nothing in this
    function's call graph needs it.

    Parameters
    ----------
    delta_p : jax.Array, shape (n_delta,)
        Probe detuning grid, rad/s.
    atom_field_magnitudes_v_per_m : jax.Array, shape (n_atoms,)
        Each atom's local DC field magnitude, V/m.
    atom_weights : jax.Array, shape (n_atoms,)
        Relative population weights; normalized internally (need not
        already sum to `1`).
    alpha0_au : jax.Array
        Scalar polarizability, atomic units (a0^3).
    system : LadderSystemJax

    Returns
    -------
    jax.Array, shape (n_delta,), complex128
    """
    weights = atom_weights / jnp.sum(atom_weights)
    fields = jnp.asarray(atom_field_magnitudes_v_per_m)
    delta_p = jnp.asarray(delta_p)

    shift_hz = rydberg_quadratic_stark_shift_hz_jax(alpha0_au, fields)
    atom_delta_c = delta_c + 2.0 * jnp.pi * shift_hz  # (n_atoms,)

    nodes, gh_weights = _gauss_hermite_nodes_weights(n_velocity_points)
    nodes_j = jnp.asarray(nodes)
    gh_weights_j = jnp.asarray(gh_weights)
    sigma_v = jnp.sqrt(BOLTZMANN_K * temperature_k / mass_kg)
    velocities = jnp.sqrt(2.0) * sigma_v * nodes_j  # (n_velocity_points,)

    k_p = 2.0 * jnp.pi / system.wavelength_probe_m
    k_c = 2.0 * jnp.pi / system.wavelength_coupling_m

    # Broadcasting to (n_atoms, n_velocity_points, n_delta): the atom axis
    # carries each atom's own Stark-shifted delta_c, the velocity axis
    # carries the shared Doppler shift, the delta axis the probe grid.
    shifted_delta_p = delta_p[None, None, :] - k_p * velocities[None, :, None]
    shifted_delta_c = atom_delta_c[:, None, None] + k_c * velocities[None, :, None]

    chi = ladder_susceptibility_jax(
        shifted_delta_p,
        shifted_delta_c,
        delta_rf,
        e_probe_v_per_m,
        e_coupling_v_per_m,
        e_rf_v_per_m,
        system,
    )  # (n_atoms, n_velocity_points, n_delta)

    doppler_avg = jnp.sum(gh_weights_j[None, :, None] * chi, axis=1)  # (n_atoms, n_delta)
    return jnp.sum(weights[:, None] * doppler_avg, axis=0)  # (n_delta,)


# ---------------------------------------------------------------------------
# Section C: differentiable cell field model (new for WP41, not a port)
# ---------------------------------------------------------------------------


def cell_field_magnitude_v_per_m_jax(
    positions_m: jnp.ndarray,
    e_uniform_v_per_m: jnp.ndarray,
    gradient_v_per_m_per_m: jnp.ndarray,
    patch_amplitude_v_per_m: jnp.ndarray,
    patch_position_m: jnp.ndarray,
    patch_softening_m: float,
) -> jnp.ndarray:
    """A three-parameter, differentiable DC field model over a vapor
    cell: a uniform background, a linear gradient along the cell axis
    (`z`), and one localized wall-patch bump. This is the
    field-reconstruction demonstrator's own forward field model, built
    for WP41 as this module's own construction: every other function in
    this module is a direct port of a published formula, and this
    function's own design choices are stated in full below.

    **Relationship to the Phase A patch model.** The reference module's
    own wall-patch demonstrator
    (:class:`cliffordclock.integrator.rydberg_cell_response.WallPatch`,
    :func:`~cliffordclock.integrator.rydberg_cell_response.patch_field_v_per_m`)
    superposes point-charge Coulomb fields (`E ~ q/(4*pi*eps0*r^2)`,
    vector-valued, arbitrarily many patches) matching Patrick et al.
    2025's own photoionized-surface-charge phenomenology (arXiv:2502.07018).
    This function reuses that model's SHAPE, a single, positive,
    localized bump that decays with the squared distance from one fixed
    point on the cell wall, in place of its exact functional form, for
    two stated reasons. First, this is a SCALAR field-magnitude model
    (the quantity :func:`rydberg_quadratic_stark_shift_hz_jax` actually
    consumes): a single-parameter "one wall-patch amplitude" (this work
    package's own instruction) needs a scalar bump, so the reference
    model's vector composition carries no benefit here. Second, the exact
    Coulomb form's `1/r^2` singularity at `r=0` needs a branch to stay
    `jax.grad`-safe (the reference module's own `if r_mag < 1e-9:
    continue` is a data-dependent Python branch, valid for the
    reference's plain `numpy` code, that a `jax.jit` trace cannot follow;
    even a traceable version of that branch would still evaluate the
    discarded branch's own gradient, which still diverges at `r=0`).
    This function instead adds a fixed softening length
    `patch_softening_m` to the squared distance before dividing (a
    Plummer-style smooth regularization, common in point-source field
    models for this reason): the field equals `patch_amplitude_v_per_m`
    at the patch location itself, decays with the true inverse-square
    shape once `distance >> patch_softening_m`, and its gradient stays
    finite and continuous everywhere, the patch location included.
    `patch_amplitude_v_per_m` is a field amplitude (V/m) directly, a
    physically equivalent reparameterization of the reference model's own
    charge parameter, appropriate for a demonstrator whose recovered
    quantity is "how strong is this patch's field": it carries no
    `eps0`/`4*pi` unit bookkeeping through the fit.

    The `e_uniform_v_per_m + gradient_v_per_m_per_m * z` background term
    is a first-order Taylor expansion of a smoothly varying DC field
    along the cell's long axis, the simplest inhomogeneity shape beyond a
    uniform field and the natural complement to one localized patch.

    Parameters
    ----------
    positions_m : jax.Array, shape (n_atoms, 3)
        Atom positions, meters. Fixed input: only the three amplitude
        parameters below are fit in the field-reconstruction
        demonstrator.
    e_uniform_v_per_m, gradient_v_per_m_per_m, patch_amplitude_v_per_m : jax.Array
        Scalars. The three differentiable fit parameters.
    patch_position_m : jax.Array, shape (3,)
        Fixed wall location the patch bump is centered on, held constant
        throughout the fit.
    patch_softening_m : float
        Fixed softening length, held constant throughout the fit. Static:
        this function is called with a plain Python float here throughout
        this module, so it carries no differentiability requirement of
        its own.

    Returns
    -------
    jax.Array, shape (n_atoms,)
        Field magnitude at each atom's position, V/m. Always positive
        for the parameter ranges this module's own demonstrator uses
        (`benchmarks/run_rydberg_field_reconstruction.py` keeps the
        background term comfortably above the maximum patch contribution
        so the total never crosses zero); nothing in this function
        enforces positivity, since the downstream Stark shift is `~ E^2`
        and is insensitive to the field's sign.
    """
    positions_m = jnp.asarray(positions_m)
    z = positions_m[:, 2]
    background = e_uniform_v_per_m + gradient_v_per_m_per_m * z

    patch_position_m = jnp.asarray(patch_position_m)
    delta = positions_m - patch_position_m[None, :]
    r_sq = jnp.sum(delta * delta, axis=-1)
    softening_sq = patch_softening_m**2
    patch_term = patch_amplitude_v_per_m * softening_sq / (r_sq + softening_sq)

    return background + patch_term


def rb85_field_reconstruction_forward_model_jax(
    delta_p: jnp.ndarray,
    positions_m: jnp.ndarray,
    atom_weights: jnp.ndarray,
    e_uniform_v_per_m: jnp.ndarray,
    gradient_v_per_m_per_m: jnp.ndarray,
    patch_amplitude_v_per_m: jnp.ndarray,
    patch_position_m: jnp.ndarray,
    patch_softening_m: float,
    alpha0_au: jnp.ndarray,
    system: LadderSystemJax,
    *,
    delta_c: jnp.ndarray | float = 0.0,
    delta_rf: jnp.ndarray | float = 0.0,
    e_probe_v_per_m: jnp.ndarray | float = 1.0,
    e_coupling_v_per_m: jnp.ndarray | float = 1.0,
    e_rf_v_per_m: jnp.ndarray | float = 0.0,
    temperature_k: jnp.ndarray | float = 320.0,
    mass_kg: jnp.ndarray | float = RB85_MASS_KG,
    n_velocity_points: int = 33,
) -> jnp.ndarray:
    """The field-reconstruction demonstrator's full forward model:
    :func:`cell_field_magnitude_v_per_m_jax` (the three fit parameters,
    `e_uniform_v_per_m`/`gradient_v_per_m_per_m`/`patch_amplitude_v_per_m`)
    feeding :func:`compose_inhomogeneous_eit_spectrum_jax`, with the
    imaginary part of the composed susceptibility returned as the
    observable: the standard atomic-physics convention that probe
    absorption is proportional to `Im(chi)`, the real, measurable
    quantity a probe-transmission spectrum reports (Holloway et al.
    2014's own Eq. 1, `epsilon = epsilon_0*(1+chi)`, an absorptive medium
    for `Im(chi) != 0`).

    `jax.grad` of a scalar loss built on this function's output, with
    respect to `e_uniform_v_per_m`/`gradient_v_per_m_per_m`/
    `patch_amplitude_v_per_m`, is the gradient
    `benchmarks/run_rydberg_field_reconstruction.py` supplies to
    `scipy.optimize.minimize` for the field-reconstruction fit.

    Returns
    -------
    jax.Array, shape (n_delta,), float64
        `Im(chi)` at every point of the `delta_p` grid.
    """
    fields = cell_field_magnitude_v_per_m_jax(
        positions_m,
        e_uniform_v_per_m,
        gradient_v_per_m_per_m,
        patch_amplitude_v_per_m,
        patch_position_m,
        patch_softening_m,
    )
    spectrum = compose_inhomogeneous_eit_spectrum_jax(
        delta_p,
        fields,
        atom_weights,
        alpha0_au,
        system,
        delta_c=delta_c,
        delta_rf=delta_rf,
        e_probe_v_per_m=e_probe_v_per_m,
        e_coupling_v_per_m=e_coupling_v_per_m,
        e_rf_v_per_m=e_rf_v_per_m,
        temperature_k=temperature_k,
        mass_kg=mass_kg,
        n_velocity_points=n_velocity_points,
    )
    return jnp.imag(spectrum)
