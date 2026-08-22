# SPDX-License-Identifier: AGPL-3.0-or-later
"""Interaction bivector construction (CONVENTIONS.md E14a, E16, E18, E21).

All functions here are pure, batched over arbitrary leading (``...``) axes,
and jit/vmap/grad-safe (no data-dependent Python control flow on traced
values). Positions/fields/gradients are SI (m, V/m, V/m^2) at these
function boundaries; everything returned is dimensionless (Compton units),
per CONVENTIONS.md section 10.

WP3 scope note: this module treats its ``delta_e``/``grad_delta_e``
arguments as already being the *perturbation* field ``δE`` of E11 (i.e. the
residual after subtracting the analytical baseline ``E_0``) -- the
baseline/residual decomposition itself is WP2 scope
(``cliffordclock.fields.decompose``), upstream of this integrator. This
matches CONVENTIONS.md section 9's V1/V2 validation cases, which are
stated directly in terms of ``δE``.

WP7 scope note (CONVENTIONS.md E14b, Sprint 2): :func:`pivot_perturbation_stark`
below implements the *physical* quadratic DC-Stark coupling, alongside
(not replacing) the E14a linear validation coupling above -- the E14a
functions are untouched by WP7. See :func:`pivot_perturbation_stark`'s
docstring for its field-decomposition argument convention.

WP16 scope note (CONVENTIONS.md E16/E18/E21 instantiated for E14b, rotor-
Stark unification): :func:`spin_connection_stark`,
:func:`scalar_rate_perturbation_stark`, and :func:`build_omega_stark`
below instantiate the pivot-general E15-E18 chain (already stated in
CONVENTIONS.md for a general ``P(r)``) with the E14b Stark pivot instead
of the E14a linear pivot -- mirroring :func:`spin_connection`,
:func:`scalar_rate_perturbation`, and :func:`build_omega` structurally
(same assembly, same E10 precision discipline) with a different pivot
source. Purely additive: every E14a function above is untouched (see the
WP16 builder report for the reviewer diff). These functions take the
*total* field ``E(r)`` (not an E11 baseline/perturbation split), matching
how :func:`pivot_perturbation_stark` is actually invoked by the pipeline
today (``cliffordclock.pipeline._make_stark_rate_fn``: no pipeline-level
E11 split exists for the E14b coupling) -- E14b's ``|E|^2`` needs only the
total field, never a separate baseline/residual pair.

WP21 scope note (CONVENTIONS.md E34/E35, ion-clock electric-quadrupole
shift, Tier 2): :func:`quadrupole_pivot_perturbation` computes the
per-point scalar pivot term ``(P-1)_Q`` (E34/E35) from the LOCAL field
gradient tensor (E13, already available everywhere the Stark functions
above are called -- no new field capability needed, dossier section 2)
for a FIXED ``(J, m_J, quantization axis)``. Composed additively into
``(P-1)`` (E35, mirroring E33's BBR composition) at
``cliffordclock.pipeline``'s WP21 call sites; unlike BBR (spatially
uniform, so its spin-connection gradient is EXACTLY zero) the quadrupole
term's own spatial variation through ``G(r)`` is not carried into the
spin connection here -- CONVENTIONS.md E35's "spin-connection scope
limit" note explains why (third-derivative field data the smoother does
not expose) and bounds the omission. :func:`quadrupole_three_orientation_average`
implements the exact cancellation identity (E35 A2) as a standing test
primitive (not a pipeline-time-savings shortcut -- the pipeline's
``averaging_mode="three_orientation"`` composes the identically-zero
result directly, since the cancellation is EXACT, not merely averaged to
zero over many samples).

WP22 scope note (CONVENTIONS.md E36, gravitational-redshift pivot term):
:func:`grav_pivot_perturbation` computes the per-position scalar pivot
term ``(P-1)_grav`` (E36) from a height coordinate (:func:`height_along_axis`
projects a position onto a configured "up" direction). Threading mirrors
:func:`bbr_pivot_perturbation`'s keyword-only composition pattern exactly
(a new ``grav_pivot_perturbation`` parameter on
:func:`pivot_perturbation_stark`/:func:`spin_connection_stark`/
:func:`scalar_rate_perturbation_stark`/:func:`build_omega_stark`, default
``0.0``) -- but, like :func:`quadrupole_pivot_perturbation`, the VALUE
threaded in is generally per-position (it varies with the caller's height,
not spatially uniform like BBR's single-radiation-temperature scalar).
Per the project's G9 theory sign-off record (A2),
the term is a pure metric clock-rate effect (depends only on position
in the potential, not on any electromagnetic field) and reaches the rotor
through the *scalar* pivot only: it shifts :func:`spin_connection_stark`'s
`P` denominator exactly like BBR/quadrupole (never that function's
numerator/gradient term) -- see that function's WP22 docstring note for
why this is provably inconsequential here (not merely bounded, unlike
quadrupole's third-derivative limitation).

WP20 scope note (CONVENTIONS.md E32/E33, blackbody-radiation shift):
:func:`bbr_pivot_perturbation` computes the uniform-T BBR scalar pivot
term ``(P-1)_BBR`` (E32); :func:`bbr_pivot_uncertainty` propagates its
registry coefficient uncertainties (and, optionally, a radiation-
temperature uncertainty) into a reported fractional uncertainty (G7
sign-off A4#2-3). Per E33 ("independent scalar perturbations compose
additively in (P-1)"), the BBR term is composed into the pivot by a new
keyword-only ``bbr_pivot_perturbation`` parameter threaded through
:func:`stark_pivot_terms`'s callers (:func:`pivot_perturbation_stark`,
:func:`spin_connection_stark`, :func:`scalar_rate_perturbation_stark`,
:func:`build_omega_stark`) -- defaulting to ``0.0`` so every existing
call site (and every shipped example, which has no ``environment``
config section) is completely unaffected. This is a genuine composition,
not a bolt-on: because BBR is spatially uniform in this MVP
(``environment.radiation_temperature_K`` is a single scalar, not a
T(r) map), its gradient is exactly zero, so it contributes to the
*numerator* `P-1` everywhere `pivot_perturbation_stark` is evaluated but
never to :func:`spin_connection_stark`'s *gradient* term
(``d(P-1)/dr_k`` is unaffected -- only the `P` denominator there shifts
by the same additive amount, at the O(1)-precision the denominator
already tolerates). See ``cliffordclock.pipeline``'s WP20 module-docstring
note for how the pipeline resolves ``bbr_pivot_perturbation`` once per
run and threads it to every evaluation-mode accumulator (fast_path,
secular, classical direct batched/streaming, rotor worldline).

WP29 Tier 1 scope note (CONVENTIONS.md E37, multi-surface thermal
environment): :func:`bbr_environment_pivot_perturbation` generalizes
:func:`bbr_pivot_perturbation` from a single radiation temperature to `N`
surfaces, each with a solid-angle weight, a temperature, and an optional
temperature uncertainty; at most one surface may also carry an
emissivity, representing PTB's reflective-enclosure-plus-apertures
topology (:func:`_bbr_effective_weights`, exact for PTB's own one-
aperture case). E32's static and dynamic terms are evaluated against the
environment's per-moment weighted sums (:func:`_bbr_weighted_moments`)
instead of a single `T`.
The resolved scalar composes into the pivot through the exact same
keyword-only ``bbr_pivot_perturbation`` parameter as the uniform-T case
(the composition point does not change; only how the pipeline resolves the
scalar upstream of it does), so it is still spatially uniform within the
atom cloud (every atom sees the same enclosure, no per-atom solid-angle
map in this tier) and its spin-connection contribution is exactly zero for
the same reason WP20's is. See ``cliffordclock.pipeline``'s WP29
module-docstring note for
``environment.radiation_environment``/:class:`RadiationEnvironmentConfig`.

WP30 scope note (CONVENTIONS.md E38, quantum-motional second-order-Doppler
pivot term): :func:`motional_pivot_perturbation` computes ``(P-1)_motional
= -<v^2>/(2c^2)`` from a set of trapped-motion normal modes
(:class:`MotionalMode`, each carrying a mode ORDINARY frequency ``f_i`` in
hertz, converted internally via ``omega_i = 2*pi*f_i``, and a mean
vibrational occupation ``n_bar_i``) plus an optional measured excess-
micromotion rms velocity, evaluated against the species' registry mass
(`~cliffordclock.ensemble.species.Species.mass_kg`, never hand-typed);
:func:`motional_pivot_uncertainty` propagates the per-mode/EMM input
uncertainties. Threading mirrors :func:`bbr_pivot_perturbation`'s
keyword-only composition pattern exactly (a new
``motional_pivot_perturbation`` parameter on
:func:`pivot_perturbation_stark`/:func:`spin_connection_stark`/
:func:`scalar_rate_perturbation_stark`/:func:`build_omega_stark`, default
``0.0``): this project's motional state is one state per run, spatially
uniform across the atom cloud exactly like BBR's single radiation
temperature (a per-atom motional map is future work, CONVENTIONS.md E38's
composition note), so it shifts :func:`spin_connection_stark`'s `P`
denominator only, never its numerator/gradient term. **No double-counting
with the existing kinematic second-order Doppler carried by E15/E21's**
``sqrt(1-v^2/c^2)`` **factor:** every call site this parameter composes at
evaluates STATIC (``v = 0`` exactly) lattice/lattice_extended quadrature
nodes, so the classical kinematic contribution is identically zero there,
which is precisely why the quantum-motional term is otherwise missing and
precisely why adding it cannot double-count (CONVENTIONS.md E38's central
argument, also why ``cliffordclock.pipeline`` rejects any
`environment.motional_state` paired with `ensemble.regime: classical`,
where velocities are real and sampled, at config-parse time).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp

from cliffordclock.cl13 import IDX_E01, IDX_E02, IDX_E03, IDX_E12
from cliffordclock.constants import (
    ELECTRON_MASS,
    HBAR,
    LAMBDA_BAR_COMPTON,
    PLANCK_H,
    SPEED_OF_LIGHT,
)
from cliffordclock.ensemble.species import (
    BBR_REFERENCE_TEMPERATURE_K,
    EA0_SQUARED_SI,
    BbrCoefficients,
    Species,
    StarkCoefficients,
)

#: m_e c^2 (J), the E14a pivot denominator.
_M_E_C2 = ELECTRON_MASS * SPEED_OF_LIGHT**2


def pivot_perturbation(delta_e: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """``P(r) − 1``, computed directly (E14a perturbation, E10 precision discipline).

    ``P(r) − 1 = δE(r)·μ / (m_e c^2)``. This is *not* the same computation
    as ``pivot(delta_e, mu) - 1.0``: E10 requires never accumulating an
    absolute (near-unity) quantity when only the perturbation is needed,
    because forming ``1.0 + x`` for ``|x| ≲ 2e-16`` (the float64 relative
    epsilon) silently rounds `x` away entirely -- exactly the target
    magnitude of this project's fractional shifts (~1e-18). Every
    downstream consumer that only needs the perturbation (E21's
    :func:`scalar_rate_perturbation`, most crucially) must go through this
    function, not through :func:`pivot` followed by a subtraction.

    Parameters
    ----------
    delta_e : jax.Array, shape (..., 3)
        Perturbation field ``δE(r)`` (E11), V/m.
    mu : jax.Array, shape (3,)
        Explicit effective dipole moment (E14a), C·m. Broadcasts against
        any leading batch axes of `delta_e`.

    Returns
    -------
    jax.Array, shape (...,)
        ``P(r) − 1``, dimensionless.
    """
    delta_e = jnp.asarray(delta_e, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    return jnp.sum(delta_e * mu, axis=-1) / _M_E_C2


def pivot(delta_e: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """Scalar pivot ``P(r)`` (E14a, MVP linear validation coupling).

    ``P(r) = 1 + δE(r)·μ / (m_e c^2)``. This is the literal E14a
    quantity, useful where `P` itself (not the perturbation `P − 1`) is
    needed at only O(1) relative precision -- e.g. as the denominator in
    :func:`spin_connection`'s ``∂_k P / P``. Callers that need the
    perturbation `P − 1` at full (sub-1e-16-absolute) precision must use
    :func:`pivot_perturbation` instead (see its docstring: this function's
    ``1 + x`` sum is exactly the E10 failure mode for tiny `x`).

    Parameters
    ----------
    delta_e : jax.Array, shape (..., 3)
        Perturbation field ``δE(r)`` (E11), V/m.
    mu : jax.Array, shape (3,)
        Explicit effective dipole moment (E14a), C·m.

    Returns
    -------
    jax.Array, shape (...,)
        ``P(r)``, dimensionless.
    """
    return 1.0 + pivot_perturbation(delta_e, mu)


def spin_connection(
    delta_e: jnp.ndarray, grad_delta_e: jnp.ndarray, mu: jnp.ndarray
) -> jnp.ndarray:
    """Spin connection boost components ``ω_{0k}(r) = ∂_k ln P(r)`` (E16).

    Computed analytically (chain rule through E14a) rather than by
    autodiff, since this module receives `grad_delta_e` as a pre-supplied
    tensor (not a differentiable closure) -- see module docstring.

    Parameters
    ----------
    delta_e : jax.Array, shape (..., 3)
        Perturbation field ``δE(r)`` (E11), V/m.
    grad_delta_e : jax.Array, shape (..., 3, 3)
        Gradient tensor ``grad_delta_e[..., i, j] = ∂_i δE_j`` (E13), V/m^2.
    mu : jax.Array, shape (3,)
        Effective dipole moment (E14a), C·m.

    Returns
    -------
    jax.Array, shape (..., 3)
        ``ω_{0k}(r)`` for spatial ``k`` = array index + 1 (i.e. index 0-2
        correspond to physical ``k = 1, 2, 3``), units 1/m.
    """
    delta_e = jnp.asarray(delta_e, dtype=jnp.float64)
    grad_delta_e = jnp.asarray(grad_delta_e, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    p = pivot(delta_e, mu)
    # d(P)/dr_k = (1/m_e c^2) sum_j mu_j * grad_delta_e[..., k, j] (E14a chain rule).
    d_p_dr = jnp.einsum("...kj,j->...k", grad_delta_e, mu) / _M_E_C2
    return d_p_dr / p[..., None]


def scalar_rate_perturbation(delta_e: jnp.ndarray, v: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """Instantaneous fractional rate perturbation ``δω̃(r, v)`` (E21).

    ``δω̃(r, v) = P(r) √(1 − v²/c²) − 1``, algebraically expanded as
    ``δω̃ = (γ_v⁻¹ − 1) + (P − 1)·γ_v⁻¹`` (with ``γ_v⁻¹ = √(1 − v²/c²)``)
    and evaluated in that expanded form rather than literally as
    ``P·γ_v⁻¹ − 1`` (E10 precision discipline): using the literal form
    would compute `P` via :func:`pivot`'s ``1 + (P − 1)`` sum, which
    rounds the perturbation `P − 1` away entirely once it drops below the
    float64 relative epsilon (~2e-16) -- exactly this project's target
    1e-18 shift regime (see :func:`pivot_perturbation`). The expanded
    form instead uses :func:`pivot_perturbation` directly, so `P − 1`
    never passes through a ``1 + x`` rounding step. At ``v = 0`` (the
    static-node regime, E21 note), ``γ_v⁻¹ = 1`` exactly, the kinematic
    term vanishes exactly, and this reduces to exactly ``P − 1`` with no
    precision loss whatsoever.

    The same E10 discipline applies to the kinematic term itself for
    ``v ≠ 0``: it is computed as ``−x / (1 + γ_v⁻¹)`` with ``x = v²/c²``
    (algebraically identical to ``γ_v⁻¹ − 1``, by rationalizing
    ``(γ_v⁻¹ − 1)(γ_v⁻¹ + 1) = −x``) rather than literally as
    ``γ_v⁻¹ − 1``, which catastrophically cancels to exactly ``0.0`` for
    any ``v/c ≲ 1e−8`` -- squarely inside the realistic cold-atom regime
    (e.g. Sr @ 1 µK has ``v/c ≈ 3.3e−11``) -- and would silently drop the
    second-order Doppler shift entirely.

    Parameters
    ----------
    delta_e : jax.Array, shape (..., 3)
        Perturbation field ``δE(r)`` (E11), V/m.
    v : jax.Array, shape (..., 3)
        Velocity, m/s.
    mu : jax.Array, shape (3,)
        Effective dipole moment (E14a), C·m.

    Returns
    -------
    jax.Array, shape (...,)
        ``δω̃``, dimensionless.
    """
    p_minus_1 = pivot_perturbation(delta_e, mu)
    v = jnp.asarray(v, dtype=jnp.float64)
    v2 = jnp.sum(v * v, axis=-1)
    x = v2 / SPEED_OF_LIGHT**2
    gamma_inv = jnp.sqrt(1.0 - x)
    # kinematic = gamma_inv - 1.0, rewritten to avoid catastrophic
    # cancellation (E10 precision discipline): for x = v^2/c^2 below
    # ~1e-8 (the realistic cold-atom regime, e.g. Sr @ 1 uK has
    # v/c ~ 3.3e-11), 1.0 - x rounds to 1.0 in float64 and
    # sqrt(1.0 - x) - 1.0 evaluates to exactly 0.0, silently zeroing the
    # second-order Doppler shift. The algebraically identical
    # -x / (1 + sqrt(1 - x)) has no such cancellation: rationalizing,
    # (sqrt(1-x) - 1)(sqrt(1-x) + 1) = (1-x) - 1 = -x, so
    # sqrt(1-x) - 1 = -x / (sqrt(1-x) + 1) exactly, and this form stays
    # accurate all the way down to x ~ 0.
    kinematic = -x / (1.0 + gamma_inv)
    return kinematic + p_minus_1 * gamma_inv


def build_omega(
    delta_e: jnp.ndarray, grad_delta_e: jnp.ndarray, v: jnp.ndarray, mu: jnp.ndarray
) -> jnp.ndarray:
    """Interaction bivector ``Ω(r)`` (E18, G0 item 3 resolved reading).

    ``Ω(r) = (P(r) γ_v⁻¹ − 1) B̂_C + ω_boost(r)`` with:

    - ``B̂_C = e_1 ∧ e_2`` (the fixed internal-circulation plane, E18
      convention), coefficient ``P γ_v⁻¹ − 1`` = the E21 scalar rate
      perturbation.
    - ``ω_boost = Σ_k (v^k/c) ω̃_{0k}(r) (e_k ∧ e_0)``, with
      ``ω̃_{0k} = λ̄_C ∂_k ln P`` (E18). Since orthogonal vectors'
      wedge equals their geometric product, ``e_k ∧ e_0 = e_k e_0 =
      −e_0 e_k``, i.e. ``−1`` times the E2 basis blade at index
      ``IDX_E0k``.
    - No additional ``ω_rot`` terms (MVP: no magnetic/real-rotation
      effects modeled, per E18's resolved note).

    Parameters
    ----------
    delta_e : jax.Array, shape (..., 3)
        Perturbation field ``δE(r)`` (E11), V/m.
    grad_delta_e : jax.Array, shape (..., 3, 3)
        Gradient tensor, ``[..., i, j] = ∂_i δE_j`` (E13), V/m^2.
    v : jax.Array, shape (..., 3)
        Velocity, m/s.
    mu : jax.Array, shape (3,)
        Effective dipole moment (E14a), C·m.

    Returns
    -------
    jax.Array, shape (..., 16)
        ``Ω(r)``, a bivector-only multivector (all non-bivector
        components exactly zero), dimensionless (rate in Compton units,
        E17).
    """
    delta_e = jnp.asarray(delta_e, dtype=jnp.float64)
    grad_delta_e = jnp.asarray(grad_delta_e, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)

    rotation_coeff = scalar_rate_perturbation(delta_e, v, mu)  # (...,)
    omega_0k = spin_connection(delta_e, grad_delta_e, mu)  # (..., 3), 1/m
    omega_tilde_0k = LAMBDA_BAR_COMPTON * omega_0k  # (..., 3), dimensionless (E18)
    boost_coeff = (v / SPEED_OF_LIGHT) * omega_tilde_0k  # (..., 3), coefficient of e_k ^ e_0

    batch_shape = jnp.broadcast_shapes(rotation_coeff.shape, boost_coeff.shape[:-1])
    omega = jnp.zeros(batch_shape + (16,), dtype=jnp.float64)
    omega = omega.at[..., IDX_E12].set(jnp.broadcast_to(rotation_coeff, batch_shape))
    omega = omega.at[..., IDX_E01].set(jnp.broadcast_to(-boost_coeff[..., 0], batch_shape))
    omega = omega.at[..., IDX_E02].set(jnp.broadcast_to(-boost_coeff[..., 1], batch_shape))
    omega = omega.at[..., IDX_E03].set(jnp.broadcast_to(-boost_coeff[..., 2], batch_shape))
    return omega


def stark_pivot_terms(
    e0: jnp.ndarray,
    delta_e: jnp.ndarray,
    species_or_coeffs: Species | StarkCoefficients,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Term-by-term decomposition of the E14b quadratic DC-Stark pivot.

    ``P(r) − 1 = Δν(r)/ν₀ = −(Δα/2)·|E(r)|² / (h ν₀)``, equivalently
    ``P − 1 = k_S |E|² / ν₀`` with ``k_S = −Δα/(2h)`` (E14b's "equivalent
    per-species input"). ``k_S`` and ``ν₀`` are resolved from
    `species_or_coeffs` via its ``resolve_stark_coefficient_hz_per_v2_m2``
    method (`cliffordclock.ensemble.species.Species` or
    `cliffordclock.ensemble.species.StarkCoefficients`); a species with no
    DC-Stark data (e.g. ``Al27+`` in the Sprint 1/WP7 registry) raises a
    clear `ValueError` there.

    Field-decomposition argument convention (E10/E11, deviation from the
    WP7 spec sketch -- see the WP7 builder report's AMBIGUITY note): this
    function takes the E11 baseline/perturbation split ``e0``
    (``E_0(r)``) and `delta_e` (``δE(r)``) as *separate* arguments, rather
    than a single combined total-field array. E14b's ``|E|²`` is expanded
    and evaluated **term-by-term** --
    ``|E|² = |E₀|² + 2E₀·δE + |δE|²`` -- with each term individually
    scaled by the (tiny) prefactor ``k_S/ν₀`` *before* summing. Writing
    ``k = Δα/(2hν₀)`` (so ``k_S/ν₀ = −k``), the three returned terms are:

    - ``baseline = −k|E₀|²`` -- the "no gradient" DC-Stark shift.
    - ``cross = −k·2(E₀·δE)`` -- the precision-critical, gradient-driven
      piece this decomposition exists to protect (E10): forming a
      combined total field ``E₀+δE`` first (then squaring) is
      numerically safe *in this module's realistic parameter regime* but
      is not a provably safe pattern in general -- if ``δE`` is small
      enough relative to ``E₀`` (e.g. ``|δE|/|E₀| ≲ 1e-16``), summing the
      *vectors* before squaring can lose `delta_e` entirely in the
      addition, before the physics prefactor is ever applied. Computing
      the three dot products (``E₀·E₀``, ``E₀·δE``, ``δE·δE``)
      independently and scaling each by the prefactor before combining
      keeps `cross` a clean, independently-rounded floating-point
      product no matter how small ``δE`` is relative to ``E₀`` -- it
      never depends on a vector addition retaining a small operand.
    - ``quadratic = −k|δE|²`` -- second order in `delta_e`, negligible at
      realistic magnitudes but included for exactness.

    Exposing these three terms individually (rather than only their sum,
    as :func:`pivot_perturbation_stark` returns) serves two purposes: it
    is what makes the term-by-term precision discipline above testable at
    all -- ``tests/test_stark_pivot.py`` (WP7 test contract item 3)
    checks `cross` in isolation against a 50-digit `decimal` reference,
    since reconstructing it by subtracting two *summed* `P − 1` scalars
    would reintroduce exactly the catastrophic-cancellation pattern this
    decomposition exists to avoid -- and it directly serves a future
    fast-path/analytics need to consume the gradient-driven `cross` term
    on its own (e.g. for a gradient-only systematic-budget line item)
    without re-deriving it from the summed pivot.

    Parameters
    ----------
    e0 : jax.Array, shape (..., 3)
        Baseline field ``E₀(r)`` (E11), V/m.
    delta_e : jax.Array, shape (..., 3)
        Perturbation field ``δE(r)`` (E11), V/m. Broadcasts against `e0`.
    species_or_coeffs : Species | StarkCoefficients
        Species (from `cliffordclock.ensemble.species.get_species`) or an
        explicit `StarkCoefficients` override providing ``k_S``/``Δα``
        and ``ν₀``. Not batched; the same coefficients apply to every
        element of `e0`/`delta_e`'s leading axes.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        ``(baseline, cross, quadratic)``, each shape (...,), dimensionless.
        ``pivot_perturbation_stark`` is exactly their sum.

    Raises
    ------
    ValueError
        If `species_or_coeffs` has no resolvable DC-Stark coefficient
        (propagated from ``resolve_stark_coefficient_hz_per_v2_m2``).
    """
    e0 = jnp.asarray(e0, dtype=jnp.float64)
    delta_e = jnp.asarray(delta_e, dtype=jnp.float64)
    k_s = species_or_coeffs.resolve_stark_coefficient_hz_per_v2_m2()
    nu_0 = species_or_coeffs.clock_frequency_hz
    prefactor = k_s / nu_0  # (V/m)^-2, dimensionless once multiplied by |E|^2 (E14b)

    e0_sq = jnp.sum(e0 * e0, axis=-1)
    cross_dot = jnp.sum(e0 * delta_e, axis=-1)
    delta_e_sq = jnp.sum(delta_e * delta_e, axis=-1)

    baseline = prefactor * e0_sq
    cross = prefactor * (2.0 * cross_dot)
    quadratic = prefactor * delta_e_sq
    return baseline, cross, quadratic


def pivot_perturbation_stark(
    e0: jnp.ndarray,
    delta_e: jnp.ndarray,
    species_or_coeffs: Species | StarkCoefficients,
    *,
    bbr_pivot_perturbation: jnp.ndarray | float = 0.0,
    quadrupole_pivot_perturbation: jnp.ndarray | float = 0.0,
    grav_pivot_perturbation: jnp.ndarray | float = 0.0,
    motional_pivot_perturbation: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """``P(r) − 1`` under the physical quadratic DC-Stark coupling (E14b).

    Exactly the sum of :func:`stark_pivot_terms`'s three decomposed
    terms (single source of truth for the E14b formula -- see that
    function's docstring for the term-by-term precision discipline this
    relies on, E10) plus `bbr_pivot_perturbation` (E33 additive scalar
    composition, WP20/CONVENTIONS.md E32-E33) plus
    `quadrupole_pivot_perturbation` (E35, WP21 Tier 2) plus
    `grav_pivot_perturbation` (E36, WP22) plus `motional_pivot_perturbation`
    (E38, WP30): ``P(r) − 1 = (P−1)_stark + (P−1)_BBR + (P−1)_Q +
    (P−1)_grav + (P−1)_motional``. All four default to ``0.0`` (an
    exact no-op: ``x + 0.0 == x`` in IEEE 754 for any finite `x`), so
    every pre-WP20/WP21/WP22/WP30 call site is unaffected -- see
    :func:`bbr_pivot_perturbation` for computing the BBR term from a
    registry species and radiation temperature, the module-level
    :func:`quadrupole_pivot_perturbation` (WP21, note the same name --
    this parameter is that function's already-evaluated *result*,
    computed by the caller from the local field-gradient tensor and
    passed in here, exactly mirroring how `bbr_pivot_perturbation` the
    parameter relates to `bbr_pivot_perturbation` the function) for
    computing the quadrupole term itself, the module-level
    :func:`grav_pivot_perturbation` (WP22, same "parameter vs. function"
    naming pattern -- computed by the caller from each point's height via
    :func:`height_along_axis` and passed in here) for the gravitational
    term, and the module-level :func:`motional_pivot_perturbation` (WP30,
    the same "parameter vs. function" naming pattern, computed by the
    caller once per run from the configured motional modes and passed in
    here, exactly like `bbr_pivot_perturbation`) for the quantum-motional
    second-order-Doppler term.

    ``tests/test_stark_pivot.py`` (WP7 test contract item 3) verifies the
    `cross` term of :func:`stark_pivot_terms` against a 50-digit `decimal`
    reference at ``|E₀| = 1e5`` V/m with a `delta_e` sized to produce a
    ~1e-19-level `cross` contribution, and separately verifies that the
    naive combined-square evaluation loses several digits of that same
    term by comparison -- the numerical justification for evaluating
    term-by-term here.

    Parameters
    ----------
    e0 : jax.Array, shape (..., 3)
        Baseline field ``E₀(r)`` (E11), V/m.
    delta_e : jax.Array, shape (..., 3)
        Perturbation field ``δE(r)`` (E11), V/m. Broadcasts against `e0`.
    species_or_coeffs : Species | StarkCoefficients
        Species (from `cliffordclock.ensemble.species.get_species`) or an
        explicit `StarkCoefficients` override providing ``k_S``/``Δα``
        and ``ν₀``. Not batched; the same coefficients apply to every
        element of `e0`/`delta_e`'s leading axes.
    bbr_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_BBR`` (E32, WP20), a single scalar (this MVP's BBR term
        is spatially uniform, so it never varies across `e0`/`delta_e`'s
        batch axes) added into the returned `P − 1` (E33 composition).
    quadrupole_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_Q`` (E34/E35, WP21), the already-evaluated quadrupole
        pivot term (module-level :func:`quadrupole_pivot_perturbation`,
        called by the caller against the LOCAL field-gradient tensor --
        unlike `bbr_pivot_perturbation` this generally DOES vary across
        `e0`/`delta_e`'s batch axes, since the quadrupole shift depends on
        position through the gradient) added into the returned `P − 1`
        (E35 composition).
    grav_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_grav`` (E36, WP22), the already-evaluated gravitational
        pivot term (module-level :func:`grav_pivot_perturbation`, called
        by the caller against each point's height -- like
        `quadrupole_pivot_perturbation`, this generally DOES vary across
        `e0`/`delta_e`'s batch axes) added into the returned `P − 1`
        (E36 composition, additive per E33's pattern; G9 sign-off A2:
        no cross term with the Stark/BBR terms at this project's working
        precision).
    motional_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_motional`` (E38, WP30), the already-evaluated quantum-
        motional second-order-Doppler pivot term (module-level
        :func:`motional_pivot_perturbation`), a single scalar (this
        project's motional state is one state per run, spatially uniform
        across `e0`/`delta_e`'s batch axes, exactly like
        `bbr_pivot_perturbation`) added into the returned `P − 1` (E38
        composition, E33's additive pattern).

    Returns
    -------
    jax.Array, shape (...,)
        ``P(r) − 1``, dimensionless.

    Raises
    ------
    ValueError
        If `species_or_coeffs` has no resolvable DC-Stark coefficient
        (propagated from ``resolve_stark_coefficient_hz_per_v2_m2``).
    """
    baseline, cross, quadratic = stark_pivot_terms(e0, delta_e, species_or_coeffs)
    return (
        baseline
        + cross
        + quadratic
        + bbr_pivot_perturbation
        + quadrupole_pivot_perturbation
        + grav_pivot_perturbation
        + motional_pivot_perturbation
    )


# ---------------------------------------------------------------------------
# WP16: rotor construction under the E14b Stark pivot (CONVENTIONS.md
# E15-E18 instantiated for the quadratic coupling, additive -- see module
# docstring's WP16 scope note). Structurally mirrors spin_connection /
# scalar_rate_perturbation / build_omega above; only the pivot source
# differs.
# ---------------------------------------------------------------------------


def spin_connection_stark(
    e_total: jnp.ndarray,
    grad_e_total: jnp.ndarray,
    species_or_coeffs: Species | StarkCoefficients,
    *,
    bbr_pivot_perturbation: jnp.ndarray | float = 0.0,
    quadrupole_pivot_perturbation: jnp.ndarray | float = 0.0,
    grav_pivot_perturbation: jnp.ndarray | float = 0.0,
    motional_pivot_perturbation: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Spin connection boost components ``ω_{0k}(r) = ∂_k ln P(r)`` (E16) under E14b.

    WP20 note: `bbr_pivot_perturbation` (E32/E33) shifts the ``P`` used in
    this function's denominator (``P = 1 + (P−1)_stark + (P−1)_BBR``) but
    contributes nothing to the *numerator* ``d(P−1)/dr_k`` -- correct
    because this MVP's BBR term is spatially uniform (a single scalar
    radiation temperature, not a T(r) map), so ``∇ln P_BBR = 0`` exactly
    (CONVENTIONS.md E33: "the spin connection generalizes by linearity:
    ∂_k ln P picks up each term's gradient (zero for uniform T ...)").

    WP21 note (CONVENTIONS.md E35's "spin-connection scope limit"):
    `quadrupole_pivot_perturbation` likewise shifts only this function's
    `P` denominator, never the numerator -- but for a DIFFERENT reason
    than BBR's: the quadrupole term is NOT spatially uniform (it varies
    through the local gradient tensor `grad_e_total`), so
    `∇ln P_Q` is not exactly zero in principle. Its exact numerator
    contribution would require the THIRD spatial derivative of the field
    (the gradient OF `grad_e_total`), which this project's field-smoother
    (E12) does not expose -- a deliberate, documented, bounded scope
    limit (CONVENTIONS.md E35), not an oversight.

    WP22 note (CONVENTIONS.md E36's "rotor carries it through the scalar
    pivot only"): `grav_pivot_perturbation` likewise shifts only this
    function's `P` denominator, never the numerator -- for a THIRD reason,
    stronger than either of the above: `(P-1)_grav`'s true gradient along
    `height_along_axis`'s "up" direction is `g/c^2` exactly (no unmodeled
    higher derivative, unlike quadrupole's third-derivative gap), but the
    only caller of this omitted numerator term is `ω_boost`
    (:func:`build_omega_stark`), whose own coefficient is `(v/c) *
    lambda_bar_C * omega_0k` -- multiplied by velocity `v`. Every call
    site this project ships evaluates static (`v = 0`) lattice/
    lattice-extended nodes (CONVENTIONS.md section 15; the G9 sign-off's
    "rotor carries it through the scalar pivot only" ruling), so
    `ω_boost`'s gravitational contribution is not merely small but
    IDENTICALLY zero for every configuration this project supports --
    provably inconsequential, not just bounded, so the omission is
    threaded through the same `P`-denominator-only pattern as
    `bbr_pivot_perturbation`/`quadrupole_pivot_perturbation` above for API
    consistency rather than out of numerical necessity.

    WP30 note (CONVENTIONS.md E38's composition note): `motional_pivot_perturbation`
    shifts only this function's `P` denominator, never the numerator --
    for the SAME reason as `bbr_pivot_perturbation`: this project's
    motional state is one state per run, spatially uniform across the
    atom cloud exactly like BBR's single radiation temperature (a per-atom
    motional map is future work, CONVENTIONS.md E38's composition note),
    so ``∇ln P_motional = 0`` exactly.

    E14b's pivot is ``P(r) − 1 = prefactor·|E(r)|²`` with ``prefactor =
    k_S/ν₀ = −Δα/(2hν₀)`` (:func:`stark_pivot_terms`). Differentiating,
    ``∂_k(P−1) = prefactor·∂_k|E|² = 2·prefactor·(E·∂_kE)`` -- equivalently
    ``−(Δα/hν₀)·(E·∂_kE)`` since ``2·prefactor = −Δα/(hν₀)`` (the WP16
    spec's stated form). Computed directly as a product/contraction of `E`
    and `grad_E`, never by subtracting two summed pivots (E10 precision
    discipline: the same reasoning as E14a's :func:`spin_connection`, and
    as :func:`stark_pivot_terms`'s `cross` term).

    ``E13`` index convention: ``grad_e_total[..., i, j] = ∂_i E_j``, so
    ``∂_k E_j`` for fixed spatial index ``k`` is ``grad_e_total[..., k, j]``
    -- ``(E·∂_kE) = Σ_j E_j · grad_e_total[..., k, j]``, contracted below via
    ``einsum("...kj,...j->...k", grad_e_total, e_total)``, exactly mirroring
    :func:`spin_connection`'s ``einsum("...kj,j->...k", grad_delta_e, mu)``
    contraction pattern (there over the fixed dipole `mu`; here over the
    field itself, since E14b's pivot is quadratic in `E` rather than linear).

    `P` in the ``ω_{0k} = ∂_k(P−1)/P`` denominator is only needed at O(1)
    relative precision (it multiplies an already-tiny numerator), so it is
    formed via the ordinary ``1 + (P−1)`` sum -- the same pattern
    :func:`spin_connection` uses for E14a's `P` (:func:`pivot`) -- not
    :func:`pivot_perturbation_stark`'s precision-protected numerator path,
    which only matters for the *perturbation* `P − 1` itself.

    Parameters
    ----------
    e_total : jax.Array, shape (..., 3)
        Total field ``E(r)`` (V/m) -- see module docstring's WP16 scope
        note on why this is the total field, not an E11 split.
    grad_e_total : jax.Array, shape (..., 3, 3)
        Gradient tensor, ``[..., i, j] = ∂_i E_j`` (E13), V/m^2.
    species_or_coeffs : Species | StarkCoefficients
        Resolved DC-Stark coefficients (`k_S`/`Δα`, `ν₀`); see
        :func:`stark_pivot_terms`. Not batched.
    bbr_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_BBR`` (E32, WP20); see :func:`pivot_perturbation_stark`.
        Affects only the ``P`` denominator here (see the note above).
    quadrupole_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_Q`` (E34/E35, WP21); see :func:`pivot_perturbation_stark`.
        Affects only the ``P`` denominator here (see the WP21 note above).
    grav_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_grav`` (E36, WP22); see :func:`pivot_perturbation_stark`.
        Affects only the ``P`` denominator here (see the WP22 note above).
    motional_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_motional`` (E38, WP30); see :func:`pivot_perturbation_stark`.
        Affects only the ``P`` denominator here (see the WP30 note above).

    Returns
    -------
    jax.Array, shape (..., 3)
        ``ω_{0k}(r)`` for spatial ``k`` = array index + 1, units 1/m.

    Raises
    ------
    ValueError
        If `species_or_coeffs` has no resolvable DC-Stark coefficient
        (propagated from ``resolve_stark_coefficient_hz_per_v2_m2``).
    """
    e_total = jnp.asarray(e_total, dtype=jnp.float64)
    grad_e_total = jnp.asarray(grad_e_total, dtype=jnp.float64)
    zeros = jnp.zeros_like(e_total)
    p = 1.0 + pivot_perturbation_stark(
        e_total,
        zeros,
        species_or_coeffs,
        bbr_pivot_perturbation=bbr_pivot_perturbation,
        quadrupole_pivot_perturbation=quadrupole_pivot_perturbation,
        grav_pivot_perturbation=grav_pivot_perturbation,
        motional_pivot_perturbation=motional_pivot_perturbation,
    )

    k_s = species_or_coeffs.resolve_stark_coefficient_hz_per_v2_m2()
    nu_0 = species_or_coeffs.clock_frequency_hz
    prefactor = k_s / nu_0  # (V/m)^-2, dimensionless once multiplied by |E|^2 (E14b)

    # d(P-1)/dr_k = 2 * prefactor * (E . d_k E) = 2 * prefactor *
    # sum_j E_j * grad_e_total[..., k, j] (E13 convention: grad[i,j] = d_i E_j).
    # BBR (WP20) contributes nothing here: uniform T => grad ln P_BBR = 0
    # (see this function's WP20 docstring note) -- only `p` above shifted.
    d_p_minus_1_dr = 2.0 * prefactor * jnp.einsum("...kj,...j->...k", grad_e_total, e_total)
    return d_p_minus_1_dr / p[..., None]


def scalar_rate_perturbation_stark(
    e_total: jnp.ndarray,
    v: jnp.ndarray,
    species_or_coeffs: Species | StarkCoefficients,
    *,
    bbr_pivot_perturbation: jnp.ndarray | float = 0.0,
    quadrupole_pivot_perturbation: jnp.ndarray | float = 0.0,
    grav_pivot_perturbation: jnp.ndarray | float = 0.0,
    motional_pivot_perturbation: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Instantaneous fractional rate perturbation ``δω̃(r, v)`` (E21) under E14b.

    ``δω̃(r, v) = P(r)√(1 − v²/c²) − 1``, evaluated with exactly the same
    E10-safe algebraic expansion as E14a's :func:`scalar_rate_perturbation`
    (``δω̃ = (γ_v⁻¹ − 1) + (P − 1)·γ_v⁻¹``, kinematic term rewritten as
    ``−x/(1+γ_v⁻¹)`` to avoid catastrophic cancellation at realistic
    cold-atom ``v/c``) -- see that function's docstring for the full
    precision-discipline rationale, which applies unchanged here; only
    `P − 1`'s source differs (:func:`pivot_perturbation_stark`, E14b,
    instead of :func:`pivot_perturbation`, E14a).

    Deliberately *not* shared code with
    ``cliffordclock.pipeline._make_stark_rate_fn``'s inline `rate_fn` body
    (which reimplements the identical rewrite for the same reason that
    function's own docstring gives: an independently-accurate,
    independently-testable evaluator at each call site, not a
    pipeline-private helper reused across module boundaries).

    Parameters
    ----------
    e_total : jax.Array, shape (..., 3)
        Total field ``E(r)`` (V/m).
    v : jax.Array, shape (..., 3)
        Velocity, m/s.
    species_or_coeffs : Species | StarkCoefficients
        Resolved DC-Stark coefficients; see :func:`stark_pivot_terms`.
    bbr_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_BBR`` (E32, WP20); see :func:`pivot_perturbation_stark`.
        Composed into `p_minus_1` before the `gamma_inv` weighting, so the
        BBR contribution picks up the same (utterly negligible at
        realistic cold-atom `v/c`) kinematic weighting as the Stark term
        -- exact per E21/E33, not an approximation.
    quadrupole_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_Q`` (E34/E35, WP21); see :func:`pivot_perturbation_stark`.
        Composed into `p_minus_1` the same way as `bbr_pivot_perturbation`
        above (E35).
    grav_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_grav`` (E36, WP22); see :func:`pivot_perturbation_stark`.
        Composed into `p_minus_1` the same way as `bbr_pivot_perturbation`
        above (E36) -- this is the rotor's "scalar pivot" route the G9
        sign-off refers to (the coefficient of the `B̂_C` rotation plane in
        :func:`build_omega_stark`, below).
    motional_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_motional`` (E38, WP30); see :func:`pivot_perturbation_stark`.
        Composed into `p_minus_1` the same way as `bbr_pivot_perturbation`
        above (E38): this term is a *separate* physical mechanism from the
        `kinematic`/`gamma_inv` term computed below. CONVENTIONS.md E38's
        no-double-counting argument is that `gamma_inv` here is evaluated
        at the CLASSICAL trajectory velocity `v` (identically zero at
        every static lattice/lattice_extended node this parameter is ever
        composed for), while `motional_pivot_perturbation` supplies the
        QUANTUM motional-state expectation `-<v^2>/(2c^2)` that a `v=0`
        classical velocity cannot see; the two never double-count the same
        physics.

    Returns
    -------
    jax.Array, shape (...,)
        ``δω̃``, dimensionless.
    """
    e_total = jnp.asarray(e_total, dtype=jnp.float64)
    p_minus_1 = pivot_perturbation_stark(
        e_total,
        jnp.zeros_like(e_total),
        species_or_coeffs,
        bbr_pivot_perturbation=bbr_pivot_perturbation,
        quadrupole_pivot_perturbation=quadrupole_pivot_perturbation,
        grav_pivot_perturbation=grav_pivot_perturbation,
        motional_pivot_perturbation=motional_pivot_perturbation,
    )
    v = jnp.asarray(v, dtype=jnp.float64)
    v2 = jnp.sum(v * v, axis=-1)
    x = v2 / SPEED_OF_LIGHT**2
    gamma_inv = jnp.sqrt(1.0 - x)
    # Same E10-safe rewrite as scalar_rate_perturbation (E21): see that
    # function's docstring for the catastrophic-cancellation rationale.
    kinematic = -x / (1.0 + gamma_inv)
    return kinematic + p_minus_1 * gamma_inv


def build_omega_stark(
    e_total: jnp.ndarray,
    grad_e_total: jnp.ndarray,
    species_or_coeffs: Species | StarkCoefficients,
    v: jnp.ndarray,
    *,
    bbr_pivot_perturbation: jnp.ndarray | float = 0.0,
    quadrupole_pivot_perturbation: jnp.ndarray | float = 0.0,
    grav_pivot_perturbation: jnp.ndarray | float = 0.0,
    motional_pivot_perturbation: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Interaction bivector ``Ω(r)`` (E18) under the E14b quadratic DC-Stark pivot.

    WP20 (E32/E33): `bbr_pivot_perturbation` composes the uniform-T BBR
    scalar pivot term into ``Ω`` consistently -- it reaches the rotation
    coefficient (via :func:`scalar_rate_perturbation_stark`) exactly as
    the Stark term does, and reaches :func:`spin_connection_stark`'s `P`
    denominator only (its gradient contribution is exactly zero for
    uniform T, WP20 design item 3 / CONVENTIONS.md E33).

    ``Ω(r) = (P(r) γ_v⁻¹ − 1) B̂_C + ω_boost(r)`` -- exactly
    :func:`build_omega`'s E18 structure (same ``B̂_C = e_1∧e_2`` plane,
    same ``ω_boost = Σ_k (v^k/c) ω̃_{0k}(r) (e_k∧e_0)`` assembly into the
    16-component bivector, same ``ω̃_{0k} = λ̄_C ∂_k ln P`` non-
    dimensionalization), with the pivot source swapped for E14b's
    quadratic Stark coupling (:func:`scalar_rate_perturbation_stark`,
    :func:`spin_connection_stark`) instead of E14a's linear coupling.
    Per WP16's scope, this is the "builder's choice" instantiation of the
    pivot-general E15-E18 chain CONVENTIONS.md already states for a
    general ``P(r)`` -- no new physics, no new equation numbers.

    Parameters
    ----------
    e_total : jax.Array, shape (..., 3)
        Total field ``E(r)`` (V/m) -- see module docstring's WP16 scope
        note.
    grad_e_total : jax.Array, shape (..., 3, 3)
        Gradient tensor, ``[..., i, j] = ∂_i E_j`` (E13), V/m^2.
    species_or_coeffs : Species | StarkCoefficients
        Resolved DC-Stark coefficients; see :func:`stark_pivot_terms`.
    v : jax.Array, shape (..., 3)
        Velocity, m/s.
    bbr_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_BBR`` (E32, WP20); see :func:`pivot_perturbation_stark`.
    quadrupole_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_Q`` (E34/E35, WP21); see :func:`pivot_perturbation_stark`.
        Reaches the rotation coefficient exactly as the Stark/BBR terms
        do, and :func:`spin_connection_stark`'s `P` denominator only (its
        exact numerator/gradient contribution is out of scope, CONVENTIONS.md
        E35's spin-connection scope-limit note).
    grav_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_grav`` (E36, WP22); see :func:`pivot_perturbation_stark`.
        Reaches the rotation coefficient (the `B̂_C` plane) exactly as the
        Stark/BBR/quadrupole terms do -- the "rotor carries it through the
        scalar pivot only" the G9 sign-off states -- and
        :func:`spin_connection_stark`'s `P` denominator only, never
        `ω_boost`'s numerator (see that function's WP22 docstring note:
        provably zero-effect for every `v = 0` static-node call site this
        project ships, not merely bounded).
    motional_pivot_perturbation : jax.Array | float, default 0.0
        ``(P−1)_motional`` (E38, WP30); see :func:`pivot_perturbation_stark`.
        Reaches the rotation coefficient (the `B̂_C` plane) exactly as the
        Stark/BBR/quadrupole/grav terms do; see
        :func:`scalar_rate_perturbation_stark`'s WP30 docstring note for
        why this never double-counts the classical kinematic term already
        carried by `v`, and :func:`spin_connection_stark`'s `P`
        denominator only (CONVENTIONS.md E38's composition note: spatially
        uniform, exactly like `bbr_pivot_perturbation`).

    Returns
    -------
    jax.Array, shape (..., 16)
        ``Ω(r)``, a bivector-only multivector (all non-bivector
        components exactly zero), dimensionless (rate in Compton units,
        E17).

    Raises
    ------
    ValueError
        If `species_or_coeffs` has no resolvable DC-Stark coefficient
        (propagated from ``resolve_stark_coefficient_hz_per_v2_m2``).
    """
    e_total = jnp.asarray(e_total, dtype=jnp.float64)
    grad_e_total = jnp.asarray(grad_e_total, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)

    rotation_coeff = scalar_rate_perturbation_stark(
        e_total,
        v,
        species_or_coeffs,
        bbr_pivot_perturbation=bbr_pivot_perturbation,
        quadrupole_pivot_perturbation=quadrupole_pivot_perturbation,
        grav_pivot_perturbation=grav_pivot_perturbation,
        motional_pivot_perturbation=motional_pivot_perturbation,
    )  # (...,)
    omega_0k = spin_connection_stark(
        e_total,
        grad_e_total,
        species_or_coeffs,
        bbr_pivot_perturbation=bbr_pivot_perturbation,
        quadrupole_pivot_perturbation=quadrupole_pivot_perturbation,
        grav_pivot_perturbation=grav_pivot_perturbation,
        motional_pivot_perturbation=motional_pivot_perturbation,
    )  # (..., 3), 1/m
    omega_tilde_0k = LAMBDA_BAR_COMPTON * omega_0k  # (..., 3), dimensionless (E18)
    boost_coeff = (v / SPEED_OF_LIGHT) * omega_tilde_0k  # (..., 3), coefficient of e_k ^ e_0

    batch_shape = jnp.broadcast_shapes(rotation_coeff.shape, boost_coeff.shape[:-1])
    omega = jnp.zeros(batch_shape + (16,), dtype=jnp.float64)
    omega = omega.at[..., IDX_E12].set(jnp.broadcast_to(rotation_coeff, batch_shape))
    omega = omega.at[..., IDX_E01].set(jnp.broadcast_to(-boost_coeff[..., 0], batch_shape))
    omega = omega.at[..., IDX_E02].set(jnp.broadcast_to(-boost_coeff[..., 1], batch_shape))
    omega = omega.at[..., IDX_E03].set(jnp.broadcast_to(-boost_coeff[..., 2], batch_shape))
    return omega


# ---------------------------------------------------------------------------
# WP20/WP29 Tier 1: blackbody-radiation shift pivot term (CONVENTIONS.md
# E32/E33/E37). Pure Python float arithmetic (not jax-batched): the BBR term
# is a per-run config-level scalar (one resolved environment, one species),
# not a per-atom/per-node batched quantity, mirroring the plain-float style
# of `Species.resolve_stark_coefficient_hz_per_v2_m2`/
# `cliffordclock.pipeline._stark_coupling_provenance_note`'s `k_s`, not the
# jax.Array style of the batched pivot functions above.
#
# WP29 Tier 1 scope note (CONVENTIONS.md E37, multi-surface thermal
# environment): the single-temperature functions below
# (`bbr_pivot_perturbation`/`bbr_pivot_uncertainty`, E32) are implemented as
# the single-surface case of the general multi-surface functions further
# down this section (`bbr_environment_pivot_perturbation`/
# `bbr_environment_pivot_uncertainty`, E37): both call the same private
# per-moment coefficient evaluation (`_bbr_weighted_moments`), so a uniform
# environment reduces to the E32 scalar path bit for bit, an exact match,
# not just close numerical agreement (`tests/test_bbr_environment.py`'s
# reduction test).
# ---------------------------------------------------------------------------

#: Absolute tolerance for `RadiationSurface.weight` sums (CONVENTIONS.md
#: E37): the input solid-angle fractions must sum to 1 within this margin,
#: enforced both by `cliffordclock.pipeline._parse_radiation_environment`
#: (parse time, before a species is even resolved) and by
#: `_bbr_validate_environment` below (evaluation time, for direct callers of
#: `bbr_environment_pivot_perturbation`/`bbr_environment_pivot_uncertainty`
#: that bypass the pipeline entirely).
BBR_ENVIRONMENT_WEIGHT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RadiationSurface:
    """One surface of a multi-surface BBR thermal environment (CONVENTIONS.md E37).

    Attributes
    ----------
    name : str
        Label for this surface, used only in error messages and pipeline
        report notes (not a registry key).
    weight : float
        Effective solid-angle fraction `w_i = Omega_i/(4*pi)` this surface
        subtends at the atoms, before any emissivity correction. The full
        set of `weight` values across an environment's surfaces must sum to
        1 within `BBR_ENVIRONMENT_WEIGHT_TOLERANCE`; this is an input the
        lab supplies (from geometry, an FEA model, or a ray-traced exchange-
        factor calculation), not a value this tier computes from CAD.
    temperature_k : float
        This surface's radiation temperature, kelvin. Must lie in the
        species' `BbrCoefficients.validity_min_k`/`validity_max_k` window
        (50-350 K for the registry's current entries), exactly like E32's
        single `temperature_k`.
    temperature_uncertainty_k : float
        1-sigma uncertainty on `temperature_k`, kelvin. Default `0.0` (no
        uncertainty on this surface's temperature); must be `>= 0`.
    emissivity : float or None
        Interior emissivity `epsilon`, in `(0, 1]`, of the reflective
        enclosure this surface represents (CONVENTIONS.md E37's PTB
        enclosure-and-apertures topology, `_bbr_effective_weights`).
        `None` (the default, and the ordinary case for every other
        surface in an environment): this surface is a direct-view
        aperture/window, its `weight` used as given with no correction.
        At most one surface across an entire environment may set
        `emissivity`; that surface is the enclosure whose reflections
        amplify every other (aperture) surface's effective weight, and
        `_bbr_validate_environment` rejects more than one.
    """

    name: str
    weight: float
    temperature_k: float
    temperature_uncertainty_k: float = 0.0
    emissivity: float | None = None


def _bbr_validate_environment(
    surfaces: Sequence[RadiationSurface], coeffs: BbrCoefficients
) -> None:
    """Raise `ValueError` if `surfaces` violates E37's weight/window/shape invariants.

    Called by every public entry point below
    (`bbr_environment_pivot_perturbation`, `bbr_environment_pivot_uncertainty`,
    `bbr_environment_effective_temperatures`) so a direct caller (bypassing
    `cliffordclock.pipeline`'s own parse-time checks entirely) still gets a
    clear rejection instead of a silently wrong shift.
    """
    if not surfaces:
        raise ValueError("radiation environment must have at least one RadiationSurface")

    total_weight = math.fsum(surface.weight for surface in surfaces)
    if abs(total_weight - 1.0) > BBR_ENVIRONMENT_WEIGHT_TOLERANCE:
        raise ValueError(
            "radiation environment surface weights must sum to 1 (tolerance "
            f"{BBR_ENVIRONMENT_WEIGHT_TOLERANCE:g}); got sum={total_weight!r} across "
            f"{len(surfaces)} surface(s): {[surface.name for surface in surfaces]!r}"
        )

    enclosure_names = [surface.name for surface in surfaces if surface.emissivity is not None]
    if len(enclosure_names) > 1:
        raise ValueError(
            "radiation environment: at most one surface may carry an emissivity "
            "(CONVENTIONS.md E37's enclosure-and-apertures topology: one reflective "
            f"enclosure plus direct-view apertures); got {len(enclosure_names)} surfaces "
            f"with emissivity set: {enclosure_names!r}. Multi-reflector radiosity (more "
            "than one partially-reflective enclosure surface) is out of scope for this "
            "tier, future work."
        )

    for surface in surfaces:
        if not (coeffs.validity_min_k <= surface.temperature_k <= coeffs.validity_max_k):
            raise ValueError(
                f"radiation environment surface {surface.name!r}: temperature_k="
                f"{surface.temperature_k!r} K is outside the validated BBR fit range "
                f"[{coeffs.validity_min_k}, {coeffs.validity_max_k}] K (CONVENTIONS.md "
                "E32/E37): hard rejection, not a silently-extrapolated fit past its "
                "published support."
            )
        if surface.temperature_uncertainty_k < 0.0:
            raise ValueError(
                f"radiation environment surface {surface.name!r}: "
                f"temperature_uncertainty_k={surface.temperature_uncertainty_k!r} must be >= 0"
            )
        if surface.emissivity is not None and not (0.0 < surface.emissivity <= 1.0):
            raise ValueError(
                f"radiation environment surface {surface.name!r}: emissivity="
                f"{surface.emissivity!r} must lie in (0, 1] (CONVENTIONS.md E37's PTB "
                "aperture form)"
            )


def _bbr_effective_weights(surfaces: Sequence[RadiationSurface]) -> list[float]:
    """Per-surface effective solid-angle fraction (CONVENTIONS.md E37's PTB
    enclosure-and-apertures topology).

    Nosske et al. (arXiv:2507.14030, PTB's transportable-clock paper) model
    the atoms as sitting inside a single reflective enclosure of interior
    emissivity `epsilon`, pierced by one or more apertures that leak in a
    different temperature: the enclosure's own reflections give the leaked-
    in radiation more chances to reach the atoms than its raw geometric
    solid angle alone would suggest. Their published closed form, for one
    aperture of raw fraction `w = Omega/4pi`, is `Omega_eff/4pi = 1 /
    [1 + (4pi/Omega - 1) * epsilon]`, equivalently `w_eff = w / (w +
    (1 - w) * epsilon)`.

    `_bbr_validate_environment` guarantees at most one `RadiationSurface`
    carries an `emissivity`; that surface is the enclosure, every other
    surface is a direct-view aperture. Writing `W = sum` of the apertures'
    raw `weight` (their combined raw fraction, jointly forming the single
    lumped aperture PTB's formula treats), each aperture's effective
    weight is `w_i_eff = w_i / (W + (1 - W) * epsilon)`: PTB's own
    single-aperture formula with `w` replaced by the combined `W`, then
    split across the individual apertures in proportion to their own raw-
    weight share of `W`. Summing every `w_i_eff` over the apertures
    reproduces PTB's combined effective fraction exactly, `sum_i w_i_eff =
    W / (W + (1 - W) * epsilon)`; for a single aperture (`W = w_1`) this is
    PTB's own formula unchanged, character for character. The enclosure
    then gets whatever effective weight is left, `1 - sum_i w_i_eff`,
    never a value derived from its own raw `weight`: PTB's derivation is a
    two-temperature mixture (the enclosure and the leaked-in aperture
    temperature), so the two effective weights are complementary by
    construction, not independently renormalized shares of every surface's
    weight (an earlier, incorrect implementation of this function did
    exactly that renormalization, which does not reduce to PTB's formula;
    see `tests/test_bbr_environment.py`'s dedicated kill-test).

    An environment with no `emissivity` set on any surface returns every
    raw `weight` unchanged: for a uniform single-surface environment
    (`weight=1.0`, no emissivity) this returns `[1.0]` exactly.

    Multi-reflector radiosity (more than one partially-reflective
    enclosure surface, each contributing its own reflected share) is out
    of scope for this tier; `_bbr_validate_environment` rejects more than
    one `emissivity`-carrying surface.
    """
    enclosure_indices = [
        index for index, surface in enumerate(surfaces) if surface.emissivity is not None
    ]
    if not enclosure_indices:
        return [surface.weight for surface in surfaces]

    # `_bbr_validate_environment` guarantees exactly one such index.
    enclosure_index = enclosure_indices[0]
    epsilon = surfaces[enclosure_index].emissivity
    assert epsilon is not None  # narrows the type for mypy; guaranteed by the list above

    aperture_weight_total = math.fsum(
        surface.weight for index, surface in enumerate(surfaces) if index != enclosure_index
    )
    denominator = aperture_weight_total + (1.0 - aperture_weight_total) * epsilon

    weights_eff = [0.0] * len(surfaces)
    for index, surface in enumerate(surfaces):
        if index != enclosure_index:
            weights_eff[index] = surface.weight / denominator
    aperture_weight_eff_total = math.fsum(
        weight for index, weight in enumerate(weights_eff) if index != enclosure_index
    )
    weights_eff[enclosure_index] = 1.0 - aperture_weight_eff_total
    return weights_eff


def _bbr_moment_powers(coeffs: BbrCoefficients) -> list[int]:
    """The `(T/T0)` powers E37's per-moment sums need: `4` and `6` always
    (the static term and the dynamic-anchor uncertainty's leading power,
    CONVENTIONS.md section 13's uncertainty note), plus every power
    `coeffs.dyn_coeffs_hz` actually carries (`{6, 8, 10}` for Sr-87,
    `{6, 8}` for Yb-171).
    """
    return sorted({4, 6} | set(coeffs.dyn_coeffs_hz.keys()))


def _bbr_weighted_moments(
    surfaces: Sequence[RadiationSurface], coeffs: BbrCoefficients
) -> dict[int, float]:
    """`M_n = sum_i w_eff_i * (T_i/T0)^n` for every power E37/E32 need (CONVENTIONS.md E37).

    A single surface with `weight=1.0` (and no emissivity, so
    `_bbr_effective_weights` returns `[1.0]` exactly) gives `M_n =
    (T_1/T0)^n` bit for bit: `math.fsum` of a single term returns that term
    unchanged, so this equals E32's `t_ratio**n` exactly, an identical
    value, not an approximation of it. This bit-exactness is what makes
    `bbr_pivot_perturbation`'s
    single-surface reduction (module docstring's WP29 Tier 1 scope note)
    a structural guarantee, not a numerical coincidence.
    """
    weights_eff = _bbr_effective_weights(surfaces)
    t0 = BBR_REFERENCE_TEMPERATURE_K
    return {
        n: math.fsum(
            weight * (surface.temperature_k / t0) ** n
            for weight, surface in zip(weights_eff, surfaces, strict=True)
        )
        for n in _bbr_moment_powers(coeffs)
    }


def _bbr_coefficient_uncertainty_frac(
    coeffs: BbrCoefficients, moments: dict[int, float], nu_0_hz: float
) -> float:
    """Registry coefficient-uncertainty contribution (CONVENTIONS.md section
    13's uncertainty note / G7 sign-off A4#2), generalized from E32's single
    `(T/T0)^n` powers to E37's per-moment sums `M_n`. Shared by
    `bbr_pivot_uncertainty` (single surface, `temperature_uncertainty_k is
    None` branch) and `bbr_environment_pivot_uncertainty`, so both compute
    this term identically.
    """
    sigma_stat_hz = coeffs.nu_stat_300k_uncertainty_hz * moments[4]
    sigma_dyn_hz = coeffs.dyn_anchor_uncertainty_hz * moments[6]
    return math.sqrt(sigma_stat_hz**2 + sigma_dyn_hz**2) / nu_0_hz


def bbr_environment_pivot_perturbation(
    surfaces: Sequence[RadiationSurface], species: Species
) -> float:
    """``(P−1)_BBR`` for a multi-surface thermal environment (CONVENTIONS.md E37).

    Generalizes E32's single-temperature formula by replacing each
    `(T/T0)^n` power with the environment's weighted moment `M_n = sum_i
    w_eff_i * (T_i/T0)^n` (`_bbr_weighted_moments`, `w_eff_i` the
    emissivity-corrected fraction from `_bbr_effective_weights`):

        (P-1)_BBR = [Delta_nu_stat * M_4 + sum_n c_n * M_n] / nu_0

    with `c_n` the same per-species `dyn_coeffs_hz` registry entries E32
    uses. A uniform environment (one `RadiationSurface` with `weight=1.0`)
    reduces to E32's `bbr_pivot_perturbation` bit for bit, not just
    numerically (`_bbr_weighted_moments`'s docstring; the single-surface
    reduction test in `tests/test_bbr_environment.py` pins this).

    Parameters
    ----------
    surfaces : Sequence[RadiationSurface]
        The enclosure's surfaces. Weights must sum to 1 within
        `BBR_ENVIRONMENT_WEIGHT_TOLERANCE`; every `temperature_k` must lie
        in `species`' resolved BBR validity window.
    species : Species
        Species with a resolvable `BbrCoefficients` entry.

    Returns
    -------
    float
        ``(P−1)_BBR``, dimensionless.

    Raises
    ------
    ValueError
        If `species` has no resolvable BBR coefficients, `surfaces` is
        empty, its weights do not sum to 1 within tolerance, or any
        surface's temperature/emissivity/uncertainty is out of range
        (`_bbr_validate_environment`).
    """
    coeffs = species.resolve_bbr_coefficients()
    _bbr_validate_environment(surfaces, coeffs)
    moments = _bbr_weighted_moments(surfaces, coeffs)
    dyn_hz = sum(coeff * moments[n] for n, coeff in coeffs.dyn_coeffs_hz.items())
    delta_nu_hz = coeffs.nu_stat_300k_hz * moments[4] + dyn_hz
    return delta_nu_hz / species.clock_frequency_hz


def bbr_environment_effective_temperatures(
    surfaces: Sequence[RadiationSurface], species: Species
) -> dict[int, float]:
    """Per-moment effective temperatures `T_eff,n = T0 * M_n^(1/n)` (CONVENTIONS.md E37).

    `T_eff,4` is the temperature a single uniform bath would need to match
    this environment's static (`T^4`) moment; `T_eff,6`/`T_eff,8`/`T_eff,10`
    are the equivalent matches for the dynamic term's powers. For a uniform
    environment every `T_eff,n` equals the single shared temperature; for a
    non-uniform one they generally differ; that divergence is exactly the
    mismatch the project's internal BBR thermal-environment dossier
    quantifies against the registry coefficients (crossing `1e-18` at an
    11 K spread, `1e-17` by 35 K, for a 50/50 two-surface split).

    Parameters
    ----------
    surfaces : Sequence[RadiationSurface]
    species : Species
        Species with a resolvable `BbrCoefficients` entry.

    Returns
    -------
    dict[int, float]
        `{n: T_eff,n}` for every power `_bbr_moment_powers` resolves for
        `species` (always includes `4` and `6`, plus every power the
        species' `dyn_coeffs_hz` carries).

    Raises
    ------
    ValueError
        Same conditions as `bbr_environment_pivot_perturbation`.
    """
    coeffs = species.resolve_bbr_coefficients()
    _bbr_validate_environment(surfaces, coeffs)
    moments = _bbr_weighted_moments(surfaces, coeffs)
    t0 = BBR_REFERENCE_TEMPERATURE_K
    return {n: t0 * moment ** (1.0 / n) for n, moment in moments.items()}


def bbr_environment_pivot_uncertainty(
    surfaces: Sequence[RadiationSurface],
    species: Species,
    *,
    correlated: bool = False,
) -> tuple[float, bool]:
    """Propagated fractional uncertainty on `bbr_environment_pivot_perturbation`
    (CONVENTIONS.md E37).

    **Coefficient uncertainty**, always included: identical to E32's
    `bbr_pivot_uncertainty` (`_bbr_coefficient_uncertainty_frac`), with the
    single `(T/T0)^n` powers generalized to the environment's weighted
    moments `M_4`/`M_6`.

    **Per-surface temperature uncertainty.** Writing `a_i = w_eff_i *
    d(Delta_nu_hz)/dT` evaluated at surface `i`'s own `temperature_k` (the
    same polynomial derivative E32's uncertainty note uses, scaled by that
    surface's effective weight), two combination modes are available:

    - `correlated=False` (the default): the surfaces' temperature errors
      are treated as independent and combined in quadrature,
      `sigma_T = sqrt(sum_i (a_i * sigma_{T_i})^2)`.
    - `correlated=True`: the surfaces' temperature errors are treated as
      moving together (a shared calibration-chain error affecting every
      sensor coherently, the linear-pooling motivation Aeppli's 2025 JILA
      thesis gives for its own four correlated temperature estimates, per
      the project's internal BBR thermal-environment dossier part A), so
      the per-surface terms are summed before taking the magnitude,
      `sigma_T = abs(sum_i (a_i * sigma_{T_i}))`. For same-sign partials
      (the ordinary case here: every registry coefficient is negative, so
      every `a_i` is negative) this is an L1 norm against `independent`'s
      L2 norm, so `correlated >= independent` always, strictly greater
      whenever more than one surface carries a nonzero
      `temperature_uncertainty_k`.

    The two contributions combine in quadrature, exactly as E32's
    coefficient/temperature terms do.

    Parameters
    ----------
    surfaces : Sequence[RadiationSurface]
    species : Species
        Species with a resolvable `BbrCoefficients` entry.
    correlated : bool
        Temperature-uncertainty combination mode across surfaces; see
        above. Default `False` (independent, quadrature).

    Returns
    -------
    tuple[float, bool]
        ``(sigma_fractional, temperature_uncertainty_included)``, the
        second element `True` iff at least one surface carries a nonzero
        `temperature_uncertainty_k` (mirrors `bbr_pivot_uncertainty`'s
        `temperature_uncertainty_k is None` -> `False` semantics: an
        environment where every surface has the default `0.0` uncertainty
        propagates no temperature-uncertainty contribution, the same
        "conditional on exact T" case E32 already reports).

    Raises
    ------
    ValueError
        Same conditions as `bbr_environment_pivot_perturbation`.
    """
    coeffs = species.resolve_bbr_coefficients()
    _bbr_validate_environment(surfaces, coeffs)
    nu_0 = species.clock_frequency_hz
    moments = _bbr_weighted_moments(surfaces, coeffs)
    weights_eff = _bbr_effective_weights(surfaces)
    sigma_coeff_frac = _bbr_coefficient_uncertainty_frac(coeffs, moments, nu_0)

    t0 = BBR_REFERENCE_TEMPERATURE_K
    per_surface_terms_hz = []
    for weight, surface in zip(weights_eff, surfaces, strict=True):
        t_ratio = surface.temperature_k / t0
        d_delta_nu_dt_hz_per_k = coeffs.nu_stat_300k_hz * 4.0 * t_ratio**3 / t0 + sum(
            coeff * n * t_ratio ** (n - 1) / t0 for n, coeff in coeffs.dyn_coeffs_hz.items()
        )
        per_surface_terms_hz.append(
            weight * d_delta_nu_dt_hz_per_k * surface.temperature_uncertainty_k
        )

    temperature_uncertainty_included = any(
        surface.temperature_uncertainty_k > 0.0 for surface in surfaces
    )
    sigma_t_hz = (
        abs(math.fsum(per_surface_terms_hz))
        if correlated
        else math.sqrt(math.fsum(term**2 for term in per_surface_terms_hz))
    )
    sigma_t_frac = sigma_t_hz / nu_0
    combined_frac = math.sqrt(sigma_coeff_frac**2 + sigma_t_frac**2)
    return combined_frac, temperature_uncertainty_included


def bbr_pivot_perturbation(temperature_k: float, species: Species) -> float:
    """``(P−1)_BBR`` (CONVENTIONS.md E32, G7-corrected sign).

    ``(P−1)_BBR = [Δν_stat·(T/T₀)⁴ + Δν_dyn(T)] / ν₀``, ``T₀ =
    BBR_REFERENCE_TEMPERATURE_K = 300 K``, with ``Δν_dyn(T) =
    Σ_n dyn_coeffs_hz[n]·(T/T₀)ⁿ`` the registry's per-species dynamic-term
    polynomial (a fit to the exact Planck-weighted integral, Lisdat et al.
    PR Research 3, L042036 (2021) Eq. 6-7, NOT a Taylor series; see
    `cliffordclock.ensemble.species.BbrCoefficients`'s docstring). **No
    leading minus**: the sign lives inside `Δν_stat < 0`, exactly as E14b
    carries it (``P−1 = Δν/ν₀``), the G7 theory sign-off's mandatory
    correction (the project's theory sign-off record (G7), A1) to an earlier
    double-negated draft. Mandatory regression:
    ``bbr_pivot_perturbation(300.0, get_species("Sr87")) < 0`` and
    ``≈ −5.3e-15`` (`tests/test_bbr_pivot.py`).

    WP29 Tier 1 note (CONVENTIONS.md E37): implemented as the single-surface
    case of `bbr_environment_pivot_perturbation`, `weight=1.0` and no
    emissivity, so this function's result is bit-for-bit identical to
    calling that function with one `RadiationSurface` (module section
    docstring; `tests/test_bbr_environment.py`'s reduction test).

    Parameters
    ----------
    temperature_k : float
        Radiation temperature `T`, kelvin (``environment.radiation_temperature_K``).
    species : Species
        Species with a resolvable `BbrCoefficients` entry (`Sr87`/`Yb171`
        in the WP20 registry).

    Returns
    -------
    float
        ``(P−1)_BBR``, dimensionless.

    Raises
    ------
    ValueError
        If `species` has no resolvable BBR coefficients (propagated from
        `Species.resolve_bbr_coefficients`), or `temperature_k` is outside
        `species`' resolved BBR validity window (CONVENTIONS.md E32/E37;
        this was previously enforced only at
        `cliffordclock.pipeline`'s config-parse boundary, now also enforced
        here as a structural consequence of sharing
        `bbr_environment_pivot_perturbation`'s validation).
    """
    return bbr_environment_pivot_perturbation(
        (RadiationSurface(name="uniform", weight=1.0, temperature_k=temperature_k),),
        species,
    )


def bbr_pivot_uncertainty(
    temperature_k: float,
    species: Species,
    temperature_uncertainty_k: float | None = None,
) -> tuple[float, bool]:
    """Propagated fractional uncertainty on `bbr_pivot_perturbation` (G7 sign-off A4#2-3).

    **Coefficient uncertainty (A4#2, always included).** The registry's
    static and dynamic-anchor uncertainties are propagated through the
    same `(T/T₀)` powers as their central values (`nu_stat_300k_hz` scales
    with `Δν_stat`'s ``(T/T₀)⁴``; `dyn_anchor_uncertainty_hz`, the
    dominant, anchor-level uncertainty on the *summed* dynamic term, not a
    per-coefficient covariance, see `BbrCoefficients`'s docstring, is
    scaled by the leading dynamic power ``(T/T₀)⁶``) and combined in
    quadrature (independent error sources: the static term is Middelmann's
    directly-measured Δα-based value, the dynamic term is a separately
    fitted/anchored quantity). At Sr87/300 K this reproduces the G7
    sign-off's cited magnitudes: static ≈1.4e-19, dynamic ≈7.7e-19,
    combined ≈7.8e-19 (the sign-off's "≈8e-19"; corrected from the theory
    brief's original mis-reading of Middelmann's "(6)" as ±6 mHz instead
    of the last-digit ±0.00006 Hz, G7 sign-off A4#2). One deliberate
    simplification for Yb171: `dyn_anchor_uncertainty_hz` is ν_dyn,6's
    ±0.34 mHz alone, not its quadrature sum with ν_dyn,8's ±0.020 mHz,
    the difference is 0.17% of the anchor (~1.2e-21 fractional on the
    clock), far below the 1e-19 floor (WP20 review nit, accepted).

    **This uncertainty is "arithmetic-reproduction fidelity", never
    "BBR accuracy"** (G7 sign-off A4#2c): it is the propagated uncertainty
    of the *registry's published coefficients*, not an independent
    assessment of the underlying physics. Callers presenting this number
    (e.g. `cliffordclock.pipeline`'s report notes) must label it as such.

    **Temperature uncertainty (A4#3, opt-in).** When
    `temperature_uncertainty_k` is given, propagated via the *exact*
    polynomial derivative ``∂Δν/∂T = Δν_stat·4·(T/T₀)³/T₀ +
    Σ_n dyn_coeffs_hz[n]·n·(T/T₀)^(n−1)/T₀`` (not just the leading
    ``4Δν/T`` approximation the sign-off used for its order-of-magnitude
    estimate) and combined in quadrature with the coefficient uncertainty.
    When omitted, the second return value is `False`, callers must then
    emit an explicit "conditional on exact T" note (G7 sign-off A4#3:
    "silent exactness is not defensible at 1e-19" once σ_T exceeds the
    floor, which it does even for JILA-class 4 mK in-vacuum thermometry).

    WP29 Tier 1 note (CONVENTIONS.md E37): the `temperature_uncertainty_k`
    is-not-`None` branch delegates to `bbr_environment_pivot_uncertainty`
    with one `RadiationSurface`, `weight=1.0`, `correlated` unused (a single
    surface has no other surface to correlate with, both modes coincide),
    bit-for-bit identical to this function's own historical formula (the
    single-surface `d(Delta_nu_hz)/dT` derivative equals E32's `d_delta_nu_dt_hz_per_k`
    exactly, and `sqrt(x**2) == abs(x)` for every finite `x` in IEEE 754
    double precision). The `is None` branch is kept as a direct calculation
    (not a delegated call) so it returns `sigma_coeff_frac` itself rather
    than `sqrt(sigma_coeff_frac**2 + 0.0**2)`, an identical value but
    without a redundant round trip through `sqrt`.

    Parameters
    ----------
    temperature_k : float
        Radiation temperature `T`, kelvin.
    species : Species
        Species with a resolvable `BbrCoefficients` entry.
    temperature_uncertainty_k : float or None
        1-sigma uncertainty on `temperature_k`, kelvin. `None`: the
        returned uncertainty excludes any σ_T contribution.

    Returns
    -------
    tuple[float, bool]
        ``(sigma_fractional, temperature_uncertainty_included)``.

    Raises
    ------
    ValueError
        If `species` has no resolvable BBR coefficients.
    """
    if temperature_uncertainty_k is None:
        coeffs = species.resolve_bbr_coefficients()
        surfaces = (RadiationSurface(name="uniform", weight=1.0, temperature_k=temperature_k),)
        _bbr_validate_environment(surfaces, coeffs)
        moments = _bbr_weighted_moments(surfaces, coeffs)
        sigma_coeff_frac = _bbr_coefficient_uncertainty_frac(
            coeffs, moments, species.clock_frequency_hz
        )
        return sigma_coeff_frac, False

    sigma_frac, _ = bbr_environment_pivot_uncertainty(
        (
            RadiationSurface(
                name="uniform",
                weight=1.0,
                temperature_k=temperature_k,
                temperature_uncertainty_k=temperature_uncertainty_k,
            ),
        ),
        species,
    )
    return sigma_frac, True


# ---------------------------------------------------------------------------
# WP21 Tier 2: ion-clock electric-quadrupole shift (CONVENTIONS.md E34/E35).
# Pure JAX, batched over arbitrary leading (...) axes like the E14a/E14b
# functions above (the gradient tensor is per-point; Theta/J/m_J/axis are
# per-run scalars, mirroring stark_pivot_terms's species_or_coeffs
# convention).
# ---------------------------------------------------------------------------


def traceless_symmetric_gradient(grad_e: jnp.ndarray) -> jnp.ndarray:
    """Traceless symmetric part of the field-gradient tensor (CONVENTIONS.md E34, G8 A5#3).

    The quadrupole interaction couples only to the traceless symmetric
    part of ``grad_E[i, j] = d_i E_j`` (E13): the physical field is
    exactly traceless in vacuum (``div E = 0``) and its curl is zero
    (electrostatics), but a *fitted* (RBF-smoothed, E12) tensor carries a
    small numerical trace and antisymmetric part -- this function removes
    both before any quadrupole contraction, per the G8 sign-off's explicit
    shipping requirement ("confirm the code subtracts it").

    Parameters
    ----------
    grad_e : jax.Array, shape (..., 3, 3)
        Gradient tensor, ``grad_e[..., i, j] = d_i E_j`` (E13), V/m^2.

    Returns
    -------
    jax.Array, shape (..., 3, 3)
        The symmetrized, detraced tensor: ``0.5*(G + G^T) - Tr(G)/3 * I``.
    """
    grad_e = jnp.asarray(grad_e, dtype=jnp.float64)
    sym = 0.5 * (grad_e + jnp.swapaxes(grad_e, -1, -2))
    trace = jnp.trace(sym, axis1=-2, axis2=-1)
    eye = jnp.eye(3, dtype=jnp.float64)
    return sym - (trace[..., None, None] / 3.0) * eye


def quadrupole_mj_factor(j: float, m_j: float) -> float:
    """The ``[J(J+1) - 3*m_J^2] / [J(2J-1)]`` factor (CONVENTIONS.md E34).

    Primary-text form (Roos et al., quant-ph/0701215v1, Eq. 1) -- note the
    ``J(J+1) - 3*m_J^2`` ordering, opposite in sign to an earlier WP21
    draft's ``3*m_J^2 - J(J+1)``; see CONVENTIONS.md section 14's
    reconciliation against Itano 2000 Eq. 46.

    Parameters
    ----------
    j : float
        Total angular momentum J of the state.
    m_j : float
        Magnetic quantum number, ``-J <= m_j <= J``.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``j < 1`` (J=0 or J=1/2: no quadrupole coupling exists at all,
        CONVENTIONS.md E34's immunity note -- the denominator ``J(2J-1)``
        is zero or the state carries no rank-2 moment by construction) or
        if ``abs(m_j) > j``.
    """
    if j < 1.0 - 1e-9:
        raise ValueError(
            f"quadrupole_mj_factor: j={j!r} < 1 has no quadrupole coupling "
            "(CONVENTIONS.md E34: J=0 and J=1/2 states carry no rank-2 electric-"
            "quadrupole moment -- Theta=0 by construction, not merely a formula "
            "singularity)."
        )
    if abs(m_j) > j + 1e-9:
        raise ValueError(f"quadrupole_mj_factor: abs(m_j)={abs(m_j)!r} > j={j!r}")
    denom = j * (2.0 * j - 1.0)
    return (j * (j + 1.0) - 3.0 * m_j**2) / denom


def quadrupole_shift_joules(
    grad_e_total: jnp.ndarray,
    quantization_axis: jnp.ndarray,
    theta_au: float,
    j: float,
    m_j: float,
) -> jnp.ndarray:
    """Electric-quadrupole level shift ``Delta_E_Q(J, m_J)`` (CONVENTIONS.md E34), joules.

    Coordinate-free form (E34's derived reduction of the Itano/Roos
    axial-plus-asymmetric formula, algebraically identical -- see
    CONVENTIONS.md section 14 for the derivation):

    ``Delta_E_Q(J, m_J) = (Theta_SI(J)/2) * [J(J+1)-3*m_J^2]/[J(2J-1)]
    * (n_hat^T . G(r) . n_hat)``

    with ``G(r)`` the traceless symmetric gradient tensor
    (:func:`traceless_symmetric_gradient`) and ``Theta_SI(J) =
    theta_au * cliffordclock.ensemble.species.EA0_SQUARED_SI`` (single
    factor of `e`, G8 sign-off gate edit 2 -- see that constant's
    docstring).

    Parameters
    ----------
    grad_e_total : jax.Array, shape (..., 3, 3)
        Gradient tensor, ``[..., i, j] = d_i E_j`` (E13), V/m^2.
    quantization_axis : jax.Array, shape (3,)
        Quantization-axis direction; need not be pre-normalized (this
        function normalizes it).
    theta_au : float
        Electric-quadrupole moment Theta(J), atomic units (= e*a0^2); sign
        carries physical meaning (CONVENTIONS.md E34's sign-discipline
        note).
    j : float
        Total angular momentum J of the state.
    m_j : float
        Magnetic quantum number.

    Returns
    -------
    jax.Array, shape (...,)
        ``Delta_E_Q``, joules.

    Raises
    ------
    ValueError
        Propagated from :func:`quadrupole_mj_factor` (``j < 1`` or
        ``abs(m_j) > j``).
    """
    grad_e_total = jnp.asarray(grad_e_total, dtype=jnp.float64)
    quantization_axis = jnp.asarray(quantization_axis, dtype=jnp.float64)
    n_hat = quantization_axis / jnp.linalg.norm(quantization_axis)
    gradient_tensor = traceless_symmetric_gradient(grad_e_total)
    # n_hat^T . G . n_hat, contracted over the last two axes of G.
    contraction = jnp.einsum("i,...ij,j->...", n_hat, gradient_tensor, n_hat)
    theta_si = theta_au * EA0_SQUARED_SI
    mj_factor = quadrupole_mj_factor(j, m_j)
    return 0.5 * theta_si * mj_factor * contraction


def quadrupole_pivot_perturbation(
    grad_e_total: jnp.ndarray,
    quantization_axis: jnp.ndarray,
    theta_au: float,
    j: float,
    m_j: float,
    nu_0_hz: float,
) -> jnp.ndarray:
    """``(P-1)_Q`` (CONVENTIONS.md E34/E35): the quadrupole-shift pivot term.

    ``(P-1)_Q = Delta_E_Q(J, m_J) / (h * nu_0)``
    (:func:`quadrupole_shift_joules` divided by the clock photon energy,
    E35's per-configuration-scalar composition into ``(P-1)``).

    Parameters
    ----------
    grad_e_total : jax.Array, shape (..., 3, 3)
        Gradient tensor (E13), V/m^2.
    quantization_axis : jax.Array, shape (3,)
        Quantization-axis direction (need not be pre-normalized).
    theta_au : float
        Theta(J), atomic units.
    j : float
        Total angular momentum J.
    m_j : float
        Magnetic quantum number.
    nu_0_hz : float
        Clock transition frequency, hertz (the SAME transition whose
        upper state has this quadrupole moment -- CONVENTIONS.md E35).

    Returns
    -------
    jax.Array, shape (...,)
        ``(P-1)_Q``, dimensionless.

    Raises
    ------
    ValueError
        Propagated from :func:`quadrupole_mj_factor`.
    """
    shift_j = quadrupole_shift_joules(grad_e_total, quantization_axis, theta_au, j, m_j)
    return shift_j / (PLANCK_H * nu_0_hz)


#: Standard-basis orthonormal triad used by
#: :func:`quadrupole_three_orientation_average` when no explicit `axes` is
#: given -- any orthonormal triad gives the same (exactly zero) result
#: (CONVENTIONS.md E35 A2: "independent of the gradient's own
#: orientation"), so the choice is arbitrary.
_STANDARD_TRIAD = (
    jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
    jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64),
    jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
)


def quadrupole_three_orientation_average(
    grad_e_total: jnp.ndarray,
    theta_au: float,
    j: float,
    m_j: float,
    nu_0_hz: float,
    axes: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Average of ``(P-1)_Q`` over three mutually orthonormal quantization
    axes (CONVENTIONS.md E35 A2).

    Exact zero for ANY gradient tensor and ANY orthonormal triad (proof:
    CONVENTIONS.md section 14 -- ``sum_i n_hat_i^T . G . n_hat_i = Tr(G) =
    0`` since `G` is traceless by construction). Implemented by literally
    evaluating and averaging three real
    :func:`quadrupole_pivot_perturbation` calls (not shortcut to a literal
    ``0.0``) so this function is a numerical test of the identity,
    not a restatement of it -- see `tests/test_quadrupole_pivot.py`'s
    machine-precision cancellation test.

    Parameters
    ----------
    grad_e_total : jax.Array, shape (..., 3, 3)
        Gradient tensor (E13), V/m^2.
    theta_au, j, m_j, nu_0_hz : see :func:`quadrupole_pivot_perturbation`.
    axes : tuple of 3 jax.Array, shape (3,), or None
        An orthonormal triad; defaults to :data:`_STANDARD_TRIAD`.

    Returns
    -------
    jax.Array, shape (...,)
        The average of the three per-axis ``(P-1)_Q`` values (machine-
        precision zero, not exactly `0.0` in floating point, per E20-style
        rounding).

    Raises
    ------
    ValueError
        Propagated from :func:`quadrupole_mj_factor`.
    """
    triad = axes if axes is not None else _STANDARD_TRIAD
    values = jnp.stack(
        [
            quadrupole_pivot_perturbation(grad_e_total, axis, theta_au, j, m_j, nu_0_hz)
            for axis in triad
        ],
        axis=0,
    )
    return jnp.mean(values, axis=0)


# ---------------------------------------------------------------------------
# WP22: gravitational-redshift pivot term (CONVENTIONS.md section 15, E36).
# Pure JAX, batched over arbitrary leading (...) axes like the BBR/
# quadrupole terms above -- height is per-position (varies across an
# extended-lattice ensemble's sites), mirroring quadrupole_pivot_perturbation's
# per-point batching, even though it threads into pivot_perturbation_stark/
# spin_connection_stark/scalar_rate_perturbation_stark/build_omega_stark via
# the same keyword-only additive-composition pattern bbr_pivot_perturbation
# established (E33). See the module docstring's WP22 scope note.
# ---------------------------------------------------------------------------


def height_along_axis(positions_m: jnp.ndarray, up_axis: jnp.ndarray) -> jnp.ndarray:
    """Height coordinate ``h(r) = up_hat · r`` (CONVENTIONS.md E36).

    The projection of a position onto a configured "up" direction --
    purely geometric (no physics beyond a dot product), split out from
    :func:`grav_pivot_perturbation` so a caller can compute a run's
    height-extent (for the ~10 m validity-bound warning, CONVENTIONS.md
    section 15 / G9 sign-off A3) without evaluating the pivot term itself.

    Parameters
    ----------
    positions_m : jax.Array, shape (..., 3)
        Position(s) ``r``, meters, in whatever coordinate frame `up_axis`
        is also expressed in (this project does not otherwise distinguish
        lab frames -- the caller's `trap`/`field` positions and `up_axis`
        must already share one).
    up_axis : jax.Array, shape (3,)
        Direction of increasing height; need not be pre-normalized (this
        function normalizes it) and need not be a coordinate axis (an
        arbitrary "up" direction is supported, e.g. a tilted optical
        table). Must not be the zero vector -- unchecked here (this
        function is called from inside `jax.lax.scan`/`jax.vmap`-traced
        code, e.g. the worldline rotor accumulator, where a Python-level
        ``if`` on a traced norm is not permitted, the same reason
        :func:`quadrupole_shift_joules`'s analogous `n_hat` normalization
        has no such check either); a zero `up_axis` silently divides to
        `nan`/`inf`. Validated instead at config-parse time
        (`cliffordclock.pipeline._parse_gravity`'s
        ``"up_axis must not be the zero vector"`` check), before any value
        reaches this function through the pipeline.

    Returns
    -------
    jax.Array, shape (...,)
        ``h(r)``, meters, measured from the coordinate origin (NOT from
        any configured reference height -- :func:`grav_pivot_perturbation`
        subtracts `reference_height_m` separately).
    """
    positions_m = jnp.asarray(positions_m, dtype=jnp.float64)
    up_axis = jnp.asarray(up_axis, dtype=jnp.float64)
    up_hat = up_axis / jnp.linalg.norm(up_axis)
    return jnp.sum(positions_m * up_hat, axis=-1)


def grav_pivot_perturbation(
    height_m: jnp.ndarray, g_m_s2: float, reference_height_m: float = 0.0
) -> jnp.ndarray:
    """``(P−1)_grav`` (CONVENTIONS.md section 15, E36): the gravitational-redshift pivot term.

    ``(P−1)_grav(r) = U(r)/c² = g·(h(r) − h_ref)/c²`` -- the weak-field
    leading term of the metric proper-time ratio (G9 sign-off A1: "the
    leading term of the metric proper-time ratio
    √(g₀₀(r)/g₀₀(r_ref)) − 1 ≈ (U(r) − U_ref)/c²"; higher-order metric
    terms are ``O((gΔh/c²)²) ~ 1e-32`` over a millimetre and irrelevant).

    **Sign convention (G9 sign-off A1, CONFIRMED): a HIGHER clock runs
    FASTER.** Under the E14b/E21 convention `(P−1) = Δν/ν₀`, this means
    `(P−1)_grav > 0` for `h > h_ref` -- `tests/test_gravity_pivot.py` pins
    `grav_pivot_perturbation(+1.0, STANDARD_GRAVITY) > 0` as a standing
    sign regression, with that physical statement in the test comment
    (G9 sign-off A1's required sign regression).

    **Magnitude (G9 sign-off A1, the "computed never transcribed" gate
    catch): the regression must compute `g/c²` from
    `cliffordclock.constants` at call time, never assert against a
    hard-coded literal** -- an earlier brief and dossier draft both
    transcribed `1.0912e-16/m`, one digit off from the correct
    `g/c² = STANDARD_GRAVITY / SPEED_OF_LIGHT**2 = 1.0911370e-16 per metre
    = 1.0911370e-19 per mm` (at `g = STANDARD_GRAVITY = 9.80665 m/s²`,
    exact `c = 299792458 m/s`). This function does not hard-code that
    quotient at all -- it divides by `SPEED_OF_LIGHT**2` directly, so the
    correct value falls out of the constants module by construction; the
    test-side discipline (computing `g/c²` independently, from the
    constants module, and asserting equality against this function's
    output) is what the gate's catch actually protects against, not
    anything special about this function's own arithmetic.

    Parameters
    ----------
    height_m : jax.Array, shape (...,)
        Height coordinate ``h(r)`` (e.g. from :func:`height_along_axis`),
        meters.
    g_m_s2 : float
        Local gravitational acceleration, m/s². `cliffordclock.pipeline`'s
        `GravityConfig` defaults this to
        `cliffordclock.constants.STANDARD_GRAVITY` (9.80665 m/s², exact by
        definition) -- but at the 1e-19 fractional level the physically
        correct input is the LAB'S OWN SURVEYED LOCAL value (CONVENTIONS.md
        section 15 / G9 sign-off B1), which this function takes as an
        explicit, required argument rather than defaulting internally, so
        every call site states which one it used.
    reference_height_m : float, default 0.0
        The reference height ``h_ref`` at which `(P−1)_grav` is defined to
        be exactly zero (e.g. the ensemble's trap/sample center). Default
        `0.0`: `height_m` is then measured directly from the coordinate
        origin.

    Returns
    -------
    jax.Array, shape (...,)
        ``(P−1)_grav``, dimensionless.
    """
    height_m = jnp.asarray(height_m, dtype=jnp.float64)
    return g_m_s2 * (height_m - reference_height_m) / SPEED_OF_LIGHT**2


# ---------------------------------------------------------------------------
# WP30: quantum-motional second-order-Doppler (time-dilation) pivot term
# (CONVENTIONS.md section 16, E38). Pure Python float arithmetic (not
# jax-batched), mirroring the WP20/WP29 BBR functions' style, not the
# batched jax.Array style of the E14a/E14b/quadrupole/gravity functions
# above: like BBR's radiation temperature, this project's motional state is
# a per-run config-level scalar (one motional state, one species), not a
# per-atom/per-node batched quantity.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionalMode:
    """One normal (secular) motional mode contributing to E38's ``<v^2>``
    (CONVENTIONS.md section 16).

    Attributes
    ----------
    frequency_hz : float
        This mode's ORDINARY frequency ``f_i``, hertz, e.g. as reported
        directly by resolved-sideband thermometry, NOT the angular
        frequency. `motional_pivot_perturbation`/
        `motional_mean_squared_velocity_m2_s2` convert internally via
        ``omega_i = 2*pi*f_i`` (CONVENTIONS.md E38's explicit hbar/m/2pi
        convention: supplying an already-angular frequency here would
        silently overstate the shift by ``(2*pi)^2 ~ 39.5x``). Must be
        `> 0`.
    n_bar : float
        Mean vibrational occupation number of this mode (from sideband
        thermometry, or an equivalent Doppler-limit statement). Must be
        `>= 0`; `n_bar = 0` is the ground-state limit (the zero-point
        `1/2` term alone still contributes).
    n_bar_uncertainty : float
        1-sigma uncertainty on `n_bar`. Default `0.0` (no uncertainty
        contribution from this mode's occupation). Must be `>= 0`.
    frequency_uncertainty_hz : float
        1-sigma uncertainty on `frequency_hz`, hertz. Default `0.0`. Must
        be `>= 0`.
    name : str
        Label for this mode (e.g. ``"axial"``, ``"radial_1"``), used only
        in error messages and pipeline report notes, not a registry key.
    """

    frequency_hz: float
    n_bar: float
    n_bar_uncertainty: float = 0.0
    frequency_uncertainty_hz: float = 0.0
    name: str = ""


def _validate_motional_modes(modes: Sequence[MotionalMode], v_rms_emm_m_s: float) -> None:
    """Raise `ValueError` if `modes`/`v_rms_emm_m_s` violates E38's input invariants.

    Called by every public entry point below
    (`motional_mean_squared_velocity_m2_s2`, `motional_pivot_perturbation`,
    `motional_pivot_uncertainty`) so a direct caller (bypassing
    `cliffordclock.pipeline`'s own parse-time checks entirely) still gets a
    clear rejection instead of a silently wrong shift, mirroring
    `_bbr_validate_environment`'s role for the E37 functions.
    """
    if not modes:
        raise ValueError("motional_state must have at least one MotionalMode")
    for mode in modes:
        if mode.frequency_hz <= 0.0:
            raise ValueError(
                f"motional mode {mode.name!r}: frequency_hz={mode.frequency_hz!r} must be > 0"
            )
        if mode.n_bar < 0.0:
            raise ValueError(f"motional mode {mode.name!r}: n_bar={mode.n_bar!r} must be >= 0")
        if mode.n_bar_uncertainty < 0.0:
            raise ValueError(
                f"motional mode {mode.name!r}: n_bar_uncertainty="
                f"{mode.n_bar_uncertainty!r} must be >= 0"
            )
        if mode.frequency_uncertainty_hz < 0.0:
            raise ValueError(
                f"motional mode {mode.name!r}: frequency_uncertainty_hz="
                f"{mode.frequency_uncertainty_hz!r} must be >= 0"
            )
    if v_rms_emm_m_s < 0.0:
        raise ValueError(f"v_rms_emm_m_s={v_rms_emm_m_s!r} must be >= 0")


def motional_mean_squared_velocity_m2_s2(
    modes: Sequence[MotionalMode], species: Species, v_rms_emm_m_s: float = 0.0
) -> float:
    """``<v^2>`` (CONVENTIONS.md E38): the velocity-variance expectation over
    the motional state, plus the optional excess-micromotion contribution.

    ``<v^2> = sum_i (hbar*omega_i/m)*(n_bar_i + 1/2) + v_rms_emm_m_s^2``
    with ``omega_i = 2*pi*frequency_hz`` (mode frequencies are ORDINARY
    frequencies, e.g. from sideband thermometry, not angular) and ``m``
    the species' registry mass (`species.mass_kg`, never hand-typed).
    `math.fsum` accumulates the per-mode sum (E10-style precision
    discipline, mirroring `_bbr_weighted_moments`'s use of the same
    compensated-summation primitive).

    Parameters
    ----------
    modes : Sequence[MotionalMode]
        The trap's normal modes contributing to the motional state: one
        motional state per run (CONVENTIONS.md E38's composition note: a
        per-atom motional map is future work).
    species : Species
        Supplies `mass_kg`.
    v_rms_emm_m_s : float, default 0.0
        Optional measured rms excess-micromotion (EMM) velocity, m/s
        (CONVENTIONS.md E38's EMM scope note: this project does not model
        the trap RF dynamics that produce EMM, a genuine roadmap package;
        this input takes the lab's own measured EMM characterization,
        already reduced to an equivalent velocity, as given). Default
        `0.0`: no EMM contribution. Must be `>= 0`.

    Returns
    -------
    float
        ``<v^2>``, m^2/s^2.

    Raises
    ------
    ValueError
        `modes` is empty, or any mode/`v_rms_emm_m_s` value is invalid
        (`_validate_motional_modes`).
    """
    _validate_motional_modes(modes, v_rms_emm_m_s)
    mass_kg = species.mass_kg
    modal_sum = math.fsum(
        (HBAR * 2.0 * math.pi * mode.frequency_hz / mass_kg) * (mode.n_bar + 0.5) for mode in modes
    )
    return modal_sum + v_rms_emm_m_s**2


def motional_pivot_perturbation(
    modes: Sequence[MotionalMode], species: Species, v_rms_emm_m_s: float = 0.0
) -> float:
    """``(P-1)_motional`` (CONVENTIONS.md section 16, E38): the quantum-motional
    second-order-Doppler (time-dilation) pivot term.

    ``(P-1)_motional = -<v^2> / (2*c^2)``
    (:func:`motional_mean_squared_velocity_m2_s2`), the same
    second-order-Doppler form E15/E21's kinematic factor already carries
    along classical trajectories, here evaluated as the EXPECTATION VALUE
    of the atom's motional-state velocity-squared operator in place of a
    classical instantaneous velocity. **No double-counting**: every call
    site this function's result is composed at (via the
    ``motional_pivot_perturbation`` keyword-only parameter on
    :func:`pivot_perturbation_stark`/:func:`spin_connection_stark`/
    :func:`scalar_rate_perturbation_stark`/:func:`build_omega_stark`)
    evaluates STATIC (``v = 0`` exactly) lattice/lattice_extended
    quadrature nodes, where E21's classical kinematic term is identically
    zero, precisely why this term is otherwise missing there and precisely
    why adding it here cannot double-count (CONVENTIONS.md E38's central
    argument). `cliffordclock.pipeline` enforces the complementary half of
    this argument (rejecting `environment.motional_state` under
    `ensemble.regime: classical`, where velocities are real and sampled)
    at config-parse time, not here: this function has no `regime` concept
    of its own, by the same design as `bbr_pivot_perturbation`/
    `grav_pivot_perturbation` above.

    Parameters
    ----------
    modes, species, v_rms_emm_m_s : see :func:`motional_mean_squared_velocity_m2_s2`.

    Returns
    -------
    float
        ``(P-1)_motional``, dimensionless.

    Raises
    ------
    ValueError
        Propagated from :func:`motional_mean_squared_velocity_m2_s2`.
    """
    mean_v2 = motional_mean_squared_velocity_m2_s2(modes, species, v_rms_emm_m_s)
    return -mean_v2 / (2.0 * SPEED_OF_LIGHT**2)


def motional_pivot_uncertainty(
    modes: Sequence[MotionalMode],
    species: Species,
    v_rms_emm_m_s: float = 0.0,
    v_rms_emm_uncertainty_m_s: float = 0.0,
) -> float:
    """Propagated fractional uncertainty on `motional_pivot_perturbation`
    (CONVENTIONS.md section 16, E38's uncertainty-propagation note).

    Independent-error quadrature combination of every partial derivative
    times its input's 1-sigma uncertainty, mirroring `bbr_pivot_uncertainty`'s
    "arithmetic-reproduction fidelity, not an independent accuracy claim"
    framing (CONVENTIONS.md section 13's uncertainty note): this propagates
    the uncertainty of the *supplied* mode/EMM inputs through the formula,
    not an independent assessment of the underlying trap physics. Writing
    ``omega_i = 2*pi*f_i``:

    - Each mode's `n_bar_i`: ``d(P-1)/d(n_bar_i) = -(hbar*omega_i/m)/(2c^2)``.
    - Each mode's `f_i`: ``d(P-1)/d(f_i) = -(hbar*2*pi*(n_bar_i+1/2)/m)/(2c^2)``.
    - `v_rms_emm_m_s`: ``d(P-1)/d(v_rms_emm) = -v_rms_emm/c^2``.

    Every term's contribution (`partial * sigma_input`) is squared and
    summed via `math.fsum` before the final `sqrt` (E10-style compensated
    summation, mirroring `bbr_environment_pivot_uncertainty`'s pattern).

    Parameters
    ----------
    modes, species, v_rms_emm_m_s : see :func:`motional_mean_squared_velocity_m2_s2`.
    v_rms_emm_uncertainty_m_s : float, default 0.0
        1-sigma uncertainty on `v_rms_emm_m_s`, m/s. Must be `>= 0`.

    Returns
    -------
    float
        Propagated 1-sigma fractional uncertainty, dimensionless.

    Raises
    ------
    ValueError
        Propagated from :func:`motional_mean_squared_velocity_m2_s2`, or
        `v_rms_emm_uncertainty_m_s` is negative.
    """
    _validate_motional_modes(modes, v_rms_emm_m_s)
    if v_rms_emm_uncertainty_m_s < 0.0:
        raise ValueError(f"v_rms_emm_uncertainty_m_s={v_rms_emm_uncertainty_m_s!r} must be >= 0")

    mass_kg = species.mass_kg
    two_c2 = 2.0 * SPEED_OF_LIGHT**2
    terms_sq = []
    for mode in modes:
        omega_i = 2.0 * math.pi * mode.frequency_hz
        d_dn_bar = -(HBAR * omega_i / mass_kg) / two_c2
        d_df_hz = -(HBAR * 2.0 * math.pi * (mode.n_bar + 0.5) / mass_kg) / two_c2
        terms_sq.append((d_dn_bar * mode.n_bar_uncertainty) ** 2)
        terms_sq.append((d_df_hz * mode.frequency_uncertainty_hz) ** 2)
    d_d_vrms = -v_rms_emm_m_s / SPEED_OF_LIGHT**2
    terms_sq.append((d_d_vrms * v_rms_emm_uncertainty_m_s) ** 2)
    return math.sqrt(math.fsum(terms_sq))
