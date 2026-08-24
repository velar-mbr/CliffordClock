# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coherent (phase-resolved) rotor composition and Ramsey visibility
(CONVENTIONS.md section 8, E39).

Each worldline `k` in an ensemble carries an accumulated perturbation
phase `ΔΦ_k` (E22). This module turns that phase into a unit rotor
confined to the engine's internal-circulation plane `B̂_C = e_1 ∧ e_2`
(E18), composes the ensemble's rotors into a single (deliberately
NOT unit-norm) multivector via a population-weighted LINEAR sum, and
reads the Ramsey fringe visibility/phase off that sum's projection onto
`B̂_C`. See `docs/CONVENTIONS.md` section 8 (E39) for the governing
equations and the two-classic-errors discussion this module's kill tests
(``tests/test_coherent_visibility.py``) encode directly.

Scope boundary (binding, stated everywhere this module's output reaches a
report): valid only for GAUSSIAN-distributed accumulated phases (thermal,
coherent, squeezed motional states -- genuine positive distributions over
worldlines). Non-Gaussian motional states (Fock `n >= 1`, cat states) are
out of scope for this worldline-ensemble representation.
"""

from __future__ import annotations

import jax.numpy as jnp

from cliffordclock.cl13 import IDX_E12, IDX_SCALAR, exp_bivector

__all__ = [
    "phase_to_rotor",
    "coherent_rotor_composition",
    "ramsey_visibility_and_phase",
]


def phase_to_rotor(phase: jnp.ndarray) -> jnp.ndarray:
    """Build each worldline's unit ``B̂_C``-plane phase-factor rotor (E39).

    ``R_k = exp(ΔΦ_k · B̂_C) = cos(ΔΦ_k) + sin(ΔΦ_k)·e_12`` (E6's
    `exp_bivector`, `B̂_C = e_1∧e_2` per E18), the rotor-algebra restatement
    of the coherence function's own ``exp(i·ΔΦ_k)`` phase factor
    (`cliffordclock.analytics.stats.coherence_function`, E26) under the
    identification ``e_12 <-> i``: this function's output, linearly
    weighted-summed by :func:`coherent_rotor_composition` and projected by
    :func:`ramsey_visibility_and_phase`, reproduces
    `coherence_function`'s endpoint value exactly (same weights, same
    `phase`, evaluated at `t = t_interrogation_s`) -- a cross-check, not a
    coincidence, since both express the same population-weighted coherent
    sum in different (complex-number vs. rotor-algebra) notation.

    Deliberately built from the FULL accumulated phase `ΔΦ_k` (E22, the
    primary scalar observable, G0 item 3), not from a worldline's own
    dynamical rotor `R_k = exp(−½ Ω dτ̃)`-chain output
    (`cliffordclock.integrator.worldline.WorldlineResult.r_final`): that
    dynamical rotor's own `B̂_C`-plane bivector angle is `ΔΦ_k / 2` (the
    standard spin-1/2-type "half angle" convention E19's `exp(−½ Ω dτ̃)`
    step already carries, see `cliffordclock.integrator.stepper.rotor_plane_angle`'s
    docstring), so summing `r_final` values directly would read off the
    wrong angle scale and break the Gaussian closure identity's exact
    ``exp(−σ_Φ²/2)`` coefficient. Building fresh full-angle phase rotors
    here from `ΔΦ_k` directly keeps the two representations (dynamical
    rotor chain, Ramsey coherent sum) exact and decoupled from any
    second-order E24 rotor/scalar cross-check divergence.

    Parameters
    ----------
    phase : jax.Array, shape (..., )
        Accumulated perturbation phases ``ΔΦ_k`` (E22), dimensionless.

    Returns
    -------
    jax.Array, shape (..., 16)
        Unit rotors confined to the `B̂_C` plane (only the scalar and
        `e_12` components are nonzero).
    """
    phase = jnp.asarray(phase, dtype=jnp.float64)
    bivector = jnp.zeros(phase.shape + (16,), dtype=jnp.float64).at[..., IDX_E12].set(phase)
    return exp_bivector(bivector)


def coherent_rotor_composition(rotors: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """The ensemble coherence object ``M = Σ_k p_k R_k`` (E39).

    A population-weighted LINEAR sum of the ensemble's unit rotors
    (typically :func:`phase_to_rotor`'s output) -- **never renormalized**.
    `M` is deliberately NOT a rotor (`⟨M M̃⟩_0 != 1` in general): its
    departure from unit norm IS the decoherence signal E39 exists to
    report (:func:`ramsey_visibility_and_phase`). Renormalizing `M` back
    to a unit rotor (the "average of rotations should itself be a
    rotation" instinct) silently erases exactly this signal -- kill test
    (b) in ``tests/test_coherent_visibility.py`` encodes this directly.

    Parameters
    ----------
    rotors : jax.Array, shape (M, 16)
        One (typically unit) multivector per worldline, dtype float64.
    weights : jax.Array, shape (M,)
        The ensemble's own PROBABILITY weights `p_k` (E23's convention:
        uniform ``1/M`` for classical Monte-Carlo, quadrature weights for
        lattice motional nodes), already normalized to sum to 1. This
        function does not renormalize `weights` itself -- the caller
        supplies exactly the weights that define the population average.

    Returns
    -------
    jax.Array, shape (16,)
        ``M = Σ_k p_k R_k``, dimensionless. NOT unit-norm in general.

    Raises
    ------
    ValueError
        `rotors` is not shape ``(M, 16)``, or `weights` is not shape
        ``(M,)`` matching `rotors`.
    """
    rotors = jnp.asarray(rotors, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    if rotors.ndim != 2 or rotors.shape[-1] != 16:
        raise ValueError(f"rotors must have shape (M, 16); got {rotors.shape}")
    if weights.shape != (rotors.shape[0],):
        raise ValueError(
            f"weights must have shape ({rotors.shape[0]},), matching rotors; got {weights.shape}"
        )
    return jnp.sum(weights[:, None] * rotors, axis=0)


def ramsey_visibility_and_phase(m: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Ramsey fringe visibility and phase from the coherence object ``M`` (E39).

    Projects `m` onto the `B̂_C = e_1∧e_2` plane (E18's fixed internal-
    circulation plane, the Ramsey interrogation's own rotation plane) by
    reading off only its scalar and `e_12` components -- any component of
    `m` outside this plane (e.g. from composing genuinely dynamical
    worldline rotors carrying a nonzero `ω_boost`, rather than
    :func:`phase_to_rotor`'s deliberately plane-confined output) is
    dropped, not folded into the visibility. For a plane-confined
    multivector ``m = c + s·e_12``, the plane projection's `⟨m̃m⟩_0`-like
    scalar norm is exactly ``c² + s²`` (since ``e_12² = -1``), so:

        visibility V = sqrt(c² + s²)      (E39; V <= 1, with equality
                                            iff every composed phase is
                                            identical -- the "no spread,
                                            no decoherence" limit)
        phase       = atan2(s, c)         (E39; the fringe phase, the
                                            argument of ``c + s·e_12``
                                            under ``e_12 <-> i``)

    Gaussian closure (the validation identity, CONVENTIONS.md section 8):
    for Gaussian-distributed accumulated phases ``ΔΦ_k ~ N(μ, σ_Φ²)``,
    ``V = exp(-σ_Φ²/2)`` exactly, matching
    `cliffordclock.analytics.stats.dephasing_time_t2star`'s own Gaussian
    inhomogeneous-dephasing assumption (E27) from the SAME
    characteristic-function identity ``E[exp(i·ΔΦ)] = exp(iμ - σ²/2)``.

    Scope boundary: valid only for Gaussian-distributed accumulated
    phases (thermal, coherent, squeezed motional states); non-Gaussian
    motional states (Fock `n >= 1`, cat states) are out of scope for the
    worldline-ensemble representation this function consumes.

    Parameters
    ----------
    m : jax.Array, shape (..., 16)
        The ensemble coherence object (:func:`coherent_rotor_composition`'s
        output), or any multivector to project.

    Returns
    -------
    visibility : jax.Array, shape (...,)
        ``V``, dimensionless, `0 <= V <= 1` when `m` is a genuine
        population-weighted sum of unit `B̂_C`-plane rotors (a convex
        combination of unit-modulus values).
    phase : jax.Array, shape (...,)
        The fringe phase, radians, in ``(-pi, pi]``.
    """
    m = jnp.asarray(m, dtype=jnp.float64)
    c = m[..., IDX_SCALAR]
    s = m[..., IDX_E12]
    visibility = jnp.sqrt(c * c + s * s)
    phase = jnp.arctan2(s, c)
    return visibility, phase
