# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fast-path evaluation for the lattice and classical-periodic regimes (WP8).

Implements ``docs/CONVENTIONS.md`` v1.1.0-draft section 12 (Sprint-2
additions, awaiting G4 sign-off): E29 (lattice static-state fast path, an
exact algebraic corollary of the already-approved E21-E23), E30 (secular
averaging for classical periodic motion), and E31 (the step-size rule for
direct integration). None of these change the underlying physics -- E29 is
exact, E30/E31 are controlled numerical approximations with stated
validity bounds (see each function's docstring).

Every function here is **coupling-agnostic**: instead of taking a field
callable plus an explicit E14a ``mu`` (like
:mod:`cliffordclock.integrator.omega` does), they take a `RateFn` -- a
plain ``(pos, v) -> delta_omega_tilde`` callable implementing E21 for
*whatever* pivot-perturbation coupling model is in play. Today's only
coupling is E14a (explicit linear ``mu``); :mod:`cliffordclock.pipeline`
wires it up via a small closure over
:func:`cliffordclock.integrator.omega.scalar_rate_perturbation`. A future
E14b (quadratic DC-Stark, WP7) coupling slots in behind the same `RateFn`
signature with no change to this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from cliffordclock.cl13 import IDX_SCALAR
from cliffordclock.constants import TAU_COMPTON
from cliffordclock.ensemble.traps import HarmonicTrap
from cliffordclock.integrator.worldline import EnsembleResult, kahan_sum

__all__ = [
    "DEFAULT_POINTS_PER_PERIOD",
    "RateFn",
    "SecularResult",
    "lattice_shift_expectation",
    "secular_average_shift",
    "secular_average_shift_ensemble",
    "select_dtau",
]

#: A coupling-agnostic pivot-perturbation rate callable (E21):
#: ``pos`` shape ``(..., 3)`` m, ``v`` shape ``(..., 3)`` m/s ->
#: ``delta_omega_tilde`` shape ``(...,)``, dimensionless. See module
#: docstring: any coupling model (E14a today, a future E14b, ...) can be
#: wrapped into this signature.
RateFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

#: E31's default resolution ``N_res``: points per trap period, used both
#: by :func:`select_dtau`'s step-size rule and by
#: :func:`secular_average_shift`'s internal one-orbit sub-stepping.
DEFAULT_POINTS_PER_PERIOD = 100

_IDENTITY_ROTOR = jnp.zeros(16, dtype=jnp.float64).at[IDX_SCALAR].set(1.0)


def select_dtau(trap: HarmonicTrap, points_per_period: int = DEFAULT_POINTS_PER_PERIOD) -> float:
    """E31 step-size rule: resolve the *trap* timescale, not the Compton one.

    ``dτ̃ = T_orb / (points_per_period · τ_c)``, with ``T_orb = 2π /
    min(trap.omega_xyz)`` the longest (slowest, most conservative) period
    among the trap's axes, so every axis is resolved at least this
    finely. Rationale (E31): the exponential-midpoint stepper's (E19)
    local error is set by how fast ``Ω`` varies -- i.e. trap dynamics
    (``Ω̈``), not the Compton scale -- and ``exp`` of a bivector lands
    exactly on the rotor group at any ``dτ̃`` (E20's norm preservation is
    step-size independent), so this is a practical accuracy rule, not a
    stability requirement. ``points_per_period=100`` is E31's default
    ``N_res``; ``docs/timescales.md`` documents the accuracy study this
    default is based on.

    Parameters
    ----------
    trap : HarmonicTrap
        Supplies the per-axis angular trap frequencies (mass-independent:
        `HarmonicTrap.omega_xyz` is the angular frequency directly, so
        the orbit period is the same for any particle mass in the trap).
    points_per_period : int, default 100
        E31's ``N_res``: resolution points per trap period.

    Returns
    -------
    float
        ``dτ̃``, dimensionless (Compton units, E9).
    """
    if points_per_period < 1:
        raise ValueError(f"points_per_period must be >= 1, got {points_per_period}")
    omega_min = min(float(w) for w in trap.omega_xyz)
    if omega_min <= 0.0:
        raise ValueError(f"trap.omega_xyz must be all positive, got {trap.omega_xyz}")
    period_s = 2.0 * math.pi / omega_min
    period_tilde = period_s / TAU_COMPTON
    return period_tilde / points_per_period


def lattice_shift_expectation(
    rate_fn: RateFn,
    nodes: jnp.ndarray,
    weights: jnp.ndarray,
    t_interrogation_s: float,
) -> EnsembleResult:
    """E29 lattice fast path: exact corollary of E21-E23 for a
    time-independent field and a stationary motional state.

    For each static quadrature node, ``v = 0`` exactly (no motion) so the
    E21 instantaneous rate ``δω̃_q`` is *constant* in time and E22's
    integral is trivial: ``ΔΦ_q = δω̃_q · T̃`` for any interrogation time
    ``T`` -- exact up to the quadrature's own accuracy (WP4), with **no
    time integration at all** (constant cost in `T`). This is the default
    execution path for ``ensemble.regime: lattice`` configs (see
    :mod:`cliffordclock.pipeline`); the worldline integrator (E17-E19)
    remains available as an explicit cross-check
    (``integration.mode: worldline``) and must agree exactly on static
    nodes at any step count (E29's own statement; see
    ``tests/test_fastpath_lattice.py``'s Tier A vs Tier C test).

    Per-node values are kept (not just the weighted mean) so downstream
    per-atom analytics (T2*, coherence function, line profile -- all E25-
    E28 consumers of per-atom ``ΔΦ_i``, WP5) work unchanged. The weighted
    expectation ``⟨Δν/ν₀⟩ = Σ_q w_q δω̃_q`` (E23) is realized by the
    existing analytics machinery downstream
    (`cliffordclock.analytics.stats.mean_fractional_shift`) from this
    function's per-node `phase` and the (validated, passed-through)
    `weights` -- not recomputed here, to avoid a second, divergeable
    implementation of E23.

    Parameters
    ----------
    rate_fn : RateFn
        ``(pos, v) -> δω̃`` (E21); coupling-agnostic, see module
        docstring.
    nodes : jax.Array, shape (M, 3)
        Static quadrature-node positions (WP4), meters.
    weights : jax.Array, shape (M,)
        Quadrature weights, dimensionless; validated for shape only (the
        weighted expectation itself is computed downstream, E23).
    t_interrogation_s : float
        Interrogation time ``T``, seconds. Any value -- the fast path's
        defining property is O(1) cost in `T`.

    Returns
    -------
    EnsembleResult
        Per-node results, `WorldlineResult`-shaped for drop-in reuse by
        :mod:`cliffordclock.pipeline`: `phase` is E29's ``ΔΦ_q`` per
        node; `phase_rotor` equals `phase` exactly (E29 is an exact
        scalar corollary -- there is no separate rotor integration to
        diverge from); `norm_error`/`max_norm_drift` are exactly zero (no
        rotor state is ever advanced); `n_steps` is zero (no time
        stepping); `r_final` is the identity rotor for every node.
    """
    nodes = jnp.asarray(nodes, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    if nodes.ndim != 2 or nodes.shape[-1] != 3:
        raise ValueError(f"nodes must have shape (M, 3); got {nodes.shape}")
    m = nodes.shape[0]
    if weights.shape != (m,):
        raise ValueError(f"weights must have shape ({m},), matching nodes; got {weights.shape}")
    if t_interrogation_s <= 0.0:
        raise ValueError("t_interrogation_s must be positive")

    v_zero = jnp.zeros_like(nodes)
    delta_omega_tilde = jnp.asarray(rate_fn(nodes, v_zero), dtype=jnp.float64)  # (M,), E21 (v=0)
    if delta_omega_tilde.shape != (m,):
        raise ValueError(
            f"rate_fn(nodes, v=0) must return shape ({m},); got {delta_omega_tilde.shape}"
        )

    t_tilde = t_interrogation_s / TAU_COMPTON
    phase = delta_omega_tilde * t_tilde  # E29: Delta_Phi_q = delta_omega~_q * T_tilde, exact.

    r_final = jnp.broadcast_to(_IDENTITY_ROTOR, (m, 16))
    zeros_m = jnp.zeros(m, dtype=jnp.float64)
    return EnsembleResult(
        r_final=r_final,
        phase=phase,
        phase_rotor=phase,
        fractional_shift=delta_omega_tilde,
        norm_error=zeros_m,
        max_norm_drift=zeros_m,
        n_steps=jnp.zeros(m, dtype=jnp.int64),
    )


class SecularResult(NamedTuple):
    """One atom's E30 secular-averaged result (`WorldlineResult`-shaped).

    Attributes
    ----------
    r_final, phase, phase_rotor, fractional_shift, norm_error,
    max_norm_drift, n_steps
        Same meaning/shape convention as
        `cliffordclock.integrator.worldline.WorldlineResult`, so a batch
        of `SecularResult` can be repacked into an
        `~cliffordclock.integrator.worldline.EnsembleResult` for pipeline
        reuse (see :func:`secular_average_shift_ensemble`). `r_final` is
        always the identity rotor and `norm_error`/`max_norm_drift` are
        always exactly zero -- E30's scalar-only pipeline never advances
        a rotor state; `phase_rotor` always equals `phase` for the same
        reason. `n_steps` is the number of one-orbit sub-steps used for
        the internal quadrature (not related to `t_interrogation_s`).
    mean_delta_omega_tilde : jax.Array, scalar
        ``⟨δω̃⟩_orb`` (E30), dimensionless -- equal to `fractional_shift`,
        kept as a separately named field for clarity and direct testing.
    epsilon_bound_phase : jax.Array, scalar
        E30's documented partial-orbit remainder bound, ``|ε| ≤ T̃_orb ·
        max_t |δω̃(t) − ⟨δω̃⟩_orb|``, dimensionless (phase units). Bounds
        the truncation error from `t_interrogation_s` not being an exact
        integer multiple of the orbit period (this implementation always
        evaluates ``⟨δω̃⟩_orb`` over exactly one full orbit, so this bound
        applies to the *last, possibly-partial* orbit within
        `t_interrogation_s` -- see :func:`secular_average_shift`'s
        docstring).
    t_orbit_s : jax.Array, scalar
        The closed-form trap orbit period ``T_orb = 2π/ω`` used, seconds.
    """

    r_final: jnp.ndarray
    phase: jnp.ndarray
    phase_rotor: jnp.ndarray
    fractional_shift: jnp.ndarray
    norm_error: jnp.ndarray
    max_norm_drift: jnp.ndarray
    n_steps: jnp.ndarray
    mean_delta_omega_tilde: jnp.ndarray
    epsilon_bound_phase: jnp.ndarray
    t_orbit_s: jnp.ndarray


def _check_isotropic(trap: HarmonicTrap) -> float:
    """Validate `trap` is isotropic and return the common angular frequency.

    E30's closed-form one-orbit average requires a *periodic* classical
    orbit (CONVENTIONS.md section 12, E30 validity bound); a 3D harmonic
    trap only guarantees a closed orbit for *arbitrary* initial
    conditions when all three axes share one angular frequency (otherwise
    the motion is in general a non-closing Lissajous curve). Anisotropic-
    but-commensurate traps are a straightforward generalization left out
    of scope (WP8 non-goal: "no anharmonic secular theory" -- the same
    spirit applies to the commensurate-frequency case, which is not
    requested).
    """
    omega = [float(w) for w in trap.omega_xyz]
    if omega[0] <= 0.0 or omega[1] <= 0.0 or omega[2] <= 0.0:
        raise ValueError(f"trap.omega_xyz must be all positive, got {trap.omega_xyz}")
    isotropic = math.isclose(omega[0], omega[1], rel_tol=1e-9) and math.isclose(
        omega[1], omega[2], rel_tol=1e-9
    )
    if not isotropic:
        raise ValueError(
            "secular_average_shift requires an isotropic HarmonicTrap (omega_xyz all "
            f"equal) for a guaranteed closed one-period orbit (E30 validity bound: "
            f"'periodic classical motion'); got omega_xyz={trap.omega_xyz}. Use "
            "integration.mode='direct' (Tier B(i)) for anisotropic traps."
        )
    return omega[0]


def _shm_trajectory(
    omega0: float,
    center: jnp.ndarray,
    initial_position: jnp.ndarray,
    initial_velocity: jnp.ndarray,
    n_steps: int,
    dt_s: float,
) -> jnp.ndarray:
    """Exact closed-form harmonic-oscillator trajectory, ``(n_steps + 1, 3)`` samples.

    ``r(t) = center + Δr0·cos(ω0 t) + (v0/ω0)·sin(ω0 t)``: the exact
    solution of `HarmonicTrap`'s (mass-independent) equation of motion
    ``a = -ω0² (r - center)`` -- the closed form the harmonic trap
    literally integrates to (CONVENTIONS.md section 9, V4: "harmonic
    trap, classical atom ... r(t) sinusoidal"), not a numerical
    approximation. Used in place of
    `cliffordclock.ensemble.classical.propagate_verlet` (which only
    conserves energy to Verlet's own truncation order) so the one-orbit
    average below is limited only by the `rate_fn` quadrature, not by any
    additional trajectory-integration error.
    """
    delta_r0 = initial_position - center
    t = jnp.arange(n_steps + 1, dtype=jnp.float64) * dt_s
    phase = omega0 * t
    cos_p = jnp.cos(phase)[:, None]
    sin_p = jnp.sin(phase)[:, None]
    return (
        center[None, :] + delta_r0[None, :] * cos_p + (initial_velocity / omega0)[None, :] * sin_p
    )


def secular_average_shift(
    rate_fn: RateFn,
    trap: HarmonicTrap,
    initial_position: jnp.ndarray,
    initial_velocity: jnp.ndarray,
    t_interrogation_s: float,
    *,
    points_per_period: int = DEFAULT_POINTS_PER_PERIOD,
) -> SecularResult:
    """E30 secular averaging: ``ΔΦ = ⟨δω̃⟩_orb · T̃ + ε`` for a classical
    atom in periodic motion through a static field.

    ``⟨δω̃⟩_orb`` is the one-orbit line integral of `rate_fn` (E21) along
    the trap's *exact* closed-form harmonic orbit
    (:func:`_shm_trajectory`), discretized with the same midpoint-sample
    convention as
    `cliffordclock.integrator.worldline.integrate_worldline` (E19-style:
    field/rate evaluated at the linear midpoint between consecutive
    position samples, finite-difference velocity) and Kahan-summed
    (E22's compensated-summation discipline, E10) -- so this agrees with
    that "direct integrator" at the per-step level whenever `rate_fn` is
    built from the same coupling model, up to the two paths' different
    orbit-fraction/step-count choices.

    Validity (E30, CONVENTIONS.md section 12): static field, periodic
    classical motion, ``T ≫ T_orb``. Not valid for drifting/chaotic
    trajectories or time-dependent fields -- use direct integration
    there. Requires an *isotropic* `trap` so the orbit is guaranteed to
    close for arbitrary initial conditions (see `_check_isotropic`);
    raises `ValueError` otherwise.

    Parameters
    ----------
    rate_fn : RateFn
        ``(pos, v) -> δω̃`` (E21); see module docstring.
    trap : HarmonicTrap
        Must be isotropic (``omega_xyz`` all equal, within a relative
        tolerance of ``1e-9``).
    initial_position : jax.Array, shape (3,)
        Orbit initial position, meters.
    initial_velocity : jax.Array, shape (3,)
        Orbit initial velocity, m/s.
    t_interrogation_s : float
        Total interrogation time ``T``, seconds (typically ``≫`` the
        orbit period ``T_orb``).
    points_per_period : int, default 100
        E31's resolution for the internal one-orbit quadrature (see
        :func:`select_dtau`).

    Returns
    -------
    SecularResult
    """
    if t_interrogation_s <= 0.0:
        raise ValueError("t_interrogation_s must be positive")
    omega0 = _check_isotropic(trap)
    center = jnp.asarray(trap.center, dtype=jnp.float64)
    initial_position = jnp.asarray(initial_position, dtype=jnp.float64)
    initial_velocity = jnp.asarray(initial_velocity, dtype=jnp.float64)

    dtau = select_dtau(trap, points_per_period)
    dt_s = dtau * TAU_COMPTON
    t_orbit_s = 2.0 * math.pi / omega0
    n_steps_orbit = max(1, round(t_orbit_s / dt_s))

    traj = _shm_trajectory(omega0, center, initial_position, initial_velocity, n_steps_orbit, dt_s)
    pos_a, pos_b = traj[:-1], traj[1:]
    pos_mid = 0.5 * (pos_a + pos_b)
    v_mid = (pos_b - pos_a) / dt_s
    delta_omega_tilde_mid = jnp.asarray(rate_fn(pos_mid, v_mid), dtype=jnp.float64)

    increments = delta_omega_tilde_mid * dtau
    phase_orbit = kahan_sum(increments)  # Delta_Phi_orb (E22-style, one orbit).
    t_orbit_tilde = n_steps_orbit * dtau
    mean_delta_omega_tilde = phase_orbit / t_orbit_tilde  # <delta_omega~>_orb (E30).

    t_tilde_total = t_interrogation_s / TAU_COMPTON
    phase_total = mean_delta_omega_tilde * t_tilde_total  # E30: Delta_Phi = <..>_orb * T_tilde.

    epsilon_bound = t_orbit_tilde * jnp.max(jnp.abs(delta_omega_tilde_mid - mean_delta_omega_tilde))

    return SecularResult(
        r_final=_IDENTITY_ROTOR,
        phase=phase_total,
        phase_rotor=phase_total,
        fractional_shift=mean_delta_omega_tilde,
        norm_error=jnp.asarray(0.0, dtype=jnp.float64),
        max_norm_drift=jnp.asarray(0.0, dtype=jnp.float64),
        n_steps=jnp.asarray(n_steps_orbit, dtype=jnp.int64),
        mean_delta_omega_tilde=mean_delta_omega_tilde,
        epsilon_bound_phase=epsilon_bound,
        t_orbit_s=jnp.asarray(t_orbit_s, dtype=jnp.float64),
    )


def secular_average_shift_ensemble(
    rate_fn: RateFn,
    trap: HarmonicTrap,
    initial_positions: jnp.ndarray,
    initial_velocities: jnp.ndarray,
    t_interrogation_s: float,
    *,
    points_per_period: int = DEFAULT_POINTS_PER_PERIOD,
) -> EnsembleResult:
    """`jax.vmap` of :func:`secular_average_shift` over an ensemble of atoms.

    Every atom shares `trap`/`t_interrogation_s`/`points_per_period` but
    has its own initial conditions (E30).

    Parameters
    ----------
    rate_fn, trap, t_interrogation_s, points_per_period
        See :func:`secular_average_shift`.
    initial_positions : jax.Array, shape (M, 3)
        Meters.
    initial_velocities : jax.Array, shape (M, 3)
        m/s.

    Returns
    -------
    EnsembleResult
        Only the `WorldlineResult`-common fields are stacked;
        `SecularResult`'s extra diagnostic fields (`mean_delta_omega_tilde`,
        `epsilon_bound_phase`, `t_orbit_s`) are dropped by this adapter
        (call :func:`secular_average_shift` directly, per atom, for those
        diagnostics).
    """

    def run_one(pos0: jnp.ndarray, vel0: jnp.ndarray) -> SecularResult:
        return secular_average_shift(
            rate_fn, trap, pos0, vel0, t_interrogation_s, points_per_period=points_per_period
        )

    batched = jax.vmap(run_one)(
        jnp.asarray(initial_positions, dtype=jnp.float64),
        jnp.asarray(initial_velocities, dtype=jnp.float64),
    )
    return EnsembleResult(
        r_final=batched.r_final,
        phase=batched.phase,
        phase_rotor=batched.phase_rotor,
        fractional_shift=batched.fractional_shift,
        norm_error=batched.norm_error,
        max_norm_drift=batched.max_norm_drift,
        n_steps=batched.n_steps,
    )
