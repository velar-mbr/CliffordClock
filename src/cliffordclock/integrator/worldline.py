# SPDX-License-Identifier: AGPL-3.0-or-later
"""Worldline rotor integrator (CONVENTIONS.md E9, E17-E24).

Integrates the rotor evolution equation (E17) along a *given* worldline
(precomputed positions -- this module does not move atoms; that is the
ensemble sampler's job) via ``jax.lax.scan`` over Compton-unit steps, and
an ensemble of worldlines via ``jax.vmap``. Non-goals, explicit: no
trajectory generation, no ensemble statistics/T2*/reports (those live in
the analytics package), no adjoint/backward integration (this is a
clean-room, forward-time-only implementation by design), no adaptive step
control (fixed ``dτ̃`` for the MVP).

Numerical-precision discipline (E10, non-negotiable):

- Only *perturbation* quantities are accumulated -- never the absolute
  Compton phase. This falls out of :mod:`cliffordclock.integrator.omega`:
  ``Ω`` is already zero when unperturbed (``P = 1``, ``v = 0``), so the
  rotor `R` integrated here never carries the ``~1e20`` rad/s absolute
  Compton rate.
- Phase accumulation (E22) uses compensated (Kahan) summation
  (:func:`kahan_sum` / ``_kahan_add``) threaded through the
  ``lax.scan`` carry, not a plain running sum.
- float64 throughout (inherited from ``cliffordclock`` package import-time
  x64 config); tests assert dtype.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cliffordclock.cl13 import IDX_SCALAR, normalize_rotor, rotor_norm_sq
from cliffordclock.constants import TAU_COMPTON
from cliffordclock.integrator.stepper import rotor_step

#: A field callable: position(s) (m) -> (E (V/m), grad_E (V/m^2)), matching
#: ``FieldSmoother.evaluate``'s combined-callable convention (WP2). Per
#: this WP's scope note (see ``omega.py``), the returned `E` is treated as
#: the perturbation field ``δE`` (E11).
FieldFn = Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]

#: Default renormalization cadence (E20), in steps. Chosen from the norm-drift
#: measurement in ``tests/test_integrator_worldline.py``
#: (``test_norm_preservation_one_million_steps``): geometric-product
#: composition of near-unit rotors accumulates float64 rounding roughly
#: linearly in step count, empirically << 1e-12 per 1000-step interval for
#: this integrator's per-step operation count (one `exp_bivector` + one
#: `geometric_product`), while renormalizing this often costs a negligible
#: fraction of total FLOPs. Exposed as the `renorm_every` keyword so
#: callers can tighten/loosen it.
DEFAULT_RENORM_EVERY = 1000

# NOTE (jit-compatibility, MAJOR 4 fix): earlier revisions of this module
# bundled the dynamic, differentiable `mu` array and the static Python int
# `renorm_every` together into a single `IntegratorParams` NamedTuple
# argument. Under `jax.jit`, every leaf of a pytree argument is traced
# unless the *whole argument* is named in `static_argnums`/`static_argnames`
# -- and `renorm_every` cannot be marked static without also freezing `mu`,
# since they shared one argument. That made `int(params.renorm_every)`
# inside `integrate_worldline` raise a `ConcretizationTypeError` under jit
# (a Tracer has no concrete int value at trace time), so the WP3 spec
# requirement "must run end-to-end under jit" was unmet.
#
# The fix: `mu` and `renorm_every` are now separate arguments to
# `integrate_worldline`/`integrate_ensemble` -- `mu` positional (dynamic,
# differentiable), `renorm_every` a plain-`int` keyword (static). A caller
# that wants to `jax.jit` either function passes
# `static_argnames=("renorm_every", "n_steps")` (both control array shapes
# or Python-level control flow, so both must stay static), or simply
# closes over `renorm_every`/`n_steps` in an outer function instead of
# passing them through the jitted call at all. No `float()`/`int()`
# coercion of a possibly-traced value remains anywhere on the scan path.


class WorldlineResult(NamedTuple):
    """Result of integrating one worldline's rotor (E17-E24).

    Attributes
    ----------
    r_final : jax.Array, shape (16,)
        Final rotor.
    phase : jax.Array, scalar
        Accumulated perturbation phase ``ΔΦ`` (E22), dimensionless --
        the *primary* observable pipeline (scalar E21, G0 item 3).
    phase_rotor : jax.Array, scalar
        Accumulated rotor-extracted phase (E24 standing cross-check),
        dimensionless. Compare to `phase`: equal at first order in any
        boost term present; a resolved second-order (``O(ω_boost²)``)
        divergence is not a bug (E24 acceptance criterion).
    fractional_shift : jax.Array, scalar
        ``(Δν/ν₀) = ΔΦ / T̃`` (E23), dimensionless.
    norm_error : jax.Array, scalar
        ``|⟨R R̃⟩_0 − 1|`` at the final rotor (E20 diagnostic).
    max_norm_drift : jax.Array, scalar
        Worst-case ``|⟨R R̃⟩_0 − 1|`` observed at *any* step during the
        run (E20 diagnostic: bounds the per-renormalization-interval
        drift, since it is a maximum over every step, not just
        renormalization checkpoints).
    n_steps : jax.Array, scalar int
        Number of integration steps taken.
    """

    r_final: jnp.ndarray
    phase: jnp.ndarray
    phase_rotor: jnp.ndarray
    fractional_shift: jnp.ndarray
    norm_error: jnp.ndarray
    max_norm_drift: jnp.ndarray
    n_steps: jnp.ndarray


class EnsembleResult(NamedTuple):
    """Batched :class:`WorldlineResult`: every field gains a leading ``(M,)`` axis."""

    r_final: jnp.ndarray
    phase: jnp.ndarray
    phase_rotor: jnp.ndarray
    fractional_shift: jnp.ndarray
    norm_error: jnp.ndarray
    max_norm_drift: jnp.ndarray
    n_steps: jnp.ndarray


def _kahan_add(
    total: jnp.ndarray, comp: jnp.ndarray, value: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One step of Kahan (compensated) summation (E10).

    Parameters
    ----------
    total : jax.Array, scalar
        Running sum so far.
    comp : jax.Array, scalar
        Running compensation (low-order bits lost in previous additions).
    value : jax.Array, scalar
        Value to add.

    Returns
    -------
    (new_total, new_comp) : tuple of jax.Array, scalar
    """
    y = value - comp
    t = total + y
    new_comp = (t - total) - y
    return t, new_comp


def kahan_sum(values: jnp.ndarray) -> jnp.ndarray:
    """Compensated (Kahan) summation of a 1-D array (E10).

    Exposed standalone (in addition to being used inline in
    :func:`integrate_worldline`'s scan body) so tests can directly probe
    compensated-summation order-stability (WP3 test contract item 7).

    Parameters
    ----------
    values : jax.Array, shape (N,)
        Values to sum, float64.

    Returns
    -------
    jax.Array, scalar
        ``Σ values``, computed with error-free (Kahan) accumulation.
    """
    values = jnp.asarray(values, dtype=jnp.float64)

    def body(
        carry: tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        total, comp = carry
        return _kahan_add(total, comp, x), None

    init = (jnp.asarray(0.0, dtype=jnp.float64), jnp.asarray(0.0, dtype=jnp.float64))
    (total, _comp), _ = jax.lax.scan(body, init, values)
    return total


def _prepare_trajectory(trajectory: jnp.ndarray, n_steps: int | None) -> jnp.ndarray:
    """Normalize `trajectory` to a dense ``(n_steps + 1, 3)`` position array.

    Per the WP3 spec: "trajectory: positions (T, 3) in meters ... For
    quadrature-node (static-position) inputs, a (3,) position is
    broadcast over time." Broadcasting a static ``(3,)`` position to
    ``(n_steps + 1, 3)`` also makes the finite-difference velocity used
    by :func:`integrate_worldline` come out exactly zero, with no special
    casing needed elsewhere.
    """
    trajectory = jnp.asarray(trajectory, dtype=jnp.float64)
    if trajectory.ndim == 1:
        if trajectory.shape != (3,):
            raise ValueError(f"static trajectory must have shape (3,), got {trajectory.shape}")
        if n_steps is None:
            raise ValueError("n_steps is required when trajectory is a static (3,) position")
        return jnp.broadcast_to(trajectory, (n_steps + 1, 3))
    if trajectory.ndim != 2 or trajectory.shape[-1] != 3:
        raise ValueError(f"trajectory must have shape (T, 3) or (3,), got {trajectory.shape}")
    if trajectory.shape[0] < 2:
        raise ValueError("trajectory must have at least 2 samples (>= 1 step)")
    return trajectory


def integrate_worldline(
    field_fn: FieldFn,
    trajectory: jnp.ndarray,
    dtau: ArrayLike,
    mu: jnp.ndarray,
    *,
    renorm_every: int = DEFAULT_RENORM_EVERY,
    n_steps: int | None = None,
) -> WorldlineResult:
    """Integrate the rotor (E17) along one worldline.

    Time loop via ``jax.lax.scan``; each step advances between two
    consecutive `trajectory` samples using the exponential-midpoint
    scheme (E19): the field/velocity are evaluated at the linear
    midpoint position between the two samples (exact for a worldline
    sampled at the same cadence as `dtau`, which is what this WP assumes
    -- trajectories are precomputed by the ensemble sampler, WP4, at the
    integrator's own step cadence). The finite-difference velocity
    ``v = (r_{k+1} − r_k) / (dτ̃ · τ_c)`` is used for the E18 boost term
    and E21 kinematic factor.

    Differentiable end-to-end (``jax.grad`` of `phase` w.r.t. a
    field-scaling parameter closed over by `field_fn`, *and* w.r.t. `mu`,
    both work, verified by the differentiability tests in
    ``tests/test_integrator_worldline.py``) and safe under ``jax.jit``
    with both `dtau` and `mu` traced (`renorm_every` and `n_steps` must
    stay static -- e.g. via
    ``jax.jit(integrate_worldline, static_argnames=("renorm_every", "n_steps"))``,
    since they control Python-level control flow / array shapes, not
    array values).

    Parameters
    ----------
    field_fn : FieldFn
        ``pos -> (delta_E, grad_delta_E)``. `pos` here is a single
        position of shape ``(3,)`` (the per-step midpoint); see
        `FieldFn`.
    trajectory : jax.Array, shape (T, 3) or (3,)
        Positions, meters. If ``(3,)``, a static position broadcast over
        `n_steps` steps (`n_steps` is then required).
    dtau : float or jax.Array (scalar)
        Fixed step size ``dτ̃`` (Compton units, E9). No adaptive control
        (WP3 non-goal). May be a traced value under ``jax.jit``.
    mu : jax.Array, shape (3,)
        Effective dipole moment (E14a), C·m. May be a traced value under
        ``jax.jit`` (and is the ``jax.grad`` target in the E14a-coupling
        differentiability test).
    renorm_every : int, default DEFAULT_RENORM_EVERY
        Renormalize the rotor (E20) every this many steps. Static
        (a plain Python int, never traced -- see the jit-compatibility
        note above `DEFAULT_RENORM_EVERY`).
    n_steps : int, optional
        Number of steps to take; required (and only used) when
        `trajectory` is a static ``(3,)`` position. Static, like
        `renorm_every`.

    Returns
    -------
    WorldlineResult
    """
    dtau = jnp.asarray(dtau, dtype=jnp.float64)
    traj = _prepare_trajectory(trajectory, n_steps)
    n = traj.shape[0] - 1
    mu = jnp.asarray(mu, dtype=jnp.float64)
    dt_phys = dtau * TAU_COMPTON  # seconds per step (E9), for finite-difference velocity.

    r0 = jnp.zeros(16, dtype=jnp.float64).at[IDX_SCALAR].set(1.0)
    init = (
        r0,
        jnp.asarray(0.0, dtype=jnp.float64),  # phase: running sum
        jnp.asarray(0.0, dtype=jnp.float64),  # phase: Kahan compensation
        jnp.asarray(0.0, dtype=jnp.float64),  # phase_rotor: running sum
        jnp.asarray(0.0, dtype=jnp.float64),  # phase_rotor: Kahan compensation
        jnp.asarray(0.0, dtype=jnp.float64),  # max_norm_drift so far
        jnp.asarray(0, dtype=jnp.int64),  # step index (1-based after increment)
    )
    xs = (traj[:-1], traj[1:])

    def body(
        carry: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
        xs_t: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[
        tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
        None,
    ]:
        r, phase_sum, phase_c, rot_sum, rot_c, max_drift, step_idx = carry
        pos_a, pos_b = xs_t
        pos_mid = 0.5 * (pos_a + pos_b)
        v = (pos_b - pos_a) / dt_phys
        e_mid, grad_e_mid = field_fn(pos_mid)
        r_next, dphase = rotor_step(r, e_mid, grad_e_mid, v, mu, dtau)

        phase_sum, phase_c = _kahan_add(phase_sum, phase_c, dphase.scalar)
        rot_sum, rot_c = _kahan_add(rot_sum, rot_c, dphase.rotor)

        norm_err = jnp.abs(rotor_norm_sq(r_next) - 1.0)
        max_drift = jnp.maximum(max_drift, norm_err)

        step_idx = step_idx + 1
        do_renorm = (step_idx % renorm_every) == 0
        r_next = jax.lax.cond(do_renorm, normalize_rotor, lambda x: x, r_next)

        return (r_next, phase_sum, phase_c, rot_sum, rot_c, max_drift, step_idx), None

    (r_final, phase_sum, _pc, rot_sum, _rc, max_drift, _idx), _ = jax.lax.scan(body, init, xs)

    t_tilde = n * dtau
    fractional_shift = phase_sum / t_tilde
    norm_error = jnp.abs(rotor_norm_sq(r_final) - 1.0)

    return WorldlineResult(
        r_final=r_final,
        phase=phase_sum,
        phase_rotor=rot_sum,
        fractional_shift=fractional_shift,
        norm_error=norm_error,
        max_norm_drift=max_drift,
        n_steps=jnp.asarray(n, dtype=jnp.int64),
    )


def integrate_ensemble(
    field_fn: FieldFn,
    trajectories: jnp.ndarray,
    dtau: ArrayLike,
    mu: jnp.ndarray,
    *,
    renorm_every: int = DEFAULT_RENORM_EVERY,
    n_steps: int | None = None,
) -> EnsembleResult:
    """Integrate the rotor along a batch of worldlines.

    ``jax.vmap`` of :func:`integrate_worldline`, sharing `dtau`, `mu`, and
    `renorm_every` across the batch (only `trajectories` is vmapped over).
    Like `integrate_worldline`, safe under ``jax.jit`` with `dtau` and `mu`
    traced and `renorm_every`/`n_steps` static.

    Parameters
    ----------
    field_fn : FieldFn
        Shared field callable, see :func:`integrate_worldline`.
    trajectories : jax.Array, shape (M, T, 3)
        Positions for `M` worldlines, meters.
    dtau : float or jax.Array (scalar)
        Fixed step size ``dτ̃`` (Compton units), shared across the batch.
        May be a traced value under ``jax.jit``.
    mu : jax.Array, shape (3,)
        Effective dipole moment (E14a), C·m, shared across the batch. May
        be a traced value under ``jax.jit``.
    renorm_every : int, default DEFAULT_RENORM_EVERY
        Renormalize the rotor (E20) every this many steps. Static.
    n_steps : int, optional
        Forwarded to :func:`integrate_worldline` (only relevant for
        static per-trajectory positions, not used with ``(M, T, 3)``
        inputs). Static.

    Returns
    -------
    EnsembleResult
        Every :class:`WorldlineResult` field, stacked along a new
        leading ``(M,)`` axis.
    """
    trajectories = jnp.asarray(trajectories, dtype=jnp.float64)

    def run_one(traj: jnp.ndarray) -> WorldlineResult:
        return integrate_worldline(
            field_fn, traj, dtau, mu, renorm_every=renorm_every, n_steps=n_steps
        )

    batched = jax.vmap(run_one)(trajectories)
    return EnsembleResult(*batched)
