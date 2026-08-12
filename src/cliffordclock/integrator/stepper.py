# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single-step rotor kernel (CONVENTIONS.md E17, E19, E21, E24).

Design order: exponential midpoint (E19, G0 item 6 confirmed), order 2 --
the field/velocity are evaluated at the *midpoint* of each ``dτ̃`` step, not
its left endpoint, which is what makes the scheme order 2 rather than
order 1 (verified empirically by the convergence-order test in
``tests/test_integrator_stepper.py``, since CONVENTIONS.md E19 is itself
marked ``[INTERPRETATION]`` on this point). The midpoint field/velocity
sample is the caller's responsibility (``worldline.py``); this module's
:func:`rotor_step` takes a single already-midpoint-evaluated
``(delta_e, grad_delta_e, v)`` triple per call.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from cliffordclock.cl13 import IDX_E12, IDX_SCALAR, exp_bivector, geometric_product
from cliffordclock.integrator.omega import build_omega


class PhaseIncrement(NamedTuple):
    """One step's perturbation-phase increment, both accumulation pipelines.

    CONVENTIONS.md G0 item 3 (E18/E21/E24 resolution): the scalar E21
    pipeline is the *primary* observable; the rotor pipeline is the
    geometric integrator and a standing cross-check (E24). Both are
    computed per step so callers can accumulate (and compare) both.

    Attributes
    ----------
    scalar : jax.Array, shape (...,)
        Primary E21/E22 perturbation phase increment,
        ``δω̃(mid) · dτ̃``, dimensionless.
    rotor : jax.Array, shape (...,)
        E24 rotor-extracted phase increment for this step (see
        :func:`rotor_plane_angle`), dimensionless. Equals `scalar` to
        machine precision whenever the step's ``Ω`` has no boost
        component (pure ``B̂_C``-plane rotation); diverges from it at
        ``O(ω_boost²)`` otherwise (E24 acceptance criterion).
    """

    scalar: jnp.ndarray
    rotor: jnp.ndarray


def rotor_plane_angle(r: jnp.ndarray) -> jnp.ndarray:
    """Rotation angle of a rotor in the fixed ``B̂_C = e_1 ∧ e_2`` plane (E24).

    For a rotor confined to the ``B̂_C`` plane, ``r = exp(b·e_12) =
    cos(b) + sin(b)·e_12`` exactly (E6), so this recovers ``−2b`` exactly
    (matching the sign/scale convention of E17's ``dR/dτ̃ = −½ Ω R``, so
    that applied to a single-step ``exp(−½ Ω_e12 dτ̃ · e_12)`` factor it
    reproduces ``Ω_e12 · dτ̃`` -- the same quantity as
    :func:`~cliffordclock.integrator.omega.scalar_rate_perturbation` times
    ``dτ̃``, by construction of E18's resolved reading). When `r` carries
    components outside the ``B̂_C`` plane (from a nonzero ``ω_boost``,
    E18), this ``atan2``-based extraction only reads off the *effective*
    ``B̂_C``-plane rotation, which is the E24 cross-check quantity: it
    agrees with the scalar E21 pipeline at first order in ``ω_boost`` and
    diverges at ``O(ω_boost²)`` (E24 acceptance criterion).

    Parameters
    ----------
    r : jax.Array, shape (..., 16)
        A rotor (or rotor-like multivector), typically a single step's
        ``exp(−½ Ω dτ̃)`` factor.

    Returns
    -------
    jax.Array, shape (...,)
        Angle, radians (dimensionless in Compton units).
    """
    r = jnp.asarray(r, dtype=jnp.float64)
    return -2.0 * jnp.arctan2(r[..., IDX_E12], r[..., IDX_SCALAR])


def rotor_step(
    r: jnp.ndarray,
    delta_e: jnp.ndarray,
    grad_delta_e: jnp.ndarray,
    v: jnp.ndarray,
    mu: jnp.ndarray,
    dtau: ArrayLike,
) -> tuple[jnp.ndarray, PhaseIncrement]:
    """Advance a rotor by one ``dτ̃`` step (E17, E19 exponential midpoint).

    1. Build ``Ω`` (E18) from the (caller-supplied, already
       midpoint-evaluated) field/velocity triple.
    2. ``R_next = exp(−½ Ω dτ̃) · R`` (E19; :func:`~cliffordclock.cl13.exp_bivector`
       from WP1).
    3. Extract both the primary scalar (E21/E22) and rotor-extracted
       (E24) phase increments for this step.

    Renormalization (E20) is *not* applied here: it is a periodic
    (every ``renorm_every`` steps), stateful decision that only makes
    sense across a sequence of steps, so it lives in
    ``worldline.py``'s scan body, not in this stateless single-step
    primitive.

    Parameters
    ----------
    r : jax.Array, shape (..., 16)
        Current rotor.
    delta_e : jax.Array, shape (..., 3)
        Perturbation field ``δE`` at the step's midpoint evaluation
        point (E19), V/m.
    grad_delta_e : jax.Array, shape (..., 3, 3)
        Gradient tensor at the same point, ``[..., i, j] = ∂_i δE_j``
        (E13), V/m^2.
    v : jax.Array, shape (..., 3)
        Velocity at the same point, m/s.
    mu : jax.Array, shape (3,)
        Effective dipole moment (E14a), C·m.
    dtau : float or jax.Array (scalar)
        Step size ``dτ̃``, dimensionless (Compton units, E9). May be a
        traced value under ``jax.jit``.

    Returns
    -------
    r_next : jax.Array, shape (..., 16)
        The advanced rotor.
    dphase : PhaseIncrement
        This step's perturbation phase increment (both pipelines).
    """
    r = jnp.asarray(r, dtype=jnp.float64)
    dtau = jnp.asarray(dtau, dtype=jnp.float64)
    omega = build_omega(delta_e, grad_delta_e, v, mu)
    generator = (-0.5 * dtau) * omega
    delta_r = exp_bivector(generator)
    r_next = geometric_product(delta_r, r)

    # NIT 9: build_omega's IDX_E12 component *is* scalar_rate_perturbation's
    # value by construction (E18: Omega's B_hat_C coefficient is exactly the
    # E21 rotation_coeff), so it is read off `omega` here rather than
    # recomputed by a second scalar_rate_perturbation call -- avoids the
    # duplicate pivot/kinematic computation while staying exact.
    dphase_scalar = omega[..., IDX_E12] * dtau
    dphase_rotor = rotor_plane_angle(delta_r)
    return r_next, PhaseIncrement(scalar=dphase_scalar, rotor=dphase_rotor)
