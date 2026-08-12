# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compton-scaled rotor path integrator.

Integrates the rotor evolution equation
``dR(τ̃)/dτ̃ = −½ Ω(r(τ̃)) R(τ̃)`` (CONVENTIONS.md E17) along worldlines,
non-dimensionalized by the electron Compton duration. See
``docs/CONVENTIONS.md`` for the governing equations (E14a, E15-E24).

- :mod:`cliffordclock.integrator.omega` -- interaction bivector ``Ω``
  (E14a, E16, E18) and the scalar observable (E21).
- :mod:`cliffordclock.integrator.stepper` -- single-step exponential
  midpoint kernel (E17, E19, E24).
- :mod:`cliffordclock.integrator.worldline` -- ``lax.scan``/``vmap``
  worldline and ensemble integration (E9, E20, E22, E23).
"""

from cliffordclock.integrator.omega import (
    build_omega,
    pivot,
    pivot_perturbation,
    scalar_rate_perturbation,
    spin_connection,
)
from cliffordclock.integrator.stepper import PhaseIncrement, rotor_plane_angle, rotor_step
from cliffordclock.integrator.worldline import (
    DEFAULT_RENORM_EVERY,
    EnsembleResult,
    FieldFn,
    WorldlineResult,
    integrate_ensemble,
    integrate_worldline,
    kahan_sum,
)

__all__ = [
    "DEFAULT_RENORM_EVERY",
    "EnsembleResult",
    "FieldFn",
    "PhaseIncrement",
    "WorldlineResult",
    "build_omega",
    "integrate_ensemble",
    "integrate_worldline",
    "kahan_sum",
    "pivot",
    "pivot_perturbation",
    "rotor_plane_angle",
    "rotor_step",
    "scalar_rate_perturbation",
    "spin_connection",
]
