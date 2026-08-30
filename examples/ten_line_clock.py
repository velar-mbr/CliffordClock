#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The whole CliffordClock model, expanded from the ten-line listing in
``docs/MODEL.md`` into a runnable script. Two terms are enough to
reproduce a published result: gravitational redshift (``p = g h / c^2``)
and second-order Doppler (``p = -v^2 / 2c^2``). A small ensemble of atoms,
raised in height and sharing one draw of near-ground-state secular
velocities, reproduces the 33 cm case from Chou et al. 2010 (Science 329,
1630): predicted ``g*dh/c^2 = 3.6e-17``, measured ``(4.1 +/- 1.6)e-17``.
A second, 1 cm case is this script's own illustration of the same
formula at the 1e-18 scale; Chou et al. mention 1 cm only as a future
geodetic-resolution goal. Each case also prints the gap between the full
product and its first-order (additive) sum, the model's cross term, a
signed float alongside its analytic estimate. Plain NumPy only, seeded
and deterministic, runs in well under a second.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Term = Callable[[dict[str, float]], float]

STANDARD_GRAVITY = 9.80665  # m/s^2, exact by international definition
SPEED_OF_LIGHT = 299_792_458.0  # m/s, exact
# Sr-87 1S0-3P0 clock transition, Hz (this codebase's SR87 species record,
# src/cliffordclock/ensemble/species.py).
NU_0 = 429_228_004_229_873.4
RAMSEY_TIME_S = 1.0  # a full one-second interrogation, a real clock's own scale
N_ATOMS = 2000
N_STEPS = 5  # trapezoid nodes; state is constant per atom, so any count > 2 is exact


class Worldline:
    """A fixed height and velocity for the Ramsey period, on a time grid
    from 0 to ``T`` (``proper_time``)."""

    def __init__(self, height_m: float, speed_mps: float, proper_time: np.ndarray) -> None:
        self._state = {"height_m": height_m, "speed_mps": speed_mps}
        self.proper_time = proper_time

    def state(self, t: float) -> dict[str, float]:
        del t  # height and speed do not change over this Ramsey period
        return self._state


def analytic_grav_term(height_m: float) -> float:
    """``g*h/c^2``, closed-form gravitational redshift (CONVENTIONS.md E36)."""
    return STANDARD_GRAVITY * height_m / SPEED_OF_LIGHT**2


def p_gravity(state: dict[str, float]) -> float:
    return analytic_grav_term(state["height_m"])


def p_doppler(state: dict[str, float]) -> float:
    """Second-order Doppler pivot term: CONVENTIONS.md E21."""
    return -(state["speed_mps"] ** 2) / (2.0 * SPEED_OF_LIGHT**2)


def product_minus_one(state: dict[str, float], terms: list[Term]) -> float:
    """``prod(1 + p_k) - 1``, without ever forming ``1 + p_k`` as a
    float: each ``p_k`` is order ``1e-17`` or smaller, and adding it to
    ``1.0`` in float64 rounds away most of it. ``acc`` tracks ``(partial
    product) - 1`` directly; ``acc + p + acc*p`` is ``(1+acc)(1+p) - 1``.
    """
    acc = 0.0
    for p in terms:
        val = p(state)
        acc = acc + val + acc * val
    return acc


def clock_phase(worldline: Worldline, terms: list[Term], nu_0: float) -> float:
    """``docs/MODEL.md``'s ``clock_phase``, integrating the rate's excess
    over ``nu_0``: ``nu_0*2*pi*T`` is the same huge common phase for
    every atom, so it cancels out of ``shift`` and drops out of
    ``visibility``'s phasor length, keeping the arithmetic inside what
    float64 can represent.
    """
    t = worldline.proper_time
    rate_excess = np.array([nu_0 * product_minus_one(worldline.state(ti), terms) for ti in t])
    return float(2.0 * np.pi * np.trapezoid(rate_excess, t))


def compute_case(height_diff_m: float, rng: np.random.Generator) -> dict[str, float]:
    """A reference clock at height 0 against the same clock raised by
    ``height_diff_m``, sharing one draw of secular velocities so the
    Doppler term cancels in the difference and only height enters it."""
    speeds_mps = np.linalg.norm(rng.normal(0.0, 0.3, size=(N_ATOMS, 3)), axis=1)
    times = np.linspace(0.0, RAMSEY_TIME_S, N_STEPS)
    terms: list[Term] = [p_gravity, p_doppler]

    def ensemble_shift(height_m: float) -> tuple[float, float]:
        phases = np.array(
            [clock_phase(Worldline(height_m, v, times), terms, NU_0) for v in speeds_mps]
        )
        shift = float(np.mean(phases)) / (2.0 * np.pi * NU_0 * RAMSEY_TIME_S)
        return shift, abs(np.mean(np.exp(1j * phases)))

    shift_ref, _ = ensemble_shift(0.0)
    shift_raised, visibility = ensemble_shift(height_diff_m)
    composed = shift_raised - shift_ref

    mean_doppler = float(np.mean([p_doppler({"speed_mps": v}) for v in speeds_mps]))
    additive = analytic_grav_term(height_diff_m) - analytic_grav_term(0.0)
    analytic = analytic_grav_term(height_diff_m)
    return {
        "height_diff_m": height_diff_m,
        "analytic": analytic,
        "composed": composed,
        "additive": additive,
        "cross_term_gap": composed - additive,
        "cross_term_analytic": analytic * mean_doppler,
        "visibility": visibility,
    }


def print_case(case: dict[str, float]) -> None:
    print(f"\nHeight difference: {case['height_diff_m'] * 100:.0f} cm")
    print(f"  analytic g*dh/c^2              = {case['analytic']:.6e}")
    print(f"  composed (product) shift       = {case['composed']:.6e}")
    print(f"  additive (sum) shift           = {case['additive']:.6e}")
    print(f"  composed - additive, measured  = {case['cross_term_gap']:.3e}")
    print(f"  cross term, analytic estimate  = {case['cross_term_analytic']:.3e}")
    print(f"  ensemble visibility            = {case['visibility']:.9f}")


def main() -> None:
    rng = np.random.default_rng(seed=0)
    print("CliffordClock model, ten lines: raising a clock's gravitational")
    print("redshift, against Chou et al. 2010 (Science 329, 1630).")
    print_case(compute_case(0.33, rng))
    print("  Chou et al. 2010 measured (4.1 +/- 1.6)e-17 for this rise.")
    print("\n1 cm case: this script's own illustration of the same formula")
    print("at the 1e-18 scale. Chou et al. mention 1 cm only as a future")
    print("geodetic-resolution goal.")
    print_case(compute_case(0.01, rng))


if __name__ == "__main__":
    main()
