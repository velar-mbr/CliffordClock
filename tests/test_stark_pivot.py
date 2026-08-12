# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for cliffordclock.integrator.omega.pivot_perturbation_stark
(CONVENTIONS.md E14b, WP7).

Covers WP7 test contract items 1-4 (item 6, pipeline e2e, is deferred to
the follow-up plumbing WP; item 5, species data, lives in
tests/test_stark_species.py; item 7, unit hygiene, is enforced by the
tolerance/atol discipline used throughout this file and by the lint/type
commands the builder report records).
"""

from __future__ import annotations

import decimal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.constants import ELECTRON_MASS, PLANCK_H, SPEED_OF_LIGHT
from cliffordclock.ensemble.species import StarkCoefficients, get_species
from cliffordclock.integrator.omega import (
    pivot_perturbation,
    pivot_perturbation_stark,
    stark_pivot_terms,
)

_M_E_C2 = ELECTRON_MASS * SPEED_OF_LIGHT**2


# ---------------------------------------------------------------------------
# Item 1: formula pin.
# ---------------------------------------------------------------------------


def test_formula_pin_sr87_at_1000_v_per_m() -> None:
    """P - 1 for Sr87 at |E| = 1000 V/m equals the hand-computed
    -Delta_alpha |E|^2 / (2 h nu_0) to rtol 1e-12, atol=0 (WP7 test
    contract item 1).

    Physical interpretation: this is the fractional frequency shift a
    stationary 87Sr lattice-clock atom would show sitting in a uniform
    1000 V/m stray field with no field gradient at all -- a ~7e-15-level
    systematic, i.e. already above the clock's target 1e-18 floor by
    three orders of magnitude, illustrating why stray DC fields must be
    controlled at the sub-1000-V/m level (or characterized and
    subtracted) for a 1e-18 clock.
    """
    species = get_species("Sr87")
    assert species.delta_alpha_dc_si is not None
    e_mag = 1000.0
    e0 = jnp.array([e_mag, 0.0, 0.0])
    delta_e = jnp.zeros(3)

    expected = (
        -(species.delta_alpha_dc_si / 2.0) * e_mag**2 / (PLANCK_H * species.clock_frequency_hz)
    )
    got = pivot_perturbation_stark(e0, delta_e, species)
    np.testing.assert_allclose(float(got), expected, rtol=1e-12, atol=0)
    # Sanity: the documented physical interpretation above.
    assert -1e-14 < float(got) < -1e-15


def test_formula_pin_yb171_at_1000_v_per_m() -> None:
    """Same as above, Yb171, cross-checking the other populated species."""
    species = get_species("Yb171")
    assert species.delta_alpha_dc_si is not None
    e_mag = 1000.0
    e0 = jnp.array([0.0, e_mag, 0.0])
    delta_e = jnp.zeros(3)

    expected = (
        -(species.delta_alpha_dc_si / 2.0) * e_mag**2 / (PLANCK_H * species.clock_frequency_hz)
    )
    got = pivot_perturbation_stark(e0, delta_e, species)
    np.testing.assert_allclose(float(got), expected, rtol=1e-12, atol=0)


def test_dtype_is_float64() -> None:
    species = get_species("Sr87")
    e0 = jnp.array([1000.0, 0.0, 0.0])
    delta_e = jnp.zeros(3)
    assert pivot_perturbation_stark(e0, delta_e, species).dtype == jnp.float64


def test_batched() -> None:
    """pivot_perturbation_stark batches over a leading axis."""
    species = get_species("Sr87")
    e0 = jnp.broadcast_to(jnp.array([1000.0, 0.0, 0.0]), (5, 3))
    delta_e = jnp.stack([jnp.array([float(i), 0.0, 0.0]) for i in range(5)])
    got = pivot_perturbation_stark(e0, delta_e, species)
    assert got.shape == (5,)
    # delta_e = 0 (first row) reproduces the item-1 formula pin exactly.
    expected0 = (
        -(species.delta_alpha_dc_si / 2.0) * 1000.0**2 / (PLANCK_H * species.clock_frequency_hz)
    )
    np.testing.assert_allclose(float(got[0]), expected0, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Item 2: bridge identity.
# ---------------------------------------------------------------------------


def test_bridge_identity_linearized_e14b_matches_e14a_pivot() -> None:
    """Linearizing E14b about E0 (via jax.grad w.r.t. delta_e at
    delta_e=0) equals the *unmodified* E14a pivot_perturbation with
    mu_eff = -Delta_alpha * E0 and an h*nu_0 denominator, to rtol 1e-10
    (WP7 test contract item 2, CONVENTIONS.md E14b "Bridge identity").

    pivot_perturbation (E14a) is hardcoded to the m_e*c^2 denominator (by
    design -- WP7 does not touch it). To route the *unmodified* E14a
    function through an h*nu_0 denominator instead, this test rescales
    the dipole argument by (m_e c^2)/(h nu_0): since
    pivot_perturbation(delta_e, mu) = delta_e.mu / (m_e c^2), feeding it
    mu' = mu_eff * (m_e c^2)/(h nu_0) makes
    pivot_perturbation(delta_e, mu') = delta_e.mu_eff / (h nu_0) exactly
    -- CONVENTIONS.md E14b's own suggested technique ("Use this to give
    MVP linear tests physically meaningful mu values").
    """
    species = get_species("Sr87")
    assert species.delta_alpha_dc_si is not None
    e0 = jnp.array([3.0e4, -1.5e4, 7.0e3])  # V/m, an arbitrary nonzero bias field

    def pivot_stark_of_delta_e(delta_e: jnp.ndarray) -> jnp.ndarray:
        return pivot_perturbation_stark(e0, delta_e, species)

    # Directional-derivative-free: the full gradient at delta_e=0 IS the
    # first-order (linear-in-delta_e) coefficient, since E14b's |E|^2 is
    # an exactly quadratic form.
    grad_at_zero = jax.grad(pivot_stark_of_delta_e)(jnp.zeros(3))

    mu_eff = -species.delta_alpha_dc_si * e0  # C.m-equivalent vector, E14b bridge identity
    mu_for_pivot_perturbation = mu_eff * _M_E_C2 / (PLANCK_H * species.clock_frequency_hz)

    # A probe delta_e; the bridge identity is linear, so any nonzero probe
    # exercises the same comparison.
    probe_delta_e = jnp.array([12.3, -45.6, 78.9])
    got = jnp.sum(grad_at_zero * probe_delta_e)
    expected = pivot_perturbation(probe_delta_e, mu_for_pivot_perturbation)

    np.testing.assert_allclose(float(got), float(expected), rtol=1e-10, atol=0)


def test_bridge_identity_gradient_vector_matches_mu_eff_over_h_nu0() -> None:
    """Direct check of the gradient vector itself against mu_eff/(h nu_0)
    (CONVENTIONS.md E14b), independent of routing through
    pivot_perturbation -- a second, non-circular check of the same
    identity.
    """
    species = get_species("Yb171")
    assert species.delta_alpha_dc_si is not None
    e0 = jnp.array([5.0e4, 2.0e4, -3.0e4])

    grad_at_zero = jax.grad(lambda de: pivot_perturbation_stark(e0, de, species))(jnp.zeros(3))
    mu_eff = -species.delta_alpha_dc_si * e0
    expected = mu_eff / (PLANCK_H * species.clock_frequency_hz)

    np.testing.assert_allclose(np.asarray(grad_at_zero), np.asarray(expected), rtol=1e-10, atol=0)


# ---------------------------------------------------------------------------
# Item 3: cancellation discipline.
# ---------------------------------------------------------------------------


def _decimal_pivot_perturbation_stark(
    e0: jnp.ndarray, delta_e: jnp.ndarray, k_s: float, nu_0: float
) -> decimal.Decimal:
    """Independent 50-digit-precision reference for P-1 (E14b), computed
    by the same term-by-term decomposition but in `decimal.Decimal`
    arithmetic, which does not silently round at float64 precision.
    """
    decimal.getcontext().prec = 50
    dec = decimal.Decimal
    e0_dec = [dec(float(x)) for x in np.asarray(e0)]
    de_dec = [dec(float(x)) for x in np.asarray(delta_e)]
    e0_sq = sum((x * x for x in e0_dec), start=dec(0))
    cross = sum((a * b for a, b in zip(e0_dec, de_dec, strict=True)), start=dec(0))
    de_sq = sum((x * x for x in de_dec), start=dec(0))
    prefactor = dec(float(k_s)) / dec(float(nu_0))
    return prefactor * (e0_sq + 2 * cross + de_sq)


def test_cancellation_gradient_term_survives_at_1e19_shift_level() -> None:
    """BLOCKER-class regression per orchestrator instruction: with
    |E0| = 1e5 V/m and delta_e sized to produce a ~1e-19-level shift
    contribution, the gradient-driven (`cross`) term must not be rounded
    away (WP7 test contract item 3).

    This checks `stark_pivot_terms`'s `cross` output *directly* against a
    50-digit `decimal` reference, tight rtol, atol=0. It deliberately does
    **not** reconstruct the target quantity by subtracting two summed
    `pivot_perturbation_stark`/`P - 1` scalars (``float(got) -
    float(baseline_only)``): that pattern re-sums `cross` back into a
    near-unity-scale total before subtracting, i.e. it reintroduces
    exactly the catastrophic-cancellation antipattern this test exists to
    catch, and a naive ``|E0 + delta_e|**2`` implementation of
    `pivot_perturbation_stark` would pass a subtraction-based version of
    this test bit-identically (see the discrimination proof below, which
    demonstrates that failure mode explicitly and confirms this test does
    distinguish the correct term-by-term implementation from it).
    """
    species = get_species("Sr87")
    assert species.delta_alpha_dc_si is not None
    k_s = species.resolve_stark_coefficient_hz_per_v2_m2()
    nu_0 = species.clock_frequency_hz

    e0_mag = 1.0e5  # V/m
    # Solve analytically for the delta_e (along E0) that makes the cross
    # term ((k_s/nu_0) * 2 * e0_mag * delta_e_x) equal to a target
    # ~1e-19 P-1 contribution.
    target_gradient_shift = 1.0e-19
    delta_e_x = target_gradient_shift * nu_0 / (2.0 * k_s * e0_mag)

    e0 = jnp.array([e0_mag, 0.0, 0.0])
    delta_e = jnp.array([delta_e_x, 0.0, 0.0])

    decimal_cross_term = (
        decimal.Decimal(float(k_s))
        / decimal.Decimal(float(nu_0))
        * 2
        * decimal.Decimal(float(e0_mag))
        * decimal.Decimal(float(delta_e_x))
    )
    # The reference itself should actually land near the target 1e-19
    # magnitude, or the test isn't exercising what it claims to. abs=0:
    # the default abs=1e-12 would dominate this ~1e-19-magnitude
    # comparison and make the check vacuous.
    assert abs(float(decimal_cross_term)) == pytest.approx(target_gradient_shift, rel=0.05, abs=0)

    # Full-sum sanity check (item 3 also covers the summed pivot).
    decimal_reference = _decimal_pivot_perturbation_stark(e0, delta_e, k_s, nu_0)
    got = pivot_perturbation_stark(e0, delta_e, species)
    np.testing.assert_allclose(float(got), float(decimal_reference), rtol=1e-9, atol=0)

    # The load-bearing check: the decomposed `cross` term in isolation,
    # against the decimal reference, tight rtol, atol=0.
    _, cross, _ = stark_pivot_terms(e0, delta_e, species)
    assert float(cross) != 0.0, (
        "gradient-driven Stark term collapsed to zero (catastrophic cancellation regression, E10)"
    )
    np.testing.assert_allclose(float(cross), float(decimal_cross_term), rtol=1e-9, atol=0)

    # --- Discrimination proof -------------------------------------------
    # A naive implementation that never keeps the cross term as an
    # independently-rounded product -- instead forming the combined total
    # field |E0 + delta_e|^2 and subtracting a separately-computed
    # baseline |E0|^2 to "extract" the cross contribution -- is exactly
    # the pattern stark_pivot_terms's decomposition exists to avoid. It
    # must measurably fail the same tight tolerance used above, proving
    # this test discriminates the correct term-by-term implementation
    # from the naive collapsed-then-subtracted one (rather than both
    # passing, which was the BLOCKER this test replaces).
    prefactor = k_s / nu_0
    naive_full = prefactor * (e0_mag + delta_e_x) ** 2
    naive_baseline = prefactor * e0_mag**2
    naive_cross = naive_full - naive_baseline

    naive_relative_error = abs(naive_cross - float(decimal_cross_term)) / abs(
        float(decimal_cross_term)
    )
    documented_bound = 1e-8  # empirically ~3e-8 at this magnitude; see companion test below
    assert naive_relative_error > documented_bound, (
        "naive combined-square-then-subtract failed to demonstrate degradation "
        "at this magnitude -- the discrimination proof is not exercising the "
        "failure mode it claims to, so it can no longer prove this test tells "
        "correct from naive"
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(naive_cross, float(decimal_cross_term), rtol=1e-9, atol=0)


def test_naive_combined_square_loses_precision_at_the_same_magnitude() -> None:
    """Standalone companion to the discrimination proof embedded in
    :func:`test_cancellation_gradient_term_survives_at_1e19_shift_level`
    above: demonstrates *why* the term-by-term decomposition in
    `stark_pivot_terms` matters, quantitatively, on its own. Literally
    forming ``|E0 + delta_e|^2`` (add the vectors, then square) and
    subtracting two independently-computed pivot values loses several
    significant digits of the ~1e-19-level cross term at this same
    |E0| = 1e5 V/m regime, while `stark_pivot_terms`'s term-by-term
    evaluation (checked above) matches the decimal reference to ~1e-16
    relative -- i.e. the same regime where the naive pattern the module
    explicitly avoids would already be measurably degraded. This test
    does not exercise the module under test; it exercises the failure
    mode documented in `stark_pivot_terms`'s docstring, in plain float64
    Python arithmetic.
    """
    species = get_species("Sr87")
    assert species.delta_alpha_dc_si is not None
    k_s = species.resolve_stark_coefficient_hz_per_v2_m2()
    nu_0 = species.clock_frequency_hz

    e0_mag = 1.0e5
    target_gradient_shift = 1.0e-19
    delta_e_x = target_gradient_shift * nu_0 / (2.0 * k_s * e0_mag)

    prefactor = k_s / nu_0
    naive_full = prefactor * (e0_mag + delta_e_x) ** 2
    naive_baseline = prefactor * e0_mag**2
    naive_gradient = naive_full - naive_baseline

    decimal_gradient_term = float(
        decimal.Decimal(float(k_s))
        / decimal.Decimal(float(nu_0))
        * 2
        * decimal.Decimal(float(e0_mag))
        * decimal.Decimal(float(delta_e_x))
    )

    naive_relative_error = abs(naive_gradient - decimal_gradient_term) / abs(decimal_gradient_term)
    # The naive pattern is measurably degraded (empirically ~3e-8 relative
    # error at this magnitude) -- far above float64's ~1e-16 floor.
    assert naive_relative_error > 1e-8


# ---------------------------------------------------------------------------
# Item 4: quadratic scaling.
# ---------------------------------------------------------------------------


def test_quadratic_scaling_across_three_decades() -> None:
    """Shift scales as |E|^2 across 3 decades of field (WP7 test contract
    item 4): P-1 at field magnitude 10x is exactly 100x P-1 at the
    smaller magnitude (uniform field, delta_e=0, so this isolates the
    baseline |E0|^2 term).
    """
    species = get_species("Sr87")
    magnitudes = [1.0e2, 1.0e3, 1.0e4, 1.0e5]  # V/m, 3 decades
    shifts = []
    for mag in magnitudes:
        e0 = jnp.array([mag, 0.0, 0.0])
        shifts.append(float(pivot_perturbation_stark(e0, jnp.zeros(3), species)))

    for i in range(len(magnitudes) - 1):
        ratio_field = magnitudes[i + 1] / magnitudes[i]
        ratio_shift = shifts[i + 1] / shifts[i]
        np.testing.assert_allclose(ratio_shift, ratio_field**2, rtol=1e-12, atol=0)


def test_quadratic_scaling_with_direction_change() -> None:
    """Scaling holds when the field direction (not just magnitude) is
    varied, confirming |E|^2 = E.E rather than a component-wise formula.
    """
    species = get_species("Yb171")
    base = jnp.array([3.0, -4.0, 0.0])  # |base| = 5
    for scale in (1e2, 1e3, 1e4):
        e0 = base * scale
        got = float(pivot_perturbation_stark(e0, jnp.zeros(3), species))
        expected = (
            -(species.delta_alpha_dc_si / 2.0)
            * jnp.sum(e0 * e0)
            / (PLANCK_H * species.clock_frequency_hz)
        )
        np.testing.assert_allclose(got, float(expected), rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# jit/vmap-safety.
# ---------------------------------------------------------------------------


def test_jit_safe() -> None:
    species = get_species("Sr87")
    e0 = jnp.array([1000.0, 0.0, 0.0])
    delta_e = jnp.array([1.0, -2.0, 0.5])

    jitted = jax.jit(lambda e0_, de_: pivot_perturbation_stark(e0_, de_, species))
    eager = pivot_perturbation_stark(e0, delta_e, species)
    np.testing.assert_allclose(float(jitted(e0, delta_e)), float(eager), rtol=1e-14, atol=0)


def test_vmap_safe() -> None:
    species = get_species("Sr87")
    e0 = jnp.array([[1000.0, 0.0, 0.0], [0.0, 2000.0, 0.0], [0.0, 0.0, 500.0]])
    delta_e = jnp.zeros((3, 3))

    vmapped = jax.vmap(lambda e0_, de_: pivot_perturbation_stark(e0_, de_, species))(e0, delta_e)
    direct = pivot_perturbation_stark(e0, delta_e, species)
    np.testing.assert_allclose(np.asarray(vmapped), np.asarray(direct), rtol=1e-14, atol=0)


def test_grad_safe() -> None:
    """grad w.r.t. delta_e is well-defined and finite (used directly by
    the bridge-identity tests above; this test isolates the grad-safety
    requirement on its own)."""
    species = get_species("Sr87")
    e0 = jnp.array([1000.0, -500.0, 250.0])

    grad_fn = jax.grad(lambda de: pivot_perturbation_stark(e0, de, species))
    g = grad_fn(jnp.array([1.0, 1.0, 1.0]))
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# StarkCoefficients override path (exercises the "species_or_coeffs" union).
# ---------------------------------------------------------------------------


def test_explicit_stark_coefficients_override_matches_species_path() -> None:
    """pivot_perturbation_stark accepts a StarkCoefficients override that
    reproduces the species-registry result exactly, when constructed from
    the same underlying numbers -- this is the mechanism the deferred
    pipeline plumbing (config `coupling: {type: stark_dc, ...}` override
    fields) will drive.
    """
    species = get_species("Sr87")
    assert species.delta_alpha_dc_si is not None
    coeffs = StarkCoefficients(
        clock_frequency_hz=species.clock_frequency_hz,
        delta_alpha_dc_si=species.delta_alpha_dc_si,
    )
    e0 = jnp.array([1000.0, 0.0, 0.0])
    delta_e = jnp.array([0.1, 0.2, -0.3])

    from_species = pivot_perturbation_stark(e0, delta_e, species)
    from_coeffs = pivot_perturbation_stark(e0, delta_e, coeffs)
    np.testing.assert_allclose(float(from_species), float(from_coeffs), rtol=1e-14, atol=0)


def test_explicit_stark_coefficients_override_for_unregistered_species() -> None:
    """StarkCoefficients works standalone for a species/value not backed
    by any registry lookup at all, demonstrating the documented override
    path for Species.resolve_stark_coefficient_hz_per_v2_m2's ValueError
    (which every currently-registered species now avoids -- WP21 added
    Delta_alpha for Al27+/In115+ -- but the override path itself is
    general and still exercised here with an illustrative value).
    """
    hypothetical_ion_alpha = 1.0e-40  # illustrative, not a cited literature value
    coeffs = StarkCoefficients(
        clock_frequency_hz=1_121_015_393_207_857.4,
        delta_alpha_dc_si=hypothetical_ion_alpha,
    )
    e0 = jnp.array([1000.0, 0.0, 0.0])
    got = pivot_perturbation_stark(e0, jnp.zeros(3), coeffs)
    expected = -(hypothetical_ion_alpha / 2.0) * 1000.0**2 / (PLANCK_H * coeffs.clock_frequency_hz)
    np.testing.assert_allclose(float(got), expected, rtol=1e-12, atol=0)
