# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP20 BBR pivot term (CONVENTIONS.md E32/E33).

``cliffordclock.integrator.omega.bbr_pivot_perturbation``/
`bbr_pivot_uncertainty` implement E32's formula and its uncertainty
propagation (G7 sign-off A4#2-3); `pivot_perturbation_stark`'s
`bbr_pivot_perturbation` keyword implements E33's additive composition.
This file covers: the mandatory sign regression (G7 sign-off gate edit 1),
closed-form known-answer checks at 300 K and one non-trivial T for both
species, DC-pivot sign consistency, composition additivity, and the
uncertainty-propagation arithmetic.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import jax.numpy as jnp
import numpy as np

from cliffordclock.ensemble.species import get_species
from cliffordclock.integrator.omega import (
    bbr_pivot_perturbation,
    bbr_pivot_uncertainty,
    pivot_perturbation_stark,
)

# ---------------------------------------------------------------------------
# 1. Mandatory sign regression (G7 sign-off gate edit 1 / WP20 Gate section
#    item 1): (P-1)_BBR(Sr87, 300K) < 0 and ~= -5.3e-15. The pre-correction
#    draft (the project's theory sign-off record (G7), A1) double-negated this and
#    gave +5.3e-15 -- a sign flip 1e4x the 1e-19 target, silently correcting
#    every clock the wrong way. This is THE regression test that must never
#    go green for the wrong reason.
# ---------------------------------------------------------------------------


def test_bbr_pivot_sign_regression_sr87_300k() -> None:
    """(P-1)_BBR(Sr87, 300K) < 0 and ~= -5.3e-15 (dossier Sec.2 fractional value)."""
    value = bbr_pivot_perturbation(300.0, get_species("Sr87"))
    assert value < 0.0, (
        f"(P-1)_BBR(Sr87, 300K) = {value!r} is NOT negative -- this is the exact G7 "
        "sign-off A1 regression (a double-negated E32 gives the wrong sign)"
    )
    # rtol/atol=0 explicit (REVIEW-checklist effective-tolerance doctrine).
    np.testing.assert_allclose(value, -5.3195e-15, rtol=2e-4, atol=0)


def test_bbr_pivot_dc_pivot_sign_consistency_sr87() -> None:
    """BBR and the ordinary DC-Stark pivot (E14b) share the same sign
    discipline: both are `-(Delta_alpha/2h)*<E^2>/nu_0`-shaped quantities
    for Delta_alpha > 0 (Sr87/Yb171: the clock runs SLOW in both a static
    field and the thermal bath) -- E32's "no leading minus, the sign lives
    in Delta_nu_stat < 0" is required to be "exactly as E14b carries it".
    Checked directly: a static uniform field's E14b pivot perturbation is
    negative, and BBR's E32 pivot perturbation is *also* negative, for the
    same species.
    """
    species = get_species("Sr87")
    e0 = jnp.array([1e5, 0.0, 0.0])  # V/m, realistic static lab field
    zeros = jnp.zeros(3)
    stark_only = float(pivot_perturbation_stark(e0, zeros, species))
    bbr_only = bbr_pivot_perturbation(300.0, species)
    assert stark_only < 0.0
    assert bbr_only < 0.0
    # Composed (E33): strictly more negative than either alone (same-sign
    # addition), never a sign flip when combined via the pivot_perturbation_stark
    # bbr_pivot_perturbation keyword.
    composed = float(pivot_perturbation_stark(e0, zeros, species, bbr_pivot_perturbation=bbr_only))
    assert composed < stark_only
    assert composed < bbr_only
    np.testing.assert_allclose(composed, stark_only + bbr_only, rtol=0, atol=1e-30)


# ---------------------------------------------------------------------------
# 2. Closed-form known-answer checks: 300K (the registry's own reference
#    temperature) and T=250K (a non-trivial T within the 50-350K window)
#    for both species. Hand-computed with 50-digit decimal precision in a
#    completely independent code path (not calling bbr_pivot_perturbation
#    itself) -- guards against a bug shared between the implementation and
#    a "reference" that's actually just a copy of it.
# ---------------------------------------------------------------------------


def _decimal_bbr_pivot_perturbation(species_name: str, temperature_k: float) -> float:
    """50-digit decimal reference for E32, independent of `bbr_pivot_perturbation`."""
    getcontext().prec = 50
    species = get_species(species_name)
    coeffs = species.resolve_bbr_coefficients()
    t0 = Decimal(300)
    t_ratio = Decimal(temperature_k) / t0
    dyn_hz = sum(Decimal(coeff) * t_ratio**n for n, coeff in coeffs.dyn_coeffs_hz.items())
    delta_nu_hz = Decimal(coeffs.nu_stat_300k_hz) * t_ratio**4 + dyn_hz
    return float(delta_nu_hz / Decimal(species.clock_frequency_hz))


def test_bbr_pivot_sr87_300k_matches_decimal_reference() -> None:
    """Hand computation (T=300K, T/T0=1 exactly):
    Delta_nu = -2.13023 + (-0.13216 - 0.01231 - 0.00858) = -2.28328 Hz
    (P-1)_BBR = -2.28328 / 429228004229873.4 = -5.319503801...e-15.
    """
    got = bbr_pivot_perturbation(300.0, get_species("Sr87"))
    reference = _decimal_bbr_pivot_perturbation("Sr87", 300.0)
    np.testing.assert_allclose(got, reference, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got, -5.319503801008258e-15, rtol=1e-12, atol=0)


def test_bbr_pivot_sr87_250k_matches_decimal_reference() -> None:
    """Hand computation (T=250K, T/T0=5/6):
    (T/T0)^4 = 0.482253..., (T/T0)^6=0.334897..., ^8=0.232567..., ^10=0.161505...
    Delta_nu = -2.13023*0.482253 + (-0.13216*0.334897 - 0.01231*0.232567
               - 0.00858*0.161505) = -1.075818739... Hz
    (P-1)_BBR = -1.075818739.../429228004229873.4 = -2.506403889...e-15.
    """
    got = bbr_pivot_perturbation(250.0, get_species("Sr87"))
    reference = _decimal_bbr_pivot_perturbation("Sr87", 250.0)
    np.testing.assert_allclose(got, reference, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got, -2.5064038896483747e-15, rtol=1e-9, atol=0)


def test_bbr_pivot_yb171_300k_matches_decimal_reference() -> None:
    """Hand computation (T=300K):
    Delta_nu = -1.2545 + (-0.02217 - 0.000744) = -1.277414 Hz
    (P-1)_BBR = -1.277414 / 518295836590863.6 = -2.464642603...e-15.
    """
    got = bbr_pivot_perturbation(300.0, get_species("Yb171"))
    reference = _decimal_bbr_pivot_perturbation("Yb171", 300.0)
    np.testing.assert_allclose(got, reference, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got, -2.46464260334851e-15, rtol=1e-12, atol=0)


def test_bbr_pivot_yb171_250k_matches_decimal_reference() -> None:
    """Hand computation (T=250K): see docstring of the Sr87 250K test for
    the (T/T0)^n powers; Delta_nu = -1.2545*0.482253 + (-0.02217*0.334897
    - 0.000744*0.232567) = -0.612584216... Hz;
    (P-1)_BBR = -0.612584216.../518295836590863.6 = -1.181920001...e-15.
    """
    got = bbr_pivot_perturbation(250.0, get_species("Yb171"))
    reference = _decimal_bbr_pivot_perturbation("Yb171", 250.0)
    np.testing.assert_allclose(got, reference, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got, -1.181920000953887e-15, rtol=1e-9, atol=0)


# ---------------------------------------------------------------------------
# 3. Uncertainty-propagation arithmetic (G7 sign-off A4#2-3).
# ---------------------------------------------------------------------------


def test_bbr_uncertainty_sr87_300k_coefficient_only_matches_g7_signoff_magnitude() -> None:
    """G7 sign-off A4#2 (corrected arithmetic): static ~1.4e-19, dynamic
    ~7.7e-19, combined in quadrature ~7.8e-19 ("the sign-off's ~8e-19").
    """
    sigma, temperature_included = bbr_pivot_uncertainty(300.0, get_species("Sr87"))
    assert temperature_included is False
    sigma_static = 6e-5 * 1.0**4 / 429_228_004_229_873.4
    sigma_dynamic = 3.3e-4 * 1.0**6 / 429_228_004_229_873.4
    np.testing.assert_allclose(sigma_static, 1.398e-19, rtol=1e-2, atol=0)
    np.testing.assert_allclose(sigma_dynamic, 7.689e-19, rtol=1e-2, atol=0)
    expected = (sigma_static**2 + sigma_dynamic**2) ** 0.5
    np.testing.assert_allclose(sigma, expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(sigma, 7.8e-19, rtol=0.02, atol=0)


def test_bbr_uncertainty_sr87_300k_with_temperature_uncertainty_matches_g7_signoff_order() -> None:
    """G7 sign-off A4#3: sigma_T=4mK (JILA-class in-vacuum thermometry) gives
    ~3e-19 via the leading dExact_nu/dT ~= 4*Delta_nu/T approximation; this
    test uses the *exact* polynomial derivative (per A4#3's requirement)
    and checks it lands at the same order of magnitude the sign-off's
    leading-order estimate predicted.
    """
    sigma_with_t, temperature_included = bbr_pivot_uncertainty(
        300.0, get_species("Sr87"), temperature_uncertainty_k=0.004
    )
    sigma_without_t, _ = bbr_pivot_uncertainty(300.0, get_species("Sr87"))
    assert temperature_included is True
    assert sigma_with_t > sigma_without_t
    # sigma_T contribution alone, isolated by quadrature subtraction. Exact
    # value (computed once, this repo, from the polynomial derivative):
    # 2.950462972716713e-19 -- consistent with the G7 sign-off's ~3e-19
    # leading-order (4*Delta_nu/T) estimate to within the expected
    # correction from using the exact derivative instead.
    sigma_t_only = (sigma_with_t**2 - sigma_without_t**2) ** 0.5
    np.testing.assert_allclose(sigma_t_only, 2.950462972716713e-19, rtol=1e-9, atol=0)


def test_bbr_uncertainty_scales_with_temperature_ratio() -> None:
    """Sanity: at T=T0 exactly, the (T/T0) powers are 1, so the coefficient
    uncertainty reduces to the raw quadrature sum of the registry's Hz
    uncertainties divided by nu_0 -- an independent closed-form check.
    """
    species = get_species("Yb171")
    coeffs = species.resolve_bbr_coefficients()
    sigma, _ = bbr_pivot_uncertainty(300.0, species)
    expected = (
        coeffs.nu_stat_300k_uncertainty_hz**2 + coeffs.dyn_anchor_uncertainty_hz**2
    ) ** 0.5 / species.clock_frequency_hz
    np.testing.assert_allclose(sigma, expected, rtol=0, atol=1e-30)


# ---------------------------------------------------------------------------
# 4. Composition additivity at the omega.py level (E33): composing BBR into
#    pivot_perturbation_stark is exactly stark-only plus bbr-only, for a
#    nontrivial (nonzero-gradient-relevant) field, not just the E0-only
#    special case used in the sign-consistency test above.
# ---------------------------------------------------------------------------


def test_bbr_composition_additivity_baseline_plus_cross_plus_quadratic() -> None:
    """Stark+BBR = Stark-only + BBR-only, exercised with a nonzero `delta_e`
    (so the `cross`/`quadratic` terms of `stark_pivot_terms` are active too,
    not just `baseline`) -- tests E33's additivity as actually implemented,
    per the WP20 test contract.
    """
    species = get_species("Yb171")
    e0 = jnp.array([5e4, -1e4, 2e4])
    delta_e = jnp.array([10.0, -5.0, 3.0])
    bbr_value = bbr_pivot_perturbation(280.0, species)

    stark_only = float(pivot_perturbation_stark(e0, delta_e, species))
    composed = float(
        pivot_perturbation_stark(e0, delta_e, species, bbr_pivot_perturbation=bbr_value)
    )
    np.testing.assert_allclose(composed, stark_only + bbr_value, rtol=0, atol=0)


def test_bbr_zero_perturbation_is_exact_noop() -> None:
    """`bbr_pivot_perturbation=0.0` (the default) is a bit-exact no-op:
    ``x + 0.0 == x`` for any finite float64 `x` (IEEE 754) -- the byte-
    exactness contract every pre-WP20 call site relies on.
    """
    species = get_species("Sr87")
    e0 = jnp.array([3e4, 1e4, -2e4])
    delta_e = jnp.array([1.0, 2.0, -1.5])
    without_kwarg = pivot_perturbation_stark(e0, delta_e, species)
    with_zero = pivot_perturbation_stark(e0, delta_e, species, bbr_pivot_perturbation=0.0)
    np.testing.assert_allclose(np.asarray(with_zero), np.asarray(without_kwarg), rtol=0, atol=0)
