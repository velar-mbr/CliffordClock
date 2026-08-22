# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP29 Tier 1 multi-surface BBR thermal environment (CONVENTIONS.md E37).

Covers: the bit-exact reduction of the multi-surface functions to the
existing single-temperature E32 path for a uniform environment, a
two-surface toy case checked against an independent `decimal`-precision
reference (mirroring `tests/test_bbr_pivot.py`'s pattern), weight-
normalization/validity-window/emissivity-range rejection, PTB's published
enclosure-and-apertures emissivity closed form (two- and three-surface
cases, each checked against an independent reference built directly from
PTB's own formula), the independent-vs-correlated uncertainty combination
modes, and pipeline-level config parsing/threading. Several kill-test
comments below spell out exactly which implementation bug each
discriminating assertion would catch, including the G10-flagged
enclosure/aperture weight bug (an earlier renormalize-against-the-total
scheme that did not reduce to PTB's own two-surface formula).
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import pytest

from cliffordclock.ensemble.species import get_species
from cliffordclock.integrator.omega import (
    BBR_ENVIRONMENT_WEIGHT_TOLERANCE,
    RadiationSurface,
    bbr_environment_effective_temperatures,
    bbr_environment_pivot_perturbation,
    bbr_environment_pivot_uncertainty,
    bbr_pivot_perturbation,
    bbr_pivot_uncertainty,
)
from cliffordclock.pipeline import (
    EnvironmentConfig,
    PipelineConfig,
    PipelineConfigError,
    RadiationEnvironmentConfig,
    _parse_environment,
    run_pipeline_full,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Independent decimal-precision reference (not calling any function under
# test), mirroring `tests/test_bbr_pivot.py::_decimal_bbr_pivot_perturbation`.
# `surfaces` is a list of `(weight, temperature_k)` with NO emissivity
# correction: this reference is only valid for plain solid-angle-weighted
# environments. The PTB enclosure-and-apertures topology (section 4 below)
# has its own, separate independent reference, built directly from PTB's
# published closed form, not from this function and not from
# `cliffordclock.integrator.omega._bbr_effective_weights`; see that
# section's comment for why a shared helper here would risk baking a
# shared bug into both the implementation and its own test.
# ---------------------------------------------------------------------------


def _decimal_bbr_environment_pivot_perturbation(
    species_name: str, surfaces: list[tuple[float, float]]
) -> float:
    getcontext().prec = 50
    species = get_species(species_name)
    coeffs = species.resolve_bbr_coefficients()
    t0 = Decimal(300)

    weights = [Decimal(weight) for weight, _temperature in surfaces]

    powers = sorted({4, 6} | set(coeffs.dyn_coeffs_hz.keys()))
    moments = {}
    for n in powers:
        acc = Decimal(0)
        for weight, (_weight, temperature) in zip(weights, surfaces, strict=True):
            acc += weight * (Decimal(temperature) / t0) ** n
        moments[n] = acc

    dyn_hz = Decimal(0)
    for n, coeff in coeffs.dyn_coeffs_hz.items():
        dyn_hz += Decimal(coeff) * moments[n]
    delta_nu_hz = Decimal(coeffs.nu_stat_300k_hz) * moments[4] + dyn_hz
    return float(delta_nu_hz / Decimal(species.clock_frequency_hz))


# ---------------------------------------------------------------------------
# 1. Bit-exact reduction to the single-temperature E32 path (CONVENTIONS.md
#    E37's "exact reduction" property): a uniform, single-surface
#    environment must equal `bbr_pivot_perturbation`/`bbr_pivot_uncertainty`
#    bit for bit, an exact match, not just a numerically close one; both
#    functions share `_bbr_weighted_moments`, so this is a structural
#    guarantee.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("temperature_k", [300.0, 250.0, 77.0, 340.0])
@pytest.mark.parametrize("species_name", ["Sr87", "Yb171"])
def test_uniform_environment_reduces_bitwise_to_scalar_perturbation(
    species_name: str, temperature_k: float
) -> None:
    species = get_species(species_name)
    scalar_value = bbr_pivot_perturbation(temperature_k, species)
    env_value = bbr_environment_pivot_perturbation(
        (RadiationSurface(name="uniform", weight=1.0, temperature_k=temperature_k),), species
    )
    # rtol=atol=0: this must be bit-for-bit identical, an exact match.
    np.testing.assert_allclose(env_value, scalar_value, rtol=0, atol=0)
    assert env_value == scalar_value


def test_uniform_environment_reduces_bitwise_to_scalar_uncertainty_no_temperature_sigma() -> None:
    species = get_species("Sr87")
    scalar_sigma, scalar_included = bbr_pivot_uncertainty(300.0, species)
    env_sigma, env_included = bbr_environment_pivot_uncertainty(
        (RadiationSurface(name="uniform", weight=1.0, temperature_k=300.0),), species
    )
    assert env_included is False
    assert scalar_included is False
    np.testing.assert_allclose(env_sigma, scalar_sigma, rtol=0, atol=0)


def test_uniform_environment_reduces_bitwise_to_scalar_uncertainty_with_temperature_sigma() -> None:
    species = get_species("Sr87")
    scalar_sigma, scalar_included = bbr_pivot_uncertainty(
        300.0, species, temperature_uncertainty_k=0.004
    )
    env_sigma, env_included = bbr_environment_pivot_uncertainty(
        (
            RadiationSurface(
                name="uniform", weight=1.0, temperature_k=300.0, temperature_uncertainty_k=0.004
            ),
        ),
        species,
    )
    assert env_included is True
    assert scalar_included is True
    np.testing.assert_allclose(env_sigma, scalar_sigma, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 2. Two-surface toy vs. the independent decimal reference (dossier part C's
#    "Field deployment, 40K spread" row: f=0.5, T1=280K, T2=320K).
#
#    Hand computation (Sr87, T0=300K):
#      M4 = 0.5*(280/300)^4 + 0.5*(320/300)^4 = 0.5*0.752915... + 0.5*1.301324...
#         = 1.026686...
#      M6, M8, M10 computed the same way with powers 6/8/10.
#      Delta_nu = -2.13023*M4 + (-0.13216*M6 - 0.01231*M8 - 0.00858*M10)
#      (P-1)_BBR = Delta_nu / 429228004229873.4
#    matching the dossier's T_eff,4 ~= 302.0K for this row (this test's own
#    T_eff,4, computed independently below, lands at ~301.98K, consistent).
# ---------------------------------------------------------------------------


def test_two_surface_toy_matches_decimal_reference() -> None:
    species = get_species("Sr87")
    surfaces = (
        RadiationSurface(name="cold_half", weight=0.5, temperature_k=280.0),
        RadiationSurface(name="warm_half", weight=0.5, temperature_k=320.0),
    )
    got = bbr_environment_pivot_perturbation(surfaces, species)
    reference = _decimal_bbr_environment_pivot_perturbation("Sr87", [(0.5, 280.0), (0.5, 320.0)])
    np.testing.assert_allclose(got, reference, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got, -5.480254637657586e-15, rtol=1e-12, atol=0)


def test_two_surface_toy_effective_temperatures_diverge_by_moment() -> None:
    """T_eff,4 != T_eff,6 != T_eff,8 != T_eff,10 for this non-uniform
    environment (CONVENTIONS.md E37's "the report should expose these so a
    user sees T_eff,4 vs T_eff,6 diverge" requirement); a uniform
    environment would give the SAME value for every n (see the reduction
    test below).
    """
    species = get_species("Sr87")
    surfaces = (
        RadiationSurface(name="cold_half", weight=0.5, temperature_k=280.0),
        RadiationSurface(name="warm_half", weight=0.5, temperature_k=320.0),
    )
    t_eff = bbr_environment_effective_temperatures(surfaces, species)
    assert set(t_eff.keys()) == {4, 6, 8, 10}
    np.testing.assert_allclose(t_eff[4], 301.9817580353671, rtol=1e-12, atol=0)
    np.testing.assert_allclose(t_eff[6], 303.2583851611937, rtol=1e-12, atol=0)
    np.testing.assert_allclose(t_eff[8], 304.47759354804356, rtol=1e-12, atol=0)
    np.testing.assert_allclose(t_eff[10], 305.62572907768225, rtol=1e-12, atol=0)
    assert t_eff[4] < t_eff[6] < t_eff[8] < t_eff[10]


def test_uniform_environment_effective_temperatures_all_equal() -> None:
    species = get_species("Sr87")
    surfaces = (RadiationSurface(name="uniform", weight=1.0, temperature_k=300.0),)
    t_eff = bbr_environment_effective_temperatures(surfaces, species)
    for value in t_eff.values():
        np.testing.assert_allclose(value, 300.0, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. Weight-normalization and validity-window violations raise.
# ---------------------------------------------------------------------------


def test_unnormalized_weights_raise() -> None:
    species = get_species("Sr87")
    surfaces = (
        RadiationSurface(name="a", weight=0.5, temperature_k=300.0),
        RadiationSurface(name="b", weight=0.6, temperature_k=300.0),
    )
    with pytest.raises(ValueError, match="must sum to 1"):
        bbr_environment_pivot_perturbation(surfaces, species)


def test_weights_within_tolerance_do_not_raise() -> None:
    """A sum within `BBR_ENVIRONMENT_WEIGHT_TOLERANCE` (1e-9) of 1 is accepted."""
    species = get_species("Sr87")
    nudged_weight = 0.5 + BBR_ENVIRONMENT_WEIGHT_TOLERANCE / 2
    surfaces = (
        RadiationSurface(name="a", weight=0.5, temperature_k=300.0),
        RadiationSurface(name="b", weight=nudged_weight, temperature_k=300.0),
    )
    bbr_environment_pivot_perturbation(surfaces, species)  # must not raise


def test_empty_surfaces_raise() -> None:
    species = get_species("Sr87")
    with pytest.raises(ValueError, match="at least one"):
        bbr_environment_pivot_perturbation((), species)


def test_surface_temperature_outside_validity_window_raises() -> None:
    species = get_species("Sr87")
    surfaces = (
        RadiationSurface(name="a", weight=0.5, temperature_k=300.0),
        RadiationSurface(name="b", weight=0.5, temperature_k=400.0),  # > 350 K
    )
    with pytest.raises(ValueError, match="outside the validated BBR fit range"):
        bbr_environment_pivot_perturbation(surfaces, species)


def test_negative_temperature_uncertainty_raises() -> None:
    species = get_species("Sr87")
    surfaces = (
        RadiationSurface(
            name="a", weight=1.0, temperature_k=300.0, temperature_uncertainty_k=-0.01
        ),
    )
    with pytest.raises(ValueError, match="must be >= 0"):
        bbr_environment_pivot_perturbation(surfaces, species)


def test_emissivity_out_of_range_raises() -> None:
    species = get_species("Sr87")
    surfaces = (RadiationSurface(name="a", weight=1.0, temperature_k=300.0, emissivity=1.5),)
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\]"):
        bbr_environment_pivot_perturbation(surfaces, species)


# ---------------------------------------------------------------------------
# 4. PTB enclosure-and-apertures emissivity form (CONVENTIONS.md E37,
#    Nosske et al. arXiv:2507.14030) against hand-computed cases, checked
#    against PTB's own published closed form evaluated independently here
#    (NOT by calling `cliffordclock.integrator.omega._bbr_effective_weights`
#    or reusing `_decimal_bbr_environment_pivot_perturbation`'s no-
#    emissivity-only code path, since either would risk baking a shared
#    bug into both the implementation and its own test, exactly
#    the class of bug the G10 review caught in an earlier version of this
#    function, which renormalized every surface's corrected weight against
#    the total instead of giving the enclosure the complement; see the
#    kill-test at the end of this section for the concrete numbers).
#
#    Two-surface hand computation (one enclosure, one aperture): PTB's
#    published closed form for a single aperture of raw weight w seen
#    through an enclosure of interior emissivity eps is
#    w_eff = w / (w + (1 - w) * eps). For w=0.1, eps=0.5:
#      w_eff_aperture = 0.1 / (0.1 + 0.9*0.5) = 0.1/0.55 = 2/11 = 0.181818...
#      w_eff_enclosure = 1 - 2/11 = 9/11 = 0.818182...
#    (the enclosure gets the complement directly, never a renormalized
#    share of its own raw weight against the aperture's corrected one).
# ---------------------------------------------------------------------------


def test_ptb_two_surface_enclosure_aperture_matches_published_closed_form() -> None:
    species = get_species("Sr87")
    t0 = Decimal(300)

    # PTB's own closed form, coded directly from the paper's formula, not
    # from any function under test.
    w_aperture, epsilon = Decimal("0.1"), Decimal("0.5")
    w_aperture_eff = w_aperture / (w_aperture + (Decimal(1) - w_aperture) * epsilon)
    w_enclosure_eff = Decimal(1) - w_aperture_eff
    assert float(w_aperture_eff) == pytest.approx(0.18181818181818182, abs=0)
    assert float(w_enclosure_eff) == pytest.approx(0.8181818181818182, abs=0)

    # T_eff,4 built directly from these independently-derived weights (a
    # direct, weight-level check, not only a check of the final shift):
    # a single reflective enclosure at 100K plus a room-temperature (300K)
    # aperture.
    t_enclosure, t_aperture = Decimal(100), Decimal(300)
    m4 = w_enclosure_eff * (t_enclosure / t0) ** 4 + w_aperture_eff * (t_aperture / t0) ** 4
    t_eff4_reference = 300.0 * float(m4) ** (1.0 / 4.0)

    coeffs = species.resolve_bbr_coefficients()
    dyn_hz = sum(
        Decimal(coeff)
        * (w_enclosure_eff * (t_enclosure / t0) ** n + w_aperture_eff * (t_aperture / t0) ** n)
        for n, coeff in coeffs.dyn_coeffs_hz.items()
    )
    delta_nu_hz = Decimal(coeffs.nu_stat_300k_hz) * m4 + dyn_hz
    shift_reference = float(delta_nu_hz / Decimal(species.clock_frequency_hz))

    surfaces = (
        RadiationSurface(name="shield", weight=0.9, temperature_k=100.0, emissivity=0.5),
        RadiationSurface(name="aperture", weight=0.1, temperature_k=300.0),
    )
    got = bbr_environment_pivot_perturbation(surfaces, species)
    t_eff = bbr_environment_effective_temperatures(surfaces, species)

    np.testing.assert_allclose(got, shift_reference, rtol=1e-9, atol=0)
    np.testing.assert_allclose(got, -1.0176625717213562e-15, rtol=1e-9, atol=0)
    np.testing.assert_allclose(t_eff[4], t_eff4_reference, rtol=1e-9, atol=0)
    np.testing.assert_allclose(t_eff[4], 198.56415698802806, rtol=1e-9, atol=0)

    # Kill-test: the G10-flagged bug renormalized every surface's raw-or-
    # corrected weight against the total sum instead of giving the
    # enclosure the complement. For this exact configuration that scheme
    # computes raw_enclosure = 0.9/(0.9+0.1*0.5) = 0.947368... (WRONG: it
    # applies the aperture correction formula to the enclosure's own raw
    # weight, which the enclosure never should be corrected by),
    # raw_aperture = 0.1 unmodified, then divides each by their sum
    # 1.047368...: enclosure -> 0.904511..., aperture -> 0.095489...,
    # nowhere close to the correct (0.818182, 0.181818) pair above (7.6%
    # and 47% off respectively). If `_bbr_effective_weights` ever
    # regresses to that scheme, `got` above would land near
    # -3.2e-16 instead of -1.02e-15, more than 3x off, so this assertion
    # would fail loudly.
    old_wrong_enclosure_weight = 0.9045226130653267
    old_wrong_aperture_weight = 0.09547738693467338
    assert abs(w_enclosure_eff - Decimal(old_wrong_enclosure_weight)) > Decimal("0.05")
    assert abs(w_aperture_eff - Decimal(old_wrong_aperture_weight)) > Decimal("0.05")


# ---------------------------------------------------------------------------
#    Three-surface hand computation (one enclosure, two apertures): the
#    apertures' combined raw weight W = 0.2 + 0.1 = 0.3 forms the single
#    lumped aperture PTB's formula treats; with enclosure emissivity
#    eps=0.6:
#      W_eff = W / (W + (1-W)*eps) = 0.3 / (0.3 + 0.7*0.6) = 0.3/0.72 = 5/12
#    split proportionally across the two apertures by their own raw share
#    of W (2:1 for aperture1:aperture2, matching their raw 0.2:0.1 ratio):
#      w_eff_aperture1 = 0.2/(0.3+0.7*0.6) = 0.2/0.72 = 5/18 = 0.277778...
#      w_eff_aperture2 = 0.1/(0.3+0.7*0.6) = 0.1/0.72 = 5/36 = 0.138889...
#      w_eff_enclosure = 1 - (5/18 + 5/36) = 1 - 5/12 = 7/12 = 0.583333...
#    (5/18 + 5/36 = 10/36 + 5/36 = 15/36 = 5/12, matching W_eff above
#    exactly: the individual apertures' effective weights sum to PTB's own
#    combined-aperture value, the "split proportionally" property this
#    topology relies on).
# ---------------------------------------------------------------------------


def test_ptb_three_surface_enclosure_two_apertures_matches_published_closed_form() -> None:
    species = get_species("Sr87")
    surfaces = (
        RadiationSurface(name="shield", weight=0.7, temperature_k=100.0, emissivity=0.6),
        RadiationSurface(name="aperture1", weight=0.2, temperature_k=300.0),
        RadiationSurface(name="aperture2", weight=0.1, temperature_k=320.0),
    )
    got = bbr_environment_pivot_perturbation(surfaces, species)
    t_eff = bbr_environment_effective_temperatures(surfaces, species)

    np.testing.assert_allclose(got, -2.4809068203051256e-15, rtol=1e-9, atol=0)
    np.testing.assert_allclose(t_eff[4], 247.70357411196898, rtol=1e-9, atol=0)

    # Independent reference: PTB's closed form applied to the combined
    # aperture weight W, split proportionally, exactly the hand computation
    # in the comment above (not calling any function under test).
    t0 = Decimal(300)
    w1, w2, epsilon = Decimal("0.2"), Decimal("0.1"), Decimal("0.6")
    combined_aperture_weight = w1 + w2
    denominator = combined_aperture_weight + (Decimal(1) - combined_aperture_weight) * epsilon
    w1_eff = w1 / denominator
    w2_eff = w2 / denominator
    enclosure_eff = Decimal(1) - (w1_eff + w2_eff)
    assert float(w1_eff) == pytest.approx(5.0 / 18.0, abs=0)
    assert float(w2_eff) == pytest.approx(5.0 / 36.0, abs=0)
    assert float(enclosure_eff) == pytest.approx(7.0 / 12.0, abs=0)
    # The proportional-split property: the two apertures' effective
    # weights keep their raw 2:1 ratio exactly.
    np.testing.assert_allclose(float(w1_eff / w2_eff), 2.0, rtol=0, atol=0)

    coeffs = species.resolve_bbr_coefficients()
    weights_temps = [
        (enclosure_eff, Decimal(100)),
        (w1_eff, Decimal(300)),
        (w2_eff, Decimal(320)),
    ]
    m4 = sum((w * (t / t0) ** 4 for w, t in weights_temps), Decimal(0))
    dyn_hz = sum(
        Decimal(coeff) * sum((w * (t / t0) ** n for w, t in weights_temps), Decimal(0))
        for n, coeff in coeffs.dyn_coeffs_hz.items()
    )
    delta_nu_hz = Decimal(coeffs.nu_stat_300k_hz) * m4 + dyn_hz
    shift_reference = float(delta_nu_hz / Decimal(species.clock_frequency_hz))
    np.testing.assert_allclose(got, shift_reference, rtol=1e-9, atol=0)


def test_more_than_one_emissivity_surface_raises() -> None:
    """At most one surface may model the reflective enclosure (CONVENTIONS.md
    E37's scope boundary: multi-reflector radiosity, more than one
    partially-reflective enclosure surface, is future work).
    """
    species = get_species("Sr87")
    surfaces = (
        RadiationSurface(name="a", weight=0.5, temperature_k=300.0, emissivity=0.5),
        RadiationSurface(name="b", weight=0.5, temperature_k=300.0, emissivity=0.9),
    )
    with pytest.raises(ValueError, match="at most one surface"):
        bbr_environment_pivot_perturbation(surfaces, species)


def test_emissivity_one_reduces_to_naive_weighting() -> None:
    """`epsilon=1` (a perfectly absorbing interior) must give the same
    result as no emissivity at all (CONVENTIONS.md E37: "epsilon = 1...
    reduces this to w_eff = w, the naive geometric weighting"), regardless
    of which surface plays the enclosure role.
    """
    species = get_species("Sr87")
    with_epsilon_one = bbr_environment_pivot_perturbation(
        (
            RadiationSurface(name="a", weight=0.3, temperature_k=280.0, emissivity=1.0),
            RadiationSurface(name="b", weight=0.7, temperature_k=320.0),
        ),
        species,
    )
    without_emissivity = bbr_environment_pivot_perturbation(
        (
            RadiationSurface(name="a", weight=0.3, temperature_k=280.0),
            RadiationSurface(name="b", weight=0.7, temperature_k=320.0),
        ),
        species,
    )
    np.testing.assert_allclose(with_epsilon_one, without_emissivity, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# 5. Kill-test: a moment-exponent swap (using M6 where M4 belongs, and vice
#    versa, for the static term and the leading dynamic-anchor term) would
#    silently change the result; this test computes both the correct
#    value (via the function under test, cross-checked against the decimal
#    reference) and the exponent-swapped WRONG value (independently, in
#    this test) and asserts they differ well beyond float noise, so a
#    regression that introduced this swap would fail this test.
# ---------------------------------------------------------------------------


def test_moment_exponent_swap_kill_test() -> None:
    species = get_species("Sr87")
    coeffs = species.resolve_bbr_coefficients()
    surfaces = (
        RadiationSurface(name="bulk", weight=0.9, temperature_k=300.0),
        RadiationSurface(name="warm_patch", weight=0.1, temperature_k=340.0),
    )
    got = bbr_environment_pivot_perturbation(surfaces, species)
    reference = _decimal_bbr_environment_pivot_perturbation("Sr87", [(0.9, 300.0), (0.1, 340.0)])
    np.testing.assert_allclose(got, reference, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got, -5.6863779779293556e-15, rtol=1e-12, atol=0)

    # The WRONG value a static<->leading-dynamic exponent swap would give:
    # Delta_nu_wrong = nu_stat*M6 + c_6*M4 + c_8*M8 + c_10*M10 (4 and 6
    # swapped only where the static term and the n=6 dynamic term meet).
    t_eff = bbr_environment_effective_temperatures(surfaces, species)
    t0 = 300.0
    m4 = (t_eff[4] / t0) ** 4
    m6 = (t_eff[6] / t0) ** 6
    m8 = (t_eff[8] / t0) ** 8
    m10 = (t_eff[10] / t0) ** 10
    delta_nu_wrong_hz = (
        coeffs.nu_stat_300k_hz * m6
        + coeffs.dyn_coeffs_hz[6] * m4
        + coeffs.dyn_coeffs_hz[8] * m8
        + coeffs.dyn_coeffs_hz[10] * m10
    )
    wrong_value = delta_nu_wrong_hz / species.clock_frequency_hz
    # Independently pinned wrong value (computed via the decimal reference
    # with the same swap, see the implementation notes above): -5.90e-15.
    np.testing.assert_allclose(wrong_value, -5.904827327180233e-15, rtol=1e-6, atol=0)
    assert abs(got - wrong_value) > 1e-16, (
        "the correct and exponent-swapped values must differ by far more than float64 "
        "noise here; if they ever match, the implementation has silently regressed to "
        "using the wrong moment for the static or leading-dynamic term"
    )


# ---------------------------------------------------------------------------
# 6. Uncertainty: independent vs. correlated combination modes.
#
#    Hand computation (Sr87): a_i = w_i * d(Delta_nu_hz)/dT|_{T_i}. For
#    surfaces A (w=0.5, T=300K, sigma_T=4mK) and B (w=0.5, T=320K,
#    sigma_T=6mK): a_A*sigma_A = -6.332e-5 Hz, a_B*sigma_B = -1.174e-4 Hz
#    (both negative, same sign, since every registry coefficient is
#    negative). independent = sqrt(sum of squares) ~= 1.334e-4 Hz;
#    correlated = |sum| ~= 1.808e-4 Hz, strictly larger (L1 >= L2 norm
#    for a same-sign vector with more than one nonzero component).
# ---------------------------------------------------------------------------


def test_uncertainty_independent_vs_correlated_direction() -> None:
    species = get_species("Sr87")
    surfaces = (
        RadiationSurface(
            name="a", weight=0.5, temperature_k=300.0, temperature_uncertainty_k=0.004
        ),
        RadiationSurface(
            name="b", weight=0.5, temperature_k=320.0, temperature_uncertainty_k=0.006
        ),
    )
    sigma_independent, included_independent = bbr_environment_pivot_uncertainty(
        surfaces, species, correlated=False
    )
    sigma_correlated, included_correlated = bbr_environment_pivot_uncertainty(
        surfaces, species, correlated=True
    )
    assert included_independent is True
    assert included_correlated is True
    # Independently hand-derived pins (see comment above); an independent
    # nonzero rtol, no vacuous tolerance.
    np.testing.assert_allclose(sigma_independent, 1.0129187238500215e-18, rtol=1e-9, atol=0)
    np.testing.assert_allclose(sigma_correlated, 1.0520138695305229e-18, rtol=1e-9, atol=0)
    assert sigma_correlated > sigma_independent


def test_uncertainty_single_surface_independent_equals_correlated() -> None:
    """With only one surface there is nothing to correlate against: both
    combination modes must agree exactly (both reduce to the same single
    `|a_1 * sigma_1|` term).
    """
    species = get_species("Sr87")
    surfaces = (
        RadiationSurface(
            name="uniform", weight=1.0, temperature_k=300.0, temperature_uncertainty_k=0.004
        ),
    )
    sigma_independent, _ = bbr_environment_pivot_uncertainty(surfaces, species, correlated=False)
    sigma_correlated, _ = bbr_environment_pivot_uncertainty(surfaces, species, correlated=True)
    np.testing.assert_allclose(sigma_independent, sigma_correlated, rtol=0, atol=0)


def test_uncertainty_zero_surface_uncertainties_reports_not_included() -> None:
    species = get_species("Sr87")
    surfaces = (
        RadiationSurface(name="a", weight=0.5, temperature_k=300.0),
        RadiationSurface(name="b", weight=0.5, temperature_k=320.0),
    )
    sigma, included = bbr_environment_pivot_uncertainty(surfaces, species)
    assert included is False
    assert sigma > 0.0  # coefficient uncertainty alone is still nonzero


# ---------------------------------------------------------------------------
# 7. Pipeline config parsing: mutual exclusion, weight/window rejection,
#    and threading into a fast_path run matches the direct function call.
# ---------------------------------------------------------------------------


def test_radiation_environment_absent_leaves_environment_off() -> None:
    assert _parse_environment(None).radiation_environment is None
    assert _parse_environment({}).radiation_environment is None


def test_radiation_environment_parses_surfaces_and_correlated_flag() -> None:
    cfg = _parse_environment(
        {
            "radiation_environment": {
                "surfaces": [
                    {"name": "a", "weight": 0.6, "temperature_K": 300.0},
                    {
                        "name": "b",
                        "weight": 0.4,
                        "temperature_K": 320.0,
                        "temperature_uncertainty_K": 0.01,
                        "emissivity": 0.9,
                    },
                ],
                "correlated": True,
            }
        }
    )
    assert cfg.radiation_temperature_k is None
    assert isinstance(cfg.radiation_environment, RadiationEnvironmentConfig)
    assert cfg.radiation_environment.correlated is True
    assert len(cfg.radiation_environment.surfaces) == 2
    assert cfg.radiation_environment.surfaces[0].name == "a"
    assert cfg.radiation_environment.surfaces[1].emissivity == 0.9


def test_radiation_environment_and_radiation_temperature_mutually_exclusive() -> None:
    with pytest.raises(PipelineConfigError, match="radiation_temperature_K.*radiation_environment"):
        _parse_environment(
            {
                "radiation_temperature_K": 300.0,
                "radiation_environment": {
                    "surfaces": [{"name": "a", "weight": 1.0, "temperature_K": 300.0}]
                },
            }
        )


def test_radiation_environment_unnormalized_weights_rejected_at_parse_time() -> None:
    with pytest.raises(PipelineConfigError, match="must sum to 1"):
        _parse_environment(
            {
                "radiation_environment": {
                    "surfaces": [
                        {"name": "a", "weight": 0.5, "temperature_K": 300.0},
                        {"name": "b", "weight": 0.6, "temperature_K": 300.0},
                    ]
                }
            }
        )


def test_radiation_environment_window_violation_rejected_at_parse_time() -> None:
    with pytest.raises(PipelineConfigError, match="outside the validated BBR fit range"):
        _parse_environment(
            {
                "radiation_environment": {
                    "surfaces": [{"name": "a", "weight": 1.0, "temperature_K": 500.0}]
                }
            }
        )


def test_radiation_environment_requires_stark_dc_coupling(tmp_path: Path) -> None:
    data = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
        "coupling": {"type": "linear_mu", "mu": [1e-30, 0.0, 0.0]},
        "ensemble": {
            "regime": "lattice",
            "temperature_uK": 1.0,
            "motional_n": [0, 0, 0],
            "n_quad": 1,
        },
        "integration": {"time_s": 1.0},
        "environment": {
            "radiation_environment": {
                "surfaces": [{"name": "a", "weight": 1.0, "temperature_K": 300.0}]
            }
        },
        "output": {"directory": str(tmp_path)},
    }
    with pytest.raises(PipelineConfigError, match="radiation_environment requires"):
        PipelineConfig.from_dict(data)


def _base_lattice_stark_dict(output_dir: Path) -> dict[str, object]:
    return {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
        "coupling": {"type": "stark_dc"},
        "ensemble": {
            "regime": "lattice",
            "temperature_uK": 1.0,
            "motional_n": [0, 0, 0],
            "n_quad": 1,
        },
        "integration": {"time_s": 1.0},
        "output": {"directory": str(output_dir)},
    }


def test_radiation_environment_threading_matches_direct_function_call_fast_path(
    tmp_path: Path,
) -> None:
    """The reported shift delta from turning on `radiation_environment`
    must equal `bbr_environment_pivot_perturbation` called directly, to
    1e-25 absolute (the single lattice quadrature node here has `v=0`
    exactly, fast_path/E29, so `gamma_inv=1` exactly and there is no
    kinematic-weighting correction to account for).
    """
    without_data = _base_lattice_stark_dict(tmp_path / "without")
    without_config = PipelineConfig.from_dict(without_data)
    without_result = run_pipeline_full(without_config)

    with_data = _base_lattice_stark_dict(tmp_path / "with")
    with_data["environment"] = {
        "radiation_environment": {
            "surfaces": [
                {"name": "cold_half", "weight": 0.5, "temperature_K": 280.0},
                {"name": "warm_half", "weight": 0.5, "temperature_K": 320.0},
            ]
        }
    }
    with_config = PipelineConfig.from_dict(with_data)
    with_result = run_pipeline_full(with_config)

    delta = with_result.report.mean_fractional_shift - without_result.report.mean_fractional_shift
    species = get_species("Sr87")
    expected = bbr_environment_pivot_perturbation(
        (
            RadiationSurface(name="cold_half", weight=0.5, temperature_k=280.0),
            RadiationSurface(name="warm_half", weight=0.5, temperature_k=320.0),
        ),
        species,
    )
    np.testing.assert_allclose(delta, expected, rtol=0, atol=1e-25)


def test_radiation_environment_report_note_mentions_e37_and_effective_temperatures(
    tmp_path: Path,
) -> None:
    data = _base_lattice_stark_dict(tmp_path)
    data["environment"] = {
        "radiation_environment": {
            "surfaces": [
                {"name": "cold_half", "weight": 0.5, "temperature_K": 280.0},
                {"name": "warm_half", "weight": 0.5, "temperature_K": 320.0},
            ],
            "correlated": True,
        }
    }
    config = PipelineConfig.from_dict(data)
    result = run_pipeline_full(config)
    notes = result.report.uncertainty_notes
    assert "E37" in notes
    assert "T_eff,4" in notes
    assert "T_eff,6" in notes
    assert "correlated" in notes
    assert "cold_half" in notes
    assert "warm_half" in notes


def test_environment_dataclass_equality_unaffected_when_radiation_environment_absent() -> None:
    """Every pre-WP29 `EnvironmentConfig()` comparison in the existing test
    suite (e.g. `tests/test_bbr_pipeline.py`, `tests/test_gravity_pivot.py`)
    relies on `EnvironmentConfig() == EnvironmentConfig()` staying true;
    the new `radiation_environment` field must default to `None` so that
    equality is untouched.
    """
    assert EnvironmentConfig() == EnvironmentConfig()
    assert EnvironmentConfig().radiation_environment is None
