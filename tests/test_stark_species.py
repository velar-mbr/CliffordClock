# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP7 (CONVENTIONS.md E14b) species polarizability data.

Covers WP7 test contract item 5: cited Delta-alpha/k_S values pinned, and
the a.u.->SI conversion round-trips a literature a.u. value. WP21
superseded the original "Al27+ raises a clear error" case: Al27+ (J=0 ->
J=0) now carries a registry Delta_alpha -- see
`tests/test_ion_species.py` for the WP21 registry pins and
`test_al27plus_now_has_stark_data_wp21` below.
"""

from __future__ import annotations

import numpy as np
import pytest

from cliffordclock.constants import PLANCK_H
from cliffordclock.ensemble.species import (
    ALPHA_AU_TO_SI,
    StarkCoefficients,
    get_species,
)


def test_sr87_delta_alpha_pinned_to_middelmann_2012() -> None:
    """Sr87 Delta-alpha matches Middelmann et al., PRL 109, 263004 (2012),
    "Delta_alpha = 4.07873(11) x 10^-39 Cm^2/V" (arXiv:1208.2848).
    """
    species = get_species("Sr87")
    assert species.delta_alpha_dc_si is not None
    # rtol/atol=0 explicit (REVIEW-checklist effective-tolerance item):
    # the paper's value is precise to 6 significant figures (4.07873(11)),
    # so rtol=1e-6 pins the transcribed digits without masking drift.
    np.testing.assert_allclose(species.delta_alpha_dc_si, 4.07873e-39, rtol=1e-6, atol=0)


def test_sr87_delta_alpha_consistent_with_porsev_theory_cross_check() -> None:
    """Independent (theoretical) second source cited by Middelmann et al.:
    Porsev et al. give Delta_alpha = 4.305(59) x 10^-39 Cm^2/V, "more than
    3 sigma" from the experimental value but "within the typical range of
    deviation between calculated and measured values" per the paper. This
    is a loose sanity cross-check (order-of-magnitude/same-sign agreement
    within ~10%), not a precision pin -- the registry value is the
    experimental (Middelmann) one.
    """
    species = get_species("Sr87")
    assert species.delta_alpha_dc_si is not None
    porsev_theory = 4.305e-39
    relative_deviation = abs(species.delta_alpha_dc_si - porsev_theory) / porsev_theory
    assert relative_deviation < 0.10, (
        "Sr87 registry Delta-alpha deviates from the independent Porsev "
        f"theory cross-check by {relative_deviation:.1%}, more than the "
        "expected few-percent range"
    )


def test_yb171_delta_alpha_pinned_to_sherman_2012() -> None:
    """Yb171 Delta-alpha matches Sherman et al., PRL 108, 153002 (2012),
    Table II "this work" row: 2.40269(5) x 10^-39 [C m^2/V] (arXiv:1112.2766).
    """
    species = get_species("Yb171")
    assert species.delta_alpha_dc_si is not None
    np.testing.assert_allclose(species.delta_alpha_dc_si, 2.40269e-39, rtol=1e-6, atol=0)


def test_yb171_delta_alpha_consistent_with_porsev_derevianko_theory_cross_check() -> None:
    """Independent (theoretical) second source, from the same paper's
    Table II: S.G. Porsev, A. Derevianko, Phys. Rev. A 74, 020502 (2006),
    Delta_alpha = 2.56(26) x 10^-39 Cm^2/V. Loose sanity cross-check, as
    for Sr87 above.
    """
    species = get_species("Yb171")
    assert species.delta_alpha_dc_si is not None
    porsev_derevianko_theory = 2.56e-39
    relative_deviation = abs(species.delta_alpha_dc_si - porsev_derevianko_theory) / (
        porsev_derevianko_theory
    )
    assert relative_deviation < 0.10, (
        "Yb171 registry Delta-alpha deviates from the independent "
        f"Porsev-Derevianko theory cross-check by {relative_deviation:.1%}"
    )


def test_alpha_au_to_si_round_trips_yb_literature_a_u_value() -> None:
    """WP7 test contract item 5: a.u.->SI round-trips a literature a.u.
    value. Sherman et al. Table II "this work" row also reports the same
    measurement in atomic units: 145.726(3) a.u. Multiplying by
    ALPHA_AU_TO_SI must reproduce the registry's SI Delta-alpha to within
    the a.u. value's quoted precision (6 significant figures).
    """
    species = get_species("Yb171")
    assert species.delta_alpha_dc_si is not None
    yb_au_value = 145.726
    round_tripped_si = yb_au_value * ALPHA_AU_TO_SI
    np.testing.assert_allclose(round_tripped_si, species.delta_alpha_dc_si, rtol=1e-5, atol=0)


def test_alpha_au_to_si_matches_codata_4pi_eps0_a0_cubed() -> None:
    """ALPHA_AU_TO_SI = 4 pi eps0 a0^3 (CONVENTIONS.md E14b), independently
    recomputed from CODATA 2022 eps0 and a0 -- not merely re-asserting the
    literal constant, since that would be tautological.
    """
    eps0_2022 = 8.8541878128e-12  # F/m, CODATA
    a0_2022 = 5.29177210544e-11  # m, Bohr radius, CODATA 2022
    recomputed = 4.0 * np.pi * eps0_2022 * a0_2022**3
    np.testing.assert_allclose(ALPHA_AU_TO_SI, recomputed, rtol=1e-6, atol=0)


def test_stark_coefficient_derived_from_delta_alpha_matches_e14b_formula() -> None:
    """stark_coefficient_hz_per_v2_m2 == -delta_alpha_dc_si / (2h) exactly
    (CONVENTIONS.md E14b "equivalent per-species input"), for both
    populated species.
    """
    for name in ("Sr87", "Yb171"):
        species = get_species(name)
        assert species.delta_alpha_dc_si is not None
        assert species.stark_coefficient_hz_per_v2_m2 is not None
        expected = -species.delta_alpha_dc_si / (2.0 * PLANCK_H)
        np.testing.assert_allclose(
            species.stark_coefficient_hz_per_v2_m2, expected, rtol=1e-14, atol=0
        )
        np.testing.assert_allclose(
            species.resolve_stark_coefficient_hz_per_v2_m2(), expected, rtol=1e-14, atol=0
        )


def test_al27plus_now_has_stark_data_wp21() -> None:
    """WP21 supersedes WP7's Al27+ exclusion: Al27+ is J=0 -> J=0, so
    E14b's scalar Delta_alpha treatment applies as-is (no tensor/
    quadrupole term needed for this transition -- that machinery is for
    the D/F-state ions, `cliffordclock.ensemble.species.QUADRUPOLE_MOMENTS`).
    See `tests/test_ion_species.py` for the pinned value/citation.
    """
    species = get_species("Al27+")
    assert species.delta_alpha_dc_si is not None
    assert species.stark_coefficient_hz_per_v2_m2 is not None
    # Does not raise (the pre-WP21 behavior this test replaces).
    species.resolve_stark_coefficient_hz_per_v2_m2()


def test_in115plus_now_has_stark_data_wp21() -> None:
    """WP21: In115+ is likewise J=0 -> J=0, registered with theory-derived
    Delta_alpha (Safronova et al. 2011)."""
    species = get_species("In115+")
    assert species.delta_alpha_dc_si is not None
    species.resolve_stark_coefficient_hz_per_v2_m2()


class TestStarkCoefficientsOverride:
    """`StarkCoefficients` lets a caller bypass the species registry."""

    def test_from_delta_alpha_only(self) -> None:
        coeffs = StarkCoefficients(clock_frequency_hz=1e15, delta_alpha_dc_si=1e-39)
        expected = -1e-39 / (2.0 * PLANCK_H)
        np.testing.assert_allclose(
            coeffs.resolve_stark_coefficient_hz_per_v2_m2(), expected, rtol=1e-14, atol=0
        )

    def test_from_stark_coefficient_only(self) -> None:
        coeffs = StarkCoefficients(
            clock_frequency_hz=1e15, stark_coefficient_hz_per_v2_m2=-1.234e-6
        )
        np.testing.assert_allclose(
            coeffs.resolve_stark_coefficient_hz_per_v2_m2(), -1.234e-6, rtol=1e-14, atol=0
        )

    def test_both_fields_consistent_is_accepted(self) -> None:
        delta_alpha = 1e-39
        k_s = -delta_alpha / (2.0 * PLANCK_H)
        coeffs = StarkCoefficients(
            clock_frequency_hz=1e15,
            delta_alpha_dc_si=delta_alpha,
            stark_coefficient_hz_per_v2_m2=k_s,
        )
        np.testing.assert_allclose(
            coeffs.resolve_stark_coefficient_hz_per_v2_m2(), k_s, rtol=1e-14, atol=0
        )

    def test_both_fields_inconsistent_raises(self) -> None:
        with pytest.raises(ValueError, match="inconsistent"):
            StarkCoefficients(
                clock_frequency_hz=1e15,
                delta_alpha_dc_si=1e-39,
                stark_coefficient_hz_per_v2_m2=1.0,  # wildly inconsistent
            )

    def test_neither_field_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            StarkCoefficients(clock_frequency_hz=1e15)
