# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP20 (CONVENTIONS.md E32) BBR registry data.

Mirrors tests/test_stark_species.py's role for the E14b Delta-alpha
registry: pins the transcribed literature coefficients, checks the
`Al27+` no-data error path, and checks the module-level validity-window
constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from cliffordclock.ensemble.species import (
    BBR_CROSS_VERIFIED_MAX_K,
    BBR_REFERENCE_TEMPERATURE_K,
    BBR_VALIDITY_MAX_K,
    BBR_VALIDITY_MIN_K,
    get_species,
)


def test_bbr_reference_temperature_is_300k() -> None:
    """T0 = 300 K (dossier Sec.1; every source paper normalizes here)."""
    assert BBR_REFERENCE_TEMPERATURE_K == 300.0


def test_bbr_validity_window_is_50_to_350k() -> None:
    """G7 sign-off gate edit 5: hard window 50-350 K."""
    assert BBR_VALIDITY_MIN_K == 50.0
    assert BBR_VALIDITY_MAX_K == 350.0


def test_bbr_cross_verified_max_is_300k() -> None:
    """G7 sign-off B4: the PTB<->JILA 1e-19 cross-verification statement
    covers T <= 300 K only; 300-350 K is in-fit-range but unverified.
    """
    assert BBR_CROSS_VERIFIED_MAX_K == 300.0


def test_sr87_bbr_static_pinned_to_middelmann_2012() -> None:
    """Sr87 Delta-nu_stat(300K) = -2.13023(6) Hz, Middelmann et al., PRL 109,
    263004 (2012) -- unchanged across the Lisdat/Aeppli/PTB dynamic-term
    revisions (dossier Sec.2).
    """
    coeffs = get_species("Sr87").resolve_bbr_coefficients()
    # rtol/atol=0 explicit (REVIEW-checklist effective-tolerance item): the
    # paper quotes 6 significant figures.
    np.testing.assert_allclose(coeffs.nu_stat_300k_hz, -2.13023, rtol=1e-6, atol=0)
    # G7 sign-off A4#2 correction: the "(6)" is the LAST digit -> +/-0.00006 Hz,
    # not +/-0.006 Hz and not +/-6 mHz as the uncorrected theory brief read it.
    np.testing.assert_allclose(coeffs.nu_stat_300k_uncertainty_hz, 6e-5, rtol=1e-9, atol=0)


def test_sr87_bbr_dynamic_pinned_to_ptb_2025_rescaled_polynomial() -> None:
    """Sr87 dynamic coefficients: PTB-2025 rescaled polynomial
    (arXiv:2507.14030), eta_6/eta_8/eta_10 = -0.13216/-0.01231/-0.00858 Hz,
    anchored to Aeppli 2024's +/-0.33 mHz (dossier Sec.7, G7 sign-off B2).
    """
    coeffs = get_species("Sr87").resolve_bbr_coefficients()
    assert set(coeffs.dyn_coeffs_hz) == {6, 8, 10}
    np.testing.assert_allclose(coeffs.dyn_coeffs_hz[6], -0.13216, rtol=1e-6, atol=0)
    np.testing.assert_allclose(coeffs.dyn_coeffs_hz[8], -0.01231, rtol=1e-6, atol=0)
    np.testing.assert_allclose(coeffs.dyn_coeffs_hz[10], -0.00858, rtol=1e-6, atol=0)
    np.testing.assert_allclose(coeffs.dyn_anchor_uncertainty_hz, 0.00033, rtol=1e-6, atol=0)


def test_sr87_bbr_dynamic_polynomial_sums_to_dossier_300k_value() -> None:
    """Sum of the three dynamic coefficients at T=300K reproduces the
    dossier's quoted -153.05 mHz (dossier Sec.7, "sum at 300 K: -153.05 mHz").
    """
    coeffs = get_species("Sr87").resolve_bbr_coefficients()
    total_hz = sum(coeffs.dyn_coeffs_hz.values())
    np.testing.assert_allclose(total_hz, -0.15305, rtol=2e-4, atol=0)


def test_yb171_bbr_static_and_dynamic_t6_pinned_to_hassan_2025() -> None:
    """Yb171: Delta-nu_stat(300K) = -1.2545(10) Hz, nu_dyn,6 = -22.17(34) mHz
    (Hassan et al., arXiv:2506.05304 (2025)).
    """
    coeffs = get_species("Yb171").resolve_bbr_coefficients()
    np.testing.assert_allclose(coeffs.nu_stat_300k_hz, -1.2545, rtol=1e-6, atol=0)
    np.testing.assert_allclose(coeffs.nu_stat_300k_uncertainty_hz, 1.0e-3, rtol=1e-6, atol=0)
    np.testing.assert_allclose(coeffs.dyn_coeffs_hz[6], -22.17e-3, rtol=1e-6, atol=0)
    np.testing.assert_allclose(coeffs.dyn_anchor_uncertainty_hz, 0.34e-3, rtol=1e-6, atol=0)


def test_yb171_bbr_dynamic_t8_matches_beloy_eta2_derivation() -> None:
    """nu_dyn,8 = nu_stat_300k * eta_2 = -1.2545 * 0.000593 = -0.744 mHz
    (Beloy et al., PRL 113, 260801 (2014); G7 sign-off B3 "arithmetic ...
    verified"). Independently re-derived here (not just re-reading the
    registry literal) to catch a transcription error in the stored value.
    """
    coeffs = get_species("Yb171").resolve_bbr_coefficients()
    eta_2 = 0.000593
    expected_hz = coeffs.nu_stat_300k_hz * eta_2
    np.testing.assert_allclose(coeffs.dyn_coeffs_hz[8], expected_hz, rtol=1e-3, atol=0)
    np.testing.assert_allclose(coeffs.dyn_coeffs_hz[8], -0.744e-3, rtol=1e-3, atol=0)


def test_yb171_bbr_eta1_order_mapping_cross_check_against_measured_nu_dyn6() -> None:
    """G7 sign-off B3: nu_stat_300k * eta_1 = -1.2545 * 0.01745 ~= -21.9 mHz,
    which agrees with Hassan's *directly measured* nu_dyn,6 = -22.17(34) mHz
    -- confirms the eta_n -> T^(4+2n) order mapping is pinned by physics,
    not an unverified dataset-index inference.
    """
    coeffs = get_species("Yb171").resolve_bbr_coefficients()
    eta_1 = 0.01745
    predicted_hz = coeffs.nu_stat_300k_hz * eta_1
    measured_hz = coeffs.dyn_coeffs_hz[6]
    relative_deviation = abs(predicted_hz - measured_hz) / abs(measured_hz)
    assert relative_deviation < 0.02, (
        f"eta_1 order-mapping cross-check deviates by {relative_deviation:.1%} from "
        "Hassan's measured nu_dyn,6 -- expected <2%"
    )


def test_al27_plus_has_no_bbr_coefficients() -> None:
    """Al27+ (ion clock): no BBR fit in scope (WP20 registry covers Sr87/Yb171 only)."""
    species = get_species("Al27+")
    assert species.bbr_coefficients is None
    with pytest.raises(ValueError, match="no BBR shift data"):
        species.resolve_bbr_coefficients()


def test_sr87_yb171_bbr_validity_windows_match_module_constants() -> None:
    for name in ("Sr87", "Yb171"):
        coeffs = get_species(name).resolve_bbr_coefficients()
        assert coeffs.validity_min_k == BBR_VALIDITY_MIN_K
        assert coeffs.validity_max_k == BBR_VALIDITY_MAX_K
        assert coeffs.cross_verified_max_k == BBR_CROSS_VERIFIED_MAX_K
