# SPDX-License-Identifier: AGPL-3.0-or-later
"""Species registry pin tests (WP4 test contract item 5)."""

import pytest

from cliffordclock import constants
from cliffordclock.ensemble.species import get_species


def test_sr87_mass_pinned_to_ame2020() -> None:
    """Sr-87 mass matches AME2020 (Wang et al. 2021), 86.908877497(50) u."""
    species = get_species("Sr87")
    expected_kg = 86.908877497 * constants.ATOMIC_MASS_UNIT
    assert species.mass_kg == pytest.approx(expected_kg, rel=1e-12, abs=0)
    # Sanity: mass in amu is close to the mass number.
    assert species.mass_kg / constants.ATOMIC_MASS_UNIT == pytest.approx(87, rel=1e-2, abs=0)


def test_sr87_clock_frequency_pinned_to_bipm() -> None:
    """Sr-87 5s^2 1S0 -> 5s5p 3P0 clock frequency matches the BIPM recommended value."""
    species = get_species("Sr87")
    assert species.clock_frequency_hz == pytest.approx(429_228_004_229_873.4, rel=1e-15, abs=0)
    assert species.clock_wavelength_m == pytest.approx(
        constants.SPEED_OF_LIGHT / 429_228_004_229_873.4, rel=1e-15, abs=0
    )
    # Sanity: Sr clock wavelength is the well-known ~698 nm.
    assert species.clock_wavelength_m == pytest.approx(698e-9, rel=1e-3, abs=0)


def test_yb171_mass_pinned_to_ame2020() -> None:
    """Yb-171 mass matches AME2020 (Wang et al. 2021), 170.9363302(2) u."""
    species = get_species("Yb171")
    expected_kg = 170.9363302 * constants.ATOMIC_MASS_UNIT
    assert species.mass_kg == pytest.approx(expected_kg, rel=1e-12, abs=0)
    assert species.mass_kg / constants.ATOMIC_MASS_UNIT == pytest.approx(171, rel=1e-2, abs=0)


def test_yb171_clock_frequency_pinned_to_bipm() -> None:
    """Yb-171 6s^2 1S0 -> 6s6p 3P0 clock frequency matches the BIPM recommended value."""
    species = get_species("Yb171")
    assert species.clock_frequency_hz == pytest.approx(518_295_836_590_863.6, rel=1e-15, abs=0)
    # Sanity: Yb clock wavelength is the well-known ~578 nm.
    assert species.clock_wavelength_m == pytest.approx(578e-9, rel=1e-3, abs=0)


def test_al27plus_mass_pinned_to_ame2020() -> None:
    """Al-27 mass matches AME2020 (Wang et al. 2021), 26.98153853(11) u."""
    species = get_species("Al27+")
    expected_kg = 26.98153853 * constants.ATOMIC_MASS_UNIT
    assert species.mass_kg == pytest.approx(expected_kg, rel=1e-12, abs=0)
    assert species.mass_kg / constants.ATOMIC_MASS_UNIT == pytest.approx(27, rel=1e-2, abs=0)


def test_al27plus_clock_frequency_pinned_to_brewer_2019() -> None:
    """Al27+ 3s^2 1S0 -> 3s3p 3P0 clock frequency matches Brewer et al. (2019)."""
    species = get_species("Al27+")
    assert species.clock_frequency_hz == pytest.approx(1_121_015_393_207_857.4, rel=1e-15, abs=0)
    # Sanity: Al+ clock wavelength is the well-known ~267 nm (deep UV).
    assert species.clock_wavelength_m == pytest.approx(267e-9, rel=1e-2, abs=0)


def test_al27plus_is_singly_ionized() -> None:
    assert get_species("Al27+").charge_state == 1


def test_in115plus_mass_pinned() -> None:
    """In-115 mass matches NIST Atomic Weights, 114.903878776(12) u
    (WP21; consistent with the AME2020 vintage the other species use)."""
    species = get_species("In115+")
    expected_kg = 114.903878776 * constants.ATOMIC_MASS_UNIT
    # abs=0: at ~1.9e-25 kg, pytest.approx's default abs=1e-12 would
    # swallow the whole comparison (WP21 review nit).
    assert species.mass_kg == pytest.approx(expected_kg, rel=1e-12, abs=0)
    assert species.mass_kg / constants.ATOMIC_MASS_UNIT == pytest.approx(115, rel=1e-2, abs=0)


def test_in115plus_is_singly_ionized() -> None:
    assert get_species("In115+").charge_state == 1


def test_sr87_and_yb171_are_neutral() -> None:
    assert get_species("Sr87").charge_state == 0
    assert get_species("Yb171").charge_state == 0


def test_get_species_unknown_name_lists_valid_names() -> None:
    with pytest.raises(KeyError, match="Sr87"):
        get_species("Not-A-Species")


def test_registry_has_exactly_four_species() -> None:
    """WP4 non-goals originally scoped three species (Sr87/Yb171/Al27+);
    WP21 (CONVENTIONS.md E34, ion-clock support) added In115+ -- no
    species beyond these four.
    """
    for name in ("Sr87", "Yb171", "Al27+", "In115+"):
        assert get_species(name).name == name
    with pytest.raises(KeyError):
        get_species("Cs133")
