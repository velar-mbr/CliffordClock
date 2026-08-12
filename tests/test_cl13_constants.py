# SPDX-License-Identifier: AGPL-3.0-or-later
"""Derived Compton-scale constants tests (docs/CONVENTIONS.md E7-E8, section 3 note).

Added under the orchestrator's G0-resolution scope grant (docs/CONVENTIONS.md
section 3 note / G0 item 1): `OMEGA_COMPTON`, `T_COMPTON_CYCLE`, and
`LAMBDA_BAR_COMPTON` in `cliffordclock.constants`. Kept in this WP1-owned
file (rather than `tests/test_scaffold.py`) to avoid touching a file other
work packages may also be editing concurrently.
"""

import math

from cliffordclock import constants


def test_omega_compton_times_tau_compton_is_exactly_one() -> None:
    """omega_C * tau_c == 1 exactly in fp64 (E7-E8: omega_C := 1 / tau_c)."""
    assert constants.OMEGA_COMPTON * constants.TAU_COMPTON == 1.0


def test_tau_compton_matches_literature_value() -> None:
    """tau_c = hbar / (m_e c^2) (E7) matches an independently computed literature value.

    Independent reference: CODATA 2022 electron Compton wavelength
    lambda_C = 2.42631023538e-12 m (https://physics.nist.gov/cgi-bin/cuu/Value?compwl).
    tau_c = lambda_C / (2 pi c).
    """
    compton_wavelength = 2.42631023538e-12  # m, CODATA 2022 lambda_C
    independent_tau_c = compton_wavelength / (2 * math.pi * constants.SPEED_OF_LIGHT)
    assert abs(independent_tau_c - constants.TAU_COMPTON) / independent_tau_c < 1e-6


def test_omega_compton_matches_literature_value() -> None:
    """omega_C = m_e c^2 / hbar = 2 pi c / lambda_C (E8) matches an independent literature value."""
    compton_wavelength = 2.42631023538e-12  # m, CODATA 2022 lambda_C
    independent_omega_c = 2 * math.pi * constants.SPEED_OF_LIGHT / compton_wavelength
    assert abs(independent_omega_c - constants.OMEGA_COMPTON) / independent_omega_c < 1e-6


def test_t_compton_cycle_matches_literature_value() -> None:
    """T_C = 2 pi tau_c = h / (m_e c^2) = lambda_C / c (section 3 note) matches literature.

    This is the source spec's tabulated (mislabeled) value, here pinned
    against the literature Compton wavelength independently of tau_c.
    """
    compton_wavelength = 2.42631023538e-12  # m, CODATA 2022 lambda_C
    independent_t_c = compton_wavelength / constants.SPEED_OF_LIGHT
    assert abs(independent_t_c - constants.T_COMPTON_CYCLE) / independent_t_c < 1e-6


def test_lambda_bar_compton_matches_literature_value() -> None:
    """lambda_bar_C = c tau_c (E18 scaling length) equals the standard reduced Compton
    wavelength of the electron; matches an independent literature value.

    Independent reference: CODATA 2022 electron Compton wavelength
    lambda_C = 2.42631023538e-12 m; lambda_bar_C = lambda_C / (2 pi).
    """
    compton_wavelength = 2.42631023538e-12  # m, CODATA 2022 lambda_C
    independent_lambda_bar_c = compton_wavelength / (2 * math.pi)
    assert (
        abs(independent_lambda_bar_c - constants.LAMBDA_BAR_COMPTON) / independent_lambda_bar_c
        < 1e-6
    )
