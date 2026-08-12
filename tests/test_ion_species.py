# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP21 (CONVENTIONS.md E34) ion-clock registry data.

Mirrors `tests/test_stark_species.py`/`tests/test_bbr_species.py`'s role:
pins the transcribed literature values (Al27+/In115+ Delta_alpha,
EA0_SQUARED_SI, the QUADRUPOLE_MOMENTS table) and the "no data" error
paths. Also covers the WP21 Tier-1 test-contract item "known-answer tests
against the literature Delta_alpha arithmetic (KA-style); cross-mode
agreement" for the two new ion species -- mirroring
`tests/test_stark_pivot.py`'s Sr87/Yb171 formula-pin tests. Pure-formula
quadrupole-shift tests live in `tests/test_quadrupole_pivot.py`.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock import constants
from cliffordclock.constants import PLANCK_H
from cliffordclock.ensemble.species import (
    ALPHA_AU_TO_SI,
    EA0_SQUARED_SI,
    ION_HYPERFINE_E2_BUDGET_NOTES,
    ION_MICROMOTION_NOTES,
    QUADRUPOLE_MOMENTS,
    get_quadrupole_moment,
    get_species,
)
from cliffordclock.integrator.omega import pivot_perturbation_stark
from cliffordclock.pipeline import (
    CouplingConfig,
    EnsembleConfig,
    FieldConfig,
    IntegrationConfig,
    OutputConfig,
    PipelineConfig,
    SyntheticFieldConfig,
    TrapConfig,
    run_pipeline_full,
)

# ---------------------------------------------------------------------------
# EA0_SQUARED_SI (G8 sign-off gate edit 2).
# ---------------------------------------------------------------------------


def test_ea0_squared_si_matches_gate_quoted_value() -> None:
    """Pinned to the G8 sign-off's quoted ``4.4866e-40`` (their rounding)."""
    np.testing.assert_allclose(EA0_SQUARED_SI, 4.4866e-40, rtol=2e-5, atol=0)


def test_ea0_squared_si_is_computed_from_e_and_a0_not_hand_transcribed() -> None:
    """`EA0_SQUARED_SI` == `e * a0^2` from `cliffordclock.constants`, exactly
    (a live check that the two pinned constants and this derived one never
    silently diverge, not just a numeric coincidence).
    """
    assert EA0_SQUARED_SI == constants.ELEMENTARY_CHARGE * constants.BOHR_RADIUS**2


def test_ea0_squared_si_applies_e_exactly_once() -> None:
    """The G8 'double-e' trap, directly: `EA0_SQUARED_SI` must equal
    `e * a0^2`, NOT `e^2 * a0^2` (double-counting `e`) and NOT plain `a0^2`
    (omitting `e` entirely) -- both wrong by a factor of `e` in either
    direction.
    """
    a0_sq = constants.BOHR_RADIUS**2
    e = constants.ELEMENTARY_CHARGE
    assert pytest.approx(e * a0_sq, rel=1e-15, abs=0) == EA0_SQUARED_SI
    assert pytest.approx(e * e * a0_sq, rel=1e-6, abs=0) != EA0_SQUARED_SI
    assert pytest.approx(a0_sq, rel=1e-6, abs=0) != EA0_SQUARED_SI


def test_ea0_squared_si_dimensional_round_trip() -> None:
    """``[C m^2] . [V/m^2] = J`` (CONVENTIONS.md E34): a representative
    quadrupole-moment-times-gradient product lands in a physically sane
    joule range for a realistic ion-trap gradient, not many orders of
    magnitude off (a coarse sanity check catching a stray missing/extra
    factor of `e`, `a0`, or `4`).
    """
    theta_si = 2.0 * EA0_SQUARED_SI  # ~ a D-state Theta, C m^2
    gradient_v_per_m2 = 1.0e8  # a realistic ion-trap dE/dz, V/m^2
    shift_j = 0.25 * theta_si * gradient_v_per_m2
    # h*nu_0 for an optical clock transition is ~1e-19 J; a realistic
    # quadrupole shift is a small fraction of that, not many orders larger
    # (a double-e bug would land ~1e-19 smaller; a missing-e bug ~1e19 larger).
    assert 1e-35 < abs(shift_j) < 1e-15


# ---------------------------------------------------------------------------
# Al27+ / In115+ Species entries.
# ---------------------------------------------------------------------------


def test_al27_plus_delta_alpha_pinned_to_wei_2024_secondary() -> None:
    """G8 sign-off B1: adopt Wei et al. 2024's 0.416(14) a.u. as the
    registry pin (secondary label; see the registry docstring), with
    Brewer 2019's 0.426(58) a.u. as the primary-text fallback recorded
    there.
    """
    species = get_species("Al27+")
    assert species.delta_alpha_dc_si is not None
    np.testing.assert_allclose(
        species.delta_alpha_dc_si, 0.416 * ALPHA_AU_TO_SI, rtol=1e-12, atol=0
    )


def test_al27_plus_delta_alpha_consistent_with_brewer_2019_within_1sigma() -> None:
    """Wei 2024 (0.416(14) a.u.) and Brewer 2019 (0.426(58) a.u.) agree at
    ~0.17sigma (dossier section 1) -- a live check, not just an assertion.
    """
    wei = 0.416
    wei_sigma = 0.014
    brewer = 0.426
    brewer_sigma = 0.058
    diff = abs(wei - brewer)
    combined_sigma = (wei_sigma**2 + brewer_sigma**2) ** 0.5
    assert diff / combined_sigma < 0.3  # dossier: 0.17sigma


def test_al27_plus_has_no_bbr_coefficients() -> None:
    """WP21: no independent static/dynamic BBR split is invented for
    Al27+ (only a single measured-shift-at-one-temperature datum exists).
    """
    species = get_species("Al27+")
    assert species.bbr_coefficients is None
    with pytest.raises(ValueError, match="no BBR shift data"):
        species.resolve_bbr_coefficients()


def test_in115_plus_registered_with_safronova_2011_theory_delta_alpha() -> None:
    """G8 sign-off B2: Safronova 2011 theory value Delta_alpha(0) = 2.01
    a.u. pinned; the un-cross-checked 2024 '3.3(3)' value is NOT pinned.
    """
    species = get_species("In115+")
    assert species.delta_alpha_dc_si is not None
    np.testing.assert_allclose(species.delta_alpha_dc_si, 2.01 * ALPHA_AU_TO_SI, rtol=1e-12, atol=0)
    # The un-cross-checked value would be very different in a.u. terms;
    # confirm it is nowhere near 3.3 a.u.
    assert species.delta_alpha_dc_si / ALPHA_AU_TO_SI != pytest.approx(3.3, rel=0.05, abs=0)


def test_in115_plus_has_no_bbr_coefficients() -> None:
    species = get_species("In115+")
    assert species.bbr_coefficients is None
    with pytest.raises(ValueError, match="no BBR shift data"):
        species.resolve_bbr_coefficients()


def test_in115_plus_clock_frequency_pinned_to_ohtsubo_2017() -> None:
    """Ohtsubo, Li, Matsubara, Ido, Hayasaka, Opt. Express 25, 11725
    (2017), arXiv:1703.02717: 1 267 402 452 901 049.9(6.9) Hz.
    """
    species = get_species("In115+")
    np.testing.assert_allclose(
        species.clock_frequency_hz, 1_267_402_452_901_049.9, rtol=0, atol=1e-3
    )


def test_four_species_registered() -> None:
    for name in ("Sr87", "Yb171", "Al27+", "In115+"):
        get_species(name)  # does not raise


# ---------------------------------------------------------------------------
# QUADRUPOLE_MOMENTS registry (G8 sign-off B3).
# ---------------------------------------------------------------------------


def test_quadrupole_moments_registry_has_expected_states() -> None:
    assert set(QUADRUPOLE_MOMENTS) == {
        "Ca+:D5/2",
        "Sr+:D5/2",
        "Ba+:D5/2",
        "Yb+:D3/2",
        "Yb+:F7/2",
    }


def test_ca_plus_theta_pinned_primary_verified() -> None:
    """Ca+ D5/2: 1.83(1) ea0^2, Roos et al. quant-ph/0701215v1 -- upgraded
    to PRIMARY VERIFIED (dossier section 6, owner-supplied full read).
    """
    moment = get_quadrupole_moment("Ca+:D5/2")
    assert moment.j == 2.5
    np.testing.assert_allclose(moment.theta_au, 1.83, rtol=0, atol=1e-9)
    np.testing.assert_allclose(moment.theta_au_uncertainty, 0.01, rtol=0, atol=1e-9)
    assert moment.verification == "primary"


def test_sr_plus_theta_pinned_shaniv_2016() -> None:
    """Sr+ D5/2: 2.973(+26/-33) ea0^2, Shaniv/Akerman/Ozeri 2016 --
    supersedes Barwood 2004's 2.6(3) (G8 sign-off B3).
    """
    moment = get_quadrupole_moment("Sr+:D5/2")
    assert moment.j == 2.5
    np.testing.assert_allclose(moment.theta_au, 2.973, rtol=0, atol=1e-9)
    assert moment.verification == "secondary"


def test_yb_plus_f7_2_theta_is_negative_sign_anchor() -> None:
    """Yb+ F7/2: -0.041(5) ea0^2 (Huntemann 2012, PRIMARY) -- the
    registry's negative-Theta sign anchor (G8 gate edit 1, A1 regression).
    """
    moment = get_quadrupole_moment("Yb+:F7/2")
    assert moment.j == 3.5
    assert moment.theta_au < 0.0
    np.testing.assert_allclose(moment.theta_au, -0.041, rtol=0, atol=1e-9)
    assert moment.verification == "primary"


def test_every_d_state_theta_is_positive_except_yb_f7_2() -> None:
    for key, moment in QUADRUPOLE_MOMENTS.items():
        if key == "Yb+:F7/2":
            assert moment.theta_au < 0.0
        else:
            assert moment.theta_au > 0.0


def test_get_quadrupole_moment_unknown_state_raises() -> None:
    with pytest.raises(KeyError, match="Unknown quadrupole state"):
        get_quadrupole_moment("Not+:A/State")


# ---------------------------------------------------------------------------
# Micromotion boundary / hyperfine-E2 budget notes (G8 gate edits 4/5/6).
# ---------------------------------------------------------------------------


def test_ion_micromotion_notes_present_for_both_ion_species() -> None:
    for name in ("Al27+", "In115+"):
        assert name in ION_MICROMOTION_NOTES
        note = ION_MICROMOTION_NOTES[name]
        # Shared-cause wording (G8 gate edit 5, required edit).
        assert "same stray" in note
        # Strongest-for-J=0 wording (G8 gate edit 5, required edit).
        assert "J=0" in note


def test_ion_hyperfine_e2_budget_notes_present_for_both_ion_species() -> None:
    for name in ("Al27+", "In115+"):
        assert name in ION_HYPERFINE_E2_BUDGET_NOTES
        note = ION_HYPERFINE_E2_BUDGET_NOTES[name]
        assert "Beloy" in note
        assert "budget line" in note


def test_micromotion_notes_not_present_for_neutral_atom_species() -> None:
    for name in ("Sr87", "Yb171"):
        assert name not in ION_MICROMOTION_NOTES
        assert name not in ION_HYPERFINE_E2_BUDGET_NOTES


# ---------------------------------------------------------------------------
# Tier-1 KA-style formula pins (WP21 test contract item 3): "known-answer
# tests against the literature Delta_alpha arithmetic", mirroring
# tests/test_stark_pivot.py's Sr87/Yb171 formula-pin tests, now for the two
# ion species.
# ---------------------------------------------------------------------------


def test_al27_plus_formula_pin_at_1000_v_per_m() -> None:
    """P - 1 for Al27+ at |E| = 1000 V/m equals the hand-computed
    -Delta_alpha |E|^2 / (2 h nu_0), same KA-style check as
    tests/test_stark_pivot.py's Sr87/Yb171 cases (WP7 test contract item
    1), now exercised against the WP21 Al27+ registry entry.
    """
    species = get_species("Al27+")
    assert species.delta_alpha_dc_si is not None
    e_mag = 1000.0
    e0 = jnp.array([e_mag, 0.0, 0.0])
    delta_e = jnp.zeros(3)
    expected = (
        -(species.delta_alpha_dc_si / 2.0) * e_mag**2 / (PLANCK_H * species.clock_frequency_hz)
    )
    got = pivot_perturbation_stark(e0, delta_e, species)
    np.testing.assert_allclose(float(got), expected, rtol=1e-12, atol=0)


def test_in115_plus_formula_pin_at_1000_v_per_m() -> None:
    species = get_species("In115+")
    assert species.delta_alpha_dc_si is not None
    e_mag = 1000.0
    e0 = jnp.array([0.0, e_mag, 0.0])
    delta_e = jnp.zeros(3)
    expected = (
        -(species.delta_alpha_dc_si / 2.0) * e_mag**2 / (PLANCK_H * species.clock_frequency_hz)
    )
    got = pivot_perturbation_stark(e0, delta_e, species)
    np.testing.assert_allclose(float(got), expected, rtol=1e-12, atol=0)


def _lattice_stark_config(species: str, output_dir: str, mode: str) -> PipelineConfig:
    return PipelineConfig(
        species=species,
        trap=TrapConfig(omega_xyz=(2.0e5, 2.0e5, 2.0e5)),
        field_config=FieldConfig(
            synthetic=SyntheticFieldConfig(kind="uniform", params={"e0": [10.0, 0.0, -5.0]})
        ),
        ensemble=EnsembleConfig(
            regime="lattice", temperature_uK=1.0, motional_n=(0, 0, 0), n_quad=1
        ),
        integration=IntegrationConfig(mode=mode, time_s=1.0)
        if mode == "fast_path"
        else IntegrationConfig(mode=mode, dtau=0.5, steps=200),
        coupling=CouplingConfig(type="stark_dc"),
        output=OutputConfig(directory=output_dir),
    )


@pytest.mark.parametrize("species_name", ["Al27+", "In115+"])
def test_ion_species_fast_path_and_worldline_agree_exactly(
    species_name: str, tmp_path: Path
) -> None:
    """Cross-mode agreement (WP21 test contract item 3), the same E29
    static-node exact-reduction argument
    `tests/test_e2e.py::test_step0_stark_dc_fast_path_and_worldline_agree_exactly`
    already exercises for Sr87, now exercised for both WP21 ion species.
    """
    fast_cfg = _lattice_stark_config(species_name, str(tmp_path / "fast"), "fast_path")
    worldline_cfg = _lattice_stark_config(species_name, str(tmp_path / "world"), "worldline")
    fast_result = run_pipeline_full(fast_cfg)
    worldline_result = run_pipeline_full(worldline_cfg)
    np.testing.assert_allclose(
        fast_result.report.mean_fractional_shift,
        worldline_result.report.mean_fractional_shift,
        rtol=0,
        atol=0,
    )
    # Nonzero (not the vacuous "both sides are 0.0" case) -- the ion
    # species' smaller Delta_alpha gives a smaller shift than Sr87's own
    # version of this test, so the floor is lower, not absent.
    assert abs(fast_result.report.mean_fractional_shift) > 1e-25
