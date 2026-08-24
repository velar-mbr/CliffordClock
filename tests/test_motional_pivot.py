# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP30 quantum-motional second-order-Doppler (time-dilation)
pivot term (CONVENTIONS.md section 16, E38).

``cliffordclock.integrator.omega.motional_pivot_perturbation``/
``motional_pivot_uncertainty`` implement E38's formula and its uncertainty
propagation; ``pivot_perturbation_stark``'s new ``motional_pivot_perturbation``
keyword implements its additive composition (E33's pattern), mirroring
``bbr_pivot_perturbation`` exactly (spatially uniform, one motional state
per run). This file covers: the hand-computed two-mode regression (Decimal
cross-check), the ground-state zero-point floor, two independently-coded
kill tests (forgotten zero-point term, forgotten 2*pi conversion),
uncertainty partials against finite differences, input validation, rotor/
spin-connection composition scope, and the pipeline-level config parsing,
the no-double-counting regime restriction (the central CONVENTIONS.md E38
argument), cross-mode agreement, byte-exactness of shipped examples, and
report-note content for ``environment.motional_state``.
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from cliffordclock.constants import ATOMIC_MASS_UNIT, HBAR, SPEED_OF_LIGHT
from cliffordclock.ensemble.species import get_species
from cliffordclock.integrator.omega import (
    ClockIonMathieuParameters,
    MotionalMode,
    axial_coulomb_curvature,
    bbr_pivot_perturbation,
    build_omega_stark,
    clock_ion_mathieu_parameters,
    grav_pivot_perturbation,
    motional_mean_squared_velocity_m2_s2,
    motional_pivot_perturbation,
    motional_pivot_uncertainty,
    pivot_perturbation_stark,
    predicted_partner_bare_radial_frequencies_hz,
    quadrupole_pivot_perturbation,
    radial_micromotion_enhancement,
    spin_connection_stark,
    stark_pivot_terms,
    two_ion_participations,
    two_ion_radial_participations,
)
from cliffordclock.pipeline import (
    _FAST_PATH_DOPPLER_EXCLUSION_NOTE,
    _FAST_PATH_MOTIONAL_INCLUDED_NOTE,
    EnvironmentConfig,
    MotionalStateConfig,
    PipelineConfig,
    PipelineConfigError,
    _parse_environment,
    _parse_motional_state,
    run_pipeline_full,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

_AL27_PLUS = get_species("Al27+")

# ---------------------------------------------------------------------------
# 1. Hand-computed two-mode regression (Decimal cross-check), Al+-like
#    numbers: f ~ few MHz, n_bar ~ 0.05, mass 26.98 u (species registry).
# ---------------------------------------------------------------------------

#: Two illustrative Al+-like modes (CONVENTIONS.md E38's own worked example
#: class): near-ground-state occupation (n_bar ~ 0.05) at a couple of MHz,
#: the realistic sideband-thermometry regime for a well-cooled trapped ion
#: (cf. the registry's own Al27+ secular-motion citation,
#: ``cliffordclock.ensemble.species.ION_MICROMOTION_NOTES["Al27+"]``:
#: Brewer et al. 2019's measured secular-motion budget line is
#: -17.3(2.9)e-19, the same 1e-19-to-1e-18 order of magnitude this
#: two-mode case lands in).
_TWO_MODES = (
    MotionalMode(
        name="axial",
        frequency_hz=2.0e6,
        n_bar=0.05,
        n_bar_uncertainty=0.01,
        frequency_uncertainty_hz=1.0e3,
    ),
    MotionalMode(
        name="radial",
        frequency_hz=4.0e6,
        n_bar=0.05,
        n_bar_uncertainty=0.02,
        frequency_uncertainty_hz=2.0e3,
    ),
)


def _decimal_two_mode_reference() -> Decimal:
    """Independent 50-digit `decimal` computation of the two-mode `_TWO_MODES`
    case's `(P-1)_motional`, coded directly from CONVENTIONS.md E38's formula
    (hand-written here, not calling anything from `cliffordclock.integrator.omega`)::

        <v^2> = sum_i (hbar * (2*pi*f_i) / m) * (n_bar_i + 1/2)
        (P-1)_motional = -<v^2> / (2*c^2)
    """
    getcontext().prec = 50
    hbar = Decimal(repr(HBAR))
    c = Decimal(repr(SPEED_OF_LIGHT))
    m = Decimal(repr(_AL27_PLUS.mass_kg))
    pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")
    total_v2 = Decimal(0)
    for mode in _TWO_MODES:
        omega = 2 * pi * Decimal(repr(mode.frequency_hz))
        total_v2 += (hbar * omega / m) * (Decimal(repr(mode.n_bar)) + Decimal("0.5"))
    return -total_v2 / (2 * c * c)


def test_motional_pivot_two_mode_hand_computed_regression() -> None:
    """The real engine function matches an independently-coded 50-digit
    `decimal` reference to float64 noise (CONVENTIONS.md E38's worked
    formula), and the sanity-checked magnitude lands in the realistic
    1e-19-to-1e-18 trapped-ion-clock secular-motion range (never
    absurdly large or small: not off by the (2*pi)^2 ~= 39.5 factor a
    missing angular-frequency conversion would produce, nor by orders of
    magnitude from an outright wrong formula).
    """
    value = motional_pivot_perturbation(_TWO_MODES, _AL27_PLUS)
    reference = float(_decimal_two_mode_reference())
    np.testing.assert_allclose(value, reference, rtol=1e-12, atol=0)

    # Sanity: negative (a redshift, exactly like the classical kinematic
    # term it generalizes), and within an order of magnitude of 1e-18 --
    # the realistic trapped-ion secular-motion scale (CONVENTIONS.md E38
    # motivation; Brewer et al. 2019's own -17.3e-19 secular-motion row,
    # already cited in this project's Al27+ registry notes).
    assert value < 0.0
    assert 1e-19 < abs(value) < 1e-17


def test_motional_mean_squared_velocity_matches_pivot_perturbation() -> None:
    """`(P-1)_motional = -<v^2>/(2c^2)` exactly, cross-checking the two public
    entry points against each other."""
    mean_v2 = motional_mean_squared_velocity_m2_s2(_TWO_MODES, _AL27_PLUS)
    value = motional_pivot_perturbation(_TWO_MODES, _AL27_PLUS)
    np.testing.assert_allclose(value, -mean_v2 / (2.0 * SPEED_OF_LIGHT**2), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 2. Ground-state limit: n_bar=0 for every mode leaves the zero-point (1/2)
#    term alone, NOT zero.
# ---------------------------------------------------------------------------


def test_motional_pivot_ground_state_limit_is_nonzero_zero_point_floor() -> None:
    ground_state_modes = (MotionalMode(name="axial", frequency_hz=2.0e6, n_bar=0.0),)
    value = motional_pivot_perturbation(ground_state_modes, _AL27_PLUS)
    assert value != 0.0, (
        "n_bar=0 must NOT give a zero shift: the +1/2 zero-point term alone "
        "still contributes (CONVENTIONS.md E38's ground-state-limit note)"
    )
    assert value < 0.0

    # Exactly half of the n_bar=0.5-equivalent (i.e. the n_bar=0 case IS the
    # n_bar_i + 1/2 = 0.5 case): matches a hand pin at full precision.
    expected = -(HBAR * 2.0 * math.pi * 2.0e6 / _AL27_PLUS.mass_kg * 0.5) / (
        2.0 * SPEED_OF_LIGHT**2
    )
    np.testing.assert_allclose(value, expected, rtol=1e-14, atol=0)


def test_motional_pivot_higher_n_bar_gives_larger_magnitude_shift() -> None:
    low = motional_pivot_perturbation((MotionalMode(frequency_hz=2.0e6, n_bar=0.0),), _AL27_PLUS)
    high = motional_pivot_perturbation((MotionalMode(frequency_hz=2.0e6, n_bar=5.0),), _AL27_PLUS)
    assert abs(high) > abs(low)


# ---------------------------------------------------------------------------
# 3. Kill tests: two independently-coded WRONG implementations, computed
#    directly in this test file (not calling the real engine functions),
#    asserted to differ from the real function's output.
# ---------------------------------------------------------------------------


def _wrong_forgot_zero_point(modes: tuple[MotionalMode, ...], mass_kg: float) -> float:
    """A plausible bug: uses `n_bar_i` alone, dropping the `+1/2` zero-point term."""
    total_v2 = 0.0
    for mode in modes:
        omega = 2.0 * math.pi * mode.frequency_hz
        total_v2 += (HBAR * omega / mass_kg) * mode.n_bar  # BUG: missing "+ 0.5"
    return -total_v2 / (2.0 * SPEED_OF_LIGHT**2)


def _wrong_used_f_not_omega(modes: tuple[MotionalMode, ...], mass_kg: float) -> float:
    """A plausible bug: uses the ORDINARY frequency `f_i` directly as if it
    were the angular frequency, skipping the `omega_i = 2*pi*f_i` conversion.
    """
    total_v2 = 0.0
    for mode in modes:
        total_v2 += (HBAR * mode.frequency_hz / mass_kg) * (mode.n_bar + 0.5)  # BUG: no 2*pi
    return -total_v2 / (2.0 * SPEED_OF_LIGHT**2)


def test_motional_pivot_kill_test_forgotten_zero_point_term() -> None:
    real_value = motional_pivot_perturbation(_TWO_MODES, _AL27_PLUS)
    wrong_value = _wrong_forgot_zero_point(_TWO_MODES, _AL27_PLUS.mass_kg)
    # The two must disagree by a real, non-vacuous margin (not float noise).
    assert abs(real_value - wrong_value) / abs(real_value) > 0.05
    np.testing.assert_allclose(wrong_value, -2.4682552764810302e-20, rtol=1e-9, atol=0)


def test_motional_pivot_kill_test_used_frequency_not_angular_frequency() -> None:
    real_value = motional_pivot_perturbation(_TWO_MODES, _AL27_PLUS)
    wrong_value = _wrong_used_f_not_omega(_TWO_MODES, _AL27_PLUS.mass_kg)
    # Missing the 2*pi factor understates |shift| by exactly 2*pi (linear in
    # omega, unlike the (2*pi)^2 overstatement a caller who instead SUPPLIED
    # an angular frequency as frequency_hz would produce, per CONVENTIONS.md
    # E38's explicit warning); still a large, non-vacuous discrepancy.
    assert abs(real_value - wrong_value) / abs(real_value) > 0.5
    np.testing.assert_allclose(wrong_value, -4.321185308710697e-20, rtol=1e-9, atol=0)


# ---------------------------------------------------------------------------
# 4. Uncertainty propagation: partials against independently-computed finite
#    differences (perturbing exactly one input at a time).
# ---------------------------------------------------------------------------


def test_motional_pivot_uncertainty_n_bar_partial_matches_finite_difference() -> None:
    base_shift = motional_pivot_perturbation(_TWO_MODES, _AL27_PLUS)
    eps = 1e-6
    perturbed = (
        MotionalMode(
            name=_TWO_MODES[0].name,
            frequency_hz=_TWO_MODES[0].frequency_hz,
            n_bar=_TWO_MODES[0].n_bar + eps,
        ),
        _TWO_MODES[1],
    )
    finite_diff = (motional_pivot_perturbation(perturbed, _AL27_PLUS) - base_shift) / eps
    analytic = -(HBAR * 2.0 * math.pi * _TWO_MODES[0].frequency_hz / _AL27_PLUS.mass_kg) / (
        2.0 * SPEED_OF_LIGHT**2
    )
    np.testing.assert_allclose(finite_diff, analytic, rtol=1e-6, atol=0)


def test_motional_pivot_uncertainty_frequency_partial_matches_finite_difference() -> None:
    base_shift = motional_pivot_perturbation(_TWO_MODES, _AL27_PLUS)
    eps_hz = 1.0
    perturbed = (
        MotionalMode(
            name=_TWO_MODES[0].name,
            frequency_hz=_TWO_MODES[0].frequency_hz + eps_hz,
            n_bar=_TWO_MODES[0].n_bar,
        ),
        _TWO_MODES[1],
    )
    finite_diff = (motional_pivot_perturbation(perturbed, _AL27_PLUS) - base_shift) / eps_hz
    analytic = -(HBAR * 2.0 * math.pi * (_TWO_MODES[0].n_bar + 0.5) / _AL27_PLUS.mass_kg) / (
        2.0 * SPEED_OF_LIGHT**2
    )
    np.testing.assert_allclose(finite_diff, analytic, rtol=1e-6, atol=0)


def test_motional_pivot_uncertainty_v_rms_emm_partial_matches_finite_difference() -> None:
    # Evaluated at a NONZERO base_v: d(P-1)/d(v_rms_emm) is proportional to
    # v_rms_emm itself (the term is quadratic in v_rms_emm), so at base_v=0
    # the true derivative is exactly zero and a finite difference there
    # would be dominated entirely by curvature, not the linear term this
    # test targets.
    base_v = 5.0
    eps = 1e-3
    base_shift = motional_pivot_perturbation(_TWO_MODES, _AL27_PLUS, v_rms_emm_m_s=base_v)
    perturbed_shift = motional_pivot_perturbation(
        _TWO_MODES, _AL27_PLUS, v_rms_emm_m_s=base_v + eps
    )
    finite_diff = (perturbed_shift - base_shift) / eps
    analytic = -base_v / SPEED_OF_LIGHT**2
    # rtol=2e-4, not 1e-6: this term is exactly quadratic in v_rms_emm
    # (-v_rms_emm^2/(2c^2)), so a forward finite difference at eps=1e-3
    # carries a real O(eps/base_v) curvature truncation error (~1e-4
    # relative here) distinct from float64 noise, still four orders of
    # magnitude tighter than a vacuous bound.
    np.testing.assert_allclose(finite_diff, analytic, rtol=2e-4, atol=0)


def test_motional_pivot_uncertainty_combines_every_input_in_quadrature() -> None:
    """`motional_pivot_uncertainty` matches an independently-assembled
    quadrature sum of every mode's n_bar/frequency partial times its
    uncertainty, plus the EMM partial times its uncertainty, computed
    here from the analytic partials directly (CONVENTIONS.md E38's
    uncertainty-propagation note), not by calling the function under test
    a second time.
    """
    v_rms_emm = 5.0
    v_rms_emm_unc = 0.5
    terms_sq = []
    for mode in _TWO_MODES:
        omega = 2.0 * math.pi * mode.frequency_hz
        d_dn = -(HBAR * omega / _AL27_PLUS.mass_kg) / (2.0 * SPEED_OF_LIGHT**2)
        d_df = -(HBAR * 2.0 * math.pi * (mode.n_bar + 0.5) / _AL27_PLUS.mass_kg) / (
            2.0 * SPEED_OF_LIGHT**2
        )
        terms_sq.append((d_dn * mode.n_bar_uncertainty) ** 2)
        terms_sq.append((d_df * mode.frequency_uncertainty_hz) ** 2)
    d_dv = -v_rms_emm / SPEED_OF_LIGHT**2
    terms_sq.append((d_dv * v_rms_emm_unc) ** 2)
    expected = math.sqrt(sum(terms_sq))

    actual = motional_pivot_uncertainty(
        _TWO_MODES, _AL27_PLUS, v_rms_emm_m_s=v_rms_emm, v_rms_emm_uncertainty_m_s=v_rms_emm_unc
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_motional_pivot_uncertainty_zero_when_every_input_uncertainty_is_zero() -> None:
    modes = (MotionalMode(frequency_hz=2.0e6, n_bar=0.05),)
    assert motional_pivot_uncertainty(modes, _AL27_PLUS) == 0.0


# ---------------------------------------------------------------------------
# 5. Input validation (omega.py level).
# ---------------------------------------------------------------------------


def test_motional_pivot_rejects_empty_modes() -> None:
    with pytest.raises(ValueError, match="at least one MotionalMode"):
        motional_pivot_perturbation((), _AL27_PLUS)


def test_motional_pivot_rejects_non_positive_frequency() -> None:
    with pytest.raises(ValueError, match="frequency_hz=0.0.*must be > 0"):
        motional_pivot_perturbation((MotionalMode(frequency_hz=0.0, n_bar=0.0),), _AL27_PLUS)
    with pytest.raises(ValueError, match="must be > 0"):
        motional_pivot_perturbation((MotionalMode(frequency_hz=-1.0, n_bar=0.0),), _AL27_PLUS)


def test_motional_pivot_rejects_negative_n_bar() -> None:
    with pytest.raises(ValueError, match="n_bar=-0.1.*must be >= 0"):
        motional_pivot_perturbation((MotionalMode(frequency_hz=1e6, n_bar=-0.1),), _AL27_PLUS)


def test_motional_pivot_rejects_negative_uncertainties() -> None:
    with pytest.raises(ValueError, match="n_bar_uncertainty=-0.1.*must be >= 0"):
        motional_pivot_perturbation(
            (MotionalMode(frequency_hz=1e6, n_bar=0.0, n_bar_uncertainty=-0.1),), _AL27_PLUS
        )
    with pytest.raises(ValueError, match="frequency_uncertainty_hz=-1.0.*must be >= 0"):
        motional_pivot_perturbation(
            (MotionalMode(frequency_hz=1e6, n_bar=0.0, frequency_uncertainty_hz=-1.0),),
            _AL27_PLUS,
        )


def test_motional_pivot_rejects_negative_v_rms_emm() -> None:
    with pytest.raises(ValueError, match="v_rms_emm_m_s=-1.0.*must be >= 0"):
        motional_pivot_perturbation(
            (MotionalMode(frequency_hz=1e6, n_bar=0.0),), _AL27_PLUS, v_rms_emm_m_s=-1.0
        )


# ---------------------------------------------------------------------------
# 6. Composition additivity (E33's pattern) vs. Stark/BBR/quadrupole/gravity,
#    at the omega.py level.
# ---------------------------------------------------------------------------


def test_motional_pivot_composes_additively_with_stark() -> None:
    species = get_species("Sr87")
    e0 = jnp.array([1e5, 0.0, 0.0])
    zeros = jnp.zeros(3)
    stark_only = float(pivot_perturbation_stark(e0, zeros, species))
    motional_only = motional_pivot_perturbation(_TWO_MODES, _AL27_PLUS)
    composed = float(
        pivot_perturbation_stark(e0, zeros, species, motional_pivot_perturbation=motional_only)
    )
    np.testing.assert_allclose(composed, stark_only + motional_only, rtol=0, atol=1e-30)


def test_motional_pivot_composes_additively_with_every_other_term() -> None:
    """All five terms (Stark, BBR, quadrupole, gravity, motional) sum
    linearly in (P-1) with no cross term at this project's working
    precision.
    """
    species = get_species("Sr87")
    e0 = jnp.array([1e5, 0.0, 2e4])
    zeros = jnp.zeros(3)
    stark_only = float(pivot_perturbation_stark(e0, zeros, species))
    bbr_only = bbr_pivot_perturbation(300.0, species)
    grad_e = jnp.array([[0.0, 0.0, 1e3], [0.0, 0.0, 0.0], [1e3, 0.0, 0.0]])
    quad_only = float(
        quadrupole_pivot_perturbation(
            grad_e, jnp.array([0.0, 0.0, 1.0]), -0.041, 3.5, 3.5, species.clock_frequency_hz
        )
    )
    grav_only = float(grav_pivot_perturbation(jnp.asarray(0.5), 9.80665))
    motional_only = motional_pivot_perturbation(_TWO_MODES, _AL27_PLUS)

    composed = float(
        pivot_perturbation_stark(
            e0,
            zeros,
            species,
            bbr_pivot_perturbation=bbr_only,
            quadrupole_pivot_perturbation=quad_only,
            grav_pivot_perturbation=grav_only,
            motional_pivot_perturbation=motional_only,
        )
    )
    np.testing.assert_allclose(
        composed,
        stark_only + bbr_only + quad_only + grav_only + motional_only,
        rtol=0,
        atol=1e-28,
    )


# ---------------------------------------------------------------------------
# 7. Rotor/spin-connection scope: motional shifts the P denominator only,
#    exactly like BBR (spatially uniform).
# ---------------------------------------------------------------------------


def test_motional_shifts_spin_connection_denominator_only() -> None:
    species = get_species("Sr87")
    e_total = jnp.array([1e4, 2e4, 3e4])
    grad_e = jnp.array([[1e2, 2e2, 3e2], [4e2, 5e2, 6e2], [7e2, 8e2, 9e2]])

    omega_0k_no_motional = spin_connection_stark(e_total, grad_e, species)
    motional_value = 1e-8
    omega_0k_with_motional = spin_connection_stark(
        e_total, grad_e, species, motional_pivot_perturbation=motional_value
    )

    baseline, cross, quadratic = stark_pivot_terms(e_total, jnp.zeros_like(e_total), species)
    p_no_motional = 1.0 + float(baseline + cross + quadratic)
    p_with_motional = p_no_motional + motional_value

    expected_ratio = p_no_motional / p_with_motional
    actual_ratio = np.asarray(omega_0k_with_motional) / np.asarray(omega_0k_no_motional)
    np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-12, atol=0)


def test_motional_reaches_the_rotation_coefficient_at_v_zero() -> None:
    """At v=0 (every lattice/lattice_extended node), the motional term reaches
    the `B_hat_C` rotation-plane coefficient by exactly its own value
    (gamma_inv=1 exactly), and omega_boost stays identically zero
    regardless of the motional term, mirroring the gravity/BBR tests.
    """
    from cliffordclock.cl13 import IDX_E01, IDX_E02, IDX_E03, IDX_E12

    species = get_species("Sr87")
    e_total = jnp.array([1e4, 0.0, 0.0])
    grad_e = jnp.zeros((3, 3))
    v = jnp.zeros(3)

    omega_no_motional = build_omega_stark(e_total, grad_e, species, v)
    omega_with_motional = build_omega_stark(
        e_total, grad_e, species, v, motional_pivot_perturbation=1e-10
    )

    for idx in (IDX_E01, IDX_E02, IDX_E03):
        assert float(omega_no_motional[idx]) == 0.0
        assert float(omega_with_motional[idx]) == 0.0

    delta = float(omega_with_motional[IDX_E12]) - float(omega_no_motional[IDX_E12])
    np.testing.assert_allclose(delta, 1e-10, rtol=1e-9, atol=0)


# ---------------------------------------------------------------------------
# 8. Pipeline-level config parsing / validation.
# ---------------------------------------------------------------------------


def test_environment_motional_state_absent_defaults_to_none() -> None:
    assert _parse_environment(None).motional_state is None
    assert _parse_environment({}).motional_state is None
    assert _parse_motional_state(None) is None


def test_environment_motional_state_parses_modes() -> None:
    cfg = _parse_motional_state(
        {
            "modes": [
                {"name": "axial", "frequency_Hz": 2.0e6, "n_bar": 0.05},
                {
                    "name": "radial",
                    "frequency_Hz": 4.0e6,
                    "n_bar": 0.05,
                    "n_bar_uncertainty": 0.01,
                    "frequency_uncertainty_Hz": 1.0e3,
                },
            ]
        }
    )
    assert cfg is not None
    assert len(cfg.modes) == 2
    assert cfg.modes[0] == MotionalMode(name="axial", frequency_hz=2.0e6, n_bar=0.05)
    assert cfg.modes[1] == MotionalMode(
        name="radial",
        frequency_hz=4.0e6,
        n_bar=0.05,
        n_bar_uncertainty=0.01,
        frequency_uncertainty_hz=1.0e3,
    )
    assert cfg.v_rms_emm_m_s == 0.0
    assert cfg.v_rms_emm_uncertainty_m_s == 0.0


def test_environment_motional_state_parses_emm_fields() -> None:
    cfg = _parse_motional_state(
        {
            "modes": [{"name": "axial", "frequency_Hz": 2.0e6, "n_bar": 0.05}],
            "v_rms_emm_m_s": 0.02,
            "v_rms_emm_uncertainty_m_s": 0.005,
        }
    )
    assert cfg is not None
    assert cfg.v_rms_emm_m_s == 0.02
    assert cfg.v_rms_emm_uncertainty_m_s == 0.005


def test_environment_motional_state_rejects_empty_modes_list() -> None:
    with pytest.raises(PipelineConfigError, match="modes must be a non-empty list"):
        _parse_motional_state({"modes": []})
    with pytest.raises(PipelineConfigError, match="modes must be a non-empty list"):
        _parse_motional_state({})


def test_environment_motional_state_rejects_non_positive_frequency() -> None:
    with pytest.raises(PipelineConfigError, match="frequency_Hz=0.0.*must be > 0"):
        _parse_motional_state({"modes": [{"frequency_Hz": 0.0, "n_bar": 0.0}]})


def test_environment_motional_state_rejects_negative_n_bar() -> None:
    with pytest.raises(PipelineConfigError, match="n_bar=-0.1.*must be >= 0"):
        _parse_motional_state({"modes": [{"frequency_Hz": 1e6, "n_bar": -0.1}]})


def test_environment_motional_state_rejects_negative_v_rms_emm() -> None:
    with pytest.raises(PipelineConfigError, match="v_rms_emm_m_s=-1.0.*must be >= 0"):
        _parse_motional_state(
            {"modes": [{"frequency_Hz": 1e6, "n_bar": 0.0}], "v_rms_emm_m_s": -1.0}
        )


def _al_lattice_stark_dict(tmp_path: Path, *, regime: str = "lattice") -> dict[str, object]:
    ensemble: dict[str, object] = {
        "regime": regime,
        "temperature_uK": 1.0,
        "motional_n": [0, 0, 0],
        "n_quad": 1,
    }
    if regime == "lattice_extended":
        ensemble.update(
            {
                "n_sites": 3,
                "site_spacing_m": 1e-6,
                "site_axis": [0.0, 0.0, 1.0],
                "site_envelope": "uniform",
            }
        )
    return {
        "species": "Al27+",
        "trap": {"omega_xyz": [2.0e6, 2.0e6, 2.0e6], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 0.0]}}},
        "coupling": {"type": "stark_dc"},
        "ensemble": ensemble,
        "integration": {"time_s": 1e-4},
        "output": {"directory": str(tmp_path)},
    }


def _motional_section() -> dict[str, object]:
    return {
        "motional_state": {
            "modes": [
                {"name": "axial", "frequency_Hz": 2.0e6, "n_bar": 0.05},
                {"name": "radial", "frequency_Hz": 4.0e6, "n_bar": 0.05},
            ]
        }
    }


def test_environment_motional_state_requires_stark_dc_coupling(tmp_path: Path) -> None:
    data = _al_lattice_stark_dict(tmp_path)
    data["coupling"] = {"mu": [1.0e-30, 0.0, 0.0]}
    data["environment"] = _motional_section()
    with pytest.raises(PipelineConfigError, match="environment.motional_state requires"):
        PipelineConfig.from_dict(data)


def test_environment_motional_state_with_stark_dc_coupling_accepted(tmp_path: Path) -> None:
    data = _al_lattice_stark_dict(tmp_path)
    data["environment"] = _motional_section()
    config = PipelineConfig.from_dict(data)
    assert config.environment.motional_state is not None
    assert len(config.environment.motional_state.modes) == 2


# ---------------------------------------------------------------------------
# 9. The no-double-counting regime restriction (CONVENTIONS.md E38's central
#    argument, the config-parse-time kill test).
# ---------------------------------------------------------------------------


def test_environment_motional_state_rejects_classical_regime(tmp_path: Path) -> None:
    data: dict[str, object] = {
        "species": "Al27+",
        "trap": {"omega_xyz": [2.0e6, 2.0e6, 2.0e6], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 0.0]}}},
        "coupling": {"type": "stark_dc"},
        "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 10, "seed": 0},
        "integration": {"mode": "direct", "time_s": 1e-6},
        "environment": _motional_section(),
        "output": {"directory": str(tmp_path)},
    }
    with pytest.raises(PipelineConfigError, match="ensemble.regime != 'classical'") as excinfo:
        PipelineConfig.from_dict(data)
    message = str(excinfo.value)
    assert "double-count" in message
    assert "sqrt(1 - v^2/c^2)" in message


@pytest.mark.parametrize("regime", ["lattice", "lattice_extended"])
def test_environment_motional_state_accepts_static_node_regimes(
    regime: str, tmp_path: Path
) -> None:
    data = _al_lattice_stark_dict(tmp_path, regime=regime)
    data["environment"] = _motional_section()
    data["integration"] = {"mode": "fast_path", "time_s": 1e-4}
    config = PipelineConfig.from_dict(data)
    assert config.environment.motional_state is not None


# ---------------------------------------------------------------------------
# 10. Static-node regimes' reported shift matches a direct
#     motional_pivot_perturbation call to 1e-25 absolute (zero applied field
#     isolates the motional term exactly).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ["lattice", "lattice_extended"])
@pytest.mark.parametrize("mode", ["fast_path", "worldline"])
def test_motional_report_matches_direct_function_call_to_1e_25_abs(
    regime: str, mode: str, tmp_path: Path
) -> None:
    data = _al_lattice_stark_dict(tmp_path, regime=regime)
    data["environment"] = _motional_section()
    data["integration"] = {"mode": mode, "time_s": 1e-4}
    config = PipelineConfig.from_dict(data)
    result = run_pipeline_full(config)

    expected = motional_pivot_perturbation(
        (
            MotionalMode(name="axial", frequency_hz=2.0e6, n_bar=0.05),
            MotionalMode(name="radial", frequency_hz=4.0e6, n_bar=0.05),
        ),
        _AL27_PLUS,
    )
    # Zero applied field (uniform e0=[0,0,0]) means the Stark term is
    # exactly zero, so the entire reported shift IS the motional term.
    np.testing.assert_allclose(result.report.mean_fractional_shift, expected, rtol=0, atol=1e-25)


def test_motional_fast_path_matches_worldline_rotor_crosscheck(tmp_path: Path) -> None:
    """E29's exact-agreement claim, extended with the motional term active:
    static v=0 quadrature nodes mean the rotor's omega_boost is identically
    zero, so mode="worldline" must reproduce mode="fast_path" exactly.
    """
    base = _al_lattice_stark_dict(tmp_path / "fp")
    base["environment"] = _motional_section()

    fast_path_data = dict(base)
    fast_path_data["integration"] = {"mode": "fast_path", "time_s": 1e-4}
    fast_path_result = run_pipeline_full(PipelineConfig.from_dict(fast_path_data))

    worldline_data = dict(base)
    worldline_data["output"] = {"directory": str(tmp_path / "wl")}
    worldline_data["integration"] = {"mode": "worldline", "time_s": 1e-4}
    worldline_result = run_pipeline_full(PipelineConfig.from_dict(worldline_data))

    np.testing.assert_allclose(
        worldline_result.report.mean_fractional_shift,
        fast_path_result.report.mean_fractional_shift,
        rtol=1e-9,
        atol=0,
    )
    assert abs(fast_path_result.report.mean_fractional_shift) > 1e-19


# ---------------------------------------------------------------------------
# 11. Excess micromotion (EMM) input.
# ---------------------------------------------------------------------------


def test_motional_emm_contribution_composes_report_level(tmp_path: Path) -> None:
    without_emm = dict(_al_lattice_stark_dict(tmp_path / "without"))
    without_emm["environment"] = _motional_section()
    without_emm["integration"] = {"mode": "fast_path", "time_s": 1e-4}
    without_result = run_pipeline_full(PipelineConfig.from_dict(without_emm))

    with_emm_section = _motional_section()
    with_emm_section["motional_state"]["v_rms_emm_m_s"] = 10.0
    with_emm = dict(_al_lattice_stark_dict(tmp_path / "with"))
    with_emm["environment"] = with_emm_section
    with_emm["integration"] = {"mode": "fast_path", "time_s": 1e-4}
    with_result = run_pipeline_full(PipelineConfig.from_dict(with_emm))

    delta = with_result.report.mean_fractional_shift - without_result.report.mean_fractional_shift
    expected_delta = -(10.0**2) / (2.0 * SPEED_OF_LIGHT**2)
    np.testing.assert_allclose(delta, expected_delta, rtol=1e-9, atol=0)


# ---------------------------------------------------------------------------
# 12. Byte-exactness of shipped examples, fast_path note supersession,
#     report-note content: the pipeline half of the WP30 test contract.
# ---------------------------------------------------------------------------


def test_no_shipped_example_uses_motional_state_key() -> None:
    example_paths = sorted(_EXAMPLES_DIR.glob("*.yaml"))
    assert len(example_paths) >= 5, "expected several shipped example configs"
    for path in example_paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        environment = data.get("environment")
        if environment:
            assert "motional_state" not in environment, (
                f"{path.name} unexpectedly sets 'motional_state:'"
            )


def test_fast_path_doppler_exclusion_note_present_without_motional_state(
    tmp_path: Path,
) -> None:
    data = _al_lattice_stark_dict(tmp_path)
    data["integration"] = {"mode": "fast_path", "time_s": 1e-4}
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    notes = result.report.uncertainty_notes
    assert _FAST_PATH_DOPPLER_EXCLUSION_NOTE in notes
    assert _FAST_PATH_MOTIONAL_INCLUDED_NOTE not in notes


def test_fast_path_motional_included_note_supersedes_exclusion_note(tmp_path: Path) -> None:
    data = _al_lattice_stark_dict(tmp_path)
    data["environment"] = _motional_section()
    data["integration"] = {"mode": "fast_path", "time_s": 1e-4}
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    notes = result.report.uncertainty_notes
    assert _FAST_PATH_MOTIONAL_INCLUDED_NOTE in notes
    assert _FAST_PATH_DOPPLER_EXCLUSION_NOTE not in notes


def test_motional_report_note_content(tmp_path: Path) -> None:
    data = _al_lattice_stark_dict(tmp_path)
    data["environment"] = _motional_section()
    data["integration"] = {"mode": "fast_path", "time_s": 1e-4}
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    notes = result.report.uncertainty_notes

    assert "CONVENTIONS.md section 16 E38" in notes
    assert "'axial'" in notes
    assert "'radial'" in notes
    assert "n_bar=0.05" in notes
    assert "<v^2>=" in notes
    assert "(P-1)_motional=" in notes
    assert "roadmap package" in notes
    assert "excess micromotion" in notes


def test_motional_report_note_states_emm_input_when_present(tmp_path: Path) -> None:
    data = _al_lattice_stark_dict(tmp_path)
    section = _motional_section()
    section["motional_state"]["v_rms_emm_m_s"] = 0.02
    data["environment"] = section
    data["integration"] = {"mode": "fast_path", "time_s": 1e-4}
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    assert "v_rms_emm_m_s=0.02" in result.report.uncertainty_notes


# ---------------------------------------------------------------------------
# 13. Dataclass sanity.
# ---------------------------------------------------------------------------


def test_motional_state_config_is_frozen_and_equal_by_value() -> None:
    a = MotionalStateConfig(modes=(MotionalMode(frequency_hz=1e6, n_bar=0.0),))
    b = MotionalStateConfig(modes=(MotionalMode(frequency_hz=1e6, n_bar=0.0),))
    assert a == b
    with pytest.raises(Exception):  # noqa: B017 (frozen dataclass, any exception type is fine)
        a.v_rms_emm_m_s = 1.0  # type: ignore[misc]


def test_environment_config_default_has_motional_state_none() -> None:
    assert EnvironmentConfig().motional_state is None


# ---------------------------------------------------------------------------
# 14. WP31: per-mode participation factors for multi-ion crystals
#     (CONVENTIONS.md section 16's participation-factor extension).
# ---------------------------------------------------------------------------


def test_participation_default_is_bitwise_identical_to_pre_wp31_results() -> None:
    """`MotionalMode.participation` defaults to `1.0`; every WP30-era
    computation (the `_TWO_MODES` regression at the top of this file) is
    bitwise-unchanged by the WP31 formula generalization.
    """
    reference = float(_decimal_two_mode_reference())
    value = motional_pivot_perturbation(_TWO_MODES, _AL27_PLUS)
    np.testing.assert_allclose(value, reference, rtol=1e-12, atol=0)

    # Explicit participation=1.0 modes give the IDENTICAL float64 result
    # to modes built without the keyword at all (true bitwise identity,
    # not just tolerance-close).
    explicit_modes = tuple(
        MotionalMode(
            name=m.name,
            frequency_hz=m.frequency_hz,
            n_bar=m.n_bar,
            n_bar_uncertainty=m.n_bar_uncertainty,
            frequency_uncertainty_hz=m.frequency_uncertainty_hz,
            participation=1.0,
        )
        for m in _TWO_MODES
    )
    assert motional_pivot_perturbation(explicit_modes, _AL27_PLUS) == value
    assert motional_mean_squared_velocity_m2_s2(
        explicit_modes, _AL27_PLUS
    ) == motional_mean_squared_velocity_m2_s2(_TWO_MODES, _AL27_PLUS)
    assert motional_pivot_uncertainty(explicit_modes, _AL27_PLUS) == motional_pivot_uncertainty(
        _TWO_MODES, _AL27_PLUS
    )


def test_two_ion_participations_hand_derived_mu_25_over_27() -> None:
    """Hand-derived regression at `mu = m2/m1 = 25/27` (CONVENTIONS.md
    section 16, WP31; Wubbena et al. 2012 Eqs. 12-14, the closed-form
    two-ion axial in-phase/out-of-phase eigenvector solution):

        mu = 25/27
        root = sqrt(1 - mu + mu^2)
        b1_sq (in-phase / "COM") = (1 - mu + root) / (2*root)
        b2_sq (out-of-phase / "STR") = 1 - b1_sq

    Hand computation (double-checked against a plain calculator, not
    copied from the implementation):

        mu = 25/27 = 0.9259259259259259
        mu^2 = 0.8573388203017833
        1 - mu + mu^2 = 0.9314128943758574
        root = sqrt(0.9314128943758574) = 0.9650973496885469
        1 - mu + root = 1.0000000000... - 0.9259259259... + 0.9650973496...
                       = 1.0391714237...
        b1_sq = 1.0391714237... / (2*0.9650973496...) = 0.5383764778...
        b2_sq = 1 - 0.5383764778... = 0.4616235221...

    `two_ion_participations` applies the SAME (axial) closed form to all
    six standard-ordering slots (a documented approximation for the
    radial pairs, see that function's docstring), so this hand-derived
    pair should appear identically at indices (0, 1), (2, 3), and (4, 5).
    """
    m_clock = 27.0
    m_partner = 25.0
    mu = m_partner / m_clock
    root = math.sqrt(1.0 - mu + mu * mu)
    expected_b1_sq = (1.0 - mu + root) / (2.0 * root)
    expected_b2_sq = 1.0 - expected_b1_sq

    np.testing.assert_allclose(expected_b1_sq, 0.5383764778226668, rtol=0, atol=1e-15)
    np.testing.assert_allclose(expected_b2_sq, 0.46162352217733316, rtol=0, atol=1e-15)

    result = two_ion_participations(m_clock, m_partner)
    assert len(result) == 6
    for pair_start in (0, 2, 4):
        np.testing.assert_allclose(result[pair_start], expected_b1_sq, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result[pair_start + 1], expected_b2_sq, rtol=0, atol=1e-14)


def test_two_ion_participations_sum_rule() -> None:
    """The closed form's sum rule: `b1_sq + b2_sq = 1` exactly for every
    mode pair (the clock ion's participation summed across a mode PAIR
    equals 1 -- derived directly from the closed form's own
    `b2_sq = 1 - b1_sq` construction, and equivalently, per the mass-
    weighted-orthogonal-transformation derivation in
    `two_ion_participations`'s docstring, the same identity governing
    both ions' participation summed across a single mode). Checked at
    several representative mass ratios, not just mu=1.
    """
    for m_clock, m_partner in [(27.0, 25.0), (1.0, 1.0), (1.0, 10.0), (40.0, 9.0), (171.0, 1.0)]:
        result = two_ion_participations(m_clock, m_partner)
        for pair_start in (0, 2, 4):
            total = result[pair_start] + result[pair_start + 1]
            np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-13)


def test_two_ion_participations_kill_test_equal_mass_limit_is_one_half() -> None:
    """Kill test: at the equal-mass limit (`m_clock == m_partner`), each
    ion carries EXACTLY half of every mode -- `participation = 0.5`, not
    `1.0`. An implementation that returned `1.0` at equal mass (e.g. one
    that failed to account for the second ion's share at all) is caught
    directly by this assertion.
    """
    result = two_ion_participations(30.0, 30.0)
    for value in result:
        np.testing.assert_allclose(value, 0.5, rtol=0, atol=1e-14)
        assert abs(value - 1.0) > 0.49  # explicit: nowhere near the "no partner" value


def test_two_ion_participations_rejects_non_positive_masses() -> None:
    with pytest.raises(ValueError, match="m_clock"):
        two_ion_participations(0.0, 25.0)
    with pytest.raises(ValueError, match="m_clock"):
        two_ion_participations(-1.0, 25.0)
    with pytest.raises(ValueError, match="m_partner"):
        two_ion_participations(27.0, 0.0)
    with pytest.raises(ValueError, match="m_partner"):
        two_ion_participations(27.0, -5.0)


def test_participation_validated_in_range_zero_to_one() -> None:
    """`MotionalMode.participation` must satisfy `0 < participation <= 1`
    -- both engine-level (`_validate_motional_modes`, exercised via
    `motional_pivot_perturbation`) and pipeline-parse-level validation.
    """
    bad_mode_zero = MotionalMode(frequency_hz=1e6, n_bar=0.0, participation=0.0)
    with pytest.raises(ValueError, match="participation"):
        motional_pivot_perturbation((bad_mode_zero,), _AL27_PLUS)

    bad_mode_negative = MotionalMode(frequency_hz=1e6, n_bar=0.0, participation=-0.1)
    with pytest.raises(ValueError, match="participation"):
        motional_pivot_perturbation((bad_mode_negative,), _AL27_PLUS)

    bad_mode_too_big = MotionalMode(frequency_hz=1e6, n_bar=0.0, participation=1.5)
    with pytest.raises(ValueError, match="participation"):
        motional_pivot_perturbation((bad_mode_too_big,), _AL27_PLUS)

    # 1.0 itself (the boundary) is valid.
    good_mode = MotionalMode(frequency_hz=1e6, n_bar=0.0, participation=1.0)
    motional_pivot_perturbation((good_mode,), _AL27_PLUS)  # does not raise


def test_environment_motional_state_parses_participation() -> None:
    """`environment.motional_state.modes[].participation` (WP31) parses
    into `MotionalMode.participation`, default `1.0` when omitted.
    """
    config = _parse_motional_state(
        {
            "modes": [
                {"name": "axial", "frequency_Hz": 2.0e6, "n_bar": 0.05, "participation": 0.54},
                {"name": "radial", "frequency_Hz": 4.0e6, "n_bar": 0.05},
            ]
        }
    )
    assert config is not None
    assert config.modes[0].participation == 0.54
    assert config.modes[1].participation == 1.0


def test_environment_motional_state_rejects_bad_participation() -> None:
    for bad_value in (0.0, -0.2, 1.2):
        with pytest.raises(PipelineConfigError, match="participation"):
            _parse_motional_state(
                {"modes": [{"frequency_Hz": 2.0e6, "n_bar": 0.05, "participation": bad_value}]}
            )


def test_participation_composes_into_velocity_squared_expectation() -> None:
    """A halved participation factor exactly halves that mode's
    contribution to `<v^2>` (E38's per-mode term is linear in
    `participation`), a direct algebraic check independent of the
    two-ion closed form.
    """
    full = MotionalMode(frequency_hz=3.0e6, n_bar=0.1, participation=1.0)
    half = MotionalMode(frequency_hz=3.0e6, n_bar=0.1, participation=0.5)
    v2_full = motional_mean_squared_velocity_m2_s2((full,), _AL27_PLUS)
    v2_half = motional_mean_squared_velocity_m2_s2((half,), _AL27_PLUS)
    np.testing.assert_allclose(v2_half, 0.5 * v2_full, rtol=1e-14, atol=0)


# ---------------------------------------------------------------------------
# WP32: two-ion RADIAL participation factors reconstructed from the measured
# normal-mode spectrum (CONVENTIONS.md section 16, WP32; `axial_coulomb_curvature`/
# `two_ion_radial_participations`). Covers: axial-curvature hand computation,
# a numpy-eigendecomposition round trip (build a synthetic 2x2 problem,
# diagonalize it, invert the resulting frequencies, recover the inputs), the
# disambiguation kill test (equal mass, and the physical branch choice for
# unequal mass), the feasibility guard, input validation, uncertainty
# propagation against an independently-coded finite difference, and a
# benchmark regression pinning the Al27+/Mg25+ case's reconstructed totals.
# ---------------------------------------------------------------------------

_M_AL27 = _AL27_PLUS.mass_kg
_M_MG25 = 24.985837 * ATOMIC_MASS_UNIT


def test_axial_coulomb_curvature_hand_computed() -> None:
    """`axial_coulomb_curvature` at the Al27+/Mg25+ mass ratio and
    Marshall et al.'s own axial-COM mode frequency (2.16 MHz), hand
    computed directly from Wubbena Eq. 7/12 (double-checked against a
    plain calculator, not copied from the implementation):

        mu = m_Mg25/m_Al27, root = sqrt(1-mu+mu^2)
        omega_com = 2*pi*2.16e6
        omega_z1 = omega_com / sqrt((1+mu-root)/mu)
        c = m_Al27 * omega_z1^2 / 2
    """
    mu = _M_MG25 / _M_AL27
    root = math.sqrt(1.0 - mu + mu * mu)
    omega_com = 2.0 * math.pi * 2.16e6
    omega_z1 = omega_com / math.sqrt((1.0 + mu - root) / mu)
    expected_c = _M_AL27 * omega_z1 * omega_z1 / 2.0

    c, c_uncertainty = axial_coulomb_curvature(_M_AL27, _M_MG25, 2.16e6)
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-24)
    # A sanity range check on the physical magnitude (a few pN/m for a
    # few-micron ion spacing), catching a stray missing/extra factor that a
    # bitwise self-comparison against the same formula cannot.
    assert 1e-13 < c < 1e-10
    assert c_uncertainty == 0.0


def test_axial_coulomb_curvature_uncertainty_matches_analytic_partial() -> None:
    """`c` depends on `axial_com_frequency_hz` only through its square, so
    `sigma_c = c * 2 * sigma_f / f` exactly -- checked against an
    independently-coded finite difference (not the function's own
    analytic-partial line).
    """
    f = 2.16e6
    sigma_f = 500.0
    c_plus, _ = axial_coulomb_curvature(_M_AL27, _M_MG25, f + sigma_f)
    c_minus, _ = axial_coulomb_curvature(_M_AL27, _M_MG25, f - sigma_f)
    finite_difference_sigma_c = (c_plus - c_minus) / 2.0

    _, sigma_c = axial_coulomb_curvature(_M_AL27, _M_MG25, f, sigma_f)
    np.testing.assert_allclose(sigma_c, finite_difference_sigma_c, rtol=1e-6, atol=0)


def test_axial_coulomb_curvature_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="m_clock_kg"):
        axial_coulomb_curvature(0.0, _M_MG25, 2.16e6)
    with pytest.raises(ValueError, match="m_partner_kg"):
        axial_coulomb_curvature(_M_AL27, -1.0, 2.16e6)
    with pytest.raises(ValueError, match="axial_com_frequency_hz"):
        axial_coulomb_curvature(_M_AL27, _M_MG25, 0.0)
    with pytest.raises(ValueError, match="axial_com_frequency_uncertainty_hz"):
        axial_coulomb_curvature(_M_AL27, _M_MG25, 2.16e6, -1.0)


def test_two_ion_radial_participations_round_trip_recovers_bare_frequencies() -> None:
    """Round trip (WP32 deliverable 3): build a synthetic 2x2 radial
    eigenproblem from KNOWN bare frequencies/masses/coupling, diagonalize
    it with `numpy.linalg.eigh` (an implementation independent of
    `two_ion_radial_participations`' own branch-selection code), feed the
    two resulting mode frequencies back into
    `two_ion_radial_participations`, and check the recovered bare
    frequencies and participations match the synthetic inputs and numpy's
    own eigenvector components to near machine precision.
    """
    m_clock, m_partner = 4.4803898868635304e-26, 4.1489958508166885e-26  # Al27+, Mg25+
    c = 3.976554191127463e-12
    wr_clock_true = 2.0 * math.pi * 3.9e6
    wr_partner_true = 2.0 * math.pi * 4.8e6  # partner (lighter) has the higher bare frequency

    c_prime = c / math.sqrt(m_clock * m_partner)
    a = wr_clock_true**2 - c / m_clock
    b = wr_partner_true**2 - c / m_partner
    matrix = np.array([[a, c_prime], [c_prime, b]])
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)  # ascending order
    lambda_lo, lambda_hi = eigenvalues
    f_lo = math.sqrt(lambda_lo) / (2.0 * math.pi)
    f_hi = math.sqrt(lambda_hi) / (2.0 * math.pi)
    clock_participation_at_lo = eigenvectors[0, 0] ** 2
    clock_participation_at_hi = eigenvectors[0, 1] ** 2

    # Feed the lower-frequency mode in as "com", the higher as "str" (the
    # inversion does not care about the label, only the two eigenvalues).
    result = two_ion_radial_participations(m_clock, m_partner, c, f_lo, f_hi)

    np.testing.assert_allclose(
        result.bare_frequency_clock_hz, wr_clock_true / (2.0 * math.pi), rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        result.bare_frequency_partner_hz, wr_partner_true / (2.0 * math.pi), rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        result.com_participation, clock_participation_at_lo, rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        result.str_participation, clock_participation_at_hi, rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        result.com_participation + result.str_participation, 1.0, rtol=0, atol=1e-12
    )


def test_two_ion_radial_participations_kill_test_equal_mass_is_ambiguous() -> None:
    """Kill test (WP32 deliverable 1): at the equal-mass limit the
    RF-pseudopotential disambiguation rule (the lighter ion has the higher
    bare radial frequency) supplies no distinguishing direction, so the
    branch choice is genuinely undefined; this must raise, not silently
    pick one of the two equally-plausible branches.
    """
    with pytest.raises(ValueError, match="m_clock_kg == m_partner_kg"):
        two_ion_radial_participations(
            30.0 * ATOMIC_MASS_UNIT, 30.0 * ATOMIC_MASS_UNIT, 1e-12, 4.0e6, 3.0e6
        )


def test_two_ion_radial_participations_disambiguation_picks_physical_branch() -> None:
    """The disambiguation rule picks whichever branch gives the LIGHTER
    ion the higher bare radial frequency, regardless of which ion is
    labeled `clock` -- swapping which ion is lighter must swap which of
    `bare_frequency_clock_hz`/`bare_frequency_partner_hz` comes out larger.
    """
    heavier, lighter = 27.0 * ATOMIC_MASS_UNIT, 25.0 * ATOMIC_MASS_UNIT
    c = 3.976554191127463e-12

    clock_is_heavier = two_ion_radial_participations(heavier, lighter, c, 4.22e6, 3.48e6)
    assert clock_is_heavier.bare_frequency_partner_hz > clock_is_heavier.bare_frequency_clock_hz

    clock_is_lighter = two_ion_radial_participations(lighter, heavier, c, 4.22e6, 3.48e6)
    assert clock_is_lighter.bare_frequency_clock_hz > clock_is_lighter.bare_frequency_partner_hz


def test_two_ion_radial_participations_feasibility_guard_raises_for_infeasible_spectrum() -> None:
    """Feasibility guard (WP32 deliverable 2): measured radial mode
    frequencies too close together for the computed Coulomb coupling
    (`(lambda_hi-lambda_lo)^2 < 4*c'^2`) must raise, naming the numbers,
    never silently clamp or return a complex-valued result.
    """
    c = 3.976554191127463e-12  # the real Al27+/Mg25+ axial-derived coupling
    with pytest.raises(ValueError, match="infeasible radial mode pair"):
        two_ion_radial_participations(_M_AL27, _M_MG25, c, 4.22e6, 4.21e6)


def test_two_ion_radial_participations_rejects_invalid_input() -> None:
    c = 3.976554191127463e-12
    with pytest.raises(ValueError, match="m_clock_kg"):
        two_ion_radial_participations(0.0, _M_MG25, c, 4.22e6, 3.48e6)
    with pytest.raises(ValueError, match="m_partner_kg"):
        two_ion_radial_participations(_M_AL27, -1.0, c, 4.22e6, 3.48e6)
    with pytest.raises(ValueError, match="coulomb_curvature_n_per_m"):
        two_ion_radial_participations(_M_AL27, _M_MG25, 0.0, 4.22e6, 3.48e6)
    with pytest.raises(ValueError, match="radial_com_frequency_hz"):
        two_ion_radial_participations(_M_AL27, _M_MG25, c, 0.0, 3.48e6)
    with pytest.raises(ValueError, match="radial_str_frequency_hz"):
        two_ion_radial_participations(_M_AL27, _M_MG25, c, 4.22e6, 0.0)
    with pytest.raises(ValueError, match="radial_com_frequency_hz == radial_str_frequency_hz"):
        two_ion_radial_participations(_M_AL27, _M_MG25, c, 4.22e6, 4.22e6)
    with pytest.raises(ValueError, match="coulomb_curvature_uncertainty_n_per_m"):
        two_ion_radial_participations(
            _M_AL27, _M_MG25, c, 4.22e6, 3.48e6, coulomb_curvature_uncertainty_n_per_m=-1.0
        )
    with pytest.raises(ValueError, match="radial_com_frequency_uncertainty_hz"):
        two_ion_radial_participations(
            _M_AL27, _M_MG25, c, 4.22e6, 3.48e6, radial_com_frequency_uncertainty_hz=-1.0
        )
    with pytest.raises(ValueError, match="radial_str_frequency_uncertainty_hz"):
        two_ion_radial_participations(
            _M_AL27, _M_MG25, c, 4.22e6, 3.48e6, radial_str_frequency_uncertainty_hz=-1.0
        )


def test_two_ion_radial_participations_uncertainty_matches_finite_difference() -> None:
    """Uncertainty propagation sanity (WP32 deliverable 4): the reported
    `com_participation_uncertainty`/`str_participation_uncertainty` must
    match an INDEPENDENTLY-coded central finite difference over each
    uncertain input, combined in quadrature -- not merely reproduce the
    function's own internal arithmetic line by line.
    """
    c = 3.976554191127463e-12
    sigma_c = c * 1e-4
    f_com, sigma_f_com = 4.22e6, 2000.0
    f_str, sigma_f_str = 3.48e6, 1500.0

    def participations(c_val: float, f_com_val: float, f_str_val: float) -> tuple[float, float]:
        r = two_ion_radial_participations(_M_AL27, _M_MG25, c_val, f_com_val, f_str_val)
        return r.com_participation, r.str_participation

    p_com_0, p_str_0 = participations(c, f_com, f_str)

    d_com_dc = (
        participations(c + sigma_c, f_com, f_str)[0] - participations(c - sigma_c, f_com, f_str)[0]
    ) / 2.0
    d_com_dfcom = (
        participations(c, f_com + sigma_f_com, f_str)[0]
        - participations(c, f_com - sigma_f_com, f_str)[0]
    ) / 2.0
    d_com_dfstr = (
        participations(c, f_com, f_str + sigma_f_str)[0]
        - participations(c, f_com, f_str - sigma_f_str)[0]
    ) / 2.0
    expected_com_uncertainty = math.sqrt(d_com_dc**2 + d_com_dfcom**2 + d_com_dfstr**2)

    d_str_dc = (
        participations(c + sigma_c, f_com, f_str)[1] - participations(c - sigma_c, f_com, f_str)[1]
    ) / 2.0
    d_str_dfcom = (
        participations(c, f_com + sigma_f_com, f_str)[1]
        - participations(c, f_com - sigma_f_com, f_str)[1]
    ) / 2.0
    d_str_dfstr = (
        participations(c, f_com, f_str + sigma_f_str)[1]
        - participations(c, f_com, f_str - sigma_f_str)[1]
    ) / 2.0
    expected_str_uncertainty = math.sqrt(d_str_dc**2 + d_str_dfcom**2 + d_str_dfstr**2)

    result = two_ion_radial_participations(
        _M_AL27,
        _M_MG25,
        c,
        f_com,
        f_str,
        coulomb_curvature_uncertainty_n_per_m=sigma_c,
        radial_com_frequency_uncertainty_hz=sigma_f_com,
        radial_str_frequency_uncertainty_hz=sigma_f_str,
    )
    assert result.com_participation == p_com_0
    assert result.str_participation == p_str_0
    np.testing.assert_allclose(
        result.com_participation_uncertainty, expected_com_uncertainty, rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        result.str_participation_uncertainty, expected_str_uncertainty, rtol=1e-9, atol=0
    )


def test_two_ion_radial_participations_uncertainty_zero_when_every_input_exact() -> None:
    c = 3.976554191127463e-12
    result = two_ion_radial_participations(_M_AL27, _M_MG25, c, 4.22e6, 3.48e6)
    assert result.com_participation_uncertainty == 0.0
    assert result.str_participation_uncertainty == 0.0
    assert result.bare_frequency_clock_uncertainty_hz == 0.0
    assert result.bare_frequency_partner_uncertainty_hz == 0.0


def test_two_ion_radial_participations_raises_ambiguous_within_uncertainty() -> None:
    """A nominal spectrum whose branch choice is cleanly resolved at the
    point estimate can still have its uncertainty band reach into
    genuinely ambiguous territory: this is exactly the case the finite-
    difference uncertainty propagation samples directly (both +/-1-sigma
    points re-run the SAME feasibility/disambiguation checks the nominal
    point passed), so a large-enough supplied frequency uncertainty must
    raise -- not silently report a participation the data cannot actually
    support at that precision.

    Constructed synthetically (mirrors the round-trip test's forward-
    construction method): two bare radial frequencies 70 kHz apart give a
    cleanly resolved nominal reconstruction, but even a modest (2 kHz)
    frequency uncertainty is enough for the DOWNWARD sample to land back
    in the region where neither/both branches satisfy the disambiguation
    rule.
    """
    m_clock, m_partner = 27.0 * ATOMIC_MASS_UNIT, 25.0 * ATOMIC_MASS_UNIT
    c = 3.976554191127463e-12  # the real Al27+/Mg25+ axial-derived coupling
    c_prime = c / math.sqrt(m_clock * m_partner)
    wr_clock_true = 2.0 * math.pi * 4.0e6
    wr_partner_true = 2.0 * math.pi * 4.07e6  # 70 kHz above the clock ion's bare frequency
    a = wr_clock_true**2 - c / m_clock
    b = wr_partner_true**2 - c / m_partner
    matrix = np.array([[a, c_prime], [c_prime, b]])
    eigenvalues, _ = np.linalg.eigh(matrix)
    lambda_lo, lambda_hi = eigenvalues
    f_lo = math.sqrt(lambda_lo) / (2.0 * math.pi)
    f_hi = math.sqrt(lambda_hi) / (2.0 * math.pi)

    nominal = two_ion_radial_participations(m_clock, m_partner, c, f_lo, f_hi)
    np.testing.assert_allclose(
        abs(nominal.bare_frequency_clock_hz - nominal.bare_frequency_partner_hz),
        70000.0,
        rtol=1e-6,
        atol=0,
    )

    with pytest.raises(ValueError, match="ambiguous radial quadrant"):
        two_ion_radial_participations(
            m_clock,
            m_partner,
            c,
            f_lo,
            f_hi,
            radial_com_frequency_uncertainty_hz=2000.0,
            radial_str_frequency_uncertainty_hz=2000.0,
        )


# ---------------------------------------------------------------------------
# WP33: mode-specific intrinsic-micromotion enhancement for radial secular
# modes (CONVENTIONS.md section 16, WP33; `clock_ion_mathieu_parameters`/
# `radial_micromotion_enhancement`/`predicted_partner_bare_radial_
# frequencies_hz`). Covers: a synthetic forward/inverse round trip, the
# Marshall Al27+/Mg25+ hand-computed values (cross-checked against
# `omega.py`'s own WP33 comment block derivation), the enhancement
# formula's a=0/q=0 special cases, the mandatory over-determination
# (partner-prediction) consistency test pinned to this session's actual
# run, finite-difference uncertainty propagation, and every guard.
# ---------------------------------------------------------------------------

#: Marshall et al.'s published RF trap-drive frequency
#: (arXiv:2504.13071v2: "Ω/2π = 70.86 MHz").
_MARSHALL_RF_DRIVE_HZ = 70.86e6

#: WP32's own pinned reconstructed clock-ion bare radial frequencies for
#: the Al27+/Mg25+ crystal (`tests/test_motional_al_ion_radial_benchmark.py`'s
#: own provenance comment: generated by `python benchmarks/run_motional_al_ion.py`).
_MARSHALL_BARE_CLOCK_X_HZ = 3946771.607526584
_MARSHALL_BARE_CLOCK_Y_HZ = 5084690.883277048
_MARSHALL_BARE_PARTNER_X_HZ = 4360931.835761294
_MARSHALL_BARE_PARTNER_Y_HZ = 5497385.853262768


def test_clock_ion_mathieu_parameters_round_trip_recovers_synthetic_inputs() -> None:
    """Forward-construct a synthetic clock-ion Mathieu solution (`q`, `a_x`,
    `a_y`, `a_z` satisfying the Laplace constraint `a_x+a_y=-a_z`), derive
    the axial Coulomb curvature and the two bare radial frequencies that
    solution implies (WP33 comment block steps 1-3's own forward formulas),
    then invert with `clock_ion_mathieu_parameters` and check the recovered
    parameters match the synthetic originals to near machine precision --
    the closed-form solve is a genuine algebraic inverse of the forward
    Mathieu relations, not merely self-consistent by construction.
    """
    m_clock = 27.0 * ATOMIC_MASS_UNIT
    q_true = 0.22
    a_x_true = -0.0040
    a_z_true = 0.0031
    a_y_true = -a_z_true - a_x_true  # Laplace constraint

    rf_drive_hz = 65.0e6
    omega_rf = 2.0 * math.pi * rf_drive_hz
    omega_z_true = (omega_rf / 2.0) * math.sqrt(a_z_true)
    c_true = m_clock * omega_z_true * omega_z_true / 2.0
    omega_x_true = (omega_rf / 2.0) * math.sqrt(a_x_true + q_true * q_true / 2.0)
    omega_y_true = (omega_rf / 2.0) * math.sqrt(a_y_true + q_true * q_true / 2.0)
    f_x_true = omega_x_true / (2.0 * math.pi)
    f_y_true = omega_y_true / (2.0 * math.pi)

    result = clock_ion_mathieu_parameters(m_clock, c_true, rf_drive_hz, f_x_true, f_y_true)
    np.testing.assert_allclose(result.mathieu_q, q_true, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.mathieu_a_x, a_x_true, rtol=1e-9, atol=1e-16)
    np.testing.assert_allclose(result.mathieu_a_y, a_y_true, rtol=1e-9, atol=1e-16)
    np.testing.assert_allclose(result.mathieu_a_z, a_z_true, rtol=1e-9, atol=1e-16)
    # Laplace constraint holds exactly on the recovered values too.
    np.testing.assert_allclose(
        result.mathieu_a_x + result.mathieu_a_y, -result.mathieu_a_z, rtol=0, atol=1e-15
    )


def test_clock_ion_mathieu_parameters_hand_computed_marshall() -> None:
    """The clock (Al27+) ion's Mathieu parameters for Marshall et al.'s
    trap, computed by hand from WP32's own pinned bare radial frequencies
    (`tests/test_motional_al_ion_radial_benchmark.py`) and the published
    RF drive frequency. Cross-checked two ways: against a value pinned
    from `python benchmarks/run_motional_al_ion.py`'s WP33 case output
    this session, AND against an independently re-derived `a_z` via
    `axial_coulomb_curvature`'s own `c` (not reused from the pinned test
    module, computed fresh here)."""
    c, _ = axial_coulomb_curvature(_M_AL27, _M_MG25, 2.16e6)
    result = clock_ion_mathieu_parameters(
        _M_AL27, c, _MARSHALL_RF_DRIVE_HZ, _MARSHALL_BARE_CLOCK_X_HZ, _MARSHALL_BARE_CLOCK_Y_HZ
    )
    np.testing.assert_allclose(result.mathieu_q, 0.19127799732774156, rtol=1e-9, atol=0)
    np.testing.assert_allclose(result.mathieu_a_x, -0.005884496084657496, rtol=1e-9, atol=0)
    np.testing.assert_allclose(result.mathieu_a_y, 0.0023025499448954367, rtol=1e-9, atol=0)
    np.testing.assert_allclose(result.mathieu_a_z, 0.0035819461397620595, rtol=1e-9, atol=0)
    # Independent cross-check: a_z from axial_coulomb_curvature's own c,
    # recomputed here via omega_z1 = sqrt(2*c/m_clock) directly (not
    # reusing the function's internals), must agree with the returned a_z.
    omega_z1_independent = math.sqrt(2.0 * c / _M_AL27)
    omega_rf = 2.0 * math.pi * _MARSHALL_RF_DRIVE_HZ
    a_z_independent = 4.0 * omega_z1_independent**2 / omega_rf**2
    np.testing.assert_allclose(result.mathieu_a_z, a_z_independent, rtol=1e-12, atol=0)
    # Laplace constraint.
    np.testing.assert_allclose(
        result.mathieu_a_x + result.mathieu_a_y, -result.mathieu_a_z, rtol=0, atol=1e-15
    )


def test_clock_ion_mathieu_parameters_uncertainty_matches_finite_difference() -> None:
    """Uncertainty propagation sanity (mirrors
    `test_two_ion_radial_participations_uncertainty_matches_finite_difference`'s
    style): an INDEPENDENTLY-coded central finite difference over each
    uncertain input, combined in quadrature, must match the reported
    uncertainty -- not merely reproduce the function's own internal
    arithmetic line by line."""
    c, sigma_c = 3.976554191127463e-12, 3.976554191127463e-12 * 1e-4
    rf_hz, sigma_rf = _MARSHALL_RF_DRIVE_HZ, 1.0e3
    f_x, sigma_f_x = _MARSHALL_BARE_CLOCK_X_HZ, 2000.0
    f_y, sigma_f_y = _MARSHALL_BARE_CLOCK_Y_HZ, 1500.0

    def solve(
        c_val: float, rf_val: float, fx_val: float, fy_val: float
    ) -> ClockIonMathieuParameters:
        return clock_ion_mathieu_parameters(_M_AL27, c_val, rf_val, fx_val, fy_val)

    def partials(field: str) -> float:
        plus = getattr(solve(c + sigma_c, rf_hz, f_x, f_y), field)
        minus = getattr(solve(c - sigma_c, rf_hz, f_x, f_y), field)
        d_c = (plus - minus) / 2.0
        plus = getattr(solve(c, rf_hz + sigma_rf, f_x, f_y), field)
        minus = getattr(solve(c, rf_hz - sigma_rf, f_x, f_y), field)
        d_rf = (plus - minus) / 2.0
        plus = getattr(solve(c, rf_hz, f_x + sigma_f_x, f_y), field)
        minus = getattr(solve(c, rf_hz, f_x - sigma_f_x, f_y), field)
        d_fx = (plus - minus) / 2.0
        plus = getattr(solve(c, rf_hz, f_x, f_y + sigma_f_y), field)
        minus = getattr(solve(c, rf_hz, f_x, f_y - sigma_f_y), field)
        d_fy = (plus - minus) / 2.0
        return math.sqrt(d_c**2 + d_rf**2 + d_fx**2 + d_fy**2)

    expected_q_unc = partials("mathieu_q")
    expected_a_x_unc = partials("mathieu_a_x")
    expected_a_y_unc = partials("mathieu_a_y")
    expected_a_z_unc = partials("mathieu_a_z")

    result = clock_ion_mathieu_parameters(
        _M_AL27,
        c,
        rf_hz,
        f_x,
        f_y,
        coulomb_curvature_uncertainty_n_per_m=sigma_c,
        rf_drive_frequency_uncertainty_hz=sigma_rf,
        radial_bare_frequency_clock_x_uncertainty_hz=sigma_f_x,
        radial_bare_frequency_clock_y_uncertainty_hz=sigma_f_y,
    )
    np.testing.assert_allclose(result.mathieu_q_uncertainty, expected_q_unc, rtol=1e-9, atol=0)
    np.testing.assert_allclose(result.mathieu_a_x_uncertainty, expected_a_x_unc, rtol=1e-9, atol=0)
    np.testing.assert_allclose(result.mathieu_a_y_uncertainty, expected_a_y_unc, rtol=1e-9, atol=0)
    np.testing.assert_allclose(result.mathieu_a_z_uncertainty, expected_a_z_unc, rtol=1e-9, atol=0)


def test_clock_ion_mathieu_parameters_uncertainty_zero_when_every_input_exact() -> None:
    """Marshall's own inputs carry no per-mode frequency uncertainty
    (Table S2), so the real benchmark case's Mathieu-parameter uncertainty
    is exactly zero -- confirmed directly, not merely assumed."""
    c, _ = axial_coulomb_curvature(_M_AL27, _M_MG25, 2.16e6)
    result = clock_ion_mathieu_parameters(
        _M_AL27, c, _MARSHALL_RF_DRIVE_HZ, _MARSHALL_BARE_CLOCK_X_HZ, _MARSHALL_BARE_CLOCK_Y_HZ
    )
    assert result.mathieu_q_uncertainty == 0.0
    assert result.mathieu_a_x_uncertainty == 0.0
    assert result.mathieu_a_y_uncertainty == 0.0
    assert result.mathieu_a_z_uncertainty == 0.0


def test_clock_ion_mathieu_parameters_rejects_invalid_input() -> None:
    c = 3.976554191127463e-12
    with pytest.raises(ValueError, match="m_clock_kg"):
        clock_ion_mathieu_parameters(0.0, c, _MARSHALL_RF_DRIVE_HZ, 4.0e6, 5.0e6)
    with pytest.raises(ValueError, match="coulomb_curvature_n_per_m"):
        clock_ion_mathieu_parameters(_M_AL27, 0.0, _MARSHALL_RF_DRIVE_HZ, 4.0e6, 5.0e6)
    with pytest.raises(ValueError, match="rf_drive_frequency_hz"):
        clock_ion_mathieu_parameters(_M_AL27, c, 0.0, 4.0e6, 5.0e6)
    with pytest.raises(ValueError, match="radial_bare_frequency_clock_x_hz"):
        clock_ion_mathieu_parameters(_M_AL27, c, _MARSHALL_RF_DRIVE_HZ, 0.0, 5.0e6)
    with pytest.raises(ValueError, match="radial_bare_frequency_clock_y_hz"):
        clock_ion_mathieu_parameters(_M_AL27, c, _MARSHALL_RF_DRIVE_HZ, 4.0e6, 0.0)
    with pytest.raises(ValueError, match="coulomb_curvature_uncertainty_n_per_m"):
        clock_ion_mathieu_parameters(
            _M_AL27,
            c,
            _MARSHALL_RF_DRIVE_HZ,
            4.0e6,
            5.0e6,
            coulomb_curvature_uncertainty_n_per_m=-1.0,
        )
    with pytest.raises(ValueError, match="rf_drive_frequency_uncertainty_hz"):
        clock_ion_mathieu_parameters(
            _M_AL27, c, _MARSHALL_RF_DRIVE_HZ, 4.0e6, 5.0e6, rf_drive_frequency_uncertainty_hz=-1.0
        )
    with pytest.raises(ValueError, match="radial_bare_frequency_clock_x_uncertainty_hz"):
        clock_ion_mathieu_parameters(
            _M_AL27,
            c,
            _MARSHALL_RF_DRIVE_HZ,
            4.0e6,
            5.0e6,
            radial_bare_frequency_clock_x_uncertainty_hz=-1.0,
        )
    with pytest.raises(ValueError, match="radial_bare_frequency_clock_y_uncertainty_hz"):
        clock_ion_mathieu_parameters(
            _M_AL27,
            c,
            _MARSHALL_RF_DRIVE_HZ,
            4.0e6,
            5.0e6,
            radial_bare_frequency_clock_y_uncertainty_hz=-1.0,
        )


def test_radial_micromotion_enhancement_equals_two_at_a_zero() -> None:
    """`F_axis = 2` exactly when `a_axis = 0` (WP33 comment block step 5:
    equal secular and micromotion energy)."""
    for q in (0.05, 0.19, 0.5, 1.0):
        np.testing.assert_allclose(radial_micromotion_enhancement(q, 0.0), 2.0, rtol=0, atol=0)


def test_radial_micromotion_enhancement_matches_berkeland_formula() -> None:
    """Direct formula check against Berkeland Eq. 10's bracket,
    `1 + q^2/(2*a+q^2)`, computed independently here."""
    for q, a in ((0.191278, -0.005884), (0.191278, 0.002303), (0.3, 0.01), (0.1, -0.001)):
        expected = 1.0 + q * q / (2.0 * a + q * q)
        np.testing.assert_allclose(
            radial_micromotion_enhancement(q, a), expected, rtol=1e-14, atol=0
        )


def test_radial_micromotion_enhancement_axial_case_is_identity() -> None:
    """`q=0` (no RF component, the axial direction) gives `F_axis=1`
    identically for any nonzero `a_axis` -- no intrinsic micromotion
    without an RF Mathieu parameter to drive it."""
    for a in (0.001, 0.5, 3.0):
        np.testing.assert_allclose(radial_micromotion_enhancement(0.0, a), 1.0, rtol=0, atol=0)


def test_radial_micromotion_enhancement_rejects_negative_q() -> None:
    with pytest.raises(ValueError, match="mathieu_q"):
        radial_micromotion_enhancement(-0.1, 0.001)


def test_radial_micromotion_enhancement_rejects_unphysical_denominator() -> None:
    """`2*a_axis+q^2<=0` -- an unconfined radial mode -- must raise, not
    silently return a negative or divide-by-zero result."""
    with pytest.raises(ValueError, match="unphysical radial confinement"):
        radial_micromotion_enhancement(0.1, -1.0)
    with pytest.raises(ValueError, match="unphysical radial confinement"):
        radial_micromotion_enhancement(0.0, 0.0)


def test_predicted_partner_bare_radial_frequencies_matches_marshall_run() -> None:
    """MANDATORY OVER-DETERMINATION CHECK (WP33 comment block step 4): the
    clock ion's own solved Mathieu parameters, mass-scaled to the partner
    ion, predict bare radial frequencies pinned here to
    `python benchmarks/run_motional_al_ion.py`'s actual WP33 case output
    this session -- `4336114.00587478` Hz (X), `5468073.1484562885` Hz
    (Y) -- landing within `-0.57%`/`-0.53%` of WP32's own SEPARATELY
    reconstructed partner frequencies (`_MARSHALL_BARE_PARTNER_X_HZ`/
    `_MARSHALL_BARE_PARTNER_Y_HZ`), both sub-1%-relative."""
    c, _ = axial_coulomb_curvature(_M_AL27, _M_MG25, 2.16e6)
    mathieu = clock_ion_mathieu_parameters(
        _M_AL27, c, _MARSHALL_RF_DRIVE_HZ, _MARSHALL_BARE_CLOCK_X_HZ, _MARSHALL_BARE_CLOCK_Y_HZ
    )
    predicted_x_hz, predicted_y_hz = predicted_partner_bare_radial_frequencies_hz(
        mathieu, _M_AL27, _M_MG25, _MARSHALL_RF_DRIVE_HZ
    )
    np.testing.assert_allclose(predicted_x_hz, 4336114.00587478, rtol=1e-9, atol=0)
    np.testing.assert_allclose(predicted_y_hz, 5468073.1484562885, rtol=1e-9, atol=0)

    relative_deviation_x = (
        predicted_x_hz - _MARSHALL_BARE_PARTNER_X_HZ
    ) / _MARSHALL_BARE_PARTNER_X_HZ
    relative_deviation_y = (
        predicted_y_hz - _MARSHALL_BARE_PARTNER_Y_HZ
    ) / _MARSHALL_BARE_PARTNER_Y_HZ
    np.testing.assert_allclose(relative_deviation_x, -0.005690946527299086, rtol=1e-6, atol=0)
    np.testing.assert_allclose(relative_deviation_y, -0.005332117044155117, rtol=1e-6, atol=0)
    # The falsifiable claim itself: both branches agree at the sub-1%-relative
    # level, well inside the few-percent band the ~3-significant-figure
    # published mode frequencies support.
    assert abs(relative_deviation_x) < 0.01
    assert abs(relative_deviation_y) < 0.01


def test_predicted_partner_bare_radial_frequencies_round_trip_equal_mass() -> None:
    """At `m_partner == m_clock` (mass ratio 1), the "predicted partner"
    frequencies must reproduce the clock ion's OWN bare radial
    frequencies exactly -- the mass-scaling relation's trivial fixed
    point, a sanity check independent of any real dataset."""
    m_clock = 27.0 * ATOMIC_MASS_UNIT
    mathieu = ClockIonMathieuParameters(
        mathieu_q=0.2,
        mathieu_a_x=-0.003,
        mathieu_a_y=0.001,
        mathieu_a_z=0.002,
        mathieu_q_uncertainty=0.0,
        mathieu_a_x_uncertainty=0.0,
        mathieu_a_y_uncertainty=0.0,
        mathieu_a_z_uncertainty=0.0,
    )
    rf_drive_hz = 70.0e6
    omega_rf = 2.0 * math.pi * rf_drive_hz
    expected_fx = (omega_rf / 2.0) * math.sqrt(mathieu.mathieu_a_x + mathieu.mathieu_q**2 / 2.0)
    expected_fy = (omega_rf / 2.0) * math.sqrt(mathieu.mathieu_a_y + mathieu.mathieu_q**2 / 2.0)
    predicted_x_hz, predicted_y_hz = predicted_partner_bare_radial_frequencies_hz(
        mathieu, m_clock, m_clock, rf_drive_hz
    )
    np.testing.assert_allclose(predicted_x_hz, expected_fx / (2.0 * math.pi), rtol=1e-12, atol=0)
    np.testing.assert_allclose(predicted_y_hz, expected_fy / (2.0 * math.pi), rtol=1e-12, atol=0)


def test_predicted_partner_bare_radial_frequencies_rejects_invalid_input() -> None:
    mathieu = ClockIonMathieuParameters(
        mathieu_q=0.2,
        mathieu_a_x=-0.003,
        mathieu_a_y=0.001,
        mathieu_a_z=0.002,
        mathieu_q_uncertainty=0.0,
        mathieu_a_x_uncertainty=0.0,
        mathieu_a_y_uncertainty=0.0,
        mathieu_a_z_uncertainty=0.0,
    )
    with pytest.raises(ValueError, match="m_clock_kg"):
        predicted_partner_bare_radial_frequencies_hz(mathieu, 0.0, _M_MG25, 70.0e6)
    with pytest.raises(ValueError, match="m_partner_kg"):
        predicted_partner_bare_radial_frequencies_hz(mathieu, _M_AL27, 0.0, 70.0e6)
    with pytest.raises(ValueError, match="rf_drive_frequency_hz"):
        predicted_partner_bare_radial_frequencies_hz(mathieu, _M_AL27, _M_MG25, 0.0)


def test_predicted_partner_bare_radial_frequencies_raises_for_unconfined_prediction() -> None:
    """A sufficiently heavy hypothetical partner ion drives the mass-scaled
    `a_x_partner + q_partner^2/2` negative (the `q^2` term, quadratic in
    the mass ratio, shrinks faster than the linear `a_x` term as the
    partner mass grows), an unconfined predicted radial mode -- must
    raise, not silently return a complex/negative frequency. Uses the real
    Marshall Al27+ clock-ion solution (whose `a_x < 0`) with a 10x-heavier-
    than-Al27+ hypothetical partner, well past this configuration's own
    threshold (~84 u, computed from `a_x`/`q` directly)."""
    c, _ = axial_coulomb_curvature(_M_AL27, _M_MG25, 2.16e6)
    mathieu = clock_ion_mathieu_parameters(
        _M_AL27, c, _MARSHALL_RF_DRIVE_HZ, _MARSHALL_BARE_CLOCK_X_HZ, _MARSHALL_BARE_CLOCK_Y_HZ
    )
    m_partner_too_heavy = _M_AL27 * 10.0
    with pytest.raises(ValueError, match="unconfined"):
        predicted_partner_bare_radial_frequencies_hz(
            mathieu, _M_AL27, m_partner_too_heavy, _MARSHALL_RF_DRIVE_HZ
        )
