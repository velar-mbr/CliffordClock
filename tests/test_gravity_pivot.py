# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP22 Part 1 gravitational-redshift pivot term
(CONVENTIONS.md section 15, E36).

``cliffordclock.integrator.omega.grav_pivot_perturbation``/
``height_along_axis`` implement E36's formula; ``pivot_perturbation_stark``'s
new ``grav_pivot_perturbation`` keyword implements its additive
composition (E33's pattern), mirroring ``bbr_pivot_perturbation`` exactly.
This file covers the G9 sign-off's BINDING regressions: the computed-
magnitude regression (never a document literal), the sign regression (with
the physical "higher runs faster" statement), the rotor "scalar pivot
only" scope (identically zero boost contribution at v=0, not merely
small), the ~10 m extent warn, and composition additivity vs
Stark/BBR/quadrupole -- plus the pipeline-level config parsing/validation,
cross-mode agreement, byte-exactness of shipped examples, and report-note
content for ``environment.gravity``.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from cliffordclock.cl13 import IDX_E01, IDX_E02, IDX_E03, IDX_E12
from cliffordclock.constants import SPEED_OF_LIGHT, STANDARD_GRAVITY
from cliffordclock.ensemble.species import get_species
from cliffordclock.integrator.omega import (
    bbr_pivot_perturbation,
    build_omega_stark,
    grav_pivot_perturbation,
    height_along_axis,
    pivot_perturbation_stark,
    quadrupole_pivot_perturbation,
    spin_connection_stark,
    stark_pivot_terms,
)
from cliffordclock.pipeline import (
    GRAVITY_EXTENT_WARN_M,
    EnvironmentConfig,
    GravityConfig,
    PipelineConfig,
    PipelineConfigError,
    _gravity_extent_warn_note,
    _parse_environment,
    _parse_gravity,
    run_pipeline_full,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

# ---------------------------------------------------------------------------
# 1. Mandatory computed-magnitude regression (G9 sign-off A1, the "computed
#    never transcribed" gate catch -- the brief AND the dossier both
#    transcribed 1.0912e-16/m, one digit off from the correct 1.0911370e-16/m).
# ---------------------------------------------------------------------------


def test_grav_pivot_magnitude_matches_computed_g_over_c_squared() -> None:
    """The regression computes g/c^2 from cliffordclock.constants directly
    (never hard-codes 1.0911 or 1.0912) and asserts the implementation
    matches it exactly -- the actual content of the G9 gate's catch.
    """
    expected = STANDARD_GRAVITY / SPEED_OF_LIGHT**2
    value = float(grav_pivot_perturbation(jnp.asarray(1.0), STANDARD_GRAVITY))
    np.testing.assert_allclose(value, expected, rtol=1e-15, atol=0)


def test_grav_pivot_magnitude_is_1_0911370e_minus_16_per_metre_not_1_0912() -> None:
    """Descriptive pin (not authoritative -- the test above is): the
    computed value is ~1.0911370e-16/m at standard g, NOT the transcribed
    1.0912e-16/m that both an earlier theory brief and the Bothwell
    dossier's first draft carried (CONVENTIONS.md section 15).
    """
    value = float(grav_pivot_perturbation(jnp.asarray(1.0), STANDARD_GRAVITY))
    np.testing.assert_allclose(value, 1.0911370e-16, rtol=1e-7, atol=0)
    # The wrong (transcribed) value differs at the 5th significant figure --
    # confirm this regression would actually catch the transcription slip.
    wrong_transcribed_value = 1.0912e-16
    assert abs(value - wrong_transcribed_value) / value > 1e-5


def test_grav_pivot_magnitude_per_mm_matches_decimal_reference() -> None:
    """Independent 50-digit `decimal` cross-check of g/c^2 in per-mm units
    (the Bothwell benchmark's natural unit), computed in a code path that
    shares no arithmetic with `grav_pivot_perturbation` itself.
    """
    getcontext().prec = 50
    g = Decimal("9.80665")
    c = Decimal("299792458")
    h_mm = Decimal("1")
    expected_per_mm = float(g * (h_mm / Decimal(1000)) / c**2)
    value_per_mm = float(grav_pivot_perturbation(jnp.asarray(1e-3), STANDARD_GRAVITY))
    np.testing.assert_allclose(value_per_mm, expected_per_mm, rtol=1e-13, atol=0)
    np.testing.assert_allclose(value_per_mm, 1.0911370e-19, rtol=1e-7, atol=0)


# ---------------------------------------------------------------------------
# 2. Mandatory sign regression (G9 sign-off A1): (P-1)_grav(+1 m) > 0, with
#    the higher-clock-runs-faster physical statement.
# ---------------------------------------------------------------------------


def test_grav_pivot_sign_regression_higher_clock_runs_faster() -> None:
    """A clock HIGHER in the gravitational potential runs FASTER: under the
    E14b/E21 (P-1) = Delta_nu/nu_0 convention, (P-1)_grav(+1m) > 0 for a
    site above the reference height (CONVENTIONS.md section 15 E36, G9
    sign-off A1). A sign-flipped implementation would silently invert
    every clock's reported redshift.
    """
    value = float(grav_pivot_perturbation(jnp.asarray(1.0), STANDARD_GRAVITY))
    assert value > 0.0, (
        f"(P-1)_grav(+1m) = {value!r} is NOT positive -- a clock higher in the "
        "gravitational potential must run FASTER (G9 sign-off A1 sign regression)"
    )


def test_grav_pivot_sign_below_reference_is_negative() -> None:
    """A clock LOWER than the reference height runs SLOWER: (P-1)_grav < 0
    for h < h_ref (the mirror-image case of the mandatory sign regression).
    """
    value = float(grav_pivot_perturbation(jnp.asarray(-1.0), STANDARD_GRAVITY))
    assert value < 0.0


def test_grav_pivot_zero_at_reference_height() -> None:
    value = float(grav_pivot_perturbation(jnp.asarray(0.0), STANDARD_GRAVITY))
    assert value == 0.0


def test_grav_pivot_reference_height_shifts_the_zero_crossing() -> None:
    """`reference_height_m` translates where (P-1)_grav = 0: evaluating at
    `h = h_ref` is always zero, and `grav_pivot_perturbation(h, g, h_ref)`
    equals `grav_pivot_perturbation(h - h_ref, g, 0)` exactly (the formula
    only depends on `h - h_ref`).
    """
    h_ref = 2.5
    at_reference = float(grav_pivot_perturbation(jnp.asarray(h_ref), STANDARD_GRAVITY, h_ref))
    assert at_reference == 0.0

    h = 5.0
    shifted = float(grav_pivot_perturbation(jnp.asarray(h), STANDARD_GRAVITY, h_ref))
    unshifted = float(grav_pivot_perturbation(jnp.asarray(h - h_ref), STANDARD_GRAVITY, 0.0))
    np.testing.assert_allclose(shifted, unshifted, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 3. height_along_axis: geometry, batching, normalization, zero-vector guard.
# ---------------------------------------------------------------------------


def test_height_along_axis_projects_onto_unit_axis() -> None:
    positions = jnp.array([[0.0, 0.0, 3.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    heights = height_along_axis(positions, jnp.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(np.asarray(heights), [3.0, 3.0, 0.0], rtol=0, atol=0)


def test_height_along_axis_normalizes_non_unit_axis() -> None:
    """`up_axis` need not be pre-normalized -- a (0,0,2) axis gives the same
    heights as (0,0,1)."""
    positions = jnp.array([[0.0, 0.0, 5.0]])
    heights_unit = height_along_axis(positions, jnp.array([0.0, 0.0, 1.0]))
    heights_scaled = height_along_axis(positions, jnp.array([0.0, 0.0, 2.0]))
    np.testing.assert_allclose(np.asarray(heights_unit), np.asarray(heights_scaled), rtol=0, atol=0)


def test_height_along_axis_tilted_axis() -> None:
    """An arbitrary (non-coordinate-axis) "up" direction is supported."""
    up = jnp.array([1.0, 1.0, 0.0])  # unnormalized; norm = sqrt(2)
    position = jnp.array([1.0, 1.0, 0.0])
    height = float(height_along_axis(position, up))
    np.testing.assert_allclose(height, np.sqrt(2.0), rtol=1e-14, atol=0)


def test_height_along_axis_zero_vector_is_not_validated_here() -> None:
    """A zero `up_axis` is deliberately NOT checked inside `height_along_axis`
    itself (it must stay traceable under `jax.lax.scan`/`jax.vmap`, e.g. the
    worldline rotor accumulator, where a Python-level `if` on a traced norm
    raises `jax.errors.ConcretizationTypeError`) -- it silently divides to
    nan, and the zero-vector rejection instead lives at config-parse time
    (`cliffordclock.pipeline._parse_gravity`, tested separately below).
    """
    result = height_along_axis(jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]))
    assert bool(jnp.isnan(result))


def test_height_along_axis_batches_over_leading_axes() -> None:
    positions = jnp.zeros((4, 5, 3)).at[..., 2].set(jnp.arange(20).reshape(4, 5))
    heights = height_along_axis(positions, jnp.array([0.0, 0.0, 1.0]))
    assert heights.shape == (4, 5)
    np.testing.assert_allclose(np.asarray(heights), np.arange(20).reshape(4, 5), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 4. Composition additivity (E33's pattern) vs. Stark/BBR/quadrupole, at the
#    omega.py level (pipeline-level composition additivity is covered in
#    section 7 below).
# ---------------------------------------------------------------------------


def test_grav_pivot_composes_additively_with_stark() -> None:
    species = get_species("Sr87")
    e0 = jnp.array([1e5, 0.0, 0.0])
    zeros = jnp.zeros(3)
    stark_only = float(pivot_perturbation_stark(e0, zeros, species))
    grav_only = float(grav_pivot_perturbation(jnp.asarray(0.5), STANDARD_GRAVITY))
    composed = float(
        pivot_perturbation_stark(e0, zeros, species, grav_pivot_perturbation=grav_only)
    )
    np.testing.assert_allclose(composed, stark_only + grav_only, rtol=0, atol=1e-30)


def test_grav_pivot_composes_additively_with_stark_bbr_and_quadrupole() -> None:
    """All four terms (Stark, BBR, quadrupole, gravity) sum linearly in
    (P-1) with no cross term at this project's working precision -- the
    exact quantity `pivot_perturbation_stark` computes when every
    keyword-only term is supplied simultaneously.
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
    grav_only = float(grav_pivot_perturbation(jnp.asarray(0.5), STANDARD_GRAVITY))

    composed = float(
        pivot_perturbation_stark(
            e0,
            zeros,
            species,
            bbr_pivot_perturbation=bbr_only,
            quadrupole_pivot_perturbation=quad_only,
            grav_pivot_perturbation=grav_only,
        )
    )
    np.testing.assert_allclose(
        composed, stark_only + bbr_only + quad_only + grav_only, rtol=0, atol=1e-28
    )


# ---------------------------------------------------------------------------
# 5. Rotor scope (G9 sign-off: "the rotor carries it through the scalar
#    pivot only") -- the boost (omega_boost) contribution is IDENTICALLY
#    zero at v=0 regardless of the gravity term's value, not merely small;
#    the rotation-plane (B_hat_C) coefficient DOES carry the gravity term.
# ---------------------------------------------------------------------------


def test_gravity_reaches_only_the_rotation_coefficient_at_v_zero() -> None:
    species = get_species("Sr87")
    e_total = jnp.array([1e4, 0.0, 0.0])
    grad_e = jnp.zeros((3, 3))
    v = jnp.zeros(3)  # every lattice/lattice_extended node is static

    omega_no_grav = build_omega_stark(e_total, grad_e, species, v)
    omega_with_grav = build_omega_stark(e_total, grad_e, species, v, grav_pivot_perturbation=1e-10)

    for idx in (IDX_E01, IDX_E02, IDX_E03):
        assert float(omega_no_grav[idx]) == 0.0
        assert float(omega_with_grav[idx]) == 0.0, (
            "omega_boost must be identically zero at v=0 regardless of the "
            "gravity term (G9 sign-off: 'the rotor carries it through the "
            "scalar pivot only')"
        )

    # The B_hat_C rotation-plane coefficient DOES change by exactly the
    # gravity contribution (gamma_inv=1 exactly at v=0, so no kinematic
    # weighting dilutes it).
    delta = float(omega_with_grav[IDX_E12]) - float(omega_no_grav[IDX_E12])
    np.testing.assert_allclose(delta, 1e-10, rtol=1e-9, atol=0)


def test_gravity_shifts_spin_connection_denominator_only() -> None:
    """`spin_connection_stark`'s numerator (`d(P-1)/dr_k`, driven by the
    field gradient) is completely insensitive to `grav_pivot_perturbation`
    -- only the `P` denominator shifts. Verified by checking the returned
    boost-source array scales by exactly `p_without_grav / p_with_grav`
    between the two calls (CONVENTIONS.md section 15's rotor-scope note).
    """
    species = get_species("Sr87")
    # Every row/component below is nonzero (and non-degenerate against
    # e_total) so every d(P-1)/dr_k numerator component is itself nonzero
    # -- otherwise a 0/0 component would make the ratio check vacuous.
    e_total = jnp.array([1e4, 2e4, 3e4])
    grad_e = jnp.array([[1e2, 2e2, 3e2], [4e2, 5e2, 6e2], [7e2, 8e2, 9e2]])

    omega_0k_no_grav = spin_connection_stark(e_total, grad_e, species)
    grav_value = 1e-8
    omega_0k_with_grav = spin_connection_stark(
        e_total, grad_e, species, grav_pivot_perturbation=grav_value
    )

    baseline, cross, quadratic = stark_pivot_terms(e_total, jnp.zeros_like(e_total), species)
    p_no_grav = 1.0 + float(baseline + cross + quadratic)
    p_with_grav = p_no_grav + grav_value

    expected_ratio = p_no_grav / p_with_grav
    actual_ratio = np.asarray(omega_0k_with_grav) / np.asarray(omega_0k_no_grav)
    np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# 6. Pipeline-level config parsing / validation.
# ---------------------------------------------------------------------------


def test_environment_gravity_absent_defaults_to_none() -> None:
    assert _parse_environment(None).gravity is None
    assert _parse_environment({}).gravity is None
    assert _parse_gravity(None) is None


def test_environment_gravity_defaults() -> None:
    cfg = _parse_gravity({})
    assert cfg is not None
    assert cfg.g_m_s2 == STANDARD_GRAVITY
    assert cfg.up_axis == (0.0, 0.0, 1.0)
    assert cfg.reference_height_m == 0.0


def test_environment_gravity_parses_explicit_fields() -> None:
    cfg = _parse_gravity({"g_m_s2": 9.796, "up_axis": [0.0, 0.0, -1.0], "reference_height_m": 0.1})
    assert cfg == GravityConfig(g_m_s2=9.796, up_axis=(0.0, 0.0, -1.0), reference_height_m=0.1)


def test_environment_gravity_rejects_non_positive_g() -> None:
    with pytest.raises(PipelineConfigError, match="g_m_s2 must be > 0"):
        _parse_gravity({"g_m_s2": 0.0})
    with pytest.raises(PipelineConfigError, match="g_m_s2 must be > 0"):
        _parse_gravity({"g_m_s2": -9.8})


def test_environment_gravity_rejects_zero_up_axis() -> None:
    with pytest.raises(PipelineConfigError, match="up_axis must not be the zero vector"):
        _parse_gravity({"up_axis": [0.0, 0.0, 0.0]})


def _base_lattice_stark_dict(tmp_path: Path) -> dict[str, object]:
    return {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 0.0]}}},
        "coupling": {"type": "stark_dc"},
        "ensemble": {
            "regime": "lattice",
            "temperature_uK": 1.0,
            "motional_n": [0, 0, 0],
            "n_quad": 1,
        },
        "integration": {"time_s": 1.0},
        "output": {"directory": str(tmp_path)},
    }


def test_environment_gravity_requires_stark_dc_coupling(tmp_path: Path) -> None:
    data = _base_lattice_stark_dict(tmp_path)
    data["coupling"] = {"mu": [1.0e-30, 0.0, 0.0]}
    data["environment"] = {"gravity": {"g_m_s2": 9.80665}}
    with pytest.raises(PipelineConfigError, match="environment.gravity requires"):
        PipelineConfig.from_dict(data)


def test_environment_gravity_with_stark_dc_coupling_accepted(tmp_path: Path) -> None:
    data = _base_lattice_stark_dict(tmp_path)
    data["environment"] = {"gravity": {"g_m_s2": 9.80665}}
    config = PipelineConfig.from_dict(data)
    assert config.environment.gravity is not None
    assert config.environment.gravity.g_m_s2 == 9.80665


# ---------------------------------------------------------------------------
# 7. Byte-exactness of shipped examples, cross-mode agreement, composition
#    additivity at the report level, report-note content -- the pipeline
#    half of the WP22 Part 1 test contract.
# ---------------------------------------------------------------------------


def test_no_shipped_example_uses_gravity_key() -> None:
    example_paths = sorted(_EXAMPLES_DIR.glob("*.yaml"))
    assert len(example_paths) >= 5, "expected several shipped example configs"
    for path in example_paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        environment = data.get("environment")
        if environment:
            assert "gravity" not in environment, f"{path.name} unexpectedly sets 'gravity:'"


@pytest.mark.parametrize(
    ("example_name", "expected_shift"),
    [
        # Reproduced from tests/test_bbr_pipeline.py's own pinned snapshot
        # (unaffected by WP22: gravity defaults to off, so every shipped
        # example's output is byte-identical to its pre-WP22 value too).
        ("lattice_sr87.yaml", 1.2180934201244512e-10),
        ("lattice_sr87_stark.yaml", -7.170523825042889e-17),
        ("quadrupole_classical.yaml", 4.699865771114046e-14),
        ("realistic_lattice_sr87.yaml", -7.723398671928127e-19),
        ("showcase_gradient_dispersion_sr87.yaml", -9.965266591533704e-17),
        ("comsol_electrode_sr87.yaml", -1.1092455754730931e-14),
    ],
)
def test_shipped_example_output_byte_identical_with_wp22_present(
    example_name: str, expected_shift: float, tmp_path: Path
) -> None:
    data = yaml.safe_load((_EXAMPLES_DIR / example_name).read_text(encoding="utf-8"))
    data["output"] = dict(data.get("output") or {})
    data["output"]["directory"] = str(tmp_path)
    config = PipelineConfig.from_dict(data)
    assert config.environment == EnvironmentConfig()  # gravity off, as constructed

    result = run_pipeline_full(config)

    np.testing.assert_allclose(result.report.mean_fractional_shift, expected_shift, rtol=0, atol=0)


@pytest.mark.slow
def test_gravity_lattice_fast_path_matches_worldline_rotor_crosscheck(tmp_path: Path) -> None:
    """E29's exact-agreement claim, extended with gravity active: static
    v=0 quadrature nodes mean the rotor's omega_boost is identically zero
    (section 5 above), so mode="worldline" must reproduce mode="fast_path"
    exactly, mirroring test_bbr_pipeline.py's analogous BBR test.
    """
    base = _base_lattice_stark_dict(tmp_path / "fp")
    base["ensemble"] = dict(base["ensemble"], motional_n=[0, 0, 0], n_quad=1)
    base["trap"] = dict(base["trap"], center=[0.0, 0.0, 1.5])
    base["environment"] = {"gravity": {"g_m_s2": 9.80665}}

    fast_path_data = dict(base)
    fast_path_data["integration"] = {"mode": "fast_path", "time_s": 1.0}
    fast_path_config = PipelineConfig.from_dict(fast_path_data)
    fast_path_result = run_pipeline_full(fast_path_config)

    worldline_data = dict(base)
    worldline_data["output"] = {"directory": str(tmp_path / "wl")}
    worldline_data["integration"] = {"mode": "worldline", "time_s": 1.0}
    worldline_config = PipelineConfig.from_dict(worldline_data)
    worldline_result = run_pipeline_full(worldline_config)

    np.testing.assert_allclose(
        worldline_result.report.mean_fractional_shift,
        fast_path_result.report.mean_fractional_shift,
        rtol=1e-9,
        atol=0,
    )
    # Non-vacuous: gravity is actually contributing (not accidentally zero
    # -- the trap sits 1.5m above the reference height).
    assert abs(fast_path_result.report.mean_fractional_shift) > 1e-17


def test_gravity_composition_additivity_at_report_level(tmp_path: Path) -> None:
    """The single lattice quadrature node here has v=0 exactly (fast_path,
    E29) and sits at a known height, so the reported shift delta from
    turning gravity on must equal `grav_pivot_perturbation` at that height
    to within a few floating-point rounding steps.
    """
    height_m = 2.0
    without_data = _base_lattice_stark_dict(tmp_path / "without")
    without_data["trap"] = dict(without_data["trap"], center=[0.0, 0.0, height_m])
    without_config = PipelineConfig.from_dict(without_data)
    without_result = run_pipeline_full(without_config)

    with_data = _base_lattice_stark_dict(tmp_path / "with")
    with_data["trap"] = dict(with_data["trap"], center=[0.0, 0.0, height_m])
    with_data["environment"] = {"gravity": {"g_m_s2": 9.80665}}
    with_config = PipelineConfig.from_dict(with_data)
    with_result = run_pipeline_full(with_config)

    delta = with_result.report.mean_fractional_shift - without_result.report.mean_fractional_shift
    expected = float(grav_pivot_perturbation(jnp.asarray(height_m), 9.80665))
    np.testing.assert_allclose(delta, expected, rtol=0, atol=1e-28)


def test_gravity_report_note_content(tmp_path: Path) -> None:
    data = _base_lattice_stark_dict(tmp_path)
    data["environment"] = {"gravity": {"g_m_s2": 9.796, "reference_height_m": 0.1}}
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    notes = result.report.uncertainty_notes

    assert "environment.gravity (CONVENTIONS.md section 15, E36)" in notes
    assert "g_m_s2=9.796" in notes
    assert "HIGHER clock" in notes
    assert "FASTER" in notes
    assert "SURVEYED LOCAL g" in notes


def test_gravity_report_note_uses_default_g_when_omitted(tmp_path: Path) -> None:
    data = _base_lattice_stark_dict(tmp_path)
    data["environment"] = {"gravity": {}}
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    assert f"g_m_s2={STANDARD_GRAVITY!r}" in result.report.uncertainty_notes


# ---------------------------------------------------------------------------
# 8. The ~10 m extent warning (G9 sign-off A3, gate edit 3): runtime guard.
# ---------------------------------------------------------------------------


def test_gravity_extent_warning_absent_at_lab_scale(tmp_path: Path) -> None:
    """A millimetre-to-metre-scale sample (every realistic config this
    project ships) must NOT trigger the extent warning.
    """
    data = _base_lattice_stark_dict(tmp_path)
    data["trap"] = dict(data["trap"], center=[0.0, 0.0, 0.5])
    data["environment"] = {"gravity": {"g_m_s2": 9.80665}}
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    assert "height-extent WARNING" not in result.report.uncertainty_notes


def test_gravity_extent_warning_fires_beyond_ten_metres(tmp_path: Path) -> None:
    """The warning is about the SAMPLE's own sampled height EXTENT (the
    spread across positions actually evaluated), not the absolute height of
    a single site relative to the origin -- a single-site `lattice` config
    has zero internal extent regardless of how high its one node sits, so
    this test uses `lattice_extended` (WP22 Part 2) with widely spaced
    sites to genuinely exceed GRAVITY_EXTENT_WARN_M, a real runtime guard,
    not just a constant that is never reached.
    """
    span_m = GRAVITY_EXTENT_WARN_M + 2.0
    data: dict[str, object] = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 0.0]}}},
        "coupling": {"type": "stark_dc"},
        "ensemble": {
            "regime": "lattice_extended",
            "temperature_uK": 1.0,
            "motional_n": [0, 0, 0],
            "n_quad": 1,
            "n_sites": 3,
            "site_spacing_m": span_m / 2.0,
            "site_axis": [0.0, 0.0, 1.0],
            "site_envelope": "uniform",
        },
        "integration": {"mode": "fast_path", "time_s": 1.0},
        "environment": {"gravity": {"g_m_s2": 9.80665}},
        "output": {"directory": str(tmp_path)},
    }
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    assert "height-extent WARNING" in result.report.uncertainty_notes
    assert f"{GRAVITY_EXTENT_WARN_M:.0f} m" in result.report.uncertainty_notes


def test_gravity_extent_warning_boundary_is_inclusive_of_the_margin() -> None:
    """A span exactly at the threshold does not warn (`<=`, matching
    `_gravity_extent_warn_note`'s own comparison)."""
    gravity = GravityConfig()
    positions = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, GRAVITY_EXTENT_WARN_M]])
    assert _gravity_extent_warn_note(gravity, positions) is None

    positions_over = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, GRAVITY_EXTENT_WARN_M + 1e-9]])
    assert _gravity_extent_warn_note(gravity, positions_over) is not None
