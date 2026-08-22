# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pipeline-level tests for the WP20 BBR shift (CONVENTIONS.md E32/E33).

Covers the WP20 test contract items that need the full pipeline (config
parsing/validation, cross-mode agreement, composition additivity at the
report level, report-note content, and byte-exactness of the shipped
examples) -- the pure-formula checks (sign regression, closed-form known
answers, uncertainty arithmetic) live in `tests/test_bbr_pivot.py`, and the
registry data pins live in `tests/test_bbr_species.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from cliffordclock.ensemble.species import get_species
from cliffordclock.integrator.omega import bbr_pivot_perturbation
from cliffordclock.pipeline import (
    EnvironmentConfig,
    PipelineConfig,
    PipelineConfigError,
    _parse_environment,
    run_pipeline_full,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

# ---------------------------------------------------------------------------
# 1. Config parsing / validation.
# ---------------------------------------------------------------------------


def test_environment_absent_defaults_to_bbr_off() -> None:
    assert _parse_environment(None) == EnvironmentConfig()


def test_environment_empty_mapping_defaults_to_bbr_off() -> None:
    """An explicit but empty ``environment: {}`` behaves exactly like the
    key's absence (no `radiation_temperature_K` key inside it either).
    """
    assert _parse_environment({}) == EnvironmentConfig()


def test_environment_parses_temperature_and_uncertainty() -> None:
    cfg = _parse_environment(
        {"radiation_temperature_K": 300.0, "radiation_temperature_uncertainty_K": 0.004}
    )
    assert cfg.radiation_temperature_k == 300.0
    assert cfg.radiation_temperature_uncertainty_k == 0.004


def test_environment_uncertainty_without_temperature_rejected() -> None:
    with pytest.raises(PipelineConfigError, match="requires environment.radiation_temperature_K"):
        _parse_environment({"radiation_temperature_uncertainty_K": 0.01})


def test_environment_negative_uncertainty_rejected() -> None:
    with pytest.raises(PipelineConfigError, match="must be >= 0"):
        _parse_environment(
            {"radiation_temperature_K": 300.0, "radiation_temperature_uncertainty_K": -1.0}
        )


@pytest.mark.parametrize("temperature_k", [49.999, 0.0, -10.0, 350.001, 1000.0])
def test_environment_temperature_outside_window_rejected(temperature_k: float) -> None:
    """G7 sign-off gate edit 5: hard PipelineConfigError outside 50-350K
    (both edges, plus degenerate/negative values).
    """
    with pytest.raises(PipelineConfigError, match="validated BBR fit range"):
        _parse_environment({"radiation_temperature_K": temperature_k})


@pytest.mark.parametrize("temperature_k", [50.0, 300.0, 350.0])
def test_environment_temperature_at_window_edges_accepted(temperature_k: float) -> None:
    """The window is inclusive: exactly 50K and exactly 350K are valid."""
    cfg = _parse_environment({"radiation_temperature_K": temperature_k})
    assert cfg.radiation_temperature_k == temperature_k


def _base_lattice_stark_dict(tmp_path: Path) -> dict[str, object]:
    """A minimal, fast lattice `coupling.type='stark_dc'` config dict
    (mirrors `examples/lattice_sr87_stark.yaml`'s structure).
    """
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
        "output": {"directory": str(tmp_path)},
    }


def test_environment_temperature_requires_stark_dc_coupling(tmp_path: Path) -> None:
    """G7 sign-off / WP20 design: BBR needs the species' registry
    `BbrCoefficients`, resolved independently of `coupling.type` -- but
    `coupling.type='linear_mu'` has no equivalent quantity, so setting
    `environment.radiation_temperature_K` together with it is rejected at
    config-parse time (`PipelineConfig.from_dict`'s cross-field check),
    not silently ignored.
    """
    data = _base_lattice_stark_dict(tmp_path)
    data["coupling"] = {"mu": [1.0e-30, 0.0, 0.0]}
    data["environment"] = {"radiation_temperature_K": 300.0}
    with pytest.raises(PipelineConfigError, match="requires coupling.type='stark_dc'"):
        PipelineConfig.from_dict(data)


def test_environment_temperature_with_stark_dc_coupling_accepted(tmp_path: Path) -> None:
    data = _base_lattice_stark_dict(tmp_path)
    data["environment"] = {"radiation_temperature_K": 300.0}
    config = PipelineConfig.from_dict(data)
    assert config.environment.radiation_temperature_k == 300.0


def test_environment_temperature_species_without_bbr_data_raises_config_error(
    tmp_path: Path,
) -> None:
    """`Al27+` has a `stark_coefficient_hz_per_v2_m2` override path (so
    `coupling.type='stark_dc'` alone is satisfiable) but no `BbrCoefficients`
    registry entry -- `run_pipeline_full` must surface that as a
    `PipelineConfigError` (via `_resolve_bbr_pivot_perturbation`'s
    `ValueError` -> `PipelineConfigError` wrap), not an unhandled `ValueError`.
    """
    data = _base_lattice_stark_dict(tmp_path)
    data["species"] = "Al27+"
    data["coupling"] = {"type": "stark_dc", "stark_coefficient_hz_per_v2_m2": 1.0e-6}
    data["environment"] = {"radiation_temperature_K": 300.0}
    config = PipelineConfig.from_dict(data)
    with pytest.raises(PipelineConfigError, match="no BBR shift data"):
        run_pipeline_full(config)


# ---------------------------------------------------------------------------
# 2. Byte-exactness of shipped examples (WP20 test contract).
# ---------------------------------------------------------------------------


def test_no_shipped_example_uses_environment_key() -> None:
    """No `examples/*.yaml` config opts into BBR -- the WP20 acceptance
    criterion is that every shipped example's output is byte-identical to
    its pre-WP20 value, which this structural check protects independent
    of any numeric regression below.
    """
    example_paths = sorted(_EXAMPLES_DIR.glob("*.yaml"))
    assert len(example_paths) >= 5, "expected several shipped example configs"
    for path in example_paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "environment" not in data, f"{path.name} unexpectedly sets 'environment:'"


@pytest.mark.parametrize(
    ("example_name", "expected_shift"),
    [
        # Pinned 2026-08-11, captured bit-for-bit identical both immediately
        # before and immediately after the WP20 BBR changes landed (a
        # deliberate before/after snapshot diff, not just a single
        # post-change capture) -- see the WP20 builder report.
        ("lattice_sr87.yaml", 1.2180934201244512e-10),  # coupling.type=linear_mu
        ("lattice_sr87_stark.yaml", -7.170523825042889e-17),  # coupling.type=stark_dc
        ("quadrupole_classical.yaml", 4.699865771114046e-14),
        ("realistic_lattice_sr87.yaml", -7.723398671928127e-19),
        ("showcase_gradient_dispersion_sr87.yaml", -9.965266591533704e-17),
        ("comsol_electrode_sr87.yaml", -1.1092455754730931e-14),
    ],
)
def test_shipped_example_output_byte_identical_to_pre_wp20_snapshot(
    example_name: str, expected_shift: float, tmp_path: Path
) -> None:
    data = yaml.safe_load((_EXAMPLES_DIR / example_name).read_text(encoding="utf-8"))
    data["output"] = dict(data.get("output") or {})
    data["output"]["directory"] = str(tmp_path)
    config = PipelineConfig.from_dict(data)
    assert config.environment == EnvironmentConfig()  # BBR off, as constructed

    result = run_pipeline_full(config)

    # rtol=0 (true bit-for-bit) held in the pinning dev venv's jax release
    # but not in a freshly resolved venv on a newer jax/XLA (observed: jax
    # 0.11.1, ~1.24e-16 relative on showcase_gradient_dispersion_sr87.yaml,
    # ~half a float64 ULP) -- XLA reduction/fusion scheduling for this
    # unchanged BBR-off arithmetic differs by jax version, so exact equality
    # is not a portable contract. rtol=1e-14 is ~2 orders of magnitude
    # looser than that observed ULP-level drift so it still comfortably
    # absorbs cross-jax-version rounding, while staying ~4+ orders of
    # magnitude tighter than any physically meaningful shift here, so it
    # still catches a real numeric regression in the BBR-off code path.
    np.testing.assert_allclose(
        result.report.mean_fractional_shift, expected_shift, rtol=1e-14, atol=0
    )


# ---------------------------------------------------------------------------
# 3. Cross-mode agreement with BBR active (WP20 test contract: fast_path =
#    direct = streaming = rotor with BBR on).
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bbr_lattice_fast_path_matches_worldline_rotor_crosscheck(tmp_path: Path) -> None:
    """E29's exact-agreement claim, extended with BBR active (WP20: "extend
    the WP16 head-to-head test with BBR active rather than duplicating
    it" -- this is the pipeline-level half; `tests/test_integrator_stark_rotor.py`
    has the direct `build_omega_stark`-level extension). Static v=0
    quadrature nodes mean the rotor's `omega_boost` is identically zero, so
    `mode="worldline"` must reproduce `mode="fast_path"` exactly (same
    tolerance style as
    `test_wp8_lattice_fast_path_is_default_and_worldline_is_explicit_crosscheck`).
    """
    base = _base_lattice_stark_dict(tmp_path / "fp")
    base["environment"] = {"radiation_temperature_K": 300.0}

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
    # Non-vacuous: BBR is actually contributing (not accidentally zero).
    assert abs(fast_path_result.report.mean_fractional_shift) > 1e-16


def _tiny_classical_stark_dict(output_dir: Path) -> dict[str, object]:
    return {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
        "coupling": {"type": "stark_dc"},
        "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 20, "seed": 1},
        "integration": {"dtau": 0.5, "steps": 300},
        "environment": {"radiation_temperature_K": 300.0},
        "output": {"directory": str(output_dir)},
    }


def test_bbr_classical_direct_batched_matches_streaming(tmp_path: Path) -> None:
    """WP19's batched<->streaming agreement, extended with BBR active."""
    batched_data = _tiny_classical_stark_dict(tmp_path / "batched")
    batched_data["integration"] = dict(batched_data["integration"], evaluation="batched")
    batched_config = PipelineConfig.from_dict(batched_data)
    batched_result = run_pipeline_full(batched_config)

    streaming_data = _tiny_classical_stark_dict(tmp_path / "streaming")
    streaming_data["integration"] = dict(streaming_data["integration"], evaluation="streaming")
    streaming_config = PipelineConfig.from_dict(streaming_data)
    streaming_result = run_pipeline_full(streaming_config)

    np.testing.assert_allclose(
        streaming_result.report.mean_fractional_shift,
        batched_result.report.mean_fractional_shift,
        rtol=1e-9,
        atol=0,
    )
    assert abs(batched_result.report.mean_fractional_shift) > 1e-16


# ---------------------------------------------------------------------------
# 4. Composition additivity at the report level (E33): Stark+BBR minus
#    Stark-only equals the standalone BBR-only pivot value.
# ---------------------------------------------------------------------------


def test_bbr_composition_additivity_at_report_level(tmp_path: Path) -> None:
    """The single lattice quadrature node here has v=0 exactly (fast_path,
    E29), so `gamma_inv=1` exactly and the reported shift delta from
    turning BBR on must equal `bbr_pivot_perturbation(300, Sr87)` to within
    a couple of floating-point rounding steps of the ~1e-15-magnitude
    quantities involved (not merely "close" -- the atol below is ~50
    ulps of the shift magnitude, not a loose physics tolerance).
    """
    without_bbr_data = _base_lattice_stark_dict(tmp_path / "without")
    without_bbr_config = PipelineConfig.from_dict(without_bbr_data)
    without_bbr_result = run_pipeline_full(without_bbr_config)

    with_bbr_data = _base_lattice_stark_dict(tmp_path / "with")
    with_bbr_data["environment"] = {"radiation_temperature_K": 300.0}
    with_bbr_config = PipelineConfig.from_dict(with_bbr_data)
    with_bbr_result = run_pipeline_full(with_bbr_config)

    delta = (
        with_bbr_result.report.mean_fractional_shift
        - without_bbr_result.report.mean_fractional_shift
    )
    expected_bbr_value = bbr_pivot_perturbation(300.0, get_species("Sr87"))
    np.testing.assert_allclose(delta, expected_bbr_value, rtol=0, atol=1e-28)


# ---------------------------------------------------------------------------
# 5. Report note content (T/coefficients/citations, uncertainty labeling,
#    M1/E2 budget line, B4 300-350K note).
# ---------------------------------------------------------------------------


def test_bbr_report_note_content_at_300k(tmp_path: Path) -> None:
    data = _base_lattice_stark_dict(tmp_path)
    data["environment"] = {"radiation_temperature_K": 300.0}
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    notes = result.report.uncertainty_notes

    assert "BBR (CONVENTIONS.md E32)" in notes
    assert "T=300.0" in notes
    assert "arithmetic-reproduction fidelity" in notes
    assert "M1/E2 multipole BBR contributions: modeled out, magnitude ~6e-20 each" in notes
    assert "conditional on exact T" in notes  # no radiation_temperature_uncertainty_K given
    # T=300K is AT the cross-verified edge, not beyond it -- no B4 caveat.
    assert "beyond the PTB<->JILA" not in notes


def test_bbr_report_note_flags_beyond_cross_verified_range_above_300k(tmp_path: Path) -> None:
    data = _base_lattice_stark_dict(tmp_path)
    data["environment"] = {"radiation_temperature_K": 320.0}
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    notes = result.report.uncertainty_notes

    assert "beyond the PTB<->JILA 1e-19-class cross-verification band" in notes


def test_bbr_report_note_with_temperature_uncertainty_omits_conditional_note(
    tmp_path: Path,
) -> None:
    data = _base_lattice_stark_dict(tmp_path)
    data["environment"] = {
        "radiation_temperature_K": 300.0,
        "radiation_temperature_uncertainty_K": 0.004,
    }
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    notes = result.report.uncertainty_notes

    assert "conditional on exact T" not in notes
    assert "radiation_temperature_uncertainty_K propagation" in notes
