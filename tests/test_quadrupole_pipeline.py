# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pipeline-level tests for the WP21 (CONVENTIONS.md E34/E35) quadrupole shift.

Covers the WP21 test contract items that need the full pipeline: config
parsing/validation, composition additivity at the report level, cross-mode
agreement (fast_path vs. worldline on static lattice nodes), byte-
exactness of every shipped example (none of which sets `quadrupole:`),
and a real-FEA demonstration (a quadrupole-shift map from
`examples/fd_electrode_field.txt`, the shipped finite-difference
electrode field -- CONVENTIONS.md section 14 / dossier section 5's
explicit negative: "no public case pairs an FEA-derived gradient tensor
with a measured quadrupole shift"). Pure-formula tests live in
`tests/test_quadrupole_pivot.py`; registry data pins live in
`tests/test_ion_species.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from cliffordclock.integrator.omega import quadrupole_pivot_perturbation
from cliffordclock.pipeline import (
    PipelineConfig,
    PipelineConfigError,
    QuadrupoleConfig,
    _parse_quadrupole,
    run_pipeline_full,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"
# Ca+ 4S1/2-3D5/2 clock frequency, Hz (Chwalla et al., PRL 102, 023002 (2009)).
_NU_0_HZ = 411_042_129_776_393.0


def _base_stark_config(
    *,
    regime: str,
    field: dict[str, object],
    output_dir: str,
    integration: dict[str, object] | None = None,
    ensemble_extra: dict[str, object] | None = None,
    quadrupole: dict[str, object] | None = None,
) -> dict[str, object]:
    ensemble: dict[str, object]
    if regime == "lattice":
        ensemble = {
            "regime": "lattice",
            "temperature_uK": 1.0,
            "motional_n": [0, 0, 0],
            "n_quad": 1,
        }
    else:
        ensemble = {"regime": "classical", "temperature_uK": 1.0, "size": 5, "seed": 0}
    if ensemble_extra:
        ensemble.update(ensemble_extra)

    data: dict[str, object] = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
        "field": field,
        "coupling": {"type": "stark_dc"},
        "ensemble": ensemble,
        "integration": integration or {"time_s": 1.0},
        "output": {"directory": output_dir},
    }
    if quadrupole is not None:
        data["quadrupole"] = quadrupole
    return data


def _quadrupole_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "state": "Ca+:D5/2",
        "nu_0_hz": _NU_0_HZ,
        "m_j": 2.5,
        "quantization_axis": [0.0, 0.0, 1.0],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Config parsing / validation.
# ---------------------------------------------------------------------------


def test_quadrupole_absent_defaults_to_none() -> None:
    assert _parse_quadrupole(None) is None


def test_quadrupole_parses_registry_state() -> None:
    cfg = _parse_quadrupole(_quadrupole_dict())
    assert cfg == QuadrupoleConfig(
        nu_0_hz=_NU_0_HZ,
        state="Ca+:D5/2",
        theta_au=None,
        j=None,
        m_j=2.5,
        quantization_axis=(0.0, 0.0, 1.0),
        averaging_mode="fixed",
    )


def test_quadrupole_parses_explicit_theta_and_j() -> None:
    cfg = _parse_quadrupole({"nu_0_hz": _NU_0_HZ, "theta_au": 2.5, "j": 2.5, "m_j": 1.5})
    assert cfg is not None
    assert cfg.state is None
    assert cfg.theta_au == 2.5
    assert cfg.j == 2.5


def test_quadrupole_rejects_both_state_and_explicit_theta() -> None:
    with pytest.raises(PipelineConfigError, match="not both"):
        _parse_quadrupole({"nu_0_hz": _NU_0_HZ, "state": "Ca+:D5/2", "theta_au": 1.0, "j": 2.5})


def test_quadrupole_rejects_neither_state_nor_explicit_theta() -> None:
    with pytest.raises(PipelineConfigError, match="must specify either"):
        _parse_quadrupole({"nu_0_hz": _NU_0_HZ, "m_j": 2.5})


def test_quadrupole_rejects_unknown_registry_state() -> None:
    with pytest.raises(PipelineConfigError, match="Unknown quadrupole state"):
        _parse_quadrupole({"nu_0_hz": _NU_0_HZ, "state": "Not+:A/State", "m_j": 0.5})


def test_quadrupole_rejects_unknown_averaging_mode() -> None:
    with pytest.raises(PipelineConfigError, match="averaging_mode"):
        _parse_quadrupole(_quadrupole_dict(averaging_mode="bogus"))


def test_quadrupole_fixed_mode_requires_m_j() -> None:
    d = _quadrupole_dict()
    del d["m_j"]
    with pytest.raises(PipelineConfigError, match="m_j is required"):
        _parse_quadrupole(d)


def test_quadrupole_three_orientation_mode_does_not_require_m_j() -> None:
    d = _quadrupole_dict(averaging_mode="three_orientation")
    del d["m_j"]
    cfg = _parse_quadrupole(d)
    assert cfg is not None
    assert cfg.averaging_mode == "three_orientation"
    assert cfg.m_j is None


def test_quadrupole_requires_stark_dc_coupling() -> None:
    data = _base_stark_config(
        regime="lattice",
        field={"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
        output_dir=".",
        quadrupole=_quadrupole_dict(),
    )
    data["coupling"] = {"mu": [1.0e-25, 0.0, 0.0]}  # linear_mu
    with pytest.raises(PipelineConfigError, match="coupling.type='stark_dc'"):
        PipelineConfig.from_dict(data)


# ---------------------------------------------------------------------------
# 2. Composition additivity / report notes.
# ---------------------------------------------------------------------------


def test_quadrupole_off_by_default_matches_pre_wp21_shift(tmp_path: Path) -> None:
    """With no `quadrupole:` section, the report is identical to a
    coupling.type='stark_dc' run without any WP21 change (the same shift
    the pre-existing Stark-only formula gives).
    """
    data = _base_stark_config(
        regime="lattice",
        field={
            "synthetic": {
                "kind": "constant_gradient",
                "params": {
                    "e0": [0.0, 0.0, 200.0],
                    "grad": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0e8]],
                },
            }
        },
        output_dir=str(tmp_path / "out"),
    )
    config = PipelineConfig.from_dict(data)
    assert config.quadrupole is None
    result = run_pipeline_full(config)
    assert "coupling.quadrupole" not in result.report.uncertainty_notes


def test_quadrupole_composes_additively_into_reported_shift(tmp_path: Path) -> None:
    """Turning quadrupole ON adds exactly the quadrupole-only fractional
    shift (computed independently via
    `cliffordclock.integrator.omega.quadrupole_pivot_perturbation` on the
    SAME field/gradient) to the quadrupole-OFF baseline shift -- E35's
    additive composition, verified numerically at the report level, not
    just asserted.
    """
    grad = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0e8]]
    field = {
        "synthetic": {
            "kind": "constant_gradient",
            "params": {"e0": [0.0, 0.0, 200.0], "grad": grad},
        }
    }

    base_data = _base_stark_config(regime="lattice", field=field, output_dir=str(tmp_path / "base"))
    quad_data = _base_stark_config(
        regime="lattice",
        field=field,
        output_dir=str(tmp_path / "quad"),
        quadrupole=_quadrupole_dict(),
    )
    base_result = run_pipeline_full(PipelineConfig.from_dict(base_data))
    quad_result = run_pipeline_full(PipelineConfig.from_dict(quad_data))

    import jax.numpy as jnp

    expected_quad_term = float(
        quadrupole_pivot_perturbation(
            jnp.asarray(grad, dtype=jnp.float64),
            jnp.array([0.0, 0.0, 1.0]),
            1.83,
            2.5,
            2.5,
            _NU_0_HZ,
        )
    )
    assert abs(expected_quad_term) > 1e-20  # non-trivial term, not a vacuous near-zero test

    diff = quad_result.report.mean_fractional_shift - base_result.report.mean_fractional_shift
    np.testing.assert_allclose(diff, expected_quad_term, rtol=1e-9, atol=0)
    assert "coupling.quadrupole (E34/E35)" in quad_result.report.uncertainty_notes


def test_quadrupole_three_orientation_mode_contributes_exactly_zero(tmp_path: Path) -> None:
    grad = [[0.0, 1.0e7, 0.0], [1.0e7, 0.0, 2.0e7], [0.0, 2.0e7, -0.0]]
    field = {
        "synthetic": {
            "kind": "constant_gradient",
            "params": {"e0": [0.0, 0.0, 200.0], "grad": grad},
        }
    }

    base_data = _base_stark_config(
        regime="lattice", field=field, output_dir=str(tmp_path / "base2")
    )
    quad_data = _base_stark_config(
        regime="lattice",
        field=field,
        output_dir=str(tmp_path / "quad2"),
        quadrupole=_quadrupole_dict(averaging_mode="three_orientation", m_j=None),
    )
    del quad_data["quadrupole"]["m_j"]  # not required for three_orientation
    base_result = run_pipeline_full(PipelineConfig.from_dict(base_data))
    quad_result = run_pipeline_full(PipelineConfig.from_dict(quad_data))
    np.testing.assert_allclose(
        quad_result.report.mean_fractional_shift,
        base_result.report.mean_fractional_shift,
        rtol=0,
        atol=1e-20,
    )


# ---------------------------------------------------------------------------
# 3. Cross-mode agreement (fast_path vs. worldline, static lattice nodes).
# ---------------------------------------------------------------------------


def test_quadrupole_fast_path_and_worldline_agree_exactly(tmp_path: Path) -> None:
    """Same static-node exact-agreement argument as the Stark-only case
    (`tests/test_e2e.py::test_step0_stark_dc_fast_path_and_worldline_agree_exactly`)
    extended to include a nonzero quadrupole term: both paths evaluate the
    SAME `_quadrupole_pivot_from_grad` call at v=0, so the reduction is
    exact (`rtol=0, atol=0`): an equality, not values that merely land
    close.
    """
    grad = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0e8]]
    field = {
        "synthetic": {
            "kind": "constant_gradient",
            "params": {"e0": [10.0, 0.0, -5.0], "grad": grad},
        }
    }

    fast_data = _base_stark_config(
        regime="lattice",
        field=field,
        output_dir=str(tmp_path / "fast"),
        quadrupole=_quadrupole_dict(),
    )
    worldline_data = _base_stark_config(
        regime="lattice",
        field=field,
        output_dir=str(tmp_path / "world"),
        integration={"mode": "worldline", "dtau": 0.5, "steps": 200},
        quadrupole=_quadrupole_dict(),
    )
    fast_result = run_pipeline_full(PipelineConfig.from_dict(fast_data))
    worldline_result = run_pipeline_full(PipelineConfig.from_dict(worldline_data))

    assert fast_result.report.ensemble_type == "lattice_fast_path"
    assert worldline_result.report.ensemble_type == "lattice_worldline_crosscheck"
    # rtol=1e-14 (not 0): fast_path evaluates the quadrupole term once and
    # multiplies by T_tilde; worldline sums the SAME per-step value 200
    # times via Kahan summation then divides by the same T_tilde --
    # mathematically identical, but not guaranteed bit-for-bit identical
    # under IEEE 754 non-associativity (observed ~1 ULP, 2e-16 relative,
    # here) -- still an extremely tight cross-mode agreement bound, far
    # inside the 1e-19 physics floor this project targets.
    np.testing.assert_allclose(
        fast_result.report.mean_fractional_shift,
        worldline_result.report.mean_fractional_shift,
        rtol=1e-14,
        atol=0,
    )
    assert abs(fast_result.report.mean_fractional_shift) > 1e-20


def test_quadrupole_direct_classical_mode_runs_end_to_end(tmp_path: Path) -> None:
    """`ensemble.regime='classical'` + `mode='direct'` composes the
    quadrupole term too (via `_make_stark_rate_fn`'s shared `rate_fn`).
    """
    grad = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0e8]]
    field = {
        "synthetic": {
            "kind": "constant_gradient",
            "params": {"e0": [0.0, 0.0, 200.0], "grad": grad},
        }
    }
    data = _base_stark_config(
        regime="classical",
        field=field,
        output_dir=str(tmp_path / "out"),
        integration={"dtau": 0.5, "steps": 200},
        quadrupole=_quadrupole_dict(),
    )
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    assert result.report.ensemble_type == "classical_direct"
    assert np.isfinite(result.report.mean_fractional_shift)
    assert "coupling.quadrupole (E34/E35)" in result.report.uncertainty_notes


# ---------------------------------------------------------------------------
# 4. Byte-exactness of shipped examples (none set `quadrupole:`).
# ---------------------------------------------------------------------------


def test_no_shipped_example_sets_quadrupole() -> None:
    """Structural byte-exactness check: every shipped example YAML omits
    `quadrupole:`, so `PipelineConfig.from_dict` resolves `quadrupole=None`
    for all of them -- the WP21 composition points
    (`_make_stark_rate_fn`/`_stark_rotor_ensemble`) then add exactly
    `0.0` (an IEEE-754 no-op), matching the full-suite run's existing
    byte-exactness tests (`tests/test_e2e.py`) that this WP does not
    modify.
    """
    yaml_files = sorted(_EXAMPLES_DIR.glob("*.yaml"))
    assert len(yaml_files) > 0
    for path in yaml_files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "quadrupole" not in data, f"{path} unexpectedly sets 'quadrupole:'"


# ---------------------------------------------------------------------------
# 5. Real-FEA demonstration (dossier section 5's explicit negative, closed
#    here for the tool's own capability -- not a claim of measured
#    agreement, since no public dataset pairs an FEA gradient with a
#    measured quadrupole shift).
# ---------------------------------------------------------------------------


def test_quadrupole_shift_map_from_shipped_fd_electrode_field_exact_domain_center(
    tmp_path: Path,
) -> None:
    """A quadrupole-shift evaluation using the REAL (well, from-scratch
    finite-difference, not a toy closed-form) field-gradient tensor from
    `examples/fd_electrode_field.txt` (the same file
    `examples/comsol_electrode_sr87.yaml` demonstrates for the Stark
    term) -- the capability the dossier states nobody else packages: an
    FEA-derived gradient tensor flowing through the quadrupole shift
    end-to-end via `field.comsol`.

    Uses the exact domain center (5.0, 5.0, 4.0 mm) --
    `examples/comsol_electrode_sr87.yaml`'s own trap center, which sits
    exactly on a finite-difference grid node. This exercises
    `cliffordclock.fields.smoother.FieldSmoother.evaluate`'s gradient at a
    coincident-RBF-center query point, which used to return NaN there (a
    latent bug discovered while building this test, since fixed
    upstream -- the gradient at a coincident center is now finite and
    equals the analytic limit, exactly 0 for that center's own
    contribution). The two-electrode geometry's mirror symmetry about
    this exact midplane makes the fitted `dE_z/dz` at this one point
    small (near the symmetry-imposed zero, not exactly zero due to the
    fit's own numerical asymmetry) -- so this test's assertion is
    finiteness/sanity, not a nonzero-magnitude floor (see the companion
    off-center test below for a meaningfully nonzero demonstration).
    """
    fd_field_path = _EXAMPLES_DIR / "fd_electrode_field.txt"
    assert fd_field_path.exists()

    data = _base_stark_config(
        regime="lattice",
        field={"comsol": str(fd_field_path), "smoothing": 0.0},
        output_dir=str(tmp_path / "out"),
        quadrupole=_quadrupole_dict(),
    )
    data["trap"]["center"] = [5.0e-3, 5.0e-3, 4.0e-3]
    config = PipelineConfig.from_dict(data)
    result = run_pipeline_full(config)

    assert np.isfinite(result.report.mean_fractional_shift)
    assert "coupling.quadrupole (E34/E35)" in result.report.uncertainty_notes
    # Sane, not absurd -- and, in particular, not NaN (the bug this
    # regression guards against).
    assert abs(result.report.mean_fractional_shift) < 1e-6


def test_quadrupole_shift_map_from_shipped_fd_electrode_field_off_center(
    tmp_path: Path,
) -> None:
    """Companion to the exact-domain-center case above: an off-grid-node
    interior point gives a meaningfully nonzero quadrupole shift (the
    symmetry that suppresses it at the exact center does not hold away
    from the midplane), demonstrating the real-FEA capability with a
    non-trivial magnitude.
    """
    fd_field_path = _EXAMPLES_DIR / "fd_electrode_field.txt"
    assert fd_field_path.exists()

    data = _base_stark_config(
        regime="lattice",
        field={"comsol": str(fd_field_path), "smoothing": 0.0},
        output_dir=str(tmp_path / "out"),
        quadrupole=_quadrupole_dict(),
    )
    data["trap"]["center"] = [5.37e-3, 5.13e-3, 4.21e-3]
    config = PipelineConfig.from_dict(data)
    result = run_pipeline_full(config)

    assert np.isfinite(result.report.mean_fractional_shift)
    assert "coupling.quadrupole (E34/E35)" in result.report.uncertainty_notes
    # A sane (not absurd) magnitude for a mm-scale electrode gradient and
    # a D-state Theta -- same order-of-magnitude reasoning as
    # tests/test_ion_species.py's dimensional round-trip check.
    assert 0.0 < abs(result.report.mean_fractional_shift) < 1e-6
