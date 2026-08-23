# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP22 Part 2 extended-lattice ensemble regime
(CONVENTIONS.md section 15).

``cliffordclock.ensemble.lattice.extended_lattice_nodes`` builds the
site-and-motional geometry; ``ensemble.regime='lattice_extended'``
(``cliffordclock.pipeline``) dispatches it through the SAME
``fast_path``/``worldline`` accumulators the `lattice` regime uses, and
``cliffordclock.pipeline._build_site_map`` assembles the per-site
frequency map (the Bothwell observable) plus the gate-edit-4 dispersion-
labeling numbers.

Covers: the site geometry (envelope weighting, offsets, byte-exactness of
the untouched `lattice` regime), config parsing/validation, cross-mode
agreement (fast_path = worldline) with gravity active, hand-computed
per-site-map pins, and the test-pinned dispersion-labeling wording.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from cliffordclock.constants import SPEED_OF_LIGHT, STANDARD_GRAVITY
from cliffordclock.ensemble.lattice import (
    VALID_SITE_ENVELOPES,
    extended_lattice_nodes,
    hermite_gaussian_nodes,
)
from cliffordclock.ensemble.species import get_species
from cliffordclock.ensemble.traps import HarmonicTrap
from cliffordclock.pipeline import (
    LATTICE_EXTENDED_DISPERSION_LABEL_NOTE,
    EnsembleConfig,
    PipelineConfig,
    PipelineConfigError,
    _parse_ensemble,
    run_pipeline_full,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

# ---------------------------------------------------------------------------
# 1. Site geometry: `extended_lattice_nodes` unit tests (envelope weighting,
#    offsets, byte-exactness of the untouched `hermite_gaussian_nodes`).
# ---------------------------------------------------------------------------


def test_extended_lattice_nodes_offsets_symmetric_about_trap_center() -> None:
    species = get_species("Sr87")
    trap = HarmonicTrap(omega_xyz=(2.0e5, 2.0e5, 2.0e5), center=(0.0, 0.0, 1.0))
    geometry = extended_lattice_nodes(
        species, trap, (0, 0, 0), 1, 4, 0.5, (0.0, 0.0, 1.0), "uniform", None
    )
    # n_sites=4: offsets (i - 1.5)*0.5 for i=0..3 -> [-0.75, -0.25, 0.25, 0.75]
    np.testing.assert_allclose(
        np.asarray(geometry.site_offsets_m), [-0.75, -0.25, 0.25, 0.75], rtol=0, atol=0
    )
    # site_centers = trap.center + offset*axis_hat
    expected_z = np.array([1.0 - 0.75, 1.0 - 0.25, 1.0 + 0.25, 1.0 + 0.75])
    np.testing.assert_allclose(np.asarray(geometry.site_centers)[:, 2], expected_z, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(geometry.site_centers)[:, :2], 0.0, rtol=0, atol=0)


def test_extended_lattice_nodes_uniform_envelope_equal_weights() -> None:
    species = get_species("Sr87")
    trap = HarmonicTrap(omega_xyz=(2.0e5, 2.0e5, 2.0e5))
    geometry = extended_lattice_nodes(
        species, trap, (0, 0, 0), 1, 5, 1.0, (0.0, 0.0, 1.0), "uniform", None
    )
    np.testing.assert_allclose(np.asarray(geometry.site_weights), 0.2, rtol=0, atol=1e-15)
    assert float(jnp.sum(geometry.site_weights)) == pytest.approx(1.0, abs=1e-14)


def test_extended_lattice_nodes_gaussian_envelope_hand_computed() -> None:
    """3 sites, spacing 1.0, sigma 1.0 -> offsets [-1, 0, 1]; hand-computed
    raw Gaussian weights exp(-0.5*offset^2/sigma^2), normalized.
    """
    species = get_species("Sr87")
    trap = HarmonicTrap(omega_xyz=(2.0e5, 2.0e5, 2.0e5))
    geometry = extended_lattice_nodes(
        species, trap, (0, 0, 0), 1, 3, 1.0, (0.0, 0.0, 1.0), "gaussian", 1.0
    )
    raw = np.array([np.exp(-0.5), 1.0, np.exp(-0.5)])
    expected = raw / raw.sum()
    np.testing.assert_allclose(np.asarray(geometry.site_weights), expected, rtol=1e-14, atol=0)
    assert float(jnp.sum(geometry.site_weights)) == pytest.approx(1.0, abs=1e-14)


def test_extended_lattice_nodes_local_weights_sum_to_one_and_match_lattice_regime() -> None:
    """Every site's local Hermite-Gauss quadrature is bit-for-bit the SAME
    as a plain `lattice` regime call against a `center=(0,0,0)` trap
    (byte-exactness of the underlying `hermite_gaussian_nodes` machinery --
    the extended regime reuses it verbatim, not a re-derivation).
    """
    species = get_species("Sr87")
    trap = HarmonicTrap(omega_xyz=(2.0e5, 2.1e5, 2.2e5), center=(3.0, -1.0, 0.5))
    geometry = extended_lattice_nodes(
        species, trap, (1, 0, 2), 4, 3, 0.4, (0.0, 0.0, 1.0), "uniform", None
    )
    assert float(jnp.sum(geometry.local_weights)) == pytest.approx(1.0, abs=1e-13)

    reference_trap = HarmonicTrap(omega_xyz=trap.omega_xyz, center=(0.0, 0.0, 0.0))
    ref_nodes, ref_weights = hermite_gaussian_nodes(species, reference_trap, (1, 0, 2), 4)
    np.testing.assert_allclose(
        np.asarray(geometry.local_weights), np.asarray(ref_weights), rtol=0, atol=0
    )
    # Site 1 (the middle of 3, offset 0 by construction) reproduces the
    # reference nodes exactly (translated by its own, zero, offset).
    n_local = ref_nodes.shape[0]
    site1_nodes = np.asarray(geometry.nodes).reshape(3, n_local, 3)[1]
    np.testing.assert_allclose(
        site1_nodes, np.asarray(ref_nodes) + np.asarray(trap.center), rtol=0, atol=0
    )


def test_extended_lattice_nodes_weights_sum_to_one() -> None:
    species = get_species("Sr87")
    trap = HarmonicTrap(omega_xyz=(2.0e5, 2.0e5, 2.0e5))
    geometry = extended_lattice_nodes(
        species, trap, (0, 1, 0), 3, 7, 0.2, (1.0, 0.0, 0.0), "gaussian", 0.5
    )
    assert float(jnp.sum(geometry.weights)) == pytest.approx(1.0, abs=1e-12)
    assert geometry.nodes.shape == (7 * 3**3, 3)
    assert geometry.weights.shape == (7 * 3**3,)


@pytest.mark.parametrize(
    ("n_sites", "spacing", "axis", "envelope", "sigma", "match"),
    [
        (0, 1.0, (0.0, 0.0, 1.0), "uniform", None, "n_sites must be >= 1"),
        (3, 0.0, (0.0, 0.0, 1.0), "uniform", None, "site_spacing_m must be > 0"),
        (3, 1.0, (0.0, 0.0, 0.0), "uniform", None, "zero vector"),
        (3, 1.0, (0.0, 0.0, 1.0), "triangular", None, "envelope must be one of"),
        (3, 1.0, (0.0, 0.0, 1.0), "gaussian", None, "envelope_sigma_m"),
        (3, 1.0, (0.0, 0.0, 1.0), "gaussian", -1.0, "envelope_sigma_m"),
    ],
)
def test_extended_lattice_nodes_validates_inputs(
    n_sites: int,
    spacing: float,
    axis: tuple[float, float, float],
    envelope: str,
    sigma: float | None,
    match: str,
) -> None:
    species = get_species("Sr87")
    trap = HarmonicTrap(omega_xyz=(2.0e5, 2.0e5, 2.0e5))
    with pytest.raises(ValueError, match=match):
        extended_lattice_nodes(species, trap, (0, 0, 0), 1, n_sites, spacing, axis, envelope, sigma)


def test_extended_lattice_nodes_single_site_reduces_to_lattice_regime() -> None:
    """n_sites=1 degenerates to exactly the `lattice` regime's own single
    Hermite-Gauss quadrature at `trap.center` (weight 1.0)."""
    species = get_species("Sr87")
    trap = HarmonicTrap(omega_xyz=(2.0e5, 2.0e5, 2.0e5), center=(0.1, 0.2, 0.3))
    geometry = extended_lattice_nodes(
        species, trap, (0, 1, 0), 4, 1, 1.0, (0.0, 0.0, 1.0), "uniform", None
    )
    ref_nodes, ref_weights = hermite_gaussian_nodes(species, trap, (0, 1, 0), 4)
    np.testing.assert_allclose(np.asarray(geometry.nodes), np.asarray(ref_nodes), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(geometry.weights), np.asarray(ref_weights), rtol=0, atol=1e-15
    )


def test_valid_site_envelopes_constant() -> None:
    assert VALID_SITE_ENVELOPES == ("gaussian", "uniform")


# ---------------------------------------------------------------------------
# 2. Pipeline config parsing / validation.
# ---------------------------------------------------------------------------


def _lattice_extended_ensemble_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "regime": "lattice_extended",
        "temperature_uK": 1.0,
        "motional_n": [0, 0, 0],
        "n_quad": 1,
        "n_sites": 3,
        "site_spacing_m": 0.5,
        "site_axis": [0.0, 0.0, 1.0],
        "site_envelope": "gaussian",
        "site_envelope_sigma_m": 0.5,
    }
    base.update(overrides)
    return base


def test_parse_ensemble_lattice_extended_valid() -> None:
    cfg = _parse_ensemble(_lattice_extended_ensemble_dict())
    assert cfg.regime == "lattice_extended"
    assert cfg.n_sites == 3
    assert cfg.site_spacing_m == 0.5
    assert cfg.site_envelope == "gaussian"
    assert cfg.site_envelope_sigma_m == 0.5


def test_parse_ensemble_lattice_extended_defaults() -> None:
    data = _lattice_extended_ensemble_dict()
    del data["site_axis"]
    del data["site_envelope"]
    cfg = _parse_ensemble(data)
    assert cfg.site_axis == (0.0, 0.0, 1.0)
    assert cfg.site_envelope == "gaussian"


def test_parse_ensemble_lattice_extended_requires_motional_n() -> None:
    data = _lattice_extended_ensemble_dict()
    del data["motional_n"]
    with pytest.raises(PipelineConfigError, match="motional_n is required"):
        _parse_ensemble(data)


def test_parse_ensemble_lattice_extended_requires_n_sites() -> None:
    data = _lattice_extended_ensemble_dict()
    del data["n_sites"]
    with pytest.raises(PipelineConfigError, match="n_sites is required"):
        _parse_ensemble(data)


def test_parse_ensemble_lattice_extended_rejects_non_positive_n_sites() -> None:
    with pytest.raises(PipelineConfigError, match="n_sites is required"):
        _parse_ensemble(_lattice_extended_ensemble_dict(n_sites=0))


def test_parse_ensemble_lattice_extended_requires_site_spacing() -> None:
    data = _lattice_extended_ensemble_dict()
    del data["site_spacing_m"]
    with pytest.raises(PipelineConfigError, match="site_spacing_m is required"):
        _parse_ensemble(data)


def test_parse_ensemble_lattice_extended_rejects_zero_site_axis() -> None:
    with pytest.raises(PipelineConfigError, match="site_axis must not be the zero vector"):
        _parse_ensemble(_lattice_extended_ensemble_dict(site_axis=[0.0, 0.0, 0.0]))


def test_parse_ensemble_lattice_extended_rejects_bad_envelope() -> None:
    with pytest.raises(PipelineConfigError, match="site_envelope must be one of"):
        _parse_ensemble(_lattice_extended_ensemble_dict(site_envelope="triangular"))


def test_parse_ensemble_lattice_extended_gaussian_requires_sigma() -> None:
    data = _lattice_extended_ensemble_dict()
    del data["site_envelope_sigma_m"]
    with pytest.raises(PipelineConfigError, match="site_envelope_sigma_m is required"):
        _parse_ensemble(data)


def test_parse_ensemble_lattice_extended_uniform_ignores_sigma() -> None:
    data = _lattice_extended_ensemble_dict(site_envelope="uniform")
    del data["site_envelope_sigma_m"]
    cfg = _parse_ensemble(data)
    assert cfg.site_envelope_sigma_m is None


def test_valid_ensemble_regimes_includes_lattice_extended() -> None:
    from cliffordclock.pipeline import VALID_ENSEMBLE_REGIMES

    assert VALID_ENSEMBLE_REGIMES == ("classical", "lattice", "lattice_extended")


def test_unknown_regime_still_rejected() -> None:
    with pytest.raises(PipelineConfigError, match="ensemble.regime must be one of"):
        _parse_ensemble({"regime": "nonexistent", "temperature_uK": 1.0})


# ---------------------------------------------------------------------------
# 3. Byte-exactness: the existing `lattice`/`classical` regimes are entirely
#    untouched by WP22 (no shipped example uses `lattice_extended`).
# ---------------------------------------------------------------------------


def test_no_shipped_example_uses_lattice_extended_regime() -> None:
    example_paths = sorted(_EXAMPLES_DIR.glob("*.yaml"))
    assert len(example_paths) >= 5
    for path in example_paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["ensemble"]["regime"] != "lattice_extended", (
            f"{path.name} unexpectedly uses ensemble.regime=lattice_extended"
        )


@pytest.mark.parametrize(
    ("example_name", "expected_shift"),
    [
        ("lattice_sr87.yaml", 1.2180934201244512e-10),
        ("lattice_sr87_stark.yaml", -7.170523825042889e-17),
        ("realistic_lattice_sr87.yaml", -7.723398671928127e-19),
    ],
)
def test_shipped_lattice_examples_byte_identical_with_wp22_present(
    example_name: str, expected_shift: float, tmp_path: Path
) -> None:
    data = yaml.safe_load((_EXAMPLES_DIR / example_name).read_text(encoding="utf-8"))
    assert data["ensemble"]["regime"] == "lattice"
    data["output"] = dict(data.get("output") or {})
    data["output"]["directory"] = str(tmp_path)
    config = PipelineConfig.from_dict(data)
    result = run_pipeline_full(config)
    # rtol=1e-12, not exact: the same cross-platform XLA scheduling drift
    # documented at test_bbr_pipeline.py's sibling snapshot test (runner
    # linux/x86 measured 5.0e-16 relative here, 2026-08-22) breaks a
    # bit-for-bit contract; the bound absorbs measured version and
    # platform drift while leaving any real numeric regression 4+ orders
    # of magnitude outside it.
    np.testing.assert_allclose(
        result.report.mean_fractional_shift, expected_shift, rtol=1e-12, atol=0
    )


# ---------------------------------------------------------------------------
# 4. Pipeline-level runs: PipelineResult.site_map is None for non-extended
#    regimes; cross-mode agreement (fast_path = worldline) with gravity
#    active; per-site map hand-computed pins; envelope-weighting composition
#    through the report; dispersion-labeling wording pin.
# ---------------------------------------------------------------------------


def _base_lattice_extended_dict(tmp_path: Path, **ensemble_overrides: object) -> dict[str, object]:
    return {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 0.0]}}},
        "coupling": {"type": "stark_dc"},
        "ensemble": _lattice_extended_ensemble_dict(**ensemble_overrides),
        "integration": {"mode": "fast_path", "time_s": 1.0},
        "output": {"directory": str(tmp_path)},
    }


def test_site_map_is_none_for_lattice_and_classical_regimes(tmp_path: Path) -> None:
    lattice_data: dict[str, object] = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
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
    result = run_pipeline_full(PipelineConfig.from_dict(lattice_data))
    assert result.site_map is None


def test_site_map_hand_computed_gravity_only_uniform_envelope(tmp_path: Path) -> None:
    """3 uniformly-weighted sites at offsets [-1, 0, +1] m along z, gravity
    the ONLY active pivot term (field is exactly zero everywhere): each
    site's mean shift is exactly `g/c^2 * offset` (no BBR, no Stark, no
    motional spread -- n_quad=1 puts every site's single node exactly at
    its own center). The fit is then EXACT: slope = g/c^2, intercept = 0,
    total spread = g/c^2 (the pure gradient), and the gradient-removed
    residual spread is exactly 0 (a perfectly linear map has no residual).
    """
    data = _base_lattice_extended_dict(
        tmp_path,
        n_sites=3,
        site_spacing_m=1.0,
        site_envelope="uniform",
        site_envelope_sigma_m=None,
    )
    data["environment"] = {"gravity": {"g_m_s2": STANDARD_GRAVITY}}
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    site_map = result.site_map
    assert site_map is not None

    g_over_c2 = STANDARD_GRAVITY / SPEED_OF_LIGHT**2
    offsets = [site.offset_m for site in site_map.sites]
    np.testing.assert_allclose(offsets, [-1.0, 0.0, 1.0], rtol=0, atol=0)

    shifts = [site.mean_fractional_shift for site in site_map.sites]
    np.testing.assert_allclose(shifts, [-g_over_c2, 0.0, g_over_c2], rtol=1e-9, atol=0)

    weights = [site.weight for site in site_map.sites]
    np.testing.assert_allclose(weights, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], rtol=1e-13, atol=0)

    np.testing.assert_allclose(site_map.slope_per_m, g_over_c2, rtol=1e-9, atol=0)
    np.testing.assert_allclose(site_map.intercept, 0.0, rtol=0, atol=1e-25)
    np.testing.assert_allclose(site_map.total_spread_fractional, g_over_c2, rtol=1e-9, atol=0)
    np.testing.assert_allclose(
        site_map.gradient_removed_residual_spread_fractional, 0.0, rtol=0, atol=1e-25
    )
    assert site_map.dispersion_label_note == LATTICE_EXTENDED_DISPERSION_LABEL_NOTE
    assert site_map.site_axis == (0.0, 0.0, 1.0)


def test_site_map_local_field_varies_by_site_position() -> None:
    """Part 2 item 1's "site positions feed every pivot term (local field
    ...)" claim, checked directly (not merely gravity's): a spatially
    varying field (constant_gradient) makes the per-site Stark shift
    genuinely differ across sites -- not evaluated once at a single global
    position and broadcast.
    """
    from cliffordclock.pipeline import EnsembleConfig as _EnsembleConfig
    from cliffordclock.pipeline import _build_site_map

    species = get_species("Sr87")
    trap = HarmonicTrap(omega_xyz=(2.0e5, 2.0e5, 2.0e5))
    geometry = extended_lattice_nodes(
        species, trap, (0, 0, 0), 1, 3, 0.01, (0.0, 0.0, 1.0), "uniform", None
    )
    from cliffordclock.fields.synthetic import as_field_fn, constant_gradient_field
    from cliffordclock.integrator import fastpath
    from cliffordclock.pipeline import _make_stark_rate_fn

    # A nonzero e0 baseline breaks the +z/-z symmetry a pure gradient alone
    # would have (|E|^2 is even in z for e0=0, so the two outer sites would
    # coincidentally match) -- e0 + z*grad makes |E(z)|^2, and hence each
    # site's E14b Stark shift, genuinely distinct at all three sites.
    grad = jnp.zeros((3, 3)).at[2, 2].set(1e6)  # dE_z/dz, V/m^2
    e_fn, grad_fn = constant_gradient_field(jnp.array([0.0, 0.0, 1e3]), grad)
    field_fn = as_field_fn(e_fn, grad_fn)
    rate_fn = _make_stark_rate_fn(field_fn, species)
    ensemble_result = fastpath.lattice_shift_expectation(
        rate_fn, geometry.nodes, geometry.weights, 1.0
    )
    ensemble_cfg = _EnsembleConfig(
        regime="lattice_extended",
        temperature_uK=1.0,
        motional_n=(0, 0, 0),
        n_quad=1,
        n_sites=3,
        site_spacing_m=0.01,
        site_axis=(0.0, 0.0, 1.0),
        site_envelope="uniform",
    )
    site_map = _build_site_map(ensemble_cfg, geometry, ensemble_result)
    shifts = [site.mean_fractional_shift for site in site_map.sites]
    # Not all equal -- the local field (and hence the E14b Stark shift)
    # genuinely differs across sites.
    assert len({round(s, 30) for s in shifts}) == 3
    # E_z(z) = 1e3 + z*1e6: |E|^2 at z=-0.01/0/+0.01 is 8.1e7/1e6/1.21e8 --
    # the Stark shift magnitude (proportional to |E|^2) must follow the same
    # strict ordering.
    assert abs(shifts[1]) < abs(shifts[0]) < abs(shifts[2])


@pytest.mark.slow
def test_lattice_extended_fast_path_matches_worldline_with_gravity_active(tmp_path: Path) -> None:
    """E29's exact-agreement claim, extended to `lattice_extended` with
    gravity active: every node is static (v=0), so mode="worldline" must
    reproduce mode="fast_path" exactly.
    """
    base = _base_lattice_extended_dict(
        tmp_path / "fp", n_sites=3, site_spacing_m=0.3, site_envelope="uniform"
    )
    del base["ensemble"]["site_envelope_sigma_m"]  # type: ignore[union-attr]
    base["environment"] = {"gravity": {"g_m_s2": 9.80665}}

    fast_path_data = dict(base)
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
    # Non-vacuous: gravity is actually contributing per-site -- NOT checked
    # via the ensemble MEAN (a symmetric 3-site uniform envelope has sites
    # at [-0.3, 0, +0.3]m, so the linear gravity term's mean cancels to
    # EXACTLY zero by symmetry, a real feature, not a bug) but via the
    # site map's slope, which is insensitive to that cancellation.
    assert fast_path_result.site_map is not None
    assert abs(fast_path_result.site_map.slope_per_m) > 0.0
    # THE discriminating cross-mode assertion (WP22 review blocker 2):
    # the mean comparison above cancels the gravity term by symmetry, so
    # a worldline branch that silently dropped gravity (the exact bug
    # class found and fixed in the pre-existing lattice branch mid-build)
    # would still pass it. The slope comparison does not cancel: with
    # gravity dropped, worldline slope reads 0.0 vs fast_path ~1.09e-16,
    # an 11-orders-of-magnitude miss at this rtol (reviewer kill-tested
    # by monkeypatching gravity=None into _stark_rotor_ensemble).
    assert worldline_result.site_map is not None
    np.testing.assert_allclose(
        worldline_result.site_map.slope_per_m,
        fast_path_result.site_map.slope_per_m,
        rtol=1e-9,
        atol=0,
    )


@pytest.mark.slow
def test_lattice_extended_linear_mu_worldline_cross_check_runs(tmp_path: Path) -> None:
    """`coupling.type='linear_mu'` (the E14a validation coupling) also
    works with `lattice_extended` via the `worldline` rotor path, exactly
    as it does for the plain `lattice` regime -- no gravity/BBR/quadrupole
    (all `coupling.type='stark_dc'`-only), just the geometry itself.
    """
    data: dict[str, object] = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
        "coupling": {"mu": [1.0e-30, 0.0, 0.0]},
        "ensemble": _lattice_extended_ensemble_dict(
            n_sites=2, site_spacing_m=0.1, site_envelope="uniform"
        ),
        "integration": {"mode": "worldline", "time_s": 1.0},
        "output": {"directory": str(tmp_path)},
    }
    del data["ensemble"]["site_envelope_sigma_m"]  # type: ignore[index]
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    assert result.site_map is not None
    assert len(result.site_map.sites) == 2


def test_lattice_extended_gaussian_envelope_weights_composed_into_report(tmp_path: Path) -> None:
    """A Gaussian envelope's site weights actually reach the reported
    ensemble mean (E23): the ensemble mean with a Gaussian envelope
    concentrated near a single (near-zero-height) site differs from the
    same geometry's uniform-envelope mean, when gravity gives each site a
    distinct height.
    """
    gaussian_data = _base_lattice_extended_dict(
        tmp_path / "g",
        n_sites=5,
        site_spacing_m=1.0,
        site_envelope="gaussian",
        site_envelope_sigma_m=0.3,
    )
    gaussian_data["environment"] = {"gravity": {"g_m_s2": 9.80665}}
    gaussian_result = run_pipeline_full(PipelineConfig.from_dict(gaussian_data))

    uniform_data = _base_lattice_extended_dict(
        tmp_path / "u",
        n_sites=5,
        site_spacing_m=1.0,
        site_envelope="uniform",
        site_envelope_sigma_m=None,
    )
    uniform_data["environment"] = {"gravity": {"g_m_s2": 9.80665}}
    uniform_result = run_pipeline_full(PipelineConfig.from_dict(uniform_data))

    # Both are (near-)zero by symmetry (the envelope is symmetric about the
    # middle site, offset 0) -- so compare the SPREAD, not the mean: a
    # tightly concentrated Gaussian envelope has much smaller total spread
    # than the uniform envelope over the same 5-site, 4m-wide geometry.
    np.testing.assert_allclose(gaussian_result.report.mean_fractional_shift, 0.0, atol=1e-24)
    np.testing.assert_allclose(uniform_result.report.mean_fractional_shift, 0.0, atol=1e-24)
    assert gaussian_result.site_map is not None
    assert uniform_result.site_map is not None
    assert (
        gaussian_result.site_map.total_spread_fractional
        < uniform_result.site_map.total_spread_fractional
    )


# ---------------------------------------------------------------------------
# 5. Dispersion-labeling wording pin (G9 sign-off A4#2, gate edit 4).
# ---------------------------------------------------------------------------


def test_dispersion_label_note_wording_pinned() -> None:
    note = LATTICE_EXTENDED_DISPERSION_LABEL_NOTE
    assert "DETERMINISTIC" in note
    assert "stochastic" in note
    assert "site_map.slope_per_m" in note
    assert "gradient_removed_residual_spread_fractional" in note
    assert "total_spread_fractional" in note
    assert "Do not read t2_star_s alone as a decoherence time" in note


def test_dispersion_label_note_present_in_report_uncertainty_notes(tmp_path: Path) -> None:
    data = _base_lattice_extended_dict(tmp_path, n_sites=3, site_spacing_m=0.2)
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    assert LATTICE_EXTENDED_DISPERSION_LABEL_NOTE in result.report.uncertainty_notes


def test_dispersion_label_note_absent_for_plain_lattice_regime(tmp_path: Path) -> None:
    """The dispersion-labeling note is specific to `lattice_extended` -- a
    plain `lattice` run must not carry it (would be a false/irrelevant
    claim about a single-site ensemble with no per-site map at all).
    """
    data: dict[str, object] = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
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
    result = run_pipeline_full(PipelineConfig.from_dict(data))
    assert "lattice_extended dispersion labeling" not in result.report.uncertainty_notes


# ---------------------------------------------------------------------------
# 6. EnsembleConfig dataclass defaults (new fields do not disturb existing
#    positional/keyword construction).
# ---------------------------------------------------------------------------


def test_ensemble_config_new_fields_default_to_none_or_documented_default() -> None:
    cfg = EnsembleConfig(regime="lattice", temperature_uK=1.0)
    assert cfg.n_sites is None
    assert cfg.site_spacing_m is None
    assert cfg.site_axis == (0.0, 0.0, 1.0)
    assert cfg.site_envelope == "gaussian"
    assert cfg.site_envelope_sigma_m is None
