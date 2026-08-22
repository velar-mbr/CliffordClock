# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end analytical validation (WP6 scope item 3) -- the G3 core.

Implements the four WP6 test-contract cases against
``cliffordclock.pipeline``/``cliffordclock.cli``:

- **Case A** (``test_case_a_*``): null result -- uniform synthetic field,
  classical ensemble; reported mean rotor-rate shift beyond the scalar
  baseline < 1e-19; report/CSV files well-formed.
- **Case B** (``test_case_b_*``): constant-gradient field, single static
  atom (a lattice node pinned at the trap center); pipeline result matches
  the CONVENTIONS.md V2 closed form to 1e-12 relative.
- **Case C** (``test_case_c_*``): quadrupole field, M=100 classical
  ensemble, short integration; agreement with the independent plain-NumPy
  ``tests/reference_impl.py`` cross-check to 1e-10 relative on the mean
  shift, **plus** a rotor-path companion assertion (WP6 review MAJOR 1
  fix) that routes the same trajectories through ``exp_bivector``/
  ``geometric_product`` (the E24 rotor-extracted phase) and checks
  agreement with the same external reference within an E24-derived
  tolerance. The primary scalar-vs-scalar comparison alone exercises
  trajectory generation, E14a/E21/E22 wiring, and E23 aggregation, but
  *zero* of the rotor kernel (the library's primary phase is read
  directly off the interaction bivector's e12 component,
  ``integrator/stepper.py``'s ``dphase_scalar`` -- never composed through
  an actual rotor); the rotor kernel's own algebraic correctness is
  independently oracle-tested in WP1 (``tests/test_cl13_oracle.py``). See
  ``tests/reference_impl.py``'s module docstring "Scope note" and
  ``test_case_c_rotor_path_e24_cross_check`` below for the rest of the
  story.
- **Case D** (``test_case_d_*``): CLI smoke test -- ``cliffordclock run`` on
  ``examples/quadrupole_classical.yaml`` via subprocess.

Also covers the WP6-introduced public surface directly: the
``fields.synthetic.as_field_fn`` adapter (WP3-review interface note) and
``PipelineConfig``/CLI error handling, since this is the only test file
this work package may add new tests to (aside from the narrow M=1/dtype
additions in ``tests/test_analytics_*.py``).
"""

from __future__ import annotations

import csv
import decimal
import json
import math
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

import reference_impl
from cliffordclock.analytics import write_json, write_line_profile_csv
from cliffordclock.cli import main as cli_main
from cliffordclock.constants import ELECTRON_MASS, SPEED_OF_LIGHT, TAU_COMPTON
from cliffordclock.ensemble.classical import sample_maxwell_boltzmann
from cliffordclock.ensemble.lattice import hermite_gaussian_nodes
from cliffordclock.ensemble.species import get_species
from cliffordclock.ensemble.traps import HarmonicTrap
from cliffordclock.fields import load_field_comsol
from cliffordclock.fields.synthetic import as_field_fn, constant_gradient_field, uniform_field
from cliffordclock.integrator import fastpath
from cliffordclock.integrator.worldline import EnsembleResult
from cliffordclock.pipeline import (
    _RAMSEY_VISIBILITY_NOTE,
    DEFAULT_MAX_TRAJECTORY_MEMORY_GB,
    MAX_ROTOR_NORM_ERROR,
    VALID_EVALUATION_MODES,
    PhysicsValidationError,
    PipelineConfig,
    PipelineConfigError,
    PipelineResult,
    _auto_renorm_every,
    _build_field_fn,
    _estimate_trajectory_memory_gb,
    _resolve_evaluation_mode,
    _resolve_stark_coupling,
    _stark_rotor_ensemble,
    _validate_physics,
    run_pipeline,
    run_pipeline_full,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

_M_E_C2 = ELECTRON_MASS * SPEED_OF_LIGHT**2


# ---------------------------------------------------------------------------
# Case A -- null result (uniform field, classical ensemble).
# ---------------------------------------------------------------------------


def _case_a_config(output_dir: Path) -> PipelineConfig:
    return PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1000.0]}}},
            "coupling": {"mu": [0.0, 0.0, 1.0e-25]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 30, "seed": 0},
            "integration": {"dtau": 0.5, "steps": 200},
            "output": {"directory": str(output_dir)},
        }
    )


def _assert_numeric_file_match(regenerated: bytes, committed: bytes, context: str) -> None:
    """Structure-exact, numerically tolerant file comparison for generated
    field exports. Byte identity held within one platform but the FD
    solver's conjugate-gradient iterates differ in low-order digits
    across BLAS/platform (runner-measured 2026-08-22: digit-level diffs
    deep in the float text on linux/x86 vs the committed macOS output),
    so the portable contract is: identical line count, identical
    non-numeric text, every number equal to 1e-9 relative with a
    per-file magnitude-scaled absolute floor for zero crossings.
    """
    num_re = re.compile(rb"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")
    reg_lines = regenerated.replace(b"\r\n", b"\n").split(b"\n")
    com_lines = committed.replace(b"\r\n", b"\n").split(b"\n")
    assert len(reg_lines) == len(com_lines), (
        f"{context}: line count {len(reg_lines)} vs {len(com_lines)}"
    )
    reg_nums: list[float] = []
    com_nums: list[float] = []
    for i, (rl, cl) in enumerate(zip(reg_lines, com_lines, strict=True)):
        assert num_re.sub(b"#", rl) == num_re.sub(b"#", cl), (
            f"{context}: non-numeric text differs at line {i + 1}"
        )
        reg_nums.extend(float(m) for m in num_re.findall(rl))
        com_nums.extend(float(m) for m in num_re.findall(cl))
    assert len(reg_nums) == len(com_nums), context
    com_arr = np.asarray(com_nums)
    reg_arr = np.asarray(reg_nums)
    atol = 1e-9 * float(np.max(np.abs(com_arr))) if com_arr.size else 0.0
    np.testing.assert_allclose(reg_arr, com_arr, rtol=1e-9, atol=atol, err_msg=context)


def test_case_a_uniform_field_shift_beyond_baseline_below_1e19(tmp_path: Path) -> None:
    """V1 (CONVENTIONS.md section 9): uniform field, grad_E = 0 everywhere.

    Even though the classical ensemble's atoms move (thermal motion in the
    harmonic trap), the boost term (E18) is exactly zero for every atom at
    every step because it is proportional to `grad_delta_e`, which is
    identically zero for a uniform field -- so the rotor-extracted phase
    (E24) must agree with the primary scalar phase (E21/E22) to within
    numerical precision, per atom, across the whole ensemble.
    """
    config = _case_a_config(tmp_path / "out")
    result = run_pipeline_full(config)

    shift_beyond_baseline = float(
        jnp.max(jnp.abs(result.ensemble_result.phase_rotor - result.ensemble_result.phase))
    )
    assert shift_beyond_baseline < 1e-19


def test_case_a_report_and_csv_well_formed(tmp_path: Path) -> None:
    """Case A also requires well-formed report.json + line_profile.csv output."""
    out_dir = tmp_path / "out"
    config = _case_a_config(out_dir)
    result = run_pipeline_full(config)

    report_path = out_dir / "report.json"
    csv_path = out_dir / "line_profile.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(result.report, report_path)
    write_line_profile_csv(result.line_profile_freqs_hz, result.line_profile_amplitude, csv_path)

    with report_path.open(encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["report_schema"] == "1.1"
    assert loaded["species_name"] == "Sr87"
    assert loaded["ensemble_size"] == 30
    assert np.isfinite(loaded["mean_fractional_shift"])

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0][0].startswith("#")
    assert len(rows) - 1 == result.line_profile_freqs_hz.shape[0]
    # Every data row parses as two floats.
    for row in rows[1:]:
        float(row[0])
        float(row[1])


# ---------------------------------------------------------------------------
# Case B -- constant gradient, single static atom, closed form (V2).
# ---------------------------------------------------------------------------


def test_case_b_constant_gradient_single_atom_matches_closed_form(tmp_path: Path) -> None:
    """V2 (CONVENTIONS.md section 9): a single static atom (one lattice
    node, `motional_n=(0,0,0)`, `n_quad=1`, pinned exactly at the trap
    center by Gauss-Hermite construction) in a constant-gradient field.

    ``ΔΦ = (P(r0) - 1) * T̃ = [(G.(r0-r_ref)).mu / (m_e c^2)] * T̃`` (E14a,
    V2), so ``⟨Δν/ν₀⟩ = ΔΦ/T̃ = P(r0) - 1`` exactly. WP6 tolerance: 1e-12
    relative.
    """
    r0 = (0.01, -0.02, 0.03)
    grad = [[1.0e3, 2.0e3, 0.0], [0.0, -1.0e3, 5.0e2], [3.0e2, 0.0, -2.0e3]]
    mu = (1.0e-25, 2.0e-25, -3.0e-25)
    dtau = 1.0
    steps = 1000

    config = PipelineConfig.from_dict(
        {
            "species": "Yb171",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": list(r0)},
            "field": {
                "synthetic": {
                    "kind": "constant_gradient",
                    "params": {"e0": [0.0, 0.0, 0.0], "grad": grad},
                }
            },
            "coupling": {"mu": list(mu)},
            "ensemble": {
                "regime": "lattice",
                "temperature_uK": 1.0,
                "motional_n": [0, 0, 0],
                "n_quad": 1,
            },
            "integration": {"dtau": dtau, "steps": steps},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    result = run_pipeline_full(config)

    assert result.report.ensemble_size == 1
    # A single lattice node at n_quad=1 is pinned exactly at the trap
    # center by Gauss-Hermite construction (see ensemble.lattice).
    np.testing.assert_allclose(np.asarray(result.trajectories[0]), np.asarray(r0), rtol=0, atol=0)

    grad_arr = np.asarray(grad, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    r0_arr = np.asarray(r0, dtype=np.float64)
    delta_e_r0 = r0_arr @ grad_arr
    expected_p_minus_1 = float(np.dot(delta_e_r0, mu_arr)) / _M_E_C2

    np.testing.assert_allclose(
        result.report.mean_fractional_shift, expected_p_minus_1, rtol=1e-12, atol=0
    )
    # M=1: variance-based statistics are undefined by construction (WP5;
    # cliffordclock.analytics.stats.weighted_phase_stats), not silently
    # dropped -- the pipeline reports them as NaN rather than raising.
    assert np.isnan(result.report.shift_std_error)
    assert np.isnan(result.report.t2_star_s)


def test_lattice_uniform_field_zero_phase_variance_reports_infinite_t2star(tmp_path: Path) -> None:
    """WP9 regression (latent WP5 edge case found building the known-answer
    validation suite): a lattice ensemble (M > 1 quadrature nodes) in a
    spatially *uniform* synthetic field gives every node the identical
    accumulated phase -- here exactly zero, since ``e0`` is orthogonal to
    ``mu`` -- so the weighted phase variance (E25) is exactly ``0.0``.
    Pre-fix, `dephasing_time_t2star` (E27) then raised an unhandled
    ``ZeroDivisionError`` that propagated as a raw traceback out of
    `run_pipeline_full` / the ``cliffordclock run`` CLI. It must instead complete
    and report ``t2_star_s = +inf`` (E27's ``sigma_Phi -> 0+`` limit: no
    inhomogeneous dephasing), with the disambiguating note recorded and
    the JSON output still strict-RFC-8259 valid (`null`, never an
    ``Infinity`` token) -- see ``docs/report-schema.md``'s null
    convention. Affects any coupling identically (purely a zero-variance
    ensemble property), exercised here through the default lattice fast
    path (E29).
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
            "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
            "ensemble": {
                "regime": "lattice",
                "temperature_uK": 1.0,
                "motional_n": [0, 0, 0],
                "n_quad": 4,
            },
            "integration": {"time_s": 1.0},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    result = run_pipeline_full(config)  # Pre-fix: ZeroDivisionError from analytics/stats.py.

    assert result.report.ensemble_size == 4**3
    assert result.report.ensemble_type == "lattice_fast_path"
    assert result.report.mean_fractional_shift == 0.0
    assert result.report.shift_std_error == 0.0  # zero spread: SEM defined, exactly 0
    assert np.isposinf(result.report.t2_star_s)
    assert "T2* is infinite" in result.report.uncertainty_notes

    report_path = tmp_path / "out" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result.report, report_path)

    def _raise_on_non_finite_constant(token: str) -> float:
        raise ValueError(f"strict JSON parser encountered non-finite constant: {token!r}")

    with report_path.open(encoding="utf-8") as f:
        loaded = json.load(f, parse_constant=_raise_on_non_finite_constant)
    assert loaded["t2_star_s"] is None
    assert loaded["shift_std_error"] == 0.0


# ---------------------------------------------------------------------------
# Case C -- quadrupole field, M=100 classical ensemble, independent cross-check.
# ---------------------------------------------------------------------------

# Shared Case C run parameters (used by both the scalar-path cross-check and
# the rotor-path E24 companion assertion below, so the two tests can never
# silently drift apart on what "Case C" means).
_CASE_C_K = 2.0e6
_CASE_C_MU = (1.0e-24, -5.0e-25, 3.0e-25)
_CASE_C_DTAU = 0.5
_CASE_C_STEPS = 60
_CASE_C_M = 100


def _case_c_config(output_dir: Path) -> PipelineConfig:
    return PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [3.0e5, 3.0e5, 4.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "quadrupole", "params": {"k": _CASE_C_K}}},
            "coupling": {"mu": list(_CASE_C_MU)},
            "ensemble": {
                "regime": "classical",
                "temperature_uK": 2.0,
                "size": _CASE_C_M,
                "seed": 7,
            },
            "integration": {"dtau": _CASE_C_DTAU, "steps": _CASE_C_STEPS},
            "output": {"directory": str(output_dir)},
        }
    )


def test_case_c_quadrupole_independent_cross_check(tmp_path: Path) -> None:
    """Case C, scalar path: agreement with tests/reference_impl.py's
    independent, plain-NumPy re-implementation of E14a/E21/E22, to 1e-10
    relative on the ensemble mean shift -- the sprint's strongest anti-
    self-confirmation test (WP6 spec). If this disagrees beyond tolerance,
    that is a finding to report, not a tolerance to loosen.

    This assertion alone exercises trajectory generation and the E14a/E21/
    E22 scalar phase pipeline + E23 aggregation, but *not* the rotor kernel
    (``exp_bivector``/``geometric_product``/the structure tensor) at all --
    see ``test_case_c_rotor_path_e24_cross_check`` below for the companion
    assertion that does (WP6 review MAJOR 1 fix).
    """
    config = _case_c_config(tmp_path / "out")
    result = run_pipeline_full(config)
    assert result.report.ensemble_size == _CASE_C_M

    trajectories = np.asarray(result.trajectories)  # (M, steps+1, 3)
    mu_arr = np.asarray(_CASE_C_MU, dtype=np.float64)
    reference_phases = np.array(
        [
            reference_impl.accumulate_phase_midpoint(
                trajectories[i], _CASE_C_DTAU, _CASE_C_K, mu_arr
            )
            for i in range(_CASE_C_M)
        ]
    )
    t_tilde = _CASE_C_STEPS * _CASE_C_DTAU
    reference_mean_shift = reference_impl.mean_fractional_shift(reference_phases, t_tilde)

    np.testing.assert_allclose(
        result.report.mean_fractional_shift, reference_mean_shift, rtol=1e-10, atol=0
    )

    # Sanity: the comparison is non-vacuous (the shift is not just zero to
    # float precision, which would trivially "agree" at any tolerance).
    assert abs(reference_mean_shift) > 1e-20


def test_case_c_rotor_path_e24_cross_check(tmp_path: Path) -> None:
    """Case C, rotor path (WP6 review MAJOR 1 fix): the E24 rotor-extracted
    phase agrees with the same independent reference the scalar-path test
    above uses, within a tolerance derived from the E24 acceptance
    criterion (CONVENTIONS.md section 6): first-order equality is
    required; second-order (``O(ω_boost²)``) divergence is permitted.

    Unlike the scalar-path cross-check, ``EnsembleResult.phase_rotor`` is
    accumulated by composing one ``exp_bivector``/``geometric_product``
    rotor step per trajectory step (``cliffordclock.integrator.stepper.rotor_step``),
    so this assertion is the one that actually routes Case C through the
    rotor kernel end-to-end against an external reference number -- the
    scalar-path test above never touches it (see that test's docstring and
    ``tests/reference_impl.py``'s module-docstring "Scope note").

    **Tolerance derivation.** ``tests/reference_impl.py.e24_second_order_bound_phase``
    computes, per atom and independent of ``cliffordclock``, an analytic
    upper bound on the leading (``O(ω_boost²)``) rotor/scalar divergence
    for *these* Case C trajectories/parameters (see that function's
    docstring for the derivation). The ensemble-mean of those per-atom
    bounds, converted to fractional-shift units (``/ t_tilde``, matching
    E23), is combined with a numerical-noise floor at the same relative
    tolerance as the scalar-path cross-check (1e-10, loosened by 10x to
    ``1e-9`` to allow for the rotor path's extra floating-point operations
    per step -- ``exp_bivector``'s 12-term Taylor series plus 10 repeated
    squarings, vs. the scalar path's direct arithmetic) to form the test's
    tolerance, with a further 10x safety margin on the physical term for
    the bound formula's own leading-order truncation:

    ``tolerance = 10 * physical_bound_shift + 1e-9 * |reference_mean_shift|``

    For Case C's actual parameters (thermal Sr-87 at 2 uK, so
    ``v/c ~ 2e-10``), the physical second-order term is astronomically
    small (``~1e-69`` in shift units at the time this test was written --
    the boost bivector is minuscule at these non-relativistic thermal
    velocities) -- far below the numerical floor, consistent with
    CONVENTIONS.md's own note that this divergence is "far below the
    1e-18 floor for realistic gradients". The measured tolerance here is
    therefore dominated entirely by the numerical floor term, not the
    physical one; both are computed and combined regardless, per the WP6
    review's requirement to justify the tolerance from the E24 acceptance
    criterion rather than pick a number that happens to pass.
    """
    config = _case_c_config(tmp_path / "out")
    result = run_pipeline_full(config)
    assert result.report.ensemble_size == _CASE_C_M

    trajectories = np.asarray(result.trajectories)  # (M, steps+1, 3)
    mu_arr = np.asarray(_CASE_C_MU, dtype=np.float64)
    t_tilde = _CASE_C_STEPS * _CASE_C_DTAU

    reference_phases = np.array(
        [
            reference_impl.accumulate_phase_midpoint(
                trajectories[i], _CASE_C_DTAU, _CASE_C_K, mu_arr
            )
            for i in range(_CASE_C_M)
        ]
    )
    reference_mean_shift = reference_impl.mean_fractional_shift(reference_phases, t_tilde)

    per_atom_second_order_bounds = np.array(
        [
            reference_impl.e24_second_order_bound_phase(
                trajectories[i], _CASE_C_DTAU, _CASE_C_K, mu_arr
            )
            for i in range(_CASE_C_M)
        ]
    )
    physical_bound_shift = float(np.mean(per_atom_second_order_bounds)) / t_tilde

    numerical_floor_rtol = 1e-9
    physical_safety_factor = 10.0
    tolerance = physical_safety_factor * physical_bound_shift + numerical_floor_rtol * abs(
        reference_mean_shift
    )

    rotor_mean_shift = float(jnp.mean(result.ensemble_result.phase_rotor)) / t_tilde
    diff = abs(rotor_mean_shift - reference_mean_shift)

    assert diff < tolerance, (
        f"rotor-path mean shift {rotor_mean_shift!r} disagrees with the independent "
        f"reference {reference_mean_shift!r} by {diff!r}, exceeding the E24-derived "
        f"tolerance {tolerance!r} (physical term {physical_bound_shift!r}, numerical "
        f"floor {numerical_floor_rtol * abs(reference_mean_shift)!r})"
    )
    # Non-vacuous: the tolerance must be a meaningful (tight) bound relative
    # to the shift itself, not so loose it passes trivially regardless of
    # what the rotor path computes.
    assert tolerance < 1e-4 * abs(reference_mean_shift)


# ---------------------------------------------------------------------------
# Case D -- CLI smoke test.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_case_d_cli_smoke_quadrupole_classical(tmp_path: Path) -> None:
    """``cliffordclock run`` on examples/quadrupole_classical.yaml via subprocess (WP6 spec)."""
    config_path = _EXAMPLES_DIR / "quadrupole_classical.yaml"
    assert config_path.exists(), f"missing example config: {config_path}"
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cliffordclock.cli",
            "run",
            str(config_path),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "mean fractional shift" in result.stdout

    report_path = out_dir / "report.json"
    csv_path = out_dir / "line_profile.csv"
    assert report_path.exists()
    assert csv_path.exists()

    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)
    assert report["report_schema"] == "1.1"
    assert np.isfinite(report["mean_fractional_shift"])

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) > 1
    assert rows[0][0].startswith("#")


def test_case_d_cli_smoke_lattice_sr87(tmp_path: Path) -> None:
    """The second example (lattice regime) also runs cleanly via the CLI."""
    config_path = _EXAMPLES_DIR / "lattice_sr87.yaml"
    assert config_path.exists(), f"missing example config: {config_path}"
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cliffordclock.cli",
            "run",
            str(config_path),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert (out_dir / "report.json").exists()
    assert (out_dir / "line_profile.csv").exists()


# ---------------------------------------------------------------------------
# Fast-path mode coverage (pipeline-level; CONVENTIONS.md section 12,
# E29-E31; see docs/timescales.md).
# ---------------------------------------------------------------------------


def test_wp8_lattice_one_second_demo_sem_much_less_than_shift(tmp_path: Path) -> None:
    """WP8 test contract item 4: the shipped lattice example at a genuine
    T = 1 s interrogation runs in < 60 s CPU and reports SEM << |shift|.
    """
    config = PipelineConfig.from_yaml(_EXAMPLES_DIR / "lattice_sr87.yaml")
    config = replace(config, output=replace(config.output, directory=str(tmp_path / "out")))
    assert config.integration.time_s == 1.0

    start = time.perf_counter()
    result = run_pipeline_full(config)
    elapsed_s = time.perf_counter() - start

    assert elapsed_s < 60.0, f"1-second demo took {elapsed_s:.2f}s CPU, exceeding the 60s bound"
    assert result.report.interrogation_time_s == 1.0
    assert result.report.ensemble_type == "lattice_fast_path"
    shift = abs(result.report.mean_fractional_shift)
    sem = result.report.shift_std_error
    assert shift > 0.0
    assert sem < 1e-3 * shift, f"SEM {sem!r} is not << |shift| {shift!r} for the shipped config"


def test_wp8_lattice_fast_path_is_default_and_worldline_is_explicit_crosscheck(
    tmp_path: Path,
) -> None:
    """`ensemble.regime: lattice` defaults to `integration.mode: fast_path`
    (E29); `integration.mode: worldline` selects the rotor-integrator
    cross-check on the same nodes, agreeing per E29 (WP8 scope item 1).

    Uses a small `time_s` here (not the 1 s example) so the worldline
    cross-check can actually complete the equivalent Compton-unit steps
    (see `tests/test_fastpath_lattice.py` for the full-precision, direct
    fast-path-vs-worldline unit test at rtol 1e-12).
    """
    base = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        # (Not `quadrupole`: its zero-baseline, odd-in-r field integrates to
        # exactly zero over these symmetric quadrature nodes, which would
        # make the "non-vacuous" check below vacuous. `gaussian_bump` has a
        # nonzero baseline across the node cluster, like `examples/lattice_sr87.yaml`.)
        "field": {
            "synthetic": {
                "kind": "gaussian_bump",
                "params": {
                    "amplitude": [10.0, 0.0, -5.0],
                    "center": [0.0, 0.0, 0.0],
                    "width": 1.0e-6,
                },
            }
        },
        "coupling": {"mu": [1.0e-25, -2.0e-25, 1.5e-25]},
        "ensemble": {
            "regime": "lattice",
            "temperature_uK": 1.0,
            "motional_n": [0, 0, 0],
            "n_quad": 3,
        },
    }
    default_cfg = PipelineConfig.from_dict(
        {
            **base,
            "integration": {"dtau": 0.5, "steps": 2000},
            "output": {"directory": str(tmp_path / "default")},
        }
    )
    worldline_cfg = PipelineConfig.from_dict(
        {
            **base,
            "integration": {"mode": "worldline", "dtau": 0.5, "steps": 2000},
            "output": {"directory": str(tmp_path / "worldline")},
        }
    )
    assert default_cfg.integration.mode == "auto"

    default_result = run_pipeline_full(default_cfg)
    worldline_result = run_pipeline_full(worldline_cfg)

    assert default_result.report.ensemble_type == "lattice_fast_path"
    assert worldline_result.report.ensemble_type == "lattice_worldline_crosscheck"
    np.testing.assert_allclose(
        default_result.report.mean_fractional_shift,
        worldline_result.report.mean_fractional_shift,
        rtol=1e-12,
        atol=0,
    )
    assert abs(default_result.report.mean_fractional_shift) > 1e-20


def test_wp8_classical_secular_mode_via_pipeline(tmp_path: Path) -> None:
    """`integration.mode: secular` (E30) runs end-to-end through the
    pipeline for a classical ensemble in an isotropic trap.
    """
    omega = 2.0e5
    t_orbit_s = 2.0 * 3.141592653589793 / omega
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [omega, omega, omega], "center": [0.0, 0.0, 0.0]},
            "field": {
                "synthetic": {
                    "kind": "constant_gradient",
                    "params": {
                        "e0": [0.0, 0.0, 0.0],
                        "grad": [[1.0e3, 0.0, 0.0], [0.0, -1.0e3, 0.0], [0.0, 0.0, 0.0]],
                    },
                }
            },
            "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 20, "seed": 1},
            "integration": {"mode": "secular", "time_s": 5.0 * t_orbit_s},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    result = run_pipeline_full(config)
    assert result.report.ensemble_type == "classical_secular_average"
    assert result.report.ensemble_size == 20
    np.testing.assert_allclose(
        result.report.interrogation_time_s, 5.0 * t_orbit_s, rtol=1e-14, atol=0
    )
    assert np.isfinite(result.report.mean_fractional_shift)
    assert "integration.mode=secular" in result.report.uncertainty_notes


def test_wp8_pipeline_rejects_secular_for_anisotropic_trap(tmp_path: Path) -> None:
    """E30's validity bound (periodic motion, isotropic trap) surfaces as
    a `PipelineConfigError`, not a raw `ValueError`, at the pipeline
    boundary.
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 3.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
            "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5, "seed": 0},
            "integration": {"mode": "secular", "time_s": 1e-6},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    with pytest.raises(PipelineConfigError, match="isotropic"):
        run_pipeline_full(config)


def test_wp8_pipeline_rejects_mode_invalid_for_regime() -> None:
    """`integration.mode: secular` is classical-only; `regime: lattice` rejects it."""
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
            "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
            "ensemble": {
                "regime": "lattice",
                "temperature_uK": 1.0,
                "motional_n": [0, 0, 0],
                "n_quad": 2,
            },
            "integration": {"mode": "secular", "time_s": 1.0},
        }
    )
    with pytest.raises(PipelineConfigError, match="not valid for ensemble.regime"):
        run_pipeline_full(config)


def test_wp8_classical_direct_mode_auto_selects_dtau_from_time_s(tmp_path: Path) -> None:
    """`integration.mode: direct` (the classical default) with `dtau`
    omitted auto-selects it via E31's `select_dtau`, and computes `steps`
    from `time_s`; the report's `uncertainty_notes` records both the
    resolved `dtau`/`steps` and that `dtau` was auto-selected.
    """
    omega = 2.0e5
    t_orbit_s = 2.0 * 3.141592653589793 / omega
    time_s = 3.0 * t_orbit_s
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [omega, omega, omega], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
            "coupling": {"mu": [1.0e-30, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5, "seed": 2},
            "integration": {"time_s": time_s},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    assert config.integration.dtau is None
    result = run_pipeline_full(config)

    expected_dtau = fastpath.select_dtau(
        HarmonicTrap(omega_xyz=(omega, omega, omega)), config.integration.points_per_period
    )
    assert f"dtau={expected_dtau!r}" in result.report.uncertainty_notes
    assert "dtau_auto_selected=True" in result.report.uncertainty_notes
    np.testing.assert_allclose(result.report.interrogation_time_s, time_s, rtol=1e-3, atol=0)
    assert np.isfinite(result.report.mean_fractional_shift)


# ---------------------------------------------------------------------------
# WP8 review MAJOR 1 -- large-dtau + realistic mu NaN-evasion fix.
# ---------------------------------------------------------------------------

# Reproduction scenario (WP8 review MAJOR 1): a realistic E14a mu (same
# order as examples/quadrupole_classical.yaml's, ~1e-25) combined with the
# auto-selected dtau (~2.4e14 for this trap) drives the per-step rotor
# generator angle to ~8863 rad -- far past exp_bivector's fixed-order
# Taylor convergence range (direct probing: still finite but wildly wrong,
# |R| ~ 1e56, at 5000 rad; NaN/Inf by 10000 rad). A hotter (500 uK)
# ensemble is used so enough atoms sample far enough from the trap center
# for the quadrupole field to actually drive some atoms' per-step angle
# into that regime (a 1 uK ensemble at this mu stays under the
# MAX_PER_STEP_ROTOR_ANGLE_RAD guard and never needed the fix at all).
_MAJOR1_REPRO_TRAP_OMEGA = 2.0e5
_MAJOR1_REPRO_K = 5.0e6
_MAJOR1_REPRO_MU = (1.0e-25, -2.0e-25, 1.5e-25)


def _major1_repro_config(
    output_dir: Path, *, dtau: float | None, steps: int | None
) -> PipelineConfig:
    integration: dict[str, object] = {}
    if dtau is not None:
        integration["dtau"] = dtau
        assert steps is not None
        integration["steps"] = steps
    else:
        integration["time_s"] = 1.0e-9
    return PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {
                "omega_xyz": [_MAJOR1_REPRO_TRAP_OMEGA] * 3,
                "center": [0.0, 0.0, 0.0],
            },
            "field": {"synthetic": {"kind": "quadrupole", "params": {"k": _MAJOR1_REPRO_K}}},
            "coupling": {"mu": list(_MAJOR1_REPRO_MU)},
            "ensemble": {
                "regime": "classical",
                "temperature_uK": 500.0,
                "size": 50,
                "seed": 0,
            },
            "integration": integration,
            "output": {"directory": str(output_dir)},
        }
    )


def test_wp8_major1_auto_dtau_realistic_mu_auto_tightens_not_silent_nan(tmp_path: Path) -> None:
    """WP8 review MAJOR 1 fix, auto-`dtau` half: the reviewer's reproduction
    scenario (realistic E14a `mu`, classical `direct` mode, `dtau` omitted
    so it is auto-selected) used to silently pass `_validate_physics`
    with `NaN` rotor-diagnostic fields (the primary scalar `phase` stayed
    finite, so only `phase` being checked let it through). It must now
    either auto-tighten `dtau` and produce a fully finite result, or raise
    `PhysicsValidationError` loudly -- never silently return a `NaN`-
    contaminated result. This run auto-tightens.
    """
    config = _major1_repro_config(tmp_path / "out", dtau=None, steps=None)
    assert config.integration.dtau is None  # dtau omitted -> auto-selected.

    result = run_pipeline_full(config)  # Must not raise; must not be silently NaN-contaminated.

    for name, arr in (
        ("phase", result.ensemble_result.phase),
        ("phase_rotor", result.ensemble_result.phase_rotor),
        ("r_final", result.ensemble_result.r_final),
        ("norm_error", result.ensemble_result.norm_error),
        ("max_norm_drift", result.ensemble_result.max_norm_drift),
    ):
        assert bool(jnp.all(jnp.isfinite(arr))), f"{name} contains non-finite values"

    assert np.isfinite(result.report.mean_fractional_shift)
    assert "integration.dtau auto-tightened" in result.report.uncertainty_notes
    assert "dtau_auto_selected=True" in result.report.uncertainty_notes
    # The auto-tightened dtau is strictly smaller than E31's naive auto-select.
    naive_dtau = fastpath.select_dtau(
        HarmonicTrap(omega_xyz=(_MAJOR1_REPRO_TRAP_OMEGA,) * 3),
        config.integration.points_per_period,
    )
    assert f"dtau={naive_dtau!r}" not in result.report.uncertainty_notes


def test_wp8_major1_explicit_dtau_realistic_mu_raises_loudly(tmp_path: Path) -> None:
    """WP8 review MAJOR 1 fix, explicit-`dtau` half: the same reproduction
    scenario, but with `integration.dtau` given explicitly (E31's naive
    auto-select value, so the same ~8863 rad per-step generator angle) --
    silently overriding an explicit user step size would be a worse
    surprise than failing loudly, so this must raise `PipelineConfigError`
    instead of auto-tightening.
    """
    naive_dtau = fastpath.select_dtau(
        HarmonicTrap(omega_xyz=(_MAJOR1_REPRO_TRAP_OMEGA,) * 3), fastpath.DEFAULT_POINTS_PER_PERIOD
    )
    config = _major1_repro_config(tmp_path / "out", dtau=naive_dtau, steps=5)
    assert config.integration.dtau == naive_dtau

    with pytest.raises(PipelineConfigError, match="exp_bivector"):
        run_pipeline_full(config)


def test_wp8_major1_lattice_worldline_auto_dtau_realistic_mu_auto_tightens(
    tmp_path: Path,
) -> None:
    """WP8 review MAJOR 1 fix, lattice `worldline`-mode half: the same
    auto-tightening (not silent NaN) behavior applies to
    `integration.mode: worldline` (the lattice E29-vs-E17 cross-check),
    not only classical `direct` mode -- both call the rotor integrator
    and both resolve `dtau` through the same guarded path
    (`_resolve_dtau_steps_worldline` / `_resolve_dtau_steps_direct`).
    A wider quadrature (`motional_n=(3,3,3)`, `n_quad=6`) is used so some
    nodes sample far enough from the trap center for the quadrupole field
    to drive the per-step angle past the guard at this project's
    realistic-mu regime.
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {
                "omega_xyz": [_MAJOR1_REPRO_TRAP_OMEGA] * 3,
                "center": [0.0, 0.0, 0.0],
            },
            "field": {"synthetic": {"kind": "quadrupole", "params": {"k": _MAJOR1_REPRO_K}}},
            "coupling": {"mu": list(_MAJOR1_REPRO_MU)},
            "ensemble": {
                "regime": "lattice",
                "temperature_uK": 1.0,
                "motional_n": [3, 3, 3],
                "n_quad": 6,
            },
            "integration": {"mode": "worldline", "time_s": 1.0e-9},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    assert config.integration.dtau is None

    result = run_pipeline_full(config)

    assert result.report.ensemble_type == "lattice_worldline_crosscheck"
    assert "integration.dtau auto-tightened" in result.report.uncertainty_notes
    for arr in (
        result.ensemble_result.phase,
        result.ensemble_result.phase_rotor,
        result.ensemble_result.r_final,
        result.ensemble_result.norm_error,
        result.ensemble_result.max_norm_drift,
    ):
        assert bool(jnp.all(jnp.isfinite(arr)))


def test_physics_validation_error_on_non_finite_norm_drift() -> None:
    """`_validate_physics` rejects a non-finite rotor-diagnostic field
    (`max_norm_drift`) even when the primary scalar `phase` stays finite
    -- the exact "NaN evasion" the WP8 review flagged (MAJOR 1a): the
    pre-fix `_validate_physics` only checked `phase` for finiteness, so
    `max_norm_drift=NaN` made the subsequent `max_norm_err >
    MAX_ROTOR_NORM_ERROR` comparison silently evaluate to `False` (NumPy:
    any comparison against NaN is `False`) instead of raising.
    """
    bad = EnsembleResult(
        r_final=jnp.zeros((2, 16)),
        phase=jnp.array([1.0e-18, 2.0e-18]),  # finite -- this is the point.
        phase_rotor=jnp.array([1.0e-18, 2.0e-18]),
        fractional_shift=jnp.array([1.0e-18, 2.0e-18]),
        norm_error=jnp.array([0.0, 0.0]),
        max_norm_drift=jnp.array([0.0, float("nan")]),
        n_steps=jnp.array([10, 10]),
    )
    with pytest.raises(PhysicsValidationError, match="non-finite"):
        _validate_physics(bad)


# ---------------------------------------------------------------------------
# WP8 review MAJOR 2 -- renorm_every auto-selection at auto-selected dtau.
# ---------------------------------------------------------------------------


def test_wp8_major2_auto_dtau_direct_mode_keeps_norm_drift_under_1e12(tmp_path: Path) -> None:
    """WP8 review MAJOR 2 fix: an auto-selected `dtau` (E31) at moderate
    parameters (no MAJOR 1 tightening triggered) also auto-tightens
    `renorm_every` (`DEFAULT_RENORM_EVERY=1000` was tuned for Compton-
    scale `dtau`, not E31-scale), so `max_norm_drift` stays under E20's
    `1e-12` sanity bound over a many-trap-period `direct`-mode run --
    unlike the un-tightened default cadence (docs/timescales.md's Tier
    B(i) finding).
    """
    omega = 2.0e5
    t_orbit_s = 2.0 * 3.141592653589793 / omega
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [omega, omega, omega], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
            "coupling": {"mu": [1.0e-30, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5, "seed": 2},
            "integration": {"time_s": 20.0 * t_orbit_s},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    assert config.integration.dtau is None
    assert not config.integration.renorm_every_was_explicit

    result = run_pipeline_full(config)

    assert "integration.renorm_every auto-selected" in result.report.uncertainty_notes
    max_drift = float(jnp.max(result.ensemble_result.max_norm_drift))
    assert max_drift < 1e-12, (
        f"max_norm_drift {max_drift!r} not < 1e-12 (E20) with auto renorm_every"
    )
    assert bool(jnp.all(jnp.isfinite(result.ensemble_result.phase)))


def test_wp8_major2_auto_dtau_worldline_mode_keeps_norm_drift_under_1e12(tmp_path: Path) -> None:
    """WP8 review MAJOR 2 fix, lattice `worldline`-mode half: the same
    auto-`renorm_every` behavior applies to `integration.mode: worldline`
    (also a rotor-stepping, `exp_bivector`-calling mode) when its `dtau`
    is auto-selected, not only to classical `direct` mode.
    """
    omega = 2.0e5
    t_orbit_s = 2.0 * 3.141592653589793 / omega
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [omega, omega, omega], "center": [0.0, 0.0, 0.0]},
            "field": {
                "synthetic": {
                    "kind": "gaussian_bump",
                    "params": {
                        "amplitude": [10.0, 0.0, -5.0],
                        "center": [0.0, 0.0, 0.0],
                        "width": 1.0e-6,
                    },
                }
            },
            "coupling": {"mu": [1.0e-30, 0.0, 0.0]},
            "ensemble": {
                "regime": "lattice",
                "temperature_uK": 1.0,
                "motional_n": [0, 0, 0],
                "n_quad": 3,
            },
            "integration": {"mode": "worldline", "time_s": 10.0 * t_orbit_s},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    assert config.integration.dtau is None
    assert not config.integration.renorm_every_was_explicit

    result = run_pipeline_full(config)

    assert result.report.ensemble_type == "lattice_worldline_crosscheck"
    assert "integration.renorm_every auto-selected" in result.report.uncertainty_notes
    max_drift = float(jnp.max(result.ensemble_result.max_norm_drift))
    assert max_drift < 1e-12, (
        f"max_norm_drift {max_drift!r} not < 1e-12 (E20) with auto renorm_every"
    )


def test_wp8_major2_explicit_renorm_every_always_honored(tmp_path: Path) -> None:
    """An explicit `integration.renorm_every` is always honored unchanged,
    even when `dtau` is auto-selected -- the auto-tightening in
    :func:`cliffordclock.pipeline._resolve_renorm_every` only applies when
    the caller left `renorm_every` at its YAML default.
    """
    omega = 2.0e5
    t_orbit_s = 2.0 * 3.141592653589793 / omega
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [omega, omega, omega], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
            "coupling": {"mu": [1.0e-30, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5, "seed": 2},
            "integration": {"time_s": 3.0 * t_orbit_s, "renorm_every": 1000},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    assert config.integration.dtau is None
    assert config.integration.renorm_every_was_explicit

    result = run_pipeline_full(config)

    assert "renorm_every=1000" in result.report.uncertainty_notes
    assert "integration.renorm_every auto-selected" not in result.report.uncertainty_notes


# ---------------------------------------------------------------------------
# Pre-flight trajectory-memory guard (pipeline._check_trajectory_memory) and
# WP19's streaming/chunked evaluation, which turns this guard into an
# advisory *selector* rather than a hard wall for ensemble.regime='classical'
# + integration.mode='direct' (both coupling.type values).
#
# The direct/worldline modes materialize dense (M, steps+1, 3) float64
# trajectories; an auto-selected E31 dtau (trap-period resolution) over a
# long time_s can push steps to 1e8+, and combined with M in the hundreds-
# to-thousands that is a silent multi-TB allocation attempt that can lock
# up the host. Pre-WP19, the guard rejected such configs outright
# (PipelineConfigError). WP19: for classical+direct under
# integration.evaluation="auto" (the default), the SAME estimate now
# switches to an O(M)-memory streaming accumulator instead of raising --
# the "pathological" configs below are genuinely runnable now (in bounded
# memory), but this test file still never actually executes them at their
# full ~1e8-step scale (compute time, not memory, would then be the
# limiting factor -- a lax.scan with ~2e8 iterations is not a multi-second
# test even though it is O(M) in memory). Instead:
#   - the ORIGINAL "pathological" dicts are used with
#     integration.evaluation="batched" forced explicitly, preserving the
#     exact pre-WP19 hard-reject behavior/test as a regression check for
#     that explicit-request path;
#   - the dispatch DECISION for the unmodified pathological (M, steps) is
#     checked arithmetically, calling `_estimate_trajectory_memory_gb`/
#     `_resolve_evaluation_mode` directly -- no trajectory or accumulator
#     code runs at all;
#   - separate, deliberately small/fast "miniature analog" configs (an
#     artificially tiny `max_trajectory_memory_gb`, not huge M/steps) drive
#     a REAL default-`evaluation="auto"` run through the actual streaming
#     accumulator end to end, verifying it dispatches, runs, and reports a
#     finite, sane result with the expected uncertainty_notes text.
# ---------------------------------------------------------------------------


def _memory_guard_pathological_dict(output_dir: Path) -> dict[str, object]:
    """Classical-direct config resolving to ~1.9e8 steps x M=1000 (~18 TB
    estimated): omega=2e5 rad/s gives an auto-selected dt of
    T_orb/100 ~ 3.1e-7 s, so time_s=60 s resolves to ~1.9e8 steps. The tiny
    `mu` (1e-30, same as the WP8 auto-dtau test) keeps the generator-angle
    guard disengaged so the memory guard is what trips.
    """
    return {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
        "coupling": {"mu": [1.0e-30, 0.0, 0.0]},
        "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 1000, "seed": 0},
        "integration": {"time_s": 60.0},
        "output": {"directory": str(output_dir)},
    }


def test_memory_guard_trips_on_pathological_classical_direct_batched_forced(
    tmp_path: Path,
) -> None:
    """`integration.evaluation="batched"` forced explicitly on the pathological
    classical `direct`-mode config: `PipelineConfigError` before any
    trajectory is allocated, message actionable -- the pre-WP19 hard-reject
    behavior, preserved unchanged when batched is requested explicitly (WP19
    only changes the *default*, `evaluation="auto"`; see
    `test_memory_guard_auto_dispatch_arithmetic_pathological_classical_direct`
    for what `"auto"` now does with this exact config, checked without
    running it).
    """
    data = _memory_guard_pathological_dict(tmp_path / "out")
    data["integration"]["evaluation"] = "batched"
    config = PipelineConfig.from_dict(data)
    assert config.integration.max_trajectory_memory_gb == DEFAULT_MAX_TRAJECTORY_MEMORY_GB

    with pytest.raises(PipelineConfigError, match="max_trajectory_memory_gb") as excinfo:
        run_pipeline_full(config)

    message = str(excinfo.value)
    assert "ensemble.size" in message
    assert "secular" in message
    assert "fast_path" in message


def test_memory_guard_trips_on_pathological_stark_scalar_direct_batched_forced(
    tmp_path: Path,
) -> None:
    """The stark_dc classical-`direct` branch, `integration.evaluation="batched"`
    forced: same hard-reject behavior as the `linear_mu` case above,
    preserved unchanged for an explicit batched request (WP19).
    """
    data = _memory_guard_pathological_dict(tmp_path / "out")
    data["coupling"] = {"type": "stark_dc"}
    data["integration"]["evaluation"] = "batched"
    config = PipelineConfig.from_dict(data)

    with pytest.raises(PipelineConfigError, match="max_trajectory_memory_gb"):
        run_pipeline_full(config)


def test_memory_guard_auto_dispatch_arithmetic_pathological_classical_direct(
    tmp_path: Path,
) -> None:
    """WP19 acceptance criterion: the *dispatch decision* for the original
    ~1.9e8-step / M=1000 (~18 TB estimated) incident config is checked
    arithmetically -- `_estimate_trajectory_memory_gb`/`_resolve_evaluation_mode`
    directly, reproducing exactly the `(n_atoms, steps)` `run_pipeline_full`
    would resolve for this config -- without ever running the trajectory or
    accumulator (never executing a ~2e8-iteration scan in a test; see this
    section's header comment). Confirms `integration.evaluation="auto"`
    (the default) would switch this exact incident-scale config to the
    streaming accumulator instead of raising.
    """
    data = _memory_guard_pathological_dict(tmp_path / "out")
    config = PipelineConfig.from_dict(data)
    assert config.integration.evaluation == "auto"

    # Reproduce run_pipeline_full's own steps resolution (E31 auto-dtau,
    # time_s-driven) without running anything.
    trap = HarmonicTrap(omega_xyz=config.trap.omega_xyz, center=config.trap.center)
    dtau = fastpath.select_dtau(trap, config.integration.points_per_period)
    steps = max(1, round(config.integration.time_s / (dtau * TAU_COMPTON)))
    assert steps > 1.0e8, f"fixture drifted: expected ~1.9e8 steps, got {steps}"

    _traj_gb, _smoother_gb, estimated_gb = _estimate_trajectory_memory_gb(1000, steps)
    assert estimated_gb > 1000.0  # comfortably into the "multi-TB" regime the docstring claims

    mode, note = _resolve_evaluation_mode(
        config.integration.evaluation, estimated_gb, config.integration.max_trajectory_memory_gb
    )
    assert mode == "streaming"
    assert note is not None
    assert "switched to streaming evaluation (memory-bounded)" in note


def test_memory_guard_auto_dispatch_arithmetic_pathological_stark_scalar_direct(
    tmp_path: Path,
) -> None:
    """Same arithmetic-only dispatch check as
    `test_memory_guard_auto_dispatch_arithmetic_pathological_classical_direct`,
    for the `coupling.type='stark_dc'` cell (`_resolve_dtau_steps_scalar`'s
    call site) -- also resolves to `"streaming"` under the default `"auto"`.
    """
    data = _memory_guard_pathological_dict(tmp_path / "out")
    data["coupling"] = {"type": "stark_dc"}
    config = PipelineConfig.from_dict(data)

    trap = HarmonicTrap(omega_xyz=config.trap.omega_xyz, center=config.trap.center)
    dtau = fastpath.select_dtau(trap, config.integration.points_per_period)
    steps = max(1, round(config.integration.time_s / (dtau * TAU_COMPTON)))

    _traj_gb, _smoother_gb, estimated_gb = _estimate_trajectory_memory_gb(1000, steps)
    mode, note = _resolve_evaluation_mode(
        config.integration.evaluation, estimated_gb, config.integration.max_trajectory_memory_gb
    )
    assert mode == "streaming"
    assert note is not None


def _tiny_budget_analog_dict(output_dir: Path, *, coupling: dict[str, object]) -> dict[str, object]:
    """A small, fast (< 1 s) classical-`direct` config with an artificially
    tiny `max_trajectory_memory_gb` (not a huge M/steps) -- forces the same
    `estimated_gb > max_gb` auto-dispatch condition the real incident
    configs trip, but cheaply, so this test actually executes the streaming
    accumulator end to end (WP19 acceptance criterion: "a real moderate run
    with RSS sanity margin", separate from the arithmetic-only checks
    above). ``dtau``/``steps`` explicit (Compton-scale) rather than
    `time_s`-driven, so this stays fast regardless of the trap/dtau
    auto-selection machinery.

    ``max_trajectory_memory_gb=1e-4``: chosen (for M=20, steps=300) to sit
    strictly between the batched estimate (~5.8e-4 GB, so it trips
    auto-dispatch) and the streaming path's own much smaller O(M) estimate
    (~2.0e-5 GB, so :func:`~cliffordclock.pipeline._check_streaming_memory`
    does *not* also trip on the same artificially tight budget -- that
    would defeat the point of this fixture).
    """
    return {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
        "coupling": coupling,
        "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 20, "seed": 1},
        "integration": {"dtau": 0.5, "steps": 300, "max_trajectory_memory_gb": 1.0e-4},
        "output": {"directory": str(output_dir)},
    }


def test_memory_guard_auto_runs_streaming_miniature_analog_linear_mu(tmp_path: Path) -> None:
    """A real (not arithmetic-only) `evaluation="auto"` run of a small,
    fast, `linear_mu` classical-`direct` config whose artificially tiny
    `max_trajectory_memory_gb` forces the same auto-dispatch-to-streaming
    path the real incident configs would take -- exercises
    `_direct_rotor_ensemble_streaming` end to end (not just the dispatch
    arithmetic) and checks the result is sane and the dispatch note lands
    in `uncertainty_notes`.
    """
    data = _tiny_budget_analog_dict(tmp_path / "out", coupling={"mu": [1.0e-25, 0.0, 0.0]})
    config = PipelineConfig.from_dict(data)
    assert config.integration.evaluation == "auto"

    result = run_pipeline_full(config)

    assert np.isfinite(result.report.mean_fractional_shift)
    assert "switched to streaming evaluation (memory-bounded)" in result.report.uncertainty_notes
    assert "evaluation=streaming" in result.report.uncertainty_notes
    assert result.trajectories.shape == (20, 2, 3)  # no trajectory_stride: initial+final only


def test_memory_guard_auto_runs_streaming_miniature_analog_stark_dc(tmp_path: Path) -> None:
    """Same as `test_memory_guard_auto_runs_streaming_miniature_analog_linear_mu`,
    `coupling.type='stark_dc'` (`_stark_scalar_ensemble_streaming`).
    """
    data = _tiny_budget_analog_dict(tmp_path / "out", coupling={"type": "stark_dc"})
    config = PipelineConfig.from_dict(data)

    result = run_pipeline_full(config)

    assert np.isfinite(result.report.mean_fractional_shift)
    assert "switched to streaming evaluation (memory-bounded)" in result.report.uncertainty_notes
    assert result.trajectories.shape == (20, 2, 3)


def test_memory_guard_trips_on_pathological_lattice_worldline(tmp_path: Path) -> None:
    """The lattice `worldline` cross-check mode broadcasts its static nodes
    to a dense (M, steps+1, 3) trajectory, so it has the same runaway-
    allocation failure mode and the same guard (via
    `_resolve_dtau_steps_worldline`). M = n_quad^3 = 512 nodes here.
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
            "coupling": {"mu": [1.0e-30, 0.0, 0.0]},
            "ensemble": {"regime": "lattice", "temperature_uK": 1.0, "motional_n": [0, 0, 0]},
            "integration": {"mode": "worldline", "time_s": 60.0},
            "output": {"directory": str(tmp_path / "out")},
        }
    )

    with pytest.raises(PipelineConfigError, match="max_trajectory_memory_gb"):
        run_pipeline_full(config)


def test_memory_guard_normal_config_unaffected(tmp_path: Path) -> None:
    """A realistic small config (the WP8 auto-dtau scenario: M=5, ~300
    steps, ~100 kB estimated) runs to completion unchanged under the
    default 2 GB limit -- the guard exists for pathological configs only.
    """
    omega = 2.0e5
    t_orbit_s = 2.0 * 3.141592653589793 / omega
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [omega, omega, omega], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
            "coupling": {"mu": [1.0e-30, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5, "seed": 2},
            "integration": {"time_s": 3.0 * t_orbit_s},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    assert config.integration.max_trajectory_memory_gb == DEFAULT_MAX_TRAJECTORY_MEMORY_GB

    result = run_pipeline_full(config)

    assert np.isfinite(result.report.mean_fractional_shift)


def test_memory_guard_override_honored_both_directions(tmp_path: Path) -> None:
    """`integration.max_trajectory_memory_gb` is honored: a small config
    (M=5, 2000 explicit steps, ~1 MB estimated) trips an artificially tiny
    limit, and the identical config with an ample explicit limit runs --
    demonstrating the override path without ever attempting a large
    allocation.

    WP19 note: under the default `evaluation="auto"`, the tiny-limit case
    below no longer raises via the *batched*-path guard
    (`_check_trajectory_memory`) -- it auto-dispatches to streaming first
    (batched estimate ~9.6e-4 GB > the 1e-6 GB limit), and *that* is what
    raises instead: `_check_streaming_memory`'s much smaller O(M) estimate
    (~5.1e-6 GB) still exceeds this deliberately absurd 1e-6 GB limit. The
    net behavior this test asserts (`max_trajectory_memory_gb` this tiny
    is honored, i.e. still rejected) is unchanged; a separate case with
    `evaluation="batched"` forced confirms the original batched-path
    guard itself still rejects the identical config directly (not merely
    via the streaming fallback), preserving that regression coverage too.
    """
    base: dict[str, object] = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
        "coupling": {"mu": [1.0e-30, 0.0, 0.0]},
        "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5, "seed": 2},
        "output": {"directory": str(tmp_path / "out")},
    }
    integration = {"dtau": 0.5, "steps": 2000}

    tight = PipelineConfig.from_dict(
        {**base, "integration": {**integration, "max_trajectory_memory_gb": 1.0e-6}}
    )
    with pytest.raises(PipelineConfigError, match="max_trajectory_memory_gb"):
        run_pipeline_full(tight)

    tight_batched_forced = PipelineConfig.from_dict(
        {
            **base,
            "integration": {
                **integration,
                "max_trajectory_memory_gb": 1.0e-6,
                "evaluation": "batched",
            },
        }
    )
    with pytest.raises(PipelineConfigError, match="max_trajectory_memory_gb") as excinfo:
        run_pipeline_full(tight_batched_forced)
    assert "streaming" not in str(
        excinfo.value
    )  # the original batched-path guard, not the fallback

    ample = PipelineConfig.from_dict(
        {**base, "integration": {**integration, "max_trajectory_memory_gb": 1.0}}
    )
    result = run_pipeline_full(ample)
    assert np.isfinite(result.report.mean_fractional_shift)


def test_memory_guard_rejects_non_positive_limit() -> None:
    """`integration.max_trajectory_memory_gb` must be positive; zero or
    negative values are rejected at config-parse time.
    """
    for bad in (0.0, -1.0):
        with pytest.raises(PipelineConfigError, match="max_trajectory_memory_gb"):
            PipelineConfig.from_dict(
                {
                    "species": "Sr87",
                    "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
                    "field": {
                        "synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}
                    },
                    "coupling": {"mu": [1.0e-30, 0.0, 0.0]},
                    "ensemble": {
                        "regime": "classical",
                        "temperature_uK": 1.0,
                        "size": 5,
                        "seed": 0,
                    },
                    "integration": {"time_s": 1.0, "max_trajectory_memory_gb": bad},
                    "output": {"directory": "."},
                }
            )


def _minimal_integration_config_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
        "coupling": {"mu": [1.0e-30, 0.0, 0.0]},
        "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5, "seed": 0},
        "integration": {"time_s": 1.0, **overrides},
        "output": {"directory": "."},
    }
    return base


def test_integration_evaluation_defaults_to_auto() -> None:
    """`integration.evaluation` defaults to `"auto"` when omitted (WP19),
    matching `VALID_EVALUATION_MODES[0]`.
    """
    config = PipelineConfig.from_dict(_minimal_integration_config_dict())
    assert config.integration.evaluation == "auto"
    assert VALID_EVALUATION_MODES == ("auto", "batched", "streaming")


@pytest.mark.parametrize("mode", ["auto", "batched", "streaming"])
def test_integration_evaluation_accepts_every_valid_mode(mode: str) -> None:
    config = PipelineConfig.from_dict(_minimal_integration_config_dict(evaluation=mode))
    assert config.integration.evaluation == mode


def test_integration_evaluation_rejects_invalid_value() -> None:
    with pytest.raises(PipelineConfigError, match="integration.evaluation"):
        PipelineConfig.from_dict(_minimal_integration_config_dict(evaluation="parallel"))


def test_integration_trajectory_stride_defaults_to_none() -> None:
    config = PipelineConfig.from_dict(_minimal_integration_config_dict())
    assert config.integration.trajectory_stride is None


def test_integration_trajectory_stride_accepts_positive_int() -> None:
    config = PipelineConfig.from_dict(_minimal_integration_config_dict(trajectory_stride=50))
    assert config.integration.trajectory_stride == 50


def test_integration_trajectory_stride_rejects_non_positive() -> None:
    for bad in (0, -5):
        with pytest.raises(PipelineConfigError, match="trajectory_stride"):
            PipelineConfig.from_dict(_minimal_integration_config_dict(trajectory_stride=bad))


def test_memory_guard_trips_via_smoother_evaluation_term_stark_scalar_batched_forced(
    tmp_path: Path,
) -> None:
    """Review fix (MAJOR): the base trajectory-only estimate misses the
    dominant cost of `_stark_scalar_ensemble`'s `run_one`, which calls
    `rate_fn` **once** on an atom's entire `(steps, 3)` midpoint
    trajectory (`vmap`-ed over `M`) rather than once per step -- so
    against a `FieldSmoother`-backed field (`field.csv`/`field.comsol`),
    the smoother's internal `(N, K, 3)` intermediates (`N = M x steps`
    query points, `K` RBF fit points), tripled again by `jax.jacfwd` for
    `grad_E`, dominate real peak RSS even when the base
    `4 x M x (steps+1) x 3 x 8` bytes trajectory term is trivially small.

    This config (M=100, steps=2000, K=216 from a 6x6x6 CSV grid) is a
    guard-passing-but-lethal case for the pre-fix guard: the base term
    alone is ~0.02 GB (comfortably under the default 2 GB limit), but the
    real allocation (calibrated from measured RSS,
    :data:`cliffordclock.pipeline._TRAJECTORY_MEMORY_FACTOR_SMOOTHER`) is
    over 10 GB. `integration.evaluation="batched"` is forced explicitly
    (WP19: under the default `"auto"`, this same estimate now dispatches
    to the streaming accumulator instead of raising -- see
    ``test_memory_guard_auto_streaming_moderate_run_rss_sanity`` below for
    that path exercised for real, with an RSS measurement) so this test
    keeps covering the pre-WP19 hard-reject behavior, still available on
    explicit request: the guard must trip on this config, in milliseconds,
    with no trajectory or smoother-evaluation work actually performed.
    """
    csv_path = tmp_path / "field.csv"
    k = 500.0
    rows = ["x,y,z,Ex,Ey,Ez"]
    axis = np.linspace(-0.01, 0.01, 6)  # 6^3 = 216 RBF fit points.
    for x in axis:
        for y in axis:
            for z in axis:
                rows.append(f"{x},{y},{z},{k * x},{k * y},{-2.0 * k * z}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"csv": str(csv_path)},
            "coupling": {"type": "stark_dc"},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 100, "seed": 3},
            "integration": {"dtau": 0.5, "steps": 2000, "evaluation": "batched"},
            "output": {"directory": str(tmp_path / "out")},
        }
    )

    start = time.perf_counter()
    with pytest.raises(PipelineConfigError, match="max_trajectory_memory_gb") as excinfo:
        run_pipeline_full(config)
    elapsed_s = time.perf_counter() - start
    assert elapsed_s < 30.0, (
        f"guard should reject before any trajectory/smoother-evaluation work: {elapsed_s:.2f}s"
    )

    message = str(excinfo.value)
    assert "K=216" in message
    assert "smoother" in message.lower()


def test_memory_guard_smoother_term_not_applied_to_synthetic_field(tmp_path: Path) -> None:
    """The exact same (M, steps) that trips the smoother-augmented guard
    above (:func:`test_memory_guard_trips_via_smoother_evaluation_term_stark_scalar`)
    runs to completion for a closed-form `field.synthetic` field instead:
    no `FieldSmoother`, hence no `K`, so only the base trajectory-only
    term applies (well under the default 2 GB limit) -- confirming the
    smoother-evaluation term is additive on top of the base guard for
    smoother-backed fields specifically, not a general tightening that
    would also affect synthetic fields.
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
            "coupling": {"type": "stark_dc"},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 100, "seed": 3},
            "integration": {"dtau": 0.5, "steps": 2000},
            "output": {"directory": str(tmp_path / "out")},
        }
    )

    result = run_pipeline_full(config)

    assert np.isfinite(result.report.mean_fractional_shift)


def test_memory_guard_trips_on_lattice_worldline_stark_dc(tmp_path: Path) -> None:
    """Review fix (MAJOR): missing regression coverage for the lattice
    `worldline` + `coupling.type='stark_dc'` branch
    (`run_pipeline_full`'s `mode == "worldline" and
    config.coupling.type == "stark_dc"` cell, which builds `traj_dense`
    and calls :func:`cliffordclock.pipeline._stark_rotor_ensemble` -- a
    different call site than
    :func:`test_memory_guard_trips_on_pathological_lattice_worldline`
    above, which only exercises the `linear_mu` cell of the same
    `mode="worldline"` dispatch). It shares the guarded resolver
    (`_resolve_dtau_steps_worldline` -> `_resolve_dtau_steps_direct`) with
    that cell, so the guard already trips here too, but nothing
    previously pinned that at this specific call site -- a regression at
    just this cell could otherwise ship silently.

    `time_s=60.0` at this trap resolves to an estimated multi-TB
    `traj_dense` allocation for `M=512` (`n_quad=8` default -> `8**3`
    quadrature nodes); the guard must reject it in a fraction of a
    second, before `traj_dense` is ever built.
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
            "coupling": {"type": "stark_dc"},
            "ensemble": {"regime": "lattice", "temperature_uK": 1.0, "motional_n": [0, 0, 0]},
            "integration": {"mode": "worldline", "time_s": 60.0},
            "output": {"directory": str(tmp_path / "out")},
        }
    )

    start = time.perf_counter()
    with pytest.raises(PipelineConfigError, match="max_trajectory_memory_gb") as excinfo:
        run_pipeline_full(config)
    elapsed_s = time.perf_counter() - start
    assert elapsed_s < 30.0, f"guard should reject before traj_dense is built: {elapsed_s:.2f}s"

    message = str(excinfo.value)
    assert "ensemble.size" in message


# ---------------------------------------------------------------------------
# Secular-mode trajectory-memory guard (MINOR review fix): points_per_period
# has no upper bound, and secular_average_shift_ensemble (fastpath.py)
# vmaps a (points_per_period + 1, 3) closed-form-orbit trajectory over M
# atoms -- the same runaway-allocation failure mode as direct/worldline,
# just keyed on points_per_period instead of a resolved step count.
# fastpath.py itself is out of scope (module docstring interface note 4);
# the check is applied at the pipeline call site, before dispatch.
# ---------------------------------------------------------------------------


def test_memory_guard_trips_on_pathological_secular_points_per_period(tmp_path: Path) -> None:
    """An absurdly large `integration.points_per_period` under
    `mode='secular'` is rejected by the same `_check_trajectory_memory`
    estimate (`M x (points_per_period + 1)`), applied at the
    `run_pipeline_full` call site before
    `fastpath.secular_average_shift_ensemble` is dispatched -- in a
    fraction of a second, never attempting the underlying
    `(points_per_period + 1, 3)`-per-atom orbit trajectory.
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
            "coupling": {"mu": [1.0e-30, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 1000, "seed": 0},
            "integration": {
                "mode": "secular",
                "time_s": 1.0,
                "points_per_period": 100_000_000_000,
            },
            "output": {"directory": str(tmp_path / "out")},
        }
    )

    start = time.perf_counter()
    with pytest.raises(PipelineConfigError, match="max_trajectory_memory_gb") as excinfo:
        run_pipeline_full(config)
    elapsed_s = time.perf_counter() - start
    assert elapsed_s < 30.0, (
        f"guard should reject before any orbit trajectory work: {elapsed_s:.2f}s"
    )

    message = str(excinfo.value)
    assert "ensemble.size" in message


def test_memory_guard_normal_secular_config_unaffected(tmp_path: Path) -> None:
    """A realistic `mode='secular'` config at the default
    `points_per_period=100` (E31's default `N_res`) runs to completion
    unchanged -- the secular guard added above only rejects pathological
    `points_per_period` values, not ordinary secular runs.
    """
    omega = 2.0e5
    t_orbit_s = 2.0 * math.pi / omega
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [omega, omega, omega], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
            "coupling": {"mu": [1.0e-24, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5, "seed": 2},
            "integration": {"mode": "secular", "time_s": 5.0 * t_orbit_s},
            "output": {"directory": str(tmp_path / "out")},
        }
    )

    result = run_pipeline_full(config)

    assert np.isfinite(result.report.mean_fractional_shift)
    assert "integration.mode=secular" in result.report.uncertainty_notes


# ---------------------------------------------------------------------------
# WP19 -- streaming (O(M)-memory) accumulators: streaming ≡ batched agreement,
# and the showcase example run via forced streaming without its own
# max_trajectory_memory_gb override (the acceptance-criterion "runs to
# completion in bounded memory" case, with a measured RSS margin).
# ---------------------------------------------------------------------------


def _streaming_agreement_result_pair(
    tmp_path: Path, *, coupling: dict[str, object], field: dict[str, object]
) -> tuple[PipelineResult, PipelineResult]:
    """Run an identical moderate classical-`direct` config once batched, once
    streaming, and return `(batched_result, streaming_result)`.
    """
    integration: dict[str, object] = {"dtau": 0.5, "steps": 500}
    base: dict[str, object] = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": field,
        "coupling": coupling,
        "ensemble": {"regime": "classical", "temperature_uK": 5.0, "size": 40, "seed": 7},
        "output": {"directory": str(tmp_path / "out")},
    }
    batched = run_pipeline_full(
        PipelineConfig.from_dict({**base, "integration": {**integration, "evaluation": "batched"}})
    )
    streaming = run_pipeline_full(
        PipelineConfig.from_dict(
            {**base, "integration": {**integration, "evaluation": "streaming"}}
        )
    )
    return batched, streaming


def test_streaming_matches_batched_linear_mu_direct(tmp_path: Path) -> None:
    """Streaming ≡ batched agreement (WP19 acceptance criterion), `linear_mu`
    + `mode='direct'` (the rotor path, `_direct_rotor_ensemble_streaming` vs.
    `worldline.integrate_ensemble`), a moderate config (M=40, 500 steps, a
    spatially-varying closed-form field so the boost/gradient terms are
    genuinely exercised, not just the uniform-field null case).

    Measured on this exact config: bitwise-identical per-atom `phase` (the
    fused streaming scan processes steps in the same order, with the same
    per-step `rotor_step`/Kahan-sum calls, as the batched
    dense-trajectory-then-scan path -- Kahan summation's whole purpose is
    making that reordering not matter, and here it turns out identical to
    the last bit). Documented bound: `rtol=1e-12, atol=0` (far looser than
    the measured 0.0, in this project's usual "tight, not vacuous" style
    for an algebraic-identity claim -- see ``docs/timescales.md``'s Tier A
    ≡ Tier C precedent for the same `rtol=1e-12` choice).
    """
    field = {
        "synthetic": {
            "kind": "gaussian_bump",
            "params": {"amplitude": [0.0, 0.0, 500.0], "center": [0.0, 0.0, 0.0], "width": 1.0e-4},
        }
    }
    batched, streaming = _streaming_agreement_result_pair(
        tmp_path, coupling={"mu": [1.0e-25, 2.0e-25, -1.0e-25]}, field=field
    )

    # Positive confirmation each side ran its own code path (not a silent
    # fallthrough): streaming keeps initial+final positions only, batched
    # keeps the dense (steps+1) trajectory.
    assert streaming.trajectories.shape == (40, 2, 3)
    assert batched.trajectories.shape == (40, 501, 3)

    phi_b = np.asarray(batched.ensemble_result.phase)
    phi_s = np.asarray(streaming.ensemble_result.phase)
    np.testing.assert_allclose(phi_b, phi_s, rtol=1e-12, atol=0)
    assert np.isfinite(phi_b).all() and np.any(phi_b != 0.0)  # a non-trivial result

    np.testing.assert_allclose(
        batched.report.mean_fractional_shift,
        streaming.report.mean_fractional_shift,
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        batched.report.shift_std_error,
        streaming.report.shift_std_error,
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(batched.ensemble_result.phase_rotor),
        np.asarray(streaming.ensemble_result.phase_rotor),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(batched.ensemble_result.r_final),
        np.asarray(streaming.ensemble_result.r_final),
        rtol=1e-9,
        atol=1e-30,
    )


def test_streaming_matches_batched_stark_dc_direct_smoother_field(tmp_path: Path) -> None:
    """Streaming ≡ batched agreement, `coupling.type='stark_dc'` + `mode='direct'`
    (`_stark_scalar_ensemble_streaming` vs. `_stark_scalar_ensemble`) against
    a real `FieldSmoother`-backed field (`field.csv`) -- the exact failure
    mode this WP exists to fix (the batched path's single whole-trajectory
    `rate_fn` call, `_TRAJECTORY_MEMORY_FACTOR_SMOOTHER`'s comment).

    Unlike the `linear_mu` rotor case above, this is *not* measured
    bitwise-identical: the batched path evaluates `FieldSmoother.evaluate`
    once per atom on its whole `(steps, 3)` trajectory (one large `vmap`
    batch), while streaming evaluates it once per step on an `(M, 3)`
    batch -- a different `jax.vmap` batching structure over the same
    underlying RBF evaluation, which JAX/XLA is not obligated to lower
    identically bit-for-bit (see
    ``tests/test_fields_smoother.py``'s chunk_size=1 vs. >=2 ulp-level
    finding for the same underlying cause). Measured on this exact config:
    max relative difference ~1.7e-16 (machine-epsilon level, not a
    physical discrepancy). `rtol=1e-10` here is a documented, deliberately
    loose (~1e6x margin over the measured value) tight bound, not a
    vacuous one.
    """
    csv_path = tmp_path / "field.csv"
    k = 500.0
    rows = ["x,y,z,Ex,Ey,Ez"]
    axis = np.linspace(-0.01, 0.01, 6)  # 6^3 = 216 RBF fit points.
    for x in axis:
        for y in axis:
            for z in axis:
                rows.append(f"{x},{y},{z},{k * x},{k * y},{-2.0 * k * z}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    batched, streaming = _streaming_agreement_result_pair(
        tmp_path, coupling={"type": "stark_dc"}, field={"csv": str(csv_path)}
    )

    # Positive confirmation each side ran its own code path (see the
    # linear_mu agreement test above for the shape rationale).
    assert streaming.trajectories.shape == (40, 2, 3)
    assert batched.trajectories.shape == (40, 501, 3)

    phi_b = np.asarray(batched.ensemble_result.phase)
    phi_s = np.asarray(streaming.ensemble_result.phase)
    np.testing.assert_allclose(phi_b, phi_s, rtol=1e-10, atol=0)
    assert np.isfinite(phi_b).all() and np.any(phi_b != 0.0)

    np.testing.assert_allclose(
        batched.report.mean_fractional_shift,
        streaming.report.mean_fractional_shift,
        rtol=1e-10,
        atol=0,
    )
    np.testing.assert_allclose(
        batched.report.shift_std_error,
        streaming.report.shift_std_error,
        rtol=1e-10,
        atol=0,
    )


def test_streaming_trajectory_stride_produces_subsampled_positions(tmp_path: Path) -> None:
    """`integration.trajectory_stride` retains a periodic position snapshot
    instead of the streaming default (initial + final position only) --
    exercises `_run_streaming_scan`'s block-loop subsampling path (WP19).
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
            "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 6, "seed": 4},
            "integration": {
                "dtau": 0.5,
                "steps": 100,
                "evaluation": "streaming",
                "trajectory_stride": 25,
            },
            "output": {"directory": str(tmp_path / "out")},
        }
    )

    result = run_pipeline_full(config)

    # 100 steps at stride 25 -> 4 blocks -> initial + 4 snapshots = 5 samples.
    assert result.trajectories.shape == (6, 5, 3)
    assert np.isfinite(np.asarray(result.trajectories)).all()

    # The final snapshot must equal the position a full (non-strided) run's
    # trajectory would end at -- cross-check against evaluation="batched"
    # on the identical config (same seed/dtau/steps): batched keeps the
    # full dense trajectory, so its last sample is the ground truth.
    batched_config = replace(config, integration=replace(config.integration, evaluation="batched"))
    batched_result = run_pipeline_full(batched_config)
    np.testing.assert_allclose(
        np.asarray(result.trajectories)[:, -1, :],
        np.asarray(batched_result.trajectories)[:, -1, :],
        rtol=1e-12,
        atol=0,
    )


def test_streaming_trajectory_stride_non_divisor_final_short_block(tmp_path: Path) -> None:
    """`_run_streaming_scan`'s final SHORT block: `steps` not an exact
    multiple of `trajectory_stride` (100 steps at stride 30 -> blocks of
    30/30/30/10). The divisible case is covered above; this pins the
    partial-final-block arithmetic (WP19 review nit).
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 100.0]}}},
            "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 6, "seed": 4},
            "integration": {
                "dtau": 0.5,
                "steps": 100,
                "evaluation": "streaming",
                "trajectory_stride": 30,
            },
            "output": {"directory": str(tmp_path / "out")},
        }
    )

    result = run_pipeline_full(config)

    # ceil(100 / 30) = 4 blocks (30 + 30 + 30 + 10) -> initial + 4 = 5 samples.
    assert result.trajectories.shape == (6, 5, 3)
    assert np.isfinite(np.asarray(result.trajectories)).all()

    # The short final block must land on the same endpoint (and phases) as
    # an unstrided streaming run of the identical config.
    unstrided_config = replace(
        config, integration=replace(config.integration, trajectory_stride=None)
    )
    unstrided_result = run_pipeline_full(unstrided_config)
    np.testing.assert_allclose(
        np.asarray(result.trajectories)[:, -1, :],
        np.asarray(unstrided_result.trajectories)[:, -1, :],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(result.ensemble_result.phase),
        np.asarray(unstrided_result.ensemble_result.phase),
        rtol=0,
        atol=0,
    )


def test_showcase_forced_streaming_without_override_matches_batched_and_bounds_rss(
    tmp_path: Path,
) -> None:
    """WP19 acceptance criterion: the showcase example
    (``examples/showcase_gradient_dispersion_sr87.yaml``) runs via FORCED
    streaming (`integration.evaluation="streaming"`) with its own
    ``max_trajectory_memory_gb: 8.0`` override REMOVED (falling back to the
    2 GB default -- irrelevant to streaming's dispatch anyway, since
    `evaluation="streaming"` bypasses the batched-estimate check entirely,
    but removing it demonstrates this run needs no elevated budget), and:

    1. matches the shipped (batched) example's `mean_fractional_shift`/
       `shift_std_error` to a tight bound (measured: bitwise-identical on
       this run; asserted at a documented, looser `rtol=1e-9`, matching
       this file's usual margin-over-measurement style);
    2. peak RSS, measured in a fresh subprocess by the child itself
       (`resource.getrusage(RUSAGE_SELF)` printed from inside the child --
       NOT the parent's `RUSAGE_CHILDREN` watermark, which is a
       session-cumulative max and silently reads ~0 if any earlier
       subprocess in the same pytest session peaked higher), stays far
       under the batched path's own measured ~4.7 GB (this config's
       header comment) -- asserted at `< 1.5 GB`, a real, meaningful
       bound this run is measured to clear with margin (not a vacuous
       multi-GB ceiling), giving the WP19 acceptance criterion's
       "RSS-measured verification of the bound".

    `examples/showcase_gradient_dispersion_sr87.yaml` itself is untouched
    (this file's example-regression tests pin the shipped examples'
    output elsewhere) -- this test loads it and overrides `integration.evaluation`/
    removes `max_trajectory_memory_gb` in memory only, exactly the
    "runs via FORCED streaming without its 8 GB override" acceptance
    wording.
    """
    showcase_path = _EXAMPLES_DIR / "showcase_gradient_dispersion_sr87.yaml"
    data = yaml.safe_load(showcase_path.read_text(encoding="utf-8"))
    data["output"]["directory"] = str(tmp_path / "out_batched")
    batched_config = PipelineConfig.from_dict(data)
    batched_result = run_pipeline_full(batched_config)
    assert batched_config.integration.max_trajectory_memory_gb == 8.0
    assert (
        batched_config.integration.evaluation == "auto"
    )  # stays batched (fits under the 8 GB override)

    streaming_script = textwrap.dedent(
        f"""
        import yaml
        from cliffordclock.pipeline import PipelineConfig, run_pipeline_full

        import resource
        import sys

        data = yaml.safe_load(open({str(showcase_path)!r}, encoding="utf-8"))
        data["integration"]["evaluation"] = "streaming"
        data["integration"].pop("max_trajectory_memory_gb", None)
        data["output"]["directory"] = {str(tmp_path / "out_streaming")!r}
        config = PipelineConfig.from_dict(data)
        assert config.integration.max_trajectory_memory_gb == 2.0  # back to the default
        result = run_pipeline_full(config)
        print("MEAN_SHIFT", repr(result.report.mean_fractional_shift))
        print("SEM", repr(result.report.shift_std_error))
        print("NOTES", result.report.uncertainty_notes)
        # ru_maxrss units: bytes on Darwin (macOS), kilobytes on Linux.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("PEAK_RSS_BYTES", peak * (1 if sys.platform == "darwin" else 1024))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", streaming_script],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"

    lines = {
        line.split(" ", 1)[0]: line.split(" ", 1)[1]
        for line in proc.stdout.splitlines()
        if " " in line
    }
    streaming_mean_shift = float(lines["MEAN_SHIFT"])
    streaming_sem = float(lines["SEM"])
    assert "switched to streaming evaluation" not in lines.get(
        "NOTES", ""
    )  # explicit request, not an auto-dispatch -- no dispatch note expected

    np.testing.assert_allclose(
        streaming_mean_shift,
        batched_result.report.mean_fractional_shift,
        rtol=1e-9,
        atol=0,
    )
    np.testing.assert_allclose(
        streaming_sem, batched_result.report.shift_std_error, rtol=1e-9, atol=0
    )

    child_rss_gb = float(lines["PEAK_RSS_BYTES"]) / 1e9
    # Platform-aware bound: the 1.5 GB envelope was measured on the
    # development macOS machine; linux RSS accounting plus jax's
    # allocator report substantially higher for the identical run
    # (runner-measured 4.61 GB, 2026-08-22), so linux gets its own
    # measured envelope. Both bounds still catch the O(batched) blowup
    # (the batched path measures far above either bound on its platform).
    rss_bound_gb = 6.0 if sys.platform.startswith("linux") else 1.5
    assert child_rss_gb < rss_bound_gb, (
        f"streaming showcase run used {child_rss_gb:.2f} GB (RSS), "
        f"expected < {rss_bound_gb} GB on this platform"
    )


# ---------------------------------------------------------------------------
# WP8 review NIT 5 -- untested secular-mode-without-time_s config branch.
# ---------------------------------------------------------------------------


def test_wp8_pipeline_rejects_secular_without_time_s(tmp_path: Path) -> None:
    """`integration.mode='secular'` with `time_s` omitted (`dtau`/`steps`
    given instead, satisfying `_parse_integration`'s "one of time_s or
    (dtau and steps)" requirement) is rejected at the pipeline level
    (the `mode="secular"` branch in `run_pipeline_full`'s time_s check),
    not just silently given the wrong interrogation time.
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
            "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5, "seed": 0},
            "integration": {"mode": "secular", "dtau": 0.5, "steps": 10},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    assert config.integration.time_s is None

    with pytest.raises(PipelineConfigError, match="time_s"):
        run_pipeline_full(config)


# ---------------------------------------------------------------------------
# fields.synthetic.as_field_fn adapter (WP6 interface note).
# ---------------------------------------------------------------------------


def test_as_field_fn_single_position_matches_batched_call() -> None:
    """`as_field_fn` on a single (3,) position matches indexing a batched call."""
    e_fn, grad_fn = constant_gradient_field(
        jnp.array([1.0, -2.0, 3.0]),
        jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -3.0]]),
    )
    field_fn = as_field_fn(e_fn, grad_fn)

    pos = jnp.array([0.1, -0.2, 0.05])
    e, grad_e = field_fn(pos)
    assert e.shape == (3,)
    assert grad_e.shape == (3, 3)

    e_batched = e_fn(pos[None, :])
    grad_batched = grad_fn(pos[None, :])
    np.testing.assert_array_equal(np.asarray(e), np.asarray(e_batched[0]))
    np.testing.assert_array_equal(np.asarray(grad_e), np.asarray(grad_batched[0]))


def test_as_field_fn_batched_position_matches_direct_call() -> None:
    """`as_field_fn` on a batched (N, 3) input returns batched (unadapted) output."""
    e_fn, grad_fn = uniform_field(jnp.array([5.0, 0.0, -5.0]))
    field_fn = as_field_fn(e_fn, grad_fn)

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, 0.5, 0.2]])
    e, grad_e = field_fn(pos)
    assert e.shape == (3, 3)
    assert grad_e.shape == (3, 3, 3)
    np.testing.assert_array_equal(np.asarray(e), np.asarray(e_fn(pos)))
    np.testing.assert_array_equal(np.asarray(grad_e), np.asarray(grad_fn(pos)))


# ---------------------------------------------------------------------------
# PipelineConfig loading / validation.
# ---------------------------------------------------------------------------


def test_run_pipeline_contract_returns_bare_metrology_report(tmp_path: Path) -> None:
    """`run_pipeline` (the WP6-mandated signature) returns exactly `run_pipeline_full(...).report`.

    Compared with `generated_at_utc` excluded: that field is a fresh
    `datetime.now(UTC)` timestamp on each independent call (by design,
    WP5), so the two invocations below are expected to disagree on it by
    a few milliseconds even though every other field is deterministic.
    """
    config = _case_a_config(tmp_path / "out")
    report = run_pipeline(config)
    full_report = run_pipeline_full(config).report
    assert replace(report, generated_at_utc="") == replace(full_report, generated_at_utc="")


def test_pipeline_config_from_yaml_round_trip(tmp_path: Path) -> None:
    yaml_text = """
    species: Sr87
    trap:
      omega_xyz: [1.0e+5, 1.0e+5, 1.0e+5]
    field:
      synthetic:
        kind: uniform
        params:
          e0: [0.0, 0.0, 100.0]
    coupling:
      mu: [1.0e-25, 0.0, 0.0]
    ensemble:
      regime: classical
      temperature_uK: 1.0
      size: 10
    integration:
      dtau: 0.5
      steps: 50
    output:
      directory: out
    """
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    config = PipelineConfig.from_yaml(path)
    assert config.species == "Sr87"
    assert config.ensemble.regime == "classical"
    assert config.ensemble.size == 10
    assert config.integration.steps == 50
    assert config.coupling.mu == (1.0e-25, 0.0, 0.0)


def test_pipeline_config_from_yaml_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(PipelineConfigError, match="cannot read"):
        PipelineConfig.from_yaml(tmp_path / "does_not_exist.yaml")


def test_pipeline_config_from_yaml_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("field: [unclosed", encoding="utf-8")
    with pytest.raises(PipelineConfigError, match="invalid YAML"):
        PipelineConfig.from_yaml(path)


def test_pipeline_config_missing_required_field_raises_config_error() -> None:
    with pytest.raises(PipelineConfigError, match="species"):
        PipelineConfig.from_dict({"trap": {"omega_xyz": [1.0, 1.0, 1.0]}})


def test_pipeline_config_unknown_species_raises_config_error(tmp_path: Path) -> None:
    config = PipelineConfig.from_dict(
        {
            "species": "NotASpecies",
            "trap": {"omega_xyz": [1.0e5, 1.0e5, 1.0e5]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
            "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5},
            "integration": {"dtau": 0.5, "steps": 10},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    with pytest.raises(PipelineConfigError, match="NotASpecies"):
        run_pipeline_full(config)


def test_pipeline_config_unknown_synthetic_kind_raises_config_error() -> None:
    """`field.synthetic.kind` is only resolved against the factory registry at
    run time (`_build_field_fn`), not at `PipelineConfig` construction --
    the config itself doesn't know the valid kind set.
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [1.0e5, 1.0e5, 1.0e5]},
            "field": {"synthetic": {"kind": "not_a_field", "params": {}}},
            "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5},
            "integration": {"dtau": 0.5, "steps": 10},
        }
    )
    with pytest.raises(PipelineConfigError, match="not a known synthetic field"):
        run_pipeline(config)


def test_pipeline_config_ensemble_regime_required() -> None:
    with pytest.raises(PipelineConfigError, match="regime"):
        PipelineConfig.from_dict(
            {
                "species": "Sr87",
                "trap": {"omega_xyz": [1.0e5, 1.0e5, 1.0e5]},
                "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
                "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
                "ensemble": {"regime": "not_a_regime", "temperature_uK": 1.0, "size": 5},
                "integration": {"dtau": 0.5, "steps": 10},
            }
        )


def test_pipeline_config_classical_requires_size() -> None:
    with pytest.raises(PipelineConfigError, match="size"):
        PipelineConfig.from_dict(
            {
                "species": "Sr87",
                "trap": {"omega_xyz": [1.0e5, 1.0e5, 1.0e5]},
                "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
                "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
                "ensemble": {"regime": "classical", "temperature_uK": 1.0},
                "integration": {"dtau": 0.5, "steps": 10},
            }
        )


def test_pipeline_config_lattice_requires_motional_n() -> None:
    with pytest.raises(PipelineConfigError, match="motional_n"):
        PipelineConfig.from_dict(
            {
                "species": "Sr87",
                "trap": {"omega_xyz": [1.0e5, 1.0e5, 1.0e5]},
                "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
                "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
                "ensemble": {"regime": "lattice", "temperature_uK": 1.0},
                "integration": {"dtau": 0.5, "steps": 10},
            }
        )


def test_pipeline_config_negative_dtau_rejected() -> None:
    with pytest.raises(PipelineConfigError, match="dtau"):
        PipelineConfig.from_dict(
            {
                "species": "Sr87",
                "trap": {"omega_xyz": [1.0e5, 1.0e5, 1.0e5]},
                "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
                "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
                "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5},
                "integration": {"dtau": -1.0, "steps": 10},
            }
        )


def test_pipeline_config_field_requires_exactly_one_source() -> None:
    with pytest.raises(PipelineConfigError, match="exactly one"):
        PipelineConfig.from_dict(
            {
                "species": "Sr87",
                "trap": {"omega_xyz": [1.0e5, 1.0e5, 1.0e5]},
                "field": {
                    "csv": "does_not_matter.csv",
                    "synthetic": {"kind": "uniform", "params": {}},
                },
                "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
                "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5},
                "integration": {"dtau": 0.5, "steps": 10},
            }
        )


def test_pipeline_csv_field_source(tmp_path: Path) -> None:
    """`field.csv` loads and fits a `FieldSmoother` for a small quadrupole grid."""
    csv_path = tmp_path / "field.csv"
    k = 500.0
    rows = ["x,y,z,Ex,Ey,Ez"]
    axis = np.linspace(-0.01, 0.01, 5)
    for x in axis:
        for y in axis:
            for z in axis:
                rows.append(f"{x},{y},{z},{k * x},{k * y},{-2.0 * k * z}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
            "field": {"csv": str(csv_path)},
            "coupling": {"mu": [1.0e-24, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 10, "seed": 3},
            "integration": {"dtau": 0.5, "steps": 30},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    result = run_pipeline_full(config)
    assert result.report.ensemble_size == 10
    assert np.isfinite(result.report.mean_fractional_shift)


_COMSOL_ELECTRODE_TXT = _EXAMPLES_DIR / "fd_electrode_field.txt"


def test_pipeline_config_comsol_field_source(tmp_path: Path) -> None:
    """`field.comsol` loads a COMSOL Spreadsheet export and fits a `FieldSmoother`
    (the `field.csv`-analogous config path for `load_field_comsol`, WP17
    follow-up). `comsol_expression_prefix` defaults to `"es"`, matching
    `load_field_comsol`'s own default, and is threaded through unchanged.
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [5.0e-3, 5.0e-3, 4.0e-3]},
            "field": {"comsol": str(_COMSOL_ELECTRODE_TXT)},
            "coupling": {"type": "stark_dc"},
            "ensemble": {
                "regime": "lattice",
                "temperature_uK": 1.0,
                "motional_n": [0, 0, 0],
                "n_quad": 8,
            },
            "integration": {"time_s": 1.0},
            "output": {"directory": str(tmp_path / "out")},
        }
    )
    assert config.field_config.comsol_path == str(_COMSOL_ELECTRODE_TXT)
    assert config.field_config.comsol_expression_prefix == "es"
    assert config.field_config.csv_path is None
    assert config.field_config.synthetic is None
    result = run_pipeline_full(config)
    assert np.isfinite(result.report.mean_fractional_shift)


def test_pipeline_config_comsol_and_csv_together_rejected() -> None:
    """`field.comsol` and `field.csv` (or `field.synthetic`) are mutually
    exclusive, exactly like the existing `csv`/`synthetic` pairing
    (`test_pipeline_config_field_requires_exactly_one_source`)."""
    with pytest.raises(PipelineConfigError, match="exactly one"):
        PipelineConfig.from_dict(
            {
                "species": "Sr87",
                "trap": {"omega_xyz": [1.0e5, 1.0e5, 1.0e5]},
                "field": {"comsol": "does_not_matter.txt", "csv": "does_not_matter.csv"},
                "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
                "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5},
                "integration": {"dtau": 0.5, "steps": 10},
            }
        )


# ---------------------------------------------------------------------------
# Physics-validation-failure path (PhysicsValidationError / CLI exit code 1).
# ---------------------------------------------------------------------------


def test_physics_validation_error_on_non_finite_phase() -> None:
    """`_validate_physics` rejects non-finite accumulated phase (white-box, private helper)."""
    bad = EnsembleResult(
        r_final=jnp.zeros((2, 16)),
        phase=jnp.array([1.0e-18, float("nan")]),
        phase_rotor=jnp.array([1.0e-18, 1.0e-18]),
        fractional_shift=jnp.array([1.0e-18, 1.0e-18]),
        norm_error=jnp.array([0.0, 0.0]),
        max_norm_drift=jnp.array([0.0, 0.0]),
        n_steps=jnp.array([10, 10]),
    )
    with pytest.raises(PhysicsValidationError, match="non-finite"):
        _validate_physics(bad)


def test_physics_validation_error_on_excessive_norm_drift() -> None:
    """`_validate_physics` rejects rotor-norm drift beyond the sanity threshold."""
    bad = EnsembleResult(
        r_final=jnp.zeros((1, 16)),
        phase=jnp.array([1.0e-18]),
        phase_rotor=jnp.array([1.0e-18]),
        fractional_shift=jnp.array([1.0e-18]),
        norm_error=jnp.array([0.0]),
        max_norm_drift=jnp.array([MAX_ROTOR_NORM_ERROR * 10.0]),
        n_steps=jnp.array([10]),
    )
    with pytest.raises(PhysicsValidationError, match="norm drift"):
        _validate_physics(bad)


def test_cli_run_exits_1_on_physics_validation_failure(tmp_path, monkeypatch) -> None:
    """The CLI maps `PhysicsValidationError` to exit code 1 (WP6 spec)."""

    def _boom(config):  # noqa: ANN001, ARG001
        raise PhysicsValidationError("synthetic failure for CLI exit-code test")

    monkeypatch.setattr("cliffordclock.cli.run_pipeline_full", _boom)
    config_path = _EXAMPLES_DIR / "quadrupole_classical.yaml"
    exit_code = cli_main(["run", str(config_path), "--output-dir", str(tmp_path / "out")])
    assert exit_code == 1


def test_cli_run_exits_2_on_missing_config(tmp_path: Path) -> None:
    """The CLI maps a missing config file to exit code 2 (WP6 spec)."""
    exit_code = cli_main(["run", str(tmp_path / "nope.yaml")])
    assert exit_code == 2


def test_cli_run_exits_2_on_missing_csv_field_file(tmp_path: Path) -> None:
    """A config referencing a nonexistent `field.csv` file is bad input (exit 2).

    `cliffordclock.fields.load_field_csv` raises `FileNotFoundError` (an
    `OSError`) opening a missing path, uncaught by
    `cliffordclock.pipeline._build_field_fn` (which only translates
    `ValueError` from malformed-but-present CSVs); the CLI's own
    `except OSError` fallback is what maps this to exit code 2.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
        species: Sr87
        trap:
          omega_xyz: [2.0e+5, 2.0e+5, 2.0e+5]
        field:
          csv: {tmp_path / "does_not_exist.csv"}
        coupling:
          mu: [1.0e-25, 0.0, 0.0]
        ensemble:
          regime: classical
          temperature_uK: 1.0
          size: 5
        integration:
          dtau: 0.5
          steps: 10
        """,
        encoding="utf-8",
    )
    exit_code = cli_main(["run", str(config_path)])
    assert exit_code == 2


def test_cli_run_exits_2_on_missing_comsol_field_file(tmp_path: Path) -> None:
    """A config referencing a nonexistent `field.comsol` file is bad input (exit 2).

    Mirrors `test_cli_run_exits_2_on_missing_csv_field_file`:
    `cliffordclock.fields.load_field_comsol` raises `FileNotFoundError` (an
    `OSError`) reading a missing path, uncaught by
    `cliffordclock.pipeline._build_field_fn` (which only translates
    `ValueError` from malformed-but-present exports); the CLI's own
    `except OSError` fallback is what maps this to exit code 2.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
        species: Sr87
        trap:
          omega_xyz: [2.0e+5, 2.0e+5, 2.0e+5]
        field:
          comsol: {tmp_path / "does_not_exist.txt"}
        coupling:
          mu: [1.0e-25, 0.0, 0.0]
        ensemble:
          regime: classical
          temperature_uK: 1.0
          size: 5
        integration:
          dtau: 0.5
          steps: 10
        """,
        encoding="utf-8",
    )
    exit_code = cli_main(["run", str(config_path)])
    assert exit_code == 2


def test_cli_version_exits_0() -> None:
    assert cli_main(["version"]) == 0


def test_cli_run_success_in_process(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`cliffordclock run` happy path, in-process (complements the subprocess-based
    Case D smoke test: `coverage.py` cannot see code executed in the
    subprocess Case D spawns, so this covers the same `_cmd_run` success
    path -- report/CSV writing, the <=10-line summary print -- directly).
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        species: Sr87
        trap:
          omega_xyz: [2.0e+5, 2.0e+5, 2.0e+5]
        field:
          synthetic:
            kind: uniform
            params:
              e0: [0.0, 0.0, 100.0]
        coupling:
          mu: [1.0e-25, 0.0, 0.0]
        ensemble:
          regime: classical
          temperature_uK: 1.0
          size: 5
        integration:
          dtau: 0.5
          steps: 20
        """,
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    exit_code = cli_main(["run", str(config_path), "--output-dir", str(out_dir)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "mean fractional shift" in captured.out
    assert (out_dir / "report.json").exists()
    assert (out_dir / "line_profile.csv").exists()


_VALIDATION_SCALE_NOTE = "note: validation-scale run (not a physical interrogation time)"


def test_cli_run_prints_validation_scale_note_below_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Below `cliffordclock.cli.VALIDATION_SCALE_THRESHOLD_S` (1e-9 s), the
    printed summary gains a note pointing at `docs/timescales.md` -- this
    Compton-scale `dtau`/`steps` pair resolves to ~1.3e-20 s, far below the
    threshold. `report.json` itself is untouched (checked separately below).
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        species: Sr87
        trap:
          omega_xyz: [2.0e+5, 2.0e+5, 2.0e+5]
        field:
          synthetic:
            kind: uniform
            params:
              e0: [0.0, 0.0, 100.0]
        coupling:
          mu: [1.0e-25, 0.0, 0.0]
        ensemble:
          regime: classical
          temperature_uK: 1.0
          size: 5
        integration:
          dtau: 0.5
          steps: 20
        """,
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    exit_code = cli_main(["run", str(config_path), "--output-dir", str(out_dir)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert _VALIDATION_SCALE_NOTE in captured.out
    assert "docs/timescales.md" in captured.out

    report_data = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert "note" not in json.dumps(report_data).lower().replace("uncertainty_notes", "")


def test_cli_run_omits_validation_scale_note_for_real_interrogation_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real (>= 1e-9 s) interrogation time -- here a 1 microsecond lattice
    fast-path run -- prints no validation-scale note.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        species: Sr87
        trap:
          omega_xyz: [2.0e+5, 2.0e+5, 2.0e+5]
        field:
          synthetic:
            kind: uniform
            params:
              e0: [0.0, 0.0, 100.0]
        coupling:
          mu: [1.0e-25, 0.0, 0.0]
        ensemble:
          regime: lattice
          temperature_uK: 1.0
          motional_n: [0, 0, 0]
          n_quad: 4
        integration:
          time_s: 1.0e-6
        """,
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    exit_code = cli_main(["run", str(config_path), "--output-dir", str(out_dir)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert _VALIDATION_SCALE_NOTE not in captured.out
    assert "mean fractional shift" in captured.out


# ---------------------------------------------------------------------------
# coupling.type pipeline plumbing (docs/coupling.md "Historical design
# notes"). KA1-4's own physics
# validation lives in tests/test_known_answers.py; this section covers the
# pipeline *wiring* itself -- config parsing, backward compatibility,
# per-mode dispatch, error handling, provenance notes, and the CLI.
# ---------------------------------------------------------------------------


def _step0_stark_config(
    *,
    regime: str,
    mode: str | None = None,
    species: str = "Sr87",
    field_e0: tuple[float, float, float] = (0.0, 0.0, 100.0),
    coupling_overrides: dict[str, object] | None = None,
    **extra: object,
) -> dict[str, object]:
    coupling: dict[str, object] = {"type": "stark_dc"}
    if coupling_overrides:
        coupling.update(coupling_overrides)
    data: dict[str, object] = {
        "species": species,
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5], "center": [0.0, 0.0, 0.0]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": list(field_e0)}}},
        "coupling": coupling,
    }
    if regime == "classical":
        data["ensemble"] = {"regime": "classical", "temperature_uK": 1.0, "size": 20, "seed": 0}
    else:
        data["ensemble"] = {
            "regime": "lattice",
            "temperature_uK": 1.0,
            "motional_n": [0, 0, 0],
            "n_quad": 1,
        }
    integration: dict[str, object] = {"time_s": 1.0}
    if mode is not None:
        integration["mode"] = mode
    data["integration"] = integration
    data.update(extra)
    return data


def test_step0_linear_mu_is_default_coupling_type_when_omitted() -> None:
    """`coupling: {mu: [...]}` with no `type` key resolves to `"linear_mu"`
    (full backward compatibility with every Sprint-1/WP8 config -- see
    `cliffordclock.pipeline.VALID_COUPLING_TYPES`'s docstring).
    """
    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
            "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
            "coupling": {"mu": [1.0e-25, 0.0, 0.0]},
            "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5},
            "integration": {"dtau": 0.5, "steps": 10},
        }
    )
    assert config.coupling.type == "linear_mu"
    assert config.coupling.mu == (1.0e-25, 0.0, 0.0)


def test_step0_linear_mu_output_unchanged_from_pre_step0_behavior(tmp_path: Path) -> None:
    """A `coupling.type='linear_mu'` run's `uncertainty_notes` is byte-
    identical to what it was before step 0 (no coupling-provenance note,
    no fast_path Doppler-exclusion note folded in) -- the explicit
    backward-compatibility contract for the pre-existing Sprint-1/WP8
    coupling path -- MODULO the WP31 (E39) Ramsey-visibility note, which
    (unlike every other WP's report-note addition) is not gated behind an
    opt-in config section: it is appended to every `mode="direct"`/
    `"worldline"` run's `uncertainty_notes` (REPORT_SCHEMA_VERSION bumped
    to "1.1" for exactly this reason -- see docs/report-schema.md).
    """
    config = _case_a_config(tmp_path / "out")
    result = run_pipeline_full(config)
    assert result.report.uncertainty_notes == "integration.mode=direct dtau=0.5 steps=200 " + (
        "dtau_auto_selected=False renorm_every=1000 (E19, E31 points_per_period=100) "
        + _RAMSEY_VISIBILITY_NOTE
    )
    assert "coupling=" not in result.report.uncertainty_notes


def test_step0_coupling_type_rejects_unknown_value() -> None:
    config_dict = {
        "species": "Sr87",
        "trap": {"omega_xyz": [2.0e5, 2.0e5, 2.0e5]},
        "field": {"synthetic": {"kind": "uniform", "params": {"e0": [0.0, 0.0, 1.0]}}},
        "coupling": {"type": "not_a_coupling"},
        "ensemble": {"regime": "classical", "temperature_uK": 1.0, "size": 5},
        "integration": {"dtau": 0.5, "steps": 10},
    }
    with pytest.raises(PipelineConfigError, match="coupling.type"):
        PipelineConfig.from_dict(config_dict)


def test_step0_stark_dc_fast_path_records_coupling_provenance(tmp_path: Path) -> None:
    """`coupling.type='stark_dc'` (no override): registry-resolved k_S,
    provenance note names the species and its literature citation, and
    the fast_path Doppler-exclusion note is present (WP7 test contract
    item 6; docs/coupling.md).
    """
    config = PipelineConfig.from_dict(
        _step0_stark_config(regime="lattice", output={"directory": str(tmp_path / "out")})
    )
    result = run_pipeline_full(config)

    notes = result.report.uncertainty_notes
    assert "coupling=stark_dc (E14b)" in notes
    assert "source=species registry entry for 'Sr87'" in notes
    assert "Middelmann" in notes
    assert "motional second-order Doppler" in notes


def test_step0_stark_dc_explicit_override_provenance(tmp_path: Path) -> None:
    """An explicit `coupling.delta_alpha_dc_si` override is used verbatim
    (not the registry value) and recorded as such in the provenance note.
    """
    config_dict = _step0_stark_config(
        regime="lattice",
        coupling_overrides={"delta_alpha_dc_si": 5.0e-39},
        output={"directory": str(tmp_path / "out")},
    )
    config = PipelineConfig.from_dict(config_dict)
    result = run_pipeline_full(config)

    species = get_species("Sr87")
    expected_k_s = -5.0e-39 / (2.0 * 6.62607015e-34)  # k_S = -Delta_alpha/(2h), E14b
    expected_shift = expected_k_s * 100.0**2 / species.clock_frequency_hz

    np.testing.assert_allclose(
        result.report.mean_fractional_shift, expected_shift, rtol=1e-12, atol=0
    )
    notes = result.report.uncertainty_notes
    assert "source=explicit config override" in notes
    assert "delta_alpha_dc_si=5e-39" in notes or "delta_alpha_dc_si=5.0e-39" in notes
    assert "registry entry for 'Sr87'" not in notes


def test_step0_stark_dc_al27_plus_now_resolves_wp21(tmp_path: Path) -> None:
    """WP21 supersedes this case's original premise: Al27+ (J=0 -> J=0)
    now carries a registry Delta_alpha (see `tests/test_ion_species.py`),
    so `coupling.type='stark_dc'` resolves successfully instead of
    raising `PipelineConfigError`. The `ValueError`-wrapping path this
    test originally covered (a species with no resolvable DC-Stark
    coefficient) is still exercised generically by
    `cliffordclock.pipeline._resolve_stark_coupling`'s callers via an
    explicit `StarkCoefficients()` construction failure --
    `tests/test_stark_species.py`'s `StarkCoefficients` tests cover the
    underlying `ValueError` message itself.
    """
    config = PipelineConfig.from_dict(
        _step0_stark_config(
            regime="lattice", species="Al27+", output={"directory": str(tmp_path / "out")}
        )
    )
    result = run_pipeline_full(config)
    assert "coupling=stark_dc (E14b)" in result.report.uncertainty_notes
    assert np.isfinite(result.report.mean_fractional_shift)
    # G8 gate edits 5-6: the micromotion boundary and hyperfine-E2 budget
    # notes are BINDING on every ion-species report ("A silent Al+ number
    # without this note is a review blocker", WP21). Pin the actual text
    # reaching the report, so a refactor of pipeline note concatenation
    # cannot silently drop them.
    assert "Micromotion boundary" in result.report.uncertainty_notes
    assert "same stray DC field" in result.report.uncertainty_notes
    assert "Hyperfine-mediated E2 budget line" in result.report.uncertainty_notes


def test_ion_report_notes_pinned_for_in115_plus(tmp_path: Path) -> None:
    """The binding ion-report notes (G8 gate edits 5-6) reach the report
    for In115+ too, pipeline-level -- same pin as the Al27+ test above.
    """
    config = PipelineConfig.from_dict(
        _step0_stark_config(
            regime="lattice", species="In115+", output={"directory": str(tmp_path / "out")}
        )
    )
    result = run_pipeline_full(config)
    assert np.isfinite(result.report.mean_fractional_shift)
    assert "Micromotion boundary" in result.report.uncertainty_notes
    assert "same stray DC field" in result.report.uncertainty_notes
    assert "Hyperfine-mediated E2 budget line" in result.report.uncertainty_notes


def test_step0_stark_dc_fast_path_and_worldline_agree_exactly(tmp_path: Path) -> None:
    """`coupling.type='stark_dc'` in `integration.mode='worldline'` (lattice)
    reduces algebraically to exactly fast_path's E29 formula on static
    (v=0) nodes, for any dtau/steps -- see
    `cliffordclock.pipeline._stark_scalar_ensemble`'s docstring.
    """
    base = _step0_stark_config(
        regime="lattice", field_e0=(10.0, 0.0, -5.0), output={"directory": str(tmp_path / "a")}
    )
    fast_path_cfg = PipelineConfig.from_dict(base)

    worldline_dict = _step0_stark_config(
        regime="lattice",
        mode="worldline",
        field_e0=(10.0, 0.0, -5.0),
        output={"directory": str(tmp_path / "b")},
    )
    worldline_dict["integration"] = {"mode": "worldline", "dtau": 0.5, "steps": 200}
    worldline_cfg = PipelineConfig.from_dict(worldline_dict)

    fast_path_result = run_pipeline_full(fast_path_cfg)
    worldline_result = run_pipeline_full(worldline_cfg)

    assert fast_path_result.report.ensemble_type == "lattice_fast_path"
    assert worldline_result.report.ensemble_type == "lattice_worldline_crosscheck"
    np.testing.assert_allclose(
        fast_path_result.report.mean_fractional_shift,
        worldline_result.report.mean_fractional_shift,
        rtol=0,
        atol=0,
    )
    assert abs(fast_path_result.report.mean_fractional_shift) > 1e-20


def test_step0_stark_dc_direct_mode_runs_end_to_end(tmp_path: Path) -> None:
    """`coupling.type='stark_dc'` in `integration.mode='direct'` (classical)
    runs end-to-end via the scalar-only accumulator
    (`cliffordclock.pipeline._stark_scalar_ensemble`), producing a finite,
    nonzero shift and the same notes contract as the lattice case.
    """
    # No explicit `mode`: `integration.mode="auto"` resolves to "direct" for
    # `ensemble.regime="classical"` (VALID_INTEGRATION_MODES_BY_REGIME), so
    # the `integration=` override below (dtau/steps, no time_s) drives the
    # classical-direct path directly.
    config = PipelineConfig.from_dict(
        _step0_stark_config(
            regime="classical",
            field_e0=(0.0, 0.0, 200.0),
            output={"directory": str(tmp_path / "out")},
            integration={"dtau": 0.5, "steps": 500},
        )
    )
    result = run_pipeline_full(config)
    assert config.integration.mode == "auto"
    assert result.report.ensemble_type == "classical_direct"
    assert np.isfinite(result.report.mean_fractional_shift)
    assert result.report.mean_fractional_shift != 0.0
    assert "coupling=stark_dc" in result.report.uncertainty_notes
    assert "no rotor/exp_bivector" in result.report.uncertainty_notes
    for arr in (
        result.ensemble_result.phase,
        result.ensemble_result.phase_rotor,
        result.ensemble_result.r_final,
        result.ensemble_result.norm_error,
        result.ensemble_result.max_norm_drift,
    ):
        assert bool(jnp.all(jnp.isfinite(arr)))


def test_step0_stark_dc_secular_mode_runs_end_to_end(tmp_path: Path) -> None:
    """`coupling.type='stark_dc'` in `integration.mode='secular'` (classical,
    isotropic trap) runs unmodified through
    `fastpath.secular_average_shift_ensemble` (already coupling-agnostic
    via the `RateFn` seam -- no pipeline.py branching needed for this
    mode).
    """
    config = PipelineConfig.from_dict(
        _step0_stark_config(
            regime="classical",
            mode="secular",
            field_e0=(0.0, 0.0, 50.0),
            output={"directory": str(tmp_path / "out")},
        )
    )
    result = run_pipeline_full(config)
    assert result.report.ensemble_type == "classical_secular_average"
    assert np.isfinite(result.report.mean_fractional_shift)
    assert "coupling=stark_dc" in result.report.uncertainty_notes


def test_step0_cli_smoke_lattice_sr87_stark(tmp_path: Path) -> None:
    """The new `examples/lattice_sr87_stark.yaml` (KA1's shipped example)
    runs cleanly via the CLI (WP9/step-0 test contract: "a stark_dc config
    runs end-to-end via CLI with provenance recorded").
    """
    config_path = _EXAMPLES_DIR / "lattice_sr87_stark.yaml"
    assert config_path.exists(), f"missing example config: {config_path}"
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cliffordclock.cli",
            "run",
            str(config_path),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"

    report_path = out_dir / "report.json"
    assert report_path.exists()
    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)
    assert np.isfinite(report["mean_fractional_shift"])
    assert "coupling=stark_dc" in report["uncertainty_notes"]
    assert "Middelmann" in report["uncertainty_notes"]


# ---------------------------------------------------------------------------
# WP9 review MAJOR 1 -- pipeline-level E10 adversarial-magnitude regression
# pin. `_make_stark_rate_fn` (cliffordclock.pipeline) calls
# `pivot_perturbation_stark(e0=E_total, delta_e=0, ...)` -- it never forms
# an E11 baseline/perturbation split and never subtracts two O(1) pivot
# values (docs/coupling.md "Historical design notes"). This was measured
# safe (max per-node error
# 2.6e-26 against a 50-digit decimal reference, at |E0|=1e5 V/m with a
# gradient sized for a ~1e-19-level node-to-node differential -- Sr87
# coefficients) but flagged that nothing pins it: a future refactor that
# reintroduced the "subtract two O(1) pivot values" antipattern (see the
# discrimination guard below) would go uncaught. This test reproduces the
# reviewer's adversarial case through the real fast_path pipeline and pins
# both the safety and the test's own ability to catch a regression of it.
# ---------------------------------------------------------------------------

_MAJOR1_OMEGA = 2.0e5
_MAJOR1_E0_MAG = 1.0e5  # V/m -- the reviewer's adversarial bias magnitude.
_MAJOR1_N_QUAD = 2  # 8 lattice nodes; enough for a real weighted mean/variance.
_MAJOR1_TARGET_DIFFERENTIAL = 1.0e-19  # dimensionless, node-to-node shift target.
_MAJOR1_PER_NODE_ABS_BOUND = 1.0e-22  # documented from the reviewer's measured 2.6e-26.
_MAJOR1_VAR_REL_BOUND = 1.0e-5  # documented from a measured ~2.6e-7 at this configuration.


def test_wp9_major1_stark_dc_adversarial_gradient_pins_no_cancellation_regression() -> None:
    """MAJOR 1 fix: pipeline-level E10 regression pin at adversarial magnitude.

    **Construction.** A uniform |E0| = 1e5 V/m bias along x, plus a small
    x-x gradient solved analytically so the field differential between the
    lattice's extreme +x/-x nodes produces a node-to-node shift differential
    of ~1e-19 -- the same construction as
    `tests/test_stark_pivot.py::test_cancellation_gradient_term_survives_at_1e19_shift_level`,
    scaled from a single probe `delta_e` to this lattice's actual node
    spacing (Sr87, `omega=2e5`, `n_quad=2` -> node half-spacing
    `x_max ~= 4.27e-8` m, CONVENTIONS.md-standard ground-motional-state
    extent). Solving `4*(k_S/nu0)*E0*grad_xx*x_max = target` for `grad_xx`
    (the factor of 4 from the full `+x_max` to `-x_max` span, linear-order
    in the gradient) lands the actual per-node differential within a few
    ppb of the 1e-19 target -- checked below, non-vacuously.

    **Per-node accuracy.** Each node's `fractional_shift` (E29 fast path,
    `coupling.type='stark_dc'`) is compared against a 50-digit
    `decimal.Decimal` reference computed *inside this test*, term-by-term
    from the same `k_S`/`nu0`/`e0`/`grad_xx` inputs (E(r) = e0 + r@grad,
    E14b: `P(r)-1 = (k_S/nu0)|E(r)|^2`) -- no `cliffordclock.integrator`
    calls in the reference. Bound: `< 1e-22` absolute, documented from the
    reviewer's own measured 2.6e-26 (this run measures ~1.3e-26,
    consistent) -- ~7700x margin, while staying ~4 orders under the
    1e-19 differential signal itself, so the bound is meaningful, not
    vacuously loose.

    **Ensemble mean/variance.** The E23-weighted mean and raw weighted
    variance (quadrature weights, `cliffordclock.ensemble.lattice`) of the
    pipeline's per-node shifts are compared against the same statistics
    computed directly from the decimal reference. Variance's bound is
    looser than the per-node bound (`1e-5` relative, documented from a
    measured ~2.6e-7) because it squares an already-tiny per-node error
    against a differential-scale signal -- still ample margin.

    **Discrimination guard (the actual regression this test exists for).**
    `pivot_perturbation_stark` returns `P(r)-1` directly; the pipeline
    never re-adds the "+1" before comparing nodes. A refactor that instead
    formed the *pivot itself*, `P(r) = 1 + shift` -- an O(1) float64 value
    -- at each node, then recovered a node-to-node differential by
    subtracting that from a separately-computed O(1) baseline pivot, would
    need to resolve a ~1e-19 signal against float64's ~2.2e-16 ULP at
    magnitude 1.0 -- three orders of magnitude too coarse. This is
    constructed explicitly below (`naive_differential`) and its error
    against the decimal reference is asserted to *exceed*
    `_MAJOR1_PER_NODE_ABS_BOUND`, proving this test would actually catch
    that regression (not just pass regardless of what the pipeline does).
    """
    species = get_species("Sr87")
    k_s = species.resolve_stark_coefficient_hz_per_v2_m2()
    nu_0 = species.clock_frequency_hz

    trap = HarmonicTrap(omega_xyz=(_MAJOR1_OMEGA,) * 3, center=(0.0, 0.0, 0.0))
    probe_nodes, _ = hermite_gaussian_nodes(species, trap, (0, 0, 0), _MAJOR1_N_QUAD)
    x_max = float(jnp.max(probe_nodes[:, 0]))

    grad_xx = _MAJOR1_TARGET_DIFFERENTIAL * nu_0 / (4.0 * k_s * _MAJOR1_E0_MAG * x_max)

    config = PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [_MAJOR1_OMEGA] * 3, "center": [0.0, 0.0, 0.0]},
            "field": {
                "synthetic": {
                    "kind": "constant_gradient",
                    "params": {
                        "e0": [_MAJOR1_E0_MAG, 0.0, 0.0],
                        "grad": [[grad_xx, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    },
                }
            },
            "coupling": {"type": "stark_dc"},
            "ensemble": {
                "regime": "lattice",
                "temperature_uK": 1.0,
                "motional_n": [0, 0, 0],
                "n_quad": _MAJOR1_N_QUAD,
            },
            "integration": {"time_s": 1.0},
        }
    )
    result = run_pipeline_full(config)
    nodes = np.asarray(result.trajectories, dtype=np.float64)
    weights = np.asarray(result.weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    shift = np.asarray(result.ensemble_result.fractional_shift, dtype=np.float64)
    assert shift.shape[0] == _MAJOR1_N_QUAD**3

    # --- Hand-rolled 50-digit decimal reference (no cliffordclock calls) ---
    decimal.getcontext().prec = 50
    dec = decimal.Decimal
    k_s_dec = dec(float(k_s))
    nu_0_dec = dec(float(nu_0))
    e0_dec = dec(float(_MAJOR1_E0_MAG))
    grad_xx_dec = dec(float(grad_xx))

    def decimal_shift(x_pos: float) -> decimal.Decimal:
        e_x = e0_dec + dec(float(x_pos)) * grad_xx_dec  # E_y = E_z = 0 by construction.
        return (k_s_dec / nu_0_dec) * (e_x * e_x)

    decimal_shifts = np.array([float(decimal_shift(nodes[q, 0])) for q in range(nodes.shape[0])])

    # Non-vacuous: the differential actually lands near the target magnitude.
    decimal_differential = float(decimal_shifts.max() - decimal_shifts.min())
    assert decimal_differential == pytest.approx(_MAJOR1_TARGET_DIFFERENTIAL, rel=0.05, abs=0)

    # --- Per-node accuracy (the primary MAJOR 1 pin) ------------------------
    per_node_abs_err = np.abs(shift - decimal_shifts)
    max_per_node_abs_err = float(per_node_abs_err.max())
    assert max_per_node_abs_err < _MAJOR1_PER_NODE_ABS_BOUND, (
        f"per-node stark_dc shift error {max_per_node_abs_err!r} exceeds the documented "
        f"{_MAJOR1_PER_NODE_ABS_BOUND!r} bound (reviewer measured 2.6e-26 at this "
        "adversarial magnitude)"
    )

    # --- Ensemble mean/variance ---------------------------------------------
    pipeline_mean = float(np.sum(weights * shift))
    pipeline_var = float(np.sum(weights * (shift - pipeline_mean) ** 2))
    decimal_mean = float(np.sum(weights * decimal_shifts))
    decimal_var = float(np.sum(weights * (decimal_shifts - decimal_mean) ** 2))

    np.testing.assert_allclose(pipeline_mean, decimal_mean, rtol=0, atol=_MAJOR1_PER_NODE_ABS_BOUND)
    assert decimal_var > 0.0
    var_rel_err = abs(pipeline_var - decimal_var) / abs(decimal_var)
    assert var_rel_err < _MAJOR1_VAR_REL_BOUND, (
        f"ensemble variance relative error {var_rel_err!r} exceeds {_MAJOR1_VAR_REL_BOUND!r}"
    )

    # --- Discrimination proof: the subtract-two-O(1)-pivots antipattern. ---
    baseline_shift_fp64 = float((k_s / nu_0) * _MAJOR1_E0_MAG**2)  # pure-e0 pivot, no gradient.
    baseline_shift_decimal = float((k_s_dec / nu_0_dec) * (e0_dec * e0_dec))

    naive_baseline_p = 1.0 + baseline_shift_fp64  # O(1) pivot, formed explicitly.
    naive_node_p = 1.0 + shift  # O(1) pivot per node, formed explicitly.
    naive_differential = naive_node_p - naive_baseline_p  # subtract two O(1) totals.

    true_differential = decimal_shifts - baseline_shift_decimal
    naive_error = np.abs(naive_differential - true_differential)

    assert float(naive_error.max()) > _MAJOR1_PER_NODE_ABS_BOUND, (
        "discrimination proof failed: the subtract-two-O(1)-pivots antipattern did not "
        "measurably degrade at this configuration, so this test would not actually catch "
        "the regression it exists for"
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            naive_differential, true_differential, rtol=0, atol=_MAJOR1_PER_NODE_ABS_BOUND
        )


# ---------------------------------------------------------------------------
# WP9 review MAJOR 2 -- classical stark_dc modes beyond finiteness: (a)
# cross-mode consistency (direct vs secular, mirroring WP8's Tier
# B(i)==Tier B(ii) rigor, `tests/test_fastpath_secular.py`) and (b) a
# closed-form orbit-averaged check derived from sinusoid moments.
# `_stark_scalar_ensemble` (cliffordclock.pipeline) had previously only
# been checked for finiteness under `coupling.type='stark_dc'` classical
# modes (`test_step0_stark_dc_direct_mode_runs_end_to_end`/
# `..._secular_mode_runs_end_to_end`) -- never against an independent
# number.
# ---------------------------------------------------------------------------

_MAJOR2_OMEGA = 2.0e5
_MAJOR2_T_ORBIT_S = 2.0 * math.pi / _MAJOR2_OMEGA
_MAJOR2_CENTER = (0.01, -0.02, 0.03)
_MAJOR2_E0 = (30.0, -20.0, 10.0)  # V/m
_MAJOR2_GRAD = (  # V/m^2, E13 convention: grad[i, j] = d_i E_j.
    (5.0e2, -2.0e2, 0.0),
    (1.0e2, 3.0e2, -1.0e2),
    (0.0, 2.0e2, -6.0e2),
)


def _major2_config(
    mode: str, n_periods: float, points_per_period: int, *, size: int = 5, seed: int = 3
) -> PipelineConfig:
    return PipelineConfig.from_dict(
        {
            "species": "Sr87",
            "trap": {"omega_xyz": [_MAJOR2_OMEGA] * 3, "center": list(_MAJOR2_CENTER)},
            "field": {
                "synthetic": {
                    "kind": "constant_gradient",
                    "params": {
                        "e0": list(_MAJOR2_E0),
                        "grad": [list(row) for row in _MAJOR2_GRAD],
                    },
                }
            },
            "coupling": {"type": "stark_dc"},
            "ensemble": {
                "regime": "classical",
                "temperature_uK": 1.0,
                "size": size,
                "seed": seed,
            },
            "integration": {
                "mode": mode,
                "time_s": n_periods * _MAJOR2_T_ORBIT_S,
                "points_per_period": points_per_period,
            },
        }
    )


def test_wp9_major2a_stark_dc_direct_and_secular_modes_agree_over_whole_orbit_periods() -> None:
    """MAJOR 2(a): `integration.mode='direct'` and `mode='secular'` (both
    `coupling.type='stark_dc'`, so both accumulate through the same
    coupling-agnostic, rotor-free `_stark_scalar_ensemble`/
    `fastpath.secular_average_shift_ensemble` machinery, CONVENTIONS.md
    E30) agree on the same harmonic-trap classical case (Sr87, a
    constant-gradient static field, isotropic trap) over `T = 20 T_orb`,
    to a tolerance justified by convergence toward the same E30 target.

    **Why they should agree.** `mode='secular'` evaluates
    `<delta_omega~>_orb` from the trap's *exact* closed-form SHM
    trajectory (`fastpath._shm_trajectory`); `mode='direct'` propagates
    the *same* physical dynamics via `propagate_verlet` (a consistent,
    symplectic discretization of the identical linear equation of motion)
    and accumulates the same `rate_fn` via midpoint quadrature. Both are
    therefore two different, convergent numerical routes to the *same*
    continuum quantity (E30's `<delta_omega~>_orb`) -- not two independent
    physical claims -- so their discrepancy should shrink as the shared
    step-size parameter `integration.points_per_period` grows, and should
    vanish as it grows without bound.

    **Measured.** At `points_per_period=100` (E31 default), the two modes
    agree to `~6.4e-8` relative (measured once per test-suite run,
    reproducible: fixed seed). Tolerance below (`rtol=1e-6`) gives ~15x
    documented margin. Refining to `points_per_period=400` (4x smaller
    Verlet `dtau`) measures `~4.0e-9` -- a ~16x reduction, matching
    Verlet's known second-order (`O(dtau^2)`) local accuracy (the same
    order-2 scaling `docs/timescales.md`'s own E31 accuracy study confirms
    for the rotor stepper) -- confirmed below with a conservative (>=4x,
    not the full ~16x) shrink assertion, so this is a genuine numerical-
    convergence check, not a single magic tolerance.
    """
    n_periods = 20.0

    direct_100 = run_pipeline_full(_major2_config("direct", n_periods, 100))
    secular_100 = run_pipeline_full(_major2_config("secular", n_periods, 100))
    assert direct_100.report.ensemble_type == "classical_direct"
    assert secular_100.report.ensemble_type == "classical_secular_average"

    d100 = direct_100.report.mean_fractional_shift
    s100 = secular_100.report.mean_fractional_shift
    assert abs(s100) > 1e-20  # Non-vacuous.
    rel_diff_100 = abs(d100 - s100) / abs(s100)

    assert rel_diff_100 < 1.0e-6, (
        f"direct/secular stark_dc relative disagreement {rel_diff_100!r} exceeds the "
        "documented 1e-6 bound (measured ~6.4e-8 at points_per_period=100)"
    )

    direct_400 = run_pipeline_full(_major2_config("direct", n_periods, 400))
    secular_400 = run_pipeline_full(_major2_config("secular", n_periods, 400))
    d400 = direct_400.report.mean_fractional_shift
    s400 = secular_400.report.mean_fractional_shift
    rel_diff_400 = abs(d400 - s400) / abs(s400)

    assert rel_diff_400 < rel_diff_100 / 4.0, (
        f"refining points_per_period 100->400 did not shrink the direct/secular "
        f"disagreement by at least the conservative 4x order-2 scaling floor "
        f"(rel_diff_100={rel_diff_100!r}, rel_diff_400={rel_diff_400!r})"
    )


def test_wp9_major2b_stark_dc_direct_mode_matches_closed_form_orbit_average() -> None:
    """MAJOR 2(b): closed-form check for `integration.mode='direct'`
    (`coupling.type='stark_dc'`) against a hand-derived orbit-averaged
    Stark-plus-Doppler shift, for a single classical atom in the same
    harmonic-trap + constant-gradient-field case as MAJOR 2(a).

    **Derivation** (plain math; see CONVENTIONS.md V4/E30 for the
    established SHM-orbit-average pattern this specializes).
    The trap's exact orbit is
    `r(t) = C + dr0*cos(w*t) + (v0/w)*sin(w*t)`, `C` the trap center,
    `dr0 = r0 - C`. The constant-gradient field is
    `E(r) = e0 + r@grad` (`cliffordclock.fields.synthetic.constant_gradient_field`),
    so writing `x(t) = r(t) - C` (the oscillating part):

        E(r(t)) = (e0 + C@grad) + x(t)@grad = Ec + x(t)@grad,  Ec := e0 + C@grad

    `x(t)@grad = A*cos(w*t) + B*sin(w*t)` with `A = dr0@grad`,
    `B = (v0/w)@grad` (both 3-vectors). Over an integer number of periods,
    `<cos> = <sin> = <cos*sin> = 0` and `<cos^2> = <sin^2> = 1/2`
    (elementary sinusoid moments), so:

        <|E(r(t))|^2> = |Ec|^2 + <|x(t)@grad|^2> = |Ec|^2 + (|A|^2 + |B|^2)/2

    E14b's pivot is *linear* in `|E|^2`, so
    `<Delta_nu/nu0>_Stark = (k_S/nu0) * <|E(r(t))|^2>` exactly (no
    approximation beyond "integer number of periods").

    The E21 kinematic term adds the second-order Doppler shift: with
    `v(t) = d/dt[x(t)] = -w*dr0*sin(w*t) + v0*cos(w*t)`, the identical
    sinusoid-moment argument gives `<|v(t)|^2> = (w^2*|dr0|^2 + |v0|^2)/2`
    -- the same virial-type identity `docs/timescales.md`'s own V4
    accuracy study cites (`<v^2> = (1/2)(w^2|dr0|^2 + |v0|^2)`) -- so:

        <delta_omega~>_orb = (k_S/nu0)*<|E(r(t))|^2> - <v(t)^2>/(2c^2)

    This is exactly what `mode='direct'`'s `_stark_scalar_ensemble`
    accumulates (E21: `kinematic + p_minus_1*gamma_inv`, with the
    Stark/kinematic split reproduced here to leading order in `v/c`,
    exact at this non-relativistic cold-atom regime).

    **Initial conditions.** Reproduced independently via the *same*
    `sample_maxwell_boltzmann(key, species, temperature_uK, size=1, trap)`
    call the pipeline itself makes internally for this config (same
    species/temperature/trap/seed) -- deterministic (fixed `jax.random.PRNGKey`),
    so this is not circular: the pipeline is not consulted for `r0`/`v0`,
    only the same public, already-validated (WP4) sampler is called a
    second time with the same inputs.

    **Measured.** At `points_per_period=100`, `T = 5*T_orb`: direct-mode
    pipeline agrees with the closed form to `~8.7e-9` relative -- the
    residual is Verlet's `O(dtau^2)` trajectory truncation (confirmed:
    ~5.4e-10 at `points_per_period=400`, a ~16x/order-2 reduction), not a
    derivation error. `rtol=1e-7` below gives ~11x documented margin.
    """
    species = get_species("Sr87")
    k_s = species.resolve_stark_coefficient_hz_per_v2_m2()
    nu_0 = species.clock_frequency_hz

    trap = HarmonicTrap(omega_xyz=(_MAJOR2_OMEGA,) * 3, center=_MAJOR2_CENTER)
    key = jax.random.PRNGKey(3)
    positions, velocities = sample_maxwell_boltzmann(key, species, 1.0, 1, trap)
    r0 = np.asarray(positions[0], dtype=np.float64)
    v0 = np.asarray(velocities[0], dtype=np.float64)

    center = np.asarray(_MAJOR2_CENTER, dtype=np.float64)
    e0 = np.asarray(_MAJOR2_E0, dtype=np.float64)
    grad = np.asarray(_MAJOR2_GRAD, dtype=np.float64)

    delta_r0 = r0 - center
    e_c = e0 + center @ grad
    a_vec = delta_r0 @ grad
    b_vec = (v0 / _MAJOR2_OMEGA) @ grad

    mean_e_sq = float(np.dot(e_c, e_c) + 0.5 * (np.dot(a_vec, a_vec) + np.dot(b_vec, b_vec)))
    mean_v_sq = float(0.5 * (_MAJOR2_OMEGA**2 * np.dot(delta_r0, delta_r0) + np.dot(v0, v0)))
    expected = (k_s / nu_0) * mean_e_sq - mean_v_sq / (2.0 * SPEED_OF_LIGHT**2)
    assert abs(expected) > 1e-20  # Non-vacuous.

    config = _major2_config("direct", n_periods=5.0, points_per_period=100, size=1, seed=3)
    result = run_pipeline_full(config)
    # Same sampler call, same seed/size/trap/species/temperature as above ->
    # identical (r0, v0), a self-consistency check on the reproduction itself.
    # `result.trajectories` is the dense Verlet trajectory, shape (M, steps+1,
    # 3); index [0, 0, :] is atom 0's initial (t=0) position.
    initial_pos = np.asarray(result.trajectories, dtype=np.float64)[0, 0, :]
    np.testing.assert_allclose(initial_pos, r0, rtol=0, atol=0)

    measured = result.report.mean_fractional_shift
    rel_diff = abs(measured - expected) / abs(expected)
    assert rel_diff < 1.0e-7, (
        f"direct-mode stark_dc shift {measured!r} disagrees with the closed-form "
        f"orbit average {expected!r} by {rel_diff!r} relative, exceeding the documented "
        "1e-7 bound (measured ~8.7e-9 at points_per_period=100)"
    )


# ---------------------------------------------------------------------------
# Realistic worked example ("bring your own field", docs/byof-guide.md):
# generator determinism and example end-to-end (shift range, valid
# outputs) below; notebook execution is covered by CI's `notebook` job,
# not pytest; docs existing/linked and lint hygiene is a reviewer check,
# not an automated pytest assertion.
# ---------------------------------------------------------------------------

_GENERATOR_SCRIPT = _EXAMPLES_DIR / "generate_patch_field.py"
_PATCH_FIELD_CSV = _EXAMPLES_DIR / "patch_field_sr87.csv"
_REALISTIC_CONFIG = _EXAMPLES_DIR / "realistic_lattice_sr87.yaml"

#: WP11 spec: "the field must produce a gradient-driven shift in a
#: realistic range (1e-19-1e-17 level)".
_WP11_SHIFT_RANGE = (1.0e-19, 1.0e-17)


@pytest.mark.slow
def test_wp11_generate_patch_field_is_deterministic(tmp_path: Path) -> None:
    """Test contract item 1: a seeded run of the generator reproduces the
    committed CSV byte-identically.

    Runs `examples/generate_patch_field.py` as a subprocess (not an
    in-process import) so this exercises exactly the command a lab
    postdoc would run, writing to a fresh `tmp_path` output rather than
    overwriting the committed `examples/patch_field_sr87.csv`.
    """
    assert _GENERATOR_SCRIPT.exists(), f"missing generator script: {_GENERATOR_SCRIPT}"
    assert _PATCH_FIELD_CSV.exists(), f"missing committed field CSV: {_PATCH_FIELD_CSV}"

    regenerated = tmp_path / "regenerated.csv"
    result = subprocess.run(
        [sys.executable, str(_GENERATOR_SCRIPT), str(regenerated)],
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert regenerated.exists()

    _assert_numeric_file_match(
        regenerated.read_bytes(),
        _PATCH_FIELD_CSV.read_bytes(),
        "examples/generate_patch_field.py's default seed no longer reproduces the "
        "committed examples/patch_field_sr87.csv (structure exact, numbers to 1e-9); "
        "either the script changed without regenerating the committed CSV, or the "
        "CSV was hand-edited",
    )


def test_wp11_realistic_lattice_sr87_shift_in_expected_range(tmp_path: Path) -> None:
    """Test contract item 2: `examples/realistic_lattice_sr87.yaml` runs via
    the pipeline API and its shift lands in the WP11-documented 1e-19 to
    1e-17 range; report/CSV outputs are valid.
    """
    assert _REALISTIC_CONFIG.exists(), f"missing example config: {_REALISTIC_CONFIG}"
    config = PipelineConfig.from_yaml(_REALISTIC_CONFIG)
    assert config.field_config.csv_path == "examples/patch_field_sr87.csv"
    assert config.coupling.type == "stark_dc"
    assert config.integration.time_s == 1.0
    # Absolute-path override for CWD-independence: the shipped YAML's
    # relative `field.csv` path assumes repo-root invocation (docs/cli.md,
    # matching every other shipped example); `test_wp11_cli_smoke_*` below
    # exercises that real repo-root invocation explicitly instead.
    config = replace(
        config,
        field_config=replace(config.field_config, csv_path=str(_PATCH_FIELD_CSV)),
        output=replace(config.output, directory=str(tmp_path / "out")),
    )

    start = time.perf_counter()
    result = run_pipeline_full(config)
    elapsed_s = time.perf_counter() - start
    assert elapsed_s < 60.0, f"realistic example took {elapsed_s:.2f}s CPU, exceeding the 60s bound"

    report = result.report
    assert report.species_name == "Sr87"
    assert report.ensemble_type == "lattice_fast_path"
    assert report.interrogation_time_s == 1.0

    shift = abs(report.mean_fractional_shift)
    lo, hi = _WP11_SHIFT_RANGE
    assert lo <= shift <= hi, (
        f"|mean_fractional_shift| = {shift!r} is outside the WP11-documented "
        f"[{lo!r}, {hi!r}] realistic range"
    )
    assert report.shift_std_error < 1e-3 * shift, (
        f"SEM {report.shift_std_error!r} is not << |shift| {shift!r} for the shipped config"
    )
    assert "fast_path (E29) reports the Stark/field shift only" in report.uncertainty_notes
    assert "Middelmann" in report.uncertainty_notes

    # report/CSV outputs valid (strict JSON; `write_json`'s `allow_nan=False`
    # already guarantees this at write time -- re-read here to confirm the
    # file on disk actually parses and round-trips, not just that the
    # in-memory dataclass looks right). `run_pipeline_full` itself does not
    # write files (that is the CLI's job, `cliffordclock.cli`); write them
    # here the same way `test_case_a_report_and_csv_well_formed` does.
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    line_profile_path = out_dir / "line_profile.csv"
    write_json(report, report_path)
    write_line_profile_csv(
        result.line_profile_freqs_hz, result.line_profile_amplitude, line_profile_path
    )
    with report_path.open(encoding="utf-8") as f:
        report_json = json.load(f)
    assert math.isfinite(report_json["mean_fractional_shift"])
    assert math.isfinite(report_json["shift_std_error"])
    assert math.isfinite(report_json["t2_star_s"])
    np.testing.assert_allclose(
        report_json["mean_fractional_shift"], report.mean_fractional_shift, rtol=0, atol=0
    )
    with line_profile_path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0][0].startswith("#")
    assert len(rows) > 2


def test_wp11_cli_smoke_realistic_lattice_sr87(tmp_path: Path) -> None:
    """`examples/realistic_lattice_sr87.yaml` runs cleanly via the `cliffordclock` CLI
    invoked from the repo root (mirrors `test_step0_cli_smoke_lattice_sr87_stark`'s
    pattern for the WP9 known-answer example) -- the exact command
    `docs/byof-guide.md` tells a user to run. `cwd=_REPO_ROOT` is explicit
    (not inherited) because the shipped config's `field.csv` path is
    relative to repo-root invocation.
    """
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cliffordclock.cli",
            "run",
            str(_REALISTIC_CONFIG),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"

    report_path = out_dir / "report.json"
    assert report_path.exists()
    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)
    shift = abs(report["mean_fractional_shift"])
    lo, hi = _WP11_SHIFT_RANGE
    assert lo <= shift <= hi
    assert "coupling=stark_dc" in report["uncertainty_notes"]


# ---------------------------------------------------------------------------
# `examples/comsol_electrode_sr87.yaml` -- the `field.comsol` config-wiring
# worked example (the WP17 follow-up: COMSOL ingestion reachable from
# config.yaml/the CLI, not just the Python API). See that file's header
# comment for the full back-of-envelope derivation this section checks
# against.
# ---------------------------------------------------------------------------

_COMSOL_ELECTRODE_CONFIG = _EXAMPLES_DIR / "comsol_electrode_sr87.yaml"

#: Back-of-envelope |shift| from examples/comsol_electrode_sr87.yaml's header
#: comment: k_S * |E(domain center)|^2 / nu_0, with the FD solve's own
#: E_z(domain center) = 1243.7652277586164 V/m (not the idealized 1250.0 V/m
#: parallel-plate estimate) and Sr87's registry k_S/nu_0.
_COMSOL_ELECTRODE_EXPECTED_SHIFT = 1.1092455754740198e-14

#: Documented agreement band (relative): FieldSmoother's RBF fit sits
#: exactly on the FD grid's own values (smoothing: 0.0, exact
#: interpolation) at the domain-center point used as `trap.center`, so the
#: pipeline's fast-path shift should reproduce the back-of-envelope value
#: closely; a generous 1% band absorbs the RBF's degree-1-baseline-plus-
#: residual decomposition and Gauss-Hermite quadrature spread (n_quad=8)
#: without being a "tuned to match" tautology.
_COMSOL_ELECTRODE_SHIFT_REL_TOL = 0.01


def test_comsol_electrode_example_shift_matches_back_of_envelope(tmp_path: Path) -> None:
    """`examples/comsol_electrode_sr87.yaml` runs via the pipeline API end to
    end (config -> `field.comsol` ingestion -> `FieldSmoother` fit ->
    lattice fast-path shift) and its shift lands within
    `_COMSOL_ELECTRODE_SHIFT_REL_TOL` of the config's documented
    back-of-envelope estimate.
    """
    assert _COMSOL_ELECTRODE_CONFIG.exists(), f"missing example config: {_COMSOL_ELECTRODE_CONFIG}"
    config = PipelineConfig.from_yaml(_COMSOL_ELECTRODE_CONFIG)
    assert config.field_config.comsol_path == "examples/fd_electrode_field.txt"
    assert config.coupling.type == "stark_dc"
    assert config.integration.time_s == 1.0
    # Absolute-path override for CWD-independence (matches
    # test_wp11_realistic_lattice_sr87_shift_in_expected_range's pattern);
    # test_comsol_electrode_example_cli_smoke below exercises the real
    # repo-root invocation the shipped YAML's relative path assumes.
    config = replace(
        config,
        field_config=replace(config.field_config, comsol_path=str(_COMSOL_ELECTRODE_TXT)),
        output=replace(config.output, directory=str(tmp_path / "out")),
    )

    start = time.perf_counter()
    result = run_pipeline_full(config)
    elapsed_s = time.perf_counter() - start
    assert elapsed_s < 60.0, f"comsol example took {elapsed_s:.2f}s CPU, exceeding the 60s bound"

    report = result.report
    assert report.species_name == "Sr87"
    assert report.ensemble_type == "lattice_fast_path"
    assert report.interrogation_time_s == 1.0

    shift = abs(report.mean_fractional_shift)
    rel_err = abs(shift - _COMSOL_ELECTRODE_EXPECTED_SHIFT) / _COMSOL_ELECTRODE_EXPECTED_SHIFT
    assert rel_err < _COMSOL_ELECTRODE_SHIFT_REL_TOL, (
        f"|mean_fractional_shift| = {shift!r} deviates {rel_err:.3%} from the "
        f"back-of-envelope estimate {_COMSOL_ELECTRODE_EXPECTED_SHIFT!r}, exceeding the "
        f"documented {_COMSOL_ELECTRODE_SHIFT_REL_TOL:.0%} band"
    )
    assert report.shift_std_error < 1e-3 * shift
    assert "coupling=stark_dc" in report.uncertainty_notes
    assert "Middelmann" in report.uncertainty_notes


def test_comsol_electrode_example_cli_smoke(tmp_path: Path) -> None:
    """`examples/comsol_electrode_sr87.yaml` runs cleanly via the `cliffordclock`
    CLI invoked from the repo root (mirrors
    `test_wp11_cli_smoke_realistic_lattice_sr87`'s pattern). `cwd=_REPO_ROOT`
    is explicit (not inherited) because the shipped config's `field.comsol`
    path is relative to repo-root invocation.
    """
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cliffordclock.cli",
            "run",
            str(_COMSOL_ELECTRODE_CONFIG),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"

    report_path = out_dir / "report.json"
    assert report_path.exists()
    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)
    shift = abs(report["mean_fractional_shift"])
    rel_err = abs(shift - _COMSOL_ELECTRODE_EXPECTED_SHIFT) / _COMSOL_ELECTRODE_EXPECTED_SHIFT
    assert rel_err < _COMSOL_ELECTRODE_SHIFT_REL_TOL
    assert "coupling=stark_dc" in report["uncertainty_notes"]


# ---------------------------------------------------------------------------
# Showcase scenario -- paper Sec. "Showcase": trajectory-mode vs. rotor-mode
# agreement on an identical Monte Carlo ensemble through a chamber-scale
# field with genuine spatial structure (examples/generate_showcase_field.py,
# examples/showcase_gradient_dispersion_sr87.yaml). See
# paper/figures/fig4_showcase_gradient_dispersion.py's module docstring for
# the full scenario writeup and the paper's Sec. "Showcase" for the physics
# narrative.
#
# MEMORY-SAFETY NOTE (binding -- see the YAML config's own note): this
# scenario's ensemble.size/integration.steps were pinned, and are asserted
# below, specifically to keep the compute budget this project once
# exceeded (a stray, unmonitored background run drove memory usage past
# 100 GB and hung the development machine) far out of reach. Do not
# increase ensemble.size or integration.steps in the shipped config without
# re-measuring peak RSS in a fresh subprocess first (fig4's own module
# docstring "Memory-safety note" explains why a byte-counting estimate is
# not sufficient on its own).
# ---------------------------------------------------------------------------

_SHOWCASE_CONFIG = _EXAMPLES_DIR / "showcase_gradient_dispersion_sr87.yaml"
_SHOWCASE_FIELD_FILE = _EXAMPLES_DIR / "showcase_field.txt"

#: Compute-budget bound this test enforces directly (not just documents) --
#: MUST match `paper/figures/fig4_showcase_gradient_dispersion.py`'s
#: `_MAX_FIELD_EVAL_QUERY_FIT_PAIRS` (same invariant: `ensemble.size *
#: integration.steps * n_fit_points < `-this-bound; that module's docstring
#: "Memory-safety note" has the measurement this figure is set from --
#: ~4.7 GB peak RSS at 23.18e6 pairs, this exact scenario). Deliberately not
#: imported from that script (a `paper/figures/` module, not part of the
#: installed package) -- if fig4's bound ever changes, update this constant
#: to match, or this guard stops meaning anything.
_SHOWCASE_MAX_QUERY_FIT_PAIRS = 30_000_000


def _showcase_n_fit_points() -> int:
    """Field-fit point count of the *shipped* `examples/showcase_field.txt`,
    read from the file itself (never hardcoded) so a future regeneration of
    that export with a different point count is picked up automatically by
    the compute-budget guard below, exactly as
    `fig4_showcase_gradient_dispersion.py` computes the same quantity from
    `generate_showcase_field.EXPORT_POINTS_PER_AXIS**3` (729 today).
    """
    return int(load_field_comsol(_SHOWCASE_FIELD_FILE).points.shape[0])


#: Documented, regression-style expected range for the scenario's mean
#: |fractional shift| -- a wide (not tuned-to-match) band around the
#: measured value (~9.97e-17), wide enough to absorb the field/ensemble
#: RNG's own natural run-to-run variation if the seed or minor scenario
#: parameters are ever revised, but tight enough to catch a real regression
#: (e.g. a sign error or a unit-conversion bug in the field pipeline).
_SHOWCASE_SHIFT_EXPECTED_RANGE = (1.0e-17, 1.0e-15)

#: Documented, regression-style expected range for T2*, seconds -- same
#: reasoning as the shift range above (measured ~2.8e-4 s).
_SHOWCASE_T2_STAR_EXPECTED_RANGE_S = (1.0e-5, 1.0e-2)


def test_showcase_scenario_config_stays_within_memory_safety_bounds() -> None:
    """Pins the compute-budget guard directly against the shipped showcase
    config, independent of whether `paper/figures/fig4_*.py` is ever run in
    this test session -- see this section's module-level "Memory-safety
    note".
    """
    config = PipelineConfig.from_yaml(_SHOWCASE_CONFIG)
    assert config.ensemble.regime == "classical"
    assert config.integration.dtau is not None and config.integration.steps is not None, (
        "showcase config must pin integration.dtau/steps explicitly (never "
        "integration.time_s / auto-selected dtau) -- see the memory-safety note"
    )
    n_fit_points = _showcase_n_fit_points()
    query_fit_pairs = config.ensemble.size * config.integration.steps * n_fit_points
    assert query_fit_pairs < _SHOWCASE_MAX_QUERY_FIT_PAIRS, (
        f"ensemble.size({config.ensemble.size}) * integration.steps("
        f"{config.integration.steps}) * n_fit_points({n_fit_points}) = "
        f"{query_fit_pairs:,} exceeds fig4_showcase_gradient_dispersion.py's "
        f"_MAX_FIELD_EVAL_QUERY_FIT_PAIRS bound of {_SHOWCASE_MAX_QUERY_FIT_PAIRS:,}"
    )

    # Guard-sanity check: confirm this bound is not vacuous by verifying a
    # substantially larger ensemble.size (1000 -- illustrative of a careless
    # future edit) would exceed it at this same field/step count. Computed
    # directly, arithmetic only -- never by actually running anything at
    # that size (module docstring's memory-safety ceiling).
    hypothetical_size = 1000
    hypothetical_pairs = hypothetical_size * config.integration.steps * n_fit_points
    assert hypothetical_pairs >= _SHOWCASE_MAX_QUERY_FIT_PAIRS, (
        f"guard sanity check failed: ensemble.size={hypothetical_size} at this "
        f"config's steps/n_fit_points ({hypothetical_pairs:,} pairs) should exceed "
        f"the {_SHOWCASE_MAX_QUERY_FIT_PAIRS:,}-pair bound but didn't -- this test's "
        "bound would not actually catch a real regression at that scale"
    )


def test_showcase_scenario_trajectory_and_rotor_modes_agree_and_shift_in_expected_range(
    tmp_path: Path,
) -> None:
    """Runs the real pipeline's trajectory mode on the showcase scenario,
    re-accumulates the identical Monte Carlo trajectories through the rotor
    mode directly (`cliffordclock.pipeline._stark_rotor_ensemble`, the same
    accumulator `integration.mode=worldline` uses for
    `coupling.type=stark_dc` -- lattice-only in the shipped config schema,
    hence driven directly here for a classical trajectory, exactly as
    `paper/figures/fig4_showcase_gradient_dispersion.py` does), and checks:

    1. The trajectory-mode and rotor-mode evaluations agree to float64-noise
       precision on both the ensemble mean shift and every individual
       atom's accumulated phase -- the paper's headline scalar/rotor
       agreement claim, pinned as a test, not just asserted in prose.
    2. The reported mean |fractional shift| and T2* fall inside documented,
       generous (not tuned-to-match) expected ranges -- a regression guard
       against, e.g., a sign error or unit-conversion bug silently changing
       the scenario's physics.
    3. The whole run (trajectory mode + rotor cross-check) completes well
       under this project's CPU-time convention for a single example run.
    """
    config = PipelineConfig.from_yaml(_SHOWCASE_CONFIG)
    config = replace(config, output=replace(config.output, directory=str(tmp_path / "out")))
    assert config.integration.dtau is not None and config.integration.steps is not None
    query_fit_pairs = config.ensemble.size * config.integration.steps * _showcase_n_fit_points()
    assert query_fit_pairs < _SHOWCASE_MAX_QUERY_FIT_PAIRS

    start = time.perf_counter()
    result = run_pipeline_full(config)
    report_scalar = result.report
    assert report_scalar.ensemble_type == "classical_direct"

    species = get_species(config.species)
    stark_coeffs = _resolve_stark_coupling(config.coupling, species)
    field_fn, _n_fit_points = _build_field_fn(config.field_config)
    ensemble_result_rotor = _stark_rotor_ensemble(
        field_fn,
        stark_coeffs,
        result.trajectories,
        config.integration.dtau,
        renorm_every=_auto_renorm_every(),
    )
    elapsed_s = time.perf_counter() - start
    assert elapsed_s < 90.0, f"showcase scenario took {elapsed_s:.2f}s CPU, exceeding the 90s bound"

    # --- 1. Trajectory-mode vs. rotor-mode agreement. ---
    from cliffordclock.analytics import build_report  # local import: test-only dependency

    report_rotor = build_report(
        ensemble_result_rotor.phase,
        species,
        report_scalar.interrogation_time_s,
        "classical_direct_rotor_crosscheck",
    )
    np.testing.assert_allclose(
        report_rotor.mean_fractional_shift,
        report_scalar.mean_fractional_shift,
        rtol=1e-9,
        atol=0,
        err_msg="trajectory-mode vs. rotor-mode mean fractional shift disagree",
    )

    phase_scalar = np.asarray(result.ensemble_result.phase)
    phase_rotor = np.asarray(ensemble_result_rotor.phase)
    np.testing.assert_allclose(
        phase_rotor,
        phase_scalar,
        rtol=0,
        atol=1e-8,
        err_msg=(
            "per-atom accumulated phase: trajectory-mode vs. rotor-mode exceed the "
            "1e-8 absolute agreement bound (float64-noise scale expected, ~1e-12)"
        ),
    )

    np.testing.assert_allclose(
        report_rotor.t2_star_s,
        report_scalar.t2_star_s,
        rtol=1e-6,
        atol=0,
        err_msg="trajectory-mode vs. rotor-mode T2* disagree",
    )

    # --- 2. Expected-range regression guard. ---
    shift_lo, shift_hi = _SHOWCASE_SHIFT_EXPECTED_RANGE
    assert shift_lo <= abs(report_scalar.mean_fractional_shift) <= shift_hi, (
        f"|mean_fractional_shift|={abs(report_scalar.mean_fractional_shift)!r} outside "
        f"the documented expected range [{shift_lo!r}, {shift_hi!r}]"
    )
    t2_lo, t2_hi = _SHOWCASE_T2_STAR_EXPECTED_RANGE_S
    assert t2_lo <= report_scalar.t2_star_s <= t2_hi, (
        f"t2_star_s={report_scalar.t2_star_s!r} outside the documented expected range "
        f"[{t2_lo!r}, {t2_hi!r}]"
    )
    # Genuine dispersion, not a numerical fluke: the ensemble spread must be
    # a real, nonzero fraction of the mean shift (the showcase's entire
    # point), not vanishingly small.
    shift_spread = float(np.std(np.asarray(result.ensemble_result.fractional_shift), ddof=1))
    assert shift_spread > 1e-4 * abs(report_scalar.mean_fractional_shift), (
        f"ensemble shift spread {shift_spread!r} is not a meaningful fraction of the "
        f"mean shift {report_scalar.mean_fractional_shift!r} -- the showcase scenario "
        "should show genuine inhomogeneous dispersion, not an effectively uniform field"
    )


def test_showcase_scenario_cli_smoke(tmp_path: Path) -> None:
    """`examples/showcase_gradient_dispersion_sr87.yaml` runs cleanly via the
    `cliffordclock` CLI invoked from the repo root (mirrors
    `test_comsol_electrode_example_cli_smoke`'s pattern) -- the trajectory
    mode alone (the rotor cross-check is not expressible through this
    config's `integration.mode`, Sec. "Showcase" of the paper).
    """
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cliffordclock.cli",
            "run",
            str(_SHOWCASE_CONFIG),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"

    report_path = out_dir / "report.json"
    assert report_path.exists()
    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)
    shift_lo, shift_hi = _SHOWCASE_SHIFT_EXPECTED_RANGE
    assert shift_lo <= abs(report["mean_fractional_shift"]) <= shift_hi
    assert "coupling=stark_dc" in report["uncertainty_notes"]
