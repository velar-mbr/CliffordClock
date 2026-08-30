# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ``examples/six_line_clock.py``, the runnable expansion of the
six-line model listing in ``docs/MODEL.md``.

Loads the script by file path, since it is a standalone example outside
the ``cliffordclock`` package, and checks its headline numbers: the
analytic gravitational-redshift formula, the composed (product) ensemble
shift against that analytic value, and the gap between the composed shift
and its additive (first-order) approximation, the model's cross term.
A subprocess run covers the script the way a lab user invokes it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "examples" / "six_line_clock.py"


def _load_six_line_clock() -> ModuleType:
    spec = importlib.util.spec_from_file_location("six_line_clock", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tlc = _load_six_line_clock()

# Chou et al. 2010 (Science 329, 1630): raising a clock by 33 cm changes
# its gravitational redshift by g*dh/c^2. STANDARD_GRAVITY and
# SPEED_OF_LIGHT below are transcribed independently of the script's own
# module-level constants, so this checks the formula, independently of
# the script's own copies of the same two numbers.
STANDARD_GRAVITY = 9.80665
SPEED_OF_LIGHT = 299_792_458.0


def test_analytic_grav_term_matches_g_dh_over_c2_33cm() -> None:
    """The 33 cm case: ``g*dh/c^2`` computed independently in this test
    matches the script's own closed form to rtol 1e-12."""
    expected = STANDARD_GRAVITY * 0.33 / SPEED_OF_LIGHT**2
    np.testing.assert_allclose(tlc.analytic_grav_term(0.33), expected, rtol=1e-12, atol=0.0)


def test_analytic_grav_term_matches_g_dh_over_c2_1cm() -> None:
    """The 1 cm case, same check: rtol 1e-12 against an independently
    computed ``g*dh/c^2``."""
    expected = STANDARD_GRAVITY * 0.01 / SPEED_OF_LIGHT**2
    np.testing.assert_allclose(tlc.analytic_grav_term(0.01), expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("height_diff_m", [0.33, 0.01])
def test_composed_shift_matches_analytic(height_diff_m: float) -> None:
    """The full per-atom product, averaged over the ensemble and
    differenced between the raised and reference heights, reproduces the
    closed-form ``g*dh/c^2`` to rtol 1e-9: five orders of margin above the
    ~1e-16 relative floating-point noise this construction
    reaches (the paired velocities make the Doppler term cancel exactly)."""
    rng = np.random.default_rng(seed=0)
    case = tlc.compute_case(height_diff_m, rng)
    np.testing.assert_allclose(case["composed"], case["analytic"], rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("height_diff_m", [0.33, 0.01])
def test_composed_vs_additive_gap_is_far_below_any_effect_row(height_diff_m: float) -> None:
    """The gap between the composed (product) and additive (sum) shifts
    is the model's cross term. Both terms here are small (grav ~1e-17 to
    1e-18, Doppler ~1e-18), so the analytic cross term sits near 1e-35,
    and this test's own construction shows the gap comes out at or below
    1e-28. This bound is meaningful: the composed and additive shifts it
    separates sit at 1e-17 to 1e-18, nine to eleven orders of magnitude
    above it."""
    rng = np.random.default_rng(seed=0)
    case = tlc.compute_case(height_diff_m, rng)
    assert abs(case["cross_term_gap"]) < 1e-28
    assert abs(case["cross_term_analytic"]) < 1e-28


@pytest.mark.parametrize("height_diff_m", [0.33, 0.01])
def test_visibility_is_high_and_bounded(height_diff_m: float) -> None:
    """Ensemble visibility (the phasor modulus, this model's coherence
    observable) stays close to 1 for near-ground-state secular
    velocities, and never exceeds 1 (a modulus of a mean of unit
    vectors)."""
    rng = np.random.default_rng(seed=0)
    case = tlc.compute_case(height_diff_m, rng)
    assert 0.99 < case["visibility"] <= 1.0


def test_compute_case_is_deterministic() -> None:
    """Two independently seeded runs of the same case reproduce the same
    numbers bit for bit: no hidden nondeterminism (unseeded RNG, set
    iteration order) has crept into the ensemble construction."""
    case_a = tlc.compute_case(0.33, np.random.default_rng(seed=0))
    case_b = tlc.compute_case(0.33, np.random.default_rng(seed=0))
    assert case_a == case_b


def test_script_runs_as_a_subprocess_and_prints_the_two_cases() -> None:
    """The script a lab user runs (``python
    examples/six_line_clock.py``) exits cleanly, well under a second as
    its module docstring promises, inside this test's 5-second subprocess
    timeout, and prints both height cases."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "Height difference: 33 cm" in result.stdout
    assert "Height difference: 1 cm" in result.stdout
    assert "Chou et al. 2010" in result.stdout
