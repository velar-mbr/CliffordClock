# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smoke tests for the repository scaffold (WP0).

These tests exist to prove the packaging, fp64 configuration, CLI entry
point, and physical constants are wired up correctly before any physics code
lands in later work packages.
"""

import math
import subprocess
import sys

import jax.numpy as jnp
import pytest

import cliffordclock
from cliffordclock import constants
from cliffordclock.cli import main


def test_package_imports() -> None:
    """The top-level package and all stub subpackages import cleanly."""
    import cliffordclock.analytics
    import cliffordclock.cl13
    import cliffordclock.ensemble
    import cliffordclock.fields
    import cliffordclock.integrator

    assert cliffordclock.analytics is not None
    assert cliffordclock.cl13 is not None
    assert cliffordclock.ensemble is not None
    assert cliffordclock.fields is not None
    assert cliffordclock.integrator is not None


def test_fp64_enabled() -> None:
    """JAX must run in 64-bit mode: the 1e-18 precision target is otherwise unreachable."""
    assert jnp.zeros(1).dtype == jnp.float64


def test_version_attribute() -> None:
    """The package exposes a non-empty __version__ string."""
    assert isinstance(cliffordclock.__version__, str)
    assert cliffordclock.__version__


def test_cli_version_exits_zero_and_prints_version() -> None:
    """`cliffordclock version` exits 0 and prints the installed package version."""
    result = subprocess.run(
        [sys.executable, "-m", "cliffordclock.cli", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == cliffordclock.__version__


def test_cli_run_requires_config_argument() -> None:
    """`cliffordclock run` is implemented as of WP6 (superseding the WP0 stub, which
    raised `NotImplementedError`): it now requires a `config` path
    argument, enforced by `argparse` itself (exit code 2, "bad input" --
    see `cliffordclock.cli` and `tests/test_e2e.py` for the full `cliffordclock run`
    test contract).
    """
    with pytest.raises(SystemExit) as exc_info:
        main(["run"])
    assert exc_info.value.code == 2


def test_tau_compton_matches_literature_value() -> None:
    """TAU_COMPTON = hbar / (m_e c^2) matches an independently computed literature value.

    Independent reference computation (CODATA 2022 electron Compton
    wavelength lambda_C = 2.42631023538e-12 m, see
    https://physics.nist.gov/cgi-bin/cuu/Value?compwl):
    tau_c = lambda_C / (2 pi c) ~= 1.28809e-21 s.
    """
    compton_wavelength = 2.42631023538e-12  # m, CODATA 2022 lambda_C
    independent_tau_c = compton_wavelength / (2 * math.pi * constants.SPEED_OF_LIGHT)

    assert pytest.approx(independent_tau_c, rel=1e-6, abs=0) == constants.TAU_COMPTON
    # Sanity check against the order-of-magnitude value quoted in the plan.
    assert pytest.approx(1.288e-21, rel=1e-3, abs=0) == constants.TAU_COMPTON
