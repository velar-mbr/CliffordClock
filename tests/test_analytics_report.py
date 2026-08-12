# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for cliffordclock.analytics.report.

Covers the metrology-report writer's test contract (JSON/CSV output).
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from cliffordclock import __version__ as package_version
from cliffordclock.analytics.report import (
    CONVENTIONS_VERSION,
    REPORT_SCHEMA_VERSION,
    MetrologyReport,
    build_report,
    write_json,
    write_line_profile_csv,
)
from cliffordclock.analytics.stats import coherence_function, line_profile
from cliffordclock.ensemble.species import get_species

# ---------------------------------------------------------------------------
# Test contract item 5: JSON round trip.
# ---------------------------------------------------------------------------


def _build_sample_report() -> MetrologyReport:
    rng = np.random.default_rng(555)
    phi = 1.0e-18 + 5.0e-20 * rng.standard_normal(1_000)
    species = get_species("Sr87")
    return build_report(
        phi,
        species,
        t_interrogation_s=1.0,
        ensemble_type="classical_monte_carlo",
        config_hash="deadbeef" * 4,
        uncertainty_notes="synthetic test data, no systematic budget",
        generated_at_utc="2026-08-08T00:00:00+00:00",
    )


def test_write_json_round_trip_bit_identical_floats(tmp_path):
    report = _build_sample_report()
    path = tmp_path / "report.json"
    write_json(report, path)

    with path.open(encoding="utf-8") as f:
        loaded = json.load(f)

    # Every float field round-trips bit-identically.
    for field in ("interrogation_time_s", "mean_fractional_shift", "shift_std_error", "t2_star_s"):
        original = getattr(report, field)
        assert loaded[field] == original, f"{field} did not round-trip bit-identically"
        assert math.copysign(1.0, loaded[field]) == math.copysign(1.0, original)

    assert loaded["ensemble_size"] == report.ensemble_size
    assert loaded["config_hash"] == report.config_hash
    assert loaded["species_name"] == "Sr87"


def test_write_json_schema_fields_complete_and_versioned(tmp_path):
    report = _build_sample_report()
    path = tmp_path / "report.json"
    write_json(report, path)

    with path.open(encoding="utf-8") as f:
        loaded = json.load(f)

    expected_fields = {
        "report_schema",
        "conventions_version",
        "package_version",
        "generated_at_utc",
        "config_hash",
        "species_name",
        "ensemble_type",
        "ensemble_size",
        "interrogation_time_s",
        "mean_fractional_shift",
        "shift_std_error",
        "t2_star_s",
        "uncertainty_notes",
    }
    assert set(loaded.keys()) == expected_fields
    assert loaded["report_schema"] == "1.0" == REPORT_SCHEMA_VERSION
    assert loaded["conventions_version"] == "1.1.0" == CONVENTIONS_VERSION
    assert loaded["package_version"] == package_version


def test_write_json_stable_key_order(tmp_path):
    """Key order in the raw JSON text matches `MetrologyReport`'s field
    declaration order (report_schema first, per the WP5 spec's literal
    `"report_schema": "1.0"` example).
    """
    report = _build_sample_report()
    path = tmp_path / "report.json"
    write_json(report, path)

    text = path.read_text(encoding="utf-8")
    parsed_in_order = list(json.loads(text, object_pairs_hook=lambda pairs: pairs))
    keys_in_order = [k for k, _ in parsed_in_order]
    assert keys_in_order[0] == "report_schema"
    assert keys_in_order == list(MetrologyReport.__dataclass_fields__.keys())


def _build_m1_report() -> MetrologyReport:
    """A single-atom (M=1) report: `shift_std_error`/`t2_star_s` undefined
    (NaN in memory), mirroring what `cliffordclock.pipeline._build_report`
    constructs for M=1 ensembles (`build_report` itself rejects M=1 by
    design -- see `test_build_report_rejects_single_atom_variance_undefined`
    above -- so this test file builds the M=1 `MetrologyReport` directly).
    """
    return MetrologyReport(
        report_schema=REPORT_SCHEMA_VERSION,
        conventions_version=CONVENTIONS_VERSION,
        package_version=package_version,
        generated_at_utc="2026-08-08T00:00:00+00:00",
        config_hash="deadbeef" * 4,
        species_name="Sr87",
        ensemble_type="lattice_quadrature",
        ensemble_size=1,
        interrogation_time_s=1.0,
        mean_fractional_shift=1.23456789e-18,
        shift_std_error=float("nan"),
        t2_star_s=float("nan"),
        uncertainty_notes=(
            "single-atom ensemble (M=1): shift_std_error/t2_star_s are undefined (NaN)."
        ),
    )


def test_write_json_m1_report_is_strict_json_with_null_undefined_fields(tmp_path):
    """MAJOR 2 fix: an M=1 report's NaN fields serialize as JSON `null`, not
    a bare `NaN` token (RFC 8259 forbids the latter). Verified with a
    strict parser: `parse_constant` is only invoked by `json.loads` when it
    encounters a literal `NaN`/`Infinity`/`-Infinity` token in the text, so
    making it raise proves those tokens are *absent* -- a much stronger
    check than a lenient parser merely "not crashing".
    """
    report = _build_m1_report()
    path = tmp_path / "report.json"
    write_json(report, path)
    text = path.read_text(encoding="utf-8")

    def _raise_on_non_finite_constant(token: str) -> float:
        raise ValueError(f"strict JSON parser encountered non-finite constant: {token!r}")

    loaded = json.loads(text, parse_constant=_raise_on_non_finite_constant)

    assert loaded["shift_std_error"] is None
    assert loaded["t2_star_s"] is None
    # Finite fields on the same report still round-trip bit-identically
    # even though other fields on it are null.
    assert loaded["mean_fractional_shift"] == report.mean_fractional_shift
    assert loaded["interrogation_time_s"] == report.interrogation_time_s
    assert loaded["ensemble_size"] == 1


def test_write_json_finite_report_round_trip_unaffected_by_null_handling(tmp_path):
    """(ii) The pre-existing bit-identical float round trip for a normal
    (all-finite) report is unaffected by the MAJOR 2 null-handling fix --
    no field of an all-finite report is ever converted to null.
    """
    report = _build_sample_report()
    path = tmp_path / "report.json"
    write_json(report, path)

    with path.open(encoding="utf-8") as f:
        loaded = json.load(f)

    for field in ("interrogation_time_s", "mean_fractional_shift", "shift_std_error", "t2_star_s"):
        original = getattr(report, field)
        assert loaded[field] is not None
        assert loaded[field] == original
        assert math.copysign(1.0, loaded[field]) == math.copysign(1.0, original)


def test_build_report_config_hash_optional():
    report = build_report(
        np.array([1.0e-18, 2.0e-18, 3.0e-18]),
        get_species("Yb171"),
        t_interrogation_s=0.5,
        ensemble_type="lattice_quadrature",
    )
    assert report.config_hash is None


def test_build_report_zero_variance_t2star_inf_noted_and_serializes_null(tmp_path):
    """WP9 regression (latent WP5 edge case): a zero-phase-variance
    ensemble (every atom accumulated the identical phase -- here exactly
    zero, as a lattice ensemble in a uniform field orthogonal to ``mu``
    produces) must build a report with ``t2_star_s == +inf``
    (`dephasing_time_t2star`, E27's ``sigma_Phi -> 0+`` limit), not raise
    ``ZeroDivisionError``; `build_report` appends the disambiguating
    `uncertainty_notes` entry (a JSON `null` in ``t2_star_s`` otherwise
    cannot be told apart from the M=1 undefined case, see
    ``docs/report-schema.md``); and `write_json` serializes the inf as a
    strict-JSON `null` (RFC 8259 has no `Infinity` token) while every
    finite field still round-trips bit-identically.
    """
    report = build_report(
        np.zeros(64),
        get_species("Sr87"),
        t_interrogation_s=1.0,
        ensemble_type="lattice_fast_path",
        uncertainty_notes="pre-existing note",
        generated_at_utc="2026-08-10T00:00:00+00:00",
    )
    assert math.isinf(report.t2_star_s)
    assert report.t2_star_s > 0.0
    assert report.shift_std_error == 0.0  # zero spread: SEM is defined, exactly 0
    assert report.mean_fractional_shift == 0.0
    assert "pre-existing note" in report.uncertainty_notes
    assert "T2* is infinite" in report.uncertainty_notes

    path = tmp_path / "report.json"
    write_json(report, path)
    text = path.read_text(encoding="utf-8")

    def _raise_on_non_finite_constant(token: str) -> float:
        raise ValueError(f"strict JSON parser encountered non-finite constant: {token!r}")

    loaded = json.loads(text, parse_constant=_raise_on_non_finite_constant)
    assert loaded["t2_star_s"] is None
    assert loaded["shift_std_error"] == 0.0
    assert loaded["mean_fractional_shift"] == 0.0


def test_build_report_rejects_single_atom_variance_undefined():
    """WP6 addition: M=1 boundary through the public analytics API
    (`build_report`, the top of that API -- WP5 review coverage gap).
    `shift_std_error`/`dephasing_time_t2star` are undefined for a single
    atom (see `weighted_phase_stats`, E25); `build_report` must propagate
    that as a clean `ValueError`, not a silent/garbage report.
    """
    with pytest.raises(ValueError, match="undefined"):
        build_report(
            np.array([1.0e-18]),
            get_species("Sr87"),
            t_interrogation_s=1.0,
            ensemble_type="classical_monte_carlo",
        )


# ---------------------------------------------------------------------------
# Test contract item 6: CSV round trip.
# ---------------------------------------------------------------------------


def test_write_line_profile_csv_loadtxt_round_trip(tmp_path):
    n = 256
    dt_s = 1.0 / (n - 1)
    t_grid_s = np.arange(n) * dt_s
    phi_final = np.full(50, 2.0 * math.pi * 20.0 * 1.0)
    coherence = coherence_function(phi_final, 1.0, t_grid_s)
    freqs_hz, amplitude = line_profile(coherence, dt_s)

    path = tmp_path / "line_profile.csv"
    write_line_profile_csv(freqs_hz, amplitude, path)

    loaded = np.loadtxt(path, delimiter=",")
    np.testing.assert_array_equal(loaded[:, 0], freqs_hz)
    np.testing.assert_array_equal(loaded[:, 1], amplitude)


def test_write_line_profile_csv_parses_with_csv_module(tmp_path):
    freqs_hz = np.array([-2.5, -1.0, 0.0, 1.0, 2.5])
    amplitude = np.array([0.1, 0.4, 1.0, 0.4, 0.1])
    path = tmp_path / "line_profile.csv"
    write_line_profile_csv(freqs_hz, amplitude, path)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[0][0].startswith("#")
    data_rows = rows[1:]
    assert len(data_rows) == freqs_hz.shape[0]
    parsed_freqs = np.array([float(r[0]) for r in data_rows])
    parsed_amps = np.array([float(r[1]) for r in data_rows])
    np.testing.assert_array_equal(parsed_freqs, freqs_hz)
    np.testing.assert_array_equal(parsed_amps, amplitude)
