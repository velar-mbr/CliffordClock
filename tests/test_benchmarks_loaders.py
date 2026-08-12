# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for `benchmarks/loaders.py` and `benchmarks/run_benchmarks.py`
(the dataset-benchmark parsing, classification, and reproducibility-case
test contract; see `benchmarks/RESULTS.md`).

``benchmarks/`` is a top-level directory (not part of the installed
package, see `benchmarks/SOURCES.md`), so it is not importable via the
normal `cliffordclock.*` namespace; this file adds it to `sys.path`
directly, mirroring how `benchmarks/run_benchmarks.py` itself imports
`loaders` and how `tests/test_known_answers.py` already imports the
top-level `tests/reference_impl.py` module (pytest's default "prepend"
import mode already puts `tests/` on `sys.path`; `benchmarks/` needs the
same treatment explicitly since it is a sibling directory, not `tests/`
itself).

Scope, per the WP file: "parsing correctness against small committed
fixture excerpts; config-mapping unit tests; runner smoke test on one
case. The full benchmark run itself is a script ... not a unit test" --
this file does not re-run or duplicate `benchmarks/run_benchmarks.main()`'s
full report generation (that is `benchmarks/results/wp10_results.json`,
a committed script output), it tests the loading/classification logic
that feeds it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import loaders  # noqa: E402
import run_benchmarks  # noqa: E402

_FIXTURES_DIR = _BENCHMARKS_DIR / "fixtures"


# ---------------------------------------------------------------------------
# loaders.load_jila_table1 -- parsing correctness
# ---------------------------------------------------------------------------


def test_load_jila_table1_parses_all_rows_from_shipped_fixture() -> None:
    """The shipped fixture is the full JILA Table I transcription (10 rows:
    9 systematics + total); every row must parse with the right dtype/shape
    of fields and in file order."""
    entries = loaders.load_jila_table1(_FIXTURES_DIR / "jila_2403_10664_table1.csv")

    assert [e.shift_name for e in entries] == [
        "BBR",
        "Lattice Light",
        "Second Order Zeeman",
        "Density",
        "First order Zeeman",
        "Background Gas",
        "DC Stark",
        "Tunneling",
        "Minor Shifts",
        "Total Shift",
    ]
    for entry in entries:
        assert isinstance(entry.shift_e19, float)
        assert isinstance(entry.uncertainty_e19, float)
        assert isinstance(entry.uncertainty_is_upper_bound, bool)
        assert isinstance(entry.in_engine_scope, bool)
        assert entry.scope_note  # non-empty


def test_load_jila_table1_bbr_row_matches_paper_table_i_verbatim() -> None:
    """Spot-check the dominant systematic (BBR) against the paper's Table I
    verbatim value (arXiv:2403.10664v2, Table I, row "BBR"): shift
    -48417.2e-19, uncertainty 7.3e-19 -- catches a transcription-digit-swap
    bug (the kind of error `docs/validation.md`'s ALPHA_AU_TO_SI history
    warns this codebase is not immune to)."""
    entries = loaders.load_jila_table1(_FIXTURES_DIR / "jila_2403_10664_table1.csv")
    bbr = next(e for e in entries if e.shift_name == "BBR")

    np.testing.assert_allclose(bbr.shift_e19, -48417.2, rtol=0, atol=1e-9)
    np.testing.assert_allclose(bbr.uncertainty_e19, 7.3, rtol=0, atol=1e-9)
    np.testing.assert_allclose(bbr.shift_fractional, -48417.2e-19, rtol=1e-12, atol=0)
    assert bbr.in_engine_scope is False


def test_load_jila_table1_dc_stark_row_matches_paper_table_i_verbatim() -> None:
    """The one in-scope row: Table I value -1.0e-19 +/- 0.1e-19, and marked
    in_engine_scope=True (unlike every other row)."""
    entries = loaders.load_jila_table1(_FIXTURES_DIR / "jila_2403_10664_table1.csv")
    dc_stark = next(e for e in entries if e.shift_name == "DC Stark")

    np.testing.assert_allclose(dc_stark.shift_e19, -1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(dc_stark.uncertainty_e19, 0.1, rtol=0, atol=1e-9)
    assert dc_stark.in_engine_scope is True

    # Exactly one row is in scope -- everything else must be False.
    others = [e for e in entries if e.shift_name != "DC Stark"]
    assert all(not e.in_engine_scope for e in others)


def test_jila_dc_stark_precise_value_matches_main_text_verbatim() -> None:
    """`loaders.JILA_DC_STARK_PRECISE` is the main-text prose value (more
    precise than Table I's rounded -1.0+/-0.1e-19): "-9.8 +/- 0.7 x 10^-20"
    (arXiv:2403.10664v2, "DC Stark Shift" section)."""
    precise = loaders.JILA_DC_STARK_PRECISE
    np.testing.assert_allclose(precise.shift_fractional, -9.8e-20, rtol=1e-12, atol=0)
    np.testing.assert_allclose(precise.uncertainty_fractional, 0.7e-20, rtol=1e-12, atol=0)
    assert precise.in_engine_scope is True


def test_load_jila_table1_missing_column_raises_value_error(tmp_path: Path) -> None:
    """A malformed fixture (missing a required column) fails loudly, not
    silently (e.g. by defaulting a field to 0.0/False)."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("shift_name,shift_e19\nBBR,-48417.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        loaders.load_jila_table1(bad_csv)


def test_load_jila_table1_non_numeric_field_raises_value_error(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "shift_name,shift_e19,uncertainty_e19,uncertainty_is_upper_bound,"
        "in_engine_scope,scope_note\n"
        "BBR,not_a_number,7.3,false,false,note\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed row"):
        loaders.load_jila_table1(bad_csv)


def test_load_jila_table1_empty_file_raises_value_error(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text(
        "shift_name,shift_e19,uncertainty_e19,uncertainty_is_upper_bound,"
        "in_engine_scope,scope_note\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no data rows"):
        loaders.load_jila_table1(empty_csv)


# ---------------------------------------------------------------------------
# loaders.load_nist_phase_csv -- parsing correctness
# ---------------------------------------------------------------------------


def test_load_nist_phase_csv_parses_yb_clock_excerpt() -> None:
    """The shipped 20-row excerpt of the real NIST Yb-clock-phase file
    (`benchmarks/SOURCES.md`: checksummed at retrieval, first 20 rows)
    parses to the exact published values."""
    series = loaders.load_nist_phase_csv(
        _FIXTURES_DIR / "nist_m32206_yb_clock_phase_excerpt.csv", phase_units="rad"
    )
    assert series.time_s.shape == (20,)
    assert series.phase.shape == (20,)
    assert series.time_s.dtype == np.float64
    assert series.phase.dtype == np.float64
    assert series.phase_units == "rad"
    assert series.source_file == "nist_m32206_yb_clock_phase_excerpt.csv"

    # Time column is the sample index 0..19 (no header row in the source).
    np.testing.assert_allclose(series.time_s, np.arange(20, dtype=np.float64), rtol=0, atol=0)
    # First published phase value, byte-verbatim from the fetched file
    # (benchmarks/SOURCES.md sha256 c00f2c5c...).
    np.testing.assert_allclose(series.phase[0], 34.77484936207629573, rtol=1e-14, atol=0)


def test_load_nist_phase_csv_parses_10ghz_excerpt() -> None:
    series = loaders.load_nist_phase_csv(
        _FIXTURES_DIR / "nist_m32206_10ghz_phase_excerpt.csv", phase_units="mrad"
    )
    assert series.time_s.shape == (20,)
    assert series.phase_units == "mrad"
    np.testing.assert_allclose(series.phase[0], 1.292666488240893141, rtol=1e-14, atol=0)


def test_load_nist_phase_csv_wrong_field_count_raises_value_error(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("0.0 1.0 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 2 whitespace-separated fields"):
        loaders.load_nist_phase_csv(bad_csv, phase_units="rad")


def test_load_nist_phase_csv_non_numeric_field_raises_value_error(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("0.0 not_a_number\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric field"):
        loaders.load_nist_phase_csv(bad_csv, phase_units="rad")


# ---------------------------------------------------------------------------
# run_benchmarks -- config-mapping / classification unit tests
# ---------------------------------------------------------------------------


def test_classify_jila_table1_marks_every_row_not_comparable() -> None:
    """The WP10 finding (benchmarks/RESULTS.md): no JILA Table I row is
    forward-comparable under this engine's current scope. This test pins
    that classification so a future edit cannot silently flip a row to
    "comparable" (and therefore claim a residual) without this test
    failing and forcing a conscious review."""
    entries = loaders.load_jila_table1(_FIXTURES_DIR / "jila_2403_10664_table1.csv")
    verdicts = run_benchmarks.classify_jila_table1(entries)

    assert len(verdicts) == len(entries)
    assert all(v.comparable is False for v in verdicts)
    assert all(v.kpi_verdict == "N/A" for v in verdicts)
    # in_engine_scope is preserved from the source entry, not recomputed.
    scope_by_name = {v.shift_name: v.in_engine_scope for v in verdicts}
    assert scope_by_name["DC Stark"] == True  # noqa: E712 -- explicit bool check
    assert scope_by_name["BBR"] == False  # noqa: E712
    # The DC Stark row gets the specific "why not comparable" reason, not
    # the generic out-of-scope reason.
    dc_stark_verdict = next(v for v in verdicts if v.shift_name == "DC Stark")
    assert "residual stray-field magnitude" in dc_stark_verdict.reason
    assert "tuned parameter" in dc_stark_verdict.reason


def test_classify_nist_series_marks_not_comparable_with_full_dataset_size() -> None:
    """`classify_nist_series` reports the *full* published dataset size
    (44,002 samples, `benchmarks/SOURCES.md`) alongside the excerpt's
    actual row count, so the JSON output cannot be misread as "the full
    dataset only has 20 samples"."""
    series = loaders.load_nist_phase_csv(
        _FIXTURES_DIR / "nist_m32206_yb_clock_phase_excerpt.csv", phase_units="rad"
    )
    verdict = run_benchmarks.classify_nist_series(series)

    assert verdict.comparable is False
    assert verdict.kpi_verdict == "N/A"
    assert verdict.excerpt_n_samples == 20
    assert verdict.full_dataset_n_samples == 44002
    assert "Allan-deviation" in verdict.reason or "instability" in verdict.reason


# ---------------------------------------------------------------------------
# run_benchmarks -- runner smoke test (WP10 scope item 4: "runner smoke
# test on one case"). This exercises the one real pipeline invocation the
# runner performs (the illustrative DC-Stark sweep), via the actual
# coupling.type=stark_dc + lattice fast-path pipeline -- not a mock.
# ---------------------------------------------------------------------------


def test_run_dc_stark_context_sweep_smoke() -> None:
    """Smoke test: the real pipeline runs end-to-end for every swept field
    without error, and the results are physically sane (monotonically
    more negative fractional shift with increasing field magnitude, all
    finite, matching the E14b quadratic-in-field form)."""
    points = run_benchmarks.run_dc_stark_context_sweep()

    assert len(points) == len(run_benchmarks.DC_STARK_CONTEXT_SWEEP_V_PER_M)
    fields = [p.field_v_per_m for p in points]
    assert fields == list(run_benchmarks.DC_STARK_CONTEXT_SWEEP_V_PER_M)

    shifts = np.array([p.predicted_fractional_shift for p in points])
    assert np.all(np.isfinite(shifts))
    assert np.all(shifts < 0.0)  # DC Stark always red-shifts (Delta_alpha > 0 for Sr87)
    # Strictly more negative as field grows (quadratic in |E|, monotonic for E > 0).
    assert np.all(np.diff(shifts) < 0.0)

    # E14b is quadratic in field: doubling E (10 -> 20 V/m) should
    # quadruple the shift, to float64 precision (single static lattice
    # node, no quadrature/statistical noise -- see tests/test_known_answers.py
    # KA1/KA2 for why n_quad=1 uniform-field cases are exact).
    shift_10 = next(p.predicted_fractional_shift for p in points if p.field_v_per_m == 10.0)
    shift_20 = next(p.predicted_fractional_shift for p in points if p.field_v_per_m == 20.0)
    np.testing.assert_allclose(shift_20, 4.0 * shift_10, rtol=1e-10, atol=0)


def test_run_dc_stark_context_sweep_never_reports_a_kpi_verdict() -> None:
    """Structural guard: `DcStarkSweepPoint` has no `kpi_verdict`/`comparable`
    field at all -- the type itself cannot be mistaken for a benchmark
    case (see benchmarks/MAPPING.md's circularity argument for why this
    sweep must never look like a residual/PASS)."""
    from dataclasses import fields as dc_fields

    field_names = {f.name for f in dc_fields(run_benchmarks.DcStarkSweepPoint)}
    assert "kpi_verdict" not in field_names
    assert "comparable" not in field_names
    assert "residual" not in field_names


def test_build_report_smoke() -> None:
    """End-to-end smoke test of `build_report` against the shipped
    fixtures: no exception, all rows present (10 JILA + 2 NIST + 1 USTC =
    13 not-applicable, + 1 NPL reproducibility case = 14 total), KPI
    summary counts are internally consistent, and nothing budget-only is
    silently marked comparable (WP10 follow-up, 2026-08-10: report schema
    2.0 -- see `benchmarks/run_benchmarks.build_report`)."""
    report = run_benchmarks.build_report(
        jila_fixture=_FIXTURES_DIR / "jila_2403_10664_table1.csv",
        nist_yb_fixture=_FIXTURES_DIR / "nist_m32206_yb_clock_phase_excerpt.csv",
        nist_10ghz_fixture=_FIXTURES_DIR / "nist_m32206_10ghz_phase_excerpt.csv",
    )

    assert report["wp10_report_schema"] == "2.0"
    assert len(report["jila_2403_10664_table1"]) == 10
    assert len(report["nist_m32206"]) == 2
    assert len(report["ustc_metrologia_63_025002"]) == 1
    assert len(report["dc_stark_context_sweep_species_sr87"]["points"]) == 6
    assert "npl_1706_01944_reproducibility_case" in report

    summary = report["kpi_summary"]
    assert summary["not_applicable_rows"] == 13  # 10 JILA + 2 NIST + 1 USTC
    assert summary["reproducibility_cases_total"] == 1
    assert summary["blind_prediction_cases_total"] == 0
    assert summary["blind_prediction_cases_met"] == 0
    assert summary["total_rows_considered"] == 14
    assert summary["total_rows_considered"] == (
        summary["not_applicable_rows"]
        + summary["reproducibility_cases_total"]
        + summary["blind_prediction_cases_total"]
    )
    # Not massaged: the reproducibility met-count must never exceed the
    # total (a tautological-looking guard that would catch a copy-paste
    # bug inflating "met" independently of "total").
    assert summary["reproducibility_cases_met"] <= summary["reproducibility_cases_total"]


def test_build_report_ustc_row_is_not_applicable_and_cites_ref_30() -> None:
    """The USTC DC-Stark budget row (Task B) is classified the same way
    as JILA's: in scope but not comparable, and its reason must name the
    follow-up-candidate reference explicitly (ref [30], Li J et al 2024
    Metrologia 61 015006) so a reviewer can find the next authorization
    target without re-reading the whole paper."""
    report = run_benchmarks.build_report(
        jila_fixture=_FIXTURES_DIR / "jila_2403_10664_table1.csv",
        nist_yb_fixture=_FIXTURES_DIR / "nist_m32206_yb_clock_phase_excerpt.csv",
        nist_10ghz_fixture=_FIXTURES_DIR / "nist_m32206_10ghz_phase_excerpt.csv",
    )
    (ustc_row,) = report["ustc_metrologia_63_025002"]

    assert ustc_row["in_engine_scope"] is True
    assert ustc_row["comparable"] is False
    assert ustc_row["kpi_verdict"] == "N/A"
    np.testing.assert_allclose(ustc_row["published_shift_fractional"], 0.0, atol=0)
    np.testing.assert_allclose(ustc_row["published_uncertainty_fractional"], 0.1e-19, atol=0)
    assert ustc_row["uncertainty_is_upper_bound"] is True
    assert "Li J et al 2024 Metrologia 61 015006" in ustc_row["reason"]
    assert "NOT" in ustc_row["reason"] and "fetched" in ustc_row["reason"]


def test_render_markdown_table_smoke() -> None:
    """`render_markdown_table` runs on a real report and includes every
    row's name and KPI verdict (the artifact a human reviewer reads), the
    NPL reproducibility case, and the USTC row -- and never uses the
    "PASS"/"FAIL" vocabulary this project reserves for nothing in WP10
    (budget rows use "N/A", the NPL case uses "MET"/"NOT MET")."""
    report = run_benchmarks.build_report(
        jila_fixture=_FIXTURES_DIR / "jila_2403_10664_table1.csv",
        nist_yb_fixture=_FIXTURES_DIR / "nist_m32206_yb_clock_phase_excerpt.csv",
        nist_10ghz_fixture=_FIXTURES_DIR / "nist_m32206_10ghz_phase_excerpt.csv",
    )
    markdown = run_benchmarks.render_markdown_table(report)

    assert "DC Stark" in markdown
    assert "BBR" in markdown
    assert "N/A" in markdown
    assert "NPL" in markdown
    assert "USTC" in markdown
    assert "MET" in markdown
    assert "PASS" not in markdown
    assert "FAIL" not in markdown


# ---------------------------------------------------------------------------
# loaders -- NPL/USTC follow-up data structures (Coordinator follow-up,
# 2026-08-10): AsymmetricMeasurement/PublishedBand parsing correctness and
# the exact published values, transcribed and independently re-verified
# from the fetched arXiv:1706.01944 text and the owner-provided USTC PDF.
# ---------------------------------------------------------------------------


def test_npl_residual_field_matches_arxiv_1706_01944_verbatim() -> None:
    """`loaders.NPL_RESIDUAL_FIELD_V_PER_M`: E = 1.52 V/m, stat
    +0.62/-0.22, sys +0.05/-0.03 (arXiv:1706.01944v1, Section IV) --
    pinned against a transcription error on this benchmark's one
    independently-sourced field input."""
    field = loaders.NPL_RESIDUAL_FIELD_V_PER_M
    np.testing.assert_allclose(field.nominal, 1.52, rtol=0, atol=1e-9)
    np.testing.assert_allclose(field.stat_lo, 0.22, rtol=0, atol=1e-9)
    np.testing.assert_allclose(field.stat_hi, 0.62, rtol=0, atol=1e-9)
    np.testing.assert_allclose(field.sys_lo, 0.03, rtol=0, atol=1e-9)
    np.testing.assert_allclose(field.sys_hi, 0.05, rtol=0, atol=1e-9)
    assert field.units == "V/m"
    assert "1706.01944" in field.citation


def test_asymmetric_measurement_combined_bounds_are_per_side_quadrature() -> None:
    """`AsymmetricMeasurement.combined_lo`/`combined_hi`: independent
    stat/sys combined in quadrature *per side*, never symmetrized between
    sides -- the precise, non-Gaussian-pretense method the WP10 follow-up
    instruction requires, verified against a hand-computed reference."""
    field = loaders.NPL_RESIDUAL_FIELD_V_PER_M
    expected_sigma_hi = (0.62**2 + 0.05**2) ** 0.5
    expected_sigma_lo = (0.22**2 + 0.03**2) ** 0.5

    np.testing.assert_allclose(
        field.combined_hi, field.nominal + expected_sigma_hi, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        field.combined_lo, field.nominal - expected_sigma_lo, rtol=1e-12, atol=0
    )
    # The two sides are NOT symmetric (stat/sys are themselves asymmetric),
    # so combined_hi - nominal must differ from nominal - combined_lo --
    # a test that would catch an accidental symmetrization bug.
    assert (field.combined_hi - field.nominal) != pytest.approx(field.nominal - field.combined_lo)


def test_npl_published_shift_matches_arxiv_1706_01944_verbatim() -> None:
    """`loaders.NPL_PUBLISHED_SHIFT`: -1.6 (+0.4/-1.6) x 10^-20
    (arXiv:1706.01944v1, Section IV) -- the band this project's
    reproducibility case is checked against."""
    published = loaders.NPL_PUBLISHED_SHIFT
    np.testing.assert_allclose(published.nominal, -1.6e-20, rtol=0, atol=1e-24)
    np.testing.assert_allclose(published.lo, -3.2e-20, rtol=0, atol=1e-24)
    np.testing.assert_allclose(published.hi, -1.2e-20, rtol=0, atol=1e-24)
    assert "1706.01944" in published.citation


def test_published_band_rejects_nominal_outside_bounds() -> None:
    """`PublishedBand.__post_init__` fails loudly on a malformed band
    (nominal outside [lo, hi]) rather than silently accepting nonsense --
    guards against a future transcription slip in this or a later WP."""
    with pytest.raises(ValueError, match="not within"):
        loaders.PublishedBand(nominal=5.0, lo=-1.0, hi=1.0, units="x", citation="test")


def test_ustc_dc_stark_constraint_matches_metrologia_63_025002_verbatim() -> None:
    """`loaders.USTC_DC_STARK_CONSTRAINT`: Table 3 "DC Stark" row = 0
    (<0.1) x 10^-19 (Jia et al., Metrologia 63, 025002 (2026)) -- own
    independent re-verification from the owner-provided PDF, page 10
    (printed page 9), Sec. 3.5 + Table 3 (printed page 10)."""
    entry = loaders.USTC_DC_STARK_CONSTRAINT
    np.testing.assert_allclose(entry.shift_e19, 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(entry.uncertainty_e19, 0.1, rtol=0, atol=1e-9)
    assert entry.uncertainty_is_upper_bound is True
    assert entry.in_engine_scope is True


# ---------------------------------------------------------------------------
# run_benchmarks -- _bands_overlap (interval-overlap test, precisely
# defined per the coordinator's follow-up instruction: "define precisely
# and document"). Two closed intervals [lo1, hi1]/[lo2, hi2] overlap iff
# lo1 <= hi2 and lo2 <= hi1.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lo1", "hi1", "lo2", "hi2", "expected"),
    [
        # Identical intervals.
        (0.0, 1.0, 0.0, 1.0, True),
        # Partial overlap, either order.
        (0.0, 2.0, 1.0, 3.0, True),
        (1.0, 3.0, 0.0, 2.0, True),
        # One fully contains the other.
        (0.0, 10.0, 2.0, 3.0, True),
        (2.0, 3.0, 0.0, 10.0, True),
        # Touching at a single point counts as overlap (closed intervals).
        (0.0, 1.0, 1.0, 2.0, True),
        (1.0, 2.0, 0.0, 1.0, True),
        # Disjoint, either order.
        (0.0, 1.0, 2.0, 3.0, False),
        (2.0, 3.0, 0.0, 1.0, False),
        # Negative-valued bands (the actual NPL/JILA use case).
        (-3.29e-20, -1.208e-20, -3.2e-20, -1.2e-20, True),
        (-5.0e-20, -4.0e-20, -3.2e-20, -1.2e-20, False),
    ],
)
def test_bands_overlap_precise_definition(
    lo1: float, hi1: float, lo2: float, hi2: float, expected: bool
) -> None:
    assert run_benchmarks._bands_overlap(lo1, hi1, lo2, hi2) is expected


def test_bands_overlap_rejects_malformed_interval() -> None:
    """A caller-supplied ``lo > hi`` interval is a bug, not silently
    handled -- catches an accidentally-swapped-bound error early."""
    with pytest.raises(AssertionError):
        run_benchmarks._bands_overlap(1.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# run_benchmarks -- the NPL reproducibility case (Task A step 4: "the
# reproducibility assertion"). Exercises three real pipeline calls (not
# mocked) at the field's combined asymmetric bounds, per WP10's labeling
# discipline (no tuning, no algebraic shortcut standing in for an actual run).
# ---------------------------------------------------------------------------


def test_run_npl_reproducibility_case_bands_overlap_and_verdict_met() -> None:
    """The core reproducibility assertion: the predicted band (from three
    real ``coupling.type: stark_dc`` pipeline runs at NPL's field bounds)
    overlaps NPL's own published shift band, so `kpi_verdict == "MET"`.

    Precise definition of "overlap" (never "every digit matches"):
    :func:`run_benchmarks._bands_overlap`, a closed-interval overlap test.
    This is the case's entire pass/fail criterion -- documented here and
    in `run_npl_reproducibility_case`'s docstring, not implicit.
    """
    case = run_benchmarks.run_npl_reproducibility_case()

    assert case.case_class == "reproducibility"
    assert case.bands_overlap is True
    assert case.kpi_verdict == "MET"

    # The predicted band must itself be internally ordered (lo <= nominal
    # <= hi in shift-space, i.e. "more negative" to "less negative") --
    # already asserted at runtime inside run_npl_reproducibility_case,
    # re-checked here as a black-box property of the returned case.
    assert case.predicted_shift_lo <= case.predicted_shift_nominal <= case.predicted_shift_hi

    # Published band, transcribed verbatim (see test_npl_published_shift_
    # matches_arxiv_1706_01944_verbatim for the citation-level pin).
    np.testing.assert_allclose(case.published_shift_lo, -3.2e-20, rtol=0, atol=1e-24)
    np.testing.assert_allclose(case.published_shift_nominal, -1.6e-20, rtol=0, atol=1e-24)
    np.testing.assert_allclose(case.published_shift_hi, -1.2e-20, rtol=0, atol=1e-24)

    # Predicted nominal must be close to NPL's own nominal (both derived
    # from the same Middelmann Delta_alpha and the same nominal field) --
    # loose rtol because NPL's "-1.6e-20" is itself rounded to 2 sig figs.
    np.testing.assert_allclose(case.predicted_shift_nominal, -1.6e-20, rtol=0.05, atol=0)

    # Non-vacuous: the predicted band must actually be a band, not a
    # degenerate point (would trivially "overlap" almost anything).
    assert case.predicted_shift_lo < case.predicted_shift_hi


def test_run_npl_reproducibility_case_uses_registry_species_delta_alpha() -> None:
    """The case must resolve its coupling coefficient from the *species
    registry* (Middelmann-sourced, same as KA1/KA2), not a hand-entered
    constant -- so a future registry update automatically flows through."""
    from cliffordclock.ensemble.species import get_species

    case = run_benchmarks.run_npl_reproducibility_case()
    sr87 = get_species("Sr87")

    assert case.species_name == "Sr87"
    np.testing.assert_allclose(case.delta_alpha_dc_si, sr87.delta_alpha_dc_si, rtol=1e-15, atol=0)
    assert "Middelmann" in case.delta_alpha_citation


def test_run_npl_reproducibility_case_field_bounds_from_combined_uncertainty() -> None:
    """The pipeline is run at `AsymmetricMeasurement.combined_lo`/
    `combined_hi`, not at the raw stat-only or sys-only bounds -- pins the
    exact field magnitudes fed to the pipeline against
    `benchmarks/loaders.NPL_RESIDUAL_FIELD_V_PER_M`."""
    case = run_benchmarks.run_npl_reproducibility_case()
    field = loaders.NPL_RESIDUAL_FIELD_V_PER_M

    np.testing.assert_allclose(case.field_lo_v_per_m, field.combined_lo, rtol=1e-12, atol=0)
    np.testing.assert_allclose(case.field_nominal_v_per_m, field.nominal, rtol=0, atol=0)
    np.testing.assert_allclose(case.field_hi_v_per_m, field.combined_hi, rtol=1e-12, atol=0)
    assert case.field_lo_v_per_m < case.field_nominal_v_per_m < case.field_hi_v_per_m


def test_run_npl_reproducibility_case_rotor_bands_overlap_and_verdict_met() -> None:
    """WP16: the same reproducibility assertion as
    `test_run_npl_reproducibility_case_bands_overlap_and_verdict_met`, but
    for the rotor-path re-run (`integration.mode='worldline'`, the true
    Cl(1,3) rotor via `cliffordclock.pipeline._stark_rotor_ensemble`) --
    same published band, same overlap criterion
    (`run_benchmarks._bands_overlap`), same `kpi_verdict` vocabulary
    (`"MET"`/`"NOT MET"`, never `"PASS"`/`"FAIL"`)."""
    case = run_benchmarks.run_npl_reproducibility_case_rotor()

    assert case.case_class == "reproducibility"
    assert case.bands_overlap is True
    assert case.kpi_verdict == "MET"
    assert case.predicted_shift_lo <= case.predicted_shift_nominal <= case.predicted_shift_hi

    np.testing.assert_allclose(case.published_shift_lo, -3.2e-20, rtol=0, atol=1e-24)
    np.testing.assert_allclose(case.published_shift_nominal, -1.6e-20, rtol=0, atol=1e-24)
    np.testing.assert_allclose(case.published_shift_hi, -1.2e-20, rtol=0, atol=1e-24)
    np.testing.assert_allclose(case.predicted_shift_nominal, -1.6e-20, rtol=0.05, atol=0)
    assert case.predicted_shift_lo < case.predicted_shift_hi


def test_run_npl_reproducibility_case_rotor_matches_scalar_case_to_float64_precision() -> None:
    """The direct WP16 cross-check: the rotor-path re-run's predicted band
    must match the scalar (fast-path) case's predicted band to essentially
    float64 precision -- both accumulate the *same* E14b pivot over the
    *same* (uniform-field, single static quadrature node, v=0) case, where
    Omega is exactly constant across every step (see
    `run_benchmarks._NPL_ROTOR_DTAU`'s docstring), so any residual
    difference is pure floating-point rounding, not a physics
    discrepancy. `rtol=1e-10` is deliberately loose relative to the
    observed agreement (~1e-16 relative, i.e. float64 noise) -- it is a
    "these are the same physics, not coincidentally close" gate, not a
    precision claim; the exact observed agreement is far tighter than
    this bound requires."""
    scalar_case = run_benchmarks.run_npl_reproducibility_case()
    rotor_case = run_benchmarks.run_npl_reproducibility_case_rotor()

    assert rotor_case.kpi_verdict == scalar_case.kpi_verdict
    assert rotor_case.bands_overlap == scalar_case.bands_overlap
    np.testing.assert_allclose(
        rotor_case.predicted_shift_lo, scalar_case.predicted_shift_lo, rtol=1e-10, atol=0
    )
    np.testing.assert_allclose(
        rotor_case.predicted_shift_nominal,
        scalar_case.predicted_shift_nominal,
        rtol=1e-10,
        atol=0,
    )
    np.testing.assert_allclose(
        rotor_case.predicted_shift_hi, scalar_case.predicted_shift_hi, rtol=1e-10, atol=0
    )
    # Field bounds and species provenance are identical inputs regardless
    # of accumulator -- pinned here so a future edit that accidentally
    # changes the rotor case's inputs (not just its integration mode) is
    # caught.
    np.testing.assert_allclose(
        rotor_case.field_lo_v_per_m, scalar_case.field_lo_v_per_m, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        rotor_case.field_hi_v_per_m, scalar_case.field_hi_v_per_m, rtol=0, atol=0
    )
    assert rotor_case.species_name == scalar_case.species_name


def test_build_report_includes_rotor_crosscheck_without_double_counting() -> None:
    """WP16: `build_report`'s rotor-path re-run is informational -- present
    in the report, verdict matching the scalar case -- but must NOT be
    counted a second time in `kpi_summary`'s totals (it is the same case
    re-run through a different accumulator, not an independent
    reproducibility case; see `build_report`'s docstring)."""
    report = run_benchmarks.build_report(
        jila_fixture=_FIXTURES_DIR / "jila_2403_10664_table1.csv",
        nist_yb_fixture=_FIXTURES_DIR / "nist_m32206_yb_clock_phase_excerpt.csv",
        nist_10ghz_fixture=_FIXTURES_DIR / "nist_m32206_10ghz_phase_excerpt.csv",
    )

    rotor_block = report["npl_1706_01944_reproducibility_case_rotor_crosscheck"]
    assert rotor_block["case"]["kpi_verdict"] == "MET"
    assert rotor_block["bands_overlap_and_verdict_match_scalar_case"] is True

    # Unchanged from test_build_report_smoke's counts: the rotor re-run
    # does not inflate reproducibility_cases_total/total_rows_considered.
    summary = report["kpi_summary"]
    assert summary["reproducibility_cases_total"] == 1
    assert summary["total_rows_considered"] == 14


def test_classify_ustc_dc_stark_smoke() -> None:
    """`classify_ustc_dc_stark` (Task B): same not-comparable/N/A shape as
    the JILA rows, referencing the exact USTC citation."""
    verdict = run_benchmarks.classify_ustc_dc_stark()

    assert verdict.comparable is False
    assert verdict.kpi_verdict == "N/A"
    assert verdict.in_engine_scope is True
    assert "Metrologia 63" in verdict.citation
    assert "025002" in verdict.citation


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
