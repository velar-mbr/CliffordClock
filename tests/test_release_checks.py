# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for tools/release_checks.py (WP28).

``tools/`` is a standalone script directory outside the ``cliffordclock``
package (like ``benchmarks/``, see ``tests/test_benchmarks_loaders.py``), so
it is added to ``sys.path`` directly rather than imported through the normal
``cliffordclock.*`` namespace.

Every scanner gets fixture-string positive AND negative cases, including a
planted-violation case per WP28's instruction ("a scanner that can't catch a
planted violation is the tool failing its own purpose").
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import release_checks as rc  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# prose-scan: em dash / dash-as-punctuation
# ---------------------------------------------------------------------------


def test_prose_scan_catches_unicode_em_dash_planted_violation():
    text = "This sentence has an em dash — right there in the middle."
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert any("em dash" in f.message for f in findings)
    assert all(f.severity == "FAIL" for f in findings if "em dash" in f.message)


def test_prose_scan_no_em_dash_negative_case():
    text = "This sentence has no dash of any kind."
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert findings == []


def test_prose_scan_catches_ascii_dash_as_punctuation_planted_violation():
    text = "Two things happened -- and neither was expected."
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert any("dash-as-punctuation" in f.message for f in findings)


def test_prose_scan_allowlists_numeric_ranges():
    # LaTeX/markdown numeric-range convention: digit--digit, no spaces.
    text = "The fit is valid over 50--350 K."
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert findings == []


def test_prose_scan_ignores_markdown_table_separator_rows():
    text = "| Quantity | Value |\n|---|---|\n| a | b |\n"
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert findings == []


def test_prose_scan_ignores_horizontal_rule():
    text = "Some text.\n\n---\n\nMore text.\n"
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert findings == []


def test_prose_scan_strips_inline_code_spans_no_false_positive_on_cli_flags():
    text = "Run it with `--fast` to skip the slow checks."
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert findings == []


def test_prose_scan_strips_fenced_code_blocks():
    text = "Prose above.\n\n```\nsome --code-- with fake dashes\n```\n\nProse below.\n"
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert findings == []


def test_prose_scan_allowlist_suppresses_deliberate_keep():
    text = "Two things happened -- and neither was expected."
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=[],
        meta_slop_minor=[],
        allowed=["Two things happened -- and neither was expected."],
    )
    assert findings == []


# ---------------------------------------------------------------------------
# prose-scan: honest-family words
# ---------------------------------------------------------------------------


def test_prose_scan_catches_honest_family_word_planted_violation():
    text = "The validation status is honestly stated here."
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert any("honest-family" in f.message for f in findings)


def test_prose_scan_honest_family_negative_case():
    text = "The validation status is stated clearly here."
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert findings == []


def test_prose_scan_honest_family_no_false_positive_on_substring():
    # "dishonest" must not trigger the \bhonest\b word-boundary match.
    text = "That would be a dishonestly framed comparison, which we avoid."
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert findings == []


# ---------------------------------------------------------------------------
# prose-scan: meta-slop phrases (configurable, fatal vs. minor)
# ---------------------------------------------------------------------------


def test_prose_scan_catches_fatal_meta_slop_phrase():
    text = "It is worth noting that this result agrees with theory."
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=["it is worth noting"],
        meta_slop_minor=[],
        allowed=[],
    )
    assert len(findings) == 1
    assert findings[0].severity == "FAIL"


def test_prose_scan_minor_meta_slop_phrase_is_not_fatal():
    text = "We used method A rather than method B for this case."
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=[],
        meta_slop_minor=["rather than"],
        allowed=[],
    )
    assert len(findings) == 1
    assert findings[0].severity == "MINOR"
    assert rc._status_from_findings(findings) == "PASS"


def test_prose_scan_meta_slop_negative_case():
    text = "This paragraph contains none of the configured phrases."
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=["it is worth noting"],
        meta_slop_minor=["rather than"],
        allowed=[],
    )
    assert findings == []


# ---------------------------------------------------------------------------
# tolerance-scan
# ---------------------------------------------------------------------------


def test_tolerance_scan_catches_bare_approx_rel_without_abs_planted_violation():
    text = "assert result == pytest.approx(expected, rel=1e-6)\n"
    findings = rc._scan_tolerance_text("fixture_test.py", text)
    assert len(findings) == 1
    assert "rel=" in findings[0].message
    assert findings[0].line == 1


def test_tolerance_scan_approx_with_abs_is_clean():
    text = "assert result == pytest.approx(expected, rel=1e-6, abs=0)\n"
    findings = rc._scan_tolerance_text("fixture_test.py", text)
    assert findings == []


def test_tolerance_scan_approx_without_rel_is_not_flagged():
    # Spec targets the specific "bare approx(..., rel=...) without abs="
    # pattern; a plain approx(x) call with no kwargs at all is out of scope.
    text = "assert result == pytest.approx(expected)\n"
    findings = rc._scan_tolerance_text("fixture_test.py", text)
    assert findings == []


def test_tolerance_scan_catches_assert_allclose_missing_atol_planted_violation():
    text = "np.testing.assert_allclose(actual, expected, rtol=1e-8)\n"
    findings = rc._scan_tolerance_text("fixture_test.py", text)
    assert len(findings) == 1
    assert "atol=" in findings[0].message


def test_tolerance_scan_assert_allclose_with_atol_is_clean():
    text = "np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=0)\n"
    findings = rc._scan_tolerance_text("fixture_test.py", text)
    assert findings == []


def test_tolerance_scan_reports_correct_line_number():
    text = "line1\nline2\nassert x == pytest.approx(y, rel=1e-3)\n"
    findings = rc._scan_tolerance_text("fixture_test.py", text)
    assert findings[0].line == 3


def test_tolerance_scan_allowlist_skips_exact_call_text_only():
    """An allowlisted call text is skipped; any other violation in the same
    text still surfaces (the allowlist is per-call, not per-file)."""
    text = (
        "assert a == pytest.approx(expected, rel=1e-6)\n"
        "assert b == pytest.approx(other, rel=1e-6)\n"
    )
    allowed = ["pytest.approx(expected, rel=1e-6)"]
    findings = rc._scan_tolerance_text("fixture_test.py", text, allowed)
    assert len(findings) == 1
    assert findings[0].line == 2


def test_tolerance_scan_runs_clean_against_current_repo():
    """The real tests/ tree, with the real allowlist, has zero findings
    (the planted-violation fixtures in this file are allowlisted; every
    other tolerance has been made explicit)."""
    result = rc.tolerance_scan(rc.load_allowlist())
    assert result.status == "PASS", [f.format() for f in result.findings]


# ---------------------------------------------------------------------------
# citation-check
# ---------------------------------------------------------------------------


def test_citation_check_catches_wrong_year_planted_violation():
    surname_years = {"Roos": {2006}}
    text = "Roos et al. (2099) measured the two-ion quadrupole slope."
    findings = rc._scan_citations_in_text("fixture.md", text, surname_years)
    assert len(findings) == 1
    assert "2099" in findings[0].message
    assert "2006" in findings[0].message


def test_citation_check_correct_year_is_clean():
    surname_years = {"Roos": {2006}}
    text = "Roos et al. (2006) measured the two-ion quadrupole slope."
    findings = rc._scan_citations_in_text("fixture.md", text, surname_years)
    assert findings == []


def test_citation_check_allows_any_pinned_year_for_a_shared_surname():
    # A surname legitimately spans multiple distinct papers/years.
    surname_years = {"Ye": {2015, 2022, 2024}}
    text = "Ye and coauthors (2022) reported the redshift measurement."
    findings = rc._scan_citations_in_text("fixture.md", text, surname_years)
    assert findings == []


def test_citation_check_ignores_unknown_surname():
    surname_years = {"Roos": {2006}}
    text = "Smith et al. (1999) is not a pinned citation at all."
    findings = rc._scan_citations_in_text("fixture.md", text, surname_years)
    assert findings == []


def test_citation_check_ignores_surname_with_no_nearby_year():
    surname_years = {"Roos": {2006}}
    text = "Roos designed a clever ion-trap sequence for this experiment."
    findings = rc._scan_citations_in_text("fixture.md", text, surname_years)
    assert findings == []


def test_citation_check_reports_correct_line_number():
    surname_years = {"Bothwell": {2022}}
    text = "line one\nline two\nBothwell et al. (1900) is wrong here.\n"
    findings = rc._scan_citations_in_text("fixture.md", text, surname_years)
    assert findings[0].line == 3


def test_citation_check_does_not_mistake_page_number_digits_for_a_year():
    # Regression: "020502(R)" contains the substring "2050", which looks
    # like a (19|20)dd year to a naive scan but is a PRA eid/page number.
    surname_years = {"Porsev": {2006, 2011}}
    text = "Porsev & Derevianko PRA 74, 020502(R) (2006) with its 2012 erratum."
    findings = rc._scan_citations_in_text("fixture.md", text, surname_years)
    assert findings == []


def test_citation_check_does_not_mistake_iso_date_for_a_year():
    # Regression: this repo's own session-date convention ("2026-08-11")
    # sits right next to citations in benchmarks/SOURCES.md-style prose and
    # must not be read as a contradicting publication year.
    surname_years = {"Roos": {2006}}
    text = "Roos et al., the two-ion quadrupole-shift benchmark (owner-supplied, 2026-08-11)"
    findings = rc._scan_citations_in_text("fixture.md", text, surname_years)
    assert findings == []


# ---------------------------------------------------------------------------
# headline-check regexes
# ---------------------------------------------------------------------------


def test_headline_two_repro_regex_positive():
    assert rc.TWO_REPRO_RE.search("the pipeline carries **two reproducibility cases**, zero blind")


def test_headline_two_repro_regex_negative():
    assert rc.TWO_REPRO_RE.search("the pipeline carries one reproducibility case") is None


def test_headline_zero_blind_regex_positive():
    assert rc.ZERO_BLIND_RE.search("Zero blind predictions, unchanged.")


def test_headline_zero_blind_regex_negative():
    assert rc.ZERO_BLIND_RE.search("one blind prediction exists") is None


def test_headline_stale_phrase_regex_planted_violation():
    assert rc.STALE_HEADLINE_RE.search("this pass found 1 of 1 reproducibility cases met")
    assert rc.STALE_HEADLINE_RE.search("only one reproducibility case has been checked")


def test_headline_stale_phrase_regex_negative():
    assert rc.STALE_HEADLINE_RE.search("the project now has two reproducibility cases") is None


def test_headline_check_runs_clean_against_current_repo():
    # Smoke test: the real README/docs/validation.md/benchmarks/RESULTS.md
    # should already state the ratified two-reproducibility-case headline
    # consistently (WP22/WP23 per the project's sprint-4 planning record).
    allowlist = rc.load_allowlist()
    result = rc.headline_check(allowlist)
    assert result.status == "PASS", [f.format() for f in result.findings]


# ---------------------------------------------------------------------------
# internal-path-check
# ---------------------------------------------------------------------------


def test_internal_path_regex_catches_plan_reference_planted_violation():
    m = rc.INTERNAL_PATH_RE.search("see `plan/notes/ion-clock-dossier.md` section 6.")
    assert m is not None
    assert m.group(0) == "plan/notes/ion-clock-dossier.md"


def test_internal_path_regex_catches_internal_reference_planted_violation():
    text = "ratified in internal/signoffs/G7_physics_signoff_theory.md B5."
    m = rc.INTERNAL_PATH_RE.search(text)
    assert m is not None
    assert m.group(0) == "internal/signoffs/G7_physics_signoff_theory.md"


def test_internal_path_regex_negative_case():
    text = "the project's theory sign-off record (G7) B5 ratified this label."
    assert rc.INTERNAL_PATH_RE.search(text) is None


def test_internal_path_regex_no_false_positive_on_deeper_path_segment():
    # A "plan"/"internal" directory nested inside an unrelated, legitimate
    # path (e.g. a vendored third-party subpackage) must not be mistaken
    # for a reference to THIS project's own private top-level directory.
    text = "see vendor/internal/utils.py or build/plan/cache.json for detail."
    assert rc.INTERNAL_PATH_RE.search(text) is None


def test_internal_path_regex_no_false_positive_on_bare_word():
    # "internal"/"plan" used as ordinary English words, with no following
    # slash, are not path references.
    text = "this is an internal detail we plan to document later."
    assert rc.INTERNAL_PATH_RE.search(text) is None


def test_scan_internal_paths_in_text_catches_planted_violation():
    text = "Provenance: `plan/notes/bbr-formalism-dossier.md` section 4/7."
    findings = rc._scan_internal_paths_in_text("fixture.md", text, allowed=[])
    assert len(findings) == 1
    assert "plan/notes/bbr-formalism-dossier.md" in findings[0].message


def test_scan_internal_paths_in_text_clean_case():
    text = "Provenance: Lisdat et al., PRR 3, L042036 (2021)."
    findings = rc._scan_internal_paths_in_text("fixture.md", text, allowed=[])
    assert findings == []


def test_scan_internal_paths_in_text_respects_allowlist():
    text = "Provenance: `plan/notes/bbr-formalism-dossier.md` section 4/7."
    findings = rc._scan_internal_paths_in_text(
        "fixture.md", text, allowed=["Provenance: `plan/notes/bbr-formalism-dossier.md`"]
    )
    assert findings == []


def test_scan_internal_paths_in_text_reports_correct_line_number():
    text = "line one\nline two\nsee internal/signoffs/G8_physics_signoff_theory.md here.\n"
    findings = rc._scan_internal_paths_in_text("fixture.md", text, allowed=[])
    assert findings[0].line == 3


def test_scan_internal_paths_in_text_applies_line_offset():
    text = "see plan/notes/ion-clock-dossier.md section 6.\n"
    findings = rc._scan_internal_paths_in_text("fixture.md", text, allowed=[], line_offset=99)
    assert findings[0].line == 100


def test_changelog_unreleased_span_excludes_older_history():
    text = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Added\n"
        "- references plan/notes/foo-dossier.md here.\n\n"
        "## 0.1.0.dev0 -- initial release\n\n"
        "### Added\n"
        "- an older entry that references plan/notes/bar-dossier.md too.\n"
    )
    span = rc._changelog_unreleased_span(text)
    assert span is not None
    start, end = span
    scanned = text[start:end]
    assert "plan/notes/foo-dossier.md" in scanned
    assert "plan/notes/bar-dossier.md" not in scanned


def test_changelog_unreleased_span_returns_none_without_unreleased_heading():
    text = "# Changelog\n\n## 0.1.0.dev0 -- initial release\n\nNo Unreleased section here.\n"
    assert rc._changelog_unreleased_span(text) is None


def test_internal_path_check_runs_clean_against_current_repo():
    # Smoke test: after the internal-path scrub, no public file should
    # reference a plan/ or internal/ path (regenerated benchmark results
    # and re-executed notebooks included).
    allowlist = rc.load_allowlist()
    result = rc.internal_path_check(allowlist)
    assert result.status == "PASS", [f.format() for f in result.findings]


def test_public_tracked_files_excludes_private_directories():
    files = {str(p.relative_to(REPO_ROOT)) for p in rc._public_tracked_files()}
    assert not any(f.startswith("plan/") for f in files)
    assert not any(f.startswith("internal/") for f in files)
    assert not any(f.startswith(".claude/") for f in files)
    # Sanity: real public files ARE included.
    assert "README.md" in files
    assert "CHANGELOG.md" in files


# ---------------------------------------------------------------------------
# JSON / notebook normalization (determinism-check, notebooks-check)
# ---------------------------------------------------------------------------


def test_normalize_json_strips_ignored_timestamp_key():
    a = {"generated_at_utc": "2026-08-11T00:00:00Z", "value": 1}
    b = {"generated_at_utc": "2026-08-12T00:00:00Z", "value": 1}
    assert rc._normalize_json(a) == rc._normalize_json(b)


def test_normalize_json_catches_real_value_difference_planted_violation():
    a = {"generated_at_utc": "2026-08-11T00:00:00Z", "value": 1}
    b = {"generated_at_utc": "2026-08-11T00:00:00Z", "value": 2}
    assert rc._normalize_json(a) != rc._normalize_json(b)


def test_normalize_json_handles_nested_structures():
    a = {"outer": {"generated_at_utc": "x", "inner": [1, 2, {"generated_at_utc": "y", "z": 3}]}}
    b = {
        "outer": {
            "generated_at_utc": "different",
            "inner": [1, 2, {"generated_at_utc": "also-different", "z": 3}],
        }
    }
    assert rc._normalize_json(a) == rc._normalize_json(b)


def _minimal_notebook(source: str, outputs: list[dict]) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "source": [source],
                "execution_count": 1,
                "outputs": outputs,
                "metadata": {},
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_diff_notebook_outputs_identical_is_none():
    nb = _minimal_notebook(
        "1 + 1", [{"output_type": "execute_result", "data": {"text/plain": ["2"]}}]
    )
    assert rc._diff_notebook_outputs(nb, nb) is None


def test_diff_notebook_outputs_ignores_execution_count():
    a = _minimal_notebook(
        "1 + 1", [{"output_type": "execute_result", "data": {"text/plain": ["2"]}}]
    )
    b = json.loads(json.dumps(a))
    a["cells"][0]["execution_count"] = 1
    b["cells"][0]["execution_count"] = 7
    assert rc._diff_notebook_outputs(a, b) is None


def test_diff_notebook_outputs_catches_output_mismatch_planted_violation():
    a = _minimal_notebook(
        "1 + 1", [{"output_type": "execute_result", "data": {"text/plain": ["2"]}}]
    )
    b = _minimal_notebook(
        "1 + 1", [{"output_type": "execute_result", "data": {"text/plain": ["99"]}}]
    )
    diff = rc._diff_notebook_outputs(a, b)
    assert diff is not None
    assert "cell 0" in diff


def test_diff_notebook_outputs_catches_cell_count_mismatch():
    a = _minimal_notebook("1 + 1", [])
    b = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    diff = rc._diff_notebook_outputs(a, b)
    assert diff is not None
    assert "cell count" in diff


def test_diff_notebook_outputs_volatile_pattern_masks_timing_line_only():
    """A wall-clock timing line matched by a volatile pattern compares
    equal across runs, while a physics number differing in the SAME
    stream output still fails (list-of-lines and plain-string text
    fields alike)."""
    volatile = [re.compile(r"wall time\s*:\s*[0-9][0-9.]* s")]
    text_a = ["wall time : 0.495 s\n", "shift : -7.72e-19\n"]
    text_b = "wall time : 1.203 s\nshift : -7.72e-19\n"
    a = _minimal_notebook("run()", [{"output_type": "stream", "name": "stdout", "text": text_a}])
    b = _minimal_notebook("run()", [{"output_type": "stream", "name": "stdout", "text": text_b}])
    assert rc._diff_notebook_outputs(a, b, volatile) is None
    # Without the pattern, the timing difference is caught.
    assert rc._diff_notebook_outputs(a, b) is not None
    # With the pattern, a real physics-number drift is still caught.
    c = json.loads(json.dumps(b))
    c["cells"][0]["outputs"][0]["text"] = "wall time : 1.203 s\nshift : -9.99e-19\n"
    diff = rc._diff_notebook_outputs(a, c, volatile)
    assert diff is not None
    assert "cell 0" in diff


def test_notebooks_check_allowlist_volatile_patterns_compile():
    """The shipped allowlist's volatile patterns are valid regexes and
    match notebook 02's actual wall-time line format."""
    patterns = rc.load_allowlist().get("notebooks_check", {}).get("volatile_patterns", [])
    assert patterns, "expected at least the wall-time pattern"
    compiled = [re.compile(p) for p in patterns]
    line = "wall time                : 0.495 s  (documented bound: < 60 s CPU)"
    assert any(p.search(line) for p in compiled)


# ---------------------------------------------------------------------------
# suite-check summary-line parsing
# ---------------------------------------------------------------------------


def test_pytest_summary_regex_extracts_counts():
    line = "===== 512 passed, 2 skipped in 34.21s ====="
    m = rc.PYTEST_SUMMARY_RE.search(line)
    assert m is not None
    assert "512 passed" in m.group("counts")
    assert "2 skipped" in m.group("counts")


def test_ruff_error_count_regex_extracts_count_planted_violation():
    m = rc.RUFF_ERROR_COUNT_RE.search("Found 3 errors.")
    assert m is not None
    assert m.group(1) == "3"


def test_mypy_error_count_regex_extracts_count():
    m = rc.MYPY_ERROR_COUNT_RE.search("Found 5 errors in 2 files (checked 40 source files)")
    assert m is not None
    assert m.group(1) == "5"


# ---------------------------------------------------------------------------
# Configuration loading (real files: tools/bibliography.toml, allowlist)
# ---------------------------------------------------------------------------


def test_load_bibliography_returns_nonempty_well_formed_entries():
    papers = rc.load_bibliography()
    assert len(papers) > 0
    required = {"key", "authors", "surnames", "title", "venue", "year", "source"}
    for paper in papers:
        missing = required - set(paper)
        assert not missing, f"{paper.get('key')} missing fields: {missing}"
        assert isinstance(paper["surnames"], list)
        assert isinstance(paper["year"], list)


def test_load_bibliography_keys_are_unique():
    papers = rc.load_bibliography()
    keys = [p["key"] for p in papers]
    assert len(keys) == len(set(keys))


def test_surname_years_merges_across_entries_for_shared_surnames():
    papers = rc.load_bibliography()
    surname_years = rc._surname_years(papers)
    # "Ye" is a real co-author on more than one pinned paper (Aeppli2024,
    # Bothwell2022, Ludlow2015) -- the merged year set must include all of them,
    # not silo per-entry (a silo would make a legitimate second-paper
    # citation look like a fabricated byline).
    assert "Ye" in surname_years
    assert len(surname_years["Ye"]) >= 2


def test_load_allowlist_has_expected_sections():
    allowlist = rc.load_allowlist()
    assert "fatal" in allowlist.get("meta_slop", {})
    assert "rather than" in allowlist["meta_slop"]["minor"]
    assert "benchmarks/RESULTS.md" in allowlist.get("headline_check", {}).get(
        "frozen_artifacts", []
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_checks_registry_has_all_eight_wp28_subcommands():
    expected = {
        "prose-scan",
        "tolerance-scan",
        "citation-check",
        "headline-check",
        "internal-path-check",
        "determinism-check",
        "notebooks-check",
        "suite-check",
    }
    assert set(rc.CHECKS) == expected


def test_fast_skip_set_matches_checks_6_through_8():
    assert {"determinism-check", "notebooks-check", "suite-check"} == rc.FAST_SKIP


def test_main_list_exits_zero(capsys):
    exit_code = rc.main(["--list"])
    assert exit_code == 0
    out = capsys.readouterr().out
    for name in rc.CHECKS:
        assert name in out


def test_main_rejects_unknown_only_value():
    with pytest.raises(SystemExit):
        rc.main(["--only", "not-a-real-check"])


def test_main_only_runs_requested_checks(capsys, monkeypatch):
    # Stub out the two cheap, filesystem-only checks so this test is fast
    # and independent of the current repo's prose content.
    calls: list[str] = []

    def fake_prose_scan(allowlist):
        calls.append("prose-scan")
        return rc.CheckResult("prose-scan", "PASS", [])

    def fake_tolerance_scan(allowlist):
        calls.append("tolerance-scan")
        return rc.CheckResult("tolerance-scan", "PASS", [])

    monkeypatch.setitem(rc.CHECKS, "prose-scan", fake_prose_scan)
    monkeypatch.setitem(rc.CHECKS, "tolerance-scan", fake_tolerance_scan)

    exit_code = rc.main(["--only", "prose-scan,tolerance-scan"])
    assert exit_code == 0
    assert calls == ["prose-scan", "tolerance-scan"]


def test_main_exit_code_nonzero_on_fail(monkeypatch):
    def failing_check(allowlist):
        return rc.CheckResult("prose-scan", "FAIL", [rc.Finding("x.md", 1, "planted failure")])

    monkeypatch.setitem(rc.CHECKS, "prose-scan", failing_check)
    exit_code = rc.main(["--only", "prose-scan"])
    assert exit_code == 1


# ---------------------------------------------------------------------------
# Integration: determinism-check must never leave the working tree modified
# ---------------------------------------------------------------------------


def test_determinism_check_restores_committed_files_regardless_of_outcome():
    # Snapshot the WHOLE benchmarks/results/ directory, not just the pinned
    # .json targets -- each regeneration script also rewrites a sibling .md
    # summary as a side effect, and a real bug here once left those .md
    # files modified after a determinism-check run (caught by inspecting
    # `git status` after running this test manually).
    results_dir = REPO_ROOT / "benchmarks" / "results"
    before = {p: p.read_bytes() for p in results_dir.rglob("*") if p.is_file()}

    result = rc.determinism_check(rc.load_allowlist())

    assert result.status in {"PASS", "FAIL"}
    after = {p: p.read_bytes() for p in results_dir.rglob("*") if p.is_file()}
    assert after == before, "determinism-check left benchmarks/results/ modified"


# ---------------------------------------------------------------------------
# Wave-review blocker fixes (Sprint 4 day-one integration): regression tests.
# ---------------------------------------------------------------------------


def test_prose_scan_strips_multiline_inline_code_span():
    """A single-backtick span opened on one line and closed on the next
    (a wrapped CLI command) must not fire dash findings from its interior
    (wave-review blocker 4)."""
    text = (
        "Run `cliffordclock run benchmarks/beta_case_x/config.yaml --output-dir\n"
        "/tmp/out` and inspect the report.\n"
    )
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert findings == []


def test_prose_scan_strips_markdown_link_targets():
    """Badge/link URL targets (status-pre--beta) are addresses, not prose."""
    text = "![status: pre-beta](https://img.shields.io/badge/status-pre--beta-orange)\n"
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert findings == []


def test_year_regex_full_iso_guard_keeps_hyphenated_year_ranges():
    """The ISO-date guard must match the full -MM-DD shape only, so a
    hyphenated year-range citation still yields its years (blocker 3)."""
    assert rc.YEAR_RE.findall("Roos (2006-2019) revisited") == ["2006", "2019"]
    assert rc.YEAR_RE.findall("retrieved 2026-08-11") == []
    assert rc.YEAR_RE.findall("PRA 74, 020502(R) (2006)") == ["2006"]


def test_latex_en_dash_typography_not_flagged():
    """Name pairs and math-mode ranges are typography, not prose dashes."""
    line = r"the Hermite--Gauss quadrature over $50$--$350$~K and 12--48 V"
    cleaned = rc.LATEX_EN_DASH_RE.sub(lambda m: m.group(0).replace("--", "  "), line)
    assert "--" not in cleaned
    prose = r"a real dash -- used as punctuation -- must survive stripping"
    cleaned2 = rc.LATEX_EN_DASH_RE.sub(lambda m: m.group(0).replace("--", "  "), prose)
    assert " -- " in cleaned2


def test_latex_en_dash_brace_closed_ranges_not_flagged():
    """Math-mode exponent ranges and \\eqref reference ranges are typography
    (the three paper/main.tex false positives found in the Sprint 4 dash
    purge): $10^{-17}$--$10^{-19}$, $\\sim10^{-19}$--$10^{-26}$, and
    \\eqref{eq:e11}--E12 must all be blanked, while a prose dash after a
    closing brace context elsewhere in the line still survives."""
    for line in (
        r"fractional shifts at $10^{-17}$--$10^{-19}$, resolved",
        r"the absolute-error pin, $\sim10^{-19}$--$10^{-26}$, in this",
        r"(Eqs.~\eqref{eq:e11}--E12 below), and no restriction",
    ):
        cleaned = rc.LATEX_EN_DASH_RE.sub(lambda m: m.group(0).replace("--", "  "), line)
        assert "--" not in cleaned, line
    prose = r"the pivot $P(r)$ -- the scalar factor -- stays untouched"
    cleaned2 = rc.LATEX_EN_DASH_RE.sub(lambda m: m.group(0).replace("--", "  "), prose)
    assert " -- " in cleaned2
