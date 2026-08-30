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
# prose-scan: Python docstrings (module/class/function, wrap-aware)
#
# The G19 gate incident this exists for: a banned "rather than" split
# across a line wrap in a TEST DOCSTRING was invisible to a scanner that
# only covered markdown/notebooks, and to a naive single-line grep.
# ---------------------------------------------------------------------------


def test_python_docstrings_extracts_module_class_and_function_docstrings(tmp_path):
    """AST extraction finds all three docstring kinds, each at the source
    line its own string literal begins on (not the ``def``/``class`` line
    above it)."""
    src = tmp_path / "fixture_module.py"
    src.write_text(
        '"""Module docstring line 1.\n'
        "Line 2.\n"
        '"""\n'
        "\n"
        "\n"
        "class Foo:\n"
        '    """Class docstring."""\n'
        "\n"
        "    def method(self):\n"
        '        """Method docstring."""\n'
        "        return 1\n"
        "\n"
        "\n"
        "def bare_function():\n"
        '    """Function docstring."""\n'
        "    return 2\n"
    )
    entries = rc.python_docstrings(src)
    texts_by_line = {line: text for line, text in entries}
    assert len(entries) == 4
    assert texts_by_line[1] == "Module docstring line 1.\nLine 2.\n"
    assert texts_by_line[7] == "Class docstring."
    assert texts_by_line[10] == "Method docstring."
    assert texts_by_line[15] == "Function docstring."


def test_python_docstrings_negative_case_code_strings_and_comments_unscanned(tmp_path):
    """A banned phrase sitting in an ordinary string assignment or a ``#``
    comment (never a docstring to Python) is not extracted -- only the
    genuine docstring's text is, confirming the ast-based extraction does
    not fall back to a string/regex scan that would catch those too."""
    src = tmp_path / "fixture_module.py"
    src.write_text(
        '"""Clean module docstring, no banned content."""\n'
        "\n"
        "# rather than doing it differently, this comment says so\n"
        'NOTE = "rather than the alternative, we chose this — see docs"\n'
        "\n"
        "\n"
        "def f():\n"
        '    """Clean function docstring."""\n'
        '    local = "rather than something else — an em dash too"\n'
        "    return local\n"
    )
    entries = rc.python_docstrings(src)
    assert [text for _line, text in entries] == [
        "Clean module docstring, no banned content.",
        "Clean function docstring.",
    ]


def test_python_docstrings_returns_empty_for_unparseable_file(tmp_path):
    """A file with a syntax error contributes no docstrings rather than
    raising, so one broken fixture file cannot crash the whole scan."""
    src = tmp_path / "broken.py"
    src.write_text("def f(:\n    pass\n")
    assert rc.python_docstrings(src) == []


def test_python_source_files_covers_the_four_docstring_source_dirs():
    """The real repo's enumeration returns only ``*.py`` files under
    src/, tests/, benchmarks/, and examples/, and is non-empty (a
    regression against silently scanning zero files)."""
    files = rc.python_source_files()
    assert files, "expected at least one Python source file"
    for path in files:
        rel = path.relative_to(REPO_ROOT)
        assert rel.parts[0] in {"src", "tests", "benchmarks", "examples"}
        assert rel.suffix == ".py"
        assert "__pycache__" not in rel.parts


def test_scan_docstring_text_catches_em_dash_planted_violation():
    findings = rc._scan_docstring_text(
        "fixture.py",
        "Uses an em dash — right here in the docstring.",
        10,
        meta_slop_fatal=[],
        meta_slop_minor=[],
        allowed=[],
    )
    assert any("em dash" in f.message and "docstring" in f.message for f in findings)
    assert all(f.severity == "FAIL" for f in findings if "em dash" in f.message)


def test_scan_docstring_text_catches_honest_family_word_planted_violation():
    findings = rc._scan_docstring_text(
        "fixture.py",
        "The result is honestly reported here.",
        1,
        meta_slop_fatal=[],
        meta_slop_minor=[],
        allowed=[],
    )
    assert any("honest-family" in f.message for f in findings)


def test_scan_docstring_text_skips_dash_as_punctuation_check():
    """Docstrings routinely carry a NumPy-style section-heading underline
    (``----------``) or an argparse-style ``--flag`` with no code fence
    around them; the dash-as-punctuation check that markdown prose-scan
    applies (only after fence-stripping) is deliberately not run against
    docstring text at all, so neither false-positives here."""
    text = "Parameters\n----------\nRun with --fast to skip slow checks."
    findings = rc._scan_docstring_text(
        "fixture.py", text, 1, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert findings == []


def test_scan_docstring_text_catches_wrapped_fatal_phrase_planted_violation_and_shifts_line():
    """The G19 incident itself: a banned phrase split across a hard-wrapped
    docstring line, at a docstring that does not start at line 1 -- the
    reported line must be the FILE's line (start_line-relative), not a
    line number local to the docstring's own text."""
    text = "The lattice pivot is not\nmerely a bookkeeping device in this framing."
    findings = rc._scan_docstring_text(
        "fixture.py",
        text,
        42,
        meta_slop_fatal=["not merely"],
        meta_slop_minor=[],
        allowed=[],
    )
    assert len(findings) == 1
    assert findings[0].severity == "FAIL"
    assert findings[0].line == 42
    assert "docstring" in findings[0].message


def test_scan_docstring_text_catches_wrapped_minor_phrase_planted_violation():
    text = "We used method A rather\nthan method B for this case."
    findings = rc._scan_docstring_text(
        "fixture.py",
        text,
        1,
        meta_slop_fatal=[],
        meta_slop_minor=["rather than"],
        allowed=[],
    )
    assert len(findings) == 1
    assert findings[0].severity == "MINOR"
    assert rc._status_from_findings(findings) == "PASS"


def test_scan_docstring_text_allowlist_suppresses_deliberate_keep():
    findings = rc._scan_docstring_text(
        "fixture.py",
        "Uses an em dash — right here in the docstring.",
        1,
        meta_slop_fatal=[],
        meta_slop_minor=[],
        allowed=["Uses an em dash — right here in the docstring."],
    )
    assert findings == []


def test_scan_docstring_text_negative_case_clean_docstring():
    findings = rc._scan_docstring_text(
        "fixture.py",
        "A clean docstring with no banned content at all.",
        1,
        meta_slop_fatal=["it is worth noting"],
        meta_slop_minor=["rather than"],
        allowed=[],
    )
    assert findings == []


def test_prose_scan_allow_docstrings_section_is_a_separate_namespace_from_allow():
    """[prose_scan.allow_docstrings] is read independently of
    [prose_scan.allow]: an allowlist entry under one section does not
    suppress a finding filed under the other, so a path collision between
    a markdown file and a same-named docstring source file (impossible in
    this repo today, but not structurally prevented) could never
    cross-allow one for the other."""
    allowlist = {
        "prose_scan": {
            "allow": {"fixture.py": ["Uses an em dash — right here."]},
        }
    }
    findings = rc._scan_docstring_text(
        "fixture.py",
        "Uses an em dash — right here.",
        1,
        meta_slop_fatal=[],
        meta_slop_minor=[],
        allowed=allowlist.get("prose_scan", {}).get("allow_docstrings", {}).get("fixture.py", []),
    )
    assert len(findings) == 1


def test_prose_scan_runs_clean_of_fail_findings_against_current_repo():
    """The full prose-scan check, including its docstring pass over
    src/, tests/, benchmarks/, and examples/, is FAIL-clean against the
    real repo with the real allowlist (WP28's own bar: MINOR findings,
    e.g. the pervasive "rather than" design-decision phrasing throughout
    this codebase's docstrings, are surfaced but never block, matching
    every other prose-scan surface)."""
    result = rc.prose_scan(rc.load_allowlist())
    assert result.status == "PASS", [f.format() for f in result.findings if f.severity == "FAIL"]


def test_paper_tex_prose_texts_reads_both_paper_and_composition_tex(tmp_path, monkeypatch):
    """`paper_tex_prose_texts` must not be a single-file scanner: it has to
    walk every path in `PAPER_TEX_RELPATHS`, main paper and composition
    companion paper alike, or the composition paper's prose is invisible
    to prose-scan and citation-check (the gap a gate review found)."""
    monkeypatch.setattr(rc, "REPO_ROOT", tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text("Main paper prose.\n")
    composition_dir = paper_dir / "composition"
    composition_dir.mkdir()
    (composition_dir / "main.tex").write_text("Composition paper prose.\n")

    texts = rc.paper_tex_prose_texts()

    assert [rel for rel, _ in texts] == ["paper/main.tex", "paper/composition/main.tex"]
    assert texts[0][1] == "Main paper prose.\n"
    assert texts[1][1] == "Composition paper prose.\n"


def test_paper_tex_prose_texts_skips_a_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "REPO_ROOT", tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text("Main paper prose.\n")
    # No paper/composition/main.tex planted.

    texts = rc.paper_tex_prose_texts()

    assert [rel for rel, _ in texts] == ["paper/main.tex"]


def test_prose_scan_catches_em_dash_planted_violation_in_composition_paper(tmp_path, monkeypatch):
    """Mirrors test_prose_scan_catches_unicode_em_dash_planted_violation, but
    plants the violation in paper/composition/main.tex specifically, so this
    fails again if the composition paper is ever dropped back out of the
    scanned path list."""
    monkeypatch.setattr(rc, "REPO_ROOT", tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text("Clean main paper prose.\n")
    composition_dir = paper_dir / "composition"
    composition_dir.mkdir()
    (composition_dir / "main.tex").write_text(
        "This sentence has an em dash — right there in the middle.\n"
    )

    result = rc.prose_scan({})

    assert result.status == "FAIL"
    assert any(
        f.file == "paper/composition/main.tex" and f.severity == "FAIL" for f in result.findings
    )


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


def test_internal_path_regex_allows_public_gate_records():
    """plan/reviews/ holds this repository's committed, public gate
    records (the working-repo convention since the repo transition), so
    citing one by path is legitimate; every sibling plan/ path stays
    flagged."""
    ok = "corrected per plan/reviews/G11-e38-motional-time-dilation.md, section A3."
    assert rc.INTERNAL_PATH_RE.search(ok) is None
    still_flagged = "see plan/notes/ion-clock-dossier.md for the extraction."
    assert rc.INTERNAL_PATH_RE.search(still_flagged) is not None
    also_flagged = "tracked in plan/STATUS.md and internal/signoffs/G8_physics_signoff_theory.md."
    assert rc.INTERNAL_PATH_RE.search(also_flagged) is not None


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


# `ruff format --check`'s stdout shape changed between ruff releases: older
# ruff lists one "Would reformat: path.py" line per file, while newer ruff
# (confirmed on 0.16.4) renders a multi-line diagnostic diff per file. A
# naive "count nonempty stdout lines" implementation reported a single
# unformatted file as "8 file(s) would be reformatted" under the new
# rendering (WP-tracked bug); `_count_ruff_would_reformat` must get the
# right count under both shapes, plus degrade gracefully if ruff's summary
# line is ever missing.


def test_count_ruff_would_reformat_old_one_line_per_file_shape():
    stdout = "Would reformat: src/a.py\nWould reformat: src/b.py\n2 files would be reformatted\n"
    assert rc._count_ruff_would_reformat(stdout) == 2


def test_count_ruff_would_reformat_old_shape_singular():
    stdout = "Would reformat: src/a.py\n1 file would be reformatted\n"
    assert rc._count_ruff_would_reformat(stdout) == 1


def test_count_ruff_would_reformat_new_diagnostic_diff_shape_one_file():
    """Regression for the planted-violation bug: one unformatted file's
    multi-line diff (ruff >= 0.13ish rendering) must count as 1, not as
    however many lines the diff happens to span."""
    stdout = (
        "unformatted: File would be reformatted\n"
        " --> src/cliffordclock/pipeline.py:12:9\n"
        "  |\n"
        "  - def foo( x,y ):\n"
        "  -     return x+y\n"
        "1 + def foo(x, y):\n"
        "2 +     return x + y\n"
        "  |\n"
        "\n"
        "1 file would be reformatted\n"
    )
    assert rc._count_ruff_would_reformat(stdout) == 1


def test_count_ruff_would_reformat_new_diagnostic_diff_shape_two_files():
    stdout = (
        "unformatted: File would be reformatted\n"
        " --> src/a.py:1:9\n"
        "  |\n"
        "  - def foo( x,y ):\n"
        "1 + def foo(x, y):\n"
        "  |\n"
        "\n"
        "unformatted: File would be reformatted\n"
        " --> src/b.py:1:9\n"
        "  |\n"
        "  - def bar( a,b ):\n"
        "1 + def bar(a, b):\n"
        "  |\n"
        "\n"
        "2 files would be reformatted\n"
    )
    assert rc._count_ruff_would_reformat(stdout) == 2


def test_count_ruff_would_reformat_mixed_summary_variant():
    """ruff appends ", N files already formatted" to the summary line when
    the run is a mix of clean and unformatted files; the regex must still
    extract the reformatted count, not the total."""
    stdout = (
        "unformatted: File would be reformatted\n"
        " --> src/a.py:1:9\n"
        "  |\n"
        "  - def foo( x,y ):\n"
        "1 + def foo(x, y):\n"
        "  |\n"
        "\n"
        "1 file would be reformatted, 12 files already formatted\n"
    )
    assert rc._count_ruff_would_reformat(stdout) == 1


def test_count_ruff_would_reformat_falls_back_to_path_count_without_summary_line():
    """If ruff's summary line is ever absent (future format change, or
    stdout truncated), fall back to counting distinct file paths named in
    the diagnostics rather than misreporting a line count."""
    stdout = (
        "unformatted: File would be reformatted\n"
        " --> src/a.py:1:9\n"
        "  |\n"
        "  - def foo( x,y ):\n"
        "1 + def foo(x, y):\n"
        "  |\n"
    )
    assert rc._count_ruff_would_reformat(stdout) == 1


def test_count_ruff_would_reformat_returns_placeholder_when_uncountable():
    """With neither a summary line nor any recognizable per-file marker,
    report '?' rather than a fabricated count."""
    assert rc._count_ruff_would_reformat("some unrecognized ruff output\n") == "?"


# ---------------------------------------------------------------------------
# suite-check: two-lane pytest split (WP28 upgrade: the suite outgrew a
# single 1800s subprocess timeout on a healthy repo once the slow-marked
# coupled-Floquet/JAX-core/generator tests pushed total runtime past 30
# minutes; two lanes mirror .github/workflows/ci.yml's own test/test-slow
# job split, each with its own timeout). Only the lane CONSTRUCTION is
# unit-tested here, via a mocked ``subprocess.run``: actually running the
# real suite end to end from this test module would take the better part
# of an hour on a healthy repo, exactly the cost this restructuring
# exists to keep off the fast-path check.
# ---------------------------------------------------------------------------


def test_pytest_lanes_pinned_names_markers_and_timeouts():
    """CI parity (`.github/workflows/ci.yml`): the fast lane deselects
    ``slow`` at the same 1800s budget the single-lane check used to run
    the whole suite at, and the slow lane covers only ``slow`` at a
    budget with measured headroom over CI's own 90-minute cap."""
    assert rc.PYTEST_LANES == [
        ("fast", "not slow", 1800.0),
        ("slow", "slow", 5400.0),
    ]


def test_run_pytest_lane_reports_summary_and_no_findings_on_success(monkeypatch):
    def fake_run(cmd, **kwargs):
        assert cmd == [sys.executable, "-m", "pytest", "-q", "-m", "not slow"]
        assert kwargs["timeout"] == 1800.0
        return rc.subprocess.CompletedProcess(
            cmd, 0, stdout="512 passed, 2 skipped in 34.21s\n", stderr=""
        )

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    detail, findings = rc._run_pytest_lane("fast", "not slow", 1800.0)
    assert detail == "pytest (fast): 512 passed, 2 skipped"
    assert findings == []


def test_run_pytest_lane_catches_nonzero_exit_planted_violation(monkeypatch):
    def fake_run(cmd, **kwargs):
        return rc.subprocess.CompletedProcess(
            cmd, 1, stdout="1 failed, 511 passed in 30.0s\n", stderr=""
        )

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    detail, findings = rc._run_pytest_lane("slow", "slow", 5400.0)
    assert "1 failed" in detail
    assert len(findings) == 1
    assert findings[0].severity == "FAIL"
    assert "slow" in findings[0].message
    assert "1 failed" in findings[0].message


def test_run_pytest_lane_names_the_lane_on_timeout_planted_violation(monkeypatch):
    """The per-lane timeout message names the lane, so a hang is
    diagnosable without re-deriving which of the two subprocesses it
    was from the elapsed wall-clock time alone."""

    def fake_run(cmd, **kwargs):
        raise rc.subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    detail, findings = rc._run_pytest_lane("slow", "slow", 5400.0)
    assert "TIMEOUT" in detail
    assert "slow" in detail
    assert len(findings) == 1
    assert findings[0].severity == "FAIL"
    assert "slow" in findings[0].message
    assert "5400" in findings[0].message


def test_suite_check_runs_both_lanes_with_pinned_markers_and_timeouts(monkeypatch):
    calls: list[tuple[str, str, float]] = []

    def fake_lane(name: str, marker_expr: str, timeout_s: float) -> tuple[str, list[rc.Finding]]:
        calls.append((name, marker_expr, timeout_s))
        return f"pytest ({name}): 1 passed", []

    monkeypatch.setattr(rc, "_run_pytest_lane", fake_lane)
    monkeypatch.setattr(
        rc.subprocess,
        "run",
        lambda cmd, **kwargs: rc.subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    result = rc.suite_check({})
    assert calls == [("fast", "not slow", 1800.0), ("slow", "slow", 5400.0)]
    assert result.status == "PASS"
    assert "pytest (fast): 1 passed" in result.detail
    assert "pytest (slow): 1 passed" in result.detail


def test_suite_check_fails_if_either_lane_fails_planted_violation(monkeypatch):
    def fake_lane(name: str, marker_expr: str, timeout_s: float) -> tuple[str, list[rc.Finding]]:
        if name == "slow":
            return f"pytest ({name}): 1 failed", [
                rc.Finding(f"pytest ({name})", None, "slow lane exited 1: 1 failed")
            ]
        return f"pytest ({name}): 1 passed", []

    monkeypatch.setattr(rc, "_run_pytest_lane", fake_lane)
    monkeypatch.setattr(
        rc.subprocess,
        "run",
        lambda cmd, **kwargs: rc.subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    result = rc.suite_check({})
    assert result.status == "FAIL"
    assert any("slow" in f.message for f in result.findings)


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


@pytest.mark.slow
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


# ---------------------------------------------------------------------------
# prose-scan: wrapped-line phrase matching + fenced-block comment prose
# (prose-audit follow-up: a phrase split across a markdown hard line-wrap
# escaped the line-by-line scan, and a phrase inside a fenced YAML block's
# config comment was removed from scanning by fence-stripping).
# ---------------------------------------------------------------------------


def test_prose_scan_catches_wrapped_fatal_phrase_planted_violation():
    """The exact prose-audit escape: a banned two-word phrase hard-wrapped
    so its first word ends one source line and its second word starts
    the next (see the fixture text below for the literal phrase)."""
    text = "The lattice pivot is not\nmerely a bookkeeping device in this framing.\n"
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=["not merely"],
        meta_slop_minor=[],
        allowed=[],
    )
    assert len(findings) == 1
    assert findings[0].severity == "FAIL"
    assert findings[0].line == 1


def test_prose_scan_catches_wrapped_minor_phrase_planted_violation():
    text = "We used method A rather\nthan method B for this case.\n"
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


def test_prose_scan_wrapped_phrase_reports_line_where_phrase_begins():
    text = (
        "First sentence of the paragraph sits on this line.\n"
        "We used method A rather\n"
        "than method B for this case.\n"
    )
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=[],
        meta_slop_minor=["rather than"],
        allowed=[],
    )
    assert len(findings) == 1
    assert findings[0].line == 2


def test_prose_scan_blank_line_is_a_paragraph_boundary():
    # A paragraph break between the two words is not a wrapped phrase.
    text = "The pivot is not\n\nmerely is how the next paragraph starts.\n"
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=["not merely"],
        meta_slop_minor=[],
        allowed=[],
    )
    assert findings == []


def test_prose_scan_list_items_do_not_join_into_one_paragraph():
    # Two adjacent bullets are separate units; joining them would
    # fabricate a phrase neither one contains.
    text = "- the pivot was chosen, not\n- merely defaulted to by the loader\n"
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=["not merely"],
        meta_slop_minor=[],
        allowed=[],
    )
    assert findings == []


def test_prose_scan_wrapped_bullet_continuation_still_joins():
    # A single bullet whose own text wraps onto an unmarked continuation
    # line is one unit; the phrase inside it must still be caught.
    text = "- the lattice pivot is not\n  merely a bookkeeping device here\n"
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=["not merely"],
        meta_slop_minor=[],
        allowed=[],
    )
    assert len(findings) == 1
    assert findings[0].line == 1


def test_prose_scan_heading_does_not_join_with_body():
    text = "## Chosen, not\nmerely defaulted is what the body says next.\n"
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=["not merely"],
        meta_slop_minor=[],
        allowed=[],
    )
    assert findings == []


def test_prose_scan_wrapped_phrase_in_blockquote_is_caught():
    text = "> the lattice pivot is not\n> merely a bookkeeping device\n"
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=["not merely"],
        meta_slop_minor=[],
        allowed=[],
    )
    assert len(findings) == 1


def test_prose_scan_wrapped_phrase_allowlisted_by_first_line_snippet():
    # The allowlist stays line-based: an entry matching the line the
    # phrase starts on suppresses the finding, as for unwrapped phrases.
    text = "We used method A rather\nthan method B for this case.\n"
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=[],
        meta_slop_minor=["rather than"],
        allowed=["We used method A rather"],
    )
    assert findings == []


def test_prose_scan_catches_phrase_in_fenced_yaml_comment_planted_violation():
    """The prose-audit fenced-block escape: a banned phrase in a config
    comment that fence-stripping used to remove from scanning."""
    text = "```yaml\n# choose the secular sampler rather than the default\nmode: secular\n```\n"
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
    assert findings[0].line == 2
    assert "fenced-block comment" in findings[0].message


def test_prose_scan_wrapped_fenced_comment_phrase_is_caught():
    text = "```yaml\n# this sampler is not\n# merely a default choice\nmode: secular\n```\n"
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=["not merely"],
        meta_slop_minor=[],
        allowed=[],
    )
    assert len(findings) == 1
    assert findings[0].line == 2


def test_prose_scan_fenced_code_lines_stay_unscanned_for_phrases():
    # The decision is comments-only: a phrase inside actual fenced CODE
    # (here a flag value, not authored prose) stays out of scope.
    text = "```bash\ncliffordclock run --note 'rather than default'\n```\n"
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=["rather than"],
        meta_slop_minor=[],
        allowed=[],
    )
    assert findings == []


def test_prose_scan_fenced_comment_with_cli_flag_fires_no_dash_finding():
    # Fenced comments are scanned for phrases ONLY; a --flag mentioned in
    # a comment must not trigger the dash-as-punctuation check.
    text = "```yaml\n# pass --fast to skip the slow checks\nmode: secular\n```\n"
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert findings == []


def test_prose_scan_fenced_comment_allowlisted_by_raw_source_line():
    # Allowlist entries match the file's raw line (with its "#" marker),
    # not the extracted comment text.
    text = "```yaml\n# choose the secular sampler rather than the default\n```\n"
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=[],
        meta_slop_minor=["rather than"],
        allowed=["# choose the secular sampler rather than the default"],
    )
    assert findings == []


# ---------------------------------------------------------------------------
# prose-scan: clarity-read heuristics (prose-review skill, "The clarity
# read" and "'Exactly' and 'precisely'") -- long sentences, "because" plus
# a 3+-comma qualifier chain, and the exactly/precisely emphasis words.
# All MINOR, never FAIL: these surface candidates for the human clarity
# read, not verdicts.
# ---------------------------------------------------------------------------


def _n_word_sentence(n: int) -> str:
    """A single sentence of exactly ``n`` whitespace-separated words."""
    return " ".join(f"w{i}" for i in range(n)) + "."


def test_prose_scan_catches_long_sentence_planted_violation():
    text = _n_word_sentence(55)
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    long_findings = [f for f in findings if "sentence runs" in f.message]
    assert len(long_findings) == 1
    assert long_findings[0].severity == "MINOR"
    assert rc._status_from_findings(findings) == "PASS"


def test_prose_scan_long_sentence_negative_case():
    text = "This is a short sentence with few words."
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert not any("sentence runs" in f.message for f in findings)


def test_prose_scan_long_sentence_boundary_45_words_not_flagged():
    text = _n_word_sentence(45)
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert not any("sentence runs" in f.message for f in findings)


def test_prose_scan_long_sentence_boundary_46_words_flagged():
    text = _n_word_sentence(46)
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert any("sentence runs" in f.message for f in findings)


def test_prose_scan_long_sentence_allowlist_suppresses():
    text = _n_word_sentence(55) + "\n"
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=[],
        meta_slop_minor=[],
        allowed=[text.strip()],
    )
    assert not any("sentence runs" in f.message for f in findings)


def test_prose_scan_long_sentence_wrapped_across_hard_wrap_still_measured_whole():
    # A single 50-word sentence hard-wrapped across two source lines must
    # still be measured as one sentence (see _wrapped_paragraphs), and the
    # finding reports the line the sentence starts on.
    first_line = " ".join(f"w{i}" for i in range(30))
    second_line = " ".join(f"w{i}" for i in range(30, 50)) + "."
    text = f"{first_line}\n{second_line}\n"
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    long_findings = [f for f in findings if "sentence runs" in f.message]
    assert len(long_findings) == 1
    assert long_findings[0].line == 1


def test_prose_scan_catches_because_qualifier_chain_planted_violation():
    text = (
        "The result drifts because the reference clock free runs, the "
        "servo lags behind, and the calibration, which is periodic, "
        "cannot keep pace."
    )
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    chain_findings = [f for f in findings if "qualifier chain" in f.message]
    assert len(chain_findings) == 1
    assert chain_findings[0].severity == "MINOR"
    assert rc._status_from_findings(findings) == "PASS"


def test_prose_scan_because_with_too_few_commas_not_flagged():
    text = (
        "The result drifts because the reference clock free runs, and "
        "the servo lags, which is slow."
    )
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert not any("qualifier chain" in f.message for f in findings)


def test_prose_scan_comma_chain_without_because_not_flagged():
    text = (
        "The clock drifts slowly, the servo lags a little, and the "
        "calibration, which is periodic, keeps pace."
    )
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert not any("qualifier chain" in f.message for f in findings)


def test_prose_scan_catches_exactly_planted_violation():
    text = "The offset is exactly what the model predicts for this configuration."
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    emphasis_findings = [f for f in findings if "emphasis word" in f.message]
    assert len(emphasis_findings) == 1
    assert emphasis_findings[0].severity == "MINOR"
    assert "'exactly'" in emphasis_findings[0].message.lower()
    assert rc._status_from_findings(findings) == "PASS"


def test_prose_scan_catches_precisely_planted_violation():
    text = "This is precisely the tool a lab needs for this measurement."
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    emphasis_findings = [f for f in findings if "emphasis word" in f.message]
    assert len(emphasis_findings) == 1
    assert emphasis_findings[0].severity == "MINOR"


def test_prose_scan_exactly_precisely_negative_case():
    text = "This is the tool a lab needs for this measurement."
    findings = rc._scan_prose_text(
        "fixture.md", text, strip_code=True, meta_slop_fatal=[], meta_slop_minor=[], allowed=[]
    )
    assert not any("emphasis word" in f.message for f in findings)


def test_prose_scan_exactly_allowlist_suppresses():
    text = "The offset is exactly what the model predicts for this configuration.\n"
    findings = rc._scan_prose_text(
        "fixture.md",
        text,
        strip_code=True,
        meta_slop_fatal=[],
        meta_slop_minor=[],
        allowed=[text.strip()],
    )
    assert not any("emphasis word" in f.message for f in findings)


def test_prose_scan_clarity_heuristics_apply_to_unstripped_tex_prose():
    # strip_code=False is the paper/main.tex path; the clarity heuristics
    # and the exactly/precisely check run there too, same as meta-slop.
    text = "The correction is exactly " + _n_word_sentence(50)
    findings = rc._scan_prose_text(
        "paper/main.tex",
        text,
        strip_code=False,
        meta_slop_fatal=[],
        meta_slop_minor=[],
        allowed=[],
    )
    assert any("emphasis word" in f.message for f in findings)
    assert any("sentence runs" in f.message for f in findings)


def test_wrapped_paragraphs_maps_offsets_to_source_lines():
    lines = ["first line here", "second line here", "", "fourth line alone"]
    paragraphs = rc._wrapped_paragraphs(lines)
    assert [text for text, _ in paragraphs] == [
        "first line here second line here",
        "fourth line alone",
    ]
    assert paragraphs[0][1] == [(0, 1), (16, 2)]
    assert paragraphs[1][1] == [(0, 4)]


def test_fenced_comment_lines_preserve_line_positions():
    text = "prose above\n```yaml\n# a comment\nkey: value\n```\nprose below\n"
    assert rc._fenced_comment_lines(text) == ["", "", "a comment", "", "", "", ""]
