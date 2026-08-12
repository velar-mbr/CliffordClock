# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mechanized release-review checks for CliffordClock (WP28).

This module mechanizes every check a human reviewer repeats by hand for a
release/beta review: em-dash/honest-family/meta-slop prose greps, bare-
``pytest.approx``/missing-``atol`` tolerance scans, citation-byline vs. a
pinned bibliography, headline-phrase consistency, internal-only
``plan/``/``internal/`` path-reference leaks, benchmark-JSON determinism,
notebook re-execution, and the suite/lint/mypy invocation, so an agent
review spends its tokens on physics and judgment, not on grepping. See
the project's mechanized-checks review checklist and the WP28 sprint
record for the originating spec.

Zero new runtime dependencies: this module uses only the standard library
plus ``nbconvert`` for ``notebooks-check`` (already an optional project
dependency, ``pyproject.toml``'s ``notebooks`` extra); if it is not
installed, ``notebooks-check`` reports ``SKIP`` rather than failing.

Usage
-----
    python tools/release_checks.py                      # run every check
    python tools/release_checks.py --fast                # skip checks 6-8
    python tools/release_checks.py --only prose-scan,tolerance-scan
    python tools/release_checks.py --list                # list check names

Exit code is nonzero if any check's status is ``FAIL``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
ALLOWLIST_PATH = TOOLS_DIR / "release_checks_allowlist.toml"
BIBLIOGRAPHY_PATH = TOOLS_DIR / "bibliography.toml"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """One thing a check noticed, at (or near) a specific location.

    Parameters
    ----------
    file : str
        Path (relative to the repo root) the finding was found in.
    line : int | None
        Line number, or ``None`` when the finding is not tied to one line
        (e.g. a whole-notebook re-execution mismatch).
    message : str
        One-sentence, human-readable description of the finding.
    severity : str
        ``"FAIL"`` (contributes to the owning check's FAIL status) or
        ``"MINOR"`` (reported, but does not by itself fail the check;
        e.g. the WP28-mandated "rather than" meta-slop phrase, which is
        "flagged-not-fatal" per the WP28 sprint record).
    """

    file: str
    line: int | None
    message: str
    severity: str = "FAIL"

    def format(self) -> str:
        """Render as ``"[SEVERITY] file:line: message"``."""
        loc = f"{self.file}:{self.line}" if self.line is not None else self.file
        return f"[{self.severity}] {loc}: {self.message}"


@dataclass
class CheckResult:
    """The outcome of one named check."""

    name: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    findings: list[Finding] = field(default_factory=list)
    detail: str = ""


def _status_from_findings(findings: list[Finding]) -> str:
    return "FAIL" if any(f.severity == "FAIL" for f in findings) else "PASS"


# ---------------------------------------------------------------------------
# Shared config loading
# ---------------------------------------------------------------------------


def load_allowlist() -> dict[str, Any]:
    """Load ``tools/release_checks_allowlist.toml`` (empty dict if absent)."""
    if not ALLOWLIST_PATH.exists():
        return {}
    return tomllib.loads(ALLOWLIST_PATH.read_text())


def load_bibliography() -> list[dict[str, Any]]:
    """Load ``tools/bibliography.toml``'s ``[[paper]]`` array."""
    if not BIBLIOGRAPHY_PATH.exists():
        return []
    data = tomllib.loads(BIBLIOGRAPHY_PATH.read_text())
    result: list[dict[str, Any]] = data.get("paper", [])
    return result


def _is_allowed(snippet: str, allowed: list[str]) -> bool:
    return any(a in snippet for a in allowed)


# ---------------------------------------------------------------------------
# Public-file enumeration
# ---------------------------------------------------------------------------


def public_markdown_files() -> list[Path]:
    """README.md + every docs/**/*.md + every benchmarks/**/*.md, sorted."""
    files: list[Path] = []
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        files.append(readme)
    docs_dir = REPO_ROOT / "docs"
    if docs_dir.exists():
        files.extend(sorted(docs_dir.rglob("*.md")))
    benchmarks_dir = REPO_ROOT / "benchmarks"
    if benchmarks_dir.exists():
        files.extend(
            sorted(p for p in benchmarks_dir.rglob("*.md") if "__pycache__" not in p.parts)
        )
    return files


def public_notebook_files() -> list[Path]:
    nb_dir = REPO_ROOT / "notebooks"
    if not nb_dir.exists():
        return []
    return sorted(nb_dir.glob("*.ipynb"))


def notebook_markdown_text(nb_path: Path) -> list[tuple[str, str]]:
    """Return ``[(cell_label, joined_source_text), ...]`` for markdown cells.

    ``cell_label`` looks like ``"notebooks/06_foo.ipynb#cell3"`` (0-indexed
    among ALL cells, so it is stable if non-markdown cells are interleaved).
    """
    try:
        nb = json.loads(nb_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    rel = str(nb_path.relative_to(REPO_ROOT))
    out: list[tuple[str, str]] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        text = "".join(source) if isinstance(source, list) else str(source)
        out.append((f"{rel}#cell{idx}", text))
    return out


LATEX_COMMENT_RE = re.compile(r"(?<!\\)%.*$")
#: LaTeX double-hyphen usages that are typography, not prose punctuation
#: (wave-review blocker 4): name pairs (Hermite--Gauss), numeric ranges
#: incl. math-mode-wrapped ones ($50$--$350$, $10^{-17}$--$10^{-19}$),
#: and reference ranges (\eqref{eq:e11}--E12).
LATEX_EN_DASH_RE = re.compile(
    r"(?:[A-Za-z]--[A-Za-z])"  # name pairs: Hermite--Gauss
    r"|(?:\$?[0-9][0-9.]*\$?--\$?[0-9])"  # ranges: 50--350, $50$--$350$
    r"|(?:\}\$--\$)"  # brace-closed math ranges: $10^{-17}$--$10^{-19}$
    r"|(?:\}--[A-Za-z0-9])"  # brace-closed ref ranges: \eqref{eq:e11}--E12
)


def paper_tex_prose_text() -> tuple[str, str] | None:
    """Return ``(relpath, comment_stripped_text)`` for ``paper/main.tex``."""
    tex_path = REPO_ROOT / "paper" / "main.tex"
    if not tex_path.exists():
        return None
    lines = tex_path.read_text().split("\n")
    stripped = [LATEX_COMMENT_RE.sub("", line) for line in lines]
    # Blank the legitimate en-dash typography so only true prose dashes
    # (" -- " with surrounding spaces, or "---") remain scannable.
    stripped = [
        LATEX_EN_DASH_RE.sub(lambda m: m.group(0).replace("--", "  "), line) for line in stripped
    ]
    return "paper/main.tex", "\n".join(stripped)


# ---------------------------------------------------------------------------
# 1. prose-scan
# ---------------------------------------------------------------------------

EM_DASH = "—"
# Dash-as-punctuation: ASCII "--"/"---" NOT immediately flanked by digits on
# both sides (a LaTeX/markdown numeric range like "50--350" is legitimate and
# allowlisted by this lookaround, not by a separate list).
DASH_PUNCT_RE = re.compile(r"(?<!\d)-{2,3}(?!\d)")
# Markdown table-separator / horizontal-rule rows ("---", "|---|---|",
# "| :-- | --: |") are not prose punctuation; skip them entirely.
TABLE_OR_RULE_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")

HONEST_FAMILY_RE = re.compile(
    r"\b(honestly|honesty|honest)\b",
    re.IGNORECASE,
)

FENCE_MARKERS = ("```", "~~~")


def _strip_markdown_code(text: str) -> str:
    """Blank out fenced code blocks and inline code spans, preserving line
    numbers and (for inline spans) column count, so prose scans do not fire
    on CLI flags (``--fast``), code identifiers, or JSON keys inside code.
    """
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in lines:
        stripped = line.strip()
        if not in_fence and stripped.startswith(FENCE_MARKERS):
            in_fence = True
            fence_marker = stripped[:3]
            out.append("")
            continue
        if in_fence:
            if stripped.startswith(fence_marker):
                in_fence = False
            out.append("")
            continue
        out.append(re.sub(r"`[^`]*`", lambda m: " " * len(m.group(0)), line))

    # Second pass: a single-backtick span opened on one line and closed on
    # a later one (a wrapped CLI command) survives the per-line sub above
    # and its contents would fire as prose (wave-review blocker 4). Track
    # open-span state across the already-processed lines and blank the
    # interior of any such multi-line span.
    in_span = False
    for i, line in enumerate(out):
        if not line:
            continue
        n_ticks = line.count("`")
        if in_span:
            if n_ticks % 2 == 1:
                close = line.index("`")
                out[i] = " " * (close + 1) + line[close + 1 :]
                in_span = False
            else:
                out[i] = " " * len(line)
        elif n_ticks % 2 == 1:
            open_ = line.rindex("`")
            out[i] = line[:open_] + " " * (len(line) - open_)
            in_span = True

    # Strip markdown link/image TARGETS (badge URLs like status-pre--beta
    # carry double hyphens that are addresses, not prose) while keeping
    # the link text scannable.
    out = [
        re.sub(r"(\]\()([^)]*)(\))", lambda m: m.group(1) + " " * len(m.group(2)) + m.group(3), ln)
        for ln in out
    ]
    return "\n".join(out)


def _scan_prose_text(
    relpath: str,
    raw_text: str,
    *,
    strip_code: bool,
    meta_slop_fatal: list[str],
    meta_slop_minor: list[str],
    allowed: list[str],
) -> list[Finding]:
    text = _strip_markdown_code(raw_text) if strip_code else raw_text
    findings: list[Finding] = []
    for lineno, line in enumerate(text.split("\n"), start=1):
        if TABLE_OR_RULE_RE.match(line):
            continue
        snippet = line.strip()
        if not snippet:
            continue
        if _is_allowed(snippet, allowed):
            continue
        if EM_DASH in line:
            findings.append(
                Finding(relpath, lineno, f"Unicode em dash (U+2014) in prose: {snippet[:120]!r}")
            )
        m = DASH_PUNCT_RE.search(line)
        if m:
            findings.append(
                Finding(
                    relpath,
                    lineno,
                    f"dash-as-punctuation {m.group(0)!r} (ASCII em/en-dash substitute) "
                    f"in prose: {snippet[:120]!r}",
                )
            )
        if HONEST_FAMILY_RE.search(line):
            findings.append(
                Finding(relpath, lineno, f"honest-family word in prose: {snippet[:120]!r}")
            )
        low = line.lower()
        for phrase in meta_slop_fatal:
            if phrase.lower() in low:
                findings.append(
                    Finding(
                        relpath,
                        lineno,
                        f"meta-slop phrase {phrase!r} in prose: {snippet[:120]!r}",
                        severity="FAIL",
                    )
                )
        for phrase in meta_slop_minor:
            if phrase.lower() in low:
                findings.append(
                    Finding(
                        relpath,
                        lineno,
                        f"meta-slop phrase {phrase!r} (flagged, not fatal) in prose: "
                        f"{snippet[:120]!r}",
                        severity="MINOR",
                    )
                )
    return findings


def prose_scan(allowlist: dict[str, Any]) -> CheckResult:
    """Em dash / dash-as-punctuation / honest-family / meta-slop scan.

    Scans README.md, docs/**/*.md, benchmarks/**/*.md, notebook markdown
    cells, and paper/main.tex prose (after LaTeX-comment stripping).
    """
    meta_slop_cfg = allowlist.get("meta_slop", {})
    meta_slop_fatal = list(meta_slop_cfg.get("fatal", []))
    meta_slop_minor = list(meta_slop_cfg.get("minor", []))
    allow_map = allowlist.get("prose_scan", {}).get("allow", {})

    findings: list[Finding] = []
    for md_path in public_markdown_files():
        rel = str(md_path.relative_to(REPO_ROOT))
        text = md_path.read_text(errors="replace")
        findings.extend(
            _scan_prose_text(
                rel,
                text,
                strip_code=True,
                meta_slop_fatal=meta_slop_fatal,
                meta_slop_minor=meta_slop_minor,
                allowed=allow_map.get(rel, []),
            )
        )
    for nb_path in public_notebook_files():
        for cell_label, cell_text in notebook_markdown_text(nb_path):
            rel = str(nb_path.relative_to(REPO_ROOT))
            findings.extend(
                _scan_prose_text(
                    cell_label,
                    cell_text,
                    strip_code=True,
                    meta_slop_fatal=meta_slop_fatal,
                    meta_slop_minor=meta_slop_minor,
                    allowed=allow_map.get(rel, []),
                )
            )
    tex = paper_tex_prose_text()
    if tex is not None:
        rel, text = tex
        findings.extend(
            _scan_prose_text(
                rel,
                text,
                strip_code=False,
                meta_slop_fatal=meta_slop_fatal,
                meta_slop_minor=meta_slop_minor,
                allowed=allow_map.get(rel, []),
            )
        )
    return CheckResult("prose-scan", _status_from_findings(findings), findings)


# ---------------------------------------------------------------------------
# 2. tolerance-scan
# ---------------------------------------------------------------------------

# One level of nested parens is handled (covers the vast majority of
# real-world approx(...)/assert_allclose(...) call sites).
_BALANCED_ARGS = r"((?:[^()]|\([^()]*\))*)"
APPROX_CALL_RE = re.compile(r"pytest\.approx\(" + _BALANCED_ARGS + r"\)", re.DOTALL)
ALLCLOSE_CALL_RE = re.compile(r"(?<!\w)assert_allclose\(" + _BALANCED_ARGS + r"\)", re.DOTALL)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _scan_tolerance_text(
    relpath: str, text: str, allowed: list[str] | None = None
) -> list[Finding]:
    allowed = allowed or []
    findings: list[Finding] = []
    for m in APPROX_CALL_RE.finditer(text):
        args = m.group(1)
        if "rel=" in args and "abs=" not in args:
            if _is_allowed(m.group(0), allowed):
                continue
            findings.append(
                Finding(
                    relpath,
                    _line_of(text, m.start()),
                    f"pytest.approx(...) uses rel= without abs= (default abs=1e-12 "
                    f"silently dominates small quantities): {m.group(0)[:140]!r}",
                )
            )
    for m in ALLCLOSE_CALL_RE.finditer(text):
        args = m.group(1)
        if "atol=" not in args:
            if _is_allowed(m.group(0), allowed):
                continue
            findings.append(
                Finding(
                    relpath,
                    _line_of(text, m.start()),
                    f"assert_allclose(...) missing explicit atol=: {m.group(0)[:140]!r}",
                )
            )
    return findings


def tolerance_scan(allowlist: dict[str, Any]) -> CheckResult:
    """Flag bare ``pytest.approx(..., rel=...)`` without ``abs=`` and
    ``assert_allclose(...)`` without ``atol=`` across ``tests/**``.

    Per the project's mechanized-checks review checklist: ``pytest.approx(x, rel=r)``
    without ``abs=0`` silently applies a default ``abs=1e-12`` that
    dominates for small quantities; prefer ``assert_allclose(rtol=...,
    atol=0)``.

    ``[tolerance_scan.allow]`` in the allowlist maps a file path to exact
    call-text substrings that are deliberate keeps (today: this scanner's
    own planted-violation fixture strings in its test file).
    """
    tests_dir = REPO_ROOT / "tests"
    if not tests_dir.exists():
        return CheckResult("tolerance-scan", "PASS", [], detail="tests/ not found")
    allow_map = allowlist.get("tolerance_scan", {}).get("allow", {})
    findings: list[Finding] = []
    for py_path in sorted(tests_dir.rglob("*.py")):
        if "__pycache__" in py_path.parts:
            continue
        rel = str(py_path.relative_to(REPO_ROOT))
        text = py_path.read_text(errors="replace")
        findings.extend(_scan_tolerance_text(rel, text, allow_map.get(rel, [])))
    return CheckResult("tolerance-scan", _status_from_findings(findings), findings)


# ---------------------------------------------------------------------------
# 3. citation-check
# ---------------------------------------------------------------------------

# A standalone 4-digit year: not embedded in a longer digit run (excludes
# journal page/eid numbers like "020502(R)" which contain a "2050"-looking
# substring) and not the start of an ISO date like "2026-08-11" (this
# project's own session-date convention, common right next to citations in
# benchmarks/SOURCES.md-style prose). The date guard matches the FULL
# ISO shape (-MM-DD) so a hyphenated year-range citation like
# "Roos (2006-2019)" still yields both years (wave-review blocker 3).
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)(?!-\d{2}-\d{2}\b)")


def _surname_years(papers: list[dict[str, Any]]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for paper in papers:
        years = paper.get("year", [])
        for surname in paper.get("surnames", []):
            out.setdefault(surname, set()).update(years)
    return out


def _scan_citations_in_text(
    relpath: str, text: str, surname_years: dict[str, set[int]]
) -> list[Finding]:
    findings: list[Finding] = []
    for surname, years in surname_years.items():
        if not years:
            continue
        for m in re.finditer(rf"\b{re.escape(surname)}\b", text):
            window = text[m.end() : m.end() + 120]
            newline_pos = window.find("\n")
            if newline_pos != -1:
                window = window[:newline_pos]
            ym = YEAR_RE.search(window)
            if ym is None:
                continue
            year = int(ym.group(1))
            if year not in years:
                findings.append(
                    Finding(
                        relpath,
                        _line_of(text, m.start()),
                        f"citation byline {surname!r} followed by year {year}, which "
                        f"contradicts tools/bibliography.toml's pinned year(s) "
                        f"{sorted(years)} for {surname!r}",
                    )
                )
    return findings


def citation_check(allowlist: dict[str, Any]) -> CheckResult:  # noqa: ARG001
    """Grep public files for a pinned author surname followed nearby by a
    year, and flag any such byline fragment whose year contradicts
    ``tools/bibliography.toml`` (mechanizes the fabricated-byline catches
    this project's reviews have made by hand, e.g. benchmarks/SOURCES.md
    section 8's "an earlier staged draft carried a fabricated middle
    initial, caught there").
    """
    papers = load_bibliography()
    if not papers:
        return CheckResult(
            "citation-check",
            "FAIL",
            [Finding("tools/bibliography.toml", None, "bibliography is empty or missing")],
        )
    surname_years = _surname_years(papers)
    findings: list[Finding] = []
    for md_path in public_markdown_files():
        rel = str(md_path.relative_to(REPO_ROOT))
        text = md_path.read_text(errors="replace")
        findings.extend(_scan_citations_in_text(rel, text, surname_years))
    for nb_path in public_notebook_files():
        for cell_label, cell_text in notebook_markdown_text(nb_path):
            findings.extend(_scan_citations_in_text(cell_label, cell_text, surname_years))
    tex = paper_tex_prose_text()
    if tex is not None:
        rel, text = tex
        findings.extend(_scan_citations_in_text(rel, text, surname_years))
    return CheckResult("citation-check", _status_from_findings(findings), findings)


# ---------------------------------------------------------------------------
# 4. headline-check
# ---------------------------------------------------------------------------

TWO_REPRO_RE = re.compile(r"\btwo\b[^.]{0,40}\breproducibility case", re.IGNORECASE)
ZERO_BLIND_RE = re.compile(r"\bzero\b[^.]{0,40}\bblind predictions?\b", re.IGNORECASE)
STALE_HEADLINE_RE = re.compile(r"\b1 of 1\b|\bone reproducibility case\b", re.IGNORECASE)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including markdown hard line-wraps) to a
    single space, so a phrase split across a wrapped source line (e.g. "TWO"
    at the end of one line, "reproducibility cases" starting the next) is
    still matched as one phrase.
    """
    return re.sub(r"\s+", " ", text)


def headline_check(allowlist: dict[str, Any]) -> CheckResult:
    """Check the validation-headline phrasing (read from
    ``benchmarks/RESULTS.md``'s executive summary, the authoritative
    source) is echoed consistently in README.md/docs/validation.md, and
    that no OTHER public file carries the stale "1 of 1"/"one
    reproducibility case" phrasing.
    """
    findings: list[Finding] = []
    results_path = REPO_ROOT / "benchmarks" / "RESULTS.md"
    if not results_path.exists():
        return CheckResult(
            "headline-check",
            "FAIL",
            [Finding("benchmarks/RESULTS.md", None, "authoritative source file is missing")],
        )
    results_text = _normalize_whitespace(results_path.read_text())
    if not TWO_REPRO_RE.search(results_text):
        findings.append(
            Finding(
                "benchmarks/RESULTS.md",
                None,
                "executive summary does not state the authoritative "
                "'two ... reproducibility case(s)' headline",
            )
        )
    if not ZERO_BLIND_RE.search(results_text):
        findings.append(
            Finding(
                "benchmarks/RESULTS.md",
                None,
                "executive summary does not state the authoritative "
                "'zero ... blind predictions' headline",
            )
        )

    for target in ("README.md", "docs/validation.md"):
        p = REPO_ROOT / target
        if not p.exists():
            findings.append(Finding(target, None, "expected public headline surface is missing"))
            continue
        text = _normalize_whitespace(p.read_text())
        if not TWO_REPRO_RE.search(text):
            findings.append(
                Finding(
                    target,
                    None,
                    "does not echo the 'two reproducibility cases' headline "
                    "(benchmarks/RESULTS.md is the authoritative source)",
                )
            )
        if not ZERO_BLIND_RE.search(text):
            findings.append(
                Finding(
                    target,
                    None,
                    "does not echo the 'zero blind predictions' headline "
                    "(benchmarks/RESULTS.md is the authoritative source)",
                )
            )

    frozen = set(allowlist.get("headline_check", {}).get("frozen_artifacts", []))
    for md_path in public_markdown_files():
        rel = str(md_path.relative_to(REPO_ROOT))
        if rel in frozen:
            continue
        text = md_path.read_text(errors="replace")
        for lineno, line in enumerate(text.split("\n"), start=1):
            if STALE_HEADLINE_RE.search(line):
                findings.append(
                    Finding(
                        rel,
                        lineno,
                        "stale '1 of 1'/'one reproducibility case' phrasing outside the "
                        f"pinned frozen-artifact allowlist: {line.strip()[:140]!r}",
                    )
                )

    return CheckResult("headline-check", _status_from_findings(findings), findings)


# ---------------------------------------------------------------------------
# 5. internal-path-check
# ---------------------------------------------------------------------------

# Matches a literal path reference into either private top-level directory
# (the literal strings "plan" or "internal", each followed by a slash and
# at least one more path-ish character), not itself preceded by a word
# character or another slash (so it does not fire on an unrelated deeper
# path segment, e.g. a vendored third-party "internal" subpackage).
INTERNAL_PATH_RE = re.compile(r"(?<![\w/])(?:plan|internal)/[\w.\-/]+")

# Directories this repo's own export tooling excludes from the public
# release tree (see this project's fresh-history export script, WP12/
# WP13), or that never leave this working copy at all: nothing under them
# needs scanning (self-reference is fine), and files everywhere else must
# not point INTO them.
_PRIVATE_DIR_NAMES = frozenset({"plan", "internal", ".claude", ".git"})

# Directories full of generated/vendored/cache content: irrelevant either
# way, skipped purely to keep the scan fast and quiet.
_SKIP_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".venv",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".hypothesis",
        "node_modules",
        "dist",
    }
)

# Extensions worth grepping for a stray path reference: prose/doc/source
# files a human or agent might type a citation-style path into. Dotfiles
# (e.g. ``.gitignore``, which legitimately lists ``/internal/`` as an
# ignore rule -- the mechanism that keeps it private, not a leak) and
# extensionless files are skipped by construction, since they never match
# this set.
_SCANNED_SUFFIXES = frozenset({".md", ".py", ".tex", ".ipynb", ".toml", ".cff", ".rst", ".txt"})


def _public_tracked_files() -> list[Path]:
    """Every git-tracked file outside this repo's private directories.

    Uses ``git ls-files`` (not a filesystem walk) so untracked scratch
    files never masquerade as "public", and filters out the same
    directories this project's fresh-history export script excludes from
    the public export tree (its ``plan`` and ``.claude`` top-level
    directories) plus this repo's own ``internal`` directory (a second,
    equally private documentation directory that also never ships).
    """
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    files: list[Path] = []
    for raw in proc.stdout.split(b"\x00"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="replace")
        rel_path = Path(rel)
        if any(part in _PRIVATE_DIR_NAMES or part in _SKIP_DIR_NAMES for part in rel_path.parts):
            continue
        if rel_path.suffix not in _SCANNED_SUFFIXES:
            continue
        files.append(REPO_ROOT / rel_path)
    return files


def _changelog_unreleased_span(text: str) -> tuple[int, int] | None:
    """Byte offsets of ``CHANGELOG.md``'s "Unreleased" section body.

    Returns ``(start, end)`` spanning from just after the ``##
    [Unreleased]`` heading to just before the next ``## `` heading (the
    first already-released version), or ``None`` if no "Unreleased"
    heading is found. Older, already-released history entries are allowed
    to keep private-directory path references -- they describe this
    project's internal process at the time it happened, exactly as a
    changelog should -- so only the still-unreleased section is scanned.
    """
    m = re.search(r"^## \[Unreleased\]\s*$", text, re.MULTILINE)
    if m is None:
        return None
    start = m.end()
    m2 = re.search(r"^## (?!\[Unreleased\])", text[start:], re.MULTILINE)
    end = start + m2.start() if m2 else len(text)
    return start, end


def _scan_internal_paths_in_text(
    relpath: str, text: str, allowed: list[str], *, line_offset: int = 0
) -> list[Finding]:
    """Return one :class:`Finding` per un-allowlisted match of
    :data:`INTERNAL_PATH_RE` in ``text``. ``line_offset`` shifts reported
    line numbers when ``text`` is a slice of a larger file (e.g.
    ``CHANGELOG.md``'s "Unreleased" section)."""
    findings: list[Finding] = []
    for m in INTERNAL_PATH_RE.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.start())
        if line_end == -1:
            line_end = len(text)
        snippet = text[line_start:line_end].strip()
        if _is_allowed(snippet, allowed):
            continue
        lineno = line_offset + text.count("\n", 0, m.start()) + 1
        findings.append(
            Finding(
                relpath,
                lineno,
                f"internal-only path reference {m.group(0)!r} (excluded from the "
                f"public export) in: {snippet[:160]!r}",
            )
        )
    return findings


def internal_path_check(allowlist: dict[str, Any]) -> CheckResult:
    """Grep every public file for a literal path into a private directory.

    ADR-0005 export-gate item, mechanized here: this project's two
    private top-level directories are excluded from the public export (by
    this project's fresh-history export script, and this repo's own
    private documentation directory), so a public file naming a path
    inside either one is pointing at something that will not exist in the
    released tree. ``CHANGELOG.md``'s older, already-released history
    entries are exempt (see :func:`_changelog_unreleased_span`); only its
    "Unreleased" section is scanned like every other public file.
    """
    allow_map = allowlist.get("internal_path_check", {}).get("allow", {})
    findings: list[Finding] = []
    for path in _public_tracked_files():
        rel = str(path.relative_to(REPO_ROOT))
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        allowed = allow_map.get(rel, [])

        if rel == "CHANGELOG.md":
            span = _changelog_unreleased_span(text)
            if span is None:
                continue
            start, end = span
            findings.extend(
                _scan_internal_paths_in_text(
                    rel,
                    text[start:end],
                    allowed,
                    line_offset=text.count("\n", 0, start),
                )
            )
        else:
            findings.extend(_scan_internal_paths_in_text(rel, text, allowed))

    return CheckResult("internal-path-check", _status_from_findings(findings), findings)


# ---------------------------------------------------------------------------
# 6. determinism-check
# ---------------------------------------------------------------------------

DETERMINISM_TARGETS: list[tuple[str, str]] = [
    ("benchmarks/run_benchmarks.py", "benchmarks/results/wp10_results.json"),
    (
        "benchmarks/run_bbr_jila_arithmetic_reproduction.py",
        "benchmarks/results/wp20_bbr_arithmetic_reproduction.json",
    ),
    ("benchmarks/run_roos_quadrupole_slope.py", "benchmarks/results/roos_quadrupole_slope.json"),
    ("benchmarks/run_bothwell_redshift.py", "benchmarks/results/bothwell_redshift.json"),
]
IGNORED_JSON_KEYS = {"generated_at_utc"}


def _normalize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize_json(v) for k, v in obj.items() if k not in IGNORED_JSON_KEYS}
    if isinstance(obj, list):
        return [_normalize_json(v) for v in obj]
    return obj


RESULTS_DIR_REL = "benchmarks/results"


def _snapshot_dir(dir_path: Path) -> dict[Path, bytes]:
    if not dir_path.exists():
        return {}
    return {p: p.read_bytes() for p in dir_path.rglob("*") if p.is_file()}


def _restore_snapshot(dir_path: Path, snapshot: dict[Path, bytes]) -> None:
    """Put every file back exactly as ``snapshot`` recorded it, and delete
    anything that appeared since (there should not be anything new, but this
    keeps the guarantee absolute rather than best-effort).
    """
    for path, content in snapshot.items():
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)
    if dir_path.exists():
        for p in dir_path.rglob("*"):
            if p.is_file() and p not in snapshot:
                p.unlink()


def determinism_check(allowlist: dict[str, Any]) -> CheckResult:  # noqa: ARG001
    """Regenerate each committed benchmark JSON via its script and diff
    against the committed content (``generated_at_utc`` ignored).

    Each script also regenerates a sibling ``.md`` summary (and possibly
    other files) as a side effect of running, so the WHOLE
    ``benchmarks/results/`` directory is snapshotted before running
    anything and unconditionally restored afterward (in a ``finally``),
    so this check never leaves the working tree modified, no matter which
    files a script touches or how it exits.
    """
    results_dir = REPO_ROOT / RESULTS_DIR_REL
    snapshot = _snapshot_dir(results_dir)
    findings: list[Finding] = []
    try:
        for script_rel, json_rel in DETERMINISM_TARGETS:
            script_path = REPO_ROOT / script_rel
            json_path = REPO_ROOT / json_rel
            if not script_path.exists() or json_path not in snapshot:
                findings.append(
                    Finding(
                        json_rel,
                        None,
                        f"missing {script_rel} or {json_rel}; cannot check determinism",
                    )
                )
                continue
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                findings.append(
                    Finding(
                        json_rel,
                        None,
                        f"{script_rel} exited {proc.returncode} while regenerating: "
                        f"{proc.stderr.strip()[-300:]}",
                    )
                )
                continue
            regenerated_bytes = json_path.read_bytes()
            try:
                orig_obj = _normalize_json(json.loads(snapshot[json_path]))
                regen_obj = _normalize_json(json.loads(regenerated_bytes))
            except json.JSONDecodeError as exc:
                findings.append(
                    Finding(json_rel, None, f"could not parse JSON for comparison: {exc}")
                )
                continue
            if orig_obj != regen_obj:
                findings.append(
                    Finding(
                        json_rel,
                        None,
                        f"regenerating via {script_rel} produced content different from the "
                        "committed file (generated_at_utc ignored)",
                    )
                )
    finally:
        _restore_snapshot(results_dir, snapshot)
    return CheckResult("determinism-check", _status_from_findings(findings), findings)


# ---------------------------------------------------------------------------
# 7. notebooks-check
# ---------------------------------------------------------------------------

NOTEBOOK_RUNTIME_BUDGET_S = 180.0


def _normalize_volatile_text(text: str, volatile_res: list[re.Pattern[str]]) -> str:
    """Replace every match of a volatile-output pattern with a fixed token.

    Applied identically to the committed and the re-executed outputs, so a
    print that is legitimately different on every run (a wall-clock timing
    line, for example) compares equal while everything around it still
    byte-compares.
    """
    for pattern in volatile_res:
        text = pattern.sub("<volatile>", text)
    return text


def _normalize_notebook_cells(
    nb: dict[str, Any], volatile_res: list[re.Pattern[str]] | None = None
) -> list[dict[str, Any]]:
    volatile_res = volatile_res or []
    cells = []
    for cell in nb.get("cells", []):
        entry: dict[str, Any] = {"cell_type": cell.get("cell_type"), "source": cell.get("source")}
        if cell.get("cell_type") == "code":
            outputs = []
            for out in cell.get("outputs", []):
                o = dict(out)
                o.pop("execution_count", None)
                metadata = o.get("metadata")
                if isinstance(metadata, dict):
                    o["metadata"] = {k: v for k, v in metadata.items() if k not in {"id"}}
                if volatile_res and isinstance(o.get("text"), (str, list)):
                    text = o["text"]
                    joined = "".join(text) if isinstance(text, list) else text
                    o["text"] = _normalize_volatile_text(joined, volatile_res)
                outputs.append(o)
            entry["outputs"] = outputs
        cells.append(entry)
    return cells


def _diff_notebook_outputs(
    orig_nb: dict[str, Any],
    new_nb: dict[str, Any],
    volatile_res: list[re.Pattern[str]] | None = None,
) -> str | None:
    a = _normalize_notebook_cells(orig_nb, volatile_res)
    b = _normalize_notebook_cells(new_nb, volatile_res)
    if a == b:
        return None
    if len(a) != len(b):
        return f"cell count differs ({len(a)} vs {len(b)})"
    for i, (ca, cb) in enumerate(zip(a, b, strict=True)):
        if ca != cb:
            return f"cell {i} outputs differ after re-execution"
    return "notebook differs after re-execution (unlocalized)"


def notebooks_check(allowlist: dict[str, Any]) -> CheckResult:
    """Re-execute each ``notebooks/*.ipynb`` and byte-compare (normalized)
    outputs against the committed version; flag any notebook whose
    re-execution runtime exceeds 180s.

    ``[notebooks_check] volatile_patterns`` in the allowlist is a list of
    regexes; every match in an output's text (committed and re-executed
    alike) is replaced with a fixed token before comparison, so prints
    that legitimately differ on every run (wall-clock timings) do not
    fail the byte-compare while the numbers around them still must match.
    """
    volatile_res = [
        re.compile(p) for p in allowlist.get("notebooks_check", {}).get("volatile_patterns", [])
    ]
    try:
        import nbconvert  # noqa: F401
    except ImportError:
        return CheckResult(
            "notebooks-check",
            "SKIP",
            [],
            detail=(
                "nbconvert not installed (pyproject.toml's 'notebooks' extra); "
                "skipping re-execution"
            ),
        )

    notebooks = public_notebook_files()
    findings: list[Finding] = []
    if not notebooks:
        return CheckResult("notebooks-check", "PASS", findings, detail="no notebooks found")

    for nb_path in notebooks:
        rel = str(nb_path.relative_to(REPO_ROOT))
        original_bytes = nb_path.read_bytes()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / nb_path.name
            start = time.monotonic()
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--ExecutePreprocessor.timeout=300",
                    "--output",
                    str(out_path),
                    str(nb_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            elapsed = time.monotonic() - start
            if proc.returncode != 0:
                findings.append(
                    Finding(rel, None, f"re-execution failed: {proc.stderr.strip()[-400:]}")
                )
                continue
            if elapsed > NOTEBOOK_RUNTIME_BUDGET_S:
                findings.append(
                    Finding(
                        rel,
                        None,
                        f"re-execution runtime {elapsed:.1f}s exceeds the "
                        f"{NOTEBOOK_RUNTIME_BUDGET_S:.0f}s budget",
                    )
                )
            new_bytes = out_path.read_bytes()

        try:
            orig_nb = json.loads(original_bytes)
            new_nb = json.loads(new_bytes)
        except json.JSONDecodeError as exc:
            findings.append(Finding(rel, None, f"could not parse notebook JSON: {exc}"))
            continue
        diff = _diff_notebook_outputs(orig_nb, new_nb, volatile_res)
        if diff is not None:
            findings.append(
                Finding(rel, None, f"re-executed outputs differ from committed outputs: {diff}")
            )

    return CheckResult("notebooks-check", _status_from_findings(findings), findings)


# ---------------------------------------------------------------------------
# 8. suite-check
# ---------------------------------------------------------------------------

PYTEST_SUMMARY_RE = re.compile(
    r"(?P<counts>\d+ (?:passed|failed|error|skipped|xfailed|xpassed|deselected)"
    r"(?:, \d+ (?:passed|failed|error|skipped|xfailed|xpassed|deselected))*)"
)
RUFF_ERROR_COUNT_RE = re.compile(r"Found (\d+) error")
MYPY_ERROR_COUNT_RE = re.compile(r"Found (\d+) error")


def _last_nonempty_line(text: str) -> str:
    lines = [line for line in text.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""


def suite_check(allowlist: dict[str, Any]) -> CheckResult:  # noqa: ARG001
    """Run pytest, ``ruff check``, ``ruff format --check``, and mypy;
    parse exact counts out of each tool's own summary line and report them.
    """
    findings: list[Finding] = []
    detail_lines: list[str] = []

    pytest_proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    # Search the whole output (not just the literal last line) for the LAST
    # summary-count match: pytest's dot/percentage progress can end up on
    # the same captured line as other text depending on terminal-width
    # wrapping in a non-tty subprocess, so the true "N passed" summary is
    # not always the last newline-delimited line.
    pytest_matches = list(PYTEST_SUMMARY_RE.finditer(pytest_proc.stdout))
    pytest_counts = (
        pytest_matches[-1].group("counts")
        if pytest_matches
        else _last_nonempty_line(pytest_proc.stdout)
    )
    detail_lines.append(f"pytest: {pytest_counts}")
    if pytest_proc.returncode != 0:
        findings.append(
            Finding("pytest", None, f"pytest exited {pytest_proc.returncode}: {pytest_counts}")
        )

    ruff_check_proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."], cwd=REPO_ROOT, capture_output=True, text=True
    )
    ruff_out = ruff_check_proc.stdout.strip()
    ruff_m = RUFF_ERROR_COUNT_RE.search(ruff_out)
    ruff_count = ruff_m.group(1) if ruff_m else ("0" if ruff_check_proc.returncode == 0 else "?")
    detail_lines.append(f"ruff check: {ruff_count} error(s)")
    if ruff_check_proc.returncode != 0:
        findings.append(
            Finding("ruff check", None, f"{ruff_count} error(s): {_last_nonempty_line(ruff_out)}")
        )

    ruff_fmt_proc = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    would_reformat = len(
        [line for line in ruff_fmt_proc.stdout.strip().split("\n") if line.strip()]
    )
    fmt_status = (
        "clean" if ruff_fmt_proc.returncode == 0 else f"{would_reformat} file(s) would reformat"
    )
    detail_lines.append(f"ruff format --check: {fmt_status}")
    if ruff_fmt_proc.returncode != 0:
        findings.append(
            Finding("ruff format --check", None, f"{would_reformat} file(s) would be reformatted")
        )

    mypy_proc = subprocess.run(
        [sys.executable, "-m", "mypy", "src"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    mypy_summary_line = _last_nonempty_line(mypy_proc.stdout) or _last_nonempty_line(
        mypy_proc.stderr
    )
    mypy_m = MYPY_ERROR_COUNT_RE.search(mypy_summary_line)
    mypy_count = mypy_m.group(1) if mypy_m else ("0" if mypy_proc.returncode == 0 else "?")
    detail_lines.append(f"mypy: {mypy_count} error(s) ({mypy_summary_line})")
    if mypy_proc.returncode != 0:
        findings.append(Finding("mypy", None, f"{mypy_count} error(s): {mypy_summary_line}"))

    return CheckResult(
        "suite-check", _status_from_findings(findings), findings, detail="\n".join(detail_lines)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CHECKS: dict[str, Any] = {
    "prose-scan": prose_scan,
    "tolerance-scan": tolerance_scan,
    "citation-check": citation_check,
    "headline-check": headline_check,
    "internal-path-check": internal_path_check,
    "determinism-check": determinism_check,
    "notebooks-check": notebooks_check,
    "suite-check": suite_check,
}
FAST_SKIP = {"determinism-check", "notebooks-check", "suite-check"}


def run_checks(names: list[str]) -> list[CheckResult]:
    """Run the named checks in ``CHECKS`` order and return their results."""
    allowlist = load_allowlist()
    ordered = [name for name in CHECKS if name in names]
    return [CHECKS[name](allowlist) for name in ordered]


def print_report(results: list[CheckResult]) -> None:
    for result in results:
        print(f"=== {result.name}: {result.status} ===")
        for finding in result.findings:
            print("  " + finding.format())
        if result.detail:
            for line in result.detail.split("\n"):
                print("  " + line)
        print()
    print("=== SUMMARY ===")
    for result in results:
        print(f"{result.name}: {result.status} ({len(result.findings)} finding(s))")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="release_checks",
        description="Mechanized release-review checks for CliffordClock (WP28).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"comma-separated subset of checks to run: {', '.join(CHECKS)}",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=f"skip the expensive checks ({', '.join(sorted(FAST_SKIP))})",
    )
    parser.add_argument("--list", action="store_true", help="list available check names and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in CHECKS:
            print(name)
        return 0

    selected = list(CHECKS)
    if args.only:
        requested = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in requested if s not in CHECKS]
        if unknown:
            parser.error(f"unknown check(s): {', '.join(unknown)}; available: {', '.join(CHECKS)}")
        selected = requested
    if args.fast:
        selected = [s for s in selected if s not in FAST_SKIP]

    results = run_checks(selected)
    print_report(results)
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
