# tools/

Standalone scripts, outside the `cliffordclock` package (`src/`), that
support development/release workflow rather than shipping in the wheel
(mirrors `benchmarks/`'s relationship to `src/`; see
`pyproject.toml`'s `[tool.setuptools.packages.find]`).

## `release_checks.py` (WP28)

Mechanized release-review checks: the greps and invocations every
release/beta review has repeated by hand, so an agent review spends its
tokens on physics and judgment instead. See the project's mechanized-
checks review checklist for how reviewers should use it, and the WP28
sprint record for the originating spec.

```bash
source .venv/bin/activate
python tools/release_checks.py              # run all eight checks
python tools/release_checks.py --fast       # skip the slow ones (6-8)
python tools/release_checks.py --only prose-scan,citation-check
python tools/release_checks.py --list       # list check names
```

Exit code is nonzero if any check's status is `FAIL`. Checks:

1. `prose-scan`: em dash / dash-as-punctuation / honest-family words /
   configurable meta-slop phrases over public files, plus every module,
   class, and function docstring under `src/`, `tests/`, `benchmarks/`,
   and `examples/` (extracted via `ast`, wrap-aware, minus the
   dash-as-punctuation check, which false-positives on docstring
   scaffolding like a NumPy-style section underline).
2. `tolerance-scan`: bare `pytest.approx(..., rel=...)` without `abs=`
   and `assert_allclose(...)` without `atol=` over `tests/**`.
3. `citation-check`: public-file author-surname+year bylines checked
   against the pinned `bibliography.toml`.
4. `headline-check`: validation-headline phrase consistency across
   README/docs/benchmarks/RESULTS.md.
5. `internal-path-check`: greps every public file for a literal `plan/`
   or `internal/` path reference (both excluded from the public export;
   `CHANGELOG.md`'s already-released history entries are exempt).
6. `determinism-check`: regenerates `benchmarks/results/*.json` and
   diffs against committed content (timestamps ignored).
7. `notebooks-check`: re-executes `notebooks/*.ipynb` and byte-compares
   (normalized) outputs; flags runtime > 180s.
8. `suite-check`: runs pytest in two lanes (fast: `-m "not slow"`,
   1800s timeout; slow: `-m slow`, 5400s timeout, mirroring
   `.github/workflows/ci.yml`'s own two-job split), then ruff/mypy, and
   parses exact counts; fails if either lane, ruff, or mypy fails.

Config: `bibliography.toml` (pinned citation records) and
`release_checks_allowlist.toml` (deliberate prose-scan keeps, the
configurable meta-slop phrase list, and headline-check's frozen-artifact
allowlist). Edit those files for deliberate keeps; never special-case a
file path inside the scanner code.

Zero new runtime dependencies: standard library only, plus `nbconvert`
for `notebooks-check` (already an optional project dependency,
`pyproject.toml`'s `notebooks` extra); `notebooks-check` reports `SKIP`
rather than failing if it is not installed.

Tests: `tests/test_release_checks.py` (fixture-string unit tests per
scanner, including planted-violation cases, plus a few real-repo smoke
tests for the cheap checks and an integration test proving
`determinism-check` never leaves `benchmarks/results/` modified).
