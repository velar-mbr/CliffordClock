# G21 gate: the model entry path (MODEL.md, ten_line_clock, docs/terms, README ladder)

Independent review of the "How it works" entry path: `docs/MODEL.md`
(the one-page model statement with the canonical ten-line listing),
`examples/ten_line_clock.py` (a runnable numpy-only expansion of the
listing), `tests/test_ten_line_clock.py`, the README navigation ladder,
and the nine per-effect one-pagers plus index under `docs/terms/`. Two
reviewers worked the two halves in parallel; every FATAL finding below
was fixed, re-verified against its source, and pushed before this
record was written.

## Physics of the example: PASS after recomputation

The reviewer recomputed g·dh/c² for both printed cases (0.33 m gives
3.600751991825412e-17, 0.01 m gives 1.0911369672198218e-18) and matched
the script's output, confirmed the second-order Doppler sign and
prefactor, hand-derived the cross-term estimates (-5.4e-35 and
-1.6e-36) and the visibility (0.999995) from the ensemble parameters,
and ran the script twice with bit-identical output. The excess-phase
formulation and the `product_minus_one` accumulator were both verified
by hand as algebraic identities to the canonical listing, with no
approximation involved. All 10 tests pass and each tolerance was traced
to a failure mode that would trip it.

## FATAL findings, both fixed

The review returned two FATALs. First, MODEL.md had inverted the
codebase's own naming: it called the small per-term correction p the
pivot and hung the lapse-function identification on it, while
`omega.py` defines `pivot(...)` as the full ratio P = 1 + correction
and names the small terms `*_pivot_perturbation`. The fix restores the
convention: P carries the lapse-function and redshift-factor
identification, each p is a pivot perturbation. Second, the example's
docstring attributed both printed cases to Chou et al. 2010, Science
329, 1630, while the paper reports one measurement: a 33 cm rise
measured at (4.1 ± 1.6)e-17 against the 3.6e-17 prediction. The fix
labels the 33 cm case with both numbers in their proper roles and
presents the 1 cm case as this script's own illustration at the 1e-18
scale. Three MINORs (an unverified magnitude-reporting claim, an
imprecise "second moment" description of the visibility, and a stale
test docstring) were fixed in the same pass, and the magnitude claim
was also removed from two test docstrings the fix scope had missed.

## The one-pagers: PASS after three narrow fixes

The one-pagers reviewer checked every E-label against CONVENTIONS.md,
every displayed formula term by term against its source docstring,
every code excerpt character for character against `src/`, every
validation number against the file the page cites, and every citation
field for field against a repo source, and found three FATALs, each a
single-location fix: the sideband page quoted a verdict field literal
("MET") that the artifact does not contain (it contains "PASS"); the
coherence page overstated a test as a cross-check against
`dephasing_time_t2star` when the test compares against the Gaussian
closure computed from the sample's own phase variance; and the BBR page
displayed the multi-surface moment with raw weights where CONVENTIONS.md
§13 defines it with emissivity-corrected effective weights. Seven
MINORs (two under-attributed sources, a dropped citation year, invented
comments inside excerpts presented as quotations, an unsourced ranking
claim, function identifiers named mid-prose, and undefined KA labels)
were fixed in the same pass. A spot recheck of the three FATAL lines
against their sources confirmed the fixes before merge.

## Battery: PASS

The full eight-check battery ran on the merged branch. One finding
surfaced and was resolved: ruff 0.16 formats fenced code blocks inside
markdown and wanted a PEP 8 blank line inside the canonical listing,
which is coordinated character for character with the companion paper's
Listing 1. The fix scopes ruff to Python sources via `extend-exclude`
in pyproject.toml, since the docs' listings are deliberate pseudocode.
After the fix, all eight checks pass, including notebooks-check
re-execution and both pytest lanes.

## Verdict: PASS

The entry path is merged as one branch: MODEL.md, the runnable example
with its tests, the nine one-pagers with index, and the README ladder
linking Quickstart, How it works, the per-term pages, the notebooks,
and the validation record in that order.
