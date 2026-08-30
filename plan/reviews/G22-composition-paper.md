# G22 gate: the composition companion paper (paper/composition)

Independent review of `paper/composition/main.tex`, a six-page
standalone companion paper comparing the field's additive error-budget
practice with the engine's composition law: one scalar proper-time-rate
factor per atom worldline, every systematic a multiplicative factor,
the fractional shift as the mean of the ensemble phase distribution and
the Ramsey visibility as the modulus of the mean phasor. The reviewer
checked the mathematics, traced every case-study number to its claimed
repo source, verified all ten citations, rebuilt the PDF, and read the
prose cold. Every FATAL finding was fixed, re-verified, and pushed
before this record was written.

## Mathematics and copy fidelity: PASS

The composition law, the first-order expansion, and the Gaussian
closure were verified as stated. The E37 BBR numbers (the 11 K and 19 K
mismatch cases, the PTB comparison at -3.325e-15 against the published
-3.32(7)e-15 and -3.33(3)e-15 bands) match paper/main.tex and were
re-generated from the live engine. The Al+ six-row sigma table (0.10,
14.11, 14.01, 1.62, 1.62, 0.08) matches notebook 13's generated table
row for row, verdicts included. The E39 visibility case (N = 4000, five
swept r values, 0.65% worst deviation) matches notebook 13. The
Bothwell 0.85-sigma method disagreement was recomputed from the repo's
own quoted values and confirmed at 0.846 sigma. Listing 1 matches the
canonical ten-line listing byte for byte.

## Citations: PASS after one FATAL

The two citations new to the repo were web-verified against publisher
records: Ashby, Living Rev. Relativity 6, 1 (2003), confirmed via its
ADS record, and Mehlstäubler, Grosche, Lisdat, Schmidt, Denker, Rep.
Prog. Phys. 81, 064401 (2018), confirmed via the IOPscience page with
author order intact. The FATAL: refs.bib claimed its eight shared
entries were copied verbatim from paper/refs.bib while every one had
dropped its doi field and five had also dropped eprint fields. The fix
re-copied all eight entries and verified each diff-identical to its
source, making the provenance claim true.

## Arithmetic FATAL, fixed

The paper summarized its two cross-term bounds (1e-33 for Stark times
BBR, 1e-31 for gravitational times field) as sitting roughly fifteen
orders of magnitude below the 1e-18 to 1e-19 floor, in the abstract and
the body. Fifteen holds for the first bound only; the second sits
twelve to thirteen orders below. All three occurrences now read "at
least twelve orders of magnitude," which holds for both. The underlying
bounds themselves were copied faithfully and were not changed.

## Claims discipline: PASS

No priority claims anywhere; the scope paragraph is present in abstract
and body (the cross terms are far below today's floor, the value of the
product form is structural, and no published total is being corrected);
the prior-practice section credits relativistic timing chains and
chronometric geodesy as precedent and claims scope only; Clifford
algebra, geometric algebra, and rotors appear nowhere outside the
product name; MET and NOT MET verdicts match the repo's artifacts.

## Scanner gap, closed

The prose scanner had hardcoded its .tex target to paper/main.tex, so
the new paper was invisible to the automated gate. The scanner now
iterates both paper paths, three new tests in test_release_checks.py
cover the extension including a planted violation attributed to the new
file, and the newly scanned paper surfaced zero fatal findings. Four
"second moment" descriptions of the visibility were also reworded to
mean-and-variance language during the fix pass, matching the corrected
MODEL.md wording on the sibling branch.

## Battery: PASS

The full eight-check battery ran on the branch after the fixes: all
eight checks pass, including notebooks-check re-execution and both
pytest lanes. The rebuilt PDF is six pages with no LaTeX errors and no
undefined references.

## Verdict: PASS
