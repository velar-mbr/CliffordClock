# G24 gate: full Rydberg Stark maps (WP40)

Independent review of `cliffordclock.integrator.rydberg_stark_map`
(quantum-defect Hamiltonian assembly, Marinescu model potential,
inward Numerov radial integrals, adiabatic diagonalization with
overlap-based connectivity, crossover detection, convergence sweeps,
and the map-sourced E44 shift path), the fix this work applied to
Phase A's Numerov integrator, the WP40 benchmarks and ARC fixture
generator, CONVENTIONS.md section 20, and the docs pages. The reviewer
re-derived the Numerov recurrence by hand, reran every benchmark,
verified the ARC fixture bit-for-bit against a live pinned
installation, and checked every equation and coefficient against the
source PDFs and the pinned ARC source code.

## The Numerov corrections: PASS

Cross-validating against ARC exposed two integration errors. First, a
latent sign error in Phase A's outward Numerov recurrence, present in
shipped code: the reviewer re-derived the standard recurrence,
confirmed the pre-fix form wrong, and reproduced the builder's
exact-solution check (eight orders of magnitude divergence pre-fix,
~1e-10 agreement post-fix). Blast-radius analysis confirmed the only
consumer was the disclosed factor-of-two dipole-moment cross-check,
whose ratio moves from 1.489 to 1.728 across the fix, inside its
stated tolerance both ways, so no gated Phase A result changes.
Second, outward integration is unstable past the classical turning
point for this module's coupled-manifold use; the build switched to
inward integration from the outer boundary, the direction ARC's own
paper states with its reason, which the reviewer verified verbatim in
the source PDF. Post-switch, matrix elements agree with ARC to within
1% where they had been wrong by orders of magnitude.

## Transcription and sources: PASS, after one fix loop

Hamiltonian assembly, angular-momentum algebra, and the Rb model
potential and quantum defects were verified against the ARC paper and
the pinned ARC source byte-for-byte. Zimmerman et al. 1979 appears
only as historical origin, never at equation level, per the standing
rule for a source that was not directly readable. The gate found one
FATAL in the sources list: a fabricated author initial (W. J. for
O'Sullivan, whose verified initials are M. S., as the project's own
bibliography already carried). The fix loop corrected it and swept the
tree for other occurrences.

## Crossover and validity window: PASS

The computed quadratic-to-map crossovers replace the Phase A
order-of-magnitude estimates: 115.80, 82.65, 50.34, and 7.97 V/cm for
n = 30, 32, 35, 50, each above its Inglis-Teller estimate as expected
for an under-guard. The map's small-field curvature reproduces the
Phase A registry polarizabilities to 4.91% worst case against a 15%
tolerance, kill-tested both ways. The n = 35 crossover's exact-digit
agreement with ARC was interrogated: the two overlap curves differ at
neighboring grid points and land in the same ~1.5 V/cm grid bin, a
quantization coincidence, and CONVENTIONS now carries that caveat.

## ARC cross-validation: PASS

Eigenvalue agreement against ARC at the pinned release (v.3.10.2,
commit 4b4573e) reaches 2.05% worst case in the gated low-field tier
against a 5% tolerance, class independent implementation
reproduction. Beyond that tier, independently built Hamiltonians
legitimately track onto swapped branches through crossings; that
behavior is reported with its reasoning and deliberately not
tolerance-gated. The committed fixture was verified genuine by
regenerating one case live against the pinned installation,
bit-identical.

## Published anchors: PASS

The composite the formalism dossier prescribed: the Holloway low-field
endpoint reproduces at 1.44% against 10%; the O'Sullivan and Stoicheff
nS crossing-field method check lands at 21.5% against a 25% tolerance
whose width is disclosed with its reason (a method check on a
different angular momentum); the Grimmel supplementary-data fetch
returned no machine-readable data, so that comparison is qualitative,
class computable comparison, with the live-network fetch's
reproducibility caveat documented.

## Convergence: PASS, extended by the fix loop

The production basis is stable to 0.1% against enlargement for the
32D5/2 and 50D5/2 states at low field. The gate flagged that the
committed study never exercised the 50D5/2 state near its own
crossover, the regime the dossier called load-bearing; the fix loop
added a crossover-field stability sweep to the benchmark, which shows
7.9707 V/cm stable to better than 0.001% from the second-smallest
basis onward, matching the gate's own independent measurement.

## Prose and truthfulness: PASS, after one fix loop

The build's first push carried nineteen FAIL-severity prose findings
in its own new files (dash punctuation and one banned word), which
broke an existing repository gate test while the branch claimed all
checks green; the builder corrected the prose and the fix loop
re-verified, reworded the verification claims to final-state truth,
and corrected two drifted test counts (43 new tests and 40 Phase A
tests, confirmed by collection).

## Battery: PASS

The full eight-check battery is green on the branch after merging the
base branch's notebook citation fix; both pytest lanes, ruff, format,
and strict mypy clean.

## Verdict: PASS
