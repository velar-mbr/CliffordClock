# G18 gate: two community lattice-light-shift models, harmonic/operational and BO+WKB (WP36 Phase 1)

Independent review of `cliffordclock.integrator.lattice_light_shift`
(E40/E41, CONVENTIONS.md 1.10.0), `benchmarks/run_lattice_light_shift.py`,
and `tests/test_lattice_light_shift.py`. The reviewer recomputed every
transcribed equation against the typeset source PDFs directly and
reimplemented Model B's BO+WKB machinery independently, checking the
benchmark numbers with the reviewer's own solvers.

## Transcription fidelity: PASS, with the zeta-form finding

Every equation cited under E40/E41 (Ushijima et al. 2018 Eqs. 1-2;
Beloy et al. 2020 Eqs. 1-2, 4-6, 8-11, 13, 16-21; Bothwell et al. 2025
Eqs. 1, 6 and Appendix A Table I) was checked character by character
against the typeset PDF text, including the two equations (Beloy Eqs. 4
and 11) the project's own research dossier had flagged as unclear in an
earlier ar5iv-based extraction. Both resolve cleanly in the typeset
source and match the implementation.

The reviewer's closest scrutiny fell on the two radial-thermal reduction
factors, since a mismatched form here would silently corrupt Targets 1
and 2 without any visible symptom. Ushijima et al. 2018's own Eq. 2 gives
the LINEAR form `zeta_j(u) ~= 1 - j*kB*Tr/(u*E_R)`; Kim et al. 2023's main
text ("use of an effective depth, `uj = (1 + j*kB*Tr/(u0*Er))^-1 * uj0`")
and Bothwell et al. 2025's Eq. 1 context give the exact RECIPROCAL of
that form. Both are real, independently attested formulas in the primary
sources themselves, and the code keeps them fully separate:
`ushijima_reduction_factor` implements the first and
is the only one Target 1 calls; `jila_reduction_factor` implements the
second and is the only one Target 2 calls. The reviewer traced every call
site to confirm no path mixes the two.

## Numbers: PASS, independently recomputed at all four table points

Targets 1 and 2 were recomputed in full from the cited coefficients with
the reviewer's own script, not by re-running the shipped code and
checking its output against itself. Target 1 (Ushijima et al. 2018's
`nu_LS(u,delta_L,0)=0` and `d(nu_LS)/du=0` solved simultaneously)
reproduces `u_op=71.7` against the published `72(2)`, `delta_L_op=5.29
MHz` against the published `5.3(2) MHz`, both inside the stated
uncertainty. Target 2 (Kim et al. 2023's coefficients through the
reciprocal reduction factor at `u0=15.06`, `delta_L=10.5 MHz`,
`Tr=120 nK`) gives `-0.056(2.22)e-19` against Aeppli et al. 2024's
published `-0.1(3.2)e-19`; the bands overlap.

Model B's BO+WKB machinery (finite-difference axial solve, WKB
turning-radius density of states, the numerically-stabilized thermal
average) was reimplemented independently and run at all four of Bothwell
et al. 2025's own Appendix A Table I points (`u0` = 56.8, 66.4, 86.2,
112.2 `E_R`; `Tr` = 650, 550, 600, 720 nK), Yb-171, `n_z=0`. The
reviewer's own solver matches the shipped implementation's claim: worst
case `8.84e-4` relative error against the published BO+WKB column across
all twelve values (four rows times X/Y/Z), comfortably inside the 1%
case tolerance and genuinely better than 0.1 percent. This is the
strongest single check in the work package: an independent published
cross-check table, reproduced end to end with zero fitted parameters, on
numbers this review independently recomputed from the reviewer's own
solver.

## Species trap: confirmed, and now docstring-documented

The build record's own account of an earlier Sr-87-vs-Yb-171 species
mismatch (5-15% disagreement, closer to the harmonic column than the
BO+WKB one) was reproduced by the reviewer: `X`/`Y`/`Z` cancel atomic
mass and lattice waist exactly out of their defining ratio, but the
species' own recoil energy `E_R` still sets the `kB*Tr/E_R`
thermal-weighting scale, so reusing a published `(u0, Tr)` pair against
the wrong species evaluates a different physical trap depth with no
other symptom. This fix loop adds the warning directly to
`make_site_potential`'s and `axial_thermal_factors`'s own docstrings (not
only CONVENTIONS.md), so a caller reading `help()` output sees it without
opening the conventions document.

## Class calibration: PASS

Target 3a (Bothwell et al. 2025's own Table I) is classified
`arithmetic_reproduction`: a closed-form evaluation against the paper's
own published inputs, checked against that paper's own published result,
zero fitted parameters. Target 3b (the paper's headline `alpha~M1E2`
coefficient) is held at `computable_comparison`, deliberately distinct
from `arithmetic_reproduction`: the published `-1.41(9)e-18`/`-1.45(8)e-18`
pair is the output of Bothwell et al.'s own nonlinear fit against raw,
unpublished scan data, and no published input lets this project rerun
that fit. What is computed instead, both models evaluated at the paper's
own stated operating conditions using each model's own published
coefficient column, is correctly reported as a narrower, different claim
from the coefficient-level number it sits beside. The classification
boundary is drawn in the right place and stated plainly in both the
benchmark output and CONVENTIONS.md.

## Verdict: PASS after one fix loop

The physics, the transcriptions, and the numbers were correct on first
submission; this gate FAILED only on prose. The fix loop removed eleven
trailing negated tails across CONVENTIONS.md section 17,
`lattice_light_shift.py`, and `run_lattice_light_shift.py`, each rewritten
to state the positive fact directly (a ruled-out alternative that matters
now gets its own sentence with a reason, per house style); corrected one
genuine overclaim, CONVENTIONS.md's "reproduces... exactly, all four table
rows" contradicted its own next sentence's "better than 1e-3 relative,"
now reads "to better than 0.1 percent, all four table rows," which is
what the numbers actually show; and added the species-trap warning to the
two docstrings named above. `ruff check .`, `ruff format --check .`,
`mypy src/`, the full `test_lattice_light_shift.py` suite (28 fast tests
plus both slow tests run once each, all 30 green), and
`tools/release_checks.py --fast` (prose-scan, tolerance-scan,
citation-check, headline-check, internal-path-check) all PASS with zero
new findings; the four remaining prose-scan MINOR hits are pre-existing,
outside this diff, in sections 13/14/15 of CONVENTIONS.md.

**Flag for the next rebase.** Section 17 and E40/E41 were numbered against
the highest section/equation number on `main` at the time this branch was
cut (section 16, E39). If another work package lands on `main` first and
also claims section 17 or E40/E41, this section needs renumbering at
merge/rebase time, a mechanical, purely numerical fix that carries no
physics or prose content of its own.
