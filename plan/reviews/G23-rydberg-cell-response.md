# G23 gate: Rydberg vapor-cell response (E43/E44, WP39 Phase A)

Independent review of `cliffordclock.integrator.rydberg_cell_response`
(E43 quadratic Rydberg Stark term, E44 EIT and Autler-Townes observable,
CONVENTIONS.md section 19), `benchmarks/run_rydberg_cell_response.py`,
`tests/test_rydberg_cell_response.py`, notebook 16, and the docs/terms
page. The reviewer downloaded and read the primary PDFs (Holloway et al.
2014, Mohapatra et al. 2007, the Sedlacek et al. 2012 preprint, Mack et
al. 2011, Sanguinetti et al. 2009, Yerokhin et al. 2016, and the
Fleischhauer, Imamoglu, Marangos Rev. Mod. Phys. 77, 633 (2005) review),
re-derived the load-bearing Doppler result from first principles,
recomputed every calibration and cross-check number by hand, and ran
deliberate physics breaks against the test suite.

## The Doppler-mismatch resolution: PASS

The primary sources disagree on the thermal-vapor scaling of the
observed Autler-Townes splitting: Holloway et al. 2014 (Eq. 12, page 7)
and Mohapatra et al. 2007 give a lambda_c/lambda_p reduction, while the
Sedlacek et al. 2012 preprint's prose states the reciprocal. The build
resolved this from the two-photon resonance condition for a moving atom
in the counter-propagating ladder: the probe-resonant velocity class
maps the coupling leg's Doppler shift onto the probe-detuning axis, so
the splitting appears compressed by lambda_c/lambda_p in probe-detuning
space. The reviewer reproduced this derivation independently and
confirmed the code implements the reduction direction. Sedlacek's
reciprocal statement carries no equation number and no corroboration,
and is treated as an error in an informal explanation. A dedicated test
confirms the reciprocal form cannot fit Holloway's own printed
calibration data; the reviewer flipped the factor in source and
confirmed the test fails.

## Formula transcription: PASS

The EIT susceptibility was checked symbol by symbol against the RMP
review (Eq. 13, page 639) and Holloway et al.'s Eqs. (1)-(4) and (12)
against the arXiv PDF. The fix loop added a code-level check the
original build had only claimed in a docstring: the module's
Omega_RF = 0 reduction equals the RMP three-level form times an exact
factor of two under the recorded variable correspondence, verified over
a detuning grid at float precision and encoded as
`test_three_level_reduction_matches_rmp_pole`.

## Calibration reproduction (C3): PASS, with the claim scoped

All three of Holloway et al. 2014 Fig. 15's printed (splitting, field)
pairs reproduce with residuals 0.349%, 0.148%, and 0.200% against a 1%
tolerance; the reviewer recomputed all three by hand and matched the
artifact. The transition dipole moment is backed out of the same three
pairs, so the check's evidentiary content is over-determination: three
pairs constrain one parameter, and the three individually backed-out
values agree to 0.6%. The artifact, notebook, and CONVENTIONS wording
state this scoping, class arithmetic_reproduction, and the reviewer
found the wording scrupulous. An independent Numerov quantum-defect
calculation cross-checks the dipole moment at a disclosed
pure-Coulomb-approximation tolerance.

## Polarizability cross-check (C4): PASS

Every tabulated (n, alpha0) pair was verified against Yerokhin et al.
2016 Table IV in the PDF. The two independent source lineages (the
O'Sullivan and Stoicheff measurements as cross-tabulated by Yerokhin,
and Yerokhin's own Dirac-Fock theory) agree at 2.88%, 1.98%, and 0.35%
for n = 30, 35, 50 against a 5% tolerance. The derived
alpha0(32D5/2) = 1.4146e10 a0^3 power-law interpolation is labeled as
derived throughout. The fix loop restored the n = 30 experimental
uncertainty (0.936(8)e10 a0^3) that the first build had dropped.

## Citations: PASS, after one fix loop

The gate found one FATAL: the Yerokhin et al. 2016 entry carried a
fabricated title in three files, while its byline, venue, DOI, and
every numeric value drawn from its Table IV were correct. The fix loop
replaced the title with the published one ("Electric dipole
polarizabilities of Rydberg states of alkali-metal atoms", Phys. Rev. A
94, 032503 (2016)), re-verified against the arXiv record, and a
tree-wide search confirms no occurrence of the fabricated string
remains. All other new bibliography entries were verified byline-exact
against their arXiv pages, including the disclosed substitutions for
paywalled sources (Mack et al. 2011 and Sanguinetti et al. 2009 in
place of Li et al. 2003; Yerokhin's cross-tabulation in place of the
O'Sullivan and Stoicheff originals). Page citations for Holloway's
Eq. (12) were corrected to page 7 in both files that carried them.

## Limit checks and kill-tests (C5): PASS, after one fix loop

Zero field returns the unperturbed line byte-identically and a uniform
field returns a pure shift with zero added width; sign-flip and
Doppler-direction breaks are caught by the suite, verified by breaking
each in source. The gate found the original doubled-coefficient check
vacuous at the formula level: halving the physical 1/2 prefactor inside
the shift function passed all 38 tests. The fix loop added a
magnitude-level pin computing the expected shift independently inside
the test from the registry coefficient and SI constants; re-applying
the break now fails that test and only that test. CONVENTIONS section
19's description was corrected to match what is enforced.

## Demonstrator (C6): PASS

The wall-patch field produces the reported phenomenology of Patrick et
al., arXiv:2502.07018: the line shift and the per-atom Stark spread
both grow monotonically with patch charge and with shrinking cell
radius, deterministic across reruns. The comparison is labeled
computable_comparison, pinned by a test, and the disclosed deviations
(illustrative field scales above the paper's near-wall estimates; a
spread-based broadening metric) are stated where the comparison
appears. The notebook kicker names the surface-charge problem as
current, supported by a literature check that found no general fix
through 2026.

## Species and isotope labels: PASS, after one fix loop

Mack et al. 2011's Table V is 87Rb; a test docstring had labeled it
85Rb. The fix loop corrected the label and recorded the cross-isotope
justification from Mack's own side-by-side 85Rb and 87Rb fits.

## Prose: PASS, after one fix loop

The read-through found twelve trailing-negated-appositive tails in
CONVENTIONS section 19, a banned construction in the notebook opener,
one meta-commentary cell opener, one hollow intensifier, and one tail
on the docs/terms page. All were rewritten under the project standard;
the section 19 tail count is now zero. Notebook edits went through
nbformat and the notebook re-executed cleanly.

## Battery: PASS

The full eight-check battery is green on the branch: prose, tolerance,
citation, headline, internal-path, determinism, notebooks re-execution,
and both pytest lanes with ruff, format, and strict mypy clean. The
suite carries 40 tests for this capability.

## Verdict: PASS

E43 and E44 are gate-passed on branch rydberg-cell-response: the
Doppler-mismatch physics resolved and kill-tested, three-pair
calibration reproduced at the sub-0.35% level with its evidentiary
scope stated, polarizabilities anchored to two independent lineages,
and the surface-charge demonstrator delivering the documented
phenomenology under a pinned qualitative label.
