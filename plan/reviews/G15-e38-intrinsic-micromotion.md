# G15 gate: mode-specific intrinsic-micromotion enhancement (WP33)

Independent physics and code review of `clock_ion_mathieu_parameters`,
`radial_micromotion_enhancement`, and
`predicted_partner_bare_radial_frequencies_hz` (E38 extension,
CONVENTIONS.md 1.9.0), the third Al27+/Mg25+ benchmark variant, and the
composition argument behind it. The reviewer recomputed every number
with an independent script, verified every cited equation against the
primary Berkeland PDF, and ran its own kill-tests beyond the shipped
suite.

## Conventions and sources: PASS, verbatim

Every equation citation checked against Berkeland et al., J. Appl.
Phys. 83, 5025 (1998): the Mathieu equation (Eq. 4), the trap-specific
`a_x = a_y = -(1/2) a_z` (Eq. 5) with the project's generalization to
`a_x + a_y = -a_z` re-derived independently from Wubbena's asymmetric
dc convention (coefficients trace to zero for any split), `q_z = 0`
(Eq. 6, so the axial factor is identically one), the leading-order
secular frequency (Eq. 9), and the kinetic-energy bracket (Eq. 10),
which is algebraically identical to `F = 1 + q^2/(2a + q^2)` and
consistent with Eq. 12's doubled-energy special case. The solved
`q ~ 0.19-0.25` and `|a| ~ 0.003-0.008` sit inside the stated
leading-order regime, below Berkeland's own worked `q ~ 0.28` example.

## Numbers: PASS, exact to every digit

The Mathieu solve (q = 0.19127799732774156, a_x =
-0.005884496084657496, a_y = 0.0023025499448954367, a_z =
0.0035819461397620595), the per-axis factors (F_x =
2.474206597939098, F_y = 1.8882050328916802), the per-mode ratios
(1.04/0.86/0.98/0.94 Marshall; 0.88-0.93 Brewer), and the corrected
total (-1.063848e-17 +/- 3.373e-19, 1.62 sigma from the published
-114.6(3.8)e-19, NOT MET with bands nearly touching) all reproduced
independently. Both papers' RF drive frequencies and Brewer's full
Table S2 transcription verified verbatim against the fetched texts.

## The over-determination check: PASS, and shown to be as tight as the inputs allow

The solved clock-ion trap parameters predict the partner ion's bare
radial frequencies at -0.569/-0.533 percent (Marshall) and
-0.949/-0.789 percent (Brewer) against WP32's independent
reconstruction. The reviewer's own sensitivity analysis, perturbing
each 3-significant-figure input frequency by half its last digit,
moves the deviation by up to 0.2 percentage points per input, so the
sub-1-percent agreement is exactly the precision the published inputs
support.

## Composition and residual: PASS, with the justification now written down

The reviewer judged the central inference sound: intrinsic micromotion
is a per-ion kinematic response whose modulation factor depends only
on the ion's own Mathieu parameters, while the Coulomb coupling enters
only the RF-cycle-averaged secular dynamics, so the clock ion's
participation-weighted share of a coupled mode carries the same
per-ion factor, and the parameters correctly derive from the bare
frequencies. That argument, previously implicit, is now stated in the
derivation comment block per the review's recommendation. The
residual (x_str at -14 percent, total at 1.62 sigma) is characterized
as consistent with next-order Mathieu corrections (order q^2, a few
percent) amplified on the X axis by its small negative a_x, with the
input-rounding sensitivity of the same size; the review found no
evidence of a missing physical mechanism, and the caveat text reports
the closure as materially narrowed without claiming it complete.

## Fix loop

PASS, after one fix loop: the review found two banned constructions in
the new derivation comment block; both were reworded and the
composition justification added in the same pass, with lint, types,
the 96 motional tests, and the fast release battery re-verified green.
