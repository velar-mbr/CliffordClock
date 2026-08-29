# G17 gate: the coupled two-ion Floquet solve and the constrained fit (WP35)

Independent review of `coupled_two_ion_floquet_modes`,
`coupled_two_ion_mathieu_parameters`, `constrained_two_ion_mathieu_fit`,
and the WP35 benchmark case, across three rounds. The reviewer built
its own 4D coupled-Mathieu monodromy integration, its own Fourier-mode
decomposition, its own Gauss-Newton solver, and its own uncertainty
propagation, sharing no code with the module, and recomputed every
headline number in every round. Because this work flips a headline
verdict to MET, the review was run at the project's maximum severity.

## The evidentiary standard the gate enforced

The first coupled implementation solved each transverse axis
separately, two measured frequencies fixing two parameters per axis.
The review identified that such a fit reproduces its inputs by
construction, carries no leftover degree of freedom, omitted the
partner-ion cross-check its own derivation comments promised, and left
the disclosed two-percent mismatch between the two solved q values
outside the uncertainty budget. The gate failed the round on those
grounds while confirming all of its arithmetic, and the fix loop
rebuilt the headline around the physically constrained model.

## The constrained fit: PASS, verified exactly

One shared RF parameter q per ion (with the q_x = -q_y sign
convention) and the dc-split fraction alpha are the only unknowns,
with a_z fixed independently from the measured axial frequency; a
Gauss-Newton least-squares fit against all four measured radial mode
frequencies gives a genuine four-equation, two-unknown
over-determination. The reviewer reproduced the solution (q =
0.190645, alpha = 1.625194), all four frequency residuals (-782.7,
+761.9, -3862.1, +3721.0 Hz; 0.019 to 0.078 percent, reported in full
as the fit's falsifiable output), the per-mode ratios
(1.0070/1.0031/1.0220/0.9865/1.0150/0.9925), the partner-ion check
(-0.07/-0.11 percent on the primary trap, -0.18/-0.14 percent on the
independently published second trap), and the three-component budget
(thermometry 3.763e-19, rounding 1.022e-20, model structure
1.591e-19, combined 4.087e-19) to every reported digit. The total,
-1.141472e-17, sits at 0.08 sigma from the published -114.6(3.8)e-19:
MET, class arithmetic reproduction.

## Budget scrutiny

The rounding channel's drop from the per-axis model's 2.25e-19 to
1.02e-20 was verified and explained: an over-determined regression
averages any single input's influence against the other data points.
The model-structure channel, defined as the spread between the
constrained and per-axis totals, was checked against an alternative
proxy (the fit residuals propagated through the full chain), which
came out a factor of roughly 44 smaller, so the reported channel is
the conservative choice. The antisymmetric residual pattern (in-phase
modes under-predicted, out-of-phase over-predicted, on both axes and
in both traps) is disclosed in full as structured signal from physics
outside the idealized Mathieu model, with no specific mechanism
asserted.

## Verdict

PASS, after two fix loops: the evidentiary rebuild described above,
then four wording fixes in comment text. The per-axis solve is
retained as a labeled diagnostic whose caveat states plainly that it
reproduces its own inputs by construction and carries no independent
evidentiary weight. The limit checks tie the coupled solve to its
predecessors: the coupling to zero recovers WP34's single-ion exact
Floquet result, and the RF drive to zero recovers WP32's static
eigenproblem, each verified independently.
