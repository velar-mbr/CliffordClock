# G16 gate: numerically exact Floquet enhancement and input-rounding uncertainty (WP34)

Independent physics and code review of `mathieu_floquet_solve`,
`radial_micromotion_enhancement_exact`,
`clock_ion_mathieu_parameters_exact`, and the rounding-bound
uncertainty channel. The reviewer wrote its own continued-fraction
solver and an independent monodromy-matrix integration, recomputed
every headline number, and probed the physics behind the residual's
structure.

## Math: PASS, cross-checked to 1e-14

The exact characteristic exponent agrees between the reviewer's own
continued-fraction and monodromy implementations to 1e-14 across the
solved parameter points and Berkeland's own worked example. The q to
zero limit gives F = 1, the small-q expansion reduces to Berkeland's
bracket, and the worked example lands at F = 1.9835 against the
leading-order 1.98.

## Numbers: PASS, exact

The exact solve (q = 0.190083, a_x = -5.809e-3, a_y = +2.227e-3), the
per-axis factors (F_x = 2.46768, F_y = 1.89344), the per-mode ratios,
the total (-1.06359e-17), both uncertainty components (thermometry
3.368e-19; rounding 7.32e-21, a factor 46 below thermometry), the
1.62 sigma NOT MET verdict, and the improved Brewer partner
deviations all reproduced independently to every reported digit.
WP30-33 artifacts confirmed bit-for-bit frozen, and the pre-WP34
module confirmed byte-identical as a file prefix.

## The review's physics finding

Probing whether the within-axis opposite-sign deviations could ever
be produced by a mechanism of the implemented class, the review
derived a mode-frequency-dependent enhancement candidate: Berkeland's
factor assumes the ion oscillates at its own bare Floquet frequency,
while an ion in a collective mode runs at the Coulomb-shifted mode
frequency, which differs between the in-phase and out-of-phase modes
of one axis. A rough evaluation of that candidate moved the total to
within roughly 0.2 sigma of the published value. The finding was
labeled well motivated and unconfirmed, and it became WP35's build
specification.

## Verdict

PASS, after one fix loop: five wording violations in new text were
reworded, and the structural claim's scope was narrowed from
Mathieu-order physics in general to the single-per-axis-enhancement
model actually implemented, since the review's own candidate is
Mathieu-order physics the broader wording would have foreclosed. The
residual attribution G15 carried (next-order Mathieu scale) is
superseded by this gate: the next-order terms are now computed and
small, and the remaining structure pointed at the mode-frequency
mechanism WP35 resolves.
