# G14 gate: two-ion radial participations reconstructed from the measured mode spectrum (WP32)

Independent physics and code review of `two_ion_radial_participations`
and `axial_coulomb_curvature` (E38 extension, CONVENTIONS.md 1.8.0),
the Al27+/Mg25+ benchmark variant, and the reconciliation question the
work was built to answer. The reviewer recomputed every number from
standard constants with an independent script, re-derived the physics
from first principles, and verified the primary sources page by page.

## Derivation: PASS

The Coulomb curvature was re-derived from scratch: expanding the
inter-ion potential to second order gives the axial-stiffening,
radial-softening-by-half form `U_quad = c dz^2 - (c/2)(dx^2 + dy^2)`
with `c = e^2/(4 pi eps0 d^3)`, and combining with the shared axial
spring constant `k_z = m_i w_zi^2` (Wubbena et al., PRA 85, 043412
(2012), Eq. 7, verified verbatim against the PDF) gives `c = k_z / 2`
exactly, with no permittivity constant surviving. Wubbena Eqs. 12-14
(axial closed forms) were verified verbatim, and Eqs. 15-18 confirm
the radial closed form requires trap-geometry parameters unavailable
from the measured spectrum, which is the justification for WP32's
inversion approach.

## Numbers: PASS, exact

Every claimed value reproduced independently: the Coulomb curvature
`3.976554191127463e-12 N/m` and its axial-stretch cross-check at
`1.736e-3` relative; both branches' reconstructed bare frequencies
(X: clock 3.946772 MHz, partner 4.360932 MHz; Y: clock 5.084691 MHz,
partner 5.497386 MHz) with the lighter partner higher in both, as the
disambiguation rule requires; participations (X: 0.21383/0.78617,
Y: 0.16629/0.83371) confirmed a second way by direct 2x2
eigendecomposition to 1e-13 relative; the corrected total
`-5.716945188613549e-18 +/- 1.535e-19`, 14.0 sigma from the published
`-114.6(3.8)e-19`, reported NOT MET with no tuning.

## Guards: PASS, kill-tested

Beyond the shipped tests, the reviewer flipped the clock/partner roles
(participations swap exactly), forced the wrong disambiguation branch
(ratios degrade to 0.09x-3x, confirming the chosen branch is physical),
and drove every guard: equal masses, equal frequencies, infeasible
frequencies, negative curvature, and ambiguity within propagated
uncertainty all raise the documented errors. Lint, format, mypy, the
72 motional tests, and the fast release-check battery are clean, with
zero new prose findings in the diff.

## Reconciliation finding: the radial factor is intrinsic micromotion, and it partially closes

The per-mode ratios (axial at 1.007 and 1.003; radial at 0.42, 0.35,
0.52, 0.50) pointed at a near-factor-two on radial modes. The
predecessor paper from the same experimental lineage (Brewer et al.,
arXiv:1902.07694, same species pair) carries the confirming footnote
on its equivalent table: the transverse values include the shift due
to intrinsic micromotion. Testing the candidates quantitatively on
both papers' data: full-energy attribution fails even axially (Brewer
axial off by 1.8x-2.1x), participation weighting matches both papers'
axial rows to a few percent, and participation times two closes the Y
branch in both independent datasets (Marshall 1.03/1.00, Brewer
0.92/0.97) while leaving the X branch short by a reproducible 20-35
percent in both. The residual has an identified mechanism: the
leading-order intrinsic-micromotion enhancement is mode-specific,
`1 + q^2/(2a + q^2)`, equal to two only when the Mathieu a-parameter
vanishes, and the trap's dc asymmetry splits a_x from a_y. The path to
computing that factor per axis from published inputs (the paper's
stated drive frequency with the reconstructed bare radial frequencies)
is the follow-on work item, WP33.

## Verdict

PASS. The implementation, physics, guards, and artifacts are correct
as shipped, with the honest NOT MET verdict and its now-understood
mechanism recorded in the caveats. The reconciliation is a physics
finding, and the mode-specific enhancement factor is the identified
next step.
