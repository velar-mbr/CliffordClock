# G25 gate: the differentiable Rydberg chain and field reconstruction (WP41)

Independent review of `cliffordclock.integrator.rydberg_cell_response_jax`
(the JAX port of the quadratic Stark, ladder susceptibility, Doppler
averaging, and per-atom composition, plus a three-parameter
differentiable cell-field model), `benchmarks/run_rydberg_field_reconstruction.py`
and its artifacts, the test suite, notebook 17, and CONVENTIONS.md
section 21. The reviewer reran every benchmark, probed adversarial
gradient points beyond the builder's own suite, audited the
finite-difference step tuning with an independent seven-step
sensitivity scan, and broke the uncertainty machinery deliberately.

## Scope decision carried from the pre-task

The chain is quadratic-path only. The pre-task's dossier records the
reason: the JAX eigensolve's reverse-mode gradient has a documented
ill-conditioning failure mode near degenerate eigenvalues, the exact
regime a Stark map's avoided crossings occupy. Extending gradient-based
fitting beyond the quadratic window is a separate future step with its
own sub-gate, and the notebook states this scope.

## Agreement and gradients: PASS

The JAX path agrees with the numpy reference at machine precision
(worst cases 3.183e-15 single-atom, 8.359e-16 composed, against a 1e-7
tolerance), reproduced by the reviewer, including at the
validity-guard boundary itself, a field the builder's own grid never
reached. A function-by-function read found no silent divergence.
Gradients agree with central finite differences of the numpy reference
to 1.748e-6 worst case against a 1e-5 tolerance. The reviewer's
independent step-size scan confirmed the tolerance-clearing conclusion
is stable across neighboring steps; one docstring sentence
overclaiming stability at a tighter step was corrected in the fix loop
with the measured values. NaN sweeps are clean, including two
adversarial probes the fix loop then encoded as regression tests: the
field at the guard boundary and the patch amplitude at zero.

## Determinism and memory: PASS

Bitwise-deterministic in-process and across fresh subprocesses.
Measured peak RSS 0.40 GB for the heaviest value-and-grad call,
guarded at 1.5 GB (macOS) and 2.0 GB (linux) with the measurement
recorded beside the bound.

## The fit demonstrator: PASS, after one fix loop

Eight synthetic cases (planted truths, seeded noise, varied starting
points) fit by L-BFGS-B through the differentiable chain: 8 of 8
converge with positive-definite Hessians. Coverage as observed: 1 of 8
cases recovers all three parameters within their reported 1-sigma
Laplace uncertainties and 6 of 8 within 2-sigma. The gate's FATAL was
a wording claim of "correctly calibrated Laplace uncertainties" in the
benchmark's claim string and a tuning docstring, which the observed
coverage contradicts; the fix loop reworded both to report coverage as
data, wired to the same variables the artifact computes, and the
regenerated artifact is numerically unchanged. The near-bound
exception case was independently verified as a true stationary point
(recovered value 4.0 from the bound, unconstrained gradient ~1.2e-5).
The indefinite-Hessian discipline (flag false, NaN sigmas) fires
correctly on a reviewer-constructed Hessian not in the test file. The
demonstrator is labeled synthetic throughout, with no published-data
claims and the validity-window guard enforced inside the fit loop.

## The field model: PASS

The three-parameter cell field (uniform background, axial gradient,
one softened wall patch) is labeled as this module's own construction
throughout, with the softening documented as a differentiability
choice and its relation to the Phase A patch model stated.

## Notebook and prose: PASS, after one fix loop

Notebook 17 carries full narrative from the first commit. The gate
found nine banned constructions across five cells; the fix loop
rewrote each per the project standard, including a positive rewrite of
the closing scope passage that preserves its no-published-data and
no-priority-claim content. All code-cell outputs were verified
byte-identical through the rewrite. Every number in prose matches an
executed output or artifact.

## Citations and hygiene: PASS

The diff introduces no new citations; all three sources are reused
from the gated Phase A work. No priority claims. The
two-reproducibility-case phrasing is untouched. CONVENTIONS section
numbering is flagged provisional pending the sibling Stark-map
branch's allocation, for the coordinator to reconcile at merge.

## Battery: PASS

The full eight-check battery is green on the branch, including
notebook re-execution and both pytest lanes; 74 tests cover the two
Rydberg modules.

## Verdict: PASS
