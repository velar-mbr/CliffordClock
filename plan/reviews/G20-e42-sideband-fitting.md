# G20 gate: sideband-spectrum forward model and gradient-based fitting (WP38)

Independent review of `cliffordclock.integrator.sideband_spectrum_jax`
(E42, CONVENTIONS.md §18), `benchmarks/run_sideband_spectrum.py`,
`benchmarks/run_sideband_fit.py`, and `tests/test_sideband_spectrum_jax.py`.
The reviewer transcribed every cited equation against both source PDFs
directly, cloned `large-lattice-model` at its pinned commit into a
separate environment and re-derived the fixture's own reference values
bit-exactly, reproduced the 12-case fitting grid, and hand-checked the
Hessian this gate's own blocker turns on.

## Transcription fidelity: PASS

Every equation §18 cites was checked against the typeset source PDF
directly: Blatt, Thomsen, Campbell, Ludlow, Swallows, Martin, Boyd, Ye,
PRA 80, 052703 (2009), arXiv:0906.1419 (all 12 pages, the harmonic path,
Eqs. 3-5, 8, 13-20, and Appendix A Eqs. A1-A2), and Goti, Petrucciani,
Condio, Levi, Calonico, Pizzocaro, arXiv:2508.08164 v2 (the first 8
pages, the BO+WKB path, Eqs. 1-2, 4-5, 8-9). Both papers' shared
harmonic motional spectrum (Blatt Eq. 3, Goti Eq. 1) match character by
character, as do the trap-frequency definitions (Blatt Eqs. 4-5) and the
longitudinal energy gap (Blatt Eq. 8, Goti Eq. 2). The BO+WKB density of
states (Goti Eq. 8) is algebraically identical to
`lattice_light_shift_jax.bo_wkb_density_of_states_jax`'s own Beloy et
al. 2020 Eq. 11, an equivalence CONVENTIONS.md §18 states directly and
the reviewer confirmed by symbolic expansion of both forms. The red
sideband's own zeroed ground-state population weight matches the
condition Blatt et al. 2009's text following their Eq. 12 states
("there is no contribution from the longitudinal ground state to the
red sideband").

## The Laguerre catch: found by the build's own tests, fixed at machine precision

`laguerre_values` evaluates the physicists' Laguerre polynomials Blatt
Eq. 14's carrier Rabi frequency needs by the standard three-term
recurrence, since `jax` ships no generalized Laguerre function. An
earlier draft's own recurrence coefficients were transposed; the
project's own `TestLaguerreValues` (checking every value against
`scipy.special.eval_genlaguerre(n, 0, x)`) caught the mismatch before
this gate's own review began, and the recurrence was corrected to the
standard form, `(n+1)*L_{n+1}(x) = (2n+1-x)*L_n(x) - n*L_{n-1}(x)`. The
reviewer independently re-ran the comparison at five points
(`x = 0.0, 0.3, 1.7, 5.0, 12.3`, orders `n = 0` through `10`): worst
relative error `3.83e-15` against `scipy`, a few floating-point-rounding
ULPs, machine precision. This is the project's established
discipline working as designed: a transcription slip caught by the
build's own test suite before ever reaching an external reviewer.

## Three-tier independent-implementation validation: MET, reviewer re-derivation bit-exact

`benchmarks/run_sideband_spectrum.py` compares against
`large-lattice-model` (github.com/inrim/large-lattice-model, MIT,
INRIM), a real third-party implementation that solves the same
Born-Oppenheimer+WKB eigenproblem with exact Mathieu-function
characteristic values, a numerical method independent of this project's
own finite-difference solver. The reviewer cloned that repository at
the fixture's own pinned commit
(`f569907cdf2f08a9081386139211a205b3c42624`) into a dedicated
environment, ran its own `U`, `R`, `DeltaU`, and `sidebands` functions
at the fixture's own recorded inputs, and matched
`benchmarks/fixtures/wp38_inrim_large_lattice_model_reference.json`'s
own committed numbers bit-exactly. This is a new evidentiary class this
gate confirms works as intended: `independent_implementation_reproduction`
reproduces an independent CODE implementation's own output. This
project's established `arithmetic_reproduction` class reproduces a
paper's own published table. The reviewer's own bit-exact
re-derivation is the strongest form the newer class can take, since it
rules out a transcription error anywhere between the fixture and this
project's own consuming code.

Three tiers, tightest to loosest, all reproduced on re-running this
project's own benchmark against that fixture:

1. **Band-bottom eigenvalue** (`independent_implementation_reproduction`):
   worst relative error `1.06e-7` against a `1e-4` tolerance. **PASS.**
2. **Franck-Condon detuning** (`independent_implementation_reproduction`):
   worst relative error `4.73e-3` against a `2e-2` tolerance, excluding
   4 of 36 points within `5 E_R` of the band top (the module's own
   documented resolution limit, each such point carrying its own
   `near_band_top` flag). **PASS.**
3. **Full sideband shape** (`computable_comparison`, bridged across the
   documented Lorentzian peak-height, linewidth, and integration-domain
   convention differences): worst peak-position difference `500 Hz` on
   a `~33-35 kHz` sideband, minimum shape correlation `0.9339`. **MET.**

## The 12-case fitting grid: reproduced bit-for-bit

`benchmarks/run_sideband_fit.py`'s synthetic round-trip (harmonic and
BO+WKB paths, three truth `(u0, Tr)` pairs, two noise seeds) was
re-run twice independently. Every recovered parameter matches to full
float64 precision between runs (for example, the harmonic `u0=100`,
`seed=0` case's own recovered `u0`, `100.95529256359293`, identical to
the last digit both times); only the artifact's own `generated_at_utc`
timestamp differs. `numpy.random.default_rng(seed)`'s fixed integer
seeds and this project's own static-shape jit discipline together give
this reproducibility, the same guarantee every other benchmark in this
project already relies on.

## Gradients and jit: clean

`TestBowkbGradients` (both `jax.grad(..., wrt=u0)` and `wrt=Tr`) matches
central finite differences of the same function; `TestJitDeterminism`
confirms repeated jit-compiled calls return bitwise-identical output.
Both were re-run by the reviewer and pass without modification.

## THE BLOCKER: a saddle point silently reported as the most confident row in the table

`run_sideband_fit.py`'s harmonic `u0=100`, `seed=0` case is a saddle
point of the negative log-likelihood. The reviewer computed the Hessian
at the reported optimum directly:

    [[   81.08,     -4.8656e7],
     [-4.8656e7,     2.6364e13]]

with eigenvalues `[-8.72, 2.636e13]`, one negative: an INDEFINITE
Hessian, the signature of a saddle point. `scipy.optimize.minimize`'s
own `L-BFGS-B` reported `success=True` regardless, since its stopping
test checks gradient norm alone and says nothing about the Hessian's
own curvature. The pre-fix code inverted that Hessian anyway; its raw
covariance diagonal came out negative (`cov[0,0] = -0.1147`,
`cov[1,1] = -3.53e-13`), and `sigma = sqrt(max(cov[i,i], 0.0))` clamped
both invalid variances to zero. The artifact printed
`100.96 +/- 0.00` and `1.959 +/- 0.00`: the row with the least
trustworthy uncertainty in the entire table rendered as the row with
the most confidence in it, indistinguishable at a glance from a
textbook-perfect fit.

This is an artifact-integrity finding. The forward model, the
gradients, and the optimizer's own recovered central values were never
wrong; what failed is the step between a number and its own claimed
precision. A reader scanning a results table trusts a small `+/-`
figure most where that trust is least earned; a zero can mean "measured
tightly," and it can mean "this number was never checkable." The two
must render differently.

**The fix**, applied and independently re-verified by the reviewer:

1. `run_one_fit`'s Hessian-to-uncertainty step is now a standalone
   function, `laplace_uncertainties(hessian)`, computing
   `np.linalg.eigvalsh(hessian)` before ever inverting. A new
   `hessian_positive_definite` field on `FitCase` is `True` only when
   every eigenvalue is strictly positive; when it is `False` (a saddle
   point, matching the reviewer's own eigenvalues above) or the
   inversion itself raises `np.linalg.LinAlgError` (a singular
   Hessian), both reported uncertainties are `nan`. A clamped zero is
   removed from the code path entirely.
2. Both artifacts now surface the flag. The JSON carries
   `hessian_positive_definite` on every case plus a report-level
   `n_hessian_positive_definite` count (`11/12`). The Markdown table
   gains a `Hessian PD` column, the flagged row's own uncertainty cells
   render `+/- nan` in place of a numeric value, and a dedicated
   paragraph beneath the table names the row by its own truth values
   and states the Laplace approximation is invalid there.
3. `TestRoundTripFitConvergence` now calls the same
   `laplace_uncertainties` function (imported from the benchmark script,
   the pattern `tests/test_roos_benchmark.py` already establishes) in
   place of its own separate, unclamped `sqrt(cov[i,i])`, and asserts
   `hessian_positive_definite` directly. A future regression toward a
   saddle point now fails that assertion loudly, with the offending
   eigenvalues printed in the failure message. A new
   `TestLaplaceUncertaintyReportingPath` class feeds
   `laplace_uncertainties` three fabricated Hessians with no fit, no
   forward model, and no jax involved: the reviewer's own indefinite
   matrix above (flag flips `False`, both sigmas `nan`), a
   positive-definite control case (flag `True`, sigmas finite and
   correct to the fourth decimal), and a singular case (flag `False`,
   sigmas `nan`). All three pass; the planted violation is caught.
4. CONVENTIONS.md §18's own sentence describing this row no longer says
   "local-optimum/ill-conditioned-Hessian," language that understates
   what happened. It now states directly that `L-BFGS-B` stops at a
   saddle point there, gives the Hessian's own eigenvalues, states that
   the Laplace approximation requires a positive-definite Hessian and
   is therefore invalid at that optimum, and names the
   `hessian_positive_definite` flag as what catches it.

Re-running `run_sideband_fit.py` after the fix reproduces the same 11
unaffected rows' central values and uncertainties to full precision (the
same bit-for-bit reproducibility the section above confirms) and renders
the twelfth row as:

    | harmonic | 100.0 | 1.00 | 0 | 100.96 +/- nan | 1.959 +/- nan | False | True | False | False |

## MINOR 1: the fixture's own non-default argument, now recorded

`benchmarks/fixtures/wp38_inrim_large_lattice_model_reference.json`'s
`sideband_spectra` rows were generated by `large-lattice-model`'s own
`sidebands()` at `fac=20`. That function's own default is `fac=10`
("controlling the number of lorentzian functions used to calculate the
sideband shape," per that function's own docstring); the fixture's
`fac=20` was discoverable only by sweeping the function's own arguments
against the fixture's own values. The fixture's `provenance` field now
carries a `non_default_arguments` entry naming the function, the value
used, the library's own default, which rows it applies to, and why; the
regeneration recipe in `run_sideband_spectrum.py`'s own module
docstring states the same fact directly, so a reader following that
recipe reproduces this exact fixture.

## MINOR 2: the em-dash substitute

`run_sideband_spectrum.py`'s module docstring carried a plain-ASCII
double-hyphen em-dash substitute (`" -- "`) around line 87, alongside a
trailing negated tail (", not this project's own") in the same
sentence. Both are now split into separate declarative sentences,
consistent with this project's prose-review discipline.

## Verification after the fix loop

`ruff check .` and `ruff format --check .` both PASS with zero findings.
`mypy src/` PASSES (30 source files) at the project's own 3.12 target.
The full `tests/test_sideband_spectrum_jax.py` file (25 tests, run
foreground in two chunks, fast then `slow`) PASSES, including the three
new planted-violation tests and the reconciled round-trip test.
`tools/release_checks.py --fast` PASSES; `prose-scan`'s own 698 findings
are unchanged from before this fix loop, and every one of them is a
pre-existing sentence-length/table-scanner artifact in `paper/main.tex`
or in the auto-generated benchmark Markdown tables, where the scanner
reads a long data row as one long sentence. This fix loop's own
hand-written prose additions carry zero such findings.
A whitespace-normalized sweep (collapsing line wraps before matching, so
a banned phrase split across a wrap cannot hide) of every string changed
in this fix loop found two more instances of the same trailing-negation
pattern this gate's own minor 2 named (in a docstring and a test
docstring added during the fix itself); both are corrected in place, and
the sweep is now clean across every file this loop touched.
`benchmarks/run_sideband_spectrum.py` and `benchmarks/run_sideband_fit.py`
were both re-run to regenerate `benchmarks/results/wp38_sideband_spectrum.{json,md}`
and `benchmarks/results/wp38_sideband_fit.{json,md}`.

## Verdict: PASS after one fix loop

The physics, the transcription, the independent-implementation
cross-validation, and the fitting demonstration's own central values
were correct on first submission. This gate FAILED on one blocker: a
saddle point's invalid Laplace uncertainty rendered as a clamped zero,
the most confident-looking row in a table whose entire purpose is
reporting calibrated uncertainty. The fix closes it at its root: an
eigenvalue check runs before any inversion is trusted, and the result
on failure is `nan`. It surfaces everywhere a reader would look (JSON,
Markdown table, Markdown narrative, CONVENTIONS.md), and a new test
plants the same failure mode this gate found and checks the fix catches
it. Both minors (the fixture's undocumented non-default argument, the
em-dash substitute) are fixed. All checks PASS; the fitting grid and
the three-tier cross-validation reproduce bit-for-bit against the
reviewer's own independent re-derivation.
