# G19 gate: differentiable JAX BO+WKB implementation (WP37)

Independent review of `cliffordclock.integrator.lattice_light_shift_jax`
(E41 addendum, CONVENTIONS.md 1.13.0) and
`tests/test_lattice_light_shift_jax.py`. The reviewer independently
recomputed the AGREEMENT and GRADIENT contracts against the G18-gated
reference implementation, reproduced the module's own reported
turning-radius `NaN`-gradient bug at its claimed grid resolutions,
confirmed the fix converges to the identical physical root, swept the
module's other clamp and boundary-selection sites for the same hazard,
and hand-verified the `tangent_solve` implicit-function-theorem formula
and the bisection's floating-point precision.

## Agreement and gradients: PASS, independently recomputed

`axial_thermal_factors_jax`'s `X`/`Y`/`Z`, evaluated at all four of
Bothwell et al. 2025's Appendix A Table I points (`u0` = 56.8/66.4/86.2/
112.2 `E_R`, Yb-171, `n_z=0`), against `lattice_light_shift.axial_thermal_factors`'s
own converged output:

| u0 (E_R) | Tr (nK) | X rel err | Y rel err | Z rel err |
|---|---|---|---|---|
| 56.8 | 650 | 9.97e-9 | 1.05e-7 | 1.75e-8 |
| 66.4 | 550 | 9.33e-9 | 1.17e-7 | 1.68e-8 |
| 86.2 | 600 | 9.03e-9 | 1.36e-7 | 1.65e-8 |
| 112.2 | 720 | 8.91e-9 | 1.57e-7 | 1.65e-8 |

Worst case `1.57e-7` (`Y`, `u0=112.2 E_R`), two orders of magnitude
inside the module's own `1e-6` AGREEMENT contract. The reviewer's
independent recomputation confirms the correct attribution directly: `Y`
grows monotonically with `u0` across all four rows, so the worst case
sits at the DEEPEST point, `u0=112.2`, not the shallowest
(`u0=56.8`'s own `Y` error, `1.05e-7`, is in fact the SMALLEST of the
four). An earlier draft of this module's own docstring, the
`TestAgreementWithReference` docstring, and CONVENTIONS.md's own §17
addendum all attributed the worst case to `u0=56.8`; all three are now
corrected to `u0=112.2`.

`jax.grad` of the light shift with respect to `u0` and `Tr`, checked
against central finite differences of the REFERENCE implementation
(the strongest available check: independent numerical methods on each
side of the comparison), at the same four points:

| u0 (E_R) | Tr (nK) | grad_u0 rel err | grad_Tr rel err |
|---|---|---|---|
| 56.8 | 650 | 3.88e-8 | 4.93e-8 |
| 66.4 | 550 | 3.18e-8 | 4.50e-8 |
| 86.2 | 600 | 2.16e-8 | 3.45e-8 |
| 112.2 | 720 | 1.34e-8 | 2.54e-8 |

Worst case `4.93e-8` relative, four orders of magnitude inside the
`1e-4` GRADIENT contract. `X`/`Y`/`Z`'s own values and both gradients
were recomputed from the module's public functions directly (not read
back from the module's own test-suite cache), matching the numbers
above to the reported precision.

## The turning-radius clamp bug: reproduced, and the fix confirmed correct

The module's own docstring reports an intermittent `NaN` gradient from
differentiating the turning-radius root-find (`turning_radius_m_jax`,
Eq. 10's `Rnz(E)`) through the reference module's own "clamp an unbound
state's energy to `0.0`" convention. The reviewer reproduced this
directly: at `axial_grid_n in {21, 81, 161}`, `jax.grad` of `rho_max`
with respect to `waist_m` returned `NaN`; at `axial_grid_n in {41, 321,
1281}` (including the module's own production resolution) it returned a
finite value. This resolution-dependence is exactly what the mechanism
predicts: the clamp makes the axial energy flat (`0.0`) across an entire
ray of radii beyond the true band edge, so `tangent_solve`'s `y /
g(1.0)` divides by zero whenever the fixed-iteration bisection's final
floating-point step lands on that flat side, a coin flip that depends on
where the discretized crossing point falls relative to the bisection's
own halving sequence at each grid resolution.

The fix, root-finding against the RAW (un-clamped) eigenvalue inside
`turning_radius_m_jax`'s own `f` only
(`_axial_band_energy_er_at_rho_unclamped`), was checked for correctness
two ways. First, the fixed-point itself: at every resolution the fix
made finite (`41`, `321`, `1281`), the returned `rho_max` is IDENTICAL
before and after the fix to the reported floating-point precision (the
clamped and unclamped eigenvalues agree everywhere strictly inside the
band, so the root itself cannot move; only the derivative's well-
posedness at the root changes). Second, the gradient's own value: at
`AXIAL_GRID_N_JAX=1281`, `d(rho_max)/d(waist_m)` matches the analytic
scaling `rho_max/waist_m` (exact for this problem's `kappa ~ 1/waist_m`
factorization) to 15 significant digits (`1.3720281773666503` computed
vs. `1.37202817736665` analytic).

**No sibling flat-region hazard, swept across all six of the module's
clamp/boundary-selection sites.** The reviewer enumerated every
`jnp.minimum`/`jnp.maximum`/`jnp.where` in the module and the one
boolean band-membership condition alongside them: (1) `axial_energies_er_jax`'s
`jnp.minimum(energies[:n_states], 0.0)`, the clamp definition itself;
(2) `harmonic_density_of_states_closed_form_jax`'s `jnp.maximum(bracket,
0.0)`, Eq. 4's own closed-form clamp; (3)-(4) the two `jnp.where` calls
inside `turning_radius_m_jax`'s `solve` bisection body (`lo`/`hi`
updates); (5) `bo_wkb_density_of_states_jax`'s final `jnp.where(in_band,
value, 0.0)`; (6) that same function's `in_band` boolean
(`(energy_er >= e0) & (energy_er <= 0.0)`). Sites (1) and (2) are never
root-find targets: the reference-matching clamped value from (1) feeds
only direct evaluations (band-bottom checks, shape-factor samples) and
(2) is a standalone algebraic evaluator, so neither can produce the
"flat region under an active root-find" configuration the bug needs.
Sites (3)-(4) sit inside `solve`, which `jax.lax.custom_root` restricts
to the forward pass by construction: the backward pass runs only
`tangent_solve`, so `solve`'s own comparisons stay outside gradient
computation regardless of any flatness in the function they bisect.
Sites (5)-(6) wrap the ALREADY-FIXED
`turning_radius_m_jax` call's output; they select between a finite
value and a hard `0.0` based on a boundary condition evaluated outside
the root-find itself, the same "select, don't feed a root-find" pattern
(1) and (2) have. No sibling hazard exists.

## Hand implicit-function-theorem check

`tangent_solve(g, y) = y / g(jnp.ones(()))` was checked directly against
the standard 1D implicit-function-theorem derivation. At a root `rho*`
of `f(rho, theta) = 0`, `d(rho*)/d(theta) = -(df/dtheta) / (df/drho)`,
evaluated at `(rho*, theta)`. `jax.lax.custom_root` supplies `g` as `f`
linearized at the root with respect to its FIRST argument only (`rho`),
so `g(1.0) = df/drho` at the root; the theta-dependence is handled
separately, by `custom_root`'s own outer VJP machinery composing this
scalar linear solve with `df/dtheta`. `tangent_solve`'s `y / g(1.0)`
therefore solves the correct scalar linear equation
(`(df/drho)*dx = y`) for `dx`, matching the formula's denominator
exactly. This is JAX's own documented idiom for a scalar root
(`lambda g, y: y / g(1.0)`, from `jax.lax.custom_root`'s docstring), used
here unmodified.

## Bisection precision

`BISECTION_ITERS = 60` halves an outward bracket of
`DEFAULT_RHO_BRACKET_WAIST_MULTIPLE * waist_m` (`10 * waist_m`, tens of
microns for realistic lattice waists). `60` halvings resolve that
bracket to `bracket / 2^60`, roughly `5e-25` m for a `50` micron waist:
far below the smallest representable relative step at the root's own
scale (float64 carries 52 mantissa bits, so ~53 halvings already exhaust
the mantissa's own resolution at any fixed exponent). `60` iterations therefore drive the bisection to the double-precision-
representable value closest to the true root. The seven extra
iterations beyond the mantissa's own limit cost a negligible fraction
of the eigensolve-dominated per-call runtime.

## Dtype discipline

`TestFloat64Discipline::test_shift_and_factors_are_float64` confirms
`bo_wkb_fractional_light_shift_jax`'s shift and `ThermalShapeFactorsJax`'s
`x_nz`/`y_nz`/`z_nz` are all `jnp.float64`, inherited from
`cliffordclock`'s own package-level `jax_enable_x64` configuration
(`cliffordclock/__init__.py`, which must run before any `jax.numpy` array
is built anywhere in this package). The module's own dense `N=1281`
eigenproblem subtracts `O(10-100) E_R` numbers to resolve an `O(1e-5)
E_R` convergence tolerance; float32's ~7 decimal digits cannot represent
that subtraction meaningfully, the same reasoning
`cliffordclock.integrator.worldline`'s own docstring states for its
float64 requirement.

## Verdict: PASS after one fix loop

The physics, the numerics, and the differentiability machinery were
correct on first submission; this gate FAILED only on four docstring
items, all in `wt-jax`. The fix loop:

1. `tests/test_lattice_light_shift_jax.py`,
   `test_grad_wrt_waist_is_finite_and_near_zero`'s docstring carried a
   "rather than" construction split across a line wrap ("checked below
   (`mass_kg`) rather than against a fixed absolute bound"); reworded to
   state the comparison the test actually performs directly.
2. `TestAgreementWithReference`'s docstring misattributed the worst-case
   `X`/`Y`/`Z` error to `Y` at `u0=56.8` (`1.05e-7`); the true worst case
   is `Y` at `u0=112.2` (`1.57e-7`), confirmed by the reviewer's own
   independent recomputation above. Corrected in the test docstring, the
   module docstring, and CONVENTIONS.md's §17 addendum (three sites).
3. Three trailing negated tails in the test file (", not `jax.Array`s
   tied to a specific trace)", ", not `pytest.approx`)", ", not a
   determinism bug") rewritten per the project's prose-review skill:
   each now states the positive fact directly, and gives the ruled-out
   alternative's own reason its own clause or sentence, in place of a
   bare negation tacked onto the end.
4. CONVENTIONS.md's dense NaN-bug changelog sentence split into three:
   symptom and mechanism, the flat-region location and division-by-zero
   consequence, and the fix with its same-root guarantee.

A whitespace-normalized sweep of all three changed files (collapsing
line wraps before pattern-matching, so a banned phrase split across a
wrap cannot hide from a naive line-by-line grep) confirms zero remaining
instances of `rather than`, `instead of`, or a trailing `, not `/`,
never ` tail in the reviewer's own added prose. `ruff check .`, `ruff
format --check .`, `mypy src/` (29 files), the fast-lane test suite (10
tests, 2.8 s), and `tools/release_checks.py --fast` all PASS with zero
new findings attributable to this diff: `prose-scan` finds one residual
MINOR sentence-length note at the new changelog entry's boundary with
the pre-existing `1.12.0` entry, a scanner tokenization artifact (the
scanner's sentence-boundary regex does not recognize `.*` followed by a
newline and another `*` as a split point) already present, in far more
severe form (a 167-word merged span), at the document's own pre-existing
`1.12.0`-to-`1.11.0` boundary; not a defect in this entry's own prose,
and not touched further since fixing it would mean editing unrelated,
already-gated changelog text.

## Addendum (2026-08-29): PR #19 OOM, confirmed and fixed

PR #19's slow lane died twice on the CI runner with no failed step
recorded, the external signature of an OOM kill. The arithmetic pointed
at this module's radial evaluation: `axial_thermal_factors_jax` vmapped
`RHO_GRID_N_JAX=321` dense `(1281, 1281)` float64 eigenproblems, WITH
their gradient graphs, in one materializing batch.

**Confirmed first, before any change to the gated code.** A
fresh-subprocess `resource.getrusage(RUSAGE_SELF).ru_maxrss` reading
(the same methodology `tests/test_e2e.py`'s own RSS guard uses) for one
production-resolution `value_and_grad` call at the `u0=56.8` table
point measured `31,483,969,536` bytes, `~29.3 GiB` (`~31.5 GB` decimal)
peak RSS on the development macOS machine. This is roughly `8x` the
coordinator's own rough `4+ GB` estimate and confirms the hypothesis
with a wide margin: the batched Hamiltonians, eigenvectors, and `eigh`'s
own backward-pass residuals for all `321` matrices lived at once.

**The fix: a memory-bounded schedule, no change to the gated numerics.**
`jax.lax.map(sample, rhos, batch_size=RHO_MAP_BATCH_SIZE)` replaces the
single `jax.vmap` call, `RHO_MAP_BATCH_SIZE=16`. Chunking alone measured
`~13-15 GB` across `batch_size in {1, 4, 16}`, a real reduction but far
short of a safe runner budget: `jax.lax.map` compiles to a `scan`, and
reverse-mode autodiff through a `scan` still saves every chunk's own
residuals for its backward pass. Wrapping the per-`rho` `sample` closure
in `jax.checkpoint` closes the remaining gap by discarding those
residuals after the forward pass and recomputing them fresh during the
backward pass; combined with `batch_size=16`, the SAME call at the same
table point measured `2,230,992,896` bytes, `~2.08 GiB` (`~2.23 GB`
decimal) peak RSS, a `14.1x` reduction. `RHO_MAP_BATCH_SIZE`'s own
docstring in the module carries this full measurement.

**Re-verified contracts, at production resolution, after the schedule
change.**

| u0 (E_R) | X rel err | Y rel err | Z rel err | grad_u0 rel err | grad_Tr rel err |
|---|---|---|---|---|---|
| 56.8 | 9.97e-9 | 1.05e-7 | 1.75e-8 | 3.88e-8 | 4.93e-8 |
| 66.4 | 9.33e-9 | 1.17e-7 | 1.68e-8 | 3.18e-8 | 4.50e-8 |
| 86.2 | 9.03e-9 | 1.36e-7 | 1.65e-8 | 2.16e-8 | 3.45e-8 |
| 112.2 | 8.91e-9 | 1.57e-7 | 1.65e-8 | 1.34e-8 | 2.54e-8 |

Identical to the gate's own table above at the reported precision: worst
case `X`/`Y`/`Z` agreement stays `1.57e-7` (`Y`, `u0=112.2`), worst case
gradient agreement stays `4.93e-8` (`Tr`, `u0=56.8`), both unchanged from
the pre-fix numbers this record already carries. Comparing the raw
values directly (the `56.8` point's `X`, computed both before and after
the schedule change) shows the summation-order effect the schedule
change can introduce: `0.7854876724879748` before, `0.7854876724879742`
after, a `~7.6e-16` relative shift, a few floating-point ULPs. Every
comparable value across all four points shifts by `1e-16`-to-`1e-14`
relative, many orders of magnitude inside both the `1e-6` AGREEMENT and
`1e-4` GRADIENT gate tolerances. `jax.jit` determinism (two calls to the
same compiled executable, bitwise-identical) and the offline
convergence study (all five of its own tests) were re-run at production
resolution and PASS unchanged.

**Timing impact.** Forward-only evaluations (the offline convergence
study, the density-of-states tests, `turning_radius_m_jax` calls outside
a gradient) are unaffected, since `jax.checkpoint` only adds
recomputation cost when a backward pass actually runs. `value_and_grad`
calls measure roughly `2x` the pre-fix wall time (`~60 s` to `~106-115
s` for one production-resolution call), the expected memory-for-
recompute trade `jax.checkpoint` makes.

**New guard.** `TestMemoryBound::test_production_call_stays_under_the_memory_bound`
(`tests/test_lattice_light_shift_jax.py`, slow-marked) measures one
production `value_and_grad` call's peak RSS in a fresh child subprocess
and asserts it under a platform-aware bound (`3.5 GB` macOS, `4.0 GB`
linux), comfortably above the `~2.2 GB` measured floor and comfortably
below the pre-fix `~31 GB` blowup, so a future regression toward
materializing batching fails this test before it reaches a CI runner's
own memory limit.

Verdict: PASS. The OOM was a genuine runner-facing defect in this
module's own scheduling, found, confirmed by direct measurement before
any code change, and fixed without altering a single gated number beyond
last-ULP floating-point noise.
