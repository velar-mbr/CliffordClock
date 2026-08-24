# Timescales: the three-tier fast-path architecture

## The practical answer

- **Real interrogation times (microseconds to seconds) are the norm.** 
  The physics integrand this tool accumulates contains no
  fast carrier: it varies only on the atom's own motional timescale
  (trap dynamics, microseconds to milliseconds), so real interrogation
  windows are cheap to compute exactly or near-exactly. There is no
  femtosecond ceiling to work around.
- **Which mode to use:**
  - `ensemble.regime: lattice` (optical-lattice clocks): use the default
    `fast_path`. Set `integration.time_s` to your real interrogation
    time; cost is independent of `T`.
  - `ensemble.regime: classical` (ion-trap-style, general motion): use
    the default `direct`; omit `integration.dtau` and let the tool pick
    a trap-resolved step size automatically.
  - Classical + purely periodic motion in an isotropic trap: `secular`
    gives the same time-independent cost as the lattice fast path.
  - `worldline`/Compton-scale `direct` (`dτ̃ ~ 1`): a validation/
    cross-check mode only, never required for a physical result, but
    always available to double-check one.
- **Real limits.** `select_dtau`'s automatic step size is accurate to
  ~3e-9 relative on the closed-form test case backing it (measured, not
  assumed: see the accuracy study below); the rotor-diagnostic fields
  (not the primary reported shift) can drift at very large step sizes if
  a run also uses very strong coupling, which the pipeline guards against
  automatically (see "Safety net" below). If you just want a number for a
  real experiment, the defaults above are the right call.

## Why real interrogation times are tractable

Early runs of this tool stepped the rotor at Compton-scale `dτ̃ ≈ 1`,
which limited tractable runs to attosecond-or-shorter simulated
interrogation windows: physically meaningless to a clock user (real
interrogation times are microseconds to seconds). **That was an artifact
of the step-size choice, not a requirement of the physics.** The
observable integrand `δω̃(r(t), v)` (CONVENTIONS.md E21) contains **no
Compton-frequency content**: the fast `~10²⁰ rad/s` carrier is removed
analytically by the perturbation formulation (E10), so `δω̃` varies only
on the atom's own motional timescale: trap dynamics, microseconds to
milliseconds for realistic ion/lattice traps. This document explains the
three-tier architecture (CONVENTIONS.md section 12, E29-E31) that
exploits this, the accuracy study behind it, and when to use each tier.

## The three tiers

| Tier | Regime | Mode | Cost in `T` | What it computes |
|---|---|---|---|---|
| **A** | lattice (quantum motional state) | `fast_path` (default) | O(1): no time stepping | `ΔΦ_q = δω̃_q · T̃` per node (E29), exact |
| **B(i)** | classical | `direct` (default) | O(steps), `dτ̃` auto-selected | The rotor integrator (E17-E24), stepped at auto-selected trap-period resolution (E31) |
| **B(ii)** | classical, periodic (isotropic trap) | `secular` | O(1) sub-steps per orbit, independent of `T` | `ΔΦ = ⟨δω̃⟩_orb · T̃` (E30), one-orbit average × `T` |
| **C** | either | `worldline` (lattice) / Compton-fine `direct` (classical) | O(T̃) | The full rotor integrator at Compton-scale `dτ̃ ~ 1`, the original Compton-scale validation mode |

Tier A is the default for `ensemble.regime: lattice`; Tier B(i) is the
default for `ensemble.regime: classical`. Tier C remains fully available
(and is still exercised by the full test suite) as a cross-check and
validation mode; it is what Tier A/B are validated *against* below, not
a deprecated path.

## Tier A: lattice fast path (E29)

For a **static** motional state (Hermite-Gauss quadrature nodes) in
a **time-independent** field, every node has `v = 0` and a *constant*
`δω̃_q = δω̃(r_q, 0)` (E21). E22's time integral of a constant is trivial:

```
ΔΦ_q = δω̃_q · T̃          (E29)
⟨Δν/ν₀⟩ = Σ_q w_q δω̃_q    (E23, unchanged)
```

No time integration at all; the cost is independent of `T`. Per-node
values are still kept, so T2*/coherence/line-profile analytics (E25-E28)
work exactly as before.

**Config:** `ensemble.regime: lattice` + `integration.time_s: <seconds>`
(direct input; `integration.dtau`/`steps` also still accepted for
backward compatibility, converted to seconds via `steps * dtau * τ_c`).
`integration.mode: worldline` selects the Tier-C rotor integrator on the
same static nodes instead, as an explicit cross-check.

**Validated ("Tier A ≡ Tier C"):**
`tests/test_fastpath_lattice.py::test_tier_a_equals_tier_c_worldline_on_static_nodes`
matches the fast path against `integrate_ensemble` (Tier C) on identical
static nodes to **rtol 1e-12, atol=0**, an exact algebraic identity, not
an approximation. A second check confirms the equality extends to *any*
`T` (Tier C's own phase is provably linear in step count for a static
node, matching the fast path's linearity by construction). Also
demonstrated via the pipeline directly in
`tests/test_e2e.py::test_wp8_lattice_fast_path_is_default_and_worldline_is_explicit_crosscheck`.

**The 1-second demo.** `examples/lattice_sr87.yaml`
now specifies a genuine `integration.time_s: 1.0`, a real, second-scale
interrogation. Measured (`tests/test_e2e.py::test_wp8_lattice_one_second_demo_sem_much_less_than_shift`):

| Quantity | Value |
|---|---|
| Wall time (in-process) | ~0.4 s |
| Wall time (CLI subprocess, incl. Python startup) | ~1.1 s |
| Interrogation time | exactly 1.000000 s |
| Ensemble size | 64 quadrature nodes |
| Mean fractional shift | `+1.218093e-10` |
| SEM | `7.597e-14` |
| SEM / \|shift\| | `~6.2e-4` (SEM ≪ shift) |

Both well under the 60 s CPU bound, and the SEM margin is achieved
plainly: the Gaussian-bump field's width (1 µm) is chosen to be much
wider than the Sr-87 ground-motional-state extent at this trap frequency
(~61 nm), so the field is nearly uniform across the ensemble's nodes,
not by loosening a tolerance.

## Tier B(i): large-`dτ̃` direct integration + `select_dtau` (E31)

The rotor integrator itself is unchanged (E17-E24); what changes is
the step size. E31's rule: resolve the *trap* period, not the Compton
period.

```
select_dtau(trap, points_per_period=100) -> dτ̃ = T_orb / (points_per_period · τ_c)
```

with `T_orb = 2π / min(trap.omega_xyz)` (the slowest/longest-period trap
axis, so every axis is resolved at least this finely).
`integration.mode: direct` (the classical default) uses this
automatically whenever `integration.dtau` is omitted; `integration.time_s`
then determines `steps`. An explicit `dtau`/`steps` pair is still honored
unchanged (backward compatible with every earlier config).

### Safety net: the per-step rotor generator angle bound

`select_dtau`'s rule above resolves *trap* dynamics only; it says
nothing about the coupling strength (`coupling.mu`, E14a) or field
magnitude, both of which set how large the per-step rotor generator
(`-½ dτ̃ Ω`, E19) is. A large enough generator angle can push
`exp_bivector`'s fixed-order Taylor evaluation into badly wrong (or
non-finite) rotor-diagnostic output, even though the primary scalar phase
(E21/E22, which is what `mean_fractional_shift` reports) stays finite
throughout. Two layers guard against this (`cliffordclock.pipeline`):

1. **Pre-flight, in dτ̃ resolution.** Before `mode="direct"`/`"worldline"`
   commit to a step size, the estimated worst-case per-step generator
   angle is checked against a conservative threshold. If exceeded: an
   **auto-selected** `dτ̃` is silently tightened and a note is recorded in
   the report's `uncertainty_notes`; an **explicit** `integration.dtau`
   is instead rejected with `PipelineConfigError`: silently overriding
   a user's explicit step size would be a worse surprise than failing the
   run outright.
2. **Backstop, in `_validate_physics`.** Every `EnsembleResult` field
   (`phase`, `phase_rotor`, `r_final`, `norm_error`, `max_norm_drift`) is
   checked for finiteness, so any run that still produces a non-finite
   rotor-diagnostic value raises `PhysicsValidationError` with a message
   pointing at the likely cause. The pipeline never silently returns a
   partially-`NaN` result.

`examples/quadrupole_classical.yaml`'s explicit Compton-scale `dtau: 0.5`
is nowhere near this guard; it only engages at large auto-selected `dτ̃`
combined with a coupling strong enough to matter. Full numeric detail and
the bug this fixed: "Design history" below.

### Safety net: the trajectory-memory guard (advisory + selector, WP19)

The time-stepping modes (`direct`/`worldline`) materialize a dense
`(M, steps + 1, 3)` float64 position trajectory. `select_dtau`'s
trap-period resolution keeps `steps` proportional to
`time_s / T_orb × points_per_period`, so a long `integration.time_s`
(seconds-scale, at a `~10⁵ rad/s` trap) resolves to `steps ~ 1e7-1e8`;
combined with an `ensemble.size` in the hundreds-to-thousands, that is a
silent 100+ GB (up to multi-TB) allocation attempt, enough to lock up
the host. Before propagating any trajectory, the pipeline estimates the
dominant allocation as `4 × M × (steps + 1) × 3 × 8` bytes (the position
array plus its documented same-shape siblings: the Verlet velocity
trajectory and transient sampler/scan copies, see
`cliffordclock.pipeline._check_trajectory_memory`), against
`integration.max_trajectory_memory_gb` (default **2 GB**).

**What happens when the estimate is exceeded now depends on
`integration.mode` and `integration.evaluation` (`auto` | `batched` |
`streaming`, default `auto`):**

- `mode: worldline` and `mode: secular` have no memory-bounded
  alternative implementation (WP19 scope, below); exceeding the
  estimate always rejects the config with `PipelineConfigError`, exactly
  as before WP19. The error is actionable: reduce `ensemble.size`, pass
  an explicit coarser `integration.dtau` (fewer steps), or switch to a
  mode that never materializes a dense trajectory: `secular`
  (Tier B(ii)) or the lattice `fast_path` (Tier A), both O(1) in
  `integration.time_s`. Users who want a larger batched
  allocation set `integration.max_trajectory_memory_gb` explicitly
  (`docs/cli.md`).
- `ensemble.regime: classical` + `mode: direct` (both `coupling.type`
  values) has an O(M)-memory **streaming** accumulator (WP19,
  below). Under `evaluation: auto` (the default), exceeding the estimate
  no longer raises: the pipeline switches to the streaming accumulator
  instead and records `"switched to streaming evaluation
  (memory-bounded)"` in the report's `uncertainty_notes`.
  `evaluation: batched` forces the pre-WP19 hard-reject behavior even
  for this combination (an explicit "I want the fast path or an error,
  never the slower streaming fallback" request); `evaluation: streaming`
  forces the streaming path unconditionally, even for a config that
  would fit comfortably batched (e.g. to measure/verify the streaming
  path itself, or to guarantee a memory-bounded run regardless of the
  estimate's accuracy). `integration.max_trajectory_memory_gb` keeps its
  meaning as the switch threshold in `auto` mode.

Effectively every config is runnable now: the one combination with an
automatic memory-bounded fallback (`ensemble.regime: classical` +
`mode: direct`) is also the one the original incident/reviewer reports
were about. `worldline`/`secular` keep the pre-WP19 hard-reject guard as
their only memory-safety mechanism (reduce `ensemble.size`/`dtau`, or
switch to a different mode, per the error message), a deliberate WP19
scope limit (below), not an oversight.

**Chunked evaluation (WP19, bounds the smoother's `N × K` term
independent of `N`).**
`cliffordclock.fields.smoother.chunked_apply`/`FieldSmoother.evaluate_chunked`
evaluate a `FieldSmoother`-backed field in fixed-size query chunks
(default 4096 points), bounding peak memory to `chunk_size × K × 3 × 8 × factor` bytes
regardless of how many query points are evaluated in total. Verified
bitwise-identical to the unchunked path for `chunk_size ≥ 2` (a `≤ 1`
ulp difference at `chunk_size = 1`, a `jax.vmap`-batch-of-one XLA
lowering detail, see `tests/test_fields_smoother.py`'s
`test_evaluate_chunked_matches_unchunked_exactly`). The streaming
accumulators below route every per-step field/`rate_fn` call through
this wrapper, so a `FieldSmoother`-backed field's evaluation cost stays
bounded independent of ensemble size `M` too, not just of `steps`.

**Fused streaming accumulators (WP19, bounds the `M × T` trajectory
term).** `cliffordclock.pipeline._stark_scalar_ensemble_streaming`
(`coupling.type: stark_dc`) and `_direct_rotor_ensemble_streaming`
(`coupling.type: linear_mu`) fuse velocity-Verlet propagation with the
per-step E19 midpoint / E21-E22 phase accumulation into a single
`jax.lax.scan` over the whole ensemble at once, mirroring how
`worldline.integrate_ensemble`/`_stark_rotor_ensemble` already stream
per-step field evaluation, except the fused scan also *advances* the
Verlet state, so the dense `(M, steps + 1, 3)` trajectory is never
materialized at all: memory is `O(M)`, independent of `steps`. Numerical
agreement with the batched path is measured directly:
bitwise-identical for the `linear_mu` rotor path on the configs tested
(`tests/test_e2e.py::test_streaming_matches_batched_linear_mu_direct`),
and `~1.7e-16` relative (machine-epsilon level, a different `jax.vmap`
batching structure over the identical underlying computation, not a
physical discrepancy) for the `stark_dc` scalar path against a real
`FieldSmoother`-backed field
(`test_streaming_matches_batched_stark_dc_direct_smoother_field`), both
asserted at a documented, deliberately loose bound
(`rtol=1e-12`/`1e-10` respectively).
`examples/showcase_gradient_dispersion_sr87.yaml` (the config whose
`max_trajectory_memory_gb: 8.0` override this document's earlier
revision cited as a "guard-passing-but-lethal" ~4.7 GB batched case) now
runs via forced streaming, with that override *removed*, in a measured
peak RSS of ~0.36 GB (vs. the batched path's measured ~4.69 GB on the
same machine) and a bitwise-identical `mean_fractional_shift`/SEM to the
batched run
(`test_showcase_forced_streaming_without_override_matches_batched_and_bounds_rss`),
the shipped example itself is unchanged and stays batched by default
(its override still comfortably covers the batched estimate).

**Trajectories in streaming mode.** `PipelineResult.trajectories` is
`(M, 2, 3)` (initial + final position only) by default for a streaming
run: the streaming path's whole point is never materializing the dense
array, so there is nothing denser to return for free. Diagnostics that
need a fuller trajectory (plots, notebooks) can set
`integration.trajectory_stride` (an explicit, opt-in, still-bounded
`O(M × steps / trajectory_stride)` subsampling, see `docs/cli.md`) or
use `evaluation: batched` (budget permitting) for the full dense array.

**What exactly the streaming accumulators cover, and what remains
batched-only:**

- **`direct` (classical), both `coupling.type` values.** Covered, as
  above.
- **`worldline` (lattice cross-check) and `secular`.** Explicitly **out
  of scope** for WP19 (unchanged from the pre-WP19 guard): `worldline`'s
  static-node broadcast trajectory and `secular`'s closed-form one-orbit
  quadrature are both validation/O(1)-cost paths already, not the
  `time_s`-driven-blowup failure mode this WP targets, and the WP19 plan
  is explicit that no changes to the rotor accumulator
  (`worldline.integrate_ensemble`) or the `fastpath`/secular math are in
  scope. `secular`'s `rate_fn` call has the same whole-trajectory shape
  as `_stark_scalar_ensemble`'s batched path (so, in principle, the same
  smoother-evaluation memory profile), but neither the base trajectory
  term nor a streaming accumulator has been extended to it here;
  `points_per_period`'s default (100) keeps this well below the base
  term in every configuration this project ships, and both `worldline`
  and `secular` keep the pre-WP19 hard-reject guard as their only
  memory-safety mechanism. A future WP could extend streaming to these
  modes using the same fused-scan pattern; not done here.

### Accuracy study

CONVENTIONS.md V4 (harmonic trap, classical atom, linear-gradient field)
has a closed form: the field-gradient (pivot) contribution time-averages
to its value at the trap center (a linear field's oscillating part
cancels exactly over full periods), plus the second-order Doppler term
`-⟨v²⟩/2c²` (SHM virial identity, `⟨v²⟩ = ½(ω²|Δr₀|² + |v₀|²)`). Sweeping
`points_per_period` (equivalently `dτ̃`) at a fixed 3-period span
(`tests/test_fastpath_select_dtau.py`), against an isotropic
`ω = 2×10⁵ rad/s` trap and a target-regime pivot magnitude
(`P(center) - 1 ≈ 3.77×10⁻¹⁸`, chosen to match this project's actual
1e-18 target regime, see the finding below for why):

| points/period | `dτ̃` | `dτ` (s) | relative phase error | max rotor-norm drift |
|---|---|---|---|---|
| 25 | 9.76e+14 | 1.26e-06 | 4.54e-08 | 1.09e-13 |
| 50 | 4.88e+14 | 6.28e-07 | 1.14e-08 | 1.13e-13 |
| **100 (E31 default)** | **2.44e+14** | **3.14e-07** | **2.84e-09** | **1.14e-13** |
| 200 | 1.22e+14 | 1.57e-07 | 7.10e-10 | 8.55e-14 |
| 400 | 6.10e+13 | 7.85e-08 | 1.78e-10 | 8.54e-14 |
| 800 | 3.05e+13 | 3.93e-08 | 4.44e-11 | 7.86e-14 |

Fitting `log(error)` vs `log(dτ̃)` over this sweep gives a measured slope
of **1.999**, the expected order-2 scaling region (E19's design order),
confirmed directly (`test_v4_error_scales_order_two_with_dtau`). At the
default `points_per_period=100` over ≥ 3 trap periods, the phase matches
the closed form to **2.84e-9 relative** (inside this project's
`rtol ≤ 1e-8` accuracy bound with ample margin) and rotor-norm drift stays
`< 1.2e-13`, inside the `< 1e-12` bound. `select_dtau`'s default
(`N_res = 100`) is confirmed adequate; this study found no reason to
tighten or relax it.

### `renorm_every` at large `dτ̃`

At large auto-selected `dτ̃`, the per-step rotor-renormalization cadence
matters more than it does at Compton scale: `cliffordclock.pipeline`
auto-selects a tighter `renorm_every` whenever `integration.dtau` is auto-selected and
`integration.renorm_every` is left unset, targeting `max_norm_drift <
1e-12` (E20). An explicit `integration.dtau` or explicit
`integration.renorm_every` is always honored unchanged. This only affects
the E20/E24 rotor-diagnostic fields (`norm_error`/`max_norm_drift`/
`phase_rotor`): **the primary scalar phase that `mean_fractional_shift`
reports is never affected**, at any `renorm_every`: it is accumulated
directly, never through the rotor exponential. `select_dtau` and the
fast-path/secular-averaging code paths are entirely unaffected (neither
ever calls the rotor exponential). Full measurement detail: "Design
history" below.

## Tier B(ii): secular averaging (E30)

For a classical atom in **periodic** motion (an isotropic harmonic trap,
so the orbit is guaranteed to close for arbitrary initial conditions) and
a **static** field:

```
ΔΦ = ⟨δω̃⟩_orb · T̃ + ε
⟨δω̃⟩_orb = (1/T̃_orb) ∮ δω̃(r(t̃), v(t̃)) dt̃   (one-orbit line integral)
|ε| ≤ T̃_orb · max_t |δω̃(t̃) − ⟨δω̃⟩_orb|         (documented remainder bound)
```

The one-orbit integral is evaluated once (E19-style midpoint quadrature
along the trap's exact closed-form SHM orbit, Kahan-summed), then scaled
by the *full* interrogation time `T`: cost independent of `T`, like
Tier A. `integration.mode: secular` requires `integration.time_s` and an
isotropic trap; `cliffordclock.integrator.fastpath.secular_average_shift`
raises a clear `ValueError` (surfaced as `PipelineConfigError` by the
pipeline) otherwise. **Validity bounds (E30): static field, periodic
motion, `T ≫ T_orb`.** Not valid for drifting/chaotic trajectories or
time-dependent fields; use `mode: direct` (Tier B(i)) there.

**Validated ("Tier B(ii) ≡ Tier B(i)"),
`tests/test_fastpath_secular.py`:**

- **Exact period multiple** (`T = 5 T_orb`): secular and direct
  integration (Tier B(i), same `dτ̃`) agree to **~3.6e-16 relative**,
  floating-point summation-order noise, not a physical approximation
  (both evaluate the same midpoint-quadrature/Kahan-sum of the same
  `rate_fn` along numerically the same periodic trajectory). Tested at
  `rtol=1e-9`, ~1e6x looser than the measured discrepancy.
- **Partial orbit** (`T = 4.37 T_orb`, the realistic case): the
  discrepancy (`5.8e-8`, absolute phase units) stays within E30's own
  documented remainder bound (`8.4e-7` for this case, ~14x looser than
  the actual discrepancy, a bound tight enough to be meaningful); this
  is the test that exercises E30's approximation, beyond ordinary
  numerical noise.

## Tier C: the Compton-fine worldline integrator (validation mode)

The original Compton-scale rotor integrator (E17-E24) at `dτ̃ ~ 1` remains
fully available and is still exercised by the full test suite
unchanged (`tests/test_integrator_worldline.py`,
`tests/test_integrator_stepper.py`, and the Case A-D suite in `tests/test_e2e.py`).
It is what Tiers A/B(i)/B(ii) are validated against above, and remains
the tool of record for regimes outside E29/E30's validity bounds
(time-dependent fields, anisotropic-trap or non-periodic classical
motion, or cross-checking a fast-path result from first
principles).

## Design history (historical, superseded narrative, kept for the record)

Everything above is current reference. This section is the postmortem
narrative behind the "Safety net" and "`renorm_every` at large `dτ̃`"
mechanisms above: how the bugs were found and characterized during
development. Nothing here changes current behavior; read the sections
above for that.

### How the per-step rotor generator angle bound was found

`select_dtau`'s step-size rule resolves *trap* dynamics only; it says
nothing about coupling strength or field magnitude, both of which set how
large the per-step rotor generator is. At a realistic E14a `mu`
(`~1e-25`) combined with a large auto-selected `dτ̃`, the per-step
generator angle can reach thousands of radians, far past the rotor
exponential's fixed-order (12-term Taylor, 10-halving) convergence range.
Direct probing at increasing bivector magnitude found it still returns a
*finite* but badly wrong rotor (`|R| ~ 1e56`, nowhere near a unit rotor)
at a 5000 rad generator angle, and `NaN`/`Inf` by 10000 rad, while the
primary scalar phase stays finite throughout, so only the rotor-diagnostic
fields (`norm_error`/`max_norm_drift`/`phase_rotor`/`r_final`) go bad.
Before the fix, `cliffordclock.pipeline._validate_physics` only checked
the primary `phase` for finiteness, so a `NaN`-contaminated
`max_norm_drift` silently passed the sanity check (`NaN` compares `False`
against anything in NumPy) without raising: a garbage-in-garbage-out
failure mode with no error at all. The fix added two layers: a pre-flight
check on the estimated worst-case per-step generator angle (threshold
`MAX_PER_STEP_ROTOR_ANGLE_RAD = 0.5` rad, four orders of magnitude below
where the rotor exponential was observed to start failing), and a
backstop finiteness check on every diagnostic field in
`_validate_physics`, not just `phase`. Reproduction: a 500 uK classical
ensemble at a realistic-mu quadrupole setup gave a naive auto-selected
`dτ̃` with an estimated per-step angle of `~8863` rad; the fix
auto-tightens `dτ̃` down to `~1.4e10` (from the naive `~2.4e14`) and
produces a fully finite result, or, with that same `dτ̃` given explicitly
instead, raises `PipelineConfigError` loudly (`tests/test_e2e.py`'s
`test_wp8_major1_*` tests).

### How the `renorm_every` cadence question was found

At Compton-scale `dτ̃ ~ 1`, a single rotor-exponential call's
floating-point floor is far below `1e-16` (the generator's magnitude is
astronomically tiny), so the default renormalization cadence
(`renorm_every=1000`) keeps accumulated rotor-norm drift well under
`1e-12` even over a million steps. At large auto-selected `dτ̃`, the
per-step generator magnitude is no longer vanishingly small (`~5e-4` rad
in the accuracy-study sweep above), and the rotor exponential's own
per-step floor (`~1e-13`) is correspondingly larger. Left unrenormalized
for hundreds of steps at the coarser default cadence, this accumulates
past `1e-12` (measured `4.5e-11` for a 500-step, 5-period case). This is a
cadence question, not a correctness bug: norm preservation being
step-size-independent is a property of the *exact* exponential map, and
the fixed-order Taylor implementation's own error floor is what a
tighter cadence compensates for. An early framing of this finding
described the coarse default cadence as "unaffected for practical
purposes," based on a tiny-coupling probe (`mu = 1e-33`, chosen to stay
inside the rotor exponential's convergence range at large `dτ̃`) with
~20x margin under the pipeline's coarse sanity threshold. That framing
undercounted the realistic-coupling case: a longer run at realistic
coupling can push drift past `1e-12` comfortably at the coarser cadence.
The fix: the pipeline auto-selects a tighter `renorm_every` whenever
`dτ̃` is auto-selected and
`renorm_every` is left unset, targeting `max_norm_drift < 1e-12` against
a documented per-call drift floor (`renorm_every = max(1, floor(1e-12 /
2e-13)) = 5`). Measured with the fix: a moderate-coupling classical
`direct` run at auto-`dtau` over 20 trap periods (2000 steps) keeps
`max_norm_drift` at `~3.9e-14` with the auto-selected `renorm_every=5`; a
lattice `worldline`-mode run at auto-`dtau` over 10 periods (1000 steps)
keeps `max_norm_drift` at `~5.1e-13`. This is a documented, conservative
heuristic, not an exact bound, but every case measured stays orders of
magnitude under the pipeline's coarse sanity threshold regardless
(`tests/test_e2e.py`'s `test_wp8_major2_*` tests).
