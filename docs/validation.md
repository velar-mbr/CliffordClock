# Validation status

This page includes every case CliffordClock has
been checked against, with the formula/source, the tolerance, and the
measured agreement. "Passed" here means "agrees with an independently
computed or independently sourced number to the stated tolerance."

See `docs/CONVENTIONS.md` for every cited equation. Tolerance doctrine 
throughout this project: explicit `rtol` with `atol=0` via 
`assert_allclose`, never a bare `pytest.approx`.

**TLDR.** 
This engine's validation is
exact closed-form self-consistency (V1-V4) plus five literature
known-answer checks (KA1-5, textbook formulas and published
polarizabilities) plus **two experimental reproducibility cases**
against public data (NPL Rydberg electrometry and Bothwell et al.'s
mm-scale gravitational-redshift measurement, explicitly *not* blind
predictions; see below), plus one weaker-class arithmetic-reproduction
case against JILA's own published BBR row (explicitly *not* a
reproducibility case either; see below), plus a Ca+:D5/2 ion-clock
quadrupole-slope case against Roos et al.'s measured two-ion Fig. 4a slope
(two labeled variants: a cross-vintage comparison against an independent
theory Theta, and an arithmetic-reproduction variant against Roos's own
extracted Theta; see below). Zero blind predictions exist because no 
public source lets this engine predict a shift nobody
had already computed from the same inputs. **(If you have one please share)**

## Summary

| Case | What it checks | Reference | Tolerance | Status |
|---|---|---|---|---|
| V1 | Uniform field, at rest: rotor-rate shift beyond the scalar baseline | Closed form (CONVENTIONS.md section 9) | `< 1e-19` | Pass (`tests/test_e2e.py::test_case_a_*`) |
| V2 | Constant gradient, single static atom | Closed form (CONVENTIONS.md section 9) | rtol `1e-12` | Pass (`tests/test_e2e.py::test_case_b_*`) |
| V3/E24 | Quadrupole field, M=100 classical ensemble, scalar path **and** rotor path | Independent plain-NumPy reference (`tests/reference_impl.py`) | rtol `1e-10` (scalar), E24-derived bound (rotor) | Pass (`tests/test_e2e.py::test_case_c_*`) |
| V4 | Harmonic trap, linear-gradient field, large-`dτ̃` accuracy sweep | Closed form (CONVENTIONS.md section 9) | Order-2 convergence, measured table | Pass (`notebooks/02_step_size_study.ipynb`, `docs/timescales.md`) |
| **KA1** | Sr87 uniform-field DC-Stark shift | Textbook formula, literature Δα | rtol `1e-10` | **Pass**, exact to float64 (`tests/test_known_answers.py::test_ka1_*`) |
| **KA2** | Yb171 uniform-field DC-Stark shift | Textbook formula, literature Δα | rtol `1e-10` | **Pass**, exact to float64 (`tests/test_known_answers.py::test_ka2_*`) |
| **KA3** | Yb171 linear-gradient DC-Stark: mean shift + phase spread | Independent Gaussian-moment reference (`tests/reference_impl.py`) | rtol `1e-8` | **Pass**, mean exact, variance rtol `5.5e-11` (`tests/test_known_answers.py::test_ka3_*`) |
| **KA4** | Sr87 second-order Doppler shift, classical thermal ensemble | Equipartition (`-⟨v²⟩/2c² = -3k_BT/2mc²`) | `5×SEM` (statistical) | **Pass**, `0.32σ` measured (`tests/test_known_answers.py::test_ka4_*`) |
| **KA5** | Sr87/Yb171 blackbody-radiation shift at 300 K + one non-trivial T | Closed-form E32 polynomial, hand-computed to 50-digit `decimal` precision | rtol `1e-12` | **Pass**, exact to float64 (`tests/test_bbr_pivot.py`) |

V1-V4 are this engine's self-consistency validation (it agrees with
*its own* closed forms and an independent scalar re-implementation).
**KA1-4 are the first cases that check the engine against numbers the
optical-clock community already knows**: literature-cited
polarizabilities, the textbook DC-Stark formula, and the standard
equipartition second-order-Doppler result, at realistic (not
femtosecond) interrogation times, via the physical E14b coupling and the
fast-path architecture (`docs/timescales.md`). **KA5** extends this to
the E32 BBR pivot term's closed-form polynomial (a formula check,
not yet a check against a published *row*; see the arithmetic-
reproduction case below for that).

**The dataset benchmark (`benchmarks/`):** comparison against a published
experimental *dataset*, for example NIST/JILA lattice-clock shift
measurements, with the residuals reported as found. KA1-4
validate the tool reproduces the right *equations*; the dataset benchmark
checks it against real *measurements*. The result, across six authorized
sources
(arXiv:2403.10664 + its PRL Supplemental Material follow-up,
data.nist.gov DOI 10.18434/M32206, arXiv:1706.01944, Metrologia
63,025002, and Metrologia 61,015006): **1 of 1 reproducibility case MET**
within this dataset-benchmark pass. NPL's Rydberg-electrometry paper
(arXiv:1706.01944) publishes an
independent stray-field measurement, and this engine's `coupling.type:
stark_dc` pipeline, given that field and the Middelmann-sourced `Δα`
this project's registry already uses, predicts a DC-Stark shift band
that overlaps NPL's own published band, labeled a
*reproducibility* check (NPL already combined the same two ingredients),
not a blind prediction. Project-wide, the reproducibility count is
**two**, with the Bothwell mm-scale gravitational-redshift case (below)
built as a separate benchmark pass the same way the BBR and Roos cases
are. **0 of 0 blind-prediction cases**: that
category remains empty; no authorized source yet lets this engine
predict a shift nobody had already computed from the same inputs. The
most promising lead for closing that gap (Li J et al 2024, Metrologia
61,015006, USTC's own reference for its applied-field characterization)
was authorized and attempted but could not be accessed by any legitimate
route (no arXiv preprint, no open-access license, ResearchGate blocked);
it is reported as its own "not accessed" outcome, not folded into a
negative finding. **13 rows not-applicable**: 8 of
9 JILA systematic-shift line items and most of USTC's Table 3 are
outside this engine's physics scope entirely; JILA's and USTC's own
DC-Stark rows are in scope but lack a published independent field input
(unlike NPL's); the NIST dataset measures a different physical quantity
(phase/Allan-deviation instability, not a systematic shift). Full gap
analysis and citations: `benchmarks/RESULTS.md`, `benchmarks/MAPPING.md`.

**The BBR arithmetic-reproduction case (WP20):** a second, structurally
*weaker* benchmark case, kept separate from the dataset-benchmark totals
above. `benchmarks/run_bbr_jila_arithmetic_reproduction.py` evaluates
this engine's real BBR pivot functions
(`cliffordclock.integrator.omega.bbr_pivot_perturbation`/
`bbr_pivot_uncertainty`) with the pinned Sr87 registry coefficients at
JILA's own published operating temperature (`T = 293.282(4) K`), and
compares against JILA's own published BBR row (arXiv:2403.10664 Table I:
`-4.84172(73)×10⁻¹⁵`): predicted `-4.841743×10⁻¹⁵`, residual
`-2.251×10⁻²⁰`, bands overlap, `kpi_verdict = "MET"`. **Binding label**
(the project's theory sign-off record (G7), B5): **"arithmetic reproduction of a
published standard-formula evaluation"**, explicitly weaker than the
NPL `"reproducibility"` case above, since JILA's own row is itself
computed (their T and their coefficients through the standard BBR
formula), not an independent measurement of the shift. Agreement is
expected almost by construction (the registry's dynamic polynomial is
itself anchored/rescaled to this exact JILA value); this case
demonstrates the engine's arithmetic and provenance chain, not
independent BBR physics validation. Full write-up:
`benchmarks/RESULTS.md`'s "Arithmetic-reproduction case: JILA BBR row"
section; provenance: `benchmarks/MAPPING.md`'s WP20 addendum,
`benchmarks/SOURCES.md` section 1.

**The Roos quadrupole-slope case (WP21 ion-clock benchmark):** a third
benchmark case, also kept separate from the dataset-benchmark totals
above, exercising the WP21 Tier-2 electric-quadrupole-shift module
(CONVENTIONS.md E34/E35). `benchmarks/run_roos_quadrupole_slope.py`
evaluates this engine's real quadrupole-shift functions
(`cliffordclock.integrator.omega.quadrupole_shift_joules`/
`quadrupole_mj_factor`) with the species registry's Ca+:D5/2 entry
(`cliffordclock.ensemble.species.QUADRUPOLE_MOMENTS`) against Roos et
al.'s (quant-ph/0701215v1) measured two-ion Fig. 4a slope,
`a = 2.975(2) Hz*mm^2/V`, in two labeled variants (**binding label,
the project's theory sign-off record (G8), B4**, the same ruling
already ratified for the analogous Barwood case): a headline
**cross-vintage comparison** against Itano's independent theory value
(`Theta_theory = 1.917 ea0^2`, predicted `|a_pred| = 3.115229` Hz*mm^2/V,
residual `+4.71%`, `kpi_verdict = "NOT MET"`, EXPECTED, recovering the
literature's own known ~4.75% theory-vs-measurement Theta tension, not an
engine defect), and a secondary, explicitly circular **arithmetic
reproduction** against Roos's own extracted Theta (`1.83(1) ea0^2`,
predicted `|a_pred| = 2.973849` Hz*mm^2/V, residual `-0.04%`,
`kpi_verdict = "MET"`, plus an exact float64-precision closed-loop check
of this engine's coefficient against Roos's own stated `Theta =
(5/12)*h*a` conversion relation). A structural pin, the two-ion state's
24/5 enhancement over a single ion's shift, dossier-stated and
independently recovered here from real engine calls, backs
both variants. Full write-up: `benchmarks/RESULTS.md`'s "Cross-vintage
comparison: Roos et al. quadrupole slope" section; provenance:
`benchmarks/MAPPING.md`'s Roos-benchmark addendum, `benchmarks/SOURCES.md`
section 7.

**A numerical-safety note on how KA1-4 get their numbers.** The pipeline
evaluates the E14b Stark pivot directly from the total field, with no E11
baseline/perturbation split (see `docs/coupling.md`'s "Historical design
notes" appendix for why); this is measured safe against catastrophic
cancellation through at least ~1e8 V/m stray-field biases, and an
adversarial-magnitude regression test
(`tests/test_e2e.py::test_wp9_major1_stark_dc_adversarial_gradient_pins_no_cancellation_regression`)
pins this at a |E0| = 1e5 V/m benchmark.

## KA1: Sr87 uniform-field DC-Stark shift

**Formula (CONVENTIONS.md E14b):**

```
Δν/ν₀ = -(Δα/2)|E|²/(h ν₀)
```

**Source:** Δα = 4.07873(11)×10⁻³⁹ C²m²J⁻¹, T. Middelmann, S. Falke, C.
Lisdat, U. Sterr, "High Accuracy Correction of Blackbody Radiation Shift in
an Optical Lattice Clock", Phys. Rev. Lett. 109, 263004 (2012),
arXiv:1208.2848 (`cliffordclock.ensemble.species.SR87`).

**Test:** `tests/test_known_answers.py::test_ka1_sr87_uniform_field_dc_stark_matches_textbook_formula`.
Full pipeline (species registry → synthetic uniform field → lattice fast
path, E29 → report), at a genuine 1 s interrogation, `coupling.type:
stark_dc` (no override; coefficient resolved from the registry).

**Quotable number** (100 V/m stray field, 1 s Ramsey interrogation):

```
Δν/ν₀ = -7.170524e-17
Δν    = -3.077790e-02 Hz
```

**Measured agreement:** exact to float64 precision (relative error `0.0`
against the independently-written textbook formula, well inside the
`rtol=1e-10` target); the pipeline's E14b wiring reproduces the DC-Stark
shift a lab would predict by hand from the published Δα.

**Runnable example:** `examples/lattice_sr87_stark.yaml`
(`cliffordclock run examples/lattice_sr87_stark.yaml`).

## KA2: Yb171 uniform-field DC-Stark shift

Same formula and pipeline path as KA1.

**Source:** Δα = 2.40269(5)×10⁻³⁹ C²m²J⁻¹ (= 145.726(3) a.u.), J.A.
Sherman, N.D. Lemke, N. Hinkley, M. Pizzocaro, R.W. Fox, A.D. Ludlow, C.W.
Oates, "High-Accuracy Measurement of Atomic Polarizability in an Optical
Lattice Clock", Phys. Rev. Lett. 108, 153002 (2012), arXiv:1112.2766, Table
II (`cliffordclock.ensemble.species.YB171`).

**Test:** `tests/test_known_answers.py::test_ka2_yb171_uniform_field_dc_stark_matches_textbook_formula`.

**Quotable number** (100 V/m stray field, 1 s Ramsey interrogation):

```
Δν/ν₀ = -3.498114e-17
Δν    = -1.813058e-02 Hz
```

**Measured agreement:** exact to float64 precision (relative error `0.0`),
same as KA1.

## KA3: gradient line shift + broadening (Yb171)

A field linear in position (`E(r) = E₀ + G·r`, CONVENTIONS.md V2's
constant-gradient field) makes the E14b pivot `P(r) - 1` a *quadratic*
polynomial in position. Over a ground-motional-state (`n=0,0,0`) lattice
ensemble, whose position density is an exact multivariate Gaussian,
unlike excited motional states, both the mean shift and its variance
(the inhomogeneous-broadening contribution the shift's spread would
produce) are known in closed form via Gaussian-moment perturbation theory.

**Reference:** `tests/reference_impl.py::stark_shift_mean_and_variance`
derives, independently (plain NumPy, no `cliffordclock.integrator`/`cl13`
imports):

```
⟨Δν(r)/ν₀⟩_ψ = (k_S/ν₀) · [|E₀|² + Σ_i ⟨r_i²⟩ · ‖A_row_i‖²]
Var(Δν(r)/ν₀)_ψ = (k_S/ν₀)² · [bᵀΣb + 2·tr((CΣ)²)]
```

(`A` the gradient tensor, `b = 2A·E₀`, `C = AAᵀ`, `Σ = diag(σ_x², σ_y²,
σ_z²)` the ground-state position variances `σ_k² = ħ/(2mω_k)`; see the
module's derivation comment for the full Wick's-theorem argument, including
why the linear-in-`r` term drops from the mean and why the linear/quadratic
covariance term vanishes from the variance.)

**Test:**
`tests/test_known_answers.py::test_ka3_gradient_field_mean_and_variance_match_gaussian_moment_reference`,
species Yb171, `E₀ = (30, -20, 10)` V/m, a modest gradient tensor, lattice
regime (`n_quad=6`, exact for the required polynomial degree), `coupling.type:
stark_dc`, 1 s interrogation.

**Quotable numbers:**

```
⟨Δν/ν₀⟩_ψ  = -4.897360e-18
sqrt(Var)_ψ =  4.656821e-24
```

**Measured agreement:** mean exact to float64 (relative error `0.0`);
variance rtol `5.5e-11`, both far inside the `rtol=1e-8` target.

## KA4: second-order Doppler shift (Sr87)

**Derivation (equipartition, per-axis accounting; see the test's
docstring for the full argument):** in a 3D isotropic harmonic trap, each
of the 3 Cartesian velocity components carries `(1/2)k_BT` of kinetic
energy by equipartition, so `⟨v²⟩ = 3k_BT/m` and the E21 kinematic term
gives

```
⟨Δν/ν₀⟩_kinematic = -⟨v²⟩/(2c²) = -3k_BT/(2mc²)
```

The pipeline's `integration.mode: secular` (E30) reports each atom's
*orbit-time-averaged* `v(t)²`, not its instantaneous value at `t=0`; by the
virial theorem for a harmonic oscillator this time-average equals half the
atom's total mechanical energy, and, because the ensemble draws both
positions and velocities from the correct thermal (Boltzmann) distributions,
the ensemble mean of that time-average converges to *exactly* the same
`3k_BT/(2mc²)` value as the naive instantaneous-equipartition formula.

**Test:**
`tests/test_known_answers.py::test_ka4_second_order_doppler_matches_equipartition`,
Sr87, T = 5 μK, M = 5000 classical atoms, `field: uniform, e0=[0,0,0]`
(isolates the kinematic term: the E14b Stark contribution is then
identically zero for any species), `coupling.type: stark_dc`,
`integration.mode: secular`, 1 s interrogation.

**Quotable numbers:**

```
Equipartition prediction: -7.983437e-21
Measured (M=5000):        -8.003764e-21 +/- 6.33e-23 (SEM)
```

**Measured agreement:** `0.32σ` (well inside the `5σ` statistical
tolerance); this is a genuine Monte Carlo comparison (unlike KA1-3's
exact closed forms), so the tolerance is stated in multiples of the
pipeline's own reported standard error, not a bare relative tolerance; the
deviation stays comfortably under 1σ across multiple seeds checked during
development.

This case is also a real uncertainty-budget line item, not just a
consistency check: the second-order Doppler shift is a genuine systematic
every optical-lattice-clock uncertainty budget carries, and this is the
tool computing it exactly (E21's kinematic term), not approximating it.

## KA5: blackbody-radiation shift closed-form check (Sr87/Yb171)

**Formula (CONVENTIONS.md E32):**

```
(P-1)_BBR = [Δν_stat·(T/T₀)⁴ + Δν_dyn(T)] / ν₀ ,   T₀ = 300 K
```

**Source:** registry coefficients (`cliffordclock.ensemble.species.SR87`/
`YB171`'s `bbr_coefficients`), each citation-traced to Middelmann/
Lisdat/Aeppli/arXiv:2507.14030 (Sr87) or Hassan/Beloy (Yb171); see
CONVENTIONS.md §13 and `docs/coupling.md`'s "Blackbody-radiation shift"
section.

**Test:** `tests/test_bbr_pivot.py`: `bbr_pivot_perturbation` at
`T = 300 K` (the registry's own reference temperature) and `T = 250 K` (a
non-trivial `T` inside the `[50, 350] K` validity window), both species,
each checked against an independently hand-computed 50-digit `decimal`
reference (not a copy of the implementation).

**Quotable numbers (T = 300 K):**

```
Sr87:  (P-1)_BBR = -5.319504e-15   (Δν = -2.28328 Hz)
Yb171: (P-1)_BBR = -2.464643e-15   (Δν = -1.277414 Hz)
```

**Measured agreement:** exact to float64 precision (rtol `1e-12` against
the independent decimal reference) at both temperatures, both species.
This is a closed-form arithmetic check of the E32 formula (including its
mandatory sign regression, `tests/test_bbr_pivot.py::test_bbr_pivot_sign_regression_sr87_300k`),
distinct from the check against an external published *row* (the
JILA-2024 arithmetic-reproduction case, now built; see "The BBR
arithmetic-reproduction case (WP20)" above, labeled a weaker
class than a reproducibility case).

## Non-goals of this validation pass

- **A genuine blind-prediction-grade BBR case** (an engine prediction
  checked against an independently measured field/temperature the
  engine's own inputs did not already combine) remains out of scope;
  the JILA-2024 arithmetic-reproduction case (above) is deliberately a
  weaker class, since JILA's own row is itself computed, not
  independently measured. T(r) spatial maps, solid-angle
  effective-temperature computation, and stochastic BBR-field sampling
  are also explicitly out of scope (uniform `T` only); see
  CONVENTIONS.md §13 and `docs/coupling.md`.
- No fitted or tuned parameters anywhere: every expected value above is
  derived from a literature formula/citation or an independent
  re-implementation, computed once and never adjusted to make a test pass.
  If a case had missed its target, that would be reported here as a
  finding, not silently corrected.

**The Bothwell mm-scale gravitational-redshift case (WP22 extended-
lattice benchmark):** a fourth case, extending the extended-lattice
ensemble regime (`ensemble.regime: lattice_extended`, CONVENTIONS.md
section 15) and its new gravitational-redshift pivot term (E36) against
Bothwell et al.'s (Nature 602, 420 (2022)) real, single-apparatus
measurement of General Relativity's gravitational time dilation across a
millimetre-scale atomic sample. `benchmarks/run_bothwell_redshift.py`
configures the REAL per-site pipeline to their sample geometry (Gaussian
envelope, ~5900-site computational grid, real 406.5 nm magic-lattice
spacing) at their surveyed local gravity, and compares the fitted
per-site slope against BOTH their corrected measurements, MET at
0.48-sigma and 0.70-sigma respectively, both bracketing the prediction.
Labeled **`"reproducibility"`, with the caveat INVERTED from the BBR
case's**: unlike NPL's nontrivial reconstruction, the `g/c^2`
arithmetic here is textbook and the authors computed it themselves;
what this case validates is the extended-sample MACHINERY
(per-site geometry, envelope weighting, map assembly) producing the right
measured-map slope end-to-end, with zero adjustable inputs. The
blind-prediction count stays unchanged. **This is the project's second
reproducibility case (owner-ratified 2026-08-11), alongside NPL's above**:
the project headline is now two reproducibility cases, zero blind
predictions. See `benchmarks/RESULTS.md`'s "Reproducibility case:
Bothwell..." section and `benchmarks/MAPPING.md`'s WP22 addendum for the
full method and provenance.
