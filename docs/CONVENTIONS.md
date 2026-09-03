# Physics & Numerical Conventions: CliffordClock

**Version:** 1.14.0 · **plus §18's E42 addition (2026-08-29, WP38 Phase
2)**, the differentiable sideband-spectrum forward model (harmonic and
BO+WKB paths) built on §17's E41 JAX core, specified directly by the
project owner, likewise not carrying its own separate formalism sign-off
record; see §18's own changelog entry for the full record. **Status:
reviewed and approved** (2026-08-11, per
the project's G9 theory sign-off record, following
the owner's trigger after reviewing the Fortier/Luiten/Margolis survey
(Optica 13, 143 (2026)): mm-scale extended samples and their gravitational
redshift observable), **plus §13's E37 addition (2026-08-22, WP29 Tier 1)**,
a multi-surface generalization of the already-approved E32/E33 BBR term
specified directly by the project owner following the project's internal
BBR thermal-environment research dossier, not carrying its own separate
formalism sign-off record, **plus §16's E38 addition (2026-08-22, WP30)**,
the quantum-motional second-order-Doppler (time-dilation) pivot term
specified directly by the project owner, likewise not carrying its own
separate formalism sign-off record, **plus §8's E39 addition and §16's
participation-factor extension (2026-08-22, WP31)**, both specified
directly by the project owner, likewise not carrying their own separate
formalism sign-off record, **plus §16's radial-spectrum-reconstruction
addition (2026-08-23, WP32)**, replacing the axial-form-as-radial
approximation with a genuine two-ion radial eigenproblem inversion for a
caller who supplies the lab's own measured radial mode frequencies,
specified directly by the project owner, likewise not carrying its own
separate formalism sign-off record, **plus §16's mode-specific intrinsic-
micromotion enhancement addition (2026-08-23, WP33)**, closing the
reconciliation the G14 gate review identified (WP32's radial rows already
include a published intrinsic-micromotion shift whose correct enhancement
factor is mode-specific, not a uniform factor of two), specified directly
by the project owner,
likewise not carrying its own separate formalism sign-off record. §15
(E36) was approved 2026-08-11
conditional on the A1 computed-magnitude regression and the A4 extended-mode
dispersion-labeling edit; see §15's sign-off note for the itemized record.
§1–§11 (E1–E28) were reviewed and
approved 2026-08-08 (signature/metric convention, the ħ-vs-h resolution,
the E14 coupling split, the rotor/Ω construction, the T₂* formula, and the
exponential-midpoint stepper; see §11 for the itemized record). §12
(E29–E31), the E14b implementation path, the full-precision
`ALPHA_AU_TO_SI`, and the E6 range-reduction note were reviewed and
approved 2026-08-09 (see §11/§12 for the itemized records). §13 (E32–E33,
the blackbody-radiation shift) was reviewed and approved 2026-08-11 (G7:
Part A items 1–3 approved with one mandatory sign correction and two
required documentation edits; Part B items 1–5 all ratified with edits;
see §13's sign-off note for the itemized record). §14 (E34–E35, the
ion-clock electric-quadrupole shift, WP21 Tier 2) was approved 2026-08-11
conditional on the A1 sign discipline (G8: primary-text verification
against Itano 2000 Eq. 46 and Roos et al. quant-ph/0701215v1 Eq. 1,
required before wiring; see §14's sign-off note for the itemized
record and its AMBIGUITY flag on the absolute-triple anchor).

This document is the single source of truth for every equation implemented
in this repository. Code must cite equation numbers (E1,
E2, …) in docstrings. Disagreement between code and this document is a bug
in the code. Items marked **[INTERPRETATION]** record editorial choices
this document makes that are not forced by the cited equations themselves
(a combination, a scaling, or a convention fixed here); they are
enumerated again in §11 for explicit human review.

**Production path vs. general engine (read this before §5–§6; updated
WP16).** The production `coupling.type: stark_dc` path (E14b) computes
the scalar observable E21 with a scalar phase accumulator for
`integration.mode: direct`/`fast_path`/`secular`. For `integration.mode:
worldline` (the lattice-regime cross-check tier) it instead constructs
the true Cl(1,3) rotor `Ω` (E17–E18) directly from the E14b pivot
(`build_omega_stark`, instantiating the pivot-general E15–E18 chain below
with E14b instead of E14a), and the rotor and scalar paths are **directly
verified against each other**: `tests/test_integrator_stark_rotor.py`
checks first-order rotor↔scalar agreement at realistic parameters, the
permitted E24 O(ω_boost²) divergence under a deliberately exaggerated
boost, a uniform-field null, and a v=0 static-node case, and the NPL
reproducibility benchmark (`benchmarks/run_benchmarks.py`) re-runs through
the rotor path with the same MET verdict as the scalar path. This direct
test now sits alongside the chain that originally established the
equivalence (still valid derivation background, not superseded): (i) the
rotor integrator is checked against the scalar formulation for the linear
validation coupling (E14a) by an executed, standing cross-check (E24);
(ii) the physical DC-Stark coupling (E14b) is checked against that same
linear coupling as its exact linearization about the bias field by a
second executed test (the bridge identity); and (iii) the independent
theory review confirmed that the rotor's additional (boost) content
alters the observable only at second order, far below measurability for
any realistic configuration. `integration.mode: direct` (classical-
ensemble trajectories) still uses the scalar accumulator only: a rotor
cross-check there is not this WP's target and remains a deferred,
post-beta candidate (see `cliffordclock.pipeline`'s module docstring mode
table). The Cl(1,3) rotor integrator remains the general formalism this
library is built around: the foundation for geometric effects beyond
DC-Stark (magnetic couplings, transport-induced geometric phases) that a
future coupling could need.

## 1. Notation

- `e_0, e_1, e_2, e_3`: orthonormal spacetime basis vectors; `I = e_0123`
  the pseudoscalar. Bold `r`, `v`, `E` are spatial 3-vectors; `k ∈ {1,2,3}`
  spatial indices.
- `R̃` denotes the reverse of multivector `R`.
- `m_e`: electron mass; `c`: speed of light; `ħ = h/2π`.

## 2. Clifford algebra Cl(1,3)

**(E1) Signature.** Metric `η = diag(+1, −1, −1, −1)`:
`e_0² = +1`, `e_k² = −1`, `e_μ·e_ν = 0` for `μ ≠ ν`.
(Source: Hestenes STA as referenced throughout the spec. The spec does not
state the signature explicitly; mostly-plus Cl(3,1) would also be
"a" Clifford spacetime algebra. **[INTERPRETATION]**: Cl(1,3) is named
explicitly in the spec's type annotations, so `e_0² = +1`.)

**(E2) Basis ordering** for the 16-component array `A[..., 16]`
(fixed, load-bearing for every array in the codebase):

| index | blade | grade |
|---|---|---|
| 0 | 1 | 0 |
| 1–4 | e_0, e_1, e_2, e_3 | 1 |
| 5–10 | e_01, e_02, e_03, e_12, e_13, e_23 | 2 |
| 11–14 | e_012, e_013, e_023, e_123 | 3 |
| 15 | e_0123 | 4 |

Blade `e_{μν…}` means the ordered product `e_μ e_ν …` with `μ < ν < …`.

**(E3) Geometric product.** `(AB)_k = Σ_{i,j} T[i,j,k] A_i B_j` with the
structure tensor `T ∈ {−1,0,+1}^{16×16×16}` generated programmatically
from E1–E2 (each blade pair multiplies to ± one blade; sign from
anticommutation count and metric contractions).

**(E4) Reverse.** `R̃` reverses factor order in each blade: sign
`(−1)^{g(g−1)/2}` per grade `g` → grades (0,1,2,3,4) pick up
(+, +, −, −, +).

**(E5) Rotor.** A rotor satisfies `R R̃ = 1` (scalar). Rotor norm² :=
`⟨R R̃⟩_0`.

**(E6) Bivector exponential.** `exp(B) = Σ B^n/n!`, evaluated by scaled
Taylor series with squaring; for rotors used here `exp(B)` with
bivector `B` yields `R R̃ = 1` to numerical precision. Implementation
note (later hardening): the compact (rotation-like) invariant
component is first range-reduced modulo 2π via the exact invariant split
`B² = s + pI` (an exact identity, not a new physical formula), so large
rotation angles stay accurate; non-compact (boost) components are not
reduced and overflow for boost invariant ≳ 710 (documented in the code's
accuracy contract).

## 3. Constants & non-dimensionalization

**(E7) Compton duration** (the fundamental time unit):
`τ_c = ħ/(m_e c²) ≈ 1.288 088 67e−21 s`.

**(E8) Compton angular rate:** `ω_C = m_e c²/ħ = 1/τ_c ≈ 7.763 44e20 rad/s`.

> **Resolved (independent theory review, 2026-08-08).** The source's tabulated
> `8.0908e−21 s` and `1.2356e20` are `T_C = 2π τ_c = h/(m_e c²)` and
> `ν_C = m_e c²/h` (a 2π-different quantity, mislabeled `ω_C` there). The
> ħ-based angular forms E7–E8 satisfy `ω_C τ_c = 1` exactly. The observable
> is provably invariant to the choice:
> `Δν/ν = ΔΦ/T̃ = (∫δω̃ dt/τ_c)/(T/τ_c) = (1/T)∫δω̃ dt`: τ_c cancels
> identically. Pinned by a constants test (`ω_C·τ_c == 1`;
> `T_C = 2π τ_c`).

**(E9) Dimensionless time:** `t̃ = t/τ_c`; interrogation time
`T̃ = T_interrogation/τ_c`.

**(E10) Precision discipline.** Never accumulate absolute Compton phase
(`~ω_C T ≈ 10²⁰` rad). All integration accumulates *perturbation*
quantities only: dimensionless numbers of order the fractional shift.
fp64 everywhere; compensated (Kahan) summation in long accumulations.

## 4. Field decomposition & smoothing

**(E11) Decomposition:** `E_total(r) = E_0(r) + δE(r)`, with `E_0` a
low-order analytical baseline (uniform + linear, exactly differentiable)
fitted by least squares, and `δE` the residual handled by the smoother.

**(E12) RBF smoother:** `δE_smooth(r) = Σ_{j=1}^{K} γ_j φ(‖r − c_j‖)`
with thin-plate-spline kernels `φ(r) = r³` or `φ(r) = r² ln r`
(implementation documents which), fitted per vector component; `∇E`
obtained by exact differentiation of the fitted form (JAX autodiff of the
evaluator).

**(E13) Gradient tensor convention:** `grad_E[i, j] = ∂_i E_j` in V/m²,
positions in meters, fields in V/m.

## 5. Pivot & spin connection

**(E14a) Scalar pivot (MVP validation coupling; linear, explicit μ):**
`P(r) = 1 + δE(r)·μ / (m_e c²)`
where `μ` is an explicit user-supplied effective dipole vector (C·m).
This is the original validation form: a known, closed-form, linear-in-E coupling used
to validate the integrator and phase-accumulation pipeline (V1/V2).

> **Resolved (independent theory review; intent confirmed by project owner, 2026-08-08).**
> The tool models systematics measurable by the atomic-clock community,
> the ordinary **differential DC-Stark shift** (state-specific, quadratic
> in E), not a universal field redshift. Clock states carry no permanent
> dipole, so the physical coupling is second-order Stark via the
> transition's differential static scalar polarizability `Δα`.

**(E14b) Physical coupling (quadratic DC Stark):**
`P(r) − 1 = Δν(r)/ν₀ = −(Δα/2)·|E(r)|² / (h ν₀)` , with `ν₀` the clock
transition frequency (NOT `m_e c²`). Equivalent per-species input: a Stark
coefficient `k_S` (Hz/(V/m)²) with `P − 1 = k_S|E|²/ν₀`. Literature
polarizabilities in atomic units convert via
`α[SI] = α[a.u.] × 1.64877727436e−41 C²m²J⁻¹` (= 4πε₀a₀³, the CODATA
atomic unit of electric polarizability, pinned at full precision; an
earlier theory note's `…772` was a theory-side transcription digit-swap,
caught during development and confirmed corrected by independent theory
review).
**Bridge identity:** expanding about the E11 baseline
(`|E|² = |E₀|² + 2E₀·δE + |δE|²`), the gradient-driven part to first order
in δE is `δE·μ_eff/(hν₀)` with `μ_eff = −Δα·E₀`, i.e. E14a is exactly the
linearization of E14b about the bias field when the denominator is `hν₀`
and `μ = −Δα·E₀`. Use this to give MVP linear tests physically meaningful
μ values. Scope: for J=0→J=0 lattice clocks (Sr, Yb) the scalar `Δα` is
the whole story; ion clocks add tensor/quadrupole terms (post-MVP). The
kinematic factor `√(1−v²/c²)` in E15/E21 is the one genuinely universal
effect and is carried exactly.

**(E15) Proper-time modulation:** `dτ = P(r) dt` (at rest);
with motion, `dτ = P(r) √(1 − v²/c²) dt`. **[INTERPRETATION]**: the
rest-case pivot modulation and E21's kinematic velocity factor are
combined multiplicatively here.

**(E16) Spin connection (boost components):**
`ω_{0k}(r) = ∂_k ln P(r)` , units 1/m.

## 6. Rotor dynamics

**(E17) Evolution equation:** `dR(τ̃)/dτ̃ = −½ Ω(r(τ̃)) R(τ̃)` with `τ̃`
in Compton units and `Ω` a bivector (dimensionless rate in Compton units).

**(E18) Ω structure:** `Ω(r) = ω_boost + I ω_rot` where:
- `ω_boost = Σ_k (v^k/c) ω̃_{0k}(r) · (e_k ∧ e_0)`: the frame-tilt (boost)
  bivector picked up while moving through the gradient, with `ω̃_{0k}` the
  spin connection non-dimensionalized by the reduced Compton length
  `λ̄_C = c τ_c` (i.e. `ω̃_{0k} = λ̄_C ∂_k ln P`).
  **[INTERPRETATION]**: E16 supplies only the components `ω_{0k}`; the
  explicit contraction written here, the `v^k/c` factor and the `λ̄_C`
  scaling, follows from dimensional consistency in Compton units.
- `ω_rot`: internal-rotation angular-rate 3-vector mapped through `I`.
  **MVP sets the *perturbation* rotation rate from the pivot:** the rotor's
  internal circulation runs at `P(r) γ_v⁻¹ − 1` relative to the unperturbed
  Compton rate, in the fixed internal circulation plane `B̂_C = e_1 ∧ e_2`
  (convention). No magnetic/rotational lab effects in MVP → no additional
  `ω_rot` terms.

> **Resolved (independent theory review, 2026-08-08).** Scalar E21 is the primary observable
> pipeline; the rotor E17 with `Ω = (Pγ⁻¹ − 1)·B̂_C + ω_boost` is the
> geometric integrator and **safety net**: `ω_boost` is a timelike bivector
> governing frame orientation (Thomas/Wigner-type precession) and alters
> the `B̂_C` rotation rate only at O(ω_boost²), far below the 1e−18 floor
> for realistic gradients, but captured automatically by the rotor if a
> gradient were ever steep enough to matter. No additional `ω_rot` terms
> in MVP (no magnetic/real-rotation effects modeled). Dimensional scheme
> (`λ̄_C = c τ_c = 3.86159e−13 m`, `v^k/c` contraction) confirmed.

**(E19) Discrete step (exponential integrator, midpoint field
evaluation):** `R(t̃+Δt̃) = exp(−½ Ω(r(t̃+Δt̃/2)) Δt̃) R(t̃)`, design
order 2. **[INTERPRETATION]**: source shows left-endpoint evaluation;
midpoint chosen to actually achieve order 2, verified by the rotor
integrator's convergence test.

**(E20) Normalization:** `|⟨R R̃⟩_0 − 1| < 1e−12` maintained over ≥10⁶
steps (periodic renormalization allowed; drift per interval must be
documented).

## 7. Observable extraction

**(E21) Instantaneous fractional rate perturbation** (dimensionless):
`δω̃(r, v) = Δω_C/ω_C = P(r) √(1 − v²/c²) − 1`.

For the lattice (static-node) regime, `v = 0`: `δω̃ = P(r) − 1`.

> Note: E21 includes *kinematic* time dilation. For thermal ensembles this
> is the (real) second-order Doppler shift; it is part of the reported
> shift, not noise. Tests that target *field-gradient-only* effects use
> `v = 0` cases.

**(E22) Accumulated perturbation phase per atom** (dimensionless):
`ΔΦ_i = ∫₀^T̃ δω̃(r_i(t̃), v_i(t̃)) dt̃` (discretized with compensated
summation; equivalently `ΔΦ_i = Σ_k δω̃(t̃_k) Δt̃`).

**(E23) Per-atom and ensemble fractional frequency shift:**
`(Δν/ν₀)_i = ΔΦ_i / T̃` ; `⟨Δν/ν₀⟩ = Σ_i w_i (ΔΦ_i / T̃)` with normalized
weights `w_i` (uniform `1/M` for classical MC; quadrature weights for
lattice nodes). This equals the equivalent per-atom form
`Δν/ν₀ = (1/M) Σ ΔΦ_i/(ω_C T_interrogation)` under E7–E9
(`ω_C T = T̃` when phases are expressed in physical radians).

**(E24) Rotor↔scalar consistency:** the rotor-extracted phase (angle
advance of `R` in `B̂_C` relative to unperturbed) must equal E22 within
integrator tolerance, a standing cross-check, not two independent
physics definitions. **Acceptance criterion:** equality is required at
*first order*; disagreement at first order is a bug. Divergence at
*second order* (the O(ω_boost²) term) in a steep gradient is permitted,
and if it occurs, **the rotor is the truth** and the scalar E21 is the
approximation. Tests must be written so a steep-gradient case is not
mis-scored as a rotor bug.

## 8. Ensemble analytics

**(E25) Phase variance:** `σ_Φ² = 1/(M−1) Σ_i (ΔΦ_i − ΔΦ̄)²` (weighted
generalization for quadrature nodes; computed by Welford/two-pass, never
`E[x²]−E[x]²`).

**(E26) Coherence function:** `C(t) = (1/M) Σ_i exp(i ΔΦ_i(t))`
(weighted), with `ΔΦ_i(t)` the running accumulated phase.

**(E27) Inhomogeneous dephasing time:**
`T₂* = √2 · T_interrogation / σ_Φ` (seconds).

> **Resolved (independent theory review, 2026-08-08).** Standard inhomogeneous dephasing:
> Gaussian frequency-offset spread gives `C(t) = exp(−σ_ω²t²/2)`, 1/e at
> `t = √2/σ_ω`, with `σ_ω = σ_Φ/T`; hence E27. The source's bare
> `√2/σ_Φ` treated σ_Φ as a rate.

**(E28) Line profile:** `I(ω)` = Fourier transform of `C(t)` over the
interrogation window; reported as `(frequency offset in Hz, normalized
amplitude)`.

**(E39) Coherent (phase-resolved) rotor composition and Ramsey
visibility (v1.7.0, WP31).** Each worldline `k` carries an accumulated
perturbation phase `ΔΦ_k` (E22) and, via `exp(ΔΦ_k · B̂_C)` (E6, `B̂_C =
e₁∧e₂` per E18), a unit rotor `R_k` confined to the engine's internal-
circulation plane (`R_k R̃_k = 1`, even subalgebra): the rotor-algebra
restatement of E26's own `exp(i ΔΦ_k)` phase factor under the
identification `e₁₂ <-> i`. The ensemble's coherence object is the
POPULATION-WEIGHTED COHERENT SUM of these per-worldline rotor phase
factors:

    M = Σ_k p_k R_k

with `p_k` the ensemble's existing PROBABILITY weights (E23's own
convention: uniform `1/M` for classical Monte-Carlo, quadrature weights
for lattice motional nodes). `M` is **deliberately not a rotor**: its
modulus, projected onto the `B̂_C` plane, IS the Ramsey fringe visibility
`V`, and `1 − V` the decoherence:

    V = sqrt(⟨M̃M⟩₀-like scalar norm, B̂_C-plane projection) <= 1
    fringe phase = argument of M in the B̂_C plane

**The combiner sums the group elements linearly and never renormalizes**
(the central rule, with its reason). Two classic errors this rule
guards against:

(a) Summing the per-worldline PHASES and exponentiating the mean
    (`|exp(i·Σ p_k ΔΦ_k)|`) gives modulus `1` identically, for ANY phase
    spread: no visibility loss is representable at all, because the
    spread information is discarded before the modulus is ever taken.
(b) Renormalizing `M` back to a unit rotor (the "average of rotations
    should itself be a rotation" instinct) erases exactly the signal E39
    exists to report: a renormalized `M` has modulus `1` by construction,
    regardless of the true spread.

Both errors are encoded directly as kill tests
(`tests/test_coherent_visibility.py`): a phase-averaging combiner and a
renormalizing combiner must each be shown to disagree with the real
combiner on a spread ensemble, landing at `V = 1` where the real
combiner does not.

**Gaussian closure (the validation identity).** For Gaussian-distributed
accumulated phases, `ΔΦ_k ~ N(μ, σ_Φ²)`, the characteristic-function
identity `E[exp(i·ΔΦ)] = exp(iμ − σ_Φ²/2)` (the SAME identity E27's own
Gaussian inhomogeneous-dephasing derivation already uses) gives

    V = exp(−σ_Φ²/2)   exactly.

**Scope boundary (Gaussian motional states only, stated everywhere this
module's output reaches a report).** Valid only for GAUSSIAN-distributed
accumulated phases: thermal, coherent, and squeezed motional states,
because those are genuine positive distributions over worldlines (a
classical-ensemble sampler can draw real worldlines from them). Fock
states with `n >= 1` and cat states are NOT Gaussian-distributed
positive distributions over worldlines and are out of scope for this
worldline-ensemble representation; they are flagged as such, not
silently misreported.

**Squeezed-motional-state sampling.** The classical ensemble sampler
(`cliffordclock.ensemble.classical.sample_maxwell_boltzmann`) accepts an
optional per-axis squeezing parameter `r` (config: `ensemble.squeezing_r`,
absent by default: today's unsqueezed thermal sampling, reproduced
bitwise): the trap-frame POSITION quadrature variance is scaled by
`exp(-2r)` and the VELOCITY quadrature variance by `exp(+2r)` per axis
(the engine's convention: positive `r` squeezes position and
antisqueezes velocity, preserving the phase-space area exactly since
`exp(-r)·exp(+r) = 1`). The SAME sampled `(positions, velocities)` draw
feeds every downstream classical-ensemble observable: the pre-existing
mean-shift/E23 pipeline and the new `ramsey_visibility`/`ramsey_phase`
report fields both consume this one draw (class-`i` consistency: no
separate, independently-resampled distribution for the two).

**Implementation.** `cliffordclock.integrator.coherence.phase_to_rotor`
builds each worldline's `R_k`; `coherent_rotor_composition` computes `M`
(a plain population-weighted linear sum, no renormalization);
`ramsey_visibility_and_phase` projects `M` onto `B̂_C` and returns
`(V, phase)`. Wired into the worldline ensemble path
(`cliffordclock.pipeline.run_pipeline_full`): a run whose resolved
`integration.mode` is `"direct"` or `"worldline"` (the two modes that run
a genuine per-worldline dynamical phase accumulation, as opposed to
`"fast_path"`/`"secular"`'s closed-form expectations) reports
`ramsey_visibility`/`ramsey_phase` (`MetrologyReport`,
`REPORT_SCHEMA_VERSION` bumped to `"1.1"`), with the Gaussian-only scope
note folded into `uncertainty_notes` whenever they are populated.

## 9. Closed-form validation cases (used by the integrator and pipeline test suites)

**(V1) Uniform field, at rest:** `∇E = 0`, `v = 0` ⇒ `δE·μ` constant ⇒
`ΔΦ = (P − 1) T̃` exactly; the *gradient-driven* part of the shift is
zero; rotor-rate shift beyond this scalar baseline must satisfy
`|shift| < 1e−19` (spec Test Contract 1).

**(V2) Constant gradient, static atom at r₀:**
`δE(r) = G·(r − r_ref)` ⇒
`ΔΦ = (P(r₀) − 1) T̃ = [(G·(r₀−r_ref))·μ/(m_e c²)] · T̃`.
Full closed form; the rotor integrator must match to 1e−14 relative.

**(V3) Constant Ω rotor:** `R(τ̃) = exp(−½ Ω τ̃) R(0)` exactly, used for
integrator convergence-order measurement.

**(V4) Harmonic trap, classical atom, linear-gradient field:** with
`r(t)` sinusoidal, `ΔΦ = ∫(P(r(t̃))γ⁻¹ − 1)dt̃` integrates in closed form
(time-average of the sinusoid + second-order Doppler `−⟨v²⟩/2c²` term);
used in the pipeline's Case C tolerance design (`tests/test_e2e.py`).

## 10. Units at API boundaries

SI in, SI out; dimensionless only internally: positions m, fields V/m,
gradients V/m², time s (converted to Compton units internally per E9),
temperature µK at sampler APIs, output shift dimensionless, T₂* seconds,
line-profile frequency Hz. Every public docstring states units and shapes.

## 11. Sign-off record (all items resolved 2026-08-08)

Reviewed and confirmed by an independent theory review of this
document's equations, 2026-08-08; proceed
authorized by project owner. (The full review record is an internal
document not included in this distribution; the summary below is
self-contained.)

1. **ħ vs h** (E7–E8): CONFIRMED ħ-based; the reviewed values identified as
   `T_C = 2πτ_c` and `ν_C`; observable invariant (§3 note).
2. **μ** (E14): RESOLVED: MVP linear form E14a confirmed as validation
   coupling; physical coupling specified as quadratic DC-Stark
   E14b with `hν₀` denominator and `μ_eff = −Δα·E₀` bridge. Not yet
   implemented (recorded caveat).
3. **Ω construction / rotor role** (E18, E21, E24): CONFIRMED: scalar E21
   primary, rotor is geometric integrator + safety net, `ω_boost` retained,
   E24 first-order/second-order acceptance criterion added.
4. **T₂*** (E27): CONFIRMED `T₂* = √2·T_interrogation/σ_Φ`.
5. **Signature** (E1): CONFIRMED `η = (+,−,−,−)`, Cl(1,3); only internal
   consistency of the structure tensor with E1–E2 is required.
6. **Midpoint stepper** (E19): CONFIRMED: exponential midpoint, order 2,
   exactly rotor-group-preserving.

## 12. Fast-path additions (v1.1.0)

Motivation: the observable integrand `δω̃(r(t̃), v)` (E21) contains **no
Compton-frequency content**: the fast carrier was removed analytically
(E10 perturbation discipline). Its time variation is bounded by the
atom's motion through the field, i.e. trap dynamics (µs–ms). The
following formalize the fast evaluation paths that exploit this, and the
step-size rule for direct integration. None of these change the physics
of E14–E28; E29 is an exact corollary, E30–E31 are controlled numerical
approximations with stated validity bounds.

**(E29) Lattice static-state fast path (exact corollary of E21–E23).**
For a time-independent field and a stationary motional state `|ψ⟩`
represented by quadrature nodes/weights `(r_q, w_q)`, each node has
`v = 0` and constant `δω̃_q = δω̃(r_q, 0)`, so E22 integrates trivially:
`ΔΦ_q = δω̃_q · T̃` and `⟨Δν/ν₀⟩ = Σ_q w_q δω̃_q`. No time integration;
exact for any interrogation time T up to quadrature accuracy. This is
the default execution path for lattice-regime runs; the worldline
integrator (E17–E19) remains available as a cross-check mode and must
agree exactly (static nodes, any step count). **Scope:** because
every node has `v = 0`, E29 returns the **field (Stark) shift only**:
it omits the motional second-order Doppler `−⟨v²⟩/2c²` carried by the
lattice state's velocity spread, a real, separately-budgeted clock
systematic. Do not read E29's `⟨Δν/ν₀⟩` as the *total* clock shift.
(E30, by contrast, integrates the full `δω̃(r, v)`: Stark + Doppler. A
lattice motional Doppler would require a phase-space representation with
node velocities, not E29.)

**(E30) Secular averaging (classical periodic motion).** For a static
field and a periodic classical orbit `r(t̃)` with period `T̃_orb`:
`ΔΦ = ⟨δω̃⟩_orb · T̃ + ε`, where
`⟨δω̃⟩_orb = (1/T̃_orb) ∮ δω̃(r(t̃), v(t̃)) dt̃` (one-orbit line
integral, computed with the direct integrator), and the partial-orbit
remainder obeys `|ε| ≤ T̃_orb · max_t|δω̃ − ⟨δω̃⟩_orb|`. Validity:
static field, periodic (e.g. harmonic-trap) motion, `T ≫ T_orb`. Not
valid for drifting/chaotic trajectories or time-dependent fields; use
direct integration there.

**(E31) Step-size rule for direct integration (timescale separation).**
The exponential-midpoint stepper's (E19) local error per step scales as
`O(|Ω̈| dτ̃³)` where `Ω̈` is set by trap dynamics, NOT the Compton
scale; and `exp` of a bivector lands exactly on the rotor group at any
`dτ̃`, so norm preservation (E20) is step-size independent. Sufficient
practical rule: `dτ̃ ≤ T̃_orb / N_res` with `N_res = 100` as the default
resolution (points per trap period); the step-size accuracy study empirically
maps error vs `dτ̃` against the V4 closed form and may tighten or relax
`N_res` with documented evidence. Compton-scale stepping (`dτ̃ ~ 1`) is
a validation mode only, never a physical requirement.

## 13. Blackbody-radiation shift (v1.2.0, WP20)

Motivation (project owner, 2026-08-11): the BBR shift is the dominant
systematic in real optical-lattice clocks, so it enters the same scalar
pivot `P(r)` as the DC-Stark term (E14b), uniform-T only in this MVP.
Provenance for every coefficient and citation below:
Lisdat et al., PRR 3, L042036 (2021); Aeppli et al., PRL 133, 023401
(2024).

**(E32) BBR pivot term.** For radiation temperature `T` (kelvin, uniform
in the MVP), `T₀ = 300 K`:

    (P−1)_BBR = [Δν_stat·(T/T₀)⁴ + Δν_dyn(T)] / ν₀

with `Δν_stat = −(Δα(0)/2h)·⟨E²⟩_{T₀}` and `⟨E²⟩_T = (4σ_SB/ε₀c)·T⁴` (the
full-energy-density convention, `u = ε₀⟨E²⟩`; `√⟨E²⟩ = 831.9 V/m` at
300 K; the equipartition convention gives 588 V/m and is WRONG, per the
Middelmann round-trip closure check, dossier §1). **No leading minus: the
sign lives inside `Δν_stat < 0`, exactly as E14b carries it**
(`P−1 = Δν/ν₀`): an earlier draft double-negated this and gave the wrong
sign; the G7 theory sign-off's mandatory correction
(the project's theory sign-off record (G7), A1). Mandatory regression:
`(P−1)_BBR(Sr87, 300 K) < 0` and `≈ −5.3e-15`
(`tests/test_bbr_pivot.py::test_bbr_pivot_sign_regression_sr87_300k`).

`Δν_dyn(T)` is a per-species even power series in `(T/T₀)` whose
coefficients are FITS to the exact Planck-weighted integral (Lisdat et
al., PR Research 3, L042036 (2021) Eq. 6–7), NOT a Taylor expansion: the
Taylor series in `1/y` has zero convergence radius for Sr's dominant
transition, and the historical 3-term truncation cost Sr clocks 4.7e-18
(dossier §2). Registry (`cliffordclock.ensemble.species.BbrCoefficients`,
consumed by `cliffordclock.integrator.omega.bbr_pivot_perturbation`):

- **Sr-87:** static `Δν_stat(300 K) = −2.13023(6) Hz` (Middelmann et al.,
  PRL 109, 263004 (2012), unchanged across every dynamic-term revision);
  dynamic: the PTB-2025 rescaled polynomial (arXiv:2507.14030)
  `{6: −0.13216, 8: −0.01231, 10: −0.00858} Hz`, anchored to Aeppli et
  al.'s (arXiv:2403.10664) `±0.33 mHz` at 300 K, shape from Lisdat 2021,
  cross-verified to 1e-19 against the full calculation for `T ≤ 300 K`.
  Shape-vs-anchor reasoning: the temperature *shape* comes from the fixed
  atomic transition spectrum (Lisdat), the overall *scale* from the
  ³D₁-state lifetime (which Aeppli remeasured more precisely); rescaling
  the published shape to the newer anchor is the physically correct
  operation, not an ad hoc reconciliation, and is exactly what the source
  authors themselves published. Truncation beyond `T¹⁰` is covered by the
  fit's own 1e-19 cross-verification, not a separate coefficient-ratio
  bound (Sr's η-series is not monotone-convergent, unlike Yb's below; see
  the registry docstring).
- **Yb-171:** static `Δν_stat(300 K) = −1.2545(10) Hz` and
  `ν_dyn,6 = −22.17(34) mHz` (Hassan et al., arXiv:2506.05304 (2025),
  direct measurement, unaffected by Sr's Taylor-divergence problem);
  `ν_dyn,8 = −0.744(20) mHz`, derived from Beloy et al.'s (PRL 113,
  260801 (2014)) `η₂ = 0.000593(16)` via `ν_dyn,8 = Δν_stat·η₂`. Truncation
  bound via the coefficient *ratio* (not a dataset order-index estimate):
  `|ν_dyn,10| ≲ (ν_dyn,8/ν_dyn,6)·|ν_dyn,8| ≈ 0.034 × 0.744 mHz ≈ 5e-20`
  fractional (Yb's η-series is monotone-convergent, so the next term is
  suppressed at least as fast), safely below the 1e-19 floor and not
  included as a registry coefficient.
- Validity window `50–350 K` for both species (the fit range); hard
  `PipelineConfigError` outside it (`environment.radiation_temperature_K`,
  `cliffordclock.ensemble.species.BBR_VALIDITY_MIN_K`/`_MAX_K`): silently
  extrapolating a fit past its support is exactly how wrong clock
  corrections get made. Split confidence: `T ≤ 300 K`
  (`BBR_CROSS_VERIFIED_MAX_K`) carries the full PTB↔JILA 1e-19-class
  cross-verification; `300 < T ≤ 350 K` is in-fit-range but beyond that
  cross-verification statement, and the report carries an explicit note
  in that band.

The BBR field's own fluctuations enter only through `⟨E²⟩` (the
deterministic self-averaged expectation over ~1e14 independent bath
cycles per interrogation), no stochastic term, matching every published
evaluation (dossier §4). M1/E2 multipole contributions (`≈6e-20` each,
Porsev & Derevianko PRA 74, 020502(R) (2006) with its 2012 erratum, PRA
86, 029904, via Lisdat 2021) are modeled out and carried as an explicit
budget line in the pipeline report, not silently omitted: the scalar-pivot
E1 model is bounded, not implicitly claimed complete.

**Uncertainty.** Registry coefficient uncertainties propagate into the
reported BBR uncertainty (`cliffordclock.integrator.omega.bbr_pivot_uncertainty`),
combined in quadrature, at Sr87/300 K: static `≈1.4e-19`, dynamic
`≈7.7e-19` (dominated by the Aeppli anchor), combined `≈8e-19`. This is
**arithmetic-reproduction fidelity** (does the code evaluate the formula
correctly), **never BBR accuracy**: the physical BBR uncertainty is
dominated by these registry coefficients, not by float64 rounding, and
every report note or docs passage stating a "1e-19"-class BBR number must
say so explicitly. An optional `environment.radiation_temperature_uncertainty_K`
propagates via the *exact* polynomial derivative `∂Δν/∂T`
(leading behavior `≈4Δν/T`; e.g. `σ_T = 4 mK ⇒ ≈3e-19`, above the 1e-19
floor); when omitted, the report states its BBR uncertainty is
conditional on exact `T` rather than silently claiming exactness.

**(E33) Scalar pivot composition.** Independent scalar (state-energy)
perturbations compose additively in `(P−1)`:

    P(r) − 1 = (P−1)_stark(r) + (P−1)_BBR + …

Exact at working order: the DC×BBR **second-order** cross term
`2 E_DC·⟨E_BBR⟩` time-averages to zero exactly (`⟨E_BBR⟩ = 0` as a vector,
isotropic thermal field). The **4th-order (hyperpolarizability) cross
term** `6β·E_DC²⟨E_BBR²⟩` is also omitted by the linear pivot sum (it is
not the sum of the pure-DC and pure-BBR hyperpolarizability shifts),
bounded for a Gaussian bath at `≤3e-4 ×` (BBR hyperpolarizability shift
`~1e-18`) `≈1e-22`, negligible versus 1e-19. **Scope note: the engine
models the dipole polarizability `α` only, never the hyperpolarizability
`β`**: both the pure and cross `β` terms are out of scope by
construction, and the bound above shows this scope boundary is safe. The
multiplicative `(P−1)_stark·(P−1)_BBR ~ 1e-33` product correction is
likewise neglected. The spin connection generalizes by linearity:
`∂_k ln P` picks up each term's gradient (zero for BBR in this MVP,
uniform `T`, but the composition is implemented generally, via a
keyword-only `bbr_pivot_perturbation` parameter threaded through
`cliffordclock.integrator.omega.pivot_perturbation_stark`/
`spin_connection_stark`/`scalar_rate_perturbation_stark`/
`build_omega_stark`, so a future `T(r)` map drops in without a signature
change to the pivot-composition call sites). Composed into every
evaluation mode: `fast_path`/`secular`/classical `direct` (batched and
streaming) via `cliffordclock.pipeline._make_stark_rate_fn`'s shared
`rate_fn`, and the rotor worldline via
`cliffordclock.pipeline._stark_rotor_ensemble`'s `build_omega_stark` calls.

**G7 sign-off record (2026-08-11, the project's theory sign-off record (G7)):**
Part A (formalism): A1 E32 structure confirmed, sign corrected (above,
with the mandatory regression test); A2 E33 additivity confirmed, 4th-order
hyperpolarizability bound added; A3 rotor scope confirmed (J=0→J=0 scalar
polarizability only, isotropic-bath averaging, no bivector/anisotropic
channel), M1/E2 promoted from footnote to an explicit budget line. Part B
(clock-literature ratifications): B1 the full-energy-density `⟨E²⟩`
convention ratified (Middelmann round-trip closure); B2 Sr-87 coefficients
ratified with citations embedded and the shape-vs-anchor sentence recorded
(above); B3 Yb-171 coefficients ratified, arithmetic independently
re-verified, truncation bound switched to the coefficient-ratio argument;
B4 the 50–350 K validity window ratified with the 300 K cross-verification
split; B5 the JILA 2024 benchmark-case label ("arithmetic reproduction of
a published standard-formula evaluation") ratified: that benchmark case
itself is a separate, later WP step with its own review, not part of this
sign-off's implementation scope.

**(E37) Multi-surface thermal environment (v1.5.0, WP29 Tier 1).** Motivation
(the field-deployment problem class: outside a shielded laboratory, the
blackbody radiation distribution across a clock's visible surfaces is a
larger and less controlled systematic than the DC electric field, a point
practitioners in the community emphasize): E32's single radiation temperature `T` is
the `⟨T⁴⟩`-matched effective temperature of the atoms' real surroundings, and
because the dynamic term scales as `T⁶` through `T¹⁰`, a `T` chosen to match
the static moment does not in general reproduce the higher moments of a
non-uniform environment. The project's internal BBR thermal-environment
dossier quantifies the resulting mismatch against the Sr-87 registry
coefficients: it
crosses `1e-18` at an 11 K spread across two surfaces and `1e-17` by 35 K, well
within what an uncontrolled or partially thermally controlled enclosure can
show. E37 replaces the single `T` with an explicit multi-surface description
and evaluates E32's static and dynamic terms directly against that
description's per-moment sums, so the mismatch above is computed exactly
instead of approximated away.

*Surfaces and weights.* The enclosure is described as `N` surfaces, each
carrying an effective solid-angle fraction `w_i` (`Omega_i/4pi`), a
temperature `T_i`, an optional temperature uncertainty `sigma_{T_i}`, and an
optional emissivity `epsilon_i`. The fractions are an input, not a computed
quantity: v1 takes `w_i` as supplied by the lab (from geometry, an FEA model,
or a ray-traced exchange-factor calculation, per the dossier's part A survey
of how every published evaluation already does this reduction), with no CAD
import in this tier. `sum_i w_i = 1` is enforced at parse time and again at
evaluation time, both to a `1e-9` absolute tolerance
(`cliffordclock.integrator.omega.BBR_ENVIRONMENT_WEIGHT_TOLERANCE`); an
unnormalized set of fractions is a configuration error, not something the
engine silently renormalizes.

*Emissivity correction: one enclosure, one or more apertures.* PTB's
transportable-clock paper (Nosske et al., arXiv:2507.14030) models the
atoms as sitting inside a single reflective enclosure of interior
emissivity `epsilon`, pierced by one or more apertures leaking in a
different temperature; the enclosure's own reflections give that leaked-in
radiation more chances to reach the atoms than its raw geometric solid
angle alone would suggest. Their published closed form, for one aperture
of raw fraction `w = Omega/4pi`, is

    Omega_eff/4pi = 1 / [1 + (4pi/Omega - 1) * epsilon]

equivalently `w_eff = w / (w + (1 - w) * epsilon)`. `epsilon = 1` (a
perfectly absorbing interior) reduces this to `w_eff = w`, the naive
geometric weighting; as `epsilon` drops toward 0 (a more reflective
interior), `w_eff` grows past `w`.

E37 carries this topology directly: at most one surface in an environment
may set an `epsilon_i`, and that surface is the enclosure; every other
surface is a direct-view aperture. Writing `W` for the apertures' combined
raw fraction (`sum` of their `w_i`, jointly forming the single lumped
aperture PTB's formula treats), each aperture's effective fraction is

    w_i_eff = w_i / (W + (1 - W) * epsilon)

PTB's own single-aperture formula with `w` replaced by the combined `W`,
then split across the individual apertures in proportion to their own raw
share of `W`; summing every `w_i_eff` over the apertures reproduces PTB's
combined effective fraction exactly. For a single aperture (`W = w_1`)
this is PTB's formula unchanged, character for character: an
illustrative round-number check of this project's own choosing, `w=0.1`,
`epsilon=0.5`, gives `w_eff = 0.1/0.55 = 0.181818...` from that formula
(the paper itself publishes only its apparatus values, aperture fraction
`1.17(3)e-3` with `epsilon = 0.926(43)`). The enclosure then gets whatever effective fraction is
left, `1 - sum_i w_i_eff` (here `0.818182...`), never a value computed
from its own raw `weight`: PTB's derivation is a two-temperature mixture
(the enclosure, the leaked-in aperture temperature), so the two effective
fractions are complementary by construction, not independently
renormalized shares of every surface's weight. An environment with no
`epsilon_i` set on any surface returns every raw `w_i` unchanged.
**Scope boundary:** multi-reflector radiosity (more than one partially-
reflective enclosure surface, each contributing its own reflected share)
is out of scope for this tier and is rejected with a configuration error
naming the boundary; a future tier that needs it is genuine future work,
not an oversight here. Implemented in
`cliffordclock.integrator.omega._bbr_effective_weights`.

*Per-moment sums.* Write `T0 = BBR_REFERENCE_TEMPERATURE_K = 300 K` and let
`M_n = sum_i w_eff_i * (T_i/T0)^n` be the `n`-th weighted moment over the
(effective-weight-corrected) surfaces. E32's static and dynamic terms are
evaluated directly against these moments in place of a single `(T/T0)^n`
power:

    (P-1)_BBR = [Delta_nu_stat * M_4 + sum_n c_n * M_n] / nu_0

with `c_n` the same per-species `dyn_coeffs_hz` registry entries E32 already
uses. Equivalently, each moment defines its own per-moment effective
temperature `T_eff,n = T0 * M_n^(1/n)`; for a non-uniform environment
`T_eff,4` (the static-term match) and `T_eff,6`/`T_eff,8`/`T_eff,10` (the
dynamic-term matches) are generally different numbers, and that divergence is
exactly the mismatch the dossier's part C quantifies. The pipeline report
exposes all of a run's `T_eff,n` values (one per registry-dynamic power plus
`n=4`) so a user can see this divergence directly instead of inferring it
from the shift alone. Implemented in
`cliffordclock.integrator.omega.bbr_environment_pivot_perturbation` (the
scalar shift) and `bbr_environment_effective_temperatures` (the `T_eff,n`
values), both consuming
`cliffordclock.integrator.omega._bbr_weighted_moments`.

*Exact reduction to E32.* A uniform environment (one surface, `w_1 = 1`, no
emissivity) makes every `M_n` equal to `(T_1/T0)^n` exactly, so E37 reduces to
E32 term for term, not just to numerical agreement: `E32`'s scalar
`bbr_pivot_perturbation(T, species)` is implemented as the single-surface call
`bbr_environment_pivot_perturbation((RadiationSurface(weight=1.0,
temperature_k=T, ...),), species)`, so the two paths share the same
coefficient-evaluation code and agree bit for bit, an exact reproduction
(`tests/test_bbr_environment.py`'s reduction test).

*Validity window.* The `50-350 K` fit-range window E32 already states
(`BBR_VALIDITY_MIN_K`/`BBR_VALIDITY_MAX_K`, per-species
`BbrCoefficients.validity_min_k`/`validity_max_k`) applies to every surface's
`T_i` individually: a single out-of-window surface is rejected with the same
class of error the single-temperature path raises (`PipelineConfigError` at
the pipeline's config-parse boundary; `ValueError` from
`cliffordclock.integrator.omega` when the environment functions are called
directly), for the same reason E32 states it: silently extrapolating the fit
past its published support is exactly how wrong clock corrections get made.

*Uncertainty.* Per-surface temperature uncertainties `sigma_{T_i}` propagate
through the same analytic-derivative pattern `bbr_pivot_uncertainty` already
uses for a single `T` (CONVENTIONS.md section 13's uncertainty note above):
writing `a_i = w_eff_i * d(Delta_nu_hz)/dT` evaluated at `T_i` (the same
polynomial derivative E32's uncertainty note gives, scaled by the surface's
own effective weight), two combination modes are supported.
**Independent** (the default): the surfaces' temperature errors are treated
as uncorrelated and combined in quadrature,
`sigma_T = sqrt(sum_i (a_i * sigma_{T_i})^2)`. **Correlated**
(`correlated=true`): the surfaces' temperature errors are treated as moving
together (a single shared calibration-chain error affecting every sensor
coherently, the motivation Aeppli's 2025 JILA thesis gives for combining its
own four correlated temperature estimates by linear pooling instead of
independent averaging, per the dossier's part A), so the per-surface terms
are summed linearly before taking the magnitude,
`sigma_T = |sum_i (a_i * sigma_{T_i})|`. For same-sign partials (the ordinary
case: every registry coefficient here is negative, so every `a_i` is
negative) the correlated combination is never smaller than the independent
one, and strictly larger whenever more than one surface carries a nonzero
uncertainty, since an L1 norm is never smaller than the corresponding L2
norm. Either `sigma_T` is combined in quadrature with the same
coefficient-uncertainty term E32's uncertainty note already computes
(`arithmetic-reproduction fidelity`, not an independent BBR-accuracy claim,
the same caveat as the single-`T` path). Implemented in
`cliffordclock.integrator.omega.bbr_environment_pivot_uncertainty`.

**Scope boundary.** E37 is position-independent within the atom cloud in
this tier: every atom sees the identical enclosure description, exactly as
E32's single `T` is spatially uniform across the cloud today. A per-atom
solid-angle map (the atoms' own extent changing which fraction of each
surface they see, the dossier's part D/E "per-atom effective moments"
product) is future work, not built here; E37's spin-connection contribution
is therefore exactly zero for the same reason E32's is (CONVENTIONS.md
section 13's composition note above), and the same keyword-only
`bbr_pivot_perturbation` composition point E33 already threads through every
evaluation mode carries E37's resolved scalar with no further signature
change. Config surface: `environment.radiation_environment` (a list of
per-surface `name`/`weight`/`temperature_K`/`temperature_uncertainty_K`/
`emissivity` entries plus a `correlated` flag), mutually exclusive with
`environment.radiation_temperature_K`
(`cliffordclock.pipeline.EnvironmentConfig`). The surfaces list is given
either inline as above or, equivalently, via
`environment.radiation_environment.surfaces_file`, a path to a plain-text
surfaces table (WP29 Tier 1 Part 1,
`cliffordclock.pipeline._load_radiation_surfaces_file`, docs/cli.md's
"Surfaces table file format" section); the two forms are mutually
exclusive with each other and produce the identical parsed `surfaces`
tuple, so this section's formalism is unchanged by which form a config
uses.

## 14. Ion-clock electric-quadrupole shift (v1.3.0, WP21 Tier 2)

Motivation (owner, 2026-08-11: "what would it take to add [ions]?"): a
D/F-state ion clock's electric-quadrupole moment `Theta` couples to the
electric-field gradient, giving a first-order (in the gradient) shift for
J>=1 upper clock states, distinct from the second-order (in the field)
DC-Stark/BBR terms E14b/E32. Provenance for the coefficients and citations
below: Roos et al., Nature 443, 316 (2006), quant-ph/0701215v1
(Eq. 1/Fig. 4a); Itano, J. Res. NIST 105, 829 (2000), Eq. 46, and Itano,
Phys. Rev. A 73, 022510 (2006); Barwood et al., PRL 93, 133001 (2004);
the per-entry citations in this section; and the project's G8 theory
sign-off record.

**(E34) Quadrupole level shift.** Canonical axially-symmetric fine-
structure form (Itano, J. Res. NIST 105, 829 (2000), Eq. 46, the
hyperfine/Euler-angle generalization the code's general form below
instantiates; Roos et al., quant-ph/0701215v1 (published as Nature 443,
316 (2006)), Eq. 1, the fine-structure-only form the code adopts
verbatim, both PRIMARY TEXT, read directly per G8 gate edit 1a):

    Delta_E_Q(J, m_J) = (Theta_SI(J)/4) * [J(J+1) - 3*m_J^2] / [J(2J-1)]
                         * [(3*cos^2(beta) - 1) - eps*sin^2(beta)*cos(2*alpha)]
                         * (dE_z/dz)

with `beta`/`alpha` the polar/azimuthal angle of the quantization
(B-field) axis in the field gradient's own principal-axis frame, `eps`
the gradient's asymmetry parameter (0 for an axially symmetric gradient),
`dE_z/dz` the gradient's principal-axis magnitude (Roos's/Itano's `A`,
`dE_z/dz = 4*A`, derived below), and `Theta_SI(J) = Theta_au(J) *
EA0_SQUARED_SI` (no separate factor of `e`; **the unit-conversion pin,
G8 gate edit 2**: `Theta` is tabulated in atomic units "= e*a0^2", so
`Theta_SI` already carries one factor of `e` through
`EA0_SQUARED_SI = e*a0^2`
(`cliffordclock.ensemble.species.EA0_SQUARED_SI`, CODATA 2022
`e = 1.602176634e-19 C`, `a0 = 5.29177210544e-11 m`, both already pinned
in `cliffordclock.constants`; computed, not hand-transcribed, at
`4.4865515185255e-40 C*m^2`, matching the gate's quoted `4.4866e-40` to
its stated precision), writing an *additional* leading `e*Theta(J)`
(as an earlier draft equation did) would double-count it, exactly the G4
`ALPHA_AU_TO_SI` digit-swap lesson's unit-pin discipline applied here).

**Primary-text transcription (G8 gate edit 1a).** Roos et al.'s Eq. 1, as
read directly from the owner-supplied preprint (quant-ph/0701215v1),
p. 6:

    hbar*Delta_nu = (1/4) * (dE_z/dz) * Theta(D,j) * [j(j+1) - 3*m_j^2]
                    / [j(2j-1)] * (3*cos^2(beta) - 1)

(their `Delta_nu` an angular-frequency shift, i.e. `hbar*Delta_nu` is the
energy shift `Delta_E_Q`; asymmetric generalization on the same page:
`(3*cos^2(beta)-1) - eps*sin^2(beta)*cos(2*alpha)`), with the **m_J-factor
ordering `[J(J+1) - 3*m_J^2]`** (opposite in sign to an earlier WP21
draft's `[3*m_J^2 - J(J+1)]`) and **no leading minus sign**. This is
adopted verbatim as E34's leading form. Itano's Eq. 46 (PMC4877145,
PRIMARY TEXT, hyperfine `F,M_F` form, read directly):

    <gJFM_F|H_Q|gJFM_F> = -2*[3*M_F^2 - F(F+1)] * A * <gJF||Theta(2)||gJF>
                            * sqrt[(2F+3)(2F+2)(2F+1) / (2F(2F-1))]
                            * [(3*cos^2(beta)-1) - eps*sin^2(beta)*(cos^2(alpha)-sin^2(alpha))]

(the asymmetry bracket's `(cos^2(alpha) - sin^2(alpha))` is the
double-angle identity `cos(2*alpha)`; Dube 2005's Eq. 1 quotes the same
Itano bracket in the same squared form. An earlier transcription here
wrote `cos(2a)-sin(2a)`, a misreading with no code impact: the
implementation contracts the full gradient tensor and never uses this
trigonometric decomposition.)

**Reconciliation (required by G8, shown not asserted).** Distributing
Itano's leading `-2[3*M_F^2 - F(F+1)]` gives `+2[F(F+1) - 3*M_F^2]`:
the SAME sign as Roos's `[J(J+1) - 3*m_J^2]` factor (for `I=0`, `F=J`,
`M_F=m_J`). The two forms' remaining prefactors (Itano's reduced
hyperfine matrix element `<gJF||Theta(2)||gJF>` times the `F`-dependent
normalization square root, vs. Roos's directly-tabulated stretched-state
`Theta(D,j)`) differ by the Wigner-Eckart reduced-matrix-element
normalization relating the two conventions: a magnitude factor this
project does not independently re-derive (it is not needed: Roos's Eq. 1
is adopted directly, verbatim, as the numerical formula; Itano's Eq. 46
is cited only to confirm the *sign* reconciles, which it does). Itano's
field-gradient parameter `A` is defined via his Eq. 31's potential
`Phi(x',y',z') = A*[(x'^2+y'^2-2z'^2) + eps*(x'^2-y'^2)]` in the
gradient's own principal-axis frame (his Eqs. 32-34: `grad_E_0^(2)' =
-2A`); since `E = -grad(Phi)`, `dE_z'/dz' = -d^2(Phi)/dz'^2 = 4*A`
exactly (a two-line derivation this project performs itself, since the
paper states the tensor-component relation `grad_E_0^(2)' = -2A` but not
the Cartesian `A = (dE_z/dz)/4` step explicitly); hence E34's `dE_z/dz`
above (Roos's own gradient-magnitude variable) is Itano's `4*A`.

**Coordinate-free reduction (this project's implementation form).**
Substituting `dE_z/dz = 4*A` and expanding the bracket's `beta`/`alpha`/
`eps` dependence against a diagonalized traceless-symmetric gradient
tensor with principal-frame eigenvalues `(-2*A*(1+eps), -2*A*(1-eps),
4*A)` (direct consequence of the Eq. 31 potential above) shows, by
explicit rotation of the quantization axis unit vector `n_hat` by angles
`(beta, alpha)` into that frame, that

    n_hat^T . G . n_hat = 2*A * [(3*cos^2(beta)-1) - eps*sin^2(beta)*cos(2*alpha)]

where `G` is the gradient tensor in the LAB frame (any frame, the
identity is coordinate-independent), i.e. the entire angular/asymmetry
bracket of E34 is exactly `n_hat^T . G . n_hat / (2*A)`. Substituting into
E34 (`(dE_z/dz) = 4*A`) gives the coordinate-free, fully general (any
`eps`, no principal-axis diagonalization needed) equivalent form this
project implements directly against the field-gradient tensor the
smoother already delivers (E13):

    Delta_E_Q(J, m_J) = (Theta_SI(J)/2) * [J(J+1) - 3*m_J^2] / [J(2J-1)]
                         * (n_hat^T . G(r) . n_hat)

with `G(r)` the traceless symmetric part of `grad_E[i,j] = d_i E_j` (E13)
at the ion's position and `n_hat` the quantization-axis unit vector. This
is algebraically identical to the Itano/Roos axial-plus-asymmetric form
above, via the coordinate-free reduction just derived, and is what
`cliffordclock.integrator.omega.quadrupole_pivot_perturbation` evaluates;
`tests/test_quadrupole_pivot.py` cross-checks the two forms
numerically at several `(beta, alpha, eps)` against each other as a
standing consistency test, not just documentation.

**Sign discipline (G8 gate edit 1, status).** The leading sign above is
Roos Eq. 1 as directly transcribed from primary text (owner-supplied
preprint, read in full), the strongest verification standard the gate
requires ("verify... against primary Itano 2000 Eq. 46"; Roos's Eq. 1 is
now the second, independently-confirming primary source, per the G8 gate
mid-build update). Two regressions are pinned as a result
(`tests/test_quadrupole_pivot.py`):
- **Convention-free m_J-ratio (structure only, no sign dependence):**
  `Delta_E_Q(J=5/2, m_J=5/2) / Delta_E_Q(J=5/2, m_J=1/2) = -1.25` for a
  D_5/2 state at fixed `(beta, gradient)`: `[J(J+1)-3*(5/2)^2]/[J(2J-1)]
  = -1` and `[J(J+1)-3*(1/2)^2]/[J(2J-1)] = 0.8`, ratio `-1/0.8 = -1.25`
  (sign-convention-independent: flipping the overall formula sign flips
  both numerator and denominator, leaving the ratio unchanged).
- **Yb+ F_7/2 negative-Theta anchor:** `Theta(Yb+, F_7/2) = -0.041(5)`
  (Huntemann et al., PRL 108, 090801 (2012), PRIMARY, dossier §3) is
  negative while every D-state `Theta` in the registry is positive; the
  implementation must produce an OPPOSITE-sign shift for the F_7/2 state
  vs. a positive-Theta D-state under an otherwise identical
  `(gradient, m_J, axis)`: a real regression on whether the code
  respects the registry's `Theta` sign (catches, e.g., an accidental
  `abs(Theta)`), not a full independent sign derivation.

**AMBIGUITY (flagged, not fabricated, per the gate's explicit fallback
clause):** the gate's preferred *absolute* anchor is a measured
`(gradient, state, shift-sign)` triple from Dube et al., PRL 95, 033001
(2005) or Barwood et al., PRL 93, 133001 (2004). At build time both
were inaccessible to the builder (APS subscription-only, no arXiv/ar5iv
preprint); the owner has since supplied Dube 2005 in primary text, and
its extraction (ion-clock dossier section 7) CONFIRMS the E34 sign form
as a third independent primary source and adds the magic-m_J^2
intercept pin (m_J^2 = J(J+1)/3, now a shipped regression test). One
binding caution from that extraction: Dube's measured angle-scan slope
is ~95% micromotion-induced tensor Stark and only ~5% quadrupole (their
own model), so it must never be cited as a pure quadrupole absolute
anchor without that attribution. The G8 gate mid-build update supplied a THIRD candidate anchor,
Roos et al.'s Fig. 3a (an angle scan of a two-ion entangled state's
measured shift, swinging from  ~+70 Hz through zero to ~-40 Hz following
`(3*cos^2(beta)-1)` at a stated positive applied `dE_z/dz`) and Fig. 4a
(a measured slope against a mechanically-calibrated gradient). This
builder did NOT independently re-derive that two-ion entangled-state
sign chain (`Theta = (5/12)*h*a` for the correlated state `Psi_1`
involves two-ion coupled-gradient and entanglement factors beyond a
single-particle `m_J`-factor lookup; reproducing it correctly is the
scope of the separate, later Roos/Barwood benchmark WP the gate
explicitly excludes from this build). E34's sign is therefore pinned
directly to Roos's own primary-text Eq. 1 (transcribed above, satisfying
the gate's "verify against primary text" requirement) plus the two
structure-only regressions above. STATUS UPDATE (2026-08-11): the
Roos-slope benchmark has since shipped
(`benchmarks/run_roos_quadrupole_slope.py`, its own review PASS) and
recovers the full two-ion 24/5 factor chain from the engine's own m_J
factors against the measured Fig. 4a slope; it deliberately reports the
slope magnitude and records the electrode-polarity absolute sign as a
standing documented open point (its review ruled the deferral
legitimate: Fig. 4a publishes only the positive gradient magnitude,
and the polarity convention is not quotable from the primary text).

**(E35) Quadrupole pivot composition.** For a FIXED `(J, m_J,
quantization axis)`, `Delta_E_Q/(h*nu_0)` is a single per-point scalar
and composes additively into `(P-1)` exactly like the DC-Stark and BBR
terms (E33):

    P(r) - 1 = (P-1)_stark(r) + (P-1)_BBR + (P-1)_Q(r) + ...

**No cross term (G8 A2, confirmed).** The quadrupole shift is first
order in the gradient (a permanent moment of the D/F state); DC-Stark and
BBR are second order in the field (via `Delta_alpha`), different field
quantities (`E` vs `grad_E`) and different orders, so no problematic
product term exists at this project's working precision (same argument
E33 already gives for the DC-Stark/BBR cross term).

**Three-orthogonal-orientation averaging identity (G8 A2, exact,
gradient-orientation-independent, provable directly, not just
asserted).** For three mutually orthonormal quantization axes
`n_hat_i` (`i=1,2,3`), `sum_i (n_hat_i^T . G . n_hat_i) = Tr(G) = 0`
exactly, because `G` is traceless by construction (the traceless
symmetric part of `grad_E`), independent of `G`'s own eigenvalues/
orientation. So `sum_i Delta_E_Q(J, m_J; n_hat_i) = 0` exactly, for ANY
gradient tensor: the standard "average over three mutually perpendicular
B-field orientations cancels the quadrupole shift" identity (Itano;
demonstrated experimentally by Dube et al. 2005), implemented as a
supported evaluation convenience (`averaging_mode="three_orientation"`,
`cliffordclock.pipeline.QuadrupoleConfig`) and tested at machine
precision (`tests/test_quadrupole_pivot.py`). **Scope caveat (G8, for the
record):** the engine computes the *ideal* (exact-trace) cancellation;
a real clock's residual after averaging (Dube: 5e-18) comes from
imperfect axis orthogonality/timing, a separate, out-of-scope systematic
this project does not model: the report must not present the engine's
exact null as an achievable experimental one.

**Traceless symmetric part (G8 A5#3, shipping requirement).** The
quadrupole interaction couples only to the traceless symmetric part of
`grad_E[i,j] = d_i E_j`; `quadrupole_pivot_perturbation` symmetrizes
(`(G + G^T)/2`) and removes the trace (`G - Tr(G)/3 * I`) before
contracting with `n_hat`: required because the *fitted* RBF-smoothed
tensor (E12) carries a small numerical trace even though the physical
field is exactly traceless in vacuum (`div E = 0`); the antisymmetric
part (which encodes `curl E`, physically zero in electrostatics, but
again not exactly zero in a fitted tensor) never couples and is dropped
by the symmetrization.

**A3 (rotor scope, confirmed) and the spin-connection scope limit.** For
fixed `(J, m_J, quantization axis)` the quadrupole shift is a scalar rate
at a point: no bivector/rotor content beyond the existing scalar
composition (G8 A3: "the tensor character is fully contained in the
*fixed* angular factor... once beta and m_J are external labels rather
than evolved degrees of freedom, that factor is a scalar coefficient").
It is composed into the pivot NUMERATOR `(P-1)` at every evaluation mode
(`cliffordclock.pipeline._make_stark_rate_fn` for
`fast_path`/`secular`/`direct` batched+streaming;
`cliffordclock.pipeline._stark_rotor_ensemble` for `worldline`), exactly
as E33 composes BBR. **Scope limit (documented, not silently omitted):**
unlike BBR's exactly-zero spin-connection gradient (uniform `T`), the
quadrupole term's own value varies with position through `G(r)`, so its
exact contribution to the spin connection `d_k ln P` (E16) would require
`d_k G(r)`, i.e. the THIRD spatial derivative of the electrostatic
potential: a field capability this project's smoother (E12, which
differentiates its fitted form once, to produce `grad_E`) does not
expose. This project does NOT implement that third-derivative
contribution; the quadrupole term reaches `Omega`'s rotation coefficient
(`B_hat_C` plane, via the same route as the Stark/BBR numerator) but not
`omega_boost`'s gradient-sourced piece. Bound: `omega_boost` already
alters the observable only at `O(omega_boost^2)` for the *existing*
DC-Stark spin connection (CONVENTIONS.md section 6's resolved note,
`~1e-19` floor for realistic gradients); the quadrupole term's own
spin-connection piece is a strictly smaller, second-derivative-of-a-
first-order-small-quantity correction on top of that already-negligible
channel, so the omission is bounded well below the 1e-19 floor for every
realistic configuration this project targets, a scope limit analogous
to (not identical to) BBR's exact-zero case, documented here rather than
silently dropped.

## 15. Gravitational-redshift pivot term and the extended-lattice ensemble (v1.4.0, WP22)

Motivation (owner, 2026-08-11, after reviewing the Fortier/Luiten/Margolis
survey, Optica 13, 143 (2026)): the tool's dispersion observables should
land on the mainstream lattice-clock frontier, mm-scale extended samples
(Bothwell et al., Nature 602, 420 (2022)), not only the chamber-scale
showcase. Provenance for every coefficient and citation below:
Bothwell et al., Nature 602, 420 (2022), arXiv:2109.12238, and the
project's G9 theory sign-off record.

**(E36) Gravitational-redshift pivot term.** For a site/atom at height
`h` relative to a configured reference height `h_ref`, the standard
general-relativistic weak-field clock-rate factor enters the pivot as

    (P-1)_grav(r) = U(r)/c^2 = g * (h(r) - h_ref) / c^2 ,   h(r) = up_hat . r

the leading term of the metric proper-time ratio
`sqrt(g_00(r)/g_00(r_ref)) - 1 ~= (U(r) - U_ref)/c^2` (higher-order metric
terms are `O((g*Delta_h/c^2)^2) ~ 1e-32` over a millimetre, irrelevant).
`up_hat` is a configured unit "up" direction
(`environment.gravity.up_axis`,
`cliffordclock.integrator.omega.height_along_axis`) and `g` a configured
local gravitational acceleration (`environment.gravity.g_m_s2`, default
`cliffordclock.constants.STANDARD_GRAVITY` = 9.80665 m/s^2, exact by
international definition).

**Sign convention (G9 sign-off A1, CONFIRMED): a HIGHER clock runs
FASTER.** Under E14b/E21's `(P-1) = Delta_nu/nu_0` convention,
`(P-1)_grav > 0` for `h > h_ref` (`Delta_nu/nu = +g*Delta_h/c^2 ~= +1.09e-16`
per metre, `~= +1.09e-19` per mm at standard g). Mandatory regression
(`tests/test_gravity_pivot.py`): `grav_pivot_perturbation(+1.0,
STANDARD_GRAVITY) > 0`, with the higher-runs-faster physical statement in
the test comment.

**Magnitude (G9 sign-off A1, "computed never transcribed"):**

    g / c^2 = STANDARD_GRAVITY / SPEED_OF_LIGHT**2 = 1.0911370e-16 per metre
            = 1.0911370e-19 per millimetre   (g = 9.80665 m/s^2, c = 299792458 m/s exact)

An earlier theory brief and the Bothwell dossier's first draft both
transcribed `1.0912e-16/m`, one digit off in the fourth significant
figure, the third gate in a row this discipline has caught (G6/G7 before
it). **The regression must assert the value computed from
`cliffordclock.constants.STANDARD_GRAVITY`/`SPEED_OF_LIGHT` at call time,
never a literal from any document** (`tests/test_gravity_pivot.py`).

**Composition (E33's pattern, additive, no cross term).** The
gravitational redshift is a metric effect on the clock rate; it depends
only on position in the potential, not on any electromagnetic field, so
it composes additively in `(P-1)` with the DC-Stark/BBR/quadrupole terms:

    P(r) - 1 = (P-1)_stark(r) + (P-1)_BBR + (P-1)_Q(r) + (P-1)_grav(r) + ...

with no cross term at any order this project resolves (G9 sign-off A2: a
genuine cross term is `O(g*Delta_h/c^2 * field-shift) ~ 1e-16 * 1e-15 ~
1e-31`, negligible). **E36 and the existing kinematic second-order
Doppler (`-<v^2>/2c^2`, E21) are the two leading terms of the same
proper-time expansion and are added ONCE EACH**: E36 is the *potential*
term, the kinematic factor is the *velocity* term, and there is no g x v
cross term at this project's resolution (`O(v^2 * g*Delta_h/c^4)`,
utterly negligible for laboratory velocities). Threading mirrors
`bbr_pivot_perturbation`'s keyword-only composition pattern exactly (a
new `grav_pivot_perturbation` parameter on
`cliffordclock.integrator.omega.pivot_perturbation_stark`/
`spin_connection_stark`/`scalar_rate_perturbation_stark`/
`build_omega_stark`, default `0.0`) but, like the quadrupole term, the
value threaded in is per-position (varies with height), computed by the
caller (`cliffordclock.pipeline._grav_pivot_from_position`) from each
point's own position via `height_along_axis`.

**Rotor scope (G9 sign-off: "the rotor carries it through the scalar
pivot only").** The term reaches `Omega` through the `B_hat_C` rotation
coefficient exactly as the Stark/BBR/quadrupole terms do, and
`spin_connection_stark`'s `P` denominator only, never `omega_boost`'s
numerator. Unlike the quadrupole term's spin-connection omission (bounded
by an unmodeled third field derivative), E36's true gradient along
`up_hat` is `g/c^2` exactly, with no missing derivative; the omission is
nonetheless provably inconsequential rather than merely bounded, because
`omega_boost`'s coefficient carries an explicit factor of velocity `v`
(E18), and every configuration this project ships evaluates STATIC
(`v = 0`) lattice/lattice-extended nodes, so the omitted term is
identically zero for every call site that exists, not just small.

**Validity bounds (G9 sign-off A3, CONFIRMED).** Tidal (height-dependence
of `g`) over a sample: `delta_g/g ~= -2*Delta_h/R_E`, `~3e-10` relative
over 1 mm, so uniform-`g` errs by `~(Delta_h/R_E)*(g*Delta_h/c^2) ~ 3e-29`
over a millimetre sample, exact for all practical purposes.
**Where uniform-g stops being exact at 1e-19:** the error grows as
`epsilon ~= (g/(c^2*R_E)) * Delta_h^2`; setting `epsilon = 1e-19` gives
`Delta_h_max ~= sqrt(1e-19 * c^2 * R_E / g) ~= 76 m`. So uniform g is good
to `<<1e-19` across any lab-scale sample and stays below 1e-19 up to
`~76 m`. `cliffordclock.pipeline.GRAVITY_EXTENT_WARN_M = 10.0` warns (not
a hard reject, per the gate's own "warn (or cap)" wording) when a run's
sampled height extent exceeds this, an order-of-magnitude margin below
the ~76 m bound, and documents that beyond lab scale the physically
correct input is a surveyed potential difference (or a height-dependent
g/geoid model), not `g*Delta_h` with a single `g`. No other GR term is
relevant at lab resolution for static/slowly-moving samples (frame-
dragging, gravitomagnetic, and g x v cross terms are all `<<1e-19`); E36
gives the height-DIFFERENCE redshift relative to a local `h_ref` and
deliberately does not carry the absolute geoid/Earth-rotation offset (a
separate, out-of-scope constant: the engine reports differences within
the sample, not an absolute geopotential).

**Local g vs. standard g (G9 sign-off B1).** At the 1e-19 level standard
gravity is a placeholder; the LAB'S OWN SURVEYED local value is the
physically correct input (e.g. Boulder, CO's USGS-surveyed
`g = 9.796 m/s^2`, van Westrum / NOAA Tech. Memo NOS NGS-77 (2019), the
Bothwell benchmark case's pinned input, `benchmarks/run_bothwell_redshift.py`).
Standard g gives `1.0911e-19/mm`; the std-vs-Boulder slope difference is
`~=0.012e-20/mm` (`1.0911` vs. `1.0900e-19/mm`), about 8x SMALLER than
Bothwell's own stated `<0.1e-20/mm` prediction uncertainty: well within
it, not merely comparable to it (G9's correction of an earlier
overstatement, preserved in G9's own wording). `environment.gravity.g_m_s2` defaults to standard gravity
with this local-value recommendation documented (report note,
`cliffordclock.pipeline._gravity_provenance_note`).

### The `lattice_extended` ensemble regime (WP22 Part 2, engineering)

`ensemble.regime: lattice_extended` (`cliffordclock.pipeline.EnsembleConfig`,
`cliffordclock.ensemble.lattice.extended_lattice_nodes`) distributes
`n_sites` copies of the `lattice` regime's single-site Hermite-Gauss
motional quadrature (E29-exact) along a configured axis
(`ensemble.site_axis`), spaced by `ensemble.site_spacing_m`, with a
Gaussian-or-uniform site-occupation envelope (`ensemble.site_envelope`).
Every site's own position feeds every pivot term already in scope at that
call site (the local field/Stark term via `field_fn`, the uniform BBR
term, and E36's height-dependent redshift) through the SAME `rate_fn`/
`_stark_rotor_ensemble` accumulators the `lattice` regime uses: no new
evaluation-mode machinery; `fast_path` (E29, the default) and `worldline`
(the E17-E24 rotor cross-check) both apply unchanged, since every
extended-lattice node remains a static (`v = 0`) point exactly like a
`lattice` node. The existing `lattice` regime (single site) is entirely
untouched: byte-identical for every shipped example, none of which sets
`ensemble.regime: lattice_extended`.

**Output: the per-site frequency map** (`PipelineResult.site_map`,
`cliffordclock.pipeline.LatticeExtendedSiteMap`), the Bothwell
observable: each site's own position, weighted mean fractional shift
(over that site's local motional quadrature only), and normalized
occupation weight, plus a weighted-least-squares linear-gradient fit
across the site means (`slope_per_m`, the map's headline number).

**Dispersion labeling (G9 sign-off A4, required edit, BINDING).** For an
extended sample the frequency spread across sites is dominated by the
DETERMINISTIC linear gradient (higher/lower sites tick at a
systematically different rate, not a stochastic process, and in
principle refocusable), not stochastic sampling. Reporting a per-site
map, an ensemble spread, and a T2*/linewidth together risks (a)
double-counting the gradient (once in the map, once in a linewidth) or
(b) misreading deterministic, inhomogeneous broadening as stochastic
decoherence. **Ruling:** every `lattice_extended` report carries a
test-pinned note (`cliffordclock.pipeline.LATTICE_EXTENDED_DISPERSION_LABEL_NOTE`)
stating that `MetrologyReport.t2_star_s`/`shift_std_error` include the
deterministic gradient, and `LatticeExtendedSiteMap` additionally reports
BOTH the total spread (`total_spread_fractional`, the same combined
number in fractional-shift units) AND the gradient-removed residual
spread (`gradient_removed_residual_spread_fractional`: the weighted
standard deviation of each site's mean shift after subtracting the
best-fit linear gradient), extending the showcase's existing SEM-vs-T2*
discipline (`docs/tutorial.md`) to the deterministic-vs-stochastic axis.

### G9 sign-off record (2026-08-11, the project's theory sign-off record (G9))

**Part A (formalism): approved for build, one correction, one required
edit.** A1: E36's form and sign CONFIRMED; the stated magnitude
`1.0912e-16/m` CORRECTED to the computed `1.0911370e-16/m` (a
transcription slip in both the brief and the dossier), the regression
must pin the computed value, never a document literal, and the Bothwell
coordinate-sign mapping must be stated wherever the benchmark case
appears (below). A2: composition/non-coupling CONFIRMED, including the
"added once each" consistency point against the existing kinematic
second-order Doppler. A3: validity bounds CONFIRMED, with the ~10 m
warn threshold recommendation (an order-of-magnitude margin below the
~76 m 1e-19 bound). A4: dispersion labeling approved WITH the required
deterministic-vs-stochastic labeling edit (above).

**Part B (benchmark ratifications): all five ratified.** B1: Bothwell's
surveyed `g = 9.796 m/s^2` pinned as the case's reference input (engine
default stays standard g), with the two number corrections above
(`1.0911e-19/mm`, "well within" not "comparable to" their uncertainty).
B2: compare the predicted slope against BOTH corrected measurements,
method A `-9.8(2.3)e-20/mm` and method B `-1.28(27)e-19/mm`
(band-overlap MET at 0.48-sigma/0.70-sigma). B3: the Gaussian
envelope/~1 mm span/6.04 um bins/406.5 nm INFERRED site-spacing geometry
ratified (discretization-only, slope-independent). B4: classification
"reproducibility" with the inverted-NPL caveat (the g/c^2 arithmetic is
textbook; the tool's contribution is the extended-sample machinery
producing the measured-map slope end-to-end with zero adjustable
inputs), does not change the blind-prediction count. B5: Zheng et al.
(Nature 602, 425 (2022), arXiv:2109.12237) ratified as the named second
extended-lattice benchmark candidate, post-beta.

**Coordinate-sign mapping (G9 sign-off A1, required wherever the
Bothwell case appears).** Bothwell's own z-axis puts LOWER physical
positions at LARGER coordinate, so their reported redshift gradient is
NEGATIVE (`-10.9e-20/mm`) while this engine's `(P-1)_grav` increases with
physical height (positive slope in the engine's own `up_axis`/`offset_m`
convention). `benchmarks/run_bothwell_redshift.py` computes the slope in
the engine's own physical-height convention and then negates it for the
comparison, with the mapping stated explicitly in the script and its
report, so the sign agreement with Bothwell's published gradient is
deliberate, not coincidental.

## 16. Quantum-motional second-order Doppler (time-dilation) pivot term (v1.6.0, WP30)

Motivation (project owner): the second-order (quadratic) Doppler shift from
an ion or atom's own residual motion in its trap is the DOMINANT systematic
of trapped-ion optical clocks (e.g. the Al-27+ secular-motion budget line
already carried in this project's ion-species report notes,
`cliffordclock.ensemble.species.ION_MICROMOTION_NOTES["Al27+"]`: "-17.3(2.9)e-19",
Brewer et al., Phys. Rev. Lett. 123, 033201 (2019)) and a real budget row
for lattice clocks too. E32/E33's BBR term and E36's gravitational term are
both scalar, position-dependent (or uniform) pivot contributions; E38 is
the third scalar contribution this document adds, and the first driven by
the atom's *motional quantum state* itself, not its field environment or
height.

**(E38) Motional time-dilation term.** For an atom/ion in harmonic
confinement, near its motional ground state, with `N` normal modes indexed
`i`, each having an ORDINARY (not angular) mode frequency `f_i` (hertz,
e.g. as reported directly by resolved-sideband thermometry) and a mean
vibrational occupation number `n_bar_i` (also from sideband thermometry, or
an equivalent Doppler-limit statement), the mode ANGULAR frequency is

    omega_i = 2 * pi * f_i

and the velocity-variance expectation over the motional state is

    <v^2> = sum_i (hbar * omega_i / m) * (n_bar_i + 1/2)

with `m` the SPECIES mass (`cliffordclock.ensemble.species.Species.mass_kg`,
resolved from the registry, never hand-typed) and `hbar`
(`cliffordclock.constants.HBAR`) this document's `hbar = h/2pi` (§1). The
fractional time-dilation shift is the second-order-Doppler form already
used everywhere else in this document (E21's kinematic factor, `sqrt(1 -
v^2/c^2) - 1 ~= -v^2/(2c^2)` for `v << c`), here evaluated as an
EXPECTATION VALUE over the quantum motional state, in place of a classical
instantaneous velocity:

    (P-1)_motional = -<v^2> / (2 * c^2)

**CAREFUL: mode frequencies are ordinary frequencies, not angular.** Sideband
thermometry reports `f_i` in hertz (or MHz); supplying an already-angular
value here would silently overstate `<v^2>`, and hence the shift, by a
factor of `(2*pi)^2 ~= 39.5`. The explicit `omega_i = 2*pi*f_i` conversion
above is mandatory at every call site; `cliffordclock.integrator.omega.MotionalMode.frequency_hz`'s
docstring states this explicitly and
`cliffordclock.integrator.omega.motional_mean_squared_velocity_m2_s2`/
`motional_pivot_perturbation` perform the conversion internally so no
caller ever multiplies by `2*pi` (or fails to) on its own.

**Ground-state limit.** `n_bar_i = 0` for every mode does not make
`(P-1)_motional` zero: the `+1/2` zero-point term alone still contributes
`<v^2> = sum_i (hbar*omega_i/2m)`, the irreducible quantum-mechanical
motional-Doppler floor even for an atom cooled exactly to its motional
ground state.

**Per-mode participation factors for multi-ion crystals (v1.7.0, WP31).**
In a mixed-species crystal each normal mode's energy divides between the
trapped ions by the squared components of the mass-weighted eigenvectors:
the CLOCK ion does not carry a mode's ENTIRE `<v^2>` contribution unless
it is the only ion sharing that mode. The formula generalizes to

    <v^2> = sum_i (hbar * omega_i / m_clock) * participation_i * (n_bar_i + 1/2)

with `participation_i` in `(0, 1]` the clock ion's squared mass-weighted
eigenvector component in mode `i`
(`cliffordclock.integrator.omega.MotionalMode.participation`), default
`1.0`, reproducing today's single-species formula bitwise. For a
TWO-ION crystal the eigenvector components are closed-form functions of
the mass ratio `mu = m_partner/m_clock` for the AXIAL pair (the standard
two-ion normal-mode solution: an in-phase and an out-of-phase mode; the
equal-mass limit gives `participation = 1/2` for both modes, each ion
carrying exactly half); `cliffordclock.integrator.omega.two_ion_participations`
implements this closed form, applying it (as a documented approximation,
not an additional exact result) to the two radial pairs too, since the
full radial closed form additionally depends on trap RF/DC geometry
parameters beyond the mass ratio alone; see that function's docstring
for the derivation, its citation, and the radial scope caveat.
`benchmarks/run_motional_al_ion.py` compares both the single-mass
(`participation=1.0`) and participation-corrected variants against a
published two-ion crystal's per-mode data.

**Radial participation factors from spectrum reconstruction (v1.8.0,
WP32).** `two_ion_participations`'s radial rows are a documented
approximation (the axial mu-only closed form reused for radial, since the
full radial closed form needs trap RF/DC geometry parameters this project
has no way to supply). WP32 replaces that approximation for a caller who
has the lab's own measured radial mode frequencies, by inverting the
two-ion radial normal-mode problem directly against them instead of
assuming a trap geometry. Per transverse direction, the coupled radial
equations of motion in mass-weighted coordinates form a 2x2 eigenproblem

    [[omega_r1^2 - c/m_clock, c'], [c', omega_r2^2 - c/m_partner]]

with `omega_r1`, `omega_r2` the two ions' own (unknown) bare radial
frequencies, `c = e^2/(4*pi*eps0*d^3)` the Coulomb curvature at the
equilibrium spacing `d`, and `c' = c/sqrt(m_clock*m_partner)`. The
transverse Coulomb curvature is exactly half the axial curvature's
magnitude and opposite in sign (a standard two-ion-crystal
electrostatics result, worked from `U_int = e^2/(4*pi*eps0*|r2-r1|)`
expanded to second order in the axial and transverse displacements about
`d`); `c` itself is recovered from the AXIAL confinement alone, since the
two ions share an axial spring constant `k_z = m_clock*omega_z1^2 =
m_partner*omega_z2^2` (Wubbena et al. 2012 Eq. 7: the DC field gradient
sets a mass-independent per-ion spring constant), giving `d^3 =
(e^2/4*pi*eps0)*2/k_z` from the equilibrium force balance and hence `c =
k_z/2` exactly, with `k_z` itself recovered from the measured axial
IN-PHASE (COM) mode frequency by inverting Wubbena Eq. 12
(`cliffordclock.integrator.omega.axial_coulomb_curvature`). The two
measured radial mode frequencies squared are the matrix's two
eigenvalues; their sum and product (trace/determinant) invert to the two
diagonal entries up to a branch swap (the QUADRANT AMBIGUITY), resolved
by requiring the LIGHTER ion to carry the higher bare radial frequency
(RF pseudopotential confinement scales as `1/mass` at fixed trap drive,
Wubbena's own `omega_p` expression before Eq. 3), and
`cliffordclock.integrator.omega.two_ion_radial_participations` implements
the full inversion, raising a `ValueError` naming the numbers whenever
the measured frequencies are infeasible for the computed coupling
(`(lambda_hi-lambda_lo)^2 < 4*c'^2`) or the branch choice is not
resolved (equal masses, neither/both branches satisfying the
disambiguation rule, or the winning branch's separation not clearing its
own propagated uncertainty, including implicitly, since the same
checks re-run at every finite-difference uncertainty sample point).
`benchmarks/run_motional_al_ion.py`'s WP32 case reconstructs both
transverse branches for the Al27+/Mg25+ crystal from Marshall et al.'s
own published mode frequencies and reports the result with no tuning:
the reconstruction's per-mode ratios against Marshall's own published
per-mode weights sit in the same rough range as WP31's approximation
(neither matches well), and the reconstructed-participation TOTAL lands
at essentially the same total-level deviation from the published band as
WP31's total (~14.0 sigma versus ~14.11 sigma, still `NOT MET`). The
reason is structural, not a flaw in the reconstruction: each mode-pair's
clock-ion participations sum to `1.0` exactly regardless of how a pair's
total is split between its COM and STR members, and Marshall's own
COM/STR `(n_bar+1/2)`-weighted magnitudes for a given branch are
comparable in size, so redistributing participation within a radial pair
moves the per-mode ratios without moving that pair's own total by much,
and getting the split right need not by itself close a total-level gap
this size. This is reported as an open empirical finding, not resolved
here.

**Mode-specific intrinsic-micromotion enhancement (v1.9.0, WP33), closing
the reconciliation.** The project's G14 gate review, independently
re-deriving WP32's physics from first principles and verifying every
cited equation against the primary sources page by page, identified the
mechanism behind WP32's open gap: Marshall et al.'s (and, per Brewer et
al., arXiv:1902.07694, Table S2 footnote (a), same species pair)
published radial rows already include the shift due to INTRINSIC
micromotion, the unavoidable micromotion that accompanies secular
motion itself (an ion carried through the RF field's spatially-varying
amplitude by its own secular oscillation), distinct from EXCESS
micromotion (the EMM paragraph below, a stray-DC-field effect this
project's `v_rms_emm_m_s` channel already accepts as a lab-supplied
input). Testing the candidate quantitatively: participation times a
UNIFORM factor of two closes the Y branch in both independent datasets
(Marshall 1.03/1.00, Brewer 0.92/0.97) while leaving X reproducibly
20-35% short in both. The residual has an identified mechanism, since
the leading-order intrinsic-micromotion enhancement is MODE-SPECIFIC,

    F_axis = 1 + q^2 / (2*a_axis + q^2)

(Berkeland, Miller, Bergquist, Itano, Wineland, J. Appl. Phys. 83, 5025
(1998) Eq. 10, the same paper this document's EMM paragraph below
already cites as canonical), equal to `2` only when the trap's Mathieu
`a` parameter vanishes for that axis; the trap's DC asymmetry splits
`a_x` from `a_y`, so a single factor of `2` cannot be exactly right for
both radial branches at once. WP33 solves for the CLOCK ion's own
leading-order Mathieu parameters directly from published/WP32-
reconstructed inputs alone: the trap's published RF drive frequency
`Omega` (Marshall: `Omega/2pi = 70.86 MHz`), the clock ion's own bare
axial frequency (`omega_z,clock = sqrt(2*c/m_clock)`, reusing
`axial_coulomb_curvature`'s own Coulomb curvature `c`, not re-deriving
Wubbena Eq. 12's inversion), and the two WP32-reconstructed clock-ion
bare radial frequencies, via the leading-order relation
`omega_i = (Omega/2)*sqrt(a_i + q_i^2/2)` (Berkeland Eq. 9, `q_z=0` for
the axial direction, Berkeland Eq. 6) together with the Laplace
constraint `a_x + a_y = -a_z` (the general statement for any static
quadrupole DC potential; Berkeland Eq. 5's `a_x=a_y=-a_z/2` is the
special radially-symmetric case). This is two equations (the two
WP32-reconstructed radial frequencies) in two unknowns (`a_x`, `q`, with
`a_y` and `a_z` following from the Laplace constraint and the axial
frequency respectively), zero degrees of freedom, no trap-geometry
parameter (`alpha`/`epsilon`) supplied as input.
`cliffordclock.integrator.omega.clock_ion_mathieu_parameters` implements
this closed form (`ClockIonMathieuParameters`, with finite-difference
uncertainty propagation mirroring `two_ion_radial_participations`'s own
style); `radial_micromotion_enhancement` implements `F_axis` directly,
literally Berkeland's own Eq. 10 bracket, not independently re-derived.

**MANDATORY over-determination check.** Because `a_i` and `q_i` are each
LINEAR in `1/mass` at fixed trap drive voltage/geometry/charge (Berkeland
Eqs. 5-6, the same charge for both singly-charged ions), the clock ion's
solved Mathieu parameters mass-scale directly to predict the PARTNER
ion's own bare radial frequencies
(`cliffordclock.integrator.omega.predicted_partner_bare_radial_
frequencies_hz`), a genuinely independent, falsifiable test, since
nothing in the Mathieu-parameter solve's own inputs (`Omega`, the clock
ion's own axial/radial frequencies) ever touches the partner ion's
frequencies. Both datasets pass this check at the sub-1%-relative level
(Marshall: `-0.57%`/`-0.53%` on X/Y; Brewer: `-0.95%`/`-0.79%`), well
inside the few-percent band the published mode frequencies' own
~3-significant-figure reporting precision supports; reported whatever
it said, not tuned toward agreement, and it passed.

**Result.** Multiplying WP32's radial participations by `F_x`/`F_y`
(Marshall: `F_x=2.474`, `F_y=1.888`) moves the four radial per-mode
ratios (predicted/published) from WP32's `0.42/0.35/0.52/0.50` to
`1.04/0.86/0.98/0.94`, landing near `1.0` on every mode, an improvement
on every individual mode, not just on average; the corrected TOTAL moves
from WP32's `-5.72e-18` (~14.0 sigma from Marshall's published
`-114.6(3.8)e-19`) to `-1.064e-17` (~1.6 sigma), still technically
`NOT MET` (the predicted and published bands do not quite overlap) but a
near-complete closure of the gap the G14 gate review identified, not a
marginal shift. The SAME enhancement, applied to Brewer et al.'s
independently published trap (`Omega/2pi=40.72 MHz`, different mode
frequencies, same species pair), moves the four radial per-mode ratios
against Brewer's own published `TDS/quantum` row (which Table S2's
footnote (a) already states includes the transverse intrinsic-
micromotion shift) from a plain-participation `0.38/0.38/0.46/0.49` to
`0.90/0.88/0.88/0.93`, the same qualitative improvement in a second,
independent dataset, not a coincidence of the first.
`benchmarks/run_motional_al_ion.py`'s WP33 case
(`run_motional_al_ion_intrinsic_micromotion_enhanced_case`) and its
Brewer consistency check (`run_wp33_brewer_consistency_check`) report
both results in full, with no tuning; Brewer's own total-level
`-17.3(2.9)e-19` row is NOT reproduced (its `n_bar` input is a 95%-CI
bound combined with a heating rate through Brewer's own time-dependent
Eq. 3, not the static point estimate this project's E38 formula
consumes; the same missing-input reason WP30-32 already use Marshall
instead of Brewer for their own total-level cases).

**Regime of validity.** Both Berkeland's `omega_i` formula (step above)
and its `F_axis` result are FIRST ORDER in the Mathieu parameters
(Berkeland's own "`|q_i|<<1` and `|a_i|<<1`" caveat, citing Landau &
Lifshitz's perturbative Mathieu solution); this project's own solved
`q ~ 0.19-0.25`, `|a_i| ~ 0.003-0.008` sit inside that regime (smaller
than Berkeland's own worked `q~0.28` example). No published input in
either dataset used here supports evaluating the exact Mathieu
characteristic-value relation instead, so the leading-order form is used
as is, with its regime stated explicitly instead of silently assumed.

**Remaining scope boundary.** With the two-ion partition now consumed for
both the axial pair (exact, WP31), the radial pairs from a lab-supplied
measured spectrum (WP32), and the radial intrinsic-micromotion
enhancement from that same spectrum plus the trap's published RF drive
frequency (WP33), what remains open is (i) `N > 2`-ion crystals, which
need a numeric normal-mode eigensolver (no closed form in general), (ii)
the exact (not leading-order) Mathieu treatment, should a future dataset
need it outside `|a_i|,|q_i| << 1`, (iii) the excess-micromotion RF
dynamics package (unrelated to intrinsic micromotion: see the EMM
paragraph below), and (iv) the residual sub-2-sigma gap WP33's own
Marshall case still reports, materially narrowed from WP32's ~14 sigma
but not fully closed.

**Optional second input channel: excess micromotion (EMM).** Ion-trap
clocks additionally accumulate a time-dilation shift from excess
micromotion, the atom's forced oscillation at the trap's RF drive frequency
when displaced from the RF null by a stray DC field (Berkeland, Miller,
Bergquist, Itano, Wineland, J. Appl. Phys. 83, 5025 (1998), the canonical
treatment already cited by this project's ion-species micromotion-boundary
notes). This project does not model the RF trap dynamics that PRODUCE EMM
(the stray-field-induced displacement from the RF null, the resulting
time-varying velocity at the RF drive frequency); that treatment is a
genuine roadmap package, not built here. Instead, `E38` accepts a lab's
own MEASURED EMM characterization, already reduced by the lab to an
equivalent rms velocity
`v_rms_emm_m_s` (a common form published EMM characterizations already
report, or one an equivalent published fractional shift can be converted
to), adding its square directly to the motional-state `<v^2>`:

    <v^2> = sum_i (hbar * omega_i / m) * (n_bar_i + 1/2) + v_rms_emm^2

This is exactly the same "characterization taken as an input, not
independently re-derived" pattern E37 already uses for a lab's own
solid-angle/emissivity characterization (§13's "Surfaces and weights" note:
"the fractions are an input, not a computed quantity"): `v_rms_emm_m_s`
is the lab's own number, not something this tier computes from trap RF
parameters.

**No double-counting (the central argument).** The engine already carries
second-order Doppler through the kinematic factor `sqrt(1 - v^2/c^2)` in
E15/E21 along CLASSICAL trajectories: for `ensemble.regime="classical"`
(the Monte-Carlo ensemble, real trajectories with real, nonzero sampled
velocities), this kinematic factor is already the physically correct,
complete second-order-Doppler shift for that ensemble's classical motion,
and composing E38 there on top of it would double-count the same physics
under two different accounting schemes. The trapped regimes
(`ensemble.regime="lattice"`/`"lattice_extended"`) instead evaluate at
STATIC quadrature nodes (`v = 0` exactly, E29's own scope statement: "every
node has `v = 0`... [E29] omits the motional second-order Doppler... a
real, separately-budgeted clock systematic"), which is precisely WHY the
quantum-motional term is missing there and precisely why adding E38 in
those regimes cannot double-count: the classical kinematic contribution
`-v^2/(2c^2)` is IDENTICALLY zero at a static node (`v = 0` makes the
kinematic term vanish exactly, an exact cancellation, not an approximate
one), so E38 supplies genuinely new physics at that node, never a second
accounting of physics already present. `cliffordclock.pipeline` enforces the
complementary half of this argument mechanically:
`environment.motional_state` paired with `ensemble.regime: classical`
raises `PipelineConfigError` at config-parse time, naming this exact
double-counting hazard; `ensemble.regime: lattice`/`lattice_extended` are
the only regimes E38 may be configured for.

**Composition (E33's pattern, spatially uniform).** Like E32's BBR term
(a single radiation temperature/environment, not a `T(r)` map), E38's
motional state is ONE motional state per run: every atom in the ensemble
is assumed to share the identical set of normal-mode occupations, not a
per-atom motional map (a genuine extension for future work, exactly
analogous to E37's own per-atom-solid-angle-map scope boundary, §13's
"Scope boundary" note). `E38` composes additively into `(P-1)` per E33:

    P(r) - 1 = (P-1)_stark(r) + (P-1)_BBR + (P-1)_Q(r) + (P-1)_grav(r) + (P-1)_motional + ...

Because `(P-1)_motional` is spatially uniform, its gradient is exactly
zero, so it reaches `spin_connection_stark`'s `P` denominator exactly as
BBR's/gravity's own "shifts the denominator only" composition note
already describes, contributing nothing to that function's
numerator/gradient term.
Threading mirrors `bbr_pivot_perturbation`'s keyword-only composition
pattern exactly: a new `motional_pivot_perturbation` parameter on
`cliffordclock.integrator.omega.pivot_perturbation_stark`/
`spin_connection_stark`/`scalar_rate_perturbation_stark`/
`build_omega_stark`, default `0.0`, composed into every evaluation mode
(`fast_path`/`worldline`, the only two modes `ensemble.regime:
lattice`/`lattice_extended` support) via `cliffordclock.pipeline`'s
`_make_stark_rate_fn`/`_stark_rotor_ensemble`, exactly as
`bbr_pivot_perturbation` already is.

**Uncertainty propagation.** Writing `omega_i = 2*pi*f_i`, the partial
derivatives of `(P-1)_motional` with respect to each input are:

    d(P-1)/d(n_bar_i)   = -(hbar * omega_i / m) / (2*c^2)
    d(P-1)/d(f_i)       = -(hbar * 2*pi * (n_bar_i + 1/2) / m) / (2*c^2)
    d(P-1)/d(v_rms_emm) = -v_rms_emm / c^2

Each mode's `n_bar_i`/`f_i` 1-sigma uncertainty and `v_rms_emm`'s own
1-sigma uncertainty propagate through these exact partials and combine in
quadrature (independent-error assumption, the same "arithmetic-
reproduction fidelity, not an independent physics-accuracy claim" caveat
E32's uncertainty note already states for its own registry-coefficient
propagation): this is the propagated uncertainty of the SUPPLIED mode/EMM
inputs, not an independent assessment of the underlying trap physics.
Implemented in `cliffordclock.integrator.omega.motional_pivot_uncertainty`.

**Config surface.** `environment.motional_state` (a list of per-mode
`name`/`frequency_Hz`/`n_bar`/`n_bar_uncertainty`/`frequency_uncertainty_Hz`
entries plus an optional `v_rms_emm_m_s`/`v_rms_emm_uncertainty_m_s` pair,
`cliffordclock.pipeline.MotionalStateConfig`), requiring
`coupling.type: stark_dc` (mirrors E32/E36/E34's own cross-field
requirement: the composition point is the E14b Stark rate function) and
rejected for `ensemble.regime: classical` (the no-double-counting argument
above). The pipeline report's `uncertainty_notes` states every configured
mode by name with its frequency and `n_bar`, the resolved `<v^2>`, the
resulting `(P-1)_motional` shift, the EMM input when present, the
propagated uncertainty, and the excess-micromotion roadmap-boundary note
verbatim (`cliffordclock.pipeline._resolve_motional_pivot_perturbation`).

**Exact Floquet treatment of intrinsic micromotion (v1.10.0, WP34).** WP33's
`F_axis = 1 + q^2/(2*a_axis+q^2)` and its `omega_i = (Omega/2)*sqrt(a_i +
q_i^2/2)` are both FIRST ORDER in the Mathieu parameters (Berkeland's own
`|q_i|<<1`, `|a_i|<<1` caveat). WP34 replaces both with the numerically
exact Floquet solution of the Mathieu equation
`u'' + [a + 2*q*cos(2*tau)]*u = 0` (Berkeland Eq. 4 convention): writing the
stable solution `u(tau) = exp(i*beta*tau)*phi(tau)` with `phi` periodic,
`phi(tau) = sum_n c_n*exp(i*2*n*tau)`, the Fourier coefficients satisfy the
standard tridiagonal recursion `q*c_(m-1) + [a-(beta+2*m)^2]*c_m +
q*c_(m+1) = 0`; eliminating the off-diagonal entries via two one-sided
continued fractions reduces the Hill-determinant stability condition to a
single scalar equation in `beta`, solved by Newton root-find at a
truncation depth chosen by an explicit convergence test (doubling the
depth until both `beta` and the retained Fourier ratios stop changing,
raising if a ceiling is reached first).
`cliffordclock.integrator.omega.mathieu_floquet_solve` implements this
(`MathieuFloquetSolution`); this session's own cross-check against
independent monodromy-matrix ODE integration (the trace of the one-period
fundamental-solution matrix) confirms agreement to float64 working
precision. `omega_sec = beta*Omega/2` EXACTLY, the definition of the
characteristic exponent itself, and the exact velocity-variance
enhancement is `F_exact = sum_n c_n^2*(beta+2*n)^2 / beta^2` (`c_0=1`
normalization, `MathieuFloquetSolution.velocity_enhancement_exact`/
`radial_micromotion_enhancement_exact`). Two checks confirm this reduces
correctly: `q -> 0` gives `F_exact -> 1` exactly, and substituting the
leading-order relation `beta^2 = a+q^2/2` into `F_exact`'s own formula
reproduces Berkeland's bracket EXACTLY at that substitution, the same
algebraic expression on both sides once `beta` is fixed there. WP33's
closed-form `(q, a_x, a_y)` inversion is re-solved by 2D Newton iteration
against the exact `beta(a, q)` map (`clock_ion_mathieu_parameters_exact`),
using WP33's own closed form as the initial guess; the exact map has no
algebraic inverse, so this step needs a numerical solve. The mandatory
partner-ion over-determination check is re-run with the exact
partner-frequency prediction
(`predicted_partner_bare_radial_frequencies_hz_exact`, the SAME
mass-scaling relation WP33 uses, only the `beta` evaluation swapped). The
axial direction needs no exact treatment: `q_z=0` (Berkeland Eq. 6)
reduces the Mathieu equation to the plain harmonic oscillator exactly, so
`beta=sqrt(a_z)` and `F_axial=1` are already exact at WP33's own leading
order. `benchmarks/run_motional_al_ion.py`'s WP34 case
(`run_motional_al_ion_exact_intrinsic_micromotion_enhanced_case`) and its
Brewer consistency check (`run_wp34_brewer_consistency_check`) report both
datasets with the exact treatment: the Al27+/Mg25+ Marshall total moves
from WP33's `-1.064e-17` (1.62 sigma) to `-1.064e-17` (1.62 sigma, to the
precision reported), the two staying close because this project's own
`(a, q)` sit well inside Berkeland's stated leading-order validity regime,
where the exact-vs-leading-order correction to `F_x`/`F_y` is at the
sub-1%-relative level. This closeness is a structural fact that holds for
any dataset in that regime: `F_axis` depends only on the axis's own
`(a, q)`, so the SAME factor multiplies both a pair's COM and STR members
in both the leading-order and exact treatments, and that factor can only
scale an existing within-axis deviation, never flip its sign. The
opposite-sign per-mode deviation the G14/G15 gate reviews found within the
X axis (`x_com` above 1.0, `x_str` below 1.0) survives the exact treatment
unchanged for exactly this reason. That residual sits outside the
single-per-axis-enhancement-factor model WP33 and WP34 both implement;
a genuinely per-mode mechanism, still within Mathieu-order physics but
coupling the two ions' motion beyond that single shared factor, remains
open, alongside the published rows' own per-mode calibration chain as a
second open candidate. The TOTAL remains the right level at which to
compare this project's reconstruction against Marshall's published
number.

**Input-rounding uncertainty (v1.10.0, WP34, Part 2).** Marshall's (and
Brewer's) published mode frequencies and RF drive frequency are each
stated to a fixed number of decimal places with no measurement
uncertainty at all; the WP30-33 cases above propagate zero for that
missing channel, understating the true uncertainty on their totals. WP34
adds a ROUNDING-bound channel: each published frequency carries a
half-last-digit bound (e.g. `2.16` MHz carries `+/-0.005` MHz), propagated
by finite differences through the FULL reconstruction chain (the axial
spring constant, the radial inversion, the participations, the exact
Mathieu solve, the exact enhancement, and the total) and combined in
quadrature with the existing phonon-number (thermometry) uncertainty. The
two components are reported SEPARATELY and labeled as such
(`predicted_total_uncertainty_nbar_fractional`/
`predicted_total_uncertainty_rounding_fractional`/
`predicted_total_uncertainty_combined_fractional`, WP34's benchmark case
record), so a reader can see what comes from rounding and what from
thermometry without the two being folded silently into one number. This
rounding channel is a BOUND: the true value could differ from the printed
one by up to half its last digit. It is documented throughout as a bound,
distinct in kind from a measured 1-sigma uncertainty even where the two
combine in quadrature. For the Al27+/Mg25+ Marshall dataset the
rounding channel comes out roughly an order of magnitude smaller than the
thermometry channel, leaving the reported band essentially where the
thermometry channel alone would place it; the rounding channel is still
carried through and reported in full, on the same footing as the
thermometry channel, instead of being assumed negligible and dropped.

**The coupled two-ion Floquet solve (v1.11.0, WP35), the most complete
treatment in this lineage.** WP33 and WP34 both write the clock ion's
per-mode velocity variance as `participation_i * F_axis`:
`participation_i` from WP32's SEPARATE secular-only 2x2 eigenproblem,
`F_axis` from the ion's OWN uncoupled Mathieu solve. A single per-axis
`F_axis` multiplies BOTH the COM and STR members of a pair identically,
so this factorization cannot, by construction, produce a relative shift
between them; WP34's own structural finding names this directly. The
physical gap: `F_axis` assumes the ion oscillates at its own BARE Floquet
frequency, but the clock ion's slow motion in a collective (Coulomb-
coupled) mode actually runs at that mode's own shifted quasi-frequency,
different for COM and STR of the same axis. WP35 removes the
factorization: per transverse axis, the two ions' time-periodic equations
of motion, coupled by the SAME Coulomb curvature `c` WP32 uses (mass-
weighted coordinates, `y1'' = -[Omega_1(t)^2 - c/m1]*y1 - c'*y2`, `y2''`
symmetric, `c' = c/sqrt(m1*m2)`), are integrated directly as a
4-dimensional linear time-periodic (Hill) system: its 4x4 monodromy
matrix over one RF period (`scipy.integrate.solve_ivp`, `DOP853`) has two
complex-conjugate eigenvalue pairs giving the two collective quasi-
frequencies exactly (the SAME `beta = arg(mu)/pi` convention
`mathieu_floquet_solve` uses). Propagating each mode's eigenvector
through one period and Fourier-decomposing the periodic part gives, per
ion `i` and mode `k`, a genuine Fourier series `c_i,k,n`; the clock ion's
participation (`|c_1,k,0|^2 / sum_j|c_j,k,0|^2`) and its EXACT per-mode
micromotion enhancement (`sum_n|c_1,k,n|^2*(beta_k+2n)^2 /
(|c_1,k,0|^2*beta_k^2)`) both come from this ONE decomposition, with no
separately-computed single-ion `F` composed in afterward.
`cliffordclock.integrator.omega.coupled_two_ion_floquet_modes` implements
this; verified against two EXACT limits: `c -> 0` reproduces WP34's own
single-ion exact Floquet result for each ion separately, and `q -> 0`
(no RF) reproduces WP32's own static secular participation decomposition
exactly, both to float64 working precision. A reviewer's cheap, non-
self-consistent forced-oscillator estimate
(`mathieu_forced_oscillator_enhancement`: the ion's own leading nearest-
neighbor Mathieu sidebands evaluated at the collective mode's externally
supplied quasi-frequency, `F(beta_mode) = 1 +
sum_(k!=0)c_k^2*(2k+beta_mode)^2/beta_mode^2` with `c_k =
q/((2k+beta_mode)^2-a)`) is reported alongside as its own labeled
comparison column. A note on energy bookkeeping: WP31/WP32's plain
participations sum to `1.0` across the two ions sharing a mode, correct
for secular motion alone; once enhancement multiplies participation, the
two ions' `participation*enhancement` shares add to something larger
than `1.0` in general, because the RF drive itself supplies the
additional kinetic energy the enhancement factor accounts for. That
larger sum is the expected signature of a driven system.

**The constrained fit (v1.11.0, WP35 Part 2), this WP's own headline
result (G17 gate fix loop).** The clock ion's own `(q, a_axis)`, solved
per axis from that SAME axis's own two measured mode frequencies
(`coupled_two_ion_mathieu_parameters`), fit each axis's two unknowns
exactly by construction (two equations, two unknowns, zero residual):
the resulting near-unity per-mode ratios carry no independent evidence
the underlying model is right, since a perfect fit there is guaranteed
regardless of the physics. Berkeland Eq. 6 states the trap has ONE RF
parameter magnitude per ion (`q_x = -q_y`); combined with the Laplace
constraint `a_x+a_y=-a_z` (`a_z` fixed exactly from the measured axial
frequency, the same relation WP34 uses) and a single DC-split fraction
`alpha` (`a_x=-alpha*a_z`, `a_y=-(1-alpha)*a_z`, satisfying the
constraint for any `alpha`), the whole crystal's radial dynamics reduce
to just two unknowns, `(q, alpha)`, predicting FOUR measured frequencies
(X-COM, X-STR, Y-COM, Y-STR) through the SAME coupled-Floquet forward
model. This is genuinely over-determined: Gauss-Newton least squares
(`cliffordclock.integrator.omega.constrained_two_ion_mathieu_fit`)
minimizes the summed squared frequency residuals and reports those
residuals exactly as computed. They are this fit's own falsifiable
output, the same epistemic shape as WP32/WP33/WP34's own over-
determination checks. The over-determination PARTNER check WP35's
own derivation promised is implemented for this fit directly: mass-scale
`(q, a_x, a_y)` to the partner ion and evaluate its own uncoupled exact
Floquet beta (`predicted_partner_bare_radial_frequencies_hz_exact`,
WP34's own function, reused unchanged), compared against WP32's
independently reconstructed partner bare frequency. For the Al27+/Mg25+
Marshall dataset the constrained fit gives `q=0.190645`,
`alpha=+1.625194` (outside `[0, 1]`; nothing in the Laplace constraint
requires it to stay inside), `a_x=-5.821e-03`, `a_y=+2.239e-03`, with
frequency residuals `-782.7 Hz` (X-COM, `-0.019%`), `+761.9 Hz` (X-STR,
`+0.022%`), `-3862.1 Hz` (Y-COM, `-0.072%`), `+3721.0 Hz` (Y-STR,
`+0.078%`): small but genuinely nonzero, quantifying whatever real
physics (trap anharmonicity, higher multipoles, or any other departure
from the idealized Mathieu model) this two-parameter fit cannot absorb.
The partner check lands at `-0.07%`/`-0.11%` (X/Y), tighter than WP34's
own `-0.45%`/`-0.40%`. The four radial per-mode ratios move to
`1.02/0.99/1.02/0.99` (X-COM/X-STR/Y-COM/Y-STR), and the total to
`-1.1415e-17`.

**The three-component uncertainty budget (WP35 Part 2).** THREE separately
labeled components: thermometry (`+/-3.763e-19`, the existing per-mode
`n_bar` channel), input rounding (`+/-1.022e-20`, WP34 Part 2's channel,
propagated through the constrained-fit chain), and model structure
(`+/-1.591e-19`, new here, `|constrained total - per-axis-diagnostic
total|`, G17 item 3's own "at minimum" bound): the constrained fit and
the per-axis diagnostic model the SAME physics two different ways, and
their spread is a direct measure of how much the reported agreement
depends on which of the two defensible models is used. Combined
in quadrature, `+/-4.087e-19`; the total lands `0.08` sigma from
Marshall's published `-114.6(3.8)e-19`: `MET`, inside the published band
even with all three components counted. The per-axis diagnostic case is
kept, unchanged, as its own labeled variant (`run_motional_al_ion_
coupled_floquet_case`): its own total, `-1.157e-17`, also lands `MET`
(`0.20` sigma against thermometry and rounding alone, no model-structure
term of its own to add, since it IS one of the two models being
compared). The same constrained-fit treatment applied to Brewer et al.'s
independently published trap gives residuals of a similar relative size,
partner deviations `-0.18%`/`-0.14%`, and four radial ratios of
`0.95/0.95/0.96/0.97`, the same qualitative pattern in a second,
independent dataset. This result is classified `arithmetic_reproduction`
(an inversion against, and comparison to, published inputs, the same
binding classification WP30-34 already carry). `reproducibility` and
`blind_prediction` are stronger classes this case does not claim.


## 17. Lattice light shift: two community models (v1.12.0, WP36 Phase 1)

Motivation (project owner): lattice optical clocks need the light shift
from their own confining lattice modeled and cancelled, the dominant
systematic after BBR for a well-shielded Sr/Yb lattice clock. Two models
are in active community use side by side. Bothwell et al. 2025's own
words state this directly: "we elect to perform an additional analysis
using a Born-Oppenheimer+WKB treatment... which better captures
axial-radial couplings," a second treatment they run alongside their
harmonic-basis main result, keeping both. This section specifies BOTH
models faithfully and picks neither as a winner. **Scope boundary,
stated here and in the implementing module's own docstring:** this
section and `cliffordclock.integrator.lattice_light_shift` are functions
and benchmarks only. Neither model is wired into `cliffordclock.pipeline`'s
config surface in this phase; that wiring is later work, once both models
are validated on their own terms.

Implemented in `cliffordclock.integrator.lattice_light_shift`. Benchmarked
in `benchmarks/run_lattice_light_shift.py`.

### E40: the Katori-lineage harmonic/operational model

Provenance: Katori, Ovsiannikov, Marmo, Palchikov, "Strategies for
reducing the light shift in atomic clocks," PRA 91, 052503 (2015) (the
functional form's lineage: E1 polarizability, multipolar M1+E2 term,
hyperpolarizability, motional-state dependence); Ushijima, Takamoto,
Katori, "Operational magic intensity for Sr optical lattice clocks," PRL
121, 263202 (2018), arXiv:1812.11815 (the operationalized Eq. 1/Eq. 2 form
and the Table I coefficients this project reproduces). Every equation
below is transcribed verbatim from the typeset PDF (not from a search
snippet or an ar5iv AI summary), each fetched and read directly this
session.

**The light shift** (Ushijima et al. 2018 Eq. 1):

    h*nu_LS(u, delta_L, n_z) ~=
        [d(alpha~E1)/dnu * delta_L - alpha~qm] * (n_z + 1/2) * u^(1/2)
      - [d(alpha~E1)/dnu * delta_L + (3/2)*beta~*(n_z^2 + n_z + 1/2)] * u
      + 2*beta~*(n_z + 1/2) * u^(3/2)
      - beta~ * u^2

with `u = U/E_R` the reduced (peak) trap depth, `delta_L` the lattice-laser
detuning from the E1 magic frequency, `n_z` the axial vibrational quantum
number, `d(alpha~E1)/dnu` the E1-polarizability-difference slope, `alpha~qm`
the combined M1+E2 polarizability difference, `beta~` the
hyperpolarizability, all three already in the paper's own `.../h` (hertz)
convention, so `nu_LS` (not `h*nu_LS`) is what the implementation returns
directly. Implemented in `harmonic_light_shift_hz`
(`HarmonicLatticeCoefficients` carries the three coefficients plus their
1-sigma uncertainties and a citation string per instance).

**Radial-thermal reduction, TWO DISTINCT FORMS (do not interchange).**
Ushijima et al. 2018's own Eq. 2 defines a radial-thermal average of `u`'s
`j`-th power, approximated by the LINEAR form

    zeta_j(u) ~= 1 - j*kB*Tr/(u*E_R)     (Ushijima 2018 Eq. 2)

applied term-by-term (`u^(1/2) -> zeta_(1/2)(u)*u^(1/2)`, and so on for
`j = 1, 3/2, 2`). Kim, Aeppli, Bothwell, Ye, PRL 130, 113203 (2023) (main
text, transcribed directly: "use of an effective depth, `uj = (1 +
j*kB*Tr/(u0*Er))^-1 * uj0`") and Bothwell et al. 2025 (their Eq. 1 context,
identical form) instead use the EXACT RECIPROCAL,

    zeta_j(u) = (1 + j*kB*Tr/(u*E_R))^-1     (Kim 2023 / Bothwell 2025)

The two agree to leading order in `j*kB*Tr/(u*E_R) << 1` but are
algebraically different formulas from different papers in the same
lineage. `ushijima_reduction_factor` implements the first;
`jila_reduction_factor` implements the second; `harmonic_light_shift_hz`'s
`reduction_form` argument selects between them (or `"none"`, the bare
Eq. 1 with no radial folding, the form Ushijima et al. 2018 themselves use
to derive their own operational point, Eqs. 14-15).

**Operational point, computed by direct numerical solve.** Ushijima et al. 2018's own
`u_op = 72(2) E_R` at `delta_L_op = 5.3(2) MHz` is the simultaneous
solution of `nu_LS(u_op, delta_L_op, 0) = 0` and
`d(nu_LS)/du|_(u_op, delta_L_op, 0) = 0` (their Eqs. 14-15 give an
approximate closed form for this point; this project solves the FULL
Eq. 1 directly, the exact target their approximation estimates).
Because Eq. 1 is exactly linear in `delta_L` at fixed `u`, the two-variable
system reduces to one equation in `u` alone
(`solve_harmonic_operational_point`'s docstring gives the derivation);
solved by `scipy.optimize.brentq`. Reproduced: `u_op = 71.7` (published
`72(2)`), `delta_L_op = 5.29 MHz` (published `5.3(2) MHz`), both within the
published uncertainty (`benchmarks/run_lattice_light_shift.py` Target 1).

**Coefficient registry**, each entry's own citation carried in its
`HarmonicLatticeCoefficients.citation` field:

- `USHIJIMA_2018_SR87`: Sr-87, RIKEN, Table I (Target 1's own coefficients).
- `KIM_2023_SR87`: Sr-87, JILA, main text (Target 2 reuses these:
  Aeppli et al. 2024's own words, "identical atomic coefficients as in
  Ref. [19]", their Ref. 19 being Kim et al. 2023). Uses the JILA
  reciprocal reduction factor, not Ushijima's linear form.
- `BOTHWELL_2025_YB171_HARMONIC`/`_BOWKB`: Yb-171, NIST, Table III's two
  columns. Unlike the two Sr entries above, these coefficients are
  ALREADY normalized by the clock frequency (Bothwell's own Eq. 1: "we
  have divided the clock shift (`delta_nu_LS`) by the clock frequency
  (`nu_c`)"): `e1_slope_per_hz` is per-hertz-of-`nu_c` directly, and
  `m1e2_hz`/`hyperpolarizability_hz` are dimensionless fractions despite
  the field names (kept for API uniformity). `harmonic_light_shift_hz`
  is a pure coefficient-algebra evaluator with no unit assumption of its
  own, so it accepts this convention directly and returns the fractional
  `nu_LS/nu_c`, not `nu_LS` in hertz; callers must track which
  convention a given coefficient set uses, since the function cannot
  detect the mismatch on its own.

**Uncertainty propagation.** `harmonic_light_shift_uncertainty_hz`
propagates each of the three coefficients' own 1-sigma uncertainty through
a central finite-difference partial derivative, combined in quadrature via
`math.fsum` (this document's established compensated-summation
discipline). This is arithmetic-reproduction fidelity: it verifies the
formula is evaluated correctly against the coefficients supplied. E32's
own uncertainty note states the same scope for its registry coefficients,
whose accuracy is a separate, physics-level question this propagation
does not address.

### E41: the NIST Born-Oppenheimer+WKB (BO+WKB) model

Provenance: Beloy, McGrew, Zhang, Nicolodi, Fasano, Hassan, Brown, Ludlow,
"Modeling motional energy spectra and lattice light shifts in optical
lattice clocks," PRA 101, 053416 (2020), arXiv:2004.06224 (the full
construction); Bothwell, Hunt, Siegel, Hassan, Grogan, Kobayashi, Gibble,
Porsev, Safronova, Brown, Beloy, Ludlow, "Lattice light shift evaluations
in a dual-ensemble Yb optical lattice clock," PRL 134, 033201 (2025),
arXiv:2409.10782 (the practical evaluation form, Eq. 6, and the
harmonic-vs-BO+WKB comparison this project reproduces). Every equation
below is transcribed verbatim from the typeset PDF (`pdftotext -layout`
against the fetched arXiv PDF, cross-read against the running prose around
each equation to resolve column-layout ambiguity), not from an ar5iv AI
summary: the two equations flagged in this project's research dossier as
needing a verbatim re-pull (Eqs. 4 and 11) are transcribed below exactly
as they appear in the source.

**The site potential** (Beloy et al. 2020 Eq. 1, cylindrical coordinates
`(rho, phi, z)` about the lattice axis):

    U(rho, z) = -(E0/2)^2 * alpha_E1 * exp(-kappa^2*rho^2) * cos^2(k*z)

with `kappa = sqrt(2)/w` (`w` the lattice beam's `1/e^2` intensity
radius), `k = 2*pi/lambda`, peak depth `D = (E0/2)^2*alpha_E1`, and recoil
energy `E_R = hbar^2*k^2/(2m)` (`recoil_energy_j`, algebraically identical
to Ushijima's and Bothwell's own `E_R` definitions in wavelength/frequency
form; CONVENTIONS.md keeps one function for all three). Implemented in
`SitePotential`/`make_site_potential`.

**The key simplification this project's numerics exploit.** Because the
potential factors exactly as `-D*exp(-kappa^2*rho^2)*cos^2(kz)`, the axial
Born-Oppenheimer eigenproblem at any fixed `rho` (Beloy's Eq. 5) is
IDENTICAL in form to the on-axis (`rho=0`) problem with a rescaled local
depth `D(rho)/E_R = depth_er*exp(-kappa^2*rho^2)`
(`_local_depth_er`): one 1D finite-difference solver, called once per
`rho` grid point, replaces a genuine 2D solve.

**Axial Born-Oppenheimer separation** (Beloy et al. 2020 Eq. 5):

    [-hbar^2/(2m) d^2/dz^2 + U(rho,z)] Z_nz(rho,z) = U_nz(rho) Z_nz(rho,z)

solved numerically by finite-difference diagonalization on the
dimensionless domain `x = k*z in [-pi/2, pi/2]` (one lattice site,
Dirichlet boundary conditions, `Beloy`'s own Eq. 13 integration domain,
justified by the deep-lattice/negligible-tunneling assumption both this
paper and Ushijima et al. 2018 make): kinetic operator `-d^2/dx^2` with
coefficient exactly `1` in `E_R` units, potential `-D(rho)/E_R*cos^2(x)`,
tridiagonal matrix, `scipy.linalg.eigh_tridiagonal`. Implemented in
`_axial_fd_solve`/`axial_energies_er`; EVERY call carries an explicit
convergence guard (grid resolution doubled until the lowest `n_states`
eigenvalues change by less than a stated tolerance between successive
resolutions, or `LatticeLightShiftConvergenceError` is raised naming the
residual: `AXIAL_GRID_N0=161` up to `AXIAL_GRID_N_MAX=40961`,
`AXIAL_ENERGY_TOL_ER=1e-5`).

**Harmonic-limit consistency check (this project's own numerical
verification, mirroring Beloy et al. 2020's own Section VI).** Feeding the
SAME finite-difference solver Beloy's Eq. 2 harmonic potential
(`potential="harmonic"`), in place of the true `cos^2` site potential,
recovers the exact 1D quantum-harmonic-oscillator spectrum
`E_n/E_R = 2*sqrt(D/E_R)*(n+1/2)` to the solver's own convergence
tolerance (`tests/test_lattice_light_shift.py`): at `D/E_R=50`, `n=0,1,2`,
the finite-difference solve gives `-42.9289, -28.7867, -14.6438`
against the closed-form `-42.9289, -28.7868, -14.6447`. The TRUE `cos^2`
potential's ground state at the same depth is `-43.19 E_R`, deeper than
the harmonic approximation (the physically correct direction: `cos^2(x) ~=
1 - x^2 + x^4/3` near the well bottom, an ATTRACTIVE quartic correction on
top of the harmonic term, not the repulsive/softening quartic a reader
might expect by analogy to other trap potentials).

**WKB radial quantization** (Beloy et al. 2020 Eqs. 8-9):

    phi_l,nz(E) = sqrt(2m/hbar^2) * integral_R sqrt(E - U_nz(rho) - hbar^2*l^2/(2m*rho^2)) d(rho)   (Eq. 8)
    phi_l,nz(E_nrho,l,nz) = pi*(nrho + 1/2)                                                          (Eq. 9)

(the subscript `R` on the integral restricting to the region where the
integrand is real). This project does not evaluate the phase integral or
its quantization condition directly (Eq. 9's role is superseded, for this
project's purposes, by the density-of-states route below, Beloy's own
Eqs. 10-11 derivation).

**Density of states, Eq. 4 (harmonic closed form) and Eq. 11 (general),
BOTH transcribed verbatim from the typeset PDF, resolving the dossier's
flagged extraction gap:**

    G^HO_nz(E) = (kappa/k)^(-2) / (4*D*E_R) * [E + D - 2*sqrt(D*E_R)*(n_z+1/2)]     (Eq. 4)

    G_nz(E) = (1/4) * (2*m/hbar^2) * [R_nz(E)]^2                                    (Eq. 11)

with `R_nz(E)` the classical turning radius (`Unz(Rnz(E)) = E`, `Rnz` the
inverse function of `Unz`, stated directly below Beloy's Eq. 10). Eq. 4 is
implemented in `harmonic_density_of_states_closed_form`; Eq. 11 in
`bo_wkb_density_of_states`, via `turning_radius_m` (bracket-then-`brentq`
root find on the tabulated `U_nz(rho)`). Algebraic AND numerical
consistency confirmed (`tests/test_lattice_light_shift.py`): feeding the
closed-form harmonic `R_nz(E)^2` directly into Eq. 11's formula recovers
Eq. 4 to floating-point precision; running the FULL numerical machinery
(finite-difference axial solve, turning-radius root find) with
`potential="harmonic"` recovers Eq. 4 to better than `1e-4` relative error
across the checked `(n_z, E)` grid, the cross-check Beloy et al.
2020 themselves perform in their own Section VI.

**Thermally-averaged trap-depth-reduction factors** (Beloy et al. 2020
Eq. 21, the per-`n_z`-isolated form of their Eqs. 19-20, transcribed
verbatim):

    X_nz = integral_0^Rnz(0) [x_nz(rho)*rho*(exp(-Unz(rho)/kB*Tr) - 1)] d(rho)
           / integral_0^Rnz(0) [rho*(exp(-Unz(rho)/kB*Tr) - 1)] d(rho)

(analogous forms for `Y_nz`/`Z_nz` with `y_nz(rho)`/`z_nz(rho)` in the
numerator, same denominator), where `x_nz(rho)`, `y_nz(rho)`, `z_nz(rho)`
are Beloy's Eq. 13 dimensionless axial-overlap shape factors
(`e^(-kappa^2*rho^2)` times the axial eigenfunction's own `cos^2(kz)`/
`sin^2(kz)`/`cos^4(kz)` expectation value at that `rho`), evaluated
directly from the finite-difference eigenvector. Implemented in
`axial_thermal_factors`.

**Numerical stability (an explicit, documented reformulation that
preserves the exact value of Eq. 21's ratio).** `Unz(rho)` can be tens of `E_R` in magnitude at
`rho=0`; evaluating `exp(-Unz(rho)/kB*Tr)` directly overflows float64 for
realistic deep, cold lattices (`|Unz(0)|/(kB*Tr)` routinely exceeds 700).
`axial_thermal_factors` factors out the common, dominant exponential
`exp(-Unz(0)/kB*Tr)` (identical in numerator and denominator, so it
cancels in the ratio and is never computed explicitly) and integrates the
shifted, bounded integrand `exp(-(Unz(rho)-Unz(0))/kB*Tr) -
exp(Unz(0)/kB*Tr)` instead, which is `<= 1` everywhere on the integration
domain by construction. Carries its own convergence guard (radial grid
points and the shared axial finite-difference resolution both doubled
until `X`/`Y`/`Z` each stabilize within `THERMAL_FACTOR_TOL=1e-4`, or
`LatticeLightShiftConvergenceError` is raised).

**Validation: reproduces Bothwell et al. 2025's own published BO+WKB
column to better than 0.1 percent, all four table rows.** Bothwell et al. 2025's Appendix A
Table I publishes `X`/`Y`/`Z` (harmonic AND BO+WKB columns) at four
`(u0, Tr)` points for Yb-171, `n_z=0`. `axial_thermal_factors` reproduces
the BO+WKB column at all four points to better than `1e-3` relative error
(`benchmarks/run_lattice_light_shift.py` Target 3a):

| u0 (E_R) | Tr (nK) | X pred / pub | Y pred / pub | Z pred / pub |
|---|---|---|---|---|
| 56.8 | 650 | 0.7855 / 0.785 | 0.0608 / 0.0608 | 0.6455 / 0.645 |
| 66.4 | 550 | 0.8378 / 0.838 | 0.0580 / 0.058 | 0.7187 / 0.719 |
| 86.2 | 600 | 0.8643 / 0.864 | 0.0515 / 0.0515 | 0.7588 / 0.759 |
| 112.2 | 720 | 0.8786 / 0.879 | 0.0454 / 0.0454 | 0.7813 / 0.781 |

This is the strongest single validation available in this work package:
an independent published cross-check table, reproduced end to end (real
finite-difference axial solve, real WKB-turning-radius density of states,
real thermal averaging) with zero fitted parameters. (An earlier attempt
at this same table used Sr-87's mass and wavelength in place of
Yb-171's, disagreed with the published BO+WKB column by 5-15%, and
matched the harmonic column more closely than the BO+WKB one. The cause
was that species mismatch: `X`/`Y`/`Z` cancel atomic mass and
lattice waist exactly out of their defining ratio, so neither one
matters, but the species' own recoil energy `E_R` still enters through
the `kB*Tr/E_R` thermal-weighting ratio, so using the wrong species'
`E_R` for a published `(u0, Tr)` pair silently evaluates a different
physical trap depth than the table intends. Recorded here as the kind of
provenance trap this document's discipline exists to catch.)

**Light-shift evaluation** (Bothwell et al. 2025 Eq. 6, transcribed
verbatim, specialized to a single dominant band `W_nz=1`):

    delta_nu_LS/nu_c ~= -[ (d(alpha~E1)/dnu)*delta_L*X(n_z,u0,Tr)*u0
                          + alpha~M1E2*Y(n_z,u0,Tr)*u0
                          + beta~*Z(n_z,u0,Tr)*u0^2 ]

using the SAME three polarizability coefficients Model A tabulates (in
Bothwell's own already-`nu_c`-normalized convention, `BOTHWELL_2025_YB171_HARMONIC`/
`_BOWKB`), weighted by the BO+WKB `X`/`Y`/`Z` factors, a direct WKB-derived
weighting that replaces the harmonic model's classical-thermal folding
factor. Implemented in `bo_wkb_fractional_light_shift`.

**Beloy et al. 2020's own conclusion, carried forward here as this
project's own explicitly stated scope limit:** the BO+WKB model has no
analytic lineshape to fit sideband/Doppler/carrier spectroscopy against,
unlike the harmonic model's perturbative fitting protocols. A WKB-native
fitting procedure is a genuine, separately-scoped future capability;
every case in `benchmarks/run_lattice_light_shift.py` evaluates the model
at STATED operating conditions from a paper's own text, the only
evaluation mode this phase implements.

### Reproduction targets and their classification

Per this project's established evidentiary-class discipline (arithmetic
reproduction: zero fitted parameters, a closed-form or direct-formula
evaluation against a paper's own published inputs, compared against that
paper's own published result):

- **Target 1** (E40): Ushijima et al. 2018's `u_op=72(2) E_R` at
  `delta_L_op=5.3(2) MHz`. `arithmetic_reproduction`. MET (both within
  published uncertainty).
- **Target 2** (E40): Aeppli et al. 2024's `-0.1(3.2)e-19` lattice-light
  budget line at `15.06(17) E_R`, `Tr~=120 nK`. `arithmetic_reproduction`.
  MET (predicted `-0.056e-19 +/- 2.22e-19`, band overlaps).
- **Target 3a** (E41): Bothwell et al. 2025's own Table I harmonic-vs-BO+WKB
  `X`/`Y`/`Z` comparison, all four rows. `arithmetic_reproduction`. MET
  (worst relative error `8.8e-4` against a `1%` tolerance).
- **Target 3b** (E41): Bothwell et al. 2025's headline `alpha~M1E2 =
  -1.41(9)e-18` (harmonic) vs. `-1.45(8)e-18` (BO+WKB), Table III.
  **`computable_comparison`, explicitly NOT `arithmetic_reproduction`**:
  these numbers are outputs of the paper's own nonlinear fit against raw,
  unpublished scan data, not a closed-form function of any published
  input. What this project computes instead: both models evaluated at the
  paper's own stated operating conditions (`u0=100 E_R`, comfortably
  inside their stated `<140 E_R` range, `Tr=600 nK`, `n_z=0`, on-magic
  detuning), using each model's own published coefficient column, with the
  resulting model-difference reported (`+2.16e-18` fractional at this
  point; the two models' own `alpha~M1E2` coefficients differ by `2.84%`,
  a separate, narrower comparison than the full-shift difference, kept as
  its own distinct number alongside it).
- **Density-of-states contrast** (E41): both models' cumulative
  axial-band-0 (`n_z=0`) radial state count from the band bottom to one
  thermal quantum above it, Yb-171 at `u0=100 E_R`, over
  `Tr in {50, 100, 200, 400, 800, 1600} nK`. The BO+WKB/harmonic ratio
  grows from `1.05` to `1.12` over this range: a direct computation, at
  this project's own chosen conditions, of the dossier's qualitative
  claim that the true site potential's radial degeneracy grows faster
  than the harmonic potential's as radial temperature rises, grounded in
  the papers' own equations.

### E41 addendum: a differentiable JAX implementation (v1.13.0, WP37)

Phase 2's spectrum fitting needs gradients of the light shift with
respect to its physical inputs, and `jax.grad` cannot trace the adaptive,
data-dependent convergence loops E41's own reference implementation
uses. `cliffordclock.integrator.lattice_light_shift_jax` reimplements
this section's BO+WKB chain (the axial separation, the WKB radial
quantization and Eq. 11 density of states, Eq. 21's `X`/`Y`/`Z` factors,
and Eq. 6's light shift) in `jax.numpy`, differentiable end to end and
compatible with `jax.jit`. No new physics: every formula is the one this
section already specifies, evaluated by a different numerical route.
Model A's closed-form Eq. 1 is ported alongside it (pure coefficient
algebra, no eigenproblem), so the differentiable module carries both
community models.

**Fixed resolution in place of adaptive convergence.** The reference
module's own numerics double a grid resolution until a convergence guard
is satisfied. That data-dependent loop length is what
`jax.jit`/`jax.grad` cannot trace, so this module fixes the axial
finite-difference grid at 1281 points and the radial quadrature at 321
points instead. An offline convergence study
(`tests/test_lattice_light_shift_jax.py::TestOfflineConvergenceStudy`),
run once against the reference module's own converged output, chose this
resolution and verified it directly. The study found the reference
module's own default convergence guard settles at axial grid 1295 and
321 radial points at all four of Bothwell et al. 2025's Table I points;
matching that resolution reproduces the reference's `X`/`Y`/`Z` to
better than `1.57e-7` relative, worst case (`Y`, `u0=112.2 E_R`),
comfortably inside this module's `1e-6` agreement bar.

**The turning-radius root-find** (Eq. 10's `Rnz(E)`, the inverse of
`Unz(rho)`) is differentiated via `jax.lax.custom_root`'s implicit-
function-theorem path around a fixed-iteration-count bisection: the
closed-form-bracketed formulation this work package's own instructions
name directly. A fixed-iteration bisection supplies the root's numeric
value; the implicit function theorem supplies its gradient from the
root-finding function's own derivative there. See the module's own
docstring for the derivation and for a kink the reference module's
"clamp an unbound state's energy to `0.0`" convention introduces at the
root-find's `E=0` endpoint, and the unclamped-eigenvalue fix this
module's own tests found necessary to keep `jax.grad` finite there.

**Validated at the four G18 table points, Yb-171.** `jax.grad` of the
light shift with respect to `u0` and `Tr` matches central finite
differences of the REFERENCE implementation at the same four points:
independent numerical methods on each side, the strongest available
check. The worst-case agreement is `4.9e-8` relative, four orders of
magnitude inside the `1e-4` gate requirement. The forward evaluation
jit-compiles and returns bitwise-identical output across repeated calls
on the same inputs.

Scope, unchanged from this section's own: no spectrum or lineshape model
(WP38, waiting on a research round this work package does not do), no
pipeline wiring. See
`cliffordclock/integrator/lattice_light_shift_jax.py`'s own module
docstring for the full derivation and
`tests/test_lattice_light_shift_jax.py` for the agreement, gradient,
jit-determinism, and offline convergence-study tests.

## 18. Sideband-spectrum forward model (v1.14.0, WP38 Phase 2)

Motivation (project owner, following the project's internal lattice-
light-shift research dossier's Phase 2 addendum): E41's BO+WKB model has
no analytic lineshape to fit sideband spectroscopy against (Beloy et al.
2020's own stated conclusion, carried forward at E41's end above); the
field's thermometry is instead done in the harmonic model (E40's own
Blatt-lineage machinery) and fed into a separately-evaluated,
better-justified light shift, a stacked model-mismatch error Goti et al.
2025 quantifies directly ("discrepancies up to a factor of two in
extracted temperatures... relative frequency deviations up to 8e-17").
This section specifies the differentiable sideband-spectrum forward
model that closes that gap: gradients of the clock-transition excitation
spectrum with respect to trap depth and temperature, through BOTH the
harmonic validation anchor and the BO+WKB capability, on the same JAX
core E41's addendum already established.

Implemented in `cliffordclock.integrator.sideband_spectrum_jax`.
Cross-validated in `benchmarks/run_sideband_spectrum.py`. Fitting
demonstration in `benchmarks/run_sideband_fit.py`.

### E42: the sideband-spectrum forward model, two paths

Provenance: Blatt, Thomsen, Campbell, Ludlow, Swallows, Martin, Boyd, Ye,
"Rabi spectroscopy and excitation inhomogeneity in a one-dimensional
optical lattice clock," PRA 80, 052703 (2009), arXiv:0906.1419 (the
harmonic path: every equation transcribed verbatim from the typeset PDF,
all 12 pages read directly, page image by page image); Goti, Petrucciani,
Condio, Levi, Calonico, Pizzocaro, "Atomic thermometry in optical lattice
clocks," arXiv:2508.08164 (v2, 2 Sept 2025) (the BO+WKB path: Eqs. 1-9
transcribed verbatim, the first 8 pages of the typeset PDF read
directly).

**The shared harmonic-oscillator motional spectrum** (Blatt Eq. 3; Goti
Eq. 1, the identical formula, `h` factored out):

    E_{nx,ny,nz}/h ~= nu_z*(nz+1/2) + nu_r*(nx+ny+1)
                       - (nu_rec/2)*(nz^2+nz+1/2)
                       - nu_rec*(nu_r/nu_z)*(nx+ny+1)*(nz+1/2)

with `nu_z = 2*nu_rec*sqrt(u0)` (Blatt Eq. 4), `nu_r =
sqrt(U0/(m*pi^2*w0^2))` (Blatt Eq. 5), `nu_rec = E_R/h`. The longitudinal
energy gap (Blatt Eq. 8; Goti Eq. 2, identical), generalized to the
combined radial quantum number `n_r = n_x+n_y`:

    gamma(nz) = nu_z - nu_rec*(nz+1) - nu_rec*(nu_r/nu_z)*(n_r+1)

Implemented in `blatt_trap_frequencies_hz`, `longitudinal_energy_hz`,
`blue_sideband_detuning_hz` (this formula), `red_sideband_detuning_hz`
(the algebraic symmetry `-gamma(nz-1)`, derived directly from the same
Eq. 3/Eq. 1; neither paper numbers this symmetry separately).

**Path A, the harmonic validation anchor.** The carrier
(`harmonic_carrier_excitation_probability`, Blatt Eqs. 13-20,
transcribed verbatim): Rabi frequency `Omega_{nx,nz} = Omega_0 *
e^(-eta_z^2/2) * e^(-eta_x^2/2) * L_{nx}(eta_x^2) * L_{nz}(eta_z^2)`
(Eq. 14, `L_n` the physicists' Laguerre polynomial, evaluated by the
standard three-term recurrence, `laguerre_values`, since `jax` ships no
built-in generalized Laguerre), the single-motional-state Rabi-flopping
probability `p_e(n,delta,t) = [Omega^2/(Omega^2+delta^2)] *
sin^2[pi*t*sqrt(Omega^2+delta^2)]` (Eq. 17), and the thermally-averaged
`P_e(delta,t) = sum_{nx,nz} q_{nx}(Tr)*q_{nz}(Tz)*p_e(n,delta,t)` (Eq.
18, Boltzmann weights Eqs. 19-20). The sideband
(`harmonic_sideband_shape`, Blatt Appendix A Eqs. A1-A2; Goti Eq. 4's
population weight, restated from the same Blatt appendix): a
population-normalized sum of power-broadened Lorentzians, `shape(delta)
= sum_{nz,nr} w(nz,nr) / (1 + [(delta-detuning(nz,nr))/gamma]^2)`, `w`
from Goti Eq. 4 (`(nr+1)*exp[-h*nu_r*(nr+1)/(kB*Tr)] *
exp[-goti_e00_hz(nz)/(kB*Tz)]`, `goti_e00_hz` Goti Eq. 1 at `nx=ny=0`).
This is the FULL `(nz, nr)` sum (Eqs. A1-A2). Eqs. A3-A5 go on to reduce
that sum further, to the shallow-sideband-edge slope alone; this
project keeps the full sum, since a differentiable forward model useful
for fitting an entire sideband needs the full lineshape. Blatt et al.
2009's own text
following their Eq. 12, "there is no contribution from the longitudinal
ground state to the red sideband," is implemented directly: the red
sideband's own `n_z=0` population weight is zeroed before normalizing.

**Path B, the BO+WKB capability.** Goti et al. 2025's Eqs. 5-9,
transcribed verbatim:

    delta_nu = [U_{nz'}(rc) - U_{nz}(rc)] / h,  rc = R_{nz}(E)        (Eq. 5)
    sigma_blue(delta) proportional-to
        sum_{nz} integral_E G_{nz}(E)*p_{nz}(E)
                 / (1+[(delta-delta_nu(E))/gamma]^2) dE                (Eq. 8)
    p_{nz}(E) proportional-to
        exp[-(E-U_{nz}(0))/(kB*Tr)] * exp[-U_{nz}(0)/(kB*Tz)]          (Eq. 9)

with `G_{nz}(E) = (m/(2*hbar^2))*[R_{nz}(E)]^2` (Goti Eq. 8's own density
of states, algebraically identical to Beloy et al. 2020 Eq. 11, already
implemented as `lattice_light_shift_jax.bo_wkb_density_of_states_jax`).
Implemented in `bowkb_sideband_shape`, via a NEW numerical route:
`build_band_energy_table` precomputes each needed axial band's
`U_nz(rho)/E_R` on a small, fixed radial grid (one batched `jax.vmap`
over `jax.numpy.linalg.eigh`, `AXIAL_GRID_N_SPECTRUM=321` axial points,
`RHO_TABLE_N=129` radial points), and `condon_point_m`/`condon_detuning_hz`
find the classical turning radius/Franck-Condon point by
`jax.numpy.interp` against that table. `lattice_light_shift_jax.turning_radius_m_jax`
instead finds the same physical quantity through per-energy bisection.
This turns roughly `N_z*N_E*BISECTION_ITERS` dense eigensolves per
spectrum call into roughly `N_z*RHO_TABLE_N`, the resolution/tractability
trade this section's own module docstring derives in full (a spectrum
needs many Franck-Condon evaluations per call; a single light-shift
evaluation, E41's own use case, needs one turning radius per `(nz, Tr)`
pair). The integration domain evaluates the EXACT target-band-boundedness
condition directly (`U_{nz'}(rc) <= 0`), masked on the same fixed energy
quadrature grid every query `delta` reuses; Eq. 8 itself approximates
that same condition through an `E_max=-h*delta` cutoff, valid, their
own words, "for a deep vertical lattice."

**A known, stated resolution limit near the band top.** This section's
own cross-validation (`benchmarks/run_sideband_spectrum.py`'s tier 2)
excludes points near the band top from its tolerance check and reports
them separately, with a flag every reader can find in that check's own
rows; folding those points into one looser tolerance would hide the
reason for the difference. Within about 5 `E_R` of a band's top (`E -> 0`, the classical turning
radius growing without bound as the local trap depth vanishes), the
finite, linearly-spaced radial table's linear interpolation loses
accuracy fast, and `bowkb_sideband_shape`'s own energy quadrature keeps
a `2%`-of-band-depth margin below `E=0` for this reason (see that
function's own docstring and the benchmark case's own docstring for the
measured numbers).

**No BO+WKB carrier formula exists in either paper.** Both
`harmonic_full_spectrum` and `bowkb_full_spectrum` share the SAME
carrier component (`harmonic_carrier_excitation_probability`): neither
Beloy et al. 2020 nor Goti et al. 2025 proposes a distinct BO+WKB
carrier treatment, and the Lamb-Dicke-regime carrier is dominated by the
ground axial band, where the harmonic and true `cos^2` potentials
already agree closely (E41's own G18-gated harmonic-limit consistency
check).

**Amplitude convention.** Neither paper fixes one. Both state their own
sideband cross section `\propto` (proportional to), a shape with a
scale left open; Blatt et al. 2009's own Fig. 2 fits the carrier and
each sideband with independently fitted amplitudes.
`harmonic_sideband_shape`/`bowkb_sideband_shape` return a
population-normalized shape bounded `[0, 1]`; `harmonic_full_spectrum`/
`bowkb_full_spectrum` take explicit `blue_amplitude`/`red_amplitude`
scale arguments, mirroring that same per-feature fitting practice.

### Cross-validation against an independent implementation (Deliverable 2)

`benchmarks/run_sideband_spectrum.py` cross-validates against
`large-lattice-model` (github.com/inrim/large-lattice-model, MIT
license, (c) 2021-2024 Marco Pizzocaro, INRIM), a real, public,
third-party implementation of Beloy et al. 2020's model. It solves the
axial eigenproblem with EXACT Mathieu-function characteristic values, a
different numerical method from this project's own finite-difference
solver, and is cited as reference [50] in Goti et al. 2025's own Fig.
4/Fig. 7 fits. This project reads only that repository's numeric
OUTPUT, generated once in a separate environment
(`benchmarks/fixtures/wp38_inrim_large_lattice_model_reference.json`,
commit hash pinned in that fixture's own `provenance` field); no code
from that repository enters this project. This comparison earns a NEW
evidentiary class, `independent_implementation_reproduction`. This
project's established `arithmetic_reproduction` class is reserved for a
PAPER's own published number; this new class reproduces an independent
CODE implementation's output at matched inputs.

Three tiers, tightest to loosest:

1. **Band-bottom eigenvalue** (`independent_implementation_reproduction`):
   this project's ALREADY G18-gated finite-difference solver
   (`lattice_light_shift.axial_energies_er`) vs. large-lattice-model's
   exact Mathieu-characteristic-value `U(0,D,nz)`, at `D in {56.8, 80,
   100, 150} E_R`, `nz in {0,1,2,3}`. Worst relative error `1.06e-7`
   (tolerance `1e-4`). **MET.**
2. **Franck-Condon detuning** (`independent_implementation_reproduction`):
   this WORK PACKAGE's own `condon_detuning_hz` (spectrum-scale
   resolution) vs. large-lattice-model's `DeltaU(R(E,D,0),D,0,1)`.
   Worst relative error `4.7e-3` (tolerance `2e-2`), excluding points
   within `5 E_R` of the band top (the resolution limit noted above;
   those points carry their own `near_band_top` flag in the full row
   listing). **MET.**
3. **Full sideband shape** (`computable_comparison`): the two sides use
   different, documented lineshape conventions (see the benchmark's own
   `CONVENTION_BRIDGES` for the Lorentzian peak-height factor of `2`,
   the state-dependent vs. fixed linewidth, and the integration-domain
   cutoff). This project's `computable_comparison` class covers a
   comparison bridged across a convention gap like this one. Peak
   positions agree within
   `500 Hz` on a `~33 kHz` sideband (`1.5%`), and shape correlation is
   `>= 0.93` across both moderate-depth conditions checked.

### Fitting demonstration (Deliverable 3)

`benchmarks/run_sideband_fit.py` runs a synthetic round-trip: the same
forward model generates the spectrum and fits it back
(`generator == fitter`), across a small, fixed grid of truth `(u0, Tr)`
pairs and deterministic noise seeds (`numpy.random.default_rng(seed)`).
Both parameters are fit back by `scipy.optimize.minimize` (`L-BFGS-B`,
`jac=True`) supplied EXACT gradients from `jax.value_and_grad` of the
forward model, with Laplace/Hessian-based 1-sigma uncertainties
(`jax.hessian` of the negative log-likelihood at the optimum, checked
for positive definiteness with `np.linalg.eigvalsh` before inversion).
12/12 fits (6 grid points, harmonic and BO+WKB paths) converged, and
recovered parameters land within their own reported 2-sigma uncertainty
in 11/12 cases. The one exception, harmonic path, `u0=100`, `seed=0`,
lands outside for a specific reason: `L-BFGS-B` stops at a SADDLE point
of the negative log-likelihood there, with Hessian eigenvalues
`[-8.72, 2.636e13]`, one negative. The Laplace approximation requires a
positive-definite Hessian, so it is invalid at that optimum.
`run_sideband_fit.py`'s own `hessian_positive_definite` flag catches
this directly (`False` for that one case), and both artifacts (JSON and
Markdown) report its uncertainty as `nan`, with the Markdown table
flagging the row by name.

**Stated at its calibration.** This demonstrates the first
GRADIENT-based (autodiff) fit of a BO+WKB-class sideband lineshape.
`large-lattice-model`'s own `fit.py` (`get_fit_sidebands`) already
supplies a working, non-differentiable fitter (a numba-jitted forward
model paired with a finite-difference-Jacobian `scipy.optimize`
routine), and Goti et al. 2025 used that code to fit real IT-Yb1
spectroscopy (Figs. 4, 7).

**The Goti et al. 2025 real-scan fit, assessed here.** Figs. 4 and 7
plot real sideband scans as discrete scatter markers, the strongest
figure-digitization candidate found across both research sweeps. This
project's own check of the underlying PDF's text/vector layer found the
paper's prose and equations, and no separately recoverable per-marker
coordinate stream. Extracting exact coordinates from these figures
would need pixel-level digitization of the published art, placing that
extraction in the figure-digitization class: weaker evidence than
either this section's synthetic fits or the independent-implementation
cross-validation above. This work package's own instruction says
plainly: do not force it. No real-scan fit is shipped; the raw scan
data behind those figures is recorded as the named partnership ask,
should INRIM be approached.

## 19. Rydberg vapor-cell response: quadratic Stark shift and EIT/Autler-Townes observable (v1.15.0, WP39 Phase A)

Motivation (project owner): Rydberg-atom RF electrometry and sensing
groups each hand-roll the same chain (a field over a vapor-cell atom
region, a per-atom Rydberg Stark response, and the resulting EIT/AT
spectrum) with no open tool providing it end to end. This section adds
that chain's atom-side half: field in, per-atom Rydberg shift, EIT/AT
spectrum out. The EM side (cell and waveguide simulation) stays with the
user's own tools; this project keeps its standing posture of consuming
field exports rather than solving electrostatics itself, the same
posture `cliffordclock.fields.io` already carries for FEA-exported field
grids.

Species and states: Rb-85, the 5S1/2-5P3/2-32D5/2-33P3/2 ladder. The
anchor is Holloway, Gordon, Jefferts, Schwarzkopf, Anderson, Miller,
Thaicharoen, Raithel, "Broadband Rydberg Atom-Based Electric-Field Probe
for SI-Traceable, Self-Calibrated Measurements," IEEE Trans. Antennas
Propag. 62, 6169 (2014), arXiv:1405.7066, chosen over the Sedlacek et
al. 2012 candidate (Nature Physics 8, 819 (2012), arXiv:1205.4461)
because its Fig. 15 prints three (splitting, field) calibration pairs
directly, with no plot digitization required. Every equation and figure
cited below was read from the paper's own arXiv PDF text or page image
this session. Implemented in
`cliffordclock.integrator.rydberg_cell_response`. Benchmarked in
`benchmarks/run_rydberg_cell_response.py`.

### E43: quadratic Stark shift of a single Rydberg state

**The formula and its sign.** `Delta_f = -(1/2) * alpha0 * E^2 / h`
(Yerokhin, Buhmann, Fritzsche, Surzhykov, Phys. Rev. A 94, 032503
(2016), arXiv:1608.04515, Eq. 5), `alpha0` the state's scalar
polarizability in atomic units (a0^3), `E` the local DC field. This is
the same sign and prefactor convention this project already uses for
`cliffordclock.ensemble.species`'s DC-Stark term (E14b), so E43 reuses
`ALPHA_AU_TO_SI` (`4*pi*eps0*a0^3`) rather than re-deriving that
conversion a second time. Implemented in
`rydberg_quadratic_stark_shift_hz`.

**Unit conversion, shown explicitly (dossier risk 3).** O'Sullivan and
Stoicheff, Phys. Rev. A 31, 2718 (1985) and Phys. Rev. A 33, 1640
(1986), publish `alpha0` directly in MHz/(V/cm)^2, a measured frequency-
shift-per-field-squared coefficient. Yerokhin et al. 2016 and this
section's registry both use the atomic-unit (a0^3) convention.
`alpha0_au_to_mhz_per_vcm2` converts between the two, derived in five
steps from constants already pinned in `cliffordclock.constants` and
`cliffordclock.ensemble.species.ALPHA_AU_TO_SI` (its own docstring shows
all five); a hand-computed value (alpha0 = 1e10 a.u. gives
k = -1.244159 MHz/(V/cm)^2) is pinned in
`tests/test_rydberg_cell_response.py::TestUnitConversion`.

**Registry, two independent sources per tabulated state.** Yerokhin et
al. 2016 Table IV cross-tabulates two independent Rb-85 nD5/2 alpha0
sources at n = 30, 35, 50: their own Dirac-Fock + core-polarization
(DFCP) theory, and O'Sullivan and Stoicheff's measured values (their
refs. [32],[33]). The two agree to within 2.9% across all three states
(`benchmarks/run_rydberg_cell_response.py`'s C4 case), inside the
dossier's own stated "1-5% level" and this check's 5% tolerance.

**The calibration state, 32D5/2, is not itself tabulated.** Holloway et
al. 2014's Fig. 15 calibration uses 32D5/2, four principal quantum
numbers below the nearest tabulated row (n=30, 35, or 50, whichever is
closest to the field regime a caller needs). Rather than forcing a false
match between the calibration state and a cross-checked polarizability
state, this registry carries two different, individually well-
provenanced n: the tabulated n=30/35/50 rows above for the C4 cross-
check, and a derivation-based `alpha0(32D5/2)` for the Stark term
`compose_inhomogeneous_eit_spectrum` actually evaluates.
`derive_rb85_32d52_alpha0_au` fits `alpha0(n_star) = C * n_star^p` in
log-log space to the three tabulated rows (the Rydberg-series scaling
Gallagher, Rydberg Atoms, Cambridge Univ. Press, 1994, sec. 2.4, gives
for a diagonal scalar polarizability), separately for the theory and the
experiment rows, and averages the two fits' predictions at
`n_star(32D5/2)`. Both fits reproduce their own three inputs to better
than 4% and land within 1% of each other at n=32
(`tests/test_rydberg_cell_response.py::TestC4PolarizabilityKA`); the
fitted exponent (6.50-6.53) sits close to the n^7 scaling law, the
expected order of magnitude. This is a fitted, derivation-based number,
not a value printed in either source, and every docstring and citation
in the code says so explicitly.

**Validity guard.** The quadratic (isolated-state) treatment breaks down
once the field approaches the first avoided crossing with the n-1
manifold. O'Sullivan and Stoicheff 1985 fit this crossing field
explicitly for Rb-85 nS states (`E_crossing(V/cm) = 4.638e8/n_star^5 +
1.528e10/n_star^7`), but this project could not obtain the equivalent
nD-specific fit from their 1986 companion paper's own body text (its
byline and existence are confirmed via Yerokhin et al.'s reference list
only). `inglis_teller_field_v_per_m` instead uses the standard order-of-
magnitude Inglis-Teller estimate, `E_IT ~ 1/(3*n_star^5)` atomic units
(Gallagher, Rydberg Atoms, the same text both anchor papers cite for
their own quantum-defect and Rydberg-atom-properties background),
labeled in its own docstring as an order-of-magnitude guard rather than
a fitted, published coefficient for this series.
`rydberg_quadratic_stark_shift_hz` raises `RydbergStarkValidityError`
above `STARK_VALIDITY_MARGIN` (1/3) of this estimate, the house pattern
of triggering validity guards with margin rather than at the estimate
itself (section 13's BBR temperature guard is the precedent).

### E44: EIT/Autler-Townes ladder susceptibility

**The formula, transcribed and verified.** Holloway et al. 2014 Eqs.
(1)-(4) (arXiv:1405.7066 page 3, read directly from the PDF text this
session):

    epsilon = epsilon_0 (1 + chi)                                    (1)

    chi = [j N |wp_p| Omega_p / (eps0 |E_p|)] x
          [(Omega_RF)^2 + 4 D13 D14] /
          [D12 (Omega_RF)^2 + D14 (Omega_c)^2 + 4 D12 D13 D14]       (2)

    D_1i = gamma_1i - j Delta_p        (as printed, resonant case)   (3)

    Omega_{p,c,RF} = |E_{p,c,RF}| wp_{p,c,RF} / hbar                 (4)

for the four-level ladder |1>-|2>-|3>-|4> (5S1/2-5P3/2-32D5/2-33P3/2,
the fourth level RF-coupled to the third). `N` is the atom density,
`wp_p`, `wp_c`, `wp_RF` the probe/coupling/RF transition dipole moments,
`Delta_p = omega_o - omega_p` the probe detuning, `gamma_1i` the
coherence decay rate for the `1->i` transition. The paper's own Eq. (3)
is printed for the resonant case (`Delta_c = Delta_RF = 0`) alone, and
states the general form would carry those detunings too, citing its
ref. [21] (Sandhya and Sharma, Phys. Rev. A 55, 2155 (1997)) without
reproducing it. `ladder_susceptibility` implements the natural ladder
generalization, each `D_1i` summing the detunings of every level
between `|1>` and `|i>` (`D12` carrying `Delta_p` alone, `D13` carrying
the two-photon detuning `Delta_p + Delta_c`, `D14` the three-photon
detuning `Delta_p + Delta_c + Delta_RF`), documented in the function's
own docstring as this project's own extension for the general-detuning
case. Setting `Omega_RF = 0` cancels `D14` out of the formula
algebraically (shown in the docstring), leaving the finite three-level
pole structure.
`tests/test_rydberg_cell_response.py::TestLadderSusceptibility::
test_three_level_reduction_matches_rmp_pole` checks that reduction
against a transcribed 3-level closed form from Fleischhauer, Imamoglu,
Marangos, Rev. Mod. Phys. 77, 633 (2005) ("Electromagnetically induced
transparency: Optics in coherent media"), Eq. 13 (page 639, read
directly from the owner-supplied PDF this session), the lambda-type
linear susceptibility with the same `gamma_31`/two-photon-detuning pole
this reduction produces. Holloway et al. 2014's own susceptibility
derivation cites Sandhya and Sharma 1997 and Meystre and Sargent's
textbook, their refs. [21] and [22]; the RMP review is not among them.
This project treats the RMP formula as an independent,
separately-verified cross-check of the reduced pole structure.

**Doppler averaging.** Mohapatra, Jackson, Adams, Phys. Rev. Lett. 98,
113003 (2007), arXiv:quant-ph/0612200, Eq. (1) (page 2) gives the
velocity-resolved susceptibility for the weak-probe 3-level ladder,
counter-propagating probe (wavevector `k_p`) and coupling (`k_c`,
opposite direction) beams, with the two-photon detuning carrying
`(Delta_p + Delta_c) - (k_p - k_c)*v` for an atom moving at velocity `v`
along the beam axis. `doppler_averaged_susceptibility` extends this
structure to the 4-level case (Section H of the module), integrating
`ladder_susceptibility` over a Maxwell-Boltzmann velocity distribution
via fixed Gauss-Hermite quadrature (`doppler_velocity_grid`, exact for
smooth functions against a Gaussian weight, and deterministic by
construction: no random sampling, no seed needed). The RF leg's own
Doppler shift is dropped, following Sedlacek et al. 2012's own stated
approximation for the kinematically identical 53D5/2-54P3/2 leg
(arXiv:1205.4461 page 8: "because the wavelength of the RF field is
large, the Doppler effect on the ... transition can be neglected"), a
four-order-of-magnitude wavelength gap between the mm/cm-scale RF leg
and the optical legs that applies equally to the 68.64 GHz/32D5/2-33P3/2
leg this section uses.

**The Doppler-mismatch factor, derived and resolved (dossier risk 1,
the pre-flight blocking task).** The AT splitting observed in the probe
transmission spectrum, plotted against probe detuning, is not the bare
Rabi splitting `Omega_RF/(2*pi)`: it is rescaled by the ratio of the
probe and coupling wavelengths. Two of the dossier's three primary
sources state the rescaling in opposite directions: Holloway et al. 2014
Eq. (12) and Mohapatra et al. 2007 both give a REDUCTION factor,
`lambda_c/lambda_p` (splitting smaller than the bare value, since
`lambda_c < lambda_p` for this ladder); Sedlacek et al. 2012's own prose
(arXiv:1205.4461 page 5, no equation number) states the RECIPROCAL,
`lambda_p/lambda_c`, an ENHANCEMENT. This project resolved the conflict
by deriving the observed splitting from the ladder's own Doppler-
detuning geometry, independently of either paper's stated direction, and
by verifying the result directly against Holloway et al. 2014's own
primary-source equation and its own calibration numbers.

*The derivation.* Take the probe propagating along `+z` (wavevector
`k_p`) and the coupling beam counter-propagating (`-k_c` along `z`,
Holloway et al. 2014's own Fig. 2(b) geometry). For an atom moving at
`v_z`, the Doppler-shifted detuning each leg sees in the atom's own
frame is `Delta_p_atom = Delta_p_lab - k_p*v_z` (probe, co-propagating
with the atom's `+z` motion) and `Delta_c_atom = Delta_c_lab + k_c*v_z`
(coupling, counter-propagating). In the strong single-photon Doppler
background (width `k_p*v_thermal`, the dominant absorption feature the
narrow EIT/AT structure sits inside), the velocity class dominating the
signal at a given probe detuning `Delta_p_lab` is the one near single-
photon resonance with the PROBE leg alone, `v_z ~= Delta_p_lab/k_p`
(the probe leg being the one directly driving population out of the
ground state, so it sets which velocity class contributes the strongest
absorption background for the EIT/AT feature to sit inside). Substituting
that velocity into the COUPLING leg's own Doppler-shifted detuning gives

    Delta_c_atom = Delta_c_lab + k_c * (Delta_p_lab / k_p)
                 = Delta_c_lab + (k_c/k_p) * Delta_p_lab
                 = Delta_c_lab + (lambda_p/lambda_c) * Delta_p_lab

(`k_c/k_p = lambda_p/lambda_c`, wavevector magnitude inversely
proportional to wavelength). The RF-driven dressed states of the
Rydberg level sit at `Delta_c_atom = Delta_c_lab +/- Omega_RF/2`
(resonant RF driving); the two AT peaks in the observed probe-detuning
spectrum are where the substituted expression above hits those two
values:

    Delta_c_lab + (lambda_p/lambda_c) * Delta_p_lab = Delta_c_lab +/- Omega_RF/2
    => Delta_p_lab = +/- (lambda_c/lambda_p) * (Omega_RF/2)

giving a peak-to-peak splitting of `(lambda_c/lambda_p) *
Omega_RF/(2*pi)` on the probe-detuning axis, the REDUCTION direction.
This matches Holloway et al. 2014 and Mohapatra et al. 2007; Sedlacek
et al. 2012 states the reciprocal, addressed below. The physical
picture: the probe-resonant velocity class maps the coupling leg's own
Doppler shift onto the probe-detuning axis with an extra factor of
`lambda_p/lambda_c` (larger, since
`k_c > k_p`), so the SAME Rydberg-level energy splitting `Omega_RF`
projects onto a COMPRESSED window in probe-detuning space, by the
inverse factor `lambda_c/lambda_p`.

*Independent verification against the primary source.* Holloway et al.
2014's own Eq. (12) (arXiv:1405.7066 page 7, read directly from the PDF
this session):

    |E_RF| = 2*pi * (hbar/wp_RF) * (lambda_p/lambda_c) * Delta_f       (12)

is the algebraic inverse of the derivation above (`Delta_f =
(lambda_c/lambda_p) * Omega_RF/(2*pi)` solved for `E_RF` via `Omega_RF =
wp_RF*E_RF/hbar` gives exactly Eq. 12). The paper's own prose
immediately above Eq. (12) states the splitting direction explicitly:
"states are scaled by lambda_c/lambda_p [7]", citing Mohapatra, Jackson,
Adams, PRL 98, 113003 (2007) as reference [7] in its own bibliography
(confirmed directly from the paper's reference list this session) for
that exact scaling claim, applied there to the kinematically analogous
case of a fine-structure splitting living in the same ladder's topmost
level. Three independent lines of evidence now agree on the reduction
direction: this project's own first-principles derivation, Holloway et
al. 2014's own stated equation and prose, and Mohapatra et al. 2007's
independently published analogous case (which Holloway's own paper
cites for the claim). Sedlacek et al. 2012's reciprocal prose statement carries no equation
number and no independent corroboration found here, so this project
treats it as a physics error in that source's informal explanation. A
definitional difference between the two papers is not the explanation:
both describe the identical observable (the RF-induced splitting of the
EIT/AT doublet, measured in the probe-transmission spectrum plotted
against probe-laser detuning, same units, same axis), so a reciprocal-
direction disagreement between them cannot be resolved as two different
quantities in different unit conventions.
`autler_townes_splitting_hz` and `field_from_at_splitting_v_per_m`
implement the resolved (reduction) direction; both are exact algebraic
inverses of each other, and a test using the reciprocal (Sedlacek-
direction) formula confirms it misses Holloway's own published
calibration data by far more than the check's tolerance
(`tests/test_rydberg_cell_response.py::test_wrong_doppler_direction_fails_the_check`).

**mu_RF, derived rather than looked up (dossier risk 2).** Holloway et
al. 2014's own Eq. (11), `wp_RF = 0.49 * e * a0 * Qn`, defines `Qn` as
"the normalized radial part of the dipole moment" (`Qn = R/a0`, their
Fig. 7 caption), read from a log-log plot; the paper gives no closed
form for it. `numerov_radial_matrix_element` computes this radial
integral by outward Numerov integration of the quantum-defect
(pure-Coulomb-tail) radial Schrodinger equation from `r_min` to just
beyond the outer classical turning point (`_turning_points`), atomic
units, using effective quantum numbers `n_star` set by the Rydberg-Ritz
quantum defects below. This pure-Coulomb approximation has a known,
disclosed accuracy limit for states with real core penetration (Rb D
and P states carry quantum defects of order 1-2.6). This project uses
two independent, disclosed derivations for the registry value in place
of that direct pure-Coulomb estimate:

- `RB85_MU_RF_32D52_33P32_C_M` (the registry value E44's ladder
  susceptibility actually uses) is backed out self-consistently from
  Holloway et al. 2014's own three published Fig. 15 (splitting, field)
  pairs. `field_from_at_splitting_v_per_m` solves Eq. 12 for `mu_RF`
  given the published `E`, the dossier's own recommended "practical
  shortcut" over digitizing Fig. 7. The three pairs give
  5.2452e-27, 5.2713e-27,
  and 5.2741e-27 C.m, agreeing with each other to within 0.6%; the mean,
  5.2635e-27 C.m, is the registry value.
- `numerov_radial_matrix_element`/
  `rf_transition_dipole_moment_from_quantum_defects` are still
  implemented and exercised as an independent, quantum-defect-based
  cross-check, at a stated, wide (factor-of-2) tolerance: they reproduce
  Sedlacek et al. 2012's own independently published, quantum-defect-
  derived value for the kinematically identical 53D5/2 -> 54P3/2
  transition (`mu_RF = 1.37e-26 C.m`) to within that factor, and agree
  with the Fig.-15-backed-out registry value for 32D5/2 -> 33P3/2 to the
  same factor (`tests/test_rydberg_cell_response.py::TestMuRfDerivation`).
  Neither number is presented as a published value; both derivations are
  shown in full in the module's own docstrings.

**Quantum defects, verified against primary PDF text.** Rb-85 nD5/2:
`delta0 = 1.3464657`, `delta2 = -0.5960`, read directly from Mack,
Karlewski, Hattermann, Hoeckh, Jessen, Cano, Fortagh, Phys. Rev. A 83,
052515 (2011), arXiv:1103.6221, Table I (the paper's own reproduction of
Li, Mourachko, Noel, Gallagher, Phys. Rev. A 67, 052502 (2003), which
has no arXiv preprint and is paywalled; this project could not verify
Li et al. 2003's own printed table directly, so the number is taken from
Mack et al. 2011's reproduction of it, disclosed in the registry's own
citation string). Rb-85 nP3/2: `delta0 = 2.64157`, `delta2 = 0.304`,
from Sanguinetti, Majeed, Jones, Varcoe, J. Phys. B 42, 165004 (2009),
arXiv:0905.0571, Table 3 ("Method 3", their own preferred direct fit),
used in place of Li et al. 2003's np-series values for the same
accessibility reason: an independent, later, higher-precision, freely
verifiable measurement of the same quantity.

### Reproduction targets and their classification

- **C3, calibration KA** (`benchmarks/run_rydberg_cell_response.py`,
  `run_c3_calibration_case`): Holloway et al. 2014 Fig. 15's three
  published `(Delta_f, E)` pairs at 68.64 GHz, reproduced from the
  registry `mu_RF` via the resolved Eq. (12). `arithmetic_reproduction`.
  Worst relative error 0.35% against a 1% tolerance (Holloway et al.
  state their own quantum-defect method is accurate to <0.1% and
  separately flag an open, unquantified RF-standing-wave uncertainty, so
  this check does not claim tighter than the source itself claims to
  control). MET.
- **C4, polarizability KA**: Rb-85 nD5/2 `alpha0` at n=30, 35, 50, two
  independent sources, worst relative difference 2.88% against a 5%
  tolerance. `arithmetic_reproduction`. MET.
- **C5, limit kill-tests**: zero field returns the unperturbed line at
  the byte level; a uniform field returns a pure shift of the same
  lineshape, also byte-identical to a direct single-atom evaluation at
  that shift. A sign-flip and a doubled-coefficient deliberate break at
  `compose_inhomogeneous_eit_spectrum` both move the result away from
  the correct one, confirming these composition-level checks are armed.
  `rydberg_quadratic_stark_shift_hz`'s own 1/2 prefactor carries a
  separate, function-level pin
  (`TestQuadraticStarkShift::test_magnitude_matches_independently_computed_value_at_a_stated_point`):
  a value hand-computed in the test from the registry `alpha0` and the
  SI conversion constants, at a stated (state, field) point, tight
  enough that a dropped or altered prefactor fails it (verified this
  session by reintroducing exactly that break and confirming the new
  test alone fails). This closes the gap the composition-level
  doubled-coefficient break cannot reach on its own: doubling only the
  `alpha0` argument there is tautological at the formula level, since
  the same doubled value flows through whichever prefactor the function
  uses. `internal_structural_check`. MET.
- **C6, surface-charge demonstrator**: a wall-patch (point-charge
  superposition) field over a cylindrical vapor cell produces a line
  shift and a per-atom Stark-shift spread that both grow monotonically
  with patch charge and with a shrinking cell radius, the qualitative
  phenomenology Patrick, Schlossberger, Hammerland, Prajapati, McDonald,
  Berweger, Talashila, Artusio-Glimpse, Holloway, AVS Quantum Science 7,
  024401 (2025), arXiv:2502.07018, report (line shift and asymmetric
  broadening from photoionized surface charge patches). No printed
  numeric target exists in that paper to reproduce arithmetically: its
  field-vs-power and EIT-vs-wavelength curves are digitizable-axis
  figures. This project classifies the case `computable_comparison`;
  the stricter `arithmetic_reproduction` class requires a printed
  numeric target to reproduce, which this paper's figures do not
  provide. A 2025-2026 literature currency check found no paper
  claiming this problem solved or a field-wide mitigation
  standardized; partial, geometry-specific workarounds exist
  (all-dielectric cells, three-photon near-IR excitation) and are not
  claimed here as closing the problem.
- **C7, Doppler layer**: the full Doppler-averaged 4-level susceptibility's
  numerically extracted AT-doublet spacing lands within a stated 0.6-1.0
  band of the closed-form `(lambda_c/lambda_p)*Omega_RF/(2*pi)` limit in
  the regime where the RF-driven splitting dominates the coupling-
  induced dressing width (`tests/test_rydberg_cell_response.py::
  test_at_splitting_survives_doppler_averaging_at_the_right_scale`);
  finite decay rates and thermal averaging pull the resolved peak
  spacing in below the idealized zero-linewidth value, standard
  Autler-Townes line-pulling. A wrong-direction Doppler factor would
  move the analytic target by roughly `(lambda_p/lambda_c)^2 ~= 2.6x`
  and put the ratio far outside this band, so the check is non-vacuous.

### Scope, unchanged from the plan's own boundary

Phase A: the quadratic (isolated-state) Stark regime, scalar
polarizability only, one calibration ladder (Rb-85, 32D5/2-33P3/2), no
pipeline wiring (matching E40/E41's own scope pattern: functions and a
benchmark, `cliffordclock.integrator.rydberg_cell_response` is not on
`cliffordclock.pipeline`'s config surface in this phase). Full Stark-map
diagonalization beyond the quadratic regime, tensor polarizability,
JAX differentiability, and coupling to external EM field exports for
the cell/waveguide side are later work.

## 20. Full Rydberg Stark maps beyond the quadratic regime (v1.16.0, WP40 Phase B)

Motivation (project owner, following the WP40/41 phase-B plan): Phase
A's quadratic Stark term (E43) holds only inside a validity window; the
literature's own statement is that a full Stark map -- diagonalizing the
Rydberg manifold's Hamiltonian in a quantum-defect basis under an
applied DC field -- is required beyond it. This section adds that map:
Hamiltonian assembly in the `(n, l, j, mj)` basis, exact diagonalization
over a field grid, and adiabatic eigenvalue tracking, so a registry
state's Stark shift becomes a function of field smooth through the
quadratic window and beyond, replacing Phase A's order-of-magnitude
Inglis-Teller validity guard with a computed crossover. Implemented in
`cliffordclock.integrator.rydberg_stark_map`. Benchmarked in
`benchmarks/run_rydberg_stark_map.py`; the ARC cross-validation fixture
is generated by `benchmarks/generate_wp40_arc_reference.py`.

**Method and its source, stated precisely.** The construction --
diagonal quantum-defect energies plus an off-diagonal electric-dipole
coupling matrix, diagonalized at each field with the eigenvalue tracked
by continuity -- is universally attributed in this literature to
Zimmerman, Littman, Kash, Kleppner, Phys. Rev. A 20, 2251 (1979). That
paper predates arXiv and its own text was **not** obtained directly for
this build (every search for a legitimate free copy failed); no
equation number from it is cited anywhere in this section. What is cited
and directly verified instead: Sibalic, Pritchard, Adams, Weatherill,
"ARC: An open-source library for calculating properties of alkali
Rydberg atoms," Comp. Phys. Comm. 220, 319 (2017), arXiv:1612.05529,
read directly from the arXiv PDF this session -- its own Sec. 2.3.2
states plainly it follows "the method of Zimmerman et al."; its Eqs.
(1)-(2) (quantum-defect energies), (6)-(8) (the `x=sqrt(r)`-substituted
Numerov radial integration), (9)-(12) (Wigner-3j/6j dipole matrix
elements), and (18) (`H = H0 + E*z`, one Stark map per `mj`) are the
equations this module's own code implements. Grimmel, Mack, Karlewski,
Jessen, Reinschmidt, Sandor, Fortagh, New J. Phys. 17, 053005 (2015),
arXiv:1503.08953, read directly in full, is an independent, later,
from-scratch implementation that also describes itself as following
Zimmerman's method and whose own Hamiltonian/matrix-element structure
agrees with ARC's, corroborating ARC's own restatement without needing
Zimmerman's own text.

**Quantum-defect registry, extended.** Phase A's own nD5/2 and nP3/2
defects are reused (not re-transcribed). New for WP40: S1/2, P1/2, D3/2
(Li, Mourachko, Noel, Gallagher, PRA 67, 052502 (2003), taken via ARC's
own `Rubidium85.quantumDefect` table -- ARC's own in-code citation names
this paper, the identical pattern already used for Phase A's own nD5/2
value and the WP40 dossier's Inglis-Teller calculation); F5/2, F7/2
(Han, Jamil, Norum, Tanner, Gallagher, "Rb nf quantum defects from
millimeter-wave spectroscopy of cold 85Rb Rydberg atoms," PRA 74, 054502
(2006), byline/title confirmed via the APS DOI record, full text not
obtained, taken via ARC's reproduction); G7/2/G9/2 (Moore, Duspayev,
Cardman, Raithel, "Measurement of the Rb g-series quantum defect using
two-photon microwave spectroscopy," PRA 102, 062817 (2020) -- this one
WAS read directly, par.nsf.gov's public-access PDF: its own abstract
states `delta0 = 0.003 999 0(21)`, `delta2 = -0.0202(21)`, matching
ARC's tabulated value to all five printed digits). `l >= 5` is treated
as exactly hydrogenic (`delta0 = delta2 = 0`), a disclosed approximation
(ARC's own table stops at G; real defects at this l are already ~4e-3).

**Angular-momentum algebra.** Wigner 3-j/6-j symbols via the standard
Racah closed-form sum (Edmonds, *Angular Momentum in Quantum Mechanics*,
1957), implemented in pure numpy/scipy with log-gamma factorials (this
module's dependency policy, matching ARC's own: no symbolic-algebra
package). Verified against six hand-derivable special-case values and
the general 3-j orthogonality relation, both re-run as pytest cases
(`tests/test_rydberg_stark_map.py::TestWignerSymbols`).

**Two Numerov bugs found and fixed while building this module** (the
central engineering narrative of this section; see
`cliffordclock.integrator.rydberg_stark_map`'s own top-of-module
docstring for the full account):

1. Phase A's own `rydberg_cell_response._numerov_outward` (uniform-`r`
   grid Numerov integrator) carried a sign error in its discrete
   recursion: every `T = h^2 g/12` term had the opposite sign from the
   correct formula, found when this module's own independently
   implemented `x=sqrt(r)`-substituted integrator disagreed with the
   pre-fix output for 32D5/2->33P3/2 by 16%. Fixed (with an exact-
   solution, `y''=-y`, verification pinning the correct sign) in the
   same commit that built this section; the two independently-coded
   single-pair integrators now agree to ~0.02%. No gated Phase A check's
   stated tolerance was violated by the pre-fix bug (the affected value
   only ever fed a disclosed, wide factor-of-2 cross-check).
2. This module's own first working version integrated OUTWARD from the
   inner radius, the same direction as (1). That is unstable for a
   bound-state radial equation once carried past a state's own classical
   turning point, and its `l`-dependent small-`r` boundary condition does
   not fix a consistent RELATIVE phase between different `(n,l)` states
   sharing one Hamiltonian. A systematic, pair-by-pair check of every
   off-diagonal matrix element the 32D5/2 map basis uses against ARC's
   own `getDipoleMatrixElement` found D-P couplings within the expected
   pure-Coulomb-tail-like scale factor but D-F and same-`n` high-`l`
   couplings wrong by factors from -1379x to +532x, including outright
   sign flips -- matching ARC's own paper text, read earlier in this
   build but not registered as significant until this discrepancy forced
   a re-read: "the integration is performed inwards, starting at
   r_o[,] ... to minimise errors introduced by the approximate model
   potential at short range" (Sec. 2.2.2). Switching to inward
   integration from a common `X(r_o)=0` boundary condition reproduces
   every one of those matrix elements to within 1% of ARC's value (most
   within 0.1%), and dropped this module's own first aggregate
   quadratic-shift estimate for 32D5/2 from roughly two orders of
   magnitude too large down to within 1.3% of the same quantity computed
   by running this module's own tracking code on ARC's own Hamiltonian
   matrices directly.

**Model potential, not a pure-Coulomb tail.** Discovering bug 2 above
also surfaced that a pure `-1/r` tail (Phase A's own approximation,
adequate for its single-transition mu_RF cross-check) is not accurate
enough for a multi-state near-degenerate-manifold Hamiltonian: the
radial integrator now uses the same Marinescu, Sadeghpour, Dalgarno,
"Dispersion Coefficients for Alkali-Metal Dimers," Phys. Rev. A 49, 982
(1994), one-electron model potential ARC itself uses (`V(r) =
-Z_l(r)/r - (alpha_c/2r^4)(1-e^{-(r/r_c)^6})`, l-dependent parameters),
values taken via ARC's own `Rubidium85` class (byline/title confirmed
via the APS DOI abstract page). No spin-orbit term: this module still
gets each state's ENERGY from the empirical quantum defect, exactly as
before; the model potential only shapes the radial wavefunction (l-basis,
matching ARC's own l-basis reduced-matrix-element convention).

**Basis and Hamiltonian.** `build_basis(n0, l0, j0, mj, delta_n=5,
l_max=20)` reproduces ARC's own `defineBasis` state-for-state (cross-
checked directly: ARC's own `len(calc.basisStates)` for
`(32,2,2.5,0.5,27,37,20)` is 451, matching this module's own count for
the same arguments exactly, and ARC's own `indexOfCoupledState` (209)
matches this module's own target index for the identical basis). `l_max
=20`, `delta_n=5` is ARC's own stated convergence rule of thumb ("l_max
of 20 and n_max-n_min ~ 10") and this module's production default.
Off-diagonal matrix elements: each pair's radial integral is computed on
its OWN dedicated grid (`_radial_matrix_element_pair`, memoized), not a
basis-wide shared grid (an earlier, unsound design choice this
section's own module docstring documents finding and rejecting: a
shared grid lets low-turning-point states integrate deep into the
numerically unstable forbidden zone before the grid ends, corrupting
their own normalization).

**Diagonalization and adiabatic tracking.** `diagonalize_stark_map`
diagonalizes `H0 + E*H1` at each field and tracks the target state by
maximum overlap with the PREVIOUS step's tracked eigenvector (not the
original field-free state, which fragments once mixed far from its
zero-field character) -- the plan's own stated "eigenvalue connectivity
tracking (adiabatic following by overlap)." The per-step overlap array
is what both the crossover detector and the ARC benchmark's tiered
tolerance use.

### G24 checks

- **C1, provenance/byline**: every new quantum defect above is byline-
  and (where obtainable) title-verified against the arXiv/journal page
  directly this session; the one exception (Han et al. 2006, no arXiv,
  APS 403) is disclosed as taken via ARC's own reproduction, the
  identical discipline already applied to the Inglis-Teller defects in
  the WP40 dossier and to Phase A's own Li et al. 2003 citation.
- **C2, Hamiltonian-assembly transcription**: ARC's Eqs. (1)-(2),
  (6)-(8), (9)-(12), (18), read directly and independently re-derived by
  hand where the derivation itself mattered (the `x=sqrt(r)` substitution,
  the Numerov recursion sign). No equation number is attributed to
  Zimmerman 1979 anywhere in code or docs.
- **C3, quadratic-crossover internal consistency**
  (`benchmarks/run_rydberg_stark_map.py::run_c3_crossover_case`): the
  map's own mj-averaged (tensor-cancelling) low-field curvature vs.
  Phase A's own E43 registry `alpha0`, all four registry states (30, 32,
  35, 50 nD5/2). Worst relative error 4.91% (n=50) against a 15%
  tolerance. `arithmetic_reproduction`. MET. Kill-tested: a sign-flipped
  or doubled map `alpha0` misses the same registry value by far more
  than the tolerance.
- **C4, ARC cross-validation**
  (`run_c4_arc_validation_case`; fixture: ARC v.3.10.2, commit
  4b4573e965222e798ac59636ad7a8b3457262835, BSD-3-Clause, installed
  directly into this project's own `.venv` per the build prescription's
  own instruction -- no separate environment needed, `pip check` reports
  no conflicts; only the fixture's numeric output is committed, no ARC
  code vendored). Two tiers, per the dossier's own instruction against
  one flat tolerance across the whole field range: a gated low-field
  tier (field <= 50% of the Inglis-Teller estimate, this module's own
  tracking code applied to BOTH its own and ARC's own Hamiltonian
  matrices so the comparison isolates the Hamiltonian construction
  alone), worst relative error 2.05% (n=50) against a 5% tolerance, MET,
  `independent_implementation_reproduction`. Beyond that tier, the two
  independently-built Hamiltonians can legitimately track through a
  shared crossing onto swapped branches (verified: restricting the
  comparison to points where both curves report high step-overlap does
  not by itself close the gap, confirming a branch-identity effect, not
  a resolution or tracking bug); this is reported (crossover-location
  fields; n=35 matches ARC's own first-low-overlap field exactly, 50.34
  V/cm) rather than gated by a single numeric tolerance.
- **C5, published anchor, three-part** (`run_c5_published_anchor_case`;
  no single source combines the registry species/l/crossing coverage
  with printed, non-digitized numbers): (a) low-field reduction to
  Holloway et al. 2014 Fig. 15's fields, mj-averaged map vs. Phase A's
  own E43 closed form, worst relative error 1.44% against 10%, MET,
  `arithmetic_reproduction`; (b) O'Sullivan & Stoicheff 1985's printed
  Rb-85 nS crossing-field fit as a same-family method check (this
  module's own map built for the nS1/2 series, not the registry's nD5/2),
  printed 6.97 V/cm vs. map-detected 8.47 V/cm, 21.5% against a 25%
  tolerance, MET, `arithmetic_reproduction`; (c) Grimmel et al. 2015's
  supplementary-data URL was fetched this session and returned an HTML
  page, not a machine-readable data file, so no quantitative Grimmel
  comparison is included (per the project's standing digitization
  caution, no digitized-plot substitute either).
- **C6, basis-truncation convergence**
  (`run_c6_convergence_case`/`convergence_sweep`): 50D5/2 (the dossier's
  own flagged load-bearing risk state, computed crossover order 6-8 V/cm,
  an order of magnitude below 30D5/2's) and 32D5/2, `(delta_n, l_max)`
  swept `(2,6) -> (3,10) -> (5,14) -> (5,20)`. Both states converged well
  inside a 10% threshold; the second-largest basis already agrees with
  the largest to < 0.1% for both states. MET.
- **C7, battery + prose**: `tests/test_rydberg_stark_map.py` (44 cases:
  Wigner symbols, basis/Hamiltonian construction, diagonalization,
  kill-tested C3, C6 smoke test, the E44 `shift_fn` integration re-
  running Phase A's own C5 structural limits on the map path); ruff,
  mypy --strict, and the release-checks prose/tolerance/citation/
  headline/internal-path scans, all green.

### E44 integration

`rydberg_cell_response.compose_inhomogeneous_eit_spectrum` gained an
optional `shift_fn` keyword (backward-compatible: the default preserves
its exact prior behavior, `rydberg_quadratic_stark_shift_hz`), so the
EIT/AT observable can source its per-atom Rydberg shift from the full
map instead of the quadratic closed form -- the plan's own stated
deliverable. `rydberg_stark_map.map_sourced_stark_shift_hz` (bind `n0`
via `functools.partial` first) is the map-sourced implementation,
mj-averaged like the C3/C5(a) checks. Phase A's own C5 structural limit
checks (zero field byte-identical to the unperturbed line; a uniform
field byte-identical to a single shifted evaluation) are re-run
verbatim on this map path
(`tests/test_rydberg_stark_map.py::TestE44MapSourcedIntegration`), plus
a third test pinning that the new keyword does not alter any existing
caller's behavior.

### Scope, unchanged from the plan's own boundary

Rb-85 only, the four WP39 registry states plus whatever `n, l, j` a
caller passes through the general API. JAX differentiability through
this eigensolve is explicitly WP41's own question (the WP40/41 dossier's
own risk assessment recommends the quadratic path as WP41's primary
deliverable, with the map path as an explicit stretch goal carrying its
own sub-gate). Notebook 17, a CONVENTIONS docs/terms page beyond this
section, and further pipeline wiring are deferred per the plan (WP40
ships module + benchmarks + docs only; notebook 17 belongs to WP41).

---
*Changelog:*
*1.16.0 (2026-09-03): WP40 Phase B, specified directly by the project
owner following the project's internal Stark-map formalism research
dossier (no separate formalism sign-off ceremony recorded for this
entry): §20 added (full Rydberg Stark maps: quantum-defect `(n,l,j,mj)`
basis Hamiltonian assembly via Wigner-3j/6j dipole matrix elements and
Marinescu-model-potential Numerov radial integrals, exact diagonalization
over a field grid with adiabatic eigenvalue tracking, the computed-
crossover validity guard replacing Phase A's Inglis-Teller estimate, ARC
cross-validation, and the E44 map-sourced `shift_fn` integration). Two
Numerov bugs found and fixed in the process: a sign error in Phase A's
own uniform-`r` recursion, and this module's own initial outward- (vs.
the correct inward-) integration convention, the latter traced directly
to a passage in ARC's own paper this build had read earlier without
registering its significance. Awaiting independent theory review.*
*1.15.0 (2026-09-03): WP39 Phase A, specified directly by the project
owner following the project's internal Rydberg-cell-response research
dossier (no separate formalism sign-off ceremony recorded for this
entry): §19 added (E43 quadratic Stark shift of a single Rydberg state,
its atomic-unit unit conversion and Inglis-Teller validity guard; E44
the four-level ladder EIT/Autler-Townes susceptibility, Doppler
averaging, and the resolved Doppler-mismatch-factor derivation between
Holloway et al. 2014/Mohapatra et al. 2007's reduction direction and
Sedlacek et al. 2012's reciprocal prose statement, settled in the
reduction direction by first-principles derivation and independent
verification against Holloway's own primary-source equation). Awaiting
independent theory review.*
*1.14.0 (2026-08-29): WP38 Phase 2, specified directly by the project
owner (no separate formalism sign-off ceremony recorded for this entry):
§18 added (E42 the sideband-spectrum forward model), the differentiable
clock-transition excitation spectrum (carrier plus red/blue axial
sidebands) on the E41 JAX core, two labeled paths: the harmonic
validation anchor (Blatt et al. 2009's full carrier Rabi-flopping
machinery, Eqs. 13-20, and the full Appendix A1-A2 sideband
population/Lorentzian sum, the full `(n_z,n_r)` quantum-number sum
Eqs. A3-A5 go on to reduce further for the shallow-edge case alone) and
the BO+WKB capability (Goti et al. 2025 Eqs. 5-9, a NEW
table-interpolation numerical route replacing per-energy bisection for
spectrum-scale tractability, with its own documented resolution limit
within ~5 E_R of a band's top). Cross-validated against `large-lattice-model`
(github.com/inrim/large-lattice-model, MIT, INRIM), a real independent
open-source implementation using exact Mathieu-function characteristic
values: a new evidentiary class, `independent_implementation_reproduction`
(distinct from this project's paper-Table `arithmetic_reproduction`),
MET at `1.06e-7` (band-bottom eigenvalues) and `4.7e-3` (Franck-Condon
detunings) relative error, plus a `computable_comparison`-class full-shape
check (peak position within 1.5%, shape correlation >= 0.93) bridging
three documented lineshape-convention differences. A synthetic
gradient-based fitting demonstration (`scipy.optimize.minimize` with
`jax`-supplied exact gradients, Laplace/Hessian uncertainties, a fixed
grid of truth values and deterministic noise seeds) recovers both `(u0,
Tr)` within 2-sigma in 11/12 cases, stated as the first GRADIENT-based
(not the first) fit of a BO+WKB-class sideband lineshape, since
`large-lattice-model`'s own non-differentiable fitter already exists and
was used for Goti et al. 2025's own real IT-Yb1 fits. The Goti et al.
2025 real-scan fit (Figs. 4, 7) was assessed and declined: the
underlying PDF carries no recoverable per-marker coordinate stream, so a
defensible extraction would require pixel-level digitization, a weaker
evidentiary class than this section's own synthetic and
independent-implementation checks; the raw scan data is recorded as the
named partnership ask.*
*1.13.0 (2026-08-29): WP37, specified directly by the project owner (no
separate formalism sign-off ceremony recorded for this entry): §17
extended with a differentiable JAX implementation of the same E41 BO+WKB
physics (`cliffordclock.integrator.lattice_light_shift_jax`), built for
Phase 2's gradient-based spectrum fitting. No new physics claim: every
formula is the one this section already specifies, and Model A's Eq. 1
is ported alongside it. Every formula is evaluated at a fixed grid
resolution in place of the reference module's adaptive convergence
loops. An offline convergence study, run against the reference
implementation's own converged output, chose and verified that
resolution. Separately, `jax.grad` of the light shift with respect to `u0` and
`Tr`, checked against central finite differences of the REFERENCE
implementation at all four G18 table points, matches to `4.9e-8`
relative, worst case; `X`/`Y`/`Z` agree with the reference to better than
`1.57e-7` relative at the same points. Found and fixed in the same entry:
an intermittent `NaN` gradient, caused by differentiating the turning-
radius root-find through the reference module's own energy-clamping
convention. The clamp flattens the axial energy to `0.0` identically across
an entire ray of radii beyond the true band edge, so the root-find's
implicit-function-theorem gradient divided by zero whenever the fixed-
iteration bisection landed on that flat side, at the root-find's `E=0`
endpoint specifically. The fix: root-find against the unclamped
eigenvalue, the same physical root, with no flat region to land on.*
*1.12.0 (2026-08-29): WP36 Phase 1, specified directly by the project
owner (no separate formalism sign-off ceremony recorded for this entry):
§17 added (E40 the Katori-lineage harmonic/operational lattice-light-shift
model, Ushijima et al. 2018 Eq. 1/Eq. 2 and the two distinct radial-
thermal reduction-factor forms; E41 the NIST Born-Oppenheimer+WKB model,
Beloy et al. 2020's axial separation/WKB quantization/density-of-states
Eqs. 4-21, with the harmonic-limit consistency check and the numerically
stable thermal-averaging reformulation). Both models implemented as pure
functions in `cliffordclock.integrator.lattice_light_shift`, not wired
into the pipeline this phase. Four reproduction targets: Ushijima et al.
2018's own operational point (MET), Aeppli et al. 2024's lattice-light
budget line (MET), Bothwell et al. 2025's own published harmonic-vs-BO+WKB
X/Y/Z table, all four rows (MET, arithmetic reproduction), and Bothwell et
al. 2025's headline alpha~M1E2 coefficient (explicitly classified
`computable_comparison`, not `arithmetic_reproduction`, since the
published coefficient values are fit outputs against unpublished raw data;
the computable substitute, both models evaluated at the paper's own stated
conditions, is reported instead), plus a density-of-states contrast case
quantifying the two models' radial-degeneracy divergence as a function of
radial temperature.*
*1.11.0 (2026-08-24): WP35, specified directly by the project owner (no
separate formalism sign-off ceremony recorded for this entry): §16
extended with the coupled two-ion Floquet solve, replacing WP33/WP34's
`participation*F_axis` factorization (a single per-axis factor applied
identically to a pair's COM and STR members, which WP34's own structural
finding showed cannot in principle produce a relative shift between
them) with a direct integration of the two ions' coupled time-periodic
equations of motion: a 4x4 monodromy matrix over one RF period gives the
two collective quasi-frequencies exactly, and a per-mode Fourier
decomposition of the propagated eigenvectors gives the clock ion's
participation AND its exact micromotion enhancement from the SAME
per-mode series, no per-ion `F` computed in isolation anywhere. Verified
against two exact limits this session (`c -> 0` against WP34's own
single-ion result, `q -> 0` against WP32's own static secular
participation decomposition, both to float64 precision). G17 gate
review: the initially-reported per-axis Mathieu-parameter inversion (the
coupled system's own quasi-frequencies matched directly to that SAME
axis's own two measured frequencies) fits each axis exactly by
construction, so its own near-unity per-mode ratios carry no independent
evidence the model is correct, and the promised partner over-
determination check was not implemented; both are fixed in this same
entry. The constrained fit
(`cliffordclock.integrator.omega.constrained_two_ion_mathieu_fit`) is
this WP's own headline result: one shared Mathieu RF parameter magnitude
`q` (Berkeland Eq. 6) and a DC-split fraction `alpha` (`a_z` fixed
exactly from the measured axial frequency), fit by Gauss-Newton least
squares against all FOUR measured radial frequencies at once, a genuine
over-determination with real, nonzero residuals reported in full; the
per-axis solve is kept as a labeled diagnostic variant, unchanged. The
partner over-determination check now runs for the constrained fit on
both datasets, reusing WP34's own
`predicted_partner_bare_radial_frequencies_hz_exact` unchanged. The
uncertainty budget carries a third, new component alongside thermometry
and rounding: model structure, `|constrained total - per-axis-diagnostic
total|`, the direct cost of choosing between two defensible models.
Result: for the Al27+/Mg25+ Marshall dataset the constrained fit's four
radial per-mode ratios move from WP34's `1.02/0.86/0.98/0.94` to
`1.02/0.99/1.02/0.99`, the total to `-1.1415e-17`, landing `0.08` sigma
from Marshall's published `-114.6(3.8)e-19` with all three uncertainty
components counted: `MET`, still classified `arithmetic_reproduction`.
The partner check lands at `-0.07%`/`-0.11%` (X/Y), tighter than WP34's
own `-0.45%`/`-0.40%`. A second, independent dataset (Brewer et al.
(2019), arXiv:1902.07694) shows the same qualitative pattern. WP30-34's
own functions are untouched, G14-G16-gated record where applicable, and
every WP35 function is additive alongside them.*
*1.10.0 (2026-08-24): WP34, specified directly by the project owner (no
separate formalism sign-off ceremony recorded for this entry): §16
extended with the numerically exact Floquet treatment of intrinsic
micromotion (the continued-fraction/Newton solve for the exact Mathieu
characteristic exponent `beta(a, q)`, the exact velocity-variance
enhancement `F_exact` derived from the Floquet Fourier coefficients and
verified to reduce exactly to Berkeland's leading-order bracket at the
leading-order substitution, and the exact 2D clock-ion Mathieu inversion
and partner-ion over-determination check, all cross-checked this session
against independent monodromy-matrix ODE integration to float64
precision), replacing WP33's leading-order Mathieu bracket
SELF-CONSISTENTLY end to end, plus an input-rounding uncertainty channel
(each published frequency's half-last-digit bound, propagated by finite
differences through the full reconstruction chain and combined in
quadrature with the existing thermometry uncertainty, reported as its own
labeled component). Result: the exact treatment moves the Al27+/Mg25+
Marshall total by well under 1 percent relative to WP33's own leading-order
result, staying at essentially the same ~1.6 sigma `NOT MET` verdict, and
the rounding-uncertainty channel comes out roughly an order of magnitude
smaller than the existing thermometry channel; a second, independent
dataset (Brewer et al. (2019), arXiv:1902.07694) shows the same near-unchanged
pattern. The opposite-sign per-mode deviation within the X axis that the
G14/G15 gate reviews identified persists unchanged under the exact
treatment, a structural confirmation on top of the earlier empirical one:
the residual sits outside the single-per-axis-enhancement-factor model
WP33 and WP34 both implement, with a genuinely per-mode mechanism (still
within Mathieu-order physics) and the published per-mode calibration
chain both remaining open candidates. WP33's own functions are
untouched, G15-gated record, and every WP34 function is additive
alongside them.*
*1.9.0 (2026-08-23): WP33, specified directly by the project owner (no
separate formalism sign-off ceremony recorded for this entry): §16
extended with the mode-specific intrinsic-micromotion enhancement
addition, closing the reconciliation the G14 gate review identified for
WP32: the clock ion's leading-order Mathieu parameters (`q`, `a_x`,
`a_y`, `a_z`) solved from the trap's published RF drive frequency plus
WP32's own reconstruction (Berkeland J. Appl. Phys. 83, 5025 (1998)
Eqs. 4-6/9-10, cross-checked against Wubbena's alpha/epsilon convention
already used by WP32), the mandatory partner-ion over-determination
check (mass-scaling to a genuinely independent, falsifiable prediction),
and the per-axis enhancement factor `F_axis = 1 + q^2/(2*a_axis+q^2)`
(Berkeland's own Eq. 10, not independently re-derived) replacing WP32's
uniform factor-of-two candidate. Result: the Al27+/Mg25+ benchmark case's
radial per-mode ratios move from WP32's `0.42/0.35/0.52/0.50` to
`1.04/0.86/0.98/0.94`, and the total moves from ~14.0 sigma to ~1.6 sigma
from Marshall et al.'s published band (still `NOT MET`, materially
narrowed not fully closed); a second, independent dataset (Brewer et
al., arXiv:1902.07694, a different trap) shows the same qualitative
improvement and passes the same over-determination check at the
sub-1%-relative level.*
*1.8.0 (2026-08-23): WP32, specified directly by the project owner (no
separate formalism sign-off ceremony recorded for this entry): §16
extended with the radial-spectrum-reconstruction addition, the two-ion
RADIAL normal-mode inversion (the mass-weighted 2x2 eigenproblem, the
Coulomb-curvature derivation from the shared axial spring constant and
Wubbena Eq. 7/12, the quadrant disambiguation rule and its RF-
pseudopotential physical basis, and the feasibility guard), replacing
WP31's axial-form-as-radial approximation for a caller who supplies the
lab's own measured radial mode frequencies (WP31's approximation stays
available, unmodified, as its own labeled case), and the resulting
open finding from the Al27+/Mg25+ benchmark case: the reconstruction's
total-level agreement with Marshall et al.'s published band is
essentially unchanged from WP31's own approximation, for a structural
reason (mode-pair participation sums to `1.0` regardless of split) this
entry states but does not resolve.*
*1.7.0 (2026-08-22): WP31, specified directly by the project owner (no
separate formalism sign-off ceremony recorded for this entry): §8 added
(E39 coherent/phase-resolved rotor composition and Ramsey visibility: the
population-weighted linear-sum combiner and its no-renormalization rule,
the two-classic-errors kill-test discussion, the `B̂_C`-plane projection
convention, the Gaussian closure validation identity `V = exp(-sigma_Phi^2/2)`,
the Gaussian-motional-state-only scope boundary, and the squeezed-thermal
classical-sampler extension, `ensemble.squeezing_r`); §16 (E38) extended
with the per-mode participation-factor generalization (multi-ion
crystals), the closed-form two-ion axial eigenvector solution and its
documented radial-approximation scope caveat, and the narrowed remaining
scope boundary (N>2 crystals need a numeric eigensolver; RF/micromotion
unchanged).*
*1.6.0 (2026-08-22): WP30, specified directly by the project owner (no
separate formalism sign-off ceremony recorded for this entry, mirroring
1.5.0/E37's record): §16 added (E38 quantum-motional second-order-Doppler
pivot term: the hbar/m/2pi-explicit `<v^2>` formula, the ground-state
zero-point floor, the excess-micromotion optional input channel and its
RF-dynamics roadmap-boundary scope statement, the no-double-counting
argument against the existing E15/E21 classical kinematic term and its
CONVENTIONS-level statement of the config-parse-time
`ensemble.regime="classical"` rejection, E33-pattern spatially-uniform
composition, and uncertainty-propagation partials).*
*1.5.1 (2026-08-23): WP29 Tier 1 Part 1, tooling/input-format addition, no
formalism change (no separate sign-off ceremony recorded, same basis as
1.5.0): §13's E37 `environment.radiation_environment.surfaces` list gains
an equivalent `surfaces_file` form (a plain-text surfaces table, see
docs/cli.md), mutually exclusive with the inline `surfaces` list and
parsed into the identical intermediate structure before reaching E37's
existing weight-normalization/validity-window/emissivity-topology checks.*
*1.5.0 (2026-08-22): WP29 Tier 1, specified directly by the project owner
following the project's internal BBR thermal-environment research dossier
(no separate formalism sign-off ceremony recorded for this entry): §13
extended with E37
(multi-surface BBR thermal environment: per-surface weight/temperature/
emissivity input, the PTB aperture-form emissivity correction, per-moment
sums replacing the single-`T` static/dynamic evaluation, per-moment
effective temperatures `T_eff,n`, the exact bit-for-bit reduction to E32 for
a uniform environment, and independent/correlated per-surface temperature-
uncertainty combination modes).*
*1.4.0 (2026-08-11): WP22, G9 theory sign-off (conditional on the A1
computed-magnitude regression and the A4 dispersion-labeling edit, both
satisfied): §15 added (E36 gravitational-redshift pivot term, its sign/
magnitude/composition/rotor-scope/validity-bound record; the
`lattice_extended` ensemble regime and its per-site frequency map/
dispersion-labeling discipline; the Bothwell 2022 benchmark case's
coordinate-sign mapping).*
*1.3.0 (2026-08-11): WP21 Tier 2, G8 theory sign-off (conditional on the
A1 sign discipline, satisfied via primary-text transcription of both
Itano 2000 Eq. 46 and Roos et al. quant-ph/0701215v1 Eq. 1): §14 added
(E34 quadrupole level shift with the coordinate-free reduction and its
derivation; E35 quadrupole pivot composition, three-orientation
cancellation proof, traceless-symmetric-part requirement, and the
spin-connection scope limit). The sign pin is Roos's own Eq. 1,
transcribed directly from primary text; Dube 2005, initially
inaccessible and later owner-supplied, confirms the same form as a
third primary source (dossier section 7) and contributes the
magic-m_J^2 intercept regression.*
*1.2.0 (2026-08-11): confirmed by G7 theory sign-off, 2026-08-11: §13
added (E32 BBR pivot term, corrected sign; E33 scalar pivot composition,
extended with the BBR term and the 4th-order hyperpolarizability neglected-
term bound).*
*1.1.0 (2026-08-09): confirmed by independent theory review, 2026-08-09
(all six items confirmed): E29 scope sentence added (Stark-only, no
motional Doppler); ALPHA_AU_TO_SI pinned at full precision
1.64877727436e−41.*
*1.1.0-draft (2026-08-09): §12 added: E29 lattice fast path (exact),
E30 secular averaging, E31 step-size rule; E14b implementation path
activated. Awaiting independent theory review.*
*1.0.0 (2026-08-08): confirmed by independent theory review, 2026-08-08;
§11 items resolved per theory review; E14 split into E14a (MVP) / E14b
(physical coupling); E24 acceptance criterion added; constants notes
pinned.*
*1.0.0-draft (2026-08-08): initial transcription; awaiting theory review.*
