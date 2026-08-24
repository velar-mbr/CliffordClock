# Physics & Numerical Conventions: CliffordClock

**Version:** 1.9.0 · **Status: reviewed and approved** (2026-08-11, per
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

---
*Changelog:*
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
