# G13: Independent review of E39, coherent (phase-resolved) rotor composition and
# Ramsey visibility, plus the E38 per-mode participation-factor extension (WP31)

Reviewed: branch `motional-time-dilation` in this repository, the uncommitted diff
touching `benchmarks/loaders.py`, `benchmarks/results/wp30_motional_al_ion_arithmetic_reproduction.{json,md}`,
`benchmarks/run_motional_al_ion.py`, `docs/CONVENTIONS.md`, `docs/cli.md`,
`docs/report-schema.md`, `src/cliffordclock/analytics/report.py`,
`src/cliffordclock/ensemble/classical.py`, `src/cliffordclock/integrator/omega.py`,
`src/cliffordclock/pipeline.py`, `tests/test_analytics_report.py`, `tests/test_e2e.py`,
`tests/test_motional_pivot.py`, and `tools/bibliography.toml`, plus the new
`src/cliffordclock/integrator/coherence.py` and `tests/test_coherent_visibility.py`.
Reviewer ran every check with `.venv/bin/python`, using independent scripts and a
scratch copy of the source tree (`/private/tmp/.../scratchpad/kill_test_scratch`) for
the B1 kill-test reproduction; no shipped test file was reused as a verification
oracle. Primary sources (Wübbena et al. 2012, James 1998, Marshall et al. 2025) were
fetched and checked directly (PDF text extraction plus page-image inspection for the
equations), not taken on the diff's own say-so. No still-running background suite was
relied on; the full fast-lane battery (`pytest -q -m "not slow"`) was run synchronously
and used only as a non-regression signal, never as the source of a physics or tolerance
verdict.

## Part A: physics gate

### A1. The E39 observable: Gaussian closure and the projection

Verdict: PASS.

**Derivation.** For accumulated phases `ΔΦ_k ~ N(μ, σ_Φ²)` and probability weights
`p_k` summing to 1, `M = Σ_k p_k R_k` with `R_k = cos(ΔΦ_k) + sin(ΔΦ_k)·e_12` is, under
the identification `e_12 <-> i`, exactly the complex sum `Σ_k p_k e^{iΔΦ_k}`. In the
large-ensemble (or exact-expectation) limit this is the characteristic function of a
Gaussian evaluated at argument 1: `E[e^{iΔΦ}] = e^{iμ - σ_Φ²/2}`. The plane projection
`sqrt(c²+s²)` of `c + s·e_12` is exactly `|c+si|`, so `V = |E[e^{iΔΦ}]| = exp(-σ_Φ²/2)`
and `phase = arg(E[e^{iΔΦ}]) = μ`, independent of `μ` for `V` and independent of `σ_Φ`
for `phase` -- both are the standard Ramsey-fringe visibility/phase.

**Independent numerical check** (own ensembles, seeds, and code path, not the shipped
tests): a pure-NumPy complex-arithmetic recomputation at three `(μ, σ)` pairs with
1.5-3M samples reproduced `exp(-σ²/2)` to `2e-4`-`1e-5` relative error and `phase = μ`
to `<1e-3` absolute, e.g. `μ=-0.4, σ=1.1`: `V_empirical=0.546069` vs
`V_theory=0.546074` (rel. err. `-1.07e-05`). Calling the shipped module directly
(`phase_to_rotor`/`coherent_rotor_composition`/`ramsey_visibility_and_phase`) on a
fresh `N(0.55, 0.42)` ensemble of 1.5M draws (seed `31415926`, not used anywhere in
the shipped tests) gave `V=0.9157139` against theory `0.9155777` and `phase=0.5499248`
against `μ=0.55`; called against the pre-existing, independently-reviewed
`cliffordclock.analytics.stats.coherence_function` on the identical `phi`/`weights`,
the module's own `M` scalar/`e_12` components (`0.7807045335044911`,
`0.47857321101123407`) matched `coherence_function`'s real/imaginary parts
(`0.7807045335044926`, `0.4785732110112341`) to 15 significant figures -- an exact
cross-check between two independently-implemented (complex-number vs. rotor-algebra)
computations of the same physical quantity, not a coincidence given both express the
same population-weighted coherent sum.

**Projection extracts the right observable.** `c=⟨M⟩_scalar`, `s=⟨M⟩_{e12}` are exactly
the real/imaginary parts of the Ramsey coherence's own complex representation; `V` and
`phase` as implemented are the standard visibility/fringe-phase pair, confirmed by the
`coherence_function` cross-check above (same endpoint value, independently computed).

### A2. THE FULL-ANGLE CRUX

Verdict: PASS -- the full-angle fresh-rotor construction is the unique correct
realization of the ratified rule, not merely an equivalent or defensible choice.

**The algebra.** `stepper.rotor_step` builds each step's dynamical-rotor factor as
`delta_r = exp(-0.5*dtau*omega)` (E19, spinor/half-angle convention: `dR/dτ̃ = -½ Ω R`),
and its own `PhaseIncrement.scalar` (the primary E21/E22 observable, accumulated into
`ΔΦ_k`) is read off `omega[..., IDX_E12] * dtau` directly -- the *undoubled* rate. Its
`PhaseIncrement.rotor` (the E24 cross-check) is `rotor_plane_angle(delta_r) =
-2*atan2(delta_r[e12], delta_r[scalar])`: the code explicitly **doubles** the raw
half-angle rotor's own bivector angle to recover the same physical quantity as
`dphase_scalar`. This means: for a trajectory confined to the `B̂_C` plane (no boost
component, exactly the uniform-field/pure-kinematic configuration WP31's own pipeline
test uses), the composed dynamical rotor `WorldlineResult.r_final`'s own `B̂_C`-plane
bivector angle is `ΔΦ_k/2` by construction -- multiplying half-angle exponentials of a
common bivector direction just adds their (half) angles. `ΔΦ_k` itself is the *full*
physical phase: confirmed independently by `coherence_function` (E26, already
reviewed/gated) already using `exp(i·ΔΦ_i(t))` with the identical, undoubled `ΔΦ` as
its own argument -- this precedent was set before WP31 and is untouched by this diff.

Physically: the atomic coherence `ρ_ge` in a two-level Ramsey experiment accumulates
phase at the *full* detuning rate (`ρ_ge ∝ e^{iΔω t}`), not a half-angle spinor rate --
the half-angle convention belongs to the GA rotor's own double-sided-sandwich transport
law (`R v R̃` doubles the effective rotation for spatial vectors), a bookkeeping
convention of *this engine's* rotor representation, not a statement about the physical
Ramsey phase itself. `ΔΦ_k`'s bivector angle (full) is therefore the correct argument
for `e^{iΔΦ_k}`, and `r_final`'s own angle (half, in the no-boost limit) is the wrong
one by a factor of 2 in angle, hence 4 in the Gaussian exponent (`(θ/2)²/2 = θ²/8` vs.
`θ²/2`).

**Numerical confirmation of the crux itself** (own script, sigma=0.6, N=2M): using the
full angle gives `ln(V)=-0.179822` against `-σ²/2=-0.180000`; using the half angle
gives `ln(V)=-0.044955` against `-σ²/8=-0.045000`; the ratio of the two log-visibilities
is `4.0000` to 4 significant figures -- exactly the factor-4 exponent error the task
brief anticipates, reproduced independently.

**Reading "identified with the even-grade content of the worldline rotor."** This
phrase specifies *what kind of object* represents a phase factor (an even-grade,
`B̂_C`-plane-confined multivector `c+s·e_12` with `c²+s²=1`), not that the specific
pre-existing dynamical rotor object (`r_final`) must be reused verbatim. Given `ΔΦ_k`
is independently defined (G0 item 3, pre-WP31) as *the* primary phase observable and
`e^{-iφ_n}` in the ratified rule's own language is manifestly a full-angle phase
factor, the only even-grade `B̂_C`-plane object whose angle equals `φ_n = ΔΦ_k` is
`exp(ΔΦ_k · B̂_C)` -- exactly what `phase_to_rotor` builds. Reusing `r_final` instead
would silently substitute a *different*, wrong-by-a-known-factor angle. The builder's
choice is correct, not merely one of several acceptable readings.

### A3. Squeezing sign convention and monotonicity

Verdict: PASS.

Convention: position-quadrature variance `∝ exp(-2r)`, velocity-quadrature variance
`∝ exp(+2r)`, `r>0` defined (explicitly, in `classical.py`'s docstring) to squeeze
position and antisqueeze velocity. This is a labeling choice, not one physics forces in
either direction, and it is stated once and applied consistently everywhere it's used
(`classical.py`, `pipeline.py`, `CONVENTIONS.md` §8) -- no internal contradiction found.
The product `exp(-r)*exp(+r)=1` preserves the phase-space area exactly, confirmed
algebraically and trivially in the code (`sigma_pos = ...*exp(-r)`,
`sigma_v = ...*exp(+r)`).

Gated to `regime="classical"` only, enforced in `_parse_ensemble` for both `"lattice"`
and `"lattice_extended"` (`PipelineConfigError`, confirmed by reading the diff and by
`test_squeezing_r_rejected_for_lattice_regime`/`..._lattice_extended_regime`).

**Monotonicity of `V(r)` for the pipeline test's configuration.** The test config uses
`field.synthetic.kind="uniform"` with `e0=[0,0,0]` and `coupling.mu=[0,0,0]`: the
field-coupled term of the perturbation phase is identically zero regardless of
position, so squeezing the position quadrature has *no* effect on `ΔΦ_k` in this
configuration -- only the velocity quadrature (driving the pure kinematic
second-order-Doppler term, `delta_omega = sqrt(1-v²/c²)-1`) matters. For
`v ~ N(0, σ_v²)`, `Var(v²) = 2σ_v⁴` is monotonically increasing in `σ_v²`, and
`σ_v² ∝ exp(+2r)` is monotonically increasing in `r`, so the phase spread `σ_Φ`
increases monotonically with `r` and `V=exp(-σ_Φ²/2)` decreases monotonically -- exactly
the claimed shape, and reproduced by direct execution (own script, not the shipped
test): `r=0.0..0.6` gives `σ_Φ=0.135..0.448`, `V=0.991..0.910`, strictly decreasing at
every step, matching `exp(-σ_Φ²/2)` computed from that same run's own phases to
relative deviations of `5.5e-5` (`r=0`) through `6.2e-3` (`r=0.6`).

### A4. Participation factors: Wübbena et al. 2012 transcription and framing

Verdict: PASS.

**Character-level check against the fetched primary source** (arXiv:1202.2730,
page-image inspection of the actual typeset equations, not OCR/`pdftotext` text which
garbles the nested-fraction layout): Eq. (14) reads exactly
`b1,z² = (1-μ+sqrt(1-μ+μ²)) / (2·sqrt(1-μ+μ²))`, matching `two_ion_participations`'
`b1_sq = (1.0 - mu + root) / (2.0 * root)` term-for-term. The in-phase/out-of-phase
mode identification is confirmed from the paper's own Eqs. (10)-(11) and the equal-mass
limit of Eqs. (12)-(13) (`ω_i,z(μ=1)=ω_z`, `ω_o,z(μ=1)=√3·ω_z`): the in-phase mode is
the lower-frequency COM mode, the out-of-phase mode is the higher-frequency stretch
mode, matching the code's `axial_com <- b1`, `axial_str <- b2` assignment.

**Equal-mass limit and sum rule.** At `μ=1`: `root=1`, `b1_sq=(1-1+1)/2=0.5`, matching
the claimed `1/2`. The paper states `b1²+b2²=1` explicitly (line following Eq. 9); the
code's `b2_sq = 1.0 - b1_sq` enforces this identically by construction. Verified
further by direct execution across five mass ratios (`μ` from 1 to 30), all summing to
`1.0` to `1e-13`.

**James 1998 claim.** Fetched the full text (arXiv:quant-ph/9702053): the potential
energy (Eq. 2.1) uses a single mass `M` for all `N` ions ("`M` is the mass of each
ion"), with no mass-ratio or mixed-species term anywhere searched (`grep` for "mass
ratio", "mixed species", "unequal mass" found zero hits). The `N=2` equal-mass special
case (Eq. 3.10) gives eigenvalue `μ₂=3` (stretch mode at `√3·ω_z`) and eigenvector
`b^(1)=(1/√2)(1,1)` (COM participation `1/2`) -- exactly `two_ion_participations`' own
`μ=1` limit. The bibliography entry's claim is accurate on both counts (equal-mass-only
scope, and the genuine `μ=1` cross-check).

**Bibliography framing.** The James1998 entry's `source` field states plainly that "the
mu-dependent unequal-mass eigenvector formula itself is NOT derivable from this paper
-- see the Wubbena2012 entry below for its actual primary source." This is the correct
framing: James 1998 is cited as a background cross-check (its `N=2` equal-mass special
case), not as the source of the closed form actually implemented. Properly framed --
PASS, not a citation-integrity concern.

**Radial scope caveat, independently verified against the primary source.** Fetched and
read the page images for Eqs. (15)-(18): `ω_{i/o,x,y}` and `b1,x,y²` depend on `μ` AND
`a`, and `a` (Eq. 18) is itself a function of both `μ` and `ε` -- confirmed the paper
defines `ε = ω_p/ω_z` (text preceding Eq. 12) with `ω_p` the RF pseudopotential
frequency and `ω_x, ω_y` split from it by the DC asymmetry parameter `α` (Eqs. 4-5,
Eq. 1's own trap-potential definition). This exactly matches the docstring's claimed
dependence on "alpha... and epsilon = omega_p/omega_z." Algebraic check: substituting
`ε=0` into Eq. (17)-(18) reduces `b1,x,y²` *exactly* to the axial `b1,z²` formula (Eq.
14) -- confirming `two_ion_participations`' radial approximation is precisely the
`ε=0` limit of the true radial closed form, not an arbitrary substitute. Since typical
linear Paul traps have `ε` far from 0 (radial confinement usually RF-dominated), this
also explains, from first principles, why the benchmark shows the radial pairs
disagreeing substantially while the axial pair (exact, no `ε` dependence) agrees to
sub-percent -- consistent, not merely asserted.

### A5. The two-variant benchmark story

Verdict: PASS.

Regenerated the benchmark independently (`.venv/bin/python benchmarks/run_motional_al_ion.py`)
and confirmed the output is byte-identical to the working tree's version modulo the
`generated_at_utc` timestamp -- the numbers in the diff are genuinely reproducible, not
hand-edited. Restored the working tree to its exact pre-run state afterward (verified
byte-for-byte) so this review stayed read-only outside the gate record. Cross-checked
`MARSHALL_AL_ION_FREQUENCY_SHIFT_PER_QUANTUM` against the fetched Marshall et al.
arXiv:2504.13071v2 PDF's Table S2 directly: `-0.95, -1.42, -1.77, -6.48, -1.42, -6.53`
(×1e-19) match character-for-character in the same column order (Axial COM, Axial STR,
X COM, X STR, Y COM, Y STR).

**(a) Radial-approximation disclosure and defensibility; the participation=1.0
alternative.** Computed, independently of the shipped code path, three variants of the
participation-corrected total using the engine's own `motional_pivot_perturbation`:

| Variant | Total (P-1)_motional | Overlaps published band `[-1.184e-17,-1.108e-17]`? |
|---|---|---|
| Shipped (μ-only formula, all 6 modes) | `-5.759e-18` | No |
| Hybrid (axial exact, radial forced to `participation=1.0`) | `-9.963e-18` | No |
| Single-mass (`participation=1.0` everywhere, WP30 baseline) | `-1.151e-17` | Yes |

Forcing radial participation to `1.0` does not fix the "NOT MET" verdict (still
outside the published band by ~13%, vs. the shipped variant's ~50%) -- there is no
free-lunch alternative available without a true `ε`-aware radial eigensolver. Per-mode,
the direction of error is mixed: the μ-only formula is *closer* to the published value
for the COM-type radial modes (`x_com` ratio `1.056` vs. what `participation=1.0` would
give, `~1.96`) but *farther* for the STR-type radial modes (`x_str` ratio `0.204` vs.
`participation=1.0`'s `~0.44`). Neither choice is uniformly better, and the disclosure
("the four RADIAL modes do NOT match well... this closed form cannot supply [the
trap-geometry dependence] from masses alone") is accurate as the blanket honesty
statement it's used as; it does not need to enumerate this per-mode mixed-direction
detail to be adequate. The current choice (documented μ-only approximation, radial
`kpi_verdict` reported honestly as its own `NOT MET`) is defensible and, on the totals
evidence above, no worse than the alternative the task raises.

**(b) Two-verdict labeling.** The JSON uses two distinct top-level keys
(`marshall_..._secular_motion_arithmetic_reproduction_case` vs.
`marshall_..._participation_corrected_variant_case`) and two distinctly-named verdict
fields (`kpi_verdict` for the single-mass case, `total_kpi_verdict` for the
participation-corrected case) -- not the same field name reused, so a reader cannot
merge or confuse them by key collision. The markdown renders them as clearly separate
sections with their own tables. Adequate.

**(c) Does axial agreement justify strengthened language; is any present that
shouldn't be?** Independently recomputed axial per-mode ratios:
`axial_com=1.0070` (0.70% high), `axial_str=1.0031` (0.31% high) -- both *better* than
the module's own "a few percent" characterization, i.e. the stated language is
conservative relative to the actual result, not inflated. No improper strengthening
found. (Note for the record: the task brief's stated range "axial per-mode agreement
0.3-7%" does not match this reviewer's recomputation from the shipped/regenerated
numbers, which gives 0.31%-0.70% for the two axial modes specifically; if the 7% figure
was meant to include the one radial mode that happens to land close, `x_com` at 5.60%,
that mode is still correctly classified `is_axial=False` in the code and not folded
into the "axial" language anywhere in the diff. This is a discrepancy in the review
brief's framing, not a defect found in the diff.)

### A6. Bylines

Verdict: PASS.

`Wubbena2012`: authors "Wübbena, Jannes B. and Amairi, Sana and Mandel, Olaf and
Schmidt, Piet O.", venue "Phys. Rev. A 85, 043412", year 2012 -- matches the fetched
arXiv:1202.2730 abstract page exactly. `James1998`: author "James, D. F. V.", venue
"Appl. Phys. B 66, 181", year 1998 -- matches CrossRef's authoritative metadata for
DOI 10.1007/s003400050373 exactly (`container-title="Applied Physics B: Lasers and
Optics"`, `volume=66`, `page=181-190`, `published-print=1998-02-01`, author
`D.F.V. James`); an initial arXiv-page fetch rendered the title with a stray comma
("...ions, with application...") not present in CrossRef's or the bibliography's
version, resolved as a fetch-summarization artifact, not a real discrepancy, by
checking the authoritative publisher record directly.

## Part A verdict: PASS, approve.

## Part B: code review

### B1. Kill-test reproduction

Verdict: PASS.

Reproduced both classic-error bugs by editing a scratch copy of
`coherence.py` (`rsync`'d `src`/`tests` only, outside the repo, PYTHONPATH-overridden
so the installed editable package resolved to the scratch copy) and running the
shipped, unmodified `tests/test_coherent_visibility.py` against each:

- **Bug (a), phase-averaging combiner** (`coherent_rotor_composition` rewritten to
  average each rotor's own extracted angle, then rebuild one unit rotor at the mean
  angle): `test_kill_test_a_phase_averaging_combiner_gives_visibility_one` failed, as
  intended (`assert abs(float(v_real) - v_bad) > 0.05` fails since both land at `1.0`).
  Four other tests (`test_visibility_equals_one_iff_all_phases_equal`,
  `test_gaussian_closure_identity_on_synthetic_ensemble`,
  `test_kill_test_b_renormalizing_combiner_gives_visibility_one`,
  `test_pipeline_squeezing_r_sweep_visibility_decreases_and_matches_closure`) also
  failed -- broad coverage, not a single brittle assertion.
- **Bug (b), renormalizing combiner** (`coherent_rotor_composition` normalizes `M` to
  unit norm before returning): the same five tests failed, including both named kill
  tests.

Restored the scratch file from a pre-edit backup after each run; reran the unmodified
file against the restored copy to confirm a clean pass (`15 passed`) before treating
the reproduction as concluded.

### B2. The no-opt-in report change

Verdict: ACCEPT.

`ramsey_visibility`/`ramsey_phase` populate unconditionally for `integration.mode` in
`{"direct","worldline"}` -- no config section gates it, unlike every other WP's report
addition (the updated `test_step0_linear_mu_output_unchanged_from_pre_step0_behavior`
docstring says this explicitly: "unlike every other WP's report-note addition, is not
gated behind an opt-in config section"). This is a real, self-flagged departure from
house style. Weighing it: the two new fields require no *new* physical input the user
must consciously supply (unlike participation, `environment.motional_state`, or
`squeezing_r`, all of which need a new config key) -- they are a derived statistic from
data (`ensemble_result.phase`, existing weights) the `direct`/`worldline` path already
computes for every such run, at negligible added cost. `REPORT_SCHEMA_VERSION` is
correctly bumped `1.0 -> 1.1` (this project's own stated bump rule: "on any
`MetrologyReport` field/type/shape change"), documented in `docs/report-schema.md`,
`docs/cli.md`, and `CONVENTIONS.md`'s changelog, and the one affected byte-exactness
test (`test_step0_linear_mu_output_unchanged_from_pre_step0_behavior`) was updated with
an explicit, honest note about the exception rather than silently patched. Given the
package's own compatibility posture (pre-1.0, `0.1.0.post1`, actively iterating report
schema across recent WPs), the schema-version-bump-plus-changelog discipline is
sufficient per this project's own established standard. No change requested.

### B3. Tolerance discipline

Verdict: PASS, with one NOTE.

The Gaussian-closure synthetic-ensemble test
(`test_gaussian_closure_identity_on_synthetic_ensemble`) carries a full, checkable
N-dependent derivation for its `rtol=1e-3` (cumulant expansion, `O(σ⁴/N)` scaling,
measured `~2e-5` at `N=200,000`, stated 50x margin) -- excellent discipline. The kill
tests and kill-test-adjacent assertions use `atol=1e-12`/`1e-13` for exact algebraic
identities (no `N`-derivation needed, correctly). `tests/test_motional_pivot.py`'s new
participation tests use `atol=1e-13`/`1e-14`/`1e-15` for closed-form/hand-derived
arithmetic, also correctly exact-tolerance, not MC-derived.

**NOTE:** `test_pipeline_squeezing_r_sweep_visibility_decreases_and_matches_closure`
uses `rtol=1.5e-2` comparing the pipeline's `ramsey_visibility` against
`exp(-σ_sample²/2)` from the same run's `N=6000` ensemble, with no comment deriving
this figure from `N` the way the synthetic-ensemble test does (only a comment
explaining the physical/numerical *setup* choices: field/coupling isolation, `dtau`
scale). Independently reproduced the actual relative deviations at each swept `r`
(own script, not the shipped test): `5.5e-5` (`r=0`) up to `6.2e-3` (`r=0.6`) -- so the
chosen tolerance carries a real but modest ~2.4x margin at the worst point, not the
~50x the other test documents, and not dangerously tight either. Not a blocker (the
value is empirically safe and the test passed on an independent rerun), but it is a
literal partial miss against this project's own stated tolerance-discipline bar ("the
MC-closure tolerances must derive from N, stated in comments"); a follow-up comment
deriving/bounding `1.5e-2` from `N=6000` analogous to the existing derivation would
close this gap.

### B4. Config validation at both layers

Verdict: PASS.

`participation`: range-checked `0 < p <= 1` both in `omega._validate_motional_modes`
(engine layer, exercised via `motional_pivot_perturbation`/`..._uncertainty`, defense
in depth for direct dataclass construction) and in `pipeline._parse_motional_state`
(parse layer, `PipelineConfigError`). Confirmed by reading both diff hunks and by the
new `test_participation_validated_in_range_zero_to_one` and
`test_environment_motional_state_rejects_bad_participation` tests, both of which pass.
`squeezing_r`: parse-layer finiteness check plus regime gate (`classical` only,
`PipelineConfigError` for `lattice`/`lattice_extended`) in `_parse_ensemble`, and a
runtime type check in `sample_maxwell_boltzmann` (`TypeError` if `trap` is not a
`HarmonicTrap`) -- confirmed `HarmonicTrap` is currently the only concrete `Trap`
implementation (`Trap` itself is a `Protocol`), so this is legitimate forward-looking
defense in depth, not dead code.

### B5. mypy strict, ruff, prose-scan, citation-check, internal-path-check

Verdict: PASS.

`.venv/bin/python -m mypy src/` (project's own `pyproject.toml`: `strict = true`,
`python_version = "3.12"`): `Success: no issues found in 27 source files`.
`.venv/bin/python -m ruff check .` and `ruff format --check .`: all checks passed, 144
files already formatted. `.venv/bin/python tools/release_checks.py --only
prose-scan,tolerance-scan,citation-check,headline-check,internal-path-check`: all five
PASS; the 48 MINOR "rather than" prose-scan findings are pre-existing across the wider
repo (docs, notebooks, benchmarks reports) and confirmed, by grepping the tool's own
output for the changed files' names, to contain zero hits inside any file this diff
touches or adds. (Running `mypy` directly on `benchmarks/run_motional_al_ion.py` in
isolation surfaces two missing-variable-annotation notes and import-untyped notes; this
is outside the project's own CI gate, which runs `mypy src/` only
(`.github/workflows/ci.yml`) -- not a regression against this project's actual
type-checking contract, and consistent with the pre-existing untyped style of
`benchmarks/run_benchmarks.py`, which errors identically in isolation. Not flagged as a
finding.)

### B6. The builder's five ambiguity flags

The reviewer could not locate a builder report enumerating five specific ambiguity
items for this diff, in this repository, its `plan/` tree, or the two unrelated
`.claude/worktrees/*` copies present (older doc/report work, not WP31). Following the
same resolution as G11's B5 in this situation, the five judgment calls actually visible
in the diff are identified and ruled on directly:

1. **Full-angle fresh-rotor construction instead of reusing `WorldlineResult.r_final`**
   (`coherence.phase_to_rotor`'s docstring, the diff's most explicit "deliberately NOT
   X, because Y" flag): correct, not merely defensible -- see A2's full derivation and
   numerical confirmation above.
2. **Reusing the axial-only (`μ`-only) closed form for the two radial mode pairs, as a
   "documented approximation, not an additional exact result"** (`two_ion_participations`
   docstring): defensible and well-disclosed -- see A4 (independently verified the true
   radial formula needs `ε` too, and that the used approximation is exactly its `ε=0`
   limit) and A5(a) (the `participation=1.0` alternative is not uniformly better).
3. **No config gate for `ramsey_visibility`/`ramsey_phase`, an explicit departure from
   this project's own per-WP opt-in-gating precedent** (self-flagged in the updated
   `test_e2e.py` docstring): accept -- see B2.
4. **The squeezing sign convention** (`r>0` squeezes position, antisqueezes velocity --
   `classical.py`'s docstring explicitly frames this as "the engine's convention," not
   a physics-forced choice): accept -- self-consistent throughout, arbitrary but
   clearly and singly stated, verified in A3.
5. **Keeping `James1998` in the bibliography as background despite it not being the
   source of the mass-ratio-dependent formula actually implemented** (its own `source`
   field states this outright): properly framed -- see A4/A6, verified against both the
   fetched arXiv text and CrossRef.

No sixth or hidden judgment call was found that materially changes any verdict above.

## Part B verdict: PASS, approve.

The B3 tolerance-derivation-comment gap and the B2 no-opt-in departure are recorded as
a NOTE and an accepted departure respectively; neither is a blocker, and both are
already the honest, disclosed choices of the diff rather than unexamined oversights.

## Overall verdict

**Approve E39 (WP31 coherent visibility) and the E38 participation-factor extension.**
Part A: PASS (A1-A6, no fix loop needed). Part B: PASS (B1-B6, no fix loop needed). The
shipped `V = exp(-σ_Φ²/2)` Gaussian closure identity, its full-angle construction, the
squeezing sign convention, the two-ion participation closed form and its disclosed
radial-approximation scope, and the two-variant benchmark's honest divergent verdicts
were all independently re-derived and/or re-fetched-and-checked against primary
sources in this review, not taken from the diff's own tests or its own citation
transcriptions. One NOTE is recorded for a future small documentation addition (B3);
no code or physics change is required before merge.
