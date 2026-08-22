# G11: Independent review of E38, the quantum-motional second-order-Doppler (time-dilation) pivot term (WP30)

Reviewed: branch `motional-time-dilation` in this repository, the uncommitted
diff touching `docs/CONVENTIONS.md`, `docs/cli.md`, `src/cliffordclock/integrator/omega.py`,
`src/cliffordclock/pipeline.py`, `benchmarks/loaders.py`, and `tools/bibliography.toml`,
plus the new `tests/test_motional_pivot.py`, `benchmarks/run_motional_al_ion.py`, and
`benchmarks/results/wp30_motional_al_ion_arithmetic_reproduction.{json,md}`. Reviewer ran
every check with `.venv/bin/python`. This record covers both the physics gate (Part A) and
the code review (Part B) for E38, mirroring G10's two-part structure since E38, like E37,
carries no separate spacetime-algebra formalism sign-off of its own.

## Part A: physics gate

### A1. The formula: QHO velocity variance and the second-order-Doppler relation

Verdict: PASS.

**QHO velocity-variance expectation, derived independently.** For a 1D quantum harmonic
oscillator (mass `m`, angular frequency `omega`), with ladder operators `x = sqrt(hbar/(2 m
omega))(a + a-dagger)`, `p = i*sqrt(hbar m omega/2)(a-dagger - a)`: `p^2 = -(hbar m
omega/2)(a-dagger - a)^2`. Expanding and taking `<n|...|n>` for a Fock (or thermal-mixture)
state, the `a^2`/`a-dagger^2` terms vanish (off-diagonal) and `a-dagger a + a a-dagger =
2n+1` gives `<p^2> = hbar*m*omega*(n+1/2)` exactly, so `<v^2> = <p^2>/m^2 = (hbar*omega/m)*(n
+ 1/2)`. Because this is linear in the occupation number, the same relation holds exactly for
a thermal mixture with mean occupation `n_bar` in place of `n`. Summed over `N` independent
normal modes, `<v^2> = sum_i (hbar*omega_i/m)*(n_bar_i + 1/2)`, exactly the formula
`motional_mean_squared_velocity_m2_s2` implements (`src/cliffordclock/integrator/omega.py`).

**Second-order-Doppler relation.** Proper time `dtau = dt*sqrt(1 - v^2/c^2)`, so the
fractional rate shift is `sqrt(1-v^2/c^2) - 1 ~= -v^2/(2c^2)` for `v << c` -- the same
kinematic factor E15/E21 already use elsewhere in this document. Substituting the quantum
expectation value `<v^2>` for the classical `v^2` (the standard treatment for a bound
particle's motional-state contribution to this shift) gives `(P-1)_motional =
-<v^2>/(2c^2)`, matching E38 and `motional_pivot_perturbation` exactly.

**Unit/hand computation, Al+ scale, chosen independently of the diff's own test cases.** For
a single mode at `f = 2 MHz`, `n_bar = 0.05`, Al27+ (`m = 26.98153853 u =
4.4803898868635304e-26 kg`, the registry's own `AL27_PLUS.mass_kg`, confirmed by direct
import):

    omega = 2*pi*2e6 = 1.2566370614359172e7 rad/s
    <v^2> = (HBAR*omega/m)*(0.05+0.5) = 0.016267952889333213 m^2/s^2
    (P-1)_motional = -<v^2>/(2*c^2) = -9.050269347097112e-20

A hand script (plain Python, not calling anything from `cliffordclock`) reproduces this
exactly; calling the real `motional_pivot_perturbation` on the same input returns
`-9.050269347097112e-20`, bit-identical. Units check end to end: `[hbar*omega/m] =
J*s*s^-1/kg = (kg*m^2*s^-2)/kg = m^2/s^2`, correct for a velocity-squared term, and dividing
by `2c^2` (m^2/s^2) leaves a dimensionless fractional shift.

### A2. The no-double-count argument

Verdict: PASS.

Confirmed by direct code inspection, not just the docstring's claim:

- **Static nodes in every evaluation mode, including `worldline`.** For `ensemble.regime:
  lattice`/`lattice_extended`, `mode="worldline"` builds `traj_dense =
  jnp.broadcast_to(nodes[:, None, :], (nodes.shape[0], n_steps + 1, 3))`
  (`src/cliffordclock/pipeline.py`, both the `lattice` and `lattice_extended` branches), and
  `_stark_rotor_ensemble`'s step body computes `v = (pos_b - pos_a) / dt_phys` from
  consecutive trajectory samples. Because every sample in `traj_dense` is the identical
  broadcast value, `pos_b - pos_a` is bit-exact `0.0`, not merely numerically small, so `v =
  0.0` exactly at every step. `mode="fast_path"` never constructs a trajectory at all
  (`fastpath.lattice_shift_expectation` evaluates the quadrature nodes' positions once, with
  no time-dependence), and its `rate_fn`'s own kinematic term (`kinematic = -x/(1+gamma_inv)`
  with `x = v2/c^2`) evaluates to exactly `0.0` at `v=0`, confirmed by running a live config
  (see A2 verification run below) and by the module's own `gamma_inv=1.0` composition
  reaching `p_minus_1` unmodified.
- **Rejection at parse time.** `PipelineConfig.from_dict` (`src/cliffordclock/pipeline.py`,
  around line 1142) raises `PipelineConfigError` when `environment.motional_state is not
  None and ensemble.regime == "classical"`, naming the double-counting hazard directly in the
  message (confirmed the message text names both `sqrt(1 - v^2/c^2)` and `double-count`
  literally). The reviewer called `PipelineConfig.from_dict` directly (no `run_pipeline_full`
  needed) with `ensemble.regime="classical"` plus `environment.motional_state` set and
  confirmed the raise; `tests/test_motional_pivot.py::test_environment_motional_state_rejects_classical_regime`
  exercises the identical path.
- **Composition point confirmed.** `spin_connection_stark`'s `P` denominator is built as `p =
  1.0 + pivot_perturbation_stark(..., motional_pivot_perturbation=...)`, and the returned
  numerator `d_p_minus_1_dr` is computed purely from `e_total`/`grad_e_total`, never touching
  `motional_pivot_perturbation` -- the motional term reaches only the denominator, never the
  gradient, bit-for-bit the same code shape as `bbr_pivot_perturbation`'s existing
  "denominator only" pattern (verified by reading the function body, not just its docstring).

### A3. THE CRUX: the geometric-factor question

Verdict: PASS, after one fix loop.

**Independent recomputation of the engine's own number.** A 50-digit `decimal` script, coded
directly from the six `(frequency_MHz, n_bar, n_bar_uncertainty)` triples in
`benchmarks/loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR` and the registry's Al27+ mass, gives
`(P-1)_motional = -115.08709195095335462979530429430277239188365332161e-19` and a
propagated 1-sigma uncertainty of `2.7129764380042009243983780883012306392901325688747e-19`
(n_bar-uncertainty terms only, since Table S2 publishes no frequency uncertainty). Running
`benchmarks/run_motional_al_ion.py` (the real engine calls) gives `-115.0870919...e-19`
and `+/-2.713e-19`, matching to `4.9e-20` absolute (float64 noise from the Decimal
comparison, not a discrepancy). The task-supplied headline number, `-115.09(2.71)e-19`
against Marshall's published `-114.6(3.8)e-19`, is confirmed; the two bands
(`[-1.178e-17, -1.1237e-17]` and `[-1.184e-17, -1.108e-17]`) do overlap under
`run_benchmarks._bands_overlap`'s documented closed-interval definition, so `kpi_verdict =
"MET"` is arithmetically correct as computed.

**Primary-source verification of kappa (arXiv:2504.13071v2, main text and Supplemental
Material, fetched and extracted via `pdftotext -layout`, not the lossy HTML/abstract-page
summarizer).** The paper's *only* two occurrences of kappa are:

    n_bar_Dopp = (gamma / (2*pi*f_i)) * kappa_i                (Eq. 1)

with the surrounding text: "gamma is the Doppler cooling transition linewidth, the Doppler
cooling laser detuning is gamma/2, f_i is the secular frequency for motional mode i, and
kappa_i is a geometric factor equal to 1.7 for axial modes and 2.3 for radial modes,"
immediately followed by "We find that the measurements agree within their uncertainty with
the calculated Doppler limit" -- and Table S2's "Geometric factor kappa" row, which exists
specifically to let the reader reproduce Table S2's own adjacent "Calculated Doppler limit
n_bar" row via Eq. 1. **Kappa is exclusively a Doppler-cooling-laser-geometry factor used to
predict/cross-check the achievable cooling-limit occupation number.** It is never defined
as, or used as, a mass- or amplitude-partition factor in the secular-motion time-dilation
formula anywhere in the paper. The paper's actual per-mode time-dilation weighting appears
only as a black-box "Frequency shift per quantum (1e-19)" row in Table S2 (`-0.95, -1.42,
-1.77, -6.48, -1.42, -6.53` for Axial COM/STR, X COM/STR, Y COM/STR respectively), with no
derivation given anywhere in the main text or supplement.

**The initial caveat misidentified kappa.** The diff as first reviewed
(`benchmarks/run_motional_al_ion.py`, then at lines 65-77 and 168-178) stated the geometric
factor kappa is what "partitions each mode's kinetic energy between the two different-mass
ions" and that E38 "does not consume kappa at all" as the documented gap -- i.e. it identified
kappa as the missing ion-mass-partition physics. That identification was factually wrong per
the primary source above.

**Testing whether kappa is nonetheless a usable proxy for the missing per-mode weight, and
whether a completeness/sum-rule argument closes the gap.** The reviewer computed the
engine's naive per-mode coefficient `-(hbar*omega_i)/(2*m_Al*c^2)` for all six modes and
compared to Table S2's real "frequency shift per quantum" column:

    mode        f_MHz  n_bar  naive(1e-19)  real(1e-19)  ratio(naive/real)  kappa
    axial_com    2.16   8.22      -1.7771       -0.9500             1.8707    1.7
    axial_str    3.75   4.50      -3.0853       -1.4200             2.1728    1.7
    x_com        4.22   5.68      -3.4720       -1.7700             1.9616    2.3
    x_str        3.48   6.69      -2.8632       -6.4800             0.4418    2.3
    y_com        5.37   4.31      -4.4182       -1.4200             3.1114    2.3
    y_str        4.75   4.84      -3.9081       -6.5300             0.5985    2.3

The naive/real ratio ranges from `0.44` to `3.11` -- a factor of `~7` spread -- and does not
track kappa at all: `x_com` and `y_str` share kappa `2.3` but have ratios `1.96` and `0.60`
(opposite sides of 1). Dividing the naive coefficient by kappa (the operation "consuming
kappa" would mean) does not reproduce the real values either: for `x_str`, `naive/kappa =
-1.245e-19` against a real value of `-6.48e-19`, off by a factor of `>5`, and in the *wrong
direction* (kappa only ranges 1.7-2.3, a 1.35x span, nowhere near enough to explain a 7x
per-mode spread). This is a second, independent confirmation (numerical, not just textual)
that kappa is not the physical quantity governing the per-mode time-dilation weight.

To check whether a genuine completeness identity could still explain the close *total*
agreement despite this large *per-mode* disagreement, the reviewer derived the general
two-ion coupled-oscillator normal-mode decomposition from scratch: writing the mass-weighted
transformation to normal coordinates as an orthogonal matrix `O`, ion 1's true velocity
variance is `<v1^2> = (1/m1)*sum_j O_1j^2 * hbar*omega_j*(n_bar_j+1/2)`, and orthogonality of
`O`'s row guarantees `sum_j O_1j^2 = 1` exactly, for *any* spring constants or masses -- a
real, provable identity, not numerology. The reviewer solved this system numerically for the
axial COM/STR pair (`f = 2.16, 3.75 MHz`, `m_Al = 27u`, `m_Mg = 25u`) and confirmed
`O_1,com^2 + O_1,str^2 = 1.0000` to five decimal places, verifying the identity holds for
this system. **However, this identity alone does not imply `sum_j O_1j^2 * W_j = sum_j W_j`**
(the naive engine sum) unless `W_j = hbar*omega_j*(n_bar_j+1/2)` is uniform across the modes
sharing that identity, which it plainly is not here (frequencies span 2.16-5.37 MHz, `n_bar`
spans 4.3-8.2). The observed per-mode ratios above (0.44 to 3.11, not clustered near a
common value) confirm this non-uniformity is real, not a modeling artifact. The reviewer
cannot, from the published data alone, derive a rigorous closed-form guarantee that the
*six-mode total* should land within 0.4% of the naive single-mass sum; the orthogonality
identity is real but insufficient by itself to certify the observed level of agreement.

**Ruling on A3(a) (meaningful reproduction, accidental agreement, or in-between):**
in-between, leaning toward the diff's own honest framing but for the wrong stated reason as
originally written. There is a genuine, derivable completeness/orthogonality mechanism at play
(confirmed above) that is qualitatively consistent with why a naive single-mass sum over a
*complete* mode set can land in the right neighborhood even though no individual per-mode term
is correct -- but the mechanism has nothing to do with "kappa," which is not the relevant
physical quantity at all (the real per-ion participation factor `O_1j^2` is never published by
Marshall and is not equal to, or simply derivable from, kappa). The close 0.4%-level *total*
agreement is real and independently reproduced, but its precision is better than the reviewer's
own derivation can certify from public information, and should not be presented as resting on
a completeness-property argument that implicitly invokes kappa's role incorrectly.

**Ruling on A3(b) (is the classification defensible as labeled):** the structural
classification (`case_class = "arithmetic_reproduction"`, excluded from `kpi_summary`,
`kpi_verdict = "MET"` via band overlap) was already sound and needed no change -- the
underlying numbers are correct and independently reproduced throughout. The caveat text needed
correction before this ships in front of the NIST group whose paper it cites, on three points:
(1) remove the claim that kappa "partitions each mode's kinetic energy between the two
different-mass ions" / "reflects each ion's amplitude in each normal mode" -- per Eq. 1 and
Table S2's own text, kappa is a Doppler-cooling-laser geometric factor, unrelated to the
secular-motion time-dilation formula; (2) if a completeness/sum-rule argument is invoked, do so
honestly, in terms of the real (unpublished) per-ion eigenvector participation factor,
explicitly flagged as not derivable from Marshall's published quantities, not "kappa"; (3)
state plainly that the mechanism behind the total-level agreement is not established, rather
than implying it is a known, if unverified, identity.

**The fix.** The builder rewrote `benchmarks/run_motional_al_ion.py`'s module docstring and
its `GEOMETRIC_FACTOR_CAVEAT` constant, and `benchmarks/loaders.MARSHALL_AL_ION_MODES_CITATION`,
replacing the kappa attribution with the correct statement of the missing physics: each ion's
own per-mode normal-mode amplitude (an ion-and-eigenvector-dependent participation factor, one
per ion per mode), explicitly distinguished from kappa. Kappa now appears exactly once, in a
single clarifying aside, correctly identified as the paper's own Doppler-cooling-laser geometry
factor from Eq. 1 with no role in the secular-motion row. The rewritten text states the
per-mode-differs/total-agrees distinction explicitly (per-mode contributions differ from
Marshall's own values by up to several-fold, matching the ratio table above, while the
six-mode total reproduces the published band); cites this record by name for the orthogonality
identity and repeats its qualification that the identity is "qualitatively consistent with"
the total-level agreement but "does not certify the observed precision," reporting the
mechanism as "an open empirical observation, not a proven identity"; narrows the
classification paragraph's claim to total-level reproduction only ("not... a per-mode
reproduction of Marshall's own values... does not demonstrate that this project's simplified
single-species-mass model matches Marshall's true per-ion motional physics mode by mode"); and
names the open item as a future two-mass normal-mode treatment (per-ion amplitude vectors as
an explicit input), filed alongside the RF/micromotion-dynamics package CONVENTIONS.md section
16 already scopes out of this tier.

**Re-verification.** The reviewer read the rewritten module docstring and `GEOMETRIC_FACTOR_CAVEAT`
in full against all three points from ruling A3(b) above and confirmed each is addressed: kappa
appears exactly once, in the single clarifying sentence described above, never as an
explanation for the missing physics or for the total-level agreement; the two-ion per-mode
amplitude-partition scope boundary is stated in physically accurate terms; the
per-mode-differs/total-agrees distinction is explicit and consistent with the 0.44x-3.11x
range this record establishes above; the orthogonality identity is attributed to this record by
name with its "does not certify the observed precision" qualification preserved in substance;
the classification paragraph no longer claims per-mode reproduction; and the open item (a
future two-mass normal-mode treatment, filed with the RF-dynamics package) is named plainly.
The reviewer independently regenerated `benchmarks/results/wp30_motional_al_ion_arithmetic_reproduction.{json,md}`
by re-running `benchmarks/run_motional_al_ion.py` directly and confirmed the output is
identical, field for field (aside from the timestamp), to the version already carried in the
diff, and that the `geometric_factor_caveat` field contains no occurrence of the word "kappa"
at all. The predicted value, uncertainty, and `kpi_verdict` are unchanged from the independent
recomputation above: `-115.0870919...e-19 +/- 2.713e-19` against Marshall's published
`-114.6(3.8)e-19`, bands overlapping, `"MET"`.

### A4. The EMM input channel

Verdict: PASS.

`<v^2> = sum_i(...) + v_rms_emm^2` is dimensionally sound (m^2/s^2 added to m^2/s^2) and
physically standard: EMM and secular motion are independent (uncorrelated, different
frequency regimes -- RF drive frequency vs. secular trap frequency), so their velocity
variances add incoherently, exactly as implemented. `-<v_emm^2>/2c^2` is the same
second-order-Doppler form applied to the EMM-equivalent velocity, consistent with A1's
derivation. The full-RF-dynamics boundary is stated explicitly in three places: CONVENTIONS.md
section 16 ("This project does not model the RF trap dynamics that PRODUCE EMM... a genuine
roadmap package"), `docs/cli.md`'s new section, and `pipeline._MOTIONAL_EMM_ROADMAP_NOTE`,
which the reviewer confirmed is actually emitted in a live report (see B6 below) and cites
Berkeland, Miller, Bergquist, Itano, Wineland, J. Appl. Phys. 83, 5025 (1998) -- an entry
already present in `tools/bibliography.toml` (`Berkeland1998`, pre-existing, unedited by this
diff) and already used identically in `cliffordclock.ensemble.species.ION_MICROMOTION_NOTES`.

### A5. Bylines and status

Verdict: PASS.

The reviewer fetched `arxiv.org/abs/2504.13071` directly and confirmed the 13-author byline
in `tools/bibliography.toml`'s new `Marshall2025` entry ("Marshall, Mason C. and Rodriguez
Castillo, Daniel A. and Arthur-Dworschack, Willa J. and Aeppli, Alexander and Kim, Kyungtae
and Lee, Dahyeon and Warfield, William and Hinrichs, Joost and Nardelli, Nicholas V. and
Fortier, Tara M. and Ye, Jun and Leibrandt, David R. and Hume, David B.") matches the arXiv
abstract page's byline exactly, same order, same 13 names. The paper's own PDF footer confirms
"Dated: Draft: July 16, 2025" -- consistent with the diff's "preprint as fetched; no journal
acceptance confirmed at fetch time" labeling in both `tools/bibliography.toml`'s `venue` field
and `benchmarks/loaders.MARSHALL_AL_ION_MODES_CITATION`. `.venv/bin/python
tools/release_checks.py --only citation-check` passes with zero findings. Every other citation
touched by this diff (Brewer et al., Phys. Rev. Lett. 123, 033201 (2019); Berkeland et al., J.
Appl. Phys. 83, 5025 (1998)) is a pre-existing, unedited `tools/bibliography.toml` entry
reused verbatim, confirmed by diffing the bibliography file (only one new `[[paper]]` block,
`Marshall2025`, was added).

## Part A verdict: PASS, approve.

A1, A2, A4, and A5 all PASS. A3 required one fix loop: the caveat in
`benchmarks/run_motional_al_ion.py` (and its copies in
`benchmarks/loaders.MARSHALL_AL_ION_MODES_CITATION` and the generated
`benchmarks/results/wp30_motional_al_ion_arithmetic_reproduction.{json,md}`) initially
misidentified Marshall et al.'s "geometric factor kappa" as the ion-mass/amplitude-partition
quantity missing from E38's single-species-mass formula, when the primary source defines
kappa exclusively as a Doppler-cooling-limit geometric factor (Eq. 1), unrelated to the
secular time-dilation formula. The underlying arithmetic and the `MET` band-overlap verdict
were correct throughout and did not change; the caveat's physics narrative was rewritten to
name the correct missing quantity (each ion's own per-mode normal-mode amplitude), state the
per-mode-differs/total-agrees distinction explicitly, cite this record's orthogonality
identity with its precision caveat intact, and narrow the classification claim to total-level
reproduction. The reviewer independently confirmed the rewritten text and the regenerated
result files directly (A3). `motional_pivot_perturbation`'s implementation was correct
throughout per A1/A2/A4 and required no change.

## Part B: code review

### B1. Threading parity with the BBR/grav pattern

Verdict: PASS.

`motional_pivot_perturbation` is threaded as a keyword-only parameter through
`pivot_perturbation_stark`/`spin_connection_stark`/`scalar_rate_perturbation_stark`/
`build_omega_stark` (all default `0.0`, an exact IEEE-754 no-op) exactly mirroring
`bbr_pivot_perturbation`'s signature shape. `spin_connection_stark`'s own code (not just its
docstring) confirms the denominator-only composition: `p = 1.0 +
pivot_perturbation_stark(..., motional_pivot_perturbation=...)` while the returned
`d_p_minus_1_dr / p[..., None]` numerator is built purely from `e_total`/`grad_e_total`. The
docstring's stated reason -- "this project's motional state is one state per run, spatially
uniform across the atom cloud exactly like BBR's single radiation temperature... so `grad ln
P_motional = 0` exactly" -- is the identical spatial-uniformity argument BBR's own docstring
note gives, word-for-word in structure. `cliffordclock.pipeline._make_stark_rate_fn`/
`_stark_rotor_ensemble` both gained a `motional_pivot_perturbation: float = 0.0` parameter
threaded into every `build_omega_stark`/`pivot_perturbation_stark` call inside them, matching
`bbr_pivot_perturbation`'s existing threading exactly.

### B2. Kill-test quality

Verdict: PASS.

The reviewer copied `src/` and `tests/` into a scratch directory
(`/private/tmp/.../scratchpad/killtest_copy`) and ran the real test suite against two
mutated copies of `motional_mean_squared_velocity_m2_s2`, using `PYTHONPATH` to shadow the
editable-installed package:

- **Bug 1** (drop the `+ 0.5` zero-point term): 5 tests fail, including the dedicated
  `test_motional_pivot_kill_test_forgotten_zero_point_term`, and also
  `test_motional_pivot_two_mode_hand_computed_regression`,
  `test_motional_pivot_ground_state_limit_is_nonzero_zero_point_floor`,
  `test_motional_pivot_uncertainty_frequency_partial_matches_finite_difference`, and
  `test_motional_fast_path_matches_worldline_rotor_crosscheck` (this last one via a real
  pipeline run, not a direct function call).
- **Bug 2** (drop the `2*pi` conversion, using `frequency_hz` as if already angular): 5 tests
  fail, including the dedicated
  `test_motional_pivot_kill_test_used_frequency_not_angular_frequency`, and also
  `test_motional_pivot_two_mode_hand_computed_regression`,
  `test_motional_pivot_ground_state_limit_is_nonzero_zero_point_floor`,
  `test_motional_pivot_uncertainty_n_bar_partial_matches_finite_difference`, and
  `test_motional_fast_path_matches_worldline_rotor_crosscheck`.

Both dedicated kill tests genuinely discriminate the bug they name, and both bugs are also
caught redundantly by several other tests in the file (a healthy, non-brittle test design).
Restoring the original file, all 47 tests in `tests/test_motional_pivot.py` pass.

### B3. Tolerance discipline

Verdict: PASS.

Every `np.testing.assert_allclose` call in `tests/test_motional_pivot.py` (16 call sites)
carries an explicit `rtol=` and `atol=` -- no bare/default-tolerance calls.
`.venv/bin/python tools/release_checks.py --only tolerance-scan` reports zero findings. Loose
tolerances are justified in comments where used: `rtol=2e-4` on the EMM finite-difference
partial explains the O(eps/base_v) curvature-truncation error from the term's exact
quadratic-in-v_rms_emm form; `atol=1e-25`/`1e-28`/`1e-30` on composition-additivity checks are
below the float64 noise floor of the ~1e-19-to-1e-17-scale quantities being compared, not
vacuous slack.

### B4. Parse-time rejection at both omega and pipeline layers

Verdict: PASS.

`omega._validate_motional_modes` (called by all three public entry points --
`motional_mean_squared_velocity_m2_s2`, `motional_pivot_perturbation`,
`motional_pivot_uncertainty`) rejects empty mode tuples, `frequency_hz <= 0`, `n_bar < 0`,
negative uncertainties, and negative `v_rms_emm_m_s`, raising `ValueError`. Independently,
`pipeline._parse_motional_state` re-checks every one of these same invariants at YAML-parse
time, raising `PipelineConfigError` before any engine call -- genuine defense in depth, not a
single shared code path. The classical-regime double-counting rejection lives in
`PipelineConfig.from_dict` (confirmed callable and raising without needing
`run_pipeline_full`). One non-blocking observation: like every other cross-field check in this
class (`coupling.type='stark_dc'` requirements for BBR/gravity/quadrupole/motional alike),
this validation is enforced only in `from_dict`/`from_yaml`, not in a `PipelineConfig.__post_init__`
-- a caller constructing `PipelineConfig` directly (bypassing the documented `from_dict`/
`from_yaml` entry points) would bypass the classical-regime guard. This is a pre-existing,
uniform project convention across every cross-field check in the class, not something WP30
introduces or weakens, and `from_dict`/`from_yaml` are the only construction paths the CLI,
docs, and every test use -- NOTE, not a blocker.

### B5. Design judgment calls

The reviewer could not locate a WP30 builder report enumerating five specific AMBIGUITY items
in this repository or its worktrees (`.claude/worktrees/*` contain unrelated, older CI work).
The judgment calls actually visible in the diff, ruled on directly:

1. Outright rejection of `ensemble.regime="classical"` (no partial/scaled composition option
   for a user who might want just the EMM/zero-point contribution under classical sampling):
   accept. Conservative, fail-closed, and the double-counting hazard it guards against is real
   (A2).
2. The geometric-factor/kappa caveat and its completeness-property framing in
   `benchmarks/run_motional_al_ion.py`: needed correction, resolved -- see A3 above for the
   fix and its independent re-verification.
3. Choosing Marshall 2025 over Brewer 2019 as the WP30 arithmetic-reproduction source, leaving
   Brewer's own `-17.3(2.9)e-19` unverified: accept. The docstring's stated reason (Brewer's
   own formula is a time-dependent heating-rate integral, not the static `n_bar+1/2` form E38
   implements) is accurate and the choice is transparently disclosed, not silently
   substituted.
4. `frequency_uncertainty_hz` silently defaulting to `0.0` per mode when Table S2 publishes no
   frequency uncertainty (all six Marshall modes): accept, with a note. This understates the
   propagated uncertainty relative to a full replication of Marshall's own error budget, but
   works conservatively for the `MET` verdict (a narrower predicted band still overlaps the
   published band), and is disclosed in the docstring ("frequency uncertainties are not
   published in Table S2, so `frequency_uncertainty_hz` is left at its `0.0` default").
5. `math.fsum` for the per-mode compensated summation in
   `motional_mean_squared_velocity_m2_s2`/`motional_pivot_uncertainty`: accept. Matches the
   project's established E10 precision-discipline convention, identical to
   `_bbr_weighted_moments`'s use of the same primitive.

### B6. Report-note reconstructability

Verdict: PASS.

The reviewer ran a live config (Al27+, `stark_dc`, `ensemble.regime=lattice`,
`integration.mode=fast_path`, two modes `axial`/`radial` with `n_bar_uncertainty`/
`frequency_uncertainty_Hz` set on one mode each, `v_rms_emm_m_s=0.02`,
`v_rms_emm_uncertainty_m_s=0.005`) via `python -m cliffordclock.cli run`. The resulting
`uncertainty_notes` lists both modes by name with frequency, `n_bar`, and their uncertainty
fields; the resolved `<v^2>=0.04920385866799964 m^2/s^2` (hand-verified: single-mode
`0.05`-occupation term at 2 MHz from A1, `0.016267952889333213`, plus the 4 MHz mode's exactly
double value `0.032535905778666426`, plus `v_rms_emm^2=0.0004`, summing to
`0.049203858668...`, matching to the last printed digit); `(P-1)_motional
=-2.737333805250206e-19` (matches `-<v^2>/(2c^2)` to the last digit); the `v_rms_emm_m_s=0.02`
input; the propagated uncertainty (`1.99e-21`); and the excess-micromotion roadmap note
verbatim, including the Berkeland citation. The `fast_path` note correctly supersedes:
`_FAST_PATH_MOTIONAL_INCLUDED_NOTE` appears (not `_FAST_PATH_DOPPLER_EXCLUSION_NOTE`), stating
plainly that `mean_fractional_shift` DOES include the motional term for this run. `mean
fractional shift: -2.737334e-19` in the CLI summary matches `(P-1)_motional` exactly (the
Stark term is exactly zero for `e0=[0,0,0]`), confirming end-to-end reconstructability from
the report text alone.

### B7. mypy strict, ruff, prose-scan, citation-check

Verdict: PASS.

`.venv/bin/python -m mypy` (project's own `pyproject.toml` config: `strict = true`,
`python_version = "3.12"`) on `src/cliffordclock/integrator/omega.py`,
`src/cliffordclock/pipeline.py`, `benchmarks/run_motional_al_ion.py`, and
`benchmarks/loaders.py`: no issues. `.venv/bin/python -m ruff check` and `ruff format --check`
on the same four files plus `tests/test_motional_pivot.py`: all checks passed, all files
already formatted. `.venv/bin/python tools/release_checks.py --only
prose-scan,tolerance-scan,citation-check,headline-check,internal-path-check`: all five PASS
(prose-scan's 48 MINOR "rather than" findings are pre-existing, unrelated to this diff,
confirmed by grepping the diff patches directly for banned phrases -- the one "not merely
bounded" match in the `omega.py` diff is unchanged context, not an added line).

## Part B verdict: PASS, approve.

B5 item 2 is a duplicate reference to the A3 fix loop (already counted under Part A), not a
separate code defect -- the code itself (threading, validation, tolerances, mypy/ruff
cleanliness, report-note content) had no issues at any point in this review.

## Overall verdict

**Approve E38 for WP30.** Part A: PASS. Part B: PASS. One fix loop was needed, in A3 (the
geometric-factor caveat); it is resolved and independently re-verified against the rewritten
text and the regenerated result files.

Every test module whose underlying functions gained the new `motional_pivot_perturbation`
parameter, or otherwise touch the changed code paths, passes in full: `test_stark_pivot.py`,
`test_bbr_pipeline.py`, `test_bbr_environment.py`, `test_bbr_pivot.py`,
`test_quadrupole_pivot.py`, `test_quadrupole_pipeline.py`, `test_gravity_pivot.py`,
`test_lattice_extended.py`, `test_integrator_stark_rotor.py`, `test_integrator_omega.py`,
`test_fastpath_lattice.py`, `test_fastpath_secular.py`, `test_e2e.py`, and
`test_motional_pivot.py`. A full collection pass over the entire `tests/` tree succeeds with
no import/collection errors, confirming nothing elsewhere in the project fails to load against
the changed signatures.

The engine's E38 implementation (`motional_pivot_perturbation`, `motional_mean_squared_velocity_m2_s2`,
`motional_pivot_uncertainty`), its threading through the Stark rate-function chain, its
config-parse-time validation and no-double-counting rejection, and its report-note content are
all independently verified correct: the QHO/second-order-Doppler derivation checks out from
first principles (A1), the static-node/no-double-count argument is true by direct code
inspection in every evaluation mode including `worldline` (A2), the EMM channel is
dimensionally and physically sound with its scope boundary correctly documented and cited
(A4), and the bibliography entry's byline and preprint-status labeling match the arXiv source
exactly (A5). Two independently-coded kill tests genuinely discriminate their named bugs,
confirmed by reintroducing both bugs directly in the engine formula and watching the correct
tests fail (B2); every new tolerance is explicit and justified (B3); rejection is enforced at
both the `omega.py` and `pipeline.py` layers (B4); and a live pipeline run confirms the report
note is fully reconstructable from its text alone (B6).

A3's fix loop is now closed. The initial caveat in `benchmarks/run_motional_al_ion.py` (and
its copies in `benchmarks/loaders.py` and the generated
`benchmarks/results/wp30_motional_al_ion_arithmetic_reproduction.{json,md}`) stated that
Marshall et al.'s per-mode "geometric factor kappa" (1.7 axial / 2.3 radial) was the ion-mass/
amplitude-partition quantity E38's single-species-mass formula omits. Primary-source
verification (arXiv:2504.13071v2, full text via `pdftotext`, not the abstract page) showed
kappa is defined once, in Eq. 1, exclusively as a Doppler-cooling-laser geometric factor used
to predict/cross-check the achievable cooling-limit occupation number, never used in, or
defined in relation to, the secular-motion time-dilation formula anywhere in the paper. A
direct numerical test (dividing each mode's naive single-mass coefficient by its kappa) failed
to reproduce Marshall's real per-mode "frequency shift per quantum" values by factors of up to
5x, in some cases in the wrong direction, confirming kappa was not a usable proxy for the
missing physics. The reviewer's own from-scratch two-ion normal-mode derivation confirmed a
genuine orthogonality identity (`sum_j O_1j^2 = 1` for a given ion across the complete mode
set sharing an axis) that is qualitatively consistent with why a naive single-mass sum over a
*complete* mode set can land near the correct total despite large per-mode errors -- but this
identity does not by itself certify the observed sub-percent agreement. The underlying
arithmetic (`-115.09(2.71)e-19` predicted vs. `-114.6(3.8)e-19` published, bands overlapping,
`kpi_verdict="MET"`) was correct throughout and did not change. The builder rewrote the caveat
in `benchmarks/run_motional_al_ion.py`'s module docstring and `GEOMETRIC_FACTOR_CAVEAT`
constant and in `benchmarks/loaders.MARSHALL_AL_ION_MODES_CITATION`, naming the correct
missing quantity (each ion's own per-mode normal-mode amplitude), mentioning kappa only once
as the paper's unrelated cooling-limit factor, stating the per-mode-differs/total-agrees
distinction explicitly, citing this record's orthogonality identity with its "does not certify
the observed precision" qualification intact, and narrowing the classification paragraph to
total-level reproduction only. The reviewer confirmed all of this directly in the rewritten
text and in the regenerated `benchmarks/results/wp30_motional_al_ion_arithmetic_reproduction.{json,md}`,
which reproduce the same `-115.09(2.71)e-19`/`"MET"` result and carry no occurrence of "kappa"
in the folded `geometric_factor_caveat` field.
