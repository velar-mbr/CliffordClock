# G10: Independent review of E37, the multi-surface BBR thermal environment (WP29 Tier 1)

Reviewed: branch `bbr-thermal-environment` in this repository, the uncommitted
diff touching `docs/CONVENTIONS.md`, `src/cliffordclock/integrator/omega.py`,
and `src/cliffordclock/pipeline.py`, plus the new `tests/test_bbr_environment.py`.
Reviewer ran every check with `.venv/bin/python`. This record covers both the
physics gate (Part A) and the code review (Part B) requested for this work
package, since E37 has no spacetime-algebra content and so carries no separate
theory sign-off of its own.

## Part A: physics gate

### A1. Additivity of the multi-surface shift at second order

Verdict: PASS.

The BBR shift is a second-order (dynamic Stark) perturbative effect: Farley
and Wing, Phys. Rev. A 23, 2397 (1981), the standard framework for this
regime, derives the shift as a functional linear in the field's spectral
energy density, arising from the off-resonant components of the thermal
field acting through the atom's dynamic polarizability. CliffordClock's own
existing E32 formula already reflects this: `Delta_nu_stat*(T/T0)^4 +
Delta_nu_dyn(T)` is built from `Delta_nu_stat = -(Delta_alpha(0)/2h)*<E^2>_T0`
and a dynamic term fit directly to the Planck-weighted overlap integral
(Lisdat et al., Phys. Rev. Research 3, L042036 (2021), the same source
already cited for the registry's dynamic-term shape; Middelmann, Falke,
Lisdat, Sterr, Phys. Rev. Lett. 109, 263004 (2012) for the static term). Both
terms are linear functionals of the spectral energy density at a single
temperature `T`.

The second half of the additivity claim, that the energy density of a
multi-temperature enclosure is the solid-angle-weighted sum of the individual
Planck spectral energy densities, follows from ordinary radiometry: distinct
macroscopic thermal surfaces at different temperatures are mutually
incoherent sources (their mutual coherence time is set by `hbar/kT`, on the
order of femtoseconds at 300 K, far below any interrogation or dephasing
timescale in this project's scope), so their fields superpose in intensity,
not amplitude, with no cross term. Given both halves, `Delta_nu(multi) =
sum_i w_i * Delta_nu(T_i)` exactly, and because each `Delta_nu(T_i)` is
itself the same fixed polynomial in `(T_i/T0)^n`, distributing the sum gives
`sum_i w_i * sum_n c_n*(T_i/T0)^n = sum_n c_n * sum_i w_i*(T_i/T0)^n = sum_n
c_n*M_n`, the exact identity E37 implements. The reviewer verified this
identity symbolically and numerically in A3 below with an independently
chosen case, finding agreement to 50 decimal digits.

Conditions under which additivity would fail: coherence between the sources,
which does not occur for blackbody thermal radiation from separated physical
surfaces at any temperature relevant here; operation beyond second order in
the atom-field coupling, already bounded negligible by E33's own
hyperpolarizability cross-term estimate (about `1e-22`, far below the `1e-19`
floor); a source whose true spectral radiance is not Planckian at its stated
temperature, since the registry's `dyn_coeffs_hz` fits are calibrated against
a Planck spectrum and `RadiationSurface` implicitly assumes every surface is
one; and near-field or evanescent coupling at sub-wavelength atom-surface
separations, which is out of scope for every treatment surveyed in the
dossier and for this implementation, both of which work in the far-field,
solid-angle picture only. Reflections do not break additivity: as PTB's own
paper and Bothwell's exchange-factor formalism both show, a reflected photon
still originated at some surface's own Planck emission, so a reflection only
moves weight between sources, changing which `w_i` multiplies which `T_i`
without introducing any nonlinear or cross term. That reweighting is exactly
where A2 below initially found a defect.

### A2. The PTB emissivity formula

Verdict: PASS, after one fix loop.

The reviewer fetched arXiv:2507.14030 directly (PDF text extraction, not the
abstract page) and located the formula in section 3, equations 2 through 4.
The paper's own text: a two-hole cylindrical shield of length `l` and
interior emissivity `epsilon_in`, atoms at position `z`, geometric solid
angle `Omega(z)/4pi` given by equation 2 as a function of the hole radii and
`z`. Equation 3 gives the reflection-corrected effective solid angle exactly
as

    Omega_eff(z)/4pi = 1 / [1 + (4pi/Omega(z) - 1)*epsilon_in]

which matches the implementation's docstring form character for character.
Equation 4 then gives the actual shift as

    Delta_nu_BBR_shield(z) = Delta_nu_BBR(T_shield)*(1 - Omega_eff(z)/4pi)
                              + Delta_nu_BBR(T_out)*(Omega_eff(z)/4pi)

The paper validates this against a position scan, finding a measured
differential shift of `-3.33(3)e-15` against a model prediction of
`-3.32(7)e-15`, a genuine position-resolved agreement.

The physical situation equation 3 is derived for is a specific two-region
split: one small aperture of raw weight `w = Omega/4pi` carrying the outside
temperature `T_out`, and the shield's own body carrying the complementary
weight `1 - Omega_eff/4pi` and the shield's own temperature `T_shield`.
`epsilon_in` is a single property of the shield's interior as a whole, not a
property attached to the aperture. The two weights `Omega_eff/4pi` and `1 -
Omega_eff/4pi` are complementary by construction and already sum to 1; the
paper performs no renormalization step.

The initial implementation instead attached an optional emissivity to each
surface independently, computed `raw_i = w_i` when no emissivity is given or
`raw_i = w_i/(w_i + (1-w_i)*epsilon_i)` when one is given, and then
renormalized the full set of `raw_i` by dividing every entry by their sum.
The reviewer computed both approaches directly for the exact two-surface case
the implementation's own test uses to validate this formula, an aperture
surface with `w = 0.1` and `epsilon = 0.5` paired with a shield surface at
`w = 0.9` and no emissivity. PTB's own equations, applied to this pair, give
weights `(0.181818..., 0.818182...)`. The renormalize-by-sum approach gave
`(0.168067..., 0.831933...)`, a 7.56 percent relative difference on the
aperture's weight. The reviewer swept the discrepancy across other `(w,
epsilon)` pairs: it shrinks to about 0.01 percent at PTB's own tiny-aperture
parameters (`w = 1.17e-3`, `epsilon = 0.926`), but grows to 0.52 percent at
`w = 0.05, epsilon = 0.9` and to 22.4 percent at `w = 0.3, epsilon = 0.3`.
The larger end of that range sits inside the exact "viewport running warmer
than the chamber" scenario the dossier itself uses to motivate this work
package (a single window covering five to thirty percent of the atom's solid
angle), so this was not a corner case outside the tier's intended use.

Both `docs/CONVENTIONS.md` (around line 566) and the `_bbr_effective_weights`
docstring stated that this renormalized, per-surface approach "generaliz[es]
PTB's two-surface (shield/aperture) formula to an arbitrary surface count"
and that it reduces to PTB's formula for that case. That claim was not correct:
the renormalize-by-sum scheme does not reproduce PTB's own two-surface
weights, as shown above. This went beyond an undocumented extrapolation,
which would itself have needed flagging per this review's own instructions;
it was a specific, checkable claim of exact correspondence to a published
formula that did not hold. None of the tests in `tests/test_bbr_environment.py`
caught this at the time, because the file's own "independent" decimal
reference (`_decimal_bbr_environment_pivot_perturbation`) reimplemented the
identical renormalize-by-sum algorithm instead of checking against PTB's
actual complementary-weight pair, and the one test that exercises
`epsilon = 1` is a degenerate case in which the renormalization is a no-op
and so could not expose the discrepancy.

This was the single highest-risk transcription item in the work package, per
this review's own mandate, and the reviewer's finding was that it needed a
code change, not just a documentation caveat: either the multi-surface
correction needed a derivation that actually reduces to PTB's formula at
`N = 2`, or the current behavior needed to be relabeled plainly as a
distinct, unvalidated extrapolation with its own numeric uncertainty budget,
not as a generalization of the cited paper. The builder took the first path:
rebuilding the topology instead of just changing the label.

`_bbr_effective_weights` in `src/cliffordclock/integrator/omega.py` is now
built around a single reflective enclosure plus one or more apertures,
matching PTB's own derivation directly. `_bbr_validate_environment` rejects
any environment with more than one `emissivity`-carrying surface, both in
`omega.py` (`ValueError`, "at most one surface may carry an emissivity") and,
independently, at parse time in `src/cliffordclock/pipeline.py`'s
`_parse_radiation_environment` (`PipelineConfigError`, the matching
message); the reviewer confirmed the pipeline-layer rejection directly by
calling `_parse_environment` with two `emissivity`-carrying surfaces and
catching the raised `PipelineConfigError`. The function now computes the
apertures' combined raw weight `W`, corrects each aperture's own weight by
`w_i/(W + (1-W)*epsilon)`, and gives the enclosure whatever is left,
`1 - sum` of the apertures' corrected weights, instead of renormalizing
every surface's weight against a shared total.

The reviewer recomputed the fix independently, in a fresh script calling
nothing from the implementation, for two cases:

- The project's own two-surface case (`w = 0.1` aperture, `epsilon = 0.5`
  enclosure at `w = 0.9`): PTB's closed form gives `w_aperture_eff =
  0.18181818181818181818...` and `w_enclosure_eff =
  0.81818181818181818181...`, summing to exactly `1`. Calling
  `bbr_environment_pivot_perturbation` on the matching `RadiationSurface`
  pair gives `-1.0176625717213564e-15`; the independent decimal computation
  of the full shift from those same weights gives
  `-1.0176625717213562e-15`, an absolute difference of `2.0e-31`.
- A three-surface case chosen fresh for this re-verification, not reused
  from any test: one enclosure at `w = 0.5`, `epsilon = 0.7`, `T = 150` K,
  and two apertures at `w = 0.35`, `T = 310` K and `w = 0.15`, `T = 340` K.
  The combined aperture weight is `W = 0.5`, giving `denominator = 0.5 +
  0.5*0.7 = 0.85`, aperture weights `0.35/0.85 = 0.41176470588...` and
  `0.15/0.85 = 0.17647058823...`, and an enclosure weight of
  `1 - (0.41176... + 0.17647...) = 0.41176470588...`, all summing to `1`
  exactly; the two apertures' effective weights preserve their raw `7:3`
  ratio exactly, confirming the proportional-split property the topology
  relies on. `bbr_environment_pivot_perturbation` on this configuration
  gives `-4.227008375180008e-15` against an independent decimal
  computation of `-4.2270083751800065e-15`, an absolute difference of
  `1.6e-30`, and `T_eff,4` agrees to `282.50434529878027` K in both.

Both cases confirm the code's Eq. 3-4 reduction to within float64 noise
across two independently constructed configurations, one of them matching
neither the project's tests nor the reviewer's own two-surface
re-verification. `docs/CONVENTIONS.md`'s `*Emissivity correction*`
paragraph (now titled "one enclosure, one or more apertures") and the
`_bbr_effective_weights` docstring were both rewritten to describe this
topology directly; the reviewer confirmed the earlier claim of an
"arbitrary surface count" renormalized generalization no longer appears
anywhere in the diff, and the new text states plainly that the enclosure
"never" gets a value computed from its own raw `weight`, the exact property
that failed before.

The reviewer also checked that the new tests guarding this fix are
genuinely independent, not circular.
`tests/test_bbr_environment.py::test_ptb_two_surface_enclosure_aperture_matches_published_closed_form`
and its three-surface counterpart compute their reference values with
`decimal.Decimal` arithmetic coded directly from PTB's published formula
(`w_eff = w / (w + (1-w)*epsilon)`, the combined-aperture generalization
written out by hand), calling neither `_bbr_effective_weights` nor the file's
own `_decimal_bbr_environment_pivot_perturbation` helper; the test file's own
comment block states this directly and names the reason, that either
shortcut would risk baking the same bug into the implementation and its
check at once. The reviewer confirmed this by inspection: the reference
computation in both tests is self-contained, with the weight formula,
temperature powers, and coefficient sum all written inline in the test body.
As a kill test, the reviewer reintroduced the old renormalize-by-sum scheme
directly in `_bbr_effective_weights`, in a scratch copy of the package, and
reran both new PTB tests against it. Both failed: the two-surface case
returned `-5.636994e-16` against an expected `-1.017663e-15` (44.6 percent
relative error), and the three-surface case returned `-1.651751e-15` against
an expected `-2.480907e-15` (33.4 percent relative error), both far beyond
the tests' `rtol=1e-9`.

### A3. Independent moment-algebra recomputation

Verdict: PASS.

The reviewer picked a two-surface case not used anywhere in the test suite,
`f = 0.3` at `T1 = 265 K` and `f = 0.7` at `T2 = 315 K` for Sr-87, and
computed the moments and shift with a fully independent `decimal` script
(50-digit precision, no call into any function under test):

    M_4  = 1.0335043981481480874094184302038177430617513801948
    M_6  = 1.0805847026105966430325122209239594417231248885544
    M_8  = 1.1454222471724071947900481069471428726700510433028
    M_10 = 1.2269958090733362455708585389953760018882838149535

with `sum_i w_i * sum_n eta_n*(T_i/T0)^n` and `sum_n eta_n*M_n` agreeing to
`0E-50` (an exact algebraic identity, as expected from linearity), giving
`(P-1)_BBR = -5.5193041854742226e-15`. Calling the real
`bbr_environment_pivot_perturbation` on the same surfaces gives
`-5.519304185474224e-15`, an absolute difference of `1.58e-30`, far below any
float64 noise floor. The per-moment effective temperatures from the
implementation, `T_eff,4 = 302.48186146846`, `T_eff,6 = 303.90024992460`,
`T_eff,8 = 305.13495148588`, `T_eff,10 = 306.20026530591`, match the same
manual decimal computation to 13 significant figures. The static term
correctly consumes `M_4` and the dynamic terms correctly consume `M_6`,
`M_8`, `M_10` against the Sr-87 registry's `{6: -0.13216, 8: -0.01231, 10:
-0.00858}` Hz coefficients; the reviewer confirmed a deliberately introduced
4/6 exponent swap (B2 below) reproduces the test file's own pinned "wrong"
value exactly, which independently confirms the correct implementation is
not accidentally computing the same thing. At `T = T0` for a uniform
environment, every moment reduces to exactly `1` in decimal arithmetic, the
expected exact reduction.

### A4. Per-surface validity window

Verdict: PASS, accept as written.

Enforcing the 50 to 350 K fit window on every individual surface temperature
is not just conservative, it is the provably correct and sufficient check
given A1 and A3's exactness. `M_n` is a weighted sum of `(T_i/T0)^n` with
non-negative weights summing to 1, a convex combination; the per-moment
effective temperature `T_eff,n = T0*M_n^(1/n)` is therefore a weighted power
mean of the surface temperatures and, by the power-mean inequality, is
bounded between `min(T_i)` and `max(T_i)`. If every surface temperature lies
in `[50, 350]` K, `T_eff,n` cannot leave that range for any `n`, so there is
no regime in which every surface passes the per-surface check while the
composite result draws on temperatures the fit was never validated against.
The narrower `cross_verified_max_k` sub-band (300 K for Sr-87) is a separate,
already-correctly-handled concern: the pipeline's report note flags any
surface above that threshold by name, mirroring the single-temperature path.

### A5. Bylines

Verdict: PASS, with the caveat recorded in B5 item 4 below.

Nosske, Vishwakarma, Lucke, Rahm, Poudel, Weyers, Benkler, Dorscher, Lisdat,
arXiv:2507.14030, was fetched directly and its author list matches every
citation of it added by this diff. The Aeppli 2025 JILA thesis reference
matches the entry already carried in the project's BBR thermal-environment
dossier. Middelmann et al., Phys. Rev. Lett. 109, 263004 (2012), Lisdat et
al., Phys. Rev. Research 3, L042036 (2021), and the rest of
`_BBR_SPECIES_CITATIONS` are unchanged, pre-existing, already-verified
strings that this diff reuses unedited.

## Part A verdict: PASS, approve.

One fix loop was needed, in A2 (the PTB emissivity formula): the initial
renormalize-by-sum scheme did not reduce to PTB's own two-surface weights,
the builder rebuilt the weighting around PTB's actual enclosure-plus-aperture
topology, and the rebuilt version now reproduces PTB's closed form exactly on
both the original two-surface case and a fresh three-surface case. A1, A3,
A4, and A5 passed on the first review with no changes needed.

## Part B: code review

### B1. Bitwise-reduction claim

Verdict: PASS.

Structurally, both `bbr_pivot_perturbation` and
`bbr_environment_pivot_perturbation` route through the same
`_bbr_weighted_moments`/`_bbr_effective_weights` pair, and
`bbr_pivot_perturbation` is now literally implemented as a one-surface call
into the multi-surface function. Empirically, all 8 parametrized reduction
tests in `tests/test_bbr_environment.py` pass with `rtol=0, atol=0`. The
reviewer also ran the full existing test suite; three tests fail, all
confirmed (by running the identical tests against the pre-branch commit) to
be pre-existing failures unrelated to this diff: a float-noise mismatch in
two unrelated shipped-example regression snapshots and a duplicate
bibliography key in `tools/release_checks.py`'s own test suite (closed later
in the Overall verdict section below, as a bonus fix alongside this diff's
own two fix-loop items). No new failures anywhere in the project are
introduced by this branch.

### B2. Kill-test quality

Verdict: PASS.

The reviewer copied the package into a scratch directory and introduced each
targeted bug directly, one at a time. Skipping the renormalization step in
`_bbr_effective_weights` (returning the raw, unnormalized list) makes
`test_ptb_emissivity_form_matches_hand_computed_weights` fail with `got =
-1.022711e-15`, exactly the test's own pinned `wrong_unnormalized_value`.
Swapping `M_4` and `M_6` in `bbr_environment_pivot_perturbation`'s static and
leading-dynamic terms makes `test_moment_exponent_swap_kill_test` fail with
`got = -5.904827e-15`, exactly the test's own pinned `wrong_value`. Both
kill-tests genuinely discriminate the bug they name.

### B3. Tolerance discipline, mypy, JAX compatibility, error-path callers

Verdict: PASS on all four sub-items.

`tools/release_checks.py --only tolerance-scan` reports zero findings; every
new `np.testing.assert_allclose` call carries an explicit `atol` or `rtol`.
`.venv/bin/python -m mypy --strict` on both changed source files reports no
issues. `_resolve_bbr_environment_pivot_perturbation` is called from exactly
one site, in `run_pipeline_full`, as plain Python before the `coupling.type`
branch and before any downstream evaluation; there is no `jax.jit` anywhere
in `pipeline.py`, so this resolution happens entirely at parse/build time and
the resulting scalar is closed over as a Python float, never traced. On the
widened validity check: every caller of `bbr_pivot_perturbation` or
`bbr_pivot_uncertainty` outside the test suite was checked, `paper/figures/fig5_bbr_temperature.py`
(a grid from 50 to 350 K inclusive, plus a dataset already filtered to that
window) and `benchmarks/run_bbr_jila_arithmetic_reproduction.py` (JILA's
293.282 plus or minus 0.004 K), and none use a temperature outside the
window, so none would have relied on the old non-raising behavior.

### B4. Style and prose

Verdict: PASS, after one fix loop.

`.venv/bin/python tools/release_checks.py --only prose-scan` initially
reported a `FAIL`-severity finding introduced by this diff:
`docs/CONVENTIONS.md:616`, the phrase "not merely" inside the new E37 "Exact
reduction to E32" paragraph ("agree bit for bit ... not merely to a
numerical tolerance"). The reviewer confirmed by checking out the
pre-branch commit that this specific finding was absent on main and new to
this branch; the allowlist carries one pre-existing, deliberately preserved
"not merely comparable" line for the G9 gravity text, but no equivalent
entry existed for this new line, so the automated release gate would have
failed on this diff as written. No other fatal-listed phrase ("stated
plainly," "it is worth noting," "honestly labeled") or honest-family word
appeared anywhere in the diff's added markdown, docstrings, or tests. The
same "not merely" phrase also appeared several more times in the diff's
added Python docstrings inside `omega.py`; `release_checks.py`'s prose-scan
only covers markdown, notebooks, and the paper's LaTeX, so these did not
trip the automated gate, but they carried the same banned phrase and were
flagged for editing alongside the primary instance for consistency.

`.venv/bin/python tools/release_checks.py --only prose-scan` now reports
zero `FAIL`-severity findings (48 `MINOR` "rather than" findings remain,
unrelated to this diff and already present before it). The specific
`docs/CONVENTIONS.md:616` finding from the initial pass is gone; the
paragraph it was in was rewritten as part of the A2 fix. A repeat grep for
the other fatal-listed phrases ("stated plainly," "it is worth noting,"
honest-family words) across the full diff still finds none. The handful of
pre-existing "not merely" occurrences elsewhere in `omega.py` flagged above
as a non-blocking cleanup item are untouched and are not part of the new E37
text, so they remain a cosmetic item, not a gate issue.

### B5. The five ambiguity flags

1. Validity-window widening: accept. Confirmed in B3 that no existing caller
   depended on the old non-raising behavior, and the change is a strict
   fail-closed improvement consistent with `_bbr_validate_environment`
   already being called from every public entry point for the same reason.
2. `RadiationSurface.weight` naming: accept, with a stylistic note. The field
   name is generic, but the docstring immediately disambiguates it as
   `Omega_i/(4*pi)`; a more specific name such as `solid_angle_fraction`
   would reduce reliance on the docstring but is not required.
3. `docs/cli.md` not updated: resolved. The reviewer's original pass grepped
   the entire `docs/` tree and found `radiation_environment` only in
   `docs/CONVENTIONS.md`; `docs/cli.md` documented `radiation_temperature_K`
   at length in a dedicated section with no mention of the new config key,
   so a user reading the project's own CLI reference would not have learned
   this feature exists. `docs/cli.md` now carries a dedicated "Multi-surface
   thermal environment" section documenting
   `environment.radiation_environment:`; the reviewer confirmed the section
   exists and names the config key directly.
4. No separate sign-off ceremony line: accept, resolved by this record. This
   review was commissioned as the stand-in gate for E37 specifically because
   it carries no spacetime-algebra content; producing this file closes that
   gap. `docs/CONVENTIONS.md`'s version header should be updated to cite this
   gate once it is committed, a small follow-up item, not a gate requirement.
5. `math.fsum` usage: accept. The correct choice for the weight-sum and
   moment-sum accumulations, consistent with the project's existing
   precision-discipline conventions; no issue found.

### B6. Report note quality

Verdict: PASS.

The reviewer ran a live pipeline configuration (Sr-87, `stark_dc`, two
surfaces: `viewport` at weight 0.1, 320 K, emissivity 0.9, temperature
uncertainty 0.05 K; `chamber_wall` at weight 0.9, 300 K, temperature
uncertainty 0.01 K; `correlated: true`) and read
`result.report.uncertainty_notes` directly. The note lists both surfaces by
name with their weight, temperature, uncertainty, and emissivity; the
resolved `(P-1)_BBR` value; all four per-moment effective temperatures
`T_eff,4` through `T_eff,10`; the active uncertainty combination mode; the
combined numeric uncertainty with an explicit statement of what it includes;
the M1/E2 budget line; and a correctly triggered warning that the `viewport`
surface exceeds the 300 K cross-verified band. A reader can reconstruct the
input environment from the note text alone, without access to the run's
configuration file.

## Part B verdict: PASS, approve.

One fix loop was needed, in B4 (style and prose): the initial diff tripped
the release gate's prose-scan with an uncovered "not merely" phrase, the
builder reworded the affected paragraph, and the prose-scan now reports zero
`FAIL`-severity findings. B5 item 3 (`docs/cli.md` coverage) was also closed
alongside it. B1, B2, B3, B5's remaining items, and B6 passed on the first
review with no changes needed.

## Overall verdict

Verdict: APPROVE E37 for WP29 Tier 1.

`tools/bibliography.toml` carried two entries under the key `Lodewyck2012`
before this round, the duplicate the first review pass flagged in B1 above
as a pre-existing, unrelated test failure
(`test_load_bibliography_keys_are_unique`). The duplicate has been removed;
one entry remains, keeping the first entry's `doi` field and its title's
original capitalization, with its `source` field extended to carry both
entries' provenance text: the original `"paper/refs.bib; README.md"` note
plus the deleted entry's `"README.md quickstart (the documented
stray-charge event examples/realistic_lattice_sr87.yaml brackets)"` note
and its Crossref byline-verification sentence. No provenance information
from either original entry was dropped. `.venv/bin/python -m pytest
tests/test_release_checks.py::test_load_bibliography_keys_are_unique`
passes.

`.venv/bin/python -m mypy --strict` on both changed source files: no
issues. `.venv/bin/python -m ruff check` on the three changed/added Python
files: all checks passed. `tools/release_checks.py --only tolerance-scan`:
zero findings. `.venv/bin/python -m pytest tests/test_bbr_environment.py`:
all 38 tests pass. A full `.venv/bin/python -m pytest -q` run over the whole
project shows exactly two failures, both the same pre-existing,
BBR-unrelated float-noise snapshot mismatch already identified in the
original review (`showcase_gradient_dispersion_sr87.yaml`, `1.23e-32`
absolute / `1.24e-16` relative, present on the pre-branch commit); the third
pre-existing failure the original review found,
`test_load_bibliography_keys_are_unique`, is gone, consistent with the
bibliography deduplication above. No failure anywhere in the project is new
to this round of changes.

Both fix-loop items from the original review, A2 and B4, are resolved and
independently re-verified against fresh, self-chosen numbers, not against
the diff's own tests. The one needs-change item, B5 item 3
(`docs/cli.md`), is also closed. The remaining notes are cosmetic and
non-blocking: the pre-existing "not merely" occurrences elsewhere in
`omega.py`, unrelated to E37's own text; `RadiationSurface.weight` as a
generic but adequately documented name; and `docs/CONVENTIONS.md`'s version
header, which should be updated to cite this record by name once it is
committed alongside the work package. None of these affect the approval.
