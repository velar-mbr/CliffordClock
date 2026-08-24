# Dataset benchmark parameter mapping

**Summary, for the impatient:** this document is the row-by-row (JILA
Table I) and file-by-file (NIST, NPL, USTC) reasoning behind
`benchmarks/RESULTS.md`'s headline numbers: for every published line
item, whether it maps to a `cliffordclock` pipeline config, and if not,
exactly why not. A documented "cannot map" (e.g. blackbody radiation,
outside this engine's scope; a paper reporting a shift but not the field
that produced it) is reported as a normal, acceptable outcome throughout,
with the same visibility given to every row regardless of outcome. See
`benchmarks/RESULTS.md`'s executive summary for the headline result and
`benchmarks/SOURCES.md` for citations, checksums, and access logs.

Every assumption below is reviewable and traceable to a specific source
citation. This document explains, row by row / file by file, whether a
published parameter maps to a `cliffordclock` pipeline config, and if not,
exactly why not; a documented "cannot map" is a normal, acceptable
outcome, stated here at face value.

## Engine physics scope (recap, for the classification below)

Per `docs/CONVENTIONS.md`/`docs/validation.md`, this engine currently
implements exactly two systematic-shift mechanisms:

- **E14b**: scalar DC-Stark shift, `Δν/ν0 = -(Δα/2)|E(r)|²/(hν0)`, for
  J=0→J=0 lattice clocks (`Sr87`, `Yb171` in the species registry).
- **E21**: second-order (kinematic/time-dilation) Doppler shift,
  `Δν/ν0 = -⟨v²⟩/(2c²)`.

At the time of this WP10 pass, nothing else (BBR, AC/lattice light shift,
Zeeman/magnetic-field physics, collisional/density shifts,
tunneling/band-structure physics, tensor/quadrupole polarizability for
ion clocks) existed anywhere in `src/cliffordclock`. This reflects the
documented, intentional scope of the
engine at that point (`docs/CONVENTIONS.md` E14b/E29 scope notes). The
WP20/WP21/WP22 addenda below cover the BBR, ion-quadrupole, and
gravitational-redshift physics this project has added since.

## Source 1: arXiv:2403.10664 Table I

| Row | In scope? | Mapping outcome |
|---|---|---|
| BBR | No (WP10-scope classification, see WP20 addendum below) | At the time of this WP10 pass, no temperature-dependent static+dynamic differential-polarizability model existed in this engine. **WP20 update (2026-08-11):** this physics now exists (CONVENTIONS.md E32/E33) and this exact row now has a dedicated, separately-labeled **arithmetic-reproduction** case; see "WP20 addendum" below. This row's own `in_engine_scope=false`/`comparable=false` classification in `benchmarks/fixtures/jila_2403_10664_table1.csv` is left unchanged (a WP10-era artifact of the "forward-mapped from this table's own row via a `PipelineConfig`" protocol those booleans describe, which the WP20 case does not use), so as not to alter previously-reviewed WP10 numbers/tests as a side effect of an unrelated WP20 addition. |
| Lattice Light | No | AC (trap-light-frequency-dependent) polarizability at the magic wavelength. `docs/CONVENTIONS.md` E29 explicitly scopes the lattice fast path to DC Stark only. **No mapping attempted.** |
| Second Order Zeeman | No | Needs a magnetic-field Hamiltonian term (`ξσ` coefficient, `CONVENTIONS.md` has no B-field physics at all). **No mapping attempted.** |
| Density | No | p-wave collisional shift between fermionic atoms; needs inter-atom interaction physics. This engine's ensembles are independent, non-interacting particles by construction. **No mapping attempted.** |
| First order Zeeman | No | Same as Second Order Zeeman: no B-field physics. **No mapping attempted.** |
| Background Gas | No | Collisional shift from residual background-gas atoms; no collisional physics of any kind exists. **No mapping attempted.** |
| **DC Stark** | **Yes (E14b)** | **See "DC Stark row" below: in scope, but no independent forward mapping is possible from what the paper publishes.** |
| Tunneling | No | Inter-lattice-site band-structure/Wannier-Stark tunneling. This engine models motional states as harmonic-oscillator Hermite-Gauss quadrature nodes (`cliffordclock.ensemble.lattice`), not a lattice band structure, so there is no tunneling degree of freedom to map onto. **No mapping attempted.** |
| Minor Shifts | No | An unitemized grab-bag ("all other systematic effects have uncertainties below 1e-19" per the abstract); no single mechanism to map. **No mapping attempted.** |
| Total Shift | n/a | An arithmetic sum of the rows above. |

### DC Stark row: why no forward config can be built

The paper's *only* quantitative statement about this systematic is (main
text, unlabeled "DC Stark Shift" section, page 4-5, quoted verbatim in
`benchmarks/loaders.py`'s `JILA_DC_STARK_PRECISE`): *"The total residual DC
Stark shift is −9.8 ± 0.7 × 10⁻²⁰."*, a **fractional-frequency shift**,
determined by *directly measuring the clock frequency* while alternating
quadrant-electrode fields high/low (a null/lock-in measurement). Recovering
a field magnitude from this shift would require a separately-known field
value and a polarizability formula, and this measurement method produces
neither. The paper explicitly does not report:

- the residual stray-field magnitude in V/m (or V/cm/kV/cm) at the atoms;
- which `Δα` value (if any) they used, since their method needs none:
  they measure the shift directly;
- the quadrant-electrode geometry/voltage values that would let a reader
  back out a field independently.

(Confirmed by grepping the full extracted paper text for `V/m`, `V/cm`,
`kV/cm`, `electric field`, and `polarizab`: every hit is either this
prose paragraph itself, or an unrelated dipole-matrix-element/AC-
polarizability discussion in the supplementary fitting section, sections 6-9
pages 12+, which concerns the BBR dynamic-correction fit, a systematic
distinct from DC Stark.)

**Third source investigated, 2026-08-10 (authorized for this benchmark follow-up;
supersedes the earlier "uninvestigated" review note):** the paper's
published version (PRL 133, 023401) references Supplemental Material
hosted at journals.aps.org. This was fetched-and-examined per the owner's
follow-up authorization, with one qualification: the file itself could
not be retrieved. `journals.aps.org` gates it "Subscription Required,"
and the one public route, APS's CHORUS accepted-manuscript link, could
not be downloaded through any available tool (see
`benchmarks/SOURCES.md` section 3 for the full access-attempt log). No
credential entry or paywall bypass was attempted, per this session's
binding safety rules: the owner's authorization to *fetch* a source
does not authorize bypassing a subscription gate.

In place of the file itself, `benchmarks/SOURCES.md` section 3 documents
a direct, verifiable cross-check using the arXiv v2 e-print already
fetched for source 1: the e-print's own LaTeX source bundle contains
separate `main.tex`/`supplementary.tex` files compiled into one PDF, and
every one of the main text's four citations to "[23] See Supplemental
Material" (temperature-sensor uncertainty, 3D1-lifetime datasets, the BBR
`νdyn` value, background-gas composition) corresponds exactly to one of
the compiled PDF's five supplement sections (I: 3D1/3P1 lifetime
uncertainty budget, II: Temperature Measurement, III: Dynamic BBR Shift,
IV: First Order Zeeman Shift, V: Background Gas Shift, confirmed
complete, ending in the supplement's own separate bibliography). **The DC
Stark paragraph cites no such reference**: its only citation, `[29]`, is
a generic prior-literature citation for the general stray-field effect,
not this paper's own supplemental data, and is fully self-contained,
stating only the resulting shift.

**Precise conclusion:** the Supplemental Material almost certainly
contains no DC-Stark field-magnitude, quadrant-electrode-voltage, or
`Δα` data: its five sections are demonstrably about five *other*
systematics (lifetime/BBR, temperature, first-order Zeeman, background
gas), and the paper's own citation pattern never points a reader to it
for DC Stark. This is strong, specific, checkable evidence (not an
absence-of-evidence guess): every one of the sections the Supplemental
Material *is* shown to contain is independently confirmed from the
already-fetched, checksummed arXiv v2 text. **This
benchmark's outcome for the DC Stark row is unchanged** by this follow-up: still no
independent forward-comparable case, for the same fundamental reason
documented below (no independent field input published anywhere the
project has legitimate access to, including this third source).

**Consequence:** a `PipelineConfig` needs a field magnitude
(`field.synthetic.params.e0`) as its independent input. There is no
published, independent number to put there. The two ways around this are
both illegitimate:

1. **Guess a field, report a "residual."** Any guessed field violates
   the benchmark protocol's rule that every input traces to the
   publication or to MAPPING.md; a guess traces to neither. The
   resulting "residual" would then compare an invented number against a
   real one, a meaningless comparison.
2. **Solve for the field that reproduces −9.8×10⁻²⁰ exactly**, then report
   that as a "predicted vs. published, residual ≈ 0, PASS" case. This is
   *exactly* the forbidden move per the benchmark protocol: "No parameter
   may be fitted, tuned, or selected to reduce residuals." Solving for the one
   free input that makes the output match the target *is* fitting a
   parameter to the residual, full stop, even though the underlying
   formula (E14b) is itself already independently validated (KA1/KA2,
   `tests/test_known_answers.py`, `rtol=1e-10` against the textbook
   formula with an *independently chosen* field).

`benchmarks/run_benchmarks.py` therefore does **not** produce a residual
for this row. What it *does* do, clearly separated and labeled as
non-comparison context: an illustrative sweep of the real
`coupling.type: stark_dc` pipeline over fixed, round-number field
magnitudes (1/5/10/20/50/100 V/m, chosen before any comparison existed),
so a reader can see where JILA's number sits on the same physical axis
(the implied field for their number, purely for narrative context in
`benchmarks/RESULTS.md`, is `sqrt(9.8e-20 * ν0 / |k_S|) ≈ 3.7 V/m` using
this engine's own Sr87 `Δα`, explicitly *not* reported as a validated
result, since it is definitionally circular: solving the very formula
being "checked" for its own free parameter).

## Source 2: data.nist.gov DOI 10.18434/M32206

**No mapping is possible for either file, for a different reason than
Source 1: this is not a systematic-shift measurement at all.**

The dataset is two phase-vs-time series (`Yb_Clock_phase(rad) vs time.csv`,
`10GHz_phase(mrad) vs time.csv`) used, per the record's own description, to
"calculate relative phase fluctuation and Allan deviation" for an
optical-to-microwave frequency-division scheme: it characterizes
the **short-term instability** of a specific down-conversion apparatus
(how much the down-converted microwave's phase wanders relative to the
optical clock over time). This engine's benchmark comparisons need a
**systematic frequency shift** caused by a field, temperature, or
motional configuration; the dataset measures a different quantity
entirely.

There is no field-gradient, trap, temperature, species, or motional
parameter anywhere in this dataset to populate a `PipelineConfig` with.
Even setting that aside, this engine's analytics
(`cliffordclock.analytics`) compute a mean fractional shift, its standard
error, and T2* dephasing (`docs/report-schema.md`); there is no
Allan-deviation computation anywhere in the package, and adding one would
be new statistics functionality, deliberately out of scope for a dataset
comparison pass (no new physics or analytics modules). So even a
hypothetical "compute Allan deviation from the phase series and compare
its magnitude to something" is out of reach without exceeding scope
twice over.

**Conclusion:** this dataset is ingested (parsed, checksummed, documented)
per this benchmark's instruction to ingest every authorized source, and
`benchmarks/run_benchmarks.py` classifies both files and reports, in
`benchmarks/RESULTS.md`, that they produce no comparison of any kind.

## Source 4: arXiv:1706.01944, the NPL reproducibility case (Task A, follow-up 2026-08-10)

**This is the one source, of everything examined across all four of this
benchmark's passes, that publishes an independent field magnitude for the
DC-Stark systematic**; see `benchmarks/SOURCES.md` section 4 for the
full verbatim extraction and independent verification. This changes this
benchmark's outcome: for the first time, a genuine `PipelineConfig` can
be built with every input traced to a publication.

### The two ingredients, and why this is NOT a blind prediction

NPL measured a residual stray field at their Sr atoms **independently of
the clock transition** (Rydberg-state EIT spectroscopy on a completely
different atomic state): `E = 1.52 (+0.62/-0.22 stat, +0.05/-0.03 sys)
V/m`. They then convert this field to a clock-transition DC-Stark shift
using the Middelmann et al. (PRL 109, 263004 (2012)) differential
polarizability, the exact paper this project's own `Sr87` species
registry entry (`cliffordclock.ensemble.species.SR87`) cites for `Δα`
(confirmed: `benchmarks/SOURCES.md` section 4's independent cross-check,
reproducing the paper's own "570 V/m → 1 Hz" intro example to 5
significant figures using nothing but this project's registry value).

**This means: NPL already combined the same two published ingredients
this project's pipeline would combine** (their own field + the
literature `Δα`) to arrive at their quoted `-1.6 (+0.4/-1.6) × 10⁻²⁰`.
Running this engine's `coupling.type: stark_dc` on their field with the
registry's `Δα` therefore **reconstructs a number NPL already computed
themselves**, using the same formula, the same field, and (functionally)
the same polarizability source, so it does not test whether this engine
can predict a clock shift *nobody had already computed*.

**Binding classification label (per the benchmark protocol,
verbatim): this is a "zero-free-parameter reproducibility case."** It
must never be described as "validated against an independent measurement
of the shift": the shift itself is not independent of the inputs this
engine also uses; only the *field measurement's apparatus* (Rydberg EIT
spectroscopy on an atom orders of magnitude more polarizable than the
clock state, via microwave/optical spectroscopy of a completely different
transition) is independent of the clock-transition Stark-shift physics.
What a "MET" verdict on this case legitimately demonstrates: **end-to-end
pipeline correctness against two independently published inputs/outputs,
with no fitting anywhere in the chain**, a real, non-trivial claim
(a unit conversion error, a sign error in E14b, a factor-of-2 error in
the pivot formula, or a wrong `Δα`-to-`k_S` derivation would all show up
as a band that does NOT overlap NPL's), but a categorically different
claim from "this engine independently predicted an unknown clock shift."
`benchmarks/run_benchmarks.py`'s `NplReproducibilityCase.case_class` is
literally the string `"reproducibility"`, kept structurally separate from
a (still entirely empty) `"blind_prediction"` category.

**Non-circularity in provenance (why this case has value at all,
despite the label above):** the field measurement (NPL's Rydberg
apparatus) and the polarizability (PTB's Middelmann et al. measurement)
come from two different groups, two different apparatuses, two different
papers, neither of which is this project's own output. Neither is
fitted or adjusted here. The case is legitimate evidence of pipeline
correctness; it is just not evidence of *predictive* power beyond what
the underlying, already-validated E14b formula (`tests/test_known_answers.py`
KA1/KA2) already established.

### Method: propagating the field's asymmetric uncertainty (no Gaussian pretense)

NPL's field has independent statistical and systematic uncertainties,
each itself asymmetric (`+0.62(stat)/-0.22(stat)`, `+0.05(sys)/-0.03(sys)`).
Per the benchmark protocol ("no Gaussian pretence on
asymmetric errors"), `benchmarks/loaders.py`'s `AsymmetricMeasurement`
combines statistical and systematic contributions **in quadrature, on
each side separately** (`combined_hi = nominal + sqrt(stat_hi² +
sys_hi²)`, `combined_lo = nominal - sqrt(stat_lo² + sys_lo²)`); this is
a standard, defensible combination step (independent uncertainty
sources add in quadrature), applied without ever symmetrizing the two
sides against each other or assuming a normal distribution for the
overall asymmetric result. Rigor note (added at review): per-side
quadrature is a common *approximation* for asymmetric errors, not the
only defensible treatment (e.g. Barlow-style variable-Gaussian
combination handles asymmetric-uncertainty shapes more carefully); the
MET verdict here is robust to that choice, since NPL's band sits deep
inside the predicted band. This yields:

```
E_lo     = 1.52 - sqrt(0.22^2 + 0.03^2) = 1.298... V/m
E_nominal = 1.52 V/m
E_hi     = 1.52 + sqrt(0.62^2 + 0.05^2) = 2.142... V/m
```

`benchmarks/run_benchmarks.run_npl_reproducibility_case` then runs the
**real pipeline three separate times** (`species: Sr87`, `coupling.type:
stark_dc`, lattice fast path, `n_quad=1` uniform field, 1 s
interrogation, the same machinery KA1 validates) at `E_lo`, `E_nominal`,
and `E_hi`, each a full pipeline execution. Because E14b's shift is
`-(k_S)|E|²/ν0` (monotonically more negative with `|E|`, `k_S < 0` for
Sr87), `E_hi` maps to the most-negative predicted shift and `E_lo` to the
least-negative; the resulting predicted band's asymmetry is *inherited*
through the pipeline from the field's asymmetry, a direct consequence of
running the same formula at three different field values. The pipeline
call's own monotonicity is checked programmatically at runtime
(`predicted_shift_lo <= predicted_shift_nominal <=
predicted_shift_hi`).

### Comparison and result

| Quantity | Low | Nominal | High |
|---|---|---|---|
| Field (V/m) | 1.298 | 1.52 | 2.142 |
| Predicted Δν/ν₀ (this engine) | -3.290×10⁻²⁰ | -1.657×10⁻²⁰ | -1.208×10⁻²⁰ |
| Published Δν/ν₀ (NPL) | -3.2×10⁻²⁰ | -1.6×10⁻²⁰ | -1.2×10⁻²⁰ |

**Overlap test** (`benchmarks/run_benchmarks._bands_overlap`, precisely
defined as: two closed intervals `[lo1, hi1]`/`[lo2, hi2]` overlap iff
`lo1 <= hi2 and lo2 <= hi1`, documented and unit-tested,
`tests/test_benchmarks_loaders.py::test_bands_overlap_precise_definition`):
the predicted band `[-3.290, -1.208]×10⁻²⁰` and NPL's published band
`[-3.2, -1.2]×10⁻²⁰` **overlap** (in fact NPL's band is nearly contained
within this engine's slightly wider one, and each band's nominal value
falls inside the other's band). **`kpi_verdict = "MET"`.**

### KPI-eligibility classification

Distinct from every "not applicable" budget-only row (Sources 1, 2, 5):
this is the project's first `case_class = "reproducibility"` row, with
its own MET/NOT-MET verdict (never PASS/FAIL, reserved vocabulary this
project does not use for any case in this benchmark). It is **not**
counted toward `not_applicable_rows`, and it is **not** a
`blind_prediction` case (that category remains empty). See
`benchmarks/RESULTS.md` for how this reclassifies this benchmark's
headline.

## Source 5: Metrologia 63, 025002 (2026), USTC Sr1 DC-Stark constraint (Task B, follow-up 2026-08-10)

**Same structural class as the JILA arXiv:2403.10664 DC-Stark row
(Source 1): in engine scope, but no independent field magnitude
published in this paper.** Full verbatim extraction, own independent
read of the owner-provided PDF, verified directly against the source:
`benchmarks/SOURCES.md` section 5.

Section 3.5 "Other minor systematic shifts" ("Residual DC Stark shift"
subsection, printed page 9) and Table 3 (printed page 10) constrain the
**total** DC-Stark shift to `0.0(0.1) × 10⁻¹⁹`, but the derivation is
two steps removed from a field measurement:

1. A **prior** shift measurement, itself derived from a field measured
   elsewhere: "the y-component of the field caused a shift of
   1.4(5.2) × 10⁻²¹", cited to their own reference **[30], Li J et al
   2024 Metrologia 61 015006**, a different paper from this one.
2. **Geometric/shielding-factor scaling arguments** applied to that prior
   shift, a scaling argument applied to an existing measurement, not a
   fresh derivation from a field magnitude: an 8× geometric
   factor from the ratio of viewport distances (142 mm vs 237 mm,
   `E_stat ∝ r⁻²`), and a 3× shielding factor "according to FE
   simulations" for the other two axes.

**No field magnitude in V/m appears anywhere in this paper's DC-Stark
discussion.** Building a `PipelineConfig` would require either the same
two illegitimate moves already ruled out for Source 1 (guess a field, or
solve for one that reproduces `0.0(0.1)×10⁻¹⁹`, doubly meaningless here
since the published value is consistent with zero, so "solving for a
field" would trivially suggest `E ≈ 0`, an artifact of the published
value's consistency with zero). **No mapping attempted; classified `comparable=False`,
`kpi_verdict="N/A"`,** same shape as every JILA row
(`run_benchmarks.classify_ustc_dc_stark`).

### Reference [30] follow-up: authorized, attempted, ACCESS BLOCKED (third follow-up, 2026-08-10)

**Reference [30], Li J et al 2024, Metrologia 61, 015006**, per this
paper's own text, is the paper that *actually* "characterized" the
applied external field underlying the 1.4(5.2)×10⁻²¹ y-component prior
value: it was one citation away from a potential genuine field-
magnitude source for a USTC-based case analogous to Source 4's NPL case.
The project owner subsequently authorized fetching it (2026-08-10).

**This session could not retrieve it by any route attempted, and
therefore could not examine, extract from, or classify its content.**
Full access-attempt log: `benchmarks/SOURCES.md` section 6. Unlike the
PRL 133,023401 Supplemental Material (Source 3), which had a
substitute text to cross-check against (the arXiv v2 preprint, since it
turned out to be the same LaTeX source merged into one PDF), **no
substitute exists here**: this 2024 Metrologia paper has no arXiv
preprint (confirmed via four independent search strategies) and no
free/open route of any kind (IOPscience: explicitly not open access,
"Login / Purchase / Rent" only, DOM-inspected for a hidden accepted-
manuscript link as found on the JILA PRL page, none exists;
ResearchGate: 403-blocked). No credential entry, purchase, or paywall
bypass was attempted, per this session's binding safety rules; per the
benchmark protocol, this is reported precisely and the
attempt stops here; the owner may supply the PDF directly (as was done
for Source 5's USTC 2026 paper) if a future pass wants this examined.

**Classification (binding): "not accessed, classification not possible
without a copy."** This is a *distinct* outcome from every other
category in this document:

- Not `"reproducibility"` (Source 4): no data was seen, so no case
  could be built, let alone checked.
- Not `"not_applicable"` (Sources 1, 2, 5): that classification means
  the content *was* examined and found not to map to a pipeline config;
  here, nothing was examined at all. Reporting this as "not applicable"
  would misrepresent an access failure as a content finding.
- Not a negative "budget-only, no field data" verdict either, for the
  same reason: this session has no idea whether the paper publishes a
  field magnitude, an applied-voltage/geometry characterization, a
  shift-vs-field plot (which per the benchmark protocol would
  itself be flagged as "not acceptable for a benchmark, an outreach ask
  instead", but that determination also requires having read the
  paper, which did not happen here).

**No code representation was added to `benchmarks/loaders.py`/
`benchmarks/run_benchmarks.py` for this source**: there is no data to
load or classify, and adding a placeholder entry with no real content
would risk looking like a finding where none exists. This section, plus
`benchmarks/SOURCES.md` section 6 and `benchmarks/RESULTS.md`, are the
complete record of this attempt.

## WP20 addendum: JILA BBR row arithmetic-reproduction case (2026-08-11)

**Trigger:** WP20 design item 5 / gate edit
8: once the BBR physics module (E32/E33) landed, the WP20 gate required
a benchmark case re-running the JILA BBR row (arXiv:2403.10664 Table I)
arithmetic with the pinned registry polynomial, labeled per
the project's theory sign-off record (G7) B5.

**Source: no new fetch.** Both numbers this case consumes:
`benchmarks/loaders.JILA_BBR_TEMPERATURE_K` (JILA's own in-vacuum
operating temperature, `293.282(4) K`) and
`benchmarks/loaders.JILA_BBR_PUBLISHED_SHIFT` (Table I "BBR" row,
`-4.84172(73)×10⁻¹⁵`) come from the *same* arXiv:2403.10664v2 source
already fetched and checksummed for WP10 (`benchmarks/SOURCES.md` section
1), reused and extended here per the binding instruction to avoid a
duplicate fetch. The
temperature value is additionally cross-checked against the primary text
(Lisdat et al., PRR 3, L042036 (2021); Aeppli et al., PRL 133, 023401
(2024)) (an independent
verification sweep, 2026-08-11); this WP20 pass did not re-fetch or
re-read the PDF itself.

**Why this is a *different structural class* from every row above,
including the NPL reproducibility case:** every "not_applicable" row in
this document (JILA's own DC-Stark row, USTC's DC-Stark constraint, the
NIST phase data) is not-applicable because either the physics is out of
scope or no independent field/measurement input is published. The NPL
case is `"reproducibility"` because NPL's *field* was measured
independently of the clock transition. **This BBR case is neither**: the
physics is now in scope (unlike every `not_applicable` row), but the
comparison target (JILA's own BBR row) is itself a computed row (their T
through the standard BBR formula with their own fitted coefficients),
unlike NPL's independently measured field. Hence a third,
explicitly weaker label: **"arithmetic reproduction of a published
standard-formula evaluation"** (G7 sign-off B5, ratified); see
`benchmarks/RESULTS.md`'s "Arithmetic-reproduction case: JILA BBR row"
section for the full method, numbers, and interpretation, and
`benchmarks/run_bbr_jila_arithmetic_reproduction.py` for the generating
script (a separate, dedicated script, not folded into
`benchmarks/run_benchmarks.py`'s WP10 report/`kpi_summary`, so this
document's pinned WP10 totals are unaffected).

**Non-circularity caveat (per this document's own discipline of never
suppressing an uncomfortable framing):** the
registry's Sr87 dynamic BBR polynomial is the PTB-2025 rescaling of
Lisdat's fit shape, anchored to *this exact JILA row's* dynamic-term
value (`-153.06(33) mHz`, arXiv:2507.14030). Close agreement between this
engine's prediction and JILA's own published row is therefore expected
almost by construction. This agreement is evidence the engine's
arithmetic and citation/provenance chain correctly reproduce a formula
and coefficients this project's own registry was built from. It does
not, on its own, independently confirm the underlying BBR *physics*.
`benchmarks/RESULTS.md`
states this explicitly: the tight residual is expected arithmetic
agreement, distinct from NPL's independent-measurement success.

## Roos-benchmark addendum: Ca+:D5/2 two-ion quadrupole-slope case (2026-08-11)

**Trigger:** the WP21 Tier-2 ion-clock electric-quadrupole-shift module
(CONVENTIONS.md E34/E35) landed with an explicit AMBIGUITY note (section
14) reserving Roos et al.'s (quant-ph/0701215v1) full two-ion Fig. 3a/4a
absolute triple "for the separate, later Roos/Barwood benchmark WP";
this addendum is that WP, once the owner supplied the Roos primary text
(`benchmarks/SOURCES.md` section 7).

**Source: no dataset fetch, an extension of the already-logged dossier
extraction.** `benchmarks/loaders.ROOS_MEASURED_SLOPE_HZ_MM2_PER_V`
(Fig. 4a slope, `2.975(2) Hz*mm^2/V`) and `ROOS_FIT_OFFSET_HZ` (Fig. 4a
offset, `-2.4(1) Hz`) restate the already-reviewed primary-text
extraction from Roos et al., Nature 443, 316 (2006),
quant-ph/0701215v1 (Eq. 1/Fig. 4a) as typed `PublishedBand`
constants; both Theta values (`1.83(1)` measured, `1.917` Itano theory)
were already registered in
`cliffordclock.ensemble.species.QUADRUPOLE_MOMENTS["Ca+:D5/2"]` before
this addendum (WP21) and are consumed directly from that registry.

**Why this needs a FOURTH structural class, distinct from every one
above:** the NPL case is `"reproducibility"` because NPL's own field and
NPL's own shift are directly comparable, both independent of this
engine. The BBR case is `"arithmetic_reproduction"` because the
comparison target (JILA's own BBR row) is itself computed from the same
formula/coefficients being checked. **This Roos case is neither, on
either variant:**

- Its headline (cross-vintage) variant uses a Theta independent
  of Roos's own fit (Itano's ab-initio theory value, a different vintage
  and method), unlike the BBR case's circularity, this is a real
  external comparison. But unlike the NPL case's `"reproducibility"`
  label, Roos's own applied gradient (the OTHER ingredient in the
  comparison) is calibrated from the ion's own measured `omega_z`
  (`dE_z/dz = -m*omega_z^2/e`, dossier section 6), a trap-model-derived
  value, and Roos's own fit is what produced the slope `a` being
  predicted. So this is weaker than a `"blind_prediction"` case would be
  (which needs every ingredient on both sides independent of every
  other), but stronger/more genuine than `"arithmetic_reproduction"`
  (the Theta input comes from an independent theoretical calculation,
  external to the fit being predicted).
- Its secondary variant, using Roos's own extracted Theta, IS
  `"arithmetic_reproduction"`, circular by the same logic as the BBR
  case, computed and reported for completeness (G8 sign-off B4's binding
  instruction: "lead with the cross-vintage version; compute both").

Hence the new taxonomy entry, `"cross_vintage_comparison"`
(`benchmarks/RESULTS.md`'s "Classification taxonomy" section, added by
this addendum), for the headline variant specifically, distinct from
both `"reproducibility"` and `"arithmetic_reproduction"`.

**Non-circularity caveat for the headline variant:**
even though Itano's Theta is independent of Roos's fit, the
"agreement" (or, here, the recovered ~4.7% disagreement) is bounded by
Roos's own measured-slope uncertainty AND by the fact that Itano's theory
value carries no published uncertainty at all; there is no meaningful
"combined uncertainty band" to report for this variant beyond Roos's own
tight `2.975(2)` figure. `benchmarks/RESULTS.md` reports this residual as
a recovered, literature-known theory-vs-measurement tension (G8 sign-off
B4's first nuance, applied here as it was for Barwood): the residual
measures the size of that known tension, and neither the engine's own
precision nor the theory value's precision is what this figure scores.

**Structural pin, computed not asserted:** the two-ion 24/5 enhancement
Roos states (dossier section 6) is independently recomputed here,
directly from real
`quadrupole_shift_joules`/`quadrupole_mj_factor` calls
(`benchmarks/run_roos_quadrupole_slope.py::structural_two_ion_enhancement_ratio`,
pinned by `tests/test_roos_benchmark.py`), the case's structural spine.

See `benchmarks/RESULTS.md`'s "Cross-vintage comparison: Roos et al.
quadrupole slope" section for the full method, numbers, and
interpretation, and `benchmarks/run_roos_quadrupole_slope.py` for the
generating script (a separate, dedicated script, not folded into
`benchmarks/run_benchmarks.py`'s WP10 report/`kpi_summary`, so this
document's pinned WP10 totals are unaffected, same pattern as the WP20
BBR addendum above).

## Summary

Across six authorized sources (arXiv:2403.10664 + its PRL Supplemental
Material follow-up, data.nist.gov DOI 10.18434/M32206, arXiv:1706.01944,
Metrologia 63,025002, and Metrologia 61,015006) and ~14 classifiable rows
considered:

- **1 reproducibility case, MET** (Source 4, NPL arXiv:1706.01944): this
  engine's `coupling.type: stark_dc` pipeline, given NPL's published
  field and PTB's published `Δα`, reconstructs NPL's own published
  DC-Stark shift band. Labeled: end-to-end pipeline correctness
  against independently published inputs/outputs with no fitting, NOT
  a blind prediction, since NPL combined the same two ingredients
  themselves.
- **0 blind-prediction cases**: still none available from any
  authorized source; every other in-scope row lacks the independent
  field input a genuine blind prediction would need. Source 6 (Li J et
  al 2024, Metrologia 61,015006) was the most promising lead for closing
  this gap, authorized, attempted, and **blocked** (no arXiv preprint,
  no open-access route, ResearchGate 403); its content remains entirely
  unknown to this project.
- **13 not-applicable rows**: JILA's Table I DC-Stark row (Source 1) and
  the USTC Metrologia 63,025002 DC-Stark constraint (Source 5) both
  publish only a resulting shift/bound with no independent field input
  (a mapping/data-completeness gap, confirmed to persist for JILA even
  after the Supplemental Material follow-up, Source 3); every other row
  across both papers (BBR, Zeeman, density, lattice light, background
  gas, tunneling, etc.) is physics entirely outside this engine's scope;
  the NIST M32206 dataset (Source 2) measures an entirely different
  physical quantity, a scope mismatch distinct from the field-input data
  gap affecting the DC-Stark rows above.
- **Source 6: not accessed, not classified**: a fourth, distinct
  outcome category from the three above, reported here as its own line
  item. The "not-applicable" count above excludes it: that label is
  reserved for content that was examined and found out of scope, and
  Source 6's content was never obtained to examine. See the Source 5
  section's "Reference [30] follow-up" subsection above and
  `benchmarks/SOURCES.md` section 6 for the full access-attempt log.

This is the outcome this project's binding evaluation rules ask
for, reported as found: neither a forced "everything passes" story nor a
suppressed "nothing works" one. One genuine, correctly-labeled
reproducibility success; zero blind predictions (a real capability gap in
what public data currently supports); the rest
reported as not-applicable with exact reasons. USTC's reference
[30] is flagged as the clearest lead for a future case.

## WP22 addendum: Bothwell mm-scale gravitational-redshift reproducibility case (2026-08-11)

**Trigger:** WP22 Part 3 / G9 sign-off
Part B: the extended-lattice ensemble regime (WP22 Part 2) plus the E36
gravitational-redshift pivot term (WP22 Part 1) together let this engine
reproduce a real mm-scale redshift measurement end-to-end.

**Source:** Bothwell et al., Nature 602, 420 (2022), arXiv:2109.12238
(research sweep, read cover to cover, 2026-08-11), and, for the
reference gravity, van Westrum, NOAA Technical Memorandum NOS NGS-77
(2019). See the staged `benchmarks/SOURCES.md` entry below for the full
citation/checksum-style record this document's own discipline expects.
This WP22 pass reuses the dossier's own research sweep, which already
read the preprint in full, matching the WP20 BBR addendum's precedent of
extending an already-authorized source directly from its existing
record; this pass did not fetch or checksum the preprint independently.

**Why this is `"reproducibility"`, with the caveat INVERTED from the
BBR/JILA case's:** the WP20 BBR case is `"arithmetic_reproduction"`
(weaker) because JILA's own comparison ROW is itself a computed number,
derived from the same standard formula and coefficients being checked.
Bothwell's comparison target IS an
independent measurement (a real, physically observed per-pixel frequency
map, fit to a slope), structurally the same class as the NPL case. The
caveat that keeps this from reading as a strong "blind prediction",
though, is that the underlying arithmetic (`g/c^2`) is textbook, and
Bothwell computed it themselves in their own paper; this project's
contribution is running that same textbook arithmetic through the full
extended-sample MACHINERY (geometry, envelope weighting, per-site pivot
evaluation, map fit) to reproduce their measured slope. See
`benchmarks/RESULTS.md`'s "Reproducibility case: Bothwell..." section
(staged above) for the full method and `benchmarks/run_bothwell_redshift.py`
for the generating script (a separate, dedicated script, not folded
into `benchmarks/run_benchmarks.py`'s WP10 report/`kpi_summary`, so this
document's pinned WP10 totals are unaffected by this addition alone; see
the headline-count note at the top of the staging file for the open
question of whether it SHOULD eventually join that count, given it is a
`"reproducibility"`-class case unlike the BBR arithmetic-reproduction
case).

**Data availability, a hard negative:** no deposited
per-pixel/per-slice dataset was found anywhere accessible for this paper
(dossier "Data availability"); the comparison is slope-level only against
the two corrected numbers the paper itself publishes (Table 1 + main
text). This mirrors the NPL case's own slope/band-level (not per-sample)
comparison discipline.
