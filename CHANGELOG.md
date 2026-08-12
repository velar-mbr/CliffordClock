# Changelog

All notable changes to CliffordClock are recorded here. Versions follow
[Semantic Versioning](https://semver.org/); this project is pre-release
(`0.y.z`), so minor bumps may include breaking changes. This file follows
the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

(Nothing yet.)

## [0.1.0] — 2026-08-12

### Added

- **WP27 paper expansion**: the paper gains the gravitational-redshift
  and extended-lattice physics section, the ion electric-quadrupole
  section, the Bothwell and Roos validation-case sections, and the
  per-site frequency-map figure; 18 pages, every quoted number from
  build-time generated macros.

- **Sprint 4 wave 2: public-tree scrub, README rewrite, grand-tour
  notebook, dash purge** (2026-08-12, pre-release wave for the showable
  snapshot):
  - **Internal-path scrub + `internal-path-check`:** every path
    reference into the two private top-level directories left in
    public-facing files (benchmark
    scripts and loaders, species/omega/pipeline docstrings, docs,
    tests, `paper/main.tex`, figure scripts) replaced with the primary
    citation it stood for (Roos et al., Nature 443, 316 (2006);
    Lisdat et al., PRR 3, L042036 (2021); Aeppli et al., PRL 133,
    023401 (2024); Bothwell et al., Nature 602, 420 (2022)) or a
    path-free process label ("the project's theory sign-off record
    (G8)"). `tools/release_checks.py` grows a matching
    `internal-path-check` subcommand (now 8 checks; `git ls-files`
    walk, CHANGELOG released-history exemption, allowlist support)
    with ~16 new tests, so the scrub is mechanically enforced from now
    on. Benchmark artifacts regenerated; notebooks 07/08 re-executed;
    paper rebuilt (13 pages, generated files byte-identical).
  - **README rewrite:** the showcase animation
    (`docs/assets/showcase_animation.gif`) leads, followed by a
    definition paragraph, the MVP argument (what a run of this tool
    does that a textbook coefficient formula cannot: per-atom
    composition of every term on real trajectories with dispersion,
    not one hand-computed number), an affirmative capability
    checklist, and a single pointer to `docs/roadmap.md`, which now
    also carries the blind-prediction partner program and the
    transportable/accelerating-clock direction as roadmap sections.
  - **`notebooks/10_grand_tour.ipynb`:** one chamber-scale scenario
    composed live through all three lattice-clock terms (DC Stark,
    then +BBR, then +gravity), cross-checked through the Cl(1,3) rotor engine on
    identical trajectories (agreement at the double-precision floor on
    the fully composed calculation), then bridged to the
    extended-lattice per-site view (fitted slope equal to `g/c^2` to
    4.5e-16 relative). Indexed in `docs/index.md`.
  - **Dash purge (prose-scan now PASS):** all 141 ASCII
    dash-as-punctuation findings swept from public prose (docs,
    notebooks 01-10, the four benchmark generators' report literals,
    regenerated results). Binding classification labels reworded with
    semicolons only; every test-pinned phrase preserved.
    `LATEX_EN_DASH_RE` gains math-mode exponent-range and
    `\eqref`-range alternations (3 paper false positives), with a
    regression test; two deliberate keeps documented in
    `tools/release_checks_allowlist.toml` (a verbatim APS paywall-page
    quote; a G9-pinned sentence in CONVENTIONS.md).

- **WP22: extended-lattice ensemble + gravitational redshift + Bothwell
  mm-scale benchmark** (2026-08-11, beta gate, owner trigger after
  reviewing the Fortier/Luiten/Margolis survey, Optica 13, 143 (2026);
  the project's G9 theory sign-off record):
  CONVENTIONS.md section 15 adds E36 (the gravitational-redshift pivot
  term, `(P-1)_grav = g*(h-h_ref)/c^2`, sign pinned to "a higher clock
  runs faster" and magnitude pinned to the COMPUTED `g/c^2 =
  1.0911370e-16/m`, correcting a transcription slip both the theory brief
  and the Bothwell dossier's first draft carried, `1.0912e-16/m`).
  - **Part 1 (physics):** `cliffordclock.integrator.omega.grav_pivot_perturbation`/
    `height_along_axis` implement E36; `pivot_perturbation_stark`/
    `spin_connection_stark`/`scalar_rate_perturbation_stark`/
    `build_omega_stark` gain a new keyword-only `grav_pivot_perturbation`
    parameter (default `0.0`, exact no-op), mirroring WP20's
    `bbr_pivot_perturbation` threading pattern exactly but, like the
    quadrupole term, per-position (varies with height). Reaches the rotor
    through the scalar `B_hat_C` rotation-plane coefficient only -- never
    `omega_boost`'s numerator, which is provably (not just boundedly)
    inconsequential since every lattice/lattice_extended node this
    project evaluates is static (`v=0`) and `omega_boost`'s own
    coefficient carries an explicit factor of `v`. New top-level
    `environment.gravity:` config section
    (`cliffordclock.pipeline.GravityConfig`: `g_m_s2` default
    `cliffordclock.constants.STANDARD_GRAVITY` = 9.80665 m/s^2 exact by
    definition, `up_axis`, `reference_height_m`), requiring
    `coupling.type='stark_dc'` (mirrors BBR's cross-field validation) and
    composed into every evaluation mode
    (`fast_path`/`secular`/`direct`-batched/`direct`-streaming/`worldline`)
    -- absent from every shipped example (byte-identical output). A
    runtime warning (`cliffordclock.pipeline.GRAVITY_EXTENT_WARN_M`, 10 m,
    not a hard rejection) fires when a run's sampled positions span more
    than that along `up_axis`, an order-of-magnitude margin below the
    uniform-g approximation's ~76 m, 1e-19-floor validity bound (G9
    sign-off A3).
  - **Part 2 (`lattice_extended` ensemble regime):**
    `cliffordclock.ensemble.lattice.extended_lattice_nodes` builds `n`
    copies of the `lattice` regime's own single-site Hermite-Gauss
    quadrature, distributed along a configured axis
    (`ensemble.site_axis`) at `ensemble.site_spacing_m` spacing with a
    Gaussian-or-uniform site-occupation envelope
    (`ensemble.site_envelope`/`site_envelope_sigma_m`); every site's own
    position feeds every pivot term already in scope (local field, uniform
    BBR, height-dependent gravity) through the SAME `fast_path`/`worldline`
    accumulators the `lattice` regime uses -- no new evaluation-mode
    machinery, and the existing `lattice` regime (and every shipped
    example) is entirely byte-identical. New
    `PipelineResult.site_map`/CLI `site_map.json` output
    (`cliffordclock.pipeline.LatticeExtendedSiteMap`): the per-site
    frequency map (the Bothwell observable) plus a weighted-least-squares
    linear-gradient fit (`slope_per_m`, the map's headline number) and the
    G9 sign-off's mandated dispersion-labeling split
    (`total_spread_fractional`/`gradient_removed_residual_spread_fractional`,
    separating the deterministic per-site height gradient from genuine
    stochastic spread, with a test-pinned report note,
    `cliffordclock.pipeline.LATTICE_EXTENDED_DISPERSION_LABEL_NOTE`) --
    deliberately NOT a `MetrologyReport`/`report.json` schema change.
  - **Part 3 (Bothwell 2022 benchmark case):**
    `benchmarks/run_bothwell_redshift.py` configures the REAL
    `lattice_extended` pipeline to Bothwell et al.'s (Nature 602, 420
    (2022) / arXiv:2109.12238) mm-scale Sr sample geometry (Gaussian
    envelope, ~5900-site computational grid at the real 406.5 nm magic-
    lattice site spacing [INFERRED from lambda/2] and an envelope sigma
    [INFERRED from their stated pixel-count analysis window], zero field
    to isolate the gravitational term) at their USGS-surveyed local
    `g = 9.796 m/s^2` (van Westrum, NOAA NGS-77 (2019)), and compares the
    real pipeline's fitted slope against BOTH their corrected measurements
    (method A -9.8(2.3)e-20/mm, method B -1.28(27)e-19/mm) with explicit
    band-overlap verdicts -- MET at 0.48-sigma/0.70-sigma respectively,
    matching the G9 sign-off's own pinned numbers exactly. Classified
    `case_class="reproducibility"` with the inverted-NPL caveat (the
    `g/c^2` arithmetic is textbook; what is validated is the extended-
    sample machinery producing the measured-map slope end-to-end with zero
    adjustable inputs) -- kept in a separate script/report (mirrors WP20's
    BBR-arithmetic-reproduction precedent), NOT merged into
    `benchmarks/run_benchmarks.py`/`wp10_results.json`'s frozen
    `kpi_summary` counts; whether/how this reproducibility case should
    join that headline count is left as an explicit open question for the
    coordinator/owner (this case's class differs from the BBR case's
    weaker `"arithmetic_reproduction"`, so the precedent is not a full
    match). The engine's own coordinate-sign convention (higher = faster)
    is explicitly, documentedly negated when comparing against Bothwell's
    own published convention (their z-axis increases toward lower
    physical height) -- the sign agreement is deliberate, not
    coincidental (G9 sign-off gate edit 2).
  - New tests: `tests/test_gravity_pivot.py`, `tests/test_lattice_extended.py`,
    `tests/test_bothwell_benchmark.py`.

- **WP21 Tiers 1+2: ion-clock support** (2026-08-11, owner directive:
  "what would it take to add [ions]? I think that is worth including
  before release"; G8 theory sign-off,
  the project's theory sign-off record (G8)): CONVENTIONS.md E34
  (ion-clock electric-quadrupole level shift, sign pinned to Roos et al.
  quant-ph/0701215v1 Eq. 1 primary text plus Itano 2000 Eq. 46's
  hyperfine-form sign reconciliation, and a coordinate-free reduction
  algebraically equivalent to both) and E35 (quadrupole pivot
  composition: additive in `(P-1)` per E33's pattern, exact three-
  orthogonal-orientation cancellation proof, traceless-symmetric-gradient
  requirement, documented spin-connection scope limit).
  - **Tier 1 (scalar ions):** `cliffordclock.ensemble.species.AL27_PLUS`
    now carries `delta_alpha_dc_si` (Wei et al. 2024, 0.416(14) a.u.,
    secondary, with Brewer et al. 2019's 0.426(58) a.u. primary-text
    fallback recorded); new `IN115_PLUS` species (Safronova et al. 2011
    theory Delta_alpha(0) = 2.01 a.u., primary; clock frequency from
    Ohtsubo et al. 2017, arXiv:1703.02717, primary). Neither carries
    `BbrCoefficients` (the dossier's single-datum BBR evidence for each
    does not support the independent static/dynamic split the registry
    requires -- not invented). `ION_MICROMOTION_NOTES`/
    `ION_HYPERFINE_E2_BUDGET_NOTES`: per-species report notes (shared-
    stray-field-cause micromotion boundary, strongest for J=0; the
    hyperfine-mediated E2 budget line for I != 0), carried on every
    report for these two species regardless of `coupling.type`.
  - **Tier 2 (quadrupole shift):** `cliffordclock.ensemble.species.QUADRUPOLE_MOMENTS`
    (Ca+ D5/2, primary-verified via an owner-supplied Roos et al.
    preprint read in full; Sr+ D5/2, Ba+ D5/2, Yb+ D3/2 secondary; Yb+
    F7/2 primary and negative, the registry's sign anchor) and
    `EA0_SQUARED_SI` (the atomic-unit-of-electric-quadrupole-moment
    conversion constant, computed from CODATA e/a0, not hand-transcribed
    -- the G4 `ALPHA_AU_TO_SI` digit-swap lesson applied here).
    `cliffordclock.integrator.omega.quadrupole_pivot_perturbation`/
    `quadrupole_shift_joules`/`quadrupole_mj_factor`/
    `quadrupole_three_orientation_average`/`traceless_symmetric_gradient`
    implement E34/E35; `pivot_perturbation_stark`/`spin_connection_stark`/
    `scalar_rate_perturbation_stark`/`build_omega_stark` gain a new
    keyword-only `quadrupole_pivot_perturbation` parameter (default
    `0.0`, exact no-op), mirroring WP20's `bbr_pivot_perturbation`
    threading pattern. New top-level `quadrupole:` config section
    (`cliffordclock.pipeline.QuadrupoleConfig`), composed into every
    `coupling.type='stark_dc'` evaluation mode via `_make_stark_rate_fn`/
    `_stark_rotor_ensemble`; requires `coupling.type='stark_dc'` and is
    absent from every shipped example (byte-identical output).
  - **Sign discipline (G8 gate edit 1):** the convention-free m_J ratio
    (`-1.25` for a D5/2 state) and the Yb+ F7/2 negative-Theta relative-
    sign anchor are pinned regressions. The ABSOLUTE sign anchor (a
    measured `(gradient, state, shift-sign)` triple from Dube et al. 2005
    or Barwood et al. 2004) was initially flagged **AMBIGUITY** (no
    accessible arXiv/ar5iv preprint at build time); the owner has since
    supplied Dube 2005 in primary text, which confirms the E34 sign form
    as a third independent source and adds the magic-m_J^2 intercept
    regression (ion-clock dossier section 7, with its caution that
    Dube's measured slope is ~95% micromotion tensor Stark and never a
    pure quadrupole anchor) -- E34's leading sign
    is pinned directly to Roos et al.'s own primary-text Eq. 1,
    per the gate's explicit fallback discipline ("implement the ratio +
    Yb+ anchors and flag the absolute triple as requiring owner-supplied
    primary text, do not fabricate"). The Roos two-ion entangled-state
    Fig. 3a/4a absolute-sign dataset (owner-supplied, in the dossier) is
    left for the separate, later Roos/Barwood benchmark WP, per the WP21
    instruction file's explicit exclusion.
  - New tests: `tests/test_ion_species.py` (registry pins),
    `tests/test_quadrupole_pivot.py` (pure-formula: sign regressions,
    ea0^2 unit pin, exact three-orientation cancellation, traceless-
    symmetric-part requirement, closed-form known answers both signs,
    coordinate-free-vs-literal-axial-form equivalence),
    `tests/test_quadrupole_pipeline.py` (config parsing, composition
    additivity, cross-mode agreement, shipped-example byte-exactness, a
    real-FEA quadrupole-shift-map demonstration from
    `examples/fd_electrode_field.txt`). `tests/test_ensemble_species.py`/
    `tests/test_stark_species.py`/`tests/test_stark_pivot.py`/
    `tests/test_e2e.py` updated for Al27+'s newly-populated Stark data
    (superseding their pre-WP21 "Al27+ has no Stark data" cases).
- **WP20 follow-up: JILA BBR-row arithmetic-reproduction benchmark case**
  (2026-08-11, WP20 design item 5, gate edit
  8): `benchmarks/run_bbr_jila_arithmetic_reproduction.py` evaluates the
  engine's real BBR pivot functions (`bbr_pivot_perturbation`/
  `bbr_pivot_uncertainty`) with the pinned Sr87 registry coefficients at
  JILA's own published operating temperature (`293.282(4) K`) and
  compares against JILA's own published BBR row (arXiv:2403.10664 Table
  I: `-4.84172(73)e-15`): predicted `-4.841743e-15`, residual
  `-2.251e-20`, bands overlap, `kpi_verdict = "MET"`. Labeled
  **"arithmetic reproduction of a published standard-formula evaluation"**
  (G7 sign-off B5) -- explicitly weaker than the NPL `"reproducibility"`
  case (`benchmarks/RESULTS.md`), since JILA's own row is itself computed
  and the registry's dynamic polynomial is anchored to this exact JILA
  value, so agreement is expected almost by construction. Kept as a
  separate script/report from `benchmarks/run_benchmarks.py`'s WP10
  `kpi_summary`, which is unchanged by this addition.
  `docs/validation.md`, `benchmarks/RESULTS.md`/`MAPPING.md`/`SOURCES.md`,
  and the README's "What it does today" list updated accordingly; new
  regression tests, `tests/test_bbr_benchmark.py`.
- **WP20: blackbody-radiation shift** (2026-08-11, owner directive: "for
  BBR, given this is the largest error source, we should figure out how
  to handle it"; G7 theory sign-off,
  `docs/G7_physics_signoff_theory.md`): CONVENTIONS.md E32 (the BBR pivot
  term, static + dynamic-polynomial split, G7-corrected sign) and E33
  (scalar pivot composition, extended with the BBR term and the 4th-order
  hyperpolarizability neglected-term bound).
  - `cliffordclock.ensemble.species.BbrCoefficients`: per-species
    static/dynamic BBR coefficients + uncertainties + full citations,
    populated for `Sr87` (Middelmann static, PTB-2025-rescaled dynamic
    polynomial) and `Yb171` (Hassan static + T^6, Beloy-derived T^8);
    unpopulated for `Al27+`.
  - New optional top-level `environment:` config section
    (`radiation_temperature_K`, `radiation_temperature_uncertainty_K`,
    `cliffordclock.pipeline.EnvironmentConfig`): absent means BBR off,
    every shipped example byte-identical to before this change (pinned
    regression, `tests/test_bbr_pipeline.py`). Requires
    `coupling.type: stark_dc`; hard-rejects temperatures outside
    `[50, 350]` K.
  - `cliffordclock.integrator.omega.bbr_pivot_perturbation`/
    `bbr_pivot_uncertainty`: the E32 formula and its coefficient-
    /temperature-uncertainty propagation (G7 sign-off A4#2-3), composed
    into `pivot_perturbation_stark`/`spin_connection_stark`/
    `scalar_rate_perturbation_stark`/`build_omega_stark` via a new
    keyword-only `bbr_pivot_perturbation` parameter (default `0.0`, an
    exact no-op) -- reaches every evaluation mode (`fast_path`,
    `secular`, classical `direct` batched and streaming, and the rotor
    `worldline` path) through the existing `coupling.type: stark_dc`
    plumbing, with no change to any `coupling.type: linear_mu` path.
  - Report `uncertainty_notes` gain a BBR provenance line (T,
    coefficients, citations), an M1/E2 multipole "modeled-out, `~6e-20`
    each" budget line, a `300 < T <= 350` K "beyond the PTB<->JILA
    cross-verified range" note, and explicit "arithmetic-reproduction
    fidelity" labeling wherever a `1e-19`-class BBR number is stated
    (never presented as BBR accuracy).
  - Docs: `docs/CONVENTIONS.md` §13, `docs/coupling.md`'s
    "Blackbody-radiation shift" section, `docs/cli.md`'s "Environment"
    section, `docs/validation.md`'s new KA5 row.
  - Deliberately out of scope (later work): T(r) spatial maps,
    solid-angle effective-temperature computation, stochastic BBR-field
    sampling, and hyperpolarizability/BBR-Zeeman terms. The JILA-2024
    benchmark-case reproduction, deferred at the time of this entry as a
    separate work package with its own review, is now built -- see the
    "WP20 follow-up" entry above.
  - Same `config_hash` caveat as WP19 (next entry): `report.json`'s
    `config_hash` changes for every shipped example because
    `PipelineConfig` gained the `environment` field. All physics outputs
    and the line-profile CSV are byte-identical (pinned by regression
    tests against a pre-change baseline); this is the pre-existing
    `_config_hash`-changes-on-schema-addition behavior, not a physics
    change.

- **WP19: streaming/chunked evaluation -- every configuration runnable in
  bounded memory** (2026-08-11, owner directive: "it should be runnable
  even if a naive approach uses too much memory"): the trajectory-memory
  guard (previous entry below) becomes a mode *selector*, not just a
  wall, for `ensemble.regime: classical` + `integration.mode: direct`
  (both `coupling.type` values).
  - `cliffordclock.fields.smoother.chunked_apply`/`FieldSmoother.evaluate_chunked`:
    evaluates a `FieldSmoother`-backed field over fixed-size query chunks
    (default 4096 points) instead of one unbounded call, bounding peak
    memory to `chunk_size x K x 3 x 8 x factor` bytes independent of the
    query-point count. Verified bitwise-identical to the unchunked path
    for `chunk_size >= 2` (<= 1 ulp at `chunk_size == 1`, a documented
    `jax.vmap`-batch-of-one XLA lowering detail, not a bug).
  - `cliffordclock.pipeline._stark_scalar_ensemble_streaming`/
    `_direct_rotor_ensemble_streaming`: fuse velocity-Verlet propagation
    with the per-step E19/E21-E22 phase accumulation into a single
    `jax.lax.scan` over the whole ensemble at once, so the dense
    `(M, steps + 1, 3)` trajectory (and, for `coupling.type: stark_dc`,
    the smoother's whole-trajectory evaluation) is never materialized --
    memory O(M), independent of `steps`. Per-step field/`rate_fn` calls
    are routed through `chunked_apply` too, so an unusually large
    `ensemble.size` stays bounded independent of `M` as well. Measured
    agreement with the existing batched accumulators: bitwise-identical
    for the `linear_mu` rotor path, ~1.7e-16 relative (machine-epsilon
    level) for the `stark_dc` scalar path against a real
    `FieldSmoother`-backed field.
  - New config key `integration.evaluation: auto | batched | streaming`
    (default `auto`): under `auto`, exceeding
    `integration.max_trajectory_memory_gb`'s batched-path estimate now
    switches to the streaming accumulator (noted in the report's
    `uncertainty_notes`) instead of raising `PipelineConfigError`;
    `batched`/`streaming` force one path explicitly.
    `integration.trajectory_stride` optionally retains a periodic
    position snapshot for the streaming path's `PipelineResult.trajectories`
    (which otherwise keeps only the initial/final position, `O(M)`).
    `worldline`/`secular` are out of scope (no streaming accumulator; the
    pre-WP19 hard-reject guard is unchanged there) -- see
    `docs/timescales.md`'s rewritten "Safety net" section and
    `docs/cli.md`.
  - `examples/showcase_gradient_dispersion_sr87.yaml` (the config whose
    `max_trajectory_memory_gb: 8.0` override the previous guard entry
    below documents) now also runs via forced streaming with that
    override removed: measured peak RSS ~0.36 GB (vs. the batched path's
    ~4.69 GB on the same machine), bitwise-identical
    `mean_fractional_shift`/SEM to the batched run. The shipped example
    itself is unchanged and stays on the batched path by default.
  - No changes to the `fast_path`/`secular` math or the rotor accumulator
    (`worldline.integrate_ensemble`/`cliffordclock.integrator.stepper.rotor_step`,
    reused unmodified by the new streaming accumulators); the batched
    path remains the fast default whenever a config fits its budget.
  - One provenance-field caveat to the "shipped examples unchanged"
    claim: `report.json`'s `config_hash` changes for every example,
    because `_config_hash` hashes the full config dataclass and
    `IntegrationConfig` gained two default-valued fields
    (`evaluation`, `trajectory_stride`). All physics outputs and the
    line-profile CSV are byte-identical; this is pre-existing
    `_config_hash` behavior under any config-schema addition, not a
    WP19 effect per se.

- **Paper showcase: chamber-scale field to full inhomogeneous-shift
  budget** (2026-08-11): the paper's central differentiator demonstration
  -- a CAD/FEA-style field with genuine spatial structure, propagated
  through a real Monte Carlo ensemble, reported as a full dispersion
  budget (mean shift, ensemble spread, T2*, line profile), not just a
  single mean-shift number, with the pipeline's trajectory (scalar) mode
  and Cl(1,3) rotor mode run head to head on the *identical* Monte Carlo
  trajectories.
  - `examples/generate_showcase_field.py`: a from-scratch (NumPy +
    `scipy.sparse`, Jacobi-preconditioned conjugate gradient only, no
    COMSOL) finite-difference solve of a chamber containing two
    differently-sized, differently-biased, non-mirror-symmetric electrode
    plates plus a patch-potential spot embedded in one wall -- deliberately
    unlike a plain parallel-plate capacitor (nearly linear across any
    small region) so the exported field carries genuine curvature and
    cross-terms. Committed output: `examples/showcase_field.txt`
    (729 points, COMSOL-Spreadsheet format, loaded via the existing
    `load_field_comsol` path).
  - `examples/showcase_gradient_dispersion_sr87.yaml`: the runnable
    scenario config (`cliffordclock run
    examples/showcase_gradient_dispersion_sr87.yaml`) -- Sr-87,
    `ensemble.regime: classical`, `coupling.type: stark_dc`, a
    100-rad/s isotropic trap and 50 uK temperature chosen so the
    classical thermal cloud sigma (~692 um) is a meaningful fraction of
    the field's own spatial-variation scale (documented scale reasoning
    in the config's header comment) -- a scenario-*geometry* choice, no
    physical constant touched.
  - `paper/figures/fig4_showcase_gradient_dispersion.py`: runs the real
    pipeline's trajectory mode via `run_pipeline_full`, then re-accumulates
    the *identical* Monte Carlo trajectories through the rotor mode
    directly (`cliffordclock.pipeline._stark_rotor_ensemble` -- the
    shipped config schema's `integration.mode=worldline` keyword only
    exposes this cross-check for static lattice nodes, not a moving
    classical trajectory, so the accumulator is driven directly here,
    exactly as the paper's text explains). **Measured agreement:** mean
    shift relative difference `0.0`, max per-atom phase difference
    `~1.8e-12` (float64-noise scale) -- the trajectory and rotor modes
    compute the same number two independent ways. Produces
    `figures/fig4_showcase_gradient_dispersion.pdf` (three panels:
    whole-chamber field context with electrode/patch geometry, a zoomed
    locally-fitted field with a Monte Carlo trajectory subsample colored
    by accumulated shift, and the per-atom shift distribution + line
    profile with T2* annotated) and `generated/showcase_values.tex`
    (every quoted number in the paper's new showcase section). **Measured
    numbers:** mean fractional shift `-9.965e-17`, ensemble spread
    `6.51e-18`, T2* `279.7 us` -- roughly three orders of magnitude below
    the scenario's own ~0.2 s interrogation window, a genuine,
    quantitative finding (an experiment run this way would lose coherence
    to field-gradient inhomogeneity well before the interrogation time
    itself) a lab could not get from a field-gradient map and a
    mean-shift calculation alone.
  - `notebooks/05_gradient_showcase.ipynb`: the same computation as a
    narrated notebook walkthrough (field geometry, scale reasoning,
    trajectory-mode run, rotor-mode cross-check, shift distribution + line
    profile), executed in CI (`.github/workflows/ci.yml`'s `notebook`
    job gains a step for it) and committed with real execution outputs.
  - `paper/main.tex`: the former "Worked example" section is replaced by
    a new "Showcase" section (`\S`, three subsections: scenario/scale
    reasoning, trajectory-vs-rotor agreement, the inhomogeneous shift
    budget) built around Figure 4 above; the abstract, introduction, and
    NPL section are updated to reference it. `paper/figures/fig1_worked_example.py`
    and its committed inputs are unchanged and still run by `make
    figures` (a valid, still-correct demonstration) but are no longer
    cited in the paper's text.
  - Test contract: `tests/test_e2e.py` gains
    `test_showcase_scenario_config_stays_within_memory_safety_bounds`,
    `test_showcase_scenario_trajectory_and_rotor_modes_agree_and_shift_in_expected_range`
    (rotor-vs-scalar agreement to a `1e-9`/`1e-8`/`1e-6` tolerance triad,
    a documented expected-range regression guard on the mean shift and
    T2*, and a genuine-dispersion sanity check), and
    `test_showcase_scenario_cli_smoke`.

- **Paper voice pass: evaluation modes, not "production path"; jargon
  bridges** (2026-08-11): a full rewrite pass over `paper/main.tex`
  addressing three things together, since they touch the same
  sentences throughout the paper:
  1. **Development-history voice removed.** Every construction describing
     the *codebase's own history* ("does now exist", "has since been
     closed", "previously a genuine, tracked gap", "the rotor engine's
     validated scope has grown", etc.) is rewritten as a timeless
     statement of what the pipeline *is* and *why it is useful* -- most
     visibly in the former "general formalism, not the production path"
     subsection (now "Three evaluation modes, one physical model") and
     the Limitations section's rotor-engine paragraph. The one
     "no longer a formality" phrase describing the optical-clock field's
     own multi-decade trend (not this codebase's history) is left as
     legitimate domain context. No claim or tolerance was changed --
     this is a voice pass, not a claims change; the binding
     reproducibility-vs-blind-prediction classification taxonomy and every
     validation caveat are unchanged.
  2. **"Production path" framing replaced with user-facing evaluation
     modes.** The paper now states plainly: a **fast analytic mode**
     (lattice motional states, exact, E29, no time integration), a
     **trajectory mode** (a Monte Carlo ensemble's actual classical
     motion through a spatially varying field -- what the showcase runs),
     and a **rotor mode** (the general Cl(1,3) engine) -- with the scalar
     modes stated to agree with the rotor mode to floating-point
     precision on every case the paper computes (cited: the rotor-vs-
     scalar test suite and the showcase's own head-to-head comparison),
     existing purely for speed, and the rotor mode's distinct necessity
     stated to begin exactly where a physical effect stops being a single
     number per point (magnetic couplings, internal-state structure,
     transport-induced geometric phases -- the roadmap). **Guardrail
     honored throughout:** the paper states repeatedly and explicitly
     that the scalar pivot fully handles a spatially varying field on its
     own -- nothing about a field gradient or curvature ever requires the
     rotor; the differentiator being showcased is the *pipeline*
     capability (real field + real atomic distribution -> inhomogeneous
     dispersion observables), not a rotor requirement.
  3. **Jargon bridges added at first use**: rotor, bivector, pivot,
     Gauss-Hermite quadrature, secular averaging, and worldline each get
     a short plain-English parenthetical or em-dash clause at their first
     appearance in the body text (abstract mentions are left terse, per
     normal abstract convention).
  - `paper/figures/make_figures.py` and `paper/README.md` updated:
    `fig4_showcase_gradient_dispersion.py` added to the regeneration
    list/table; `fig1_worked_example.py`'s row notes it is still built
    but no longer cited in `main.tex`'s text.
  - `paper/main.tex`'s Table~2 ("evaluation modes and their performance
    tiers") gains an explicit "Evaluation mode" column mapping each
    internal tier (A/B(i)/B(ii)/C) to its user-facing mode name.
  - Build verified: `make distclean && make all` (figures + latexmk)
    succeeds cleanly, 11 pages, no undefined references.

### Fixed (process, not code)

- **Memory-safety incident and guardrails** (2026-08-11): during this
  paper's development, an interactive exploratory Python run (a larger
  Monte Carlo ensemble at a longer, auto-selected interrogation window,
  against an earlier draft of the showcase field with a larger
  `FieldSmoother` fit point count) exceeded a 120s foreground timeout,
  was moved to a background process by the tool harness, and was never
  monitored or killed -- it went on to drive memory usage past 100 GB and
  hung the development machine. Root cause: `FieldSmoother.evaluate`'s
  cost scales with `n_query_points * n_fit_points` (JAX materializes
  several intermediate query-by-fit-point arrays for the autodiff graph,
  not one dense kernel matrix -- a naive "8 bytes/element" estimate
  undercounts the real cost by over an order of magnitude), and neither
  the field generator's export point count nor the Monte Carlo ensemble
  size/step count were bounded against that product. No `src/` code was
  at fault or changed; this was a scenario-parameter and process-discipline
  issue in the showcase work described above. Fixes, all committed:
  - `examples/generate_showcase_field.py`: solve grid capped at
    <=60 points/axis (was 61 on two axes) and export point count capped
    at 729 (`<=10,000`, both asserted at import time, iterative CG-only
    solve, as before).
  - `examples/showcase_gradient_dispersion_sr87.yaml`: `integration.dtau`/
    `steps` pinned explicitly (never `integration.time_s`/auto-selected
    `dtau`); `ensemble.size` set to 100, chosen from a direct,
    fresh-subprocess peak-RSS measurement at several sizes (40/60/80/100
    atoms -> 2.28/3.13/3.98/4.68 GB), not a byte-counting estimate.
  - `paper/figures/fig4_showcase_gradient_dispersion.py`: asserts a
    trajectory-storage bound (`<1 GB`) and a field-evaluation
    query-fit-pair bound (`<30e6`, empirically calibrated with ~1.3x
    headroom below the scenario's own 23.18e6) before running anything
    expensive, so a future parameter change reintroducing this failure
    mode fails loudly and immediately instead of silently allocating
    unbounded memory.
  - `tests/test_e2e.py::test_showcase_scenario_config_stays_within_memory_safety_bounds`
    pins the same compute-budget bound directly against the shipped
    config.

### Changed

- **Package rename** (2026-08-10): distribution `dg0-pathsolve` ->
  `cliffordclock`, Python module `dg0_pathsolve` -> `cliffordclock`, CLI
  command `dg0` -> `cliffordclock` (`cliffordclock run config.yaml`,
  `cliffordclock version`). Version stays `0.1.0.dev0`; **no compatibility
  shim** -- there are no external users yet, which is the point of doing
  this now, before any beta distribution. Every import, doc command,
  notebook cell, and provenance string (the `importlib.metadata` lookups
  in `cliffordclock.__init__` and
  `cliffordclock.analytics.report._package_version`) now use the new
  name; all four notebooks re-executed (notebook 01 now carries committed
  execution outputs for the first time, deliberately, so all four ship
  with verified outputs); paper figures/PDF regenerated from the renamed
  pipeline.

### Added

- **Pre-flight trajectory-memory guard** (2026-08-11): the time-stepping
  modes (`integration.mode: direct`/`worldline`) estimate the dense
  `(M, steps + 1, 3)` float64 trajectory allocation *before* propagating
  anything (`4 x M x (steps+1) x 3 x 8` bytes, the documented multiplier
  covering the Verlet velocity trajectory and same-shape intermediates)
  and reject configs whose estimate exceeds the new optional
  `integration.max_trajectory_memory_gb` config key (default 2 GB) with
  an actionable `PipelineConfigError` (reduce `ensemble.size`, use an
  explicit coarser `integration.dtau`, or switch to the
  `secular`/`fast_path` modes, which are O(1) in `integration.time_s`).
  Previously an auto-selected E31 `dtau` (trap-period resolution)
  combined with a long `integration.time_s` and a large ensemble could
  silently attempt a 100+ GB allocation and lock up the host.
  `coupling.type: stark_dc` + `mode: direct` against a
  `FieldSmoother`-backed field (`field.csv`/`field.comsol`) gets an
  additional term for that accumulator's whole-trajectory `rate_fn` call
  against the smoother's `(N, K, 3)` evaluation intermediates (`N = M x
  steps`, `K` fit points) -- calibrated from measured peak RSS (a
  guard-passing-but-lethal case before this addition: base term ~1.9 GB,
  real allocation ~4 TB). `integration.mode: secular` gets the base
  trajectory check too (keyed on the otherwise-unbounded
  `integration.points_per_period` instead of a resolved `steps`), applied
  at the pipeline call site since `fastpath.py` is out of scope for this
  guard. The chunked-evaluation fix that would bound the smoother-backed
  memory cost directly (rather than just estimating and rejecting it) is
  deferred as follow-up work. See `docs/timescales.md` ("Safety net: the
  trajectory-memory guard") and `docs/cli.md`.

- **COMSOL "Spreadsheet"-format field ingestion** (2026-08-10):
  `cliffordclock.fields.load_field_comsol` parses COMSOL's native
  `File > Export > Data` "Spreadsheet" export format (`%`-prefixed
  metadata + column-header block, whitespace- or comma-delimited data
  rows) directly into a `FieldGrid`, reusing `load_field_csv`'s full
  validation stack (finite values, exact/near-duplicate points). Scope is
  the documented common case only, per this project's no-silent-partial-
  parsing rule: a single, non-parameterized, 3D, real-valued export with
  `es.Ex`/`es.Ey`/`es.Ez` (or another `expression_prefix`) field columns
  in m/mm/cm and V/m/kV/m/V-per-cm. Out-of-scope or malformed input
  (COMSOL's "Sectionwise" format, a 2D export, a parameter-sweep `@` tag,
  a complex-valued cell, an unsupported unit, a `Nodes`/`Expressions`
  count mismatch, a truncated header, ...) is rejected with a specific,
  descriptive `ValueError` rather than silently mis-parsed -- verified by
  `tests/test_fields_comsol.py`'s adversarial suite, plus a real,
  independently-sourced COMSOL header excerpt (Zenodo record 3763035,
  Stocchi/Mencarelli/Pierantoni, IEEE Microwave Magazine, DOI
  10.1109/MMM.2018.2821086, CC-BY 4.0) confirming the parser handles a
  genuine export's header correctly and rejects it for the right reason
  (2D, not the swept-study `@` tag it also happens to carry).
  `examples/generate_fd_electrode_field.py` is the end-to-end proof: a
  from-scratch finite-difference Laplace solve (numpy + scipy.sparse
  only, no new dependency) of a grounded box with two +/-2 V electrode
  plates, written out through an independent small COMSOL-format writer
  (`examples/fd_electrode_field.txt`, committed, 15^3 points) and loaded
  back through `load_field_comsol` -- `E_z` at the domain midplane lands
  0.5% off the parallel-plate estimate `V/d`, with the finite-difference
  method's staircase-boundary limitation documented in the script's
  docstring.

- **`field.comsol` config/CLI wiring** (2026-08-10, completes the
  ingestion item deferred above): `field: {comsol: path}` is now a first-class
  `config.yaml` field source, on equal footing with `field: {csv: path}`
  -- `cliffordclock.pipeline.PipelineConfig` accepts exactly one of
  `csv`/`comsol`/`synthetic` (mutually exclusive, a clear
  `PipelineConfigError` otherwise), loads the export via
  `load_field_comsol`, and fits a `FieldSmoother` from it the same way
  `csv` does (`field.smoothing` applies to both; the optional
  `field.expression_prefix`, default `"es"`, is forwarded to
  `load_field_comsol`). The comsol path and expression prefix flow into
  the report's `config_hash` provenance exactly as `field.csv` already
  does. New example `examples/comsol_electrode_sr87.yaml` runs the
  committed `examples/fd_electrode_field.txt` two-electrode
  finite-difference export through `coupling.type: stark_dc` on an Sr-87
  lattice ensemble (E29 fast path, genuine 1 s interrogation); its
  reported shift (~-1.109e-14) matches a documented back-of-envelope
  `k_S|E|^2/nu_0` estimate at the field's own domain-center value
  (1243.7652 V/m) to within the 1% band `tests/test_e2e.py` checks.
  `docs/fields.md`, `docs/byof-guide.md`, and `docs/cli.md`'s "Field
  sources" section drop the "not yet wired" caveat and document the
  `field.comsol`/`field.expression_prefix` config keys directly;
  `docs/tutorial.md` gains a short "Loading a COMSOL export" section
  built on the new example.

- **NIST/JILA public-dataset benchmark** (2026-08-10): a new top-level
  `benchmarks/` tree (not part of the installed package --
  `pyproject.toml`'s `[tool.setuptools.packages.find] where = ["src"]`
  already excludes it, unchanged) ingesting the two sources authorized
  for this benchmark (2026-08-10): arXiv:2403.10664 (JILA 1D Sr-87
  optical lattice clock, "A clock with 8x10^-19 systematic uncertainty")
  and data.nist.gov DOI 10.18434/M32206 (optical-clock down-conversion
  phase/Allan-deviation data). **Headline finding (reported as found, not
  massaged):** zero rows from either source support an independent
  predicted-vs-published residual under this engine's current physics
  scope (CONVENTIONS.md E14b scalar DC Stark + E21 second-order Doppler
  only). 8 of JILA's 9 systematic-shift line items (BBR, lattice light,
  Zeeman, density, background gas, tunneling, minor shifts) are physics
  entirely outside this engine's scope; the 9th (DC Stark) is in scope
  but the paper reports only the resulting shift (`-9.8+/-0.7e-20`),
  never the residual stray-field magnitude that produced it, so no
  forward config can be built without either an unsourced guess or a
  forbidden tuned parameter. The NIST dataset is a different measurement
  category entirely (phase/frequency-division instability, not a
  systematic-shift/field-gradient measurement) and maps to nothing. Full
  reasoning, citations, and gap analysis: `benchmarks/MAPPING.md`,
  `benchmarks/RESULTS.md`. This project's binding evaluation rules treat
  a documented "cannot compare" as a successful, correctly-labeled
  outcome, not a shortfall.
  - `benchmarks/SOURCES.md`: URLs, DOIs, retrieval date (2026-08-10),
    SHA-256 checksums (cross-verified against NIST's own published
    checksums), and license/redistribution terms for every fetched
    file. The JILA PDF/TeX source is **not committed** (arXiv's default
    perpetual-non-exclusive submission license does not grant
    third-party redistribution) -- `benchmarks/fetch_data.py` re-fetches
    and checksum-verifies it on demand. The NIST CSVs are public domain
    (17 U.S.C. Section 105) but only a 20-row excerpt of each is
    committed (not a license restriction -- the full ~2.2 MB files are
    demonstrated not to map to any comparison, see below), with
    `fetch_data.py` covering the full files too.
  - `benchmarks/MAPPING.md`: row-by-row (JILA Table I) and file-by-file
    (NIST) scope classification and citation, including the full
    reasoning for why the one in-scope JILA row (DC Stark) still cannot
    be turned into an independent forward-comparable case without
    violating the "no tuned parameters" labeling rule.
  - `benchmarks/loaders.py`: typed parsers -- `load_jila_table1` (the
    hand-transcribed Table I fixture, since arXiv hosts no
    machine-readable ancillary data for this paper) and
    `load_nist_phase_csv` (the real, genuinely machine-readable NIST
    time-series format).
  - `benchmarks/run_benchmarks.py`: the benchmark runner script (not a
    CI unit test). Classifies every JILA/NIST row (`classify_jila_table1`,
    `classify_nist_series`), and runs a real, non-tuned illustrative
    sweep of the `coupling.type: stark_dc` pipeline (Sr87, lattice fast
    path, 1 s interrogation, the same KA1 machinery) over fixed
    round-number stray-field magnitudes for physical context --
    explicitly typed/labeled so it can never be mistaken for a
    residual/KPI case. Emits `benchmarks/results/wp10_results.json` + a
    generated markdown table.
  - `benchmarks/fetch_data.py`: re-downloads and SHA-256-verifies the
    full JILA PDF/source and NIST CSVs on demand.
  - `benchmarks/fixtures/`: small committed excerpts -- the full JILA
    Table I transcription (10 rows) and 20-row excerpts of both NIST
    CSVs.
  - `benchmarks/results/wp10_results.json` + `wp10_results_table.md`:
    committed script output (KPI verdict: 0 pass / 0 fail / 12
    not-applicable, of 12 rows considered).
  - `benchmarks/RESULTS.md`: the full write-up for project review --
    per-case table, gap analysis (which physics is missing and where
    it's tracked), KPI verdict, and explicit open-question/deviation
    notes.
  - Test contract: `tests/test_benchmarks_loaders.py` (17 tests) --
    parsing correctness against the committed fixtures (including a
    BBR-row and DC-Stark-row verbatim pin against the paper, guarding
    against a transcription-digit-swap), config-mapping/classification
    unit tests (pinning that every JILA/NIST row is classified
    not-comparable, so a future edit can't silently flip one without
    this test forcing a conscious review), and a runner smoke test that
    exercises the real pipeline end-to-end for the illustrative
    DC-Stark sweep (monotonicity and exact quadratic-in-field scaling
    checks, `rtol=1e-10`).
- **NPL reproducibility case + USTC second budget-only source**
  (follow-up, 2026-08-10): two more sources authorized for this
  benchmark, upgrading the benchmark story for the first time from "no
  comparison possible" to "one genuine, correctly-labeled
  reproducibility success against public data."
  - **Task A -- NPL Rydberg electrometry** (arXiv:1706.01944, Bowden et
    al., PRA 96, 023419 (2017)): independently re-verified from the
    fetched full text (not trusted from the research dossier): NPL
    measured a residual stray field at their Sr atoms **independently
    of the clock transition** (Rydberg-state EIT spectroscopy) -- `E =
    1.52 (+0.62/-0.22 stat, +0.05/-0.03 sys) V/m` -- and quote a
    resulting DC-Stark shift of `-1.6 (+0.4/-1.6) x 10^-20`, converted
    using the same Middelmann et al. (PRL 109, 263004 (2012)) `Δα` this
    project's `Sr87` species registry already cites (cross-checked:
    this project's own pipeline reproduces the paper's own "570 V/m →
    1 Hz" intro sanity-check example to 5 significant figures). Built
    the **first genuine forward-comparable case in this benchmark's
    history**: `benchmarks/loaders.py` gains `AsymmetricMeasurement`
    (per-side stat+sys quadrature combination, no Gaussian pretense on
    asymmetric errors) and `PublishedBand`; `benchmarks/run_benchmarks.py`
    gains `run_npl_reproducibility_case`, which runs the real
    `coupling.type: stark_dc` pipeline three times (at the field's
    combined low/nominal/high bounds) and compares the resulting
    predicted band `[-3.290, -1.208]×10⁻²⁰` against NPL's published band
    `[-3.2, -1.2]×10⁻²⁰` via a precisely-defined, unit-tested
    closed-interval overlap test (`_bands_overlap`). **Result: bands
    overlap, `kpi_verdict = "MET"`.** **Binding classification label:** this is
    a *"reproducibility"* case (`case_class = "reproducibility"`),
    explicitly **not** a blind prediction or "validation against an
    independent measurement of the shift" -- NPL themselves already
    combined the same two published ingredients this engine's pipeline
    combines; the case demonstrates end-to-end pipeline correctness
    against independently published inputs/outputs with no fitting, not
    independent predictive power. Kept structurally distinct from a
    (still entirely empty) `"blind_prediction"` category throughout
    `benchmarks/run_benchmarks.py`'s report schema (bumped to `"2.0"`),
    `benchmarks/RESULTS.md`, and `benchmarks/MAPPING.md`.
  - **Task B -- USTC Sr1 evaluation** (Jia et al., Metrologia 63, 025002
    (2026), CC BY 4.0, owner-provided PDF): own independent read of
    Section 3.5 "Residual DC Stark shift" and Table 3 (not trusted from
    the dossier) -- publishes a DC-Stark **budget constraint**
    (`0.0(0.1) x 10^-19` total), derived from a prior y-component shift
    measurement (`1.4(5.2) x 10^-21`, cited to their own reference [30])
    and geometric/shielding-factor scaling arguments (viewport
    distances 142/237 mm, an 8x geometric factor, a 3x FE-simulated
    shielding factor) -- same structural class as JILA's DC-Stark row:
    in scope, no independent field magnitude in *this* paper.
    `benchmarks/loaders.py` gains `USTC_DC_STARK_CONSTRAINT`;
    `benchmarks/run_benchmarks.py` gains `classify_ustc_dc_stark`
    (`comparable=False`, `kpi_verdict="N/A"`, same shape as every JILA
    row). Reference [30] (**Li J et al 2024, Metrologia 61, 015006**) is
    flagged in `benchmarks/MAPPING.md`/`benchmarks/RESULTS.md` as the
    clearest lead for a *second* reproducibility case -- explicitly
    **not authorized, not fetched, not examined** by this session.
  - **Provenance and licensing:** `benchmarks/SOURCES.md` gains
    sections 4-5 with full checksums/access logs. NPL's arXiv PDF/TeX
    source: not committed (same non-exclusive-license reasoning as the
    JILA/PRL sources), `benchmarks/fetch_data.py` extended to re-fetch
    and verify both files. USTC's PDF: CC BY 4.0 (confirmed directly
    from the PDF's own copyright block, not assumed) -- permits
    committing extracted excerpts freely with attribution; a verbatim,
    attributed excerpt of Section 3.5 + Table 3 is committed
    (`benchmarks/fixtures/ustc_metrologia_63_025002_sec3_5_table3_excerpt.txt`),
    the full 3.8 MB PDF itself is not (owner-local file; its SHA-256 is
    recorded as the provenance record instead).
  - **Test contract**: `tests/test_benchmarks_loaders.py` grows from 17
    to 39 tests -- NPL/USTC data-structure parsing/citation pins,
    `AsymmetricMeasurement`/`PublishedBand` correctness (including the
    per-side-quadrature, never-symmetrized combination method),
    `_bands_overlap`'s precise interval-overlap definition (11
    parametrized cases: identical, partial, containing, touching,
    disjoint, and the actual NPL/JILA-scale negative bands), the
    reproducibility case's core assertion (bands overlap,
    `kpi_verdict == "MET"`, registry-sourced `Δα`, field bounds traced
    to `AsymmetricMeasurement.combined_lo/hi`), and
    `classify_ustc_dc_stark`. `test_build_report_smoke`/
    `test_render_markdown_table_smoke` updated for the new report
    schema (`reproducibility_cases_total`/`_met`,
    `blind_prediction_cases_total`/`_met`, `not_applicable_rows`
    replacing the old pass/fail/not-applicable triad -- this benchmark
    has never used "PASS"/"FAIL" for any case, and now distinguishes
    "N/A" budget-only rows from "MET"/"NOT MET" reproducibility
    verdicts).
  - **KPI verdict, updated:** 1/1 reproducibility case met, 0/0
    blind-prediction cases, 13 rows not-applicable (of 14 rows
    considered) -- up from the prior follow-up's 0/0/12.
- **Realistic worked example + "bring your own field" guide**
  (2026-08-10): the demo a lab postdoc adapts in 30 minutes -- a
  physically-shaped stray-field scenario through the full pipeline
  (real CSV field import, `coupling.type: stark_dc`, a genuine 1 s
  lattice interrogation), plus the guide for swapping in a reader's own
  FEA export.
  - `examples/generate_patch_field.py`: a documented, seeded generator
    for a physically plausible optical-lattice stray-field scenario --
    a small uniform residual bias field plus six Gaussian
    patch-potential contributions (10-100 mV, J.B. Camp, T.W. Darling,
    R.E. Brown, "Macroscopic variations of surface potentials of
    conductors", J. Appl. Phys. 69, 7126 (1991)) on window/electrode
    surfaces ~2 mm from an Sr-87 trap, evaluated on a regular
    17x17x17 (4913-point) grid and written in the `docs/fields.md` CSV
    contract. Generator determinism (seeded run reproduces the
    committed CSV byte-identically) is pinned by
    `tests/test_e2e.py::test_wp11_generate_patch_field_is_deterministic`.
  - `examples/patch_field_sr87.csv`: the committed generated field
    (4913 points, ~460 KB).
  - `examples/realistic_lattice_sr87.yaml`: Sr87, `coupling.type:
    stark_dc` (E14b), `ensemble.regime: lattice`, a genuine `T = 1 s`
    interrogation, loading the CSV above through
    `cliffordclock.fields.FieldSmoother` (not a closed-form
    `field.synthetic` factory, unlike every other shipped example) --
    the config a reader adapts by swapping the `field.csv` path for
    their own export. Runs in ~2.4 s wall time, well under the 60 s CPU
    bound. **Headline numbers** (back-of-envelope cross-check vs. the
    full pipeline, both derived from the Sr87 DC-Stark coefficient
    `docs/validation.md` KA1 already validates): `|E(trap center)| ~=
    18.83 V/m` gives a back-of-envelope `Delta_nu/nu0 ~= -2.542e-18`;
    the full pipeline reports `mean_fractional_shift = -2.541762e-18
    +/- 1.079e-22 (SEM)`, `t2_star_s = 2.856964 s` -- inside the
    required [1e-19, 1e-17] demonstration range, and SEM ~4 orders of
    magnitude below the shift itself.
  - `notebooks/04_bring_your_own_field.ipynb`: the adaptation template
    -- load a CSV, inspect/smooth it (field + gradient plots along
    lines through the trap), choose species/motional state, run,
    interpret the report (including a live demonstration of
    `NearDuplicatePointsWarning`/`IllConditionedFitWarning` and the
    `smoothing`-parameter fix). Executed in CI
    (`.github/workflows/ci.yml`'s `notebook` job).
  - `docs/byof-guide.md`: the "bring your own field" guide -- CSV
    format contract, units, grid-spacing-vs-feature-size guidance,
    `smoothing` guidance for noisy exports, the three field-smoother
    warnings explained, and current limitations stated plainly (CSV
    only, scalar DC-Stark + second-order Doppler physics only,
    `Sr87`/`Yb171` only, E29 fast-path Stark-only scope). Linked from
    `README.md` and `docs/index.md`.
  - Test contract: `tests/test_e2e.py` gains
    `test_wp11_generate_patch_field_is_deterministic`,
    `test_wp11_realistic_lattice_sr87_shift_in_expected_range` (shift
    range, report/CSV validity, `run_pipeline_full` API), and
    `test_wp11_cli_smoke_realistic_lattice_sr87` (the exact `cliffordclock run`
    command the guide documents).
- **Known-answer validation against numbers the optical-clock community
  already knows** (2026-08-09), plus the `coupling: {type: stark_dc}`
  pipeline/CLI plumbing -- see `docs/validation.md` and
  `docs/coupling.md`.
  - `coupling.type: stark_dc` pipeline/CLI wiring (completes the
    coupling-abstraction test contract's final item): `config.yaml`'s
    `coupling:` block gains a `type` field (`linear_mu` | `stark_dc`,
    default `linear_mu` when omitted -- every existing config keeps
    working unchanged) plus optional
    `delta_alpha_dc_si`/`stark_coefficient_hz_per_v2_m2` override
    fields. `stark_dc` (the physical E14b DC-Stark coupling) works in
    every `integration.mode` (`fast_path`, `direct`, `secular`,
    `worldline`) via the coupling-agnostic `fastpath.RateFn` seam;
    `direct`/`worldline` run through a new scalar-only phase
    accumulator (`cliffordclock.pipeline._stark_scalar_ensemble`)
    rather than the E14a rotor path (E14b has no `Ω`-bivector
    construction in this codebase yet). The report's
    `uncertainty_notes` records the resolved coefficient and its
    provenance (species-registry citation, or explicit override), and
    `fast_path` runs additionally note the motional-second-order-Doppler
    exclusion (E29 scope). New example: `examples/lattice_sr87_stark.yaml`
    (Sr-87, a lab-typical 100 V/m stray field, genuine 1 s
    interrogation).
  - Known-answer test suite (`tests/test_known_answers.py`): KA1/KA2
    (Sr87/Yb171 uniform-field DC-Stark, matching the textbook
    `Delta_nu = -(1/2)Delta_alpha|E|^2/h` formula to rtol 1e-10 --
    exact to float64 in practice), KA3 (Yb171 linear-gradient field,
    mean shift and phase spread matching an independent Gaussian-moment
    perturbation-theory hand calculation to rtol 1e-8,
    `tests/reference_impl.py` extended with
    `stark_shift_mean_and_variance`), KA4 (Sr87 second-order Doppler
    shift, classical thermal ensemble, matching the equipartition
    prediction `-3k_BT/(2mc^2)` within 5 SEM -- measured 0.32 SEM).
  - `docs/validation.md`: every validated case (the original V1-V4
    cases plus this milestone's KA1-4), with formulas, literature
    sources, and measured agreement -- linked from the README as the
    entry point for a skeptical reviewer.
  - `notebooks/03_known_answers.ipynb`: a narrated walkthrough of
    KA1-4 (what field, what state, what the textbook predicts, what
    the engine returns), executed in CI.
- **Three-tier fast-path architecture** (CONVENTIONS.md section 12,
  E29-E31), replacing the original "large-`dtau` is unexplored
  headroom" caveat entirely -- see `docs/timescales.md` for the full
  explanation and accuracy study.
  - Lattice fast path (E29,
    `cliffordclock.integrator.fastpath.lattice_shift_expectation`): the
    default execution path for `ensemble.regime: lattice` configs.
    Static motional-state quadrature nodes need no time integration at
    all -- `ΔΦ_q = δω̃_q · T̃` exactly, at O(1) cost in the
    interrogation time. Matches the worldline (rotor) integrator
    exactly (rtol 1e-12) on identical static nodes, still available via
    `integration.mode: worldline`. `examples/lattice_sr87.yaml` now
    reports a genuine **1-second interrogation** (well under a second
    of wall time).
  - `select_dtau` (E31): automatic step-size selection
    (`cliffordclock.integrator.fastpath.select_dtau`) resolving the
    trap period, not the Compton period (`dτ̃ = T_orb / (100 τ_c)` by
    default). Used automatically by `ensemble.regime: classical`
    configs (`integration.mode: direct`, the default) when
    `integration.dtau` is omitted. A large-`dτ̃` accuracy study
    (`notebooks/02_step_size_study.ipynb`, `docs/timescales.md`)
    confirms the expected order-2 error scaling and validates the
    default 100-points-per-period resolution against the
    CONVENTIONS.md V4 closed form.
  - Secular averaging (E30,
    `cliffordclock.integrator.fastpath.secular_average_shift`): a
    one-orbit average for classical periodic motion in an isotropic
    harmonic trap (`integration.mode: secular`), validated against
    large-step direct integration on the same harmonic case.
  - `integration.time_s`: a direct, real-seconds interrogation-time
    config input (`integration.dtau`/`steps` remain fully supported,
    unchanged, for explicit/backward-compatible configs).

### Changed

- **PRL Supplemental Material re-check** (follow-up, 2026-08-10): one
  more source authorized for this benchmark -- the Supplemental
  Material of the *published* version of the JILA paper (PRL 133,
  023401), specifically to re-check whether it publishes a residual
  stray-field magnitude (V/m) the DC-Stark row still lacked. **Outcome
  unchanged.** The file itself is paywalled ("Subscription Required" on
  `journals.aps.org`; the one public route, APS's CHORUS
  accepted-manuscript link, could not be downloaded through any
  available tool) and was not retrieved -- no credential entry or
  paywall bypass was attempted, per this project's binding safety
  rules. In its place: a specific, citable cross-check against the
  already-fetched, checksummed arXiv v2 e-print (whose LaTeX source
  bundle contains separate `main.tex`/`supplementary.tex` files
  compiled into the one PDF already on hand) shows the Supplemental
  Material's five sections (3D1/3P1 lifetime uncertainty, temperature
  measurement, dynamic BBR shift, first-order Zeeman, background gas)
  do not include DC Stark, and that the paper's DC-Stark paragraph
  cites no supplemental reference at all. `benchmarks/SOURCES.md`
  section 3 (new) records the full access-attempt log;
  `benchmarks/MAPPING.md`'s "third source" note is updated from
  "uninvestigated" (a review-added transparency flag) to this precise,
  evidenced conclusion; `benchmarks/RESULTS.md` gains a "Follow-up:
  PRL Supplemental Material" section; `benchmarks/loaders.py`,
  `benchmarks/run_benchmarks.py` (the DC-Stark not-comparable reason
  string and the report's headline finding), and the committed
  `benchmarks/results/wp10_results.json`/`wp10_results_table.md` are
  updated to reference it. KPI verdict is unchanged: 0 pass / 0 fail /
  12 not-applicable.
- **Li J et al 2024 (Metrologia 61, 015006) access blocked**
  (follow-up, 2026-08-10): fetching the USTC DC-Stark row's reference
  [30] (Li J et al 2024, "A strontium lattice clock with both
  stability and uncertainty below 5×10⁻¹⁸," Metrologia 61, 015006) was
  authorized for this benchmark specifically to check whether it
  enables a **blind-prediction-grade** case (a new capability tier,
  distinct from the NPL reproducibility case: a field this engine's
  inputs did not already combine with the measurement being
  predicted). **Outcome: the paper could not be retrieved by any
  route, and its content was not examined.** Unlike the JILA PRL
  Supplemental Material follow-up, no substitute text was available to
  cross-check against -- no arXiv preprint exists for this paper
  (confirmed via four independent search strategies), IOPscience shows
  it as explicitly not open access ("Login / Purchase / Rent" only, no
  accepted-manuscript or CHORUS-style link on the page), and
  ResearchGate returned HTTP 403. No credential entry or paywall
  bypass was attempted. Per the benchmark protocol ("if gated
  everywhere, report precisely and stop"), the attempt stops here.
  **Classification (binding): "not accessed -- classification not
  possible without a copy"** -- reported as its own outcome,
  deliberately distinct from `"reproducibility"` and `"not_applicable"`
  (the latter would misrepresent an access failure as a content
  finding). No code changes were made
  (`benchmarks/loaders.py`/`benchmarks/run_benchmarks.py` untouched
  for this source -- there is nothing to load or classify).
  `benchmarks/SOURCES.md` gains section 6 (full access-attempt log);
  `benchmarks/MAPPING.md`'s Source 5 "next authorization candidate"
  note is updated to reflect the attempt and its outcome;
  `benchmarks/RESULTS.md` gains a "Follow-up 2" section and an open
  question note; `docs/validation.md`'s headline is updated to mention
  six sources and this outcome. **KPI verdict: unchanged** -- 1/1
  reproducibility case met, 0/0 blind-prediction cases, 13 rows
  not-applicable (of 14 rows considered). This attempt neither added a
  case nor changed any existing classification; it documents due
  diligence on an authorized fetch that did not succeed. The owner may
  supply the PDF directly (as done for the USTC 2026 paper) to unblock
  this in a future pass.
- **Patch-field scenario realism** (2026-08-10 reframe): a pre-release
  review noted that the original 2 mm patch standoff reads as
  ion-trap, not lattice-clock, geometry. A literature review confirmed
  that real charge-bearing surfaces in neutral-atom lattice clocks are
  cm-scale (mirrors, viewports, shields), not mm-scale.
  `examples/generate_patch_field.py` is reframed accordingly, same
  file paths and seeded-determinism design:
  - Scenario: six Gaussian patch-potential contributions from residual
    trapped charge on **in-vacuum dielectric surfaces** (mirror/viewport
    faces), `PATCH_DISTANCE_M = 25 mm` (was 2 mm), `PATCH_WIDTH_M =
    8 mm` (was 1 mm; widened because the Gaussian-bump field falloff
    otherwise suppresses the field to nothing at the new cm-scale
    standoff -- see the generator's module docstring "Tuning
    reasoning"), amplitudes drawn Uniform(0.5, 5.0) V (was 10-100 mV)
    -- Lodewyck-bracketed partially-discharged trapped-charge
    potentials, not the Camp/Darling/Brown 1991 conductor-patch mV
    range (the dossier could not re-verify that paper's specific
    "10-100 mV" figure at the cited precision; it is now kept only as
    general patch-potential background, not this scenario's amplitude
    source).
  - Primary citation: Lodewyck, Zawada, Lorini, Gurov, Lemonde, IEEE
    Trans. UFFC 59, 411 (2012), arXiv:1108.4320 -- the real SYRTE
    Sr-clock event (dielectric cavity-mirror charge, cm-scale, 3.4
    kV/m pre-mitigation / 1e-13 shift, UV-discharged to 1e-18). This
    scenario models the intermediate, partially-discharged regime
    between those endpoints. Secondary context: Beloy et al., PRL 120,
    183201 (2018) (Faraday shield, cm-scale windows); NPL,
    arXiv:2005.10857 (chamber-as-shield).
  - Regenerated `examples/patch_field_sr87.csv` (same seed `20260810`,
    same 17x17x17/4913-point grid, ~460 KB, generator still runs in
    well under 60 s CLI wall time). **New headline numbers**:
    `|E(trap center)| ~= 10.38 V/m` (was 18.83 V/m); full pipeline
    reports `mean_fractional_shift = -7.723399e-19 +/- 6.771e-24
    (SEM)` (was -2.541762e-18 +/- 1.079e-22), `t2_star_s = 45.528361 s`
    (was 2.856964 s) -- still inside the required [1e-19, 1e-17]
    demonstration range (now near its geometric center), and still ~4
    orders of magnitude above its own SEM. The `[1e-19, 1e-17]` range
    assertion itself, and the generator-determinism/CLI-smoke test
    contract, are unchanged -- `tests/test_e2e.py`'s relevant tests
    needed no code changes, only the regenerated CSV.
  - Docs updated: `examples/generate_patch_field.py` module/function
    docstrings, `examples/realistic_lattice_sr87.yaml` scenario
    comment, `docs/byof-guide.md` (grid-oversampling figure),
    `README.md` (headline numbers),
    `notebooks/04_bring_your_own_field.ipynb` (scenario narrative
    cell; computed cells re-executed against the regenerated CSV).
- **`docs/cli.md`, `README.md`**: the "conservative small `dtau`,
  femtosecond windows" framing is replaced by `docs/timescales.md`'s
  three-tier architecture and measured accuracy study.
- **`cl13.exp_bivector` large-angle hardening**: the rotor exponential
  now applies an exact 2π range reduction to the compact
  (rotation-like) invariant component before the E6 scaled-Taylor
  evaluation, replacing silent degradation at large generator angles
  (norm error 1.9e-7 at ~1e3 rad, finite garbage at ~5e3, NaN from
  ~1e4) with ≲1e-12 accuracy up to 1e4 rad and ≲1e-9 up to 1e6 rad
  (documented accuracy contract in the docstring). Inputs with compact
  angle ≤ π -- everything the original test suite exercised -- produce
  bitwise-identical results. jit/vmap/grad-safe; gradients stay finite
  at the reduction's singular points (zero, null, pure-boost
  bivectors).

### Fixed

- Known issue, not yet fixed:
  `cliffordclock.analytics.stats.dephasing_time_t2star` raises an
  unhandled `ZeroDivisionError` for any ensemble whose weighted phase
  variance is exactly zero (e.g. a spatially uniform field over more
  than one lattice quadrature node, where every node reports an
  identical shift by construction) -- a pre-existing edge case, worked
  around in `examples/lattice_sr87_stark.yaml`/KA1/KA2 by using a
  single static node instead of triggering it.

## 0.1.0.dev0 — initial release

Initial implementation: a validated, end-to-end synthetic pipeline. See
`docs/CONVENTIONS.md` (v1.0.0) for the physics this release implements.

### Added

- **Cl(1,3) algebra kernel** (`cliffordclock.cl13`): 16-component
  multivector representation, geometric product via a precomputed
  structure tensor, reverse, grade projection, rotor exponential/
  normalization (E1-E6). Cross-checked against the third-party
  pygae/`clifford` PyPI package as an independent test-oracle reference.
- **Field importer and smoother** (`cliffordclock.fields`): CSV field-grid
  ingestion, a degree-1 analytical baseline + thin-plate-spline RBF
  residual smoother with autodiff gradients (E11-E13), and closed-form
  synthetic test fields (`fields.synthetic`) with hand-derived exact
  gradients.
- **Rotor path integrator** (`cliffordclock.integrator`): the interaction
  bivector `Ω` (E14a, E16, E18), an exponential-midpoint rotor stepper
  (E17, E19), and `lax.scan`/`vmap`-based worldline/ensemble integration
  with compensated (Kahan) phase accumulation (E9, E20-E24).
- **Ensemble samplers** (`cliffordclock.ensemble`): Maxwell-Boltzmann
  Monte-Carlo sampling + velocity-Verlet propagation for the classical
  (ion-trap-style) regime, and Hermite-Gauss motional-state quadrature for
  the optical-lattice regime, plus a three-species registry (Sr-87,
  Yb-171, Al-27+).
- **Metrology analytics** (`cliffordclock.analytics`): weighted mean
  fractional shift and standard error, inhomogeneous dephasing time T2*,
  coherence function and spectral line profile (E23, E25-E28), and a
  schema-versioned `MetrologyReport` JSON + line-profile CSV writer.
- **Pipeline façade and CLI** (`cliffordclock.pipeline`,
  `cliffordclock.cli`): `run_pipeline`/`run_pipeline_full`, a declarative
  YAML `config.yaml` schema, and the `cliffordclock run`/`cliffordclock version` commands
  (see `docs/cli.md`).
- **Demo notebook**
  (`notebooks/01_end_to_end_demo.ipynb`): a plotted, narrated walkthrough
  of the full pipeline, executed in CI via `jupyter nbconvert --execute`.
- **End-to-end validation** (`tests/test_e2e.py`,
  `tests/reference_impl.py`): four analytical validation cases -- a null
  result at the 1e-19 level (V1), a constant-gradient closed-form match to
  1e-12 relative (V2), an independent plain-NumPy cross-check of the
  phase-accumulation pipeline to 1e-10 relative, and a CLI smoke test.
- Two runnable example configs (`examples/quadrupole_classical.yaml`,
  `examples/lattice_sr87.yaml`), each completing in well under a minute on
  CPU.

### Known limitations (initial release)

- No experimental data ingestion or benchmarking against published
  clock-shift measurements yet (later milestone).
- Only the linear, explicit-`mu` E14a pivot coupling is implemented; the
  physical quadratic DC-Stark coupling (E14b) is specified but not yet
  implemented.
- The shipped example configs used a conservative, small `dtau` for fast
  CI runs, limiting them to femtosecond-scale simulated interrogation
  windows. **Resolved in a later milestone** -- see the "Unreleased"
  section above and `docs/timescales.md`: real (microsecond-to-second)
  interrogation times are now the norm, backed by a systematic
  large-`dtau` accuracy study.
