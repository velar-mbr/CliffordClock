# CLI and pipeline configuration

`cliffordclock` is CliffordClock's command-line entry point
(`cliffordclock.cli`), a thin wrapper around the one-call pipeline façade
`cliffordclock.pipeline.run_pipeline`/`run_pipeline_full`. It runs
the full path (load/synthesize a field, build an ensemble, integrate the
rotor path equation, analyze the result) from a single `config.yaml`.

## Commands

```bash
cliffordclock version
cliffordclock run config.yaml [--output-dir DIR] [--radiation-surfaces PATH]
```

- `cliffordclock version` prints the installed package version.
- `cliffordclock run config.yaml` runs the pipeline and writes `report.json` +
  `line_profile.csv` into the config's `output.directory` (see schema
  below), printing a short human summary (species, ensemble, mean
  fractional shift ± SEM, T2*, and the two output paths).
- `--output-dir DIR` overrides `output.directory` from the config file
  without editing it.
- `--radiation-surfaces PATH` injects/overrides
  `environment.radiation_environment.surfaces_file` from `PATH` (a
  surfaces table file, see "Surfaces table file format" below), equivalent
  to the config file having had that key set. `PATH` is resolved relative
  to the CURRENT WORKING DIRECTORY (it came from the command line, not the
  config file). If the config already sets
  `environment.radiation_temperature_K` or an inline
  `environment.radiation_environment.surfaces` list, the normal E37
  mutual-exclusivity error fires instead of the flag silently overriding
  it.

**Exit codes:**

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | Physics-validation failure: the integrated result failed a basic sanity check (non-finite phase, or rotor-norm drift, E20, far beyond what a correctly configured run should show). Not a precision-grade check: see `cliffordclock.pipeline.PhysicsValidationError`. |
| `2` | Bad input: a malformed/unreadable config file, an unknown species or synthetic-field kind, an invalid parameter value, or a bad CLI argument (the last handled by `argparse` itself, which also exits 2). |

Errors are printed to stderr; the run summary goes to stdout.

## `config.yaml` schema

```yaml
species: Sr87                      # cliffordclock.ensemble.species registry name

trap:
  omega_xyz: [2.0e+5, 2.0e+5, 2.0e+6]  # rad/s
  center: [0.0, 0.0, 0.0]              # m, optional (default origin)

field:
  # exactly one of `csv`, `comsol`, or `synthetic`:
  csv: path/to/field_export.csv        # see docs/fields.md for the CSV format
  # comsol: path/to/comsol_export.txt  # COMSOL "Spreadsheet"-format export, see
  #                                     #   docs/fields.md / docs/byof-guide.md
  # expression_prefix: es              # optional, `comsol` only (default "es"): forwarded to
  #                                     #   load_field_comsol's expression_prefix
  # synthetic:
  #   kind: quadrupole                 # uniform | constant_gradient | quadrupole | gaussian_bump
  #   params: {k: 5.0e+6}              # forwarded to the cliffordclock.fields.synthetic factory
  smoothing: 0.0                       # optional, `csv`/`comsol` only: FieldSmoother.fit's
                                        #   Tikhonov term

coupling:
  type: linear_mu                      # optional (default "linear_mu"): linear_mu | stark_dc
  # linear_mu (E14a, the default when `type` is omitted; every existing
  # config that never wrote a `type` key keeps working unchanged):
  mu: [1.0e-25, -2.0e-25, 1.5e-25]     # C.m, the E14a effective dipole moment
  # stark_dc (E14b, the physically meaningful coupling; recommended for
  # new configs, see docs/coupling.md):
  # type: stark_dc
  # delta_alpha_dc_si: 4.07873e-39           # optional override, C^2 m^2 J^-1
  # stark_coefficient_hz_per_v2_m2: ...      # optional override, Hz.m^2.V^-2
  #                                           #   (alternative to delta_alpha_dc_si)

environment:                           # optional section (WP20, CONVENTIONS.md E32/E33);
                                        #   absent means BBR is off, byte-identical to a config
                                        #   with no environment section at all: see
                                        #   docs/coupling.md's "Blackbody-radiation shift" section
  radiation_temperature_K: 300.0       # kelvin; requires coupling.type=stark_dc; must lie in
                                        #   [50, 350] (hard PipelineConfigError outside it).
                                        #   Mutually exclusive with radiation_environment below.
  radiation_temperature_uncertainty_K: 0.004  # optional, kelvin; requires
                                        #   radiation_temperature_K to also be set
  # radiation_environment:               # optional, WP29 Tier 1, CONVENTIONS.md E37; a
  #                                       #   multi-surface alternative to radiation_temperature_K
  #                                       #   above (mutually exclusive with it). See the
  #                                       #   "Multi-surface thermal environment" section below.
  #   surfaces:                          # exactly one of `surfaces` (inline) or
  #                                       #   `surfaces_file` (a table file, see the
  #                                       #   "Surfaces table file format" section below)
  #     - name: shield                   # label only, used in error messages/report notes
  #       weight: 0.9                    # solid-angle fraction Omega_i/4pi; all surfaces'
  #                                       #   weights must sum to 1 (tolerance 1e-9)
  #       temperature_K: 100.0           # must lie in [50, 350], same window as above
  #       emissivity: 0.5                # optional, (0, 1]; at most one surface may set this
  #                                       #   (that surface is the reflective enclosure)
  #     - name: aperture
  #       weight: 0.1
  #       temperature_K: 300.0
  #       temperature_uncertainty_K: 0.01  # optional per-surface 1-sigma uncertainty, kelvin
  #   # surfaces_file: path/to/surfaces.txt  # equivalent to `surfaces` above, from a file
  #   correlated: false                  # optional (default false): per-surface temperature-
  #                                       #   uncertainty combination mode, see below
  gravity:                             # optional sub-section (WP22, CONVENTIONS.md section 15
                                        #   E36); absent means the gravitational-redshift term
                                        #   is off, byte-identical to a config with no gravity
                                        #   section at all. Requires coupling.type=stark_dc.
    g_m_s2: 9.80665                    # optional (default: STANDARD_GRAVITY, 9.80665 m/s^2,
                                        #   exact by definition); at the 1e-19 level use the
                                        #   LAB'S OWN SURVEYED LOCAL value instead (see the
                                        #   "Gravitational redshift" section below)
    up_axis: [0.0, 0.0, 1.0]           # optional (default [0,0,1]); need not be pre-normalized
    reference_height_m: 0.0            # optional (default 0.0): height (along up_axis, from the
                                        #   coordinate origin) where (P-1)_grav = 0

quadrupole:                            # optional section (WP21, CONVENTIONS.md E34/E35);
                                        #   absent means the quadrupole term is off,
                                        #   byte-identical to a config with no quadrupole
                                        #   section at all: see the "Quadrupole shift" section
                                        #   below. Requires coupling.type=stark_dc.
  state: "Ca+:D5/2"                    # a cliffordclock.ensemble.species.QUADRUPOLE_MOMENTS
                                        #   registry key, mutually exclusive with theta_au/j
  # theta_au: 1.83                     # explicit Theta(J) override, atomic units (= e*a0^2);
  # j: 2.5                             #   requires j to be given alongside it
  nu_0_hz: 411042129776393.0           # required: the clock transition frequency, Hz (the SAME
                                        #   transition whose upper state has this Theta; here,
                                        #   Ca+ 4S1/2-3D5/2, Chwalla et al., PRL 102, 023002 (2009))
  m_j: 2.5                             # required when averaging_mode="fixed"
  quantization_axis: [0.0, 0.0, 1.0]   # optional (default [0,0,1]); ignored when
                                        #   averaging_mode="three_orientation"
  averaging_mode: fixed                # optional (default "fixed"): fixed | three_orientation

ensemble:
  regime: classical                    # classical | lattice | lattice_extended
  temperature_uK: 1.0                  # microkelvin
  size: 200                            # classical only: Monte-Carlo particle count
  motional_n: [0, 0, 0]                # lattice/lattice_extended: motional quantum numbers
                                        #   (applied identically at every site for
                                        #   lattice_extended)
  n_quad: 8                            # lattice/lattice_extended: Gauss-Hermite points per
                                        #   axis, per site (default 8)
  seed: 0                              # classical only: PRNG seed (default 0)
  n_sites: 51                          # lattice_extended only (required, WP22 Part 2,
                                        #   CONVENTIONS.md section 15): number of sites along
                                        #   site_axis
  site_spacing_m: 4.065e-07            # lattice_extended only (required): center-to-center
                                        #   site spacing, meters
  site_axis: [0.0, 0.0, 1.0]           # lattice_extended only, optional (default [0,0,1]):
                                        #   direction sites are distributed along; need not be
                                        #   pre-normalized
  site_envelope: gaussian              # lattice_extended only, optional (default "gaussian"):
                                        #   gaussian | uniform site-occupation weighting
  site_envelope_sigma_m: 4.0e-04       # lattice_extended + site_envelope=gaussian only
                                        #   (required then): envelope standard deviation, m

integration:
  mode: auto                           # optional (default "auto"): auto | fast_path | worldline
                                        #   (lattice) | direct | secular (classical): see
                                        #   docs/timescales.md for the three-tier architecture
  time_s: 1.0                          # direct interrogation time, seconds, required for
                                        #   fast_path/secular; an alternative to dtau+steps for
                                        #   direct/worldline
  dtau: 0.5                            # Compton-unit step size, dτ̃ (E9); optional for
                                        #   mode=direct (auto-selected via E31's select_dtau
                                        #   when omitted)
  steps: 2000                          # number of integration steps; optional if time_s is given
  points_per_period: 100               # optional (default 100, E31 N_res): trap-period
                                        #   resolution for auto-selected dtau / secular's
                                        #   internal one-orbit quadrature
  renorm_every: 1000                   # optional, rotor renormalization cadence (E20)
  max_trajectory_memory_gb: 2.0        # optional (default 2.0): pre-flight ceiling on the
                                        #   estimated dense-trajectory allocation for
                                        #   mode=direct/worldline/secular (docs/timescales.md,
                                        #   "Safety net: the trajectory-memory guard").
                                        #   coupling.type=stark_dc + mode=direct against a
                                        #   smoother-backed field (field.csv/field.comsol)
                                        #   adds an extra term for that path's whole-trajectory
                                        #   field evaluation. worldline/secular reject a
                                        #   config over this limit outright
                                        #   (PipelineConfigError); ensemble.regime=classical +
                                        #   mode=direct instead switches to a memory-bounded
                                        #   streaming accumulator by default: see `evaluation`
                                        #   below (WP19).
  evaluation: auto                     # optional (default "auto"): auto | batched | streaming
                                        #   (WP19). Only affects ensemble.regime=classical +
                                        #   mode=direct (both coupling.type values); ignored by
                                        #   every other mode. "auto": run the fast batched
                                        #   accumulator when the max_trajectory_memory_gb
                                        #   estimate fits, else switch to the O(M)-memory
                                        #   streaming accumulator (a note is recorded in the
                                        #   report's uncertainty_notes). "batched": always run
                                        #   the batched path, raising PipelineConfigError if it
                                        #   doesn't fit (the pre-WP19 behavior, explicit).
                                        #   "streaming": always run the streaming path,
                                        #   regardless of whether the config would fit batched.
                                        #   See docs/timescales.md's "Safety net" section.
  trajectory_stride: null              # optional (default null/None), streaming path only:
                                        #   how often (in steps) to retain a position snapshot
                                        #   in the report's underlying trajectories output
                                        #   (PipelineResult.trajectories via the Python API;
                                        #   not part of report.json/line_profile.csv). null:
                                        #   only the initial/final positions are kept (O(M)).
                                        #   An explicit stride retains more, at
                                        #   O(M * steps / trajectory_stride) memory, still
                                        #   bounded, but opt-in. Ignored when evaluation
                                        #   resolves to "batched" (the batched path always
                                        #   returns the full dense trajectory).

output:
  directory: .                         # created if missing
  report_filename: report.json         # optional
  line_profile_filename: line_profile.csv  # optional
  n_time_samples: 512                  # optional: time samples for the coherence/line-profile

uncertainty_notes: ""                  # optional free-text note, forwarded to the report
```

Every field except the ones marked "optional" above is required;
`PipelineConfig.from_yaml`/`from_dict` raise a
`cliffordclock.pipeline.PipelineConfigError` with a specific field path on
any missing/invalid value.

**YAML scientific notation gotcha:** PyYAML's default (YAML 1.1) float
resolver only recognizes `1.0e+6`/`1.0e-6` (an explicit exponent sign) as a
float literal: `1.0e6` without the sign parses as a *string*. Always
write an explicit `+` or `-` in exponents, as in the schema above.

### Field sources

- `synthetic.kind` selects one of the closed-form test fields in
  `cliffordclock.fields.synthetic` (`params` are that factory's keyword
  arguments): `uniform` (`e0`), `constant_gradient` (`e0`, `grad`),
  `quadrupole` (`k`), `gaussian_bump` (`amplitude`, `center`, `width`).
  These have exact, hand-derived gradients (no smoothing/fitting);
  see `docs/fields.md`.
- `csv` loads a CSV-exported field grid (`docs/fields.md`) and fits a
  `cliffordclock.fields.smoother.FieldSmoother` to it.
- `comsol` loads a COMSOL "Spreadsheet"-format `File > Export > Data`
  export (`docs/fields.md`, `docs/byof-guide.md`'s "COMSOL exports"
  section) and fits a `FieldSmoother` to it exactly like `csv`
  (`field.smoothing` applies to both). `field.expression_prefix`
  (optional, default `"es"`) is forwarded to
  `cliffordclock.fields.load_field_comsol`'s `expression_prefix` argument.
  Pass a different value if your COMSOL model renamed the
  Electrostatics physics interface. `examples/comsol_electrode_sr87.yaml`
  is a runnable example.

### Pivot coupling (`coupling:`)

`coupling.type` selects the pivot-perturbation model (CONVENTIONS.md
section 5); see `docs/coupling.md` for the full physics/API.

- `linear_mu` (E14a; the default when `coupling.type` is omitted): an
  explicit, user-supplied effective dipole moment `coupling.mu`. A
  closed-form MVP coupling used to validate the integrator/phase-
  accumulation pipeline (Sprint 1), not a claim about the real physics
  of a clock transition (real clock states carry no permanent dipole).
- `stark_dc` (E14b; **recommended for new configs**): the physical
  second-order DC-Stark shift, `P(r) - 1 = -(Delta_alpha/2)|E(r)|^2/(h
  nu0)`. With no override fields, the coefficient is resolved from the
  `species` registry entry's cited differential polarizability
  (`cliffordclock.ensemble.species`; populated for `Sr87`/`Yb171`/`Al27+`/
  `In115+`, the latter two, singly-ionized J=0 -> J=0 clock transitions,
  WP21). `delta_alpha_dc_si`/
  `stark_coefficient_hz_per_v2_m2` optionally override the registry value
  (e.g. for a species not in the registry, or a newer measurement).
  `stark_dc` works in every `integration.mode` (`fast_path`, `direct`,
  `secular`, `worldline`); for `direct` it runs through a coupling-
  agnostic scalar phase accumulator rather than the rotor path (same
  E21/E22 physics: see `cliffordclock.pipeline._stark_scalar_ensemble`).
  For `worldline` it runs through the **true Cl(1,3) rotor** instantiated
  for E14b (`cliffordclock.integrator.omega.build_omega_stark`, via
  `cliffordclock.pipeline._stark_rotor_ensemble`), directly verified
  against the scalar formulation by `tests/test_integrator_stark_rotor.py`:
  see `docs/coupling.md`'s "Production path vs. general engine" note
  and `cliffordclock.pipeline`'s module docstring mode table for the full
  per-mode accumulator breakdown.
  The report's `uncertainty_notes` records the resolved `k_S`/`nu0` and
  their provenance (registry species + citation, or "explicit config
  override"), and for `integration.mode: fast_path` runs, an explicit
  note that the fast path omits the motional second-order Doppler shift
  (E29 scope, CONVENTIONS.md section 12), a real, separately-budgeted
  systematic, not included in that run's `mean_fractional_shift`.
  `examples/lattice_sr87_stark.yaml` is a runnable `stark_dc` example.

Shipped validation example YAMLs (`examples/quadrupole_classical.yaml`,
`examples/lattice_sr87.yaml`) stay `linear_mu` (they validate the toy
integrator path); `examples/lattice_sr87_stark.yaml` is the `stark_dc`
example, reporting a physically meaningful Sr-87 DC-Stark shift.

### Environment (`environment:`, BBR shift, WP20)

`environment.radiation_temperature_K` turns on the blackbody-radiation
shift (CONVENTIONS.md E32/E33): see `docs/coupling.md`'s
"Blackbody-radiation shift" section for the physics and API. Key points
for the config schema specifically:

- **Absent by default.** No shipped example sets this section, so every
  shipped example's output is byte-identical to a version of
  `cliffordclock` without WP20 at all.
- **Requires `coupling.type: stark_dc`.** BBR needs the species' registry
  `BbrCoefficients` (a separately published static/dynamic fit, distinct
  from `delta_alpha_dc_si`), which `linear_mu` has no equivalent of;
  setting `radiation_temperature_K` with `coupling.type: linear_mu` is a
  `PipelineConfigError` at config-load time, not a silently ignored key.
- **Hard-validated range.** `radiation_temperature_K` must lie in
  `[50, 350]` kelvin (the published fit's validity window):
  `PipelineConfigError` outside it, both edges.
- **Optional uncertainty.** `radiation_temperature_uncertainty_K`
  propagates through the exact BBR-polynomial derivative into the
  report's BBR uncertainty note; omitting it makes that note explicit
  that the BBR uncertainty is conditional on an exactly-known `T`.
- **Composed into every `integration.mode`** (`fast_path`, `secular`,
  `direct`, `worldline`) and both `integration.evaluation` values
  (`batched`, `streaming`) for `ensemble.regime: classical` +
  `integration.mode: direct`.
- Currently populated for `Sr87`/`Yb171` only (same registry-population
  pattern as `coupling.type: stark_dc`'s `delta_alpha_dc_si`); `Al27+`
  raises a clear, caught error.

### Multi-surface thermal environment (`environment.radiation_environment:`, WP29 Tier 1)

`environment.radiation_environment` is a multi-surface alternative to
`radiation_temperature_K` (CONVENTIONS.md E37): instead of one ambient
temperature, the atoms sit in an enclosure of `N` named surfaces, each
with its own solid-angle weight, temperature, optional temperature
uncertainty, and optional emissivity. It resolves to the same `(P-1)_BBR`
scalar `radiation_temperature_K` does and threads into every evaluation
mode identically.

- **Mutually exclusive with `radiation_temperature_K`.** Setting both is
  a `PipelineConfigError` at config-load time naming both keys.
- **Requires `coupling.type: stark_dc`**, and every other `radiation_temperature_K`
  cross-field requirement above (registry `BbrCoefficients`, currently
  `Sr87`/`Yb171` only).
- **`surfaces` is a non-empty list**, each entry requiring `name`,
  `weight`, and `temperature_K`; `temperature_uncertainty_K` (default
  `0.0`) and `emissivity` (default: none) are optional per surface. Every
  surface's `name` must be distinct: a duplicate name is a
  `PipelineConfigError` naming the repeated value, checked identically
  whether `surfaces` is written inline or loaded from `surfaces_file`
  below (see "Surfaces table file format"'s own note on this).
- **`surfaces_file` is an equivalent, mutually exclusive alternative to
  `surfaces`**: a path to a plain-text surfaces table (WP29 Tier 1 Part 1;
  see "Surfaces table file format" below), parsed into the exact same
  per-surface structure `surfaces` carries, before any of the checks
  below run. Exactly one of `surfaces`/`surfaces_file` is required;
  giving both, or neither, is a `PipelineConfigError`. A relative
  `surfaces_file` path is resolved against the directory containing the
  config file that names it (mirroring the CLI's own
  `--radiation-surfaces` flag, which resolves relative to the current
  working directory instead, since it comes from the command line).
- **Weights must sum to 1** across all surfaces, within a `1e-9` absolute
  tolerance: `PipelineConfigError` otherwise, checked at config-load time.
- **`temperature_K` must lie in `[50, 350]`** for every surface, the same
  hard-validated window as `radiation_temperature_K` above.
- **`emissivity`, when set, must lie in `(0, 1]`, and at most one surface
  in the whole list may set it.** That surface is treated as a single
  reflective enclosure (PTB's own topology, Nosske et al.
  arXiv:2507.14030): every other surface is a direct-view aperture, and
  the enclosure's effective weight is whatever is left after the
  apertures' weights are corrected for reflections
  (`w_i_eff = w_i / (W + (1 - W) * emissivity)`, `W` the apertures'
  combined raw weight), never an independently renormalized share.
  Setting `emissivity` on more than one surface is a `PipelineConfigError`:
  multi-reflector radiosity (more than one partially-reflective enclosure
  surface) is out of scope for this tier.
- **`correlated`** (default `false`) selects how per-surface temperature
  uncertainties combine: `false` combines them independently, in
  quadrature; `true` treats every surface's temperature error as moving
  together (a shared calibration-chain error) and combines them linearly
  before taking the magnitude, never smaller than the independent mode
  for the same inputs.
- **A uniform, single-surface environment** (`weight: 1.0`, no
  `emissivity`) reduces to `radiation_temperature_K`'s result bit for
  bit, not just numerically.
- **The report's `uncertainty_notes`** lists every surface's
  name/weight/temperature, the per-moment effective temperatures
  `T_eff,n` (one per registry dynamic-term power plus `n=4`), and the
  uncertainty combination mode.

#### Surfaces table file format (`surfaces_file`, WP29 Tier 1 Part 1)

A plain-text, UTF-8 alternative to writing `surfaces:` inline, mirroring
`docs/fields.md`'s COMSOL-format documentation approach: one surface per
line, whitespace-separated columns,

```
name weight temperature_K [temperature_uncertainty_K] [emissivity]
```

```
# Two-surface enclosure-and-aperture example (CONVENTIONS.md E37).
# name      weight  temperature_K  temperature_uncertainty_K  emissivity
shield       0.9     100.0          -                          0.5
aperture     0.1     300.0          0.01                       -
```

- `#` starts a comment, whole-line or trailing; blank lines are ignored.
  The file is read as UTF-8, tolerating (and stripping) a leading
  byte-order mark, so a plain Notepad "Save As UTF-8" file works
  unmodified.
- `name` must be a bare token (no whitespace).
- `weight` and `temperature_K` are required on every line.
- `temperature_uncertainty_K` and `emissivity` are optional trailing
  columns; write `-` for either one to leave it absent while still
  supplying the other (as `shield`'s row does above, to set `emissivity`
  with no `temperature_uncertainty_K`).
- Every column, and every surface as a whole, parses through the exact
  same checks the inline `surfaces:` list goes through (unique names,
  weight normalization, the `[50, 350]` K validity window, the emissivity
  topology rule). These run once, after the file is loaded, identically
  for both input forms, never as separate file-specific logic: a
  `surfaces_file` and the equivalent inline `surfaces:` list produce
  byte-identical pipeline results.
- Malformed input specific to the file's own grammar (wrong column
  count, a non-numeric column, a blank/reserved-token name, a missing
  file) raises `PipelineConfigError` naming the file, the 1-based line
  number, and the offending token, matching `load_field_comsol`'s error
  style (`docs/fields.md`). A duplicate surface name is instead caught by
  the shared cross-form check above; its error names the surface's index
  and value, not a file line number.

See `examples/radiation_environment_surfaces.txt` and
`examples/radiation_environment_surfaces_sr87.yaml` for a complete
worked example.

### Gravitational redshift (`environment.gravity:`, WP22)

`environment.gravity` turns on the gravitational-redshift pivot term
(CONVENTIONS.md section 15, E36): the systematic mm-scale extended
samples measure directly (Bothwell et al., Nature 602, 420 (2022);
`benchmarks/run_bothwell_redshift.py`).

- **Absent by default.** No shipped example sets this section, so every
  shipped example's output is byte-identical to a version of
  `cliffordclock` without WP22 at all.
- **Requires `coupling.type: stark_dc`.** Mirrors `radiation_temperature_K`'s
  cross-field validation exactly: the term is composed at the same E14b
  rate-function call sites as BBR/the quadrupole term.
- **`g_m_s2` defaults to standard gravity** (9.80665 m/s^2, exact by
  definition), a placeholder at the 1e-19 level. For any 1e-19-class
  comparison against a real lab site, set this to the site's own SURVEYED
  LOCAL value instead (e.g. 9.796 m/s^2 at JILA's Boulder, CO site).
- **`up_axis`/`reference_height_m`** define the height coordinate
  `h(r) = up_axis_hat . r`; `(P-1)_grav = g*(h - reference_height_m)/c^2`.
  A HIGHER clock (larger `h`) runs FASTER.
- **Composed into every `integration.mode`** the same way as BBR/the
  quadrupole term: `fast_path`, `secular`, `direct` (batched+streaming),
  and `worldline` (through the rotor's scalar `B_hat_C` rotation-plane
  coefficient only; static, `v=0` lattice/lattice_extended nodes make this
  the WHOLE contribution: see CONVENTIONS.md section 15).
- **A runtime warning** (not a config-load-time rejection) is recorded in
  the report's `uncertainty_notes` if a run's sampled positions span more
  than `cliffordclock.pipeline.GRAVITY_EXTENT_WARN_M` (10 m) along
  `up_axis`, the uniform-g approximation's validity margin
  (CONVENTIONS.md section 15).
- **Coordinate-sign convention is this project's own** (higher along
  `up_axis` = faster): a paper's own published axis convention may differ
  (e.g. Bothwell's z-axis increases toward LOWER physical height);
  `benchmarks/run_bothwell_redshift.py` documents and applies that
  specific mapping explicitly when comparing against a published number;
  this schema itself makes no such external-convention assumption.

### Quadrupole shift (`quadrupole:`, WP21)

`quadrupole:` turns on the electric-quadrupole shift (CONVENTIONS.md
E34/E35) for D/F-state ion clocks: the systematic that coexists with
(and is structurally distinct from) the DC-Stark shift for J>=1 upper
clock states.

- **Absent by default.** No shipped example sets this section.
- **Requires `coupling.type: stark_dc`.** The quadrupole term is composed
  at the same E14b rate-function call sites as the Stark/BBR terms.
- **`Theta(J)` source:** either `state` (a
  `cliffordclock.ensemble.species.QUADRUPOLE_MOMENTS` registry key, e.g.
  `"Ca+:D5/2"`) or an explicit `theta_au`+`j` override, mutually
  exclusive, exactly one form required.
- **`nu_0_hz` is always required** (the clock transition frequency for
  the SAME upper state as `Theta`): unlike the Stark/BBR terms, this is
  NOT resolved from the top-level `species:` selection, since the D/F-
  state ions the registry's `Theta` values are for (Ca+, Sr+, Ba+, Yb+)
  are not themselves registered `Species` entries (no clock-transition
  frequency is pinned for them in this WP's scope: see
  `cliffordclock.ensemble.species.QuadrupoleMoment`'s docstring).
- **`averaging_mode: fixed`** (default) evaluates at a single `m_j`/
  `quantization_axis`. **`averaging_mode: three_orientation`** implements
  the exact three-mutually-perpendicular-orientation cancellation
  identity (CONVENTIONS.md E35 A2), contributes exactly `0.0` to every
  evaluation mode, `m_j`/`quantization_axis` are ignored.
- **Composed into every `integration.mode`** the same way as BBR.
- **Traceless symmetric gradient only.** The quadrupole term uses only
  the traceless symmetric part of the field-gradient tensor (E13); its
  own contribution to the rotor's spin connection (`worldline` mode) is
  a documented, bounded scope limit, not modeled (CONVENTIONS.md E35).
- **Micromotion boundary / hyperfine-E2 budget notes.** Any run whose
  `species:` is `Al27+`/`In115+` (WP21's registered ion clocks) carries
  two report notes regardless of whether `quadrupole:` is set: the
  micromotion-boundary note (the same stray field that produces the
  modeled quadrupole/Stark shift also produces the larger, unmodeled
  excess-micromotion pathway) and the hyperfine-mediated-E2 budget line
  (a real, unmodeled second-order effect for I != 0 J=0 ions).

### Ensemble regimes

- `classical`: Maxwell-Boltzmann Monte-Carlo positions/velocities
  (`cliffordclock.ensemble.classical`). For `integration.mode: direct`
  (the default), propagated through the trap with velocity-Verlet at a
  step `dt = integration.dtau * τ_c` (E9): **the same
  `integration.dtau` value drives both the trajectory sampling and the
  rotor integrator**, so trajectory samples are spaced at exactly `dτ̃`,
  which the integrator's finite-difference velocity assumes (see
  `cliffordclock.integrator.worldline.integrate_worldline`).
  `integration.mode: secular` instead evaluates a closed-form one-orbit
  average (E30; requires an isotropic trap and periodic motion).
- `lattice`: Hermite-Gauss motional-state quadrature nodes
  (`cliffordclock.ensemble.lattice`), each a static position (no
  trajectory) weighted by its quadrature weight. `integration.mode:
  fast_path` (the default) evaluates the exact E29 quadrature expectation
  directly, at O(1) cost in `integration.time_s`;
  `integration.mode: worldline` runs the same rotor integrator as the
  classical regime, as an explicit cross-check.
- `lattice_extended` (WP22 Part 2, CONVENTIONS.md section 15): `n_sites`
  copies of the `lattice` regime's own single-site Hermite-Gauss
  quadrature (`cliffordclock.ensemble.lattice.extended_lattice_nodes`),
  distributed along `site_axis` at `site_spacing_m` spacing with a
  Gaussian-or-uniform site-occupation envelope (`site_envelope`). Every
  site's own position feeds every pivot term already in play (the local
  field/Stark shift, uniform BBR, and the height-dependent gravitational
  redshift). Uses the SAME `fast_path`/`worldline` accumulators as
  `lattice`: no new evaluation-mode machinery; the existing `lattice`
  regime (single site) is entirely untouched. **Output:** the Python API's
  `PipelineResult.site_map`
  (`cliffordclock.pipeline.LatticeExtendedSiteMap`) carries the per-site
  frequency map (each site's own position/weight/mean shift) plus a
  weighted-least-squares linear-gradient fit (`slope_per_m`, the map's
  headline number) and the gate-mandated deterministic-vs-stochastic
  dispersion-labeling split (`total_spread_fractional`/
  `gradient_removed_residual_spread_fractional`), not part of
  `report.json`'s schema (`MetrologyReport` is unchanged), but a
  test-pinned note in `report.uncertainty_notes` states that
  `t2_star_s`/`shift_std_error` include the deterministic per-site
  gradient and points to `site_map` for the split. See
  `benchmarks/run_bothwell_redshift.py` for a full worked example (the
  Bothwell et al. 2022 mm-scale redshift measurement).

### Interrogation times and the three-tier fast-path architecture

**See `docs/timescales.md`** for the full explanation: why real
(microsecond-to-second) interrogation times are the norm (not an
"unexplored headroom" caveat), the three-tier architecture
(`fast_path`/`direct`/`secular`/`worldline`, CONVENTIONS.md v1.1.0-draft
section 12, E29-E31) this schema configures, the large-`dτ̃` accuracy
study behind `select_dtau`'s automatic step-size selection, and the
1-second lattice demo (`examples/lattice_sr87.yaml`).

## Examples

`examples/quadrupole_classical.yaml` (synthetic quadrupole field, a
classical ion-trap-style ensemble) and `examples/lattice_sr87.yaml`
(synthetic Gaussian-bump field, an Sr-87 optical-lattice ensemble, a
genuine 1-second interrogation via the E29 fast path: see
`docs/timescales.md`) both run in well under a minute on CPU:

```bash
cliffordclock run examples/quadrupole_classical.yaml --output-dir /tmp/cliffordclock_out
cat /tmp/cliffordclock_out/report.json
```

`examples/comsol_electrode_sr87.yaml` is the `field.comsol` config-wiring
example: the same COMSOL "Spreadsheet" export
(`examples/fd_electrode_field.txt`) `docs/byof-guide.md`'s "COMSOL
exports" section walks through via the Python API, loaded here directly
from `config.yaml`: see the file's header comment for the physics and
back-of-envelope shift derivation.

## Output files

- `report.json`: a `cliffordclock.analytics.MetrologyReport`: see
  `docs/report-schema.md` for the field-by-field schema.
- `line_profile.csv`: the spectral line profile (E28): see
  `docs/report-schema.md`.
- `site_map.json` (WP22 Part 2, `ensemble.regime: lattice_extended` runs
  only): a `cliffordclock.pipeline.LatticeExtendedSiteMap`: the per-site
  frequency map plus the dispersion-labeling split (see the
  "Gravitational redshift" section above). Not written at all for any
  other regime. Filename configurable via `output.site_map_filename`
  (default `site_map.json`).

## Python API equivalent

The CLI is a thin wrapper; the same run is available from Python:

```python
from cliffordclock.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig.from_yaml("examples/quadrupole_classical.yaml")
report = run_pipeline(config)
print(report.mean_fractional_shift, "+/-", report.shift_std_error)
```

`run_pipeline` returns only the `MetrologyReport`;
`cliffordclock.pipeline.run_pipeline_full` returns a `PipelineResult` with
the underlying `EnsembleResult`, trajectories/weights, and line-profile
arrays too (what the CLI uses to write both output files).
