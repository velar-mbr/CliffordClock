# CliffordClock documentation

CliffordClock is an open-source (AGPLv3) Python/JAX library that computes
optical-atomic-clock fractional frequency shifts (target regime: 1e-18
level) caused by stray DC electric fields and by the atom's own motion.

## Status

**Pre-beta research code.** See the top-level [README](../README.md)'s
Status section for the two-sentence validation summary, and
[`validation.md`](validation.md) for the full record.

## Start here: Tutorial

- [`tutorial.md`](tutorial.md): install, run the physical example (with
  every summary line explained in plain language), run the validation
  example, then bring your own field. If you read nothing else, read this.

## How-to guides

Task-oriented instructions for a specific job:

- [`byof-guide.md`](byof-guide.md): "bring your own field": export a
  field grid from your own FEA/CAD tool and get a shift report for your
  own trap geometry. CSV format contract, grid-spacing guidance, smoother
  warnings, and the current scope with a pointer to
  [`docs/roadmap.md`](roadmap.md).
- [`cli.md`](cli.md): the `cliffordclock` command-line interface and the full
  `config.yaml` schema (field sources, ensemble regimes, integration
  parameters, coupling types).

## Reference

Look-up material, organized around the artifact, not a narrative:

- [`report-schema.md`](report-schema.md): the `report.json` and
  `line_profile.csv` output formats written by `cliffordclock run`.
- [`coupling.md`](coupling.md): the DC-Stark coupling's Python API
  (species registry, precision discipline, function signatures).
- [`fields.md`](fields.md): the CSV field-import format and the
  `FieldSmoother` fit/evaluate API.
- [`CONVENTIONS.md`](CONVENTIONS.md): the single source of truth for
  every equation this library implements (cited by equation number, e.g.
  E21, throughout the codebase and this documentation).

## Explanation

Background and design reasoning, for understanding *why*:

- [`validation.md`](validation.md): every case this tool has been
  checked against, with formula/source, tolerance, and measured
  agreement. Start here if you're deciding whether to trust a number this
  tool reports.
- [`timescales.md`](timescales.md): why real (microsecond-to-second)
  interrogation times are cheap for this tool, and which execution mode
  to use for your regime.
- [`coupling.md`](coupling.md): (also) the physics: why the DC-Stark
  shift is quadratic in the field, and how the validation coupling
  relates to the physical one.

## Runnable material

- [`notebooks/01_end_to_end_demo.ipynb`](../notebooks/01_end_to_end_demo.ipynb):
  an end-to-end walkthrough with plots (field/gradient visualization,
  ensemble trajectories, the spectral line profile).
- [`notebooks/02_step_size_study.ipynb`](../notebooks/02_step_size_study.ipynb):
  the step-size accuracy study behind `timescales.md`: large-`dtau`
  convergence order, a renormalization-cadence finding, secular
  averaging, and the 1-second lattice fast-path demo.
- [`notebooks/03_known_answers.ipynb`](../notebooks/03_known_answers.ipynb):
  the four known-answer validation cases (KA1-4), narrated.
- [`notebooks/04_bring_your_own_field.ipynb`](../notebooks/04_bring_your_own_field.ipynb):
  the adaptation template for your own trap's field export.
- [`notebooks/05_gradient_showcase.ipynb`](../notebooks/05_gradient_showcase.ipynb):
  the paper's showcase case: a chamber-scale field with real spatial
  structure, the full inhomogeneous shift budget, and a trajectory-mode
  vs. rotor-mode cross-check on identical trajectories.
- [`notebooks/06_npl_reproducibility.ipynb`](../notebooks/06_npl_reproducibility.ipynb):
  the NPL stray-field reproducibility case: the E14b governing
  equation, the pipeline run stage by stage (field, ensemble, coupling,
  integration), then the wrapped benchmark case reproducing the same
  numbers.
- [`notebooks/07_bbr_jila_arithmetic.ipynb`](../notebooks/07_bbr_jila_arithmetic.ipynb):
  the JILA BBR-row arithmetic-reproduction case (E32), an explicitly
  weaker class than reproducibility, staged the same way as notebook 06.
- [`notebooks/08_roos_quadrupole_slope.ipynb`](../notebooks/08_roos_quadrupole_slope.ipynb):
  the Ca+:D5/2 ion-clock quadrupole-slope case (E34/E35) against Roos
  et al.'s measured two-ion slope, built up single-ion shift by
  single-ion shift.
- [`notebooks/09_bothwell_redshift.ipynb`](../notebooks/09_bothwell_redshift.ipynb):
  the Bothwell mm-scale gravitational-redshift reproducibility case
  (E36), this project's second: the extended-lattice geometry built by
  hand, then the live reduced-site run set against the full committed
  benchmark grid.
- [`notebooks/10_grand_tour.ipynb`](../notebooks/10_grand_tour.ipynb):
  the three lattice-clock terms (DC Stark, BBR, gravity) composed live
  on one chamber-scale scenario, cross-checked through the rotor
  engine, then bridged to the extended-lattice view.
- [`notebooks/11_real_budget_slice.ipynb`](../notebooks/11_real_budget_slice.ipynb):
  one real clock, the JILA Sr system: the covered slice of its
  published systematic evaluation computed from published inputs,
  beside the lab's own numbers, in a single composed pipeline run.
- [`notebooks/12_thermal_environment.ipynb`](../notebooks/12_thermal_environment.ipynb):
  the multi-surface thermal environment (E37), quantifying live where a
  single effective temperature stops representing a real, non-uniform
  enclosure, against the PTB aperture formula and a JILA
  temperature-step check, then a field-deployment sensitivity band from
  per-surface sensor readings and uncertainties.
- [`notebooks/13_trapped_ion_quantum_motion.ipynb`](../notebooks/13_trapped_ion_quantum_motion.ipynb):
  the trapped-ion walkthrough, quantum motional states throughout: a
  ground-state ion's ~10 nm wavepacket sampled against a field with real
  curvature, the Coulomb-crystal quadrupole case recapped from notebook 08,
  the motional time-dilation row (E38) reproduced against a published Al+
  evaluation and its per-mode participation-factor correction for the
  two-ion crystal, and the coherent Ramsey fringe visibility a squeezed
  motional state leaves behind (E39), closing with the excess-micromotion
  input channel and a scope statement against the RF-dynamics roadmap.
