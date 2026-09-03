# CliffordClock

![status: pre-beta](https://img.shields.io/badge/status-pre--beta-orange)
![license: AGPLv3](https://img.shields.io/badge/license-AGPLv3-blue)

![CliffordClock showcase animation: Monte Carlo atoms dispersing through a real chamber field, colored by accumulated fractional shift, with the ensemble coherence decaying at T2*](https://raw.githubusercontent.com/velar-mbr/CliffordClock/main/docs/assets/showcase_animation.gif)

![CliffordClock trapped-ion animation: a two-ion 27Al+/25Mg+ crystal cycling through its six normal modes next to the time-dilation budget those modes add up to, compared against a published trapped-ion clock's own measured value](https://raw.githubusercontent.com/velar-mbr/CliffordClock/main/docs/assets/ion_motion_animation.gif)

![CliffordClock lattice-fit animation: a noisy synthetic sideband spectrum with a Born-Oppenheimer-plus-WKB model curve fit to it live by gradient descent, lattice depth and radial temperature converging to their truth values](https://raw.githubusercontent.com/velar-mbr/CliffordClock/main/docs/assets/lattice_fit_animation.gif)

**CliffordClock predicts how stray electric fields shift and broaden an
optical lattice clock's frequency, starting from your own field
simulation.** Export a field map from COMSOL or any FEA tool, and
describe your atoms and trap in a short config file. CliffordClock
returns the fractional frequency shift, its spread across your atom
cloud, the dephasing time T₂*, and the spectral line profile. These
numbers land at the 1×10⁻¹⁸ level today's clocks budget to.

Free and open source (AGPLv3), Python, `pip install`-able.

## Start here

- **[Quickstart](#quickstart)**: install and run a real case in a minute.
- **[How it works](docs/MODEL.md)**: the composition rule, in
  [six lines of Python](examples/six_line_clock.py); every systematic
  is a published model plugged into it. A six-page
  [companion paper](paper/composition/main.pdf) compares this
  composition with the field's budget tables.
- **[The physics, term by term](docs/terms/)**: one page per systematic,
  formula and source.
- **[Deep dives](#notebooks)**: the full pipeline, notebook by notebook.
- **[Validation record](docs/validation.md)**: every check this engine
  has passed, case by case.

## Why use CliffordClock?

Every systematic this engine models enters through one rule. The
DC-Stark shift, blackbody radiation, motional time dilation, the
lattice light shift, the electric-quadrupole shift, and gravitational
redshift each enter as a multiplicative factor on one per-atom clock
rate. The reported fractional shift is the mean of the resulting phase
distribution, and the Ramsey fringe visibility comes from the same
distribution, so one calculation produces the number a budget quotes
and the coherence a lab measures. The rule is six lines of Python
([`docs/MODEL.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/MODEL.md));
each factor inside it is a published physics model, cited in the
output.

That rule runs per atom over your own imported field. A textbook
formula gives one shift for one field value. CliffordClock computes
every atom's shift from where it sits in your trap, then rolls the
ensemble up into per-atom and per-site maps, the spread, the dephasing
time T₂* that spread implies, and the spectral line profile. The
lattice-light-shift and sideband models are differentiable end to end,
so a lab's trap depth and radial temperature can be fitted by gradient
descent through the same physics its light-shift budget uses. A
millimetre-scale extended-lattice sample runs on a laptop.

Each of these capabilities is checked against published results. Two
published measurements, NPL's stray-field reconstruction and Bothwell
et al.'s mm-scale gravitational-redshift measurement, come out of this
pipeline with zero fitted parameters. Published budget rows are
reproduced from their own inputs, among them a blackbody row and a
trapped-ion motional evaluation that lands 0.08 sigma from the
published value. The lattice models are cross-validated against an
independent open-source implementation. The
numerics are proven at 1×10⁻¹⁸ against adversarial tests, and two
independent formulations, a direct scalar calculation and a Cl(1,3)
geometric-algebra rotor engine, agree to the last digit on every case
shipped. Every coefficient carries its source, which paper and which
value, in the output.

## What it does today

- [x] **Field import**: plain CSV or COMSOL's `File > Export > Data`
  spreadsheet, straight into the config
- [x] **DC-Stark shift** with published polarizabilities for Sr-87 and
  Yb-171, or any coefficient you supply
- [x] **Second-order Doppler** carried through the exact relativistic
  kinematic factor
- [x] **Blackbody radiation**: one temperature or a full multi-surface
  environment (surfaces table, YAML, or `--radiation-surfaces`), checked
  against PTB's published formula and position scan (notebook 12)
- [x] **Trapped-ion motional time dilation** from measured mode
  frequencies and phonon numbers, with two-ion participation and
  micromotion factors reconstructed by a coupled Floquet fit; the fit
  lands at 0.08 sigma from the published Al⁺ evaluation (notebook 13)
- [x] **Ramsey fringe visibility** from the motional state, thermal,
  coherent, or squeezed
- [x] **Ion quadrupole shift** for Ca⁺/Sr⁺/Ba⁺/Yb⁺ D/F states, and
  static-field DC Stark for Al⁺/In⁺, with the micromotion boundary
  stated on every report
- [x] **Millimetre-scale lattice samples** with per-site frequency maps,
  checked against the published Bothwell mm-scale redshift measurement
- [x] **Sideband-spectrum fitting** for lattice-clock trap depth and
  radial temperature, by gradient descent through the same differentiable
  BO+WKB model the lattice light shift uses, cross-validated against an
  independent open-source implementation; a synthetic demonstration
  today, with real-scan fitting awaiting shared data (notebook 15)
- [x] **Rydberg vapor-cell response**: a Rb-85 EIT/Autler-Townes
  spectrum built from per-atom quadratic Stark shifts across a field
  map or a wall-patch model, checked against Holloway et al. 2014's
  own published calibration data (notebook 16)
- [x] **Full Rydberg Stark maps** beyond the quadratic regime: exact
  diagonalization in the quantum-defect `(n,l,j,mj)` basis with adiabatic
  eigenvalue tracking, cross-validated against ARC (an independent
  open-source implementation) and a computed avoided-crossing field
  replacing the earlier order-of-magnitude validity guard
- [x] **Real interrogation times**: a 1-second run costs seconds of
  compute
- [x] **Beyond the mean shift**: per-atom distributions, T₂*, and the
  line profile
- [x] **Numerics for 1×10⁻¹⁸**, with tests that prove it
- [x] **Reports with provenance**: JSON and CSV naming the paper behind
  every coefficient
- [x] **Validated**: five literature known-answer cases, two
  published-measurement reproductions, and two independent formulations
  (scalar and Cl(1,3) rotor) agreeing to machine precision

Every comparison carries its evidentiary class: a reproduction of a
published evaluation from its own inputs is labeled arithmetic
reproduction, a weaker class than agreement with an independent
measurement.
[`docs/validation.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/validation.md)
keeps the ledger.

See [`docs/roadmap.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/roadmap.md) for what's next, and why it's
queued the way it is.

## Quickstart

```bash
pip install cliffordclock

cliffordclock version
```

**New here? Start with [`docs/tutorial.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/tutorial.md)**: it
walks each command one at a time and explains every line of output in
plain language. The fast version, with the examples from a clone of
this repository:

```bash
git clone https://github.com/velar-mbr/CliffordClock.git
cd CliffordClock
cliffordclock run examples/realistic_lattice_sr87.yaml --output-dir /tmp/cliffordclock_out
```

```
CliffordClock run summary
  species:                Sr87
  ensemble:               lattice_fast_path (M=512)
  interrogation time:     1.000000e+00 s
  mean fractional shift:  -7.723399e-19 +/- 6.771e-24 (SEM)
  T2*:                    4.552836e+01 s
  ...
```

That's a physically realistic scenario: stray charge patches on
in-vacuum surfaces, sized to bracket a documented real event at a Sr
lattice clock ([Lodewyck et al. 2012](https://arxiv.org/abs/1108.4320)).
The field is imported from a CSV field file the way your own FEA export
would be, and the run uses a genuine 1-second interrogation time. To
point it at **your** trap:
[`docs/byof-guide.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/byof-guide.md).

## Can you trust the numbers?

This is **pre-beta research code**. Every number is checked against
exact closed forms and five literature known-answer cases with
published polarizabilities. The pipeline also carries **two
reproducibility cases** against zero blind predictions. It reconstructs
NPL's published stray-field shift from their independently measured
field, and it reconstructs Bothwell et al.'s published mm-scale
gravitational-redshift measurement from an extended-lattice sample's
per-site frequency map. Both reproductions use **zero fitted
parameters**. A blind prediction, a shift nobody had already computed
from the same published inputs, does not exist yet; getting one is the
top roadmap item. The full case-by-case record, with formulas and
sources, is
[`docs/validation.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/validation.md).

## How it works

At every point along an atom's path, the local fractional clock-rate
shift combines the field's quadratic Stark shift with the time dilation
from the atom's speed. CliffordClock integrates that local shift over
where your atoms actually travel, so the method fully handles spatially
varying fields. The same physics also runs through a general
geometric-algebra engine, a Cl(1,3) "rotor" representing the atom's
internal clock. That rotor engine agrees with the simple scalar
calculation to machine precision today, and it exists for physics a
single number per point cannot express; details in
[`docs/coupling.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/coupling.md) and
[`docs/CONVENTIONS.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/CONVENTIONS.md).
The rule that composes all of these terms into the reported shift and
visibility is six lines of Python; each term inside it is a published
physics model of its own. That rule, and why the standard additive
budget is its first-order expansion, is in
[`docs/MODEL.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/MODEL.md), with a
runnable version at
[`examples/six_line_clock.py`](https://github.com/velar-mbr/CliffordClock/blob/main/examples/six_line_clock.py).

## Notebooks

- [`notebooks/01_end_to_end_demo.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/01_end_to_end_demo.ipynb):
  start here. Walks the full pipeline by hand, field synthesis through
  smoother fit, ensemble sampling, integration, and the report, the
  same composition the CLI automates.
- [`notebooks/05_gradient_showcase.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/05_gradient_showcase.ipynb):
  the showcase behind the animation at the top of this page, a
  chamber-scale field with genuine spatial structure carried through to
  a full dispersion budget.
- [`notebooks/06_npl_reproducibility.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/06_npl_reproducibility.ipynb)
  through [`09_bothwell_redshift.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/09_bothwell_redshift.ipynb):
  the validation walkthroughs, one per case (NPL, JILA BBR, Roos
  quadrupole slope, Bothwell redshift), each stating the governing
  equation, building the config, and running the pipeline stage by
  stage.
- [`notebooks/10_grand_tour.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/10_grand_tour.ipynb): the
  grand tour. One chamber-scale scenario with the three lattice-clock
  terms composed live (DC Stark, then +BBR, then +gravity). The same
  scenario is cross-checked through the rotor engine on identical
  trajectories, then bridged to the extended-lattice per-site view.
- [`notebooks/11_real_budget_slice.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/11_real_budget_slice.ipynb):
  the closest-to-a-real-experiment demo. One real clock's published
  evaluation, the JILA Sr system, with the covered rows computed from
  published inputs beside the lab's own numbers in a single composed
  pipeline run.
- [`notebooks/12_thermal_environment.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/12_thermal_environment.ipynb):
  the multi-surface thermal environment (E37), for field-deployed
  clocks sitting in a real, non-uniform radiation environment no lab
  shield has engineered flat. It quantifies live where a single
  effective temperature stops representing that environment, and closes
  with a sensitivity band built from per-surface sensor readings and
  uncertainties.
- [`notebooks/13_trapped_ion_quantum_motion.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/13_trapped_ion_quantum_motion.ipynb):
  the trapped-ion walkthrough, quantum motional states throughout. A
  ground-state ion's ~10 nm wavepacket is sampled against a field with
  real curvature, and the Coulomb-crystal quadrupole case is recapped
  from notebook 08. The motional time-dilation row (E38) is reproduced
  against a published Al+ evaluation, with the per-mode
  participation-factor correction for the two-ion crystal closed by a
  coupled Floquet fit. The notebook closes with the coherent Ramsey
  fringe visibility a squeezed motional state leaves behind (E39), the
  excess-micromotion input channel, and a scope statement against the
  RF-dynamics roadmap.
- [`notebooks/14_lattice_light_shift.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/14_lattice_light_shift.ipynb):
  both community lattice-light-shift models run side by side (E40/E41), the
  closed-form Katori-lineage harmonic/operational model and the NIST
  Born-Oppenheimer-plus-WKB model, each validated against its own group's
  published numbers (Ushijima et al. 2018, Aeppli et al. 2024, and
  Bothwell et al. 2025's own trap-depth-reduction table), then compared
  directly through the density-of-states difference that drives the two
  models apart as radial temperature rises.
- [`notebooks/15_sideband_fitting.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/15_sideband_fitting.ipynb):
  fitting the full lattice model to sideband spectra (E42) by gradient
  descent, so a lab's own thermometry and its own light shift share one
  model. Both lineshape paths generate a carrier-plus-sidebands spectrum
  on the differentiable JAX core. The forward model is cross-validated
  against `large-lattice-model` (INRIM), an independent open-source
  implementation, then fit to a synthetic spectrum with exact autodiff
  gradients and Laplace uncertainties, including the one case whose own
  Hessian-positive-definiteness check catches a fit that should not be
  trusted.
- [`notebooks/16_rydberg_cell_response.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/16_rydberg_cell_response.ipynb):
  a Rydberg-atom RF electrometry chain (E43/E44) for Rb-85's
  5S1/2-5P3/2-32D5/2-33P3/2 ladder: a single atom's quadratic Stark
  shift, the four-level EIT/Autler-Townes susceptibility with Doppler
  averaging, and the Doppler-mismatch-factor derivation this project
  resolved between two disagreeing published forms. Reproduces Holloway
  et al. 2014's own three published calibration pairs, cross-checks the
  Rb-85 nD5/2 polarizability against two independent sources, and
  composes many atoms' shifts across a wall-patch field model into one
  EIT line, reproducing the shift-and-asymmetric-broadening
  phenomenology of Patrick et al. 2025's surface-charge distortion
  problem, still open in the current literature.

## Documentation

- [`paper/main.pdf`](https://github.com/velar-mbr/CliffordClock/blob/main/paper/main.pdf): the paper. The physical model,
  the numerical methods, the full validation record including both
  reproducibility cases, and the chamber-scale showcase, in one
  self-contained read.
- [`paper/composition/main.pdf`](https://github.com/velar-mbr/CliffordClock/blob/main/paper/composition/main.pdf): the companion
  paper. How the field sums an error budget, how this engine composes
  one, and why the two agree to first order, in six pages.
- [`docs/tutorial.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/tutorial.md): start here
- [`docs/MODEL.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/MODEL.md): the six-line composition rule, and what the rest of the code is
- [`docs/index.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/index.md): full documentation map
- [`docs/validation.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/validation.md): what's been checked
  against what
- [`docs/roadmap.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/roadmap.md): what's next, and why
- [`docs/byof-guide.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/byof-guide.md): bring your own field
- [`docs/cli.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/cli.md): CLI and config reference
- [`docs/timescales.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/timescales.md): execution modes and why
  real interrogation times are cheap
- [`docs/CONVENTIONS.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/CONVENTIONS.md): every equation, with
  units

## Contributing & License

[CONTRIBUTING.md](https://github.com/velar-mbr/CliffordClock/blob/main/CONTRIBUTING.md) for dev setup and the quality bar.
GNU AGPL-3.0-or-later, see [LICENSE](https://github.com/velar-mbr/CliffordClock/blob/main/LICENSE).
