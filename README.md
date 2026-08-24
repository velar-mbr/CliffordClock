# CliffordClock

![status: pre-beta](https://img.shields.io/badge/status-pre--beta-orange)
![license: AGPLv3](https://img.shields.io/badge/license-AGPLv3-blue)

![CliffordClock showcase animation: Monte Carlo atoms dispersing through a real chamber field, colored by accumulated fractional shift, with the ensemble coherence decaying at T2*](https://raw.githubusercontent.com/velar-mbr/CliffordClock/main/docs/assets/showcase_animation.gif)

**CliffordClock predicts how stray electric fields shift and broaden an
optical lattice clock's frequency, starting from your own field
simulation.** Export a field map from COMSOL or any FEA tool, describe
your atoms and trap in a short config file, and get back the fractional
frequency shift, its spread across your atom cloud, the dephasing time
T₂*, and the spectral line profile, at the 1×10⁻¹⁸ level today's clocks
budget to.

Free and open source (AGPLv3), Python, `pip install`-able.

## Why use CliffordClock?

A textbook formula gives you one shift for one field value: plug in a
differential polarizability and a stray-field magnitude, and you can
reproduce that number in an afternoon. What CliffordClock ships
is the full dispersion budget of a real imported field: every atom's
shift computed from where it actually sits in your trap, rolled up into
per-atom and per-site maps, the ensemble's spread, the dephasing time
T₂* that spread implies, and the spectral line profile it produces. That
budget runs through numerics proven at 1×10⁻¹⁸ against adversarial
tests, cross-checked by two independent formulations, a direct scalar
calculation and a Cl(1,3) geometric-algebra rotor engine, that agree to
the last digit on every case shipped, with every coefficient carrying
its source, which paper, which value, in the output. A
millimetre-scale extended-lattice sample runs on a laptop. And it reproduces the real world: two
published measurements, NPL's stray-field reconstruction and Bothwell et
al.'s mm-scale gravitational-redshift measurement, come out of this
pipeline with zero fitted parameters.

## What it does today

- [x] **Imports your field export**: plain CSV, or COMSOL's native
  `File > Export > Data` spreadsheet format, straight into the config
- [x] **Quadratic DC-Stark shift** with published differential
  polarizabilities for Sr-87 and Yb-171 (or any coefficient you supply)
- [x] **Second-order Doppler** (relativistic time dilation) carried
  exactly
- [x] **Blackbody-radiation shift**, with published coefficients, for a
  single uniform temperature or a multi-surface radiation environment. A
  multi-surface run takes each surface's own temperature, sensor
  uncertainty, and emissivity, entered as a plain-text surfaces table or
  inline YAML, or swapped in with the `--radiation-surfaces` CLI flag.
  That path reduces to PTB's own published enclosure-and-apertures
  formula and reproduces PTB's own measured position-resolved shift scan
  inside its quoted uncertainty (notebook 12). The single-temperature
  path is checked against JILA's published evaluation by arithmetic
  reproduction, a weaker class than an independent measurement; see
  [`docs/validation.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/validation.md)
- [x] **Real interrogation times**: a 1-second run costs seconds of
  compute
- [x] **More than the mean shift**: the per-atom shift distribution
  across your cloud, the dephasing time T₂* it implies, and the clock
  line's spectral profile
- [x] **Your atoms where they actually are**: thermal Monte-Carlo
  clouds or lattice motional states, sampled through your species and
  trap geometry
- [x] **Numerics built for 1×10⁻¹⁸**: a signal 8 orders of magnitude
  below the baseline survives, and tests prove it
- [x] **Machine-readable reports** (JSON + CSV) carrying the provenance
  of every coefficient, which paper, which value, in the output
- [x] **Checked against things you already know**: textbook Stark
  formulas with literature polarizabilities, exact closed forms, and
  five literature known-answer cases, plus two published-measurement
  reproductions (NPL's Rydberg electrometry, and Bothwell et al.'s
  mm-scale gravitational-redshift measurement); see
  [`docs/validation.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/validation.md)
- [x] **Two independent formulations that must agree**: a direct scalar
  calculation and a geometric-algebra (Cl(1,3) rotor) engine, verified
  against each other to machine precision on every case shipped
- [x] **Ion-clock systematics**: static-field DC Stark for Al⁺/In⁺, and
  the electric-quadrupole shift from your field gradient for
  Ca⁺/Sr⁺/Ba⁺/Yb⁺ D/F states, with every ion report carrying the same
  boundary line: the stray field that produces this shift also drives
  an RF-trapped ion into excess micromotion, a separate and larger
  pathway this release does not model
- [x] **Quantum-motional time dilation for trapped ions**, evaluated
  from measured mode frequencies and mean phonon numbers. For a two-ion
  crystal, per-mode participation factors are reconstructed from the
  measured mode spectrum, and the per-axis intrinsic-micromotion
  enhancement is solved from the published drive frequency. The
  published Al+ evaluation (Marshall et al.) is checked by arithmetic
  reproduction, and the total agrees at 0.10 sigma, with per-mode
  agreement documented and a characterized residual on the trap's
  smaller-a axis (notebook 13)
- [x] **Ramsey fringe visibility**, computed from the ensemble's
  motional state, thermal, coherent, or squeezed, and reported as the
  `ramsey_visibility`/`ramsey_phase` report fields (E39)
- [x] **Millimetre-scale extended-lattice samples** with per-site
  frequency maps (mean shift, spread, T₂*, and gravitational redshift
  all included), checked against the published Bothwell mm-scale
  redshift measurement

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
lattice clock ([Lodewyck et al. 2012](https://arxiv.org/abs/1108.4320)),
imported from a CSV field file exactly the way your own FEA export would
be, at a genuine 1-second interrogation. To point it at **your** trap:
[`docs/byof-guide.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/byof-guide.md).

## Can you trust the numbers?

This is **pre-beta research code**. Every number is checked against
exact closed forms and five literature known-answer cases with
published polarizabilities, and the pipeline carries **two
reproducibility cases** against zero blind predictions: it reconstructs
NPL's published stray-field shift from their independently measured
field, and it reconstructs Bothwell et al.'s published mm-scale
gravitational-redshift measurement from an extended-lattice sample's
per-site frequency map, both with **zero fitted parameters**. A blind
prediction, a shift nobody had already computed from the same published
inputs, does not exist yet; getting one is the top roadmap item. The
full case-by-case record, with formulas and sources, is
[`docs/validation.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/validation.md).

## How it works, in two sentences

At every point along an atom's path, the local fractional clock-rate
shift comes from the field (quadratic Stark) and the atom's speed (time
dilation), then integrates over where your atoms actually are: that
approach fully handles spatially varying fields. The same physics
also runs through a general geometric-algebra engine, a Cl(1,3) "rotor"
representing the atom's internal clock, which agrees with the simple
calculation to machine precision today and exists for the physics a
single number per point can't express; details in
[`docs/coupling.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/coupling.md) and
[`docs/CONVENTIONS.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/CONVENTIONS.md).

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
  terms composed live (DC Stark, then +BBR, then +gravity),
  cross-checked through the rotor engine on identical trajectories,
  then bridged to the extended-lattice per-site view.
- [`notebooks/11_real_budget_slice.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/11_real_budget_slice.ipynb):
  the closest-to-a-real-experiment demo. One real clock's published
  evaluation, the JILA Sr system, with the covered rows computed from
  published inputs beside the lab's own numbers in a single composed
  pipeline run.
- [`notebooks/12_thermal_environment.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/12_thermal_environment.ipynb):
  the multi-surface thermal environment (E37), for field-deployed
  clocks sitting in a real, non-uniform radiation environment no lab
  shield has engineered flat, quantifying live where a single effective
  temperature stops representing it and closing with a sensitivity band
  built from per-surface sensor readings and uncertainties.
- [`notebooks/13_trapped_ion_quantum_motion.ipynb`](https://github.com/velar-mbr/CliffordClock/blob/main/notebooks/13_trapped_ion_quantum_motion.ipynb):
  the trapped-ion walkthrough, quantum motional states throughout: a
  ground-state ion's ~10 nm wavepacket sampled against a field with real
  curvature, the Coulomb-crystal quadrupole case recapped from notebook 08,
  the motional time-dilation row (E38) reproduced against a published Al+
  evaluation and its per-mode participation-factor correction for the
  two-ion crystal, and the coherent Ramsey fringe visibility a squeezed
  motional state leaves behind (E39), closing with the excess-micromotion
  input channel and a scope statement against the RF-dynamics roadmap.

## Documentation

- [`paper/main.pdf`](https://github.com/velar-mbr/CliffordClock/blob/main/paper/main.pdf): the paper. The physical model,
  the numerical methods, the full validation record including both
  reproducibility cases, and the chamber-scale showcase, in one
  self-contained read.
- [`docs/tutorial.md`](https://github.com/velar-mbr/CliffordClock/blob/main/docs/tutorial.md): start here
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
