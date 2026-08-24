# Tutorial: your first hour with CliffordClock

This page is the fastest path from "nothing installed" to "I understand
what this tool just told me, and I trust it enough to point it at my own
trap." It walks three things, in order: install, the physical worked
example (with every line of output explained), and the validation example
(which looks similar but means something different). It ends with where
to go to use your own field data.

Every command below was run in a fresh virtual environment while writing
this page; the numbers shown are real output.

## 1. Install

```bash
pip install cliffordclock
```

This installs the library and the `cliffordclock` command-line tool.
The examples and notebooks this tutorial runs are stored in the
repository, so grab a clone alongside:

```bash
git clone https://github.com/velar-mbr/CliffordClock.git
cd CliffordClock
```

(Working on the code itself? Use `CONTRIBUTING.md`'s editable install
in place of the pip install.) Confirm it worked:

```bash
cliffordclock version
```

```
0.1.0
```

## 2. Run the physical example

This is the tool doing its actual job: predicting a clock's fractional
frequency shift from a physically-shaped stray electric field, at a real
1-second interrogation time.

```bash
cliffordclock run examples/realistic_lattice_sr87.yaml --output-dir /tmp/cliffordclock_out
```

```
CliffordClock run summary
  species:                Sr87
  ensemble:               lattice_fast_path (M=512)
  interrogation time:     1.000000e+00 s
  mean fractional shift:  -7.723399e-19 +/- 6.771e-24 (SEM)
  T2*:                    4.552836e+01 s
  notes:                  ...(coupling provenance and scope caveats, see below)...
  report:                 /tmp/cliffordclock_out/report.json
  line profile:           /tmp/cliffordclock_out/line_profile.csv
```

### What the scenario is

`examples/realistic_lattice_sr87.yaml` models a strontium-87 optical-
lattice clock sitting near a small residual stray electric field: a
uniform bias plus several patches of stray charge on nearby in-vacuum
surfaces (mirrors, viewports). The field magnitudes are chosen to bracket
a documented, real stray-charge event measured at a Sr optical-lattice
clock (SYRTE; [Lodewyck et al., IEEE Trans. UFFC 59, 411 (2012)](https://arxiv.org/abs/1108.4320)).
The field itself is loaded from a CSV export
(`examples/patch_field_sr87.csv`) the same way this tool would ingest a
field exported from your own FEA/CAD simulation. See
[`byof-guide.md`](byof-guide.md) for that path.

A stray electric field shifts a clock transition's frequency through the
**DC-Stark effect**: the field slightly perturbs the atom's energy levels,
proportional to the field strength squared, weighted by the transition's
differential polarizability (a published, species-specific number). This
tool evaluates that shift directly from the field and a literature-cited
polarizability: no fitting, no tuned parameters.

### What every line of output means

- **species**: which atom/ion the clock uses (`Sr87` here); this
  selects the literature polarizability used below.
- **ensemble**: how the atom's motional state is represented, and how
  many samples (`M`) that representation uses. `lattice_fast_path` means
  the atoms sit at fixed, quantized positions in an optical lattice (not
  flying around a trap), and the tool used its fast, exact evaluation
  path for that case: no time-stepping needed, so it costs the same
  whether you ask for a microsecond or an hour of interrogation.
- **interrogation time**: how long the clock probes the atoms
  for, in seconds. `1.000000e+00 s` is a real, physically meaningful
  Ramsey/Rabi interrogation time.
- **mean fractional shift**: the predicted shift, `Δν/ν₀`
  (dimensionless: a fraction of the clock's transition frequency), caused
  by the stray field. `-7.723399e-19` means the clock's frequency reads
  about 0.77 parts in 10¹⁸ *low* because of this field, comfortably
  inside the 1e-18-level regime this tool targets.
- **`±` (SEM)**: the standard error of that mean, from spreading the
  calculation over the ensemble's `M` samples. `6.771e-24` is about four
  orders of magnitude smaller than the shift itself, so the reported
  number is not noise-dominated: the ensemble size and field structure
  resolve the shift cleanly.
- **T2\***: the *inhomogeneous dephasing time*: how long it takes for
  the spread in individual atoms' shifts (because they sit at slightly
  different positions, seeing slightly different field strengths) to wash
  out coherence across the ensemble. `45.5 s` here is long compared to
  the 1 s interrogation, meaning the field is close enough to uniform
  across this trap that dephasing is not a practical concern for this
  scenario.
- **notes**: free-text provenance: which coupling coefficient was used,
  where it's cited from, and a caveat that this particular calculation
  mode reports the field-induced (Stark) shift only. The separate
  motional (Doppler) shift is covered by the validation example's
  secular mode (see `docs/coupling.md` and `docs/validation.md` KA4).
  Safe to skip on a first read; useful when you need to cite exactly
  where a number came from.
- **report / line profile**: the two files written: `report.json` (the
  numbers above plus metadata, machine-readable, see
  [`report-schema.md`](report-schema.md)) and `line_profile.csv` (the
  predicted spectral line shape).

## 3. Run the validation example

```bash
cliffordclock run examples/quadrupole_classical.yaml --output-dir /tmp/cliffordclock_out
```

```
CliffordClock run summary
  species:                Sr87
  ensemble:               classical_direct (M=200)
  interrogation time:     1.288089e-18 s
  mean fractional shift:  +4.699866e-14 +/- 4.853e-14 (SEM)
  T2*:                    2.654232e-09 s
  notes:                  ...
  note: validation-scale run (not a physical interrogation time) -- see docs/timescales.md
  report:                 /tmp/cliffordclock_out/report.json
  line profile:           /tmp/cliffordclock_out/line_profile.csv
```

This looks like the same kind of report, but it means something
different, and the tool tells you so directly: the extra
**`note: validation-scale run`** line. Two things give it away even
without that line: the interrogation time (`1.29e-18 s`, a fraction of
an attosecond, far shorter than any real clock interrogation) and the
mean shift's own error bar (`SEM` is larger than the shift itself, i.e.
this number is statistical noise).

This example exists to exercise the tool's integrator directly, one
Compton-scale step at a time (the fundamental timescale the underlying
physics is formulated in). The toy case validates that stepping
mechanism; predicting a real clock's frequency shift needs the full
pipeline, with its registry-sourced species data and coupling
coefficients, which section 2's example supplies. It is useful for
development and cross-checking; the physical example in section 2 is
what to cite when you need a number.
Whenever `cliffordclock run`'s resolved interrogation time is below one nanosecond,
you'll see this note; treat it as a signal to check
[`docs/timescales.md`](timescales.md) for why real interrogation times
are normally cheap to compute directly, and why this particular example
deliberately doesn't use one.

## 4. Bring your own field

Once you trust what these two examples are telling you, the next step is
pointing the tool at your own trap: export a field grid from your own
FEA/CAD tool as a CSV, adapt `examples/realistic_lattice_sr87.yaml` to
load it, and set your species and trap parameters.

**[`byof-guide.md`](byof-guide.md)** is the full guide: CSV format
contract, grid-spacing guidance, what the smoother warnings mean, and
current limitations.
**[`notebooks/04_bring_your_own_field.ipynb`](../notebooks/04_bring_your_own_field.ipynb)**
is the copy-and-edit starting point, with "REPLACE THIS CELL" markers at
each of the four things you need to change.

## 5. Loading a COMSOL export (optional)

If your FEA tool is COMSOL, you can skip the CSV conversion entirely and
point `config.yaml` at your native export:

```bash
cliffordclock run examples/comsol_electrode_sr87.yaml --output-dir /tmp/cliffordclock_out
```

```
CliffordClock run summary
  species:                Sr87
  ensemble:               lattice_fast_path (M=512)
  interrogation time:     1.000000e+00 s
  mean fractional shift:  -1.109246e-14 +/- 1.148e-25 (SEM)
  T2*:                    2.686396e+03 s
  notes:                  ...(coupling provenance and scope caveats, see below)...
  report:                 /tmp/cliffordclock_out/report.json
  line profile:           /tmp/cliffordclock_out/line_profile.csv
```

`examples/comsol_electrode_sr87.yaml` loads
`examples/fd_electrode_field.txt`: a from-scratch finite-difference
solve of a two-electrode capacitor, written out in COMSOL's native
"Spreadsheet" export format (`examples/generate_fd_electrode_field.py`),
directly via `field: {comsol: examples/fd_electrode_field.txt}`, the same
`config.yaml` key as `field: {csv: ...}` above but for COMSOL's native
format. The reported shift (`-1.109246e-14`) matches a documented
back-of-envelope estimate (the file's own header comment derives it from
the DC-Stark formula and the solve's domain-center field, ~1244 V/m) to
well under 1%.

**[`byof-guide.md`](byof-guide.md)**'s "COMSOL exports" section has the
full export-dialog settings (which "Data format" to pick, unit handling,
the `expression_prefix` option for renamed physics interfaces) and current
format limitations (Spreadsheet text export only, no VTK/mesh formats).

## Where to go next

- [`docs/validation.md`](validation.md): the full validation record:
  every case this tool has been checked against, with sources and
  measured agreement.
- [`docs/coupling.md`](coupling.md): the DC-Stark physics in full, and
  its Python API.
- [`docs/cli.md`](cli.md): the complete `config.yaml` schema.
- [`docs/index.md`](index.md): the full documentation map.
