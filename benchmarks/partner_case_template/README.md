# Partner benchmark case: template

**What this is:** a template showing exactly what data a beta
collaborator needs to send (or run locally and report back on) to become
a real, comparable benchmark case, the thing `benchmarks/RESULTS.md`
found none of the current public sources could supply (see
`benchmarks/MAPPING.md`: the DC-Stark row is in this engine's scope, but
every currently-authorized public source reports a measured shift
without the independent field magnitude that produced it, and solving
backwards for that field would make the field magnitude a fitted
parameter chosen to match the measured shift, which this project's
benchmarks never allow). A collaborator supplying *both* halves (a
characterized field and the shift it produced) closes exactly that gap.

This directory is the template `benchmarks/beta_case_<name>/` gets
copied from once real data arrives (see "Turning this into a real case"
below).

## What plugs in where

| You provide | Goes in | Format |
|---|---|---|
| A characterized stray-field magnitude or map | `field_grid_template.csv` (replace with your data) | `docs/byof-guide.md` CSV contract: header `x,y,z,Ex,Ey,Ez`, positions in **meters**, field in **V/m**. A single-point "field" (uniform, no map) is fine too: see the note in the CSV template. |
| Your trap/species parameters | `config_template.yaml` | `species`, `trap.omega_xyz`/`center`: see `docs/cli.md` for the full schema. |
| Your measured DC-Stark shift (the number you compare the tool's prediction against) | `expected_data_schema.md`'s `partner_case_expected.yaml` template | fractional shift, uncertainty (1σ or bound), and provenance: mirrors `benchmarks/loaders.py`'s `SystematicShiftEntry` shape, the same structure the WP10 JILA case uses. |

## Steps (what "within a day" looks like)

1. Copy this directory: `cp -r benchmarks/partner_case_template
   benchmarks/beta_case_<short-name>` (e.g. `beta_case_syrte`).
2. Replace `field_grid_template.csv` with the collaborator's actual field
   export (or a synthetic `field.synthetic` block in the config if the
   field is well described by a closed form: uniform/gradient/quadrupole
   are already built in, see `docs/cli.md`).
3. Fill in `config_template.yaml`'s placeholders (`species`,
   `trap.omega_xyz`/`center`, `field.csv` path, `coupling.type: stark_dc`:
   no override needed for `Sr87`/`Yb171`).
4. Fill in `expected_data_schema.md`'s `partner_case_expected.yaml`
   template with the collaborator's measured shift, uncertainty, and a
   one-line provenance note (what was measured, how, and the citation or
   internal reference if unpublished).
5. Run `cliffordclock run benchmarks/beta_case_<name>/config.yaml --output-dir
   /tmp/beta_case_<name>` and read `mean_fractional_shift` from
   `report.json`.
6. Compute the residual (`predicted - measured`) and compare against the
   measured uncertainty, the same way `docs/validation.md`'s KA1-4 cases
   report "measured agreement": this is the step no current public
   source lets `benchmarks/run_benchmarks.py` reach (see
   `benchmarks/MAPPING.md`'s explanation of why every current row's
   `comparable` field is `False`).
7. Write the result up the way `benchmarks/RESULTS.md` documents WP10's
   cases: what was compared, the formula/reference, the tolerance, and
   the measured agreement, reported as found, including if the residual is
   larger than hoped. A "this doesn't match yet" result is real,
   reportable information (physics gap, field-characterization
   uncertainty, or a modeling assumption that doesn't hold for this
   apparatus): reporting it faithfully is the collaboration succeeding,
   exactly as designed.

## What this case will NOT need

- No business or internal-process information of any kind: this is a
  physics comparison (field in, shift out).
- No data has to leave the collaborator's machine if they'd rather run
  `cliffordclock run` themselves and send back only the residual: a data-stays-
  local option is always available and does not depend on this template.
- No change to `src/cliffordclock`: a real partner case is config +
  data, run through the existing pipeline exactly as shipped.

## See also

- `docs/byof-guide.md`: the CSV contract, grid-spacing guidance, and the
  smoothing parameter you'll likely want `> 0` for a real (noisy)
  measured field, unlike this repo's noiseless synthetic examples.
- `examples/realistic_lattice_sr87.yaml`: a complete, runnable config in
  the same shape `config_template.yaml` below is derived from.
- `benchmarks/MAPPING.md` / `benchmarks/RESULTS.md`: the classification
  methodology and labeling discipline (no tuned parameters, ever) a real
  partner case must also follow.
- `benchmarks/loaders.py`: the typed structures the WP10 public-source
  cases use; `expected_data_schema.md` mirrors the same shape for a
  partner-supplied case.
