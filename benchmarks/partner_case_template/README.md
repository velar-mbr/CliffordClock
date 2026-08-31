# Partner benchmark case: template

This directory is a template for a partner benchmark case: the data a
collaborator sends, or runs locally and reports on, to become a real,
comparable benchmark. `benchmarks/RESULTS.md` found that no current
public source can supply such a case for the DC-Stark row. Every
authorized public source reports a measured shift without the
independent field magnitude that produced it. Solving backwards for
that field would turn the field magnitude into a fitted parameter
chosen to match the measured shift, and this project's benchmarks
never allow a tuned parameter (`benchmarks/MAPPING.md` documents the
survey). A collaborator supplying both halves, a characterized field
and the shift it produced, closes that gap.

Once real data arrives, this directory is copied to
`benchmarks/beta_case_<name>/` and filled in; the "Steps" section
below walks it.

## What plugs in where

| You provide | Goes in | Format |
|---|---|---|
| A characterized stray-field magnitude or map | `field_grid_template.csv` (replace with your data) | `docs/byof-guide.md` CSV contract: header `x,y,z,Ex,Ey,Ez`, positions in **meters**, field in **V/m**. A single-point uniform field works too; see the note in the CSV template. |
| Your trap/species parameters | `config_template.yaml` | `species`, `trap.omega_xyz`/`center`: see `docs/cli.md` for the full schema. |
| Your measured DC-Stark shift (the number you compare the tool's prediction against) | `expected_data_schema.md`'s `partner_case_expected.yaml` template | fractional shift, uncertainty (1σ or bound), and provenance: mirrors `benchmarks/loaders.py`'s `SystematicShiftEntry` shape, the same structure the WP10 JILA case uses. |

## Steps (about a day end to end)

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
6. Compute the residual (`predicted - measured`) and compare it against
   the measured uncertainty, the way `docs/validation.md`'s KA1-4 cases
   report measured agreement. This comparison is the step no current
   public source lets `benchmarks/run_benchmarks.py` reach;
   `benchmarks/MAPPING.md` explains why every current row's
   `comparable` field is `False`.
7. Write the result up the way `benchmarks/RESULTS.md` documents WP10's
   cases: what was compared, the formula or reference, the tolerance,
   and the measured agreement, reported as found. A residual larger
   than the measured uncertainty is a reportable result in its own
   right: it points to a physics gap, a field-characterization
   uncertainty, or a modeling assumption that fails for this apparatus.

## Scope and privacy

- The comparison uses physics data only, a field in and a shift out.
  Business and internal-process information stays out of it.
- Data can stay on the collaborator's machine. They run
  `cliffordclock run` themselves and send back only the residual; that
  option is always available and does not depend on this template.
- A real partner case is config plus data, run through the existing
  pipeline as shipped. `src/cliffordclock` is unchanged.

## See also

- `docs/byof-guide.md`: the CSV contract, grid-spacing guidance, and the
  smoothing parameter you'll likely want `> 0` for a real (noisy)
  measured field, unlike this repo's noiseless synthetic examples.
- `examples/realistic_lattice_sr87.yaml`: a complete, runnable config in
  the same shape `config_template.yaml` is derived from.
- `benchmarks/MAPPING.md` / `benchmarks/RESULTS.md`: the classification
  methodology and labeling discipline (no tuned parameters, ever) a real
  partner case must also follow.
- `benchmarks/loaders.py`: the typed structures the WP10 public-source
  cases use; `expected_data_schema.md` mirrors the same shape for a
  partner-supplied case.
