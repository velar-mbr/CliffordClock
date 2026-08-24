# Partner case: expected-value schema

This is the "other half" of a partner benchmark case: `config_template.yaml`
+ `field_grid_template.csv` describe the *input* (trap, species, field);
this file describes the *comparison target*: the collaborator's measured
shift the tool's prediction gets checked against, plus enough provenance
that the resulting case is auditable the way every existing case in
`docs/validation.md` / `benchmarks/RESULTS.md` is.

The shape mirrors `benchmarks/loaders.py`'s `SystematicShiftEntry`
dataclass (the structure the WP10 JILA case already uses); a partner
case is the same shape, just with `comparable: true` and a real residual
computed, which no current public-source row can legitimately claim (see
`benchmarks/MAPPING.md`).

## `partner_case_expected.yaml`: copy and fill in

```yaml
# benchmarks/beta_case_<short-name>/expected.yaml

case_id: <short-name>              # e.g. "syrte_2027": matches the
                                    # benchmarks/beta_case_<short-name>/
                                    # directory name.

collaborator: <PLACEHOLDER>        # Institution/group name, or an
                                    # anonymized label ("Lab A") if the
                                    # collaborator would rather not be
                                    # named in this repo yet, their
                                    # choice; confirm with them before
                                    # committing a real name.

species: <PLACEHOLDER e.g. Sr87>   # Must match config.yaml's `species`.

measured_shift_fractional: <PLACEHOLDER>       # e.g. -7.2e-19
measured_uncertainty_fractional: <PLACEHOLDER> # 1-sigma, unless
                                                # `uncertainty_is_upper_bound`
uncertainty_is_upper_bound: false  # true if the collaborator's number is
                                    # a bound ("shift no larger than X"),
                                    # not a symmetric 1-sigma value:
                                    # same convention as
                                    # benchmarks/loaders.py's
                                    # SystematicShiftEntry.

measurement_method: >-
  <PLACEHOLDER: how the collaborator measured this, e.g. "external
  calibration electrode, 1 kV -> 26 Hz sensitivity, field nulled via UV
  discharge" (Lodewyck-style) or "quadrant-electrode field alternation,
  clock-frequency lock-in" (JILA-style). This is what makes the number
  auditable.>

field_source: >-
  <PLACEHOLDER: how the field in field_grid.csv was characterized,
  e.g. "FEA simulation of trapped charge on build-up-cavity mirrors,
  calibrated against the external-electrode measurement" or "in-situ
  Rydberg-atom field mapping". This is the field-characterization
  provenance the current public sources are missing
  (benchmarks/MAPPING.md): it's the single most important field in
  this whole schema.>

citation_or_reference: >-
  <PLACEHOLDER: a paper citation if published, or "unpublished, shared
  under the CliffordClock beta program, <date>" if not. Either way, this
  is the provenance trail a skeptical reader (docs/validation.md's
  target audience) will want to check.>

date_received: <PLACEHOLDER YYYY-MM-DD>

notes: >-
  <PLACEHOLDER: anything else relevant, known caveats on the field
  characterization, whether the ensemble/temperature assumptions in
  config.yaml are the collaborator's actual apparatus parameters or an
  approximation, etc. A stated caveat here is exactly the kind of
  content docs/validation.md's existing cases already carry (e.g. KA1-4's
  stated tolerances and measured agreement, beyond a bare "PASS").>
```

## Computing and reporting the residual

Once `cliffordclock run` (or the pipeline API) has produced `report.json`'s
`mean_fractional_shift`:

```
residual = report["mean_fractional_shift"] - measured_shift_fractional
```

Report the residual **and** whether it falls inside
`measured_uncertainty_fractional` (or, for a Monte Carlo ensemble case,
inside some stated multiple of `report["shift_std_error"]`, the same
convention `docs/validation.md`'s KA4 uses: `0.32σ`, stated explicitly,
beyond a bare "passed"). If the residual is *larger* than the stated
uncertainty, report that too, with the same reported-as-found discipline
`benchmarks/RESULTS.md` uses for WP10's negative result: a mismatch is
real information (a physics-scope gap this engine doesn't model yet, a
field-characterization uncertainty larger than assumed, or a modeling
assumption, e.g. `ensemble.regime`/`motional_n`/temperature, that
doesn't match the real apparatus). Report it exactly as it comes out.

## Turning this into a real case

This template stays independent of `benchmarks/run_benchmarks.py`: that
script's current job (classifying the WP10 public sources as found) is
unrelated to a partner case, and no wiring should be built for data that
doesn't exist yet. Once real data arrives, either:

- Keep it simple: a `benchmarks/beta_case_<name>/` directory (config +
  field CSV + `expected.yaml` per this schema) plus a short markdown
  writeup in the same style as `benchmarks/RESULTS.md`, run manually via
  `cliffordclock run` and reported by hand, sufficient for the first case.
- Or, if/when there are enough partner cases to want automation, extend
  `benchmarks/loaders.py`/`run_benchmarks.py` to also load
  `beta_case_*/expected.yaml` files and compute residuals the same way
  the JILA/NIST loaders parse their sources, a natural follow-up WP once
  real data exists to build and test it against.
