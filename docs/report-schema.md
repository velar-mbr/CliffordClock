# Metrology report schema

`cliffordclock.analytics.report` writes two machine-readable
outputs from a `MetrologyReport`: a schema-versioned JSON report and a
companion line-profile CSV. Formulas are `docs/CONVENTIONS.md` E23
(mean fractional shift), E25 (phase variance), E27 (T2*); see
`cliffordclock.analytics.stats` for the implementations.

## `write_json` output

Top-level object, schema version `"report_schema": "1.0"`. Key order is
stable (matches the field order below; do not rely on alphabetical
order). Floats are written with full precision (`repr`-roundtrip safe;
Python's `json` encoder already guarantees this for finite floats, so no
extra handling is needed on read).

**Null convention.** Non-finite float fields are written as JSON `null`,
not a bare `NaN`/`Infinity` token; neither is valid JSON (RFC 8259
section 6 permits only finite numbers); `write_json` passes
`allow_nan=False` to `json.dump` so any non-finite value that reaches it
without first being converted to `null` raises loudly instead of emitting
invalid JSON. Two cases currently produce a `null`:

1. **Undefined** (`shift_std_error` and `t2_star_s` together): single-atom
   M=1 ensembles: their E25 sample-variance statistics need >= 2
   effective samples, see `cliffordclock.pipeline._build_report`. In
   memory these are `float("nan")`.
2. **Infinite T2*** (`t2_star_s` only): the ensemble phase variance is
   exactly zero (every atom accumulated the identical phase, e.g. a
   lattice ensemble in a spatially uniform field), so E27's
   `sigma_Phi -> 0+` limit gives `T2* = +inf`, a *defined*, physically
   meaningful value (no inhomogeneous dephasing) that JSON simply cannot
   represent. In memory this is `float("inf")`
   (`cliffordclock.analytics.stats.dephasing_time_t2star`).

On read, `uncertainty_notes` disambiguates: `build_report` appends a
"zero ensemble phase variance: T2* is infinite ..." note in case 2, and
`cliffordclock.pipeline._build_report` appends a "single-atom ensemble
(M=1) ..." note in case 1: recover the in-memory convention by mapping
`null` to `float("nan")` or `float("inf")` accordingly if your consumer
needs to. Every report float field *other* than
`shift_std_error`/`t2_star_s` is always finite and round-trips
bit-identically; only those two can ever be `null`.

| field | type | units | notes |
|---|---|---|---|
| `report_schema` | string | n/a | this schema's version, currently `"1.0"` |
| `conventions_version` | string | n/a | `docs/CONVENTIONS.md` version the formulas trace to (currently `"1.1.0"`) |
| `package_version` | string | n/a | `cliffordclock` package version (`importlib.metadata`) |
| `generated_at_utc` | string | n/a | ISO-8601 UTC timestamp |
| `config_hash` | string or `null` | n/a | caller-supplied hash of the input configuration; `null` if not supplied (populated by the pipeline façade, not by this module itself) |
| `species_name` | string | n/a | atomic species registry name, e.g. `"Sr87"` |
| `ensemble_type` | string | n/a | free-text ensemble/regime label, e.g. `"classical_direct"`, `"classical_secular_average"`, `"lattice_fast_path"`, `"lattice_worldline_crosscheck"`, `"lattice_extended_fast_path"`, or `"lattice_extended_worldline_crosscheck"` (WP22; not a closed enum: see `docs/timescales.md` for what each `integration.mode` produces) |
| `ensemble_size` | integer | n/a | number of atoms/nodes `M` |
| `interrogation_time_s` | float | seconds | interrogation time `T` |
| `mean_fractional_shift` | float | dimensionless | weighted `<Δν/ν₀>` (E23) |
| `shift_std_error` | float or `null` | dimensionless | standard error of `mean_fractional_shift`; `null` when undefined (M=1): see the null convention above |
| `t2_star_s` | float or `null` | seconds | inhomogeneous dephasing time `T2*` (E27); `null` when undefined (M=1) or infinite (zero phase variance): see the null convention above |
| `uncertainty_notes` | string | n/a | free-text systematic-uncertainty notes; no budget model (out of scope for this module) |

## `write_line_profile_csv` output

Two columns, no systematic uncertainty encoded. First line is a
`#`-prefixed comment header (skipped by `numpy.loadtxt`'s default
`comments="#"`, or droppable manually via the stdlib `csv` module); data
rows are `repr`-precision floats.

| column | units | notes |
|---|---|---|
| `frequency_offset_hz` | hertz | offset from the clock's nominal frequency (`cliffordclock.analytics.stats.line_profile`, E28) |
| `amplitude` | dimensionless | normalized spectral amplitude, `\|FFT(C)\| / T` |

Example:

```
# frequency_offset_hz,amplitude
-2.5,0.1
-1.0,0.4
0.0,1.0
1.0,0.4
2.5,0.1
```

## Future work

Allan deviation / overlapping Allan variance is deliberately out of scope
for schema 1.0 (single-interrogation statistics only); it is the natural
next statistic when multi-shot simulation lands (post-MVP), and will bump
the schema version when added.
