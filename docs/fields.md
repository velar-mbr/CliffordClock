# Field importer & smoother

`cliffordclock.fields` ingests a CAD/FEA-exported E-field grid and turns it
into a smooth, analytically differentiable `E(r)`, `∇E(r)` evaluator for the
rest of the pipeline (CONVENTIONS.md §4, E11-E13).

## CSV format

```
x,y,z,Ex,Ey,Ez
-0.001,-0.001,-0.001,123.4,-56.7,8.9
...
```

- Header row required; the six columns may appear in any order (looked up
  by name).
- Positions in **meters**, field components in **V/m** (CONVENTIONS.md §10).
- One row per sample point. Points forming the complete Cartesian product
  of three axes are detected as a **regular grid**; anything else (a
  scattered point cloud, or a partial/malformed attempt at a grid) is
  either accepted as scattered data or rejected with an informative
  `ValueError`: see `load_field_csv` docstring for the exact error cases
  (missing column, non-finite value, duplicate point, short/empty file).

```python
from cliffordclock.fields import load_field_csv

grid = load_field_csv("my_export.csv")
grid.regular  # True if it's a full x/y/z product grid
grid.shape  # (nx, ny, nz) if regular, else None
```

## COMSOL format

`load_field_comsol` reads the header/column-header block COMSOL writes for
`File > Export > Data` with the **Spreadsheet** file type
(`.txt`/`.csv`/`.dat`): a `%`-prefixed metadata block, a final
`%`-prefixed column-header line, then one data row per sample point:

```
% Model:      my_model.mph
% Version:    COMSOL 6.0.0.318
% Date:       Jan 1 2026, 00:00
% Dimension:  3
% Nodes:      1000
% Expressions: 3
% Description: Electric field
% Length unit: mm
% X    Y    Z    es.Ex (V/m)   es.Ey (V/m)   es.Ez (V/m)
1.0    2.0  3.0  123.4         -56.7         8.9
...
```

```python
from cliffordclock.fields import load_field_comsol

grid = load_field_comsol("my_export.txt")
grid = load_field_comsol("my_export.txt", expression_prefix="es")  # default
```

- **Only the `Spreadsheet` export type is supported**: COMSOL's
  `Sectionwise` format (and any file whose first line does not start with
  `%`) is rejected with a clear error naming the problem. Whitespace- and
  comma-delimited (`.csv`-exported) variants of the Spreadsheet format are
  both accepted (auto-detected from the column-header line).
- **Length units** `m`, `mm`, `cm` are converted to meters; **field-component
  units** `V/m`, `kV/m`, `V/cm` are converted to V/m. Any other unit string
  raises a `ValueError` naming it; there is no silent misconversion.
- **Field columns** are matched by name: `{expression_prefix}.Ex`,
  `{expression_prefix}.Ey`, `{expression_prefix}.Ez` (default prefix `es`,
  COMSOL's built-in Electrostatics physics interface). Pass a different
  `expression_prefix` if your model renamed the interface. Extra
  expression columns beyond the three field components (e.g. `es.normE`)
  are ignored, matching the CSV loader's look-up-by-name convention,
  unless a `% Expressions:` count line is present and disagrees with the
  actual column count, which raises.
- **Only 3D, non-parameterized, real-valued exports are in scope**: a
  `% Dimension:` line other than `3`, a column-header line
  containing an `@ param=value` tag (a parameter-sweep/multi-study
  export), or a complex-valued (frequency-domain) data cell all raise a
  descriptive `ValueError` naming the problem, so none of these get
  silently mis-parsed or truncated. `docs/byof-guide.md`'s "COMSOL exports" section has the exact
  export-dialog settings that keep you in this supported scope.
- **Header cross-checks:** if present, `% Nodes:` and `% Expressions:`
  are checked against the actual data-row and column counts and raise on
  mismatch.
- Same validation stack as `load_field_csv`: non-finite values, duplicate
  points, and near-duplicate points all raise/warn exactly as described
  above.
- `FieldGrid.metadata` additionally carries the header's `model`,
  `version`, `date`, `description`, `length_unit`, `field_units`, and
  `expression_prefix`.
- **Wired into `config.yaml`/the CLI**: `field: {comsol: path}` loads a
  COMSOL export and fits a `FieldSmoother` the same way `field: {csv:
  path}` does (`field.smoothing` applies to both; `field.expression_prefix`,
  optional, default `"es"`, is forwarded to `load_field_comsol`): see
  `docs/cli.md`'s "Field sources" section and
  `examples/comsol_electrode_sr87.yaml` for a runnable example. The Python
  API above still works directly, e.g. in a notebook or a small adapter
  script.

## Fitting and evaluating a smoother

The field is decomposed as `E = E_0 + δE` (E11): `E_0` is a degree-1
(uniform + linear) analytical baseline fitted by least squares, and `δE` is
the residual fitted by a thin-plate-spline RBF (`φ(r) = r² ln r`, E12).
Splitting out the baseline keeps `δE` small, which keeps the RBF fit and
its gradients well-conditioned for the 1e-18-level shift arithmetic
downstream.

```python
from cliffordclock.fields import FieldSmoother

smoother = FieldSmoother.fit(grid, method="auto", smoothing=0.0)
E, grad_E = smoother.evaluate(pos)  # pos: (N, 3) m -> E: (N, 3) V/m, grad_E: (N, 3, 3) V/m^2
```

- `grad_E[..., i, j] = ∂_i E_j` (E13), computed by `jax.jacfwd` of the same
  function that produces `E`, so the two are exactly consistent and
  `evaluate` is C^∞ in the interpolant.
- `evaluate` is pure JAX: it works under `jax.jit`, `jax.vmap`, and as a
  leaf inside a caller's `jax.grad`/`jax.jacfwd` (e.g. the rotor
  integrator).
- `smoothing > 0` adds Tikhonov regularization to the RBF fit, trading
  exact interpolation for robustness to noisy input data.
- Only `method="rbf"` (or `"auto"`, which currently always resolves to it)
  is implemented; tensor B-splines are documented future work.
- The fit is capped at `smoother.MAX_FIT_POINTS` (~20,000) points: the RBF
  fit solves a dense `(N, N)` linear system, an O(N³) operation.
- Querying `evaluate` outside the fit data's bounding box raises an
  `OutOfBoundsWarning` (values are still returned, extrapolated).

## Synthetic test fields

`cliffordclock.fields.synthetic` provides closed-form fields with
hand-derived exact gradients (uniform, constant-gradient, spherical
quadrupole, Gaussian bump) plus a `sample_on_grid` helper, used throughout
the test suite as ground truth independent of the smoother's own autodiff.
