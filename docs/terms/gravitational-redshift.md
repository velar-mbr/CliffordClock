# Gravitational-redshift pivot term

E36 in `docs/CONVENTIONS.md`, used together with the extended-lattice
ensemble regime (section 15). This is General Relativity's own
gravitational time dilation, evaluated across a millimetre-scale atomic
sample.

## What it is

A clock higher in a gravitational potential runs faster, by the
weak-field leading term of the metric proper-time ratio. Across a
millimetre-scale optical-lattice sample, the height difference between
the top and bottom of the cloud is large enough that this term produces
a measurable per-site frequency slope, the effect Bothwell et al. (2022)
measured directly inside a single Sr lattice-clock apparatus. At
`g = 9.80665 m/s²`, the slope is `g/c² ≈ 1.0911370e-16` per metre.

## The formula

$$(P-1)_{\text{grav}}(r) = \frac{U(r)}{c^2} = \frac{g\cdot(h(r)-h_{\text{ref}})}{c^2}$$

`g` is the local gravitational acceleration, `h(r)` the atom's height
along the configured "up" direction, `h_ref` the reference height (the
sample center by default), and `c` the speed of light. A higher clock
gives `(P-1)_grav > 0`, matching the convention `(P-1) = Δν/ν₀`.

## The code

```python
# src/cliffordclock/integrator/omega.py::grav_pivot_perturbation
def grav_pivot_perturbation(
    height_m: jnp.ndarray, g_m_s2: float, reference_height_m: float = 0.0
) -> jnp.ndarray:
    height_m = jnp.asarray(height_m, dtype=jnp.float64)
    return g_m_s2 * (height_m - reference_height_m) / SPEED_OF_LIGHT**2
```

`g_m_s2` is a required argument with no internal default, since the
physically correct input at this precision is the lab's own surveyed
local gravity. The real implementation lives in
`src/cliffordclock/integrator/omega.py::grav_pivot_perturbation`.

## How it is checked

The extended-lattice ensemble regime (`ensemble.regime:
lattice_extended`) with this pivot term reconstructs Bothwell et al.'s
(Nature 602, 420 (2022)) published mm-scale gravitational-redshift
measurement. The real per-site pipeline, run over about 5900 sites at
their sample geometry and their surveyed local gravity `g = 9.796 m/s²`,
predicts a slope of `-1.0900e-19/mm`, against their two corrected
measurements of `-9.8(2.3)e-20/mm` and `-1.28(27)e-19/mm`. Both land
`kpi_verdict = "MET"`, at `0.48σ` and `0.70σ` respectively, with the
prediction between them (`docs/validation.md`, `benchmarks/RESULTS.md`).

## Sources

- T. Bothwell, C. J. Kennedy, A. Aeppli, D. Kedar, J. M. Robinson,
  E. Oelker, A. Staron, J. Ye, "Resolving the gravitational redshift
  across a millimetre-scale atomic sample," Nature 602, 420-424 (2022),
  arXiv:2109.12238.
- D. van Westrum, NOAA Technical Memorandum NOS NGS-77 (2019), the
  surveyed local gravity Bothwell et al. cite for their apparatus.
