# Blackbody-radiation shift

E32/E33 (single-temperature) and E37 (multi-surface thermal environment)
in `docs/CONVENTIONS.md`. This is the shift from the thermal radiation
field every clock sits in, the dominant systematic in real optical-
lattice clocks.

## What it is

Every clock chamber radiates like a blackbody at some temperature, and
that radiation field polarizes the clock states the same way a static
field does, so it shifts the transition frequency. The static part of
the shift scales as `T⁴`, and a per-species dynamic correction adds
higher powers of `T` fit to the exact Planck-weighted integral. At
`T = 300 K`, the shift is `-5.319504e-15` for Sr-87 and `-2.464643e-15`
for Yb-171. Outside a shielded lab, a clock's surroundings are rarely one
uniform temperature. E37 replaces the single `T` with a set of surfaces,
each with its own weight and temperature, and the resulting mismatch
against a single-temperature model crosses `1e-18` at an 11 K spread
across two surfaces and `1e-17` by 35 K.

## The formula

$$(P-1)_{\text{BBR}} = \frac{\Delta\nu_{\text{stat}}\cdot(T/T_0)^4 + \Delta\nu_{\text{dyn}}(T)}{\nu_0}, \quad T_0 = 300\text{ K}$$

`Δν_stat` and the per-species dynamic polynomial `Δν_dyn(T) =
Σ c_n(T/T₀)ⁿ` come from the species registry. For a multi-surface
environment (E37), each `(T/T₀)ⁿ` power is replaced by a weighted moment
`M_n = Σ_i w_eff_i(T_i/T_0)ⁿ` over the `N` surfaces, each carrying its own
solid-angle weight `w_i` (with `Σw_i = 1`):

$$(P-1)_{\text{BBR}} = \frac{\Delta\nu_{\text{stat}}\cdot M_4 + \sum_n c_n M_n}{\nu_0}$$

`w_eff_i = w_i/(w_i+(1-w_i)·ε_i)` is the emissivity/aperture-corrected
weight (Nosske et al. formula, `docs/CONVENTIONS.md` section 13); it
reduces to the raw `w_i` when every surface is a blackbody, `ε_i = 1`
(emissivity handling absent). A uniform, single-surface environment makes
every `M_n` equal to `(T_1/T_0)ⁿ`, so this reduces to the
single-temperature formula term for term.

## The code

```python
# src/cliffordclock/integrator/omega.py::bbr_pivot_perturbation
def bbr_pivot_perturbation(temperature_k: float, species: Species) -> float:
    return bbr_environment_pivot_perturbation(
        (RadiationSurface(name="uniform", weight=1.0, temperature_k=temperature_k),),
        species,
    )
```

The single-temperature function is implemented as the one-surface case of
the multi-surface function, so the two paths share one code path with no
separate formula to keep in sync. The real implementation lives in
`src/cliffordclock/integrator/omega.py::bbr_environment_pivot_perturbation`.

## How it is checked

KA5 (a known-answer test, the repo's term for a check against a
published anchor value) checks the E32 closed-form polynomial at
`T = 300 K` and `T = 250 K`
against a 50-digit `decimal` reference for both species, at `rtol=1e-12`
(`docs/validation.md`). Separately, the engine's real BBR functions are
run at JILA's own published operating temperature, `T = 293.282(4) K`.
That run reproduces JILA's own published BBR row (arXiv:2403.10664
Table I, `-4.84172(73)e-15`) with predicted `-4.841743e-15`, residual
`-2.251e-20`, `kpi_verdict = "MET"`. This case is labeled arithmetic
reproduction, weaker than a reproducibility check: it reproduces a
published value from the paper's own quoted inputs (their `T` and their
coefficients run through the same formula), while a reproducibility case
checks a prediction against an independently measured result. The
registry's own polynomial is anchored to this same JILA value
(`docs/validation.md`, `benchmarks/RESULTS.md`).

## Sources

- Lisdat et al., Phys. Rev. Research 3, L042036 (2021) (dynamic-term
  Planck-weighted fit, shape).
- Aeppli et al., Phys. Rev. Lett. 133, 023401 (2024), arXiv:2403.10664
  (Sr-87 anchor value and reproduction target).
- T. Middelmann, S. Falke, C. Lisdat, U. Sterr, Phys. Rev. Lett. 109,
  263004 (2012) (Sr-87 static term).
- Hassan et al., arXiv:2506.05304 (2025) (Yb-171 static and dynamic
  terms).
- Beloy et al., Phys. Rev. Lett. 113, 260801 (2014) (Yb-171 dynamic
  coefficient `ν_dyn,8`, derived from their `η₂` parameter).
- Nosske et al., arXiv:2507.14030 (multi-surface aperture/emissivity
  formula, E37).
