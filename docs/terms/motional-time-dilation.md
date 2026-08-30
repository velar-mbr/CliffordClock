# Trapped-ion motional time dilation

E38 in `docs/CONVENTIONS.md`, with the WP31-WP35 trapped-ion extensions:
per-mode participation factors (WP31), two-ion radial-spectrum
reconstruction from a lab's own measured mode frequencies (WP32),
mode-specific intrinsic-micromotion enhancement (WP33), an exact Floquet
treatment of intrinsic micromotion (WP34), and a coupled two-ion Floquet
solve with a constrained fit (WP35).

## What it is

A trapped ion's motional state gives it a nonzero velocity even at zero
temperature, and that motion carries the same second-order-Doppler time
dilation a classical moving atom carries. The engine evaluates it as an
expectation value over the motional state, in place of the instantaneous
classical velocity a moving atom's trajectory supplies. For a single ion,
this needs only the trap's normal-mode frequencies and phonon numbers.
For a two-ion crystal, each mode's
contribution is weighted by the clock ion's own participation in that
mode, which the RF trapping field's intrinsic micromotion enhances
unevenly across modes. Reconstructed against a published Al⁺ clock
evaluation, the full treatment lands the total time-dilation shift at
`0.08σ` from the published value.

## The formula

$$\langle v^2\rangle = \sum_i \frac{\hbar\omega_i}{m}\cdot\text{participation}_i\cdot(\bar{n}_i + \tfrac{1}{2}) + v_{\text{rms,emm}}^2$$

$$(P-1)_{\text{motional}} = -\frac{\langle v^2\rangle}{2c^2}$$

`ω_i = 2π f_i` is each mode's angular frequency, `m` the species mass,
`participation_i` the clock ion's squared mass-weighted eigenvector
component in mode `i` (`1.0` for a single trapped ion), `n̄_i` the mode's
phonon number, and `v_rms,emm` an optional measured excess-micromotion
velocity. This is the same `-⟨v²⟩/2c²` form the classical second-order
Doppler shift uses, evaluated over a quantum motional-state expectation
value in place of an instantaneous velocity.

## The code

```python
# src/cliffordclock/integrator/omega.py::motional_pivot_perturbation
mean_v2 = motional_mean_squared_velocity_m2_s2(modes, species, v_rms_emm_m_s)
return -mean_v2 / (2.0 * SPEED_OF_LIGHT**2)
```

The real implementation lives in
`src/cliffordclock/integrator/omega.py::motional_pivot_perturbation`
and `motional_mean_squared_velocity_m2_s2`. The two-ion participation and
micromotion-enhancement machinery (`two_ion_participations`,
`two_ion_radial_participations`, `radial_micromotion_enhancement_exact`,
`coupled_two_ion_floquet_modes`, `constrained_two_ion_mathieu_fit`) lives
in the same file.

## How it is checked

`benchmarks/run_motional_al_ion.py` reconstructs Marshall et al.'s
published Al⁺ clock evaluation. The WP35 constrained two-ion Floquet fit
gives a total of `-1.1415e-17`, landing `0.08σ` from the published
`-114.6(3.8)e-19` with a three-component uncertainty budget (thermometry,
input rounding, model structure) combined in quadrature to `±4.087e-19`:
`kpi_verdict = "MET"`, classified arithmetic reproduction. The same
constrained-fit treatment applied to Brewer et al.'s independently
published trap gives partner deviations of `-0.18%`/`-0.14%`
(`docs/CONVENTIONS.md` section 16).

## Sources

- Marshall, Rodriguez Castillo, Arthur-Dworschack, Aeppli, Kim, Lee,
  Warfield, Hinrichs, Nardelli, Fortier, Ye, Leibrandt, Hume,
  "High-Stability Single-Ion Clock with 5.5x10⁻¹⁹ Systematic
  Uncertainty," arXiv:2504.13071v2 (2025), Supplemental Material
  Table S2, the published secular-motion mode frequencies and
  time-dilation total this benchmark reconstructs.
- Brewer, Chen, Hankin, Clements, Chou, Wineland, Hume, Leibrandt,
  "27Al+ Quantum-Logic Clock with a Systematic Uncertainty below
  10⁻¹⁸," Phys. Rev. Lett. 123, 033201 (2019), arXiv:1902.07694,
  Supplemental Material Table S2, the independently published trap
  dataset used as the second reconstruction target for the
  constrained-fit method.
- Berkeland, Miller, Bergquist, Itano, Wineland, J. Appl. Phys. 83, 5025
  (1998) (radial-secular Mathieu-parameter background
  `docs/CONVENTIONS.md` section 16 cites).
