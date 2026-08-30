# Lattice light shift

E40 (the Katori-lineage harmonic/operational model) and E41 (the NIST
Born-Oppenheimer+WKB model) in `docs/CONVENTIONS.md` section 17, plus a
differentiable JAX implementation of E41 (v1.13.0, WP37). Both community
models run side by side.

## What it is

The lattice light that traps atoms for interrogation also shifts the
clock transition, since the two clock states polarize differently in the
trapping light. Model A (E40) treats the trap as a harmonic well and
expands the shift in the reduced trap depth `u`. Model B (E41) solves the
true `cos²` site potential directly by finite-difference diagonalization
and a WKB radial density of states, a more expensive treatment that
stays more accurate as the radial temperature rises relative to the trap
depth. At Ushijima et al.'s operational magic point, `u_op ≈ 72 E_R`, the shift
crosses zero by design; at a real JILA operating point (`15.06 E_R`,
`Tr ≈ 120 nK`), the predicted budget line is `-0.056e-19`, against a
published `-0.1(3.2)e-19`.

## The formula

Model A (Ushijima et al. 2018 Eq. 1, `u = U/E_R` the reduced trap depth,
`δ_L` the laser detuning from the magic frequency, `n_z` the axial
vibrational quantum number):

$$h\nu_{LS} \approx \left[\frac{d\tilde\alpha_{E1}}{d\nu}\delta_L - \tilde\alpha_{qm}\right](n_z+\tfrac12)u^{1/2} - \left[\frac{d\tilde\alpha_{E1}}{d\nu}\delta_L + \tfrac32\tilde\beta(n_z^2+n_z+\tfrac12)\right]u + 2\tilde\beta(n_z+\tfrac12)u^{3/2} - \tilde\beta u^2$$

Model B (Bothwell et al. 2025 Eq. 6) uses the same three polarizability
coefficients, weighted by thermally-averaged factors `X`, `Y`, `Z`
(Beloy et al. 2020 Eq. 21) computed from the true site potential's
axial eigenstates and WKB radial turning points, in place of Model A's
`u`-power expansion:

$$\frac{\delta\nu_{LS}}{\nu_c} \approx -\left[\frac{d\tilde\alpha_{E1}}{d\nu}\delta_L X\, u_0 + \tilde\alpha_{M1E2} Y\, u_0 + \tilde\beta Z\, u_0^2\right]$$

## The code

```python
# src/cliffordclock/integrator/lattice_light_shift.py::harmonic_light_shift_hz
e1_slope = coeffs.e1_slope_per_hz
m1e2 = coeffs.m1e2_hz
beta = coeffs.hyperpolarizability_hz

term1 = (e1_slope * detuning_hz - m1e2) * (n_z + 0.5) * u_pow(0.5)
term2 = -(e1_slope * detuning_hz + 1.5 * beta * (n_z**2 + n_z + 0.5)) * u_pow(1.0)
term3 = 2.0 * beta * (n_z + 0.5) * u_pow(1.5)
term4 = -beta * u_pow(2.0)
return term1 + term2 + term3 + term4
```

The real implementation lives in
`src/cliffordclock/integrator/lattice_light_shift.py::harmonic_light_shift_hz`
(Model A) and `axial_energies_er`/`bo_wkb_density_of_states`/
`bo_wkb_fractional_light_shift` (Model B), with a differentiable JAX
version of Model B in `lattice_light_shift_jax.py`.

## How it is checked

Model A reproduces Ushijima et al.'s published operational point,
`u_op = 71.7` against `72(2)`, `δ_L,op = 5.29 MHz` against `5.3(2) MHz`,
both within the published uncertainty, and Aeppli et al.'s published
budget line as noted above; both `kpi_verdict = "MET"`. Model B
reproduces Bothwell et al.'s own published `X`/`Y`/`Z` table to better
than `0.1%` at all four checked `(u₀, Tr)` points, worst relative error
`8.8e-4` against a `1%` tolerance, `kpi_verdict = "MET"`
(`docs/CONVENTIONS.md` section 17).

## Sources

- Katori, Ovsiannikov, Marmo, Palchikov, "Strategies for reducing the
  light shift in atomic clocks," Phys. Rev. A 91, 052503 (2015).
- Ushijima, Takamoto, Katori, "Operational magic intensity for Sr optical
  lattice clocks," Phys. Rev. Lett. 121, 263202 (2018), arXiv:1812.11815.
- Kim, Aeppli, Bothwell, Ye, Phys. Rev. Lett. 130, 113203 (2023).
- Beloy, McGrew, Zhang, Nicolodi, Fasano, Hassan, Brown, Ludlow,
  "Modeling motional energy spectra and lattice light shifts in optical
  lattice clocks," Phys. Rev. A 101, 053416 (2020), arXiv:2004.06224.
- Bothwell, Hunt, Siegel, Hassan, Grogan, Kobayashi, Gibble, Porsev,
  Safronova, Brown, Beloy, Ludlow, "Lattice light shift evaluations in a
  dual-ensemble Yb optical lattice clock," Phys. Rev. Lett. 134, 033201
  (2025), arXiv:2409.10782.
- Aeppli, Kim, Warfield, Safronova, Ye, Phys. Rev. Lett. 133, 023401
  (2024), arXiv:2403.10664 (the published lattice-light-shift budget
  line reproduced above).
