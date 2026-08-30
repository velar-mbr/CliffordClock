# Sideband-spectrum forward model and fitting

E42 in `docs/CONVENTIONS.md` section 18. Two forward-model paths (Path A,
harmonic; Path B, BO+WKB) generate a carrier-plus-sidebands spectrum, and
both are differentiable so a lab's own sideband scan can be fit directly.

## What it is

Sideband spectroscopy probes an atom's motional state by driving the
carrier and the red/blue motional sidebands and reading off the
population that ends up excited. Path A builds the spectrum from the
harmonic-trap Rabi-flopping probability and a thermally-weighted sum of
power-broadened Lorentzians, the same harmonic energy spectrum Model A's
lattice light shift uses. Path B builds the blue-sideband shape from the
true site potential's WKB density of states and Franck-Condon detuning,
the same machinery Model B's lattice light shift uses, so a lab's own
thermometry and its own light-shift budget share one underlying model.
Both paths are implemented with exact JAX gradients, so trap depth and
radial temperature can be recovered from a scan by gradient descent.

## The formula

The shared harmonic-oscillator motional spectrum (Blatt et al. 2009
Eq. 3; Goti et al. 2025 Eq. 1):

$$\frac{E_{n_x,n_y,n_z}}{h} \approx \nu_z(n_z+\tfrac12) + \nu_r(n_x+n_y+1) - \frac{\nu_{\text{rec}}}{2}(n_z^2+n_z+\tfrac12) - \nu_{\text{rec}}\frac{\nu_r}{\nu_z}(n_x+n_y+1)(n_z+\tfrac12)$$

The blue-sideband detuning (Blatt Eq. 8; Goti Eq. 2), the longitudinal
energy gap for a `n_z → n_z+1` transition:

$$\gamma(n_z) = \nu_z - \nu_{\text{rec}}(n_z+1) - \nu_{\text{rec}}\frac{\nu_r}{\nu_z}(n_r+1)$$

`ν_z`/`ν_r` are the axial/radial trap frequencies, `ν_rec` the recoil
frequency, and `n_r = n_x + n_y` the combined radial quantum number.

## The code

```python
# src/cliffordclock/integrator/sideband_spectrum_jax.py::blue_sideband_detuning_hz
def blue_sideband_detuning_hz(
    n_z: jnp.ndarray, n_r: jnp.ndarray, nu_z: jnp.ndarray, nu_r: jnp.ndarray, nu_rec: jnp.ndarray
) -> jnp.ndarray:
    return nu_z - nu_rec * (n_z + 1.0) - nu_rec * (nu_r / nu_z) * (n_r + 1.0)
```

The real implementation lives in
`src/cliffordclock/integrator/sideband_spectrum_jax.py`, in
`blatt_trap_frequencies_hz`, `harmonic_full_spectrum` (Path A), and
`bowkb_full_spectrum` (Path B).

## How it is checked

`benchmarks/run_sideband_spectrum.py` cross-validates against
`large-lattice-model` (INRIM), an independent public implementation of
the Beloy et al. 2020 model. Band-bottom eigenvalues agree to a worst
relative error of `1.06e-7`. Franck-Condon detunings agree to `4.7e-3`.
Both land `kpi_verdict = "PASS"`.

`benchmarks/run_sideband_fit.py` runs a synthetic generate-and-refit
round trip with exact gradients and Laplace uncertainties. 12/12 fits
converge, and recovered parameters land within their own reported
2-sigma uncertainty in 11/12 cases. The one exception is flagged
directly by its own Hessian positive-definiteness check
(`docs/CONVENTIONS.md` section 18).

## Sources

- Blatt, Thomsen, Campbell, Ludlow, Swallows, Martin, Boyd, Ye, "Rabi
  spectroscopy and excitation inhomogeneity in a one-dimensional optical
  lattice clock," Phys. Rev. A 80, 052703 (2009), arXiv:0906.1419.
- Goti, Petrucciani, Condio, Levi, Calonico, Pizzocaro, "Atomic
  thermometry in optical lattice clocks," arXiv:2508.08164 (v2, 2 Sept
  2025).
- `large-lattice-model` (github.com/inrim/large-lattice-model, MIT
  license, (c) 2021-2024 Marco Pizzocaro, INRIM), the independent
  implementation used for cross-validation.
