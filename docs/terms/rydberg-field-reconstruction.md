# Differentiable Rydberg field-to-spectrum chain and field reconstruction

E45 in `docs/CONVENTIONS.md` section 21. A `jax.numpy` port of the
quadratic-Stark/EIT chain, differentiable end to end, plus a
gradient-based demonstrator that recovers a vapor cell's own field
distribution from a synthetic measured spectrum.

## What it is

A Rydberg-atom RF/DC sensor's calibration question runs backward from
the forward chain `docs/terms/rydberg-cell-response.md` describes.
Given a measured EIT/Autler-Townes spectrum, what field distribution
across the vapor cell produced it? Answering that by gradient descent
needs `jax.grad` through the same physics. This module ports the
quadratic-Stark path to `jax.numpy`. The full Stark-map eigensolve
carries its own documented gradient-conditioning risk near an avoided
crossing, a separate, harder problem left for future work. The ported
chain's gradients are checked against finite differences of the
original implementation. A three-parameter field
model sits on top of it: a uniform background, a linear gradient along
the cell's long axis, and one localized wall-patch bump, reusing the
shape of the existing surface-charge demonstrator's patch field. A
synthetic round-trip then fits that field model back from a noisy
composed spectrum, with Laplace uncertainties on the recovered field.

## The formula

The ported quadratic Stark shift (Yerokhin et al. 2016 Eq. 5, unchanged
from the reference module):

$$\Delta f = -\frac{1}{2}\frac{\alpha_0 E^2}{h}$$

The new field model, a softened point-source bump added to a linear
background:

$$E(\mathbf{r}) = E_0 + g \cdot z + A \frac{s^2}{|\mathbf{r}-\mathbf{r}_{\text{patch}}|^2 + s^2}$$

with `E_0` the uniform background, `g` the axial gradient, `A` the
patch amplitude, and `s` a fixed softening length that keeps the bump's
gradient finite everywhere, including at the patch itself.

## The code

```python
# src/cliffordclock/integrator/rydberg_cell_response_jax.py::cell_field_magnitude_v_per_m_jax
def cell_field_magnitude_v_per_m_jax(
    positions_m, e_uniform_v_per_m, gradient_v_per_m_per_m, patch_amplitude_v_per_m,
    patch_position_m, patch_softening_m,
):
    z = positions_m[:, 2]
    background = e_uniform_v_per_m + gradient_v_per_m_per_m * z
    r_sq = jnp.sum((positions_m - patch_position_m[None, :]) ** 2, axis=-1)
    patch_term = patch_amplitude_v_per_m * patch_softening_m**2 / (r_sq + patch_softening_m**2)
    return background + patch_term
```

The full chain lives in
`src/cliffordclock/integrator/rydberg_cell_response_jax.py`:
`ladder_susceptibility_jax` and `doppler_averaged_susceptibility_jax`
port the reference module's own four-level ladder and Doppler average;
`compose_inhomogeneous_eit_spectrum_jax` sums many atoms' own
Stark-shifted spectra; `rb85_field_reconstruction_forward_model_jax`
combines the field model with that composition into one scalar
observable, `Im(chi)`, the probe-absorption spectrum a fit optimizes
against.

## How it is checked

`tests/test_rydberg_cell_response_jax.py` checks agreement against the
reference module (machine precision, no eigensolve on either side),
`jax.grad` against central finite differences of the reference
(`1.7e-6` relative, worst case), a `NaN` sweep across extreme field,
temperature, and Rabi-frequency inputs, jit determinism across fresh
processes, and a measured memory bound.

`benchmarks/run_rydberg_field_reconstruction.py` runs the field-
reconstruction demonstrator: `scipy.optimize.minimize` with exact
`jax`-supplied gradients recovers a planted three-parameter field
distribution from a synthetic composed spectrum, across a grid of four
truth values and two noise seeds. 8/8 fits converge and report a
positive-definite Hessian; 6/8 recover all three parameters within
their own reported 2-sigma Laplace uncertainty. Both exceptions trace
to a stated cause (the wall-patch amplitude's own weaker identifiability
in this geometry, and one optimizer-bound-proximity case at the grid's
largest truth values), the same `hessian_positive_definite` discipline
`docs/terms/sideband-spectrum.md`'s own fit demonstrator uses.

## Sources

- Yerokhin, Buhmann, Fritzsche, Surzhykov, "Electric dipole
  polarizabilities of Rydberg states of alkali-metal atoms," Phys. Rev.
  A 94, 032503 (2016), arXiv:1608.04515 (the quadratic Stark shift, same
  source `docs/terms/rydberg-cell-response.md` already cites).
- Holloway, Gordon, Jefferts, Schwarzkopf, Anderson, Miller, Thaicharoen,
  Raithel, "Broadband Rydberg Atom-Based Electric-Field Probe for
  SI-Traceable, Self-Calibrated Measurements," IEEE Trans. Antennas
  Propag. 62, 6169 (2014), arXiv:1405.7066 (the ladder susceptibility
  this module ports).
- L. Patrick, N. Schlossberger, D. F. Hammerland, N. Prajapati, T.
  McDonald, S. Berweger, R. Talashila, A. B. Artusio-Glimpse, C. L.
  Holloway, "Imaging of induced surface charge distribution effects in
  glass vapor cells used for Rydberg atom-based sensors," AVS Quantum
  Science 7, 024401 (2025), arXiv:2502.07018 (the wall-patch
  phenomenology the field model's SHAPE reuses, same source
  `docs/terms/rydberg-cell-response.md` already cites for its own
  surface-charge demonstrator).
