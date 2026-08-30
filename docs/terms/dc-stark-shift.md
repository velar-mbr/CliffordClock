# DC-Stark shift

E14a/E14b in `docs/CONVENTIONS.md`. This is the shift a stray DC electric
field puts on an optical clock transition.

## What it is

A clock state carries no permanent electric dipole, so a static field
shifts its energy through the second-order (quadratic) Stark effect: the
field polarizes the atom, and the induced dipole's energy scales as the
field squared. The two clock states polarize by different amounts, so
the transition frequency shifts by an amount set by their differential
polarizability, `Δα`. At a realistic chamber stray field of 100 V/m, this
shift lands around `-7e-17` for Sr-87 and `-3.5e-17` for Yb-171, both at
the fractional level a lattice-clock budget tracks.

## The formula

$$\frac{\Delta\nu(r)}{\nu_0} = -\frac{\Delta\alpha}{2}\frac{|E(r)|^2}{h\nu_0}$$

`Δα` is the transition's differential static scalar polarizability
(C²m²J⁻¹), `E(r)` the local field (V/m), `ν₀` the clock transition
frequency, and `h` Planck's constant. An earlier validation-only form,
`P(r) = 1 + δE(r)·μ/(m_e c²)` with a user-supplied dipole `μ` (E14a), is
the linearization of this formula about a bias field once the
denominator `m_e c²` is replaced by `hν₀` and `μ = −Δα·E₀`, and it
remains in the codebase in that substituted form as the closed-form case
the integrator validates against.

## The code

```python
# src/cliffordclock/integrator/omega.py::stark_pivot_terms
prefactor = k_s / nu_0  # (V/m)^-2, dimensionless once multiplied by |E|^2 (E14b)

e0_sq = jnp.sum(e0 * e0, axis=-1)
cross_dot = jnp.sum(e0 * delta_e, axis=-1)
delta_e_sq = jnp.sum(delta_e * delta_e, axis=-1)

baseline = prefactor * e0_sq
cross = prefactor * (2.0 * cross_dot)
quadratic = prefactor * delta_e_sq
```

The three terms of `|E₀+δE|²` are scaled by the Stark prefactor and
summed separately, so a small gradient field `δE` next to a much larger
bias field `E₀` never gets lost inside a squared vector sum. The real
implementation lives in
`src/cliffordclock/integrator/omega.py::pivot_perturbation_stark`, which
sums these terms and composes them with the BBR, quadrupole,
gravitational, and motional pivot terms.

## How it is checked

KA1 (a known-answer test, the repo's term for a check against a
published anchor value) reproduces the textbook uniform-field formula to
float64 precision for Sr-87, and KA2 does the same for Yb-171, both
against literature `Δα` values. KA3 checks the mean shift and its
variance for a linear-gradient field against an independent
Gaussian-moment reference (`docs/validation.md`). The engine
also reconstructs NPL's own published stray-field DC-Stark measurement
(arXiv:1706.01944) from their independently measured field: predicted
band `[-3.290, -1.208]e-20`, published band `[-3.2, -1.2]e-20`, the bands
overlap and `kpi_verdict = "MET"` (`benchmarks/RESULTS.md`).

## Sources

- T. Middelmann, S. Falke, C. Lisdat, U. Sterr, "High Accuracy Correction
  of Blackbody Radiation Shift in an Optical Lattice Clock," Phys. Rev.
  Lett. 109, 263004 (2012), arXiv:1208.2848 (Sr-87 `Δα`).
- J.A. Sherman, N.D. Lemke, N. Hinkley, M. Pizzocaro, R.W. Fox, A.D.
  Ludlow, C.W. Oates, "High-Accuracy Measurement of Atomic Polarizability
  in an Optical Lattice Clock," Phys. Rev. Lett. 108, 153002 (2012),
  arXiv:1112.2766 (Yb-171 `Δα`).
