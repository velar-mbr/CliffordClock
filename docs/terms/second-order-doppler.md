# Second-order Doppler shift (thermal ensemble)

E21's kinematic term in `docs/CONVENTIONS.md`. This is the time-dilation
shift a moving atom in a classical thermal ensemble carries, independent
of any field.

## What it is

A moving clock runs slow by the special-relativistic factor
`√(1-v²/c²)`, so an atom's own thermal motion through a lattice trap
shifts its clock frequency even with no external field present. For a
3D isotropic harmonic trap in thermal equilibrium, equipartition puts
`(1/2)k_BT` of kinetic energy on each velocity component, and the
ensemble-averaged shift works out to `-3k_BT/(2mc²)`. At Sr-87 and
`T = 5 μK`, that is about `-8.0e-21`, small next to the DC-Stark and BBR
terms but a real line item in every optical-lattice-clock uncertainty
budget.

## The formula

$$\left\langle\frac{\Delta\nu}{\nu_0}\right\rangle_{\text{kinematic}} = -\frac{\langle v^2\rangle}{2c^2} = -\frac{3k_BT}{2mc^2}$$

`v` is the atom's velocity, `c` the speed of light, `k_B` Boltzmann's
constant, `T` the ensemble temperature, and `m` the atomic mass. The
engine evaluates the exact kinematic factor `√(1-v²/c²)` at each atom's
actual velocity along its trajectory. The equipartition form above is the
closed-form limit a thermal ensemble converges to under that exact
evaluation.

## The code

```python
# src/cliffordclock/integrator/omega.py::scalar_rate_perturbation
v2 = jnp.sum(v * v, axis=-1)
x = v2 / SPEED_OF_LIGHT**2
gamma_inv = jnp.sqrt(1.0 - x)
kinematic = -x / (1.0 + gamma_inv)
return kinematic + p_minus_1 * gamma_inv
```

The kinematic term is computed as `-x/(1+γ⁻¹)`, algebraically identical
to `γ⁻¹-1` but stable down to `v/c` far below the realistic cold-atom
range (Sr at 1 μK has `v/c ≈ 3.3e-11`), where the literal subtraction
rounds to exactly zero in float64. `p_minus_1` is the field-based pivot
term (the DC-Stark shift and its companions), added to the kinematic
term at the end. The real implementation lives in
`src/cliffordclock/integrator/omega.py::scalar_rate_perturbation`.

## How it is checked

KA4 is a known-answer test, the repo's term for a check against a
published anchor value. It draws M = 5000 classical Sr-87 atoms at
`T = 5 μK` from the correct thermal position and velocity distributions
and integrates them through `integration.mode: secular`. The measured
mean shift, `-8.003764e-21 ± 6.33e-23` (SEM), agrees with the
equipartition prediction `-7.983437e-21` at `0.32σ`, well inside the
project's `5σ` statistical tolerance (`docs/validation.md`).

## Sources

- Equipartition theorem, standard statistical mechanics, applied to a 3D
  isotropic harmonic trap per `docs/validation.md`'s KA4 write-up.
