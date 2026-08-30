# Ion electric-quadrupole shift

E34/E35 in `docs/CONVENTIONS.md`. This is the shift a trapped ion's
D/F clock states pick up from the local electric-field gradient, through
the state's own electric-quadrupole moment.

## What it is

A trapped ion's clock states with angular momentum `J ≥ 1` carry an
electric-quadrupole moment, so the local field gradient at the ion's
position shifts each `m_J` sublevel by an amount set by that moment and
the gradient's projection onto the ion's quantization axis. For the
validated Ca⁺:D5/2 case, that gradient sensitivity is large enough
(`a ≈ 2.97 Hz·mm²/V`) to matter at the fractional level a clock budget
tracks. Because the shift depends on `m_J` and the quantization-axis
orientation, averaging a state's population over three orthogonal
orientations cancels it exactly, a standard technique the engine models
directly.

## The formula

$$\Delta E_Q(J, m_J) = \frac{\Theta(J)}{2}\cdot\frac{J(J+1)-3m_J^2}{J(2J-1)}\cdot\left(\hat{n}^T G(r) \hat{n}\right)$$

$$(P-1)_Q = \frac{\Delta E_Q(J, m_J)}{h\nu_0}$$

`Θ(J)` is the state's electric-quadrupole moment, `J` and `m_J` the
angular-momentum quantum numbers, `n̂` the quantization-axis direction,
`G(r)` the traceless symmetric field-gradient tensor, and `ν₀` the clock
transition frequency.

## The code

```python
# src/cliffordclock/integrator/omega.py::quadrupole_shift_joules
n_hat = quantization_axis / jnp.linalg.norm(quantization_axis)
gradient_tensor = traceless_symmetric_gradient(grad_e_total)
contraction = jnp.einsum("i,...ij,j->...", n_hat, gradient_tensor, n_hat)
theta_si = theta_au * EA0_SQUARED_SI
mj_factor = quadrupole_mj_factor(j, m_j)
return 0.5 * theta_si * mj_factor * contraction
```

The real implementation lives in
`src/cliffordclock/integrator/omega.py::quadrupole_shift_joules`, and the
pivot-term conversion in `quadrupole_pivot_perturbation` in the same
file.

## How it is checked

`benchmarks/run_roos_quadrupole_slope.py` evaluates the engine's
quadrupole functions against the registry's Ca⁺:D5/2 entry and compares
against Roos et al.'s (quant-ph/0701215v1) measured two-ion Fig. 4a
slope, `a = 2.975(2) Hz·mm²/V`, in two labeled variants. The headline
cross-vintage comparison, against Itano's independent theory value
`Θ_theory = 1.917 ea₀²`, predicts `|a_pred| = 3.115229 Hz·mm²/V`,
residual `+4.71%`, `kpi_verdict = "NOT MET"`, an expected result that
recovers the literature's own known ~4.75% theory-versus-measurement
tension. The secondary arithmetic reproduction, against Roos's own
extracted `Θ = 1.83(1) ea₀²`, predicts `|a_pred| = 2.973849 Hz·mm²/V`,
residual `-0.04%`, `kpi_verdict = "MET"` (`docs/validation.md`,
`benchmarks/RESULTS.md`).

## Sources

- Roos et al., Nature 443, 316 (2006), quant-ph/0701215v1, Eq. 1 and
  Fig. 4a (the measured slope and the quadrupole-moment extraction).
- Itano, J. Res. NIST 105, 829 (2000), Eq. 46 (the canonical
  axially-symmetric quadrupole-shift form).
- Itano, Phys. Rev. A 73, 022510 (2006) (the independent theory value
  for Ca⁺:D5/2's quadrupole moment used in the cross-vintage comparison).
