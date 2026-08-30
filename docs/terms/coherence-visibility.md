# Ramsey fringe visibility and coherence

E39 in `docs/CONVENTIONS.md`. This is the loss of Ramsey fringe contrast
a spread of accumulated phase across an ensemble produces.

## What it is

Every atom in an ensemble accumulates its own perturbation phase over the
interrogation time, and when those phases spread across the population,
the Ramsey fringe the ensemble produces loses contrast even though each
atom's own phase evolution stays coherent. This module turns each atom's
accumulated phase into a unit phase factor, sums those factors across the
ensemble's population weights, and reads the fringe visibility and phase
off that sum. For a Gaussian-distributed spread of accumulated phases
(thermal, coherent, or squeezed motional states), the visibility reduces
to a closed form matching the same Gaussian characteristic-function
identity the project's `T₂*` dephasing-time formula uses.

## The formula

$$M = \sum_k p_k\, e^{i\Delta\Phi_k}, \qquad V = |M|, \qquad \phi = \arg(M)$$

$$V = e^{-\sigma_\Phi^2/2} \quad \text{(Gaussian closure, for } \Delta\Phi_k \sim \mathcal{N}(\mu, \sigma_\Phi^2)\text{)}$$

`ΔΦ_k` is worldline `k`'s accumulated perturbation phase (E22), `p_k` its
population weight, `V` the visibility (`0 ≤ V ≤ 1`), and `σ_Φ²` the
ensemble's phase variance. `V = 1` only when every worldline's
accumulated phase is identical, the no-spread limit with no decoherence.

## The code

```python
# src/cliffordclock/integrator/coherence.py::ramsey_visibility_and_phase
c = m[..., IDX_SCALAR]
s = m[..., IDX_E12]
visibility = jnp.sqrt(c * c + s * s)
phase = jnp.arctan2(s, c)
return visibility, phase
```

`m` is the population-weighted sum of the ensemble's phase factors,
built by `coherent_rotor_composition` and never renormalized: the
departure of that sum from unit magnitude is the decoherence signal this
module exists to report. The real implementation lives in
`src/cliffordclock/integrator/coherence.py::phase_to_rotor`,
`coherent_rotor_composition`, and `ramsey_visibility_and_phase`.

## How it is checked

`tests/test_coherent_visibility.py` includes two kill tests targeting
known failure modes. One confirms that renormalizing the composed sum
back to unit magnitude erases the decoherence signal. The Gaussian
closure identity `V = exp(-σ_Φ²/2)` is checked directly as a validation
identity against `dephasing_time_t2star`'s own `T₂*` formula, which
shares the same underlying characteristic-function argument
(`docs/CONVENTIONS.md` section 8).

## Sources

- `docs/CONVENTIONS.md` section 8 (E26-E28, E39), the project's own
  derivation of the coherence function, `T₂*`, and the Ramsey visibility
  identity from the ensemble's accumulated-phase statistics.
