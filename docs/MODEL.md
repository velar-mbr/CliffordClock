# How it works

CliffordClock's physical model reduces to one idea: each atom in the clock
carries a single number, its clock rate, and every systematic effect is a
multiplicative correction to that number. Blackbody radiation, the Stark
shift from stray fields, the second-order Doppler shift from motion, the
AC Stark shift from a lattice trap, the electric quadrupole shift, and
gravitational redshift each contribute a small fractional correction `p`.
Every correction is evaluated at the atom's own position and velocity at
that instant. The product of `(1 + p)` over every active term is what the
rest of this codebase calls the pivot: the object the relativity
literature calls the lapse function or the redshift factor, the ratio
between an atom's local clock rate and a freely running reference rate.
Each small `p` is a pivot perturbation, the fractional piece one
systematic effect contributes to that ratio. The clock rate at time `t`
is the bare transition frequency times the pivot, and an atom's
accumulated phase is that rate integrated along its own path through the
interrogation.

Two more steps turn one atom's phase into the numbers a lab reports. An
ensemble of atoms samples the trap, each atom with its own trajectory and
so its own accumulated phase; the mean of those phases, divided by the
bare oscillation over the same time, gives the ensemble's fractional
frequency shift. The same set of phases, treated as unit vectors on the
complex plane and averaged, gives a vector, and its length is the Ramsey
fringe visibility. Phases that stay aligned average to length one; phases
spread out by the ensemble's spread in position and velocity average to
something shorter. The shift is the mean of the phase distribution
across atoms; the visibility is the modulus of the mean phasor, the
distribution's coherence. For small spreads, that modulus is set by the
phase variance.

This is the whole model:

```python
def clock_phase(worldline, terms, nu_0):
    rate = lambda t: nu_0 * prod(1.0 + p(worldline.state(t)) for p in terms)
    return 2.0 * pi * integrate(rate, worldline.proper_time)

phases = [clock_phase(w, terms, NU_0) for w in ensemble]
shift = mean(phases) / (2.0 * pi * NU_0 * T) - 1.0
visibility = abs(mean(exp(1j * array(phases))))
```

The additive budget every optical-clock systematics table publishes is
the linear term of this same product. Expanding it gives
`prod(1+p_k) = 1 + sum(p_k) + O(p^2)`; today's per-effect terms sit near
`p_k ~ 1e-15`, so the quadratic remainder lands near `1e-30`, far below
any clock's measurement floor. That smallness is structural, a property
of how small each `p_k` already is, and CliffordClock evaluates the full
product per atom regardless of size, before any ensemble averaging runs.
The additive table a systematics budget prints is what falls out of that
per-atom product at leading order.

Each `p` in that product is a published model. `src/cliffordclock/`
implements the DC-Stark shift, blackbody radiation, the electric
quadrupole shift, motional time dilation, the lattice light shift, and
gravitational redshift, each cited to its source paper in
`docs/CONVENTIONS.md`. The tests, benchmarks, and notebooks in this
repository are the evidence that each term is evaluated correctly:
known-answer checks against textbook formulas, reproductions of published
experimental measurements, and cross-checks between independent
implementations of the same physics.

Five places take this further:

- [`paper/composition/main.pdf`](../paper/composition/main.pdf): a
  six-page companion paper comparing this composition law with the
  field's additive budget tables, with case studies where the
  difference in structure paid.
- [`examples/ten_line_clock.py`](../examples/ten_line_clock.py): a
  runnable, self-contained version of the code above, in plain NumPy,
  reproducing a published gravitational-redshift measurement.
- [`docs/terms/`](terms/): a one-page reference per systematic (BBR,
  Stark, quadrupole, motional time dilation, lattice light shift,
  gravitational redshift), each stating its formula and its source.
- [`notebooks/`](../notebooks/): the full pipeline walked by hand, field
  import through the spectral line profile, one notebook per case.
- [`docs/validation.md`](validation.md): the ledger of every check this
  engine has passed, case by case, with formula, source, and tolerance.
