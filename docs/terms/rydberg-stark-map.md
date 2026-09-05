# Full Rydberg Stark maps

E43-extension in `docs/CONVENTIONS.md` section 20. A registry Rydberg
state's Stark shift, as a function of field, computed by exact
diagonalization instead of a closed-form quadratic formula, so it stays
correct through and beyond the field where the quadratic (isolated-
state) treatment breaks down.

## What it is

The quadratic Stark shift (E43, `docs/terms/rydberg-cell-response.md`)
only holds while a Rydberg state stays isolated from its neighbors. Push
the field higher and the state starts mixing strongly with nearby
`(n, l, j)` states, an avoided crossing, where the shift stops being
a clean parabola. The textbook fix, attributed throughout this
literature to Zimmerman, Littman, Kash, Kleppner (1979): build the full
Hamiltonian in a truncated quantum-defect basis around the state of
interest, diagonalize it at each field value on a grid, and track each
eigenvalue's identity by continuity from one field step to the next. The
result is a smooth curve through the quadratic window and the crossing
alike, a Stark *map*, not a single coefficient.

This module builds that map for Rb-85's registry nD5/2 states (n = 30,
32, 35, 50) and replaces the earlier order-of-magnitude Inglis-Teller
validity guard with the map's own computed first-avoided-crossing field.

## The formula

Field-free energies from the Rydberg-Ritz quantum-defect formula, the
Hamiltonian, and the basis-restricting selection rule (ARC's own Eqs. 1,
18, and the sentence beneath Eq. 9):

$$E_{n,l,j} = -\frac{1}{2(n-\delta_{n,l,j})^2} \qquad H = H_0 + E\hat{z} \qquad \Delta l = \pm 1,\ \Delta m_j = 0$$

Off-diagonal dipole matrix elements combine a radial integral (computed
by Numerov integration of the radial Schrödinger equation, inward from
a large outer radius) with angular factors from Wigner 3-j/6-j symbols
(ARC's Eqs. 9-12).

## The code

```python
# src/cliffordclock/integrator/rydberg_stark_map.py::stark_map_registry_state
def stark_map_registry_state(
    n0: int, field_v_per_m: NDArray, *, l0: int = 2, j0: float = 2.5, mj: float = 0.5,
    delta_n: int = 5, l_max: int = 20,
) -> StarkMapResult:
    hamiltonian = stark_hamiltonian(n0, l0, j0, mj, delta_n=delta_n, l_max=l_max)
    return diagonalize_stark_map(hamiltonian, field_v_per_m)
```

`diagonalize_stark_map` tracks the target state at each field by maximum
overlap with the previous step's own tracked eigenvector, adiabatic
following, not a fixed eigenvalue index. `first_crossover_field_v_per_m`
walks that same overlap trace to find where it first drops sharply,
`stark_validity_field_v_per_m` uses that as the computed validity guard
(falling back to the Inglis-Teller estimate if the map itself fails).

## How it is checked

`benchmarks/run_rydberg_stark_map.py` runs four cases against all four
registry states. C3 compares the map's own low-field curvature
(mj-averaged over `mj = 1/2, 3/2, 5/2` to cancel the tensor
polarizability term) against the E43 registry's `alpha0`: worst relative
error 4.91% against a 15% tolerance, `arithmetic_reproduction`,
kill-tested against a sign-flipped and a doubled coefficient. C4
cross-validates the full field-swept eigenvalue curves against ARC
(Alkali Rydberg Calculator, v.3.10.2, an independent open-source
implementation), this module's own tracking code applied to both
Hamiltonians so the comparison isolates the Hamiltonian construction
itself: worst low-field relative error 2.05% against a 5% tolerance,
`independent_implementation_reproduction`. C5 checks three published
anchors (Holloway et al. 2014's low-field calibration, O'Sullivan and
Stoicheff's printed nS crossing-field fit as a same-family method check,
and Grimmel et al. 2015's supplementary data, which turned out not to be
machine-readable this session). C6 sweeps the basis size
(`delta_n, l_max`) for 50D5/2 (the registry state closest to its own
crossing, an order of magnitude below 30D5/2's), and confirms both the
low-field shift and the first-crossover field itself are stable well
inside their own stated tolerances before the production basis size.

Two Numerov bugs surfaced while building this module, both found by
comparing against ARC rather than trusting internal agreement alone: a
sign error in an existing, separate integrator (`rydberg_cell_response`)
and this module's own initial choice to integrate the radial equation
outward instead of inward, the wrong direction for a multi-state
Hamiltonian's relative phases. Both are described in full, with the
exact-solution and pair-by-pair verifications that found them, in
`cliffordclock.integrator.rydberg_stark_map`'s own module docstring.

## Sources

- N. Sibalic, J. D. Pritchard, C. S. Adams, K. J. Weatherill, "ARC: An
  open-source library for calculating properties of alkali Rydberg
  atoms," Comp. Phys. Comm. 220, 319 (2017), arXiv:1612.05529 (the
  Hamiltonian, matrix-element, and Numerov-substitution equations this
  module implements, and the independent-implementation cross-check).
- M. L. Zimmerman, M. G. Littman, M. M. Kash, D. Kleppner, "Stark
  structure of the Rydberg states of alkali-metal atoms," Phys. Rev. A
  20, 2251 (1979) (the method's historical origin; not read directly,
  no equation number attributed to it anywhere in this module).
- J. Grimmel, M. Mack, F. Karlewski, F. Jessen, M. Reinschmidt, N.
  Sandor, J. Fortagh, "Measurement and numerical calculation of Rubidium
  Rydberg Stark spectra," New J. Phys. 17, 053005 (2015),
  arXiv:1503.08953 (an independent from-scratch implementation
  corroborating ARC's own restatement of the method).
- M. Marinescu, H. R. Sadeghpour, A. Dalgarno, "Dispersion Coefficients
  for Alkali-Metal Dimers," Phys. Rev. A 49, 982 (1994) (the one-electron
  model potential shaping this module's radial wavefunctions).
- W. Li, I. Mourachko, M. W. Noel, T. F. Gallagher, Phys. Rev. A 67,
  052502 (2003); J. Han, Y. Jamil, D. V. L. Norum, P. J. Tanner, T. F.
  Gallagher, Phys. Rev. A 74, 054502 (2006); K. Moore, A. Duspayev, R.
  Cardman, G. Raithel, "Measurement of the Rb g-series quantum defect
  using two-photon microwave spectroscopy," Phys. Rev. A 102, 062817
  (2020) (the S/P/D, F, and G quantum defects extending this module's
  basis beyond Phase A's own registry).
- O'Sullivan, M. S. and Stoicheff, B. P., "Scalar polarizabilities and
  avoided crossings of high Rydberg states in Rb," Phys. Rev. A 31, 2718
  (1985) (the printed nS crossing-field fit used as a same-family method
  check).
