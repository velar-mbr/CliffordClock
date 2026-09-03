# Rydberg vapor-cell response

E43/E44 in `docs/CONVENTIONS.md` section 19. A field over a vapor-cell
atom region goes in; a per-atom Rydberg Stark shift and the resulting
EIT/Autler-Townes spectrum come out.

## What it is

Rydberg-atom RF electrometry reads an electric field off an atom's own
energy levels. A coupling laser and an RF source together split a
transparency window a probe laser sees in the vapor cell
(electromagnetically induced transparency, EIT). The splitting scales
linearly with the RF field's amplitude, an SI-traceable measurement
chain that needs no external calibration. A static or slowly varying
field acts on the same Rydberg state through the ordinary quadratic
Stark effect, shifting the atom's line.

This module puts both pieces together for Rb-85's 32D5/2 state, the
state Holloway et al. 2014's own calibration data (Fig. 15, 68.64 GHz)
anchors. A single atom's field gives a quadratic Stark shift. Many
atoms across an inhomogeneous field compose into one observed EIT/AT
line, shifted, broadened, and asymmetric depending on how uneven the
field is across the cell. The field can come from an imported CSV map
or a closed-form wall-patch model.

## The formula

The quadratic Stark shift of a single Rydberg state (Yerokhin et al.
2016 Eq. 5):

$$\Delta f = -\frac{1}{2}\frac{\alpha_0 E^2}{h}$$

The Autler-Townes splitting observed in the probe-transmission
spectrum, plotted against probe-laser detuning, carries a Doppler-
mismatch factor between the probe and coupling laser wavelengths
(CONVENTIONS.md section 19 derives this factor from the ladder's own
Doppler-detuning geometry and verifies it against Holloway et al.
2014's own Eq. 12):

$$\Delta f_{\text{AT}} = \frac{\lambda_c}{\lambda_p}\frac{\mu_{\text{RF}} E}{2\pi\hbar}$$

## The code

```python
# src/cliffordclock/integrator/rydberg_cell_response.py::rydberg_quadratic_stark_shift_hz
def rydberg_quadratic_stark_shift_hz(
    alpha0_au: float, field_v_per_m: float, n_star: float, *, margin: float = STARK_VALIDITY_MARGIN
) -> float:
    guard = margin * inglis_teller_field_v_per_m(n_star)
    if field_v_per_m > guard:
        raise RydbergStarkValidityError(...)
    alpha0_si = alpha0_au * ALPHA_AU_TO_SI
    shift_j = -0.5 * alpha0_si * field_v_per_m**2
    return shift_j / constants.PLANCK_H
```

The full ladder susceptibility, Doppler averaging, and Autler-Townes
splitting live in the same module's `ladder_susceptibility`,
`doppler_averaged_susceptibility`, and `autler_townes_splitting_hz`.
`compose_inhomogeneous_eit_spectrum` sums many atoms' own Stark-shifted
spectra, weighted by their local field, into one composed line.

## How it is checked

`benchmarks/run_rydberg_cell_response.py` runs four cases. C3 reproduces
Holloway et al. 2014 Fig. 15's three published (splitting, field) pairs
at 68.64 GHz to a worst relative error of 0.35% against a 1% tolerance,
`arithmetic_reproduction`. C4 checks the Rb-85 nD5/2 scalar
polarizability against two independent sources (O'Sullivan and
Stoicheff's measurement, Yerokhin et al.'s theory) at n = 30, 35, 50,
worst relative difference 2.88% against a 5% tolerance,
`arithmetic_reproduction`. C5 confirms zero field returns the
unperturbed line and a uniform field returns a pure shift with zero
added width, both checked at the byte level against a direct single-atom
evaluation, with a deliberate sign-flip and a deliberate doubled
coefficient both confirmed to break the match. C6 demonstrates the
qualitative phenomenology of Patrick et al. 2025's surface-charge
distortion problem (a wall-patch field's line shift and per-atom
Stark-shift spread both growing with patch charge and shrinking cell
radius), `computable_comparison`: no printed numeric target exists in
that paper to reproduce arithmetically. All four cases report
`kpi_verdict = "MET"`.

`tests/test_rydberg_cell_response.py` additionally cross-checks the RF
transition dipole moment `mu_RF` (derived from Rydberg-Ritz quantum
defects via direct Numerov integration, not looked up) against
Sedlacek et al. 2012's independently published value for the
kinematically identical 53D5/2 -> 54P3/2 transition, and against this
module's own Fig.-15-backed-out registry value for 32D5/2 -> 33P3/2,
both within a stated, disclosed factor-of-2 tolerance.

## Sources

- C. L. Holloway, J. A. Gordon, S. Jefferts, A. Schwarzkopf, D. A.
  Anderson, S. A. Miller, N. Thaicharoen, G. Raithel, "Broadband
  Rydberg Atom-Based Electric-Field Probe for SI-Traceable,
  Self-Calibrated Measurements," IEEE Trans. Antennas Propag. 62, 6169
  (2014), arXiv:1405.7066 (the anchor: Eqs. 1-4, 11, 12; Fig. 15's
  calibration data).
- M. Mack, F. Karlewski, H. Hattermann, S. Hoeckh, F. Jessen, D. Cano,
  J. Fortagh, "Measurement of absolute transition frequencies of 87Rb
  to nS and nD Rydberg states by means of electromagnetically induced
  transparency," Phys. Rev. A 83, 052515 (2011), arXiv:1103.6221 (the
  Rb-85 nD5/2 quantum defects).
- B. Sanguinetti, H. O. Majeed, M. L. Jones, B. T. H. Varcoe,
  "Precision measurements of quantum defects in the nP3/2 Rydberg
  states of 85Rb," J. Phys. B 42, 165004 (2009), arXiv:0905.0571 (the
  Rb-85 nP3/2 quantum defects).
- V. A. Yerokhin, S. Y. Buhmann, S. Fritzsche, A. Surzhykov,
  "Model-potential approach to the calculation of dipole
  polarizabilities of alkali-metal atoms," Phys. Rev. A 94, 032503
  (2016), arXiv:1608.04515 (the scalar-polarizability theory/experiment
  cross-tabulation).
- J. A. Sedlacek, A. Schwettmann, H. Kubler, R. Low, T. Pfau, J. P.
  Shaffer, "Microwave electrometry with Rydberg atoms in a vapor cell
  using bright atomic resonances," Nature Physics 8, 819 (2012),
  arXiv:1205.4461 (the mu_RF cross-check target; the rejected
  Doppler-mismatch-factor prose statement, resolved in
  CONVENTIONS.md section 19).
- A. K. Mohapatra, T. R. Jackson, C. S. Adams, "Coherent optical
  detection of highly excited Rydberg states using electromagnetically
  induced transparency," Phys. Rev. Lett. 98, 113003 (2007),
  arXiv:quant-ph/0612200 (Doppler-averaged susceptibility; the
  independent corroboration of the resolved Doppler-mismatch factor).
- M. Fleischhauer, A. Imamoglu, J. P. Marangos, "Electromagnetically
  induced transparency: Optics in coherent media," Rev. Mod. Phys. 77,
  633 (2005) (the 3-level susceptibility cross-check).
- L. Patrick, N. Schlossberger, D. F. Hammerland, N. Prajapati, T.
  McDonald, S. Berweger, R. Talashila, A. B. Artusio-Glimpse, C. L.
  Holloway, "Imaging of induced surface charge distribution effects in
  glass vapor cells used for Rydberg atom-based sensors," AVS Quantum
  Science 7, 024401 (2025), arXiv:2502.07018 (the surface-charge
  demonstrator's phenomenology target).
