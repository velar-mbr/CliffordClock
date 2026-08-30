# WP38 Deliverable 2: sideband-spectrum cross-validation against large-lattice-model

Generated: 2026-08-30T04:14:32.243986+00:00

Independent oracle: large-lattice-model (github.com/inrim/large-lattice-model), MIT license, INRIM, commit `f569907cdf2f08a9081386139211a205b3c42624`. No code from that repository enters CliffordClock; only its numeric output (a fixture JSON, generated in a separate environment) is compared here.

## Tier 1: axial band-bottom eigenvalue reproduction

**Classification: independent_implementation_reproduction**

Worst relative error: 1.06e-07 (tolerance 0%). **kpi_verdict: PASS**

| D (E_R) | n_z | CliffordClock (E_R) | INRIM Mathieu (E_R) | rel. err |
|---|---|---|---|---|
| 56.8 | 0 | -49.522666 | -49.522666 | 1.42e-09 |
| 56.8 | 1 | -35.530212 | -35.530212 | 9.14e-09 |
| 56.8 | 2 | -22.738976 | -22.738976 | 3.19e-08 |
| 56.8 | 3 | -11.222857 | -11.222856 | 1.06e-07 |
| 80.0 | 0 | -71.313386 | -71.313386 | 1.51e-09 |
| 80.0 | 1 | -54.491064 | -54.491063 | 8.63e-09 |
| 80.0 | 2 | -38.839295 | -38.839294 | 2.82e-08 |
| 80.0 | 3 | -24.506024 | -24.506022 | 7.37e-08 |
| 100.0 | 0 | -90.256779 | -90.256779 | 1.49e-09 |
| 100.0 | 1 | -71.314861 | -71.314861 | 8.45e-09 |
| 100.0 | 2 | -53.520943 | -53.520942 | 2.65e-08 |
| 100.0 | 3 | -37.013512 | -37.013510 | 6.55e-08 |
| 150.0 | 0 | -138.007999 | -138.007999 | 4.42e-10 |
| 150.0 | 1 | -114.559094 | -114.559093 | 2.30e-09 |
| 150.0 | 2 | -92.224232 | -92.224231 | 6.88e-09 |
| 150.0 | 3 | -71.105426 | -71.105425 | 1.39e-08 |

## Tier 2: Franck-Condon detuning reproduction

**Classification: independent_implementation_reproduction**

Worst relative error, excluding points within 5 E_R of the band top (see this case's own docstring for why): 4.73e-03 (tolerance 2%). **kpi_verdict: PASS**. 4 of 36 points excluded as near-band-top.

## Tier 3: full sideband-shape comparison (convention-bridged)

**Classification: computable_comparison**

Convention bridges:

- **lorentzian_peak_height**: large-lattice-model's own lorentzian(x,x0,w) (latticemodel.py) peaks at 0.5 (den=1+(x-x0)^2/w^2, returns 0.5/den); this project's harmonic_sideband_shape/bowkb_sideband_shape (Blatt et al. 2009 App. A1's own Lorentzian form) peak at 1. Bridged by comparing peak-normalized SHAPE, canceling the height difference before any comparison runs.
- **linewidth_convention**: large-lattice-model scales each (nz, rc) state's own Lorentzian width by its normalized Rabi frequency (rabi_ho, latticemodel.py, Wineland and Itano 1979 Eq. 31), so `wc` is a per-state-scaled half-width; this project's linewidth_hz is a single FIXED half-width shared by every (n_z, E) term (module docstring's own stated Blatt et al. 2009 approximation, 'given by the carrier Rabi frequency'). Bridged by using large-lattice-model's own wc=2000 Hz directly as this project's linewidth_hz, an approximate match between two different conventions.
- **integration_domain_e_max**: large-lattice-model's own sidebands() fixes E_max=0.0 for both blue and red (sidebands.py); this project's bowkb_sideband_shape masks by the target-band-boundedness condition directly (module docstring's 'Integration domain' section). Both converge to the same physical boundary for a deep trap; bridged by comparing shapes at moderate depth (D=80-100 E_R), where the two conditions land close together.
- **depth_definition**: Both sides define D/u0 identically: peak trap depth in units of the recoil energy E_R = h^2/(2*m*lambda^2) (large-lattice-model's settings.py; this project's recoil_energy_j_jax). No bridging needed.
- **temperature_convention**: Both sides take Tz/Tr as ordinary kelvin, two independent temperatures for the longitudinal and radial degrees of freedom (Beloy et al. 2020's own two-temperature ansatz, Goti et al. 2025 Eq. 9). No bridging needed.

| D (E_R) | Tz (uK) | Tr (uK) | sideband | INRIM peak (Hz) | predicted peak (Hz) | diff (Hz) | shape corr. |
|---|---|---|---|---|---|---|---|
| 80.0 | 0.30 | 0.30 | blue | 33000 | 32500 | 500 | 0.9339 |
| 80.0 | 0.30 | 0.30 | red | -33000 | -32500 | 500 | 0.9416 |
| 100.0 | 1.00 | 1.00 | blue | 35000 | 34500 | 500 | 0.9852 |
| 100.0 | 1.00 | 1.00 | red | -34500 | -34000 | 500 | 0.9864 |

Worst peak-position difference: 500 Hz. Minimum shape correlation: 0.9339.

