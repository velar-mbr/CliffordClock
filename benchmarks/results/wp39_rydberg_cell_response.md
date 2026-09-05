# WP39 Rydberg vapor-cell response benchmark cases (generated)

Generated: 2026-09-03T06:38:25.685866+00:00

This report validates the Rydberg vapor-cell response module (CONVENTIONS.md section 19, E43/E44) against its two anchor papers' own published numbers, its own structural limits, and a qualitative reproduction of the surface-charge distortion problem.

## C3: Holloway et al. 2014 Fig. 15 calibration, all three pairs

**Classification: arithmetic_reproduction**

| Published Delta_f (MHz) | Published |E| (V/m) | Predicted Delta_f (MHz) | Relative error |
|---|---|---|---|
| 4.35 | 0.89 | 4.3652 | 0.349% |
| 20.09 | 4.09 | 20.0602 | 0.148% |
| 48.31 | 9.83 | 48.2132 | 0.200% |

Tolerance: 1%. Worst relative error: 0.349%. **kpi_verdict: MET**

Source: Holloway, Gordon, Jefferts, Schwarzkopf, Anderson, Miller, Thaicharoen, Raithel, IEEE Trans. Antennas Propag. 62, 6169 (2014) [arXiv:1405.7066], Fig. 15

## C4: Rb-85 nD5/2 scalar polarizability, two independent sources

**Classification: arithmetic_reproduction**

| n | n_star | alpha0 theory (a0^3) | alpha0 experiment (a0^3) | Relative difference |
|---|---|---|---|---|
| 30 | 28.654 | 9.090e+09 | 9.360e+09 | 2.88% |
| 35 | 33.654 | 2.580e+10 | 2.530e+10 | 1.98% |
| 50 | 48.654 | 2.880e+11 | 2.890e+11 | 0.35% |

Tolerance: 5%. Worst relative difference: 2.88%. **kpi_verdict: MET**

Derived alpha0(32D5/2) = 1.4146e+10 a0^3. Power-law fit through the three tabulated rows above (n_star^p scaling), averaged across the theory and experiment fits; not a value printed in either source (see rcr.derive_rb85_32d52_alpha0_au docstring).

Sources: Yerokhin, Buhmann, Fritzsche, Surzhykov, PRA 94, 032503 (2016) [arXiv:1608.04515], Table IV, DFCP; O'Sullivan & Stoicheff, PRA 31, 2718 (1985) / PRA 33, 1640 (1986), as tabulated in Yerokhin et al. 2016 Table IV

## C5: limit kill-tests

**Classification: internal_structural_check**

| Check | Result |
|---|---|
| Zero field byte-identical to the unperturbed line | True |
| Uniform field byte-identical to a pure shift | True |
| Sign-flip kill-test armed (deliberately broken case differs) | True |
| Doubled-coefficient kill-test armed (deliberately broken case differs) | True |

**kpi_verdict: MET**

## C6: surface-charge demonstrator

**Classification: demonstrator. Evidentiary class: computable_comparison.**

No 2025-2026 follow-up literature claims this problem solved or a field-wide standardized mitigation adopted (dossier currency check, September 2026). Partial, geometry-specific workarounds exist (all-dielectric cells, three-photon near-IR excitation) and are not claimed here as a general fix.

| Condition | Cell radius (mm) | Patch charge (fC) | Line shift (MHz) | Per-atom shift spread (MHz) | Full-line width (MHz) | Asymmetry |
|---|---|---|---|---|---|---|
| no charge | 12.5 | 0.0 | +0.0000 | 0.0000 | 10.2195 | +0.000 |
| weak patch | 12.5 | 1000.0 | -0.0006 | 2.8270 | 10.2191 | +0.003 |
| strong patch | 12.5 | 4000.0 | -0.0049 | 45.2322 | 10.2097 | +0.018 |
| strong patch, small cell | 6.0 | 4000.0 | -0.0261 | 118.1456 | 10.1381 | +0.018 |

Line shift grows with patch charge: True. Per-atom shift spread grows with patch charge: True. Per-atom shift spread grows as the cell shrinks: True. **kpi_verdict: MET**

Source: Patrick, Schlossberger, Hammerland, Prajapati, McDonald, Berweger, Talashila, Artusio-Glimpse, Holloway, AVS Quantum Science 7, 024401 (2025) [arXiv:2502.07018]

