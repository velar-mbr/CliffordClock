# WP36 lattice light shift benchmark cases (generated)

Generated: 2026-08-29T12:49:04.721426+00:00

This report validates both community lattice-light-shift models (the Katori-lineage harmonic/operational model, and the NIST Born-Oppenheimer+WKB model) against their defining papers, before either is wired into the pipeline (a later phase).

## Target 1: Ushijima et al. 2018 operational point

**Classification: arithmetic_reproduction**

| Quantity | Predicted | Published |
|---|---|---|
| u_op | 71.725 | 72.0(2) |
| delta_L_op (MHz) | 5.286 | 5.3(0.2) |
| u_op within published uncertainty | True | |
| detuning_op within published uncertainty | True | |

Source: Ushijima, Takamoto, Katori, PRL 121, 263202 (2018), main text and Eqs. 14-15

## Target 2: Aeppli et al. 2024 lattice-light-shift budget line

**Classification: arithmetic_reproduction**

| Quantity | Value |
|---|---|
| u0 | 15.06 |
| detuning (MHz) | 10.5 |
| Tr (nK) | 120 |
| Predicted shift (1e-19) | -0.056 |
| Predicted uncertainty (1e-19) | +/-2.218 |
| Published shift (1e-19) | -0.100 |
| Published uncertainty (1e-19) | +/-3.200 |
| Bands overlap | True |
| **kpi_verdict** | **MET** |

Source: Aeppli, Kim, Warfield, Safronova, Ye, PRL 133, 023401 (2024), Table I; coefficients and Tr from Kim, Aeppli, Bothwell, Ye, PRL 130, 113203 (2023)

## Target 3a: Bothwell et al. 2025 Table I, X/Y/Z reproduction

**Classification: arithmetic_reproduction**

| u0 (E_R) | Tr (nK) | X pred | X pub (BO+WKB) | Y pred | Y pub | Z pred | Z pub | max rel. err |
|---|---|---|---|---|---|---|---|---|
| 56.8 | 650 | 0.7855 | 0.7850 | 0.0608 | 0.0608 | 0.6455 | 0.6450 | 7.18e-04 |
| 66.4 | 550 | 0.8378 | 0.8380 | 0.0580 | 0.0580 | 0.7187 | 0.7190 | 4.59e-04 |
| 86.2 | 600 | 0.8643 | 0.8640 | 0.0515 | 0.0515 | 0.7588 | 0.7590 | 4.24e-04 |
| 112.2 | 720 | 0.8786 | 0.8790 | 0.0454 | 0.0454 | 0.7813 | 0.7810 | 8.84e-04 |

Worst relative error across all 4 rows: 8.84e-04 (tolerance 1%). **kpi_verdict: MET**

Source: Bothwell, Hunt, Siegel, Hassan, Grogan, Kobayashi, Gibble, Porsev, Safronova, Brown, Beloy, Ludlow, PRL 134, 033201 (2025), Appendix A Table I

## Target 3b: Bothwell et al. 2025 headline coefficient, computable comparison

**Classification: computable_comparison**

computable comparison, NOT an arithmetic reproduction of the paper's own fitted coefficient values: alpha~M1E2 in Bothwell et al. 2025's Table III is an output of their own nonlinear fit against raw (unpublished) scan data; reproducing it would require running that same fit against their raw data, which this module does not have. What is computed here is each model's own light-shift prediction at the paper's stated operating conditions, using that same paper's own published coefficient columns, and the resulting fractional-shift difference between the two models: the part of Target 3 that published inputs alone can settle.

| Quantity | Value |
|---|---|
| u0 | 100.0 |
| Tr (nK) | 600 |
| Harmonic-model shift (fractional) | +2.053e-17 |
| BO+WKB-model shift (fractional) | +2.269e-17 |
| Model difference (fractional) | +2.160e-18 |
| Published alpha~M1E2/h, harmonic (Hz) | -1.410e-18 |
| Published alpha~M1E2/h, BO+WKB (Hz) | -1.450e-18 |
| Published relative difference between the two models' alpha~M1E2 | +2.837% |

Source: Bothwell, Hunt, Siegel, Hassan, Grogan, Kobayashi, Gibble, Porsev, Safronova, Brown, Beloy, Ludlow, PRL 134, 033201 (2025), Table III and main text (u0 < 140 E_R, Tr ~= 600 nK)

## Density-of-states contrast

Species Yb171, u0=100.0, n_z=0. Cumulative number of radial states from the band bottom up to one radial-temperature thermal quantum above it, both models.

| Tr (nK) | Cumulative states (cos2/BO+WKB) | Cumulative states (harmonic) | Ratio |
|---|---|---|---|
| 50 | 2.9887e+01 | 2.8339e+01 | 1.0546 |
| 100 | 1.1977e+02 | 1.1335e+02 | 1.0566 |
| 200 | 4.8088e+02 | 4.5342e+02 | 1.0606 |
| 400 | 1.9382e+03 | 1.8137e+03 | 1.0686 |
| 800 | 7.8741e+03 | 7.2547e+03 | 1.0854 |
| 1600 | 3.2541e+04 | 2.9019e+04 | 1.1214 |

