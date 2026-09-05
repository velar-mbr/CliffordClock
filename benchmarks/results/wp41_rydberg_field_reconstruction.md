# WP41: differentiable Rydberg field-to-spectrum chain and field reconstruction

Generated: 2026-09-03T16:08:40.833614+00:00

Synthetic round-trip: generator == fitter. Gradient-based optimization through this project's own differentiable quadratic-Stark/EIT chain recovers a known three-parameter cell field distribution (uniform background, linear axial gradient, one wall-patch amplitude) from a synthetic composed EIT spectrum. 1/8 cases recover all three parameters within their own reported 1-sigma Laplace uncertainty; 6/8 within 2-sigma, the observed coverage on this eight-case truth/seed grid. No real Rydberg-sensor scan is fit here. Every truth value and every optimizer-bound corner is checked to stay inside the guarded quadratic-Stark validity window before any fit runs (2110.9 V/m for the Rb-85 32D5/2 registry state). Every case's Hessian is checked for positive definiteness (hessian_positive_definite) before its inverse is trusted as a covariance; at a saddle point the reported uncertainty is nan.

## C1: agreement, JAX vs the numpy reference

Single-atom worst-case relative error: `3.183e-15` at {'temperature_k': 500.0, 'e_coupling_v_per_m': 800.0, 'e_rf_v_per_m': 10.0, 'field_v_per_m': 420.0}. Composed (multi-atom) worst-case relative error: `8.359e-16` at {'temperature_k': 500.0, 'e_coupling_v_per_m': 800.0, 'e_rf_v_per_m': 10.0, 'delta_c_hz': -18849555.92153876, 'delta_rf_hz': 6283185.307179586}. Tolerance `1e-07`. MET: `True`.

## C2: gradient validation, jax.grad vs central finite differences

Worst-case relative error `1.748e-06` (argument `e_coupling_v_per_m`), tolerance `1e-05`. MET: `True`.

| argument | jax.grad | central FD | relative error |
|---|---|---|---|
| field_v_per_m | 4.994862e-09 | 4.994862e-09 | 1.119e-09 |
| temperature_k | -6.432896e-08 | -6.432896e-08 | 2.787e-10 |
| e_coupling_v_per_m | -6.800475e-11 | -6.800463e-11 | 1.748e-06 |
| e_rf_v_per_m | -2.815622e-07 | -2.815621e-07 | 2.321e-07 |

## C5: field-reconstruction fit grid

8/8 fits converged. 8/8 report a positive-definite Hessian at the optimum, the condition the Laplace uncertainty below requires. 1/8 recovered all three parameters within their own reported 1-sigma Laplace uncertainty; 6/8 within 2-sigma.

| truth E0 (V/m) | truth grad (V/m/m) | truth patch (V/m) | seed | recovered E0 | recovered grad | recovered patch | Hessian PD | converged | 1-sigma | 2-sigma |
|---|---|---|---|---|---|---|---|---|---|---|
| 180.0 | 800.0 | 60.0 | 0 | 180.1 +/- 0.6 | 800.6 +/- 12.2 | 62.0 +/- 8.5 | True | True | True | True |
| 180.0 | 800.0 | 60.0 | 1 | 180.3 +/- 0.7 | 785.6 +/- 13.2 | 55.8 +/- 9.5 | True | True | False | True |
| 220.0 | -1200.0 | 90.0 | 0 | 218.9 +/- 0.7 | -1162.8 +/- 16.9 | 114.7 +/- 10.1 | True | True | False | False |
| 220.0 | -1200.0 | 90.0 | 1 | 219.2 +/- 0.7 | -1186.3 +/- 16.7 | 92.7 +/- 10.0 | True | True | False | True |
| 150.0 | 1500.0 | 40.0 | 0 | 149.7 +/- 0.7 | 1523.2 +/- 16.3 | 43.4 +/- 7.6 | True | True | False | True |
| 150.0 | 1500.0 | 40.0 | 1 | 148.9 +/- 0.7 | 1490.8 +/- 15.7 | 52.6 +/- 6.6 | True | True | False | True |
| 300.0 | -1800.0 | 120.0 | 0 | 291.1 +/- 1.7 | -1602.3 +/- 45.9 | 196.0 +/- 20.9 | True | True | False | False |
| 300.0 | -1800.0 | 120.0 | 1 | 298.3 +/- 1.6 | -1762.5 +/- 40.8 | 125.8 +/- 12.1 | True | True | False | True |

