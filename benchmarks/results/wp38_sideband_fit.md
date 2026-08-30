# WP38 Deliverable 3: gradient-based sideband-fitting demonstration

Generated: 2026-08-30T04:18:11.017864+00:00

Synthetic round-trip: generator == fitter. Gradient-based optimization through this project's own differentiable forward model recovers known (u0, Tr) with correctly calibrated Laplace uncertainties. Real-data model accuracy is a separate question; run_sideband_spectrum.py and Goti et al. 2025 address it directly. large-lattice-model's own get_fit_sidebands is a non-differentiable BO+WKB sideband fitter that already exists and was used for Goti et al. 2025's own real-data fits; this file's own contribution is the first GRADIENT-based (autodiff) fit of the BO+WKB model. Every case's Hessian is checked for positive definiteness (hessian_positive_definite) before its inverse is trusted as a covariance. At a saddle point, the Hessian carries a negative eigenvalue and the Laplace approximation does not apply; the reported uncertainty there is nan.

12/12 fits converged. 11/12 report a positive-definite Hessian at the optimum, the condition the Laplace uncertainty below requires. 7/12 recovered both parameters within their own reported 1-sigma Laplace uncertainty; 11/12 within 2-sigma.

| model | truth u0 | truth Tr (uK) | seed | recovered u0 | recovered Tr (uK) | Hessian PD | converged | 1-sigma | 2-sigma |
|---|---|---|---|---|---|---|---|---|---|
| harmonic | 80.0 | 0.60 | 0 | 80.06 +/- 0.30 | 0.584 +/- 0.059 | True | True | True | True |
| harmonic | 80.0 | 0.60 | 1 | 80.36 +/- 0.31 | 0.651 +/- 0.073 | True | True | False | True |
| harmonic | 100.0 | 1.00 | 0 | 100.96 +/- nan | 1.959 +/- nan | False | True | False | False |
| harmonic | 100.0 | 1.00 | 1 | 100.30 +/- 0.36 | 1.189 +/- 0.209 | True | True | True | True |
| harmonic | 120.0 | 1.50 | 0 | 119.50 +/- 0.46 | 1.330 +/- 0.277 | True | True | False | True |
| harmonic | 120.0 | 1.50 | 1 | 120.23 +/- 0.41 | 1.777 +/- 0.443 | True | True | True | True |
| bowkb | 80.0 | 0.60 | 0 | 80.51 +/- 0.49 | 0.624 +/- 0.029 | True | True | False | True |
| bowkb | 80.0 | 0.60 | 1 | 80.03 +/- 0.24 | 0.603 +/- 0.017 | True | True | True | True |
| bowkb | 100.0 | 1.00 | 0 | 100.01 +/- 0.49 | 0.996 +/- 0.028 | True | True | True | True |
| bowkb | 100.0 | 1.00 | 1 | 99.91 +/- 0.51 | 0.994 +/- 0.029 | True | True | True | True |
| bowkb | 120.0 | 1.50 | 0 | 119.14 +/- 0.55 | 1.467 +/- 0.033 | True | True | False | True |
| bowkb | 120.0 | 1.50 | 1 | 119.64 +/- 0.56 | 1.485 +/- 0.033 | True | True | True | True |

**Hessian not positive definite.** The row(s) below stopped at a saddle point of the negative log-likelihood: the Hessian at the reported optimum carries a negative eigenvalue, so the Laplace approximation is invalid there. Each such row's own uncertainty is reported as `nan`.

- `harmonic`, truth `u0=100.0`, `Tr=1.00 uK`, seed `0`: recovered `u0=100.96`, `Tr=1.959 uK`, Hessian eigenvalues carry at least one negative value; the Laplace uncertainty at this optimum is undefined.

## Goti et al. 2025 real-scan fit: assessment

**Decision: declined.**

Goti et al. 2025 Figs. 4/7 plot real IT-Yb1 sideband scans as discrete scatter markers, the strongest figure-digitization candidate found in this project's research sweep. The underlying PDF's text/vector layer carries the paper's prose and equations, and no separately recoverable per-marker coordinate stream. Extracting exact coordinates from these figures would need pixel-level digitization of the published art, placing that extraction in the figure-digitization class, weaker than either this file's synthetic fits or run_sideband_spectrum.py's independent-implementation cross-validation. This work package's own instruction says plainly: do not force it. No real-scan fit is shipped here; the raw scan data behind Figs. 4/7 (detuning, excitation fraction, per-point uncertainties) is recorded as the named partnership ask.

