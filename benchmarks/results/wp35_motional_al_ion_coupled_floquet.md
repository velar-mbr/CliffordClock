# WP35 motional Al+ ion coupled-two-ion-Floquet benchmark case (generated)

Generated: 2026-08-24T17:21:04.893327+00:00

## WP35 HEADLINE: the constrained two-ion Mathieu fit (constrained_two_ion_mathieu_fit, Al27+/Mg25+, Marshall, G17 gate fix loop)

**WP35 HEADLINE (G17 gate fix loop): this case fits a SINGLE shared Mathieu RF parameter magnitude q (Berkeland Eq. 6, q_x=-q_y) and a DC-split fraction alpha (a_x=-alpha*a_z, a_y=-(1-alpha)*a_z, satisfying the Laplace constraint a_x+a_y=-a_z for any alpha) against ALL FOUR measured radial mode frequencies at once, with a_z fixed exactly from the measured axial frequency (the same relation WP34 uses). Four equations, two unknowns: a genuine over-determination, solved by Gauss-Newton least squares (cliffordclock.integrator.omega.constrained_two_ion_mathieu_fit), with real, nonzero frequency residuals reported in full as this case's own falsifiable output, the same epistemic shape as WP32/WP33/WP34's own over-determination checks. This replaces the per-axis diagnostic case's own two-equations-two-unknowns-per-axis solve, which fits each axis exactly by construction, so it alone provides no independent evidence the underlying model is correct. Reported per-mode and total-level agreement below is whatever this fit gives against Marshall's own published per-mode and total rows, with no tuning; the fit's own residuals, and a model-structure uncertainty component derived from them, are reported alongside so a reader can see exactly how much of the agreement is forced and how much is earned. A note on energy bookkeeping: WP31/WP32's plain participations sum to 1.0 across the two ions sharing a mode, correct for secular motion alone; once enhancement multiplies participation, the two ions' participation*enhancement shares add to something larger than 1.0 in general, because the RF drive itself supplies the additional kinetic energy the enhancement factor accounts for. That larger sum is the expected signature of a driven system.**

**Constrained fit: q=0.190645, alpha=+1.625194, a_x=-5.821357e-03, a_y=+2.239411e-03, a_z=+3.581946e-03. Frequency residuals (predicted-measured, the fit's own falsifiable output): X-COM -782.7 Hz (-0.0185%), X-STR +761.9 Hz (+0.0219%), Y-COM -3862.1 Hz (-0.0719%), Y-STR +3721.0 Hz (+0.0783%). These residuals quantify whatever real physics (trap anharmonicity, higher multipoles, or any other departure from the idealized Mathieu model) this two-parameter constrained fit cannot absorb; the resulting total lands at 0.08 sigma from the published total (MET).**

**Over-determination check (constrained fit): mass-scaling the constrained fit's own (q=0.190645, a_x=-5.821357e-03, a_y=+2.239411e-03) to the partner ion (Mg25+) predicts bare radial frequencies of 4.357681e+06 Hz (X) and 5.491603e+06 Hz (Y), against WP32's own SEPARATELY reconstructed 4.360932e+06 Hz (X) and 5.497386e+06 Hz (Y); relative deviations -0.0745% (X) and -0.1052% (Y), both well inside the few-percent band the published mode frequencies' own ~3-significant-figure reporting precision supports.**

**Model-structure uncertainty (G17 item 3, a component this project's earlier motional cases did not carry): the per-axis diagnostic case's own total is -1.157386e-17, against this case's constrained total of -1.141472e-17; the spread, +/-1.591e-19, is reported as its own labeled uncertainty component, standing for how much of the total depends on which of the two defensible Mathieu models is used, beside the thermometry and rounding channels.**

**Input-rounding uncertainty channel (WP34 Part 2, reused unchanged here): propagating each published frequency's half-last-digit rounding bound through this constrained-fit chain gives a total-level rounding uncertainty of +/-1.022e-20 (fractional), against the thermometry (n_bar) channel's own +/-3.763e-19, roughly 37x. This channel is a BOUND on rounding, distinct from a measured uncertainty, and is labeled as such throughout this case's own record.**

| Quantity | Value |
|---|---|
| Constrained q | 0.190645 |
| Constrained alpha (DC split) | +1.625194 |
| Constrained a_x | -5.821357e-03 |
| Constrained a_y | +2.239411e-03 |
| Constrained a_z (fixed from axial) | +3.581946e-03 |

| Mode | Axial? | Measured freq (Hz) | Predicted freq (Hz) | Participation | Enhancement | Predicted shift/quantum | Published shift/quantum | Ratio |
|---|---|---|---|---|---|---|---|---|
| axial_com | True | 2.160000e+06 | 2.160000e+06 | 0.5383 | 1.0000 | -9.5667e-20 | -9.5000e-20 | +1.0070 |
| axial_str | True | 3.750000e+06 | 3.750000e+06 | 0.4617 | 1.0000 | -1.4244e-19 | -1.4200e-19 | +1.0031 |
| x_com | False | 4.220000e+06 | 4.219217e+06 | 0.2270 | 2.2949 | -1.8089e-19 | -1.7700e-19 | +1.0220 |
| x_str | False | 3.480000e+06 | 3.480762e+06 | 0.7722 | 2.8915 | -6.3927e-19 | -6.4800e-19 | +0.9865 |
| y_com | False | 5.370000e+06 | 5.366138e+06 | 0.1803 | 1.8094 | -1.4413e-19 | -1.4200e-19 | +1.0150 |
| y_str | False | 4.750000e+06 | 4.753721e+06 | 0.8190 | 2.0248 | -6.4810e-19 | -6.5300e-19 | +0.9925 |

| Quantity | Value |
|---|---|
| Constrained-fit total (P-1)_motional | -1.141472e-17 |
| Uncertainty component: thermometry (n_bar), 1-sigma | +/-3.763e-19 |
| Uncertainty component: input rounding, 1-sigma | +/-1.022e-20 |
| Uncertainty component: model structure, 1-sigma | +/-1.591e-19 |
| Uncertainty, combined in quadrature, 1-sigma | +/-4.087e-19 |
| Band | [-1.182340e-17, -1.100604e-17] |
| Total bands overlap | True |
| **total_kpi_verdict** | **MET** |

Per-mode published-value citation: Marshall et al., arXiv:2504.13071v2 (2025), Supplemental Material Table S2, "Frequency shift per quantum (10^-19)" row: -0.95, -1.42, -1.77, -6.48, -1.42, -6.53 for Axial COM, Axial STR, X COM, X STR, Y COM, Y STR respectively (same mode order as MARSHALL_AL_ION_MODES_MHZ_NBAR). This is the paper's own PUBLISHED per-mode time-dilation weight (distinct from the unrelated 'Geometric factor kappa' row, a Doppler-cooling-laser geometry factor, MARSHALL_AL_ION_MODES_CITATION's own caveat). Re-fetched and confirmed against the arXiv PDF text this session.

## WP35 constrained-fit Brewer et al. (2019, arXiv:1902.07694) consistency check (second, independent dataset)

**Brewer et al.'s own total-level secular-motion row (-17.3(2.9)e-19) is NOT reproduced here, unchanged reason from WP33/WP34/the per-axis diagnostic case's own Brewer checks: Table S2 publishes a 95%-CI BOUND on n_bar_0 combined with a per-mode heating rate through Brewer's own time-dependent Eq. 3, a different input shape from the static n_bar point estimate this project's formula consumes. What IS available, the RF drive frequency, all six mode frequencies, and a per-mode TDS/quantum row that already includes the transverse intrinsic-micromotion shift (footnote a), is what this consistency check uses: the constrained fit's own per-mode ratios and partner over-determination check above, both independent of n_bar.**

| Quantity | Value |
|---|---|
| Constrained q | 0.246887 |
| Constrained alpha | +1.540194 |
| Constrained a_x | -7.737639e-03 |
| Constrained a_y | +2.713831e-03 |
| Frequency residual, X-COM (Hz) | -4193.1 |
| Frequency residual, X-STR (Hz) | +4007.6 |
| Frequency residual, Y-COM (Hz) | -4888.8 |
| Frequency residual, Y-STR (Hz) | +4908.1 |
| Partner X relative deviation | -0.1799% |
| Partner Y relative deviation | -0.1404% |
| Per-mode ratio (pred/pub), x_com | +0.9490 |
| Per-mode ratio (pred/pub), x_str | +0.9512 |
| Per-mode ratio (pred/pub), y_com | +0.9594 |
| Per-mode ratio (pred/pub), y_str | +0.9669 |

## WP35 diagnostic variant: the per-axis solve (coupled_two_ion_floquet_modes/coupled_two_ion_mathieu_parameters, Al27+/Mg25+, Marshall)

**WP35 DIAGNOSTIC VARIANT: this case fits EACH axis's own two unknowns (q, a_axis) from that SAME axis's own two measured frequencies alone: two equations, two unknowns, zero residual by construction, so its near-unity per-mode ratios carry no independent evidence that the underlying model is right (a perfect fit here is guaranteed regardless of the physics). This case's OWN scope note: it replaces WP33/WP34's participation*enhancement factorization (a single per-axis F_axis applied identically to a pair's COM and STR members) with the coupled two-ion Floquet solve: the two ions' time-periodic equations of motion, coupled by the SAME Coulomb curvature WP32 uses, are integrated directly (cliffordclock.integrator.omega.coupled_two_ion_floquet_modes), and each mode's clock-ion participation AND micromotion enhancement are both read from the SAME per-mode Fourier decomposition of the coupled solution, no per-ion F factorization anywhere. The clock ion's own (q, a_axis) are solved fully self-consistently per axis: the coupled system's own two quasi-frequencies are matched directly to the two measured mode frequencies (coupled_two_ion_mathieu_parameters), with no WP32 bare-frequency reconstruction consumed as an intermediate step. Verified this session against two exact limits: c -> 0 reproduces WP34's own single-ion exact Floquet result for each ion separately, and q -> 0 (RF -> 0) reproduces WP32's own static secular participation decomposition exactly; both checks are in this WP's own test suite. A second, cheap comparison column (mathieu_forced_oscillator_enhancement) evaluates a reviewer's forced-oscillator estimate at each mode's own exact quasi-frequency, using the leading nearest-neighbor sideband approximation in place of the full coupled solve; it is reported alongside the rigorous result, its own labeled column. Reported per-mode and total-level agreement below is whatever this reconstruction gives against Marshall's own published per-mode and total rows, with no tuning. See the CONSTRAINED fit case for this WP's own headline result and its genuine over-determination check.**

**Fully self-consistent per-axis solve: X axis gives q=0.191625, a_x=-6.016634e-03; Y axis gives q=0.195942, a_y=+1.162899e-03. Each axis is solved INDEPENDENTLY from that axis's own two measured mode frequencies alone, with no cross-axis constraint used in the solve itself. The two independently-solved q values differ by +2.25% relative, a genuine self-consistency check on the whole reconstruction chain. WP34's own single shared exact solve for this dataset gives q=0.190083, a_x=-5.809270e-03, a_y=+2.227324e-03, reported alongside for direct comparison.**

**Per-mode comparison against Marshall et al.'s own published 'Frequency shift per quantum' row (Table S2), coupled two-ion Floquet solve: the two AXIAL modes are unchanged (enhancement=1.0 identically, q_z=0). The four RADIAL modes use the COUPLED solve's own participation and enhancement, both read from the SAME per-mode Fourier decomposition. The resulting total lands at 0.20 sigma from the published total (MET). Reported as run, no tuning.**

| Mode | Axial? | Mode freq (Hz) | Participation | Enhancement (exact) | Enhancement (reviewer estimate) | Predicted (exact) | Predicted (reviewer) | Published | Ratio (exact) | Ratio (reviewer) |
|---|---|---|---|---|---|---|---|---|---|---|
| axial_com | True | 2.160000e+06 | 0.5383 | 1.0000 | 1.0000 | -9.5667e-20 | -9.5667e-20 | -9.5000e-20 | +1.0070 | +1.0070 |
| axial_str | True | 3.750000e+06 | 0.4617 | 1.0000 | 1.0000 | -1.4244e-19 | -1.4244e-19 | -1.4200e-19 | +1.0031 | +1.0031 |
| x_com | False | 4.220000e+06 | 0.2258 | 2.3076 | 3.1181 | -1.8092e-19 | -2.4446e-19 | -1.7700e-19 | +1.0221 | +1.3811 |
| x_str | False | 3.480000e+06 | 0.7734 | 2.9116 | 4.1076 | -6.4474e-19 | -9.0957e-19 | -6.4800e-19 | +0.9950 | +1.4037 |
| y_com | False | 5.370000e+06 | 0.1752 | 1.8535 | 2.3770 | -1.4346e-19 | -1.8399e-19 | -1.4200e-19 | +1.0103 | +1.2957 |
| y_str | False | 4.750000e+06 | 0.8241 | 2.0837 | 2.7556 | -6.7109e-19 | -8.8748e-19 | -6.5300e-19 | +1.0277 | +1.3591 |

| Quantity | Value |
|---|---|
| Per-axis-solve total (P-1)_motional | -1.157386e-17 |
| **total_kpi_verdict** | **MET** |

## WP35 per-axis-diagnostic Brewer et al. (2019, arXiv:1902.07694) consistency check

**Brewer et al.'s own total-level secular-motion row (-17.3(2.9)e-19) is NOT reproduced here, unchanged reason from WP33/WP34's own Brewer checks: Table S2 publishes a 95%-CI BOUND on n_bar_0 combined with a per-mode heating rate through Brewer's own time-dependent Eq. 3, a different input shape from the static n_bar point estimate this project's formula consumes. What IS available, the RF drive frequency, all six mode frequencies, and a per-mode TDS/quantum row that already includes the transverse intrinsic-micromotion shift (footnote a), is what this consistency check uses: the coupled-Floquet per-mode ratios above, independent of n_bar.**

| Quantity | Value |
|---|---|
| X-axis per-axis solve q | 0.254745 |
| Y-axis per-axis solve q | 0.257082 |
| Per-mode ratio (pred/pub), x_com | +0.9431 |
| Per-mode ratio (pred/pub), x_str | +0.9985 |
| Per-mode ratio (pred/pub), y_com | +0.9471 |
| Per-mode ratio (pred/pub), y_str | +1.0174 |
