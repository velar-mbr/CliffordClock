# WP33 motional Al+ ion intrinsic-micromotion-enhanced benchmark case (generated)

Generated: 2026-08-24T14:24:08.286647+00:00

## WP33: intrinsic-micromotion-enhanced variant (clock_ion_mathieu_parameters/radial_micromotion_enhancement, Al27+/Mg25+, Marshall)

**WP33 SCOPE NOTE: this case multiplies WP32's reconstructed radial participations by the leading-order intrinsic-micromotion enhancement factor F_axis = 1 + q^2/(2*a_axis+q^2) (Berkeland, Miller, Bergquist, Itano, Wineland, J. Appl. Phys. 83, 5025 (1998) Eq. 10), with (q, a_x, a_y) the clock (Al27+) ion's own leading-order Mathieu parameters solved from the trap's published RF drive frequency, the axial Coulomb curvature (WP32), and the two WP32-reconstructed clock-ion bare radial frequencies (cliffordclock.integrator.omega.clock_ion_mathieu_parameters): two equations, two unknowns, zero degrees of freedom, no trap-geometry parameter (alpha/epsilon) supplied as input. The axial modes are unchanged from WP31/WP32 (q_z=0, F_axial=1 identically: no intrinsic micromotion along the trap axis). Before use, the reconstruction chain is checked by an independent, falsifiable test: the clock ion's solved Mathieu parameters are mass-scaled to predict the PARTNER (Mg25+) ion's own bare radial frequencies, compared against WP32's separately-reconstructed partner frequencies (from the two-ion eigenproblem inversion, an entirely different calculation); see this case's own partner_prediction_note for the result. Reported per-mode and total-level agreement below is whatever this reconstruction gives against Marshall's own published per-mode and total rows; see this case's own enhancement_note for the result stated in full.**

**Over-determination check: mass-scaling the clock ion's own solved Mathieu parameters (q=0.191278, a_x=-5.884496e-03, a_y=+2.302550e-03, a_z=+3.581946e-03) to the partner ion (Mg25+) predicts bare radial frequencies of 4.336114e+06 Hz (X) and 5.468073e+06 Hz (Y), against WP32's own SEPARATELY reconstructed 4.360932e+06 Hz (X) and 5.497386e+06 Hz (Y); relative deviations -0.5691% (X) and -0.5332% (Y), both sub-1%-relative, well inside the few-percent band the published mode frequencies' own ~3-significant-figure reporting precision supports. This is an independent, falsifiable test (nothing in the Mathieu-parameter solve's own inputs touches the partner ion's frequencies at all) of the WHOLE reconstruction chain's internal consistency, reported as run.**

**Per-mode comparison against Marshall et al.'s own published 'Frequency shift per quantum' row (Table S2): the two AXIAL modes are unchanged from WP31/WP32 (enhancement=1.0 identically, q_z=0). The four RADIAL modes now use participation*enhancement (F_x=2.4742, F_y=1.8882) in place of WP32's plain participation; the resulting per-mode ratios (predicted/published) land at 1.62 sigma from the published total (NOT MET). Reported as run.**

| Quantity | Value |
|---|---|
| Clock-ion Mathieu q | 0.191278 |
| Clock-ion Mathieu a_x | -5.884496e-03 |
| Clock-ion Mathieu a_y | +2.302550e-03 |
| Clock-ion Mathieu a_z | +3.581946e-03 |
| Enhancement F_x | 2.4742 |
| Enhancement F_y | 1.8882 |
| Predicted partner bare freq, X (Hz) | 4.336114e+06 |
| WP32-reconstructed partner bare freq, X (Hz) | 4.360932e+06 |
| Partner X relative deviation | -0.5691% |
| Predicted partner bare freq, Y (Hz) | 5.468073e+06 |
| WP32-reconstructed partner bare freq, Y (Hz) | 5.497386e+06 |
| Partner Y relative deviation | -0.5332% |

| Mode | Axial? | Participation | Enhancement | Predicted shift/quantum | Published shift/quantum | Ratio (pred/pub) |
|---|---|---|---|---|---|---|
| axial_com | True | 0.5383 | 1.0000 | -9.5667e-20 | -9.5000e-20 | +1.0070 |
| axial_str | True | 0.4617 | 1.0000 | -1.4244e-19 | -1.4200e-19 | +1.0031 |
| x_com | False | 0.2138 | 2.4742 | -1.8369e-19 | -1.7700e-19 | +1.0378 |
| x_str | False | 0.7862 | 2.4742 | -5.5693e-19 | -6.4800e-19 | +0.8595 |
| y_com | False | 0.1663 | 1.8882 | -1.3873e-19 | -1.4200e-19 | +0.9770 |
| y_str | False | 0.8337 | 1.8882 | -6.1521e-19 | -6.5300e-19 | +0.9421 |

| Quantity | Value |
|---|---|
| Enhancement-corrected total (P-1)_motional | -1.063848e-17 |
| Enhancement-corrected uncertainty (1-sigma) | +/-3.373e-19 |
| Enhancement-corrected band | [-1.097582e-17, -1.030115e-17] |
| Total bands overlap | False |
| **total_kpi_verdict** | **NOT MET** |

Per-mode published-value citation: Marshall et al., arXiv:2504.13071v2 (2025), Supplemental Material Table S2, "Frequency shift per quantum (10^-19)" row: -0.95, -1.42, -1.77, -6.48, -1.42, -6.53 for Axial COM, Axial STR, X COM, X STR, Y COM, Y STR respectively (same mode order as MARSHALL_AL_ION_MODES_MHZ_NBAR). This is the paper's own PUBLISHED per-mode time-dilation weight (distinct from the unrelated 'Geometric factor kappa' row, a Doppler-cooling-laser geometry factor, MARSHALL_AL_ION_MODES_CITATION's own caveat). Re-fetched and confirmed against the arXiv PDF text this session.

## WP33 Brewer et al. (2019, arXiv:1902.07694) consistency check (second, independent dataset)

**Brewer et al.'s own total-level secular-motion row (-17.3(2.9)e-19) is NOT reproduced here: Table S2 publishes a 95%-CI BOUND on n_bar_0 (zero-point energy excluded) combined with a per-mode heating rate n_bar_dot through Brewer's own Eq. 3 (a time-dependent model over the 150 ms interrogation time), not the static n_bar point estimate E38's formula consumes; the SAME missing-input reason run_motional_al_ion.py's module docstring already states for why WP30/31/32 use Marshall et al. instead of Brewer for their own total-level cases. What IS available from Brewer's Table S2, the RF drive frequency, all six mode frequencies, and a per-mode TDS/quantum row that already includes the transverse intrinsic-micromotion shift (footnote a), is what this consistency check uses: the over-determination check and the per-mode ratios above, both independent of n_bar.**

| Quantity | Value |
|---|---|
| Clock-ion Mathieu q | 0.248485 |
| Clock-ion Mathieu a_x | -7.898300e-03 |
| Clock-ion Mathieu a_y | +2.874493e-03 |
| Clock-ion Mathieu a_z | +5.023807e-03 |
| Enhancement F_x | 2.3438 |
| Enhancement F_y | 1.9148 |
| Partner X relative deviation | -0.9494% |
| Partner Y relative deviation | -0.7891% |
| Per-mode ratio (pred/pub), x_com | +0.8991 |
| Per-mode ratio (pred/pub), x_str | +0.8843 |
| Per-mode ratio (pred/pub), y_com | +0.8762 |
| Per-mode ratio (pred/pub), y_str | +0.9325 |
