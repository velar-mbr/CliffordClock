# WP34 motional Al+ ion exact-Floquet-enhanced benchmark case (generated)

Generated: 2026-08-24T14:24:08.305356+00:00

## WP34: exact-Floquet-enhanced variant (clock_ion_mathieu_parameters_exact/radial_micromotion_enhancement_exact, Al27+/Mg25+, Marshall)

**WP34 SCOPE NOTE: this case replaces WP33's leading-order Mathieu bracket (F_axis = 1 + q^2/(2*a_axis+q^2)) with the numerically exact Floquet velocity-variance enhancement (cliffordclock.integrator.omega.radial_micromotion_enhancement_exact), and WP33's closed-form (q, a_x, a_y) inversion with a 2D Newton solve against the exact Mathieu characteristic exponent (clock_ion_mathieu_parameters_exact), the SAME two-equations-two-unknowns structure. The exact beta(a, q) map has no closed-form inverse, so this step requires a numerical solve. Both the exact continued-fraction solve and the exact partner-frequency over-determination check were verified this session against independent monodromy-matrix ODE integration (agreement to float64 precision); see cliffordclock.integrator.omega's own WP34 comment block for the full derivation. This case ALSO adds an input-rounding uncertainty channel WP33 did not carry: Marshall's published frequencies are given to a fixed number of decimal places with no stated measurement uncertainty, so each one carries an unavoidable rounding bound (half its last printed digit), propagated by finite differences through the full reconstruction chain and combined in quadrature with the existing phonon-number (thermometry) uncertainty, reported as a SEPARATE labeled component (see this case's own predicted_total_uncertainty_nbar_fractional/predicted_total_uncertainty_rounding_fractional/predicted_total_uncertainty_combined_fractional fields): a bound on rounding, distinct from a measured uncertainty and labeled separately for that reason. Reported per-mode and total-level agreement below is whatever this reconstruction gives against Marshall's own published per-mode and total rows; see this case's own enhancement_note and structural_note for the results stated in full, with no tuning.**

**Over-determination check (exact): mass-scaling the clock ion's own EXACT-solved Mathieu parameters (q=0.190083, a_x=-5.809270e-03, a_y=+2.227324e-03, a_z=+3.581946e-03) to the partner ion (Mg25+) and evaluating the EXACT beta(a, q) predicts bare radial frequencies of 4.341117e+06 Hz (X) and 5.475275e+06 Hz (Y), against WP32's own SEPARATELY reconstructed 4.360932e+06 Hz (X) and 5.497386e+06 Hz (Y); relative deviations -0.4544% (X) and -0.4022% (Y). WP33's own leading-order solve, re-evaluated for this same dataset, gives deviations -0.5691% (X) and -0.5332% (Y): the exact treatment moves the over-determination check's own margin, both branches staying sub-1%-relative, well inside the few-percent band the published mode frequencies' own ~3-significant-figure reporting precision supports.**

**Per-mode comparison against Marshall et al.'s own published 'Frequency shift per quantum' row (Table S2), exact-Floquet enhancement: the two AXIAL modes are unchanged (enhancement=1.0 identically, q_z=0). The four RADIAL modes use EXACT participation*enhancement (exact F_x=2.4677 vs leading-order 2.4742; exact F_y=1.8934 vs leading-order 1.8882); the resulting total lands at 1.62 sigma from the published total (NOT MET), against 1.62 sigma for WP33's own leading-order treatment of this SAME radial-spectrum reconstruction (re-evaluated here for an apples-to-apples comparison, thermometry uncertainty only). The two totals stay close: the exact-vs-leading-order correction to F_x/F_y here is at the sub-1%-relative level, since this project's own (a, q) sit well inside Berkeland's stated leading-order validity regime, too small to move the total-level verdict. Reported as run, no tuning.**

**Input-rounding uncertainty channel (Part 2 of this WP): propagating each published frequency's half-last-digit rounding bound (+/-5000 Hz on each of the six mode frequencies, +/-5000 Hz on the RF drive frequency) through the full reconstruction chain gives a total-level rounding uncertainty of +/-7.325e-21 (fractional), against the thermometry (n_bar) channel's own +/-3.368e-19: the rounding bound is roughly 46x smaller than the thermometry uncertainty here, so it leaves the reported band essentially where the thermometry channel alone would place it. It is still reported as its OWN separately labeled component here, since Marshall et al.'s Table S2 states no uncertainty at all for these inputs. This channel is a BOUND on rounding, distinct from a measured uncertainty, and is labeled as such throughout this case's own record.**

**Structural question (does the exact treatment split COM from STR within an axis): the answer is NO, by construction: F_axis depends only on the axis's own (a, q), so the identical F_x multiplies both x_com and x_str, and the identical F_y multiplies both y_com and y_str, in both the leading-order and exact treatments. The X-axis per-mode ratios here are x_com=+1.0351 (+3.51%) and x_str=+0.8572 (-14.28%), opposite-sign deviations from 1.0 within the same axis, essentially unchanged from WP33's leading-order pattern: a single per-axis multiplicative enhancement factor scales an existing within-axis deviation, and cannot flip its sign. This locates the remaining scatter outside the single-per-axis-enhancement-factor model implemented here (WP33's and WP34's own F_axis, applied identically to both a pair's COM and STR members). A genuinely per-mode mechanism, still within Mathieu-order physics but coupling the two ions' motion beyond that single shared factor, remains open; the published rows' own per-mode calibration chain (for example Marshall's own COM/STR mode-frequency measurement or per-mode Doppler-cooling-limit corrections) is another candidate this case does not rule out either. The Y-axis ratios (y_com=+0.9797, y_str=+0.9447) both land on the same side of 1.0, so this sign flip is specific to the X branch. Given this, the TOTAL is the right level at which to compare this project's reconstruction against Marshall's published number; the individual per-mode rows carry this additional, uncorrected scatter.**

| Quantity | Exact (WP34) | Leading order (WP33) |
|---|---|---|
| Clock-ion Mathieu q | 0.190083 | 0.191278 |
| Clock-ion Mathieu a_x | -5.809270e-03 | -5.884496e-03 |
| Clock-ion Mathieu a_y | +2.227324e-03 | +2.302550e-03 |
| Clock-ion Mathieu a_z | +3.581946e-03 | +3.581946e-03 |
| Enhancement F_x | 2.4677 | 2.4742 |
| Enhancement F_y | 1.8934 | 1.8882 |

| Predicted partner bare freq, X (Hz) | 4.341117e+06 |
| WP32-reconstructed partner bare freq, X (Hz) | 4.360932e+06 |
| Partner X relative deviation | -0.4544% |
| Predicted partner bare freq, Y (Hz) | 5.475275e+06 |
| WP32-reconstructed partner bare freq, Y (Hz) | 5.497386e+06 |
| Partner Y relative deviation | -0.4022% |

| Mode | Axial? | Participation | Enhancement | Predicted shift/quantum | Published shift/quantum | Ratio (pred/pub) | Rounding uncertainty |
|---|---|---|---|---|---|---|---|
| axial_com | True | 0.5383 | 1.0000 | -9.5667e-20 | -9.5000e-20 | +1.0070 | +/-2.21e-22 |
| axial_str | True | 0.4617 | 1.0000 | -1.4244e-19 | -1.4200e-19 | +1.0031 | +/-1.90e-22 |
| x_com | False | 0.2138 | 2.4677 | -1.8321e-19 | -1.7700e-19 | +1.0351 | +/-5.04e-21 |
| x_str | False | 0.7862 | 2.4677 | -5.5546e-19 | -6.4800e-19 | +0.8572 | +/-4.55e-21 |
| y_com | False | 0.1663 | 1.8934 | -1.3911e-19 | -1.4200e-19 | +0.9797 | +/-4.16e-21 |
| y_str | False | 0.8337 | 1.8934 | -6.1692e-19 | -6.5300e-19 | +0.9447 | +/-3.57e-21 |

| Quantity | Value |
|---|---|
| Exact-enhancement-corrected total (P-1)_motional | -1.063590e-17 |
| Uncertainty component: thermometry (n_bar), 1-sigma | +/-3.368e-19 |
| Uncertainty component: input rounding, 1-sigma | +/-7.325e-21 |
| Uncertainty, combined in quadrature, 1-sigma | +/-3.369e-19 |
| Band | [-1.097280e-17, -1.029899e-17] |
| Total bands overlap | False |
| **total_kpi_verdict** | **NOT MET** |

Per-mode published-value citation: Marshall et al., arXiv:2504.13071v2 (2025), Supplemental Material Table S2, "Frequency shift per quantum (10^-19)" row: -0.95, -1.42, -1.77, -6.48, -1.42, -6.53 for Axial COM, Axial STR, X COM, X STR, Y COM, Y STR respectively (same mode order as MARSHALL_AL_ION_MODES_MHZ_NBAR). This is the paper's own PUBLISHED per-mode time-dilation weight (distinct from the unrelated 'Geometric factor kappa' row, a Doppler-cooling-laser geometry factor, MARSHALL_AL_ION_MODES_CITATION's own caveat). Re-fetched and confirmed against the arXiv PDF text this session.

## WP34 Brewer et al. (2019, arXiv:1902.07694) consistency check (second, independent dataset, exact treatment)

**Brewer et al.'s own total-level secular-motion row (-17.3(2.9)e-19) is NOT reproduced here, unchanged reason from WP33's own Brewer check: Table S2 publishes a 95%-CI BOUND on n_bar_0 combined with a per-mode heating rate through Brewer's own time-dependent Eq. 3, a different input shape from the static n_bar point estimate this project's formula consumes. What IS available, the RF drive frequency, all six mode frequencies, and a per-mode TDS/quantum row that already includes the transverse intrinsic-micromotion shift (footnote a), is what this consistency check uses: the exact over-determination check and the exact per-mode ratios above, both independent of n_bar.**

| Quantity | Value |
|---|---|
| Clock-ion Mathieu q | 0.245805 |
| Clock-ion Mathieu a_x | -7.730668e-03 |
| Clock-ion Mathieu a_y | +2.706861e-03 |
| Clock-ion Mathieu a_z | +5.023807e-03 |
| Enhancement F_x | 2.3365 |
| Enhancement F_y | 1.9224 |
| Partner X relative deviation | -0.7514% |
| Partner Y relative deviation | -0.5676% |
| Per-mode ratio (pred/pub), x_com | +0.8963 |
| Per-mode ratio (pred/pub), x_str | +0.8816 |
| Per-mode ratio (pred/pub), y_com | +0.8796 |
| Per-mode ratio (pred/pub), y_str | +0.9362 |
