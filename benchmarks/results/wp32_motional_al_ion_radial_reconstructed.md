# WP32 motional Al+ ion radial-spectrum-reconstructed benchmark case (generated)

Generated: 2026-08-23T14:33:02.172712+00:00

## WP32: radial-spectrum-reconstructed variant (two_ion_radial_participations, Al27+/Mg25+)

**WP32 SCOPE NOTE: this case's X/Y participations are NOT the axial mu-only closed form reused for radial (WP31's documented approximation, still reported unmodified in the participation-corrected variant case above); they are reconstructed directly from Marshall et al.'s own measured axial-COM and radial mode frequencies (cliffordclock.integrator.omega.axial_coulomb_curvature/two_ion_radial_participations), inverting the two-ion radial eigenproblem for each transverse direction's two unknown bare radial frequencies with no trap RF/DC geometry parameter (epsilon, alpha) as input. The disambiguation assumption (RF pseudopotential scaling: the lighter ion, Mg25+, carries the higher bare radial frequency) is applied identically to both the X and Y branches. Reported per-mode and total-level agreement below is whatever this reconstruction gives against Marshall's own published per-mode and total rows, with no tuning; see this case's own participation_note for the result stated in full.**

**Radial-spectrum-reconstructed per-mode comparison against Marshall et al.'s own published 'Frequency shift per quantum' row (Table S2): the two AXIAL modes are unchanged from the WP31 variant (two_ion_participations' exact mu-only closed form). The four RADIAL modes now use two_ion_radial_participations' reconstruction from the measured X/Y spectra instead of the axial-form approximation; the resulting per-mode ratios (predicted/published) sit in the same rough range as WP31's radial rows, landing at 14.01 sigma from the published total (NOT MET), essentially unchanged from WP31's own radial-approximation total. Because each mode-pair's clock-ion participations sum to 1.0 exactly regardless of how the pair's total is split between its COM and STR members, and Marshall's own COM/STR (n_bar+1/2)-weighted magnitudes for a given branch are comparable in size, redistributing participation within a radial pair moves the per-mode ratios without moving the pair's own total much, a structural reason a correctly reconstructed split need not by itself close a total-level gap this size. The reconstruction, its cross-check, and this result are reported as run, with no tuning.**

| Quantity | Value |
|---|---|
| Coulomb curvature c (N/m), from axial COM | 3.976554e-12 |
| Coulomb curvature c (N/m), cross-check from axial STR | 3.983457e-12 |
| Cross-check relative deviation | +1.7360e-03 |
| Bare radial frequency, clock ion, X branch (Hz) | 3.946772e+06 |
| Bare radial frequency, partner ion, X branch (Hz) | 4.360932e+06 |
| Bare radial frequency, clock ion, Y branch (Hz) | 5.084691e+06 |
| Bare radial frequency, partner ion, Y branch (Hz) | 5.497386e+06 |

| Mode | Axial? | Participation | Predicted shift/quantum | Published shift/quantum | Ratio (pred/pub) |
|---|---|---|---|---|---|
| axial_com | True | 0.5383 | -9.5667e-20 | -9.5000e-20 | +1.0070 |
| axial_str | True | 0.4617 | -1.4244e-19 | -1.4200e-19 | +1.0031 |
| x_com | False | 0.2138 | -7.4243e-20 | -1.7700e-19 | +0.4195 |
| x_str | False | 0.7862 | -2.2509e-19 | -6.4800e-19 | +0.3474 |
| y_com | False | 0.1663 | -7.3472e-20 | -1.4200e-19 | +0.5174 |
| y_str | False | 0.8337 | -3.2582e-19 | -6.5300e-19 | +0.4990 |

| Quantity | Value |
|---|---|
| Reconstructed-participation total (P-1)_motional | -5.716945e-18 |
| Reconstructed-participation uncertainty (1-sigma) | +/-1.535e-19 |
| Reconstructed-participation band | [-5.870449e-18, -5.563441e-18] |
| Total bands overlap | False |
| **total_kpi_verdict** | **NOT MET** |

Per-mode published-value citation: Marshall et al., arXiv:2504.13071v2 (2025), Supplemental Material Table S2, "Frequency shift per quantum (10^-19)" row: -0.95, -1.42, -1.77, -6.48, -1.42, -6.53 for Axial COM, Axial STR, X COM, X STR, Y COM, Y STR respectively (same mode order as MARSHALL_AL_ION_MODES_MHZ_NBAR). This is the paper's own PUBLISHED per-mode time-dilation weight (distinct from the unrelated 'Geometric factor kappa' row, a Doppler-cooling-laser geometry factor, MARSHALL_AL_ION_MODES_CITATION's own caveat). Re-fetched and confirmed directly against the arXiv PDF text this session.
