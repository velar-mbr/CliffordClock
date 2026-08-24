# WP30 motional Al+ ion benchmark case (generated)

Generated: 2026-08-24T02:37:00.182799+00:00

## Arithmetic-reproduction case: Marshall et al. arXiv:2504.13071 Table I "Secular motion" row

**Classification label (binding): arithmetic reproduction of a published standard-formula evaluation (arithmetic-reproduction fidelity, validates the engine's E38 implementation and unit chain, NOT an independent motional-Doppler physics prediction; Marshall et al.'s secular-motion row is computed from their own measured inputs through the same standard formula)**

**SCOPE CAVEAT (corrected per the project's G11 gate record, section A3): Marshall et al.'s six modes are TWO-ION (27Al+/25Mg+) crystal normal modes. The physically complete per-mode evaluation partitions each mode's zero-point and thermal motion between the two ions by their normal-mode amplitudes. THIS case (participation=1.0 throughout, the single-species-mass formula) does not consume that partition, by deliberate construction: it isolates the WP30 single-mass TOTAL-level reproduction as a case independent of the WP31 participation-corrected variant reported alongside it below. As a result this case's per-mode contributions differ from Marshall's per-mode values by up to several-fold, while summing over the complete six-mode set reproduces their published TOTAL inside both uncertainty bands (an open empirical observation about this total-level agreement, not a proven identity; see the G11 gate record's orthogonality-identity discussion, which is qualitatively consistent with but does not by itself certify the observed precision). WP31 (CONVENTIONS.md section 16's participation-factor extension, `cliffordclock.integrator.omega.two_ion_participations`) now consumes the two-ion partition; see this report's participation-corrected variant case for the per-mode and total-level result that closed-form treatment gives (axial modes match well; radial modes do not, a disclosed, different scope boundary of THAT closed form, not this one). What remains open after WP31 is N>2-ion crystals (a numeric normal-mode eigensolver, no closed form in general) and the RF/micromotion dynamics package (unrelated to participation).**

| Quantity | Value |
|---|---|
| Species | Al27+ |
| Number of modes | 6 |
| <v^2> (m^2/s^2) | 2.068702e+00 |
| Predicted (P-1)_motional | -1.150871e-17 |
| Predicted uncertainty (1-sigma) | +/-2.713e-19 |
| Predicted band | [-1.178001e-17, -1.123741e-17] |
| Published (Marshall Table I "Secular motion") | -1.146000e-17 |
| Published band | [-1.184000e-17, -1.108000e-17] |
| Residual (predicted - published) | -4.871e-20 |
| Bands overlap | True |
| **kpi_verdict** | **MET** |

This is NOT counted toward `benchmarks/results/wp10_results.json`'s `kpi_summary` (reproducibility/blind-prediction/not-applicable) totals; it is a structurally distinct, weaker class (`case_class = "arithmetic_reproduction"`), tracked in this separate report. See this script's module docstring for the full SOURCES provenance, the Brewer 2019 alternative-source discussion, and the two-mass normal-mode scope caveat.

## WP31: participation-corrected variant (two_ion_participations, Al27+/Mg25+)

**Participation-corrected per-mode comparison against Marshall et al.'s own published 'Frequency shift per quantum' row (Table S2): the two AXIAL modes, where two_ion_participations' closed form is exact (a function of the Al+/Mg25+ mass ratio alone), match the published per-mode values to a few percent, a substantial improvement over the single-mass (participation=1.0) variant's ~2x per-mode disagreement there. The four RADIAL modes do NOT match well: the true radial eigenvector additionally depends on trap RF/DC geometry parameters (two_ion_participations' own documented scope caveat) this closed form cannot supply from masses alone. Because the radial STR pair carries the largest published per-mode magnitudes, the participation-corrected TOTAL does not reproduce Marshall's published band as closely as the single-mass total does; both totals are reported with their own kpi_verdict, not merged into one number.**

| Mode | Axial? | Participation | Predicted shift/quantum | Published shift/quantum | Ratio (pred/pub) |
|---|---|---|---|---|---|
| axial_com | True | 0.5383 | -9.5667e-20 | -9.5000e-20 | +1.0070 |
| axial_str | True | 0.4617 | -1.4244e-19 | -1.4200e-19 | +1.0031 |
| x_com | False | 0.5383 | -1.8690e-19 | -1.7700e-19 | +1.0560 |
| x_str | False | 0.4617 | -1.3219e-19 | -6.4800e-19 | +0.2040 |
| y_com | False | 0.5383 | -2.3784e-19 | -1.4200e-19 | +1.6749 |
| y_str | False | 0.4617 | -1.8043e-19 | -6.5300e-19 | +0.2763 |

Axial mean |ratio-1| deviation: 0.0051. Radial mean |ratio-1| deviation: 0.5626.

| Quantity | Value |
|---|---|
| Participation-corrected total (P-1)_motional | -5.759425e-18 |
| Participation-corrected uncertainty (1-sigma) | +/-1.370e-19 |
| Participation-corrected band | [-5.896383e-18, -5.622467e-18] |
| Total bands overlap | False |
| **total_kpi_verdict** | **NOT MET** |

Per-mode published-value citation: Marshall et al., arXiv:2504.13071v2 (2025), Supplemental Material Table S2, "Frequency shift per quantum (10^-19)" row: -0.95, -1.42, -1.77, -6.48, -1.42, -6.53 for Axial COM, Axial STR, X COM, X STR, Y COM, Y STR respectively (same mode order as MARSHALL_AL_ION_MODES_MHZ_NBAR). This is the paper's own PUBLISHED per-mode time-dilation weight (distinct from the unrelated 'Geometric factor kappa' row, a Doppler-cooling-laser geometry factor, MARSHALL_AL_ION_MODES_CITATION's own caveat). Re-fetched and confirmed against the arXiv PDF text this session.
