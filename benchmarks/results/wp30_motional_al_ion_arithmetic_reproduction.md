# WP30 motional Al+ ion benchmark case (generated)

Generated: 2026-08-22T20:51:52.194155+00:00

## Arithmetic-reproduction case: Marshall et al. arXiv:2504.13071 Table I "Secular motion" row

**Classification label (binding): arithmetic reproduction of a published standard-formula evaluation (arithmetic-reproduction fidelity, validates the engine's E38 implementation and unit chain, NOT an independent motional-Doppler physics prediction; Marshall et al.'s own secular-motion row is itself computed from their own measured inputs through the same standard formula)**

**SCOPE CAVEAT (corrected per the project's G11 gate record, section A3): Marshall et al.'s six modes are TWO-ION (27Al+/25Mg+) crystal normal modes. The physically complete per-mode evaluation partitions each mode's zero-point and thermal motion between the two ions by their own normal-mode amplitudes, a quantity this project's E38 formula does not consume (one species/mass for every mode, CONVENTIONS.md section 16): a documented scope boundary, not an oversight. As a result the engine's per-mode contributions differ from Marshall's own per-mode values by up to several-fold, while summing over the complete six-mode set reproduces their published TOTAL inside both uncertainty bands. The G11 gate record derives a genuine orthogonality identity over the two-ion normal-mode basis that is qualitatively consistent with this total-level agreement despite the per-mode differences, but that record also shows the identity alone does not certify the observed precision: the mechanism is reported as an open empirical observation, not a proven identity. The open item is a full two-mass normal-mode treatment (per-ion amplitude vectors as explicit input), belonging to the same future package as the RF/micromotion dynamics treatment already flagged out of scope for this tier.**

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
