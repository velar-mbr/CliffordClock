# WP20 BBR benchmark case (generated)

Generated: 2026-08-12T13:57:40.177739+00:00

## Arithmetic-reproduction case: JILA arXiv:2403.10664 Table I 'BBR' row

**Classification label (binding, G7 sign-off B5): arithmetic reproduction of a published standard-formula evaluation (arithmetic-reproduction fidelity, NOT BBR accuracy; explicitly weaker than a reproducibility case; JILA's own BBR row is itself computed, not an independently measured shift)**

| Quantity | Value |
|---|---|
| Species | Sr87 |
| Temperature T | 293.282 +/- 0.004 K |
| Predicted (P-1)_BBR (nominal T) | -4.841743e-15 |
| Predicted, temperature band (T +/- 0.004 K) | [-4.842017e-15, -4.841468e-15] |
| Predicted, coefficient-uncertainty band | +/-6.832e-19 |
| Predicted, combined (coefficient+T) band | [-4.842479e-15, -4.841006e-15] |
| Published (JILA Table I 'BBR') | -4.841720e-15 |
| Published band | [-4.842450e-15, -4.840990e-15] |
| Residual (predicted - published) | -2.251e-20 |
| Bands overlap | True |
| **kpi_verdict** | **MET** |

This is NOT counted toward `benchmarks/results/wp10_results.json`'s `kpi_summary` (reproducibility/blind-prediction/not-applicable) totals; it is a structurally distinct, weaker class (`case_class = "arithmetic_reproduction"`), tracked in this separate report. See `benchmarks/RESULTS.md` for the full write-up and why agreement here is expected almost by construction.
