# WP22 Bothwell mm-scale redshift benchmark case (generated)

Generated: 2026-08-12T13:57:42.804755+00:00

## Reproducibility case: Bothwell et al., Nature 602, 420 (2022) / arXiv:2109.12238

**Classification label (binding, G9 sign-off B4): reproducibility, with the INVERTED-NPL caveat: the g/c^2 arithmetic is textbook and the authors computed it themselves trivially (unlike NPL's differential-polarizability reconstruction); what this case validates is the extended-sample MACHINERY (per-site geometry, Gaussian-envelope weighting, map assembly) producing the right measured-map slope end-to-end against a published measured map, with zero adjustable inputs; it does not change the blind-prediction count.**

| Quantity | Value |
|---|---|
| Sites (computational grid) | 5945 |
| Site spacing (INFERRED, lambda/2) | 4.0650e-07 m |
| Envelope sigma (INFERRED) | 4.0267e-04 m |
| Reference g (Bothwell surveyed, B1) | 9.796 m/s^2 |
| Predicted slope (engine's own height convention) | +1.089952e-16 /m |
| Predicted slope (Bothwell's coordinate convention, sign-mapped) | -1.0900e-19 /mm |
| Predicted slope (+/-1.5-sigma-windowed cross-check) | -1.0900e-19 /mm |
| Measured, method A (14-dataset campaign) | -9.80e-20 [-1.21e-19, -7.50e-20] /mm |
| Measured, method B (synchronous two-region) | -1.28e-19 [-1.55e-19, -1.01e-19] /mm |
| Sigma distance, method A | 0.48 sigma |
| Sigma distance, method B | 0.70 sigma |
| **kpi_verdict, method A** | **MET** |
| **kpi_verdict, method B** | **MET** |

The published corrected slopes already have per-pixel density and second-order-Zeeman corrections applied before the linear fit, plus budget-level corrections (their Table 1, units 1e-20/mm): BBR 0(0.3), lattice light -0.5(0.1), DC Stark +0.3(0.2), pixel calibration 0(0.8), so the corrected slope targets the ISOLATED gravitational gradient E36 predicts. This script's own field is configured to exactly zero (field.synthetic.kind='uniform', e0=[0,0,0]) for the same reason: no DC-Stark/BBR/quadrupole term is active, isolating E36 to match.

Bothwell's own DC-Stark gradient row, +0.3(0.2)e-20/mm (their Table 1 budget-level corrections), is in-scope physics for this engine (CONVENTIONS.md E14b) but is a separate systematic this case does not model (the comparison target already has it corrected out, see isolation_note); it enters this report as a narrative cross-reference only.

Not counted toward `benchmarks/results/wp10_results.json`'s `kpi_summary` totals; this is a separate script/report (mirrors `benchmarks/run_bbr_jila_arithmetic_reproduction.py`'s precedent), see the module docstring's headline-count note.
