# WP10 benchmark summary (generated)

Generated: 2026-08-12T13:57:39.308158+00:00

## Reproducibility case: NPL arXiv:1706.01944 (Sr87 DC Stark)

Zero-free-parameter reproducibility, NOT a blind prediction (see benchmarks/MAPPING.md): this engine's coupling.type=stark_dc pipeline, given NPL's published residual field and this project's Middelmann-sourced species-registry Delta_alpha (the same source NPL cites), against NPL's own published shift band.

| Quantity | Low | Nominal | High |
|---|---|---|---|
| Field (V/m) | 1.2980 | 1.5200 | 2.1420 |
| Predicted Δν/ν₀ | -3.2900e-20 | -1.6567e-20 | -1.2080e-20 |
| Published Δν/ν₀ (NPL) | -3.2000e-20 | -1.6000e-20 | -1.2000e-20 |

**Bands overlap: True; kpi_verdict: MET**

**WP16 rotor-path re-run (informational, not a second KPI row):** kpi_verdict=MET (same-verdict-as-scalar-case=True); integration.mode=worldline true Cl(1,3) rotor (`cliffordclock.pipeline._stark_rotor_ensemble`) instead of the E29 scalar fast path used above.

| Source | Effect / file | Published value | In scope | Comparable | KPI verdict |
|---|---|---|---|---|---|
| JILA 2403.10664 Table I | BBR | -4.842e-15 ± 7.3e-19 | False | False | N/A |
| JILA 2403.10664 Table I | Lattice Light | -1.000e-20 ± 3.2e-19 | False | False | N/A |
| JILA 2403.10664 Table I | Second Order Zeeman | -8.551e-17 ± 1.0e-19 | False | False | N/A |
| JILA 2403.10664 Table I | Density | -1.100e-19 ± 9.0e-20 | False | False | N/A |
| JILA 2403.10664 Table I | First order Zeeman | +0.000e+00 ± 7.0e-20 | False | False | N/A |
| JILA 2403.10664 Table I | Background Gas | -4.700e-19 ± 5.0e-20 | False | False | N/A |
| JILA 2403.10664 Table I | DC Stark | -1.000e-19 ± 1.0e-20 | True | False | N/A |
| JILA 2403.10664 Table I | Tunneling | +0.000e+00 ± 1.0e-20 | False | False | N/A |
| JILA 2403.10664 Table I | Minor Shifts | +0.000e+00 ± 1.0e-20 | False | False | N/A |
| JILA 2403.10664 Table I | Total Shift | -4.928e-15 ± 8.1e-19 | False | False | N/A |
| USTC Metrologia 63,025002 | DC Stark (USTC Sr1, Table 3 + Sec. 3.5) | +0.000e+00 ± 1.0e-20 | True | False | N/A |
| NIST M32206 | nist_m32206_yb_clock_phase_excerpt.csv (20/44002 samples, excerpt/full) | n/a (phase time series) | False | False | N/A |
| NIST M32206 | nist_m32206_10ghz_phase_excerpt.csv (20/44002 samples, excerpt/full) | n/a (phase time series) | False | False | N/A |

**KPI summary:** 1/1 reproducibility case(s) met, 0/0 blind-prediction case(s) met, 13 rows not-applicable (of 14 rows considered).

1 of 1 reproducibility case(s) met (NPL arXiv:1706.01944: this engine's coupling.type=stark_dc pipeline, given NPL's published residual field and PTB's published Delta_alpha, reconstructs NPL's own published DC-Stark shift band; NOT a blind prediction, since NPL combined the same two ingredients themselves; see benchmarks/MAPPING.md for why this label is binding). 0 blind-prediction cases (still none available from any authorized source). 13 rows remain not-applicable: JILA's Table I DC-Stark row and the USTC Metrologia 63,025002 DC-Stark constraint both publish only a resulting shift/bound with no independent field input; every other row (BBR, Zeeman, density, lattice light, background gas, etc., both papers) is physics outside this engine's scope; the NIST M32206 dataset measures a different physical quantity entirely. See benchmarks/RESULTS.md for the full gap analysis.

## Illustrative DC-Stark field-magnitude context (Sr87, not a benchmark case)

| Field (V/m) | Predicted Δν/ν₀ |
|---|---|
| 1.0 | -7.170524e-21 |
| 5.0 | -1.792631e-19 |
| 10.0 | -7.170524e-19 |
| 20.0 | -2.868210e-18 |
| 50.0 | -1.792631e-17 |
| 100.0 | -7.170524e-17 |

For reference (not a residual): JILA's actively-nulled published residual is -9.8e-20, which the table above shows sits between the 1 and 5 V/m rows, i.e. at a scale consistent with a few-V/m residual field, well below the ~19 V/m unshielded-patch-field example in `examples/realistic_lattice_sr87.yaml` (WP11).
