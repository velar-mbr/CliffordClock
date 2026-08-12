# Roos-slope ion benchmark case (generated)

Generated: 2026-08-12T14:04:18.087815+00:00

## Structural pin: two-ion enhancement

`hbar*Delta_1 / Delta_E_Q(-5/2)` = +4.800000 (must equal +24/5 = +4.8)

## Headline: cross-vintage comparison (independent theory Theta)

**Classification label (binding, G8 sign-off B4): cross-vintage comparison: Roos et al.'s measured Fig. 4a slope against an INDEPENDENT theory Theta (Itano, Phys. Rev. A 73, 022510 (2006)), not against Roos's own extracted Theta; weaker than a blind prediction (Roos's own applied gradient is itself trap-model calibrated, and their own fit produced the slope being predicted), but a genuine external comparison, distinct from arithmetic reproduction (G8 sign-off B4)**

| Quantity | Value |
|---|---|
| Theta (Itano theory) | 1.917 ea0^2 (no published uncertainty) |
| Predicted slope \|a_pred\| | 3.115229 Hz*mm^2/V |
| Measured slope (Roos Fig. 4a) | 2.975 +/- 0.002 Hz*mm^2/V |
| Residual (predicted - measured) | +0.140229 Hz*mm^2/V (+4.7136%) |
| Bands overlap | False |
| **kpi_verdict** | **NOT MET** (expected: recovers the known Theta theory-vs-measurement tension, not an engine defect) |

## Secondary: arithmetic reproduction (Roos's own Theta, circular)

**Classification label (binding, G8 sign-off B4): arithmetic reproduction: Roos et al.'s own extracted Theta, inverted from their own Fig. 4a fit via Theta = (5/12)*h*a, used to predict that SAME fit's slope; circular by construction, a closed-loop factor-consistency check of this engine's chain against Roos's own published conversion relation, never an independent validation (G8 sign-off B4)**

| Quantity | Value |
|---|---|
| Theta (Roos's own extraction) | 1.83 +/- 0.01 ea0^2 |
| Predicted slope \|a_pred\| | 2.973849 Hz*mm^2/V |
| Predicted slope band | [2.957599, 2.990100] Hz*mm^2/V |
| Measured slope (Roos Fig. 4a) | 2.975 +/- 0.002 Hz*mm^2/V |
| Residual (predicted - measured) | -0.001151 Hz*mm^2/V (-0.0387%) |
| Bands overlap | True |
| **kpi_verdict** | **MET** |
| Theta round-trip check ((5/12)*h*a_pred) | 1.830000 (input 1.830000; equality pinned to 1e-9 absolute in `tests/test_roos_benchmark.py`) |

## Offset (documentation only, not part of the slope comparison)

Delta_0/(2*pi) = -2.4 +/- 0.1 Hz at zero applied gradient; not combined with either slope prediction above (module docstring step 4: partly second-order Zeeman, a mechanism this engine does not model, partly an uncharacterized residual stray field).

## KPI-summary impact

Does not increment benchmarks/results/wp10_results.json's reproducibility_cases_total (stays 1) or blind_prediction_cases_total (stays 0); tracked separately, same pattern as benchmarks/run_bbr_jila_arithmetic_reproduction.py.

Source: Roos et al., Nature 443, 316 (2006), quant-ph/0701215v1, Eq. 1 + Fig. 4a + p.5-6/p.9 text; benchmarks/SOURCES.md section 7 (owner-supplied primary text, provenance note).
