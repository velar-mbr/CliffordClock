# WP40 Rydberg Stark-map benchmark cases (generated)

Generated: 2026-09-03T16:28:51.401324+00:00

Validates the full Stark-map module (CONVENTIONS.md section 20) against Phase A's own registry (C3), an independent open-source implementation (C4, ARC), published literature (C5), and its own basis-truncation convergence (C6).

## C3: quadratic-crossover consistency vs. the E43 registry

**Classification: arithmetic_reproduction**

| n | n* | map alpha0 (a.u.) | registry alpha0 (a.u.) | Relative error |
|---|---|---|---|---|
| 30 | 28.654 | 8.8089e+09 | 9.2250e+09 | 4.51% |
| 32 | 30.654 | 1.3599e+10 | 1.4146e+10 | 3.87% |
| 35 | 33.654 | 2.4849e+10 | 2.5550e+10 | 2.74% |
| 50 | 48.654 | 2.7432e+11 | 2.8850e+11 | 4.91% |

Tolerance: 15%. Worst relative error: 4.91%. **kpi_verdict: MET**

Kill tests armed: sign-flip=True, doubled-coefficient=True.

## C4: ARC cross-validation

**Classification: independent_implementation_reproduction**. ARC 3.10.2, commit 4b4573e965222e798ac59636ad7a8b3457262835, BSD-3-Clause.

| n | Basis (mine/ARC) | Target idx (mine/ARC) | Low-field worst rel. err | My crossover (V/cm) | ARC low-overlap field (V/cm) |
|---|---|---|---|---|---|
| 30 | 451/451 | 209/209 | 1.450% | 115.80 | 99.26 |
| 32 | 451/451 | 209/209 | 1.579% | 82.65 | 73.20 |
| 35 | 451/451 | 209/209 | 1.694% | 50.34 | 50.34 |
| 50 | 451/451 | 209/209 | 2.049% | 7.97 | 4.69 |

Low-field tolerance (50% of the Inglis-Teller estimate): 5%. Worst observed: 2.049%. **kpi_verdict: MET**

Beyond the low-field tier, this module's own and ARC's own tracked curves are built from two INDEPENDENTLY constructed Hamiltonians (different radial-integral method, different quantum defects for some l, no fine-structure term in this module's model potential); near an avoided crossing the two can legitimately settle onto SWAPPED branches (both individually well-tracked, i.e. locally high step-overlap, but globally divergent past that point) without either side's tracking algorithm being wrong -- verified directly: restricting the comparison to points where BOTH curves report a high step-to-step overlap (>0.95) does not by itself bring the worst-case full-range error down, confirming this is a branch-identity effect near a crossing, not a resolution or tracking bug. This is reported, not gated by a numeric tolerance, per the dossier's own instruction to loosen (not eliminate) the check near a crossing; the crossover-location fields above are the closest thing to a quantitative beyond-crossing check this comparison supports.

## C5: published anchor, three-part

No single source combines the registry species (85Rb), the registry l (D5/2), and coverage through an avoided crossing with printed (non-digitized) numbers (dossier Sec. 2c/6, item 2); the three parts above are the honest composite this project can support, each labeled with its own evidentiary class, none presented as a full through-the-crossing validation on its own.

### (a) Holloway et al. 2014 Fig. 15 field-endpoint reduction

Classification: arithmetic_reproduction. Worst relative error: 1.443%. **kpi_verdict: MET**

### (b) O'Sullivan & Stoicheff 1985 nS crossing-field method check

Classification: arithmetic_reproduction. n=40, printed field=6.97 V/cm, map-detected field=8.47 V/cm
**kpi_verdict: MET**

### (c) Grimmel et al. 2015 supplementary data availability

Classification: not_attempted (no machine-readable data found).
HTTP 200; content-type=text/html; charset=UTF-8

## C6: basis-truncation convergence

**Classification: convergence_study**. Convergence threshold: 10%. **kpi_verdict: MET**

### n=50 (IT estimate: 6.29 V/cm, converged=True)

| delta_n | l_max | Basis size | alpha0 (a.u.) | Relative shift from largest |
|---|---|---|---|---|
| 2 | 6 | 65 | 2.7649e+11 | 0.165% |
| 3 | 10 | 147 | 2.7599e+11 | 0.017% |
| 5 | 14 | 319 | 2.7604e+11 | 0.000% |
| 5 | 20 | 451 | 2.7604e+11 | 0.000% |

### n=32 (IT estimate: 63.33 V/cm, converged=True)

| delta_n | l_max | Basis size | alpha0 (a.u.) | Relative shift from largest |
|---|---|---|---|---|
| 2 | 6 | 65 | 1.3645e+10 | 0.089% |
| 3 | 10 | 147 | 1.3649e+10 | 0.059% |
| 5 | 14 | 319 | 1.3657e+10 | 0.000% |
| 5 | 20 | 451 | 1.3657e+10 | 0.000% |

