# Effect one-pagers

CliffordClock's systematics budget is built from separate physical
effects, each computed by its own small piece of code. Each page below
names one effect, states its governing formula, shows the function that
evaluates it, and gives the validation number that checks it against a
closed form, a published value, or an independent implementation. A
clock physicist can read any one page in under two minutes and check it
against their own version of the same effect.

| Effect | E-label(s) | What it computes | Typical magnitude |
|---|---|---|---|
| [DC-Stark shift](dc-stark-shift.md) | E14a/E14b | Stray-field quadratic Stark shift | `-7.17e-17` (Sr-87, 100 V/m) |
| [Second-order Doppler shift (thermal ensemble)](second-order-doppler.md) | E21 | Relativistic time dilation from classical thermal motion | `-8.0e-21` (Sr-87, 5 μK) |
| [Blackbody-radiation shift](blackbody-radiation.md) | E32/E33/E37 | Thermal-radiation-field shift, single or multi-surface | `-5.32e-15` (Sr-87, 300 K) |
| [Ion electric-quadrupole shift](quadrupole-shift.md) | E34/E35 | Field-gradient coupling to a D/F state's quadrupole moment | `2.97 Hz·mm²/V` slope (Ca⁺:D5/2) |
| [Gravitational-redshift pivot term](gravitational-redshift.md) | E36 | General-relativistic time dilation across a mm-scale sample | `1.09e-16` per metre (`g/c²`) |
| [Trapped-ion motional time dilation](motional-time-dilation.md) | E38 (WP31-35) | Quantum motional-state second-order Doppler shift | `-1.14e-17` total (Al⁺/Mg⁺ two-ion) |
| [Ramsey fringe visibility and coherence](coherence-visibility.md) | E39 | Dephasing loss from a spread of accumulated phase | `V = exp(-σ_Φ²/2)` (Gaussian closure) |
| [Lattice light shift](lattice-light-shift.md) | E40/E41 | Trapping-light-induced differential shift, two community models | crosses zero near `u_op ≈ 72 E_R` (Sr-87) |
| [Sideband-spectrum forward model and fitting](sideband-spectrum.md) | E42 | Differentiable carrier-plus-sidebands lineshape, fit for trap depth and radial temperature | `12/12` synthetic fits converge |
| [Rydberg vapor-cell response](rydberg-cell-response.md) | E43/E44 | Per-atom quadratic Stark shift and the resulting EIT/Autler-Townes spectrum | `48.31 MHz` splitting at `9.83 V/m` (Rb-85 32D5/2-33P3/2, 68.64 GHz) |
| [Full Rydberg Stark maps](rydberg-stark-map.md) | E43-extension | Exact-diagonalization Stark shift beyond the quadratic regime, cross-validated against ARC | `2.05%` worst low-field error vs. ARC (Rb-85 50D5/2) |
| [Rydberg field reconstruction](rydberg-field-reconstruction.md) | E45 | Differentiable JAX port of the quadratic-Stark/EIT chain, fit for a cell's own field distribution | `6/8` synthetic fits recover all three field parameters within 2-sigma |

Every formula and validation number on these pages is copied from
`docs/CONVENTIONS.md`, `docs/validation.md`, or `benchmarks/RESULTS.md`;
see those files for the full derivations, sign-off records, and
per-species coefficient registries.
