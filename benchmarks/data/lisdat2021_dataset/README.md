# Lisdat et al. 2021 blackbody-radiation dataset

Source: Christian Lisdat, Sören Dörscher, Ingo Nosske, Uwe Sterr,
"Additional data for the publication 'The blackbody radiation shift in
strontium lattice clocks revisited'," Physikalisch-Technische
Bundesanstalt (PTB) Open Access Repository, DOI
[10.7795/720.20210928](https://doi.org/10.7795/720.20210928).

License: **CC BY-ND 4.0** (Creative Commons Attribution-NoDerivatives
4.0 International). The four data files below and `info.txt` are
redistributed here **unmodified**, exactly as downloaded from the PTB
repository, which this license permits; no derivative or edited version
of any file is committed anywhere in this repository. Any reuse of
these files outside this project must preserve the same attribution and
must not distribute a modified copy, per the license terms.

## Files

- `info.txt`: the dataset's own manifest, unmodified, describing every
  file in the original archive (including files not copied here).
- `BBR_shift.dat`: the temperature-dependent blackbody-radiation shift
  for the strontium clock transition. Columns: temperature (K), total
  BBR shift (Hz), the dynamic contribution to the BBR shift (Hz), the
  fractional uncertainty of the static contribution (in units of
  $10^{-19}$), the fractional uncertainty of the dynamic contribution
  (in units of $10^{-19}$), and the residuals of the dynamic-term
  parametrization (Eqs. 6 and 7 of the source manuscript, in units of
  $10^{-19}$).
- `G(3D1_T).dat`: the exact Planck-integral function $G$ (the source
  manuscript's Eq. 3) for strontium's dominant BBR-coupled transition,
  $5s4d\,^3D_1$-$5s5p\,^3P_0$ at 115.1662621 THz, tabulated as a
  function of temperature. Columns: temperature (K), $G$.
- `Approximation_G(n)_Sr.dat`, `Approximation_G(n)_Yb.dat`: the data
  behind Fig. 1 of the source manuscript, showing how a truncated
  power-series approximation to $G$ (the manuscript's Eq. 4) converges,
  or fails to converge, with the truncation order $n$, for strontium and
  ytterbium respectively. Columns: truncation order $n$, the
  approximate solution $G_n(y)$ at $y=18.4$ (Sr) or $y=34.5$ (Yb),
  corresponding to $T=300$~K, and the fractional deviation of $G_n(y)$
  from the full integral.

This project's own registry (`cliffordclock.ensemble.species.SR87`) uses
a PTB-2025 rescaling of this dataset's fit *shape*, applied onto
the more precise Aeppli et al. 2024 anchor (see
Lisdat et al., PRR 3, L042036 (2021); Aeppli et al., PRL 133, 023401
(2024), and
`paper/main.tex` Sec. "The blackbody-radiation pivot term"); this
dataset's own coefficients are not used directly. This
dataset's own tabulated `BBR_shift.dat` values are used by
`paper/figures/fig5_bbr_temperature.py` purely as a published-data
overlay for comparison, not as an input to the registry.
