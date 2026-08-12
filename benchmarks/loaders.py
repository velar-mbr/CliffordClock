# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed loaders for this project's authorized public-dataset sources.

``benchmarks/`` is not part of the installed package (excluded from the
wheel by ``pyproject.toml``'s ``[tool.setuptools.packages.find] where =
["src"]`` -- only ``src/`` is packaged; nothing here needs to change that).
This module parses the two datasets the project owner authorized fetching
from (2026-08-10, see ``benchmarks/SOURCES.md``) into typed structures, and
carries no physics of its own -- see ``cliffordclock.pipeline``/
``cliffordclock.ensemble.species`` for that.

- **arXiv:2403.10664** (Aeppli, Kim, Warfield, Safronova, Ye, "A clock with
  8x10^-19 systematic uncertainty", 2024): a JILA 1D Sr-87 optical lattice
  clock systematic-uncertainty paper. It has **no machine-readable
  ancillary data files** (confirmed: arXiv hosts only the PDF and a TeX
  source bundle for this submission, no "Ancillary files" section) -- so
  the only ingestible content is Table I's published numbers, hand-
  transcribed into ``benchmarks/fixtures/jila_2403_10664_table1.csv``
  (see that file and ``benchmarks/MAPPING.md`` for the exact citation of
  every row; this is normal and expected for this kind of benchmark).
- **data.nist.gov DOI 10.18434/M32206** ("Data for 'Coherent Optical
  Clock Down-Conversion for Microwave Frequencies with 10-18
  Instability'", Nakamura et al., 2020): two machine-readable CSV time
  series (Yb-clock optical phase and 10 GHz microwave phase vs. time,
  ~44002 samples each). This *is* genuinely machine-readable data, parsed
  here by :func:`load_nist_phase_csv` -- but (see ``benchmarks/MAPPING.md``)
  it is a phase/Allan-deviation instability record for an optical-to-
  microwave frequency-division scheme, not a systematic-shift/field-
  gradient measurement, so it does not map to any pipeline config; the
  parser exists to demonstrate genuine ingestion and is exercised by
  ``tests/test_benchmarks_loaders.py``, not because ``run_benchmarks.py``
  produces a residual from it.

Two sources authorized as a follow-up (2026-08-10, see
``benchmarks/SOURCES.md`` sections 4-5):

- **arXiv:1706.01944** (Bowden, Hobson, Huillery, Gill, Jones, Hill,
  "Rydberg Electrometry for Optical Lattice Clocks", Phys. Rev. A 96,
  023419 (2017), NPL): measures a *residual stray field* at their Sr
  atoms **independently of the clock transition** (Rydberg-state EIT
  spectroscopy) -- ``E = 1.52 V/m`` with asymmetric statistical and
  systematic uncertainties -- then converts it to a clock-transition
  DC-Stark shift using the same Middelmann et al. (PRL 109, 263004
  (2012)) differential polarizability this project's `Sr87` species
  registry entry already uses. This is the one source, of everything
  examined for WP10, that publishes an independent field magnitude for
  the DC-Stark systematic -- see `NPL_RESIDUAL_FIELD_V_PER_M` and
  `NPL_PUBLISHED_SHIFT` below, and `benchmarks/MAPPING.md`/
  `benchmarks/RESULTS.md` for why this is a *reproducibility* case, not
  a blind prediction.
- **Metrologia 63, 025002 (2026)** (Jia et al., USTC Sr1 clock,
  "Improved systematic evaluation of a strontium optical clock with
  uncertainty below 1e-18", CC BY 4.0, owner-provided PDF): publishes a
  DC-Stark systematic *budget constraint* (Table 3, section 3.5) but,
  like the JILA arXiv:2403.10664 row, no independent field magnitude --
  see `USTC_DC_STARK_CONSTRAINT` below and `benchmarks/MAPPING.md`.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SystematicShiftEntry:
    """One row of a published clock systematic-uncertainty budget table.

    Parameters
    ----------
    shift_name : str
        The effect's name, verbatim from the source table's first column.
    shift_e19 : float
        Correction/shift value, in units of ``1e-19`` fractional frequency
        (the source table's own units -- kept un-rescaled here so every
        stored number matches the publication digit-for-digit; callers
        multiply by ``1e-19`` to get a bare fractional-frequency value).
    uncertainty_e19 : float
        1-sigma uncertainty (or, if `uncertainty_is_upper_bound`, an upper
        bound), same units as `shift_e19`.
    uncertainty_is_upper_bound : bool
        True if the source table gives this uncertainty as an upper bound
        (e.g. published as ``"<0.1"``) rather than a symmetric 1-sigma value.
    in_engine_scope : bool
        True if this effect's physics is, in principle, within this
        engine's current scope (CONVENTIONS.md E14b scalar DC Stark + E21
        second-order Doppler only, per `docs/validation.md`/`docs/CONVENTIONS.md`).
        Being in scope does *not* imply a comparison is actually possible
        -- see `scope_note` and `benchmarks/MAPPING.md`.
    scope_note : str
        A one-line citation/rationale for the `in_engine_scope` classification.
    """

    shift_name: str
    shift_e19: float
    uncertainty_e19: float
    uncertainty_is_upper_bound: bool
    in_engine_scope: bool
    scope_note: str

    @property
    def shift_fractional(self) -> float:
        """`shift_e19` rescaled to a bare fractional-frequency value."""
        return self.shift_e19 * 1e-19

    @property
    def uncertainty_fractional(self) -> float:
        """`uncertainty_e19` rescaled to a bare fractional-frequency value."""
        return self.uncertainty_e19 * 1e-19


def load_jila_table1(path: Path) -> list[SystematicShiftEntry]:
    """Parse the JILA arXiv:2403.10664 Table I fixture CSV.

    Parameters
    ----------
    path : Path
        Path to a CSV with columns ``shift_name, shift_e19, uncertainty_e19,
        uncertainty_is_upper_bound, in_engine_scope, scope_note`` (see
        ``benchmarks/fixtures/jila_2403_10664_table1.csv``, the shipped
        full-table transcription).

    Returns
    -------
    list[SystematicShiftEntry]
        One entry per data row, in file order.

    Raises
    ------
    ValueError
        If a row is missing a required column or a numeric/boolean field
        does not parse.
    """
    entries: list[SystematicShiftEntry] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {
            "shift_name",
            "shift_e19",
            "uncertainty_e19",
            "uncertainty_is_upper_bound",
            "in_engine_scope",
            "scope_note",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"{path}: missing required column(s) {sorted(missing)}")
        for row_num, row in enumerate(reader, start=2):
            try:
                entries.append(
                    SystematicShiftEntry(
                        shift_name=row["shift_name"],
                        shift_e19=float(row["shift_e19"]),
                        uncertainty_e19=float(row["uncertainty_e19"]),
                        uncertainty_is_upper_bound=_parse_bool(row["uncertainty_is_upper_bound"]),
                        in_engine_scope=_parse_bool(row["in_engine_scope"]),
                        scope_note=row["scope_note"],
                    )
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{row_num}: malformed row {row!r}: {exc}") from exc
    if not entries:
        raise ValueError(f"{path}: no data rows found")
    return entries


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise ValueError(f"not a recognized boolean: {value!r}")


#: The JILA paper's *precise* main-text DC-Stark value, quoted from prose
#: (not Table I, which rounds to one decimal place in units of 1e-19).
#: Verbatim source (arXiv:2403.10664v2, unlabeled "DC Stark Shift" section,
#: page 4-5, the paragraph beginning "Stray electric fields can shift the
#: clock transition frequency"): "The total residual DC Stark shift is
#: -9.8 +/- 0.7 x 10^-20." No independent stray-field magnitude (V/m) is
#: given anywhere in the arXiv preprint, nor (per a specific cross-check,
#: not merely an inaccessibility assumption) in the published version's
#: Supplemental Material -- see `benchmarks/SOURCES.md` section 3 and
#: `benchmarks/MAPPING.md` for why this means the value below cannot be
#: turned into a predicted-vs-published residual case (WP10 labeling
#: discipline: no field to feed the engine, and solving for one from this very
#: number would be exactly the "tuned parameter" the labeling discipline
#: forbids).
JILA_DC_STARK_PRECISE = SystematicShiftEntry(
    shift_name="DC Stark (main text, precise value)",
    shift_e19=-0.98,
    uncertainty_e19=0.07,
    uncertainty_is_upper_bound=False,
    in_engine_scope=True,
    scope_note=(
        "CONVENTIONS.md E14b scalar DC-Stark shift -- in scope, but not an "
        "independent forward-comparable case (see benchmarks/MAPPING.md)."
    ),
)


@dataclass(frozen=True)
class NistPhaseSeries:
    """A parsed NIST DOI 10.18434/M32206 phase-vs-time CSV.

    Parameters
    ----------
    time_s : np.ndarray
        Sample index / time column, shape ``(N,)``, as published (the
        source files' first column is a plain integer sample count, not
        an explicitly labeled unit -- see `benchmarks/SOURCES.md`).
    phase : np.ndarray
        Phase column, shape ``(N,)``, units per `phase_units`.
    phase_units : str
        Unit of the `phase` column as given in the source file's own
        title (``"rad"`` for the Yb-clock file, ``"mrad"`` for the 10 GHz
        microwave file).
    source_file : str
        The originating filename (for provenance in downstream reports).
    """

    time_s: np.ndarray
    phase: np.ndarray
    phase_units: str
    source_file: str


def load_nist_phase_csv(path: Path, phase_units: str) -> NistPhaseSeries:
    """Parse a NIST M32206 whitespace-delimited ``time phase`` CSV.

    Parameters
    ----------
    path : Path
        Path to a NIST M32206 data file (or a fixture excerpt of one --
        the format is identical: two whitespace-separated float columns,
        no header row).
    phase_units : str
        Unit label to attach to the parsed phase column (``"rad"`` or
        ``"mrad"`` for the two published files -- not itself present in
        the file, which carries no header/unit row).

    Returns
    -------
    NistPhaseSeries
        The parsed two-column series.

    Raises
    ------
    ValueError
        If any row does not have exactly two whitespace-separated numeric
        fields.
    """
    times: list[float] = []
    phases: list[float] = []
    with path.open(encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_num}: expected 2 whitespace-separated fields, "
                    f"got {len(fields)}: {line!r}"
                )
            try:
                t, p = float(fields[0]), float(fields[1])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_num}: non-numeric field: {line!r}") from exc
            times.append(t)
            phases.append(p)
    return NistPhaseSeries(
        time_s=np.asarray(times, dtype=np.float64),
        phase=np.asarray(phases, dtype=np.float64),
        phase_units=phase_units,
        source_file=path.name,
    )


@dataclass(frozen=True)
class AsymmetricMeasurement:
    """A published quantity with independent, asymmetric statistical and
    systematic uncertainties (e.g. NPL's residual-field measurement).

    Deliberately keeps ``stat``/``sys`` and ``lo``/``hi`` separate rather
    than collapsing them into a single symmetric sigma -- WP10's follow-up
    instruction is explicit: "no Gaussian pretence on asymmetric errors."
    The one combination this class *does* perform (`combined_lo`/
    `combined_hi`) is the standard, non-Gaussian-assuming one: independent
    statistical and systematic contributions on the same side are added in
    quadrature (`sqrt(stat**2 + sys**2)`), *keeping the low and high sides
    separate* -- it never symmetrizes the two sides against each other.

    Parameters
    ----------
    nominal : float
        The central published value.
    stat_lo, stat_hi : float
        Statistical uncertainty, low/high side (both stored as positive
        magnitudes; `nominal - stat_lo` / `nominal + stat_hi` are the
        statistical-only bounds).
    sys_lo, sys_hi : float
        Systematic uncertainty, low/high side, same convention.
    units : str
        Unit of `nominal` (e.g. ``"V/m"``).
    citation : str
        Exact source location (paper, section, page, verbatim quote).
    """

    nominal: float
    stat_lo: float
    stat_hi: float
    sys_lo: float
    sys_hi: float
    units: str
    citation: str

    @property
    def combined_lo(self) -> float:
        """Low-side bound: ``nominal - sqrt(stat_lo**2 + sys_lo**2)``."""
        return self.nominal - math.sqrt(self.stat_lo**2 + self.sys_lo**2)

    @property
    def combined_hi(self) -> float:
        """High-side bound: ``nominal + sqrt(stat_hi**2 + sys_hi**2)``."""
        return self.nominal + math.sqrt(self.stat_hi**2 + self.sys_hi**2)


@dataclass(frozen=True)
class PublishedBand:
    """A published central value with an asymmetric confidence band,
    already expressed as absolute low/high bounds (not stat+sys
    components) -- e.g. a paper's own quoted "X (+a/-b)" result.

    Parameters
    ----------
    nominal : float
        The central published value.
    lo : float
        The lower (more negative, for a negative shift) bound.
    hi : float
        The upper (less negative) bound.
    units : str
        Unit of `nominal`/`lo`/`hi`.
    citation : str
        Exact source location (paper, section, page, verbatim quote).
    """

    nominal: float
    lo: float
    hi: float
    units: str
    citation: str

    def __post_init__(self) -> None:
        if not self.lo <= self.nominal <= self.hi:
            raise ValueError(
                f"PublishedBand: nominal {self.nominal!r} not within [{self.lo!r}, {self.hi!r}]"
            )


#: NPL's Rydberg-EIT-measured residual stray field at their Sr atoms,
#: **independent of the clock transition itself** (arXiv:1706.01944,
#: Bowden et al., PRA 96, 023419 (2017), Section IV "Rydberg electrometry
#: using EIT", the paragraph beginning "Next the external field was
#: switched off..."). Verbatim (own extraction from the fetched arXiv v1
#: PDF text, cross-checked against the layout-mangled pdftotext output --
#: the paragraph reads, un-mangling the column wrap): "A fit to the
#: resulting splitting revealed a residual electric field of 1.52
#: (+0.62(stat)/-0.22(stat)) (+0.05(sys)/-0.03(sys)) V/m most likely due
#: to patch potential on the surrounding chamber."
NPL_RESIDUAL_FIELD_V_PER_M = AsymmetricMeasurement(
    nominal=1.52,
    stat_lo=0.22,
    stat_hi=0.62,
    sys_lo=0.03,
    sys_hi=0.05,
    units="V/m",
    citation=(
        'Bowden, Hobson, Huillery, Gill, Jones, Hill, "Rydberg Electrometry '
        'for Optical Lattice Clocks", arXiv:1706.01944v1 (PRA 96, 023419 '
        '(2017)), Section IV, p.5: "residual electric field of '
        '1.52 +0.62(stat)/-0.22(stat) +0.05(sys)/-0.03(sys) Vm^-1"'
    ),
)

#: NPL's own quoted clock-transition DC-Stark shift, converted from
#: `NPL_RESIDUAL_FIELD_V_PER_M` using the Middelmann et al. (PRL 109,
#: 263004 (2012)) differential polarizability -- the same citation this
#: project's `cliffordclock.ensemble.species.SR87` registry entry uses
#: (confirmed: arXiv:1706.01944's only Delta-alpha-relevant reference,
#: [3], is exactly this Middelmann paper, cited in the introduction's
#: "570 V/m yields a DC Stark shift of 1 Hz [3]" sanity-check example --
#: see benchmarks/SOURCES.md section 4 and benchmarks/MAPPING.md for the
#: full citation-chain verification, including an independent pipeline
#: cross-check of that very 570 V/m -> 1 Hz example).
#: Verbatim (arXiv:1706.01944v1, Section IV, same page): "Translating
#: this electric field and corresponding uncertainty to the DC Stark
#: shift of the 1S0-3P0 clock transition results in a fractional
#: frequency shift of -1.6 (+0.4/-1.6) x 10^-20."
NPL_PUBLISHED_SHIFT = PublishedBand(
    nominal=-1.6e-20,
    lo=-3.2e-20,  # -1.6 - 1.6, x1e-20
    hi=-1.2e-20,  # -1.6 + 0.4, x1e-20
    units="fractional frequency",
    citation=(
        "Bowden et al., arXiv:1706.01944v1 (PRA 96, 023419 (2017)), Section "
        'IV, p.5: "fractional frequency shift of -1.6 +0.4/-1.6 x 10^-20"'
    ),
)

#: USTC Sr1 clock's DC-Stark systematic *budget constraint* (Jia et al.,
#: "Improved systematic evaluation of a strontium optical clock with
#: uncertainty below 1e-18", Metrologia 63, 025002 (2026), CC BY 4.0).
#: Same structural class as the JILA arXiv:2403.10664 Table I DC-Stark
#: row (`JILA_DC_STARK_PRECISE`): in engine scope (E14b), but no
#: independent field magnitude published -- see benchmarks/MAPPING.md.
#: Verbatim, Section 3.5 "Other minor systematic shifts", subsection
#: "Residual DC Stark shift" (printed page 9 of the PDF): "this shift
#: originates from static electric fields due to charge accumulation on
#: viewports [60]. As reported in our previous work [30], the
#: y-component of the field caused a shift of 1.4(5.2) x 10^-21. Given
#: Delta-nu_DC ~ E^2_stat and E_stat ~ r^-2 dependence, the shorter
#: y-axis viewport distance (142 mm vs 237 mm for x,z) yield an 8x larger
#: shift along y. Electrostatic shielding further suppresses x,z-axis
#: fields by a factor of 3 according to FE simulations. Combining these,
#: we constrain the total shift to 0.0(0.1) x 10^-19." Table 3 (printed
#: page 10) lists the same value as its "DC Stark" row: "0 <0.1" (units
#: 1e-19). Reference [30] is "Li J et al 2024 Metrologia 61 015006" --
#: the group's own *previous* evaluation, which per this quote actually
#: characterized an applied external field (the source of the "1.4(5.2)
#: x 10^-21" y-component prior) -- see benchmarks/MAPPING.md for why this
#: is flagged as the next authorization candidate, not fetched here.
USTC_DC_STARK_CONSTRAINT = SystematicShiftEntry(
    shift_name="DC Stark (USTC Sr1, Table 3 + Sec. 3.5)",
    shift_e19=0.0,
    uncertainty_e19=0.1,
    uncertainty_is_upper_bound=True,
    in_engine_scope=True,
    scope_note=(
        "CONVENTIONS.md E14b scalar DC-Stark shift -- in scope, but (like "
        "the JILA arXiv:2403.10664 DC Stark row) no independent field "
        "magnitude published in this paper; see benchmarks/MAPPING.md."
    ),
)

# ---------------------------------------------------------------------------
# WP20 addendum: JILA arXiv:2403.10664 "BBR" row + operating temperature.
#
# Same source as `JILA_DC_STARK_PRECISE`/`JILA_DC_STARK_CONSTRAINT` above
# (arXiv:2403.10664v2, already fetched and checksummed for WP10 -- see
# `benchmarks/SOURCES.md` section 1; not re-fetched here, per the binding
# instruction to extend that logged source rather than duplicate it). These
# two values feed `benchmarks/run_bbr_jila_arithmetic_reproduction.py`'s
# **arithmetic-reproduction** case (WP20 design item 5, the project's
# theory sign-off record (G7) B5) -- explicitly NOT a
# `"reproducibility"`-class case like `NPL_RESIDUAL_FIELD_V_PER_M` above:
# JILA's own BBR row is itself computed (their T through the standard BBR
# formula with their own coefficients), not an independent measurement of
# the shift.
# ---------------------------------------------------------------------------

#: JILA's in-vacuum RTD operating temperature (arXiv:2403.10664v2, main
#: text temperature-measurement statement, cross-checked against the
#: Supplemental Material's own "II. Temperature Measurement" section
#: title, `benchmarks/SOURCES.md` section 3): "T = 20.132(4) degC =
#: 293.282(4) K". Independently re-verified against the primary text in
#: this project's internal review sweep (2026-08-11) -- this session did
#: not re-fetch the PDF, per the instruction to extend
#: the already-logged arXiv:2403.10664 source (section 1 above) rather
#: than duplicate the fetch. `lo`/`hi` are the published symmetric 1-sigma
#: bounds (`nominal +/- 0.004 K`); consumed directly by
#: `bbr_pivot_perturbation`, the real engine function -- not hand
#: arithmetic.
JILA_BBR_TEMPERATURE_K = PublishedBand(
    nominal=293.282,
    lo=293.282 - 0.004,
    hi=293.282 + 0.004,
    units="K",
    citation=(
        "Aeppli, Kim, Warfield, Safronova, Ye, arXiv:2403.10664v2 (PRL 133, "
        "023401 (2024)), main-text in-vacuum RTD temperature statement: "
        '"T = 20.132(4) degC = 293.282(4) K"; cross-checked against the '
        "primary text in this project's internal review sweep (2026-08-11) "
        "-- see benchmarks/SOURCES.md section 1 for the "
        "already-logged arXiv:2403.10664 fetch/checksum this value extends."
    ),
)

#: JILA's published BBR row, Table I "BBR": `-48417.2(73) x 10^-19` =
#: `-4.84172(73) x 10^-15` fractional (same fixture value as
#: `benchmarks/fixtures/jila_2403_10664_table1.csv`'s "BBR" row, restated
#: here as a `PublishedBand` for the WP20 arithmetic-reproduction case's
#: direct overlap test -- see `benchmarks/run_bbr_jila_arithmetic_reproduction.py`).
JILA_BBR_PUBLISHED_SHIFT = PublishedBand(
    nominal=-48417.2e-19,
    lo=-48417.2e-19 - 7.3e-19,
    hi=-48417.2e-19 + 7.3e-19,
    units="fractional frequency",
    citation=(
        "Aeppli, Kim, Warfield, Safronova, Ye, arXiv:2403.10664v2 (PRL 133, "
        '023401 (2024)), Table I "BBR" row: -48417.2(73) x 10^-19 = '
        "-4.84172(73) x 10^-15 -- same source/checksum as "
        "benchmarks/fixtures/jila_2403_10664_table1.csv (benchmarks/SOURCES.md "
        "section 1)."
    ),
)

# ---------------------------------------------------------------------------
# Roos-benchmark addendum: Roos et al., quant-ph/0701215v1 (published Nature
# 443, 316 (2006)), Fig. 4a's measured two-ion quadrupole-shift slope and
# Fig. 4a's fitted offset.
#
# Source: owner-supplied primary text (local file, not a fresh fetch from
# this session -- see `benchmarks/SOURCES.md` section 7 for the provenance
# note; no checksum table is recorded there, per the binding instruction not
# to perform "fetch hash theater" for an owner-supplied file this session
# did not itself retrieve). Both numbers below are already independently
# extracted and cross-checked against the primary text (Roos et al.,
# Nature 443, 316 (2006), quant-ph/0701215v1, Eq. 1/Fig. 4a
# + p.9 text); not re-transcribed from the PDF a second time here,
# consistent with the "extend an already-logged source" pattern
# `JILA_BBR_TEMPERATURE_K`/`JILA_BBR_PUBLISHED_SHIFT` above use for WP20.
# Consumed by `benchmarks/run_roos_quadrupole_slope.py`'s cross-vintage/
# arithmetic-reproduction case (G8 sign-off B4 labeling).
# ---------------------------------------------------------------------------

#: Roos et al.'s Fig. 4a linear-fit slope of the two-ion entangled state
#: Psi_1's measured quadrupole-shift parity-oscillation frequency against
#: the mechanically-calibrated APPLIED axial gradient dE_z/dz (12-48
#: V/mm^2 range, beta=0), quant-ph/0701215v1 p.9: "a = 2.975(2)
#: Hz mm^2/V" (uncertainty stated as "<0.1%" of the fit slope; the
#: `PublishedBand` below uses the paper's own explicit "(2)" digit rather
#: than re-deriving one from the "<0.1%" prose, since "<0.1%" of 2.975 is
#: 0.003, slightly looser than the quoted "(2)" -- the tighter, explicitly
#: stated digit is used, per this project's general "quote the paper's own
#: number, do not re-round it" discipline).
ROOS_MEASURED_SLOPE_HZ_MM2_PER_V = PublishedBand(
    nominal=2.975,
    lo=2.975 - 0.002,
    hi=2.975 + 0.002,
    units="Hz*mm^2/V",
    citation=(
        "Roos, Chwalla, Kim, Riebe, Blatt, 'Designer atoms for quantum "
        "metrology,' quant-ph/0701215v1 (Nature 443, 316 (2006)), "
        'Fig. 4a + p.9: linear-fit slope "a = 2.975(2) Hz mm^2/V" of the '
        "measured two-ion quadrupole shift against the applied, "
        "omega_z-calibrated axial gradient dE_z/dz = 12-48 V/mm^2 at "
        "beta=0 -- see Roos et al., Nature 443, 316 (2006), "
        "quant-ph/0701215v1, Eq. 1/Fig. 4a."
    ),
)

#: Roos et al.'s Fig. 4a linear-fit y-intercept (the shift at zero applied
#: gradient), quant-ph/0701215v1 p.9: offset "Delta_0/(2*pi) = -2.4(1) Hz",
#: of which -2.9 Hz is second-order Zeeman at the 2.9 G bias field (a B-field
#: physics mechanism this engine does not model, CONVENTIONS.md has no
#: Zeeman term) and the remainder is attributed by the paper to a residual
#: stray quadrupole field -- NOT a quadrupole-vs-applied-gradient slope
#: quantity, so it does not enter the slope comparison
#: `benchmarks/run_roos_quadrupole_slope.py` performs (see that script's
#: module docstring for why).
ROOS_FIT_OFFSET_HZ = PublishedBand(
    nominal=-2.4,
    lo=-2.4 - 0.1,
    hi=-2.4 + 0.1,
    units="Hz",
    citation=(
        "Roos et al., quant-ph/0701215v1 (Nature 443, 316 (2006)), Fig. 4a "
        '+ p.9: fitted offset "Delta_0/(2*pi) = -2.4(1) Hz," of which '
        "-2.9 Hz is second-order Zeeman at the 2.9 G bias (the remainder "
        "attributed to a residual stray quadrupole field) -- see "
        "Roos et al., Nature 443, 316 (2006), quant-ph/0701215v1, "
        "Eq. 1/Fig. 4a."
    ),
)
