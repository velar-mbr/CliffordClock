# SPDX-License-Identifier: AGPL-3.0-or-later
"""Metrology report assembly and machine-readable output (WP5 scope item 2).

Assembles a frozen `MetrologyReport` from ensemble phase statistics
(:mod:`cliffordclock.analytics.stats`, E23/E25/E27) plus run provenance
(package version, `docs/CONVENTIONS.md` version, an optional caller-supplied
config hash, species, ensemble metadata, timestamps), and writes it to
schema-versioned JSON and a companion line-profile CSV. See
``docs/report-schema.md`` for the field-by-field schema description.

No new physics is implemented in this module -- all formulas are in
`cliffordclock.analytics.stats`.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from cliffordclock.analytics.stats import (
    dephasing_time_t2star,
    mean_fractional_shift,
    shift_std_error,
)
from cliffordclock.ensemble.species import Species

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "CONVENTIONS_VERSION",
    "MetrologyReport",
    "build_report",
    "write_json",
    "write_line_profile_csv",
]

#: `MetrologyReport` JSON schema version (WP5 spec: `"report_schema": "1.0"`).
#: Bump on any `MetrologyReport` field/type/shape change -- independent of
#: both the package version and `CONVENTIONS_VERSION` below.
#: 1.1 (WP31, CONVENTIONS.md section 8 E39): added `ramsey_visibility`/
#: `ramsey_phase`, both `float | None` (see docs/report-schema.md).
REPORT_SCHEMA_VERSION = "1.1"

#: `docs/CONVENTIONS.md` version this module's formulas (via
#: `cliffordclock.analytics.stats`: E23, E25, E27) were transcribed
#: against. **Bump this constant by hand whenever
#: `docs/CONVENTIONS.md`'s own "Version:" header changes** (WP5
#: orchestrator instruction 8) -- it is not read from the file
#: automatically, so a report always states the equation set it was
#: actually computed against, even if the docs are later revised.
CONVENTIONS_VERSION = "1.1.0"


def _package_version() -> str:
    """`cliffordclock`'s installed package version (`importlib.metadata`).

    Mirrors `cliffordclock.__version__`'s fallback (recomputed locally
    rather than imported, to keep this module's provenance lookup
    self-contained and independent of the top-level package's import-time
    side effects).
    """
    try:
        return version("cliffordclock")
    except PackageNotFoundError:  # pragma: no cover - only hit if package not installed
        return "0.0.0+unknown"


@dataclass(frozen=True)
class MetrologyReport:
    """One metrology run's report (WP5 scope item 2).

    Field declaration order is the canonical JSON key order written by
    `write_json` (WP5 test contract item 5: "stable key order"); do not
    reorder fields without a `REPORT_SCHEMA_VERSION` bump.

    Attributes
    ----------
    report_schema : str
        Schema version, currently `REPORT_SCHEMA_VERSION`.
    conventions_version : str
        `docs/CONVENTIONS.md` version the formulas trace to,
        `CONVENTIONS_VERSION`.
    package_version : str
        `cliffordclock` package version (`importlib.metadata`).
    generated_at_utc : str
        ISO-8601 UTC timestamp of report generation.
        **[INTERPRETATION]**: the WP5 spec says "timestamps" (plural);
        a single generation timestamp is recorded because no second
        timestamp concept exists among WP5's inputs (simulation
        start/duration provenance belongs to the WP6 pipeline config,
        reachable via `config_hash`).
    config_hash : str or None
        Caller-supplied provenance hash of the run's input configuration
        (e.g. a hash of the `config.yaml` driving the pipeline). WP5 does
        not compute this itself -- computing it is WP6 pipeline scope;
        this field exists so WP6 can populate it (WP5 orchestrator
        instruction 8). ``None`` when not supplied.
    species_name : str
        Atomic species registry name (`cliffordclock.ensemble.species.Species.name`).
    ensemble_type : str
        Free-text ensemble/regime label (e.g. ``"classical_monte_carlo"``
        or ``"lattice_quadrature"``); WP5's non-goals do not fix a closed
        vocabulary, so this is not an enum.
    ensemble_size : int
        Number of atoms/nodes ``M`` in the ensemble.
    interrogation_time_s : float
        Interrogation time ``T``, seconds.
    mean_fractional_shift : float
        Weighted ``⟨Δν/ν₀⟩`` (E23), dimensionless.
    shift_std_error : float
        Standard error of `mean_fractional_shift`, dimensionless.
        ``float("nan")`` when undefined for the run (single-atom, M=1
        ensembles: E25's sample variance needs >= 2 effective samples --
        see `cliffordclock.pipeline._build_report`). This field stays a
        plain `float` (not `float | None`) so in-memory/numeric code keeps
        ordinary NaN-propagation semantics without unwrapping an Optional
        at every call site; the JSON *serialization* boundary is where
        this becomes a documented `null` instead of a bare (RFC-8259-
        invalid) `NaN` token -- see `write_json` and
        `docs/report-schema.md`. Every other float field in this
        dataclass is always finite for any successfully-built report.
    t2_star_s : float
        Inhomogeneous dephasing time ``T2*`` (E27), seconds. Same NaN-for-
        undefined / null-on-write convention as `shift_std_error` (both
        are undefined together, from the same M=1 variance computation).
        Additionally ``float("inf")`` when the ensemble phase variance is
        exactly zero (`dephasing_time_t2star`, E27's ``σ_Φ -> 0+`` limit:
        no inhomogeneous dephasing) -- a *defined* value, unlike the M=1
        NaN, but equally non-representable in RFC-8259 JSON, so it is
        also written as `null`; `build_report` records which case
        produced the `null` in `uncertainty_notes`.
    uncertainty_notes : str
        Free-text systematic-uncertainty notes (WP5 non-goal: no budget
        modeling -- this field only, default ``""``).
    ramsey_visibility : float or None
        The Ramsey fringe visibility ``V`` (WP31, CONVENTIONS.md section 8
        E39), `0 <= V <= 1`, dimensionless. `None` unless the run used a
        genuine per-worldline dynamical phase accumulation
        (`integration.mode` in ``("direct", "worldline")``,
        `cliffordclock.pipeline.run_pipeline_full`'s mode table) --
        `fast_path`/`secular` runs leave this `None`. Valid only for
        Gaussian-distributed accumulated phases (thermal, coherent,
        squeezed motional states); see `uncertainty_notes` for the
        Gaussian-only scope note recorded whenever this field is
        populated.
    ramsey_phase : float or None
        The Ramsey fringe phase, radians, in ``(-pi, pi]`` (E39). `None`
        under the same condition as `ramsey_visibility` (both are
        populated, or neither is).
    """

    report_schema: str
    conventions_version: str
    package_version: str
    generated_at_utc: str
    config_hash: str | None
    species_name: str
    ensemble_type: str
    ensemble_size: int
    interrogation_time_s: float
    mean_fractional_shift: float
    shift_std_error: float
    t2_star_s: float
    uncertainty_notes: str
    ramsey_visibility: float | None = None
    ramsey_phase: float | None = None


def build_report(
    phi: ArrayLike,
    species: Species,
    t_interrogation_s: float,
    ensemble_type: str,
    *,
    weights: ArrayLike | None = None,
    config_hash: str | None = None,
    uncertainty_notes: str = "",
    generated_at_utc: str | None = None,
    ramsey_visibility: float | None = None,
    ramsey_phase: float | None = None,
) -> MetrologyReport:
    """Assemble a `MetrologyReport` from ensemble phases and run provenance.

    Computes `mean_fractional_shift`, `shift_std_error`, and
    `dephasing_time_t2star` (E23, E25, E27) from `phi`/`weights`/
    `t_interrogation_s` via :mod:`cliffordclock.analytics.stats`, and fills
    in provenance (package/CONVENTIONS.md versions, timestamp,
    `config_hash`).

    Parameters
    ----------
    phi : array_like, shape (M,)
        Accumulated perturbation phases ``ΔΦ_i`` (E22), dimensionless. May
        be a `jax.Array` (e.g. `EnsembleResult.phase` from
        `cliffordclock.integrator.worldline`); converted to numpy float64.
    species : Species
        Atomic species (report provenance only: `species.name` is
        recorded. **[INTERPRETATION]**: none of E23/E25/E27 require
        ``ν₀`` directly -- the fractional-shift pipeline is already
        ν₀-normalized (E21-E22) -- so `species` does not otherwise
        participate in the shift/T2* arithmetic here).
    t_interrogation_s : float
        Interrogation time ``T``, seconds.
    ensemble_type : str
        Free-text ensemble/regime label; see `MetrologyReport.ensemble_type`.
    weights : array_like, shape (M,), optional
        Ensemble weights (E23); uniform ``1/M`` if omitted.
    config_hash : str, optional
        Caller-supplied config-provenance hash; see
        `MetrologyReport.config_hash`. ``None`` by default -- WP5 does not
        compute one itself.
    uncertainty_notes : str, default ""
        Free-text systematic-uncertainty notes.
    generated_at_utc : str, optional
        ISO-8601 UTC timestamp override (for reproducible tests); defaults
        to `datetime.now(UTC).isoformat()`.
    ramsey_visibility : float, optional
        The Ramsey fringe visibility (WP31, E39); see
        `MetrologyReport.ramsey_visibility`. `None` by default -- this
        module computes no rotor-composition physics itself (WP5 scope:
        "No new physics is implemented in this module"), so the caller
        (`cliffordclock.pipeline`) supplies the already-computed value.
    ramsey_phase : float, optional
        The Ramsey fringe phase (E39); see `MetrologyReport.ramsey_phase`.

    Returns
    -------
    MetrologyReport
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    if phi_arr.ndim != 1 or phi_arr.shape[0] < 1:
        raise ValueError(f"phi must be 1-D with at least one atom; got shape {phi_arr.shape}")
    ensemble_size = int(phi_arr.shape[0])

    mean_shift = mean_fractional_shift(phi_arr, t_interrogation_s, weights)
    sem = shift_std_error(phi_arr, t_interrogation_s, weights)
    t2_star = dephasing_time_t2star(phi_arr, t_interrogation_s, weights)

    if math.isinf(t2_star):
        # Disambiguate the JSON `null` this serializes to (see
        # `MetrologyReport.t2_star_s` / docs/report-schema.md): infinite
        # T2* (zero phase variance, E27 limit) vs. undefined (M=1 NaN).
        note = (
            "zero ensemble phase variance: T2* is infinite (no inhomogeneous "
            "dephasing, E27 sigma_Phi -> 0 limit); t2_star_s is written as null."
        )
        uncertainty_notes = f"{uncertainty_notes} {note}".strip() if uncertainty_notes else note

    timestamp = generated_at_utc if generated_at_utc is not None else datetime.now(UTC).isoformat()

    return MetrologyReport(
        report_schema=REPORT_SCHEMA_VERSION,
        conventions_version=CONVENTIONS_VERSION,
        package_version=_package_version(),
        generated_at_utc=timestamp,
        config_hash=config_hash,
        species_name=species.name,
        ensemble_type=ensemble_type,
        ensemble_size=ensemble_size,
        interrogation_time_s=float(t_interrogation_s),
        mean_fractional_shift=mean_shift,
        shift_std_error=sem,
        t2_star_s=t2_star,
        uncertainty_notes=uncertainty_notes,
        ramsey_visibility=ramsey_visibility,
        ramsey_phase=ramsey_phase,
    )


def _sanitize_non_finite_floats(data: dict[str, Any]) -> dict[str, Any]:
    """Replace non-finite floats (NaN/Inf) with JSON `null` (MAJOR 2 fix).

    A bare `NaN`/`Infinity` token is not valid JSON (RFC 8259 section 6
    permits only finite numbers): Python's `json.dump` happily emits one
    by default (`allow_nan=True`, the historical behavior this module used
    to rely on), which a strict parser (e.g. Node's `JSON.parse`) raises
    on and a lenient one (`jq`) silently turns into `null` anyway -- so
    the previous output was already being read as "undefined" by every
    consumer that didn't outright crash. This function makes that
    "undefined -> null" convention explicit and documented (see
    `docs/report-schema.md`) instead of an accident of `json.dump`
    defaults, for every non-finite value in the flat report dict
    (currently `shift_std_error`/`t2_star_s` can be NaN for M=1
    ensembles, and `t2_star_s` can additionally be `+inf` for
    zero-phase-variance ensembles -- see `MetrologyReport`'s field
    docstrings -- but this walks all fields generically rather than
    hand-listing them, so a future non-finite field is covered
    automatically).

    Parameters
    ----------
    data : dict[str, Any]
        A flat `dataclasses.asdict(report)` mapping (no nested
        dicts/lists of floats in `MetrologyReport`, so a single top-level
        pass is sufficient).

    Returns
    -------
    dict[str, Any]
        `data` with every non-finite `float` value replaced by `None`
        (which `json.dump` renders as `null`); everything else unchanged.
    """
    return {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in data.items()
    }


def write_json(report: MetrologyReport, path: str | Path) -> None:
    """Write `report` to `path` as schema-versioned JSON (test contract item 5).

    Key order matches `MetrologyReport`'s field declaration order exactly
    (``dataclasses.asdict`` preserves declaration order; `json.dump` does
    not reorder unless ``sort_keys=True``, which is not passed). Floats
    are written via Python's default `json` float encoding, which uses
    `repr(float)` internally (the shortest string that round-trips
    exactly) -- so no explicit precision handling is needed for the
    round-trip-safety requirement (WP5 test contract item 5; verified
    directly in ``tests/test_analytics_report.py``) for every field that
    is actually finite.

    Non-finite fields (`shift_std_error`/`t2_star_s` when undefined for
    M=1 ensembles, or `t2_star_s` when infinite for zero-phase-variance
    ensembles) are the one exception: they are written as JSON `null`
    (see :func:`_sanitize_non_finite_floats` and
    `docs/report-schema.md`), not a bare `NaN` token, since the latter is
    not valid per RFC 8259 and breaks strict JSON parsers. `allow_nan=False`
    is passed to `json.dump` as a backstop -- it makes `json.dump` raise
    `ValueError` instead of silently emitting an invalid `NaN`/`Infinity`
    token for any value `_sanitize_non_finite_floats` did not already
    convert to `None` (there should never be one, but this turns a future
    regression into a loud failure here rather than invalid JSON on disk).

    Parameters
    ----------
    report : MetrologyReport
    path : str or pathlib.Path
        Destination file path; overwritten if it exists.
    """
    data: dict[str, Any] = _sanitize_non_finite_floats(asdict(report))
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def write_line_profile_csv(
    freqs_hz: ArrayLike,
    amplitude: ArrayLike,
    path: str | Path,
) -> None:
    """Write a two-column line-profile CSV: `frequency_offset_hz`, `amplitude`.

    The header row is written as a single ``#``-prefixed comment (parsed
    by `numpy.loadtxt`'s default ``comments="#"``, or skippable with the
    stdlib `csv` module by dropping the first row); data rows are written
    with `repr` for full-precision, round-trip-safe floats (WP5 test
    contract item 6).

    Parameters
    ----------
    freqs_hz : array_like, shape (T,)
        Frequency offsets, hertz (e.g. from
        `cliffordclock.analytics.stats.line_profile`).
    amplitude : array_like, shape (T,)
        Normalized spectral amplitude, dimensionless, matching `freqs_hz`.
    path : str or pathlib.Path
        Destination file path; overwritten if it exists.
    """
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    amps = np.asarray(amplitude, dtype=np.float64)
    if freqs.shape != amps.shape:
        raise ValueError(
            f"freqs_hz and amplitude must have the same shape; got {freqs.shape} vs {amps.shape}"
        )
    if freqs.ndim != 1:
        raise ValueError(f"freqs_hz/amplitude must be 1-D; got shape {freqs.shape}")

    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# frequency_offset_hz", "amplitude"])
        for fr, am in zip(freqs.tolist(), amps.tolist(), strict=True):
            writer.writerow([repr(float(fr)), repr(float(am))])
