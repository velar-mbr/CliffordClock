# SPDX-License-Identifier: AGPL-3.0-or-later
"""CSV and COMSOL ingestion for FEA-exported electric field grids.

Reads the CSV export format described in ``docs/fields.md``: a header row
``x,y,z,Ex,Ey,Ez`` followed by one row per sample point, positions in
meters and field components in V/m (units per CONVENTIONS.md §10). Regular
(axis-aligned, product) grids are detected automatically; genuinely
scattered point clouds are accepted and flagged ``regular=False`` rather
than rejected, since Sprint-2 solvers may want unstructured input too.

Also reads COMSOL's "Spreadsheet" export format (WP17,
:func:`load_field_comsol`): the ``%``-prefixed header/column-header block
COMSOL writes for ``File > Export > Data`` with the "Spreadsheet" file
type, ending in a static (non-swept) 3D electrostatic study's
``es.Ex``/``es.Ey``/``es.Ez`` columns. See ``docs/fields.md`` and
``docs/byof-guide.md`` for the format contract and export-dialog guidance.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

#: Header columns required in every field CSV, in the order documented in
#: docs/fields.md. Column *order* in the file does not matter (looked up by
#: name); all six must be present.
REQUIRED_COLUMNS: tuple[str, ...] = ("x", "y", "z", "Ex", "Ey", "Ez")

#: Default near-duplicate-point threshold for :func:`check_near_duplicate_points`,
#: as a fraction of the point cloud's bounding-box diagonal.
NEAR_DUPLICATE_REL_TOL = 1e-9


class NearDuplicatePointsWarning(UserWarning):
    """Warned when two distinct sample points sit far closer together than the domain scale.

    Points that are not exactly coincident (so they pass the exact-duplicate
    check in :func:`load_field_csv`) but are, say, 1e-9 m apart on a
    millimeter-scale domain are almost certainly duplicated/noisy sample
    locations rather than a genuine sub-nanometer feature of the field. Left
    alone they make :class:`~cliffordclock.fields.smoother.FieldSmoother`'s
    RBF kernel matrix (built from pairwise distances between fit points)
    severely ill-conditioned -- see
    :class:`~cliffordclock.fields.smoother.IllConditionedFitWarning`.
    """


def check_near_duplicate_points(
    points: NDArray[np.float64],
    values: NDArray[np.float64] | None = None,
    *,
    rel_tol: float = NEAR_DUPLICATE_REL_TOL,
    context: str = "",
) -> None:
    """Detect near- (but not exactly-) duplicate sample points.

    Shared validation helper used both by :func:`load_field_csv` (on the raw
    CSV point cloud) and by
    :meth:`cliffordclock.fields.smoother.FieldSmoother.fit` (on the fit's
    RBF centers, which may come from ``synthetic.py`` or hand-built
    :class:`FieldGrid`\\ s that never go through this module's CSV path).
    It lives here rather than being duplicated in ``smoother.py`` because
    the check is about point-cloud geometry, not about the RBF fit itself.

    Uses a ``scipy.spatial.cKDTree`` nearest-neighbor query -- O(N log N)
    and O(N) memory -- rather than an O(N^2) full pairwise distance matrix,
    so it stays cheap even near :data:`~cliffordclock.fields.smoother.MAX_FIT_POINTS`.

    Note: exact coincident-coordinate duplicates (distance exactly 0) are
    handled separately by :func:`load_field_csv`'s dedicated check, which
    unconditionally rejects them; this function targets the *near*-duplicate
    regime that check cannot see.

    Parameters
    ----------
    points : NDArray[np.float64], shape (N, 3)
        Sample positions, meters.
    values : NDArray[np.float64], shape (N, 3), optional
        Field values at ``points``. When given, a near-duplicate pair whose
        values disagree (outside :func:`numpy.allclose`'s default
        tolerance) is treated as corrupted/inconsistent input and raised;
        a pair with matching values is treated as merely redundant
        sampling and only warned about. When omitted, any near-duplicate
        pair is warned about.
    rel_tol : float, default 1e-9
        Points closer than ``rel_tol`` times the point cloud's bounding-box
        diagonal are flagged.
    context : str, default ""
        Optional prefix (e.g. a source file path) for the raised/warned
        message.

    Raises
    ------
    ValueError
        A near-duplicate pair has disagreeing ``values``.

    Warns
    -----
    NearDuplicatePointsWarning
        A near-duplicate pair was found and not raised as an error.
    """
    n = points.shape[0]
    if n < 2:
        return
    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if bbox_diag == 0.0:
        return  # every point at the same location; nothing to compare against

    threshold = rel_tol * bbox_diag
    tree = cKDTree(points)
    # k=2: column 0 is each point matched to itself (distance 0), column 1
    # is the true nearest *other* point.
    nn_dist, nn_idx = tree.query(points, k=2)
    nearest = nn_dist[:, 1]
    i = int(np.argmin(nearest))
    min_dist = float(nearest[i])
    if min_dist >= threshold:
        return
    j = int(nn_idx[i, 1])

    prefix = f"{context}: " if context else ""
    if values is not None and not np.allclose(values[i], values[j]):
        raise ValueError(
            f"{prefix}near-duplicate points with differing values: points {i} and "
            f"{j} are {min_dist:.3e} m apart (< {rel_tol:.0e} x the {bbox_diag:.3e} m "
            f"domain diagonal = {threshold:.3e} m) but report different field values "
            f"{values[i].tolist()} vs {values[j].tolist()}; this is almost certainly "
            "duplicated or corrupted sample data, not a genuine feature of the field"
        )
    warnings.warn(
        f"{prefix}near-duplicate points found: points {i} and {j} are {min_dist:.3e} m "
        f"apart, below {rel_tol:.0e} x the {bbox_diag:.3e} m domain diagonal "
        f"({threshold:.3e} m); this can make an RBF fit's kernel matrix severely "
        "ill-conditioned -- consider deduplicating near-coincident points or "
        "increasing FieldSmoother.fit's `smoothing`",
        NearDuplicatePointsWarning,
        stacklevel=3,
    )


@dataclass(frozen=True, eq=False)
class FieldGrid:
    """An ingested E-field point cloud, possibly on a regular grid.

    Attributes
    ----------
    points : NDArray[np.float64], shape (N, 3)
        Sample positions in meters.
    values : NDArray[np.float64], shape (N, 3)
        Field vectors ``(Ex, Ey, Ez)`` in V/m at each point.
    regular : bool
        ``True`` if ``points`` forms the complete Cartesian product of
        three 1D axes (an axis-aligned regular grid), ``False`` for
        scattered/incomplete point clouds.
    axes : tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
        Sorted unique ``(x, y, z)`` axis coordinates, in meters, when
        ``regular`` is ``True``; ``None`` otherwise.
    shape : tuple[int, int, int] | None
        ``(len(x_axis), len(y_axis), len(z_axis))`` when ``regular`` is
        ``True``; ``None`` otherwise.
    metadata : dict[str, Any]
        Free-form provenance info (e.g. source file path, point count).

    Notes
    -----
    ``eq=False`` because the default dataclass equality would compare the
    ``points``/``values`` arrays with ``==``, which raises when used in a
    boolean context (e.g. inside ``assert grid1 == grid2``).
    """

    points: NDArray[np.float64]
    values: NDArray[np.float64]
    regular: bool
    axes: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
    shape: tuple[int, int, int] | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _detect_regular_grid(
    points: NDArray[np.float64],
) -> tuple[
    bool,
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None,
    tuple[int, int, int] | None,
]:
    """Detect whether ``points`` is the full Cartesian product of its axes.

    Returns ``(regular, axes, shape)``, with ``axes`` typed as the exact
    3-tuple :attr:`FieldGrid.axes` expects (rather than a variable-length
    ``tuple[NDArray, ...]``), so callers can assign the result straight
    into a ``FieldGrid`` without a ``# type: ignore``. ``points`` is
    assumed already checked for duplicate rows by the caller.
    """
    x_axis = np.unique(points[:, 0])
    y_axis = np.unique(points[:, 1])
    z_axis = np.unique(points[:, 2])
    shape = (len(x_axis), len(y_axis), len(z_axis))
    expected_n = shape[0] * shape[1] * shape[2]
    n = points.shape[0]
    if expected_n != n:
        return False, None, None

    xi = np.searchsorted(x_axis, points[:, 0])
    yi = np.searchsorted(y_axis, points[:, 1])
    zi = np.searchsorted(z_axis, points[:, 2])
    flat_index = (xi * shape[1] + yi) * shape[2] + zi
    if len(np.unique(flat_index)) != expected_n:
        # Same axis values repeated but not covering every combination.
        return False, None, None

    return True, (x_axis, y_axis, z_axis), shape


def load_field_csv(path: str | Path) -> FieldGrid:
    """Load a CSV-exported E-field grid.

    Parameters
    ----------
    path : str | Path
        Path to a CSV file with header ``x,y,z,Ex,Ey,Ez`` (column order in
        the file is irrelevant; columns are looked up by name). Positions
        are meters, field components V/m.

    Returns
    -------
    FieldGrid
        The parsed point cloud. ``regular`` is set automatically.

    Raises
    ------
    ValueError
        On any malformed input: missing/empty file, missing required
        column(s), non-numeric cell, wrong column count on a data row,
        non-finite (NaN/Inf) coordinate or field value, duplicate
        ``(x, y, z)`` points (an internally inconsistent "grid" — the same
        location cannot carry two different field values), or near- (but
        not exactly-) duplicate points with disagreeing values (see
        :func:`check_near_duplicate_points`).

    Warns
    -----
    NearDuplicatePointsWarning
        Near-duplicate points were found with agreeing values; see
        :func:`check_near_duplicate_points`.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: CSV file is empty (no header row)") from None

        header = [h.strip() for h in header]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(
                f"{path}: missing required column(s) {missing}; "
                f"header must contain all of {REQUIRED_COLUMNS}, got {header}"
            )
        col_idx = {c: header.index(c) for c in REQUIRED_COLUMNS}

        rows: list[list[float]] = []
        for lineno, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            if len(row) < len(header):
                raise ValueError(
                    f"{path}: line {lineno}: expected {len(header)} columns, got {len(row)}"
                )
            try:
                parsed = [float(row[col_idx[c]]) for c in REQUIRED_COLUMNS]
            except ValueError as exc:
                raise ValueError(f"{path}: line {lineno}: non-numeric value ({exc})") from exc
            rows.append(parsed)

    if not rows:
        raise ValueError(f"{path}: no data rows found")

    data = np.asarray(rows, dtype=np.float64)
    points = np.ascontiguousarray(data[:, :3])
    values = np.ascontiguousarray(data[:, 3:])

    if not np.all(np.isfinite(points)):
        raise ValueError(f"{path}: non-finite (NaN or Inf) coordinate value(s) found")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: non-finite (NaN or Inf) field value(s) found")

    _, counts = np.unique(points, axis=0, return_counts=True)
    n_duplicated = int(np.sum(counts > 1))
    if n_duplicated:
        raise ValueError(
            f"{path}: inconsistent grid — {n_duplicated} duplicate (x, y, z) "
            "point location(s) with distinct rows; each location must appear once"
        )

    check_near_duplicate_points(points, values, context=str(path))

    regular, axes, shape = _detect_regular_grid(points)
    metadata: dict[str, Any] = {"source_path": str(path), "n_points": points.shape[0]}
    return FieldGrid(
        points=points,
        values=values,
        regular=regular,
        axes=axes,
        shape=shape,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# COMSOL "Spreadsheet" export format (WP17)
# ---------------------------------------------------------------------------

#: Length-unit conversion factors to meters, as they appear on a COMSOL
#: export's ``% Length unit:`` line. Anything else raises a loud
#: :class:`ValueError` naming the unsupported unit (WP17 Phase-A scope: the
#: MVP supports the common metric length units; no imperial units, no
#: unusual COMSOL unit strings).
_COMSOL_LENGTH_UNIT_TO_M: dict[str, float] = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}

#: Field-component-unit conversion factors to V/m, as they appear in
#: parentheses after an ``es.Ex``/``es.Ey``/``es.Ez``-style column name
#: (e.g. ``es.Ex (kV/m)``). ``V/cm`` -> ``V/m`` is x100 (1 V/cm = 1 V per
#: 0.01 m = 100 V/m), not x10 -- easy sign to get backwards, called out here.
_COMSOL_FIELD_UNIT_TO_V_PER_M: dict[str, float] = {"V/m": 1.0, "kV/m": 1e3, "V/cm": 1e2}

#: The three field-component suffixes this loader recognizes, matched
#: against ``f"{expression_prefix}.{suffix}"`` (default prefix ``"es"``,
#: COMSOL's built-in Electrostatics physics interface tag).
_COMSOL_COMPONENT_SUFFIXES: tuple[str, ...] = ("Ex", "Ey", "Ez")


def _comsol_get_metadata(metadata_raw: dict[str, str], key: str) -> str | None:
    """Case-insensitive lookup into a COMSOL header's ``{key: value}`` metadata dict."""
    for k, v in metadata_raw.items():
        if k.lower() == key.lower():
            return v
    return None


def load_field_comsol(path: str | Path, *, expression_prefix: str = "es") -> FieldGrid:
    """Load a COMSOL "Spreadsheet"-format ``File > Export > Data`` E-field export.

    Parses the ``%``-prefixed header block COMSOL writes for the
    Spreadsheet export type (``.txt``/``.csv``/``.dat``):

    .. code-block:: text

        % Model:      my_model.mph
        % Version:    COMSOL 6.0.0.318
        % Date:       Jan 1 2026, 00:00
        % Dimension:  3
        % Nodes:      1000
        % Expressions: 3
        % Description: Electric field
        % Length unit: mm
        % X    Y    Z    es.Ex (V/m)   es.Ey (V/m)   es.Ez (V/m)
        1.0    2.0  3.0  123.4         -56.7         8.9
        ...

    Whitespace- and comma-delimited (``.csv``-exported) variants are both
    accepted (detected from the final header line). Positions are
    converted to meters and field components to V/m per the header's
    declared units (``docs/byof-guide.md``'s "COMSOL exports" section has
    the exact export-dialog settings a user should pick); everything is
    validated with the same checks :func:`load_field_csv` uses (finite
    values, exact/near-duplicate points).

    Only the documented common case is supported (WP17 Phase-A scope
    rule): a single, non-parameterized 3D static-study export with
    ``es.Ex``/``es.Ey``/``es.Ez`` (or another ``expression_prefix``)
    columns in m/mm/cm and V/m/kV/m/V per cm. Parameter-sweep exports
    (a trailing ``@ param=value`` tag), 2D exports, complex-valued
    (frequency-domain) exports, and COMSOL's "Sectionwise" format are all
    explicitly out of scope and rejected with a descriptive error rather
    than silently mis-parsed.

    Parameters
    ----------
    path : str | Path
        Path to the exported file.
    expression_prefix : str, default "es"
        The physics-interface tag COMSOL prefixes the field-component
        expression names with (``es`` = the built-in Electrostatics
        interface, giving ``es.Ex``/``es.Ey``/``es.Ez``). Pass a different
        prefix if your model uses a renamed or non-default physics
        interface.

    Returns
    -------
    FieldGrid
        The parsed grid, in the same shape :func:`load_field_csv` returns.
        ``metadata`` additionally carries the COMSOL header fields found
        (``model``, ``version``, ``date``, ``description``, ``length_unit``,
        ``field_units``, ``expression_prefix``).

    Raises
    ------
    ValueError
        On any malformed or out-of-scope input: not a ``%``-header
        Spreadsheet export (e.g. "Sectionwise" format), truncated/missing
        header fields (``Dimension``, ``Length unit``), a non-3D export, a
        parameterized/swept-study ``@`` tag, an unrecognized
        ``expression_prefix.Ex/Ey/Ez`` column, an unsupported length or
        field unit, a malformed/non-numeric or complex-valued data cell,
        a data row with the wrong column count, a ``Nodes``/``Expressions``
        count that disagrees with the actual data, non-finite coordinate
        or field values, duplicate ``(x, y, z)`` points, or near-duplicate
        points with disagreeing values (see
        :func:`check_near_duplicate_points`).

    Warns
    -----
    NearDuplicatePointsWarning
        Near-duplicate points were found with agreeing values; see
        :func:`check_near_duplicate_points`.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or all(line.strip() == "" for line in lines):
        raise ValueError(f"{path}: COMSOL export file is empty")

    if not lines[0].lstrip().startswith("%"):
        raise ValueError(
            f"{path}: does not look like a COMSOL 'Spreadsheet' export -- the first line "
            "does not start with '%'. This loader only supports COMSOL's Spreadsheet "
            "export format (File > Export > Data, file type 'Spreadsheet'); the "
            "'Sectionwise' format (and other export types) are not supported -- "
            "see docs/byof-guide.md's 'COMSOL exports' section for the exact dialog "
            "settings to use."
        )

    header_lines: list[str] = []
    body_start = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith("%"):
            header_lines.append(line)
            body_start = i + 1
        else:
            break

    if len(header_lines) < 2:
        raise ValueError(
            f"{path}: truncated or malformed COMSOL header -- expected a metadata block "
            "(Model/Version/Dimension/.../Length unit lines) followed by a final "
            "'%'-prefixed column-header line (e.g. '% X Y Z es.Ex (V/m) es.Ey (V/m) "
            f"es.Ez (V/m)'), but found only {len(header_lines)} '%'-prefixed line(s) "
            "before the data started"
        )

    column_header_line = header_lines[-1]
    metadata_lines = header_lines[:-1]

    metadata_raw: dict[str, str] = {}
    for line in metadata_lines:
        content = line.lstrip().lstrip("%").strip()
        if ":" not in content:
            continue
        key, _, value = content.partition(":")
        metadata_raw[key.strip()] = value.strip()

    dimension_str = _comsol_get_metadata(metadata_raw, "Dimension")
    if dimension_str is None:
        raise ValueError(
            f"{path}: truncated or malformed COMSOL header -- missing the required "
            "'% Dimension:' line"
        )
    try:
        dimension = int(dimension_str)
    except ValueError:
        raise ValueError(
            f"{path}: unparseable '% Dimension:' value {dimension_str!r} (expected an integer)"
        ) from None

    if dimension != 3:
        raise ValueError(
            f"{path}: {dimension}D COMSOL export is not supported by this field loader -- "
            "it requires a 3D export ('% Dimension: 3'). Re-export from a 3D model/study; "
            "2D (and other non-3D) exports are out of scope (WP17 non-goals)."
        )

    length_unit = _comsol_get_metadata(metadata_raw, "Length unit")
    if length_unit is None:
        raise ValueError(
            f"{path}: truncated or malformed COMSOL header -- missing the required "
            "'% Length unit:' line"
        )
    if length_unit not in _COMSOL_LENGTH_UNIT_TO_M:
        raise ValueError(
            f"{path}: unsupported length unit {length_unit!r} on the '% Length unit:' "
            f"line (supported: {sorted(_COMSOL_LENGTH_UNIT_TO_M)}); re-export with a "
            "supported length unit, or convert the file before loading it"
        )
    length_factor = _COMSOL_LENGTH_UNIT_TO_M[length_unit]

    nodes_str = _comsol_get_metadata(metadata_raw, "Nodes")
    declared_nodes: int | None = None
    if nodes_str is not None:
        try:
            declared_nodes = int(nodes_str)
        except ValueError:
            raise ValueError(
                f"{path}: unparseable '% Nodes:' value {nodes_str!r} (expected an integer)"
            ) from None

    # Parameter-sweep exports tag every study point's column header with a
    # trailing "@ param=value"; out of scope (WP17 non-goals). Checked only
    # now (after the Dimension gate above) so a 2D *and* swept file like the
    # WP17 Phase-A real-world fixture is rejected for its Dimension, the
    # more specific and actionable of the two problems.
    if "@" in column_header_line:
        raise ValueError(
            f"{path}: COMSOL column header contains an '@ param=value' tag -- this is a "
            "parameterized/swept-study export (e.g. a frequency, voltage, or geometry "
            "sweep), which is out of scope for this loader (WP17 non-goals: no "
            "parameter-sweep/multi-dataset files). Re-export a single static study "
            "without any parametric sweep."
        )

    header_content = column_header_line.lstrip().lstrip("%").strip()
    use_comma = "," in header_content
    header_tokens = (
        [t.strip() for t in header_content.split(",") if t.strip() != ""]
        if use_comma
        else header_content.split()
    )

    if len(header_tokens) < dimension:
        raise ValueError(
            f"{path}: column-header line has fewer tokens ({len(header_tokens)}) than "
            f"the declared Dimension ({dimension}): {header_tokens!r}"
        )

    position_tokens = header_tokens[:dimension]
    expected_position_labels = ["X", "Y", "Z"][:dimension]
    if [t.upper() for t in position_tokens] != expected_position_labels:
        raise ValueError(
            f"{path}: expected position columns {expected_position_labels} at the start "
            f"of the column-header line, got {position_tokens!r}"
        )

    expr_tokens = header_tokens[dimension:]
    columns: list[tuple[str, str | None]] = []
    i = 0
    while i < len(expr_tokens):
        name = expr_tokens[i]
        i += 1
        unit: str | None = None
        if i < len(expr_tokens) and expr_tokens[i].startswith("("):
            unit_token = expr_tokens[i]
            if not unit_token.endswith(")"):
                raise ValueError(
                    f"{path}: malformed unit annotation {unit_token!r} following "
                    f"column {name!r} in the column-header line"
                )
            unit = unit_token[1:-1]
            i += 1
        columns.append((name, unit))

    expressions_str = _comsol_get_metadata(metadata_raw, "Expressions")
    if expressions_str is not None:
        try:
            declared_expressions = int(expressions_str)
        except ValueError:
            raise ValueError(
                f"{path}: unparseable '% Expressions:' value {expressions_str!r} "
                "(expected an integer)"
            ) from None
        if declared_expressions != len(columns):
            raise ValueError(
                f"{path}: header declares '% Expressions: {declared_expressions}' but "
                f"the column-header line lists {len(columns)} expression column(s): "
                f"{[name for name, _ in columns]!r}"
            )

    name_to_index_unit = {name: (idx, unit) for idx, (name, unit) in enumerate(columns)}
    expected_names = {comp: f"{expression_prefix}.{comp}" for comp in _COMSOL_COMPONENT_SUFFIXES}
    missing = [name for name in expected_names.values() if name not in name_to_index_unit]
    if missing:
        available = [name for name, _ in columns]
        raise ValueError(
            f"{path}: expected field columns {list(expected_names.values())} "
            f"(expression_prefix={expression_prefix!r}) not found in the column header; "
            f"available expression column(s): {available!r}. Pass a different "
            "`expression_prefix` if your model's Electrostatics interface was renamed."
        )

    field_factors: dict[str, float] = {}
    field_col_index: dict[str, int] = {}
    field_units: dict[str, str] = {}
    for comp, name in expected_names.items():
        idx, unit = name_to_index_unit[name]
        if unit is None:
            raise ValueError(
                f"{path}: column {name!r} is missing a unit annotation (expected e.g. "
                f"'{name} (V/m)')"
            )
        if unit not in _COMSOL_FIELD_UNIT_TO_V_PER_M:
            raise ValueError(
                f"{path}: unsupported field unit {unit!r} for column {name!r} "
                f"(supported: {sorted(_COMSOL_FIELD_UNIT_TO_V_PER_M)}); re-export with a "
                "supported unit, or convert the file before loading it"
            )
        field_factors[comp] = _COMSOL_FIELD_UNIT_TO_V_PER_M[unit]
        field_col_index[comp] = dimension + idx
        field_units[comp] = unit

    total_columns = dimension + len(columns)
    body_lines = lines[body_start:]
    rows: list[list[float]] = []
    for lineno, raw_line in enumerate(body_lines, start=body_start + 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        tokens = [t.strip() for t in stripped.split(",")] if use_comma else stripped.split()
        if len(tokens) != total_columns:
            raise ValueError(
                f"{path}: line {lineno}: expected {total_columns} column(s), got {len(tokens)}"
            )
        try:
            parsed = [float(t) for t in tokens]
        except ValueError as exc:
            raise ValueError(
                f"{path}: line {lineno}: non-numeric or complex-valued value ({exc}) -- "
                "this loader does not support complex-valued (frequency-domain study) "
                "COMSOL exports"
            ) from exc
        rows.append(parsed)

    if not rows:
        raise ValueError(f"{path}: no data rows found after the header")

    if declared_nodes is not None and declared_nodes != len(rows):
        raise ValueError(
            f"{path}: header declares '% Nodes: {declared_nodes}' but {len(rows)} data "
            "row(s) were found"
        )

    data = np.asarray(rows, dtype=np.float64)
    points = np.ascontiguousarray(data[:, :dimension]) * length_factor
    values = np.empty((data.shape[0], 3), dtype=np.float64)
    for axis_idx, comp in enumerate(_COMSOL_COMPONENT_SUFFIXES):
        values[:, axis_idx] = data[:, field_col_index[comp]] * field_factors[comp]
    values = np.ascontiguousarray(values)

    if not np.all(np.isfinite(points)):
        raise ValueError(f"{path}: non-finite (NaN or Inf) coordinate value(s) found")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: non-finite (NaN or Inf) field value(s) found")

    _, counts = np.unique(points, axis=0, return_counts=True)
    n_duplicated = int(np.sum(counts > 1))
    if n_duplicated:
        raise ValueError(
            f"{path}: inconsistent grid -- {n_duplicated} duplicate (x, y, z) point "
            "location(s) with distinct rows; each location must appear once"
        )

    check_near_duplicate_points(points, values, context=str(path))

    regular, axes, shape = _detect_regular_grid(points)
    metadata: dict[str, Any] = {
        "source_path": str(path),
        "n_points": points.shape[0],
        "format": "comsol",
        "expression_prefix": expression_prefix,
        "length_unit": length_unit,
        "field_units": field_units,
    }
    for key in ("Model", "Version", "Date", "Description"):
        field_value = _comsol_get_metadata(metadata_raw, key)
        if field_value is not None:
            metadata[key.lower()] = field_value

    return FieldGrid(
        points=points,
        values=values,
        regular=regular,
        axes=axes,
        shape=shape,
        metadata=metadata,
    )
