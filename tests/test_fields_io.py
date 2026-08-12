# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for cliffordclock.fields.io (WP2 test contract item 6: IO errors).

Covers CSV ingestion: valid regular-grid and scattered-cloud loads, and
informative-exception behavior on malformed input (missing column,
non-finite value, duplicate/inconsistent grid point, empty file,
non-numeric cell, short row).
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pytest

from cliffordclock.fields.io import (
    NearDuplicatePointsWarning,
    check_near_duplicate_points,
    load_field_csv,
)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _regular_grid_rows(n: int = 3) -> list[list[str]]:
    rows = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x, y, z = float(i) * 1e-3, float(j) * 1e-3, float(k) * 1e-3
                rows.append([str(x), str(y), str(z), str(x + 1.0), str(y - 1.0), str(-2.0 * z)])
    return rows


def test_load_regular_grid_detected(tmp_path: Path) -> None:
    """A full Cartesian-product grid is flagged regular=True with correct shape/axes."""
    path = tmp_path / "grid.csv"
    n = 4
    _write_csv(path, ["x", "y", "z", "Ex", "Ey", "Ez"], _regular_grid_rows(n))

    grid = load_field_csv(path)

    assert grid.regular is True
    assert grid.shape == (n, n, n)
    assert grid.points.shape == (n**3, 3)
    assert grid.values.shape == (n**3, 3)
    assert grid.points.dtype == np.float64
    assert grid.values.dtype == np.float64
    assert grid.axes is not None
    assert all(len(axis) == n for axis in grid.axes)
    assert grid.metadata["n_points"] == n**3


def test_load_column_order_independent(tmp_path: Path) -> None:
    """Required columns are looked up by name, not position."""
    path = tmp_path / "reordered.csv"
    _write_csv(
        path,
        ["Ez", "y", "Ex", "x", "Ey", "z"],
        [["-2.0", "0.0", "1.0", "0.0", "-1.0", "0.0"]],
    )
    grid = load_field_csv(path)
    # rtol=0, atol=0: the CSV literals are exactly representable doubles,
    # so column-name lookup must reproduce them exactly (points are all 0.0,
    # where any nonzero tolerance would be an abs-bound decision).
    np.testing.assert_allclose(grid.points[0], [0.0, 0.0, 0.0], rtol=0, atol=0)
    np.testing.assert_allclose(grid.values[0], [1.0, -1.0, -2.0], rtol=0, atol=0)


def test_load_scattered_cloud_flagged_not_regular(tmp_path: Path) -> None:
    """A genuinely scattered (non-grid) point cloud is accepted, flagged regular=False."""
    path = tmp_path / "scattered.csv"
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1e-3, 1e-3, size=(50, 3))
    rows = [[str(x), str(y), str(z), str(x), str(y), str(z)] for x, y, z in pts]
    _write_csv(path, ["x", "y", "z", "Ex", "Ey", "Ez"], rows)

    grid = load_field_csv(path)

    assert grid.regular is False
    assert grid.axes is None
    assert grid.shape is None
    assert grid.points.shape == (50, 3)


def test_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "missing_col.csv"
    _write_csv(path, ["x", "y", "z", "Ex", "Ey"], [["0", "0", "0", "1", "1"]])
    with pytest.raises(ValueError, match="missing required column"):
        load_field_csv(path)


def test_non_finite_value_raises(tmp_path: Path) -> None:
    path = tmp_path / "nan.csv"
    _write_csv(
        path,
        ["x", "y", "z", "Ex", "Ey", "Ez"],
        [["0", "0", "0", "nan", "1", "1"]],
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_field_csv(path)


def test_non_finite_coordinate_raises(tmp_path: Path) -> None:
    path = tmp_path / "inf_coord.csv"
    _write_csv(
        path,
        ["x", "y", "z", "Ex", "Ey", "Ez"],
        [["inf", "0", "0", "1", "1", "1"]],
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_field_csv(path)


def test_duplicate_point_raises_inconsistent_grid(tmp_path: Path) -> None:
    """Two rows claiming the same (x, y, z) with different field values is malformed input."""
    path = tmp_path / "dup.csv"
    _write_csv(
        path,
        ["x", "y", "z", "Ex", "Ey", "Ez"],
        [
            ["0", "0", "0", "1", "1", "1"],
            ["0", "0", "0", "2", "2", "2"],
        ],
    )
    with pytest.raises(ValueError, match="inconsistent grid"):
        load_field_csv(path)


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_field_csv(path)


def test_no_data_rows_raises(tmp_path: Path) -> None:
    path = tmp_path / "header_only.csv"
    _write_csv(path, ["x", "y", "z", "Ex", "Ey", "Ez"], [])
    with pytest.raises(ValueError, match="no data rows"):
        load_field_csv(path)


def test_non_numeric_value_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad_value.csv"
    _write_csv(
        path,
        ["x", "y", "z", "Ex", "Ey", "Ez"],
        [["0", "0", "0", "not_a_number", "1", "1"]],
    )
    with pytest.raises(ValueError, match="non-numeric"):
        load_field_csv(path)


def test_short_row_raises(tmp_path: Path) -> None:
    path = tmp_path / "short_row.csv"
    _write_csv(path, ["x", "y", "z", "Ex", "Ey", "Ez"], [["0", "0", "0", "1", "1"]])
    with pytest.raises(ValueError, match="columns"):
        load_field_csv(path)


def test_near_duplicate_points_differing_values_raises(tmp_path: Path) -> None:
    """FINDING 2(b): near- (not exactly-) coincident points with different values.

    Distinct (x, y, z) tuples, so the exact-duplicate check does not fire,
    but the two points are far below the 1e-9-times-domain-diagonal
    near-duplicate threshold apart and report inconsistent field values --
    almost certainly corrupted/duplicated sample data.
    """
    path = tmp_path / "near_dup_bad.csv"
    rows = _regular_grid_rows(4)
    rows.append(["1.0000000000005e-3", "0", "0", "999.0", "999.0", "999.0"])
    _write_csv(path, ["x", "y", "z", "Ex", "Ey", "Ez"], rows)
    with pytest.raises(ValueError, match="near-duplicate points with differing values"):
        load_field_csv(path)


def test_near_duplicate_points_same_value_warns(tmp_path: Path) -> None:
    """FINDING 2(b): near-duplicate points with agreeing values warn instead of raising."""
    path = tmp_path / "near_dup_ok.csv"
    rows = _regular_grid_rows(4)
    # Same field value as the point at (1e-3, 0, 0) (Ex = x + 1.0 = 1.001)
    # it sits a fraction of a nanometer from.
    rows.append(["1.0000000000005e-3", "0", "0", "1.001", "-1.0", "0.0"])
    _write_csv(path, ["x", "y", "z", "Ex", "Ey", "Ez"], rows)

    with pytest.warns(NearDuplicatePointsWarning):
        grid = load_field_csv(path)
    assert grid.points.shape[0] == len(rows)


def test_no_near_duplicate_warning_on_well_separated_points(tmp_path: Path) -> None:
    """Clean regular-grid data triggers no NearDuplicatePointsWarning."""
    path = tmp_path / "clean.csv"
    _write_csv(path, ["x", "y", "z", "Ex", "Ey", "Ez"], _regular_grid_rows(4))
    with warnings.catch_warnings():
        warnings.simplefilter("error", NearDuplicatePointsWarning)
        load_field_csv(path)


def test_check_near_duplicate_points_helper_no_values_warns() -> None:
    """check_near_duplicate_points without a `values` array always warns, never raises."""
    points = np.array([[0.0, 0.0, 0.0], [1e-15, 0.0, 0.0], [1.0, 1.0, 1.0]])
    with pytest.warns(NearDuplicatePointsWarning):
        check_near_duplicate_points(points)
