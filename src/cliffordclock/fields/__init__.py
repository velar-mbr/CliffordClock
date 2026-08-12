# SPDX-License-Identifier: AGPL-3.0-or-later
"""Electric field importer and C² smoother.

Ingests CSV-exported (:func:`load_field_csv`) or COMSOL-exported
(:func:`load_field_comsol`) E-field grids, decomposes them into an
analytical baseline plus residual (E11, :func:`fit_baseline`), and fits a
thin-plate-spline RBF smoother (E12, :class:`FieldSmoother`) whose
``evaluate`` is pure JAX and differentiable end to end. See
``docs/fields.md`` for the CSV/COMSOL formats and usage, and
:mod:`cliffordclock.fields.synthetic` for closed-form test fields.
"""

from cliffordclock.fields.decompose import Baseline, fit_baseline, residual
from cliffordclock.fields.io import (
    FieldGrid,
    NearDuplicatePointsWarning,
    check_near_duplicate_points,
    load_field_comsol,
    load_field_csv,
)
from cliffordclock.fields.smoother import (
    DEFAULT_CHUNK_SIZE,
    FieldSmoother,
    IllConditionedFitWarning,
    OutOfBoundsWarning,
    chunked_apply,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "Baseline",
    "FieldGrid",
    "FieldSmoother",
    "IllConditionedFitWarning",
    "NearDuplicatePointsWarning",
    "OutOfBoundsWarning",
    "check_near_duplicate_points",
    "chunked_apply",
    "fit_baseline",
    "load_field_comsol",
    "load_field_csv",
    "residual",
]
