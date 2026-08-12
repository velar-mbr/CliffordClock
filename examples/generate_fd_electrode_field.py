#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate ``examples/fd_electrode_field.txt``, a from-scratch open-FEA COMSOL fixture.

The end-to-end proof that :func:`cliffordclock.fields.io.load_field_comsol`
actually works against a field nobody hand-typed: this script (a) solves a
real (if simple) electrostatics boundary-value problem with a
finite-difference Laplace solver built from nothing but ``numpy`` +
``scipy.sparse`` (no COMSOL, no paid software, no new dependency beyond
what this project already ships with), then (b) writes the solved field
out through a small COMSOL-"Spreadsheet"-format writer (also in this
file), independent of ``load_field_comsol``'s own parsing code. Loading
the committed output back through ``load_field_comsol`` (the round trip
exercised by ``tests/test_fields_comsol.py``'s end-to-end test) is
therefore a genuine test of the loader against an independently generated
file, not a fixture massaged to match the parser.

Physics of the scenario
------------------------

**Geometry.** A grounded conducting box (``BOX_*_M``, 10 mm x 10 mm x
8 mm) contains two square, axis-aligned electrode plates
(``PLATE_XY_LO_M``-``PLATE_XY_HI_M`` on a side, 6 mm x 6 mm) held at
``+PLATE_VOLTAGE_V`` and ``-PLATE_VOLTAGE_V`` (+/-2 V), facing each other
across a ``PLATE_SEPARATION_M`` = 3.2 mm gap, centered in the box on all
three axes. This is the simplest electrostatics problem with a clean,
independently-checkable answer: a parallel-plate capacitor, inside a
grounded enclosure so the boundary-value problem is well-posed on a finite
domain (an isolated pair of plates in free space has no natural outer
boundary condition for a box-truncated finite-difference grid).

**Solve.** Laplace's equation (``grad^2 V = 0``, no free charge away from
the electrodes) is discretized on a regular grid (spacing ``H_M`` = 0.2 mm)
with the standard second-order 7-point stencil: each free (non-electrode,
non-wall) node's potential equals the average of its six axis-neighbors.
Electrode and wall nodes are Dirichlet boundary conditions (fixed
potential), assembled into a sparse linear system over the free nodes only
and solved by (Jacobi-preconditioned) conjugate gradient
(``scipy.sparse.linalg.cg``) -- the system is symmetric positive-definite
(a discrete Laplacian restricted to its free-node block), so CG is the
natural choice; it converges to ``rtol=1e-10`` in well under a second at
this grid size (~107k total nodes, ~92k free). The field is then
``E = -grad V`` (``numpy.gradient``, same central-difference stencil
family as the solve, so the field it reports is discretely consistent with
the potential that produced it).

Back-of-envelope check (this is the "proof", not just a description)
-----------------------------------------------------------------------

At the domain center -- equidistant between the two plates, deep inside
both plates' 6 mm footprint (2.1 mm margin from each plate's own edge, so
comfortably clear of the edge-fringing region) -- the ideal
infinite-parallel-plate estimate is::

    E_ideal = (2 * PLATE_VOLTAGE_V) / PLATE_SEPARATION_M
            = (2 * 2.0 V) / 3.2e-3 m
            = 1250.0 V/m

The finite-difference solve (:func:`main` prints the live numbers) gives
``E_z(center) ~= 1243.8 V/m`` -- **-0.5% off the ideal estimate**, i.e. the
finite plate size (width/separation ratio 6/3.2 ~= 1.9) and the grounded
box's own image-charge-like influence are both small, second-order
corrections at the exact center, consistent with plain electrostatics
intuition and not tuned to land there (:data:`PLATE_SEPARATION_M`,
:data:`PLATE_VOLTAGE_V`, and the box/plate geometry were fixed first from
"a physically sensible mm-scale two-electrode capacitor", and the small
resulting deviation is simply whatever the solve returns).

**Staircase-boundary accuracy caveat.** A 7-point finite-difference
Laplacian represents every Dirichlet boundary -- plate faces, plate edges,
box walls -- as a step function on the Cartesian grid: a node is either
"on the electrode" (fixed potential) or "off the electrode" (a free
unknown solved from its neighbors), with no sub-cell information about
where between two grid nodes the physical electrode edge actually sits.
Away from any edge (e.g. at this scenario's domain-center check point,
deep inside the plates' footprint) this costs nothing beyond the solve's
normal ``O(h^2)`` truncation error. Near an edge or corner, though, the
staircased boundary systematically under- or over-resolves the true field
concentration there (the field genuinely diverges at an idealized sharp
conducting edge; the discrete grid instead reports a large but finite,
grid-spacing-dependent value) -- exactly the region
``examples/fd_electrode_field.txt``'s exported sub-cube (see "Export"
below) is deliberately centered *away* from, but real FEA users pointing
this loader at their own near-electrode exports should keep in mind that
refining the mesh changes the near-edge answer, not just its precision.

**Export.** Only a small interior sub-cube of the full solve
(``EXPORT_POINTS_PER_AXIS`` = 15 points/axis, 3375 points total -- well
under :data:`cliffordclock.fields.smoother.MAX_FIT_POINTS`) is written to
``examples/fd_electrode_field.txt``: strictly between the two plates in z
(never touching an electrode's own Dirichlet-fixed potential, which is not
a meaningfully "smooth field" sample point) and centered in x/y, so every
exported point sits in the well-resolved interior region the back-of-
envelope check above describes, not the staircased edge region the
caveat above describes. Positions are written in **millimeters** (COMSOL's
common default -- and a genuine exercise of ``load_field_comsol``'s mm ->
m unit conversion, not meters copied straight through); field components
in **V/m**.

Determinism
------------

No randomness anywhere in this script: the finite-difference solve is a
deterministic linear solve of a fixed system from fixed geometry constants
(IEEE 754 double-precision arithmetic is deterministic given fixed inputs
and a fixed sequence of operations). Re-running this script reproduces
``examples/fd_electrode_field.txt`` byte-identically (values written via
``repr(float)``, Python's shortest round-trip representation).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]
import scipy.sparse.linalg as spla  # type: ignore[import-untyped]
from numpy.typing import NDArray

#: Finite-difference grid spacing, meters (module docstring "Solve").
H_M = 0.2e-3

#: Grounded-box interior dimensions, meters -- lateral (x, y) and depth (z).
BOX_XY_M = 10.0e-3
BOX_Z_M = 8.0e-3

#: Grid point counts per axis, derived from the box size and spacing.
NX = round(BOX_XY_M / H_M) + 1  # 51
NY = NX
NZ = round(BOX_Z_M / H_M) + 1  # 41

#: Electrode potential, volts (+/- this value on the two plates).
PLATE_VOLTAGE_V = 2.0

#: Electrode z grid-index positions (module docstring "Geometry") --
#: symmetric about the box's z-center with equal 2.4 mm margins to the
#: grounded top/bottom walls.
PLATE_Z_INDEX_LO = 12  # z = 2.4 mm, held at +PLATE_VOLTAGE_V
PLATE_Z_INDEX_HI = 28  # z = 5.6 mm, held at -PLATE_VOLTAGE_V

#: Plate separation, meters -- derived from the z-index spacing above.
PLATE_SEPARATION_M = (PLATE_Z_INDEX_HI - PLATE_Z_INDEX_LO) * H_M

#: Electrode x/y grid-index footprint (a square, centered in the box) --
#: 6 mm x 6 mm plates with 2 mm margins to the grounded side walls.
PLATE_XY_INDEX_LO = 10  # x, y = 2.0 mm
PLATE_XY_INDEX_HI = 40  # x, y = 8.0 mm

#: Exported sub-cube size, points per axis (module docstring "Export").
EXPORT_POINTS_PER_AXIS = 15

#: Exported sub-cube index bounds (inclusive), centered on the domain and
#: strictly interior to both the plate gap (z) and the plate footprint
#: (x, y) -- module docstring "Export". Centered on the true midpoint of
#: each range (not an endpoint-derived formula that could land on or past
#: an electrode's own Dirichlet-fixed node), with a symmetric half-span of
#: ``(EXPORT_POINTS_PER_AXIS - 1) // 2`` grid cells on each side.
_EXPORT_HALF_SPAN = (EXPORT_POINTS_PER_AXIS - 1) // 2
_Z_CENTER = (PLATE_Z_INDEX_LO + PLATE_Z_INDEX_HI) // 2
_EXPORT_Z_LO = _Z_CENTER - _EXPORT_HALF_SPAN
_EXPORT_Z_HI = _Z_CENTER + _EXPORT_HALF_SPAN
_XY_CENTER = (PLATE_XY_INDEX_LO + PLATE_XY_INDEX_HI) // 2
_EXPORT_XY_LO = _XY_CENTER - _EXPORT_HALF_SPAN
_EXPORT_XY_HI = _XY_CENTER + _EXPORT_HALF_SPAN

assert PLATE_Z_INDEX_LO < _EXPORT_Z_LO and _EXPORT_Z_HI < PLATE_Z_INDEX_HI, (
    "export sub-cube must sit strictly between the two electrode planes in z"
)

#: Default output path (relative to this file's directory).
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "fd_electrode_field.txt"

#: The six axis-neighbor offsets used by the 7-point Laplacian stencil.
_NEIGHBOR_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


def solve_potential() -> NDArray[np.float64]:
    """Solve Laplace's equation for the two-electrode grounded-box scenario.

    Returns
    -------
    NDArray[np.float64], shape (NX, NY, NZ)
        The potential ``V(x, y, z)``, volts, on the full solve grid.
    """
    ii, jj, kk = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")

    is_dirichlet = np.zeros((NX, NY, NZ), dtype=bool)
    dirichlet_val = np.zeros((NX, NY, NZ), dtype=np.float64)

    # Grounded outer box walls.
    wall = (ii == 0) | (ii == NX - 1) | (jj == 0) | (jj == NY - 1) | (kk == 0) | (kk == NZ - 1)
    is_dirichlet[wall] = True  # dirichlet_val already 0 there

    # Electrode plates.
    plate_xy = (
        (ii >= PLATE_XY_INDEX_LO)
        & (ii <= PLATE_XY_INDEX_HI)
        & (jj >= PLATE_XY_INDEX_LO)
        & (jj <= PLATE_XY_INDEX_HI)
    )
    plate_pos = plate_xy & (kk == PLATE_Z_INDEX_LO)
    plate_neg = plate_xy & (kk == PLATE_Z_INDEX_HI)
    is_dirichlet[plate_pos] = True
    dirichlet_val[plate_pos] = PLATE_VOLTAGE_V
    is_dirichlet[plate_neg] = True
    dirichlet_val[plate_neg] = -PLATE_VOLTAGE_V

    free_mask = ~is_dirichlet
    free_ids = np.flatnonzero(free_mask)  # indices into the raveled (NX*NY*NZ,) array
    n_free = free_ids.size
    free_rank = -np.ones(NX * NY * NZ, dtype=np.int64)
    free_rank[free_ids] = np.arange(n_free)

    is_dirichlet_flat = is_dirichlet.ravel()
    dirichlet_val_flat = dirichlet_val.ravel()
    ii_free, jj_free, kk_free = ii.ravel()[free_ids], jj.ravel()[free_ids], kk.ravel()[free_ids]

    # Assemble the sparse system A x = b over free nodes only (Dirichlet
    # neighbors move to the right-hand side): diagonal 6 (7-point stencil,
    # module docstring "Solve"), off-diagonal -1 per free neighbor. Each of
    # the 6 neighbor directions is handled as one vectorized pass over all
    # free nodes at once (free nodes are never on the outer wall by
    # construction, so every neighbor index below is always in-bounds).
    row_idx = [free_rank[free_ids]]
    col_idx = [free_rank[free_ids]]
    values = [np.full(n_free, 6.0)]
    b = np.zeros(n_free, dtype=np.float64)

    for di, dj, dk in _NEIGHBOR_OFFSETS:
        ni, nj, nk = ii_free + di, jj_free + dj, kk_free + dk
        neighbor_flat = (ni * NY + nj) * NZ + nk
        neighbor_is_dirichlet = is_dirichlet_flat[neighbor_flat]

        b_contrib_rank = free_rank[free_ids[neighbor_is_dirichlet]]
        np.add.at(b, b_contrib_rank, dirichlet_val_flat[neighbor_flat[neighbor_is_dirichlet]])

        free_neighbor_mask = ~neighbor_is_dirichlet
        row_idx.append(free_rank[free_ids[free_neighbor_mask]])
        col_idx.append(free_rank[neighbor_flat[free_neighbor_mask]])
        values.append(np.full(int(np.count_nonzero(free_neighbor_mask)), -1.0))

    a_matrix = sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(row_idx), np.concatenate(col_idx))),
        shape=(n_free, n_free),
    )

    # Jacobi (diagonal) preconditioner -- the matrix diagonal is a uniform
    # 6, so this is a cheap constant rescaling that still meaningfully
    # accelerates CG convergence on the discrete Laplacian's spread-out
    # eigenvalue spectrum.
    diag = a_matrix.diagonal()
    preconditioner = spla.LinearOperator(a_matrix.shape, matvec=lambda x: x / diag)
    x, info = spla.cg(a_matrix, b, rtol=1e-10, maxiter=20_000, M=preconditioner)
    if info != 0:
        raise RuntimeError(f"conjugate-gradient solve did not converge (info={info})")

    v_flat = np.zeros(NX * NY * NZ, dtype=np.float64)
    v_flat[free_ids] = x
    v_flat[is_dirichlet.ravel()] = dirichlet_val_flat[is_dirichlet.ravel()]
    return v_flat.reshape(NX, NY, NZ)


def field_from_potential(
    v_grid: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``E = -grad(V)`` on the full solve grid via central differences.

    Returns
    -------
    tuple of three NDArray[np.float64], each shape (NX, NY, NZ)
        ``(Ex, Ey, Ez)``, V/m.
    """
    grad_x, grad_y, grad_z = np.gradient(v_grid, H_M, edge_order=2)
    return (
        np.asarray(-grad_x, dtype=np.float64),
        np.asarray(-grad_y, dtype=np.float64),
        np.asarray(-grad_z, dtype=np.float64),
    )


def export_subcube(
    v_grid: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract the interior export sub-cube's positions (m) and field (V/m).

    Returns
    -------
    positions_m : NDArray[np.float64], shape (EXPORT_POINTS_PER_AXIS**3, 3)
    field_v_per_m : NDArray[np.float64], shape (EXPORT_POINTS_PER_AXIS**3, 3)
    """
    ex, ey, ez = field_from_potential(v_grid)

    i_sl = slice(_EXPORT_XY_LO, _EXPORT_XY_HI + 1)
    j_sl = slice(_EXPORT_XY_LO, _EXPORT_XY_HI + 1)
    k_sl = slice(_EXPORT_Z_LO, _EXPORT_Z_HI + 1)

    x_axis = np.arange(_EXPORT_XY_LO, _EXPORT_XY_HI + 1) * H_M
    y_axis = x_axis.copy()
    z_axis = np.arange(_EXPORT_Z_LO, _EXPORT_Z_HI + 1) * H_M
    xg, yg, zg = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    positions_m = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=-1)

    field_v_per_m = np.stack(
        [ex[i_sl, j_sl, k_sl].ravel(), ey[i_sl, j_sl, k_sl].ravel(), ez[i_sl, j_sl, k_sl].ravel()],
        axis=-1,
    )
    return positions_m, field_v_per_m


def write_comsol_spreadsheet(
    path: Path,
    positions_m: NDArray[np.float64],
    field_v_per_m: NDArray[np.float64],
) -> None:
    """Write ``positions_m``/``field_v_per_m`` in COMSOL's "Spreadsheet" export format.

    An independent writer from :func:`cliffordclock.fields.io.load_field_comsol`'s
    parser -- this is what makes the round trip in
    ``tests/test_fields_comsol.py``'s end-to-end test a genuine check of the
    loader, not a fixture hand-massaged to fit it. Positions are converted
    to **millimeters** (``docs/byof-guide.md``'s "COMSOL exports" section:
    a common real-world COMSOL default, and a genuine exercise of the
    loader's unit conversion) and written with the same header grammar
    ``docs/fields.md`` documents: a ``%``-prefixed metadata block, a final
    ``%``-prefixed column-header line, then whitespace-separated data rows.
    Each float is written via ``repr``, Python's shortest round-trip
    representation (module docstring "Determinism").
    """
    n = positions_m.shape[0]
    positions_mm = positions_m * 1.0e3
    header_lines = [
        "% Model:       generate_fd_electrode_field.py (synthetic, not a real COMSOL export)",
        "% Version:     n/a",
        "% Date:        generated by cliffordclock/examples",
        "% Dimension:   3",
        f"% Nodes:       {n}",
        "% Expressions: 3",
        "% Description: Finite-difference Laplace solve, two-electrode grounded box",
        "% Length unit: mm",
        "% X    Y    Z    es.Ex (V/m)   es.Ey (V/m)   es.Ez (V/m)",
    ]
    with path.open("w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        for pos, e in zip(positions_mm, field_v_per_m, strict=True):
            row = [repr(float(v)) for v in (*pos, *e)]
            f.write(" ".join(row) + "\n")


def main(output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Solve, export, and print the back-of-envelope parallel-plate cross-check."""
    v_grid = solve_potential()
    positions_m, field_v_per_m = export_subcube(v_grid)
    write_comsol_spreadsheet(output_path, positions_m, field_v_per_m)

    # Domain-center check point (module docstring "Back-of-envelope check").
    ic = (PLATE_XY_INDEX_LO + PLATE_XY_INDEX_HI) // 2
    kc = (PLATE_Z_INDEX_LO + PLATE_Z_INDEX_HI) // 2
    _, _, ez_full = field_from_potential(v_grid)
    e_center = float(ez_full[ic, ic, kc])
    e_ideal = 2.0 * PLATE_VOLTAGE_V / PLATE_SEPARATION_M
    deviation_pct = 100.0 * (e_center - e_ideal) / e_ideal

    print(f"Wrote {positions_m.shape[0]} points to {output_path}")
    print(f"Solve grid: {NX} x {NY} x {NZ} = {NX * NY * NZ} nodes, spacing {H_M * 1e3:.2f} mm")
    print(f"Plate separation: {PLATE_SEPARATION_M * 1e3:.2f} mm, +/-{PLATE_VOLTAGE_V} V")
    print(f"E_z(domain center) = {e_center:.4f} V/m (solved)")
    print(f"E_ideal = 2V/d      = {e_ideal:.4f} V/m (parallel-plate estimate)")
    print(f"Deviation: {deviation_pct:+.3f}%")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"output COMSOL-spreadsheet-format path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    main(output_path=_args.output)
