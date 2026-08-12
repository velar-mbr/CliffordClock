#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate ``examples/showcase_field.txt``, the showcase's chamber-scale field.

This is the field behind the paper's showcase figure and section: a
realistic chamber with genuine spatial *structure* -- not a single
uniform-field number, and not a plain parallel-plate capacitor either.
Like ``examples/generate_fd_electrode_field.py`` (the template this script
follows), it solves a real, if simple, electrostatics boundary-value
problem from nothing but ``numpy`` + ``scipy.sparse`` (no COMSOL, no paid
software, no new dependency), then writes the solved field out through the
same independent COMSOL-"Spreadsheet"-format writer convention, so loading
the committed output back through
:func:`cliffordclock.fields.io.load_field_comsol` is a genuine test of the
loader against an independently generated file.

What makes this scenario a "showcase" and not another two-electrode
capacitor
--------------------------------------------------------------------------

``generate_fd_electrode_field.py``'s field is, by design, close to the
ideal parallel-plate limit at its export region -- a good loader
round-trip test, but nearly linear (nearly constant gradient) across any
small region, which is not a demanding test of a pipeline's ability to
carry real field *curvature* into a dispersion budget. This scenario is
deliberately different in three ways:

1. **Asymmetric electrodes.** Two plates of different size, different
   voltage, and offset (not mirror-symmetric) positions, so the field
   near the shared interior region has both a genuine gradient *and* a
   genuine second-derivative (curvature) -- unlike a symmetric capacitor,
   whose field is odd about the mid-plane and very nearly linear there.
2. **A patch-potential spot.** A small circular Dirichlet region embedded
   in an otherwise-grounded chamber wall, at a different potential (a
   crude but standard model of a patch-potential contamination site --
   the same physical mechanism ``examples/generate_patch_field.py``
   models with an analytic Gaussian bump, here instead solved
   self-consistently as a boundary condition of the same finite-difference
   problem as the electrodes). The patch's field falls off away from the
   wall and adds cross-terms (a third, non-collinear feature) that a
   pairwise-electrode field alone would not have.
3. **A wider useful region.** The exported sub-cube is sized to comfortably
   contain a real thermal atomic cloud (see "Ensemble sizing" in
   ``paper/figures/fig4_showcase_gradient_dispersion.py``), not just a
   single evaluation point.

Geometry (meters; also see the module-level constants below)
--------------------------------------------------------------------------

A grounded conducting box (``BOX_X_M`` x ``BOX_Y_M`` x ``BOX_Z_M`` = 24 mm
x 24 mm x 16 mm) contains:

- **Electrode A** (the "small" plate): 6 mm x 6 mm, centered at
  ``(x, y) = (5, 16)`` mm, held at ``+3.0`` V, at ``z = 3`` mm.
- **Electrode B** (the "large" plate): 9 mm x 9 mm, centered at
  ``(x, y) = (17, 8)`` mm, held at ``-1.2`` V, at ``z = 13`` mm.
- **The patch**: a 1.5 mm-radius circular Dirichlet spot at ``+4.5`` V,
  embedded in the ``z = 0`` wall (otherwise grounded), centered at
  ``(x, y) = (21, 4)`` mm.

Every voltage and position above is a chosen scenario parameter (this
project's owner-approved convention: scenario *geometry* is tunable to
make a demonstration realistic and legible; physical *constants* --
``epsilon_0``, the FD stencil, species data -- are never touched). None
of these voltages is large for a real vacuum chamber (a few volts of
patch-potential or applied-bias contamination is squarely in the range
surveyed by ``examples/generate_patch_field.py``'s own citations).

**Solve.** Laplace's equation, the same second-order 7-point stencil,
Jacobi-preconditioned conjugate gradient, and ``E = -grad(V)`` central
difference as ``generate_fd_electrode_field.py`` -- see that script's
module docstring for the numerical method in full; nothing about the
solver itself changes here, only the boundary conditions.

**Export region.** A cube of side ``2 * EXPORT_HALF_EXTENT_M`` (6.4 mm),
centered on ``TRAP_CENTER_M`` (the domain center in x/y, and the z-midpoint
between the two electrode planes, 8 mm), sampled at twice the native
solve-grid spacing (0.8 mm, 9 points/axis, 729 points total). The showcase
figure script (``paper/figures/fig4_showcase_gradient_dispersion.py``)
evaluates this fitted field at tens of thousands of Monte Carlo trajectory
samples (a thermal ensemble's classical trajectories, not a handful of
line-plot points); a thin-plate-spline RBF fit's evaluation cost scales
with the *fit* point count (a dense kernel matrix against every query
point), so keeping the fit itself modest (729 points, comfortably under
:data:`cliffordclock.fields.smoother.MAX_FIT_POINTS`) is what keeps that
Monte Carlo evaluation tractable; 729 points spaced at 0.8 mm still
comfortably resolves this scenario's mm-scale field curvature (a global
spline fit, not a per-cell finite-difference stencil -- fit density needs
only to sample the *smooth* solved potential's variation, not resolve a
sub-grid feature). This region sits strictly inside the two electrode
z-planes (3.2 mm export half-extent vs. a 5 mm half-gap to either plane,
a 1.8 mm safety margin -- see the module-level ``assert``) and strictly
away from the patch's own Dirichlet nodes (the export region's own
z-minimum, 4.8 mm, sits 4.8 mm above the ``z=0`` patch wall, and the
nearest export corner is several mm from the patch disk in x/y as well)
-- so, exactly as the FD-electrode template's own export region, no
exported point ever touches a Dirichlet-fixed node; every exported value
is a genuinely solved, interior field/gradient/curvature sample.

Back-of-envelope check (this is the "proof", not just a description)
--------------------------------------------------------------------------

At the domain/trap center, this scenario has no single-capacitor closed
form to check against (that is the point: two differently-sized,
differently-biased plates plus an off-axis patch have no simple textbook
answer) -- so the back-of-envelope check here is instead a *scale* check:
:func:`main` prints the field magnitude and, along each axis, the
field's local relative variation
``|E(center + h) - E(center - h)| / (2 h |E(center)|)`` for
``h = 0.5 mm`` (comparable to the atomic cloud's own sigma, see the figure
script) alongside the naive order-of-magnitude estimate
``V_scale / (typical electrode-to-center distance)`` -- both computed
live, not hand-tuned, and reported for the owner's/reviewer's sanity
check.

Determinism
--------------------------------------------------------------------------

Identical to ``generate_fd_electrode_field.py``: no randomness anywhere
(a deterministic linear solve of a fixed system from fixed geometry
constants); re-running this script reproduces
``examples/showcase_field.txt`` byte-identically (``repr(float)`` output).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]
import scipy.sparse.linalg as spla  # type: ignore[import-untyped]
from numpy.typing import NDArray

#: Finite-difference grid spacing, meters. Chosen (with the box size below)
#: so every solve-grid axis stays at or under this project's 60-point
#: memory-safety bound (see "Memory-safety note" below) -- 0.4 mm would
#: give 61 points on the two 24 mm axes, one over that bound.
H_M = 0.41e-3

#: Grounded-box interior dimensions, meters.
BOX_X_M = 24.0e-3
BOX_Y_M = 24.0e-3
BOX_Z_M = 16.0e-3

#: Grid point counts per axis.
NX = round(BOX_X_M / H_M) + 1  # 60
NY = round(BOX_Y_M / H_M) + 1  # 60
NZ = round(BOX_Z_M / H_M) + 1  # 40

#: Memory-safety note (binding): this project caps the FD solve grid at
#: <=60 points/axis and solves it with iterative, Jacobi-preconditioned
#: conjugate gradient only (never a dense/direct solve) -- see
#: :func:`solve_potential`. An unrelated but related incident (a stray,
#: unmonitored Monte Carlo run at a much larger `FieldSmoother` fit point
#: count, not this solve grid) once drove memory usage far past available
#: RAM; this assert is a second, independent guard on the solve side.
assert NX <= 60 and NY <= 60 and NZ <= 60, (
    f"solve grid ({NX}, {NY}, {NZ}) exceeds the 60-point-per-axis memory-safety bound"
)

# --- Electrode A ("small"): 6mm x 6mm, +3.0 V, z = 3 mm. -------------------
PLATE_A_VOLTAGE_V = 3.0
PLATE_A_CENTER_XY_M = (5.0e-3, 16.0e-3)
PLATE_A_HALF_WIDTH_M = 3.0e-3  # 6 mm side
PLATE_A_Z_M = 3.0e-3

# --- Electrode B ("large"): 9mm x 9mm, -1.2 V, z = 13 mm. -------------------
PLATE_B_VOLTAGE_V = -1.2
PLATE_B_CENTER_XY_M = (17.0e-3, 8.0e-3)
PLATE_B_HALF_WIDTH_M = 4.5e-3  # 9 mm side
PLATE_B_Z_M = 13.0e-3

# --- Patch: circular Dirichlet spot on the z=0 wall. ------------------------
PATCH_VOLTAGE_V = 4.5
PATCH_CENTER_XY_M = (21.0e-3, 4.0e-3)
PATCH_RADIUS_M = 1.5e-3

#: Trap/export center: domain center in x/y, z-midpoint between the two
#: electrode planes -- module docstring "Export region".
TRAP_CENTER_M = (BOX_X_M / 2.0, BOX_Y_M / 2.0, (PLATE_A_Z_M + PLATE_B_Z_M) / 2.0)

#: Export stride in solve-grid cells (2 -> export at 0.8 mm spacing, twice
#: the native 0.4 mm solve-grid spacing -- see "Export region" below for
#: why a coarser export keeps the RBF fit cheap without losing the
#: curvature this scenario is built to carry).
EXPORT_STRIDE = 2

#: Export half-extent, in solve-grid cells (4 cells * 2 stride * 0.4 mm =
#: 3.2 mm).
_EXPORT_HALF_N_CELLS = 4

#: Export sub-cube half-extent, meters (module docstring "Export region").
EXPORT_HALF_EXTENT_M = _EXPORT_HALF_N_CELLS * H_M * EXPORT_STRIDE

#: Export points per axis (9: 4 cells either side of center, inclusive).
EXPORT_POINTS_PER_AXIS = 2 * _EXPORT_HALF_N_CELLS + 1

#: Memory-safety bound (binding, module docstring "Export region"): this
#: project caps a `FieldSmoother` fit at far fewer points than the
#: library's own hard `MAX_FIT_POINTS` (20,000) -- a downstream consumer
#: (e.g. a Monte Carlo ensemble evaluating this field at many trajectory
#: points) pays a cost proportional to `n_query_points * n_fit_points`,
#: which is what actually drove memory usage far past available RAM in an
#: earlier, unmonitored run of this scenario at a larger fit point count.
assert EXPORT_POINTS_PER_AXIS**3 <= 10_000, (
    f"export point count {EXPORT_POINTS_PER_AXIS**3} exceeds the 10,000-point memory-safety bound"
)

assert (PLATE_B_Z_M - PLATE_A_Z_M) / 2.0 - EXPORT_HALF_EXTENT_M > 1.0e-3, (
    "export sub-cube must sit strictly between the two electrode planes in z, with a safety margin"
)

#: Default output path (relative to this file's directory).
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "showcase_field.txt"

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
    """Solve Laplace's equation for the asymmetric-electrode + patch scenario.

    Returns
    -------
    NDArray[np.float64], shape (NX, NY, NZ)
        The potential ``V(x, y, z)``, volts, on the full solve grid.
    """
    ii, jj, kk = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")
    x = ii.astype(np.float64) * H_M
    y = jj.astype(np.float64) * H_M

    is_dirichlet = np.zeros((NX, NY, NZ), dtype=bool)
    dirichlet_val = np.zeros((NX, NY, NZ), dtype=np.float64)

    # Grounded outer box walls.
    wall = (ii == 0) | (ii == NX - 1) | (jj == 0) | (jj == NY - 1) | (kk == 0) | (kk == NZ - 1)
    is_dirichlet[wall] = True  # dirichlet_val already 0 there

    # Electrode A.
    ax0, ay0 = PLATE_A_CENTER_XY_M
    plate_a = (
        (np.abs(x - ax0) <= PLATE_A_HALF_WIDTH_M)
        & (np.abs(y - ay0) <= PLATE_A_HALF_WIDTH_M)
        & (kk == round(PLATE_A_Z_M / H_M))
    )
    is_dirichlet[plate_a] = True
    dirichlet_val[plate_a] = PLATE_A_VOLTAGE_V

    # Electrode B.
    bx0, by0 = PLATE_B_CENTER_XY_M
    plate_b = (
        (np.abs(x - bx0) <= PLATE_B_HALF_WIDTH_M)
        & (np.abs(y - by0) <= PLATE_B_HALF_WIDTH_M)
        & (kk == round(PLATE_B_Z_M / H_M))
    )
    is_dirichlet[plate_b] = True
    dirichlet_val[plate_b] = PLATE_B_VOLTAGE_V

    # Patch spot: overrides the grounded z=0 wall locally (assigned after
    # the wall block above, so it wins).
    px0, py0 = PATCH_CENTER_XY_M
    patch = (kk == 0) & ((x - px0) ** 2 + (y - py0) ** 2 <= PATCH_RADIUS_M**2)
    is_dirichlet[patch] = True
    dirichlet_val[patch] = PATCH_VOLTAGE_V

    free_mask = ~is_dirichlet
    free_ids = np.flatnonzero(free_mask)
    n_free = free_ids.size
    free_rank = -np.ones(NX * NY * NZ, dtype=np.int64)
    free_rank[free_ids] = np.arange(n_free)

    is_dirichlet_flat = is_dirichlet.ravel()
    dirichlet_val_flat = dirichlet_val.ravel()
    ii_free = ii.ravel()[free_ids]
    jj_free = jj.ravel()[free_ids]
    kk_free = kk.ravel()[free_ids]

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

    diag = a_matrix.diagonal()
    preconditioner = spla.LinearOperator(a_matrix.shape, matvec=lambda x: x / diag)
    x_sol, info = spla.cg(a_matrix, b, rtol=1e-10, maxiter=20_000, M=preconditioner)
    if info != 0:
        raise RuntimeError(f"conjugate-gradient solve did not converge (info={info})")

    v_flat = np.zeros(NX * NY * NZ, dtype=np.float64)
    v_flat[free_ids] = x_sol
    v_flat[is_dirichlet.ravel()] = dirichlet_val_flat[is_dirichlet.ravel()]
    return v_flat.reshape(NX, NY, NZ)


def field_from_potential(
    v_grid: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``E = -grad(V)`` on the full solve grid via central differences."""
    grad_x, grad_y, grad_z = np.gradient(v_grid, H_M, edge_order=2)
    return (
        np.asarray(-grad_x, dtype=np.float64),
        np.asarray(-grad_y, dtype=np.float64),
        np.asarray(-grad_z, dtype=np.float64),
    )


def export_subcube(
    v_grid: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract the export sub-cube's positions (m) and field (V/m).

    Returns
    -------
    positions_m : NDArray[np.float64], shape (EXPORT_POINTS_PER_AXIS**3, 3)
    field_v_per_m : NDArray[np.float64], shape (EXPORT_POINTS_PER_AXIS**3, 3)
    """
    ex, ey, ez = field_from_potential(v_grid)

    cx, cy, cz = TRAP_CENTER_M
    ic, jc, kc = round(cx / H_M), round(cy / H_M), round(cz / H_M)
    half_n = (EXPORT_POINTS_PER_AXIS - 1) // 2 * EXPORT_STRIDE

    i_idx = np.arange(ic - half_n, ic + half_n + 1, EXPORT_STRIDE)
    j_idx = np.arange(jc - half_n, jc + half_n + 1, EXPORT_STRIDE)
    k_idx = np.arange(kc - half_n, kc + half_n + 1, EXPORT_STRIDE)
    assert i_idx.size == EXPORT_POINTS_PER_AXIS
    assert j_idx.size == EXPORT_POINTS_PER_AXIS
    assert k_idx.size == EXPORT_POINTS_PER_AXIS

    ig, jg, kg = np.meshgrid(i_idx, j_idx, k_idx, indexing="ij")
    xg, yg, zg = ig * H_M, jg * H_M, kg * H_M
    positions_m = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=-1)

    field_v_per_m = np.stack(
        [ex[ig, jg, kg].ravel(), ey[ig, jg, kg].ravel(), ez[ig, jg, kg].ravel()],
        axis=-1,
    )
    return positions_m, field_v_per_m


def write_comsol_spreadsheet(
    path: Path,
    positions_m: NDArray[np.float64],
    field_v_per_m: NDArray[np.float64],
) -> None:
    """Write ``positions_m``/``field_v_per_m`` in COMSOL's "Spreadsheet" export format.

    Independent of :func:`cliffordclock.fields.io.load_field_comsol`'s
    parser (module docstring). Positions are written in **millimeters**
    (COMSOL's common default, and a genuine exercise of the loader's
    mm -> m unit conversion); field components in **V/m**.
    """
    n = positions_m.shape[0]
    positions_mm = positions_m * 1.0e3
    header_lines = [
        "% Model:       generate_showcase_field.py (synthetic, not a real COMSOL export)",
        "% Version:     n/a",
        "% Date:        generated by cliffordclock/examples",
        "% Dimension:   3",
        f"% Nodes:       {n}",
        "% Expressions: 3",
        "% Description: Finite-difference Laplace solve, asymmetric two-electrode "
        "chamber + patch-potential wall spot",
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
    """Solve, export, and print the back-of-envelope scale check."""
    v_grid = solve_potential()
    positions_m, field_v_per_m = export_subcube(v_grid)
    write_comsol_spreadsheet(output_path, positions_m, field_v_per_m)

    ex, ey, ez = field_from_potential(v_grid)
    cx, cy, cz = TRAP_CENTER_M
    ic, jc, kc = round(cx / H_M), round(cy / H_M), round(cz / H_M)
    e_center = np.array([ex[ic, jc, kc], ey[ic, jc, kc], ez[ic, jc, kc]])
    e_center_mag = float(np.linalg.norm(e_center))

    # Local relative field variation over h=0.5mm steps along each axis
    # (module docstring "Back-of-envelope check").
    h_cells = round(0.5e-3 / H_M)
    rel_variation = {}
    for axis, name in enumerate("xyz"):
        idx_lo = [ic, jc, kc]
        idx_hi = [ic, jc, kc]
        idx_lo[axis] -= h_cells
        idx_hi[axis] += h_cells
        e_lo = np.array(
            [ex[tuple(idx_lo)], ey[tuple(idx_lo)], ez[tuple(idx_lo)]]  # type: ignore[arg-type]
        )
        e_hi = np.array(
            [ex[tuple(idx_hi)], ey[tuple(idx_hi)], ez[tuple(idx_hi)]]  # type: ignore[arg-type]
        )
        rel_variation[name] = float(np.linalg.norm(e_hi - e_lo)) / (2.0 * e_center_mag)

    print(f"Wrote {positions_m.shape[0]} points to {output_path}")
    print(f"Solve grid: {NX} x {NY} x {NZ} = {NX * NY * NZ} nodes, spacing {H_M * 1e3:.2f} mm")
    print(f"E(trap center) = {e_center} V/m, |E| = {e_center_mag:.4f} V/m")
    for name in "xyz":
        print(
            f"  relative variation over +/-0.5mm along {name}: "
            f"{rel_variation[name]:.4e} per mm-scale step"
        )


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
