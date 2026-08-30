#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate ``examples/patch_field_sr87.csv`` (the "bring your own field" worked example).

Produces a physically plausible optical-lattice-clock stray-field scenario:
a small uniform residual bias field (representing incomplete DC-field
nulling, a routine lab procedure) plus six Gaussian **patch-potential**
contributions from residual trapped charge on in-vacuum dielectric surfaces
(mirror/viewport faces) at a cm-scale standoff around the trap, evaluated
on a regular grid near the trap center and written in the standard CSV
contract (``docs/fields.md``: header ``x,y,z,Ex,Ey,Ez``, meters / V/m).

This is the CSV a lab postdoc's own COMSOL/FEA export would resemble --
the demo `examples/realistic_lattice_sr87.yaml` loads exactly this file
through the same `field: {csv: ...}` path a real export would use. See
`docs/byof-guide.md` for the full "bring your own field" adaptation guide.

Physics of the scenario
------------------------

**Reframed 2026-08-10** (following a literature review): this scenario
originally modeled bare-metal "patch potentials" (grain-
boundary work-function variation on conductor surfaces) at a 2 mm, mm-
scale standoff. A reviewer flagged that no documented neutral-atom
lattice clock has ordinary chamber surfaces that close -- 2 mm is
ion-trap/calibration-fixture territory (Brownnutt et al., Rev. Mod. Phys.
87, 1419 (2015)). The dossier's literature survey found the real,
documented event class is different: **trapped charge accumulating on
in-vacuum dielectric surfaces** (cavity mirrors, viewports) at **cm-scale**
distances from the atoms. This module now models that scenario instead --
same field/shift order of magnitude (already physically conservative, see
below), reframed geometry and citations.

- **Lodewyck, Zawada, Lorini, Gurov, Lemonde**, IEEE Trans. UFFC 59, 411
  (2012), arXiv:1108.4320 -- the primary citation: the canonical stray-
  field event in a real Sr lattice clock (LNE-SYRTE). Charges trapped on
  **dielectric build-up-cavity mirrors "a few centimeters from the
  atoms"** produced a field of **3.4 kV/m** at the atoms and a
  **1e-13-level** fractional clock shift, pre-mitigation; a UV-light
  discharge procedure reduced this to the **1e-18 level**, with residual
  sensitivity calibrated via an external electrode (1 kV applied -> 26 Hz
  shift). This module models the intermediate, **partially-discharged**
  regime between those two endpoints: volt-scale (not kV-scale) residual
  potentials on dielectric surfaces after an incomplete/early-stage UV
  discharge cycle -- exactly the "systematic you must characterize and
  correct" a BYOF-style tool exists for, not either extreme.
- **Beloy et al.** (NIST Yb), PRL 120, 183201 (2018), arXiv:1803.10737 --
  secondary context: an in-vacuum Faraday shield built specifically
  because "DC Stark shifts... observed as large as 1e-13" in the field;
  shield windows sit ~50 mm from the atoms (100 mm pair separation),
  confirming the cm-scale (not mm-scale) standoff for real charge-bearing
  surfaces near lattice-clock atoms.
- **NPL Sr** (arXiv:2005.10857) -- secondary context: background field
  budgeted via Rydberg spectroscopy in a chamber where "the large steel
  chamber with comparatively small viewports proves to be quite an
  effective shield" -- again cm-scale-or-larger surfaces, DC-Stark line
  constrained to -1.6(16)e-20.
- **Camp, Darling, Brown**, J. Appl. Phys. 69, 7126 (1991) -- kept only as
  *general background* on patch-potential magnitudes existing at all
  (Kelvin-probe survey of 7 conductors; alloys are directionally least
  uniform, gold/graphite stayed at their ~1 mV probe floor). The
  dossier's fetched-source check could **not** re-verify the specific
  "10-100 mV" figure this module previously attributed to this paper at
  that precision, so it is no longer used as the quantitative source for
  any parameter here -- this module's volt-scale trapped-charge
  potentials are Lodewyck-anchored (above), not Camp-derived.

This generator draws six patch amplitudes from Uniform(0.5, 5.0) V (random
sign) -- see "Geometry" below for how this range was chosen relative to
Lodewyck's two documented endpoints, and "Tuning reasoning" for why the
*width* parameter, not the amplitude range, absorbed the adjustment needed
to land the target shift (CONVENTIONS.md-style: parameters come from cited
physics or bounded, documented geometry tuning, never adjusted post hoc to
hit a target number by changing a physics constant).

**Geometry.** Six patches sit at `PATCH_DISTANCE_M` = 25 mm from the trap
center, one per +/-x/y/z direction -- modeling six in-vacuum dielectric
surfaces (e.g. build-up-cavity mirror faces and vacuum-viewport faces)
surrounding the trap at the cm-scale standoff Lodewyck/Beloy/NPL document,
squarely inside the WP-specified 20-30 mm band. Each patch is modeled as a
Gaussian potential bump, `V_patch(r) = A * exp(-|r - r_patch|^2 / (2 w^2))`
with `w = PATCH_WIDTH_M = 8 mm`, giving a field contribution
`E_patch(r) = -grad V_patch(r) = (A / w^2) (r - r_patch) exp(-|r -
r_patch|^2 / (2 w^2))`. The total field is the uniform bias plus the sum
of all six patch contributions -- superposition, since each patch's
potential is an independent (linear) source term.

**Model idealization:** each "patch" here is an
*isotropic 3D Gaussian potential bump* centered on the surface location
-- a deliberately simple, smooth, closed-form idealization -- NOT an
oriented 2D charged-surface element, which is what a charged mirror face
physically is and which would have a direction-dependent (surface-normal)
near/far-field falloff. The demo's purpose is a realistic field
*magnitude and spatial scale* at the atoms with a clean analytic form the
determinism test can pin, not a boundary-element electrostatics solve;
users bringing real FEA exports get the real geometry's field by
construction.

**Tuning reasoning (documented per dossier's "never tune physics
constants" rule).** This Gaussian-bump functional form's field falls off
as `exp(-d^2 / 2w^2)` between the patch center and the evaluation point,
so once the standoff `d` grew from the old 2 mm to the cm-scale 20-30 mm
band, a literal "few mm" width (the loose guidance that motivated this
WP) makes `d/w` too large: e.g. `d=25mm, w=3mm` gives
`exp(-625/18) = exp(-34.7) ~ 8e-16`, suppressing the field to nothing
for any amplitude in the cited volt range.
The width that keeps `d/w` at the same order as the original,
working scenario (`d/w = 2` there) is `w ~ 8-12 mm`, i.e. a *patch that
covers roughly a third to a half of its own standoff distance* -- still a
localized region on a several-cm optic, not a "few mm" dot, but well
short of "the whole mirror is one patch." This is not a stretch of the
cited physics: Lodewyck describes charge distributed over the mirror
surface itself (cm-scale), so a patch footprint of order 1 cm on a
cm-scale optic a few cm from the atoms is the literature-consistent
choice, not the "few mm" figure, which was only ever early rough
verbal shorthand for "small compared to the standoff." `PATCH_WIDTH_M`
(8 mm) is therefore the one geometry parameter tuned beyond the initial
guidance, and it is tuned to keep the functional form usable at the new
distance, not to hit the target shift number directly -- the amplitude
range (0.5-5 V, Lodewyck-bracketed, see below) and distance (25 mm,
literature-typical) were fixed first from the citations, and only then
was the width chosen so the field would not vanish; the resulting shift
landing in-range is a consequence, not a fitted outcome (see "Back-of-
envelope check" below for the actual numbers, which land near the
geometric center of the required [1e-19, 1e-17] demonstration range
(docs/byof-guide.md) without further adjustment).

**Amplitude range vs. Lodewyck's endpoints.** Lodewyck's paper reports
field and shift, not the mirror's surface potential directly, so there is
no single citable "the potential was X volts" figure to draw from
verbatim. What is documented is the *bracket*: kV-scale-equivalent
charging pre-mitigation (3.4 kV/m at cm range) reduced to a field so
small it is unmeasured at the 1e-18 shift level post-UV-discharge. A
"partially discharged" intermediate state -- meaningfully cleaned up but
not fully neutralized -- sitting at volt-scale local potentials (0.5-5 V)
on a cm-scale dielectric surface a few cm away is the natural order-of-
magnitude midpoint consistent with that bracket, and is the range this
module draws from.

**Grid.** The exported grid spans a small region around the trap center
(`GRID_HALF_EXTENT_M = 250 um`, `GRID_POINTS_PER_AXIS = 17`, a regular
17x17x17 = 4913-point grid, spacing ~31 um) -- vastly larger than an
Sr-87 lattice ground-motional-state extent (~61 nm at a typical
2e5 rad/s trap frequency, `docs/timescales.md`), so every quadrature node
the pipeline queries sits deep inside the fit's bounding box, and vastly
smaller than the patches' 8 mm width (~250x oversampled), so the field
varies smoothly (near-linear-plus-curvature) across the exported domain --
exactly the regime `cliffordclock.fields.FieldSmoother`'s degree-1
baseline + RBF residual split (CONVENTIONS.md E11-E12) is designed for.
This mirrors what a real FEA export would look like: a fine mesh sampled
only in the small region of physical interest around the trap, not the
whole chamber.

Back-of-envelope check (docs/byof-guide.md demonstration-range contract)
--------------------------------------------------------------------------

At the trap center this scenario's total field is (seed
`SEED`, computed by :func:`main` below) approximately::

    |E(0)| ~= 10.4 V/m

The Sr-87 DC-Stark coefficient (CONVENTIONS.md E14b,
`cliffordclock.ensemble.species.SR87`, cross-checked against
`docs/validation.md` KA1's quoted "100 V/m -> Delta_nu/nu0 =
-7.170524e-17") gives `k_S / nu0 = 7.170524e-21` per (V/m)^2, so::

    Delta_nu/nu0 ~= -(k_S/nu0) * |E(0)|^2 ~= -(7.170524e-21) * 10.4^2
                 ~= -7.7e-19

-- squarely inside the required [1e-19, 1e-17] demonstration range
(close to its geometric center, `sqrt(1e-19 * 1e-17) = 1e-18`), matching
what the full pipeline reports (`examples/realistic_lattice_sr87.yaml`,
`docs/byof-guide.md`). One-line anchor against the two documented real
endpoints: this scenario's field (~10.4 V/m) is ~330x below Lodewyck's
unmitigated event (3.4 kV/m, 1e-13 shift) and its shift (~7.7e-19) is
~80x above a modern mitigated budget (~1e-20, Beloy/NPL) -- i.e. the
"partially discharged, still needs correcting" middle ground the demo is
meant to represent. This scenario's *geometry* (patch distance/width) was
tuned to land in this range starting from the Lodewyck-bracketed
amplitude range and cm-scale distance -- never by adjusting the physics
constants (`ALPHA_AU_TO_SI`, the species registry's `Delta_alpha`, or the
E14b formula itself).

Determinism (pinned by the determinism test)
-----------------------------------------------

All randomness (`AMPLITUDES_V`'s draw) comes from a single seeded
`numpy.random.default_rng(SEED)` call in a fixed order; every other
quantity is a closed-form function of fixed constants. Re-running this
script reproduces `examples/patch_field_sr87.csv` byte-identically (same
NumPy/Python floating-point semantics -- IEEE 754 double-precision
arithmetic is deterministic given fixed inputs and operation order; each
CSV field is written via `repr(float)`, Python's shortest round-trip
representation, so the same float always prints the same digits). The
seed (`SEED = 20260810`) is unchanged from the original scenario -- this
reframe changed geometry/amplitude-range constants, not the PRNG call
site or order, so the same seed still applies deterministically to the
(now differently-scaled) draw.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

#: PRNG seed for the patch-amplitude draw (module docstring "Determinism").
#: Arbitrary but fixed -- chosen as this WP's authoring date (2026-08-10),
#: not tuned.
SEED = 20260810

#: Small uniform residual bias field, V/m -- representing typical
#: incomplete DC stray-field nulling after a lab's compensation-electrode
#: procedure, on top of the dielectric-surface trapped charge (deliberately
#: modest: this scenario's shift comes mostly from the patch-potential
#: *gradient* structure, not a large uniform bias).
BIAS_FIELD_V_PER_M: tuple[float, float, float] = (0.3, -0.2, 0.5)

#: Number of patch sources: one per in-vacuum dielectric surface
#: surrounding the trap (+/-x, +/-y, +/-z) -- e.g. build-up-cavity mirror
#: faces and vacuum-viewport faces (module docstring "Geometry").
N_PATCHES = 6

#: Distance from the trap center to each patch source, meters -- a
#: cm-scale in-vacuum dielectric-surface (mirror/viewport) standoff,
#: Lodewyck-2012-style ("a few centimeters from the atoms"); 25 mm sits
#: inside the WP-specified 20-30 mm band (module docstring "Geometry").
PATCH_DISTANCE_M = 25.0e-3

#: Gaussian patch width (`w` in `V_patch`), meters -- chosen relative to
#: `PATCH_DISTANCE_M` so the Gaussian falloff does not suppress the field
#: to nothing at the new cm-scale standoff (module docstring "Tuning
#: reasoning"); a localized ~1 cm charge region on a several-cm dielectric
#: optic, not the whole mirror.
PATCH_WIDTH_M = 8.0e-3

#: Patch potential amplitude range, volts (0.5-5 V, Lodewyck-2012-bracketed
#: partially-discharged trapped-charge potentials -- module docstring
#: "Amplitude range vs. Lodewyck's endpoints"; NOT the Camp/Darling/Brown
#: 1991 mV-scale range, per the dossier's caveat on that citation).
PATCH_AMPLITUDE_RANGE_V: tuple[float, float] = (0.5, 5.0)

#: Exported grid half-extent around the trap center, meters (250 um).
GRID_HALF_EXTENT_M = 2.5e-4

#: Grid points per axis (17^3 = 4913 total -- within the ~15-21/axis
#: guidance, small enough for a fast CLI run and a sub-MB CSV).
GRID_POINTS_PER_AXIS = 17

#: Default output path (relative to this file's directory).
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "patch_field_sr87.csv"


def patch_positions() -> NDArray[np.float64]:
    """The six patch centers, shape (6, 3), meters: +/-`PATCH_DISTANCE_M` on each axis."""
    d = PATCH_DISTANCE_M
    return np.array(
        [
            [d, 0.0, 0.0],
            [-d, 0.0, 0.0],
            [0.0, d, 0.0],
            [0.0, -d, 0.0],
            [0.0, 0.0, d],
            [0.0, 0.0, -d],
        ],
        dtype=np.float64,
    )


def patch_amplitudes_v(seed: int = SEED) -> NDArray[np.float64]:
    """Six patch potential amplitudes, volts (signed), drawn from the Lodewyck-bracketed
    partially-discharged trapped-charge range.

    Parameters
    ----------
    seed : int, default `SEED`
        PRNG seed (module docstring "Determinism").

    Returns
    -------
    NDArray[np.float64], shape (6,)
        Signed amplitudes `A` for `V_patch(r) = A * exp(-|r - r_patch|^2 / (2 w^2))`,
        volts. Magnitude drawn uniformly from `PATCH_AMPLITUDE_RANGE_V`; sign
        drawn independently (patches can be net-positive or net-negative
        relative to the nominal bias -- trapped charge on different
        dielectric faces need not have consistent sign, module docstring
        "Amplitude range vs. Lodewyck's endpoints").
    """
    rng = np.random.default_rng(seed)
    lo, hi = PATCH_AMPLITUDE_RANGE_V
    magnitude = rng.uniform(lo, hi, size=N_PATCHES)
    sign = rng.choice(np.array([-1.0, 1.0]), size=N_PATCHES)
    return np.asarray(magnitude * sign, dtype=np.float64)


def field_at(
    positions_m: NDArray[np.float64],
    patch_centers_m: NDArray[np.float64],
    patch_amplitudes_v: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate the total (bias + patch-superposition) field at `positions_m`.

    Parameters
    ----------
    positions_m : NDArray[np.float64], shape (N, 3)
        Query positions, meters.
    patch_centers_m : NDArray[np.float64], shape (6, 3)
        Patch source positions, meters (:func:`patch_positions`).
    patch_amplitudes_v : NDArray[np.float64], shape (6,)
        Patch potential amplitudes, volts (:func:`patch_amplitudes_v`).

    Returns
    -------
    NDArray[np.float64], shape (N, 3)
        `E(r) = E_bias + sum_k -grad V_patch_k(r)`, V/m (module docstring
        "Physics of the scenario").
    """
    bias = np.asarray(BIAS_FIELD_V_PER_M, dtype=np.float64)
    e = np.broadcast_to(bias, positions_m.shape).astype(np.float64).copy()
    w2 = PATCH_WIDTH_M**2
    for center, amplitude in zip(patch_centers_m, patch_amplitudes_v, strict=True):
        diff = positions_m - center[None, :]  # (N, 3)
        r2 = np.sum(diff * diff, axis=-1)  # (N,)
        envelope = np.exp(-r2 / (2.0 * w2))  # (N,)
        e += (amplitude / w2) * diff * envelope[:, None]
    return e


def sample_grid_points() -> NDArray[np.float64]:
    """The regular 17x17x17 grid of sample positions, shape (N, 3), meters."""
    axis = np.linspace(-GRID_HALF_EXTENT_M, GRID_HALF_EXTENT_M, GRID_POINTS_PER_AXIS)
    mesh = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1).astype(np.float64)


def write_csv(
    path: Path, positions_m: NDArray[np.float64], field_v_per_m: NDArray[np.float64]
) -> None:
    """Write `positions_m`/`field_v_per_m` in the ``docs/fields.md`` CSV contract.

    Each float is written via `repr`, Python's shortest round-trip
    representation -- deterministic given a fixed float value (module
    docstring "Determinism").
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "Ex", "Ey", "Ez"])
        for pos, e in zip(positions_m, field_v_per_m, strict=True):
            writer.writerow([repr(float(v)) for v in (*pos, *e)])


def main(output_path: Path = DEFAULT_OUTPUT_PATH, seed: int = SEED) -> None:
    """Generate and write the patch-field CSV, printing the back-of-envelope check."""
    centers = patch_positions()
    amplitudes = patch_amplitudes_v(seed)
    points = sample_grid_points()
    field = field_at(points, centers, amplitudes)

    write_csv(output_path, points, field)

    e_center = field_at(np.zeros((1, 3)), centers, amplitudes)[0]
    e_center_mag = float(np.linalg.norm(e_center))
    # k_S/nu0 for Sr87 (CONVENTIONS.md E14b), from docs/validation.md KA1's
    # quoted "100 V/m -> Delta_nu/nu0 = -7.170524e-17" (a back-of-envelope
    # cross-check independent of importing cliffordclock, deliberately: this
    # script has no runtime dependency on the package it demonstrates).
    ka1_factor_per_v2_m2 = 7.170524e-17 / 1.0e4
    shift_estimate = -ka1_factor_per_v2_m2 * e_center_mag**2

    print(f"Wrote {points.shape[0]} points to {output_path}")
    print(f"Patch amplitudes (V): {[round(float(a), 4) for a in amplitudes]}")
    print(f"E(trap center) = {e_center} V/m, |E| = {e_center_mag:.4f} V/m")
    print(f"Back-of-envelope Delta_nu/nu0 estimate (Sr87, E14b): {shift_estimate:.3e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"PRNG seed for the patch-amplitude draw (default: {SEED})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    main(output_path=_args.output, seed=_args.seed)
