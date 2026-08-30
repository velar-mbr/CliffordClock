# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differentiable JAX port of the BO+WKB lattice-light-shift chain
(WP37, CONVENTIONS.md section 17 addendum, E41): the same physics
`cliffordclock.integrator.lattice_light_shift` implements and the G18 gate
already approved, reimplemented so that ``jax.grad``/``jax.jit`` work
end to end. This module adds no new physics claim; every equation below is
the SAME equation the reference module cites (Beloy et al. 2020 Eqs. 1, 5,
11, 19-21; Bothwell et al. 2025 Eq. 6), evaluated by a numerical route
chosen for differentiability.

**Why this module exists as a second implementation.** `jax.jit` needs a
fixed computational graph, and reverse-mode autodiff needs a fixed number
of primal operations to build its tape; a `jax.numpy` drop-in for the
existing module cannot supply either, because the reference module's own
numerics are adaptive. Every routine there doubles a grid resolution until
a convergence guard is satisfied, raising
:class:`~cliffordclock.integrator.lattice_light_shift.LatticeLightShiftConvergenceError`
when it never converges (its own module docstring): a data-dependent loop
length is what `jax.jit`/`jax.grad` cannot trace. This
module FIXES every grid resolution at a value chosen offline (see
"Chosen resolution" below), verified once against the reference's own
converged output over a documented input domain (see "Validated input
domain" below), the substitute for the adaptive re-verification a traced
function structurally cannot run on every call.

**Chosen resolution, and its verified error bound.** The axial
finite-difference Hamiltonian (Beloy Eq. 5, the same dimensionless domain
`x = k*z in [-pi/2, pi/2]`, Dirichlet boundary conditions the reference
module uses) is solved as a DENSE `jax.numpy.linalg.eigh` at a fixed grid
resolution ``AXIAL_GRID_N_JAX = 1281``. An offline convergence study
(`tests/test_lattice_light_shift_jax.py::TestOfflineConvergenceStudy`, run
once against the reference module's own converged output at all four of
Bothwell et al. 2025's Table I points, Yb-171) first measured the axial
ground-band energy's relative error at this resolution at `3.6e-7` to
`7.2e-8` across the four points (worst case the shallowest, `u0=56.8 E_R`;
finite-difference truncation error scales `O(1/N^2)`, so shallower traps,
with a smaller axial curvature, converge slightly slower). That number
alone underestimates the error the `X`/`Y`/`Z` radial integral compounds
across many axial solves (:func:`axial_thermal_factors_jax`'s own
docstring), so the same study went on to measure `X`/`Y`/`Z` DIRECTLY at
several `(AXIAL_GRID_N_JAX, RHO_GRID_N_JAX)` pairs against the reference's
own converged output. The finding that fixed the second number: the
reference module's own default convergence guard for these four points
settles at axial grid `1295` and `321` radial points (every one of the
four table points lands on the identical rung of the reference's doubling
ladder), and reproducing THAT SAME `(axial_grid, n_rho)` pair with a dense
solver in place of the reference's tridiagonal one reproduces `X`/`Y`/`Z`
to `~1e-11` relative, confirming the two solvers diagonalize the identical
operator with no implementation discrepancy. Holding `AXIAL_GRID_N_JAX`
at `1281` (close enough to the reference's own `1295` that the axial
grid itself contributes negligible error) and matching the reference's
`n_rho=321` directly gives ``RHO_GRID_N_JAX = 321``, this module's actual
choice; at this resolution the worst-case measured `X`/`Y`/`Z` relative
error across the four table points is `1.57e-7` (`Y`, `u0=112.2 E_R`;
`X`/`Z` errors are smaller, `~1e-8`), an order of magnitude inside the
`1e-6` AGREEMENT contract. An intermediate resolution the study also
measured, `RHO_GRID_N_JAX=257` (below the reference's own `n_rho=321`),
left a `4e-6`-to`-9e-6` residual. At that below-settling resolution, the
RADIAL quadrature was the dominant error source, which is why `n_rho` is
this module's main resolution lever.

**Validated input domain.** `AXIAL_GRID_N_JAX`/`RHO_GRID_N_JAX` are
verified over `u0 in [50, 120] E_R`, `Tr in [500, 750] nK`, Yb-171
(the four G18 table points and their immediate neighborhood): the range
this work package's gate exercises. A caller evaluating far outside this
range (a much shallower or much hotter trap) gets a result from the SAME
fixed grid, with no guard to catch degraded accuracy, because catching it
would require the adaptive doubling this module deliberately does not do
inside the traced core (see above). :func:`axial_thermal_factors_jax` and
:func:`bo_wkb_fractional_light_shift_jax` validate their scalar inputs
(positivity, `Tr > 0`) OUTSIDE the jitted core, in plain Python, before
tracing begins; they do not validate that the inputs fall inside the
convergence-verified domain, since doing so would require running the
reference's OWN adaptive solver on every call, defeating the purpose of a
fixed-shape differentiable core. A caller who needs the convergence
guarantee at a materially different `(u0, Tr)` regime should rerun the
offline study at that regime first.

**The differentiable turning-radius root-find.** Beloy et al. 2020's
classical turning radius `R_nz(E)` (`Unz(Rnz(E)) = E`, stated below their
Eq. 10) is the root of a strictly monotonic function of `rho` at fixed
`(u0, n_z, E)`: `U_nz(0)` is the axial band's minimum (most negative) and
`U_nz(rho) -> 0` as `rho` grows and the local trap depth vanishes
(:func:`_local_depth_er_jax`), so `f(rho) = U_nz(rho) - E` crosses zero
exactly once on `rho in [0, rho_bracket_m]` for any `E` inside the band's
range. This module finds that root with
``jax.lax.custom_root(f, initial_guess, solve, tangent_solve)``: `solve`
runs a FIXED-iteration-count bisection (`BISECTION_ITERS = 60`, chosen so
`bracket / 2**60` is far below double-precision resolution for any
realistic `rho_bracket_m`) that supplies the root's numeric value.
`tangent_solve` implements the scalar implicit-function-theorem solve
JAX's own documentation gives for a scalar root, ``lambda g, y: y /
g(1.0)`` (`g` is `f` linearized at the root, so `g(1.0)` is `df/drho`
there, and `y / g(1.0)` is the scalar linear solve `dg/drho * drho = y`
for `drho`). `custom_root` wires these two pieces together so `jax.grad`
differentiates the ROOT with respect to every closed-over parameter of
`f` (`u0`, the site geometry, `n_z`) using only `f`'s own derivative at
the root; the bisection's comparisons stay entirely inside `solve` and
take no part in that backward pass. This is the formulation the work
package's own instructions name explicitly ("a closed-form-bracketed
bisection with implicit-function-theorem gradients via
`jax.lax.custom_root`").

**A kink `tangent_solve` cannot divide through, and its fix.** `f` above
uses the RAW (un-clamped) eigenvalue
(:func:`_axial_band_energy_er_at_rho_unclamped`); every other function in
this module uses the clamped one
(:func:`_axial_band_energy_er_at_rho`, matching the reference module's
own "clamp an unbound state's energy to exactly `0.0`" convention). The
clamp makes `U_nz(rho)/E_R` exactly `0.0` across the entire ray of `rho`
beyond the crossing, in addition to at the crossing point itself, so a
root-find against the clamped value has no unique target: every `rho` on
that ray satisfies it exactly. Landing on the clamped side of the ray
gives `df/drho = 0` there (the clamp's own subgradient), and
`tangent_solve`'s `y / g(1.0)` then divides by zero. This module's own
tests caught it directly: an early version of this function returned
`NaN` gradients intermittently, at specific `AXIAL_GRID_N_JAX` values,
because whether the bisection's final floating-point step lands just
before or just after the crossing is resolution-dependent. The unclamped
eigenvalue is smooth and crosses zero at the SAME physical `rho` (it
agrees with the clamped value everywhere except beyond that single
crossing), so using it inside `f` removes the kink and leaves the root
and every other function's physics unchanged.

**Float64, required.** This package enables `jax.config.jax_enable_x64`
as an import-time side effect of ``import cliffordclock``
(`cliffordclock/__init__.py`'s own docstring: "the 1e-18 target precision
is unreachable with JAX's default 32-bit dtype"). The lattice light shift
itself lives at the `1e-19` fractional-frequency scale (Bothwell et al.
2025's own headline number), and this module's dense `N=1281` eigenvalue
problem subtracts two O(10-100) `E_R` numbers to resolve an O(1e-5) `E_R`
convergence tolerance; float32's ~7 decimal digits of precision cannot
represent that subtraction meaningfully. This module relies on
:mod:`cliffordclock`'s own `__init__.py`, which already must run first
for `jax.numpy` to be usable at all in this package, to configure x64;
this docstring documents the dependency explicitly, matching
:mod:`cliffordclock.integrator.worldline`'s own stated pattern
("float64 throughout (inherited from `cliffordclock` package import-time
x64 config)").

**Species trap (same warning as the reference module).** `X`/`Y`/`Z`
cancel atomic mass and lattice waist exactly out of their defining ratio
(Beloy Eq. 21), but the species' own recoil energy `E_R` still sets the
`kB*Tr/E_R` thermal-weighting scale every downstream evaluation uses.
Passing `(u0, Tr)` from a published table together with the wrong
species' `mass_kg`/`wavelength_m` evaluates a different physical trap
depth than the table intends, with no other symptom
(:func:`make_site_potential_jax`'s and :func:`axial_thermal_factors_jax`'s
own docstrings repeat this warning, matching
:func:`cliffordclock.integrator.lattice_light_shift.make_site_potential`'s
own documented finding).

**Scope boundary.** No spectrum or lineshape model: that is WP38, gated
on a still-pending research round. No pipeline wiring. This module is
functions only, validated against the already-gated reference
implementation and against central finite differences of that same
reference implementation; see `tests/test_lattice_light_shift_jax.py`
for the full agreement and gradient checks.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from cliffordclock.constants import BOLTZMANN_K, HBAR, PLANCK_H

__all__ = [
    "AXIAL_GRID_N_JAX",
    "RHO_GRID_N_JAX",
    "BISECTION_ITERS",
    "recoil_energy_j_jax",
    "ushijima_reduction_factor_jax",
    "jila_reduction_factor_jax",
    "harmonic_light_shift_hz_jax",
    "SitePotentialJax",
    "make_site_potential_jax",
    "axial_energies_er_jax",
    "harmonic_density_of_states_closed_form_jax",
    "turning_radius_m_jax",
    "bo_wkb_density_of_states_jax",
    "ThermalShapeFactorsJax",
    "axial_thermal_factors_jax",
    "bo_wkb_fractional_light_shift_jax",
]

# ---------------------------------------------------------------------------
# Fixed resolutions (offline-validated; see module docstring)
# ---------------------------------------------------------------------------

#: Static axial finite-difference grid size, dense `jnp.linalg.eigh`. Fixed
#: (not adaptive) because `jax.jit`/`jax.grad` need a static computational
#: graph; the offline convergence study
#: (`tests/test_lattice_light_shift_jax.py`) pins its verified error bound.
AXIAL_GRID_N_JAX = 1281

#: Static radial quadrature point count for the `X`/`Y`/`Z` trapezoidal
#: integral (:func:`axial_thermal_factors_jax`), on the reparametrized
#: domain `rho = s*rho_max`, `s in [0, 1]` (`rho_max` itself the
#: differentiable turning radius at `E=0`, so the domain rescales
#: correctly under `jax.grad` without needing the endpoint itself to be
#: static). Matches the reference module's own converged `n_rho=321` at
#: all four G18 table points (see module docstring's "Chosen resolution"
#: section).
RHO_GRID_N_JAX = 321

#: Chunk size for the radial evaluation's memory-bounded schedule
#: (:func:`axial_thermal_factors_jax`): `jax.lax.map(sample, rhos,
#: batch_size=RHO_MAP_BATCH_SIZE)`, replacing a single materializing
#: `jax.vmap` over all `RHO_GRID_N_JAX` points. A CI runner OOM-killed
#: PR #19's slow lane twice; a fresh-subprocess `resource.getrusage`
#: reading (`tests/test_lattice_light_shift_jax.py`'s own RSS guard)
#: measured the plain-`vmap` path at `~31 GB` peak RSS for one
#: `value_and_grad` call, the batched Hamiltonians, eigenvectors, and
#: `eigh`'s own backward-pass residuals for all 321 dense `(1281,
#: 1281)` float64 eigenproblems living at once. `batch_size` alone
#: only partially fixes this: `jax.lax.map` compiles to a `scan`, and
#: reverse-mode autodiff through a `scan` still saves every chunk's own
#: residuals for the backward pass, so chunking without checkpointing
#: measured `~13-15 GB` across `batch_size in {1, 4, 16}`, a real
#: reduction but far short of the CI runner's budget. Wrapping the
#: per-`rho` `sample` closure in `jax.checkpoint` (below) discards
#: those residuals after the forward pass and recomputes them fresh
#: during the backward pass; combined with `batch_size=16`, the SAME
#: call measured `~2.2 GB` peak RSS, at
#: roughly `2x` the single-call wall time, the expected memory-for-
#: recompute trade `jax.checkpoint` makes. `16` balances chunk-loop
#: overhead against memory;
#: `tests/test_lattice_light_shift_jax.py`'s own RSS guard pins the
#: measured bound this choice achieves.
RHO_MAP_BATCH_SIZE = 16

#: Fixed bisection iteration count for the turning-radius root-find
#: (:func:`turning_radius_m_jax`). `60` halvings of any realistic
#: `rho_bracket_m` (tens of microns to millimeters) land far below
#: float64's ~15-16 decimal digits of resolution, matching the reference
#: module's own `brentq(..., xtol=1e-12*site.waist_m)` in spirit.
BISECTION_ITERS = 60

#: Default outward bracket multiple for the turning-radius root-find,
#: matching :func:`cliffordclock.integrator.lattice_light_shift.turning_radius_m`'s
#: own default (`10 * site.waist_m`): comfortably beyond where any bound
#: axial state persists for realistic depths.
DEFAULT_RHO_BRACKET_WAIST_MULTIPLE = 10.0


# ---------------------------------------------------------------------------
# Shared helpers (Model A and Model B both use recoil energy)
# ---------------------------------------------------------------------------


def recoil_energy_j_jax(wavelength_m: jnp.ndarray, mass_kg: jnp.ndarray) -> jnp.ndarray:
    """`E_R = h^2/(2*m*lambda^2)` (CONVENTIONS.md E41; identical formula to
    :func:`cliffordclock.integrator.lattice_light_shift.recoil_energy_j`,
    reimplemented in `jax.numpy` so it stays differentiable with respect to
    `wavelength_m`/`mass_kg` when those flow from a caller's traced
    pytree). No input validation here (this is the differentiable core;
    see the module docstring's "validated outside the jitted core" note).

    Parameters
    ----------
    wavelength_m : jax.Array or float
        Lattice laser wavelength, meters.
    mass_kg : jax.Array or float
        Atomic mass, kilograms.

    Returns
    -------
    jax.Array
        `E_R`, joules.
    """
    return PLANCK_H**2 / (2.0 * mass_kg * wavelength_m**2)


# ---------------------------------------------------------------------------
# Model A: Katori-lineage harmonic/operational model (E40), ported verbatim
# ---------------------------------------------------------------------------


def ushijima_reduction_factor_jax(
    u: jnp.ndarray, j: float, radial_temperature_k: jnp.ndarray, recoil_energy_j_value: jnp.ndarray
) -> jnp.ndarray:
    """`zeta_j(u) ~= 1 - j*kB*Tr/(u*E_R)` (Ushijima et al. 2018 Eq. 2,
    LINEAR form; same formula as
    :func:`cliffordclock.integrator.lattice_light_shift.ushijima_reduction_factor`,
    reimplemented in `jax.numpy`). See that function's docstring for the
    "not the same formula as the reciprocal form" warning, which applies
    identically here.
    """
    return 1.0 - j * BOLTZMANN_K * radial_temperature_k / (u * recoil_energy_j_value)


def jila_reduction_factor_jax(
    u: jnp.ndarray, j: float, radial_temperature_k: jnp.ndarray, recoil_energy_j_value: jnp.ndarray
) -> jnp.ndarray:
    """`zeta_j(u) = (1 + j*kB*Tr/(u*E_R))^-1` (Kim et al. 2023 / Bothwell
    et al. 2025's RECIPROCAL form; same formula as
    :func:`cliffordclock.integrator.lattice_light_shift.jila_reduction_factor`,
    reimplemented in `jax.numpy`).
    """
    return 1.0 / (1.0 + j * BOLTZMANN_K * radial_temperature_k / (u * recoil_energy_j_value))


def harmonic_light_shift_hz_jax(
    u: jnp.ndarray,
    detuning_hz: jnp.ndarray,
    n_z: jnp.ndarray,
    e1_slope_per_hz: jnp.ndarray,
    m1e2_hz: jnp.ndarray,
    hyperpolarizability_hz: jnp.ndarray,
    *,
    reduction_form: str = "none",
    radial_temperature_k: jnp.ndarray | None = None,
    recoil_energy_j_value: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Model A's light shift (Ushijima et al. 2018 Eq. 1), a direct
    `jax.numpy` port of
    :func:`cliffordclock.integrator.lattice_light_shift.harmonic_light_shift_hz`
    with no numerical difference from that function: this is pure
    coefficient algebra (four powers of `u`, no eigenproblem, no
    root-find), so both implementations evaluate the SAME closed-form
    expression and agree to floating-point precision
    (`TestModelAPortEquivalence`, `tests/test_lattice_light_shift_jax.py`).

    `coeffs` is unpacked into three scalar arguments here, so every
    differentiable input is a plain `jax.Array` leaf: the explicit-
    argument convention
    :mod:`cliffordclock.integrator.worldline` already establishes for
    this package's JAX modules, in place of a nested-dataclass pytree.

    `reduction_form` is a plain Python string, read at trace time to
    select between two DIFFERENT closed-form expressions (like `n_z`'s
    role in the reference module), so it must be passed as a static
    argument under `jax.jit` (`static_argnames=("reduction_form",)`).

    Parameters, returns: same physical contract as the reference
    function, `u`/`detuning_hz`/`n_z`/the three coefficients/
    `radial_temperature_k`/`recoil_energy_j_value` all differentiable
    `jax.Array` or Python-scalar inputs.
    """
    if reduction_form == "none":

        def u_pow(j: float) -> jnp.ndarray:
            return u**j
    else:
        if radial_temperature_k is None or recoil_energy_j_value is None:
            raise ValueError(
                "radial_temperature_k and recoil_energy_j_value are required when "
                f"reduction_form={reduction_form!r}"
            )
        reduction_fn = (
            ushijima_reduction_factor_jax
            if reduction_form == "ushijima_linear"
            else jila_reduction_factor_jax
        )

        def u_pow(j: float) -> jnp.ndarray:
            zeta = reduction_fn(u, j, radial_temperature_k, recoil_energy_j_value)
            return zeta * u**j

    term1 = (e1_slope_per_hz * detuning_hz - m1e2_hz) * (n_z + 0.5) * u_pow(0.5)
    term2 = -(
        e1_slope_per_hz * detuning_hz + 1.5 * hyperpolarizability_hz * (n_z**2 + n_z + 0.5)
    ) * u_pow(1.0)
    term3 = 2.0 * hyperpolarizability_hz * (n_z + 0.5) * u_pow(1.5)
    term4 = -hyperpolarizability_hz * u_pow(2.0)
    return term1 + term2 + term3 + term4


# ---------------------------------------------------------------------------
# Model B: NIST Born-Oppenheimer + WKB model (E41), differentiable core
# ---------------------------------------------------------------------------


class SitePotentialJax(NamedTuple):
    """JAX-differentiable counterpart to
    :class:`cliffordclock.integrator.lattice_light_shift.SitePotential`: a
    plain `NamedTuple` of `jax.Array` scalars, a valid pytree
    `jax.jit`/`jax.grad` can build, pass through, and differentiate
    directly (a frozen dataclass, the reference class's own choice, is
    not automatically a pytree).

    Attributes
    ----------
    depth_er : jax.Array
        `u0`, the peak reduced trap depth. The primary differentiable
        input this work package's gate exercises.
    waist_m, wavelength_m, mass_kg : jax.Array
        Same physical meaning as the reference `SitePotential`'s fields.
    recoil_energy_j_value, kappa_per_m, k_per_m : jax.Array
        Derived quantities (:func:`make_site_potential_jax`), kept as
        explicit fields (not recomputed on every use) so a caller who
        wants to differentiate with respect to `wavelength_m`/`mass_kg`
        directly still can, by building this tuple from traced leaves in
        the first place.
    """

    depth_er: jnp.ndarray
    waist_m: jnp.ndarray
    wavelength_m: jnp.ndarray
    mass_kg: jnp.ndarray
    recoil_energy_j_value: jnp.ndarray
    kappa_per_m: jnp.ndarray
    k_per_m: jnp.ndarray


def make_site_potential_jax(
    depth_er: jnp.ndarray, waist_m: jnp.ndarray, wavelength_m: jnp.ndarray, mass_kg: jnp.ndarray
) -> SitePotentialJax:
    """Build a :class:`SitePotentialJax`, the differentiable counterpart to
    :func:`cliffordclock.integrator.lattice_light_shift.make_site_potential`
    (same formulas: `kappa = sqrt(2)/w`, `k = 2*pi/lambda`,
    :func:`recoil_energy_j_jax`).

    No input validation here (`depth_er > 0`/`waist_m > 0` are checked by
    the reference module's own `make_site_potential` and, for this
    module's public entry points, by the plain-Python guard at the top of
    :func:`axial_thermal_factors_jax`/:func:`bo_wkb_fractional_light_shift_jax`
    -- see the module docstring's "validated outside the jitted core"
    note): this constructor is meant to be called freely from inside a
    jitted/traced context.

    **Species trap** (same warning as
    :func:`cliffordclock.integrator.lattice_light_shift.make_site_potential`):
    `mass_kg`/`wavelength_m` set this site's `E_R`, which controls the
    thermal weighting (`kB*Tr/E_R`) every downstream `X`/`Y`/`Z`
    evaluation uses. Reusing a `(u0, Tr)` pair from a published table with
    the wrong species' mass/wavelength evaluates a different physical
    trap depth than the table intends, even though `waist_m` itself never
    affects the result.
    """
    e_r = recoil_energy_j_jax(wavelength_m, mass_kg)
    return SitePotentialJax(
        depth_er=jnp.asarray(depth_er),
        waist_m=jnp.asarray(waist_m),
        wavelength_m=jnp.asarray(wavelength_m),
        mass_kg=jnp.asarray(mass_kg),
        recoil_energy_j_value=e_r,
        kappa_per_m=jnp.sqrt(2.0) / waist_m,
        k_per_m=2.0 * jnp.pi / wavelength_m,
    )


def _local_depth_er_jax(
    depth_er: jnp.ndarray, kappa_per_m: jnp.ndarray, rho_m: jnp.ndarray
) -> jnp.ndarray:
    """`D(rho)/E_R = depth_er * exp(-kappa^2*rho^2)`, same factorization
    :func:`cliffordclock.integrator.lattice_light_shift._local_depth_er`
    exploits.
    """
    return depth_er * jnp.exp(-(kappa_per_m**2) * rho_m**2)


def _axial_grid(grid_n: int) -> tuple[jnp.ndarray, float]:
    """The fixed axial finite-difference grid (static `grid_n`), same
    domain as :func:`cliffordclock.integrator.lattice_light_shift._axial_fd_solve`:
    `x = k*z in [-pi/2, pi/2]`, `grid_n` interior points, Dirichlet
    boundary conditions at the domain edges.
    """
    dx = math.pi / (grid_n + 1)
    x_grid = -math.pi / 2.0 + dx * jnp.arange(1, grid_n + 1, dtype=jnp.float64)
    return x_grid, dx


def _axial_hamiltonian_dense(
    depth_local_er: jnp.ndarray, x_grid: jnp.ndarray, dx: float
) -> jnp.ndarray:
    """Dense `(grid_n, grid_n)` axial Hamiltonian at fixed local depth,
    `E_R` units (`-d^2/dx^2` has coefficient exactly `1` in these units,
    same as the reference module's `_axial_fd_solve`). This module uses
    `jax.numpy.linalg.eigh` on the full dense matrix, because
    `scipy.linalg.eigh_tridiagonal`'s tridiagonal-specialized solver has
    no JAX counterpart with a differentiable `eigh`. `AXIAL_GRID_N_JAX`
    is chosen small enough that the dense solve stays fast (see the
    module docstring's resolution note), and the tridiagonal structure
    is exactly recovered in the dense matrix (all off-tridiagonal entries
    `0`), so the two solvers diagonalize the identical operator.
    """
    grid_n = x_grid.shape[0]
    v = -depth_local_er * jnp.cos(x_grid) ** 2
    diag = (2.0 / dx**2) + v
    off = jnp.full(grid_n - 1, -1.0 / dx**2, dtype=jnp.float64)
    return jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)


def _axial_solve_dense(
    depth_local_er: jnp.ndarray, x_grid: jnp.ndarray, dx: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Full dense eigendecomposition at fixed local depth, ascending
    eigenvalues (`jax.numpy.linalg.eigh`'s own convention, matching
    `scipy.linalg.eigh_tridiagonal`'s). Differentiable with respect to
    `depth_local_er` via `jax.numpy.linalg.eigh`'s built-in VJP.

    Returns
    -------
    energies : jax.Array, shape (grid_n,)
    eigvecs : jax.Array, shape (grid_n, grid_n)
        Columns are L2-normalized eigenvectors (`sum(v**2) == 1`), same
        normalization convention as the reference module's
        `eigh_tridiagonal` output.
    """
    h = _axial_hamiltonian_dense(depth_local_er, x_grid, dx)
    energies, eigvecs = jnp.linalg.eigh(h)
    return energies, eigvecs


def axial_energies_er_jax(
    depth_local_er: jnp.ndarray, n_states: int, x_grid: jnp.ndarray, dx: float
) -> jnp.ndarray:
    """Fixed-resolution axial band energies `U_nz(rho=0-equivalent)/E_R`
    at one local depth (CONVENTIONS.md E41, Beloy et al. 2020 Eq. 5), the
    differentiable counterpart to
    :func:`cliffordclock.integrator.lattice_light_shift.axial_energies_er`.
    No convergence loop (see module docstring): `x_grid`/`dx` come from
    :func:`_axial_grid` at the fixed `AXIAL_GRID_N_JAX`.

    A state whose energy is `>= 0` (unbound at this depth) is clamped to
    exactly `0.0` via `jnp.minimum`, matching the reference module's
    convention and remaining differentiable (subgradient `0` on the
    clamped branch, matching the reference's own hard clamp, which is
    not itself differentiated either).

    Parameters
    ----------
    depth_local_er : jax.Array
        Local peak depth `D(rho)/E_R`.
    n_states : int
        Number of lowest axial bands to return. Static (selects a slice
        of the eigenvalue array, fixed at trace time).
    x_grid, dx
        From :func:`_axial_grid`.

    Returns
    -------
    jax.Array, shape (n_states,)
    """
    energies, _ = _axial_solve_dense(depth_local_er, x_grid, dx)
    return jnp.minimum(energies[:n_states], 0.0)


def harmonic_density_of_states_closed_form_jax(
    depth_er: jnp.ndarray,
    recoil_energy_j_value: jnp.ndarray,
    kappa_per_m: jnp.ndarray,
    k_per_m: jnp.ndarray,
    n_z: int,
    energy_j: jnp.ndarray,
) -> jnp.ndarray:
    """`G^HO_nz(E)` (Beloy et al. 2020 Eq. 4), a direct `jax.numpy` port of
    :func:`cliffordclock.integrator.lattice_light_shift.harmonic_density_of_states_closed_form`
    (pure algebra, no eigenproblem): same formula,

        G^HO_nz(E) = (kappa/k)^(-2) / (4*D*E_R) * [E + D - 2*sqrt(D*E_R)*(n_z+1/2)]

    with `D = depth_er * recoil_energy_j_value` the peak depth in joules.
    A negative bracket clamps to `0.0` via `jnp.maximum`
    (Beloy's own stated convention), same subgradient-at-the-clamp
    behavior as :func:`axial_energies_er_jax`.
    """
    d_joules = depth_er * recoil_energy_j_value
    e_r = recoil_energy_j_value
    kappa_over_k_sq = (kappa_per_m / k_per_m) ** 2
    bracket = energy_j + d_joules - 2.0 * jnp.sqrt(d_joules * e_r) * (n_z + 0.5)
    return jnp.maximum(bracket, 0.0) / (kappa_over_k_sq * 4.0 * d_joules * e_r)


def _axial_band_energy_er_at_rho(
    depth_er: jnp.ndarray,
    kappa_per_m: jnp.ndarray,
    n_z: int,
    rho_m: jnp.ndarray,
    x_grid: jnp.ndarray,
    dx: float,
) -> jnp.ndarray:
    """`U_nz(rho)/E_R`, the fixed-resolution counterpart to
    :func:`cliffordclock.integrator.lattice_light_shift.axial_band_energy_er`,
    `potential="cos2"` only (the true site potential). The reference
    module's own `"harmonic"` branch exists solely for that module's own
    internal harmonic-limit consistency tests; this module's own tests
    validate against the reference's `"cos2"` output directly.
    """
    depth_local = _local_depth_er_jax(depth_er, kappa_per_m, rho_m)
    energies = axial_energies_er_jax(depth_local, n_z + 1, x_grid, dx)
    return energies[n_z]


def _axial_band_energy_er_at_rho_unclamped(
    depth_er: jnp.ndarray,
    kappa_per_m: jnp.ndarray,
    n_z: int,
    rho_m: jnp.ndarray,
    x_grid: jnp.ndarray,
    dx: float,
) -> jnp.ndarray:
    """The RAW (un-clamped) `n_z`-th eigenvalue at `rho`, this module's
    internal input to :func:`turning_radius_m_jax`'s root-find only.
    :func:`_axial_band_energy_er_at_rho` above (which clamps a `>= 0`
    eigenvalue to exactly `0.0`, matching the reference module's own
    convention) is the public physical quantity every other caller uses.
    The module docstring's "A kink `tangent_solve` cannot divide
    through, and its fix" section derives why the root-find specifically
    needs the smooth, unclamped curve: the clamped value is flat (exactly
    `0.0`) across an entire ray of `rho` beyond the true crossing, giving
    `tangent_solve` a zero denominator whenever the bisection lands on
    that ray, an intermittent `NaN` this module's own tests caught.
    """
    depth_local = _local_depth_er_jax(depth_er, kappa_per_m, rho_m)
    energies, _ = _axial_solve_dense(depth_local, x_grid, dx)
    return energies[n_z]


def turning_radius_m_jax(
    depth_er: jnp.ndarray,
    kappa_per_m: jnp.ndarray,
    n_z: int,
    energy_er: jnp.ndarray,
    x_grid: jnp.ndarray,
    dx: float,
    rho_bracket_m: jnp.ndarray,
) -> jnp.ndarray:
    """`R_nz(E)`, the classical turning radius (Beloy et al. 2020's
    notation, `Unz(Rnz(E)) = E`), differentiable via
    ``jax.lax.custom_root``. See the module docstring's "differentiable
    turning-radius root-find" section for the full derivation; in brief,
    `f(rho) = U_nz(rho) - E` is strictly increasing from `f(0) <= 0` to
    `f(rho_bracket_m) >= 0` for any `E` inside the band's range, bisected
    to convergence by a fixed-iteration-count `solve` (never itself
    differentiated), with the root's gradient supplied by the scalar
    implicit-function-theorem `tangent_solve`.

    Parameters
    ----------
    depth_er, kappa_per_m : jax.Array
        Site parameters (:class:`SitePotentialJax` fields).
    n_z : int
        Static.
    energy_er : jax.Array
        Target energy, `E_R` units. Caller's responsibility to keep this
        inside `[U_nz(0), 0]` (this function does not validate that range
        under trace, matching the module's "validate outside the jitted
        core" discipline; the reference module's own `turning_radius_m`
        raises `ValueError` for an out-of-range target, which this
        function cannot do under `jax.jit`).
    x_grid, dx
        From :func:`_axial_grid`.
    rho_bracket_m : jax.Array
        Outward bracket bound, meters (:data:`DEFAULT_RHO_BRACKET_WAIST_MULTIPLE`
        `* waist_m` is the usual choice). Differentiable: if built from
        `waist_m`, `jax.grad` with respect to `waist_m` flows through the
        bracket bound too (it does not affect the ROOT's value once the
        root is bracketed correctly, only where the bisection searches,
        so this dependency contributes nothing to the gradient in
        practice, consistent with the reference module's turning radius
        not depending on the search bound either).

    Returns
    -------
    jax.Array, scalar
        `R_nz(E)`, meters.
    """

    def f(rho: jnp.ndarray) -> jnp.ndarray:
        return (
            _axial_band_energy_er_at_rho_unclamped(depth_er, kappa_per_m, n_z, rho, x_grid, dx)
            - energy_er
        )

    def solve(
        f_inner: Callable[[jnp.ndarray], jnp.ndarray], initial_guess: jnp.ndarray
    ) -> jnp.ndarray:  # noqa: ARG001 - custom_root's required signature
        lo0 = jnp.zeros((), dtype=jnp.float64)
        hi0 = rho_bracket_m

        def body(
            _i: jnp.ndarray, carry: tuple[jnp.ndarray, jnp.ndarray]
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            lo, hi = carry
            mid = 0.5 * (lo + hi)
            go_right = f_inner(mid) < 0.0
            lo = jnp.where(go_right, mid, lo)
            hi = jnp.where(go_right, hi, mid)
            return lo, hi

        lo, hi = jax.lax.fori_loop(0, BISECTION_ITERS, body, (lo0, hi0))
        result: jnp.ndarray = 0.5 * (lo + hi)
        return result

    def tangent_solve(g: Callable[[jnp.ndarray], jnp.ndarray], y: jnp.ndarray) -> jnp.ndarray:
        return y / g(jnp.ones(()))

    root: jnp.ndarray = jax.lax.custom_root(
        f, jnp.zeros((), dtype=jnp.float64), solve, tangent_solve
    )
    return root


def bo_wkb_density_of_states_jax(
    depth_er: jnp.ndarray,
    kappa_per_m: jnp.ndarray,
    mass_kg: jnp.ndarray,
    recoil_energy_j_value: jnp.ndarray,
    n_z: int,
    energy_j: jnp.ndarray,
    x_grid: jnp.ndarray,
    dx: float,
    rho_bracket_m: jnp.ndarray,
) -> jnp.ndarray:
    """`G_nz(E) = (1/4)*(2*m/hbar^2)*[R_nz(E)]^2` (Beloy et al. 2020
    Eq. 11), the differentiable counterpart to
    :func:`cliffordclock.integrator.lattice_light_shift.bo_wkb_density_of_states`,
    via :func:`turning_radius_m_jax`. Returns `0.0` outside the band's
    range (`E < U_nz(0)` or `E > 0`) using `jnp.where`, keeping the
    function traceable: both branches are evaluated (the turning-radius
    root-find runs even when its result is discarded), a deliberate trade
    of extra compute for a static computational graph.
    """
    e_r_j = recoil_energy_j_value
    energy_er = energy_j / e_r_j
    e0 = _axial_band_energy_er_at_rho(
        depth_er, kappa_per_m, n_z, jnp.zeros((), dtype=jnp.float64), x_grid, dx
    )
    in_band = (energy_er >= e0) & (energy_er <= 0.0)
    r_nz = turning_radius_m_jax(depth_er, kappa_per_m, n_z, energy_er, x_grid, dx, rho_bracket_m)
    value = 0.25 * (2.0 * mass_kg / HBAR**2) * r_nz**2
    return jnp.where(in_band, value, 0.0)


class ThermalShapeFactorsJax(NamedTuple):
    """Differentiable counterpart to
    :class:`cliffordclock.integrator.lattice_light_shift.ThermalShapeFactors`.
    `rho_max_m`/`n_rho_points`/`axial_grid_n` are informational, fixed by
    this module's static resolution choice, kept for output parity with
    the reference's own result type.
    """

    x_nz: jnp.ndarray
    y_nz: jnp.ndarray
    z_nz: jnp.ndarray
    rho_max_m: jnp.ndarray
    n_rho_points: int
    axial_grid_n: int


def _shape_integrand_at_rho(
    depth_er: jnp.ndarray,
    kappa_per_m: jnp.ndarray,
    n_z: int,
    rho_m: jnp.ndarray,
    x_grid: jnp.ndarray,
    dx: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One `rho` sample's `(x_nz(rho), y_nz(rho), z_nz(rho), U_nz(rho)/E_R)`
    (Beloy et al. 2020 Eq. 13), the differentiable counterpart to
    :func:`cliffordclock.integrator.lattice_light_shift._shape_integrand_at_rho`.
    This function always evaluates the same fixed
    `AXIAL_GRID_N_JAX`-resolution eigenproblem and returns a numeric
    tuple every time. `jax.jit` cannot trace a Python-level branch, so
    this function has no counterpart to the reference's own "band not
    bound at this `rho`" `None` return; that reference branch only fires
    at `u0` values far outside this module's validated domain (see
    module docstring). A caller outside that domain sees whatever
    numbers this fixed-resolution eigenproblem happens to produce,
    potentially describing an unbound state.
    """
    depth_local = _local_depth_er_jax(depth_er, kappa_per_m, rho_m)
    energies, eigvecs = _axial_solve_dense(depth_local, x_grid, dx)
    e_nz = energies[n_z]
    v = eigvecs[:, n_z]
    v2 = v * v
    cos2 = jnp.cos(x_grid) ** 2
    sin2 = 1.0 - cos2
    cos4 = cos2 * cos2
    kappa2_rho2 = (kappa_per_m * rho_m) ** 2
    x_nz = jnp.exp(-kappa2_rho2) * jnp.sum(v2 * cos2)
    y_nz = jnp.exp(-kappa2_rho2) * jnp.sum(v2 * sin2)
    z_nz = jnp.exp(-2.0 * kappa2_rho2) * jnp.sum(v2 * cos4)
    return x_nz, y_nz, z_nz, e_nz


def axial_thermal_factors_jax(
    site: SitePotentialJax,
    n_z: int,
    radial_temperature_k: jnp.ndarray,
    *,
    rho_bracket_waist_multiple: float = DEFAULT_RHO_BRACKET_WAIST_MULTIPLE,
    axial_grid_n: int = AXIAL_GRID_N_JAX,
    rho_grid_n: int = RHO_GRID_N_JAX,
    rho_map_batch_size: int = RHO_MAP_BATCH_SIZE,
) -> ThermalShapeFactorsJax:
    """Ensemble-averaged trap-depth-reduction factors `X`, `Y`, `Z`
    (CONVENTIONS.md E41, Beloy et al. 2020 Eq. 21), the differentiable
    fixed-resolution counterpart to
    :func:`cliffordclock.integrator.lattice_light_shift.axial_thermal_factors`.
    Same defining ratio,

        X_nz = integral_0^Rnz(0) [x_nz(rho)*rho*(exp(-Unz(rho)/kB*Tr)-1)] drho
               / integral_0^Rnz(0) [rho*(exp(-Unz(rho)/kB*Tr)-1)] drho

    (analogous for `Y`/`Z`), evaluated by:

    1. :func:`turning_radius_m_jax` at `E=0` for the differentiable upper
       bound `Rnz(0)` (`rho_max`).
    2. A radial grid REPARAMETRIZED as `rho = s*rho_max`,
       `s = linspace(0, 1, RHO_GRID_N_JAX)` (fixed shape, `s` itself never
       differentiated): `jax.grad` with respect to any parameter `rho_max`
       depends on (`u0`, the site geometry) flows through this
       substitution automatically via the standard chain rule, so the
       INTEGRATION DOMAIN's own dependence on `u0`/`Tr` is captured
       without needing to differentiate through the root-find a second
       time inside the integral.
    3. A memory-bounded schedule over the radial grid,
       `jax.lax.map(sample, rhos, batch_size=RHO_MAP_BATCH_SIZE)`
       (`RHO_MAP_BATCH_SIZE`'s own docstring has the measured numbers): a
       single materializing `jax.vmap` over all `RHO_GRID_N_JAX` axial
       eigensolves (:func:`_shape_integrand_at_rho`) peaked at `~31 GB`
       RSS for one `value_and_grad` call, enough to OOM a constrained CI
       runner. `jax.checkpoint` on the per-`rho` sample, combined with
       `batch_size=16` chunking, brings that down to `~2.2 GB`, at
       roughly `2x` the wall time.
    4. The SAME numerically-stabilized shifted-exponential reformulation
       the reference module uses (`exp(-(Unz(rho)-Unz(0))/kB*Tr) -
       exp(Unz(0)/kB*Tr)`, bounded `<= 1` everywhere, avoiding the
       float64 overflow the reference module's own docstring documents
       for the naive `exp(-Unz(rho)/kB*Tr)` form), integrated by
       `jax.numpy.trapezoid`.

    **Numerical scope, stated because it differs from the reference
    module's:** no convergence guard, no `Tr=0` special case (the
    reference module's exact `X=Z=1, Y=0` limit at `Tr=0` is a Python-level
    branch this module does not trace; `radial_temperature_k` must be
    `> 0`, checked below, outside the jitted core).

    **Species trap** (same warning as the reference's own
    `axial_thermal_factors`): `X`/`Y`/`Z` cancel `site.mass_kg` and
    `site.waist_m` exactly out of their defining ratio, so neither
    affects the result; `site`'s recoil energy `E_R` still enters through
    `kB*Tr/E_R`. Building `site` with the wrong species' mass/wavelength
    against a published `(u0, Tr)` pair silently evaluates a different
    physical trap depth than intended.

    Parameters
    ----------
    site : SitePotentialJax
    n_z : int
        Static.
    radial_temperature_k : jax.Array
        `Tr`, kelvin. Must be `> 0` (checked below, before tracing).
    rho_bracket_waist_multiple : float, default 10.0
        Passed to :func:`turning_radius_m_jax` as
        `rho_bracket_m = rho_bracket_waist_multiple * site.waist_m`.
    axial_grid_n : int, default AXIAL_GRID_N_JAX
        Static axial finite-difference grid size. Overridable ONLY so the
        offline convergence study
        (`tests/test_lattice_light_shift_jax.py::TestOfflineConvergenceStudy`)
        can call this exact production code path at other resolutions and
        pin the trend that justified `AXIAL_GRID_N_JAX`'s default; a
        physical evaluation should use the default.
    rho_grid_n : int, default RHO_GRID_N_JAX
        Static radial quadrature point count. Same overridable-for-the-
        convergence-study-only role as `axial_grid_n`.
    rho_map_batch_size : int, default RHO_MAP_BATCH_SIZE
        Chunk size for the radial evaluation's `jax.lax.map` schedule
        (`RHO_MAP_BATCH_SIZE`'s own docstring has the measured memory
        numbers this default achieves). Overridable for the same
        testing purpose as `axial_grid_n`/`rho_grid_n`: the module's own
        RSS-guard test passes a larger value to reproduce the pre-fix
        memory blowup as a regression check. A physical evaluation
        should use the default.

    Returns
    -------
    ThermalShapeFactorsJax

    Raises
    ------
    ValueError
        `n_z < 0` or `radial_temperature_k <= 0` -- checked with a plain
        Python comparison against the CONCRETE value of
        `radial_temperature_k`; calling this function under `jax.jit`
        with a TRACED `radial_temperature_k` skips this check (a Tracer
        has no concrete value to compare), matching this module's
        documented "validate outside the jitted core" discipline: a
        caller who wants the check should call this function eagerly
        first, or validate `Tr > 0` in their own un-jitted wrapper.
    """
    if n_z < 0:
        raise ValueError(f"n_z must be >= 0, got {n_z}")
    if not isinstance(radial_temperature_k, jax.core.Tracer) and float(radial_temperature_k) <= 0.0:
        raise ValueError(f"radial_temperature_k must be > 0, got {radial_temperature_k}")

    x_grid, dx = _axial_grid(axial_grid_n)
    rho_bracket_m = rho_bracket_waist_multiple * site.waist_m
    rho_max = turning_radius_m_jax(
        site.depth_er,
        site.kappa_per_m,
        n_z,
        jnp.zeros((), dtype=jnp.float64),
        x_grid,
        dx,
        rho_bracket_m,
    )

    s = jnp.linspace(0.0, 1.0, rho_grid_n, dtype=jnp.float64)
    rhos = s * rho_max

    @jax.checkpoint
    def sample(rho: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return _shape_integrand_at_rho(site.depth_er, site.kappa_per_m, n_z, rho, x_grid, dx)

    x_arr, y_arr, z_arr, u_arr = jax.lax.map(sample, rhos, batch_size=rho_map_batch_size)

    kt = BOLTZMANN_K * radial_temperature_k
    u0_j = u_arr[0] * site.recoil_energy_j_value
    shifted = jnp.exp(-(u_arr * site.recoil_energy_j_value - u0_j) / kt) - jnp.exp(u0_j / kt)
    weight = rhos * shifted
    denom = jnp.trapezoid(weight, rhos)
    num_x = jnp.trapezoid(x_arr * weight, rhos)
    num_y = jnp.trapezoid(y_arr * weight, rhos)
    num_z = jnp.trapezoid(z_arr * weight, rhos)

    return ThermalShapeFactorsJax(
        x_nz=num_x / denom,
        y_nz=num_y / denom,
        z_nz=num_z / denom,
        rho_max_m=rho_max,
        n_rho_points=rho_grid_n,
        axial_grid_n=axial_grid_n,
    )


def bo_wkb_fractional_light_shift_jax(
    n_z: int,
    u0: jnp.ndarray,
    detuning_hz: jnp.ndarray,
    radial_temperature_k: jnp.ndarray,
    e1_slope_per_hz: jnp.ndarray,
    m1e2_hz: jnp.ndarray,
    hyperpolarizability_hz: jnp.ndarray,
    waist_m: jnp.ndarray,
    wavelength_m: jnp.ndarray,
    mass_kg: jnp.ndarray,
) -> tuple[jnp.ndarray, ThermalShapeFactorsJax]:
    """Model B's fractional light shift `delta_nu_LS/nu_c` (Bothwell et
    al. 2025 Eq. 6), the differentiable counterpart to
    :func:`cliffordclock.integrator.lattice_light_shift.bo_wkb_fractional_light_shift`:

        delta_nu_LS/nu_c ~= -[ (d(alpha~E1)/dnu)*delta_L*X(n_z,u0,Tr)*u0
                              + alpha~M1E2*Y(n_z,u0,Tr)*u0
                              + beta~*Z(n_z,u0,Tr)*u0^2 ]

    with `X`/`Y`/`Z` from :func:`axial_thermal_factors_jax`.
    Differentiable end to end with respect to `u0`, `detuning_hz`,
    `radial_temperature_k`, the three coefficients, and (through
    `E_R`/`kappa`) `waist_m`/`wavelength_m`/`mass_kg`: `jax.grad` of this
    function's first return value with respect to any of these arguments
    traces through the SAME `axial_thermal_factors_jax` computational
    graph, no separate code path.

    **Unit contract, identical to the reference function's:** `coeffs`
    (here, `e1_slope_per_hz`/`m1e2_hz`/`hyperpolarizability_hz`) must be
    in Bothwell's OWN normalization (already per-`nu_c` /
    dimensionless-fraction; `BOTHWELL_2025_YB171_HARMONIC`/`_BOWKB`'s
    convention in the reference module). This is a DIFFERENT convention
    from the Ushijima/Kim `.../h` hertz convention
    `harmonic_light_shift_hz_jax` expects, and this function has no way
    to tell the two apart at the value level: doing so would need a
    species-specific `nu_c`, which no argument here carries.

    This function builds its own :class:`SitePotentialJax` internally
    from `waist_m`/`wavelength_m`/`mass_kg`, taking those three physical
    inputs directly. The reference function instead takes a pre-built
    `site` argument and checks `site.depth_er == u0`, a design that lets
    a caller build the site once and reuse it across many calls; this
    module's fixed-resolution, jit-compiled core has no matching need
    for that reuse, since site construction is a handful of scalar
    arithmetic operations, cheap next to the eigensolves that dominate
    every call regardless.

    Parameters
    ----------
    n_z : int
        Static.
    u0 : jax.Array
        Peak reduced trap depth, `E_R` units. Differentiable.
    detuning_hz, radial_temperature_k : jax.Array
        `delta_L` (hertz), `Tr` (kelvin). Differentiable. `Tr` must be
        `> 0` (see :func:`axial_thermal_factors_jax`'s own validation
        note).
    e1_slope_per_hz, m1e2_hz, hyperpolarizability_hz : jax.Array
        Bothwell-convention coefficients. Differentiable.
    waist_m, wavelength_m, mass_kg : jax.Array
        Site geometry / species. Differentiable.

    Returns
    -------
    tuple[jax.Array, ThermalShapeFactorsJax]
        `(delta_nu_LS/nu_c, the ThermalShapeFactorsJax used)`.
    """
    site = make_site_potential_jax(u0, waist_m, wavelength_m, mass_kg)
    factors = axial_thermal_factors_jax(site, n_z, radial_temperature_k)
    shift = -(
        e1_slope_per_hz * detuning_hz * factors.x_nz * u0
        + m1e2_hz * factors.y_nz * u0
        + hyperpolarizability_hz * factors.z_nz * u0**2
    )
    return shift, factors
