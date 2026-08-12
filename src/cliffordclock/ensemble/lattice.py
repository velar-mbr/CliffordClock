# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: N803
# `temperature_uK` is the CONVENTIONS.md-mandated parameter name at the
# sampler API boundary (docs/CONVENTIONS.md section 10: "temperature uK at
# sampler APIs"); pep8-naming's N803 (lowercase-argument-name) would
# otherwise flag the embedded capital K.
"""Quantum lattice regime: Hermite-Gauss motional-state quadrature.

For atoms held in optical-lattice sites deep enough to be treated as 3D
independent harmonic oscillators, position expectation values are computed
exactly (up to quadrature-order-limited exactness) via Gauss-Hermite
quadrature against the harmonic-oscillator eigenstate probability density,
rather than by classical Monte-Carlo sampling.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import hermite as np_hermite

from cliffordclock import constants
from cliffordclock.ensemble.species import Species
from cliffordclock.ensemble.traps import HarmonicTrap

#: Valid `extended_lattice_nodes` `envelope` values (WP22 Part 2).
VALID_SITE_ENVELOPES: tuple[str, ...] = ("gaussian", "uniform")


def _hermite_physicist(
    n: int, x: np.ndarray[Any, np.dtype[np.float64]]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Evaluate the physicists' Hermite polynomial H_n at points `x`.

    Uses `numpy.polynomial.hermite` (the "physicists'" convention, weight
    function ``exp(-x^2)``), matching the standard quantum-harmonic-
    oscillator wavefunction convention.
    """
    coeffs = np.zeros(n + 1, dtype=np.float64)
    coeffs[n] = 1.0
    result: np.ndarray[Any, np.dtype[np.float64]] = np_hermite.hermval(x, coeffs)
    return result


def _axis_quadrature(
    n: int, n_quad: int, x0: float
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """1D Gauss-Hermite quadrature nodes/weights against ``|psi_n(x)|^2``.

    Standard physicists'-convention Gauss-Hermite quadrature
    ``{xi_i, w_i}`` (`numpy.polynomial.hermite.hermgauss`) integrates
    ``integral P(xi) exp(-xi^2) dxi`` exactly for any polynomial `P` of
    degree <= ``2*n_quad - 1``. The harmonic-oscillator eigenstate density
    is ``|psi_n(x)|^2 dx = H_n(xi)^2 exp(-xi^2) / (2^n n! sqrt(pi)) dxi``
    with ``xi = x / x0``; folding ``H_n(xi)^2`` into the quadrature weights
    below makes ``sum_q w_q f(x_q) = integral f(x) |psi_n(x)|^2 dx`` exact
    whenever `f` (pulled back to `xi`) is a polynomial of degree
    <= ``2*n_quad - 1 - 2*n``.

    Returns
    -------
    nodes : np.ndarray
        Shape ``(n_quad,)``, meters (``x0 * xi_i``).
    weights : np.ndarray
        Shape ``(n_quad,)``, dimensionless probability weights (sum to 1).
    """
    xi, wi = np_hermite.hermgauss(n_quad)
    hn = _hermite_physicist(n, xi)
    norm = 1.0 / (2.0**n * math.factorial(n) * math.sqrt(math.pi))
    weights = wi * hn**2 * norm
    nodes = x0 * xi
    return nodes, weights


def hermite_gaussian_nodes(
    species: Species,
    trap: HarmonicTrap,
    motional_n: tuple[int, int, int],
    n_quad: int,
) -> tuple[jax.Array, jax.Array]:
    """3D tensor-product Gauss-Hermite quadrature against a motional eigenstate density.

    For the harmonic-oscillator eigenstate ``psi_{nx,ny,nz}(r)`` of `trap`
    (frequencies `trap.omega_xyz`) for a particle of mass
    `species.mass_kg`, returns nodes and weights such that
    ``sum_q w_q f(r_q) ~= <psi | f(r_hat) | psi>`` for `f` a polynomial,
    exact up to the quadrature's exactness degree (see `_axis_quadrature`).
    Each axis characteristic length is ``x0_k = sqrt(hbar / (m omega_k))``.

    Parameters
    ----------
    species : Species
        Atomic species (supplies mass).
    trap : HarmonicTrap
        Harmonic trap (supplies per-axis angular frequencies and center).
    motional_n : tuple[int, int, int]
        Motional quantum numbers ``(nx, ny, nz)``.
    n_quad : int
        Number of 1D Gauss-Hermite quadrature points per axis. The
        resulting quadrature is exact for polynomials of per-axis degree
        <= ``2*n_quad - 1 - 2*n_k``.

    Returns
    -------
    nodes : jax.Array
        Shape ``(n_quad**3, 3)``, meters, dtype float64.
    weights : jax.Array
        Shape ``(n_quad**3,)``, dimensionless, dtype float64, summing to 1.
    """
    omega = np.asarray(trap.omega_xyz, dtype=np.float64)
    center = np.asarray(trap.center, dtype=np.float64)
    x0 = np.sqrt(constants.HBAR / (species.mass_kg * omega))

    axis_nodes: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    axis_weights: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for k in range(3):
        nodes_k, weights_k = _axis_quadrature(motional_n[k], n_quad, float(x0[k]))
        axis_nodes.append(nodes_k)
        axis_weights.append(weights_k)

    grid_x, grid_y, grid_z = np.meshgrid(*axis_nodes, indexing="ij")
    wx, wy, wz = np.meshgrid(*axis_weights, indexing="ij")

    nodes = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1) + center
    weights = (wx * wy * wz).ravel()

    return jnp.asarray(nodes, dtype=jnp.float64), jnp.asarray(weights, dtype=jnp.float64)


class ExtendedLatticeGeometry(NamedTuple):
    """The site-and-motional geometry an extended-lattice ensemble
    resolves to (WP22 Part 2, `cliffordclock.pipeline.EnsembleConfig`
    `regime="lattice_extended"`).

    Attributes
    ----------
    nodes : jax.Array, shape (n_sites * n_local, 3)
        Every site's local Hermite-Gauss quadrature nodes, concatenated
        site-major (site 0's `n_local` nodes first, then site 1's, ...) --
        i.e. ``nodes.reshape(n_sites, n_local, 3)`` recovers the per-site
        grouping, meters.
    weights : jax.Array, shape (n_sites * n_local,)
        ``site_weights[s] * local_weights[q]`` for each `(s, q)` pair,
        flattened in the same site-major order as `nodes`; sums to 1
        exactly (both factors individually sum to 1, so the product does
        too -- see :func:`extended_lattice_nodes`).
    site_centers : jax.Array, shape (n_sites, 3)
        Each site's trap center, meters.
    site_weights : jax.Array, shape (n_sites,)
        Normalized (sum-to-1) site-occupation envelope weight.
    site_offsets_m : jax.Array, shape (n_sites,)
        Signed coordinate of each site along the configured `site_axis`
        (unit-normalized internally), relative to `trap.center`, meters --
        the natural x-axis for a per-site linear-gradient fit (WP22 Part
        2 gate edit 4's "best-fit linear gradient along the axis").
    local_weights : jax.Array, shape (n_local,)
        The SAME per-site Hermite-Gauss quadrature weights for every site
        (every site shares one trap/motional-state configuration by
        construction, CONVENTIONS.md section 15 Part 2 item 1); sums to 1.
        Kept separately (not just folded into `weights`) so a caller can
        recover each site's own weighted mean
        (``nodes.reshape(n_sites, n_local, ...) @ local_weights``) without
        re-deriving it from the flattened, envelope-scaled `weights`.
    """

    nodes: jax.Array
    weights: jax.Array
    site_centers: jax.Array
    site_weights: jax.Array
    site_offsets_m: jax.Array
    local_weights: jax.Array


def extended_lattice_nodes(
    species: Species,
    trap: HarmonicTrap,
    motional_n: tuple[int, int, int],
    n_quad: int,
    n_sites: int,
    site_spacing_m: float,
    site_axis: tuple[float, float, float],
    envelope: str,
    envelope_sigma_m: float | None,
) -> ExtendedLatticeGeometry:
    """Extended-lattice site geometry: `n_sites` copies of
    :func:`hermite_gaussian_nodes`'s single-site quadrature, distributed
    along a configured axis with a Gaussian-or-uniform occupation envelope
    (WP22 Part 2, CONVENTIONS.md section 15).

    Each site is an independent copy of the SAME local trap/motional
    configuration (`trap.omega_xyz`, `motional_n`, `n_quad`) -- i.e. this
    models an extended sample of otherwise-identical lattice sites (the
    Bothwell 2022 magic-wavelength-lattice geometry this WP targets: every
    site the same depth/frequency, differing only in position), NOT a
    site-dependent trap. Implemented as exactly ONE
    :func:`hermite_gaussian_nodes` call (against a `center=(0,0,0)`
    template trap) broadcast to every site's own center, rather than
    `n_sites` separate calls -- both cheaper and, since every site's local
    quadrature is then bit-for-bit the same array (only translated), makes
    `ExtendedLatticeGeometry.local_weights` exactly shared/reusable by
    every site (see that field's docstring), not merely numerically close
    across sites.

    Site centers are evenly spaced at `site_spacing_m` along `site_axis`
    (unit-normalized internally), symmetric about `trap.center`:
    ``offset_i = (i − (n_sites−1)/2) · site_spacing_m`` for
    ``i = 0, ..., n_sites−1`` (so `trap.center` is always the envelope's
    geometric center, whatever `n_sites`'s parity).

    Parameters
    ----------
    species : Species
        Atomic species (supplies mass, via :func:`hermite_gaussian_nodes`).
    trap : HarmonicTrap
        The (identical, per-site) local trap; `trap.center` is the
        envelope's geometric center.
    motional_n : tuple[int, int, int]
        Motional quantum numbers, applied identically at every site (see
        function docstring).
    n_quad : int
        Gauss-Hermite quadrature points per axis, per site.
    n_sites : int
        Number of sites along `site_axis`. Must be `>= 1`.
    site_spacing_m : float
        Center-to-center spacing between adjacent sites, meters. Must be
        `> 0`.
    site_axis : tuple[float, float, float]
        Direction along which sites are distributed; need not be
        pre-normalized (normalized internally). Must not be the zero
        vector.
    envelope : str
        `"gaussian"`: site weight `\\propto exp(-0.5*(offset/envelope_sigma_m)**2)`
        (requires `envelope_sigma_m`). `"uniform"`: every site equally
        weighted. See :data:`VALID_SITE_ENVELOPES`.
    envelope_sigma_m : float or None
        Gaussian envelope standard deviation, meters; required (and must
        be `> 0`) when `envelope="gaussian"`, ignored otherwise.

    Returns
    -------
    ExtendedLatticeGeometry

    Raises
    ------
    ValueError
        `n_sites < 1`, `site_spacing_m <= 0`, `site_axis` is the zero
        vector, `envelope` is not in :data:`VALID_SITE_ENVELOPES`, or
        `envelope="gaussian"` with `envelope_sigma_m` missing/non-positive.
    """
    if n_sites < 1:
        raise ValueError(f"extended_lattice_nodes: n_sites must be >= 1, got {n_sites}")
    if site_spacing_m <= 0.0:
        raise ValueError(
            f"extended_lattice_nodes: site_spacing_m must be > 0, got {site_spacing_m!r}"
        )
    axis = np.asarray(site_axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        raise ValueError("extended_lattice_nodes: site_axis must not be the zero vector")
    axis_hat = axis / axis_norm

    idx = np.arange(n_sites, dtype=np.float64)
    offsets_m = (idx - (n_sites - 1) / 2.0) * site_spacing_m  # (n_sites,), symmetric about 0

    if envelope == "gaussian":
        if envelope_sigma_m is None or envelope_sigma_m <= 0.0:
            raise ValueError(
                "extended_lattice_nodes: envelope='gaussian' requires envelope_sigma_m > 0, "
                f"got {envelope_sigma_m!r}"
            )
        raw_weights = np.exp(-0.5 * (offsets_m / envelope_sigma_m) ** 2)
    elif envelope == "uniform":
        raw_weights = np.ones(n_sites, dtype=np.float64)
    else:
        raise ValueError(
            f"extended_lattice_nodes: envelope must be one of {VALID_SITE_ENVELOPES}, "
            f"got {envelope!r}"
        )
    site_weights_np = raw_weights / np.sum(raw_weights)

    center = np.asarray(trap.center, dtype=np.float64)
    site_centers_np = center[None, :] + offsets_m[:, None] * axis_hat[None, :]

    local_trap = HarmonicTrap(omega_xyz=trap.omega_xyz, center=(0.0, 0.0, 0.0))
    local_nodes, local_weights = hermite_gaussian_nodes(species, local_trap, motional_n, n_quad)

    site_centers = jnp.asarray(site_centers_np, dtype=jnp.float64)
    site_weights = jnp.asarray(site_weights_np, dtype=jnp.float64)
    nodes = (site_centers[:, None, :] + local_nodes[None, :, :]).reshape(-1, 3)
    weights = (site_weights[:, None] * local_weights[None, :]).reshape(-1)

    return ExtendedLatticeGeometry(
        nodes=nodes,
        weights=weights,
        site_centers=site_centers,
        site_weights=site_weights,
        site_offsets_m=jnp.asarray(offsets_m, dtype=jnp.float64),
        local_weights=local_weights,
    )


def thermal_occupation(temperature_uK: float, trap: HarmonicTrap) -> jax.Array:
    """Mean Bose-Einstein motional occupation number per trap axis.

    ``n_bar(omega) = 1 / (exp(hbar omega / (k_B T)) - 1)``, the thermal
    equilibrium mean phonon number of a quantum harmonic oscillator mode
    of angular frequency `omega` at temperature `T` (independent of
    particle mass). A helper for choosing representative `motional_n`
    values to pass to `hermite_gaussian_nodes`.

    Parameters
    ----------
    temperature_uK : float
        Ensemble temperature, microkelvin.
    trap : HarmonicTrap
        Harmonic trap (supplies per-axis angular frequencies).

    Returns
    -------
    jax.Array
        Shape ``(3,)``, dimensionless mean occupation number per axis,
        dtype float64.
    """
    temperature_k = temperature_uK * 1e-6
    omega = jnp.asarray(trap.omega_xyz, dtype=jnp.float64)
    exponent = constants.HBAR * omega / (constants.BOLTZMANN_K * temperature_k)
    return 1.0 / jnp.expm1(exponent)
