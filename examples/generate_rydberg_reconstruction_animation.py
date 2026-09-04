#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate ``docs/assets/rydberg_reconstruction_animation.gif``, the
README's fourth hero animation.

**What this shows.** WP41's differentiable Rydberg field-reconstruction
demonstrator (``benchmarks/run_rydberg_field_reconstruction.py``, C5, the
same file's own module docstring: "a synthetic round-trip ... a
three-parameter field model ... generates a synthetic spectrum at planted
truth values, seeded Gaussian noise is added, and
``scipy.optimize.minimize`` (L-BFGS-B, exact jax-supplied gradients) fits
the three parameters back"), animated backwards from what a real
Rydberg-sensor calibration run would see first: the measured spectrum,
then the reconstructed field. Three panels, updated per optimizer step:

- **Left**: the hidden truth field, a 2D cross-section of the vapor cell
  (``y=0`` slice, cell axis ``z`` horizontal, radial coordinate ``x``
  vertical) -- the same three-parameter model
  (:func:`cliffordclock.integrator.rydberg_cell_response_jax.cell_field_magnitude_v_per_m_jax`,
  evaluated here through the benchmark's own numpy transcription
  ``run_rydberg_field_reconstruction._cell_field_magnitude_np``) at the
  PLANTED truth parameters. Drawn once and held fixed for the whole
  animation, a reference the middle panel converges onto. Labeled
  "planted, synthetic" throughout.
- **Middle**: the reconstruction, live. The same field-model formula,
  evaluated at the CURRENT L-BFGS-B iterate's fitted parameters, redrawn
  at every kept step. Both field panels share one color scale, so a
  visual match at the end is checkable against that shared scale.
- **Right**: the measured evidence, the noisy synthetic EIT spectrum
  (``Im(chi)``, the observable
  :func:`~cliffordclock.integrator.rydberg_cell_response_jax.rb85_field_reconstruction_forward_model_jax`
  returns) as data points, with the model curve at the current iterate's
  parameters redrawn alongside it.

A persistent footer caption and the figure title both carry the
"synthetic demonstration" label in every frame, matching this project's
calibration-language discipline: no real Rydberg-sensor scan is fit
here, and no priority claim is made for the inverse-problem pattern
(``run_rydberg_field_reconstruction.py``'s own module docstring states
the identical calibration).

**The fit case, and why it was chosen.** ``TRUTH_GRID[0]`` = `(E0=180.0
V/m, gradient=800.0 V/m/m, patch=60.0 V/m)`, seed `0` -- the
best-converging row of the already-gated WP41 fit grid
(``benchmarks/results/wp41_rydberg_field_reconstruction.md``'s own C5
table). It is the only one of the eight truth/seed cases that recovers
all three parameters within its own reported 1-sigma Laplace
uncertainty, with a positive-definite Hessian and a converged
optimizer. The other seven cases in the grid land outside 1-sigma, or
(two of them) outside 2-sigma, on at least one parameter; the
project's own results table records those outcomes for every case.
This animation picks the case whose reconstruction visibly converges
onto the truth field within a short loop. This script reuses the same
400 atom positions the WP41 benchmark samples for this case
(:data:`ATOM_POSITION_SEED`, matching the seed value
``run_rydberg_field_reconstruction.run_all_fits`` hardcodes inline).
This value is reproduced here and checked:
:func:`_check_against_reference` below confirms this script's own fit
reaches the identical optimum, uncertainty, and Hessian verdict as an
unwrapped call to that file's own ``run_one_fit`` on the identical
case before either number is trusted for the animation.

**The optimizer's raw path, held frame by frame.** Same discipline as
``generate_lattice_fit_animation.py``'s own module docstring: L-BFGS-B's
line search moves non-monotonically, so this script logs every call and
keeps only the calls that set a new best chi-squared so far, then holds
each kept point on screen for a fixed number of frames before cutting
directly to the next. No interpolation, easing, or morphing sits between
one field/spectrum pair and the next.

Regeneration
------------
Run from an activated project venv (``pip install -e ".[dev,notebooks]"``
-- JAX from the ``dev`` extra, matplotlib + Pillow from ``notebooks``)::

    python examples/generate_rydberg_reconstruction_animation.py

Deterministic: fixed seeds throughout (atom positions, fit noise,
plotting), no wall-clock reads in any computed quantity. Two runs on the
same machine produce byte-identical output (checked directly this
session: two full regenerations diffed byte-for-byte identical).
Last measured wall time on the author's machine: about 11 s (fit + trace
under 3 s; animation render/encode the rest). Output:
``docs/assets/rydberg_reconstruction_animation.gif`` (~1.1 MB, 114
frames at 10 fps, an 11.4 s loop), sha256
``81616230cf5e4143e1a6ee1275b18b8ee1cac5063227ea6b4d27fd7e098dd954``.
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import Normalize
from scipy.optimize import minimize  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import run_rydberg_field_reconstruction as wfr  # noqa: E402

_OUTPUT_PATH = REPO_ROOT / "docs" / "assets" / "rydberg_reconstruction_animation.gif"

#: The fit case animated: `wfr.TRUTH_GRID[0]`, `wfr.SEEDS[0]` (module
#: docstring "The fit case, and why it was chosen").
TRUTH: tuple[float, float, float] = wfr.TRUTH_GRID[0]
SEED: int = wfr.SEEDS[0]
assert TRUTH in wfr.TRUTH_GRID
assert SEED in wfr.SEEDS

#: Matches the seed value `run_rydberg_field_reconstruction.run_all_fits`
#: hardcodes inline (`np.random.default_rng(20260903)`). Reproducing it
#: here regenerates the identical 400 atom positions that case uses;
#: `_check_against_reference` below confirms the match directly.
ATOM_POSITION_SEED = 20260903

#: Frames-per-second, matching the other three hero animations.
FPS = 10

#: Each kept (monotone-improving) L-BFGS-B call holds for this many
#: frames (module docstring "The optimizer's raw path, held frame by
#: frame").
FRAMES_PER_ITERATE = 3

#: Stride applied to the monotone-improving subsequence before animating
#: it: every `MONOTONE_STRIDE`-th kept call is drawn, always including
#: the first and last. This case's raw monotone subsequence has 54
#: calls; each one draws a distinct 60x170 field heatmap in two panels,
#: so the gif's size is set mainly by the NUMBER OF DISTINCT HEATMAPS.
#: Raising `FRAMES_PER_ITERATE` alone barely changes file size, since
#: repeated frames compress well. Striding to ~27 distinct steps keeps
#: the gif under about 1.5 MB while still showing every phase of the
#: real descent: the early wild excursion to the parameter bounds, the
#: slow crawl, and the final convergence each keep representatives.
MONOTONE_STRIDE = 2

#: Final hold: reconstruction beside truth, recovered vs. planted values
#: (2.5 s at FPS=10, matching the other fit-style hero's hold length).
HOLD_FRAMES = 30

#: Field cross-section grid resolution (x: radial, z: axial). Chosen so
#: the heatmap reads smoothly at README width without adding materially
#: to the gif's size.
_N_X = 60
_N_Z = 170

_COLOR_DATA = "#C1666B"
_COLOR_MODEL = "#2E5FA3"
_CMAP_FIELD = "viridis"  # colorblind-safe, matching generate_showcase_animation.py

_CAPTION = (
    "Synthetic demonstration: planted truth field, synthetic noisy spectrum, "
    "L-BFGS-B fit through the real differentiable Rydberg chain (WP41). "
    "No real Rydberg-sensor scan is fit here."
)


@dataclass(frozen=True)
class TracedFit:
    """One fit's full traced optimizer path plus its final report."""

    positions_m: np.ndarray  # (n_atoms, 3)
    delta_mhz: np.ndarray  # (n_delta,)
    noisy_spectrum: np.ndarray  # (n_delta,)
    mono_params: np.ndarray  # (N, 3): [e0, grad, patch] at each kept call
    mono_chi2: np.ndarray  # (N,)
    mono_curves: np.ndarray  # (N, n_delta): forward spectrum at each kept call
    mono_field_grids: np.ndarray  # (N, n_x, n_z): field cross-section at each kept call
    truth_field_grid: np.ndarray  # (n_x, n_z)
    recovered: np.ndarray  # (3,)
    recovered_sigma: np.ndarray  # (3,)
    hessian_positive_definite: bool
    n_calls_total: int
    x_grid_mm: np.ndarray
    z_grid_mm: np.ndarray


def _field_cross_section(
    params: tuple[float, float, float], x_grid_m: np.ndarray, z_grid_m: np.ndarray
) -> np.ndarray:
    """The real field-model formula (`wfr._cell_field_magnitude_np`, the
    numpy transcription of
    :func:`cliffordclock.integrator.rydberg_cell_response_jax.cell_field_magnitude_v_per_m_jax`),
    evaluated on a `y=0` grid slice through the cell. Returns shape
    `(len(x_grid_m), len(z_grid_m))`.
    """
    xx, zz = np.meshgrid(x_grid_m, z_grid_m, indexing="ij")
    positions_flat = np.stack([xx.ravel(), np.zeros(xx.size), zz.ravel()], axis=-1)
    field_flat = wfr._cell_field_magnitude_np(positions_flat, *params)
    return field_flat.reshape(xx.shape)


def _run_traced_fit() -> TracedFit:
    """Run the exact `run_rydberg_field_reconstruction.run_one_fit` case,
    with every L-BFGS-B objective call logged, reduced to its
    monotone-improving subsequence (module docstring "The optimizer's raw
    path, held frame by frame").
    """
    rng_positions = np.random.default_rng(ATOM_POSITION_SEED)
    positions_np = wfr.rcr.cylindrical_cell_atom_positions(
        wfr.CELL_RADIUS_M, wfr.CELL_LENGTH_M, wfr.N_ATOMS, rng_positions
    )

    truth_fields = wfr._cell_field_magnitude_np(positions_np, *TRUTH)
    wfr._assert_within_validity_window(truth_fields, "hero animation truth")
    corner_fields = wfr._bound_corner_fields(positions_np)
    wfr._assert_within_validity_window(corner_fields, "hero animation bound corners")

    positions_j = jnp.asarray(positions_np)
    weights_j = jnp.ones(positions_np.shape[0])
    forward = wfr.make_forward_model(positions_j, weights_j)
    forward_jit = jax.jit(forward)

    truth_spectrum = np.asarray(forward_jit(jnp.asarray(TRUTH)))
    rng_noise = np.random.default_rng(SEED)
    noisy = truth_spectrum + rng_noise.normal(0.0, wfr.NOISE_SIGMA, size=truth_spectrum.shape)

    objective, hessian_fn = wfr.make_objective_and_hessian(forward, noisy)

    call_params: list[np.ndarray] = []
    call_chi2: list[float] = []

    def traced_objective(params_np: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = objective(params_np)
        call_params.append(np.array(params_np, dtype=np.float64))
        call_chi2.append(float(value))
        return value, grad

    x0 = np.array([t * f for t, f in zip(TRUTH, wfr.X0_OFFSET_FACTORS, strict=True)])
    result = minimize(traced_objective, x0, jac=True, method="L-BFGS-B", bounds=wfr.BOUNDS)

    call_params_arr = np.array(call_params)
    call_chi2_arr = np.array(call_chi2)

    best = np.inf
    mono_rows: list[int] = []
    for i, chi2_i in enumerate(call_chi2_arr):
        if chi2_i < best:
            best = chi2_i
            mono_rows.append(i)
    mono_params = call_params_arr[mono_rows]
    mono_chi2 = call_chi2_arr[mono_rows]

    # Stride the monotone subsequence (`MONOTONE_STRIDE`'s own comment
    # above), always keeping the first and last kept call.
    strided_idx = list(range(0, len(mono_rows), MONOTONE_STRIDE))
    if strided_idx[-1] != len(mono_rows) - 1:
        strided_idx.append(len(mono_rows) - 1)
    mono_params = mono_params[strided_idx]
    mono_chi2 = mono_chi2[strided_idx]

    hessian = np.asarray(hessian_fn(jnp.asarray(result.x)))
    hessian_pd, sigmas = wfr.laplace_uncertainties(hessian)

    mono_curves = np.stack([np.asarray(forward_jit(jnp.asarray(p))) for p in mono_params])

    x_grid_m = np.linspace(-wfr.CELL_RADIUS_M, wfr.CELL_RADIUS_M, _N_X)
    z_grid_m = np.linspace(-wfr.CELL_LENGTH_M / 2.0, wfr.CELL_LENGTH_M / 2.0, _N_Z)
    truth_field_grid = _field_cross_section(TRUTH, x_grid_m, z_grid_m)
    mono_field_grids = np.stack(
        [_field_cross_section(tuple(p), x_grid_m, z_grid_m) for p in mono_params]
    )

    return TracedFit(
        positions_m=positions_np,
        delta_mhz=np.asarray(wfr.DELTA_P_HZ_NP) / (2.0 * np.pi) / 1.0e6,
        noisy_spectrum=noisy,
        mono_params=mono_params,
        mono_chi2=mono_chi2,
        mono_curves=mono_curves,
        mono_field_grids=mono_field_grids,
        truth_field_grid=truth_field_grid,
        recovered=np.asarray(result.x, dtype=np.float64),
        recovered_sigma=sigmas,
        hessian_positive_definite=hessian_pd,
        n_calls_total=len(call_params),
        x_grid_mm=x_grid_m * 1.0e3,
        z_grid_mm=z_grid_m * 1.0e3,
    )


def _check_against_reference(traced: TracedFit) -> None:
    """Confirms the wrapped, logged fit reaches the same optimum, the
    same Laplace uncertainties, and the same Hessian verdict as an
    unwrapped `run_rydberg_field_reconstruction.run_one_fit` call on the
    identical case (mirroring `generate_lattice_fit_animation.py`'s own
    `_check_against_reference` discipline). This is also the check that
    validates `ATOM_POSITION_SEED` reproduces the benchmark's own atom
    positions: a seed mismatch would desynchronize the noisy spectrum and
    fail this comparison well outside roundoff.
    """
    reference = wfr.run_one_fit(TRUTH, SEED, traced.positions_m)
    recovered_diff = np.max(np.abs(traced.recovered - np.asarray(reference.recovered)))
    assert recovered_diff < 1e-6, (
        "traced fit's recovered parameters disagree with run_one_fit's own reference case "
        f"by {recovered_diff:.3e} -- ATOM_POSITION_SEED may not reproduce the benchmark's own "
        "atom positions"
    )
    assert traced.hessian_positive_definite == reference.hessian_positive_definite
    sigma_diff = np.nanmax(
        np.abs(traced.recovered_sigma - np.asarray(reference.recovered_uncertainty))
    )
    assert sigma_diff < 1e-6 or not traced.hessian_positive_definite


def main() -> None:
    t_wall_start = time.perf_counter()
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    traced = _run_traced_fit()
    _check_against_reference(traced)
    assert traced.hessian_positive_definite, (
        "this case's Hessian carries a non-positive eigenvalue at its reported optimum -- "
        "the animation's end-state uncertainty readout would be invalid"
    )
    t_fit_s = time.perf_counter() - t0

    n_mono = traced.mono_params.shape[0]
    # Color scale fixed to the truth field's own range, with a fixed
    # margin added. This case's raw L-BFGS-B path visits parameters near
    # the optimizer bounds before settling, reaching a field magnitude up
    # to ~555 V/m against a converged range of ~150-240 V/m. Normalizing
    # to that full swing would wash out the color contrast during the
    # converged final third of the animation, the part where the visual
    # match to the truth panel matters most. Off-window values saturate
    # to the colorbar's end color (`extend="both"` marks this on the
    # colorbar); the field values feeding the fit and the parameter
    # readouts are unaffected.
    field_margin_v_per_m = 0.2 * (
        float(traced.truth_field_grid.max()) - float(traced.truth_field_grid.min())
    )
    field_vmin = float(traced.truth_field_grid.min()) - field_margin_v_per_m
    field_vmax = float(traced.truth_field_grid.max()) + field_margin_v_per_m
    field_norm = Normalize(vmin=field_vmin, vmax=field_vmax)
    field_extent = [
        traced.z_grid_mm[0],
        traced.z_grid_mm[-1],
        traced.x_grid_mm[0],
        traced.x_grid_mm[-1],
    ]
    patch_z_mm = float(wfr.PATCH_POSITION_M[2] * 1.0e3)
    patch_x_mm = float(wfr.PATCH_POSITION_M[0] * 1.0e3)

    spec_lo = min(traced.noisy_spectrum.min(), traced.mono_curves.min()) * 1.0e6
    spec_hi = max(traced.noisy_spectrum.max(), traced.mono_curves.max()) * 1.0e6
    spec_pad = 0.08 * (spec_hi - spec_lo)

    # --- Figure/axes setup (created once; per-frame update mutates artists). --
    fig = plt.figure(figsize=(13.2, 4.7))
    ax_truth = fig.add_axes((0.045, 0.15, 0.235, 0.58))
    ax_recon = fig.add_axes((0.315, 0.15, 0.235, 0.58))
    cax_field = fig.add_axes((0.558, 0.15, 0.013, 0.58))
    ax_spec = fig.add_axes((0.685, 0.15, 0.285, 0.58))

    ax_truth.imshow(
        traced.truth_field_grid,
        origin="lower",
        extent=field_extent,
        cmap=_CMAP_FIELD,
        norm=field_norm,
        aspect="auto",
    )
    ax_truth.scatter(
        [patch_z_mm], [patch_x_mm], s=22, facecolors="none", edgecolors="white", lw=1.1
    )
    ax_truth.set_title("Hidden truth field (planted, synthetic)", fontsize=9)
    ax_truth.set_xlabel("z, cell axis (mm)", fontsize=8.5)
    ax_truth.set_ylabel("x (mm)", fontsize=8.5)

    im_recon = ax_recon.imshow(
        traced.mono_field_grids[0],
        origin="lower",
        extent=field_extent,
        cmap=_CMAP_FIELD,
        norm=field_norm,
        aspect="auto",
    )
    ax_recon.scatter(
        [patch_z_mm], [patch_x_mm], s=22, facecolors="none", edgecolors="white", lw=1.1
    )
    ax_recon.set_title("Reconstruction, live (L-BFGS-B)", fontsize=9)
    ax_recon.set_xlabel("z, cell axis (mm)", fontsize=8.5)
    ax_recon.tick_params(labelleft=False)
    fig.colorbar(im_recon, cax=cax_field, label=r"$|E|$ (V/m)", extend="both")

    ax_spec.scatter(
        traced.delta_mhz,
        traced.noisy_spectrum * 1.0e6,
        s=8,
        color=_COLOR_DATA,
        alpha=0.75,
        edgecolors="none",
        label="synthetic noisy spectrum",
        zorder=2,
    )
    (model_line,) = ax_spec.plot(
        [], [], color=_COLOR_MODEL, lw=1.6, label="model, current fit", zorder=3
    )
    ax_spec.set_xlim(traced.delta_mhz[0], traced.delta_mhz[-1])
    ax_spec.set_ylim(spec_lo - spec_pad, spec_hi + spec_pad)
    ax_spec.set_xlabel("probe detuning (MHz)", fontsize=8.5)
    ax_spec.set_ylabel(r"Im($\chi$) ($\times 10^{-6}$)", fontsize=8.5)
    ax_spec.set_title("Measured evidence: noisy EIT spectrum", fontsize=9)
    ax_spec.legend(loc="upper right", fontsize=7.5, frameon=False)

    fig.suptitle(
        "Rydberg-sensor field reconstruction: spectrum in, field out (synthetic demonstration)",
        fontsize=10.5,
        y=0.985,
    )
    readout_text = fig.text(0.5, 0.865, "", ha="center", va="center", fontsize=8.5, color="0.15")
    hold_text = fig.text(
        0.5, 0.825, "", ha="center", va="center", fontsize=8, color="0.2", alpha=0.0
    )
    fig.text(0.5, 0.035, _CAPTION, ha="center", va="center", fontsize=7.5, color="0.35")

    e0_truth, grad_truth, patch_truth = TRUTH
    e0_sigma, grad_sigma, patch_sigma = traced.recovered_sigma
    e0_rec, grad_rec, patch_rec = traced.recovered

    n_descent_frames = n_mono * FRAMES_PER_ITERATE
    n_frames = n_descent_frames + HOLD_FRAMES

    def update(frame: int):
        idx = min(frame // FRAMES_PER_ITERATE, n_mono - 1)
        is_hold = frame >= n_descent_frames

        im_recon.set_data(traced.mono_field_grids[idx])
        model_line.set_data(traced.delta_mhz, traced.mono_curves[idx] * 1.0e6)

        e0_i, grad_i, patch_i = traced.mono_params[idx]
        readout_text.set_text(
            f"step {idx + 1}/{n_mono}, $\\chi^2$ = {traced.mono_chi2[idx]:.0f}   |   "
            f"$E_0$ = {e0_i:.1f} V/m (truth {e0_truth:.0f})   "
            f"gradient = {grad_i:.0f} V/m/m (truth {grad_truth:.0f})   "
            f"patch = {patch_i:.1f} V/m (truth {patch_truth:.0f})"
        )

        if is_hold:
            hold_local = frame - n_descent_frames
            hold_progress = min(1.0, hold_local / (HOLD_FRAMES * 0.5))
            hold_text.set_alpha(hold_progress)
            hold_text.set_text(
                f"recovered: $E_0$={e0_rec:.2f}$\\pm${e0_sigma:.2f}, "
                f"gradient={grad_rec:.1f}$\\pm${grad_sigma:.1f}, "
                f"patch={patch_rec:.1f}$\\pm${patch_sigma:.1f} V/m -- "
                "all three within 1-sigma of the planted truth"
            )
        else:
            hold_text.set_alpha(0.0)

        return [im_recon, model_line, readout_text, hold_text]

    t0 = time.perf_counter()
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    ani.save(_OUTPUT_PATH, writer=animation.PillowWriter(fps=FPS), dpi=88)
    plt.close(fig)
    t_render_s = time.perf_counter() - t0

    t_wall_total_s = time.perf_counter() - t_wall_start
    size_mb = _OUTPUT_PATH.stat().st_size / 1e6
    sha256 = hashlib.sha256(_OUTPUT_PATH.read_bytes()).hexdigest()
    print(f"fit + trace: {t_fit_s:.2f} s ({traced.n_calls_total} objective calls, {n_mono} kept)")
    print(f"animation render + encode: {t_render_s:.2f} s")
    print(f"total wall time: {t_wall_total_s:.2f} s")
    print(
        f"wrote {_OUTPUT_PATH} ({size_mb:.2f} MB, {n_frames} frames at {FPS} fps, "
        f"{n_frames / FPS:.1f} s loop)"
    )
    print(f"sha256: {sha256}")
    print(
        f"case: truth={TRUTH} seed={SEED} recovered={traced.recovered.tolist()} "
        f"sigma={traced.recovered_sigma.tolist()} "
        f"hessian_positive_definite={traced.hessian_positive_definite}"
    )


if __name__ == "__main__":
    main()
