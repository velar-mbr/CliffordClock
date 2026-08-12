#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figure 4 (showcase): a chamber-scale field's inhomogeneous shift budget.

The pitch this figure exists to make: from a CAD/FEA-exported field with
real spatial structure to the full inhomogeneous-shift budget of an atomic
cloud -- mean shift, spread, T2*, and line profile -- computed by the
*real* pipeline, in both its trajectory (scalar) evaluation mode and the
Cl(1,3) rotor engine, on identical Monte Carlo trajectories. Both modes
agree, which is itself part of the message (Sec. "Showcase" of the paper):
the scalar path is not an approximation to some more-correct rotor
calculation here, and the rotor engine is not required to get this right
-- but the two are directly, quantitatively cross-checked on this exact
scenario, not just on the simpler cases in Table
:ref:`tab:validation`/``tests/test_integrator_stark_rotor.py``.

**What this script runs.**

1. Loads ``examples/showcase_gradient_dispersion_sr87.yaml`` (Sr-87,
   ``ensemble.regime: classical``, ``coupling.type: stark_dc``, the field
   ``examples/showcase_field.txt`` -- see
   ``examples/generate_showcase_field.py``'s module docstring for the
   field's physics: two asymmetric electrodes plus a patch-potential wall
   spot, solved from scratch by finite differences, no COMSOL) and runs it
   through :func:`cliffordclock.pipeline.run_pipeline_full` exactly as the
   ``cliffordclock`` CLI would -- the scalar **trajectory mode**
   (``integration.mode: direct``, the classical-regime default): a real
   Maxwell-Boltzmann-sampled, velocity-Verlet-propagated Monte Carlo
   ensemble accumulating the E14b DC-Stark phase along each atom's actual
   motion through the fitted field.
2. Takes the *exact* Monte Carlo trajectories that run produced
   (``PipelineResult.trajectories``) and re-accumulates the identical
   scenario through the true Cl(1,3) rotor path
   (:func:`cliffordclock.pipeline._stark_rotor_ensemble` -- the same
   accumulator ``integration.mode: worldline`` uses for
   ``coupling.type: stark_dc``, here driven directly because that mode is
   only wired up for static lattice nodes in the shipped pipeline, not a
   moving classical trajectory; see the YAML config's own header comment).
   This is not a second, independent simulation: it is the *same*
   trajectories, the *same* field, the *same* species -- only the
   accumulator (scalar rate-function summation vs. the full rotor
   exponential-map integrator) differs, which is exactly what makes the
   comparison a direct, non-vacuous cross-check rather than two unrelated
   numbers that happen to be close.
3. Builds both evaluations' :class:`~cliffordclock.analytics.MetrologyReport`
   (mean shift, SEM, T2*) via the same
   :func:`~cliffordclock.analytics.build_report` the pipeline itself calls,
   and both coherence functions / line profiles via
   :func:`~cliffordclock.analytics.coherence_function` /
   :func:`~cliffordclock.analytics.line_profile`.
4. Writes ``generated/showcase_values.tex`` (every number the paper's
   showcase section quotes) and ``figures/fig4_showcase_gradient_dispersion.pdf``
   (three panels: chamber-scale field context with electrode/patch
   geometry, a zoomed local field with a trajectory subsample colored by
   accumulated shift, and the per-atom shift distribution + line profile
   with T2* annotated).

**Ensemble sizing** (why the cloud is chamber-scale, not lattice-site
scale): see ``examples/showcase_gradient_dispersion_sr87.yaml``'s header
comment and ``examples/generate_showcase_field.py``'s module docstring.
Short version: a real lattice-site cloud (tens of nanometers,
``docs/timescales.md``) never samples genuine chamber-scale field
curvature; this scenario instead scales a classical, chamber-scale
trapping stage's temperature/trap frequency so its cloud sigma
(~692 um) is a real, meaningful fraction of the field's own spatial
variation scale -- a scenario-*geometry* choice, never a change to any
physical constant.

**Memory-safety note (binding).** ``examples/showcase_gradient_dispersion_sr87.yaml``
pins ``integration.dtau``/``steps`` explicitly rather than auto-selecting
them from ``integration.time_s`` -- see that file's own memory-safety
comment for why (an earlier, unmonitored background run at a longer
auto-selected interrogation window and a larger field-fit point count
drove the field-evaluation cost far past available memory). This script
asserts the same compute-budget bounds at run time
(:data:`_MAX_TRAJECTORY_BYTES`, :data:`_MAX_FIELD_EVAL_QUERY_FIT_PAIRS`)
before running anything expensive, so a future parameter change that
would silently reintroduce the same failure mode fails loudly and early
instead.
"""

# Note: this module has a few matplotlib-stub typing nits (e.g. `subplots`
# unpacking, `imshow`/`ScalarMappable` return types) left unfixed -- CI's
# `mypy` invocation scopes to `src/` only (`.github/workflows/ci.yml`), so
# `paper/figures/` is not type-checked and these are not build-breaking.

from __future__ import annotations

import sys
import time

import common  # noqa: E402  (sets up sys.path; must be imported first)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.patches import Circle, Rectangle  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "examples"))
import generate_showcase_field as showcase_gen  # noqa: E402

from cliffordclock.analytics import build_report, coherence_function, line_profile  # noqa: E402
from cliffordclock.constants import BOLTZMANN_K  # noqa: E402
from cliffordclock.ensemble.species import get_species  # noqa: E402
from cliffordclock.pipeline import (  # noqa: E402
    PipelineConfig,
    _auto_renorm_every,
    _build_field_fn,
    _resolve_stark_coupling,
    _stark_rotor_ensemble,
    run_pipeline_full,
)

_CONFIG_PATH = common.REPO_ROOT / "examples" / "showcase_gradient_dispersion_sr87.yaml"

#: Zoomed local-field panel half-extent, meters -- matches the field
#: generator's own exported half-extent (showcase_gen.EXPORT_HALF_EXTENT_M)
#: so the zoomed slice never queries the FieldSmoother outside its fit
#: bounding box.
#: 95% of the field generator's own export half-extent -- strictly inside
#: the FieldSmoother fit's bounding box (the full extent's own edge can
#: sit fractionally outside it after floating-point round-trip through
#: the generator's grid-index arithmetic), so this slice never triggers
#: `OutOfBoundsWarning` extrapolation.
_ZOOM_HALF_EXTENT_M = 0.95 * showcase_gen.EXPORT_HALF_EXTENT_M
_SLICE_N = 120

#: Number of Monte Carlo trajectories to overlay in panel (ii) -- a
#: legible subsample, not all `ensemble.size` atoms.
_N_TRAJ_SHOWN = 40

#: Memory-safety bounds (module docstring "Memory-safety note"). Trajectory
#: storage (`M * (steps+1) * 3 * 8` bytes, positions only -- this script
#: never materializes a velocity trajectory) must stay far under 1 GB.
_MAX_TRAJECTORY_BYTES = 1_000_000_000

#: The diagnosed root cause of the incident this note responds to: a
#: `FieldSmoother.evaluate` call's cost scales with
#: `n_query_points * n_fit_points` (JAX materializes several intermediate
#: query-by-fit-point arrays for the autodiff graph, not just one dense
#: kernel matrix), not just the trajectory array size above. Bounding this
#: product directly is what actually prevents a recurrence -- pinning
#: `ensemble.size`/`integration.steps` alone would not, if the field's own
#: fit-point count were later increased. This bound (30e6 pairs) is set
#: from a direct measurement, not a byte-counting estimate (an earlier
#: draft's "8 bytes/element" estimate undercounted the real cost by over
#: an order of magnitude): this exact scenario at 23.18e6 pairs
#: (`examples/showcase_gradient_dispersion_sr87.yaml`'s memory-safety
#: note) measured ~4.7 GB peak RSS end to end (pipeline run + rotor
#: cross-check) in a fresh subprocess, giving ~1.3x headroom below this
#: bound.
_MAX_FIELD_EVAL_QUERY_FIT_PAIRS = 30_000_000


def _check_compute_budget(ensemble_size: int, n_steps: int, n_fit_points: int) -> None:
    """Fail loudly, before running anything expensive, if this scenario's
    parameters would exceed this project's documented memory-safety bounds
    (module docstring). Raises `AssertionError` with the offending numbers
    rather than silently proceeding into a multi-GB (or worse) allocation.
    """
    trajectory_bytes = ensemble_size * (n_steps + 1) * 3 * 8
    assert trajectory_bytes < _MAX_TRAJECTORY_BYTES, (
        f"trajectory storage {trajectory_bytes:,} bytes exceeds the "
        f"{_MAX_TRAJECTORY_BYTES:,}-byte safety bound (ensemble_size={ensemble_size}, "
        f"n_steps={n_steps}) -- reduce ensemble.size or integration.steps"
    )
    query_fit_pairs = ensemble_size * n_steps * n_fit_points
    assert query_fit_pairs < _MAX_FIELD_EVAL_QUERY_FIT_PAIRS, (
        f"field-evaluation cost {query_fit_pairs:,} query-fit pairs exceeds the "
        f"{_MAX_FIELD_EVAL_QUERY_FIT_PAIRS:,}-pair safety bound (ensemble_size="
        f"{ensemble_size}, n_steps={n_steps}, n_fit_points={n_fit_points}) -- this is "
        "the exact mechanism that hung the development machine once already "
        "(module docstring 'Memory-safety note'); reduce ensemble.size, "
        "integration.steps, or the field generator's export point count"
    )


def _fmt_sci(x: float, sig: int = 4) -> str:
    """Render ``x`` as LaTeX scientific notation, e.g. ``-1.232\\times10^{-16}``."""
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10.0**exponent)
    return f"{mantissa:.{sig - 1}f}\\times10^{{{exponent}}}"


def main() -> None:
    common.reset_tex_macro_file("showcase_values.tex")

    # --- 0. Pre-flight compute-budget check (module docstring). -------------
    config = PipelineConfig.from_yaml(_CONFIG_PATH)
    assert config.integration.dtau is not None and config.integration.steps is not None, (
        "showcase config must pin integration.dtau/steps explicitly, never "
        "integration.time_s (auto-selected dtau) -- see the memory-safety note"
    )
    assert config.ensemble.size is not None, (
        "showcase config must pin ensemble.size explicitly -- see the memory-safety note"
    )
    _check_compute_budget(
        ensemble_size=config.ensemble.size,
        n_steps=config.integration.steps,
        n_fit_points=showcase_gen.EXPORT_POINTS_PER_AXIS**3,
    )

    # --- 1. Real pipeline run: trajectory (scalar) mode. --------------------
    t0 = time.perf_counter()
    result = run_pipeline_full(config)
    elapsed_scalar_s = time.perf_counter() - t0
    report_scalar = result.report

    species_full = get_species(config.species)
    stark_coeffs = _resolve_stark_coupling(config.coupling, species_full)

    trajectories = result.trajectories  # (M, T, 3)
    # dtau is pinned explicitly in the config (never auto-selected here);
    # read it directly rather than back-deriving it from interrogation_time_s.
    dtau = config.integration.dtau

    # --- 2. Identical trajectories through the true Cl(1,3) rotor path. ----
    field_fn, _n_fit_points = _build_field_fn(config.field_config)
    t0 = time.perf_counter()
    ensemble_result_rotor = _stark_rotor_ensemble(
        field_fn, stark_coeffs, trajectories, dtau, renorm_every=_auto_renorm_every()
    )
    elapsed_rotor_s = time.perf_counter() - t0
    report_rotor = build_report(
        ensemble_result_rotor.phase,
        species_full,
        report_scalar.interrogation_time_s,
        "classical_direct_rotor_crosscheck",
    )

    # --- 3. Agreement metrics (the message: scalar and rotor agree). -------
    ensemble_result_scalar = result.ensemble_result
    phase_scalar = np.asarray(ensemble_result_scalar.phase)
    phase_rotor = np.asarray(ensemble_result_rotor.phase)
    max_abs_phase_diff = float(np.max(np.abs(phase_scalar - phase_rotor)))
    shift_scalar = np.asarray(ensemble_result_scalar.fractional_shift)
    shift_rotor = np.asarray(ensemble_result_rotor.fractional_shift)
    max_abs_shift_diff = float(np.max(np.abs(shift_scalar - shift_rotor)))
    mean_shift_rel_diff = abs(
        report_scalar.mean_fractional_shift - report_rotor.mean_fractional_shift
    ) / abs(report_scalar.mean_fractional_shift)

    shift_spread_scalar = float(np.std(shift_scalar, ddof=1))
    shift_spread_rotor = float(np.std(shift_rotor, ddof=1))

    print(f"Scalar (trajectory-mode) run: {elapsed_scalar_s:.2f} s")
    print(f"Rotor cross-check run: {elapsed_rotor_s:.2f} s")
    print(
        f"scalar mean={report_scalar.mean_fractional_shift:+.6e} "
        f"spread={shift_spread_scalar:.4e} sem={report_scalar.shift_std_error:.4e} "
        f"t2*={report_scalar.t2_star_s:.6e} s"
    )
    print(
        f"rotor  mean={report_rotor.mean_fractional_shift:+.6e} "
        f"spread={shift_spread_rotor:.4e} sem={report_rotor.shift_std_error:.4e} "
        f"t2*={report_rotor.t2_star_s:.6e} s"
    )
    print(
        f"agreement: max|phase diff|={max_abs_phase_diff:.3e}, "
        f"max|shift diff|={max_abs_shift_diff:.3e}, "
        f"mean-shift rel diff={mean_shift_rel_diff:.3e}"
    )

    # --- 4. Line profiles (both paths). -------------------------------------
    n_time_samples = config.output.n_time_samples
    dt_s = report_scalar.interrogation_time_s / (n_time_samples - 1)
    t_grid_s = np.arange(n_time_samples, dtype=np.float64) * dt_s
    coherence_scalar = coherence_function(
        ensemble_result_scalar.phase, report_scalar.interrogation_time_s, t_grid_s
    )
    coherence_rotor = coherence_function(
        ensemble_result_rotor.phase, report_rotor.interrogation_time_s, t_grid_s
    )
    freqs_hz_scalar, amp_scalar = line_profile(coherence_scalar, dt_s)
    freqs_hz_rotor, amp_rotor = line_profile(coherence_rotor, dt_s)
    # Convert frequency offset (Hz) to fractional-shift units (Delta_nu/nu0)
    # so panel (iii) can share one x-axis with the per-atom shift histogram.
    nu0 = species_full.clock_frequency_hz
    freq_shift_units_scalar = freqs_hz_scalar / nu0
    freq_shift_units_rotor = freqs_hz_rotor / nu0

    # --- 5. Ensemble/field scale numbers, for the paper's scale-reasoning --
    trap_omega = float(np.asarray(config.trap.omega_xyz)[0])
    cloud_sigma_m = np.sqrt(
        BOLTZMANN_K * config.ensemble.temperature_uK * 1e-6 / (species_full.mass_kg * trap_omega**2)
    )
    e_center, _grad_center = field_fn(np.asarray([config.trap.center], dtype=np.float64))
    e_center_mag = float(np.linalg.norm(np.asarray(e_center)[0]))

    # --- 6. Figure: 3 panels. ------------------------------------------------
    fig, (ax_chamber, ax_zoom, ax_dist) = plt.subplots(1, 3, figsize=(11.5, 3.6))

    # Panel (i): whole-chamber XY field-magnitude slice at z = trap center,
    # from a fresh run of the field generator's own FD solve (geometry
    # context only -- the pipeline computation above uses solely the
    # committed showcase_field.txt export via FieldSmoother, never this
    # full-chamber solve).
    v_grid = showcase_gen.solve_potential()
    ex, ey, ez = showcase_gen.field_from_potential(v_grid)
    e_mag_full = np.sqrt(ex**2 + ey**2 + ez**2)
    ic = round(config.trap.center[0] / showcase_gen.H_M)
    jc = round(config.trap.center[1] / showcase_gen.H_M)
    kc = round(config.trap.center[2] / showcase_gen.H_M)

    # Live measurement (not a placeholder): relative field-magnitude
    # variation over one cloud sigma along each axis, from the full FD
    # solve -- the number the paper's showcase-scale-reasoning text quotes
    # ("the field varies by a genuine, measured fraction across the
    # cloud").
    h_cells = max(1, round(float(cloud_sigma_m) / showcase_gen.H_M))
    e_at_center = np.array([ex[ic, jc, kc], ey[ic, jc, kc], ez[ic, jc, kc]])
    e_at_center_mag_full = float(np.linalg.norm(e_at_center))
    rel_variation_per_sigma = []
    for axis in range(3):
        idx_lo = [ic, jc, kc]
        idx_hi = [ic, jc, kc]
        idx_lo[axis] -= h_cells
        idx_hi[axis] += h_cells
        e_lo = np.array([ex[tuple(idx_lo)], ey[tuple(idx_lo)], ez[tuple(idx_lo)]])
        e_hi = np.array([ex[tuple(idx_hi)], ey[tuple(idx_hi)], ez[tuple(idx_hi)]])
        rel_variation_per_sigma.append(
            float(np.linalg.norm(e_hi - e_lo)) / (2.0 * e_at_center_mag_full)
        )
    max_rel_variation_per_sigma = max(rel_variation_per_sigma)
    slice_xy = e_mag_full[:, :, kc]  # (NX, NY)
    extent_mm = [0.0, showcase_gen.BOX_X_M * 1e3, 0.0, showcase_gen.BOX_Y_M * 1e3]
    im = ax_chamber.imshow(
        slice_xy.T,
        origin="lower",
        extent=extent_mm,
        cmap="viridis",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax_chamber, label=r"$|E|$ (V/m)", fraction=0.046, pad=0.04)

    def _rect_patch(
        center_xy_m: tuple[float, float], half_w_m: float, **kwargs: object
    ) -> Rectangle:
        cx, cy = center_xy_m
        side_mm = 2.0 * half_w_m * 1e3
        return Rectangle(((cx - half_w_m) * 1e3, (cy - half_w_m) * 1e3), side_mm, side_mm, **kwargs)

    ax_chamber.add_patch(
        _rect_patch(
            showcase_gen.PLATE_A_CENTER_XY_M,
            showcase_gen.PLATE_A_HALF_WIDTH_M,
            fill=False,
            edgecolor="white",
            linestyle="--",
            linewidth=1.2,
        )
    )
    ax_chamber.text(
        showcase_gen.PLATE_A_CENTER_XY_M[0] * 1e3,
        showcase_gen.PLATE_A_CENTER_XY_M[1] * 1e3,
        "A\n(z=3mm)",
        color="white",
        ha="center",
        va="center",
        fontsize=7,
    )
    ax_chamber.add_patch(
        _rect_patch(
            showcase_gen.PLATE_B_CENTER_XY_M,
            showcase_gen.PLATE_B_HALF_WIDTH_M,
            fill=False,
            edgecolor="white",
            linestyle="--",
            linewidth=1.2,
        )
    )
    ax_chamber.text(
        showcase_gen.PLATE_B_CENTER_XY_M[0] * 1e3,
        showcase_gen.PLATE_B_CENTER_XY_M[1] * 1e3,
        "B (z=13mm)",
        color="white",
        ha="center",
        va="center",
        fontsize=7,
    )
    px, py = showcase_gen.PATCH_CENTER_XY_M
    ax_chamber.add_patch(
        Circle(
            (px * 1e3, py * 1e3),
            showcase_gen.PATCH_RADIUS_M * 1e3,
            fill=False,
            edgecolor=common.COLOR_REFERENCE,
            linestyle="--",
            linewidth=1.2,
        )
    )
    ax_chamber.text(
        px * 1e3,
        py * 1e3 - 1.8,
        "patch (z=0)",
        color=common.COLOR_REFERENCE,
        ha="center",
        va="top",
        fontsize=6,
    )
    cx_mm, cy_mm = config.trap.center[0] * 1e3, config.trap.center[1] * 1e3
    ax_chamber.plot([cx_mm], [cy_mm], marker="+", color="white", markersize=8, mew=1.5)
    ax_chamber.text(cx_mm + 0.6, cy_mm + 0.6, "cloud", color="white", fontsize=6)
    ax_chamber.set_xlabel("x (mm)")
    ax_chamber.set_ylabel("y (mm)")
    ax_chamber.set_title(
        f"Chamber (z={config.trap.center[2] * 1e3:.0f} mm slice):\n"
        "electrodes A/B (dashed white), patch (dashed red)"
    )

    # Panel (ii): zoomed local field (the actual FieldSmoother fit the
    # pipeline used, from the committed export) with a trajectory
    # subsample, colored by each shown atom's final fractional shift.
    cx_m, cy_m, cz_m = config.trap.center
    xg = np.linspace(cx_m - _ZOOM_HALF_EXTENT_M, cx_m + _ZOOM_HALF_EXTENT_M, _SLICE_N)
    yg = np.linspace(cy_m - _ZOOM_HALF_EXTENT_M, cy_m + _ZOOM_HALF_EXTENT_M, _SLICE_N)
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    query = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, cz_m)], axis=-1)
    e_zoom, _grad_zoom = field_fn(np.asarray(query, dtype=np.float64))
    e_zoom_mag = np.linalg.norm(np.asarray(e_zoom), axis=-1).reshape(_SLICE_N, _SLICE_N)
    extent_zoom_um = [
        (cx_m - _ZOOM_HALF_EXTENT_M - cx_m) * 1e6,
        (cx_m + _ZOOM_HALF_EXTENT_M - cx_m) * 1e6,
        (cy_m - _ZOOM_HALF_EXTENT_M - cy_m) * 1e6,
        (cy_m + _ZOOM_HALF_EXTENT_M - cy_m) * 1e6,
    ]
    im2 = ax_zoom.imshow(
        e_zoom_mag.T,
        origin="lower",
        extent=extent_zoom_um,
        cmap="viridis",
        aspect="equal",
    )
    fig.colorbar(im2, ax=ax_zoom, label=r"$|E|$ (V/m)", fraction=0.046, pad=0.04)

    rng = np.random.default_rng(0)
    show_idx = rng.choice(trajectories.shape[0], size=_N_TRAJ_SHOWN, replace=False)
    traj_np = np.asarray(trajectories)
    shift_for_color = shift_scalar
    norm = plt.Normalize(vmin=float(shift_for_color.min()), vmax=float(shift_for_color.max()))
    cmap = plt.get_cmap("plasma")
    for idx in show_idx:
        xs_um = (traj_np[idx, :, 0] - cx_m) * 1e6
        ys_um = (traj_np[idx, :, 1] - cy_m) * 1e6
        ax_zoom.plot(xs_um, ys_um, color=cmap(norm(shift_for_color[idx])), lw=0.6, alpha=0.85)
    ax_zoom.set_xlim(extent_zoom_um[0], extent_zoom_um[1])
    ax_zoom.set_ylim(extent_zoom_um[2], extent_zoom_um[3])
    ax_zoom.set_xlabel(r"$x$ ($\mu$m), trap center at origin")
    ax_zoom.set_ylabel(r"$y$ ($\mu$m)")
    ax_zoom.set_title(
        f"Local fitted field + {_N_TRAJ_SHOWN} MC trajectories\n(colored by accumulated shift)"
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_zoom, label=r"$\Delta\nu/\nu_0$", fraction=0.046, pad=0.11)

    # Panel (iii): per-atom shift distribution (histogram, left axis) +
    # line profile (right axis), shared x-axis in fractional-shift units,
    # T2* annotated. Zoomed to a window a few multiples of the line's own
    # width (set by T2*), not the full FFT Nyquist range.
    ax_dist.hist(
        shift_scalar,
        bins=30,
        color=common.COLOR_ENGINE,
        alpha=0.55,
        label="per-atom shift (scalar)",
    )
    ax_dist.axvline(report_scalar.mean_fractional_shift, color=common.COLOR_ENGINE, lw=1.2, ls="--")
    ax_dist.set_xlabel(r"$\Delta\nu/\nu_0$")
    ax_dist.set_ylabel("atom count")

    ax_line = ax_dist.twinx()
    half_window = 6.0 * shift_spread_scalar
    window_mask = (
        np.abs(freq_shift_units_scalar - report_scalar.mean_fractional_shift) < half_window
    )
    ax_line.plot(
        freq_shift_units_scalar[window_mask],
        amp_scalar[window_mask],
        color=common.COLOR_REFERENCE,
        lw=1.3,
        label="line profile (scalar)",
    )
    ax_line.plot(
        freq_shift_units_rotor[window_mask],
        amp_rotor[window_mask],
        color="black",
        lw=0.8,
        ls=":",
        label="line profile (rotor)",
    )
    ax_line.set_ylabel("line-profile amplitude")
    ax_dist.set_xlim(
        report_scalar.mean_fractional_shift - half_window,
        report_scalar.mean_fractional_shift + half_window,
    )
    ax_dist.set_title(
        f"Shift distribution + line profile\n$T_2^*$ = {report_scalar.t2_star_s * 1e6:.1f} $\\mu$s"
    )
    lines1, labels1 = ax_dist.get_legend_handles_labels()
    lines2, labels2 = ax_line.get_legend_handles_labels()
    ax_dist.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper right")

    fig.tight_layout()
    fig.savefig(common.FIGURES_DIR / "fig4_showcase_gradient_dispersion.pdf")
    plt.close(fig)

    # --- 7. Generated \input macros. -----------------------------------------
    common.write_tex_macro("ShowcaseFieldCenter", f"{e_center_mag:.2f}", "showcase_values.tex")
    common.write_tex_macro("ShowcaseTrapOmega", f"{trap_omega:.0f}", "showcase_values.tex")
    common.write_tex_macro(
        "ShowcaseTemperatureUK",
        f"{config.ensemble.temperature_uK:.0f}",
        "showcase_values.tex",
    )
    common.write_tex_macro(
        "ShowcaseCloudSigmaUm", f"{cloud_sigma_m * 1e6:.0f}", "showcase_values.tex"
    )
    common.write_tex_macro("ShowcaseEnsembleSize", str(config.ensemble.size), "showcase_values.tex")
    common.write_tex_macro(
        "ShowcaseInterrogationTime",
        f"{report_scalar.interrogation_time_s:.3f}",
        "showcase_values.tex",
    )
    common.write_tex_macro(
        "ShowcaseMeanShift", _fmt_sci(report_scalar.mean_fractional_shift), "showcase_values.tex"
    )
    common.write_tex_macro(
        "ShowcaseShiftSpread", _fmt_sci(shift_spread_scalar), "showcase_values.tex"
    )
    common.write_tex_macro(
        "ShowcaseShiftSem", _fmt_sci(report_scalar.shift_std_error), "showcase_values.tex"
    )
    common.write_tex_macro(
        "ShowcaseTtwoStarUs", f"{report_scalar.t2_star_s * 1e6:.1f}", "showcase_values.tex"
    )
    common.write_tex_macro(
        "ShowcaseTtwoStarRotorUs", f"{report_rotor.t2_star_s * 1e6:.1f}", "showcase_values.tex"
    )
    common.write_tex_macro(
        "ShowcaseMeanShiftRelDiff", _fmt_sci(mean_shift_rel_diff), "showcase_values.tex"
    )
    common.write_tex_macro(
        "ShowcaseMaxPhaseDiff", _fmt_sci(max_abs_phase_diff), "showcase_values.tex"
    )
    common.write_tex_macro(
        "ShowcaseMaxShiftDiff", _fmt_sci(max_abs_shift_diff), "showcase_values.tex"
    )
    common.write_tex_macro(
        "ShowcaseRelFieldVariation",
        f"{max_rel_variation_per_sigma * 100.0:.1f}",
        "showcase_values.tex",
    )

    print(f"Wrote {common.FIGURES_DIR / 'fig4_showcase_gradient_dispersion.pdf'}")
    print(f"Wrote {common.GENERATED_DIR / 'showcase_values.tex'}")


if __name__ == "__main__":
    main()
