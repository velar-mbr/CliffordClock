#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate ``docs/assets/showcase_animation.gif``, the README's headline
animation.

**What this shows.** The same real chamber-scale field and Monte Carlo
ensemble as the paper's showcase figure
(``paper/figures/fig4_showcase_gradient_dispersion.py``, built from
``examples/showcase_gradient_dispersion_sr87.yaml`` + the field export
``examples/showcase_field.txt`` -- see that figure script's module
docstring for the underlying physics). Two panels, updated frame by
frame:

- **Left**: the fitted local field magnitude (``FieldSmoother``, the same
  object the pipeline itself evaluates) as a backdrop, with each atom's
  Monte Carlo trajectory drawn up to the current frame's elapsed time,
  colored by that atom's own accumulated fractional shift *so far*
  (``Delta_nu/nu0``, the running mean rate along the true trajectory --
  see "Per-frame accumulated shift" below).
- **Right**: the ensemble coherence function ``|C(t)| = |<exp(i Delta
  Phi_i(t))>|`` (E26, :func:`cliffordclock.analytics.coherence_function`,
  the same call ``fig4_showcase_gradient_dispersion.py`` uses for its
  line-profile panel) revealed progressively in step with the left
  panel's frame count, with T2* marked.

**Why the two panels use different time axes (read before changing
either).** This scenario's own config header
(``examples/showcase_gradient_dispersion_sr87.yaml``) documents that its
T2* is set by where each atom sits in the field at the *start* of the
interrogation window, not by how long the window runs -- the per-atom
shift spread (and hence T2*) is fixed almost immediately, while the left
panel's trajectories keep moving for the full window to show real spatial
motion through the field. Concretely: this scenario's real T2* (order
100 microseconds) is shorter than a single trajectory step
(``dtau * TAU_COMPTON``, order 100s of microseconds here), so a coherence
curve plotted against the same real elapsed-trajectory-time axis as the
left panel would collapse to zero within the first frame or two -- not a
bug, just two different physical timescales in the same scenario. The
right panel therefore plots coherence against its own microsecond-scale
axis (a handful of T2*, independently sampled, exactly the reconstructed
``coherence_function`` call ``fig4_showcase_gradient_dispersion.py``
already makes), and "reveals" that fixed curve in step with the
animation's frame counter as a visualization device -- the curve itself
is real, computed output; only the pacing of its reveal is decoupled from
the left panel's real elapsed time. Both axes are labeled in real units so
this distinction is visible, not implied.

**Per-frame accumulated shift.** The pipeline's own scalar accumulator
(:func:`cliffordclock.pipeline._stark_scalar_ensemble`) keeps only the
final per-atom phase (a ``jax.lax.scan`` reduction, no per-step history).
This script instead calls the identical rate function
(:func:`cliffordclock.pipeline._make_stark_rate_fn`, built from the same
``_build_field_fn``/``_resolve_stark_coupling`` the pipeline and
``fig4_showcase_gradient_dispersion.py`` both use) on the pipeline's own
committed trajectories, keeping every step's contribution instead of
collapsing them, so each frame's color is a real per-step-accumulated
quantity, not an interpolation. Sanity-checked at the final step against
the real pipeline's own ``EnsembleResult.phase`` (see
``_check_shift_reconstruction`` below) before this script trusts its own
running total for anything the figure displays.

**Reduced configuration (regeneration speed; visible in the figure
itself, see ``_REDUCED_DEMO_LABEL``).** The shipped showcase config uses
a 100-atom ensemble over 318 steps; this animation instead uses
:data:`N_ATOMS_DEMO`/:data:`N_STEPS_DEMO` atoms/steps of the *same*
scenario (same field, species, trap, temperature, seed, dtau) -- fewer
Monte Carlo trajectories and a shorter integration window, never a
different field or a different physical coefficient. Every frame is
still built from a real pipeline run of the real scenario, just at a
demo-appropriate size for a repository asset that regenerates in around
a minute.

Regeneration
------------
Run from an activated project venv (``pip install -e ".[notebooks]"`` --
matplotlib + Pillow only, both already required by that extra; no new
dependency)::

    python examples/generate_showcase_animation.py

Last measured wall time on the author's machine: about 12 s total
(roughly 2 s pipeline run + 1 s per-step rate reconstruction + 1 s field
backdrop solve + 8 s animation render/GIF encode), comfortably under the
WP25 60 s target. Output: ``docs/assets/showcase_animation.gif``
(~121 frames at 10 fps, about a 12 s loop).
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import Normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import generate_showcase_field as showcase_gen  # noqa: E402

from cliffordclock.analytics import coherence_function  # noqa: E402
from cliffordclock.constants import TAU_COMPTON  # noqa: E402
from cliffordclock.ensemble.species import get_species  # noqa: E402
from cliffordclock.pipeline import (  # noqa: E402
    PipelineConfig,
    _build_field_fn,
    _make_stark_rate_fn,
    _resolve_stark_coupling,
    run_pipeline_full,
)

_CONFIG_PATH = REPO_ROOT / "examples" / "showcase_gradient_dispersion_sr87.yaml"
_OUTPUT_PATH = REPO_ROOT / "docs" / "assets" / "showcase_animation.gif"

#: Reduced-demo sizing (module docstring). Kept well inside the "40-60
#: atoms" WP25 guidance and small enough that the field-evaluation cost
#: (~ N_ATOMS_DEMO * N_STEPS_DEMO * 729 fit points) stays two orders of
#: magnitude under the showcase figure's own measured memory-safety bound
#: (paper/figures/fig4_showcase_gradient_dispersion.py's
#: _MAX_FIELD_EVAL_QUERY_FIT_PAIRS).
N_ATOMS_DEMO = 48
N_STEPS_DEMO = 120

#: Frames-per-second for the GIF; N_STEPS_DEMO + 1 frames at this rate
#: gives a ~12 s loop (WP25's 8-15 s target).
FPS = 10

#: Number of samples for the right-panel coherence curve (its own
#: microsecond-scale axis -- module docstring "different time axes").
N_COHERENCE_SAMPLES = 220

#: The coherence curve's x-axis spans this many T2* (a decayed contrast
#: worth showing, not just the initial drop).
COHERENCE_WINDOW_T2_MULT = 6.0

#: Zoomed local-field panel half-extent, meters -- must stay inside the
#: field generator's own exported half-extent, exactly like
#: fig4_showcase_gradient_dispersion.py's _ZOOM_HALF_EXTENT_M (same
#: reasoning: never query the FieldSmoother outside its fit bounding box).
_ZOOM_HALF_EXTENT_M = 0.95 * showcase_gen.EXPORT_HALF_EXTENT_M
_SLICE_N = 90

_COLOR_FIELD_CMAP = "viridis"
_COLOR_SHIFT_CMAP = "plasma"
_COLOR_COHERENCE = "#2E5FA3"
_COLOR_T2STAR = "#C1666B"

_REDUCED_DEMO_LABEL = (
    f"Reduced demo: {N_ATOMS_DEMO} atoms, {N_STEPS_DEMO} steps of the paper's "
    "showcase scenario (shipped showcase: 100 atoms, 318 steps). "
    "Regenerate: examples/generate_showcase_animation.py"
)


def _build_reduced_config() -> PipelineConfig:
    """The showcase config, with a smaller ensemble and shorter window.

    Same species, trap, temperature, seed, field, coupling, and per-step
    ``dtau`` as ``examples/showcase_gradient_dispersion_sr87.yaml`` --
    only ``ensemble.size`` and ``integration.steps`` shrink (module
    docstring "Reduced configuration").
    """
    base = PipelineConfig.from_yaml(_CONFIG_PATH)
    assert base.integration.dtau is not None and base.integration.steps is not None
    assert base.ensemble.size is not None
    return dataclasses.replace(
        base,
        ensemble=dataclasses.replace(base.ensemble, size=N_ATOMS_DEMO),
        integration=dataclasses.replace(base.integration, steps=N_STEPS_DEMO),
    )


def _per_step_fractional_shift(
    config: PipelineConfig, trajectories: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom, per-step accumulated fractional shift along the real trajectories.

    Reruns the pipeline's own E14b rate function
    (:func:`cliffordclock.pipeline._make_stark_rate_fn`) at the same
    midpoint/velocity samples :func:`cliffordclock.pipeline._stark_scalar_ensemble`
    uses, but keeps every step instead of reducing to one final phase, so
    each frame's trajectory color is a real quantity, not an
    interpolation (module docstring "Per-frame accumulated shift").

    Returns
    -------
    cumulative_phase : numpy.ndarray, shape (M, S)
        Running accumulated phase after each of the ``S`` steps.
    fractional_shift_running : numpy.ndarray, shape (M, S)
        ``cumulative_phase / elapsed_proper_time``, the running mean
        shift rate an atom has experienced up to that step.
    """
    species_full = get_species(config.species)
    stark_coeffs = _resolve_stark_coupling(config.coupling, species_full)
    field_fn, _n_fit_points = _build_field_fn(config.field_config)
    rate_fn = _make_stark_rate_fn(field_fn, stark_coeffs)

    assert config.integration.dtau is not None
    dtau = config.integration.dtau
    dt_phys_s = dtau * TAU_COMPTON

    pos_a, pos_b = trajectories[:, :-1, :], trajectories[:, 1:, :]
    pos_mid = 0.5 * (pos_a + pos_b)
    v_mid = (pos_b - pos_a) / dt_phys_s
    m, s, _ = pos_mid.shape
    domega = np.asarray(
        rate_fn(pos_mid.reshape(-1, 3), v_mid.reshape(-1, 3)), dtype=np.float64
    ).reshape(m, s)

    cumulative_phase = np.cumsum(domega * dtau, axis=1)
    elapsed_tau = np.arange(1, s + 1, dtype=np.float64) * dtau
    fractional_shift_running = cumulative_phase / elapsed_tau[None, :]
    return cumulative_phase, fractional_shift_running


def _check_shift_reconstruction(cumulative_phase: np.ndarray, ensemble_phase: np.ndarray) -> None:
    """The reconstructed final phase must match the real pipeline's own
    `EnsembleResult.phase` (this script's own per-step sum vs. the
    pipeline's Kahan-summed reduction -- both float64, same terms, no
    reason for more than roundoff-level disagreement).
    """
    final_phase = cumulative_phase[:, -1]
    max_abs_diff = float(np.max(np.abs(final_phase - np.asarray(ensemble_phase))))
    assert max_abs_diff < 1e-8, (
        "per-step rate reconstruction disagrees with the real pipeline's final "
        f"phase by {max_abs_diff:.3e} (expected roundoff-level agreement) -- "
        "the per-frame trajectory colors would not be trustworthy"
    )


def _chamber_field_backdrop(
    config: PipelineConfig,
) -> tuple[np.ndarray, list[float]]:
    """Zoomed local field magnitude (V/m) around the trap center, for the
    left panel's backdrop -- same region the pipeline's own FieldSmoother
    fit covers (mirrors fig4_showcase_gradient_dispersion.py panel ii).
    """
    field_fn, _n_fit_points = _build_field_fn(config.field_config)
    cx_m, cy_m, cz_m = config.trap.center
    xg = np.linspace(cx_m - _ZOOM_HALF_EXTENT_M, cx_m + _ZOOM_HALF_EXTENT_M, _SLICE_N)
    yg = np.linspace(cy_m - _ZOOM_HALF_EXTENT_M, cy_m + _ZOOM_HALF_EXTENT_M, _SLICE_N)
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    query = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, cz_m)], axis=-1)
    e_zoom, _grad_zoom = field_fn(np.asarray(query, dtype=np.float64))
    e_mag = np.linalg.norm(np.asarray(e_zoom), axis=-1).reshape(_SLICE_N, _SLICE_N)
    extent_um = [
        -_ZOOM_HALF_EXTENT_M * 1e6,
        _ZOOM_HALF_EXTENT_M * 1e6,
        -_ZOOM_HALF_EXTENT_M * 1e6,
        _ZOOM_HALF_EXTENT_M * 1e6,
    ]
    return e_mag, extent_um


def main() -> None:
    t_wall_start = time.perf_counter()
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    config = _build_reduced_config()

    t0 = time.perf_counter()
    result = run_pipeline_full(config)
    t_pipeline_s = time.perf_counter() - t0
    report = result.report
    trajectories = np.asarray(result.trajectories, dtype=np.float64)  # (M, S+1, 3)

    t0 = time.perf_counter()
    cumulative_phase, shift_running = _per_step_fractional_shift(config, trajectories)
    _check_shift_reconstruction(cumulative_phase, np.asarray(result.ensemble_result.phase))
    t_reconstruct_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    field_mag, extent_um = _chamber_field_backdrop(config)
    t_field_s = time.perf_counter() - t0

    # --- Right-panel coherence curve, own microsecond-scale axis (module ---
    # docstring "Why the two panels use different time axes").
    t2_star_s = report.t2_star_s
    coherence_window_s = min(report.interrogation_time_s, COHERENCE_WINDOW_T2_MULT * t2_star_s)
    t_grid_coh_s = np.linspace(0.0, coherence_window_s, N_COHERENCE_SAMPLES)
    coherence = coherence_function(
        result.ensemble_result.phase, report.interrogation_time_s, t_grid_coh_s
    )
    contrast = np.abs(coherence)
    t_grid_coh_us = t_grid_coh_s * 1e6

    # --- Figure/axes setup (created once; per-frame update mutates artists). --
    # Explicit axes placement (rather than plt.subplots + colorbar(ax=...))
    # so the two left-panel colorbars (field magnitude, accumulated shift)
    # get enough dedicated horizontal room that their tick labels never
    # collide -- the automatic layout crowded them together.
    fig = plt.figure(figsize=(12.5, 4.4))
    ax_traj = fig.add_axes((0.055, 0.12, 0.36, 0.72))
    cax_field = fig.add_axes((0.445, 0.12, 0.018, 0.72))
    cax_shift = fig.add_axes((0.545, 0.12, 0.018, 0.72))
    ax_coh = fig.add_axes((0.68, 0.12, 0.29, 0.72))

    im = ax_traj.imshow(
        field_mag.T,
        origin="lower",
        extent=extent_um,
        cmap=_COLOR_FIELD_CMAP,
        aspect="equal",
    )
    fig.colorbar(im, cax=cax_field, label=r"$|E|$ (V/m)")
    ax_traj.set_xlabel(r"$x$ ($\mu$m), trap center at origin")
    ax_traj.set_ylabel(r"$y$ ($\mu$m)")
    ax_traj.set_title("Chamber field + Monte Carlo trajectories")

    shift_vmin = float(np.min(shift_running))
    shift_vmax = float(np.max(shift_running))
    norm = Normalize(vmin=shift_vmin, vmax=shift_vmax)
    cmap = plt.get_cmap(_COLOR_SHIFT_CMAP)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    shift_cbar = fig.colorbar(sm, cax=cax_shift, label=r"accumulated $\Delta\nu/\nu_0$")
    shift_cbar.formatter.set_powerlimits((0, 0))
    shift_cbar.locator = plt.MaxNLocator(nbins=4)
    shift_cbar.update_ticks()

    cx_m, cy_m, _cz_m = config.trap.center
    traj_um = np.stack(
        [(trajectories[:, :, 0] - cx_m) * 1e6, (trajectories[:, :, 1] - cy_m) * 1e6], axis=-1
    )  # (M, S+1, 2)

    m_atoms = traj_um.shape[0]
    lines = [ax_traj.plot([], [], lw=0.7, alpha=0.85, color="0.6")[0] for _ in range(m_atoms)]
    heads = ax_traj.scatter(
        traj_um[:, 0, 0], traj_um[:, 0, 1], s=6, c="0.6", zorder=3, edgecolors="none"
    )
    ax_traj.set_xlim(extent_um[0], extent_um[1])
    ax_traj.set_ylim(extent_um[2], extent_um[3])
    elapsed_text = ax_traj.text(
        0.02,
        0.97,
        "",
        transform=ax_traj.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="white",
    )

    ax_coh.set_xlim(0.0, t_grid_coh_us[-1])
    ax_coh.set_ylim(0.0, 1.05)
    ax_coh.set_xlabel(r"$t$ ($\mu$s) -- own clock, decoupled from the left panel's ms-scale time")
    ax_coh.set_ylabel(r"$|C(t)|$ (ensemble contrast)")
    ax_coh.set_title("Coherence decay ($T_2^*$)")
    ax_coh.axhline(1.0 / np.e, color="0.6", lw=0.8, ls=":")
    ax_coh.text(
        t_grid_coh_us[-1],
        1.0 / np.e,
        " $1/e$",
        color="0.5",
        fontsize=7,
        ha="right",
        va="bottom",
    )
    t2_star_us = t2_star_s * 1e6
    ax_coh.axvline(t2_star_us, color=_COLOR_T2STAR, lw=1.0, ls="--")
    ax_coh.text(
        t2_star_us,
        1.0,
        f"  $T_2^*$ = {t2_star_us:.1f} $\\mu$s",
        color=_COLOR_T2STAR,
        fontsize=7,
        ha="left",
        va="top",
    )
    (coh_line,) = ax_coh.plot([], [], color=_COLOR_COHERENCE, lw=1.6)
    (coh_head,) = ax_coh.plot([], [], marker="o", ms=4, color=_COLOR_COHERENCE)

    fig.suptitle(_REDUCED_DEMO_LABEL, fontsize=7.5, y=0.995)

    total_elapsed_ms = N_STEPS_DEMO * config.integration.dtau * TAU_COMPTON * 1e3
    n_frames = N_STEPS_DEMO + 1

    def update(frame: int):
        for atom in range(m_atoms):
            lines[atom].set_data(traj_um[atom, : frame + 1, 0], traj_um[atom, : frame + 1, 1])
            if frame >= 1:
                lines[atom].set_color(cmap(norm(shift_running[atom, frame - 1])))
        heads.set_offsets(traj_um[:, frame, :])
        if frame >= 1:
            heads.set_color([cmap(norm(shift_running[atom, frame - 1])) for atom in range(m_atoms)])

        elapsed_ms = frame * config.integration.dtau * TAU_COMPTON * 1e3
        elapsed_text.set_text(f"elapsed $t$ = {elapsed_ms:.1f} / {total_elapsed_ms:.1f} ms")

        reveal_frac = frame / (n_frames - 1)
        reveal_n = max(1, int(round(reveal_frac * N_COHERENCE_SAMPLES)))
        coh_line.set_data(t_grid_coh_us[:reveal_n], contrast[:reveal_n])
        coh_head.set_data([t_grid_coh_us[reveal_n - 1]], [contrast[reveal_n - 1]])
        return [*lines, heads, elapsed_text, coh_line, coh_head]

    t0 = time.perf_counter()
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    ani.save(_OUTPUT_PATH, writer=animation.PillowWriter(fps=FPS), dpi=88)
    plt.close(fig)
    t_render_s = time.perf_counter() - t0

    t_wall_total_s = time.perf_counter() - t_wall_start
    size_mb = _OUTPUT_PATH.stat().st_size / 1e6
    print(f"pipeline run: {t_pipeline_s:.2f} s")
    print(f"per-step rate reconstruction: {t_reconstruct_s:.2f} s")
    print(f"field backdrop solve: {t_field_s:.2f} s")
    print(f"animation render + encode: {t_render_s:.2f} s")
    print(f"total wall time: {t_wall_total_s:.2f} s")
    print(f"wrote {_OUTPUT_PATH} ({size_mb:.2f} MB, {n_frames} frames at {FPS} fps)")
    print(
        f"report: mean_fractional_shift={report.mean_fractional_shift:+.4e} "
        f"t2_star_s={report.t2_star_s:.4e}"
    )


if __name__ == "__main__":
    main()
