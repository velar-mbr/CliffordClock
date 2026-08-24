#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate ``docs/assets/ion_motion_animation.gif``, a companion hero
animation for the trapped-ion audience.

**What this shows.** The published two-ion 27Al+/25Mg+ crystal from
Marshall et al. (arXiv:2504.13071v2), animated through its six secular
normal modes, next to the time-dilation budget those modes add up to.
Every number on screen comes from the real engine, through the same
call path ``benchmarks/run_motional_al_ion.py``'s WP35 constrained-fit
case already uses:

- **Left panel**: the two ions on their trap axis, cycling through the
  six modes (axial COM/STR, then radial X COM/STR, then radial Y
  COM/STR) in Marshall et al.'s own Table S2 order. Each mode's
  displayed frequency and mean phonon number ``n_bar`` come from
  ``benchmarks/loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR``. Each ion's
  oscillation amplitude is that ion's own mass-weighted participation
  factor (:func:`cliffordclock.integrator.omega.two_ion_participations`
  for the axial pair; for the radial pairs, the SAME per-mode coupled
  Fourier decomposition
  ``run_motional_al_ion.run_motional_al_ion_constrained_floquet_case``
  fits the clock ion's Mathieu parameters against, one shared RF
  parameter q and a dc-split fraction alpha solved from all four
  measured radial mode frequencies at once), square-rooted to turn an
  energy fraction back into an amplitude and scaled only for
  visibility. COM modes move both ions the same direction (in-phase);
  STR modes move them opposite directions (out-of-phase), the sign
  structure ``two_ion_participations``'s own docstring derives from the
  coupled normal-mode eigenvectors. The oscillation RATE shown is a
  visualization convenience: a real MHz-scale oscillation has no
  legible frame rate in a gif, so each mode plays a fixed
  number of cycles, the same kind of documented, labeled pacing
  decoupling ``generate_showcase_animation.py``'s own module docstring
  already uses for its coherence panel.
- **Right panel**: a bar that accumulates each mode's own
  time-dilation contribution (``predicted_shift_per_quantum *
  (n_bar + 1/2)``, the exact per-mode term the constrained-fit case
  sums internally to reach its own total) as that mode plays on the
  left, colored by whether the mode is axial (participation alone,
  micromotion enhancement fixed at 1.0) or radial (participation and
  micromotion enhancement both read from the same coupled two-ion
  Floquet solution at the fitted q and dc split, with no separate
  per-axis enhancement factor multiplied in). Once all six modes have
  played, Marshall et al.'s own published band
  (``benchmarks/loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT``) appears
  around it. This script computes the same sigma separation and
  verdict ``run_motional_al_ion.py``'s WP35 constrained-fit case
  reports (0.08 sigma, ``MET``) itself, by the identical formula that
  case function uses internally.

This script never hand-types a physics number: every frequency, phonon
number, participation, enhancement factor, contribution, and the final
sigma separation is either read from ``benchmarks/loaders.py``'s
transcribed publication tables or computed by calling
``benchmarks/run_motional_al_ion.py``'s own WP35 constrained-fit case
function, which itself calls the real engine functions in
``cliffordclock.integrator.omega``. A runtime check (mirroring
``generate_showcase_animation.py``'s own
``_check_shift_reconstruction``) confirms this script's own per-mode
sum reproduces the case function's total before trusting it for the
animation. Nothing here is tuned against Marshall's published total:
the fitted Mathieu parameters solve from the four MEASURED radial mode
frequencies alone, the same over-determination
``plan/reviews/G17-e38-coupled-floquet.md`` reviewed. The caption's
"zero free parameters" line is cashable against three inputs: the
measured mode frequencies, the mean phonon numbers, and the published
RF drive frequency, all three read live from ``benchmarks/loaders.py``.

Regeneration
------------
Run from an activated project venv (``pip install -e ".[notebooks]"``
-- matplotlib + Pillow only, both already required by that extra; no
new dependency)::

    python examples/generate_ion_motion_animation.py

Deterministic: no wall-clock reads, no randomness. Two runs on the
same machine produce byte-identical output. Output:
``docs/assets/ion_motion_animation.gif``.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import loaders  # noqa: E402
import run_motional_al_ion  # noqa: E402

_OUTPUT_PATH = REPO_ROOT / "docs" / "assets" / "ion_motion_animation.gif"

#: Table S2 mode order, matching `run_motional_al_ion._MODE_NAMES` and
#: `loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR`.
MODE_ORDER: tuple[str, ...] = ("axial_com", "axial_str", "x_com", "x_str", "y_com", "y_str")

#: Plain-language label and displacement axis for each mode. "axis" is
#: "x" (the trap axis, drawn horizontally) for the two axial modes, "y"
#: (drawn vertically, transverse to the trap axis) for both radial
#: pairs: a 2D schematic draws the two orthogonal radial directions
#: (X, Y) the same way on screen, and the mode label is what tells
#: them apart, as Marshall et al.'s own table does.
MODE_DISPLAY: dict[str, tuple[str, str]] = {
    "axial_com": ("Axial COM (in-phase)", "x"),
    "axial_str": ("Axial STR (out-of-phase)", "x"),
    "x_com": ("Radial X COM (in-phase)", "y"),
    "x_str": ("Radial X STR (out-of-phase)", "y"),
    "y_com": ("Radial Y COM (in-phase)", "y"),
    "y_str": ("Radial Y STR (out-of-phase)", "y"),
}

#: `True` for the two COM modes (both ions move the same direction),
#: `False` for the two STR modes (opposite directions). The sign
#: structure comes from `two_ion_participations`'s own docstring,
#: which derives it from the coupled normal-mode eigenvector matrix
#: (`[[b1, b2], [b2, -b1]]`).
MODE_IN_PHASE: dict[str, bool] = {name: name.endswith("com") for name in MODE_ORDER}

#: Frames-per-second for the GIF.
FPS = 10

#: Each mode plays for this many frames (2.5 s at FPS=10).
FRAMES_PER_MODE = 25

#: Final hold, totals side by side (3 s at FPS=10).
HOLD_FRAMES = 30

N_MODES = len(MODE_ORDER)
N_FRAMES = FRAMES_PER_MODE * N_MODES + HOLD_FRAMES

#: Oscillation cycles shown per mode, chosen so `sin(2*pi*CYCLES*1.0)
#: == 0`: every mode's animation starts and ends at the ions'
#: equilibrium position, giving a clean handoff between modes at the
#: seam.
CYCLES_PER_MODE = 2.5

#: Visual amplitude scale (dimensionless plot units) applied to each
#: ion's sqrt(participation) so both ions stay clear of each other and
#: of the axis labels across every mode in the shipped dataset.
AMPLITUDE_SCALE = 0.45

#: Equilibrium half-spacing (plot units) between the two ions'
#: schematic axial positions, chosen only so both ions and their
#: labels stay clear of each other in the frame. This project's own
#: `two_ion_radial_participations` derivation works from a Coulomb
#: curvature in newtons per meter, and Marshall et al.'s paper
#: publishes mode frequencies; neither source gives a real equilibrium
#: spacing in meters to draw from.
ION_HALF_SPACING = 0.55

_COLOR_CLOCK_ION = "C0"
_COLOR_PARTNER_ION = "C1"
_COLOR_AXIAL_CONTRIBUTION = "C0"
_COLOR_RADIAL_CONTRIBUTION = "C2"
_COLOR_PUBLISHED = "C1"

_CAPTION_LINE_1 = (
    "Inputs: measured mode frequencies and mean phonon numbers (Marshall et al. 2025, Table S2),"
)
_RF_DRIVE_MHZ = loaders.MARSHALL_AL_ION_RF_DRIVE_FREQUENCY_HZ / 1.0e6
_CAPTION_LINE_2 = f"published RF drive frequency ({_RF_DRIVE_MHZ:.2f} MHz), zero free parameters."


def _mode_inputs() -> dict[str, tuple[float, float, float]]:
    """Each mode's ``(frequency_hz, n_bar, n_bar_uncertainty)``, from
    `loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR` (MHz converted to Hz).
    """
    return {
        name: (frequency_mhz * 1.0e6, n_bar, n_bar_uncertainty)
        for name, frequency_mhz, n_bar, n_bar_uncertainty in loaders.MARSHALL_AL_ION_MODES_MHZ_NBAR
    }


def _per_mode_contributions_1e19(
    case: run_motional_al_ion.MotionalAlIonConstrainedFloquetCase,
    mode_inputs: dict[str, tuple[float, float, float]],
) -> dict[str, float]:
    """Each mode's own share of the WP35 constrained-fit total, in 1e-19 units.

    ``contribution_i = predicted_shift_per_quantum_i * (n_bar_i + 1/2)``,
    the exact per-mode term the constrained-fit case sums internally
    (with a sign flip already folded into `predicted_shift_per_quantum`)
    to reach `case.predicted_total_nominal`; verified below before this
    script trusts it for the animation.
    """
    per_mode_by_name = {mode.name: mode for mode in case.per_mode}
    contributions = {}
    for name in MODE_ORDER:
        mode = per_mode_by_name[name]
        _frequency_hz, n_bar, _n_bar_uncertainty = mode_inputs[name]
        contributions[name] = mode.predicted_shift_per_quantum * (n_bar + 0.5) * 1.0e19
    return contributions


def _check_total_reconstruction(
    contributions_1e19: dict[str, float],
    case: run_motional_al_ion.MotionalAlIonConstrainedFloquetCase,
) -> None:
    """The six per-mode contributions must sum to the case's own total.

    Mirrors `generate_showcase_animation.py`'s own
    `_check_shift_reconstruction` discipline: a derived per-part
    quantity is checked against the real engine's own whole before it
    is trusted for display.
    """
    reconstructed_total = math.fsum(contributions_1e19.values()) * 1.0e-19
    published_total = case.predicted_total_nominal
    rel_diff = abs(reconstructed_total - published_total) / abs(published_total)
    assert rel_diff < 1.0e-9, (
        "per-mode contribution reconstruction disagrees with the WP35 constrained-fit case's "
        f"own predicted_total_nominal by a relative {rel_diff:.3e} (expected roundoff-level "
        "agreement) -- the animation's bar heights would not be trustworthy"
    )


def _deviation_sigma(
    case: run_motional_al_ion.MotionalAlIonConstrainedFloquetCase,
    published: loaders.PublishedBand,
) -> float:
    """Sigma separation between the predicted and published totals.

    The identical formula `run_motional_al_ion.py`'s WP35 constrained-fit
    case function uses internally to build its own `fit_note` ("0.08
    sigma"), computed here directly from the case's own returned fields.
    """
    combined_sigma = math.sqrt(
        case.predicted_total_uncertainty_combined_fractional**2
        + (published.hi - published.nominal) ** 2
    )
    return abs(case.predicted_total_nominal - published.nominal) / combined_sigma


def _mode_frame_state(frame: int) -> tuple[int, float, bool]:
    """Which mode is animating at `frame`, how far through its own
    growth/oscillation slot (0.0-1.0), and whether `frame` is in the
    final hold.
    """
    if frame < FRAMES_PER_MODE * N_MODES:
        mode_index = frame // FRAMES_PER_MODE
        local = frame % FRAMES_PER_MODE
        progress = local / (FRAMES_PER_MODE - 1)
        return mode_index, progress, False
    return N_MODES - 1, 1.0, True


def main() -> None:
    t_wall_start = time.perf_counter()
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    case = run_motional_al_ion.run_motional_al_ion_constrained_floquet_case()
    mode_inputs = _mode_inputs()
    contributions_1e19 = _per_mode_contributions_1e19(case, mode_inputs)
    _check_total_reconstruction(contributions_1e19, case)

    published = loaders.MARSHALL_AL_ION_SECULAR_MOTION_SHIFT
    deviation_sigma = _deviation_sigma(case, published)
    verdict = case.total_kpi_verdict

    per_mode_by_name = {mode.name: mode for mode in case.per_mode}
    predicted_total_1e19 = case.predicted_total_nominal * 1.0e19
    predicted_sigma_1e19 = case.predicted_total_uncertainty_combined_fractional * 1.0e19
    published_nominal_1e19 = published.nominal * 1.0e19
    published_lo_1e19 = published.lo * 1.0e19
    published_hi_1e19 = published.hi * 1.0e19

    # --- Figure/axes setup (created once; per-frame update mutates artists). --
    # Vertical bands, top to bottom: a per-frame mode readout, the two
    # panel titles, the two plot areas, and a footer for the persistent
    # input caption plus the hold-only verdict line -- kept as separate
    # fig-level text objects (not per-axes titles) so long strings never
    # collide with each other regardless of axes width.
    fig = plt.figure(figsize=(11.2, 4.6))
    ax_crystal = fig.add_axes((0.045, 0.16, 0.46, 0.54))
    ax_bar = fig.add_axes((0.62, 0.16, 0.34, 0.54))

    # -- Left panel: the two-ion crystal. --
    ax_crystal.axhline(0.0, color="0.8", lw=0.9, zorder=1)
    ax_crystal.axvline(-ION_HALF_SPACING, color="0.88", lw=0.7, ls=":", zorder=1)
    ax_crystal.axvline(ION_HALF_SPACING, color="0.88", lw=0.7, ls=":", zorder=1)
    ax_crystal.set_xlim(-1.3, 1.3)
    ax_crystal.set_ylim(-1.05, 1.05)
    ax_crystal.set_xticks([])
    ax_crystal.set_yticks([])
    for spine in ax_crystal.spines.values():
        spine.set_visible(False)

    clock_point = ax_crystal.scatter(
        [-ION_HALF_SPACING], [0.0], s=520, c=_COLOR_CLOCK_ION, zorder=3, edgecolors="white", lw=1.2
    )
    partner_point = ax_crystal.scatter(
        [ION_HALF_SPACING], [0.0], s=520, c=_COLOR_PARTNER_ION, zorder=3, edgecolors="white", lw=1.2
    )
    ax_crystal.text(
        -ION_HALF_SPACING,
        -0.85,
        "27Al+ (clock)",
        ha="center",
        va="top",
        fontsize=8.5,
        color=_COLOR_CLOCK_ION,
        fontweight="bold",
    )
    ax_crystal.text(
        ION_HALF_SPACING,
        -0.85,
        "25Mg+ (partner)",
        ha="center",
        va="top",
        fontsize=8.5,
        color=_COLOR_PARTNER_ION,
        fontweight="bold",
    )
    ax_crystal.set_title("Two-ion Al+/Mg+ crystal (six secular normal modes)", fontsize=9, pad=8)
    mode_text = fig.text(0.275, 0.895, "", ha="center", va="center", fontsize=8.5)

    # -- Right panel: the accumulating time-dilation bar. --
    bar_x = 0.0
    bar_width = 0.6
    bar_rects: dict[str, Rectangle] = {}
    bottom = 0.0
    for name in MODE_ORDER:
        color = (
            _COLOR_AXIAL_CONTRIBUTION if name.startswith("axial") else _COLOR_RADIAL_CONTRIBUTION
        )
        rect = Rectangle(
            (bar_x - bar_width / 2.0, bottom),
            bar_width,
            0.0,
            facecolor=color,
            edgecolor="white",
            lw=0.6,
        )
        ax_bar.add_patch(rect)
        bar_rects[name] = rect
        bottom += abs(contributions_1e19[name])

    max_height = abs(published_hi_1e19) * 1.32
    ax_bar.set_xlim(-1.3, 1.5)
    ax_bar.set_ylim(0.0, max_height)
    ax_bar.set_xticks([bar_x])
    ax_bar.set_xticklabels(["this engine\n(coupled, constrained fit)"], fontsize=7.5)
    ax_bar.set_ylabel(r"|shift| contribution ($\times10^{-19}$)", fontsize=8.5)
    ax_bar.set_title("Time-dilation budget (WP35)", fontsize=9, pad=8)

    published_band = Rectangle(
        (-1.3, min(abs(published_lo_1e19), abs(published_hi_1e19))),
        2.8,
        abs(abs(published_hi_1e19) - abs(published_lo_1e19)),
        facecolor=_COLOR_PUBLISHED,
        alpha=0.0,
        zorder=0,
        edgecolor="none",
    )
    ax_bar.add_patch(published_band)
    published_line = ax_bar.axhline(
        abs(published_nominal_1e19), color=_COLOR_PUBLISHED, lw=1.2, ls="--", alpha=0.0
    )

    total_label = ax_bar.text(
        bar_x, 0.0, "", ha="center", va="bottom", fontsize=8, color="0.15", fontweight="bold"
    )
    published_label = ax_bar.text(
        bar_x,
        abs(published_hi_1e19) + max_height * 0.045,
        "",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=_COLOR_PUBLISHED,
        alpha=0.0,
    )
    verdict_text = fig.text(
        0.5,
        0.115,
        "",
        ha="center",
        va="center",
        fontsize=9,
        color="0.1",
        fontweight="bold",
        alpha=0.0,
    )
    fig.text(
        0.5,
        0.035,
        _CAPTION_LINE_1 + "\n" + _CAPTION_LINE_2,
        ha="center",
        va="center",
        fontsize=7,
        color="0.35",
    )

    def update(frame: int):
        mode_index, progress, is_hold = _mode_frame_state(frame)
        active_name = MODE_ORDER[mode_index]
        display_name, axis = MODE_DISPLAY[active_name]
        frequency_hz, n_bar, _n_bar_uncertainty = mode_inputs[active_name]
        mode = per_mode_by_name[active_name]
        clock_participation = mode.participation_clock
        partner_participation = 1.0 - clock_participation
        in_phase = MODE_IN_PHASE[active_name]

        if not is_hold:
            phase = 2.0 * math.pi * CYCLES_PER_MODE * progress
            osc = math.sin(phase)
            clock_amp = AMPLITUDE_SCALE * math.sqrt(clock_participation)
            partner_amp = AMPLITUDE_SCALE * math.sqrt(partner_participation)
            clock_dir = 1.0
            partner_dir = 1.0 if in_phase else -1.0

            if axis == "x":
                clock_xy = (-ION_HALF_SPACING + clock_dir * clock_amp * osc, 0.0)
                partner_xy = (ION_HALF_SPACING + partner_dir * partner_amp * osc, 0.0)
            else:
                clock_xy = (-ION_HALF_SPACING, clock_dir * clock_amp * osc)
                partner_xy = (ION_HALF_SPACING, partner_dir * partner_amp * osc)
            clock_point.set_offsets([clock_xy])
            partner_point.set_offsets([partner_xy])

            mode_text.set_text(
                f"{display_name}: f = {frequency_hz / 1.0e6:.2f} MHz, n_bar = {n_bar:.2f}\n"
                f"participation: Al {clock_participation:.1%} / Mg {partner_participation:.1%}"
            )
        else:
            clock_point.set_offsets([(-ION_HALF_SPACING, 0.0)])
            partner_point.set_offsets([(ION_HALF_SPACING, 0.0)])
            mode_text.set_text("All six modes shown above -- totals at right")

        # -- Right panel: stacked bar growth, mode by mode. --
        bottom_height = 0.0
        for i, name in enumerate(MODE_ORDER):
            full_height = abs(contributions_1e19[name])
            if is_hold or i < mode_index:
                height = full_height
            elif i == mode_index:
                height = full_height * progress
            else:
                height = 0.0
            bar_rects[name].set_y(bottom_height)
            bar_rects[name].set_height(height)
            bottom_height += full_height if (is_hold or i <= mode_index) else 0.0

        running_total = sum(
            abs(contributions_1e19[name])
            for i, name in enumerate(MODE_ORDER)
            if is_hold or i < mode_index
        ) + (abs(contributions_1e19[active_name]) * progress if not is_hold else 0.0)
        if is_hold:
            # Placed inside the bar, just below its top edge: this variant's
            # total lands close to published_label's own position above the
            # bar, and the two labels would collide if both sat up there.
            total_label.set_position((bar_x, running_total - max_height * 0.02))
            total_label.set_va("top")
            total_label.set_color("white")
            total_label.set_fontsize(8)
            total_label.set_text(f"{abs(predicted_total_1e19):.1f} +/- {predicted_sigma_1e19:.1f}")
        else:
            total_label.set_position((bar_x, running_total + max_height * 0.015))
            total_label.set_va("bottom")
            total_label.set_color("0.15")
            total_label.set_fontsize(8)
            total_label.set_text(f"{running_total:.1f}")

        if is_hold:
            hold_local = frame - FRAMES_PER_MODE * N_MODES
            hold_progress = min(1.0, hold_local / (HOLD_FRAMES * 0.4))
            published_band.set_alpha(0.25 * hold_progress)
            published_line.set_alpha(0.9 * hold_progress)
            published_label.set_alpha(hold_progress)
            published_label.set_text(
                f"Marshall et al. published: {abs(published_nominal_1e19):.1f} +/- "
                f"{(published_hi_1e19 - published_nominal_1e19):.1f}"
            )
            verdict_text.set_alpha(hold_progress)
            verdict_text.set_text(
                f"{deviation_sigma:.2f} sigma from the published band ({verdict})"
            )
        else:
            published_band.set_alpha(0.0)
            published_line.set_alpha(0.0)
            published_label.set_alpha(0.0)
            verdict_text.set_alpha(0.0)

        return [
            clock_point,
            partner_point,
            mode_text,
            *bar_rects.values(),
            published_band,
            published_line,
            total_label,
            published_label,
            verdict_text,
        ]

    t0 = time.perf_counter()
    ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, blit=False)
    ani.save(_OUTPUT_PATH, writer=animation.PillowWriter(fps=FPS), dpi=88)
    plt.close(fig)
    t_render_s = time.perf_counter() - t0

    t_wall_total_s = time.perf_counter() - t_wall_start
    size_mb = _OUTPUT_PATH.stat().st_size / 1e6
    print(f"animation render + encode: {t_render_s:.2f} s")
    print(f"total wall time: {t_wall_total_s:.2f} s")
    print(
        f"wrote {_OUTPUT_PATH} ({size_mb:.2f} MB, {N_FRAMES} frames at {FPS} fps, "
        f"{N_FRAMES / FPS:.1f} s loop)"
    )
    print("per-mode contributions (1e-19 units):")
    for name in MODE_ORDER:
        mode = per_mode_by_name[name]
        print(
            f"  {name}: participation={mode.participation_clock:.4f} "
            f"enhancement={mode.enhancement_clock:.4f} contribution={contributions_1e19[name]:+.3f}"
        )
    print(
        f"predicted total: {predicted_total_1e19:+.2f} +/- {predicted_sigma_1e19:.2f} (1e-19); "
        f"published: {published_nominal_1e19:+.2f} +/- "
        f"{(published_hi_1e19 - published_nominal_1e19):.2f} (1e-19); "
        f"deviation={deviation_sigma:.3f} sigma; verdict={verdict}"
    )
    print(
        f"rf_drive_frequency_hz={loaders.MARSHALL_AL_ION_RF_DRIVE_FREQUENCY_HZ:.4e} "
        f"({_RF_DRIVE_MHZ:.2f} MHz)"
    )


if __name__ == "__main__":
    main()
