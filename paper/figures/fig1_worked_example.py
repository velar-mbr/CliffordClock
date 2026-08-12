#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figure 1 (worked example): the Lodewyck-style patch-field demo.

Runs the *real* pipeline on ``examples/realistic_lattice_sr87.yaml`` (the
WP11 "bring your own field" worked example: a small uniform residual bias
field plus six Gaussian patch-potential contributions on in-vacuum
dielectric surfaces ~25 mm from an Sr-87 trap, modeling the partially-
discharged intermediate regime of the Lodewyck et al. (IEEE Trans. UFFC
59, 411 (2012)) SYRTE stray-field event) and produces:

1. ``figures/fig1_worked_example.pdf`` -- a two-panel figure: (a) the
   field magnitude along a line through the trap center (the same
   ``FieldSmoother`` the pipeline itself uses to evaluate the field,
   applied to the committed CSV export), and (b) the resulting DC-Stark
   pivot shift ``P(r) - 1`` along the same line.
2. ``generated/worked_example_values.tex`` -- every quoted number in the
   paper's Worked Example section, as ``\\newcommand`` macros, computed
   directly from this run (never hand-typed).

Model idealization caveat (stated in the paper's text, not just here):
each "patch" in the generating scenario is an isotropic 3D Gaussian
potential bump, not an oriented 2D charged-surface element -- see
``examples/generate_patch_field.py``'s module docstring, "Model
idealization (stated plainly)".
"""

from __future__ import annotations

import common  # noqa: E402  (sets up sys.path; must be imported first)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from cliffordclock.ensemble.species import get_species  # noqa: E402
from cliffordclock.fields.io import load_field_csv  # noqa: E402
from cliffordclock.fields.smoother import FieldSmoother  # noqa: E402
from cliffordclock.pipeline import PipelineConfig, run_pipeline_full  # noqa: E402

_CONFIG_PATH = common.REPO_ROOT / "examples" / "realistic_lattice_sr87.yaml"
_CSV_PATH = common.REPO_ROOT / "examples" / "patch_field_sr87.csv"

#: Half-extent of the line-slice plot, meters -- matches the generator's
#: exported grid half-extent (examples/generate_patch_field.py,
#: GRID_HALF_EXTENT_M) so the slice never queries outside the fit's bbox.
_SLICE_HALF_EXTENT_M = 2.5e-4
_SLICE_N_POINTS = 400


def _fmt_sci(x: float, sig: int = 4) -> str:
    """Render ``x`` as LaTeX scientific notation, e.g. ``-7.723\\times10^{-19}``."""
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10.0**exponent)
    return f"{mantissa:.{sig - 1}f}\\times10^{{{exponent}}}"


def main() -> None:
    common.reset_tex_macro_file("worked_example_values.tex")

    # --- Run the real pipeline (the exact shipped WP11 example config). ---
    config = PipelineConfig.from_yaml(_CONFIG_PATH)
    result = run_pipeline_full(config)
    report = result.report
    species = get_species(report.species_name)

    # --- Field slice along x through the trap center, via the same ---
    # --- FieldSmoother.fit/evaluate the pipeline itself uses. ---
    grid = load_field_csv(_CSV_PATH)
    smoother = FieldSmoother.fit(grid, smoothing=0.0)

    x = np.linspace(-_SLICE_HALF_EXTENT_M, _SLICE_HALF_EXTENT_M, _SLICE_N_POINTS)
    positions = np.zeros((_SLICE_N_POINTS, 3), dtype=np.float64)
    positions[:, 0] = x
    e_vec, _grad = smoother.evaluate(positions)
    e_vec = np.asarray(e_vec)
    e_mag = np.linalg.norm(e_vec, axis=-1)

    k_s = species.resolve_stark_coefficient_hz_per_v2_m2()
    pivot_minus_1 = (k_s / species.clock_frequency_hz) * e_mag**2

    e_center = float(e_mag[np.argmin(np.abs(x))])

    # --- Figure: two panels sharing the x-axis. ---
    fig, (ax_field, ax_shift) = plt.subplots(2, 1, figsize=(5.0, 5.2), sharex=True)

    ax_field.plot(x * 1e6, e_mag, color=common.COLOR_ENGINE, lw=1.6)
    ax_field.axvline(0.0, color=common.COLOR_NEUTRAL, lw=0.8, ls=":")
    ax_field.set_ylabel(r"$|E(x,0,0)|$ (V/m)")
    ax_field.set_title("Worked example: patch-potential stray field (Sr-87)")

    ax_shift.plot(x * 1e6, pivot_minus_1, color=common.COLOR_REFERENCE, lw=1.6)
    ax_shift.axvline(0.0, color=common.COLOR_NEUTRAL, lw=0.8, ls=":")
    ax_shift.axhline(report.mean_fractional_shift, color=common.COLOR_ENGINE, lw=1.0, ls="--")
    ax_shift.set_xlabel(r"$x$ ($\mu$m), trap center at origin")
    ax_shift.set_ylabel(r"$P(x,0,0)-1 = \Delta\nu/\nu_0$")

    fig.tight_layout()
    fig.savefig(common.FIGURES_DIR / "fig1_worked_example.pdf")
    plt.close(fig)

    # --- Generated \input macros (every number quoted in the text). ---
    common.write_tex_macro(
        "WorkedExampleFieldCenter", f"{e_center:.2f}", "worked_example_values.tex"
    )
    common.write_tex_macro(
        "WorkedExampleShift",
        _fmt_sci(report.mean_fractional_shift),
        "worked_example_values.tex",
    )
    common.write_tex_macro(
        "WorkedExampleShiftSem", _fmt_sci(report.shift_std_error), "worked_example_values.tex"
    )
    common.write_tex_macro(
        # Note: avoid "T2" as a substring in LaTeX macro names -- it
        # collides with hyperref's font-encoding detection and produces a
        # spurious "Missing \begin{document}" error (confirmed via a
        # minimal reproduction during this WP's PDF build).
        "WorkedExampleTtwoStar",
        f"{report.t2_star_s:.2f}",
        "worked_example_values.tex",
    )
    common.write_tex_macro(
        "WorkedExampleEnsembleSize", str(report.ensemble_size), "worked_example_values.tex"
    )
    common.write_tex_macro(
        "WorkedExampleInterrogationTime",
        f"{report.interrogation_time_s:.1f}",
        "worked_example_values.tex",
    )
    common.write_tex_macro(
        "WorkedExampleDeltaAlpha",
        _fmt_sci(species.delta_alpha_dc_si),
        "worked_example_values.tex",
    )

    print(f"Figure 1: |E(center)| = {e_center:.3f} V/m")
    print(
        f"Figure 1: mean_fractional_shift = {report.mean_fractional_shift:+.6e} "
        f"+/- {report.shift_std_error:.3e} (SEM), t2_star_s = {report.t2_star_s:.4f}"
    )
    print(f"Wrote {common.FIGURES_DIR / 'fig1_worked_example.pdf'}")
    print(f"Wrote {common.GENERATED_DIR / 'worked_example_values.tex'}")


if __name__ == "__main__":
    main()
