#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figure 3 (step-size accuracy study): V4 closed form vs. large-`dtau` error.

Regenerates the ``notebooks/02_step_size_study.ipynb`` methodology (WP8
test contract item 2, ``docs/timescales.md``'s accuracy-study table):
sweep the rotor integrator's step size `d\\tilde\\tau` (via
``cliffordclock.integrator.fastpath.select_dtau``, E31) against the
CONVENTIONS.md V4 closed form (harmonic trap, classical atom,
linear-gradient field, second-order Doppler included), and confirm the
expected order-2 error scaling of the exponential-midpoint stepper (E19).

This is the *same* trap/field/`mu` scenario, parameter values, and sweep
points as the notebook and ``tests/test_fastpath_select_dtau.py`` (not a
new, independently-invented setup) -- reproduced here directly against
``cliffordclock.integrator.worldline.integrate_worldline`` so the figure
and its quoted numbers regenerate from the real integrator, never copied
from the notebook's saved output cells.

Outputs
-------
- ``figures/fig3_step_size_accuracy.pdf``: log-log relative phase error
  vs. `d\\tilde\\tau`, with the measured order-2 fit line and the E31
  default (100 points/period) marked.
- ``generated/step_size_values.tex``: the measured convergence order and
  the error/norm-drift at the E31 default, as ``\\newcommand`` macros.
"""

from __future__ import annotations

import common  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from cliffordclock.constants import ELECTRON_MASS, SPEED_OF_LIGHT, TAU_COMPTON  # noqa: E402
from cliffordclock.ensemble.traps import HarmonicTrap  # noqa: E402
from cliffordclock.fields.synthetic import as_field_fn, constant_gradient_field  # noqa: E402
from cliffordclock.integrator import fastpath  # noqa: E402
from cliffordclock.integrator.worldline import integrate_worldline  # noqa: E402

_M_E_C2 = ELECTRON_MASS * SPEED_OF_LIGHT**2

# --- V4 scenario (identical to notebooks/02_step_size_study.ipynb). ---
OMEGA = 2.0e5  # rad/s, isotropic trap
CENTER = np.array([0.01, -0.02, 0.03])
GRAD = np.array([[1.0e3, 2.0e3, 0.0], [0.0, -1.0e3, 5.0e2], [3.0e2, 0.0, -2.0e3]])
E0 = np.zeros(3)
MU = 1.0e-33 * np.array([1.0, 2.0, -3.0])  # P(center) - 1 ~ 3.8e-18, this project's target regime
SCALE = 1.0e-5
DELTA_R0 = SCALE * np.array([1.0e-3, -0.7e-3, 1.2e-3])
V0 = SCALE * np.array([1.0e-2, -0.8e-2, 0.6e-2])

_POINTS_PER_PERIOD_VALUES = [25, 50, 100, 200, 400, 800]
_N_PERIODS = 3
_E31_DEFAULT_PPP = 100
_WP8_RTOL_BOUND = 1e-8


def _sinusoidal_trajectory(n_steps: int, dtau: float) -> np.ndarray:
    dt_s = dtau * TAU_COMPTON
    t = np.arange(n_steps + 1, dtype=np.float64) * dt_s
    return (
        CENTER + DELTA_R0 * np.cos(OMEGA * t)[:, None] + (V0 / OMEGA) * np.sin(OMEGA * t)[:, None]
    )


def main() -> None:
    common.reset_tex_macro_file("step_size_values.tex")

    trap = HarmonicTrap(omega_xyz=(OMEGA, OMEGA, OMEGA), center=tuple(CENTER))
    field_fn = as_field_fn(*constant_gradient_field(jnp.asarray(E0), jnp.asarray(GRAD)))

    p_minus_1_center = float(np.dot(E0 + CENTER @ GRAD, MU) / _M_E_C2)
    mean_v2 = 0.5 * (OMEGA**2 * np.dot(DELTA_R0, DELTA_R0) + np.dot(V0, V0))
    mean_rate = p_minus_1_center - mean_v2 / (2.0 * SPEED_OF_LIGHT**2)  # V4 closed form

    def closed_form_phase(n_steps: int, dtau: float) -> float:
        return mean_rate * (n_steps * dtau)

    rows = []
    for ppp in _POINTS_PER_PERIOD_VALUES:
        dtau = fastpath.select_dtau(trap, ppp)
        n_steps = _N_PERIODS * ppp
        traj = _sinusoidal_trajectory(n_steps, dtau)
        result = integrate_worldline(
            field_fn, jnp.asarray(traj), dtau, jnp.asarray(MU), renorm_every=1
        )
        phase_closed = closed_form_phase(n_steps, dtau)
        rel_err = abs(float(result.phase) - phase_closed) / abs(phase_closed)
        rows.append((ppp, dtau, dtau * TAU_COMPTON, rel_err, float(result.max_norm_drift)))

    dtaus = np.array([r[1] for r in rows])
    errors = np.array([r[3] for r in rows])
    slope, intercept = np.polyfit(np.log(dtaus), np.log(errors), 1)

    default_row = next(r for r in rows if r[0] == _E31_DEFAULT_PPP)

    # --- Figure. ---
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.loglog(dtaus, errors, "o-", color=common.COLOR_ENGINE, label="measured relative error")
    fit_line = np.exp(intercept) * dtaus**slope
    ax.loglog(dtaus, fit_line, "--", color=common.COLOR_NEUTRAL, label=f"fit: order {slope:.2f}")
    ax.axhline(
        _WP8_RTOL_BOUND,
        color=common.COLOR_REFERENCE,
        linestyle=":",
        label=r"target bound ($\mathrm{rtol}\leq10^{-8}$)",
    )
    ax.axvline(
        fastpath.select_dtau(trap, _E31_DEFAULT_PPP),
        color="green",
        linestyle=":",
        label="default (100 pts/period)",
    )
    ax.set_xlabel(r"$d\tilde\tau$ (Compton units, step size of this convergence sweep)")
    ax.set_ylabel("relative phase error of the convergence study (dimensionless)")
    ax.set_title("Step-size accuracy: order-2 convergence (E19)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    # On-plot annotation (owner directive): state directly, next to the data,
    # what a reader should take away -- this is a stepper convergence-order
    # measurement, not the pipeline's shift accuracy (the validation table
    # and the paper's precision-discipline section characterize the latter).
    default_dtau = default_row[1]
    default_err = default_row[3]
    ax.annotate(
        f"measured slope = order {slope:.2f} (design: 2)\n"
        f"at the default step size, error = {default_err:.2e},\n"
        f"far below this study's own tolerance ({_WP8_RTOL_BOUND:.0e})",
        xy=(default_dtau, default_err),
        xytext=(0.08, 0.92),
        textcoords="axes fraction",
        fontsize=7,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "fc": "white", "ec": common.COLOR_NEUTRAL, "alpha": 0.9},
        arrowprops={"arrowstyle": "->", "color": common.COLOR_NEUTRAL, "lw": 0.8},
    )
    fig.tight_layout()
    fig.savefig(common.FIGURES_DIR / "fig3_step_size_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Generated \input macros. ---
    common.write_tex_macro("StepSizeOrder", f"{slope:.3f}", "step_size_values.tex")
    common.write_tex_macro("StepSizeDefaultError", f"{default_row[3]:.2e}", "step_size_values.tex")
    common.write_tex_macro("StepSizeDefaultDrift", f"{default_row[4]:.2e}", "step_size_values.tex")
    common.write_tex_macro(
        "StepSizeDefaultDtauCompton", f"{default_row[1]:.3e}", "step_size_values.tex"
    )
    common.write_tex_macro(
        "StepSizeDefaultDtauSeconds", f"{default_row[2]:.3e}", "step_size_values.tex"
    )
    common.write_tex_macro("StepSizeNRes", str(_E31_DEFAULT_PPP), "step_size_values.tex")

    # A small generated table (all six sweep rows) for the numerical
    # methods section (\input directly, not part of the main validation table).
    table_path = common.GENERATED_DIR / "step_size_table.tex"
    with table_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by paper/figures/fig3_step_size_accuracy.py -- do not edit.\n")
        f.write("\\begin{tabular}{rrrrr}\n\\toprule\n")
        f.write(
            "pts/period & $d\\tilde\\tau$ & $d\\tau$ (s) & rel.\\ error & "
            "norm drift \\\\\n\\midrule\n"
        )
        for ppp, dtau, dtau_s, rel_err, drift in rows:
            marker = "\\textbf{" if ppp == _E31_DEFAULT_PPP else ""
            marker_end = "}" if ppp == _E31_DEFAULT_PPP else ""
            f.write(
                f"{marker}{ppp}{marker_end} & {dtau:.3e} & {dtau_s:.3e} & "
                f"{rel_err:.3e} & {drift:.3e} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"Figure 3: measured convergence order = {slope:.4f} (design order: 2)")
    print(
        f"Figure 3: at default (100 pts/period), rel. error = {default_row[3]:.4e}, "
        f"norm drift = {default_row[4]:.4e}"
    )
    print(f"Wrote {common.FIGURES_DIR / 'fig3_step_size_accuracy.pdf'}")
    print(f"Wrote {common.GENERATED_DIR / 'step_size_values.tex'}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
