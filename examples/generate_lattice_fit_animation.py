#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate ``docs/assets/lattice_fit_animation.gif``, the README's third
hero animation.

**What this shows.** One case from the WP38 gradient-based sideband-fit
demonstration (``benchmarks/run_sideband_fit.py``, gated alongside the
differentiable BO+WKB forward model in
``cliffordclock.integrator.sideband_spectrum_jax``): a synthetic Yb-171
sideband spectrum generated from known truth ``(u0=100 E_R, Tr=1.00
uK)``, corrupted with the same deterministic 1% Gaussian noise every run
(seed 0), fit back by ``scipy.optimize.minimize`` supplied exact
``jax.value_and_grad`` gradients of the BO+WKB forward model. Every
number this script displays comes from that real fit, run live:

- **Left panel**: the noisy synthetic spectrum as data points, with the
  model curve redrawn at each point in the optimizer's own path.
- **Right panels**: the two fitted parameters, lattice depth (``u0``) and
  radial temperature (``Tr``), traced across the same optimizer path,
  each against its own truth line. The 1-sigma Laplace band around each
  final value appears once the descent reaches its last frame.

**The same functions `run_one_fit` calls.** This script calls
:func:`run_sideband_fit._forward`, :func:`run_sideband_fit._make_loss_and_grad`,
and :func:`run_sideband_fit.laplace_uncertainties` unchanged, for the same
``(model="bowkb", truth_u0=100.0, truth_tr_k=1.0e-6, seed=0)`` case
already reported in ``benchmarks/results/wp38_sideband_fit.json`` (one
of the module's own 12-case ``TRUTH_GRID`` x ``SEEDS`` x model grid).
The one piece this script adds is a thin wrapper around the jit-compiled
objective that logs every parameter vector `scipy.optimize.minimize`
evaluates, so the optimizer's real path can be drawn; :func:`_check_against_reference`
then confirms the wrapped run reaches the identical optimum, uncertainty,
and Hessian verdict as an unwrapped call to ``run_one_fit`` on the same
case, before either number is trusted for the animation.

**The optimizer's raw path, held frame by frame.** `L-BFGS-B`'s line
search moves non-monotonically: several of its trial points land far
worse than the point before them. This script logs every call, then
keeps only the calls that set a new best chi-squared so far, a monotone
subsequence of the real trajectory. Each kept point holds on screen for
a fixed number of frames and then cuts directly to the next; no
interpolation, easing, or smoothing sits between one fitted curve and
the next.

**Why this differs from the other two hero animations' framing.** Both
``generate_showcase_animation.py`` and ``generate_ion_motion_animation.py``
can claim zero free parameters: every number they animate is read from a
config file or a published table. This animation is a fit. Two
parameters, lattice depth and radial temperature, are solved from the
spectrum, and the caption says so.

Regeneration
------------
Run from an activated project venv (``pip install -e ".[notebooks]"`` --
matplotlib + Pillow only, both already required by that extra; no new
dependency)::

    python examples/generate_lattice_fit_animation.py

Deterministic: fixed seed (0), no wall-clock reads. Two runs on the same
machine produce byte-identical output. Output:
``docs/assets/lattice_fit_animation.gif``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.optimize import minimize  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import run_sideband_fit as rsf  # noqa: E402

_OUTPUT_PATH = REPO_ROOT / "docs" / "assets" / "lattice_fit_animation.gif"

#: The case animated: one row of `run_sideband_fit.TRUTH_GRID` x
#: `run_sideband_fit.SEEDS`, the BO+WKB model, already reported at
#: `benchmarks/results/wp38_sideband_fit.json` as `recovered_u0 =
#: 100.01 +/- 0.49` and `recovered_tr_k = 0.996 +/- 0.028 uK` (a
#: well-conditioned, positive-definite-Hessian case, the condition
#: ``main``'s own assert checks before this hero animates it).
FIT_MODEL: rsf.Model = "bowkb"
TRUTH_U0 = 100.0
TRUTH_TR_K = 1.0e-6
FIT_SEED = 0
assert (TRUTH_U0, TRUTH_TR_K) in rsf.TRUTH_GRID
assert FIT_SEED in rsf.SEEDS

#: Frames-per-second, matching the other two hero animations
#: (`generate_showcase_animation.py`, `generate_ion_motion_animation.py`).
FPS = 10

#: Each kept (monotone-improving) optimizer call holds for this many
#: frames before cutting to the next -- long enough to read, short
#: enough that the descent stays the visible subject (module docstring
#: "The optimizer's raw path, held frame by frame").
FRAMES_PER_ITERATE = 8

#: Final hold: recovered vs. truth, uncertainty bands, caption (2.5 s at
#: FPS=10, inside the brief's 2-3 s hold-frame target).
HOLD_FRAMES = 25

#: Colors matching `paper/figures/common.py`'s own established palette
#: (`COLOR_REFERENCE`/`COLOR_ENGINE`/`COLOR_NEUTRAL`): data/published
#: values in the reference color, this engine's own predicted/model
#: values in the engine color, truth/reference lines neutral.
_COLOR_DATA = "#C1666B"
_COLOR_MODEL = "#2E5FA3"
_COLOR_TRUTH = "#6B6B6B"

_CAPTION_LINE_1 = (
    "The Born-Oppenheimer + WKB lattice model, fit to this spectrum by gradient descent."
)
_CAPTION_LINE_2 = (
    "Two fitted parameters, lattice depth and radial temperature; every number here is "
    "computed live."
)


@dataclass(frozen=True)
class TracedFit:
    """One fit's full traced optimizer path plus its final report."""

    delta_khz: np.ndarray
    noisy_spectrum: np.ndarray
    mono_params: np.ndarray  # (N, 2): [u0, tr_k] at each kept call
    mono_chi2: np.ndarray  # (N,)
    mono_curves: np.ndarray  # (N, n_delta): forward spectrum at each kept call
    recovered_u0: float
    recovered_u0_sigma: float
    recovered_tr_k: float
    recovered_tr_k_sigma: float
    hessian_positive_definite: bool
    n_calls_total: int


def _run_traced_fit(model: rsf.Model, truth_u0: float, truth_tr_k: float, seed: int) -> TracedFit:
    """Run the exact `run_sideband_fit.run_one_fit` case, with every
    optimizer call logged, and reduce the log to its monotone-improving
    subsequence (module docstring "The optimizer's raw path, held frame
    by frame").

    Calls `run_sideband_fit._forward`, `_make_loss_and_grad`, and
    `laplace_uncertainties` unchanged; the only new code here is the
    logging wrapper and the monotone-subsequence filter, both pure
    bookkeeping around the real fit.
    """
    truth_spectrum = np.asarray(rsf._forward(model, jnp.asarray(truth_u0), jnp.asarray(truth_tr_k)))
    rng = np.random.default_rng(seed)
    noisy = truth_spectrum + rng.normal(0.0, rsf.NOISE_SIGMA, size=truth_spectrum.shape)

    objective, hessian_fn = rsf._make_loss_and_grad(model, noisy, rsf.NOISE_SIGMA)

    call_params: list[np.ndarray] = []
    call_chi2: list[float] = []

    def traced_objective(params_np: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = objective(params_np)
        call_params.append(np.array(params_np, dtype=np.float64))
        call_chi2.append(float(value))
        return value, grad

    x0 = np.array([truth_u0 * 1.25, truth_tr_k * 0.6])
    bounds = [(10.0, 300.0), (50e-9, 8e-6)]
    result = minimize(traced_objective, x0, jac=True, method="L-BFGS-B", bounds=bounds)

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

    hessian = np.asarray(hessian_fn(jnp.asarray(result.x)))
    hessian_pd, sigma_u0, sigma_tr = rsf.laplace_uncertainties(hessian)

    mono_curves = np.stack(
        [
            np.asarray(rsf._forward(model, jnp.asarray(u0_i), jnp.asarray(tr_i)))
            for u0_i, tr_i in mono_params
        ]
    )

    return TracedFit(
        delta_khz=np.asarray(rsf.DELTA_GRID_HZ) / 1.0e3,
        noisy_spectrum=noisy,
        mono_params=mono_params,
        mono_chi2=mono_chi2,
        mono_curves=mono_curves,
        recovered_u0=float(result.x[0]),
        recovered_u0_sigma=sigma_u0,
        recovered_tr_k=float(result.x[1]),
        recovered_tr_k_sigma=sigma_tr,
        hessian_positive_definite=hessian_pd,
        n_calls_total=len(call_params),
    )


def _check_against_reference(traced: TracedFit) -> None:
    """Confirms the wrapped, logged fit reaches the same optimum, the same
    Laplace uncertainties, and the same Hessian verdict as an unwrapped
    `run_sideband_fit.run_one_fit` call on the identical case. That
    agreement is what makes the logging wrapper's neutrality verifiable.
    """
    reference = rsf.run_one_fit(FIT_MODEL, TRUTH_U0, TRUTH_TR_K, FIT_SEED)
    assert abs(traced.recovered_u0 - reference.recovered_u0) < 1e-6, (
        "traced fit's recovered u0 disagrees with run_one_fit's own reference case"
    )
    assert abs(traced.recovered_tr_k - reference.recovered_tr_k) < 1e-12, (
        "traced fit's recovered Tr disagrees with run_one_fit's own reference case"
    )
    assert traced.hessian_positive_definite == reference.hessian_positive_definite
    assert abs(traced.recovered_u0_sigma - reference.recovered_u0_uncertainty) < 1e-6
    assert abs(traced.recovered_tr_k_sigma - reference.recovered_tr_k_uncertainty) < 1e-12


def main() -> None:
    t_wall_start = time.perf_counter()
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    traced = _run_traced_fit(FIT_MODEL, TRUTH_U0, TRUTH_TR_K, FIT_SEED)
    _check_against_reference(traced)
    assert traced.hessian_positive_definite, (
        "this case's Hessian carries a non-positive eigenvalue at its reported optimum -- "
        "the Laplace uncertainty bands this hero animates would be invalid"
    )
    t_fit_s = time.perf_counter() - t0

    n_mono = traced.mono_params.shape[0]
    recovered_tr_uk = traced.recovered_tr_k * 1.0e6
    recovered_tr_sigma_uk = traced.recovered_tr_k_sigma * 1.0e6
    truth_tr_uk = TRUTH_TR_K * 1.0e6
    mono_u0 = traced.mono_params[:, 0]
    mono_tr_uk = traced.mono_params[:, 1] * 1.0e6

    # --- Figure/axes setup (created once; per-frame update mutates artists). --
    fig = plt.figure(figsize=(12.0, 4.6))
    ax_spec = fig.add_axes((0.055, 0.16, 0.46, 0.68))
    ax_u0 = fig.add_axes((0.62, 0.57, 0.34, 0.30))
    ax_tr = fig.add_axes((0.62, 0.16, 0.34, 0.30))

    # -- Left panel: noisy spectrum + the descending model curve. --
    ax_spec.scatter(
        traced.delta_khz,
        traced.noisy_spectrum,
        s=9,
        color=_COLOR_DATA,
        alpha=0.75,
        edgecolors="none",
        label="synthetic noisy spectrum",
        zorder=2,
    )
    (model_line,) = ax_spec.plot(
        [], [], color=_COLOR_MODEL, lw=1.6, label="BO+WKB model, current fit", zorder=3
    )
    y_lo = min(traced.noisy_spectrum.min(), traced.mono_curves.min())
    y_hi = max(traced.noisy_spectrum.max(), traced.mono_curves.max())
    y_pad = 0.06 * (y_hi - y_lo)
    ax_spec.set_xlim(traced.delta_khz[0], traced.delta_khz[-1])
    ax_spec.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax_spec.set_xlabel("probe detuning from the carrier (kHz)")
    ax_spec.set_ylabel("excitation probability")
    ax_spec.set_title("Sideband spectrum: fitting live")
    ax_spec.legend(loc="upper right", fontsize=8, frameon=False)
    step_text = ax_spec.text(
        0.02, 0.95, "", transform=ax_spec.transAxes, ha="left", va="top", fontsize=8.5, color="0.2"
    )

    fig.suptitle(
        f"Yb-171 BO+WKB sideband fit -- truth $u_0$={TRUTH_U0:.0f} $E_R$, "
        f"$T_r$={truth_tr_uk:.2f} uK, seed={FIT_SEED} (1% Gaussian noise)",
        fontsize=9.5,
        y=0.985,
    )

    # -- Right panels: the two fitted-parameter traces. --
    u0_pad = 0.06 * (mono_u0.max() - mono_u0.min())
    ax_u0.set_xlim(0, n_mono - 1)
    ax_u0.set_ylim(mono_u0.min() - u0_pad, mono_u0.max() + u0_pad)
    ax_u0.axhline(TRUTH_U0, color=_COLOR_TRUTH, lw=1.0, ls="--", zorder=1)
    ax_u0.set_ylabel(r"$u_0$ ($E_R$)", fontsize=9)
    ax_u0.set_title("Fitted parameters, converging", fontsize=9.5, pad=6)
    ax_u0.tick_params(labelbottom=False)
    (u0_trace,) = ax_u0.plot([], [], color=_COLOR_MODEL, lw=1.4, marker="o", ms=3, zorder=3)
    u0_band = ax_u0.axhspan(0.0, 0.0, color=_COLOR_MODEL, alpha=0.0, zorder=2)
    u0_readout = ax_u0.text(
        0.98,
        0.92,
        "",
        transform=ax_u0.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        alpha=0.0,
        bbox={"boxstyle": "round", "fc": "white", "ec": "none", "pad": 0.2},
    )

    tr_pad = 0.06 * (mono_tr_uk.max() - mono_tr_uk.min())
    ax_tr.set_xlim(0, n_mono - 1)
    ax_tr.set_ylim(mono_tr_uk.min() - tr_pad, mono_tr_uk.max() + tr_pad)
    ax_tr.axhline(truth_tr_uk, color=_COLOR_TRUTH, lw=1.0, ls="--", zorder=1)
    ax_tr.set_xlabel("optimizer step (monotone-improving calls only)", fontsize=8.5)
    ax_tr.set_ylabel(r"$T_r$ ($\mu$K)", fontsize=9)
    (tr_trace,) = ax_tr.plot([], [], color=_COLOR_MODEL, lw=1.4, marker="o", ms=3, zorder=3)
    tr_band = ax_tr.axhspan(0.0, 0.0, color=_COLOR_MODEL, alpha=0.0, zorder=2)
    tr_readout = ax_tr.text(
        0.98,
        0.92,
        "",
        transform=ax_tr.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        alpha=0.0,
        bbox={"boxstyle": "round", "fc": "white", "ec": "none", "pad": 0.2},
    )

    fig.text(
        0.5,
        0.045,
        _CAPTION_LINE_1 + "\n" + _CAPTION_LINE_2,
        ha="center",
        va="center",
        fontsize=7.5,
        color="0.3",
    )

    n_descent_frames = n_mono * FRAMES_PER_ITERATE
    n_frames = n_descent_frames + HOLD_FRAMES

    def update(frame: int):
        idx = min(frame // FRAMES_PER_ITERATE, n_mono - 1)
        is_hold = frame >= n_descent_frames

        model_line.set_data(traced.delta_khz, traced.mono_curves[idx])
        u0_trace.set_data(np.arange(idx + 1), mono_u0[: idx + 1])
        tr_trace.set_data(np.arange(idx + 1), mono_tr_uk[: idx + 1])
        step_text.set_text(
            f"optimizer step {idx + 1}/{n_mono}, $\\chi^2$ = {traced.mono_chi2[idx]:.1f}"
        )

        if is_hold:
            hold_local = frame - n_descent_frames
            hold_progress = min(1.0, hold_local / (HOLD_FRAMES * 0.5))
            # `axhspan` returns a `Rectangle` in a blended transform (x in
            # axes-fraction, y in data coordinates): only y0/height move.
            u0_band.set_y(traced.recovered_u0 - traced.recovered_u0_sigma)
            u0_band.set_height(2.0 * traced.recovered_u0_sigma)
            u0_band.set_alpha(0.22 * hold_progress)
            u0_readout.set_alpha(hold_progress)
            u0_readout.set_text(
                f"{traced.recovered_u0:.2f} $\\pm$ {traced.recovered_u0_sigma:.2f} "
                f"(truth {TRUTH_U0:.2f})"
            )

            tr_band.set_y(recovered_tr_uk - recovered_tr_sigma_uk)
            tr_band.set_height(2.0 * recovered_tr_sigma_uk)
            tr_band.set_alpha(0.22 * hold_progress)
            tr_readout.set_alpha(hold_progress)
            tr_readout.set_text(
                f"{recovered_tr_uk:.3f} $\\pm$ {recovered_tr_sigma_uk:.3f} "
                f"(truth {truth_tr_uk:.2f})"
            )
        else:
            u0_band.set_alpha(0.0)
            u0_readout.set_alpha(0.0)
            tr_band.set_alpha(0.0)
            tr_readout.set_alpha(0.0)

        return [
            model_line,
            u0_trace,
            tr_trace,
            step_text,
            u0_band,
            u0_readout,
            tr_band,
            tr_readout,
        ]

    t0 = time.perf_counter()
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    ani.save(_OUTPUT_PATH, writer=animation.PillowWriter(fps=FPS), dpi=145)
    plt.close(fig)
    t_render_s = time.perf_counter() - t0

    t_wall_total_s = time.perf_counter() - t_wall_start
    size_mb = _OUTPUT_PATH.stat().st_size / 1e6
    print(f"fit + trace: {t_fit_s:.2f} s ({traced.n_calls_total} objective calls, {n_mono} kept)")
    print(f"animation render + encode: {t_render_s:.2f} s")
    print(f"total wall time: {t_wall_total_s:.2f} s")
    print(
        f"wrote {_OUTPUT_PATH} ({size_mb:.2f} MB, {n_frames} frames at {FPS} fps, "
        f"{n_frames / FPS:.1f} s loop)"
    )
    print(
        f"case: model={FIT_MODEL} truth_u0={TRUTH_U0:.1f} truth_tr_k={TRUTH_TR_K:.2e} "
        f"seed={FIT_SEED}"
    )
    print(
        f"recovered: u0={traced.recovered_u0:.2f} +/- {traced.recovered_u0_sigma:.2f}, "
        f"tr_uk={recovered_tr_uk:.3f} +/- {recovered_tr_sigma_uk:.3f}, "
        f"hessian_positive_definite={traced.hessian_positive_definite}"
    )


if __name__ == "__main__":
    main()
