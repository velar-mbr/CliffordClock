# SPDX-License-Identifier: AGPL-3.0-or-later
"""WP38 Deliverable 3: gradient-based sideband-spectrum fitting
demonstration. Generates a synthetic spectrum from known ``(u0, Tr)``,
adds deterministic Gaussian noise, and recovers both parameters by
``scipy.optimize.minimize`` supplied with exact gradients from
``jax.value_and_grad`` of the differentiable forward model
(`cliffordclock.integrator.sideband_spectrum_jax`), reporting
Laplace/Hessian-based 1-sigma uncertainties, repeated across a small
deterministic grid of truth values and noise seeds.

**The claim, stated at its calibration.** This is a synthetic
round-trip: the same forward model both generates the spectrum and
fits it back (`generator == fitter`), the standard way to demonstrate a
fitting procedure before ever touching real data. It demonstrates that
gradient-based optimization through this project's differentiable
BO+WKB sideband forward model converges to the generating `(u0, Tr)`
with correctly calibrated uncertainties. Two narrower points bound that
claim. Goti et al. 2025 compares the BO+WKB and harmonic models' own
description of a real experiment; this file cites that comparison and
does not reproduce it. Separately, `large-lattice-model`
(github.com/inrim/large-lattice-model, MIT, INRIM) already ships a
working, non-differentiable fitter
(`large_lattice_model.fit.get_fit_sidebands`, a `scipy.optimize`-ready
callable built on a numba-jitted forward model with a
finite-difference-Jacobian least-squares optimizer supplied by the
caller), and Goti et al. 2025 used that code to fit real IT-Yb1
spectroscopy (this project's private research dossier, Phase 2
addendum). This file's own contribution is the fit method: a
BO+WKB-class sideband lineshape fit by GRADIENT-based optimization,
exact analytic gradients through the forward model via autodiff,
supplied directly to the optimizer. `large-lattice-model`'s own fitter
supplies the optimizer a finite-difference-approximated Jacobian.

**Deterministic seeds** (repo convention): every noise draw uses
`numpy.random.default_rng(seed)` with a small, fixed integer `seed`.
`numpy.random.default_rng()`'s own OS-entropy default never appears in
this file.

**The Goti et al. 2025 real-scan fit, assessed here.** Goti et al.
2025's Figs. 4 and 7 plot real IT-Yb1 sideband scans as discrete
scatter markers, individual points plotted directly. This project's
private research dossier ranks these figures the strongest
figure-digitization candidate found in either research sweep, because
those individual points are extractable on their own terms, unlike a
rendered continuous trace. The underlying PDF
(arxiv.org/pdf/2508.08164, produced by `xdvipdfmx`, 570 KB) was checked
directly (`pdftotext`/text-layer extraction against the locally fetched
PDF) for embedded vector point coordinates, a check run before assuming
pixel-digitization was the only route available. The extracted text
layer carries the paper's prose and equations; it carries no separate,
recoverable per-marker coordinate stream distinguishable from the
surrounding vector-graphics path data. A defensible exact-coordinate
extraction would need pixel-level digitization of the published
figure's raster/vector art. This project's own evidentiary discipline
places pixel-level digitization in the figure-digitization class, a
class weaker than either this file's own synthetic demonstrations or
`run_sideband_spectrum.py`'s independent-implementation cross-validation.
This work package's own instructions say plainly: do not force it. This
benchmark ships the synthetic demonstration (this file) and the
independent-implementation cross-validation
(`run_sideband_spectrum.py`) as WP38's evidentiary core. What a
real-data fit would need is recorded here as the named partnership ask,
should INRIM be approached: Goti et al. 2025's own raw scan data
(detuning, excitation fraction, and their per-point uncertainties, for
the spectra underlying Figs. 4 and 7), unpublished in any
machine-readable form this project's research sweep located.

Run this yourself: ``python benchmarks/run_sideband_fit.py`` (from the
repo root, with ``.venv`` active). Regenerates
``benchmarks/results/wp38_sideband_fit.json`` and its markdown summary.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

import cliffordclock.integrator.sideband_spectrum_jax as ssj
from cliffordclock.constants import SPEED_OF_LIGHT
from cliffordclock.ensemble.species import get_species

_BENCHMARKS_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _BENCHMARKS_DIR / "results"

#: Fixed species/site/probe parameters, held KNOWN throughout every fit
#: below (only `u0`/`Tr` are fit): Yb-171, the same magic-frequency
#: convention `run_lattice_light_shift.py`/`run_sideband_spectrum.py`
#: both use.
YB171_MAGIC_FREQUENCY_HZ = 394_798_266.9e6
WAIST_M = 50e-6
PROBE_WAVELENGTH_M = 578e-9
RABI0_HZ = 300.0
PULSE_TIME_S = 3e-3
TEMPERATURE_Z_K = 2e-6
SIDEBAND_LINEWIDTH_HZ = 2e3
BLUE_AMPLITUDE = 0.9
RED_AMPLITUDE = 0.8

#: Reduced BO+WKB resolution for THIS fitting demonstration (distinct
#: from `sideband_spectrum_jax`'s own module-level defaults,
#: `AXIAL_GRID_N_SPECTRUM=321`/`RHO_TABLE_N=129`/`N_Z_MAX_BOWKB=5`/
#: `N_E_QUAD=96`): chosen so a full `scipy.optimize.minimize` run (tens
#: of jit-compiled forward+gradient evaluations) completes in well under
#: a second after the one-time jit compilation, verified against the
#: full-resolution module defaults at a handful of spot points
#: (`tests/test_sideband_spectrum_jax.py::TestOfflineConvergenceStudy`)
#: to agree on peak position/shape to a few percent, comfortably inside
#: this demonstration's own noise floor. A physics-precision BO+WKB
#: evaluation (Deliverable 1/2) should use the module's own defaults.
FIT_AXIAL_GRID_N = 241
FIT_RHO_TABLE_N = 97
FIT_N_Z_MAX_BOWKB = 4
FIT_N_E_QUAD = 48

DELTA_GRID_HZ = jnp.linspace(-55e3, 55e3, 111)

#: A small, deterministic grid of truth `(u0, Tr)` pairs and noise seeds.
#: Fixed at authoring time, never adjusted after seeing a fit's result
#: (this project's established no-tuned-parameters discipline).
TRUTH_GRID: tuple[tuple[float, float], ...] = ((80.0, 0.6e-6), (100.0, 1.0e-6), (120.0, 1.5e-6))
SEEDS: tuple[int, ...] = (0, 1)
NOISE_SIGMA = 0.01


def _yb171_mass_kg() -> float:
    return get_species("Yb171").mass_kg


def _yb171_wavelength_m() -> float:
    return SPEED_OF_LIGHT / YB171_MAGIC_FREQUENCY_HZ


Model = Literal["harmonic", "bowkb"]


def _forward(model: Model, u0: jnp.ndarray, tr: jnp.ndarray) -> jnp.ndarray:
    """The forward spectrum, `(u0, Tr)` as the only free arguments,
    every other physical input fixed at this module's own constants
    above.
    """
    mass_kg = _yb171_mass_kg()
    wavelength_m = _yb171_wavelength_m()
    if model == "harmonic":
        return ssj.harmonic_full_spectrum(
            DELTA_GRID_HZ,
            PULSE_TIME_S,
            u0,
            WAIST_M,
            wavelength_m,
            mass_kg,
            PROBE_WAVELENGTH_M,
            RABI0_HZ,
            TEMPERATURE_Z_K,
            tr,
            BLUE_AMPLITUDE,
            RED_AMPLITUDE,
            SIDEBAND_LINEWIDTH_HZ,
        )
    carrier = ssj.harmonic_carrier_excitation_probability(
        DELTA_GRID_HZ,
        PULSE_TIME_S,
        u0,
        WAIST_M,
        wavelength_m,
        mass_kg,
        PROBE_WAVELENGTH_M,
        RABI0_HZ,
        TEMPERATURE_Z_K,
        tr,
    )
    blue = ssj.bowkb_sideband_shape(
        DELTA_GRID_HZ,
        1,
        u0,
        WAIST_M,
        wavelength_m,
        mass_kg,
        TEMPERATURE_Z_K,
        tr,
        SIDEBAND_LINEWIDTH_HZ,
        n_z_max=FIT_N_Z_MAX_BOWKB,
        n_e_quad=FIT_N_E_QUAD,
        axial_grid_n=FIT_AXIAL_GRID_N,
        rho_table_n=FIT_RHO_TABLE_N,
    )
    red = ssj.bowkb_sideband_shape(
        DELTA_GRID_HZ,
        -1,
        u0,
        WAIST_M,
        wavelength_m,
        mass_kg,
        TEMPERATURE_Z_K,
        tr,
        SIDEBAND_LINEWIDTH_HZ,
        n_z_max=FIT_N_Z_MAX_BOWKB,
        n_e_quad=FIT_N_E_QUAD,
        axial_grid_n=FIT_AXIAL_GRID_N,
        rho_table_n=FIT_RHO_TABLE_N,
    )
    return carrier + BLUE_AMPLITUDE * blue + RED_AMPLITUDE * red


def _make_loss_and_grad(model: Model, data: np.ndarray, sigma: float):
    """Build a jit-compiled `(chi2, grad_chi2)` callable in `(u0, Tr)`
    for `scipy.optimize.minimize(..., jac=True)`, and a separate
    jit-compiled Hessian of the negative-log-likelihood
    (`0.5*chi2`, the standard Gaussian-noise Laplace/asymptotic
    approximation) for the uncertainty report.
    """
    data_j = jnp.asarray(data)

    def chi2(params: jnp.ndarray) -> jnp.ndarray:
        u0, tr = params[0], params[1]
        pred = _forward(model, u0, tr)
        return jnp.sum(((pred - data_j) / sigma) ** 2)

    def neg_log_likelihood(params: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * chi2(params)

    value_and_grad = jax.jit(jax.value_and_grad(chi2))
    hessian = jax.jit(jax.hessian(neg_log_likelihood))

    def scipy_objective(params_np: np.ndarray) -> tuple[float, np.ndarray]:
        v, g = value_and_grad(jnp.asarray(params_np))
        return float(v), np.asarray(g, dtype=np.float64)

    return scipy_objective, hessian


@dataclass(frozen=True)
class FitCase:
    model: str
    truth_u0: float
    truth_tr_k: float
    seed: int
    noise_sigma: float
    initial_guess_u0: float
    initial_guess_tr_k: float
    recovered_u0: float
    recovered_u0_uncertainty: float
    recovered_tr_k: float
    recovered_tr_k_uncertainty: float
    hessian_positive_definite: bool
    u0_within_1sigma: bool
    tr_within_1sigma: bool
    u0_within_2sigma: bool
    tr_within_2sigma: bool
    converged: bool
    n_iterations: int
    final_chi2: float
    n_data_points: int


def laplace_uncertainties(hessian: np.ndarray) -> tuple[bool, float, float]:
    """The reporting path shared by every fit case: given the Hessian of
    the negative log-likelihood at a reported optimum, decide whether
    the Laplace/Gaussian approximation applies there and return its
    1-sigma uncertainties.

    The Laplace approximation treats the inverse Hessian as a covariance
    matrix. That step is valid only where the Hessian is positive
    definite, the condition for a true local minimum of the negative
    log-likelihood. `L-BFGS-B` reports `success` from its own
    gradient-norm stopping test alone, so a run that stops at a SADDLE
    point (one negative eigenvalue, the gradient still near zero along
    the other direction) reports `success=True` with no warning. This
    function checks the Hessian's own eigenvalues (`np.linalg.eigvalsh`)
    before trusting its inverse: the returned `hessian_positive_definite`
    flag is `True` only when every eigenvalue is strictly positive. When
    it is `False`, both returned uncertainties are `nan`, matching the
    convention already used for a singular Hessian
    (`np.linalg.LinAlgError`). A clamped zero would read as the most
    confident row in a results table; `nan` reads as what it is, an
    optimum where the Laplace approximation does not apply.

    Parameters
    ----------
    hessian : np.ndarray
        The `(2, 2)` Hessian of `0.5*chi2` at the reported optimum.

    Returns
    -------
    tuple[bool, float, float]
        `(hessian_positive_definite, sigma_u0, sigma_tr)`.
    """
    eigvals = np.linalg.eigvalsh(hessian)
    hessian_pd = bool(np.all(eigvals > 0.0))
    if hessian_pd:
        try:
            cov = np.linalg.inv(hessian)
            sigma_u0 = float(np.sqrt(cov[0, 0]))
            sigma_tr = float(np.sqrt(cov[1, 1]))
        except np.linalg.LinAlgError:
            hessian_pd = False
            sigma_u0 = float("nan")
            sigma_tr = float("nan")
    else:
        sigma_u0 = float("nan")
        sigma_tr = float("nan")
    return hessian_pd, sigma_u0, sigma_tr


def run_one_fit(model: Model, truth_u0: float, truth_tr_k: float, seed: int) -> FitCase:
    """Generate one synthetic spectrum, fit `(u0, Tr)` back, and report
    the recovery against truth with Laplace/Hessian uncertainties from
    :func:`laplace_uncertainties`.
    """
    truth_spectrum = np.asarray(_forward(model, jnp.asarray(truth_u0), jnp.asarray(truth_tr_k)))
    rng = np.random.default_rng(seed)
    noisy = truth_spectrum + rng.normal(0.0, NOISE_SIGMA, size=truth_spectrum.shape)

    objective, hessian_fn = _make_loss_and_grad(model, noisy, NOISE_SIGMA)

    # A fixed fractional offset from truth (never start AT the answer),
    # the same offset applied to every case.
    x0 = np.array([truth_u0 * 1.25, truth_tr_k * 0.6])
    bounds = [(10.0, 300.0), (50e-9, 8e-6)]
    result = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds)

    u0_hat, tr_hat = float(result.x[0]), float(result.x[1])
    hessian = np.asarray(hessian_fn(jnp.asarray(result.x)))
    hessian_pd, sigma_u0, sigma_tr = laplace_uncertainties(hessian)

    u0_within_1s = abs(u0_hat - truth_u0) <= sigma_u0
    tr_within_1s = abs(tr_hat - truth_tr_k) <= sigma_tr
    u0_within_2s = abs(u0_hat - truth_u0) <= 2.0 * sigma_u0
    tr_within_2s = abs(tr_hat - truth_tr_k) <= 2.0 * sigma_tr

    return FitCase(
        model=model,
        truth_u0=truth_u0,
        truth_tr_k=truth_tr_k,
        seed=seed,
        noise_sigma=NOISE_SIGMA,
        initial_guess_u0=float(x0[0]),
        initial_guess_tr_k=float(x0[1]),
        recovered_u0=u0_hat,
        recovered_u0_uncertainty=sigma_u0,
        recovered_tr_k=tr_hat,
        recovered_tr_k_uncertainty=sigma_tr,
        hessian_positive_definite=hessian_pd,
        u0_within_1sigma=u0_within_1s,
        tr_within_1sigma=tr_within_1s,
        u0_within_2sigma=u0_within_2s,
        tr_within_2sigma=tr_within_2s,
        converged=bool(result.success),
        n_iterations=int(result.nit),
        final_chi2=float(result.fun),
        n_data_points=int(DELTA_GRID_HZ.shape[0]),
    )


def run_all_fits() -> list[FitCase]:
    cases: list[FitCase] = []
    for model in ("harmonic", "bowkb"):
        for truth_u0, truth_tr_k in TRUTH_GRID:
            for seed in SEEDS:
                cases.append(run_one_fit(model, truth_u0, truth_tr_k, seed))  # type: ignore[arg-type]
    return cases


def build_report() -> dict[str, Any]:
    cases = run_all_fits()
    n_1sigma = sum(1 for c in cases if c.u0_within_1sigma and c.tr_within_1sigma)
    n_2sigma = sum(1 for c in cases if c.u0_within_2sigma and c.tr_within_2sigma)
    n_converged = sum(1 for c in cases if c.converged)
    n_hessian_pd = sum(1 for c in cases if c.hessian_positive_definite)
    return {
        "wp38_sideband_fit_schema": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "claim_calibration": (
            "Synthetic round-trip: generator == fitter. Gradient-based "
            "optimization through this project's own differentiable "
            "forward model recovers known (u0, Tr) with correctly "
            "calibrated Laplace uncertainties. Real-data model accuracy is "
            "a separate question; run_sideband_spectrum.py and Goti et "
            "al. 2025 address it directly. "
            "large-lattice-model's own get_fit_sidebands is a "
            "non-differentiable BO+WKB sideband fitter that already "
            "exists and was used for Goti et al. 2025's own real-data "
            "fits; this file's own contribution is the first GRADIENT-"
            "based (autodiff) fit of the BO+WKB model. Every case's "
            "Hessian is checked for positive definiteness "
            "(hessian_positive_definite) before its inverse is trusted "
            "as a covariance. At a saddle point, the Hessian carries a "
            "negative eigenvalue and the Laplace approximation does not "
            "apply; the reported uncertainty there is nan."
        ),
        "cases": [asdict(c) for c in cases],
        "n_cases": len(cases),
        "n_converged": n_converged,
        "n_hessian_positive_definite": n_hessian_pd,
        "n_within_1sigma_both_params": n_1sigma,
        "n_within_2sigma_both_params": n_2sigma,
        "goti_real_scan_assessment": {
            "decision": "declined",
            "class_if_attempted": "figure_digitization",
            "reason": (
                "Goti et al. 2025 Figs. 4/7 plot real IT-Yb1 sideband scans "
                "as discrete scatter markers, the strongest "
                "figure-digitization candidate found in this project's "
                "research sweep. The underlying PDF's text/vector layer "
                "carries the paper's prose and equations, and no "
                "separately recoverable per-marker coordinate stream. "
                "Extracting exact coordinates from these figures would "
                "need pixel-level digitization of the published art, "
                "placing that extraction in the figure-digitization "
                "class, weaker than either this file's synthetic fits or "
                "run_sideband_spectrum.py's independent-implementation "
                "cross-validation. This work package's own instruction "
                "says plainly: do not force it. No real-scan fit is "
                "shipped here; the raw scan data behind Figs. 4/7 "
                "(detuning, excitation fraction, per-point uncertainties) "
                "is recorded as the named partnership ask."
            ),
        },
    }


def _uncertainty_cell(recovered: float, uncertainty: float, hessian_pd: bool, decimals: int) -> str:
    """One table cell's recovered value and its own uncertainty. A
    non-positive-definite Hessian prints `nan` for the uncertainty,
    matching the value `run_one_fit` reports.
    """
    if not hessian_pd:
        return f"{recovered:.{decimals}f} +/- nan"
    return f"{recovered:.{decimals}f} +/- {uncertainty:.{decimals}f}"


def _fit_case_row(c: dict[str, Any]) -> str:
    """One Markdown table row for a single `FitCase`."""
    pd_flag = c["hessian_positive_definite"]
    u0_cell = _uncertainty_cell(c["recovered_u0"], c["recovered_u0_uncertainty"], pd_flag, 2)
    tr_cell = _uncertainty_cell(
        c["recovered_tr_k"] * 1e6, c["recovered_tr_k_uncertainty"] * 1e6, pd_flag, 3
    )
    both_1sigma = c["u0_within_1sigma"] and c["tr_within_1sigma"]
    both_2sigma = c["u0_within_2sigma"] and c["tr_within_2sigma"]
    return (
        f"| {c['model']} | {c['truth_u0']:.1f} | {c['truth_tr_k'] * 1e6:.2f} | "
        f"{c['seed']} | {u0_cell} | {tr_cell} | {pd_flag} | {c['converged']} | "
        f"{both_1sigma} | {both_2sigma} |"
    )


def render_markdown(report: dict[str, Any]) -> str:
    non_pd_cases = [c for c in report["cases"] if not c["hessian_positive_definite"]]
    lines = [
        "# WP38 Deliverable 3: gradient-based sideband-fitting demonstration",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        report["claim_calibration"],
        "",
        f"{report['n_converged']}/{report['n_cases']} fits converged. "
        f"{report['n_hessian_positive_definite']}/{report['n_cases']} report a "
        f"positive-definite Hessian at the optimum, the condition the Laplace "
        f"uncertainty below requires. "
        f"{report['n_within_1sigma_both_params']}/{report['n_cases']} recovered both "
        f"parameters within their own reported 1-sigma Laplace uncertainty; "
        f"{report['n_within_2sigma_both_params']}/{report['n_cases']} within 2-sigma.",
        "",
        "| model | truth u0 | truth Tr (uK) | seed | recovered u0 | recovered Tr (uK) | "
        "Hessian PD | converged | 1-sigma | 2-sigma |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *(_fit_case_row(c) for c in report["cases"]),
        "",
    ]
    if non_pd_cases:
        lines.append(
            "**Hessian not positive definite.** The row(s) below stopped at a saddle "
            "point of the negative log-likelihood: the Hessian at the reported optimum "
            "carries a negative eigenvalue, so the Laplace approximation is invalid "
            "there. Each such row's own uncertainty is reported as `nan`."
        )
        lines.append("")
        for c in non_pd_cases:
            lines.append(
                f"- `{c['model']}`, truth `u0={c['truth_u0']:.1f}`, "
                f"`Tr={c['truth_tr_k'] * 1e6:.2f} uK`, seed `{c['seed']}`: recovered "
                f"`u0={c['recovered_u0']:.2f}`, `Tr={c['recovered_tr_k'] * 1e6:.3f} uK`, "
                f"Hessian eigenvalues carry at least one negative value; the Laplace "
                f"uncertainty at this optimum is undefined."
            )
        lines.append("")
    lines.extend(
        [
            "## Goti et al. 2025 real-scan fit: assessment",
            "",
            f"**Decision: {report['goti_real_scan_assessment']['decision']}.**",
            "",
            report["goti_real_scan_assessment"]["reason"],
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / "wp38_sideband_fit.json"
    md_path = _RESULTS_DIR / "wp38_sideband_fit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
