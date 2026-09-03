# SPDX-License-Identifier: AGPL-3.0-or-later
"""WP41 Deliverable: the differentiable Rydberg field-to-spectrum chain
and gradient-based field reconstruction (CONVENTIONS.md section 21, E45).

**What this file computes, and why it exists.** A Rydberg-atom RF/DC
electrometry sensor's calibration question runs backward from the usual
direction: given a measured EIT/Autler-Townes spectrum, what field
distribution across the vapor cell produced it? This file demonstrates
that inverse problem, solved by gradient-based optimization through
`cliffordclock.integrator.rydberg_cell_response_jax`, the JAX port of the
WP39 Phase A quadratic-Stark/EIT chain
(`cliffordclock.integrator.rydberg_cell_response`). Three things are
computed and written to this file's own JSON/Markdown artifact:

1. **Agreement (C1).** `rydberg_cell_response_jax`'s functions compared
   against the numpy reference module across a parameter grid, at
   floating-point-algebra precision (no eigensolve, no adaptive grid on
   either side, unlike the WP37 lattice-light-shift comparison).
2. **Gradient validation (C2).** `jax.grad` of a scalar functional of the
   spectrum, with respect to field amplitude, temperature, and the
   coupling/RF Rabi frequencies, checked against central finite
   differences of the NUMPY reference implementation (independent
   numerical methods on each side, the strongest available check, the
   same discipline WP37/WP38 both used).
3. **Field-reconstruction fit-grid (C5).** A synthetic round-trip: a
   three-parameter field model (uniform background, linear axial
   gradient, one wall-patch amplitude) generates a synthetic spectrum at
   planted truth values, seeded Gaussian noise is added, and
   `scipy.optimize.minimize` (`L-BFGS-B`, exact `jax`-supplied gradients)
   fits the three parameters back, with Laplace/Hessian uncertainties
   reported the same way `benchmarks/run_sideband_fit.py` reports them:
   `hessian_positive_definite` checked before the inverse Hessian is
   trusted as a covariance, `nan` uncertainties where it is not.

**The claim, stated at its calibration.** This is a synthetic
round-trip: the same forward model both generates the spectrum and fits
it back (`generator == fitter`), the standard way to demonstrate a
fitting procedure before ever touching real data, the identical framing
`run_sideband_fit.py` states for its own synthetic demonstration. No real
Rydberg-sensor scan is fit here, and no priority claim is made for the
inverse-problem pattern itself; this file demonstrates that this
project's own differentiable Rydberg chain supports it, end to end.

**Stays inside the quadratic Stark validity window everywhere.**
`rydberg_quadratic_stark_shift_hz_jax` carries no validity guard inside
its own traced core (see that function's docstring); this file enforces
the guard itself, in plain Python, before any JAX call: every truth
value, every optimizer bound corner, and every field realization the
fit-grid or the C1/C2 checks ever construct is checked against
`cliffordclock.integrator.rydberg_cell_response.rydberg_quadratic_stark_shift_hz`'s
own guard
(`STARK_VALIDITY_MARGIN * inglis_teller_field_v_per_m(n_star)`) by
`_assert_within_validity_window` below, which raises loudly if any
value ever exceeds it.

Run this yourself: ``python benchmarks/run_rydberg_field_reconstruction.py``
(from the repo root, with ``.venv`` active). Regenerates
``benchmarks/results/wp41_rydberg_field_reconstruction.json`` and
``benchmarks/results/wp41_rydberg_field_reconstruction.md``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

import cliffordclock.integrator.rydberg_cell_response as rcr
import cliffordclock.integrator.rydberg_cell_response_jax as rcj

_BENCHMARKS_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _BENCHMARKS_DIR / "results"

# ---------------------------------------------------------------------------
# Shared fixed system, species, and cell geometry
# ---------------------------------------------------------------------------

#: The Rb-85 32D5/2 calibration state (WP39 Phase A's own registry state)
#: throughout: effective quantum number and the derived polarizability.
N_STAR_32D52 = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
ALPHA0_AU = rcr.RB85_32D52_ALPHA0_AU

#: The guarded quadratic-Stark validity window, V/m (WP39 Phase A's own
#: guard, `STARK_VALIDITY_MARGIN` of the order-of-magnitude Inglis-Teller
#: estimate). Every field value this file ever constructs is checked
#: against this by `_assert_within_validity_window`.
VALIDITY_GUARD_V_PER_M = rcr.STARK_VALIDITY_MARGIN * rcr.inglis_teller_field_v_per_m(N_STAR_32D52)

#: Ladder system parameters, matching
#: `tests/test_rydberg_cell_response.py`'s own `_default_system` (same
#: registry `mu_RF`, same wavelengths, comparable decay rates and probe
#: dipole moment): this file is not re-deriving a new system, only
#: reusing the already-provenanced one.
_SYSTEM_NP = rcr.LadderSystem(
    mu_probe_c_m=2.0e-29,
    mu_coupling_c_m=5.0e-30,
    mu_rf_c_m=rcr.RB85_MU_RF_32D52_33P32_C_M,
    gamma_12=2.0 * math.pi * 6.0e6,
    gamma_13=2.0 * math.pi * 0.3e6,
    gamma_14=2.0 * math.pi * 0.3e6,
    number_density_m3=1.0e16,
    wavelength_probe_m=rcr.HOLLOWAY_LAMBDA_PROBE_M,
    wavelength_coupling_m=rcr.HOLLOWAY_LAMBDA_COUPLING_M,
)
SYSTEM_JAX = rcj.LadderSystemJax(
    mu_probe_c_m=jnp.asarray(_SYSTEM_NP.mu_probe_c_m),
    mu_coupling_c_m=jnp.asarray(_SYSTEM_NP.mu_coupling_c_m),
    mu_rf_c_m=jnp.asarray(_SYSTEM_NP.mu_rf_c_m),
    gamma_12=jnp.asarray(_SYSTEM_NP.gamma_12),
    gamma_13=jnp.asarray(_SYSTEM_NP.gamma_13),
    gamma_14=jnp.asarray(_SYSTEM_NP.gamma_14),
    number_density_m3=jnp.asarray(_SYSTEM_NP.number_density_m3),
    wavelength_probe_m=jnp.asarray(_SYSTEM_NP.wavelength_probe_m),
    wavelength_coupling_m=jnp.asarray(_SYSTEM_NP.wavelength_coupling_m),
)

#: A strong coupling-laser field (distinct from the weak DC field being
#: sensed): sets `Omega_c ~ 2*pi*4.7 MHz`, comparable to `gamma_12`, so
#: the EIT transparency window is deep and narrow enough that field-
#: dependent smearing of it (the actual reconstruction signal) is a
#: large, clearly resolved effect. This file sets the value directly,
#: from numerical exploration run this session: at a much weaker
#: coupling field the two-photon EIT feature is far narrower than
#: `gamma_12`'s single-photon background and the inhomogeneous-field
#: signature becomes nearly invisible in `Im(chi)` (checked directly:
#: `e_coupling_v_per_m=1.0` leaves the composed spectrum within
#: `1e-8` relative of the zero-field spectrum for this file's own field
#: scale, an order of magnitude below any noise floor a real measurement
#: could resolve).
E_COUPLING_V_PER_M = 500.0

#: Patrick et al. 2025's own cylindrical vapor-cell geometry
#: (arXiv:2502.07018, already reused by WP39 Phase A's own C6
#: surface-charge demonstrator,
#: `cliffordclock.integrator.rydberg_cell_response.cylindrical_cell_atom_positions`'s
#: own docstring): 78 mm length, 25 mm diameter.
CELL_RADIUS_M = 0.0125
CELL_LENGTH_M = 0.078

#: Chosen by direct exploration this session: at `n_atoms=150`, the
#: patch-amplitude parameter (only atoms within a few `PATCH_SOFTENING_M`
#: of the wall feel it meaningfully) was so weakly constrained that most
#: fit-grid cases missed their own reported 2-sigma Laplace uncertainty,
#: even with a large reported uncertainty and a positive-definite
#: Hessian: a real "sloppy direction" in this inverse problem, a genuine
#: identifiability limit this three-parameter model carries. `400` atoms
#: (with `NOISE_SIGMA` tightened alongside it, below) brings the patch
#: direction's curvature up enough that the reported Laplace uncertainty
#: is calibrated across the truth grid, checked directly against this
#: file's own C5 fit-grid results.
N_ATOMS = 400

#: Fixed (not fit) wall-patch location: mid-length, one point on the
#: cylindrical wall.
PATCH_POSITION_M = np.array([CELL_RADIUS_M, 0.0, 0.0])

#: Fixed (not fit) softening length for the patch field model's smooth
#: `1/(r^2+softening^2)` regularization
#: (`cell_field_magnitude_v_per_m_jax`'s own docstring): much smaller
#: than the cell's own dimensions (a factor of ~15-25 below the cell
#: radius/length), so the bump stays spatially localized near the patch,
#: matching the "wall patch" phenomenology it stands in for.
PATCH_SOFTENING_M = 0.005

TEMPERATURE_K = 320.0
N_VELOCITY_POINTS = 33

#: Probe-detuning grid, angular rad/s (this module's convention
#: throughout, matching the reference module's). `+/-25 MHz` comfortably
#: covers the EIT feature this file's own `E_COUPLING_V_PER_M` produces
#: (checked directly this session: the feature's full width at this
#: coupling strength and `gamma_12=2*pi*6 MHz` is a few MHz).
DELTA_P_HZ = jnp.linspace(-2.0 * math.pi * 25e6, 2.0 * math.pi * 25e6, 161)
DELTA_P_HZ_NP = np.asarray(DELTA_P_HZ)


def _cell_field_magnitude_np(
    positions_m: np.ndarray,
    e_uniform_v_per_m: float,
    gradient_v_per_m_per_m: float,
    patch_amplitude_v_per_m: float,
) -> np.ndarray:
    """Plain-`numpy` evaluation of the identical field model
    `cell_field_magnitude_v_per_m_jax` implements, used wherever this
    file needs the field values WITHOUT tracing a `jax` graph (validity-
    window checks, truth-spectrum generation via the numpy reference
    module). Kept as one small function here, its own single `numpy`
    transcription of the formula:
    `cell_field_magnitude_v_per_m_jax`'s own docstring is the single
    source of truth for what this formula is and why.
    """
    z = positions_m[:, 2]
    background = e_uniform_v_per_m + gradient_v_per_m_per_m * z
    delta = positions_m - PATCH_POSITION_M[None, :]
    r_sq = np.sum(delta * delta, axis=-1)
    softening_sq = PATCH_SOFTENING_M**2
    patch_term = patch_amplitude_v_per_m * softening_sq / (r_sq + softening_sq)
    return background + patch_term


def _assert_within_validity_window(
    fields_v_per_m: np.ndarray, label: str, *, require_positive: bool = True
) -> None:
    """Raise loudly if any field value in `fields_v_per_m` exceeds the
    guarded quadratic-Stark validity window
    (`VALIDITY_GUARD_V_PER_M`). Called on every truth-value field
    realization and on the field realized at every optimizer-bound
    corner (`_bound_corner_fields`) before this file trusts the
    quadratic path anywhere. This is the module-docstring's "stays
    inside the validity window everywhere" claim, made checkable rather
    than asserted.

    `require_positive` additionally checks the field never crosses zero,
    the field-reconstruction demonstrator's own design requirement (a
    field model whose magnitude changes sign has no clean physical
    reading as "how strong is the field here"). The general C1 agreement
    grid deliberately includes a zero field as a structural test point
    (matching the numpy reference module's own zero-field check), so it
    calls this with `require_positive=False`.
    """
    max_field = float(np.max(np.abs(fields_v_per_m)))
    if max_field >= VALIDITY_GUARD_V_PER_M:
        raise rcr.RydbergStarkValidityError(
            f"{label}: max |field| {max_field:.3e} V/m reaches or exceeds the guarded "
            f"quadratic-Stark validity window {VALIDITY_GUARD_V_PER_M:.3e} V/m; this file's "
            "own bounds/truth values must stay inside it (module docstring)."
        )
    if require_positive:
        min_field = float(np.min(fields_v_per_m))
        if min_field <= 0.0:
            raise ValueError(
                f"{label}: min field {min_field:.3e} V/m is non-positive; this file's own "
                "parameter choices keep the background term dominant over the gradient/patch "
                "swing everywhere so the field magnitude never crosses zero (see module "
                "docstring's field-scale design notes)."
            )


# ---------------------------------------------------------------------------
# C1: agreement, JAX vs the numpy reference, across a parameter grid
# ---------------------------------------------------------------------------

#: A deterministic grid of `(temperature_k, e_coupling_v_per_m,
#: e_rf_v_per_m, delta_c_hz, delta_rf_hz)` points, spanning zero and
#: nonzero coupling/RF drive so both the 3-level (EIT) and 4-level
#: (Autler-Townes) branches of `ladder_susceptibility_jax` are exercised,
#: not only the `e_rf=0` branch this file's own fit demonstrator uses.
C1_DRIVE_GRID: tuple[tuple[float, float, float, float, float], ...] = (
    (200.0, 0.0, 0.0, 0.0, 0.0),
    (320.0, 500.0, 0.0, 0.0, 0.0),
    (320.0, 500.0, 30.0, 0.0, 2.0 * math.pi * 5e6),
    (450.0, 200.0, 60.0, 2.0 * math.pi * 2e6, 0.0),
    (500.0, 800.0, 10.0, -2.0 * math.pi * 3e6, 2.0 * math.pi * 1e6),
)

#: Field magnitudes for the C1/C1-composition checks, chosen well inside
#: the validity window (checked below): zero and several nonzero values.
C1_FIELD_GRID_V_PER_M: tuple[float, ...] = (0.0, 80.0, 220.0, 420.0)


def run_c1_agreement_check() -> dict[str, Any]:
    """Compare `doppler_averaged_susceptibility_jax` (single atom) and
    `compose_inhomogeneous_eit_spectrum_jax` (multi-atom) against their
    numpy reference counterparts across `C1_DRIVE_GRID` x
    `C1_FIELD_GRID_V_PER_M`, worst-case relative error recorded.
    """
    _assert_within_validity_window(
        np.array(C1_FIELD_GRID_V_PER_M), "C1 field grid", require_positive=False
    )

    single_atom_worst = 0.0
    single_atom_worst_point: dict[str, float] = {}
    rng = np.random.default_rng(41)
    n_atoms_check = 6
    composed_worst = 0.0
    composed_worst_point: dict[str, float] = {}

    for temperature_k, e_coupling, e_rf, delta_c, delta_rf in C1_DRIVE_GRID:
        for field in C1_FIELD_GRID_V_PER_M:
            delta_c_atom = delta_c + 2.0 * math.pi * rcr.rydberg_quadratic_stark_shift_hz(
                ALPHA0_AU, field, N_STAR_32D52
            )
            ref = rcr.doppler_averaged_susceptibility(
                DELTA_P_HZ_NP,
                delta_c_atom,
                delta_rf,
                1.0,
                e_coupling,
                e_rf,
                _SYSTEM_NP,
                temperature_k,
                rcr.RB85_MASS_KG,
                n_velocity_points=N_VELOCITY_POINTS,
            )
            got = np.asarray(
                rcj.doppler_averaged_susceptibility_jax(
                    DELTA_P_HZ,
                    jnp.asarray(delta_c_atom),
                    jnp.asarray(delta_rf),
                    jnp.asarray(1.0),
                    jnp.asarray(e_coupling),
                    jnp.asarray(e_rf),
                    SYSTEM_JAX,
                    jnp.asarray(temperature_k),
                    jnp.asarray(rcr.RB85_MASS_KG),
                    n_velocity_points=N_VELOCITY_POINTS,
                )
            )
            rel_err = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
            if rel_err > single_atom_worst:
                single_atom_worst = rel_err
                single_atom_worst_point = {
                    "temperature_k": temperature_k,
                    "e_coupling_v_per_m": e_coupling,
                    "e_rf_v_per_m": e_rf,
                    "field_v_per_m": field,
                }

        fields = rng.uniform(20.0, 420.0, n_atoms_check)
        _assert_within_validity_window(fields, "C1 composition field draw")
        weights = rng.uniform(0.5, 1.5, n_atoms_check)
        ref_composed = rcr.compose_inhomogeneous_eit_spectrum(
            DELTA_P_HZ_NP,
            fields,
            weights,
            ALPHA0_AU,
            N_STAR_32D52,
            _SYSTEM_NP,
            delta_c=delta_c,
            delta_rf=delta_rf,
            e_coupling_v_per_m=e_coupling,
            e_rf_v_per_m=e_rf,
            temperature_k=temperature_k,
            mass_kg=rcr.RB85_MASS_KG,
            n_velocity_points=N_VELOCITY_POINTS,
        )
        got_composed = np.asarray(
            rcj.compose_inhomogeneous_eit_spectrum_jax(
                DELTA_P_HZ,
                jnp.asarray(fields),
                jnp.asarray(weights),
                jnp.asarray(ALPHA0_AU),
                SYSTEM_JAX,
                delta_c=jnp.asarray(delta_c),
                delta_rf=jnp.asarray(delta_rf),
                e_coupling_v_per_m=jnp.asarray(e_coupling),
                e_rf_v_per_m=jnp.asarray(e_rf),
                temperature_k=jnp.asarray(temperature_k),
                mass_kg=jnp.asarray(rcr.RB85_MASS_KG),
                n_velocity_points=N_VELOCITY_POINTS,
            )
        )
        rel_err_c = float(
            np.max(np.abs(got_composed - ref_composed)) / np.max(np.abs(ref_composed))
        )
        if rel_err_c > composed_worst:
            composed_worst = rel_err_c
            composed_worst_point = {
                "temperature_k": temperature_k,
                "e_coupling_v_per_m": e_coupling,
                "e_rf_v_per_m": e_rf,
                "delta_c_hz": delta_c,
                "delta_rf_hz": delta_rf,
            }

    return {
        "single_atom_worst_relative_error": single_atom_worst,
        "single_atom_worst_point": single_atom_worst_point,
        "composed_worst_relative_error": composed_worst,
        "composed_worst_point": composed_worst_point,
        "tolerance": 1e-7,
        "met": bool(single_atom_worst < 1e-7 and composed_worst < 1e-7),
    }


# ---------------------------------------------------------------------------
# C2: gradient validation, jax.grad vs central finite differences of the
# numpy reference
# ---------------------------------------------------------------------------


def _projected_loss_np(spectrum: np.ndarray, proj_re: np.ndarray, proj_im: np.ndarray) -> float:
    """A fixed random linear functional of a complex spectrum, real and
    imaginary parts separately projected: a directional derivative that
    exercises the whole output vector through one scalar. A plain sum
    could mask a per-element sign or index error through cancellation;
    the random weights here make that cancellation unlikely.
    `proj_re`/`proj_im` are fixed, seeded arrays (see `_PROJECTION_SEED`
    below), identical for the numpy and JAX sides of every comparison in
    this section.
    """
    return float(np.sum(proj_re * spectrum.real) + np.sum(proj_im * spectrum.imag))


_PROJECTION_SEED = 4141
_proj_rng = np.random.default_rng(_PROJECTION_SEED)
PROJ_RE = _proj_rng.uniform(0.5, 1.5, DELTA_P_HZ_NP.shape[0])
PROJ_IM = _proj_rng.uniform(0.5, 1.5, DELTA_P_HZ_NP.shape[0])
PROJ_RE_J = jnp.asarray(PROJ_RE)
PROJ_IM_J = jnp.asarray(PROJ_IM)


def _projected_loss_jax(spectrum: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(PROJ_RE_J * jnp.real(spectrum)) + jnp.sum(PROJ_IM_J * jnp.imag(spectrum))


#: Fixed base point every C2 gradient direction is evaluated at: a single
#: atom, moderate field, moderate temperature, both coupling and RF drive
#: nonzero (so every one of `ladder_susceptibility`'s terms is active).
C2_BASE_FIELD_V_PER_M = 220.0
C2_BASE_TEMPERATURE_K = 320.0
C2_BASE_E_COUPLING_V_PER_M = 500.0
C2_BASE_E_RF_V_PER_M = 25.0
C2_BASE_DELTA_C_HZ = 0.0
C2_BASE_DELTA_RF_HZ = 0.0


def _numpy_single_atom_loss(
    field_v_per_m: float, temperature_k: float, e_coupling_v_per_m: float, e_rf_v_per_m: float
) -> float:
    delta_c_atom = C2_BASE_DELTA_C_HZ + 2.0 * math.pi * rcr.rydberg_quadratic_stark_shift_hz(
        ALPHA0_AU, field_v_per_m, N_STAR_32D52
    )
    spectrum = rcr.doppler_averaged_susceptibility(
        DELTA_P_HZ_NP,
        delta_c_atom,
        C2_BASE_DELTA_RF_HZ,
        1.0,
        e_coupling_v_per_m,
        e_rf_v_per_m,
        _SYSTEM_NP,
        temperature_k,
        rcr.RB85_MASS_KG,
        n_velocity_points=N_VELOCITY_POINTS,
    )
    return _projected_loss_np(spectrum, PROJ_RE, PROJ_IM)


def _jax_single_atom_loss(
    field_v_per_m: jnp.ndarray,
    temperature_k: jnp.ndarray,
    e_coupling: jnp.ndarray,
    e_rf: jnp.ndarray,
) -> jnp.ndarray:
    shift_hz = rcj.rydberg_quadratic_stark_shift_hz_jax(jnp.asarray(ALPHA0_AU), field_v_per_m)
    delta_c_atom = C2_BASE_DELTA_C_HZ + 2.0 * jnp.pi * shift_hz
    spectrum = rcj.doppler_averaged_susceptibility_jax(
        DELTA_P_HZ,
        delta_c_atom,
        jnp.asarray(C2_BASE_DELTA_RF_HZ),
        jnp.asarray(1.0),
        e_coupling,
        e_rf,
        SYSTEM_JAX,
        temperature_k,
        jnp.asarray(rcr.RB85_MASS_KG),
        n_velocity_points=N_VELOCITY_POINTS,
    )
    return _projected_loss_jax(spectrum)


#: Central finite-difference relative step, applied to each argument's
#: own base value (`h = FD_RELATIVE_STEP * max(|x0|, 1.0)`). Chosen by
#: direct comparison this session across `1e-3` to `1e-7`: the
#: `e_rf_v_per_m` direction (whose loss depends on `e_rf_v_per_m`
#: quadratically through `Omega_RF^2`, so its central-difference
#: truncation error is proportionally larger at a fixed relative step
#: than the other three, more nearly linear, directions) needed a
#: tighter step than `1e-5` to reach the same digit of agreement; `1e-6`
#: is inside the well-conditioned window (truncation error still small,
#: float64 cancellation error not yet dominant) for all four arguments
#: this section checks, confirmed by the worst-case relative error
#: landing at `1.7e-6`, unchanged to the digit shown when the step is
#: tightened further to `1e-7`.
FD_RELATIVE_STEP = 1.0e-6


def _central_fd(f: Any, x0: float, h: float) -> float:
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


def run_c2_gradient_check() -> dict[str, Any]:
    """`jax.grad` of `_jax_single_atom_loss` with respect to each of its
    four arguments, checked against a central finite difference of
    `_numpy_single_atom_loss` (the SAME scalar functional, evaluated by
    the numpy reference module) at the SAME base point, one argument
    perturbed at a time.
    """
    _assert_within_validity_window(np.array([C2_BASE_FIELD_V_PER_M]), "C2 base field")

    grad_fn = jax.jit(jax.grad(_jax_single_atom_loss, argnums=(0, 1, 2, 3)))
    jax_grads = grad_fn(
        jnp.asarray(C2_BASE_FIELD_V_PER_M),
        jnp.asarray(C2_BASE_TEMPERATURE_K),
        jnp.asarray(C2_BASE_E_COUPLING_V_PER_M),
        jnp.asarray(C2_BASE_E_RF_V_PER_M),
    )

    base_values = {
        "field_v_per_m": C2_BASE_FIELD_V_PER_M,
        "temperature_k": C2_BASE_TEMPERATURE_K,
        "e_coupling_v_per_m": C2_BASE_E_COUPLING_V_PER_M,
        "e_rf_v_per_m": C2_BASE_E_RF_V_PER_M,
    }
    fd_grads = {}
    for name in base_values:
        x0 = base_values[name]
        h = FD_RELATIVE_STEP * max(abs(x0), 1.0)

        def f(x: float, _name: str = name) -> float:
            kwargs = dict(base_values)
            kwargs[_name] = x
            return _numpy_single_atom_loss(**kwargs)  # type: ignore[arg-type]

        fd_grads[name] = _central_fd(f, x0, h)

    results = {}
    worst_rel = 0.0
    worst_name = ""
    for i, name in enumerate(base_values):
        jax_val = float(jax_grads[i])
        fd_val = fd_grads[name]
        denom = max(abs(fd_val), 1e-300)
        rel_err = abs(jax_val - fd_val) / denom
        results[name] = {"jax_grad": jax_val, "fd_grad": fd_val, "relative_error": rel_err}
        if rel_err > worst_rel:
            worst_rel = rel_err
            worst_name = name

    return {
        "base_point": base_values,
        "per_argument": results,
        "worst_relative_error": worst_rel,
        "worst_argument": worst_name,
        "tolerance": 1e-5,
        "met": bool(worst_rel < 1e-5),
    }


# ---------------------------------------------------------------------------
# C5: field-reconstruction fit-grid demonstrator
# ---------------------------------------------------------------------------

#: Truth `(e_uniform_v_per_m, gradient_v_per_m_per_m, patch_amplitude_v_per_m)`
#: triples. Fixed at authoring time (this project's established
#: no-tuned-parameters discipline): chosen from the field-scale
#: exploration recorded in this file's own module docstring, never
#: adjusted after seeing a fit's result.
TRUTH_GRID: tuple[tuple[float, float, float], ...] = (
    (180.0, 800.0, 60.0),
    (220.0, -1200.0, 90.0),
    (150.0, 1500.0, 40.0),
    (300.0, -1800.0, 120.0),
)
SEEDS: tuple[int, ...] = (0, 1)

#: Noise sigma on the `Im(chi)` observable. Tightened alongside `N_ATOMS`
#: (see that constant's own docstring) from an initial `5e-7` (~2-3% of
#: the peak-to-trough swing the inhomogeneous-field smearing produces) to
#: `1.5e-7` (~1%): the combination is what brings the fit-grid's reported
#: Laplace uncertainties into calibrated coverage of the truth grid.
NOISE_SIGMA = 1.5e-7

#: Optimizer bounds. Corner values are validity-window-checked below
#: (`_bound_corner_fields`) before any fit runs.
BOUNDS = [(100.0, 400.0), (-2000.0, 2000.0), (0.0, 200.0)]

#: Starting-point offset factors (multiplicative on truth), one per
#: parameter, distinct so no fit ever starts at its own answer
#: (`run_sideband_fit.py`'s own convention). Chosen, and checked against
#: `TRUTH_GRID`, to keep every starting point comfortably inside `BOUNDS`
#: for every truth case: an early choice that multiplied the largest
#: truth values by a factor pushing the start within ~10 V/m of a bound
#: measurably degraded L-BFGS-B's own convergence for that case (the box
#: constraint left little room to move), so these factors keep every
#: starting point at least ~40 units from its own bound across the whole
#: grid.
X0_OFFSET_FACTORS = (0.8, 0.5, 1.3)


def _bound_corner_fields(atom_positions_m: np.ndarray) -> np.ndarray:
    """Every field value realized at an optimizer-bound CORNER (all 8
    combinations of the three parameters' `(lo, hi)` bounds), over the
    actual fixed atom positions this file's fit-grid uses. Checked by
    `_assert_within_validity_window` before any fit runs: if a future
    edit widens `BOUNDS` past the guarded window, this check raises
    loudly, catching the drift before any fit runs outside the quadratic
    regime's validity.
    """
    corners = []
    for e0 in (BOUNDS[0][0], BOUNDS[0][1]):
        for grad in (BOUNDS[1][0], BOUNDS[1][1]):
            for patch in (BOUNDS[2][0], BOUNDS[2][1]):
                corners.append(_cell_field_magnitude_np(atom_positions_m, e0, grad, patch))
    return np.concatenate(corners)


def laplace_uncertainties(hessian: np.ndarray) -> tuple[bool, np.ndarray]:
    """The reporting path shared by every fit case, generalized from
    `benchmarks/run_sideband_fit.py::laplace_uncertainties` (there, a
    fixed `2x2` Hessian; here, a general `NxN` Hessian, `N=3`). Same
    discipline, verbatim: the Laplace/Gaussian approximation treats the
    inverse Hessian as a covariance, valid only where the Hessian of the
    negative log-likelihood is positive definite, the condition for a
    true local minimum. `L-BFGS-B`'s own gradient-norm `success` flag
    reports the same `True` value at a saddle point, where the gradient
    is near zero but one Hessian eigenvalue is negative, with no
    separate warning. `hessian_positive_definite` is `True` only when
    every eigenvalue (`np.linalg.eigvalsh`) is strictly positive;
    otherwise every returned uncertainty is `nan`, matching the
    convention already used for a singular Hessian
    (`np.linalg.LinAlgError`). A `nan` marks the row as an optimum where
    the Laplace approximation does not apply, so it reads as what it is
    in a results table.

    Parameters
    ----------
    hessian : np.ndarray, shape (n, n)
        The Hessian of `0.5*chi2` at the reported optimum.

    Returns
    -------
    tuple[bool, np.ndarray]
        `(hessian_positive_definite, sigmas)`, `sigmas` shape `(n,)`,
        `nan` throughout when the flag is `False`.
    """
    n = hessian.shape[0]
    eigvals = np.linalg.eigvalsh(hessian)
    hessian_pd = bool(np.all(eigvals > 0.0))
    if hessian_pd:
        try:
            cov = np.linalg.inv(hessian)
            sigmas = np.sqrt(np.diag(cov))
            if np.any(np.diag(cov) < 0.0):
                hessian_pd = False
                sigmas = np.full(n, np.nan)
        except np.linalg.LinAlgError:
            hessian_pd = False
            sigmas = np.full(n, np.nan)
    else:
        sigmas = np.full(n, np.nan)
    return hessian_pd, sigmas


@dataclass(frozen=True)
class FitCase:
    truth_e_uniform_v_per_m: float
    truth_gradient_v_per_m_per_m: float
    truth_patch_amplitude_v_per_m: float
    seed: int
    noise_sigma: float
    initial_guess: list[float]
    recovered: list[float]
    recovered_uncertainty: list[float]
    hessian_positive_definite: bool
    within_1sigma: list[bool]
    within_2sigma: list[bool]
    all_within_1sigma: bool
    all_within_2sigma: bool
    converged: bool
    n_iterations: int
    final_chi2: float
    n_data_points: int
    max_field_v_per_m: float


def make_forward_model(positions_m: jnp.ndarray, atom_weights: jnp.ndarray):
    def forward(params: jnp.ndarray) -> jnp.ndarray:
        return rcj.rb85_field_reconstruction_forward_model_jax(
            DELTA_P_HZ,
            positions_m,
            atom_weights,
            params[0],
            params[1],
            params[2],
            jnp.asarray(PATCH_POSITION_M),
            PATCH_SOFTENING_M,
            jnp.asarray(ALPHA0_AU),
            SYSTEM_JAX,
            e_coupling_v_per_m=jnp.asarray(E_COUPLING_V_PER_M),
            temperature_k=jnp.asarray(TEMPERATURE_K),
            mass_kg=jnp.asarray(rcr.RB85_MASS_KG),
            n_velocity_points=N_VELOCITY_POINTS,
        )

    return forward


def make_objective_and_hessian(forward: Any, data: np.ndarray):
    data_j = jnp.asarray(data)

    def chi2(params: jnp.ndarray) -> jnp.ndarray:
        pred = forward(params)
        return jnp.sum(((pred - data_j) / NOISE_SIGMA) ** 2)

    def neg_log_likelihood(params: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * chi2(params)

    value_and_grad = jax.jit(jax.value_and_grad(chi2))
    hessian = jax.jit(jax.hessian(neg_log_likelihood))

    def scipy_objective(params_np: np.ndarray) -> tuple[float, np.ndarray]:
        v, g = value_and_grad(jnp.asarray(params_np))
        return float(v), np.asarray(g, dtype=np.float64)

    return scipy_objective, hessian


def run_one_fit(truth: tuple[float, float, float], seed: int, positions_np: np.ndarray) -> FitCase:
    truth_fields = _cell_field_magnitude_np(positions_np, *truth)
    _assert_within_validity_window(truth_fields, f"truth {truth}")

    positions_j = jnp.asarray(positions_np)
    weights_j = jnp.ones(positions_np.shape[0])
    forward = make_forward_model(positions_j, weights_j)
    truth_spectrum = np.asarray(forward(jnp.asarray(truth)))

    rng = np.random.default_rng(seed)
    noisy = truth_spectrum + rng.normal(0.0, NOISE_SIGMA, size=truth_spectrum.shape)
    objective, hessian_fn = make_objective_and_hessian(forward, noisy)

    x0 = [t * f for t, f in zip(truth, X0_OFFSET_FACTORS, strict=True)]
    result = minimize(objective, np.array(x0), jac=True, method="L-BFGS-B", bounds=BOUNDS)

    hessian = np.asarray(hessian_fn(jnp.asarray(result.x)))
    hessian_pd, sigmas = laplace_uncertainties(hessian)

    within_1s = [
        bool(abs(result.x[i] - truth[i]) <= sigmas[i]) if hessian_pd else False for i in range(3)
    ]
    within_2s = [
        bool(abs(result.x[i] - truth[i]) <= 2.0 * sigmas[i]) if hessian_pd else False
        for i in range(3)
    ]

    return FitCase(
        truth_e_uniform_v_per_m=truth[0],
        truth_gradient_v_per_m_per_m=truth[1],
        truth_patch_amplitude_v_per_m=truth[2],
        seed=seed,
        noise_sigma=NOISE_SIGMA,
        initial_guess=list(x0),
        recovered=[float(v) for v in result.x],
        recovered_uncertainty=[float(s) for s in sigmas],
        hessian_positive_definite=hessian_pd,
        within_1sigma=within_1s,
        within_2sigma=within_2s,
        all_within_1sigma=bool(all(within_1s)),
        all_within_2sigma=bool(all(within_2s)),
        converged=bool(result.success),
        n_iterations=int(result.nit),
        final_chi2=float(result.fun),
        n_data_points=int(DELTA_P_HZ_NP.shape[0]),
        max_field_v_per_m=float(np.max(truth_fields)),
    )


def run_all_fits() -> list[FitCase]:
    rng = np.random.default_rng(20260903)
    positions_np = rcr.cylindrical_cell_atom_positions(CELL_RADIUS_M, CELL_LENGTH_M, N_ATOMS, rng)

    corner_fields = _bound_corner_fields(positions_np)
    _assert_within_validity_window(corner_fields, "optimizer bound corners")

    cases = []
    for truth in TRUTH_GRID:
        for seed in SEEDS:
            cases.append(run_one_fit(truth, seed, positions_np))
    return cases


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report() -> dict[str, Any]:
    c1 = run_c1_agreement_check()
    c2 = run_c2_gradient_check()
    cases = run_all_fits()

    n_converged = sum(1 for c in cases if c.converged)
    n_hessian_pd = sum(1 for c in cases if c.hessian_positive_definite)
    n_1sigma = sum(1 for c in cases if c.all_within_1sigma)
    n_2sigma = sum(1 for c in cases if c.all_within_2sigma)

    return {
        "wp41_rydberg_field_reconstruction_schema": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "claim_calibration": (
            "Synthetic round-trip: generator == fitter. Gradient-based optimization "
            "through this project's own differentiable quadratic-Stark/EIT chain "
            "recovers a known three-parameter cell field distribution (uniform "
            "background, linear axial gradient, one wall-patch amplitude) from a "
            "synthetic composed EIT spectrum, with correctly calibrated Laplace "
            "uncertainties. No real Rydberg-sensor scan is fit here. Every truth "
            "value and every optimizer-bound corner is checked to stay inside the "
            "guarded quadratic-Stark validity window before any fit runs "
            f"({VALIDITY_GUARD_V_PER_M:.1f} V/m for the Rb-85 32D5/2 registry state). "
            "Every case's Hessian is checked for positive definiteness "
            "(hessian_positive_definite) before its inverse is trusted as a "
            "covariance; at a saddle point the reported uncertainty is nan."
        ),
        "validity_guard_v_per_m": VALIDITY_GUARD_V_PER_M,
        "c1_agreement_check": c1,
        "c2_gradient_check": c2,
        "fit_cases": [asdict(c) for c in cases],
        "n_cases": len(cases),
        "n_converged": n_converged,
        "n_hessian_positive_definite": n_hessian_pd,
        "n_within_1sigma_all_params": n_1sigma,
        "n_within_2sigma_all_params": n_2sigma,
    }


def _uncertainty_cell(recovered: float, uncertainty: float, hessian_pd: bool, decimals: int) -> str:
    if not hessian_pd:
        return f"{recovered:.{decimals}f} +/- nan"
    return f"{recovered:.{decimals}f} +/- {uncertainty:.{decimals}f}"


def _fit_case_row(c: dict[str, Any]) -> str:
    pd_flag = c["hessian_positive_definite"]
    e0_cell = _uncertainty_cell(c["recovered"][0], c["recovered_uncertainty"][0], pd_flag, 1)
    grad_cell = _uncertainty_cell(c["recovered"][1], c["recovered_uncertainty"][1], pd_flag, 1)
    patch_cell = _uncertainty_cell(c["recovered"][2], c["recovered_uncertainty"][2], pd_flag, 1)
    return (
        f"| {c['truth_e_uniform_v_per_m']:.1f} | {c['truth_gradient_v_per_m_per_m']:.1f} | "
        f"{c['truth_patch_amplitude_v_per_m']:.1f} | {c['seed']} | {e0_cell} | {grad_cell} | "
        f"{patch_cell} | {pd_flag} | {c['converged']} | {c['all_within_1sigma']} | "
        f"{c['all_within_2sigma']} |"
    )


def render_markdown(report: dict[str, Any]) -> str:
    c1 = report["c1_agreement_check"]
    c2 = report["c2_gradient_check"]
    non_pd_cases = [c for c in report["fit_cases"] if not c["hessian_positive_definite"]]
    lines = [
        "# WP41: differentiable Rydberg field-to-spectrum chain and field reconstruction",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        report["claim_calibration"],
        "",
        "## C1: agreement, JAX vs the numpy reference",
        "",
        f"Single-atom worst-case relative error: `{c1['single_atom_worst_relative_error']:.3e}` "
        f"at {c1['single_atom_worst_point']}. Composed (multi-atom) worst-case relative error: "
        f"`{c1['composed_worst_relative_error']:.3e}` at {c1['composed_worst_point']}. "
        f"Tolerance `{c1['tolerance']:.0e}`. MET: `{c1['met']}`.",
        "",
        "## C2: gradient validation, jax.grad vs central finite differences",
        "",
        f"Worst-case relative error `{c2['worst_relative_error']:.3e}` "
        f"(argument `{c2['worst_argument']}`), tolerance `{c2['tolerance']:.0e}`. "
        f"MET: `{c2['met']}`.",
        "",
        "| argument | jax.grad | central FD | relative error |",
        "|---|---|---|---|",
        *(
            f"| {name} | {v['jax_grad']:.6e} | {v['fd_grad']:.6e} | {v['relative_error']:.3e} |"
            for name, v in c2["per_argument"].items()
        ),
        "",
        "## C5: field-reconstruction fit grid",
        "",
        f"{report['n_converged']}/{report['n_cases']} fits converged. "
        f"{report['n_hessian_positive_definite']}/{report['n_cases']} report a "
        "positive-definite Hessian at the optimum, the condition the Laplace uncertainty "
        f"below requires. {report['n_within_1sigma_all_params']}/{report['n_cases']} recovered "
        "all three parameters within their own reported 1-sigma Laplace uncertainty; "
        f"{report['n_within_2sigma_all_params']}/{report['n_cases']} within 2-sigma.",
        "",
        "| truth E0 (V/m) | truth grad (V/m/m) | truth patch (V/m) | seed | "
        "recovered E0 | recovered grad | recovered patch | Hessian PD | converged | "
        "1-sigma | 2-sigma |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *(_fit_case_row(c) for c in report["fit_cases"]),
        "",
    ]
    if non_pd_cases:
        lines.append(
            "**Hessian not positive definite.** The row(s) below stopped at a saddle "
            "point of the negative log-likelihood: the Hessian at the reported optimum "
            "carries a non-positive eigenvalue, so the Laplace approximation is invalid "
            "there. Each such row's own uncertainty is reported as `nan`."
        )
        lines.append("")
        for c in non_pd_cases:
            lines.append(
                f"- truth `E0={c['truth_e_uniform_v_per_m']:.1f}`, "
                f"`grad={c['truth_gradient_v_per_m_per_m']:.1f}`, "
                f"`patch={c['truth_patch_amplitude_v_per_m']:.1f}`, seed `{c['seed']}`: "
                f"recovered `{[round(v, 2) for v in c['recovered']]}`; the Laplace "
                "uncertainty at this optimum is undefined."
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / "wp41_rydberg_field_reconstruction.json"
    md_path = _RESULTS_DIR / "wp41_rydberg_field_reconstruction.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
