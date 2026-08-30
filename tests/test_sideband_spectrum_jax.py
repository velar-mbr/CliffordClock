# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for WP38's differentiable sideband-spectrum forward model
(`cliffordclock.integrator.sideband_spectrum_jax`).

- **Lineshape limits**: the carrier's zero-pulse-time behavior, and both
  models' `[0, 1]` boundedness.
- **Sideband positions**: the blue/red harmonic-path peaks sit at the
  axial spacing (Blatt et al. 2009 Eq. 4's `nu_z`, corrected by the
  anharmonic terms Eq. 8 states explicitly), and the red sideband
  carries no ground-state (`n_z=0`) contribution (Blatt et al. 2009, text
  following their Eq. 12).
- **Harmonic-vs-BO+WKB direction**: in the regime Goti et al. 2025's own
  Fig. 1 identifies (`k_B*T/D > 0.2`, atoms populating the trap edge),
  the harmonic model's own sideband sits at a HIGHER detuning than the
  BO+WKB model's, Goti et al. 2025's own stated finding ("the harmonic
  oscillator approximation will predict a higher sideband frequency for
  atoms with a given energy").
- **Gradient checks**: `jax.grad` of both paths against central finite
  differences of the SAME function (an internal-consistency check; the
  cross-implementation check lives in `benchmarks/run_sideband_spectrum.py`).
- **Round-trip fit convergence**: one small, fast, deterministic
  synthetic fit (a compact version of `benchmarks/run_sideband_fit.py`'s
  own demonstration).
- **Saddle-point uncertainty reporting**: a planted-violation test of
  `run_sideband_fit.laplace_uncertainties`, the reporting path a G20
  review found silently clamping a saddle point's invalid uncertainty
  to a confident-looking `+/- 0.00`; a fabricated indefinite Hessian,
  fed directly to that function, must flip its `hessian_positive_definite`
  flag to `False` and report `nan`, matching the existing convention for
  a singular Hessian.
- **jit determinism**: repeated calls to the jit-compiled forward
  functions return bitwise-identical output.
- **The offline convergence study**: pins `AXIAL_GRID_N_SPECTRUM`'s
  measured error bound against the G18-gated reference solver
  (mirrors `test_lattice_light_shift_jax.py::TestOfflineConvergenceStudy`'s
  own pattern). This resolution is checked once here, offline; the
  module itself holds it fixed on every call, per its own static-shape
  discipline.

**Timing.** Every BO+WKB (`bowkb_*`) test is `@pytest.mark.slow`: each
call batches a `jax.vmap` over `jax.numpy.linalg.eigh` at
`AXIAL_GRID_N_SPECTRUM` resolution, several bands deep, uncompiled by
default (module-level defaults) at several seconds per call (see
`test_timing_sanity_note` for a measured figure). Tests that override the
resolution down to `benchmarks/run_sideband_fit.py`'s own reduced,
jit-compiled profile run fast and are NOT marked slow; this is noted at
each such test's own docstring.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import eval_genlaguerre  # type: ignore[import-untyped]

import cliffordclock.integrator.lattice_light_shift as lls
import cliffordclock.integrator.sideband_spectrum_jax as ssj
from cliffordclock.constants import SPEED_OF_LIGHT
from cliffordclock.ensemble.species import get_species

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

from run_sideband_fit import laplace_uncertainties  # noqa: E402

WAVELENGTH_M = SPEED_OF_LIGHT / 394_798_266.9e6
WAIST_M = 50e-6
PROBE_WAVELENGTH_M = 578e-9
MASS_KG = get_species("Yb171").mass_kg

#: Reduced BO+WKB resolution shared by every test in this file that does
#: not specifically need the module's own (much slower) default
#: resolution; matches `benchmarks/run_sideband_fit.py`'s own profile.
FAST_AXIAL_GRID_N = 161
FAST_RHO_TABLE_N = 65
FAST_N_Z_MAX = 3
FAST_N_E_QUAD = 32


def _bowkb_kwargs() -> dict[str, int]:
    return {
        "n_z_max": FAST_N_Z_MAX,
        "n_e_quad": FAST_N_E_QUAD,
        "axial_grid_n": FAST_AXIAL_GRID_N,
        "rho_table_n": FAST_RHO_TABLE_N,
    }


# ---------------------------------------------------------------------------
# Laguerre polynomial sanity check
# ---------------------------------------------------------------------------


class TestLaguerreValues:
    """`laguerre_values` against `scipy.special.eval_genlaguerre(n, 0, x)`
    (the ordinary Laguerre polynomial, `alpha=0`), an independent
    special-function implementation.
    """

    @pytest.mark.parametrize("x", [0.0, 0.3, 1.7, 5.0])
    def test_matches_scipy_genlaguerre(self, x: float) -> None:
        n_max = 8
        values = ssj.laguerre_values(jnp.asarray(x), n_max)
        for n in range(n_max + 1):
            expected = eval_genlaguerre(n, 0, x)
            assert float(values[n]) == pytest.approx(expected, rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# Carrier lineshape limits
# ---------------------------------------------------------------------------


class TestCarrierLimits:
    def test_zero_pulse_time_gives_zero_excitation(self) -> None:
        """Blatt et al. 2009 Eq. 17: `p_e(n,delta,t) = ... *
        sin^2[pi*t*sqrt(...)]`, identically zero at `t=0` for every
        `delta`, `n`: no pulse, no excitation.
        """
        delta = jnp.linspace(-50e3, 50e3, 21)
        p_e = ssj.harmonic_carrier_excitation_probability(
            delta, 0.0, 80.0, WAIST_M, WAVELENGTH_M, MASS_KG, PROBE_WAVELENGTH_M, 300.0, 2e-6, 2e-6
        )
        assert np.allclose(np.asarray(p_e), 0.0, atol=1e-12)

    def test_bounded_in_zero_one(self) -> None:
        """A physical probability: never negative, never above 1, over a
        broad detuning sweep and a pi-pulse-scale pulse time.
        """
        delta = jnp.linspace(-100e3, 100e3, 51)
        p_e = np.asarray(
            ssj.harmonic_carrier_excitation_probability(
                delta,
                2e-3,
                80.0,
                WAIST_M,
                WAVELENGTH_M,
                MASS_KG,
                PROBE_WAVELENGTH_M,
                300.0,
                2e-6,
                2e-6,
            )
        )
        assert np.all(p_e >= -1e-10)
        assert np.all(p_e <= 1.0 + 1e-10)

    def test_carrier_peaks_at_zero_detuning(self) -> None:
        """On resonance, the ground-band Rabi frequency dominates and the
        excitation should peak at `delta=0` for a pulse time tuned near
        the ground-state pi-pulse condition.
        """
        nu_z, _, _ = ssj.blatt_trap_frequencies_hz(80.0, WAIST_M, WAVELENGTH_M, MASS_KG)
        eta_z2 = (1.0 / PROBE_WAVELENGTH_M) ** 2 * (6.62607015e-34 / (2.0 * MASS_KG * float(nu_z)))
        omega0 = 300.0 * float(jnp.exp(-0.5 * eta_z2))
        t_pi = 1.0 / (2.0 * omega0)
        delta = jnp.linspace(-2e3, 2e3, 41)
        p_e = np.asarray(
            ssj.harmonic_carrier_excitation_probability(
                delta,
                t_pi,
                80.0,
                WAIST_M,
                WAVELENGTH_M,
                MASS_KG,
                PROBE_WAVELENGTH_M,
                300.0,
                1e-7,
                1e-7,
            )
        )
        assert int(np.argmax(p_e)) == len(delta) // 2 or abs(
            float(delta[int(np.argmax(p_e))])
        ) < float(delta[1] - delta[0])


# ---------------------------------------------------------------------------
# Sideband positions (harmonic path)
# ---------------------------------------------------------------------------


class TestHarmonicSidebandPositions:
    def test_blue_sideband_peaks_near_axial_spacing(self) -> None:
        """Blatt et al. 2009 Eq. 8: `gamma(n_z=0, n_r=0) = nu_z -
        nu_rec`, close to but slightly below `nu_z` (Eq. 4). The blue
        sideband's own peak should sit near this value for a cold trap
        (population concentrated at `n_z=n_r=0`).
        """
        u0 = 100.0
        nu_z, nu_r, nu_rec = ssj.blatt_trap_frequencies_hz(u0, WAIST_M, WAVELENGTH_M, MASS_KG)
        expected = float(nu_z - nu_rec)
        delta = jnp.linspace(0.0, 2.0 * expected, 400)
        shape = np.asarray(
            ssj.harmonic_sideband_shape(
                delta, 1, u0, WAIST_M, WAVELENGTH_M, MASS_KG, 3e-7, 3e-7, 500.0
            )
        )
        peak_delta = float(delta[np.argmax(shape)])
        assert peak_delta == pytest.approx(expected, rel=0.05, abs=1.0)

    def test_red_sideband_is_mirrored_for_a_cold_trap(self) -> None:
        """A cold, deep trap should show near-mirror-symmetric blue/red
        peak positions (both dominated by the `n_z=0<->1` transition).
        """
        u0 = 100.0
        delta_pos = jnp.linspace(0.0, 50e3, 200)
        delta_neg = -delta_pos
        blue = np.asarray(
            ssj.harmonic_sideband_shape(
                delta_pos, 1, u0, WAIST_M, WAVELENGTH_M, MASS_KG, 3e-7, 3e-7, 500.0
            )
        )
        red = np.asarray(
            ssj.harmonic_sideband_shape(
                delta_neg, -1, u0, WAIST_M, WAVELENGTH_M, MASS_KG, 3e-7, 3e-7, 500.0
            )
        )
        blue_peak = float(delta_pos[np.argmax(blue)])
        red_peak = float(delta_neg[np.argmax(red)])
        assert abs(blue_peak + red_peak) < 0.02 * blue_peak

    def test_red_sideband_has_no_ground_state_contribution(self) -> None:
        """Blatt et al. 2009, text following Eq. 12: "There is no
        contribution from the longitudinal ground state to the red
        sideband." Restricting the sum to `n_z_max=0` (only the ground
        band available as a STARTING state) should give an identically
        zero red-sideband shape, since the sole candidate band is
        excluded.
        """
        delta = jnp.linspace(-50e3, 50e3, 21)
        red = ssj.harmonic_sideband_shape(
            delta, -1, 100.0, WAIST_M, WAVELENGTH_M, MASS_KG, 3e-7, 3e-7, 500.0, n_z_max=0
        )
        assert np.allclose(np.asarray(red), 0.0, atol=1e-12)

    def test_invalid_sign_raises(self) -> None:
        with pytest.raises(ValueError, match="sign must be"):
            ssj.harmonic_sideband_shape(
                jnp.array([0.0]), 0, 100.0, WAIST_M, WAVELENGTH_M, MASS_KG, 1e-6, 1e-6, 500.0
            )


# ---------------------------------------------------------------------------
# Harmonic-vs-BO+WKB lineshape difference direction
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHarmonicVsBowkbDirection:
    """Goti et al. 2025's own stated finding: "the harmonic oscillator
    approximation will predict a higher sideband frequency for atoms
    with a given energy" (their Fig. 8 discussion). Checked here at a
    condition inside their own identified `k_B*T/D > 0.2` regime (their
    Fig. 1's own threshold for the harmonic approximation to visibly
    break down): `u0=80 E_R`, `Tr=3 uK` gives `k_B*T/D ~= 0.39` (E_R for
    Yb-171 at the magic wavelength is `~97 nK`, `run_sideband_fit.py`'s
    own constants).
    """

    def test_bowkb_peak_sits_at_or_below_harmonic_peak(self) -> None:
        u0 = 80.0
        tz = tr = 3e-6
        delta = jnp.linspace(0.0, 50e3, 100)
        harmonic = np.asarray(
            ssj.harmonic_sideband_shape(delta, 1, u0, WAIST_M, WAVELENGTH_M, MASS_KG, tz, tr, 2e3)
        )
        bowkb = np.asarray(
            ssj.bowkb_sideband_shape(
                delta, 1, u0, WAIST_M, WAVELENGTH_M, MASS_KG, tz, tr, 2e3, **_bowkb_kwargs()
            )
        )
        harmonic_peak = float(delta[np.argmax(harmonic)])
        bowkb_peak = float(delta[np.argmax(bowkb)])
        assert bowkb_peak <= harmonic_peak + 1e-9

    def test_bowkb_and_harmonic_converge_in_the_cold_deep_limit(self) -> None:
        """At low `k_B*T/D` (deep, cold trap: `u0=150 E_R`, `Tr=200 nK`,
        `k_B*T/D ~= 0.014`), Goti et al. 2025's own Fig. 1 predicts the
        two models should agree closely (population concentrated near
        the well bottom, where the true `cos^2` and harmonic potentials
        agree to high accuracy, this project's own G18-gated
        harmonic-limit consistency check).
        """
        u0 = 150.0
        tz = tr = 2e-7
        delta = jnp.linspace(0.0, 60e3, 120)
        harmonic = np.asarray(
            ssj.harmonic_sideband_shape(delta, 1, u0, WAIST_M, WAVELENGTH_M, MASS_KG, tz, tr, 2e3)
        )
        bowkb = np.asarray(
            ssj.bowkb_sideband_shape(
                delta, 1, u0, WAIST_M, WAVELENGTH_M, MASS_KG, tz, tr, 2e3, **_bowkb_kwargs()
            )
        )
        harmonic_peak = float(delta[np.argmax(harmonic)])
        bowkb_peak = float(delta[np.argmax(bowkb)])
        assert abs(harmonic_peak - bowkb_peak) / harmonic_peak < 0.05


# ---------------------------------------------------------------------------
# Gradient checks vs. finite differences
# ---------------------------------------------------------------------------


class TestHarmonicGradients:
    def test_grad_wrt_u0_and_tr_matches_finite_differences(self) -> None:
        def f(u0: jnp.ndarray, tr: jnp.ndarray) -> jnp.ndarray:
            spec = ssj.harmonic_full_spectrum(
                jnp.linspace(-40e3, 40e3, 21),
                3e-3,
                u0,
                WAIST_M,
                WAVELENGTH_M,
                MASS_KG,
                PROBE_WAVELENGTH_M,
                300.0,
                2e-6,
                tr,
                0.8,
                0.7,
                2e3,
            )
            return jnp.sum(spec)

        grad_fn = jax.grad(f, argnums=(0, 1))
        u0, tr = 90.0, 1.5e-6
        g_u0, g_tr = grad_fn(u0, tr)

        du0, dtr = 1e-3, 1e-9
        fd_u0 = (f(u0 + du0, tr) - f(u0 - du0, tr)) / (2 * du0)
        fd_tr = (f(u0, tr + dtr) - f(u0, tr - dtr)) / (2 * dtr)

        assert float(g_u0) == pytest.approx(float(fd_u0), rel=1e-3, abs=1e-6)
        assert float(g_tr) == pytest.approx(float(fd_tr), rel=1e-3, abs=1e-2)


@pytest.mark.slow
class TestBowkbGradients:
    """BO+WKB gradients through the table-interpolation route
    (`jax.numpy.interp`, no `jax.lax.custom_root`): marked slow because
    each finite-difference comparison needs 2-4 forward evaluations at
    `FAST_*` resolution, uncompiled.
    """

    def test_grad_wrt_u0_matches_finite_differences(self) -> None:
        def f(u0: jnp.ndarray) -> jnp.ndarray:
            spec = ssj.bowkb_sideband_shape(
                jnp.linspace(0.0, 40e3, 21),
                1,
                u0,
                WAIST_M,
                WAVELENGTH_M,
                MASS_KG,
                3e-6,
                3e-6,
                2e3,
                **_bowkb_kwargs(),
            )
            return jnp.sum(spec)

        u0 = 90.0
        g = float(jax.grad(f)(jnp.asarray(u0)))
        du0 = 0.05
        fd = float((f(jnp.asarray(u0 + du0)) - f(jnp.asarray(u0 - du0))) / (2 * du0))
        assert g == pytest.approx(fd, rel=5e-2, abs=1e-3)

    def test_grad_wrt_tr_matches_finite_differences(self) -> None:
        def f(tr: jnp.ndarray) -> jnp.ndarray:
            spec = ssj.bowkb_sideband_shape(
                jnp.linspace(0.0, 40e3, 21),
                1,
                90.0,
                WAIST_M,
                WAVELENGTH_M,
                MASS_KG,
                3e-6,
                tr,
                2e3,
                **_bowkb_kwargs(),
            )
            return jnp.sum(spec)

        tr = 3e-6
        g = float(jax.grad(f)(jnp.asarray(tr)))
        dtr = 1e-8
        fd = float((f(jnp.asarray(tr + dtr)) - f(jnp.asarray(tr - dtr))) / (2 * dtr))
        assert g == pytest.approx(fd, rel=5e-2, abs=1.0)


# ---------------------------------------------------------------------------
# jit determinism
# ---------------------------------------------------------------------------


class TestJitDeterminism:
    def test_harmonic_full_spectrum_jit_is_deterministic(self) -> None:
        jitted = jax.jit(
            lambda u0, tr: ssj.harmonic_full_spectrum(
                jnp.linspace(-40e3, 40e3, 21),
                3e-3,
                u0,
                WAIST_M,
                WAVELENGTH_M,
                MASS_KG,
                PROBE_WAVELENGTH_M,
                300.0,
                2e-6,
                tr,
                0.8,
                0.7,
                2e3,
            )
        )
        first = np.asarray(jitted(90.0, 1.5e-6))
        second = np.asarray(jitted(90.0, 1.5e-6))
        assert np.array_equal(first, second)

    @pytest.mark.slow
    def test_bowkb_sideband_shape_jit_is_deterministic(self) -> None:
        kwargs = _bowkb_kwargs()

        def call(u0: jnp.ndarray, tr: jnp.ndarray) -> jnp.ndarray:
            return ssj.bowkb_sideband_shape(
                jnp.linspace(0.0, 40e3, 21),
                1,
                u0,
                WAIST_M,
                WAVELENGTH_M,
                MASS_KG,
                3e-6,
                tr,
                2e3,
                **kwargs,
            )

        jitted = jax.jit(call)
        first = np.asarray(jitted(90.0, 3e-6))
        second = np.asarray(jitted(90.0, 3e-6))
        assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# Round-trip fit convergence
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRoundTripFitConvergence:
    """A compact, self-contained version of
    `benchmarks/run_sideband_fit.py`'s own synthetic round-trip: fixed
    seed, fixed truth, `FAST_*` BO+WKB resolution.
    """

    def test_bowkb_fit_recovers_u0_and_tr_within_2sigma(self) -> None:
        truth_u0, truth_tr = 100.0, 1.0e-6
        delta = jnp.linspace(-40e3, 40e3, 41)
        kwargs = _bowkb_kwargs()

        def forward(u0: jnp.ndarray, tr: jnp.ndarray) -> jnp.ndarray:
            carrier = ssj.harmonic_carrier_excitation_probability(
                delta, 3e-3, u0, WAIST_M, WAVELENGTH_M, MASS_KG, PROBE_WAVELENGTH_M, 300.0, 2e-6, tr
            )
            blue = ssj.bowkb_sideband_shape(
                delta, 1, u0, WAIST_M, WAVELENGTH_M, MASS_KG, 2e-6, tr, 2e3, **kwargs
            )
            red = ssj.bowkb_sideband_shape(
                delta, -1, u0, WAIST_M, WAVELENGTH_M, MASS_KG, 2e-6, tr, 2e3, **kwargs
            )
            return carrier + 0.9 * blue + 0.8 * red

        truth_spectrum = np.asarray(forward(jnp.asarray(truth_u0), jnp.asarray(truth_tr)))
        rng = np.random.default_rng(0)
        sigma = 0.01
        noisy = truth_spectrum + rng.normal(0.0, sigma, size=truth_spectrum.shape)
        noisy_j = jnp.asarray(noisy)

        def chi2(params: jnp.ndarray) -> jnp.ndarray:
            pred = forward(params[0], params[1])
            return jnp.sum(((pred - noisy_j) / sigma) ** 2)

        value_and_grad = jax.jit(jax.value_and_grad(chi2))

        def objective(params_np: np.ndarray) -> tuple[float, np.ndarray]:
            v, g = value_and_grad(jnp.asarray(params_np))
            return float(v), np.asarray(g, dtype=np.float64)

        x0 = np.array([truth_u0 * 1.2, truth_tr * 0.7])
        result = minimize(
            objective, x0, jac=True, method="L-BFGS-B", bounds=[(10.0, 300.0), (50e-9, 8e-6)]
        )
        hessian = np.asarray(jax.hessian(lambda p: 0.5 * chi2(p))(jnp.asarray(result.x)))
        # Same reporting path `run_sideband_fit.run_one_fit` uses: trust
        # the inverse Hessian as a covariance only where every eigenvalue
        # is strictly positive, reporting `nan` otherwise. This case's
        # own optimum is a true minimum, asserted directly below; a
        # regression that pushed the fit toward a saddle point surfaces
        # here as a failed assertion on `hessian_positive_definite`.
        hessian_pd, sigma_u0, sigma_tr = laplace_uncertainties(hessian)

        assert result.success
        assert hessian_pd, f"Hessian not positive definite: eigvals={np.linalg.eigvalsh(hessian)}"
        assert abs(result.x[0] - truth_u0) <= 3.0 * sigma_u0
        assert abs(result.x[1] - truth_tr) <= 3.0 * sigma_tr


# ---------------------------------------------------------------------------
# Planted-violation test: the saddle-point reporting path itself
# ---------------------------------------------------------------------------


class TestLaplaceUncertaintyReportingPath:
    """Direct tests of `run_sideband_fit.laplace_uncertainties`, the
    reporting path this project's G20 review found silently reporting
    `+/- 0.00` at a saddle point (`benchmarks/run_sideband_fit.py`'s
    harmonic `u0=100`, `seed=0` case). `L-BFGS-B`'s own `success` flag
    checks gradient norm only, so it says nothing about whether the
    Hessian at that point is positive definite, and `sqrt(max(x, 0.0))`
    clamped a negative covariance diagonal to a confident-looking zero.
    Every test here feeds the reporting path a fabricated Hessian
    directly, with no fit, no forward model, and no jax, so the planted
    violation is unambiguous.
    """

    def test_indefinite_hessian_flags_and_reports_nan(self) -> None:
        """The exact indefinite Hessian this project's G20 review found
        at the harmonic `u0=100`, `seed=0` saddle point, eigenvalues
        `[-8.72, 2.636e13]`, one negative.
        """
        hessian = np.array([[81.08, -4.8656e7], [-4.8656e7, 2.6364e13]])
        eigvals = np.linalg.eigvalsh(hessian)
        assert eigvals[0] < 0.0
        assert eigvals[1] > 0.0

        hessian_pd, sigma_u0, sigma_tr = laplace_uncertainties(hessian)

        assert hessian_pd is False
        assert np.isnan(sigma_u0)
        assert np.isnan(sigma_tr)

    def test_positive_definite_hessian_reports_finite_sigmas(self) -> None:
        """A control case: a genuinely positive-definite Hessian reports
        the flag `True` and finite, positive uncertainties. The planted
        violation above checks the failure branch specifically, against
        a function proven here to report a real value on the success
        branch.
        """
        hessian = np.array([[4.0, 0.0], [0.0, 9.0]])

        hessian_pd, sigma_u0, sigma_tr = laplace_uncertainties(hessian)

        assert hessian_pd is True
        assert sigma_u0 == pytest.approx(0.5)
        assert sigma_tr == pytest.approx(1.0 / 3.0)

    def test_singular_hessian_flags_false_and_reports_nan(self) -> None:
        """A singular Hessian, one eigenvalue exactly zero and the other
        positive: the same flag and the same `nan` convention apply,
        matching the existing `np.linalg.LinAlgError` branch this
        function's docstring describes.
        """
        hessian = np.array([[1.0, 0.0], [0.0, 0.0]])

        hessian_pd, sigma_u0, sigma_tr = laplace_uncertainties(hessian)

        assert hessian_pd is False
        assert np.isnan(sigma_u0)
        assert np.isnan(sigma_tr)


# ---------------------------------------------------------------------------
# Offline convergence study: AXIAL_GRID_N_SPECTRUM's own error bound
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestOfflineConvergenceStudy:
    """Pins `AXIAL_GRID_N_SPECTRUM`/`RHO_TABLE_N`'s measured error bound
    against the G18-gated reference finite-difference solver
    (`lattice_light_shift.axial_energies_er`, converged to its own
    `AXIAL_ENERGY_TOL_ER=1e-5`), mirroring
    `test_lattice_light_shift_jax.py::TestOfflineConvergenceStudy`'s own
    pattern. Reports the trend, resolution up -> error down, the same
    trend `benchmarks/run_sideband_spectrum.py`'s tier 1 independently
    confirms against large-lattice-model's own exact Mathieu-function
    values.
    """

    def test_band_bottom_error_below_1e_minus_3_at_default_resolution(self) -> None:
        depth_er = 80.0
        n_states = 4
        reference = lls.axial_energies_er(depth_er, n_states)
        from cliffordclock.integrator.lattice_light_shift_jax import make_site_potential_jax

        site = make_site_potential_jax(depth_er, WAIST_M, WAVELENGTH_M, MASS_KG)
        worst = 0.0
        for nz in range(n_states):
            table = ssj.build_band_energy_table(site, nz)
            pred = float(table.energy_er[0])
            rel_err = abs(pred - float(reference[nz])) / abs(float(reference[nz]))
            worst = max(worst, rel_err)
        assert worst < 1e-3

    def test_error_shrinks_as_rho_table_resolution_increases(self) -> None:
        depth_er = 80.0
        n_z = 3
        reference = float(lls.axial_energies_er(depth_er, n_z + 1)[n_z])
        from cliffordclock.integrator.lattice_light_shift_jax import make_site_potential_jax

        site = make_site_potential_jax(depth_er, WAIST_M, WAVELENGTH_M, MASS_KG)
        errors = []
        for rho_n in (33, 65, 129):
            table = ssj.build_band_energy_table(site, n_z, rho_table_n=rho_n)
            pred = float(table.energy_er[0])
            errors.append(abs(pred - reference) / abs(reference))
        assert errors[0] >= errors[1] >= errors[2]


@pytest.mark.slow
def test_timing_sanity_note(capsys: pytest.CaptureFixture[str]) -> None:
    """Measures one `bowkb_sideband_shape` call's wall time at the
    module's own DEFAULT (not `FAST_*`) resolution and prints it
    directly, giving a change to `AXIAL_GRID_N_SPECTRUM`/`RHO_TABLE_N`
    its own visible signal. The `elapsed < 600.0` ceiling is generous
    (machine-dependent wall time); correctness lives in this file's
    other tests.
    """
    delta = jnp.linspace(0.0, 40e3, 21)
    t0 = time.perf_counter()
    ssj.bowkb_sideband_shape(delta, 1, 90.0, WAIST_M, WAVELENGTH_M, MASS_KG, 2e-6, 2e-6, 2e3)
    elapsed = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[timing] one default-resolution bowkb_sideband_shape call: {elapsed:.1f} s")
    assert elapsed < 600.0
