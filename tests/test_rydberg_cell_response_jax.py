# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for WP41's differentiable JAX Rydberg field-to-spectrum chain
(`cliffordclock.integrator.rydberg_cell_response_jax`, CONVENTIONS.md
section 21, E45): the gate's own correctness contract, checked directly.

- **AGREEMENT (C1)**: `ladder_susceptibility_jax`,
  `doppler_averaged_susceptibility_jax`, and
  `compose_inhomogeneous_eit_spectrum_jax` match the numpy reference
  module's own functions at floating-point-algebra precision (no
  eigensolve, no adaptive grid on either side, unlike the WP37
  lattice-light-shift comparison). A larger, deterministic grid sweep
  and the field-reconstruction fit-grid demonstrator live in
  `benchmarks/run_rydberg_field_reconstruction.py`; this file's own
  checks are a fast-lane regression guard on the same functions.
- **GRADIENTS (C2)**: `jax.grad` of a scalar functional of the spectrum
  matches central finite differences of the NUMPY reference, with a
  dedicated NaN sweep across extreme field/temperature/detuning/Rabi
  inputs (the WP37 clamp-bug lesson, applied proactively here even
  though this module's own docstring argues no clamp/where site exists
  in its physics chain).
- **DETERMINISM (C3)**: the jit-compiled forward call is bitwise
  deterministic across repeated calls in one process and across fresh
  subprocesses.
- **MEMORY (C4)**: peak RSS of the heaviest call this module supports
  (the full field-reconstruction forward model plus its gradient, at the
  benchmark's own production atom count) stays under a measured,
  guarded bound.
- **Field model structure**: `cell_field_magnitude_v_per_m_jax`'s own
  smoothness/positivity/no-NaN-at-the-patch-center claims, checked
  directly.
- **Laplace-uncertainty reporting path**: `laplace_uncertainties` from
  `benchmarks/run_rydberg_field_reconstruction.py`, imported the same
  way `tests/test_sideband_spectrum_jax.py` imports
  `run_sideband_fit.laplace_uncertainties`, with a planted-violation
  test of the indefinite-Hessian branch.
"""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cliffordclock.integrator.rydberg_cell_response as rcr
import cliffordclock.integrator.rydberg_cell_response_jax as rcj

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

from run_rydberg_field_reconstruction import laplace_uncertainties  # noqa: E402

N_STAR_32D52 = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
ALPHA0_AU = rcr.RB85_32D52_ALPHA0_AU


def _default_system() -> rcr.LadderSystem:
    """Matches `tests/test_rydberg_cell_response.py::_default_system`
    exactly: this file is not re-provenancing a new system, only reusing
    the already-checked one.
    """
    return rcr.LadderSystem(
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


def _default_system_jax() -> rcj.LadderSystemJax:
    s = _default_system()
    return rcj.LadderSystemJax(
        mu_probe_c_m=jnp.asarray(s.mu_probe_c_m),
        mu_coupling_c_m=jnp.asarray(s.mu_coupling_c_m),
        mu_rf_c_m=jnp.asarray(s.mu_rf_c_m),
        gamma_12=jnp.asarray(s.gamma_12),
        gamma_13=jnp.asarray(s.gamma_13),
        gamma_14=jnp.asarray(s.gamma_14),
        number_density_m3=jnp.asarray(s.number_density_m3),
        wavelength_probe_m=jnp.asarray(s.wavelength_probe_m),
        wavelength_coupling_m=jnp.asarray(s.wavelength_coupling_m),
    )


DELTA_P_NP = np.linspace(-2.0 * math.pi * 25e6, 2.0 * math.pi * 25e6, 81)
DELTA_P_J = jnp.asarray(DELTA_P_NP)


# ---------------------------------------------------------------------------
# C1: agreement, JAX vs the numpy reference
# ---------------------------------------------------------------------------


class TestAgreement:
    def test_ladder_susceptibility_matches_reference(self) -> None:
        system_np = _default_system()
        system_jax = _default_system_jax()
        ref = rcr.ladder_susceptibility(DELTA_P_NP, 1.0e6, 0.0, 1.0, 400.0, 20.0, system_np)
        got = np.asarray(
            rcj.ladder_susceptibility_jax(
                DELTA_P_J,
                jnp.asarray(1.0e6),
                jnp.asarray(0.0),
                jnp.asarray(1.0),
                jnp.asarray(400.0),
                jnp.asarray(20.0),
                system_jax,
            )
        )
        rel_err = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
        assert rel_err < 1e-10

    @pytest.mark.parametrize(
        ("temperature_k", "e_coupling", "e_rf"),
        [(200.0, 0.0, 0.0), (320.0, 500.0, 0.0), (450.0, 300.0, 40.0)],
    )
    def test_doppler_averaged_susceptibility_matches_reference(
        self, temperature_k: float, e_coupling: float, e_rf: float
    ) -> None:
        system_np = _default_system()
        system_jax = _default_system_jax()
        ref = rcr.doppler_averaged_susceptibility(
            DELTA_P_NP,
            0.0,
            0.0,
            1.0,
            e_coupling,
            e_rf,
            system_np,
            temperature_k,
            rcr.RB85_MASS_KG,
            n_velocity_points=33,
        )
        got = np.asarray(
            rcj.doppler_averaged_susceptibility_jax(
                DELTA_P_J,
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.asarray(1.0),
                jnp.asarray(e_coupling),
                jnp.asarray(e_rf),
                system_jax,
                jnp.asarray(temperature_k),
                jnp.asarray(rcr.RB85_MASS_KG),
                n_velocity_points=33,
            )
        )
        rel_err = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
        assert rel_err < 1e-10

    def test_compose_inhomogeneous_eit_spectrum_matches_reference(self) -> None:
        system_np = _default_system()
        system_jax = _default_system_jax()
        rng = np.random.default_rng(7)
        fields = rng.uniform(20.0, 400.0, 8)
        weights = rng.uniform(0.5, 1.5, 8)
        ref = rcr.compose_inhomogeneous_eit_spectrum(
            DELTA_P_NP,
            fields,
            weights,
            ALPHA0_AU,
            N_STAR_32D52,
            system_np,
            e_coupling_v_per_m=500.0,
            temperature_k=320.0,
            mass_kg=rcr.RB85_MASS_KG,
            n_velocity_points=33,
        )
        got = np.asarray(
            rcj.compose_inhomogeneous_eit_spectrum_jax(
                DELTA_P_J,
                jnp.asarray(fields),
                jnp.asarray(weights),
                jnp.asarray(ALPHA0_AU),
                system_jax,
                e_coupling_v_per_m=jnp.asarray(500.0),
                temperature_k=jnp.asarray(320.0),
                mass_kg=jnp.asarray(rcr.RB85_MASS_KG),
                n_velocity_points=33,
            )
        )
        rel_err = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
        assert rel_err < 1e-10

    def test_zero_field_matches_reference_exactly_in_shape(self) -> None:
        """The reference module's own C5 zero-field kill-test checks
        byte-identical agreement against a direct single-atom evaluation
        (its own general-vs-uniform-path branch, this module's docstring
        explains why this module does not reproduce bit-for-bit). This
        test instead checks the weaker, still meaningful claim this
        module DOES make: at zero field, the JAX composition matches the
        numpy reference's own general path to the same floating-point
        tolerance every other agreement check in this class uses.
        """
        system_np = _default_system()
        system_jax = _default_system_jax()
        fields = np.zeros(5)
        weights = np.ones(5)
        ref = rcr.compose_inhomogeneous_eit_spectrum(
            DELTA_P_NP,
            fields,
            weights,
            ALPHA0_AU,
            N_STAR_32D52,
            system_np,
            e_coupling_v_per_m=500.0,
            temperature_k=320.0,
            mass_kg=rcr.RB85_MASS_KG,
            n_velocity_points=33,
        )
        got = np.asarray(
            rcj.compose_inhomogeneous_eit_spectrum_jax(
                DELTA_P_J,
                jnp.asarray(fields),
                jnp.asarray(weights),
                jnp.asarray(ALPHA0_AU),
                system_jax,
                e_coupling_v_per_m=jnp.asarray(500.0),
                temperature_k=jnp.asarray(320.0),
                mass_kg=jnp.asarray(rcr.RB85_MASS_KG),
                n_velocity_points=33,
            )
        )
        rel_err = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
        assert rel_err < 1e-10


# ---------------------------------------------------------------------------
# C2: gradients vs central finite differences of the numpy reference, and
# the NaN sweep
# ---------------------------------------------------------------------------


def _numpy_loss(field_v_per_m: float, temperature_k: float) -> float:
    system_np = _default_system()
    delta_c = (
        2.0 * math.pi * rcr.rydberg_quadratic_stark_shift_hz(ALPHA0_AU, field_v_per_m, N_STAR_32D52)
    )
    spectrum = rcr.doppler_averaged_susceptibility(
        DELTA_P_NP,
        delta_c,
        0.0,
        1.0,
        500.0,
        0.0,
        system_np,
        temperature_k,
        rcr.RB85_MASS_KG,
        n_velocity_points=33,
    )
    return float(np.sum(spectrum.imag))


def _jax_loss(field_v_per_m: jnp.ndarray, temperature_k: jnp.ndarray) -> jnp.ndarray:
    system_jax = _default_system_jax()
    shift_hz = rcj.rydberg_quadratic_stark_shift_hz_jax(jnp.asarray(ALPHA0_AU), field_v_per_m)
    delta_c = 2.0 * jnp.pi * shift_hz
    spectrum = rcj.doppler_averaged_susceptibility_jax(
        DELTA_P_J,
        delta_c,
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(500.0),
        jnp.asarray(0.0),
        system_jax,
        temperature_k,
        jnp.asarray(rcr.RB85_MASS_KG),
        n_velocity_points=33,
    )
    return jnp.sum(jnp.imag(spectrum))


class TestGradients:
    def test_gradient_matches_central_finite_difference_of_reference(self) -> None:
        field0, temp0 = 220.0, 320.0
        grad_fn = jax.jit(jax.grad(_jax_loss, argnums=(0, 1)))
        g_field, g_temp = grad_fn(jnp.asarray(field0), jnp.asarray(temp0))

        h_field = 1e-6 * field0
        fd_field = (_numpy_loss(field0 + h_field, temp0) - _numpy_loss(field0 - h_field, temp0)) / (
            2.0 * h_field
        )
        h_temp = 1e-6 * temp0
        fd_temp = (_numpy_loss(field0, temp0 + h_temp) - _numpy_loss(field0, temp0 - h_temp)) / (
            2.0 * h_temp
        )

        assert abs(float(g_field) - fd_field) / abs(fd_field) < 1e-5
        assert abs(float(g_temp) - fd_temp) / abs(fd_temp) < 1e-5


class TestNaNSweep:
    """Direct evaluation of `jax.grad` at extreme inputs. This module's
    own docstring argues no `jnp.where`/clip site exists in the physics
    chain to create a flat region or a divide-by-zero subgradient; this
    class checks that argument empirically, with a direct sweep.
    """

    @pytest.mark.parametrize("field_v_per_m", [0.0, 1e-6, 1.0, 2000.0, -500.0])
    def test_stark_shift_gradient_finite_at_extreme_fields(self, field_v_per_m: float) -> None:
        g = jax.grad(rcj.rydberg_quadratic_stark_shift_hz_jax, argnums=1)(
            jnp.asarray(ALPHA0_AU), jnp.asarray(field_v_per_m)
        )
        assert math.isfinite(float(g))

    @pytest.mark.parametrize(
        ("temperature_k", "e_coupling", "e_rf", "delta_rf"),
        [
            (1e-3, 0.0, 0.0, 0.0),
            (1e6, 0.0, 0.0, 0.0),
            (320.0, 0.0, 0.0, 0.0),
            (320.0, 1e6, 0.0, 0.0),
            (320.0, 0.0, 1e6, 0.0),
            (320.0, 500.0, 30.0, 1e9),
            (320.0, 500.0, 30.0, -1e9),
        ],
    )
    def test_doppler_averaged_gradient_finite_at_extreme_drives(
        self, temperature_k: float, e_coupling: float, e_rf: float, delta_rf: float
    ) -> None:
        system_jax = _default_system_jax()

        def loss(
            temp: jnp.ndarray, ec: jnp.ndarray, erf: jnp.ndarray, drf: jnp.ndarray
        ) -> jnp.ndarray:
            spectrum = rcj.doppler_averaged_susceptibility_jax(
                DELTA_P_J,
                jnp.asarray(0.0),
                drf,
                jnp.asarray(1.0),
                ec,
                erf,
                system_jax,
                temp,
                jnp.asarray(rcr.RB85_MASS_KG),
                n_velocity_points=33,
            )
            return jnp.sum(jnp.imag(spectrum)) + jnp.sum(jnp.real(spectrum))

        grads = jax.grad(loss, argnums=(0, 1, 2, 3))(
            jnp.asarray(temperature_k),
            jnp.asarray(e_coupling),
            jnp.asarray(e_rf),
            jnp.asarray(delta_rf),
        )
        for g in grads:
            assert math.isfinite(float(g)), (
                f"non-finite grad at T={temperature_k}, ec={e_coupling}, erf={e_rf}, drf={delta_rf}"
            )

    @pytest.mark.parametrize("delta_p_edge", [-2.0 * math.pi * 1e9, 2.0 * math.pi * 1e9, 0.0])
    def test_gradient_finite_at_detuning_window_edges(self, delta_p_edge: float) -> None:
        system_jax = _default_system_jax()

        def loss(field: jnp.ndarray) -> jnp.ndarray:
            shift_hz = rcj.rydberg_quadratic_stark_shift_hz_jax(jnp.asarray(ALPHA0_AU), field)
            delta_c = 2.0 * jnp.pi * shift_hz
            spectrum = rcj.doppler_averaged_susceptibility_jax(
                jnp.asarray([delta_p_edge]),
                delta_c,
                jnp.asarray(0.0),
                jnp.asarray(1.0),
                jnp.asarray(500.0),
                jnp.asarray(0.0),
                system_jax,
                jnp.asarray(320.0),
                jnp.asarray(rcr.RB85_MASS_KG),
                n_velocity_points=33,
            )
            return jnp.sum(jnp.imag(spectrum))

        g = jax.grad(loss)(jnp.asarray(220.0))
        assert math.isfinite(float(g))

    def test_field_model_gradient_finite_at_the_patch_center(self) -> None:
        """The one site this module's docstring names as a potential
        singularity (the softened patch term): its gradient must stay
        finite EXACTLY at the patch location, `r=0`, not just nearby.
        """
        patch_position = jnp.array([0.0125, 0.0, 0.0])
        position_at_patch = jnp.array([[0.0125, 0.0, 0.0]])

        def loss(patch_amplitude: jnp.ndarray) -> jnp.ndarray:
            field = rcj.cell_field_magnitude_v_per_m_jax(
                position_at_patch,
                jnp.asarray(200.0),
                jnp.asarray(0.0),
                patch_amplitude,
                patch_position,
                0.005,
            )
            return jnp.sum(field)

        value, g = jax.value_and_grad(loss)(jnp.asarray(90.0))
        assert math.isfinite(float(value))
        assert math.isfinite(float(g))
        # At r=0 the patch term is exactly patch_amplitude (softening_sq /
        # (0 + softening_sq) == 1), so d(field)/d(patch_amplitude) == 1.
        assert float(g) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# C3: jit determinism, in-process and cross-process
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_doppler_averaged_susceptibility_jit_is_deterministic(self) -> None:
        system_jax = _default_system_jax()
        jitted = jax.jit(
            lambda field, temp: rcj.doppler_averaged_susceptibility_jax(
                DELTA_P_J,
                2.0
                * jnp.pi
                * rcj.rydberg_quadratic_stark_shift_hz_jax(jnp.asarray(ALPHA0_AU), field),
                jnp.asarray(0.0),
                jnp.asarray(1.0),
                jnp.asarray(500.0),
                jnp.asarray(0.0),
                system_jax,
                temp,
                jnp.asarray(rcr.RB85_MASS_KG),
                n_velocity_points=33,
            )
        )
        first = np.asarray(jitted(jnp.asarray(220.0), jnp.asarray(320.0)))
        second = np.asarray(jitted(jnp.asarray(220.0), jnp.asarray(320.0)))
        assert np.array_equal(first, second)

    def test_forward_model_bitwise_deterministic_across_fresh_processes(self) -> None:
        """The same call, run in two SEPARATE fresh Python processes
        (not just twice within one process's compilation cache), prints
        an identical full-precision `repr` of its output. This checks a
        stronger claim than in-process determinism: no dependence on
        anything specific to one process's compilation cache or
        allocator state.
        """
        script = textwrap.dedent(
            """
            import numpy as np
            import jax.numpy as jnp
            import cliffordclock.integrator.rydberg_cell_response_jax as rcj

            system = rcj.LadderSystemJax(
                mu_probe_c_m=jnp.asarray(2.0e-29), mu_coupling_c_m=jnp.asarray(5.0e-30),
                mu_rf_c_m=jnp.asarray(5.2635e-27), gamma_12=jnp.asarray(3.7699e7),
                gamma_13=jnp.asarray(1.8850e6), gamma_14=jnp.asarray(1.8850e6),
                number_density_m3=jnp.asarray(1.0e16), wavelength_probe_m=jnp.asarray(780.24e-9),
                wavelength_coupling_m=jnp.asarray(481.75e-9),
            )
            delta_p = jnp.linspace(-1.5e8, 1.5e8, 41)
            out = rcj.doppler_averaged_susceptibility_jax(
                delta_p, jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(500.0),
                jnp.asarray(0.0), system, jnp.asarray(320.0), jnp.asarray(1.409993199e-25),
                n_velocity_points=33,
            )
            print(repr(np.asarray(out).tobytes()))
            """
        )
        results = []
        for _ in range(2):
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
                cwd=_REPO_ROOT,
            )
            assert proc.returncode == 0, f"stderr={proc.stderr!r}"
            results.append(proc.stdout.strip())
        assert results[0] == results[1]


# ---------------------------------------------------------------------------
# C4: memory bound
# ---------------------------------------------------------------------------


class TestMemoryBound:
    """Peak RSS of one production-scale `value_and_grad` call through the
    full field-reconstruction forward model
    (`rb85_field_reconstruction_forward_model_jax`, at
    `benchmarks/run_rydberg_field_reconstruction.py`'s own `n_atoms=400`
    production scale), measured the same way
    `tests/test_lattice_light_shift_jax.py::TestMemoryBound` measures it:
    a fresh CHILD subprocess reads its own
    `resource.getrusage(RUSAGE_SELF).ru_maxrss`.
    """

    def test_production_call_stays_under_the_memory_bound(self) -> None:
        script = textwrap.dedent(
            """
            import resource
            import sys
            import numpy as np
            import jax
            import jax.numpy as jnp
            import cliffordclock.integrator.rydberg_cell_response as rcr
            import cliffordclock.integrator.rydberg_cell_response_jax as rcj

            n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
            alpha0 = rcr.RB85_32D52_ALPHA0_AU
            system = rcj.LadderSystemJax(
                mu_probe_c_m=jnp.asarray(2.0e-29), mu_coupling_c_m=jnp.asarray(5.0e-30),
                mu_rf_c_m=jnp.asarray(rcr.RB85_MU_RF_32D52_33P32_C_M),
                gamma_12=jnp.asarray(3.7699e7), gamma_13=jnp.asarray(1.8850e6),
                gamma_14=jnp.asarray(1.8850e6), number_density_m3=jnp.asarray(1.0e16),
                wavelength_probe_m=jnp.asarray(rcr.HOLLOWAY_LAMBDA_PROBE_M),
                wavelength_coupling_m=jnp.asarray(rcr.HOLLOWAY_LAMBDA_COUPLING_M),
            )
            rng = np.random.default_rng(0)
            positions = rcr.cylindrical_cell_atom_positions(0.0125, 0.078, 400, rng)
            positions_j = jnp.asarray(positions)
            weights_j = jnp.ones(400)
            patch_position = jnp.array([0.0125, 0.0, 0.0])
            delta_p = jnp.linspace(-1.5708e8, 1.5708e8, 161)

            def forward(params):
                return rcj.rb85_field_reconstruction_forward_model_jax(
                    delta_p, positions_j, weights_j, params[0], params[1], params[2],
                    patch_position, 0.005, jnp.asarray(alpha0), system,
                    e_coupling_v_per_m=jnp.asarray(500.0), temperature_k=jnp.asarray(320.0),
                    mass_kg=jnp.asarray(rcr.RB85_MASS_KG), n_velocity_points=33,
                )

            def loss(params):
                return jnp.sum(forward(params) ** 2)

            value_and_grad_fn = jax.jit(jax.value_and_grad(loss))
            value, grad = value_and_grad_fn(jnp.asarray([220.0, -1200.0, 90.0]))
            print("LOSS", repr(float(value)))
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print("PEAK_RSS_BYTES", peak * (1 if sys.platform == "darwin" else 1024))
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
            cwd=_REPO_ROOT,
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"

        lines = {
            line.split(" ", 1)[0]: line.split(" ", 1)[1]
            for line in proc.stdout.splitlines()
            if " " in line
        }
        assert math.isfinite(float(lines["LOSS"]))

        child_rss_gb = float(lines["PEAK_RSS_BYTES"]) / 1e9
        # This module's own physics chain has no eigensolve and no
        # per-atom Python loop (broadcast composition, module docstring):
        # measured ~0.3 GB on the development macOS machine for this
        # exact call, comfortably under a bound set with large margin
        # above that measurement (unlike the lattice-light-shift module's
        # own multi-GB eigensolve workload, there is no reason to expect
        # this module's memory profile to differ meaningfully by
        # platform, but the same linux/mac split is kept for consistency
        # with the rest of this package's memory-bound tests).
        rss_bound_gb = 2.0 if sys.platform.startswith("linux") else 1.5
        assert child_rss_gb < rss_bound_gb, (
            f"production value_and_grad call used {child_rss_gb:.2f} GB (RSS), "
            f"expected < {rss_bound_gb} GB on this platform"
        )


# ---------------------------------------------------------------------------
# Field model structure
# ---------------------------------------------------------------------------


class TestCellFieldModel:
    def test_uniform_only_reduces_to_the_constant(self) -> None:
        positions = jnp.array([[0.0, 0.0, 0.0], [0.001, 0.002, 0.01], [-0.005, 0.0, -0.02]])
        field = rcj.cell_field_magnitude_v_per_m_jax(
            positions,
            jnp.asarray(200.0),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.array([0.0125, 0.0, 0.0]),
            0.005,
        )
        assert np.allclose(np.asarray(field), 200.0)

    def test_patch_peaks_exactly_at_amplitude_at_the_patch_location(self) -> None:
        patch_position = jnp.array([0.0125, 0.0, 0.0])
        positions = jnp.array([[0.0125, 0.0, 0.0]])
        field = rcj.cell_field_magnitude_v_per_m_jax(
            positions,
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(75.0),
            patch_position,
            0.005,
        )
        assert float(field[0]) == pytest.approx(75.0)

    def test_patch_decays_with_distance(self) -> None:
        patch_position = jnp.array([0.0125, 0.0, 0.0])
        positions = jnp.array(
            [[0.0125, 0.0, 0.0], [0.0125, 0.0, 0.01], [0.0125, 0.0, 0.03], [0.0125, 0.0, 0.05]]
        )
        field = np.asarray(
            rcj.cell_field_magnitude_v_per_m_jax(
                positions,
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.asarray(100.0),
                patch_position,
                0.005,
            )
        )
        assert np.all(np.diff(field) < 0.0)
        assert field[-1] < 1.0


# ---------------------------------------------------------------------------
# Laplace-uncertainty reporting path: planted-violation tests
# ---------------------------------------------------------------------------


class TestLaplaceUncertaintyReportingPath:
    """Direct tests of `run_rydberg_field_reconstruction.laplace_uncertainties`,
    the 3-parameter generalization of `run_sideband_fit.py`'s own
    reporting path (`tests/test_sideband_spectrum_jax.py`'s own
    `TestLaplaceUncertaintyReportingPath` established this
    planted-violation pattern). Every test here feeds the reporting path
    a fabricated Hessian directly, no fit and no jax involved, so the
    planted violation is unambiguous.
    """

    def test_indefinite_hessian_flags_and_reports_nan(self) -> None:
        hessian = np.array([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 3.0]])
        eigvals = np.linalg.eigvalsh(hessian)
        assert np.any(eigvals < 0.0)

        hessian_pd, sigmas = laplace_uncertainties(hessian)

        assert hessian_pd is False
        assert np.all(np.isnan(sigmas))

    def test_positive_definite_hessian_reports_finite_sigmas(self) -> None:
        hessian = np.diag([4.0, 9.0, 16.0])

        hessian_pd, sigmas = laplace_uncertainties(hessian)

        assert hessian_pd is True
        assert sigmas == pytest.approx([0.5, 1.0 / 3.0, 0.25])

    def test_singular_hessian_flags_false_and_reports_nan(self) -> None:
        hessian = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])

        hessian_pd, sigmas = laplace_uncertainties(hessian)

        assert hessian_pd is False
        assert np.all(np.isnan(sigmas))
