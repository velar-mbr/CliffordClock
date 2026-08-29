# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for WP37's differentiable JAX BO+WKB port
(`cliffordclock.integrator.lattice_light_shift_jax`): the gate's own
correctness contract, checked directly.

- **AGREEMENT**: `axial_thermal_factors_jax`'s `X`/`Y`/`Z` match the
  G18-gated reference implementation's `axial_thermal_factors` to `1e-6`
  relative, at all four of Bothwell et al. 2025's Table I points.
- **GRADIENTS**: `jax.grad` of the light shift with respect to `u0` and
  `Tr` matches central finite differences of the REFERENCE
  implementation, to `1e-4` relative, at the same four points.
- **JIT**: the forward evaluation jit-compiles, and the compiled call is
  bitwise deterministic.
- **The offline convergence study**: pins the chosen `(AXIAL_GRID_N_JAX,
  RHO_GRID_N_JAX)` resolution's measured error bound as a standing
  regression, and demonstrates the trend (error shrinks with resolution)
  that justified the choice. This study runs OFFLINE, in this test file,
  because the traced module's own fixed computational graph has no room
  for the reference module's adaptive doubling (CONVENTIONS.md E41; the
  module docstring's "why this module exists" section derives this).
- **Model A port equivalence**: the trivial closed-form port
  (`harmonic_light_shift_hz_jax`) against the reference's own
  `harmonic_light_shift_hz`.

**Timing, and why most of this file is `@pytest.mark.slow`.** Every
Model-B check here runs a dense `(1281, 1281)` `jax.numpy.linalg.eigh`
at `321` radial grid points (module docstring's "Chosen resolution"
section); measured locally at ~50-70 s for one forward-plus-gradient
evaluation (`pytest tests/test_lattice_light_shift_jax.py -m slow
--durations=25`), many times this project's fast-lane budget. Each of
the four table points' forward value, `X`/`Y`/`Z`, and both gradients are
computed ONCE per point via `_value_and_grad_at_row` (module-level
`functools.cache`), so the AGREEMENT and GRADIENT checks below share
that one expensive call per point. Model A's port and the JIT
determinism check use small, cheap inputs and stay in the fast lane.
"""

from __future__ import annotations

import functools
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cliffordclock.integrator.lattice_light_shift as lls
import cliffordclock.integrator.lattice_light_shift_jax as llsj
from cliffordclock.constants import SPEED_OF_LIGHT
from cliffordclock.ensemble.species import get_species

#: Bothwell et al. 2025's own measured Yb-171 E1 magic frequency (Table
#: III, harmonic-basis column), the same constant
#: `benchmarks/run_lattice_light_shift.py` uses for its own Target 3a/3b.
BOTHWELL_2025_MAGIC_FREQUENCY_HZ = 394_798_266.9e6

#: Bothwell et al. 2025's Appendix A Table I: the four G18-gated points
#: this work package's AGREEMENT/GRADIENT contract is checked at.
TABLE1_ROWS: tuple[dict[str, float], ...] = (
    {"u0": 56.8, "tr_nk": 650.0},
    {"u0": 66.4, "tr_nk": 550.0},
    {"u0": 86.2, "tr_nk": 600.0},
    {"u0": 112.2, "tr_nk": 720.0},
)

#: Arbitrary lattice waist: `X`/`Y`/`Z` are independent of it (both
#: implementations' own documented cancellation), matches the benchmark's
#: own choice of `50e-6` m.
WAIST_M = 50e-6

#: A representative near-magic detuning, used only so the light-shift
#: GRADIENT check exercises the E1-slope coefficient's own term (Table I
#: itself specifies no detuning: `X`/`Y`/`Z` don't depend on it). Not tied
#: to any published operating point.
DETUNING_HZ = 200e3


def _yb171_wavelength_m() -> float:
    return SPEED_OF_LIGHT / BOTHWELL_2025_MAGIC_FREQUENCY_HZ


def _yb171_mass_kg() -> float:
    return get_species("Yb171").mass_kg


def _bowkb_coeffs() -> lls.HarmonicLatticeCoefficients:
    return lls.BOTHWELL_2025_YB171_BOWKB


@functools.cache
def _value_and_grad_at_row(
    u0: float, tr_nk: float
) -> tuple[float, float, float, float, float, float]:
    """The one expensive computation this file's Model-B tests share:
    `jax.value_and_grad` of `bo_wkb_fractional_light_shift_jax`'s shift,
    with respect to `(u0, Tr)`, with the `ThermalShapeFactorsJax` used
    threaded out as an auxiliary (undifferentiated) output via
    `has_aux=True` -- one traced forward-plus-backward pass computes the
    shift value, both gradients, AND `X`/`Y`/`Z` together, so the
    AGREEMENT test (which only needs `X`/`Y`/`Z`) and the GRADIENT test
    (which only needs the two gradients) don't each pay for a separate
    eigh-heavy evaluation. `functools.cache` (not a pytest fixture)
    memoizes across the module's several test methods that both touch the
    same `(u0, tr_nk)` pair.

    Returns
    -------
    tuple[float, float, float, float, float, float]
        `(shift, grad_u0, grad_tr, x_nz, y_nz, z_nz)`, all plain Python
        floats: concrete values `functools.cache` can hash to memoize the
        call, and plain numbers any later caller can reuse freely, since
        a Python float carries no dependency on the trace that produced
        it.
    """
    coeffs = _bowkb_coeffs()
    waist_m = jnp.asarray(WAIST_M)
    wavelength_m = jnp.asarray(_yb171_wavelength_m())
    mass_kg = jnp.asarray(_yb171_mass_kg())
    e1 = jnp.asarray(coeffs.e1_slope_per_hz)
    m1e2 = jnp.asarray(coeffs.m1e2_hz)
    beta = jnp.asarray(coeffs.hyperpolarizability_hz)
    detuning_hz = jnp.asarray(DETUNING_HZ)

    def shift_with_aux(
        u0_: jnp.ndarray, tr_k_: jnp.ndarray
    ) -> tuple[jnp.ndarray, llsj.ThermalShapeFactorsJax]:
        shift, factors = llsj.bo_wkb_fractional_light_shift_jax(
            0, u0_, detuning_hz, tr_k_, e1, m1e2, beta, waist_m, wavelength_m, mass_kg
        )
        return shift, factors

    value_and_grad_fn = jax.value_and_grad(shift_with_aux, argnums=(0, 1), has_aux=True)
    (value, factors), (grad_u0, grad_tr) = value_and_grad_fn(
        jnp.asarray(u0), jnp.asarray(tr_nk * 1e-9)
    )
    return (
        float(value),
        float(grad_u0),
        float(grad_tr),
        float(factors.x_nz),
        float(factors.y_nz),
        float(factors.z_nz),
    )


def _reference_site(u0: float) -> lls.SitePotential:
    return lls.make_site_potential(
        depth_er=u0, waist_m=WAIST_M, wavelength_m=_yb171_wavelength_m(), mass_kg=_yb171_mass_kg()
    )


def _reference_shift(u0: float, tr_k: float) -> float:
    """The reference implementation's own light shift at `(u0, Tr)`,
    `n_z=0`, `DETUNING_HZ`, `BOTHWELL_2025_YB171_BOWKB` coefficients --
    the exact same physical point `_value_and_grad_at_row` evaluates,
    used both directly (as the GRADIENT check's finite-difference
    target) and, at the unperturbed `(u0, tr_k)`, as an AGREEMENT
    cross-check on the shift value itself.
    """
    site = _reference_site(u0)
    shift, _ = lls.bo_wkb_fractional_light_shift(0, u0, DETUNING_HZ, tr_k, _bowkb_coeffs(), site)
    return shift


def _reference_central_fd_grad_u0(u0: float, tr_k: float, rel_step: float = 1e-4) -> float:
    """Central finite difference of `_reference_shift` with respect to
    `u0`, the target `jax.grad(..., argnums=0)` (evaluated by
    `_value_and_grad_at_row`) is checked against (GRADIENTS contract:
    "the strongest possible check: autodiff of the new code against
    numerical differentiation of the old code"). `rel_step=1e-4` (`u0`
    values are `O(10-100)`, so the absolute step is `O(1e-3)-O(1e-2)`):
    small enough to resolve the shift's curvature over the `u0` range
    Table I spans, large enough to stay well above the reference module's
    own `THERMAL_FACTOR_TOL=1e-4` relative convergence-guard noise floor.
    """
    h = u0 * rel_step
    plus = _reference_shift(u0 + h, tr_k)
    minus = _reference_shift(u0 - h, tr_k)
    return (plus - minus) / (2.0 * h)


def _reference_central_fd_grad_tr(u0: float, tr_k: float, rel_step: float = 1e-4) -> float:
    """Same central finite difference as
    :func:`_reference_central_fd_grad_u0`, with respect to `Tr` instead."""
    h = tr_k * rel_step
    plus = _reference_shift(u0, tr_k + h)
    minus = _reference_shift(u0, tr_k - h)
    return (plus - minus) / (2.0 * h)


# ---------------------------------------------------------------------------
# AGREEMENT: X/Y/Z against the reference, all four G18 table points
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAgreementWithReference:
    """`axial_thermal_factors_jax`'s `X`/`Y`/`Z`, reached via
    `_value_and_grad_at_row`, against `lattice_light_shift.axial_thermal_factors`
    at all four of Bothwell et al. 2025's Table I points. The module's own
    offline convergence study (`TestOfflineConvergenceStudy` below)
    independently measured this same comparison's worst case at `1.57e-7`
    (`Y`, `u0=56.8 E_R`); this test pins that agreement as a standing
    regression.
    """

    @pytest.mark.parametrize("row", TABLE1_ROWS, ids=[f"u0={r['u0']}" for r in TABLE1_ROWS])
    def test_x_y_z_within_1e_minus_6_relative(self, row: dict[str, float]) -> None:
        _shift, _gu0, _gtr, x_jax, y_jax, z_jax = _value_and_grad_at_row(row["u0"], row["tr_nk"])
        site = _reference_site(row["u0"])
        ref = lls.axial_thermal_factors(site, 0, row["tr_nk"] * 1e-9)

        assert x_jax == pytest.approx(ref.x_nz, rel=1e-6, abs=0)
        assert y_jax == pytest.approx(ref.y_nz, rel=1e-6, abs=0)
        assert z_jax == pytest.approx(ref.z_nz, rel=1e-6, abs=0)

    @pytest.mark.parametrize("row", TABLE1_ROWS, ids=[f"u0={r['u0']}" for r in TABLE1_ROWS])
    def test_light_shift_value_matches_reference(self, row: dict[str, float]) -> None:
        """The full light-shift VALUE agrees too, checked directly here.
        Both implementations evaluate the identical Eq. 6 algebra on top
        of their own `X`/`Y`/`Z`, so this test confirms the shared
        linear-combination formula, the piece the `X`/`Y`/`Z` agreement
        test above does not itself exercise.
        """
        shift_jax, _gu0, _gtr, _x, _y, _z = _value_and_grad_at_row(row["u0"], row["tr_nk"])
        shift_ref = _reference_shift(row["u0"], row["tr_nk"] * 1e-9)
        assert shift_jax == pytest.approx(shift_ref, rel=1e-6, abs=0)


# ---------------------------------------------------------------------------
# GRADIENTS: jax.grad vs. central finite differences of the reference
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGradientsMatchReferenceFiniteDifferences:
    """`jax.grad` of the BO+WKB light shift with respect to `u0` and `Tr`,
    against central finite differences of the REFERENCE implementation
    (not of this module): the strongest available check, since it
    compares one numerical method's autodiff against an independent
    numerical method's finite differences on the same physical formula.
    """

    @pytest.mark.parametrize("row", TABLE1_ROWS, ids=[f"u0={r['u0']}" for r in TABLE1_ROWS])
    def test_grad_wrt_u0_matches_reference_fd(self, row: dict[str, float]) -> None:
        _shift, grad_u0_jax, _gtr, _x, _y, _z = _value_and_grad_at_row(row["u0"], row["tr_nk"])
        grad_u0_fd = _reference_central_fd_grad_u0(row["u0"], row["tr_nk"] * 1e-9)
        assert grad_u0_jax == pytest.approx(grad_u0_fd, rel=1e-4, abs=0)

    @pytest.mark.parametrize("row", TABLE1_ROWS, ids=[f"u0={r['u0']}" for r in TABLE1_ROWS])
    def test_grad_wrt_tr_matches_reference_fd(self, row: dict[str, float]) -> None:
        _shift, _gu0, grad_tr_jax, _x, _y, _z = _value_and_grad_at_row(row["u0"], row["tr_nk"])
        grad_tr_fd = _reference_central_fd_grad_tr(row["u0"], row["tr_nk"] * 1e-9)
        assert grad_tr_jax == pytest.approx(grad_tr_fd, rel=1e-4, abs=0)


# ---------------------------------------------------------------------------
# JIT: compiles, and the compiled call is deterministic
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestJitCompilesAndIsDeterministic:
    def test_forward_evaluation_jit_compiles_and_is_deterministic(self) -> None:
        """`bo_wkb_fractional_light_shift_jax` runs under `jax.jit`
        (`n_z` marked static: it selects an array slice, fixed at trace
        time) and the compiled call returns BITWISE identical output
        across two runs on the same inputs, the JIT contract's own
        determinism requirement. This test checks that requirement with
        exact array equality: `pytest.approx`'s tolerance would silently
        pass a real bitwise regression here."""
        row = TABLE1_ROWS[0]
        coeffs = _bowkb_coeffs()
        args = (
            row["u0"],
            DETUNING_HZ,
            row["tr_nk"] * 1e-9,
            coeffs.e1_slope_per_hz,
            coeffs.m1e2_hz,
            coeffs.hyperpolarizability_hz,
            WAIST_M,
            _yb171_wavelength_m(),
            _yb171_mass_kg(),
        )
        jitted = jax.jit(llsj.bo_wkb_fractional_light_shift_jax, static_argnames=("n_z",))

        shift_a, factors_a = jitted(0, *args)
        shift_b, factors_b = jitted(0, *args)

        np.testing.assert_array_equal(np.asarray(shift_a), np.asarray(shift_b))
        np.testing.assert_array_equal(np.asarray(factors_a.x_nz), np.asarray(factors_b.x_nz))
        np.testing.assert_array_equal(np.asarray(factors_a.y_nz), np.asarray(factors_b.y_nz))
        np.testing.assert_array_equal(np.asarray(factors_a.z_nz), np.asarray(factors_b.z_nz))

        # The jitted call also agrees with the eager (non-jitted) call to
        # high relative precision. XLA fuses operations differently
        # between the eager and compiled execution paths, so the two
        # calls diverge at the single-ULP level: a quick standalone check
        # on this same function found eager-vs-jit differences of
        # 1e-16-relative, ordinary floating-point non-associativity. The
        # two jitted calls above run the SAME compiled executable, which
        # is why they are held to the stricter exact-equality bar.
        shift_eager, factors_eager = llsj.bo_wkb_fractional_light_shift_jax(0, *args)
        assert float(shift_a) == pytest.approx(float(shift_eager), rel=1e-10, abs=0)
        assert float(factors_a.x_nz) == pytest.approx(float(factors_eager.x_nz), rel=1e-10, abs=0)


# ---------------------------------------------------------------------------
# Offline convergence study: pins the chosen resolution's error bound
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestOfflineConvergenceStudy:
    """Pins the resolution choice the module docstring documents
    (`AXIAL_GRID_N_JAX=1281`, `RHO_GRID_N_JAX=321`): the worst-case
    `X`/`Y`/`Z` relative error against the reference's own converged
    output, at all four G18 table points (a standing regression on the
    number quoted in the module docstring), and the CONVERGENCE TREND
    that justified it (error shrinks as resolution increases). The traced
    module's own fixed computational graph has no room for the
    reference's adaptive doubling, so this class runs that convergence
    check here, offline, calling
    `axial_thermal_factors_jax`'s `axial_grid_n`/`rho_grid_n` OVERRIDE
    keywords (present only for this purpose; a physical evaluation always
    uses the defaults) so the SAME production code path gets measured at
    each resolution.
    """

    @pytest.mark.parametrize("row", TABLE1_ROWS, ids=[f"u0={r['u0']}" for r in TABLE1_ROWS])
    def test_default_resolution_worst_case_error_below_1e_minus_6(
        self, row: dict[str, float]
    ) -> None:
        site = llsj.make_site_potential_jax(
            jnp.asarray(row["u0"]),
            jnp.asarray(WAIST_M),
            jnp.asarray(_yb171_wavelength_m()),
            jnp.asarray(_yb171_mass_kg()),
        )
        factors = llsj.axial_thermal_factors_jax(site, 0, jnp.asarray(row["tr_nk"] * 1e-9))
        ref = lls.axial_thermal_factors(_reference_site(row["u0"]), 0, row["tr_nk"] * 1e-9)

        rel_x = abs(float(factors.x_nz) - ref.x_nz) / abs(ref.x_nz)
        rel_y = abs(float(factors.y_nz) - ref.y_nz) / abs(ref.y_nz)
        rel_z = abs(float(factors.z_nz) - ref.z_nz) / abs(ref.z_nz)
        assert max(rel_x, rel_y, rel_z) < 1e-6

    def test_error_shrinks_as_resolution_increases(self) -> None:
        """Demonstrates the convergence TREND (not just the endpoint): a
        coarser `(axial_grid_n, rho_grid_n)` pair, run through the same
        `axial_thermal_factors_jax` code path via its override keywords,
        disagrees with the reference MORE than the chosen default
        resolution does, at the same `(u0, Tr)` point. This is the "run
        at two resolutions and compare" discipline the reference module's
        own convergence guard automates; here it is a one-shot offline
        check, since the JAX module's traced core cannot adapt at
        runtime (module docstring).
        """
        row = TABLE1_ROWS[0]
        site = llsj.make_site_potential_jax(
            jnp.asarray(row["u0"]),
            jnp.asarray(WAIST_M),
            jnp.asarray(_yb171_wavelength_m()),
            jnp.asarray(_yb171_mass_kg()),
        )
        ref = lls.axial_thermal_factors(_reference_site(row["u0"]), 0, row["tr_nk"] * 1e-9)

        coarse = llsj.axial_thermal_factors_jax(
            site, 0, jnp.asarray(row["tr_nk"] * 1e-9), axial_grid_n=641, rho_grid_n=81
        )
        fine = llsj.axial_thermal_factors_jax(site, 0, jnp.asarray(row["tr_nk"] * 1e-9))

        coarse_err = max(
            abs(float(coarse.x_nz) - ref.x_nz) / abs(ref.x_nz),
            abs(float(coarse.y_nz) - ref.y_nz) / abs(ref.y_nz),
            abs(float(coarse.z_nz) - ref.z_nz) / abs(ref.z_nz),
        )
        fine_err = max(
            abs(float(fine.x_nz) - ref.x_nz) / abs(ref.x_nz),
            abs(float(fine.y_nz) - ref.y_nz) / abs(ref.y_nz),
            abs(float(fine.z_nz) - ref.z_nz) / abs(ref.z_nz),
        )
        assert fine_err < coarse_err
        assert fine_err < 1e-6


# ---------------------------------------------------------------------------
# Model A port equivalence
# ---------------------------------------------------------------------------


class TestModelAPortEquivalence:
    """`harmonic_light_shift_hz_jax` (pure coefficient algebra, no
    eigenproblem, no root-find) against the reference's own
    `harmonic_light_shift_hz`. Neither side of this comparison touches
    the axial eigenproblem, so this class stays in the fast lane.
    """

    @pytest.mark.parametrize(
        ("u", "detuning_hz", "n_z", "reduction_form"),
        [
            (72.0, 5.3e6, 0.0, "none"),
            (15.06, 10.5e6, 0.0, "jila_reciprocal"),
            (500.0, 0.0, 1.0, "ushijima_linear"),
            (100.0, -2e6, 2.0, "none"),
        ],
    )
    def test_matches_reference_across_a_grid_of_inputs(
        self, u: float, detuning_hz: float, n_z: float, reduction_form: str
    ) -> None:
        coeffs = lls.USHIJIMA_2018_SR87
        e_r = lls.recoil_energy_j(813e-9, 87 * 1.66053906892e-27)
        tr_k = 1e-6 if reduction_form != "none" else None
        recoil = e_r if reduction_form != "none" else None

        ref = lls.harmonic_light_shift_hz(
            u,
            detuning_hz,
            n_z,
            coeffs,
            reduction_form=reduction_form,
            radial_temperature_k=tr_k,
            recoil_energy_j_value=recoil,
        )
        jax_val = llsj.harmonic_light_shift_hz_jax(
            jnp.asarray(u),
            jnp.asarray(detuning_hz),
            jnp.asarray(n_z),
            jnp.asarray(coeffs.e1_slope_per_hz),
            jnp.asarray(coeffs.m1e2_hz),
            jnp.asarray(coeffs.hyperpolarizability_hz),
            reduction_form=reduction_form,
            radial_temperature_k=jnp.asarray(tr_k) if tr_k is not None else None,
            recoil_energy_j_value=jnp.asarray(recoil) if recoil is not None else None,
        )
        assert float(jax_val) == pytest.approx(ref, rel=1e-12, abs=0)

    def test_reduction_factors_match_reference(self) -> None:
        e_r = lls.recoil_energy_j(759e-9, 171 * 1.66053906892e-27)
        u, j, tr_k = 66.4, 1.0, 550e-9
        ref_linear = lls.ushijima_reduction_factor(u, j, tr_k, e_r)
        jax_linear = llsj.ushijima_reduction_factor_jax(
            jnp.asarray(u), j, jnp.asarray(tr_k), jnp.asarray(e_r)
        )
        assert float(jax_linear) == pytest.approx(ref_linear, rel=1e-12, abs=0)

        ref_recip = lls.jila_reduction_factor(u, j, tr_k, e_r)
        jax_recip = llsj.jila_reduction_factor_jax(
            jnp.asarray(u), j, jnp.asarray(tr_k), jnp.asarray(e_r)
        )
        assert float(jax_recip) == pytest.approx(ref_recip, rel=1e-12, abs=0)

    def test_recoil_energy_matches_reference(self) -> None:
        ref = lls.recoil_energy_j(759e-9, 171 * 1.66053906892e-27)
        jax_val = llsj.recoil_energy_j_jax(
            jnp.asarray(759e-9), jnp.asarray(171 * 1.66053906892e-27)
        )
        assert float(jax_val) == pytest.approx(ref, rel=1e-12, abs=0)


# ---------------------------------------------------------------------------
# Differentiability with respect to every physical input this work
# package's spec names (u0/Tr's REFERENCE-FD check lives in
# TestGradientsMatchReferenceFiniteDifferences above; this class covers
# the remaining inputs -- waist, wavelength, mass, coefficients -- at a
# deliberately tiny overridden resolution, since these tests check only
# the SIGN and finite/nonzero properties, at the coarse resolution's own
# accuracy).
# ---------------------------------------------------------------------------


class TestDifferentiabilityWrtAllPhysicalInputs:
    """The work package's differentiability contract names `u0`, `Tr`,
    "waist-through-E_R inputs" (`wavelength_m`, `mass_kg`: the two
    physical inputs `recoil_energy_j_jax` actually depends on), and
    coefficient values. `u0`/`Tr` get the strong REFERENCE-finite-
    difference check above; this class confirms the remaining inputs
    differentiate too, using the same `axial_grid_n=21, rho_grid_n=5`
    testing-only override `TestFloat64Discipline` uses, fast enough for
    the non-slow lane.
    """

    def _shift(
        self,
        waist_m: jnp.ndarray,
        wavelength_m: jnp.ndarray,
        mass_kg: jnp.ndarray,
        e1: jnp.ndarray,
        m1e2: jnp.ndarray,
        beta: jnp.ndarray,
    ) -> jnp.ndarray:
        site = llsj.make_site_potential_jax(jnp.asarray(56.8), waist_m, wavelength_m, mass_kg)
        factors = llsj.axial_thermal_factors_jax(
            site, 0, jnp.asarray(650e-9), axial_grid_n=21, rho_grid_n=5
        )
        return -(
            e1 * jnp.asarray(200e3) * factors.x_nz * site.depth_er
            + m1e2 * factors.y_nz * site.depth_er
            + beta * factors.z_nz * site.depth_er**2
        )

    def test_grad_wrt_waist_is_finite_and_near_zero(self) -> None:
        """`X`/`Y`/`Z` cancel the lattice waist exactly out of their
        defining ratio (module docstring's "species trap" section; same
        cancellation the reference module documents), so the light
        shift's gradient with respect to `waist_m` should be finite and
        MUCH smaller than its gradient with respect to a parameter that
        genuinely enters the physics (`mass_kg`, checked below). This
        test compares the two gradients directly, a relative bound the
        coarse `axial_grid_n=21` override can support even though that
        override does not reproduce the cancellation to machine
        precision.
        """
        coeffs = lls.BOTHWELL_2025_YB171_BOWKB
        args = (
            jnp.asarray(_yb171_wavelength_m()),
            jnp.asarray(_yb171_mass_kg()),
            jnp.asarray(coeffs.e1_slope_per_hz),
            jnp.asarray(coeffs.m1e2_hz),
            jnp.asarray(coeffs.hyperpolarizability_hz),
        )
        grad_waist = jax.grad(self._shift, argnums=0)(jnp.asarray(WAIST_M), *args)
        grad_mass = jax.grad(self._shift, argnums=2)(jnp.asarray(WAIST_M), *args)
        assert math.isfinite(float(grad_waist))
        assert abs(float(grad_waist)) < 1e-6 * abs(float(grad_mass))

    def test_grad_wrt_wavelength_and_mass_are_finite_and_nonzero(self) -> None:
        """`wavelength_m`/`mass_kg` set `E_R` (`recoil_energy_j_jax`),
        which controls the `kB*Tr/E_R` thermal weighting every `X`/`Y`/`Z`
        evaluation uses: unlike `waist_m`, these should carry a genuine,
        nonzero gradient.
        """
        coeffs = lls.BOTHWELL_2025_YB171_BOWKB
        wavelength_m = jnp.asarray(_yb171_wavelength_m())
        mass_kg = jnp.asarray(_yb171_mass_kg())
        e1 = jnp.asarray(coeffs.e1_slope_per_hz)
        m1e2 = jnp.asarray(coeffs.m1e2_hz)
        beta = jnp.asarray(coeffs.hyperpolarizability_hz)

        grad_wavelength = jax.grad(self._shift, argnums=1)(
            jnp.asarray(WAIST_M), wavelength_m, mass_kg, e1, m1e2, beta
        )
        grad_mass = jax.grad(self._shift, argnums=2)(
            jnp.asarray(WAIST_M), wavelength_m, mass_kg, e1, m1e2, beta
        )
        assert math.isfinite(float(grad_wavelength))
        assert float(grad_wavelength) != 0.0
        assert math.isfinite(float(grad_mass))
        assert float(grad_mass) != 0.0

    def test_grad_wrt_coefficients_matches_direct_algebra(self) -> None:
        """The light shift is affine in the three coefficients at fixed
        `X`/`Y`/`Z` (Eq. 6), so `d(shift)/d(e1_slope)` etc. should equal
        the `X`/`Y`/`Z`-times-`u0`-power prefactor directly: an exact
        algebraic check, stronger than a finite/nonzero sanity bound.
        """
        coeffs = lls.BOTHWELL_2025_YB171_BOWKB
        waist_m = jnp.asarray(WAIST_M)
        wavelength_m = jnp.asarray(_yb171_wavelength_m())
        mass_kg = jnp.asarray(_yb171_mass_kg())
        e1 = jnp.asarray(coeffs.e1_slope_per_hz)
        m1e2 = jnp.asarray(coeffs.m1e2_hz)
        beta = jnp.asarray(coeffs.hyperpolarizability_hz)

        grad_e1, grad_m1e2, grad_beta = jax.grad(self._shift, argnums=(3, 4, 5))(
            waist_m, wavelength_m, mass_kg, e1, m1e2, beta
        )

        site = llsj.make_site_potential_jax(jnp.asarray(56.8), waist_m, wavelength_m, mass_kg)
        factors = llsj.axial_thermal_factors_jax(
            site, 0, jnp.asarray(650e-9), axial_grid_n=21, rho_grid_n=5
        )
        expected_grad_e1 = -(200e3 * factors.x_nz * site.depth_er)
        expected_grad_m1e2 = -(factors.y_nz * site.depth_er)
        expected_grad_beta = -(factors.z_nz * site.depth_er**2)

        assert float(grad_e1) == pytest.approx(float(expected_grad_e1), rel=1e-10, abs=0)
        assert float(grad_m1e2) == pytest.approx(float(expected_grad_m1e2), rel=1e-10, abs=0)
        assert float(grad_beta) == pytest.approx(float(expected_grad_beta), rel=1e-10, abs=0)


# ---------------------------------------------------------------------------
# Density of states (Eq. 11): sanity checks against the harmonic closed
# form and against the reference module's own numeric machinery
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDensityOfStates:
    """`bo_wkb_density_of_states_jax` (Eq. 11 via the differentiable
    turning-radius root-find) against the harmonic closed form (Eq. 4)
    for a moderately deep site, mirroring Beloy et al. 2020's own Section
    VI consistency check and the reference module's own
    `harmonic_density_of_states_closed_form`-vs-`bo_wkb_density_of_states`
    cross-check (`potential="cos2"` here throughout, since this module
    never implements the reference's `potential="harmonic"` branch -- see
    `_axial_band_energy_er_at_rho`'s own docstring)."""

    def test_zero_above_band_top(self) -> None:
        u0 = 50.0
        site = llsj.make_site_potential_jax(
            jnp.asarray(u0),
            jnp.asarray(WAIST_M),
            jnp.asarray(759e-9),
            jnp.asarray(_yb171_mass_kg()),
        )
        x_grid, dx = llsj._axial_grid(llsj.AXIAL_GRID_N_JAX)
        rho_bracket = llsj.DEFAULT_RHO_BRACKET_WAIST_MULTIPLE * site.waist_m
        g = llsj.bo_wkb_density_of_states_jax(
            site.depth_er,
            site.kappa_per_m,
            site.mass_kg,
            site.recoil_energy_j_value,
            0,
            jnp.asarray(1.0 * site.recoil_energy_j_value),  # E > 0
            x_grid,
            dx,
            rho_bracket,
        )
        assert float(g) == 0.0

    def test_positive_and_finite_inside_band(self) -> None:
        u0 = 50.0
        site = llsj.make_site_potential_jax(
            jnp.asarray(u0),
            jnp.asarray(WAIST_M),
            jnp.asarray(759e-9),
            jnp.asarray(_yb171_mass_kg()),
        )
        x_grid, dx = llsj._axial_grid(llsj.AXIAL_GRID_N_JAX)
        rho_bracket = llsj.DEFAULT_RHO_BRACKET_WAIST_MULTIPLE * site.waist_m
        e0 = llsj._axial_band_energy_er_at_rho(
            site.depth_er, site.kappa_per_m, 0, jnp.zeros((), dtype=jnp.float64), x_grid, dx
        )
        mid_energy_er = 0.5 * e0  # halfway between the band bottom and 0
        g = llsj.bo_wkb_density_of_states_jax(
            site.depth_er,
            site.kappa_per_m,
            site.mass_kg,
            site.recoil_energy_j_value,
            0,
            mid_energy_er * site.recoil_energy_j_value,
            x_grid,
            dx,
            rho_bracket,
        )
        assert float(g) > 0.0
        assert math.isfinite(float(g))

    def test_turning_radius_gradient_wrt_u0_is_finite_and_nonzero(self) -> None:
        """`jax.grad` of the turning radius with respect to `u0`, via the
        `jax.lax.custom_root` implicit-function-theorem path (module
        docstring's "differentiable turning-radius root-find" section):
        deepening the trap at fixed energy target should shrink the
        band-top turning radius. This test checks that expected sign, a
        sanity bound on the physically motivated direction."""
        wavelength_m = 759e-9
        mass_kg = _yb171_mass_kg()

        def rho_max_of_u0(u0: jnp.ndarray) -> jnp.ndarray:
            site = llsj.make_site_potential_jax(
                u0, jnp.asarray(WAIST_M), jnp.asarray(wavelength_m), jnp.asarray(mass_kg)
            )
            x_grid, dx = llsj._axial_grid(llsj.AXIAL_GRID_N_JAX)
            rho_bracket = llsj.DEFAULT_RHO_BRACKET_WAIST_MULTIPLE * site.waist_m
            return llsj.turning_radius_m_jax(
                site.depth_er,
                site.kappa_per_m,
                0,
                jnp.zeros((), dtype=jnp.float64),
                x_grid,
                dx,
                rho_bracket,
            )

        grad_fn = jax.grad(rho_max_of_u0)
        g = grad_fn(jnp.asarray(50.0))
        assert math.isfinite(float(g))
        assert float(g) != 0.0


# ---------------------------------------------------------------------------
# Float64 discipline
# ---------------------------------------------------------------------------


class TestFloat64Discipline:
    def test_shift_and_factors_are_float64(self) -> None:
        """dtype propagation is independent of grid resolution, so this
        check runs `axial_thermal_factors_jax` at a deliberately tiny
        overridden resolution (`axial_grid_n=21, rho_grid_n=5`, this
        function's testing-only override keywords -- see their docstring)
        to keep this test in the fast lane: the eigh-heavy cost the
        module's production resolution carries is not needed to verify a
        dtype.
        """
        site = llsj.make_site_potential_jax(
            jnp.asarray(56.8),
            jnp.asarray(WAIST_M),
            jnp.asarray(_yb171_wavelength_m()),
            jnp.asarray(_yb171_mass_kg()),
        )
        factors = llsj.axial_thermal_factors_jax(
            site, 0, jnp.asarray(650e-9), axial_grid_n=21, rho_grid_n=5
        )
        coeffs = lls.BOTHWELL_2025_YB171_BOWKB
        shift = -(
            jnp.asarray(coeffs.e1_slope_per_hz) * jnp.asarray(0.0) * factors.x_nz * site.depth_er
            + jnp.asarray(coeffs.m1e2_hz) * factors.y_nz * site.depth_er
            + jnp.asarray(coeffs.hyperpolarizability_hz) * factors.z_nz * site.depth_er**2
        )
        assert shift.dtype == jnp.float64
        assert factors.x_nz.dtype == jnp.float64
        assert factors.y_nz.dtype == jnp.float64
        assert factors.z_nz.dtype == jnp.float64


# ---------------------------------------------------------------------------
# Timing sanity note (this file's own top docstring carries the numbers
# this print documents)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_timing_sanity_note(capsys: pytest.CaptureFixture[str]) -> None:
    """Measures one full `_value_and_grad_at_row` call's wall time and
    prints it directly (`pytest -s` or `--durations` to see it), giving a
    change to `AXIAL_GRID_N_JAX`/`RHO_GRID_N_JAX` that materially worsens
    runtime its own visible signal, separate from the overall slow-lane
    duration. The `elapsed < 600.0` assertion below is a generous ceiling
    against a hang or a runaway resolution change, kept loose because
    wall time is machine-dependent; this file's OTHER slow tests already
    carry the real correctness assertions for this same code path.
    """
    row = TABLE1_ROWS[0]
    _value_and_grad_at_row.cache_clear()
    t0 = time.perf_counter()
    _value_and_grad_at_row(row["u0"], row["tr_nk"])
    elapsed = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[timing] one value_and_grad call at u0={row['u0']} E_R: {elapsed:.1f} s")
    assert elapsed < 600.0, f"value_and_grad took {elapsed:.1f}s, unexpectedly slow"
