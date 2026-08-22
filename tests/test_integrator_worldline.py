# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for cliffordclock.integrator.worldline (CONVENTIONS.md E9, E17-E24).

Covers the rotor-integrator test contract: normalization, energy/phase
consistency, and long-run stability; convergence order is covered in
``tests/test_integrator_stepper.py``.

Uses ``cliffordclock.fields.synthetic`` for
closed-form test fields, via the small ``_as_field_fn`` adapter below (MAJOR
6 fix -- the previous ``integrator._closed_form_fields`` fallback module has
been deleted: WP2's ``fields.synthetic`` was already committed before that
fallback module was written, so the fallback's justification was stale).
``fields.synthetic`` factories return two separate callables ``(e_fn,
grad_fn)`` batched over a leading ``(N, 3)`` axis; the integrator's
``FieldFn`` wants one combined callable ``pos -> (E, grad_E)`` for a single
``(3,)`` position (matching ``FieldSmoother.evaluate``'s convention) -- this
is exactly what the adapter bridges. A public adapter is WP6 scope, not
this WP; this one is deliberately local/private to the test suite.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.cl13 import IDX_SCALAR
from cliffordclock.constants import ELECTRON_MASS, SPEED_OF_LIGHT
from cliffordclock.fields.synthetic import constant_gradient_field as _synth_constant_gradient_field
from cliffordclock.fields.synthetic import uniform_field as _synth_uniform_field
from cliffordclock.integrator.omega import pivot_perturbation
from cliffordclock.integrator.stepper import rotor_step
from cliffordclock.integrator.worldline import (
    DEFAULT_RENORM_EVERY,
    FieldFn,
    integrate_ensemble,
    integrate_worldline,
    kahan_sum,
)

_M_E_C2 = ELECTRON_MASS * SPEED_OF_LIGHT**2
_IDENTITY = jnp.zeros(16, dtype=jnp.float64).at[IDX_SCALAR].set(1.0)


_SynthFieldPair = tuple[Callable[..., jnp.ndarray], Callable[..., jnp.ndarray]]


def _as_field_fn(synth_pair: _SynthFieldPair) -> FieldFn:
    """Adapt a `fields.synthetic` `(e_fn, grad_fn)` pair to the integrator's
    single combined-callable `FieldFn` convention (MAJOR 6 fix; see module
    docstring). `fields.synthetic` callables are hard-coded for a leading
    `(N, 3)` batch axis (e.g. `grad_fn` reads `pos.shape[0]` as `N`), while
    the integrator calls `field_fn` with a single un-batched `(3,)`
    position each step -- so this adapter also bridges that batch-of-one
    shape mismatch, not just the two-callables-vs-one-callable shape.
    """
    e_fn, grad_fn = synth_pair

    def field_fn(pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        pos_batched = jnp.asarray(pos, dtype=jnp.float64)[None, :]
        return e_fn(pos_batched)[0], grad_fn(pos_batched)[0]

    return field_fn


def uniform_field(e0: jnp.ndarray) -> FieldFn:
    return _as_field_fn(_synth_uniform_field(e0))


def constant_gradient_field(e0: jnp.ndarray, grad: jnp.ndarray) -> FieldFn:
    return _as_field_fn(_synth_constant_gradient_field(e0, grad))


def test_v1_uniform_field_gradient_driven_shift_below_1e19() -> None:
    """WP3 test contract item 1 (CONVENTIONS.md V1): uniform E, grad_E = 0,
    static atom (v=0) -- the rotor-rate shift beyond the scalar E21
    baseline is < 1e-19 in absolute terms (no boost term is present, since
    grad = 0, so rotor and scalar pipelines should agree almost exactly).
    """
    field_fn = uniform_field(jnp.array([0.0, 0.0, 1000.0]))
    mu = jnp.array([0.0, 0.0, 1e-35])  # sized so P - 1 ~ 1.2e-18 (E14a).

    result = integrate_worldline(field_fn, jnp.zeros(3), 1.0, mu, n_steps=1000)

    shift_beyond_scalar_baseline = abs(float(result.phase_rotor) - float(result.phase))
    assert shift_beyond_scalar_baseline < 1e-19


def test_v2_constant_gradient_closed_form() -> None:
    """WP3 test contract item 4 (CONVENTIONS.md V2): static atom at r0 in a
    constant-gradient field; total accumulated phase matches the closed
    form (P(r0) - 1) * T_tilde to 1e-14 relative.
    """
    grad = jnp.array([[1e3, 2e3, 0.0], [0.0, -1e3, 5e2], [3e2, 0.0, -2e3]])
    e0 = jnp.zeros(3)
    field_fn = constant_gradient_field(e0, grad)
    r0 = jnp.array([0.01, -0.02, 0.03])
    mu = jnp.array([1e-25, 2e-25, -3e-25])
    dtau = 1.0
    n_steps = 1000

    result = integrate_worldline(field_fn, r0, dtau, mu, n_steps=n_steps)

    delta_e_r0 = e0 + r0 @ grad
    p_minus_1_r0 = pivot_perturbation(delta_e_r0, mu)
    t_tilde = n_steps * dtau
    expected_phase = p_minus_1_r0 * t_tilde

    # BLOCKER 2: a rel-only pytest.approx comparison silently uses a
    # default abs of 1e-12, which dominates for this ~3.8e-7-scale phase and
    # would let a ~1e-6 relative error slip through. assert_allclose with
    # atol=0 makes rtol the only thing that governs (matches the pattern
    # already used in test_batch_equivalence below).
    np.testing.assert_allclose(float(result.phase), float(expected_phase), rtol=1e-14, atol=0)


@pytest.mark.slow
def test_norm_preservation_one_million_steps() -> None:
    """WP3 test contract item 2: |<R R~>_0 - 1| < 1e-12 after 1,000,000
    Compton steps in a constant-gradient field, with renormalization-
    interval drift diagnostics also < 1e-12.
    """
    grad = jnp.array([[1e3, 0.0, 0.0], [0.0, 1e3, 0.0], [0.0, 0.0, -2e3]])
    e0 = jnp.array([0.0, 0.0, 1000.0])
    field_fn = constant_gradient_field(e0, grad)
    r0 = jnp.array([0.01, -0.02, 0.03])
    mu = jnp.array([1e-22, 0.0, 0.0])

    result = integrate_worldline(
        field_fn, r0, 1.0, mu, renorm_every=DEFAULT_RENORM_EVERY, n_steps=1_000_000
    )

    assert result.n_steps == 1_000_000
    assert float(result.norm_error) < 1e-12
    assert float(result.max_norm_drift) < 1e-12


def test_batch_equivalence() -> None:
    """WP3 test contract item 5: integrate_ensemble over 10 trajectories
    equals a Python loop of integrate_worldline, to 1e-15.
    """
    grad = jnp.array([[5e2, 0.0, 0.0], [0.0, -3e2, 0.0], [0.0, 0.0, -2e2]])
    e0 = jnp.array([100.0, 0.0, -50.0])
    field_fn = constant_gradient_field(e0, grad)
    mu = jnp.array([1e-24, -2e-24, 1e-24])
    dtau = 0.25

    rng = np.random.default_rng(0)
    n_traj, n_steps = 10, 40
    trajectories = jnp.asarray(rng.uniform(-0.05, 0.05, size=(n_traj, n_steps + 1, 3)))

    ensemble_result = integrate_ensemble(field_fn, trajectories, dtau, mu)

    loop_phases = []
    loop_phase_rotors = []
    loop_norm_errors = []
    for i in range(n_traj):
        single = integrate_worldline(field_fn, trajectories[i], dtau, mu)
        loop_phases.append(float(single.phase))
        loop_phase_rotors.append(float(single.phase_rotor))
        loop_norm_errors.append(float(single.norm_error))

    np.testing.assert_allclose(
        np.asarray(ensemble_result.phase), np.asarray(loop_phases), rtol=1e-15, atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(ensemble_result.phase_rotor), np.asarray(loop_phase_rotors), rtol=1e-15, atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(ensemble_result.norm_error), np.asarray(loop_norm_errors), rtol=1e-15, atol=1e-30
    )


def test_differentiability_grad_matches_finite_difference() -> None:
    """WP3 test contract item 6: jax.grad of total phase w.r.t. a field-
    scaling parameter matches central finite differences to 1e-6 relative.
    """

    def phase_of_scale(scale: jnp.ndarray) -> jnp.ndarray:
        e0 = scale * jnp.array([0.0, 0.0, 1000.0])
        grad = scale * jnp.array([[1e3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1e3]])
        field_fn = constant_gradient_field(e0, grad)
        pos = jnp.array([0.01, 0.02, -0.01])
        mu = jnp.array([1e-22, 0.0, 0.0])
        result = integrate_worldline(field_fn, pos, 0.5, mu, n_steps=20)
        return result.phase

    scale0 = 1.0
    analytic = float(jax.grad(phase_of_scale)(jnp.asarray(scale0)))

    h = 1e-4
    finite_diff = (
        float(phase_of_scale(jnp.asarray(scale0 + h)))
        - float(phase_of_scale(jnp.asarray(scale0 - h)))
    ) / (2 * h)

    rel_error = abs(analytic - finite_diff) / abs(finite_diff)
    assert rel_error < 1e-6


def test_precision_floor_stable_under_summation_reorder() -> None:
    """WP3 test contract item 7: with a gradient sized to produce a
    1e-18-level fractional shift, the computed shift is stable to < 1%
    under a change of summation order -- evidence the compensated (Kahan)
    accumulation works.

    The per-step increments are deliberately *not* all identical (a jitter
    modulates the field amplitude per step): summing a sequence of
    identical floats is trivially order-invariant regardless of summation
    method, which would defeat the point of this test (it must actually
    distinguish "the accumulator handles reordering" from "there was
    nothing for reordering to disturb").

    NOTE (MAJOR 5): this test alone is *not* discriminating against a
    broken/naive accumulator -- naive summation happens to also pass it,
    with ~13 orders of magnitude of headroom (the reordering perturbation
    here is far larger than what naive summation actually loses). It is
    kept as WP3 test-contract item 7's literal reorder-stability check;
    see `test_kahan_sum_is_discriminating_against_naive_summation` below
    for the torture-test case that actually fails under naive summation.
    """
    mu = jnp.array([0.0, 0.0, 1e-35])
    grad_delta_e = jnp.zeros((3, 3))
    v = jnp.zeros(3)
    dtau = 1.0
    n = 100_000

    # Amplitude jitter in [0.5, 1.5] x 1000 V/m -- keeps P - 1 at the same
    # 1e-18 order of magnitude every step, while making consecutive
    # increments genuinely different values (not a repeated constant).
    key = jax.random.PRNGKey(1)
    jitter = 1.0 + 0.5 * jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)

    def body(r: jnp.ndarray, amp: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        delta_e = jnp.array([0.0, 0.0, 1000.0]) * amp
        r_next, dphase = rotor_step(r, delta_e, grad_delta_e, v, mu, dtau)
        return r_next, dphase.scalar

    _r_final, increments = jax.lax.scan(body, _IDENTITY, jitter)
    assert increments.shape == (n,)
    assert increments.dtype == jnp.float64
    assert len(jnp.unique(increments)) > n // 2, "increments are not actually varying"

    mean_increment = float(jnp.mean(increments))
    assert 1e-19 < abs(mean_increment) < 1e-17  # confirms the 1e-18-level regime.

    original_sum = kahan_sum(increments)
    perm = jax.random.permutation(jax.random.PRNGKey(0), n)
    shuffled_sum = kahan_sum(increments[perm])

    rel_diff = abs(float(shuffled_sum) - float(original_sum)) / abs(float(original_sum))
    assert rel_diff < 0.01


def _naive_sequential_sum(values: jnp.ndarray) -> jnp.ndarray:
    """Plain (uncompensated) sequential running sum, via the same
    `lax.scan` shape as `kahan_sum` minus the compensation term -- a
    reference "what if Kahan were broken/removed" implementation, used
    only to demonstrate that `kahan_sum` actually beats it (MAJOR 5).
    """

    def body(total: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        return total + x, None

    total, _ = jax.lax.scan(body, jnp.asarray(0.0, dtype=jnp.float64), values)
    return total


def test_kahan_sum_is_discriminating_against_naive_summation() -> None:
    """MAJOR 5 fix: a torture-test case that actually distinguishes
    compensated (Kahan) summation from naive summation, unlike
    `test_precision_floor_stable_under_summation_reorder` above.

    A large O(1) transient first term followed by 2,000,000 increments of
    1e-18 is the textbook failure case for naive fp64 summation: each
    increment is far below the ULP of a running total near 1.0
    (~2.22e-16), so a naive running sum is provably stuck at exactly
    1.0 -- it never moves. Kahan/compensated summation recovers the
    increments via its running compensation term. Both are checked
    against a `math.fsum` reference (exact double-precision summation,
    immune to this failure mode by construction), so a broken or
    accidentally removed compensation step fails this test.
    """
    n = 2_000_000
    transient = 1.0
    increment = 1e-18
    values_list = [transient] + [increment] * n
    values = jnp.asarray(values_list, dtype=jnp.float64)

    reference = math.fsum(values_list)

    naive_total = float(_naive_sequential_sum(values))
    assert naive_total == 1.0, "naive summation unexpectedly resolved the small increments"

    kahan_total = float(kahan_sum(values))
    assert kahan_total != naive_total, "kahan_sum failed to distinguish itself from naive summation"

    # Kahan must land within a handful of ULPs of the exact reference --
    # far tighter than naive summation's ~2e-12 (100%-of-signal) error.
    np.testing.assert_allclose(kahan_total, reference, rtol=0, atol=1e-13)


def test_static_broadcast_trajectory_gives_zero_velocity() -> None:
    """A static (3,) trajectory (quadrature-node convention) broadcasts to
    zero finite-difference velocity, so the boost term (E18) is exactly
    zero regardless of the field's gradient.
    """
    grad = jnp.array([[1e5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    field_fn = constant_gradient_field(jnp.zeros(3), grad)
    mu = jnp.array([1e-20, 0.0, 0.0])

    result = integrate_worldline(field_fn, jnp.array([0.1, 0.0, 0.0]), 1.0, mu, n_steps=100)
    # No boost => rotor and scalar pipelines agree to near machine precision.
    assert abs(float(result.phase_rotor) - float(result.phase)) < 1e-14 * max(
        1.0, abs(float(result.phase))
    )


def test_worldline_result_dtype_float64() -> None:
    field_fn = uniform_field(jnp.array([0.0, 0.0, 1.0]))
    mu = jnp.array([1e-25, 0.0, 0.0])
    result = integrate_worldline(field_fn, jnp.zeros(3), 1.0, mu, n_steps=10)
    assert result.r_final.dtype == jnp.float64
    assert result.phase.dtype == jnp.float64
    assert result.phase_rotor.dtype == jnp.float64
    assert result.fractional_shift.dtype == jnp.float64
    assert result.norm_error.dtype == jnp.float64
    assert result.max_norm_drift.dtype == jnp.float64


def test_rejects_missing_n_steps_for_static_trajectory() -> None:
    field_fn = uniform_field(jnp.array([0.0, 0.0, 1.0]))
    mu = jnp.array([1e-25, 0.0, 0.0])
    with pytest.raises(ValueError, match="n_steps"):
        integrate_worldline(field_fn, jnp.zeros(3), 1.0, mu)


def test_integrate_worldline_runs_under_jit_with_traced_dtau_and_mu() -> None:
    """MAJOR 4 fix: `integrate_worldline` must run end-to-end under
    `jax.jit` (WP3 spec) with `dtau` and `mu` traced. `field_fn`,
    `renorm_every`, and `n_steps` are marked static -- `field_fn` is a
    plain (hashable-by-identity) Python callable, and `renorm_every`/
    `n_steps` control array shapes / a Python-level cadence check against
    a concrete int, so none of the three can be traced.

    Regression coverage for the previous `IntegratorParams` design, where
    `mu` (dynamic) and `renorm_every` (static) were bundled into a single
    NamedTuple argument: jitting that argument as a whole traced *both*
    fields, and the old `int(params.renorm_every)` inside the scan path
    then raised `ConcretizationTypeError` at trace time.
    """
    grad = jnp.array([[1e3, 0.0, 0.0], [0.0, -1e3, 0.0], [0.0, 0.0, 0.0]])
    e0 = jnp.array([0.0, 0.0, 500.0])
    field_fn = constant_gradient_field(e0, grad)
    r0 = jnp.array([0.01, -0.02, 0.03])
    mu = jnp.array([1e-24, 2e-24, -1e-24])
    dtau = 0.5
    n_steps = 200

    jitted = jax.jit(integrate_worldline, static_argnames=("field_fn", "renorm_every", "n_steps"))

    result_jit = jitted(
        field_fn,
        r0,
        jnp.asarray(dtau),
        mu,
        renorm_every=DEFAULT_RENORM_EVERY,
        n_steps=n_steps,
    )
    result_eager = integrate_worldline(
        field_fn, r0, dtau, mu, renorm_every=DEFAULT_RENORM_EVERY, n_steps=n_steps
    )

    result_fields = (
        "r_final",
        "phase",
        "phase_rotor",
        "fractional_shift",
        "norm_error",
        "max_norm_drift",
        "n_steps",
    )
    for field in result_fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(result_jit, field)), np.asarray(getattr(result_eager, field))
        )


def test_integrate_worldline_grad_wrt_scale_and_mu_works_under_jit() -> None:
    """MAJOR 4 fix, part (b): `jax.grad` works, under `jax.jit`, both
    w.r.t. a field-scale parameter closed over by `field_fn` (as in
    `test_differentiability_grad_matches_finite_difference`, but jitted)
    and w.r.t. `mu` itself -- only possible now that `mu` is a first-class
    dynamic argument of `integrate_worldline` rather than a field buried
    inside a static-pytree `IntegratorParams`.
    """
    grad_tensor = jnp.array([[1e3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1e3]])
    pos = jnp.array([0.01, 0.02, -0.01])
    dtau = 0.5
    n_steps = 20

    def phase_of_scale_and_mu(scale: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        e0 = scale * jnp.array([0.0, 0.0, 1000.0])
        grad = scale * grad_tensor
        field_fn = constant_gradient_field(e0, grad)
        result = integrate_worldline(
            field_fn, pos, dtau, mu, renorm_every=DEFAULT_RENORM_EVERY, n_steps=n_steps
        )
        return result.phase

    scale0 = jnp.asarray(1.0)
    mu0 = jnp.array([1e-22, 0.0, 0.0])

    grad_scale_eager = jax.grad(phase_of_scale_and_mu, argnums=0)(scale0, mu0)
    grad_mu_eager = jax.grad(phase_of_scale_and_mu, argnums=1)(scale0, mu0)

    grad_scale_jit = jax.jit(jax.grad(phase_of_scale_and_mu, argnums=0))(scale0, mu0)
    grad_mu_jit = jax.jit(jax.grad(phase_of_scale_and_mu, argnums=1))(scale0, mu0)

    assert jnp.isfinite(grad_scale_jit)
    assert bool(jnp.all(jnp.isfinite(grad_mu_jit)))
    assert float(jnp.max(jnp.abs(grad_mu_jit))) > 0.0, "d(phase)/d(mu) is vacuously zero"

    np.testing.assert_allclose(
        np.asarray(grad_scale_jit), np.asarray(grad_scale_eager), rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(grad_mu_jit), np.asarray(grad_mu_eager), rtol=1e-12, atol=0
    )
