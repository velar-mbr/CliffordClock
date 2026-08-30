# SPDX-License-Identifier: AGPL-3.0-or-later
"""Direct rotor<->scalar cross-check for the E14b Stark pivot (WP16).

CONVENTIONS.md E15-E18 state the pivot -> spin-connection -> Ω chain for a
*general* P(r); WP7 instantiated the pivot itself (E14b, quadratic DC
Stark) and WP16 instantiates the rest of that chain (E16 spin connection,
E18 Ω) with the same E14b pivot, via
``cliffordclock.integrator.omega.build_omega_stark`` (see that module for
the physics). Before this WP, no test ran the Cl(1,3) rotor against the
`coupling.type='stark_dc'` scalar formulation directly -- CONVENTIONS.md's
"production path vs. general engine" note rested the claim on a three-step
chain (E24 for the *linear* E14a coupling, the E14a/E14b bridge identity,
and an independent theory review's second-order bound) instead. This file
is the direct test that chain predicted would agree, mirroring
``tests/test_integrator_stepper.py``'s E14a E24 tests structurally (same
rotor-vs-scalar-phase-increment comparison, same second-order-divergence
ratio-scaling design) with the E14b pivot substituted for E14a's.

Four groups of tests (WP16 scope item 3):

1. Realistic-regime first-order agreement (E24 acceptance criterion).
2. Exaggerated-boost regime: O(omega_boost^2) ratio-scaling (E24's
   permitted second-order divergence).
3. Uniform-field null (V1-style invariance): zero field gradient forces
   omega_boost to exactly zero, regardless of velocity.
4. v=0 static-node check: zero velocity forces omega_boost to exactly
   zero, regardless of field gradient -- the regime every existing
   `coupling.type='stark_dc'` + `integration.mode='worldline'` pipeline
   call site (lattice quadrature nodes) actually runs in.

Groups 1-4 exercise no pipeline-level plumbing (that is
``cliffordclock.pipeline._stark_rotor_ensemble``, exercised end-to-end by
the pipeline/benchmark test suites for its one production call site,
which always passes static v=0 lattice nodes); those groups test the
rotor construction itself, one step at a time, exactly as
``test_integrator_stepper.py`` does for E14a.

Two more groups close a review-identified coverage gap (WP16 review,
2026-08-10):

5. Non-symmetric-gradient orientation guard: every case above uses a
   zero or single-diagonal-element `grad_e_total`, so
   :func:`~cliffordclock.integrator.omega.spin_connection_stark`'s
   `einsum("...kj,...j->...k", grad_e_total, e_total)` contraction
   (E13: `grad[i, j] = d_i E_j`) is indistinguishable from its transposed
   twin (`einsum("...jk,...j->...k", ...)`) in every existing test --
   mirrors `tests/test_fields_smoother.py`'s
   `test_grad_orientation_nonsymmetric_gradient` orientation-trap pattern
   with a genuinely non-symmetric gradient and an explicit
   transposed-contraction guard assertion.
6. `_stark_rotor_ensemble` time-varying-Ω convergence: groups 1-4 (and
   every real pipeline call site) only ever pass a trajectory with `v=0`
   at every step (static lattice quadrature nodes) or a single
   hand-picked static `(E, grad_E, v)` triple -- no existing test drives
   the multi-step ensemble accumulator itself with a trajectory whose
   position, and hence Ω, genuinely varies step to step. This group
   builds a synthetic polynomial field and a constant-velocity moving
   trajectory, then checks the step-halving error ratio for O(dτ²)
   convergence (proving true midpoint evaluation, not e.g. an
   endpoint-evaluated stand-in that would only be first-order accurate).
   `_stark_rotor_ensemble` is pipeline-private, but this exercises the
   function directly (no config/CLI plumbing) in the same spirit as
   groups 1-4's direct calls into `build_omega_stark` -- the pipeline's
   one production call site (`integration.mode='worldline'`,
   `coupling.type='stark_dc'`) cannot itself construct a moving
   trajectory (lattice nodes are always static), so this coverage can
   only be reached by calling the accumulator directly.

A seventh group extends (rather than duplicates) group 4 for WP20
(CONVENTIONS.md E32/E33, the blackbody-radiation shift): the same v=0
static-node head-to-head, with a nonzero `bbr_pivot_perturbation` composed
into both sides, per the WP20 acceptance criterion "extend the WP16
rotor-scalar head-to-head test with BBR active rather than duplicating it".
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.cl13 import (
    IDX_E01,
    IDX_E02,
    IDX_E03,
    IDX_E12,
    IDX_SCALAR,
    exp_bivector,
    geometric_product,
)
from cliffordclock.constants import LAMBDA_BAR_COMPTON, SPEED_OF_LIGHT, TAU_COMPTON
from cliffordclock.ensemble.species import Species, StarkCoefficients, get_species
from cliffordclock.fields.synthetic import as_field_fn
from cliffordclock.integrator.omega import (
    bbr_pivot_perturbation,
    build_omega_stark,
    spin_connection_stark,
)
from cliffordclock.integrator.stepper import rotor_plane_angle
from cliffordclock.integrator.worldline import FieldFn as CombinedFieldFn
from cliffordclock.pipeline import _stark_rotor_ensemble

_IDENTITY = jnp.zeros(16, dtype=jnp.float64).at[IDX_SCALAR].set(1.0)


def _rotor_step_stark(
    r: jnp.ndarray,
    e_total: jnp.ndarray,
    grad_e_total: jnp.ndarray,
    v: jnp.ndarray,
    species_or_coeffs: Species | StarkCoefficients,
    dtau: float,
    *,
    bbr_pivot_perturbation: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One E17/E19 exponential-midpoint step under the E14b Stark pivot.

    Deliberately *not* a call into ``cliffordclock.integrator.stepper.rotor_step``
    (which is hardwired to E14a's ``build_omega``/`mu` signature -- see
    that module's docstring) -- this reimplements `rotor_step`'s exact
    body (E19: ``R_next = exp(-1/2 * Omega * dtau) * R``, plus both phase
    increments) with :func:`~cliffordclock.integrator.omega.build_omega_stark`
    in place of :func:`~cliffordclock.integrator.omega.build_omega` --
    same formulas, same per-step structure, only the pivot source differs
    (WP16 scope: `stepper.py` itself stays untouched).

    Returns
    -------
    (r_next, dphase_scalar, dphase_rotor) : tuple of jax.Array
        `r_next`: the advanced rotor. `dphase_scalar`: the primary E21/E22
        phase increment (``Omega``'s `B_hat_C`-plane coefficient times
        `dtau`). `dphase_rotor`: the E24 rotor-extracted phase increment
        (:func:`~cliffordclock.integrator.stepper.rotor_plane_angle` of
        this step's `exp_bivector` factor).
    """
    omega = build_omega_stark(
        e_total,
        grad_e_total,
        species_or_coeffs,
        v,
        bbr_pivot_perturbation=bbr_pivot_perturbation,
    )
    generator = (-0.5 * dtau) * omega
    delta_r = exp_bivector(generator)
    r_next = geometric_product(delta_r, r)
    dphase_scalar = omega[..., IDX_E12] * dtau
    dphase_rotor = rotor_plane_angle(delta_r)
    return r_next, dphase_scalar, dphase_rotor


# ---------------------------------------------------------------------------
# 1. Realistic regime: first-order agreement (E24 acceptance criterion).
# ---------------------------------------------------------------------------


def test_realistic_regime_first_order_agreement() -> None:
    """Realistic Sr87 field magnitude + realistic cold-atom v/c: the
    rotor<->scalar discrepancy is unmeasurably small, per E24's "equality
    required at first order" criterion.

    Parameters: `species=Sr87` (registry-sourced k_S/nu0, the same
    coefficients every physically-meaningful `coupling.type='stark_dc'`
    example in this repository uses), field magnitude `10 V/m` (NPL
    residual-field scale, see `benchmarks/loaders.NPL_RESIDUAL_FIELD_V_PER_M`),
    a *deliberately exaggerated* synthetic gradient `1e18 V/m^2` (mirrors
    E14a's own realistic-regime test,
    `test_e24_first_order_agreement_at_realistic_boost_scale` in
    `test_integrator_stepper.py`, which also pairs a physically-tiny
    coupling with a synthetically large gradient -- the point in both
    cases is that even an unrealistically steep gradient cannot make the
    boost term matter once `v/c` itself is realistic), and
    `v/c = 1e-6` (thermal cold-atom scale, matching E14a's test and the
    `scalar_rate_perturbation` docstring's own Sr-@-1uK citation).

    Tolerance derivation: identical reasoning to E14a's realistic-regime
    test bound. The rotor<->scalar discrepancy is `O(omega_boost^2)`
    (E24); `omega_boost = (v/c) * lambda_bar_C * omega_tilde_0k` with
    `lambda_bar_C ~ 3.86e-13 m` and, at these parameters,
    `omega_tilde_0k ~ -5.5e-14` (measured: see the docstring of
    `cliffordclock.integrator.omega.spin_connection_stark`'s formula) --
    so `omega_boost ~ 1e-6 * 3.86e-13 * 5.5e-14 ~ 2e-32`, and its square
    is far below float64's ~1e-16 relative epsilon at any scale this
    module's phase values reach. `1e-18` is the same absolute floor
    E14a's analogous test uses (the project's target measurability
    floor, CONVENTIONS.md section 1/E10) -- comfortably above float64
    noise (~1e-16 relative to a ~1e-14 phase value here) and comfortably
    below any physically meaningful shift.
    """
    species = get_species("Sr87")
    e_total = jnp.array([10.0, 0.0, 0.0])
    grad_e_total = jnp.zeros((3, 3)).at[0, 0].set(1e18)
    v = jnp.array([1e-6 * SPEED_OF_LIGHT, 0.0, 0.0])
    dtau = 0.05

    _r_next, dphase_scalar, dphase_rotor = _rotor_step_stark(
        _IDENTITY, e_total, grad_e_total, v, species, dtau
    )
    # Non-vacuous: the primary phase increment itself must be nonzero, so
    # this is a genuine agreement check, not 0 == 0.
    assert abs(float(dphase_scalar)) > 1e-16
    discrepancy = abs(float(dphase_rotor) - float(dphase_scalar))
    assert discrepancy < 1e-18


# ---------------------------------------------------------------------------
# 2. Exaggerated-boost regime: O(omega_boost^2) ratio scaling.
# ---------------------------------------------------------------------------


def test_exaggerated_boost_second_order_divergence_scales_quadratically() -> None:
    """Deliberately exaggerated field/gradient/velocity to numerically
    exercise the omega_boost machinery, then confirm the E24-permitted
    O(omega_boost^2) discrepancy scaling -- mirrors
    `test_e24_second_order_divergence_scales_quadratically_with_boost` in
    `test_integrator_stepper.py` (E14a), same ratio-4-per-doubling design,
    same acceptance band.

    Parameters chosen by direct search (documented, not tuned to a
    result): `species=Sr87`, field `1e10 V/m` and gradient `1e18 V/m^2`
    (both deliberately unrealistic -- this test's entire purpose is to
    force `omega_boost` into a numerically resolvable regime, exactly as
    E14a's analogous test uses an unrealistic `grad_delta_e[0,0]=1e18`),
    `dtau=0.05`, and a velocity scale `v/c = 0.08 * s` for
    `s in {1, 2, 4, 8}` -- large enough that `omega_boost` (hence its
    square) clears the float64 rounding-noise floor by two orders of
    magnitude at `s=1` (measured discrepancy ~1.8e-15; the noise floor is
    the observed phase magnitude ~0.036 times the ~2e-16 float64 relative
    epsilon, i.e. ~7e-18 -- the `> 5e-16` sanity assertion below sits
    comfortably between the two, never within a factor of 3 of either),
    while staying well inside the non-relativistic regime the E21
    kinematic-term algebra assumes.
    """
    species = get_species("Sr87")
    e_total = jnp.array([1e10, 0.0, 0.0])
    grad_e_total = jnp.zeros((3, 3)).at[0, 0].set(1e18)
    dtau = 0.05

    scales = [1.0, 2.0, 4.0, 8.0]
    discrepancies = []
    for s in scales:
        v = jnp.array([0.08 * s * SPEED_OF_LIGHT, 0.0, 0.0])
        _r_next, dphase_scalar, dphase_rotor = _rotor_step_stark(
            _IDENTITY, e_total, grad_e_total, v, species, dtau
        )
        discrepancies.append(abs(float(dphase_rotor) - float(dphase_scalar)))

    assert discrepancies[0] > 5e-16, "boost machinery not actually exercised (discrepancy ~0)"

    ratios = [discrepancies[i + 1] / discrepancies[i] for i in range(len(discrepancies) - 1)]
    # Doubling the boost-driving velocity should ~quadruple the discrepancy
    # (O(omega_boost^2)); same generous leading-order-asymptotic band as
    # E14a's analogous test (test_integrator_stepper.py).
    for ratio in ratios:
        assert 3.5 < ratio < 4.6, f"expected ~4x scaling per doubling, got {ratio}"


# ---------------------------------------------------------------------------
# 3. Uniform-field null (V1-style invariance).
# ---------------------------------------------------------------------------


def test_uniform_field_null_rotor_matches_scalar_exactly() -> None:
    """Zero field gradient (a spatially uniform field) forces omega_boost
    to *exactly* zero (`spin_connection_stark`'s einsum contraction
    against a `grad_e_total` of all zeros returns exactly `0.0`, not just
    a small value) -- regardless of velocity. This is CONVENTIONS.md V1's
    invariance ("uniform field ... the gradient-driven part of the shift
    is zero") reproduced through the E14b rotor path specifically: with
    `Omega` confined to the pure `B_hat_C = e_1^e_2` plane (no `e_k^e_0`
    boost components at all), `rotor_plane_angle` recovers the scalar
    phase increment exactly, to the same float64-rounding-only tolerance
    as `test_rotor_plane_angle_matches_scalar_phase_for_pure_e12_rotation`
    in `test_integrator_stepper.py` (E14a's zero-boost case) -- `abs=1e-14`.

    A large velocity (`v/c = 0.1`, deliberately far from the realistic
    cold-atom regime) is used specifically to demonstrate this is an
    *exact algebraic identity* (the boost bivector is zero, a stronger
    claim than merely being small), not a coincidence of small `v`.
    """
    species = get_species("Sr87")
    e_total = jnp.array([1e10, 0.0, 0.0])
    grad_e_total = jnp.zeros((3, 3))  # uniform field: no spatial variation at all
    v = jnp.array([0.1 * SPEED_OF_LIGHT, 0.0, 0.0])
    dtau = 0.05

    omega = build_omega_stark(e_total, grad_e_total, species, v)
    # The boost bivector components (e_k ^ e_0) are exactly zero, not just
    # small -- pinned before the phase-agreement check below.
    for idx_boost in (5, 6, 7):  # IDX_E01, IDX_E02, IDX_E03
        assert float(omega[idx_boost]) == 0.0

    _r_next, dphase_scalar, dphase_rotor = _rotor_step_stark(
        _IDENTITY, e_total, grad_e_total, v, species, dtau
    )
    assert abs(float(dphase_scalar)) > 1e-16  # non-vacuous
    assert pytest.approx(float(dphase_scalar), rel=0, abs=1e-14) == float(dphase_rotor)


# ---------------------------------------------------------------------------
# 4. v=0 static-node check.
# ---------------------------------------------------------------------------


def test_v_zero_static_node_rotor_matches_scalar_exactly() -> None:
    """Zero velocity forces omega_boost to *exactly* zero
    (`boost_coeff = (v/c) * omega_tilde_0k` with `v` identically `0`),
    regardless of field gradient -- the regime every existing
    `coupling.type='stark_dc'` + `integration.mode='worldline'` pipeline
    call site actually runs in (lattice Hermite-Gauss quadrature nodes,
    static positions, `v=0` at every integration step; see
    `cliffordclock.pipeline._stark_rotor_ensemble`'s docstring). Uses the
    *same* deliberately steep gradient as the exaggerated-boost test above
    (`1e18 V/m^2`) specifically to show the null holds even when
    `omega_0k` itself is large -- it is `v`, not the gradient, doing the
    zeroing.
    """
    species = get_species("Sr87")
    e_total = jnp.array([1e10, 0.0, 0.0])
    grad_e_total = jnp.zeros((3, 3)).at[0, 0].set(1e18)
    v = jnp.zeros(3)
    dtau = 0.05

    omega = build_omega_stark(e_total, grad_e_total, species, v)
    for idx_boost in (5, 6, 7):  # IDX_E01, IDX_E02, IDX_E03
        assert float(omega[idx_boost]) == 0.0

    _r_next, dphase_scalar, dphase_rotor = _rotor_step_stark(
        _IDENTITY, e_total, grad_e_total, v, species, dtau
    )
    assert abs(float(dphase_scalar)) > 1e-16  # non-vacuous
    assert pytest.approx(float(dphase_scalar), rel=0, abs=1e-14) == float(dphase_rotor)


def test_v_zero_static_node_rotor_matches_scalar_exactly_with_bbr_active() -> None:
    """WP20 extension of the test above ("extend the WP16 head-to-head test
    with BBR active rather than duplicating it"): the same v=0 static-node
    regime, but with a nonzero `bbr_pivot_perturbation` (CONVENTIONS.md
    E32/E33) composed into both `build_omega_stark` and `_rotor_step_stark`.
    `omega_boost` is still identically zero here (v=0 unconditionally
    zeroes it, per the base test above -- BBR does not change that, since
    BBR's own gradient contribution is exactly zero for uniform T,
    `spin_connection_stark`'s WP20 docstring note), so the rotor and scalar
    phase increments must still agree to the same tight bound as the
    BBR-off case, proving E33's composition does not silently reintroduce
    a rotor<->scalar discrepancy.
    """
    species = get_species("Sr87")
    e_total = jnp.array([1e10, 0.0, 0.0])
    grad_e_total = jnp.zeros((3, 3)).at[0, 0].set(1e18)
    v = jnp.zeros(3)
    dtau = 0.05
    bbr_value = bbr_pivot_perturbation(300.0, species)
    assert bbr_value < 0.0  # non-vacuous: the E32 sign regression, reused as a sanity check

    omega = build_omega_stark(e_total, grad_e_total, species, v, bbr_pivot_perturbation=bbr_value)
    for idx_boost in (5, 6, 7):  # IDX_E01, IDX_E02, IDX_E03
        assert float(omega[idx_boost]) == 0.0

    _r_next, dphase_scalar, dphase_rotor = _rotor_step_stark(
        _IDENTITY, e_total, grad_e_total, v, species, dtau, bbr_pivot_perturbation=bbr_value
    )
    assert abs(float(dphase_scalar)) > 1e-16  # non-vacuous
    np.testing.assert_allclose(float(dphase_rotor), float(dphase_scalar), rtol=0, atol=1e-14)

    # Non-vacuousness / additivity check that BBR actually moved the
    # accumulated phase (not just that rotor==scalar trivially), done at a
    # REALISTIC field magnitude (100 V/m, a lab-typical stray field --
    # matching examples/lattice_sr87_stark.yaml) rather than the deliberately
    # unrealistic 1e10 V/m above. At 1e10 V/m the Stark term itself is
    # non-perturbative (`p_minus_1 ~ -0.72`, order-1, chosen there purely to
    # make omega_0k huge for the boost-null check), so adding a ~1e-15
    # BBR correction to an O(1) sum loses ~all of it to float64 rounding
    # (measured ~0.2% relative error there) -- an intrinsic floating-point
    # fact about adding a tiny correction to an O(1) number, not a bug in
    # the E33 composition. At a realistic field the Stark term is itself
    # perturbative (comparable magnitude to BBR), so the two add exactly;
    # `tests/test_bbr_pivot.py`'s dedicated composition-additivity tests
    # already check this at `atol=0`, so this is a lighter sanity echo of
    # that fact at the rotor-step level, not a duplicate precision claim.
    e_realistic = jnp.array([100.0, 0.0, 0.0])
    _r_realistic, dphase_scalar_with_bbr, _rot = _rotor_step_stark(
        _IDENTITY, e_realistic, grad_e_total, v, species, dtau, bbr_pivot_perturbation=bbr_value
    )
    _r_off, dphase_scalar_off, _rot_off = _rotor_step_stark(
        _IDENTITY, e_realistic, grad_e_total, v, species, dtau
    )
    np.testing.assert_allclose(
        float(dphase_scalar_with_bbr) - float(dphase_scalar_off),
        bbr_value * dtau,
        rtol=1e-9,
        atol=0,
    )


# ---------------------------------------------------------------------------
# 5. Non-symmetric-gradient orientation guard (MAJOR 1, WP16 review).
# ---------------------------------------------------------------------------

# Hand-derived reference values for the case below (see
# test_spin_connection_stark_matches_hand_derived_nonsymmetric_gradient's
# docstring for the full derivation). Recorded once here, at full float64
# precision, as the shared source of truth for both tests in this section.
_NONSYM_E_TOTAL = jnp.array([1.0, 2.0, 3.0])
_NONSYM_GRAD_E_TOTAL = jnp.array(
    [
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
        [70.0, 80.0, 90.0],
    ]
)
_NONSYM_OMEGA_0K_EXPECTED = jnp.array(
    [-2.007746671012009e-18, -4.5891352480274485e-18, -7.170523825042889e-18]
)


def test_spin_connection_stark_matches_hand_derived_nonsymmetric_gradient() -> None:
    """MAJOR 1 (WP16 review): `spin_connection_stark` against a hand-derived
    reference for a genuinely non-symmetric `grad_e_total`, plus a guard
    that the transposed contraction does *not* match.

    Every existing test of `spin_connection_stark`/`build_omega_stark`
    (groups 1-4 above, and `tests/test_integrator_omega.py`) uses a
    `grad_e_total` that is either exactly zero or has only a single
    diagonal element set. Both are trivially symmetric (`G == G.T`), so a
    transposed-axes bug in the function's `einsum("...kj,...j->...k",
    grad_e_total, e_total)` contraction -- the E13 convention has
    `grad_e_total[..., i, j] = d_i E_j`, so `(E . d_k E) = sum_j E_j *
    grad_e_total[..., k, j]` -- would be completely undetectable by any
    shipped test. This mirrors `tests/test_fields_smoother.py`'s
    `test_grad_orientation_nonsymmetric_gradient` orientation-trap
    pattern: use a genuinely non-symmetric `G` and assert both "matches
    G" and "does not match G.T".

    **Derivation** (E14b/E16, plain math -- see `spin_connection_stark`'s
    own docstring for the general formula this specializes).
    `e_total = E = [1, 2, 3]` V/m, `grad_e_total = G = [[10, 20, 30],
    [40, 50, 60], [70, 80, 90]]` V/m^2 (E13: `G[i, j] = d_i E_j`;
    deliberately non-symmetric, `G[0, 1] = 20 != 40 = G[1, 0]`, etc.).
    Sr87: `Delta_alpha = 4.07873e-39` C.m^2/V (Middelmann et al. 2012,
    the species registry value), `h = 6.62607015e-34` J.s
    (`constants.PLANCK_H`), `nu_0 = 429_228_004_229_873.4` Hz (species
    registry). `k_S = -Delta_alpha / (2h) = -3.077789630705917e-06`
    Hz.m^2.V^-2; `prefactor = k_S / nu_0 = -7.170523825042889e-21`
    (V/m)^-2.

    `d(P-1)/dr_k = 2 * prefactor * (E . d_k E)`, with `(E . d_k E) =
    sum_j E_j * G[k, j]` (E13 convention -- fix the *first* index of `G`
    to `k`, contract the *second* against `E`):

        k=0: 1*10 + 2*20 + 3*30 = 140
        k=1: 1*40 + 2*50 + 3*60 = 320
        k=2: 1*70 + 2*80 + 3*90 = 500

    so `d(P-1)/dr = 2 * prefactor * [140, 320, 500] =
    [-2.007746671012009e-18, -4.5891352480274485e-18,
    -7.170523825042889e-18]`. `P` in the `omega_0k = d(P-1)/dr_k / P`
    denominator: `P - 1 = prefactor * |E|^2 = prefactor * 14 =
    -1.0038733355060045e-19`, so `P` rounds to exactly `1.0` at float64
    precision -- the denominator division is a no-op at this precision,
    and `omega_0k = d(P-1)/dr` to full float64 precision, matching the
    module docstring's own note that `P` is only needed at O(1) relative
    precision here.

    These full-precision values were independently cross-checked by a
    plain-Python (non-JAX) re-derivation of the same formula. The WP16
    reviewer's own independently hand-derived values (`-2.00775e-18,
    -4.58914e-18, -7.17052e-18`, quoted to 6 significant figures) agree
    with the full-precision values below to their stated precision.

    The guard: contracting the *transposed* convention instead
    (`(E . d_k E)_wrong = sum_j E_j * G[j, k]`, i.e. what a swapaxes /
    transposed-contraction regression would compute) gives
    `[300, 360, 420]` for `2*prefactor*(...)` -- a completely different
    vector, qualitatively distinct from the correct one rather than a
    small perturbation of it.
    """
    species = get_species("Sr87")

    omega_0k = spin_connection_stark(_NONSYM_E_TOTAL, _NONSYM_GRAD_E_TOTAL, species)
    np.testing.assert_allclose(
        np.asarray(omega_0k), np.asarray(_NONSYM_OMEGA_0K_EXPECTED), rtol=1e-10, atol=0.0
    )

    # Orientation guard: the transposed contraction must NOT match --
    # mirrors test_fields_smoother.py's orientation-trap pattern. The
    # divergence is O(1) relative (not float64 rounding), confirming
    # this is a genuine orientation trap rather than noise.
    omega_0k_transposed_grad = spin_connection_stark(
        _NONSYM_E_TOTAL, _NONSYM_GRAD_E_TOTAL.T, species
    )
    rel_diff = np.abs(
        (np.asarray(omega_0k_transposed_grad) - np.asarray(_NONSYM_OMEGA_0K_EXPECTED))
        / np.asarray(_NONSYM_OMEGA_0K_EXPECTED)
    )
    assert np.all(rel_diff > 0.1), (
        f"transposed-gradient contraction too close to the correct answer "
        f"(rel diff {rel_diff}) -- guard would not catch a transposed-axes regression"
    )


def test_build_omega_stark_boost_components_carry_nonsymmetric_gradient() -> None:
    """MAJOR 1 (WP16 review), extended to `build_omega_stark`: the
    boost-blade components (`e_k ^ e_0`, `IDX_E01/E02/E03`) must carry the
    same non-symmetric `spin_connection_stark` values through the
    `(v^k/c) * lambda_bar_C` scaling (E18) -- not just
    `spin_connection_stark` in isolation.

    Uses the same non-symmetric `(E, grad_E)` case as
    `test_spin_connection_stark_matches_hand_derived_nonsymmetric_gradient`,
    with a velocity that has all three components nonzero
    (`v = (0.1, 0.2, 0.3) * c`) so all three boost components are
    exercised at once (unlike groups 1-4 above, which only ever use a
    single-axis `v`).

    **Derivation.** `boost_coeff_k = (v_k / c) * lambda_bar_C *
    omega_0k`, and `build_omega_stark` stores `IDX_E0k = -boost_coeff_k`
    (see its body: `omega.at[..., IDX_E0k].set(-boost_coeff[..., k])`).
    With `lambda_bar_C = 3.8615926719863036e-13` m
    (`constants.LAMBDA_BAR_COMPTON`) and the hand-derived `omega_0k` from
    the sibling test:

        IDX_E01 = -(0.1 * lambda_bar_C * omega_0k[0]) = 7.753099831984871e-32
        IDX_E02 = -(0.2 * lambda_bar_C * omega_0k[1]) = 3.544274208907369e-31
        IDX_E03 = -(0.3 * lambda_bar_C * omega_0k[2]) = 8.306892677126646e-31

    (independently cross-checked against the actual `build_omega_stark`
    output while deriving this test: matches to float64 precision.)
    """
    species = get_species("Sr87")
    v = jnp.array([0.1, 0.2, 0.3]) * SPEED_OF_LIGHT

    omega16 = build_omega_stark(_NONSYM_E_TOTAL, _NONSYM_GRAD_E_TOTAL, species, v)

    v_over_c = np.array([0.1, 0.2, 0.3])
    expected_boost = -(v_over_c * LAMBDA_BAR_COMPTON * np.asarray(_NONSYM_OMEGA_0K_EXPECTED))
    actual_boost = np.asarray(
        [float(omega16[IDX_E01]), float(omega16[IDX_E02]), float(omega16[IDX_E03])]
    )
    np.testing.assert_allclose(actual_boost, expected_boost, rtol=1e-9, atol=0.0)

    # Orientation guard, propagated through to the assembled bivector:
    # the transposed-gradient boost components must differ by O(1)
    # relative, not just float64 rounding.
    omega16_transposed_grad = build_omega_stark(_NONSYM_E_TOTAL, _NONSYM_GRAD_E_TOTAL.T, species, v)
    actual_boost_transposed = np.asarray(
        [
            float(omega16_transposed_grad[IDX_E01]),
            float(omega16_transposed_grad[IDX_E02]),
            float(omega16_transposed_grad[IDX_E03]),
        ]
    )
    rel_diff = np.abs((actual_boost_transposed - expected_boost) / expected_boost)
    assert np.all(rel_diff > 0.1), (
        f"transposed-gradient boost components too close to the correct answer "
        f"(rel diff {rel_diff}) -- guard would not catch a transposed-axes regression"
    )


# ---------------------------------------------------------------------------
# 6. `_stark_rotor_ensemble` time-varying-Omega convergence (MAJOR 2, WP16
# review). Genuinely time-varying field along a genuinely moving
# trajectory -- the pipeline's one production call site for this function
# (`integration.mode='worldline'`, `coupling.type='stark_dc'`, lattice
# quadrature nodes) always passes a *static* trajectory (v=0 every step),
# so this coverage can only be reached by calling the accumulator
# directly with a hand-built trajectory, exactly as this file's other
# sections call `build_omega_stark` directly.
# ---------------------------------------------------------------------------

_MAJOR2_E0 = 1.0e8  # V/m, deliberately large synthetic field (see docstring)
_MAJOR2_V_X = 0.05 * SPEED_OF_LIGHT  # m/s, constant velocity along x
_MAJOR2_T_TAU = 5.0  # total tau~ (Compton units, E9) integrated over
# x_typ set to half the trajectory's total x-displacement at the coarsest
# resolution tested, so E_z's k1*x/k2*x^2 terms are an O(1) fraction of
# E_z(0) over the run (a genuinely time-varying field, not a barely-
# perturbed one) -- see the first test's docstring for the numeric check.
_MAJOR2_X_TYP = (_MAJOR2_V_X * _MAJOR2_T_TAU * TAU_COMPTON) / 2.0
_MAJOR2_K1 = 1.0 / _MAJOR2_X_TYP  # 1/m
_MAJOR2_K2 = 1.0 / _MAJOR2_X_TYP**2  # 1/m^2


def _major2_field_fn() -> CombinedFieldFn:
    """``E(r) = (0, 0, E0*(1 + k1*x + k2*x^2))`` -- a field that varies
    only along x, quadratically, mirroring the WP16 review's polynomial-
    field template. Wrapped via `as_field_fn` into the combined
    ``pos -> (E, grad_E)`` convention `_stark_rotor_ensemble` expects.
    """

    def e_fn(pos: jnp.ndarray) -> jnp.ndarray:
        x = pos[:, 0]
        e_z = _MAJOR2_E0 * (1.0 + _MAJOR2_K1 * x + _MAJOR2_K2 * x**2)
        zeros = jnp.zeros_like(x)
        return jnp.stack([zeros, zeros, e_z], axis=-1)

    def grad_fn(pos: jnp.ndarray) -> jnp.ndarray:
        x = pos[:, 0]
        d_ez_dx = _MAJOR2_E0 * (_MAJOR2_K1 + 2.0 * _MAJOR2_K2 * x)
        grad = jnp.zeros((x.shape[0], 3, 3), dtype=jnp.float64)
        return grad.at[:, 0, 2].set(d_ez_dx)  # E13: grad[i, j] = d_i E_j -> grad[0, 2] = d_x E_z

    return as_field_fn(e_fn, grad_fn)


def _major2_trajectory(n_steps: int) -> tuple[jnp.ndarray, float]:
    """Constant-velocity trajectory along x, sampled at `n_steps + 1`
    equally-spaced tau~ points covering `[0, _MAJOR2_T_TAU]`. Positions
    are exactly affine in tau~ (`x = v_x * t_phys(tau~)`), so sampling at
    any resolution is exact -- no trajectory-discretization error is
    introduced, meaning any measured convergence error below is purely
    the rotor stepper's own quadrature error, not a sampling artifact.
    """
    dtau = _MAJOR2_T_TAU / n_steps
    dt_phys = dtau * TAU_COMPTON
    t_idx = jnp.arange(n_steps + 1, dtype=jnp.float64)
    x = _MAJOR2_V_X * (t_idx * dt_phys)
    zeros = jnp.zeros_like(x)
    trajectory = jnp.stack([x, zeros, zeros], axis=-1)[None, :, :]  # (M=1, T, 3)
    return trajectory, dtau


_MAJOR2_STEP_COUNTS = [16, 32, 64, 128, 256, 512]  # 5 doublings


def test_stark_rotor_ensemble_time_varying_omega_converges_at_second_order() -> None:
    """MAJOR 2 (WP16 review): `_stark_rotor_ensemble` driven by a
    genuinely time-varying Omega (quadratic field along a moving
    trajectory) shows clean step-halving O(dtau~^2) convergence -- proof
    that the accumulator genuinely evaluates the field at the step
    *midpoint* each iteration (E19 exponential-midpoint), not e.g. at a
    step endpoint (which would only be first-order accurate; see
    `test_stark_rotor_ensemble_endpoint_evaluation_would_fail_convergence`
    below for the explicit discriminating comparison).

    **Why no closed form is available, and how error is measured
    instead.** Unlike `test_integrator_stepper.py`'s
    `test_convergence_order_matches_design_order` (E14a, which has a
    closed-form exact phase because its Omega stays in a single fixed
    bivector plane), this case's Omega direction itself changes across
    the run (`build_omega_stark`'s boost-plane components turn on and
    off with the field-driven `omega_0k`, on top of the rotation-plane
    component varying with `E_z(x)^2`), so no simple closed form exists.
    Self-convergence (Richardson-style step doubling) is used instead:
    for a scheme with true global error `phase(N) = phase_exact + C*h^2 +
    O(h^3)` (`h = T/N`), the *consecutive-difference* sequence
    `e_N = |phase(N) - phase(2N)|` satisfies `e_N = (3/4)*C*h^2 +
    O(h^3)`, so successive ratios `e_N / e_{2N} -> 4` as `N -> infinity`
    -- the same discriminating signature a closed-form check would give,
    without needing one.

    **Parameters** (documented, not tuned to a result -- chosen by the
    reasoning in each comment). `species=Sr87`. Field magnitude
    `E0=1e8` V/m: deliberately large and unrealistic (same spirit as
    this file's own exaggerated-boost test and `test_integrator_stepper.py`'s
    `test_e24_second_order_divergence_scales_quadratically_with_boost`),
    chosen so the Stark-driven part of Omega's time variation clears the
    float64 rounding floor by many orders of magnitude -- at a
    physically realistic `E0~10 V/m` the Stark contribution to the phase
    is drowned out by the (exactly, hence error-free at any resolution)
    linear-in-tau kinematic term from the constant `v_x`, since a
    *constant*-velocity trajectory's kinematic contribution to Omega
    never varies and midpoint quadrature integrates it exactly
    regardless of step count. `v_x = 0.05c`: large enough for a
    non-degenerate moving trajectory, non-relativistic enough to stay
    inside E21's kinematic-term algebra. `T_tau~ = 5.0`: total run
    length; combined with `v_x` this gives a total x-displacement of
    `~9.65e-14` m at the coarsest resolution tested, and `_MAJOR2_X_TYP`
    is set to half of that so the field's `k1*x + k2*x^2` terms are an
    O(1) fraction of `E_z(0)` by the end of the run (measured: `E_z`
    ranges from `E0` at `x=0` to `~2.4*E0` at the trajectory's end) --
    genuinely time-varying, not a barely-perturbed uniform field.

    **Measured** (this environment, float64): step counts `[16, 32, 64,
    128, 256, 512]` give consecutive-difference ratios
    `[3.99879, 3.99970, 3.99992, 3.99998]` -- all within `0.0013` of the
    ideal `4.0`, tightening monotonically toward `4.0` as expected of a
    genuine leading-order `O(h^2)` asymptotic. The `3.9 < ratio < 4.1`
    window below gives >60x margin around the measured spread while
    remaining tight enough to exclude `~2` (first-order) convergence by
    a wide margin -- see the sibling endpoint-comparison test for what
    that failure mode actually measures.
    """
    species = get_species("Sr87")
    field_fn = _major2_field_fn()

    phases = []
    for n_steps in _MAJOR2_STEP_COUNTS:
        trajectory, dtau = _major2_trajectory(n_steps)
        result = _stark_rotor_ensemble(field_fn, species, trajectory, dtau)
        phases.append(float(result.phase[0]))

    phases_arr = np.asarray(phases)
    consecutive_diffs = np.abs(np.diff(phases_arr))
    assert consecutive_diffs[0] > 1e-8, (
        "consecutive-resolution phase differences are at the float64 noise "
        "floor -- the time-varying part of Omega is not actually being "
        "exercised (test is not measuring anything)"
    )
    ratios = consecutive_diffs[:-1] / consecutive_diffs[1:]
    for ratio in ratios:
        assert 3.9 < ratio < 4.1, (
            f"step-halving error ratio {ratio!r} outside the O(dtau~^2) "
            "window [3.9, 4.1] (measured ~3.999-4.000 at these parameters); "
            "expected ~4x-per-doubling from true midpoint evaluation"
        )


def _major2_endpoint_biased_phase(
    species: Species | StarkCoefficients, field_fn: CombinedFieldFn, n_steps: int
) -> float:
    """Reimplements `_stark_rotor_ensemble`'s single-trajectory accumulation
    loop with Omega evaluated at each step's LEFT ENDPOINT (`pos_a`)
    instead of the true midpoint (`pos_mid = 0.5*(pos_a+pos_b)`) --
    deliberately *not* a call into `_stark_rotor_ensemble` itself, since
    the whole point is to simulate what a *different*, only first-order-
    accurate implementation would measure. Same per-step formulas
    (E17/E19/E21) and same trajectory/field/species as
    `test_stark_rotor_ensemble_time_varying_omega_converges_at_second_order`
    otherwise, so the only difference between the two convergence curves
    is midpoint vs. endpoint evaluation.
    """
    trajectory, dtau = _major2_trajectory(n_steps)
    trajectory = trajectory[0]
    dt_phys = dtau * TAU_COMPTON
    r0 = jnp.zeros(16, dtype=jnp.float64).at[IDX_SCALAR].set(1.0)

    def body(
        carry: tuple[jnp.ndarray, jnp.ndarray], xs_t: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        r, phase = carry
        pos_a, pos_b = xs_t
        v = (pos_b - pos_a) / dt_phys
        e_a, grad_a = field_fn(pos_a)  # endpoint, not midpoint -- the bug being simulated
        omega = build_omega_stark(e_a, grad_a, species, v)
        generator = (-0.5 * dtau) * omega
        delta_r = exp_bivector(generator)
        r_next = geometric_product(delta_r, r)
        phase_next = phase + omega[IDX_E12] * dtau
        return (r_next, phase_next), None

    xs = (trajectory[:-1], trajectory[1:])
    (_r_final, phase_final), _ = jax.lax.scan(body, (r0, jnp.asarray(0.0, dtype=jnp.float64)), xs)
    return float(phase_final)


_MAJOR2_ENDPOINT_STEP_COUNTS = [16, 32, 64, 128, 256]  # 3 doublings


def test_stark_rotor_ensemble_endpoint_evaluation_would_fail_convergence() -> None:
    """MAJOR 2 (WP16 review): explicit discriminating comparison -- an
    endpoint-evaluated stand-in (`_major2_endpoint_biased_phase`, same
    field/trajectory/species as the true-midpoint test above) shows
    step-halving ratios near `2`, not `4`, confirming the `[3.9, 4.1]`
    window used above would actually catch a midpoint-vs-endpoint
    regression rather than passing regardless of which one is
    implemented.

    **Why ratio~2 vs ratio~4 discriminates.** Midpoint quadrature is
    exact for any function affine in `tau~` over a step, leaving
    `O(h^2)` global error (E19); left-endpoint (rectangle-rule)
    quadrature is only exact for constants, leaving `O(h)` global error.
    By the same Richardson argument as the true-midpoint test's
    docstring, an `O(h)`-error scheme's consecutive-difference ratio
    `e_N / e_{2N} -> 2`, not `4`. `2` and `4` are far enough apart
    (a factor of 2) that float64 noise or reasonable parameter choices
    cannot confuse the two regimes.

    **Measured** (this environment, float64): step counts `[16, 32, 64,
    128, 256]` give ratios `[1.955, 1.978, 1.989]` -- converging toward
    2 from below, exactly the first-order signature, and nowhere near
    the true implementation's `[3.9, 4.1]` window.
    """
    species = get_species("Sr87")
    field_fn = _major2_field_fn()

    phases = [
        _major2_endpoint_biased_phase(species, field_fn, n_steps)
        for n_steps in _MAJOR2_ENDPOINT_STEP_COUNTS
    ]
    phases_arr = np.asarray(phases)
    consecutive_diffs = np.abs(np.diff(phases_arr))
    assert consecutive_diffs[0] > 1e-6, "endpoint-biased phase differences too small to measure"
    ratios = consecutive_diffs[:-1] / consecutive_diffs[1:]
    for ratio in ratios:
        assert 1.5 < ratio < 3.0, (
            f"endpoint-biased step-halving ratio {ratio!r} expected in the ~2 "
            "(O(dtau~) convergence) band, clear of the true midpoint "
            "implementation's ~4 (O(dtau~^2)) band asserted above -- if this "
            "ratio drifted near 4 the two implementations would no longer be "
            "distinguishable by this test's window"
        )


def test_stark_rotor_ensemble_moving_trajectory_midpoints_differ_from_endpoints() -> None:
    """MAJOR 2 (WP16 review): kills the `pos_a`/`pos_b` degeneracy blind
    spot directly -- for the moving trajectory used above, `pos_mid =
    0.5*(pos_a+pos_b)` is neither `pos_a` nor `pos_b` at any step. This
    is exactly the condition the pipeline's one production call site for
    `_stark_rotor_ensemble` (static lattice nodes, `v=0` always) never
    satisfies, so it is the condition this file's convergence tests above
    need in order to be measuring anything beyond that degenerate case.
    """
    trajectory, _dtau = _major2_trajectory(8)
    trajectory = trajectory[0]
    pos_a = np.asarray(trajectory[:-1])
    pos_b = np.asarray(trajectory[1:])
    pos_mid = 0.5 * (pos_a + pos_b)

    assert np.all(np.abs(pos_mid[:, 0] - pos_a[:, 0]) > 0.0), "midpoints coincide with pos_a"
    assert np.all(np.abs(pos_mid[:, 0] - pos_b[:, 0]) > 0.0), "midpoints coincide with pos_b"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
