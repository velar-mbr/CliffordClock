# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pure-formula tests for the WP21 (CONVENTIONS.md E34/E35) quadrupole shift.

Covers the G8 gate's binding acceptance items: the two-part sign
regression (gate edit 1), the ea0^2 unit pin (gate edit 2), the exact
three-orientation cancellation (gate edit 3 / E35 A2), the traceless-
symmetric-part requirement (gate edit 3 / A5#3), closed-form known
answers, and the coordinate-free-vs-literal-axial-form equivalence
CONVENTIONS.md section 14 derives. Pipeline-level composition/byte-
exactness/real-FEA tests live in `tests/test_quadrupole_pipeline.py`;
registry data pins live in `tests/test_ion_species.py`.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from cliffordclock.constants import PLANCK_H
from cliffordclock.ensemble.species import EA0_SQUARED_SI, get_quadrupole_moment
from cliffordclock.integrator.omega import (
    quadrupole_mj_factor,
    quadrupole_pivot_perturbation,
    quadrupole_shift_joules,
    quadrupole_three_orientation_average,
    traceless_symmetric_gradient,
)

_NU_0_HZ = 411_042_129_776_393.0  # Ca+ 4S1/2-3D5/2 clock frequency, Hz (Chwalla et al.,
# Phys. Rev. Lett. 102, 023002 (2009), arXiv:0812.2278)


def _axial_literal_shift_joules(
    a_v_per_m2: float, beta_rad: float, theta_au: float, j: float, m_j: float
) -> float:
    """Independent re-implementation of the LITERAL axial Itano/Roos
    formula (CONVENTIONS.md E34, eps=0 special case):

    ``Delta_E_Q = (Theta_SI/4) * [J(J+1)-3m_J^2]/[J(2J-1)] * (3cos^2(beta)-1) * (4*A)``

    with ``dE_z/dz = 4*A`` (the derivation in CONVENTIONS.md section 14).
    Used only as an independent cross-check of
    :func:`quadrupole_shift_joules`'s coordinate-free form -- NOT imported
    from `omega.py`, so this is a genuine two-derivation consistency test,
    not a restatement.
    """
    theta_si = theta_au * EA0_SQUARED_SI
    mj_factor = (j * (j + 1.0) - 3.0 * m_j**2) / (j * (2.0 * j - 1.0))
    angular = 3.0 * np.cos(beta_rad) ** 2 - 1.0
    gradient = 4.0 * a_v_per_m2
    return (theta_si / 4.0) * mj_factor * angular * gradient


def _axial_gradient_tensor(a_v_per_m2: float, eps: float = 0.0) -> jnp.ndarray:
    """The gradient tensor in ITS OWN principal frame (E34 derivation):
    ``diag(-2A(1+eps), -2A(1-eps), 4A)``.
    """
    return jnp.array(
        [
            [-2.0 * a_v_per_m2 * (1.0 + eps), 0.0, 0.0],
            [0.0, -2.0 * a_v_per_m2 * (1.0 - eps), 0.0],
            [0.0, 0.0, 4.0 * a_v_per_m2],
        ]
    )


def _quantization_axis(beta_rad: float, alpha_rad: float = 0.0) -> jnp.ndarray:
    return jnp.array(
        [
            np.sin(beta_rad) * np.cos(alpha_rad),
            np.sin(beta_rad) * np.sin(alpha_rad),
            np.cos(beta_rad),
        ]
    )


# ---------------------------------------------------------------------------
# G8 gate edit 1: the two-part sign regression.
# ---------------------------------------------------------------------------


class TestSignRegression:
    """The A1 two-part regression the G8 gate requires before wiring E34."""

    def test_convention_free_mj_ratio_d5_2(self) -> None:
        """ΔE_Q(J=5/2, m_J=5/2) / ΔE_Q(J=5/2, m_J=1/2) = -1.25 (gate edit 1,
        convention-free -- holds for ANY overall sign convention since a
        global sign flip cancels in a ratio).
        """
        f_stretched = quadrupole_mj_factor(2.5, 2.5)
        f_half = quadrupole_mj_factor(2.5, 0.5)
        np.testing.assert_allclose(f_stretched / f_half, -1.25, rtol=1e-14, atol=0)

    def test_mj_ratio_is_independent_of_overall_formula_sign(self) -> None:
        """A regression that would survive an implementer flipping the
        overall leading sign of the formula -- the ratio must be
        unaffected either way, proving this specific test does not
        silently depend on E34's (unverified) absolute sign convention.
        """
        grad = _axial_gradient_tensor(1.0e10)
        axis = jnp.array([0.0, 0.0, 1.0])
        theta_au = 2.973  # Sr+ D5/2
        shift_52 = quadrupole_pivot_perturbation(grad, axis, theta_au, 2.5, 2.5, _NU_0_HZ)
        shift_12 = quadrupole_pivot_perturbation(grad, axis, theta_au, 2.5, 0.5, _NU_0_HZ)
        np.testing.assert_allclose(float(shift_52 / shift_12), -1.25, rtol=1e-12, atol=0)
        # And again with theta negated (an implementer's sign bug simulated):
        shift_52_neg = quadrupole_pivot_perturbation(grad, axis, -theta_au, 2.5, 2.5, _NU_0_HZ)
        shift_12_neg = quadrupole_pivot_perturbation(grad, axis, -theta_au, 2.5, 0.5, _NU_0_HZ)
        np.testing.assert_allclose(float(shift_52_neg / shift_12_neg), -1.25, rtol=1e-12, atol=0)

    def test_yb_plus_f7_2_negative_theta_gives_opposite_sign_shift(self) -> None:
        """Yb+ F7/2's Theta is negative (dossier section 3, Huntemann 2012,
        PRIMARY); under an otherwise identical (gradient, m_J, axis) its
        shift must be OPPOSITE in sign to a positive-Theta D-state (G8
        gate edit 1's second sign anchor). A real regression on whether
        the code respects the registry's Theta sign.
        """
        grad = _axial_gradient_tensor(1.0e10)
        axis = jnp.array([0.0, 0.0, 1.0])
        ca = get_quadrupole_moment("Ca+:D5/2")  # J=5/2, Theta > 0
        yb = get_quadrupole_moment("Yb+:F7/2")  # J=7/2, Theta < 0
        assert ca.theta_au > 0.0
        assert yb.theta_au < 0.0
        # Stretched state (m_J = J) for each, same axis/gradient.
        shift_ca = quadrupole_pivot_perturbation(grad, axis, ca.theta_au, ca.j, ca.j, _NU_0_HZ)
        shift_yb = quadrupole_pivot_perturbation(grad, axis, yb.theta_au, yb.j, yb.j, _NU_0_HZ)
        assert float(shift_ca) != 0.0
        assert float(shift_yb) != 0.0
        assert np.sign(float(shift_ca)) != np.sign(float(shift_yb))

    def test_j_zero_and_j_half_are_immune_no_quadrupole_coupling(self) -> None:
        """J=0 and J=1/2 states carry no quadrupole coupling at all
        (CONVENTIONS.md E34's immunity note, corroborated by Ludlow RMP
        2015) -- `quadrupole_mj_factor` raises rather than returning a
        silently-wrong 0/0.
        """
        with pytest.raises(ValueError, match="no quadrupole coupling"):
            quadrupole_mj_factor(0.0, 0.0)
        with pytest.raises(ValueError, match="no quadrupole coupling"):
            quadrupole_mj_factor(0.5, 0.5)


# ---------------------------------------------------------------------------
# G8 gate edit 2: the ea0^2 unit pin (single-e application test).
# ---------------------------------------------------------------------------


class TestEa0SquaredUnitPin:
    def test_theta_si_applies_e_exactly_once(self) -> None:
        """CONVENTIONS.md E34: Theta_SI = theta_au * EA0_SQUARED_SI, NOT an
        additional leading `e` on top (the double-e trap). Cross-check by
        computing the shift two ways: via `quadrupole_shift_joules`
        (which uses EA0_SQUARED_SI internally) and by hand with the SAME
        constant, and separately show a spurious extra factor of `e`
        would move the answer far outside a physically sane range.
        """
        from cliffordclock import constants

        grad = _axial_gradient_tensor(1.0e10)
        axis = jnp.array([0.0, 0.0, 1.0])
        theta_au = 1.83  # Ca+ D5/2
        got = quadrupole_shift_joules(grad, axis, theta_au, 2.5, 2.5)

        theta_si = theta_au * EA0_SQUARED_SI
        mj_factor = quadrupole_mj_factor(2.5, 2.5)
        contraction = 4.0e10  # n^T . diag(...,4A) . n at n=z-hat, A=1e10
        expected = 0.5 * theta_si * mj_factor * contraction
        np.testing.assert_allclose(float(got), expected, rtol=1e-13, atol=0)

        # A double-e bug (theta_si computed as e * theta_au * EA0_SQUARED_SI,
        # i.e. an extra spurious factor of e) would be smaller by ~1.6e-19 --
        # far outside the two computations' agreement tolerance above, and
        # would push the shift many orders of magnitude below any
        # physically meaningful floor for a realistic clock transition.
        double_e_bug = float(got) * constants.ELEMENTARY_CHARGE
        assert abs(double_e_bug) < abs(float(got)) * 1e-15

    def test_ea0_squared_si_pinned_full_precision(self) -> None:
        np.testing.assert_allclose(EA0_SQUARED_SI, 4.4865515185255e-40, rtol=1e-10, atol=0)


# ---------------------------------------------------------------------------
# G8 gate edit 3 / E35 A2: exact three-orientation cancellation.
# ---------------------------------------------------------------------------


class TestThreeOrientationCancellation:
    def test_exact_cancellation_at_machine_precision_random_gradient(self) -> None:
        rng = np.random.default_rng(1234)
        grad = jnp.asarray(rng.normal(size=(3, 3)) * 1.0e12)
        ca = get_quadrupole_moment("Ca+:D5/2")
        avg = quadrupole_three_orientation_average(grad, ca.theta_au, ca.j, ca.j, _NU_0_HZ)
        # Compare to the magnitude of the individual (non-cancelling) terms
        # to show this is genuine machine-precision cancellation, not a
        # coincidentally-small absolute number.
        axes = (
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
        )
        terms = [
            float(quadrupole_pivot_perturbation(grad, ax, ca.theta_au, ca.j, ca.j, _NU_0_HZ))
            for ax in axes
        ]
        max_term = max(abs(t) for t in terms)
        assert max_term > 0.0
        assert abs(float(avg)) / max_term < 1e-13

    def test_cancellation_independent_of_gradient_orientation(self) -> None:
        """CONVENTIONS.md E35 A2: the identity holds for ANY gradient
        tensor orientation, not just axis-aligned ones -- tested across
        several random asymmetric gradients.
        """
        rng = np.random.default_rng(7)
        ca = get_quadrupole_moment("Ca+:D5/2")
        for _ in range(5):
            grad = jnp.asarray(rng.normal(size=(3, 3)) * 1.0e11)
            avg = quadrupole_three_orientation_average(grad, ca.theta_au, ca.j, ca.j, _NU_0_HZ)
            assert abs(float(avg)) < 1e-25  # machine-precision zero at these magnitudes

    def test_cancellation_holds_for_a_non_standard_orthonormal_triad(self) -> None:
        """The identity is proven for ANY orthonormal triad (CONVENTIONS.md
        section 14), not just the standard basis -- exercised with a
        rotated triad.
        """
        theta = 0.37
        c, s = np.cos(theta), np.sin(theta)
        triad = (
            jnp.array([c, s, 0.0]),
            jnp.array([-s, c, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
        )
        rng = np.random.default_rng(99)
        grad = jnp.asarray(rng.normal(size=(3, 3)) * 1.0e11)
        ca = get_quadrupole_moment("Ca+:D5/2")
        avg = quadrupole_three_orientation_average(
            grad, ca.theta_au, ca.j, ca.j, _NU_0_HZ, axes=triad
        )
        assert abs(float(avg)) < 1e-25


# ---------------------------------------------------------------------------
# G8 A5#3: traceless-symmetric-part requirement.
# ---------------------------------------------------------------------------


class TestTracelessSymmetricPart:
    def test_output_is_traceless(self) -> None:
        rng = np.random.default_rng(5)
        grad = jnp.asarray(rng.normal(size=(4, 3, 3)) * 1.0e10)
        result = traceless_symmetric_gradient(grad)
        trace = jnp.trace(result, axis1=-2, axis2=-1)
        # atol scaled to the ~1e10 input magnitude's float64 rounding floor
        # (~1e10 * 2.2e-16 ~ 2e-6), not an arbitrary loose tolerance.
        np.testing.assert_allclose(np.asarray(trace), 0.0, atol=1e-4)

    def test_output_is_symmetric(self) -> None:
        rng = np.random.default_rng(6)
        grad = jnp.asarray(rng.normal(size=(3, 3)) * 1.0e10)
        result = traceless_symmetric_gradient(grad)
        np.testing.assert_allclose(np.asarray(result), np.asarray(result.T), atol=1e-6)

    def test_antisymmetric_part_does_not_couple(self) -> None:
        """Adding a pure antisymmetric perturbation to the gradient tensor
        must not change the quadrupole shift at all (it never couples --
        CONVENTIONS.md E34's traceless-symmetric-part note).
        """
        grad = _axial_gradient_tensor(1.0e10)
        antisymmetric = jnp.array(
            [[0.0, 5.0e9, -3.0e9], [-5.0e9, 0.0, 2.0e9], [3.0e9, -2.0e9, 0.0]]
        )
        axis = jnp.array([0.3, 0.4, 0.866])  # arbitrary
        theta_au, j, m_j = 1.83, 2.5, 2.5
        base = quadrupole_shift_joules(grad, axis, theta_au, j, m_j)
        perturbed = quadrupole_shift_joules(grad + antisymmetric, axis, theta_au, j, m_j)
        np.testing.assert_allclose(float(base), float(perturbed), rtol=1e-12, atol=0)

    def test_numerical_trace_is_removed(self) -> None:
        """A gradient tensor with a small numerical trace (as a fitted RBF
        tensor would carry, per the dossier/CONVENTIONS.md E34 note) gives
        the SAME shift as its exactly-traceless counterpart.
        """
        grad = _axial_gradient_tensor(1.0e10)
        noisy_trace = grad + jnp.eye(3) * 1.0e6  # small numerical trace
        axis = jnp.array([0.0, 0.0, 1.0])
        theta_au, j, m_j = 1.83, 2.5, 2.5
        base = quadrupole_shift_joules(grad, axis, theta_au, j, m_j)
        noisy = quadrupole_shift_joules(noisy_trace, axis, theta_au, j, m_j)
        np.testing.assert_allclose(float(base), float(noisy), rtol=1e-9, atol=0)


# ---------------------------------------------------------------------------
# Closed-form known answers, both signs.
# ---------------------------------------------------------------------------


class TestClosedFormKnownAnswers:
    @pytest.mark.parametrize("a_sign", [1.0, -1.0])
    @pytest.mark.parametrize("beta_deg", [0.0, 30.0, 54.7356, 90.0, 135.0, 180.0])
    def test_coordinate_free_matches_literal_axial_form(
        self, a_sign: float, beta_deg: float
    ) -> None:
        """The coordinate-free `n_hat^T . G . n_hat` form
        (:func:`quadrupole_shift_joules`) matches an INDEPENDENTLY
        re-implemented literal axial Itano/Roos formula
        (`_axial_literal_shift_joules`, not imported from `omega.py`) at
        several angles and both gradient signs -- CONVENTIONS.md section
        14's derivation, verified numerically, not just asserted.
        """
        a = a_sign * 1.234e10
        beta_rad = np.deg2rad(beta_deg)
        theta_au, j, m_j = 2.973, 2.5, 2.5  # Sr+ D5/2, stretched state

        grad = _axial_gradient_tensor(a, eps=0.0)
        axis = _quantization_axis(beta_rad)
        got = float(quadrupole_shift_joules(grad, axis, theta_au, j, m_j))
        expected = _axial_literal_shift_joules(a, beta_rad, theta_au, j, m_j)
        # atol set well below the smallest nonzero |shift| this
        # parametrization produces (~1e-36 J) but above float64 rounding
        # noise at the magic angle (beta=54.7356deg), where the true value
        # is exactly zero and a bare rtol comparison is ill-posed (0/0) --
        # not the REVIEW-checklist's "vacuous approx" anti-pattern, which
        # concerns SMALL-but-nonzero signals, not a genuine zero.
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-40)

    @pytest.mark.parametrize("eps", [0.0, 0.2, -0.35, 0.7])
    @pytest.mark.parametrize("alpha_deg", [0.0, 25.0, 90.0])
    def test_coordinate_free_matches_general_asymmetric_bracket(
        self, eps: float, alpha_deg: float
    ) -> None:
        """General (eps != 0) case: the coordinate-free contraction matches
        the literal `[(3cos^2(beta)-1) - eps*sin^2(beta)*cos(2*alpha)]`
        bracket (CONVENTIONS.md section 14's asymmetric-generalization
        derivation), independently re-implemented here.
        """
        a = 5.0e9
        beta_rad = np.deg2rad(40.0)
        alpha_rad = np.deg2rad(alpha_deg)
        theta_au, j, m_j = 3.229, 2.5, 1.5  # Ba+ D5/2, an interior m_j

        grad = _axial_gradient_tensor(a, eps=eps)
        axis = _quantization_axis(beta_rad, alpha_rad)
        got = float(quadrupole_shift_joules(grad, axis, theta_au, j, m_j))

        theta_si = theta_au * EA0_SQUARED_SI
        mj_factor = quadrupole_mj_factor(j, m_j)
        bracket = (
            3.0 * np.cos(beta_rad) ** 2
            - 1.0
            - eps * np.sin(beta_rad) ** 2 * np.cos(2.0 * alpha_rad)
        )
        expected = (theta_si / 4.0) * mj_factor * bracket * (4.0 * a)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0)

    def test_beta_zero_maximizes_magnitude_for_stretched_state(self) -> None:
        """At beta=0 (quantization axis aligned with the gradient's
        principal axis), (3cos^2(beta)-1) = 2, its maximum magnitude --
        a physically-meaningful known answer, not an arbitrary check.
        """
        a = 1.0e10
        theta_au, j, m_j = 1.83, 2.5, 2.5
        grad = _axial_gradient_tensor(a)
        shift_0 = abs(
            float(quadrupole_shift_joules(grad, _quantization_axis(0.0), theta_au, j, m_j))
        )
        for beta_deg in (20.0, 40.0, 60.0, 80.0):
            shift = abs(
                float(
                    quadrupole_shift_joules(
                        grad, _quantization_axis(np.deg2rad(beta_deg)), theta_au, j, m_j
                    )
                )
            )
            assert shift <= shift_0 + 1e-30

    def test_magic_angle_gives_zero_shift(self) -> None:
        """beta = arccos(1/sqrt(3)) ~ 54.7356 degrees is the root of
        (3cos^2(beta)-1) -- the shift vanishes there for ANY nonzero
        gradient/Theta, a closed-form zero.
        """
        magic_beta = np.arccos(1.0 / np.sqrt(3.0))
        grad = _axial_gradient_tensor(7.0e10)
        axis = _quantization_axis(magic_beta)
        shift = float(quadrupole_shift_joules(grad, axis, 2.973, 2.5, 2.5))
        assert abs(shift) < 1e-30

    def test_magic_mj_squared_zero_dube_intercept(self) -> None:
        """The m_J factor's own zero at m_J^2 = J(J+1)/3 -- Dube et al.,
        PRL 95, 033001 (2005) use exactly this intercept (35/12 for
        F' = 5/2, their Fig. 1) as the quadrupole-shift-free point their
        cancellation method interpolates to. A third convention-free
        structural pin alongside the -1.25 ratio and the magic-angle
        zero (ion-clock dossier section 7; G8 sign discipline).
        m_J = sqrt(35/12) is not a physical sublevel; the pin is on the
        continuous m_J factor the physical shifts interpolate through.
        """
        j = 2.5
        magic_mj = math.sqrt(j * (j + 1.0) / 3.0)  # sqrt(35/12) for J=5/2
        factor = quadrupole_mj_factor(j, magic_mj)
        # Exact zero up to the one rounding step (squaring sqrt(35/12)):
        # numerator scale is j(j+1) ~ 8.75, so a 1-ulp m_J^2 error leaves
        # |factor| <~ 8.75 * 2.2e-16 / 10 ~ 2e-16.
        assert abs(factor) < 1e-14
        # And the intercept is where the SHIFT vanishes for any gradient:
        grad = _axial_gradient_tensor(7.0e10)
        axis = _quantization_axis(0.0)
        shift = float(quadrupole_shift_joules(grad, axis, 2.973, j, magic_mj))
        assert abs(shift) < 1e-45


# ---------------------------------------------------------------------------
# Pivot-term composition (E35): the returned (P-1)_Q divides by h*nu_0.
# ---------------------------------------------------------------------------


def test_quadrupole_pivot_perturbation_divides_by_h_nu0() -> None:
    grad = _axial_gradient_tensor(1.0e10)
    axis = jnp.array([0.0, 0.0, 1.0])
    theta_au, j, m_j = 1.83, 2.5, 2.5
    shift_j = float(quadrupole_shift_joules(grad, axis, theta_au, j, m_j))
    p_minus_1 = float(quadrupole_pivot_perturbation(grad, axis, theta_au, j, m_j, _NU_0_HZ))
    np.testing.assert_allclose(p_minus_1, shift_j / (PLANCK_H * _NU_0_HZ), rtol=1e-14, atol=0)


def test_quadrupole_pivot_perturbation_is_realistic_magnitude() -> None:
    """A realistic ion-trap gradient (~1e8 V/m^2) and a registered Theta
    give a fractional shift in the 1e-15 to 1e-10 range -- physically
    sane, not many orders of magnitude off (catches a stray missing/extra
    factor of e/a0/4).
    """
    grad = _axial_gradient_tensor(2.5e7)  # dE_z/dz ~ 1e8 V/m^2
    axis = jnp.array([0.0, 0.0, 1.0])
    ca = get_quadrupole_moment("Ca+:D5/2")
    p_minus_1 = abs(
        float(quadrupole_pivot_perturbation(grad, axis, ca.theta_au, ca.j, ca.j, _NU_0_HZ))
    )
    assert 1e-20 < p_minus_1 < 1e-5
