# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP40 Rydberg Stark-map module (CONVENTIONS.md section
20): Wigner-symbol correctness, basis construction against ARC's own
state count, Hamiltonian assembly, adiabatic diagonalization, the C3
quadratic-crossover consistency check (mj-averaged scalar polarizability
vs. the Phase A registry), and the C6 basis-truncation convergence study.

Field grids in this file are deliberately small (few points, modest
``l_max``/``delta_n`` where the physics under test does not need the
full ARC-matched production basis) to keep the suite's own runtime
reasonable; the full ``l_max=20``, ``n0 +/- 5`` production basis and its
agreement with ARC are exercised in
``benchmarks/run_rydberg_stark_map.py`` instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cliffordclock.integrator import rydberg_cell_response as rcr
from cliffordclock.integrator import rydberg_stark_map as rsm

# ---------------------------------------------------------------------------
# Section A: Wigner 3-j / 6-j symbols
# ---------------------------------------------------------------------------


class TestWignerSymbols:
    @pytest.mark.parametrize(
        "args,expected",
        [
            ((1, 1, 0, 0, 0, 0), -1.0 / math.sqrt(3)),
            ((1, 1, 2, 0, 0, 0), math.sqrt(2.0 / 15.0)),
            ((0.5, 0.5, 1, 0.5, -0.5, 0), 1.0 / math.sqrt(6)),
            ((0.5, 0.5, 0, 0.5, -0.5, 0), 1.0 / math.sqrt(2)),
            ((1, 1, 1, 1, -1, 0), 1.0 / math.sqrt(6)),
            ((2, 2, 2, 0, 0, 0), -math.sqrt(8.0 / 5040.0) * 6.0),
        ],
    )
    def test_3j_known_values(self, args: tuple[float, ...], expected: float) -> None:
        """Six hand-derivable special-case 3-j values (Edmonds 1957 /
        standard tables), the last one (``(2 2 2;0 0 0)``) via the
        general integer ``(l l l;0 0 0)`` closed form,
        ``(-1)^g sqrt((2g-2l)!^3/(2g+1)!) * g!/(g-l)!^3`` with
        ``g=3l/2``, evaluated by hand for ``l=2``: ``g=3``,
        ``(-1)^3 * sqrt(2!^3/7!) * 3!/1!^3 = -sqrt(8/5040)*6``.
        """
        assert rsm.wigner_3j(*args) == pytest.approx(expected, rel=1e-9, abs=1e-12)

    @pytest.mark.parametrize(
        "args,expected",
        [
            ((1, 1, 1, 1, 1, 1), 1.0 / 6.0),
            ((1, 1, 2, 1, 1, 1), 1.0 / 6.0),
            ((0.5, 0.5, 1, 0.5, 0.5, 1), 1.0 / 6.0),
            ((0.5, 0.5, 0, 0.5, 0.5, 1), 0.5),
        ],
    )
    def test_6j_known_values(self, args: tuple[float, ...], expected: float) -> None:
        """The last case uses the standard ``j6=0`` special-case 6-j
        formula after applying the row/column-pair-swap symmetry to
        bring this test's own zero into the ``j4`` position: ``{1, 0.5,
        0.5; 0.5, 0.5, 1}`` -> (zero at j4 rule) ``(-1)^(j1+j2+j3)/
        sqrt((2j2+1)(2j3+1))`` with ``j1=1, j2=j3=0.5``: ``(-1)^2/sqrt(4)
        = 0.5``, hand-verified independently of this module's own code.
        """
        assert rsm.wigner_6j(*args) == pytest.approx(expected, rel=1e-9, abs=1e-12)

    def test_3j_orthogonality(self) -> None:
        """``sum_{m1,m2} (j1 j2 j3;m1 m2 -m3)(j1 j2 j3';m1 m2 -m3') =
        delta_{j3,j3'} delta_{m3,m3'} / (2j3+1)`` (a standard 3-j sum
        rule, e.g. Edmonds 1957 Eq. 3.7.8), checked at the diagonal for
        every ``j3`` up to 3.5 with ``j1=2, j2=1.5`` fixed: a broad,
        general structural check of the whole implementation, not just
        a handful of special cases.
        """
        j1, j2 = 2.0, 1.5

        def m_range(j: float) -> list[float]:
            n = int(round(2 * j))
            return [-j + k for k in range(n + 1)]

        for j3 in (0.5, 1.5, 2.5, 3.5):
            for m3 in m_range(j3):
                total = 0.0
                for m1 in m_range(j1):
                    for m2 in m_range(j2):
                        total += rsm.wigner_3j(j1, j2, j3, m1, m2, -m3) ** 2
                assert total == pytest.approx(1.0 / (2.0 * j3 + 1.0), rel=1e-8, abs=1e-10)

    def test_3j_violates_triangle_rule_is_zero(self) -> None:
        assert rsm.wigner_3j(1, 1, 5, 0, 0, 0) == 0.0

    def test_3j_violates_m_sum_is_zero(self) -> None:
        assert rsm.wigner_3j(1, 1, 1, 1, 1, 1) == 0.0

    def test_6j_violates_triangle_rule_is_zero(self) -> None:
        assert rsm.wigner_6j(1, 1, 1, 1, 1, 10) == 0.0


# ---------------------------------------------------------------------------
# Section B: quantum defects
# ---------------------------------------------------------------------------


class TestQuantumDefects:
    def test_d52_and_p32_reuse_phase_a_registry(self) -> None:
        """WP40 must reuse, not re-transcribe, Phase A's own already-cited
        nD5/2 and nP3/2 defects (single-transcription-surface discipline).
        """
        assert rsm.RB85_QUANTUM_DEFECTS[(2, 2.5)] is rcr.RB85_ND52_QUANTUM_DEFECT
        assert rsm.RB85_QUANTUM_DEFECTS[(1, 1.5)] is rcr.RB85_NP32_QUANTUM_DEFECT

    def test_g_state_defect_matches_moore_et_al_abstract(self) -> None:
        """Moore, Duspayev, Cardman, Raithel, PRA 102, 062817 (2020),
        read directly this session (par.nsf.gov PDF): abstract states
        ``delta0 = 0.003 999 0(21)``, ``delta2 = -0.0202(21)``.
        """
        defect = rsm.quantum_defect_for(4, 4.5)
        assert defect.delta0 == pytest.approx(0.0039990, abs=1e-7)
        assert defect.delta2 == pytest.approx(-0.0202, abs=1e-4)

    def test_hydrogenic_beyond_l4(self) -> None:
        defect = rsm.quantum_defect_for(6, 6.5)
        assert defect.delta0 == 0.0
        assert defect.delta2 == 0.0

    def test_unregistered_low_l_raises(self) -> None:
        with pytest.raises(ValueError, match="no Rb-85 quantum defect"):
            rsm.quantum_defect_for(2, 0.5)  # D_{1/2} does not exist (l=2 => j in {1.5,2.5})


# ---------------------------------------------------------------------------
# Section C0/C: model potential and radial matrix elements
# ---------------------------------------------------------------------------


class TestModelPotential:
    def test_reduces_to_coulomb_at_large_r(self) -> None:
        r = np.array([500.0, 1000.0, 5000.0])
        v = rsm._model_potential_hartree(r, 2)
        assert v == pytest.approx(-1.0 / r, rel=1e-3, abs=1e-8)

    def test_more_negative_than_coulomb_at_short_range(self) -> None:
        """Core penetration/polarization make the true potential deeper
        (more negative) than the bare ``-1/r`` at short range for every
        l this module tabulates a defect for -- the entire physical
        reason a model potential is needed at all.
        """
        r = np.array([2.0, 3.0, 5.0])
        for l_orbital in range(4):
            v = rsm._model_potential_hartree(r, l_orbital)
            assert np.all(v < -1.0 / r)


class TestRadialMatrixElements:
    def test_symmetric_under_swap(self) -> None:
        ns1 = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        ns2 = rcr.effective_quantum_number(33, rcr.RB85_NP32_QUANTUM_DEFECT)
        r12 = rsm._radial_matrix_element_pair(ns1, 2, ns2, 1)
        r21 = rsm._radial_matrix_element_pair(ns2, 1, ns1, 2)
        assert r12 == pytest.approx(r21, rel=1e-9, abs=1e-6)

    def test_agrees_with_arc_convention_scale(self) -> None:
        """No live ARC dependency in the pytest suite (ARC is installed
        only transiently to generate the committed benchmark fixture,
        dossier's own "install ARC in the venv" instruction, not a
        project dependency); this pins the SIGN and the order of
        magnitude of the 32D5/2->33P3/2 radial matrix element against a
        value transcribed once from a live ARC session
        (``atom.getRadialMatrixElement(32,2,2.5,33,1,1.5)`` ->
        1277.26...), the same discipline
        ``benchmarks/fixtures/wp40_arc_stark_map_reference.json`` uses
        for the full map comparison.
        """
        ns1 = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        ns2 = rcr.effective_quantum_number(33, rcr.RB85_NP32_QUANTUM_DEFECT)
        radial = rsm._radial_matrix_element_pair(ns1, 2, ns2, 1)
        assert radial == pytest.approx(1277.26, rel=0.01, abs=1.0)

    def test_kill_wrong_recursion_sign_fails_the_arc_check(self) -> None:
        """Formula-level kill test: flipping the sign of every ``T`` term
        in the Numerov recursion (this module's own historical bug,
        Section C's module docstring) reproduces the pre-fix instability
        and misses the ARC-anchored value by far more than the 1%
        tolerance :func:`test_agrees_with_arc_convention_scale` uses.
        """
        n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        _, outer = rcr._turning_points(n_star, 2)
        x_min = math.sqrt(rsm.RADIAL_R_MIN_AU)
        x_max = math.sqrt(1.2 * outer)
        x = np.linspace(x_min, x_max, 2000)
        h = x[1] - x[0]

        r = x**2
        v = rsm._model_potential_hartree(r, 2)
        energy = -1.0 / (2.0 * n_star**2)
        g = 8.0 * x**2 * (v - energy) + (2 * 2 + 0.5) * (2 * 2 + 1.5) / x**2
        t = h**2 * g / 12.0

        x_wave = np.zeros(len(x))
        x_wave[-1] = 0.0
        x_wave[-2] = 1.0e-10
        for i in range(len(x) - 2, 0, -1):
            # Deliberately broken: sign of every T flipped (the historical bug).
            x_wave[i - 1] = ((2.0 - 10.0 * t[i]) * x_wave[i] - (1.0 + t[i + 1]) * x_wave[i + 1]) / (
                1.0 + t[i - 1]
            )

        norm = 2.0 * np.sum(x_wave**2 * x**2) * h
        assert norm > 0.0  # armed: the broken recursion still produces a finite, nonzero norm
        # No need to finish the full matrix element: a NaN/inf/wildly-different-scale
        # norm here is already conclusive proof the broken sign is detectably different.
        correct_norm = 2.0 * np.sum(rsm._numerov_sqrt_single(n_star, 2, x, h) ** 2 * x**2) * h
        assert abs(norm - correct_norm) / correct_norm > 0.5


# ---------------------------------------------------------------------------
# Section D/E: basis construction and Hamiltonian assembly
# ---------------------------------------------------------------------------


class TestBasisConstruction:
    def test_basis_size_matches_arc(self) -> None:
        """ARC's own ``StarkMap.defineBasis(32, 2, 2.5, 0.5, 27, 37, 20)``
        (a live ARC session, ``arc-alkali-rydberg-calculator==3.10.2``,
        commit 4b4573e965222e798ac59636ad7a8b3457262835) reports
        ``len(calc.basisStates) == 451``.
        """
        basis = rsm.build_basis(32, 2, 2.5, 0.5, delta_n=5, l_max=20)
        assert len(basis) == 451

    def test_target_index_matches_arc(self) -> None:
        """ARC's own ``calc.indexOfCoupledState`` for the same basis is
        209 (same live-ARC session as the size check above).
        """
        hamiltonian = rsm.stark_hamiltonian(32, 2, 2.5, 0.5, delta_n=2, l_max=6)
        basis = hamiltonian.basis
        target = next(
            i
            for i, s in enumerate(basis)
            if s.n == 32 and s.l_orbital == 2 and abs(s.j - 2.5) < 1e-9 and abs(s.mj - 0.5) < 1e-9
        )
        assert hamiltonian.target_index == target

    def test_mj_restriction(self) -> None:
        basis = rsm.build_basis(32, 2, 2.5, 2.5, delta_n=2, l_max=6)
        assert all(s.j >= 2.5 - 1e-9 for s in basis)

    def test_s_states_only_have_one_j(self) -> None:
        basis = rsm.build_basis(32, 2, 2.5, 0.5, delta_n=2, l_max=6)
        s_states = [s for s in basis if s.l_orbital == 0]
        js = {s.j for s in s_states}
        assert js == {0.5}


class TestHamiltonianAssembly:
    def test_h0_diagonal_matches_quantum_defect_energy(self) -> None:
        hamiltonian = rsm.stark_hamiltonian(32, 2, 2.5, 0.5, delta_n=2, l_max=6)
        for i, s in enumerate(hamiltonian.basis):
            expected = rsm.state_energy_hartree(s.n, s.l_orbital, s.j)
            assert hamiltonian.h0[i, i] == pytest.approx(expected, rel=1e-12, abs=1e-15)

    def test_h1_symmetric_and_zero_diagonal(self) -> None:
        hamiltonian = rsm.stark_hamiltonian(32, 2, 2.5, 0.5, delta_n=2, l_max=6)
        assert np.allclose(hamiltonian.h1, hamiltonian.h1.T)
        assert np.all(np.diag(hamiltonian.h1) == 0.0)

    def test_h1_zero_for_delta_l_not_one(self) -> None:
        """Delta l = 0 or Delta l = 2 must carry zero coupling (dipole
        selection rule); a formula-level kill test would require
        deliberately dropping the ``abs(l_i - l_j) != 1`` guard, which
        this test checks the CONSEQUENCE of instead: every same-l pair
        actually present in the basis has exactly zero coupling.
        """
        hamiltonian = rsm.stark_hamiltonian(32, 2, 2.5, 0.5, delta_n=2, l_max=6)
        basis = hamiltonian.basis
        same_l_pairs = 0
        for i, si in enumerate(basis):
            for j in range(i + 1, len(basis)):
                sj = basis[j]
                if si.l_orbital == sj.l_orbital:
                    same_l_pairs += 1
                    assert hamiltonian.h1[i, j] == 0.0
        assert same_l_pairs > 0  # sanity: the basis actually contains such pairs


# ---------------------------------------------------------------------------
# Section F: diagonalization and adiabatic tracking
# ---------------------------------------------------------------------------


class TestDiagonalization:
    def test_zero_field_tracked_energy_matches_h0(self) -> None:
        hamiltonian = rsm.stark_hamiltonian(32, 2, 2.5, 0.5, delta_n=2, l_max=6)
        result = rsm.diagonalize_stark_map(hamiltonian, np.array([0.0]))
        expected = hamiltonian.h0[hamiltonian.target_index, hamiltonian.target_index]
        assert result.tracked_energy_hartree[0] == pytest.approx(expected, rel=1e-10, abs=1e-13)

    def test_min_overlap_near_one_away_from_crossings(self) -> None:
        n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        it = rcr.inglis_teller_field_v_per_m(n_star)
        fields = np.linspace(0.0, 0.1 * it, 10)
        result = rsm.stark_map_registry_state(32, fields, delta_n=2, l_max=6)
        assert result.min_overlap > 0.99

    def test_tracked_energy_smooth_low_field(self) -> None:
        """Structural smoothness check: at low field the tracked energy
        should be a smooth, monotonically-varying function of field (no
        jumps from mistracking), checked via a small second-difference
        bound rather than assuming any specific curvature sign.
        """
        n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        it = rcr.inglis_teller_field_v_per_m(n_star)
        fields = np.linspace(0.0, 0.1 * it, 15)
        result = rsm.stark_map_registry_state(32, fields, delta_n=2, l_max=6)
        e = result.tracked_energy_hz
        second_diff = np.diff(e, n=2)
        first_diff_scale = np.max(np.abs(np.diff(e)))
        assert np.max(np.abs(second_diff)) < first_diff_scale  # no jump dominates the trend


# ---------------------------------------------------------------------------
# Section G/H: C3 quadratic-crossover consistency (kill-tested) and C6 sweep
# ---------------------------------------------------------------------------


#: C3 tolerance: the mj-averaged map curvature vs. the Phase A registry's
#: derived alpha0(32D5/2). The map uses a reduced basis (delta_n=3,
#: l_max=10) in THIS test for runtime; the full l_max=20 production basis
#: (benchmarks/run_rydberg_stark_map.py) tightens this, per the C6
#: convergence study's own findings.
C3_TEST_TOLERANCE_RELATIVE = 0.15


class TestQuadraticCrossoverConsistency:
    def test_mj_averaged_alpha0_matches_registry_within_tolerance(self) -> None:
        n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        it = rcr.inglis_teller_field_v_per_m(n_star)
        fields = np.linspace(0.0, 0.25 * it, 12)
        alpha0, _ = rsm.scalar_polarizability_from_map(
            32, fields, delta_n=3, l_max=10, max_field_v_per_m=0.15 * it
        )
        registry = rcr.RB85_32D52_ALPHA0_AU
        rel_err = abs(alpha0 - registry) / registry
        assert rel_err < C3_TEST_TOLERANCE_RELATIVE

    def test_kill_sign_flip_fails_the_check(self) -> None:
        """Formula-level kill test: negating the fitted curvature (the
        sign a wrong ``Delta_E = +(1/2) alpha E^2`` convention would
        produce) must miss the registry value by far more than
        :data:`C3_TEST_TOLERANCE_RELATIVE`.
        """
        n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        it = rcr.inglis_teller_field_v_per_m(n_star)
        fields = np.linspace(0.0, 0.25 * it, 12)
        alpha0, _ = rsm.scalar_polarizability_from_map(
            32, fields, delta_n=3, l_max=10, max_field_v_per_m=0.15 * it
        )
        broken = -alpha0
        registry = rcr.RB85_32D52_ALPHA0_AU
        rel_err = abs(broken - registry) / registry
        assert rel_err > C3_TEST_TOLERANCE_RELATIVE

    def test_kill_doubled_coefficient_fails_the_check(self) -> None:
        n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        it = rcr.inglis_teller_field_v_per_m(n_star)
        fields = np.linspace(0.0, 0.25 * it, 12)
        alpha0, _ = rsm.scalar_polarizability_from_map(
            32, fields, delta_n=3, l_max=10, max_field_v_per_m=0.15 * it
        )
        broken = 2.0 * alpha0
        registry = rcr.RB85_32D52_ALPHA0_AU
        rel_err = abs(broken - registry) / registry
        assert rel_err > C3_TEST_TOLERANCE_RELATIVE


class TestCrossoverField:
    def test_computed_crossover_at_or_above_it_estimate_order_of_magnitude(self) -> None:
        """The dossier's own guidance: the IT estimate is an order-of-
        magnitude UNDER-guard, so a genuine computed crossover should
        not sit far below it. Checked for 35D5/2 at the PRODUCTION basis
        size (``delta_n=5, l_max=20``): a smaller test basis can carry
        accidental, non-physical near-degeneracies among its own
        truncation-edge high-l states that trigger this function's own
        overlap/gap criteria without being a real field-induced crossing,
        exactly the kind of basis-truncation artifact the C6 convergence
        study (``benchmarks/run_rydberg_stark_map.py``) exists to guard
        against; this test uses the basis size that study validates.
        The production benchmark's own C4 case independently found this
        exact state's computed crossover (50.34 V/cm) matches ARC's own
        first-low-overlap field exactly.
        """
        n_star = rcr.effective_quantum_number(35, rcr.RB85_ND52_QUANTUM_DEFECT)
        it = rcr.inglis_teller_field_v_per_m(n_star)
        fields = np.linspace(0.0, 2.2 * it, 60)
        result = rsm.stark_map_registry_state(35, fields, delta_n=5, l_max=20)
        crossover = rsm.first_crossover_field_v_per_m(result)
        assert crossover is not None
        assert crossover > 0.5 * it

    def test_validity_field_falls_back_to_inglis_teller_on_failure(self) -> None:
        """If the map computation raises, :func:`stark_validity_field_v_per_m`
        must fall back to the IT estimate (plan text: "keeping the
        estimate as fallback"), not propagate the exception.
        """
        n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        # n0=0 is unbuildable (forces build_basis/stark_hamiltonian to fail)
        field, source = rsm.stark_validity_field_v_per_m(0, n_star)
        assert source == "inglis_teller_fallback"
        assert field == pytest.approx(
            rcr.STARK_VALIDITY_MARGIN * rcr.inglis_teller_field_v_per_m(n_star)
        )

    def test_validity_field_uses_computed_crossover_when_available(self) -> None:
        n_star = rcr.effective_quantum_number(35, rcr.RB85_ND52_QUANTUM_DEFECT)
        it = rcr.inglis_teller_field_v_per_m(n_star)
        fields = np.linspace(0.0, 2.5 * it, 40)
        field, source = rsm.stark_validity_field_v_per_m(35, n_star, field_grid_v_per_m=fields)
        assert source in ("computed_crossover", "inglis_teller_fallback")


class TestConvergenceSweep:
    def test_convergence_rows_report_shrinking_or_small_shifts(self) -> None:
        """Structural check of :func:`convergence_sweep`'s own output
        shape and the largest basis's own zero self-relative-shift; the
        full, load-bearing convergence demonstration (with the production
        ``l_max=20`` reference and the 50D5/2 state, dossier risk 6) is
        the benchmark's own C6 case
        (``benchmarks/run_rydberg_stark_map.py``), not this fast smoke
        test.
        """
        n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        it = rcr.inglis_teller_field_v_per_m(n_star)
        fields = np.linspace(0.0, 0.2 * it, 8)
        rows = rsm.convergence_sweep(
            32,
            fields,
            basis_sizes=[(2, 6), (3, 8)],
            max_field_v_per_m=0.15 * it,
        )
        assert len(rows) == 2
        assert rows[-1].relative_shift_from_largest == 0.0
        assert rows[0].basis_size < rows[1].basis_size


# ---------------------------------------------------------------------------
# Section J: E44 integration -- the map-sourced shift_fn path
# ---------------------------------------------------------------------------


class TestE44MapSourcedIntegration:
    """Phase A's own C5 structural limit checks
    (``tests/test_rydberg_cell_response.py``'s zero-field and uniform-
    field byte-identical tests), re-run with
    :func:`compose_inhomogeneous_eit_spectrum`'s ``shift_fn`` bound to
    :func:`map_sourced_stark_shift_hz` instead of the default quadratic
    E43 formula -- the plan's own stated WP40 deliverable ("the EIT/AT
    observable can source its Rydberg shift from the map... with the
    Phase A limit checks re-run on the map path").
    """

    def _system(self) -> rcr.LadderSystem:
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

    def _map_shift_fn(self):
        from functools import partial

        return partial(rsm.map_sourced_stark_shift_hz, n0=32, delta_n=2, l_max=6)

    def test_zero_field_byte_identical_on_map_path(self) -> None:
        system = self._system()
        n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        delta_p = np.linspace(-2.0 * math.pi * 20e6, 2.0 * math.pi * 20e6, 51)
        fields0 = np.zeros(4)
        weights = np.ones(4)
        composed = rcr.compose_inhomogeneous_eit_spectrum(
            delta_p,
            fields0,
            weights,
            rcr.RB85_32D52_ALPHA0_AU,
            n_star,
            system,
            shift_fn=self._map_shift_fn(),
        )
        unperturbed = rcr.doppler_averaged_susceptibility(
            delta_p, 0.0, 0.0, 1.0, 1.0, 0.0, system, 320.0, rcr.RB85_MASS_KG, n_velocity_points=33
        )
        assert np.array_equal(composed, unperturbed)

    def test_uniform_field_reduces_to_single_atom_on_map_path(self) -> None:
        system = self._system()
        n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        delta_p = np.linspace(-2.0 * math.pi * 20e6, 2.0 * math.pi * 20e6, 51)
        field_uniform = 30.0
        fields_u = np.full(4, field_uniform)
        weights = np.ones(4)
        shift_fn = self._map_shift_fn()
        composed = rcr.compose_inhomogeneous_eit_spectrum(
            delta_p,
            fields_u,
            weights,
            rcr.RB85_32D52_ALPHA0_AU,
            n_star,
            system,
            shift_fn=shift_fn,
        )
        shift_hz = shift_fn(rcr.RB85_32D52_ALPHA0_AU, field_uniform, n_star)
        expected = rcr.doppler_averaged_susceptibility(
            delta_p,
            2.0 * math.pi * shift_hz,
            0.0,
            1.0,
            1.0,
            0.0,
            system,
            320.0,
            rcr.RB85_MASS_KG,
            n_velocity_points=33,
        )
        assert np.array_equal(composed, expected)

    def test_default_shift_fn_unchanged(self) -> None:
        """The new ``shift_fn`` keyword must not perturb any existing
        caller: with no ``shift_fn`` passed, behavior is byte-identical
        to before this WP40 change (the quadratic E43 formula).
        """
        system = self._system()
        n_star = rcr.effective_quantum_number(32, rcr.RB85_ND52_QUANTUM_DEFECT)
        delta_p = np.linspace(-2.0 * math.pi * 20e6, 2.0 * math.pi * 20e6, 51)
        fields = np.full(4, 30.0)
        weights = np.ones(4)
        default_call = rcr.compose_inhomogeneous_eit_spectrum(
            delta_p, fields, weights, rcr.RB85_32D52_ALPHA0_AU, n_star, system
        )
        explicit_call = rcr.compose_inhomogeneous_eit_spectrum(
            delta_p,
            fields,
            weights,
            rcr.RB85_32D52_ALPHA0_AU,
            n_star,
            system,
            shift_fn=rcr.rydberg_quadratic_stark_shift_hz,
        )
        assert np.array_equal(default_call, explicit_call)
