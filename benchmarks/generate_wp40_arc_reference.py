# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generates ``benchmarks/fixtures/wp40_arc_stark_map_reference.json``,
the WP40 C4 ARC cross-validation fixture (CONVENTIONS.md section 20).

**Provenance and the "install ARC in the venv" instruction.** Unlike the
WP38 INRIM fixture (``benchmarks/fixtures/wp38_inrim_large_lattice_model_reference.json``),
which was generated in a SEPARATE virtual environment from this
project's own (that library is not published on PyPI under a name
compatible with this project's own dependency set), ARC
(``arc-alkali-rydberg-calculator``, PyPI) installs cleanly alongside
this project's own dependencies with no conflicts (verified this
session: ``pip check`` reports no broken requirements after installing
both). The build prescription's own instruction is "install ARC in the
venv" (this project's own ``.venv``, not a separate one), so this script
runs there directly and imports :mod:`cliffordclock.integrator.rydberg_stark_map`
alongside ``arc`` in the same process -- simpler than the INRIM pattern,
and it lets this script reuse this project's OWN adiabatic-tracking code
(:func:`rydberg_stark_map.diagonalize_stark_map`) UNCHANGED on ARC's own
Hamiltonian matrices, isolating exactly what C4 needs to isolate: does
this module's own Hamiltonian construction (quantum-defect energies +
Numerov/Wigner dipole couplings) agree with ARC's, holding the
diagonalization and eigenvalue-tracking code fixed and identical on both
sides. ARC is NOT added to ``pyproject.toml``: only this fixture's own
numeric output is committed, the same "published-table-as-fixture"
posture the WP38 fixture already established for an independent
implementation's OUTPUT (as opposed to a paper's own printed number).

**ARC pin**, read from the installed package this session (re-verify at
regeneration time; record whichever commit is actually installed, this
dossier's own instruction against letting a pin note go stale):

- PyPI package: ``arc-alkali-rydberg-calculator``, installed version
  3.10.2 (``pip show`` this session).
- Tag ``v.3.10.2``, commit ``4b4573e965222e798ac59636ad7a8b3457262835``
  (github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator; installed via
  ``pip install "arc-alkali-rydberg-calculator @ git+https://github.com/
  nikolasibalic/ARC-Alkali-Rydberg-Calculator.git@4b4573e965222e798ac59636ad7a8b3457262835"``
  this session).
- License: BSD-3-Clause (PyPI project page and the CPC paper's own
  "Program summary" box both state this identically).

**What is recorded, per registry state.** ARC's own ``StarkMap.mat1``/
``mat2`` (field-independent diagonal / field-proportional off-diagonal
parts of the Stark Hamiltonian, ARC's own units of GHz and GHz/(V/m)
respectively -- read directly from ``defineBasis``'s own source this
session) for the SAME basis parameters this module's own production
default uses (``nMin=n0-5, nMax=n0+5, lMax=20``), converted to this
project's own atomic-unit convention
(:data:`cliffordclock.integrator.rydberg_cell_response.ATOMIC_UNIT_FIELD_V_PER_M`,
:data:`...HARTREE_TO_HZ`), then diagonalized and adiabatically tracked
by THIS PROJECT'S OWN CODE
(:func:`cliffordclock.integrator.rydberg_stark_map.diagonalize_stark_map`)
over the same field grid the benchmark itself uses. The fixture stores
the resulting tracked-energy curve (Hz), not ARC's raw matrices (which
would make the fixture far larger and would not by itself demonstrate
anything without also committing to a specific tracking algorithm; the
tracked curve IS the comparison object C4 needs).

Run this yourself (from the repo root, ``.venv`` active, ARC installed
per the pin above): ``python benchmarks/generate_wp40_arc_reference.py``.
Regenerates ``benchmarks/fixtures/wp40_arc_stark_map_reference.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from arc import Rubidium85, StarkMap  # type: ignore[import-untyped]

from cliffordclock.integrator import rydberg_cell_response as rcr
from cliffordclock.integrator import rydberg_stark_map as rsm

_BENCHMARKS_DIR = Path(__file__).resolve().parent
_FIXTURE_PATH = _BENCHMARKS_DIR / "fixtures" / "wp40_arc_stark_map_reference.json"

REGISTRY_N_VALUES: tuple[int, ...] = (30, 32, 35, 50)
DELTA_N = 5
L_MAX = 20
N_FIELD_POINTS = 60
FIELD_RANGE_IT_MULTIPLE = 2.2


def _arc_version() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "show", "arc-alkali-rydberg-calculator"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in out.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _arc_hamiltonian_in_atomic_units(n0: int) -> rsm.StarkHamiltonian:
    atom = Rubidium85()
    calc = StarkMap(atom)
    calc.defineBasis(n0, 2, 2.5, 0.5, n0 - DELTA_N, n0 + DELTA_N, L_MAX, progressOutput=False)
    mat1_hartree = calc.mat1 * 1.0e9 / rcr.HARTREE_TO_HZ
    mat2_hartree_per_au_field = (
        calc.mat2 * 1.0e9 * rcr.ATOMIC_UNIT_FIELD_V_PER_M / rcr.HARTREE_TO_HZ
    )
    return rsm.StarkHamiltonian(
        basis=[],  # not needed downstream; this fixture only reads tracked energies
        h0=mat1_hartree,
        h1=mat2_hartree_per_au_field,
        target_index=calc.indexOfCoupledState,
    )


def generate() -> dict[str, Any]:
    states: dict[str, Any] = {}
    for n0 in REGISTRY_N_VALUES:
        n_star = rcr.effective_quantum_number(n0, rcr.RB85_ND52_QUANTUM_DEFECT)
        it_field_v_per_m = rcr.inglis_teller_field_v_per_m(n_star)
        fields = np.linspace(0.0, FIELD_RANGE_IT_MULTIPLE * it_field_v_per_m, N_FIELD_POINTS)

        hamiltonian = _arc_hamiltonian_in_atomic_units(n0)
        result = rsm.diagonalize_stark_map(hamiltonian, fields)

        states[str(n0)] = {
            "n_star": n_star,
            "inglis_teller_field_v_per_m": it_field_v_per_m,
            "field_v_per_m": fields.tolist(),
            "tracked_energy_hz": result.tracked_energy_hz.tolist(),
            "min_overlap": result.min_overlap,
            "step_overlaps": result.step_overlaps.tolist(),
            "basis_size": hamiltonian.h0.shape[0],
            "target_index": hamiltonian.target_index,
        }

    return {
        "wp40_arc_stark_map_reference_schema": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "package": "arc-alkali-rydberg-calculator",
            "installed_version": _arc_version(),
            "repo": "https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator",
            "tag": "v.3.10.2",
            "commit": "4b4573e965222e798ac59636ad7a8b3457262835",
            "license": "BSD-3-Clause",
            "install_method": (
                'pip install "arc-alkali-rydberg-calculator @ '
                "git+https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator.git"
                '@4b4573e965222e798ac59636ad7a8b3457262835", installed into THIS '
                "project's own .venv (no separate environment needed: pip check "
                "reports no conflicts), per the build prescription's own instruction"
            ),
            "note": (
                "Generated by benchmarks/generate_wp40_arc_reference.py. No ARC code "
                "is vendored; only these numeric outputs (ARC's own mat1/mat2, "
                "converted to this project's atomic-unit convention, then "
                "diagonalized and adiabatically tracked by THIS PROJECT'S OWN "
                "rydberg_stark_map.diagonalize_stark_map) are copied into "
                "CliffordClock as a benchmark fixture. Holding the tracking "
                "algorithm fixed and identical on both sides isolates the C4 "
                "comparison to the Hamiltonian construction alone."
            ),
            "basis_parameters": {
                "delta_n": DELTA_N,
                "l_max": L_MAX,
                "target_state": "n0, l=2 (D), j=2.5 (5/2), mj=0.5",
            },
        },
        "states": states,
    }


def main() -> None:
    report = generate()
    _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE_PATH.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote {_FIXTURE_PATH}")


if __name__ == "__main__":
    main()
