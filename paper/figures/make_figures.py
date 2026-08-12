#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Orchestrator: run every figure/table generator in dependency order.

Usage (from the repo root, with the project's ``.venv`` active):

    python paper/figures/make_figures.py

Regenerates every PDF figure under ``paper/figures/`` and every
``\\input``-able ``.tex`` value/table file under ``paper/generated/`` by
calling the real ``cliffordclock`` pipeline and ``benchmarks/`` code --
see each script's module docstring for what it computes.
``table_validation.py`` reads ``generated/step_size_values.tex``, so
``fig3_step_size_accuracy`` must run first; the rest are independent.

Note: ``fig1_worked_example.py`` still runs (its output is a valid,
still-correct pipeline demonstration) but its figure/numbers are no
longer cited in ``main.tex`` -- the paper's former "Worked example"
section was replaced by the showcase section
(``fig4_showcase_gradient_dispersion.py``), which makes the same kind of
demonstration with a field that has genuine spatial structure and reports
the resulting dispersion budget (spread, T2*, line profile), not just a
single mean-shift number. See ``CHANGELOG.md``.

``fig6_precision_ladder.py`` was removed with its figure (owner review,
2026-08-12): plotting relative agreements and absolute fractional-shift
errors on one axis invited a dimensionally invalid comparison, and its
five numbers all live in the validation table and verification prose
with their relative/absolute character attached.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_FIGURES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FIGURES_DIR.parent.parent

# Dependency order: fig3 writes generated/step_size_values.tex, which
# table_validation.py reads for the V4 row.
_SCRIPTS = [
    "fig1_worked_example.py",
    "fig2_npl_band.py",
    "bbr_jila_values.py",
    "fig5_bbr_temperature.py",
    "roos_values.py",
    "fig7_bothwell_sitemap.py",
    "fig3_step_size_accuracy.py",
    "fig4_showcase_gradient_dispersion.py",
    "table_validation.py",
    "table_budget_slice.py",
]


def main() -> None:
    for script in _SCRIPTS:
        path = _FIGURES_DIR / script
        print(f"\n=== Running {script} ===")
        start = time.perf_counter()
        # cwd=repo root: examples/*.yaml configs use paths relative to the
        # repo root (the same cwd `cliffordclock run` is documented to be invoked
        # from), e.g. `field.csv: examples/patch_field_sr87.csv`.
        subprocess.run([sys.executable, str(path)], check=True, cwd=str(_REPO_ROOT))
        elapsed = time.perf_counter() - start
        print(f"=== {script} done in {elapsed:.1f} s ===")
    print("\nAll figures and generated tables regenerated.")


if __name__ == "__main__":
    main()
