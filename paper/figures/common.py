# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared setup for every ``paper/figures/make_*.py`` script.

Every figure script in this directory calls the *real* ``cliffordclock``
pipeline (and, where relevant, ``benchmarks/`` scripts) -- nothing in
``paper/main.tex`` is a hand-typed pipeline output. This module puts
``src/`` and ``benchmarks/`` on ``sys.path`` (mirroring what
``benchmarks/run_benchmarks.py`` and the notebooks already do) and
centralizes the figure/`\\input`-file output directories and a consistent
matplotlib style, so every script in this directory produces visually
consistent, theme-neutral (print-ready) PDF figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

PAPER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PAPER_DIR.parent
FIGURES_DIR = PAPER_DIR / "figures"
GENERATED_DIR = PAPER_DIR / "generated"

_SRC_DIR = REPO_ROOT / "src"
_BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
_TESTS_DIR = REPO_ROOT / "tests"

for _p in (_SRC_DIR, _BENCHMARKS_DIR, _TESTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

GENERATED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# A single, print-ready (light-background, colorblind-tolerant) style used
# by every figure script -- arXiv PDFs are read on paper and on both light
# and dark screen themes, so figures use a neutral white background with a
# high-contrast, non-fully-saturated palette rather than any app-specific
# theming.
mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # embed TrueType, not Type 3, for arXiv compliance
        "ps.fonttype": 42,
    }
)

#: Consistent categorical palette (colorblind-tolerant, matches the
#: project's `dataviz` conventions: no fully-saturated primaries).
COLOR_ENGINE = "#2E5FA3"  # this engine's predicted values
COLOR_REFERENCE = "#C1666B"  # external/reference/published values
COLOR_NEUTRAL = "#6B6B6B"


def write_tex_macro(name: str, value: str, filename: str) -> None:
    """Append (or create) a ``\\newcommand`` macro definition in ``generated/<filename>``.

    Parameters
    ----------
    name : str
        Macro name without the leading backslash, e.g. ``"KaOneShift"``.
    value : str
        The LaTeX-ready value (already formatted, e.g. ``"-7.17\\times10^{-17}"``).
    filename : str
        Target file under ``paper/generated/`` (created/appended).
    """
    path = GENERATED_DIR / filename
    line = f"\\newcommand{{\\{name}}}{{{value}}}\n"
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        f.write(line)


def reset_tex_macro_file(filename: str) -> None:
    """Truncate ``generated/<filename>`` so a re-run does not append duplicates."""
    path = GENERATED_DIR / filename
    if path.exists():
        path.unlink()
