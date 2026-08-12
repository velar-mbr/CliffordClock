# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integrity checks for the committed Lisdat et al. 2021 dataset fixture.

``benchmarks/data/lisdat2021_dataset/`` carries five files redistributed
verbatim from the PTB Open Access Repository (DOI 10.7795/720.20210928)
under CC BY-ND 4.0, a No-Derivatives license: this project may cite and
use the published values, and may redistribute the files unmodified, but
may never commit an edited copy. This test pins each file's SHA-256
digest against the checksum recorded when the files were first added, so
any future accidental edit (a line-ending conversion, a stray
reformat, a partial re-download) fails loudly here rather than silently
producing a "verbatim" dataset that no longer is one. It also sanity-checks
that ``BBR_shift.dat`` (the file
``paper/figures/fig5_bbr_temperature.py`` reads) parses the way that
script assumes: a temperature column covering the engine's 50-350 K
validity window, in ascending order.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

_DATASET_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "data" / "lisdat2021_dataset"

#: SHA-256 digests recorded when these files were copied verbatim from the
#: PTB Open Access Repository archive (2026-08-11). Any change to this
#: dict should only ever happen alongside a full re-verification that the
#: new file is still an unmodified, verbatim copy of the published data
#: (never a hand-edited one) -- see the directory's README.md.
_EXPECTED_SHA256 = {
    "info.txt": "1219f7decc48d4efe65164bebc65d6122780cdf286d442c092198545c00e2bcf",
    "BBR_shift.dat": "f98ea52a7621a8299fc494a66621093836cbe1cf296ed069c96d8f4a17425b56",
    "G(3D1_T).dat": "926b238c2a9d26b43575c6a4e128b16b2c9b2b12fe6e40980278386b9413fe1f",
    "Approximation_G(n)_Sr.dat": "b4f72232a61a96c7a04c90c7197da4736a739539da6fcac24b438f83692ec4c1",
    "Approximation_G(n)_Yb.dat": "8453549f1062483b43bb84f4965c405926998770efbdaf6c8c1410403307e10a",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("filename", sorted(_EXPECTED_SHA256))
def test_dataset_file_present(filename: str) -> None:
    assert (_DATASET_DIR / filename).is_file(), f"missing dataset file: {filename}"


@pytest.mark.parametrize("filename", sorted(_EXPECTED_SHA256))
def test_dataset_file_matches_recorded_checksum(filename: str) -> None:
    """Pins each file byte-for-byte, so no accidental edit passes unnoticed.

    A CC BY-ND 4.0 (No-Derivatives) license permits verbatim
    redistribution only; if this test ever fails after an intentional,
    verified re-download (not a hand edit), update
    ``_EXPECTED_SHA256`` and the directory's README.md together.
    """
    digest = _sha256(_DATASET_DIR / filename)
    assert digest == _EXPECTED_SHA256[filename], (
        f"{filename} no longer matches its recorded checksum -- this dataset directory "
        "must only ever contain verbatim copies of the published files (CC BY-ND 4.0)"
    )


def test_readme_present_and_documents_license() -> None:
    readme = _DATASET_DIR / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    assert "CC BY-ND 4.0" in text
    assert "10.7795/720.20210928" in text


def test_bbr_shift_dat_covers_engine_validity_window() -> None:
    """``fig5_bbr_temperature.py`` overlays this file's total-shift column over 50-350 K."""
    data = np.genfromtxt(
        _DATASET_DIR / "BBR_shift.dat",
        skip_header=2,
        delimiter="\t",
        usecols=(0, 1, 2, 3, 4),
        dtype=float,
    )
    temperatures = data[:, 0]
    assert np.all(np.diff(temperatures) > 0), "temperature column must be strictly ascending"
    assert temperatures.min() <= 50.0
    assert temperatures.max() >= 350.0
    # The total-shift column (index 1) must be negative everywhere in the
    # engine's validity window (same sign convention as
    # cliffordclock.integrator.omega.bbr_pivot_perturbation: the clock
    # runs slow in the thermal bath).
    window = (temperatures >= 50.0) & (temperatures <= 350.0)
    assert np.all(data[window, 1] < 0.0)
