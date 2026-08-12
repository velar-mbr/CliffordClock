# SPDX-License-Identifier: AGPL-3.0-or-later
"""Re-fetch the full WP10 source files that are *not* committed to the repo.

``benchmarks/fixtures/`` ships small excerpts/transcriptions only (see
``benchmarks/SOURCES.md``): the JILA and NPL arXiv PDF/TeX sources are not
redistributed here (arXiv's default "perpetual, non-exclusive" submission
license does not grant third-party redistribution rights -- see
``benchmarks/SOURCES.md``), and the two NIST CSVs, while public domain and
freely redistributable, are ~2.2 MB each of data this benchmark does not
use (see ``benchmarks/MAPPING.md`` -- the dataset is a phase/Allan-
deviation instability record, not a systematic-shift measurement, so it
never enters a residual computation). This script re-downloads all six
files (network access required) and verifies each against the SHA-256
checksum recorded in ``benchmarks/SOURCES.md`` at original retrieval time
(2026-08-10), so a reader can independently confirm nothing has silently
changed upstream. (The USTC Metrologia 63,025002 PDF is owner-provided,
not fetched over the network -- see ``benchmarks/SOURCES.md`` section 5
for its provenance checksum; this script does not handle it.)

Usage::

    python benchmarks/fetch_data.py [--dest DIR]

Exits non-zero (with a clear message) if any download's SHA-256 does not
match the recorded checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

#: (relative filename, source URL, expected SHA-256), exactly the six
#: network-fetched files documented in benchmarks/SOURCES.md.
_FILES: tuple[tuple[str, str, str], ...] = (
    (
        "arxiv_2403.10664v2.pdf",
        "https://arxiv.org/pdf/2403.10664v2",
        "f222fe4de23aa4a5f7f3daa13c428767299a217a2849f2f2f9472f56dfef4871",
    ),
    (
        "arxiv_2403.10664v2_source.tar.gz",
        "https://arxiv.org/e-print/2403.10664v2",
        "f8c88d5046b086dac5bf588f03f52c094ef2eac453dd26399668524f8f684d95",
    ),
    (
        "Yb_Clock_phase(rad)_vs_time.csv",
        "https://data.nist.gov/od/ds/mds2-2206/Yb_Clock_phase%28rad%29%20vs%20time.csv",
        "c00f2c5c03c3ef0cf346de9917b66800eccd4147cc86af88feae9d02725baad9",
    ),
    (
        "10GHz_phase(mrad)_vs_time.csv",
        "https://data.nist.gov/od/ds/mds2-2206/10GHz_phase%28mrad%29%20vs%20time.csv",
        "9d715e33e8440dc7b84833ee51a48884e7e7091cc632f482ab2812fc466793be",
    ),
    (
        "arxiv_1706.01944v1.pdf",
        "https://arxiv.org/pdf/1706.01944v1",
        "f8b24d5a3cd8a7b2d1a2bd078b866526229883c919649519fd122fa7d118ef51",
    ),
    (
        "arxiv_1706.01944v1_source.tar.gz",
        "https://arxiv.org/e-print/1706.01944v1",
        "bc0b435aa91320e4a1b3f866e31740f5914d441380b1ca42abbcdb61346d4f55",
    ),
)


def fetch_and_verify(dest_dir: Path) -> bool:
    """Download every file in `_FILES` into `dest_dir` and verify its
    SHA-256 against the recorded checksum.

    Parameters
    ----------
    dest_dir : Path
        Destination directory (created if missing).

    Returns
    -------
    bool
        True if every file downloaded and matched its recorded checksum.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for filename, url, expected_sha256 in _FILES:
        target = dest_dir / filename
        print(f"Fetching {url} -> {target}")
        urllib.request.urlretrieve(url, target)  # noqa: S310 (fixed, documented URLs above)
        actual_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            print(
                f"  CHECKSUM MISMATCH for {filename}: expected {expected_sha256}, "
                f"got {actual_sha256} -- upstream content has changed since "
                "2026-08-10 retrieval; do not trust this file without investigating.",
                file=sys.stderr,
            )
            all_ok = False
        else:
            print(f"  OK: sha256 {actual_sha256} matches benchmarks/SOURCES.md")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "_fetched",
        help="Destination directory (default: benchmarks/data/_fetched/, gitignored)",
    )
    args = parser.parse_args()
    ok = fetch_and_verify(args.dest)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
