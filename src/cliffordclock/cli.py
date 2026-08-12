# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command-line entry point for CliffordClock (WP6 scope item 2).

``cliffordclock run config.yaml [--output-dir DIR]`` runs the full pipeline
(:mod:`cliffordclock.pipeline`) and writes ``report.json`` +
``line_profile.csv`` (plus ``site_map.json`` for
``ensemble.regime: lattice_extended`` runs, WP22 Part 2), printing a short
human summary. ``cliffordclock version`` prints the installed package
version. See ``docs/cli.md`` for the ``config.yaml`` schema.

Exit codes: ``0`` success, ``1`` physics-validation failure
(:class:`~cliffordclock.pipeline.PhysicsValidationError`), ``2`` bad input
(:class:`~cliffordclock.pipeline.PipelineConfigError`, a missing/unreadable
file, or a bad CLI argument -- the last handled by ``argparse`` itself,
which also exits 2). Errors go to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from cliffordclock import __version__
from cliffordclock.analytics import MetrologyReport, write_json, write_line_profile_csv
from cliffordclock.pipeline import (
    LatticeExtendedSiteMap,
    PhysicsValidationError,
    PipelineConfig,
    PipelineConfigError,
    run_pipeline_full,
)

#: `LatticeExtendedSiteMap` JSON schema version (WP22 Part 2) -- independent
#: of `cliffordclock.analytics.report.REPORT_SCHEMA_VERSION`, mirroring that
#: constant's own "bump on any field/type/shape change" role for this
#: separate (not part of `MetrologyReport`) output file.
SITE_MAP_SCHEMA_VERSION = "1.0"


def _write_site_map_json(site_map: LatticeExtendedSiteMap, path: Path) -> None:
    """Write `site_map` (WP22 Part 2) to `path` as schema-versioned JSON.

    Mirrors `cliffordclock.analytics.report.write_json`'s style (stable key
    order via `dataclasses.asdict`, `SITE_MAP_SCHEMA_VERSION` as the first
    key) but is not that function itself: `LatticeExtendedSiteMap` is not a
    `MetrologyReport` and every one of its fields is always finite (no
    NaN/Inf sanitization is needed the way `write_json` needs for
    `shift_std_error`/`t2_star_s`).

    Parameters
    ----------
    site_map : LatticeExtendedSiteMap
    path : pathlib.Path
        Destination file path; overwritten if it exists.
    """
    data = {"site_map_schema": SITE_MAP_SCHEMA_VERSION, **asdict(site_map)}
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


#: Below this resolved interrogation time (seconds), the run is not a
#: physically meaningful interrogation window -- it is a validation-scale
#: probe (e.g. a Compton-scale `dtau`/`steps` pair used to exercise the
#: integrator directly). See ``docs/timescales.md`` for what a *real*
#: interrogation time looks like and why validation-scale runs exist.
VALIDATION_SCALE_THRESHOLD_S = 1e-9


def _print_summary(
    report: MetrologyReport,
    report_path: Path,
    line_profile_path: Path,
    site_map_path: Path | None = None,
) -> None:
    """Print a <=12-line human summary: shift +/- SEM, T2*, and (when the
    resolved interrogation time is not physically realistic) a note
    pointing at ``docs/timescales.md``.

    The real interrogation time (`report.interrogation_time_s`, printed
    unconditionally, already prominent) and `report.uncertainty_notes`
    (which embeds `integration.mode` plus its effective dτ̃/steps or T) are
    both printed so a real-time-scale (e.g. 1 s lattice) run is legible at a
    glance. When `report.interrogation_time_s` is below
    `VALIDATION_SCALE_THRESHOLD_S`, one extra line flags the run as a
    validation-scale probe rather than a physical interrogation time --
    this never changes `report.json`, only the printed summary.

    Parameters
    ----------
    site_map_path : pathlib.Path or None
        The written ``site_map.json`` path (WP22 Part 2,
        `ensemble.regime: lattice_extended` runs only); `None` (the
        default, every other regime) omits the extra summary line.
    """
    print("CliffordClock run summary")
    print(f"  species:                {report.species_name}")
    print(f"  ensemble:               {report.ensemble_type} (M={report.ensemble_size})")
    print(f"  interrogation time:     {report.interrogation_time_s:.6e} s")
    print(
        f"  mean fractional shift:  {report.mean_fractional_shift:+.6e} "
        f"+/- {report.shift_std_error:.3e} (SEM)"
    )
    print(f"  T2*:                    {report.t2_star_s:.6e} s")
    if report.uncertainty_notes:
        print(f"  notes:                  {report.uncertainty_notes}")
    if report.interrogation_time_s < VALIDATION_SCALE_THRESHOLD_S:
        print(
            "  note: validation-scale run (not a physical interrogation "
            "time) -- see docs/timescales.md"
        )
    print(f"  report:                 {report_path}")
    print(f"  line profile:           {line_profile_path}")
    if site_map_path is not None:
        print(f"  site map:               {site_map_path}")


def _cmd_run(config_path: str, output_dir: str | None) -> int:
    """Run ``cliffordclock run``; returns the process exit code (see module docstring)."""
    try:
        config = PipelineConfig.from_yaml(config_path)
        if output_dir is not None:
            config = replace(config, output=replace(config.output, directory=output_dir))
        result = run_pipeline_full(config)
    except PipelineConfigError as exc:
        print(f"cliffordclock run: invalid configuration: {exc}", file=sys.stderr)
        return 2
    except PhysicsValidationError as exc:
        print(f"cliffordclock run: physics validation failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"cliffordclock run: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(config.output.directory)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"cliffordclock run: cannot create output directory {out_dir}: {exc}", file=sys.stderr
        )
        return 2

    report_path = out_dir / config.output.report_filename
    line_profile_path = out_dir / config.output.line_profile_filename
    write_json(result.report, report_path)
    write_line_profile_csv(
        result.line_profile_freqs_hz, result.line_profile_amplitude, line_profile_path
    )

    site_map_path: Path | None = None
    if result.site_map is not None:
        site_map_path = out_dir / config.output.site_map_filename
        _write_site_map_json(result.site_map, site_map_path)

    _print_summary(result.report, report_path, line_profile_path, site_map_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``cliffordclock`` command-line interface.

    Parameters
    ----------
    argv : Sequence[str] or None
        Command-line arguments, excluding the program name. Defaults to
        ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Process exit code (see module docstring).
    """
    parser = argparse.ArgumentParser(prog="cliffordclock")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="Print the installed package version")
    run_parser = subparsers.add_parser(
        "run", help="Run an end-to-end simulation from a config.yaml"
    )
    run_parser.add_argument("config", help="Path to a config.yaml file")
    run_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Override the config's output.directory",
    )

    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "run":
        return _cmd_run(args.config, args.output_dir)

    parser.print_help()  # pragma: no cover - unreachable: subparsers are required
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
