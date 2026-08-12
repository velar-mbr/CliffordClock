# SPDX-License-Identifier: AGPL-3.0-or-later
"""Metrology analytics outputs (fractional shift, T2*, line profile, report writers).

Turns the rotor path integrator's per-atom accumulated perturbation
phases (`ΔΦ_i`, E22) into the numbers metrologists consume
(`cliffordclock.analytics.stats`, E23/E25-E28) and machine-readable
reports (`cliffordclock.analytics.report`). See ``docs/CONVENTIONS.md``
sections 7-8.
"""

from __future__ import annotations

from cliffordclock.analytics.report import (
    CONVENTIONS_VERSION,
    REPORT_SCHEMA_VERSION,
    MetrologyReport,
    build_report,
    write_json,
    write_line_profile_csv,
)
from cliffordclock.analytics.stats import (
    WeightedPhaseStats,
    coherence_function,
    dephasing_time_t2star,
    line_profile,
    mean_fractional_shift,
    shift_std_error,
    weighted_phase_stats,
)

__all__ = [
    "CONVENTIONS_VERSION",
    "REPORT_SCHEMA_VERSION",
    "MetrologyReport",
    "WeightedPhaseStats",
    "build_report",
    "coherence_function",
    "dephasing_time_t2star",
    "line_profile",
    "mean_fractional_shift",
    "shift_std_error",
    "weighted_phase_stats",
    "write_json",
    "write_line_profile_csv",
]
