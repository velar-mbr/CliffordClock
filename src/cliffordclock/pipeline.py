# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: N806, N815
# `temperature_uK` mirrors the CONVENTIONS.md-mandated sampler-API
# parameter name (docs/CONVENTIONS.md section 10: "temperature uK at
# sampler APIs"); pep8-naming's N806/N815 would otherwise flag the
# embedded capital K on this dataclass field and the local variable that
# feeds it.
"""One-call pipeline façade: config -> report.

``run_pipeline``/``run_pipeline_full`` compose the field, ensemble,
integrator, and analytics modules end to end -- load/synthesize a field,
build an ensemble, integrate the rotor path equation, and analyze the
result -- with **no new physics**. Every equation is implemented upstream
(``fields``, ``ensemble``, ``integrator``, ``analytics``); this module only
wires them together per a declarative :class:`PipelineConfig` (see
``docs/cli.md`` for the YAML schema and ``docs/CONVENTIONS.md`` for the
equations).

Interface note (binding on this module):

1. :mod:`cliffordclock.fields.synthetic` factories return two callables
   ``(e_fn, grad_fn)``; :meth:`~cliffordclock.fields.smoother.FieldSmoother.evaluate`
   and the integrator's :data:`~cliffordclock.integrator.worldline.FieldFn`
   use one combined ``pos -> (E, grad_E)`` callable. This module
   standardizes on the combined convention everywhere via the public
   :func:`~cliffordclock.fields.synthetic.as_field_fn` adapter -- no
   per-call-site ad-hoc lambda.
2. Trajectory sampling and the integrator's ``dtau`` are derived from the
   *same* config value (``integration.dtau``): the classical-ensemble
   Verlet propagator's step ``dt`` (seconds) is computed as
   ``dtau * TAU_COMPTON`` (E9), so trajectory samples are spaced at
   exactly ``dτ̃`` -- the assumption the integrator's finite-difference
   velocity depends on (documented in
   ``cliffordclock.integrator.worldline.integrate_worldline``, not
   runtime-enforced there). This applies to ``integration.mode`` values
   ``"direct"``/``"worldline"`` (see :func:`run_pipeline_full` below);
   the fast-path modes (note 3) do not sample a discretized trajectory.
3. **Fast-path modes (CONVENTIONS.md section 12).**
   ``ensemble.regime: lattice`` configs default to
   ``integration.mode: fast_path`` (E29: exact, no time stepping,
   ``interrogation.time_s``-driven); ``ensemble.regime: classical``
   configs default to ``integration.mode: direct`` (the existing rotor
   integrator, with ``integration.dtau`` auto-selected via
   :func:`cliffordclock.integrator.fastpath.select_dtau` (E31) when
   omitted) and may opt into ``integration.mode: secular`` (E30, periodic
   harmonic motion only). ``integration.mode: worldline`` is the explicit
   lattice cross-check against the rotor integrator (E29's own claim:
   must agree exactly on static nodes). See ``docs/timescales.md`` for
   the three-tier architecture this implements and ``docs/cli.md`` for
   the full config schema.
4. **Large-dτ̃ safety net.** The ``"direct"``/
   ``"worldline"`` dτ̃-resolution path
   (:func:`_resolve_dtau_steps_direct`) pre-flight-checks the estimated
   per-step rotor generator angle against
   :data:`MAX_PER_STEP_ROTOR_ANGLE_RAD` (an auto-selected dτ̃ is
   tightened to respect it; an explicit dτ̃ that violates it is rejected
   loudly, :class:`PipelineConfigError`) and auto-tightens
   ``renorm_every`` when dτ̃ is auto-selected
   (:func:`_resolve_renorm_every`) -- both because
   :func:`cliffordclock.cl13.exp_bivector`'s fixed-order Taylor
   evaluation has a convergence range and a per-call floor that only
   become relevant at E31-scale dτ̃, not at the original Compton-scale dτ̃.
   The same resolvers (plus :func:`_resolve_dtau_steps_scalar`) also
   pre-flight-check the estimated dense ``(M, steps+1, 3)`` trajectory
   allocation against ``integration.max_trajectory_memory_gb``
   (:func:`_check_trajectory_memory`) -- an auto-selected E31 dτ̃ over a
   long ``time_s`` can otherwise silently attempt a 100+ GB allocation.
   For ``coupling.type='stark_dc'`` + ``mode='direct'``
   (:func:`_stark_scalar_ensemble`) with a
   :class:`~cliffordclock.fields.smoother.FieldSmoother`-backed field
   (``field.csv``/``field.comsol``), the same check also adds a term for
   the smoother's ``(N, K, 3)`` evaluation intermediates (`N = M x
   steps` query points, `K` fit points) -- see
   :data:`_TRAJECTORY_MEMORY_FACTOR_SMOOTHER`; closed-form
   ``field.synthetic`` fields are unaffected. ``integration.mode:
   secular`` gets the same base trajectory check too (at the
   `run_pipeline_full` call site, keyed on ``points_per_period`` rather
   than a resolved `steps`) since it is otherwise unbounded in
   ``integration.points_per_period``. See ``docs/timescales.md`` and
   :func:`_validate_physics`.

   **WP19 update: this estimate is now an advisory *selector*, not just a
   wall, for ``ensemble.regime="classical"`` + ``mode="direct"`` (both
   `coupling.type` values).** Under ``integration.evaluation="auto"``
   (default), exceeding ``max_trajectory_memory_gb`` no longer raises
   :class:`PipelineConfigError` for that one combination -- it switches to
   an O(M)-memory streaming accumulator instead
   (:func:`_stark_scalar_ensemble_streaming`/
   :func:`_direct_rotor_ensemble_streaming`, dispatched by
   :func:`_resolve_evaluation_mode`) and records a note in the report's
   `uncertainty_notes`. ``mode="worldline"``/``"secular"`` (and
   ``mode="direct"`` with ``evaluation="batched"`` explicitly requested)
   keep the pre-WP19 hard-reject behavior unchanged. See
   ``docs/timescales.md``'s rewritten "Safety net" section.
5. **Mode table (which accumulator actually runs a given config, WP16).**
   Every ``(coupling.type, integration.mode)`` combination below is a
   distinct code path; only the ``coupling.type="stark_dc"`` +
   ``mode="worldline"`` cell changed under WP16 (it now runs the true
   Cl(1,3) rotor, replacing the scalar stand-in it used before -- every
   other cell is unchanged):

   | ``coupling.type`` | ``mode`` | accumulator |
   |---|---|---|
   | ``linear_mu`` | ``direct`` (batched) | rotor, `worldline.integrate_ensemble` (E17-E24) |
   | ``linear_mu`` | ``direct`` (streaming, WP19) | rotor, `_direct_rotor_ensemble_streaming` |
   | ``linear_mu`` | ``worldline`` | rotor, `worldline.integrate_ensemble` (E17-E24) |
   | ``linear_mu`` | ``fast_path``/``secular`` | scalar, `fastpath` (E29/E30) |
   | ``stark_dc`` | ``direct`` (batched) | scalar, `_stark_scalar_ensemble` (footnote 1) |
   | ``stark_dc`` | ``direct`` (streaming, WP19) | scalar, `_stark_scalar_ensemble_streaming` |
   | ``stark_dc`` | ``worldline`` | rotor, `_stark_rotor_ensemble` (footnote 2) |
   | ``stark_dc`` | ``fast_path``/``secular`` | scalar, `fastpath` (footnote 3) |

   "batched" vs. "streaming" (WP19): see :func:`_resolve_evaluation_mode`
   and ``integration.evaluation`` (default ``"auto"``, dispatches on the
   ``max_trajectory_memory_gb`` estimate). Both produce the same physics
   to a documented tight numerical bound (streaming-vs-batched agreement
   tests, ``tests/test_e2e.py``).

   Footnotes: (1) E21/E22 only -- classical-ensemble trajectories, not a
   lattice static node, so a rotor cross-check is not the E29-style
   exact-agreement case WP16 targets. (2) E17-E24; WP16's new direct
   rotor<->scalar cross-check target -- see
   ``tests/test_integrator_stark_rotor.py`` and ``docs/CONVENTIONS.md``'s
   production-path note. (3) via :func:`_make_stark_rate_fn`, unchanged
   by WP16.
6. **Blackbody-radiation shift (WP20, CONVENTIONS.md E32/E33).** An
   optional top-level ``environment:`` config section
   (:class:`EnvironmentConfig`) carries ``radiation_temperature_K``
   (absent -> BBR off, every table row above unaffected -- see
   :func:`cliffordclock.integrator.omega.bbr_pivot_perturbation`'s
   docstring for the E32 formula). When present, `coupling.type` must be
   ``"stark_dc"`` (BBR needs the species' registry
   `~cliffordclock.ensemble.species.BbrCoefficients`, not a
   coupling-agnostic quantity) and the resolved ``(P−1)_BBR`` scalar is
   composed (E33) into *every* ``stark_dc`` accumulator in the table
   above via a keyword-only ``bbr_pivot_perturbation`` parameter threaded
   through :func:`_make_stark_rate_fn` (covers ``fast_path``/``secular``/
   ``direct`` batched+streaming, all via the shared `rate_fn` closure) and
   :func:`_stark_rotor_ensemble` (``worldline``). See
   :func:`_resolve_bbr_pivot_perturbation`.
7. **Ion-clock electric-quadrupole shift (WP21, CONVENTIONS.md E34/E35).**
   An optional top-level ``quadrupole:`` config section
   (:class:`QuadrupoleConfig`) mirrors `environment:`'s pattern (absent ->
   off, requires `coupling.type='stark_dc'`) but composes a
   per-point scalar (not a single per-run constant like BBR): the
   quadrupole shift depends on the LOCAL field-gradient tensor (E13),
   already available at every `field_fn`/`_make_stark_rate_fn` call site.
   :func:`_quadrupole_pivot_from_grad` evaluates it from each point's
   `grad_E` and is threaded into `pivot_perturbation_stark`/
   `spin_connection_stark`/`scalar_rate_perturbation_stark`/
   `build_omega_stark`'s (all in `cliffordclock.integrator.omega`) new
   keyword-only ``quadrupole_pivot_perturbation`` parameter, mirroring
   `bbr_pivot_perturbation`'s threading exactly -- covering the same four
   evaluation-mode cells (`fast_path`/`secular`/`direct` batched
   +streaming via the shared `rate_fn`; `worldline` via
   :func:`_stark_rotor_ensemble`). Every ion-species (`Al27+`/`In115+`)
   report also carries the micromotion-boundary and hyperfine-E2
   budget-line notes (`cliffordclock.ensemble.species.ION_MICROMOTION_NOTES`/
   `ION_HYPERFINE_E2_BUDGET_NOTES`) regardless of whether `quadrupole:` is
   set, per the G8 sign-off's shipping requirement. See
   :func:`_resolve_quadrupole_theta_j`/:func:`_quadrupole_pivot_from_grad`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray

from cliffordclock.analytics import (
    CONVENTIONS_VERSION,
    REPORT_SCHEMA_VERSION,
    MetrologyReport,
    build_report,
    coherence_function,
    line_profile,
    mean_fractional_shift,
    weighted_phase_stats,
)
from cliffordclock.cl13 import (
    IDX_E12,
    IDX_SCALAR,
    exp_bivector,
    geometric_product,
    normalize_rotor,
    rotor_norm_sq,
)
from cliffordclock.constants import SPEED_OF_LIGHT, STANDARD_GRAVITY, TAU_COMPTON
from cliffordclock.ensemble.classical import propagate_verlet, sample_maxwell_boltzmann
from cliffordclock.ensemble.lattice import (
    VALID_SITE_ENVELOPES,
    ExtendedLatticeGeometry,
    extended_lattice_nodes,
    hermite_gaussian_nodes,
)
from cliffordclock.ensemble.species import (
    BBR_REFERENCE_TEMPERATURE_K,
    BBR_VALIDITY_MAX_K,
    BBR_VALIDITY_MIN_K,
    ION_HYPERFINE_E2_BUDGET_NOTES,
    ION_MICROMOTION_NOTES,
    Species,
    StarkCoefficients,
    get_quadrupole_moment,
    get_species,
)
from cliffordclock.ensemble.traps import HarmonicTrap
from cliffordclock.fields import FieldSmoother, load_field_comsol, load_field_csv
from cliffordclock.fields.smoother import DEFAULT_CHUNK_SIZE, chunked_apply
from cliffordclock.fields.synthetic import (
    FieldFn as SynthFieldFn,
)
from cliffordclock.fields.synthetic import (
    GradFn as SynthGradFn,
)
from cliffordclock.fields.synthetic import (
    as_field_fn,
    constant_gradient_field,
    gaussian_bump_field,
    quadrupole_field,
    uniform_field,
)
from cliffordclock.integrator import fastpath
from cliffordclock.integrator.omega import (
    bbr_pivot_perturbation as _bbr_pivot_perturbation_e32,
)
from cliffordclock.integrator.omega import (
    bbr_pivot_uncertainty,
    build_omega_stark,
    height_along_axis,
    pivot_perturbation_stark,
    quadrupole_three_orientation_average,
    scalar_rate_perturbation,
)
from cliffordclock.integrator.omega import (
    grav_pivot_perturbation as _grav_pivot_perturbation_e36,
)
from cliffordclock.integrator.omega import (
    quadrupole_pivot_perturbation as _quadrupole_pivot_perturbation_e34,
)
from cliffordclock.integrator.stepper import rotor_plane_angle, rotor_step
from cliffordclock.integrator.worldline import (
    DEFAULT_RENORM_EVERY,
    EnsembleResult,
    integrate_ensemble,
    kahan_sum,
)
from cliffordclock.integrator.worldline import (
    FieldFn as CombinedFieldFn,
)

__all__ = [
    "CouplingConfig",
    "EnsembleConfig",
    "EnvironmentConfig",
    "FieldConfig",
    "GravityConfig",
    "IntegrationConfig",
    "LatticeExtendedSiteMap",
    "OutputConfig",
    "PhysicsValidationError",
    "PipelineConfig",
    "PipelineConfigError",
    "PipelineResult",
    "QuadrupoleConfig",
    "SiteMapEntry",
    "SyntheticFieldConfig",
    "TrapConfig",
    "run_pipeline",
    "run_pipeline_full",
]


class PipelineConfigError(ValueError):
    """Malformed or invalid :class:`PipelineConfig` input (CLI exit code 2, "bad input")."""


class PhysicsValidationError(RuntimeError):
    """A pipeline run produced physically invalid output.

    CLI exit code 1, "physics-validation failure".

    Raised by :func:`run_pipeline_full` when the integrated ensemble result
    fails a basic sanity check: non-finite accumulated phase, or rotor-norm
    drift (E20) far beyond what a correctly configured run should ever
    show. This is a coarse trip-wire, not a precision-grade physics check
    (those live in the per-module test suites and ``tests/test_e2e.py``); it
    exists so a badly misconfigured CLI run (e.g. ``dtau`` far too large)
    fails loudly with a distinct exit code rather than silently reporting
    garbage.
    """


# ---------------------------------------------------------------------------
# Config schema (see docs/cli.md for the YAML documentation).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticFieldConfig:
    """One synthetic field factory selection (``field.synthetic`` in YAML).

    Attributes
    ----------
    kind : str
        One of ``"uniform"``, ``"constant_gradient"``, ``"quadrupole"``,
        ``"gaussian_bump"`` (the :mod:`cliffordclock.fields.synthetic`
        factories).
    params : dict[str, Any]
        Keyword arguments forwarded to the chosen factory.
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FieldConfig:
    """Field source (``field:`` in YAML): exactly one of `csv_path`/`comsol_path`/`synthetic`.

    Attributes
    ----------
    csv_path : str or None
        Path to a CSV-exported field grid (``cliffordclock.fields.load_field_csv``),
        fit with :class:`~cliffordclock.fields.smoother.FieldSmoother`.
    comsol_path : str or None
        Path to a COMSOL "Spreadsheet"-format ``File > Export > Data``
        export (``cliffordclock.fields.load_field_comsol``), fit with
        :class:`~cliffordclock.fields.smoother.FieldSmoother` exactly like
        `csv_path` (see `smoothing`). See ``docs/fields.md``/
        ``docs/byof-guide.md`` for the export-format contract.
    comsol_expression_prefix : str
        Physics-interface tag forwarded to
        ``load_field_comsol(..., expression_prefix=...)`` when `comsol_path`
        is used; ignored otherwise. Default ``"es"`` (COMSOL's built-in
        Electrostatics interface) matches `load_field_comsol`'s own default.
    synthetic : SyntheticFieldConfig or None
        A closed-form synthetic test field (exact gradients, no smoothing).
    smoothing : float
        Tikhonov regularization forwarded to ``FieldSmoother.fit`` when
        `csv_path` or `comsol_path` is used; ignored for `synthetic`.
    """

    csv_path: str | None = None
    comsol_path: str | None = None
    comsol_expression_prefix: str = "es"
    synthetic: SyntheticFieldConfig | None = None
    smoothing: float = 0.0


@dataclass(frozen=True)
class TrapConfig:
    """Harmonic trap parameters (``trap:`` in YAML).

    Attributes
    ----------
    omega_xyz : tuple[float, float, float]
        Angular trap frequencies, rad/s.
    center : tuple[float, float, float]
        Trap center, meters.
    """

    omega_xyz: tuple[float, float, float]
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)


#: Valid `EnsembleConfig.regime` values.
VALID_ENSEMBLE_REGIMES: tuple[str, ...] = ("classical", "lattice", "lattice_extended")


@dataclass(frozen=True)
class EnsembleConfig:
    """Ensemble sampling parameters (``ensemble:`` in YAML).

    Attributes
    ----------
    regime : str
        ``"classical"`` (Maxwell-Boltzmann Monte-Carlo + Verlet trajectory,
        :mod:`cliffordclock.ensemble.classical`), ``"lattice"``
        (Hermite-Gauss motional-state quadrature,
        :mod:`cliffordclock.ensemble.lattice`), or ``"lattice_extended"``
        (WP22 Part 2: `n_sites` copies of the `lattice` regime's local
        Hermite-Gauss quadrature distributed along `site_axis` -- the
        extended-sample mode CONVENTIONS.md section 15/`environment.gravity`
        (E36) targets, :func:`~cliffordclock.ensemble.lattice.extended_lattice_nodes`).
    temperature_uK : float
        Ensemble temperature, microkelvin.
    size : int or None
        Number of Monte-Carlo particles; required for `regime="classical"`.
    motional_n : tuple[int, int, int] or None
        Motional quantum numbers; required for `regime="lattice"` and
        `regime="lattice_extended"` (applied identically at every site in
        the latter case).
    n_quad : int
        Gauss-Hermite quadrature points per axis (`regime="lattice"`/
        `"lattice_extended"` only).
    seed : int
        PRNG seed for `regime="classical"` sampling.
    n_sites : int or None
        Number of extended-lattice sites along `site_axis`; required (and
        must be `>= 1`) for `regime="lattice_extended"`, ignored otherwise.
    site_spacing_m : float or None
        Center-to-center spacing between adjacent sites, meters; required
        (and must be `> 0`) for `regime="lattice_extended"`, ignored
        otherwise.
    site_axis : tuple[float, float, float]
        Direction sites are distributed along (`regime="lattice_extended"`
        only; need not be pre-normalized). Default ``(0, 0, 1)``.
    site_envelope : str
        Site-occupation envelope kind, `regime="lattice_extended"` only:
        ``"gaussian"`` (default) or ``"uniform"``; see
        :data:`~cliffordclock.ensemble.lattice.VALID_SITE_ENVELOPES`.
    site_envelope_sigma_m : float or None
        Gaussian envelope standard deviation, meters; required (and must
        be `> 0`) when `regime="lattice_extended"` and
        `site_envelope="gaussian"`, ignored otherwise.
    """

    regime: str
    temperature_uK: float
    size: int | None = None
    motional_n: tuple[int, int, int] | None = None
    n_quad: int = 8
    seed: int = 0
    n_sites: int | None = None
    site_spacing_m: float | None = None
    site_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    site_envelope: str = "gaussian"
    site_envelope_sigma_m: float | None = None


#: Valid ``integration.mode`` values, keyed by ``ensemble.regime``.
#: ``"auto"`` (the default) resolves to the first entry for the run's
#: regime. See :func:`_resolve_integration_mode`.
VALID_INTEGRATION_MODES_BY_REGIME: dict[str, tuple[str, ...]] = {
    "lattice": ("fast_path", "worldline"),
    "lattice_extended": ("fast_path", "worldline"),
    "classical": ("direct", "secular"),
}
VALID_INTEGRATION_MODES: tuple[str, ...] = (
    "auto",
    "fast_path",
    "worldline",
    "direct",
    "secular",
)

#: Default ceiling, in GB (10^9 bytes), on the pre-flight estimate of the
#: dominant dense-trajectory allocation for the time-stepping modes
#: (``"direct"``/``"worldline"``) -- see :func:`_check_trajectory_memory`.
#: 2 GB comfortably covers every example config and test in this repo
#: (all well under 100 MB) while stopping the silent 100+ GB allocations
#: an auto-selected E31 `dtau` over a long ``integration.time_s`` can
#: otherwise attempt (M x (steps+1) x 3 float64 positions with `steps`
#: reaching 1e6-1e7 -- enough to lock up a host mid-run). Under
#: ``integration.evaluation: auto`` (the default, WP19) this ceiling no
#: longer means "reject the config": for ``ensemble.regime=classical`` +
#: ``mode=direct``, exceeding it now means "run the memory-bounded
#: streaming accumulator instead of the fast batched one" -- see
#: :func:`_resolve_evaluation_mode` and ``docs/timescales.md``. It keeps
#: its original hard-reject meaning for every mode WP19 does not stream
#: (``worldline``, ``secular``) and for ``integration.evaluation: batched``
#: explicitly requested.
DEFAULT_MAX_TRAJECTORY_MEMORY_GB = 2.0

#: Valid ``integration.evaluation`` values (WP19). ``"auto"`` (default):
#: run the fast batched path when the pre-flight estimate
#: (:func:`_check_trajectory_memory`'s formula) fits
#: ``integration.max_trajectory_memory_gb``, else switch to the
#: memory-bounded streaming accumulator (a note is recorded in the
#: report's ``uncertainty_notes``). ``"batched"``: always run the batched
#: path, preserving the pre-WP19 hard-reject behavior of
#: ``max_trajectory_memory_gb`` (:class:`PipelineConfigError` if the
#: estimate is over budget). ``"streaming"``: always run the streaming
#: path, regardless of the batched-path estimate (e.g. to force a
#: memory-bounded run of a config that *would* fit batched, for
#: measurement/comparison purposes). Only ``ensemble.regime="classical"``
#: + ``integration.mode="direct"`` has a streaming accumulator today (WP19
#: scope: the ``worldline``/``secular``/``fast_path`` modes are unaffected
#: by this key and always run their existing batched path -- see
#: ``docs/timescales.md``). See :func:`_resolve_evaluation_mode`.
VALID_EVALUATION_MODES: tuple[str, ...] = ("auto", "batched", "streaming")


@dataclass(frozen=True)
class IntegrationConfig:
    """Rotor/fast-path integration parameters (``integration:`` in YAML).

    See ``docs/timescales.md`` for the three-tier architecture
    (CONVENTIONS.md section 12) this configures.

    Attributes
    ----------
    mode : str
        One of ``"auto"`` (default; resolves to ``"fast_path"`` for
        ``ensemble.regime: lattice``, ``"direct"`` for
        ``ensemble.regime: classical``), ``"fast_path"`` (E29, lattice
        only), ``"worldline"`` (the rotor integrator, lattice-only
        explicit cross-check), ``"direct"`` (the rotor integrator,
        classical only -- Tier B(i)), or ``"secular"`` (E30, classical
        only, periodic/isotropic-trap motion -- Tier B(ii)).
    dtau : float or None
        Fixed Compton-unit step size ``dτ̃`` (E9) for `mode` values that
        time-step (``"direct"``/``"worldline"``). Also drives the
        classical-ensemble Verlet trajectory step (module docstring,
        interface note 2). If omitted for `mode="direct"`, auto-selected
        via :func:`cliffordclock.integrator.fastpath.select_dtau` (E31).
    steps : int or None
        Number of integration steps, for `mode` values that time-step. If
        omitted (with `time_s` given), computed from `time_s` and `dtau`.
    time_s : float or None
        Direct interrogation-time input, seconds.
        Required for `mode="fast_path"`/`"secular"`; for
        `mode="direct"`/`"worldline"` it is an alternative to specifying
        `steps` directly. At least one of `time_s` or (`dtau` and
        `steps`) must be given.
    points_per_period : int
        E31's ``N_res``: trap-period resolution used by the auto-selected
        `dtau` (``mode="direct"``) and by `mode="secular"`'s internal
        one-orbit quadrature. Default 100 (E31).
    renorm_every : int
        Rotor renormalization cadence (E20), for `mode` values that
        advance a rotor (``"direct"``/``"worldline"``). Defaults to
        :data:`~cliffordclock.integrator.worldline.DEFAULT_RENORM_EVERY`
        (1000, tuned for Compton-scale `dtau`); when `dtau` is
        auto-selected (E31) *and* this field was left at its YAML default
        (`renorm_every_was_explicit` False), :func:`_resolve_renorm_every`
        auto-tightens it instead -- an explicit value
        here is always honored unchanged.
    renorm_every_was_explicit : bool
        Whether `renorm_every` was explicitly given in the source config
        (as opposed to defaulted). Not itself a YAML field; derived by
        :func:`_parse_integration` from whether the ``renorm_every`` key
        was present. See `renorm_every`.
    max_trajectory_memory_gb : float
        Ceiling, GB (10^9 bytes), on the pre-flight estimate of the dense
        ``(M, steps + 1, 3)`` trajectory allocation the time-stepping
        modes (``"direct"``/``"worldline"``) materialize. For
        ``ensemble.regime="classical"`` + ``mode="direct"`` under
        ``evaluation="auto"`` (default, WP19), exceeding it switches to
        the streaming accumulator instead of raising; every other
        mode/`evaluation` combination still rejects with
        :class:`PipelineConfigError` *before* anything is allocated
        (:func:`_check_trajectory_memory`). Default
        :data:`DEFAULT_MAX_TRAJECTORY_MEMORY_GB` (2 GB). The fast-path
        modes (``"fast_path"``/``"secular"``) never materialize a dense
        trajectory and ignore this field.
    evaluation : str
        One of :data:`VALID_EVALUATION_MODES` (``"auto"`` default,
        ``"batched"``, ``"streaming"``, WP19) -- selects between the fast
        batched accumulator and the O(M)-memory streaming accumulator for
        ``ensemble.regime="classical"`` + ``mode="direct"``; ignored by
        every other mode. See :data:`VALID_EVALUATION_MODES` and
        :func:`_resolve_evaluation_mode`.
    trajectory_stride : int or None
        WP19: for the streaming accumulator only (`evaluation` resolves
        to ``"streaming"``), how often (in steps) to retain a position
        snapshot in :class:`PipelineResult`'s `trajectories` output, at a
        memory cost of ``O(M * steps / trajectory_stride)`` -- still
        bounded, but no longer strictly O(M), so this is opt-in. `None`
        (default): only the initial and final positions are retained
        (`O(M)`, the streaming path's normal case) -- sufficient for the
        primary scalar-phase/report output, which never depends on
        `trajectories`; diagnostics that need denser trajectories (plots,
        notebooks) should set this explicitly or use
        ``evaluation="batched"`` (:func:`_check_trajectory_memory`
        permitting) for the full dense array. Ignored when `evaluation`
        resolves to ``"batched"`` (the batched path always returns the
        full dense trajectory, as before WP19).
    """

    dtau: float | None = None
    steps: int | None = None
    time_s: float | None = None
    mode: str = "auto"
    points_per_period: int = fastpath.DEFAULT_POINTS_PER_PERIOD
    renorm_every: int = DEFAULT_RENORM_EVERY
    renorm_every_was_explicit: bool = False
    max_trajectory_memory_gb: float = DEFAULT_MAX_TRAJECTORY_MEMORY_GB
    evaluation: str = "auto"
    trajectory_stride: int | None = None


#: Valid ``coupling.type`` values (see ``docs/coupling.md`` "API for the
#: pipeline plumbing"). ``linear_mu`` is the code-level default when
#: ``coupling.type`` is omitted from the YAML -- full backward
#: compatibility with every existing config, which writes
#: ``coupling: {mu: [...]}`` with no ``type`` key at all. Note:
#: ``docs/coupling.md`` separately calls ``stark_dc`` "the documented
#: default for new configs" -- a *prose* recommendation for what a human
#: should write in a new config, not a change to what an omitted ``type``
#: key resolves to.
VALID_COUPLING_TYPES: tuple[str, ...] = ("linear_mu", "stark_dc")


@dataclass(frozen=True)
class CouplingConfig:
    """Pivot coupling (``coupling:`` in YAML): E14a (linear) or E14b (quadratic DC-Stark).

    Attributes
    ----------
    type : str
        ``"linear_mu"`` (default) or ``"stark_dc"``; see
        :data:`VALID_COUPLING_TYPES`.
    mu : tuple[float, float, float] or None
        Explicit effective dipole moment ``μ`` (E14a), C·m. Required (and
        only meaningful) for ``type="linear_mu"``.
    delta_alpha_dc_si : float or None
        Optional explicit override of the transition's differential
        static scalar polarizability ``Δα`` (E14b), C²m²J⁻¹. ``type="stark_dc"``
        only; when omitted (along with `stark_coefficient_hz_per_v2_m2`),
        the coefficient is resolved from the species registry instead
        (``cliffordclock.ensemble.species.get_species(species).resolve_stark_coefficient_hz_per_v2_m2()``).
    stark_coefficient_hz_per_v2_m2 : float or None
        Optional explicit override of the equivalent Stark coefficient
        ``k_S`` (E14b), Hz·m²·V⁻² -- an alternative to `delta_alpha_dc_si`.
        ``type="stark_dc"`` only.
    """

    type: str = "linear_mu"
    mu: tuple[float, float, float] | None = None
    delta_alpha_dc_si: float | None = None
    stark_coefficient_hz_per_v2_m2: float | None = None


@dataclass(frozen=True)
class GravityConfig:
    """Gravitational-redshift pivot-term parameters (``environment.gravity:``
    in YAML, WP22 Part 1, CONVENTIONS.md section 15 E36).

    Attributes
    ----------
    g_m_s2 : float
        Local gravitational acceleration, m/s². Defaults to
        `cliffordclock.constants.STANDARD_GRAVITY` (9.80665 m/s², exact by
        international definition) -- a placeholder at the 1e-19 level: the
        physically correct input for a real 1e-19-class comparison is the
        LAB'S OWN SURVEYED LOCAL value (CONVENTIONS.md section 15 / G9
        sign-off B1; e.g. Boulder, CO's 9.796 m/s², see
        `benchmarks/run_bothwell_redshift.py`), which differs from
        standard gravity by parts in 1e3.
    up_axis : tuple[float, float, float]
        Direction of increasing height; need not be pre-normalized
        (normalized internally by
        :func:`~cliffordclock.integrator.omega.height_along_axis`). Default
        ``(0, 0, 1)``.
    reference_height_m : float
        The height (along `up_axis`, from the coordinate origin) at which
        ``(P−1)_grav`` is defined to be exactly zero. Default `0.0`: height
        is then measured directly from the coordinate origin -- for a
        `regime="lattice_extended"` ensemble whose `trap.center` is at the
        origin (the common case), this makes the ensemble's own geometric
        center the reference height with no extra configuration.
    """

    g_m_s2: float = STANDARD_GRAVITY
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    reference_height_m: float = 0.0


@dataclass(frozen=True)
class EnvironmentConfig:
    """Thermal-bath / environment parameters (``environment:`` in YAML, WP20/WP22).

    CONVENTIONS.md E32/E33 (blackbody-radiation shift, uniform-T MVP) and
    section 15 E36 (gravitational redshift, WP22). See the module
    docstring's "Blackbody-radiation shift" interface note for how BBR
    composes into every evaluation mode; WP22's module-docstring note
    describes the gravity term's identical threading pattern.

    Attributes
    ----------
    radiation_temperature_k : float or None
        BBR ambient radiation temperature, kelvin. `None` (the default,
        and the key's absence in YAML): BBR is off -- every existing
        example config, and every ``coupling.type="linear_mu"`` config,
        is completely unaffected (byte-identical output;
        ``tests/test_e2e.py``'s byte-exactness tests cover this). When
        given, `coupling.type` must be ``"stark_dc"``
        (:class:`PipelineConfig`'s cross-field validation raises
        :class:`PipelineConfigError` otherwise -- BBR needs the species'
        registry `~cliffordclock.ensemble.species.BbrCoefficients`, not a
        coupling-agnostic quantity) and the value must lie in
        ``[BBR_VALIDITY_MIN_K, BBR_VALIDITY_MAX_K]`` = ``[50, 350]``
        kelvin (the published fit range, dossier Sec.2/Sec.7; G7 sign-off
        gate edit 5: hard rejection, not a silently-extrapolated fit).
    radiation_temperature_uncertainty_k : float or None
        Optional 1-sigma uncertainty on `radiation_temperature_k`, kelvin;
        requires `radiation_temperature_k` to be set. When given,
        propagated through the BBR polynomial's exact derivative into the
        reported BBR uncertainty (G7 sign-off A4#3,
        :func:`~cliffordclock.integrator.omega.bbr_pivot_uncertainty`);
        when omitted, the report states its BBR uncertainty is
        "conditional on exact T" instead of silently claiming exactness
        (G7 sign-off: "silent exactness is not defensible at 1e-19").
    gravity : GravityConfig or None
        Gravitational-redshift pivot-term parameters (WP22 Part 1,
        CONVENTIONS.md section 15 E36). `None` (the default, and the
        key's absence in YAML): gravity is off -- every existing example
        config is completely unaffected (byte-identical output). When
        given, `coupling.type` must be ``"stark_dc"`` (mirrors
        `radiation_temperature_k`'s cross-field validation exactly -- the
        gravity term is composed at the same E14b rate-function call
        sites as BBR/the quadrupole term, G9 sign-off "mirrors BBR
        exactly").
    """

    radiation_temperature_k: float | None = None
    radiation_temperature_uncertainty_k: float | None = None
    gravity: GravityConfig | None = None


#: Valid `QuadrupoleConfig.averaging_mode` values.
VALID_QUADRUPOLE_AVERAGING_MODES: tuple[str, ...] = ("fixed", "three_orientation")


@dataclass(frozen=True)
class QuadrupoleConfig:
    """Electric-quadrupole shift parameters (``quadrupole:`` in YAML, WP21 Tier 2).

    CONVENTIONS.md E34/E35 (ion-clock electric-quadrupole shift). See the
    module docstring's WP21 interface note for how this composes into
    every evaluation mode, mirroring :class:`EnvironmentConfig`'s WP20 BBR
    pattern. Named as a top-level section (not nested under ``coupling:``)
    since the coefficients it needs (`Theta`, `J`) are independent of the
    DC-Stark coupling's `Delta_alpha`/`k_S` -- a deliberate deviation from
    the WP21 instruction file's literal "``coupling.quadrupole``" phrasing
    (itself hedged: "or an ion-species-implied term"), documented as an
    AMBIGUITY in the WP21 builder report.

    Attributes
    ----------
    state : str or None
        A `cliffordclock.ensemble.species.QUADRUPOLE_MOMENTS` registry key
        (e.g. ``"Ca+:D5/2"``), supplying `theta_au`/`j` from the registry.
        Mutually exclusive with `theta_au`/`j` given explicitly; exactly
        one of (`state`) or (`theta_au` and `j`) must be given.
    theta_au : float or None
        Explicit electric-quadrupole moment Theta(J), atomic units
        (= e*a0^2); overrides the registry when given directly (mirrors
        `StarkCoefficients`'s "explicit override" pattern for a state not
        in `QUADRUPOLE_MOMENTS`). Requires `j` to also be given.
    j : float or None
        Total angular momentum J of the state; required with `theta_au`,
        ignored (derived from the registry) when `state` is given.
    nu_0_hz : float
        Clock transition frequency, hertz -- the SAME transition whose
        upper state has this quadrupole moment (CONVENTIONS.md E35). Not
        looked up from the run's `species:` (WP21 Tier 2's D/F-state ions
        are not registered `Species` entries -- see `QuadrupoleMoment`'s
        docstring); an explicit, required input.
    m_j : float or None
        Magnetic quantum number for a FIXED-axis evaluation. Required when
        `averaging_mode="fixed"`; ignored (every `m_j` cancels identically)
        when `averaging_mode="three_orientation"`.
    quantization_axis : tuple[float, float, float]
        Quantization-axis direction (need not be pre-normalized). Default
        ``(0, 0, 1)``. Ignored when `averaging_mode="three_orientation"`
        (CONVENTIONS.md E35 A2: the cancellation identity is independent
        of the gradient's -- and hence any single axis choice's --
        orientation; the three-orientation evaluator always uses its own
        internal orthonormal triad,
        `cliffordclock.integrator.omega._STANDARD_TRIAD`).
    averaging_mode : str
        ``"fixed"`` (default): evaluate at the single `(m_j,
        quantization_axis)` given. ``"three_orientation"``: the standard
        three-mutually-perpendicular-orientation averaging identity
        (CONVENTIONS.md E35 A2) -- composes the EXACT (not merely
        averaged-over-samples) cancellation, i.e. contributes `0.0` to
        every evaluation mode's `(P-1)_Q`.
    """

    nu_0_hz: float
    state: str | None = None
    theta_au: float | None = None
    j: float | None = None
    m_j: float | None = None
    quantization_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    averaging_mode: str = "fixed"


@dataclass(frozen=True)
class OutputConfig:
    """Output file destinations (``output:`` in YAML).

    Attributes
    ----------
    directory : str
        Output directory (created if missing).
    report_filename : str
        JSON report filename, written by the CLI.
    line_profile_filename : str
        Line-profile CSV filename, written by the CLI.
    site_map_filename : str
        Per-site frequency map JSON filename (WP22 Part 2,
        :class:`LatticeExtendedSiteMap`), written by the CLI ALONGSIDE
        `report_filename` when the run's `ensemble.regime` is
        ``"lattice_extended"`` (``PipelineResult.site_map`` is not `None`);
        not written at all for every other regime.
    n_time_samples : int
        Number of uniformly spaced time samples across
        ``[0, T_interrogation]`` at which the coherence function (E26) and
        line profile (E28) are evaluated.
    """

    directory: str = "."
    report_filename: str = "report.json"
    line_profile_filename: str = "line_profile.csv"
    site_map_filename: str = "site_map.json"
    n_time_samples: int = 512


@dataclass(frozen=True)
class PipelineConfig:
    """The full ``cliffordclock run`` configuration (see ``docs/cli.md`` for the YAML schema).

    Attributes
    ----------
    species : str
        Atomic species registry name (``cliffordclock.ensemble.species.get_species``).
    trap : TrapConfig
    field_config : FieldConfig
        Field source (the ``field:`` YAML section; named `field_config`,
        not `field`, to avoid shadowing `dataclasses.field` within this
        class body).
    ensemble : EnsembleConfig
    integration : IntegrationConfig
    coupling : CouplingConfig
    environment : EnvironmentConfig
        Thermal-bath parameters (``environment:`` in YAML, WP20); see that
        class's docstring. Defaults to BBR off.
    quadrupole : QuadrupoleConfig or None
        Electric-quadrupole shift parameters (``quadrupole:`` in YAML,
        WP21 Tier 2); see :class:`QuadrupoleConfig`'s docstring. `None`
        (the default, and the key's absence in YAML): the quadrupole term
        is off -- every existing example config is completely unaffected
        (byte-identical output). Requires `coupling.type='stark_dc'` (same
        reasoning as `environment`: the composition point is the E14b
        Stark rate function).
    output : OutputConfig
    uncertainty_notes : str
        Free-text note forwarded to ``MetrologyReport.uncertainty_notes``.
    """

    species: str
    trap: TrapConfig
    field_config: FieldConfig
    ensemble: EnsembleConfig
    integration: IntegrationConfig
    coupling: CouplingConfig
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    quadrupole: QuadrupoleConfig | None = None
    output: OutputConfig = field(default_factory=OutputConfig)
    uncertainty_notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load and validate a :class:`PipelineConfig` from a YAML file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a ``config.yaml`` (see ``docs/cli.md``).

        Returns
        -------
        PipelineConfig

        Raises
        ------
        PipelineConfigError
            The file cannot be read, is not valid YAML, is not a mapping
            at the top level, or fails schema validation (see
            :meth:`from_dict`).
        """
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelineConfigError(f"cannot read config file {path}: {exc}") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PipelineConfigError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise PipelineConfigError(f"{path}: config must be a YAML mapping at the top level")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Build and validate a :class:`PipelineConfig` from a parsed mapping.

        Parameters
        ----------
        data : dict[str, Any]
            Parsed config (e.g. from ``yaml.safe_load``).

        Returns
        -------
        PipelineConfig

        Raises
        ------
        PipelineConfigError
            A required field is missing, has the wrong shape/type, or
            has an invalid value (e.g. unknown ``ensemble.regime``,
            non-positive ``integration.dtau``).
        """
        species = str(_require(data, "species", "config"))
        trap_cfg = _parse_trap(_require(data, "trap", "config"))
        field_cfg = _parse_field(_require(data, "field", "config"))
        ensemble_cfg = _parse_ensemble(_require(data, "ensemble", "config"))
        integration_cfg = _parse_integration(_require(data, "integration", "config"))
        coupling_cfg = _parse_coupling(_require(data, "coupling", "config"))
        environment_cfg = _parse_environment(data.get("environment"))
        quadrupole_cfg = _parse_quadrupole(data.get("quadrupole"))
        output_cfg = _parse_output(data.get("output"))
        uncertainty_notes = str(data.get("uncertainty_notes", ""))
        if environment_cfg.radiation_temperature_k is not None and coupling_cfg.type != "stark_dc":
            raise PipelineConfigError(
                "environment.radiation_temperature_K requires coupling.type='stark_dc' "
                f"(got coupling.type={coupling_cfg.type!r}): the BBR shift (CONVENTIONS.md "
                "E32) is composed into the species' registry-resolved DC-Stark polarizability "
                "data, which coupling.type='linear_mu' has no equivalent of."
            )
        if quadrupole_cfg is not None and coupling_cfg.type != "stark_dc":
            raise PipelineConfigError(
                "quadrupole: requires coupling.type='stark_dc' "
                f"(got coupling.type={coupling_cfg.type!r}): the quadrupole shift "
                "(CONVENTIONS.md E34/E35) is composed at the E14b Stark rate function's "
                "call sites, which coupling.type='linear_mu' does not use."
            )
        if environment_cfg.gravity is not None and coupling_cfg.type != "stark_dc":
            raise PipelineConfigError(
                "environment.gravity requires coupling.type='stark_dc' "
                f"(got coupling.type={coupling_cfg.type!r}): the gravitational-redshift term "
                "(CONVENTIONS.md section 15 E36) is composed at the same E14b Stark "
                "rate-function call sites as environment.radiation_temperature_K/BBR "
                "(G9 sign-off: 'E36 mirrors BBR exactly'), which coupling.type='linear_mu' "
                "does not use."
            )
        return cls(
            species=species,
            trap=trap_cfg,
            field_config=field_cfg,
            ensemble=ensemble_cfg,
            integration=integration_cfg,
            coupling=coupling_cfg,
            environment=environment_cfg,
            quadrupole=quadrupole_cfg,
            output=output_cfg,
            uncertainty_notes=uncertainty_notes,
        )


# ---------------------------------------------------------------------------
# Config parsing helpers.
# ---------------------------------------------------------------------------


def _require(data: dict[str, Any], key: str, context: str) -> Any:
    if not isinstance(data, dict):
        raise PipelineConfigError(f"{context}: must be a mapping, got {type(data).__name__}")
    if key not in data or data[key] is None:
        raise PipelineConfigError(f"{context}: missing required field {key!r}")
    return data[key]


def _as_float_tuple3(value: Any, context: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise PipelineConfigError(f"{context}: expected a 3-element list, got {value!r}")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise PipelineConfigError(f"{context}: expected 3 numeric values, got {value!r}") from exc


def _as_int_tuple3(value: Any, context: str) -> tuple[int, int, int]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise PipelineConfigError(f"{context}: expected a 3-element list, got {value!r}")
    try:
        return (int(value[0]), int(value[1]), int(value[2]))
    except (TypeError, ValueError) as exc:
        raise PipelineConfigError(f"{context}: expected 3 integer values, got {value!r}") from exc


def _parse_trap(data: Any) -> TrapConfig:
    if not isinstance(data, dict):
        raise PipelineConfigError(f"trap: must be a mapping, got {type(data).__name__}")
    omega = _as_float_tuple3(_require(data, "omega_xyz", "trap"), "trap.omega_xyz")
    center = _as_float_tuple3(data.get("center", [0.0, 0.0, 0.0]), "trap.center")
    return TrapConfig(omega_xyz=omega, center=center)


def _parse_field(data: Any) -> FieldConfig:
    if not isinstance(data, dict):
        raise PipelineConfigError(f"field: must be a mapping, got {type(data).__name__}")
    csv_path = data.get("csv")
    comsol_path = data.get("comsol")
    synthetic_data = data.get("synthetic")
    n_sources = sum(source is not None for source in (csv_path, comsol_path, synthetic_data))
    if n_sources > 1:
        raise PipelineConfigError(
            "field: specify exactly one of 'csv', 'comsol', or 'synthetic', not more than one"
        )
    if n_sources == 0:
        raise PipelineConfigError("field: must specify one of 'csv', 'comsol', or 'synthetic'")
    smoothing = float(data.get("smoothing", 0.0))
    if smoothing < 0:
        raise PipelineConfigError(f"field.smoothing must be >= 0, got {smoothing}")
    if synthetic_data is not None:
        if not isinstance(synthetic_data, dict) or "kind" not in synthetic_data:
            raise PipelineConfigError("field.synthetic: must be a mapping with a 'kind' field")
        kind = str(synthetic_data["kind"])
        params = synthetic_data.get("params", {})
        if not isinstance(params, dict):
            raise PipelineConfigError("field.synthetic.params: must be a mapping")
        return FieldConfig(
            synthetic=SyntheticFieldConfig(kind=kind, params=dict(params)), smoothing=smoothing
        )
    if comsol_path is not None:
        expression_prefix = str(data.get("expression_prefix", "es"))
        return FieldConfig(
            comsol_path=str(comsol_path),
            comsol_expression_prefix=expression_prefix,
            smoothing=smoothing,
        )
    return FieldConfig(csv_path=str(csv_path), smoothing=smoothing)


def _parse_ensemble(data: Any) -> EnsembleConfig:
    if not isinstance(data, dict):
        raise PipelineConfigError(f"ensemble: must be a mapping, got {type(data).__name__}")
    regime = str(_require(data, "regime", "ensemble"))
    if regime not in VALID_ENSEMBLE_REGIMES:
        raise PipelineConfigError(
            f"ensemble.regime must be one of {VALID_ENSEMBLE_REGIMES}, got {regime!r}"
        )
    temperature_uK = float(_require(data, "temperature_uK", "ensemble"))
    size_raw = data.get("size")
    size = int(size_raw) if size_raw is not None else None
    motional_n_raw = data.get("motional_n")
    motional_n = (
        _as_int_tuple3(motional_n_raw, "ensemble.motional_n")
        if motional_n_raw is not None
        else None
    )
    n_quad = int(data.get("n_quad", 8))
    seed = int(data.get("seed", 0))
    n_sites_raw = data.get("n_sites")
    n_sites = int(n_sites_raw) if n_sites_raw is not None else None
    site_spacing_raw = data.get("site_spacing_m")
    site_spacing_m = float(site_spacing_raw) if site_spacing_raw is not None else None
    site_axis = _as_float_tuple3(data.get("site_axis", [0.0, 0.0, 1.0]), "ensemble.site_axis")
    site_envelope = str(data.get("site_envelope", "gaussian"))
    site_envelope_sigma_raw = data.get("site_envelope_sigma_m")
    site_envelope_sigma_m = (
        float(site_envelope_sigma_raw) if site_envelope_sigma_raw is not None else None
    )

    if regime == "classical":
        if size is None or size < 1:
            raise PipelineConfigError("ensemble.size is required (>= 1) when regime='classical'")
    elif regime == "lattice":
        if motional_n is None:
            raise PipelineConfigError("ensemble.motional_n is required when regime='lattice'")
        if n_quad < 1:
            raise PipelineConfigError(f"ensemble.n_quad must be >= 1, got {n_quad}")
    elif regime == "lattice_extended":
        if motional_n is None:
            raise PipelineConfigError(
                "ensemble.motional_n is required when regime='lattice_extended' "
                "(applied identically at every site)"
            )
        if n_quad < 1:
            raise PipelineConfigError(f"ensemble.n_quad must be >= 1, got {n_quad}")
        if n_sites is None or n_sites < 1:
            raise PipelineConfigError(
                "ensemble.n_sites is required (>= 1) when regime='lattice_extended'"
            )
        if site_spacing_m is None or site_spacing_m <= 0:
            raise PipelineConfigError(
                "ensemble.site_spacing_m is required (> 0) when regime='lattice_extended'"
            )
        if all(component == 0.0 for component in site_axis):
            raise PipelineConfigError("ensemble.site_axis must not be the zero vector")
        if site_envelope not in VALID_SITE_ENVELOPES:
            raise PipelineConfigError(
                f"ensemble.site_envelope must be one of {VALID_SITE_ENVELOPES}, "
                f"got {site_envelope!r}"
            )
        if site_envelope == "gaussian" and (
            site_envelope_sigma_m is None or site_envelope_sigma_m <= 0
        ):
            raise PipelineConfigError(
                "ensemble.site_envelope_sigma_m is required (> 0) when "
                "regime='lattice_extended' and site_envelope='gaussian'"
            )

    return EnsembleConfig(
        regime=regime,
        temperature_uK=temperature_uK,
        size=size,
        motional_n=motional_n,
        n_quad=n_quad,
        seed=seed,
        n_sites=n_sites,
        site_spacing_m=site_spacing_m,
        site_axis=site_axis,
        site_envelope=site_envelope,
        site_envelope_sigma_m=site_envelope_sigma_m,
    )


def _parse_integration(data: Any) -> IntegrationConfig:
    if not isinstance(data, dict):
        raise PipelineConfigError(f"integration: must be a mapping, got {type(data).__name__}")

    mode = str(data.get("mode", "auto"))
    if mode not in VALID_INTEGRATION_MODES:
        raise PipelineConfigError(
            f"integration.mode must be one of {VALID_INTEGRATION_MODES}, got {mode!r}"
        )

    dtau_raw = data.get("dtau")
    dtau = float(dtau_raw) if dtau_raw is not None else None
    steps_raw = data.get("steps")
    steps = int(steps_raw) if steps_raw is not None else None
    time_s_raw = data.get("time_s")
    time_s = float(time_s_raw) if time_s_raw is not None else None
    points_per_period = int(data.get("points_per_period", fastpath.DEFAULT_POINTS_PER_PERIOD))
    renorm_every_raw = data.get("renorm_every")
    renorm_every_was_explicit = renorm_every_raw is not None
    renorm_every = int(renorm_every_raw) if renorm_every_raw is not None else DEFAULT_RENORM_EVERY
    max_traj_raw = data.get("max_trajectory_memory_gb")
    max_trajectory_memory_gb = (
        float(max_traj_raw) if max_traj_raw is not None else DEFAULT_MAX_TRAJECTORY_MEMORY_GB
    )
    evaluation = str(data.get("evaluation", "auto"))
    if evaluation not in VALID_EVALUATION_MODES:
        raise PipelineConfigError(
            f"integration.evaluation must be one of {VALID_EVALUATION_MODES}, got {evaluation!r}"
        )
    trajectory_stride_raw = data.get("trajectory_stride")
    trajectory_stride = int(trajectory_stride_raw) if trajectory_stride_raw is not None else None
    if trajectory_stride is not None and trajectory_stride < 1:
        raise PipelineConfigError(
            f"integration.trajectory_stride must be >= 1, got {trajectory_stride}"
        )

    if dtau is not None and dtau <= 0:
        raise PipelineConfigError(f"integration.dtau must be positive, got {dtau}")
    if steps is not None and steps < 1:
        raise PipelineConfigError(f"integration.steps must be >= 1, got {steps}")
    if time_s is not None and time_s <= 0:
        raise PipelineConfigError(f"integration.time_s must be positive, got {time_s}")
    if time_s is None and (dtau is None or steps is None):
        raise PipelineConfigError(
            "integration: must specify either 'time_s', or both 'dtau' and 'steps'"
        )
    if points_per_period < 1:
        raise PipelineConfigError(
            f"integration.points_per_period must be >= 1, got {points_per_period}"
        )
    if renorm_every < 1:
        raise PipelineConfigError(f"integration.renorm_every must be >= 1, got {renorm_every}")
    if not max_trajectory_memory_gb > 0:
        raise PipelineConfigError(
            f"integration.max_trajectory_memory_gb must be positive, got {max_trajectory_memory_gb}"
        )

    return IntegrationConfig(
        dtau=dtau,
        steps=steps,
        time_s=time_s,
        mode=mode,
        points_per_period=points_per_period,
        renorm_every=renorm_every,
        renorm_every_was_explicit=renorm_every_was_explicit,
        max_trajectory_memory_gb=max_trajectory_memory_gb,
        evaluation=evaluation,
        trajectory_stride=trajectory_stride,
    )


def _parse_coupling(data: Any) -> CouplingConfig:
    if not isinstance(data, dict):
        raise PipelineConfigError(f"coupling: must be a mapping, got {type(data).__name__}")
    coupling_type = str(data.get("type", "linear_mu"))
    if coupling_type not in VALID_COUPLING_TYPES:
        raise PipelineConfigError(
            f"coupling.type must be one of {VALID_COUPLING_TYPES}, got {coupling_type!r}"
        )
    if coupling_type == "linear_mu":
        mu = _as_float_tuple3(_require(data, "mu", "coupling"), "coupling.mu")
        return CouplingConfig(type=coupling_type, mu=mu)

    assert coupling_type == "stark_dc"
    delta_alpha_raw = data.get("delta_alpha_dc_si")
    stark_coeff_raw = data.get("stark_coefficient_hz_per_v2_m2")
    try:
        delta_alpha = float(delta_alpha_raw) if delta_alpha_raw is not None else None
        stark_coeff = float(stark_coeff_raw) if stark_coeff_raw is not None else None
    except (TypeError, ValueError) as exc:
        raise PipelineConfigError(
            "coupling.delta_alpha_dc_si/stark_coefficient_hz_per_v2_m2 must be numeric, got "
            f"delta_alpha_dc_si={delta_alpha_raw!r} "
            f"stark_coefficient_hz_per_v2_m2={stark_coeff_raw!r}"
        ) from exc
    return CouplingConfig(
        type=coupling_type,
        delta_alpha_dc_si=delta_alpha,
        stark_coefficient_hz_per_v2_m2=stark_coeff,
    )


def _parse_environment(data: Any) -> EnvironmentConfig:
    """Parse the optional ``environment:`` YAML section (WP20, :class:`EnvironmentConfig`).

    `data` is `None` when the key is absent from the config (the default,
    common case) -- returns `EnvironmentConfig()` (BBR off), matching
    every other optional-section parser's ``data.get(...)`` -> `None` ->
    default-value pattern in this module.
    """
    if data is None:
        return EnvironmentConfig()
    if not isinstance(data, dict):
        raise PipelineConfigError(f"environment: must be a mapping, got {type(data).__name__}")

    temperature_raw = data.get("radiation_temperature_K")
    temperature_k = float(temperature_raw) if temperature_raw is not None else None
    uncertainty_raw = data.get("radiation_temperature_uncertainty_K")
    temperature_uncertainty_k = float(uncertainty_raw) if uncertainty_raw is not None else None

    if temperature_uncertainty_k is not None and temperature_k is None:
        raise PipelineConfigError(
            "environment.radiation_temperature_uncertainty_K requires "
            "environment.radiation_temperature_K to also be set"
        )
    if temperature_k is not None and not (
        BBR_VALIDITY_MIN_K <= temperature_k <= BBR_VALIDITY_MAX_K
    ):
        raise PipelineConfigError(
            f"environment.radiation_temperature_K={temperature_k!r} K is outside the "
            f"validated BBR fit range [{BBR_VALIDITY_MIN_K}, {BBR_VALIDITY_MAX_K}] K "
            "(CONVENTIONS.md E32; G7 sign-off gate edit 5) -- hard rejection, not a "
            "silently-extrapolated fit past its published support."
        )
    if temperature_uncertainty_k is not None and temperature_uncertainty_k < 0:
        raise PipelineConfigError(
            "environment.radiation_temperature_uncertainty_K must be >= 0, got "
            f"{temperature_uncertainty_k}"
        )

    gravity = _parse_gravity(data.get("gravity"))

    return EnvironmentConfig(
        radiation_temperature_k=temperature_k,
        radiation_temperature_uncertainty_k=temperature_uncertainty_k,
        gravity=gravity,
    )


def _parse_gravity(data: Any) -> GravityConfig | None:
    """Parse the optional ``environment.gravity:`` YAML section (WP22 Part 1,
    :class:`GravityConfig`).

    `data` is `None` when the key is absent (the default, common case) --
    returns `None` (gravity off), matching `_parse_environment`'s own
    ``data.get(...)`` -> `None` -> off pattern for `environment:` itself.
    """
    if data is None:
        return None
    if not isinstance(data, dict):
        raise PipelineConfigError(
            f"environment.gravity: must be a mapping, got {type(data).__name__}"
        )

    g_m_s2 = float(data.get("g_m_s2", STANDARD_GRAVITY))
    if g_m_s2 <= 0:
        raise PipelineConfigError(f"environment.gravity.g_m_s2 must be > 0, got {g_m_s2!r}")
    up_axis = _as_float_tuple3(data.get("up_axis", [0.0, 0.0, 1.0]), "environment.gravity.up_axis")
    if all(component == 0.0 for component in up_axis):
        raise PipelineConfigError("environment.gravity.up_axis must not be the zero vector")
    reference_height_m = float(data.get("reference_height_m", 0.0))

    return GravityConfig(
        g_m_s2=g_m_s2,
        up_axis=up_axis,
        reference_height_m=reference_height_m,
    )


def _parse_quadrupole(data: Any) -> QuadrupoleConfig | None:
    """Parse the optional ``quadrupole:`` YAML section (WP21, :class:`QuadrupoleConfig`).

    `data` is `None` when the key is absent (the default, common case) --
    returns `None` (quadrupole off), matching `_parse_environment`'s
    ``data.get(...)`` -> `None` -> off pattern.
    """
    if data is None:
        return None
    if not isinstance(data, dict):
        raise PipelineConfigError(f"quadrupole: must be a mapping, got {type(data).__name__}")

    nu_0_hz = float(_require(data, "nu_0_hz", "quadrupole"))
    state = data.get("state")
    state = str(state) if state is not None else None
    theta_au_raw = data.get("theta_au")
    theta_au = float(theta_au_raw) if theta_au_raw is not None else None
    j_raw = data.get("j")
    j = float(j_raw) if j_raw is not None else None

    if state is not None and (theta_au is not None or j is not None):
        raise PipelineConfigError(
            "quadrupole: specify either 'state' (a QUADRUPOLE_MOMENTS registry key) or "
            "both 'theta_au' and 'j', not both forms"
        )
    if state is None:
        if theta_au is None or j is None:
            raise PipelineConfigError(
                "quadrupole: must specify either 'state' or both 'theta_au' and 'j'"
            )
    else:
        try:
            get_quadrupole_moment(state)
        except KeyError as exc:
            raise PipelineConfigError(str(exc)) from exc

    averaging_mode = str(data.get("averaging_mode", "fixed"))
    if averaging_mode not in VALID_QUADRUPOLE_AVERAGING_MODES:
        raise PipelineConfigError(
            "quadrupole.averaging_mode must be one of "
            f"{VALID_QUADRUPOLE_AVERAGING_MODES}, got {averaging_mode!r}"
        )
    m_j_raw = data.get("m_j")
    m_j = float(m_j_raw) if m_j_raw is not None else None
    if averaging_mode == "fixed" and m_j is None:
        raise PipelineConfigError(
            "quadrupole.m_j is required when quadrupole.averaging_mode='fixed'"
        )
    quantization_axis = _as_float_tuple3(
        data.get("quantization_axis", [0.0, 0.0, 1.0]), "quadrupole.quantization_axis"
    )
    if averaging_mode == "fixed" and math.sqrt(sum(c * c for c in quantization_axis)) == 0.0:
        raise PipelineConfigError("quadrupole.quantization_axis must be nonzero")

    return QuadrupoleConfig(
        nu_0_hz=nu_0_hz,
        state=state,
        theta_au=theta_au,
        j=j,
        m_j=m_j,
        quantization_axis=quantization_axis,
        averaging_mode=averaging_mode,
    )


def _parse_output(data: Any) -> OutputConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise PipelineConfigError(f"output: must be a mapping, got {type(data).__name__}")
    n_time_samples = int(data.get("n_time_samples", 512))
    if n_time_samples < 2:
        raise PipelineConfigError(f"output.n_time_samples must be >= 2, got {n_time_samples}")
    return OutputConfig(
        directory=str(data.get("directory", ".")),
        report_filename=str(data.get("report_filename", "report.json")),
        line_profile_filename=str(data.get("line_profile_filename", "line_profile.csv")),
        site_map_filename=str(data.get("site_map_filename", "site_map.json")),
        n_time_samples=n_time_samples,
    )


# ---------------------------------------------------------------------------
# Pipeline execution.
# ---------------------------------------------------------------------------

#: Synthetic field factories addressable from ``field.synthetic.kind``.
_SYNTHETIC_FIELD_FACTORIES: dict[str, Callable[..., tuple[SynthFieldFn, SynthGradFn]]] = {
    "uniform": uniform_field,
    "constant_gradient": constant_gradient_field,
    "quadrupole": quadrupole_field,
    "gaussian_bump": gaussian_bump_field,
}

#: Coarse rotor-norm-drift (E20) sanity threshold for :func:`_validate_physics`.
#: Well above the ~1e-15 floor a correctly configured run shows (see
#: ``tests/test_integrator_worldline.py::test_norm_preservation_one_million_steps``,
#: which requires < 1e-12 with the *default* renorm cadence) -- this is a
#: coarse CLI-facing trip-wire against a badly misconfigured run, not a
#: precision bound. No exact number for this check is pinned elsewhere;
#: it is chosen deliberately loose (see below).
MAX_ROTOR_NORM_ERROR = 1e-9

#: Documented bound on the *estimated* worst-case
#: per-step rotor generator angle -- the magnitude of the bivector
#: ``generator = (-0.5 * dtau) * Omega`` fed to
#: :func:`cliffordclock.cl13.exp_bivector` each step (E19,
#: ``integrator/stepper.py::rotor_step``). `exp_bivector` evaluates a
#: *scaled* Taylor series (halved 10 times, a 12-term truncation, then
#: repeatedly squared back up, ``cliffordclock.cl13.ops``); that fixed-
#: order scheme is only accurate -- and, at large enough arguments,
#: numerically stable at all -- well inside its convergence range.
#: Direct probing shows `exp_bivector` still
#: returns a *finite* but badly wrong rotor (``|R| ~ 1e56``, not a unit
#: rotor) at a generator angle of 5000 rad, and NaN/Inf by 10000 rad; 0.5
#: rad is chosen with four orders of magnitude of margin below that
#: failure onset -- a loud-failure trip-wire, not a tight bound. At
#: realistic Compton-scale `dtau` (the original validation regime) the
#: generator angle is
#: ``~1e-19`` or smaller and this bound is never approached; it only
#: starts to matter at E31's large auto-selected `dtau` combined with a
#: realistic E14a `mu` (see :func:`_resolve_dtau_steps_direct` and
#: ``docs/timescales.md``).
MAX_PER_STEP_ROTOR_ANGLE_RAD = 0.5

#: Conservative, documented per-call rotor-norm-drift floor of
#: `exp_bivector`'s fixed-order (12-term Taylor / 10-halving) evaluation
#: at E31-scale (large-`dtau`) generator angles: the step-size accuracy
#: study
#: (``docs/timescales.md``, Tier B(i) "Finding: `renorm_every` at large
#: dτ̃") measures ~8.9e-14 rotor-norm-drift per `exp_bivector` call in
#: that regime; this module's own reproduction measured up to ~1.02e-13
#: per call on other E31-scale configurations (lattice nodes, a localized
#: field). Rounded up to `2e-13` here -- roughly double the largest
#: measured value, not the raw measurement -- so :func:`_auto_renorm_every`
#: below keeps a comfortable margin rather than landing exactly on the
#: `_AUTO_RENORM_EVERY_DRIFT_BOUND` edge.
_EXP_BIVECTOR_LARGE_DTAU_NORM_FLOOR = 2e-13

#: E20's accumulated-rotor-norm-drift target this module's auto-selected
#: `renorm_every` is chosen to respect -- the same
#: `< 1e-12` bound ``tests/test_integrator_worldline.py``'s
#: ``test_norm_preservation_one_million_steps`` holds the *default*
#: cadence (`DEFAULT_RENORM_EVERY`, tuned for Compton-scale `dtau`) to.
_AUTO_RENORM_EVERY_DRIFT_BOUND = 1e-12

#: Multiplier applied on top of the single dense ``(M, steps + 1, 3)``
#: float64 position array in :func:`_check_trajectory_memory`'s pre-flight
#: estimate. The position trajectory is the dominant allocation, but it is
#: never alone: the classical Verlet sampler
#: (``ensemble.classical._verlet_trajectory``) builds a same-shape velocity
#: trajectory alongside it plus same-shape ``moveaxis``/``concatenate``
#: intermediates, and every accumulator's scan feeds on ``traj[:-1]``/
#: ``traj[1:]`` slice copies of the whole array. Up to ~4 same-shape
#: arrays can be live simultaneously on the worst path, so the estimate
#: charges 4x the base array -- a documented coarse multiplier for a
#: loud-failure trip-wire, not exact peak-RSS accounting.
_TRAJECTORY_MEMORY_FACTOR = 4.0

#: Multiplier on the *second* term of :func:`_check_trajectory_memory`'s
#: estimate -- the ``N x K x 3 x 8`` byte cost of evaluating a
#: :class:`~cliffordclock.fields.smoother.FieldSmoother`-backed field
#: (``field.csv``/``field.comsol``) at ``N = M x steps`` query points
#: against its ``K`` fitted RBF centers. This term only applies to
#: :func:`_resolve_dtau_steps_scalar` (`coupling.type='stark_dc'`,
#: `mode='direct'`): unlike every other accumulator in this module (which
#: calls `field_fn`/`rate_fn` once per step inside a `lax.scan`, so the
#: live smoother intermediate is only ``(M, K, 3)``),
#: :func:`_stark_scalar_ensemble`'s `run_one` calls `rate_fn` **once** on
#: an atom's *entire* `(steps, 3)` midpoint trajectory, `vmap`-ed over `M`
#: -- so `FieldSmoother.evaluate`'s internal `(N, K, 3)` `diffs`/`phi`
#: intermediates (`_field_at_point`, `vmap`-ed) are all live at once, and
#: `jax.jacfwd` (for `grad_E`, computed but unused by this rate_fn --
#: E14b's pivot needs only `E`, see :func:`_make_stark_rate_fn`) triples
#: that again. Calibrated from measured peak RSS at N=8000 (M x steps),
#: K=729 RBF fit points: naive ``N x K x 3 x 8`` bytes = 140 MB, measured
#: 1.56 GB -- a ~11.1x factor. 12 is a documented, slightly conservative
#: round-up of that measurement, in the same spirit as
#: :data:`_TRAJECTORY_MEMORY_FACTOR`: a loud-failure trip-wire, not exact
#: peak-RSS accounting. Deliberately not chased further: the real fix is
#: chunking `rate_fn`'s evaluation to bound memory independent of `N`,
#: deferred as follow-up work (not implemented here -- see
#: `docs/timescales.md`, "Safety net: the trajectory-memory guard").
# Reviewer calibration note: the actual/naive RSS ratio is N-dependent --
# measured 24x at N=500 falling monotonically to ~5.6x at N=100,000 -- so 12
# sits mid-range: mildly optimistic only in a narrow N~1000-3000 band where
# absolute allocations are <0.5 GB regardless, and increasingly conservative
# in the large-N regime this guard exists to defend against.
_TRAJECTORY_MEMORY_FACTOR_SMOOTHER = 12.0


def _estimate_trajectory_memory_gb(
    n_atoms: int,
    steps: int,
    *,
    n_smoother_fit_points: int | None = None,
) -> tuple[float, float, float]:
    """The batched-path memory estimate :func:`_check_trajectory_memory` checks (WP19).

    Factored out of :func:`_check_trajectory_memory` so the evaluation-mode
    dispatch (:func:`_resolve_evaluation_mode`) can query the same estimate
    *without* raising -- the auto-dispatch decision (switch to streaming
    vs. run batched) needs the number before deciding whether raising is
    even appropriate. See :func:`_check_trajectory_memory` for the formula
    documentation (unchanged, just relocated).

    Returns
    -------
    tuple[float, float, float]
        ``(trajectory_gb, smoother_gb, total_gb)``; `total_gb` is exactly
        their sum.
    """
    trajectory_gb = _TRAJECTORY_MEMORY_FACTOR * n_atoms * (steps + 1) * 3 * 8 / 1e9
    smoother_gb = 0.0
    n_query_points = n_atoms * steps
    if n_smoother_fit_points is not None:
        smoother_gb = (
            _TRAJECTORY_MEMORY_FACTOR_SMOOTHER * n_query_points * n_smoother_fit_points * 3 * 8
        ) / 1e9
    return trajectory_gb, smoother_gb, trajectory_gb + smoother_gb


def _check_trajectory_memory(
    n_atoms: int,
    steps: int,
    max_gb: float,
    *,
    n_smoother_fit_points: int | None = None,
) -> None:
    """Pre-flight resource guard for the dense-trajectory time-stepping modes.

    ``integration.mode: direct``/``worldline`` materialize a dense
    ``(M, steps + 1, 3)`` float64 position trajectory (plus same-shape
    siblings/intermediates, :data:`_TRAJECTORY_MEMORY_FACTOR`). With an
    auto-selected E31 `dtau` (trap-period resolution) over a long
    ``integration.time_s``, `steps` can reach 1e6-1e7; combined with an
    ensemble size in the hundreds-to-thousands that is a silent 100+ GB
    allocation attempt -- enough to lock up the host. This check runs
    *before* any trajectory is allocated and rejects such configs loudly
    instead.

    When `n_smoother_fit_points` is given (the field is
    :class:`~cliffordclock.fields.smoother.FieldSmoother`-backed, i.e.
    ``field.csv``/``field.comsol``, with `K` known from the fitted
    smoother), a second term
    (:data:`_TRAJECTORY_MEMORY_FACTOR_SMOOTHER` ``x N x K x 3 x 8`` bytes,
    ``N = n_atoms x steps``) is added, covering
    :func:`_stark_scalar_ensemble`'s single whole-trajectory `rate_fn`
    call (see :data:`_TRAJECTORY_MEMORY_FACTOR_SMOOTHER`'s comment for
    why only that call site needs it) -- this term alone can dwarf the
    base trajectory term for a realistic `K` even when `steps`/`n_atoms`
    are individually small enough to pass the base check. Closed-form
    synthetic fields have no `K` and skip this term entirely (the
    trajectory-only estimate already covers them exactly as before).

    Parameters
    ----------
    n_atoms : int
        Ensemble size `M` (Monte-Carlo particles or lattice quadrature
        nodes).
    steps : int
        Resolved number of integration steps.
    max_gb : float
        Ceiling in GB (10^9 bytes);
        ``integration.max_trajectory_memory_gb``.
    n_smoother_fit_points : int or None
        The fitted `FieldSmoother`'s number of RBF centers `K`
        (:func:`_build_field_fn`'s second return value), or `None` for a
        closed-form `field.synthetic` field.

    WP19 note: this function's own behavior is unchanged -- it still
    always raises when the estimate exceeds `max_gb`. What changed is
    *which call sites still call it*: for ``ensemble.regime="classical"``
    + ``mode="direct"`` under ``integration.evaluation="auto"`` (default)
    or ``"streaming"``, `run_pipeline_full` no longer calls this function
    at all when it would raise -- it calls
    :func:`_resolve_evaluation_mode` first and dispatches to the streaming
    accumulator instead (see :data:`DEFAULT_MAX_TRAJECTORY_MEMORY_GB`'s
    updated docstring). Every other mode (`worldline`, `secular`) and
    ``evaluation="batched"`` explicitly still call this function directly
    and get the original hard-reject behavior.

    Raises
    ------
    PipelineConfigError
        The estimate (trajectory term, plus the smoother term when
        `n_smoother_fit_points` is given) exceeds `max_gb`.
    """
    trajectory_gb, smoother_gb, estimated_gb = _estimate_trajectory_memory_gb(
        n_atoms, steps, n_smoother_fit_points=n_smoother_fit_points
    )
    if estimated_gb > max_gb:
        n_query_points = n_atoms * steps
        if n_smoother_fit_points is not None:
            breakdown = (
                f"trajectory term {trajectory_gb:.1f} GB = "
                f"{_TRAJECTORY_MEMORY_FACTOR:g} x M x (steps+1) x 3 x 8 bytes, plus "
                f"smoother-evaluation term {smoother_gb:.1f} GB = "
                f"{_TRAJECTORY_MEMORY_FACTOR_SMOOTHER:g} x N x K x 3 x 8 bytes "
                f"(N=M*steps={n_query_points} query points, K={n_smoother_fit_points} "
                "RBF fit points)"
            )
        else:
            breakdown = f"{_TRAJECTORY_MEMORY_FACTOR:g} x M x (steps+1) x 3 x 8 bytes"
        raise PipelineConfigError(
            f"this configuration would materialize dense trajectories of an estimated "
            f"{estimated_gb:.1f} GB (ensemble size M={n_atoms}, steps={steps}: "
            f"{breakdown}, see cliffordclock.pipeline._check_trajectory_memory), "
            f"exceeding the integration.max_trajectory_memory_gb limit of {max_gb:g} GB. "
            "Reduce ensemble.size, use an explicit coarser integration.dtau (fewer "
            "steps), or switch to a mode that never materializes a dense trajectory over "
            "integration.time_s: ensemble.regime='lattice' with "
            "integration.mode='fast_path' (Tier A, exact quadrature) for lattice "
            "ensembles, or integration.mode='secular' (classical, Tier B(ii)) with a "
            "bounded integration.points_per_period for classical periodic motion -- "
            "secular is not an unconditional escape from this guard, it is checked by "
            "this same estimate too (docs/timescales.md). If you genuinely want an "
            "allocation this large, raise integration.max_trajectory_memory_gb "
            "explicitly."
        )


#: Note text folded into a run's `uncertainty_notes` whenever
#: :func:`_resolve_evaluation_mode` auto-dispatches to the streaming
#: accumulator (WP19 acceptance criterion: this exact phrase). ``{gb}`` is
#: filled with the batched-path estimate that triggered the switch.
_STREAMING_DISPATCH_NOTE_TEMPLATE = (
    "switched to streaming evaluation (memory-bounded): the batched-path estimate "
    "({gb:.1f} GB) exceeded integration.max_trajectory_memory_gb; the O(M)-memory "
    "streaming accumulator (WP19) was used instead, producing numerically equivalent "
    "results (see docs/timescales.md)"
)


def _resolve_evaluation_mode(
    requested: str, estimated_batched_gb: float, max_gb: float
) -> tuple[str, str | None]:
    """Resolve ``integration.evaluation`` against the batched-path memory estimate (WP19).

    Only called for ``ensemble.regime="classical"`` + ``mode="direct"``
    (the one combination with a streaming accumulator today, both
    `coupling.type` values -- see :func:`_stark_scalar_ensemble_streaming`/
    :func:`_direct_rotor_ensemble_streaming`).

    Parameters
    ----------
    requested : str
        ``config.integration.evaluation``, already validated to be one of
        :data:`VALID_EVALUATION_MODES` by :func:`_parse_integration`.
    estimated_batched_gb : float
        The batched path's pre-flight estimate
        (:func:`_estimate_trajectory_memory_gb`'s `total_gb`) for this
        run's resolved `(n_atoms, steps)` (and, for `coupling.type=
        'stark_dc'`, `n_smoother_fit_points`).
    max_gb : float
        ``integration.max_trajectory_memory_gb``.

    Returns
    -------
    tuple[str, str | None]
        ``(resolved_mode, dispatch_note)`` -- `resolved_mode` is
        ``"batched"`` or ``"streaming"``; `dispatch_note` is non-`None`
        exactly when `requested="auto"` and the batched estimate forced
        an auto-dispatch to streaming (fold into `uncertainty_notes`).
        `requested="batched"`/`"streaming"` never produce a note -- an
        explicit request is not a "switch", it is what was asked for.
    """
    if requested in ("batched", "streaming"):
        return requested, None
    assert requested == "auto"
    if estimated_batched_gb > max_gb:
        return "streaming", _STREAMING_DISPATCH_NOTE_TEMPLATE.format(gb=estimated_batched_gb)
    return "batched", None


#: Multiplier on :func:`_estimate_streaming_memory_gb`'s ``M x 16 x 8``
#: base term (the largest single streaming-accumulator carry array, the
#: rotor-path's ``(M, 16)`` rotor state -- :func:`_stark_scalar_ensemble_streaming`'s
#: carry is smaller, just a handful of ``(M,)`` Kahan accumulators, so
#: charging the rotor path's larger carry unconditionally is a
#: conservative superset covering both). Covers the position/velocity
#: carry (``(M, 3)`` each) and phase/Kahan-compensation/rotor-phase
#: accumulators (``(M,)`` each) plus XLA scan-carry double-buffering, in
#: the same documented-trip-wire-not-exact-accounting spirit as
#: :data:`_TRAJECTORY_MEMORY_FACTOR`. This check is deliberately generous
#: (WP19 plan: "practically unreachable") -- streaming's whole point is
#: that its memory is O(M) regardless of `steps`, so only an ensemble
#: size in the millions-to-billions could approach any realistic budget.
_STREAMING_MEMORY_FACTOR = 8.0


def _estimate_streaming_memory_gb(n_atoms: int) -> float:
    """O(M) peak-memory estimate for the streaming accumulators (WP19).

    Unlike :func:`_estimate_trajectory_memory_gb`, independent of `steps`
    entirely -- the whole point of the streaming accumulators
    (:func:`_stark_scalar_ensemble_streaming`/
    :func:`_direct_rotor_ensemble_streaming`) is that no array they
    allocate scales with the number of integration steps. Per-step field
    evaluation is bounded independently by
    :data:`cliffordclock.fields.smoother.DEFAULT_CHUNK_SIZE`
    (:func:`cliffordclock.fields.smoother.chunked_apply`), not by `n_atoms`.
    """
    return _STREAMING_MEMORY_FACTOR * n_atoms * 16 * 8 / 1e9


def _check_streaming_memory(n_atoms: int, max_gb: float) -> None:
    """Pre-flight guard for the streaming accumulators' O(M) carry state (WP19).

    Mirrors :func:`_check_trajectory_memory`'s role for the batched path,
    against :func:`_estimate_streaming_memory_gb`'s much smaller estimate.
    Called unconditionally before either streaming accumulator runs
    (whether dispatched by `evaluation="auto"` or requested explicitly via
    `evaluation="streaming"`) -- the WP19 plan's own framing of this case
    ("practically unreachable") holds up: at the default 2 GB budget this
    only trips for `n_atoms` in the tens of millions.

    Raises
    ------
    PipelineConfigError
        `n_atoms` is large enough that even the O(M) streaming carry state
        exceeds `max_gb`.
    """
    estimated_gb = _estimate_streaming_memory_gb(n_atoms)
    if estimated_gb > max_gb:
        raise PipelineConfigError(
            f"this configuration's streaming-accumulator carry state is an estimated "
            f"{estimated_gb:.1f} GB (ensemble size M={n_atoms}: "
            f"{_STREAMING_MEMORY_FACTOR:g} x M x 16 x 8 bytes, see "
            "cliffordclock.pipeline._check_streaming_memory), exceeding the "
            f"integration.max_trajectory_memory_gb limit of {max_gb:g} GB even for the "
            "O(M)-memory streaming accumulator (WP19). Reduce ensemble.size, or raise "
            "integration.max_trajectory_memory_gb explicitly if you genuinely want an "
            "ensemble this large."
        )


def _build_field_fn(field_cfg: FieldConfig) -> tuple[CombinedFieldFn, int | None]:
    """Build the combined ``pos -> (E, grad_E)`` callable for `field_cfg`.

    Returns
    -------
    tuple[CombinedFieldFn, int | None]
        ``(field_fn, n_smoother_fit_points)``. `n_smoother_fit_points` is
        the fitted :class:`~cliffordclock.fields.smoother.FieldSmoother`'s
        number of RBF centers `K` (``field.csv``/``field.comsol``,
        threaded to :func:`_check_trajectory_memory`'s smoother-evaluation
        term via :func:`_resolve_dtau_steps_scalar`), or `None` for a
        closed-form ``field.synthetic`` field (no smoother -> no RBF
        query-point memory term).
    """
    if field_cfg.synthetic is not None:
        factory = _SYNTHETIC_FIELD_FACTORIES.get(field_cfg.synthetic.kind)
        if factory is None:
            valid = ", ".join(sorted(_SYNTHETIC_FIELD_FACTORIES))
            raise PipelineConfigError(
                f"field.synthetic.kind={field_cfg.synthetic.kind!r} is not a known synthetic "
                f"field; valid kinds: {valid}"
            )
        try:
            e_fn, grad_fn = factory(**field_cfg.synthetic.params)
        except TypeError as exc:
            raise PipelineConfigError(
                f"field.synthetic.params invalid for kind={field_cfg.synthetic.kind!r}: {exc}"
            ) from exc
        return as_field_fn(e_fn, grad_fn), None

    if field_cfg.comsol_path is not None:
        try:
            grid = load_field_comsol(
                field_cfg.comsol_path, expression_prefix=field_cfg.comsol_expression_prefix
            )
        except ValueError as exc:
            raise PipelineConfigError(f"field.comsol={field_cfg.comsol_path!r}: {exc}") from exc
        smoother = FieldSmoother.fit(grid, smoothing=field_cfg.smoothing)
        return smoother.evaluate, int(smoother.centers.shape[0])

    assert field_cfg.csv_path is not None  # enforced by _parse_field / FieldConfig construction
    try:
        grid = load_field_csv(field_cfg.csv_path)
    except ValueError as exc:
        raise PipelineConfigError(f"field.csv={field_cfg.csv_path!r}: {exc}") from exc
    smoother = FieldSmoother.fit(grid, smoothing=field_cfg.smoothing)
    return smoother.evaluate, int(smoother.centers.shape[0])


def _make_e14a_rate_fn(field_fn: CombinedFieldFn, mu: jnp.ndarray) -> fastpath.RateFn:
    """Build a `fastpath.RateFn` from `field_fn` + today's E14a coupling.

    This is the one place :mod:`cliffordclock.integrator.fastpath`'s
    coupling-agnostic ``(pos, v) -> δω̃`` signature is wired up to a
    concrete coupling model (E14a, explicit linear `mu`) -- a future E14b
    (quadratic DC-Stark) coupling would be wired up here too, with
    no change to `fastpath.py` itself (see that module's docstring and
    ``docs/coupling.md``).
    """

    def rate_fn(pos: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        delta_e, _grad_delta_e = field_fn(pos)
        return scalar_rate_perturbation(delta_e, v, mu)

    return rate_fn


#: Short literature citations for the report's Stark-coupling provenance
#: note. Mirrors the citations already
#: carried in ``cliffordclock.ensemble.species``'s module-level docstrings
#: for ``SR87``/``YB171`` -- reproduced here (not imported: species.py's
#: citation text lives in comments, not a field this module can read) so
#: the report can quote a source for a registry-resolved coefficient, not
#: just the bare number.
_STARK_SPECIES_CITATIONS: dict[str, str] = {
    "Sr87": "Middelmann et al., Phys. Rev. Lett. 109, 263004 (2012)",
    "Yb171": "Sherman et al., Phys. Rev. Lett. 108, 153002 (2012)",
}


def _resolve_stark_coupling(
    coupling: CouplingConfig, species: Species
) -> Species | StarkCoefficients:
    """Resolve `coupling`'s E14b coefficients to a `Species`/`StarkCoefficients`.

    ``docs/coupling.md`` "API for the pipeline plumbing" wiring note: when
    either override field is given, build a `StarkCoefficients` (species
    supplies only `clock_frequency_hz`, not its polarizability data); when
    neither is given, use `species` directly so
    ``resolve_stark_coefficient_hz_per_v2_m2`` reads the registry.

    Raises
    ------
    ValueError
        Propagated from `StarkCoefficients.__post_init__` (inconsistent
        explicit overrides) -- callers must catch this and re-raise as
        :class:`PipelineConfigError` (see `run_pipeline_full`).
    """
    has_override = (
        coupling.delta_alpha_dc_si is not None
        or coupling.stark_coefficient_hz_per_v2_m2 is not None
    )
    if has_override:
        return StarkCoefficients(
            clock_frequency_hz=species.clock_frequency_hz,
            delta_alpha_dc_si=coupling.delta_alpha_dc_si,
            stark_coefficient_hz_per_v2_m2=coupling.stark_coefficient_hz_per_v2_m2,
        )
    return species


def _stark_coupling_provenance_note(coupling: CouplingConfig, species: Species, k_s: float) -> str:
    """Coupling-provenance note.

    ``docs/coupling.md`` "API for the pipeline plumbing": "the report ...
    should record the coupling type and coefficient provenance" --
    folded into the report's `uncertainty_notes` (`report-schema.md` has
    no dedicated field for this; free text is the documented extension
    point, since systematic-budget modeling is out of scope for the
    analytics module).
    """
    is_override = (
        coupling.delta_alpha_dc_si is not None
        or coupling.stark_coefficient_hz_per_v2_m2 is not None
    )
    if is_override:
        source = (
            "explicit config override (delta_alpha_dc_si="
            f"{coupling.delta_alpha_dc_si!r}, stark_coefficient_hz_per_v2_m2="
            f"{coupling.stark_coefficient_hz_per_v2_m2!r})"
        )
    else:
        citation = _STARK_SPECIES_CITATIONS.get(
            species.name, "species registry, see docs/coupling.md"
        )
        source = f"species registry entry for {species.name!r} ({citation})"
    return (
        f"coupling=stark_dc (E14b): k_S={k_s!r} Hz.m^-2.V^-2, nu_0={species.clock_frequency_hz!r} "
        f"Hz, source={source}"
    )


#: Folded into `uncertainty_notes` for `coupling.type='stark_dc'` runs in
#: `integration.mode='fast_path'` (CONVENTIONS.md E29 scope): static
#: lattice quadrature nodes have `v = 0` exactly, so the fast path's
#: `⟨Δν/ν₀⟩` never includes the motional second-order Doppler shift --
#: a real, separately-budgeted clock systematic (E30 carries it; E29 does
#: not, by construction). Not added for `coupling.type='linear_mu'` runs:
#: this keeps `uncertainty_notes` byte-identical for the existing
#: `linear_mu` example configs, which predate this note and do not use
#: the E14b coupling it describes.
_FAST_PATH_DOPPLER_EXCLUSION_NOTE = (
    "fast_path (E29) reports the Stark/field shift only: static v=0 quadrature nodes carry "
    "no motional second-order Doppler contribution (CONVENTIONS.md E29 scope); "
    "that term is a separate, real clock systematic not included in this run's "
    "mean_fractional_shift."
)

# ---------------------------------------------------------------------------
# WP20: blackbody-radiation shift (CONVENTIONS.md E32/E33). See the module
# docstring's "Blackbody-radiation shift" interface note for the composition
# this feeds; :func:`_resolve_bbr_pivot_perturbation` is the single call site
# `run_pipeline_full` uses to compute the (P-1)_BBR scalar (once per run) and
# its report note.
# ---------------------------------------------------------------------------

#: Short literature citations for the report's BBR provenance note. Mirrors
#: `_STARK_SPECIES_CITATIONS`'s role; reproduced here (not imported) since
#: `species.py`'s citation text lives in comments, not a field this module
#: can read. G7 sign-off gate edit 6: citations embedded in the registry
#: entry AND surfaced in the report.
_BBR_SPECIES_CITATIONS: dict[str, str] = {
    "Sr87": (
        "static: Middelmann et al., PRL 109, 263004 (2012); dynamic shape: Lisdat et al., "
        "PR Research 3, L042036 (2021); dynamic anchor: Aeppli et al., arXiv:2403.10664; "
        "rescaled coefficients: arXiv:2507.14030"
    ),
    "Yb171": (
        "static + dynamic (T^6): Hassan et al., arXiv:2506.05304 (2025); dynamic (T^8) "
        "derived via Beloy et al., PRL 113, 260801 (2014) eta_2"
    ),
}

#: M1/E2 multipole BBR budget line (G7 sign-off gate edit 3, A3): a fixed,
#: order-of-magnitude "modeled-out" budget entry, not a per-species/per-T
#: computed value -- the sign-off states this generically ("magnitude ~6e-20
#: each"), citing Porsev & Derevianko PRA 74, 020502(R) (2006) *with its 2012
#: erratum* (PRA 86, 029904), routed through Lisdat 2021 (the erratum-safe
#: citation path the sign-off requires, A3 "confirm the 6e-20 is the
#: post-erratum value"). Always appended when BBR is active (both species):
#: the WP20 registry has no independent, per-species M1/E2 evaluation to
#: refine this with.
_BBR_M1_E2_BUDGET_NOTE = (
    "M1/E2 multipole BBR contributions: modeled out, magnitude ~6e-20 each (Porsev & "
    "Derevianko, PRA 74, 020502(R) (2006), erratum PRA 86, 029904 (2012), via Lisdat et al., "
    "PR Research 3, L042036 (2021)); not included in mean_fractional_shift (E1 scalar-"
    "polarizability model only, CONVENTIONS.md E33 scope note)."
)


def _resolve_bbr_pivot_perturbation(
    environment: EnvironmentConfig, species: Species
) -> tuple[float, str | None]:
    """``(P−1)_BBR`` (E32) plus its report note, or ``(0.0, None)`` if BBR is off.

    Single call site `run_pipeline_full` uses to compute the WP20 BBR
    scalar once per run (:func:`cliffordclock.integrator.omega.bbr_pivot_perturbation`)
    and assemble its report-note text: T/coefficients/citations (G7 sign-off
    gate edit 6), the coefficient-uncertainty propagation labeled
    "arithmetic-reproduction fidelity" (never "BBR accuracy", gate edit 4c),
    the M1/E2 budget line (gate edit 3), and the 300-350 K
    beyond-cross-verified-range note when applicable (gate edit 5 / B4).

    Parameters
    ----------
    environment : EnvironmentConfig
    species : Species
        Resolved run species (used for its `bbr_coefficients` registry
        entry, regardless of any `coupling.delta_alpha_dc_si`/
        `stark_coefficient_hz_per_v2_m2` override -- BBR's static/dynamic
        fit is a separate, independently published measurement, not
        derivable from a generic Δα override).

    Returns
    -------
    tuple[float, str | None]
        ``(bbr_pivot_perturbation, note)``. `note` is `None` iff
        `environment.radiation_temperature_k` is `None` (BBR off).

    Raises
    ------
    PipelineConfigError
        `species` has no resolvable `BbrCoefficients` (e.g. `Al27+`);
        wraps the underlying `ValueError`.
    """
    temperature_k = environment.radiation_temperature_k
    if temperature_k is None:
        return 0.0, None

    try:
        bbr_value = _bbr_pivot_perturbation_e32(temperature_k, species)
        sigma_frac, temperature_uncertainty_included = bbr_pivot_uncertainty(
            temperature_k, species, environment.radiation_temperature_uncertainty_k
        )
        coeffs = species.resolve_bbr_coefficients()
    except ValueError as exc:
        raise PipelineConfigError(str(exc)) from exc

    dyn_terms = ", ".join(f"n={n}:{coeff!r}Hz" for n, coeff in sorted(coeffs.dyn_coeffs_hz.items()))
    citation = _BBR_SPECIES_CITATIONS.get(species.name, "see docs/coupling.md")
    t0 = BBR_REFERENCE_TEMPERATURE_K
    parts = [
        f"BBR (CONVENTIONS.md E32): T={temperature_k!r}K, (P-1)_BBR={bbr_value!r} "
        f"(nu_stat({t0:.0f}K)={coeffs.nu_stat_300k_hz!r}Hz, dynamic coeffs [{dyn_terms}]; "
        f"{citation})",
        (
            f"BBR coefficient uncertainty (arithmetic-reproduction fidelity, NOT an "
            f"independent BBR-accuracy claim -- G7 sign-off A4#2c): {sigma_frac:.2e} "
            "fractional, 1-sigma, combining the static and dynamic-anchor registry "
            "uncertainties in quadrature"
            + (
                " (includes radiation_temperature_uncertainty_K propagation via the exact "
                "polynomial derivative, G7 sign-off A4#3)"
                if temperature_uncertainty_included
                else " (conditional on exact T: environment.radiation_temperature_uncertainty_K "
                "was not given, G7 sign-off A4#3 -- set it to also propagate sigma_T)"
            )
        ),
        _BBR_M1_E2_BUDGET_NOTE,
    ]
    if temperature_k > coeffs.cross_verified_max_k:
        parts.append(
            f"T={temperature_k!r}K is in-fit-range but beyond the PTB<->JILA 1e-19-class "
            f"cross-verification band (T<={coeffs.cross_verified_max_k:.0f}K, G7 sign-off B4) "
            "-- treat the BBR uncertainty above as less certain in this band."
        )
    return bbr_value, " ".join(parts)


# ---------------------------------------------------------------------------
# WP21 Tier 2: ion-clock electric-quadrupole shift (CONVENTIONS.md E34/E35).
# Unlike WP20's BBR term (a single scalar computed once per run,
# :func:`_resolve_bbr_pivot_perturbation`), the quadrupole term depends on
# the LOCAL field-gradient tensor, so it cannot be precomputed once -- the
# per-run constants (Theta, J, m_J-or-averaging-mode, quantization axis,
# nu_0) are resolved once (:func:`_resolve_quadrupole_theta_j`) and
# :func:`_quadrupole_pivot_from_grad` is called at every point `grad_e_total`
# is evaluated (`_make_stark_rate_fn`'s `rate_fn`, `_stark_rotor_ensemble`'s
# per-step body), mirroring `omega.pivot_perturbation_stark`'s own per-point
# evaluation pattern more than BBR's per-run one.
# ---------------------------------------------------------------------------


def _resolve_quadrupole_theta_j(quadrupole: QuadrupoleConfig) -> tuple[float, float]:
    """``(theta_au, j)`` for `quadrupole` -- registry lookup or explicit override.

    Parameters
    ----------
    quadrupole : QuadrupoleConfig

    Returns
    -------
    tuple[float, float]

    Raises
    ------
    PipelineConfigError
        `quadrupole.state` is not a `QUADRUPOLE_MOMENTS` registry key
        (already validated by `_parse_quadrupole`, so this should not
        occur for a config built via `PipelineConfig.from_dict`/
        `from_yaml`; guarded here too for direct `PipelineConfig`
        construction).
    """
    if quadrupole.state is not None:
        try:
            moment = get_quadrupole_moment(quadrupole.state)
        except KeyError as exc:
            raise PipelineConfigError(str(exc)) from exc
        return moment.theta_au, moment.j
    assert quadrupole.theta_au is not None  # enforced by _parse_quadrupole
    assert quadrupole.j is not None  # enforced by _parse_quadrupole
    return quadrupole.theta_au, quadrupole.j


def _quadrupole_pivot_from_grad(
    quadrupole: QuadrupoleConfig, theta_au: float, j: float, grad_e_total: jnp.ndarray
) -> jnp.ndarray:
    """``(P-1)_Q`` at one or more points, from the LOCAL gradient tensor (CONVENTIONS.md E34/E35).

    Dispatches on `quadrupole.averaging_mode`:
    ``"fixed"``: :func:`~cliffordclock.integrator.omega.quadrupole_pivot_perturbation`
    at `quadrupole.m_j`/`quadrupole.quantization_axis`.
    ``"three_orientation"``:
    :func:`~cliffordclock.integrator.omega.quadrupole_three_orientation_average`
    (CONVENTIONS.md E35 A2's exact cancellation -- the `m_j` passed is
    irrelevant to the *result* (it cancels along with the axis sum for
    ANY valid `m_j`), so the stretched state `m_j=j` is used as an
    arbitrary valid representative; :func:`~cliffordclock.integrator.omega.quadrupole_mj_factor`
    still requires `j >= 1`).

    Parameters
    ----------
    quadrupole : QuadrupoleConfig
    theta_au, j : float
        From :func:`_resolve_quadrupole_theta_j`.
    grad_e_total : jax.Array, shape (..., 3, 3)
        Gradient tensor (E13), V/m^2.

    Returns
    -------
    jax.Array, shape (...,)
        ``(P-1)_Q``, dimensionless.
    """
    if quadrupole.averaging_mode == "three_orientation":
        return quadrupole_three_orientation_average(
            grad_e_total, theta_au, j, j, quadrupole.nu_0_hz
        )
    assert quadrupole.m_j is not None  # enforced by _parse_quadrupole
    return _quadrupole_pivot_perturbation_e34(
        grad_e_total,
        jnp.asarray(quadrupole.quantization_axis, dtype=jnp.float64),
        theta_au,
        j,
        quadrupole.m_j,
        quadrupole.nu_0_hz,
    )


def _quadrupole_provenance_note(quadrupole: QuadrupoleConfig, theta_au: float, j: float) -> str:
    """Quadrupole-provenance report note (mirrors `_stark_coupling_provenance_note`)."""
    if quadrupole.state is not None:
        moment = get_quadrupole_moment(quadrupole.state)
        source = (
            f"registry state {quadrupole.state!r} ({moment.source}, verification="
            f"{moment.verification!r})"
        )
    else:
        source = f"explicit config override (theta_au={theta_au!r}, j={j!r})"
    mode_desc = (
        "three-orientation averaging (CONVENTIONS.md E35 A2, exact cancellation)"
        if quadrupole.averaging_mode == "three_orientation"
        else f"m_j={quadrupole.m_j!r}, quantization_axis={quadrupole.quantization_axis!r}"
    )
    return (
        f"coupling.quadrupole (E34/E35): theta_au={theta_au!r}, j={j!r}, "
        f"nu_0={quadrupole.nu_0_hz!r} Hz, {mode_desc}, source={source}. Traceless-symmetric "
        "gradient only (G8 A5#3); spin-connection gradient contribution not modeled "
        "(CONVENTIONS.md E35 scope limit, bounded)."
    )


# ---------------------------------------------------------------------------
# WP22 Part 1: gravitational-redshift pivot term (CONVENTIONS.md section 15,
# E36). Unlike WP20's BBR term (a single scalar computed once per run) but
# like WP21's quadrupole term, the gravity term depends on each point's own
# position -- height, not the field-gradient tensor -- so it is evaluated at
# every point :func:`_make_stark_rate_fn`'s `rate_fn`/
# :func:`_stark_rotor_ensemble`'s per-step body calls, mirroring
# :func:`_quadrupole_pivot_from_grad`'s per-point-evaluation pattern.
# ---------------------------------------------------------------------------


def _grav_pivot_from_position(gravity: GravityConfig, pos: jnp.ndarray) -> jnp.ndarray:
    """``(P-1)_grav`` at one or more points, from their position (CONVENTIONS.md E36).

    Parameters
    ----------
    gravity : GravityConfig
    pos : jax.Array, shape (..., 3)
        Position(s), meters.

    Returns
    -------
    jax.Array, shape (...,)
        ``(P-1)_grav``, dimensionless.
    """
    height_m = height_along_axis(pos, jnp.asarray(gravity.up_axis, dtype=jnp.float64))
    return _grav_pivot_perturbation_e36(height_m, gravity.g_m_s2, gravity.reference_height_m)


def _gravity_provenance_note(gravity: GravityConfig) -> str:
    """Gravity-provenance report note (mirrors `_quadrupole_provenance_note`/BBR's note)."""
    return (
        f"environment.gravity (CONVENTIONS.md section 15, E36): g_m_s2={gravity.g_m_s2!r}, "
        f"up_axis={gravity.up_axis!r} (unit-normalized internally), "
        f"reference_height_m={gravity.reference_height_m!r}. Sign convention (G9 sign-off "
        "A1): (P-1)_grav = g*(h-h_ref)/c^2 with h = up_axis_hat . r -- a HIGHER clock "
        "(larger h along up_axis) runs FASTER, so (P-1)_grav > 0 for h > h_ref. g_m_s2 "
        f"defaults to standard gravity ({STANDARD_GRAVITY!r} m/s^2, exact by "
        "international definition); at the 1e-19 level the physically correct input is "
        "the LAB'S OWN SURVEYED LOCAL g (which can differ from standard g by parts in "
        "1e3, e.g. Boulder CO's 9.796 m/s^2, see benchmarks/run_bothwell_redshift.py), "
        "not the standard-gravity default -- set g_m_s2 explicitly for any 1e-19-class "
        "comparison against a real site (G9 sign-off B1)."
    )


#: Recommended cap on the sampled height extent (max-min along
#: `environment.gravity.up_axis`) before CliffordClock's uniform-`g` E36
#: model is warned as approaching the edge of its stated validity (G9
#: sign-off A3, CONVENTIONS.md section 15): the uniform-g approximation's
#: error grows as `(g/(c^2*R_E))*(delta_h)^2` and reaches the 1e-19 floor
#: only near `delta_h ~ 76 m` -- this 10 m threshold keeps an
#: order-of-magnitude margin below that bound, per the gate's explicit
#: recommendation ("a ~10 m cap gives an order-of-magnitude margin").
#: A WARN, not a hard PipelineConfigError (the gate's own wording, "warn
#: (or cap)"): unlike environment.radiation_temperature_K's hard-edged fit
#: range, this is a physics validity margin, not a fit-support boundary,
#: and every configuration this project ships (mm-to-m-scale lab samples)
#: sits far below it.
GRAVITY_EXTENT_WARN_M = 10.0


def _gravity_extent_warn_note(gravity: GravityConfig, positions_m: jnp.ndarray) -> str | None:
    """A G9-sign-off-A3 validity-bound warning note, or `None` if `positions_m`'s
    sampled height extent is within :data:`GRAVITY_EXTENT_WARN_M`.

    Parameters
    ----------
    gravity : GravityConfig
    positions_m : jax.Array, shape (..., 3)
        Every position this run actually sampled/evaluated at (e.g. the
        run's `trajectories`, reshaped to `(-1, 3)`) -- an approximation
        for time-stepping regimes whose motion is not densely sampled
        (e.g. `mode="secular"`'s closed-form orbit, only its initial
        conditions), documented here rather than silently understated.

    Returns
    -------
    str or None
    """
    heights_m = height_along_axis(
        jnp.reshape(jnp.asarray(positions_m, dtype=jnp.float64), (-1, 3)),
        jnp.asarray(gravity.up_axis, dtype=jnp.float64),
    )
    span_m = float(jnp.max(heights_m) - jnp.min(heights_m))
    if span_m <= GRAVITY_EXTENT_WARN_M:
        return None
    return (
        f"environment.gravity height-extent WARNING: sampled positions span "
        f"{span_m:.6g} m along up_axis, exceeding the {GRAVITY_EXTENT_WARN_M:.0f} m "
        "margin CliffordClock recommends for the uniform-g E36 model (CONVENTIONS.md "
        "section 15 / G9 sign-off A3: the uniform-g approximation is exact to <<1e-19 "
        "at lab/mm scale and stays below the 1e-19 floor out to ~76 m, an order-of-"
        "magnitude margin below this warning threshold) -- beyond this scale, prefer a "
        "surveyed potential difference or a height-dependent g/geoid model rather than "
        "a single uniform g."
    )


def _make_stark_rate_fn(
    field_fn: CombinedFieldFn,
    species_or_coeffs: Species | StarkCoefficients,
    *,
    bbr_pivot_perturbation: float = 0.0,
    quadrupole: QuadrupoleConfig | None = None,
    gravity: GravityConfig | None = None,
) -> fastpath.RateFn:
    """Build a `fastpath.RateFn` from `field_fn` + the E14b physical DC-Stark coupling.

    WP20 (E32/E33): `bbr_pivot_perturbation` (see
    :func:`_resolve_bbr_pivot_perturbation`) is composed into `p_minus_1`
    before the `gamma_inv` weighting, mirroring
    :func:`cliffordclock.integrator.omega.scalar_rate_perturbation_stark`'s
    handling exactly -- this is the single call site every `mode` that
    dispatches through a `fastpath.RateFn` shares (`fast_path`, `secular`,
    `direct` batched and streaming; see the module docstring's WP20 note),
    so composing it here threads BBR into all four at once. Defaults to
    ``0.0``, an exact IEEE-754 no-op, so every pre-WP20 call site
    (including every shipped example, none of which sets
    ``environment.radiation_temperature_K``) is byte-for-byte unaffected.

    Mirrors :func:`_make_e14a_rate_fn`'s role for the linear coupling:
    this is the one place the coupling-agnostic `fastpath.RateFn` seam
    (module docstring interface note 3; ``cliffordclock.integrator.fastpath``'s
    own docstring) is wired up to E14b
    (:func:`cliffordclock.integrator.omega.pivot_perturbation_stark`).
    Because it is a plain `RateFn`, it plugs into
    :func:`cliffordclock.integrator.fastpath.lattice_shift_expectation`
    (`mode="fast_path"`) and
    :func:`cliffordclock.integrator.fastpath.secular_average_shift_ensemble`
    (`mode="secular"`) with **no changes to `fastpath.py` itself**; `mode="direct"`
    still needs a different accumulation path since
    :func:`cliffordclock.integrator.worldline.integrate_ensemble` remains
    hardwired to the E14a linear `mu` coupling -- see
    :func:`_stark_scalar_ensemble` for the scalar-only accumulator
    `mode="direct"` uses instead. **`mode="worldline"` (WP16, CONVENTIONS.md
    E16/E18 instantiated for E14b): no longer scalar-only** -- see
    :func:`_stark_rotor_ensemble`, which builds the true Cl(1,3) rotor via
    :func:`cliffordclock.integrator.omega.build_omega_stark` (additive to
    `omega.py`; `worldline.py`/`stepper.py` remain untouched, so this
    module implements its own E17/E19 midpoint-stepping scan rather than
    calling `integrate_ensemble`, which stays E14a-`mu`-specific). See the
    module docstring's mode table (interface note 5) and
    ``docs/CONVENTIONS.md``'s production-path note, now updated to cite
    the direct rotor<->scalar cross-check this enables
    (``tests/test_integrator_stark_rotor.py``).

    **No pipeline-level E11 baseline/perturbation split.** `field_fn(pos)`
    returns this pipeline's single combined-field abstraction -- the
    *total* field `E(r)`, with no separate `E_0`/`δE` decomposition
    exposed at this layer (`cliffordclock.fields.decompose`'s
    `Baseline`/`residual` machinery is used only internally by
    `FieldSmoother.fit`, for RBF-conditioning purposes -- `FieldSmoother.evaluate`,
    like every synthetic field factory, returns the combined total field).
    This rate_fn therefore calls `pivot_perturbation_stark` with
    ``e0=E(r)`` (the entire field) and ``delta_e=0``:
    `stark_pivot_terms`'s `baseline` term alone then evaluates the full,
    exact E14b formula `P(r) - 1 = k_S|E(r)|^2/nu_0` (its `cross`/
    `quadratic` terms are identically zero since `delta_e` is identically
    zero) -- exact, not a truncation, given this pipeline has no genuine
    E11 split to feed those terms non-trivially. A pipeline-level E11
    split would require this module to expose separate `E_0`/`δE` inputs
    at the `field_fn` layer, which it deliberately does not (see the
    "No pipeline-level E11 baseline/perturbation split" note above).

    The kinematic (second-order Doppler, E21) term is combined with the
    same E10 catastrophic-cancellation-avoiding algebraic rewrite as
    :func:`cliffordclock.integrator.omega.scalar_rate_perturbation` --
    reimplemented here (not imported: that function is hardwired to the
    E14a `pivot_perturbation`, not a general `p_minus_1` input) so this
    rate_fn's kinematic handling has the identical accuracy at the
    realistic cold-atom `v/c` regime.

    Parameters
    ----------
    field_fn : CombinedFieldFn
        ``pos -> (E, grad_E)``; `grad_E` is unused here (E14b's pivot
        needs only the field itself, not its gradient -- `grad_E` matters
        for E14a's `spin_connection`/boost term, not the E14b scalar
        pivot).
    species_or_coeffs : Species | StarkCoefficients
        Resolved DC-Stark coefficients; see :func:`_resolve_stark_coupling`.
    bbr_pivot_perturbation : float, default 0.0
        ``(P−1)_BBR`` (E32, WP20); see :func:`_resolve_bbr_pivot_perturbation`.
    quadrupole : QuadrupoleConfig or None, default None
        Electric-quadrupole shift parameters (E34/E35, WP21); `None`
        (default): no quadrupole term, byte-identical to pre-WP21
        behavior. When given, `grad_E` (unused by the pure E14b Stark term
        above) is used to evaluate the quadrupole term at each point via
        :func:`_quadrupole_pivot_from_grad`, composed additively into
        `p_minus_1` exactly like `bbr_pivot_perturbation`.
    gravity : GravityConfig or None, default None
        Gravitational-redshift pivot-term parameters (E36, WP22); `None`
        (default): no gravity term, byte-identical to pre-WP22 behavior.
        When given, `pos` (already available -- no new field capability
        needed, unlike the quadrupole term's `grad_E`) is used to evaluate
        each point's height via
        :func:`~cliffordclock.integrator.omega.height_along_axis` and the
        gravitational pivot term via
        :func:`~cliffordclock.integrator.omega.grav_pivot_perturbation`,
        composed additively into `p_minus_1` exactly like
        `bbr_pivot_perturbation`/`quadrupole`.

    Returns
    -------
    fastpath.RateFn
    """
    quadrupole_theta_j = _resolve_quadrupole_theta_j(quadrupole) if quadrupole is not None else None

    def rate_fn(pos: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        e_total, grad_e = field_fn(pos)
        e_total = jnp.asarray(e_total, dtype=jnp.float64)
        quadrupole_value: jnp.ndarray | float = 0.0
        if quadrupole is not None:
            assert quadrupole_theta_j is not None
            theta_au, j = quadrupole_theta_j
            quadrupole_value = _quadrupole_pivot_from_grad(quadrupole, theta_au, j, grad_e)
        grav_value: jnp.ndarray | float = 0.0
        if gravity is not None:
            grav_value = _grav_pivot_from_position(gravity, pos)
        p_minus_1 = pivot_perturbation_stark(
            e_total,
            jnp.zeros_like(e_total),
            species_or_coeffs,
            bbr_pivot_perturbation=bbr_pivot_perturbation,
            quadrupole_pivot_perturbation=quadrupole_value,
            grav_pivot_perturbation=grav_value,
        )
        v = jnp.asarray(v, dtype=jnp.float64)
        v2 = jnp.sum(v * v, axis=-1)
        x = v2 / SPEED_OF_LIGHT**2
        gamma_inv = jnp.sqrt(1.0 - x)
        # Same E10-safe rewrite as omega.scalar_rate_perturbation (E21):
        # gamma_inv - 1 = -x/(1+gamma_inv), avoiding the catastrophic
        # cancellation of sqrt(1-x)-1 at realistic cold-atom x ~ 1e-20.
        kinematic = -x / (1.0 + gamma_inv)
        return kinematic + p_minus_1 * gamma_inv

    return rate_fn


def _resolve_dtau_steps_scalar(
    integration_cfg: IntegrationConfig,
    trap: HarmonicTrap,
) -> tuple[float, int, bool]:
    """``(dtau, steps, dtau_was_auto)`` for `coupling.type='stark_dc'` in
    `mode="direct"` (:func:`_stark_scalar_ensemble`).

    WP16 note: `mode="worldline"` under `coupling.type='stark_dc'` no
    longer uses this resolver -- it now runs the true rotor
    (:func:`_stark_rotor_ensemble`) and so needs the same
    generator-angle pre-flight guard as the `linear_mu` rotor path; see
    :func:`_resolve_dtau_steps_worldline`, reused unchanged for it
    (already coupling-agnostic: it only calls the coupling-agnostic
    `rate_fn`, not anything E14a-specific).

    Mirrors :func:`_resolve_dtau_steps_direct`'s dtau-or-auto-select
    (E31's :func:`~cliffordclock.integrator.fastpath.select_dtau`) and
    steps-or-from-`time_s` resolution, but **omits** the rotor-generator-
    angle pre-flight guard (:data:`MAX_PER_STEP_ROTOR_ANGLE_RAD`,
    :func:`_estimate_max_generator_angle`): that guard exists to protect
    `cliffordclock.cl13.exp_bivector`'s fixed-order Taylor evaluation,
    which :func:`_stark_scalar_ensemble` never calls (no bivector/rotor
    state is ever built by *this* accumulator -- see its docstring; the
    `mode="worldline"` rotor accumulator, :func:`_stark_rotor_ensemble`,
    does call `exp_bivector` and uses the guarded resolver instead, per
    the WP16 note above). A plain Kahan-summed float64 accumulation
    of E21's `delta_omega_tilde` has no comparable convergence-radius
    failure mode at any dtau this project's realistic parameters reach.

    WP19 note: this resolver no longer calls the trajectory-memory guard
    itself (:func:`_check_trajectory_memory`) -- unlike
    :func:`_resolve_dtau_steps_direct` (still called by the `worldline`
    resolver, which has no streaming alternative and so still enforces
    the guard unconditionally at its own call site), this resolver's only
    call site is the `mode="direct"` + `coupling.type='stark_dc'` cell,
    which now has a streaming accumulator (WP19,
    :func:`_stark_scalar_ensemble_streaming`) to dispatch to instead of
    unconditionally raising -- `run_pipeline_full` makes that dispatch
    decision itself (via :func:`_estimate_trajectory_memory_gb` and
    :func:`_resolve_evaluation_mode`) using the `n_atoms`/
    `n_smoother_fit_points` it already has at hand, after calling this
    resolver for `dtau`/`steps` (this function no longer takes those two
    parameters at all -- WP19 API change, this is a private function with
    exactly one call site).
    """
    dtau_was_auto = integration_cfg.dtau is None
    dtau = (
        integration_cfg.dtau
        if integration_cfg.dtau is not None
        else fastpath.select_dtau(trap, integration_cfg.points_per_period)
    )
    if integration_cfg.steps is not None:
        steps = integration_cfg.steps
    else:
        assert integration_cfg.time_s is not None
        steps = max(1, round(integration_cfg.time_s / (dtau * TAU_COMPTON)))
    return dtau, steps, dtau_was_auto


def _stark_scalar_ensemble(
    rate_fn: fastpath.RateFn,
    trajectories: jnp.ndarray,
    dtau: float,
) -> EnsembleResult:
    """Coupling-agnostic scalar phase accumulation (E19 midpoint, E21/E22 Kahan
    sum) for `coupling.type='stark_dc'` in `mode="direct"`.

    WP16 note: this accumulator's `mode="worldline"` use has been
    *replaced* by :func:`_stark_rotor_ensemble` (the true Cl(1,3) rotor
    path, via :func:`cliffordclock.integrator.omega.build_omega_stark`);
    it remains the `mode="direct"` implementation only (classical-ensemble
    trajectories, where a rotor cross-check is not this WP's target -- see
    the module docstring's mode table). Historical context for why a
    scalar-only path existed at all: before WP16, E14b's quadratic pivot
    had no `Ω`-bivector construction anywhere in this codebase, so this
    function mirrors ``cliffordclock.integrator.worldline.integrate_ensemble``'s
    midpoint-evaluation + Kahan-summation phase accumulation but calls the
    coupling-agnostic `rate_fn` (E21) directly instead of composing a
    rotor. `phase_rotor` is set equal to `phase` (no separate rotor
    cross-check is computed by *this* accumulator); `norm_error`/
    `max_norm_drift` are exactly zero (no rotor state is ever advanced) --
    the same convention
    ``fastpath.lattice_shift_expectation``/``fastpath.secular_average_shift``
    already use for their own rotor-free result fields.

    For a *static* trajectory (every time sample the same position), the
    finite-difference velocity is exactly zero at every step, so this
    reduces algebraically to exactly `rate_fn(node, 0) * n_steps * dtau =
    rate_fn(node, 0) * T̃` -- E29's own formula -- for *any* `n_steps`/
    `dtau`, matching `fastpath.lattice_shift_expectation` exactly
    regardless of the different code path. (`mode="direct"` trajectories
    are not generally static, so this reduction is incidental there, not
    the reason this accumulator is used for that mode.)

    Parameters
    ----------
    rate_fn : fastpath.RateFn
        ``(pos, v) -> delta_omega_tilde`` (E21); see :func:`_make_stark_rate_fn`.
    trajectories : jax.Array, shape (M, T, 3)
        Dense per-atom position trajectories, meters (``T - 1`` steps).
    dtau : float
        Fixed step size ``dτ̃`` (Compton units, E9).

    Returns
    -------
    EnsembleResult
    """
    trajectories = jnp.asarray(trajectories, dtype=jnp.float64)
    m = trajectories.shape[0]
    n = trajectories.shape[1] - 1
    dt_phys = dtau * TAU_COMPTON

    def run_one(traj: jnp.ndarray) -> jnp.ndarray:
        pos_a, pos_b = traj[:-1], traj[1:]
        pos_mid = 0.5 * (pos_a + pos_b)
        v_mid = (pos_b - pos_a) / dt_phys
        domega = jnp.asarray(rate_fn(pos_mid, v_mid), dtype=jnp.float64)
        return kahan_sum(domega * dtau)

    phase = jax.vmap(run_one)(trajectories)  # (M,)
    t_tilde = n * dtau
    fractional_shift = phase / t_tilde
    zeros_m = jnp.zeros(m, dtype=jnp.float64)
    # Identity rotor, index 0 = scalar 1 (E2 basis ordering) -- no rotor
    # state is ever advanced by this scalar-only accumulator.
    identity_rotor = jnp.zeros(16, dtype=jnp.float64).at[0].set(1.0)
    r_final = jnp.broadcast_to(identity_rotor, (m, 16))
    return EnsembleResult(
        r_final=r_final,
        phase=phase,
        phase_rotor=phase,
        fractional_shift=fractional_shift,
        norm_error=zeros_m,
        max_norm_drift=zeros_m,
        n_steps=jnp.full((m,), n, dtype=jnp.int64),
    )


def _kahan_add_scalar(
    total: jnp.ndarray, comp: jnp.ndarray, value: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One step of Kahan (compensated) summation (E10), online/carry form.

    Reimplemented here rather than imported (``cliffordclock.integrator.worldline``
    only exposes the whole-array ``kahan_sum``, already used by
    :func:`_stark_scalar_ensemble` above; the online per-step form used
    inside a ``lax.scan`` carry is a private helper there,
    ``worldline._kahan_add``) so :func:`_stark_rotor_ensemble` does not
    reach across a module's private-function boundary -- mirrors this
    file's existing pattern of small, independently-accurate
    reimplementations at each call site (e.g. :func:`_make_stark_rate_fn`'s
    kinematic-term rewrite) rather than sharing pipeline-private helpers
    with `worldline.py`.

    Parameters
    ----------
    total : jax.Array, scalar
        Running sum so far.
    comp : jax.Array, scalar
        Running compensation (low-order bits lost in previous additions).
    value : jax.Array, scalar
        Value to add.

    Returns
    -------
    (new_total, new_comp) : tuple of jax.Array, scalar
    """
    y = value - comp
    t = total + y
    new_comp = (t - total) - y
    return t, new_comp


def _stark_rotor_ensemble(
    field_fn: CombinedFieldFn,
    species_or_coeffs: Species | StarkCoefficients,
    trajectories: jnp.ndarray,
    dtau: float,
    *,
    renorm_every: int = DEFAULT_RENORM_EVERY,
    bbr_pivot_perturbation: float = 0.0,
    quadrupole: QuadrupoleConfig | None = None,
    gravity: GravityConfig | None = None,
) -> EnsembleResult:
    """True Cl(1,3) rotor accumulation (E17-E24) for `coupling.type='stark_dc'`
    in `mode="worldline"` (WP16 -- replaces :func:`_stark_scalar_ensemble`
    for this one mode; see the module docstring's mode table).

    WP20 (E32/E33): `bbr_pivot_perturbation` is threaded into every
    per-step :func:`~cliffordclock.integrator.omega.build_omega_stark`
    call below -- this is the `mode="worldline"` (rotor) half of the
    module docstring's WP20 note; :func:`_make_stark_rate_fn` handles the
    other three modes. Defaults to ``0.0``, an exact no-op.

    Implements the same exponential-midpoint step (E19) and E20
    renormalization/E22 Kahan-summation bookkeeping as
    ``cliffordclock.integrator.worldline.integrate_worldline``'s scan body
    and ``cliffordclock.integrator.stepper.rotor_step`` -- same formulas,
    same per-step structure -- but built directly from
    :func:`cliffordclock.cl13.exp_bivector`/:func:`~cliffordclock.cl13.geometric_product`
    and :func:`cliffordclock.integrator.omega.build_omega_stark` rather
    than calling those functions, because both are hardwired to the E14a
    linear `mu` coupling (`rotor_step` calls `build_omega`, not a
    pluggable pivot source) and this WP's scope keeps `worldline.py`/
    `stepper.py` untouched (see the WP16 builder report). `species_or_coeffs`
    is a plain Python object (not a JAX array), so it is closed over here
    exactly as `_make_stark_rate_fn`'s `rate_fn` closes over it -- never
    threaded through `lax.scan`/`vmap` as a traced value.

    Both the primary scalar phase (`phase`, `omega[..., IDX_E12] * dtau`
    per step, the same quantity :func:`_stark_scalar_ensemble`/
    `_make_stark_rate_fn`'s `rate_fn` compute) and the E24 rotor-extracted
    phase (`phase_rotor`, via :func:`~cliffordclock.integrator.stepper.rotor_plane_angle`
    on each step's `exp_bivector` factor) are accumulated, so this is a
    genuine standing E24 cross-check for every `mode="worldline"`
    `coupling.type='stark_dc'` run, not just the dedicated unit tests in
    ``tests/test_integrator_stark_rotor.py``.

    Parameters
    ----------
    field_fn : CombinedFieldFn
        ``pos -> (E, grad_E)``; the *total* field (see
        ``cliffordclock.integrator.omega``'s WP16 module-docstring note --
        E14b's pivot needs no E11 baseline/perturbation split).
    species_or_coeffs : Species | StarkCoefficients
        Resolved DC-Stark coefficients; see
        :func:`~cliffordclock.integrator.omega.stark_pivot_terms`.
    trajectories : jax.Array, shape (M, T, 3)
        Dense per-atom position trajectories, meters (``T - 1`` steps).
        The only call site today (`mode="worldline"`, `ensemble.regime:
        lattice`) always passes a *static* trajectory (every time sample
        the same quadrature node, `v = 0` at every step) -- see
        ``tests/test_integrator_stark_rotor.py``'s v=0 static-node test
        class for why `ω_boost` is then identically zero and this reduces
        to a pure `B̂_C`-plane rotation; the function itself places no
        such restriction on `trajectories`.
    dtau : float
        Fixed step size ``dτ̃`` (Compton units, E9).
    renorm_every : int, default DEFAULT_RENORM_EVERY
        Renormalize the rotor (E20) every this many steps; see
        :func:`_resolve_renorm_every`.
    bbr_pivot_perturbation : float, default 0.0
        ``(P−1)_BBR`` (E32, WP20); see :func:`_resolve_bbr_pivot_perturbation`.
    quadrupole : QuadrupoleConfig or None, default None
        Electric-quadrupole shift parameters (E34/E35, WP21); `None`
        (default): no quadrupole term, byte-identical to pre-WP21
        behavior. When given, evaluated from each step's `grad_e_mid`
        (E13, already computed for `build_omega_stark`'s spin-connection
        argument) via :func:`_quadrupole_pivot_from_grad` and threaded
        into `build_omega_stark`'s own `quadrupole_pivot_perturbation`.
    gravity : GravityConfig or None, default None
        Gravitational-redshift pivot-term parameters (E36, WP22); `None`
        (default): no gravity term, byte-identical to pre-WP22 behavior.
        When given, evaluated from each step's `pos_mid` via
        :func:`_grav_pivot_from_position` and threaded into
        `build_omega_stark`'s own `grav_pivot_perturbation`.

    Returns
    -------
    EnsembleResult
    """
    quadrupole_theta_j = _resolve_quadrupole_theta_j(quadrupole) if quadrupole is not None else None
    trajectories = jnp.asarray(trajectories, dtype=jnp.float64)
    m = trajectories.shape[0]
    n = trajectories.shape[1] - 1
    dt_phys = dtau * TAU_COMPTON

    def run_one(traj: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r0 = jnp.zeros(16, dtype=jnp.float64).at[IDX_SCALAR].set(1.0)
        init = (
            r0,
            jnp.asarray(0.0, dtype=jnp.float64),  # phase: running sum
            jnp.asarray(0.0, dtype=jnp.float64),  # phase: Kahan compensation
            jnp.asarray(0.0, dtype=jnp.float64),  # phase_rotor: running sum
            jnp.asarray(0.0, dtype=jnp.float64),  # phase_rotor: Kahan compensation
            jnp.asarray(0.0, dtype=jnp.float64),  # max_norm_drift so far
            jnp.asarray(0, dtype=jnp.int64),  # step index (1-based after increment)
        )
        xs = (traj[:-1], traj[1:])

        def body(
            carry: tuple[
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ],
            xs_t: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[
            tuple[
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ],
            None,
        ]:
            r, phase_sum, phase_c, rot_sum, rot_c, max_drift, step_idx = carry
            pos_a, pos_b = xs_t
            pos_mid = 0.5 * (pos_a + pos_b)
            v = (pos_b - pos_a) / dt_phys
            e_mid, grad_e_mid = field_fn(pos_mid)

            quadrupole_value: jnp.ndarray | float = 0.0
            if quadrupole is not None:
                assert quadrupole_theta_j is not None
                theta_au, j = quadrupole_theta_j
                quadrupole_value = _quadrupole_pivot_from_grad(quadrupole, theta_au, j, grad_e_mid)
            grav_value: jnp.ndarray | float = 0.0
            if gravity is not None:
                grav_value = _grav_pivot_from_position(gravity, pos_mid)

            omega = build_omega_stark(
                e_mid,
                grad_e_mid,
                species_or_coeffs,
                v,
                bbr_pivot_perturbation=bbr_pivot_perturbation,
                quadrupole_pivot_perturbation=quadrupole_value,
                grav_pivot_perturbation=grav_value,
            )
            generator = (-0.5 * dtau) * omega
            delta_r = exp_bivector(generator)
            r_next = geometric_product(delta_r, r)

            dphase_scalar = omega[..., IDX_E12] * dtau
            dphase_rotor = rotor_plane_angle(delta_r)

            phase_sum, phase_c = _kahan_add_scalar(phase_sum, phase_c, dphase_scalar)
            rot_sum, rot_c = _kahan_add_scalar(rot_sum, rot_c, dphase_rotor)

            norm_err = jnp.abs(rotor_norm_sq(r_next) - 1.0)
            max_drift = jnp.maximum(max_drift, norm_err)

            step_idx = step_idx + 1
            do_renorm = (step_idx % renorm_every) == 0
            r_next = jax.lax.cond(do_renorm, normalize_rotor, lambda x: x, r_next)

            return (r_next, phase_sum, phase_c, rot_sum, rot_c, max_drift, step_idx), None

        (r_final, phase_sum, _pc, rot_sum, _rc, max_drift, _idx), _ = jax.lax.scan(body, init, xs)
        return r_final, phase_sum, rot_sum, max_drift

    r_final, phase, phase_rotor, max_drift = jax.vmap(run_one)(trajectories)
    t_tilde = n * dtau
    fractional_shift = phase / t_tilde
    norm_error = jnp.abs(jax.vmap(rotor_norm_sq)(r_final) - 1.0)
    return EnsembleResult(
        r_final=r_final,
        phase=phase,
        phase_rotor=phase_rotor,
        fractional_shift=fractional_shift,
        norm_error=norm_error,
        max_norm_drift=max_drift,
        n_steps=jnp.full((m,), n, dtype=jnp.int64),
    )


# ---------------------------------------------------------------------------
# WP19: streaming (O(M)-memory) accumulators for ensemble.regime='classical' +
# mode='direct', both coupling.type values. Fuse velocity-Verlet propagation
# (cliffordclock.ensemble.classical.propagate_verlet's step body,
# reimplemented here -- see _verlet_step's docstring for why) with the
# per-step phase accumulation into ONE jax.lax.scan over the whole ensemble
# at once (carry shape (M, ...)), so no array scales with `n_steps` -- unlike
# the batched path, which first materializes a dense (M, steps+1, 3)
# trajectory (_check_trajectory_memory's "trajectory term") and, for
# coupling.type='stark_dc', additionally evaluates rate_fn once on that
# whole trajectory ("smoother-evaluation term"). Per-step field/rate_fn
# calls are also routed through cliffordclock.fields.smoother.chunked_apply,
# bounding the smoother-evaluation cost independent of ensemble size `M`
# too (not just `n_steps`). See docs/timescales.md's rewritten "Safety net"
# section and _resolve_evaluation_mode for the dispatch this feeds.
# ---------------------------------------------------------------------------


def _verlet_step(
    trap: HarmonicTrap, pos: jnp.ndarray, vel: jnp.ndarray, dt: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One velocity-Verlet step (WP19), shared by both streaming accumulators.

    Reimplements (not imports) the step body of
    :func:`cliffordclock.ensemble.classical._verlet_trajectory` (a
    private helper of that module -- reaching into another module's
    private function is avoided elsewhere in this file too, see
    :func:`_kahan_add_scalar`'s docstring for the same rationale): a
    small, independently-verified reimplementation at this call site
    rather than a cross-module private-function reach.

    Parameters
    ----------
    trap : HarmonicTrap
    pos, vel : jax.Array, shape (M, 3)
    dt : float or jax.Array (scalar)
        Step size, seconds.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(pos_new, vel_new)``, each shape (M, 3).
    """
    acc = trap.acceleration(pos)
    pos_new = pos + vel * dt + 0.5 * acc * dt**2
    acc_new = trap.acceleration(pos_new)
    vel_new = vel + 0.5 * (acc + acc_new) * dt
    return pos_new, vel_new


#: Generic over the scan carry's concrete tuple type (WP19's two streaming
#: accumulators have different, fixed-length carries -- a 4-tuple for the
#: scalar accumulator, a 9-tuple for the rotor one). Bound to
#: ``tuple[jnp.ndarray, ...]`` (variable-length) so `_run_streaming_scan`
#: can still index into it generically (`init[position_index]`); a
#: TypeVar (not a plain `tuple[jnp.ndarray, ...]` parameter/return
#: annotation) is what lets mypy unify `step_fn`'s exact fixed-length
#: carry type with `init`'s and the return value's, rather than rejecting
#: the call as a variance mismatch (a fixed-length tuple is not a
#: `Callable` parameter-position subtype of a variable-length one).
_CarryT = TypeVar("_CarryT", bound="tuple[jnp.ndarray, ...]")


def _run_streaming_scan(
    step_fn: Callable[[_CarryT, None], tuple[_CarryT, None]],
    init: _CarryT,
    n_steps: int,
    *,
    trajectory_stride: int | None,
    position_index: int,
) -> tuple[_CarryT, jnp.ndarray]:
    """Run `step_fn` for `n_steps` with no per-step stacking (WP19's core
    memory bound), optionally recording position snapshots.

    ``jax.lax.scan(step_fn, init, xs=None, length=n_steps)`` (the same
    "no `xs`, just a step count" pattern
    :func:`~cliffordclock.integrator.worldline.kahan_sum` and
    :func:`~cliffordclock.ensemble.classical.propagate_verlet`'s own scan
    use) never materializes an array proportional to `n_steps`: the scan's
    `ys` output is `None` at every step. `trajectory_stride` optionally
    breaks the single scan into a Python-level loop of fixed-size blocks
    (each an independent `lax.scan` call), recording the carry's position
    entry at each block boundary -- `O(M * ceil(n_steps / trajectory_stride))`
    memory, controllable by the caller, rather than the batched path's
    `O(M * n_steps)`.

    Parameters
    ----------
    step_fn : Callable[[carry, None], tuple[carry, None]]
        A `jax.lax.scan`-compatible step function (ignores its second
        argument; `init`/every returned carry share one pytree structure,
        a plain tuple here).
    init : tuple[jax.Array, ...]
        Initial carry.
    n_steps : int
        Number of steps.
    trajectory_stride : int or None
        `None`: only the initial and final positions are recorded
        (`n_snapshots = 2`). An explicit stride: a snapshot every
        `trajectory_stride` steps (plus the initial state and, if
        `n_steps` is not an exact multiple, one final short block).
    position_index : int
        Index into the carry tuple of the position array to snapshot
        (shape (M, 3)).

    Returns
    -------
    tuple[tuple[jax.Array, ...], jax.Array]
        ``(final_carry, snapshots)``; `snapshots` has shape
        ``(M, n_snapshots, 3)``.
    """
    initial_positions = init[position_index]
    if trajectory_stride is None:
        final_carry, _ = jax.lax.scan(step_fn, init, xs=None, length=n_steps)
        snapshots = jnp.stack([initial_positions, final_carry[position_index]], axis=1)
        return final_carry, snapshots

    snapshots_list = [initial_positions]
    carry = init
    remaining = n_steps
    while remaining > 0:
        block = min(trajectory_stride, remaining)
        carry, _ = jax.lax.scan(step_fn, carry, xs=None, length=block)
        snapshots_list.append(carry[position_index])
        remaining -= block
    snapshots = jnp.stack(snapshots_list, axis=1)
    return carry, snapshots


def _stark_scalar_ensemble_streaming(
    rate_fn: fastpath.RateFn,
    trap: HarmonicTrap,
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    dtau: float,
    n_steps: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    trajectory_stride: int | None = None,
) -> tuple[EnsembleResult, jnp.ndarray]:
    """WP19 streaming counterpart of :func:`_stark_scalar_ensemble` (O(M) memory).

    Fuses velocity-Verlet propagation (:func:`_verlet_step`) with the E19
    midpoint / E21-E22 Kahan-summed scalar phase accumulation into a
    single `jax.lax.scan` over the *whole ensemble at once* (carry shape
    ``(M, ...)``, via :func:`_run_streaming_scan`) -- instead of first
    materializing the dense ``(M, steps+1, 3)`` Verlet trajectory
    (:func:`~cliffordclock.ensemble.classical.propagate_verlet`) and then
    calling `rate_fn` **once** on an atom's entire midpoint trajectory
    (:func:`_stark_scalar_ensemble`'s `run_one`, `vmap`-ed over `M`) --
    the whole-trajectory call this WP exists to eliminate (see
    :data:`_TRAJECTORY_MEMORY_FACTOR_SMOOTHER`'s comment). No array this
    function allocates scales with `n_steps`.

    Numerically identical formula to :func:`_stark_scalar_ensemble`: the
    same consecutive-Verlet-position pair at each step (this fuses the
    identical Verlet step body `propagate_verlet` uses, via
    :func:`_verlet_step`) gives the same midpoint position/finite-
    difference velocity, fed to the same `rate_fn`, scaled by the same
    `dtau` and Kahan-summed the same way -- only the *order* of
    "propagate all steps, then evaluate" vs. "propagate one step,
    evaluate, repeat" differs, and Kahan summation's whole purpose is
    making that reordering not matter to more than rounding-noise
    precision (measured bound: see
    ``tests/test_e2e.py``'s streaming-vs-batched agreement tests).

    Parameters
    ----------
    rate_fn : fastpath.RateFn
        ``(pos, v) -> delta_omega_tilde`` (E21); see :func:`_make_stark_rate_fn`.
    trap : HarmonicTrap
        Supplies the Verlet acceleration this function fuses in, rather
        than consuming a precomputed trajectory.
    positions, velocities : jax.Array, shape (M, 3)
        Initial Maxwell-Boltzmann draw (the same inputs
        `propagate_verlet` takes).
    dtau : float
        Fixed step size dτ̃ (Compton units, E9); `dtau * TAU_COMPTON` is
        both the Verlet step and the finite-difference-velocity
        denominator (module docstring interface note 2).
    n_steps : int
        Number of steps.
    chunk_size : int, default DEFAULT_CHUNK_SIZE
        Forwarded to `chunked_apply` for the per-step `rate_fn` call --
        bounds the smoother-evaluation cost independent of `M` too (see
        the module-level WP19 section comment above).
    trajectory_stride : int or None
        See `IntegrationConfig.trajectory_stride`.

    Returns
    -------
    tuple[EnsembleResult, jax.Array]
        ``(ensemble_result, trajectories)`` -- see :func:`_run_streaming_scan`
        for `trajectories`' shape.
    """
    positions = jnp.asarray(positions, dtype=jnp.float64)
    velocities = jnp.asarray(velocities, dtype=jnp.float64)
    m = positions.shape[0]
    dt_phys = dtau * TAU_COMPTON
    zeros_m = jnp.zeros(m, dtype=jnp.float64)

    def step(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], _: None
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
        pos, vel, phase_sum, phase_c = carry
        pos_new, vel_new = _verlet_step(trap, pos, vel, dt_phys)
        pos_mid = 0.5 * (pos + pos_new)
        v_mid = (pos_new - pos) / dt_phys
        domega = jnp.asarray(
            chunked_apply(rate_fn, pos_mid, v_mid, chunk_size=chunk_size), dtype=jnp.float64
        )
        phase_sum, phase_c = _kahan_add_scalar(phase_sum, phase_c, domega * dtau)
        return (pos_new, vel_new, phase_sum, phase_c), None

    init = (positions, velocities, zeros_m, zeros_m)
    (_pos_f, _vel_f, phase, _phase_c), trajectories = _run_streaming_scan(
        step, init, n_steps, trajectory_stride=trajectory_stride, position_index=0
    )

    t_tilde = n_steps * dtau
    fractional_shift = phase / t_tilde
    # Identity rotor (index 0 = scalar 1, E2 basis ordering): no rotor state
    # is ever advanced by this scalar-only accumulator, mirroring
    # _stark_scalar_ensemble's convention exactly.
    identity_rotor = jnp.zeros(16, dtype=jnp.float64).at[0].set(1.0)
    r_final = jnp.broadcast_to(identity_rotor, (m, 16))
    ensemble_result = EnsembleResult(
        r_final=r_final,
        phase=phase,
        phase_rotor=phase,
        fractional_shift=fractional_shift,
        norm_error=zeros_m,
        max_norm_drift=zeros_m,
        n_steps=jnp.full((m,), n_steps, dtype=jnp.int64),
    )
    return ensemble_result, trajectories


def _direct_rotor_ensemble_streaming(
    field_fn: CombinedFieldFn,
    trap: HarmonicTrap,
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    dtau: float,
    mu: jnp.ndarray,
    n_steps: int,
    *,
    renorm_every: int = DEFAULT_RENORM_EVERY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    trajectory_stride: int | None = None,
) -> tuple[EnsembleResult, jnp.ndarray]:
    """WP19 streaming counterpart of `integrate_ensemble` for `coupling.type=
    'linear_mu'` + `mode='direct'` (O(M) memory).

    Fuses velocity-Verlet propagation (:func:`_verlet_step`) with the
    E17-E24 rotor accumulation into one `jax.lax.scan` over the whole
    ensemble at once (carry shape ``(M, ...)``, via
    :func:`_run_streaming_scan`), instead of `propagate_verlet` (dense
    ``(M, steps+1, 3)`` trajectory) followed by
    :func:`~cliffordclock.integrator.worldline.integrate_ensemble`
    (`jax.vmap` of a per-atom `lax.scan` over that dense trajectory).

    Reuses :func:`~cliffordclock.integrator.stepper.rotor_step`
    **unmodified** (WP19 non-goal: no changes to the rotor accumulator)
    for the per-step bivector/rotor update -- already batched over
    arbitrary leading axes (``cliffordclock.cl13``/``omega.py``'s ``...``
    broadcasting convention), so it accepts this function's ``(M, ...)``
    carry directly with no extra `vmap` wrapper needed. This function only
    supplies the fused Verlet-stepping orchestration `worldline.py`
    itself does not (:func:`~cliffordclock.integrator.worldline.integrate_ensemble`
    consumes an already-dense trajectory) -- mirroring how
    :func:`_stark_rotor_ensemble` already builds its own scan around
    `cl13`/`omega.py` primitives directly rather than calling
    `integrate_ensemble` (which is hardwired to a precomputed
    trajectory).

    Parameters
    ----------
    field_fn : CombinedFieldFn
        ``pos -> (E, grad_E)``.
    trap, positions, velocities, dtau, n_steps : see :func:`_stark_scalar_ensemble_streaming`.
    mu : jax.Array, shape (3,)
        Effective dipole moment (E14a), C·m.
    renorm_every : int, default DEFAULT_RENORM_EVERY
        Rotor renormalization cadence (E20).
    chunk_size : int, default DEFAULT_CHUNK_SIZE
        Forwarded to `chunked_apply` for the per-step `field_fn` call.
    trajectory_stride : int or None
        See `IntegrationConfig.trajectory_stride`.

    Returns
    -------
    tuple[EnsembleResult, jax.Array]
        ``(ensemble_result, trajectories)``; see :func:`_run_streaming_scan`.
    """
    positions = jnp.asarray(positions, dtype=jnp.float64)
    velocities = jnp.asarray(velocities, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    m = positions.shape[0]
    dt_phys = dtau * TAU_COMPTON
    zeros_m = jnp.zeros(m, dtype=jnp.float64)
    r0 = jnp.zeros(16, dtype=jnp.float64).at[IDX_SCALAR].set(1.0)
    r0_batch = jnp.broadcast_to(r0, (m, 16))

    CarryT = tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]

    def step(carry: CarryT, _: None) -> tuple[CarryT, None]:
        pos, vel, r, phase_sum, phase_c, rot_sum, rot_c, max_drift, step_idx = carry
        pos_new, vel_new = _verlet_step(trap, pos, vel, dt_phys)
        pos_mid = 0.5 * (pos + pos_new)
        v_mid = (pos_new - pos) / dt_phys
        e_mid, grad_e_mid = chunked_apply(field_fn, pos_mid, chunk_size=chunk_size)
        r_next, dphase = rotor_step(r, e_mid, grad_e_mid, v_mid, mu, dtau)

        phase_sum, phase_c = _kahan_add_scalar(phase_sum, phase_c, dphase.scalar)
        rot_sum, rot_c = _kahan_add_scalar(rot_sum, rot_c, dphase.rotor)

        norm_err = jnp.abs(rotor_norm_sq(r_next) - 1.0)
        max_drift = jnp.maximum(max_drift, norm_err)

        step_idx = step_idx + 1
        do_renorm = (step_idx % renorm_every) == 0
        r_next = jax.lax.cond(do_renorm, normalize_rotor, lambda x: x, r_next)

        return (
            pos_new,
            vel_new,
            r_next,
            phase_sum,
            phase_c,
            rot_sum,
            rot_c,
            max_drift,
            step_idx,
        ), None

    init = (
        positions,
        velocities,
        r0_batch,
        zeros_m,
        zeros_m,
        zeros_m,
        zeros_m,
        zeros_m,
        jnp.asarray(0, dtype=jnp.int64),
    )
    (
        (_pos_f, _vel_f, r_final, phase, _pc, phase_rotor, _rc, max_drift, _idx),
        trajectories,
    ) = _run_streaming_scan(
        step, init, n_steps, trajectory_stride=trajectory_stride, position_index=0
    )

    t_tilde = n_steps * dtau
    fractional_shift = phase / t_tilde
    norm_error = jnp.abs(rotor_norm_sq(r_final) - 1.0)
    ensemble_result = EnsembleResult(
        r_final=r_final,
        phase=phase,
        phase_rotor=phase_rotor,
        fractional_shift=fractional_shift,
        norm_error=norm_error,
        max_norm_drift=max_drift,
        n_steps=jnp.full((m,), n_steps, dtype=jnp.int64),
    )
    return ensemble_result, trajectories


def _resolve_integration_mode(mode: str, regime: str) -> str:
    """Resolve ``integration.mode`` ("auto" or explicit) against `regime`.

    Raises
    ------
    PipelineConfigError
        `mode` is explicit but not valid for `regime` (e.g.
        ``mode="secular"`` with ``regime="lattice"``).
    """
    valid_for_regime = VALID_INTEGRATION_MODES_BY_REGIME[regime]
    if mode == "auto":
        return valid_for_regime[0]
    if mode not in valid_for_regime:
        raise PipelineConfigError(
            f"integration.mode={mode!r} is not valid for ensemble.regime={regime!r}; "
            f"valid modes: {valid_for_regime}"
        )
    return mode


def _resolve_time_s(integration_cfg: IntegrationConfig) -> float:
    """Resolve the total interrogation time ``T`` (seconds) from `integration_cfg`.

    `integration_cfg.time_s` directly if given, else the legacy
    ``steps * dtau * τ_c`` formula (both guaranteed present by
    :func:`_parse_integration`'s validation whenever `time_s` is absent).
    """
    if integration_cfg.time_s is not None:
        return integration_cfg.time_s
    assert integration_cfg.dtau is not None and integration_cfg.steps is not None
    return integration_cfg.steps * integration_cfg.dtau * TAU_COMPTON


def _estimate_max_generator_angle(
    rate_fn: fastpath.RateFn,
    sample_positions: jnp.ndarray,
    sample_velocities: jnp.ndarray,
    dtau: float,
) -> float:
    """Cheap pre-flight estimate of the worst-case per-step rotor generator
    angle at step size `dtau`.

    ``generator = (-0.5 * dtau) * Omega`` (module docstring interface
    note 4, ``integrator/stepper.py::rotor_step``); `Omega`'s dominant
    (`B_hat_C`-plane) coefficient *is* `rate_fn`'s E21 ``delta_omega~``
    (E18: `build_omega`'s `IDX_E12` component), so
    ``0.5 * dtau * max|rate_fn(sample_positions, sample_velocities)|`` is
    a cheap, exact-up-to-the-omitted-boost-term estimate of the worst
    generator angle any step of the run will produce -- no trajectory
    integration required, just one `rate_fn` call over an
    already-available position/velocity sample (the classical ensemble's
    initial Maxwell-Boltzmann draw, or the lattice's static quadrature
    nodes at `v=0`). The boost bivector components (E18, ``e_k ^ e_0``)
    are omitted from this estimate -- they are subleading at realistic
    `v/c` (CONVENTIONS.md's E24 note) -- so this is a coarse guard against
    `exp_bivector` overflow (:data:`MAX_PER_STEP_ROTOR_ANGLE_RAD`), not an
    exact per-step bound.

    Parameters
    ----------
    rate_fn : cliffordclock.integrator.fastpath.RateFn
        ``(pos, v) -> delta_omega~`` (E21); see
        :func:`_make_e14a_rate_fn`.
    sample_positions, sample_velocities : jax.Array, shape (M, 3)
        A cheap position/velocity sample to evaluate `rate_fn` at.
    dtau : float
        Candidate step size ``dτ̃`` (Compton units, E9).

    Returns
    -------
    float
        The estimated worst-case per-step generator angle, radians.
    """
    domega = jnp.asarray(rate_fn(sample_positions, sample_velocities), dtype=jnp.float64)
    max_abs_domega = float(jnp.max(jnp.abs(domega)))
    return 0.5 * abs(dtau) * max_abs_domega


def _resolve_dtau_steps_direct(
    integration_cfg: IntegrationConfig,
    trap: HarmonicTrap,
    rate_fn: fastpath.RateFn,
    sample_positions: jnp.ndarray,
    sample_velocities: jnp.ndarray,
) -> tuple[float, int, bool, str | None]:
    """Resolve ``(dtau, steps, dtau_was_auto_selected, tighten_note)`` for
    `mode="direct"` (Tier B(i)) and (via :func:`_resolve_dtau_steps_worldline`)
    `mode="worldline"`.

    `dtau`: `integration_cfg.dtau` if given, else E31's
    :func:`~cliffordclock.integrator.fastpath.select_dtau` (auto-selected
    from `trap`/`points_per_period`). `steps`: `integration_cfg.steps` if
    given, else computed from `integration_cfg.time_s` and the resolved
    `dtau`. At least one of `time_s` or (`dtau` and `steps`) is guaranteed
    present by :func:`_parse_integration`.

    **Pre-flight generator-angle guard.** Before `steps` is resolved, the estimated
    worst-case per-step rotor generator angle
    (:func:`_estimate_max_generator_angle`, sampled cheaply at
    `sample_positions`/`sample_velocities`) is checked against
    :data:`MAX_PER_STEP_ROTOR_ANGLE_RAD`. If it exceeds the bound:

    - an **auto-selected** `dtau` is silently tightened (scaled down by
      exactly the ratio needed to bring the estimate back to the bound --
      the estimate is linear in `dtau`, so one rescaling suffices) and a
      human-readable `tighten_note` is returned for the caller to fold
      into the report's `uncertainty_notes`;
    - an **explicit** `dtau` is rejected loudly with
      :class:`PipelineConfigError` instead -- silently overriding a
      user's explicit step size would be a worse surprise than failing
      the run.

    This is what stops a realistic E14a `mu` combined with an
    auto-selected large `dtau` from driving `exp_bivector`'s per-step
    argument into its overflow regime; see
    ``docs/timescales.md``.

    WP19 note: this function no longer applies the trajectory-memory
    guard (:func:`_check_trajectory_memory`) itself -- that call moved to
    the caller, since the two call sites now differ: `mode="worldline"`
    (via :func:`_resolve_dtau_steps_worldline`, no streaming alternative)
    still enforces it unconditionally right after calling this function;
    `mode="direct"` + `coupling.type='linear_mu'` (`run_pipeline_full`)
    instead uses the estimate to decide between the batched path and the
    streaming accumulator (WP19, :func:`_direct_rotor_ensemble_streaming`)
    via :func:`_resolve_evaluation_mode`.

    Returns
    -------
    tuple[float, int, bool, str | None]
        ``(dtau, steps, dtau_was_auto, tighten_note)``; `tighten_note` is
        `None` unless an auto-selected `dtau` was tightened.

    Raises
    ------
    PipelineConfigError
        An explicit `dtau` gives an estimated per-step generator angle
        beyond :data:`MAX_PER_STEP_ROTOR_ANGLE_RAD`.
    """
    dtau_was_auto = integration_cfg.dtau is None
    dtau = (
        integration_cfg.dtau
        if integration_cfg.dtau is not None
        else fastpath.select_dtau(trap, integration_cfg.points_per_period)
    )

    estimated_angle = _estimate_max_generator_angle(
        rate_fn, sample_positions, sample_velocities, dtau
    )
    tighten_note: str | None = None
    if estimated_angle > MAX_PER_STEP_ROTOR_ANGLE_RAD:
        if not dtau_was_auto:
            raise PipelineConfigError(
                f"integration.dtau={dtau!r} gives an estimated worst-case per-step rotor "
                f"generator angle of {estimated_angle:.3e} rad (sampled over the ensemble's "
                "initial positions/velocities), exceeding the "
                f"{MAX_PER_STEP_ROTOR_ANGLE_RAD} rad bound "
                "cliffordclock.cl13.exp_bivector's fixed-order (12-term Taylor, 10-halving) "
                "evaluation is only valid within (see docs/timescales.md); beyond it, "
                "exp_bivector can silently return a badly wrong or non-finite rotor even "
                "though the primary scalar phase (E21/E22) stays finite. Reduce "
                "integration.dtau, or use integration.mode='secular' (Tier B(ii)) or "
                "ensemble.regime='lattice' (Tier A, no time stepping at all) instead."
            ) from None
        old_dtau = dtau
        dtau = dtau * (MAX_PER_STEP_ROTOR_ANGLE_RAD / estimated_angle)
        tighten_note = (
            f"integration.dtau auto-tightened from {old_dtau!r} to {dtau!r} (estimated "
            f"worst-case per-step rotor generator angle {estimated_angle:.3e} rad exceeded "
            f"the {MAX_PER_STEP_ROTOR_ANGLE_RAD} rad exp_bivector convergence bound, "
            "see docs/timescales.md)"
        )

    if integration_cfg.steps is not None:
        steps = integration_cfg.steps
    else:
        assert integration_cfg.time_s is not None
        steps = max(1, round(integration_cfg.time_s / (dtau * TAU_COMPTON)))
    return dtau, steps, dtau_was_auto, tighten_note


def _resolve_dtau_steps_worldline(
    integration_cfg: IntegrationConfig,
    trap: HarmonicTrap,
    rate_fn: fastpath.RateFn,
    nodes: jnp.ndarray,
) -> tuple[float, int, bool, str | None]:
    """Resolve ``(dtau, steps, dtau_was_auto, tighten_note)`` for
    `mode="worldline"` (lattice E29-vs-E17 cross-check).

    Same resolution (and the same pre-flight generator-angle guard) as
    :func:`_resolve_dtau_steps_direct`, sampled at the static lattice
    `nodes` with `v = 0` (their only physically meaningful velocity).

    **Pre-flight trajectory-memory guard, applied here unconditionally**
    (WP19 note: `_resolve_dtau_steps_direct` itself no longer applies it --
    see that function's WP19 note). `mode="worldline"` has no streaming
    accumulator (WP19 scope: only `ensemble.regime="classical"` +
    `mode="direct"` does), so this call site keeps the pre-WP19
    hard-reject behavior unchanged: :class:`PipelineConfigError` before
    the dense ``(M, steps + 1, 3)`` broadcast trajectory is ever built.
    """
    dtau, steps, dtau_was_auto, tighten_note = _resolve_dtau_steps_direct(
        integration_cfg, trap, rate_fn, nodes, jnp.zeros_like(nodes)
    )
    _check_trajectory_memory(int(nodes.shape[0]), steps, integration_cfg.max_trajectory_memory_gb)
    return dtau, steps, dtau_was_auto, tighten_note


def _auto_renorm_every() -> int:
    """Auto-select `renorm_every` (E20) for an auto-selected `dtau`.

    `DEFAULT_RENORM_EVERY` (1000, ``worldline.py``) was tuned at Compton-
    scale `dtau ~ 1`, where a single `exp_bivector` call's rotor-
    norm-drift floor is far below `1e-16` and 1000 unrenormalized steps
    stay comfortably under `_AUTO_RENORM_EVERY_DRIFT_BOUND`. At E31's
    large auto-selected `dtau`, that per-call floor is
    `_EXP_BIVECTOR_LARGE_DTAU_NORM_FLOOR` -- no longer negligible -- so
    the same 1000-step cadence can accumulate past the bound; see
    ``docs/timescales.md``. This picks the largest cadence that
    still keeps `_EXP_BIVECTOR_LARGE_DTAU_NORM_FLOOR * renorm_every`
    under `_AUTO_RENORM_EVERY_DRIFT_BOUND`.

    Returns
    -------
    int
        ``max(1, floor(_AUTO_RENORM_EVERY_DRIFT_BOUND / _EXP_BIVECTOR_LARGE_DTAU_NORM_FLOOR))``
        -- 5 at this module's current constants.
    """
    return max(
        1,
        math.floor(_AUTO_RENORM_EVERY_DRIFT_BOUND / _EXP_BIVECTOR_LARGE_DTAU_NORM_FLOOR),
    )


def _resolve_renorm_every(
    integration_cfg: IntegrationConfig, dtau_was_auto: bool
) -> tuple[int, str | None]:
    """Resolve the rotor-renormalization cadence (E20) for a time-stepping mode.

    When `dtau` was auto-selected (E31) *and* the
    caller left `integration.renorm_every` at its YAML default (i.e. did
    not set it explicitly, `IntegrationConfig.renorm_every_was_explicit`
    is `False`), auto-select a tighter cadence (:func:`_auto_renorm_every`)
    instead of `DEFAULT_RENORM_EVERY` -- otherwise the accumulated rotor-
    norm drift (E20) can exceed `_AUTO_RENORM_EVERY_DRIFT_BOUND` at large
    `dtau` (see :func:`_auto_renorm_every`). An explicit `integration.dtau`
    or an explicit `integration.renorm_every` is always honored unchanged
    (backward compatible with every existing config).

    Returns
    -------
    tuple[int, str | None]
        ``(renorm_every, note)``; `note` is `None` unless the cadence was
        auto-tightened.
    """
    if dtau_was_auto and not integration_cfg.renorm_every_was_explicit:
        auto_value = _auto_renorm_every()
        note = (
            f"integration.renorm_every auto-selected={auto_value} (dtau auto-selected; "
            "exp_bivector's large-dtau per-call rotor-norm-drift floor, docs/timescales.md, "
            f"would otherwise exceed the E20 {_AUTO_RENORM_EVERY_DRIFT_BOUND:.0e} target at "
            f"the default cadence of {DEFAULT_RENORM_EVERY})"
        )
        return auto_value, note
    return integration_cfg.renorm_every, None


def _validate_physics(ensemble_result: EnsembleResult) -> None:
    """Coarse sanity check on an integrated ensemble (see :class:`PhysicsValidationError`).

    **Full-field finiteness check.** Every `EnsembleResult` field is checked
    for finiteness, not just `phase`. An earlier version only checked `phase` (the
    primary E21/E22 scalar observable, accumulated directly from
    ``omega[..., IDX_E12] * dtau`` and never routed through
    `exp_bivector`, see `cliffordclock.integrator.worldline`'s module
    docstring) was checked; the rotor-diagnostic fields
    (`phase_rotor`/`r_final`/`norm_error`/`max_norm_drift`, all produced
    by composing `exp_bivector`/`geometric_product` rotor steps, E19-E20/
    E24) were not, so a run whose per-step rotor generator angle exceeded
    `exp_bivector`'s convergence range (a large auto-selected `dtau`
    combined with a realistic E14a `mu`) could leave `max_norm_drift`
    (or any of its rotor-pipeline siblings) `NaN` while `phase` stayed
    finite -- and since NumPy's ``max_norm_err > MAX_ROTOR_NORM_ERROR``
    comparison is `False` whenever `max_norm_err` is `NaN`, the very next
    check silently passed instead of raising (a "NaN evasion" failure
    mode). :func:`_resolve_dtau_steps_direct`'s pre-flight
    guard (`MAX_PER_STEP_ROTOR_ANGLE_RAD`) now catches this earlier for
    the `"direct"`/`"worldline"` modes it covers; this check remains as
    the final backstop for any path that could still produce a
    non-finite result.
    """
    fields_to_check: dict[str, np.ndarray] = {
        "phase": np.asarray(ensemble_result.phase),
        "phase_rotor": np.asarray(ensemble_result.phase_rotor),
        "r_final": np.asarray(ensemble_result.r_final),
        "norm_error": np.asarray(ensemble_result.norm_error),
        "max_norm_drift": np.asarray(ensemble_result.max_norm_drift),
    }
    non_finite_fields = [
        name for name, arr in fields_to_check.items() if not np.all(np.isfinite(arr))
    ]
    if non_finite_fields:
        raise PhysicsValidationError(
            "non-finite value(s) in the integrated ensemble result "
            f"({', '.join(non_finite_fields)}; NaN/Inf). The most likely cause is a "
            "per-step rotor generator angle outside exp_bivector's fixed-order Taylor "
            "convergence range (see docs/timescales.md and "
            "cliffordclock.pipeline.MAX_PER_STEP_ROTOR_ANGLE_RAD) -- typically a large "
            "integration.dtau (explicit, or auto-selected for an unusually slow trap) "
            "combined with a large coupling.mu / field magnitude. Reduce integration.dtau, "
            "or use integration.mode='secular' or ensemble.regime='lattice' "
            "(integration.mode='fast_path', the default there), neither of which ever "
            "calls exp_bivector."
        )

    max_norm_err = float(np.max(fields_to_check["max_norm_drift"]))
    if max_norm_err > MAX_ROTOR_NORM_ERROR:
        raise PhysicsValidationError(
            f"rotor norm drift {max_norm_err:.3e} exceeds the sanity threshold "
            f"{MAX_ROTOR_NORM_ERROR:.0e} (E20); the integration may be unstable "
            "(check integration.dtau / integration.steps / integration.renorm_every)"
        )


def _config_hash(config: PipelineConfig) -> str:
    """A short, deterministic provenance hash of `config` (report's `config_hash`)."""
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _package_version() -> str:
    """`cliffordclock`'s installed package version.

    Duplicates `cliffordclock.analytics.report._package_version` (private
    to that module) rather than importing it, for the same reason that
    function gives for not importing `cliffordclock.__version__`: keeping
    this module's own provenance lookup self-contained. `REPORT_SCHEMA_VERSION`/
    `CONVENTIONS_VERSION` below *are* imported (public constants -- no
    reason to duplicate a literal that must stay in sync automatically).
    """
    try:
        return version("cliffordclock")
    except PackageNotFoundError:  # pragma: no cover - only hit if package not installed
        return "0.0.0+unknown"


def _build_report(
    phi: jnp.ndarray,
    species: Species,
    t_interrogation_s: float,
    ensemble_type: str,
    weights: jnp.ndarray | None,
    config_hash: str,
    uncertainty_notes: str,
) -> MetrologyReport:
    """Build the `MetrologyReport`, tolerating the M=1 (single-atom) case.

    `cliffordclock.analytics.build_report` requires >= 2 atoms: its
    variance-based statistics (`shift_std_error`, `t2_star_s`) are
    undefined for a single atom by construction of `weighted_phase_stats`
    (E25's sample variance needs >= 2 effective samples) -- this is
    correct behavior for that function (see the M=1 boundary tests
    in ``tests/test_analytics_stats.py``/``tests/test_analytics_report.py``),
    not a bug to work around there.

    A single static atom / single lattice node is nonetheless a
    legitimate pipeline input (CONVENTIONS.md V2's closed-form validation
    case is exactly this), so this wrapper does not let that ValueError
    propagate out of :func:`run_pipeline_full`: for `ensemble_size == 1`
    it reports `mean_fractional_shift` alone (still exact, E23, since
    `mean_fractional_shift` itself works at M=1) with `shift_std_error`/
    `t2_star_s` set to NaN and a note recorded, instead of the
    `MetrologyReport` `build_report` would have produced.
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    if phi_arr.shape[0] >= 2:
        return build_report(
            phi_arr,
            species,
            t_interrogation_s,
            ensemble_type,
            weights=weights,
            config_hash=config_hash,
            uncertainty_notes=uncertainty_notes,
        )

    shift = mean_fractional_shift(phi_arr, t_interrogation_s, weights)
    note = "single-atom ensemble (M=1): shift_std_error/t2_star_s are undefined (NaN)."
    combined_notes = f"{uncertainty_notes} {note}".strip() if uncertainty_notes else note
    return MetrologyReport(
        report_schema=REPORT_SCHEMA_VERSION,
        conventions_version=CONVENTIONS_VERSION,
        package_version=_package_version(),
        generated_at_utc=datetime.now(UTC).isoformat(),
        config_hash=config_hash,
        species_name=species.name,
        ensemble_type=ensemble_type,
        ensemble_size=1,
        interrogation_time_s=float(t_interrogation_s),
        mean_fractional_shift=shift,
        shift_std_error=float("nan"),
        t2_star_s=float("nan"),
        uncertainty_notes=combined_notes,
    )


def _line_profile_arrays(
    phi: jnp.ndarray,
    weights: jnp.ndarray | None,
    t_interrogation_s: float,
    n_time_samples: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coherence function (E26) + line profile (E28) on a uniform time grid."""
    dt_s = t_interrogation_s / (n_time_samples - 1)
    t_grid_s = np.arange(n_time_samples, dtype=np.float64) * dt_s
    coherence = coherence_function(phi, t_interrogation_s, t_grid_s, weights=weights)
    return line_profile(coherence, dt_s)


# ---------------------------------------------------------------------------
# WP22 Part 2: the per-site frequency map (the Bothwell observable) and its
# gate-edit-4 dispersion-labeling numbers -- `ensemble.regime='lattice_extended'`
# only. Deliberately a SEPARATE output from `MetrologyReport` (not a schema
# change to that frozen, versioned dataclass): every other regime's report
# shape/byte-exactness is untouched by this WP.
# ---------------------------------------------------------------------------

#: Gate edit 4 (G9 sign-off A4#2, the project's theory sign-off record (G9)):
#: for the extended-lattice mode, deterministic-gradient broadening must be
#: labeled distinctly from stochastic dephasing in every output that reports
#: spread/T2*/linewidth, so a reader cannot double-count the gradient (once
#: in the per-site map, once in a linewidth) or misread it as decoherence.
#: Test-pinned wording (`tests/test_lattice_extended.py`), extending the
#: showcase's existing SEM-vs-T2* discipline (`docs/tutorial.md`).
LATTICE_EXTENDED_DISPERSION_LABEL_NOTE = (
    "lattice_extended dispersion labeling (CONVENTIONS.md section 15, G9 sign-off A4#2): "
    "this run's t2_star_s/shift_std_error (MetrologyReport) are computed over the FULL "
    "site-and-motional ensemble and therefore include the DETERMINISTIC per-site height "
    "gradient (higher/lower sites tick at a systematically different rate -- not a "
    "stochastic process, and in principle refocusable) mixed together with any genuine "
    "stochastic (motional-quadrature) spread. Use site_map.slope_per_m (the headline "
    "gradient) and site_map.total_spread_fractional / "
    "site_map.gradient_removed_residual_spread_fractional "
    "to separate them: total_spread_fractional is the SAME combined (deterministic + "
    "stochastic) number as t2_star_s in fractional-shift units, while "
    "gradient_removed_residual_spread_fractional is what remains after subtracting the "
    "best-fit linear gradient from each site's own mean shift -- the stochastic-only "
    "part. Do not read t2_star_s alone as a decoherence time for this ensemble regime: "
    "most of its narrowing here is the deterministic gradient, not irreversible dephasing."
)


@dataclass(frozen=True)
class SiteMapEntry:
    """One extended-lattice site's entry in the per-site frequency map
    (WP22 Part 2, CONVENTIONS.md section 15 -- "the Bothwell observable").

    Attributes
    ----------
    site_index : int
        Index into the site array, `0 <= site_index < n_sites`.
    position_m : tuple[float, float, float]
        This site's trap center, meters.
    offset_m : float
        Signed coordinate along `ensemble.site_axis` (unit-normalized),
        relative to `trap.center` -- the natural x-axis `slope_per_m` is
        fit against (:data:`LatticeExtendedSiteMap.slope_per_m`).
    weight : float
        Normalized (sum-to-1 across all sites) site-occupation envelope
        weight (``ensemble.site_envelope``).
    mean_fractional_shift : float
        This site's own weighted mean fractional shift ``Δν/ν₀``, over
        ONLY this site's local Hermite-Gauss motional-quadrature nodes
        (E23-style weighted mean, restricted to this site).
    """

    site_index: int
    position_m: tuple[float, float, float]
    offset_m: float
    weight: float
    mean_fractional_shift: float


@dataclass(frozen=True)
class LatticeExtendedSiteMap:
    """The per-site frequency map plus gate-edit-4 dispersion-labeling
    numbers for an ``ensemble.regime='lattice_extended'`` run (WP22 Part
    2, CONVENTIONS.md section 15).

    Attributes
    ----------
    site_axis : tuple[float, float, float]
        `ensemble.site_axis`, unit-normalized.
    sites : tuple[SiteMapEntry, ...]
        One entry per site, in site-index order.
    slope_per_m : float
        **The headline number** (G9 sign-off B2/gate edit 6): the
        best-fit (weighted least squares, weighted by `SiteMapEntry.weight`)
        linear-gradient slope ``d(mean_fractional_shift)/d(offset_m)``
        across the sites' own means -- the quantity the Bothwell et al.
        2022 measured-map slope is compared against
        (`benchmarks/run_bothwell_redshift.py`).
    intercept : float
        The same fit's intercept (the fitted mean fractional shift at
        `offset_m = 0`).
    total_spread_fractional : float
        The full ensemble's weighted standard deviation of the fractional
        shift (site occupation x local motional quadrature weights
        combined), dimensionless -- the SAME quantity `t2_star_s`
        (`MetrologyReport`) is derived from, in fractional-shift rather
        than phase/time units (`t2_star_s = sqrt(2) * T /
        (total_spread_fractional * T_tilde)`). Includes the deterministic
        per-site gradient (see :data:`LATTICE_EXTENDED_DISPERSION_LABEL_NOTE`).
    gradient_removed_residual_spread_fractional : float
        The weighted standard deviation of each site's `mean_fractional_shift`
        AFTER subtracting the best-fit line (`slope_per_m * offset_m +
        intercept`) -- gate edit 4's "gradient-removed residual spread":
        what remains once the deterministic linear gradient is fit out,
        i.e. the non-deterministic (stochastic/higher-order) part of the
        site-to-site spread. `0.0` when there are fewer than 2 sites
        (no gradient to fit).
    dispersion_label_note : str
        Always :data:`LATTICE_EXTENDED_DISPERSION_LABEL_NOTE`, verbatim
        (gate edit 4: "test-pinned wording, same pattern as the ion
        notes").
    """

    site_axis: tuple[float, float, float]
    sites: tuple[SiteMapEntry, ...]
    slope_per_m: float
    intercept: float
    total_spread_fractional: float
    gradient_removed_residual_spread_fractional: float
    dispersion_label_note: str


def _weighted_linear_fit(
    x: NDArray[np.float64], y: NDArray[np.float64], w: NDArray[np.float64]
) -> tuple[float, float]:
    """Weighted least-squares fit ``y ~= slope*x + intercept``.

    Standard closed-form weighted normal equations (centered on the
    weighted means of `x`/`y`, avoiding the catastrophic-cancellation-
    prone "sum of squares minus square of sum" form). Degenerate inputs
    (fewer than 2 points, or all `x` identical -- zero variance in `x`,
    so no gradient is resolvable) return ``(0.0, weighted_mean(y))``: a
    flat "fit" at the data's own weighted mean, which is the well-defined
    least-squares answer when the design matrix has no `x`-direction
    information to give a slope.

    Parameters
    ----------
    x, y, w : numpy.ndarray, shape (N,)
        Sample points, values, and weights (need not be pre-normalized).

    Returns
    -------
    tuple[float, float]
        ``(slope, intercept)``.
    """
    w_sum = math.fsum(w.tolist())
    x_bar = math.fsum((w * x).tolist()) / w_sum
    y_bar = math.fsum((w * y).tolist()) / w_sum
    if x.shape[0] < 2:
        return 0.0, y_bar
    s_xx = math.fsum((w * (x - x_bar) ** 2).tolist())
    if s_xx <= 0.0:
        return 0.0, y_bar
    s_xy = math.fsum((w * (x - x_bar) * (y - y_bar)).tolist())
    slope = s_xy / s_xx
    intercept = y_bar - slope * x_bar
    return slope, intercept


def _weighted_std_or_nan(x: NDArray[np.float64], w: NDArray[np.float64]) -> float:
    """Weighted standard deviation of `x` (reusing `weighted_phase_stats`'s
    bias-corrected reliability-weights estimator, E25's precision
    discipline, for a generic array -- not literally a phase). `0.0` when
    fewer than 2 points; `float('nan')` if the effective sample size
    implied by `w` is still below 2 (mirrors `shift_std_error`'s M=1
    convention -- undefined, not silently zero)."""
    if x.shape[0] < 2:
        return 0.0
    try:
        stats = weighted_phase_stats(x, w)
    except ValueError:
        return float("nan")
    return math.sqrt(stats.variance)


def _build_site_map(
    ensemble_cfg: EnsembleConfig,
    geometry: ExtendedLatticeGeometry,
    ensemble_result: EnsembleResult,
) -> LatticeExtendedSiteMap:
    """Assemble the WP22 Part 2 per-site frequency map from a completed
    `ensemble.regime='lattice_extended'` run.

    Parameters
    ----------
    ensemble_cfg : EnsembleConfig
        The run's `ensemble:` config (`n_sites`/`site_axis`).
    geometry : ExtendedLatticeGeometry
        The site/node geometry `extended_lattice_nodes` built.
    ensemble_result : EnsembleResult
        The completed integration result (`fast_path` or `worldline`);
        `.fractional_shift` and `.weights`-compatible ordering must match
        `geometry.nodes`'s site-major layout.

    Returns
    -------
    LatticeExtendedSiteMap
    """
    assert ensemble_cfg.n_sites is not None  # enforced by _parse_ensemble
    n_sites = ensemble_cfg.n_sites
    local_weights = np.asarray(geometry.local_weights, dtype=np.float64)
    n_local = local_weights.shape[0]

    shift = np.asarray(ensemble_result.fractional_shift, dtype=np.float64).reshape(n_sites, n_local)
    site_mean_shift = shift @ local_weights  # (n_sites,): each site's own weighted mean (E23-style)

    site_offsets_m = np.asarray(geometry.site_offsets_m, dtype=np.float64)
    site_weights = np.asarray(geometry.site_weights, dtype=np.float64)
    site_centers = np.asarray(geometry.site_centers, dtype=np.float64)

    slope, intercept = _weighted_linear_fit(site_offsets_m, site_mean_shift, site_weights)
    residual = site_mean_shift - (slope * site_offsets_m + intercept)
    gradient_removed_residual_spread = _weighted_std_or_nan(residual, site_weights)

    full_weights = np.asarray(geometry.weights, dtype=np.float64)
    full_shift = np.asarray(ensemble_result.fractional_shift, dtype=np.float64)
    total_spread = _weighted_std_or_nan(full_shift, full_weights)

    axis = np.asarray(ensemble_cfg.site_axis, dtype=np.float64)
    axis_unit = axis / np.linalg.norm(axis)
    axis_hat = (float(axis_unit[0]), float(axis_unit[1]), float(axis_unit[2]))

    sites = tuple(
        SiteMapEntry(
            site_index=i,
            position_m=(
                float(site_centers[i, 0]),
                float(site_centers[i, 1]),
                float(site_centers[i, 2]),
            ),
            offset_m=float(site_offsets_m[i]),
            weight=float(site_weights[i]),
            mean_fractional_shift=float(site_mean_shift[i]),
        )
        for i in range(n_sites)
    )
    return LatticeExtendedSiteMap(
        site_axis=axis_hat,
        sites=sites,
        slope_per_m=float(slope),
        intercept=float(intercept),
        total_spread_fractional=total_spread,
        gradient_removed_residual_spread_fractional=gradient_removed_residual_spread,
        dispersion_label_note=LATTICE_EXTENDED_DISPERSION_LABEL_NOTE,
    )


class PipelineResult(NamedTuple):
    """Full output of :func:`run_pipeline_full`.

    `run_pipeline` (the primary entry point) returns only `.report`;
    this richer result is exposed for callers (tests, notebooks, the CLI)
    that need the underlying ensemble diagnostics or line-profile arrays
    too.

    Attributes
    ----------
    report : MetrologyReport
        The assembled metrology report (E23, E25, E27).
    ensemble_result : EnsembleResult
        Raw per-atom integrator output, including both the primary
        scalar phase and the E24 rotor cross-check phase.
    trajectories : jax.Array
        The positions integrated over: shape ``(M, T, 3)`` for
        `regime="classical"` with `integration.mode="direct"` **and the
        batched accumulator** (the full discretized Verlet trajectory),
        ``(M, 3)`` (static per-node positions) for `regime="lattice"`,
        and ``(M, 3)`` (initial conditions only -- E30 integrates a
        closed-form orbit, not a discretized trajectory) for
        `regime="classical"` with `integration.mode="secular"`. **WP19:**
        for `regime="classical"` + `mode="direct"` with the *streaming*
        accumulator (`integration.evaluation` resolved to `"streaming"`),
        shape is ``(M, 2, 3)`` by default (initial + final position only
        -- the streaming path never materializes the full dense
        trajectory, that is the point) or
        ``(M, ceil(steps / integration.trajectory_stride) + 1, 3)`` when
        `integration.trajectory_stride` is set; see
        `IntegrationConfig.trajectory_stride`.
    weights : jax.Array or None
        Ensemble weights (``None`` for classical: uniform ``1/M``;
        quadrature weights for lattice).
    line_profile_freqs_hz, line_profile_amplitude : numpy.ndarray
        The line profile (E28), as written to the line-profile CSV.
    site_map : LatticeExtendedSiteMap or None
        The per-site frequency map (WP22 Part 2) for
        `regime="lattice_extended"` runs; `None` for every other regime.
    """

    report: MetrologyReport
    ensemble_result: EnsembleResult
    trajectories: jnp.ndarray
    weights: jnp.ndarray | None
    line_profile_freqs_hz: NDArray[np.float64]
    line_profile_amplitude: NDArray[np.float64]
    site_map: LatticeExtendedSiteMap | None = None


def run_pipeline_full(config: PipelineConfig) -> PipelineResult:
    """Run the full pipeline, returning the report plus underlying diagnostics.

    load/synthesize field -> build ensemble -> integrate -> analyze
    -> report. No new physics: every step delegates to the
    ``fields``, ``ensemble``, ``integrator``, and ``analytics``
    modules.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    PipelineResult

    Raises
    ------
    PipelineConfigError
        `config` references an unknown species/synthetic-field kind, or a
        malformed CSV field file.
    PhysicsValidationError
        The integrated result fails the coarse sanity check in
        :func:`_validate_physics`.
    """
    try:
        species = get_species(config.species)
    except KeyError as exc:
        raise PipelineConfigError(str(exc)) from exc

    trap = HarmonicTrap(omega_xyz=config.trap.omega_xyz, center=config.trap.center)
    field_fn, n_smoother_fit_points = _build_field_fn(config.field_config)

    mu: jnp.ndarray | None
    stark_coeffs: Species | StarkCoefficients | None
    coupling_note: str | None
    bbr_note: str | None
    quadrupole_note: str | None
    gravity_note: str | None
    # WP20 (E32/E33): resolved once per run, before the coupling.type branch,
    # since `_parse_environment`/`PipelineConfig.from_dict`'s cross-field
    # validation already guarantees radiation_temperature_k is None whenever
    # coupling.type='linear_mu' -- bbr_value stays the exact-zero default in
    # that branch (never computed against `species`, matching the "no
    # coupling-agnostic BBR quantity" design note).
    bbr_value, bbr_note = _resolve_bbr_pivot_perturbation(config.environment, species)
    # WP21 (E34/E35): same cross-field validation guarantees config.quadrupole
    # is None whenever coupling.type='linear_mu'.
    quadrupole_note = None
    if config.quadrupole is not None:
        theta_au, j = _resolve_quadrupole_theta_j(config.quadrupole)
        quadrupole_note = _quadrupole_provenance_note(config.quadrupole, theta_au, j)
    # WP22 (E36): same cross-field validation guarantees config.environment.gravity
    # is None whenever coupling.type='linear_mu'.
    gravity = config.environment.gravity
    gravity_note = _gravity_provenance_note(gravity) if gravity is not None else None
    if config.coupling.type == "linear_mu":
        assert config.coupling.mu is not None  # enforced by _parse_coupling
        mu = jnp.asarray(config.coupling.mu, dtype=jnp.float64)
        stark_coeffs = None
        rate_fn = _make_e14a_rate_fn(field_fn, mu)
        coupling_note = None
    else:
        assert config.coupling.type == "stark_dc"
        mu = None
        try:
            stark_coeffs = _resolve_stark_coupling(config.coupling, species)
            k_s = stark_coeffs.resolve_stark_coefficient_hz_per_v2_m2()
        except ValueError as exc:
            raise PipelineConfigError(str(exc)) from exc
        rate_fn = _make_stark_rate_fn(
            field_fn,
            stark_coeffs,
            bbr_pivot_perturbation=bbr_value,
            quadrupole=config.quadrupole,
            gravity=gravity,
        )
        coupling_note = _stark_coupling_provenance_note(config.coupling, species, k_s)

    mode = _resolve_integration_mode(config.integration.mode, config.ensemble.regime)

    weights: jnp.ndarray | None
    mode_note: str
    geometry: ExtendedLatticeGeometry | None = None
    if config.ensemble.regime == "classical":
        assert config.ensemble.size is not None  # enforced by _parse_ensemble
        key = jax.random.PRNGKey(config.ensemble.seed)
        positions, velocities = sample_maxwell_boltzmann(
            key, species, config.ensemble.temperature_uK, config.ensemble.size, trap
        )
        weights = None

        if mode == "direct" and config.coupling.type == "stark_dc":
            # Tier B(i) under coupling.type='stark_dc': no rotor -- see
            # _stark_scalar_ensemble's docstring (omega.py/worldline.py
            # are not touched here, so E14b has no Omega-bivector
            # construction to drive the rotor integrator with; this is a
            # deliberate, deferred design choice, not a gap).
            dtau, n_steps, dtau_was_auto = _resolve_dtau_steps_scalar(config.integration, trap)
            dt_seconds = dtau * TAU_COMPTON
            n_atoms = int(positions.shape[0])
            _traj_gb, _smoother_gb, estimated_gb = _estimate_trajectory_memory_gb(
                n_atoms, n_steps, n_smoother_fit_points=n_smoother_fit_points
            )
            eval_mode, dispatch_note = _resolve_evaluation_mode(
                config.integration.evaluation,
                estimated_gb,
                config.integration.max_trajectory_memory_gb,
            )
            ensemble_type = "classical_direct"
            if eval_mode == "batched":
                # Re-derive the same estimate and raise with the full,
                # actionable message when requested explicitly
                # (evaluation="batched") or (defensively) if it somehow
                # still resolved to "batched" over budget.
                _check_trajectory_memory(
                    n_atoms,
                    n_steps,
                    config.integration.max_trajectory_memory_gb,
                    n_smoother_fit_points=n_smoother_fit_points,
                )
                trajectories = propagate_verlet(trap, positions, velocities, dt_seconds, n_steps)
                ensemble_result = _stark_scalar_ensemble(rate_fn, trajectories, dtau)
            else:
                assert eval_mode == "streaming"
                _check_streaming_memory(n_atoms, config.integration.max_trajectory_memory_gb)
                ensemble_result, trajectories = _stark_scalar_ensemble_streaming(
                    rate_fn,
                    trap,
                    positions,
                    velocities,
                    dtau,
                    n_steps,
                    trajectory_stride=config.integration.trajectory_stride,
                )
            t_interrogation_s = n_steps * dt_seconds
            # WP19: the "evaluation=..." marker is only appended for
            # eval_mode="streaming" -- when the batched path runs (the
            # unmodified pre-WP19 default), mode_note stays byte-identical
            # to its pre-WP19 text (see
            # tests/test_e2e.py::test_step0_linear_mu_output_unchanged_from_pre_step0_behavior's
            # analogous byte-exactness contract for the linear_mu branch
            # below; the shipped examples' outputs must not change when
            # they run batched, per the WP19 acceptance criteria).
            mode_note = (
                f"integration.mode=direct dtau={dtau!r} steps={n_steps} "
                f"dtau_auto_selected={dtau_was_auto} (E21/E22 "
                "scalar-only phase accumulation via rate_fn, no rotor/exp_bivector for "
                f"coupling.type=stark_dc; E31 points_per_period="
                f"{config.integration.points_per_period})"
            )
            if eval_mode == "streaming":
                mode_note = f"{mode_note} evaluation=streaming"
            if dispatch_note is not None:
                mode_note = f"{mode_note}; {dispatch_note}"
        elif mode == "direct":
            # Tier B(i): the rotor integrator (E17-E24), dtau auto-selected
            # via E31 (fastpath.select_dtau) when omitted (module docstring,
            # interface notes 2-3). The pre-flight generator-angle/renorm
            # safety net (interface note 4) pre-flight-checks the per-step
            # rotor generator angle and (for auto-selected dtau) the
            # renorm_every cadence.
            dtau, n_steps, dtau_was_auto, tighten_note = _resolve_dtau_steps_direct(
                config.integration, trap, rate_fn, positions, velocities
            )
            dt_seconds = dtau * TAU_COMPTON
            n_atoms = int(positions.shape[0])
            _traj_gb, _smoother_gb, estimated_gb = _estimate_trajectory_memory_gb(n_atoms, n_steps)
            eval_mode, dispatch_note = _resolve_evaluation_mode(
                config.integration.evaluation,
                estimated_gb,
                config.integration.max_trajectory_memory_gb,
            )
            ensemble_type = "classical_direct"
            renorm_every, renorm_note = _resolve_renorm_every(config.integration, dtau_was_auto)
            assert mu is not None  # linear_mu coupling: enforced by _parse_coupling
            if eval_mode == "batched":
                _check_trajectory_memory(
                    n_atoms, n_steps, config.integration.max_trajectory_memory_gb
                )
                trajectories = propagate_verlet(trap, positions, velocities, dt_seconds, n_steps)
                ensemble_result = integrate_ensemble(
                    field_fn, trajectories, dtau, mu, renorm_every=renorm_every
                )
            else:
                assert eval_mode == "streaming"
                _check_streaming_memory(n_atoms, config.integration.max_trajectory_memory_gb)
                ensemble_result, trajectories = _direct_rotor_ensemble_streaming(
                    field_fn,
                    trap,
                    positions,
                    velocities,
                    dtau,
                    mu,
                    n_steps,
                    renorm_every=renorm_every,
                    trajectory_stride=config.integration.trajectory_stride,
                )
            t_interrogation_s = n_steps * dt_seconds
            # WP19: same "only mention evaluation for streaming" rule as
            # the stark_dc branch above -- preserves the batched path's
            # pre-WP19 byte-exact mode_note text.
            mode_note = (
                f"integration.mode=direct dtau={dtau!r} steps={n_steps} "
                f"dtau_auto_selected={dtau_was_auto} renorm_every={renorm_every} "
                f"(E19, E31 points_per_period={config.integration.points_per_period})"
            )
            if eval_mode == "streaming":
                mode_note = f"{mode_note} evaluation=streaming"
            if tighten_note is not None:
                mode_note = f"{mode_note}; {tighten_note}"
            if renorm_note is not None:
                mode_note = f"{mode_note}; {renorm_note}"
            if dispatch_note is not None:
                mode_note = f"{mode_note}; {dispatch_note}"
        else:
            assert mode == "secular"
            # Tier B(ii): E30 secular averaging (isotropic-trap periodic motion).
            if config.integration.time_s is None:
                raise PipelineConfigError(
                    "integration.time_s is required for integration.mode='secular'"
                )
            t_interrogation_s = config.integration.time_s
            trajectories = positions  # Initial conditions only (E30: closed-form orbit).
            ensemble_type = "classical_secular_average"
            # secular_average_shift_ensemble (fastpath.py) has the same
            # dense-trajectory shape as direct/worldline, just with
            # `steps` fixed at `points_per_period` (not `time_s`-driven):
            # secular_average_shift builds one closed-form-orbit
            # trajectory of shape (points_per_period + 1, 3) per atom
            # (`_shm_trajectory`), `vmap`-ed over `M` -- and unlike
            # `time_s`, `points_per_period` has no upper bound enforced
            # anywhere else (fastpath.py is out of scope for this guard;
            # checked here at the pipeline call site instead, before
            # dispatch). Reuses the base trajectory-only term
            # (n_smoother_fit_points intentionally omitted here: the
            # smoother-evaluation term is calibrated for and scoped to
            # _stark_scalar_ensemble's single whole-trajectory rate_fn
            # call, see _TRAJECTORY_MEMORY_FACTOR_SMOOTHER's comment;
            # extending it to this call site is deferred, matching the
            # docs/timescales.md note).
            _check_trajectory_memory(
                int(positions.shape[0]),
                config.integration.points_per_period,
                config.integration.max_trajectory_memory_gb,
            )
            try:
                ensemble_result = fastpath.secular_average_shift_ensemble(
                    rate_fn,
                    trap,
                    positions,
                    velocities,
                    t_interrogation_s,
                    points_per_period=config.integration.points_per_period,
                )
            except ValueError as exc:
                raise PipelineConfigError(str(exc)) from exc
            mode_note = (
                f"integration.mode=secular T={t_interrogation_s!r}s "
                f"(E30, points_per_period={config.integration.points_per_period})"
            )
    elif config.ensemble.regime == "lattice":
        assert config.ensemble.motional_n is not None  # enforced by _parse_ensemble
        nodes, node_weights = hermite_gaussian_nodes(
            species, trap, config.ensemble.motional_n, config.ensemble.n_quad
        )
        trajectories = nodes
        weights = node_weights

        if mode == "fast_path":
            # Tier A: E29 exact quadrature-expectation fast path -- O(1) cost in T.
            # Scope per E29: Stark/field shift only -- static v=0 nodes carry
            # no motional second-order Doppler.
            t_interrogation_s = _resolve_time_s(config.integration)
            ensemble_type = "lattice_fast_path"
            ensemble_result = fastpath.lattice_shift_expectation(
                rate_fn, nodes, node_weights, t_interrogation_s
            )
            mode_note = f"integration.mode=fast_path T={t_interrogation_s!r}s (E29, exact)"
        elif mode == "worldline" and config.coupling.type == "stark_dc":
            # WP16: the true Cl(1,3) rotor path (E17-E24) for coupling.type=
            # 'stark_dc' -- replaces the scalar stand-in previously used
            # here (see the module docstring's mode table and
            # _stark_rotor_ensemble's docstring). Static lattice nodes
            # (v=0 always) mean ω_boost is identically zero for this call
            # site specifically -- tests/test_integrator_stark_rotor.py
            # exercises ω_boost directly (nonzero v) as a standalone
            # integrator-level cross-check, since no config here ever
            # drives a moving atom through this branch. Same pre-flight
            # generator-angle/renorm safety net as the linear_mu rotor
            # path below, via the same (already coupling-agnostic)
            # resolvers.
            assert stark_coeffs is not None  # stark_dc coupling: resolved above
            dtau, n_steps, dtau_was_auto, tighten_note = _resolve_dtau_steps_worldline(
                config.integration, trap, rate_fn, nodes
            )
            dt_seconds = dtau * TAU_COMPTON
            traj_dense = jnp.broadcast_to(nodes[:, None, :], (nodes.shape[0], n_steps + 1, 3))
            ensemble_type = "lattice_worldline_crosscheck"
            renorm_every, renorm_note = _resolve_renorm_every(config.integration, dtau_was_auto)
            ensemble_result = _stark_rotor_ensemble(
                field_fn,
                stark_coeffs,
                traj_dense,
                dtau,
                renorm_every=renorm_every,
                bbr_pivot_perturbation=bbr_value,
                quadrupole=config.quadrupole,
                gravity=gravity,
            )
            t_interrogation_s = n_steps * dt_seconds
            mode_note = (
                f"integration.mode=worldline dtau={dtau!r} steps={n_steps} "
                f"renorm_every={renorm_every} dtau_auto_selected={dtau_was_auto} "
                "(E17-E24 Cl(1,3) rotor, coupling.type=stark_dc: true rotor path, WP16)"
            )
            if tighten_note is not None:
                mode_note = f"{mode_note}; {tighten_note}"
            if renorm_note is not None:
                mode_note = f"{mode_note}; {renorm_note}"
        else:
            assert mode == "worldline"
            # Explicit cross-check: the rotor integrator (E17-E24) on the
            # same static nodes -- must agree with fast_path exactly (E29).
            # Same pre-flight generator-angle/renorm safety net as
            # mode="direct" above.
            dtau, n_steps, dtau_was_auto, tighten_note = _resolve_dtau_steps_worldline(
                config.integration, trap, rate_fn, nodes
            )
            dt_seconds = dtau * TAU_COMPTON
            ensemble_type = "lattice_worldline_crosscheck"
            renorm_every, renorm_note = _resolve_renorm_every(config.integration, dtau_was_auto)
            assert mu is not None  # linear_mu coupling: enforced by _parse_coupling
            ensemble_result = integrate_ensemble(
                field_fn,
                trajectories,
                dtau,
                mu,
                renorm_every=renorm_every,
                n_steps=n_steps,
            )
            t_interrogation_s = n_steps * dt_seconds
            mode_note = (
                f"integration.mode=worldline dtau={dtau!r} steps={n_steps} "
                f"renorm_every={renorm_every} (E17-E24)"
            )
            if tighten_note is not None:
                mode_note = f"{mode_note}; {tighten_note}"
            if renorm_note is not None:
                mode_note = f"{mode_note}; {renorm_note}"
    else:
        assert config.ensemble.regime == "lattice_extended"
        assert config.ensemble.motional_n is not None  # enforced by _parse_ensemble
        assert config.ensemble.n_sites is not None  # enforced by _parse_ensemble
        assert config.ensemble.site_spacing_m is not None  # enforced by _parse_ensemble
        # WP22 Part 2: n_sites copies of the `lattice` regime's single-site
        # Hermite-Gauss quadrature, distributed along ensemble.site_axis with
        # a Gaussian-or-uniform occupation envelope -- every site's own
        # position then feeds every pivot term already resolved above (the
        # local field via `field_fn`, uniform BBR via `bbr_value`, and the
        # gravitational redshift via `gravity`, all through the SAME `rate_fn`/
        # `_stark_rotor_ensemble` call sites the `lattice` branch above uses --
        # no new evaluation-mode machinery, per the WP's "trajectory/rotor
        # modes apply unchanged" design note).
        geometry = extended_lattice_nodes(
            species,
            trap,
            config.ensemble.motional_n,
            config.ensemble.n_quad,
            config.ensemble.n_sites,
            config.ensemble.site_spacing_m,
            config.ensemble.site_axis,
            config.ensemble.site_envelope,
            config.ensemble.site_envelope_sigma_m,
        )
        nodes = geometry.nodes
        node_weights = geometry.weights
        trajectories = nodes
        weights = node_weights

        if mode == "fast_path":
            # Tier A: E29 exact quadrature-expectation fast path -- O(1) cost in T,
            # exactly as the `lattice` regime's own fast_path branch (every
            # extended-lattice node is equally static, v=0).
            t_interrogation_s = _resolve_time_s(config.integration)
            ensemble_type = "lattice_extended_fast_path"
            ensemble_result = fastpath.lattice_shift_expectation(
                rate_fn, nodes, node_weights, t_interrogation_s
            )
            mode_note = (
                f"integration.mode=fast_path T={t_interrogation_s!r}s (E29, exact; "
                f"ensemble.regime=lattice_extended, WP22, n_sites={config.ensemble.n_sites}, "
                f"site_spacing_m={config.ensemble.site_spacing_m!r}, "
                f"site_envelope={config.ensemble.site_envelope!r})"
            )
        elif mode == "worldline" and config.coupling.type == "stark_dc":
            # WP16-style true Cl(1,3) rotor path, same as the `lattice` regime's
            # own stark_dc worldline branch, just against the extended node set.
            assert stark_coeffs is not None  # stark_dc coupling: resolved above
            dtau, n_steps, dtau_was_auto, tighten_note = _resolve_dtau_steps_worldline(
                config.integration, trap, rate_fn, nodes
            )
            dt_seconds = dtau * TAU_COMPTON
            traj_dense = jnp.broadcast_to(nodes[:, None, :], (nodes.shape[0], n_steps + 1, 3))
            ensemble_type = "lattice_extended_worldline_crosscheck"
            renorm_every, renorm_note = _resolve_renorm_every(config.integration, dtau_was_auto)
            ensemble_result = _stark_rotor_ensemble(
                field_fn,
                stark_coeffs,
                traj_dense,
                dtau,
                renorm_every=renorm_every,
                bbr_pivot_perturbation=bbr_value,
                quadrupole=config.quadrupole,
                gravity=gravity,
            )
            t_interrogation_s = n_steps * dt_seconds
            mode_note = (
                f"integration.mode=worldline dtau={dtau!r} steps={n_steps} "
                f"renorm_every={renorm_every} dtau_auto_selected={dtau_was_auto} "
                "(E17-E24 Cl(1,3) rotor, coupling.type=stark_dc: true rotor path; "
                f"ensemble.regime=lattice_extended, WP22, n_sites={config.ensemble.n_sites})"
            )
            if tighten_note is not None:
                mode_note = f"{mode_note}; {tighten_note}"
            if renorm_note is not None:
                mode_note = f"{mode_note}; {renorm_note}"
        else:
            assert mode == "worldline"
            # Explicit linear_mu cross-check: the rotor integrator (E17-E24) on
            # the same static extended-lattice nodes -- must agree with
            # fast_path exactly (E29), same as the `lattice` regime's own
            # linear_mu worldline branch.
            dtau, n_steps, dtau_was_auto, tighten_note = _resolve_dtau_steps_worldline(
                config.integration, trap, rate_fn, nodes
            )
            dt_seconds = dtau * TAU_COMPTON
            ensemble_type = "lattice_extended_worldline_crosscheck"
            renorm_every, renorm_note = _resolve_renorm_every(config.integration, dtau_was_auto)
            assert mu is not None  # linear_mu coupling: enforced by _parse_coupling
            ensemble_result = integrate_ensemble(
                field_fn,
                trajectories,
                dtau,
                mu,
                renorm_every=renorm_every,
                n_steps=n_steps,
            )
            t_interrogation_s = n_steps * dt_seconds
            mode_note = (
                f"integration.mode=worldline dtau={dtau!r} steps={n_steps} "
                f"renorm_every={renorm_every} (E17-E24; ensemble.regime=lattice_extended, WP22)"
            )
            if tighten_note is not None:
                mode_note = f"{mode_note}; {tighten_note}"
            if renorm_note is not None:
                mode_note = f"{mode_note}; {renorm_note}"

    site_map: LatticeExtendedSiteMap | None = None
    if config.ensemble.regime == "lattice_extended":
        assert geometry is not None  # set in the lattice_extended branch above
        site_map = _build_site_map(config.ensemble, geometry, ensemble_result)

    _validate_physics(ensemble_result)

    config_hash = _config_hash(config)
    # WP21 (G8 sign-off gate edits 5/6): the micromotion boundary and
    # hyperfine-E2 budget-line notes are test-pinned "on every ion-species
    # report" -- independent of coupling.type (an ion species run under
    # coupling.type='linear_mu' still carries them; the shared-cause stray
    # field this note describes is not specific to the E14b Stark path).
    ion_notes = " ".join(
        note
        for note in (
            ION_MICROMOTION_NOTES.get(species.name),
            ION_HYPERFINE_E2_BUDGET_NOTES.get(species.name),
        )
        if note is not None
    )
    if coupling_note is None:
        # coupling.type='linear_mu': byte-identical to behavior before the
        # stark_dc coupling was added (full backward compatibility), except
        # for the ion-species notes above (empty string for every non-ion
        # species already registered before WP21, so still byte-identical
        # for them).
        combined_notes = (
            f"{config.uncertainty_notes} {mode_note} {ion_notes}".strip()
            if ion_notes
            else (
                f"{config.uncertainty_notes} {mode_note}".strip()
                if config.uncertainty_notes
                else mode_note
            )
        )
    else:
        extra_notes = coupling_note
        if mode == "fast_path":
            extra_notes = f"{extra_notes} {_FAST_PATH_DOPPLER_EXCLUSION_NOTE}"
        if bbr_note is not None:
            extra_notes = f"{extra_notes} {bbr_note}"
        if quadrupole_note is not None:
            extra_notes = f"{extra_notes} {quadrupole_note}"
        if gravity_note is not None:
            assert gravity is not None  # gravity_note is only set when gravity is configured
            extra_notes = f"{extra_notes} {gravity_note}"
            extent_warn_note = _gravity_extent_warn_note(gravity, trajectories)
            if extent_warn_note is not None:
                extra_notes = f"{extra_notes} {extent_warn_note}"
        if config.ensemble.regime == "lattice_extended":
            extra_notes = f"{extra_notes} {LATTICE_EXTENDED_DISPERSION_LABEL_NOTE}"
        if ion_notes:
            extra_notes = f"{extra_notes} {ion_notes}"
        combined_notes = (
            f"{config.uncertainty_notes} {mode_note} {extra_notes}".strip()
            if config.uncertainty_notes
            else f"{mode_note} {extra_notes}".strip()
        )

    report = _build_report(
        ensemble_result.phase,
        species,
        t_interrogation_s,
        ensemble_type,
        weights,
        config_hash,
        combined_notes,
    )

    freqs_hz, amplitude = _line_profile_arrays(
        ensemble_result.phase, weights, t_interrogation_s, config.output.n_time_samples
    )

    return PipelineResult(
        report=report,
        ensemble_result=ensemble_result,
        trajectories=trajectories,
        weights=weights,
        line_profile_freqs_hz=freqs_hz,
        line_profile_amplitude=amplitude,
        site_map=site_map,
    )


def run_pipeline(config: PipelineConfig) -> MetrologyReport:
    """The one-call path: config -> report.

    load/synthesize field -> fit smoother -> build ensemble -> integrate
    -> analyze -> report. Thin wrapper around :func:`run_pipeline_full`
    for callers that only need the report; see that function's docstring
    for the full composition and :class:`PipelineResult` for the richer
    output (ensemble diagnostics, line-profile arrays) the CLI writes to
    disk.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    MetrologyReport
    """
    return run_pipeline_full(config).report
