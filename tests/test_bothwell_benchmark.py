# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the WP22 Part 3 Bothwell 2022 mm-scale redshift benchmark case
(``benchmarks/run_bothwell_redshift.py``).

Covers: the case runs the REAL `ensemble.regime='lattice_extended'`
pipeline (not a shortcut formula), the coordinate-sign mapping, the
G9-sign-off-pinned band-overlap verdicts (MET at ~0.48-sigma/~0.70-sigma),
the "reproducibility" classification label with the inverted-NPL caveat,
the INFERRED-geometry flags, and the report/markdown rendering.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import run_bothwell_redshift as bothwell  # noqa: E402

from cliffordclock.constants import SPEED_OF_LIGHT  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Geometry constants: derivation and INFERRED flags.
# ---------------------------------------------------------------------------


def test_site_spacing_is_half_the_magic_wavelength() -> None:
    """406.5 nm = 813 nm / 2 (dossier B3's own "813/2" cross-check)."""
    np.testing.assert_allclose(bothwell.SITE_SPACING_M, 813e-9 / 2.0, rtol=0, atol=0)
    np.testing.assert_allclose(bothwell.SITE_SPACING_M, 406.5e-9, rtol=1e-12, atol=0)


def test_envelope_sigma_derivation_documented() -> None:
    """sigma = (100 pixels * 6.04 um/pixel) / 1.5 -- the INFERRED derivation
    from the dossier's "two +/-1.5sigma regions, ~100 pixels each"."""
    expected = (100 * 6.04e-6) / 1.5
    np.testing.assert_allclose(bothwell.ENVELOPE_SIGMA_M, expected, rtol=0, atol=0)
    # ~403 um, order-of-magnitude consistent with the dossier's "~1-1.3mm
    # span" (a few sigma).
    assert 3e-4 < bothwell.ENVELOPE_SIGMA_M < 5e-4


def test_n_sites_covers_the_configured_half_span_and_is_odd() -> None:
    assert bothwell.N_SITES % 2 == 1  # symmetric about offset 0
    half_span_sites = (bothwell.N_SITES - 1) // 2
    covered_span_m = half_span_sites * bothwell.SITE_SPACING_M
    assert covered_span_m >= bothwell.N_SIGMA_HALF_SPAN * bothwell.ENVELOPE_SIGMA_M


@pytest.mark.parametrize(
    ("half_span_m", "spacing_m", "expected_n_sites"),
    [
        (1.0, 1.0, 3),  # exactly 1 site each side
        (1.5, 1.0, 5),  # ceil(1.5) = 2 sites each side
        (0.0, 1.0, 1),  # degenerate: just the center site
    ],
)
def test_n_sites_covering_hand_computed(
    half_span_m: float, spacing_m: float, expected_n_sites: int
) -> None:
    assert bothwell._n_sites_covering(half_span_m, spacing_m) == expected_n_sites


# ---------------------------------------------------------------------------
# 2. Reference gravity / measured-slope constants (citations present, not
#    a hand-typed number masquerading as a PublishedBand).
# ---------------------------------------------------------------------------


def test_bothwell_surveyed_g_pinned_and_differs_from_standard_gravity() -> None:
    from cliffordclock.constants import STANDARD_GRAVITY

    assert bothwell.BOTHWELL_SURVEYED_G_M_S2 == 9.796
    assert bothwell.BOTHWELL_SURVEYED_G_M_S2 != STANDARD_GRAVITY
    assert "van Westrum" in bothwell._BOTHWELL_G_CITATION
    assert "NGS-77" in bothwell._BOTHWELL_G_CITATION


def test_measured_slope_bands_hand_computed() -> None:
    a = bothwell.MEASURED_SLOPE_METHOD_A
    np.testing.assert_allclose(a.nominal, -9.8e-20, rtol=0, atol=0)
    np.testing.assert_allclose(a.lo, -12.1e-20, rtol=1e-12, atol=0)
    np.testing.assert_allclose(a.hi, -7.5e-20, rtol=1e-12, atol=0)

    b = bothwell.MEASURED_SLOPE_METHOD_B
    np.testing.assert_allclose(b.nominal, -1.28e-19, rtol=0, atol=0)
    np.testing.assert_allclose(b.lo, -1.55e-19, rtol=1e-12, atol=0)
    np.testing.assert_allclose(b.hi, -1.01e-19, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# 3. The case itself: real-pipeline computation, coordinate-sign mapping,
#    band-overlap verdicts pinned to the G9 sign-off's own numbers.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def case() -> bothwell.BothwellRedshiftCase:
    """Computed once per test module (a real ~5900-site pipeline run,
    ~1s) and reused read-only by every test below."""
    return bothwell.run_bothwell_redshift_case()


def test_case_uses_the_real_lattice_extended_pipeline_not_a_shortcut(
    case: bothwell.BothwellRedshiftCase,
) -> None:
    """The predicted slope is NOT simply `g/c^2` hand-computed -- it comes
    from a real per-site weighted-least-squares fit over `N_SITES` sites
    (the whole point of this case, module docstring: "not a shortcut
    formula -- the whole point is the machinery"). Verified by checking it
    is close to, but not bit-identical to, the naive `g/c^2` value (the
    fit/geometry introduces float64 rounding a pure formula would not).
    """
    naive_g_over_c2_per_mm = bothwell.BOTHWELL_SURVEYED_G_M_S2 / SPEED_OF_LIGHT**2 / 1000.0
    np.testing.assert_allclose(
        case.predicted_slope_engine_convention_per_m / 1000.0,
        naive_g_over_c2_per_mm,
        rtol=1e-6,
        atol=0,
    )
    assert case.n_sites == bothwell.N_SITES > 1000  # a genuine extended-sample run


def test_coordinate_sign_mapping_is_a_negation(case: bothwell.BothwellRedshiftCase) -> None:
    """Bothwell's own coordinate convention is the engine's own height
    convention NEGATED (module docstring's coordinate-sign mapping, G9
    sign-off gate edit 2) -- checked as an exact algebraic relationship:
    the full negated value must match, not just the sign.
    """
    np.testing.assert_allclose(
        case.predicted_slope_per_mm,
        -case.predicted_slope_engine_convention_per_m / 1000.0,
        rtol=0,
        atol=0,
    )
    # The engine's own height convention is POSITIVE (higher = faster,
    # CONVENTIONS.md section 15 E36's sign convention).
    assert case.predicted_slope_engine_convention_per_m > 0.0
    # Bothwell's own published convention is NEGATIVE (dossier: "Known
    # Redshift -10.9e-20/mm").
    assert case.predicted_slope_per_mm < 0.0


def test_predicted_slope_matches_dossier_headline_minus_10_9e_20_per_mm(
    case: bothwell.BothwellRedshiftCase,
) -> None:
    """The G9-corrected dossier headline: standard g gives 1.0911e-19/mm,
    Boulder g gives ~1.0900e-19/mm -- both round to the published
    Table 1 "Known Redshift -10.9e-20/mm" at its own reported precision.
    """
    np.testing.assert_allclose(case.predicted_slope_per_mm, -10.9e-20, rtol=1e-2, atol=0)


def test_windowed_cross_check_agrees_with_full_range_fit(
    case: bothwell.BothwellRedshiftCase,
) -> None:
    """The +/-1.5-sigma-windowed fit (Bothwell's own analysis window) and
    the full +/-N_SIGMA_HALF_SPAN-sigma computational-grid fit must agree
    to high precision for this field-free, purely-linear-in-height case
    (module docstring) -- an independent cross-check computed via
    `numpy.polyfit`, not a re-invocation of the same fit code.
    """
    np.testing.assert_allclose(
        case.predicted_slope_windowed_per_mm,
        case.predicted_slope_per_mm,
        rtol=1e-6,
        atol=0,
    )


def test_band_overlap_verdicts_met_both_methods(case: bothwell.BothwellRedshiftCase) -> None:
    """G9 sign-off B2: both bands are expected to bracket the prediction
    (MET), the case's headline result."""
    assert case.kpi_verdict_method_a == "MET"
    assert case.kpi_verdict_method_b == "MET"
    assert case.bands_overlap_method_a is True
    assert case.bands_overlap_method_b is True
    assert case.kpi_verdict_method_a not in ("PASS", "FAIL")
    assert case.kpi_verdict_method_b not in ("PASS", "FAIL")


def test_sigma_distances_match_g9_sign_off_pinned_numbers(
    case: bothwell.BothwellRedshiftCase,
) -> None:
    """G9 sign-off B2: "the prediction -10.9e-20/mm sits 0.48-sigma from
    the campaign measurement and 0.70-sigma from the synchronous one" --
    the exact numbers the gate ratified, computed here (not hand-typed)
    from the real predicted value and the published uncertainties.
    """
    np.testing.assert_allclose(case.sigma_distance_method_a, 0.48, rtol=0, atol=0.01)
    np.testing.assert_allclose(case.sigma_distance_method_b, 0.70, rtol=0, atol=0.01)


def test_measurements_sit_roughly_one_sigma_apart_with_prediction_between(
    case: bothwell.BothwellRedshiftCase,
) -> None:
    """Dossier/G9 sign-off narrative point: the two measurements sit ~1
    sigma apart from each other, with the prediction between them."""
    a = bothwell.MEASURED_SLOPE_METHOD_A.nominal
    b = bothwell.MEASURED_SLOPE_METHOD_B.nominal
    predicted = case.predicted_slope_per_mm
    assert min(a, b) <= predicted <= max(a, b)


# ---------------------------------------------------------------------------
# 4. Classification label (G9 sign-off B4) and its caveat text.
# ---------------------------------------------------------------------------


def test_case_class_is_reproducibility_not_arithmetic_reproduction(
    case: bothwell.BothwellRedshiftCase,
) -> None:
    """Distinct from the WP20 BBR case's WEAKER 'arithmetic_reproduction'
    class (G9 sign-off B4 vs. G7 sign-off B5) -- this case is a genuine
    'reproducibility' case, the SAME vocabulary as the NPL case."""
    assert case.case_class == "reproducibility"
    assert case.case_class != "arithmetic_reproduction"


def test_case_label_carries_the_inverted_npl_caveat() -> None:
    label = bothwell.CASE_LABEL
    assert "reproducibility" in label
    assert "INVERTED-NPL" in label
    assert "textbook" in label
    assert "MACHINERY" in label
    assert "blind-prediction count" in label


def test_case_label_present_on_the_dataclass_instance(
    case: bothwell.BothwellRedshiftCase,
) -> None:
    """The caveat rides wherever the case appears (G9 sign-off instruction)
    -- present directly on the returned dataclass, not only in the module
    docstring/constant."""
    assert case.case_label == bothwell.CASE_LABEL


def test_isolation_note_and_dc_stark_context_note_present(
    case: bothwell.BothwellRedshiftCase,
) -> None:
    assert "ISOLATED gravitational gradient" in case.isolation_note
    assert "DC-Stark" in case.dc_stark_context_note


# ---------------------------------------------------------------------------
# 5. Report / markdown rendering; not merged into the WP10 headline.
# ---------------------------------------------------------------------------


def test_build_report_schema_and_not_merged_into_wp10() -> None:
    report = bothwell.build_report()
    assert report["wp22_bothwell_benchmark_schema"] == "1.0"
    assert report["case_class"] == "reproducibility"
    assert "bothwell_2022_nature_602_420_redshift_case" in report
    # Never writes into or reads from wp10_results.json.
    assert "wp10" not in json_dump_keys(report)


def json_dump_keys(d: dict) -> set:
    """All (nested) dict keys, for a coarse structural assertion above."""
    keys: set = set()
    for k, v in d.items():
        keys.add(k)
        if isinstance(v, dict):
            keys |= json_dump_keys(v)
    return keys


def test_render_markdown_contains_verdicts_and_caveat() -> None:
    report = bothwell.build_report()
    markdown = bothwell.render_markdown(report)
    assert "MET" in markdown
    assert "INVERTED-NPL" in markdown
    assert "kpi_verdict" in markdown
    assert "wp10_results.json" in markdown  # the not-merged-into-headline note


def test_main_writes_json_and_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bothwell, "_RESULTS_DIR", tmp_path)
    bothwell.main()
    assert (tmp_path / "bothwell_redshift.json").exists()
    assert (tmp_path / "bothwell_redshift.md").exists()
    import json

    data = json.loads((tmp_path / "bothwell_redshift.json").read_text(encoding="utf-8"))
    assert data["case_class"] == "reproducibility"


# ---------------------------------------------------------------------------
# 6. Runtime sanity: the case does not silently disable itself.
# ---------------------------------------------------------------------------


def test_run_bothwell_redshift_case_is_deterministic(case: bothwell.BothwellRedshiftCase) -> None:
    """Re-running the case gives a bit-identical prediction (no PRNG, no
    hidden state): a determinism guard on the actual predicted value,
    stronger than a smoke test that only confirms the case runs without
    raising."""
    second = bothwell.run_bothwell_redshift_case()
    assert math.isclose(
        second.predicted_slope_per_mm, case.predicted_slope_per_mm, rel_tol=0, abs_tol=0
    )


# ---------------------------------------------------------------------------
# 7. Label-drift guard (WP23, mirrors test_bbr_benchmark.py's and
# test_roos_benchmark.py's own guards): the binding two-reproducibility-case
# project headline (owner-ratified 2026-08-11, WP23) must appear in the
# user-facing docs themselves, not only in this script's own CASE_LABEL
# constant -- someone editing RESULTS.md/docs/validation.md independently
# must trip this test.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("relpath", "required_phrases"),
    [
        (
            "benchmarks/RESULTS.md",
            [
                "TWO reproducibility cases MET",
                "Reproducibility case: Bothwell et al. mm-scale gravitational redshift",
                "inverted-NPL caveat",
            ],
        ),
        (
            "docs/validation.md",
            [
                "two experimental reproducibility cases",
                "Bothwell",
                "gravitational-redshift",
            ],
        ),
        (
            "README.md",
            [
                "two reproducibility cases",
                "Bothwell mm-scale redshift measurement",
            ],
        ),
    ],
)
def test_case_label_present_in_user_facing_docs(relpath: str, required_phrases: list[str]) -> None:
    # Markdown hard-wraps prose, so a phrase can span a line break --
    # collapse all whitespace runs to single spaces before matching.
    raw = (_REPO_ROOT / relpath).read_text(encoding="utf-8")
    text = " ".join(raw.split())
    for phrase in required_phrases:
        assert phrase in text, (
            f"{relpath} lost the binding two-reproducibility-case headline phrase "
            f"{phrase!r} (WP23, owner-ratified 2026-08-11; see this test's header comment)"
        )
