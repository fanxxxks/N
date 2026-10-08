"""p54 / WP16 16-C (§4): the percentile-through-combo identity, pinned.

Assertion source: ``reports/wp16/FROZEN_C.md`` §4 (frozen 2026-09-23).

FROZEN_C §4.1 fixes the fact: ``combo_signal`` re-ranks every member
internally (``rank_normalize_signal`` at r01_baseline.py:203-209), and a
per-column strictly monotone member transform ``T`` satisfies
``rank(T(s)) == rank(s)`` — so the percentile scale of
``percentile_normalize_signal`` is erased *by construction*, not found
empirically.  §4.2.2 requires a small test that pins this identity so
nobody can later "fix" it silently; this file is that test.  It is green
on arrival by design (it pins the frozen fact, like p52's oracle anchors),
and it must stay green.

§4.2.1 requires the re-run measurement panel to present the identity as
STRUCTURAL with the frozen reason string; ``test_wp16c_measurement_output``
checks the panel artifacts (skipped until ``reports/wp16c/`` exists).

Fences (§4.3): ``r01_baseline.py`` is not modified here, the combo
definition is untouched, ``R01_COMBO_ARMS`` untouched; the no-re-rank
aggregate exists only inside the measurement script as a contrast arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ashare_model.r01_baseline import (
    combo_signal,
    percentile_normalize_signal,
)

_WP16C_DIR = Path(__file__).resolve().parents[1] / "reports" / "wp16c"

#: FROZEN_C §4.2.1 — the reason string the measurement output must carry
#: (verbatim Chinese clause; matched as a substring).
_STRUCTURAL_REASON_MARK = "消去任何列内严格单调尺度"


def _market(seed: int = 7, n_stocks: int = 9, n_dates: int = 16):
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 1.0, size=(n_stocks, n_dates))
    signal[2, 3] = np.nan
    signal[5, 10] = np.nan
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    mask[1, 2] = False  # PIT-masked cells stay excluded everywhere
    mask[4, 9] = False
    return signal, mask


def _affine(signal: np.ndarray, mask: np.ndarray, slope: float, offset: float):
    """An arbitrary strictly increasing affine member transform (NaN and
    masked cells preserved)."""
    out = np.asarray(signal, dtype=np.float64) * slope + offset
    return np.where(np.isfinite(signal), out, np.nan)


# --- §4.2.2: single member -----------------------------------------------------


@pytest.mark.parametrize(
    "name,transform",
    [
        ("percentile", percentile_normalize_signal),
        ("affine_3.7x+2.1", lambda s, m: _affine(s, m, 3.7, 2.1)),
        ("affine_0.5x-1", lambda s, m: _affine(s, m, 0.5, -1.0)),
    ],
)
def test_single_member_strictly_monotone_transform_leaves_combo_unchanged(
    name, transform
):
    signal, mask = _market()
    base = combo_signal([(1.0, signal)], mask)
    transformed = combo_signal([(1.0, transform(signal, mask))], mask)
    # §4.2.2: tolerance 0 (exact), per column, NaN cells equal.
    np.testing.assert_array_equal(base, transformed)


# --- §4.2.2: multi member ------------------------------------------------------


def test_multi_member_independent_monotone_transforms_leave_combo_unchanged():
    signal_a, mask = _market(seed=7)
    signal_b, _ = _market(seed=8)
    signal_c, _ = _market(seed=9)
    members = [(1.0, signal_a), (-1.0, signal_b), (1.0, signal_c)]
    transformed = [
        (1.0, percentile_normalize_signal(signal_a, mask)),
        (-1.0, _affine(signal_b, mask, 2.5, 0.75)),
        (1.0, percentile_normalize_signal(_affine(signal_c, mask, 0.3, -2.0), mask)),
    ]
    base = combo_signal(members, mask)
    other = combo_signal(transformed, mask)
    np.testing.assert_array_equal(base, other)
    # A real combo: some finite cells exist (the comparison is not vacuous).
    assert np.isfinite(base).any()


def test_percentile_is_strictly_monotone_per_column():
    """The transform family the identity covers is genuinely strictly
    monotone per column (ties preserved), on the same fixture."""
    signal, mask = _market()
    pct = percentile_normalize_signal(signal, mask)
    for j in range(signal.shape[1]):
        a = signal[:, j]
        b = pct[:, j]
        sel = np.isfinite(a) & np.isfinite(b) & mask[:, j]
        if sel.sum() < 2:
            continue
        order_a = np.argsort(a[sel], kind="stable")
        order_b = np.argsort(b[sel], kind="stable")
        assert np.array_equal(order_a, order_b)


# --- §4.2.1: the measurement panel presents the identity as structural --------


def test_wp16c_measurement_output_presents_structural_identity():
    """Gated on the panel artifacts (like p53's rerun-output test): the
    JSON carries the frozen reason clause, per-fold raw-vs-pct max
    abs difference 0, and equal raw/pct engine totals — all labelled as
    by-construction, plus the no-re-rank contrast arm."""
    path = _WP16C_DIR / "combo_pct_measurement.json"
    if not path.exists():
        pytest.skip("wp16c percentile panel not yet produced")
    report = json.loads(path.read_text(encoding="utf-8"))
    reason = report.get("structural_identity_reason", "")
    assert _STRUCTURAL_REASON_MARK in reason
    assert "combo_signal" in reason
    folds = report.get("folds", [])
    assert folds, "panel must report at least one fold"
    for fold in folds:
        combo = fold["combo_raw_vs_pct"]
        assert combo["max_abs_diff"] == 0.0
        assert combo["identity"] == "structural"
        assert combo["engine_total_raw"] == pytest.approx(combo["engine_total_pct"])
        # The honest contrast arm exists (scale NOT erased there).
        assert "contrast_no_rerank" in fold
