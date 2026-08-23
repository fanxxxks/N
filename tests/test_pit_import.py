from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_data.pit_import import (
    bs_code_to_ts_code,
    merge_list_dates,
    month_end_samples,
    timeline_to_intervals,
)


def test_merge_list_dates_prefers_fetched_and_preserves_others():
    stocks = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH", "600999.SH"],
            "name": ["A", "B", "C"],
            "industry": [None, "bank", None],
            "list_date": ["20100101", None, "20150101"],
            "is_st": [False, True, False],
        }
    )
    fetched = pd.DataFrame(
        {
            # Overlapping: fetched wins.  Missing fetched value: preserved.
            "ts_code": ["000001.SZ", "600000.SH", "601888.SH"],
            "list_date": ["20090101", "20200101", "20200201"],
        }
    )
    merged = merge_list_dates(stocks, fetched)
    assert merged["ts_code"].tolist() == [
        "000001.SZ",
        "600000.SH",
        "600999.SH",
        "601888.SH",
    ]
    by_code = merged.set_index("ts_code")
    assert by_code.loc["000001.SZ", "list_date"] == "20090101"  # fetched wins
    assert by_code.loc["600000.SH", "list_date"] == "20200101"  # null filled
    assert by_code.loc["600999.SH", "list_date"] == "20150101"  # untouched
    # The appended delisted member gets neutral display defaults.
    assert by_code.loc["601888.SH", "name"] == "601888.SH"
    assert by_code.loc["601888.SH", "list_date"] == "20200201"
    assert bool(by_code.loc["601888.SH", "is_st"]) is False
    # Overlapping rows keep their display fields.
    assert by_code.loc["000001.SZ", "name"] == "A"


def test_merge_list_dates_with_nan_fetched_never_overwrites():
    stocks = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "name": ["A"], "industry": [None],
         "list_date": ["20100101"], "is_st": [False]}
    )
    fetched = pd.DataFrame({"ts_code": ["000001.SZ"], "list_date": [np.nan]})
    merged = merge_list_dates(stocks, fetched)
    assert merged.iloc[0]["list_date"] == "20100101"


def test_merge_list_dates_on_empty_stock_table():
    stocks = pd.DataFrame(columns=["ts_code", "name", "industry", "list_date", "is_st"])
    fetched = pd.DataFrame({"ts_code": ["600999.SH"], "list_date": ["20080101"]})
    merged = merge_list_dates(stocks, fetched)
    assert len(merged) == 1
    assert merged.iloc[0]["list_date"] == "20080101"
    assert merged.iloc[0]["name"] == "600999.SH"


def test_bs_code_to_ts_code():
    assert bs_code_to_ts_code("sh.600000") == "600000.SH"
    assert bs_code_to_ts_code("sz.000001") == "000001.SZ"
    with pytest.raises(ValueError):
        bs_code_to_ts_code("bj.430047")


def test_month_end_samples_covers_pre_year_and_final_session():
    sessions = ["20240201", "20240215", "20240301", "20240329"]
    samples = month_end_samples(sessions)
    # Every session's month is represented by its latest session.
    assert "20240215" in samples
    assert "20240329" in samples
    # The 12 pre-year pseudo-months precede the first session.
    assert samples[0] == "20230128"
    assert "20231228" in samples
    # The final session is always the last sample.
    assert samples[-1] == "20240329"
    assert len(samples) == len(set(samples))


def test_timeline_to_intervals_compresses_runs_and_reentries():
    dates = ["20240101", "20240201", "20240301", "20240401", "20240501"]
    present = [True, True, False, True, True]
    assert timeline_to_intervals(dates, present) == [
        ("20240101", "20240301"),
        ("20240401", "99991231"),
    ]
    # All-present collapses to one open interval; all-absent to none.
    assert timeline_to_intervals(dates, [True] * 5) == [("20240101", "99991231")]
    assert timeline_to_intervals(dates, [False] * 5) == []
