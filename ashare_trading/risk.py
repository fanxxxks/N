"""Paper-trading filters."""

from __future__ import annotations

import numpy as np


def limit_rate(ts_code: str, name: str = "") -> float:
    if "ST" in name.upper():
        return 0.05
    prefix = ts_code.split(".")[0][:3]
    if prefix in {"300", "301", "688", "689"}:
        return 0.20
    return 0.10


def is_one_word_limit_up(
    open_: float,
    high: float,
    low: float,
    pre_close: float,
    ts_code: str,
    name: str = "",
) -> bool:
    if pre_close <= 0:
        return False
    rate = limit_rate(ts_code, name)
    change = open_ / pre_close - 1.0
    return (
        np.isclose(open_, high)
        and np.isclose(open_, low)
        and change >= rate - 0.005
    )


def is_one_word_limit_down(
    open_: float,
    high: float,
    low: float,
    pre_close: float,
    ts_code: str,
    name: str = "",
) -> bool:
    if pre_close <= 0:
        return False
    rate = limit_rate(ts_code, name)
    change = open_ / pre_close - 1.0
    return (
        np.isclose(open_, high)
        and np.isclose(open_, low)
        and change <= -rate + 0.005
    )


def is_suspended(open_: float, volume: float) -> bool:
    return open_ <= 0 or volume <= 0
