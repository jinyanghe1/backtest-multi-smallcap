import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from tools.backtest_mvp.factors.operators import (
    correlation,
    decay_linear,
    delta,
    group_rank,
    rank,
    scale,
    signed_power,
    ts_argmax,
    ts_argmin,
    ts_rank,
)


def _panel_series():
    dates = pd.date_range("2024-01-01", periods=4)
    symbols = ["a", "b"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    return pd.Series([1, 4, 2, 3, 3, 2, 4, 1], index=idx, dtype=float)


def test_time_series_operators_group_by_symbol():
    s = _panel_series()
    ranked = ts_rank(s, 2)
    assert ranked.xs("a", level="symbol").iloc[-1] == 1.0
    assert ranked.xs("b", level="symbol").iloc[-1] == 0.5

    d = delta(s, 1)
    assert d.xs("a", level="symbol").iloc[-1] == 1.0
    assert d.xs("b", level="symbol").iloc[-1] == -1.0

    decayed = decay_linear(s, 2)
    assert decayed.xs("a", level="symbol").iloc[-1] == (3 * 1 + 4 * 2) / 3


def test_cross_sectional_rank_and_group_rank():
    s = _panel_series()
    ranked = rank(s)
    first_date = s.index.get_level_values("date")[0]
    assert ranked.loc[(first_date, "a")] == 0.5
    assert ranked.loc[(first_date, "b")] == 1.0

    group = pd.Series(["g", "g"] * 4, index=s.index)
    grouped = group_rank(s, group)
    assert grouped.loc[(first_date, "a")] == 0.5
    assert grouped.loc[(first_date, "b")] == 1.0


def test_other_operators():
    s = _panel_series()
    corr = correlation(s, s, 2)
    assert corr.dropna().iloc[-1] == 1.0
    assert scale(pd.Series([1.0, -1.0])).tolist() == [0.5, -0.5]
    assert ts_argmax(pd.Series([1.0, 3.0, 2.0]), 3).iloc[-1] == 1
    assert ts_argmin(pd.Series([1.0, 3.0, 2.0]), 3).iloc[-1] == 0
    assert signed_power(pd.Series([-2.0, 3.0]), 2).tolist() == [-4.0, 9.0]

