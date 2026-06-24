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
    ts_decay_exp,
    ts_rank,
    winsorize,
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


def test_winsorize_clips_outliers():
    # Use a flat series (no MultiIndex) so winsorize operates on all values
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0])
    result = winsorize(s, std=1.0, level=None)
    # The outlier (1000) should be clipped — mean+1*std will be < 1000
    assert result.iloc[-1] < 1000.0
    # Non-outlier values should be unchanged
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 2.0
    # The clipped value should equal mean + 1*std
    mu = s.mean()
    sigma = s.std()
    expected_clip = mu + 1.0 * sigma
    assert abs(result.iloc[-1] - expected_clip) < 0.01


def test_winsorize_no_clip_without_outliers():
    s = _panel_series()
    result = winsorize(s, std=4.0)
    # With no outliers, values should be unchanged
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 4.0


def test_ts_decay_exp_smoothes_series():
    s = _panel_series()
    result = ts_decay_exp(s, window=3, factor=0.5)
    # The result should have the same index
    assert result.index.equals(s.index)
    # Exponential decay should smooth: last value should be between raw and mean
    last_a = result.xs("a", level="symbol").iloc[-1]
    raw_last_a = s.xs("a", level="symbol").iloc[-1]
    # With factor=0.5 (alpha=0.5), the EWM is heavily weighted to recent
    assert last_a > 0
    # EWM should not exceed the max of the series
    assert last_a <= raw_last_a + 0.01

