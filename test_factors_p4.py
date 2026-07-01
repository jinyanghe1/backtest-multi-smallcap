import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging

import numpy as np
import pandas as pd
import pytest

from tools.backtest_mvp.factors import factor_library as fl
from tools.backtest_mvp.factors.factor_library import (
    FACTOR_REGISTRY,
    beta_arbitrage,
    compute_all_factors,
    fifty_two_week_high_proximity,
    information_discreteness,
    momentum_quality,
    prospect_theory_value,
    seasonality_same_month,
    tail_return_spread,
    trend_strength,
    downside_beta,
    trailing_max_drawdown,
    analyst_revision,
)


NEW_FACTORS = [
    fifty_two_week_high_proximity,
    seasonality_same_month,
    downside_beta,
    information_discreteness,
    prospect_theory_value,
    trailing_max_drawdown,
]


def _panel_from_close(close_wide: pd.DataFrame) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [close_wide.index, close_wide.columns], names=["date", "symbol"]
    )
    close = close_wide.stack()
    close.index.names = ["date", "symbol"]
    panel = pd.DataFrame(index=idx)
    panel["close"] = close.reindex(idx).astype(float)
    panel["open"] = panel["close"]
    panel["high"] = panel["close"] * 1.01
    panel["low"] = panel["close"] * 0.99
    panel["volume"] = 1000.0
    panel["amount"] = panel["close"] * panel["volume"]
    panel["vwap"] = panel["close"]
    panel["mcap"] = panel["close"] * 1_000_000
    panel["pb"] = 1.5
    panel["turnover"] = 0.01
    panel["shareholders"] = 1000
    panel["industry_code"] = "I1"
    return panel


def _random_panel(days: int = 900, symbols: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2020-01-01", periods=days)
    names = [f"S{i:02d}" for i in range(symbols)]
    market = rng.normal(0.0002, 0.012, (days, 1))
    idio = rng.normal(0.0001, 0.018, (days, symbols))
    close = 100 * np.exp(np.cumsum(market + idio, axis=0))
    panel = _panel_from_close(pd.DataFrame(close, index=dates, columns=names))
    panel["high"] = panel["close"] * (1.005 + rng.random(len(panel)) * 0.01)
    panel["low"] = panel["close"] * (0.995 - rng.random(len(panel)) * 0.01)
    panel["open"] = panel["close"] * (1 + rng.normal(0, 0.002, len(panel)))
    panel["volume"] = rng.integers(10_000, 100_000, len(panel)).astype(float)
    panel["amount"] = panel["volume"] * panel["close"]
    panel["turnover"] = rng.uniform(0.001, 0.05, len(panel))
    panel["pb"] = rng.uniform(0.5, 5.0, len(panel))
    panel["industry_code"] = [f"I{i % 3}" for i in range(len(panel))]
    return panel


@pytest.mark.parametrize("factor", NEW_FACTORS)
def test_new_factors_return_aligned_series(factor):
    panel = _random_panel()
    result = factor(panel)
    assert isinstance(result, pd.Series)
    assert result.index.equals(panel.index)
    assert result.notna().any()


@pytest.mark.parametrize("factor", NEW_FACTORS)
def test_new_factors_handle_short_history_gracefully(factor):
    dates = pd.bdate_range("2024-01-01", periods=10)
    close = pd.DataFrame({"A": np.linspace(10, 11, len(dates)), "B": 10.0}, index=dates)
    panel = _panel_from_close(close)
    result = factor(panel)
    assert result.index.equals(panel.index)
    assert len(result) == len(panel)


def test_f031_near_rolling_high_scores_higher():
    dates = pd.bdate_range("2024-01-01", periods=80)
    close = pd.DataFrame(
        {"NEAR": np.linspace(10, 20, 80), "FAR": np.r_[np.linspace(10, 20, 40), np.linspace(19, 8, 40)]},
        index=dates,
    )
    result = fifty_two_week_high_proximity(_panel_from_close(close), window=30).xs(dates[-1], level="date")
    assert result["NEAR"] > result["FAR"]


def test_f032_uses_prior_same_month_history():
    dates = pd.to_datetime(
        ["2019-12-31", "2020-01-31", "2020-12-31", "2021-01-29", "2021-12-31", "2022-01-31"]
    )
    close = pd.DataFrame(
        {
            "JAN_WINNER": [100, 120, 100, 130, 100, 101],
            "JAN_LOSER": [100, 90, 100, 80, 100, 101],
        },
        index=dates,
    )
    result = seasonality_same_month(_panel_from_close(close)).xs(pd.Timestamp("2022-01-31"), level="date")
    assert result["JAN_WINNER"] > result["JAN_LOSER"]


def test_f033_downside_beta_penalizes_high_downside_exposure():
    dates = pd.bdate_range("2024-01-01", periods=140)
    base = np.tile([0.01, -0.012], 70)
    returns = pd.DataFrame(
        {
            "HIGH_DOWNSIDE": np.where(base < 0, 2.0 * base, 0.5 * base),
            "LOW_DOWNSIDE": np.where(base < 0, 0.2 * base, 0.5 * base),
            "MARKET_1": base,
            "MARKET_2": base * 0.9,
        },
        index=dates,
    )
    close = 100 * (1 + returns).cumprod()
    result = downside_beta(_panel_from_close(close), window=90).xs(dates[-1], level="date")
    assert result["LOW_DOWNSIDE"] > result["HIGH_DOWNSIDE"]


def test_f034_continuous_momentum_scores_above_jump_momentum():
    dates = pd.bdate_range("2024-01-01", periods=90)
    continuous_ret = np.full(90, 0.002)
    jump_ret = np.zeros(90)
    jump_ret[35] = (1 + 0.002) ** 60 - 1
    close = pd.DataFrame(
        {
            "CONTINUOUS": 100 * np.cumprod(1 + continuous_ret),
            "JUMP": 100 * np.cumprod(1 + jump_ret),
            "FLAT": 100.0,
        },
        index=dates,
    )
    result = information_discreteness(_panel_from_close(close), window=60).xs(dates[-1], level="date")
    assert result["CONTINUOUS"] > result["JUMP"]


def test_f035_high_prospect_theory_value_is_penalized():
    dates = pd.bdate_range("2024-01-01", periods=90)
    close = pd.DataFrame(
        {
            "HIGH_TK": 100 * np.cumprod(np.full(90, 1.003)),
            "LOW_TK": 100 * np.cumprod(np.full(90, 0.997)),
            "MIXED": 100 * np.cumprod(1 + np.tile([0.002, -0.002], 45)),
        },
        index=dates,
    )
    result = prospect_theory_value(_panel_from_close(close), window=60).xs(dates[-1], level="date")
    assert result["LOW_TK"] > result["HIGH_TK"]


def test_f036_deep_trough_scores_higher():
    dates = pd.bdate_range("2024-01-01", periods=90)
    deep = np.r_[np.linspace(100, 130, 30), np.linspace(130, 60, 30), np.linspace(60, 90, 30)]
    shallow = np.linspace(100, 110, 90)
    close = pd.DataFrame({"DEEP": deep, "SHALLOW": shallow}, index=dates)
    result = trailing_max_drawdown(_panel_from_close(close), window=60).xs(dates[-1], level="date")
    assert result["DEEP"] > result["SHALLOW"]


@pytest.mark.parametrize(
    "factor,kwargs",
    [
        (fifty_two_week_high_proximity, {"window": 30}),
        (information_discreteness, {"window": 60}),
        (trailing_max_drawdown, {"window": 60}),
    ],
)
def test_future_rows_do_not_change_earlier_factor_values(factor, kwargs):
    panel = _random_panel(days=120, symbols=5)
    baseline = factor(panel, **kwargs)
    cutoff = panel.index.get_level_values("date").unique()[80]
    corrupted = panel.copy()
    future = corrupted.index.get_level_values("date") > cutoff
    corrupted.loc[future, ["close", "high", "low", "open"]] *= 100
    changed = factor(corrupted, **kwargs)
    pd.testing.assert_series_equal(
        baseline.loc[(cutoff, slice(None))],
        changed.loc[(cutoff, slice(None))],
        check_names=False,
    )


def test_new_factors_have_low_correlation_with_nearest_existing_factors():
    panel = _random_panel()
    pairs = {
        "F031-F023": (fifty_two_week_high_proximity(panel), trend_strength(panel)),
        "F032-F030": (seasonality_same_month(panel), analyst_revision(panel)),
        "F033-F021": (downside_beta(panel), beta_arbitrage(panel)),
        "F034-F025": (information_discreteness(panel), momentum_quality(panel)),
        "F035-F026": (prospect_theory_value(panel), tail_return_spread(panel)),
        "F036-F026": (trailing_max_drawdown(panel), tail_return_spread(panel)),
    }
    correlations = {}
    for name, (new_factor, old_factor) in pairs.items():
        aligned = pd.concat([new_factor, old_factor], axis=1).dropna()
        correlations[name] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    assert correlations
    assert all(abs(corr) < 0.8 for corr in correlations.values())


def test_compute_all_factors_log_errors_true_warns(caplog, monkeypatch):
    def broken_factor(panel):
        raise RuntimeError("synthetic failure")

    monkeypatch.setitem(FACTOR_REGISTRY, "FBAD", broken_factor)
    caplog.set_level(logging.WARNING, logger=fl.__name__)
    compute_all_factors(_random_panel(days=80, symbols=4), log_errors=True)
    assert "FBAD" in caplog.text
    assert "synthetic failure" in caplog.text


def test_compute_all_factors_default_stays_silent(caplog, monkeypatch):
    def broken_factor(panel):
        raise RuntimeError("silent failure")

    monkeypatch.setitem(FACTOR_REGISTRY, "FBAD", broken_factor)
    caplog.set_level(logging.WARNING, logger=fl.__name__)
    compute_all_factors(_random_panel(days=80, symbols=4))
    assert "FBAD" not in caplog.text
    assert "silent failure" not in caplog.text
