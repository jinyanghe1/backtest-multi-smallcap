import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from tools.backtest_mvp.engine import (
    CrossSectionalEngine,
    compute_monthly_ic_heatmap,
    _compute_max_drawdown_recovery_time,
    _compute_rolling_sharpe,
)


def _panels_with_predictive_signal():
    dates = pd.bdate_range("2024-01-01", periods=120)
    symbols = ["a", "b", "c", "d", "e"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    signal_vals = [5.0, 4.0, 3.0, 2.0, 1.0] * len(dates)
    factor_panel = pd.DataFrame({"alpha_signal": signal_vals}, index=idx)
    returns = []
    for _ in dates:
        returns.extend([0.003, 0.002, 0.001, -0.001, -0.002])
    return_panel = pd.DataFrame({"daily_return": returns}, index=idx)
    return factor_panel, return_panel


# ── T01: monthly_ic_heatmap ──

def test_monthly_ic_heatmap_from_ic_series():
    """Standalone function should pivot IC series into year × month."""
    dates = pd.date_range("2024-01-31", periods=6, freq="ME")
    ic = pd.Series([0.1, 0.2, 0.05, -0.1, 0.15, 0.08], index=dates, name="ic")
    heatmap = compute_monthly_ic_heatmap(ic)
    assert heatmap.index.name == "year"
    assert heatmap.columns.name == "month"
    assert 2024 in heatmap.index
    assert heatmap.loc[2024, 1] == 0.1
    assert heatmap.loc[2024, 2] == 0.2
    # Columns should be 1..12
    assert list(heatmap.columns) == list(range(1, 13))


def test_monthly_ic_heatmap_empty():
    """Empty IC series should return empty DataFrame."""
    assert compute_monthly_ic_heatmap(pd.Series(dtype=float)).empty
    assert compute_monthly_ic_heatmap(None).empty


def test_monthly_ic_heatmap_multi_year():
    """Heatmap should handle multiple years."""
    dates = pd.to_datetime(["2023-06-15", "2023-12-20", "2024-03-10", "2024-09-15"])
    ic = pd.Series([0.1, 0.2, 0.15, 0.05], index=dates)
    heatmap = compute_monthly_ic_heatmap(ic)
    assert 2023 in heatmap.index
    assert 2024 in heatmap.index
    assert heatmap.loc[2023, 6] == 0.1
    assert heatmap.loc[2024, 3] == 0.15


def test_monthly_ic_heatmap_in_backtest_result():
    """BacktestResult should contain monthly_ic_heatmap after run()."""
    fp, rp = _panels_with_predictive_signal()
    engine = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp,
        n_stocks=2, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    result = engine.run(ranking_factor="alpha_signal", ascending=False)
    assert result.monthly_ic_heatmap is not None
    assert not result.monthly_ic_heatmap.empty
    assert 2024 in result.monthly_ic_heatmap.index


# ── T04: max_drawdown_recovery_time ──

def test_max_drawdown_recovery_time_basic():
    """Recovery time from trough to new high."""
    # equity goes 1.0 → 0.8 (trough) → 1.1 (new high)
    dates = pd.date_range("2024-01-01", periods=5)
    eq = pd.Series([1.0, 0.9, 0.8, 0.95, 1.1], index=dates)
    recovery = _compute_max_drawdown_recovery_time(eq)
    # Trough is at index 2 (0.8), new high at index 4 (1.1)
    # Days between 2024-01-03 and 2024-01-05 = 2
    assert recovery == 2


def test_max_drawdown_recovery_time_no_recovery():
    """If curve never recovers, return 0."""
    dates = pd.date_range("2024-01-01", periods=4)
    eq = pd.Series([1.0, 0.9, 0.8, 0.85], index=dates)
    recovery = _compute_max_drawdown_recovery_time(eq)
    assert recovery == 0


def test_max_drawdown_recovery_time_in_result():
    """BacktestResult should contain max_drawdown_recovery_time."""
    fp, rp = _panels_with_predictive_signal()
    engine = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp,
        n_stocks=2, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    result = engine.run(ranking_factor="alpha_signal", ascending=False)
    assert hasattr(result, "max_drawdown_recovery_time")
    assert isinstance(result.max_drawdown_recovery_time, int)


# ── T12: rolling_sharpe ──

def test_rolling_sharpe_basic():
    """Rolling Sharpe should produce a Series of floats."""
    monthly = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0, 0.02,
                         0.01, -0.02, 0.03, 0.01, 0.02, 0.0,
                         0.01, 0.02])
    rs = _compute_rolling_sharpe(monthly, window=6)
    assert isinstance(rs, pd.Series)
    assert len(rs) > 0
    assert rs.notna().all()


def test_rolling_sharpe_empty():
    """Empty or too-short returns should return empty Series."""
    assert _compute_rolling_sharpe(pd.Series(dtype=float)).empty
    assert _compute_rolling_sharpe(pd.Series([0.01])).empty


def test_rolling_sharpe_in_result():
    """BacktestResult should contain rolling_sharpe."""
    fp, rp = _panels_with_predictive_signal()
    engine = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp,
        n_stocks=2, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    result = engine.run(ranking_factor="alpha_signal", ascending=False)
    assert hasattr(result, "rolling_sharpe")
    assert result.rolling_sharpe is not None


# ── T07: turnover_attribution ──

def test_turnover_attribution_in_result():
    """BacktestResult should contain turnover_attribution dict."""
    fp, rp = _panels_with_predictive_signal()
    engine = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp,
        n_stocks=2, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    result = engine.run(ranking_factor="alpha_signal", ascending=False)
    assert hasattr(result, "turnover_attribution")
    attr = result.turnover_attribution
    assert isinstance(attr, dict)
    assert "rebalance_turnover" in attr
    assert "price_drift_turnover" in attr
    assert "total_turnover" in attr
    assert attr["total_turnover"] == attr["rebalance_turnover"] + attr["price_drift_turnover"]
    assert attr["rebalance_turnover"] >= 0
    assert attr["price_drift_turnover"] >= 0
