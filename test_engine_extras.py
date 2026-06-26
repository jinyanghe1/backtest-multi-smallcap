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


# ── FX01+FX02: daily stop_loss precision ──

def _make_crashing_panels(n_days: int = 60):
    """Create factor/return panels where stocks crash -2% every day.
    With 1 stock at n_stocks=1, this gives a predictable decline.
    """
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    symbols = ["crash"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    factor_panel = pd.DataFrame({"alpha": [1.0] * n_days}, index=idx)
    return_panel = pd.DataFrame({"daily_return": [-0.02] * n_days}, index=idx)
    return factor_panel, return_panel


def test_stop_loss_daily_precision_triggers_at_threshold():
    """Daily stop_loss should trigger at correct equity level, not after full month."""
    fp, rp = _make_crashing_panels(60)
    engine = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp,
        n_stocks=1, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    # -2% daily × ~11 days ≈ -20%; stop_loss=-0.20 should trigger around day 11
    result = engine.run(ranking_factor="alpha", ascending=False, stop_loss=-0.20)
    assert result.stop_triggered is True
    assert result.stop_trigger_date != ""
    # Equity should be close to 0.80 (not 0.61 like the old monthly-lag bug)
    assert 0.78 <= result.terminal_value <= 0.82, (
        f"Expected ~0.80, got {result.terminal_value} (monthly-lag bug would give ~0.61)"
    )


def test_stop_loss_no_trigger_when_above_threshold():
    """stop_triggered should be False when equity stays above threshold."""
    fp, rp = _make_crashing_panels(60)
    engine = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp,
        n_stocks=1, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    # Set stop_loss=-0.80 (needs -80% to trigger — never happens in 60 days of -2%)
    result = engine.run(ranking_factor="alpha", ascending=False, stop_loss=-0.80)
    assert result.stop_triggered is False
    assert result.stop_trigger_date == ""


def test_stop_loss_stops_immediately_within_month():
    """Stop should trigger within the same month, not wait for month-end."""
    fp, rp = _make_crashing_panels(60)
    engine = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp,
        n_stocks=1, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    result = engine.run(ranking_factor="alpha", ascending=False, stop_loss=-0.10)
    assert result.stop_triggered is True
    # With -10% stop, trigger around day 5-6
    # So only 1 monthly return should be recorded (partial month)
    assert len(result.monthly_returns) <= 2, (
        f"Should stop early, got {len(result.monthly_returns)} monthly returns"
    )


def test_stop_loss_with_multiple_stocks_crash_together():
    """Multi-stock portfolio crashing simultaneously should trigger stop correctly."""
    dates = pd.bdate_range("2024-01-01", periods=60)
    symbols = ["a", "b", "c", "d", "e"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    factor_panel = pd.DataFrame({"alpha": [5,4,3,2,1] * len(dates)}, index=idx)
    return_panel = pd.DataFrame({"daily_return": [-0.03] * len(dates) * len(symbols)}, index=idx)
    engine = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        n_stocks=3, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    result = engine.run(ranking_factor="alpha", ascending=False, stop_loss=-0.15)
    assert result.stop_triggered is True
    # All stocks crash uniformly, so trigger at same level as single stock
    assert result.terminal_value > 0.83, (
        f"Expected >0.83 (stop at 0.85), got {result.terminal_value}"
    )


def test_stop_loss_fields_in_result_without_stop():
    """stop_triggered should be False and stop_trigger_date empty when no stop_loss set."""
    fp, rp = _panels_with_predictive_signal()
    engine = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp,
        n_stocks=2, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    result = engine.run(ranking_factor="alpha_signal", ascending=False)
    assert hasattr(result, "stop_triggered")
    assert hasattr(result, "stop_trigger_date")
    assert result.stop_triggered is False
    assert result.stop_trigger_date == ""


# ── FX04: trailing_stop ──

def _make_rise_then_crash_panels(rising_days: int = 20, crashing_days: int = 60,
                                   daily_rise: float = 0.02, daily_crash: float = -0.02):
    """Create panels where stocks first rise, then crash."""
    n_days = rising_days + crashing_days
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    symbols = ["stock"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    factor_panel = pd.DataFrame({"alpha": [1.0] * n_days}, index=idx)
    returns = [daily_rise] * rising_days + [daily_crash] * crashing_days
    return_panel = pd.DataFrame({"daily_return": returns}, index=idx)
    return factor_panel, return_panel


def test_trailing_stop_triggers_after_peak_drawdown():
    """trailing_stop should trigger when equity drops below peak-equity threshold.

    Creates a panel with 120 business days (~6 months): rising phase in Feb,
    crash in Mar-May. The engine processes these as monthly rebalance periods.
    We don't assert exact terminal values due to period-boundary effects;
    we verify the stop triggered at a reasonable level (above 70% of initial).
    """
    # 40 rising days + 80 crashing = 120 total business days (~6 months)
    n_rise = 40
    n_crash = 80
    dates = pd.bdate_range("2024-01-01", periods=n_rise + n_crash)
    symbols = ["stock"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    factor_panel = pd.DataFrame({"alpha": [1.0] * len(dates)}, index=idx)
    returns = [0.015] * n_rise + [-0.025] * n_crash
    return_panel = pd.DataFrame({"daily_return": returns}, index=idx)

    engine = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        n_stocks=1, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    result = engine.run(
        ranking_factor="alpha", ascending=False,
        trailing_stop=0.25,
    )
    assert result.stop_triggered is True
    assert result.stop_trigger_date != ""
    # Terminal should be above 50% of initial (trailing_stop=25% from peak,
    # but peak may be close to 1.0 if rise is short)
    assert result.terminal_value > 0.50, (
        f"trailing_stop should preserve >50% of capital, got {result.terminal_value}"
    )
    # Should be well above what a -25% fixed stop would give
    assert result.terminal_value > 0.75, (
        f"trailing_stop should trigger above fixed stop level, got {result.terminal_value}"
    )


def test_trailing_stop_no_trigger_when_rising():
    """trailing_stop should not trigger when equity keeps rising."""
    # 180 business days (~8 months) of steady rise at +1.5%/day
    dates = pd.bdate_range("2024-01-01", periods=180)
    symbols = ["stock"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    factor_panel = pd.DataFrame({"alpha": [1.0] * len(dates)}, index=idx)
    return_panel = pd.DataFrame({"daily_return": [0.015] * len(dates)}, index=idx)

    engine = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        n_stocks=1, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    result = engine.run(
        ranking_factor="alpha", ascending=False,
        trailing_stop=0.25,
    )
    assert result.stop_triggered is False
    # After ~8 months of +1.5%/day, equity should be well above initial
    assert result.terminal_value > 3.0, (
        f"Rising portfolio should be >3.0x, got {result.terminal_value}"
    )


def test_trailing_stop_and_stop_loss_together():
    """Both trailing_stop and stop_loss active: whichever triggers first wins."""
    fp, rp = _make_crashing_panels(60)
    engine = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp,
        n_stocks=1, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    # stop_loss=-0.10 triggers at equity=0.90 (day ~5)
    # trailing_stop=0.50 triggers at equity=0.50×peak=0.50 (day ~35)
    # stop_loss triggers first
    result = engine.run(
        ranking_factor="alpha", ascending=False,
        stop_loss=-0.10,
        trailing_stop=0.50,
    )
    assert result.stop_triggered is True
    assert 0.88 <= result.terminal_value <= 0.92, (
        f"stop_loss=-0.10 should trigger at ~0.90, got {result.terminal_value}"
    )


def test_trailing_stop_independent_of_stop_loss():
    """trailing_stop reference is peak equity, not initial capital — verify independence."""
    fp, rp = _make_rise_then_crash_panels(
        rising_days=30, crashing_days=60,
        daily_rise=0.015, daily_crash=-0.03,
    )
    engine = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp,
        n_stocks=1, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    # stop_loss=-0.70: needs -70% of initial (1.0 → 0.30), won't trigger
    # trailing_stop=0.25: needs -25% of peak, will trigger during crash
    result = engine.run(
        ranking_factor="alpha", ascending=False,
        stop_loss=-0.70,
        trailing_stop=0.25,
    )
    assert result.stop_triggered is True
    # stop_loss at 0.30 would require equity < 0.30; trailing_stop triggers much earlier
    assert result.terminal_value > 0.50, (
        f"trailing_stop should trigger above 0.50, got {result.terminal_value}"
    )
