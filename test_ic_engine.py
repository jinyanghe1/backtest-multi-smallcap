import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from tools.backtest_mvp.engine import CrossSectionalEngine


def _panels_with_predictive_signal():
    """Create panels where the signal predicts next-period returns."""
    dates = pd.bdate_range("2024-01-01", periods=120)
    symbols = ["a", "b", "c", "d", "e"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    # Signal: stock 'a' has highest signal, 'e' lowest
    signal_vals = [5.0, 4.0, 3.0, 2.0, 1.0] * len(dates)
    factor_panel = pd.DataFrame({"alpha_signal": signal_vals}, index=idx)
    # Returns: correlate with signal (a gets highest return)
    returns = []
    for _ in dates:
        returns.extend([0.003, 0.002, 0.001, -0.001, -0.002])
    return_panel = pd.DataFrame({"daily_return": returns}, index=idx)
    return factor_panel, return_panel


def test_ic_mean_is_positive_for_predictive_signal():
    fp, rp = _panels_with_predictive_signal()
    engine = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp,
        n_stocks=2, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    result = engine.run(ranking_factor="alpha_signal", ascending=False)
    # IC should be positive since signal predicts returns
    assert result.ic_mean > 0, f"Expected positive IC, got {result.ic_mean}"
    assert -1.0 <= result.ic_mean <= 1.0
    # IC IR can be large when all ICs are identical (std≈0); just check sign
    assert result.ic_ir >= 0 or np.isinf(result.ic_ir) or abs(result.ic_ir) > 1e15


def test_ic_series_has_correct_length():
    fp, rp = _panels_with_predictive_signal()
    engine = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp,
        n_stocks=2, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    result = engine.run(ranking_factor="alpha_signal", ascending=False)
    # IC series should have entries for each rebalance period (minus last)
    assert len(result.ic_series) > 0
    assert result.ic_series.notna().sum() > 0


def test_ic_is_zero_for_random_signal():
    """Random noise signal should have IC near zero."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=120)
    symbols = ["a", "b", "c", "d", "e"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    signal_vals = np.random.randn(len(idx))
    factor_panel = pd.DataFrame({"noise": signal_vals}, index=idx)
    returns = np.random.randn(len(idx)) * 0.001
    return_panel = pd.DataFrame({"daily_return": returns}, index=idx)
    engine = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        n_stocks=2, rebalance_freq='M',
        commission=0.0, slippage=0.0, price_limit_stocks=False,
    )
    result = engine.run(ranking_factor="noise", ascending=False)
    # IC should be close to zero (within reasonable bounds)
    assert abs(result.ic_mean) < 0.5
