import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from tools.backtest_mvp.research_loop import AlphaCandidate, ResearchLoop, validate_metrics
from tools.backtest_mvp.research_loop.validators import validate_signal_coverage


def _panels():
    dates = pd.bdate_range("2024-01-01", periods=90)
    symbols = ["a", "b", "c"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    signal = [3.0, 2.0, 1.0] * len(dates)
    factor_panel = pd.DataFrame({"alpha_signal": signal}, index=idx)
    returns = []
    for _date in dates:
        returns.extend([0.002, 0.001, -0.001])
    return_panel = pd.DataFrame({"daily_return": returns}, index=idx)
    return factor_panel, return_panel


def test_research_loop_simulate_updates_candidate_metrics():
    factor_panel, return_panel = _panels()
    alpha = AlphaCandidate(expr="alpha_signal")
    loop = ResearchLoop(n_stocks=1, rebalance_freq="M")
    result = loop.simulate(alpha, factor_panel, return_panel, ascending=False)

    assert result.terminal_value > 0
    assert "sharpe" in alpha.metrics
    assert alpha.turnover >= 0


def test_validate_metrics_uses_percent_units():
    factor_panel, return_panel = _panels()
    loop = ResearchLoop(n_stocks=1, rebalance_freq="M")
    result = loop.simulate(AlphaCandidate(expr="alpha_signal"), factor_panel, return_panel)
    validation = validate_metrics(result, thresholds={
        "sharpe": -10,
        "fitness": -10,
        "turnover": (0, 100),
        "drawdown": 100,
        "self_correlation": 0.7,
    })

    assert validation.passed


def test_signal_coverage_validator_ok():
    factor_panel, _ = _panels()
    result = validate_signal_coverage(factor_panel, "alpha_signal")
    assert result["status"] == "ok"
    assert result["coverage"] == 1.0


def test_signal_coverage_validator_warning():
    """Coverage between 10% and 30% should be 'warning'."""
    dates = pd.bdate_range("2024-01-01", periods=100)
    symbols = ["a", "b", "c"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    # 20% non-null: 1 in 5
    vals = [3.0, np.nan, np.nan, np.nan, np.nan] * 60  # 300 total, 60 non-null = 20%
    panel = pd.DataFrame({"sparse_signal": vals}, index=idx)
    result = validate_signal_coverage(panel, "sparse_signal")
    assert result["status"] == "warning"
    assert 0.1 <= result["coverage"] < 0.3


def test_signal_coverage_validator_error():
    """Coverage below 10% should be 'error'."""
    dates = pd.bdate_range("2024-01-01", periods=100)
    symbols = ["a", "b", "c"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    vals = [np.nan] * 300
    vals[0] = 1.0  # only 1 non-null out of 300
    panel = pd.DataFrame({"rare_signal": vals}, index=idx)
    result = validate_signal_coverage(panel, "rare_signal")
    assert result["status"] == "error"
    assert result["coverage"] < 0.1


def test_signal_coverage_missing_column():
    panel, _ = _panels()
    result = validate_signal_coverage(panel, "nonexistent")
    assert result["status"] == "error"

