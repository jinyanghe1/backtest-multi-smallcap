import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from tools.backtest_mvp.research_loop import AlphaCandidate, ResearchLoop, validate_metrics


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

