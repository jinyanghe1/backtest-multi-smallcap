import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from tools.backtest_mvp.run import run_template_backtest


def _panels():
    dates = pd.bdate_range("2024-01-01", periods=90)
    symbols = ["a", "b", "c"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    factor_panel = pd.DataFrame({
        "mcap": [1.0, 2.0, 3.0] * len(dates),
        "pb": [1.0, 2.0, 3.0] * len(dates),
        "mom20d": [0.3, 0.2, 0.1] * len(dates),
        "vol20d": [0.1, 0.2, 0.3] * len(dates),
        "max_ret": [0.01, 0.02, 0.03] * len(dates),
    }, index=idx)
    returns = []
    for _ in dates:
        returns.extend([0.002, 0.001, -0.001])
    return_panel = pd.DataFrame({"daily_return": returns}, index=idx)
    return factor_panel, return_panel


def test_run_template_backtest_golden_combo():
    factor_panel, return_panel = _panels()
    result = run_template_backtest(
        "golden_combo",
        factor_panel,
        return_panel,
        n_stocks=1,
        template_kwargs={"window": 2},
    )

    assert result.terminal_value > 0
    assert result.annual_return is not None

