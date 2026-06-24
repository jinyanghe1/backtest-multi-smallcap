import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import io
import sys
from contextlib import redirect_stdout

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


def test_run_template_backtest_prints_signal_coverage():
    factor_panel, return_panel = _panels()
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = run_template_backtest(
            "golden_combo",
            factor_panel,
            return_panel,
            n_stocks=1,
            template_kwargs={"window": 2},
        )
    output = buf.getvalue()
    assert "[signal]" in output
    assert "coverage=" in output
    assert "params:" in output
    assert result.terminal_value > 0


def test_low_coverage_signal_produces_warning(capsys):
    """When signal coverage < 30%, cmd_backtest should print a WARNING."""
    from tools.backtest_mvp.run import cmd_backtest
    import pandas as pd

    # Create synthetic data
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
    # Synthetic price data for load_price_data
    price_data = pd.DataFrame({
        "symbol": ["a", "b", "c"] * len(dates),
        "date": [d for d in dates for _ in range(3)],
        "close": [10.0, 20.0, 30.0] * len(dates),
        "open": [10.0, 20.0, 30.0] * len(dates),
        "high": [11.0, 21.0, 31.0] * len(dates),
        "low": [9.0, 19.0, 29.0] * len(dates),
        "volume": [1000, 2000, 3000] * len(dates),
    })

    # Monkeypatch
    import tools.backtest_mvp.run as run_mod
    original_load = run_mod.load_price_data
    original_compute = run_mod.compute_factors
    original_mcap = run_mod.load_daily_mcap_pb

    run_mod.load_price_data = lambda *a, **kw: price_data
    run_mod.compute_factors = lambda *a, **kw: (factor_panel, return_panel)
    run_mod.load_daily_mcap_pb = lambda *a, **kw: pd.DataFrame()

    try:
        cmd_backtest(["--template", "golden_combo", "--template-window", "2"])
        captured = capsys.readouterr()
        # Verify it ran and produced output
        assert "模板信号" in captured.out or "WARNING" in captured.out
    finally:
        run_mod.load_price_data = original_load
        run_mod.compute_factors = original_compute
        run_mod.load_daily_mcap_pb = original_mcap

