import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from tools.backtest_mvp.factors.templates import add_template_signals, golden_combo, template_fundamental_value, template_mean_reversion, template_value_momentum


def _panel():
    dates = pd.date_range("2024-01-01", periods=5)
    symbols = ["a", "b"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    return pd.DataFrame({
        "close": [10, 20, 11, 19, 12, 18, 13, 17, 14, 16],
        "roe_ttm": [0.1, 0.2, 0.11, 0.19, 0.12, 0.18, 0.13, 0.17, 0.14, 0.16],
        "mcap": [10, 20, 10, 20, 10, 20, 10, 20, 10, 20],
        "pb": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "mom20d": [0.1, 0.2] * 5,
        "vol20d": [0.3, 0.4] * 5,
        "max_ret": [0.05, 0.10] * 5,
        "sw_industry_2": ["g", "g"] * 5,
    }, index=idx)


def test_template_outputs_panel_aligned_series():
    panel = _panel()
    signal = template_fundamental_value(panel, window=2)
    assert signal.index.equals(panel.index)
    assert signal.notna().sum() > 0


def test_golden_combo_and_add_template_signals():
    panel = _panel()
    signal = golden_combo(panel, window=2)
    assert signal.index.equals(panel.index)

    enriched = add_template_signals(panel, ["golden_combo"], golden_combo={"window": 2})
    assert "golden_combo" in enriched.columns
    assert enriched.index.equals(panel.index)


def test_value_momentum_template():
    panel = _panel()
    signal = template_value_momentum(
        panel,
        fundamental_field="roe_ttm",
        fundamental_window=2,
        momentum_window=2,
        decay_window=2,
    )
    assert signal.index.equals(panel.index)
    assert signal.notna().sum() > 0

    enriched = add_template_signals(
        panel, ["value_momentum"],
        value_momentum={"fundamental_window": 2, "momentum_window": 2, "decay_window": 2},
    )
    assert "value_momentum" in enriched.columns


def test_mean_reversion_template():
    panel = _panel()
    signal = template_mean_reversion(
        panel,
        short_window=2,
        long_window=3,
        decay_window=2,
    )
    assert signal.index.equals(panel.index)
    assert signal.notna().sum() > 0

    enriched = add_template_signals(
        panel, ["mean_reversion"],
        mean_reversion={"short_window": 2, "decay_window": 2},
    )
    assert "mean_reversion" in enriched.columns

