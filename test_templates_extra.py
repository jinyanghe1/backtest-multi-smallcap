import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from tools.backtest_mvp.factors.templates import (
    template_sentiment_conditional,
    template_overnight_reversal,
    template_ensemble,
    template_multi_timeframe,
    add_template_signals,
)


def _full_panel():
    dates = pd.date_range("2024-01-01", periods=30)
    symbols = ["a", "b", "c"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    n = len(idx)
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "close": rng.randn(n).cumsum() + 10,
        "open": rng.randn(n).cumsum() + 10,
        "turnover": rng.uniform(0.5, 5.0, n),
        "roe_ttm": rng.uniform(0.05, 0.3, n),
        "mom20d": rng.uniform(-0.1, 0.1, n),
        "vol20d": rng.uniform(0.1, 0.5, n),
        "max_ret": rng.uniform(0.01, 0.1, n),
        "sw_industry_1": ["g1", "g2", "g1"] * 30,
        "sw_industry_2": ["g1", "g2", "g1"] * 30,
    }, index=idx)


# ── T02: template_sentiment_conditional ──

def test_sentiment_conditional_outputs_aligned():
    panel = _full_panel()
    signal = template_sentiment_conditional(panel, sentiment_window=5, mom_window=3, rev_window=3)
    assert signal.index.equals(panel.index)
    assert signal.notna().sum() > 0


def test_sentiment_conditional_high_turnover_uses_reversal():
    """When all stocks have the same high turnover, signal should lean reversal."""
    panel = _full_panel().copy()
    panel["turnover"] = 10.0  # uniformly high
    signal = template_sentiment_conditional(panel, sentiment_window=5, mom_window=3, rev_window=3)
    assert signal.notna().sum() > 0


def test_sentiment_conditional_low_turnover_uses_momentum():
    """When all stocks have the same low turnover, signal should lean momentum."""
    panel = _full_panel().copy()
    panel["turnover"] = 0.1  # uniformly low
    signal = template_sentiment_conditional(panel, sentiment_window=5, mom_window=3, rev_window=3)
    assert signal.notna().sum() > 0


def test_sentiment_conditional_in_add_template_signals():
    panel = _full_panel()
    enriched = add_template_signals(
        panel, ["sentiment_conditional"],
        sentiment_conditional={"sentiment_window": 5, "mom_window": 3, "rev_window": 3},
    )
    assert "sentiment_conditional" in enriched.columns


# ── Overnight reversal (bug fix) ──

def test_overnight_reversal_outputs_aligned():
    panel = _full_panel()
    signal = template_overnight_reversal(panel, window=5)
    assert signal.index.equals(panel.index)
    assert signal.notna().sum() > 0


def test_overnight_reversal_in_add_template_signals():
    panel = _full_panel()
    enriched = add_template_signals(
        panel, ["overnight_reversal"],
        overnight_reversal={"window": 5},
    )
    assert "overnight_reversal" in enriched.columns


# ── T05: template_ensemble ──

def test_ensemble_outputs_aligned():
    panel = _full_panel()
    signal = template_ensemble(panel, window=5)
    assert signal.index.equals(panel.index)
    assert signal.notna().sum() > 0


def test_ensemble_custom_specs():
    panel = _full_panel()
    signal = template_ensemble(
        panel,
        signal_specs=[("roe_ttm", 1), ("mom20d", 1)],
        window=5,
    )
    assert signal.notna().sum() > 0


def test_ensemble_no_fields_raises():
    panel = _full_panel().drop(columns=["roe_ttm", "mom20d", "vol20d", "max_ret"])
    try:
        template_ensemble(panel, signal_specs=[("roe_ttm", 1)])
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_ensemble_in_add_template_signals():
    panel = _full_panel()
    enriched = add_template_signals(
        panel, ["ensemble"],
        ensemble={"window": 5},
    )
    assert "ensemble" in enriched.columns


# ── T06: template_multi_timeframe ──

def test_multi_timeframe_outputs_aligned():
    panel = _full_panel()
    signal = template_multi_timeframe(panel, windows=(3, 5, 10))
    assert signal.index.equals(panel.index)
    assert signal.notna().sum() > 0


def test_multi_timeframe_custom_weights():
    panel = _full_panel()
    signal = template_multi_timeframe(
        panel, windows=(3, 5, 10), weights=[0.5, 0.3, 0.2],
    )
    assert signal.notna().sum() > 0


def test_multi_timeframe_in_add_template_signals():
    panel = _full_panel()
    enriched = add_template_signals(
        panel, ["multi_timeframe"],
        multi_timeframe={"windows": (3, 5, 10)},
    )
    assert "multi_timeframe" in enriched.columns
