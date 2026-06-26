"""Unit and factor-backtest tests for template strategies (E-H).

Verifies: (1) strategy definitions are well-formed,
(2) template signals compute without errors,
(3) backtest produces non-zero results (= strategy is alive).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import pytest

from tools.backtest_mvp.engine import CrossSectionalEngine
from tools.backtest_mvp.strategies_v2 import (
    TEMPLATE_STRATEGIES,
    strategy_regime_momentum,
    strategy_sentiment_conditional,
    strategy_ensemble,
    strategy_multi_timeframe,
    template_micro_cap_filter,
)
from tools.backtest_mvp.factors.templates import add_template_signals


# ── Helper: create synthetic factor/return panels ──

def _make_synthetic_panels(n_days: int = 252, n_stocks: int = 100):
    """Create synthetic factor/return panels with realistic columns.

    252 trading days ≈ 1 year, 100 stocks.
    Each stock has random returns and plausible factor values.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=n_days)

    records = []
    for i in range(n_stocks):
        symbol = f"STOCK{i:04d}"
        base_close = rng.uniform(5, 50)
        daily_returns = rng.normal(0.0005, 0.025, n_days).clip(-0.1, 0.1)
        closes = base_close * np.cumprod(1 + daily_returns)
        opens = closes * rng.uniform(0.98, 1.02, n_days)

        for j, date in enumerate(dates):
            records.append({
                "date": date,
                "symbol": symbol,
                "close": closes[j],
                "open": opens[j],
                "mcap": rng.uniform(5, 50) * (1 + 0.02 * j / n_days),
                "pb": rng.uniform(0.5, 8.0),
                "pe": rng.choice([rng.uniform(5, 100), np.nan], p=[0.75, 0.25]),
                "mom20d": daily_returns[max(0, j-20):j+1].sum() if j >= 20 else 0.0,
                "mom60d": daily_returns[max(0, j-60):j+1].sum() if j >= 60 else 0.0,
                "turnover": rng.uniform(0.2, 5.0),
                "vol20d": abs(daily_returns[max(0, j-20):j+1].std() * np.sqrt(252)) if j >= 20 else 0.3,
                "max_ret": rng.uniform(0, 0.1),
                "daily_return": daily_returns[j],
            })

    df = pd.DataFrame(records)
    factor_panel = df.set_index(["date", "symbol"])[
        ["mcap", "pb", "pe", "mom20d", "mom60d", "turnover", "vol20d",
         "max_ret", "close", "open"]
    ].sort_index()
    return_panel = df.set_index(["date", "symbol"])[["daily_return"]].sort_index()
    return factor_panel, return_panel


# ── T01: Strategy definitions are well-formed ──

def test_template_strategies_have_required_fields():
    """All template strategies should have name, ranking_factor, template, n_stocks."""
    for s in TEMPLATE_STRATEGIES:
        assert "name" in s, f"{s} missing name"
        assert "template" in s, f"{s} missing template"
        assert "ranking_factor" in s, f"{s} missing ranking_factor"
        assert "n_stocks" in s, f"{s} missing n_stocks"
        assert s["template"] == s["ranking_factor"], (
            f"{s['name']}: template={s['template']} != ranking_factor={s['ranking_factor']}"
        )


def test_template_strategies_have_stop_loss():
    """Template strategies should have a stop_loss to survive bear markets."""
    for s in TEMPLATE_STRATEGIES:
        assert "stop_loss" in s, f"{s['name']} missing stop_loss"
        assert s["stop_loss"] is not None, f"{s['name']}: stop_loss should not be None"


# ── T02: Template signal computation ──

def test_regime_momentum_signal_computes():
    """regime_momentum template should produce valid signals."""
    fp, _ = _make_synthetic_panels(60, 50)
    enriched = add_template_signals(fp, ["regime_momentum"])
    assert "regime_momentum" in enriched.columns
    signal = enriched["regime_momentum"]
    assert signal.notna().sum() > 0
    # Signal should be rank-based (between 0 and 1)
    assert 0 <= signal.max() <= 1.1
    assert -0.1 <= signal.min() <= 1


def test_sentiment_conditional_signal_computes():
    """sentiment_conditional template should produce valid signals."""
    fp, _ = _make_synthetic_panels(60, 50)
    enriched = add_template_signals(fp, ["sentiment_conditional"])
    assert "sentiment_conditional" in enriched.columns
    signal = enriched["sentiment_conditional"]
    assert signal.notna().sum() > 0


def test_ensemble_signal_computes():
    """ensemble template should work with factor panel fields."""
    fp, _ = _make_synthetic_panels(60, 50)
    enriched = add_template_signals(fp, ["ensemble"])
    assert "ensemble" in enriched.columns
    signal = enriched["ensemble"]
    assert signal.notna().sum() > 0


def test_multi_timeframe_signal_computes():
    """multi_timeframe template should produce valid signals."""
    fp, _ = _make_synthetic_panels(60, 50)
    enriched = add_template_signals(
        fp, ["multi_timeframe"],
        multi_timeframe={"windows": (5, 10, 20), "weights": (0.5, 0.3, 0.2)},
    )
    assert "multi_timeframe" in enriched.columns
    signal = enriched["multi_timeframe"]
    assert signal.notna().sum() > 0


# ── T03: Template filter function ──

def test_template_micro_cap_filter():
    """template_micro_cap_filter should filter to micro-cap non-ST stocks."""
    fp, _ = _make_synthetic_panels(60, 50)
    # Get snapshot for a specific date
    snapshot = fp.xs(fp.index.get_level_values(0)[0], level=0)
    result = template_micro_cap_filter(snapshot, None, 0)
    assert isinstance(result, list)
    assert len(result) > 0  # should have some stocks


# ── T04: Factor backtest — annual return ≠ ~0% proves strategy is alive ──

@pytest.mark.slow
def test_backtest_regime_momentum_alive():
    """regime_momentum strategy should produce non-zero annual return."""
    fp, rp = _make_synthetic_panels(252, 100)
    enriched = add_template_signals(
        fp, ["regime_momentum"],
        regime_momentum={"price_field": "close"},
    )
    engine = CrossSectionalEngine(
        factor_panel=enriched, return_panel=rp,
        n_stocks=25, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=False,
    )
    result = engine.run(
        universe_filter=template_micro_cap_filter,
        ranking_factor="regime_momentum",
        ascending=False,
        stop_loss=-0.40,
    )
    assert result.stop_triggered is False, (
        f"Strategy stopped at {result.stop_trigger_date}"
    )
    # With random returns on synthetic data, annual return should be non-zero
    assert abs(result.annual_return) > 0.01, (
        f"Expected |annual_return| > 0.01%, got {result.annual_return}% (strategy is dead)"
    )


@pytest.mark.slow
def test_backtest_ensemble_alive():
    """ensemble strategy should produce non-zero annual return."""
    fp, rp = _make_synthetic_panels(252, 100)
    enriched = add_template_signals(fp, ["ensemble"], ensemble={"window": 20})
    engine = CrossSectionalEngine(
        factor_panel=enriched, return_panel=rp,
        n_stocks=25, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=False,
    )
    result = engine.run(
        universe_filter=template_micro_cap_filter,
        ranking_factor="ensemble",
        ascending=False,
        stop_loss=-0.40,
    )
    assert result.stop_triggered is False
    assert abs(result.annual_return) > 0.01


@pytest.mark.slow
def test_backtest_multi_timeframe_alive():
    """multi_timeframe strategy should produce non-zero annual return."""
    fp, rp = _make_synthetic_panels(252, 100)
    enriched = add_template_signals(
        fp, ["multi_timeframe"],
        multi_timeframe={
            "windows": (5, 10, 20, 60),
            "weights": (0.4, 0.3, 0.2, 0.1),
        },
    )
    engine = CrossSectionalEngine(
        factor_panel=enriched, return_panel=rp,
        n_stocks=25, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=False,
    )
    result = engine.run(
        universe_filter=template_micro_cap_filter,
        ranking_factor="multi_timeframe",
        ascending=False,
        stop_loss=-0.40,
    )
    assert result.stop_triggered is False
    assert abs(result.annual_return) > 0.01


@pytest.mark.slow
def test_backtest_sentiment_conditional_alive():
    """sentiment_conditional strategy should produce non-zero annual return."""
    fp, rp = _make_synthetic_panels(252, 100)
    enriched = add_template_signals(
        fp, ["sentiment_conditional"],
        sentiment_conditional={
            "turnover_field": "turnover",
            "price_field": "close",
        },
    )
    engine = CrossSectionalEngine(
        factor_panel=enriched, return_panel=rp,
        n_stocks=20, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=False,
    )
    result = engine.run(
        universe_filter=template_micro_cap_filter,
        ranking_factor="sentiment_conditional",
        ascending=False,
        stop_loss=-0.40,
    )
    assert result.stop_triggered is False
    assert abs(result.annual_return) > 0.01
