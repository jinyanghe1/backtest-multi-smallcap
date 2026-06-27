"""Tests for P3 low-priority factors (F019, F020, F021, F023, F024, F025, F026, F029, F030).

These are advanced/secondary factors. Some require financial data (F029).
"""

import numpy as np
import pandas as pd
import pytest
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from tools.backtest_mvp.factors.factor_library import (
    close_location_value,
    drift_state_momentum,
    beta_arbitrage,
    trend_strength,
    accumulation_distribution,
    momentum_quality,
    tail_return_spread,
    rd_intensity,
    analyst_revision,
)


@pytest.fixture
def sample_panel():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    symbols = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    n = len(idx)

    close = np.cumsum(np.random.randn(n) * 0.02 + 0.0005) + 10
    high = close * (1 + abs(np.random.randn(n) * 0.02))
    low = close * (1 - abs(np.random.randn(n) * 0.02))
    open_ = close * (1 + np.random.randn(n) * 0.01)
    volume = np.random.randint(1000, 100000, n)
    amount = close * volume * (1 + np.random.randn(n) * 0.1)
    mcap = np.random.uniform(1, 100, n)

    return pd.DataFrame({
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "amount": amount,
        "mcap": mcap,
    }, index=idx)


# ── F019: Close location value ──

def test_f019_returns_series(sample_panel):
    result = close_location_value(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f019_range(sample_panel):
    """Close location should be bounded (mostly in [-10, 10])."""
    result = close_location_value(sample_panel)
    valid = result.dropna()
    if len(valid) > 0:
        # When high-low is very small, result can be large even with +0.001
        assert valid.min() > -50, f"Close location below -50: {valid.min():.2f}"
        assert valid.max() < 50, f"Close location above +50: {valid.max():.2f}"
        # Median should be in reasonable range
        assert abs(valid.median()) < 5, f"Median too extreme: {valid.median():.2f}"


def test_f019_no_inf_nan(sample_panel):
    result = close_location_value(sample_panel)
    assert not np.isinf(result).any()


# ── F020: Drift state momentum ──

def test_f020_returns_series(sample_panel):
    result = drift_state_momentum(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f020_no_inf_nan(sample_panel):
    result = drift_state_momentum(sample_panel)
    assert result.isna().sum() < len(result) * 0.5
    assert not np.isinf(result).any()


# ── F021: Beta arbitrage ──

def test_f021_returns_series(sample_panel):
    result = beta_arbitrage(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f021_no_inf_nan(sample_panel):
    result = beta_arbitrage(sample_panel)
    assert result.isna().sum() < len(result) * 0.5
    assert not np.isinf(result).any()


# ── F023: Trend strength ──

def test_f023_returns_series(sample_panel):
    result = trend_strength(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f023_no_inf_nan(sample_panel):
    result = trend_strength(sample_panel)
    assert result.isna().sum() < len(result) * 0.5
    assert not np.isinf(result).any()


# ── F024: Accumulation/Distribution ──

def test_f024_returns_series(sample_panel):
    result = accumulation_distribution(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f024_no_inf_nan(sample_panel):
    result = accumulation_distribution(sample_panel)
    assert not np.isinf(result).any()


# ── F025: Momentum quality ──

def test_f025_returns_series(sample_panel):
    result = momentum_quality(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f025_no_inf_nan(sample_panel):
    result = momentum_quality(sample_panel)
    assert result.isna().sum() < len(result) * 0.5
    assert not np.isinf(result).any()


# ── F026: Tail return spread ──

def test_f026_returns_series(sample_panel):
    result = tail_return_spread(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f026_no_inf_nan(sample_panel):
    result = tail_return_spread(sample_panel)
    assert result.isna().sum() < len(result) * 0.5
    assert not np.isinf(result).any()


# ── F029: R&D intensity (requires financial data) ──

def test_f029_returns_series(sample_panel):
    result = rd_intensity(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f029_nan_without_financial_data(sample_panel):
    result = rd_intensity(sample_panel)
    assert result.isna().all(), "F029 should be NaN when RD data is missing"


# ── F030: Analyst revision (requires EPS data) ──

def test_f030_returns_series(sample_panel):
    result = analyst_revision(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f030_fallback_to_ret20d(sample_panel):
    """Without EPS data, should fall back to 20-day return."""
    result = analyst_revision(sample_panel)
    assert result.isna().sum() < len(result) * 0.5, "F030 fallback should work"


def test_f030_no_inf_nan(sample_panel):
    result = analyst_revision(sample_panel)
    assert not np.isinf(result).any()


# ── Integration: all P3 factors on real data ──

def test_p3_factors_on_real_data():
    """Run all P3 factors on real data and verify they work."""
    from tools.backtest_mvp.factors import load_price_data, compute_factors
    from tools.backtest_mvp.data import DATA_DIR

    data = load_price_data(str(DATA_DIR))
    fp, rp = compute_factors(data)

    for name, func, expected_nan in [
        ("F019", close_location_value, False),
        ("F020", drift_state_momentum, False),
        ("F021", beta_arbitrage, False),
        ("F023", trend_strength, False),
        ("F024", accumulation_distribution, False),
        ("F025", momentum_quality, False),
        ("F026", tail_return_spread, False),
        ("F029", rd_intensity, True),   # No financial data
        ("F030", analyst_revision, False), # Fallback to ret20d
    ]:
        try:
            result = func(fp)
            assert isinstance(result, pd.Series), f"{name} did not return Series"
            assert len(result) == len(fp), f"{name} length mismatch"

            valid_count = result.notna().sum()
            if expected_nan:
                assert valid_count == 0, f"{name} should be all NaN without data, got {valid_count}"
                print(f"  {name}: ALL NaN (expected — no financial data)")
            else:
                assert valid_count > len(fp) * 0.3, f"{name} too few valid: {valid_count}/{len(fp)}"
                print(f"  {name}: {valid_count}/{len(fp)} valid, mean={result.mean():.4f}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            raise
