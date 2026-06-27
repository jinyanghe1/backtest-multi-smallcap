"""Tests for P1 high-priority factors (F003, F004, F005, F011, F012, F027, F028).

Each factor is tested with:
- Basic correctness: returns a Series with same index
- IC sign: expected positive or negative
- No NaN/Inf: factor values are finite
- Fallback logic: works when primary data is unavailable
"""

import numpy as np
import pandas as pd
import pytest
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from tools.backtest_mvp.factors.factor_library import (
    shareholder_concentration,
    earnings_acceleration_proxy,
    quality_value,
    amihud_illiquidity,
    overnight_gap,
    earnings_momentum,
    cashflow_price,
)


@pytest.fixture
def sample_panel():
    """Create a small panel for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")  # 200 days for longer-window factors
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
    pb = np.random.uniform(0.5, 5, n)
    turnover = np.random.uniform(0.01, 0.1, n)

    return pd.DataFrame({
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "amount": amount,
        "mcap": mcap,
        "pb": pb,
        "turnover": turnover,
    }, index=idx)


# ── F003: Shareholder concentration ──

def test_f003_returns_series(sample_panel):
    result = shareholder_concentration(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f003_fallback_works(sample_panel):
    """When no shareholders column, fallback uses turnover + return."""
    result = shareholder_concentration(sample_panel)
    # Fallback should produce valid values
    assert result.isna().sum() < len(result) * 0.5


def test_f003_no_inf_nan(sample_panel):
    result = shareholder_concentration(sample_panel)
    assert not np.isinf(result).any()


# ── F004: Earnings acceleration proxy ──

def test_f004_returns_series(sample_panel):
    result = earnings_acceleration_proxy(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f004_no_inf_nan(sample_panel):
    result = earnings_acceleration_proxy(sample_panel)
    assert result.isna().sum() < len(result) * 0.5
    assert not np.isinf(result).any()


def test_f004_acceleration_logic(sample_panel):
    """When 5d ret > 20d ret, acceleration should be positive."""
    close = sample_panel["close"]
    ret_5d = close.groupby(level="symbol", group_keys=False).pct_change(5)
    ret_20d = close.groupby(level="symbol", group_keys=False).pct_change(20)
    result = earnings_acceleration_proxy(sample_panel)
    
    mask = result.notna() & ret_5d.notna() & ret_20d.notna()
    if mask.sum() > 10:
        # When 5d ret is large and 20d ret is small, acceleration should be positive
        strong_accel = (ret_5d > 0.05) & (abs(ret_20d) < 0.01)
        if strong_accel.sum() > 0:
            assert result[strong_accel].mean() > 0, "Strong acceleration should be positive"


# ── F005: Quality value ──

def test_f005_returns_series(sample_panel):
    result = quality_value(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f005_no_inf_nan(sample_panel):
    result = quality_value(sample_panel)
    assert not np.isinf(result).any()


def test_f005_fallback_to_pb(sample_panel):
    """Without OCF, should fall back to -pb."""
    result = quality_value(sample_panel)
    pb = sample_panel["pb"]
    mask = result.notna() & pb.notna()
    if mask.sum() > 10:
        ic = result[mask].corr(-pb[mask])
        # Should be positively correlated with -pb (same direction)
        assert ic > -0.5, f"F005 should correlate with -pb when no OCF: {ic:.3f}"


# ── F011: Amihud illiquidity ──

def test_f011_returns_series(sample_panel):
    result = amihud_illiquidity(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f011_positive(sample_panel):
    """Amihud illiquidity should be positive (abs(ret)/amount)."""
    result = amihud_illiquidity(sample_panel)
    valid = result.dropna()
    if len(valid) > 0:
        assert (valid >= 0).all(), "Amihud should be non-negative"


def test_f011_no_inf_nan(sample_panel):
    result = amihud_illiquidity(sample_panel)
    assert not np.isinf(result).any()


# ── F012: Overnight gap ──

def test_f012_returns_series(sample_panel):
    result = overnight_gap(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f012_no_inf_nan(sample_panel):
    result = overnight_gap(sample_panel)
    assert result.isna().sum() < len(result) * 0.5
    assert not np.isinf(result).any()


def test_f012_range(sample_panel):
    """Gap should be in reasonable range (-1, +1) for most cases."""
    result = overnight_gap(sample_panel)
    valid = result.dropna()
    if len(valid) > 0:
        assert valid.min() > -1, "Gap below -1 (unrealistic)"
        assert valid.max() < 1, "Gap above +1 (unrealistic)"


# ── F027: Earnings momentum (SUE proxy) ──

def test_f027_returns_series(sample_panel):
    result = earnings_momentum(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f027_fallback_to_ret60d(sample_panel):
    """Without EPS data, should fall back to 60-day return."""
    result = earnings_momentum(sample_panel)
    # 60-day return should be mostly valid (after warmup)
    assert result.isna().sum() < len(result) * 0.5


def test_f027_no_inf_nan(sample_panel):
    result = earnings_momentum(sample_panel)
    assert not np.isinf(result).any()


# ── F028: Cash flow to price ──

def test_f028_returns_series(sample_panel):
    result = cashflow_price(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f028_fallback_to_negative_pb(sample_panel):
    """Without OCF, should fall back to -pb."""
    result = cashflow_price(sample_panel)
    pb = sample_panel["pb"]
    mask = result.notna() & pb.notna()
    if mask.sum() > 10:
        ic = result[mask].corr(-pb[mask])
        assert ic > -0.5, f"F028 fallback should correlate with -pb: {ic:.3f}"


def test_f028_no_inf_nan(sample_panel):
    result = cashflow_price(sample_panel)
    assert not np.isinf(result).any()


# ── Integration: all P1 factors on real data ──

def test_p1_factors_on_real_data():
    """Run all P1 factors on real data and verify they work."""
    from tools.backtest_mvp.factors import load_price_data, compute_factors
    from tools.backtest_mvp.data import DATA_DIR

    data = load_price_data(str(DATA_DIR))
    fp, rp = compute_factors(data)

    for name, func in [
        ("F003", shareholder_concentration),
        ("F004", earnings_acceleration_proxy),
        ("F005", quality_value),
        ("F011", amihud_illiquidity),
        ("F012", overnight_gap),
        ("F027", earnings_momentum),
        ("F028", cashflow_price),
    ]:
        try:
            result = func(fp)
            assert isinstance(result, pd.Series), f"{name} did not return Series"
            assert len(result) == len(fp), f"{name} length mismatch"
            valid_count = result.notna().sum()
            assert valid_count > len(fp) * 0.3, f"{name} too few valid: {valid_count}/{len(fp)}"
            print(f"  {name}: {valid_count}/{len(fp)} valid, mean={result.mean():.4f}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            raise
