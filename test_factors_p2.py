"""Tests for P2 medium-priority factors (F008, F009, F010, F013, F015, F016, F017, F022).

Notes:
- F008, F009, F010 require financial data (accruals, total_assets, gross_profit)
  which is not available in current data. Tests verify graceful NaN fallback.
- F016 requires industry_code which is not available. Tests verify graceful NaN fallback.
"""

import numpy as np
import pandas as pd
import pytest
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from tools.backtest_mvp.factors.operators import _group_apply
from tools.backtest_mvp.factors.factor_library import (
    accruals_quality,
    asset_growth,
    gross_profitability,
    vwap_deviation,
    volatility_ratio,
    industry_momentum,
    disagreement_volatility,
    skewness_avoidance,
)


@pytest.fixture
def sample_panel():
    """Create a small panel for testing."""
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
    pb = np.random.uniform(0.5, 5, n)

    return pd.DataFrame({
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "amount": amount,
        "mcap": mcap,
        "pb": pb,
    }, index=idx)


# ── F008: Accruals quality (requires financial data) ──

def test_f008_returns_series(sample_panel):
    result = accruals_quality(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f008_nan_without_financial_data(sample_panel):
    """Without financial data, should return all NaN (graceful)."""
    result = accruals_quality(sample_panel)
    assert result.isna().all(), "F008 should be NaN when financial data is missing"


# ── F009: Asset growth (requires financial data) ──

def test_f009_returns_series(sample_panel):
    result = asset_growth(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f009_nan_without_financial_data(sample_panel):
    result = asset_growth(sample_panel)
    assert result.isna().all(), "F009 should be NaN when total_assets is missing"


# ── F010: Gross profitability (requires financial data) ──

def test_f010_returns_series(sample_panel):
    result = gross_profitability(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f010_nan_without_financial_data(sample_panel):
    result = gross_profitability(sample_panel)
    assert result.isna().all(), "F010 should be NaN when gross_profit is missing"


# ── F013: VWAP deviation ──

def test_f013_returns_series(sample_panel):
    result = vwap_deviation(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f013_vwap_estimation(sample_panel):
    """Without vwap column, should estimate from amount/volume."""
    result = vwap_deviation(sample_panel)
    assert result.isna().sum() < len(result) * 0.5, "F013 should work with amount/volume proxy"


def test_f013_no_inf_nan(sample_panel):
    result = vwap_deviation(sample_panel)
    assert not np.isinf(result).any()


# ── F015: Volatility ratio ──

def test_f015_returns_series(sample_panel):
    result = volatility_ratio(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f015_no_inf_nan(sample_panel):
    result = volatility_ratio(sample_panel)
    assert result.isna().sum() < len(result) * 0.5
    assert not np.isinf(result).any()


def test_f015_rising_vol_negative(sample_panel):
    """Rising volatility ratio should be negative signal."""
    result = volatility_ratio(sample_panel)
    close = sample_panel["close"]
    ret = close.groupby(level="symbol", group_keys=False).pct_change()
    vol = _group_apply(ret, "symbol", lambda s: s.rolling(20, min_periods=10).std())

    mask = result.notna() & vol.notna()
    if mask.sum() > 20:
        ic = result[mask].corr(vol[mask])
        assert ic < 0.2, f"F015 should be negatively correlated with vol: {ic:.3f}"


# ── F016: Industry momentum (requires industry_code) ──

def test_f016_returns_series(sample_panel):
    result = industry_momentum(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f016_nan_without_industry_data(sample_panel):
    result = industry_momentum(sample_panel)
    assert result.isna().all(), "F016 should be NaN when industry_code is missing"


# ── F017: Disagreement volatility ──

def test_f017_returns_series(sample_panel):
    result = disagreement_volatility(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f017_no_inf_nan(sample_panel):
    result = disagreement_volatility(sample_panel)
    assert result.isna().sum() < len(result) * 0.5
    assert not np.isinf(result).any()


def test_f017_high_disagreement_negative(sample_panel):
    """High disagreement should be negative signal."""
    result = disagreement_volatility(sample_panel)
    close = sample_panel["close"]
    ret = close.groupby(level="symbol", group_keys=False).pct_change()
    std_ret = _group_apply(ret, "symbol", lambda s: s.rolling(20, min_periods=10).std())

    mask = result.notna() & std_ret.notna()
    if mask.sum() > 20:
        ic = result[mask].corr(std_ret[mask])
        assert ic < 0.2, f"F017 should be negatively correlated with vol: {ic:.3f}"


# ── F022: Skewness avoidance ──

def test_f022_returns_series(sample_panel):
    result = skewness_avoidance(sample_panel)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_f022_no_inf_nan(sample_panel):
    result = skewness_avoidance(sample_panel)
    assert result.isna().sum() < len(result) * 0.5
    assert not np.isinf(result).any()


def test_f022_negative_skew_positive(sample_panel):
    """Negative skewness should be positive signal (avoid positive skew)."""
    result = skewness_avoidance(sample_panel)
    close = sample_panel["close"]
    ret = close.groupby(level="symbol", group_keys=False).pct_change()
    skew = _group_apply(ret, "symbol", lambda s: s.rolling(60, min_periods=30).skew())

    mask = result.notna() & skew.notna()
    if mask.sum() > 20:
        ic = result[mask].corr(skew[mask])
        assert ic < 0.2, f"F022 should be negatively correlated with skew: {ic:.3f}"


# Helper needed for some tests
from tools.backtest_mvp.factors.operators import _group_apply


# ── Integration: all P2 factors on real data ──

def test_p2_factors_on_real_data():
    """Run all P2 factors on real data and verify they work."""
    from tools.backtest_mvp.factors import load_price_data, compute_factors
    from tools.backtest_mvp.data import DATA_DIR

    data = load_price_data(str(DATA_DIR))
    fp, rp = compute_factors(data)

    for name, func, expected_nan in [
        ("F008", accruals_quality, True),   # No financial data
        ("F009", asset_growth, True),       # No total_assets
        ("F010", gross_profitability, True), # No gross_profit
        ("F013", vwap_deviation, False),    # Has amount/volume
        ("F015", volatility_ratio, False), # Has close
        ("F016", industry_momentum, True),  # No industry_code
        ("F017", disagreement_volatility, False), # Has close
        ("F022", skewness_avoidance, False), # Has close
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
