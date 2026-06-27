"""Tests for factor preprocessing pipeline.

Covers ROADMAP_P1 UC1-2 test cases:
    1. test_winsorize_mad
    2. test_winsorize_preserves_order
    3. test_zscore_cross_sectional
    4. test_preprocess_pipeline_full
"""

import numpy as np
import pandas as pd
import pytest

from tools.backtest_mvp.factors.preprocessing import (
    winsorize_mad,
    winsorize_percentile,
    winsorize_cross_sectional,
    zscore_cross_sectional,
    rank_normalize,
    preprocess_pipeline,
)


# ── Winsorization tests ───────────────────────────────────────────────

def test_winsorize_mad():
    """Outlier (100) should be clipped to median + 5*MAD bounds."""
    np.random.seed(42)
    values = np.random.randn(99) * 2
    values = np.append(values, 100.0)  # extreme outlier
    series = pd.Series(values, index=[f"s{i:03d}" for i in range(100)])

    result = winsorize_mad(series, n_mad=5.0, level=None)

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    upper_bound = median + 5 * mad

    assert result.max() <= upper_bound, f"Max {result.max()} > bound {upper_bound}"
    # The outlier should be clipped
    assert result.iloc[-1] < 100.0


def test_winsorize_preserves_order():
    """Winsorization should preserve rank order (Spearman corr = 1.0)."""
    np.random.seed(42)
    values = np.random.randn(50)
    series = pd.Series(values, index=[f"s{i:03d}" for i in range(50)])

    result = winsorize_mad(series, n_mad=5.0, level=None)

    rank_corr = series.corr(result, method="spearman")
    assert rank_corr >= 0.9999, f"Rank correlation changed: {rank_corr}"


def test_winsorize_percentile():
    """Percentile winsorization: clip to p1 and p99."""
    np.random.seed(42)
    values = np.random.randn(100) * 10
    series = pd.Series(values, index=[f"s{i:03d}" for i in range(100)])

    result = winsorize_percentile(series, lower=0.01, upper=0.99, level=None)

    p1 = np.percentile(values, 1)
    p99 = np.percentile(values, 99)
    assert result.min() >= p1
    assert result.max() <= p99


def test_winsorize_cross_sectional_panel():
    """Winsorize per-date for panel data."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    symbols = [f"s{i:03d}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    np.random.seed(42)
    values = np.random.randn(len(idx)) * 5
    # Add outlier to first date
    values[0] = 100.0
    series = pd.Series(values, index=idx)

    result = winsorize_cross_sectional(series, method="mad", level="date")

    # First date's outlier should be clipped
    assert result.iloc[0] < 100.0
    # Other dates should have different bounds (independent per date)
    assert result.notna().sum() > 0


# ── Standardization tests ─────────────────────────────────────────────

def test_zscore_cross_sectional():
    """Per-date zscore: mean ≈ 0, std ≈ 1."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    symbols = [f"s{i:03d}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    np.random.seed(42)
    values = np.random.randn(len(idx)) * 10 + 50
    series = pd.Series(values, index=idx)

    result = zscore_cross_sectional(series, level="date")

    # Per-date mean ≈ 0, std ≈ 1
    for date in dates:
        day_values = result.xs(date, level="date")
        assert abs(day_values.mean()) < 0.01, f"Date {date} mean = {day_values.mean():.4f}"
        assert abs(day_values.std(ddof=1) - 1.0) < 0.01, f"Date {date} std = {day_values.std():.4f}"


def test_zscore_constant_series():
    """Constant series → all zeros."""
    series = pd.Series([5.0] * 10, index=[f"s{i:03d}" for i in range(10)])
    result = zscore_cross_sectional(series, level=None)
    assert (result == 0.0).all()


def test_rank_normalize():
    """Rank normalization maps to approximately normal distribution."""
    np.random.seed(42)
    values = np.random.randn(100) * 10 + 50
    series = pd.Series(values, index=[f"s{i:03d}" for i in range(100)])

    result = rank_normalize(series, level=None)

    # Should be approximately standard normal
    assert abs(result.mean()) < 0.5
    assert 0.5 < result.std() < 2.0


# ── Pipeline tests ──────────────────────────────────────────────────────

def test_preprocess_pipeline_full():
    """Full pipeline: winsorize → zscore → neutralize → re-zscore."""
    np.random.seed(42)
    n = 50
    symbols = [f"s{i:03d}" for i in range(n)]

    # Raw factor with size bias: factor = 0.5 * size + noise
    size = np.random.randn(n) * 2 + 10
    noise = np.random.randn(n) * 0.5
    factor = 0.5 * size + noise
    factor_series = pd.Series(factor, index=symbols)

    controls = pd.DataFrame({"size": size}, index=symbols)

    result = preprocess_pipeline(
        factor_series, controls=controls,
        config={"neutralize": {"strength": 1.0, "enabled": True}}
    )

    # After full neutralization, size correlation should be near zero
    size_corr = result.corr(pd.Series(size, index=symbols))
    assert abs(size_corr) < 0.15, f"Size correlation after neutralization: {size_corr:.4f}"
    # Result should be standardized (mean ≈ 0, std ≈ 1)
    assert abs(result.mean()) < 0.1
    assert 0.8 < result.std() < 1.2


def test_preprocess_pipeline_no_controls():
    """Pipeline without controls: winsorize + zscore only."""
    np.random.seed(42)
    values = np.random.randn(50) * 10 + 50
    series = pd.Series(values, index=[f"s{i:03d}" for i in range(50)])

    result = preprocess_pipeline(series, controls=None)

    assert abs(result.mean()) < 0.1
    assert 0.8 < result.std() < 1.2


def test_preprocess_pipeline_panel():
    """Pipeline on panel data (MultiIndex)."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    symbols = [f"s{i:03d}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    np.random.seed(42)
    values = np.random.randn(len(idx)) * 10 + 50
    factor = pd.Series(values, index=idx)

    # Controls: size (same per date)
    size_values = np.random.randn(len(idx)) * 2 + 10
    controls = pd.DataFrame({"size": size_values}, index=idx)

    result = preprocess_pipeline(factor, controls=controls)

    assert result.index.equals(factor.index)
    # Per-date standardization
    for date in dates:
        day_values = result.xs(date, level="date")
        assert abs(day_values.mean()) < 0.1


def test_preprocess_pipeline_blend_strength():
    """strength=0.5 should reduce size correlation vs strength=0."""
    np.random.seed(42)
    n = 50
    symbols = [f"s{i:03d}" for i in range(n)]
    # Factor with strong size bias
    size = np.random.randn(n) * 2 + 10
    factor = pd.Series(0.8 * size + np.random.randn(n) * 0.3, index=symbols)
    controls = pd.DataFrame({"x": size}, index=symbols)

    full = preprocess_pipeline(factor, controls=controls, config={"neutralize": {"strength": 1.0}})
    half = preprocess_pipeline(factor, controls=controls, config={"neutralize": {"strength": 0.5}})
    none = preprocess_pipeline(factor, controls=controls, config={"neutralize": {"strength": 0.0}})

    # strength=1 should have lowest size correlation, strength=0 highest
    size_full = abs(full.corr(pd.Series(size, index=symbols)))
    size_half = abs(half.corr(pd.Series(size, index=symbols)))
    size_none = abs(none.corr(pd.Series(size, index=symbols)))
    
    assert size_full < size_half < size_none, \
        f"size_corr: full={size_full:.3f}, half={size_half:.3f}, none={size_none:.3f}"
