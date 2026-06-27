"""Tests for factor neutralization operators.

References the 5 test cases from ROADMAP_P1 UC1-1:
    1. test_neutralize_ols_basic
    2. test_neutralize_ols_min_obs
    3. test_neutralize_ols_multiindex
    4. test_neutralize_ols_industry
    5. test_neutralize_ols_preserve_rank
"""

import numpy as np
import pandas as pd
import pytest

from tools.backtest_mvp.factors.neutralization import (
    neutralize_ols_residual,
    neutralize_by_sector,
    neutralize_by_size,
    neutralize_by_both,
    neutralize_blend,
)


# ── Helpers ──────────────────────────────────────────────────────────

def make_panel(n_dates: int = 5, n_stocks: int = 10) -> pd.DataFrame:
    """Build a synthetic panel for testing."""
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    symbols = [f"s{i:03d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    return idx


# ── UC1-1 test: test_neutralize_ols_basic ──────────────────────────────

def test_neutralize_ols_basic():
    """Regress known linear y = 2*x + noise; residual should be uncorrelated with x."""
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    noise = np.random.randn(n) * 0.1
    y = 2.0 * x + noise

    factor = pd.Series(y, index=[f"s{i:03d}" for i in range(n)])
    controls = pd.DataFrame({"x": x}, index=factor.index)

    residual = neutralize_ols_residual(factor, controls, min_obs=10)

    assert len(residual) == len(factor)
    # Residual should be uncorrelated with x (|corr| < 0.05)
    corr = residual.corr(pd.Series(x, index=factor.index))
    assert abs(corr) < 0.05, f"Residual x-correlation {corr:.4f} > 0.05"
    # Residual mean ≈ 0 (since OLS includes constant)
    assert abs(residual.mean()) < 0.01


# ── UC1-1 test: test_neutralize_ols_min_obs ────────────────────────────

def test_neutralize_ols_min_obs():
    """When observations < min_obs, return original factor unchanged."""
    n = 5  # less than min_obs=10
    factor = pd.Series(np.random.randn(n), index=[f"s{i:03d}" for i in range(n)])
    controls = pd.DataFrame({"x": np.random.randn(n)}, index=factor.index)

    residual = neutralize_ols_residual(factor, controls, min_obs=10)

    # Should return original values (not NaN, not modified)
    pd.testing.assert_series_equal(residual, factor)


# ── UC1-1 test: test_neutralize_ols_multiindex ───────────────────────────

def test_neutralize_ols_multiindex():
    """Input MultiIndex [date, symbol] → output index fully aligned."""
    idx = make_panel(n_dates=3, n_stocks=10)
    np.random.seed(42)
    factor = pd.Series(np.random.randn(len(idx)), index=idx)
    controls = pd.DataFrame(
        {"x1": np.random.randn(len(idx)), "x2": np.random.randn(len(idx))},
        index=idx,
    )

    residual = neutralize_ols_residual(factor, controls, min_obs=5)

    assert residual.index.equals(factor.index)
    assert len(residual) == len(factor)
    # All dates should have residuals (each has 10 stocks >= min_obs=5)
    assert residual.notna().sum() > 0


# ── UC1-1 test: test_neutralize_ols_industry ─────────────────────────────

def test_neutralize_ols_industry():
    """After sector neutralization, each industry mean ≈ 0."""
    np.random.seed(42)
    n = 60
    symbols = [f"s{i:03d}" for i in range(n)]
    # 3 industries, 20 stocks each
    industries = ["电子"] * 20 + ["银行"] * 20 + ["医药"] * 20
    sector = pd.Series(industries, index=symbols)

    # Factor with strong industry bias: 电子=+5, 银行=-3, 医药=0
    base = np.random.randn(n) * 0.1
    for i in range(20):
        base[i] += 5.0      # 电子
    for i in range(20, 40):
        base[i] -= 3.0      # 银行
    factor = pd.Series(base, index=symbols)

    residual = neutralize_by_sector(factor, sector, min_sector_size=3)

    # After neutralization, each industry mean ≈ 0 (±0.01 tolerance)
    for ind in ["电子", "银行", "医药"]:
        ind_mean = residual[sector == ind].mean()
        assert abs(ind_mean) < 0.01, f"Industry {ind} mean = {ind_mean:.4f}"


# ── UC1-1 test: test_neutralize_ols_preserve_rank ────────────────────────

def test_neutralize_ols_preserve_rank():
    """After neutralization, Spearman rank correlation > 0.7 with original."""
    np.random.seed(42)
    n = 100
    symbols = [f"s{i:03d}" for i in range(n)]
    factor = pd.Series(np.random.randn(n), index=symbols)
    size = pd.Series(np.random.randn(n) * 2 + 10, index=symbols)  # log mcap proxy

    residual = neutralize_by_size(factor, size)

    rank_corr = factor.corr(residual, method="spearman")
    assert rank_corr > 0.7, f"Rank correlation {rank_corr:.4f} < 0.7"


# ── Additional tests ───────────────────────────────────────────────────

def test_neutralize_by_size_linear():
    """Factor = 2 * log_mcap + noise → residual ≈ noise (mean ≈ 0)."""
    np.random.seed(42)
    n = 50
    symbols = [f"s{i:03d}" for i in range(n)]
    log_mcap = pd.Series(np.random.randn(n) * 2 + 10, index=symbols)
    noise = np.random.randn(n) * 0.5
    factor = 2.0 * log_mcap + noise

    residual = neutralize_by_size(factor, log_mcap)

    assert abs(residual.mean()) < 0.1
    # Residual should have near-zero correlation with log_mcap
    corr = residual.corr(log_mcap)
    assert abs(corr) < 0.1


def test_neutralize_by_both():
    """Sequential neutralization works on panel data."""
    idx = make_panel(n_dates=2, n_stocks=20)
    np.random.seed(42)
    factor = pd.Series(np.random.randn(len(idx)), index=idx)

    # Build sector (same per date)
    sectors = ["A"] * 7 + ["B"] * 7 + ["C"] * 6
    sector_map = pd.Series(sectors * 2, index=idx)  # 2 dates
    log_mcap = pd.Series(np.random.randn(len(idx)) * 2 + 10, index=idx)

    residual = neutralize_by_both(factor, sector_map, log_mcap, min_sector_size=3)

    assert residual.index.equals(factor.index)
    assert not residual.isna().all()


def test_neutralize_blend():
    """Blend strength=0.5 preserves 50% of original factor."""
    np.random.seed(42)
    n = 50
    symbols = [f"s{i:03d}" for i in range(n)]
    factor = pd.Series(np.random.randn(n), index=symbols)
    controls = pd.DataFrame({"x": np.random.randn(n)}, index=symbols)

    residual = neutralize_ols_residual(factor, controls)
    blended = neutralize_blend(factor, controls, strength=0.5)

    expected = 0.5 * factor + 0.5 * residual
    pd.testing.assert_series_equal(blended, expected, check_names=False)


def test_neutralize_empty_controls():
    """Empty controls DataFrame should return original factor."""
    factor = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    controls = pd.DataFrame(index=factor.index)

    residual = neutralize_ols_residual(factor, controls)
    pd.testing.assert_series_equal(residual, factor)


def test_neutralize_no_common_index():
    """No common index between factor and controls → return NaN series."""
    factor = pd.Series([1.0, 2.0], index=["a", "b"])
    controls = pd.DataFrame({"x": [1.0, 2.0]}, index=["c", "d"])

    residual = neutralize_ols_residual(factor, controls)
    assert residual.isna().all()


def test_neutralize_constant_controls():
    """Controls with zero variance → return original factor (can't regress)."""
    factor = pd.Series(np.random.randn(20), index=[f"s{i:03d}" for i in range(20)])
    controls = pd.DataFrame({"x": [1.0] * 20}, index=factor.index)  # zero variance

    residual = neutralize_ols_residual(factor, controls)
    pd.testing.assert_series_equal(residual, factor)
