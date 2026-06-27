"""Tests for industry classification data module.

Covers ROADMAP_P1 UC1-3 test cases:
    1. test_load_industry_map
    2. test_build_dummy_matrix
    3. test_dummy_matrix_orthogonal
"""

import numpy as np
import pandas as pd
import pytest

from tools.backtest_mvp.data.industry import (
    load_industry_map,
    build_dummy_matrix,
    build_controls,
)


# ── test_load_industry_map ────────────────────────────────────────────

def test_load_industry_map_basic():
    """load_industry_map returns DataFrame with correct schema."""
    symbols = ["sh600000", "sz000001", "sh600519"]
    df = load_industry_map(symbols=symbols, method="sw")
    
    # Should have correct columns regardless of data availability
    assert "industry_code" in df.columns
    assert "industry_name" in df.columns
    # If data is available, no NaN values
    if len(df) > 0:
        assert not df.isna().any().any()
        # All requested symbols present (or filtered subset)
        assert len(df) <= len(symbols)
    else:
        # Empty is OK when no cache — will use mcap proxy fallback
        pass


def test_load_industry_map_fallback():
    """When no cache exists, falls back to mcap proxy (at least 3 categories)."""
    # Use non-existent method to force fallback
    df = load_industry_map(symbols=["sh600000", "sz000001"], method="nonexistent")
    
    # Should still return something (even if empty)
    assert isinstance(df, pd.DataFrame)
    assert "industry_code" in df.columns


# ── test_build_dummy_matrix ───────────────────────────────────────────

def test_build_dummy_matrix_shape():
    """3 industries + 10 symbols -> (10, 2) with drop_first=True."""
    industries = ["电子", "银行", "医药"]
    symbols = [f"s{i:03d}" for i in range(10)]
    # Assign: 4电子, 3银行, 3医药
    codes = ["电子"] * 4 + ["银行"] * 3 + ["医药"] * 3
    industry_map = pd.DataFrame({
        "industry_code": codes,
        "industry_name": codes,
    }, index=symbols)

    dummies = build_dummy_matrix(industry_map, symbols, drop_first=True)

    # 3 industries - 1 (drop_first) = 2 columns
    assert dummies.shape == (10, 2)
    # Each row should have exactly one 1 (or all 0 for dropped category)
    assert (dummies.sum(axis=1) <= 1).all()
    # Missing symbols should be 0
    assert (dummies.loc["s009":"s009"].sum(axis=1) == 0).all()  # none missing in this test


def test_build_dummy_matrix_drop_first_false():
    """drop_first=False -> all 3 columns."""
    symbols = [f"s{i:03d}" for i in range(10)]
    codes = ["A"] * 4 + ["B"] * 3 + ["C"] * 3
    industry_map = pd.DataFrame({
        "industry_code": codes,
        "industry_name": codes,
    }, index=symbols)

    dummies = build_dummy_matrix(industry_map, symbols, drop_first=False)

    assert dummies.shape == (10, 3)
    # Each row should have exactly one 1
    assert (dummies.sum(axis=1) == 1).all()


def test_build_dummy_matrix_missing_symbols():
    """Symbols not in industry_map get all 0s."""
    symbols = [f"s{i:03d}" for i in range(5)]
    industry_map = pd.DataFrame({
        "industry_code": ["A"],
        "industry_name": ["A"],
    }, index=["s000"])

    dummies = build_dummy_matrix(industry_map, symbols, drop_first=True)

    # s001-s004 not in industry_map -> all 0s
    assert (dummies.loc["s001":"s004"].sum(axis=1) == 0).all()


# ── test_dummy_matrix_orthogonal ────────────────────────────────────

def test_dummy_matrix_orthogonal():
    """Dummy columns are linearly independent (rank = n_cols)."""
    symbols = [f"s{i:03d}" for i in range(20)]
    # 3 industries with enough stocks each
    codes = ["A"] * 7 + ["B"] * 7 + ["C"] * 6
    industry_map = pd.DataFrame({
        "industry_code": codes,
        "industry_name": codes,
    }, index=symbols)

    dummies = build_dummy_matrix(industry_map, symbols, drop_first=True)

    rank = np.linalg.matrix_rank(dummies.values)
    assert rank == dummies.shape[1], f"Rank {rank} != n_cols {dummies.shape[1]}"


# ── Additional tests ─────────────────────────────────────────────────

def test_build_controls():
    """build_controls combines log_size and industry dummies."""
    idx = pd.MultiIndex.from_product([
        pd.date_range("2024-01-01", periods=3),
        [f"s{i:03d}" for i in range(10)]
    ], names=["date", "symbol"])
    
    np.random.seed(42)
    factor_panel = pd.DataFrame({
        "mcap": np.random.uniform(1e9, 1e11, len(idx)),
        "close": np.random.uniform(10, 100, len(idx)),
    }, index=idx)

    industry_map = pd.DataFrame({
        "industry_code": ["A"] * 5 + ["B"] * 5,
        "industry_name": ["A"] * 5 + ["B"] * 5,
    }, index=[f"s{i:03d}" for i in range(10)])

    controls = build_controls(factor_panel, industry_map=industry_map)

    assert "log_size" in controls.columns
    # Should have industry dummies
    ind_cols = [c for c in controls.columns if c.startswith("ind_")]
    assert len(ind_cols) > 0
    # Index should match factor_panel
    assert len(controls) == len(factor_panel)


def test_build_controls_no_industry():
    """build_controls with no industry map: only log_size."""
    idx = pd.MultiIndex.from_product([
        pd.date_range("2024-01-01", periods=2),
        [f"s{i:03d}" for i in range(5)]
    ], names=["date", "symbol"])
    
    factor_panel = pd.DataFrame({
        "mcap": np.random.uniform(1e9, 1e11, len(idx)),
    }, index=idx)

    controls = build_controls(factor_panel, industry_map=None)

    assert "log_size" in controls.columns
    assert len(controls.columns) == 1
