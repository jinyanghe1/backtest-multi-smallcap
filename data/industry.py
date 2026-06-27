"""Industry classification data for A-share stocks.

Supports loading industry maps from local cache or generating a proxy
from market-cap quantiles when no industry data is available.

Industry systems:
    - SW  (申万行业, 3 levels: 一级/二级/三级)
    - CITICS  (中信行业)
    - ZZ  (中证行业)

References:
    - ziyan916/multi-factor-quant: src/data/universe.py
    - PuYuan-scott/quant-multifactor: src/industry_map.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from tools.backtest_mvp.data import DATA_DIR


def _get_industry_cache_path(method: str = "sw") -> Path:
    """Return the path for cached industry data."""
    return DATA_DIR / f"industry_map_{method}.csv"


def load_industry_map(
    symbols: list[str] | None = None,
    method: str = "sw",
    level: int = 1,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """Load industry classification for A-share stocks.

    Parameters
    ----------
    symbols : list of str, optional
        Stock symbols to filter (e.g. ["sh600000", "sz000001"]).
        If None, load all available.
    method : str, default "sw"
        Industry classification system: "sw" | "citics" | "zz".
    level : int, default 1
        Industry level (1=一级, 2=二级, 3=三级).  Only applicable for SW.
    data_dir : str, optional
        Override default data directory.

    Returns
    -------
    pd.DataFrame
        Columns: [symbol, industry_code, industry_name]
        Index: symbol

    Fallback
    --------
    If no local cache exists and no industry data is available, falls back
    to a market-cap-based proxy: 大盘股/中盘股/小盘股 as pseudo-industries.
    """
    cache_dir = Path(data_dir) if data_dir else DATA_DIR
    cache_path = cache_dir / f"industry_map_{method}_l{level}.csv"

    # Try to load from cache
    if cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"symbol": str, "industry_code": str, "industry_name": str})
        df = df.set_index("symbol")
        if symbols:
            df = df[df.index.isin(symbols)]
        return df

    # Try to load a generic industry map
    generic_cache = cache_dir / f"industry_map_{method}.csv"
    if generic_cache.exists():
        df = pd.read_csv(generic_cache, dtype={"symbol": str})
        df = df.set_index("symbol")
        # If level is specified, use the appropriate column
        if level > 1 and f"industry_code_l{level}" in df.columns:
            df = df.rename(columns={
                f"industry_code_l{level}": "industry_code",
                f"industry_name_l{level}": "industry_name",
            })
        if symbols:
            df = df[df.index.isin(symbols)]
        return df

    # No industry data available at all — fallback to mcap proxy
    return _build_mcap_proxy_industry(symbols, cache_dir)


def _build_mcap_proxy_industry(symbols: list[str] | None, cache_dir: Path) -> pd.DataFrame:
    """Build pseudo-industry from market-cap quantiles when no real industry data.

    Uses latest available mcap data to split stocks into 3 quantiles:
    - 大盘股 (large): top 33% by mcap
    - 中盘股 (mid): middle 33%
    - 小盘股 (small): bottom 33%

    This is a crude but effective fallback for size-neutralization.
    """
    # Try to load mcap from cached data
    mcap_data = []
    if symbols:
        for sym in symbols:
            parquet_path = cache_dir / f"{sym}.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                if "mcap" in df.columns or "market_cap" in df.columns:
                    col = "mcap" if "mcap" in df.columns else "market_cap"
                    latest = df[col].dropna().iloc[-1] if not df[col].dropna().empty else np.nan
                    mcap_data.append({"symbol": sym, "mcap": latest})
    else:
        # Load all cached parquet files for mcap
        for f in cache_dir.glob("*.parquet"):
            if f.name.startswith("adv_") or f.name.startswith("industry_"):
                continue
            try:
                df = pd.read_parquet(f)
                if "mcap" in df.columns or "market_cap" in df.columns:
                    col = "mcap" if "mcap" in df.columns else "market_cap"
                    sym = f.stem
                    latest = df[col].dropna().iloc[-1] if not df[col].dropna().empty else np.nan
                    mcap_data.append({"symbol": sym, "mcap": latest})
            except Exception:
                continue

    if not mcap_data:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=["industry_code", "industry_name"]).astype({
            "industry_code": str, "industry_name": str
        })

    mcap_df = pd.DataFrame(mcap_data).set_index("symbol")
    mcap_df = mcap_df.dropna()

    # Quantile-based pseudo-industries
    mcap_df["q"] = pd.qcut(mcap_df["mcap"], q=3, labels=["small", "mid", "large"])

    result = pd.DataFrame({
        "industry_code": mcap_df["q"].astype(str),
        "industry_name": mcap_df["q"].map({
            "small": "小盘股",
            "mid": "中盘股",
            "large": "大盘股",
        }).astype(str),
    })
    return result


def build_dummy_matrix(
    industry_map: pd.DataFrame,
    symbols: list[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """Build industry dummy variables for OLS regression.

    Parameters
    ----------
    industry_map : pd.DataFrame
        DataFrame with index=symbol, columns=[industry_code, ...].
    symbols : list of str
        Symbols to include in the dummy matrix.
    drop_first : bool, default True
        Drop first category to avoid collinearity (standard in OLS).

    Returns
    -------
    pd.DataFrame
        Dummy matrix with index=symbol, columns=ind_xxx.
        Excludes symbols not in industry_map.

    Example
    -------
    >>> industry_map = load_industry_map(symbols=["sh600000", "sz000001"])
    >>> dummies = build_dummy_matrix(industry_map, ["sh600000", "sz000001"])
    >>> dummies.shape
    (2, n_industries - 1)  # drop_first=True
    """
    # Filter to requested symbols
    filtered = industry_map[industry_map.index.isin(symbols)]
    if filtered.empty:
        return pd.DataFrame(index=symbols)

    codes = filtered["industry_code"].astype(str)

    # Get dummies
    dummies = pd.get_dummies(codes, prefix="ind", drop_first=drop_first)
    dummies.index = filtered.index

    # Reindex to include all requested symbols (missing = 0)
    dummies = dummies.reindex(symbols, fill_value=0.0)

    return dummies


def build_controls(
    factor_panel: pd.DataFrame,
    industry_map: pd.DataFrame | None = None,
    size_col: str = "mcap",
) -> pd.DataFrame:
    """Build control variables DataFrame for neutralization.

    Combines:
    - log(size) as a continuous control
    - industry dummies as categorical controls

    Parameters
    ----------
    factor_panel : pd.DataFrame
        Factor panel with MultiIndex [date, symbol] or columns including size_col.
    industry_map : pd.DataFrame, optional
        Industry classification. If None, only size control is used.
    size_col : str, default "mcap"
        Column name for market cap in factor_panel.

    Returns
    -----
    pd.DataFrame
        Control variables with same index as factor_panel.
        Columns: [log_size, ind_xxx, ...]
    """
    # Extract size
    if size_col in factor_panel.columns:
        size = factor_panel[size_col].copy()
    elif "close" in factor_panel.columns and "shares" in factor_panel.columns:
        # Compute mcap from close * shares
        size = factor_panel["close"] * factor_panel["shares"]
    else:
        # Cannot build size control
        size = pd.Series(np.nan, index=factor_panel.index)

    log_size = np.log(size.replace(0, np.nan).replace(-np.inf, np.nan))
    log_size = log_size.rename("log_size")

    controls = pd.DataFrame({"log_size": log_size})

    # Add industry dummies if available
    if industry_map is not None and not industry_map.empty:
        # For panel data, we need industry dummies per date
        # This is a simplification: industry is static (doesn't change per date)
        if isinstance(factor_panel.index, pd.MultiIndex) and "symbol" in factor_panel.index.names:
            symbols = factor_panel.index.get_level_values("symbol").unique()
        elif isinstance(factor_panel.index, pd.MultiIndex) and len(factor_panel.index.levels) == 2:
            symbols = factor_panel.index.get_level_values(1).unique()
        else:
            symbols = factor_panel.index.tolist()

        dummies = build_dummy_matrix(industry_map, symbols.tolist() if hasattr(symbols, 'tolist') else list(symbols))

        # Merge dummies with controls by symbol
        if isinstance(factor_panel.index, pd.MultiIndex):
            # For each date, attach the same industry dummies
            dummies_panel = pd.DataFrame(index=factor_panel.index)
            for col in dummies.columns:
                dummies_panel[col] = dummies.reindex(
                    factor_panel.index.get_level_values(1 if len(factor_panel.index.levels) == 2 else 0),
                    fill_value=0.0
                ).values
            controls = pd.concat([controls, dummies_panel], axis=1)
        else:
            controls = pd.concat([controls, dummies], axis=1)

    return controls


# ── Convenience exports ───────────────────────────────────────────────

__all__ = [
    "load_industry_map",
    "build_dummy_matrix",
    "build_controls",
]
