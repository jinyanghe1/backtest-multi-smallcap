"""Factor neutralization — OLS regression residuals to remove unwanted exposures.

Implements the industry-standard pipeline:
    factor = β₀ + β₁·controls + residual
    → use residual as the "pure" alpha signal

References:
    - ziyan916/multi-factor-quant: src/pipeline/neutralization.py
    - PuYuan-scott/quant-multifactor: src/neutralization.py
    - Barra USE4 / CNE5 risk model methodology
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .operators import _has_level, _group_apply


def neutralize_ols_residual(
    factor: pd.Series,
    controls: pd.DataFrame,
    min_obs: int = 10,
) -> pd.Series:
    """Cross-sectional OLS neutralization (date-by-date).

    For each trading day, regress the factor on the control variables and
    return the residual.  The residual represents the factor with the
    control exposures stripped out.

    Parameters
    ----------
    factor : pd.Series
        Raw factor values.  Index may be:
        - MultiIndex [date, symbol]  →  groupby date
        - single Index [symbol]      →  single cross-section
    controls : pd.DataFrame
        Control variables (e.g. log(mcap), industry dummies).
        Index must align with ``factor`` (same level names or single index).
    min_obs : int, default 10
        Minimum observations in a cross-section to run regression.
        Below this threshold the original factor is returned unchanged.

    Returns
    -------
    pd.Series
        Residuals with the same index as ``factor``.  NaN where regression
        failed or observations were insufficient.

    Algorithm
    ---------
    1. Align factor and controls on common index.
    2. If MultiIndex with ``date`` level: groupby date, run OLS per group.
    3. If single index: run single OLS.
    4. Residual = y - X @ β (where X includes controls + constant).
    5. Reindex to original factor index to preserve shape.
    """
    # Align indices --------------------------------------------------------
    common_idx = factor.index.intersection(controls.index)
    if len(common_idx) == 0:
        return pd.Series(np.nan, index=factor.index)

    y = factor.loc[common_idx].astype(float)
    X = controls.loc[common_idx].astype(float)

    # Add constant
    X = X.copy()
    if "const" not in X.columns:
        X.insert(0, "const", 1.0)

    # Determine grouping strategy ------------------------------------------
    if _has_level(y.index, "date"):
        # Panel data: groupby date
        residuals = []
        for date, group in y.groupby(level="date"):
            if len(group) < min_obs:
                residuals.append(group)
                continue

            x_date = X.loc[group.index]
            # Drop columns with zero variance (except constant)
            var_mask = x_date.std() > 0
            if "const" in x_date.columns:
                var_mask["const"] = True
            x_date = x_date.loc[:, var_mask]
            if x_date.shape[1] <= 1:  # only constant left
                residuals.append(group)
                continue

            try:
                beta = np.linalg.lstsq(x_date.values, group.values, rcond=None)[0]
                predicted = x_date.values @ beta
                resid = pd.Series(group.values - predicted, index=group.index)
                residuals.append(resid)
            except (np.linalg.LinAlgError, ValueError):
                residuals.append(group)

        result = pd.concat(residuals)
    else:
        # Single cross-section
        if len(y) < min_obs:
            return y.reindex(factor.index)

        # Drop columns with zero variance (except constant)
        var_mask = X.std() > 0
        if "const" in X.columns:
            var_mask["const"] = True
        X = X.loc[:, var_mask]
        if X.shape[1] <= 1:
            return y.reindex(factor.index)

        try:
            beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
            predicted = X.values @ beta
            result = pd.Series(y.values - predicted, index=y.index)
        except (np.linalg.LinAlgError, ValueError):
            result = y

    return result.reindex(factor.index)


def neutralize_by_sector(
    factor: pd.Series,
    sector: pd.Series,
    min_sector_size: int = 3,
) -> pd.Series:
    """Sector-neutralize a factor via OLS on industry dummies.

    Parameters
    ----------
    factor : pd.Series
        Factor values (index = symbol, or MultiIndex [date, symbol]).
    sector : pd.Series
        Industry classification (same index as factor).
    min_sector_size : int, default 3
        Sectors with fewer than this many stocks are dropped from dummies.

    Returns
    -------
    pd.Series
        Residuals after regressing on sector dummies + constant.
    """
    if factor is None or len(factor.dropna()) < 10:
        return factor

    common = factor.index.intersection(sector.index)
    if len(common) < 10:
        return factor

    y = factor.loc[common].dropna().astype(float)
    sector_aligned = sector.loc[y.index]

    # Build sector dummies (drop small sectors)
    sector_counts = sector_aligned.value_counts()
    valid_sectors = sector_counts[sector_counts >= min_sector_size].index

    if len(valid_sectors) <= 1:
        return factor

    X = pd.DataFrame({"const": 1.0}, index=y.index)
    for sec in valid_sectors:
        X[f"sector_{sec}"] = (sector_aligned == sec).astype(float)

    try:
        beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
        predicted = X.values @ beta
        residuals = pd.Series(y.values - predicted, index=y.index)
    except Exception:
        return factor

    return residuals.reindex(factor.index)


def neutralize_by_size(
    factor: pd.Series,
    log_mkt_cap: pd.Series,
) -> pd.Series:
    """Size-neutralize a factor via OLS on log(market cap).

    Parameters
    ----------
    factor : pd.Series
        Factor values.
    log_mkt_cap : pd.Series
        log(market cap) values (same index as factor).

    Returns
    -------
    pd.Series
        Residuals after regressing on log_mkt_cap + constant.
    """
    if factor is None or len(factor.dropna()) < 10:
        return factor

    common = factor.index.intersection(log_mkt_cap.index)
    if len(common) < 10:
        return factor

    y = factor.loc[common].dropna().astype(float)
    x = log_mkt_cap.loc[y.index].astype(float)

    if x.std() == 0 or pd.isna(x.std()):
        return factor

    X = np.column_stack([np.ones(len(x)), x.values])

    try:
        beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
        predicted = X @ beta
        residuals = pd.Series(y.values - predicted, index=y.index)
    except Exception:
        return factor

    return residuals.reindex(factor.index)


def neutralize_by_both(
    factor: pd.Series,
    sector: pd.Series,
    log_mkt_cap: pd.Series,
    min_sector_size: int = 3,
) -> pd.Series:
    """Sequential neutralization: sector first, then size.

    This is the standard institutional approach (Barra style):
    1. Remove sector effects (largest bias first)
    2. Remove residual size effects

    Parameters
    ----------
    factor : pd.Series
        Factor values.
    sector : pd.Series
        Industry classification.
    log_mkt_cap : pd.Series
        log(market cap).
    min_sector_size : int, default 3

    Returns
    -------
    pd.Series
        Double-residuals (sector-neutralized, then size-neutralized).
    """
    factor = neutralize_by_sector(factor, sector, min_sector_size)
    factor = neutralize_by_size(factor, log_mkt_cap)
    return factor


def neutralize_blend(
    factor: pd.Series,
    controls: pd.DataFrame,
    strength: float = 0.5,
    min_obs: int = 10,
) -> pd.Series:
    """Lightweight neutralization: blend original factor with residual.

    Instead of full neutralization (which can "de-activate" a factor),
    this removes only ``strength`` fraction of the control exposure.

    Parameters
    ----------
    factor : pd.Series
        Raw factor values.
    controls : pd.DataFrame
        Control variables (same index as factor).
    strength : float, default 0.5
        0.0 = no neutralization, 1.0 = full neutralization.
    min_obs : int, default 10

    Returns
    -------
    pd.Series
        blended = (1 - strength) * factor + strength * residual
    """
    residual = neutralize_ols_residual(factor, controls, min_obs)
    # Preserve original factor where residual is NaN
    blended = factor.copy()
    mask = ~residual.isna()
    blended.loc[mask] = (1 - strength) * factor.loc[mask] + strength * residual.loc[mask]
    return blended
