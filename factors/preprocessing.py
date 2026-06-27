"""Factor preprocessing pipeline — winsorization, standardization, neutralization.

Chain:  winsorize → zscore → neutralize → re-zscore

This is the standard institutional factor-processing pipeline used by
Barra, MSCI, and major quant funds.  Each step operates cross-sectionally
by date for panel data.

References:
    - ziyan916/multi-factor-quant: src/pipeline/{outliers,standardization,processor}.py
    - WorldQuant BRAIN factor formula documentation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .operators import _has_level, _group_apply
from .neutralization import neutralize_ols_residual, neutralize_blend


# ═══════════════════════════════════════════════════════════════════════
# Winsorization (outlier handling)
# ═══════════════════════════════════════════════════════════════════════

def winsorize_mad(series: pd.Series, n_mad: float = 5.0, level: str | int = "date") -> pd.Series:
    """MAD winsorization — robust to extreme outliers.

    median = median(x)
    MAD = median(|x - median|)
    Clip to [median - n_mad * MAD, median + n_mad * MAD]

    Parameters
    ----------
    series : pd.Series
        Factor values.  MultiIndex [date, symbol] or single index.
    n_mad : float, default 5.0
        Number of MADs for clipping bounds (3–5 is standard).
    level : str or int, default "date"
        Groupby level for cross-sectional operation.

    Returns
    -------
    pd.Series
        Winsorized values with same index.
    """
    def _mad(s: pd.Series) -> pd.Series:
        if len(s.dropna()) < 5:
            return s
        median = s.median()
        mad = np.median(np.abs(s - median))
        if mad == 0:
            return s
        upper = median + n_mad * mad
        lower = median - n_mad * mad
        return s.clip(lower, upper)

    if level is not None and _has_level(series.index, level):
        return series.groupby(level=level, group_keys=False).apply(_mad)
    return _mad(series)


def winsorize_percentile(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
    level: str | int = "date",
) -> pd.Series:
    """Percentile winsorization — clip extreme tails.

    Parameters
    ----------
    series : pd.Series
    lower : float, default 0.01
        Lower percentile (0.01 = bottom 1%).
    upper : float, default 0.99
        Upper percentile (0.99 = top 1%).
    level : str or int, default "date"

    Returns
    -------
    pd.Series
    """
    def _pct(s: pd.Series) -> pd.Series:
        if len(s.dropna()) < 10:
            return s
        lo = s.quantile(lower)
        hi = s.quantile(upper)
        return s.clip(lo, hi)

    if level is not None and _has_level(series.index, level):
        return series.groupby(level=level, group_keys=False).apply(_pct)
    return _pct(series)


def winsorize_std(series: pd.Series, n_std: float = 4.0, level: str | int = "date") -> pd.Series:
    """Standard-deviation winsorization (already exists in operators.py).

    Kept here for completeness; delegates to the existing operator.
    """
    from .operators import winsorize as _winsorize_std
    return _winsorize_std(series, std=n_std, level=level)


def winsorize_cross_sectional(
    series: pd.Series,
    method: str = "mad",
    n_mad: float = 5.0,
    lower: float = 0.01,
    upper: float = 0.99,
    level: str | int = "date",
) -> pd.Series:
    """Unified winsorization entry point.

    Parameters
    ----------
    method : str, default "mad"
        "mad" | "percentile" | "std"
    n_mad : float, default 5.0
        For "mad" method.
    lower, upper : float
        For "percentile" method.
    level : str or int, default "date"

    Returns
    -------
    pd.Series
    """
    if method == "mad":
        return winsorize_mad(series, n_mad=n_mad, level=level)
    elif method == "percentile":
        return winsorize_percentile(series, lower=lower, upper=upper, level=level)
    elif method == "std":
        return winsorize_std(series, n_std=n_mad, level=level)
    else:
        raise ValueError(f"Unknown winsorize method: {method}")


# ═══════════════════════════════════════════════════════════════════════
# Standardization
# ═══════════════════════════════════════════════════════════════════════

def zscore_cross_sectional(series: pd.Series, level: str | int = "date") -> pd.Series:
    """Z-score standardization: z = (x - μ) / σ, per cross-section.

    Parameters
    ----------
    series : pd.Series
    level : str or int, default "date"

    Returns
    -------
    pd.Series
        Standardized values with mean≈0, std≈1 per cross-section.
    """
    def _zscore(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        mu = s.mean()
        sigma = s.std(ddof=1)  # sample std
        if sigma == 0 or pd.isna(sigma):
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sigma

    if level is not None and _has_level(series.index, level):
        return series.groupby(level=level, group_keys=False).apply(_zscore)
    return _zscore(series)


def rank_normalize(series: pd.Series, level: str | int = "date") -> pd.Series:
    """Rank normalization: map to normal distribution via inverse CDF.

    Advantages over z-score:
    - Completely immune to outliers (uses ranks, not values).
    - Works with any distribution shape.

    Disadvantages:
    - Loses distance information (PE=5 vs PE=500 become equally spaced).

    Parameters
    ----------
    series : pd.Series
    level : str or int, default "date"

    Returns
    -------
    pd.Series
        Values mapped to standard normal distribution via ppf.
    """
    def _rank_norm(s: pd.Series) -> pd.Series:
        ranked = s.rank(pct=True)
        # Clip to (0,1) to avoid ±inf at boundaries
        ranked = ranked.clip(0.001, 0.999)
        result = pd.Series(stats.norm.ppf(ranked), index=s.index)
        return result.fillna(0.0)

    if level is not None and _has_level(series.index, level):
        return series.groupby(level=level, group_keys=False).apply(_rank_norm)
    return _rank_norm(series)


# ═══════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════

def preprocess_pipeline(
    factor: pd.Series,
    controls: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.Series:
    """Complete factor-preprocessing pipeline.

    Steps:
        1. Winsorize (cross-sectional, default MAD 5×)
        2. Z-score standardization (cross-sectional)
        3. Neutralize (OLS residual on controls, optional)
        4. Re-standardize (cross-sectional, post-neutralization)

    Parameters
    ----------
    factor : pd.Series
        Raw factor values (MultiIndex [date, symbol] or single index).
    controls : pd.DataFrame, optional
        Control variables for neutralization.  If None, skip neutralization.
    config : dict, optional
        Pipeline configuration.  Defaults:
        {
            "winsorize": {"method": "mad", "n_mad": 5.0},
            "standardize": {"method": "zscore"},  # "zscore" | "rank"
            "neutralize": {"strength": 0.5, "min_obs": 10},
            "restandardize": True,
        }

    Returns
    -------
    pd.Series
        Clean, standardized, optionally neutralized factor.

    Notes
    -----
    The ``strength=0.5`` default is a key design choice: it removes 50% of
    the control exposure while preserving the original rank information.
    This avoids the "de-activation" problem where full neutralization
    destroys the factor's predictive power.
    """
    cfg = {
        "winsorize": {"method": "mad", "n_mad": 5.0, "level": "date"},
        "standardize": {"method": "zscore", "level": "date"},
        "neutralize": {"strength": 0.5, "min_obs": 10, "enabled": True},
        "restandardize": True,
    }
    if config:
        cfg.update(config)

    result = factor.copy()

    # Step 1: Winsorize
    w_cfg = cfg["winsorize"]
    result = winsorize_cross_sectional(
        result,
        method=w_cfg.get("method", "mad"),
        n_mad=w_cfg.get("n_mad", 5.0),
        level=w_cfg.get("level", "date"),
    )

    # Step 2: Standardize
    s_cfg = cfg["standardize"]
    level = s_cfg.get("level", "date")
    if s_cfg.get("method", "zscore") == "zscore":
        result = zscore_cross_sectional(result, level=level)
    else:
        result = rank_normalize(result, level=level)

    # Step 3: Neutralize (optional)
    n_cfg = cfg.get("neutralize", {})
    if n_cfg.get("enabled", True) and controls is not None and not controls.empty:
        strength = n_cfg.get("strength", 0.5)
        min_obs = n_cfg.get("min_obs", 10)
        if strength >= 1.0:
            result = neutralize_ols_residual(result, controls, min_obs=min_obs)
        else:
            result = neutralize_blend(result, controls, strength=strength, min_obs=min_obs)
        # Fill NaN residuals with 0 (no alpha bias)
        result = result.fillna(0.0)

    # Step 4: Re-standardize (post-neutralization)
    if cfg.get("restandardize", True):
        result = zscore_cross_sectional(result, level=level)

    return result
