"""Panel-safe factor operators in a WorldQuant-style vocabulary."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _has_level(index: pd.Index, level: str | int) -> bool:
    return isinstance(index, pd.MultiIndex) and (
        isinstance(level, int) or level in index.names
    )


def _group_apply(series: pd.Series, group_level: str | int, func) -> pd.Series:
    if _has_level(series.index, group_level):
        return series.groupby(level=group_level, group_keys=False).apply(func)
    return func(series)


def ts_rank(
    series: pd.Series,
    window: int,
    group_level: str | int = "symbol",
    min_periods: int | None = None,
) -> pd.Series:
    """Rank the latest value within each rolling time-series window."""
    min_periods = window if min_periods is None else min_periods

    def _calc(s: pd.Series) -> pd.Series:
        return s.rolling(window, min_periods=min_periods).apply(
            lambda values: pd.Series(values).rank(pct=True).iloc[-1],
            raw=False,
        )

    return _group_apply(series, group_level, _calc)


def delay(series: pd.Series, periods: int, group_level: str | int = "symbol") -> pd.Series:
    """Lag a series by periods, grouped by symbol for panel data."""
    return _group_apply(series, group_level, lambda s: s.shift(periods))


def delta(series: pd.Series, periods: int, group_level: str | int = "symbol") -> pd.Series:
    """Difference a series by periods, grouped by symbol for panel data."""
    return _group_apply(series, group_level, lambda s: s.diff(periods))


def decay_linear(
    series: pd.Series,
    window: int,
    group_level: str | int = "symbol",
    min_periods: int | None = None,
) -> pd.Series:
    """Apply linearly increasing weights inside each rolling window."""
    min_periods = window if min_periods is None else min_periods

    def _weighted(values) -> float:
        arr = np.asarray(values, dtype=float)
        if np.isnan(arr).any():
            return np.nan
        weights = np.arange(1, len(arr) + 1, dtype=float)
        return float(np.dot(arr, weights) / weights.sum())

    return _group_apply(
        series,
        group_level,
        lambda s: s.rolling(window, min_periods=min_periods).apply(_weighted, raw=True),
    )


def correlation(
    x: pd.Series,
    y: pd.Series,
    window: int,
    group_level: str | int = "symbol",
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling correlation between two aligned series."""
    min_periods = window if min_periods is None else min_periods
    x_aligned, y_aligned = x.align(y, join="inner")
    if _has_level(x_aligned.index, group_level):
        df = pd.DataFrame({"x": x_aligned, "y": y_aligned})

        def _calc(g: pd.DataFrame) -> pd.Series:
            return g["x"].rolling(window, min_periods=min_periods).corr(g["y"])

        return df.groupby(level=group_level, group_keys=False).apply(_calc)
    return x_aligned.rolling(window, min_periods=min_periods).corr(y_aligned)


def rank(series: pd.Series, level: str | int = "date") -> pd.Series:
    """Percentile rank cross-sectionally by date for panel data."""
    if _has_level(series.index, level):
        return series.groupby(level=level, group_keys=False).rank(pct=True)
    return series.rank(pct=True)


def group_rank(series: pd.Series, group: pd.Series, pct: bool = True) -> pd.Series:
    """Rank values within a group, and within date for MultiIndex panels."""
    aligned_value, aligned_group = series.align(group, join="left")
    df = pd.DataFrame({"value": aligned_value, "group": aligned_group})
    if _has_level(df.index, "date"):
        ranked = df.groupby([df.index.get_level_values("date"), "group"], dropna=False)["value"].rank(pct=pct)
    elif isinstance(df.index, pd.MultiIndex):
        ranked = df.groupby([df.index.get_level_values(0), "group"], dropna=False)["value"].rank(pct=pct)
    else:
        ranked = df.groupby("group", dropna=False)["value"].rank(pct=pct)
    ranked.index = df.index
    return ranked


def scale(series: pd.Series, level: str | int | None = "date") -> pd.Series:
    """Scale values by sum(abs(x)), preserving sign."""
    def _scale(s: pd.Series) -> pd.Series:
        denom = s.abs().sum()
        if denom == 0 or pd.isna(denom):
            return pd.Series(np.nan, index=s.index)
        return s / denom

    if level is not None and _has_level(series.index, level):
        return series.groupby(level=level, group_keys=False).apply(_scale)
    return _scale(series)


def ts_argmax(
    series: pd.Series,
    window: int,
    group_level: str | int = "symbol",
    min_periods: int | None = None,
) -> pd.Series:
    """0-based position of the maximum value within each rolling window."""
    min_periods = window if min_periods is None else min_periods

    def _argmax(values) -> float:
        arr = np.asarray(values, dtype=float)
        if np.isnan(arr).all():
            return np.nan
        return float(np.nanargmax(arr))

    return _group_apply(series, group_level, lambda s: s.rolling(window, min_periods=min_periods).apply(_argmax, raw=True))


def ts_argmin(
    series: pd.Series,
    window: int,
    group_level: str | int = "symbol",
    min_periods: int | None = None,
) -> pd.Series:
    """0-based position of the minimum value within each rolling window."""
    min_periods = window if min_periods is None else min_periods

    def _argmin(values) -> float:
        arr = np.asarray(values, dtype=float)
        if np.isnan(arr).all():
            return np.nan
        return float(np.nanargmin(arr))

    return _group_apply(series, group_level, lambda s: s.rolling(window, min_periods=min_periods).apply(_argmin, raw=True))


def signed_power(series: pd.Series, alpha: float) -> pd.Series:
    """Raise abs(x) to alpha while preserving sign."""
    return np.sign(series) * (series.abs() ** alpha)


def winsorize(series: pd.Series, std: float = 4.0, level: str | int | None = "date") -> pd.Series:
    """Cap values beyond ±std standard deviations from the cross-sectional mean.

    Operates per-date for panel data (MultiIndex with a date level).
    Values outside [mean - std*sigma, mean + std*sigma] are clipped to the bounds.
    """
    def _winsor(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sigma = s.std()
        if sigma == 0 or pd.isna(sigma):
            return s
        lo = mu - std * sigma
        hi = mu + std * sigma
        return s.clip(lower=lo, upper=hi)

    if level is not None and _has_level(series.index, level):
        return series.groupby(level=level, group_keys=False).apply(_winsor)
    return _winsor(series)


def ts_decay_exp(
    series: pd.Series,
    window: int,
    factor: float = 0.5,
    group_level: str | int = "symbol",
    min_periods: int | None = None,
) -> pd.Series:
    """Exponentially weighted decay within each rolling window.

    Weights follow a geometric progression with ratio `factor`
    (most recent value gets weight factor^0 = 1, previous gets factor^1, etc.).
    """
    min_periods = window if min_periods is None else min_periods
    alpha = 1.0 - factor  # pandas ewm alpha parameter

    def _calc(s: pd.Series) -> pd.Series:
        return s.ewm(alpha=alpha, min_periods=min_periods, adjust=False).mean()

    return _group_apply(series, group_level, _calc)


def ts_std_dev(
    series: pd.Series,
    window: int,
    group_level: str | int = "symbol",
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling standard deviation within each group."""
    min_periods = window if min_periods is None else min_periods
    return _group_apply(
        series, group_level,
        lambda s: s.rolling(window, min_periods=min_periods).std(),
    )


def ts_quantile(
    series: pd.Series,
    window: int,
    quantile: float = 0.5,
    group_level: str | int = "symbol",
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling quantile within each group."""
    min_periods = window if min_periods is None else min_periods
    return _group_apply(
        series, group_level,
        lambda s: s.rolling(window, min_periods=min_periods).quantile(quantile),
    )

