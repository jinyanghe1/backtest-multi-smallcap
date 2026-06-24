"""Panel-level alpha templates built from the operator layer."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .operators import decay_linear, delta, group_rank, rank, ts_decay_exp, ts_rank, winsorize


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"factor_panel missing required columns: {missing}")


def template_fundamental_value(
    factor_panel: pd.DataFrame,
    field: str = "roe_ttm",
    window: int = 126,
    group_col: str = "sw_industry_2",
) -> pd.Series:
    """group_rank(ts_rank(field, window), group_col)."""
    _require_columns(factor_panel, [field])
    signal = ts_rank(factor_panel[field], window)
    if group_col in factor_panel.columns:
        return group_rank(signal, factor_panel[group_col])
    return rank(signal)


def template_technical_momentum(
    factor_panel: pd.DataFrame,
    price_field: str = "close",
    delta_periods: int = 1,
    ts_rank_window: int = 8,
    decay_window: int = 20,
    group_col: str = "sw_industry_1",
) -> pd.Series:
    """decay_linear(ts_rank(delta(price), ts_rank_window), decay_window), optionally neutralized."""
    _require_columns(factor_panel, [price_field])
    signal = decay_linear(ts_rank(delta(factor_panel[price_field], delta_periods), ts_rank_window), decay_window)
    if group_col in factor_panel.columns:
        return group_rank(signal, factor_panel[group_col])
    return rank(signal)


def template_multi_factor_blend(
    signals: Sequence[pd.Series],
    weights: Sequence[float] | None = None,
    group: pd.Series | None = None,
) -> pd.Series:
    if not signals:
        raise ValueError("signals must not be empty")
    ranked = [rank(signal) for signal in signals]
    if weights is None:
        weights = [1.0 / len(ranked)] * len(ranked)
    if len(weights) != len(ranked):
        raise ValueError("weights length must match signals length")
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("weights must not sum to zero")
    blended = sum(signal * weight for signal, weight in zip(ranked, weights)) / total_weight
    if group is not None:
        return group_rank(blended, group)
    return blended


GOLDEN_COMBO_SIGNALS = {
    "mcap": -1,
    "pb": -1,
    "mom20d": 1,
    "vol20d": -1,
    "max_ret": -1,
    "roe_ttm": 1,
    "revenue_growth_ttm": 1,
}


def golden_combo(factor_panel: pd.DataFrame, window: int = 20, group_col: str = "sw_industry_2") -> pd.Series:
    signals = []
    for field, direction in GOLDEN_COMBO_SIGNALS.items():
        if field in factor_panel.columns:
            signal = ts_rank(factor_panel[field], window)
            signals.append(signal if direction > 0 else -signal)
    if not signals:
        raise KeyError("factor_panel has none of the supported golden_combo fields")
    group = factor_panel[group_col] if group_col in factor_panel.columns else None
    return template_multi_factor_blend(signals, group=group)


def template_value_momentum(
    factor_panel: pd.DataFrame,
    fundamental_field: str = "roe_ttm",
    price_field: str = "close",
    fundamental_window: int = 126,
    momentum_window: int = 20,
    decay_window: int = 10,
    fundamental_weight: float = 0.6,
    group_col: str = "sw_industry_2",
) -> pd.Series:
    """Blend fundamental value with short-term momentum.

    60% fundamental (ts_rank of ROE) + 40% momentum (decay_linear of ts_rank of delta).
    Higher fundamental_weight => more stable, lower turnover.
    """
    _require_columns(factor_panel, [fundamental_field, price_field])
    value_signal = ts_rank(factor_panel[fundamental_field], fundamental_window)
    mom_signal = decay_linear(
        ts_rank(delta(factor_panel[price_field], 1), momentum_window),
        decay_window,
    )
    signals = [value_signal, mom_signal]
    weights = [fundamental_weight, 1.0 - fundamental_weight]
    group = factor_panel[group_col] if group_col in factor_panel.columns else None
    return template_multi_factor_blend(signals, weights=weights, group=group)


def template_mean_reversion(
    factor_panel: pd.DataFrame,
    price_field: str = "close",
    short_window: int = 5,
    long_window: int = 20,
    decay_window: int = 5,
    group_col: str = "sw_industry_1",
) -> pd.Series:
    """Short-term mean reversion: sell recent winners, buy recent losers.

    Signal = -ts_rank(delta(price, 1), short_window) smoothed by decay_linear.
    Reversion is stronger after sharp moves, so we use short windows.
    """
    _require_columns(factor_panel, [price_field])
    raw = ts_rank(delta(factor_panel[price_field], 1), short_window)
    signal = decay_linear(-raw, decay_window)
    if group_col in factor_panel.columns:
        return group_rank(signal, factor_panel[group_col])
    return rank(signal)


def template_fundamental_quality(
    factor_panel: pd.DataFrame,
    window: int = 126,
    group_col: str = "sw_industry_2",
) -> pd.Series:
    """Composite quality signal: ROE + gross_margin + revenue_growth.

    Equal-weight blend of three fundamental ts_rank signals, group-neutralized.
    Requires roe_ttm, gross_margin, revenue_growth_ttm columns.
    """
    available = [f for f in ["roe_ttm", "gross_margin", "revenue_growth_ttm"]
                 if f in factor_panel.columns]
    if not available:
        raise KeyError("factor_panel has none of: roe_ttm, gross_margin, revenue_growth_ttm")
    signals = [ts_rank(factor_panel[f], window) for f in available]
    group = factor_panel[group_col] if group_col in factor_panel.columns else None
    return template_multi_factor_blend(signals, group=group)


def template_low_volatility(
    factor_panel: pd.DataFrame,
    vol_field: str = "vol20d",
    window: int = 20,
    group_col: str = "sw_industry_1",
) -> pd.Series:
    """Low-volatility signal: prefer stocks with lower recent volatility.

    Signal = -ts_rank(vol_field, window), group-neutralized.
    Lower volatility → higher signal value.
    """
    _require_columns(factor_panel, [vol_field])
    signal = -ts_rank(factor_panel[vol_field], window)
    if group_col in factor_panel.columns:
        return group_rank(signal, factor_panel[group_col])
    return rank(signal)


def template_regime_momentum(
    factor_panel: pd.DataFrame,
    price_field: str = "close",
    vol_window: int = 20,
    mom_window: int = 10,
    rev_window: int = 5,
    vol_threshold: float = 0.03,
    group_col: str = "sw_industry_1",
) -> pd.Series:
    """Regime-switching momentum/reversal.

    In low-volatility regime (rolling vol < vol_threshold): use momentum.
    In high-volatility regime: use mean reversion.

    Source: arxiv 2410.14841 + springer 41260-024-00372
    """
    _require_columns(factor_panel, [price_field])
    returns = delta(factor_panel[price_field], 1)
    rolling_vol = returns.groupby(level="symbol", group_keys=False).rolling(vol_window).std()
    rolling_vol = rolling_vol.reset_index(level=0, drop=True)

    mom_signal = ts_rank(delta(factor_panel[price_field], 1), mom_window)
    rev_signal = -ts_rank(delta(factor_panel[price_field], 1), rev_window)

    is_low_vol = rolling_vol < vol_threshold
    signal = pd.Series(np.nan, index=factor_panel.index)
    signal[is_low_vol] = mom_signal[is_low_vol]
    signal[~is_low_vol] = rev_signal[~is_low_vol]

    if group_col in factor_panel.columns:
        return group_rank(signal, factor_panel[group_col])
    return rank(signal)


def add_template_signals(
    factor_panel: pd.DataFrame,
    template_names: Sequence[str],
    **kwargs,
) -> pd.DataFrame:
    """Append requested template signals as new factor columns."""
    result = factor_panel.copy()
    for name in template_names:
        if name == "fundamental_value":
            result[name] = template_fundamental_value(result, **kwargs.get(name, {}))
        elif name == "technical_momentum":
            result[name] = template_technical_momentum(result, **kwargs.get(name, {}))
        elif name == "golden_combo":
            result[name] = golden_combo(result, **kwargs.get(name, {}))
        elif name == "value_momentum":
            result[name] = template_value_momentum(result, **kwargs.get(name, {}))
        elif name == "mean_reversion":
            result[name] = template_mean_reversion(result, **kwargs.get(name, {}))
        elif name == "fundamental_quality":
            result[name] = template_fundamental_quality(result, **kwargs.get(name, {}))
        elif name == "low_volatility":
            result[name] = template_low_volatility(result, **kwargs.get(name, {}))
        elif name == "regime_momentum":
            result[name] = template_regime_momentum(result, **kwargs.get(name, {}))
        elif name == "overnight_reversal":
            result[name] = template_overnight_reversal(result, **kwargs.get(name, {}))
        else:
            raise KeyError(f"unknown template: {name}")
    return result

