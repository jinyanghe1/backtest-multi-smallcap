"""Data-driven factor combination utilities.

This module turns a dictionary of raw factor Series into a decorrelated,
IC-weighted composite signal suitable for cross-sectional ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from tools.backtest_mvp.factors.neutralization import neutralize_blend


@dataclass
class CombineResult:
    """Result of a decorrelated IC-weighted factor combination."""

    composite: pd.Series
    selected: list[str]
    weights: dict[str, float]
    ic: dict[str, float]
    ic_ir: dict[str, float]
    corr_matrix: pd.DataFrame


def _date_level(index: pd.Index) -> str | int | None:
    if isinstance(index, pd.MultiIndex):
        if "date" in index.names:
            return "date"
        return 0
    return None


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """Z-score a Series by date when possible, preserving the input index."""

    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    level = _date_level(series.index)
    if level is not None:
        return series.groupby(level=level, group_keys=False).apply(_zscore)
    return _zscore(series)


def _spearman_ic(factor: pd.Series, returns: pd.Series) -> tuple[float, float, float, float]:
    """Compute mean cross-sectional Spearman IC and IC-IR."""
    aligned = pd.DataFrame({"factor": factor, "return": returns}).dropna()
    if aligned.empty:
        return 0.0, 1.0, 0.0, 0.0

    ic_values: list[float] = []
    level = _date_level(aligned.index)
    if level is not None:
        for _, group in aligned.groupby(level=level):
            if len(group) < 2:
                continue
            if group["factor"].nunique(dropna=True) < 2 or group["return"].nunique(dropna=True) < 2:
                continue
            corr = group["factor"].corr(group["return"], method="spearman")
            if not pd.isna(corr):
                ic_values.append(float(corr))
    else:
        if aligned["factor"].nunique(dropna=True) >= 2 and aligned["return"].nunique(dropna=True) >= 2:
            corr = aligned["factor"].corr(aligned["return"], method="spearman")
            if not pd.isna(corr):
                ic_values.append(float(corr))

    if not ic_values:
        return 0.0, 1.0, 0.0, 0.0

    ic_s = pd.Series(ic_values, dtype=float)
    ic_mean = float(ic_s.mean())
    ic_std = float(ic_s.std()) if len(ic_s) > 1 else 0.0
    ic_ir = ic_mean / ic_std if ic_std > 0 else (np.sign(ic_mean) * np.inf if ic_mean else 0.0)
    ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_s))) if ic_std > 0 else 0.0
    return ic_mean, ic_std, float(ic_ir), float(ic_tstat)


def _controls_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    controls = pd.DataFrame(index=panel.index)
    if "mcap" in panel.columns:
        controls["log_mcap"] = np.log(panel["mcap"].replace(0, np.nan).astype(float))
    industry_col = "industry_code" if "industry_code" in panel.columns else "industry"
    if industry_col in panel.columns:
        dummies = pd.get_dummies(panel[industry_col].astype(str), prefix="ind", drop_first=True)
        controls = pd.concat([controls, dummies.astype(float)], axis=1)
    return controls.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _neutralize_factors(
    factor_values: dict[str, pd.Series],
    panel: pd.DataFrame | None,
) -> dict[str, pd.Series]:
    if panel is None:
        return factor_values
    controls = _controls_from_panel(panel)
    if controls.empty:
        return factor_values
    return {
        name: neutralize_blend(factor.reindex(panel.index), controls, strength=1.0)
        for name, factor in factor_values.items()
    }


def _finite_strength(value: float) -> float:
    if pd.isna(value):
        return 0.0
    if np.isinf(value):
        return 1e12
    return float(abs(value))


def _decorrelate_ordered(
    factors: dict[str, pd.Series],
    candidates: list[str],
    max_corr: float,
) -> list[str]:
    selected: list[str] = []
    for name in candidates:
        if not selected:
            selected.append(name)
            continue
        max_abs_corr = 0.0
        for chosen in selected:
            aligned = pd.DataFrame({"candidate": factors[name], "chosen": factors[chosen]}).dropna()
            if len(aligned) < 2:
                continue
            if aligned["candidate"].nunique(dropna=True) < 2 or aligned["chosen"].nunique(dropna=True) < 2:
                continue
            corr = aligned["candidate"].corr(aligned["chosen"])
            if not pd.isna(corr):
                max_abs_corr = max(max_abs_corr, abs(float(corr)))
        if max_abs_corr < max_corr:
            selected.append(name)
    return selected


def _safe_corr_matrix(columns: dict[str, pd.Series], names: list[str]) -> pd.DataFrame:
    """Pairwise Pearson correlations without numpy warnings on constants."""
    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for left in names:
        for right in names:
            if left == right:
                matrix.loc[left, right] = 1.0
                continue
            aligned = pd.DataFrame({"left": columns[left], "right": columns[right]}).dropna()
            if len(aligned) < 2:
                continue
            if aligned["left"].nunique(dropna=True) < 2 or aligned["right"].nunique(dropna=True) < 2:
                continue
            matrix.loc[left, right] = aligned["left"].corr(aligned["right"])
    return matrix


def combine_factors(
    factor_values: dict[str, pd.Series],
    fwd_returns: pd.Series,
    *,
    method: str = "ic_ir",
    max_corr: float = 0.6,
    neutralize: bool = False,
    panel: pd.DataFrame | None = None,
) -> CombineResult:
    """Build a decorrelated, sign-aligned, IC-weighted composite factor.

    Parameters
    ----------
    factor_values:
        Mapping from factor name to panel-aligned factor Series.
    fwd_returns:
        Forward returns used only for estimating IC statistics.
    method:
        ``"equal"``, ``"ic"``, or ``"ic_ir"`` weighting.
    max_corr:
        Greedy decorrelation threshold. A candidate is kept only when its
        absolute correlation with every selected factor is below this value.
    neutralize:
        If true and ``panel`` is supplied, neutralize each factor against size
        and industry controls before scoring and combining.
    panel:
        Factor panel carrying ``mcap`` and ``industry_code``/``industry``.
    """
    if method not in {"equal", "ic", "ic_ir"}:
        raise ValueError("method must be one of: equal, ic, ic_ir")

    if not factor_values:
        return CombineResult(
            composite=pd.Series(dtype=float),
            selected=[],
            weights={},
            ic={},
            ic_ir={},
            corr_matrix=pd.DataFrame(dtype=float),
        )

    processed = {name: pd.to_numeric(series, errors="coerce") for name, series in factor_values.items()}
    if neutralize:
        processed = _neutralize_factors(processed, panel)

    common_index = fwd_returns.index
    for series in processed.values():
        common_index = common_index.intersection(series.index)
    if len(common_index) == 0:
        common_index = fwd_returns.index

    processed = {name: series.reindex(common_index) for name, series in processed.items()}
    returns = fwd_returns.reindex(common_index)

    ic: dict[str, float] = {}
    ic_ir: dict[str, float] = {}
    for name, factor in processed.items():
        ic_mean, _, ir, _ = _spearman_ic(factor, returns)
        ic[name] = float(ic_mean)
        ic_ir[name] = float(ir)

    ordered = sorted(processed, key=lambda n: (_finite_strength(ic_ir[n]), _finite_strength(ic[n])), reverse=True)
    selected = _decorrelate_ordered(processed, ordered, max_corr)

    if not selected:
        return CombineResult(
            composite=pd.Series(dtype=float, index=common_index),
            selected=[],
            weights={},
            ic=ic,
            ic_ir=ic_ir,
            corr_matrix=pd.DataFrame(dtype=float),
        )

    if method == "equal":
        raw_weights = {name: 1.0 for name in selected}
    elif method == "ic":
        raw_weights = {name: _finite_strength(ic[name]) for name in selected}
    else:
        raw_weights = {name: _finite_strength(ic_ir[name]) for name in selected}
    total = sum(raw_weights.values())
    if total <= 0 or not np.isfinite(total):
        weights = {name: 1.0 / len(selected) for name in selected}
    else:
        weights = {name: float(raw_weights[name] / total) for name in selected}

    standardized: dict[str, pd.Series] = {}
    for name in selected:
        sign = -1.0 if ic[name] < 0 else 1.0
        standardized[name] = _cross_sectional_zscore(sign * processed[name]).reindex(common_index)

    composite = pd.Series(0.0, index=common_index, dtype=float)
    for name, weight in weights.items():
        composite = composite.add(standardized[name].fillna(0.0) * weight, fill_value=0.0)

    corr_matrix = _safe_corr_matrix(standardized, selected) if selected else pd.DataFrame(dtype=float)
    return CombineResult(
        composite=composite,
        selected=selected,
        weights=weights,
        ic=ic,
        ic_ir=ic_ir,
        corr_matrix=corr_matrix,
    )


def _forward_returns_from_close(panel: pd.DataFrame, fwd_period: int) -> pd.Series:
    if "close" not in panel.columns:
        return pd.Series(dtype=float, index=panel.index)
    close = pd.to_numeric(panel["close"], errors="coerce")
    if _date_level(close.index) is not None:
        future = close.groupby(level="symbol", group_keys=False).shift(-fwd_period)
        return future / close - 1.0
    return close.pct_change(fwd_period).shift(-fwd_period)


def _infer_current_date(snapshot: pd.DataFrame, state: dict) -> pd.Timestamp | None:
    if "date" in snapshot.attrs:
        return pd.Timestamp(snapshot.attrs["date"])
    if isinstance(snapshot.index, pd.MultiIndex):
        level = "date" if "date" in snapshot.index.names else 0
        return pd.Timestamp(max(snapshot.index.get_level_values(level)))
    return state.get("current_date")


def _history_source(snapshot: pd.DataFrame, state: dict) -> pd.DataFrame:
    panel = snapshot.attrs.get("panel")
    if panel is not None:
        return panel
    if isinstance(snapshot.index, pd.MultiIndex):
        return snapshot
    panel = state.get("panel")
    if panel is not None:
        return panel
    return snapshot


def make_composite_strategy_def(
    factor_funcs: dict[str, Callable[[pd.DataFrame], pd.Series]],
    *,
    method: str = "ic_ir",
    max_corr: float = 0.6,
    n_stocks: int = 30,
    universe_filter=None,
    lookback: int = 252,
    fwd_period: int = 20,
) -> dict:
    """Create an engine strategy_def with an adaptive composite ``ranking_fn``.

    Existing engines call ``ranking_fn(snapshot) -> pd.Series`` after applying
    ``universe_filter``. Because that contract supplies only the current
    cross-section, this function's ranking function also accepts either a
    MultiIndex historical panel as ``snapshot`` or ``snapshot.attrs['panel']``
    plus ``snapshot.attrs['date']``. In the normal engine path, the wrapper
    universe filter records the rebalance date; callers that need true trailing
    history should pass the full panel through ``snapshot.attrs['panel']`` or
    bind factor functions that close over their data source.

    No-lookahead guarantee: for a rebalance at date ``t``, factor values are
    computed on rows with date ``<= t`` only. IC weights are estimated on dates
    ``<= t - fwd_period`` so every forward return used for weighting is fully
    known strictly before the rebalance date.
    """
    if method not in {"equal", "ic", "ic_ir"}:
        raise ValueError("method must be one of: equal, ic, ic_ir")

    state: dict = {"current_date": None, "panel": None}

    def recording_filter(snapshot: pd.DataFrame, dates, step: int):
        if dates is not None and step is not None and step < len(dates):
            state["current_date"] = pd.Timestamp(dates[step])
        if "panel" in snapshot.attrs:
            state["panel"] = snapshot.attrs["panel"]
        if universe_filter is not None:
            return universe_filter(snapshot, dates, step)
        return list(snapshot.index)

    def ranking_fn(snapshot: pd.DataFrame) -> pd.Series:
        current_date = _infer_current_date(snapshot, state)
        source = _history_source(snapshot, state)
        if current_date is None:
            current_date = pd.Timestamp.max

        if isinstance(source.index, pd.MultiIndex):
            date_level = "date" if "date" in source.index.names else 0
            dates = pd.DatetimeIndex(pd.to_datetime(source.index.get_level_values(date_level))).unique().sort_values()
            train_end_pos = dates.searchsorted(current_date, side="right") - fwd_period - 1
            start_pos = max(0, train_end_pos - lookback + 1)
            train_dates = dates[start_pos : train_end_pos + 1] if train_end_pos >= 0 else pd.DatetimeIndex([])
            current_dates = dates[dates <= current_date]
            trailing_dates = current_dates[-lookback:] if len(current_dates) else pd.DatetimeIndex([])
            train_panel = source.loc[source.index.get_level_values(date_level).isin(train_dates)] if len(train_dates) else source.iloc[0:0]
            trailing_panel = source.loc[source.index.get_level_values(date_level).isin(trailing_dates)] if len(trailing_dates) else source.iloc[0:0]
            known_panel = source.loc[source.index.get_level_values(date_level).isin(current_dates)] if len(current_dates) else source.iloc[0:0]
        else:
            train_panel = source.iloc[0:0]
            trailing_panel = source
            known_panel = source

        current_symbols = snapshot.index
        if isinstance(current_symbols, pd.MultiIndex):
            symbol_level = "symbol" if "symbol" in current_symbols.names else -1
            current_symbols = current_symbols.get_level_values(symbol_level).unique()

        if train_panel.empty:
            factor_now = {name: func(trailing_panel).dropna() for name, func in factor_funcs.items()}
            scores = pd.Series(0.0, index=current_symbols, dtype=float)
            used = 0
            for factor in factor_now.values():
                if isinstance(factor.index, pd.MultiIndex) and "date" in factor.index.names:
                    try:
                        cross = factor.xs(current_date, level="date")
                    except KeyError:
                        cross = factor.groupby(level="symbol").tail(1)
                        cross.index = cross.index.get_level_values("symbol")
                else:
                    cross = factor
                scores = scores.add(_cross_sectional_zscore(cross).reindex(current_symbols).fillna(0.0), fill_value=0.0)
                used += 1
            return scores / used if used else scores

        train_factors = {name: func(train_panel) for name, func in factor_funcs.items()}
        train_returns = _forward_returns_from_close(known_panel, fwd_period).reindex(train_panel.index)
        combined = combine_factors(
            train_factors,
            train_returns,
            method=method,
            max_corr=max_corr,
            neutralize=False,
        )

        current_factor_values: dict[str, pd.Series] = {}
        for name in combined.selected or list(factor_funcs):
            values = factor_funcs[name](trailing_panel)
            if isinstance(values.index, pd.MultiIndex) and "date" in values.index.names:
                try:
                    current_factor_values[name] = values.xs(current_date, level="date")
                except KeyError:
                    tail = values.groupby(level="symbol").tail(1)
                    tail.index = tail.index.get_level_values("symbol")
                    current_factor_values[name] = tail
            else:
                current_factor_values[name] = values

        scores = pd.Series(0.0, index=current_symbols, dtype=float)
        selected = combined.selected or list(current_factor_values)
        if not selected:
            return scores
        weights = combined.weights or {name: 1.0 / len(selected) for name in selected}
        for name in selected:
            sign = -1.0 if combined.ic.get(name, 0.0) < 0 else 1.0
            cross = _cross_sectional_zscore(sign * current_factor_values[name]).reindex(current_symbols)
            scores = scores.add(cross.fillna(0.0) * weights.get(name, 0.0), fill_value=0.0)
        return scores

    return {
        "name": "Decorrelated IC-weighted composite",
        "universe_filter": recording_filter,
        "ranking_factor": None,
        "ascending": False,
        "ranking_fn": ranking_fn,
        "n_stocks": n_stocks,
        "factor_weights": None,
        "combiner_method": method,
        "max_corr": max_corr,
        "lookback": lookback,
        "fwd_period": fwd_period,
    }
