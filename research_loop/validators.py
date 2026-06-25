"""Metric validation for alpha candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


METRIC_THRESHOLDS = {
    "sharpe": 1.5,
    "fitness": 1.0,
    "turnover": (1.0, 20.0),  # engine reports percent points
    "drawdown": 15.0,         # engine reports negative percent points
    "self_correlation": 0.7,
}


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    metrics: dict[str, float]
    failures: dict[str, str] = field(default_factory=dict)


def _result_metrics(result: Any, self_correlation: float | None = None) -> dict[str, float]:
    annual_return = float(getattr(result, "annual_return", 0.0))
    sharpe = float(getattr(result, "sharpe_ratio", 0.0))
    turnover = float(getattr(result, "avg_turnover", 0.0))
    drawdown = abs(float(getattr(result, "max_drawdown", 0.0)))
    turnover_decimal = max(turnover / 100.0, 1e-12)
    return_decimal = max(annual_return / 100.0, 0.0)
    fitness = sharpe * np.sqrt(return_decimal / turnover_decimal) if turnover_decimal > 0 else 0.0
    metrics = {
        "annual_return": annual_return,
        "sharpe": sharpe,
        "fitness": float(fitness),
        "turnover": turnover,
        "drawdown": drawdown,
    }
    if self_correlation is not None:
        metrics["self_correlation"] = float(self_correlation)
    return metrics


def validate_metrics(
    result: Any,
    thresholds: dict | None = None,
    self_correlation: float | None = None,
) -> ValidationResult:
    """Validate engine metrics, respecting BacktestResult's percent units."""
    thresholds = thresholds or METRIC_THRESHOLDS
    metrics = _result_metrics(result, self_correlation)
    failures: dict[str, str] = {}

    if metrics["sharpe"] < thresholds["sharpe"]:
        failures["sharpe"] = f"{metrics['sharpe']:.2f} < {thresholds['sharpe']:.2f}"
    if metrics["fitness"] < thresholds["fitness"]:
        failures["fitness"] = f"{metrics['fitness']:.2f} < {thresholds['fitness']:.2f}"

    low_turnover, high_turnover = thresholds["turnover"]
    if not (low_turnover <= metrics["turnover"] <= high_turnover):
        failures["turnover"] = f"{metrics['turnover']:.2f} not in [{low_turnover:.2f}, {high_turnover:.2f}]"

    if metrics["drawdown"] > thresholds["drawdown"]:
        failures["drawdown"] = f"{metrics['drawdown']:.2f} > {thresholds['drawdown']:.2f}"

    if "self_correlation" in metrics and metrics["self_correlation"] >= thresholds["self_correlation"]:
        failures["self_correlation"] = (
            f"{metrics['self_correlation']:.2f} >= {thresholds['self_correlation']:.2f}"
        )

    return ValidationResult(passed=not failures, metrics=metrics, failures=failures)


def validate_signal_coverage(
    factor_panel: "pd.DataFrame",
    signal_name: str,
    warn_threshold: float = 0.3,
    error_threshold: float = 0.1,
) -> dict:
    """Validate signal non-null coverage in a factor panel.

    Parameters
    ----------
    factor_panel : DataFrame with MultiIndex (date, symbol)
    signal_name : str, column to check
    warn_threshold : float, coverage below this triggers "warning"
    error_threshold : float, coverage below this triggers "error"

    Returns
    -------
    dict with keys: signal, total, non_null, coverage, status, per_date_min, per_date_max
    status is one of "ok", "warning", "error"
    """
    import pandas as pd

    if signal_name not in factor_panel.columns:
        return {
            "signal": signal_name,
            "total": 0,
            "non_null": 0,
            "coverage": 0.0,
            "status": "error",
            "per_date_min": 0.0,
            "per_date_max": 0.0,
            "error": f"column '{signal_name}' not found in factor_panel",
        }

    col = factor_panel[signal_name]
    total = len(col)
    non_null = int(col.notna().sum())
    coverage = non_null / total if total > 0 else 0.0

    if isinstance(factor_panel.index, __import__("pandas").MultiIndex) and "date" in factor_panel.index.names:
        per_date = col.groupby(level="date").apply(lambda s: s.notna().mean())
        per_date_min = float(per_date.min()) if len(per_date) > 0 else 0.0
        per_date_max = float(per_date.max()) if len(per_date) > 0 else 0.0
    else:
        per_date_min = per_date_max = coverage

    if coverage < error_threshold:
        status = "error"
    elif coverage < warn_threshold:
        status = "warning"
    else:
        status = "ok"

    return {
        "signal": signal_name,
        "total": total,
        "non_null": non_null,
        "coverage": round(coverage, 4),
        "status": status,
        "per_date_min": round(per_date_min, 4),
        "per_date_max": round(per_date_max, 4),
    }


def compute_hit_rate(
    factor_panel: "pd.DataFrame",
    return_panel: "pd.DataFrame",
    signal_name: str,
) -> float:
    """Compute hit rate: fraction of rebalance dates where signal direction
    matches realized return direction.

    For each date, compute Spearman correlation sign between signal and
    forward returns. Hit rate = fraction of dates with positive correlation.
    """
    import pandas as pd

    if signal_name not in factor_panel.columns:
        return 0.0
    signal = factor_panel[signal_name]
    returns = return_panel["daily_return"] if "daily_return" in return_panel.columns else return_panel.iloc[:, 0]

    # Align
    common_idx = signal.index.intersection(returns.index)
    if len(common_idx) == 0:
        return 0.0

    dates = common_idx.get_level_values("date").unique() if "date" in signal.index.names else pd.Index([0])
    hits = 0
    total = 0
    for d in dates:
        if "date" in signal.index.names:
            s = signal.xs(d, level="date")
            r = returns.xs(d, level="date")
        else:
            s = signal
            r = returns
        common = s.dropna().index.intersection(r.dropna().index)
        if len(common) < 3:
            continue
        corr = s.loc[common].rank().corr(r.loc[common].rank())
        if pd.notna(corr):
            total += 1
            if corr > 0:
                hits += 1
    return hits / total if total > 0 else 0.0


def compute_signal_turnover(
    factor_panel: "pd.DataFrame",
    signal_name: str,
) -> float:
    """Compute signal turnover: mean absolute change in signal rank between
    consecutive dates, normalized to [0, 1].

    High turnover → signal is unstable → high trading costs.
    """
    import pandas as pd

    if signal_name not in factor_panel.columns:
        return 0.0
    signal = factor_panel[signal_name]
    if "date" not in signal.index.names:
        return 0.0

    # Rank cross-sectionally per date
    ranked = signal.groupby(level="date", group_keys=False).rank(pct=True)
    # Compute turnover per date transition
    dates = ranked.index.get_level_values("date").unique().sort_values()
    turnovers = []
    for i in range(1, len(dates)):
        prev = ranked.xs(dates[i - 1], level="date")
        curr = ranked.xs(dates[i], level="date")
        common = prev.dropna().index.intersection(curr.dropna().index)
        if len(common) < 2:
            continue
        diff = (curr.loc[common] - prev.loc[common]).abs()
        turnovers.append(float(diff.mean()) / 2.0)  # normalize to [0, 1]

    return float(np.mean(turnovers)) if turnovers else 0.0


def factor_decay_halflife(
    ic_decay: dict[int, float],
) -> float | None:
    """Estimate the half-life of factor IC decay.

    Given the output of ``CrossSectionalEngine.compute_ic_decay`` (a
    ``{lag: mean_ic}`` mapping), find the lag at which IC decays to
    half of its initial value (IC at the smallest available lag).

    If IC never drops below half, returns the largest lag.
    If IC is zero or negative at the first lag, returns ``None``.

    Source: AlphaAgent (arxiv 2502.16789) — decay-resistant factor design.
    """
    if not ic_decay:
        return None
    sorted_lags = sorted(ic_decay.keys())
    ic_1 = ic_decay[sorted_lags[0]]
    if ic_1 is None or ic_1 <= 0:
        return None
    half_ic = ic_1 / 2.0
    for lag in sorted_lags:
        val = ic_decay[lag]
        if val is not None and val <= half_ic:
            # Linear interpolation between previous and current lag
            prev_lag = sorted_lags[0] if lag == sorted_lags[0] else sorted_lags[sorted_lags.index(lag) - 1]
            prev_val = ic_decay[prev_lag]
            if lag == prev_lag or prev_val is None or prev_val == val:
                return float(lag)
            # Interpolate: lag_est = prev_lag + (prev_val - half) / (prev_val - val) * (lag - prev_lag)
            frac = (prev_val - half_ic) / (prev_val - val)
            return float(prev_lag + frac * (lag - prev_lag))
    # IC never drops to half — return the largest lag tested
    return float(sorted_lags[-1])

