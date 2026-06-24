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

