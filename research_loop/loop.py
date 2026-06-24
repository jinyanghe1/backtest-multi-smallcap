"""Thin research-loop adapter over the existing cross-sectional engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .validators import ValidationResult, validate_metrics

try:
    from tools.backtest_mvp.engine import BacktestResult, CrossSectionalEngine
except ModuleNotFoundError:
    from engine import BacktestResult, CrossSectionalEngine


@dataclass
class AlphaCandidate:
    expr: str
    signal_col: Optional[str] = None
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    drawdown: float = 0.0
    self_correlation: float = 0.0
    status: str = "ACTIVE"
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class ResearchLoop:
    """Minimal orchestration layer; template generation remains outside this class."""

    def __init__(
        self,
        n_stocks: int = 30,
        rebalance_freq: str = "M",
        thresholds: Optional[dict] = None,
        engine_cls=CrossSectionalEngine,
    ):
        self.n_stocks = n_stocks
        self.rebalance_freq = rebalance_freq
        self.thresholds = thresholds
        self.engine_cls = engine_cls

    def simulate(
        self,
        alpha: AlphaCandidate,
        factor_panel: pd.DataFrame,
        return_panel: pd.DataFrame,
        ascending: bool = False,
    ) -> BacktestResult:
        signal_col = alpha.signal_col or alpha.expr
        if signal_col not in factor_panel.columns:
            raise KeyError(f"signal column not found in factor_panel: {signal_col}")

        engine = self.engine_cls(
            factor_panel=factor_panel,
            return_panel=return_panel,
            n_stocks=self.n_stocks,
            rebalance_freq=self.rebalance_freq,
        )
        result = engine.run(ranking_factor=signal_col, ascending=ascending)
        validation = validate_metrics(result, self.thresholds, alpha.self_correlation)
        alpha.metrics = validation.metrics
        alpha.sharpe = validation.metrics["sharpe"]
        alpha.fitness = validation.metrics["fitness"]
        alpha.turnover = validation.metrics["turnover"]
        alpha.drawdown = validation.metrics["drawdown"]
        return result

    def validate_metrics(self, result: BacktestResult, self_correlation: float | None = None) -> ValidationResult:
        return validate_metrics(result, self.thresholds, self_correlation)

    def check_self_correlation(self, alpha_signal: pd.Series, existing_signals: list[pd.Series]) -> float:
        if not existing_signals:
            return 0.0
        correlations = []
        for signal in existing_signals:
            left, right = alpha_signal.align(signal, join="inner")
            if len(left.dropna()) < 3 or len(right.dropna()) < 3:
                continue
            corr = left.rank().corr(right.rank())
            if pd.notna(corr):
                correlations.append(abs(float(corr)))
        return max(correlations) if correlations else 0.0
