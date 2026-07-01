"""Minimal research-loop interfaces."""

from .loop import AlphaCandidate, ResearchLoop
from .validators import METRIC_THRESHOLDS, ValidationResult, validate_metrics
from .factor_deployment import (
    DeploymentVerdict,
    build_returns_matrix,
    evaluate_deployment,
    factor_long_short_returns,
)

__all__ = [
    "AlphaCandidate",
    "ResearchLoop",
    "METRIC_THRESHOLDS",
    "ValidationResult",
    "validate_metrics",
    "DeploymentVerdict",
    "build_returns_matrix",
    "evaluate_deployment",
    "factor_long_short_returns",
]

