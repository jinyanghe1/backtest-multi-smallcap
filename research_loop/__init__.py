"""Minimal research-loop interfaces."""

from .loop import AlphaCandidate, ResearchLoop
from .validators import METRIC_THRESHOLDS, ValidationResult, validate_metrics

__all__ = [
    "AlphaCandidate",
    "ResearchLoop",
    "METRIC_THRESHOLDS",
    "ValidationResult",
    "validate_metrics",
]

