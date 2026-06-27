"""Factor package public API.

Legacy factor-panel builders remain re-exported so existing CLI/tests keep
working after the package split.
"""

from .legacy import load_price_data, load_daily_mcap_pb, compute_factors
from .neutralization import (
    neutralize_ols_residual,
    neutralize_by_sector,
    neutralize_by_size,
    neutralize_by_both,
    neutralize_blend,
)
from .preprocessing import (
    winsorize_mad,
    winsorize_percentile,
    winsorize_cross_sectional,
    zscore_cross_sectional,
    rank_normalize,
    preprocess_pipeline,
)

__all__ = [
    # Legacy
    "load_price_data",
    "load_daily_mcap_pb",
    "compute_factors",
    # Neutralization
    "neutralize_ols_residual",
    "neutralize_by_sector",
    "neutralize_by_size",
    "neutralize_by_both",
    "neutralize_blend",
    # Preprocessing
    "winsorize_mad",
    "winsorize_percentile",
    "winsorize_cross_sectional",
    "zscore_cross_sectional",
    "rank_normalize",
    "preprocess_pipeline",
]

