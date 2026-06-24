"""Factor package public API.

Legacy factor-panel builders remain re-exported so existing CLI/tests keep
working after the package split.
"""

from .legacy import load_price_data, load_daily_mcap_pb, compute_factors

__all__ = [
    "load_price_data",
    "load_daily_mcap_pb",
    "compute_factors",
]

