"""Data package public API.

This package preserves the historical ``tools.backtest_mvp.data`` imports while
adding the new provider/resolver interfaces.
"""

from .legacy import (
    DATA_DIR,
    fetch_microcap_symbols,
    fetch_stock_kline,
    fetch_stock_quote,
    download_microcap_universe,
    get_data_summary,
)

__all__ = [
    "DATA_DIR",
    "fetch_microcap_symbols",
    "fetch_stock_kline",
    "fetch_stock_quote",
    "download_microcap_universe",
    "get_data_summary",
]

