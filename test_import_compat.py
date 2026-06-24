import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_legacy_package_imports():
    from tools.backtest_mvp.factors import compute_factors, load_daily_mcap_pb, load_price_data
    from tools.backtest_mvp.data import DATA_DIR, fetch_stock_kline, get_data_summary

    assert callable(compute_factors)
    assert callable(load_daily_mcap_pb)
    assert callable(load_price_data)
    assert DATA_DIR.name == "data_cache"
    assert callable(fetch_stock_kline)
    assert callable(get_data_summary)


def test_local_imports_after_package_split():
    project = Path(__file__).parent
    sys.path.insert(0, str(project))
    from factors import compute_factors
    from data import DATA_DIR

    assert callable(compute_factors)
    assert DATA_DIR.name == "data_cache"

