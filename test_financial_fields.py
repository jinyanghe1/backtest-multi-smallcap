import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from tools.backtest_mvp.factors.financial_fields import derive_financial_fields, get_field_spec


def test_derive_financial_fields_uses_notice_date_not_report_date():
    financials = pd.DataFrame({
        "symbol": ["sh600000"],
        "report_date": [pd.Timestamp("2023-12-31")],
        "notice_date": [pd.Timestamp("2024-01-03")],
        "bps": [5.0],
        "eps": [1.0],
        "revenue": [100.0],
        "net_profit": [10.0],
        "total_equity": [50.0],
    })
    prices = pd.DataFrame({
        "symbol": ["sh600000", "sh600000"],
        "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")],
        "close": [10.0, 12.0],
    })

    derived = derive_financial_fields(financials, prices, fields=["bps", "pb"])
    assert pd.isna(derived.loc[0, "bps"])
    assert derived.loc[1, "bps"] == 5.0
    assert derived.loc[1, "pb"] == 12.0 / 5.0


def test_field_spec_marks_unavailable_fields():
    assert get_field_spec("gross_margin").status == "unavailable"


def test_roa_ttm_derivation_uses_notice_date():
    """roa_ttm = net_profit_ttm / total_assets, aligned by notice_date."""
    financials = pd.DataFrame({
        "symbol": ["sh600000"] * 5,
        "report_date": pd.date_range("2023-03-31", periods=5, freq="QE"),
        "notice_date": pd.date_range("2023-04-28", periods=5, freq="91D"),
        "net_profit": [10.0, 10.0, 10.0, 10.0, 10.0],
        "total_assets": [200.0, 200.0, 200.0, 200.0, 200.0],
        "total_equity": [50.0, 50.0, 50.0, 50.0, 50.0],
    })
    prices = pd.DataFrame({
        "symbol": ["sh600000"] * 3,
        "date": [pd.Timestamp("2023-04-01"), pd.Timestamp("2023-08-01"), pd.Timestamp("2024-02-01")],
        "close": [10.0, 11.0, 12.0],
    })
    derived = derive_financial_fields(financials, prices, fields=["roa_ttm"])
    # After 4 quarters of TTM (notice_date 2024-01-26), net_profit_ttm = 40, total_assets = 200 => roa = 0.2
    # The price date 2024-02-01 is after the 4th notice_date, so roa_ttm should be available
    last_row = derived[derived["date"] == pd.Timestamp("2024-02-01")]
    assert not last_row.empty
    val = last_row["roa_ttm"].iloc[0]
    assert not pd.isna(val), "roa_ttm should be non-null after 4 quarters of data"
    assert abs(val - 0.2) < 0.01

