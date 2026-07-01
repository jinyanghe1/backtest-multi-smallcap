import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math

import pandas as pd
import pytest

from tools.backtest_mvp.data import shareholders as sh
from tools.backtest_mvp.data.shareholders import (
    SCHEMA_COLUMNS,
    attach_to_panel,
    fetch_shareholders,
    load_shareholders,
    normalize_shareholders,
    update_cache,
)
from tools.backtest_mvp.factors.factor_library import shareholder_concentration


def test_fetch_shareholders_normalizes_chinese_columns_and_computes_qoq(monkeypatch):
    raw = pd.DataFrame({
        "股东户数统计截止日": ["2024-03-31", "2024-06-30", "2024-09-30"],
        "股东户数": ["1,000", "900", "990"],
        "户均持股市值": ["10万", "11万", "12万"],
    })
    monkeypatch.setattr(sh, "_call_akshare_shareholders", lambda symbol: raw)

    df = fetch_shareholders("sz000001")

    assert list(df.columns) == SCHEMA_COLUMNS
    assert df["symbol"].tolist() == ["000001", "000001", "000001"]
    assert pd.api.types.is_integer_dtype(df["holder_count"])
    assert df["holder_count"].tolist() == [1000, 900, 990]
    assert math.isclose(df.loc[1, "holder_count_qoq"], -0.1)
    assert math.isclose(df.loc[2, "holder_count_qoq"], 0.1)
    assert df.loc[0, "avg_holding_value"] == 100000


def test_normalize_uses_provider_qoq_for_one_row_percentage():
    raw = pd.DataFrame({
        "截止日": ["2024-03-31"],
        "股东户数-本次": [1000],
        "股东户数-增减比例": ["5.5"],
        "户均流通股": [123.0],
    })
    df = normalize_shareholders(raw, symbol="000001")
    assert math.isclose(df.loc[0, "holder_count_qoq"], 0.055)
    assert df.loc[0, "avg_float_per_holder"] == 123.0


def test_asof_merge_attaches_until_next_report(tmp_path):
    pd.DataFrame({
        "symbol": ["000001", "000001"],
        "report_date": pd.to_datetime(["2024-01-03", "2024-01-06"]),
        "holder_count": [100, 80],
        "holder_count_qoq": [float("nan"), -0.2],
        "avg_holding_value": [float("nan"), float("nan")],
        "avg_float_per_holder": [float("nan"), float("nan")],
    }).to_parquet(tmp_path / "000001.parquet", index=False)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=8, freq="D"), ["000001"]],
        names=["date", "symbol"],
    )
    panel = pd.DataFrame(index=idx)

    out = attach_to_panel(panel, cache_dir=tmp_path)

    values = out["shareholders"].droplevel("symbol").tolist()
    assert pd.isna(values[0]) and pd.isna(values[1])
    assert values[2:5] == [100.0, 100.0, 100.0]
    assert values[5:] == [80.0, 80.0, 80.0]


def test_asof_merge_no_lookahead(tmp_path):
    pd.DataFrame({
        "symbol": ["000001"],
        "report_date": pd.to_datetime(["2024-01-10"]),
        "holder_count": [100],
        "holder_count_qoq": [float("nan")],
        "avg_holding_value": [float("nan")],
        "avg_float_per_holder": [float("nan")],
    }).to_parquet(tmp_path / "000001.parquet", index=False)
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-09", "2024-01-10"]), ["000001"]],
        names=["date", "symbol"],
    )
    out = attach_to_panel(pd.DataFrame(index=idx), cache_dir=tmp_path)
    assert pd.isna(out.iloc[0]["shareholders"])
    assert out.iloc[1]["shareholders"] == 100.0


def test_multi_symbol_isolation(tmp_path):
    pd.DataFrame({
        "symbol": ["000001", "000002"],
        "report_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "holder_count": [100, 200],
        "holder_count_qoq": [float("nan"), float("nan")],
        "avg_holding_value": [float("nan"), float("nan")],
        "avg_float_per_holder": [float("nan"), float("nan")],
    }).query("symbol == '000001'").to_parquet(tmp_path / "000001.parquet", index=False)
    pd.DataFrame({
        "symbol": ["000002"],
        "report_date": pd.to_datetime(["2024-01-01"]),
        "holder_count": [200],
        "holder_count_qoq": [float("nan")],
        "avg_holding_value": [float("nan")],
        "avg_float_per_holder": [float("nan")],
    }).to_parquet(tmp_path / "000002.parquet", index=False)
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02"]), ["000001", "000002"]],
        names=["date", "symbol"],
    )
    out = attach_to_panel(pd.DataFrame(index=idx), cache_dir=tmp_path)
    assert out.loc[(pd.Timestamp("2024-01-02"), "000001"), "shareholders"] == 100.0
    assert out.loc[(pd.Timestamp("2024-01-02"), "000002"), "shareholders"] == 200.0


def test_cache_round_trip_update_then_load(monkeypatch, tmp_path):
    expected = pd.DataFrame({
        "symbol": ["000001", "000001"],
        "report_date": pd.to_datetime(["2024-03-31", "2024-06-30"]),
        "holder_count": [1000, 800],
        "holder_count_qoq": [float("nan"), -0.2],
        "avg_holding_value": [1.0, 2.0],
        "avg_float_per_holder": [float("nan"), float("nan")],
    })
    monkeypatch.setattr(sh, "fetch_shareholders", lambda symbol: expected)

    update_cache(["000001"], cache_dir=tmp_path)
    loaded = load_shareholders(["000001"], cache_dir=tmp_path)

    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), expected)


def test_incremental_update_skips_existing_symbol(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_fetch(symbol):
        calls["count"] += 1
        return pd.DataFrame({
            "symbol": [sh.normalize_symbol(symbol)],
            "report_date": pd.to_datetime(["2024-03-31"]),
            "holder_count": [1000],
            "holder_count_qoq": [float("nan")],
            "avg_holding_value": [float("nan")],
            "avg_float_per_holder": [float("nan")],
        })

    monkeypatch.setattr(sh, "fetch_shareholders", fake_fetch)
    update_cache(["000001"], cache_dir=tmp_path)
    update_cache(["000001"], cache_dir=tmp_path)

    assert calls["count"] == 1


def test_holder_count_qoq_computation_known_series():
    raw = pd.DataFrame({
        "日期": ["2024-03-31", "2024-06-30", "2024-09-30"],
        "股东户数": [100, 120, 90],
    })
    df = normalize_shareholders(raw, symbol="000001")
    assert pd.isna(df.loc[0, "holder_count_qoq"])
    assert math.isclose(df.loc[1, "holder_count_qoq"], 0.2)
    assert math.isclose(df.loc[2, "holder_count_qoq"], -0.25)


def test_empty_and_one_row_edge_cases(tmp_path):
    assert normalize_shareholders(pd.DataFrame(), symbol="000001").empty
    one = normalize_shareholders(pd.DataFrame({"日期": ["2024-03-31"], "股东户数": [100]}), symbol="000001")
    assert len(one) == 1
    one.to_parquet(tmp_path / "000001.parquet", index=False)
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-03-30", "2024-03-31"]), ["000001"]], names=["date", "symbol"]
    )
    out = attach_to_panel(pd.DataFrame(index=idx), cache_dir=tmp_path)
    assert pd.isna(out.iloc[0]["shareholders"])
    assert out.iloc[1]["shareholders"] == 100.0


def test_attach_to_panel_handles_prefixed_panel_symbols(tmp_path):
    pd.DataFrame({
        "symbol": ["000001"],
        "report_date": pd.to_datetime(["2024-01-01"]),
        "holder_count": [321],
        "holder_count_qoq": [float("nan")],
        "avg_holding_value": [float("nan")],
        "avg_float_per_holder": [float("nan")],
    }).to_parquet(tmp_path / "000001.parquet", index=False)
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02"]), ["sz000001"]], names=["date", "symbol"]
    )
    out = attach_to_panel(pd.DataFrame(index=idx), cache_dir=tmp_path)
    assert out.loc[(pd.Timestamp("2024-01-02"), "sz000001"), "shareholders"] == 321.0


def test_attached_shareholders_make_f003_use_primary_branch(tmp_path):
    reports = pd.DataFrame({
        "symbol": ["000001", "000001"],
        "report_date": pd.to_datetime(["2024-01-01", "2024-03-02"]),
        "holder_count": [100, 80],
        "holder_count_qoq": [float("nan"), -0.2],
        "avg_holding_value": [float("nan"), float("nan")],
        "avg_float_per_holder": [float("nan"), float("nan")],
    })
    reports.to_parquet(tmp_path / "000001.parquet", index=False)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=65, freq="D"), ["000001"]],
        names=["date", "symbol"],
    )
    attached = attach_to_panel(pd.DataFrame(index=idx), cache_dir=tmp_path)

    factor = shareholder_concentration(attached)

    expected = -attached["shareholders"].groupby(level="symbol", group_keys=False).pct_change(60)
    pd.testing.assert_series_equal(factor, expected)
    assert math.isclose(factor.dropna().iloc[-1], 0.2)


def test_live_akshare_smoke_for_000001():
    pytest.importorskip("akshare")
    try:
        df = fetch_shareholders("000001")
    except Exception as exc:
        pytest.skip(f"akshare/network unavailable: {exc}")
    assert set(SCHEMA_COLUMNS).issubset(df.columns)
    if df.empty:
        pytest.skip("akshare returned no shareholder rows for 000001")
    assert df["holder_count"].dropna().gt(0).all()
