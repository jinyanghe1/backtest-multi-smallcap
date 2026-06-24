import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from tools.backtest_mvp.data.field_resolver import detect_conflict, resolve


def test_detect_conflict_threshold():
    idx = pd.date_range("2024-01-01", periods=3)
    em = pd.Series([100.0, 101.0, 102.0], index=idx)
    ths_close = pd.Series([100.5, 101.5, 102.5], index=idx)
    ths_far = pd.Series([120.0, 121.0, 122.0], index=idx)

    assert not detect_conflict({"eastmoney": em, "ths": ths_close}, threshold=0.01).has_conflict
    report = detect_conflict({"eastmoney": em, "ths": ths_far}, threshold=0.01)
    assert report.has_conflict
    assert "eastmoney:ths" in report.conflicts


def test_resolve_priority_and_gap_fill():
    idx = pd.date_range("2024-01-01", periods=3)
    em = pd.Series([1.0, None, 3.0], index=idx)
    ths = pd.Series([10.0, 2.0, 30.0], index=idx)

    result = resolve("eps", {"ths": ths, "eastmoney": em}, threshold=10.0)
    assert result.source == "eastmoney"
    assert result.data.loc[idx[1]] == 2.0
    assert result.metadata["fallback_sources"] == ["ths"]


def test_resolve_ratio_percent_normalisation():
    idx = pd.date_range("2024-01-01", periods=3)
    roe = pd.Series([10.0, 20.0, 30.0], index=idx)

    result = resolve("roe_ttm", {"eastmoney": roe})
    assert result.data.iloc[-1] == 0.30

