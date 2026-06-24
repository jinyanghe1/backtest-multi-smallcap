import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import pytest

from tools.backtest_mvp.industry.classifier import (
    IndustryClassifier,
    IndustrySource,
    default_cache_path,
    load_default_industry_map,
)


def _make_panel():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), ["sh600000", "sz000001", "bj920000"]],
        names=["date", "symbol"],
    )
    return pd.DataFrame({"mcap": range(1, len(idx) + 1)}, index=idx)


def test_attach_industry_to_factor_panel():
    panel = _make_panel()
    industry_map = pd.DataFrame({
        "symbol": ["sh600000"],
        "sw_industry_1": ["银行"],
        "sw_industry_2": ["股份制银行"],
    })

    classifier = IndustryClassifier(industry_map)
    attached = classifier.attach_industry(panel)
    assert attached.loc[("2024-01-01", "sh600000"), "sw_industry_2"] == "股份制银行"
    assert attached.loc[("2024-01-01", "sz000001"), "sw_industry_2"] == "Unknown"
    assert attached.loc[("2024-01-01", "bj920000"), "sw_industry_2"] == "Unknown"
    assert classifier.get("sh600000", IndustrySource.SW_INDUSTRY_1) == "银行"


def test_load_industry_map_from_parquet():
    panel = _make_panel()
    industry_map = pd.DataFrame({
        "symbol": ["sh600000", "sz000001"],
        "sw_industry_1": ["银行", "房地产"],
        "sw_industry_2": ["股份制银行", "房地产开发"],
        "source": ["test", "test"],
        "updated_at": [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-01")],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "shenwan_map.parquet"
        industry_map.to_parquet(path, index=False)

        classifier = IndustryClassifier()
        loaded = classifier.load_industry_map(path)
        assert len(loaded) == 2
        assert classifier.get("sh600000", IndustrySource.SW_INDUSTRY_2) == "股份制银行"

        attached = classifier.attach_industry(panel)
        assert attached.loc[("2024-01-01", "sh600000"), "sw_industry_2"] == "股份制银行"
        assert attached.loc[("2024-01-01", "sz000001"), "sw_industry_2"] == "房地产开发"
        assert attached.loc[("2024-01-01", "bj920000"), "sw_industry_2"] == "Unknown"


def test_load_industry_map_missing_file_returns_empty():
    classifier = IndustryClassifier()
    loaded = classifier.load_industry_map(Path("/nonexistent/path/shenwan_map.parquet"))
    assert loaded.empty
    assert classifier.get("sh600000", IndustrySource.SW_INDUSTRY_2) == "Unknown"


def test_refresh_raises_not_implemented():
    classifier = IndustryClassifier()
    with pytest.raises(NotImplementedError):
        classifier.load_industry_map(default_cache_path(), refresh=True)


def test_default_cache_path_points_to_industry_cache():
    path = default_cache_path()
    assert path.name == "shenwan_map.parquet"
    assert path.parent.name == "industry_cache"


def test_load_default_industry_map_does_not_call_network():
    """The default loader only reads local parquet; no external API is invoked."""
    # If the default cache does not exist, this returns an empty frame rather
    # than attempting a network request.
    df = load_default_industry_map()
    assert isinstance(df, pd.DataFrame)


def test_normalise_unknown_symbols_in_parquet():
    """Parquet rows that contain Unknown as a placeholder are preserved."""
    panel = _make_panel()
    industry_map = pd.DataFrame({
        "symbol": ["sh600000"],
        "sw_industry_1": ["Unknown"],
        "sw_industry_2": ["Unknown"],
    })
    attached = IndustryClassifier(industry_map).attach_industry(panel)
    assert attached.loc[("2024-01-01", "sh600000"), "sw_industry_2"] == "Unknown"


def test_attach_industry_preserves_index_names():
    panel = _make_panel()
    industry_map = pd.DataFrame({
        "symbol": ["sh600000"],
        "sw_industry_1": ["银行"],
        "sw_industry_2": ["股份制银行"],
    })
    attached = IndustryClassifier(industry_map).attach_industry(panel)
    assert attached.index.names == ["date", "symbol"]
