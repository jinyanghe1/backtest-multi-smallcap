import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from tools.backtest_mvp.data.providers import (
    AllProvidersFailedError,
    AkshareClient,
    BatchFetchError,
    DataProvider,
    FieldNotFoundError,
    NetworkError,
)


class GoodClient:
    def fetch_price(self, symbol, field, **kwargs):
        return pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2), name=field)


class MissingClient:
    def fetch_price(self, symbol, field, **kwargs):
        raise FieldNotFoundError("missing")


class FlakyClient:
    def __init__(self):
        self.calls = 0

    def fetch_price(self, symbol, field, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise NetworkError("temporary")
        return pd.Series([1.0], index=[pd.Timestamp("2024-01-01")], name=field)


def test_provider_fallback_to_second_client():
    provider = DataProvider(
        config={"price": {"primary": "bad", "fallback": ["good"]}},
        clients={"bad": MissingClient(), "good": GoodClient()},
        retry_delays=(0, 0, 0),
        rate_limit_delay=0,
    )
    result = provider.get("price", "sh600000", "close")
    assert result.source == "good"
    assert result.data.iloc[-1] == 2.0


def test_provider_retries_network_error():
    flaky = FlakyClient()
    provider = DataProvider(
        config={"price": {"primary": "flaky", "fallback": []}},
        clients={"flaky": flaky},
        retry_delays=(0, 0, 0),
        rate_limit_delay=0,
    )
    result = provider.get("price", "sh600000", "close")
    assert result.source == "flaky"
    assert flaky.calls == 2


def test_provider_all_failed_error():
    provider = DataProvider(
        config={"price": {"primary": "bad", "fallback": []}},
        clients={"bad": MissingClient()},
        retry_delays=(0, 0, 0),
        rate_limit_delay=0,
    )
    try:
        provider.get("price", "sh600000", "close")
    except AllProvidersFailedError as exc:
        assert "bad" in exc.errors
    else:
        raise AssertionError("expected AllProvidersFailedError")


def test_akshare_client_fetches_fundamental(monkeypatch):
    from tools.backtest_mvp import financials_akshare

    def fake_fetch_quarterly_financials(symbol, cache=True, refresh=False):
        assert symbol == "sh600000"
        assert cache is False
        assert refresh is True
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-03-31", "2024-06-30"]),
            "roe": [8.0, 9.0],
        })

    monkeypatch.setattr(financials_akshare, "fetch_quarterly_financials", fake_fetch_quarterly_financials)
    series = AkshareClient().fetch_fundamental("sh600000", "roe_q", cache=False, refresh=True)

    assert series.name == "roe_q"
    assert series.iloc[-1] == 9.0


def test_data_provider_can_fallback_to_default_akshare(monkeypatch):
    from tools.backtest_mvp import financials_akshare

    def fake_fetch_quarterly_financials(symbol, cache=True, refresh=False):
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-03-31"]),
            "gross_margin": [30.5],
        })

    monkeypatch.setattr(financials_akshare, "fetch_quarterly_financials", fake_fetch_quarterly_financials)
    provider = DataProvider(
        config={"fundamental": {"primary": "missing", "fallback": ["akshare"]}},
        clients={"missing": MissingClient(), "akshare": AkshareClient()},
        retry_delays=(0, 0, 0),
        rate_limit_delay=0,
    )
    result = provider.get("fundamental", "sh600000", "gross_margin")

    assert result.source == "akshare"
    assert result.data.iloc[0] == 30.5


def test_batch_get_all_succeed():
    provider = DataProvider(
        config={"price": {"primary": "good", "fallback": []}},
        clients={"good": GoodClient()},
        retry_delays=(0, 0, 0),
        rate_limit_delay=0,
    )
    results = provider.batch_get(["sh600000", "sz000001"], "price", "close")
    assert set(results.keys()) == {"sh600000", "sz000001"}
    assert results["sh600000"].data.iloc[-1] == 2.0


def test_batch_get_partial_failure_raises_batch_error():
    class SelectiveClient:
        def fetch_price(self, symbol, field, **kwargs):
            if symbol == "bad":
                raise FieldNotFoundError("no data")
            return pd.Series([1.0], index=[pd.Timestamp("2024-01-01")], name=field)

    provider = DataProvider(
        config={"price": {"primary": "sel", "fallback": []}},
        clients={"sel": SelectiveClient()},
        retry_delays=(0, 0, 0),
        rate_limit_delay=0,
    )
    try:
        provider.batch_get(["good", "bad"], "price", "close")
    except BatchFetchError as exc:
        assert "bad" in exc.failures
        assert "good" in exc.partial_results
    else:
        raise AssertionError("expected BatchFetchError")
