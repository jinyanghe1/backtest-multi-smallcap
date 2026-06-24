"""Unified data-provider facade with deterministic fallback semantics."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import legacy

logger = logging.getLogger(__name__)


PROVIDERS = {
    "price": {"primary": "westock", "fallback": ["akshare"], "timeout": 30},
    "quote": {"primary": "westock", "fallback": ["akshare", "sina"], "timeout": 10},
    "financials": {"primary": "eastmoney", "fallback": ["ths", "sina"], "timeout": 60},
    "fundamental": {"primary": "eastmoney", "fallback": ["ths", "sina", "akshare"], "timeout": 60},
    "industry": {"primary": "akshare", "fallback": ["eastmoney", "ths"], "timeout": 30},
}


class ProviderError(Exception):
    """Base class for provider-layer failures."""


class NetworkError(ProviderError):
    """Retryable network failure."""


class RateLimitError(ProviderError):
    """Provider rate limit; retry after a longer delay."""


class FieldNotFoundError(ProviderError):
    """Field/category is unsupported by a provider."""


class DataValidationError(ProviderError):
    """Provider returned data, but it failed local validation."""


class AllProvidersFailedError(ProviderError):
    """Raised when every configured provider fails."""

    def __init__(self, category: str, symbol: str, field: str, errors: dict[str, str]):
        self.category = category
        self.symbol = symbol
        self.field = field
        self.errors = errors
        super().__init__(f"All providers failed for {category}/{symbol}/{field}: {errors}")


class BatchFetchError(ProviderError):
    """Raised when one or more symbols fail during batch_get.

    Attributes:
        failures: {symbol: AllProvidersFailedError}
        partial_results: {symbol: FetchResult} for successful fetches
    """

    def __init__(self, failures: dict, partial_results: dict | None = None):
        self.failures = failures
        self.partial_results = partial_results or {}
        symbols = list(failures.keys())
        super().__init__(f"Batch fetch failed for {len(failures)} symbol(s): {symbols}")


@dataclass(frozen=True)
class FetchResult:
    data: pd.Series | pd.DataFrame
    source: str
    category: str
    field: str
    symbol: str
    timestamp: pd.Timestamp
    metadata: dict = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        if self.data is None or self.data.empty:
            return False
        if isinstance(self.data, pd.Series):
            return not self.data.isna().all()
        return not self.data.isna().all().all()


class WestockClient:
    def fetch_price(self, symbol: str, field: str, **kwargs) -> pd.Series:
        days = kwargs.pop("days", kwargs.pop("limit", 1500))
        df = legacy.fetch_stock_kline(symbol, days=days)
        if field not in df.columns:
            raise FieldNotFoundError(f"westock price field not found: {field}")
        series = df.set_index("date")[field]
        return _slice_date_range(series, kwargs.get("start"), kwargs.get("end"))

    def fetch_quote(self, symbol: str, field: str, **kwargs) -> pd.Series:
        df = legacy.fetch_stock_quote([symbol])
        if df.empty or field not in df.columns:
            raise FieldNotFoundError(f"westock quote field not found: {field}")
        if "symbol" in df.columns:
            row = df[df["symbol"] == symbol]
            if row.empty:
                raise FieldNotFoundError(f"westock quote symbol not found: {symbol}")
            value = row.iloc[0][field]
        else:
            value = df.iloc[0][field]
        return pd.Series([value], index=[pd.Timestamp.now().normalize()], name=field)


class FinancialScriptClient:
    def __init__(self, source: str):
        self.source = source

    def fetch_financials(self, symbol: str, field: str, **kwargs) -> pd.Series:
        try:
            from tools.backtest_mvp import fetch_missing_financials as financials
        except ModuleNotFoundError:
            import fetch_missing_financials as financials

        func = {
            "eastmoney": financials.fetch_em,
            "em": financials.fetch_em,
            "ths": financials.fetch_ths,
            "sina": financials.fetch_sina,
        }.get(self.source)
        if func is None:
            raise FieldNotFoundError(f"unknown financial source: {self.source}")
        df = func(symbol)
        if df is None or df.empty or field not in df.columns:
            raise FieldNotFoundError(f"{self.source} financial field not found: {field}")
        index_col = "notice_date" if "notice_date" in df.columns else "report_date"
        series = df.set_index(pd.to_datetime(df[index_col]))[field].sort_index()
        return _slice_date_range(series, kwargs.get("start"), kwargs.get("end"))

    def fetch_fundamental(self, symbol: str, field: str, **kwargs) -> pd.Series:
        return self.fetch_financials(symbol, field, **kwargs)


class AkshareClient:
    """AkShare-backed fundamental fields.

    The underlying function lazy-loads akshare, so importing this provider stays
    cheap and unit tests can monkeypatch the fetch function without network use.
    """

    FIELD_ALIASES = {
        "roe_q": "roe",
        "gross_margin_q": "gross_margin",
        "net_profit_growth_q": "net_profit_growth",
        "revenue_growth_q": "revenue_growth",
    }

    def fetch_fundamental(self, symbol: str, field: str, **kwargs) -> pd.Series:
        try:
            from tools.backtest_mvp import financials_akshare
        except ModuleNotFoundError:
            import financials_akshare

        source_field = self.FIELD_ALIASES.get(field, field)
        df = financials_akshare.fetch_quarterly_financials(
            symbol,
            cache=kwargs.get("cache", True),
            refresh=kwargs.get("refresh", False),
        )
        if df is None or df.empty or source_field not in df.columns:
            raise FieldNotFoundError(f"akshare fundamental field not found: {field}")
        if "date" not in df.columns:
            raise DataValidationError("akshare fundamental data missing date column")
        series = df.set_index(pd.to_datetime(df["date"]))[source_field].sort_index()
        series.name = field
        return _slice_date_range(series, kwargs.get("start"), kwargs.get("end"))

    def fetch_financials(self, symbol: str, field: str, **kwargs) -> pd.Series:
        return self.fetch_fundamental(symbol, field, **kwargs)


def _slice_date_range(series: pd.Series, start=None, end=None) -> pd.Series:
    out = series.sort_index()
    if start is not None:
        out = out[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out[out.index <= pd.Timestamp(end)]
    return out


class DataProvider:
    """Unified data access with provider fallback and explicit failure reporting."""

    def __init__(
        self,
        config: Optional[dict] = None,
        clients: Optional[dict] = None,
        retry_delays: tuple[float, float, float] = (3.0, 6.0, 12.0),
        rate_limit_delay: float = 60.0,
    ):
        self.config = config or PROVIDERS
        self.retry_delays = retry_delays
        self.rate_limit_delay = rate_limit_delay
        self.clients = clients or self._default_clients()

    def _default_clients(self) -> dict:
        return {
            "westock": WestockClient(),
            "eastmoney": FinancialScriptClient("eastmoney"),
            "em": FinancialScriptClient("em"),
            "ths": FinancialScriptClient("ths"),
            "sina": FinancialScriptClient("sina"),
            "akshare": AkshareClient(),
        }

    def get(
        self,
        category: str,
        symbol: str,
        field: str,
        start: Optional[pd.Timestamp | str] = None,
        end: Optional[pd.Timestamp | str] = None,
        asof: Optional[pd.Timestamp | str] = None,
        **kwargs,
    ) -> FetchResult:
        if category not in self.config:
            raise FieldNotFoundError(f"unknown category: {category}")

        provider_cfg = self.config[category]
        providers = [provider_cfg["primary"], *provider_cfg.get("fallback", [])]
        errors: dict[str, str] = {}
        fetch_kwargs = dict(kwargs)
        fetch_kwargs.update({"start": start, "end": end, "asof": asof})

        for provider_name in providers:
            client = self.clients.get(provider_name)
            if client is None:
                errors[provider_name] = "client not configured"
                continue

            for attempt in range(3):
                try:
                    data = self._fetch(client, category, symbol, field, **fetch_kwargs)
                    if not self._validate(data, category, field):
                        raise DataValidationError(f"invalid data for {category}/{field}")
                    return FetchResult(
                        data=data,
                        source=provider_name,
                        category=category,
                        field=field,
                        symbol=symbol,
                        timestamp=pd.Timestamp.now(),
                        metadata={"attempt": attempt + 1},
                    )
                except NetworkError as exc:
                    errors[provider_name] = str(exc)
                    logger.warning("[%s] network error attempt %s: %s", provider_name, attempt + 1, exc)
                    if attempt < 2:
                        time.sleep(self.retry_delays[min(attempt, len(self.retry_delays) - 1)])
                        continue
                    break
                except RateLimitError as exc:
                    errors[provider_name] = str(exc)
                    logger.warning("[%s] rate limited: %s", provider_name, exc)
                    if attempt < 2:
                        time.sleep(self.rate_limit_delay)
                        continue
                    break
                except FieldNotFoundError as exc:
                    errors[provider_name] = str(exc)
                    break
                except DataValidationError as exc:
                    errors[provider_name] = str(exc)
                    break
                except Exception as exc:
                    errors[provider_name] = f"{type(exc).__name__}: {exc}"
                    logger.exception("[%s] unexpected provider error", provider_name)
                    break

        raise AllProvidersFailedError(category, symbol, field, errors)

    def get_series(self, category: str, symbol: str, field: str, **kwargs) -> pd.Series:
        result = self.get(category, symbol, field, **kwargs)
        if isinstance(result.data, pd.Series):
            return result.data
        if result.data.shape[1] == 1:
            return result.data.iloc[:, 0]
        raise DataValidationError("get_series received multi-column DataFrame")

    def batch_get(
        self,
        symbols: list[str],
        category: str,
        field: str,
        **kwargs,
    ) -> dict[str, "FetchResult"]:
        """Fetch data for multiple symbols.

        Returns a dict mapping symbol -> FetchResult for successful fetches.
        Symbols that fail on ALL providers raise AllProvidersFailedError,
        which is collected and re-raised as a BatchError containing all failures.
        """
        results: dict[str, FetchResult] = {}
        failures: dict[str, AllProvidersFailedError] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get(category, symbol, field, **kwargs)
            except AllProvidersFailedError as exc:
                failures[symbol] = exc
        if failures:
            raise BatchFetchError(failures, partial_results=results)
        return results

    def _fetch(self, client, category: str, symbol: str, field: str, **kwargs) -> pd.Series | pd.DataFrame:
        method = getattr(client, f"fetch_{category}", None)
        if method is not None:
            return method(symbol, field, **kwargs)
        if category == "price" and hasattr(client, "fetch_kline"):
            df = client.fetch_kline(symbol, **kwargs)
            if field not in df.columns:
                raise FieldNotFoundError(f"{field} not found")
            return df.set_index("date")[field]
        raise FieldNotFoundError(f"{client.__class__.__name__} does not support {category}/{field}")

    def _validate(self, data: pd.Series | pd.DataFrame, category: str, field: str) -> bool:
        if data is None or data.empty:
            return False
        if isinstance(data, pd.Series):
            if data.isna().all():
                return False
            return True
        if data.isna().all().all():
            return False
        return True
