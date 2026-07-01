"""Shareholder-count ingestion, cache persistence, and as-of panel merge.

Normalized shareholder schema (one row per stock/report date):

``symbol``
    Six-digit A-share stock code without exchange prefix, e.g. ``"000001"``.
``report_date``
    ``pd.Timestamp`` for the shareholder-count statistic/report cutoff date
    (AkShare columns such as ``股东户数统计截止日``/``截止日``/``日期``).
``holder_count``
    Integer number of shareholder accounts/households (``股东户数``).
``holder_count_qoq``
    Float ratio change versus the previous report for the same symbol, i.e.
    ``current / previous - 1``. Provider percentage columns are normalized to a
    ratio and only used when a previous report is unavailable.
``avg_holding_value``
    Float average market value held per shareholder account, in CNY, when
    AkShare provides ``户均持股市值``; otherwise ``NaN``.
``avg_float_per_holder``
    Float average shares/float shares held per shareholder account, in shares,
    when AkShare provides ``户均流通股``/``户均持股数量``; otherwise ``NaN``.

Cache convention: per-symbol parquet files under ``backtest_mvp/shareholders_cache``
(``{six_digit_symbol}.parquet``), matching the repo's existing per-symbol parquet
cache style. ``update_cache`` is incremental: an existing non-empty parquet file
is considered fresh and skipped unless ``refresh=True`` is passed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / "shareholders_cache"

SCHEMA_COLUMNS = [
    "symbol",
    "report_date",
    "holder_count",
    "holder_count_qoq",
    "avg_holding_value",
    "avg_float_per_holder",
]

_DATE_COLUMNS = [
    "report_date",
    "end_date",
    "date",
    "日期",
    "截止日",
    "统计截止日",
    "股东户数统计截止日",
    "股东户数统计截止日-本次",
]
_HOLDER_COLUMNS = ["holder_count", "shareholder_count", "股东户数", "股东户数-本次"]
_QOQ_COLUMNS = [
    "holder_count_qoq",
    "change_rate",
    "change_rate_pct",
    "股东户数-增减比例",
    "较上期变化",
]
_AVG_VALUE_COLUMNS = ["avg_holding_value", "avg_market_value", "户均持股市值"]
_AVG_FLOAT_COLUMNS = [
    "avg_float_per_holder",
    "avg_shares_per_household",
    "户均流通股",
    "户均持股数量",
]
_SYMBOL_COLUMNS = ["symbol", "代码", "股票代码", "证券代码"]


def normalize_symbol(symbol: str) -> str:
    """Return AkShare's six-digit A-share code, stripping sh/sz/bj prefixes."""
    text = str(symbol).strip().lower()
    if text.startswith(("sh", "sz", "bj")):
        text = text[2:]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"invalid stock symbol: {symbol!r}")
    return digits.zfill(6)[-6:]


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _parse_number(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in {"", "--", "-", "nan", "None"}:
        return np.nan
    multiplier = 1.0
    if text.endswith("亿"):
        multiplier = 100_000_000.0
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 10_000.0
        text = text[:-1]
    text = text.replace("%", "")
    return pd.to_numeric(text, errors="coerce") * multiplier


def _to_numeric(series: pd.Series) -> pd.Series:
    return series.map(_parse_number).astype("float64")


def _empty_schema() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": pd.Series(dtype="object"),
        "report_date": pd.Series(dtype="datetime64[ns]"),
        "holder_count": pd.Series(dtype="int64"),
        "holder_count_qoq": pd.Series(dtype="float64"),
        "avg_holding_value": pd.Series(dtype="float64"),
        "avg_float_per_holder": pd.Series(dtype="float64"),
    })


def _call_akshare_shareholders(symbol: str) -> pd.DataFrame:
    """Thin AkShare call isolated for monkeypatching in tests."""
    import akshare as ak

    code = normalize_symbol(symbol)
    detail = getattr(ak, "stock_zh_a_gdhs_detail_em", None)
    if detail is not None:
        return detail(symbol=code)
    return ak.stock_zh_a_gdhs(symbol=code)


def fetch_shareholders(symbol: str) -> pd.DataFrame:
    """Fetch and normalize shareholder-count history for one A-share symbol."""
    raw = _call_akshare_shareholders(symbol)
    return normalize_shareholders(raw, symbol=symbol)


def normalize_shareholders(raw: pd.DataFrame | None, symbol: str | None = None) -> pd.DataFrame:
    """Normalize AkShare shareholder DataFrames to ``SCHEMA_COLUMNS``."""
    if raw is None or raw.empty:
        return _empty_schema()

    df = raw.copy()
    code = normalize_symbol(symbol) if symbol is not None else None

    symbol_col = _first_existing(df.columns, _SYMBOL_COLUMNS)
    if symbol_col is not None:
        symbols = df[symbol_col].map(normalize_symbol)
    elif code is not None:
        symbols = pd.Series(code, index=df.index)
    else:
        raise ValueError("shareholder data has no symbol column and no symbol argument")

    date_col = _first_existing(df.columns, _DATE_COLUMNS)
    holder_col = _first_existing(df.columns, _HOLDER_COLUMNS)
    if date_col is None or holder_col is None:
        raise ValueError(
            "shareholder data missing required date/holder columns; "
            f"columns={list(df.columns)!r}"
        )

    out = pd.DataFrame({
        "symbol": symbols.astype(str),
        "report_date": pd.to_datetime(df[date_col], errors="coerce"),
        "holder_count": _to_numeric(df[holder_col]),
    })

    avg_value_col = _first_existing(df.columns, _AVG_VALUE_COLUMNS)
    avg_float_col = _first_existing(df.columns, _AVG_FLOAT_COLUMNS)
    out["avg_holding_value"] = _to_numeric(df[avg_value_col]) if avg_value_col else np.nan
    out["avg_float_per_holder"] = _to_numeric(df[avg_float_col]) if avg_float_col else np.nan

    qoq_col = _first_existing(df.columns, _QOQ_COLUMNS)
    provider_qoq = _to_numeric(df[qoq_col]) if qoq_col else pd.Series(np.nan, index=df.index)
    valid = provider_qoq.dropna()
    if not valid.empty and valid.abs().median() > 1.5:
        provider_qoq = provider_qoq / 100.0
    out["_provider_qoq"] = provider_qoq

    out = out.dropna(subset=["symbol", "report_date", "holder_count"])
    if out.empty:
        return _empty_schema()

    out = (
        out.sort_values(["symbol", "report_date"])
        .drop_duplicates(["symbol", "report_date"], keep="last")
        .reset_index(drop=True)
    )
    computed = out.groupby("symbol", group_keys=False)["holder_count"].pct_change()
    out["holder_count_qoq"] = computed.combine_first(out["_provider_qoq"])
    out["holder_count"] = out["holder_count"].round().astype("int64")

    return out[SCHEMA_COLUMNS].sort_values(["symbol", "report_date"]).reset_index(drop=True)


def _cache_path(cache_dir: str | Path, symbol: str) -> Path:
    return Path(cache_dir) / f"{normalize_symbol(symbol)}.parquet"


def _candidate_cache_paths(cache_dir: str | Path, symbol: str) -> list[Path]:
    code = normalize_symbol(symbol)
    base = Path(cache_dir)
    return [base / f"{code}.parquet", base / f"sh{code}.parquet", base / f"sz{code}.parquet", base / f"bj{code}.parquet"]


def update_cache(symbols: list[str], cache_dir: str | Path = DEFAULT_CACHE_DIR, *, refresh: bool = False) -> None:
    """Fetch missing shareholder histories into per-symbol parquet files.

    Freshness rule: a non-empty existing parquet file for a symbol is considered
    fresh and skipped. Pass ``refresh=True`` to overwrite it.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        path = _cache_path(cache, symbol)
        if not refresh and path.exists() and path.stat().st_size > 0:
            continue
        df = fetch_shareholders(symbol)
        df.to_parquet(path, index=False)


def _coerce_loaded_schema(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    if set(SCHEMA_COLUMNS).issubset(df.columns):
        out = df[SCHEMA_COLUMNS].copy()
        out["symbol"] = out["symbol"].map(normalize_symbol)
        out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
        out["holder_count"] = pd.to_numeric(out["holder_count"], errors="coerce")
        out["holder_count_qoq"] = pd.to_numeric(out["holder_count_qoq"], errors="coerce")
        out["avg_holding_value"] = pd.to_numeric(out["avg_holding_value"], errors="coerce")
        out["avg_float_per_holder"] = pd.to_numeric(out["avg_float_per_holder"], errors="coerce")
        out = out.dropna(subset=["symbol", "report_date", "holder_count"])
        if out.empty:
            return _empty_schema()
        out["holder_count"] = out["holder_count"].round().astype("int64")
        return out[SCHEMA_COLUMNS]
    return normalize_shareholders(df, symbol=symbol)


def load_shareholders(symbols: Sequence[str] | None = None, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Load cached shareholder histories for ``symbols`` from parquet files."""
    cache = Path(cache_dir)
    if symbols is None:
        paths = sorted(cache.glob("*.parquet")) if cache.exists() else []
        symbol_for_path = {path: path.stem for path in paths}
    else:
        paths = []
        symbol_for_path = {}
        for symbol in symbols:
            for path in _candidate_cache_paths(cache, symbol):
                if path.exists():
                    paths.append(path)
                    symbol_for_path[path] = symbol
                    break

    frames = []
    for path in paths:
        df = pd.read_parquet(path)
        frames.append(_coerce_loaded_schema(df, symbol_for_path.get(path)))
    if not frames:
        return _empty_schema()

    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        return _empty_schema()
    out = out.drop_duplicates(["symbol", "report_date"], keep="last")
    return out.sort_values(["symbol", "report_date"]).reset_index(drop=True)


def attach_to_panel(panel: pd.DataFrame, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Attach latest non-future shareholder count as ``shareholders``.

    ``panel`` must be indexed by a ``(date, symbol)`` MultiIndex. For each row,
    the attached value is the most recent ``holder_count`` with
    ``report_date <= date`` within the same normalized symbol.
    """
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels < 2:
        raise ValueError("panel must have a MultiIndex with date and symbol levels")

    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    symbol_level = "symbol" if "symbol" in panel.index.names else panel.index.names[1]

    out = panel.copy()
    rows = out.index.to_frame(index=False)
    dates = pd.to_datetime(rows[date_level], errors="coerce")
    original_symbols = rows[symbol_level].astype(str)
    codes = original_symbols.map(normalize_symbol)
    holder_values = pd.Series(np.nan, index=np.arange(len(out)), dtype="float64")

    histories = load_shareholders(sorted(codes.unique()), cache_dir=cache_dir)
    if histories.empty:
        out["shareholders"] = holder_values.to_numpy(dtype="float64")
        return out

    work = pd.DataFrame({"_row_id": np.arange(len(out)), "_date": dates, "_code": codes})
    for code, left in work.dropna(subset=["_date"]).groupby("_code", sort=False):
        right = histories[histories["symbol"] == code][["report_date", "holder_count"]].sort_values("report_date")
        if right.empty:
            continue
        merged = pd.merge_asof(
            left.sort_values("_date"),
            right,
            left_on="_date",
            right_on="report_date",
            direction="backward",
            allow_exact_matches=True,
        )
        holder_values.loc[merged["_row_id"].to_numpy()] = merged["holder_count"].astype("float64").to_numpy()

    out["shareholders"] = holder_values.to_numpy(dtype="float64")
    return out
