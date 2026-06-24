"""Build the cache-only Shenwan industry mapping parquet.

This module is allowed to touch external APIs when run explicitly as a script.
The runtime classifier (`industry.classifier`) remains cache-only and must
never call out to external APIs inside a backtest loop.

Example:
    python -m tools.backtest_mvp.industry.build_cache
    python -m tools.backtest_mvp.industry.build_cache --limit 50
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT / "data_cache"
DEFAULT_OUTPUT_PATH = PROJECT / "industry_cache" / "shenwan_map.parquet"
SW_STANDARD = "申银万国行业分类标准"
CNINFO_URL = "https://webapi.cninfo.com.cn/api/stock/p_stock2110"
CNINFO_START_DATE = "2009-12-27"
CNINFO_END_DATE = "2022-07-13"


def _cninfo_headers() -> dict:
    """Build the authenticated headers for the cninfo API."""
    # Lazy imports: avoid heavy akshare/py_mini_racer startup on module import.
    from akshare.stock.stock_industry_cninfo import _get_file_content_ths
    from py_mini_racer import MiniRacer

    js_code = MiniRacer()
    js_code.eval(_get_file_content_ths("cninfo.js"))
    mcode = js_code.call("getResCode1")
    return {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Content-Length": "0",
        "Host": "webapi.cninfo.com.cn",
        "Accept-Enckey": mcode,
        "Origin": "https://webapi.cninfo.com.cn",
        "Pragma": "no-cache",
        "Proxy-Connection": "keep-alive",
        "Referer": "https://webapi.cninfo.com.cn/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
    }


def normalise_symbol(raw: str) -> str:
    """Convert a raw code into the project's symbol convention.

    Examples:
        600000  -> sh600000
        000001  -> sz000001
        688981  -> sh688981
        430047  -> bj430047   (北交所)
        sh600000 -> sh600000
    """
    raw = str(raw).strip().lower()
    if raw.startswith(("sh", "sz", "bj")):
        return raw
    if len(raw) == 6 and raw.isdigit():
        if raw.startswith(("60", "68", "900")):
            return f"sh{raw}"
        if raw.startswith(("00", "30", "200")):
            return f"sz{raw}"
        if raw.startswith(("43", "83", "87", "88", "92")):
            return f"bj{raw}"
    # Fallback: infer from numeric pattern
    if re.fullmatch(r"\d{6}", raw):
        if raw.startswith("6"):
            return f"sh{raw}"
        return f"sz{raw}"
    return raw


def fetch_sw_industry_for_code(
    code: str,
    headers: dict,
    retries: int = 2,
    sleep_seconds: float = 0.3,
    timeout: float = 30.0,
) -> Optional[tuple[str, str]]:
    """Return (sw_industry_1, sw_industry_2) for a numeric code via cninfo.

    Returns None when cninfo has no Shenwan classification for the code.
    Retries on transient network/JSON errors; raises only on persistent
    failures so the caller can decide whether to abort the whole build.
    """
    params = {
        "scode": code,
        "sdate": CNINFO_START_DATE,
        "edate": CNINFO_END_DATE,
    }
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(CNINFO_URL, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(sleep_seconds * (attempt + 1))
                continue
            raise RuntimeError(f"cninfo failed for {code} after {retries + 1} attempts: {exc}") from last_exc
        break

    records = data.get("records", [])
    if not records:
        return None

    sw_records = [rec for rec in records if rec.get("F002V") == SW_STANDARD]
    if not sw_records:
        return None

    latest = max(sw_records, key=lambda rec: rec.get("VARYDATE", ""))
    sw1 = latest.get("F004V", "")
    sw2 = latest.get("F005V", "")
    sw1 = str(sw1).strip() if sw1 is not None else ""
    sw2 = str(sw2).strip() if sw2 is not None else ""
    if not sw1 and not sw2:
        return None
    return sw1, sw2


def list_symbols_in_data_dir(data_dir: Path) -> list[str]:
    """Return all parquet stem symbols from data_cache."""
    files = sorted(data_dir.glob("*.parquet"))
    return [f.stem for f in files]


def _fetch_one(
    sym: str,
    headers: dict,
    retries: int = 2,
    sleep_seconds: float = 0.3,
) -> dict:
    """Fetch a single symbol and return a record dict."""
    code = sym[2:] if sym.startswith(("sh", "sz", "bj")) else sym
    try:
        result = fetch_sw_industry_for_code(
            code, headers=headers, retries=retries, sleep_seconds=sleep_seconds
        )
    except Exception as exc:
        print(f"  ✗ {sym}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return {
            "symbol": sym,
            "sw_industry_1": "Unknown",
            "sw_industry_2": "Unknown",
            "source": "cninfo_failed",
            "updated_at": pd.Timestamp.now(),
        }

    if result is None:
        return {
            "symbol": sym,
            "sw_industry_1": "Unknown",
            "sw_industry_2": "Unknown",
            "source": "cninfo_unavailable",
            "updated_at": pd.Timestamp.now(),
        }

    sw1, sw2 = result
    return {
        "symbol": sym,
        "sw_industry_1": sw1 or "Unknown",
        "sw_industry_2": sw2 or sw1 or "Unknown",
        "source": "cninfo_shenwan",
        "updated_at": pd.Timestamp.now(),
    }


def build_shenwan_map(
    symbols: list[str],
    output_path: Optional[Path] = None,
    progress_every: int = 50,
    max_workers: int = 10,
) -> pd.DataFrame:
    """Fetch Shenwan classification for symbols and write a parquet cache.

    Parameters
    ----------
    symbols: project-style symbols, e.g. sh600000, sz000001.
    output_path: where to write the parquet; defaults to
        ``industry_cache/shenwan_map.parquet``.
    progress_every: print progress every N symbols.
    max_workers: concurrent requests to cninfo.

    Returns
    -------
    DataFrame with columns [symbol, sw_industry_1, sw_industry_2, source, updated_at].
    """
    output_path = output_path or DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = _cninfo_headers()
    total = len(symbols)
    records: list[dict] = [{} for _ in symbols]
    completed_records: list[dict] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_fetch_one, sym, headers): i
            for i, sym in enumerate(symbols)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            record = future.result()
            records[i] = record
            completed_records.append(record)
            completed += 1
            if progress_every > 0 and completed % progress_every == 0:
                covered = sum(1 for r in completed_records if r["sw_industry_2"] != "Unknown")
                print(f"  progress: {completed}/{total} ({100 * completed / total:.1f}%) covered={covered}/{completed}")

    df = pd.DataFrame(records)
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    df.to_parquet(output_path, index=False)
    failed = df[df["source"] == "cninfo_failed"].shape[0]
    print(f"Saved {len(df)} rows to {output_path}")
    if failed:
        print(f"  {failed} symbols failed after retries (marked Unknown)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build Shenwan industry mapping cache")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing *.parquet price caches",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output parquet path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N symbols (useful for smoke tests)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Explicit symbol list; overrides --data-dir discovery",
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = [normalise_symbol(s) for s in args.symbols]
    else:
        symbols = list_symbols_in_data_dir(args.data_dir)

    if args.limit is not None:
        symbols = symbols[: args.limit]

    if not symbols:
        print("No symbols found.", file=sys.stderr)
        sys.exit(1)

    print(f"Building Shenwan map for {len(symbols)} symbols...")
    df = build_shenwan_map(symbols, output_path=args.output)
    covered = df[df["sw_industry_2"] != "Unknown"].shape[0]
    print(f"Coverage: {covered}/{len(df)} ({100 * covered / len(df):.1f}%)")


if __name__ == "__main__":
    main()
