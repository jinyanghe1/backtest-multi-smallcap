#!/usr/bin/env python3
"""
A股全量日线数据批量采集脚本
==============================
使用修复后的 westock-data CLI，从 name_lookup.parquet 读取全量股票列表，
批量下载每只股票的1500日K线数据。

用法:
    python batch_download_a_share.py [--max N] [--days 1500]
"""

import subprocess
import json
import time
import pandas as pd
from pathlib import Path
from typing import Optional
import argparse

# ── 路径配置 ──
PROJECT_DIR = Path(__file__).resolve().parent
DATA_CACHE = PROJECT_DIR / "data_cache"
NAME_LOOKUP = PROJECT_DIR / "name_lookup.parquet"

WESTOCK_SCRIPT = Path("/Users/hejinyang/.workbuddy/plugins/marketplaces/experts/plugins/stock-partner-team/skills/westock-data/scripts/index.js")
NODE_BIN = Path("/Users/hejinyang/.workbuddy/binaries/node/versions/22.22.2/bin/node")


def _westock(cmd: str, timeout: int = 60) -> dict:
    """调用 westock-data CLI，解析 JSON 返回"""
    result = subprocess.run(
        [str(NODE_BIN), str(WESTOCK_SCRIPT)] + cmd.split(),
        capture_output=True, text=True, timeout=timeout,
        cwd=str(WESTOCK_SCRIPT.parent.parent),
    )
    output = result.stdout.strip()
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('[') or line.startswith('{'):
            try:
                return json.loads(line)
            except:
                pass
    return {"raw": output}


def fetch_kline(symbol: str, days: int = 1500) -> pd.DataFrame:
    """下载单只股票日线 K 线"""
    result = _westock(f"kline {symbol} --period day --limit {days} --fq qfq", timeout=30)

    if "raw" in result:
        lines = result["raw"].split('\n')
        records = []
        in_table = False
        for line in lines:
            line = line.strip()
            if '|' in line and ('date' in line or '---' in line):
                in_table = True
                continue
            if in_table and line.startswith('|'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 7:
                    try:
                        records.append({
                            "date": parts[0],
                            "open": float(parts[1]),
                            "close": float(parts[2]),
                            "high": float(parts[3]),
                            "low": float(parts[4]),
                            "volume": float(parts[5]),
                            "amount": float(parts[6]) if len(parts) > 6 else 0.0,
                        })
                    except (ValueError, IndexError):
                        continue
        df = pd.DataFrame(records)
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df = df.sort_values('date').reset_index(drop=True)
        return df
    return pd.DataFrame()


def batch_download(
    max_stocks: Optional[int] = None,
    days: int = 1500,
    skip_existing: bool = True,
    sleep_between: float = 0.5,
):
    """批量下载全量 A 股日线数据"""

    DATA_CACHE.mkdir(parents=True, exist_ok=True)

    # 1. 读取全量股票列表
    if not NAME_LOOKUP.exists():
        print(f"❌ 未找到 {NAME_LOOKUP}，请先生成股票列表")
        return

    df_names = pd.read_parquet(NAME_LOOKUP)
    all_symbols = list(df_names.index)  # 索引是 symbol

    # 2. 过滤已有数据
    existing = {f.stem for f in DATA_CACHE.glob("*.parquet") if f.stem.startswith(('sh', 'sz', 'bj'))}
    if skip_existing:
        symbols = [s for s in all_symbols if s not in existing]
        print(f"📊 全量股票: {len(all_symbols)} 只 | 已有缓存: {len(existing)} 只 | 待下载: {len(symbols)} 只")
    else:
        symbols = all_symbols
        print(f"📊 全量股票: {len(all_symbols)} 只 | 强制重新下载")

    if max_stocks:
        symbols = symbols[:max_stocks]
        print(f"  本次限制下载前 {max_stocks} 只")

    if not symbols:
        print("✅ 所有股票数据已下载完毕，无需继续")
        return

    # 3. 逐只下载
    stats = {"total": len(symbols), "downloaded": 0, "cached": 0, "failed": 0, "skipped": 0}
    start_time = time.time()

    for i, symbol in enumerate(symbols):
        cache_file = DATA_CACHE / f"{symbol}.parquet"

        if skip_existing and cache_file.exists():
            stats["cached"] += 1
            continue

        try:
            df = fetch_kline(symbol, days)
            if len(df) > 30:  # 至少 30 个交易日
                df.to_parquet(cache_file, index=False)
                stats["downloaded"] += 1
            else:
                stats["failed"] += 1
                if stats["failed"] <= 5:
                    print(f"  ⚠️ {symbol} 数据不足 ({len(df)} 行)")
        except Exception as e:
            stats["failed"] += 1
            if stats["failed"] <= 5:
                print(f"  ⚠️ {symbol} 下载失败: {str(e)[:80]}")

        # 进度报告
        if (i + 1) % 20 == 0 or (i + 1) == len(symbols):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(symbols) - i - 1) / rate if rate > 0 else 0
            print(f"  📈 进度: {i+1}/{len(symbols)} "
                  f"(下载{stats['downloaded']} 跳过{stats['cached']} 失败{stats['failed']}) "
                  f"[{rate:.1f}只/s, ETA {eta/60:.0f}min]")

        time.sleep(sleep_between)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ 完成! 总耗时 {elapsed/60:.1f}min")
    print(f"   下载: {stats['downloaded']} | 已有: {stats['cached']} | 失败: {stats['failed']}")
    print(f"   数据目录: {DATA_CACHE}")
    print(f"   总文件数: {len(list(DATA_CACHE.glob('*.parquet')))}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股全量日线数据批量采集")
    parser.add_argument("--max", type=int, default=None, help="最多下载多少只（默认全部）")
    parser.add_argument("--days", type=int, default=1500, help="每只下载多少天历史（默认1500）")
    parser.add_argument("--no-skip", action="store_true", help="不跳过已有文件，强制重新下载")
    parser.add_argument("--sleep", type=float, default=0.5, help="请求间隔秒数（默认0.5）")
    args = parser.parse_args()

    batch_download(
        max_stocks=args.max,
        days=args.days,
        skip_existing=not args.no_skip,
        sleep_between=args.sleep,
    )
