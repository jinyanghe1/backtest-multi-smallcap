"""
数据下载模块
=============
通过 westock-data CLI 或 ak share 批量下载 A 股日线数据, 缓存为 Parquet 文件。

架构:
  1. 获取全 A 股列表 (含市值/PE/PB 概览)
  2. 筛选微盘股 (市值 < 30亿)
  3. 逐只下载日线 K 线 → 存 Parquet
  4. 支持断点续传

用法:
  from tools.backtest_mvp.data import download_microcap_universe
  download_microcap_universe()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import json
import time
import subprocess
import sys

DATA_DIR = Path(__file__).parent / "data_cache"
WESTOCK_SCRIPT = Path("/Users/hejinyang/.workbuddy/plugins/marketplaces/cb_teams_marketplace/plugins/finance-data/skills/westock-data/scripts/index.js")


def _westock(cmd: str, timeout: int = 60) -> dict:
    """调用 westock-data CLI, 解析 JSON 返回"""
    import subprocess
    result = subprocess.run(
        ["node", str(WESTOCK_SCRIPT)] + cmd.split(),
        capture_output=True, text=True, timeout=timeout,
        cwd=str(Path(WESTOCK_SCRIPT).parent.parent),
    )
    # westock-data 输出 Markdown 表格或 JSON, 尝试解析 JSON
    output = result.stdout.strip()
    # 先找 JSON 数组
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('[') or line.startswith('{'):
            try:
                return json.loads(line)
            except:
                pass
    return {"raw": output}


def fetch_microcap_symbols() -> pd.DataFrame:
    """
    获取微盘股概念成分股列表 (~700 只)

    Returns:
        DataFrame with columns: code, name
    """
    print("正在获取微盘股成分股列表...")
    # 使用 westock-data sector 获取微盘股概念 (pt02GN2282)
    result = _westock("sector pt02GN2282", timeout=30)

    if "raw" in result:
        # 解析 Markdown 表格
        lines = result["raw"].split('\n')
        records = []
        in_table = False
        for line in lines:
            line = line.strip()
            if '|' in line and ('code' in line or '---' in line):
                in_table = True
                continue
            if in_table and line.startswith('|'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 2:
                    records.append({"code": parts[0], "name": parts[1]})

        df = pd.DataFrame(records)
        print(f"  获取到 {len(df)} 只微盘股")
        return df
    return pd.DataFrame()


def fetch_stock_kline(symbol: str, days: int = 1500) -> pd.DataFrame:
    """
    下载单只股票的日线 K 线数据

    Args:
        symbol: 如 sh600000, sz000001
        days: 获取天数 (默认 1500 ≈ 6 年)

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
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
                            "high": float(parts[3]),
                            "low": float(parts[4]),
                            "close": float(parts[2]),
                            "volume": float(parts[5]),
                        })
                    except (ValueError, IndexError):
                        continue

        df = pd.DataFrame(records)
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df = df.sort_values('date').reset_index(drop=True)
        return df
    return pd.DataFrame()


def fetch_stock_quote(symbols: List[str]) -> pd.DataFrame:
    """
    批量获取实时行情 (含 mcap, pb, pe, turnover)
    """
    import subprocess
    batch = ",".join(symbols[:20])  # 一次最多 20 只
    try:
        result = subprocess.run(
            ["node", str(WESTOCK_SCRIPT), "quote", batch],
            capture_output=True, text=True, timeout=60,
            cwd=str(Path(WESTOCK_SCRIPT).parent.parent),
        )
        output = result.stdout
        records = []
        in_table = False
        for line in output.split('\n'):
            line = line.strip()
            if '|' in line and ('code' in line or '---' in line):
                in_table = True
                continue
            if in_table and line.startswith('|'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 10:
                    try:
                        records.append({
                            "symbol": parts[0],
                            "name": parts[3],
                            "mcap": float(parts[29]) / 1e8 if len(parts) > 29 and parts[29] else 0,
                            "pb": float(parts[25]) if len(parts) > 25 and parts[25] else 0,
                            "pe": float(parts[22]) if len(parts) > 22 and parts[22] else 0,
                            "turnover": float(parts[19]) if len(parts) > 19 and parts[19] else 0,
                        })
                    except (ValueError, IndexError):
                        continue
        return pd.DataFrame(records)
    except:
        return pd.DataFrame()


def download_microcap_universe(
    max_stocks: int = 500,
    kline_days: int = 1500,
    skip_existing: bool = True,
    sleep_between: float = 0.3,
) -> pd.DataFrame:
    """
    下载微盘股全量数据

    Args:
        max_stocks: 最多下载多少只 (默认 500, 全部 700 只需要更长时间)
        kline_days: K 线天数
        skip_existing: 跳过已有缓存
        sleep_between: 请求间隔 (秒)

    Returns:
        DataFrame: 下载统计
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 获取微盘股列表
    symbols_df = fetch_microcap_symbols()
    if len(symbols_df) == 0:
        print("⚠️ 无法获取微盘股列表, 尝试用备选方案...")
        # 备选: 用 westock-data index 功能
        return pd.DataFrame()

    symbols = symbols_df['code'].tolist()[:max_stocks]
    print(f"准备下载 {len(symbols)} 只微盘股的日线 (每只 {kline_days} 天)...")

    # 2. 逐只下载
    stats = {"total": len(symbols), "downloaded": 0, "cached": 0, "failed": 0, "symbols": []}
    start_time = time.time()

    for i, symbol in enumerate(symbols):
        cache_file = DATA_DIR / f"{symbol}.parquet"

        if skip_existing and cache_file.exists():
            stats["cached"] += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  进度: {i+1}/{len(symbols)} ({elapsed:.0f}s) "
                      f"[下载{stats['downloaded']} 缓存{stats['cached']} 失败{stats['failed']}]")
            continue

        try:
            df = fetch_stock_kline(symbol, kline_days)
            if len(df) > 30:  # 至少 30 个交易日, 否则视为无效
                df.to_parquet(cache_file, index=False)
                stats["downloaded"] += 1
                stats["symbols"].append(symbol)
            else:
                stats["failed"] += 1
        except Exception as e:
            stats["failed"] += 1
            if stats["failed"] <= 3:
                print(f"  ⚠️ {symbol} 下载失败: {str(e)[:80]}")

        # 进度
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(symbols) - i - 1)
            print(f"  进度: {i+1}/{len(symbols)} ({elapsed:.0f}s, ETA {eta:.0f}s) "
                  f"[下载{stats['downloaded']} 缓存{stats['cached']} 失败{stats['failed']}]")

        time.sleep(sleep_between)

    elapsed = time.time() - start_time
    print(f"\n  完成! 总耗时 {elapsed:.0f}s "
          f"({elapsed/60:.1f}min)")
    print(f"  下载: {stats['downloaded']} | 缓存: {stats['cached']} | 失败: {stats['failed']}")

    return pd.DataFrame(stats)


def get_data_summary() -> pd.DataFrame:
    """检查本地缓存的数据状态"""
    if not DATA_DIR.exists():
        return pd.DataFrame()

    files = sorted(DATA_DIR.glob("*.parquet"))
    records = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            records.append({
                "symbol": f.stem,
                "rows": len(df),
                "start": df['date'].min() if 'date' in df.columns else None,
                "end": df['date'].max() if 'date' in df.columns else None,
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
        except:
            pass

    return pd.DataFrame(records)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        summary = get_data_summary()
        print(f"本地缓存: {len(summary)} 只")
        if len(summary) > 0:
            print(summary.to_string())
    else:
        download_microcap_universe(max_stocks=20, kline_days=500)
