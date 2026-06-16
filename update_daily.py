#!/usr/bin/env python3
"""
日线增量更新脚本
=================
每天运行 2 次 (午间 12:00 + 晚间 20:00), 拉取最新交易日数据追加到 Parquet 缓存。

数据流:
  westock-data kline --limit 5 → 取最近 5 天
  → 与本地 Parquet 最后一天对比
  → 仅追加新日期
  → 去重 + 排序 + 保存

用法:
  python tools/backtest_mvp/update_daily.py           # 增量: 仅有新日期才追加
  python tools/backtest_mvp/update_daily.py --force   # 强制: 重写最后5天
  python tools/backtest_mvp/update_daily.py --dry-run # 仅检查不写入
"""

import sys
import os
import time
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / "data_cache"
WESTOCK = Path(os.path.expanduser(
    "~/.workbuddy/plugins/marketplaces/cb_teams_marketplace/"
    "plugins/finance-data/skills/westock-data/scripts/index.js"
))
LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def fetch_recent(symbol: str, days: int = 5) -> pd.DataFrame:
    """拉取最近 N 天 K 线"""
    result = subprocess.run(
        ["node", str(WESTOCK), "kline", symbol,
         "--period", "day", "--limit", str(days), "--fq", "qfq"],
        capture_output=True, text=True, timeout=30,
        cwd=str(WESTOCK.parent.parent),
    )
    output = result.stdout
    records = []
    in_table = False
    for line in output.split("\n"):
        line = line.strip()
        if "|" in line and ("date" in line.lower() or "---" in line):
            in_table = True
            continue
        if in_table and line.startswith("|"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 6:
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

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def update_one(symbol: str, force: bool = False, dry_run: bool = False) -> dict:
    """更新单只股票, 返回 {'status', 'new_rows', 'symbol'}"""
    cache_file = DATA_DIR / f"{symbol}.parquet"

    # 取远程最近 5 天
    try:
        new_data = fetch_recent(symbol, days=5)
    except Exception as e:
        return {"status": "fetch_fail", "new_rows": 0, "symbol": symbol,
                "error": str(e)[:80]}

    if len(new_data) == 0:
        return {"status": "fetch_empty", "new_rows": 0, "symbol": symbol}

    if not cache_file.exists():
        if not dry_run:
            new_data.to_parquet(cache_file, index=False)
        return {"status": "created", "new_rows": len(new_data), "symbol": symbol}

    # 读本地缓存
    try:
        local = pd.read_parquet(cache_file)
        local["date"] = pd.to_datetime(local["date"])
    except:
        return {"status": "read_fail", "new_rows": 0, "symbol": symbol}

    last_local = local["date"].max()

    if force:
        # 强制重写最后 5 天: 去掉本地最后 5 天, 然后用远程数据替换
        cutoff = last_local - timedelta(days=5)
        local = local[local["date"] <= cutoff]
        to_add = new_data[new_data["date"] > cutoff]
    else:
        # 增量: 仅保留严格晚于本地的日期
        to_add = new_data[new_data["date"] > last_local]

    if len(to_add) == 0:
        return {"status": "up_to_date", "new_rows": 0, "symbol": symbol}

    # 合并 + 去重 + 排序
    combined = pd.concat([local, to_add], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

    if not dry_run:
        combined.to_parquet(cache_file, index=False)

    return {"status": "updated", "new_rows": len(to_add), "symbol": symbol,
            "before": str(last_local.date()), "after": str(combined["date"].max().date())}


def run_update(max_stocks: int = None, force: bool = False,
               dry_run: bool = False, delay: float = 0.3):
    """批量更新所有日线缓存"""
    files = sorted(DATA_DIR.glob("*.parquet"))
    symbols = [f.stem for f in files]
    if max_stocks:
        symbols = symbols[:max_stocks]

    print(f"日线增量更新 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  股票数: {len(symbols)}")
    print(f"  模式: {'强制' if force else '增量'} {'(dry-run)' if dry_run else ''}")
    print()

    stats = {"total": len(symbols), "updated": 0, "up_to_date": 0,
             "created": 0, "failed": 0, "new_rows": 0}
    start = time.time()

    for i, sym in enumerate(symbols):
        result = update_one(sym, force=force, dry_run=dry_run)
        stats[result["status"]] = stats.get(result["status"], 0) + 1
        stats["new_rows"] += result.get("new_rows", 0)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (len(symbols) - i - 1)
            print(f"  进度: {i+1}/{len(symbols)} ({elapsed:.0f}s, ETA {eta:.0f}s) "
                  f"[更新{stats.get('updated',0)} 最新{stats.get('up_to_date',0)} "
                  f"失败{stats.get('failed',0)+stats.get('fetch_fail',0)}]")

        time.sleep(delay)

    elapsed = time.time() - start
    print(f"\n{'  DRY RUN — 无写入' if dry_run else '  更新完成!'}")
    print(f"  耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  更新: {stats.get('updated', 0)} 只 ({stats['new_rows']} 行新数据)")
    print(f"  已最新: {stats.get('up_to_date', 0)}")
    print(f"  新建: {stats.get('created', 0)}")
    print(f"  失败: {stats.get('fetch_fail', 0) + stats.get('fetch_empty', 0) + stats.get('read_fail', 0)}")

    # 写入日志
    log_entry = {
        "time": datetime.now().isoformat(),
        "mode": "force" if force else "incremental",
        "dry_run": dry_run,
        "stats": stats,
        "elapsed_s": elapsed,
    }
    log_file = LOG_DIR / f"update_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    if not dry_run:
        import json
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2, default=str)

    return stats


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="日线增量更新")
    p.add_argument("--force", action="store_true", help="强制重写最后5天")
    p.add_argument("--dry-run", action="store_true", help="仅检查不写入")
    p.add_argument("--max", type=int, default=None, help="最多更新几只 (调试用)")
    p.add_argument("--delay", type=float, default=0.3, help="请求间隔 (秒)")
    args = p.parse_args()

    run_update(
        max_stocks=args.max,
        force=args.force,
        dry_run=args.dry_run,
        delay=args.delay,
    )
