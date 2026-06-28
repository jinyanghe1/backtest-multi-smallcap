"""
Data cleaning pipeline for all parquet files in data_cache.

Issues addressed:
1. 停牌日价格不一致: open/high/low = 0, close ≠ 0
   → 将 open/high/low 同步为 close（停牌日无交易，价格不变）
2. 价格逻辑修复: 确保 high >= max(open, close), low <= min(open, close)
   → 用 true_high/true_low 修正

Usage:
    python clean_price_data.py [--dry-run]
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

DATA_CACHE = Path("data_cache")

def clean_single_file(filepath: Path, dry_run: bool = False) -> dict:
    """Clean one parquet file. Returns stats dict."""
    df = pd.read_parquet(filepath)
    original_len = len(df)
    
    if not {'open', 'high', 'low', 'close'}.issubset(df.columns):
        return {"file": filepath.name, "status": "skipped", "reason": "missing columns"}
    
    changes = {
        "synced_stop_rows": 0,      # 停牌日同步
        "fixed_high": 0,            # high修正
        "fixed_low": 0,             # low修正
    }
    
    # 1. 停牌日: open=high=low=0, close≠0 → 同步为close
    stop_mask = (df['open'] == 0) & (df['high'] == 0) & (df['low'] == 0) & (df['close'] != 0)
    if stop_mask.any():
        changes["synced_stop_rows"] = int(stop_mask.sum())
        if not dry_run:
            df.loc[stop_mask, 'open'] = df.loc[stop_mask, 'close']
            df.loc[stop_mask, 'high'] = df.loc[stop_mask, 'close']
            df.loc[stop_mask, 'low'] = df.loc[stop_mask, 'close']
    
    # 2. 修正 high/low 逻辑: high应为当天最高，low应为当天最低
    true_high = df[['open', 'high', 'low', 'close']].max(axis=1)
    true_low = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    high_diff = (df['high'] != true_high).sum()
    low_diff = (df['low'] != true_low).sum()
    
    if high_diff > 0:
        changes["fixed_high"] = int(high_diff)
    if low_diff > 0:
        changes["fixed_low"] = int(low_diff)
    
    if not dry_run and (high_diff > 0 or low_diff > 0):
        df['high'] = true_high
        df['low'] = true_low
    
    # 3. 保存
    if not dry_run and sum(changes.values()) > 0:
        df.to_parquet(filepath, index=False)
    
    return {
        "file": filepath.name,
        "status": "cleaned" if sum(changes.values()) > 0 else "ok",
        **changes
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--max", type=int, default=None, help="Max files to process")
    args = parser.parse_args()
    
    files = sorted([f for f in DATA_CACHE.iterdir() 
                   if f.suffix == '.parquet' and not f.name.startswith('adv')])
    
    if args.max:
        files = files[:args.max]
    
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(files)} files...")
    
    total_synced = 0
    total_fixed_high = 0
    total_fixed_low = 0
    cleaned_count = 0
    
    for i, f in enumerate(files, 1):
        result = clean_single_file(f, dry_run=args.dry_run)
        
        if result["status"] == "cleaned":
            cleaned_count += 1
            total_synced += result.get("synced_stop_rows", 0)
            total_fixed_high += result.get("fixed_high", 0)
            total_fixed_low += result.get("fixed_low", 0)
        
        if i % 500 == 0 or i == len(files):
            print(f"  Progress: {i}/{len(files)} files ({cleaned_count} cleaned)")
    
    print(f"\n{'='*50}")
    print(f"Summary ({'DRY RUN' if args.dry_run else 'WRITTEN'}):")
    print(f"  Total files: {len(files)}")
    print(f"  Files cleaned: {cleaned_count}")
    print(f"  Stop rows synced: {total_synced}")
    print(f"  High fixed: {total_fixed_high}")
    print(f"  Low fixed: {total_fixed_low}")
    print(f"  {'='*50}")

if __name__ == "__main__":
    main()
