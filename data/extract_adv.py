#!/usr/bin/env python3
"""D0.2 从现有日线数据提取成交额/ADV

输入: data_cache/*.parquet (price data)
输出: data_cache/adv_panel.parquet
    MultiIndex: (date, symbol) -> adv_20d
"""

from pathlib import Path
import pandas as pd
import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
OUTPUT = CACHE_DIR / "adv_panel.parquet"


def extract_adv_from_parquet():
    """从现有 parquet 文件提取成交额并计算 ADV"""
    print("[D0.2] 从现有日线数据提取 ADV...")
    
    parquet_files = list(CACHE_DIR.glob("*.parquet"))
    print(f"  发现 {len(parquet_files)} 个 parquet 文件")
    
    all_amounts = []
    
    for f in parquet_files:
        symbol = f.stem  # e.g., "sh600000"
        try:
            df = pd.read_parquet(f)
            # 检查列名
            if "amount" in df.columns:
                amount_col = "amount"
            elif "成交额" in df.columns:
                amount_col = "成交额"
            elif "volume" in df.columns and "close" in df.columns:
                # 估算成交额 = volume * close
                df["amount"] = df["volume"] * df["close"]
                amount_col = "amount"
            else:
                continue
            
            df = df[["date", amount_col]].copy()
            df["symbol"] = symbol
            df = df.rename(columns={amount_col: "amount"})
            all_amounts.append(df[["date", "symbol", "amount"]])
        except Exception as e:
            pass  # 静默跳过坏文件
    
    if not all_amounts:
        print("  ⚠️ 未找到任何成交额数据")
        return None
    
    # 合并
    combined = pd.concat(all_amounts, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    print(f"  合并后: {len(combined)} 行, {combined['symbol'].nunique()} 只")
    
    # 计算 20 日 ADV (按 symbol 分组滚动)
    combined = combined.sort_values(["symbol", "date"])
    combined["adv_20d"] = combined.groupby("symbol")["amount"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    
    # 保存为 parquet
    combined = combined.set_index(["date", "symbol"]).sort_index()
    combined.to_parquet(OUTPUT)
    
    print(f"  ✅ 已保存: {OUTPUT}")
    print(f"  样本 ADV: mean={combined['adv_20d'].mean()/1e4:.0f}万, median={combined['adv_20d'].median()/1e4:.0f}万")
    
    return combined


if __name__ == "__main__":
    extract_adv_from_parquet()
