#!/usr/bin/env python3
"""D0.1 采集全量 A 股上市日期表

解决幸存者偏差的核心数据：知道每只股票什么时候上市、是否退市。

输出: data_cache/listing_dates.csv
    symbol, name, list_date, market
"""

from pathlib import Path
import sys

# 使用 default venv 中的 akshare
PYTHON = "/Users/hejinyang/.workbuddy/binaries/python/envs/default/bin/python"

import akshare as ak
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT = CACHE_DIR / "listing_dates.csv"


def fetch_listing_dates():
    """从 akshare 获取全部 A 股上市日期"""
    print("[D0.1] 拉取全量 A 股上市日期...")
    
    # 上证
    try:
        sh = ak.stock_info_sh_name_code()
        sh = sh.rename(columns={
            "证券代码": "code",
            "证券简称": "name",
            "上市日期": "list_date",
        })
        sh["market"] = "sh"
        print(f"  上证: {len(sh)} 只")
    except Exception as e:
        print(f"  ⚠️ 上证失败: {e}")
        sh = pd.DataFrame(columns=["code", "name", "list_date", "market"])
    
    # 深证
    try:
        sz = ak.stock_info_sz_name_code()
        sz = sz.rename(columns={
            "A股代码": "code",
            "A股简称": "name",
            "A股上市日期": "list_date",
        })
        sz["market"] = "sz"
        print(f"  深证: {len(sz)} 只")
    except Exception as e:
        print(f"  ⚠️ 深证失败: {e}")
        sz = pd.DataFrame(columns=["code", "name", "list_date", "market"])
    
    # 北证 (可选)
    try:
        bj = ak.stock_info_bj_name_code()
        bj = bj.rename(columns={
            "证券代码": "code",
            "证券简称": "name",
            "上市日期": "list_date",
        })
        bj["market"] = "bj"
        print(f"  北证: {len(bj)} 只")
    except Exception as e:
        print(f"  ⚠️ 北证失败: {e}")
        bj = pd.DataFrame(columns=["code", "name", "list_date", "market"])
    
    df = pd.concat([sh, sz, bj], ignore_index=True)
    
    # 统一 symbol
    df["symbol"] = df["market"] + df["code"].astype(str).str.zfill(6)
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    
    # 去重
    df = df.drop_duplicates(subset=["symbol"], keep="first")
    df = df[["symbol", "name", "list_date", "market"]].copy()
    
    print(f"\n  总计: {len(df)} 只")
    print(f"  有上市日期: {df['list_date'].notna().sum()}")
    print(f"  最早上市: {df['list_date'].min()}")
    print(f"  最新上市: {df['list_date'].max()}")
    
    # 保存
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ 已保存: {OUTPUT}")
    
    return df


if __name__ == "__main__":
    fetch_listing_dates()
