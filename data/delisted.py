"""退市数据管理模块 — PIT 无偏 universe 重建"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

import akshare as ak


PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

DELISTED_CACHE = CACHE_DIR / "delisted_stocks.csv"


@dataclass
class DelistRecord:
    """单只股票退市记录"""
    symbol: str  # 如 sh600001
    name: str
    list_date: pd.Timestamp  # 上市日
    delist_date: pd.Timestamp  # 退市日 / 终止上市日
    market: str  # "sh" | "sz"


class DelistManager:
    """
    退市股票管理器。
    
    核心功能:
    1. 从 akshare 获取全量退市列表 (上证+深证)
    2. 缓存为本地 Parquet
    3. 提供 PIT universe 查询: 给定日期，返回当时"还活着"的股票
    4. 支持增量更新
    """

    def __init__(self, cache_path: Path | None = None):
        self.cache_path = cache_path or DELISTED_CACHE
        self.df: pd.DataFrame | None = None
        self._load_cache()

    def _load_cache(self) -> None:
        """加载本地缓存"""
        if self.cache_path.exists():
            try:
                self.df = pd.read_csv(self.cache_path, encoding="utf-8-sig")
                # 解析日期列
                self.df["list_date"] = pd.to_datetime(self.df["list_date"], errors="coerce")
                self.df["delist_date"] = pd.to_datetime(self.df["delist_date"], errors="coerce")
            except Exception:
                self.df = None
        else:
            self.df = None

    def fetch_all(self, force: bool = False) -> pd.DataFrame:
        """
        从 akshare 获取退市数据。
        
        如果缓存存在且 force=False, 返回缓存。
        否则重新拉取。
        
        Returns:
            DataFrame with columns: [symbol, name, list_date, delist_date, market]
        """
        if self.df is not None and not force:
            return self.df

        print("  [delisted] 从 akshare 拉取退市列表...")
        
        # 上证退市
        try:
            sh = ak.stock_info_sh_delist()
            sh = sh.rename(columns={
                "公司代码": "code",
                "公司简称": "name",
                "上市日期": "list_date",
                "暂停上市日期": "delist_date",
            })
            sh["market"] = "sh"
        except Exception as e:
            print(f"    ⚠️ 上证退市列表失败: {e}")
            sh = pd.DataFrame(columns=["code", "name", "list_date", "delist_date", "market"])

        # 深证退市
        try:
            sz = ak.stock_info_sz_delist()
            sz = sz.rename(columns={
                "证券代码": "code",
                "证券简称": "name",
                "上市日期": "list_date",
                "终止上市日期": "delist_date",
            })
            sz["market"] = "sz"
        except Exception as e:
            print(f"    ⚠️ 深证退市列表失败: {e}")
            sz = pd.DataFrame(columns=["code", "name", "list_date", "delist_date", "market"])

        df = pd.concat([sh, sz], ignore_index=True)
        
        # 统一 symbol 格式: sh/sz + 6位代码
        df["symbol"] = df["market"] + df["code"].astype(str).str.zfill(6)
        
        # 日期解析
        df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
        df["delist_date"] = pd.to_datetime(df["delist_date"], errors="coerce")
        
        # 去重
        df = df.drop_duplicates(subset=["symbol"], keep="first")
        
        # 列排序
        df = df[["symbol", "name", "list_date", "delist_date", "market"]].copy()
        
        # 缓存
        df.to_csv(self.cache_path, index=False, encoding="utf-8-sig")
        self.df = df
        
        print(f"  [delisted] 共 {len(df)} 只退市股 (上证{len(sh)} + 深证{len(sz)})")
        print(f"  [delisted] 最早退市: {df['delist_date'].min()}, 最晚: {df['delist_date'].max()}")
        
        return df

    def get_delisted_before(self, date: str | pd.Timestamp) -> List[str]:
        """
        返回在指定日期之前已退市的股票列表。
        
        用于 PIT 回测: 这只股票在 date 这一天已经死了, 不应纳入 universe。
        """
        if self.df is None:
            self.fetch_all()
        
        ts = pd.Timestamp(date)
        mask = (self.df["delist_date"] <= ts) & (self.df["delist_date"].notna())
        return self.df.loc[mask, "symbol"].tolist()

    def is_alive(self, symbol: str, date: str | pd.Timestamp) -> bool:
        """
        判断某股票在指定日期是否"还活着"。
        
        条件:
        - 已上市 (list_date <= date)
        - 未退市 (delist_date > date 或 delist_date 为 NaN)
        """
        if self.df is None:
            self.fetch_all()
        
        ts = pd.Timestamp(date)
        row = self.df[self.df["symbol"] == symbol]
        if len(row) == 0:
            # 不在退市列表中 -> 活着 (但需验证是否已上市)
            # 这里简化为: 未知退市列表默认活着
            return True
        
        list_date = row.iloc[0]["list_date"]
        delist_date = row.iloc[0]["delist_date"]
        
        if pd.isna(list_date) or list_date > ts:
            return False  # 还没上市
        
        if pd.isna(delist_date):
            return True  # 没有退市日 -> 一直活着
        
        return delist_date > ts  # 退市日 > 查询日 -> 还活着

    def pit_universe_filter(
        self,
        snapshot: pd.DataFrame,
        all_dates: list,
        rebalance_idx: int,
    ) -> List[str]:
        """
        PIT universe filter: 返回当前日期"还活着"的股票。
        
        设计为 engine.universe_filter 的 callable 参数。
        
        Args:
            snapshot: factor_snapshot DataFrame (index=symbol)
            all_dates: engine 的 self.dates (完整日期列表)
            rebalance_idx: 当前调仓索引
        
        Returns:
            存活股票列表
        """
        if rebalance_idx >= len(all_dates):
            return []
        
        current_date = all_dates[rebalance_idx]
        
        # 获取退市黑名单
        dead = set(self.get_delisted_before(current_date))
        
        alive = [s for s in snapshot.index if s not in dead]
        return alive

    def summary(self) -> dict:
        """返回退市数据摘要"""
        if self.df is None:
            self.fetch_all()
        
        return {
            "total_delisted": len(self.df),
            "sh_count": len(self.df[self.df["market"] == "sh"]),
            "sz_count": len(self.df[self.df["market"] == "sz"]),
            "earliest_delist": self.df["delist_date"].min().strftime("%Y-%m-%d") if not self.df["delist_date"].isna().all() else None,
            "latest_delist": self.df["delist_date"].max().strftime("%Y-%m-%d") if not self.df["delist_date"].isna().all() else None,
            "cached_at": self.cache_path.stat().st_mtime if self.cache_path.exists() else None,
        }


# ── 便捷函数 ──

def get_delist_manager() -> DelistManager:
    """获取全局退市管理器实例"""
    return DelistManager()


def build_pit_universe(
    delist_mgr: DelistManager,
    symbols: List[str],
    date: str | pd.Timestamp,
) -> List[str]:
    """
    给定股票列表和日期，返回 PIT 存活的子集。
    
    这是退市数据的核心接口：在每次调仓前调用，过滤掉已退市股票。
    """
    ts = pd.Timestamp(date)
    dead = set(delist_mgr.get_delisted_before(ts))
    return [s for s in symbols if s not in dead]
