#!/usr/bin/env python3
"""
Phase 1: 引擎数据层预计算与 Fast 查询方法

核心目标:
1. 将 MultiIndex DataFrame 预 pivot 为 2D/3D numpy 数组
2. 建立 date_idx / stock_idx 映射，消除 O(N) 索引查找
3. 保持原有接口完全兼容，内部自动切换 fast 方法
4. 添加单元测试验证 fast 输出与原始输出一致

Date: 2026-06-28
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class EngineDataPrecomputation:
    """
    引擎数据预计算类。

    将 factor_panel (MultiIndex: date, stock) 和 return_panel (MultiIndex: date, stock)
    预计算为可直接 numpy 索引的 2D/3D 数组，并提供 O(1) 查询方法。
    """

    def __init__(
        self,
        factor_panel: pd.DataFrame,
        return_panel: pd.DataFrame,
    ):
        """
        Parameters
        ----------
        factor_panel : pd.DataFrame
            MultiIndex (date, stock), columns = [mcap, pb, mom20d, ...]
        return_panel : pd.DataFrame
            MultiIndex (date, stock), columns = ['daily_return']
        """
        self.factor_panel = factor_panel
        self.return_panel = return_panel

        # 1. 提取统一日期和股票列表
        self.dates = sorted(
            pd.Timestamp(d) for d in (
                set(factor_panel.index.get_level_values(0))
                & set(return_panel.index.get_level_values(0))
            )
        )
        self.stocks = sorted(
            set(factor_panel.index.get_level_values(1))
            & set(return_panel.index.get_level_values(1))
        )

        self.n_dates = len(self.dates)
        self.n_stocks = len(self.stocks)
        self.all_factor_names = list(factor_panel.columns)
        self.n_all_factors = len(self.all_factor_names)

        # 2. 建立索引映射 (O(1) 查找)
        self.date_to_idx: Dict[pd.Timestamp, int] = {d: i for i, d in enumerate(self.dates)}
        self.stock_to_idx: Dict[str, int] = {s: i for i, s in enumerate(self.stocks)}
        self.idx_to_stock: Dict[int, str] = {i: s for s, i in self.stock_to_idx.items()}
        self.idx_to_date: Dict[int, pd.Timestamp] = {i: d for d, i in self.date_to_idx.items()}

        # 3. 预 pivot 收益率矩阵 (date × stock)
        self.returns_2d = self._pivot_returns(return_panel)

        # 4. 预 pivot 因子矩阵 (date × stock × factor)
        self.factors_2d = self._pivot_factors(factor_panel)

        # 5. 预计算对数收益率 (用于向量化累积)
        self.log_returns_2d = np.log1p(self.returns_2d)

        # 6. 预计算累积收益率
        self.cum_returns_2d = np.cumprod(1 + self.returns_2d, axis=0)

    # ------------------------------------------------------------------
    # 内部: pivot 方法
    # ------------------------------------------------------------------
    def _pivot_returns(self, return_panel: pd.DataFrame) -> np.ndarray:
        """
        将 MultiIndex 收益率面板 pivot 为 2D numpy 数组 (date × stock)
        缺失值填充为 0
        """
        # unstack: (date, stock) -> date × stock
        pivot = return_panel['daily_return'].unstack(level=1)
        # 重新对齐到统一维度，缺失值填充为 0
        pivot = pivot.reindex(index=self.dates, columns=self.stocks, fill_value=0.0)
        # 确保任何残留的 NaN 也被填充为 0
        pivot = pivot.fillna(0.0)
        return pivot.values.astype(np.float64)

    def _pivot_factors(self, factor_panel: pd.DataFrame) -> np.ndarray:
        """
        将因子面板 pivot 为 3D numpy 数组 (date × stock × factor)
        缺失值填充为 NaN
        只处理数值列，跳过字符串列
        """
        # 筛选数值列
        numeric_cols = factor_panel.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_factor_names = numeric_cols
        self.n_numeric_factors = len(numeric_cols)

        factor_3d = np.full((self.n_dates, self.n_stocks, self.n_numeric_factors), np.nan, dtype=np.float64)

        for f_idx, f_name in enumerate(numeric_cols):
            if f_name not in factor_panel.columns:
                continue
            pivot = factor_panel[f_name].unstack(level=1)
            pivot = pivot.reindex(index=self.dates, columns=self.stocks)
            factor_3d[:, :, f_idx] = pivot.values

        return factor_3d

    # ------------------------------------------------------------------
    # 兼容属性 (用于 engine_fast.py 访问)
    # ------------------------------------------------------------------
    @property
    def factor_names(self) -> list:
        """返回数值因子名列表"""
        return getattr(self, 'numeric_factor_names', [])

    @property
    def n_factors(self) -> int:
        """返回数值因子数量"""
        return getattr(self, 'n_numeric_factors', 0)
    # ------------------------------------------------------------------
    def get_factor_snapshot_fast(self, date_idx: int) -> pd.DataFrame:
        """
        获取指定日期索引的因子横截面（只返回有有效数据的股票）

        Returns
        -------
        pd.DataFrame : index=stock_code, columns=numeric_factor_names
        """
        row = self.factors_2d[date_idx, :, :]  # shape: (n_stocks, n_numeric_factors)
        df = pd.DataFrame(row, index=self.stocks, columns=self.numeric_factor_names)
        # 只返回至少有一个非NaN因子的股票（保留排名因子可能为NaN的情况，由调用方过滤）
        return df.dropna(how='all')

    def get_daily_return_fast(self, date_idx: int, stock_idx_list: Optional[List[int]] = None) -> pd.Series:
        """
        获取指定日期索引的个股收益率

        Parameters
        ----------
        date_idx : int
        stock_idx_list : list[int] or None
            如果为 None，返回所有股票的收益率

        Returns
        -------
        pd.Series : index=stock_code, values=daily_return
        """
        if stock_idx_list is None:
            vals = self.returns_2d[date_idx, :]
            return pd.Series(vals, index=self.stocks)
        else:
            vals = self.returns_2d[date_idx, stock_idx_list]
            stocks = [self.idx_to_stock[i] for i in stock_idx_list]
            return pd.Series(vals, index=stocks)

    def get_period_returns_fast(
        self,
        start_idx: int,
        end_idx: int,
        stock_idx_list: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        获取持仓期间的日收益率矩阵 (dates × stocks)
        使用 log-return 求和累积

        Parameters
        ----------
        start_idx : int
            调仓日索引 (exclusive)
        end_idx : int
            下次调仓日索引 (inclusive)
        stock_idx_list : list[int] or None

        Returns
        -------
        pd.DataFrame : index=date, columns=stock_code
        """
        # 提取日期范围 [start_idx+1, end_idx] 的数据
        s = start_idx + 1
        e = end_idx + 1

        if stock_idx_list is None:
            subset = self.returns_2d[s:e, :]  # (n_dates_in_period, n_stocks)
            dates_in_range = self.dates[s:e]
            return pd.DataFrame(subset, index=dates_in_range, columns=self.stocks)
        else:
            subset = self.returns_2d[s:e, stock_idx_list]
            dates_in_range = self.dates[s:e]
            stocks = [self.idx_to_stock[i] for i in stock_idx_list]
            return pd.DataFrame(subset, index=dates_in_range, columns=stocks)

    def get_period_cumulative_returns_fast(
        self,
        start_idx: int,
        end_idx: int,
        stock_idx_list: Optional[List[int]] = None,
    ) -> pd.Series:
        """
        获取持仓期间每只股票的累积收益率 (1 + cumprod)

        Returns
        -------
        pd.Series : index=stock_code, values=cumulative_return
        """
        s = start_idx + 1
        e = end_idx + 1

        if stock_idx_list is None:
            log_sum = self.log_returns_2d[s:e, :].sum(axis=0)
            stocks = self.stocks
        else:
            log_sum = self.log_returns_2d[s:e, stock_idx_list].sum(axis=0)
            stocks = [self.idx_to_stock[i] for i in stock_idx_list]

        cum_ret = np.exp(log_sum) - 1
        return pd.Series(cum_ret, index=stocks)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def date_to_index(self, date) -> int:
        """将日期转换为索引"""
        ts = pd.Timestamp(date)
        return self.date_to_idx.get(ts, -1)

    def stock_to_index(self, stock: str) -> int:
        """将股票代码转换为索引"""
        return self.stock_to_idx.get(stock, -1)

    def stock_list_to_indices(self, stocks: List[str]) -> List[int]:
        """批量转换股票代码列表为索引列表"""
        return [self.stock_to_idx[s] for s in stocks if s in self.stock_to_idx]

    def get_date_range_indices(self, start_date, end_date) -> Tuple[int, int]:
        """获取日期范围对应的索引"""
        start_idx = self.date_to_idx.get(pd.Timestamp(start_date), 0)
        end_idx = self.date_to_idx.get(pd.Timestamp(end_date), self.n_dates - 1)
        return start_idx, end_idx

    # ------------------------------------------------------------------
    # 缓存/持久化
    # ------------------------------------------------------------------
    def save_cache(self, cache_dir: str = "engine_cache"):
        """保存预计算数据到磁盘，下次可直接加载"""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        np.save(cache_path / "returns_2d.npy", self.returns_2d)
        np.save(cache_path / "factors_2d.npy", self.factors_2d)
        np.save(cache_path / "log_returns_2d.npy", self.log_returns_2d)
        np.save(cache_path / "cum_returns_2d.npy", self.cum_returns_2d)

        # 保存元数据
        meta = {
            "dates": [str(d.date()) for d in self.dates],
            "stocks": self.stocks,
            "factor_names": self.factor_names,
        }
        pd.DataFrame([meta]).to_json(cache_path / "meta.json", orient="records")

        print(f"Cache saved to {cache_path}")

    @classmethod
    def load_cache(cls, cache_dir: str = "engine_cache"):
        """从磁盘加载预计算数据"""
        cache_path = Path(cache_dir)

        returns_2d = np.load(cache_path / "returns_2d.npy")
        factors_2d = np.load(cache_path / "factors_2d.npy")
        log_returns_2d = np.load(cache_path / "log_returns_2d.npy")
        cum_returns_2d = np.load(cache_path / "cum_returns_2d.npy")

        meta = pd.read_json(cache_path / "meta.json").iloc[0].to_dict()
        dates = [pd.Timestamp(d) for d in meta["dates"]]
        stocks = meta["stocks"]
        factor_names = meta["factor_names"]

        # 重建对象
        obj = object.__new__(cls)
        obj.dates = dates
        obj.stocks = stocks
        obj.factor_names = factor_names
        obj.n_dates = len(dates)
        obj.n_stocks = len(stocks)
        obj.n_factors = len(factor_names)
        obj.date_to_idx = {d: i for i, d in enumerate(dates)}
        obj.stock_to_idx = {s: i for i, s in enumerate(stocks)}
        obj.idx_to_stock = {i: s for s, i in obj.stock_to_idx.items()}
        obj.idx_to_date = {i: d for d, i in obj.date_to_idx.items()}
        obj.returns_2d = returns_2d
        obj.factors_2d = factors_2d
        obj.log_returns_2d = log_returns_2d
        obj.cum_returns_2d = cum_returns_2d
        return obj


# ------------------------------------------------------------------
# 快速缓存 key 生成 (用于判断缓存是否有效)
# ------------------------------------------------------------------
def generate_cache_key(factor_panel: pd.DataFrame, return_panel: pd.DataFrame) -> str:
    """
    生成缓存 key，基于数据的 hash。
    如果数据变化，key 变化，自动重新计算。
    """
    import hashlib
    import json

    # 使用数据shape + 首尾行内容作为key
    fp_info = f"{factor_panel.shape}_{factor_panel.index[0]}_{factor_panel.index[-1]}"
    rp_info = f"{return_panel.shape}_{return_panel.index[0]}_{return_panel.index[-1]}"

    key = hashlib.md5(f"{fp_info}_{rp_info}".encode()).hexdigest()[:16]
    return key
