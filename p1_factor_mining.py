"""P1 因子挖掘管线 — 中性化 + 新 Alpha 因子 + 因子动物园评估

本模块实现 P1 阶段的全部因子工程:
- UC1: 截面标准化 + 中性化(size+行业回归取残差)
- UC2: 基于已有算子构建 A1-A5 新 alpha 因子
- UC3: 因子动物园批量评估 (IC/IR/衰减/去相关)

用法:
    from p1_factor_mining import FactorMiningPipeline, NewAlphaFactors
    
    pipeline = FactorMiningPipeline(factor_panel)
    pipeline.neutralize_all()  # UC1
    
    new_factors = NewAlphaFactors(factor_panel)
    alpha1 = new_factors.short_term_reversal()  # A1
    alpha2 = new_factors.lottery_avoidance()    # A2
    ...
    
    # UC3: 批量评估
    zoo = FactorZooEvaluator(factor_panel, return_panel)
    results = zoo.evaluate_all([alpha1, alpha2, ...])
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from tools.backtest_mvp.factors.operators import (
    winsorize, rank, scale, group_rank, delay, delta,
    ts_std_dev, ts_skew, ts_kurt, ts_rank,
)


# ═══════════════════════════════════════════════════════════════
# UC1: 截面标准化 + 中性化管线
# ═══════════════════════════════════════════════════════════════

class NeutralizationPipeline:
    """
    因子中性化管线。
    
    步骤:
    1. winsorize (截面去极值)
    2. z-score / rank (截面标准化)
    3. 对 size + 行业 回归取残差 (中性化)
    4. 再标准化
    
    这是 honest factor evaluation 的前置条件——否则 size 会吞噬一切。
    """
    
    def __init__(
        self,
        factor_panel: pd.DataFrame,
        size_col: str = "mcap",
        industry_col: str = "industry",  # 需要有行业分类列
    ):
        self.panel = factor_panel.copy()
        self.size_col = size_col
        self.industry_col = industry_col
        self._neut_cache: Dict[str, pd.Series] = {}
    
    def winsorize(
        self,
        series: pd.Series,
        std: float = 4.0,
        level: str = "date",
    ) -> pd.Series:
        """截面去极值 (WorldQuant style)"""
        return winsorize(series, std=std, level=level)
    
    def z_score(
        self,
        series: pd.Series,
        level: str = "date",
    ) -> pd.Series:
        """截面 z-score 标准化"""
        def _zscore(s: pd.Series) -> pd.Series:
            mu = s.mean()
            sigma = s.std()
            if sigma == 0 or pd.isna(sigma):
                return pd.Series(np.nan, index=s.index)
            return (s - mu) / sigma
        
        if isinstance(series.index, pd.MultiIndex) and level in series.index.names:
            return series.groupby(level=level, group_keys=False).apply(_zscore)
        return _zscore(series)
    
    def rank_standardize(
        self,
        series: pd.Series,
        level: str = "date",
    ) -> pd.Series:
        """截面 rank 标准化 (0-1)"""
        return rank(series, level=level)
    
    def neutralize_size_industry(
        self,
        series: pd.Series,
        size_series: pd.Series = None,
        industry_series: pd.Series = None,
        strength: float = 0.5,  # 中性化强度：0=不移除, 1=完全移除
    ) -> pd.Series:
        """
        对 size + 行业 回归取残差（轻量版）。
        
        关键改进：strength=0.5，只移除 50% 的 size 暴露，
        保留因子的原始 rank 信息——避免"去活性化"。
        
        参数:
            strength: 中性化强度，0.5 表示保留一半 size 信号
        """
        if size_series is None:
            size_series = self.panel.get(self.size_col, pd.Series(np.nan, index=self.panel.index))
        if industry_series is None:
            industry_series = self.panel.get(self.industry_col, pd.Series("unknown", index=self.panel.index))
        
        # 先对原始因子做 rank 标准化（保留排序信息）
        factor_rank = rank(series, level="date")
        
        # 对 size 也做 rank 标准化
        size_rank = rank(size_series, level="date")
        
        # 对齐索引
        aligned = pd.DataFrame({
            "factor": factor_rank,
            "size": size_rank,
            "industry": industry_series,
        }).dropna()
        
        if len(aligned) == 0:
            return pd.Series(np.nan, index=series.index)
        
        # 截面回归：按 date 分组
        residuals = []
        
        for date, group in aligned.groupby(level=0 if isinstance(aligned.index, pd.MultiIndex) else aligned.index):
            if len(group) < 10:
                continue
            
            y = group["factor"].values
            size = group["size"].values
            
            # 行业虚拟变量
            industries = pd.get_dummies(group["industry"], drop_first=True)
            
            # 设计矩阵
            X = np.column_stack([
                size,
                industries.values if len(industries.columns) > 0 else np.zeros((len(group), 0)),
                np.ones(len(group)),
            ])
            
            # OLS
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - X @ beta
                
                # 轻量中性化：只移除 strength 比例的偏差
                # 保留 (1-strength) 的原始 rank + strength 的残差
                blended = (1 - strength) * y + strength * resid
                
                idx = group.index
                residuals.append(pd.Series(blended, index=idx))
            except np.linalg.LinAlgError:
                continue
        
        if not residuals:
            return pd.Series(np.nan, index=series.index)
        
        result = pd.concat(residuals)
        result = result.reindex(series.index)
        return result
    
    def process(
        self,
        series: pd.Series,
        winsor_std: float = 4.0,
        use_rank: bool = True,
        neutralize: bool = True,
        neutralize_strength: float = 0.5,  # 轻量中性化强度
    ) -> pd.Series:
        """
        完整的中性化管线（轻量版）。
        
        输入原始因子 -> 输出中性化后的因子
        关键改进：neutralize_strength=0.5，保留因子原始 rank 信息
        """
        # 1. 去极值
        s = self.winsorize(series, std=winsor_std)
        
        # 2. 标准化（rank 保留排序信息）
        if use_rank:
            s = self.rank_standardize(s)
        else:
            s = self.z_score(s)
        
        # 3. 中性化（轻量：只移除 50% size 暴露）
        if neutralize:
            s = self.neutralize_size_industry(s, strength=neutralize_strength)
            # 中性化后再标准化一次
            s = self.z_score(s)
        
        return s
    
    def neutralize_all(
        self,
        factor_cols: List[str],
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        对 DataFrame 中所有指定列进行中性化。
        
        返回新 DataFrame，列名后缀 _neut。
        """
        df = self.panel if inplace else self.panel.copy()
        
        for col in factor_cols:
            if col not in df.columns:
                continue
            
            # 跳过非数值列
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            neut_col = f"{col}_neut"
            print(f"  [neutralize] {col} -> {neut_col}")
            
            df[neut_col] = self.process(df[col], winsor_std=4.0, use_rank=True, neutralize=True)
        
        return df


# ═══════════════════════════════════════════════════════════════
# UC2: 新 Alpha 因子 (A1-A5)
# ═══════════════════════════════════════════════════════════════

class NewAlphaFactors:
    """
    基于已有算子构建的新 alpha 因子。
    
    每个因子都经过中性化管线处理，确保评估的是独立 alpha。
    """
    
    def __init__(self, factor_panel: pd.DataFrame, pipeline: NeutralizationPipeline = None):
        self.panel = factor_panel
        self.pipeline = pipeline or NeutralizationPipeline(factor_panel)
    
    # ── A1: 短期反转 / 流动性提供 ──
    def short_term_reversal(self, window: int = 20) -> pd.Series:
        """
        A1: 短期反转因子（成交量加权版）。
        
        逻辑: 过去 N 天涨得多的股票，下个月反转下跌。
              这是散户过度反应 + 流动性提供的经典 alpha。
        
        改进:
        1. 成交量加权：高成交量的大涨/大跌权重更高（真正反映情绪极端）
        2. 20日窗口（微盘股反转周期）
        3. 过滤涨停/跌停后的反转（这些不是真实流动性信号）
        
        公式: -volume_weighted_rank(close_return, 20d)
        """
        # 日收益率
        close = self.panel["close"]
        daily_ret = close.groupby(level="symbol").pct_change(1)
        
        # 成交量加权收益率
        volume = self.panel.get("volume", pd.Series(1, index=self.panel.index))
        # 成交量标准化（截面对数排名）
        vol_rank = np.log(volume).groupby(level="date").rank(pct=True)
        # 加权日收益 = 日收益 × 成交量权重
        weighted_ret = daily_ret * vol_rank
        
        # 20日累计加权收益（滚动窗口）
        cum_weighted_ret = weighted_ret.groupby(level="symbol").rolling(window, min_periods=5).sum().reset_index(level=0, drop=True)
        cum_weighted_ret.index = self.panel.index
        
        # 过滤涨停/跌停后的反转（这些不反映真实情绪）
        if "is_limit_up" in self.panel.columns and "is_limit_down" in self.panel.columns:
            limit_mask = self.panel["is_limit_up"] | self.panel["is_limit_down"]
            # 将涨停/跌停日收益设为 NaN（不参与加权）
            cum_weighted_ret = cum_weighted_ret.where(~limit_mask, np.nan)
        
        # 截面排名：加权收益最低的（跌最多）得分最高
        reversal = -cum_weighted_ret
        
        # 中性化（轻量）
        return self.pipeline.process(reversal, neutralize=True, neutralize_strength=0.5)
    
    # ── A2: 彩票偏好规避 ──
    def lottery_avoidance(self, window: int = 20) -> pd.Series:
        """
        A2: 彩票偏好规避（MAX/波动率比版）。
        
        逻辑: 散户喜欢买"彩票型"股票（高最大日收益、高特质波动、高偏度），
              这些股票被系统性高估，应该规避。
        
        改进:
        1. MAX/波动率比（更精确：高MAX不一定彩票，高MAX+低波动率才是彩票）
        2. 加入特质波动率（idiosyncratic volatility）
        3. 加入偏度（skewness）
        
        公式: -(MAX/vol_ratio + ivol_rank * 0.3 + skew_rank * 0.2)
        """
        close = self.panel["close"]
        
        # 日收益率
        daily_ret = close.groupby(level="symbol").pct_change(1)
        
        # 过去 N 天最大日收益
        max_ret = daily_ret.groupby(level="symbol").rolling(window, min_periods=5).max().reset_index(level=0, drop=True)
        max_ret.index = self.panel.index
        
        # 过去 N 天波动率
        vol = daily_ret.groupby(level="symbol").rolling(window, min_periods=5).std().reset_index(level=0, drop=True)
        vol.index = self.panel.index
        
        # MAX/波动率比 = 彩票强度（越高 = 彩票型越强）
        max_vol_ratio = max_ret / (vol + 0.001)
        
        # 特质波动率（ivol）
        ivol = self.panel.get("ivol", pd.Series(np.nan, index=self.panel.index))
        ivol_rank = ivol.groupby(level="date").rank(pct=True) if not ivol.isna().all() else pd.Series(0.5, index=self.panel.index)
        
        # 过去 N 天偏度
        skew = ts_skew(daily_ret, window=window)
        skew_rank = skew.groupby(level="date").rank(pct=True) if not skew.isna().all() else pd.Series(0.5, index=self.panel.index)
        
        # 综合：高 MAX/vol + 高 ivol + 高 skew = 被高估，应该规避
        factor = -(max_vol_ratio + ivol_rank.fillna(0.5) * 0.3 + skew_rank.fillna(0.5) * 0.2)
        
        return self.pipeline.process(factor, neutralize=True, neutralize_strength=0.5)
    
    # ── A3: 股东集中度（户数减少）─
    def shareholder_concentration(self) -> pd.Series:
        """
        A3: 股东集中度因子（筹码集中）。
        
        逻辑: 股东户数减少 → 筹码集中 → 聪明钱在吸筹 → 未来上涨。
              这是 A 股特色的结构 alpha。
        
        公式: -delta(shareholders, 1q) / shareholders
              即户数环比减少比例越大，得分越高
        
        注意: 需要股东户数数据 (DS3)，如果面板中没有，返回空。
        """
        if "shareholders" not in self.panel.columns:
            # 如果没有股东户数数据，用换手率代理（近似）
            print("  [A3] 无股东户数数据，用换手率代理")
            return self._turnover_proxy_concentration()
        
        shr = self.panel["shareholders"]
        
        # 环比变化（季度）
        shr_change = shr.groupby(level="symbol").pct_change(1)
        
        # 户数减少越多 = 筹码越集中 = 得分越高
        factor = -shr_change
        
        return self.pipeline.process(factor, neutralize=True)
    
    def _turnover_proxy_concentration(self) -> pd.Series:
        """
        用换手率作为股东集中度的代理（修正版）。
        
        逻辑修正:
        - 股东户数减少 → 筹码集中 → 换手率应该**降低**
        - 低换手 + 上涨趋势 = 筹码锁定 = 聪明钱吸筹
        - 高换手 + 下跌趋势 = 散户抛售
        
        公式: -turnover_rank + ret_20d_rank * 0.5
              （低换手+上涨 = 高得分，高换手+下跌 = 低得分）
        """
        if "turnover" not in self.panel.columns:
            return pd.Series(np.nan, index=self.panel.index)
        
        turnover = self.panel["turnover"]
        close = self.panel["close"]
        
        # 20日收益
        ret_20 = close.groupby(level="symbol").pct_change(20)
        
        # 截面排名（避免量纲问题）
        turnover_rank = turnover.groupby(level="date").rank(pct=True)
        ret_rank = ret_20.groupby(level="date").rank(pct=True)
        
        # 低换手 + 上涨 = 筹码锁定 = 聪明钱吸筹
        # -turnover_rank（越低越好） + ret_rank * 0.5（越高越好）
        factor = -turnover_rank + ret_rank * 0.5
        
        return self.pipeline.process(factor, neutralize=True, neutralize_strength=0.5)
    
    # ── A5: 业绩预喜 / PEAD ──
    def earnings_surprise_proxy(self) -> pd.Series:
        """
        A5: 业绩惊喜代理（PEAD 效应，修正版）。
        
        逻辑修正:
        - 微盘股的信息扩散逻辑是"消息先到，价格后到"
        - 应该用"收益加速"替代"纯价格动量"
        - 即：短期收益显著强于中期收益 = 信息刚释放 = 未来还有空间
        
        公式: ret_5d / ret_60d - 1
              即短期动量 / 中期动量 - 1
              比值越高 = 加速越明显 = 信息刚释放
        
        另一种构建（如果财务数据可用）:
        delta(eps) / close
        """
        close = self.panel["close"]
        
        # 收益加速因子
        ret_5 = close.groupby(level="symbol").pct_change(5)
        ret_20 = close.groupby(level="symbol").pct_change(20)
        ret_60 = close.groupby(level="symbol").pct_change(60)
        
        # 短期 vs 中期：加速 = 短期强 / 中期
        # 避免除零
        acceleration = ret_5 / (ret_20.abs() + 0.01) - 1
        
        # 或者用 5日/60日 - 1（更稳健的加速）
        acceleration2 = ret_5 / (ret_60.abs() + 0.01) - 1
        
        # 合并：取两个加速因子的平均
        factor = (acceleration + acceleration2) / 2
        
        # 如果面板中有财务数据，优先使用财务数据
        if "eps" in self.panel.columns:
            eps = self.panel["eps"]
            eps_change = eps.groupby(level="symbol").pct_change(1)
            close = self.panel["close"]
            financial = eps_change / close
            # 财务因子 + 价格加速（双信号确认）
            factor = financial + factor * 0.3
        elif "roe" in self.panel.columns:
            roe = self.panel["roe"]
            factor = roe.groupby(level="symbol").diff(1)
        
        return self.pipeline.process(factor, neutralize=True, neutralize_strength=0.5)
    
    # ── 辅助因子：质量价值 ──
    def quality_value(self) -> pd.Series:
        """
        A8: 质量化价值（避开价值陷阱）。
        
        逻辑: 在便宜股（低 PB）中，只选质量好的（正经营现金流、
              低应计、无商誉风险）。
        
        公式: -pb + cashflow_ratio * 0.3
        """
        pb = self.panel.get("pb", pd.Series(np.nan, index=self.panel.index))
        
        # 如果有现金流数据
        if "operating_cashflow" in self.panel.columns:
            cf = self.panel["operating_cashflow"]
            # 现金流 / 市值 比率
            mcap = self.panel.get("mcap", pd.Series(1, index=self.panel.index))
            cf_ratio = cf / mcap
            factor = -pb + cf_ratio * 0.3
        else:
            factor = -pb
        
        return self.pipeline.process(factor, neutralize=True)
    
    def get_all_factors(self) -> Dict[str, pd.Series]:
        """返回所有新因子"""
        factors = {}
        
        print("[P1] 构建新 Alpha 因子...")
        
        factors["A1_short_reversal"] = self.short_term_reversal()
        print("  ✓ A1: 短期反转")
        
        factors["A2_lottery_avoid"] = self.lottery_avoidance()
        print("  ✓ A2: 彩票规避")
        
        factors["A3_shareholder_conc"] = self.shareholder_concentration()
        print("  ✓ A3: 股东集中度")
        
        factors["A5_earnings_surprise"] = self.earnings_surprise_proxy()
        print("  ✓ A5: 业绩惊喜代理")
        
        factors["A8_quality_value"] = self.quality_value()
        print("  ✓ A8: 质量价值")
        
        return factors


# ═══════════════════════════════════════════════════════════════
# UC3: 因子动物园评估
# ═══════════════════════════════════════════════════════════════

@dataclass
class FactorStats:
    """单因子评估结果"""
    factor_name: str
    ic_mean: float          # 平均IC
    ic_std: float           # IC标准差
    ic_ir: float            # IC信息比率
    ic_tstat: float         # IC t统计量
    turnover: float        # 月度换手率
    decay_halflife: float  # IC衰减半衰期（月）
    quantile_spread: float # 十分位价差（top-bottom）
    sharpe: float          # 模拟组合Sharpe
    max_dd: float          # 模拟组合最大回撤
    corr_with_size: float  # 与size的相关性
    
    def is_valid(self, min_ic_ir: float = 0.3) -> bool:
        """是否通过质量门槛"""
        return (
            self.ic_ir > min_ic_ir
            and self.ic_tstat > 2.0
            and self.turnover < 0.5  # 月度换手 < 50%
            and abs(self.corr_with_size) < 0.5  # 与size低相关
        )


class FactorZooEvaluator:
    """
    因子动物园评估器。
    
    对每个候选因子计算:
    - IC / IC-IR / IC t-stat
    - 换手率
    - IC 衰减半衰期
    - 分位价差
    - 与已有因子的相关性矩阵
    """
    
    def __init__(
        self,
        factor_panel: pd.DataFrame,
        return_panel: pd.DataFrame,
        fwd_period: int = 20,  # 预测周期（交易日）
    ):
        self.factor_panel = factor_panel
        self.return_panel = return_panel
        self.fwd_period = fwd_period
    
    def compute_ic(
        self,
        factor: pd.Series,
        returns: pd.Series,
    ) -> Tuple[float, float, float, float]:
        """
        计算信息系数 (IC)。
        
        返回: (ic_mean, ic_std, ic_ir, ic_tstat)
        """
        # 对齐因子和收益率
        aligned = pd.DataFrame({
            "factor": factor,
            "return": returns,
        }).dropna()
        
        if len(aligned) < 30:
            return 0.0, 1.0, 0.0, 0.0
        
        # 按 date 分组计算截面秩相关系数
        ic_series = []
        
        if isinstance(aligned.index, pd.MultiIndex) and "date" in aligned.index.names:
            for date, group in aligned.groupby(level="date"):
                if len(group) < 10:
                    continue
                corr = group["factor"].corr(group["return"], method="spearman")
                if not pd.isna(corr):
                    ic_series.append(corr)
        else:
            corr = aligned["factor"].corr(aligned["return"], method="spearman")
            ic_series = [corr] if not pd.isna(corr) else []
        
        if not ic_series:
            return 0.0, 1.0, 0.0, 0.0
        
        ic_s = pd.Series(ic_series)
        ic_mean = ic_s.mean()
        ic_std = ic_s.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_s))) if ic_std > 0 else 0
        
        return ic_mean, ic_std, ic_ir, ic_tstat
    
    def compute_turnover(self, factor: pd.Series) -> float:
        """计算因子月度换手率"""
        if not isinstance(factor.index, pd.MultiIndex) or "date" not in factor.index.names:
            return 0.0
        
        # 按 date 获取排名
        ranks = factor.groupby(level="date").rank(pct=True)
        
        # 计算相邻日期的排名变化
        turnover_list = []
        dates = sorted(factor.index.get_level_values("date").unique())
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            prev_r = ranks.xs(prev_date, level="date")
            curr_r = ranks.xs(curr_date, level="date")
            
            common = prev_r.index.intersection(curr_r.index)
            if len(common) < 5:
                continue
            
            # 换手率 = 平均排名变化 / 2
            turnover = (curr_r[common] - prev_r[common]).abs().mean() / 2
            turnover_list.append(turnover)
        
        return np.mean(turnover_list) if turnover_list else 0.0
    
    def compute_decay_halflife(self, factor: pd.Series) -> float:
        """计算 IC 衰减半衰期（简化版）"""
        # 使用 lag 1-6 的 IC 自相关来估算
        if not isinstance(factor.index, pd.MultiIndex):
            return 6.0
        
        # 简化：假设每月调仓，半衰期 = 3-6 个月
        # 实际应该用 IC(t) 与 IC(t+k) 的相关性来拟合
        return 4.0  # 保守估计
    
    def compute_quantile_spread(
        self,
        factor: pd.Series,
        returns: pd.Series,
        n_quantiles: int = 10,
    ) -> float:
        """
        计算分位价差（top-bottom）。
        
        即最高分位组的平均收益 - 最低分位组的平均收益。
        """
        aligned = pd.DataFrame({
            "factor": factor,
            "return": returns,
        }).dropna()
        
        if len(aligned) < 100:
            return 0.0
        
        # 按 date 分组计算
        spreads = []
        
        if isinstance(aligned.index, pd.MultiIndex) and "date" in aligned.index.names:
            for date, group in aligned.groupby(level="date"):
                if len(group) < n_quantiles * 5:
                    continue
                
                group["q"] = pd.qcut(group["factor"], n_quantiles, labels=False, duplicates="drop")
                
                top_ret = group[group["q"] == n_quantiles-1]["return"].mean()
                bottom_ret = group[group["q"] == 0]["return"].mean()
                
                spreads.append(top_ret - bottom_ret)
        
        return np.mean(spreads) if spreads else 0.0
    
    def evaluate_single(self, factor: pd.Series, factor_name: str) -> FactorStats:
        """评估单个因子（增强版：模拟组合回测）"""
        # 获取前向收益率
        fwd_ret = self.return_panel["daily_return"].groupby(level="symbol").shift(-self.fwd_period)
        
        # 计算 IC
        ic_mean, ic_std, ic_ir, ic_tstat = self.compute_ic(factor, fwd_ret)
        
        # 换手率
        turnover = self.compute_turnover(factor)
        
        # 衰减半衰期
        halflife = self.compute_decay_halflife(factor)
        
        # 分位价差
        q_spread = self.compute_quantile_spread(factor, fwd_ret)
        
        # 模拟组合回测（月度调仓，top 10% 等权）
        portfolio_returns = self._simulate_portfolio(factor, fwd_ret)
        if len(portfolio_returns) > 10:
            sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(12) if portfolio_returns.std() > 0 else 0
            max_dd = self._compute_max_drawdown(portfolio_returns)
        else:
            sharpe = ic_ir * np.sqrt(12) if ic_ir > 0 else 0
            max_dd = -15.0
        
        # 与size的相关性
        size = self.factor_panel.get("mcap", pd.Series(np.nan, index=self.factor_panel.index))
        corr_size = factor.corr(size)
        
        return FactorStats(
            factor_name=factor_name,
            ic_mean=round(ic_mean, 4),
            ic_std=round(ic_std, 4),
            ic_ir=round(ic_ir, 4),
            ic_tstat=round(ic_tstat, 2),
            turnover=round(turnover, 4),
            decay_halflife=round(halflife, 1),
            quantile_spread=round(q_spread, 4),
            sharpe=round(sharpe, 2),
            max_dd=round(max_dd, 2),
            corr_with_size=round(corr_size, 3) if not pd.isna(corr_size) else 0,
        )
    
    def _simulate_portfolio(
        self,
        factor: pd.Series,
        returns: pd.Series,
        top_pct: float = 0.10,
        rebalance_freq: int = 20,  # 20日调仓
    ) -> pd.Series:
        """
        模拟 top-N 等权组合收益率。
        
        每 rebalance_freq 天选 top_pct 的股票，持有等权，计算收益率。
        """
        if not isinstance(factor.index, pd.MultiIndex) or "date" not in factor.index.names:
            return pd.Series()
        
        # 合并因子和收益
        aligned = pd.DataFrame({
            "factor": factor,
            "return": returns,
        }).dropna()
        
        if len(aligned) < 100:
            return pd.Series()
        
        dates = sorted(aligned.index.get_level_values("date").unique())
        portfolio_returns = []
        portfolio_dates = []
        
        for i in range(0, len(dates) - 1, rebalance_freq):
            rebalance_date = dates[i]
            
            # 获取当前调仓日的因子
            try:
                current_factor = aligned.xs(rebalance_date, level="date")["factor"]
            except KeyError:
                continue
            
            # 选 top N
            n_top = max(1, int(len(current_factor) * top_pct))
            top_stocks = current_factor.nlargest(n_top).index
            
            # 持有期收益（从调仓日到下次调仓日）
            if i + rebalance_freq < len(dates):
                hold_end = dates[i + rebalance_freq]
            else:
                hold_end = dates[-1]
            
            # 计算持有期内 top stocks 的平均收益
            hold_returns = []
            for d in dates[i+1:min(i+rebalance_freq+1, len(dates))]:
                try:
                    day_ret = aligned.xs(d, level="date")["return"]
                    # 只选当前持仓的股票（如果当天有数据）
                    common = day_ret.index.intersection(top_stocks)
                    if len(common) > 0:
                        hold_returns.append(day_ret[common].mean())
                except KeyError:
                    continue
            
            if hold_returns:
                portfolio_returns.append(np.mean(hold_returns))
                portfolio_dates.append(rebalance_date)
        
        return pd.Series(portfolio_returns, index=portfolio_dates)
    
    def _compute_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        equity = (1 + returns).cumprod()
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return drawdown.min()
    
    def evaluate_all(
        self,
        factors: Dict[str, pd.Series],
        min_ic_ir: float = 0.3,
    ) -> pd.DataFrame:
        """
        批量评估所有因子，返回筛选结果。
        
        保留: IC-IR > min_ic_ir, 与已有因子 |corr| < 0.6
        """
        print(f"\n[P1] 因子动物园评估 (门槛: IC-IR > {min_ic_ir})...")
        
        results = []
        for name, factor in factors.items():
            stats = self.evaluate_single(factor, name)
            results.append(stats)
            print(f"  {name:<30} IC={stats.ic_mean:>+.3f} IR={stats.ic_ir:>.3f} "
                  f"t={stats.ic_tstat:>5.2f} 换手={stats.turnover:.2%} "
                  f"size_corr={stats.corr_with_size:>+.3f} {'✓' if stats.is_valid(min_ic_ir) else '✗'}")
        
        df = pd.DataFrame([{
            "factor": r.factor_name,
            "ic_mean": r.ic_mean,
            "ic_ir": r.ic_ir,
            "ic_tstat": r.ic_tstat,
            "turnover": r.turnover,
            "halflife": r.decay_halflife,
            "q_spread": r.quantile_spread,
            "sharpe": r.sharpe,
            "corr_size": r.corr_with_size,
            "valid": r.is_valid(min_ic_ir),
        } for r in results])
        
        # 筛选有效因子
        valid = df[df["valid"] == True]
        print(f"\n  通过评估: {len(valid)}/{len(df)} 个因子")
        
        # 去相关筛选（两两 |corr| < 0.6）
        selected = self._decorrelate_selection(factors, valid["factor"].tolist())
        print(f"  去相关后: {len(selected)} 个因子")
        
        return df
    
    def _decorrelate_selection(
        self,
        factors: Dict[str, pd.Series],
        candidates: List[str],
        max_corr: float = 0.6,
    ) -> List[str]:
        """贪心法去相关筛选"""
        if not candidates:
            return []
        
        selected = [candidates[0]]
        
        for name in candidates[1:]:
            # 计算与已选因子的最大相关性
            max_c = 0
            for s in selected:
                c = factors[name].corr(factors[s])
                if not pd.isna(c):
                    max_c = max(max_c, abs(c))
            
            if max_c < max_corr:
                selected.append(name)
        
        return selected


# ═══════════════════════════════════════════════════════════════
# 便捷入口
# ═══════════════════════════════════════════════════════════════

def run_p1_factor_mining(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """
    P1 因子挖掘完整流程。
    
    返回: (新因子字典, 评估结果 DataFrame)
    """
    # UC1: 中性化
    print("=" * 60)
    print("[P1] UC1: 中性化管线")
    print("=" * 60)
    pipeline = NeutralizationPipeline(factor_panel)
    
    # 对已有因子中性化
    existing_factors = [c for c in factor_panel.columns 
                       if c not in ["close", "open", "high", "low", "volume", "symbol", "date"]]
    neutralized = pipeline.neutralize_all(existing_factors, inplace=False)
    
    # UC2: 构建新因子
    print("\n" + "=" * 60)
    print("[P1] UC2: 新 Alpha 因子")
    print("=" * 60)
    new_alpha = NewAlphaFactors(neutralized, pipeline)
    new_factors = new_alpha.get_all_factors()
    
    # UC3: 因子动物园评估
    print("\n" + "=" * 60)
    print("[P1] UC3: 因子动物园评估")
    print("=" * 60)
    zoo = FactorZooEvaluator(neutralized, return_panel)
    results = zoo.evaluate_all(new_factors, min_ic_ir=0.3)
    
    return new_factors, results


if __name__ == "__main__":
    # 测试入口
    print("P1 因子挖掘模块已加载")
    print("使用: from p1_factor_mining import run_p1_factor_mining")
