"""ADV (平均日成交额) 冲击模型 + 容量管理"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


@dataclass
class ADVImpactModel:
    """
    ADV 平方根冲击成本模型。
    
    公式: impact_bps = k * sqrt(order_value / ADV_value)
    
    参考:
    - Almgren et al. (2005) "Direct Estimation of Equity Market Impact"
    - Kissell (2006) "The Expanded Implementation Shortfall"
    - A股微盘实证: k ≈ 5-15 (0.5%-1.5% per sqrt(1/ADV))
    
    参数:
    - k: 冲击系数 (bps), 默认 8 (0.8%) = 中等微盘流动性
    - adv_window: 计算 ADV 的窗口天数, 默认 20
    - max_position_adv_pct: 单票持仓占 ADV 上限, 默认 5%
    """
    k: float = 8.0  # 冲击系数 (bps)
    adv_window: int = 20  # 计算 ADV 的窗口天数
    max_position_adv_pct: float = 5.0  # 单票持仓占 ADV 上限 (%)
    
    def estimate_adv(self, price_data: pd.DataFrame) -> pd.Series:
        """
        从日线数据估算 ADV (平均日成交额)。
        
        Args:
            price_data: DataFrame with columns [date, close, volume]
                       MultiIndex: (date, symbol) or symbol level
        
        Returns:
            Series indexed by (date, symbol) with ADV values
        """
        # 如果输入是 MultiIndex
        if isinstance(price_data.index, pd.MultiIndex) and "date" in price_data.index.names:
            # 按 symbol 分组计算滚动 ADV
            adv = price_data.groupby(level=1).apply(
                lambda x: x["close"] * x["volume"] * x["close"].iloc[0] / x["close"].iloc[0]  # 成交额
            ).reset_index(level=0, drop=True)
            # TODO: 实际上这里需要更仔细地处理
        else:
            # 单只股票的日线
            amount = price_data["close"] * price_data["volume"]  # 成交额
            adv = amount.rolling(self.adv_window, min_periods=5).mean()
        
        return adv
    
    def compute_impact(
        self,
        order_value: float,  # 下单金额 (元)
        adv_value: float,    # 平均日成交额 (元)
    ) -> float:
        """
        计算给定订单对 ADV 的冲击成本 (bps)。
        
        Returns:
            冲击成本 (百分比), 如 0.005 = 0.5%
        """
        if adv_value <= 0:
            return 0.0
        
        ratio = order_value / adv_value
        if ratio <= 0:
            return 0.0
        
        impact_bps = self.k * np.sqrt(ratio)
        return impact_bps / 10000  # 转换为百分比
    
    def compute_position_impact(
        self,
        position_value: float,  # 持仓金额
        adv_value: float,       # 该股票的 ADV
        is_entry: bool = True,  # 入场/出场
    ) -> float:
        """
        计算单只股票的持仓冲击成本。
        
        假设等分成 2 天执行 (入场/出场各一天):
        - 入场: order_value = position_value / 2
        - 出场: order_value = position_value / 2
        
        总冲击 = 入场冲击 + 出场冲击
        """
        order_value = position_value / 2  # 假设每天执行一半
        one_side = self.compute_impact(order_value, adv_value)
        
        if is_entry:
            return one_side  # 只算入场
        return one_side * 2  # 入场 + 出场
    
    def check_capacity(
        self,
        position_value: float,
        adv_value: float,
    ) -> Tuple[bool, float, float]:
        """
        检查持仓是否超出容量限制。
        
        Returns:
            (is_safe, position_pct_of_adv, max_allowed_value)
        """
        if adv_value <= 0:
            return False, 0.0, 0.0
        
        position_pct = position_value / adv_value * 100
        max_allowed = adv_value * (self.max_position_adv_pct / 100)
        
        is_safe = position_pct <= self.max_position_adv_pct
        
        return is_safe, position_pct, max_allowed


class PortfolioCapacityChecker:
    """
    组合容量检查器。
    
    检查整个组合是否在容量限制内, 返回容量报告。
    """
    
    def __init__(self, adv_model: ADVImpactModel = None):
        self.adv_model = adv_model or ADVImpactModel()
    
    def check_portfolio(
        self,
        positions: Dict[str, float],  # {symbol: weight} 权重 (0-1)
        portfolio_value: float,        # 组合总价值 (万元)
        adv_series: pd.Series,          # 各股票的 ADV
    ) -> dict:
        """
        检查整个组合的容量。
        
        Args:
            positions: 持仓权重
            portfolio_value: 组合总价值 (万元)
            adv_series: 各股票的 ADV (元), index=symbol
        
        Returns:
            {
                "total_wan": float,
                "is_safe": bool,
                "violations": List[str],  # 超限股票
                "avg_impact_bps": float,  # 平均冲击成本
                "capacity_score": float,  # 容量评分 (0-1)
                "min_adv_wan": float,     # 最小 ADV (万元)
            }
        """
        total_value = portfolio_value * 10000  # 万元 -> 元
        
        violations = []
        impacts = []
        total_position_value = 0
        min_adv = float('inf')
        
        for symbol, weight in positions.items():
            position_value = total_value * weight
            adv = adv_series.get(symbol, 0)
            
            if adv > 0:
                min_adv = min(min_adv, adv)
                
                is_safe, pct, max_allowed = self.adv_model.check_capacity(
                    position_value, adv
                )
                
                impact = self.adv_model.compute_impact(position_value, adv)
                impacts.append(impact)
                total_position_value += position_value
                
                if not is_safe:
                    violations.append(symbol)
        
        avg_impact = np.mean(impacts) if impacts else 0
        
        # 容量评分: 1 - 平均冲击/1% (越低越差)
        capacity_score = max(0, 1 - avg_impact / 0.01)
        
        return {
            "total_wan": portfolio_value,
            "is_safe": len(violations) == 0,
            "violations": violations,
            "avg_impact_bps": round(avg_impact * 10000, 2),
            "capacity_score": round(capacity_score, 3),
            "min_adv_wan": round(min_adv / 10000, 2) if min_adv != float('inf') else 0,
        }


def compute_impact_cost(
    replaced_frac: float,  # 调仓替换比例
    order_value_per_stock: float,  # 每只新股的下单金额
    adv_series: pd.Series,  # 各股票的 ADV
    picked: List[str],  # 新选中的股票
    k: float = 8.0,  # 冲击系数
) -> float:
    """
    计算调仓日的 ADV 冲击成本。
    
    替代 engine 中固定的 self.slippage * replaced_frac
    
    实际冲击 = mean(impact_bps(stock_i)) * replaced_frac
    
    Args:
        replaced_frac: 调仓替换比例 (0-1)
        order_value_per_stock: 每只新股的下单金额 (元)
        adv_series: 各股票的 ADV
        picked: 新选中的股票列表
        k: 冲击系数
    
    Returns:
        组合层面的冲击成本 (百分比)
    """
    if not picked or adv_series.empty:
        return 0.0
    
    model = ADVImpactModel(k=k)
    impacts = []
    
    for symbol in picked:
        adv = adv_series.get(symbol, 0)
        if adv > 0:
            impact = model.compute_impact(order_value_per_stock, adv)
            impacts.append(impact)
    
    if not impacts:
        return 0.0
    
    avg_impact = np.mean(impacts)
    
    # 组合层面的冲击 = 平均冲击 * 替换比例
    # 因为不是所有持仓都换, 只有新入场的股票需要冲击成本
    return avg_impact * replaced_frac


def compute_total_cost_with_impact(
    commission: float,  # 佣金比例
    slippage_fixed: float,  # 固定滑点 (保留)
    replaced_frac: float,
    order_value_per_stock: float,
    adv_series: pd.Series,
    picked: List[str],
    k: float = 8.0,
) -> float:
    """
    计算调仓日总成本 (佣金 + 固定滑点 + ADV 冲击)。
    
    用于 engine.run() 中第 1 天的 portfolio_return 调整。
    
    Returns:
        总成本 (百分比, 需要减去)
    """
    # 佣金
    total_cost = commission
    
    # 固定滑点 (保留但降低, 作为安全网)
    total_cost += slippage_fixed * 0.3  # 只保留 30% 作为基础成本
    
    # ADV 冲击
    adv_impact = compute_impact_cost(
        replaced_frac, order_value_per_stock, adv_series, picked, k
    )
    total_cost += adv_impact
    
    return total_cost
