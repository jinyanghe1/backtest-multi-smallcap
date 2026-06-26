"""Deflated Sharpe Ratio — 多重检验折扣

基于 Bailey & Lopez de Prado (2012) 的 Deflated Sharpe Ratio 公式。

核心思想: 当你跑 N 个策略只报最好的 Sharpe, 需要对"最优"Sharpe 做折扣,
因为最优值天然向上偏。

公式:
    DSR = N( (Sharpe_best - E[Sharpe]) / sqrt(Var(Sharpe)) * f(N, K) )

其中 N(.) 是标准正态 CDF, E[Sharpe] 和 Var(Sharpe) 是全部试验的均值和方差,
f(N, K) 是多重检验折扣因子。

更实用的版本 (简化版):
    DSR ≈ Sharpe_best * (1 - ln(N_trials) / (4 * Sharpe_best * sqrt(N_periods)))

或更保守:
    DSR = Sharpe_best - ln(N_trials) / (4 * sqrt(N_periods))

参考:
    - Bailey, D. H., & Lopez de Prado, M. (2012). "The Sharpe Ratio
      Efficient Frontier." Journal of Risk, 15(2), 3-44.
    - Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated
      Sharpe Ratio: Correcting for Selection Bias, Backtest
      Overfitting and Non-Normality." Journal of Portfolio
      Management, 40(5), 94-107.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


class TrialCounter:
    """
    试验次数计数器。
    
    记录每次回测/策略优化尝试, 确保 Deflated Sharpe 知道 N_trials。
    
    使用示例:
        counter = TrialCounter()
        for each strategy run:
            counter.add_trial(strategy_name, params, result)
        
        dsr = compute_deflated_sharpe(
            sharpe=best_sharpe,
            n_trials=counter.n_total,
            n_periods=92,  # 月数
        )
    """
    
    def __init__(self):
        self._trials = []  # 每次试验的记录
    
    def add_trial(
        self,
        strategy_name: str,
        params: dict,
        sharpe: float,
        annual_return: float = 0.0,
        max_drawdown: float = 0.0,
        is_final: bool = False,  # 是否用于最终评估
    ) -> None:
        """记录一次试验"""
        self._trials.append({
            "strategy": strategy_name,
            "params": params,
            "sharpe": sharpe,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "is_final": is_final,
        })
    
    @property
    def n_total(self) -> int:
        """总试验次数"""
        return len(self._trials)
    
    @property
    def n_strategies(self) -> int:
        """不同策略的数量"""
        return len(set(t["strategy"] for t in self._trials))
    
    @property
    def n_param_combinations(self) -> int:
        """参数组合数"""
        return len(set(
            str(sorted(t["params"].items())) for t in self._trials
        ))
    
    @property
    def sharpe_series(self) -> pd.Series:
        """所有试验的 Sharpe 序列"""
        return pd.Series([t["sharpe"] for t in self._trials])
    
    def best_result(self) -> dict:
        """返回最优试验"""
        if not self._trials:
            return {}
        best = max(self._trials, key=lambda x: x["sharpe"])
        return best
    
    def summary(self) -> dict:
        """返回试验摘要"""
        if not self._trials:
            return {"n_total": 0}
        
        sharpe_s = self.sharpe_series
        return {
            "n_total": self.n_total,
            "n_strategies": self.n_strategies,
            "n_param_combinations": self.n_param_combinations,
            "sharpe_mean": round(sharpe_s.mean(), 4),
            "sharpe_std": round(sharpe_s.std(), 4),
            "sharpe_max": round(sharpe_s.max(), 4),
            "sharpe_min": round(sharpe_s.min(), 4),
            "best_strategy": self.best_result().get("strategy", ""),
        }


def compute_deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_periods: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    method: str = "bailey_prado",
) -> float:
    """
    计算 Deflated Sharpe Ratio。
    
    Args:
        sharpe: 报告的最优 Sharpe
        n_trials: 试验次数 (跑过的策略/参数组合数)
        n_periods: 样本期数 (月数, 例如 92 = 7.6年)
        skew: 收益率偏度 (默认 0 = 正态)
        kurt: 收益率峰度 (默认 3 = 正态)
        method: "bailey_prado" (标准) | "conservative" (更保守)
    
    Returns:
        Deflated Sharpe Ratio (可能为负)
    
    Raises:
        ValueError: 如果参数无效
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_periods < 2:
        raise ValueError("n_periods must be >= 2")
    
    if method == "bailey_prado":
        # Bailey & Lopez de Prado (2014) 简化公式
        # DSR = Sharpe - ln(N_trials) / (4 * sqrt(N_periods))
        # 这是更保守的下界估计
        
        penalty = np.log(n_trials) / (4 * np.sqrt(n_periods))
        dsr = sharpe - penalty
        
    elif method == "conservative":
        # 更保守: 考虑偏度和峰度调整
        # DSR = Sharpe * (1 - ln(N_trials) / (4 * Sharpe * sqrt(N_periods)))
        
        if sharpe <= 0:
            dsr = sharpe - np.log(n_trials) / (4 * np.sqrt(n_periods))
        else:
            penalty = np.log(n_trials) / (4 * sharpe * np.sqrt(n_periods))
            dsr = sharpe * (1 - min(penalty, 0.9))  # 上限 90% 折扣
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 考虑非正态性调整 (偏度/峰度)
    if kurt != 3.0:
        # 峰度 > 3 (厚尾) -> 进一步折扣
        tail_penalty = (kurt - 3) / 12 * 0.1  # 轻微调整
        dsr -= tail_penalty
    
    return round(dsr, 4)


def deflated_sharpe_significance(
    dsr: float,
    significance_level: float = 0.05,
) -> dict:
    """
    判断 Deflated Sharpe 是否显著。
    
    Args:
        dsr: Deflated Sharpe Ratio
        significance_level: 显著性水平 (默认 5%)
    
    Returns:
        {
            "is_significant": bool,
            "significance_level": float,
            "confidence": str,  # "high" | "moderate" | "low" | "none"
            "p_value": float,
        }
    """
    # 使用正态近似: DSR ~ N(0, 1/sqrt(N_periods))
    # p-value = P(Z > DSR) = 1 - CDF(DSR)
    
    p_value = 1 - stats.norm.cdf(dsr) if dsr > 0 else 1.0
    
    is_significant = p_value < significance_level
    
    if dsr > 1.0 and p_value < 0.01:
        confidence = "high"
    elif dsr > 0.5 and p_value < 0.05:
        confidence = "moderate"
    elif dsr > 0 and p_value < 0.1:
        confidence = "low"
    else:
        confidence = "none"
    
    return {
        "is_significant": is_significant,
        "significance_level": significance_level,
        "confidence": confidence,
        "p_value": round(p_value, 6),
    }


def format_dsr_report(
    sharpe: float,
    n_trials: int,
    n_periods: int,
    dsr: float,
    significance: dict,
) -> str:
    """格式化 Deflated Sharpe 报告"""
    lines = [
        "=" * 60,
        "Deflated Sharpe 报告",
        "=" * 60,
        f"  原始 Sharpe:       {sharpe:.4f}",
        f"  试验次数:          {n_trials}",
        f"  样本期数:          {n_periods} (月)",
        f"  Deflated Sharpe:   {dsr:.4f}",
        f"  折扣幅度:          {(1 - dsr/max(sharpe, 0.001)) * 100:.1f}%",
        f"  显著性:            {significance['confidence']}",
        f"  p-value:           {significance['p_value']:.6f}",
        "=" * 60,
    ]
    return "\n".join(lines)
