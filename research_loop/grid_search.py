"""
网格搜索框架 — Grid Search for Hybrid Strategy Parameters
============================================================
对混合策略的关键参数进行系统性搜索，找到最优配置。

搜索维度:
1. 子策略权重组合 (微盘/多因子/低波)
2. 市场状态参数 (MA窗口/波动率阈值)
3. 现金缓冲配置
4. 风险平价/动量加权比例
5. 权重调整速度

输出:
- 参数组合排名表
- 最优参数配置
- 参数敏感性分析
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from tools.backtest_mvp.research_loop.hybrid_strategy import (
    HybridConfig, SubStrategy, MarketRegime,
    detect_market_regime, calculate_risk_parity_weights,
    calculate_momentum_weights, adjust_weights_for_regime,
    constrain_weights, HybridStrategy
)


@dataclass
class GridSearchResult:
    """网格搜索结果"""
    params: Dict[str, Any]       # 参数组合
    annual_return: float         # 年化收益
    sharpe_ratio: float          # 夏普比率
    max_drawdown: float          # 最大回撤
    calmar_ratio: float          # 卡玛比率
    win_rate: float              # 月度胜率
    avg_turnover: float          # 平均换手率
    terminal_value: float        # 终值
    regime_transitions: int      # 状态切换次数
    # 分状态表现
    bull_return: float = 0.0
    bear_return: float = 0.0
    # 综合评分
    composite_score: float = 0.0


class GridSearchSpace:
    """
    定义网格搜索空间
    
    每个参数可以是一个离散值列表或一个范围
    """
    
    def __init__(self):
        self.space = {}
    
    def add_param(self, name: str, values: List[Any]):
        """添加离散参数"""
        self.space[name] = values
    
    def add_range(self, name: str, start: float, end: float, step: float):
        """添加连续参数范围 (离散化)"""
        n_steps = int((end - start) / step) + 1
        self.space[name] = [start + i * step for i in range(n_steps)]
    
    def generate_combinations(self, max_evals: int = None) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        keys = list(self.space.keys())
        values = [self.space[k] for k in keys]
        
        combos = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            combos.append(params)
        
        if max_evals and len(combos) > max_evals:
            # 随机采样
            np.random.shuffle(combos)
            combos = combos[:max_evals]
        
        return combos
    
    def get_size(self) -> int:
        """获取总组合数"""
        total = 1
        for values in self.space.values():
            total *= len(values)
        return total


def create_default_grid_space() -> GridSearchSpace:
    """
    创建默认搜索空间
    
    基于对混合策略影响最大的参数
    """
    space = GridSearchSpace()
    
    # 1. 子策略权重配置 (3种预设)
    space.add_param("weight_preset", [
        "aggressive",   # 激进: 微盘50% + 多因子30% + 低波20%
        "balanced",     # 平衡: 微盘40% + 多因子35% + 低波25%
        "conservative", # 保守: 微盘30% + 多因子40% + 低波30%
    ])
    
    # 2. 市场状态参数
    space.add_param("regime_vol_threshold", [0.20, 0.25, 0.30, 0.35])  # 高波动阈值
    space.add_param("regime_trend_threshold", [0.03, 0.05, 0.07])    # 趋势阈值
    
    # 3. 现金缓冲配置
    space.add_param("cash_buffer_bull", [0.05, 0.10, 0.15])         # 牛市现金
    space.add_param("cash_buffer_bear", [0.30, 0.40, 0.50])         # 熊市现金
    space.add_param("cash_buffer_crash", [0.50, 0.60, 0.70])       # 暴跌现金
    
    # 4. 加权方法混合比例
    space.add_param("rp_mom_blend", [0.3, 0.5, 0.7])                # RP权重(0.3=30%RP+70%Momentum)
    
    # 5. 权重调整速度
    space.add_param("weight_adjustment_speed", [0.2, 0.3, 0.5])      # 调整速度
    
    return space


def params_to_config(params: Dict[str, Any]) -> HybridConfig:
    """
    将参数字典转换为 HybridConfig
    """
    # 权重预设
    presets = {
        "aggressive": (0.50, 0.30, 0.20),
        "balanced": (0.40, 0.35, 0.25),
        "conservative": (0.30, 0.40, 0.30),
    }
    
    w1, w2, w3 = presets.get(params["weight_preset"], (0.40, 0.35, 0.25))
    
    # 创建子策略 (使用占位符，实际运行时会替换)
    subs = [
        SubStrategy(name="micro_cap", strategy_def={}, weight=w1, max_weight=0.70, min_weight=0.15),
        SubStrategy(name="composite", strategy_def={}, weight=w2, max_weight=0.60, min_weight=0.15),
        SubStrategy(name="lowvol", strategy_def={}, weight=w3, max_weight=0.50, min_weight=0.10),
    ]
    
    return HybridConfig(
        name=f"grid_{params['weight_preset']}",
        sub_strategies=subs,
        regime_vol_threshold=params["regime_vol_threshold"],
        regime_trend_threshold=params["regime_trend_threshold"],
        cash_buffer_bull=params["cash_buffer_bull"],
        cash_buffer_bear=params["cash_buffer_bear"],
        cash_buffer_crash=params["cash_buffer_crash"],
        weight_adjustment_speed=params["weight_adjustment_speed"],
        risk_parity_target_vol=0.20,
    )


def evaluate_single_params(
    params: Dict[str, Any],
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    market_returns: pd.Series,
    sub_strategy_results: Dict[str, Any],
) -> GridSearchResult:
    """
    评估单个参数组合
    
    使用子策略的预计算回测结果，模拟混合策略表现
    """
    config = params_to_config(params)
    hybrid = HybridStrategy(config)
    
    # 获取调仓日
    dates = sorted(set(factor_panel.index.get_level_values(0)))
    if len(dates) < 60:
        return GridSearchResult(
            params=params, annual_return=0, sharpe_ratio=0,
            max_drawdown=0, calmar_ratio=0, win_rate=0,
            avg_turnover=0, terminal_value=1.0, regime_transitions=0
        )
    
    # 使用月度调仓日
    monthly_dates = pd.DatetimeIndex(dates).to_period('M').to_timestamp(how='end')
    monthly_dates = sorted(set(monthly_dates))
    
    # 初始化
    hybrid.initialize(market_returns, monthly_dates[0])
    
    # 模拟混合策略收益
    equity = 1.0
    equity_curve = [equity]
    monthly_rets = []
    
    # 子策略的权益曲线 (从预计算结果获取)
    sub_equities = {}
    for name in sub_strategy_results:
        if 'equity_curve' in sub_strategy_results[name]:
            sub_equities[name] = sub_strategy_results[name]['equity_curve']
    
    regime_history = []
    
    for i in range(1, len(monthly_dates)):
        date = monthly_dates[i]
        prev_date = monthly_dates[i-1]
        
        # 更新权重
        weights = hybrid.update_weights(
            market_returns, {}, date  # 简化: 使用固定权重，不依赖历史收益
        )
        
        regime_history.append(hybrid.current_regime.value)
        
        # 获取现金权重
        cash_weight = hybrid.get_cash_weight()
        strategy_weight = 1.0 - cash_weight
        
        # 计算当月混合收益
        if sub_equities:
            month_ret = 0.0
            for name, eq in sub_equities.items():
                if name in weights and prev_date in eq.index and date in eq.index:
                    w = weights.get(name, 0.0) * strategy_weight
                    sub_ret = eq.loc[date] / eq.loc[prev_date] - 1
                    month_ret += w * sub_ret
            
            # 现金收益 (假设年化2%)
            cash_ret = 0.02 / 12 * cash_weight
            month_ret += cash_ret
            
            equity *= (1 + month_ret)
            equity_curve.append(equity)
            monthly_rets.append(month_ret)
    
    # 计算指标
    if len(monthly_rets) < 3:
        return GridSearchResult(
            params=params, annual_return=0, sharpe_ratio=0,
            max_drawdown=0, calmar_ratio=0, win_rate=0,
            avg_turnover=0, terminal_value=equity, regime_transitions=0
        )
    
    monthly_rets = pd.Series(monthly_rets)
    n_years = len(monthly_rets) / 12
    
    total_ret = equity - 1.0
    annual_ret = (equity) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = monthly_rets.std() * np.sqrt(12)
    
    sharpe = (annual_ret - 0.03) / annual_vol if annual_vol > 0 else 0
    
    # 最大回撤
    eq_series = pd.Series(equity_curve)
    peak = eq_series.expanding().max()
    dd = (eq_series - peak) / peak
    max_dd = dd.min()
    
    calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0
    win_rate = (monthly_rets > 0).sum() / len(monthly_rets)
    
    # 状态切换次数
    regime_transitions = sum(
        1 for i in range(1, len(regime_history))
        if regime_history[i] != regime_history[i-1]
    )
    
    # 综合评分 (夏普 + 卡玛 - 回撤惩罚)
    drawdown_penalty = max(0, abs(max_dd) - 0.20) * 2  # 回撤>20%开始惩罚
    composite_score = sharpe * 0.4 + calmar * 0.3 - drawdown_penalty
    
    return GridSearchResult(
        params=params,
        annual_return=round(annual_ret * 100, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown=round(max_dd * 100, 2),
        calmar_ratio=round(calmar, 2),
        win_rate=round(win_rate * 100, 1),
        avg_turnover=0.0,  # 简化
        terminal_value=round(equity, 4),
        regime_transitions=regime_transitions,
        composite_score=round(composite_score, 3),
    )


def run_grid_search(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    market_returns: pd.Series,
    sub_strategy_results: Dict[str, Any],
    space: GridSearchSpace = None,
    max_evals: int = 100,
    metric: str = "composite_score",
    n_jobs: int = 1,
) -> Tuple[List[GridSearchResult], GridSearchResult]:
    """
    运行网格搜索
    
    Args:
        factor_panel: 因子面板
        return_panel: 收益面板
        market_returns: 市场指数日收益率
        sub_strategy_results: 子策略预计算回测结果
        space: 搜索空间 (None=使用默认)
        max_evals: 最大评估数 (None=全部)
        metric: 优化目标 ("sharpe", "calmar", "composite_score")
        n_jobs: 并行数 (1=串行)
    
    Returns:
        (所有结果列表, 最优结果)
    """
    if space is None:
        space = create_default_grid_space()
    
    combos = space.generate_combinations(max_evals=max_evals)
    total = len(combos)
    
    print(f"网格搜索: {total} 组参数组合 (搜索空间: {space.get_size()})")
    print(f"优化目标: {metric}")
    print("-" * 60)
    
    results = []
    
    if n_jobs > 1:
        # 并行评估
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    evaluate_single_params, combo, factor_panel, return_panel,
                    market_returns, sub_strategy_results
                ): combo for combo in combos
            }
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    results.append(result)
                    if (i + 1) % 10 == 0:
                        print(f"  进度: {i+1}/{total}")
                except Exception as e:
                    print(f"  ⚠️ 参数组合失败: {e}")
    else:
        # 串行评估
        for i, combo in enumerate(combos):
            result = evaluate_single_params(
                combo, factor_panel, return_panel,
                market_returns, sub_strategy_results
            )
            results.append(result)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{total}] SR={result.sharpe_ratio:.2f} "
                      f"DD={result.max_drawdown:.1f}% "
                      f"Score={result.composite_score:.3f}")
    
    # 按目标指标排序
    reverse = True  # 越高越好
    if metric == "max_drawdown":
        reverse = False  # 回撤越小越好
    
    results_sorted = sorted(results, key=lambda x: getattr(x, metric), reverse=reverse)
    
    best = results_sorted[0]
    
    print(f"\n{'='*60}")
    print(f"  最优参数 (按 {metric})")
    print(f"{'='*60}")
    print(f"  参数: {best.params}")
    print(f"  年化收益: {best.annual_return:.2f}%")
    print(f"  夏普比率: {best.sharpe_ratio:.2f}")
    print(f"  最大回撤: {best.max_drawdown:.2f}%")
    print(f"  卡玛比率: {best.calmar_ratio:.2f}")
    print(f"  综合评分: {best.composite_score:.3f}")
    print(f"{'='*60}")
    
    return results_sorted, best


def print_grid_search_results(
    results: List[GridSearchResult],
    top_n: int = 10,
):
    """打印网格搜索结果表格"""
    
    print(f"\n{'='*100}")
    print(f"  网格搜索结果 TOP {top_n}")
    print(f"{'='*100}")
    
    headers = ["排名", "权重配置", "波动阈值", "趋势阈值", "牛市现金", "熊市现金", "暴跌现金", "调整速度", "年化", "夏普", "回撤", "卡玛", "综合评分"]
    print("  " + " | ".join(f"{h:>8}" for h in headers))
    print("  " + "-" * 120)
    
    for i, r in enumerate(results[:top_n]):
        p = r.params
        print(f"  {i+1:>4} | {p.get('weight_preset', 'N/A'):>8} | "
              f"{p.get('regime_vol_threshold', 0):>8.2f} | "
              f"{p.get('regime_trend_threshold', 0):>8.2f} | "
              f"{p.get('cash_buffer_bull', 0):>8.2f} | "
              f"{p.get('cash_buffer_bear', 0):>8.2f} | "
              f"{p.get('cash_buffer_crash', 0):>8.2f} | "
              f"{p.get('weight_adjustment_speed', 0):>8.2f} | "
              f"{r.annual_return:>7.1f}% | "
              f"{r.sharpe_ratio:>6.2f} | "
              f"{r.max_drawdown:>6.1f}% | "
              f"{r.calmar_ratio:>6.2f} | "
              f"{r.composite_score:>8.3f}")
    
    print(f"{'='*100}")
    
    # 参数敏感性分析
    print(f"\n  参数敏感性分析")
    print(f"  {'-'*60}")
    
    for param_name in results[0].params.keys():
        param_values = {}
        for r in results[:50]:  # 只看前50
            val = r.params[param_name]
            if val not in param_values:
                param_values[val] = []
            param_values[val].append(r.composite_score)
        
        # 计算每个参数值的平均评分
        avg_scores = {v: np.mean(scores) for v, scores in param_values.items()}
        best_val = max(avg_scores, key=avg_scores.get)
        
        print(f"  {param_name:>25}: 最优={best_val}, 平均评分={avg_scores[best_val]:.3f}")
    
    print(f"{'='*100}")


def analyze_parameter_importance(
    results: List[GridSearchResult],
) -> Dict[str, float]:
    """
    分析参数重要性 (基于结果方差)
    
    原理: 如果某个参数的不同值导致结果差异很大，则该参数重要
    """
    importance = {}
    
    for param_name in results[0].params.keys():
        # 按参数值分组
        groups = {}
        for r in results:
            val = r.params[param_name]
            if val not in groups:
                groups[val] = []
            groups[val].append(r.composite_score)
        
        # 计算组间方差 / 组内方差
        group_means = [np.mean(scores) for scores in groups.values()]
        group_vars = [np.var(scores) for scores in groups.values()]
        
        between_var = np.var(group_means) if len(group_means) > 1 else 0
        within_var = np.mean(group_vars) if len(group_vars) > 0 else 1
        
        # F-statistic 近似
        f_stat = between_var / (within_var + 1e-6)
        importance[param_name] = f_stat
    
    # 归一化
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}
    
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
