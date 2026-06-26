"""Walk-forward validation for backtest_mvp

实现时序交叉验证框架：
- Train 期 (2018-2024.01): 用于参数优化/因子挖掘
- Test 期 (2024.01-2026.06): 用于验证，不可用于调参
- 可选: 验证期 (2022-2024.01) 用于超参数选择

设计约束：
- 仅使用日频数据，无日内数据
- 微盘股池（≤1000只），计算量可控
- 支持网格搜索 + 早停（计算资源有限）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
import copy

from tools.backtest_mvp.engine import CrossSectionalEngine, BacktestResult
from tools.backtest_mvp.research_loop.validators import validate_metrics, METRIC_THRESHOLDS


# ──────────────────────────────────────────────────────────────────────────────
# 数据分割工具
# ──────────────────────────────────────────────────────────────────────────────

def split_by_date(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    train_end: str = "2024-01-01",
    val_end: Optional[str] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """按时间切分 factor_panel 和 return_panel

    Returns
    -------
    dict with keys: train, val, test
    Each contains {"factor_panel": ..., "return_panel": ...}
    """
    train_end_ts = pd.Timestamp(train_end)
    train_end_ts = pd.Timestamp(train_end)
    dates = factor_panel.index.get_level_values("date").unique().sort_values()
    
    if val_end:
        val_end_ts = pd.Timestamp(val_end)
        train_dates = dates[dates < train_end_ts]
        val_dates = dates[(dates >= train_end_ts) & (dates < val_end_ts)]
        test_dates = dates[dates >= val_end_ts]
    else:
        train_dates = dates[dates < train_end_ts]
        val_dates = pd.DatetimeIndex([])
        test_dates = dates[dates >= train_end_ts]
    
    def _slice(panel, date_set):
        mask = panel.index.get_level_values("date").isin(date_set)
        return panel[mask]
    
    result = {}
    for split_name, date_set in [("train", train_dates), ("val", val_dates), ("test", test_dates)]:
        if len(date_set) > 0:
            result[split_name] = {
                "factor_panel": _slice(factor_panel, date_set),
                "return_panel": _slice(return_panel, date_set),
            }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 参数网格搜索
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GridSearchResult:
    """网格搜索结果"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame  # 所有参数组合的得分
    metric_name: str  # 优化目标指标
    

def _create_param_grid(strategy_def: dict) -> List[Dict[str, Any]]:
    """从策略定义中提取参数网格
    
    支持的参数:
    - n_stocks: 持仓数量 (e.g., [20, 30, 40])
    - stop_loss: 止损阈值 (e.g., [-0.30, -0.35, -0.40, -0.50])
    - trailing_stop: 移动止损 (e.g., [0.25, 0.30, 0.35])
    - pb_threshold: PB阈值 (e.g., [1.0, 1.5, 2.0, 3.0])
    - mcap_max: 最大市值 (e.g., [10, 15, 20, 30])
    
    参数网格通过 strategy_def 中的 "_param_grid" 键定义
    """
    param_grid = strategy_def.get("_param_grid", {})
    if not param_grid:
        # 默认参数网格
        return [{}]
    
    # 生成笛卡尔积
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    
    import itertools
    combos = list(itertools.product(*values))
    
    results = []
    for combo in combos:
        params = {k: v for k, v in zip(keys, combo)}
        results.append(params)
    return results


def _apply_params(strategy_def: dict, params: Dict[str, Any]) -> dict:
    """将参数应用到策略定义中，返回新的策略定义"""
    new_def = copy.deepcopy(strategy_def)
    for key, value in params.items():
        if key in new_def:
            new_def[key] = value
        elif key == "pb_threshold":
            # 修改 universe_filter 中的 PB 阈值
            # 这需要自定义 filter 支持参数化
            pass
        elif key == "mcap_max":
            # 修改 filter_micro_cap 的最大市值参数
            # 需要在 strategy_def 中重新构建 filter
            pass
    return new_def


# ──────────────────────────────────────────────────────────────────────────────
# Walk-Forward 核心引擎
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    """Walk-forward 验证结果"""
    strategy_name: str
    train_result: BacktestResult
    test_result: BacktestResult
    best_params: Dict[str, Any]
    optimization_metric: str
    overfit_ratio: float  # test_sharpe / train_sharpe，接近1表示不过拟合
    robustness_score: float  # 综合稳健性评分


def run_grid_search(
    strategy_def: dict,
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    metric: str = "sharpe",  # 优化目标: sharpe, annual_return, calmar
    param_grid: Optional[Dict[str, List]] = None,
    early_stop: bool = True,
    max_evals: int = 20,  # 最大评估次数（防止计算爆炸）
) -> GridSearchResult:
    """在训练数据上运行参数网格搜索
    
    Parameters
    ----------
    strategy_def : dict
        策略定义，包含 ranking_factor, universe_filter 等
    factor_panel : DataFrame
        训练期的因子面板
    return_panel : DataFrame
        训练期的收益率面板
    metric : str
        优化目标指标
    param_grid : dict
        自定义参数网格，如果为None则使用 strategy_def._param_grid
    early_stop : bool
        是否启用早停（如果连续5个参数组合得分 < 0.5 * 当前最佳，停止）
    max_evals : int
        最大评估次数
    
    Returns
    -------
    GridSearchResult
    """
    if param_grid is None:
        param_grid = strategy_def.get("_param_grid", {})
    
    if not param_grid:
        # 无参数可调，直接运行一次
        engine = CrossSectionalEngine(
            factor_panel=factor_panel,
            return_panel=return_panel,
            n_stocks=strategy_def.get("n_stocks", 30),
            rebalance_freq='M',
            commission=0.00125,
            slippage=0.002,
        )
        result = engine.run(
            universe_filter=strategy_def["universe_filter"],
            ranking_factor=strategy_def.get("ranking_factor", "mcap"),
            ascending=strategy_def.get("ascending", True),
            composite_factors=strategy_def.get("composite_factors"),
            stop_loss=strategy_def.get("stop_loss"),
            trailing_stop=strategy_def.get("trailing_stop"),
            ranking_fn=strategy_def.get("ranking_fn"),
            factor_weights=strategy_def.get("factor_weights"),
        )
        score = getattr(result, metric, 0)
        return GridSearchResult(
            best_params={},
            best_score=score,
            all_results=pd.DataFrame({"params": [{}], metric: [score]}),
            metric_name=metric,
        )
    
    # 生成参数组合
    param_combos = []
    import itertools
    keys = sorted(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in itertools.product(*values):
        param_combos.append({k: v for k, v in zip(keys, combo)})
    
    # 限制评估数量
    if len(param_combos) > max_evals:
        import random
        random.seed(42)
        param_combos = random.sample(param_combos, max_evals)
    
    print(f"  网格搜索: {len(param_combos)} 组参数 (max_evals={max_evals})")
    
    results = []
    best_score = -np.inf
    best_params = {}
    stagnant_count = 0
    
    for i, params in enumerate(param_combos):
        # 构建新策略定义
        new_def = copy.deepcopy(strategy_def)
        for key, value in params.items():
            if key in new_def:
                new_def[key] = value
        
        try:
            engine = CrossSectionalEngine(
                factor_panel=factor_panel,
                return_panel=return_panel,
                n_stocks=new_def.get("n_stocks", 30),
                rebalance_freq='M',
                commission=0.00125,
                slippage=0.002,
            )
            result = engine.run(
                universe_filter=new_def["universe_filter"],
                ranking_factor=new_def.get("ranking_factor", "mcap"),
                ascending=new_def.get("ascending", True),
                composite_factors=new_def.get("composite_factors"),
                stop_loss=new_def.get("stop_loss"),
                trailing_stop=new_def.get("trailing_stop"),
                ranking_fn=new_def.get("ranking_fn"),
                factor_weights=new_def.get("factor_weights"),
            )
            score = getattr(result, metric, 0)
            if np.isnan(score) or np.isinf(score):
                score = 0.0
        except Exception as e:
            print(f"    参数 {params} 运行失败: {e}")
            score = 0.0
            result = None
        
        results.append({
            "params": str(params),
            **params,
            metric: score,
            "annual_return": getattr(result, "annual_return", 0) if result else 0,
            "sharpe": getattr(result, "sharpe_ratio", 0) if result else 0,
            "max_drawdown": getattr(result, "max_drawdown", 0) if result else 0,
        })
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            stagnant_count = 0
        else:
            stagnant_count += 1
        
        # 早停
        if early_stop and stagnant_count >= 5 and best_score > 0:
            print(f"    早停: 连续 {stagnant_count} 次未改善")
            break
    
    df = pd.DataFrame(results)
    return GridSearchResult(
        best_params=best_params,
        best_score=best_score,
        all_results=df,
        metric_name=metric,
    )


def run_walk_forward(
    strategy_def: dict,
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    train_end: str = "2024-01-01",
    val_end: Optional[str] = None,
    optimization_metric: str = "sharpe",
    param_grid: Optional[Dict[str, List]] = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """运行完整的 walk-forward 验证
    
    流程:
    1. 按 train_end 切分数据
    2. 在训练数据上运行网格搜索，找到最优参数
    3. 在测试数据上运行最优参数，得到 out-of-sample 结果
    4. 对比 train/test 的稳健性
    
    Parameters
    ----------
    strategy_def : dict
        策略定义（可包含 _param_grid 用于参数搜索）
    factor_panel : DataFrame
        完整因子面板（全时间窗口）
    return_panel : DataFrame
        完整收益率面板（全时间窗口）
    train_end : str
        训练期截止日（默认2024-01-01）
    val_end : str, optional
        验证期截止日（用于三阶段划分：train/val/test）
    optimization_metric : str
        优化目标: sharpe, annual_return, calmar
    param_grid : dict, optional
        自定义参数网格
    verbose : bool
        是否打印详细日志
    
    Returns
    -------
    WalkForwardResult
    """
    # 1. 切分数据
    splits = split_by_date(factor_panel, return_panel, train_end, val_end)
    
    train_data = splits.get("train")
    test_data = splits.get("test")
    
    if not train_data or not test_data:
        raise ValueError("数据切分失败: train 或 test 为空")
    
    train_fp = train_data["factor_panel"]
    train_rp = train_data["return_panel"]
    test_fp = test_data["factor_panel"]
    test_rp = test_data["return_panel"]
    
    if verbose:
        train_dates = train_fp.index.get_level_values("date").unique()
        test_dates = test_fp.index.get_level_values("date").unique()
        print(f"Walk-forward: Train {train_dates[0].date()} ~ {train_dates[-1].date()} "
              f"({len(train_dates)} days) | Test {test_dates[0].date()} ~ {test_dates[-1].date()} "
              f"({len(test_dates)} days)")
    
    # 2. 训练期参数优化
    if verbose:
        print(f"\n  [Train] 参数优化 (目标: {optimization_metric})...")
    
    grid_result = run_grid_search(
        strategy_def,
        train_fp,
        train_rp,
        metric=optimization_metric,
        param_grid=param_grid,
    )
    
    if verbose:
        print(f"  [Train] 最优参数: {grid_result.best_params}")
        print(f"  [Train] 最优 {optimization_metric}: {grid_result.best_score:.2f}")
    
    # 3. 测试期验证（使用最优参数）
    if verbose:
        print(f"\n  [Test] 使用最优参数验证...")
    
    # 应用最优参数
    best_def = copy.deepcopy(strategy_def)
    for key, value in grid_result.best_params.items():
        if key in best_def:
            best_def[key] = value
    
    engine_test = CrossSectionalEngine(
        factor_panel=test_fp,
        return_panel=test_rp,
        n_stocks=best_def.get("n_stocks", 30),
        rebalance_freq='M',
        commission=0.00125,
        slippage=0.002,
    )
    test_result = engine_test.run(
        universe_filter=best_def["universe_filter"],
        ranking_factor=best_def.get("ranking_factor", "mcap"),
        ascending=best_def.get("ascending", True),
        composite_factors=best_def.get("composite_factors"),
        stop_loss=best_def.get("stop_loss"),
        trailing_stop=best_def.get("trailing_stop"),
        ranking_fn=best_def.get("ranking_fn"),
        factor_weights=best_def.get("factor_weights"),
    )
    
    # 4. 训练期基准（使用最优参数重新跑，用于对比）
    engine_train = CrossSectionalEngine(
        factor_panel=train_fp,
        return_panel=train_rp,
        n_stocks=best_def.get("n_stocks", 30),
        rebalance_freq='M',
        commission=0.00125,
        slippage=0.002,
    )
    train_result = engine_train.run(
        universe_filter=best_def["universe_filter"],
        ranking_factor=best_def.get("ranking_factor", "mcap"),
        ascending=best_def.get("ascending", True),
        composite_factors=best_def.get("composite_factors"),
        stop_loss=best_def.get("stop_loss"),
        trailing_stop=best_def.get("trailing_stop"),
        ranking_fn=best_def.get("ranking_fn"),
        factor_weights=best_def.get("factor_weights"),
    )
    
    # 5. 计算稳健性指标
    train_sharpe = train_result.sharpe_ratio if train_result.sharpe_ratio else 0.001
    test_sharpe = test_result.sharpe_ratio if test_result.sharpe_ratio else 0
    overfit_ratio = test_sharpe / train_sharpe if train_sharpe > 0 else 0
    
    # 稳健性评分：综合 test_sharpe + overfit_ratio + test_win_rate
    test_wr = test_result.win_rate / 100 if test_result.win_rate else 0
    robustness = (
        0.4 * max(0, test_sharpe) +
        0.3 * max(0, min(overfit_ratio, 1.5)) +  # 过拟合比 > 1.5 不额外奖励
        0.3 * test_wr
    )
    
    if verbose:
        print(f"\n  {'='*60}")
        print(f"  Walk-Forward 验证结果: {strategy_def['name']}")
        print(f"  {'='*60}")
        print(f"  Train 年化: {train_result.annual_return:>6.1f}% | 夏普: {train_result.sharpe_ratio:>5.2f} | 回撤: {train_result.max_drawdown:>6.1f}%")
        print(f"  Test  年化: {test_result.annual_return:>6.1f}% | 夏普: {test_result.sharpe_ratio:>5.2f} | 回撤: {test_result.max_drawdown:>6.1f}%")
        print(f"  过拟合比: {overfit_ratio:.2f} (1.0=完美, <0.5=严重过拟合)")
        print(f"  稳健性评分: {robustness:.2f} / 1.0")
        if overfit_ratio < 0.5:
            print(f"  ⚠️ 警告: 严重过拟合! Test 夏普远低于 Train")
        elif overfit_ratio > 1.2:
            print(f"  ℹ️  Test 表现优于 Train，可能受益于市场风格")
        print(f"  {'='*60}")
    
    return WalkForwardResult(
        strategy_name=strategy_def["name"],
        train_result=train_result,
        test_result=test_result,
        best_params=grid_result.best_params,
        optimization_metric=optimization_metric,
        overfit_ratio=overfit_ratio,
        robustness_score=robustness,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 批量 Walk-Forward 验证
# ──────────────────────────────────────────────────────────────────────────────

def run_walk_forward_all(
    strategies: List[dict],
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    train_end: str = "2024-01-01",
    optimization_metric: str = "sharpe",
    verbose: bool = True,
) -> pd.DataFrame:
    """对多个策略批量运行 walk-forward 验证
    
    Returns
    -------
    DataFrame with columns:
        strategy, train_annual, train_sharpe, test_annual, test_sharpe,
        overfit_ratio, robustness_score, best_params
    """
    rows = []
    for s in strategies:
        try:
            wf = run_walk_forward(
                s, factor_panel, return_panel,
                train_end=train_end,
                optimization_metric=optimization_metric,
                verbose=verbose,
            )
            rows.append({
                "strategy": wf.strategy_name,
                "train_annual": wf.train_result.annual_return,
                "train_sharpe": wf.train_result.sharpe_ratio,
                "train_drawdown": wf.train_result.max_drawdown,
                "test_annual": wf.test_result.annual_return,
                "test_sharpe": wf.test_result.sharpe_ratio,
                "test_drawdown": wf.test_result.max_drawdown,
                "overfit_ratio": wf.overfit_ratio,
                "robustness_score": wf.robustness_score,
                "best_params": str(wf.best_params),
            })
        except Exception as e:
            print(f"  ⚠️ {s['name']} 失败: {e}")
            rows.append({
                "strategy": s["name"],
                "train_annual": 0,
                "train_sharpe": 0,
                "test_annual": 0,
                "test_sharpe": 0,
                "overfit_ratio": 0,
                "robustness_score": 0,
                "best_params": str({}),
                "error": str(e),
            })
    
    df = pd.DataFrame(rows)
    if verbose and len(df) > 0:
        print(f"\n{'='*80}")
        print(f"  Walk-Forward 批量验证汇总")
        print(f"{'='*80}")
        print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    # 演示
    print("Walk-forward validation module loaded.")
    print("Usage:")
    print("  from walk_forward import run_walk_forward, run_walk_forward_all")
    print("  wf = run_walk_forward(strategy_def, factor_panel, return_panel)")
