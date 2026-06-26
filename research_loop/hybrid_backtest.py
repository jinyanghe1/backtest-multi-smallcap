"""
混合策略回测 — Hybrid Backtest
==============================
在现有回测框架上实现多策略组合+动态权重+市场状态过滤

运行流程:
1. 跑3个关键子策略 (S3微盘 + SB多因子 + SA低波)
2. 获取市场指数收益 (用于状态检测)
3. 按月动态调整权重
4. 计算混合策略收益曲线

输出:
- 混合策略回测结果
- 子策略对比
- 权重变化历史
- 市场状态历史
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from tools.backtest_mvp.engine import CrossSectionalEngine, BacktestResult
from tools.backtest_mvp.strategies import ALL_STRATEGIES
from tools.backtest_mvp.strategies_v2 import NEW_STRATEGIES
from tools.backtest_mvp.benchmark import load_benchmarks


def run_sub_strategies(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
) -> Dict[str, BacktestResult]:
    """
    跑3个关键子策略
    
    Returns:
        {策略名: BacktestResult}
    """
    # 选择3个策略
    s3 = ALL_STRATEGIES[2]   # 策略3: 极小小市值轮动
    sb = NEW_STRATEGIES[1]   # 策略B: 多因子综合
    sa = NEW_STRATEGIES[0]   # 策略A: 低波小市值
    
    strategies = [("micro_cap", s3), ("composite", sb), ("lowvol", sa)]
    results = {}
    
    print("  跑子策略回测...")
    for name, s_def in strategies:
        engine = CrossSectionalEngine(
            factor_panel=factor_panel,
            return_panel=return_panel,
            initial_capital=1.0,
            n_stocks=s_def.get("n_stocks", 30),
            rebalance_freq='M',
            commission=0.00125,
            slippage=0.002,
        )
        
        result = engine.run(
            universe_filter=s_def["universe_filter"],
            ranking_factor=s_def.get("ranking_factor", "mcap"),
            ascending=s_def.get("ascending", True),
            composite_factors=s_def.get("composite_factors"),
            stop_loss=s_def.get("stop_loss"),
            trailing_stop=s_def.get("trailing_stop"),
            ranking_fn=s_def.get("ranking_fn"),
            factor_weights=s_def.get("factor_weights"),
        )
        results[name] = result
        print(f"    {name:>12}: 年化{result.annual_return:>6.1f}% 夏普{result.sharpe_ratio:>5.2f} 回撤{result.max_drawdown:>6.1f}%")
    
    return results


def detect_market_state(
    market_returns: pd.Series,
    current_date: pd.Timestamp,
    short_window: int = 20,
    long_window: int = 60,
    vol_threshold: float = 0.25,
    trend_threshold: float = 0.03,
) -> str:
    """
    检测市场状态
    
    Returns: "bull" | "bear" | "range" | "crash"
    """
    available = market_returns[market_returns.index <= current_date]
    if len(available) < long_window:
        return "range"
    
    short_ret = available.iloc[-short_window:]
    long_ret = available.iloc[-long_window:]
    
    short_ma = (1 + short_ret).prod() - 1
    long_ma = (1 + long_ret).prod() - 1
    trend = short_ma - long_ma
    
    vol = short_ret.std() * np.sqrt(252)
    recent_5d = available.iloc[-5:]
    recent_ret = (1 + recent_5d).prod() - 1
    
    if vol > vol_threshold * 1.5 and recent_ret < -0.08:
        return "crash"
    elif vol > vol_threshold and trend < -trend_threshold:
        return "bear"
    elif trend > trend_threshold and vol < vol_threshold:
        return "bull"
    elif trend < -trend_threshold and vol < vol_threshold:
        return "bear"
    else:
        return "range"


def calculate_weights(
    state: str,
    sub_results: Dict[str, BacktestResult],
    current_date: pd.Timestamp,
    base_weights: Dict[str, float],
    weight_adjustment_speed: float = 0.3,
    prev_weights: Dict[str, float] = None,
) -> Tuple[Dict[str, float], float]:
    """
    根据市场状态计算子策略权重
    
    Returns:
        (策略权重, 现金权重)
    """
    # 状态偏好系数
    state_prefs = {
        "bull":   {"micro_cap": 1.5, "composite": 1.0, "lowvol": 0.6},
        "bear":   {"micro_cap": 0.4, "composite": 0.8, "lowvol": 1.5},
        "range":  {"micro_cap": 1.0, "composite": 1.0, "lowvol": 1.0},
        "crash":  {"micro_cap": 0.0, "composite": 0.5, "lowvol": 1.0},
    }
    
    prefs = state_prefs.get(state, state_prefs["range"])
    
    # 现金缓冲
    cash_buffer = {
        "bull": 0.10, "bear": 0.35, "range": 0.20, "crash": 0.50,
    }
    cash_weight = cash_buffer.get(state, 0.20)
    
    # 计算调整后的策略权重
    strategy_weight = 1.0 - cash_weight
    
    adjusted = {}
    for name, base_w in base_weights.items():
        pref = prefs.get(name, 1.0)
        adjusted[name] = base_w * pref
    
    # 归一化
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total * strategy_weight for k, v in adjusted.items()}
    
    # 平滑过渡
    if prev_weights is not None:
        smoothed = {}
        for name in adjusted:
            old_w = prev_weights.get(name, 0.0)
            new_w = adjusted[name]
            smoothed[name] = weight_adjustment_speed * new_w + (1 - weight_adjustment_speed) * old_w
        
        # 重新归一化到 strategy_weight
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total * strategy_weight for k, v in smoothed.items()}
        adjusted = smoothed
    
    return adjusted, cash_weight


def run_hybrid_backtest(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    base_weights: Dict[str, float] = None,
    vol_threshold: float = 0.25,
    trend_threshold: float = 0.03,
    weight_adjustment_speed: float = 0.3,
    verbose: bool = True,
) -> Tuple[BacktestResult, Dict]:
    """
    运行混合策略回测
    
    Args:
        base_weights: 基础权重 {micro_cap: 0.4, composite: 0.35, lowvol: 0.25}
        vol_threshold: 高波动阈值
        trend_threshold: 趋势阈值
        weight_adjustment_speed: 权重调整速度
    
    Returns:
        (混合策略BacktestResult, 详细信息)
    """
    if base_weights is None:
        base_weights = {"micro_cap": 0.40, "composite": 0.35, "lowvol": 0.25}
    
    # 1. 跑子策略
    if verbose:
        print("=" * 80)
        print("  混合策略回测")
        print("=" * 80)
    
    sub_results = run_sub_strategies(factor_panel, return_panel)
    
    # 2. 获取市场收益 (用国证2000作为代理)
    try:
        benchmarks = load_benchmarks()
        if '国证2000' in benchmarks:
            market_prices = benchmarks['国证2000']
            market_returns = market_prices.pct_change().dropna()
        else:
            # 用因子面板中的股票平均收益作为市场代理
            market_returns = return_panel.groupby(level=0)['daily_return'].mean()
    except:
        market_returns = return_panel.groupby(level=0)['daily_return'].mean()
    
    # 3. 获取调仓日 (使用实际交易日，而非日历月末)
    dates = sorted(set(factor_panel.index.get_level_values(0)))
    # 找到每月最后一个实际交易日
    df_dates = pd.DataFrame({'date': dates})
    df_dates['month'] = df_dates['date'].dt.to_period('M')
    monthly_dates = df_dates.groupby('month')['date'].last().tolist()
    monthly_dates = sorted(monthly_dates)
    
    # 确保 monthly_dates 在数据范围内
    if len(monthly_dates) < 3:
        print("  ⚠️ 数据不足")
        return None, {}
    
    # 4. 模拟混合策略
    equity = 1.0
    equity_curve = [equity]
    monthly_rets = []
    weight_history = []
    state_history = []
    
    prev_weights = None
    
    for i in range(1, len(monthly_dates)):
        date = monthly_dates[i]
        prev_date = monthly_dates[i-1]
        
        # 检测市场状态
        state = detect_market_state(
            market_returns, date,
            short_window=20, long_window=60,
            vol_threshold=vol_threshold, trend_threshold=trend_threshold,
        )
        
        # 计算权重
        weights, cash_weight = calculate_weights(
            state, sub_results, date,
            base_weights, weight_adjustment_speed, prev_weights,
        )
        prev_weights = weights.copy()
        
        # 计算当月收益 (使用asof处理日期不匹配)
        month_ret = 0.0
        for name, result in sub_results.items():
            eq = result.equity_curve
            # 使用asof获取最近的前一个日期的权益值
            start_val = eq.asof(prev_date)
            end_val = eq.asof(date)
            if start_val is not None and end_val is not None and start_val > 0:
                sub_ret = end_val / start_val - 1
                w = weights.get(name, 0.0)
                month_ret += w * sub_ret
        
        # 现金收益
        cash_ret = 0.02 / 12 * cash_weight
        month_ret += cash_ret
        
        equity *= (1 + month_ret)
        equity_curve.append(equity)
        monthly_rets.append(month_ret)
        
        weight_history.append({
            'date': date, 'state': state,
            **weights, 'cash': cash_weight,
        })
        state_history.append(state)
    
    # 5. 计算混合策略指标
    monthly_rets = pd.Series(monthly_rets)
    n_months = len(monthly_rets)
    n_years = n_months / 12
    
    total_ret = equity - 1.0
    annual_ret = (equity) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = monthly_rets.std() * np.sqrt(12)
    
    sharpe = (annual_ret - 0.03) / annual_vol if annual_vol > 0 else 0
    
    eq_series = pd.Series(equity_curve)
    peak = eq_series.expanding().max()
    dd = (eq_series - peak) / peak
    max_dd = dd.min()
    
    calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0
    win_rate = (monthly_rets > 0).sum() / n_months if n_months > 0 else 0
    
    # 计算回撤恢复时间
    trough_idx = dd.idxmin()
    if trough_idx < len(eq_series) - 1:
        after = eq_series.iloc[trough_idx:]
        pre_trough_peak = peak.iloc[trough_idx]
        recovery_dates = after[after >= pre_trough_peak].index
        recovery_days = int(recovery_dates[0] - trough_idx) if len(recovery_dates) > 0 else 0
    else:
        recovery_days = 0
    
    # 计算IC (简化)
    ic_mean = np.mean([r.ic_mean for r in sub_results.values() if r.ic_mean != 0])
    
    # 构建结果
    hybrid_result = BacktestResult(
        equity_curve=pd.Series(equity_curve, index=[monthly_dates[0]] + monthly_dates[1:]),
        monthly_returns=monthly_rets,
        annual_return=round(annual_ret * 100, 2),
        annual_volatility=round(annual_vol * 100, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown=round(max_dd * 100, 2),
        calmar_ratio=round(calmar, 2),
        win_rate=round(win_rate * 100, 1),
        avg_turnover=0.0,  # 简化
        terminal_value=round(equity, 4),
        positions_log=pd.DataFrame(),
        monthly_turnover_log=[],
        ic_mean=round(ic_mean, 4),
        ic_ir=0.0,
        ic_series=pd.Series(),
        quantile_spread=0.0,
        monthly_ic_heatmap=pd.DataFrame(),
        max_drawdown_recovery_time=recovery_days,
        rolling_sharpe=pd.Series(),
        turnover_attribution={},
    )
    
    details = {
        'sub_results': sub_results,
        'weight_history': pd.DataFrame(weight_history),
        'state_history': state_history,
        'state_counts': pd.Series(state_history).value_counts().to_dict(),
    }
    
    return hybrid_result, details


def run_hybrid_grid_search(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    max_evals: int = 20,
) -> List[Tuple[Dict, BacktestResult, Dict]]:
    """
    对混合策略进行简化的网格搜索
    
    搜索维度:
    - 基础权重组合 (3种)
    - 波动率阈值 (2种)
    - 权重调整速度 (2种)
    
    Returns:
        [(参数, 结果, 详情), ...]
    """
    print(f"\n{'='*80}")
    print(f"  混合策略参数网格搜索")
    print(f"{'='*80}")
    
    # 定义搜索空间
    weight_presets = [
        {"micro_cap": 0.50, "composite": 0.30, "lowvol": 0.20},  # 激进
        {"micro_cap": 0.40, "composite": 0.35, "lowvol": 0.25},  # 平衡
        {"micro_cap": 0.30, "composite": 0.40, "lowvol": 0.30},  # 保守
    ]
    vol_thresholds = [0.20, 0.30]
    adjustment_speeds = [0.2, 0.5]
    
    combinations = []
    for wp in weight_presets:
        for vt in vol_thresholds:
            for speed in adjustment_speeds:
                combinations.append({
                    'weights': wp,
                    'vol_threshold': vt,
                    'adjustment_speed': speed,
                })
    
    # 如果组合数太多，随机采样
    if max_evals and len(combinations) > max_evals:
        np.random.shuffle(combinations)
        combinations = combinations[:max_evals]
    
    print(f"  搜索 {len(combinations)} 组参数...")
    print(f"  {'-'*60}")
    
    results = []
    for i, params in enumerate(combinations):
        result, details = run_hybrid_backtest(
            factor_panel, return_panel,
            base_weights=params['weights'],
            vol_threshold=params['vol_threshold'],
            weight_adjustment_speed=params['adjustment_speed'],
            verbose=False,
        )
        
        if result is not None:
            results.append((params, result, details))
            
            # 打印进度
            w_label = "激进" if params['weights']['micro_cap'] > 0.45 else ("保守" if params['weights']['micro_cap'] < 0.35 else "平衡")
            print(f"  [{i+1}/{len(combinations)}] {w_label} | "
                  f"波动{params['vol_threshold']:.2f} | "
                  f"调整{params['adjustment_speed']:.1f} | "
                  f"年化{result.annual_return:>5.1f}% | "
                  f"夏普{result.sharpe_ratio:>5.2f} | "
                  f"回撤{result.max_drawdown:>6.1f}% | "
                  f"卡玛{result.calmar_ratio:>5.2f}")
    
    # 按夏普排序
    results_sorted = sorted(results, key=lambda x: x[1].sharpe_ratio, reverse=True)
    
    # 打印最优
    if results_sorted:
        best_params, best_result, best_details = results_sorted[0]
        print(f"\n{'='*80}")
        print(f"  最优参数配置")
        print(f"{'='*80}")
        print(f"  权重: 微盘{best_params['weights']['micro_cap']:.0%} + "
              f"多因子{best_params['weights']['composite']:.0%} + "
              f"低波{best_params['weights']['lowvol']:.0%}")
        print(f"  波动阈值: {best_params['vol_threshold']:.2f}")
        print(f"  调整速度: {best_params['adjustment_speed']:.1f}")
        print(f"  {'-'*60}")
        print(f"  年化收益: {best_result.annual_return:.2f}%")
        print(f"  夏普比率: {best_result.sharpe_ratio:.2f}")
        print(f"  最大回撤: {best_result.max_drawdown:.2f}%")
        print(f"  卡玛比率: {best_result.calmar_ratio:.2f}")
        print(f"  月度胜率: {best_result.win_rate:.1f}%")
        print(f"  终值: {best_result.terminal_value:.2f}x")
        print(f"  {'-'*60}")
        print(f"  市场状态分布:")
        for state, count in best_details['state_counts'].items():
            print(f"    {state}: {count} 个月 ({count/len(best_details['state_history']):.1%})")
        print(f"{'='*80}")
    
    return results_sorted
