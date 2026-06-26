#!/usr/bin/env python3
"""
回测运行器 — CLI + Demo
==

用法:
  # 下载数据
  python tools/backtest_mvp/run.py download --n 30

  # 检查缓存
  python tools/backtest_mvp/run.py status

  # 回测所有策略
  python tools/backtest_mvp/run.py backtest

  # 回测单个策略
  python tools/backtest_mvp/run.py backtest --strategy 3

  # 回测模板信号
  python tools/backtest_mvp/run.py backtest --template golden_combo --template-window 20
"""

import sys
import os
import numpy as np
import pandas as pd

# 确保 thinking_and_learning_with_AI 在 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.backtest_mvp.engine import CrossSectionalEngine, BacktestResult
from tools.backtest_mvp.factors import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.strategies import ALL_STRATEGIES
from tools.backtest_mvp.strategies_v2 import NEW_STRATEGIES, TEMPLATE_STRATEGIES, LITERATURE_REVIEW
from tools.backtest_mvp.data import DATA_DIR, download_microcap_universe, get_data_summary
from tools.backtest_mvp.benchmark import (
    load_benchmarks,
    compute_benchmark_stats,
    compute_excess_return,
    get_primary_benchmark,
)


def run_single_backtest(strategy_def: dict, factor_panel: pd.DataFrame,
                        return_panel: pd.DataFrame) -> BacktestResult:
    """对单个策略运行回测

    支持两种模式:
    1. 普通策略 (ranking_factor / ranking_fn / composite_factors)
    2. 模板策略: 如果 strategy_def 有 'template' 键,
       则预计算模板信号并注入 factor_panel
    """
    # 模板策略: 预计算信号
    if "template" in strategy_def:
        from tools.backtest_mvp.factors.templates import add_template_signals
        template_name = strategy_def["template"]
        template_kwargs = strategy_def.get("template_kwargs", {})
        factor_panel = add_template_signals(
            factor_panel, [template_name],
            **{template_name: template_kwargs},
        )

    engine = CrossSectionalEngine(
        factor_panel=factor_panel,
        return_panel=return_panel,
        initial_capital=1.0,
        n_stocks=strategy_def.get("n_stocks", 30),
        rebalance_freq='M',
        commission=0.00125,  # A股: 万2.5 + 千1
        slippage=0.002,
        price_limit_stocks=True,  # 过滤涨停股 (买不到)
    )

    # 支持复合排名函数 (ranking_fn) 和单因子排名 (ranking_factor)
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
    return result


def run_single_backtest_wf(strategy_def: dict, factor_panel: pd.DataFrame,
                          return_panel: pd.DataFrame,
                          train_end: str = "2024-01-01",
                          optimization_metric: str = "sharpe") -> dict:
    """对单个策略运行 walk-forward 验证

    1. 切分 train/test
    2. 在 train 上参数优化
    3. 在 test 上验证 OOS 表现
    """
    from tools.backtest_mvp.research_loop.walk_forward import run_walk_forward

    wf = run_walk_forward(
        strategy_def, factor_panel, return_panel,
        train_end=train_end,
        optimization_metric=optimization_metric,
        verbose=True,
    )
    return {
        "strategy": wf.strategy_name,
        "train_annual": wf.train_result.annual_return,
        "train_sharpe": wf.train_result.sharpe_ratio,
        "train_drawdown": wf.train_result.max_drawdown,
        "test_annual": wf.test_result.annual_return,
        "test_sharpe": wf.test_result.sharpe_ratio,
        "test_drawdown": wf.test_result.max_drawdown,
        "overfit_ratio": wf.overfit_ratio,
        "robustness_score": wf.robustness_score,
        "best_params": wf.best_params,
    }


def cmd_walk_forward(args: list):
    """运行 walk-forward 验证"""
    # 加载数据
    print("加载数据缓存...")
    data = load_price_data(str(DATA_DIR))
    if len(data) == 0:
        print("⚠️ 未找到数据! 请先运行: python tools/backtest_mvp/run.py download")
        return

    print(f"  加载 {data['symbol'].nunique()} 只股票, {len(data)} 行")

    print("加载历史 mcap/pb 面板...")
    mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
    if not mcap_pb.empty:
        print(f"  ✓ {mcap_pb['symbol'].nunique()} 只有历史 mcap/pb 数据")
    else:
        print("  ⚠️ 无历史 mcap/pb, 将使用静态近似值")

    print("计算因子...")
    factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)

    # 选择策略
    all_strategies = ALL_STRATEGIES + NEW_STRATEGIES + TEMPLATE_STRATEGIES

    if "--strategy" in args:
        idx = int(args[args.index("--strategy") + 1]) - 1
        if 0 <= idx < len(all_strategies):
            selected = [all_strategies[idx]]
        else:
            print(f"策略编号 1-{len(all_strategies)}")
            return
    else:
        # 默认只跑关键策略（避免计算爆炸）
        selected = [ALL_STRATEGIES[0], ALL_STRATEGIES[2], ALL_STRATEGIES[4],
                   NEW_STRATEGIES[0], NEW_STRATEGIES[1], NEW_STRATEGIES[3]]

    train_end = "2024-01-01"
    if "--train-end" in args:
        train_end = args[args.index("--train-end") + 1]

    metric = "sharpe"
    if "--metric" in args:
        metric = args[args.index("--metric") + 1]

    print(f"\n{'='*80}")
    print(f"  Walk-Forward 验证")
    print(f"  训练截止: {train_end} | 优化目标: {metric}")
    print(f"{'='*80}")

    rows = []
    for s in selected:
        try:
            result = run_single_backtest_wf(s, factor_panel, return_panel, train_end, metric)
            rows.append(result)
        except Exception as e:
            print(f"  ⚠️ {s['name']} 失败: {e}")
            rows.append({
                "strategy": s["name"],
                "train_annual": 0, "train_sharpe": 0, "test_annual": 0, "test_sharpe": 0,
                "overfit_ratio": 0, "robustness_score": 0, "best_params": {},
            })

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n{'='*90}")
        print("  Walk-Forward 验证结果汇总")
        print(f"{'='*90}")
        for _, row in df.iterrows():
            print(f"  {row['strategy']:<30}")
            print(f"    Train: 年化 {row['train_annual']:>6.1f}% | 夏普 {row['train_sharpe']:>5.2f} | 回撤 {row['train_drawdown']:>6.1f}%")
            print(f"    Test:  年化 {row['test_annual']:>6.1f}% | 夏普 {row['test_sharpe']:>5.2f} | 回撤 {row['test_drawdown']:>6.1f}%")
            print(f"    过拟合比: {row['overfit_ratio']:>5.2f} | 稳健评分: {row['robustness_score']:>5.2f}")
            print(f"    最优参数: {row['best_params']}")
        print(f"{'='*90}")


def print_result_table(name: str, result: BacktestResult):
    """格式化输出单个策略的结果"""
    print(f"  {name:<30} "
          f"年化 {result.annual_return:>7.2f}%  "
          f"夏普 {result.sharpe_ratio:>6.2f}  "
          f"回撤 {result.max_drawdown:>6.2f}%  "
          f"胜率 {result.win_rate:>5.1f}%  "
          f"换手 {result.avg_turnover:>5.1f}%  "
          f"终值 {result.terminal_value:>6.2f}x")


def run_template_backtest(
    template_name: str,
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    n_stocks: int = 30,
    ascending: bool = False,
    template_kwargs: dict = None,
) -> BacktestResult:
    """Compute a panel-level template signal and run it through the engine."""
    from tools.backtest_mvp.factors.templates import add_template_signals

    template_kwargs = template_kwargs or {}
    enriched = add_template_signals(
        factor_panel,
        [template_name],
        **{template_name: template_kwargs},
    )

    # Signal coverage diagnostics
    if template_name in enriched.columns:
        signal = enriched[template_name]
        total = len(signal)
        non_null = signal.notna().sum()
        coverage = non_null / total if total > 0 else 0.0
        # Per-date coverage
        if isinstance(signal.index, pd.MultiIndex) and "date" in signal.index.names:
            per_date = signal.groupby(level="date").apply(lambda s: s.notna().mean())
            min_cov = per_date.min() if len(per_date) > 0 else 0.0
            max_cov = per_date.max() if len(per_date) > 0 else 0.0
        else:
            min_cov = max_cov = coverage
        print(f"  [signal] {template_name}: coverage={coverage:.1%} "
              f"(per-date: min={min_cov:.1%} max={max_cov:.1%}), "
              f"non-null={non_null}/{total}")
        print(f"  [signal] params: {template_kwargs}")

    engine = CrossSectionalEngine(
        factor_panel=enriched,
        return_panel=return_panel,
        initial_capital=1.0,
        n_stocks=n_stocks,
        rebalance_freq='M',
        commission=0.00125,
        slippage=0.002,
        price_limit_stocks=True,
    )
    return engine.run(ranking_factor=template_name, ascending=ascending)



def run_all_backtests(factor_panel: pd.DataFrame, return_panel: pd.DataFrame,
                      strategies: list = None):
    """运行回测并输出对比表"""
    if strategies is None:
        strategies = ALL_STRATEGIES

    # 加载基准
    bms = load_benchmarks()
    bm_stats = compute_benchmark_stats(bms)
    primary_bm = get_primary_benchmark(bm_stats)
    bm_row = bm_stats[bm_stats["benchmark"] == primary_bm]
    bm_ann = bm_row["annual_return"].iloc[0] if len(bm_row) > 0 else None

    label = f"{len(strategies)} 大策略"
    print("\n" + "=" * 105)
    print(f"  {label}回测对比 (基准: {primary_bm})")
    print("=" * 105)
    header = f"  {'策略':<28} {'年化':>7}  {'夏普':>6}  {'回撤':>7}  {'胜率':>6}  {'终值':>7}  {'超额(α)':>8}"
    print(header)
    print("  " + "-" * 95)

    results = []
    for i, s in enumerate(strategies):
        try:
            result = run_single_backtest(s, factor_panel, return_panel)
            results.append(result)
            alpha_str = ""
            if bm_ann is not None:
                alpha = compute_excess_return(result.annual_return, bm_ann)
                alpha_str = f"{alpha:>+7.1f}pp"
            print(f"  {s['name']:<28} {result.annual_return:>5.1f}%  "
                  f"{result.sharpe_ratio:>5.2f}  {result.max_drawdown:>5.1f}%  "
                  f"{result.win_rate:>4.1f}%  {result.terminal_value:>5.2f}x  {alpha_str}")
        except Exception as e:
            print(f"  {s['name']:<28} 错误: {str(e)[:50]}")

    if len(results) == 0:
        print("  ⚠️ 没有策略成功运行 (数据可能不足)")
        return

    # 汇总
    print("\n  " + "=" * 85)
    # 找到最高夏普和最高收益
    best_idx = max(range(len(results)), key=lambda i: results[i].sharpe_ratio)
    best_sharpe = results[best_idx]
    best_ret_idx = max(range(len(results)), key=lambda i: results[i].annual_return)
    best_ret = results[best_ret_idx]
    best_s_name = strategies[best_idx]["name"] if best_idx < len(strategies) else "?"
    best_r_name = strategies[best_ret_idx]["name"] if best_ret_idx < len(strategies) else "?"
    print(f"  最高夏普:  {best_s_name} ({best_sharpe.sharpe_ratio:.2f})")
    print(f"  最高收益:  {best_r_name} ({best_ret.annual_return:.1f}%)")
    if bm_ann is not None:
        print(f"  基准 ({primary_bm}): {bm_ann:+.1f}%")
        print(f"  最大超额:  {best_ret.annual_return - bm_ann:+.1f} pp")
    print(f"  数据窗口:  {factor_panel.index.get_level_values(0).min().strftime('%Y-%m-%d')} ~ "
          f"{factor_panel.index.get_level_values(0).max().strftime('%Y-%m-%d')}")
    print(f"  覆盖股票:  {factor_panel.index.get_level_values(1).nunique()} 只")


def cmd_backtest(args: list):
    """回测命令"""
    # 加载数据
    print("加载数据缓存...")
    data = load_price_data(str(DATA_DIR))
    if len(data) == 0:
        print("⚠️ 未找到数据! 请先运行: python tools/backtest_mvp/run.py download")
        return

    print(f"  加载 {data['symbol'].nunique()} 只股票, {len(data)} 行")

    # 加载历史逐日 mcap/pb (公告日对齐, 无前视偏差)
    print("加载历史 mcap/pb 面板...")
    mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
    if not mcap_pb.empty:
        print(f"  ✓ {mcap_pb['symbol'].nunique()} 只有历史 mcap/pb 数据")
    else:
        print("  ⚠️ 无历史 mcap/pb, 将使用静态近似值")

    # 计算因子
    print("计算因子...")
    factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)

    # 运行
    if "--template" in args:
        template_name = args[args.index("--template") + 1]
        n_stocks = int(args[args.index("--n") + 1]) if "--n" in args else 30
        template_kwargs = {}
        if "--template-window" in args:
            template_kwargs["window"] = int(args[args.index("--template-window") + 1])
        result = run_template_backtest(
            template_name,
            factor_panel,
            return_panel,
            n_stocks=n_stocks,
            ascending=False,
            template_kwargs=template_kwargs,
        )

        # Coverage warning
        if template_name in factor_panel.columns:
            col = factor_panel[template_name]
        else:
            # The template was just computed; re-derive to check coverage
            from tools.backtest_mvp.factors.templates import add_template_signals
            enriched = add_template_signals(
                factor_panel, [template_name],
                **{template_name: template_kwargs},
            )
            col = enriched[template_name] if template_name in enriched.columns else pd.Series(dtype=float)
        if len(col) > 0:
            cov = col.notna().mean()
            if cov < 0.3:
                print(f"  ⚠️ WARNING: signal '{template_name}' coverage is only {cov:.1%} "
                      f"(< 30%). Results may be unreliable.")

        print(f"\n模板信号 {template_name}:")
        print(f"  年化: {result.annual_return}% | 夏普: {result.sharpe_ratio} | "
              f"回撤: {result.max_drawdown}% | 终值: {result.terminal_value}x")
    elif "--strategy" in args:
        idx = int(args[args.index("--strategy") + 1]) - 1
        all_strats = ALL_STRATEGIES + NEW_STRATEGIES
        if 0 <= idx < len(all_strats):
            s = all_strats[idx]
            result = run_single_backtest(s, factor_panel, return_panel)
            print(f"\n{s['name']}:")
            print(f"  年化: {result.annual_return}% | 夏普: {result.sharpe_ratio} | "
                  f"回撤: {result.max_drawdown}% | 终值: {result.terminal_value}x")
        else:
            print(f"策略编号 1-{len(all_strats)}")
    else:
        run_all_backtests(factor_panel, return_panel)
        print()
        run_all_backtests(factor_panel, return_panel, strategies=NEW_STRATEGIES)
        print()
        run_all_backtests(factor_panel, return_panel, strategies=TEMPLATE_STRATEGIES)


def cmd_download(args: list):
    """下载数据"""
    n = 30
    if "--n" in args:
        n = int(args[args.index("--n") + 1])
    print(f"开始下载 {n} 只微盘股数据...")
    download_microcap_universe(max_stocks=n, kline_days=1500, skip_existing=True)


def cmd_status():
    """检查缓存状态"""
    summary = get_data_summary()
    if len(summary) == 0:
        print("数据缓存为空。运行 download 下载数据。")
    else:
        print(f"本地缓存: {len(summary)} 只股票")
        print(f"  日期范围: {summary['start'].min()} ~ {summary['end'].max()}")
        print(f"  总大小: {summary['size_kb'].sum():.1f} KB")
        print(f"  平均每只: {summary['rows'].mean():.0f} 行")


def cmd_hybrid(args: list):
    """运行混合策略回测"""
    print("加载数据缓存...")
    data = load_price_data(str(DATA_DIR))
    if len(data) == 0:
        print("⚠️ 未找到数据! 请先运行: python tools/backtest_mvp/run.py download")
        return
    
    print(f"  加载 {data['symbol'].nunique()} 只股票, {len(data)} 行")
    
    print("加载历史 mcap/pb 面板...")
    mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
    if not mcap_pb.empty:
        print(f"  ✓ {mcap_pb['symbol'].nunique()} 只有历史 mcap/pb 数据")
    
    print("计算因子...")
    factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)
    
    # 检查是否需要网格搜索
    if "--grid" in args:
        from tools.backtest_mvp.research_loop.hybrid_backtest import run_hybrid_grid_search
        results = run_hybrid_grid_search(factor_panel, return_panel, max_evals=12)
    else:
        from tools.backtest_mvp.research_loop.hybrid_backtest import run_hybrid_backtest
        
        # 解析参数
        base_weights = None
        if "--aggressive" in args:
            base_weights = {"micro_cap": 0.50, "composite": 0.30, "lowvol": 0.20}
        elif "--conservative" in args:
            base_weights = {"micro_cap": 0.30, "composite": 0.40, "lowvol": 0.30}
        
        vol_threshold = 0.25
        if "--vol-threshold" in args:
            vol_threshold = float(args[args.index("--vol-threshold") + 1])
        
        result, details = run_hybrid_backtest(
            factor_panel, return_panel,
            base_weights=base_weights,
            vol_threshold=vol_threshold,
        )
        
        if result:
            print(f"\n{'='*80}")
            print(f"  混合策略回测结果")
            print(f"{'='*80}")
            print(f"  年化收益: {result.annual_return:.2f}%")
            print(f"  夏普比率: {result.sharpe_ratio:.2f}")
            print(f"  最大回撤: {result.max_drawdown:.2f}%")
            print(f"  卡玛比率: {result.calmar_ratio:.2f}")
            print(f"  月度胜率: {result.win_rate:.1f}%")
            print(f"  终值: {result.terminal_value:.2f}x")
            print(f"  {'-'*60}")
            print(f"  子策略对比:")
            for name, sub in details['sub_results'].items():
                print(f"    {name:>12}: 年化{sub.annual_return:>6.1f}% 夏普{sub.sharpe_ratio:>5.2f} 回撤{sub.max_drawdown:>6.1f}%")
            print(f"  {'-'*60}")
            print(f"  市场状态分布:")
            for state, count in details['state_counts'].items():
                print(f"    {state}: {count} 个月")
            print(f"{'='*80}")

def main():
    if len(sys.argv) < 2:
        print("用法: python tools/backtest_mvp/run.py [download|status|backtest|walk-forward|hybrid]")
        print()
        print("  download      - 下载微盘股数据 (--n 30 控制数量)")
        print("  status        - 查看数据缓存状态")
        print("  backtest      - 运行全部策略回测 (--strategy 1-N 选单个)")
        print("  walk-forward  - Walk-forward验证: 2024年前训练, 2024年后测试")
        print("                  (--strategy N 选单个, --train-end 2024-01-01 切分点)")
        print("                  (--metric sharpe|annual_return|calmar 优化目标)")
        print("  hybrid        - 混合策略回测 (多策略组合+动态权重+市场状态过滤)")
        print("                  (--grid 网格搜索最优参数)")
        print("                  (--aggressive 激进配置 | --conservative 保守配置)")
        print("                  (--vol-threshold 0.25 波动率阈值)")
        return

    cmd = sys.argv[1]
    args = sys.argv[1:]

    if cmd == "download":
        cmd_download(args)
    elif cmd == "status":
        cmd_status()
    elif cmd == "backtest":
        cmd_backtest(args)
    elif cmd == "walk-forward":
        cmd_walk_forward(args)
    elif cmd == "hybrid":
        cmd_hybrid(args)
    elif cmd == "v2":
        cmd_backtest(args)
        print("\n" + "=" * 95)
        print("  文献综述 (PART I — 见 strategies_v2.py 文档)")
        print("=" * 95)
        print(LITERATURE_REVIEW[:1500])
        print("\n  ... (完整综述见 tools/backtest_mvp/strategies_v2.py)")
    elif cmd == "review":
        print(LITERATURE_REVIEW)
    else:
        print(f"未知命令: {cmd}")
        print()
        print("用法: python tools/backtest_mvp/run.py [download|status|backtest|walk-forward|hybrid]")
        print()
        print("  download      - 下载微盘股数据 (--n 30 控制数量)")
        print("  status        - 查看数据缓存状态")
        print("  backtest      - 运行全部策略回测 (--strategy 1-N 选单个)")
        print("  walk-forward  - Walk-forward验证: 2024年前训练, 2024年后测试")
        print("                  (--strategy N 选单个, --train-end 2024-01-01 切分点)")
        print("                  (--metric sharpe|annual_return|calmar 优化目标)")
        print("  hybrid        - 混合策略回测 (多策略组合+动态权重+市场状态过滤)")
        print("                  (--grid 网格搜索最优参数)")
        print("                  (--aggressive 激进配置 | --conservative 保守配置)")
        print("                  (--vol-threshold 0.25 波动率阈值)")
        return


if __name__ == "__main__":
    main()
