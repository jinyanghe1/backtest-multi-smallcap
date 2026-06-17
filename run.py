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
from tools.backtest_mvp.strategies_v2 import NEW_STRATEGIES, LITERATURE_REVIEW
from tools.backtest_mvp.data import DATA_DIR, download_microcap_universe, get_data_summary
from tools.backtest_mvp.benchmark import (
    load_benchmarks,
    compute_benchmark_stats,
    compute_excess_return,
    get_primary_benchmark,
)


def run_single_backtest(strategy_def: dict, factor_panel: pd.DataFrame,
                        return_panel: pd.DataFrame) -> BacktestResult:
    """对单个策略运行回测"""
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
        ranking_fn=strategy_def.get("ranking_fn"),
        factor_weights=strategy_def.get("factor_weights"),
    )
    return result


def print_result_table(name: str, result: BacktestResult):
    """格式化输出单个策略的结果"""
    print(f"  {name:<30} "
          f"年化 {result.annual_return:>7.2f}%  "
          f"夏普 {result.sharpe_ratio:>6.2f}  "
          f"回撤 {result.max_drawdown:>6.2f}%  "
          f"胜率 {result.win_rate:>5.1f}%  "
          f"换手 {result.avg_turnover:>5.1f}%  "
          f"终值 {result.terminal_value:>6.2f}x")



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
    if "--strategy" in args:
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


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/backtest_mvp/run.py [download|status|backtest]")
        print()
        print("  download    - 下载微盘股数据 (--n 30 控制数量)")
        print("  status      - 查看数据缓存状态")
        print("  backtest    - 运行全部策略回测 (--strategy 1-6 选单个)")
        return

    cmd = sys.argv[1]
    args = sys.argv[1:]

    if cmd == "download":
        cmd_download(args)
    elif cmd == "status":
        cmd_status()
    elif cmd == "backtest":
        cmd_backtest(args)
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


if __name__ == "__main__":
    main()
