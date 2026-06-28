#!/usr/bin/env python3
"""
Test Phase 2: Vectorized backtest loop vs original engine

目标:
1. 验证 V2 引擎输出与原始引擎一致（容忍浮点误差）
2. 测量性能提升
3. 测试不同规模的股票池

Date: 2026-06-28
"""

import sys, time, numpy as np
import pandas as pd

sys.path.insert(0, '/Users/hejinyang/thinking_and_learning_with_AI')

from tools.backtest_mvp.factors.legacy import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.data import DATA_DIR
from tools.backtest_mvp.engine import CrossSectionalEngine
from tools.backtest_mvp.engine_v2 import CrossSectionalEngineV2


def run_comparison(n_stocks: int = 100, strategy: str = "microcap"):
    """
    对比原始引擎和 V2 引擎的结果
    """
    print(f"\n{'='*70}")
    print(f"  对比测试: {n_stocks} 只股票, {strategy} 策略")
    print(f"{'='*70}")

    # 1. Load data
    print(f"\n[1/3] Loading data...")
    t0 = time.time()
    data = load_price_data(str(DATA_DIR))
    all_symbols = data['symbol'].unique()
    if n_stocks < len(all_symbols):
        symbols = all_symbols[:n_stocks]
        data = data[data['symbol'].isin(symbols)]
    t1 = time.time()
    print(f"  Data: {data['symbol'].nunique()} stocks, {len(data):,} rows ({t1-t0:.1f}s)")

    # 2. Compute factors
    print(f"\n[2/3] Computing factors...")
    t0 = time.time()
    mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
    factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)
    t1 = time.time()
    print(f"  Factors: {len(factor_panel):,} rows ({t1-t0:.1f}s)")

    # 3. Define strategy
    if strategy == "microcap":
        universe_filter = lambda snapshot, dates, i: list(snapshot[snapshot['mcap'] < 50].index) if 'mcap' in snapshot.columns else list(snapshot.index)
        ranking_factor = 'mcap'
        ascending = True
    elif strategy == "low_pb":
        universe_filter = lambda snapshot, dates, i: list(snapshot[snapshot['mcap'] < 100].index) if 'mcap' in snapshot.columns else list(snapshot.index)
        ranking_factor = 'pb'
        ascending = True
    elif strategy == "momentum":
        universe_filter = lambda snapshot, dates, i: list(snapshot[snapshot['mcap'] < 100].index) if 'mcap' in snapshot.columns else list(snapshot.index)
        ranking_factor = 'mom20d'
        ascending = False
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 4. Run original engine
    print(f"\n[3/3] Running engines...")
    print(f"\n  Original engine:")
    t0 = time.time()
    engine_orig = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )
    result_orig = engine_orig.run(
        universe_filter=universe_filter,
        ranking_factor=ranking_factor,
        ascending=ascending,
    )
    t_orig = time.time() - t0
    print(f"    Time: {t_orig:.2f}s")
    print(f"    Annual: {result_orig.annual_return:.2f}%, Sharpe: {result_orig.sharpe_ratio:.2f}, DD: {result_orig.max_drawdown:.2f}%")

    # 5. Run V2 engine
    print(f"\n  V2 engine (vectorized):")
    t0 = time.time()
    engine_v2 = CrossSectionalEngineV2(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )
    result_v2 = engine_v2.run(
        universe_filter=universe_filter,
        ranking_factor=ranking_factor,
        ascending=ascending,
    )
    t_v2 = time.time() - t0
    print(f"    Time: {t_v2:.2f}s")
    print(f"    Annual: {result_v2.annual_return:.2f}%, Sharpe: {result_v2.sharpe_ratio:.2f}, DD: {result_v2.max_drawdown:.2f}%")

    # 6. Compare results
    print(f"\n{'='*70}")
    print(f"  结果对比")
    print(f"{'='*70}")

    metrics = [
        ('Annual Return', result_orig.annual_return, result_v2.annual_return, 0.1),
        ('Sharpe Ratio', result_orig.sharpe_ratio, result_v2.sharpe_ratio, 0.01),
        ('Max Drawdown', result_orig.max_drawdown, result_v2.max_drawdown, 0.1),
        ('Win Rate', result_orig.win_rate, result_v2.win_rate, 0.1),
        ('Terminal Value', result_orig.terminal_value, result_v2.terminal_value, 0.01),
    ]

    all_pass = True
    for name, orig, v2, tol in metrics:
        diff = abs(orig - v2)
        status = "PASS" if diff <= tol else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name:<20}  Orig: {orig:>8.2f}  V2: {v2:>8.2f}  Diff: {diff:>8.4f}  [{status}]")

    speedup = t_orig / t_v2 if t_v2 > 0 else float('inf')
    print(f"\n  Speedup: {speedup:.1f}x ({t_orig:.2f}s → {t_v2:.2f}s)")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'} (tolerance applied)")

    return {
        'orig_time': t_orig,
        'v2_time': t_v2,
        'speedup': speedup,
        'pass': all_pass,
        'orig_result': result_orig,
        'v2_result': result_v2,
    }


def benchmark_all_sizes():
    """
    测试不同规模的股票池
    """
    sizes = [100, 500, 1000, 2500, 5000]
    results = []

    for size in sizes:
        try:
            r = run_comparison(n_stocks=size, strategy="microcap")
            results.append((size, r['speedup'], r['pass']))
        except Exception as e:
            print(f"\n  ERROR with {size} stocks: {e}")
            results.append((size, None, False))

    print(f"\n{'='*70}")
    print(f"  Benchmark Summary")
    print(f"{'='*70}")
    print(f"  {'Size':>8} {'Speedup':>10} {'Status':>8}")
    print(f"  {'-'*30}")
    for size, speedup, passed in results:
        s = f"{speedup:.1f}x" if speedup else "N/A"
        p = "PASS" if passed else "FAIL"
        print(f"  {size:>8} {s:>10} {p:>8}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=500, help="Number of stocks to test")
    parser.add_argument("--strategy", type=str, default="microcap", help="Strategy name")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark across all sizes")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_all_sizes()
    else:
        run_comparison(n_stocks=args.size, strategy=args.strategy)
