#!/usr/bin/env python3
"""
Phase 1 测试: 验证 Fast Engine 输出与原始引擎完全一致

测试策略:
1. 使用相同输入数据，分别运行原始引擎和 Fast 引擎
2. 比较所有输出字段，差异应在浮点误差范围 (< 1e-6)
3. 使用不同规模的测试数据 (100只 / 500只 / 全量)

Date: 2026-06-28
"""

import sys, os, time
sys.path.insert(0, '/Users/hejinyang/thinking_and_learning_with_AI')

import numpy as np
import pandas as pd
from tools.backtest_mvp.factors.legacy import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.data import DATA_DIR
from tools.backtest_mvp.engine import CrossSectionalEngine
from tools.backtest_mvp.engine_fast import CrossSectionalEngineFast


def load_test_data(n_stocks: int = None):
    """加载测试数据，可限制股票数量"""
    data = load_price_data(str(DATA_DIR))

    if n_stocks is not None and n_stocks > 0:
        symbols = data['symbol'].unique()[:n_stocks]
        data = data[data['symbol'].isin(symbols)]

    mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
    factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)

    return factor_panel, return_panel


def compare_results(orig_result, fast_result, tol: float = 1e-6) -> dict:
    """
    比较两个 BacktestResult 对象的差异

    Returns
    -------
    dict with pass/fail status and max difference for each field
    """
    report = {"passed": True, "fields": {}, "errors": []}

    scalar_fields = [
        'annual_return', 'annual_volatility', 'sharpe_ratio',
        'max_drawdown', 'calmar_ratio', 'win_rate', 'avg_turnover',
        'terminal_value', 'ic_mean', 'ic_ir', 'quantile_spread',
        'max_drawdown_recovery_time', 'stop_triggered',
    ]

    for field in scalar_fields:
        orig_val = getattr(orig_result, field, None)
        fast_val = getattr(fast_result, field, None)

        if orig_val is None and fast_val is None:
            continue

        if isinstance(orig_val, (int, float)) and isinstance(fast_val, (int, float)):
            diff = abs(orig_val - fast_val)
            report["fields"][field] = {
                "orig": orig_val, "fast": fast_val, "diff": diff, "pass": diff < tol
            }
            if diff >= tol:
                report["passed"] = False
                report["errors"].append(f"{field}: diff={diff:.6f} (orig={orig_val}, fast={fast_val})")
        elif orig_val != fast_val:
            report["passed"] = False
            report["errors"].append(f"{field}: orig={orig_val} != fast={fast_val}")

    # 比较 Series 类型字段
    series_fields = ['equity_curve', 'monthly_returns', 'ic_series', 'rolling_sharpe']
    for field in series_fields:
        orig_series = getattr(orig_result, field, None)
        fast_series = getattr(fast_result, field, None)

        if orig_series is None and fast_series is None:
            continue

        if isinstance(orig_series, pd.Series) and isinstance(fast_series, pd.Series):
            # 对齐索引后比较
            merged = pd.merge(
                orig_series.rename('orig'),
                fast_series.rename('fast'),
                left_index=True, right_index=True, how='inner'
            )
            if len(merged) > 0:
                diff = (merged['orig'] - merged['fast']).abs().max()
                report["fields"][field] = {
                    "diff": diff, "pass": diff < tol, "n_points": len(merged)
                }
                if diff >= tol:
                    report["passed"] = False
                    report["errors"].append(f"{field}: max_diff={diff:.6f}")

    # 比较 DataFrame 类型字段
    df_fields = ['positions_log', 'monthly_ic_heatmap']
    for field in df_fields:
        orig_df = getattr(orig_result, field, None)
        fast_df = getattr(fast_result, field, None)

        if orig_df is None and fast_df is None:
            continue

        if isinstance(orig_df, pd.DataFrame) and isinstance(fast_df, pd.DataFrame):
            # 简单比较 shape 和 sum
            if orig_df.shape != fast_df.shape:
                report["passed"] = False
                report["errors"].append(f"{field}: shape mismatch {orig_df.shape} vs {fast_df.shape}")
            else:
                diff = (orig_df.fillna(0) - fast_df.fillna(0)).abs().max().max()
                report["fields"][field] = {
                    "diff": diff, "pass": diff < tol, "shape": orig_df.shape
                }
                if diff >= tol:
                    report["passed"] = False
                    report["errors"].append(f"{field}: max_diff={diff:.6f}")

    # 比较列表
    list_fields = ['monthly_turnover_log']
    for field in list_fields:
        orig_list = getattr(orig_result, field, None)
        fast_list = getattr(fast_result, field, None)
        if orig_list is None and fast_list is None:
            continue
        if len(orig_list) != len(fast_list):
            report["passed"] = False
            report["errors"].append(f"{field}: length mismatch {len(orig_list)} vs {len(fast_list)}")
        else:
            if len(orig_list) > 0:
                diff = max(abs(a - b) for a, b in zip(orig_list, fast_list))
                report["fields"][field] = {"diff": diff, "pass": diff < tol}
                if diff >= tol:
                    report["passed"] = False
                    report["errors"].append(f"{field}: max_diff={diff:.6f}")

    return report


def test_micro_cap_strategy(n_stocks: int = 100):
    """测试 Micro-Cap 策略"""
    print(f"\n{'='*70}")
    print(f"Test 1: Micro-Cap Strategy ({n_stocks} stocks)")
    print(f"{'='*70}")

    factor_panel, return_panel = load_test_data(n_stocks)
    print(f"Data: {factor_panel.index.get_level_values(1).nunique()} stocks, "
          f"{len(factor_panel):,} rows")

    universe_filter = lambda snapshot, dates, i: list(
        snapshot[snapshot['mcap'] < 50].index
    ) if 'mcap' in snapshot.columns else list(snapshot.index)

    # Original engine
    print("\n  [Original] Running...")
    t0 = time.time()
    orig_engine = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )
    orig_result = orig_engine.run(
        universe_filter=universe_filter, ranking_factor='mcap', ascending=True,
    )
    t_orig = time.time() - t0
    print(f"  Time: {t_orig:.2f}s")
    print(f"  Result: Annual={orig_result.annual_return:.2f}%, "
          f"Sharpe={orig_result.sharpe_ratio:.2f}, DD={orig_result.max_drawdown:.2f}%")

    # Fast engine
    print("\n  [Fast] Running...")
    t0 = time.time()
    fast_engine = CrossSectionalEngineFast(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )
    fast_result = fast_engine.run(
        universe_filter=universe_filter, ranking_factor='mcap', ascending=True,
    )
    t_fast = time.time() - t0
    print(f"  Time: {t_fast:.2f}s")
    print(f"  Result: Annual={fast_result.annual_return:.2f}%, "
          f"Sharpe={fast_result.sharpe_ratio:.2f}, DD={fast_result.max_drawdown:.2f}%")

    # Compare
    print(f"\n  Comparison:")
    report = compare_results(orig_result, fast_result, tol=1e-5)
    print(f"  Speedup: {t_orig/t_fast:.1f}x")
    print(f"  Test: {'PASSED' if report['passed'] else 'FAILED'}")
    if not report['passed']:
        for err in report['errors']:
            print(f"    ERROR: {err}")

    return report['passed'], t_orig, t_fast


def test_momentum_strategy(n_stocks: int = 100):
    """测试 Momentum 策略"""
    print(f"\n{'='*70}")
    print(f"Test 2: Momentum Strategy ({n_stocks} stocks)")
    print(f"{'='*70}")

    factor_panel, return_panel = load_test_data(n_stocks)

    universe_filter = lambda snapshot, dates, i: list(
        snapshot[snapshot['mcap'] < 100].index
    ) if 'mcap' in snapshot.columns else list(snapshot.index)

    # Original
    print("\n  [Original] Running...")
    t0 = time.time()
    orig_engine = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )
    orig_result = orig_engine.run(
        universe_filter=universe_filter, ranking_factor='mom20d', ascending=False,
    )
    t_orig = time.time() - t0
    print(f"  Time: {t_orig:.2f}s")

    # Fast
    print("\n  [Fast] Running...")
    t0 = time.time()
    fast_engine = CrossSectionalEngineFast(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )
    fast_result = fast_engine.run(
        universe_filter=universe_filter, ranking_factor='mom20d', ascending=False,
    )
    t_fast = time.time() - t0
    print(f"  Time: {t_fast:.2f}s")

    report = compare_results(orig_result, fast_result, tol=1e-5)
    print(f"\n  Speedup: {t_orig/t_fast:.1f}x")
    print(f"  Test: {'PASSED' if report['passed'] else 'FAILED'}")
    if not report['passed']:
        for err in report['errors']:
            print(f"    ERROR: {err}")

    return report['passed'], t_orig, t_fast


def test_composite_factors(n_stocks: int = 100):
    """测试复合因子策略"""
    print(f"\n{'='*70}")
    print(f"Test 3: Composite Factors ({n_stocks} stocks)")
    print(f"{'='*70}")

    factor_panel, return_panel = load_test_data(n_stocks)

    universe_filter = lambda snapshot, dates, i: list(
        snapshot[snapshot['mcap'] < 100].index
    ) if 'mcap' in snapshot.columns else list(snapshot.index)

    composite_factors = [('mcap', True), ('pb', True), ('mom20d', False)]

    # Original
    print("\n  [Original] Running...")
    t0 = time.time()
    orig_engine = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )
    orig_result = orig_engine.run(
        universe_filter=universe_filter, composite_factors=composite_factors,
    )
    t_orig = time.time() - t0
    print(f"  Time: {t_orig:.2f}s")

    # Fast
    print("\n  [Fast] Running...")
    t0 = time.time()
    fast_engine = CrossSectionalEngineFast(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )
    fast_result = fast_engine.run(
        universe_filter=universe_filter, composite_factors=composite_factors,
    )
    t_fast = time.time() - t0
    print(f"  Time: {t_fast:.2f}s")

    report = compare_results(orig_result, fast_result, tol=1e-5)
    print(f"\n  Speedup: {t_orig/t_fast:.1f}x")
    print(f"  Test: {'PASSED' if report['passed'] else 'FAILED'}")
    if not report['passed']:
        for err in report['errors']:
            print(f"    ERROR: {err}")

    return report['passed'], t_orig, t_fast


def test_neutralize(n_stocks: int = 100):
    """测试中性化策略"""
    print(f"\n{'='*70}")
    print(f"Test 4: Neutralized Strategy ({n_stocks} stocks)")
    print(f"{'='*70}")

    factor_panel, return_panel = load_test_data(n_stocks)

    universe_filter = lambda snapshot, dates, i: list(
        snapshot[snapshot['mcap'] < 100].index
    ) if 'mcap' in snapshot.columns else list(snapshot.index)

    # Original
    print("\n  [Original] Running...")
    t0 = time.time()
    orig_engine = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )
    orig_result = orig_engine.run(
        universe_filter=universe_filter, ranking_factor='mcap', ascending=True,
        neutralize=True, neutralize_strength=0.5,
    )
    t_orig = time.time() - t0
    print(f"  Time: {t_orig:.2f}s")

    # Fast
    print("\n  [Fast] Running...")
    t0 = time.time()
    fast_engine = CrossSectionalEngineFast(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )
    fast_result = fast_engine.run(
        universe_filter=universe_filter, ranking_factor='mcap', ascending=True,
        neutralize=True, neutralize_strength=0.5,
    )
    t_fast = time.time() - t0
    print(f"  Time: {t_fast:.2f}s")

    report = compare_results(orig_result, fast_result, tol=1e-5)
    print(f"\n  Speedup: {t_orig/t_fast:.1f}x")
    print(f"  Test: {'PASSED' if report['passed'] else 'FAILED'}")
    if not report['passed']:
        for err in report['errors']:
            print(f"    ERROR: {err}")

    return report['passed'], t_orig, t_fast


def test_ic_decay(n_stocks: int = 100):
    """测试 IC decay 计算"""
    print(f"\n{'='*70}")
    print(f"Test 5: IC Decay ({n_stocks} stocks)")
    print(f"{'='*70}")

    factor_panel, return_panel = load_test_data(n_stocks)

    orig_engine = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )
    fast_engine = CrossSectionalEngineFast(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )

    print("\n  [Original] IC decay...")
    t0 = time.time()
    orig_decay = orig_engine.compute_ic_decay(ranking_factor='mcap', lags=(1, 5, 10))
    t_orig = time.time() - t0
    print(f"  Time: {t_orig:.2f}s, Result: {orig_decay}")

    print("\n  [Fast] IC decay...")
    t0 = time.time()
    fast_decay = fast_engine.compute_ic_decay(ranking_factor='mcap', lags=(1, 5, 10))
    t_fast = time.time() - t0
    print(f"  Time: {t_fast:.2f}s, Result: {fast_decay}")

    passed = True
    for lag in orig_decay:
        diff = abs(orig_decay[lag] - fast_decay[lag])
        if diff > 1e-5:
            print(f"  ERROR: lag={lag}, diff={diff:.6f}")
            passed = False

    print(f"\n  Speedup: {t_orig/t_fast:.1f}x")
    print(f"  Test: {'PASSED' if passed else 'FAILED'}")
    return passed, t_orig, t_fast


def run_all_tests():
    """运行所有测试"""
    print("="*70)
    print("  Phase 1: Fast Engine 一致性测试")
    print("="*70)

    results = []

    # 100 stocks
    for n in [100, 500]:
        print(f"\n\n{'#'*70}")
        print(f"# 测试规模: {n} 只股票")
        print(f"{'#'*70}")

        p1, t1o, t1f = test_micro_cap_strategy(n)
        p2, t2o, t2f = test_momentum_strategy(n)
        p3, t3o, t3f = test_composite_factors(n)
        p4, t4o, t4f = test_neutralize(n)
        p5, t5o, t5f = test_ic_decay(n)

        total_orig = t1o + t2o + t3o + t4o + t5o
        total_fast = t1f + t2f + t3f + t4f + t5f

        all_passed = all([p1, p2, p3, p4, p5])

        print(f"\n{'='*70}")
        print(f"  Summary ({n} stocks):")
        print(f"  Total tests: 5, Passed: {sum([p1,p2,p3,p4,p5])}")
        print(f"  Total time (orig): {total_orig:.2f}s")
        print(f"  Total time (fast): {total_fast:.2f}s")
        print(f"  Overall speedup: {total_orig/total_fast:.1f}x")
        print(f"  {'='*70}")

        results.append({
            'n_stocks': n,
            'all_passed': all_passed,
            'total_orig': total_orig,
            'total_fast': total_fast,
            'speedup': total_orig / total_fast if total_fast > 0 else 0,
        })

    # 最终总结
    print(f"\n\n{'#'*70}")
    print(f"# 最终测试报告")
    print(f"{'#'*70}")
    for r in results:
        status = "PASSED" if r['all_passed'] else "FAILED"
        print(f"  {r['n_stocks']:>4} stocks: {status} | "
              f"Speedup: {r['speedup']:.1f}x | "
              f"Time: {r['total_orig']:.1f}s -> {r['total_fast']:.1f}s")

    all_passed = all(r['all_passed'] for r in results)
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
