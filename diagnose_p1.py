#!/usr/bin/env python3
"""P1 诊断脚本 — 运行因子挖掘 + 集成回测

用法:
  cd thinking_and_learning_with_AI
  PYTHONPATH=/path/to/project python3 tools/backtest_mvp/diagnose_p1.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from tools.backtest_mvp.factors import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.data import DATA_DIR
from tools.backtest_mvp.p0_engine_v2 import P0EngineV2
from tools.backtest_mvp.p1_factor_mining import run_p1_factor_mining
from tools.backtest_mvp.p0_enhanced_engine import P0EnhancedEngine
from tools.backtest_mvp.benchmark import load_benchmarks, compute_benchmark_stats, get_primary_benchmark


def run_p1_diagnosis():
    print("=" * 80)
    print("  P1 诊断: 新因子挖掘 + 集成回测")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    data = load_price_data(str(DATA_DIR))
    print(f"  加载 {data['symbol'].nunique()} 只股票, {len(data)} 行")
    
    mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
    print(f"  加载 mcap/pb: {mcap_pb['symbol'].nunique() if not mcap_pb.empty else 0} 只")
    
    print("  计算因子...")
    factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)
    
    # 2. P1 因子挖掘
    print("\n[2/4] P1 因子挖掘...")
    new_factors, eval_results = run_p1_factor_mining(factor_panel, return_panel)
    
    # 3. 集成新因子到引擎
    print("\n[3/4] 集成新因子回测...")
    
    # 将新因子注入面板
    enriched_panel = factor_panel.copy()
    for name, factor in new_factors.items():
        if not factor.isna().all():
            enriched_panel[name] = factor
            print(f"  注入因子: {name}")
    
    # 选择通过评估的因子
    valid_factors = eval_results[eval_results["valid"] == True]["factor"].tolist()
    print(f"\n  有效因子: {valid_factors}")
    
    # 构建复合策略：等权集成所有有效因子
    if len(valid_factors) >= 2:
        composite = [(f, True) for f in valid_factors]  # 全部升序（值越大越好）
        weights = {f: 1.0 / len(valid_factors) for f in valid_factors}
        
        # P0 引擎回测
        engine = P0EnhancedEngine(
            factor_panel=enriched_panel,
            return_panel=return_panel,
            initial_capital=1.0,
            n_stocks=30,
            rebalance_freq='M',
            commission=0.00125,
            slippage=0.002,
            price_limit_stocks=True,
            enable_pit_universe=True,
            enable_adv_impact=True,
            enable_risk_overlay=True,
            enable_deflated_sharpe=True,
            n_trials=5,  # 只跑5个组合
        )
        
        result = engine.run(
            composite_factors=composite,
            factor_weights=weights,
        )
        
        print(f"\n  新因子集成策略 (P0):")
        print(f"    年化: {result.annual_return:.2f}%")
        print(f"    夏普: {result.sharpe_ratio:.2f}")
        print(f"    回撤: {result.max_drawdown:.2f}%")
        if hasattr(result, 'deflated_sharpe'):
            print(f"    DSR: {result.deflated_sharpe:.4f}")
    else:
        print("  ⚠️ 有效因子不足2个，无法构建集成策略")
        result = None
    
    # 4. 对比原最佳策略
    print("\n[4/4] 对比基准...")
    
    # SC 低波+低换手 (P0 最优)
    engine_sc = P0EnhancedEngine(
        factor_panel=factor_panel,
        return_panel=return_panel,
        initial_capital=1.0,
        n_stocks=30,
        rebalance_freq='M',
        commission=0.00125,
        slippage=0.002,
        price_limit_stocks=True,
        enable_pit_universe=True,
        enable_adv_impact=True,
        enable_risk_overlay=True,
        enable_deflated_sharpe=True,
        n_trials=1,
    )
    
    # 手动构建 SC 策略 (低波+低换手)
    sc_result = engine_sc.run(
        ranking_factor="low_volatility_neut" if "low_volatility_neut" in factor_panel.columns else "low_volatility",
        ascending=True,
    )
    
    print(f"\n  SC 低波+低换手 (P0 基准):")
    print(f"    年化: {sc_result.annual_return:.2f}%")
    print(f"    夏普: {sc_result.sharpe_ratio:.2f}")
    print(f"    回撤: {sc_result.max_drawdown:.2f}%")
    
    # 5. 结论
    print("\n" + "=" * 80)
    print("  P1 诊断结论")
    print("=" * 80)
    
    if result is not None:
        print(f"\n  新因子集成 vs 基准 SC:")
        print(f"    年化: {result.annual_return:.2f}% vs {sc_result.annual_return:.2f}% (Δ{result.annual_return - sc_result.annual_return:+.2f}pp)")
        print(f"    夏普: {result.sharpe_ratio:.2f} vs {sc_result.sharpe_ratio:.2f} (Δ{result.sharpe_ratio - sc_result.sharpe_ratio:+.2f})")
        print(f"    回撤: {result.max_drawdown:.2f}% vs {sc_result.max_drawdown:.2f}% (Δ{result.max_drawdown - sc_result.max_drawdown:+.2f}pp)")
        
        if result.sharpe_ratio >= 1.0 and result.max_drawdown > -30:
            print(f"\n  ✅ 新因子集成策略通过 P0 部署检查")
        else:
            print(f"\n  ⚠️ 新因子集成策略未通过部署检查，需进一步调优")
    
    # 保存评估结果
    eval_path = Path(__file__).parent / "diagnose_p1_factors.csv"
    eval_results.to_csv(eval_path, index=False, encoding="utf-8-sig")
    print(f"\n  因子评估报告: {eval_path}")
    
    return new_factors, eval_results, result


if __name__ == "__main__":
    run_p1_diagnosis()
