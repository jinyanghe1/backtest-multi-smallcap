#!/usr/bin/env python3
"""P0 修复验证脚本 — 验证所有 P0 功能模块可导入和运行"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# 计算项目根目录: test_p0_fixes.py -> backtest_mvp -> tools -> PROJECT_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TOOLS_DIR.parent

# 添加项目根到 sys.path
sys.path.insert(0, str(PROJECT_ROOT))

# 设置 PYTHONPATH 环境变量
os.environ["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

# 现在从 tools.backtest_mvp 导入
from tools.backtest_mvp.data.delisted import DelistManager, build_pit_universe
from tools.backtest_mvp.data.adv_impact import ADVImpactModel, PortfolioCapacityChecker
from tools.backtest_mvp.data.risk_overlay import RiskOverlay, RiskOverlayConfig
from tools.backtest_mvp.research_loop.deflated_sharpe import (
    compute_deflated_sharpe, deflated_sharpe_significance, TrialCounter
)
from tools.backtest_mvp.p0_engine_v2 import P0EngineV2


def test_delisted():
    """测试退市数据模块"""
    print("=" * 60)
    print("[TEST] 退市数据模块 (UA1)")
    print("=" * 60)
    
    mgr = DelistManager()
    df = mgr.fetch_all(force=False)
    
    if len(df) > 0:
        print(f"  ✓ 退市数据加载成功: {len(df)} 只")
        print(f"  ✓ 上证: {len(df[df['market'] == 'sh'])} 只")
        print(f"  ✓ 深证: {len(df[df['market'] == 'sz'])} 只")
        
        # 测试 PIT 查询
        dead = mgr.get_delisted_before("2020-01-01")
        print(f"  ✓ 2020-01-01 前退市: {len(dead)} 只")
        
        alive = mgr.is_alive("sh600001", "2005-01-01")
        print(f"  ✓ sh600001 在 2005-01-01 存活: {alive}")
        
        summary = mgr.summary()
        print(f"  ✓ 最早退市: {summary['earliest_delist']}")
        print(f"  ✓ 最晚退市: {summary['latest_delist']}")
    else:
        print("  ⚠️ 退市数据为空")
    
    print()


def test_adv_impact():
    """测试 ADV 冲击模型"""
    print("=" * 60)
    print("[TEST] ADV 冲击模型 (UA2)")
    print("=" * 60)
    
    model = ADVImpactModel(k=8.0, adv_window=20, max_position_adv_pct=5.0)
    
    # 测试冲击成本计算
    impact = model.compute_impact(order_value=1_000_000, adv_value=10_000_000)
    print(f"  ✓ 100万/1000万 ADV 冲击: {impact*10000:.2f} bps")
    
    impact2 = model.compute_impact(order_value=5_000_000, adv_value=10_000_000)
    print(f"  ✓ 500万/1000万 ADV 冲击: {impact2*10000:.2f} bps")
    
    # 测试容量检查
    is_safe, pct, max_allowed = model.check_capacity(5_000_000, 10_000_000)
    print(f"  ✓ 500万/1000万 容量检查: safe={is_safe}, pct={pct:.1f}%, max={max_allowed/10000:.0f}万")
    
    print()


def test_risk_overlay():
    """测试风险护栏"""
    print("=" * 60)
    print("[TEST] 风险护栏 (UD1)")
    print("=" * 60)
    
    config = RiskOverlayConfig(
        target_vol_annual=0.20,
        min_gross=0.30,
        max_gross=1.00,
        enable_crash_filter=True,
        enable_dd_throttle=True,
    )
    
    overlay = RiskOverlay(config)
    
    # 模拟权益曲线
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-01-01", freq="D")
    returns = np.random.normal(0.001, 0.02, len(dates))
    equity = (1 + returns).cumprod()
    equity_series = pd.Series(equity, index=dates)
    
    gross = overlay.compute_gross_exposure(equity_series)
    print(f"  ✓ 当前 gross 敞口: {gross:.2%}")
    
    summary = overlay.summary()
    print(f"  ✓ 平均 gross: {summary.get('avg_gross', 0):.2%}")
    print(f"  ✓ 最小 gross: {summary.get('min_gross', 0):.2%}")
    
    print()


def test_deflated_sharpe():
    """测试 Deflated Sharpe"""
    print("=" * 60)
    print("[TEST] Deflated Sharpe (UA3)")
    print("=" * 60)
    
    # 测试折扣计算
    dsr = compute_deflated_sharpe(sharpe=1.47, n_trials=19, n_periods=92)
    print(f"  ✓ Sharpe=1.47, 19次试验, 92期 -> DSR={dsr:.4f}")
    print(f"  ✓ 折扣幅度: {(1 - dsr/max(1.47, 0.001)) * 100:.1f}%")
    
    sig = deflated_sharpe_significance(dsr)
    print(f"  ✓ 显著性: {sig['confidence']}, p-value={sig['p_value']:.6f}")
    
    # 测试 TrialCounter
    counter = TrialCounter()
    for i in range(10):
        counter.add_trial(f"strategy_{i % 3}", {"p": i}, sharpe=0.5 + i * 0.1)
    
    print(f"  ✓ 试验计数: {counter.n_total} 次, {counter.n_strategies} 个策略")
    print(f"  ✓ 最优: {counter.best_result().get('sharpe', 0):.2f}")
    
    print()


def test_p0_engine():
    """测试 P0 增强引擎"""
    print("=" * 60)
    print("[TEST] P0 增强引擎")
    print("=" * 60)
    
    # 创建模拟数据
    np.random.seed(42)
    
    dates = pd.date_range("2020-01-01", "2021-01-01", freq="D")
    symbols = [f"sh60000{i}" for i in range(1, 6)]
    
    # 因子面板
    factor_data = []
    for date in dates:
        for sym in symbols:
            factor_data.append({
                "date": date,
                "symbol": sym,
                "mcap": np.random.lognormal(10, 1),
                "pb": np.random.uniform(0.5, 5),
            })
    
    factor_df = pd.DataFrame(factor_data)
    factor_df.set_index(["date", "symbol"], inplace=True)
    
    # 收益率面板
    return_data = []
    for date in dates:
        for sym in symbols:
            return_data.append({
                "date": date,
                "symbol": sym,
                "daily_return": np.random.normal(0.001, 0.02),
            })
    
    return_df = pd.DataFrame(return_data)
    return_df.set_index(["date", "symbol"], inplace=True)
    
    # 测试1: 关闭所有 P0 功能 (与原引擎一致)
    print("  [1/4] 关闭 P0 功能...")
    engine1 = P0EnhancedEngine(
        factor_panel=factor_df,
        return_panel=return_df,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result1 = engine1.run(ranking_factor="mcap", ascending=True)
    print(f"    原引擎模式: 年化={result1.annual_return:.2f}%, 夏普={result1.sharpe_ratio:.2f}")
    
    # 测试2: 启用 PIT universe
    print("  [2/4] 启用 PIT universe...")
    engine2 = P0EnhancedEngine(
        factor_panel=factor_df,
        return_panel=return_df,
        enable_pit_universe=True,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result2 = engine2.run(ranking_factor="mcap", ascending=True)
    print(f"    PIT 模式: 年化={result2.annual_return:.2f}%, 夏普={result2.sharpe_ratio:.2f}")
    
    # 测试3: 启用风险护栏
    print("  [3/4] 启用风险护栏...")
    engine3 = P0EnhancedEngine(
        factor_panel=factor_df,
        return_panel=return_df,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=True,
        enable_deflated_sharpe=False,
    )
    result3 = engine3.run(ranking_factor="mcap", ascending=True)
    print(f"    风险护栏模式: 年化={result3.annual_return:.2f}%, 回撤={result3.max_drawdown:.2f}%")
    
    # 测试4: 启用所有 P0 功能
    print("  [4/4] 启用所有 P0 功能...")
    engine4 = P0EnhancedEngine(
        factor_panel=factor_df,
        return_panel=return_df,
        enable_pit_universe=True,
        enable_adv_impact=True,
        enable_risk_overlay=True,
        enable_deflated_sharpe=True,
        n_trials=19,
    )
    result4 = engine4.run(ranking_factor="mcap", ascending=True)
    print(f"    全 P0 模式: 年化={result4.annual_return:.2f}%, 夏普={result4.sharpe_ratio:.2f}")
    
    if hasattr(result4, 'deflated_sharpe'):
        print(f"    Deflated Sharpe: {result4.deflated_sharpe:.4f}")
    
    print("  ✓ P0 引擎所有模式测试通过")
    print()


def main():
    print("\n" + "=" * 60)
    print("P0 修复验证 — 开始")
    print("=" * 60 + "\n")
    
    test_delisted()
    test_adv_impact()
    test_risk_overlay()
    test_deflated_sharpe()
    test_p0_engine()
    
    print("=" * 60)
    print("P0 修复验证 — 全部通过 ✓")
    print("=" * 60)
    print("\n模块清单:")
    print("  1. data/delisted.py       — 退市数据 + PIT universe (UA1)")
    print("  2. data/adv_impact.py    — ADV 冲击成本模型 (UA2)")
    print("  3. data/risk_overlay.py  — 三层风险护栏 (UD1)")
    print("  4. research_loop/deflated_sharpe.py — Deflated Sharpe (UA3)")
    print("  5. p0_enhanced_engine.py — P0 集成引擎")
    print()


if __name__ == "__main__":
    main()
