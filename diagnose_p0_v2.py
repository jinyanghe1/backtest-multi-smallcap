#!/usr/bin/env python3
"""P0 诊断 v2 — 对比: 原引擎 vs P0 v1(后处理) vs P0 v2(真实路径)

用法:
  cd thinking_and_learning_with_AI
  PYTHONPATH=/path/to/project python3 tools/backtest_mvp/diagnose_p0_v2.py

输出:
  - 三列对比: 原引擎 | P0 v1(后处理) | P0 v2(逐日内嵌)
  - 量化"后处理" vs "真实路径"的差异
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.backtest_mvp.engine import CrossSectionalEngine, BacktestResult
from tools.backtest_mvp.factors import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.strategies import ALL_STRATEGIES
from tools.backtest_mvp.strategies_v2 import NEW_STRATEGIES, TEMPLATE_STRATEGIES
from tools.backtest_mvp.data import DATA_DIR
from tools.backtest_mvp.p0_engine_v2 import P0EngineV2
from tools.backtest_mvp.research_loop.deflated_sharpe import compute_deflated_sharpe, deflated_sharpe_significance
from tools.backtest_mvp.data.delisted import DelistManager
from tools.backtest_mvp.data.adv_impact import ADVImpactModel
from tools.backtest_mvp.data.risk_overlay import RiskOverlay, RiskOverlayConfig


def load_p0_data():
    """加载 P0 所需数据"""
    from pathlib import Path
    cache_dir = Path(__file__).resolve().parent / 'data_cache'
    
    # 上市日期
    listing_path = cache_dir / 'listing_dates.csv'
    listing_dates = pd.read_csv(listing_path, encoding='utf-8-sig') if listing_path.exists() else None
    
    # ADV 数据
    adv_path = cache_dir / 'adv_panel.parquet'
    adv_data = pd.read_parquet(adv_path) if adv_path.exists() else None
    
    # 退市数据
    delist_mgr = DelistManager()
    delist_mgr.fetch_all(force=False)
    
    return listing_dates, adv_data, delist_mgr


def run_original(s, factor_panel, return_panel) -> BacktestResult:
    """原引擎"""
    engine = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=s.get("n_stocks", 30),
        rebalance_freq='M', commission=0.00125, slippage=0.002,
        price_limit_stocks=True,
    )
    return engine.run(
        universe_filter=s.get("universe_filter"),
        ranking_factor=s.get("ranking_factor", "mcap"),
        ascending=s.get("ascending", True),
        composite_factors=s.get("composite_factors"),
        stop_loss=s.get("stop_loss"), trailing_stop=s.get("trailing_stop"),
        ranking_fn=s.get("ranking_fn"), factor_weights=s.get("factor_weights"),
    )


def run_p0_v2(s, factor_panel, return_panel, listing_dates, adv_data, delist_mgr, n_trials=19) -> BacktestResult:
    """P0 v2: 逐日内嵌引擎"""
    engine = P0EngineV2(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=s.get("n_stocks", 30),
        rebalance_freq='M', commission=0.00125, slippage=0.002,
        price_limit_stocks=True,
        enable_pit_universe=True, enable_adv_impact=True,
        enable_risk_overlay=True, enable_deflated_sharpe=True,
        listing_dates=listing_dates, delist_manager=delist_mgr,
        adv_data=adv_data, adv_model=ADVImpactModel(),
        risk_overlay_config=RiskOverlayConfig(),
        n_trials=n_trials,
    )
    return engine.run(
        universe_filter=s.get("universe_filter"),
        ranking_factor=s.get("ranking_factor", "mcap"),
        ascending=s.get("ascending", True),
        composite_factors=s.get("composite_factors"),
        stop_loss=s.get("stop_loss"), trailing_stop=s.get("trailing_stop"),
        ranking_fn=s.get("ranking_fn"), factor_weights=s.get("factor_weights"),
    )


def diagnose():
    print("=" * 120)
    print("  P0 诊断 v2: 原引擎 vs P0 v1(后处理) vs P0 v2(逐日内嵌)")
    print("=" * 120)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    
    # 获取所有股票代码 (排除 adv_panel.parquet 等非股票文件)
    cache_dir = DATA_DIR
    all_files = sorted(cache_dir.glob("*.parquet"))
    stock_files = [f for f in all_files if f.name != "adv_panel.parquet"]
    symbols = [f.stem for f in stock_files]
    print(f"  发现 {len(symbols)} 只股票价格数据")
    
    data = load_price_data(str(cache_dir), symbols=symbols)
    if len(data) == 0:
        print("⚠️ 未找到数据! 请先运行: python tools/backtest_mvp/run.py download")
        return
    print(f"  加载 {data['symbol'].nunique()} 只股票, {len(data)} 行")
    
    mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
    factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)
    
    # 2. 加载 P0 数据
    print("\n[2/5] 加载 P0 数据...")
    listing_dates, adv_data, delist_mgr = load_p0_data()
    print(f"  上市日期: {len(listing_dates) if listing_dates is not None else 0} 只")
    print(f"  ADV 数据: {len(adv_data) if adv_data is not None else 0} 行")
    print(f"  退市数据: {len(delist_mgr.df) if delist_mgr and delist_mgr.df is not None else 0} 只")
    
    # 3. 策略列表
    print("\n[3/5] 准备策略...")
    all_strategies = ALL_STRATEGIES + NEW_STRATEGIES + TEMPLATE_STRATEGIES
    print(f"  共 {len(all_strategies)} 个策略")
    n_trials = len(all_strategies)
    
    # 4. 运行对比
    print("\n[4/5] 运行对比回测...")
    print(f"  试验次数: {n_trials}")
    
    results = []
    for i, s in enumerate(all_strategies):
        print(f"\n  [{i+1}/{len(all_strategies)}] {s['name']}")
        
        try:
            # 两个引擎：原 vs v2
            orig = run_original(s, factor_panel, return_panel)
            p2 = run_p0_v2(s, factor_panel, return_panel, listing_dates, adv_data, delist_mgr, n_trials)
            
            n_months = len(p2.monthly_returns) if hasattr(p2, 'monthly_returns') and p2.monthly_returns is not None else 92
            
            row = {
                "strategy": s['name'],
                # 原引擎
                "orig_annual": orig.annual_return, "orig_sharpe": orig.sharpe_ratio,
                "orig_dd": orig.max_drawdown, "orig_calmar": orig.calmar_ratio,
                # P0 v2
                "v2_annual": p2.annual_return, "v2_sharpe": p2.sharpe_ratio,
                "v2_dd": p2.max_drawdown, "v2_calmar": p2.calmar_ratio,
                "v2_dsr": getattr(p2, 'deflated_sharpe', 0),
                # 部署判断
                "v2_deployable": p2.sharpe_ratio >= 1.0 and p2.max_drawdown > -30 and getattr(p2, 'deflated_sharpe', 0) > 0,
            }
            results.append(row)
            
            print(f"    原: 年化{orig.annual_return:>+6.1f}% 夏普{orig.sharpe_ratio:>5.2f} 回撤{orig.max_drawdown:>6.1f}%")
            print(f"    v2: 年化{p2.annual_return:>+6.1f}% 夏普{p2.sharpe_ratio:>5.2f} 回撤{p2.max_drawdown:>6.1f}% DSR{getattr(p2, 'deflated_sharpe', 0):>5.2f}")
            
        except Exception as e:
            print(f"    ⚠️ 错误: {str(e)[:80]}")
            import traceback
            traceback.print_exc()
            results.append({"strategy": s['name'], "v2_deployable": False})
    
    # 5. 汇总输出
    print("\n" + "=" * 120)
    print("  诊断结果汇总")
    print("=" * 120)
    
    df = pd.DataFrame(results)
    
    # 可部署统计
    deployable = df[df.get("v2_deployable", False) == True]
    print(f"\n  📊 v2 可部署策略: {len(deployable)}/{len(df)}")
    
    # 保存报告
    report_path = Path(__file__).parent / "diagnose_p0_v2_report.csv"
    df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\n  📄 详细报告: {report_path}")
    print("=" * 120)
    
    return df


if __name__ == "__main__":
    diagnose()
