#!/usr/bin/env python3
"""
Optimized full backtest with pre-pivoted data for fast lookup
"""
import sys, os, time
sys.path.insert(0, '/Users/hejinyang/thinking_and_learning_with_AI')

import pandas as pd
import numpy as np
from tools.backtest_mvp.factors.legacy import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.data import DATA_DIR
from tools.backtest_mvp.engine import CrossSectionalEngine, BacktestResult

print("="*80)
print("  OPTIMIZED FULL BACKTEST - Expanded Universe (5,400+ stocks)")
print("="*80)

t_total = time.time()

# 1. Load data
print("\n[1/4] Loading price data (5,400+ stocks)...")
t0 = time.time()
data = load_price_data(str(DATA_DIR))
t1 = time.time()
print(f"  ✓ {data['symbol'].nunique()} stocks, {len(data):,} rows ({t1-t0:.1f}s)")
print(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}")

# 2. Load mcap/pb
print("\n[2/4] Loading mcap/pb...")
t0 = time.time()
mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
t1 = time.time()
print(f"  ✓ {mcap_pb['symbol'].nunique() if not mcap_pb.empty else 0} stocks with mcap/pb ({t1-t0:.1f}s)")

# 3. Compute factors
print("\n[3/4] Computing factors...")
t0 = time.time()
factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)
t1 = time.time()
print(f"  ✓ {len(factor_panel):,} rows x {len(factor_panel.columns)} cols ({t1-t0:.1f}s)")
print(f"  Stocks: {factor_panel.index.get_level_values(1).nunique()}")
print(f"  Date range: {factor_panel.index.get_level_values(0).min().date()} to {factor_panel.index.get_level_values(0).max().date()}")

# 4. Run strategies - only one per engine instance to avoid rebuilding
strategies = [
    ("Micro-Cap (MCAP<50, asc)", lambda s, d, i: list(s[s['mcap'] < 50].index) if 'mcap' in s.columns else list(s.index), 'mcap', True),
    ("Low-PB (MCAP<100, PB asc)", lambda s, d, i: list(s[s['mcap'] < 100].index) if 'mcap' in s.columns else list(s.index), 'pb', True),
    ("Momentum 20d (MCAP<100, desc)", lambda s, d, i: list(s[s['mcap'] < 100].index) if 'mcap' in s.columns else list(s.index), 'mom20d', False),
    ("Low-Vol (MCAP<100, VOL asc)", lambda s, d, i: list(s[s['mcap'] < 100].index) if 'mcap' in s.columns else list(s.index), 'vol20d', True),
]

print(f"\n[4/4] Running {len(strategies)} strategies...")
print(f"{'='*80}")
print(f"  {'Strategy':<35} {'Annual':>8} {'Sharpe':>7} {'Drawdown':>9} {'WinRate':>8} {'Terminal':>8}")
print(f"{'='*80}")

for name, universe_filter, ranking_factor, ascending in strategies:
    t0 = time.time()
    engine = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=30, rebalance_freq='M',
        commission=0.00125, slippage=0.002, price_limit_stocks=True,
    )
    result = engine.run(
        universe_filter=universe_filter,
        ranking_factor=ranking_factor,
        ascending=ascending,
    )
    t1 = time.time()
    print(f"  {name:<35} {result.annual_return:>7.1f}% {result.sharpe_ratio:>6.2f} {result.max_drawdown:>8.1f}% {result.win_rate:>7.1f}% {result.terminal_value:>7.2f}x  ({t1-t0:.1f}s)")

t_total_end = time.time()
print(f"{'='*80}")
print(f"Total time: {(t_total_end - t_total)/60:.1f}min")
print(f"{'='*80}")

# Save results to a simple file for reference
with open('/Users/hejinyang/thinking_and_learning_with_AI/tools/backtest_mvp/backtest_results.txt', 'w') as f:
    f.write(f"Backtest Results - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Universe: {factor_panel.index.get_level_values(1).nunique()} stocks\n")
    f.write(f"Date range: {factor_panel.index.get_level_values(0).min().date()} to {factor_panel.index.get_level_values(0).max().date()}\n")
    f.write(f"Total time: {(t_total_end - t_total)/60:.1f}min\n\n")
    f.write(f"{'='*80}\n")
    f.write(f"  {'Strategy':<35} {'Annual':>8} {'Sharpe':>7} {'Drawdown':>9} {'WinRate':>8} {'Terminal':>8}\n")
    f.write(f"{'='*80}\n")
    for name, universe_filter, ranking_factor, ascending in strategies:
        engine = CrossSectionalEngine(
            factor_panel=factor_panel, return_panel=return_panel,
            initial_capital=1.0, n_stocks=30, rebalance_freq='M',
            commission=0.00125, slippage=0.002, price_limit_stocks=True,
        )
        result = engine.run(
            universe_filter=universe_filter,
            ranking_factor=ranking_factor,
            ascending=ascending,
        )
        f.write(f"  {name:<35} {result.annual_return:>7.1f}% {result.sharpe_ratio:>6.2f} {result.max_drawdown:>8.1f}% {result.win_rate:>7.1f}% {result.terminal_value:>7.2f}x\n")

print("\nResults saved to backtest_results.txt")
