#!/usr/bin/env python3
"""
Diagnostic script to compare original and fast engine step by step
"""
import sys
sys.path.insert(0, '/Users/hejinyang/thinking_and_learning_with_AI')

import pandas as pd
from tools.backtest_mvp.factors.legacy import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.data import DATA_DIR
from tools.backtest_mvp.engine import CrossSectionalEngine
from tools.backtest_mvp.engine_fast import CrossSectionalEngineFast

print("Loading data (100 stocks)...")
data = load_price_data(str(DATA_DIR))
symbols = data['symbol'].unique()[:100]
data = data[data['symbol'].isin(symbols)]
mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)

print(f"Factor panel: {len(factor_panel):,} rows, {factor_panel.index.get_level_values(1).nunique()} stocks")
print(f"Return panel: {len(return_panel):,} rows, {return_panel.index.get_level_values(1).nunique()} stocks")
print(f"Factor columns: {list(factor_panel.columns)}")

# Create engines
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

print(f"\nOriginal rebalance dates: {len(orig_engine.rebalance_dates)}")
print(f"Fast rebalance dates: {len(fast_engine.rebalance_dates)}")
print(f"Dates same: {list(orig_engine.rebalance_dates) == list(fast_engine.rebalance_dates)}")

# Compare first 5 rebalance dates
for i in range(min(5, len(orig_engine.rebalance_dates))):
    orig_date = orig_engine.rebalance_dates[i]
    fast_date = fast_engine.rebalance_dates[i]
    print(f"\nRebalance {i}: orig={orig_date}, fast={fast_date}, same={orig_date == fast_date}")

    # Compare snapshots
    orig_snapshot = orig_engine._get_factor_snapshot(orig_date)
    fast_idx = fast_engine._precompute.date_to_idx[fast_date]
    fast_snapshot = fast_engine._get_factor_snapshot_fast(fast_idx)

    print(f"  Orig snapshot: {len(orig_snapshot)} stocks, cols={list(orig_snapshot.columns)[:5]}...")
    print(f"  Fast snapshot: {len(fast_snapshot)} stocks, cols={list(fast_snapshot.columns)[:5]}...")

    # Check if orig stocks are a subset of fast stocks
    orig_stocks = set(orig_snapshot.index)
    fast_stocks = set(fast_snapshot.index)
    missing_in_fast = orig_stocks - fast_stocks
    extra_in_fast = fast_stocks - orig_stocks
    print(f"  Missing in fast: {len(missing_in_fast)}, Extra in fast: {len(extra_in_fast)}")

    if len(orig_snapshot) > 0 and len(fast_snapshot) > 0:
        # Compare common stocks
        common = list(orig_stocks & fast_stocks)
        if len(common) > 0:
            orig_vals = orig_snapshot.loc[common, 'mcap'].head(3)
            fast_vals = fast_snapshot.loc[common, 'mcap'].head(3)
            print(f"  mcap comparison (first 3 common):")
            for s in common[:3]:
                o = orig_snapshot.loc[s, 'mcap'] if s in orig_snapshot.index else 'N/A'
                f = fast_snapshot.loc[s, 'mcap'] if s in fast_snapshot.index else 'N/A'
                print(f"    {s}: orig={o}, fast={f}, same={o == f if o != 'N/A' and f != 'N/A' else 'N/A'}")

    # Compare daily returns for next 3 days
    if i < len(orig_engine.rebalance_dates) - 1:
        next_date = orig_engine.rebalance_dates[i + 1]
        period_dates = [d for d in orig_engine.dates if d > orig_date and d <= next_date]
        for d in period_dates[:3]:
            orig_r = orig_engine._get_daily_return(d, list(orig_snapshot.index)[:5])
            fast_r_idx = fast_engine._precompute.date_to_idx[d]
            fast_r = fast_engine._get_daily_return_fast(fast_r_idx, list(orig_snapshot.index)[:5])
            print(f"  Returns on {d}: orig={orig_r.values[:3]}, fast={fast_r.values[:3]}")

print("\n\n=== Checking rebalance_date_indices ===")
for i in range(min(5, len(fast_engine.rebalance_dates))):
    d = fast_engine.rebalance_dates[i]
    idx = fast_engine.rebalance_date_indices[i]
    expected_idx = fast_engine._precompute.dates.index(d) if d in fast_engine._precompute.dates else 'NOT FOUND'
    print(f"  {i}: {d} -> idx={idx}, expected={expected_idx}")

print("\n\n=== Checking positions_log divergence ===")
# Run both engines and compare positions_log
universe_filter = lambda snapshot, dates, i: list(snapshot[snapshot['mcap'] < 50].index) if 'mcap' in snapshot.columns else list(snapshot.index)

orig_result = orig_engine.run(universe_filter=universe_filter, ranking_factor='mcap', ascending=True)
fast_result = fast_engine.run(universe_filter=universe_filter, ranking_factor='mcap', ascending=True)

print(f"Orig positions_log shape: {orig_result.positions_log.shape}")
print(f"Fast positions_log shape: {fast_result.positions_log.shape}")

# Find missing dates
orig_dates = set(orig_result.positions_log.columns)
fast_dates = set(fast_result.positions_log.columns)
print(f"Missing in fast: {sorted(orig_dates - fast_dates)[:5]}")
print(f"Missing in orig: {sorted(fast_dates - orig_dates)[:5]}")
