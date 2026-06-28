#!/usr/bin/env python3
"""
Deep dive: Compare period dates and positions_log between orig and fast
"""
import sys
sys.path.insert(0, '/Users/hejinyang/thinking_and_learning_with_AI')

import pandas as pd
import numpy as np
from tools.backtest_mvp.factors.legacy import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.data import DATA_DIR
from tools.backtest_mvp.engine import CrossSectionalEngine
from tools.backtest_mvp.engine_fast import CrossSectionalEngineFast

print("Loading data...")
data = load_price_data(str(DATA_DIR))
symbols = data['symbol'].unique()[:500]
data = data[data['symbol'].isin(symbols)]
mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)

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

universe_filter = lambda snapshot, dates, i: list(snapshot[snapshot['mcap'] < 50].index) if 'mcap' in snapshot.columns else list(snapshot.index)

# Run both
orig_result = orig_engine.run(universe_filter=universe_filter, ranking_factor='mcap', ascending=True)
fast_result = fast_engine.run(universe_filter=universe_filter, ranking_factor='mcap', ascending=True)

print(f"\n=== Positions Log Comparison ===")
print(f"Orig: {orig_result.positions_log.shape}")
print(f"Fast: {fast_result.positions_log.shape}")

orig_dates = set(orig_result.positions_log.columns)
fast_dates = set(fast_result.positions_log.columns)

print(f"\nOnly in orig ({len(orig_dates - fast_dates)}):")
for d in sorted(orig_dates - fast_dates)[:10]:
    print(f"  {d}")

print(f"\nOnly in fast ({len(fast_dates - orig_dates)}):")
for d in sorted(fast_dates - orig_dates)[:10]:
    print(f"  {d}")

# Compare stock rows
orig_rows = set(orig_result.positions_log.index)
fast_rows = set(fast_result.positions_log.index)
print(f"\nStock overlap: {len(orig_rows & fast_rows)}")
print(f"Only in orig: {len(orig_rows - fast_rows)}")
print(f"Only in fast: {len(fast_rows - orig_rows)}")

if len(orig_rows - fast_rows) > 0:
    print(f"  Sample: {list(orig_rows - fast_rows)[:5]}")

# Find first rebalance date where columns differ
for i in range(min(len(orig_engine.rebalance_dates), 5)):
    d = orig_engine.rebalance_dates[i]
    in_orig = d in orig_result.positions_log.columns
    in_fast = d in fast_result.positions_log.columns
    print(f"\nRebalance {i} ({d}): in_orig={in_orig}, in_fast={in_fast}")
    
    if in_orig and in_fast:
        orig_pos = orig_result.positions_log[d]
        fast_pos = fast_result.positions_log[d]
        orig_picked = orig_pos[orig_pos > 0].index.tolist()
        fast_picked = fast_pos[fast_pos > 0].index.tolist()
        print(f"  orig_picked: {len(orig_picked)} stocks")
        print(f"  fast_picked: {len(fast_picked)} stocks")
        if set(orig_picked) != set(fast_picked):
            print(f"  MISMATCH: {set(orig_picked) ^ set(fast_picked)}")

# Compare equity curves - find first divergence
print(f"\n\n=== Equity Curve First Divergence ===")
orig_ec = orig_result.equity_curve
fast_ec = fast_result.equity_curve

merged = pd.merge(orig_ec.rename('orig'), fast_ec.rename('fast'), left_index=True, right_index=True, how='outer')
print(f"Merged: {len(merged)} dates")

both = merged.dropna()
print(f"Both have data: {len(both)} dates")

if len(both) > 0:
    diff = (both['orig'] - both['fast']).abs()
    bad = diff[diff > 1e-6]
    if len(bad) > 0:
        first_bad = bad.index[0]
        print(f"First divergence: {first_bad}")
        print(f"  orig: {both.loc[first_bad, 'orig']:.6f}")
        print(f"  fast: {both.loc[first_bad, 'fast']:.6f}")
        
        # Find which rebalance period this belongs to
        for i in range(len(orig_engine.rebalance_dates) - 1):
            start = orig_engine.rebalance_dates[i]
            end = orig_engine.rebalance_dates[i + 1]
            if start <= first_bad < end:
                print(f"  In rebalance period: {start} -> {end}")
                
                # Check if the period dates are the same
                orig_period_dates = [d for d in orig_engine.dates if d > start and d <= end]
                fast_start_idx = fast_engine._precompute.date_to_idx[start]
                fast_end_idx = fast_engine._precompute.date_to_idx[end]
                fast_period_dates = fast_engine._precompute.dates[fast_start_idx + 1:fast_end_idx + 1]
                
                print(f"  orig_period_dates: {len(orig_period_dates)} days")
                print(f"  fast_period_dates: {len(fast_period_dates)} days")
                print(f"  orig first: {orig_period_dates[0] if orig_period_dates else 'None'}")
                print(f"  fast first: {fast_period_dates[0] if len(fast_period_dates) > 0 else 'None'}")
                print(f"  orig last: {orig_period_dates[-1] if orig_period_dates else 'None'}")
                print(f"  fast last: {fast_period_dates[-1] if len(fast_period_dates) > 0 else 'None'}")
                break
