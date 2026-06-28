#!/usr/bin/env python3
"""
Detailed diagnostic: compare original vs fast engine step by step
"""
import sys
sys.path.insert(0, '/Users/hejinyang/thinking_and_learning_with_AI')

import pandas as pd
import numpy as np
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

print(f"Rebalance dates: {len(orig_engine.rebalance_dates)}")
print(f"Fast rebalance dates: {len(fast_engine.rebalance_dates)}")

# Run both engines
orig_result = orig_engine.run(universe_filter=universe_filter, ranking_factor='mcap', ascending=True)
fast_result = fast_engine.run(universe_filter=universe_filter, ranking_factor='mcap', ascending=True)

print(f"\n=== Positions Log Comparison ===")
print(f"Orig shape: {orig_result.positions_log.shape}")
print(f"Fast shape: {fast_result.positions_log.shape}")

orig_cols = set(orig_result.positions_log.columns)
fast_cols = set(fast_result.positions_log.columns)
print(f"Missing in fast: {len(orig_cols - fast_cols)}")
print(f"Missing in orig: {len(fast_cols - orig_cols)}")

if orig_cols != fast_cols:
    print(f"Only in orig: {sorted(orig_cols - fast_cols)[:5]}")
    print(f"Only in fast: {sorted(fast_cols - orig_cols)[:5]}")

# Compare stock overlap
orig_rows = set(orig_result.positions_log.index)
fast_rows = set(fast_result.positions_log.index)
print(f"\nStock overlap: {len(orig_rows & fast_rows)} / {len(orig_rows)} (orig) / {len(fast_rows)} (fast)")
print(f"Only in orig: {len(orig_rows - fast_rows)}")
print(f"Only in fast: {len(fast_rows - orig_rows)}")

# Compare equity curves
print(f"\n=== Equity Curve Comparison ===")
orig_ec = orig_result.equity_curve
fast_ec = fast_result.equity_curve
print(f"Orig dates: {len(orig_ec)}")
print(f"Fast dates: {len(fast_ec)}")

merged = pd.merge(
    orig_ec.rename('orig'), fast_ec.rename('fast'),
    left_index=True, right_index=True, how='outer'
)
print(f"Merged dates: {len(merged)}")
print(f"Same dates: {merged['orig'].notna().sum()} (both have data)")
print(f"Only orig: {merged['orig'].notna().sum() - merged['fast'].notna().sum()}")
print(f"Only fast: {merged['fast'].notna().sum() - merged['orig'].notna().sum()}")

# Find first difference
both = merged.dropna()
if len(both) > 0:
    diff = (both['orig'] - both['fast']).abs()
    first_diff_idx = diff[diff > 1e-6].index[0] if (diff > 1e-6).any() else None
    if first_diff_idx:
        print(f"First difference at: {first_diff_idx}")
        print(f"  orig: {both.loc[first_diff_idx, 'orig']:.6f}")
        print(f"  fast: {both.loc[first_diff_idx, 'fast']:.6f}")
        
        # Find which rebalance period this date belongs to
        for i in range(len(orig_engine.rebalance_dates) - 1):
            if orig_engine.rebalance_dates[i] <= first_diff_idx < orig_engine.rebalance_dates[i + 1]:
                print(f"  In rebalance period: {orig_engine.rebalance_dates[i]} -> {orig_engine.rebalance_dates[i+1]}")
                break

# Compare monthly returns
print(f"\n=== Monthly Returns Comparison ===")
orig_mr = orig_result.monthly_returns
fast_mr = fast_result.monthly_returns
print(f"Orig monthly: {len(orig_mr)}")
print(f"Fast monthly: {len(fast_mr)}")
if len(orig_mr) == len(fast_mr):
    diff = (orig_mr - fast_mr).abs()
    print(f"Max diff: {diff.max():.6f}")
    bad = diff[diff > 1e-6]
    if len(bad) > 0:
        print(f"First bad month: idx={bad.index[0]}, orig={orig_mr.iloc[bad.index[0]]:.6f}, fast={fast_mr.iloc[bad.index[0]]:.6f}")

# Compare IC series
print(f"\n=== IC Series Comparison ===")
orig_ic = orig_result.ic_series
fast_ic = fast_result.ic_series
print(f"Orig IC: {len(orig_ic)}")
print(f"Fast IC: {len(fast_ic)}")
if len(orig_ic) > 0 and len(fast_ic) > 0:
    merged_ic = pd.merge(
        orig_ic.rename('orig'), fast_ic.rename('fast'),
        left_index=True, right_index=True, how='outer'
    )
    both_ic = merged_ic.dropna()
    if len(both_ic) > 0:
        diff = (both_ic['orig'] - both_ic['fast']).abs()
        print(f"Max IC diff: {diff.max():.6f}")

# Compare first few positions
print(f"\n=== First 3 Rebalance Positions ===")
for i, col in enumerate(orig_result.positions_log.columns[:3]):
    orig_pos = orig_result.positions_log[col]
    orig_picked = orig_pos[orig_pos > 0].index.tolist()
    if col in fast_result.positions_log.columns:
        fast_pos = fast_result.positions_log[col]
        fast_picked = fast_pos[fast_pos > 0].index.tolist()
        same = set(orig_picked) == set(fast_picked)
        print(f"\n{col}: same={same}")
        if not same:
            print(f"  orig: {orig_picked}")
            print(f"  fast: {fast_picked}")
            print(f"  orig - fast: {set(orig_picked) - set(fast_picked)}")
            print(f"  fast - orig: {set(fast_picked) - set(orig_picked)}")
