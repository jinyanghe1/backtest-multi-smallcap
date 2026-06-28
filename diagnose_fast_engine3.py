#!/usr/bin/env python3
"""
Quick diagnostic: Find first divergence between original and fast engine
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

print(f"Orig rebalance_dates: {len(orig_engine.rebalance_dates)}")
print(f"Fast rebalance_dates: {len(fast_engine.rebalance_dates)}")
print(f"Dates same: {list(orig_engine.rebalance_dates) == list(fast_engine.rebalance_dates)}")

# Compare first few rebalance dates
for i in range(min(5, len(orig_engine.rebalance_dates))):
    o = orig_engine.rebalance_dates[i]
    f = fast_engine.rebalance_dates[i]
    print(f"  {i}: orig={o}, fast={f}, same={o==f}")

# Compare first rebalance in detail
rebal_date = orig_engine.rebalance_dates[0]
next_date = orig_engine.rebalance_dates[1]
print(f"\n=== First Rebalance: {rebal_date} -> {next_date} ===")

orig_snapshot = orig_engine._get_factor_snapshot(rebal_date)
fast_idx = fast_engine._precompute.date_to_idx[rebal_date]
fast_snapshot = fast_engine._get_factor_snapshot_fast(fast_idx)

print(f"Orig snapshot: {len(orig_snapshot)} stocks, cols={list(orig_snapshot.columns)[:5]}")
print(f"Fast snapshot: {len(fast_snapshot)} stocks, cols={list(fast_snapshot.columns)[:5]}")

# Check if same stocks
orig_stocks = set(orig_snapshot.index)
fast_stocks = set(fast_snapshot.index)
print(f"Orig stocks in fast: {len(orig_stocks & fast_stocks)}/{len(orig_stocks)}")
print(f"Missing in fast: {len(orig_stocks - fast_stocks)}")

# Compare selected
orig_selected = universe_filter(orig_snapshot, orig_engine.dates, 0)
fast_selected = universe_filter(fast_snapshot, fast_engine.dates, 0)
print(f"\nOrig selected: {len(orig_selected)} stocks")
print(f"Fast selected: {len(fast_selected)} stocks")
print(f"Same selected: {set(orig_selected) == set(fast_selected)}")

# Compare picked (ranked)
orig_valid = orig_snapshot.loc[orig_selected]['mcap'].dropna()
orig_ranked = orig_valid.nsmallest(30)
orig_picked = orig_ranked.index[:30].tolist()

fast_valid = fast_snapshot.loc[fast_selected]['mcap'].dropna()
fast_ranked = fast_valid.nsmallest(30)
fast_picked = fast_ranked.index[:30].tolist()

print(f"\nOrig picked: {len(orig_picked)}")
print(f"Fast picked: {len(fast_picked)}")
print(f"Same picked: {set(orig_picked) == set(fast_picked)}")
if set(orig_picked) != set(fast_picked):
    print(f"  orig - fast: {set(orig_picked) - set(fast_picked)}")
    print(f"  fast - orig: {set(fast_picked) - set(orig_picked)}")

# Compare period returns (the key difference!)
period_dates = [d for d in orig_engine.dates if d > rebal_date and d <= next_date]
print(f"\nPeriod dates: {len(period_dates)} days")
print(f"First 3: {period_dates[:3]}")

for d in period_dates[:3]:
    orig_r = orig_engine._get_daily_return(d, orig_picked)
    fast_r_idx = fast_engine._precompute.date_to_idx[d]
    fast_r = fast_engine._get_daily_return_fast(fast_r_idx, orig_picked)
    
    print(f"\n  {d}:")
    print(f"    orig: {orig_r.values[:3]}")
    print(f"    fast: {fast_r.values[:3]}")
    print(f"    same: {np.allclose(orig_r.values, fast_r.values, atol=1e-10)}")
    if not np.allclose(orig_r.values, fast_r.values, atol=1e-10):
        diff = np.abs(orig_r.values - fast_r.values)
        print(f"    max_diff: {diff.max()}")
        # Find which stock has the diff
        for i, s in enumerate(orig_picked[:3]):
            if not np.isclose(orig_r.values[i], fast_r.values[i], atol=1e-10):
                print(f"    {s}: orig={orig_r.values[i]}, fast={fast_r.values[i]}")

# Compare full period returns
orig_period = orig_engine._get_period_returns(rebal_date, next_date, orig_picked)
fast_period = fast_engine._get_period_returns_fast(
    fast_engine._precompute.date_to_idx[rebal_date],
    fast_engine._precompute.date_to_idx[next_date],
    orig_picked
)

print(f"\n=== Period Returns Matrix ===")
print(f"Orig period: {orig_period.shape}")
print(f"Fast period: {fast_period.shape}")
if orig_period.shape == fast_period.shape:
    diff = (orig_period.fillna(0) - fast_period.fillna(0)).abs()
    print(f"Max diff: {diff.max().max()}")
    max_diff_loc = np.unravel_index(np.argmax(diff.values), diff.shape)
    print(f"Max diff at: row={max_diff_loc[0]}, col={diff.columns[max_diff_loc[1]]}")
    print(f"Orig value: {orig_period.iloc[max_diff_loc].values[0]}")
    print(f"Fast value: {fast_period.iloc[max_diff_loc].values[0]}")
else:
    print("Shape mismatch!")
    # Check dates
    orig_d = set(orig_period.index)
    fast_d = set(fast_period.index)
    print(f"Date overlap: {len(orig_d & fast_d)}")
    print(f"Only in orig: {sorted(orig_d - fast_d)[:5]}")
    print(f"Only in fast: {sorted(fast_d - orig_d)[:5]}")
