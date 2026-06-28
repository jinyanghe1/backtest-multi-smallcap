#!/usr/bin/env python3
"""
Find first divergence across ALL rebalance periods
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

print(f"Rebalance periods: {len(orig_engine.rebalance_dates) - 1}")

for i in range(len(orig_engine.rebalance_dates) - 1):
    rebal_date = orig_engine.rebalance_dates[i]
    next_date = orig_engine.rebalance_dates[i + 1]
    
    orig_snapshot = orig_engine._get_factor_snapshot(rebal_date)
    fast_idx = fast_engine._precompute.date_to_idx[rebal_date]
    fast_snapshot = fast_engine._get_factor_snapshot_fast(fast_idx)
    
    orig_selected = universe_filter(orig_snapshot, orig_engine.dates, i)
    fast_selected = universe_filter(fast_snapshot, fast_engine.dates, i)
    
    if set(orig_selected) != set(fast_selected):
        print(f"\nDIVERGENCE at rebalance {i} ({rebal_date}):")
        print(f"  Selected mismatch!")
        print(f"  Orig: {len(orig_selected)} stocks")
        print(f"  Fast: {len(fast_selected)} stocks")
        print(f"  orig - fast: {set(orig_selected) - set(fast_selected)}")
        print(f"  fast - orig: {set(fast_selected) - set(orig_selected)}")
        break
    
    orig_valid = orig_snapshot.loc[orig_selected]['mcap'].dropna() if 'mcap' in orig_snapshot.columns else pd.Series()
    fast_valid = fast_snapshot.loc[fast_selected]['mcap'].dropna() if 'mcap' in fast_snapshot.columns else pd.Series()
    
    orig_ranked = orig_valid.nsmallest(30)
    fast_ranked = fast_valid.nsmallest(30)
    orig_picked = orig_ranked.index[:30].tolist()
    fast_picked = fast_ranked.index[:30].tolist()
    
    if set(orig_picked) != set(fast_picked):
        print(f"\nDIVERGENCE at rebalance {i} ({rebal_date}):")
        print(f"  Picked mismatch!")
        print(f"  orig - fast: {set(orig_picked) - set(fast_picked)}")
        print(f"  fast - orig: {set(fast_picked) - set(orig_picked)}")
        break
    
    # Check period returns
    period_dates = [d for d in orig_engine.dates if d > rebal_date and d <= next_date]
    for d in period_dates:
        orig_r = orig_engine._get_daily_return(d, orig_picked)
        fast_r_idx = fast_engine._precompute.date_to_idx[d]
        fast_r = fast_engine._get_daily_return_fast(fast_r_idx, orig_picked)
        
        if not np.allclose(orig_r.values, fast_r.values, atol=1e-10):
            print(f"\nDIVERGENCE at rebalance {i} ({rebal_date}), date {d}:")
            print(f"  Returns mismatch!")
            diff = np.abs(orig_r.values - fast_r.values)
            max_idx = np.argmax(diff)
            print(f"  Stock: {orig_picked[max_idx]}")
            print(f"  orig: {orig_r.values[max_idx]:.10f}")
            print(f"  fast: {fast_r.values[max_idx]:.10f}")
            break
    else:
        continue
    break
else:
    print("\nNo divergence in rebalance logic found!")
    print("The issue must be in the IC computation or other non-rebalance parts")
