#!/usr/bin/env python3
"""
Trace divergence between original and fast engine
"""
import sys
sys.path.insert(0, '/Users/hejinyang/thinking_and_learning_with_AI')

import pandas as pd
import numpy as np
from tools.backtest_mvp.factors.legacy import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.data import DATA_DIR
from tools.backtest_mvp.engine import CrossSectionalEngine
from tools.backtest_mvp.engine_fast import CrossSectionalEngineFast

print("Loading data (500 stocks)...")
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

print(f"Rebalance periods: {len(orig_engine.rebalance_dates) - 1}")
print(f"Orig dates: {len(orig_engine.dates)}")
print(f"Fast precompute dates: {len(fast_engine._precompute.dates)}")
print(f"Dates same: {list(orig_engine.dates) == list(fast_engine._precompute.dates)}")

# Compare each rebalance period
for i in range(len(orig_engine.rebalance_dates) - 1):
    rebal_date = orig_engine.rebalance_dates[i]
    next_date = orig_engine.rebalance_dates[i + 1]
    
    orig_snapshot = orig_engine._get_factor_snapshot(rebal_date)
    fast_idx = fast_engine._precompute.date_to_idx[rebal_date]
    fast_snapshot = fast_engine._get_factor_snapshot_fast(fast_idx)
    
    orig_selected = universe_filter(orig_snapshot, orig_engine.dates, i)
    fast_selected = universe_filter(fast_snapshot, fast_engine.dates, i)
    
    orig_picked = None
    fast_picked = None
    
    if set(orig_selected) != set(fast_selected):
        print(f"\n  [{i}] SELECTED MISMATCH at {rebal_date}")
        print(f"    orig: {len(orig_selected)} stocks")
        print(f"    fast: {len(fast_selected)} stocks")
        continue
    
    orig_valid = orig_snapshot.loc[orig_selected]['mcap'].dropna() if 'mcap' in orig_snapshot.columns else pd.Series()
    fast_valid = fast_snapshot.loc[fast_selected]['mcap'].dropna() if 'mcap' in fast_snapshot.columns else pd.Series()
    
    orig_ranked = orig_valid.nsmallest(30)
    fast_ranked = fast_valid.nsmallest(30)
    orig_picked = orig_ranked.index[:30].tolist()
    fast_picked = fast_ranked.index[:30].tolist()
    
    if set(orig_picked) != set(fast_picked):
        print(f"\n  [{i}] PICKED MISMATCH at {rebal_date}")
        print(f"    orig: {orig_picked}")
        print(f"    fast: {fast_picked}")
        print(f"    orig - fast: {set(orig_picked) - set(fast_picked)}")
        print(f"    fast - orig: {set(fast_picked) - set(orig_picked)}")
        
        # Show the mcap values for the differing stocks
        all_diff = list(set(orig_picked) ^ set(fast_picked))[:5]
        for s in all_diff:
            orig_mcap = orig_snapshot.loc[s, 'mcap'] if s in orig_snapshot.index else 'N/A'
            fast_mcap = fast_snapshot.loc[s, 'mcap'] if s in fast_snapshot.index else 'N/A'
            print(f"    {s}: orig_mcap={orig_mcap}, fast_mcap={fast_mcap}")
        break
    
    # Compare period returns for all dates in the period
    period_dates = [d for d in orig_engine.dates if d > rebal_date and d <= next_date]
    for d in period_dates:
        orig_r = orig_engine._get_daily_return(d, orig_picked)
        fast_r_idx = fast_engine._precompute.date_to_idx[d]
        fast_r = fast_engine._get_daily_return_fast(fast_r_idx, orig_picked)
        
        if not np.allclose(orig_r.values, fast_r.values, atol=1e-10):
            print(f"\n  [{i}] RETURNS MISMATCH at {rebal_date}, day {d}")
            diff = np.abs(orig_r.values - fast_r.values)
            max_idx = np.argmax(diff)
            print(f"    Stock: {orig_picked[max_idx]}")
            print(f"    orig: {orig_r.values[max_idx]:.10f}")
            print(f"    fast: {fast_r.values[max_idx]:.10f}")
            break
    else:
        continue
    break
else:
    print("\n  No divergence in rebalance logic!")
    print("  The issue must be in IC computation or other post-processing")
    
    # Run full backtests to compare
    print("\n  Running full backtests...")
    orig_result = orig_engine.run(universe_filter=universe_filter, ranking_factor='mcap', ascending=True)
    fast_result = fast_engine.run(universe_filter=universe_filter, ranking_factor='mcap', ascending=True)
    
    print(f"  Orig positions_log: {orig_result.positions_log.shape}")
    print(f"  Fast positions_log: {fast_result.positions_log.shape}")
    
    orig_cols = set(orig_result.positions_log.columns)
    fast_cols = set(fast_result.positions_log.columns)
    print(f"  Missing in fast: {sorted(orig_cols - fast_cols)[:5]}")
    print(f"  Missing in orig: {sorted(fast_cols - orig_cols)[:5]}")
    
    # Compare equity curves
    orig_ec = orig_result.equity_curve
    fast_ec = fast_result.equity_curve
    merged = pd.merge(orig_ec.rename('orig'), fast_ec.rename('fast'), left_index=True, right_index=True, how='outer')
    both = merged.dropna()
    if len(both) > 0:
        diff = (both['orig'] - both['fast']).abs()
        first_diff = diff[diff > 1e-6].index[0] if (diff > 1e-6).any() else None
        if first_diff:
            print(f"  First equity diff at: {first_diff}")
            print(f"    orig: {both.loc[first_diff, 'orig']:.6f}")
            print(f"    fast: {both.loc[first_diff, 'fast']:.6f}")
