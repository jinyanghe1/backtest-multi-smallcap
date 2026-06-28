#!/usr/bin/env python3
"""
Find the exact cause of divergence in rebalance 3 (2020-08-31)
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

rebal_date = orig_engine.rebalance_dates[3]  # 2020-08-31
print(f"Rebalance 3: {rebal_date}")

orig_snapshot = orig_engine._get_factor_snapshot(rebal_date)
fast_idx = fast_engine._precompute.date_to_idx[rebal_date]
fast_snapshot = fast_engine._get_factor_snapshot_fast(fast_idx)

orig_selected = universe_filter(orig_snapshot, orig_engine.dates, 3)
fast_selected = universe_filter(fast_snapshot, fast_engine.dates, 3)

print(f"Orig selected: {len(orig_selected)} stocks")
print(f"Fast selected: {len(fast_selected)} stocks")

orig_mcap = orig_snapshot.loc[orig_selected]['mcap'].dropna().sort_values()
fast_mcap = fast_snapshot.loc[fast_selected]['mcap'].dropna().sort_values()

print(f"\nOrig mcap (first 10):")
print(orig_mcap.head(10))

print(f"\nFast mcap (first 10):")
print(fast_mcap.head(10))

# Compare values for common stocks
common = list(set(orig_mcap.index) & set(fast_mcap.index))
print(f"\nCommon stocks: {len(common)}")

for s in common[:10]:
    o = orig_mcap[s]
    f = fast_mcap[s]
    if o != f:
        print(f"  {s}: orig={o:.10f}, fast={f:.10f}, diff={abs(o-f):.2e}")

# Check the boundary stocks (around rank 30)
print(f"\n=== Boundary Check (ranks 25-35) ===")
orig_ranks = orig_mcap.reset_index().reset_index().rename(columns={'index': 'rank'})
orig_ranks['rank'] = orig_ranks['rank'] + 1
print("Orig ranks 25-35:")
print(orig_ranks.iloc[24:35])

print("\nFast ranks 25-35:")
fast_ranks = fast_mcap.reset_index().reset_index().rename(columns={'index': 'rank'})
fast_ranks['rank'] = fast_ranks['rank'] + 1
print(fast_ranks.iloc[24:35])

# Check which stock is #30 in each
orig_30 = orig_mcap.iloc[29] if len(orig_mcap) > 29 else None
fast_30 = fast_mcap.iloc[29] if len(fast_mcap) > 29 else None
print(f"\nOrig #30: {orig_mcap.index[29] if len(orig_mcap) > 29 else 'N/A'} = {orig_30}")
print(f"Fast #30: {fast_mcap.index[29] if len(fast_mcap) > 29 else 'N/A'} = {fast_30}")

# Compare the raw mcap values for the two stocks that differ
print(f"\n=== Detailed Comparison for Mismatch Stocks ===")
for s in ['sh600138', 'sh600246']:
    orig_val = orig_snapshot.loc[s, 'mcap'] if s in orig_snapshot.index else 'N/A'
    fast_val = fast_snapshot.loc[s, 'mcap'] if s in fast_snapshot.index else 'N/A'
    print(f"{s}: orig_mcap={orig_val}, fast_mcap={fast_val}")
