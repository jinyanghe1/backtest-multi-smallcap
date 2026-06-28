#!/usr/bin/env python3
"""
Diagnostic: Check why snapshot mcap is constant
"""
import sys, pandas as pd
sys.path.insert(0, '/Users/hejinyang/thinking_and_learning_with_AI')

from tools.backtest_mvp.factors.legacy import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.data import DATA_DIR
from tools.backtest_mvp.engine_precompute import EngineDataPrecomputation

print('=== Diagnosing constant mcap in snapshot ===')

data = load_price_data(str(DATA_DIR))
mcap_pb = load_daily_mcap_pb(str(DATA_DIR))

factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)

print(f'factor_panel shape: {factor_panel.shape}')
print(f'factor_panel index levels: {factor_panel.index.names}')
print(f'factor_panel index type: {type(factor_panel.index.get_level_values(0)[0])}')
print(f'factor_panel mcap: {factor_panel["mcap"].describe()}')

# Check mcap unstack
mcap_series = factor_panel['mcap']
print(f'\nmcap_series index: {mcap_series.index.names}')
print(f'First 5 mcap values: {mcap_series.head().tolist()}')

pivot = mcap_series.unstack(level=1)
print(f'\npivot shape: {pivot.shape}')
print(f'pivot columns (first 5): {list(pivot.columns[:5])}')
print(f'pivot index type: {type(pivot.index[0])}')
print(f'pivot index (first 5): {list(pivot.index[:5])}')
print(f'pivot sample (first 5 rows, 5 cols):')
print(pivot.iloc[:5, :5])

# Now precompute
pre = EngineDataPrecomputation(factor_panel, return_panel)
print(f'\npre.dates type: {type(pre.dates[0])}')
print(f'pre.dates (first 5): {list(pre.dates[:5])}')
print(f'pre.stocks (first 5): {pre.stocks[:5]}')

# Check mcap_2d
mcap_idx = pre.numeric_factor_names.index('mcap') if 'mcap' in pre.numeric_factor_names else None
print(f'\nmcap_2d index: {mcap_idx}')
if mcap_idx is not None:
    print(f'mcap_2d shape: {pre.factor_2d[:, :, mcap_idx].shape}')
    print(f'mcap_2d first row: {pre.factor_2d[0, :5, mcap_idx]}')
    print(f'mcap_2d first row unique: {len(set(pre.factor_2d[0, :, mcap_idx]))}')
    print(f'mcap_2d first row describe: {pd.Series(pre.factor_2d[0, :, mcap_idx]).describe()}')

# Get snapshot and compare
snapshot = pre.get_factor_snapshot_fast(0)
print(f'\nsnapshot mcap describe: {snapshot["mcap"].describe()}')
print(f'snapshot mcap first 5: {snapshot["mcap"].head().tolist()}')
