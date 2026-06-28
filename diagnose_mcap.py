#!/usr/bin/env python3
"""
Diagnostic: Check why mcap proxy is not working
"""
import sys
sys.path.insert(0, '/Users/hejinyang/thinking_and_learning_with_AI')

from tools.backtest_mvp.factors.legacy import load_price_data, load_daily_mcap_pb
from tools.backtest_mvp.data import DATA_DIR
import pandas as pd

data = load_price_data(str(DATA_DIR))
mcap_pb = load_daily_mcap_pb(str(DATA_DIR))

print(f"data shape: {data.shape}")
print(f"data columns: {list(data.columns)}")
print(f"mcap_pb shape: {mcap_pb.shape}")

# Merge
mpb = mcap_pb[['symbol', 'date', 'mcap', 'pb']].copy()
data2 = data.merge(mpb, on=['symbol', 'date'], how='left', suffixes=('', '_hist'))

print(f"After merge, data2 shape: {data2.shape}")
print(f"After merge, mcap NaN: {data2['mcap'].isna().sum()}/{len(data2)}")
print(f"After merge, close NaN: {data2['close'].isna().sum()}/{len(data2)}")

# ffill
data2['mcap'] = data2.groupby('symbol')['mcap'].ffill()
print(f"After ffill, mcap NaN: {data2['mcap'].isna().sum()}/{len(data2)}")

# Check a specific stock
sample = data2[data2['symbol'] == 'sh300999']
print(f"\nsh300999:")
print(f"  rows: {len(sample)}")
print(f"  mcap NaN: {sample['mcap'].isna().sum()}")
print(f"  close NaN: {sample['close'].isna().sum()}")
print(f"  First few rows:")
print(sample[['symbol', 'date', 'close', 'mcap']].head())

# Apply proxy
mcap_na = data2['mcap'].isna()
print(f"\nmcap_na.sum(): {mcap_na.sum()}")
print(f"close[mcap_na] NaN: {data2.loc[mcap_na, 'close'].isna().sum()}")
print(f"close[mcap_na] sample:")
print(data2.loc[mcap_na, ['symbol', 'date', 'close', 'mcap']].head())

data2.loc[mcap_na, 'mcap'] = data2.loc[mcap_na, 'close']
print(f"After proxy, mcap NaN: {data2['mcap'].isna().sum()}")

# Check again
sample2 = data2[data2['symbol'] == 'sh300999']
print(f"\nsh300999 after proxy:")
print(f"  mcap NaN: {sample2['mcap'].isna().sum()}")
print(sample2[['symbol', 'date', 'close', 'mcap']].head())
