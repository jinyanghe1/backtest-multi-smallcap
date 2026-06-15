#!/bin/bash
# 拉取单只股票的财务数据 (shell wrapper 解决沙箱问题)
# Usage: bash fetch_one.sh sh600239

SYMBOL=$1
if [ -z "$SYMBOL" ]; then
    echo "Usage: bash fetch_one.sh <symbol>"
    exit 1
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
/Users/hejinyang/miniconda3/bin/python -c "
import akshare as ak
import pandas as pd
import numpy as np
import re, sys, traceback
from pathlib import Path

symbol_raw = '$SYMBOL'
DATA_ROOT = Path('$DIR')
FINANCIALS_CACHE = DATA_ROOT / 'financials_cache'
PROFILES_CACHE = DATA_ROOT / 'profiles_cache'
MCAP_PB_CACHE = DATA_ROOT / 'daily_mcap_pb_cache'
PRICE_CACHE = DATA_ROOT / 'data_cache'
for d in [FINANCIALS_CACHE, PROFILES_CACHE, MCAP_PB_CACHE]:
    d.mkdir(exist_ok=True)

symbol = symbol_raw[2:] if len(symbol_raw) > 2 and symbol_raw[:2] in ('sh','sz','bj') else symbol_raw
em_code = f'{symbol}.SZ' if symbol.startswith(('0','3','4','9')) else f'{symbol}.SH'

status = []

# --- Step 1: 财务指标 ---
cache_path = FINANCIALS_CACHE / f'{symbol_raw}.parquet'
fin_ok = False
if not cache_path.exists():
    try:
        df = ak.stock_financial_analysis_indicator_em(symbol=em_code, indicator='按报告期')
        keep = ['REPORT_DATE','NOTICE_DATE','BPS','EPSJB','TOTALOPERATEREVE','PARENTNETPROFIT','ROEJQ']
        available = [c for c in keep if c in df.columns]
        result = df[available].copy()
        rename = {'REPORT_DATE':'report_date','NOTICE_DATE':'notice_date','BPS':'bps','EPSJB':'eps','TOTALOPERATEREVE':'revenue','PARENTNETPROFIT':'net_profit','ROEJQ':'roe'}
        result.columns = [rename.get(c,c) for c in available]
        result['report_date'] = pd.to_datetime(result['report_date'], errors='coerce')
        result['notice_date'] = pd.to_datetime(result['notice_date'], errors='coerce')
        for col in ['bps','eps','revenue','net_profit','roe']:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')
        result = result.sort_values('report_date').reset_index(drop=True)
        result['symbol'] = symbol_raw
        result.to_parquet(cache_path, index=False)
        fin_ok = True
        status.append(f'fin={len(result)}Q')
    except Exception as e:
        status.append(f'fin_fail={str(e)[:60]}')
else:
    df = pd.read_parquet(cache_path)
    fin_ok = True
    status.append(f'fin={len(df)}Q(cached)')

# --- Step 2: 总股本 ---
profile_path = PROFILES_CACHE / f'{symbol_raw}.parquet'
pro_ok = False
if not profile_path.exists():
    try:
        df2 = ak.stock_profile_cninfo(symbol=symbol)
        raw_val = str(df2['注册资金'].iloc[0]) if '注册资金' in df2.columns else ''
        if raw_val:
            nums = re.findall(r'[\d.]+', raw_val)
            if nums:
                yi = float(nums[0]) / 10000.0
                pd.DataFrame([{'total_shares_yi': round(yi, 4)}]).to_parquet(profile_path, index=False)
                pro_ok = True
                status.append(f'shares={yi:.4f}亿')
            else:
                status.append('shares_no_num')
        else:
            status.append('shares_no_row')
    except Exception as e:
        status.append(f'shares_fail={str(e)[:60]}')
else:
    p = pd.read_parquet(profile_path)
    status.append(f'shares={p.iloc[0,0]:.4f}亿(cached)')
    pro_ok = True

# --- Step 3: 逐日 mcap/pb ---
mpb_path = MCAP_PB_CACHE / f'{symbol_raw}.parquet'
if not mpb_path.exists() and pro_ok and fin_ok:
    try:
        px_path = PRICE_CACHE / f'{symbol_raw}.parquet'
        if px_path.exists():
            px = pd.read_parquet(px_path)
            px['date'] = pd.to_datetime(px['date'])
            px = px.sort_values('date')
            # mcap
            shares = pd.read_parquet(profile_path)['total_shares_yi'].iloc[0]
            px['mcap'] = px['close'] * shares
            # pb/pe by notice_date alignment
            fin = pd.read_parquet(cache_path)
            fin = fin.dropna(subset=['notice_date']).sort_values('notice_date')
            if 'bps' in fin.columns or 'eps' in fin.columns:
                all_dates = px['date'].values
                bps = np.full(len(all_dates), np.nan)
                epss = np.full(len(all_dates), np.nan)
                for _, r in fin.iterrows():
                    nd = r['notice_date']
                    if pd.isna(nd): continue
                    mask = all_dates >= pd.Timestamp(nd)
                    if 'bps' in r and not pd.isna(r['bps']):
                        bps[mask] = r['bps']
                    if 'eps' in r and not pd.isna(r['eps']):
                        epss[mask] = r['eps']
                px['bps'] = pd.Series(bps).ffill().values
                px['eps'] = pd.Series(epss).ffill().values
                px['pb'] = px['close'] / px['bps'].replace(0, np.nan)
                px['pe'] = px['close'] / px['eps'].replace(0, np.nan)
                px = px.drop(columns=['bps','eps'], errors='ignore')
            # keep only needed cols
            keep_cols = ['date','close','mcap','pb','pe']
            px = px[[c for c in keep_cols if c in px.columns]]
            px.to_parquet(mpb_path, index=False)
            status.append('mcap_pb=ok')
        else:
            status.append('mcap_pb=no_price')
    except Exception as e:
        status.append(f'mcap_pb_fail={str(e)[:60]}')

# --- 输出 ---
print(f'{symbol_raw:>12} | {\" | \".join(status)}')
" 2>&1
