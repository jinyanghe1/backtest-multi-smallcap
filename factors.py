"""
因子计算模块
=============
根据日线 OHLCV 数据计算截面因子:
  mcap     - 总市值 (亿元)
  pb       - 市净率
  mom20d   - 近 20 日收益率
  mom60d   - 近 60 日收益率
  turnover - 换手率 (%)
  vol20d   - 近 20 日波动率 (%)

输入: Parquet 文件目录, 每只股票一个文件 (date, open, high, low, close, volume, mcap, pb, turnover)
输出:
  factor_panel:  MultiIndex DataFrame (date, stock) × [mcap, pb, mom20d, mom60d, turnover, vol20d]
  return_panel:  MultiIndex DataFrame (date, stock) × [daily_return]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import glob


def load_price_data(data_dir: str, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    加载所有股票的 Parquet 日线, 合并为统一 DataFrame

    假设文件命名: sh600000.parquet
    每列: date, open, high, low, close, volume, mcap (亿), pb, turnover (%)

    Returns:
        DataFrame with columns: symbol, date, open, high, low, close, volume, mcap, pb, turnover
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    dfs = []
    for f in files:
        symbol = f.stem  # "sh600000"
        if symbols and symbol not in symbols:
            continue
        df = pd.read_parquet(f)
        df['symbol'] = symbol
        df['date'] = pd.to_datetime(df['date'])
        dfs.append(df)

    if not dfs:
        raise ValueError("No matching symbols found")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(['symbol', 'date']).reset_index(drop=True)
    return combined


def compute_factors(price_data: pd.DataFrame,
                    min_days: int = 120) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从日线数据计算因子面板和收益率面板

    Args:
        price_data: 包含 symbol, date, close, mcap, pb, turnover 的 DataFrame
        min_days: 每只股票的最低交易日数 (过滤流动性极差或新上市)

    Returns:
        factor_panel:  MultiIndex (date, symbol) × [mcap, pb, mom20d, mom60d, turnover, vol20d]
        return_panel:  MultiIndex (date, symbol) × [daily_return]
    """
    data = price_data.copy()
    data = data.sort_values(['symbol', 'date'])

    print(f"  加载 {data['symbol'].nunique()} 只股票, {len(data)} 行日线...")

    # 计算日收益率
    data['daily_return'] = data.groupby('symbol')['close'].pct_change()

    # 计算因子 (按股票分组, 滚动计算)
    grouped = data.groupby('symbol')

    # 市值 (直接使用, 但做亿单位归一化) — 若无历史数据, 用静态近似
    if 'mcap' in data.columns:
        data['mcap'] = data['mcap'].clip(lower=0.1, upper=10000)
    else:
        print("  ⚠️ 无 mcap 列 — 回测中市值相关策略可能不准确")
        data['mcap'] = 1.0  # 占位

    # PB — 若无数据填 2.0 (中性)
    if 'pb' not in data.columns:
        data['pb'] = 2.0

    # 换手率 — 若 volume 有数据则估算, 否则用 2%
    if 'turnover' in data.columns:
        data['turnover'] = data['turnover'].clip(lower=0.001, upper=100)
    elif 'volume' in data.columns:
        # 从 volume 粗略估算 turnover (volume / avg volume of last 20 days)
        avg_vol = data.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
        data['turnover'] = (data['volume'] / avg_vol.replace(0, 1)).clip(0, 20)
    else:
        data['turnover'] = 2.0

    # 近 20/60 日动量
    data['mom20d'] = grouped['close'].transform(
        lambda x: x.pct_change(20))
    data['mom60d'] = grouped['close'].transform(
        lambda x: x.pct_change(60))

    # 20 日波动率 (年化)
    data['vol20d'] = grouped['daily_return'].transform(
        lambda x: x.rolling(20).std() * np.sqrt(252))

    # 换手率做百分比 (输入已经是百分比则保留)
    if 'turnover' in data.columns:
        if data['turnover'].median() > 50:
            data['turnover'] = data['turnover'] / 100  # 换算成小数

    # --- 过滤 ---
    # 删除前 120 天 (因子的滚动计算需要预热)
    data = data.dropna(subset=['daily_return', 'mom20d', 'mom60d', 'vol20d'])

    # 删除交易日不足的股票
    counts = data.groupby('symbol')['date'].count()
    valid_symbols = counts[counts >= min_days].index
    data = data[data['symbol'].isin(valid_symbols)]

    print(f"  过滤后: {len(valid_symbols)} 只股票, {len(data)} 行 (≥{min_days}天交易)")

    # --- 构建因子面板 (MultiIndex) ---
    factor_cols = ['mcap', 'pb', 'mom20d', 'mom60d', 'turnover', 'vol20d']
    available_cols = [c for c in factor_cols if c in data.columns]

    factor_panel = data.set_index(['date', 'symbol'])[available_cols]

    # --- 构建收益率面板 (MultiIndex) ---
    return_panel = data.set_index(['date', 'symbol'])[['daily_return']]

    return factor_panel, return_panel
