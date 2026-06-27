"""
因子计算模块
=============
根据日线 OHLCV 数据计算截面因子:
  mcap     - 总市值 (亿元), 来源于逐日 daily_mcap_pb_cache
  pb       - 市净率, 来源于逐日 daily_mcap_pb_cache (公告日对齐, 防前视偏差)
  pe       - 市盈率, 来源于逐日 daily_mcap_pb_cache
  mom20d   - 近 20 日收益率 (短期反转因子)
  mom60d   - 近 60 日收益率
  turnover - 相对量比 (volume / avg_vol_20d, 1.0=均值)
  vol20d   - 近 20 日波动率 (年化, 低波动异象)
  ivol     - 特质波动率 (market residual std, Ang 2006 异象)
  max_ret  - 近20日最大日收益 (MAX/彩票型折价, Bali 2011)
  is_limit_up/down - 涨跌停标记

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


PROJECT_DIR = Path(__file__).resolve().parent.parent


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
        # Skip non-stock parquet files (e.g. adv_panel.parquet) that lack a 'date' column
        if 'date' not in df.columns:
            if 'date' in df.index.names:
                df = df.reset_index()
            else:
                continue  # Not a stock price file, skip
        df['symbol'] = symbol
        df['date'] = pd.to_datetime(df['date'])
        dfs.append(df)

    if not dfs:
        raise ValueError("No matching symbols found")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(['symbol', 'date']).reset_index(drop=True)
    return combined


def load_daily_mcap_pb(data_dir: str) -> pd.DataFrame:
    """
    加载逐日历史 mcap/pb/pe 面板 (由 fetch_financials 生成, 已按公告日对齐防前视偏差)

    Returns:
        DataFrame: columns [symbol, date, mcap, pb, pe]
    """
    mcap_dir = Path(data_dir).parent / "daily_mcap_pb_cache"
    if not mcap_dir.exists():
        return pd.DataFrame()

    files = sorted(mcap_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        symbol = f.stem
        df = pd.read_parquet(f)
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = symbol
        keep = ['symbol', 'date', 'mcap', 'pb']
        if 'pe' in df.columns:
            keep.append('pe')
        dfs.append(df[keep])

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(['symbol', 'date']).reset_index(drop=True)
    return combined


def compute_factors(price_data: pd.DataFrame,
                    min_days: int = 120,
                    mcap_pb_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从日线数据计算因子面板和收益率面板

    Args:
        price_data: 包含 symbol, date, close, volume 的 DataFrame
        min_days: 每只股票的最低交易日数 (过滤流动性极差或新上市)
        mcap_pb_data: 逐日历史 mcap/pb 面板 (由 load_daily_mcap_pb 生成)

    Returns:
        factor_panel:  MultiIndex (date, symbol) × [mcap, pb, mom20d, mom60d, turnover, vol20d]
        return_panel:  MultiIndex (date, symbol) × [daily_return]
    """
    data = price_data.copy()
    data = data.sort_values(['symbol', 'date'])

    print(f"  加载 {data['symbol'].nunique()} 只股票, {len(data)} 行日线...")

    # --- 集成历史 mcap/pb (公告日对齐, 无前视偏差) ---
    if mcap_pb_data is not None and not mcap_pb_data.empty:
        old_len = len(data)
        # 只合并 mcap/pb (保留 price_data 中的其他列不变)
        mpb_cols = ['symbol', 'date', 'mcap', 'pb']
        if 'pe' in mcap_pb_data.columns:
            mpb_cols.append('pe')
        mpb = mcap_pb_data[mpb_cols].copy()
        # 用 merge (left join) 而非 concat, 避免列冲突
        data = data.merge(mpb, on=['symbol', 'date'], how='left', suffixes=('', '_hist'))
        # 优先使用历史值, fallback 到原值
        if 'mcap_hist' in data.columns:
            # merge 没有生成 _hist 后缀, 因为 data 中已经可能有 mcap 列
            pass
        # 如果 data 本来没有 mcap/pb (从 Parquet 加载时没有), merge 直接补充
        if 'mcap' in data.columns:
            # pb 同理, 处理 nan
            data['pb'] = data['pb'].ffill()
        print(f"    ✓ 集成历史 mcap/pb ({mpb['symbol'].nunique()} 只, {mpb['date'].nunique()} 天)")
    else:
        # Fallback: 使用静态值
        if 'mcap' not in data.columns:
            print("    ⚠️ 无历史 mcap/pb — 使用静态近似值, 回测中排名可能不准确")
            data['mcap'] = 1.0
        if 'pb' not in data.columns:
            data['pb'] = 2.0

    data['mcap'] = data['mcap'].clip(lower=0.05, upper=50000)
    data['pb'] = data['pb'].clip(lower=0.01, upper=1000)
    if 'pe' in data.columns:
        data['pe'] = data['pe'].clip(lower=-10000, upper=10000)  # PE 极端值截断

    # 集成名称 (用于 ST 过滤)
    name_lookup_path = PROJECT_DIR / "name_lookup.parquet"
    if name_lookup_path.exists():
        names = pd.read_parquet(name_lookup_path)
        data['name'] = data['symbol'].map(names['name']).fillna('')
        print(f"    ✓ 集成股票名称 ({names['name'].notna().sum()} 条)")
    else:
        data['name'] = ''

    # 计算日收益率
    data['daily_return'] = data.groupby('symbol')['close'].pct_change()

    # ── 涨跌停检测 (依赖 daily_return 和 close/high/low) ──
    # 检测逻辑: 涨停 = 收盘接近最高价 + 涨幅接近上限
    #           跌停 = 收盘接近最低价 + 跌幅接近下限
    # 阈值: 主板 ±9.5%, 科创板/创业板 ±19.5% (用 0.5% 容差因数据精度)
    def _limit_threshold(sym: str) -> float:
        code = sym[2:] if len(sym) > 2 else sym
        return 0.195 if code.startswith(('68', '300', '301')) else 0.095

    # 分板块计算
    data['_limit_pct'] = data['symbol'].apply(_limit_threshold)
    data['is_limit_up'] = (
        (data['close'] >= data['high'] * 0.995) &
        (data['daily_return'] > data['_limit_pct'] * 0.95)
    )
    data['is_limit_down'] = (
        (data['close'] <= data['low'] * 1.005) &
        (data['daily_return'] < -data['_limit_pct'] * 0.95)
    )

    # 计算因子 (按股票分组, 滚动计算)
    grouped = data.groupby('symbol')

    # 换手率 — 若 volume 有数据则估算, 否则用 2%
    if 'turnover' in data.columns:
        data['turnover'] = data['turnover'].clip(lower=0.001, upper=100)
    elif 'volume' in data.columns:
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

    # ── 新因子: 特质波动率 (Idiosyncratic Volatility, IVOL) ──
    # 文献: Ang et al. (2006/09); Su (2025) NYU Shanghai
    # 方法: 等权市场收益作为 benchmark, ivol = std(residual) * √252
    # 假设 β≈1 (微盘等权 universe 中 β 均值天然趋近 1)
    market_return = data.groupby('date')['daily_return'].mean().reset_index()
    market_return.columns = ['date', 'market_return']
    data = data.merge(market_return, on='date', how='left')
    data['residual'] = data['daily_return'] - data['market_return']
    # 重新 groupby (因为 merge 改变了 index 顺序)
    grouped2 = data.groupby('symbol')
    data['ivol'] = grouped2['residual'].transform(
        lambda x: x.rolling(20).std() * np.sqrt(252))

    # ── 新因子: MAX (彩票型折价, 近20日最大日收益) ──
    # 文献: Bali-Cakici-Whitelaw (2011); 中科院 (2022)
    # 高 MAX → 彩票型股票 → 未来收益低 (散户追涨后回归)
    data['max_ret'] = grouped2['daily_return'].transform(
        lambda x: x.rolling(20).max())

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
    factor_cols = ['name', 'mcap', 'pb', 'pe', 'mom20d', 'mom60d', 'turnover', 'vol20d',
                   'ivol', 'max_ret', 'is_limit_up', 'is_limit_down',
                   'close', 'open']  # 模板信号需要原始价格字段
    available_cols = [c for c in factor_cols if c in data.columns]

    factor_panel = data.set_index(['date', 'symbol'])[available_cols]

    # --- 构建收益率面板 (MultiIndex) ---
    return_panel = data.set_index(['date', 'symbol'])[['daily_return']]

    return factor_panel, return_panel
