"""
AkShare 财务数据接口
=====================
获取 A 股季度财务指标，合并入 factor_panel:
  - roe_q:   季度 ROE (净资产收益率 %)
  - gross_margin_q: 季度毛利率 %
  - net_profit_growth_q: 季度净利润同比增速 %
  - revenue_growth_q: 季度营收同比增速 %

数据来源: akshare (东方财富/同花顺接口)
缓存:    tools/backtest_mvp/financials_cache/ (Parquet, 按股票代码存)

用法:
  python tools/backtest_mvp/financials_akshare.py          # 全部股票
  python tools/backtest_mvp/financials_akshare.py --symbols sh600000 sz000001
  python tools/backtest_mvp/financials_akshare.py --merge  # 合并到 factor_panel

依赖:
  pip install akshare
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time

PROJECT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT / "financials_cache"
DATA_DIR = PROJECT / "data_cache"


def _ak():
    """懒加载 akshare"""
    import akshare as ak
    return ak


def fetch_quarterly_financials(symbol: str, cache=True, refresh=False) -> pd.DataFrame:
    """
    获取单只股票的季度财务指标

    参数:
        symbol: 股票代码, 如 sh600000 (需转换为 600000)
        cache: 是否使用缓存
        refresh: 是否强制刷新

    返回:
        DataFrame: columns = [date, roe, gross_margin, net_profit_growth, revenue_growth]
        date 为报告期 (季度末日期)
    """
    # 转换 symbol -> akshare 格式 (600000 或 000001)
    code = symbol[2:] if symbol.startswith(("sh", "sz")) else symbol

    cache_file = CACHE_DIR / f"{symbol}.parquet"
    if cache and cache_file.exists() and not refresh:
        return pd.read_parquet(cache_file)

    ak = _ak()
    try:
        # 东方财富个股财务分析指标 (季度)
        # symbol 格式: "600000" 或 "000001"
        df = ak.stock_financial_analysis_indicator(symbol=code)
        if df is None or df.empty:
            return pd.DataFrame()

        # 标准化列名 (不同 akshare 版本列名可能不同)
        # 常见列: 报告期, ROE, 销售毛利率, 净利润同比增长率, 营业总收入同比增长率
        col_map = {
            '报告期': 'date',
            'ROE': 'roe',
            '净资产收益率': 'roe',
            '销售毛利率': 'gross_margin',
            '毛利率': 'gross_margin',
            '净利润同比增长率': 'net_profit_growth',
            '净利润同比': 'net_profit_growth',
            '营业总收入同比增长率': 'revenue_growth',
            '营收同比': 'revenue_growth',
        }
        df = df.rename(columns=col_map)

        # 保留需要的列
        needed = ['date', 'roe', 'gross_margin', 'net_profit_growth', 'revenue_growth']
        available = [c for c in needed if c in df.columns]
        if not available:
            # 尝试模糊匹配
            for col in df.columns:
                if 'ROE' in col or '净资产' in col:
                    df = df.rename(columns={col: 'roe'})
                elif '毛利率' in col or '销售毛利' in col:
                    df = df.rename(columns={col: 'gross_margin'})
                elif '净利润同比' in col or '净利同比' in col:
                    df = df.rename(columns={col: 'net_profit_growth'})
                elif '营收同比' in col or '营业总收入同比' in col:
                    df = df.rename(columns={col: 'revenue_growth'})
            available = [c for c in needed if c in df.columns]

        df = df[available].copy()
        df['date'] = pd.to_datetime(df['date'])

        # 数值列转为 float
        for col in df.columns:
            if col != 'date':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.sort_values('date').reset_index(drop=True)

        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_file, index=False)

        return df

    except Exception as e:
        print(f"  ✗ {symbol} 财务数据获取失败: {e}")
        return pd.DataFrame()


def build_financial_panel(data_dir: str, symbols: list = None,
                           refresh: bool = False) -> pd.DataFrame:
    """
    为全部 (或指定) 股票构建财务因子面板

    返回:
        DataFrame: columns = [symbol, date, roe_q, gross_margin_q, net_profit_growth_q, revenue_growth_q]
        date 为报告期, 需用 ffill 对齐到交易日
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.parquet"))
    if symbols:
        sym_set = set(symbols)
        files = [f for f in files if f.stem in sym_set]

    print(f"获取 {len(files)} 只股票的季度财务数据...")

    all_records = []
    for i, f in enumerate(files):
        symbol = f.stem
        df = fetch_quarterly_financials(symbol, cache=True, refresh=refresh)
        if not df.empty:
            df['symbol'] = symbol
            all_records.append(df)
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(files)}")

    if not all_records:
        print("  ⚠️ 无有效财务数据")
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    return combined


def merge_financials_to_factor_panel(fp: pd.DataFrame,
                                     financial_df: pd.DataFrame) -> pd.DataFrame:
    """
    将季度财务数据合并入 factor_panel

    方法:
      1. 财务数据按 (symbol, date) 
      2. 对每个交易日, forward-fill 最近一季的财务数据
      3. 合并入 fp (新增 roe_q, gross_margin_q, net_profit_growth_q, revenue_growth_q 列)

    参数:
        fp: factor_panel, MultiIndex (date, symbol)
        financial_df: 财务数据, columns = [symbol, date, roe, gross_margin, ...]

    返回:
        合并后的 factor_panel (新增列)
    """
    # 将财务数据 pivot 为 (date, symbol) 面板
    financial_df = financial_df.copy()
    financial_df['date'] = pd.to_datetime(financial_df['date'])

    fp = fp.copy()
    fp_reset = fp.reset_index()  # columns: date, symbol, factor1, factor2, ...

    # 对每个财务因子, 按 symbol 做 asof merge
    financial_cols = [c for c in financial_df.columns
                      if c not in ('symbol', 'date')]

    for col in financial_cols:
        # pivot: symbol -> columns, date -> index
        pivot = financial_df.set_index('date')[['symbol', col]].pivot_table(
            index='date', columns='symbol', values=col
        )
        # 对 fp_reset 的每个 (date, symbol), 找最近一季的财务数据
        merged = []
        for sym in fp_reset['symbol'].unique():
            if sym not in pivot.columns:
                continue
            sym_fp = fp_reset[fp_reset['symbol'] == sym].copy()
            sym_fin = pivot[sym].dropna()
            if sym_fin.empty:
                continue
            # asof join: 对每个交易日, 找最近报告期的值
            sym_fp['__date'] = pd.to_datetime(sym_fp['date'])
            sym_fin = sym_fin.reset_index()
            sym_fin.columns = ['fin_date', col]
            sym_fin = sym_fin.sort_values('fin_date')
            # 用 pandas merge_asof
            sym_fp_sorted = sym_fp.sort_values('__date')
            merged_sym = pd.merge_asof(
                sym_fp_sorted, sym_fin,
                left_on='__date', right_on='fin_date',
                direction='backward'
            )
            merged_sym = merged_sym.drop(columns=['__date', 'fin_date'])
            merged.append(merged_sym)

        if merged:
            merged_df = pd.concat(merged, ignore_index=True)
            # 写回 fp
            fp_reset[f"{col}_q"] = merged_df[col].values if col in merged_df.columns else np.nan

    fp_new = fp_reset.set_index(['date', 'symbol'])
    # 只保留新增的财务列
    new_cols = [f"{c}_q" for c in financial_cols]
    existing_cols = [c for c in new_cols if c in fp_reset.columns]
    fp_new = fp.copy()
    for c in existing_cols:
        fp_new[c] = fp_reset[c].values

    return fp_new


def update_factor_panel_with_financials(data_dir: str, output_path: str = None):
    """
    完整流程: 获取财务数据 → 合并入 factor_panel → 保存
    """
    from factors import load_price_data, compute_factors, load_daily_mcap_pb

    print("Step 1/3: 加载 factor_panel...")
    data = load_price_data(data_dir)
    mpb = load_daily_mcap_pb(data_dir)
    fp, rp = compute_factors(data, mcap_pb_data=mpb)
    print(f"  factor_panel: {fp.shape}")

    print("\nStep 2/3: 获取财务数据...")
    financial_df = build_financial_panel(
        data_dir,
        symbols=fp.index.get_level_values(1).unique().tolist(),
        refresh=False,
    )
    if financial_df.empty:
        print("  ⚠️ 无财务数据, 退出")
        return

    print(f"\nStep 3/3: 合并财务数据到 factor_panel...")
    fp_new = merge_financials_to_factor_panel(fp, financial_df)
    print(f"  新增列: {[c for c in fp_new.columns if c not in fp.columns]}")

    if output_path:
        fp_new.to_parquet(output_path)
        print(f"  保存: {output_path}")

    return fp_new


def main():
    parser = argparse.ArgumentParser(description="AkShare 财务数据接口")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="指定股票代码 (默认全部)")
    parser.add_argument("--refresh", action="store_true",
                        help="强制刷新缓存")
    parser.add_argument("--merge", action="store_true",
                        help="合并财务数据到 factor_panel 并保存")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="日线数据目录")
    args = parser.parse_args()

    if args.merge:
        output_path = str(PROJECT / "factor_panel_with_financials.parquet")
        update_factor_panel_with_financials(args.data_dir, output_path)
    else:
        financial_df = build_financial_panel(
            args.data_dir, symbols=args.symbols, refresh=args.refresh
        )
        if not financial_df.empty:
            print(f"\n  获取完成: {len(financial_df)} 条记录")
            print(financial_df.head(20).to_string())


if __name__ == "__main__":
    main()
