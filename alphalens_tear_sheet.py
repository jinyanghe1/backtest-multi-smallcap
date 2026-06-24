"""
Alphalens 因子分析集成脚本
=================================
对 backtest_mvp 的 factor_panel 跑标准 Alphalens 分析:
  - IC 时序、IC 衰减、IC 热力图
  - 分位数组合收益 (quantile portfolio returns)
  - 多空对冲收益 (long-short spread)
  - 换手率分析 (turnover analysis)

用法:
  python tools/backtest_mvp/alphalens_tear_sheet.py           # 全部因子
  python tools/backtest_mvp/alphalens_tear_sheet.py --factor mcap pb
  python tools/backtest_mvp/alphalens_tear_sheet.py --output ./alphalens_html/

依赖:
  pip install alphalens-reloaded akshare
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from factors import load_price_data, compute_factors, load_daily_mcap_pb
from data import DATA_DIR

PROJECT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT / "alphalens_html"


def prepare_alphalens_inputs(fp: pd.DataFrame, rp: pd.DataFrame,
                             factor_name: str, prices_df: pd.DataFrame):
    """
    将 factor_panel / return_panel 转换为 Alphalens 格式

    Alphalens 0.4.x API:
      create_full_tear_sheet(
          factor_data,             # DataFrame, MultiIndex (date, asset), 单列因子值
          prices,                  # DataFrame, index=dates, columns=assets, 价格
          periods=(1, 5, 10),    # 持有期 (交易日)
          quantiles=5,
      )

    Returns:
        factor_data, prices
    """
    # factor_data: 单列 MultiIndex DataFrame
    fdata = fp[[factor_name]].dropna(subset=[factor_name]).copy()
    fdata.columns = [factor_name]

    # prices: 用 close 价重构（需要从原始 price_data 来）
    # 这里从 fp 里拿不到 close，需要从 rp 重构
    # 策略: 用 return_panel 的 daily_return 倒推价格（需要基准价）
    # 简化: 直接用 return_panel 的 index 对齐，prices 用等权市场均价占位
    # 实际上 Alphalens 只需要 prices 来计算 forward returns，
    # 我们可以用 daily_return 累计净值作为 pseudoprice

    # 构建 pseudoprice: 每只股票从 1.0 开始累积 (1 + return)
    returns = rp[['daily_return']].copy()
    returns = returns.reset_index().pivot(index='date', columns='symbol', values='daily_return')
    prices = (1 + returns).cumprod().fillna(method='ffill')

    # 对齐 factor_data 和 prices 的日期
    common_dates = fdata.index.get_level_values(0).unique().intersection(prices.index)
    fdata = fdata.loc[common_dates]
    prices = prices.loc[common_dates]

    return fdata, prices


def run_alphalens(fp, rp, factor_names, output_dir, periods=(1, 5, 10)):
    """
    对每个因子运行 Alphalens 分析，输出 HTML 报告
    """
    import alphalens as al

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建 prices (所有股票的 pseudoprice)
    returns = rp[['daily_return']].copy()
    returns = returns.reset_index().pivot(index='date', columns='symbol', values='daily_return')
    prices = (1 + returns).cumprod().fillna(method='ffill')

    results = {}
    for fac in factor_names:
        if fac not in fp.columns:
            print(f"  ⚠️ 因子 {fac} 不在 factor_panel 中, 跳过")
            continue

        print(f"\n{'='*60}")
        print(f"  因子: {fac}")
        print(f"{'='*60}")

        fdata = fp[[fac]].dropna(subset=[fac]).copy()
        # Alphalens 要求因子值列名不能和 factor name 一样？
        # 实际上 factor_data 可以是多列，但 create_full_tear_sheet 只接受单列
        # 所以这里用 fac 作为列名是可以的

        common_dates = fdata.index.get_level_values(0).unique().intersection(prices.index)
        fdata = fdata.loc[common_dates]
        prices_aligned = prices.loc[common_dates]

        # 至少要有 2 期数据
        if len(common_dates) < max(periods) + 5:
            print(f"  ⚠️ 数据不足 ({len(common_dates)} 天), 跳过")
            continue

        try:
            # Alphalens 0.4.x: create_full_tear_sheet
            # 注意: factor_data 的 column name 会用作图例
            fig = al.create_full_tear_sheet(
                fdata,
                prices_aligned,
                periods=periods,
                quantiles=5,
                show_prints=False,
                save_to_html=str(output_dir / f"alphalens_{fac}.html"),
            )
            print(f"  ✓ 报告已保存: {output_dir / f'alphalens_{fac}.html'}")
            results[fac] = "success"
        except Exception as e:
            print(f"  ✗ Alphalens 分析失败: {e}")
            results[fac] = str(e)

    return results


def run_alphalens_notebook_style(fp, rp, factor_names, output_dir):
    """
    用 alphalens.utils.get_clean_factor_and_forward_returns 手动构建分析
    (更灵活, 不依赖 create_full_tear_sheet 的 HTML 导出)
    """
    from alphalens import performance, plotting, utils, tears

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    returns = rp[['daily_return']].copy()
    returns = returns.reset_index().pivot(index='date', columns='symbol', values='daily_return')
    prices = (1 + returns).cumprod().fillna(method='ffill')

    for fac in factor_names:
        if fac not in fp.columns:
            continue

        print(f"\n  分析因子: {fac}")

        fdata = fp[[fac]].dropna(subset=[fac]).copy()
        common_dates = fdata.index.get_level_values(0).unique().intersection(prices.index)
        fdata = fdata.loc[common_dates]
        prices_aligned = prices.loc[common_dates]

        if len(common_dates) < 10:
            print(f"    ⚠️ 数据不足, 跳过")
            continue

        try:
            # 用 Alphalens 的工具函数
            factor_data = fdata[[fac]]
            prices_data = prices_aligned

            # get_clean_factor_and_forward_returns
            merged = utils.get_clean_factor_and_forward_returns(
                factor_data, prices_data, periods=(1, 5, 10),
            )

            # 生成 tear sheet (输出到 HTML)
            out_path = str(output_dir / f"{fac}_tear.html")
            tears.create_full_tear_sheet(
                factor_data, prices_data,
                periods=(1, 5, 10), quantiles=5,
                save_to_html=out_path,
            )
            print(f"    ✓ 保存: {out_path}")
        except Exception as e:
            print(f"    ✗ 失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="Alphalens 因子分析")
    parser.add_argument("--factor", nargs="+", default=None,
                        help="指定因子名 (默认全部)")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help="输出目录 (默认 tools/backtest_mvp/alphalens_html/)")
    parser.add_argument("--periods", nargs="+", type=int, default=(1, 5, 10),
                        help="持有期 (交易日)")
    args = parser.parse_args()

    print("加载因子面板...", end=" ", flush=True)
    data = load_price_data(str(DATA_DIR))
    mpb = load_daily_mcap_pb(str(DATA_DIR))
    fp, rp = compute_factors(data, mcap_pb_data=mpb)
    print(f"{fp.index.get_level_values(1).nunique()} 只, "
          f"{fp.index.get_level_values(0).nunique()} 交易日")

    factor_cols = [c for c in fp.columns
                   if c not in ('name', 'is_limit_up', 'is_limit_down')]
    if args.factor:
        factors = [f for f in args.factor if f in factor_cols]
        missing = set(args.factor) - set(factors)
        if missing:
            print(f"  ⚠️ 未找到因子: {missing}")
    else:
        factors = factor_cols

    print(f"\n对以下 {len(factors)} 个因子运行 Alphalens 分析:")
    for f in factors:
        print(f"  - {f}")
    print()

    results = run_alphalens(fp, rp, factors, args.output, periods=tuple(args.periods))

    # 汇总
    print(f"\n{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    ok = sum(1 for v in results.values() if v == "success")
    print(f"  成功: {ok}/{len(results)}")
    for fac, status in results.items():
        tag = "✓" if status == "success" else "✗"
        print(f"    {tag} {fac}: {status}")


if __name__ == "__main__":
    main()
