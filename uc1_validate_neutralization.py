#!/usr/bin/env python3
"""
UC1: 截面标准化 + 中性化管线验证
目标：验证中性化后 size 相关性 < 0.5
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.backtest_mvp.factors import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.p1_factor_mining import NeutralizationPipeline


def compute_size_correlation(factor_panel, factor_col, size_col="mcap"):
    """计算因子与 size 的截面相关性（均值）"""
    if factor_col not in factor_panel.columns:
        return np.nan

    corrs = []
    for date, group in factor_panel.groupby(level=0):
        if size_col not in group.columns:
            continue
        corr = group[factor_col].corr(group[size_col])
        if not pd.isna(corr):
            corrs.append(corr)

    return np.mean(corrs) if corrs else np.nan


def main():
    print("=" * 80)
    print("  UC1: 截面中性化管线验证")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/4] 加载 factor_panel...")
    # 脚本在 tools/backtest_mvp/，data_cache 在同级目录
    DATA_DIR = Path(__file__).resolve().parent / "data_cache"
    print(f"  DATA_DIR: {DATA_DIR}")
    stock_files = sorted([f for f in DATA_DIR.glob("*.parquet") if f.name != "adv_panel.parquet"])
    print(f"  文件数: {len(stock_files)}")
    # 只加载股票文件，排除 adv_panel.parquet
    data = load_price_data(str(DATA_DIR), symbols=[f.stem for f in stock_files])
    # 加载逐日 mcap/pb 数据
    mcap_pb_data = load_daily_mcap_pb(str(DATA_DIR))
    factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb_data)
    print(f"  factor_panel: {factor_panel.shape}, return_panel: {return_panel.shape}")

    # 2. 加载行业分类
    print("\n[2/4] 加载行业分类...")
    industry_file = DATA_DIR / "industry_classification.csv"
    if industry_file.exists():
        industry_df = pd.read_csv(industry_file, encoding="utf-8-sig")
        industry_df = industry_df.set_index("symbol")["industry_1"]
        # 合并到 factor_panel
        factor_panel["industry"] = factor_panel.index.get_level_values(1).map(industry_df)
        factor_panel["industry"] = factor_panel["industry"].fillna("其他")
        print(f"  行业分类: {factor_panel['industry'].nunique()} 类, 覆盖率 {factor_panel['industry'].notna().sum()}/{len(factor_panel)}")
    else:
        factor_panel["industry"] = "其他"
        print("  ⚠️ 未找到行业分类，使用默认值")

    # 3. 识别所有因子列
    print("\n[3/4] 识别因子列...")
    exclude_cols = ["close", "open", "high", "low", "volume", "amount", "mcap", "industry", "name"]
    factor_cols = [c for c in factor_panel.columns if c not in exclude_cols]
    print(f"  因子列: {factor_cols}")

    # 4. 计算中性化前的 size 相关性
    print("\n[4/4] 计算中性化前后的 size 相关性...")
    print(f"\n{'因子':<20} {'中性化前':>12} {'中性化后':>12} {'改善':>10}")
    print("-" * 60)

    pipeline = NeutralizationPipeline(factor_panel)
    results = []

    for col in factor_cols:
        before_corr = compute_size_correlation(factor_panel, col)

        if pd.isna(before_corr):
            continue

        # 中性化
        neut_col = f"{col}_neut"
        factor_panel[neut_col] = pipeline.process(
            factor_panel[col], winsor_std=4.0, use_rank=True, neutralize=True, neutralize_strength=0.5
        )

        after_corr = compute_size_correlation(factor_panel, neut_col)

        delta = abs(after_corr) - abs(before_corr) if not pd.isna(after_corr) else np.nan
        improved = "✅" if abs(after_corr) < 0.5 else "❌"

        print(f"{col:<20} {before_corr:>+12.3f} {after_corr:>+12.3f} {delta:>+10.3f} {improved}")
        results.append({
            "factor": col,
            "before_corr": before_corr,
            "after_corr": after_corr,
            "delta": delta,
            "pass": abs(after_corr) < 0.5 if not pd.isna(after_corr) else False,
        })

    # 5. 汇总
    print("\n" + "=" * 80)
    print("  汇总")
    print("=" * 80)
    df = pd.DataFrame(results)
    pass_count = df["pass"].sum()
    total = len(df)
    print(f"\n  通过中性化 (< 0.5): {pass_count}/{total} ({pass_count/total*100:.0f}%)")
    print(f"  平均 |correlation| 改善: {df['delta'].mean():+.3f}")
    print(f"  中性化前平均 |corr|: {df['before_corr'].abs().mean():.3f}")
    print(f"  中性化后平均 |corr|: {df['after_corr'].abs().mean():.3f}")

    if pass_count < total:
        print(f"\n  ⚠️ 未通过中性化的因子:")
        for _, r in df[~df["pass"]].iterrows():
            print(f"    {r['factor']}: {r['after_corr']:+.3f}")

    print("\n  ✅ UC1 验证完成")
    print("=" * 80)

    return df


if __name__ == "__main__":
    main()
