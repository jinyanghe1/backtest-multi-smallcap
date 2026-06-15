"""
财务数据批量轮询脚本
=====================
两步拉取:
  1. stock_financial_analysis_indicator_em → BPS, EPS, NOTICE_DATE, REPORT_DATE
  2. stock_profile_cninfo → 总股本 (注册资金 万元 → 亿股)

输出:
  financials_cache/{symbol}.parquet   — 逐季财务指标 (按 NOTICE_DATE 排序)
  profiles_cache/{symbol}.parquet     — 总股本
  daily_mcap_pb_cache/{symbol}.parquet — 逐日 mcap = close × shares, pb = close / BPS_forward_filled
                                          (按 NOTICE_DATE 对齐, 杜绝前视偏差)

用法:
  python tools/backtest_mvp/fetch_financials.py                     # 处理全部缓存股票
  python tools/backtest_mvp/fetch_financials.py --symbols sh600239 sz002630  # 指定
  python tools/backtest_mvp/fetch_financials.py --retry-failed       # 重试失败的
"""

import pandas as pd
import numpy as np
import akshare as ak
import time
import sys
from pathlib import Path
from typing import List, Optional

# --- 路径 ---
DATA_ROOT = Path(__file__).resolve().parent
PRICE_CACHE = DATA_ROOT / "data_cache"
FINANCIALS_CACHE = DATA_ROOT / "financials_cache"
PROFILES_CACHE = DATA_ROOT / "profiles_cache"
MCAP_PB_CACHE = DATA_ROOT / "daily_mcap_pb_cache"

# --- 财务字段映射 ---
# akshare 返回的列名 → 我们的简洁名
FIELD_MAP = {
    "REPORT_DATE":   "report_date",
    "NOTICE_DATE":   "notice_date",
    "BPS":           "bps",         # 每股净资产
    "EPSJB":         "eps",         # 基本每股收益
    "TOTALOPERATEREVE": "revenue",  # 营业总收入
    "PARENTNETPROFIT":  "net_profit",  # 归属净利润
    "ROEJQ":         "roe",
    "REPORT_TYPE":   "report_type",
}
KEEP_FIELDS = [k for k in FIELD_MAP if k != "REPORT_TYPE"]  # report_type just for debugging


def strip_symbol(raw: str) -> str:
    """确保 A 股代码不带后缀 (只需要 6 位数字)"""
    return raw.split(".")[0] if "." in raw else raw


def fetch_one_financials(symbol_raw: str, retries: int = 3) -> Optional[pd.DataFrame]:
    """
    拉取单只股票的逐季财务指标 (按报告期)
    Returns None if all retries fail
    """
    symbol = strip_symbol(symbol_raw)
    em_code = f"{symbol}.SZ" if symbol.startswith(("0", "3", "4")) else f"{symbol}.SH"

    for attempt in range(retries):
        try:
            df = ak.stock_financial_analysis_indicator_em(
                symbol=em_code,
                indicator="按报告期",
            )
            if df.empty:
                print(f"    ⚠️ {symbol} 返回空 DataFrame (attempt {attempt+1})")
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return None

            # 提取需要的列
            available = [c for c in KEEP_FIELDS if c in df.columns]
            if not available:
                print(f"    ⚠️ {symbol} 无有效字段")
                return None

            result = df[available].copy()
            result.columns = [FIELD_MAP[c] for c in available]

            # 类型转换
            result["report_date"] = pd.to_datetime(result["report_date"], errors="coerce")
            result["notice_date"] = pd.to_datetime(result["notice_date"], errors="coerce")
            for col in ["bps", "eps", "revenue", "net_profit", "roe"]:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors="coerce")

            result = result.sort_values("report_date").reset_index(drop=True)
            result["symbol"] = symbol_raw
            return result

        except Exception as e:
            err_msg = str(e)[:100]
            print(f"    ⚠️ {symbol} attempt {attempt+1}/{retries}: {err_msg}")
            if attempt < retries - 1:
                time.sleep(3)
            else:
                print(f"    ✗ {symbol} 全部重试失败")
                return None


def fetch_one_profile(symbol_raw: str, retries: int = 3) -> Optional[dict]:
    """
    拉取总股本, 返回 {"total_shares_yi": float} or None
    使用 cninfo 接口 (注册资金, 万元 → 亿股)
    """
    symbol = strip_symbol(symbol_raw)
    for attempt in range(retries):
        try:
            df = ak.stock_profile_cninfo(symbol=symbol)
            if df.empty:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return None
            # 注册资金 列 (万元)
            row = df[df["项目名称"] == "注册资金"]
            if row.empty:
                # 备选: 某些版本列名是 'registered_capital' 或中文
                row = df[df.iloc[:, 0].str.contains("注册资金|注册资本", na=False)]
            if not row.empty:
                raw_val = str(row["项目内容"].iloc[0])
                # 提取数字
                import re
                nums = re.findall(r"[\d.]+", raw_val)
                if nums:
                    wan = float(nums[0])
                    yi = wan / 10000.0
                    return {"total_shares_yi": round(yi, 4)}
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                print(f"    ✗ {symbol_raw} profile failed: {str(e)[:80]}")
                return None


def build_daily_mcap_pb(symbol_raw: str, shares_yi: float,
                        financials: pd.DataFrame,
                        price_data: pd.DataFrame) -> pd.DataFrame:
    """
    构建逐日 mcap/pb 面板 (无前视偏差)

    Args:
        symbol_raw: 如 sh600239
        shares_yi: 总股本(亿股)
        financials: report_date, notice_date, bps, eps (按 notice_date 排序)
        price_data: date, close (这只股票的日线)

    Returns:
        DataFrame: date, close, mcap, pb, pe
          - mcap = close × shares_yi
          - pb = close / bps_forward_filled (按 NOTICE_DATE 对齐)
          - pe = close / eps_forward_filled
    """
    px = price_data[["date", "close"]].copy()
    px = px.sort_values("date")

    if px.empty:
        return pd.DataFrame()

    # 基础 mcap (不依赖财务数据)
    px["mcap"] = px["close"] * shares_yi

    # --- PB / PE 按公告日对齐 ---
    # 只有公告日 (NOTICE_DATE) 之后才能使用该季报的 BPS/EPS
    # 公告日之前 = 上一季报的 forward-fill
    if financials is not None and not financials.empty:
        fin = financials[["notice_date", "report_date", "bps", "eps"]].copy()
        fin = fin.dropna(subset=["notice_date"]).sort_values("notice_date")
        if not fin.empty and ("bps" in fin.columns or "eps" in fin.columns):
            # 创建每日的 BPS/EPS 时间序列
            # 从最早公告日开始, 每个公告日更新 BPS/EPS 值
            all_dates = px["date"].unique()
            bps_series = np.full(len(all_dates), np.nan)
            eps_series = np.full(len(all_dates), np.nan)

            for _, row in fin.iterrows():
                nd = row["notice_date"]
                if pd.isna(nd):
                    continue
                mask = all_dates >= nd
                if "bps" in row and not pd.isna(row["bps"]):
                    bps_series[mask] = row["bps"]
                if "eps" in row and not pd.isna(row["eps"]):
                    eps_series[mask] = row["eps"]

            px["bps_ffill"] = bps_series
            px["eps_ffill"] = eps_series
            px["bps_ffill"] = px["bps_ffill"].ffill()
            px["eps_ffill"] = px["eps_ffill"].ffill()

            if "bps_ffill" in px.columns:
                px["pb"] = px["close"] / px["bps_ffill"].replace(0, np.nan)
            if "eps_ffill" in px.columns:
                px["pe"] = px["close"] / px["eps_ffill"].replace(0, np.nan)

    # 清理辅助列
    px = px.drop(columns=[c for c in ["bps_ffill", "eps_ffill"] if c in px.columns],
                 errors="ignore")

    return px


def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量拉取财务数据")
    parser.add_argument("--symbols", nargs="*", help="指定股票代码 (如 sh600239 sz002630)")
    parser.add_argument("--retry-failed", action="store_true", help="仅重试之前失败的")
    parser.add_argument("--skip-mcap-pb", action="store_true", help="跳过逐日 mcap/pb 构建")
    parser.add_argument("--delay", type=float, default=1.0, help="请求间隔 (秒)")
    args = parser.parse_args()

    # --- 确定待处理股票列表 ---
    FINANCIALS_CACHE.mkdir(exist_ok=True)
    PROFILES_CACHE.mkdir(exist_ok=True)
    MCAP_PB_CACHE.mkdir(exist_ok=True)

    if args.symbols:
        targets = args.symbols
    else:
        targets = sorted([f.stem for f in PRICE_CACHE.glob("*.parquet")])

    if args.retry_failed:
        done_fin = {f.stem for f in FINANCIALS_CACHE.glob("*.parquet")}
        done_pro = {f.stem for f in PROFILES_CACHE.glob("*.parquet")}
        targets = [t for t in targets if t not in done_fin or t not in done_pro]
        print(f"重试模式: {len(targets)} 只待处理")

    if not targets:
        print("没有需要处理的股票。")
        return

    total = len(targets)
    print(f"开始轮询 {total} 只股票...")
    print(f"请求间隔: {args.delay}s\n")

    success_fin = 0
    fail_fin = []
    success_pro = 0
    fail_pro = []
    success_mcap = 0

    for i, symbol in enumerate(targets):
        print(f"[{i+1}/{total}] {symbol} ", end="", flush=True)

        # --- Step 1: 拉取财务指标 ---
        cache_path = FINANCIALS_CACHE / f"{symbol}.parquet"
        financials_df = None
        if not cache_path.exists():
            financials_df = fetch_one_financials(symbol)
            if financials_df is not None:
                financials_df.to_parquet(cache_path, index=False)
                success_fin += 1
                n_q = len(financials_df)
                print(f"→ 财务 {n_q}期 ", end="", flush=True)
                time.sleep(args.delay)
            else:
                fail_fin.append(symbol)
                print("→ 财务 ✗ ", end="", flush=True)
        else:
            financials_df = pd.read_parquet(cache_path)
            print("→ 财务(缓存) ", end="", flush=True)

        # --- Step 2: 拉取总股本 ---
        profile_path = PROFILES_CACHE / f"{symbol}.parquet"
        shares_yi = None
        if not profile_path.exists():
            profile = fetch_one_profile(symbol)
            if profile:
                pd.DataFrame([profile]).to_parquet(profile_path, index=False)
                success_pro += 1
                shares_yi = profile["total_shares_yi"]
                print(f"→ 股本 {shares_yi}亿股 ", end="", flush=True)
                time.sleep(args.delay)
            else:
                fail_pro.append(symbol)
                print("→ 股本 ✗ ", end="", flush=True)
        else:
            profile = pd.read_parquet(profile_path)
            if "total_shares_yi" in profile.columns:
                shares_yi = float(profile["total_shares_yi"].iloc[0])
            print("→ 股本(缓存) ", end="", flush=True)

        # --- Step 3: 构建逐日 mcap/pb (可选) ---
        if not args.skip_mcap_pb and shares_yi is not None:
            mcap_pb_path = MCAP_PB_CACHE / f"{symbol}.parquet"
            if not mcap_pb_path.exists():
                try:
                    price_path = PRICE_CACHE / f"{symbol}.parquet"
                    if price_path.exists():
                        px = pd.read_parquet(price_path)
                        px["date"] = pd.to_datetime(px["date"])
                        daily = build_daily_mcap_pb(symbol, shares_yi, financials_df, px)
                        if not daily.empty:
                            daily.to_parquet(mcap_pb_path, index=False)
                            success_mcap += 1
                except Exception as e:
                    print(f"(mcap/pb: {str(e)[:40]}) ", end="")

        print()

    # --- 总汇报 ---
    print("\n" + "=" * 60)
    print("  财务数据轮询完成")
    print("=" * 60)
    c_fin = len(list(FINANCIALS_CACHE.glob("*.parquet")))
    c_pro = len(list(PROFILES_CACHE.glob("*.parquet")))
    c_mpb = len(list(MCAP_PB_CACHE.glob("*.parquet")))
    print(f"  财务指标: {c_fin}/{total} 只 (本次新增: {success_fin})")
    print(f"  总股本:   {c_pro}/{total} 只 (本次新增: {success_pro})")
    print(f"  逐日mcap/pb: {c_mpb} 只 (本次新增: {success_mcap})")

    if fail_fin:
        print(f"\n  财务失败 ({len(fail_fin)}): {', '.join(fail_fin)}")
    if fail_pro:
        print(f"  股本失败 ({len(fail_pro)}): {', '.join(fail_pro)}")
    if fail_fin or fail_pro:
        print(f"\n  重试命令: python tools/backtest_mvp/fetch_financials.py --retry-failed")


if __name__ == "__main__":
    main()
