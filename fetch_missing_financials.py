#!/usr/bin/env python3
"""
缺失财务数据补充 — 多数据源采集 + 规整 + 验证
==============================================
数据源优先级: EM(东财) → THS(同花顺) → Sina(新浪)
统一输出 schema: report_date/notice_date/bps/eps/revenue/net_profit/total_equity

Unit handling:
  - EM: all in 元
  - THS: 万/亿 mixed — needs parsing
  - Sina: all in 元

用法: python tools/backtest_mvp/fetch_missing_financials.py [--dry-run] [--source auto|em|ths|sina]
"""

import sys
import os
import time
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import akshare as ak
from tools.backtest_mvp.data import DATA_DIR, fetch_stock_kline

SCRIPT_DIR = Path(__file__).parent
FIN_DIR = SCRIPT_DIR / "financials_cache"
PROF_DIR = SCRIPT_DIR / "profiles_cache"
MB_DIR = SCRIPT_DIR / "daily_mcap_pb_cache"

FIN_DIR.mkdir(exist_ok=True)
PROF_DIR.mkdir(exist_ok=True)
MB_DIR.mkdir(exist_ok=True)

# ── Unified Schema ──
UNIFIED_COLS = [
    "report_date", "notice_date", "bps", "eps",
    "revenue", "net_profit", "total_equity",
]
# All values in 元; bps/eps in 元/股


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def parse_number(raw: str) -> float:
    """解析含单位的数字: '3.5亿'→350000000, '1200万'→12000000, '5000'→5000"""
    if raw is None or raw == "" or raw == "nan":
        return np.nan
    s = str(raw).strip()
    # 处理负值
    sign = -1 if s.startswith("-") else 1
    if s.startswith("-"):
        s = s[1:]
    # 提取数字和单位
    num_match = re.match(r"([\d,.]+)\s*(亿|万|元|%|)?", s)
    if not num_match:
        try:
            return sign * float(s)
        except ValueError:
            return np.nan
    num = float(num_match.group(1).replace(",", ""))
    unit = num_match.group(2) or ""
    if unit == "亿":
        num *= 1e8
    elif unit == "万":
        num *= 1e4
    elif unit == "%":
        num /= 100  # percentage → decimal
    return sign * num


def code_to_em(symbol: str) -> str:
    """sh600051 → 600051.SH"""
    code = symbol[2:]
    if symbol.startswith("sh"):
        return f"{code}.SH"
    elif symbol.startswith("sz"):
        return f"{code}.SZ"
    else:  # bj
        return f"{code}.BJ"


def code_to_sina(symbol: str) -> str:
    """sh600051 → sh600051"""
    return symbol


def code_to_numeric(symbol: str) -> str:
    """sh600051 → 600051"""
    return symbol[2:]


# ═══════════════════════════════════════════════════════════
# 数据源解析器
# ═══════════════════════════════════════════════════════════

def fetch_em(symbol: str) -> Optional[pd.DataFrame]:
    """
    东财 EM 接口 — BPS/EPS/NOTICE_DATE 直接可用
    Returns DataFrame with UNIFIED_COLS or None
    """
    try:
        em_code = code_to_em(symbol)
        df = ak.stock_financial_analysis_indicator_em(symbol=em_code, indicator="按报告期")
        if df is None or len(df) == 0:
            return None

        result = pd.DataFrame()
        result["report_date"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
        result["notice_date"] = pd.to_datetime(df["NOTICE_DATE"], errors="coerce")
        result["bps"] = pd.to_numeric(df["BPS"], errors="coerce")  # 元
        result["eps"] = pd.to_numeric(df.get("EPSJB", df.get("基本每股收益", pd.Series(np.nan))), errors="coerce")
        result["revenue"] = pd.to_numeric(df["TOTALOPERATEREVE"], errors="coerce")  # 元
        result["net_profit"] = pd.to_numeric(df["PARENTNETPROFIT"], errors="coerce")  # 元
        # total_equity from BPS × total shares (we'll fill later)
        result["total_equity"] = np.nan
        result = result[result["report_date"].notna()].sort_values("report_date").reset_index(drop=True)
        return result
    except Exception as e:
        print(f"    EM: {type(e).__name__}")
        return None


def fetch_ths(symbol: str) -> Optional[pd.DataFrame]:
    """
    同花顺 THS 接口 — 所有列为object，需要手动解析单位
    Returns DataFrame with UNIFIED_COLS or None
    """
    try:
        code = code_to_numeric(symbol)
        df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        if df is None or len(df) == 0:
            return None

        col_map = {
            "报告期": "report_date",
            "基本每股收益": "eps",
            "每股净资产": "bps",
            "净利润": "net_profit",
            "营业总收入": "revenue",
        }
        result = pd.DataFrame()
        for cn, en in col_map.items():
            if cn in df.columns:
                result[en] = df[cn].astype(str).replace({"False": "", "True": "", "nan": ""})
            else:
                result[en] = ""

        # Parse all numeric columns with units
        for col in ["eps", "bps", "net_profit", "revenue"]:
            if col in result.columns:
                result[col] = result[col].apply(parse_number)

        result["report_date"] = pd.to_datetime(result["report_date"], errors="coerce")
        # THS has no notice_date → report + 120 days
        result["notice_date"] = result["report_date"] + pd.DateOffset(days=120)
        result["total_equity"] = np.nan
        result = result[result["report_date"].notna()].sort_values("report_date").reset_index(drop=True)
        return result
    except Exception as e:
        print(f"    THS: {type(e).__name__}")
        return None


def fetch_sina(symbol: str) -> Optional[pd.DataFrame]:
    """
    新浪 Sina 接口 — 三表联动, 计算 BPS/EPS
    Returns DataFrame with UNIFIED_COLS or None
    """
    try:
        sina_code = code_to_sina(symbol)

        # 资产负债表 → total_equity = 归属于母公司股东权益合计
        bs = ak.stock_financial_report_sina(stock=sina_code, symbol="资产负债表")
        if bs is None or len(bs) == 0:
            return None

        # Transpose: rows are financial items, columns are periods
        # bs index: item names; bs columns: period dates (YYYYMMDD)
        bs_t = bs.T  # now rows=periods, cols=financial items

        # Find total equity row
        equity_idx = None
        for idx in bs.index:
            if "归属" in str(idx) and "股东权益" in str(idx):
                equity_idx = idx
                break
        if equity_idx is None:
            # Fallback: 股东权益合计
            for idx in bs.index:
                if "股东权益合计" in str(idx) and "归属" not in str(idx):
                    equity_idx = idx
                    break
        if equity_idx is None:
            equity_idx = bs.index[0]  # use first available

        periods = [str(c) for c in bs.columns if str(c).isdigit() and len(str(c)) == 8]
        if not periods:
            return None

        # Build result
        records = []
        for period in sorted(periods):
            report_date = pd.to_datetime(period, format="%Y%m%d")
            total_eq = parse_number(str(bs[period].get(equity_idx, np.nan)))

            records.append({
                "report_date": report_date,
                "notice_date": report_date + pd.DateOffset(days=120),
                "bps": np.nan,  # computed below
                "eps": np.nan,
                "revenue": np.nan,  # from 利润表 below
                "net_profit": np.nan,
                "total_equity": total_eq,
            })

        result = pd.DataFrame(records)

        # 利润表 → revenue & net_profit
        try:
            pl = ak.stock_financial_report_sina(stock=sina_code, symbol="利润表")
            pl_t = pl.T
            revenue_idx = None
            profit_idx = None
            for idx in pl.index:
                if "营业总收入" in str(idx) and revenue_idx is None:
                    revenue_idx = idx
                if "归属" in str(idx) and "净利润" in str(idx) and profit_idx is None:
                    profit_idx = idx
            if profit_idx is None:
                for idx in pl.index:
                    if "净利润" in str(idx) and profit_idx is None:
                        profit_idx = idx

            for i, row in result.iterrows():
                period = row["report_date"].strftime("%Y%m%d")
                if period in pl.columns:
                    if revenue_idx and revenue_idx in pl.index:
                        result.at[i, "revenue"] = parse_number(str(pl[period].get(revenue_idx, np.nan)))
                    if profit_idx and profit_idx in pl.index:
                        result.at[i, "net_profit"] = parse_number(str(pl[period].get(profit_idx, np.nan)))
        except Exception:
            pass  # Profit table optional

        # Compute BPS/EPS from raw data + profile
        profile_path = PROF_DIR / f"{symbol}.parquet"
        if profile_path.exists():
            total_shares = pd.read_parquet(profile_path)["total_shares_yi"].iloc[0]
        else:
            total_shares = 1.0  # fallback, will produce bad bps

        result["bps"] = result["total_equity"] / (total_shares * 1e8)
        result["eps"] = result["net_profit"] / (total_shares * 1e8)

        return result[result["report_date"].notna()].sort_values("report_date").reset_index(drop=True)
    except Exception as e:
        print(f"    Sina: {type(e).__name__}: {str(e)[:80]}")
        return None


# ═══════════════════════════════════════════════════════════
# 验证
# ═══════════════════════════════════════════════════════════

def validate_financials(df: pd.DataFrame, symbol: str) -> List[str]:
    """返回问题列表, 无问题返回[]"""
    issues = []
    if len(df) < 4:
        issues.append(f"仅 {len(df)} 期, 可能不够")

    # BPS不能为负 (排除技术性负资产的 ST)
    neg_bps = (df["bps"] < 0).sum()
    if neg_bps > len(df) * 0.5:
        issues.append(f"BPS为负: {neg_bps}/{len(df)} 期")

    # 日期不单调
    if not df["report_date"].is_monotonic_increasing:
        issues.append("报告期不单调递增")

    # 营收不能全部为0/NaN (新上市可能前几期为0, 但太多就有问题)
    revenue_zero = (df["revenue"].fillna(0) <= 0).sum()
    if revenue_zero > len(df) * 0.3:
        issues.append(f"营收≤0: {revenue_zero}/{len(df)} 期")

    return issues


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def rebuild_mcap_pb(symbol: str):
    """对单个股票重建逐日 mcap/pb 面板"""
    from tools.backtest_mvp.data import DATA_DIR
    fin_path = FIN_DIR / f"{symbol}.parquet"
    prof_path = PROF_DIR / f"{symbol}.parquet"
    kline_path = DATA_DIR / f"{symbol}.parquet"
    out_path = MB_DIR / f"{symbol}.parquet"

    if not fin_path.exists() or not prof_path.exists() or not kline_path.exists():
        return False

    try:
        financials = pd.read_parquet(fin_path)
        profile = pd.read_parquet(prof_path)
        kline = pd.read_parquet(kline_path)
        kline["date"] = pd.to_datetime(kline["date"])
        kline = kline.sort_values("date")

        total_shares = profile["total_shares_yi"].iloc[0]

        # Build daily mcap/pb
        daily = kline[["date", "close"]].copy()
        daily["mcap"] = (daily["close"] * total_shares).clip(lower=0.01)

        # Forward-fill BPS by NOTICE_DATE (avoids look-ahead bias)
        financials = financials.sort_values("report_date")
        financials["notice_date"] = pd.to_datetime(financials["notice_date"])
        # Map: notice_date → latest bps available
        bps_map = financials[["notice_date", "bps"]].dropna(subset=["bps"])
        bps_map = bps_map.set_index("notice_date").sort_index()
        bps_map = bps_map[~bps_map.index.duplicated(keep="last")]

        if len(bps_map) > 0:
            # Merge to daily
            daily = daily.merge(bps_map, left_on="date", right_index=True, how="left")
            daily["bps"] = daily["bps"].ffill()
            daily["pb"] = (daily["close"] / daily["bps"]).clip(lower=0.01, upper=1000)
            daily["pe"] = np.nan
        else:
            daily["pb"] = np.nan
            daily["pe"] = np.nan

        out_cols = ["date", "close", "mcap", "pb", "pe"]
        available = [c for c in out_cols if c in daily.columns]
        daily[available].to_parquet(out_path, index=False)
        return True
    except Exception as e:
        print(f"    rebuild_mcap_pb error: {e}")
        return False


def main(args: List[str]):
    dry_run = "--dry-run" in args
    force_source = None
    for src in ["em", "ths", "sina"]:
        if f"--source" in args:
            idx = args.index("--source") + 1
            if idx < len(args):
                force_source = args[idx]

    # Find missing
    existing_fin = set(f.stem for f in FIN_DIR.glob("*.parquet"))
    all_kline = set(f.stem for f in DATA_DIR.glob("*.parquet"))
    existing_prof = set(f.stem for f in PROF_DIR.glob("*.parquet"))
    missing = sorted(all_kline - existing_fin)
    missing_prof = sorted(all_kline - existing_prof)

    print(f"K线: {len(all_kline)} 只 | 财务: {len(existing_fin)} 只 | 股本: {len(existing_prof)} 只")
    print(f"缺失财务: {len(missing)} 只")
    print(f"缺失股本: {len(missing_prof)} 只")
    if dry_run:
        print(f"\n  缺失财务样本: {missing[:10]}")
        return

    if not missing:
        print("✅ 财务数据已全覆盖")
        # Still rebuild mcap/pb
        missing_mb = sorted(all_kline - set(f.stem for f in MB_DIR.glob("*.parquet")))
        if missing_mb:
            print(f"重建 mcap/pb: {len(missing_mb)} 只...")
            for sym in missing_mb:
                ok = rebuild_mcap_pb(sym)
                if not ok:
                    print(f"  {sym}: mcap/pb 重建失败")
        return

    # Step 0: Fill missing profiles first
    if missing_prof:
        print(f"\n先补股本: {len(missing_prof)} 只...")
        for sym in missing_prof:
            prof_path = PROF_DIR / f"{sym}.parquet"
            if prof_path.exists():
                continue
            try:
                profile = ak.stock_profile_cninfo(symbol=sym[2:])
                yi = parse_number(str(profile["注册资金"].iloc[0])) / 1e4
                pd.DataFrame({"total_shares_yi": [yi]}).to_parquet(prof_path, index=False)
            except Exception as e:
                print(f"  {sym}: profile失败 ({type(e).__name__})")
            time.sleep(1.0)

    # Step 1: Fetch financials
    sources_used = {"EM": 0, "THS": 0, "Sina": 0, "FAIL": 0}
    validation_issues = {}

    print(f"\n开始拉取 {len(missing)} 只财务数据...")
    start = time.time()

    for i, sym in enumerate(missing):
        result = None
        source = "FAIL"

        # Priority: EM → THS → Sina (unless force_source)
        order = ["EM", "THS", "Sina"]
        if force_source:
            order = [force_source.upper()]

        for src in order:
            if src == "EM" and sym.startswith("bj"):
                continue  # EM doesn't support BJ
            fn = {"EM": fetch_em, "THS": fetch_ths, "Sina": fetch_sina}[src]
            result = fn(sym)
            if result is not None and len(result) >= 4:
                source = src
                break

        if result is not None and len(result) >= 4:
            result["symbol"] = sym
            result["source"] = source
            result.to_parquet(FIN_DIR / f"{sym}.parquet", index=False)
            sources_used[source] += 1

            # Validate
            issues = validate_financials(result, sym)
            if issues:
                validation_issues[sym] = issues
        else:
            sources_used["FAIL"] += 1

        # Progress
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(missing) - i - 1) / rate
            print(f"  [{i+1}/{len(missing)}] {elapsed:.0f}s, ETA {eta:.0f}s | "
                  f"EM={sources_used['EM']} THS={sources_used['THS']} "
                  f"Sina={sources_used['Sina']} FAIL={sources_used['FAIL']}")

        time.sleep(1.0)

    elapsed = time.time() - start
    print(f"\n完成! {elapsed:.0f}s")
    print(f"  EM: {sources_used['EM']} | THS: {sources_used['THS']} | "
          f"Sina: {sources_used['Sina']} | FAIL: {sources_used['FAIL']}")

    # Report validation issues
    if validation_issues:
        print(f"\n验证问题 ({len(validation_issues)} 只):")
        for sym, issues in sorted(validation_issues.items())[:10]:
            print(f"  {sym}: {', '.join(issues)}")
        if len(validation_issues) > 10:
            print(f"  ... 及其他 {len(validation_issues) - 10} 只")

    # Step 2: Rebuild mcap/pb for ALL missing
    all_kline_check = set(f.stem for f in DATA_DIR.glob("*.parquet"))
    existing_mb_check = set(f.stem for f in MB_DIR.glob("*.parquet"))
    to_rebuild = sorted(all_kline_check - existing_mb_check)

    print(f"\n重建 mcap/pb: {len(to_rebuild)} 只...")
    rebuilt = 0
    for sym in to_rebuild:
        ok = rebuild_mcap_pb(sym)
        if ok:
            rebuilt += 1
    print(f"  mcap/pb 重建: {rebuilt}/{len(to_rebuild)} 成功")

    # Final stats
    final_fin = len(set(f.stem for f in FIN_DIR.glob("*.parquet")))
    final_mb = len(set(f.stem for f in MB_DIR.glob("*.parquet")))
    final_prof = len(set(f.stem for f in PROF_DIR.glob("*.parquet")))
    print(f"\n最终覆盖: 财务={final_fin} | 股本={final_prof} | mcap/pb={final_mb}")


if __name__ == "__main__":
    main(sys.argv)
