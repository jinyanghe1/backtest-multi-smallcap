#!/usr/bin/env python3
"""
P1 因子挖掘数据采集脚本
采集: 行业分类、股东户数、业绩预告、财务现金流

Usage:
    python data/collect_p1_data.py

Output:
    data_cache/industry_classification.csv
    data_cache/shareholder_count.csv
    data_cache/earnings_forecast.csv
    data_cache/financial_cashflow.csv
"""

import sys, os, json, time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import pandas as pd
import akshare as ak

# ── paths ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # backtest_mvp/
CACHE = ROOT / "data_cache"
CACHE.mkdir(parents=True, exist_ok=True)
LOG = CACHE / "collect_p1_log.json"


def _log(task: str, status: str, rows: int = 0, msg: str = ""):
    """记录采集日志"""
    entry = {"task": task, "status": status, "rows": rows, "msg": msg, "ts": datetime.now().isoformat()}
    logs = []
    if LOG.exists():
        try:
            logs = json.loads(LOG.read_text(encoding="utf-8"))
        except:
            pass
    logs.append(entry)
    LOG.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [{task}] {status}: {rows} rows {msg}")


# ══════════════════════════════════════════════
# D1. 行业分类 (Industry Classification)
# ══════════════════════════════════════════════
def collect_industry() -> pd.DataFrame:
    """采集行业分类数据。尝试多个接口，失败时降级。"""
    
    # Strategy 1: 尝试 stock_industry_pe_ratio_cninfo (巨潮)
    try:
        df = ak.stock_industry_pe_ratio_cninfo(symbol="证监会行业分类", date="20240601")
        if len(df) > 0 and len(df.columns) > 2:
            cols = df.columns.tolist()
            if any(c.startswith('20') for c in cols[:3]):
                raise ValueError("Wrong format: columns are dates")
            df = df.rename(columns={
                "证券代码": "symbol",
                "证券简称": "name",
                "行业分类": "industry_1",
                "行业分类编码": "industry_code",
            })
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].astype(str).str.zfill(6)
                df["symbol"] = df["symbol"].apply(lambda x: f"sh{x}" if x.startswith("6") else f"sz{x}")
                df["industry_2"] = ""
                df["industry_3"] = ""
                df["update_date"] = datetime.now().strftime("%Y-%m-%d")
                df = df[["symbol", "name", "industry_1", "industry_2", "industry_3", "industry_code", "update_date"]]
                _log("industry", "OK", len(df), "from stock_industry_pe_ratio_cninfo")
                return df
    except Exception as e:
        print(f"  Strategy 1 failed: {e}")
    
    # Fallback: 使用简单分类（按代码前缀）
    listing = pd.read_csv(CACHE / "listing_dates.csv", encoding="utf-8-sig")
    df = pd.DataFrame()
    df["symbol"] = listing["symbol"]
    df["name"] = listing["name"]
    
    def _simple_industry(code: str) -> str:
        prefix = code[2:4] if len(code) >= 4 else "00"
        industry_map = {
            "60": "金融地产", "00": "制造业", "30": "科技", "68": "科技",
            "88": "综合", "89": "综合", "20": "制造业", "002": "制造业",
            "300": "科技", "301": "科技", "688": "科技", "689": "科技",
        }
        return industry_map.get(prefix, "其他")
    
    df["industry_1"] = df["symbol"].apply(_simple_industry)
    df["industry_2"] = ""
    df["industry_3"] = ""
    df["industry_code"] = ""
    df["update_date"] = datetime.now().strftime("%Y-%m-%d")
    df = df[["symbol", "name", "industry_1", "industry_2", "industry_3", "industry_code", "update_date"]]
    
    _log("industry", "FALLBACK", len(df), "simple prefix-based classification")
    return df


# ══════════════════════════════════════════════
# D2. 股东户数 (Shareholder Count)
# ══════════════════════════════════════════════
def collect_shareholder_count() -> pd.DataFrame:
    """采集股东户数数据。"""
    try:
        df = ak.stock_zh_a_gdhs()
        df = df.rename(columns={
            "代码": "symbol",
            "名称": "name",
            "股东户数-本次": "shareholder_count",
            "股东户数-上次": "shareholder_count_prev",
            "股东户数-增减": "shareholder_change",
            "股东户数-增减比例": "change_rate_pct",
            "股东户数统计截止日-本次": "end_date",
            "股东户数统计截止日-上次": "end_date_prev",
            "户均持股市值": "avg_market_value",
            "户均持股数量": "avg_shares_per_household",
            "公告日期": "announce_date",
        })
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        df["symbol"] = df["symbol"].apply(lambda x: f"sh{x}" if x.startswith("6") else f"sz{x}")
        df["shareholder_count"] = pd.to_numeric(df["shareholder_count"], errors="coerce")
        df["change_rate_pct"] = pd.to_numeric(df["change_rate_pct"], errors="coerce")
        df["avg_shares_per_household"] = pd.to_numeric(df["avg_shares_per_household"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df[["symbol", "name", "end_date", "shareholder_count", 
                 "change_rate_pct", "avg_shares_per_household", "announce_date"]]
        _log("shareholder_count", "OK", len(df), "from stock_zh_a_gdhs")
        return df
    except Exception as e:
        _log("shareholder_count", "FAIL", 0, str(e))
        return pd.DataFrame(columns=["symbol", "name", "end_date", "shareholder_count", 
                                      "change_rate_pct", "avg_shares_per_household", "announce_date"])


# ══════════════════════════════════════════════
# D3. 业绩预告 (Earnings Forecast)
# ══════════════════════════════════════════════
def collect_earnings_forecast() -> pd.DataFrame:
    """采集业绩预告数据。采集最近 4 个季度。"""
    quarters = []
    for year in [2024, 2025]:
        for q in ["0331", "0630", "0930", "1231"]:
            if year == 2025 and q == "1231":
                continue
            quarters.append(f"{year}{q}")
    
    all_records = []
    for quarter in quarters:
        try:
            df = ak.stock_yjyg_em(date=quarter)
            if len(df) == 0:
                continue
            df = df.rename(columns={
                "股票代码": "symbol",
                "股票简称": "name",
                "预测指标": "forecast_indicator",
                "业绩变动": "forecast_summary",
                "预测数值": "forecast_value",
                "业绩变动幅度": "yoy_change_pct",
                "预告类型": "forecast_type",
                "上年同期值": "prev_year_value",
                "公告日期": "announce_date",
            })
            df["symbol"] = df["symbol"].astype(str).str.zfill(6)
            df["symbol"] = df["symbol"].apply(lambda x: f"sh{x}" if x.startswith("6") else f"sz{x}")
            df["report_date"] = pd.to_datetime(quarter, format="%Y%m%d").strftime("%Y-%m-%d")
            df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df["forecast_lower"] = pd.to_numeric(df["forecast_value"], errors="coerce")
            df["forecast_upper"] = df["forecast_lower"]
            df["yoy_lower"] = pd.to_numeric(df["yoy_change_pct"], errors="coerce")
            df["yoy_upper"] = df["yoy_lower"]
            df = df[["symbol", "name", "report_date", "forecast_type", "forecast_summary",
                     "forecast_lower", "forecast_upper", "yoy_lower", "yoy_upper", "announce_date"]]
            all_records.append(df)
            print(f"  Quarter {quarter}: {len(df)} records")
        except Exception as e:
            print(f"  Quarter {quarter} failed: {e}")
    
    if all_records:
        result = pd.concat(all_records, ignore_index=True)
        _log("earnings_forecast", "OK", len(result), f"{len(quarters)} quarters")
        return result
    else:
        _log("earnings_forecast", "FAIL", 0, "all quarters failed")
        return pd.DataFrame(columns=["symbol", "name", "report_date", "forecast_type", 
                                      "forecast_summary", "forecast_lower", "forecast_upper",
                                      "yoy_lower", "yoy_upper", "announce_date"])


# ══════════════════════════════════════════════
# D4. 财务现金流 (Financial Cashflow) - SIMPLIFIED
# ══════════════════════════════════════════════
def collect_financial_cashflow() -> pd.DataFrame:
    """采集财务现金流数据。"""
    # 简化版：从 stock_financial_abstract 提取关键指标
    # 由于单只查询太慢，我们只采样部分股票
    try:
        listing = pd.read_csv(CACHE / "listing_dates.csv", encoding="utf-8-sig")
        # 只采样前 50 只 + 10 只随机
        sample = pd.concat([
            listing.head(50),
            listing.sample(min(10, len(listing)), random_state=42)
        ]).drop_duplicates(subset=["symbol"])
        
        all_records = []
        for _, row in sample.iterrows():
            try:
                code = row["symbol"][2:]  # 去掉 sh/sz
                df = ak.stock_financial_abstract(symbol=code)
                if len(df) == 0:
                    continue
                # 查找 经营现金流量净额 行 (akshare 返回的指标名是"经营现金流量净额", 不是"经营活动现金流量净额")
                cf_row = df[df["指标"] == "经营现金流量净额"]
                if len(cf_row) == 0:
                    continue
                # 提取最新 4 个季度的值
                cols = [c for c in df.columns if c.startswith("20")]
                cols = sorted(cols, reverse=True)[:4]  # 最近 4 个季度
                for col in cols:
                    val = pd.to_numeric(cf_row[col].values[0], errors="coerce")
                    if pd.notna(val):
                        all_records.append({
                            "symbol": row["symbol"],
                            "report_date": col,
                            "operating_cashflow": val,
                        })
            except Exception as e:
                continue
        
        if all_records:
            result = pd.DataFrame(all_records)
            _log("financial_cashflow", "OK", len(result), f"sample {len(sample)} stocks")
            return result
        else:
            _log("financial_cashflow", "FAIL", 0, "no records extracted")
            return pd.DataFrame(columns=["symbol", "report_date", "operating_cashflow"])
    except Exception as e:
        _log("financial_cashflow", "FAIL", 0, str(e))
        return pd.DataFrame(columns=["symbol", "report_date", "operating_cashflow"])


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  P1 数据采集")
    print("=" * 60)
    
    # D1: 行业分类
    print("\n[D1] 行业分类...")
    df_ind = collect_industry()
    df_ind.to_csv(CACHE / "industry_classification.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved: {CACHE / 'industry_classification.csv'} ({len(df_ind)} rows)")
    
    # D2: 股东户数
    print("\n[D2] 股东户数...")
    df_sh = collect_shareholder_count()
    df_sh.to_csv(CACHE / "shareholder_count.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved: {CACHE / 'shareholder_count.csv'} ({len(df_sh)} rows)")
    
    # D3: 业绩预告
    print("\n[D3] 业绩预告...")
    df_ef = collect_earnings_forecast()
    df_ef.to_csv(CACHE / "earnings_forecast.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved: {CACHE / 'earnings_forecast.csv'} ({len(df_ef)} rows)")
    
    # D4: 财务现金流 (简化版)
    print("\n[D4] 财务现金流...")
    df_cf = collect_financial_cashflow()
    df_cf.to_csv(CACHE / "financial_cashflow.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved: {CACHE / 'financial_cashflow.csv'} ({len(df_cf)} rows)")
    
    print("\n" + "=" * 60)
    print("  P1 数据采集完成")
    print("=" * 60)
    print(f"\n  行业分类: {len(df_ind)} 行")
    print(f"  股东户数: {len(df_sh)} 行")
    print(f"  业绩预告: {len(df_ef)} 行")
    print(f"  财务现金流: {len(df_cf)} 行")
    print(f"\n  日志: {LOG}")
