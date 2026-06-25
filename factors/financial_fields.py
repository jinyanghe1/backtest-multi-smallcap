"""Minimal financial-field registry and derivation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FieldSpec:
    name: str
    unit: str
    source_priority: tuple[str, ...]
    formula: str | None = None
    requires: tuple[str, ...] = ()
    higher_is_better: bool | None = None
    status: str = "available"


FIELD_SPECS: dict[str, FieldSpec] = {
    "bps": FieldSpec("每股净资产", "元/股", ("eastmoney", "ths", "sina")),
    "eps": FieldSpec("基本每股收益", "元/股", ("eastmoney", "ths", "sina")),
    "revenue": FieldSpec("营业总收入", "元", ("eastmoney", "ths", "sina")),
    "net_profit": FieldSpec("归属净利润", "元", ("eastmoney", "ths", "sina")),
    "total_equity": FieldSpec("股东权益合计", "元", ("eastmoney", "sina", "ths")),
    "mcap": FieldSpec("总市值", "亿元", ("westock", "eastmoney"), "close * total_shares_yi"),
    "pb": FieldSpec("市净率", "ratio", ("eastmoney", "westock"), "close / bps"),
    "pe": FieldSpec("市盈率", "ratio", ("eastmoney", "westock"), "close / eps"),
    "gross_profit": FieldSpec("毛利润", "元", ("eastmoney", "ths", "sina")),
    "total_assets": FieldSpec("资产总计", "元", ("eastmoney", "sina", "ths")),
    "operating_cashflow": FieldSpec("经营活动现金流", "元", ("eastmoney", "ths", "sina")),
    "roe_ttm": FieldSpec("ROE(TTM)", "ratio", ("derived",), "net_profit_ttm / total_equity", ("net_profit", "total_equity"), True),
    "roa_ttm": FieldSpec("ROA(TTM)", "ratio", ("derived",), "net_profit_ttm / total_assets", ("net_profit", "total_assets"), True),
    "revenue_growth_ttm": FieldSpec("营收TTM增速", "ratio", ("derived",), "(revenue_ttm / revenue_ttm_lag4Q) - 1", ("revenue",), True),
    "profit_growth_ttm": FieldSpec("净利润TTM增速", "ratio", ("derived",), "(net_profit_ttm / net_profit_ttm_lag4Q) - 1", ("net_profit",), True),
    "gross_margin": FieldSpec("毛利率", "ratio", ("derived",), "gross_profit / revenue", ("gross_profit", "revenue"), True),
    "net_margin": FieldSpec("净利率", "ratio", ("derived",), "net_profit / revenue", ("net_profit", "revenue"), True),
    "asset_turnover": FieldSpec("资产周转率", "ratio", ("derived",), "revenue / total_assets", ("revenue", "total_assets"), True),
    "operating_cashflow_ttm": FieldSpec("经营现金流(TTM)", "元", ("derived",), "sum(operating_cashflow, 4Q)", ("operating_cashflow",), True),
    "operating_cashflow_to_revenue": FieldSpec("现金流营收比", "ratio", ("derived",), "operating_cashflow_ttm / revenue_ttm", ("operating_cashflow", "revenue"), True),
}


def get_field_spec(field: str) -> FieldSpec:
    try:
        return FIELD_SPECS[field]
    except KeyError as exc:
        raise KeyError(f"unknown financial field: {field}") from exc


def _add_ttm_fields(financials: pd.DataFrame) -> pd.DataFrame:
    df = financials.copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["notice_date"] = pd.to_datetime(df["notice_date"], errors="coerce")
    df = df.sort_values(["symbol", "report_date"])
    for col in ["revenue", "net_profit", "operating_cashflow"]:
        if col in df.columns:
            df[f"{col}_ttm"] = (
                df.groupby("symbol")[col]
                .rolling(4, min_periods=4)
                .sum()
                .reset_index(level=0, drop=True)
            )
            df[f"{col}_ttm_lag4Q"] = df.groupby("symbol")[f"{col}_ttm"].shift(4)
    if {"net_profit_ttm", "total_equity"}.issubset(df.columns):
        df["roe_ttm"] = df["net_profit_ttm"] / df["total_equity"].replace(0, np.nan)
    if {"net_profit_ttm", "total_assets"}.issubset(df.columns):
        df["roa_ttm"] = df["net_profit_ttm"] / df["total_assets"].replace(0, np.nan)
    if {"revenue_ttm", "revenue_ttm_lag4Q"}.issubset(df.columns):
        df["revenue_growth_ttm"] = df["revenue_ttm"] / df["revenue_ttm_lag4Q"].replace(0, np.nan) - 1
    if {"net_profit_ttm", "net_profit_ttm_lag4Q"}.issubset(df.columns):
        df["profit_growth_ttm"] = df["net_profit_ttm"] / df["net_profit_ttm_lag4Q"].replace(0, np.nan) - 1
    if {"gross_profit", "revenue"}.issubset(df.columns):
        df["gross_margin"] = df["gross_profit"] / df["revenue"].replace(0, np.nan)
    if {"net_profit", "revenue"}.issubset(df.columns):
        df["net_margin"] = df["net_profit"] / df["revenue"].replace(0, np.nan)
    if {"revenue", "total_assets"}.issubset(df.columns):
        df["asset_turnover"] = df["revenue"] / df["total_assets"].replace(0, np.nan)
    if {"operating_cashflow_ttm", "revenue_ttm"}.issubset(df.columns):
        df["operating_cashflow_to_revenue"] = df["operating_cashflow_ttm"] / df["revenue_ttm"].replace(0, np.nan)
    return df


def derive_financial_fields(
    financials: pd.DataFrame,
    prices: pd.DataFrame,
    fields: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Align financial fields to trade dates using notice_date only."""
    if financials.empty or prices.empty:
        return pd.DataFrame()
    requested = fields or [name for name, spec in FIELD_SPECS.items() if spec.status == "available"]
    fin = _add_ttm_fields(financials)
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")

    records = []
    for symbol, sym_px in px.sort_values("date").groupby("symbol"):
        sym_fin = fin[fin["symbol"] == symbol].sort_values("notice_date")
        if sym_fin.empty:
            merged = sym_px.copy()
            for field in requested:
                if field not in merged.columns:
                    merged[field] = np.nan
            records.append(merged)
            continue

        keep_cols = ["notice_date", *[c for c in requested if c in sym_fin.columns]]
        sym_fin = sym_fin[keep_cols].dropna(subset=["notice_date"])
        merged = pd.merge_asof(
            sym_px.sort_values("date"),
            sym_fin.sort_values("notice_date"),
            left_on="date",
            right_on="notice_date",
            direction="backward",
        )
        if "mcap" in requested and "mcap" not in merged.columns and {"close", "total_shares_yi"}.issubset(merged.columns):
            merged["mcap"] = merged["close"] * merged["total_shares_yi"]
        if "pb" in requested and "pb" not in merged.columns and {"close", "bps"}.issubset(merged.columns):
            merged["pb"] = merged["close"] / merged["bps"].replace(0, np.nan)
        if "pe" in requested and "pe" not in merged.columns and {"close", "eps"}.issubset(merged.columns):
            merged["pe"] = merged["close"] / merged["eps"].replace(0, np.nan)
        for field in requested:
            if field not in merged.columns:
                merged[field] = np.nan
        records.append(merged)

    result = pd.concat(records, ignore_index=True)
    output_cols = ["symbol", "date", *requested]
    return result[[c for c in output_cols if c in result.columns]]

