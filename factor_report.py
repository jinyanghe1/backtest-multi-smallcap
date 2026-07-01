#!/usr/bin/env python3
"""CLI for evaluating factor-library ICs and a decorrelated composite."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.backtest_mvp.factors import compute_factors, load_daily_mcap_pb, load_price_data
from tools.backtest_mvp.factors.combiner import combine_factors
from tools.backtest_mvp.factors.factor_library import FACTOR_REGISTRY, compute_all_factors
from tools.backtest_mvp.data import DATA_DIR


def _synthetic_panel() -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    symbols = [f"S{i:03d}" for i in range(30)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    d = pd.Series(idx.get_level_values("date")).factorize()[0]
    s = pd.Series(idx.get_level_values("symbol")).factorize()[0]
    close = 100 + d * 0.05 + np.sin(s * 0.4 + d * 0.1)
    panel = pd.DataFrame({
        "close": close,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "volume": 1000 + s,
        "amount": close * (1000 + s),
        "vwap": close,
        "mcap": 10 + s,
        "pb": 1 + s / 100,
        "turnover": 1 + (s % 5) / 10,
        "shareholders": 10000 + s,
        "industry_code": s % 4,
    }, index=idx)
    future = panel["close"].groupby(level="symbol", group_keys=False).shift(-20)
    fwd_returns = future / panel["close"] - 1.0
    return panel, fwd_returns


def _load_real_panel() -> tuple[pd.DataFrame, pd.Series]:
    data = load_price_data(str(DATA_DIR))
    if data.empty:
        raise RuntimeError("No cached price data found. Run the existing data download first or use --synthetic.")
    mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
    factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)
    fwd_returns = return_panel["daily_return"].groupby(level="symbol", group_keys=False).shift(-20)
    return factor_panel, fwd_returns


def build_report(panel: pd.DataFrame, fwd_returns: pd.Series, method: str, max_corr: float):
    factors = compute_all_factors(panel)
    factor_values = {col: factors[col] for col in factors.columns}
    combined = combine_factors(factor_values, fwd_returns, method=method, max_corr=max_corr)
    rows = []
    for name in sorted(factor_values):
        rows.append({
            "factor": name,
            "function": getattr(FACTOR_REGISTRY.get(name), "__name__", ""),
            "ic": combined.ic.get(name, 0.0),
            "ic_ir": combined.ic_ir.get(name, 0.0),
            "selected": name in combined.selected,
            "weight": combined.weights.get(name, 0.0),
        })
    table = pd.DataFrame(rows).sort_values("ic_ir", key=lambda s: s.abs(), ascending=False)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "max_corr": max_corr,
        "n_factors": len(factor_values),
        "selected": combined.selected,
        "weights": combined.weights,
        "factors": rows,
        "corr_matrix": combined.corr_matrix.to_dict(),
    }
    return table, payload


def persist_report(table: pd.DataFrame, payload: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"factor_report_{stamp}.csv"
    json_path = out_dir / f"factor_report_{stamp}.json"
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate factor-library ICs and build an IC-weighted composite.")
    parser.add_argument("--synthetic", action="store_true", help="Run an offline deterministic self-test panel instead of cached real data.")
    parser.add_argument("--method", choices=["equal", "ic", "ic_ir"], default="ic_ir")
    parser.add_argument("--max-corr", type=float, default=0.6)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "reports")
    args = parser.parse_args(argv)

    panel, fwd_returns = _synthetic_panel() if args.synthetic else _load_real_panel()
    table, payload = build_report(panel, fwd_returns, args.method, args.max_corr)
    json_path, csv_path = persist_report(table, payload, args.out_dir)

    print(table[["factor", "ic", "ic_ir", "selected", "weight"]].to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
