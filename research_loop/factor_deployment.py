"""因子部署裁决：把因子库 → 组合器 → 过拟合诊断串成端到端评估（roadmap WS-C 收口）。

动机
----
Phase 2/4 挖了 43 个因子，Phase 3 造了严谨的 PBO / DSR（``overfitting.py``），
但二者从未打通。本模块回答量化研究最要命的问题：

    "在挖了 43 个因子、只报告 IC 最高的那个之后，它到底是真 alpha 还是运气最优？"

流程
----
1. 对每个因子构造 **多空日频收益序列**（按截面因子值分位，做多头分位、做空尾分位，
   按 IC 符号对齐方向），堆叠成 T×N 收益矩阵。
2. 把 T×N 矩阵喂给 :func:`probability_of_backtest_overfitting` → **PBO / CSCV**：
   样本内最优因子在样本外是否退化。
3. 把 N 个因子的单期 Sharpe 喂给 :func:`deflated_sharpe_from_trials` → **DSR**：
   把 43 个因子当 43 次试验，扣除多重检验选择偏差后最优 Sharpe 是否仍显著。
4. 用组合器合成去相关 IC 加权 composite，算它自己的多空 Sharpe 与 PSR。
5. 汇总成 :class:`DeploymentVerdict`（含人类可读裁决）。

设计
----
- 纯计算，无 I/O 副作用；核心函数接受 ``(factor_values, fwd_returns)``，与真实/合成
  数据无关，便于单测。
- 复用 :mod:`combiner`（IC / IC-IR / 去相关 / composite）与
  :mod:`overfitting`（PBO / DSR / PSR），不重复造轮子。
- 零新增依赖（numpy / pandas / scipy 均已在用）。

重叠收益提示
------------
若 ``fwd_returns`` 是 k 日前视收益，逐日构造的多空序列高度重叠、自相关，会低估
Sharpe 标准误。用 ``resample_step=k`` 取非重叠子样本可缓解（真实运行推荐）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tools.backtest_mvp.factors.combiner import CombineResult, combine_factors
from tools.backtest_mvp.research_loop.overfitting import (
    deflated_sharpe_from_trials,
    probability_of_backtest_overfitting,
    probabilistic_sharpe_ratio,
    sharpe_moments,
)


def _date_level(index: pd.Index) -> str | int | None:
    if isinstance(index, pd.MultiIndex):
        return "date" if "date" in index.names else 0
    return None


def _single_period_sharpe(returns: pd.Series) -> float:
    """单期（非年化）Sharpe；样本 < 2 或零方差记 0。"""
    r = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    std = r.std(ddof=1)
    return float(r.mean() / std) if std > 0 else 0.0


def factor_long_short_returns(
    factor: pd.Series,
    fwd_returns: pd.Series,
    *,
    quantile: float = 0.3,
    sign: float = 1.0,
    min_names: int = 6,
    resample_step: int = 1,
) -> pd.Series:
    """单因子的逐期多空收益序列（按日期截面分位）。

    每个日期：按 ``sign * factor`` 排序，做多前 ``quantile`` 分位、做空后 ``quantile``
    分位，等权，收益 = 多头均值前视收益 − 空头均值前视收益。

    Args:
        factor: 因子值 Series（MultiIndex date, symbol）。
        fwd_returns: 与 ``factor`` 同索引的前视收益。
        quantile: 头/尾分位比例（0<q<=0.5）。
        sign: 方向（+1 或 −1），用于把负 IC 因子翻正。
        min_names: 单日最少可用标的数，不足则跳过该日。
        resample_step: 每隔 step 个日期取一次（>1 缓解重叠收益自相关）。

    Returns:
        以日期为索引的多空收益 Series（可能为空）。
    """
    if not 0.0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    df = pd.DataFrame(
        {
            "f": sign * pd.to_numeric(factor, errors="coerce"),
            "r": pd.to_numeric(fwd_returns, errors="coerce"),
        }
    ).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    level = _date_level(df.index)
    if level is None:
        return pd.Series(dtype=float)

    out: dict = {}
    for date, g in df.groupby(level=level):
        if len(g) < min_names or g["f"].nunique() < 2:
            continue
        k = max(1, int(round(len(g) * quantile)))
        ordered = g.sort_values("f")
        short_leg = ordered["r"].iloc[:k].mean()
        long_leg = ordered["r"].iloc[-k:].mean()
        if pd.notna(long_leg) and pd.notna(short_leg):
            out[date] = float(long_leg - short_leg)
    if not out:
        return pd.Series(dtype=float)
    series = pd.Series(out).sort_index()
    if resample_step > 1:
        series = series.iloc[::resample_step]
    return series


def build_returns_matrix(
    factor_values: dict[str, pd.Series],
    fwd_returns: pd.Series,
    *,
    ic: dict[str, float] | None = None,
    quantile: float = 0.3,
    min_names: int = 6,
    resample_step: int = 1,
    min_periods: int = 10,
    min_coverage: float = 0.6,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """构造对齐的 T×N 多空收益矩阵 + 每因子 Sharpe + 方向。

    方向按 ``ic`` 符号对齐（缺省用截面 Spearman IC 现算），使每个因子朝其"预期
    有效"方向，从而 DSR 的"最优试验"是各因子的最优取向。

    为避免少数低覆盖因子（warmup 长/中途缺失）通过 ``dropna(how="any")`` 把公共
    期数拖垮，先按 ``min_coverage`` 剔除覆盖不足的因子，再取剩余因子的公共日期。

    Returns:
        (returns_df, sharpes, signs)：
        returns_df 行=公共日期、列=因子名（丢弃有效期数 < min_periods 或覆盖
        < min_coverage 的因子）。
    """
    from tools.backtest_mvp.factors.combiner import _spearman_ic  # 局部导入避免暴露

    signs: dict[str, float] = {}
    series_map: dict[str, pd.Series] = {}
    for name, factor in factor_values.items():
        if ic is not None and name in ic:
            ic_val = ic[name]
        else:
            ic_val, _, _, _ = _spearman_ic(
                pd.to_numeric(factor, errors="coerce"), fwd_returns
            )
        sign = -1.0 if ic_val < 0 else 1.0
        signs[name] = sign
        # 逐因子先不下采样：各因子 warmup 长度不同，若在各自序列上 iloc[::step]
        # 会错开绝对日期相位，导致跨因子对齐后交集为空。改在对齐后的矩阵上统一下采样。
        ls = factor_long_short_returns(
            factor,
            fwd_returns,
            quantile=quantile,
            sign=sign,
            min_names=min_names,
            resample_step=1,
        )
        if len(ls) >= min_periods:
            series_map[name] = ls

    if not series_map:
        return pd.DataFrame(dtype=float), {}, signs

    # 覆盖率过滤：在全体因子的日期并集上，剔除覆盖 < min_coverage 的稀疏因子，
    # 让剩余因子的公共交集尽量大（保住 PBO 所需的期数）。
    union = sorted(set().union(*[set(s.index) for s in series_map.values()]))
    aligned = pd.DataFrame({name: s.reindex(union) for name, s in series_map.items()})
    coverage = aligned.notna().mean()
    keep = list(coverage[coverage >= min_coverage].index)
    if not keep:
        keep = list(aligned.columns)
    returns_df = aligned[keep].dropna(how="any")
    if resample_step > 1 and len(returns_df) > 0:
        returns_df = returns_df.iloc[::resample_step]
    sharpes = {name: _single_period_sharpe(returns_df[name]) for name in returns_df.columns}
    return returns_df, sharpes, signs


@dataclass
class DeploymentVerdict:
    """43 因子 → 部署裁决的完整诊断结果。"""

    n_factors: int
    n_periods: int
    pbo: float | None
    performance_degradation_slope: float | None
    prob_oos_loss: float | None
    best_factor: str
    best_sharpe: float
    dsr: float
    expected_max_sharpe: float
    psr_best_vs_zero: float
    sr_variance_across_factors: float
    composite_sharpe: float
    composite_psr: float
    selected: list[str] = field(default_factory=list)
    per_factor: pd.DataFrame = field(default_factory=pd.DataFrame)
    report_text: str = ""

    def to_dict(self) -> dict:
        d = {
            "n_factors": self.n_factors,
            "n_periods": self.n_periods,
            "pbo": self.pbo,
            "performance_degradation_slope": self.performance_degradation_slope,
            "prob_oos_loss": self.prob_oos_loss,
            "best_factor": self.best_factor,
            "best_sharpe": self.best_sharpe,
            "dsr": self.dsr,
            "expected_max_sharpe": self.expected_max_sharpe,
            "psr_best_vs_zero": self.psr_best_vs_zero,
            "sr_variance_across_factors": self.sr_variance_across_factors,
            "composite_sharpe": self.composite_sharpe,
            "composite_psr": self.composite_psr,
            "selected": self.selected,
            "per_factor": self.per_factor.to_dict(orient="records"),
        }
        return d


def _largest_even_leq(n: int) -> int:
    return n if n % 2 == 0 else n - 1


def _deployment_verdict_text(v: "DeploymentVerdict") -> str:
    if v.dsr >= 0.95:
        dsr_line = "稳健：扣除多重检验后最优因子仍高度显著"
    elif v.dsr >= 0.90:
        dsr_line = "尚可：去膨胀后边际显著，谨慎采纳"
    else:
        dsr_line = "存疑：大概率是 N 次试验里运气最优，勿直接部署单因子"

    if v.pbo is None:
        pbo_line = "样本期数不足，PBO 未计算"
    elif v.pbo <= 0.10:
        pbo_line = f"低过拟合风险（PBO={v.pbo:.3f}）"
    elif v.pbo <= 0.50:
        pbo_line = f"中等过拟合风险（PBO={v.pbo:.3f}）"
    else:
        pbo_line = f"高过拟合风险（PBO={v.pbo:.3f}），IS 最优多在 OOS 退化"

    if v.dsr >= 0.90 and (v.pbo is None or v.pbo <= 0.20):
        deploy = "建议：可推进组合部署（优先用去相关 composite 而非单因子）"
    elif v.composite_psr >= 0.90 and v.composite_sharpe > v.best_sharpe * 0.6:
        deploy = "建议：单因子存疑，但 composite 表现稳健，走 composite 路线"
    else:
        deploy = "建议：暂缓部署，补充样本 / 降低搜索空间 / 做 walk-forward 复核"

    lines = [
        "═══ 因子部署裁决 (roadmap WS-C 收口) ═══",
        f"  因子数 N / 期数 T      : {v.n_factors} / {v.n_periods}",
        f"  最优因子               : {v.best_factor}  (单期 Sharpe {v.best_sharpe:.4f})",
        f"  跨因子 Sharpe 方差     : {v.sr_variance_across_factors:.6f}",
        f"  E[max] 去膨胀基准      : {v.expected_max_sharpe:.4f}",
        f"  PSR(best vs 0)         : {v.psr_best_vs_zero:.4f}",
        f"  DSR(best vs E[max])    : {v.dsr:.4f}  → {dsr_line}",
    ]
    if v.pbo is not None:
        lines.append(
            f"  PBO / OOS亏损概率      : {v.pbo:.4f} / {v.prob_oos_loss:.4f}  → {pbo_line}"
        )
        lines.append(f"  性能衰减斜率(OOS~IS)   : {v.performance_degradation_slope:.4f}")
    else:
        lines.append(f"  PBO                    : —  ({pbo_line})")
    lines.append(
        f"  Composite Sharpe/PSR   : {v.composite_sharpe:.4f} / {v.composite_psr:.4f}"
        f"  (选入 {len(v.selected)} 个因子)"
    )
    lines.append(f"  部署裁决               : {deploy}")
    return "\n".join(lines)


def evaluate_deployment(
    factor_values: dict[str, pd.Series],
    fwd_returns: pd.Series,
    *,
    method: str = "ic_ir",
    max_corr: float = 0.6,
    quantile: float = 0.3,
    min_names: int = 6,
    resample_step: int = 1,
    n_partitions: int = 16,
) -> DeploymentVerdict:
    """端到端因子部署裁决：IC → 多空矩阵 → PBO/DSR → composite → 汇总。

    Args:
        factor_values: {因子名: 因子 Series}。
        fwd_returns: 前视收益（与因子同索引）。
        method / max_corr: 传给 :func:`combine_factors`。
        quantile / min_names / resample_step: 传给多空收益构造。
        n_partitions: PBO 的 CSCV 时间块数（偶数）；期数不足时自动缩减。

    Returns:
        :class:`DeploymentVerdict`。因子为空或无有效多空序列时返回空裁决。
    """
    n_input = len(factor_values)
    if n_input == 0:
        v = DeploymentVerdict(
            n_factors=0, n_periods=0, pbo=None, performance_degradation_slope=None,
            prob_oos_loss=None, best_factor="", best_sharpe=0.0, dsr=0.0,
            expected_max_sharpe=0.0, psr_best_vs_zero=0.0,
            sr_variance_across_factors=0.0, composite_sharpe=0.0, composite_psr=0.0,
        )
        v.report_text = "═══ 因子部署裁决 ═══\n  无因子输入。"
        return v

    combined: CombineResult = combine_factors(
        factor_values, fwd_returns, method=method, max_corr=max_corr
    )

    returns_df, sharpes, signs = build_returns_matrix(
        factor_values,
        fwd_returns,
        ic=combined.ic,
        quantile=quantile,
        min_names=min_names,
        resample_step=resample_step,
    )

    n_periods = int(returns_df.shape[0])
    n_factors = int(returns_df.shape[1])

    # 每因子诊断表
    rows = []
    for name in factor_values:
        rows.append(
            {
                "factor": name,
                "ic": float(combined.ic.get(name, 0.0)),
                "ic_ir": float(combined.ic_ir.get(name, 0.0)),
                "sign": float(signs.get(name, 1.0)),
                "sharpe": float(sharpes.get(name, 0.0)),
                "selected": name in combined.selected,
                "weight": float(combined.weights.get(name, 0.0)),
            }
        )
    per_factor = pd.DataFrame(rows).sort_values(
        "sharpe", key=lambda s: s.abs(), ascending=False, ignore_index=True
    )

    # 单因子 DSR（把 N 个因子当 N 次试验）
    if sharpes:
        best_factor = max(sharpes, key=lambda k: sharpes[k])
        best_sharpe = float(sharpes[best_factor])
        trial_sharpes = list(sharpes.values())
        best_series = returns_df[best_factor]
        try:
            mom = sharpe_moments(best_series.to_numpy(dtype=float))
            skew, kurt = mom["skew"], mom["kurtosis"]
        except ValueError:
            skew, kurt = 0.0, 3.0
        dsr_info = deflated_sharpe_from_trials(
            trial_sharpes, n_obs=max(n_periods, 2), skew=skew, kurt=kurt
        )
        dsr = float(dsr_info["dsr"])
        emax = float(dsr_info["expected_max_sharpe"])
        psr0 = float(dsr_info["psr_vs_zero"])
        sr_var = float(dsr_info["sr_variance"])
    else:
        best_factor, best_sharpe = "", 0.0
        dsr = emax = psr0 = sr_var = 0.0

    # PBO / CSCV（期数够才跑）
    pbo = slope = prob_oos_loss = None
    if n_factors >= 2 and n_periods >= 4:
        S = _largest_even_leq(min(n_partitions, n_periods))
        if S >= 2:
            pbo_res = probability_of_backtest_overfitting(
                returns_df.to_numpy(dtype=float), n_partitions=S
            )
            pbo = float(pbo_res["pbo"])
            slope = float(pbo_res["performance_degradation_slope"])
            prob_oos_loss = float(pbo_res["prob_oos_loss"])

    # composite 多空表现
    composite_sharpe = composite_psr = 0.0
    if len(combined.composite) > 0:
        comp_ls = factor_long_short_returns(
            combined.composite,
            fwd_returns,
            quantile=quantile,
            sign=1.0,  # composite 已按各因子 IC 对齐符号
            min_names=min_names,
            resample_step=resample_step,
        )
        composite_sharpe = _single_period_sharpe(comp_ls)
        if len(comp_ls) >= 2:
            try:
                cm = sharpe_moments(comp_ls.to_numpy(dtype=float))
                composite_psr = probabilistic_sharpe_ratio(
                    composite_sharpe, 0.0, cm["n_obs"], skew=cm["skew"], kurt=cm["kurtosis"]
                )
            except ValueError:
                composite_psr = 0.0

    verdict = DeploymentVerdict(
        n_factors=n_factors,
        n_periods=n_periods,
        pbo=pbo,
        performance_degradation_slope=slope,
        prob_oos_loss=prob_oos_loss,
        best_factor=best_factor,
        best_sharpe=best_sharpe,
        dsr=dsr,
        expected_max_sharpe=emax,
        psr_best_vs_zero=psr0,
        sr_variance_across_factors=sr_var,
        composite_sharpe=float(composite_sharpe),
        composite_psr=float(composite_psr),
        selected=list(combined.selected),
        per_factor=per_factor,
    )
    verdict.report_text = _deployment_verdict_text(verdict)
    return verdict


# ─────────────────────────────── CLI ───────────────────────────────

def _bounded_real_inputs(
    max_symbols: int, lookback_days: int, fwd_period: int
) -> tuple[dict[str, pd.Series], pd.Series]:
    """加载真实缓存数据的有界子集，返回 (factor_values, fwd_returns)。

    取成交额最高的 ``max_symbols`` 只、最近 ``lookback_days`` 个交易日，避免全量
    universe 上跑 43 因子（含 F043 嵌套回归）过慢。前视收益用 ``fwd_period`` 日累计。
    """
    from tools.backtest_mvp.data import DATA_DIR
    from tools.backtest_mvp.factors import (
        compute_factors,
        load_daily_mcap_pb,
        load_price_data,
    )
    from tools.backtest_mvp.factors.factor_library import compute_all_factors

    data = load_price_data(str(DATA_DIR))
    if data.empty:
        raise RuntimeError("无缓存价格数据；先运行数据下载或改用 --synthetic。")
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    top = list(data.groupby("symbol")["amount"].sum().sort_values(ascending=False).head(max_symbols).index)
    sub = data[data["symbol"].isin(top)]
    keep_dates = sorted(sub["date"].unique())[-lookback_days:]
    sub = sub[sub["date"].isin(keep_dates)]

    factor_panel, _ = compute_factors(sub, mcap_pb_data=load_daily_mcap_pb(str(DATA_DIR)))
    close = pd.to_numeric(factor_panel["close"], errors="coerce")
    future = close.groupby(level="symbol", group_keys=False).shift(-fwd_period)
    fwd_returns = future / close - 1.0

    factors = compute_all_factors(factor_panel, log_errors=False)
    factor_values = {c: factors[c] for c in factors.columns}
    return factor_values, fwd_returns


def _synthetic_inputs() -> tuple[dict[str, pd.Series], pd.Series]:
    rng = np.random.RandomState(7)
    dates = pd.date_range("2024-01-01", periods=140, freq="B")
    symbols = [f"S{i:03d}" for i in range(40)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    sym = pd.Series(idx.get_level_values("symbol")).factorize()[0]
    base = (sym - sym.mean()) / (sym.std() + 1e-9) + rng.normal(0, 0.05, len(idx))
    fwd = pd.Series(0.03 * base + rng.normal(0, 0.01, len(idx)), index=idx)
    fvals = {"real_alpha": pd.Series(base, index=idx)}
    for i in range(9):
        fvals[f"noise{i}"] = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    return fvals, fwd


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json
    from datetime import datetime
    from pathlib import Path

    p = argparse.ArgumentParser(description="因子部署裁决：PBO / DSR / composite 端到端诊断。")
    p.add_argument("--synthetic", action="store_true", help="用合成面板离线自检，不读真实数据。")
    p.add_argument("--max-symbols", type=int, default=80)
    p.add_argument("--lookback-days", type=int, default=400)
    p.add_argument("--fwd", type=int, default=5, help="前视收益日数（也用作 resample_step 去重叠）。")
    p.add_argument("--quantile", type=float, default=0.2)
    p.add_argument("--method", choices=["equal", "ic", "ic_ir"], default="ic_ir")
    p.add_argument("--max-corr", type=float, default=0.6)
    p.add_argument("--n-partitions", type=int, default=14)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "reports")
    args = p.parse_args(argv)

    if args.synthetic:
        factor_values, fwd_returns = _synthetic_inputs()
        step = 1
    else:
        factor_values, fwd_returns = _bounded_real_inputs(
            args.max_symbols, args.lookback_days, args.fwd
        )
        step = max(1, args.fwd)

    verdict = evaluate_deployment(
        factor_values,
        fwd_returns,
        method=args.method,
        max_corr=args.max_corr,
        quantile=args.quantile,
        resample_step=step,
        n_partitions=args.n_partitions,
    )

    print(verdict.report_text)
    print("\n── 单因子诊断（按 |Sharpe| 前 15）──")
    cols = ["factor", "ic", "ic_ir", "sign", "sharpe", "selected", "weight"]
    print(verdict.per_factor[cols].head(15).to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.out_dir / f"deployment_verdict_{stamp}.json"
    txt_path = args.out_dir / f"deployment_verdict_{stamp}.txt"
    json_path.write_text(json.dumps(verdict.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(verdict.report_text, encoding="utf-8")
    print(f"\nSaved JSON: {json_path}\nSaved TXT:  {txt_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
