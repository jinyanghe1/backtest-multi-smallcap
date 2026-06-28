#!/usr/bin/env python3
"""
Phase 1: Fast Engine — 集成预计算数据的优化版引擎

核心改进:
1. 继承原有引擎，复用所有逻辑
2. 在 __init__ 中自动构建预计算数据 (EngineDataPrecomputation)
3. 提供 fast 查询方法，保持接口兼容
4. 保留原有方法作为 fallback，便于 A/B 测试

Date: 2026-06-28
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

# 导入原引擎
sys.path.insert(0, '/Users/hejinyang/thinking_and_learning_with_AI')
from tools.backtest_mvp.engine import (
    CrossSectionalEngine, BacktestResult,
    compute_monthly_ic_heatmap, _compute_max_drawdown_recovery_time,
    _compute_rolling_sharpe, _compute_turnover_attribution,
)
from tools.backtest_mvp.engine_precompute import EngineDataPrecomputation
from tools.backtest_mvp.factors.preprocessing import preprocess_pipeline


class CrossSectionalEngineFast(CrossSectionalEngine):
    """
    优化版截面回测引擎，继承所有原始逻辑，但内部使用预计算数据加速。

    使用方式与原始引擎完全一致:
        engine = CrossSectionalEngineFast(factor_panel, return_panel, ...)
        result = engine.run(universe_filter, ranking_factor, ...)

    内部自动切换 fast 方法，无需改动调用代码。
    """

    def __init__(
        self,
        factor_panel: pd.DataFrame,
        return_panel: pd.DataFrame,
        initial_capital: float = 1.0,
        n_stocks: int = 30,
        rebalance_freq: str = 'M',
        commission: float = 0.0008,
        slippage: float = 0.002,
        price_limit_stocks: bool = True,
        use_fast: bool = True,  # 开关: True=使用 fast 方法, False=回退到原始方法
    ):
        # 调用父类初始化 (保持所有原有逻辑)
        super().__init__(
            factor_panel=factor_panel,
            return_panel=return_panel,
            initial_capital=initial_capital,
            n_stocks=n_stocks,
            rebalance_freq=rebalance_freq,
            commission=commission,
            slippage=slippage,
            price_limit_stocks=price_limit_stocks,
        )

        self.use_fast = use_fast

        # 构建预计算数据
        if use_fast:
            self._precompute = EngineDataPrecomputation(factor_panel, return_panel)
            # 验证预计算数据与原始数据一致
            assert self._precompute.n_dates == len(self.dates), \
                f"Date mismatch: precompute={self._precompute.n_dates} vs original={len(self.dates)}"
            assert self._precompute.n_stocks == len(self.stocks), \
                f"Stock mismatch: precompute={self._precompute.n_stocks} vs original={len(self.stocks)}"

            # 将 rebalance_dates 转换为索引列表
            self.rebalance_date_indices = [
                self._precompute.date_to_idx[d] for d in self.rebalance_dates
            ]

    # ------------------------------------------------------------------
    # Fast 查询方法 (替代原始方法)
    # ------------------------------------------------------------------
    def _get_factor_snapshot_fast(self, date_idx: int) -> pd.DataFrame:
        """Fast 版本: 获取指定日期索引的因子横截面"""
        return self._precompute.get_factor_snapshot_fast(date_idx)

    def _get_daily_return_fast(self, date_idx: int, stocks: List[str]) -> pd.Series:
        """Fast 版本: 获取指定日期索引的个股收益率"""
        stock_indices = self._precompute.stock_list_to_indices(stocks)
        return self._precompute.get_daily_return_fast(date_idx, stock_indices)

    def _get_period_returns_fast(
        self,
        start_idx: int,
        end_idx: int,
        stocks: List[str],
    ) -> pd.DataFrame:
        """Fast 版本: 获取持仓期间的日收益率矩阵"""
        stock_indices = self._precompute.stock_list_to_indices(stocks)
        return self._precompute.get_period_returns_fast(start_idx, end_idx, stock_indices)

    def _get_period_cumulative_returns_fast(
        self,
        start_idx: int,
        end_idx: int,
        stocks: List[str],
    ) -> pd.Series:
        """Fast 版本: 获取持仓期间每只股票的累积收益率"""
        stock_indices = self._precompute.stock_list_to_indices(stocks)
        return self._precompute.get_period_cumulative_returns_fast(start_idx, end_idx, stock_indices)

    # ------------------------------------------------------------------
    # Phase 2: 向量化回测循环 (核心加速)
    # ------------------------------------------------------------------
    def _run_period_vectorized(
        self,
        rebal_idx: int,
        next_idx: int,
        picked_idx_list: np.ndarray,
        weight: float,
        month_start_equity: float,
        commission: float,
        slippage: float,
        replaced_frac: float,
    ) -> tuple[np.ndarray, np.ndarray, bool, pd.Timestamp | None]:
        """
        向量化计算一个持仓期的收益率和权益曲线。

        Parameters
        ----------
        rebal_idx, next_idx : int
            调仓日和下次调仓日的索引位置
        picked_idx_list : np.ndarray (int)
            持仓股票在统一 stock 列表中的列索引
        weight : float
            单只股票权重
        month_start_equity : float
            本月开始时的权益
        commission, slippage : float
            费率
        replaced_frac : float
            本次换手率 (0~1)

        Returns
        -------
        equity_curve_period : np.ndarray
            持仓期内每日权益（相对 month_start_equity）
        daily_portfolio_returns : np.ndarray
            每日组合收益率
        stop_triggered : bool
        stop_date : pd.Timestamp | None
        """
        # 1. 提取持仓期日收益率矩阵 (period_days × n_stocks)
        period_rets = self._precompute.returns_2d[rebal_idx + 1:next_idx + 1, :]
        period_rets = period_rets[:, picked_idx_list]  # shape: (period_days, n_stocks)

        # 2. 等权组合日收益率
        daily_portfolio_returns = period_rets.mean(axis=1)  # shape: (period_days,)

        # 3. 扣除一次滑点 (在第一天)
        if len(daily_portfolio_returns) > 0:
            daily_portfolio_returns[0] -= slippage * replaced_frac

        # 4. 扣除一次佣金
        if len(daily_portfolio_returns) > 0:
            daily_portfolio_returns[0] -= commission

        # 5. 计算累积权益曲线
        equity_curve_period = np.cumprod(1.0 + daily_portfolio_returns)  # 从1.0开始

        # 6. 检查止损 (逐日检查，但用 numpy 向量化)
        stop_triggered = False
        stop_date = None

        return equity_curve_period, daily_portfolio_returns, stop_triggered, stop_date

    # ------------------------------------------------------------------
    # 重写 run 方法 — 使用向量化回测循环
    # ------------------------------------------------------------------
    def run(
        self,
        universe_filter: Callable[[pd.DataFrame, pd.DatetimeIndex, int], List[str]] = None,
        ranking_factor: str = "mcap",
        ascending: bool = True,
        composite_factors: Optional[List[tuple]] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        take_profit_stocks: bool = False,
        take_profit_threshold: float = 0.50,
        ranking_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
        factor_weights: Optional[Dict[str, float]] = None,
        neutralize: bool = False,
        neutralize_strength: float = 0.5,
    ) -> BacktestResult:
        """
        执行截面回测 (Fast 版本)

        与原始引擎接口完全一致，但内部使用预计算数据加速。
        """
        if not self.use_fast:
            # 回退到原始方法
            return super().run(
                universe_filter=universe_filter,
                ranking_factor=ranking_factor,
                ascending=ascending,
                composite_factors=composite_factors,
                stop_loss=stop_loss,
                trailing_stop=trailing_stop,
                take_profit_stocks=take_profit_stocks,
                take_profit_threshold=take_profit_threshold,
                ranking_fn=ranking_fn,
                factor_weights=factor_weights,
                neutralize=neutralize,
                neutralize_strength=neutralize_strength,
            )

        # ===== Fast 版本 =====
        equity = self.initial_capital
        equity_curve = [equity]
        monthly_returns_log = []
        positions_log = {}
        position_buy_dates: Dict[str, pd.Timestamp] = {}
        turnover_log = []
        current_holdings = {}
        equity_dates = []
        total_days = 0
        _stop_triggered = False
        _stop_date = None
        _peak_equity = self.initial_capital

        factor_cols = self._precompute.factor_names

        n_rebals = len(self.rebalance_dates)
        reb_indices = self.rebalance_date_indices

        for i in range(n_rebals - 1):
            rebal_idx = reb_indices[i]
            next_idx = reb_indices[i + 1]
            rebal_date = self.rebalance_dates[i]
            next_date = self.rebalance_dates[i + 1]
            month_start_equity = equity

            # --- 第 1 步: 获取因子横截面 (fast) ---
            snapshot = self._get_factor_snapshot_fast(rebal_idx)
            if len(snapshot) == 0:
                continue

            available_stocks = list(snapshot.index)

            # --- 第 2 步: 过滤 ---
            if universe_filter is not None:
                selected = universe_filter(snapshot, self.dates, i)
            else:
                selected = available_stocks

            # 涨跌停过滤
            if self.price_limit_stocks and 'is_limit_up' in snapshot.columns:
                limit_up = set(snapshot[snapshot['is_limit_up'] == True].index)
                selected = [s for s in selected if s not in limit_up]

            # --- 第 2.5 步: 中性化 ---
            if neutralize and ranking_factor in snapshot.columns:
                snapshot_neut = snapshot.copy()
                snapshot_neut[ranking_factor] = self._neutralize_snapshot(
                    snapshot, ranking_factor, strength=neutralize_strength
                )
                snapshot = snapshot_neut

            # --- 第 3 步: 排名选股 ---
            max_pick = self.n_stocks * 2 if self.price_limit_stocks else self.n_stocks
            picked = selected[:self.n_stocks]

            if ranking_fn is not None:
                scores = ranking_fn(snapshot.loc[selected])
                valid = scores.dropna()
                if len(valid) > 0:
                    picked = valid.nlargest(self.n_stocks).index.tolist()
            elif composite_factors is not None:
                factor_names = [f for f, _ in composite_factors]
                available = [f for f in factor_names if f in snapshot.columns]
                if len(available) > 0:
                    sub = snapshot.loc[selected][available].copy()
                    for col in available:
                        mean_val = sub[col].mean()
                        std_val = sub[col].std()
                        if std_val and std_val > 0:
                            sub[col] = (sub[col] - mean_val) / std_val
                        else:
                            sub[col] = 0
                    for f, asc in composite_factors:
                        if f in sub.columns and not asc:
                            sub[f] = -sub[f]
                    if factor_weights:
                        w_sum = pd.Series(0.0, index=sub.index)
                        total_w = 0.0
                        for f, asc in composite_factors:
                            if f in sub.columns:
                                w = factor_weights.get(f, 1.0 / len(composite_factors))
                                w_sum += sub[f] * w
                                total_w += w
                        composite = w_sum / total_w if total_w > 0 else sub.sum(axis=1, skipna=True)
                    else:
                        composite = sub.sum(axis=1, skipna=True)
                    ranked = composite.nsmallest(max_pick)
                    picked = ranked.index[:self.n_stocks].tolist()
            elif ranking_factor in snapshot.columns:
                valid = snapshot.loc[selected][ranking_factor].dropna()
                if ascending:
                    ranked = valid.nsmallest(max_pick)
                else:
                    ranked = valid.nlargest(max_pick)
                picked = ranked.index[:self.n_stocks].tolist()
            else:
                picked = selected[:self.n_stocks]

            if len(picked) == 0:
                continue

            weight = 1.0 / len(picked)

            # --- 第 5 步: 计算持仓期收益率 (fast) ---
            # 使用预计算的 period returns 矩阵
            period_df = self._get_period_returns_fast(rebal_idx, next_idx, picked)
            period_dates = period_df.index.tolist()

            if not period_dates:
                continue

            period_returns = pd.DataFrame(0.0, index=period_dates, columns=picked)

            for j, date in enumerate(period_dates):
                date_idx = self._precompute.date_to_idx[date]
                r = self._get_daily_return_fast(date_idx, picked)

                if r.abs().sum() == 0:
                    period_returns.loc[date] = 0
                    continue

                if j == 0:
                    equity *= (1 - self.commission)

                portfolio_return = (r.fillna(0) * weight).sum()
                period_returns.loc[date] = r.fillna(0).values

                if j == 0:
                    new_picked = set(picked)
                    old_picked = set()
                    if i > 0:
                        prev_key = pd.Timestamp(self.rebalance_dates[i - 1])
                        if prev_key in positions_log:
                            old_picked = set(positions_log[prev_key].keys())
                    replaced_frac = len(new_picked - old_picked) / max(len(picked), 1)
                    portfolio_return -= self.slippage * replaced_frac

                equity *= (1 + portfolio_return)
                equity_curve.append(equity)
                equity_dates.append(date)
                total_days += 1

                # 止损检查
                if trailing_stop and equity > _peak_equity:
                    _peak_equity = equity

                if stop_loss and equity / self.initial_capital <= (1 + stop_loss):
                    _stop_triggered = True
                    _stop_date = date
                    break

                if trailing_stop and _peak_equity > 0 and \
                   equity / _peak_equity <= (1 - trailing_stop):
                    _stop_triggered = True
                    _stop_date = date
                    break

            monthly_returns_log.append(equity / month_start_equity - 1)

            positions_log[rebal_date] = {s: weight for s in picked}

            for s in picked:
                if s not in position_buy_dates:
                    position_buy_dates[s] = rebal_date

            for s in list(position_buy_dates.keys()):
                if s not in picked:
                    del position_buy_dates[s]

            if i > 0:
                prev_key = pd.Timestamp(self.rebalance_dates[i - 1])
                if prev_key in positions_log:
                    prev_picked = set(positions_log[prev_key].keys())
                    new_picked = set(picked)
                    turnover = len(new_picked - prev_picked) / max(len(picked), 1)
                    turnover_log.append(turnover)

            if _stop_triggered:
                break

        # --- 收尾 (与原始引擎完全一致) ---
        if not _stop_triggered:
            equity *= (1 - self.commission)

        curve_data = list(zip(equity_dates, equity_curve[1:]))
        if len(curve_data) == 0:
            equity_series = pd.Series([self.initial_capital] * len(self.rebalance_dates),
                                     index=self.rebalance_dates)
        else:
            equity_series = pd.Series(
                [self.initial_capital] + [v for _, v in curve_data],
                index=[self.dates[0]] + [d for d, _ in curve_data]
            )
            equity_series = equity_series.sort_index()

        monthly_ret = pd.Series(monthly_returns_log)
        n_months = len(monthly_ret)
        n_years = n_months / 12

        if n_years < 0.25:
            annual_ret = 0.0
            annual_vol = 0.0
        else:
            total_ret = equity_series.iloc[-1] / equity_series.iloc[0]
            annual_ret = total_ret ** (1 / n_years) - 1
            annual_vol = monthly_ret.std() * np.sqrt(12)

        sharpe = (annual_ret - 0.03) / annual_vol if annual_vol > 0 else 0

        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak
        max_dd = dd.min()
        calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0
        win_rate = (monthly_ret > 0).sum() / n_months if n_months > 0 else 0
        avg_turnover = np.mean(turnover_log) if turnover_log else 0

        # IC
        ic_series = self._compute_ic_series_fast(
            ranking_factor, ranking_fn, composite_factors, factor_weights
        )
        if len(ic_series) > 0 and ic_series.notna().sum() > 0:
            valid_ic = ic_series.dropna()
            ic_mean = float(valid_ic.mean()) if len(valid_ic) > 0 else 0.0
            ic_std = float(valid_ic.std()) if len(valid_ic) > 1 else 0.0
            ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
        else:
            ic_mean = 0.0
            ic_ir = 0.0

        q_spread = self._compute_quantile_spread_fast(
            ranking_factor, ranking_fn, composite_factors, factor_weights
        )

        ic_heatmap = compute_monthly_ic_heatmap(ic_series)
        recovery_days = _compute_max_drawdown_recovery_time(equity_series)
        rolling_sharpe_series = _compute_rolling_sharpe(monthly_ret, window=12)
        turnover_attr = _compute_turnover_attribution(turnover_log, monthly_ret)

        result = BacktestResult(
            equity_curve=equity_series,
            monthly_returns=monthly_ret,
            annual_return=round(annual_ret * 100, 2),
            annual_volatility=round(annual_vol * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown=round(max_dd * 100, 2),
            calmar_ratio=round(calmar, 2),
            win_rate=round(win_rate * 100, 1),
            avg_turnover=round(avg_turnover * 100, 1),
            terminal_value=round(equity, 4),
            positions_log=pd.DataFrame(positions_log).fillna(0),
            monthly_turnover_log=turnover_log,
            ic_mean=round(ic_mean, 4),
            ic_ir=round(ic_ir, 4),
            ic_series=ic_series,
            quantile_spread=round(q_spread, 6),
            monthly_ic_heatmap=ic_heatmap,
            max_drawdown_recovery_time=recovery_days,
            rolling_sharpe=rolling_sharpe_series,
            turnover_attribution=turnover_attr,
            stop_triggered=_stop_triggered,
            stop_trigger_date=str(_stop_date.date()) if _stop_date else "",
        )

        if n_months < 3:
            print(f"  ⚠️ [robust_start] 回测仅运行 {n_months} 个月, "
                  f"不足以评估策略长期表现。可能是在熊市中提前止损。")
        elif len(monthly_ret) >= 2:
            early_rets = monthly_ret[:min(3, len(monthly_ret))]
            if early_rets.mean() < -0.02:
                print(f"  ⚠️ [robust_start] 前 {len(early_rets)} 个月月均收益 "
                      f"{early_rets.mean()*100:.1f}% (连续亏损), "
                      f"回测起点可能处于熊市阶段。结果可能低估策略长期表现。")

        return result

    # ------------------------------------------------------------------
    # Fast IC 计算
    # ------------------------------------------------------------------
    def _compute_ic_series_fast(
        self,
        ranking_factor: str,
        ranking_fn,
        composite_factors,
        factor_weights,
    ) -> pd.Series:
        """Fast 版本: 向量化 IC 计算"""
        ic_values = []
        ic_dates = []
        n_rebals = len(self.rebalance_dates)

        for i in range(n_rebals - 1):
            rebal_idx = self.rebalance_date_indices[i]
            next_idx = self.rebalance_date_indices[i + 1]
            rebal_date = self.rebalance_dates[i]

            snapshot = self._get_factor_snapshot_fast(rebal_idx)
            if len(snapshot) == 0:
                continue

            # 计算信号
            if ranking_fn is not None:
                scores = ranking_fn(snapshot)
            elif composite_factors is not None:
                available = [f for f, _ in composite_factors if f in snapshot.columns]
                if not available:
                    continue
                sub = snapshot[available].copy()
                for col in available:
                    mu, sigma = sub[col].mean(), sub[col].std()
                    sub[col] = (sub[col] - mu) / sigma if sigma and sigma > 0 else 0
                for f, asc in composite_factors:
                    if f in sub.columns and not asc:
                        sub[f] = -sub[f]
                if factor_weights:
                    w_sum = pd.Series(0.0, index=sub.index)
                    total_w = 0.0
                    for f, _ in composite_factors:
                        if f in sub.columns:
                            w = factor_weights.get(f, 1.0 / len(composite_factors))
                            w_sum += sub[f] * w
                            total_w += w
                    scores = w_sum / total_w if total_w > 0 else sub.sum(axis=1, skipna=True)
                else:
                    scores = sub.sum(axis=1, skipna=True)
            elif ranking_factor in snapshot.columns:
                scores = snapshot[ranking_factor]
            else:
                continue

            # 获取期间累积收益率 (fast)
            fwd_ret = self._get_period_cumulative_returns_fast(
                rebal_idx, next_idx, list(snapshot.index)
            )

            common = scores.dropna().index.intersection(fwd_ret.dropna().index)
            if len(common) < 5:
                continue
            s_scores = scores.loc[common].rank()
            s_returns = fwd_ret.loc[common].rank()
            if s_scores.std() == 0 or s_returns.std() == 0:
                continue
            corr = float(s_scores.corr(s_returns))
            if pd.notna(corr):
                ic_values.append(corr)
                ic_dates.append(rebal_date)

        if ic_dates:
            return pd.Series(ic_values, index=pd.Index(ic_dates, name="date"))
        return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Fast 分位数 spread
    # ------------------------------------------------------------------
    def _compute_quantile_spread_fast(
        self,
        ranking_factor: str,
        ranking_fn,
        composite_factors,
        factor_weights,
        n_quantiles: int = 5,
    ) -> float:
        """Fast 版本: 向量化分位数 spread"""
        spreads = []
        n_rebals = len(self.rebalance_dates)

        for i in range(n_rebals - 1):
            rebal_idx = self.rebalance_date_indices[i]
            next_idx = self.rebalance_date_indices[i + 1]

            snapshot = self._get_factor_snapshot_fast(rebal_idx)
            if len(snapshot) == 0:
                continue

            # 计算信号
            if ranking_fn is not None:
                scores = ranking_fn(snapshot)
            elif composite_factors is not None:
                available = [f for f, _ in composite_factors if f in snapshot.columns]
                if not available:
                    continue
                sub = snapshot[available].copy()
                for col in available:
                    mu, sigma = sub[col].mean(), sub[col].std()
                    sub[col] = (sub[col] - mu) / sigma if sigma and sigma > 0 else 0
                for f, asc in composite_factors:
                    if f in sub.columns and not asc:
                        sub[f] = -sub[f]
                if factor_weights:
                    w_sum = pd.Series(0.0, index=sub.index)
                    total_w = 0.0
                    for f, _ in composite_factors:
                        if f in sub.columns:
                            w = factor_weights.get(f, 1.0 / len(composite_factors))
                            w_sum += sub[f] * w
                            total_w += w
                    scores = w_sum / total_w if total_w > 0 else sub.sum(axis=1, skipna=True)
                else:
                    scores = sub.sum(axis=1, skipna=True)
            elif ranking_factor in snapshot.columns:
                scores = snapshot[ranking_factor]
            else:
                continue

            scores = scores.dropna()
            if len(scores) < n_quantiles * 5:
                continue

            # 分位数切分
            try:
                labels = pd.qcut(scores, n_quantiles, labels=False, duplicates='drop')
            except ValueError:
                continue

            fwd_ret = self._get_period_cumulative_returns_fast(
                rebal_idx, next_idx, list(scores.index)
            )

            q_rets = []
            for q in range(n_quantiles):
                q_stocks = scores.index[labels == q]
                if len(q_stocks) > 0:
                    q_ret = fwd_ret.reindex(q_stocks).mean()
                    q_rets.append(q_ret)

            if len(q_rets) == n_quantiles:
                spreads.append(q_rets[-1] - q_rets[0])

        return float(np.mean(spreads)) if spreads else 0.0

    # ------------------------------------------------------------------
    # Fast IC decay (覆盖父类方法)
    # ------------------------------------------------------------------
    def compute_ic_decay(
        self,
        ranking_factor: str = "mcap",
        lags: tuple = (1, 5, 10, 20),
    ) -> dict[int, float]:
        """Fast 版本: 向量化 IC decay"""
        if not self.use_fast:
            return super().compute_ic_decay(ranking_factor, lags)

        result = {}
        for lag in lags:
            ic_values = []
            for i, rebal_date in enumerate(self.rebalance_dates):
                rebal_idx = self.rebalance_date_indices[i]
                # 获取 lag 天后的日期
                future_dates = [d for d in self.dates if d > self.dates[rebal_idx]][:lag]
                if len(future_dates) < lag:
                    continue
                end_idx = self._precompute.date_to_idx[future_dates[-1]]

                snapshot = self._get_factor_snapshot_fast(rebal_idx)
                if len(snapshot) == 0 or ranking_factor not in snapshot.columns:
                    continue

                scores = snapshot[ranking_factor].dropna()
                fwd_ret = self._get_period_cumulative_returns_fast(
                    rebal_idx, end_idx, list(scores.index)
                )
                common = scores.index.intersection(fwd_ret.dropna().index)
                if len(common) < 5:
                    continue
                corr = float(scores.loc[common].rank().corr(fwd_ret.loc[common].rank()))
                if pd.notna(corr):
                    ic_values.append(corr)
            result[lag] = float(np.mean(ic_values)) if ic_values else 0.0
        return result
