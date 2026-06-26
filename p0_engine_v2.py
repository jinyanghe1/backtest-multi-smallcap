from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from tools.backtest_mvp.engine import CrossSectionalEngine, BacktestResult
from tools.backtest_mvp.data.delisted import DelistManager
from tools.backtest_mvp.data.adv_impact import ADVImpactModel
from tools.backtest_mvp.data.risk_overlay import RiskOverlay, RiskOverlayConfig
from tools.backtest_mvp.research_loop.deflated_sharpe import (
    compute_deflated_sharpe, deflated_sharpe_significance
)

# ───────────────────────────────────────────
# P0 Engine V2 — 逐日内嵌循环，真实交易路径
# ───────────────────────────────────────────

class P0EngineV2:
    """
    P0 增强版回测引擎 v2 — 逐日内嵌循环。

    与 v1 (后处理) 的根本区别:
    - v1: 先跑原引擎，再对结果打折
    - v2: 逐日模拟真实交易，在循环中实时应用 P0 修复

    P0 功能 (全部在逐日循环中内嵌):
    1. PIT Universe: 每日选股前检查上市/退市状态
    2. ADV 冲击: 调仓日逐只计算真实冲击成本
    3. Risk Overlay: 每日根据权益曲线计算 gross 敞口
    4. Deflated Sharpe: 回测结束后报告

    数据依赖 (Phase 0 已采集):
    - listing_dates.csv: 上市日期表
    - delisted_stocks.csv: 退市数据
    - adv_panel.parquet: 逐日 ADV
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
        # ── P0 配置 ──
        enable_pit_universe: bool = True,       # 默认开启
        enable_adv_impact: bool = True,        # 默认开启
        enable_risk_overlay: bool = True,      # 默认开启
        enable_deflated_sharpe: bool = True,   # 默认开启
        # ── 数据 ──
        listing_dates: pd.DataFrame = None,     # D0.1 输出
        delist_manager: DelistManager = None,  # D0.3 输出
        adv_data: pd.DataFrame = None,         # D0.2 输出
        # ── 参数 ──
        adv_model: ADVImpactModel = None,
        risk_overlay_config: RiskOverlayConfig = None,
        n_trials: int = 1,
    ):
        self.factor_panel = factor_panel
        self.return_panel = return_panel
        self.initial_capital = initial_capital
        self.n_stocks = n_stocks
        self.rebalance_freq = rebalance_freq
        self.commission = commission
        self.slippage = slippage
        self.price_limit_stocks = price_limit_stocks

        self.enable_pit_universe = enable_pit_universe
        self.enable_adv_impact = enable_adv_impact
        self.enable_risk_overlay = enable_risk_overlay
        self.enable_deflated_sharpe = enable_deflated_sharpe
        self.n_trials = n_trials

        # ── 加载数据 ──
        if enable_pit_universe:
            self._load_listing_dates(listing_dates)
            self.delist_manager = delist_manager or DelistManager()
            self.delist_manager.fetch_all()
        else:
            self.listing_dates = None
            self.delist_manager = None

        if enable_adv_impact:
            self.adv_model = adv_model or ADVImpactModel()
            self.adv_data = adv_data
        else:
            self.adv_model = None
            self.adv_data = None

        if enable_risk_overlay:
            self.risk_overlay = RiskOverlay(risk_overlay_config or RiskOverlayConfig())
        else:
            self.risk_overlay = None

        # 预计算日期/股票
        self.dates = sorted(
            pd.Timestamp(d) for d in (
                set(factor_panel.index.get_level_values(0))
                & set(return_panel.index.get_level_values(0))
            )
        )
        self.stocks = sorted(
            set(factor_panel.index.get_level_values(1))
            & set(return_panel.index.get_level_values(1))
        )

        # 生成调仓日
        all_dates = pd.DatetimeIndex(self.dates)
        if rebalance_freq == 'M':
            self.rebalance_dates = self._get_monthly_dates(all_dates)
        elif rebalance_freq == 'W':
            self.rebalance_dates = self._get_weekly_dates(all_dates)
        else:
            self.rebalance_dates = all_dates

    # ───────────────────────────────────────────
    # 日期工具
    # ───────────────────────────────────────────

    def _get_monthly_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        df = pd.DataFrame({'date': dates})
        df['month'] = dates.to_period('M')
        return pd.DatetimeIndex(df.groupby('month')['date'].last().values)

    def _get_weekly_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        df = pd.DataFrame({'date': dates})
        df['week'] = dates.to_period('W')
        return pd.DatetimeIndex(df.groupby('week')['date'].last().values)

    # ───────────────────────────────────────────
    # 数据查询
    # ───────────────────────────────────────────

    def _get_factor_snapshot(self, date: pd.Timestamp) -> pd.DataFrame:
        try:
            return self.factor_panel.xs(date, level=0, drop_level=True)
        except (KeyError, TypeError, AttributeError):
            return pd.DataFrame()

    def _get_daily_return(self, date: pd.Timestamp, stocks: List[str]) -> pd.Series:
        """获取指定日期的个股收益率"""
        try:
            r = self.return_panel.xs(date, level=0, drop_level=True)
            if isinstance(r, pd.Series):
                # 只有一列，xs 返回 Series（索引=symbol，值=return）
                return r.reindex(stocks).fillna(0)
            else:
                # DataFrame
                r = r.reindex(stocks)
                if 'daily_return' in r.columns:
                    return r['daily_return'].fillna(0)
                else:
                    return r.iloc[:, 0].fillna(0) if len(r.columns) > 0 else pd.Series(0.0, index=stocks)
        except (KeyError, AttributeError):
            return pd.Series(0.0, index=stocks)

    def _get_adv(self, symbol: str, date: pd.Timestamp) -> float:
        """获取某股票某日的 ADV (20日平均成交额)"""
        if self.adv_data is None:
            return 0.0
        try:
            return self.adv_data.loc[(date, symbol), 'adv_20d']
        except KeyError:
            return 0.0

    # ───────────────────────────────────────────
    # PIT Universe 过滤
    # ───────────────────────────────────────────

    def _load_listing_dates(self, listing_dates: pd.DataFrame = None):
        """加载上市日期表"""
        if listing_dates is not None:
            # 确保 list_date 是 datetime (传入的 DataFrame 可能是 CSV 读入的字符串)
            df = listing_dates.copy()
            df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')
            self.listing_dates = df.set_index('symbol')['list_date']
        else:
            # 尝试从文件加载
            from pathlib import Path
            cache = Path(__file__).resolve().parent.parent / 'data_cache' / 'listing_dates.csv'
            if cache.exists():
                df = pd.read_csv(cache, encoding='utf-8-sig')
                df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')
                self.listing_dates = df.set_index('symbol')['list_date']
            else:
                self.listing_dates = None

    def _is_alive(self, symbol: str, date: pd.Timestamp) -> bool:
        """检查股票在指定日期是否活着"""
        # 1. 检查是否已上市
        if self.listing_dates is not None:
            list_date = self.listing_dates.get(symbol)
            if list_date is not None and pd.notna(list_date):
                if date < list_date:
                    return False  # 还没上市
        # 2. 检查是否已退市
        if self.delist_manager is not None:
            if not self.delist_manager.is_alive(symbol, date):
                return False
        return True

    def _pit_filter(self, snapshot: pd.DataFrame, date: pd.Timestamp) -> List[str]:
        """PIT 过滤: 只保留当天还活着的股票"""
        if not self.enable_pit_universe:
            return list(snapshot.index)
        return [s for s in snapshot.index if self._is_alive(s, date)]

    # ───────────────────────────────────────────
    # 核心: 逐日回测循环
    # ───────────────────────────────────────────

    def run(
        self,
        universe_filter: Callable = None,
        ranking_factor: str = "mcap",
        ascending: bool = True,
        composite_factors: Optional[List[Tuple]] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        ranking_fn: Optional[Callable] = None,
        factor_weights: Optional[Dict[str, float]] = None,
    ) -> BacktestResult:
        """
        执行逐日回测 — P0 内嵌引擎的核心。

        与 CrossSectionalEngine.run() 的区别:
        - 逐日更新权益 (不是只在调仓日)
        - 每日应用 RiskOverlay
        - 调仓日逐只计算 ADV 冲击
        - PIT Universe 在选股前过滤
        """
        equity = self.initial_capital
        equity_curve = [equity]  # 初始权益
        equity_dates = [self.dates[0]] if self.dates else [pd.Timestamp('1970-01-01')]  # 空数据时用默认日期
        monthly_returns_log = []
        positions_log = {}
        turnover_log = []
        _stop_triggered = False
        _stop_date = None
        _peak_equity = self.initial_capital

        current_holdings = set()  # 当前持仓
        pending_rebalance = True  # 是否需要在下一个调仓日执行

        for i, date in enumerate(self.dates):
            ts = pd.Timestamp(date)
            is_rebalance = ts in self.rebalance_dates

            # ── Step 1: 如果已有持仓，计算今日收益 ──
            if len(current_holdings) > 0 and not _stop_triggered:
                stocks = list(current_holdings)
                weight = 1.0 / len(stocks)
                daily_ret = self._get_daily_return(ts, stocks)
                portfolio_return = (daily_ret * weight).sum()

                # Risk Overlay (核心 P0 修复)
                if self.enable_risk_overlay and self.risk_overlay is not None:
                    gross = self._compute_gross_exposure(equity_curve, ts)
                    portfolio_return *= gross

                # 更新权益
                equity *= (1 + portfolio_return)
                equity_curve.append(equity)
                equity_dates.append(ts)

                # 止损检查
                if trailing_stop and equity > _peak_equity:
                    _peak_equity = equity

                if stop_loss and equity / self.initial_capital <= (1 + stop_loss):
                    _stop_triggered = True
                    _stop_date = ts
                    current_holdings = set()
                    break

                if trailing_stop and _peak_equity > 0 and equity / _peak_equity <= (1 - trailing_stop):
                    _stop_triggered = True
                    _stop_date = ts
                    current_holdings = set()
                    break

            # ── Step 2: 如果是调仓日，执行选股 ──
            if is_rebalance and not _stop_triggered:
                # 获取因子横截面
                snapshot = self._get_factor_snapshot(ts)
                if len(snapshot) == 0:
                    continue

                available = list(snapshot.index)

                # PIT 过滤 (核心 P0 修复)
                alive = self._pit_filter(snapshot, ts)

                # 用户自定义 filter
                if universe_filter is not None:
                    rebalance_idx = list(self.rebalance_dates).index(ts) if ts in self.rebalance_dates else 0
                    selected = universe_filter(snapshot.loc[alive], self.dates, rebalance_idx)
                else:
                    selected = alive

                # 涨跌停过滤
                if self.price_limit_stocks and 'is_limit_up' in snapshot.columns:
                    limit_up = set(snapshot[snapshot['is_limit_up'] == True].index)
                    selected = [s for s in selected if s not in limit_up]

                # 排名选股
                picked = self._rank_stocks(
                    snapshot, selected, ranking_factor, ascending,
                    composite_factors, ranking_fn, factor_weights
                )

                if len(picked) > 0:
                    # 计算换手率
                    old_picked = current_holdings
                    new_picked = set(picked)
                    if len(old_picked) > 0:
                        turnover = len(new_picked - old_picked) / max(len(picked), 1)
                        turnover_log.append(turnover)
                    current_holdings = new_picked

                    # 记录持仓
                    weight = 1.0 / len(picked)
                    positions_log[ts] = {s: weight for s in picked}

                    # 调仓日 ADV 冲击 (核心 P0 修复)
                    if self.enable_adv_impact and self.adv_model is not None:
                        adv_drag = self._compute_adv_drag(picked, equity, weight, ts)
                        equity *= (1 - adv_drag)

                    # 手续费
                    equity *= (1 - self.commission)

                    # 记录调仓后的权益 (用于下一日计算)
                    equity_curve.append(equity)
                    equity_dates.append(ts)

        # 简化版月度收益计算 (从权益曲线)
        equity_series = pd.Series(equity_curve, index=[pd.Timestamp(d) for d in equity_dates])
        monthly_equity = equity_series.resample('ME').last()
        monthly_ret = monthly_equity.pct_change().dropna()

        # 计算指标
        n_months = len(monthly_ret)
        n_years = n_months / 12
        if n_years > 0.25:
            total_ret = equity_series.iloc[-1] / equity_series.iloc[0]
            annual_ret = total_ret ** (1 / n_years) - 1
            annual_vol = monthly_ret.std() * np.sqrt(12)
        else:
            annual_ret = 0
            annual_vol = 0

        sharpe = (annual_ret - 0.03) / annual_vol if annual_vol > 0 else 0
        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak
        max_dd = dd.min()
        calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0
        win_rate = (monthly_ret > 0).sum() / n_months if n_months > 0 else 0
        avg_turnover = np.mean(turnover_log) if turnover_log else 0

        # 使用原引擎的 IC 计算
        base_engine = CrossSectionalEngine(
            factor_panel=self.factor_panel,
            return_panel=self.return_panel,
            n_stocks=self.n_stocks,
            rebalance_freq=self.rebalance_freq,
        )
        ic_series = base_engine._compute_ic_series(
            ranking_factor, ranking_fn, composite_factors, factor_weights
        )
        if len(ic_series) > 0 and ic_series.notna().sum() > 0:
            valid_ic = ic_series.dropna()
            ic_mean = float(valid_ic.mean()) if len(valid_ic) > 0 else 0.0
            ic_std = float(valid_ic.std()) if len(valid_ic) > 1 else 0.0
            ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
        else:
            ic_mean = ic_ir = 0.0

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
            positions_log=pd.DataFrame(positions_log).fillna(0) if positions_log else pd.DataFrame(),
            monthly_turnover_log=turnover_log,
            ic_mean=round(ic_mean, 4),
            ic_ir=round(ic_ir, 4),
            ic_series=ic_series,
            stop_triggered=_stop_triggered,
            stop_trigger_date=str(_stop_date.date()) if _stop_date else "",
        )

        # Deflated Sharpe
        if self.enable_deflated_sharpe:
            dsr = compute_deflated_sharpe(
                sharpe=result.sharpe_ratio,
                n_trials=self.n_trials,
                n_periods=max(n_months, 2),
            )
            result.deflated_sharpe = dsr
            sig = deflated_sharpe_significance(dsr)
            result.deflated_sharpe_significance = sig['confidence']
            result.deflated_sharpe_pvalue = sig['p_value']

        return result

    # ───────────────────────────────────────────
    # 辅助方法
    # ───────────────────────────────────────────

    def _rank_stocks(
        self, snapshot, selected, ranking_factor, ascending,
        composite_factors, ranking_fn, factor_weights
    ) -> List[str]:
        """排名选股 (复用原引擎逻辑)"""
        max_pick = self.n_stocks * 2 if self.price_limit_stocks else self.n_stocks

        if ranking_fn is not None:
            scores = ranking_fn(snapshot.loc[selected])
            valid = scores.dropna()
            if len(valid) > 0:
                return valid.nlargest(self.n_stocks).index.tolist()

        elif composite_factors is not None:
            available = [f for f, _ in composite_factors if f in snapshot.columns]
            if len(available) > 0:
                sub = snapshot.loc[selected][available].copy()
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
                    composite = w_sum / total_w if total_w > 0 else sub.sum(axis=1, skipna=True)
                else:
                    composite = sub.sum(axis=1, skipna=True)
                ranked = composite.nsmallest(max_pick)
                return ranked.index[:self.n_stocks].tolist()

        elif ranking_factor in snapshot.columns:
            valid = snapshot.loc[selected][ranking_factor].dropna()
            if len(valid) > 0:
                if ascending:
                    ranked = valid.nsmallest(max_pick)
                else:
                    ranked = valid.nlargest(max_pick)
                return ranked.index[:self.n_stocks].tolist()

        return selected[:self.n_stocks]

    def _compute_adv_drag(self, picked: List[str], equity: float, weight: float, date: pd.Timestamp) -> float:
        """计算调仓日 ADV 冲击成本"""
        if not self.enable_adv_impact or self.adv_model is None:
            return 0.0

        order_value = equity * weight
        impacts = []
        for symbol in picked:
            adv = self._get_adv(symbol, date)
            if adv > 0:
                impact = self.adv_model.compute_impact(order_value, adv)
                impacts.append(impact)

        return np.mean(impacts) if impacts else 0.0

    def _compute_gross_exposure(self, equity_curve: list, date: pd.Timestamp) -> float:
        """计算当前 gross 敞口"""
        if not self.enable_risk_overlay or self.risk_overlay is None:
            return 1.0

        # equity_curve 记录到当前日期的所有权益值
        # 使用 equity_curve 自身的索引，不映射到 self.dates
        equity_series = pd.Series(equity_curve)
        return self.risk_overlay.compute_gross_exposure(equity_series)
