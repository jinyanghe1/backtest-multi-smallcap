"""风险护栏 — 三层防御机制

目标: 将最大回撤从 ~-50% 压到 <-30% (Calmar >= 0.5)

三层机制:
1. 波动率目标缩放: 将 gross 敞口缩放到目标年化波动率
2. 崩盘过滤器: 当市场宽度崩塌或微盘指数跌破均线时降仓
3. 滚动回撤节流: 当滚动回撤触线时降 gross

与逐日个股止损正交, 独立作用于组合层面。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RiskOverlayConfig:
    """风险护栏配置"""
    
    # ── 层1: 波动率目标 ──
    target_vol_annual: float = 0.20  # 目标年化波动率 (20%)
    min_gross: float = 0.30  # 最小 gross 敞口 (30%)
    max_gross: float = 1.00  # 最大 gross 敞口 (100%)
    vol_lookback: int = 20  # 计算波动率的窗口 (月数)
    
    # ── 层2: 崩盘过滤器 ──
    enable_crash_filter: bool = True
    breadth_threshold: float = 0.30  # 市场宽度 < 30% 时触发
    breadth_lookback: int = 20  # 宽度计算窗口 (日)
    index_below_ma: bool = True  # 微盘指数是否跌破 MA
    index_ma_window: int = 20  # 均线窗口 (日)
    crash_gross: float = 0.30  # 崩盘时的 gross 敞口 (30%)
    
    # ── 层3: 滚动回撤节流 ──
    enable_dd_throttle: bool = True
    dd_window: int = 60  # 滚动回撤窗口 (日)
    dd_threshold: float = -0.15  # 触发阈值 (-15%)
    dd_severe: float = -0.25  # 严重阈值 (-25%)
    dd_gross: float = 0.50  # 一般回撤时的 gross (50%)
    dd_severe_gross: float = 0.20  # 严重回撤时的 gross (20%)
    
    # ── 状态跟踪 ──
    track_state: bool = True  # 是否记录每日 risk state


class RiskOverlay:
    """
    组合层面的风险护栏引擎。
    
    每天运行一次, 返回当前 gross 敞口比例 (0-1)。
    策略层据此调整持仓规模。
    
    使用示例:
        overlay = RiskOverlay(RiskOverlayConfig())
        for each day:
            gross = overlay.compute_gross_exposure(equity_curve, market_data)
            portfolio_return *= gross  # 应用风险缩放
    """
    
    def __init__(self, config: RiskOverlayConfig = None):
        self.config = config or RiskOverlayConfig()
        self._state_history = []  # 每日状态记录
    
    def compute_gross_exposure(
        self,
        equity_curve: pd.Series,  # 逐日权益曲线
        market_data: pd.DataFrame = None,  # 市场数据 (可选)
    ) -> float:
        """
        计算当前 gross 敞口。
        
        取三层的最严格限制。
        
        Returns:
            gross 敞口比例 (0-1)
        """
        cfg = self.config
        
        # 层1: 波动率目标缩放
        gross_vol = self._vol_target_scaling(equity_curve)
        
        # 层2: 崩盘过滤器
        gross_crash = 1.0
        if cfg.enable_crash_filter and market_data is not None:
            gross_crash = self._crash_filter(market_data)
        
        # 层3: 滚动回撤节流
        gross_dd = 1.0
        if cfg.enable_dd_throttle:
            gross_dd = self._dd_throttle(equity_curve)
        
        # 取最严格的限制 (最小)
        gross = min(gross_vol, gross_crash, gross_dd)
        
        # 边界限制
        gross = max(gross, cfg.min_gross)
        gross = min(gross, cfg.max_gross)
        
        # 记录状态
        if cfg.track_state:
            self._state_history.append({
                "date": equity_curve.index[-1] if len(equity_curve) > 0 else None,
                "gross": gross,
                "gross_vol": gross_vol,
                "gross_crash": gross_crash,
                "gross_dd": gross_dd,
                "current_drawdown": self._current_drawdown(equity_curve),
            })
        
        return gross
    
    def _vol_target_scaling(self, equity_curve: pd.Series) -> float:
        """
        根据历史波动率缩放 gross 敞口。
        
        公式: gross = target_vol / realized_vol
        如果 realized_vol < target_vol, gross 可以 > 1 (但受 max_gross 限制)
        """
        cfg = self.config
        
        if len(equity_curve) < 5:
            return cfg.max_gross
        
        # 计算日收益率
        daily_ret = equity_curve.pct_change().dropna()
        
        # 使用最近窗口计算日波动率
        lookback = min(cfg.vol_lookback * 21, len(daily_ret))  # 月 -> 日
        recent = daily_ret.iloc[-lookback:]
        
        if len(recent) < 5:
            return cfg.max_gross
        
        daily_vol = recent.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        if annual_vol <= 0:
            return cfg.max_gross
        
        gross = cfg.target_vol_annual / annual_vol
        
        # 限制
        return max(cfg.min_gross, min(gross, cfg.max_gross))
    
    def _crash_filter(self, market_data: pd.DataFrame) -> float:
        """
        崩盘过滤器。
        
        触发条件 (任一):
        1. 市场宽度 < 阈值 (上涨股票占比 < 30%)
        2. 微盘指数跌破均线
        """
        cfg = self.config
        
        if len(market_data) < cfg.index_ma_window:
            return cfg.max_gross
        
        # 检查市场宽度
        breadth_triggered = False
        if 'breadth' in market_data.columns:
            recent_breadth = market_data['breadth'].iloc[-cfg.breadth_lookback:]
            if len(recent_breadth) > 0 and recent_breadth.mean() < cfg.breadth_threshold:
                breadth_triggered = True
        
        # 检查指数均线
        ma_triggered = False
        if cfg.index_below_ma and 'index_close' in market_data.columns:
            idx_close = market_data['index_close']
            ma = idx_close.rolling(cfg.index_ma_window).mean()
            if len(ma) > 0 and not pd.isna(ma.iloc[-1]):
                if idx_close.iloc[-1] < ma.iloc[-1]:
                    ma_triggered = True
        
        # 如果任一触发, 降到 crash_gross
        if breadth_triggered or ma_triggered:
            return cfg.crash_gross
        
        return cfg.max_gross
    
    def _dd_throttle(self, equity_curve: pd.Series) -> float:
        """
        滚动回撤节流。
        
        当滚动回撤触线时降 gross。
        """
        cfg = self.config
        
        current_dd = self._current_drawdown(equity_curve)
        
        if current_dd <= cfg.dd_severe:
            return cfg.dd_severe_gross
        elif current_dd <= cfg.dd_threshold:
            return cfg.dd_gross
        
        return cfg.max_gross
    
    def _current_drawdown(self, equity_curve: pd.Series) -> float:
        """计算当前回撤"""
        if len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve.expanding().max().iloc[-1]
        current = equity_curve.iloc[-1]
        
        if peak <= 0:
            return 0.0
        
        return (current - peak) / peak
    
    def get_state_history(self) -> pd.DataFrame:
        """返回风险状态历史记录"""
        if not self._state_history:
            return pd.DataFrame()
        return pd.DataFrame(self._state_history)
    
    def summary(self) -> dict:
        """返回风险护栏运行摘要"""
        if not self._state_history:
            return {"status": "no data"}
        
        hist = pd.DataFrame(self._state_history)
        
        return {
            "n_days": len(hist),
            "avg_gross": hist["gross"].mean() if "gross" in hist.columns else None,
            "min_gross": hist["gross"].min() if "gross" in hist.columns else None,
            "crash_filter_activated": (hist["gross_crash"] < 1.0).sum() if "gross_crash" in hist.columns else 0,
            "dd_throttle_activated": (hist["gross_dd"] < 1.0).sum() if "gross_dd" in hist.columns else 0,
        }


class SimpleMarketBreadth:
    """简单市场宽度计算器"""
    
    @staticmethod
    def compute(
        returns: pd.DataFrame,  # 逐日收益率 (date x stock)
    ) -> pd.Series:
        """
        计算每日市场宽度 (上涨股票占比)。
        
        Returns:
            Series indexed by date, values = 上涨股票占比 (0-1)
        """
        if returns.empty:
            return pd.Series(dtype=float)
        
        breadth = (returns > 0).sum(axis=1) / returns.count(axis=1)
        return breadth
