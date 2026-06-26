"""
混合策略框架 — Hybrid Strategy Framework
======================================
组合多个子策略 + 动态权重调整 + 市场状态过滤 + 风险平价仓位管理

核心设计:
1. 多策略组合: 微盘+ETF轮动+固收(简化版用现金代理)
2. 动态权重: 根据各子策略近期表现和波动率调整权重
3. 市场状态过滤: 基于沪深300/中证1000的MA/波动率识别状态
4. 风险平价: 波动率高的策略自动降低权重
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL = "bull"           # 牛市: 趋势向上，波动正常
    BEAR = "bear"           # 熊市: 趋势向下，波动放大
    RANGE = "range"         # 震荡: 无明显趋势，波动正常
    CRASH = "crash"         # 暴跌: 快速下跌，高波动
    RECOVERY = "recovery"   # 复苏: 从底部反弹


@dataclass
class SubStrategy:
    """子策略配置"""
    name: str
    strategy_def: dict          # 原始策略定义
    weight: float = 0.33        # 目标权重
    max_weight: float = 0.60    # 最大权重上限
    min_weight: float = 0.10    # 最小权重下限
    volatility_lookback: int = 20  # 波动率计算窗口(日)
    performance_lookback: int = 60  # 业绩计算窗口(日)


@dataclass
class HybridConfig:
    """混合策略配置"""
    name: str = "混合策略"
    sub_strategies: List[SubStrategy] = None
    
    # 市场状态参数
    regime_lookback_short: int = 20   # 短期MA
    regime_lookback_long: int = 60    # 长期MA
    regime_vol_threshold: float = 0.30  # 高波动阈值(年化)
    regime_trend_threshold: float = 0.05  # 趋势阈值
    
    # 仓位管理
    cash_buffer_bull: float = 0.10     # 牛市现金缓冲
    cash_buffer_bear: float = 0.40     # 熊市现金缓冲
    cash_buffer_range: float = 0.20    # 震荡现金缓冲
    cash_buffer_crash: float = 0.60    # 暴跌现金缓冲
    cash_buffer_recovery: float = 0.20  # 复苏现金缓冲
    
    # 风险平价
    risk_parity_target_vol: float = 0.20  # 目标年化波动率(20%)
    risk_parity_enabled: bool = True
    
    # 再平衡
    rebalance_freq: str = 'M'          # 月度调仓
    weight_adjustment_speed: float = 0.3  # 权重调整速度(0-1, 越大调整越快)


def detect_market_regime(
    market_returns: pd.Series,
    config: HybridConfig,
    current_date: pd.Timestamp,
) -> MarketRegime:
    """
    检测当前市场状态
    
    基于:
    - 短期 vs 长期 MA 差值 (趋势)
    - 近期波动率 (风险)
    - 近期收益分布 (极端性)
    
    Args:
        market_returns: 市场指数日收益率序列 (index=date)
        config: 混合配置
        current_date: 当前日期
    
    Returns:
        MarketRegime 枚举
    """
    if len(market_returns) < config.regime_lookback_long:
        return MarketRegime.RANGE  # 数据不足，默认震荡
    
    # 获取可用数据 (截止到当前日期)
    available = market_returns[market_returns.index <= current_date]
    if len(available) < config.regime_lookback_long:
        return MarketRegime.RANGE
    
    # 计算短期和长期收益
    short_ret = available.iloc[-config.regime_lookback_short:]
    long_ret = available.iloc[-config.regime_lookback_long:]
    
    # 趋势: 短期MA vs 长期MA
    short_ma = (1 + short_ret).prod() - 1
    long_ma = (1 + long_ret).prod() - 1
    trend = short_ma - long_ma
    
    # 波动率 (年化)
    vol = short_ret.std() * np.sqrt(252)
    
    # 近期收益 (判断是否暴跌)
    recent_5d = available.iloc[-5:]
    recent_5d_ret = (1 + recent_5d).prod() - 1
    
    # 状态判断
    if vol > config.regime_vol_threshold * 1.5:  # 极高波动
        if recent_5d_ret < -0.10:  # 近5日跌超10%
            return MarketRegime.CRASH
        elif trend < -config.regime_trend_threshold:
            return MarketRegime.BEAR
        else:
            return MarketRegime.RECOVERY
    elif vol > config.regime_vol_threshold:  # 高波动
        if trend < -config.regime_trend_threshold:
            return MarketRegime.BEAR
        elif trend > config.regime_trend_threshold:
            return MarketRegime.RECOVERY
        else:
            return MarketRegime.RANGE
    else:  # 正常波动
        if trend > config.regime_trend_threshold:
            return MarketRegime.BULL
        elif trend < -config.regime_trend_threshold:
            return MarketRegime.BEAR
        else:
            return MarketRegime.RANGE


def calculate_risk_parity_weights(
    strategy_returns: Dict[str, pd.Series],
    current_date: pd.Timestamp,
    lookback: int = 60,
) -> Dict[str, float]:
    """
    计算风险平价权重
    
    原理: 各策略对组合风险的贡献相等
    w_i ∝ 1 / σ_i
    
    Args:
        strategy_returns: {策略名: 日收益率序列}
        current_date: 当前日期
        lookback: 波动率计算窗口
    
    Returns:
        {策略名: 权重} (归一化到1.0)
    """
    inv_vols = {}
    
    for name, rets in strategy_returns.items():
        available = rets[rets.index <= current_date]
        if len(available) >= lookback // 2:
            recent = available.iloc[-lookback:]
            vol = recent.std() * np.sqrt(252)  # 年化波动率
            if vol > 0:
                inv_vols[name] = 1.0 / vol
            else:
                inv_vols[name] = 0.0
        else:
            inv_vols[name] = 1.0  # 数据不足，默认等权
    
    total = sum(inv_vols.values())
    if total == 0:
        n = len(inv_vols)
        return {k: 1.0/n for k in inv_vols}
    
    return {k: v / total for k, v in inv_vols.items()}


def calculate_momentum_weights(
    strategy_returns: Dict[str, pd.Series],
    current_date: pd.Timestamp,
    lookback: int = 60,
) -> Dict[str, float]:
    """
    计算动量加权权重
    
    原理: 近期表现好的策略给予更高权重
    w_i ∝ exp(λ * r_i)
    
    Args:
        strategy_returns: {策略名: 日收益率序列}
        current_date: 当前日期
        lookback: 业绩回看窗口
    
    Returns:
        {策略名: 权重} (归一化到1.0)
    """
    scores = {}
    
    for name, rets in strategy_returns.items():
        available = rets[rets.index <= current_date]
        if len(available) >= lookback // 2:
            recent = available.iloc[-lookback:]
            # 使用夏普近似: 收益/波动
            ret = (1 + recent).prod() - 1
            vol = recent.std() * np.sqrt(252)
            if vol > 0:
                score = ret / vol
            else:
                score = 0
            scores[name] = max(score, 0)  # 负收益的策略不给权重
        else:
            scores[name] = 1.0  # 数据不足，默认等权
    
    # Softmax 归一化
    exp_scores = {k: np.exp(v) for k, v in scores.items()}
    total = sum(exp_scores.values())
    if total == 0:
        n = len(exp_scores)
        return {k: 1.0/n for k in exp_scores}
    
    return {k: v / total for k, v in exp_scores.items()}


def adjust_weights_for_regime(
    base_weights: Dict[str, float],
    regime: MarketRegime,
    config: HybridConfig,
    sub_strategies: List[SubStrategy],
) -> Dict[str, float]:
    """
    根据市场状态调整权重
    
    - 牛市: 增加进攻型策略权重 (微盘)
    - 熊市: 增加防守型策略权重 (低波/现金)
    - 暴跌: 大幅降低所有策略权重，增加现金
    
    Args:
        base_weights: 基础权重
        regime: 当前市场状态
        config: 配置
        sub_strategies: 子策略列表
    
    Returns:
        调整后的权重
    """
    # 根据市场状态定义策略偏好
    regime_preferences = {
        MarketRegime.BULL: {"micro": 1.5, "composite": 1.2, "lowvol": 0.8, "cash": 0.0},
        MarketRegime.BEAR: {"micro": 0.5, "composite": 0.8, "lowvol": 1.5, "cash": 1.0},
        MarketRegime.RANGE: {"micro": 1.0, "composite": 1.0, "lowvol": 1.0, "cash": 0.5},
        MarketRegime.CRASH: {"micro": 0.0, "composite": 0.3, "lowvol": 0.5, "cash": 2.0},
        MarketRegime.RECOVERY: {"micro": 1.2, "composite": 1.0, "lowvol": 0.9, "cash": 0.3},
    }
    
    prefs = regime_preferences.get(regime, regime_preferences[MarketRegime.RANGE])
    
    # 为每个子策略应用偏好系数
    adjusted = {}
    for sub in sub_strategies:
        base = base_weights.get(sub.name, 0.0)
        # 根据策略类型匹配偏好
        if "micro" in sub.name.lower() or "市值" in sub.name:
            pref = prefs.get("micro", 1.0)
        elif "composite" in sub.name.lower() or "综合" in sub.name or "多因子" in sub.name:
            pref = prefs.get("composite", 1.0)
        elif "lowvol" in sub.name.lower() or "低波" in sub.name:
            pref = prefs.get("lowvol", 1.0)
        else:
            pref = 1.0
        
        adjusted[sub.name] = base * pref
    
    # 归一化
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    
    return adjusted


def constrain_weights(
    weights: Dict[str, float],
    sub_strategies: List[SubStrategy],
) -> Dict[str, float]:
    """
    约束权重在最小/最大范围内
    
    如果权重超出范围，进行截断并重新归一化
    """
    constrained = {}
    
    # 第一步: 截断到上下限
    for sub in sub_strategies:
        w = weights.get(sub.name, 0.0)
        constrained[sub.name] = np.clip(w, sub.min_weight, sub.max_weight)
    
    # 第二步: 重新归一化
    total = sum(constrained.values())
    if total > 0:
        constrained = {k: v / total for k, v in constrained.items()}
    
    return constrained


class HybridStrategy:
    """
    混合策略引擎
    
    组合多个子策略，根据市场状态动态调整权重
    """
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.sub_strategies = config.sub_strategies or []
        self.current_regime = MarketRegime.RANGE
        self.current_weights = {}
        self.weight_history = []  # 记录权重变化
        
    def initialize(self, market_returns: pd.Series, start_date: pd.Timestamp):
        """初始化权重"""
        n = len(self.sub_strategies)
        if n > 0:
            self.current_weights = {s.name: 1.0/n for s in self.sub_strategies}
        else:
            self.current_weights = {}
    
    def update_weights(
        self,
        market_returns: pd.Series,
        strategy_returns: Dict[str, pd.Series],
        current_date: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        更新权重
        
        流程:
        1. 检测市场状态
        2. 计算风险平价权重
        3. 计算动量权重
        4. 混合两种权重
        5. 根据市场状态调整
        6. 约束到上下限
        7. 平滑过渡 (避免权重突变)
        """
        if not self.sub_strategies:
            return {}
        
        # 1. 检测市场状态
        self.current_regime = detect_market_regime(
            market_returns, self.config, current_date
        )
        
        # 2. 风险平价权重
        rp_weights = calculate_risk_parity_weights(
            strategy_returns, current_date,
            lookback=self.config.sub_strategies[0].volatility_lookback if self.sub_strategies else 60
        )
        
        # 3. 动量权重
        mom_weights = calculate_momentum_weights(
            strategy_returns, current_date,
            lookback=self.config.sub_strategies[0].performance_lookback if self.sub_strategies else 60
        )
        
        # 4. 混合权重 (50% RP + 50% Momentum)
        base_weights = {}
        for name in rp_weights:
            rp_w = rp_weights.get(name, 0.0)
            mom_w = mom_weights.get(name, 0.0)
            base_weights[name] = 0.5 * rp_w + 0.5 * mom_w
        
        # 5. 根据市场状态调整
        regime_weights = adjust_weights_for_regime(
            base_weights, self.current_regime, self.config, self.sub_strategies
        )
        
        # 6. 约束到上下限
        constrained = constrain_weights(regime_weights, self.sub_strategies)
        
        # 7. 平滑过渡 (EMA风格)
        if self.current_weights:
            alpha = self.config.weight_adjustment_speed
            smoothed = {}
            for name in constrained:
                old_w = self.current_weights.get(name, 0.0)
                new_w = constrained[name]
                smoothed[name] = alpha * new_w + (1 - alpha) * old_w
            
            # 重新归一化
            total = sum(smoothed.values())
            if total > 0:
                smoothed = {k: v / total for k, v in smoothed.items()}
            self.current_weights = smoothed
        else:
            self.current_weights = constrained
        
        # 记录历史
        self.weight_history.append({
            'date': current_date,
            'regime': self.current_regime.value,
            **self.current_weights
        })
        
        return self.current_weights
    
    def get_cash_weight(self) -> float:
        """获取当前现金权重"""
        regime_cash = {
            MarketRegime.BULL: self.config.cash_buffer_bull,
            MarketRegime.BEAR: self.config.cash_buffer_bear,
            MarketRegime.RANGE: self.config.cash_buffer_range,
            MarketRegime.CRASH: self.config.cash_buffer_crash,
            MarketRegime.RECOVERY: self.config.cash_buffer_recovery,
        }
        return regime_cash.get(self.current_regime, 0.20)


def create_default_hybrid_config(
    micro_strategy: dict,
    composite_strategy: dict,
    lowvol_strategy: dict,
) -> HybridConfig:
    """
    创建默认混合策略配置
    
    包含3个子策略:
    - 微盘纯市值 (进攻)
    - 多因子综合 (平衡)
    - 低波小市值 (防守)
    """
    subs = [
        SubStrategy(
            name="micro_cap",
            strategy_def=micro_strategy,
            weight=0.40,
            max_weight=0.60,
            min_weight=0.15,
        ),
        SubStrategy(
            name="composite",
            strategy_def=composite_strategy,
            weight=0.35,
            max_weight=0.50,
            min_weight=0.15,
        ),
        SubStrategy(
            name="lowvol",
            strategy_def=lowvol_strategy,
            weight=0.25,
            max_weight=0.40,
            min_weight=0.10,
        ),
    ]
    
    return HybridConfig(
        name="混合策略: 微盘+多因子+低波",
        sub_strategies=subs,
        regime_lookback_short=20,
        regime_lookback_long=60,
        regime_vol_threshold=0.25,
        regime_trend_threshold=0.03,
        cash_buffer_bull=0.10,
        cash_buffer_bear=0.35,
        cash_buffer_range=0.20,
        cash_buffer_crash=0.50,
        cash_buffer_recovery=0.20,
        risk_parity_target_vol=0.20,
        risk_parity_enabled=True,
        weight_adjustment_speed=0.3,
    )
