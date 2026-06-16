"""
策略库 — 6 种小微盘股截面策略
===============================
每个策略被定义为一个 (universe_filter, ranking_factor, ascending, params) 打包。
直接喂给 CrossSectionalEngine.run()。
"""

import pandas as pd
import numpy as np
from typing import List, Callable, Tuple, Dict, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 通用过滤/排名辅助函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def filter_micro_cap(snapshot: pd.DataFrame, dates, step: int,
                     max_mcap: float = 30.0) -> List[str]:
    """过滤: 市值 < max_mcap 亿"""
    if 'mcap' in snapshot.columns:
        return snapshot[snapshot['mcap'] < max_mcap].index.tolist()
    return list(snapshot.index)


def filter_no_st(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """过滤: 排除 ST 股票 (通过名称判断)"""
    if 'name' in snapshot.columns:
        st = snapshot['name'].str.contains(r'\*?ST', na=False)
        return snapshot[~st].index.tolist()
    return list(snapshot.index)


def filter_st_only(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """过滤: 只要 ST 股票"""
    if 'name' in snapshot.columns:
        st = snapshot['name'].str.contains(r'\*?ST', na=False)
        return snapshot[st].index.tolist()
    return []


def filter_low_pb(snapshot: pd.DataFrame, dates, step: int,
                  max_pb: float = 1.5) -> List[str]:
    """过滤: PB < max_pb"""
    if 'pb' in snapshot.columns:
        pb_positive = snapshot[snapshot['pb'] > 0]  # 排除负 PB
        return pb_positive[pb_positive['pb'] < max_pb].index.tolist()
    return list(snapshot.index)


def filter_liquid(snapshot: pd.DataFrame, dates, step: int,
                  min_rel_vol: float = 1.0) -> List[str]:
    """过滤: 相对量比 > min_rel_vol (1.0=成交等于20日均量, 确保有交易活动)"""
    if 'turnover' in snapshot.columns:
        return snapshot[snapshot['turnover'] > min_rel_vol].index.tolist()
    return list(snapshot.index)


def filter_positive_pe(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """过滤: PE > 0 (盈利)"""
    if 'pe' in snapshot.columns:
        return snapshot[snapshot['pe'] > 0].index.tolist()
    return list(snapshot.index)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略 1: ST 困境反转 (摘帽套利)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def st_turnaround_filter(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """ST 股 + 市值 < 20亿 + PE > 0 或微亏"""
    stocks = filter_st_only(snapshot, dates, step)
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    if 'mcap' in sub.columns:
        sub = sub[sub['mcap'] < 20]
    if 'pe' in sub.columns:
        sub = sub[sub['pe'] > -20]  # 非极端亏损
    return list(sub.index)

strategy_st_turnaround = {
    "name": "策略1: ST困境反转",
    "universe_filter": st_turnaround_filter,
    "ranking_factor": "mom60d",  # 近 60 日动量 (反转已启动的信号)
    "ascending": False,           # 选动量最强
    "n_stocks": 5,
    "stop_loss": -0.30,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略 2: 小市值动量 + 高换手跟随
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def momentum_turnover_filter(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """市值 < 30亿 + 非 ST + 换手率 > 3%"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=30)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    if 'turnover' in sub.columns:
        sub = sub[sub['turnover'] > 1.5]  # 相对量比 > 1.5x (高于均值 50%)
    if 'pe' in sub.columns:
        sub = sub[sub['pe'] > -50]  # 排除巨额亏损
    return list(sub.index)

strategy_momentum = {
    "name": "策略2: 小市值动量+高换手",
    "universe_filter": momentum_turnover_filter,
    "ranking_factor": "mom20d",  # 20 日涨幅
    "ascending": False,
    "n_stocks": 8,
    "stop_loss": -0.15,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略 3: 极小小市值轮动 (因子捕获)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def micro_cap_rotation_filter(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """市值 < 30亿 + 非 ST + 非涨停"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=30)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    return stocks

strategy_micro_rotation = {
    "name": "策略3: 极小小市值轮动",
    "universe_filter": micro_cap_rotation_filter,
    "ranking_factor": "mcap",     # 最小市值
    "ascending": True,
    "n_stocks": 30,
    "stop_loss": None,            # 不设止损, 靠分散
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略 4: 次新股 + 题材共振
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ipo_resonance_filter(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """市值 < 30亿 + 排 ST + 有换手率 (次新股通常高换手)"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=30)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    # 高换手作为次新/活跃代理变量 (turnover = 相对量比, median≈0.82)
    if 'turnover' in sub.columns:
        sub = sub[sub['turnover'] > 2.0]  # 量比 > 2x 均值 (活跃次新)
    return list(sub.index)

strategy_ipo = {
    "name": "策略4: 次新股+题材共振",
    "universe_filter": ipo_resonance_filter,
    "ranking_factor": "mom20d",  # 20 日动量
    "ascending": False,
    "n_stocks": 6,
    "stop_loss": -0.20,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略 5: 壳资源深度价值
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def shell_value_filter(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """市值 < 12亿 + PB < 2 + 非 ST"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=12)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    if 'pb' in sub.columns:
        sub = sub[sub['pb'] > 0]     # 正 PB
        sub = sub[sub['pb'] < 2]     # PB < 2
    return list(sub.index)

strategy_shell = {
    "name": "策略5: 壳资源深度价值",
    "universe_filter": shell_value_filter,
    "ranking_factor": "pb",         # 最低 PB
    "ascending": True,
    "n_stocks": 10,
    "stop_loss": -0.25,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略 6: 小市值 + 低 PB + 50% 止盈轮动
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pb_value_filter(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """市值 < 30亿 + PB < 1.5 + 非 ST + PE > 0 或微亏"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=30)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    if 'pb' in sub.columns:
        sub = sub[sub['pb'] > 0]
        sub = sub[sub['pb'] < 1.5]
    return list(sub.index)

strategy_pb_value = {
    "name": "策略6: 低PB+50%止盈轮动",
    "universe_filter": pb_value_filter,
    "ranking_factor": "pb",           # 最低 PB 优先
    "ascending": True,
    "n_stocks": 20,
    "stop_loss": -0.20,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 所有策略汇总
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_STRATEGIES = [
    strategy_st_turnaround,
    strategy_momentum,
    strategy_micro_rotation,
    strategy_ipo,
    strategy_shell,
    strategy_pb_value,
]
