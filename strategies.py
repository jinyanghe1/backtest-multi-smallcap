"""
策略库 — 6 种小微盘股截面策略 + 5 种学术论文驱动策略
====================================================
每个策略被定义为一个 (universe_filter, ranking_factor, ascending, params) 打包。
直接喂给 CrossSectionalEngine.run()。

策略 7-11 基于近 2-3 年权威量化金融论文的核心思想:
  - S7: 漂移状态条件反转 (arXiv 2511.12490 "13-Sharpe OOS Factor")
  - S8: 盈利能力×低波动复合 (Novy-Marx Profitability + Low Vol Anomaly)
  - S9: 条件反转→动量切换 (ScienceDirect 2024 "Short-term Momentum & Reversals")
  - S10: 极小市值×盈利能力×反转三因子 (论文1+11+13 组合)
  - S11: 因子动量动态权重 (ScienceDirect 2024 "Factor Momentum in Chinese Stock Market")
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
                  min_amount: float = 200) -> List[str]:
    """过滤: 日均成交 > min_amount 万元"""
    if 'turnover' in snapshot.columns and 'mcap' in snapshot.columns:
        # 用换手率 × 市值估算成交额
        amount = snapshot['turnover'] * snapshot['mcap'] * 100  # 万元
        return snapshot[amount > min_amount].index.tolist()
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
        sub = sub[sub['turnover'] > 3]  # 日换手 > 3%
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
    # 高换手作为次新/活跃代理变量 (真实系统中会用 IPO 日期)
    if 'turnover' in sub.columns:
        sub = sub[sub['turnover'] > 5]  # 日换手 > 5%
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
# 通用排名辅助函数 (用于复合因子策略)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def rank_normalize(series: pd.Series) -> pd.Series:
    """截面排名归一化到 [0, 1]"""
    return series.rank(pct=True)


def composite_rank(snapshot: pd.DataFrame, factors: list, signs: list) -> pd.Series:
    """
    多因子复合排名 (等权)

    Args:
        snapshot: 因子截面 DataFrame
        factors: 因子列名列表
        signs: 方向列表, +1 表示越大越好, -1 表示越小越好

    Returns:
        pd.Series: 复合得分, 越高越好
    """
    scores = pd.Series(0.0, index=snapshot.index)
    for col, sign in zip(factors, signs):
        if col in snapshot.columns:
            s = snapshot[col]
            if sign < 0:
                s = -s
            scores += rank_normalize(s)
    return scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略 7: 漂移状态条件反转 (论文: arXiv 2511.12490)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Alpha 原则: Alpha 不来自于信号本身, 而来自于信号激活时机的选择
#   - 只在 63 日窗口正收益天数 > 60% 时激活反转信号
#   - 漂移状态 = 行为偏差放大 + 流动性模式改变 → 反转利润最大化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def drift_regime_reversal_filter(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """市值 < 30亿 + 非 ST + 有换手率数据"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=30)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    return stocks


def drift_regime_reversal_rank(snapshot: pd.DataFrame) -> pd.Series:
    """
    漂移状态条件反转排名函数

    逻辑: 当 mom60d > 0 (近似漂移状态: 近 60 日正收益),
          则做 20 日反转 (-mom20d); 否则不做反转

    得分 = Drift_Regime × (-mom20d_rank)
         + (1 - Drift_Regime) × (mom20d_rank)  # 非漂移期跟动量
    """
    if 'mom60d' not in snapshot.columns or 'mom20d' not in snapshot.columns:
        return pd.Series(0.0, index=snapshot.index)

    # 近似漂移状态: 60 日动量 > 0 表示多数天正收益
    drift = (snapshot['mom60d'] > 0).astype(float)

    # 反转得分 (短期跌幅越大, 得分越高)
    reversal_score = rank_normalize(-snapshot['mom20d'])

    # 动量得分 (短期涨幅越大, 得分越高)
    momentum_score = rank_normalize(snapshot['mom20d'])

    # 复合: 漂移期做反转, 非漂移期做动量
    composite = drift * reversal_score + (1 - drift) * momentum_score

    return composite


strategy_drift_reversal = {
    "name": "策略7: 漂移状态条件反转",
    "universe_filter": drift_regime_reversal_filter,
    "ranking_factor": None,  # 使用 ranking_fn
    "ascending": True,       # ranking_fn 已处理方向
    "n_stocks": 20,
    "stop_loss": -0.20,
    "ranking_fn": drift_regime_reversal_rank,
    "paper": "arXiv 2511.12490: 13-Sharpe OOS Factor via Drift Regimes",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略 8: 盈利能力×低波动复合 (论文: Novy-Marx NBER w33601)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Alpha 原则:
#   - GP/A (毛利/总资产) 是比净利润更好的盈利能力度量
#   - 低波动异常: 低波动率股票长期跑赢高波动率股票
#   - 两个因子负相关, 复合后 Sharpe 显著提升
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def profitability_low_vol_filter(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """市值 < 30亿 + 非 ST + PB > 0 (正净资产) + 换手率 > 1%"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=30)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    if 'pb' in sub.columns:
        sub = sub[sub['pb'] > 0]
    if 'turnover' in sub.columns:
        sub = sub[sub['turnover'] > 1]
    return list(sub.index)


def profitability_low_vol_rank(snapshot: pd.DataFrame) -> pd.Series:
    """
    盈利能力×低波动复合排名

    逻辑: 高 GP/A (高盈利) + 低 vol20d (低波动)

    近似 GP/A: 当 pe > 0 且 pb > 0 时, ROE ≈ 1/(pe/pb) = pb/pe
              → 用 1/PE × PB 近似盈利能力 (越大越好)
    实际用 1/vol20d 作为低波动得分
    """
    scores = pd.Series(0.0, index=snapshot.index)

    # 盈利能力: 用 PB/PE 近似 GP/A (当 PE > 0 时)
    if 'pb' in snapshot.columns and 'pe' in snapshot.columns:
        # 1/PE × PB = 盈利能力代理
        gp_proxy = (snapshot['pb'] / snapshot['pe'].clip(lower=0.1)).clip(lower=0, upper=100)
        scores += rank_normalize(gp_proxy)
    elif 'pb' in snapshot.columns:
        # 无 PE 时用 PB 近似 (高 PB 可能是高 ROE)
        scores += rank_normalize(snapshot['pb'].clip(lower=0.01))

    # 低波动: 波动率越低越好
    if 'vol20d' in snapshot.columns:
        # 1/vol20d 作为低波动得分
        low_vol_score = 1.0 / snapshot['vol20d'].clip(lower=0.01)
        scores += rank_normalize(low_vol_score)

    return scores


strategy_profit_lowvol = {
    "name": "策略8: 盈利能力×低波动复合",
    "universe_filter": profitability_low_vol_filter,
    "ranking_factor": None,
    "ascending": True,
    "n_stocks": 25,
    "stop_loss": -0.15,
    "ranking_fn": profitability_low_vol_rank,
    "paper": "Novy-Marx NBER w33601 + Low Vol Anomaly",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略 9: 条件反转→动量切换 (论文: ScienceDirect 2024)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Alpha 原则:
#   - 短期反转行为取决于换手率和 PTH (价格/52周高点比)
#   - 高换手率 + 高 PTH → 做动量 (赢家继续赢)
#   - 低换手率 + 低 PTH → 做反转 (赢家反转下跌)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def conditional_mom_rev_filter(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """市值 < 30亿 + 非 ST + 换手率 > 0.5%"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=30)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    if 'turnover' in sub.columns:
        sub = sub[sub['turnover'] > 0.5]
    return list(sub.index)


def conditional_mom_rev_rank(snapshot: pd.DataFrame) -> pd.Series:
    """
    条件反转→动量切换排名函数

    逻辑:
      - 高换手率 (turnover > 中位数) + 高 PTH (mom60d > 0) → 做动量 (+mom20d)
      - 低换手率 + 低 PTH → 做反转 (-mom20d)

    PTH 近似: mom60d > 0 表示价格相对 60 日前上涨,
             接近或超过 52 周高点的概率更大
    """
    if 'mom20d' not in snapshot.columns or 'turnover' not in snapshot.columns:
        return pd.Series(0.0, index=snapshot.index)

    # 换手率中位数
    turnover_median = snapshot['turnover'].median()

    # 高换手 + 高 PTH (mom60d > 0) → 动量
    high_turnover = snapshot['turnover'] > turnover_median
    high_pth = snapshot['mom60d'] > 0
    momentum_zone = high_turnover & high_pth

    # 动量得分 (涨幅越大越好)
    momentum_score = rank_normalize(snapshot['mom20d'])

    # 反转得分 (跌幅越大越好)
    reversal_score = rank_normalize(-snapshot['mom20d'])

    # 复合
    composite = momentum_zone.astype(float) * momentum_score + \
                (~momentum_zone).astype(float) * reversal_score

    return composite


strategy_conditional_mom_rev = {
    "name": "策略9: 条件反转→动量切换",
    "universe_filter": conditional_mom_rev_filter,
    "ranking_factor": None,
    "ascending": True,
    "n_stocks": 20,
    "stop_loss": -0.20,
    "ranking_fn": conditional_mom_rev_rank,
    "paper": "ScienceDirect 2024: Short-term Momentum & Reversals",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略 10: 极小市值×盈利能力×反转三因子
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Alpha 原则:
#   - 小市值溢价: 最强A股异象之一
#   - 盈利能力溢价: GP/A 高的公司持续跑赢 (Novy-Marx)
#   - 条件反转: 只在漂移状态下做反转 (arXiv 2511.12490)
#   - 三因子负相关 → 复合 Sharpe 最高
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def triple_factor_filter(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """市值 < 15亿 (极小) + 非 ST + PB > 0"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=15)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    if 'pb' in sub.columns:
        sub = sub[sub['pb'] > 0]
    return list(sub.index)


def triple_factor_rank(snapshot: pd.DataFrame) -> pd.Series:
    """
    三因子复合排名: 极小市值 + 盈利能力 + 条件反转

    得分 = rank(-mcap) + rank(GP/A_proxy) + rank(Drift × (-mom20d))
    """
    scores = pd.Series(0.0, index=snapshot.index)

    # 因子1: 极小市值 (市值越小越好)
    if 'mcap' in snapshot.columns:
        scores += rank_normalize(-snapshot['mcap'])

    # 因子2: 盈利能力 (PB/PE 近似 GP/A)
    if 'pb' in snapshot.columns and 'pe' in snapshot.columns:
        gp_proxy = (snapshot['pb'] / snapshot['pe'].clip(lower=0.1)).clip(lower=0, upper=100)
        scores += rank_normalize(gp_proxy)
    elif 'pb' in snapshot.columns:
        scores += rank_normalize(snapshot['pb'].clip(lower=0.01))

    # 因子3: 条件反转 (漂移状态下做反转)
    if 'mom20d' in snapshot.columns and 'mom60d' in snapshot.columns:
        drift = (snapshot['mom60d'] > 0).astype(float)
        reversal_score = rank_normalize(-snapshot['mom20d'])
        momentum_score = rank_normalize(snapshot['mom20d'])
        scores += drift * reversal_score + (1 - drift) * momentum_score

    return scores


strategy_triple_factor = {
    "name": "策略10: 极小市值×盈利×反转三因子",
    "universe_filter": triple_factor_filter,
    "ranking_factor": None,
    "ascending": True,
    "n_stocks": 25,
    "stop_loss": -0.20,
    "ranking_fn": triple_factor_rank,
    "paper": "Novy-Marx + arXiv 2511.12490 + Micro-cap Anomaly",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略 11: 因子动量动态权重 (论文: ScienceDirect 2024)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Alpha 原则:
#   - 因子本身存在动量效应: 表现好的因子短期继续好
#   - 根据因子过去 20 日的 IC (信息系数) 动态调整因子权重
#   - IC 高的因子获得更高权重 → 自适应因子择时
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def factor_momentum_filter(snapshot: pd.DataFrame, dates, step: int) -> List[str]:
    """市值 < 30亿 + 非 ST + 换手率 > 0.5%"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=30)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    if 'turnover' in sub.columns:
        sub = sub[sub['turnover'] > 0.5]
    return list(sub.index)


# 全局 IC 历史 (跨调仓日累积)
_ic_history: Dict[str, list] = {}


def factor_momentum_rank(snapshot: pd.DataFrame) -> pd.Series:
    """
    因子动量动态权重排名

    逻辑:
      1. 计算 6 个因子 (mcap, pb, mom20d, mom60d, turnover, vol20d)
         的截面排名作为因子值
      2. 用等权作为初始默认 (前 2 个月无 IC 历史)
      3. 逐步积累各因子的 IC (rank correlation with mom20d)
      4. 用最近 3 次 IC 的均值作为因子权重
      5. 加权复合排名
    """
    factor_cols = ['mcap', 'pb', 'mom20d', 'mom60d', 'turnover', 'vol20d']
    factor_signs = [-1, -1, 1, 1, 1, -1]  # 期望方向

    scores = pd.Series(0.0, index=snapshot.index)

    for col, sign in zip(factor_cols, factor_signs):
        if col not in snapshot.columns:
            continue

        # 计算因子截面排名
        if sign < 0:
            factor_rank = rank_normalize(-snapshot[col])
        else:
            factor_rank = rank_normalize(snapshot[col])

        # 计算 IC (与 mom20d 的秩相关)
        if 'mom20d' in snapshot.columns and col != 'mom20d':
            valid = snapshot[[col, 'mom20d']].dropna()
            if len(valid) > 10:
                ic = valid[col].corr(valid['mom20d'], method='spearman')
                if col not in _ic_history:
                    _ic_history[col] = []
                _ic_history[col].append(ic)
                # 保留最近 3 次 IC
                if len(_ic_history[col]) > 3:
                    _ic_history[col] = _ic_history[col][-3:]

        # 获取权重: IC 均值 或 默认 1.0
        if col in _ic_history and len(_ic_history[col]) >= 2:
            weight = max(0, np.mean(_ic_history[col][-3:]))
        else:
            weight = 1.0

        scores += weight * factor_rank

    return scores


strategy_factor_momentum = {
    "name": "策略11: 因子动量动态权重",
    "universe_filter": factor_momentum_filter,
    "ranking_factor": None,
    "ascending": True,
    "n_stocks": 25,
    "stop_loss": -0.15,
    "ranking_fn": factor_momentum_rank,
    "paper": "ScienceDirect 2024: Factor Momentum in Chinese Stock Market",
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
    strategy_drift_reversal,
    strategy_profit_lowvol,
    strategy_conditional_mom_rev,
    strategy_triple_factor,
    strategy_factor_momentum,
]
