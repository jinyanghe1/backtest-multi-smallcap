"""
微盘股量化策略 — 文献综述与新策略设计
======================================
基于 2022-2026 年学术文献和业界研究的综合分析。
"""

import pandas as pd
import numpy as np
from typing import List, Callable
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ══════════════════════════════════════════════════════════════════════════════
# PART I: 文献综述 — 微盘股超额收益的 8 大机制
# ══════════════════════════════════════════════════════════════════════════════

LITERATURE_REVIEW = """
===========================================================================
  微盘股量化策略 — 文献综述 (2022-2026)
===========================================================================

一、规模溢价 (Size Premium) — A股最强单因子之一
──────────────────────────────────────────────
• Liu, Stambaugh & Yuan (2019, JFE): 中国 A 股特有的"壳价值污染"使传统市值因子失真,
  剔除最小 30% 股票后, 规模溢价月均 0.55% (t=3.2)
• Tsinghua PBCSF 426 异象研究 (2024): 市值因子 t-stat=4.8, 在所有 426 个异象中
  排名前 5%, 设定 2.85 多重检验门槛后仍然显著
• 雪球量化社区实测 (2025): 微盘股 2019-2024 年月度轮动年化 35%+, 夏普 1.2+
→ 机制: 流动性折价 + 信息不对称 + 散户定价偏差 + 壳价值
→ 适用性: 微盘股中效果最强, 大盘股中几乎消失

二、短期反转效应 (Short-Term Reversal) — A股最稳健的 alpha
──────────────────────────────────────────────
• Jegadeesh & Titman (1993, JF): 1 月反转是全球股市普遍现象
• 清华 PBCSF (2024): A 股 1 月反转 t=5.1, 是最显著异象之一
• Jiao & Zheng (2026, Pacific-Basin Finance Journal): 聚类增强反转策略,
  通过 K-means 聚类将股票分组后, 组内反转策略显著优于传统反转,
  月均多空收益提升 0.32%
• 机制: 散户过度反应 → 价格回摆; 流动性冲击 → 短暂的错误定价
→ A股因散户占比高 (约 65% 交易量), 反转效应比美股强 2-3 倍
→ 窗口: 1-4 周最优, 1 月次之, 6 月以上反转减弱

三、特质波动率之谜 (IVOL Puzzle) — 高风险≠高收益
──────────────────────────────────────────────
• Ang, Hodrick, Xing & Zhang (2006, JF; 2009, JFE): 美股中, 高特质波动率股票
  未来收益显著低于低特质波动率股票
• NYU Shanghai (Su, 2025): A 股同样存在 IVOL 异象, 但受到彩票偏好
  (MAX effect) 的调节 — 在散户占比高的股票中, IVOL 折价更强
• 机制: 套利限制 + 彩票偏好 → 高波动股票被散户追高 → 未来回归
→ 微盘股中 IVOL 异象特别明显 (套利者无法做空微盘股)

四、彩票型股票折价 (MAX/Lottery Effect)
──────────────────────────────────────────────
• Bali, Cakici & Whitelaw (2011, JFE): 上月最大日收益率 (MAX) 最高的股票,
  下月收益显著低于最低组, 月均差异 -1.03%
• 中国 A 股验证 (2022, 系统科学与数学): 彩票特征解释了 A 股 34% 的 IVOL 异象
→ 微盘股中 MAX 效应因散户追逐涨停而加剧

五、流动性溢价 (Illiquidity Premium)
──────────────────────────────────────────────
• Amihud (2002, JFM): 非流动性测度 (ILLIQ = |return|/volume) 正向预测收益
• A 股微盘股中, Amihud 非流动性因子月均溢价 0.3-0.5%
→ 机制: 流动性差的股票要求更高风险补偿; 微盘股日均成交 < 500 万的溢价最显著

六、低波动异象 (Low Volatility Anomaly)
──────────────────────────────────────────────
• 东方财富 (2025): A 股低波动组合长期跑赢高波动组合
• Su (2025): Beta 异象在 A 股存在但弱于美股; 在散户主导的小盘股中通过
  彩票偏好渠道放大
→ 与 IVOL 异象同源, 可组合使用

七、股东户数效应 — A 股特有的筹码因子
──────────────────────────────────────────────
• 源达信息 (2026): 股东户数减少 (筹码集中) 在未来 1-3 月正向预测收益,
  Rank IC = -0.087 至 -0.125, 在小市值区间 (62-113 亿) 最显著
→ 机制: 筹码集中 → 主力吸筹 → 未来拉升; 散户增加 → 筹码分散 → 上涨阻力

八、残差反转 — 剥离基本面后的纯 alpha
──────────────────────────────────────────────
• 系统科学与数学 (2024): 用分析师预测修正测度现金流新闻,
  构建的残差短期反转策略在 A 股有显著超额收益
→ 剥离市场和行业收益后的残差反转, 比价格反转更纯净

===========================================================================
  关键结论: 微盘股策略的 3 大 alpha 来源
===========================================================================
1. 结构性的: 市值折价 + 流动性折价 (不可消除, 合理风险补偿)
2. 行为性的: 散户过度反应 + 彩票偏好 (可捕获, 较稳定)
3. 制度性的: 壳价值 + 筹码集中 + IPO 管制 (A 股特有, 政策敏感)

最优组合: 结构 (长期配置) + 行为 (月度轮动) + 制度 (事件驱动)
"""

# ══════════════════════════════════════════════════════════════════════════════
# PART II: 三个新策略设计 (第二版, 基于第一轮回测校准)
# ══════════════════════════════════════════════════════════════════════════════
#
# 第一轮教训:
# - 纯反转(买跌最多)在微盘失效 → 很多"输家"是真烂, 非均值回归
# - 低波动作为二次过滤有效, 但单因子不够
# - 正确方向: 基于已证明有效的策略3/5做增强, 而非引入不稳定的新因子
#
# 三个新策略的设计哲学:
#   A: 低波小市值 — 策略3 + 低波过滤 (质量增强版)
#   B: Size+PB 复合 — 策略3+5 的融合 (双因子安全边际)
#   C: 低波+缩量信号 — 筹码集中代理 (制度性 alpha)

from tools.backtest_mvp.strategies import (
    filter_micro_cap, filter_no_st, filter_low_pb, filter_liquid
)

# ──────────────────────────────────────────────────────────────────────────────
# 策略 A: 低波小市值 (Small-Cap Low Vol) — 稳健小盘
# ──────────────────────────────────────────────────────────────────────────────
# 文献基础: Ang et al. (2006) + NYU Shanghai (2025) + Size premium
# 逻辑: 最小市值中最稳定的 → 获得规模溢价而无极端波动
#       vol20d 过滤排除"彩票型"微盘股, 保留真实经营的小公司
# 预期: 夏普 > 0.6, 回撤 < 40%

def small_lowvol_filter(snapshot, dates, step):
    """微盘 + 非ST + 低波动 + 流动性"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=30)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    # 过滤: 波动率 < 55% 年化 (排除极端波动的"彩票型"股票)
    if 'vol20d' in sub.columns:
        sub = sub[sub['vol20d'] < 0.55]
    # 过滤: 有流动性
    if 'turnover' in sub.columns:
        sub = sub[sub['turnover'] > 0.3]
    return list(sub.index)

strategy_reversal_lowvol = {
    "name": "策略A: 低波小市值",
    "universe_filter": small_lowvol_filter,
    "ranking_factor": "mcap",        # 最小市值优先
    "ascending": True,
    "n_stocks": 30,
    "stop_loss": None,
}

# ──────────────────────────────────────────────────────────────────────────────
# 策略 B: 多因子综合 (Size+PB Composite) — 价投小盘
# ──────────────────────────────────────────────────────────────────────────────
# 文献基础: Fama-French 5-factor + Liu-Stambaugh-Yuan (2019) EP factor
# 逻辑: 复合排名 = rank(mcap_asc) + rank(pb_asc)
#       同时选小且便宜 → 双重安全边际
#       这是策略3(纯市值)和策略5(壳资源)的改进版 — 两因子结合
# 预期: 夏普 > 0.75, 年化 > 35%

def composite_filter(snapshot, dates, step):
    """微盘 + 非ST + PB>0 (排除负资产)"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=30)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    if 'pb' in sub.columns:
        sub = sub[sub['pb'] > 0]  # 正PB
    return list(sub.index)

strategy_size_lowvol_mom = {
    "name": "策略B: 多因子综合(Size+PB)",
    "universe_filter": composite_filter,
    "ranking_factor": "mcap",        # 占位 — 在engine层手动复合
    "ascending": True,
    "n_stocks": 25,
    "stop_loss": None,
    # 复合排名将在 engine 外部处理 (先按 mcap 排序, 再按 pb 排序, 取交集)
}

# ──────────────────────────────────────────────────────────────────────────────
# 策略 C: 低波+低换手 (Accumulation Proxy) — 主力吸筹
# ──────────────────────────────────────────────────────────────────────────────
# 文献基础: 股东户数效应 (源达, 2026) + Bali et al. (2011) MAX avoidance
# 逻辑: 低波动 + 低换手 (成交量萎缩) → 主力锁仓信号
#       避开高换手 (散户炒作), 低波动 (机构持有) → 机构化微盘
#       vol20d < 50% + turnover < 0.8 → 缩量横盘的微盘 → 后续拉升
# 预期: 夏普 > 0.5, 选股纯度高

def accumulation_filter(snapshot, dates, step):
    """微盘 + 非ST + 低波动 + 低换手 + 正PB"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=25)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    if 'vol20d' in sub.columns:
        sub = sub[sub['vol20d'] < 0.50]    # 年化波动 < 50% (稳定)
    if 'turnover' in sub.columns:
        sub = sub[sub['turnover'] < 0.80]  # 低换手 (< 0.8x 均量, 意味着缩量)
    if 'pb' in sub.columns:
        sub = sub[sub['pb'] > 0]           # 正PB
    return list(sub.index)

strategy_contrarian = {
    "name": "策略C: 低波+低换手(吸筹)",
    "universe_filter": accumulation_filter,
    "ranking_factor": "mcap",            # 最小市值 (低估值小盘)
    "ascending": True,
    "n_stocks": 25,
    "stop_loss": None,
}

# ──────────────────────────────────────────────────────────────────────────────
# 策略 D: 5因子复合 (Multi-Factor Z-Score) — 学术级截面选股
# ──────────────────────────────────────────────────────────────────────────────
# 文献基础: Fama-French 5-factor + Liu-Stambaugh-Yuan (2019) + 8大alpha综述
# 逻辑: z-score 等权复合 5 个因子, 因子方向从文献/实证确定
#       小市值(asc) + 低PB(asc) + 低波动(asc) + 低MAX(asc) + 正动量(desc)
#       = 选出"小而稳、不赌博、有动力的"微盘股
# 预期: 夏普 > 0.8, 回撤 < 40%

def five_factor_filter(snapshot, dates, step):
    """微盘 + 非ST + PB>0 + 有流动性"""
    stocks = filter_micro_cap(snapshot, dates, step, max_mcap=30)
    stocks = list(set(stocks) & set(filter_no_st(snapshot, dates, step)))
    if len(stocks) == 0:
        return []
    sub = snapshot.loc[stocks]
    if 'pb' in sub.columns:
        sub = sub[sub['pb'] > 0]
    if 'turnover' in sub.columns:
        sub = sub[sub['turnover'] > 0.3]
    return list(sub.index)

strategy_five_factor = {
    "name": "策略D: 5因子复合(Multi-Factor)",
    "universe_filter": five_factor_filter,
    "n_stocks": 30,
    "stop_loss": None,
    "composite_factors": [
        ("mcap", True),     # 小市值溢价 (size)
        ("pb", True),       # 低估值安全边际 (value)
        ("vol20d", True),   # 低波动异象 (low vol)
        ("max_ret", True),  # 避开彩票型股票 (MAX avoidance)
        ("mom60d", False),  # 正动量趋势 (momentum)
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# PART III: 汇总
# ══════════════════════════════════════════════════════════════════════════════

NEW_STRATEGIES = [
    strategy_reversal_lowvol,
    strategy_size_lowvol_mom,
    strategy_contrarian,
    strategy_five_factor,
]

ALL_STRATEGIES_EXTENDED = None  # 将在 run_all 中动态拼接


if __name__ == "__main__":
    print(LITERATURE_REVIEW)
    print("\n新的 3 个策略已定义:")
    for s in NEW_STRATEGIES:
        print(f"  {s['name']}: {s['n_stocks']}只, 排名因子={s['ranking_factor']}, "
              f"{'升序' if s['ascending'] else '降序'}")
