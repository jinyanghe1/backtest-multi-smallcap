#!/usr/bin/env python3
"""
P0 Roadmap v2 — Data-First Dependency Chain
============================================
版本: 2026-06-26-v2
原则: 先解决数据，再做代码算法修复

Phase 0: Data Collection (数据先行)
────────────────────────────────────

D0.1 [BLOCKER] 采集全量 A 股上市日期表
    目的: 解决幸存者偏差 — 知道每只股票什么时候上市、是否退市
    输入: akshare.stock_info_a_code_name()
    输出: data_cache/listing_dates.csv (symbol, name, list_date, market)
    依赖: 无
    阻塞: D0.2, D0.3, P1.1
    
D0.2 [BLOCKER] 从现有日线数据提取成交额/ADV
    目的: 为 ADV 冲击模型提供真实输入
    输入: data_cache/*.parquet (price data)
    输出: data_cache/adv_panel.parquet (MultiIndex: date, symbol -> adv_20d)
    依赖: 无 (可以并行)
    阻塞: P1.2
    
D0.3 [BLOCKER] 退市数据增量更新
    目的: 确保退市表包含最新数据
    输入: akshare.stock_info_sh_delist() + stock_info_sz_delist()
    输出: data_cache/delisted_stocks.csv (追加最新)
    依赖: 无
    阻塞: P1.1

Phase 1: Engine Rebuild (引擎重建)
────────────────────────────────────

P1.1 [CORE] 重写 P0EngineV2.run() — 逐日内嵌循环
    目的: 把"后处理"改成"真实交易路径"
    输入: D0.1 (上市日期), D0.3 (退市数据)
    输出: p0_engine_v2.py (新文件)
    关键变更:
        - 逐日循环 (而非逐月聚合)
        - 每日调用 RiskOverlay.compute_gross_exposure()
        - 调仓日逐只计算 ADV 冲击
        - PIT universe 在每日选股前过滤
    依赖: D0.1, D0.3
    阻塞: P1.2
    
P1.2 [CORE] 集成 ADV 冲击到逐日循环
    目的: 真实冲击成本 (不是事后估算)
    输入: D0.2 (ADV panel), P1.1 (逐日循环)
    输出: 修改 p0_engine_v2.py
    关键逻辑:
        order_value = equity * weight
        impact = adv_model.compute_impact(order_value, adv_today)
        portfolio_return -= impact
    依赖: D0.2, P1.1
    阻塞: P1.3
    
P1.3 [CORE] 集成 RiskOverlay 到逐日循环
    目的: 风险护栏在真实路径中生效
    输入: P1.1 (逐日循环), market_data (从 return_panel 计算)
    输出: 修改 p0_engine_v2.py
    关键逻辑:
        gross = overlay.compute_gross_exposure(equity_curve_so_far, market_data)
        portfolio_return *= gross
    依赖: P1.1
    阻塞: P2.1

Phase 2: Validation & Integration (验证集成)
────────────────────────────────────

P2.1 [TEST] 编写 P0EngineV2 单元测试
    目的: 验证新引擎行为正确
    测试覆盖:
        - PIT universe: 退市股票不会被选中
        - ADV 冲击: 微盘冲击 > 大盘冲击
        - Risk Overlay: 回撤时 gross < 1.0
        - 与原引擎对比: 关闭 P0 时结果一致
    依赖: P1.3
    阻塞: P2.2
    
P2.2 [INTEGRATION] 修改 diagnose_p0.py 使用 P0EngineV2
    目的: 重新诊断，得到真实数字
    输入: P1.3 (新引擎), P2.1 (测试通过)
    输出: diagnose_p0_v2_report.csv
    依赖: P1.3, P2.1
    阻塞: P2.3
    
P2.3 [ANALYSIS] 对比 v1 vs v2 诊断结果
    目的: 量化"后处理" vs "真实路径"的差异
    输出: 对比报告 (Markdown)
    关键问题:
        - 后处理引擎高估了多少?
        - 哪些策略被真实路径淘汰?
        - 回撤是否被真实护栏压下来?
    依赖: P2.2
    阻塞: P3.1

Phase 3: Cleanup & Documentation (清理文档)
────────────────────────────────────

P3.1 [CLEANUP] 删除/归档旧引擎
    - p0_enhanced_engine.py → deprecated/p0_enhanced_engine_v1.py
    - 更新所有 import
    依赖: P2.3
    
P3.2 [DOC] 更新 roadmap.json
    - 标记已完成项
    - 记录 v1→v2 的迁移说明
    依赖: P3.1

Dependency Graph (DAG)
=======================

D0.1 ──┐
D0.3 ──┼──> P1.1 ──> P1.2 ──> P1.3 ──> P2.1 ──> P2.2 ──> P2.3 ──> P3.1 ──> P3.2
       │              ↑
D0.2 ──┘              │
                      └── P1.2 需要 D0.2

Critical Path: D0.1 → P1.1 → P1.3 → P2.1 → P2.3 → P3.2
""", "file_path": "/Users/hejinyang/thinking_and_learning_with_AI/tools/backtest_mvp/ROADMAP_P0_v2.md"} 哪个文件路径？