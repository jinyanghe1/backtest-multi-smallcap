# P0EngineV2 单元测试计划
# test_p0_engine_v2.py — 测试覆盖规划
# Version: 2026-06-26

"""
## 测试目标
验证 P0EngineV2 的逐日内嵌循环逻辑正确，且 P0 修复在真实交易路径中生效。

## 测试原则
1. 所有数据用合成数据 (monkeypatch/mock)，禁止真实网络请求
2. 每个 P0 功能单独测试，确保隔离性
3. 与原引擎对比测试，关闭 P0 时结果应一致
4. 边界条件测试：空数据、单只股票、极端行情

## 接口覆盖

### 1. 初始化 (__init__)
- 默认参数：所有 P0 功能默认开启
- 数据加载：listing_dates 从文件/参数加载
- 日期对齐：factor_panel 和 return_panel 取交集
- 调仓日生成：月度/周度/日度

### 2. 核心运行 (run)
- 逐日循环：权益每日更新
- 调仓逻辑：只在调仓日选股
- 止损逻辑：固定止损 + 移动止损
- 月度收益计算：从权益曲线 resample

### 3. PIT Universe (_pit_filter, _is_alive)
- 未上市股票过滤：date < list_date → 排除
- 已退市股票过滤：date >= delist_date → 排除
- 活着的股票：保留
- 数据缺失：默认活着 (保守处理)

### 4. ADV 冲击 (_compute_adv_drag)
- 有 ADV 数据：计算冲击成本
- 无 ADV 数据：返回 0
- 冲击成本 > 0：验证权益被扣除
- 微盘 vs 大盘：微盘冲击 > 大盘冲击

### 5. Risk Overlay (_compute_gross_exposure)
- 正常行情：gross = 1.0
- 回撤触线：gross < 1.0
- 严重回撤：gross = 0.2 (默认配置)
- 权益曲线为空：gross = 1.0

### 6. Deflated Sharpe
- 计算正确：使用 n_trials 和 n_periods
- 显著性判断：DSR > 0 时置信度 > 0

## 测试数据策略

### 合成数据 (全部使用)
- factor_panel: 3 只股票 × 60 个交易日，含 mcap, pb, mom20d
- return_panel: 同上，含 daily_return
- listing_dates: 2 只上市，1 只未上市
- delist_manager: mock，1 只已退市
- adv_data: 2 只有数据 (ADV 高/低)，1 只无数据

### 为什么不用真实数据？
- 真实数据不可控，无法预测预期结果
- 合成数据可以精确构造边界条件
- 测试运行速度快，不依赖网络

## 测试用例清单

| 用例 | 测试功能 | 输入 | 预期结果 |
|------|---------|------|---------|
| TC01 | 初始化默认参数 | 无 | 所有 P0 功能开启 |
| TC02 | 日期对齐 | 不同日期范围 | 取交集 |
| TC03 | 关闭 P0 时与原引擎一致 | enable_*=False | 结果 ≈ 原引擎 |
| TC04 | PIT 过滤未上市 | 1 只未上市 | 未上市被排除 |
| TC05 | PIT 过滤已退市 | 1 只已退市 | 已退市被排除 |
| TC06 | PIT 无数据时默认活着 | 无 listing_dates | 全部保留 |
| TC07 | ADV 有数据时冲击 > 0 | ADV=100万 | 冲击成本 > 0 |
| TC08 | ADV 无数据时冲击 = 0 | ADV=0 | 冲击成本 = 0 |
| TC09 | ADV 微盘冲击 > 大盘 | ADV低 vs ADV高 | 低ADV冲击 > 高ADV |
| TC10 | Risk Overlay 正常时 gross=1 | 无回撤 | gross=1.0 |
| TC11 | Risk Overlay 回撤时 gross<1 | 回撤-20% | gross<1.0 |
| TC12 | Risk Overlay 严重回撤 | 回撤-30% | gross=0.2 |
| TC13 | 组合止损触发 | 权益跌破 80% | stop_triggered=True |
| TC14 | 移动止损触发 | 峰值回撤 25% | stop_triggered=True |
| TC15 | Deflated Sharpe 计算 | n_trials=10 | DSR < Sharpe |
| TC16 | 空数据 | 空面板 | 不崩溃，返回默认值 |
| TC17 | 单只股票 | 1 只 | 正常计算 |
| TC18 | 极端行情 | 连续跌停 | 止损触发或权益归零 |

## 预期结果总结
- 全部 18 个用例通过
- 关闭 P0 时与原引擎结果差异 < 1% (数值精度)
- 开启 P0 时：
  - 未上市/已退市股票永远不会被选中
  - 冲击成本降低收益 (ADV 越低，降低越多)
  - 回撤时 Risk Overlay 降低 gross，减少后续损失
  - Deflated Sharpe ≤ 原始 Sharpe

## 文件路径
- 测试文件: test_p0_engine_v2.py
- 被测文件: p0_engine_v2.py
- 数据目录: 使用内存中的合成数据，不依赖文件系统
""", "file_path": "/Users/hejinyang/thinking_and_learning_with_AI/tools/backtest_mvp/test_p0_engine_v2_plan.md"} 
