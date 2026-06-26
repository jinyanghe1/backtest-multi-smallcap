# Backtest MVP 数据字段规范 (Data Schema v2)
# 定义时间: 2026-06-26
# 用途: P1 因子挖掘数据依赖

# ────────────────────────────────────────────
# 已有数据 (P0 阶段已采集)
# ────────────────────────────────────────────

## 1. 日频价格数据 (Price Data)
# 文件: data_cache/{symbol}.parquet (每只股票一个文件)
# 来源: akshare.stock_zh_a_hist
# 更新频率: 日频
# Schema:
#   date       datetime64[ns]  交易日期 (YYYY-MM-DD)
#   open       float64         开盘价
#   high       float64         最高价
#   low        float64         最低价
#   close      float64         收盘价
#   volume     float64         成交量 (股)
# Primary Key: (symbol, date)

## 2. 上市日期表 (Listing Dates)
# 文件: data_cache/listing_dates.csv
# 来源: akshare.stock_info_sh_name_code + stock_info_sz_name_code + stock_info_bj_name_code
# 更新频率: 低频 (仅新股)
# Schema:
#   symbol     str    股票代码 (e.g., sh600000, sz000001, bj920000)
#   name       str    股票名称
#   list_date  str    上市日期 (YYYY-MM-DD)
#   market     str    市场 (sh|sz|bj)
# Primary Key: symbol

## 3. 退市数据 (Delisted Stocks)
# 文件: data_cache/delisted_stocks.csv
# 来源: akshare.stock_info_sh_delist + stock_info_sz_delist
# 更新频率: 低频
# Schema:
#   symbol       str    股票代码
#   name         str    股票名称
#   list_date    str    上市日期
#   delist_date  str    退市日期 (YYYY-MM-DD)
#   market       str    市场
# Primary Key: symbol

## 4. ADV 数据 (Average Daily Volume)
# 文件: data_cache/adv_panel.parquet
# 来源: 从日频价格数据提取 (rolling 20-day mean of amount)
# 更新频率: 日频
# Schema:
#   Index: (date, symbol) 多层级索引
#   amount       float64  日成交额 (元)
#   adv_20       float64  20日ADV (元)
#   adv_20_flag  bool     是否有20日历史
# Primary Key: (date, symbol)

# ────────────────────────────────────────────
# 新数据 (P1 阶段需要采集)
# ────────────────────────────────────────────

## 5. 行业分类 (Industry Classification) ⭐ BLOCKER for UC1
# 文件: data_cache/industry_classification.csv
# 来源: akshare.stock_industry_clf_em 或 stock_board_industry_name_em
# 更新频率: 季度 (行业分类变化慢)
# Schema:
#   symbol       str    股票代码 (e.g., sh600000)
#   name         str    股票名称
#   industry_1   str    一级行业 (申万/东财一级, e.g., "银行")
#   industry_2   str    二级行业 (e.g., "股份制银行")
#   industry_3   str    三级行业 (e.g., "银行Ⅲ")
#   industry_code str   行业代码 (e.g., "bk0475")
#   update_date  str    数据更新日期 (YYYY-MM-DD)
# Primary Key: symbol
# 注意: 行业分类可能随时间变化，需要 as-of 日期查询。但简化版用最新分类即可。

## 6. 股东户数 (Shareholder Count) ⭐ for A3 集中度
# 文件: data_cache/shareholder_count.csv
# 来源: akshare.stock_zh_a_gdhs
# 更新频率: 季度 (上市公司披露频率)
# Schema:
#   symbol       str    股票代码
#   end_date     str    报告期 (YYYY-MM-DD, e.g., "2024-03-31")
#   shareholder_count  int64  股东户数
#   change_rate  float64  较上期变动率 (%)
#   avg_shares_per_household  float64  户均持股数
# Primary Key: (symbol, end_date)
# 因子用途: 
#   - 股东户数下降 → 筹码集中 → 可能上涨 (A3)
#   - 变动率: (本期户数 - 上期户数) / 上期户数

## 7. 业绩预告 (Earnings Forecast) ⭐ for A5 PEAD
# 文件: data_cache/earnings_forecast.csv
# 来源: akshare.stock_yjyg_em
# 更新频率: 季度 (业绩预告窗口期)
# Schema:
#   symbol       str    股票代码
#   report_date  str    报告期 (YYYY-MM-DD, e.g., "2024-03-31")
#   forecast_type  str    预告类型 (预增|预减|扭亏|首亏|续盈|续亏|略增|略减|不确定)
#   forecast_summary  str  预告摘要
#   forecast_lower  float64  净利润下限 (万元)
#   forecast_upper  float64  净利润上限 (万元)
#   yoy_lower    float64  同比增长下限 (%)
#   yoy_upper    float64  同比增长上限 (%)
#   announce_date  str    公告日期 (YYYY-MM-DD) ← 关键！用于事件日定义
# Primary Key: (symbol, report_date)
# 因子用途:
#   - 预告类型映射为数值 (预增=3, 续盈=2, 略增=1, 不确定=0, 略减=-1, 预减=-2, 首亏=-3, 续亏=-4, 扭亏=2)
#   - 公告日前后收益率 → PEAD (盈余公告后价格漂移)
#   - 惊喜度 = (预告中值 - 一致预期) / 一致预期标准差

## 8. 财务现金流数据 (Financial Cashflow) ⭐ for A8 质量价值
# 文件: data_cache/financial_cashflow.csv
# 来源: akshare.stock_financial_report_sina (现金流量表)
# 更新频率: 季度
# Schema:
#   symbol       str    股票代码
#   report_date  str    报告期 (YYYY-MM-DD)
#   operating_cashflow  float64  经营活动现金流量净额 (元)
#   investing_cashflow  float64  投资活动现金流量净额 (元)
#   financing_cashflow  float64  筹资活动现金流量净额 (元)
#   net_cashflow  float64  现金及等价物净增加额 (元)
#   operating_cashflow_ttm  float64  经营活动现金流TTM (元)
#   free_cashflow  float64  自由现金流 (operating_cashflow_ttm - capex_ttm)
# Primary Key: (symbol, report_date)
# 因子用途:
#   - 经营现金流/净利润 (质量指标)
#   - 自由现金流/市值 (价值指标)
#   - 现金流趋势 (quality momentum)

## 9. 财务指标 (Financial Indicators) - 补充
# 文件: data_cache/financial_indicators.csv
# 来源: akshare.stock_financial_analysis_indicator
# 更新频率: 季度
# Schema:
#   symbol       str    股票代码
#   report_date  str    报告期
#   roe          float64  净资产收益率 (%)
#   roa          float64  总资产收益率 (%)
#   net_profit_margin  float64  销售净利率 (%)
#   gross_margin  float64  毛利率 (%)
#   debt_ratio   float64  资产负债率 (%)
#   current_ratio  float64  流动比率
#   quick_ratio  float64  速动比率
#   eps          float64  每股收益
#   bps          float64  每股净资产
# Primary Key: (symbol, report_date)
# 因子用途: 质量价值 (A8)、多因子综合

# ────────────────────────────────────────────
# 数据质量检查清单
# ────────────────────────────────────────────
# 每次采集后必须验证:
#   [ ] 字段名完全匹配 schema 定义
#   [ ] 数据类型正确 (date 为 str YYYY-MM-DD, numeric 为 float64/int64)
#   [ ] 无缺失主键 (symbol, date/report_date)
#   [ ] 重复主键检查 (groupby + count)
#   [ ] 异常值检查 (负值、极大值、零值比例)
#   [ ] 时间范围覆盖 (最小/最大日期)
#   [ ] 与 price 数据 symbol 交集 (覆盖率)
