# DATA.md — backtest-multi-smallcap 数据集定义

> 最后更新: 2026-06-17 | 覆盖: 699 只 A 股微盘股 + 4 个基准指数 + 82 只港股

---

## 一、数据集总览

| 数据集 | 文件数 | 磁盘 | 格式 | 来源 | 时间范围 |
|--------|--------|------|------|------|---------|
| A 股日线 | 699 | ~25 MB | Parquet | westock-data (东方财富) | 2020-05 ~ 2026-06 |
| 财务季报 | 699 | ~6.9 MB | Parquet | akshare EM + THS | 2014 ~ 2026 Q1 |
| 总股本 | 695 | ~2.7 MB | Parquet | akshare cninfo | 静态 (最新) |
| 逐日 mcap/pb/pe | 695 | ~25 MB | Parquet | 衍生: 日线×股本/季报 | 2020-05 ~ 2026-06 |
| 基准指数 | 4 | ~0.3 MB | Parquet | westock-data | 2018-03 ~ 2026-06 |
| 港股日线 | 82 | ~3.9 MB | Parquet | westock-data | 2020-05 ~ 2026-06 |

> **打包文件**: `data_financials.zip` (3.5 MB) — 包含 `financials_cache/` + `profiles_cache/`。
> 日线数据通过 `pipeline.sh` 可再生，未纳入 zip。

---

## 二、数据集详细 Schema

### 2.1 A 股日线 (`data_cache/{symbol}.parquet`)

| 字段 | 类型 | 含义 | 备注 |
|------|------|------|------|
| `date` | datetime64[ns] | 交易日 | 按日，无缺口 |
| `open` | float64 | 开盘价 | 前复权 (qfq) |
| `high` | float64 | 最高价 | 前复权 |
| `low` | float64 | 最低价 | 前复权 |
| `close` | float64 | 收盘价 | 前复权 |
| `volume` | float64 | 成交量 (股) | |

- **来源**: `westock-data kline {symbol} --period day --limit 2000 --fq qfq`
- **符号规则**: `sh` = 沪市, `sz` = 深市, `bj` = 北交所 (如 `sh600519`, `sz000001`)

### 2.2 财务季报 (`financials_cache/{symbol}.parquet`)

| 字段 | 类型 | 含义 | 单位 | 备注 |
|------|------|------|------|------|
| `report_date` | datetime64[ns] | 报告期截止日 | — | 如 2025-12-31 = 2025 年报 |
| `notice_date` | datetime64[ns] | 公告日 | — | EM 精确; THS = report_date + 120d |
| `bps` | float64 | 每股净资产 | 元/股 | Book Value Per Share |
| `eps` | float64 | 基本每股收益 | 元/股 | Earnings Per Share |
| `revenue` | float64 | 营业总收入 | 元 | 已归一化 (THS 万/亿→元) |
| `net_profit` | float64 | 归属净利润 | 元 | 已归一化 |
| `total_equity` | float64 | 股东权益合计 | 元 | 部分缺失 |
| `symbol` | str | 股票代码 | — | 如 `sh600519` |
| `source` | str | 数据源 | — | `EM` (东财) / `THS` (同花顺) |

- **前视偏差防护**: `notice_date` 用于因子对齐 — 公告日前只能用上一季度的数据
- **单位归一化**: `parse_number()` 函数统一处理万/亿→元
- **多源策略**: EM 主源 (125 只) → THS fallback (66 只 bj), 剩余 503 只第一批下载已全

### 2.3 总股本 (`profiles_cache/{symbol}.parquet`)

| 字段 | 类型 | 含义 | 单位 |
|------|------|------|------|
| `total_shares_yi` | float64 | 总股本 | 亿股 |

- **来源**: `akshare.stock_profile_cninfo(symbol)` → `注册资金` 列
- **转换**: `注册资金(万元) / 10000 = 亿股`
- **缺失 4 只**: bj920242/275/553/855 (cninfo 不支持北交所最新上市)

### 2.4 逐日 mcap/pb/pe (`daily_mcap_pb_cache/{symbol}.parquet`)

| 字段 | 类型 | 含义 | 公式 | 备注 |
|------|------|------|------|------|
| `date` | datetime64[ns] | 交易日 | — | |
| `close` | float64 | 收盘价 | — | 用于交叉验证 |
| `mcap` | float64 | 总市值 (亿元) | `close × total_shares_yi` | |
| `pb` | float64 | 市净率 | `close / bps_fwd` | 公告日对齐，避前视偏差 |
| `pe` | float64 | 市盈率 | `close / eps_fwd` | 同上 |

- **生成逻辑** (`fetch_missing_financials.py: rebuild_mcap_pb`):
  ```
  for each day t:
      mcap_t = close_t × total_shares_yi
      bps_used = bps[notice_date ≤ t 的最新一期]  # 前向填充
      pb_t = close_t / bps_used
  ```

### 2.5 基准指数 (`benchmarks/{name}.parquet`)

与 2.1 日线相同 Schema (`date/open/high/low/close/volume`)。

| 文件 | 指数 | 代码 |
|------|------|------|
| `中证1000.parquet` | 中证1000 | 000852 |
| `国证2000.parquet` | 国证2000 | 399303 |
| `中证500.parquet` | 中证500 | 000905 |
| `中小综指.parquet` | 中小板综指 | 399101 |

### 2.6 港股日线 (`data_cache_hk/{symbol}.parquet`)

与 2.1 日线相同 Schema。符号规则: `hk` 前缀 (如 `hk00700` = 腾讯)。

---

## 三、如何再生数据

```bash
# 全量采集 (4 阶段)
bash pipeline.sh --max 700

# 仅补缺失
python fetch_missing_financials.py

# 解压财务数据 (从 data_financials.zip)
unzip data_financials.zip

# 日线增量更新 (每天 2 次)
python update_daily.py --max 50
```

---

## 四、因子面板 (运行时生成)

通过 `factors.py: compute_factors()` 从上述数据集构建，共 12 列：

| # | 因子 | 列名 | 来源 | 文献 |
|---|------|------|------|------|
| 1 | 总市值 | `mcap` | daily_mcap_pb | 规模溢价 |
| 2 | 市净率 | `pb` | daily_mcap_pb | 价值因子 |
| 3 | 市盈率 | `pe` | daily_mcap_pb | 价值因子 |
| 4 | 20日动量 | `mom20d` | 日线 close | 短期反转 |
| 5 | 60日动量 | `mom60d` | 日线 close | 中期趋势 |
| 6 | 相对量比 | `turnover` | 日线 volume | 流动性代理 |
| 7 | 20日波动率 | `vol20d` | 日线 close | 低波动异象 |
| 8 | 特质波动率 | `ivol` | 日线 (market residual) | Ang 2006 |
| 9 | MAX 收益 | `max_ret` | 日线 close | Bali 2011 |
| 10 | 涨停标记 | `is_limit_up` | 日线 close/high | 交易约束 |
| 11 | 跌停标记 | `is_limit_down` | 日线 close/low | 预留 |
| 12 | 股票名称 | `name` | name_lookup.parquet | ST 过滤 |
