# P17 Tushare/PIT 数据底座预注册契约（四层架构 + D4/D5/D6 语义基座）

- 状态：**DRAFT — 提交契约门禁评审（t2，含用户批准门）。批准前禁止 D2/D3
  实施动代码与回填（AGENTS §2.2 门禁顺序不可倒置）；批准后本文保持前瞻性，
  禁止用实现或测量输出反向修改；测量结果只能执行预注册裁决。**
- 起草：governance-auditor（AgentTeams tushare-pit-migration t1，G0 工作包），
  2026-09-03。
- 证据基线（勘察基线惯例 [02-INT-09]）：**main @ `8ff21f6`**。§1 的全部现状
  引用（文件/行号/常量/测量日志）均在该 commit 实测；Tushare 接口权限与官方
  更新时刻引自 tushare.pro 官方文档（[关于权限 doc_id=108](https://tushare.pro/document/1?doc_id=108)），并以 D1 探针
  （`scripts/probe_tushare.py`，t3）的真实留档输出为字段映射核对权威（§5.0、
  附录 A）。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。任务路由：**数据/股票池变更**
  （AGENTS §5，G1–G8 + research_doctor + 只读审计）+ **研究语义变更**
  （§3.2：D4/D5/D6 语义切换的版本 bump 与迁移/拒绝策略由本文统一预注册基座，
  实施任务 t12/t14/t15 各自携带 RED 清单）。仲裁顺序：本文/批准 → 契约测试
  → 实现 → 测量日志（AGENTS §1）。
- 配套审计：`docs/g0_approval_evidence_audit.md`（P12/P13/P14 批准证据审计，
  同任务交付；本文不重复其内容）。

## 1. 问题陈述（全部为既有证据，非本文新测量）

### 1.1 现行数据底座的四个结构性 PIT 缺口（main @ 8ff21f6 现场引用）

1. **价格 = 前复权（qfq），历史随除权被反复改写**：`ashare_data/config.py:41`
   `adjust: str = "qfq"`——AkShare 拉取即前复权；Baostock 兜底路径同为 qfq
   （`ashare_data/akshare_client.py:553,580`，`adjustflag="2"`）。前复权价的
   时间序列**不是 append-only 事实**：每一次分红送转都会改写全部历史价格。
   这直接违反 AGENTS §4.3"append-only ledger、不得覆盖原始证据"对数据层
   的同构要求，且使"复现某日当时所见价格"不可能。
2. **申万行业 = 当前快照，非 PIT**：`sw_member` 表（`ashare_data/db.py:165-173`）
   仅 `(index_code, ts_code)`，无 in/out 日期；P14 契约 §1.4 已实证其后果——
   IND_REL_* 族为 Tier C（"申万当前快照非 PIT"），P9 实测最强的因子族在晋级
   路径不可用。
3. **财务 = 法定披露季末锚定的近似，无修订追踪**：`fundamental_pit`
   （`ashare_data/db.py:109-143`）以 announce master 法定季末日锚定可见性、
   COALESCE upsert 覆盖合并（db.py L396-412 语义），同一 (ts_code, report_date)
   只保留一行——**调整前/调整后版本并存的审计能力不存在**（修订即覆盖）。
4. **执行约束近似 + 来源单点脆弱**：涨跌停/停牌走近似交易阻断（无精确
   limit/suspension 表）；akshare 1.18.91 缺 `stock_zcfzb_em` 函数曾导致 P13
   回填 total_assets 0 行（`docs/p13_fundamental_backfill_measurement_log.md`
   §3.3）——单一免费来源的属性面漂移是已发生的事实风险，不是假设。

### 1.2 目标：Tushare Pro（2000 积分）四层 PIT 数据底座

按用户预注册路线图，把数据底座迁移到 Tushare Pro（用户 2000 积分账户，
权限覆盖本文全部 20 个核心接口，见附录 A），建立四层架构：

```text
L0 原始快照层   data/raw/tushare/<endpoint>/<ingest_id>/part-*.parquet
                （append-only，批次级完整元数据 + 内容 SHA-256）
L1 来源表层     data/ashare_v2.duckdb：source_ingest_runs 台账 + 19 张新表
                （append-only / 版本保留，单位按 Tushare 原生口径）
L2 PIT 视图层   可用时间映射 + as-of 读取规则（§6，单一实现）
L3 消费层       AshareDataLoader / 因子注册表（既有唯一正式路径，不复制第二套）
```

### 1.3 铁律（非目标式红线，§15 逐条可执行）

- 正式路径**禁止**在 Tushare 失败后静默混入 AkShare/BaoStock 数据（AkShare/
  BaoStock 只可作审计对比且必须显式标注；Tushare 失败 → 数据标 stale + 正式
  运行拒绝启动）。
- `TUSHARE_TOKEN` 为机密：只允许存在于进程环境或 git-ignored 的 `config/.env`
  （`.gitignore` 的无斜杠模式 `.env` 覆盖任意层级同名文件，已核对
  main @ 8ff21f6 的 `.gitignore` L12）；禁止进入本计划、代码、日志、异常、
  测试快照、commit 或任何输出。
- 生产库 `data/ashare.duckdb` 全程零写入；迁移在**新建** `data/ashare_v2.duckdb`
  进行（§8、§10）。

## 2. 假设

1. **H1 权限假设**：用户 2000 积分可调用附录 A 全部 20 个核心接口；D1 探针
   实测确认。`stock_st`/`share_float`（官方 3000 起）/`bak_basic` 预期无权限，
   明确报告且不影响主计划（t3 预注册一致）。
2. **H2 更新时刻假设**：官方更新时刻表（附录 A）稳定；本文按"官方窗口上界"
   保守取 available_ts（§6），官方提前量变化不破坏保守性。
3. **H3 语义假设**：(a) `daily`（未复权）+ `adj_factor`（后复权因子，历史值
   稳定）可重建复权连续序列且除权日无跳变；(b) 三张财报的版本行
   （update_flag/ann_date/f_ann_date 并存）+ §6.2 as-of 规则可复现任意交易日的
   "当时可见"报表；(c) `index_member_all` 的 in/out 区间可表达行业成员 PIT。
   任一被探针/回填证伪 → §15 停止条件。
4. **H4 隔离假设**：新库与生产库文件级隔离（ashare_v2.duckdb ≠ ashare.duckdb），
   legacy 只读共存；D4–D6 语义切换以显式配置/版本拒绝承载，无静默切换。
5. **H5 可用时间假设**：申万成员区间的 in_date 近似"可知日"存在收录滞后
   （Tushare 无成员变更公告日字段）——这正是行业成员停在 Tier B 而非 A 的
   原因（§7），由 D5 验证门量化该滞后（官方公告对账）。

## 3. 范围与非目标

### 3.1 范围（本文预注册的语义基座）

1. L0 快照层目录/元数据规范（§4）、L1 全部 19 张新表 DDL 与写入语义（§5）、
   L2 可用时间与 as-of 规则（§6）、Tier 映射（§7）、legacy 只读纪律（§8）、
   版本影响表（§9）、迁移/回填路径（§10-§11）、覆盖率门（§12，供 D7 按需
   消费）、拉取策略与限速（§13）、申万专项约束（§14）、红线与停止条件（§15）、
   验证矩阵（§16）。
2. D4/D5/D6 语义切换的**数据侧语义**在此固定；三者的实现任务（t12/t14/t15）
   各自携带 RED 清单、版本 bump 兑现与迁移/拒绝测试，本文是其预注册上游。

### 3.2 非目标（显式声明，评审按此驳回越界改动）

1. 不改现有生产库 `ashare.duckdb` 的任何 schema、行、路径；qfq daily_bar、
   `fundamental_pit`、`sw_industry_member` 原样保留为只读 legacy（§8）。
2. 不实现因子/搜索/组合语义：因子公式、标签、Reward、晋级阈值零改动（D4–D6
   实施任务与各自契约辖区）；本文不含任何研究测量，不产生 alpha/晋级结论。
3. 不新增第二套日历/universe/mask：trade_cal 仅做 parity 对比记录（t9 步骤 2），
   交易日历唯一权威仍是现有 `calendar_table`/universe contract；PIT eligibility
   唯一路径仍是 AshareDataLoader（AGENTS §5.1）。
4. 不做 UI、不切生产默认搜索器、不做 lifecycle 状态转换（E 线任务辖区）。
5. 不授权 push/远程 PR；全程仅本地（§17）。

## 4. 四层架构与批次身份（L0/L1 规范）

1. **ingest_id**：格式 `<endpoint>_<UTC yyyymmddTHHMMSSZ>_<8位小写hex随机>`，
   全局唯一、文件系统安全；贯穿 L0 目录名、source_ingest_runs 主键、L1 行级
   `ingest_id` 列——单批一次拉取 = 一个 ingest_id。
2. **L0 快照**：`data/raw/tushare/<endpoint>/<ingest_id>/part-*.parquet`；
   先落快照、后提交数据库（顺序由 t7 测试强制）。每批元数据齐全：ingest_id、
   endpoint、请求参数（parameters_json）、字段清单（fields_json）、起止时间、
   tushare SDK 版本（sdk_version 列）、代码 commit（code_commit 列）、原始
   行数（row_count）、内容 SHA-256（raw_payload_sha256）、成功/失败状态、
   错误类型/消息、是否触达接口行数上限（hit_row_limit 列）。**禁止**保存
   token、HTTP headers 或任何账户信息。
3. **source_row_hash**：对行做 canonical 序列化（键排序、UTF-8、值原样、
   不含 token/账户信息）后 SHA-256 十六进制；append-only 幂等与修订检测的
   唯一依据。**展开映射场景（附录 B）的哈希输入 = 展开后契约行全部列**，
   canonical 输入显式含 level 判别符；**禁止以源行（11 列）为哈希输入**——
   t47 keep-first 批内预去重会把同源行的多个展开行坍缩为一行（L2/L3 静默
   数据丢失，勘误 #2）。
4. **dataset_id**：`ashare_v2` 的 manifest 按既有 manifest 权威对内容重算
   （p13 先例：dataset_id 随数据内容重算是数据事实，manifest 版本常量不 bump）；
   后续测量绑定 v2 dataset_id，跨 dataset_id 结论禁止拼接（AGENTS §4.3）。

## 5. 19 张新表 DDL（D2 实现的列级权威）

### 5.0 列名权威与 curated 规则

- 身份/审计列（ingest_id、source_row_hash、available_date、ann_date 等）与
  语义列的**来源字段映射**以 Tushare 官方文档为准，并在 D1 探针留档输出中
  逐列核对；探针列名与本文映射不符 → **停止条件"来源字段无法确认"**（§15.6），
  停下修订本文，禁止静默改名或跳列。
- curated 语义列为**下限**：写入器必须断言全部必需列存在（缺列报错，
  fail-closed）；允许按来源原样追加 additive 列（列名保持 Tushare 原名、
  记入 fields_json），禁止把未登记列静默改名入库。
- 日期列一律 `VARCHAR YYYYMMDD`（与既有表约定一致）；时间戳列
  `TIMESTAMP`（Asia/Shanghai 墙钟，约定写入模块 docstring）；金额/股数单位
  在每表注释中固定为 Tushare 原生单位，L1 不做单位换算，任何换算只发生在
  L2/L3 且为显式单一路径。
- 写入语义两类（§5.2 逐表标注）：
  - **KEYED**（市场/成员类，业务键唯一）：`ON CONFLICT DO NOTHING`；被抑制
    行若 `source_row_hash` 与库内不同 → 整批失败（error_type='row_conflict'），
    零半批提交（先写后改 = 修订，必须走人工裁决 + 新预注册，禁止覆盖）。
    同内容重跑 = 幂等 no-op。
  - **APPEND**（版本/事件类，修订并存）：`ON CONFLICT (source_row_hash)
    DO NOTHING`；同 hash 重跑幂等，不同 hash 永远追加新版本行，禁止任何
    按业务键的 UPDATE/DELETE（红线 §15.4）。
- **源列→契约列映射纪律（勘误 t44 确立，常设）**：任何源列名与契约列名
  不一致（改名/语义错位/笔误），必须先经勘误在附录 B 落盘显式映射后方可
  实施；D2/D3 按映射实施，禁止静默改名或跳列（§15.6 的机制化）。

### 5.1 台账（第 1 张）

```sql
CREATE TABLE IF NOT EXISTS source_ingest_runs (
    ingest_id         VARCHAR PRIMARY KEY,
    provider          VARCHAR NOT NULL,           -- 'tushare'
    endpoint          VARCHAR NOT NULL,
    parameters_json   VARCHAR NOT NULL,           -- 请求参数 canonical JSON
    fields_json       VARCHAR NOT NULL,           -- 来源返回字段清单
    started_at        TIMESTAMP NOT NULL,         -- Asia/Shanghai
    finished_at       TIMESTAMP,                  -- running 时为 NULL
    status            VARCHAR NOT NULL,           -- running|succeeded|failed
    row_count         BIGINT  NOT NULL,           -- 原始行数（失败可为 0）
    raw_payload_sha256 VARCHAR NOT NULL,          -- 快照内容 SHA-256
    raw_path          VARCHAR NOT NULL,           -- L0 目录/文件相对路径
    code_commit       VARCHAR NOT NULL,           -- 写入代码 git SHA
    sdk_version       VARCHAR NOT NULL,           -- tushare SDK 版本
    hit_row_limit     BOOLEAN NOT NULL,           -- 行数==接口上限→需分页
    error_type        VARCHAR,                    -- status=failed 时非空
    error_message     VARCHAR                     -- status=failed 时非空
);
```

### 5.2 市场/成员类（KEYED；单位 = Tushare 原生）

```sql
-- 2. 日线原始行情（未复权；daily；价格=元，vol=手，amount=千元）
CREATE TABLE IF NOT EXISTS daily_bar_raw (
    ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, pre_close DOUBLE,
    change DOUBLE, pct_chg DOUBLE, vol DOUBLE, amount DOUBLE,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (ts_code, trade_date)
);
-- 3. 复权因子（adj_factor；后复权因子，历史值稳定是 H3a 前提）
CREATE TABLE IF NOT EXISTS daily_adj_factor (
    ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL,
    adj_factor DOUBLE NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (ts_code, trade_date)
);
-- 4. 每日指标（daily_basic；total_mv/circ_mv=万元，share 类=万股）
CREATE TABLE IF NOT EXISTS daily_basic_pit (
    ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL,
    close DOUBLE, turnover_rate DOUBLE, turnover_rate_f DOUBLE,
    volume_ratio DOUBLE, pe DOUBLE, pe_ttm DOUBLE, pb DOUBLE,
    ps DOUBLE, ps_ttm DOUBLE, dv_ratio DOUBLE, dv_ttm DOUBLE,
    total_share DOUBLE, float_share DOUBLE, free_share DOUBLE,
    total_mv DOUBLE, circ_mv DOUBLE,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (ts_code, trade_date)
);
-- 5. 涨跌停价格（stk_limit；元；官方更新=交易日9点，当日会话可用）
CREATE TABLE IF NOT EXISTS daily_price_limit (
    ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL,
    pre_close DOUBLE,  -- typed 显式字段请求供给（D1 实测 2026-09-04；探针默认响应未含此列）
    up_limit DOUBLE NOT NULL, down_limit DOUBLE NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (ts_code, trade_date)
);
-- 6. 停复牌（suspend_d；事件类但按业务键幂等；trade_date 当日会话可用，§6.1；
--    列集恢复用户目标文档原始定义：suspend_timing 为 §5.0 additive 源列）
CREATE TABLE IF NOT EXISTS daily_suspension (
    ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL,
    suspend_type VARCHAR NOT NULL,
    suspend_timing VARCHAR,  -- additive 源列（探针实证 2026-09-04）
    available_at TIMESTAMP NOT NULL,  -- 源可得时点：trade_date 当日（停牌决定于开盘前确定；经 §6.1 映射得当日会话）
    source VARCHAR NOT NULL,  -- 数据来源标识（与 source_ingest_runs.provider 一致：'tushare'）
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (ts_code, trade_date, suspend_type)
);
-- 7. 申万行业分类（index_classify；源列含 is_pub/parent_code，代际判定见 §14.3）
CREATE TABLE IF NOT EXISTS industry_taxonomy (
    index_code VARCHAR PRIMARY KEY,
    industry_name VARCHAR NOT NULL, level VARCHAR NOT NULL,
    industry_code VARCHAR NOT NULL, src VARCHAR,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL
);
-- 8. 申万行业成员 PIT（index_member_all；区间 [in_date, out_date)，
--    out_date 为空 = 仍在任；约束见 §14；源行三级展开映射见附录 B——
--    一行源数据展开为最多 3 行契约行（level=1/2/3），空级跳过）
CREATE TABLE IF NOT EXISTS industry_membership_pit (
    index_code VARCHAR NOT NULL, ts_code VARCHAR NOT NULL,
    level INTEGER NOT NULL,  -- 展开判别符（勘误 #2 授权的唯一派生列，非源列；入 canonical 哈希输入）
    in_date VARCHAR NOT NULL,
    out_date VARCHAR, is_new VARCHAR NOT NULL,
    industry_name VARCHAR NOT NULL, available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (index_code, ts_code, in_date)
);
-- 18. 两融明细（margin_detail；rzye/rqye/rzmre/rzche/rzrqye 等=元，
--     rqyl/rqmcl=股/份；available_date = trade_date 次一开市会话）
CREATE TABLE IF NOT EXISTS margin_detail_pit (
    ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL,
    rzye DOUBLE, rqye DOUBLE, rzmre DOUBLE, rqyl DOUBLE,
    rzche DOUBLE, rqchl DOUBLE, rqmcl DOUBLE, rzrqye DOUBLE,
    available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (ts_code, trade_date)
);
-- 19. 指数成分与权重（index_weight；月度；weight=%；保守可用时间=月末后）
CREATE TABLE IF NOT EXISTS index_weight_pit (
    index_code VARCHAR NOT NULL, con_code VARCHAR NOT NULL,
    trade_date VARCHAR NOT NULL, weight DOUBLE,
    available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (index_code, con_code, trade_date)
);
```

### 5.3 财务版本表（APPEND；三张同构，身份列严格按本节）

身份列（全部 NOT NULL，除 f_ann_date 可空）：`ts_code, end_date, ann_date,
f_ann_date, available_date, report_type, comp_type, update_flag,
source_row_hash, ingest_id`；**不过滤 update_flag**（0=原始/1=修订并存）；
缺失数值保持 NULL（消费侧 NaN），**禁止变 0**；ann_date 与 f_ann_date 皆缺
的行在入库时拒绝并记入 source_ingest_runs 错误（fail-closed）。

```sql
-- 9. 利润表版本表（income；金额=元）
CREATE TABLE IF NOT EXISTS income_statement_pit (
    ts_code VARCHAR NOT NULL, end_date VARCHAR NOT NULL,
    ann_date VARCHAR NOT NULL, f_ann_date VARCHAR,
    available_date VARCHAR NOT NULL,
    report_type VARCHAR NOT NULL, comp_type VARCHAR NOT NULL,
    update_flag VARCHAR NOT NULL, end_type VARCHAR,
    total_revenue DOUBLE, revenue DOUBLE, oper_cost DOUBLE,
    total_profit DOUBLE, n_income DOUBLE, n_income_attr_p DOUBLE,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
);
-- 10. 资产负债表版本表（balancesheet；金额=元）
CREATE TABLE IF NOT EXISTS balance_sheet_pit (
    ts_code VARCHAR NOT NULL, end_date VARCHAR NOT NULL,
    ann_date VARCHAR NOT NULL, f_ann_date VARCHAR,
    available_date VARCHAR NOT NULL,
    report_type VARCHAR NOT NULL, comp_type VARCHAR NOT NULL,
    update_flag VARCHAR NOT NULL, end_type VARCHAR,
    total_assets DOUBLE, total_liab DOUBLE,
    total_hldr_eqy_exc_min_int DOUBLE, total_hldr_eqy_inc_min_int DOUBLE,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
);
-- 11. 现金流量表版本表（cashflow；金额=元）
CREATE TABLE IF NOT EXISTS cashflow_statement_pit (
    ts_code VARCHAR NOT NULL, end_date VARCHAR NOT NULL,
    ann_date VARCHAR NOT NULL, f_ann_date VARCHAR,
    available_date VARCHAR NOT NULL,
    report_type VARCHAR NOT NULL, comp_type VARCHAR NOT NULL,
    update_flag VARCHAR NOT NULL, end_type VARCHAR,
    n_cashflow_act DOUBLE, n_cashflow_inv_act DOUBLE,
    n_cash_flows_fnc_act DOUBLE, c_cash_equ_end_period DOUBLE,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
);
```

### 5.4 审计/披露与事件表（APPEND）

```sql
-- 12. 财务指标（fina_indicator；仅审计/对账用，禁止进入因子正式路径）
CREATE TABLE IF NOT EXISTS financial_indicator_audit (
    ts_code VARCHAR NOT NULL, end_date VARCHAR NOT NULL,
    ann_date VARCHAR, f_ann_date VARCHAR, update_flag VARCHAR,
    report_type VARCHAR, eps DOUBLE, roe DOUBLE, roe_waa DOUBLE,
    grossprofit_margin DOUBLE, netprofit_margin DOUBLE,
    debt_to_assets DOUBLE,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
);
-- 13. 财报披露计划（disclosure_date；pre_date 预约/actual_date 实际，
--     用于可用时间对账审计）
CREATE TABLE IF NOT EXISTS financial_disclosure_schedule (
    ts_code VARCHAR NOT NULL, end_date VARCHAR NOT NULL,
    pre_date VARCHAR, actual_date VARCHAR,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
);
-- 14. 业绩预告（forecast；p_change_min/max=%，net_profit_min/max=元）
CREATE TABLE IF NOT EXISTS earnings_forecast_pit (
    ts_code VARCHAR NOT NULL, ann_date VARCHAR NOT NULL,
    end_date VARCHAR NOT NULL, type VARCHAR,
    p_change_min DOUBLE, p_change_max DOUBLE,
    net_profit_min DOUBLE, net_profit_max DOUBLE,
    last_parent_net DOUBLE, summary VARCHAR, change_reason VARCHAR,
    available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
);
-- 15. 业绩快报（express；金额=元）
CREATE TABLE IF NOT EXISTS earnings_express_pit (
    ts_code VARCHAR NOT NULL, ann_date VARCHAR NOT NULL,
    end_date VARCHAR NOT NULL,
    revenue DOUBLE, operate_profit DOUBLE, total_profit DOUBLE,
    n_income DOUBLE, total_assets DOUBLE,
    available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
);
-- 16. 回购（repurchase；vol=万股，amount=万元）
CREATE TABLE IF NOT EXISTS repurchase_event_pit (
    ts_code VARCHAR NOT NULL, ann_date VARCHAR NOT NULL,
    end_date VARCHAR, proc VARCHAR, exp_date VARCHAR,  -- proc=实施进度（源列名，勘误 t44；取值语义 D3 首批样本值落档核验）
    vol DOUBLE, amount DOUBLE, high_limit DOUBLE, low_limit DOUBLE,
    available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
);
-- 17. 股东人数（stk_holdernumber）
CREATE TABLE IF NOT EXISTS holder_number_pit (
    ts_code VARCHAR NOT NULL, ann_date VARCHAR NOT NULL,
    end_date VARCHAR NOT NULL, holder_num BIGINT,
    available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
);
```

## 6. 可用时间语义与 as-of 读取规则（L2，单一实现）

### 6.1 统一映射规则（预注册）

**available_session = 最早满足"开盘（09:30）严格晚于 available_ts"的开市
会话日期**。该规则对四类情形逐项退化正确：

| 源 | available_ts（预注册，保守取官方窗口上界） | available_session |
|---|---|---|
| daily / daily_basic | T 日 17:00（官方 15–17 点盘后更新） | T 的次一开市会话 |
| stk_limit | T 日 09:00（官方"交易日9点"） | **T 当日会话**（09:00 < 09:30，盘前已知；供执行约束同日消费） |
| margin_detail（T 日数据） | T+1 日 09:00（官方"每日9点更新"上一日明细；t27 口径"次日约8:30"取更晚的官方 9:00，保守） | T+1 当日会话（T+1 语义） |
| suspend_d | suspend_date 当日（停牌事实当日开盘即可观察确证——当日不成交即停牌，无未来信息；ann_date 留档审计） | suspend_date 当日会话 |
| forecast / express / repurchase / stk_holdernumber / 三张财报 | 公告基准日（三张财报 = max(f_ann_date, ann_date)）当日 23:59（公告日全天按"发布后"处理，保守） | 公告基准日的**下一开市会话**（F3"公告日后下一交易日才可进入因子"由此成立；公告日为非交易日时自然顺延到次一开市日） |
| index_member_all | in_date 当日 23:59（近似可知日，H5；收录滞后由 D5 验证门量化） | in_date 的下一开市会话 |
| index_weight | 所属月度结束（次月 1 日 00:00） | 月末后的下一开市会话（保守；探针/对账如证实更早可得，经契约勘误收紧） |

- available_date 列 = available_session（YYYYMMDD）；L1 中带 available_date 列
  的表由写入器按上表计算；daily_bar_raw/daily_basic_pit/daily_price_limit/
  daily_adj_factor 不设该列（规则可从 trade_date 确定性推导，避免两处真值）。
- 映射实现必须复用**现有唯一交易日历**判定开市会话；禁止第二套日历。

### 6.2 财务版本表 as-of 读取规则（D6 消费方的预注册语义）

对业务键 `K = (ts_code, end_date, report_type, comp_type)`、查询会话 `s`：

1. 候选 = 该键下 `available_date <= s` 的全部版本行（**不筛 update_flag**，
   调整前报表可审计：版本行不被覆盖删除）；
2. 胜者 = 按 `(f_ann_date, ann_date, available_date, ingest_id)` 字典序取最大
   ——修订公告后自动返回修订版，修订公告日前返回初始版（update_flag=0）；
   `(f_ann_date, ann_date)` 相同而内容不同的并列情形按 `(available_date,
   ingest_id)` 决胜并**必须**进入 D6 审计报告（正常应为 0 例）；
3. 三张表**各自独立 as-of**：利润/资产负债/现金流按各自 available_date 取，
   禁止用一表的公告日代表另一表、禁止提前拼接（AGENTS §1 时间因果）；
4. 查无候选或缺字段 → NaN，**禁止 0 或中性值伪造**（AGENTS §5.3）；
5. as-of 读取只允许一个实现（D6 落点：`ashare_data/pit_import.py`/视图层），
   消费方（factors.py 等）只调用不复制（AGENTS §1 原则 3）。

### 6.3 行业成员 as-of 规则（D5 消费方的预注册语义）

会话 `s` 的有效行业 = `in_date <= s < out_date`（out_date 为空 = 仍在任）；
同一股票同一层级同日**至多一个**有效行业——违反即 fail-closed 拒绝并阻断
（§14）。历史区间的 out_date 修订按 KEYED 冲突语义处理（禁止覆盖）。

## 7. 数据 Tier 映射（经单一注册表权威，不新建第二套 tier）

| L1 表 | 注册表 PitLevel | DataTier |
|---|---|---|
| daily_bar_raw / daily_adj_factor / daily_basic_pit / daily_price_limit / daily_suspension | PIT_DAILY | **A** |
| margin_detail_pit | PIT_CAPITAL | **B**（与现行 margin 同级） |
| industry_membership_pit | PIT 成员区间 | **B**（D5 验证门通过后；未验证维持 C——C→B 门槛核验记录在案：历史覆盖 / 官方公告对账 / 可用时间；升至 A 需新预注册修订） |
| income/balance/cashflow_statement_pit | PIT_FUNDAMENTAL | **B**（带修订追踪，PIT 保证强于 legacy 法定季末近似） |
| earnings_forecast_pit / earnings_express_pit | PIT_FUNDAMENTAL | **B** |
| repurchase_event_pit / holder_number_pit / index_weight_pit | PIT_CAPITAL | **B** |
| financial_indicator_audit / financial_disclosure_schedule / industry_taxonomy / source_ingest_runs | 非因子信号源（审计/元数据） | 不参与 tier 评定（禁止进入因子正式路径） |

- 特征级 tier 仍由既有 `feature_registry.PitLevel → data_tier` 单一映射派生；
  本文只固定**源级**归类，不新增映射实现（AGENTS §1 原则 3）。
- D6 实施时 `TIER_TIME_RULES["B"]` 文本需同步"修订追踪"语义并 bump
  DATA_TIER_VERSION（§9）；行业成员 B 级化在 D5 实施任务中按其验证门兑现。

## 8. legacy 只读清单与生产库纪律

1. **生产库 `data/ashare.duckdb` 零写入**：新管线对其只允许只读连接（parity
   审计、scope 读取）；所有 D2 测试在 tmp_path 临时 DuckDB 上进行（t7 已预
   注册同款纪律）。
2. **legacy 只读清单**（禁止覆盖/删除/回填，任何改动都构成红线 §15）：
   - `daily`（qfq 日线）及其 adj_factor 列；
   - `fundamental_pit`（含 P13 两列）；
   - `sw_industry_member`、`sw_index`；
   - `stocks`、`constituents`、`calendar_table`、`factor`、`margin` 等
     生产库全部既有表。
3. 新库 `data/ashare_v2.duckdb` 与生产库无外键/无视图耦合；D4–D6 切换以
   配置指向 + 版本拒绝承载（§10），无静默切换路径。

## 9. 版本影响表（`git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"` 全量
检索基线 = main @ 8ff21f6，33 命中；下表列相关项，其余以 p11 契约 §6.1
全量清单为准，不复制第二份）

| 版本所有者 | 旧值 | 新值 | bump 理由 / 承载 |
|---|---|---|---|
| **新增** ashare_data/tushare_store.py `TUSHARE_STORE_SCHEMA_VERSION` | （无） | **"1"** | L1 新 schema（19 表 + 台账）；新 artifact-facing 版本接入既有兼容性权威 = manifest（v2 manifest 记录该版本；消费方按 dataset_id + 该版本 fail-closed） |
| ashare_data/manifest.py MANIFEST_VERSION | 1 | 不变 | 新库 manifest 为新实例；dataset_id 按内容重算（p13 先例：数据事实非版本事件） |
| ashare_model/data_tier.py DATA_TIER_VERSION | 1 | **2**（D6 实施时） | TIER_TIME_RULES 基本面条目改为"修订追踪"语义 + 行业成员 B 级规则文本；映射机制零改动 |
| ashare_model/vocab.py GRAMMAR_VERSION | 6 | 不变（本文） | p17 不改词表；D5/D6 特征语义切换的词表后果由其实施任务按各自契约 bump |
| ashare_model/feature_registry.py FEATURE_REGISTRY_VERSION | 6 | 不变（本文） | 同上 |
| ashare_model/research_domain.py RESEARCH_DOMAIN_VERSION | 3 | 不变（本文） | 域划分零改动 |
| ashare_model/evaluation.py PROTOCOL_VERSION | "25" | 不变 | 评价协议零改动（D4 实施如触及，按 t10 修订契约 bump） |
| ashare_model/reward.py REWARD_VERSION | "15" | 不变 | 零改动 |
| ashare_model/promotion.py PROMOTION_RULE_VERSION | "2" | 不变 | 晋级门零改动（G7 与数据门两个编号空间不混用，AGENTS §5.4） |
| ashare_model/search_contract.py / searcher_bench.py | 4 / 4 | 不变 | 搜索层零改动 |
| ashare_portfolio/constructor.py / execution_spec.py | 1 / 2 | 不变（本文） | D4 实施的成交语义切换版本后果由 t12 依据 t10 修订契约兑现 |

迁移/拒绝策略总纲：**旧产物按 dataset_id + 版本集合拒绝**（既有 artifact
兼容性权威，fail-closed）；legacy 数据只读不迁移；新数据只进新库；无任何
历史改写、无回填旧库（AGENTS §4.3、§5.6）。

## 10. 迁移路径与回填步骤（D3 执行；每步先 L0 快照后入库）

1. 只读审计生产库：范围、行数、日期范围、dataset_id，留档（不写）。
2. `trade_cal`（与现有唯一日历 parity 对比记录，**不替换**日历权威）；
   `stock_basic`（含 L/D/P 状态与历史退市股——为 R4 全市场扩展储备元数据，
   见 §11）。
3. 按 trade_date 全市场逐日拉取（2014-01-01 起）：daily、adj_factor、
   daily_basic、stk_limit、suspend_d。
4. margin_detail（T+1 可用语义见 §6.1）。
5. 申万：index_classify 全量 + index_member_all 按 ts_code 逐只（§13、§14），
   is_new=Y 与历史记录并取，taxonomy 代际（含 2021 切换）显式记录。
6. 三张财报按股票全历史（2012-12-31 起，保留全部 ann_date/f_ann_date/
   report_type/update_flag 版本行）+ fina_indicator（仅审计）+ disclosure_date。
7. forecast/express（按公告日期增量、首次按年分段）、repurchase、
   stk_holdernumber、index_weight。
8. 构建 v2 manifest：dataset_id、日期范围、各表行数、来源快照 ingest_id
   清单、SHA-256、TUSHARE_STORE_SCHEMA_VERSION。
9. 门禁复核：G1–G8（`python scripts/check_production_gates.py`，经
   `ASHARE_DUCKDB_PATH` 指向 v2；禁止 --dev 冒充正式）+ `python -m
   ashare_model.research_doctor`；覆盖率/重叠/修订版本/日期连续性审计报告
   留档（evidence-only 测量日志，与本文分离，AGENTS §11.1）。

停止条件（任一触发 → 任务 failed + 完整报告，禁止静默降级或换源凑数）：
§15 全部红线。

## 11. 回填范围与 scope 纪律

1. **正式表入库 scope** = 生产库持久 scope（`persisted_scope_codes` =
   stocks ∪ PIT 成员 ∪ daily 缓存码；权威工具 `python scripts/
   check_fundamental_scope.py --report`，P2-01；P13 日志实测 5,542 码
   @ dataset b7b4dd4b，D3 运行时重算并留档）。scope 外数据进入正式表 =
   红线 §15.5（fail-closed）。
2. stock_basic 全市场（含退市）仅入 L0/审计面，用于状态/退市元数据与 R4
   未来扩展；正式表扩池（如 R4 全市场）**必须**先修订 p5 契约（t29 预注册
   路径）并显式扩展 scope 清单，禁止静默扩池。
3. 时间范围：市场数据 2014-01-01 起；财务数据 2012-12-31 起；研究评价起点
   保持 2015-01-01 不变（与既有 OOS/窗口协议一致，禁止顺手改窗口）。

## 12. 覆盖率门（供 D7 按公式依赖资格门预注册消费；本文固定数值）

| 源 | 门 | 不合格动作 |
|---|---|---|
| 日线（daily_bar_raw）/ 涨跌停（daily_price_limit） | 相对最近已完成会话滞后 ≤ **3** 会话 | 该源 fail；依赖该源的 RunSpec 拒绝（fail-closed） |
| 两融（margin_detail_pit） | 滞后 ≤ **2** 会话 | 同上 |
| 申万指数（industry_taxonomy 侧数据） | 滞后 ≤ **3** 会话 | 同上 |
| 行业成员（industry_membership_pit） | 有效区间覆盖 ≥ **98%** 且区间重叠 = **0** | 同上（重叠另触发红线 §15.3） |
| 财务关键字段（三张版本表核心列） | 当前正式 scope 报告期覆盖 ≥ **90%** | 同上 |
| 一切源 | 来源不可计算一律 **fail 不得 pass** | fail-closed（无豁免） |

- 滞后 = 最近会话 − 该源最新 available_session（按 §6.1 映射）。
- D7 的资格门是**按公式依赖**的检查（P8 FACTOR_SET_QUALIFIED 用），不是全局
  新鲜度门（禁止实现"所有可选数据都必须新鲜"的全局 G9——t19 非目标已预注册，
  本表数值供其直接引用）。

## 13. 拉取策略与限速（D1 客户端 + D3 回填的预注册参数）

| 端点组 | 模式 |
|---|---|
| daily / daily_basic / adj_factor / stk_limit / margin_detail | 按 trade_date 全市场，一日一请求（2014-01-01 起逐日） |
| income / balancesheet / cashflow | 按 ts_code 全历史逐只 |
| index_member_all | 按 ts_code 逐只（单次 2000 行上限 → 防截断，t9 步骤 5）；源行三级展开为 ≤3 契约行（勘误 #2，附录 B） |
| forecast / express | 按公告日期增量；首次按年分段 |
| repurchase / stk_holdernumber | 按公告日期或按股票分段 |
| index_classify / index_weight / trade_cal / stock_basic / disclosure_date | 全量/按指数/按日（小表） |

- 限速 ≤ **180 请求/分钟**（官方 2000 积分档 200 次/分钟上限内取保守值）；
  429/超时 → 指数退避 + 抖动重试（有上限，超限即失败留档）。
- 单次返回行数 == 接口上限 → 必须分页拆分（`hit_row_limit=true` + 明确的
  needs_pagination 信号），禁止假定已完整入库（红线 §15.8）。
- 每批 = 一个 ingest_id；先 L0 快照、后 DB 提交；失败批整批回滚并记入
  source_ingest_runs（error_type/error_message）。
- token 仅经进程环境 / config/.env（git-ignored）注入；日志/异常/快照/
  commit 中零出现（红线 §15.10）。

## 14. 申万专项约束（D5 的数据侧预注册）

1. 区间语义半开：`[in_date, out_date)`；out_date 为空 = 仍在任。
2. 同一股票同一层级同一会话**至多一个**有效行业——回填审计与 D5 消费端双重
   fail-closed 校验；发现重叠 → 红线 §15.3 停止。
3. 必须同时取 `is_new=Y` 与历史记录（行业迁移史可追溯）；`industry_taxonomy`
   保留各代分类（index_code 代际 + industry_membership_pit.is_new——真实源
   字段、代际判定单一权威），**2021 版分类切换显式记录
   taxonomy 代际版本**（D5 审计含切换边界证据）。
4. legacy `sw_industry_member` 只读保留且**退出正式计算路径**（D5 切换后
   仅审计消费；grep 验证正式计算不再读它）。
5. 行业成员可用时间近似（in_date 23:59 → 次一开市会话，H5）与 Tier B 定级
   绑定：D5 验证门（官方公告对账 + 历史覆盖 + 可用时间核验）未通过则该族
   维持 Tier C，不升级。

## 15. RED 红线与停止条件（任一命中 → 停止、保留证据、上报 captain；
禁止静默变通、禁止换源凑数）

1. **未来泄漏**：任何 available_ts/available_date 早于官方发布时刻或早于
   §6.1 预注册保守值的构造；as-of 读到公告日当日/之前的修订。
2. **正式路径静默混入 AkShare/BaoStock**：Tushare 失败后未经显式裁决换源；
   审计对比数据未标注混入正式表。Tushare 失败 → 该源标 stale、依赖它的
   正式运行拒绝启动。
3. **行业区间重叠**：同股同层同会话多有效行业（含回填审计与消费端校验）。
4. **财务修订版本被覆盖/删除**：任何对版本表的按业务键 UPDATE/DELETE；
   update_flag 被过滤或丢失。
5. **生产 scope 外数据进入正式表**（§11.1）。
6. **来源字段/单位无法确认**：探针列名/单位与本文映射不符、文档间冲突
   （如累计 vs 单期口径含糊）→ 停下修订本文，禁止猜测（P13 §5.2 先例）。
7. **运行时存在其他 DB 写者**：回填/写入窗口独占确认（AGENTS §5.6）。
8. **行数上限未分页 / 截断入库**；**L0 快照缺失或 SHA-256 与台账不一致**。
9. **生产库被写**：ashare.duckdb 出现任何来自新管线的写连接/写操作。
10. **token 泄漏**到代码、日志、异常、快照、测试快照、commit 或任何输出。

## 16. 验证矩阵

- **本文（t1，docs-only）**：`git diff --check`；契约/索引同 commit 登记
  （CONTRACTS.md 单一指针，不作状态裁决）；无代码改动 → 无 pytest/compileall
  （未运行项如实列出）。
- **D1（t3）**：探针 20 接口留档（权限/列名/行数/日期范围，脱敏）+ 客户端
  RED（token 防泄漏 / fail-closed 字段校验 / 限速退避 / 行数上限信号）。
- **D2（t7）**：schema 幂等、PK/非空/日期/哈希约束直接断言、单批事务原子
  （失败零残留 + 台账记录）、先快照后提交顺序测试、legacy 表零改动，
  全部在临时 DuckDB。
- **D3（t9）**：§10 八步行数/日期留档、G1–G8 formal + research_doctor、
  trade_cal parity、覆盖率/重叠/修订审计；evidence-only 测量日志引用被测
  commit 与 v2 dataset_id。
- **D4/D5/D6（t12/t14/t15）**：各自 RED 清单 + 全量回归 + 版本 bump 兑现
  （§9）；golden parity 逐日比较（D4）；无第二套实现 grep 验证（D5/D6）。

## 17. 批准与实施边界

- 批准流程：**t2 评审 + 用户批准门**通过后本文生效；批准记录补记于本节
  （p16 先例格式：用户裁决日期 + 依据 + git 可查时序）。**批准前 t7（D2
  schema）与 t9（D3 回填）不得启动实施**；t3（D1 客户端/探针）为只读探测
  与客户端代码，不依赖本文批准（但其探针结论是本文 §5 映射的核对权威）。
- D4/D5/D6 语义切换：以本文为数据侧基座 + 各自预注册修订/实施契约
  （D4 契约修订 = t10，经 t11 评审 + 用户批准门；D5/D6 实施任务按 §6/§7/§14
  语义携带 RED）。
- 文件所有权边界（§12 并行规则）：本文仅 docs/；实施任务落点见各任务
  inScope（D1 客户端、D2 tushare_store/db/schemas/manifest、D4-D6 语义层）；
  热点文件（evaluation.py、run_sim.py 等）单点集成。
- 本契约不授权 push/远程 PR/合并；全程仅本地，integrator 串行小 PR（AGENTS
  §12）。未运行/未验证项如实列出；本文不预写任何测量结果（AGENTS §11.1）。

## 附录 A：接口权限与官方更新时刻（D1 探针核对清单）

来源：tushare.pro 官方权限表（[doc_id=108](https://tushare.pro/document/1?doc_id=108)，2000 积分档核对日 2026-09-03）。
"权限"列 = 官方最低分值要求；"更新"列 = 官方更新时刻（available_ts 保守化见 §6.1）。

| # | 端点 | 权限 | 官方更新 | 用途（L1 表） |
|---|---|---|---|---|
| 1 | daily | 120 | 交易日 15–17 点 | daily_bar_raw |
| 2 | adj_factor | 探针确认（官方权限表未单列；社区口径 2000 起） | 随 daily | daily_adj_factor |
| 3 | daily_basic | 2000 | 交易日 15–17 点 | daily_basic_pit |
| 4 | stk_limit | 2000 | 交易日 9 点 | daily_price_limit |
| 5 | suspend_d | 探针确认（2000 档 200 次/分） | 定时 | daily_suspension |
| 6 | margin_detail | 2000 | 每日 9 点（上一交易日明细） | margin_detail_pit |
| 7 | index_classify | 2000 | 全部分类 | industry_taxonomy |
| 8 | index_member_all | 2000 | 全部数据（单次限量） | industry_membership_pit |
| 9 | income | 2000 | 全部历史，实时更新 | income_statement_pit |
| 10 | balancesheet | 2000 | 全部历史，实时更新 | balance_sheet_pit |
| 11 | cashflow | 2000 | 全部历史，实时更新 | cashflow_statement_pit |
| 12 | fina_indicator | 2000 | 随财报实时更新 | financial_indicator_audit |
| 13 | disclosure_date | 2000 | 定期更新 | financial_disclosure_schedule |
| 14 | forecast | 2000 | 全部历史，实时更新 | earnings_forecast_pit |
| 15 | express | 2000 | 全部历史，实时更新 | earnings_express_pit |
| 16 | repurchase | 2000 | 每日定时更新 | repurchase_event_pit |
| 17 | stk_holdernumber | 2000 | 不定期更新 | holder_number_pit |
| 18 | stock_basic | 120 | 每日更新 | 股票状态元数据（§11.2） |
| 19 | index_weight | 2000 | 月度成分和权重 | index_weight_pit |
| 20 | trade_cal | 120 | 定期更新 | parity 对比（不替换日历权威） |
| — | stock_st / share_float / bak_basic | 预期无权限（share_float 官方 3000 起） | — | 明确报告，不影响主计划 |

> adj_factor 与 suspend_d 的权限档位以 D1 探针实测为准：若探针显示无权限，
> 触发停止条件 §15.6（停下修订本文），禁止换源凑数。

## 附录 B：源列→契约列显式映射表（勘误 t44 确立，常设机制）

任何源列名与契约列名不一致（改名/语义错位/笔误），必须先经勘误在本表落盘
显式映射后方可实施；D2/D3 按映射实施，禁止静默改名或跳列（§5.0 常设规则、
§15.6 的机制化）。未来新增不一致必须经勘误在本表追加行；未登记的源列差异
按 §15.6 停止。

| 勘误 | 表 | 源列（探针实证 2026-09-04） | 契约列 | 备注 |
|---|---|---|---|---|
| t44 | repurchase_event_pit | proc | proc | 实施进度（文本）；取值语义 D3 首批样本值落档核验 |
| t44 | margin_detail_pit | rqchl | rqchl | p17 起草笔误 `rqche` 已更正——现同名，无映射偏移 |
| t44 | daily_suspension | trade_date / suspend_timing | trade_date / suspend_timing | 恢复用户目标文档原始定义；resume_date/ann_date 源不存在，已删 |
| t44 | industry_taxonomy | （无 is_new 列） | （is_new 已删） | index_classify 源无 is_new；代际判定归 industry_membership_pit.is_new |
| t49 | industry_membership_pit | l1/l2/l3_code + l1/l2/l3_name（源行，无 index_code 列） | index_code + industry_name + level(1/2/3) | 一行源数据展开为最多 3 行契约行；空级跳过；level 为展开判别符（入 canonical 哈希输入）；in_date/out_date/is_new 各级沿用源值；available_date 按 §6.1 由 in_date 推导 |

## 附录 C：勘误记录（t44，2026-09-04，用户批准「批准全部」）

- **触发**：t6 评审（D1 验收，verdict=pass）D2 交接②——p17 DDL 与探针源列
  名不符，按 §15.6 走勘误；起草核对另发现同类差异（见下表决范围 3/4/5）。
- **草案**：`docs/p17_erratum_draft_t42.md` @ `fc03fb4`（governance-auditor，
  t42）。
- **用户批准原文（逐字）**：「批准全部」（2026-09-04，经 captain 会话转达）。
- **表决范围与 captain 裁决**：草案建议列 **1/2/3/4/5 全部采纳，项 6
  （stk_limit 行上限增补）跳过**；两处分流裁决——(a) pre_close **保留**并补
  来源说明（typed 显式字段请求供给，D1 实测 2026-09-04），否决草案的删列
  建议；(b) daily_suspension **删除 resume_date/ann_date**、suspend_date
  统一为 trade_date，恢复用户目标文档原始定义（ts_code/trade_date/
  suspend_type/suspend_timing/available_at/source/ingest_id）。
- **执行清单（本 commit 实际改动）**：§5.0 增补源列→契约列映射纪律常设
  规则；§5.2 daily_price_limit pre_close 补来源说明（保留）；§5.2
  daily_suspension 恢复用户原始定义（suspend_date→trade_date、删
  resume_date/ann_date、增 suspend_timing/available_at/source，主键
  (ts_code, trade_date, suspend_type)）；§5.2 industry_taxonomy 删 is_new
  （连带注释修正）；§5.2 margin_detail_pit rqche→rqchl；§5.4
  repurchase_event_pit proc_date→proc；§14.3 代际判定措辞 conforming
  修正；新增附录 B。
- **残留措辞说明（超出表决范围，不改）**：§6.1 suspend_d 行的 "suspend_date"
  措辞按本勘误读作 trade_date（同一语义、同一条映射规则），§6.1 正文零改动。
- **未采纳项**：项 6 跳过——p17 未钉死 stk_limit 行上限数值，§13 分页规则
  以"行数==接口上限"信号触发，无需修改。

### 勘误 #2（t50，2026-09-04，用户批准「这次批准，接下来你自行把握即可不用再问我」）

- **触发**：t9 step5-membership（index_member_all 回填）被 §15.6 拦截——
  p17 §5.2 假设源行携带单一 index_code，探针与实测证明源行为 l1/l2/l3
  三级同行结构、无 index_code 列（11 列清单见草案）。
- **草案**：`docs/p17_erratum2_draft_t49.md` @ `5b15bbb`（governance-auditor，
  t49）。
- **用户批准原文（逐字）**：「这次批准，接下来你自行把握即可不用再问我」
  （2026-09-04，经 captain 会话转达）。
- **表决范围**：草案项 1 + 项 2 **捆绑采纳**（拆开会产生中间态错误：有展开
  无哈希钉死 = 数据丢失）；范围外核对 ④（SW 双代际 src 取值确认步骤
  预注册）与 ⑤（附录 B 衔接）随批记录。
- **授权扩展记录（如实记录，含范围解释）**：用户原文把后续处置授权给
  captain 自行把握。团队按保守解释登记其范围为——**后续勘误的批准与
  流程批准点由 captain 代行**（p18 条款框架内的自主推进延伸）；**push/
  merge 等外部状态变更、预算规模变更仍需请示用户**。该解释为本团队对
  简式批准的保守读法，用户原文未逐项列举边界；如用户意图更宽或更窄，
  以用户后续明示为准（p18 §5 撤回/修订机制适用）。
- **执行清单（本 commit 实际改动）**：§5.2 表 8 增 `level INTEGER NOT
  NULL`（展开判别符，唯一授权派生列）+ 注释（三级展开、空级跳过）；§4.3
  哈希输入定义补句（展开后行 canonical 显式含 level，禁止以源行 11 列为
  输入——t47 keep-first 坍缩风险）；§13 index_member_all 行补备注（源行
  三级展开 ≤3 契约行）；附录 B 追加 t49 展开映射行。
- **零修改确认句（修正清单第 3 项）**：§14.3 的"同一层级"经展开后即同
  level 分组，规则文本零修改；§6.1 行业成员可用时间推导与 §6.3 as-of
  规则本体零修改（in/out/is_new 沿用源值，available_date 推导输入不变）。
- **PK 维持声明**：`PRIMARY KEY (index_code, ts_code, in_date)` 不扩大
  （level 经 index_code 跨级互异而隐式区分；病态同码跨级由 KEYED 冲突
  fail-closed 拦截）。
