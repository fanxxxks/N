# P6 按预测周期拆分研究域契约

状态：预注册（实现前）
适用版本：`PROTOCOL_VERSION = 24`（由 23 提升）、`REBALANCE_POLICY_VERSION = 2`
（由 1 提升）、`RESEARCH_DOMAIN_VERSION = 1`（新增）。不 bump：
`REWARD_VERSION = 14`、`EXECUTION_SPEC_VERSION = 2`、`DATA_TIER_VERSION = 1`、
`GRAMMAR_VERSION = 2`、`MODEL_VERSION = 3`、`FEATURE_REGISTRY_VERSION = 2`、
`TARGET_CONTRACT_VERSION = 1`。

本文是 P6 测试断言与验收的来源。仲裁顺序为本文/需求、测试、实现；测量结果
不得反向改写本契约。实施完成后按惯例补 `docs/p6_measurement_log.md`。

## 0. 目标与问题陈述

现状缺陷（用户确认）：慢速基本面（ROE 等）、20–120 日动量
（MOMENTUM_20/60、RET_120）与日内/隔夜特征（OVERNIGHT_RET/INTRADAY_RET）
在同一 `daily / horizon=1` 目标下、用同一 Reward（v14 单一参数）、同一换手
约束（单一 `turnover_budget`）一起被搜索与评估——这是语义混合：不同预测周期
的信号被强制塞进同一标签与同一执行纪律。

P6 目标：把研究域按预测周期拆分。每个研究域独立声明：**特征集、目标周期
（horizon）、执行周期（rebalance 频率）、Reward 参数与换手约束**。不同域
不得共用同一 Reward 语义与同一换手约束；跨域 `best_reward`/`val_reward`
不可比（不同目标周期、不同执行纪律），artifact 必须记录域身份。

## 1. 研究域定义

新增模块 `ashare_model/research_domain.py`，常量 `RESEARCH_DOMAIN_VERSION = 1`，
只读注册表 `RESEARCH_DOMAINS` 含三个内置域与一个兼容语义 `unified`：

| 域 id | 名称 | 特征语义 | 目标周期（意图） | 合法执行点 (frequency, horizon) | 默认执行点 |
|---|---|---|---|---|---|
| `short_price_volume` | 短周期价格量 | Tier A 价格、成交、涨跌停、流动性、日内/隔夜、微结构 | 1–5 日 | `(daily,1)`、`(every_5_days,1..5)` | `(daily,1)` |
| `medium_cross_section` | 中周期横截面 | 动量、反转、波动、分布、风险、技术、行业相对、外部横截面 | 5–20 日 | `(every_5_days,5)`、`(every_10_days,5..10)` | `(every_10_days,10)` |
| `slow_fundamental` | 慢周期基本面 | 估值、质量、增长、规模、股息 | 20 日以上（无上限） | `(every_20_days,20)`；`(monthly,h≥20)` 仅在该日期轴上相邻月度信号间隔 ≥ h 时合法（运行时校验） | `(every_20_days,20)` |
| `unified` | 兼容语义 | 全部活跃特征 | 任意合法组合 | 既有规则 | 不适用（保持现状） |

说明：

- 中周期意图范围 5–20 日，但非重叠标签约束（P3 §1）下 `every_10_days` 最多
  horizon=10，因此**合法执行点只覆盖 5–10 日目标**；11–19 日目标本期不设
  执行点，需要时以慢域 `(every_20_days, 20)` 表达。这是契约范围声明，不是缺陷。
- 慢域执行周期取"月度或事件驱动"之外的**每 20 交易日**为默认：真实 A 股
  日历的春节二月只有约 15 个交易日，`(monthly, horizon≥20)` 会违反 P3
  非重叠约束，因此 monthly 仅作为日历支持时的可选执行点（运行时校验，
  fail-fast，不静默抽稀）；事件驱动（按基本面披露日调仓）为本期
  Non-goal，留待后续契约。此修订经用户裁决（2026-08-30），取代初稿中
  "慢域默认 (monthly,20)" 的条款。
- `unified` 保留为兼容与基线测量路径：不施加任何域默认值，行为与 P6 之前
  逐字节一致。新研究 campaign 必须显式声明三个域之一；`unified` 运行在
  artifact 中记录 `research_domain: "unified"`。

每个域声明：`id`、`label`、`description`、`features`（显式特征元组）、
`horizon_range`（意图 `(min, max|None)`）、`frequencies`、
`default_frequency`、`default_horizon`、`baseline_signals`（默认基线，⊆
features）、`turnover_budget`（本域 L1 换手预算）、`cost_weight`（本域
Reward 成本权重）。

每域默认 Reward/换手参数（契约声明，非调参结果；执行周期越长，单次调仓
预算越紧，因为慢信号不值得高频换手）：

| 域 | `turnover_budget` | `cost_weight` | 默认基线 |
|---|---|---|---|
| short_price_volume | 0.20 | 1.0 | REVERSAL_5, TURNOVER, ILLIQ_20, OVERNIGHT_RET, LIMIT_UP_CNT_20 |
| medium_cross_section | 0.10 | 1.0 | MOMENTUM_20, RSQ_60, REVERSAL_60, IND_REL_RET_20 |
| slow_fundamental | 0.05 | 1.0 | ROE, PE_TTM, REVENUE_YOY, DIVIDEND_YIELD, MARKET_CAP |
### 1.1 特征归属（全量、互斥、穷尽）

`FEATURE_NAMES` 的 62 个成员中，61 个活跃特征恰好属于一个域；废弃特征
`NORTHBOUND_CHG` 不归属任何域（搜索空间在域模式下天然不含它；`unified`
维持现状）。

**short_price_volume（24）**：RET_1、RET_5、RET_10、REVERSAL_5；
TURNOVER、TURNOVER_CHG、TURNOVER_MA5、TURNOVER_MA20、TURNOVER_STD20、
VOLUME_RATIO、VOLUME_IMPACT、AMPLITUDE、CLOSE_POSITION；LIMIT_UP_EVENT、
LIMIT_DOWN_EVENT、LIMIT_STREAK、LIMIT_UP_CNT_20、LIMIT_BREAK；
OVERNIGHT_RET、INTRADAY_RET；ILLIQ_20、AMOUNT_SHARE；SUSPEND_DAYS_60、
LIST_AGE。

**medium_cross_section（25）**：MOMENTUM_20、MOMENTUM_60、RET_120、
REVERSAL_60、REVERSAL_120、HIGH_52W；VOL_20、VOL_60；SKEW_20、KURT_20；
MAX_20；BETA_60、IVOL_60、RSQ_60；BIAS_20、RSI_14、ATR_14、MACD_DIF、
MACD_DEA；IND_REL_RET_5、IND_REL_RET_20、IND_REL_VOL_20、IND_REL_TURNOVER；
INDUSTRY_MOMENTUM、MARGIN_BALANCE_CHG。

**slow_fundamental（12）**：PE_TTM、PB、PS_TTM、ROE、ROA、GROSS_MARGIN、
NET_MARGIN、REVENUE_YOY、PROFIT_YOY、DEBT_RATIO、DIVIDEND_YIELD、
MARKET_CAP。

归属依据（契约条款）：按特征族语义而非数据 tier——价格/成交/涨跌停/流动性/
日内/微结构归短域；动量/反转/波动/风险/技术/分布/行业相对归中域；估值/质量/
增长/规模归慢域。`MARGIN_BALANCE_CHG`（20 日两融变化，Tier B）与
`INDUSTRY_MOMENTUM`（20 日行业动量，Tier C 快照）是横截面外部特征，归中域；
`MARKET_CAP` 是日线重构的流通市值（Tier A），归慢域（规模是慢变量）。

### 1.2 合法执行点判定

`is_legal_execution(domain, frequency, horizon)`：`frequency ∈ domain.frequencies`
且 `horizon ∈ domain.horizon_range` 且 `RebalancePolicy(frequency, horizon)`
可构造（静态非重叠上限）。慢域的 `(monthly, h≥20)` 在日期轴可得处由
`RebalancePolicy.rebalance_mask` 做运行时校验（§2）。

## 2. RebalancePolicy 新增 `every_20_days` 与 `monthly` 频率（P3 §1 修订）

`RebalancePolicy` 新增两个规范值：

- **`every_20_days`**：可调仓信号日 = 从数据集首个交易日起按全局交易日
  序号 `0, 20, 40, ...`（与 every_5/every_10 同规则，全局锚定）。静态
  非重叠上限 20（相邻信号间隔恒为 20，horizon ≤ 20 即不重叠）。
- **`monthly`**：可调仓信号日 = 完整日期轴上每个自然月的最后一个交易日
  （按日期轴预解析；在折内或验证子窗口重新锚定属于协议错误，与 weekly
  同规则）。静态非重叠上限不适用于 monthly（月间隔随日历变化）：
  `__post_init__` 允许任意正 horizon；`rebalance_mask` 在解析完整日期轴
  后校验**相邻信号间隔 ≥ horizon**，不满足即 `ValueError`（fail-fast，
  报错引用 P3 §1 条款）。该校验只在 `horizon > 1` 时执行（horizon=1
  恒成立）。真实 A 股日历的春节二月只有约 15 个交易日，因此
  `(monthly, h≥20)` 在多数真实日期轴上不可执行——这是 P3 非重叠约束的
  诚实结果，slow 域的默认执行点是日历无关的 `(every_20_days, 20)`。

- `REBALANCE_POLICY_VERSION` 1 → 2。P3 契约 §1 的频率表同步修订（新增
  `every_20_days` 与 `monthly` 两行），并注明 monthly 的日历相关校验。

## 3. 配置契约

`protocol.domain`（默认 `"unified"`）选择研究域；三个域 id 之外的任何值在
配置阶段拒绝。域模式（`domain != unified`）下的默认值规则：

1. `protocol.frequency`/`horizon` 缺省 → 域默认执行点；显式给出时必须通过
   `is_legal_execution`，否则配置阶段拒绝；
2. `protocol.baseline_signals` 缺省 → 域默认基线；显式给出时必须全部 ∈ 域
   特征集，否则配置阶段拒绝；
3. `backtest.turnover_budget` 缺省 → 域默认 `turnover_budget`；
4. `reward.cost_weight` 缺省 → 域默认 `cost_weight`；
5. 其它既有字段语义不变；`unified` 不施加任何默认值。

慢域默认执行点 `(every_20_days, 20)` 静态合法、日历无关；显式选择
`(monthly, h≥20)` 时，月度日历合法性在日期轴可得处（`resolve_folds` /
`rebalance_mask`）fail-fast 校验。

## 4. 运行语义契约（域模式）

1. **域外特征恒为中性**：协议入口（`run_protocol`）在域模式下把
   `loader.factor_tensor` 替换为域限制张量——域外特征行置 0（float32，
   复制不原地改），行序/形状不变（token id 语义不变）。VM、回测、reward、
   基线、OOS 全路径消费该张量。
2. **搜索空间限制**：四个搜索后端（RL/random/GP/TPE）在域模式下只采样域内
   特征 token（全局 token id 不变；`build_action_mask` 增加可选
   `feature_ids`，缺省 None = 现状）。基线信号经 §3.2 校验必然 ⊆ 域特征。
3. **语义缓存窗口身份**：域模式下的 `window_id` 追加 `domain:<id>` 分量，
   域内公式得分不与 `unified` 或其他域混用。
4. **产物 provenance**：协议 artifact 记录 `research_domain` 与
   `research_domain_version`；`portfolio_config.turnover_budget` 已记录域
   实际换手约束；Reward 参数默认值由版本化注册表解析，运行级覆盖写入测量
   日志。`PROTOCOL_VERSION` 23 → 24（artifact schema 增加域维度）；
   `classify_protocol` 对当前版本的 artifact 要求 `research_domain` 字段，
   缺失即 legacy。
5. **日志与指标**：`run_protocol` 启动时结构化日志记录域 id、默认/生效执行
   点、域内特征数、生效换手预算与成本权重；失败路径不变。

Non-goals：事件驱动调仓；promotion/sim/champion 流程改造（champion 流程
消费既有 unified 语义产物，域研究先行；域 champion 的晋级流程留待域研究
出结果后的后续契约）；reword v14 实现语义；跨域组合/多域并行 campaign 编排。

## 5. 版本与迁移策略

- 提升：`PROTOCOL_VERSION` 23→24；`REBALANCE_POLICY_VERSION` 1→2。
- 新增：`RESEARCH_DOMAIN_VERSION = 1`。
- 不 bump：`REWARD_VERSION`（实现语义不变；域参数是配置维度）、
  `EXECUTION_SPEC_VERSION`（portfolio_config schema 不变）、
  `DATA_TIER_VERSION`、`GRAMMAR_VERSION`、`MODEL_VERSION`、
  `FEATURE_REGISTRY_VERSION`、`TARGET_CONTRACT_VERSION`。
- 旧产物迁移/拒绝：v23 及更早 protocol artifact 按既有版本不匹配规则自动
  legacy（不转换、不重写）；新 artifact 必须携带域字段。`unified` 即
  pre-P6 语义，任何既有运行/测试/产物无需改动即可继续。
- `every_20_days` 与 `monthly` 是新增频率，不影响既有四个频率的任何既有
  语义；P3 契约文档同步修订频率表（文档与代码同 commit）。

## 6. 预注册失败测试清单（首个代码提交即写入，先 RED 后 GREEN）

代码任务遵循项目惯例：测试先行。测试名与断言语义是本契约的一部分：

1. `tests/test_research_domain.py::test_domains_partition_vocabulary_exhaustively`
   — §1.1：61 个活跃特征恰好属于一个域；域间互斥；域特征 ⊆ FEATURE_NAMES；
   NORTHBOUND_CHG 不属于任何域。
2. `tests/test_research_domain.py::test_domain_defaults_are_legal`
   — §1/§1.2：每域默认执行点通过 `is_legal_execution`；默认基线 ⊆ 域特征；
   turnover_budget/cost_weight 为正且记录。
3. `tests/test_research_domain.py::test_legal_executions_respect_non_overlap`
   — §1.2：每域合法执行点中 `RebalancePolicy(f, h)` 均可构造；短域含
   `(daily,1)`；中域不含 `(every_10_days,11)`；慢域默认
   `(every_20_days,20)` 合法、`(every_20_days,21)` 非法、`(monthly,19)`
   非法。
4. `tests/test_research_domain.py::test_domain_of_feature_resolution`
   — 解析函数对未知特征名报错；示例特征归属正确（RET_1→short、
   MOMENTUM_20→medium、ROE→slow）。
5. `tests/test_research_domain.py::test_restrict_tensor_zeroes_out_of_domain_rows`
   — §4.1：域限制张量形状不变；域内行原值保留；域外行全 0；`unified` 恒等。
6. `tests/test_research_domain.py::test_feature_token_ids_follow_global_vocab`
   — §4.2：域特征 token id 是全局 vocab id；缺省为 None（全特征）。
7. `tests/test_research_domain.py::test_protocol_config_applies_domain_defaults`
   — §3：域模式缺省 frequency/horizon/baseline 取域默认（慢域 =
   `every_20_days`/20）；显式非法组合拒绝；域外基线拒绝；`unified` 与缺省
   完全一致。
8. `tests/test_research_domain.py::test_backtest_and_reward_defaults_follow_domain`
   — §3.3/3.4：`backtest.turnover_budget`、`reward.cost_weight` 缺省取域默认。
9. `tests/test_rebalance_policy.py::test_every_20_days_schedule_is_global`
   — §2：every_20_days 掩码 = 全局序号 `0, 20, 40, ...`（固定 fixture
   精确断言）。
10. `tests/test_rebalance_policy.py::test_monthly_schedule_is_last_session_of_month`
    — §2：monthly 掩码 = 每自然月最后交易日（固定 fixture 精确断言）。
11. `tests/test_rebalance_policy.py::test_monthly_overlap_prone_axis_rejected`
    — §2：monthly + horizon 超过某相邻月间隔的日期轴 → `rebalance_mask`
    抛 ValueError；间隔足够的轴正常。
12. `tests/test_research_domain.py::test_build_action_mask_restricts_feature_ids`
    — §4.2：给定 feature_ids 时只开放这些 feature token；其余 feature
    token 为 -inf；None 时与现状一致（全特征开放）。
13. `tests/test_research_domain.py::test_random_sampling_stays_inside_domain`
    — §4.2：`sample_random_formulas(feature_ids=...)` 采样序列解码后只含域内
    特征名。
13. `tests/test_research_domain.py::test_window_id_carries_domain`
    — §4.3：域模式 window_id 含 `domain:<id>`。
14. `tests/test_research_domain.py::test_protocol_artifact_records_research_domain`
    — §4.4：build_result 产物含 `research_domain` 与 `research_domain_version`。
15. `tests/test_artifact_versions.py::test_protocol_missing_domain_is_legacy`
    — §4.4：当前协议版本但缺 `research_domain` 字段的 payload 判定 legacy。

既有测试修改白名单（契约修订举证）：

- `test_rebalance_policy.py::test_unknown_frequency_and_non_positive_horizon_are_rejected`
  中 `RebalancePolicy("monthly", horizon=1)` 不再报错（§2 新增规范值），
  未知频率改用 `"hourly"`；断言强度不变。

## 7. 验收 Definition of Done

1. §6 全部测试 GREEN，且全量 pytest 通过数 ≥ 改动前基线（逐阶段记录计数，
   不新增 skip/xfail/弱断言）；
2. `unified` 路径（缺省配置）行为逐字节不变：既有协议/回测/训练测试全部
   保持 GREEN，且域默认值不注入；
3. 测量日志 `docs/p6_measurement_log.md` 记录：改动前后 pytest 计数、域
   划分统计（61 活跃特征 24/25/12）、every_20_days/monthly 日历样例、
   window_id/artifact 样例、命令与 commit；
4. 契约文档（本文）、P3 契约 §1 频率表修订、代码、测试同 commit 提交；
5. 全部测试本地通过后合并回 main 并推送。
