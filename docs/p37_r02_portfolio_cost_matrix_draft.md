# P37 R02 组合与成本矩阵·第一轮（六格：订单门槛 × 持仓结构，周期固定）预注册契约

- 状态：**APPROVED**，用户 2026-09-19 会话直接指令「阅读AGENTS.md和AlphaGPT_整改与提升执行方案_2026-09-09.md
  完成工作包 4：把 R02 首轮缩成六格，先固定调仓周期」（p30–p36 先例：用户直接指令
  批准 + 执行模型起草；批准记录回写于本文件头）。
- 起草：执行模型（GLM Harness 会话），分支 `codex/b04-load-cache-performance`。
- 勘察基线：`codex/b04-load-cache-performance @ b3f0c33`（WP1 p35 股数基记账、
  WP2 p36 用途分层资格、WP3 p34 rev3 报告补齐与正式重跑均已在该链上交付）。
  `git status` 干净（仅 `.mimosa/` 未跟踪目录，与本包无路径交集）。
- 上位依据：整改方案 §四 R02（组合与成本结构优化）与 §六 任务契约；用户本次指令
  对 R02 首轮的**收缩修订**（六格、周期固定、第二轮再独立比较合法周期）；
  AGENTS §3.2/§4.1/§4.4/§8；p8 生命周期契约；p34 R01 冻结臂与成本阶梯（本包候选
  与 0x/1x 定义的语义来源）。
- 契约 hash（本文件正文 sha256，实现 commit 与 IdeaRecord 引用）：由实现时点实时
  计算（后置测量引用于 `docs/p37_r02_matrix_measurement_log.md`）。

## 1 问题陈述与目标

整改方案 R02 原首轮建议 12 组合（持仓数 × 最小订单 × 两组调仓周期）。用户本次
指令将其收缩为**六格**：周期先固定，一次只回答「订单门槛与持仓结构」的问题——
原方案把周期放进首轮会同时改变执行频率；若连 horizon、方向学习也变化，归因将
更困难。第二轮再独立比较合法周期。

**目标**：

1. 在真实数据身份与正式生命周期下，对冻结的 2×3 格点面板（持仓数 10/20 ×
   最低订单 1000/2500/5000 元）测量固定候选信号的真实可执行性与净空间，
   资本固定 10 万元。
2. 产出可复核面板报告：全部格点、年度表现、交易活动、真实费用、**连续目标
   权重与整手实现账本的差异**。
3. 按本契约 §7 冻结规则判定：存在「相邻配置均能成立的区域」/ 单一最高格点
   （不稳定结果）/ 毛信号有信息但执行受阻（继续执行诊断）/ 整个固定面板没有
   可靠信息（暂停扩展搜索）。

**非目标**：

- 不比较调仓周期（every_5_days/every_20_days/monthly 等合法周期属第二轮，
  另立指令与运行）；本包周期**冻结为 every_10_days / horizon 10**。
- 不做 equal_weight/optimizer 对照、0.1x/1x/10x 资本缩放、2x 成本压力
  （整改方案 R02 原文：下一轮独立对照；本包成本档仅 0x/1x，见 §5）。
- 不修改生产默认配置（`config/ashare_config.yaml`）、不改默认策略语义；
  六格变体全部经 `dataclasses.replace` 从冻结基础配置派生。
- 不运行任何搜索臂；不消费唯一语义评价预算。
- 不做晋级、paper、受保护 OOS 声明。测试年 2021–2025 为开发验证窗口。
- 不处置三个已知数据阻断（申万源新鲜度 / 1,630 码 pre-listing 限额格网
  披露语义 / 001914 前身历史限额）——它们是 p17-D5/p19/数据修复各自范围内的
  契约治理决策，超出本包授权；本包正式运行若被其 fail-closed 拒绝，按诚实
  终态落账（§9）。

## 2 数据与身份

1. 数据身份：与 p34 §2 同一链——`AshareDataLoader` confirm 模式（A02 复合确认）
   → `frozen_data_cutoff()` → 运行前漂移守卫。本轮运行前**重新实时确认**，
   不沿用任何旧值。
2. G1–G8：`ProductionGateRunner(data_config).run(mode="formal",
   purpose=historical_research, snapshot_cutoff=…)`（p36 历史基准；快照时间线
   与墙钟解耦，同一确认快照任意运行日资格一致）。
3. 生命周期：RunStore 根 = `data_config.data_dir`（r1_medium_a 即
   `data/r1/medium_a`，与 evaluation/R01 入口同一解析，无第二套根）。
   状态链：`IDEA → SPEC_LOCKED → DATA_QUALIFIED →（§5 面板证据生产）→ 裁决`。
   **本包不进入 `FACTOR_SET_QUALIFIED`/`SEARCH_PLAN_ADMITTED`**：因子集资格属
   R01 运行自己的链（R01 从未到达，亦无必要由本包代行）；本包是执行层研究，
   裁决只决定本 run 的行政走向（驻留/退役）。
4. RunSpec 冻结基础配置（`configs/r1_medium_a.yaml` 全字段原值，含
   top_n 20 / buy_rank 20 / sell_rank 30 / min_trade 5000 / cap 0.05）；六格
   面板是基础配置 + 本契约 §3 冻结格点表的**纯函数**，其 provenance 由契约
   sha（IdeaRecord 与报告均印签）+ 报告内逐格参数表承载。格点不进入
   RunSpec 的配置 hash——不存在第二套 spec 身份路径，面板内变体是测量内
   处理（与 p34 每臂同一 spec 下 0x/1x/2x 变体同型）。

## 3 冻结的六格面板（不可在见数后修改）

### 3.1 固定项（全部 = r1_medium_a.yaml 原值）

| 项 | 冻结值 |
|---|---|
| 研究域 / tier | `medium_cross_section` / `tier_set: A` |
| 频率 / 视界 | `every_10_days` / horizon 10（**本轮不比较周期**） |
| folds | 5 个：train_end 2020-12-31…2024-12-31，test_end 2021-12-31…2025-12-31 |
| 股票池 | calendar/constituents 主库 + v2 PIT store，`min_listed_sessions: 60` |
| 资金 | `initial_capital: 100000.0` |
| 调仓阈值 | `target_weight_change_threshold: 0.01` |
| 费率（1x） | commission 0.00025 / min_commission 5 / stamp 0.0005 / transfer 0.00001 / slippage 0.0005 |
| 整手 | `lot_size: 100`（A 股板） |

### 3.2 六格定义（持仓数 × 最低订单）

格点耦合参数**预先写清**（整改方案 R02 原文要求：单股上限必须与持仓数一致，
买卖排名缓冲必须预先冻结）：

| 格名 | top_n | buy_rank | sell_rank | single_weight_cap | min_trade_amount | 耦合规则 |
|---|---|---|---|---|---|---|
| `n20_m5000`（参考格） | 20 | 20 | 30 | 0.05 | 5000 | r1_medium_a 原值逐字 |
| `n20_m2500` | 20 | 20 | 30 | 0.05 | 2500 | 同上，仅 min_trade 变 |
| `n20_m1000` | 20 | 20 | 30 | 0.05 | 1000 | 同上，仅 min_trade 变 |
| `n10_m5000` | 10 | 10 | 15 | 0.10 | 5000 | 见下两条规则 |
| `n10_m2500` | 10 | 10 | 15 | 0.10 | 2500 | 同上 |
| `n10_m1000` | 10 | 10 | 15 | 0.10 | 1000 | 同上 |

耦合规则（冻结，禁止见数后调整）：

1. **buy_rank = top_n**：等权单元权重 = `min(1/buy_rank, single_weight_cap)`
   （构造器唯一语义）。buy_rank 即持股槽数。
2. **single_weight_cap = 1/top_n**（n20→0.05，n10→0.10）：满仓可表达。研究
   10 只等权时保持 5% 上限会使最大投入 ≤50%，正是整改方案点名禁止的
   「单股上限与持仓数不一致却假设满仓」。
3. **sell_rank = ⌈1.5 × top_n⌉**（n20→30，n10→15）：保持 r1_medium_a 的
   1.5× 排名缓冲比例（持仓 rank < sell_rank 者留存），缓冲随持仓数等比缩放，
   不引入新的缓冲自由度。
4. 其余字段（费用、阈值、周期、池、资金）逐字不变；格点变体仅替换上表五个
   字段（`cell_variant` 纯函数，测试断言无其他字段漂移）。

**特格诊断（整改方案 R02 特别诊断原文）**：`n20_m5000` 满足
20×5000=100000 **恰好覆盖全部本金**——叠加现金、费用、整手、价格波动及排名
缓冲后，实际建仓/换仓与理想权重可能偏离。必须测出来（§6 lotgap 与首建仓
漏斗），不能只按名义乘法断言错误或有效。

## 4 冻结候选（数据合格、有明确选择依据的少量信号）

**选择依据 = 预注册，不是 OOS 表现**：R01 正式测量至今被数据资格 fail-closed
阻断（p34 测量日志 §7.3/§7.5），从未产出任何基线 OOS 数字；不存在、也不允许
存在「按 R01 结果挑臂」。臂集与 p34 §3.2 完全一致（p21 预注册 Tier-A 基线
信号 + 冻结规则组合）：

| 臂 | 构造（冻结，同 p34 §3.2） |
|---|---|
| `baseline:MOMENTUM_20` / `baseline:RSQ_60` / `baseline:REVERSAL_60` | 裸因子直接作为信号（Tier-A 价量，数据合格性由既有 tier 契约承载） |
| `combo:all3` | 三臂成员逐日截面 rank 归一（`reward._rankdata`）等权平均；缺失成员取可得均值，全缺失置 NaN |
| `combo:momentum_reversal` | 同上，成员 = MOMENTUM_20 + REVERSAL_60 |
| `benchmark:equal_weight`（参考） | 既有 `benchmark_row`，零成本、格点无关，每折一次 |
| `control:cash`（对照） | 全 NaN 信号 → 全程空仓；空仓与格点参数无关，仅在参考格 `n20_m5000` 1x 测一次并披露 |

**方向规则（冻结；用户指令：方向学习不得随格点变化）**：每折在**参考格
`n20_m5000`（= r1_medium_a 原值）**上由既有 `baseline_candidates` 计算一次
训练窗方向（方向 = 训练窗验证段前向 rank IC 符号，`reward.signal_direction`
唯一权威，与组合参数无语义依赖），**跨全部六格冻结复用**。逐折方向连同其
证据（val_reward/val_icir）入账。组合方向 = 成员同方向规则（同 p34）。

## 5 成本档（每臂 × 每折 × 每格 × 两档）

| 档 | 定义（冻结，复用 p34 §3.3 权威 `cost_variant`） |
|---|---|
| 0x（毛） | 全部成本参数置零——归因「毛信号是否有信息」的唯一依据 |
| 1x（净） | r1_medium_a.yaml 原值（真实费用） |

2x 压力、资本缩放、optimizer 对照不在本轮（§1 非目标）。基准行零成本不参与
分档（`ref`）；现金对照在 1x（空仓定义下费用为零，如实披露）。

## 6 报告必含（每格 × 每臂 × 每折 × 每档）

全部 p34 §3.4 既有行级字段（净/毛超额、`oos_active_ir`、回撤、Sharpe、IC 双
口径、调仓应发生/实际数、订单数、抑制数、换手、持仓数、投入比例、
`cost_fraction_sum`、费用参数原文、执行漏斗五桶与抑制原因拆分）**外加**本包
新增的连续-整手差异字段（`evaluate_signal` 附加可观测，引擎与构造器零改动，
既有字段逐位不变）：

- `lotgap_l1_mean_due` / `lotgap_l1_max_due`：应调仓会话上
  ½·‖连续目标权重 − 整手实现权重‖₁ 的均值/最大（实现权重 = 引擎执行后账本
  权重，即 `positions` 记录的书；无应调仓会话时为 null）；
- `lotgap_invested_mean_due`：应调仓会话上 |Σ目标 − Σ实现| 的均值（现金尾差
  / 买不起整手 / 费用侵蚀的直接度量）；
- `funnel_initial_funding_sessions` / `funnel_initial_funding_planned_orders` /
  `funnel_initial_funding_filled_orders`：首建仓会话（构造器
  `initial_funding` 判定）数及其计划/成交订单数——特格 `n20_m5000` 的
  20×5000=100000 覆盖诊断直接消费。

汇总层：每 (格,臂) 的跨折均值统计、逐折年度分解、跨折中位/最小/最大、
`stitch_oos_series` 拼接统计（唯一拼接权威；拼接 IR 不得用折均 IR 代替，
两者并列分别命名）；每格漏斗合计；每格逐臂连续-整手差异合计；**全部格全部
臂全部折全部档入报告，不裁剪、不突出最高一行**。JSON 序列化 NaN→null。

## 7 裁决规则（冻结；执行模型不得用结果反改）

对每个成本承载臂（3 单因子 + 2 组合；基准与现金对照不参与），在每格上：

- **臂在格上「成立」** ⇔ 1x 跨折平均净超额 > 0 且 1x 平均 `oos_active_ir` > 0
  且 `invested_ratio_mean` ≥ 0.5 且 `mean_holdings_1x` ≥ ⌈top_n/2⌉（半仓
  下限；n10→5，n20→10）且失败折 < 2。
- **格「成立」** ⇔ 该格存在 ≥1 成立臂。
- **相邻**：格点图上恰好一个轴变化一步（n10_m1000–n10_m2500 相邻；
  n10_m1000–n20_m1000 相邻；n10_m1000–n20_m2500 不相邻）。
- 单年依赖（1x 净超额 > 0 的折数 < 3/5）与成本信息（0x 正 1x 非正）作为
  **旗标**随格报告，不改变成立判定（本轮的稳健性判据是相邻区域，不是
  单年依赖；年度分解全部展示）。

**verdict（四路，优先级即枚举顺序）**：

1. `stable_region_found` ⇔ 存在成立格且其至少一个相邻格也成立。
2. `isolated_single_winner` ⇔ 全面板恰有一个成立格且无成立邻居 → **记录为
   不稳定结果**：不据此选择参数，报告明示「单一最高格点不构成参数选择
   依据」。
3. 无成立格 且 存在 (格,臂) 0x 拼接超额 > 0 且 0x 拼接 active IR > 0 →
   `gross_info_execution_blocked`：毛信号有信息但执行受阻 → **继续执行诊断**
   （下一独立包），本报告必须携带阻断证据（抑制原因拆分、未成交率、投入
   比例、首建仓漏斗、lotgap 合计）。
4. 无成立格 且 无任何 (格,臂) 毛信息 → `panel_no_reliable_info`：
   **暂停扩展搜索**——第二轮周期比较与矩阵扩展均暂停，瓶颈在信号而非执行。

行政走向（p34 §3.7 同型，非晋级证据）：verdict 1/2 → run 停驻
`DATA_QUALIFIED`（非终态；第二轮周期比较另立显式指令与运行身份）；verdict
3/4 → `RETIRED`（原因引用报告 hash 与 verdict）。任何 verdict 都不产生
champion/晋级/holdout 声明；「唯一最高格点」永不构成默认参数变更依据（整改
方案 R02：默认配置只有经独立确认后再变更）。

## 8 预算（冻结）

- 墙钟上限：**6 小时**（依据：p34 §3.6 同源外推——90 次引擎折运行 ≤1 h；
  本轮 ≈ 6格×5臂×5折×2档=300 次 + 方向路径 15 次 + 基准/对照 10 次 ≈ 325
  次，warm 因子缓存命中预期 ≤3.5 h；上限含全部重试一次的机会）。超限即停
  止并记 `FAILED`，不得缩窗/减格冒充完成。
- 唯一语义评价预算：`consumed=0`（无搜索臂；RunSpec `requested_budget` 沿用
  p21 预注册值仅作记录）。
- 内存：单进程运行（16 GB 主机，p33 实测全历史装载峰值 ~7.9 GB）；禁止与
  本运行并发其他重载进程。

失败/缺折规则（同 p34 §3.5）：逐 (格,臂,折,档) 捕获失败记 `failed` 行及
原因；≥2 折失败的 (格,臂) 标 `unreliable`；不虚构零收益补位。

## 9 停止条件

- 门禁拒绝（G1–G8 任一 fail、类型化降级未消除、复合身份失效）→
  `REJECTED`（携带已登记证据）诚实落账。**当前已知数据阻断（申万源滞后
  7 会话 / 1,630 码 pre-listing 限额格网 / 001914 前身历史）若仍在，本包
  正式运行将如实落 REJECTED——这与 WP3 重跑同一落点，是测量不完整的真实
  状态；面板软件交付不受影响，数据解锁后以同一命令重跑即可。**
- 装载/引擎异常或 6 h 超限 → `FAILED`，保留失败证据，禁止缩窗/减格重跑
  冒充。
- 契约与实现冲突 → 停下修订契约，不得改实现凑数。
- 发现周期比较、optimizer、资本缩放、2x 压力或搜索需求 → 另立指令，不扩权。

## 10 与生命周期动作模式的对应（p27 §3）

- 无正式 `train`/`evaluation`/`backtest`/`paper_sim` 入口调用；面板是
  `DATA_QUALIFIED` 状态内的资格/研究证据生产，由 `LifecycleWriter` 单一守卫
  路径记录，报告 artifact 登记进 run store。
- 机器门禁裁决权不变：`qualify_data` 的验证器独立裁决；本包 §7 规则只决定
  run 在证据生产后的行政走向。

## 11 变更分类与版本影响表

本包为研究语义新增（新测量矩阵），但不改任何既有语义默认值：

| 版本 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `PROTOCOL_VERSION` | 27 | 不变 | 不运行评价协议；`evaluate_signal` 只加附加字段（p34 rev3 同例），既有字段数值零变化 |
| `EXECUTION_SPEC_VERSION` | 4 | 不变 | 引擎/构造器零改动（lotgap 从既有结果对象读取） |
| `PORTFOLIO_CONSTRUCTOR_VERSION` | 1 | 不变 | 构造器零改动；格点变体经既有 `BacktestConfig` 字段表达 |
| `REWARD_VERSION` | 15 | 不变 | 评分器零改动（方向沿用既有权威） |
| `RUNSPEC_SCHEMA_VERSION` | 3 | 不变 | 无新 spec 字段（格点 provenance 走契约 sha + 报告，§2.4） |
| `LIFECYCLE_CONTRACT_VERSION` | 2 | 不变 | 状态集/边/裁决规则零变化；`DATA_QUALIFIED → RETIRED` 为既有合法边 |
| R02 报告 artifact | — | `r02_report_schema_version=1`（新测量产物，非 `artifact_schemas` 消费者） | 与 R01 报告同型（p34 §4 版本表先例）；不进入晋级证据链 |

工程改动（零研究语义默认值变化）：

1. `ashare_model/eval_metrics.py`：`evaluate_signal` 返回值**附加** §6 五个
   lotgap/首建仓字段（从既有 `result.target_weights`/`positions`/
   `buy_weights`/`sell_weights`/`construction_diagnostics` 读取，引擎零改动，
   既有字段逐位不变）。
2. 新增 `ashare_model/r02_matrix.py`（本轮唯一驱动，复用
   `r01_baseline` 的 `cost_variant`/`combo_signal`/`cash_signal`/
   `summarize_arms`/`rows_to_jsonable`/`resolve_folds_for` 单一权威，不复制
   第二套）+ `scripts/r02_matrix.py` CLI shim + 测试。
3. `scripts/check_test_shards.py`：新测试文件同 commit 注册。

## 12 预期 RED 测试（实现前失败）

`tests/test_p37_r02_matrix.py`（同 commit 注册分片）：

- W1 格点冻结：六格参数逐项断言（含独立不变量 `cap == 1/top_n`、
  `sell_rank == ⌈1.5·top_n⌉`、参考格与 r1_medium_a 原值逐字一致）。
- W2 `cell_variant`：恰替换五字段、其余字段逐位不变；未知格名报错。
- W3 lotgap 手算：合成引擎（p35 同型）上按手算期望断言
  `lotgap_l1_mean_due`/`lotgap_invested_mean_due`/首建仓计数；无应调仓会话
  → null；既有字段数值不变。
- W4 裁决纯函数：四路 verdict 手造摘要逐项断言（含持仓下限随 top_n 缩放：
  holdings=7 在 n20 格不成立、在 n10 格成立；无成立格+毛信息 →
  `gross_info_execution_blocked`；无成立格+无毛信息 → `panel_no_reliable_info`）。
- W5 相邻定义：三组相邻/非相邻断言。
- W6 报告 json-safe + `r02_report_schema_version == 1`。
- W7 端到端（迷你双库，真实裁决器 + canary 阈值测试缝）：正 verdict 驻留
  `DATA_QUALIFIED`；负 verdict → `RETIRED`；数据门禁拒绝 → `REJECTED`
  （携带证据 artifact）；契约 sha 不符 → 昂贵计算前拒绝。

## 13 验证矩阵

1. `python -m pytest -q tests/test_p37_r02_matrix.py`
2. 相邻契约：`tests/test_p34_r01_baseline.py tests/test_p34_r01_report_wp3.py
   tests/test_p35_share_accounting.py tests/test_evaluation.py
   tests/test_baseline_harness.py tests/test_run_store.py`
3. `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi`
4. `python scripts/check_test_shards.py --check`
5. `git diff --check`
6. 全量：`python -m pytest -q tests -n auto`（结果入测量日志）
7. 正式运行后：测量日志（命令/commit/dataset id/契约与配置 hash/墙钟/内存/
   原始产物路径/sha256）。
