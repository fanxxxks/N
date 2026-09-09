# P19（D4）实施测量日志（t12，evidence-only）

- 被测 commit：**codex/d4-price-semantics @ `e36e226`**（链 `6026129` → `b0a1c00`
  [p19 契约导入，cherry-pick 自 26d2697] → `e36e226` [实现原子提交，44 文件
  +2166/−406]），worktree `D:/minequant/AlphaGPT-d4`，本地未 push。
- 环境：Windows / Python 3.13.12（D:/minequant/.venv）/ duckdb 1.5.5 /
  pytest 串行；无 GPU 路径触发。
- 数据前置（p19 §6.4）：v2 gate adapter（t53）在调度器投递 t12 时未完成，
  按契约允许的**等效 scoped 资格审计**执行：对
  `D:/minequant/AlphaGPT-d2/data/ashare_v2.duckdb` 只读审计——
  daily_bar_raw 11,886,384 行 / daily_adj_factor 12,246,154 行 /
  daily_price_limit 12,209,690 行 / daily_suspension 336,077 行 /
  daily_basic_pit 11,896,917 行，范围均 20140102..20260904；
  OHLC/vol/pre_close/amount 零空值，adj_factor 零空值零非正，
  up_limit/down_limit 零空值（limit 表 pre_close 27,628 空值不被 loader
  消费）。行数/范围与 t9 记录逐表一致（价格四表未受 t53 补跑影响）。
  注意：v2 库 dataset_id 已被 t53 补跑更新为
  `43ef1e4caace5cfc3ffc5bef335a3bf613e538dfbff89c405370848e980ce27d`，
  p19 引用的 `aa6f6d55` 为 D3 时点 id（价格表字节不变）。

## 验证矩阵（实际发生）

| 命令 | 结果 |
|---|---|
| `python -m pytest -q tests/test_price_semantics.py tests/test_golden_parity.py tests/test_backtest.py tests/test_run_sim.py tests/test_data_loader.py` | 114 passed / 0 failed，94.66s（在被测 commit 上执行） |
| `python -m pytest -q tests`（串行） | **1648 passed / 1 skipped（既有基线）/ 0 failed**，1319.22s；基线 1622 → 净增 26 |
| `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts` | exit 0 |
| `git diff --check` | exit 0 |
| `python scripts/check_test_shards.py --check` | 94 文件并集 = 全集，exit 0（test_price_semantics.py 注册 shard-data） |

RED 先行证据：实施前（02:15–02:17）`tests/test_price_semantics.py` 19/19
全部按契约预期失败（loader v2 能力缺失 "No daily bars found"、engine 缺
p19 键 DID NOT RAISE、版本常量 2!=3 等），无无关失败；实施后同文件全绿。
全量验证为串行单轮（AGENTS §10.2 回退条款），未做并行对账。

集成核验：`git merge-tree`（无变异）对 codex/tushare-schema-v1 尖端
（t53 后 `1df993b`，8 commit 仅触及 tushare_store.py /
test_tushare_store.py / d3 日志）零冲突、零双方修改文件。

## 版本影响核对表（全量 33 项，git grep 实际结果 @ 基线 6026129；行号为基线）

**bump（6 项，与 p19 §5 表逐字一致）**：

| # | 所有者 | 旧 → 新 |
|---|---|---|
| 1 | ashare_portfolio/execution_spec.py:22 EXECUTION_SPEC_VERSION | 2 → 3 |
| 2 | ashare_model/versions.py:17 PROTOCOL_VERSION | "25" → "26" |
| 3 | ashare_model/factors.py:46 FACTOR_COMPUTE_VERSION | 1 → 2 |
| 4 | ashare_model/targets.py:12 TARGET_CONTRACT_VERSION | 1 → 2 |
| 5 | ashare_model/bare_factor_backtest.py:56 BARE_FACTOR_BACKTEST_VERSION | 3 → 4 |
| 6 | ashare_model/p3_measurement.py:50 P3_MEASUREMENT_VERSION | 3 → 4 |

**不变（27 项，逐项 no-bump 依据；含 t11 指出的两处漏名）**：

| # | 所有者 | no-bump 依据 |
|---|---|---|
| 7 | ashare_portfolio/constructor.py:23 PORTFOLIO_CONSTRUCTOR_VERSION | 纯权重语义；金额=权重×资金，无价格消费（grep 证实） |
| 8 | ashare_portfolio/rebalance.py:12 REBALANCE_POLICY_VERSION | 调仓日历/进出场索引语义不变 |
| 9 | ashare_portfolio/optimizer.py:41 PORTFOLIO_OPTIMIZER_VERSION（t11 漏名） | 优化器输入=信号/权重/约束/adv，无第二套价格换算；raw 化只改变其输入数据值 |
| 10 | ashare_data/tushare_store.py:73 TUSHARE_STORE_SCHEMA_VERSION | L1 schema 零改动 |
| 11 | ashare_model/cost_matrix.py:47 FEE_MATRIX_VERSION | 费率表语义不变；金额基数变化由 EXECUTION_SPEC 3 承载 |
| 12 | ashare_data/manifest.py:54 MANIFEST_VERSION | v2 manifest 新实例、dataset_id 重算（p13/p17 先例） |
| 13 | ashare_model/data_tier.py:37 DATA_TIER_VERSION | tier 映射与时间规则未变（本任务零改动核实） |
| 14 | ashare_model/elite_archive.py:15 ELITE_ARCHIVE_VERSION | 归档结构未触 |
| 15 | ashare_model/feature_registry.py:61 FEATURE_REGISTRY_VERSION | 特征/词表未增删（公式未变，仅输入基座） |
| 16 | ashare_model/imitation.py:12 IMITATION_VERSION | 未触 |
| 17 | ashare_model/ledger.py:47 LEDGER_SCHEMA_VERSION | 未触 |
| 18 | ashare_model/lifecycle.py:94 LIFECYCLE_CONTRACT_VERSION | 未触 |
| 19 | ashare_model/searcher_adjudication.py:38 P10_ADJUDICATION_VERSION | 未触 |
| 20 | ashare_model/run_store.py:49 INDEX_SCHEMA_VERSION | 未触 |
| 21 | ashare_model/promotion.py:91 PROMOTION_RULE_VERSION | 晋级门禁语义不变，经既有 G1 pin 等值规则消费新版本集合 |
| 22 | ashare_model/runspec.py:40 RUNSPEC_SCHEMA_VERSION | 未触（版本经 _VERSION_IMPORTS 读取新值） |
| 23 | ashare_model/rl_diagnostics.py:12 RL_DIAGNOSTICS_VERSION | 未触 |
| 24 | ashare_model/semantic_cache.py:40 SEMANTIC_CACHE_VERSION | 未触 |
| 25 | ashare_model/reward.py:155 REWARD_VERSION | 公式/带域不变；输入基座变化由 PROTOCOL "26" 承载 |
| 26 | ashare_model/research_domain.py:54 RESEARCH_DOMAIN_VERSION | 未触 |
| 27 | ashare_model/search_contract.py:31 SEARCH_CONTRACT_VERSION | 未触 |
| 28 | ashare_model/searcher_bench.py:83 SEARCHER_BENCH_VERSION | 未触 |
| 29 | ashare_model/tier_reports.py:35 TIER_REPORT_VERSION（t11 漏名） | tier 报告结构与 tier 判定未变（DATA_TIER 同上零改动） |
| 30 | ashare_model/artifact_schemas.py:44 ARTIFACT_SCHEMA_VERSION | versions dict 记录新版本集合为 additive（p9/p13 先例） |
| 31 | ashare_model/admission.py:18 ADMISSION_RULE_VERSION | 准入规则与价格基座无消费关系 |
| 32 | ashare_model/alphagpt.py:33 MODEL_VERSION | 模型本体未触 |
| 33 | ashare_model/vocab.py:285 GRAMMAR_VERSION | 词表/算子未增删 |

新增 artifact-facing 版本：**无**（与契约 §5 一致）。旧 artifact 拒绝经既有
分类权威：`artifact_versions.version_matches` 等值规则 +
`promotion` G1 pin（RED-6 测试覆盖）。

## 边界与未运行项

- 本日志仅记录工程验证（AGENTS §4.1 engineering）；不构成 alpha/晋级/
  production 证据。合成数据 parity 只证明软件行为一致。
- 未运行：并行全量对账（串行回退条款）、`research_doctor`、生产库
  G1-G8（legacy 生产库未触；v2 gate adapter 归 t53/t55）、push/PR/merge
  （未授权）。
- 已知范围备注：turnover_rate 正式源切至 v2 daily_basic_pit（Tier A，
  为遗留表退役前置）；golden lot 路径在 raw 价格 + adjusted 标签基座下，
  除权事件残差落入既有 carry 归因（golden.py docstring 声明，非静默
  分歧）；fresh G8 fixtures 补 pre-listed 未来会话（周末时钟翻转缺陷，
  先于本任务存在）；factors.py 涨跌停特征族 `limit_rate` 消费保留
  （t22 原位修复范围，非 D4 执行阻断路径）。

## 附录：数据前置升级审计（captain 裁决 3a，evidence-only @ `064018a` 后）

- 触发：captain 批准等效审计路径时附两条件——(a) 四表全量双向核对 +
  L0 快照抽查；(b) t53（gate adapter + G1-G8）完成是 t12 交付硬前置。
  裁决送达时 t12 已按调度器投递完成（时序例外），本附录为条件 (a) 的
  补齐与 (b) 的收口记录；顺序例外由 captain 授权记录在案（理由 = 文件
  不相交 + AGENTS §12 并行线规则 + 门禁完成仍是交付硬条件）。
- 前置状态核验：t53 [completed]（gate adapter 773f47a/0a1eb1e/08b2016）、
  t55 [completed]（constituents ← index_weight_pit 映射修正 b2a7d8b）；
  G1-G8 最终 6/8 PASS（G1/G2/G3/G4/G5/G8 ✓），G6/G7 N/A-with-reason
  （t53/t55 output 与 d3 日志留证）。t12 完成时 t53 数据收口（dataset_id
  `43ef1e4c…`）已在库上，任务状态翻终态的先后无法事后精确判定，如实
  记录。
- 升级审计命令：`python D:/minequant/audit_price_tables_t12_upgrade.py`
  （venv python，duckdb read_only + pandas/pyarrow 读 L0 parquet，
  库 `43ef1e4c…` = t53 补跑后重建 manifest，created_at
  2026-09-04T19:12:31+00:00）。p19 引用的 `aa6f6d55` 为 D3 时点 id，其
  逐表计数与下表相同（t9 记录），本次双向核对以当前库上权威 manifest
  为准，双向一致即同时覆盖旧 id 的计数声明。

| 表 | manifest | live | 双向 | 日期范围 |
|---|---|---|---|---|
| daily_bar_raw | 11,886,384 | 11,886,384 | OK/OK | 20140102..20260904 一致 |
| daily_adj_factor | 12,246,154 | 12,246,154 | OK/OK | 20140102..20260904 一致 |
| daily_price_limit | 12,209,690 | 12,209,690 | OK/OK | 20140102..20260904 一致 |
| daily_suspension | 336,077 | 336,077 | OK/OK | 20140102..20260904 一致 |
| daily_basic_pit | 11,896,917 | 11,896,917 | OK/OK | 20140102..20260904 一致 |

- L0 快照抽查（抽样日期 20240102、20250610；端点 daily_bar_raw 与
  daily_price_limit 的全部 L0 批次文件）：live 行 10,200 / 10,228，
  快照行 10,200 / 10,228，键差 0/0，值差异 0/0 —— 两端点均 PASS
  （快照与库内容逐值一致，manifest 计数由快照批次可信支撑）。
- 结论：**AUDIT_UPGRADE PASS**——裁决 3(a) 两条件全额兑现；条件 (b) 的
  t53 完成状态已核验（任务台账 completed + 分支 commit 在案）。

## 附录二：范围裁决与零改动依据（captain 裁决，供 t13 对照）

- **inScope 仲裁**：captain 裁定 inScope 以已批准契约为准——AGENTS 仲裁
  顺序第一条（已批准且仍适用的预注册契约）优先于任务文本；t12 任务文本
  起草于契约存在之前，属过时描述。实施触碰以下文件是 p19 自身的变更面
  要求，非越界：§1.2（六个成交消费点逐点实测：backtest.py:562、
  bare_factor_backtest.py:101、diagnostics.py:255、eval_search.py:93/391、
  p3_measurement.py:107、train_windows.py:326）、§2.1（data_loader.py 为
  双价格唯一实现地）、§3 范围（上述消费点切换 + 实施落点）、§5 版本表
  （versions.py / targets.py / execution_spec.py 为版本所有者）。
- **零改动核实（有意为之且正确）**：
  - `git diff 6026129..HEAD -- ashare_model/data_tier.py` 为空：Tier A
    时间规则文本（"值在交易日收盘即确定，自次日会话起可用"）与
    p17 §6.1 的 v2 daily 表规则一致，p17 §7 的源级 tier 映射不变，故
    DATA_TIER_VERSION 1 不 bump、文件零改动。
  - `git diff 6026129..HEAD -- ashare_model/evaluation.py` 为空：
    PROTOCOL_VERSION 的单一家园自 IP-07a 起为 ashare_model/versions.py
    （evaluation.py:285 仅重导出），bump "25"→"26" 经重导出自动传播到
    全部消费方（promotion G1 pin、classify_*、runspec）；协议语义输入
    变化（标签基座 + 可交易性来源）由版本常量与消费点切换承载，无需改
    动 evaluation.py 本体。
- **t53 交付硬前置收口**：t53/t55 [completed]；G1-G8 经 adapter
  6/8 PASS（G6/G7 N/A-with-reason 留证）。t12 完成时间早于本裁决送达
  （消息交叉），完成时 t53 数据收口已在库上（dataset_id `43ef1e4c…`）；
  升级审计（上附录）在过门禁的数据上执行 PASS——硬前置的实质条件已满足，
  时序例外如实记录（上附录）。


## 附录三：后置任务登记（t15/D6 收口，captain 代行，p18 §4 回写义务）

- **其余 11 特征切源至 v2 as-of**（captain 收口指令 2026-09-05）：v2 三表
  （income/balance/cashflow）补齐股份/派息/指标列后，PE_TTM、PB、PS_TTM、
  ROE、ROA、GROSS_MARGIN、NET_MARGIN、REVENUE_YOY、PROFIT_YOY、
  DEBT_RATIO、DIVIDEND_YIELD 从 legacy fundamental_pit 切源至 v2
  as-of（经 pit_import.asof_winner_intervals 单一实现）。完成判据 =
  列覆盖审计（逐特征列出所需列与 v2 可用性核对）。与 G6 as-of、
  Tier C→B 外部对账源、G8 成员退出语义并列进入 captain 未来池。
