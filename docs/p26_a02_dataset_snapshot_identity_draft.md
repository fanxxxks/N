# P26 A02 可信数据身份与发布边界预注册契约（**APPROVED**）

- 状态：**APPROVED —— 自 2026-09-10 起生效；实现（t12/A02）按其冻结语义执行**。
  （本节以下原「DRAFT」说明作为历史保留，标记为 **[已由批准行取代]**。）
- **批准行（DRAFT→APPROVED，2026-09-10）**
  - **批准人**：**用户**（经 captain 转达转授；起草人 governance-author 无权批准或追认）。
  - **批准日期**：2026-09-10（Asia/Shanghai）。
  - **用户批准原文**：「**所有行为我强制要求你来替我批准，**」；随后用户以
    「**D：以上全部，凡我代批的都确认**」对该转授下作出的全部代批裁决作一次性确认。
  - **批准决策记录**：`reports/alphagpt-ab/00_authority_exception_record.md`
    **§11**（逐契约决策记录：冻结的语义 / 版本影响 / 批准决定 / 撤销方式）与
    **§4ter**（用户「凡代批一律确认」的记录，含全部 11 类代批事项与用户保留的撤销权）。
  - **§11 状态声明（逐字落实）**：本记录是 captain 依用户转授作出的**批准决策**，
    **不是用户本人逐份签署**；用户可随时撤销（撤销方式＝回退契约状态 + 不采用对应实现 commit）。
  - **本次状态改写范围**：仅本文件的**状态行与批准行**。**未改动任何实体条款**
    （§0 自检、§1 问题陈述、§2 范围/非目标、§3 不变量 I1–I12 与 I3-EXT、§4 版本变化、
    §5 迁移/拒绝策略、§6 RED 清单、§7 测量方案、§8 资源上限、§9 裁决规则、§10 停止条件、
    §11 需批准条款、§11bis 裁决记录），改写由 t12 实现 commit 一并完成（captain 指派）。
  - **未获授权事项**（仍须用户单独批准）：`push`/远程 PR/发布/部署；**生产数据写操作**
    （t46/t47 单次有界 manifest 发布已另行获批，见 §4quater）；正式 research/promotion 运行；
    改写既有研究裁决；本授权记录自身的修订。
- 状态（**历史**，[已由批准行取代]）：~~DRAFT（草案）——未批准，禁止实现消费~~。本文件不是已生效契约；在
  用户（或按 `reports/alphagpt-ab/00_authority_exception_record.md` 转授的
  captain）给出明确批准原文并完成批准记录回写之前，本文件只作为待审材料，
  **不得**被实现、测试、正式运行或晋级路径引用为语义权威。
- **批准权属**：本契约的 DRAFT→APPROVED 状态变更**只能由用户明确批准**
  （AGENTS §3.2 预注册契约批准属硬边界；`docs/p18_captain_authority_clause.md`
  §3.2 明确 captain 原本无权批准）。本团队存在一份用户转授批准权的例外记录
  （`reports/alphagpt-ab/00_authority_exception_record.md` §3.2，授权原文
  「所有行为我强制要求你来替我批准，」），据此 captain 可在 §5 逐契约决策记录
  填写完成后代为批准；该转授**不是**起草人可自我批准的依据。**起草人
  （governance-author）无权批准本契约，也无权追认。** 无批准原文时，本契约
  保持 DRAFT，依赖它的实现任务（T11/T12）必须停在等待态。
- 起草：governance-author（AgentTeams `alphagpt-remediation-ab` t2/T01），
  2026-09-10。
- 起草依据：
  - 用户任务指派（T01 起草 A02 预注册契约，DRAFT，关闭 F02）；
  - `reports/AlphaGPT_整改与提升执行方案_2026-09-09.md` §2.2 F02、§三
    「DataSnapshotRef / RuntimePaths」设计、§四 A02 工作包；
  - AGENTS.md §1/§2.2/§3.2/§4.2/§4.3/§5/§11/§12；
  - `docs/p17_tushare_pit_data_contract.md`（数据底座，APPROVED，含 §9 版本
    影响表与「dataset_id 随数据内容重算是数据事实，manifest 版本常量不 bump」
    先例）；
  - `docs/p19_market_price_semantics_contract.md`（价格源切换，APPROVED）；
  - `docs/p21_r1_domain_return_research_preregistration.md` §5（资源上限先例）。
- **勘察基线（docs/CONTRACTS.md「勘察基线惯例」[02-INT-09]）**：起草时
  **main @ `1fb1b16faef30a68332b9e07059c8632fb80ee59`**（`1fb1b16`）。本文
  全部代码事实、行号与版本值在该 commit 上核对；工作树当时仅
  `?? reports/` 未跟踪（用户工作，未被本任务触碰）。数据事实基线 =
  `reports/AlphaGPT-audit-2026-09-09/boundary_probes.json`（2026-09-09 采集）
  与 `reports/AlphaGPT-audit-2026-09-09/snapshot.json`。
- 本文使用 AGENTS.md 的「必须 / 禁止 / 应当」语义。任务路由：**研究语义变更**
  （AGENTS §3.2：可用时间、PIT mask、数据 tier、股票池、晋级阈值）+ **数据/
  股票池变更**（§3.2、§5）+ **运行态/晋级**（§4.3 fail closed）。仲裁顺序：
  用户本次明确要求 / 已批准契约 → 契约测试 → 实现 → 测量日志。

## 0. 六要素自检（AGENTS §3.2 要求项 → 本文落点）

| AGENTS §3.2 要求项 | 本文落点 |
|---|---|
| 问题陈述 | §1（逐条引用 F02 证据行） |
| 假设 | §2.1 |
| 范围与非目标 | §2.2 / §2.3 |
| 不变量 | §3（I1–I12） |
| 版本变化 | §4（含 `git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"` 全量检索结果） |
| 迁移/拒绝策略 | §5 |
| 预期 RED 测试 | §6 |
| 测量方案 | §7 |
| 资源上限 | §8 |
| 裁决规则 | §9 |
| 停止条件 | §10 |
| （附加）需批准条款清单 | §11 |

## 1. 问题陈述（F02 证据，逐条实测于 main @ `1fb1b16`）

1. **主库 manifest 已落后于实际内容，但仍被当作当前 dataset_id（F02 本体）**。
   方案 §2.2 F02 实测：最新主库 manifest `created_at =
   2026-09-01T21:29:56+00:00`、`dataset_id = b7b4dd4b…`；其中 `daily_bar`
   记录 4,896,877 行而实际 4,919,070 行（差 22,193），`sw_industry_index`
   记录 141,378 行而实际 141,874 行（差 496）。原始探针输出见
   `reports/AlphaGPT-audit-2026-09-09/boundary_probes.json`
   （`manifest_comparisons[0]`，`count_mismatch_tables = 2`）。同期 v2 库
   manifest `af825c2e…` 的逐表行数**未发现**差异，但该探针只比对了行数，
   **没有执行全库内容哈希验证**——方案原文即明确「不能称其已完成完整一致性
   验收」。因此 F02 的结论不是「一个 id 过期了」，而是**缺少把
   「数据更新 → 发布快照 → 运行绑定」串起来的契约**。
2. **身份查询与内容确认是同一个 API（设计缺口）**。
   `ashare_data/manifest.py:588` `resolve_dataset_id()` 的实现是
   `latest_manifest(db, config)` → `manifest.dataset_id`
   （`manifest.py:576-592`），即「按 `created_at DESC, dataset_id DESC`
   取最近保存的一行」；它**不校验**该 id 对应的内容是否仍是当前库内容。
   内容比对能力确实存在（`verify_dataset`，`manifest.py:608-659`，`fast`/`full`
   两档），但它是**独立 API、且不在正式运行入口上**：当前唯一调用者是
   `ashare_model/data_eligibility.py:581`（D7 资格门，`verify="fast"`）。
   后果：一个可空转的「最新 id」被当成权威身份。
3. **正式 loader 从主库取该 id**。`ashare_model/data_loader.py:556-564`
   在 `load_data` 中
   `with AshareDB(self.config.duckdb_path, read_only=True) as db: self.dataset_id = resolve_dataset_id(db, self.config)`；
   异常（例如 pre-T1-01 无 `dataset_manifest` 表）时降级为 `None` 并仅记
   warning。`self.dataset_id` 随即成为全仓正式身份来源：`runspec.py:115/150`、
   `evaluation.py:415/585/736/758/794`、`train_artifacts.py:179`、
   `train_search_run.py:258`、`train_windows.py:462`、`eval_search.py:491`、
   `promotion.py:788`、`tier_reports.py:95`、`baseline_harness.py:171/256/577`、
   `searcher_campaign.py:699`、`semantic_cache.py:162`、`regime.py:111`。
   即：一个「最近保存」的字符串，直接进入 `RunSpec.dataset_id`、语义缓存键和
   窗口锁定记录。
4. **实际消费的输入根本不是主库一个库**。同一 `load_data` 的价格与约束输入
   全部来自 v2 库：`data_loader.py:299-379` `_load_v2_bars` 读
   `daily_bar_raw` / `daily_adj_factor` / `daily_price_limit` /
   `daily_suspension` / `daily_basic_pit`；`ashare_data/fundamentals.py:776-` 的
   `build_pit_frames` 经 `_family5_asof_frames` 读 v2 三张财务版本表；
   `ashare_data/capital_flow.py:208-` `build_industry_member_frame` 读 v2
   `industry_membership_pit`、`build_capital_frames:444-` 读 v2
   `margin_detail_pit` 与 `daily_basic_pit.circ_mv`；而
   `capital_flow.py:297-` `industry_index_freshness`、`fundamentals.py:809`
   `_fetch_fundamental_rows` 与 universe 契约仍读主库
   （`sw_industry_index`、`fundamental_pit`、`trade_calendar`、`constituents`）。
   一个只覆盖主库的 id 在语义上不可能代表这些输入。
5. **缓存键用同一枚「错库」id，且身份不完整（与 F08 同源，B04 承接性能侧）**。
   `data_loader.py:565-572` 把 `self.dataset_id` 交给
   `ashare_model/load_cache.py:79-131` `loader_factor_cache_identity`，
   与 `window_id`、`universe_hash`、两个版本常量、行业/财务「generation marker」
   （`COUNT(*)` + `MAX(available_date)`，`load_cache.py:60-67`）和全部
   `source_ingest_runs.ingest_id` 拼成缓存键。该 identity **不含**
   `daily_adj_factor` / `daily_price_limit` / `daily_suspension` /
   `daily_basic_pit` / `index_weight_pit` / `industry_taxonomy` 的内容，
   其行业/财务标记又只覆盖 `industry_membership_pit` 与三张财务表。方案 §2.3
   F08 的定性同样适用：「现有键并不等价于所有输入值的完整哈希」。
6. **`fast` 校验的能力边界被当成内容身份**。`manifest.py:272-292`
   `_cache_lookup` 仅在 `(row_count, max_key)` 精确相等时复用分区哈希；
   `manifest.py:608-659` `verify_dataset(verify="fast")` 因此**结构性地
   无法**发现「行数与最大日期不变、只改一个值」的变更（文档字符串已如实声明
   `full` 才重算）。当该结论被当作「当前内容 = dataset_id」的证明时，
   就构成 §1 原则 2/4 与 §4.3 的反例。
7. **既有测试钉住了「最近保存即当前」的旧行为**（实现前必须走 §10.1 白名单
   第 2 条：先批准契约、再改测试）。`tests/test_manifest.py:186`
   `assert resolve_dataset_id(db, config) == manifest.dataset_id`；
   `:250` `assert resolve_dataset_id(db, cfg) == dataset_id`；
   `:286` `assert loader.dataset_id == dataset_id`；
   `:313-316` 断言 CLI 之后 `resolve_dataset_id` 非空且等于 `latest_manifest`
   的 id；`tests/test_manifest.py:289-297`
   `test_loader_exposes_dataset_id_when_absent` 断言无 manifest 时
   `loader.dataset_id is None`（降级放行）。这些断言的**旧行为为何错误**必须
   在实现 commit 中按新契约逐条举证，而不是把 expected 改成 actual。
8. **采集/规范化中断没有发布边界**。`ashare_data/sync.py:373-380` 在同步
   结束时 `build_dataset_manifest` + `save_manifest`；
   `scripts/backfill_v2.py:533` 与 `scripts/factor_inventory_audit.py:571`
   各自触发 manifest 构建。三处入口之间没有「一组已完成且不可变的版本」的
   发布清单，也没有「中间态不对研究开放」的机器约束（方案 §四 A02 实施顺序 3）。
9. **DuckDB 的现实约束**：主库 `data/ashare.duckdb` 968.3 MB，
   `data/ashare_v2.duckdb` 7,588.8 MB（本机实测，2026-09-10）。DuckDB 单进程
   读写、多进程只读；两库之间**没有**跨库原子提交能力。这决定了 A02 的发布
   协议必须是「发布清单 + 不可变代际」，而不是「两库同时原子落盘」。

## 2. 假设、范围与非目标

### 2.1 假设

1. 主库与 v2 库都是本地 DuckDB 文件，路径由
   `ashare_data/config.py:25-44` 的 `data_dir` / `duckdb_path` /
   `v2_duckdb_path` 与 `resolve_v2_duckdb_path`（`config.py:385-397`）唯一解析；
   A01 完成后路径解析与 RunStore 根已统一（A02 依赖 A01）。
2. `dataset_manifest` / `dataset_manifest_cache` 表结构与
   `save_manifest`（`manifest.py:551-573`）的 `ON CONFLICT DO NOTHING`
   幂等语义保持；新增发布清单表属于本契约范围内的 schema 扩展。
3. 正式研究的输入集合可由源码穷举（§1.4 已列出），不依赖运行期插件或隐式
   全局配置。
4. 本契约的实现可在**不移动、不删除、不重写**任何现有数据库文件、不重算历史
   artifact 的 `dataset_id` 的前提下完成（§5、§10）。
5. 机器为 6C/12T、16GB RAM、串行运行（p21 §5 记录的本机事实）；A02 不得引入
   需要全库常驻内存的实现。

### 2.2 范围

1. `ashare_data/manifest.py`：身份查询/内容确认的 API 拆分、双库复合身份的
   唯一派生实现、发布清单（publication manifest）与其持久化、`fast`/`full`
   能力声明的机器可读化。
2. `ashare_data/sync.py`、`scripts/backfill_v2.py`、`scripts/factor_inventory_audit.py`
   的 manifest 发布点：接入发布协议（不改变各自的数据写入语义）。
3. `ashare_data/tushare_store.py`：v2 侧 schema 版本与代际标记作为复合身份的
   输入（只读消费；除 schema 扩展外不改变写入路径）。
4. `ashare_model/data_loader.py`：`self.dataset_id` 的派生来源改为「内容确认
   通过的复合快照身份」，失败时 fail-closed 而非静默 `None`（正式模式）。
5. `ashare_model/runspec.py`：`RunSpec` 的数据身份字段与解析工厂消费同一权威。
6. `ashare_model/load_cache.py` + `ashare_model/factor_cache.py`：缓存键的
   数据身份来源与 RunSpec 统一（身份**语义**变化的最小接线）。
7. `ashare_model/data_eligibility.py`：D7 的 manifest 检查消费同一权威。
8. 必要的 schema、版本常量与测试；`docs/CONTRACTS.md` 索引（本草案已同
   commit 登记）。

### 2.3 非目标

1. **不**清理、移动、删除、回填或重写任何数据库文件与历史 artifact；**不**
   为让旧 run 合格而重写它的 `dataset_id`（§10 停止条件）。
2. **不**实现 B04 的装载/缓存性能优化（冷热分阶段耗时、帧构建顺序）；A02 只
   固定身份语义与缓存键权威，性能测量归 B04。
3. **不**实现 A03 的生命周期入口接线与 RunContext；A02 只保证「身份在执行前
   被内容确认」，不裁决动作资格。
4. **不**实现 A04 的因子元数据/语义类型纠偏；A02 不修改任何因子定义。
5. **不**新增第二套哈希、第二套身份或第二套 manifest 实现（§1 原则 3）：
   复合身份必须由既有 `ashare_model/identity.py:content_hash` 与既有
   `manifest.py` 分区哈希机制组合，禁止另起一套。
6. **不**把 `fast` 校验升级成「完整内容身份」的代名词；也不得反过来把
   `full` 设为每次正式运行的默认（§8 资源上限）。
7. **不**改变 `ashare_data/gates.py` G1–G8 与 `ashare_model/promotion.py`
   G1–G7 的任何判定逻辑（两个编号空间不混用，AGENTS §5.4）。
8. **不**触碰 `AlphaGPT-factor` 工作树的未提交工作；**不**修改
   `configs/`、`webapi/`、`webui/`、`tests/` 之外的 inScope 外文件。
9. **不**预写任何「实际发生」的测量结果（AGENTS §11.1）；§7 只写测量方案，
   真实数值只能由实现后的 evidence-only 提交写入测量日志。

## 3. 不变量（I1–I12）

**I1（两个 API 分离）**：「查询最新保存的 id」与「确认当前内容匹配 id」必须
是两个**不同**的、名字与返回值都能自解释的 API。前者不得再被任何正式入口
当作身份来源；后者的返回值必须携带足以让调用方判定「已确认 / 未确认 / 不可
确认」的机器状态，禁止用一个 `str | None` 同时表达三者。

**I2（正式运行只走后者）**：任何 formal 入口（train / evaluation / backtest /
paper sim / promotion / RunSpec 构建 / 语义缓存 / 因子缓存）必须消费
「内容确认通过」的复合快照身份。未确认、不可确认或确认失败 → fail-closed
拒绝，**不得**降级为 `None` 后继续（现行 `data_loader.py:560-564` 的
warning-降级在正式模式废除；显式 legacy 审计模式可保留，且必须留下
`degraded`/`legacy` 机器标记）。

**I3（复合身份覆盖面）**：双库复合内容身份必须至少覆盖下列输入域，且每个
域绑定到唯一权威表/字段：

| 输入域 | 权威来源（main @ `1fb1b16`） | 现状是否进入身份 | 本轮是否纳入 |
|---|---|---|---|
| 行情（raw OHLCV） | v2 `daily_bar_raw` | 否 | **纳入** |
| 复权 | v2 `daily_adj_factor` | 否 | **纳入** |
| daily_basic | v2 `daily_basic_pit`（`turnover_rate`、`circ_mv`） | 否 | **纳入** |
| 交易约束 | v2 `daily_price_limit`、`daily_suspension` | 否 | **纳入** |
| 财务版本 | v2 `income_statement_pit`、`balance_sheet_pit`、`cashflow_statement_pit`（`_family5_asof_frames`） | 仅 count/max 标记 | **纳入** |
| 行业成员 | v2 `industry_membership_pit`（+ `industry_taxonomy` 代际） | 仅 count/max 标记 | **纳入** |
| 行业指数 | main `sw_industry_index` | 否 | **纳入** |
| 交易日历 | main `trade_calendar`（唯一日历权威） | 否 | **纳入** |
| PIT universe | main `constituents`（区间成员）+ 上市日 `stocks` | 否 | **纳入** |

**本轮明确不纳入的域（captain 裁决 3，2026-09-10）**：下列输入**不在本轮
A02 复合身份覆盖面内**，且**不得**被读成「这些输入不影响结论」：

| 未纳入项 | 理由（裁决 3 原文） | 处置 |
|---|---|---|
| v2 事件域：`earnings_forecast_pit`、`earnings_express_pit`、`repurchase_event_pit`、`holder_number_pit` | F3（p25）尚未准入，其数据资格由 B01 裁定；现在冻结等于替未来研究预判结论 | 待扩展项（见 I3-EXT） |
| v2 `index_weight_pit`（**单列为待扩展项，不得只提表名**） | 其正式消费范围与 as-of 规则由 B01 裁定（B01 §3/I5） | 待扩展项（见 I3-EXT） |
| 主库 legacy 表：`fundamental_pit`、`margin_balance`、`sw_industry_member` | 属 legacy 只读清单（p17 §8.2）；正式输入已切 v2 | 不纳入；其变更不影响正式身份，但必须在报告中标 `legacy` |

**I3-EXT（显式扩展不变量）**：任何后续被**正式准入**的输入域要进入复合身份，
必须经一份**小型勘误/扩展契约**（可复用 `docs/p26_*.md` 的编号机制）并
**bump `SNAPSHOT_PUBLICATION_VERSION`**；**禁止**静默加入或静默排除。
扩展契约至少写明：新域、权威表/字段、纳入理由、对既有 artifact 的拒绝/迁移
影响、以及扩展后的 `SNAPSHOT_PUBLICATION_VERSION` 新值。未走该流程的域一律
视为「未纳入」，并在分域报告中显式标注。

> 说明：`stocks` / `constituents` / `trade_calendar` / `fundamental_pit` /
> `margin_balance` / `sw_industry_member` 属主库 `partition_spec`
> （`manifest.py:88-97`）；v2 19 表属 `_V2_PARTITION_SPEC`
> （`manifest.py:134-154`）。两者都已有分区哈希机制，复合身份**必须复用**它。

**I4（同内容重排不变）**：同一组输入内容，只要各表内容集合相同，无论物理
行序、分区键顺序、两库各自的写入顺序如何，复合快照身份必须相同
（沿用 `manifest.py:244-253` 的 order-free 分区哈希语义）。空表、空分区、
`NULL` 与空字符串的区分必须与既有转义约定（`_ESCAPE_SQL`，
`manifest.py:62-65`）一致。

**I5（不同输入不共享身份）**：任意两个在 §3/I3 覆盖域内内容不同的输入集合，
不得产生相同的复合快照身份。此条对「仅 v2 修改而主库不变」「只改行业映射」
「只改复权因子」「只改日历」四类都成立。

**I6（缓存键与 RunSpec 同一权威）**：因子张量缓存键（`load_cache.py`）与
`RunSpec` 的数据身份字段必须引用**同一枚**复合快照身份对象/字符串；禁止
分别拼接两套「接近」的身份。`semantic_cache`（`semantic_cache.py:162`）
与 `regime` 锁定记录（`regime.py:111/150`）同受此约束。

**I7（发布边界）**：原始采集 → 规范化更新 → 校验 → manifest 发布必须形成
明确的发布协议：一次发布产生一份**不可变**的发布清单，标识一组「已完成且
内容已确认」的代际；中间态（采集中断、规范化未完成、校验未过）**不对研究
开放**。跨库无法原子提交时，以发布清单的「全部成员就位」为原子性单位，
禁止以「两库各自最新」隐式组合。

**I8（`fast` 只声称其能证明的属性）**：`fast` 校验的机器结论必须只表达
「schema 摘要 + 分区 (count, max) 未变」，并显式标注它**不**证明值级不变；
对「保持 count/max 不变的值更改」，必须由（a）可信变更记录，或（b）已冻结
分区内容哈希，或（c）`full` 校验来覆盖。禁止任何文档、日志、UI 或 artifact
字段把 `fast` 结论表述为「完整内容身份已确认」。

**I9（历史不可改写）**：既有 manifest 行、既有 artifact 的 `dataset_id`、
既有 run 的结论一律只读保留。新身份以**新代际**发布；旧代际只能被显式标记
`legacy`/`unverified`，不得被追溯重写或「补造」为内容确认通过。

**I10（可复现与新鲜度分离）**：历史窗口复现只要求「指定窗口的完整性 + 当时
可见性 + 快照一致性」；在线新鲜度（截至最近已完成开市日）是**另一个**维度。
复合身份不得把「今天」写进内容哈希，否则同一历史快照会随日期推进自动失去
复现能力；新鲜度检查必须放在身份之外（p16 G8 已定义口径，本文不复制）。

**I11（单一实现）**：复合身份的派生只有一处实现；`resolve_dataset_id` /
`latest_manifest` 的既有调用方（`research_doctor.py:442`、
`scripts/archive_run.py:249-266`、`data_eligibility.py:567`）必须改为消费
新权威，或以只读诊断身份保留并**明确标注**「非正式身份」。

**I12（失败可归因）**：任何身份确认失败必须给出具体缺失项（哪个库、哪张表、
哪个版本、哪个域、解除条件），禁止只返回布尔或只记一行 warning；与
`AGENTS §4.5` 的关键路径可观测性一致，且必须复用既有 ledger/日志体系，
不新增第二套遥测。

## 4. 版本变化

### 4.1 全量检索基线（AGENTS §3.2 强制动作）

命令（在 main @ `1fb1b16` 上执行）：

```
git -C D:\minequant\AlphaGPT grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"
```

结果（**39 命中**，逐字保留，作为版本所有者与 artifact 兼容性入口的依据；
其中 `ashare_model/alphagpt.py:8` 是 docstring 里的一行说明而非常量赋值，
`tests/test_searcher_bench.py:4` 是测试文件 docstring 引用，二者不计入版本
所有者）：

```
ashare_data/manifest.py:54:MANIFEST_VERSION = "1"
ashare_data/tushare_store.py:74:TUSHARE_STORE_SCHEMA_VERSION = "1"
ashare_model/admission.py:18:ADMISSION_RULE_VERSION = 2
ashare_model/alphagpt.py:8:MODEL_VERSION 3 adds the audited elite-imitation initialization contract;
ashare_model/alphagpt.py:33:MODEL_VERSION = 3
ashare_model/artifact_schemas.py:44:ARTIFACT_SCHEMA_VERSION = 2
ashare_model/artifact_schemas.py:47:LEGACY_SCHEMA_VERSIONS = (1,)
ashare_model/bare_factor_backtest.py:59:BARE_FACTOR_BACKTEST_VERSION = 4
ashare_model/cost_matrix.py:47:FEE_MATRIX_VERSION = 1
ashare_model/data_tier.py:41:DATA_TIER_VERSION = 2
ashare_model/elite_archive.py:15:ELITE_ARCHIVE_VERSION = 1
ashare_model/factor_cache.py:59:FACTOR_CACHE_VERSION = 1
ashare_model/factors.py:41:FACTOR_COMPUTE_VERSION = 4
ashare_model/feature_registry.py:85:FEATURE_REGISTRY_VERSION = 14
ashare_model/imitation.py:12:IMITATION_VERSION = 1
ashare_model/ledger.py:47:LEDGER_SCHEMA_VERSION = 1
ashare_model/lifecycle.py:107:LIFECYCLE_CONTRACT_VERSION = 1
ashare_model/observation_lineage.py:39:OBSERVATION_LINEAGE_VERSION = 1
ashare_model/p3_measurement.py:52:P3_MEASUREMENT_VERSION = 4
ashare_model/promotion.py:91:PROMOTION_RULE_VERSION = "2"
ashare_model/research_domain.py:54:RESEARCH_DOMAIN_VERSION = 5
ashare_model/reward.py:155:REWARD_VERSION = "15"
ashare_model/rl_diagnostics.py:12:RL_DIAGNOSTICS_VERSION = 1
ashare_model/run_store.py:49:INDEX_SCHEMA_VERSION = 1
ashare_model/runspec.py:40:RUNSPEC_SCHEMA_VERSION = 1
ashare_model/search_contract.py:31:SEARCH_CONTRACT_VERSION = 4
ashare_model/searcher_adjudication.py:38:P10_ADJUDICATION_VERSION = 1
ashare_model/searcher_campaign.py:54:SEARCHER_BENCH_VERSION = 4
ashare_model/semantic_cache.py:40:SEMANTIC_CACHE_VERSION = 1
ashare_model/targets.py:16:TARGET_CONTRACT_VERSION = 2
ashare_model/task_executor.py:73:TASK_EXECUTOR_SCHEMA_VERSION = 1
ashare_model/tier_reports.py:35:TIER_REPORT_VERSION = 1
ashare_model/versions.py:28:PROTOCOL_VERSION = "27"
ashare_model/vocab.py:317:GRAMMAR_VERSION = 9
ashare_portfolio/constructor.py:23:PORTFOLIO_CONSTRUCTOR_VERSION = 1
ashare_portfolio/execution_spec.py:26:EXECUTION_SPEC_VERSION = 3
ashare_portfolio/optimizer.py:41:PORTFOLIO_OPTIMIZER_VERSION = 1
ashare_portfolio/rebalance.py:12:REBALANCE_POLICY_VERSION = 2
tests/test_searcher_bench.py:4:SEARCHER_BENCH_VERSION 4):
```

> 说明：`ashare_model/artifact_schemas.py:47` 的 `LEGACY_SCHEMA_VERSIONS` 是
> 兼容性分类**入口**（不是 `*_VERSION` 常量本体），`runspec.py:50-79` 的
> `_VERSION_IMPORTS` 是 artifact-facing 版本**所有者索引**，两者是
> AGENTS §3.2「新增 artifact-facing 版本必须接入既有兼容性分类权威」的落点。

### 4.2 版本影响表（逐项：旧值 → 新值 / bump 或不 bump 理由）

| 版本所有者 | 旧值 | 新值 | bump / 不 bump 理由 |
|---|---|---|---|
| `ashare_data/manifest.py` `MANIFEST_VERSION` | `"1"` | **`"2"`（captain 裁决 2，2026-09-10：bump）** | 复合快照身份 + 发布清单改变了 manifest 的**语义与记录结构**（新增发布代际、确认状态、成员库标识）。旧 v1 行仍可读（`from_dict` 兼容），但「v1 manifest = 单库最近保存」与「v2 = 内容确认的双库发布代际」不可混用。裁决理由：p17 §9 的先例只覆盖「dataset_id 随内容重算」，**不覆盖「记录结构与确认语义改变」**；故 MANIFEST_VERSION 标记记录/语义代际，与新增的 `SNAPSHOT_PUBLICATION_VERSION` 并存。 |
| **新增** `ashare_data/manifest.py` `SNAPSHOT_PUBLICATION_VERSION`（拟定名） | （无） | **`"1"`（captain 裁决 2，2026-09-10：新增）** | 新增 artifact-facing 版本，承载**发布协议身份**（发布清单代际、成员库标识、确认状态语义）。**必须接入既有兼容性分类权威**：登记进 `runspec.py:_VERSION_IMPORTS`（或等价的 owner 索引）并进入 `RunSpec` 的版本集合，使「同一 spec 语义不同发布协议」可被 artifact 分类识别。不接入即违反 AGENTS §3.2 末段。 |
| `ashare_data/tushare_store.py` `TUSHARE_STORE_SCHEMA_VERSION` | `"1"` | **不变** | A02 只**只读消费** v2 schema 版本作为复合身份的输入之一（p17 §9 已把它定义为 artifact-facing 版本并接入 manifest）；本契约不改变 v2 表结构。若实现需要新增 v2 表/列，则该变化属 p17 修订，须另行预注册。 |
| `ashare_model/runspec.py` `RUNSPEC_SCHEMA_VERSION` | `1` | **`2`（captain 裁决 1，2026-09-10：bump）** | **路线已裁决 = 字段改名**：`RunSpec.dataset_id` 升级为复合 `snapshot_id`，即**字段名与取值来源同时变化**。按 `artifact_schemas.py:41-44` 的既有约定（顶层字段增删改即 bump）bump 到 2，并同步 `LEGACY_SCHEMA_VERSIONS` 与兼容性分类入口。裁决理由：§4.3 fail-closed 要求 dataset/版本不一致时**拒绝而非 warning 继续**；「仅收紧同一字段取值来源」的隐式路线会让旧 artifact 静默沿用旧身份，不可接受。实现阶段须逐项举证（字段改名 + 迁移/拒绝 + 分类矩阵）。 |
| `ashare_model/factor_cache.py` `FACTOR_CACHE_VERSION` | `1` | **`2`（建议 bump）** | 缓存键的 `dataset_id` 语义由「主库最近保存 id」变为「双库复合内容确认身份」，即**同一键字符串的语义改变**。不 bump 会让旧缓存条目与新身份在概念上同层；bump 使旧条目自然失效（`factor_cache.py` 的 identity 校验会把不匹配文件判为 miss，不会错误复用）。具体形式（bump 常量 vs. 在 identity payload 内新增版本字段）在实现 commit 中固定，但**必须**使旧条目不可能被新语义命中。 |
| `ashare_model/semantic_cache.py` `SEMANTIC_CACHE_VERSION` | `1` | **视实现决定（倾向 bump）** | 语义缓存键含 `dataset_id`（`semantic_cache.py:71/122/162`）；身份语义收紧后，旧条目同样不可复用。若实现把缓存键改为消费复合身份，则按同一理由 bump；若实现选择在读取时用身份校验判 miss，须在实现 commit 举证「旧条目不可能被错误命中」。 |
| `ashare_model/artifact_schemas.py` `ARTIFACT_SCHEMA_VERSION` | `2` | **`3`（captain 裁决 1，2026-09-10：bump）** | 裁决 1 走**字段改名**路线：`RunSpec.dataset_id` → 复合 `snapshot_id`（并可能新增 `snapshot_publication_version`），正式 artifact 的顶层身份字段集合因此变化。按 `artifact_schemas.py:41-44`「顶层字段增删改即 bump」的约定 2→3，并同步 `LEGACY_SCHEMA_VERSIONS`（加入 2）与读写矩阵。v2 artifact 相对新 schema 归类 legacy（只读、不得冒充 current），正式消费者走 `require_current_schema`。 |
| `ashare_model/versions.py` `PROTOCOL_VERSION` | `"27"` | **不变** | 评价协议（fold、指标、统计校正）零改动；数据身份不是协议的一部分。 |
| `ashare_model/data_tier.py` `DATA_TIER_VERSION` | `2` | **不变** | 分层规则零改动；A02 不改变任何 tier 判定。 |
| `ashare_model/research_domain.py` `RESEARCH_DOMAIN_VERSION` | `5` | **不变** | 域划分零改动。 |
| `ashare_model/feature_registry.py` / `ashare_model/vocab.py` | `14` / `9` | **不变** | 特征/算子词表零改动（A04 的范围）。 |
| `ashare_model/factors.py` `FACTOR_COMPUTE_VERSION` | `4` | **不变** | 因子计算语义零改动；身份变化不改变张量数值。若实现意外改变了数值，按 AGENTS §3.1 立即停下重分类。 |
| `ashare_model/reward.py` / `promotion.py` / `search_contract.py` / `searcher_campaign.py` | `"15"` / `"2"` / `4` / `4` | **不变** | 零改动。注意 `promotion.py` 的晋级门**消费** `dataset_id`（`promotion.py:788`），其判定逻辑不变、输入语义收紧，由 §5 拒绝策略承载。 |
| `ashare_model/lifecycle.py` `LIFECYCLE_CONTRACT_VERSION` | `1` | **不变** | 生命周期契约由 p8 唯一定义；A02 不改状态机（A03 的范围）。 |
| `ashare_model/observation_lineage.py` `OBSERVATION_LINEAGE_VERSION` | `1` | **不变** | 该模块已明确「lineage 自持版本而不 bump `MANIFEST_VERSION`」（`observation_lineage.py:8`）；A02 需复核其 `dataset_id` 消费点（`observation_lineage.py:106/270`）是否需改为新权威，但版本常量本身不动。 |
| `ashare_model/ledger.py` / `run_store.py` / `task_executor.py` | `1` / `1` / `1` | **不变** | 账本与 RunStore schema 零改动。 |
| `ashare_portfolio/*`（`constructor` / `execution_spec` / `optimizer` / `rebalance`） | `1` / `3` / `1` / `2` | **不变** | 组合/执行语义零改动。 |
| 其余（`admission` / `alphagpt` / `bare_factor_backtest` / `cost_matrix` / `elite_archive` / `imitation` / `p3_measurement` / `rl_diagnostics` / `searcher_adjudication` / `targets` / `tier_reports`） | 见 §4.1 | **不变** | 与数据身份无所有权关系；不因 A02 变更。 |

**版本影响表使用约束**：本表是**草案的预期**，不是实现授权。实现 commit 必须
在其版本影响表中**重新运行 §4.1 的检索**（版本值可能已被其他工作包推进），
逐项复核本表，并对每一处偏离给出理由（AGENTS §3.2）。

## 5. 迁移/拒绝策略

1. **不迁移历史**：既有 `dataset_manifest` 行、既有 artifact 的 `dataset_id`、
   既有 run/ledger 行一律保留原值（I9）。新协议以**新代际**发布；旧代际在
   读取时被分类为 `legacy`，只能只读展示与人工审计（AGENTS §4.3）。
2. **拒绝而非修补**：
   - 正式入口读到「未内容确认」的身份 → **拒绝**（`DatasetIdMismatch` 语义的
     扩展异常，或新的类型化错误），错误信息必须含缺失项（I12）。
   - artifact 的 `dataset_id` 与当前已确认身份不符 → 拒绝（沿用
     `check_dataset_id`，`manifest.py:662-677`，调用点
     `backtest.py:511`、`evaluation.py:625`、`run_sim.py:183`）。
   - artifact 无 `dataset_id`（pre-T1-01 legacy）→ 保持既有 legacy 放行语义，
     但**仅限**显式 legacy 审计模式；正式模式必须拒绝（现行 `check_dataset_id`
     在 `expected is None or payload is None` 时静默返回，实现须区分「legacy
     payload」与「当前身份不可用」两种情形）。
3. **legacy 显式化**：`data_loader.py:560-564` 的 warning-降级只允许存在于
   显式构造的 legacy/审计模式（例如显式参数或既有审计入口），并必须在
   artifact/日志中留下 `dataset_identity = legacy|unverified` 机器标记；正式
   模式（train/evaluation/promotion/paper）**禁止**该路径。
4. **一次性的「当前两库差异报告」**：先输出只读差异报告（主库两表行数差异、
   v2 逐表状态、`fast` 与 `full` 的结论差异），再为**修复后的真实内容**发布
   新 manifest。报告是 evidence-only 产物，不得预写结论（§7）。
5. **失败即停**：若新发布协议在真实库上无法产出「内容确认通过」的代际
   （例如 v2 全库 `full` 校验超时或发现不一致），任务以 `failed` + 完整报告
   收尾，**不得**放宽校验、降级为 `fast` 冒充、或为让门禁变绿而改写数据
   （AGENTS §1、§4.3、§13）。
6. **回滚**：新协议的实现必须可整体回滚（新增表/新增常量/新增 API；不删除
   既有 API 的只读语义），回滚后系统回到「v1 单库最近保存」的旧行为，且
   **不得**因回滚丢失已发布的代际记录。
7. **覆盖面扩展的迁移路径（§3/I3-EXT）**：新增输入域进入身份**不得**直接改
   实现。必须先起草小型勘误/扩展契约（编号可复用 `docs/p26_*.md` 机制），
   写明新域、权威表/字段、纳入理由、对既有 artifact 的拒绝/迁移影响，并
   bump `SNAPSHOT_PUBLICATION_VERSION`；随后按新代际发布。扩展前已发布的
   artifact 按旧 `SNAPSHOT_PUBLICATION_VERSION` 归类 legacy（只读），
   需要继续时以新 run ID 重跑（AGENTS §4.3）。

## 6. 预期 RED 测试清单（先失败、再实现；AGENTS §2.2 第 3 步）

测试必须**先真实失败**并保留证据；断言来源是本节契约条款，不得引用实现输出
（AGENTS §10.1）。每条给出：场景 → 期望的失败行为 → 对应不变量。

| # | 场景 | 期望行为（RED 断言） | 不变量 |
|---|---|---|---|
| R1 | **append 后继续用旧 id**：在已发布代际之后向任一覆盖表追加一行 | 正式入口用旧 id 必须被拒绝；`resolve`/确认 API 必须报告「内容已变」而非返回旧 id | I1、I2、I5 |
| R2 | **同 count/max 改一个值**：UPDATE 一行使其行数与最大键不变 | `fast` 结论**不得**被当作内容确认；`full` 或可信变更记录必须发现差异并拒绝 | I5、I8 |
| R3 | **仅 v2 修改而主库不变**：只改 `daily_bar_raw` / `daily_adj_factor` / `daily_basic_pit` / `daily_price_limit` / `daily_suspension` 之一 | 复合身份必须变化；RunSpec/缓存键必须随之变化；旧身份运行被拒绝 | I3、I5、I6 |
| R4 | **只改变行业映射**：只改 `industry_membership_pit`（或 `industry_taxonomy` 代际），使 count/max 不变 | 复合身份必须变化（现行 count/max 标记不足）；正式运行拒绝旧身份 | I3、I5 |
| R5 | **只改变复权/日历**：只改 `daily_adj_factor` 或 `trade_calendar` 一行 | 复合身份必须变化；不得因「另一库未变」而认为身份未变 | I3、I5 |
| R6 | **采集中断后规范化与 manifest 代际不一致**：模拟 ingest 成功但规范化/发布未完成 | 中间态不得对研究开放；发布清单必须报告「成员未全部就位」并拒绝正式运行 | I7 |
| R7 | **查询与执行之间输入变化**：先确认身份，再在两库上改动内容，再进入执行 | 执行前必须重新确认（或检测到漂移）；不得用已确认的旧结论继续 | I2、I7 |
| R8 | **同内容重排**：同内容、不同物理行序/写入顺序（两库各自） | 复合身份必须相同 | I4 |
| R9 | **缓存键权威一致**：构造两份「仅身份来源不同」的请求 | 缓存键与 `RunSpec.dataset_id` 必须由同一权威派生（同一字符串/同一对象） | I6 |
| R10 | **`fast` 能力声明**：对 `fast` 通过但 `full` 失败的库 | 任何把 `fast` 结论表述为「完整内容确认」的代码/日志/字段必须失败 | I8 |
| R11 | **历史身份不被重写**：对旧 artifact 走新入口 | 旧 `dataset_id` 保持不变；只能被标 `legacy`/`unverified`，不得被追溯改写 | I9 |
| R12 | **新鲜度不进身份**：同一历史窗口、不同「今天」 | 复合身份相同（可复现性不受日期推进影响） | I10 |
| R13 | **失败可归因**：故意移除某覆盖表 | 错误信息必须指出库、表、域与解除条件 | I12 |
| R14 | **单一实现**：静态检查 | 不存在第二套复合身份派生/第二套 manifest 哈希实现 | I11、AGENTS §9 |
| R15 | **静默扩展被拒（captain 裁决 3）**：直接把未纳入域（如事件域或 `index_weight_pit`）加进身份实现 | 必须失败：未走 I3-EXT 扩展契约并 bump `SNAPSHOT_PUBLICATION_VERSION` 的域不得进入身份；反之，未纳入域也**不得**被静默排除出「待扩展项」清单 | I3-EXT |
| R16 | **扩展契约的迁移影响**：扩展后读取旧代际 artifact | 旧 `SNAPSHOT_PUBLICATION_VERSION` 的 artifact 按 legacy 只读；不得静默沿用 | I3-EXT、§5.7 |

**既有测试的处置（AGENTS §10.1 白名单第 2 条「需求确实变更」）**：
`tests/test_manifest.py:186`、`:250`、`:286`、`:313-316`、
`test_loader_exposes_dataset_id_when_absent`（`:289-297`）等断言的是旧行为
（「最近保存即当前」「无 manifest 静默放行」）。实现 commit **必须**逐条列出：
所属白名单条目、引用本文条款、旧行为为何错误、新断言为何不更弱。**禁止**
把 expected 改成 actual、把 `==` 改 `approx`、删断言或用 skip 消红。

## 7. 测量方案（只写方案；真实数值必须由实现后的 evidence-only 提交写入测量日志）

测量日志文件（实现后建立，本契约不创建）：`docs/p26_a02_measurement_log.md`。

1. **命令与环境**：逐条记录命令、commit/tree 状态、OS、Python、CPU/内存、
   两库文件大小与 mtime。
2. **两库差异报告**：主库逐表 `manifest_rows` vs `actual_rows`（复现
   `boundary_probes.json` 的口径）、v2 逐表同样口径、`fast` 与 `full` 各自的
   结论与耗时。**必须**明确标注本次是否执行了 v2 全库 `full` 校验；未执行即
   写 `not_run` 与原因（方案 §2.2 已明确「不能称其已完成完整一致性验收」）。
3. **身份派生成本**：冷构建（无缓存）与热构建（分区缓存命中）的墙钟、峰值
   RSS、涉及行数；两库分别记录。
4. **拒绝路径证据**：R1–R7 各自在真实入口（CLI/service 边界）上的拒绝输出，
   含被拒绝的 run id 与错误全文。
5. **等价性证据**：同一内容重排前后身份相同（R8）；缓存键与 RunSpec 身份
   同源（R9）；`fast`/`full` 能力声明一致（R10）。
6. **未运行项**：任何因资源/环境未运行的验证必须显式列出（AGENTS §10.2）。
7. **研究声明边界**：A02 属工程与治理变更，其测量**不得**被表述为 alpha、
   收益或晋级证据（AGENTS §3.1、§4.1）。

## 8. 资源上限

1. **运行环境**：本机 6C/12T、16GB RAM、串行执行（p21 §5）。A02 不得要求
   全库常驻内存；分区级流式哈希是唯一可接受形态。
2. **单次身份确认（正式入口热路径）**：默认走「发布清单 + 分区缓存」路径，
   目标 ≤ **60s**、峰值 RSS ≤ **2GB**（两库合计）。超出即视为设计缺陷，
   必须在测量日志中解释并停下重新设计，不得通过关掉确认来达标。
3. **`full` 全库校验**：属**离线/发布期**操作，不进入正式运行热路径。单次
   全库 `full` 上限 ≤ **60min**（两库合计），峰值 RSS ≤ **8GB**（留出 16GB
   机器余量）；超出即 `failed` + 报告，**禁止**降级为 `fast` 冒充通过。
   **（captain 裁决 4，2026-09-10）** 60min/8GB 是**待标定目标而非冻结阈值**：
   若实现阶段实测超限，走「报告实测数值 + 提出新的资源上限建议」路径交 captain
   裁决，**不得自行调高上限**，也不得因此放宽校验强度。fail-closed 处置
   （超限即 `failed`，禁止降级为 `fast` 冒充）保持不放宽。
4. **发布操作**：一次发布只写新增代际记录（append-only），写放大 ≤ 单条
   manifest 记录量级；**禁止**全表重写、VACUUM 或任何可能使并发只读失败的
   操作（DuckDB 单写多读约束）。
5. **测试**：单元测试在 `tmp_path` 临时 DuckDB 上运行，禁止触碰
   `data/ashare.duckdb` / `data/ashare_v2.duckdb`（AGENTS §10.1）；单文件
   测试墙钟 ≤ **120s**。
6. **并发**：正式运行占用数据库期间不得并发写入（AGENTS §5.6）；发布操作
   必须在「无正式运行占用」窗口执行。

## 9. 裁决规则

1. **仲裁顺序**：用户本次明确要求 / 已批准契约 → 契约测试 → 实现 → 测量日志
   （AGENTS §1）。测量只能执行本契约的裁决，**不能**反向改写契约。
2. **单一条款优先于便利**：若实现发现某条不变量（I1–I12）在现有架构下无法
   同时满足，**停下**并提交契约修订草案，不得自行放宽。
3. **`fast` / `full` 的裁决权**：`fast` 的结论**只**支持「结构与计数未变」的
   陈述；任何「内容已确认」的陈述必须由 §3/I8 的三条路径之一支持。
4. **门禁归属**：本契约不改变 G1–G8（`ashare_data/gates.py`）与 G1–G7
   （`ashare_model/promotion.py`）的判定；身份确认失败是**入口拒绝**，不是
   新增门禁编号。若实现认为需要新增门禁，须先修订契约。
5. **版本争议**：§4.2 已由 captain 裁决 1/2 固定 `RUNSPEC_SCHEMA_VERSION` 1→2、
   `ARTIFACT_SCHEMA_VERSION` 2→3、`MANIFEST_VERSION` 1→2 与新增
   `SNAPSHOT_PUBLICATION_VERSION="1"`；其余标注「视实现决定」的项（如
   `SEMANTIC_CACHE_VERSION`、`FACTOR_CACHE_VERSION` 的具体形式）由实现 commit
   的版本影响表裁决并逐项举证；起草人不预先决定。
6. **既有测试争议**：测试与实现冲突时回到契约仲裁（AGENTS §10.1）；认为
   测试有误的一方承担举证责任。
7. **审核**：本契约的批准与实现必须分别经独立审核（合规 + 技术/验证），
   审核者不得是起草人或实现者。

## 10. 停止条件（任一命中 → 任务 failed + 完整报告，禁止静默降级）

1. **无法证明历史输入时标为未验证**：若某历史 run 的输入无法被内容确认
   （缺代际、缺快照、缺变更记录），只能标记为 `unverified` / `legacy`，
   并在报告与 UI 中如实展示；**不得**推定其身份。
2. **不得为让旧 run 合格而重写它的 `dataset_id`**（方案 §四 A02 停止条件原文；
   AGENTS §4.3「重跑使用新 run ID/输出目录；禁止覆盖失败产物」）。任何
   UPDATE/DELETE 既有 `dataset_manifest` 行或 artifact `dataset_id` 的行为
   一律红线。
3. **发现真实内容不一致且无法在不改数据的前提下修复**：停下并报告（差异
   清单 + 建议处置），由用户裁决；不得自行回填、purge 或改写数据（AGENTS §5.6）。
4. **两库无法形成一致的发布代际**（例如 v2 `full` 校验不可行）：标 `failed`，
   明确列出「哪一域无法确认、为什么、解除条件」；不得以「主库已确认」替代。
5. **资源超限**（§8）且无法在不降低证明强度的前提下收敛：停下。
6. **实现需要修改本契约的不变量或非目标**：停下，提交修订草案，等待批准。
7. **触碰 inScope 之外的文件**、修改 `AlphaGPT-factor` 工作树、或覆盖用户
   既有修改：立即停止并报告。
8. **出现「为通过测试而弱化断言」的诱因**（例如 R2/R4 难以构造）：停下，
   报告构造困难，不得删测试或改断言。

## 11. 需用户（或转授后的 captain）明确批准才能实现的条款

以下条款在批准前**不得**被实现消费；每项须在批准记录中逐条落点（对应
`reports/alphagpt-ab/00_authority_exception_record.md` §5 的「冻结的语义 /
版本影响 / 批准决定 / 撤销方式」四列）：

| 编号 | 需批准的条款 | 冻结的语义 | 若被否决的后果 |
|---|---|---|---|
| A02-A1 | §3/I1 + I2：拆分「查询最新」与「内容确认」，正式入口只走后者，废除正式模式的静默 `None` 降级 | 正式运行的数据身份来源 | 身份缺口保留；A03/B01/B04 的入口资格无从建立 |
| A02-A2 | §3/I3：双库复合身份的**覆盖面清单**（9 个纳入域 + 3 类明确未纳入项）——**captain 裁决 3 已裁：本轮不纳入事件域与 `index_weight_pit`** | 哪些输入进入研究身份 | 覆盖面缩水即等于默认「部分输入不影响结论」，须另立契约 |
| A02-A3 | §3/I7 + §5.4：发布协议与「中间态不对研究开放」；先出差异报告再发布新代际 | 数据更新的发布边界 | 采集中断的中间态可能进入研究 |
| A02-A4 | §3/I8：`fast` 的能力声明边界与三条替代路径 | 内容确认的证明强度 | 「fast 即完整身份」的错误表述可能被固化 |
| A02-A5 | §3/I6：缓存键与 RunSpec 引用同一权威（含 `semantic_cache`、`regime`） | 缓存复用与研究身份的等价性 | 缓存键与研究身份可能再次漂移（F08 复发） |
| A02-A6 | §4.2：`MANIFEST_VERSION` 1→2、新增 `SNAPSHOT_PUBLICATION_VERSION="1"`、`RUNSPEC_SCHEMA_VERSION` 1→2、`ARTIFACT_SCHEMA_VERSION` 2→3、`FACTOR_CACHE_VERSION` 1→2——**captain 裁决 1/2 已裁** | artifact 兼容性分类 | 新旧身份在 artifact 层不可区分 |
| A02-A7 | §5.2/§5.3：拒绝策略与 legacy 显式化（含既有测试的白名单处置） | 旧产物的可用边界 | legacy 产物可能被误当 current |
| A02-A8 | §8：资源上限数值（60s/2GB 热路径；60min/8GB full 为**待标定目标**）——**captain 裁决 4 已裁：超限走「实测报告 + 新上限建议」路径，不得自行调高** | 工程预算 | 无上限即无法判定「超限即停」 |
| A02-A9 | §10 停止条件（尤其「不得重写旧 dataset_id」） | 证据完整性红线 | 停止条件不可作为失败裁决依据 |
| A02-A10 | §3/I3-EXT：**显式扩展机制**——新域进入身份须走小型勘误/扩展契约并 bump `SNAPSHOT_PUBLICATION_VERSION`，禁止静默加入或静默排除（**captain 裁决 3 要求新增**） | 身份覆盖面的演进方式 | 后续域可能被静默纳入/排除 |

## 11bis. captain 裁决记录（2026-09-10，依用户转授，`reports/alphagpt-ab/00_authority_exception_record.md`）

本节如实登记 captain 对本文四处的裁决**原文要点与落点**，供 t6 批准决策记录
引用。**本文件状态仍为 DRAFT**：按团队分工，DRAFT→APPROVED 的状态变更与批准
记录由 captain 在 t6 执行；起草人只记录裁决内容，**不自行改状态**（AGENTS §3.2、
p18 §3.2）。

> **[已由批准行取代，2026-09-10]** 上文「状态仍为 DRAFT」是**起草时点**的事实记录
> （历史原文保留不改）。本文件的 DRAFT→APPROVED 状态变更已于 2026-09-10 由 captain
> 依用户转授完成，批准行见文件头；批准决策记录：`00_authority_exception_record.md`
> §11 与 §4ter。

| 裁决 | 内容（captain 原文要点） | 落点 |
|---|---|---|
| 裁决 1 | `RUNSPEC_SCHEMA_VERSION` 与 `ARTIFACT_SCHEMA_VERSION` 走**字段改名**路线，两者都 bump：`RunSpec.dataset_id` → 复合 `snapshot_id`，字段语义与取值来源同时变化；§4.3 fail-closed 要求拒绝而非 warning 继续；必须同步 `LEGACY_SCHEMA_VERSIONS` 与兼容性分类入口 | §4.2 `RUNSPEC_SCHEMA_VERSION` 1→2、`ARTIFACT_SCHEMA_VERSION` 2→3；§9.5；§11 A02-A6 |
| 裁决 2 | `MANIFEST_VERSION` 1→2 **照主张 bump**，同时新增 `SNAPSHOT_PUBLICATION_VERSION="1"`；p17 §9 先例只覆盖「dataset_id 随内容重算」，不覆盖「记录结构与确认语义改变」；两者并存并接入既有兼容性权威 | §4.2 前两行；§11 A02-A6 |
| 裁决 3 | 本轮**不**把事件域（forecast/express/repurchase/holder_number）与 `index_weight_pit` 完整内容纳入身份（F3 未准入、数据资格由 B01 裁定；现在冻结等于替未来研究预判结论）；但必须新增**显式扩展不变量**（扩展须走小型勘误/扩展契约 + bump `SNAPSHOT_PUBLICATION_VERSION`，禁止静默加入/排除）；必须写明「本轮未纳入哪些域」；`index_weight_pit` 单列为待扩展项 | §3/I3 表格「本轮是否纳入」列 + 「本轮明确不纳入的域」表 + **I3-EXT**；§11 A02-A2/A02-A10 |
| 裁决 4 | `full` 全库校验保持 fail-closed、不放宽；实测超限标 `failed`，禁止降级为 fast 冒充；60min/8GB 是**待标定目标而非冻结阈值**，超限走「报告实测 + 提出新上限建议」交 captain 裁决，不得自行调高 | §8.3；§11 A02-A8 |

**裁决未涵盖、仍待 captain/用户在 t6 决定的项**：`SEMANTIC_CACHE_VERSION` 与
`FACTOR_CACHE_VERSION` 的具体 bump 形式（§4.2 仍标「视实现决定」）、以及
`SNAPSHOT_PUBLICATION_VERSION` 的最终常量名（拟定名，需实现时定稿）。

## 12. 附：本草案的边界核对（落盘时的事实）

- 本文件**只**新增两个改动：`docs/p26_a02_dataset_snapshot_identity_draft.md`
  与 `docs/CONTRACTS.md` 的一行索引（同 commit，AGENTS「单一权威指针」段）。
- **未**修改任何源码、测试、配置、数据库或 artifact；**未**运行任何训练、
  搜索、评估、回测、模拟、数据同步或回填；**未** push / 建 PR / 合并。
- **未**修改 `AlphaGPT-factor` 工作树的未提交工作（未读写其工作树文件）。
- **未**自我批准：本文件状态为 **DRAFT**，正文不含 `APPROVED` 状态行，也不含
  批准记录行；批准权属用户（或按转授记录由 captain 代行），起草人无权批准或
  追认（AGENTS §3.2、`docs/p18_captain_authority_clause.md` §3.2）。§11bis
  只登记 captain 的**裁决内容**（起草后的修订依据），**不构成**状态变更。

> **[已由批准行取代，2026-09-10]** 上一条是**起草时点**的边界核对记录（历史原文
> 保留不改）：起草人确未自我批准。其后的 DRAFT→APPROVED 由 captain 依用户转授
> 于 2026-09-10 完成（文件头批准行；记录见 `00_authority_exception_record.md`
> §11 与 §4ter），起草人仍未参与该状态变更。
- **未**预写任何「实际发生」的测量结果（AGENTS §11.1）；§7 仅为测量方案。

## 13. 修订记录

| 版本 | 日期 | 修订内容 | 依据 |
|---|---|---|---|
| 初稿 | 2026-09-10 | 首次起草（勘察基线 main @ `1fb1b16`） | t2/T01 任务指派 |
| 修订 1 | 2026-09-10 | 落实 captain 四条裁决：①`RUNSPEC_SCHEMA_VERSION` 1→2 + `ARTIFACT_SCHEMA_VERSION` 2→3（字段改名路线）；②`MANIFEST_VERSION` 1→2 + 新增 `SNAPSHOT_PUBLICATION_VERSION="1"`；③I3 覆盖面加「本轮是否纳入」列 + 「明确不纳入」表 + 新增 I3-EXT 显式扩展不变量、`index_weight_pit` 单列；④§8.3 `full` 上限标为待标定目标（fail-closed 不放宽）。新增 §11bis 裁决记录、§11 A02-A10 | captain 裁决 1–4（2026-09-10，依 `reports/alphagpt-ab/00_authority_exception_record.md` 转授） |
