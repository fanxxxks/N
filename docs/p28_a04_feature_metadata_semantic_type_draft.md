# P28 A04 因子元数据与语义类型纠偏预注册契约

- 状态：**APPROVED — 批准记录回写（实现包 A04 按本文件头分工条款执行）**。
  批准人：captain（依用户转授权原文「所有行为我强制要求你来替我批准，」，
  见 `reports/alphagpt-ab/00_authority_exception_record.md` §1）；批准决策
  记录：同记录 **§11（t6）七条争议点裁决第 2 条**（A04 语义类型二选一 =
  `CROSS_SECTIONAL_SIGNAL`、`GRAMMAR_VERSION` 不 bump、`FEATURE_REGISTRY_VERSION`
  14→15 必须、缓存版本 bump 授权）及 **第 8 条**（A04 按当前主仓 owner 常量
  实现、不预判 F3 合并）与 **§11 版本影响总表**；用户确认：同记录 **§4ter
  「凡我代批的都确认」**。本次实现另受用户**直接指令**（2026-09-14 会话：
  「阅读AlphaGPT_整改与提升执行方案_2026-09-09.md，完成A04」——用户本次明确
  要求，AGENTS §1 仲裁顺序第一位）。用户可随时按 §11「批准效力与撤销」行使
  撤销权。
- **批准权属**：本契约的 DRAFT→APPROVED 状态变更**只能由用户明确批准**
  （AGENTS §3.2：特征/算子定义与可用时间属硬边界；
  `docs/p18_captain_authority_clause.md` §3.2 明确 captain 原本无权批准）。
  本团队存在用户转授批准权的例外记录
  （`reports/alphagpt-ab/00_authority_exception_record.md` §3.2，授权原文
  「所有行为我强制要求你来替我批准，」），据此 captain 可在 §5 逐契约决策记录
  填写完成后代为批准；该转授**不是**起草人可自我批准的依据。**起草人
  （governance-author）无权批准本契约，也无权追认。** 无批准原文时本契约保持
  DRAFT，依赖它的实现任务（T18/T19）必须停在等待态。
- 起草：governance-author（AgentTeams `alphagpt-remediation-ab` t4/T03），
  2026-09-10。
- 起草依据：
  - 用户任务指派（T03 起草 A04 预注册契约，DRAFT，关闭 F04，含 FO-01 存量
    契约清单范围）；
  - `reports/AlphaGPT_整改与提升执行方案_2026-09-09.md` §2.2 F04、§四 A04
    工作包；
  - `reports/AlphaGPT_因子与算子开发补充方案_2026-09-10.md` §5 接线契约、
    §6 FO-01 工作包、§8 顺序关系；
  - AGENTS.md §1/§3.2/§6/§9/§10.1/§11/§13；
  - `docs/p24_f2_margin_crowding_draft.md`（F2 批次预注册，APPROVED）、
    `docs/p23_f1_slow_fundamental_quality_draft.md`（F1 批次，APPROVED）、
    `docs/p17_tushare_pit_data_contract.md` §5.2/§6.1/§6.2（数据底座与 PIT
    可见性）；
  - `docs/p7_semantic_types_contract.md`（语义类型契约）。
- **勘察基线（docs/CONTRACTS.md「勘察基线惯例」[02-INT-09]）**：起草时
  **main @ `1fb1b16faef30a68332b9e07059c8632fb80ee59`**（`1fb1b16`）。
  全部代码事实与计数在该 commit 上实测（见 §1.9 的实测脚本）。**只读评审**的
  `AlphaGPT-factor` 工作树当时位于 `codex/factor-semantic-fixes` @ `0a2ecee`，
  有 8 个已修改文件 + 1 个未跟踪文件 `ashare_data/events.py`（全部为 **F3
  新增**，见 §2.1 假设 4）；**本任务未修改该工作树的任何文件**。
- 本文使用 AGENTS.md 的「必须 / 禁止 / 应当」语义。任务路由：**研究语义变更**
  （AGENTS §3.2：特征定义、可用时间、grammar/规范化、搜索空间）+ **数据/
  股票池变更**（§5）+ **运行态/晋级**（§4.3）。仲裁顺序：用户本次明确要求 /
  已批准契约 → 契约测试 → 实现 → 测量日志。

## 0. 六要素自检（AGENTS §3.2 要求项 → 本文落点）

| AGENTS §3.2 要求项 | 本文落点 |
|---|---|
| 问题陈述 | §1（逐条引用 F04 证据行） |
| 假设 | §2.1 |
| 范围与非目标 | §2.2 / §2.3 |
| 不变量 | §3（I1–I12） |
| 版本变化 | §4（含 `git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"` 全量检索结果） |
| 迁移/拒绝策略 | §5 |
| 预期 RED 测试 | §6（R1–R14） |
| 测量方案 | §7 |
| 资源上限 | §8 |
| 裁决规则 | §9 |
| 停止条件 | §10 |
| （附加）需批准条款清单 | §11 |

## 1. 问题陈述（F04 证据，逐条实测于 main @ `1fb1b16`）

1. **两个 F2 融资因子的 `availability_rule` 与 `semantic_type` 同其真实数据
   来源不符（F04 本体）**。`ashare_model/feature_metadata.py:470-477`
   `MARGIN_BALANCE_TO_CIRC_MV` 与 `:478-485`（`"MARGIN_BUY_PRESSURE_5"` 条目）
   `MARGIN_BUY_PRESSURE_5` 均使用
   `availability_rule=_FUNDAMENTAL_TTM`（`feature_metadata.py:79-83`：
   「quarterly report, visible after the statutory disclosure season ends
   （Q1 04-30 / H1 08-31 / Q3 10-31 / annual next-year 04-30）；TTM ratios
   require four complete consecutive quarters」）与
   `semantic_type=SemanticType.FUNDAMENTAL_LIKE`
   （`feature_metadata.py:48`：slow report/scale characteristics），而
   `feature_metadata.py:464-469` 的**紧邻注释**自述真实来源：
   「the raw ratios computed in capital_flow (ashare_data) with the per-date
   cross-sectional z applied loader-side… The margin detail visibility = T+1
   (p17 §6.1); circ_mv = the trade-date same-session convention」。
   方案 §2.2 F04 原文：「availability_rule 使用季度财务 TTM 模板，
   semantic_type 为 FUNDAMENTAL_LIKE；相邻注释及其实际数据来源却是融资明细
   T+1 加 daily_basic 当日口径。」
2. **同一模块内的兄弟因子已经采用正确口径（可对照的既有先例）**：
   `MARGIN_BALANCE_CHG`（`feature_metadata.py:377-388`）与 `MARGIN_CROWD_60`
   （`:749-756`）均使用 `_MARGIN`
   （`feature_metadata.py:84-87`：「v2 margin_detail_pit (p17 §5.2 table 18):
   the detail of session t-1 becomes visible at session t (p17 §6.1 T+1
   availability)」）与 `semantic_type=SemanticType.VOLUME_LIKE`
   （`feature_metadata.py:47`）。即：**同族、同源、同可见性的四个因子，存在
   两套互相漂移的元数据口径**（AGENTS §1 原则 3 的具体违反）。
3. **`semantic_type` 是搜索层的活语义，不是文案**。`ashare_model/gp_search.py:118`
   `pset.addTerminal(name, _type_class(FEATURE_METADATA[name].semantic_type))`
   把每个特征登记为强类型终端的**类型**；
   `ashare_model/semantic_sampling.py:56-66` `_feature_type(name)` 同样读
   `FEATURE_METADATA[name].semantic_type`（`semantic_sampling.py:58` 即该
   读取行），并驱动 `type_id`
   （`semantic_sampling.py:69-74`）的掩码/状态层。方案 §2.2 F04 原文：
   「GP 和 semantic_sampling 会消费该 semantic_type，因此不能仅按文案问题处理。」
4. **类型选择会实际改变合法表达式空间（实测，见 §1.9）**。
   `ashare_model/operator_registry.py:38-40` 定义 `_SIX`（全部六型）、
   `_CONT`（除 `BOOLEAN_EVENT_SIGNAL` 的五型）与 `_XS = (CROSS_SECTIONAL_SIGNAL,)`；
   全量遍历 39 个算子后，**唯一**限定输入为单一类型的算子是
   `CS_NEUTRALIZE`（`operator_registry.py:224` 输入 `(_XS,)`）。因此：
   把 F2 从 `FUNDAMENTAL_LIKE` 改为 `CROSS_SECTIONAL_SIGNAL` 会让
   `CS_NEUTRALIZE` **对这两个特征从不可用变为可用**——这是一次真实的搜索空间
   变化，必须在契约中预注册、并在测量中给出合法动作集合的前后对照。
   （反之若改为 `VOLUME_LIKE`，则与 `MARGIN_BALANCE_CHG`/`MARGIN_CROWD_60`
   同型，但不解锁 `CS_NEUTRALIZE`。）
5. **F2 的输出确实是截面标准化量**。`ashare_model/data_loader.py:525-528`
   在装载时对这两个帧施加 `apply_cross_sectional_z(capital,
   ("MARGIN_BALANCE_TO_CIRC_MV", "MARGIN_BUY_PRESSURE_5"))`；该函数的文档字符串
   （`data_loader.py:515-524`）明确「F2 (p24 §2): the margin-crowding factors
   are pre-registered as the per-date cross-sectional z of the raw ratios」。
   方案 §2.2 F04 原文：「究竟应归入 CROSS_SECTIONAL_SIGNAL 还是 VOLUME_LIKE，
   应依据最终输出是否已经截面标准化及算子类型契约确定。」——本文 §3/I3 给出
   判定规则，把最终裁决留给批准人（§11 A04-A2）。
6. **`depends_on` 的口径也不一致（同一批因子的第二处错配）**。F2 声明
   `depends_on=("rzye", "circ_mv")`（`:476`）与 `("rzmre", "circ_mv")`
   （`:484`）——**字段名**；而 `MARGIN_CROWD_60` 声明
   `depends_on=("margin_detail_pit",)`（`:755`）——**表名**，且
   `tests/test_factor_semantic_fixes.py:288` 把 `("margin_detail_pit",)` 钉成
   断言。`ashare_model/feature_registry.py:414` 与
   `scripts/generate_registry_docs.py:98` 直接消费该字段并写入生成文档
   `docs/feature_registry.md`；FO-01 要求「每个可采样对象的单位、时间、类型、
   NaN、依赖可核对」，因此**依赖项的单位口径必须先统一**，否则清单无法核对。
7. **D7 资格门并未错误路由（重要更正，避免过度修复）**。方案 §2.2 F04 原文：
   「D7 目前把两者识别为 margin 来源，本次没有证实它错误路由到财务资格门。
   应修复确定的元数据问题。」实测确认：`ashare_model/feature_registry.py:130-164`
   的 `_family_of`/`_pit_level_of` 先查 `FACTOR_REGISTRY`，再查
   `FUNDAMENTAL_PIT_NAMES`（`ashare_data/fundamentals.py:49-69`，19 项，**不含**
   这两个 F2 因子），再查 `EXTERNAL_FACTOR_NAMES`
   （`ashare_data/capital_flow.py:41-49`，**含**这两个因子）→ 返回
   `PitLevel.PIT_CAPITAL` → `ashare_model/data_eligibility.py:166-167` 映射到
   `SOURCE_MARGIN`。**结论：D7 的 margin 路由正确，A04 不得以「修 D7」为名
   改动资格门**（§2.3 非目标 4）。
8. **存量清单不存在（FO-01 的缺口）**。当前没有任何机器可核对的「存量因子/
   算子契约清单」：`docs/feature_registry.md` 是**生成文档**（由
   `scripts/generate_registry_docs.py` 从注册表生成），但不含「单位」与
   「NaN 语义」列；`AuthoredFeatureMeta`（`feature_metadata.py:95-107`）只有
   `availability_rule / hypothesis / expected_direction / semantic_type /
   promotion_allowed / compute_cost / depends_on`——**没有单位字段、没有
   NaN 语义字段**。补充方案 §6 FO-01 要求「每个可采样对象的单位、时间、类型、
   NaN、依赖可核对」，因此 FO-01 的交付物必须是**可从单一权威派生的清单**，
   不得手工维护第二份名单（AGENTS §6 末段、§11.5）。
9. **实测计数基线（本机，main @ `1fb1b16`，只读脚本）**：
   - 特征槽位 **83**，deprecated **13**，可采样 **70**（与方案 §2.1 表格一致）；
   - 算子 **39**；
   - 语义类型直方图（全部 83）：`fundamental_like` 23、`return_like` 19、
     `volume_like` 17、`cross_sectional_signal` 10、`boolean_event_signal` 9、
     `price_like` 5；可采样 70 的分布：23 / 14 / 11 / 10 / 8 / 4；
   - 四个 margin 因子的实测状态：
     `MARGIN_BALANCE_TO_CIRC_MV`（type=`fundamental_like`、avail=季度 TTM、
     deps=`('rzye','circ_mv')`、pit=`pit_capital`、family=`external`）、
     `MARGIN_BUY_PRESSURE_5`（同上，deps=`('rzmre','circ_mv')`）、
     `MARGIN_BALANCE_CHG`（type=`volume_like`、avail=`v2 margin_detail_pit…`、
     deps=`('margin_detail_pit',)`）、`MARGIN_CROWD_60`（同前）。
   > 上述计数是**起草时实测**，实现前必须重新测量（版本与词表可能已被其他
   > 工作包推进，AGENTS §11.3）。

## 2. 假设、范围与非目标

### 2.1 假设

1. **A02/A03 的批准与实现是 A04 的正式依赖**（方案 §四依赖表：A04 依赖「无；
   合并前复核 WIP」，B01 依赖 A02/A04；A05 依赖 A01–A04）。A04 本身可在
   A02/A03 之前起草与实现元数据修复，但**其测量（搜索空间影响）必须在
   A02/A03 的正式入口可用后进行**。
2. **`FEATURE_METADATA` / `FACTOR_REGISTRY` / `OPERATOR_REGISTRY` 是既有的
   单一注册权威**；A04 只修正其内容与派生口径，不新建第二份注册表。
3. **F2 批次的族级裁决（p24）不被本次纠偏推翻**：p24 已批准的是「两个因子
   进入候选池」；本次纠偏只改元数据口径与语义类型，不改变因子数值、公式、
   PIT 可见性或族级裁决（方案 §四 A04 验收原文：「不以某成员弱结果自动废弃
   既有族级裁决」）。
4. **`AlphaGPT-factor` 工作树的 WIP 与本任务不重叠，但**版本起点已分叉**（只读评审，
   8 个 M + 1 个未跟踪，逐文件判定见下表）。评审基线：分支
   `codex/factor-semantic-fixes` @ `0a2ecee`。

**§2.1.4 WIP 逐文件判定（captain t1 确认的 8 文件清单；只读评审，未修改）**

| # | 文件 | WIP 内容（只读实测） | 对 A04 的判定 |
|---|---|---|---|
| 1 | `ashare_data/pit_import.py`（+85） | 新增 `_EVENT_TABLES` / `_EVENT_VALUE_COLUMNS` 与事件 as-of 机制（三张事件表共用 `(ts_code, end_date)` 业务键，winner = max by `(ann_date, available_date, ingest_id)`） | **可复用**：这是 FO-02 的事件 as-of 机制；A04 的 `depends_on` 口径统一（§3/I4）应与其「单一 as-of 位置」纪律一致，但 **A04 不依赖它**（A04 只改元数据）。 |
| 2 | `ashare_domain.py`（+28） | 新增 F3 四因子到 `medium_cross_section`、新建 `event` 域（`_EVENT_FEATURES = ("EXPRESS_VS_FORECAST",)`） | **无冲突**：纯新增域/成员；A04 非目标 7 禁止在 `event` 域内顺手改。 |
| 3 | `ashare_model/feature_metadata.py`（+37） | 新增四个 F3 因子条目（`FORECAST_GROWTH_MIDPOINT` / `FORECAST_UNCERTAINTY` / `EXPRESS_VS_FORECAST` / `BUYBACK_INTENSITY`），全部 `availability_rule=_FUNDAMENTAL_TTM`、`semantic_type=FUNDAMENTAL_LIKE` | **需注意（非冲突但同模式）**：这四个 F3 条目**复用了同一个季度 TTM 模板**，而它们的真实来源是公告日驱动的事件表（`ann_date` + 公告后可见）。**A04 的 I1/I2 纠偏模式对它们同样适用**——但它们是 F3 的批准范围（p25），**A04 不得顺手改**；应作为 FO-02 的输入记录（见 §2.3 非目标 5 与 §10.6）。 |
| 4 | `ashare_model/feature_registry.py`（±4） | `FEATURE_REGISTRY_VERSION = 14` → **`15`**（注释标为 p25 F3 批次） | **版本起点分叉（必须处理）**：A04 §4.2 的旧值 `14` 是 **main @ 1fb1b16** 的值；若 F3 WIP 先合并，A04 的 bump 目标应基于 **15**（即 15→16）。这正是 AGENTS §3.2「实现前必须重新运行全量检索」的实例；实现 commit 必须重测并重述本表。 |
| 5 | `ashare_model/research_domain.py`（±2） | `RESEARCH_DOMAIN_VERSION = 5` → **`6`** | **版本起点分叉**：A04 §4.2 的旧值 `5` 同源问题；实现前重测。 |
| 6 | `ashare_model/vocab.py`（±17） | 新增 `_FEATURE_NAMES_V11`（四个 F3 因子）→ 槽位 83→**87**；`GRAMMAR_VERSION = 9` → **`10`**；`resolve_formula_tokens` 的已知代际白名单 2..9 → 2..10 | **版本起点分叉 + 词表位移**：A04 §1.9 的实测计数（83 槽位/70 可采样/39 算子）与 §4.2 的 `GRAMMAR_VERSION=9` 均为 **main @ 1fb1b16** 值；F3 合并后须重测（槽位 87、可采样数随之变化）。**I7「词表不变」约束的是 A04 自身不得改词表**，不阻止 F3 的既有批准变更。 |
| 7 | `docs/feature_registry.md`（±11） | 生成文档同步：版本行 14/5 → 15/6，特征数 83 → **87**，新增 4 行 F3 条目与 `event` 域行 | **可复用（生成物纪律先例）**：F3 已经做了「注册表 + 生成文档同 commit」——A04 的 FO-01 清单（§3/I5）必须沿用同一纪律；但 A04 的清单**新增单位/NaN 列**，是 F3 未覆盖的字段。 |
| 8 | `docs/p25_f3_expectation_event_draft.md`（±2） | 状态行由 DRAFT 改为 **APPROVED & IMPLEMENTED**（用户原文「批准」+ captain 代行签署，引 89dd121 授权扩展） | **无冲突**：p25 的批准状态与 A04 无关；**不得**据此推定 A04 可自行改状态（A04 的 DRAFT 状态由本团队 t6 走批准流程）。 |
| — | `ashare_data/events.py`（未跟踪，新建） | F3 四因子的帧构建；全部读经 `ashare_data.pit_import` 的单一 as-of 机制；公告日当天不可见（`available_date` 键控） | **可复用（B01 相关）**：其「公告日+1 可见」语义与 B01 §3/I3 的四时间语义一致；但 A04 不消费它。 |

**判定结论**：A04 与 F3 WIP **无文件级冲突**（F3 只新增，未触及 `MARGIN_*`）；F3 WIP
侧存在三处版本推进（`FEATURE_REGISTRY_VERSION` 14→15、`RESEARCH_DOMAIN_VERSION`
5→6、`GRAMMAR_VERSION` 9→10）与一处计数位移（槽位 83→87）。

**captain 裁决 8（2026-09-10）：A04 按当前主仓 owner 常量实现，不预判 F3 合并。**
- `FEATURE_REGISTRY_VERSION` **14 → 15（必须 bump）**；
- `RESEARCH_DOMAIN_VERSION` **维持 5**；
- `GRAMMAR_VERSION` **维持 9**；
- 槽位/可采样/算子计数**按当前 83 实测**（§1.9）；
- F3（p25）尚未准入，且事件域已裁决不纳入 A02 身份，故**不得**把 F3 的版本起点
  当作 A04 的实现前提。
- **合并时对账**：若 F3 先合并，A04 的 bump 目标与三项计数**在合并时重新对账**
  （从新的 main 起点重测并重述 §4.2）；若 A04 先合并，F3 必须在其版本表中承接
  14→15 的既有 bump（不得回退）。两者都不得依赖对方未提交的工作树状态（AGENTS §12）。

因此 A04 的实现 commit **必须**：重新运行 §4.1 的全量检索、重新测量 §1.9 的计数、
并在版本影响表中重述每一处偏离的理由（AGENTS §3.2、§11.3）。
5. **`_FUNDAMENTAL_TTM` 等可用性模板文本是**描述性**声明**，不是机器判据；
   A04 的目标是让声明与真实来源一致，并让清单可核对。
6. 机器 6C/12T、16GB RAM、串行；A04 的测量不需要重跑搜索，只需构造合法
   动作集合的前后对照与小型固定面板。

### 2.2 范围

1. `ashare_model/feature_metadata.py`：两个 F2 因子的 `availability_rule`、
   `semantic_type`、`depends_on`；以及为 FO-01 可核对性所需的**单位/NaN**
   结构化声明（若需要新增字段）。
2. `ashare_model/feature_registry.py`：派生口径（`depends_on`/`pit_level`/
   `family` 的单一来源）与 FO-01 清单的机器输出。
3. `ashare_model/semantic_sampling.py`、`ashare_model/gp_search.py`：仅在
   类型变更引起的合法动作集合差异需要断言时改动（原则上不改逻辑）。
4. `ashare_domain.py`：F2 因子的域归属核对（`ashare_domain.py:166`）。
5. `scripts/generate_registry_docs.py` 与生成文档 `docs/feature_registry.md`：
   FO-01 清单的生成与 drift guard（同 commit 更新，AGENTS §11.5）。
6. `tests/`：新增 FO-01 清单守卫与 F2 类型纠偏的 RED 测试（新测试文件必须
   同 commit 注册进 `scripts/check_test_shards.py` 分片，否则 fail-closed 红）。
7. `docs/CONTRACTS.md` 索引（本草案已同 commit 登记）。
8. （修订 3 补列，对齐 §4.2 版本表 owner 文件——合规审核 C-F-1）
   `ashare_model/semantic_cache.py`：`SEMANTIC_CACHE_VERSION` 与
   `SemanticCacheKey` 的键布局属 §4.2 该行「视实现决定」分支（把
   `FEATURE_REGISTRY_VERSION` 纳入键 = 键布局变化即 bump），并经 t6 裁决 2
   明示缓存 bump 授权；改动限于键字段、版本常量与惰性导入，不触碰缓存
   语义/预算语义。

### 2.3 非目标

1. **不**改变任何因子的**数值/公式/计算实现**：F2 的比率计算
   （`ashare_data/capital_flow.py:444-543`）与 loader 侧 xcs-z
   （`data_loader.py:525-528`）零改动。方案 §四 A04 原文：「仅修说明与真正
   改变数值/类型应分开提交。」
2. **不**改变 p24/p23/p13 的族级批准结论，**不**重新裁决任何方向/有效性。
3. **不**新增、删除或重排任何特征槽位/算子 token：`FEATURE_NAMES` 长度与顺序、
   token id、`DEPRECATED_FEATURE_NAMES` 集合零改动（`tests/test_feature_registry.py:276-288`
   的 append-only 钉值必须继续成立）。
4. **不**修改 D7 资格门（`data_eligibility.py`）的判定逻辑或来源映射
   （§1.7 已实测路由正确）。
5. **不**实现 FO-02–FO-08（事件规范化、新候选因子、算子原型、完整接线、
   消融、收益验证）——它们是独立工作包，且 FO-08 必须等待 R01 基线。
6. **不**建立第二份手工因子名单；清单必须从注册表生成（AGENTS §6、§11.5）。
7. **不**把 `AlphaGPT-factor` 的 F3 WIP 纳入本次提交，也不修改其工作树。
8. **不**改动 `ashare_data/gates.py` G1–G8 / `promotion.py` G1–G7 判定。
9. **不**预写任何「实际发生」的测量结果（AGENTS §11.1）。

## 3. 不变量（I1–I12）

**I1（元数据与真实来源一致）**：每个特征的 `availability_rule`、
`semantic_type`、`depends_on` 必须与其**实际计算来源**一致：来源表/字段、
可见性（T+1 / 当日收盘 / 公告后 / 季末披露季）、单位、NaN 语义必须能被独立
核对。声明与实现冲突时，以已批准的数据契约（p17 §5.2/§6.1/§6.2、p19 §2）为
仲裁依据，不以实现现状反推。

**I2（F2 两因子的可见性表达联合规则）**：两个 F2 因子的
`availability_rule` 必须显式表达**联合可见性**：融资明细为 T+1
（`margin_detail_pit`，p17 §6.1），`circ_mv` 为**当日常规口径**
（`daily_basic_pit`，D4 时代无 `available_date`，见
`ashare_data/capital_flow.py:471-479` 的注释）。禁止继续使用季末披露季模板；
也禁止只表达其中一半。

**I3（语义类型判定规则，二选一必须可举证）**：
- 若最终输出**已经截面标准化**（当前是：loader 侧 `apply_cross_sectional_z`），
  则类型必须为 `CROSS_SECTIONAL_SIGNAL`（「only meaningful vs peers」，
  `feature_metadata.py:49`）；
- 若输出是**未标准化的水平量/流量量**，则类型为 `VOLUME_LIKE`（与
  `MARGIN_BALANCE_CHG`/`MARGIN_CROWD_60` 同型）。
判定必须写明依据（输出量纲 + 是否 xcs-z + 算子类型契约），并给出合法动作
集合的前后对照（§7.3）。**该判定属研究语义变更，须批准**（§11 A04-A2）。

**I4（依赖项口径统一）**：`depends_on` 在同一注册表内必须口径一致——要么统一
为**表名**（如 `margin_detail_pit`），要么统一为**表.字段**二元组。选择必须
一次性固定并写进契约；禁止同一批因子混用两种口径。既有
`tests/test_factor_semantic_fixes.py:288` 的 `("margin_detail_pit",)` 与
`tests/test_feature_metadata.py:143` 的 `spec.required_columns` 派生断言构成
兼容性约束，改动必须逐条举证（§6 末段）。

**I5（FO-01 清单：机器可核对的存量契约清单）**：必须交付一份**从注册表派生**
的清单，覆盖**全部 70 个可采样特征**与**全部 39 个算子**，每个对象至少包含：
名称、token id、家族/域、PIT level、Tier、`semantic_type`、`availability_rule`
（含可见性类别）、单位、NaN 语义、`depends_on`、`promotion_allowed`、
deprecated 状态。清单必须是**生成物**并有 drift guard（生成命令 + 同 commit
更新），禁止手工维护第二份（AGENTS §6 末段、§11.5）。**不重复建立登记权威**：
清单只是既有注册表的可读投影。

**I6（单位与 NaN 必须可核对）**：每个可采样特征必须声明单位（如
`ratio`/`pct`/`yuan`/`wan_yuan`/`shares`/`z_score`/`boolean`）与 NaN 语义
（缺失时是 NaN 还是中性 0；哪些情形允许中性化）。当前 `AuthoredFeatureMeta`
没有这两个字段，因此允许**新增字段**（属 artifact-facing schema 变化，
按 §4.2 处理）；但不得用自由文本代替结构化声明。

**I7（词表与 token 不变）**：`FEATURE_NAMES` 的长度、顺序、token id 与
`DEPRECATED_FEATURE_NAMES` 集合零改动；`GRAMMAR_VERSION` 不因本次纠偏而
bump（除非批准人裁决类型变更构成 grammar 语义变化——见 §4.2）。

**I8（搜索后端一致）**：GP / Random / TPE / RL 四个后端对修正后的类型必须
产生**一致的合法动作集合**（同一 `FORMULA_VOCAB`、同一
`OPERATOR_REGISTRY`、同一类型约束）。任一后端出现「某表达式合法、另一后端
非法」即失败。

**I9（数值不变）**：纠偏不得改变任何特征的数值。若实现导致数值变化，按
AGENTS §3.1 立即停下并重新分类（工程变更 vs 语义变更）。

**I10（族级裁决不被推翻）**：本次纠偏不改变 p24 的族级批准结论，也不把
「候选池准入」升级为「单因子有效」或「可晋级」。证据状态必须可区分
（候选允许 / 族级通过 / 单因子有效性 / 可晋级——方案 §四 A04 验收原文）。

**I11（单一实现）**：类型/单位/依赖/可见性的解析只有一处实现
（`feature_metadata` → `feature_registry` 派生），D7、生成文档、计算函数不得
各自维护字符串映射（补充方案 §5 原文）。

**I12（未来数据不改变过去）**：元数据纠偏是**面向未来**的：它改变从批准后
开始的搜索空间与资格判定；不得追溯重写既有 artifact、既有测量日志或既有
结论（AGENTS §4.3、§11.4）。

## 4. 版本变化

### 4.1 全量检索基线（AGENTS §3.2 强制动作）

命令（在 main @ `1fb1b16` 上执行）：

```
git -C D:\minequant\AlphaGPT grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"
```

结果：**39 命中**（38 个常量赋值 + 2 处 docstring 引用）。逐字清单见
`docs/p26_a02_dataset_snapshot_identity_draft.md` §4.1（**同一次检索**，此处
不复制第二份，AGENTS §11.5）。实现前必须**重新运行**该命令。

### 4.2 版本影响表（逐项：旧值 → 新值 / bump 或不 bump 理由）

| 版本所有者 | 旧值 | 新值 | bump / 不 bump 理由 |
|---|---|---|---|
| `ashare_model/feature_registry.py` `FEATURE_REGISTRY_VERSION` | `14` | **`15`（必须 bump；captain 裁决 8，2026-09-10：14→15 确认）** | 语义类型与可用性规则是注册表记录的语义字段；F2 的类型/可见性/依赖口径变化改变了「注册表描述的特征语义」。方案 §四 A04 版本节原文：「类型变化会改变可搜索表达式，必须评估 grammar/feature registry/semantic cache 及旧公式兼容性。」bump 后由 `RunSpec` 版本集合与 artifact 分类承载代际区分。裁决 8 确认本项按当前主仓 owner 常量 14→15 执行，**不预判 F3**。 |
| `ashare_model/vocab.py` `GRAMMAR_VERSION` | `9` | **维持 `9`（captain 裁决 8，2026-09-10：已裁决，不再是待决项）** | 本次不新增/删除 token、不改变词表顺序与长度（83 槽位）、不改变规范化规则（I7），故词表语义未变。**裁决 8 明确**：A04 按当前主仓 owner 常量实现，`GRAMMAR_VERSION` 维持 9；「特征的语义类型是否构成 grammar 语义」不再作为 bump 依据（原 A04-A4 的待裁决二选一已由裁决 8 关闭）。若实现意外改动了词表顺序/长度或规范化，按 §10.6 停下另立契约。 |
| `ashare_model/factors.py` `FACTOR_COMPUTE_VERSION` | `4` | **不变** | 数值计算零改动（I9）。若实现意外改变数值 → 停下重新分类（AGENTS §3.1）。 |
| `ashare_model/semantic_cache.py` `SEMANTIC_CACHE_VERSION` | `1` | **视实现决定（倾向 bump）** | 语义缓存的键含 `dataset_id`、reward/protocol 版本与窗口（`semantic_cache.py:11/71/162`）；类型/注册表版本变化会改变「同一公式是否语义等价」的判定基础。若实现把 `FEATURE_REGISTRY_VERSION` 纳入键或校验，则按同一理由 bump；否则须在实现 commit 举证「旧条目不可能被错误命中」。 |
| `ashare_model/factor_cache.py` `FACTOR_CACHE_VERSION` | `1` | **不变** | 缓存键已含 `feature_registry_version`（`load_cache.py:127`），因此 `FEATURE_REGISTRY_VERSION` bump 会自动使旧张量条目失配（identity 校验判 miss）。除非 A02 一并改变键语义，否则本包不 bump。 |
| `ashare_model/data_tier.py` `DATA_TIER_VERSION` | `2` | **不变** | `tier_of_pit_level` 的映射机制零改动；F2 的 PIT level 实测已是 `PIT_CAPITAL`（§1.7），纠偏不改变 tier。若实现改变了任何特征的 tier，必须停下重新分类并在此表登记。 |
| `ashare_model/research_domain.py` `RESEARCH_DOMAIN_VERSION` | `5` | **维持 `5`（captain 裁决 8，2026-09-10：维持 5）** | F2 的域归属（`ashare_domain.py:166`）不在本次纠偏范围；裁决 8 确认维持 5。若 FO-01 清单核对发现域归属与真实语义冲突，须**另立契约**修订，不在本包内顺手改。 |
| `ashare_model/artifact_schemas.py` `ARTIFACT_SCHEMA_VERSION` | `3`（修订 3 勘误：base 实测 3，起草表的 `2` 已被 A02 消费） | **不变（A04 未触碰）** | 若为 I6 新增结构化「单位/NaN」字段并写入 artifact 顶层，则按 `artifact_schemas.py:41-44` 约定 bump 并同步 `LEGACY_SCHEMA_VERSIONS`；A04 实现只存在于注册表/生成文档，不写入 artifact 顶层，故不 bump（决策与勘误前的表意一致）。 |
| `ashare_model/runspec.py` `RUNSPEC_SCHEMA_VERSION` | `2`（修订 3 勘误：base 实测 2，起草表的 `1` 已被 A02 消费） | **不变** | RunSpec 已通过 `feature_registry_version` 间接承载注册表代际；A04 未新增顶层字段（如显式 `feature_metadata_version`），不 bump。 |
| `ashare_model/versions.py` `PROTOCOL_VERSION` | `"27"` | **不变** | 评价协议（fold、指标、统计校正）零改动。 |
| `ashare_model/reward.py` `REWARD_VERSION` | `"15"` | **不变** | Reward 语义零改动。 |
| `ashare_model/promotion.py` `PROMOTION_RULE_VERSION` | `"2"` | **不变** | 晋级门判定零改动；F2 的 `promotion_allowed` 保持 `True`（p24 已批准候选池准入）。 |
| `ashare_model/search_contract.py` / `searcher_campaign.py` | `4` / `4` | **不变** | 搜索协议与公平性参数零改动；类型变化对合法动作集合的影响由 §7.3 测量，不改变预算/seed 口径。 |
| `ashare_model/operator_registry.py` 无版本常量 | — | — | 算子语义零改动（不新增/删除算子）；算子注册表随 `FEATURE_REGISTRY_VERSION` 一并承载（`runspec.py:40-49` 的注释先例）。 |
| 其余 26 个 `*_VERSION`（见 §4.1 指针） | 见 p26 §4.1 | **不变** | 与特征/算子元数据无所有权关系；不因 A04 变更。 |

**版本影响表使用约束**：本表是**草案的预期**，不是实现授权。实现 commit 必须
重新运行 §4.1 检索、逐项复核本表，并对每一处偏离给出理由（AGENTS §3.2）。

**已知版本起点与 F3 的关系（captain 裁决 8，2026-09-10）**：`AlphaGPT-factor` WIP
（`codex/factor-semantic-fixes` @ `0a2ecee`）已把 `FEATURE_REGISTRY_VERSION` 14→15、
`RESEARCH_DOMAIN_VERSION` 5→6、`GRAMMAR_VERSION` 9→10。本表的「旧值」列是
**main @ `1fb1b16`** 的值。**裁决 8：A04 按当前主仓 owner 常量实现，不预判 F3 合并**
——即 `FEATURE_REGISTRY_VERSION` 14→15、`RESEARCH_DOMAIN_VERSION` 维持 5、
`GRAMMAR_VERSION` 维持 9、计数按当前 83 槽位 / 70 可采样实测（§1.9）。**若 F3 先合并，
A04 的 bump 目标与三项计数在合并时重新对账**（从新的 main 起点重测，并重述本表与
§2.1.4）；届时 `FEATURE_REGISTRY_VERSION` 的自增目标相应上移，但**不得**照抄本表
数字。实现 commit 必须以**重测结果**为准。

## 5. 迁移/拒绝策略

1. **面向未来，不改历史**：纠偏只影响批准后的搜索空间与资格判定；既有
   artifact、测量日志、族级裁决一律保留原值（I12、AGENTS §11.4）。
2. **旧公式兼容性**：类型变化可能使既有公式在**新**类型检查下不再合法
   （例如把 F2 用作 `CS_NEUTRALIZE` 之外某些同型约束的第二个参数）。
   策略：
   - 既有 artifact 中的公式**按原样只读保留**（artifact 自带
     `feature_registry_version`，由 `classify_schema_version` /
     `artifact_versions` 的既有分类权威判定 legacy/current）；
   - 新运行不得把旧公式静默「修正」后重用；需要继续时以新 run ID 重跑
     （AGENTS §4.3）；
   - 若某旧公式在新类型下非法，必须**拒绝**并报告具体算子/位置，不得放宽
     类型检查让它通过。
3. **拒绝而非降级**：清单核对发现「声明与来源不符」时必须显式列出并修复，
   不得以「不影响当前搜索」为由沉默（AGENTS §4.3）。
4. **拒绝清单生成失败**：若 FO-01 清单无法从注册表完整派生（缺字段、缺单位、
   缺 NaN 语义），生成器必须 fail-closed 失败并列出缺口，不得输出部分清单
   冒充完整。
5. **WIP 隔离**：`AlphaGPT-factor` 的 F3 WIP 不得被纳入本包提交；若合并顺序
   导致冲突，暂停本包并报告（AGENTS §12 末段：共享契约变化先改契约）。
6. **回滚**：元数据纠偏必须可整体回滚（单文件/单提交）；回滚后系统回到
   「F2 使用季度 TTM 模板 + FUNDAMENTAL_LIKE」的旧口径，且不得丢失 FO-01
   清单（清单本身是独立交付物）。

## 6. 预期 RED 测试清单（先失败、再实现；AGENTS §2.2 第 3 步）

断言来源是本节条款与已批准数据契约，不得引用实现输出（AGENTS §10.1）。

| # | 场景 | 期望行为（RED 断言） | 不变量 |
|---|---|---|---|
| R1 | **F2 被当成季度数据**：读取 F2 的 `availability_rule` | 必须表达 margin T+1 + daily_basic 当日的联合规则；不得含季末披露季模板 | I1、I2 |
| R2 | **单位错配**：人民币/万元、市值单位 | FO-01 清单必须给出每因子单位，且与来源字段单位一致；错配即失败 | I6 |
| R3 | **两源到达时间不一致**：`margin_detail_pit` T+1 与 `daily_basic_pit` 当日 | 联合可见性必须显式表达两者；只写其一的声明失败 | I2 |
| R4 | **缺依赖时被填零**：F2 在缺 margin/daily_basic 时的行为 | 必须保持 NaN（不得中性化），且 FO-01 清单声明该 NaN 语义 | I6 |
| R5 | **所有搜索后端对修正后的类型有一致合法动作集合** | GP/Random/TPE/RL 对同一表达式给出同一合法性判定 | I8 |
| R6 | **未来数据改动不改变过去特征** | 纠偏只影响批准后的运行；既有 artifact 只读保留 | I12 |
| R7 | **类型判定可举证**：F2 的类型 | 必须由 §3/I3 规则推出并附合法动作集合前后对照（`CS_NEUTRALIZE` 可用性变化） | I3 |
| R8 | **依赖口径统一** | 同一注册表内不得混用表名与字段名口径 | I4 |
| R9 | **FO-01 清单完整性** | 清单必须覆盖 70 个可采样特征 + 39 个算子，且每个对象含单位/时间/类型/NaN/依赖 | I5、I6 |
| R10 | **清单是生成物** | 手工维护的第二份名单失败；drift guard 必须能发现注册表与文档不一致 | I5 |
| R11 | **词表不变** | `FEATURE_NAMES` 长度/顺序/token id 与 deprecated 集合零改动 | I7 |
| R12 | **数值不变** | 纠偏前后同一面板的因子张量逐元素相同 | I9 |
| R13 | **D7 不被顺手改** | F2 的 `pit_level`/来源映射保持 `PIT_CAPITAL`/margin；资格门判定逻辑零改动 | §2.3 非目标 4 |
| R14 | **族级裁决不被推翻** | p24 的候选池准入结论不变；证据状态可区分 | I10 |

**既有测试的处置（AGENTS §10.1 白名单第 1/2/3 条，逐条举证）**：可能受影响的
既有断言包括 `tests/test_feature_metadata.py:120-145`（`depends_on` 与
`required_columns` 的派生关系）、`tests/test_factor_semantic_fixes.py:288`
（`MARGIN_BALANCE_CHG` 的 `depends_on == ("margin_detail_pit",)`）、
`tests/test_feature_registry.py:273-303`（F2 的记录字段）、
`tests/test_vocab.py:505`、`tests/test_research_domain.py:573-594`、
`tests/test_data_tier.py:103`、`tests/test_factors.py:175`。实现 commit
**必须**逐条列出：所属白名单条目、引用本文条款、旧行为为何错误、新断言为何
不更弱。**禁止**删断言、加 skip、把 expected 改成 actual。

## 7. 测量方案（只写方案；真实数值必须由实现后的 evidence-only 提交写入测量日志）

测量日志文件（实现后建立，本契约不创建）：
`docs/p28_a04_measurement_log.md`。

1. **命令与环境**：逐条记录命令、commit/tree 状态、OS/Python、CPU/内存。
2. **FO-01 清单交付**：清单文件路径、生成命令、覆盖计数（可采样特征 70、
   算子 39）、逐对象字段完整性统计、drift guard 结果。
3. **合法动作集合前后对照**：对同一固定面板与同一预算口径，输出
   「修正前 vs 修正后」的合法动作集合差异，至少覆盖：
   - `CS_NEUTRALIZE` 对 F2 的可用性变化（§1.4 实测：唯一限定单一输入类型的
     算子）；
   - 同型约束（family-preserving 算子）下可与 F2 组合的特征集合变化；
   - 四个后端的一致性（I8）。
   差异必须**逐项解释**，不得只给总数。
4. **数值等价性**：纠偏前后同一面板的因子张量逐元素比较（期望完全相同，
   I9）；任何差异即失败并停下。
5. **声明-来源对照表**：四个 margin 因子的 `availability_rule`/`semantic_type`/
   `depends_on`/单位/NaN 与 p17 §5.2/§6.1、p19 §2 的逐条对照。
6. **未运行项**：任何因资源/环境未运行的验证必须显式列出（AGENTS §10.2）。
7. **研究声明边界**：A04 是元数据/语义纠偏，其测量**不得**被表述为 alpha、
   收益或晋级证据；族级裁决与单因子有效性的区分必须保留（I10）。

## 8. 资源上限

1. **运行环境**：6C/12T、16GB RAM、串行（p21 §5）。
2. **清单生成**：一次全量生成 ≤ **60s**、RSS ≤ **512MB**。
3. **合法动作集合对照**：固定小面板（建议 ≤ 200 股票 × ≤ 500 交易日、
   特征张量 float32），单次对照 ≤ **10min**、RSS ≤ **4GB**；禁止为此重跑
   正式搜索或训练。
4. **测试**：新增/修改测试必须在 `tmp_path` 或内存面板上运行，禁止触碰
   `data/` 下的正式数据库与产物；单文件测试墙钟 ≤ **120s**。
5. **依赖**：不得新增生产依赖（AGENTS §9）；清单生成必须复用既有
   `scripts/generate_registry_docs.py` 路径或等价的零依赖实现。

## 9. 裁决规则

1. **仲裁顺序**：用户本次明确要求 / 已批准契约 → 契约测试 → 实现 → 测量
   日志（AGENTS §1）。测量只能执行本契约的裁决，不能反向改写契约。
2. **p17/p19 优先**：可见性与单位的口径以已批准的 p17 §5.2/§6.1/§6.2 与
   p19 §2 为仲裁依据；本契约不得放宽其要求。
3. **类型二选一的裁决权**：§3/I3 的规则是**判据**，最终选择（`CROSS_SECTIONAL_SIGNAL`
   vs `VOLUME_LIKE`）由批准人在 §11 A04-A2 裁决；起草人不预先决定。
4. **版本争议**：§4.2 中标注「视实现决定」的项，由实现 commit 的版本影响表
   裁决并逐项举证；`GRAMMAR_VERSION` 是否 bump 由批准人裁决（A04-A4）。
5. **测试争议**：见 §6 末段（AGENTS §10.1）。
6. **审核**：本契约的批准与实现必须分别经独立审核（合规 + 技术/验证），
   审核者不得是起草人或实现者。
7. **不得顺手扩张**：任何「核对时发现的其它不一致」必须记录为后续小任务，
   不得在本包内顺手修复（AGENTS §9.1）。

## 10. 停止条件（任一命中 → 任务 failed + 完整报告，禁止静默降级）

1. **源语义无法确认即暂停**（补充方案 §6 FO-02 停止条件同款精神）：若某因子
   的单位/可见性无法从已批准数据契约确认，标 `unknown` 并暂停相关候选，
   **不得**推定。
2. **不得为让旧公式/旧 run 合格而放宽类型或可见性**：任何放宽类型检查、
   改写旧 artifact 的 `feature_registry_version`、或追溯重写既有测量的行为
   一律红线（AGENTS §4.3）。
3. **数值发生变化**：I9 失败即停下，重新分类（工程 vs 语义），不得继续。
4. **清单无法完整派生**：FO-01 清单缺字段/缺单位/缺 NaN 语义时，fail-closed
   失败并列出缺口；不得输出部分清单冒充完整。
5. **与 `AlphaGPT-factor` WIP 冲突**：若合并顺序导致 F3 与 A04 冲突，暂停
   本包并报告；**不得**把 F3 WIP 纳入本包提交。
6. **需要修改本契约的类型判定规则、词表或 p17/p19 口径**：停下，提交修订
   草案，等待批准。
7. **触碰 inScope 之外的文件**、修改 `AlphaGPT-factor` 工作树、或覆盖用户
   既有修改：立即停止并报告。
8. **出现「为通过测试而弱化断言」的诱因**（例如 R5 的后端一致性难以构造）：
   停下并报告，不得删测试或改断言。

## 11. 需用户（或转授后的 captain）明确批准才能实现的条款

| 编号 | 需批准的条款 | 冻结的语义 | 若被否决的后果 |
|---|---|---|---|
| A04-A1 | §3/I2：F2 两因子的联合可见性表达（margin T+1 + daily_basic 当日） | 可用时间 | F04 的可见性错配保留 |
| A04-A2 | §3/I3：语义类型判定规则与 F2 的最终类型（`CROSS_SECTIONAL_SIGNAL` vs `VOLUME_LIKE`） | 特征语义类型 → 搜索空间 | 类型错配保留；`CS_NEUTRALIZE` 可用性裁决悬空 |
| A04-A3 | §3/I4：`depends_on` 口径统一（表名 vs 表.字段） | 依赖声明口径 | 清单无法核对；生成文档口径漂移 |
| A04-A4 | §4.2：`FEATURE_REGISTRY_VERSION` 14→15 与 `GRAMMAR_VERSION` 是否 bump——**captain 裁决 8 已关闭**：14→15 执行、`RESEARCH_DOMAIN_VERSION` 维持 5、`GRAMMAR_VERSION` 维持 9 | artifact/grammar 兼容性 | 已裁决，不再悬空；实现须按裁决 8 执行 |
| A04-A5 | §3/I5 + I6：FO-01 清单的交付形态与新增「单位/NaN」结构化字段 | 存量契约清单的权威形态 | 只能继续手工核对 |
| A04-A6 | §5.2：旧公式在新型下的拒绝策略 | 旧产物可用边界 | 旧公式可能被静默修正重用 |
| A04-A7 | §2.3 非目标 4：不修改 D7 资格门 | 资格门范围 | 可能被「顺手修」引入第二套映射 |
| A04-A8 | §7.3：合法动作集合前后对照的测量口径 | 搜索空间影响的证明方式 | 无法证明类型变更的实际影响 |
| A04-A9 | §8：资源上限数值（60s/512MB 清单；10min/4GB 对照） | 工程预算 | 无上限即无法判定「超限即停」 |
| A04-A10 | §10 停止条件（尤其「源语义无法确认即暂停」与「不得放宽类型」） | 证据完整性红线 | 停止条件不可作为失败裁决依据 |

## 11bis. captain 裁决记录（2026-09-10，依用户转授，`reports/alphagpt-ab/00_authority_exception_record.md`）

本节如实登记 captain 对本文相关处的裁决**原文要点与落点**，供 t6 批准决策记录
引用。**本文件状态仍为 DRAFT**：按团队分工，DRAFT→APPROVED 的状态变更与批准
记录由 captain 在 t6 执行；起草人只记录裁决内容，**不自行改状态**（AGENTS §3.2、
p18 §3.2）。

| 裁决 | 内容（captain 原文要点） | 落点 |
|---|---|---|
| 裁决 8 | **A04 与 F3 WIP 的版本起点分叉**：A04 按**当前主仓 owner 常量**实现——`FEATURE_REGISTRY_VERSION` 14→15、`RESEARCH_DOMAIN_VERSION` **维持 5**、`GRAMMAR_VERSION` **维持 9**、槽位按**当前 83 实测**；**不预判 F3 合并**；若 F3 先合并，A04 的 bump 目标与计数在合并时重新对账。理由：F3（p25）尚未准入，且事件域已裁决不纳入 A02 身份 | §2.1.4 判定结论段；§4.2 `FEATURE_REGISTRY_VERSION` / `RESEARCH_DOMAIN_VERSION` / `GRAMMAR_VERSION` 三行；§4.2「已知版本起点与 F3 的关系」段；§11 A04-A4（已关闭） |
| （相关）F3 WIP 同模式问题登记为 FO-02 输入 | 四个 F3 因子复用同一季度 TTM 模板，而真实来源是公告日驱动的事件表；**A04 不得顺手改** | §2.1.4 第 3 行判定（「需注意（非冲突但同模式）」）；§2.3 非目标 5；§10.6 |

**裁决未涵盖、仍待 captain/用户在 t6 决定的项**：`SEMANTIC_CACHE_VERSION` 与
`FACTOR_CACHE_VERSION` 的具体 bump 形式（§4.2 仍标「视实现决定」）、以及
**A04-A2 的 F2 语义类型二选一**（`CROSS_SECTIONAL_SIGNAL` vs `VOLUME_LIKE`，
裁决 8 未涉及，仍属研究语义待批准项）。

## 12. 附：本草案的边界核对（落盘时的事实）

- 本文件**只**新增两个改动：`docs/p28_a04_feature_metadata_semantic_type_draft.md`
  与 `docs/CONTRACTS.md` 的一行索引（同 commit）。
- **未**修改任何源码、测试、配置、数据库或 artifact；**未**运行任何训练、
  搜索、评估、回测、模拟、数据同步或回填；**未** push / 建 PR / 合并。
- **未**修改 `AlphaGPT-factor` 工作树的任何文件（只读 `git status`/`git diff`
  评审其 F3 WIP，结论见 §2.1 假设 4）。
- **未**自我批准：状态为 DRAFT，正文不含 `APPROVED` 状态行，也不含批准记录行。
- **未**预写任何「实际发生」的测量结果（AGENTS §11.1）；§7 仅为测量方案。

## 13. 修订记录

| 版本 | 日期 | 修订内容 | 依据 |
|---|---|---|---|
| 初稿 | 2026-09-10 | 首次起草（勘察基线 main @ `1fb1b16`），含 8 文件 F3 WIP 逐文件判定表 | t4/T03 任务指派 |
| 修订 1 | 2026-09-10 | 落实 **captain 裁决 8**：§4.2 `FEATURE_REGISTRY_VERSION` 14→15（补裁决标注）、`RESEARCH_DOMAIN_VERSION` 维持 5、`GRAMMAR_VERSION` 维持 9（删除「待批准人裁决」措辞）；§2.1.4 与 §4.2 改为「A04 不预判 F3 合并；F3 若先合并则在合并时重新对账」；§11 A04-A4 标为已裁决；新增 §11bis 裁决记录 | captain 裁决 8（2026-09-10，依 `reports/alphagpt-ab/00_authority_exception_record.md` 转授） |
| 修订 2 | 2026-09-14 | 状态 DRAFT→APPROVED（批准记录回写，见文件头）。实现前重测（AGENTS §11.3）记录：main @ `aac4aad` 上 `SEMANTIC_CACHE_VERSION=2`/`FACTOR_CACHE_VERSION=2`（§4.2 表中的「1」已被 A02 消费）；`FEATURE_REGISTRY_VERSION=14`、`GRAMMAR_VERSION=9`、`RESEARCH_DOMAIN_VERSION=5`、槽位 83/可采样 70/算子 39 与 §1.9 一致。实现决策登记：①§4.2 语义缓存行按「把 `FEATURE_REGISTRY_VERSION` 纳入键」分支执行——`SemanticCacheKey` 新增 `feature_registry_version` 字段 + `SEMANTIC_CACHE_VERSION` 2→3（键布局变化即 bump 依据；`FACTOR_CACHE_VERSION` 不 bump——其身份已含 `FEATURE_REGISTRY_VERSION`（`load_cache.py`），FRV bump 自动失配）；②I4 落地为「家族内一致 + 禁混用」：margin 四因子统一 `表.字段`，全注册表禁止单条目内混装点式/裸式依赖；fundamental 家族的表级依赖迁移登记为后续小任务（§9.7，不顺手扩张）；③I10/I5 的四状态（implementation/research_eligibility/evidence/promotion）以**派生**字段进 FO-01 清单（单一权威派生，不建第二名单）；④D6 复核发现：`asof_winner_intervals` 为逐行 `available_date` 可见性，`_FUNDAMENTAL_TTM` 模板的「法定披露季结束」文案与实现不符（F1/family-⑤ 条目）——按 §9.7 登记为后续任务，本包不改（非目标 2）。测试精化披露（§10.1 白名单）：单位受控词表定稿 `price_adj`（复权基价格量，替代草稿 `price_yuan`）、`MARGIN_BALANCE_CHG` 单位定稿 `return`（20 日变化量）；RED 期容差 1e-9→1e-6（对齐实现的 `std+1e-9` 防除零，非弱化）；R12 增加 `tmp_path` 夹具参数 | 用户本次直接指令（2026-09-14「完成A04」）+ 授权记录 §11/§4ter；实现 commit 证据 |
| 修订 3 | 2026-09-14 | 双审核（合规 + 技术/验证，均独立于实现者）**pass** 后的勘误与对齐（evidence-only，无代码变更）：①§2.2 范围清单补列第 8 项 `ashare_model/semantic_cache.py`（对齐 §4.2「视实现决定」分支 + t6 裁决 2——合规 C-F-1）；②§4.2 勘误两行：`RUNSPEC_SCHEMA_VERSION` 旧值 1→**2**、`ARTIFACT_SCHEMA_VERSION` 旧值 2→**3**（base `aac4aad` 实测，A02 已消费；A04 未触碰两者的决策不变——技术 T-F-4）；③登记后续任务：evidence_status 派生规则区分撤回原因（技术 T-F-3）、全注册表 depends_on 口径统一（技术 T-F-5，含 INDUSTRY_MOMENTUM 既有混装）、RED 期精化保留原值痕迹（合规 C-F-3，实践改进）。完整处置清单见测量日志 §11 | 双审核报告（pass ×2，0 blocker/0 major/8 minor）；`reports/alphagpt-ab/A04_report.md` §10 |
