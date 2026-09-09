# P29 B01 PIT 与可交易数据质量预注册契约（DRAFT）

- 状态：**DRAFT（草案）——未批准，禁止实现消费**。在用户（或按
  `reports/alphagpt-ab/00_authority_exception_record.md` 转授的 captain）给出
  明确批准原文并完成批准记录回写之前，本文件只作为待审材料，**不得**被实现、
  测试、正式运行或晋级路径引用为语义权威。
- **批准权属**：本契约的 DRAFT→APPROVED 状态变更**只能由用户明确批准**
  （AGENTS §3.2：可用时间、PIT mask、数据 tier、股票池、晋级阈值属硬边界；
  `docs/p18_captain_authority_clause.md` §3.2 明确 captain 原本无权批准）。
  本团队存在用户转授批准权的例外记录
  （`reports/alphagpt-ab/00_authority_exception_record.md` §3.2，授权原文
  「所有行为我强制要求你来替我批准，」），据此 captain 可在 §5 逐契约决策记录
  填写完成后代为批准；该转授**不是**起草人可自我批准的依据。**起草人
  （governance-author）无权批准本契约，也无权追认。** 无批准原文时本契约保持
  DRAFT，依赖它的实现任务（T26/T27）必须停在等待态。
- 起草：governance-author（AgentTeams `alphagpt-remediation-ab` t5/T04），
  2026-09-10。
- 起草依据：
  - 用户任务指派（T04 起草 B01 预注册契约，DRAFT）；
  - `reports/AlphaGPT_整改与提升执行方案_2026-09-09.md` §四 B01 工作包（范围、
    实施重点、失败测试、验收、依赖）；
  - `reports/AlphaGPT_因子与算子开发补充方案_2026-09-10.md` §5 接线契约、
    §6 FO-02 前置（A02/B01 合格）、§8 顺序关系；
  - AGENTS.md §1/§3.2/§4.3/§5/§10.1/§11/§13；
  - `docs/p17_tushare_pit_data_contract.md`（数据底座，APPROVED：§5.2 表清单、
    §6.1 可见性、§6.2 财务 as-of、§12 分域资格）、
    `docs/p16_data_freshness_gate_contract.md`（G8，APPROVED）、
    `docs/p19_market_price_semantics_contract.md`（§2.4 精确交易约束，
    APPROVED）、`docs/p2_data_tier_contract.md`（Tier A/B/C）；
  - `docs/p26_a02_dataset_snapshot_identity_draft.md`（B01 的数据身份前置，
    DRAFT）、`docs/p28_a04_feature_metadata_semantic_type_draft.md`（FO-01
    单位/依赖口径前置，DRAFT）。
- **勘察基线（docs/CONTRACTS.md「勘察基线惯例」[02-INT-09]）**：起草时
  **main @ `1fb1b16faef30a68332b9e07059c8632fb80ee59`**（`1fb1b16`）。
  代码事实与行号在该 commit 上核对；**数据事实**来自本任务新增的只读探针
  （全部 `read_only=True`，未写入任何数据库，见 §1.11 的实测清单）：
  `data/ashare.duckdb`（968.3 MB，11 表）与 `data/ashare_v2.duckdb`
  （7,588.8 MB，27 表）。本任务**未**修改 `AlphaGPT-factor` 工作树。
- 本文使用 AGENTS.md 的「必须 / 禁止 / 应当」语义。任务路由：**数据/股票池
  变更**（AGENTS §3.2、§5）+ **研究语义变更**（§3.2：可用时间、PIT mask、
  数据 tier）+ **运行态/晋级**（§4.3 fail closed）。仲裁顺序：用户本次明确
  要求 / 已批准契约 → 契约测试 → 实现 → 测量日志。

## 0. 六要素自检（AGENTS §3.2 要求项 → 本文落点）

| AGENTS §3.2 要求项 | 本文落点 |
|---|---|
| 问题陈述 | §1（方案 B01 实施重点 + 本次只读实测证据） |
| 假设 | §2.1 |
| 范围与非目标 | §2.2 / §2.3 |
| 不变量 | §3（I1–I14） |
| 版本变化 | §4（§4.0 G1–G8 语义影响评估 + §4.1 `git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"` 全量检索结果 + §4.2 逐项影响表） |
| 迁移/拒绝策略 | §5 |
| 预期 RED 测试 | §6（R1–R16） |
| 测量方案 | §7 |
| 资源上限 | §8 |
| 裁决规则 | §9 |
| 停止条件 | §10 |
| （附加）需批准条款清单 | §11 |

## 1. 问题陈述（方案 B01 实施重点 + 本次只读实测证据）

方案 §四 B01 给出五项实施重点与一组失败测试。本节把每一项落到**实测事实**上，
并补充本次探针新发现的、方案未逐条列举的证据。所有数据事实均为只读探针结果
（§1.11）。

1. **时间字段存在混合格式，且字符串区间比较会静默失配（最严重，实测）**。
   方案 B01 原文：「时间字段统一规范，特别检查主库同时存在 ISO 和紧凑日期的
   历史情况；在导入边界验证，禁止字符串区间比较依赖混合格式。」实测：
   - `data/ashare.duckdb` 的 `daily_bar.trade_date` 有 **2,957 个不同取值**，
     其中**紧凑格式 2,839 个、ISO 格式 118 个**（`2015-01-05` … `2015-06-29`，
     全部属于 `000982.SZ`）；全库非 8 字符日期的行数 = **118**；
   - `trade_calendar.trade_date` **全部为紧凑格式**（样本
     `19901219/19901220/19901221`）；118 个 ISO 日期在日历中**逐字命中 0 次**，
     转紧凑后命中 **118 次**；
   - `data/ashare_v2.duckdb` 的 `daily_bar` / `daily_bar_raw` /
     `trade_calendar` **零混合**（各 3,081 / 3,081 / 8,797 个不同日期全紧凑）；
     全库 51 个日期列**无混合**。
   - **后果（可复现的静默失配）**：`ashare_data/gates.py:124-178`
     `_contract_eligible_mask` 把 `daily_bar` 的 `trade_date` 与
     `contract.sessions`（来自紧凑格式日历）做 `reindex` 对齐；ISO 行
     （`"2015-01-05"`）与紧凑会话（`"20150105"`）不相等，因此**这 118 根 bar
     在 mask/presence 中全部丢失**，该股票在其成员窗口内表现为「零 bar」，
     而它**并非**零 bar——这正是方案「不把退市前后无数据和应该存在但缺失的
     bar 混为一类」所指的混类。
   - **G7 抓不到它**：G7（`gates.py:378`）检查「每个 PIT 成员区间有日线」，
     但其判定同样走日期字符串对齐，因此 ISO 行对 G7 不可见。
   - **字典序陷阱**：`"2015-01-05" < "20260907"`（`'-'` 的码位小于任何数字），
     所以 `daily_bar` 的字典序 `MIN` 就是那条 ISO 行；G3
     （`gates.py:288-297`）的 `calendar.mn <= daily.mn` 比较因此仍为真——
     **混合格式能让区间检查「碰巧通过」**。
2. **四种时间语义尚未分离**。方案 B01 原文：「区分经济有效时间、公告/可用
   时间、实际采集时间和修订版本。带 ann_date 的历史数据不自动成为完整修订
   历史；源无法还原之处保留 Tier 限制。」实测：v2 已有 `available_date`
   （`industry_membership_pit` 16,611 行、`margin_detail_pit` 5,877,462 行）
   与公告日（`earnings_forecast_pit.ann_date` 126,720 行、
   `repurchase_event_pit.ann_date` 21,018 行），也有 `source_ingest_runs`
   台账（采集时间）；但**没有任何机器可核对的「四时间 + 修订版本」声明**，
   `ashare_data/pit_import.py:123-` `asof_winner_intervals` 只处理财务版本
   胜出者，其余族无统一口径。
3. **月度指数权重尚未明确 as-of 转日度**。方案 B01 原文：「月度指数权重快照
   按明确 as-of 规则转为日度区间。」实测 `index_weight_pit` 348,890 行，
   日期范围 `20050131`…`20260831`，按指数分：`000852.SH` 142,300、
   `000905.SH` 129,540、`000300.SH` 77,050——日期全部落在**月末**（如
   `20141031`、`20260831`）。`ashare_data/pit_import.py:71-` `timeline_to_intervals`
   与 `:50-` `month_end_samples` 存在，但**没有把「月末快照 → 日度可见区间」
   的 as-of 规则写进机器契约**，也没有「月末非交易日」的处置断言。
4. **覆盖率分母口径需要固定**。方案 B01 原文：「数据覆盖分母使用『当日已上市、
   属于目标 PIT universe、应当可观察』的单元，而不是全股票×全历史笛卡尔积。」
   实测：`ashare_model/data_eligibility.py:289-` `_financial_coverage` 以
   「季度末 × universe 上市日」为分母（`_universe_list_dates`，`:263-`），
   `:216-` `_lag_sessions` 按开市会话位置计算滞后；但**没有**对每个源统一声明
   「分母 = 当日已上市 ∧ 属于目标 PIT universe ∧ 应当可观察」，也没有把
   「应当可观察」与「源覆盖不到」区分开。
5. **可交易约束需按执行日真实值**。方案 B01 原文：「涨跌停/停牌采用执行日
   真实约束。科创板等缺口逐码逐日定位，不用硬编码固定涨跌幅填成正式证据。」
   实测：v2 有 `daily_price_limit`、`daily_suspension`，`p19 §2.4` 已规定
   「缺失记录 fail-closed、不得填 0」；`data_loader.py:455-469` 会把缺失的
   涨跌停记录记为 `price_degraded`。但**逐码逐日的缺口清单**（哪些 (股票,
   日期) 缺 `daily_price_limit`/`daily_suspension`，按板块分类）尚未作为机器
   产物存在。
6. **零事件与接口漏采尚未区分**。方案 B01 原文：「零事件与接口漏采必须由成功
   采集批次证明区分。」实测：`source_ingest_runs` 台账存在（v2），
   `scripts/backfill_v2.py` 与 `ashare_data/tushare_store.py` 记录 ingest 状态；
   但「某日某事件族零行」与「该日接口失败/未采集」的区分**没有**落到分域资格
   报告里。
7. **holder_number_pit 覆盖到 2022 年末（F3 扩张前的独立检查项）**。方案 B01
   原文：「本次 holder_number_pit 截至 2023 年的状况说明 F3 扩张前需要单独
   检查数据，不应因此全局阻断纯价格基线。」实测：`holder_number_pit.end_date`
   范围 `20101231`…`20221230`（244,658 行）——**最晚 2022-12-30**，即 2023 年
   起的股东户数缺失。该源不合格**不得**阻断纯价格基线（分域资格原则）。
8. **G1–G8 与 D7 的角色需要明确**。方案 B01 验收原文：「DataGate G1–G8 与 D7
   的角色明确。不能把 N/A 偷换为通过，也不能通过删除困难样本改善覆盖率。」
   实测：G1–G8 在 `ashare_data/gates.py:264-441`（G1 成员区间、G2 上市日、
   G3 日历覆盖、G4 区间结构、G5 PIT 契约、G6 eligible 下限、G7 零 bar 存续、
   G8 新鲜度）；D7 在 `ashare_model/data_eligibility.py:354-`（按 RunSpec 因子
   集解析所需源，逐源 coverage/lag 检查）。两者**编号空间独立**
   （AGENTS §5.4），但「某源不合格是阻断正式运行还是仅降级」的机器判据尚未
   统一成文。
9. **阈值是语义常量，不得在实现中随手改**。实测既有阈值：
   `DAILY_ELIGIBILITY_MAX_LAG_SESSIONS=3`、`PRICE_LIMIT_MAX_LAG_SESSIONS=3`、
   `MARGIN_MAX_LAG_SESSIONS=2`、`INDUSTRY_INDEX_MAX_LAG_SESSIONS=3`、
   `INDUSTRY_MEMBERSHIP_MIN_COVERAGE=0.98`、`FINANCIAL_REPORT_COVERAGE_MIN=0.90`
   （`ashare_data/gates.py:73-78`）、`MIN_ELIGIBLE_DEFAULT=100`（`:57`）、
   `FRESHNESS_TOLERANCE_SESSIONS=3`（`:63`）。B01 的分域报告必须**消费**这些
   单一权威常量，**禁止**在报告层复制一套数字（AGENTS §1 原则 3、§9）。
10. **D7 的按需原则已预注册，B01 不得扩大成全局门**。`data_eligibility.py:12`
    明确「must be fresh」的全局门是 p17 §12 的**预注册非目标**；B01 的分域
    报告必须保持「只要求该研究依赖的源合格」。
11. **本次实测清单（只读探针，未写入任何数据库）**：
    - 主库 11 表：`constituents / daily_bar / dataset_manifest /
      dataset_manifest_cache / factor_cache / fundamental_pit /
      margin_balance / stocks / sw_industry_index / sw_industry_member /
      trade_calendar`；
    - v2 库 27 表（含 p17 的 19 张 + 兼容表）；
    - 主库日期列 9 个，其中**混合格式 1 个**（`daily_bar.trade_date`）；
      v2 库日期列 51 个，**混合格式 0 个**；
    - `000982.SZ`：ISO 行 118、紧凑行 1,842、成员区间
      `000905.SH [20140127, 20150629)`，该窗口内可见行 118（即全部为 ISO 行）。
12. **证据基线**：`reports/AlphaGPT-audit-2026-09-09/research_doctor.json` 与
    `boundary_probes.json` 是 2026-09-09 的 doctor/边界探针原始产物；本次
    探针与它们的差异必须在测量日志中逐项说明（AGENTS §11.3：不得复制过时数字）。

## 2. 假设、范围与非目标

### 2.1 假设

1. **A02 与 A04 是 B01 的正式依赖**（方案 §四依赖表：B01 ← A02、A04；
   FO-02 前置「A02/B01 合格」）。B01 的**报告与实现**可以起草，但**正式资格
   结论必须等 A02 的数据身份内容确认与 A04 的单位/依赖口径固定之后**才能给出。
2. v2 库是正式数据的唯一来源（p19 §2.1）；主库的 `daily_bar` 属 legacy 只读
   清单（p17 §8.2），**但 G1–G8 仍读主库**（`gates.py:124-178` 用
   `config.duckdb_path`），因此主库的格式缺陷仍影响正式门禁。
3. 数据端点的可用范围会变化；执行前必须核对供应商官方更新记录与少量读取
   探针，**不据二手文档承诺接口权限、价格或完整历史**（方案 B01 末段原文）。
4. `source_ingest_runs` 台账是「采集是否成功」的唯一权威；`scripts/backfill_v2.py`
   与 `ashare_data/tushare_store.py` 的写入语义不变。
5. 机器 6C/12T、16GB RAM、串行；B01 的报告不得要求全库常驻内存。
6. 本契约的实现可在**不移动、不删除、不重写**任何既有数据库、不改写历史
   artifact 的前提下完成；数据修复（若有）必须另走用户批准的有状态操作
   （AGENTS §5.6）。

### 2.2 范围

1. `ashare_data/pit_import.py`：导入边界的时间规范与验证；四时间语义的机器
   声明；月末快照 → 日度区间的 as-of 规则。
2. `ashare_data/fundamentals.py`、`ashare_data/capital_flow.py`：源族的时间/
   单位/可见性声明与缺口定位（与 A04 的 FO-01 清单对接）。
3. `ashare_data/universe.py`：PIT universe 的「应当可观察」单元定义。
4. `ashare_data/gates.py`：G1–G8 对时间规范与缺口清单的消费（**判定逻辑
   不变**，见 §2.3）。
5. `ashare_model/data_eligibility.py`：分域资格报告的 coverage/lag/revision/
   quality 分项与真实缺口清单。
6. 必要的 schema/版本常量、测试与 `docs/CONTRACTS.md` 索引（本草案已同 commit
   登记）。
7. **按一个来源族拆包执行**（方案 §四 B01 范围末句）：建议顺序为
   行情/约束 → 复权 → daily_basic → 财务版本 → 行业成员/指数 → margin →
   事件族（forecast/express/repurchase/holder_number）。

### 2.3 非目标

1. **不**实现 FO-02–FO-08（事件规范化、新候选因子、算子原型、完整接线、
   消融、收益验证）；B01 只做**数据资格**。
2. **不**改变 G1–G8 的判定逻辑与阈值（§1.9 的常量保持原值）；B01 只增加
   「分项与缺口清单」的机器产物。若实现认为某阈值需要变化，**停下**另立契约。
3. **不**改变 `ashare_model/promotion.py` 的 G1–G7 晋级门（两个编号空间不
   混用，AGENTS §5.4）。
4. **不**把 D7 从「按需」改成「全局门」（p17 §12 非目标，§1.10）。
5. **不**执行任何数据库写操作（同步、回填、purge、迁移、修复）；发现数据
   缺陷只**报告**，处置由用户批准的有状态任务执行（AGENTS §5.6）。
6. **不**删除困难样本、不缩小分母、不把 `N/A` 当通过（方案 B01 验收原文）。
7. **不**新增数据源、依赖或第二套日历/覆盖率实现（AGENTS §1 原则 3、§9）。
8. **不**把 holder_number_pit 的覆盖缺口升级为全局阻断（§1.7）。
9. **不**触碰 `AlphaGPT-factor` 工作树；**不**修改 inScope 外文件。
10. **不**预写任何「实际发生」的测量结果（AGENTS §11.1）。

## 3. 不变量（I1–I14）

**I1（导入边界规范化 + fail-closed 验证）**：所有日期字段在**导入边界**统一
为规范形式（本文固定为紧凑 `YYYYMMDD`，与 `trade_calendar` 一致），并在边界
验证：非规范值必须**拒绝或显式记录并修复**，禁止让混合格式进入下游的字符串
区间比较。既有混合数据必须先出**只读缺口清单**（§1.11 已给出第一份），再
由用户批准的有状态任务修复。

**I2（禁止依赖混合格式的字符串比较）**：任何区间比较（`BETWEEN`、`<=`、
`reindex`、字典序 `MIN/MAX`）必须在**已验证同构**的日期轴上进行；实现必须
提供机器断言（例如「参与比较的两侧 100% 匹配规范式」），否则 fail-closed。

**I3（四种时间语义分离）**：每个源族必须分别声明并在数据中可核对：
（a）**经济有效时间**（数据本身描述的时间，如报告期 `end_date`、交易日）；
（b）**公告/可用时间**（`ann_date`/`available_date`，何时可被市场看到）；
（c）**实际采集时间**（`source_ingest_runs` 的 ingest 时间）；
（d）**修订版本**（同键多次披露的版本行）。四者不得混用；「带 ann_date 的
历史数据」**不自动**成为完整修订历史。

**I4（不可还原即保留 Tier 限制）**：源无法还原完整修订历史时，该族的 Tier
限制必须保留并显式记录（p2 契约的 A/B/C 语义）；禁止用「有 ann_date」推定
修订完整。

**I5（月末快照 → 日度区间的 as-of 规则）**：`index_weight_pit` 等月末快照必须
按**明确、预注册的 as-of 规则**转为日度可见区间（例如「快照值自其发布后的
首个开市会话起可见，直至下一快照发布前一日」）；「月末非交易日」必须有确定
处置（顺延到下一开市会话或按公告日），且该规则必须写进机器契约与测试。

**I6（覆盖率分母 = 应当可观察单元）**：每个源的覆盖率分母必须是「**当日已
上市 ∧ 属于目标 PIT universe ∧ 应当可观察**」的单元集合；禁止使用
「全股票 × 全历史」笛卡尔积，也禁止用缩小分母（删除困难样本）改善覆盖率。

**I7（可交易约束取执行日真实值）**：涨跌停与停牌必须来自执行日的真实记录
（`daily_price_limit`、`daily_suspension`）；缺失记录 **fail-closed**（不得填 0、
不得用硬编码涨跌幅替代），且必须给出**逐码逐日**的缺口清单（按板块分类，
含科创板等）。

**I8（零事件与漏采必须可区分）**：某日某族「零行」只有在 `source_ingest_runs`
证明**该批次成功**时才可判为「零事件」；否则必须判为「漏采/失败」并计入缺口。
两者不得互相冒充。

**I9（分域资格报告）**：每个源族必须输出 `coverage / lag / revision / quality`
四类分项与**真实缺口清单**（具体到表、字段、股票、日期），并明确该源
`pass / fail / unknown / not_applicable` 四态；`not_applicable` **不得**算作
`pass`（方案 B01 验收原文）。

**I10（按需而非全局）**：只要求该研究**依赖的**源合格；未依赖的源标
`not_applicable` 并说明（p17 §12 非目标）。holder_number_pit 的 2023 起缺口
不得阻断纯价格基线（§1.7）。

**I11（G1–G8 与 D7 角色分离）**：G1–G8 是**数据资格门**（`ashare_data/gates.py`，
CLI `scripts/check_production_gates.py`）；D7 是**按 RunSpec 因子集的源资格
门**（`ashare_model/data_eligibility.py`）。两者编号空间独立、不得混用；分域
报告必须标明每条结论属于哪个空间。

**I12（阈值单一权威）**：所有阈值/容差必须来自既有常量（§1.9），报告层不得
复制数字；任何阈值变化属研究语义变更，必须另立契约并由用户批准。

**I13（不改善覆盖率）**：禁止通过删除困难样本、缩小分母、跳过失败批次、
把 `unknown` 记为 `pass` 来改善覆盖率数字；缺口清单只增不减，除非有**新采集
证据**（AGENTS §4.3、§10.1）。

**I14（只读纪律）**：B01 的调查、报告与测试**只读**；任何写操作（同步、回填、
purge、修复）必须单独走用户批准的有状态任务，并遵守 AGENTS §5.6 的「先只读
审计精确范围 → 再执行 → 再复核 row count/日期范围/dataset ID/gates」。

## 4. 版本变化

### 4.0 G1–G8 语义影响评估（captain t5 要求 2 的专项）

**结论：B01 不改变 G1–G8 的判定逻辑与阈值，因此 `ashare_data/gates.py` 本身
不新增/不 bump 版本常量；但其「资格结论的可消费性」需要 artifact-facing 承载
（见 §4.2 的 `PRODUCTION_GATE_CONTRACT_VERSION` / `TIER_REPORT_VERSION` 两行）。**

1. **判定逻辑零改动**：G1（成员区间）、G2（上市日）、G3（日历覆盖）、G4（区间
   结构）、G5（PIT 契约）、G6（eligible 下限）、G7（零 bar 存续）、G8（新鲜度）
   的判据文本与阈值（§1.9 的 8 个常量）保持原值。B01 只**新增分项报告与缺口
   清单**，属于 §3.1 工程变更（可观测性/边界），不改研究语义。
2. **消费方式改变（须记录）**：B01 让 G1–G8 的结论**附带机器可读的分项**
   （`coverage/lag/revision/quality` + 缺口清单）。这改变的是「结论如何被消费」，
   不是「结论如何被计算」。因此：
   - 若实现只增加报告字段、不改变任何 check 的 `ok` 取值 → 不 bump
     `DATA_TIER_VERSION`，由 `TIER_REPORT_VERSION` 承载报告结构变化；
   - 若实现**确实改变**了任一 check 的判定（例如把「日期格式未验证」变成新的
     fail 条件）→ 属判定逻辑变化，必须**停下另立契约**（§2.3 非目标 2、§10.6），
     不得在本包内悄悄改。
3. **两个编号空间（AGENTS §5.4，强制重申）**：数据资格门 **G1–G8** 定义在
   `ashare_data/gates.py`（CLI `scripts/check_production_gates.py`）；晋级门
   **G1–G7** 定义在 `ashare_model/promotion.py`（`PROMOTION_RULE_VERSION="2"`）。
   **两者是相互独立的编号空间，禁止混用、禁止互相引用、禁止用一方的结论替代
   另一方。** B01 的分域报告必须逐条标注结论所属空间（§3/I11）。
4. **G8 与 p16 的关系**：G8 的机器标准由 `docs/p16_data_freshness_gate_contract.md`
   唯一定义（评估日、参考会话、lag 口径、容差 `FRESHNESS_TOLERANCE_SESSIONS=3`）；
   B01 **不**重新定义 G8，只在其报告中复用该口径并输出逐源 lag 分项。

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
| `ashare_model/data_tier.py` `DATA_TIER_VERSION` | `2` | **不变（§4.0 评估后收窄；仅在实现改变 Tier 映射时 bump）** | B01 的 I4「源无法还原即保留 Tier 限制」**不改变**既有 Tier 映射：F2 实测已是 `PIT_CAPITAL`（§1.7），G1–G8 判定零改动（§4.0）。若实现确实改变了任何族的 Tier 映射或时间规则，即属 tier 语义变化，必须 bump（p2/p17 §9 先例）——但该情形须**停下另立契约**（§4.0 第 2 条、§10.6），不得在本包内发生。实现 commit 必须逐项举证「Tier 映射零改动」。 |
| `ashare_data/tushare_store.py` `TUSHARE_STORE_SCHEMA_VERSION` | `"1"` | **视 schema 决定** | 若 I3 需要新增「采集时间/修订版本」列或新表（例如分域资格报告表），按 p17 §9 的「新 artifact-facing 版本接入既有兼容性权威」处理；若报告是**派生产物**（不入库），则不 bump。 |
| `ashare_data/gates.py` 无版本常量 | — | **（建议）新增 `PRODUCTION_GATE_CONTRACT_VERSION`** | 若 B01 给 G1–G8 增加分项/缺口清单的机器契约，该契约属 artifact-facing。新增常量必须接入既有兼容性权威（`runspec.py:_VERSION_IMPORTS` 或等价 owner 索引），并在契约中说明；若实现不改判定逻辑、只加报告字段，可由 `TIER_REPORT_VERSION` 承载（见下行）。 |
| `ashare_model/tier_reports.py` `TIER_REPORT_VERSION` | `1` | **视报告结构决定** | 若分域资格报告复用 `tier_reports` 的既有结构并新增 `coverage/lag/revision/quality` 分项字段，按「顶层字段增删改即 bump」的既有约定处理。 |
| `ashare_model/artifact_schemas.py` `ARTIFACT_SCHEMA_VERSION` | `2` | **不变（预期）** | B01 不改变 strategy/protocol/backtest/paper 四个边界产物的顶层字段；若分域报告进入正式 artifact 边界，则按 §4.1 的兼容性权威处理。 |
| `ashare_model/versions.py` `PROTOCOL_VERSION` | `"27"` | **不变** | 评价协议零改动。 |
| `ashare_model/feature_registry.py` / `vocab.py` / `factors.py` | `14` / `9` / `4` | **不变** | B01 不改变特征/算子/因子计算语义（A04 的范围）。 |
| `ashare_model/research_domain.py` `RESEARCH_DOMAIN_VERSION` | `5` | **不变** | 域划分零改动；分域资格报告按既有域/源映射组织。 |
| `ashare_model/data_eligibility.py` 无版本常量 | — | — | D7 的判定逻辑与阈值不变（I11、I12）；若新增分项字段，由 `TIER_REPORT_VERSION` 或新常量承载。 |
| `ashare_model/promotion.py` / `reward.py` / `search_contract.py` / `searcher_campaign.py` | `"2"` / `"15"` / `4` / `4` | **不变** | 晋级/奖励/搜索零改动。 |
| `ashare_model/ledger.py` / `run_store.py` / `lifecycle.py` / `observation_lineage.py` | `1` / `1` / `1` / `1` | **不变** | 账本/运行/lifecycle/lineage 零改动（A03 的范围）。 |
| 其余 `*_VERSION`（见 §4.1 指针） | 见 p26 §4.1 | **不变** | 与数据资格无所有权关系；不因 B01 变更。 |

**版本影响表使用约束**：本表是**草案的预期**，不是实现授权。实现 commit 必须
重新运行 §4.1 检索、逐项复核本表，并对每一处偏离给出理由（AGENTS §3.2）。

## 5. 迁移/拒绝策略

1. **数据不迁移，只报告**：B01 发现的时间格式混合、缺口、修订不完整一律
   先出**只读清单**；数据修复（如把 118 行 ISO 日期规范化为紧凑）属**有状态
   写操作**，必须由用户批准并遵守 AGENTS §5.6（先审计范围 → 执行 → 复核
   row count/日期范围/dataset ID/gates）。B01 实现本身**不执行**该修复。
2. **拒绝而非降级**：
   - 参与区间比较的日期轴未验证同构 → fail-closed 拒绝（I2）。
   - 涨跌停/停牌记录缺失 → 拒绝该 (股票, 日期) 的正式成交证据（I7，p19 §2.4）。
   - `source_ingest_runs` 无法证明批次成功 → 该日该族判「漏采」而非「零事件」
     （I8）。
   - `not_applicable` 被当作 `pass` → 拒绝（I9）。
3. **legacy 显式化**：主库 `daily_bar` 属 legacy 只读清单（p17 §8.2），其格式
   缺陷必须在报告中显式标 `legacy + format_mixed`；不得因为它「仍在被 G1–G8
   读取」而假装它合格。
4. **失败即停**：若分域报告在真实两库上无法给出「可核对的缺口清单」（例如
   某族缺少采集台账），任务以 `failed` + 完整报告收尾，**不得**用「推定合格」
   或缩小分母绕过。
5. **回滚**：B01 的报告/清单实现必须可整体回滚；回滚后系统回到「只有 G1–G8
   与 D7 的现状判定」，不得丢失已产出的缺口清单（清单是独立证据）。

## 6. 预期 RED 测试清单（先失败、再实现；AGENTS §2.2 第 3 步）

断言来源是本节条款与已批准数据契约，不得引用实现输出（AGENTS §10.1）。

| # | 场景 | 期望行为（RED 断言） | 不变量 |
|---|---|---|---|
| R1 | **上市前缺数据**：某股票在上市日之前无 bar | 不得计入「应当可观察」分母；缺口清单不列该项 | I6 |
| R2 | **已退市股票**：退市后无 bar | 同上（退市后不属「应当可观察」） | I6 |
| R3 | **无事件成功返回**：某日事件族零行但批次成功 | 判「零事件」（不计缺口），证据引用 ingest 批次 | I8 |
| R4 | **接口失败**：某日批次失败 | 判「漏采」，计入缺口；不得与零事件混同 | I8 |
| R5 | **月末非交易日**：快照日不是开市会话 | as-of 规则必须给出确定处置（顺延/公告日），测试钉住 | I5 |
| R6 | **报告修订/重复公告**：同键多版本 | 必须按修订版本口径处理（不倒灌、不重复计数）；无修订历史时保留 Tier 限制 | I3、I4 |
| R7 | **股票行业切换**：成员区间切换 | 行业帧按半开区间 + 可见性；切换日不得出现两个行业 | p17 §14.3 |
| R8 | **除权**：复权因子断档/非正 | 该股序列 fail-closed（不可交易 + degraded），不得 ffill/bfill 伪造 | p19 §2.1 |
| R9 | **停牌**：停牌记录存在 | 该 (股票, 日期) 不可新开仓；记录缺失即 fail-closed | I7 |
| R10 | **涨跌停缺口**：某 (股票, 日期) 缺 `daily_price_limit` | 拒绝该日成交证据；缺口清单逐码逐日列出并按板块分类 | I7 |
| R11 | **季度累计到 TTM 的非季末记录** | 必须按四个连续完整季度口径处理；非季末记录不得当作 TTM | p13 §5.3 |
| R12 | **日期格式混合**：`daily_bar.trade_date` 含 ISO 行 | 导入边界必须拒绝或规范化；区间比较必须验证同构；混合格式不得进入 mask/presence | I1、I2 |
| R13 | **覆盖率分母口径**：构造只有部分股票/日期的库 | 分母 = 已上市 ∧ 目标 PIT universe ∧ 应当可观察；不等于笛卡尔积 | I6 |
| R14 | **N/A 不是通过**：某源无数据 | 报告必须标 `unknown`/`not_applicable`，不得计入 `pass` | I9、I10 |
| R15 | **零事件 vs 漏采的台账证据** | 报告必须引用 `source_ingest_runs` 的批次状态 | I8 |
| R16 | **只读纪律**：报告生成 | 不得写入任何数据库；写操作必须被拒绝或不在本包实现 | I14 |

**既有测试的处置（AGENTS §10.1 白名单第 1/2/3 条，逐条举证）**：可能受影响的
既有断言包括 `tests/test_gates.py`（G1–G8）、`tests/test_capital_flow.py`、
`tests/test_manifest.py`、`tests/test_data_loader.py`、以及 D7 相关测试。实现
commit **必须**逐条列出：所属白名单条目、引用本文条款、旧行为为何错误、新
断言为何不更弱。**禁止**删断言、加 skip、把 expected 改成 actual。

## 7. 测量方案（只写方案；真实数值必须由实现后的 evidence-only 提交写入测量日志）

测量日志文件（实现后建立，本契约不创建）：
`docs/p29_b01_measurement_log.md`。

1. **命令与环境**：逐条记录命令、commit/tree 状态、OS/Python、CPU/内存、
   两库文件大小与 mtime。
2. **分域资格报告**：逐源族输出 `coverage / lag / revision / quality` 四态与
   真实缺口清单（表/字段/股票/日期），并标注结论所属编号空间（G1–G8 或 D7）。
3. **时间格式审计**：两库所有日期列的格式分布与混合清单（本次已给出第一份
   基线：主库 1 个混合列 / 118 行；v2 零混合）；修复后必须重测并对比。
4. **可交易约束缺口**：逐码逐日的 `daily_price_limit` / `daily_suspension`
   缺失清单与板块分类；不得用硬编码涨跌幅补齐。
5. **零事件 vs 漏采对账**：按日按族引用 `source_ingest_runs` 批次状态。
6. **阈值一致性**：报告使用的阈值必须逐项引用 §1.9 的常量与行号。
7. **未运行项**：任何因资源/环境未运行的验证必须显式列出（AGENTS §10.2）。
8. **研究声明边界**：B01 属数据资格工程，其测量**不得**被表述为 alpha、收益
   或晋级证据（AGENTS §3.1、§4.1）。

## 8. 资源上限

1. **运行环境**：6C/12T、16GB RAM、串行（p21 §5）。
2. **分域资格报告**（热路径，正式运行前）：≤ **10min**、峰值 RSS ≤ **4GB**
   （两库合计）；超出即视为设计缺陷，停下重新设计，不得通过跳过检查达标。
3. **全库只读审计**（离线，如时间格式全量扫描）：≤ **60min**、RSS ≤ **8GB**；
   必须流式/分表进行，禁止全库常驻内存。
4. **测试**：在 `tmp_path` 临时 DuckDB 上运行，禁止触碰 `data/` 下正式库
   （AGENTS §10.1）；单文件测试墙钟 ≤ **120s**。
5. **并发**：正式运行占用数据库期间不得并发写入（AGENTS §5.6）；任何写操作
   必须单独授权。
6. **网络**：正式验证不得依赖网络（AGENTS §10.1）；供应商探针属离线审计，
   必须显式标注并在测量日志中记录其时间与结果。

## 9. 裁决规则

1. **仲裁顺序**：用户本次明确要求 / 已批准契约 → 契约测试 → 实现 → 测量
   日志（AGENTS §1）。测量只能执行本契约的裁决，不能反向改写契约。
2. **p17/p16/p19 优先**：可见性、新鲜度、价格语义与交易约束的口径以已批准
   契约（p17 §6/§12、p16 §2、p19 §2.4）为仲裁依据；本契约不得放宽。
3. **阈值裁决权**：§1.9 的常量是唯一权威；任何变化必须另立契约并由用户批准
   （I12）。
4. **分域 vs 全局的裁决**：B01 保持「按需」原则（I10）；若实现认为某源应成为
   全局门，必须停下另立契约（不得在报告层偷偷升级）。
5. **N/A 语义**：`not_applicable` 与 `unknown` 都不等于 `pass`；报告必须四态
   分明（I9）。
6. **测试争议**：见 §6 末段（AGENTS §10.1）。
7. **审核**：本契约的批准与实现必须分别经独立审核（合规 + 技术/验证），
   审核者不得是起草人或实现者。

## 10. 停止条件（任一命中 → 任务 failed + 完整报告，禁止静默降级）

1. **源语义无法确认即暂停**（补充方案 §6 FO-02 停止条件同款）：若某族的
   经济有效时间/可用时间无法从已批准数据契约确认，标 `unknown` 并暂停相关
   候选，**不得**推定。
2. **不得为让门禁变绿而改数据或改口径**：任何删除困难样本、缩小分母、把
   `unknown` 记为 `pass`、用硬编码涨跌幅补缺、或改写既有 manifest/artifact
   的行为一律红线（AGENTS §4.3、§10.1）。
3. **发现真实数据缺陷**：只报告，**不**自行修复；修复需用户批准的有状态任务
   （AGENTS §5.6）。
4. **无法区分零事件与漏采**：若 `source_ingest_runs` 不足以证明批次成功，
   该族判 `unknown` 并计入缺口；不得默认「零事件」。
5. **资源超限**（§8）且无法在不降低证明强度的前提下收敛：停下。
6. **需要修改本契约的阈值、分母口径或 G1–G8 判定**：停下，提交修订草案。
7. **触碰 inScope 之外的文件**、修改 `AlphaGPT-factor` 工作树、或覆盖用户
   既有修改：立即停止并报告。
8. **出现「为通过测试而弱化断言」的诱因**（例如 R12 的混合格式难以构造）：
   停下并报告，不得删测试或改断言。

## 11. 需用户（或转授后的 captain）明确批准才能实现的条款

| 编号 | 需批准的条款 | 冻结的语义 | 若被否决的后果 |
|---|---|---|---|
| B01-A1 | §3/I1 + I2：导入边界规范化与「禁止混合格式比较」的机器断言 | 时间字段口径与区间比较前提 | 静默失配（118 行 ISO）继续存在 |
| B01-A2 | §3/I3 + I4：四时间语义分离与「不可还原即保留 Tier 限制」 | 可用时间与 Tier | 修订历史可能被推定完整 |
| B01-A3 | §3/I5：月末快照 → 日度区间的 as-of 规则（含月末非交易日处置） | 指数权重的日度可见性 | 月度快照可能被当作日度值 |
| B01-A4 | §3/I6：覆盖率分母 = 已上市 ∧ 目标 PIT universe ∧ 应当可观察 | 覆盖率口径 | 覆盖率可能被分母口径放大 |
| B01-A5 | §3/I7：可交易约束取执行日真实值 + 逐码逐日缺口清单 | 可交易性证据 | 缺口可能被硬编码填平 |
| B01-A6 | §3/I8：零事件 vs 漏采的台账证据要求 | 数据完整性判据 | 漏采可能被当作零事件 |
| B01-A7 | §3/I9 + I10：分域资格报告四态与「按需」原则（N/A 不是通过） | 资格报告的裁决语义 | 可能退化为单一全局绿 |
| B01-A8 | §4.2 的 `DATA_TIER_VERSION` / 新增 gate 契约版本 / `TIER_REPORT_VERSION` 的 bump 方向 | artifact 兼容性分类 | 新旧资格代际不可区分 |
| B01-A9 | §5.1：数据缺陷只报告、修复另走有状态任务 | 写操作边界 | 可能未授权写生产库 |
| B01-A10 | §8：资源上限数值（10min/4GB 热路径；60min/8GB 全库审计） | 工程预算 | 无上限即无法判定「超限即停」 |
| B01-A11 | §10 停止条件（尤其「源语义无法确认即暂停」与「不得改口径变绿」） | 证据完整性红线 | 停止条件不可作为失败裁决依据 |

## 12. 附：本草案的边界核对（落盘时的事实）

- 本文件**只**新增两个改动：`docs/p29_b01_pit_tradable_data_quality_draft.md`
  与 `docs/CONTRACTS.md` 的一行索引（同 commit）。
- **未**修改任何源码、测试、配置或 artifact；**未**运行任何训练、搜索、评估、
  回测、模拟、数据同步或回填；**未** push / 建 PR / 合并。
- **数据探针全部只读**：`duckdb.connect(..., read_only=True)`，仅执行
  `SELECT`/`information_schema` 查询；**未**写入任何数据库（I14）。探针脚本
  位于仓库外的临时目录，未纳入提交。
- **未**修改 `AlphaGPT-factor` 工作树的任何文件。
- **未**自我批准：状态为 DRAFT，正文不含 `APPROVED` 状态行，也不含批准记录行。
- **未**预写任何「实际发生」的测量结果（AGENTS §11.1）；§7 仅为测量方案。
