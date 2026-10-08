# P50 实验完整性与报告状态（WP14）契约

- 状态：**APPROVED**（工作包 14 任务指令 + 仲裁修复清单）。本契约是
  `ashare_model/experiment_integrity/` 包与六份 `tests/test_p50_wp14_*.py` 的规范权威，
  内容定稿自 `reports/wp14/DESIGN.md` §2/§3/§5/§7 与已落地模块/测试行为。
- 落地记录：执行阶段先行落地代码与测试（DESIGN §1 既定路径），本契约文件于
  审计修复阶段补齐（仲裁必改项：约 24 处 § 引用悬空）。语义与已落地行为逐条
  对齐——若本文件与代码行为出现分歧，以本文件为准修复代码，或按 §10 修订
  流程显式修订本文件；二者不得静默分叉。
- 上位依据：`reports/wp14/_recon/{conventions.md, report_schemas.md,
  plan_sources.md}` 三份侦察；`reports/wp14/DESIGN.md`（设计权威，Rev3 记录
  交付偏离）；四份冻结产物 `data/r1/{r01_slow_baseline, r02_matrix,
  r02_periods, r03_combos}/`。
- 红线（本契约处处受其约束）：不改 `AlphaGPT/data/**` 既有文件（唯一允许的
  写入 = r01_slow_baseline 目录内**新增** ERRATUM.md 与 errata.json）；不改
  旧候选/组合定义；不重跑回测（一切输入是已读入的 report dict / plan，纯函数）；
  勘误前瞻规则入档（§7）。

## 1 问题陈述与模块布局

WP14 的任务：对四份既有 r1 产物做**计划解析、完整性核对、状态分类与勘误**，
不引入口径差或二次排名式缺陷（P45/P46 教训在本层的形态 = 绝不把不同经济
对象折叠进一个键、绝不跨实验比较日期日历、绝不比较输入不一致的两侧）。

六项能力需求（R1–R6）与模块布局（`ashare_model/experiment_integrity/` 子包，
全部纯函数、import-light，r0x 模块一律函数体内 lazy import）：

| 模块 | 需求 | 职责 |
|---|---|---|
| `types.py` | — | §2 全部 frozen dataclass、版本常量、异常（零 IO） |
| `plan.py` | R1 | 计划解析与物化（§3.1/§8）：四实验的 L1-L4 来源快照、臂×档×折(×周期)全枚举、verdict_map |
| `precheck.py` | R2 | 数据装载前三类检查（§3.2）：域成员资格 / 依赖与方向定义 / 组合引用完整性 + 臂形状守卫 |
| `reconcile.py` | R3 | 唯一键集合对账（§3.3）：期望−实际、失败层、覆盖层、汇总层重建、provenance 锚定 |
| `status.py` | R4/R5 | 三层状态与终态标签（§3.4/§5）：无自动升级，measurement_incomplete 遮蔽经济标签 |
| （勘误生成） | R6 | ERRATUM.md / errata.json 负载生成与受限写盘（§7）；实际实现 = `reports/wp14/r01_slow_errata.py` 一次性脚本（写面守卫 = 脚本拒绝覆盖 + ER2 目录面快照，见 DESIGN Rev3） |

执行入口（实际形态，DESIGN Rev3 对齐）：`scripts/wp14_integrity.py` 为注册式
stub（`materialize` 可用，其余子命令拒绝式退出 1）；实际执行态 =
`reports/wp14/wp14_run_integrity.py`（只读跑 materialize→precheck→reconcile→status，
写 `reports/wp14/<exp>_status.json` ×4 + `runplan_materialized.json` +
`integrity_findings.md`）。

## 2 值类型与词汇表

全部 frozen dataclass（C4 惯例：不引入 Pydantic/TypedDict）。版本常量
`PLAN/RECONCILE/STATUS/ERRATA_SCHEMA_VERSION = 1`（v1 = WP14 初始；这些是
测量工件，不注册进 `artifact_schemas.py`）。

**词汇表**（单一权威，消费方一律 import，不得复制第二份字面量）：

- `ARM_KINDS = ("single", "combo", "benchmark", "control")`。benchmark/control
  是对照臂：被测量、被报告，绝不作为价值证据参与裁决（r01_baseline.py
  decide_adjudication 按名跳过对照臂的先例）。
- `STATUS_LABELS`（终态标签，§5.4 固定求值顺序）：
  `not_generated` → `measurement_incomplete` → `summary_layer_inconsistent`
  →（三层全真后查表）`economic_negative` / `economic_positive` /
  `inconclusive_mixed`。
- `ARM_MEASUREMENT_STATUSES = ("citable", "program_error", "missing",
  "partial")`：`citable` = 行齐且无失败；`program_error` = ≥1 失败行；
  `missing` = 行全缺；`partial` = 行部分缺。
- `EXPERIMENT_IDS = ("r01_slow_baseline", "r02_matrix", "r02_periods",
  "r03_combos")`。

**RowKey（唯一键，五轴恒全携带）**：`period / cell / candidate / cost_level /
fold_test_end`，实验未启用的轴取中性值（`None`/`""`）。**绝不从键里删轴**——
r02_periods 删掉 period 轴实测把 120 行折叠成 60 键×重数 2（F9）。`seed`
不入键（四产物 695 行全 null，F6）。

**异常分层（C3）**：`PlanValidationError(ValueError)` = 可构造、可定位的计划/
预检违例（消息带 `(experiment, arm, member, fold)` 定位串）；
`IntegrityRefusal(RuntimeError)` = 对账/状态/勘误的 fail-closed 拒绝（输入根本
不可审）。**findings 是数据，不是异常**。

## 3 模块契约

### 3.1 plan.py — R1 计划解析与物化

公开面：`plan_sources` / `materialize_plan` / `expected_keys` / `plan_to_json`
/ `plan_from_json`。

- **物化规则**：计划完全由 L2 config + L3 冻结代码常量重建（§8 取数表），
  绝不读报告自身反推期望（防循环论证）；期望行集 = cross(arms,
  cost_levels, folds)，每臂携带**自身的**成本阶梯（绝无统一阶梯假设）与
  自身的 cell/period 轴值（cash 臂 candidate 无前缀而 cell=参考格、benchmark
  cell=None 但 r02_periods 下按 period 展开，F19）。
- **plan_sources 的承诺（reconsume 义务）**：契约/配置路径 + sha256 快照
  「是对账锚：后续 reconcile 将其与报告 recorded provenance 比对」——该
  比较由 §3.3 的 provenance 锚定实现（RC10/RC11 钉死）。
- **fail-closed（kind 词表）**：`_arm_from_json` 对 `payload["kind"]` 不在
  `ARM_KINDS` 时抛 `PlanValidationError`，消息定位四元组（experiment、
  arm.candidate、kind、expected ARM_KINDS）。理由：所有成员级检查按精确
  kind 字符串分流，词表外 kind 会静默绕过三类检查且告警变少（实测复现：
  15 → 6，涉事臂零告警）。`plan_from_json` 对此类可定位拒绝**原样上抛**，
  不重包成笼统 malformed。
- 违例（未知 experiment_id、配置缺 folds/baseline_signals、代码常量缺失）
  抛 `PlanValidationError`，消息带 `(experiment, locator)`——一切在数据
  装载前发生。

### 3.2 precheck.py — R2 装载前三类检查

公开面：`run_precheck(plan, *, vocab=None, domain_features=None) ->
PrecheckReport`。纯函数、零数据 IO、零 loader import；两个权威
（`vocab.FEATURE_NAMES`、`research_domain.tier_restricted_features`）lazy
读取、可注入（测试与假设性注册表 what-if 不 monkeypatch）。**不抛异常、
不短路**：一次调用跑完全部检查类，返回完整、确定性排序的 issue 清单
（拒绝是调用方按 `report.ok is False` 的决定）。

检查类与 code 词表（完整）：

1. **P-DOM 研究域成员资格**：`member_outside_domain`（信号臂成员 ∉ 运行
   研究域 = KeyError 'MOMENTUM_20' 三十失败行的装载前等价物）；
   `domain_unresolvable`（域不可解析 = 可定位 finding，仅跳过 P-DOM 成员
   循环，绝不当作空域抹黑每个成员）。
2. **P-DEP 依赖与方向定义**：`member_not_in_vocab`；
   `member_missing_direction_evidence`（成员未被任何 single 臂声明 = 方向
   字典必缺）；`unknown_direction_rule`（规则 id 不在 direction.py 权威
   枚举，或 rule_version 与运行时不符）；`missing_direction_rule`（信号臂
   无方向规则；对照臂合法为 None）；`fold_window_invalid`（非空、ISO 日期、
   train<test、按 test_end 严格升序）；`fold_overlap`（测试窗两两不交，
   相邻折合法）。
3. **P-REF 组合引用完整性与臂形状**：`combo_dangling_member`；
   `combo_member_not_in_plan_singles`；`combo_duplicate_member`；
   `cost_ladder_invalid`（阶梯空；benchmark 阶梯 ≠ `("ref",)`）；
   `empty_arm_list`；**`unknown_arm_kind`**（kind ∉ ARM_KINDS——非 JSON
   路径的第二道防线，覆盖直接用 Python 构造的计划；可定位、点名臂）；
   **`comparison_arm_with_members`**（benchmark/control 臂携带非空 members
   ——对照臂按设计在所有成员级透镜之外，是把成员藏进无人阅读处的唯一
   位置；可定位、点名隐藏成员）。

`SIGNAL_ARM_KINDS = ("single", "combo")` 为成员轴承重臂；对照 kinds 由
`ARM_KINDS` **推导**（`COMPARISON_ARM_KINDS`），无第二份手写字面量。

### 3.3 reconcile.py — R3 唯一键集合对账

公开面：`row_key_of` / `reconcile_report(plan, report, run_meta=None)` /
`rebuild_failed_folds` / `reconcile_to_json`。

- **输入身份门（IntegrityRefusal，绝不 finding）**：report 非映射、无 rows
  层、行非映射、schema 家族键缺失/异值/混入外家族键（`_REPORT_SCHEMA_KEYS`：
  r01=3，其余=1）、run_meta 非映射。
- **provenance 锚定（plan_sources 承诺的兑现）**：report 携带
  `provenance.config_sha256` / `provenance.contract_sha256` 时与
  `plan.sources['config'/'contract']['sha256']` 比对——**不符抛
  `IntegrityRefusal`，消息含两侧实测值与 locator**（报告来自与计划不同的
  输入，继续对账 = 跨口径比较，P45 教训）；**字段缺失（任一侧）= 可定位
  finding `provenance_anchor_absent`**（披露而非拒绝）；十六进制摘要
  **大小写不敏感**比较（r0x 写盘方记录大写 config 摘要而 `_sha256_file`
  产小写，四产物实测）。
- **集合层 findings**：`missing_row`（期望−实际）；`duplicate_row`
  （Counter 重数 >1，绝不静默去重）；`unexpected_row`（实际−期望）；
  `unexpected_axis_value`（非活动轴上非中性值——INV5 防折叠的反向守卫）。
- **失败层**（行层权威，F12）：`failed_row`，按 plan 的 `is_primary` 分流
  primary/benchmark 两流；`failure_reason` 缺失时以哨兵串分类。
- **覆盖层（仅限报告内）**：`date_coverage_mismatch`（组内 n_dates 单值、
  daily_dates 向量一致）；`returns_length_mismatch`（n_dates ==
  len(daily_returns)）；`fold_window_overlap`（观测窗不倒置、按 period 两两
  不交）。**跨实验日期比较结构性不可能**：分组键只有本报告的
  (period, fold_test_end)。
- **汇总层重建**：`summary_layer_mismatch`——按 plan sources 三重声明执行
  （`summary_layer_path` / `summary_layer_arms` /
  `summary_layer_period_prefilter`，F20）；行层多出臂域外的臂记入
  `coverage["summary_layer_extra_arms"]` 披露、不报 issue；声明路径未知 =
  fail-visible finding。代码无按 experiment_id 的分类分支。
- `meta_row_count_mismatch`：run_meta.row_count ≠ len(rows)（run_meta 提供
  时）。

### 3.4 status.py — R4/R5 三层分类

公开面：`classify_status(plan, recon, report) -> StatusReport`。判定规则
见 §5。三条结构规则（INV4/INV10 源码级钉死）：

1. `report["completed"]` 在模块内**只出现一次**、只喂 report_generated 层。
2. 求值顺序固定：report_generated → primary_arms_measured → verdict_valid
   → label；`measurement_incomplete` 遮蔽一切经济标签。
3. **recon↔report 强制绑定**（进任何判定前）：experiment_id 相等、
   `n_rows_observed == len(rows)`、`actual_keys` 与 rows 重建键集相等——
   任一不符抛 `IntegrityRefusal`（消息含两侧实测值）。手工全空
   Reconciliation 配真实 completed=true 报告在此被物理拒绝，空真直通经济
   标签的 API 路径不存在。

## 4 CLI 与执行脚本形态

- `AlphaGPT/scripts/wp14_integrity.py`：argparse 子命令注册（materialize/
  precheck/reconcile/status/errata）。**实际形态**：`materialize` 可用；
  其余为拒绝式 stub（退出 1 并说明落点），完整执行态在
  `reports/wp14/wp14_run_integrity.py`。退出码：0 = 产出完成；1 = 拒绝/
  issue；2 = usage。写面守卫：任何位于 `AlphaGPT/data/` 下的输出路径拒绝。
- `reports/wp14/wp14_run_integrity.py`：一次性只读执行脚本——materialize→
  precheck→reconcile→status 四实验全跑，输出 `reports/wp14/` 下五个文件；
  只读证明 = main() 末尾对四份 report/run_meta 的**前后 sha256 比对**
  （任何不一致退出非零），不是名为 `_verify_readonly` 的函数。

## 5 三态语义的精确判定规则（字段级）

反捷径总则：三层是**合取链**，不是默认真。每层证据 dict 列出它读了哪些
字段——未列出的字段一律不读。

### 5.1 report_generated（"报告生成完成"）

| # | 条件 | 读的字段 |
|---|---|---|
| G1 | report JSON 可解析、非空 | （整体） |
| G2 | 实验对应 schema 版本键在且为已知值 | `r01_report_schema_version==3` / `r02_report_schema_version==1` / `r02_periods_report_schema_version==1` / `r03_report_schema_version==1` |
| G3 | `report["completed"] is True` | `completed`（**唯一**读取点，只喂本层） |

`completed=True` 到此为止——它不携带"臂测完"或"裁决可读"的任何信息（F7：
slow 报告 completed=true 且 worth_search=false 同时 30 行程序失败）。

### 5.2 primary_arms_measured（"所有主要臂测量完成"）

True 当且仅当 report_generated **且** recon 中全部成立：

| # | 条件 | recon 字段 |
|---|---|---|
| M1 | 主要臂无缺行 | `missing` 中无键属于 primary 臂 (candidate, period) 组合 |
| M2 | 全报告无重复行 | `duplicates == ()` |
| M3 | 主要臂无程序失败行 | `primary_failed_rows == ()` |
| M4 | benchmark 无失败行 | `benchmark_failed_rows == ()` |
| M5 | 日期覆盖一致 | 无 `date_coverage_mismatch` / `returns_length_mismatch` / `fold_window_overlap` |

非主要臂的程序失败行不阻断本层；它们的**缺行**由 V0 兜底（删 benchmark
全部行的构造场景靠 V0 拦下，而非信任 M4 只查失败行）。

### 5.3 verdict_valid（"研究裁决有效"）

True 当且仅当 primary_arms_measured **且**：

| # | 条件 | recon 字段 |
|---|---|---|
| V0 | 无任何缺行（含非主要臂、benchmark、control） | `missing == ()` |
| V1 | 无计划外行 | `unexpected == ()` |
| V2 | 无非活动轴取值 | 无 `unexpected_axis_value` |
| V3 | 汇总层与行层一致 | 无 `summary_layer_mismatch` |
| V4 | meta 计数一致 | 无 `meta_row_count_mismatch`（run_meta 提供时） |

### 5.4 label（终态分类，固定顺序，先到先得）

1. `not_generated`：report_generated=False。
2. `measurement_incomplete`：M1–M5、V0、V1、V2 任一违例（**含 completed=true
   的情形**）。
3. `summary_layer_inconsistent`：仅 V3/V4 违例。
4. 否则（三层全真）：按 plan sources 声明的 `adjudication_path` 取**裁决
   结局词**（bool 规范化为字面键如 `"worth_search:false"`），查
   `plan.verdict_map`（§8 逐字穷举）→ 经济标签。**fail-closed**：结局词
   不在 verdict_map（词表漂移）→ `IntegrityRefusal`（消息含词、map 键集、
   experiment_id），绝不静默落 `inconclusive_mixed`，绝不裸 KeyError；
   status.py 无任何 verdict 字面比较分支——查表是唯一路径，表穷举 + else
   拒绝配套。
5. `adjudication_quarantined = not verdict_valid`：裁决输出原样保留
   （`adjudication_as_recorded`）但不可作为经济证据读。

### 5.5 经济负结果的判定（M3 纪律）

**负结果条件（全部同时）**：程序无错（M1–M5）+ 臂齐全（V0–V4）+ 结局词
逐字命中 verdict_map 且映射 `economic_negative`。→ `economic_negative`，
不是 incomplete。**UNSTABLE/执行类词不得标负**：r02_matrix 的
`isolated_single_winner` / `multiple_disjoint_viable`（代码 note 明言
UNSTABLE）/ `gross_info_execution_blocked`（续走执行诊断）与 r02_periods
对应词全部 `inconclusive_mixed`；`panel_no_reliable_info` 一词在 r02_matrix
（无格存活且无毛信息 → 信号负）映射 negative、在 r03（等权 2/3 格存活的
混合面板）映射 inconclusive_mixed——同词不同实验不同语义，逐实验逐词按
代码 note 论证，绝无跨实验通配。反之 `measurement_incomplete` 触发时
（任一 M/V 测量缺陷），即使报告内 worth_search=false /
bottleneck="no_qualified_arm…"，也**不得**读作经济负结果——测量不全时
"没有合格臂"是未定，不是否定。

### 5.6 四产物落点（实测锚，非预设）

- **r01_slow_baseline**：30 个主臂失败行 → M3 违例 →
  `measurement_incomplete`；worth_search=false 被隔离；五个单因子 +
  cash + benchmark = citable，两组合 = program_error（行层实测 15+15，
  唯一错误串 `KeyError: 'MOMENTUM_20'`）。
- **r02_matrix / r02_periods / r03_combos**：对账全清洁（310/120/140 行、
  零缺/重/外/失败、覆盖与汇总层一致）→ 三层全真 → 标签取 verdict_map：
  stable_region_found → economic_positive；
  period_panel_no_reliable_info → economic_negative（真负结果判例）；
  equal_weight_stands → economic_positive。

## 6 验收反例矩阵（R7）

每个反例**双断言**：拒绝错误裁决 + 保留可用部分结果。

| # | 场景 | 构造 | 拒绝（RED 断言） | 保留（GREEN 断言） |
|---|---|---|---|---|
| S1 | 错误域成员 | 真实 r01 计划（combo 成员 ∉ slow 域） | precheck 报 `member_outside_domain`×5 + `member_missing_direction_evidence`×5 + `combo_member_not_in_plan_singles`×5（合计 15），locator 定位到 (arm=combo:all3, member=MOMENTUM_20)；ok=False | 五单因子与两对照臂零告警；真实报告对账后五单因子 citable |
| S2 | 缺一个主要臂 | 合成 r01 形报告删 combo:all3 全部 15 行 | missing=15 键全属该臂；label=measurement_incomplete；verdict_valid=False | 其余 110 行照常对账，臂级状态逐臂给出 |
| S3 | 重复一折 | 合成报告复制一行（126 行一键重数 2） | duplicate_row 恰点出该键（重数 2，n_rows_observed=126）；label=measurement_incomplete | 125 键全部可见；无缺行误报；其余臂 citable |
| S4 | 全组合失败单因子成功 | 合成报告复刻真实形态（30 失败行、completed=true、worth_search=false） | label=measurement_incomplete（**非** economic_negative）；adjudication_quarantined=True | 五单因子+cash+benchmark citable；两组合 program_error 且 failure_reasons 逐字保留 |

## 7 勘误文件格式（R6）

写入 `data/r1/r01_slow_baseline/`（目录内**新增**两文件；既有字节零改动；
勘误脚本拒绝覆盖已存在目标；写面守卫 = 目标目录名必须是
r01_slow_baseline 且父链在 data/r1/ 下 + ER2 目录面快照钉死目录恰为
4 文件）。单一事实源两个视图：errata.json（机器）与 ERRATUM.md（人类）
由同一 payload 渲染。

**errata.json schema（`ERRATA_SCHEMA_VERSION = 1`，已落地实例的逐字段）**：

- `target`：experiment_id / report_path / report_sha256 / report_bytes /
  completed_as_stamped / run_meta_path / run_meta_sha256 / run_meta_bytes /
  run_meta_row_count（全部实测自 file digest 与报告顶层）。
- `facts[]`：每条含 `fact` / `value` / `measured_from`（"rows layer" /
  "arm_summaries layer" / "file digest" / "code +
  research_domain.tier_restricted_features"）/ 可选 `note`。四条事实：
  failed_row_count=30（breakdown {combo:all3:15, combo:momentum_reversal:15}）、
  failure_reason_variants=["KeyError: 'MOMENTUM_20'"]、
  citable_single_factor_arms=五单因子、root_cause（域外成员 → 方向字典必缺）。
  **不抄叙述，一切实测**。
- `classification`：failed_rows_class="measurement_program_error" /
  overall_label="measurement_incomplete" / adjudication_quarantined=true /
  authority="WP14 reconcile+status (p50)" / quarantined_as_recorded（冻结
  裁决输出逐字副本：worth_search=false、flags、bottleneck）。
- `forward_rule`（红线四条）：no_rerun / no_redefinition / new_candidate_ids
  （未来 slow 组合必须新候选 ID + 新实验计划，成员取自
  tier_restricted_features('slow_fundamental','AB')）/ precheck_gate。

**ERRATUM.md 骨架**：标题与日期；靶产物（sha256/字节/completed 原样）；
三段事实（30 失败行及分布 / 唯一错误串 / 五单因子可引用），每句尾标
`（实测自产物：<层>）`；归类段（测量程序错误，非经济负结果）；
前瞻规则段（四条）；签名行引用本契约 §7（R6）。

## 8 计划解析策略：四实验来源与物化规则

计划来源四层（P §0）：L1 契约 / L2 配置 / L3 代码常量 / L4 运行记录。
物化取数表（验证数 = 与真实报告 actual 键集合作差实测）：

| 实验 | L2 配置 | L3 代码常量 | 方向规则 | key_axes | 物化规则（期望行集） | 验证数 |
|---|---|---|---|---|---|---|
| r01_slow_baseline | r1_slow_ab.yaml（folds/baseline_signals/domain/tier） | r01_baseline: R01_COMBO_ARMS, COST_LEVELS | reference_return v1，per_fold，learned_at=none | candidate, cost_level, fold_test_end | 5 单因子×{0x,1x,2x} + 2 组合×3档 + cash×3档 + benchmark×{ref}，×5 折 | 8×15+5=125 |
| r02_matrix | r1_medium_a.yaml | r02_matrix: R02_GRID/R02_REFERENCE_CELL/R02_COST_LEVELS + R01_COMBO_ARMS | reference_return v1，per_fold，reference_cell=n20_m5000 | cell, candidate, cost_level, fold_test_end | 6 格×5 信号臂×{0x,1x} + cash（仅参考格，1x）+ benchmark（ref，cell=None），×5 折 | (60+1+1)×5=310 |
| r02_periods | r1_medium_a.yaml | r02_matrix: R02_PERIODS/R02_REFERENCE_CELL/R02_COST_LEVELS + R01_COMBO_ARMS | reference_return v1，**per_(period,fold)**，reference_cell=n20_m5000 | period, cell, candidate, cost_level, fold_test_end | 每 period：5 信号臂(参考格)×{0x,1x} + cash(1x) + benchmark(ref)——cash 与 benchmark 都按 period 展开 | 12×5×2=120 |
| r03_combos | r1_medium_a.yaml | r03_combos: R03_CELLS/R03_REFERENCE_CELL/R03_LAMBDAS/R03_ADJUDICATED_ARMS/R03_MEMBERS/R03_COST_LEVELS | reference_return v1，per_fold，reference_cell=n10_m2500 | cell, candidate, cost_level, fold_test_end | 3 格×4 组合臂×{0x,1x} + 参考格 3 单因子(1x，非主要) + benchmark(ref) | 28×5=140 |

`rule_version` 锚定**运行时代码**的 `DIRECTION_RULE_VERSION`（=1），而非
物化当日值。plan sources 另声明：`adjudication_path`（F22）、
`summary_layer_path` + `summary_layer_arms` + `summary_layer_period_prefilter`
（F20：r02_periods 臂域仅 5 个 n20_m5000:: 信号臂、按 period 预过滤；其余
三实验臂域=全部臂）。

主要臂（is_primary）：r01 = 全部 7 个成本承载信号臂；r02_matrix = 30 个
cell::信号臂；r02_periods = 5 信号臂×2 period；r03 = R03_ADJUDICATED_ARMS
（combo:all3 + combo:fitted_lam050），描述性 λ 臂与参考格 singles 非主要。

**verdict_map（plan 数据，逐字穷举，无通配；未命中 = §5.4 #4 fail-closed
IntegrityRefusal）**——同词 `panel_no_reliable_info` 在 r02_matrix→negative /
r03→mixed 的分野不是笔误，两处代码 note 语义不同：

```python
VERDICT_MAPS = {
    "r01_slow_baseline": {
        "worth_search:true": "economic_positive",
        "worth_search:false": "economic_negative",
    },
    "r02_matrix": {
        "stable_region_found": "economic_positive",
        "isolated_single_winner": "inconclusive_mixed",   # UNSTABLE
        "multiple_disjoint_viable": "inconclusive_mixed", # UNSTABLE
        "gross_info_execution_blocked": "inconclusive_mixed",
        "panel_no_reliable_info": "economic_negative",
    },
    "r02_periods": {
        "stable_period_region": "economic_positive",
        "isolated_period_winner": "inconclusive_mixed",   # UNSTABLE
        "gross_info_execution_blocked_period_panel": "inconclusive_mixed",
        "period_panel_no_reliable_info": "economic_negative",
    },
    "r03_combos": {
        "fitted_increment_region": "economic_positive",
        "equal_weight_stands": "economic_positive",
        "combo_no_net_space": "economic_negative",
        "panel_no_reliable_info": "inconclusive_mixed",
    },
}
```

## 9 不变量清单与测试映射

测试为六个分阶段文件（非单一文件，DESIGN Rev3 记录），各用独立编号避免
碰撞；映射如下（INV 编号 = DESIGN §9 原编号）：

| 不变量 | 断言要点 | 落点测试 |
|---|---|---|
| INV1 | r01 报告 sha256/字节冻结（95fac732…f550b / 2049495） | test_p50_wp14_errata.py ER1、reconcile/acceptance 的 digest 锚 |
| INV2 | data/ 写面唯一（目录外拒绝 + 拒绝覆盖 + ER2 目录面=恰 4 文件） | test_p50_wp14_errata.py ER2（替代控制：勘误脚本拒绝覆盖，DESIGN Rev3） |
| INV3 | R01_COMBO_ARMS 逐字冻结；包内无 r0x 写路径 | test_p50_wp14_errata.py ER6 |
| INV4 | completed 不自动升级（字面量单读取点） | test_p50_wp14_status.py ST11/ST2 |
| INV5 | 唯一键不折叠经济对象（period 轴 + unexpected_axis_value） | test_p50_wp14_plan.py PL4、reconcile RC6 |
| INV6 | 纯函数/不重跑（源码扫描 import 面；全合成 payload） | test_p50_wp14_precheck.py PC12 等各阶段 |
| INV7 | 物化计数对账（125/310/120/140，missing=duplicate=unexpected=0） | test_p50_wp14_plan.py PL3/PL4、reconcile RC1 |
| INV8 | 真负结果不被吞（干净+worth_search:false → economic_negative；UNSTABLE 词 → mixed） | test_p50_wp14_status.py ST1/ST9、acceptance ACC2 |
| INV9 | 汇总层可由行层重建（含 r02_periods 按 period 预过滤/臂域） | test_p50_wp14_reconcile.py RC2/RC8 |
| INV10 | recon↔report 绑定（孤儿对账拒绝） | test_p50_wp14_status.py ST10 |
| R6 | 勘误主张钉死（30/15+15/唯一串/五单因子/forward_rule 四条款/measured_from） | test_p50_wp14_errata.py ER3–ER7 |
| R7 | §6 反例矩阵 S1–S4 端到端 | test_p50_wp14_acceptance.py |
| PC13/PC14 | kind 词表 fail-closed + 对照臂藏成员可定位 | test_p50_wp14_precheck.py |
| RC10/RC11 | provenance 锚定（不符拒绝含两侧值 / 缺失披露 finding / 大小写不敏感） | test_p50_wp14_reconcile.py |

## 10 修订

本契约的任何语义修订（新 code、新标签、版本 bump）必须：先改本文件并
记录修订节；同步落测试（先 RED 后 GREEN，证据入 `reports/
p50_red_evidence.txt`）；代码与测试在同一次变更中对齐。版本常量的 bump
注释是 schema 演进的唯一提示面。
