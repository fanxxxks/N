# P28 A04 测量日志（evidence-only）

- 契约：`docs/p28_a04_feature_metadata_semantic_type_draft.md`（APPROVED，批准记录回写见文件头）
- 实现分支：`codex/a04-feature-metadata-evidence`（base = main @ `aac4aad`）
- 本日志只写**实际发生**的测量（AGENTS §11.1）；契约在 `docs/p28_..._draft.md`。

## 1. 命令与环境

- 环境：Windows（PowerShell），Python 3.13.12，CPU 6C/12T / 16GB RAM，本地串行/pytest 受控并行（`-n 4`），未使用 GPU。
- 基线重测（只读探针，2026-09-14，分支 `codex/a04-feature-metadata-evidence` 创建后、实现前）：

```
slots 83 | deprecated 13 | samplable 70 | operators 39
FRV 14 | GRAMMAR 9 | RDV 5 | SEMCACHE 2 | FACTCACHE 2
type-hist(all83) {'return_like': 19, 'volume_like': 17, 'price_like': 5, 'fundamental_like': 23, 'boolean_event_signal': 9, 'cross_sectional_signal': 10}
MARGIN_BALANCE_TO_CIRC_MV | type fundamental_like | deps ('rzye', 'circ_mv') | pit pit_capital | fam external | avail quarterly-report 模板
MARGIN_BUY_PRESSURE_5     | 同上（deps ('rzmre', 'circ_mv')）
```

- 与契约 §1.9 的对照：槽位/可采样/算子/FRV/GRAMMAR/RDV 全部一致；**偏差**：`SEMANTIC_CACHE_VERSION`、`FACTOR_CACHE_VERSION` 实测为 **2**（§4.2 草案写的 1 已被 A02 消费，授权记录 §11 版本总表「A02/A04」归属一致）。按契约「实现前必须重测、以重测为准」处理（修订 2 已登记）。
- 版本 owner 全量检索（AGENTS §3.2 强制动作，实现前在 base 上重跑）：

```
main:ashare_model/data_tier.py:41:DATA_TIER_VERSION = 2
main:ashare_model/factor_cache.py:65:FACTOR_CACHE_VERSION = 2
main:ashare_model/factors.py:41:FACTOR_COMPUTE_VERSION = 4
main:ashare_model/feature_registry.py:85:FEATURE_REGISTRY_VERSION = 14
main:ashare_model/research_domain.py:54:RESEARCH_DOMAIN_VERSION = 5
main:ashare_model/semantic_cache.py:43:SEMANTIC_CACHE_VERSION = 2
main:ashare_model/vocab.py:317:GRAMMAR_VERSION = 9
```

## 2. RED 证据（commit `480d9f5`）

- 命令：`python -m pytest tests/test_p28_a04_feature_metadata.py -q`
- 结果：**23 failed / 9 passed**（7.27s）。
  - 23 个失败 = 契约违规本体（R1/R3 联合可见性缺失、R2 单位未声明、R4 nan 语义未声明、R5 类型翻转未发生、R6 语义缓存键无代际、R7 类型未修正、R8 依赖口径混用、R9/R10 清单缺失）。
  - 9 个通过 = 预期守卫（R4 行为半、R5 checker↔pset 全枚举一致性、R6 legacy 解码、R7 单型算子事实、R8 禁混装、R11、R12、R13、R14）——非目标钉值，纠偏前后都必须绿。
- 分片注册：`python scripts/check_test_shards.py --check` → `100 files across 4 shards cover all 100 test files`（新测试文件同 commit 注册进 shard-data）。

## 3. FO-01 清单交付

- 文件：`docs/feature_registry.md` §4「FO-01 存量契约清单」（由 `scripts/generate_registry_docs.py` 生成，非手工名单）。
- 生成命令：`python scripts/generate_registry_docs.py`；drift guard：`python scripts/generate_registry_docs.py --check` → `feature_registry.md is in sync with the registries`（exit 0），并有 `tests/test_registry_docs.py` + `tests/test_p28_a04_feature_metadata.py::test_r10_manifest_is_generated_and_drift_guarded` 双守卫。
- 覆盖计数（清单 `counts` 字段，生成时实测）：`feature_slots=83`、`samplable=70`、`deprecated=13`、`operators=39`。
- 逐对象字段：特征 = name/vocab_index/family/domain/pit_level/tier/semantic_type/availability_rule/unit/nan_policy/depends_on/promotion_allowed/implementation_status/research_eligibility/evidence_status/deprecated；算子 = name/vocab_index/arity/category/inputs/output/cost/stability。
- 四状态全部**派生**自单一权威（NEUTRAL_FEATURE_NAMES / DEPRECATED_FEATURE_NAMES / promotion_allowed），无第二名单。派生规则：evidence_status ∈ {preregistered_hypothesis, measurement_withdrawn, source_discontinued}。

## 4. 合法动作集合前后对照（§7.3）

类型判定依据（I3 + t6 裁决 2）：F2 的 loader 侧输出**已经是逐日横截面 z**（`data_loader.apply_cross_sectional_z`，p24 §2 预注册），按 I3 规则 → `CROSS_SECTIONAL_SIGNAL`。grammar 不 bump（未增删 token；只改算子合法输入类型）。

| 项 | 修正前（FUNDAMENTAL_LIKE） | 修正后（CROSS_SECTIONAL_SIGNAL） |
|---|---|---|
| `CS_NEUTRALIZE(F2)` | 非法（CS_NEUTRALIZE 输入限 `(_XS,)`） | **合法**（唯一解锁的新表达；全 39 算子中唯一限单输入型的算子，`test_r7_cs_neutralize_is_the_only_single_type_op` 钉值） |
| `ADD(F2, PE_TTM/LEVERAGE_CHANGE)`（同 fundamental 族算术） | 合法 | **非法**（family-preserving 要求同型） |
| `ADD(F2, IND_REL_RET_5/IND_REL_*)`（同截面族算术） | 非法 | 合法 |
| `ADD(F2, TURNOVER 等 volume 族)` | 非法 | 非法（跨族仍禁止，语义未变） |
| F2 作为根特征 | 合法 | 合法（ROOT_TYPES 不含 boolean，二者均在根集） |
| 混合族 DIV/CORR/GATE 条件位 | 合法 | 合法（DIV/CORR 跨连续族；GATE 条件位任意型——未受影响） |

- 一致性（I8/R5）：`test_r5_signature_checker_matches_typed_pset_everywhere` 对 39 个算子 × 全部类型组合做 checker↔GP 类型化 pset 的**双向穷举等价**（零失配）；RL/Random/TPE 的动作掩码与 GP 消费同一 `semantic_sampling` 层（`type_id`/`signature_legal`），既有 `test_semantic_sampling.py` 的 Random 采样合法性与 GP 类型树测试继续全绿。
- 面板级搜索行为未重跑：本包为元数据/语义类型纠偏，不涉及任何搜索运行（资源上限 §8.3 的固定面板对照以上表 + 穷举等价测试承载；未消耗语义评价预算）。

## 5. 数值等价性（I9/R12）

- 纠偏**零改动**计算文件：`ashare_data/capital_flow.py`、`ashare_model/data_loader.py`、`ashare_model/factor_formulas.py`、`ashare_model/factors.py`、`ashare_model/ops.py` 均不在 diff 中（可由实现 commit 的文件清单核对）。
- 行为见证：`tests/test_capital_flow.py::test_f2_frames_raw_ratios`（既有钉值 0.16/0.3）继续通过；新增独立 oracle `test_p28_a04_feature_metadata.py::test_r12_factor_values_are_invariant`（rzye=200/400、circ_mv=800 → 0.25/0.5，手算）通过；`apply_cross_sectional_z` 手算 z 值（±1）通过。
- `FACTOR_COMPUTE_VERSION` 不变（4）；factor 张量缓存身份含 `FEATURE_REGISTRY_VERSION` → FRV 14→15 使旧缓存条目自动失配（`load_cache.py`），无陈旧张量复用路径。

## 6. 声明-来源对照表（§7.5；p17 §5.2/§6.1、p19 §2 逐条）

| 因子 | 声明（修正后） | 来源证据 |
|---|---|---|
| MARGIN_BALANCE_TO_CIRC_MV | availability = margin_detail_pit T+1（available_date）AND daily_basic_pit 当日（trade_date 收盘已知）；type=cross_sectional_signal；unit=ratio（rzye[元]/circ_mv[万元]，常数 1e4 尺度被逐日 xcs-z 吸收——z 对正尺度变换不变）；deps=(margin_detail_pit.rzye, daily_basic_pit.circ_mv)；nan=missing_nan（缺源保持 NaN，绝不预填 0） | p17 §5.2 表 18（rzye/rzmre=元）+ 表 4（circ_mv=万元）；p17 §6.1 T+1；p24 §2（xcs-z 预注册）；`capital_flow.py:528-559`（原始比率）+ `data_loader.py:648-651`（xcs-z）+ `data_loader.py:143`（NaN 恢复） |
| MARGIN_BUY_PRESSURE_5 | 同上，分子为 rzmre 的 5 个可见会话滚动和（T+1 使第 k 列承载第 k 会话明细） | 同上 + `capital_flow.py:544-559` |
| MARGIN_BALANCE_CHG | deps 由 `("margin_detail_pit",)` 收紧为 `("margin_detail_pit.rzye",)`（I4 口径统一；消费字段实测 rzye）；unit=return；type/可见性不变 | `capital_flow.py:516-525`（wide=rzye pivot）；测试钉值更新属白名单第 2 类（已批准 p28 I4），断言更强（字段级可审计） |
| MARGIN_CROWD_60 | 同上（deps 收紧为 rzye） | `capital_flow.py:518,526`（wide 复用） |

### 登记的后续任务（§9.7，本包不顺手修）

1. **`_FUNDAMENTAL_TTM` 模板与 D6 实现的文案错配**：D6 实际为逐行 `available_date` 的公告级可见（`pit_import.asof_winner_intervals`，p17 §6.2 winner 语义），而模板文案写「法定披露季结束后可见」（Q1 04-30 等）。该模板现仍被 F1 四因子（CFO_TO_ASSETS/GROSS_PROFITABILITY/OPERATING_PROFITABILITY/LEVERAGE_CHANGE）与 family-⑤ 四因子使用——声明比实际**偏保守**（晚于真实可用），但按 I1 的仲裁口径（p17 §6.2）文案不准确。属研究语义声明修订，须另立契约修订案后统一改为公告级可用性表述。
2. **fundamental 家族 depends_on 的表级口径**（如 `("fundamental_pit",)`）向 `表.字段` 迁移——需逐因子核对 p13 字段清单，独立小任务。
3. **F3 四因子（AlphaGPT-factor WIP）复用季度模板的同模式问题**：按授权记录 §11bis 登记为 FO-02 输入，本包未触碰该工作树。

## 7. 版本影响表（实现 commit 实测重述）

| 版本所有者 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `feature_registry.py` FEATURE_REGISTRY_VERSION | 14 | **15** | 注册表记录的语义字段（类型/可用性/依赖口径）变更，改变可搜索表达式（t6 裁决 2）；RunSpec 版本集合与 artifact 分类承载代际 |
| `vocab.py` GRAMMAR_VERSION | 9 | 9（不变） | 未增删 token/未改顺序与规范化（t6 裁决 2/8） |
| `semantic_cache.py` SEMANTIC_CACHE_VERSION | 2 | **3** | 键布局变化：`SemanticCacheKey` 新增 `feature_registry_version`（必填，无默认）；旧代际条目不可比亦不可复用（p28 §4.2 的「纳入键」分支） |
| `factor_cache.py` FACTOR_CACHE_VERSION | 2 | 2（不变） | 缓存身份已含 `FEATURE_REGISTRY_VERSION`（`load_cache.py`），FRV bump 自动失配 |
| `factors.py` FACTOR_COMPUTE_VERSION | 4 | 4（不变） | 数值计算零改动（I9 有行为见证） |
| `research_domain.py` RESEARCH_DOMAIN_VERSION | 5 | 5（不变） | 域归属未动（t6 裁决 8） |
| `data_tier.py` DATA_TIER_VERSION | 2 | 2（不变） | PIT level/tier 映射零改动（R13 钉值） |
| `runspec.py`/protocol/reward/promotion/search 全组 | — | 不变 | 无所有权关系；`feature_registry_version`/`semantic_cache_version` 已在 `_VERSION_IMPORTS` owner 索引（t6 裁决 7 满足，无新增版本常量） |

## 8. 测试与验证记录

- 聚焦集（实现后）：`test_p28_a04 + test_factor_semantic_fixes + test_feature_metadata + test_feature_registry + test_semantic_cache` → **92 passed**。
- 相邻契约环：`test_semantic_sampling + test_gp_search + test_vocab + test_data_tier + test_factors + test_capital_flow + test_registry_docs + test_research_domain + test_load_cache` → **165 passed**（131s）。
- 全量：`python -m pytest -q tests -n 4` → 结果见 §10 追加段（实现 commit 工作树上执行；首个 `-n auto` 尝试因磁盘耗尽中止，与 A03 记录的磁盘事件同类，已改用受控并行度重跑）。
- `git diff --check`：实现 commit 前执行，无空白错误。
- 既有测试修改清单（§10.1 白名单，全部第 2 类——已批准 p28 契约的需求变更，无断言弱化）：
  1. `test_feature_metadata.py` / `test_feature_registry.py`（×2）/ `test_semantic_cache.py` 的版本钉值 14→15、2→3：FRV/SEMANTIC_CACHE bump 即契约内容本身；注释引用 t6 裁决 2。
  2. `test_factor_semantic_fixes.py:288` deps 钉值 `("margin_detail_pit",)`→`("margin_detail_pit.rzye",)`：p28 I4 口径统一；新断言更强（字段级）。
  3. 新测试文件内的 RED 期精化（unit 受控词表 `price_adj` 定稿、MARGIN_BALANCE_CHG=return、xcs-z 容差 1e-6）：见契约修订 2 ④，均非弱化。

## 9. 未运行项与研究声明边界

- 未重跑任何搜索/训练/正式评价（本包不涉及）；未运行 `scripts/check_production_gates.py`/`research_doctor`（无数据/universe 改动；G1–G8 与 D7 判定零改动，R13 有钉值）。
- 未 push、未建 PR、未合并（用户未授权；仅本地提交）。
- 研究声明边界：A04 是元数据/语义纠偏与可核对清单交付；其测量**不构成** alpha、收益或晋级证据。p24 的族级裁决（候选池准入）维持原值；「单因子有效」仍无任何登记（I10/R14 钉值）。
- 剩余风险：①F1/family-⑤ 的 `_FUNDAMENTAL_TTM` 文案错配已登记未修（后续契约修订）；②`AlphaGPT-factor` 的 F3 WIP 合并时须按 t6 裁决 8 对账版本起点（FRV 目标 15→16 等）；③FO-01 清单的 unit 词表是本轮 authored 声明，其逐因子源尺度链以本日志 §6 对照表为审计基准。

## 10. 全量测试追加段（实现 commit 工作树上实测）

- 命令：`python -m pytest -q tests -n 4`（受控并行；首个 `-n auto` 尝试因磁盘峰值耗尽中止——与 A03 记录的磁盘事件同类，改用 `-n 4` 重跑完成）。
- 结果：**1819 passed / 2 failed / 6 skipped**（2097.91s ≈ 35min）。
- 2 个失败均为 `tests/test_lock_files.py`（`test_pins_match_installed_environment`、`test_freeze_lock_check_passes`），根因 = 本机解释器缺 `tushare` 包元数据（`importlib.metadata.distribution('tushare')` 抛 `PackageNotFoundError`，实测复现于解释器本身）→ `freeze_lock.py --check` exit 1。归属：**pre-existing 环境失败**，与 A03 最终证据（commit `5b6812e`：「2 pre-existing env failures」）记录的同一对环境失败一致；本包 diff 未触碰任何依赖/锁文件（`git diff --name-only` 无 requirements/lock/pyproject/package 文件）。原始输出：`reports/alphagpt-ab/_a04_full_suite.txt`（未跟踪）。
- 6 个 skipped：既有基线 skip，无净新增（与 main 基线一致）。
