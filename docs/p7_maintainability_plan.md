# P7 可维护性治理主计划：巨型编排模块拆分 + 类型化产物 + 单一注册表 + 语义类型

状态：已批准基线（用户裁决 2026-07-XX 见 §0 决策记录），按小 PR 串行执行。
任务类型：Phase A/B/D 为工程变更（§3.1，parity 保护）；Phase C 为 artifact schema
变更（需迁移/拒绝策略）；Phase E 为研究语义变更（§3.2，单独预注册契约）。

本文是整个 P7 倡议的目标、非目标、阶段边界与验收来源。各行为变更阶段在动代码前
另有独立契约文档（见 §4/§6），仲裁顺序按 AGENTS.md：契约 > 测试 > 实现。

## Plan Header

- **Goal**：治理两个巨型编排模块（`evaluation.py` 2340 行、`train.py` 1577 行），
  按"变化原因"拆分；用 Pydantic schema 替代 strategy/protocol/backtest/paper state
  边界的松散 dict；建立因子+算子单一元数据注册表并令文档由注册表生成；为 GP 引入
  语义类型收缩无意义搜索空间。
- **Architecture**：保持 AGENTS.md §9 依赖方向不变；拆分只在 `ashare_model` 内部
  进行，原文件保留兼容 facade，不新增跨层依赖、不反向依赖。
- **Tech Stack**：Python 3.13、pytest、Pydantic v2（新增显式生产依赖，Phase C0
  单独 PR 处理 `.in`/lock/`pip check`）、torch（既有）。
- **Baseline/Authority Refs**：AGENTS.md（§2/§3/§9/§10/§14）、
  `docs/p6_research_domain_contract.md`（契约格式先例）、
  `ashare_model/artifact_versions.py`（legacy 分类单一权威，P7 不另建第二套）。
- **Compatibility Boundary**：Phase A/B 默认路径逐字节兼容——全部既有 import
  （`from ashare_model.evaluation import ...`、`from ashare_model.train import ...`，
  含 tests/scripts/webapi 的 40+ 处）不改；monkeypatch 面
  （`evaluation.CandidateScorer`、`evaluation.REWARD_VERSION`、
  `evaluation._build_trainer`、`train_module.batched_basket_rewards`）保持
  运行时经 facade 命名空间查找。Phase C 新 schema 只增不删字段；legacy 产物
  继续走 `artifact_versions` 分类/打戳路径，只读可审计，永不冒充 current。
- **TDD Route**：
  - Mode: auto（用户全局规则"先写契约和失败测试"构成本倡议的严格授权）
  - Decision: strict（Phase C/D/E 行为承载切片）；Phase A/B 机械移动为
    post-change regression + 导入面契约测试（A1 先写，红在"新模块不存在"，
    绿在移动完成且 facade 不变）
  - Strict authority: explicit user/project request（用户规则 + AGENTS.md §10.1）
  - Test posture: strict RED test（C/D/E）；diagnostic parity（A/B）
  - Verification: 见 §8 每阶段命令序列
- **Verification**：每 PR 全量 `python -m pytest -q tests` + compileall +
  `git diff --check`；不得用局部测试绿代替全量。

## 0. 决策记录（用户裁决）

| 决策点 | 裁决 |
|---|---|
| schema 实现 | Pydantic v2，显式加入 `requirements.in` 并同步 lock（Phase C0 独立 PR） |
| W4 语义类型 | 纳入本倡议，排最后阶段，动代码前先提交 `docs/p7_semantic_types_contract.md` |
| 基线提交 | 只提交 `AGENTS.md`；`docs/p5_implementation_plan.md` 保持未跟踪（用户私人草稿） |

## 1. 目标与非目标

目标（四条工作流，串行）：

- **A. 拆分 `evaluation.py`**：按变化原因抽出 folds/metrics/corrections/search/artifacts
  五个模块，原文件成为薄 facade（保留 `run_protocol`/`main`/版本常量/re-exports）。
- **B. 拆分 `train.py`**：抽出窗口/采样自由函数；`AshareTrainer` 类本体保留在
  facade（monkeypatch 面最重、RL 内核是内聚单元）。
- **C. 类型化产物**：strategy/protocol/backtest/paper state 四处边界使用 Pydantic
  schema，含 schema version、必填 provenance、legacy migration、reject policy、
  字段类型与默认值；消灭边界上的静默 `payload.get(..., None)`。
- **D. 单一元数据注册表**：扩展 `FeatureRecord`（数据 Tier、可用时间/滞后规则、
  经济假设、推荐预测周期、预期方向、输入语义类型、是否允许 promotion、计算成本、
  依赖字段）+ 新建算子注册表；文档由注册表生成，带漂移守卫测试。
- **E. GP 语义类型**：PriceLike/ReturnLike/VolumeLike/FundamentalLike/
  CrossSectionalSignal/Boolean-EventSignal；类型来自注册表（D 的字段），约束
  采样合法性，收缩无经济含义的搜索空间。

非目标（本倡议不做）：

- 不改变任何指标、Reward、成本、mask、执行、晋级语义（A–D 纯工程；E 的语义
  变化仅限其独立契约声明的搜索空间合法性）。
- 不重构无关模块、不顺手改名/格式化（§9"机械重构与语义变化分开提交"）。
- 不删除 legacy 产物；不重跑正式 campaign；不以本倡议结果声称任何 alpha。
- UI/webui 消费注册表为非目标（除非 webapi 已有注册表端点，接线成本极小，
  届时在 D3 内评估并显式记录决定）。
- `webapi/service.py` 的 24 处 `.get(` 治理为非目标（属 UI 防御性降级面，
  与正式研究边界分开处理，留待后续契约）。

## 2. 基线与停止条件

基线：`main @ 5a7c2b3`，工作区仅两个未跟踪文件（按 §0 处理）。
全量测试基线在 Phase 0 记录通过数/警告数，写入测量日志，后续每 PR 对比。

停止条件（任一命中即停并汇报）：

1. 全量测试出现非预期红，且 30 分钟内无法定位为本次改动引起；
2. 发现拆分必须改变任一既有语义才能继续（说明切分边界设计错误，回到计划）；
3. 既有测试断言与计划冲突——按仲裁顺序回契约判断，禁止默认改测试；
4. 任一 PR 的 `git diff` 越出本任务文件清单。

## 3. Phase A：拆分 evaluation.py（工程，parity）

目标模块图（按变化原因命名，`eval_` 前缀保持同包内聚）：

| 新模块 | 变化原因 | 迁入符号 |
|---|---|---|
| `eval_folds.py` | 折叠/窗口契约变化 | `Fold`、`FoldData`、`resolve_folds`、`epoch_slice`、`search_window_id` |
| `eval_metrics.py` | 指标/拼接语义变化 | `evaluate_signal`、`evaluate_formula`、`benchmark_row`、`_tradable_ic_mask`、`aggregate_results`、`stitch_oos_series`、`stitched_metrics`、`top_trial`、`_algorithm_of` |
| `eval_corrections.py` | 统计校正方法变化 | `_poly`、`norm_ppf`、`norm_cdf`、`psr`、`expected_max_sr`、`deflated_sharpe`、`_trial_stats`、`_stitched_pool`、`_dsr_from_pool`、`dsr_from_rows`、`_max_t_from_pool`、`max_t_from_rows`、`selfcheck_rows` |
| `eval_search.py` | 搜索后端接入变化 | `baseline_candidates`、`_build_trainer`、`run_fold`、`_SearchWindow`、`_build_search_window`、`_search_evaluator`、`_search_failed_row`、`_search_row`、`run_random_search`、`run_gp_search`、`run_tpe_search` |
| `eval_artifacts.py` | artifact schema 变化 | `_sanitize`、`universe_policy_payload`、`_regime_payload`、`_data_tier_block`、`build_result`、`_run_recorded` |
| `evaluation.py`（facade） | 编排/CLI 变化 | `PROTOCOL_VERSION`、`run_protocol`、`load_trial_rows`、`main` + 全部 re-export |

**Monkeypatch 兼容规则（硬约束）**：被测试 patch 的名字必须经 facade 运行时
查找——`eval_search` 内对 `_build_trainer`、`CandidateScorer` 的引用、
`eval_artifacts` 内对 `REWARD_VERSION` 的引用，改为函数体内惰性
`from ashare_model import evaluation as _facade` 后 `_facade.X`。被抽模块之间
直接 import 非 patch 面名字（确定性、无环：folds→metrics→corrections→search→
artifacts 单向，facade 在最顶）。

PR 切片（每片一个分支 `p7/aN-*`，串行合并回 main）：

- **A0** `p7/00-baseline`：提交 `AGENTS.md`；建 `tests/test_eval_module_split.py`
  导入面契约测试（当前全部公开名字可从 `ashare_model.evaluation` 导入、
  无循环导入）——此时即绿，作为后续每片的 parity 守卫。记录全量测试基线。
- **A1** 抽 `eval_folds.py`；**A2** `eval_metrics.py`；**A3** `eval_corrections.py`；
  **A4** `eval_search.py`；**A5** `eval_artifacts.py`。
- 每片步骤相同：verbatim 移动 → facade re-export → patch 面惰性查找 →
  最小相关测试（test_evaluation/test_stitched_oos/test_promotion/test_research_domain）
  → 全量 pytest → compileall → `git diff --check` → 合并。

验证（每片）：`python -m pytest -q tests`；`python -m compileall -q ashare_data
ashare_model ashare_portfolio ashare_trading scripts webapi`；`git diff --check`。
parity 证据：全量绿 + A0 导入面测试绿 + 不作任何语义编辑（diff 只含移动与
import 行）。

## 4. Phase B：拆分 train.py（工程，parity）

- **B1** `train_windows.py`：`_TrainWindow`、`validation_start`、
  `validation_windows`、`sample_random_formulas`、`resolve_device`、`_project_root`。
  facade re-export；`tests/test_train.py` 等对 `train_module` 的 patch 面
  （`batched_basket_rewards`）不动——`AshareTrainer` 留在 `train.py`。
- **B2** 评估后决定（先 grep 测试对私有方法的直接调用）：若 `_write_artifact`/
  `_build_rl_search_result`/`_training_contract` 无外部直接调用，抽为
  `train_artifacts.py` 模块函数 + 类内薄包装；有调用则放弃 B2 并记录理由
  （不允许为拆分破坏测试面）。B2 动代码前在本文件补一节切分清单。

验证同 Phase A。`AshareTrainer` 的 RL 内核不拆（内聚单元，"按变化原因"而非
按行数）。

## 5. Phase C：类型化产物（schema 变更）

动代码前先提交 `docs/p7_artifact_schema_contract.md`（字段清单、schema version、
provenance 必填项、migration/reject 矩阵、对 `artifact_versions.py` 分类规则的
影响）。切片：

- **C0** 依赖 PR：`requirements.in` += `pydantic`（必要性：四处正式边界的
  fail-closed 校验，webapi 已传递依赖同包；固化直接依赖防传递漂移）→
  `python scripts/freeze_lock.py` → `pip check` → `freeze_lock.py --check`。
- **C1** `ashare_model/artifact_schemas.py`：Strategy/Protocol schema + 写入侧
  （`build_result`/trainer artifact）fail-closed 校验 + 读取侧 reject 矩阵
  （current 校验、legacy 走既有分类、未知 version 拒绝）。
- **C2** backtest 结果与 `ashare_trading` paper state schema + `run_sim` 接线。

每片 strict TDD：先写"缺 provenance/未知版本被 reject、legacy 仍可读且被标
legacy、current round-trip"的红测试。

## 6. Phase D：单一注册表（工程 + 版本 bump）

- **D1** 扩展 `FeatureRecord` 九字段，`FEATURE_REGISTRY_VERSION` 2→3，默认值与
  unknown 显式策略；62 个特征的元数据逐一著录（内容工作，机械但需逐条核对
  数据源文档）；artifact 中 registry 版本引用同步。
- **D2** `operator_registry.py`：算子元数据单一来源（输入/输出语义类型、成本、
  数值稳定性备注）；`OPS_CONFIG` 保持实现/arity 来源，注册表只加描述性元数据，
  不复制实现参数（防第二权威）。
- **D3** `scripts/generate_registry_docs.py` 生成 `docs/feature_registry.md` +
  漂移守卫测试（生成物与注册表不一致即红）；README/onboarding 链接生成物，
  手工名单删除。

D1/D3 strict TDD（新字段缺失即校验红；文档漂移即红）。D1 的版本 bump 与迁移
策略在动代码前补入本文 §6 一节（轻量契约，不另立文档——字段为纯描述性
metadata，不进搜索合法性判定，搜索语义不变；若执行中发现要进判定，升级为
独立契约）。

## 7. Phase E：GP 语义类型（研究语义变更）

先提交 `docs/p7_semantic_types_contract.md`（§3.2 全要素：问题、假设、不变量、
版本 bump——grammar/search contract、legacy 公式策略、预算公平性、预期 RED
测试、测量方案、裁决规则、停止条件）。类型格与每特征/算子标注来自 Phase D
注册表字段，不建第二份类型表。属性测试须覆盖：类型合法剪枝的确定性、
legacy 公式按名解析不受影响、搜索预算计数公平性不漂移。先后比较禁止：
类型上线前后的搜索结果不得宣称 matched comparison。

## 8. 测量与汇报

每 Phase 结束更新 `docs/p7_measurement_log.md`（命令、commit、passed/skipped/
warnings、墙钟、产物路径、未运行项及原因）。最终汇报按 §14 区分：软件正确性
（测试）、工程测量（行数/模块边界/导入面）、研究证据（本倡议除 Phase E 的
契约测量外不产生研究结论）、生产就绪度（无声明）。

## 9. 风险与回滚

| 风险 | 缓解 |
|---|---|
| facade 拆分破坏 monkeypatch 面 | A0 导入面契约 + 每片全量测试；patch 面惰性查找硬约束 |
| 移动中手滑改语义 | verbatim 移动；diff 只含移动/import；逐 PR 小步 |
| pydantic 依赖固化引发 lock 冲突 | C0 独立 PR，失败即停，不挟带其他改动 |
| 62 特征元数据著录错误 | D1 逐族核对 `factors.py`/`fundamentals.py`/`capital_flow.py` 数据源；review 按族分批 |
| Phase E 搜索空间变化被误当工程变更 | 独立预注册契约 + 版本 bump + 不可比声明 |

回滚：每 PR 单点职责、串行合入 main，任一 PR 可单独 revert；legacy 产物只读
不动，无状态迁移风险。

## 10. Retirement

- Phase A/B：无删除；facade 即兼容层，长期保留（导入面即公共 API）。
- Phase C：strategy/protocol/backtest/paper state 的松散 dict 写入路径退役
  （读取侧 legacy 分类保留，永不删除）。
- Phase D：README/onboarding 中手工维护的特征名单退役，由生成物替代；
  漂移守卫测试防止第二权威复活。
