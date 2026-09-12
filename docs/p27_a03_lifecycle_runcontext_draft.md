# P27 A03 真正强制的生命周期与 RunContext 预注册契约

- 状态：**APPROVED — 批准记录回写（实现包 A03 按本文件头分工条款执行）**。
  批准人：captain（依用户转授权原文「所有行为我强制要求你来替我批准，」，
  见 `reports/alphagpt-ab/00_authority_exception_record.md` §1）；批准决策
  记录：同记录 **§11（t6）七条争议点裁决第 1 条**（A03 动作模式表 +
  「None 放行仅限非正式模式」＝A03-A1/A2 **批准**）及 **§11 版本影响总表**
  （`LIFECYCLE_CONTRACT_VERSION` 1→2 归属 A03）；用户确认：同记录 **§4ter
  第 4 项**（2026-09-10，含裁决 1–8 全部条目，用户保留逐条撤销权）。批准
  引用范围：`5efc79c` → `1578e07` → `2996cdb` → `37f7609`（契约正文 SHA）。
  起草人（governance-author）无权自改状态；本状态行由实现包随实现 commit
  写入并引用上述记录，符合 §11「DRAFT→APPROVED 的改写由各实现包在自身的
  inScope 内、随实现 commit 一并完成」的分工条款。
- **批准权属**：本契约的 DRAFT→APPROVED 状态变更**只能由用户明确批准**
  （AGENTS §3.2；`docs/p18_captain_authority_clause.md` §3.2 明确 captain
  原本无权批准）。本团队存在用户转授批准权的例外记录
  （`reports/alphagpt-ab/00_authority_exception_record.md` §3.2，授权原文
  「所有行为我强制要求你来替我批准，」），据此 captain 可在 §5 逐契约决策记录
  填写完成后代为批准；该转授**不是**起草人可自我批准的依据。**起草人
  （governance-author）无权批准本契约，也无权追认。** 无批准原文时本契约保持
  DRAFT，依赖它的实现任务（T14/T15）必须停在等待态。
- 起草：governance-author（AgentTeams `alphagpt-remediation-ab` t3/T02），
  2026-09-10。
- 起草依据：
  - 用户任务指派（T02 起草 A03 预注册契约，DRAFT，关闭 F03）；
  - `reports/AlphaGPT_整改与提升执行方案_2026-09-09.md` §2.2 F03、§三
    「RunContext」设计、§四 A03 工作包；
  - AGENTS.md §1/§2.2/§3.2/§4.1/§4.2/§4.3/§4.6/§11/§12/§13；
  - `docs/p8_research_lifecycle_contract.md`（生命周期唯一契约，APPROVED，
    §2 身份层次 / §3 证据矩阵 / §4 合法转换 / §5 资格门禁 / §6 受保护 OOS /
    §8 旧产物与 account / §9 激活边界）；
  - `docs/p7_artifact_schema_contract.md`（类型化产物边界）；
  - `docs/p26_a02_dataset_snapshot_identity_draft.md`（A03 的数据身份前置，
    DRAFT——见 §2.1 假设 1）。
- **勘察基线（docs/CONTRACTS.md「勘察基线惯例」[02-INT-09]）**：起草时
  **main @ `1fb1b16faef30a68332b9e07059c8632fb80ee59`**（`1fb1b16`）。本文
  全部代码事实、行号与版本值在该 commit 上核对；工作树当时含 A02 草案提交
  `185a086`（docs-only）与未跟踪 `reports/`（用户工作，未被本任务触碰）。
  数据/运行事实基线 = `reports/AlphaGPT-audit-2026-09-09/boundary_probes.json`
  （`gate_without_ids` / `doctor_lifecycle`）与
  `reports/AlphaGPT-audit-2026-09-09/research_doctor.json`。
- 本文使用 AGENTS.md 的「必须 / 禁止 / 应当」语义。任务路由：**研究语义变更**
  （AGENTS §3.2：PIT mask、数据 tier、晋级阈值、受保护 OOS）+ **运行态/晋级**
  （§4.1、§4.3 fail closed）+ **数据/股票池变更**（§5）。仲裁顺序：用户本次
  明确要求 / 已批准契约 → 契约测试 → 实现 → 测量日志。

## 0. 六要素自检（AGENTS §3.2 要求项 → 本文落点）

| AGENTS §3.2 要求项 | 本文落点 |
|---|---|
| 问题陈述 | §1（逐条引用 F03 证据行） |
| 假设 | §2.1 |
| 范围与非目标 | §2.2 / §2.3 |
| 不变量 | §3（I1–I13） |
| 版本变化 | §4（含 `git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"` 全量检索结果） |
| 迁移/拒绝策略 | §5 |
| 预期 RED 测试 | §6（R1–R16） |
| 测量方案 | §7 |
| 资源上限 | §8 |
| 裁决规则 | §9 |
| 停止条件 | §10 |
| （附加）需批准条款清单 | §11 |

## 1. 问题陈述（F03 证据，逐条实测于 main @ `1fb1b16`）

1. **生命周期检查已被调用，但调用点没有传递身份（F03 本体）**。
   `ashare_model/lifecycle.py:1096-1123` `assert_formal_run_lifecycle(entry_type,
   data_dir, spec_id=None, run_id=None)` 的第三、四参数都有默认值，且
   `lifecycle.py:1122-1123` 在 `run_id is None or spec_id is None` 时**直接
   `return None`**（注释：「no lifecycle identity → pre-activation legacy
   run」）。三个正式调用点全部使用两参形式：
   - `ashare_model/train.py:94` `assert_formal_run_lifecycle("train",
     data_config.data_dir)`（在 `AshareTrainer.__init__` 内）；
   - `ashare_model/evaluation.py:712-714`
     `assert_formal_run_lifecycle("evaluation", data_config.data_dir)`（在
     `main()` 内）；
   - `ashare_trading/run_sim.py:118`
     `assert_formal_run_lifecycle("paper_sim", data_config.data_dir)`（在
     `SimulationRunner.__init__` 内）。
   三处都必然命中 `return None` 分支：**检查函数被调用了，但它证明不了任何
   生命周期约束**。方案 §2.2 F03 原文：「这几处调用本身不能强制检查实际运行
   的生命周期。」
2. **运行身份在昂贵计算之后才建立（时间顺序倒置）**。三个入口的实际顺序是
   先装载/训练/评价，再冻结 RunSpec 并开 RunStore 运行：
   - `evaluation.py:700` `loader.load_data()` → `:712` 生命周期门（无身份）→
     `:735-749` `resolve_runtime_runspec(...)` → `:751` `store =
     RunStore(data_config.data_dir)` → `:753` `with store.open_run(spec) as
     handle:`；
   - `train.py:94` 生命周期门（无身份）→ `train.py:101-103` loader 装载 →
     `train_artifacts.py:173-196`（`if not trainer.loader.dataset_id:` 守卫 +
     `resolve_runtime_runspec`，即**产物写出时**才冻结 spec）→
     `train_artifacts.py:207-208` `RunStore(...)` + `open_run(spec)`；
   - `run_sim.py:118` 生命周期门（无身份）→ `run_sim.py:274-302`
     `_bind_lineage_run()` 才 `reconstruct_runspec_from_lineage` +
     `open_run(spec)`。
   `train_artifacts.py:192-195` 的注释自述：「The clean-tree SPEC_LOCKED
   evidence gate activates with the lifecycle stages (P8-06+)」——即**门禁尚未
   接线**。方案 §2.2 F03 原文：「训练保存产物时才建立 RunSpec/RunStore 的
   路径，也与『执行前锁定研究计划』的目标存在时间顺序问题。」
3. **`assert_paper_account_writable("current")` 是硬编码实参（同一缺口）**。
   `run_sim.py:119` 与 `research_doctor.py:845` 都以字面量 `"current"` 调用
   `lifecycle.py:1161-1172`；该函数只在 `account_kind != "current"` 时抛错，
   因此这两处调用**恒真**。方案 §2.2 F06 原文：「其 lifecycle 检查调用
   `assert_paper_account_writable("current")`，并没有读取实际账户分类或
   绑定。」A03 负责入口侧（必须消费真实账户身份）；doctor 侧的分层报告归
   A05。
4. **受保护 OOS 的入口保护没有接线到正式 CLI**。
   `lifecycle.py:1074-1093` `assert_protected_oos_access(state, run_type)`
   已实现（research/promotion + 至少 `SEARCH_PLAN_ADMITTED`），但其调用点只有
   `tests/test_run_store.py:1008-1015`——**生产路径零调用**。方案 §四 A03
   失败测试明确要求「在禁止状态读取受保护窗口」必须被拒绝。
5. **没有「动作模式」这一层概念**。全仓检索 `RunContext` 在 `*.py` 中零命中
   （唯一出现是 A02 草案的文档文字）。当前能区分运行类型的只有
   `artifact`/`ledger` 里的零散字段（`run_type` 仅出现在
   `lifecycle.py:1074` 的参数名与 `protocol`/`promotion` 的少量调用），没有
   「工程探针 / 历史审计 / 正式研究 / 受保护 OOS / 正式 paper」这五类动作的
   统一声明，也没有「某动作允许哪些生命周期状态」的机器表。方案 §四 A03
   实施项 2 要求把动作分类固定下来。
6. **既有测试把 None 放行钉成了契约**。`tests/test_run_store.py:1419-1421`
   三条断言 `assert_formal_run_lifecycle("train"|"evaluation"|"paper_sim",
   data_dir) is None`，函数 docstring（`:1406-1408`）自述「Pre-activation
   runs (no lifecycle events) pass through with None」。方案 §2.2 F03 原文：
   「现有 P8-15 选择测试本次 **4 passed**，其中测试明确接受缺少身份时返回
   None。这说明需要修正行为规格及入口覆盖，不能只继续增加『函数存在/文案
   存在』的测试。」按 AGENTS §10.1，这属于白名单第 2 条（需求确实变更），
   实现前必须先批准契约。
7. **「被拒绝的运行留下什么」尚无规格**。方案 §四 A03 验收要求：「被拒绝
   运行可以留下管理/拒绝事件，不能留下成功业务产物。」当前
   `LifecycleWriter`（`lifecycle.py:1184-`）只有 `record_idea` /
   `lock_spec` / `qualify_data` / `fail` / `retire` 等状态转换入口，没有
   「入口拒绝」这一事件类别，也没有「拒绝后不得写出 strategy/protocol/
   backtest/paper 产物」的机器约束。
8. **边界探针证据**：`reports/AlphaGPT-audit-2026-09-09/boundary_probes.json`
   的 `gate_without_ids` 三项（`train` / `evaluation` / `paper_sim`）**全部为
   `null`**——这正是 `assert_formal_run_lifecycle` 在缺身份时的返回值，实测
   坐实了 §1.1。

## 2. 假设、范围与非目标

### 2.1 假设

1. **A02 已批准并实现**（A03 的正式依赖，方案 §四依赖表 A03 ← A01、A02）：
   正式入口能获得「内容确认通过的复合数据身份」。若 A02 未批准/未实现，
   A03 不得以「身份稍后补」的方式绕过（§10 停止条件 6）。
2. `docs/p8_research_lifecycle_contract.md` 定义的状态集、证据矩阵与合法转换
   **不变**；A03 只改变「谁、在什么时刻、以什么身份」调用既有状态机。
3. `RunStore.open_run` / `RunHandle` / `LifecycleWriter` 的既有语义与
   append-only、hash-chained 事件链不变（`lifecycle.py:210-211` AGENTS §4.6）。
4. 正式入口集合可由源码穷举：`train.py`、`evaluation.py`、`backtest.py`、
   `run_sim.py`、`promotion.py`、`research_doctor.py`（诊断，非正式）、
   `webapi/service.py`（展示/触发）。A03 的 RunContext 必须覆盖全部正式入口。
5. 机器 6C/12T、16GB RAM、串行运行；A03 不得引入需要全库/全窗常驻内存的
   新身份计算。
6. 本契约的实现可在**不移动、不删除、不重写**既有 run 目录、lifecycle 事件
   链、artifact 的前提下完成；旧 run 一律按 legacy 只读处理。

### 2.2 范围

1. **RunContext 的单一实现**（新增或扩展现有对象，禁止第二套身份派生）：
   承载「已冻结 spec、run 身份、动作模式、store 根、预算/恢复身份」。
2. `ashare_model/lifecycle.py`：入口门禁的签名与语义（真实身份必需）、
   动作模式 → 允许状态表、入口拒绝事件类别（若需要）。
3. `ashare_model/runspec.py`：RunContext 的 spec 冻结与身份派生消费点。
4. 正式入口接线：`ashare_model/train.py`、`ashare_model/train_artifacts.py`、
   `ashare_model/evaluation.py`、`ashare_model/backtest.py`、
   `ashare_trading/run_sim.py`、`ashare_model/artifact_writer.py`。
5. `ashare_model/artifact_schemas.py` / `artifact_versions.py`：产物身份必须
   来自执行期上下文（若需要新字段）。
6. 相应测试与 `docs/CONTRACTS.md` 索引（本草案已同 commit 登记）。
7. 可拆成训练 / 评价 / 模拟三个子提交（方案 §四 A03 范围注）。

> **勘误 1（2026-09-12，用户裁决"全部按推荐进行"追认，实现包 A03
> 回写）**：§2.2 范围 4 的文件清单在实践中补充两个接触面——
> `ashare_model/data_loader.py`（I6 必需：身份先于 `load_data` 的
> `confirm_input_snapshot` 抽取 + `frozen_data_cutoff`，行为保持）与
> `ashare_model/train_entry.py`（新文件，`test_train_module_split`
> 强制的 train.py 300 行 facade 预算所迫）。另据用户裁决 1a，本契约
> §2.1 假设 4 与本节范围 4 的内部矛盾按用户决定解决：**promotion.py
> 的消费侧生命周期检查已由 A03 后续小包落地**
> （`lifecycle.assert_qualified_source_run` + promotion.main 接线，
> ≥ `SEARCH_PLAN_ADMITTED`，R7/I10 消费侧）；promotion 是只读消费，
> 不取单写者锁，故不经 `RunContext.attach`（R13 对 promotion.py 的
> 断言因此仍然成立，消费侧检查为其接替约束）。

### 2.3 非目标

1. **不**新建并行状态机：状态、转换与证据矩阵仍由 p8 唯一定义；A03 只做
   「接线 + 动作模式表 + 身份强制」。
2. **不**改变 p8 §5 的资格门禁机器标准（Data Qualified / Factor Set
   Qualified / Search Plan Admitted 的判据文本与阈值）。
3. **不**改变 A02 的数据身份派生（A03 只消费）。
4. **不**实现 A05 的分层 doctor / 动作资格报告（`research_doctor.py` 的
   HEALTHY 语义与报告结构属 A05）；A03 只保证 doctor 消费的入口身份为真。
5. **不**实现 B02 的 canary / CI 门禁、B03 的持久执行与中断恢复（A03 只
   要求 RunContext 携带恢复身份，不实现恢复语义）。
6. **不**迁移、重写或删除既有 legacy run / 事件链 / 产物；legacy 一律只读。
7. **不**改动 `ashare_data/gates.py` G1–G8 与 `ashare_model/promotion.py`
   G1–G7 的判定逻辑（两个编号空间不混用）。
8. **不**触碰 `AlphaGPT-factor` 工作树的未提交工作；**不**修改 inScope 外文件。
9. **不**预写任何「实际发生」的测量结果（AGENTS §11.1）。

## 3. 不变量（I1–I13）

**I1（单一 RunContext）**：训练、评价、回测、模拟、晋级五类正式入口必须接收
**同一类型**的 RunContext；禁止各入口各自拼装「接近的」身份对象或继续以
裸参数（`data_dir` + 字面量）表达身份。

**I2（身份必需，None 不再是正式语义）**：正式入口调用生命周期门禁时，
`spec_id` 与 `run_id` 必须是**真实、已建立**的值。`assert_formal_run_lifecycle`
的「缺身份 → 返回 None → 放行」分支只允许存在于**显式非正式模式**（工程探针 /
历史审计），且该模式必须由 RunContext 的动作模式显式声明；正式模式缺身份一律
fail-closed 拒绝。

**I3（动作模式分类固定为五类）**：动作模式 = 工程探针（engineering）、
历史审计（audit）、正式研究（research）、受保护 OOS（protected_oos）、
正式 paper（paper）。五类的**允许状态表**（本文 §3/I4）是唯一权威；未登记的
动作模式一律 fail-closed 拒绝（AGENTS §4.3）。

**I4（动作 → 允许状态表）**：

| 动作模式 | 允许的生命周期状态 | 允许的输入/输出 |
|---|---|---|
| engineering（工程探针） | 任意状态**只读**；不消费受保护 OOS；不写 current 产物 | 可装载、可跑小预算；产物必须标 `engineering`，不得进入 ledger 正式行或晋级证据 |
| audit（历史审计） | 任意状态**只读** | 可读 legacy/current 产物与事件链；**禁止**写任何 current 产物、禁止晋级、禁止写入 paper 账户 |
| research（正式研究） | `SEARCH_PLAN_ADMITTED` 或之后（`train` 阶段入口见 §3/I5 的最低状态） | 可写 current 产物；受 §3/I8 的 OOS 约束 |
| protected_oos（受保护 OOS） | `SEARCH_PLAN_ADMITTED` 或之后，且 `run_type ∈ {research, promotion}` | 仅此模式可评价锁定 OOS 窗口（复用 `assert_protected_oos_access`，`lifecycle.py:1074-1093`） |
| paper（正式 paper） | **恰好** `PAPER_OBSERVING`，且账户为运行上下文绑定的可写账户 | 可追加 paper 观察数据；禁止跨账户写入 |

**I5（搜索前资格完整，装载 ≠ 搜索）**：正式搜索（`evaluation` 的协议/搜索
路径、`train` 的搜索路径）在开始前必须满足「数据资格 + 因子集资格 + 搜索计划
准入」的完整证据链；单纯的数据装载/检查（工程探针、`--selfcheck` 类干跑）
属**不同动作**，必须显式声明为 engineering/audit，不得继承正式研究身份。
`train` 入口的最低状态与 `evaluation` 入口的最低状态在实现中必须分别固定并
写进本文 §6 的断言（不得用一个笼统阈值替代）。

**I6（身份先于昂贵计算）**：计划冻结（RunSpec 解析）与运行身份建立必须发生
在 loader 装载、搜索、评价、回测、账户写入**之前**；禁止「先算完再补身份」。
「昂贵计算」的机器定义 = 任何触发 `load_data` / 搜索 / 评价 / 回测 / 账户
写入的调用。

**I7（产物身份来自执行期上下文）**：写出的 strategy / protocol / backtest /
paper 产物必须携带**执行期间**由 RunContext 派生的 `spec_id`/`run_id`/
`candidate_id`/`account_id`；禁止执行后补造、禁止从别的 run 复制、禁止用
`new_run_id()` 临时值替代上下文身份（`backtest.py:480/487` 目前的
`new_run_id()` 只用于日志身份行，正式产物身份必须来自上下文）。

**I8（受保护 OOS 隔离）**：受保护 OOS 窗口只能由 `protected_oos` 模式在
`SEARCH_PLAN_ADMITTED` 及之后评价；engineering/audit 模式在**任何**状态都
不得读取受保护窗口（AGENTS §4.6、p8 §6）。当前生产路径零调用
`assert_protected_oos_access` 的状态必须被修正。

**I9（账户绑定）**：paper 写入的目标账户必须来自 RunContext 绑定的真实账户
身份（`account_id`），禁止硬编码 `"current"`；跨账户写入（上下文账户 ≠ 目标
账户）一律拒绝（p8 §8）。

**I10（legacy 不隐式穿透）**：legacy 产物/事件链只能在显式 audit 模式读取；
禁止正式入口隐式穿透 legacy 后生成 `current` 产物。仅拥有新 schema **不**
构成生命周期资格（p8 §9）。

**I11（拒绝的留痕边界）**：被拒绝的运行可以留下管理/拒绝事件（append-only），
但**不得**留下成功的业务产物（strategy/protocol/backtest/paper 状态文件）。
拒绝必须发生在 loader/搜索/账户写入之前，并留下可归因的机器记录。

**I12（单一身份实现）**：RunContext 的身份派生必须复用既有唯一实现
（`ashare_model/identity.py` 的 `content_hash`、`RunSpec.spec_id`、
`run_store` 的 `run_id`/`account_id`）；禁止第二套 hash 或第二套身份路径
（AGENTS §1 原则 3、§9）。

**I13（恢复身份一致）**：RunContext 携带的恢复/续跑身份（若存在）必须与
RunStore 的 `(spec_id, run_id)` 及 `spec.to_payload()` 逐字节一致；
不一致必须拒绝（`artifact_writer.reconstruct_runspec_from_lineage`
`artifact_writer.py:68-111` 的既有严格性不得放松）。B03 的恢复语义在此之上
实现，A03 只保证身份一致。

## 4. 版本变化

### 4.1 全量检索基线（AGENTS §3.2 强制动作）

命令（在 main @ `1fb1b16` 上执行）：

```
git -C D:\minequant\AlphaGPT grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"
```

结果：**39 命中**（38 个常量赋值 + `ashare_model/alphagpt.py:8` docstring 与
`tests/test_searcher_bench.py:4` docstring 各 1 处引用）。逐字清单与 A02 草案
`docs/p26_a02_dataset_snapshot_identity_draft.md` §4.1 **同一次检索**，此处不
复制第二份（AGENTS §11.5：能指针化就不复制）；实现前必须**重新运行**该命令，
因为版本值可能已被其他工作包推进。

### 4.2 版本影响表（逐项：旧值 → 新值 / bump 或不 bump 理由）

| 版本所有者 | 旧值 | 新值 | bump / 不 bump 理由 |
|---|---|---|---|
| `ashare_model/lifecycle.py` `LIFECYCLE_CONTRACT_VERSION` | `1` | **`2`（建议 bump）** | A03 改变的是生命周期**门禁的强制语义与调用契约**（真实身份必需、动作模式表、入口拒绝边界），虽不改状态集与转换图，但对「同一 run 在什么条件下被允许进入某入口」的裁决已变。bump 使 artifact/日志能区分「v1 门禁可空转」与「v2 门禁强制」两个代际。若实现证明可在不改变任何裁决的前提下完成（不可能——I2/I5 就是裁决变化），则须在实现 commit 重新举证不 bump 的理由。 |
| `ashare_model/runspec.py` `RUNSPEC_SCHEMA_VERSION` | `1` | **视字段决定** | 若 RunContext 需要把 `action_mode` 写进 RunSpec（顶层字段新增），按 `artifact_schemas.py:41-44` 约定 bump 并同步 `LEGACY_SCHEMA_VERSIONS`；若动作模式只存在于执行期上下文、不进入 RunSpec payload，则不 bump。**注意**：动作模式若进入 RunSpec，会改变 `spec_id`（`runspec.py:301-305` 对全部字段做内容哈希），因此旧 run 的 spec_id 不会与新 run 混同——这一点必须在实现 commit 显式说明。 |
| `ashare_model/artifact_schemas.py` `ARTIFACT_SCHEMA_VERSION` | `2` | **视字段决定** | 只有当产物顶层字段集合变化（例如新增 `action_mode` 或 `account_id` 语义收紧）才 bump；新增必须同时更新 `LEGACY_SCHEMA_VERSIONS` 与读写矩阵。 |
| `ashare_model/versions.py` `PROTOCOL_VERSION` | `"27"` | **不变** | 评价协议（fold、指标、统计校正）零改动；入口资格不是协议的一部分。 |
| `ashare_model/reward.py` / `alphagpt.py` / `vocab.py` / `feature_registry.py` | `"15"` / `3` / `9` / `14` | **不变** | 零改动（A04 的范围）。 |
| `ashare_model/promotion.py` `PROMOTION_RULE_VERSION` | `"2"` | **不变** | G1–G7 判定逻辑不变；A03 只保证进入晋级入口时身份为真。 |
| `ashare_model/data_tier.py` / `research_domain.py` | `2` / `5` | **不变** | 分层与域划分零改动。 |
| `ashare_model/ledger.py` `LEDGER_SCHEMA_VERSION` | `1` | **视需要决定** | 若「入口拒绝」需要新的 ledger 行类别，按既有 schema 约定处理；若复用既有 `fail` 事件则不 bump。实现 commit 必须择一举证。 |
| `ashare_model/run_store.py` `INDEX_SCHEMA_VERSION` | `1` | **不变** | RunStore 索引结构不变；A03 只消费其既有 API。 |
| `ashare_model/observation_lineage.py` `OBSERVATION_LINEAGE_VERSION` | `1` | **不变** | paper 观察链的单调追加语义不变（B03/C03 范围）。 |
| `ashare_model/factor_cache.py` / `semantic_cache.py` | `1` / `1` | **不变（A03 范围内）** | 其身份来源由 A02 裁定；A03 不改变缓存键语义。若 A03 的实现意外改变了缓存键，按 AGENTS §3.1 停下重分类。 |
| `ashare_model/data_eligibility.py` 无版本常量 | — | — | D7 资格判定不变（B01 范围）。 |
| 其余 26 个 `*_VERSION`（见 §4.1 指针） | 见 `docs/p26…` §4.1 | **不变** | 与生命周期入口身份无所有权关系；不因 A03 变更。 |

**版本影响表使用约束**：本表是**草案的预期**，不是实现授权。实现 commit 必须
重新运行 §4.1 检索、逐项复核本表，并对每一处偏离给出理由（AGENTS §3.2）。

## 5. 迁移/拒绝策略

1. **不迁移历史**：既有 run 目录、lifecycle 事件链、artifact 一律只读保留
   （I10）。旧 run 无事件链者按 p8 §9「跨激活 run 的处置」归类 legacy，
   只能 audit 只读；需要继续时以**新 run_id** 重跑（AGENTS §4.3）。
2. **拒绝而非降级**：
   - 正式模式缺 `spec_id`/`run_id` → 拒绝（不再 `return None`）。
   - 动作模式未登记 → 拒绝。
   - 动作模式与当前生命周期状态不匹配（§3/I4）→ 拒绝，错误信息必须含
     `entry_type`、`action_mode`、当前状态、允许状态集合与解除条件。
   - 受保护 OOS 在禁止状态/模式被读取 → 拒绝（复用 `assert_protected_oos_access`）。
   - 上下文账户 ≠ 目标 paper 账户 → 拒绝。
   - 拒绝必须发生在 loader/搜索/账户写入之前（I6、I11）。
3. **legacy 显式化**：现有 `return None` 语义整体迁移为「显式 audit/engineering
   模式」。迁移后，`assert_formal_run_lifecycle` 的「无事件链」情形在正式模式
   必须拒绝；在 audit/engineering 模式返回 `None` 且**必须**在 artifact/日志
   留下 `lifecycle = legacy|unverified` 机器标记。
4. **一次性对账**：实现前先输出只读对账报告（现有 run 目录中哪些有事件链、
   哪些是 legacy、各自状态），作为迁移范围的证据；报告为 evidence-only，不得
   预写结论。
5. **失败即停**：若接线后发现真实 run 因缺身份被拒绝且无法在不改写历史的前提
   下补齐，任务以 `failed` + 完整报告收尾；**禁止**为让旧 run 通过而伪造
   lifecycle 事件或改写 `spec_id`/`run_id`（AGENTS §4.3、p18 §4.3）。
6. **回滚**：实现必须可整体回滚（保留旧签名可用或提供显式非正式开关）；
   回滚后系统回到「缺身份即放行」的旧行为，且不得丢失已写入的事件链。

## 6. 预期 RED 测试清单（先失败、再实现；AGENTS §2.2 第 3 步）

**全部必须从 CLI/service 边界触发**，并断言失败发生在 loader/搜索/账户写入
之前（方案 §四 A03 失败测试原文）。断言来源是本节条款，不得引用实现输出。

| # | 场景 | 期望行为（RED 断言） | 不变量 |
|---|---|---|---|
| R1 | **缺 id**：正式入口以无身份上下文调用 | 正式模式必须拒绝（不再返回 None）；`assert_formal_run_lifecycle` 的正式分支不再接受缺身份 | I2 |
| R2 | **spec/run 不匹配**：上下文 spec_id 与 RunStore 中该 run 的 spec 不一致 | 拒绝；`reconstruct_runspec_from_lineage` 的严格性不放松 | I12、I13 |
| R3 | **缺事件日志**：run 目录存在但无 lifecycle 事件链，且模式为 research | 正式模式拒绝；audit/engineering 模式返回 None 且留下 legacy 标记 | I2、I10 |
| R4 | **非法状态**：run 处于 `IDEA`/`SPEC_LOCKED` 却调用 `evaluation` 入口 | 拒绝（状态低于最低要求） | I4、I5 |
| R5 | **终止运行被继续**：run 处于 `REJECTED`/`FAILED`/`RETIRED` 后再次进入正式入口 | 拒绝（终态无出边） | I4 |
| R6 | **跨账户 paper 写入**：上下文账户 ≠ 目标账户 | 拒绝 | I9 |
| R7 | **当前 schema 但无资格**：产物 schema 为 current 但无生命周期证据 | 拒绝（schema ≠ 资格） | I10 |
| R8 | **禁止状态读取受保护窗口**：engineering/audit 模式在任意状态读取锁定 OOS | 拒绝；`assert_protected_oos_access` 必须出现在生产路径 | I8 |
| R9 | **身份先于昂贵计算**：以可观测桩记录调用顺序 | 冻结 spec/建立身份必须早于 `load_data`/搜索/账户写入 | I6 |
| R10 | **动作模式未登记**：传入未知动作模式 | 拒绝 | I3 |
| R11 | **产物身份来自上下文**：故意让上下文身份与产物身份不一致 | 拒绝；产物必须携带上下文派生的 `spec_id`/`run_id`/`candidate_id` | I7 |
| R12 | **拒绝不留成功产物**：构造被拒绝的正式运行 | 只允许管理/拒绝事件，不得出现 strategy/protocol/backtest/paper 产物 | I11 |
| R13 | **单一 RunContext**：静态/导入面检查 | 五个正式入口接收同一类型上下文，无第二套身份拼装 | I1、I12 |
| R14 | **装载 ≠ 搜索**：工程探针/干跑动作 | 不得继承正式研究身份，不得产出可晋级证据 | I5 |
| R15 | **`paper_sim` 精确状态**：run 处于 `PAPER_OBSERVING` 之外 | 拒绝（`paper_sim` 要求恰好该状态，`lifecycle.py:1142-1148` 语义保留） | I4 |
| R16 | **硬编码账户消除**：`run_sim` 路径 | 不得以字面量 `"current"` 代替真实账户身份 | I9 |

**既有测试的处置（AGENTS §10.1 白名单第 2 条）**：
`tests/test_run_store.py:1419-1421` 的三条 `is None` 断言、
`test_p8_15_lifecycle_no_bypass_assert_formal_run_lifecycle` 的
docstring 声明（`:1406-1408`）、以及任何依赖「缺身份放行」的测试，实现
commit **必须**逐条列出：所属白名单条目、引用本文条款、旧行为为何错误、
新断言为何不更弱。**禁止**删断言、加 skip、把 expected 改成 actual。

## 7. 测量方案（只写方案；真实数值必须由实现后的 evidence-only 提交写入测量日志）

测量日志文件（实现后建立，本契约不创建）：
`docs/p27_a03_measurement_log.md`。

1. **命令与环境**：逐条记录命令、commit/tree 状态、OS/Python、CPU/内存、
   run 根目录与 RunStore 布局。
2. **接线顺序证据**：以可观测桩记录 train/evaluation/backtest/run_sim 四条
   路径的实际调用顺序（身份建立 → 门禁 → 装载/搜索/账户写入），逐条对照
   §3/I6。
3. **拒绝路径证据**：R1–R16 各自在 CLI/service 边界上的拒绝输出全文，含被
   拒绝的 run 身份与解除条件。
4. **既有 run 对账**：现有 run 目录中「有事件链 / legacy 无事件链」的清单与
   各自状态（只读扫描），作为迁移范围证据。
5. **测试影响**：受影响测试的 diff 与白名单举证（§6 末段）。
6. **未运行项**：任何因资源/环境未运行的验证必须显式列出（AGENTS §10.2）。
7. **研究声明边界**：A03 属运行态/治理变更，其测量**不得**被表述为 alpha、
   收益或晋级证据（AGENTS §3.1、§4.1）。

## 8. 资源上限

1. **运行环境**：6C/12T、16GB RAM、串行（p21 §5）。
2. **身份建立与门禁**（入口热路径）：新增的 RunContext 构建 + 生命周期重放
   必须 ≤ **5s**、峰值 RSS ≤ **512MB**（事件链为 append-only 文本，重放成本
   线性）；超出即视为设计缺陷，停下重新设计，**禁止**通过跳过门禁达标。
3. **对账扫描**（离线）：一次全量 run 目录扫描 ≤ **5min**、RSS ≤ **1GB**。
4. **测试**：新增/修改测试必须在 `tmp_path` 临时目录运行，禁止触碰
   `data/` 下的真实 run 与 DuckDB（AGENTS §10.1）；单文件测试墙钟 ≤ **120s**。
5. **并发**：同一 `(spec_id, run_id)` 的单写者约束由 RunStore 锁承载
   （`run_store.py:371-390`），A03 不得引入绕过锁的写入路径。

## 9. 裁决规则

1. **仲裁顺序**：用户本次明确要求 / 已批准契约 → 契约测试 → 实现 → 测量
   日志（AGENTS §1）。
2. **p8 优先**：状态集、转换图、证据矩阵以 `docs/p8_research_lifecycle_contract.md`
   为唯一权威；本契约只固定「入口接线 + 动作模式 + 身份强制」，冲突时以 p8
   的状态机定义为准、以本契约的入口约束收紧执行。
3. **动作模式的裁决权**：§3/I4 的表是唯一权威；新增动作模式必须修订本契约，
   不得在实现里私自扩展。
4. **`None` 的裁决**：`None` 返回值只能表示「显式非正式模式下的 legacy」，
   不得被任何正式路径消费为放行依据。
5. **测试争议**：见 §6 末段；认为测试有误的一方承担举证责任（AGENTS §10.1）。
6. **审核**：本契约的批准与实现必须分别经独立审核（合规 + 技术/验证），
   审核者不得是起草人或实现者。

## 10. 停止条件（任一命中 → 任务 failed + 完整报告，禁止静默降级）

1. **无法证明历史输入/身份时标为未验证**：旧 run 缺事件链或缺身份时，只能
   标 `legacy`/`unverified` 并如实展示；**不得**伪造事件、不得推定其状态。
2. **不得为让旧 run 通过而改写它的身份**：任何 UPDATE/DELETE/重写既有
   `lifecycle.jsonl`、`runspec.json`、artifact 身份字段的行为一律红线。
3. **A02 未批准/未实现**：若数据身份尚未内容确认，A03 不得以「先接线、身份
   后补」推进；停下等待。
4. **接线导致真实正式运行无法通过**：若真实 run 因缺资格被拒绝，报告拒绝
   原因与解除条件，由用户裁决；**禁止**降低门槛让其通过。
5. **资源超限**（§8）且无法在不降低证明强度的前提下收敛：停下。
6. **实现需要修改本契约的动作模式表或 p8 状态机**：停下，提交修订草案。
7. **触碰 inScope 之外的文件**、修改 `AlphaGPT-factor` 工作树、或覆盖用户
   既有修改：立即停止并报告。
8. **出现「为通过测试而弱化断言」的诱因**（例如 R9 的调用顺序难以观测）：
   停下并报告，不得删测试或改断言。

## 11. 需用户（或转授后的 captain）明确批准才能实现的条款

| 编号 | 需批准的条款 | 冻结的语义 | 若被否决的后果 |
|---|---|---|---|
| A03-A1 | §3/I2：正式模式缺身份一律拒绝，`None` 放行仅限显式非正式模式 | 生命周期门禁的强制力 | F03 缺口保留；A05/B02 的入口资格无从建立 |
| A03-A2 | §3/I3 + I4：五类动作模式及其允许状态表 | 动作分类与资格裁决 | 无法区分「工程探针」与「正式研究」的证据强度 |
| A03-A3 | §3/I5：搜索前完整资格、装载 ≠ 搜索 | 正式搜索的前置条件 | 搜索可能在只有装载检查的情况下开跑 |
| A03-A4 | §3/I6：身份先于昂贵计算 | 执行顺序契约 | 仍可能「先算完再补身份」 |
| A03-A5 | §3/I7 + I11：产物身份来自执行期上下文；拒绝不得留成功产物 | 产物身份与拒绝留痕边界 | 补造身份/拒绝后仍出产物的风险 |
| A03-A6 | §3/I8：受保护 OOS 只能由 protected_oos 模式读取 | OOS 隔离 | PR 开发反馈可能再次触达受保护窗口 |
| A03-A7 | §3/I9：paper 账户绑定（禁止硬编码 `"current"`） | 账户身份 | F06 的假阳性继续存在于入口 |
| A03-A8 | §4.2 的 `LIFECYCLE_CONTRACT_VERSION` 1→2 与 `RUNSPEC_SCHEMA_VERSION`/`ARTIFACT_SCHEMA_VERSION` 的 bump 方向 | artifact 兼容性分类 | 新旧门禁代际不可区分 |
| A03-A9 | §5.3 + §6 末段：legacy 显式化与既有测试的白名单处置 | 旧产物与测试的可用边界 | 测试可能被弱化消红 |
| A03-A10 | §8：资源上限数值（5s/512MB 热路径；5min/1GB 对账） | 工程预算 | 无上限即无法判定「超限即停」 |
| A03-A11 | §10 停止条件（尤其「不得改写旧身份」） | 证据完整性红线 | 停止条件不可作为失败裁决依据 |

## 12. 附：本草案的边界核对（落盘时的事实）

- 本文件**只**新增两个改动：`docs/p27_a03_lifecycle_runcontext_draft.md` 与
  `docs/CONTRACTS.md` 的一行索引（同 commit）。
- **未**修改任何源码、测试、配置、数据库或 artifact；**未**运行任何训练、
  搜索、评估、回测、模拟、数据同步或回填；**未** push / 建 PR / 合并。
- **未**修改 `AlphaGPT-factor` 工作树的未提交工作。
- **未**自我批准：状态为 DRAFT，正文不含 `APPROVED` 状态行，也不含批准记录行。
- **未**预写任何「实际发生」的测量结果（AGENTS §11.1）；§7 仅为测量方案。
