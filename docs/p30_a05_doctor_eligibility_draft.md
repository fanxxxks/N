# P30 A05 可信 doctor 与运行资格报告预注册契约（APPROVED）

- 状态：**APPROVED**。批准依据 = 用户 2026-09-14（Asia/Shanghai）会话直接指令
  「阅读AlphaGPT_整改与提升执行方案_2026-09-09.md，完成A05，开两个子agent来
  负责审核你的工作，除此之外不要开任何多余的子agent」（AGENTS §1 仲裁顺序
  第一位：用户本次明确要求）。**转授边界如实记录**：
  `reports/alphagpt-ab/00_authority_exception_record.md` §3 的 DRAFT→APPROVED
  转授仅覆盖 p26–p29 四份契约，**不含本契约**；本契约的批准依据仅为上述
  用户直接指令，不得援引该转授。用户保留对任一条款与裁决的撤销权（撤销方式 =
  回退契约状态 + 不合并对应实现 commit）。
- 起草：执行模型（本会话，用户直接指令「完成A05」），2026-09-14。
- 起草依据：
  - 用户任务指令（完成 A05；双子 agent 审核；无多余子 agent）；
  - `reports/AlphaGPT_整改与提升执行方案_2026-09-09.md` §2.2 F06、§2.3 F12、
    §四 A05 工作包、§五 第一阶段完成标准；
  - A01 技术评审遗留项 O5（`reports/alphagpt-ab/A01_report.md` §O5，用户裁决
    「O5→并入 A05」，见授权记录 §4ter 条目 10）；
  - AGENTS.md §2.2/§3.1/§4.3/§4.5/§9/§10/§11；
  - `docs/p8_research_lifecycle_contract.md`（APPROVED；§8 旧产物与 account、
    §9 激活边界）；
  - `docs/p26_a02_dataset_snapshot_identity_draft.md`（APPROVED；复合身份确认
    API）、`docs/p27_a03_lifecycle_runcontext_draft.md`（APPROVED；RunContext/
    动作模式）、`docs/p17_tushare_pit_data_contract.md` §12（D7 全局新鲜度门禁
    为预注册 NON-GOAL）、`docs/p16_data_freshness_gate_contract.md`（G8）。
- **勘察基线（docs/CONTRACTS.md「勘察基线惯例」[02-INT-09]）**：起草于分支
  `codex/a05-doctor-eligibility` @ merge commit `a18cac8`（= main @ `aac4aad`
  + A04 四提交 + A03 十二提交的合并态；A01/A02 已在 main，A03/A04 在本分支
  集成）。本文全部代码事实在该 commit 上核对。

## 0. 问题陈述

1. **F06（方案 §2.2）**：`research_doctor` 的 HEALTHY 语义过宽——lifecycle
   检查调用 `assert_paper_account_writable("current")`，用硬编码 ``"current"``
   换取恒真，不读取任何真实账户绑定；报告只有一个全局 ``healthy`` 布尔，
   把「程序可用」「数据合格」「研究可执行」「可晋级」「paper 可续跑」混为
   一谈。已确认的缺陷（2026-09-09 审计：manifest 落后、配置路径失效）与
   doctor HEALTHY 并存，即假阳性实证。
2. **四态缺失**：报告没有 `pass/fail/unknown/not_applicable` 区分；不可计算
   与通过不可区分；不适用被当作通过。
3. **无绑定**：doctor 报告不绑定配置身份、快照身份、运行模式与评价时点，
   无法回答「这份报告说的是哪个配置/哪批数据/何时」。
4. **O5（A01 遗留）**：`webapi.get_run_detail` 未返回 `runs_dir_source`/
   `runs_dir_exists`（A01 C9 字面要求；`list_runs` 已带，详情接口缺失）。

## 1. 范围与非目标

**范围**（inScope）：

- 新模块 `ashare_model/readiness.py`：分层就绪报告（六维 × 四态）的组合层，
  只读消费既有权威，不新建第二套判定；
- `ashare_model/research_doctor.py`：接线 readiness 报告节；移除
  `assert_paper_account_writable("current")` 假阳性；可选 `--spec-id/--run-id`
  绑定真实 run；文档字符串如实声明 healthy 的新边界；
- `ashare_model/data_eligibility.py`：仅为组合层暴露所需的只读辅助（若需），
  不改变 D7 判定语义；
- `webapi/service.py`、`webapi/app.py`：`/api/readiness` 状态接口；
  `get_run_detail` 补 O5 provenance；
- `scripts/check_test_shards.py`：新测试文件同 commit 注册（fail-closed 守卫）；
- 测试 `tests/test_p30_a05_doctor_eligibility.py`、现有测试的契约内适配；
- 文档：本契约（同 commit 登记 `docs/CONTRACTS.md`）、测量日志（后置
  evidence-only commit）、`research_doctor` 模块 docstring。

**非目标**（本包禁止）：

- 不改变 G1–G8、D7、lifecycle 状态机、RunContext、复合身份的任何判定语义
  （只组合、不改权威）；
- 不为 RunSpec 新增字段，不修改 `_VERSION_IMPORTS`（readiness 版本不是
  artifact-facing 版本：它永不进入 run artifact、兼容性分类或缓存键，仅自述
  于报告节内——依据 AGENTS §3.2「确实不适用时必须在契约中说明」在此说明）；
- 不实现 B01 的分域资格报告（p29 范围）；
- 不修改主库/v2 数据内容，不发布新快照（有状态操作未获授权）；
- 不改变 doctor CLI 退出码语义（0/1/2 保持），不把 readiness 状态并入
  `healthy`（见 §3.4）；
- 不删除 `data/runs`、`paper/` 等既有产物；不清理任何工作区。

## 2. 不变量

1. **只读**：readiness 层对数据库只读连接（v2 强制 read-only）、对 RunStore
   只重放不落锁（构造 `RunHandle` 走只读构造，不 `attach_run`）、对文件系统
   不写（`--output` 仍为 doctor 既有显式写）。
2. **单一权威**：G 门禁 = `ashare_data.gates.ProductionGateRunner`；快照身份
   = `runspec.confirm_composite_snapshot`；D7 = `check_runspec_data_eligibility`
   （经 FACTOR_SET_QUALIFIED 证据重放）；账户绑定 = `bound_paper_account_id`
   读路径；状态重放 = `lifecycle.replay_run`；动作模式表 =
   `lifecycle.ACTION_MODES`。readiness 不复制其中任何判定。
3. **四态语义**：`pass` / `fail` / `unknown`（不可计算，必须给原因）/
   `not_applicable`（该用途在本环境不存在，必须给原因）。`not_applicable`
   永不聚合为 `pass`；`unknown` 永不聚合为 `pass`。
4. **维度聚合规则**（冻结）：任一 `fail` → `fail`；否则任一 `unknown` →
   `unknown`；否则全部 `not_applicable` → `not_applicable`；否则 `pass`。
5. **绑定**：报告必须携带 `bindings`（配置路径+sha256、复合快照身份或其
   失败原因、action_mode="audit"、评价时点 UTC+本地 evaluation_day、git
   commit、store 根及来源、显式绑定的 run 身份）。
6. **历史/在线分离**：快照完整性回答「冻结输入是否仍是它声明的内容」（与
   今日日期无关）；paper/在线新鲜度经 G8 结果（今日相关）单独呈现——两者
   状态允许不同，且都不得相互替代。
7. **健康边界**：doctor 既有 `healthy` 仅表示「工程冲突规则（版本/dataset
   绑定/gate/fairness error）未触发」，不表示任何资格维度合格；human 输出
   必须分维呈现 readiness，禁止以统一 HEALTHY 行代替分维资格。

## 3. 报告契约（schema 冻结）

### 3.1 六个维度与检查项

| 维度 | 检查项（name） | 权威来源 | 四态判定 |
|---|---|---|---|
| runtime | config_loaded | `load_config`（doctor 主流程） | fail=加载失败（原因+路径）；pass 附 path+sha256 |
| runtime | input_paths_present | 主库/v2 路径存在性（A01 resolver + `resolve_v2_duckdb_path`） | fail=缺失（列精确路径+解除条件）；unknown=路径规则本身不可解析 |
| runtime | runtime_deps | `gather_dependencies` + gp→deap 条件 | fail=缺依赖（列名）；unknown=配置失败无法判定 gp 条件 |
| runtime | store_root_resolvable | `resolve_runtime_paths`（audit 模式） | pass 附 root+source；fail 附错误 |
| snapshot_integrity | composite_identity_confirmed | `confirm_composite_snapshot`（全量未缓存重哈希） | pass 附 composite_id/publication_id/两成员 id；fail 附漂移表+解除条件（DatasetIdMismatch 原文）；unknown=库不可读 |
| data_eligibility | production_gates_g1_g8 | `gather_gates` 结果 | pass/fail 附逐 G 明细；unknown=未计算（附原因） |
| data_eligibility | runspec_data_eligibility | 绑定 run 的 FACTOR_SET_QUALIFIED 证据内嵌 D7 结论 | pass/fail=按证据记录（附记录时 evaluation_day 语义：历史事实）；not_applicable=未绑定 run 或 run 未达 FACTOR_SET_QUALIFIED（附原因）；unknown=证据完整性失败 |
| research_ready | lifecycle_edges_activated | `ACTIVATED_EDGES == legal_edge_pairs()` | pass/fail |
| research_ready | fairness_probes | 既有 `gather_fairness_probes` | fail=探针失败（正式运行阻断）；pass 附基线状态；基线缺失/过期=pass 但 detail 携带补记义务（维持决策⑥语义：warning 非阻断） |
| research_ready | prerequisites（carried） | runtime/snapshot/gates 三处结果 | 状态逐一携带（不重算）；carried fail/unknown 使本维聚合遵循 §2.4 |
| promotion_ready | artifact_binding | doctor 既有 artifacts 冲突规则结果（非 legacy 存在时） | fail=非 legacy 产物版本/dataset 不一致（附产物名+字段）；pass=一致；not_applicable=仅有 legacy 产物（「可审计、永不晋级」）或无产物 |
| promotion_ready | source_run_qualified | `assert_qualified_source_run` + 状态 ≥ OOS_QUALIFIED | 仅绑定 run 时适用；not_applicable=未绑定 run |
| paper_ready | paper_binding_evidence | RunStore 全扫描：存在重放态=PAPER_OBSERVING 的 run，`bound_paper_account_id` 读路径取绑定账户 | pass=存在（列 spec/run/account）；not_applicable=不存在（附原因：「本环境无生命周期绑定的 paper 观察；续跑声明被拒绝」） |
| paper_ready | sim_state_vs_binding | paper 状态文件存在性 × 绑定证据 | fail=有状态文件但无绑定证据（「无账户却可续跑」的假阳性关闭点）；pass=两者一致存在；not_applicable=无状态文件且无绑定；unknown=绑定存在但状态文件缺失 |
| paper_ready | online_data_freshness（carried） | G8 结果 | 状态携带（今日相关）；与 snapshot_integrity 显式分离 |

### 3.2 聚合与 summary

- 每维 `status` 按 §2.4 聚合；报告附 `summary.blocking`（全部 fail 项：
  dimension/check/detail/release_condition）、`summary.unresolved`（全部
  unknown 项）、`summary.by_status` 计数。
- 每个非 pass 检查必须给出**解除条件**（release_condition）：缺失路径、表、
  版本或需要执行的发布/预注册动作。

### 3.3 绑定块

`bindings`: {readiness_version, action_mode:"audit", evaluated_at(UTC ISO),
evaluation_day(Asia/Shanghai YYYYMMDD), config_path, config_sha256,
git_commit, repo_root, store_root, store_root_source, bound_run|null,
snapshot: {composite_id, publication_id, primary_dataset_id, v2_dataset_id}|{error}}

### 3.4 doctor 集成边界

- `build_report(..., readiness=None)`：新可选节；`healthy` 判定**不变**（不含
  readiness）；docstring 与 human 输出明确「HEALTHY ≠ 分维资格」。
- CLI：新增 `--spec-id/--run-id`（绑定真实 run；两者必须同时给出，否则
  argparse 级报错）；配置加载失败时以最小 readiness 报告（runtime=fail、
  其余 unknown）如实呈现后仍以退出码 2 结束。
- `gather_lifecycle_checks`：删除 `assert_paper_account_writable("current")`
  恒真检查（该 API 本身保留给其真实调用方）；paper 资格改由 paper_ready
  维度以真实绑定证据呈现。
- webapi：`GET /api/readiness`（只读、防御性降级：错误显式标
  unavailable，不显示健康）；`get_run_detail` 响应增加
  `runs_dir_source`/`runs_dir_exists`（O5）。

## 4. 失败测试（RED 清单，R1–R12）

纯逻辑用 `build_readiness_report` 纯输入驱动；环境型用 tmp_path/monkeypatch
伪造最小环境，不触真实库。

- **R1**（环境依赖齐全但 manifest 失效）：runtime 全 pass、gates 传 ok 的
  输入 + `confirm_composite_snapshot` 抛 `DatasetIdMismatch`（漂移表）→
  snapshot_integrity=fail（含漂移明细与解除条件）、research_ready=fail、
  summary.blocking 非空；runtime 仍 pass（分层不互相掩盖）。
- **R2**（没有账户却声称可续跑）：paper 状态文件存在 + RunStore 无
  PAPER_OBSERVING run → paper_ready=fail（非 pass/非 not_applicable），
  detail 指明「状态文件存在但无绑定证据」；无状态文件且无绑定 →
  not_applicable（非 pass）。
- **R3**（配置错误但主库 gates 通过）：config_error 注入 → runtime=fail、
  snapshot/gates/data_eligibility/research/paper=unknown（不可计算，附原因），
  gates 传入的 ok 结果不得把 runtime 维度洗白为 pass。
- **R4**（当前仅有 legacy 产物）：artifacts 全 legacy → promotion_ready=
  not_applicable（detail「可审计、永不晋级」），绝不为 pass；存在非 legacy
  且版本/dataset 失配 → fail。
- **R5**（历史快照复现与在线新鲜度分离）：快照确认 pass + G8 fail（滞后超限）
  → snapshot_integrity=pass 且 paper_ready 携带 fail 的 carried 新鲜度检查；
  两状态同时成立、互不覆盖。
- **R6**（四态聚合规则冻结）：纯函数逐例验证 §2.4（fail>unknown>pass；
  全 N/A→not_applicable；N/A+pass→pass）。
- **R7**（绑定块）：报告缺任何 bindings 必备键即构造失败（KeyError/断言）；
  action_mode 恒为 "audit"。
- **R8**（paper 扫描真实重放）：tmp RunStore 造一个 PAPER_OBSERVING run
  （经 lifecycle writer 真实转换 + 绑定证据）→ paper_binding_evidence=pass
  且列出的账户 = 绑定证据账户；RETIRED/REJECTED run 不计入。
- **R9**（doctor 移除假阳性）：`gather_lifecycle_checks` 输出不再包含
  `paper_account_writable`（或其 "current" 恒真语义）；lifecycle_edges 检查
  保留且真实。
- **R10**（O5）：`get_run_detail` 响应含 `runs_dir_source`/`runs_dir_exists`
  且与同根 `list_runs` 一致。
- **R11**（CLI 绑定参数）：`--spec-id` 与 `--run-id` 必须成对；只给其一在
  argparse 层失败。
- **R12**（webapi readiness 端点）：monkeypatch 注入 config 失败 →
  响应显式 unavailable 且 HTTP 200 载荷内状态可见（防御性降级不伪装健康）。

### 4.1 既有测试适配（§10.1 白名单声明）

- `tests/test_research_doctor.py`：现有断言不涉及 lifecycle_checks/paper
  （grep 佐证：仅 fairness/readiness 字样，无 paper_account 断言），预期
  无需修改；若全量回归暴露遗漏断言，按白名单逐条举证后同 commit 处理。

## 5. 版本影响表

| 版本 | 旧→新 | 理由 |
|---|---|---|
| READINESS_REPORT_VERSION（新增，readiness.py 内） | —→"1" | 报告 schema 首版；自述于报告节 |
| 其余全部 `*_VERSION` | 不变 | 零研究语义变化：G/D7/lifecycle/复合身份/词表/组合/费用/晋级阈值全部只读组合 |

## 6. 验收（对本次环境的期望结论）

在本分支集成环境实测 doctor：`runtime` 应 pass（依赖齐全、路径有效）；
`snapshot_integrity` 应以真实复合身份结论呈现（A02 已发布代际）；gates 以
真实 G1–G8 结论呈现；`paper_ready` 在无绑定 paper run 时必须呈现
not_applicable 或 fail（绝不 pass）；整体结论必须能同时陈述「运行时可用 +
存在阻断项 + 旧产物可审计」，任何阻断项带具体缺失路径/表/版本与解除条件。

## 7. 停止条件

- 实现中发现需要改动任何权威模块的判定语义 → 停止，回用户裁决；
- 发现 readiness 需要写任何状态（锁、缓存、DB）→ 停止，回用户裁决；
- 全量回归出现无法归因于本包的失败 → 停止诊断，不得放宽断言凑绿；
- 涉及删除/移动历史产物或数据库写入 → 一律不执行（未获授权）。

## 8. 测量与裁决

- 测量日志：`docs/p30_a05_doctor_eligibility_measurement_log.md`（后置
  evidence-only commit，引用被测实现 SHA）；RED/GREEN、聚焦+全量、
  compileall、`git diff --check`、分片守卫全部留痕。
- 双审：两个独立审核 agent（技术审 + 合规审）；findings 全部登记处置；
  needs_revision 触发修复轮（不新建并行语义）。
