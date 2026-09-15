# P32 B03 可恢复的执行与预算控制预注册契约（草案）

状态：**APPROVED**，用户 2026-09-15 会话直接指令「完成B03」（授权记录：本会话用户消息，
对齐 p31 B02 先例「用户直接指令 + 执行模型实施」）。执行模型起草，分支
`codex/b02-entry-canary-ci` @ `2ccbbc4`（勘察基线 commit；B03 依赖 A03、B02——两者分别
在 p27/p31 交付并已合并到本分支，本分支 HEAD 即含 A01–A05、B01、B02 的线性链，故从该
HEAD 而非落后于 B02 的 `main` 创建任务分支 `codex/b03-resumable-execution-budget`）。

本文依据《AlphaGPT 整改与提升执行方案》B03 工作包与 F09 观察起草，是工程行为契约：
不改变任何研究语义（见 §6 版本影响表），先固定行为契约与验收，再 RED、再实现。

## 1 问题陈述与证据

F09（方案 §2.3）：`scripts/r1_campaign.ps1` 以 `.done` 文件跳过任务，未绑定配置/数据
身份；部分配置失败后驱动最终仍 `exit 0`；脚本未落实完整墙钟约束。B03 要求：复用
`TaskExecutor` 的执行、恢复、预算和最终状态语义，关闭 `.done` 权威、部分失败谎报成功、
无预算控制三个缺口。

勘察基线事实（2026-09-15，分支 `codex/b02-entry-canary-ci` @ `2ccbbc4`）：

* `ashare_model/task_executor.py`（E3/t5）已具备：行级独立子进程、心跳超时杀树、原子
  行结果、RunStore 账本绑定（spec_id/run_id）、plan 内容寻址注册与漂移拒绝、恢复对账
  （interrupted → 记 failed 后补执行一次；有有效结果则采纳）、单写者锁。**但没有任何
  生产调用方**（仅测试引用）；无阶段遥测、无行级预算、杀树前无归属核验、无用户中止处理。
* `scripts/r1_campaign.ps1`：`.done` 为跳过权威（L96–99、L104）；部分失败仍
  `exit 0`（L114–115）；无墙钟/内存约束；每次 leg 直接 `Start-Process` 运行
  `python -m ashare_model.evaluation`，不经 TaskExecutor、无账本绑定的尝试登记。
* `ashare_model/searcher_campaign.py`（P10/P14）是进程内搜索器基准 campaign 层，其
  状态词汇（`completed/failed`、`stopped_wall_cap`、`not_run` 带原因）、行前墙钟熔断、
  身份漂移拒绝是先例；它不提供子进程隔离与行级恢复，不能承载 R1 评估 leg（整协议
  进程级重活），也不是本包要替换的对象。
* A02/p26 提供复合快照确认权威（`AshareDataLoader.confirm_input_snapshot` +
  `frozen_data_cutoff`，全量未缓存重哈希）；A03/p27 提供入口身份接线（evaluation 正式
  模式强制 `--spec-id/--run-id`，engineering 是显式非正式探针）。

## 2 范围与非目标

**允许修改/新增**：`ashare_model/task_executor.py`、`ashare_model/campaign_executor.py`
（新增，campaign 适配器）、`ashare_model/evaluation.py`（仅阶段标记五处调用）、
`scripts/r1_campaign.py`（新增 CLI + R1 manifest）、`scripts/r1_campaign.ps1`（改为薄
委托入口）、`tests/test_b03_campaign_executor.py`（新增）、
`scripts/check_test_shards.py`（同 commit 注册新测试）、`docs/CONTRACTS.md`（同 commit
登记本契约）。

**非目标（明确不做）**：不改任何研究语义（protocol/reward/grammar/registry/…）；
不启动真实 R1 campaign 或任何数据写入；不接线正式研究身份 leg（R01 预注册后另行
接线）；不迁移/清理任何历史产物或 `.done` 文件；不改 AGENTS.md；不 push/建 PR/合并；
不并行执行多个 leg（16GB 主机一次一个重任务，TaskExecutor 串行执行保持原样）；
不实现数据装载性能优化（B04）。

## 3 行为契约

### 3.1 尝试登记与阶段遥测（I1/I2）

* **I1**：一个 campaign leg = 一次 TaskExecutor run（单行 plan）。尝试启动即先
  `record_trial` 再 spawn worker（既有行为）；trial payload 携带 `leg_id` 与阶段计划
  `stages = ["loading", "feature_build", "search", "evaluation", "artifact_write"]`。
* **I2**：worker 通过环境变量 `TASK_EXECUTOR_ROW_PHASE_FILE` 向行命令暴露阶段文件路径；
  `evaluation.py` 在五个里程碑调用 `task_executor.row_phase(name)`：`loading`
  （`main()` 中 `load_data()` 前）、`feature_build`（`run_protocol` 入口处）、`search`
  （fold 循环前）、`evaluation`（fold 循环后、`build_result` 前）、`artifact_write`
  （`main()` 中 `write_boundary_artifact` 前）。行结果携带 `phases`（JSONL 记录数组）
  与 `phase`（最后阶段）；账本 metrics 附带 `phases`/`active_phase`。`row_phase` 在无
  环境变量时 no-op；写失败仅告警不抛出（遥测不阻断研究，与 `ashare_trading/manager.py`
  的 phase 约定同型）。阶段语义是**粗粒度里程碑**（fold 循环内 baseline/搜索/训练
  归入 `search`），不是逐调用归属；selfcheck 路径不经过 `run_protocol`，只记录其
  实际进入的阶段。

### 3.2 身份与复用（I3/I4）

* **I3**：leg 身份 = `content_hash("campaign_leg_identity", {config_sha256, dataset_id,
  data_cutoff, git_commit, spec_id, seeds, requested_budget, frequency, horizon,
  research_domain, searcher, max_formula_len, lifecycle_mode})`。复用判定**只**来自
  状态文件：leg 状态为 `succeeded` 且身份哈希与当前一致、且 state 记录的 output 指针
  与当前 leg 一致 → 跳过（派生 `.done` 仅作显示，由终态成功写出；绝不读取 `.done`
  做决策；无状态记录的裸 `.done` 一律视为不可复用 → 重跑）。`spec_id` 与 output
  指针入闸是 rev 1 修订（见 §9）：`runtime_overrides.yaml`/`.env` 深合并与脏树
  可以在 leg YAML 不变的情况下改变有效运行参数（spec_id 必变），output 指针防
  「身份替另一路径的产物作保」。
* **I4**：数据身份经唯一权威解析：驱动每次启动构建一次 `AshareDataLoader`（首个 leg
  的 data_config；跨 leg 校验 data 配置一致，不一致拒绝），调用
  `confirm_input_snapshot()`（全量未缓存重哈希，代价在 §5 披露）与
  `frozen_data_cutoff()`；不派生第二套快照/截止日推导。确认失败（`DatasetIdMismatch`）
  → 拒绝启动全部 leg（exit 2），不降级。

### 3.3 预算与资源（I5）

* **I5**：行级预算由 worker 强制：`row_wall_clock_budget_s`（超时→终止子命令→failed，
  错误含 `row wall-clock budget`）、`row_peak_rss_budget_mb`（超限→终止子命令→failed，
  错误含 `row peak RSS budget`）。campaign 级墙钟上限在每 leg 启动前检查（对齐
  searcher_campaign 行前熔断模式）：超限 → `stopped_budget`，剩余 leg 记 `not_run`
  带原因，exit 1。三个预算默认 `None`（不设限），实际数值由 campaign manifest 冻结，
  代码不硬编码 16GB 主机数值。执行严格串行（TaskExecutor 原状）。

### 3.4 终态语义与退出码（I6）

* **I6**：leg 终态三义：`succeeded` / `failed`（命令失败、心跳失联、预算超限）/
  `aborted`（用户中止 SIGINT，或驱动被杀后 leg 留存 open trial——恢复时关闭为 failed
  并补执行，账本证据保留）。campaign 状态：`completed` / `failed` /
  `stopped_budget` / `aborted`。退出码：全 completed（含复用）= 0；任一 leg failed =
  1（**部分失败不得返回整体成功**）；用户中止 = 130；操作前置拒绝（锁冲突、快照
  确认失败、配置/RunSpec 解析失败、配置预检失败、账本完整性拒绝）= 2。
  rev 1 澄清：(a) 配置预检失败与身份/RunSpec 解析失败同样映射 exit 2（操作前置
  拒绝），不作为 leg 失败；(b) campaign 墙钟上限触发时，遗留的 `running` 记录
  （被杀前驱的 interrupted leg）保持 attachable、不被降级为 `not_run`，此时
  campaign 终态为 `failed`/exit 1（下一轮恢复）；(c) `aborted` 对 campaign 是
  粘性的——后续的预算停止不得把已有 abort 证据降级。

### 3.5 恢复语义（I7）

* **I7**：重启驱动时逐 leg 对账：`succeeded` 且身份一致且 output 指针一致 → 复用；
  状态为 `running/aborted` 且存在 run_id 且身份一致 → **attach 同一 run 继续**
  （TaskExecutor 把 interrupted 尝试关闭为 failed 并补执行一次，或采纳其有效结果；
  旧失败试错保留在链上）；身份在 in-flight 期间变化 → 不 attach，开**新 run**，
  state 记录 `superseded_run_id` 与原因（rev 1，review A#5）；`failed`（终态工作
  失败）→ 默认**不**自动重跑；`--retry-failed` 显式开新 run（新 run_id，旧 run 与
  账本原样保留）。恢复是「继续已验证 checkpoint（同 run 新 attempt）」还是
  「新 attempt（新 run）」由上述规则唯一决定，并在状态与日志中显式。

### 3.6 归属核验与单写者（I8/I9）

* **I8**：supervisor 在任何杀树前核验目标进程 cmdline 包含本 run 的 job 文件路径
  （心跳超时与用户中止两条路径；worker 杀的是自己 spawn 的子命令，无需核验）。
  核验失败（PID 已消失或被复用为无关进程）→ 不做树杀、记录；此时仅允许对**自有
  Popen 句柄**做单进程终止——句柄未 reap 期间 OS 不会复用该 PID（Windows 句柄
  钉住 / POSIX 未收尸僵尸），故该回退不可能命中无关进程，也不可能是树杀
  （rev 1 澄清，review A#10/B#4）。
* **I9**：campaign 级锁（`O_CREAT|O_EXCL`，内容含 pid）：第二个驱动 → `exit 2` 拒绝；
  陈旧锁只能经显式 `--clear-stale-lock` 清除，清除前用 psutil 核验持锁 pid 已死亡
  （对齐 RunStore `clear_stale_lock` 的显式管理动作语义）。状态文件的
  `campaign_id` 与当前 manifest 不一致时拒绝（exit 2）——禁止跨 campaign 混用
  leg 记录（rev 1，review A#7）。

### 3.7 驱动与入口（I10/I11/I12）

* **I10**：campaign 状态文件 `<state_dir>/campaign_state.json`（schema version 1，
  `ashare_data.io_utils.atomic_write_json` 原子写——唯一原子 JSON 权威，不引入第二套）；
  每 leg 记录：status、identity（哈希与字段）、run_id、spec_id、trial_id、result/output
  指针、telemetry、时间戳、error。`markers.log` 追加人读历史（延续 p21 约定）。
* **I11**：telemetry 提取自 leg 的协议产物（真实来源，不从实现反推）：对 rows 的
  `search_result` 求 `requested_budget/consumed_budget/proposal_count/invalid_proposals/
  semantic_duplicates` 之和；`n_candidates` 取产物字段；行墙钟/峰值 RSS 取 executor
  结果。字段缺失记 `null`，不虚构。
* **I12**：leg 以 `--lifecycle-mode engineering` 显式非正式模式运行（p27 I2 机器标记）；
  驱动不铸造正式研究身份，不产生正式证据。R1 manifest（calibration + 5 配置 leg）
  保持 p21 阶段语义：默认仅 calibration，`-RunConfigs`/`--run-configs` 进配置阶段；
  每 leg 启动前保留 A01 C5/C6 配置预检（`ashare_data.config main --check`）。

## 4 失败测试（RED → GREEN，R1–R9 对应方案九场景）

| # | 场景 | 断言要点 |
|---|---|---|
| R1 | 装载阶段崩溃 | leg 命令标记 `loading` 后 exit 3 → leg failed、active phase=loading、其余 leg 完成、exit 1 |
| R2 | 产物（状态）写一半 | 注入 `os.replace` 失败 → 旧状态完好、无半文件 |
| R3 | 同一任务重复启动 | 第二驱动被拒（exit 2）；陈旧锁需显式清除且核验 pid 已死 |
| R4 | 进程被终止 | 真实子进程杀驱动树 → 重启 attach 同 run、interrupted 关闭为 failed、补执行一次、旧证据保留 |
| R5 | 子进程失联 | 心跳超时 → leg failed（原因含 heartbeat timeout）、后续 leg 继续、exit 1 |
| R6 | 旧 marker 配置已变 | 成功 leg 的配置改动 → 新 run 重跑、旧 run 账本不动；裸 `.done` 无状态 → 重跑 |
| R7 | 部分配置失败 | 一 leg failed → exit 1、后续 leg 仍执行、计数正确、campaign_status=failed |
| R8 | 预算耗尽 | 行墙钟/行 RSS 超限 → failed 带预算原因；campaign 墙钟上限 → 剩余 not_run、stopped_budget、exit 1 |
| R9 | 用户中止 | 驱动子进程收 SIGINT/CTRL_BREAK → exit 130、leg=aborted、worker 树确实死亡；重启后该 leg 恢复补执行一次 |

附加守卫：阶段遥测（executor 层 env 注入 + evaluation 集成）、身份复用正例、telemetry
提取、trial 阶段计划登记、R1 manifest 构建（真实 6 份配置可解析）。评估入口集成
（真实 evaluation leg 跑在 B02 canary 双库 fixture 上）验证真实阶段序列
loading→feature_build→search→evaluation→artifact_write 与真实 telemetry 提取。

## 5 资源与运行上限

* 驱动每次启动一次全量双库复合确认（A02 语义，`confirm_input_snapshot`）：这是
  **接受的成本**——复用判定正是 F02「查最新 id ≠ 内容确认」教训的消费侧，不能用
  未确认的已发布 id 做复用权威。真实库上的耗时在测量日志记录。
* 预算数值不在代码/契约硬编码：manifest 冻结（row/campaign 墙钟、行 RSS）。默认
  `None` = 不设限，但 campaign 墙钟上限与行预算字段必须在 manifest 显式声明取值
  （含 `null`）。
* 严格串行：一次一个 leg、一个 worker、一个子命令。

## 6 版本影响表

| 版本 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `TASK_EXECUTOR_SCHEMA_VERSION` | 1 | 1（不变） | job/result 仅**加性可选**字段（`phase_path`、`phases`、预算字段），旧格式读取容错（P8-04 加性先例，`RunSpec.manifest_version` 同型）；无行为语义收窄 |
| `LEDGER_SCHEMA_VERSION` | 1 | 1（不变） | 账本仅经既有 `metrics`/`payload` 自由字典携带阶段与预算证据，条目 schema 不变 |
| `CAMPAIGN_STATE_SCHEMA_VERSION` | — | 1（新增文件） | 新 artifact-facing 状态文件，schema version 1；无兼容分类问题（新建即当前） |
| protocol/reward/grammar/registry/… | — | 不变 | 无研究语义变化：阶段/预算/恢复全部是工程执行层观测与控制 |

## 7 验收对照（方案 B03「完成时必须交付」）

* 崩溃/中止/恢复演练及账本一致性：R1/R4/R5/R8/R9 + 账本断言（hash 链验证由
  `ExperimentLedger` 读取时强制）。
* 失败/中止/完成含义不同：I6/R7/R9。
* 部分失败不得返回整体成功：I6/R7。
* 恢复明确是继续已验证 checkpoint 还是新的 attempt，旧失败试错保留：I7/R4/R6。
* 终止进程树前核对属于本次 run，不影响其他模型任务：I8/R9。

## 8 停止条件

涉及删除 `.done`/历史 campaign 产物、数据库操作、正式 R1 运行或研究裁决时停止并
请求用户裁决。实施中发现契约缺口（如 evaluation 阶段边界与 §3.1 不符）先修订本
契约再继续，不静默扩大范围。

## 9 修订记录

* **rev 1（2026-09-15，双审处置，approval 不变——用户直接指令「完成B03」的授权
  范围内的契约澄清与收紧，均为面向未来的行为修订，不重写已发生证据）**：
  - I3：身份 payload 增补 `spec_id`；复用门槛增补 output 指针一致性（双审 A#3：
    `runtime_overrides.yaml`/`.env` 深合并与脏树可在 leg YAML 不变时改变有效参数）。
  - I6：exit 2 澄清覆盖身份/RunSpec 解析失败与配置预检失败（双审 A#2/B#1/B#2）；
    墙钟上限扫描不降级 `running`（保持 attachable）、`aborted` 粘性优先
    （双审 A#6/B#5）。
  - I7：身份 in-flight 变化时 state 记录 `superseded_run_id` + 原因（双审 A#5）。
  - I8：核验失败回退的自有 Popen 句柄终止不变量化（双审 A#10/B#4）。
  - I9：state `campaign_id` 与 manifest 不一致 → 拒绝（双审 A#7）。
  - I10：`_executor_result_path` 布局重推导退役，改由 TaskExecutor payload 携带
    行 metrics（加性可选字段，双审 A#11）。
