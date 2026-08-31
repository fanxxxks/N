# 测试运行时优化测量日志

本文只记录实际发生的测量与验证（命令、环境、结果、不变量），对应方案为
2026-08-31 测试运行时优化第一阶段（B1 基线 / A1+A2 / B2+C2 / D1+D2）。
预注册式的计划与合规判定见团队研究产出（2026-08-31，AGENTS §10 约束下的
只读审计结论）；本文不承载计划，只承载证据。

环境口径（与 p7 终态口径一致，保证可比性）：
Windows 11（10.0.22621）/ Python 3.13.12（`C:\ProgramData\miniconda3\python.exe`，
CPU 路径，torch `2.11.0+cpu`）/ duckdb 1.5.5 / pytest 9.1.1 / Node v24.14.0；
工作目录 `D:\minequant\AlphaGPT`。

注意：`D:\minequant\.venv`（onboarding 引用）装有 `torch 2.11.0+cu128`，
偏离 base 依赖的 CPU pin 契约；本 campaign 所有测量与门禁统一使用上表
CPU 路径环境。venv 的 CPU/CUDA 契约偏差已记录为后续独立小任务线索，
不在本 campaign 范围内处理。

对比不变量（继承 p7 规则）：passed 只允许随新增合同测试增加；
skipped/warnings 不得增长。

## B1 基线（2026-08-31 14:37，main @ cbda238，PR1）

- 被测实现：`cbda2384675ace7884fe7e4e9bf73c7c521ca16c`
  （`merge: P8-05 artifact schema v2 end-to-end lineage`，main）。
- 工作树状态：tracked 干净；既有未跟踪文件 `.agent-teams/`、
  `docs/p5_implementation_plan.md`（任务前已存在，未触碰）。
- 命令：`python -m pytest -q tests --durations=100`
- 结果（首次运行，2026-08-31 14:37）：1323 passed, 5 skipped, 614 warnings
  in 779.26s（0:12:59）。该次运行的 pytest 控制台输出未完整落盘：任务系统
  仅保留会话内尾部，`logs/pytest_b1_baseline_cbda238.txt` 实为 conftest
  导出的 loguru 内存缓冲片段（约 3.7 分钟跨度），不含 pytest 汇总行、
  warnings summary 与 durations 表——构成 §11.2 证据缺口，由下条完整
  重测收口（评审 R1-F1）。
- 结果（同 SHA 完整重测，2026-08-31 15:29，campaign 权威口径）：
  **1323 passed, 5 skipped, 614 warnings in 740.36s（0:12:20）**。
- 原始产物（权威）：`logs/pytest_b1_baseline_cbda238_full.txt`——在
  detached worktree（`D:\minequant\AlphaGPT-wt\b1-remeasure`，精确检出
  cbda238）以 `cmd /c "... > file 2>&1"` 完整捕获 stdout+stderr，含
  warnings summary（L20 起）、slowest-100 durations（L76 起）与汇总行
  （L177），格式同 `logs/pytest_p6_final.txt` 先例。两次运行计数完全一致
  （1323/5/614），墙钟差 −4.99% 属机器波动。
- 5 skipped 均为登记在册的 CUDA skipif 基线（test_ops/test_train/test_vm），
  与 p7 终态持平。

### 与 p7 终态基线的对比（诚实记录，不下结论）

| 口径 | p7 终态 | B1 基线 | 变化 |
|---|---|---|---|
| passed | 1212 | 1323 | +111（净增测试定义 +82：P8-01..04 +61、P8-05 净 +21；其余 +29 为参数化展开口径差） |
| skipped | 5 | 5 | 持平 |
| warnings | 614 | 614 | 持平 |
| 墙钟 | 562.68s | 740.36s | +177.68s |
| 平均每测试 | 0.464s | 0.560s | +20.5% |

墙钟增长的分解（测试数增加 vs 每测试变慢）未做归因实验；本表只记录
观察。B1 数字自此取代 p7 终态成为本 campaign 的对比基线。

### 最慢测试（--durations=100 的头部，全量墙钟的结构性事实）

| 测试 | 耗时 | 占全量 |
|---|---|---|
| test_evaluation.py::test_run_protocol_rows_and_determinism | 126.43s | 17.1% |
| test_evaluation.py::test_run_protocol_records_universe_policy | 68.13s | 9.2% |
| test_grammar.py::test_random_samples_execute_100_percent | 19.09s | 2.6% |
| test_evaluation.py::test_cli_confirmation_smoke | 17.84s | 2.4% |
| test_evaluation.py::test_cli_smoke | 16.97s | 2.3% |
| test_archive_run.py::test_run_dir_collision_gets_suffix | 11.79s | 1.6% |
| test_archive_run.py::test_protocol_manifest_records_actual_run_scope | 9.51s | 1.3% |
| test_archive_run.py::test_protocol_mode_does_not_require_formula | 8.31s | 1.1% |
| test_archive_run.py::test_protocol_mode_archives_with_manifest_block | 8.03s | 1.1% |
| test_promotion.py::test_cli_promotion_refuses_legacy_artifact_without_dataset | 7.90s | 1.1% |

top-2 = 194.56s（26.3%）；top-10 = 294.00s（39.7%）。头部排序两次运行
同构（单条 126.43s 的协议行测试仍是分片均衡与并行收益的最大杠杆；
test_archive_run.py 以数量取胜），尾部名次有机器负载噪声（如重测中
test_data_loader 的 8.71s 条目未进 top-10）。本表以可复核的完整快照
重测口径为准；分片均衡设计依据此表。

### warnings 结构快照（PR4 归并口径的原始材料）

机器可读权威 = `docs/ci_warning_baseline.json`（PR4/D2 生成自
`logs/pytest_gate_4156de4.txt` 的 warnings summary 段：50 行集合、
614 条实例；provenance 记录被测 SHA 与环境）。本节初稿的人工分组
转录被评审 R1-F5 证伪（subtract 组 18/5/1 被误写为 20/10/2、
baseline_harness 2 遗漏、reward.py 组与 numpy 组混计、divide 聚类
遗漏 universe sentinel），现按评审 requiredFix 以**指针 + 总数**取代
人工转录：total = 614（唯一经逐项核实无误的子计数：osqp
PendingDeprecation 553）。核对工具 = `scripts/ci_warning_merge.py`
（净新增行检测，CI test-warning-merge job 已机器化）。

### 证据缺口收口

本条目关闭了 t1 审计指出的"无 per-test duration"缺口：--durations=100
头部已入库（见上表），后续 C3 拆分与 PR4 分片均以此为均衡依据。

### 本条目验证矩阵

- `git diff --check`：通过（提交前复核）；
- 全量 pytest 于候选 commit 的 CI-equivalent 复现：见下条 PR1 门禁记录。

研究结论边界：本条目为工程测量（engineering run type），不产生也不
声称任何研究结论。

## PR1 门禁（2026-08-31，codex/test-runtime-b1-baseline）

第一阶段门禁在证据 commit `44a47fb` 上运行，结果如下：

| 门禁项 | 结果 |
|---|---|
| `python -m compileall -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi` | 通过（exit 0） |
| `python scripts/freeze_lock.py --check` | 通过（pin 文件同步；注意 `--check-full` 在本机对 `requirements.lock` 报既有漂移，见下） |
| `git diff --check` | 通过 |
| web job：`npm ci` → `npm ls --depth=0` → `npm run build`（tsc -b 在 build 内） | 通过（exit 0，build 40.85s） |
| `python -m pytest -q tests`（全量） | 通过：**1323 passed / 5 skipped / 614 warnings in 852.59s**，三项计数与 B1 基线一致，零回归 |
| `python -m pip check`（共享 miniconda 环境） | **失败（任务前既有状态，非本任务引入）**。归因分两类（R1-F2 修正）：①`langchain` 系（community 0.3.24 ↔ langchain 1.3.4、langchain-core 1.4.0 等）——真契约外包，requirements `.in`/`.txt`/`requirements.lock` 全部零匹配；②`idna` 3.11——**契约内陈旧**：`requirements.lock:37` pin `idna==3.18`，本机安装 3.11，低于 optional pin `httpx2==2.12.0` 的传递要求（idna>=3.18） |
| `python -m pip check`（CI 忠实复现：%TEMP% 干净 venv，仅装 requirements.txt + requirements-optional.txt） | **通过**（"No broken requirements found"）——与 CI 的干净安装语义一致 |

环境偏差披露（均为任务前既有，已记为后续独立小任务线索）：

1. 共享 miniconda 环境存在两类偏差导致本地 `pip check` 失败：契约外包
   （langchain 系，`.in`/`.txt`/lock 零匹配）与契约内陈旧（`idna` 本机
   3.11 < `requirements.lock:37` pin 的 3.18）。CI 等价复现须在干净 venv
   中执行（本条目已这样做，通过）。
2. `python scripts/freeze_lock.py --check-full` 失败：`requirements.lock`
   全量快照相对当前环境漂移，由上述两类偏差共同构成。pin 文件（CI 门禁
   对象）是同步的；lock 卫生修复不在本 campaign 范围。

原始产物留存（R1-F3）：44a47fb 与 3192e17 两次门禁 pytest 的控制台输出
未完整落盘（同 F1 缺口；计数 1323/5/614 与两次基线运行一致）。自本修订
起，最终候选的门禁 pytest 以完整重定向留档于 `logs/pytest_gate_<sha>.txt`
（结果与路径记入 merge commit 与评审记录），后续 PR 一律沿用该口径。

合并前在最终候选 commit 上重跑同套门禁的结果记录于 merge commit
message 与评审记录（测量日志只承载测量与门禁命令矩阵本身）。

## 修订记录

- **R1（2026-08-31，独立评审 findings F1-F4）**：
  F1/B1 与门禁运行的原始产物留存口径修正——在精确检出 cbda238 的
  detached worktree 以完整 stdout+stderr 重定向重测基线，计数与首测完全
  一致（1323/5/614），权威快照 `logs/pytest_b1_baseline_cbda238_full.txt`；
  对比表与最慢测试表切换到可复核的重测口径（740.36s）。
  F2/pip check 失败归因修正：langchain=契约外（lock 零匹配）、
  idna=契约内陈旧（lock pin 3.18 vs 本机 3.11）。
  F3/自本修订起最终候选门禁 pytest 完整落盘 `logs/pytest_gate_<sha>.txt`。
  F4/+111 归因修正：净增测试定义 +82（P8-01..04 +61、P8-05 净 +21），
  其余为参数化展开口径差。
  不变量结论（1323/5/614；skipped/warnings 不增长；top-2 杠杆结构）
  在修正前后保持一致。

## PR3（2026-08-31，codex/test-runtime-b2c2-xdist，pytest-xdist 试点）

### 依赖契约（B2）

- `pytest-xdist==3.8.0`（+`execnet==2.1.2`）与 `baostock==0.9.3` 进入
  `requirements-optional.in/.txt`（freeze_lock 重生成，`.txt` 恰 +2 行）。
- baostock 为**活契约消费者**（R2-F1 对本节初稿"无消费者"声明的更正）：
  `ashare_data/akshare_client.py:532` 生产 fallback 链、
  `scripts/import_pit_universe.py:139/212` 懒加载、
  `tests/test_akshare_client.py:246` 无 skip 的无条件 import——契约闭包
  必须包含，否则全量收集即失败。仓内历史全量运行均在含 baostock 的
  miniconda 超集环境；旧 lock 未含 baostock 属 lock 陈旧，而非"无消费
  者"。matplotlib/seaborn 等残留经 grep 证实零代码 import，判断不变。
- `requirements.lock` 从干净 CI 忠实 venv 解释器重生成：attempt-1 parity
  中 `test_lock_files.py::test_lock_file_is_a_full_freeze` 正确拒绝了
  "保留陈旧 lock"的捷径（该守卫实证有效）。lock 逐项变化（R2-F4 列明）：
  `execnet==2.1.2`、`baostock==0.9.3` 新增；`torch==2.11.0+cu128 →
  2.11.0+cpu`（旧 lock 生成自 cu128 环境，按 P0-06 CPU-base 契约修正）；
  `pydantic==2.13.4 → 2.13.3`（对齐 requirements.txt L18，旧值属陈旧
   漂移）；`idna==3.18 → 3.19`、`joblib==1.5.3 → 1.6.0`（其新硬依赖
   `cloudpickle==3.1.2` 随之新增）及 click/curl_cffi/filelock/lxml/
   narwhals/platformdirs、`scs 3.2.11 → 3.3.0`、`protobuf 7.35.1 →
   7.36.0`、`Pygments 2.20.0 → 2.21.0` 等传递版本随干净闭包解析更新
   （R2-F6 更正：scs/protobuf/Pygments 系 cvxpy/streamlit 链必需传递
   依赖的**版本上跳**，仍在 lock 中，非移除；实现 commit 3bdac28 的
   message 中同一清单有相同笔误，以本日志为准）；移除经 grep 证实零
   代码 import 的环境残留（matplotlib/seaborn 及 matplotlib 依赖族
   contourpy/cycler/fonttools/kiwisolver/pyparsing）。
  `freeze_lock --check` 与干净环境 `--check-full` 双通过（后者修复前
  持续失败——lock 卫生实质改善）。
- `pip check`：干净 venv 通过（含全部新 pin）；共享 miniconda 仍失败
  （既有状态，见 PR1 披露）。

### C2 日志竞态修复

- RED：`tests/test_logging_parallel.py` 3 项预期失败（`worker_log_suffix`
  缺失 → ImportError；.log 命名无 worker 后缀 → 同秒碰撞；导出崩溃损坏
  目标文件）；命令 `python -m pytest -q tests/test_logging_parallel.py`。
- GREEN：`worker_log_suffix()`（读 `PYTEST_XDIST_WORKER`，serial 为空串，
  历史命名字节兼容）；`export_log_txt` 原子化（tmp + `os.replace`，失败
  清理 tmp）；conftest 每 worker 导出 `logs/pytest_gwN.txt`（serial 保持
  `logs/pytest.txt`）。GREEN 4/4（含既有 `test_logging.py` 串行契约）。
- 实证：xdist smoke `pytest -n 2` 8 passed；`logs/pytest_gw0.txt` /
  `pytest_gw1.txt` 每 worker 导出观测。

### C4 关联缺陷：jobmanager 冲突检测的跨秒竞态（并行门禁暴露）

- 现象：attempt-2 parity（b027d2e 树）中
  `test_jobmanager.py::test_start_conflict_while_running` 失败
  （DID NOT RAISE），第二个 start() 偷锁并双 spawn（快照 stderr 双
  "Simulation started" 行）。
- 根因：`_spawn` 在 `Popen` 返回**之后**采样 `started_at`（`_now_iso()`
  为秒级截断），而 `_is_alive` 的 PID 重用守卫把
  `create_time < started_at` 判为非本进程——Popen→采样窗口跨秒时
  （并行负载下窗口拉大）守卫误杀自己的活子进程。attempt-1/闭包运行
  绿 = 低负载下概率低；串行历史全量均未触发。
- RED（确定性）：`test_start_conflict_survives_second_boundary_spawn`
  —— Popen 后延迟 1.1s 强制跨秒，修复前必红（1 failed / 7.65s）。
- GREEN：`started_at` 采样移到 `Popen` 之前（守卫语义不变：仅拒绝创建
  于 run 启动之前的外来进程）；jobmanager 模块 16/16（9.33s）。
- 该缺陷为既有产品缺陷，由门禁并行化暴露并构成其可交付前提（与 C2
  同类处理）；闭包与后续并行运行均依赖此修复。

### 门禁命令同步

- AGENTS §10.2 第 3 步 → `python -m pytest -q tests -n auto`，附 fail-closed
  条款：仅当当次 PR 的 parity 对账已记入本日志时可用，否则回退串行命令；
  CI job 命令不变。onboarding §9.2 同步。drift guard 扩展
  （`test_local_full_suite_gate_runs_parallel`，RED 1 → GREEN 5/5）。
- PR2 评审两条 low 观察（docstring 条款定位、§9.2 切片加固）顺手修正并
  在实现 commit 中披露。

### parity 对账（被测实现 SHA 3bdac28，`-n auto` = 6 workers）

- worker 数更正（R2-F3）：xdist `-n auto` 取 psutil **物理核** = 6
  （本机逻辑核 12；gw0-gw5 六份 worker 导出佐证），非 12。
- 命令：`python -m pytest -q tests -n auto`（完整 stdout+stderr 重定向
  → `logs/pytest_parity_3bdac28.txt`）。
- 结果：**1332 passed / 5 skipped / 615 warnings / 275.64s（4:35）**。
- 计数对账（R2-F2 修正）：1332 = 1327（PR2 门禁）+ 3
  （`test_logging_parallel.py` 新契约）+ 1
  （`test_local_full_suite_gate_runs_parallel` drift guard）+ 1
  （`test_start_conflict_survives_second_boundary_spawn` 回归测试）；
  skipped 5 持平。
- warnings 口径：615 = 614 + 1。与 PR2 门禁快照做 warnings 段集合差：
  唯一差异为 `test_evaluation` 组 7→8（once-per-process 语义的 warning
  在多 worker 下重复实例），**warning 种类/消息/位置零新增**。据此确立
  可比口径：warnings"不增长"不变量在并行口径下按去重
  (类别, 消息模板, 调用位置) 集合判定（裸总数在 worker 数变化时不可直接
  比对）；该口径由 PR4 的归并 job 机器化。
- 提速：275.64s vs B1 串行基线 740.36s = **2.69x**（vs 711.37s 门禁运行
  = 2.58x）。
- 尝试史（全部留证）：attempt-1（f964c3a 树，281.67s，1 failed = lock
  契约违约，`logs/pytest_parity_attempt1_lockviolation.txt`）；attempt-2
  （b027d2e 树，307.09s，1 failed = jobmanager 跨秒竞态，
  `logs/pytest_parity_b027d2e.txt`）；attempt-3（3bdac28 树，本文数字，
  绿）。
- 干净闭包全量（R2-F1 证据）：venv 解释器 `pytest -n auto` 于 b027d2e
  树 → **1331 passed / 5 skipped / 615 warnings / 323.93s**，快照
  `logs/pytest_closure_b027d2e.txt`（含 test_akshare_client 的 baostock
  无条件 import）——契约闭包可独立运行全量套件，取代本节初稿失真的
  "无缺失"声明。

### 最终候选门禁

- 于最终候选（本条目提交后的 HEAD）运行：串行全量 pytest（严格 CI 等
  价复现）、web job（npm ci/ls/build）、compileall -j 0、freeze_lock
  --check、git diff --check、干净环境 pip check；结果记入 merge commit
  与评审记录。

## PR4（2026-08-31，codex/test-runtime-d1d2-ci-shards，CI 分片 + warning 基线）

### D1/D2 实现

- `scripts/check_test_shards.py`：88 个测试文件显式四分片（data 23 /
  model-1 17 / model-2 29 / portfolio-trading 19），`--check` fail-closed
  （遗漏/重叠/未知文件即红），`--emit` 输出分片文件清单。开发过程中该
  守卫抓到 `test_ci_sharding.py` 自身未分片——守卫有效性实证。
- ci.yml：test job → 4-leg matrix（每腿含 pip check / freeze_lock
  --check / 并集守卫步 / 分片 pytest + tee 完整日志上传）；
  `test-warning-merge` 归并 job（needs: test）下载全部分片日志并强制
  warning 基线（净新增行 = 红）。web job 不变。
- `scripts/ci_warning_merge.py`：解析 warnings summary 段 → 行集合对比
  基线；net-new = 红（须在测量日志逐项解释并 `--write-baseline` 重新
  基线）；disappeared = 信息性报告。缺分片日志 = fail closed。
- `docs/ci_warning_baseline.json`：生成自 `logs/pytest_gate_4156de4.txt`
  （4156de4 串行 CI 等价门禁：1332/5/614/730.02s），provenance 在案。
- RED：`tests/test_ci_sharding.py` 9 项预期失败（脚本缺失、并集未覆盖、
  validate 三类错误、emit CLI、CI 同步、净新增检测、缺日志 fail-closed、
  基线结构）；GREEN 10/10；相邻守卫 21 passed。

### parity（被测实现 SHA 3e82455，-n auto = 6 workers）

- 命令：`python -m pytest -q tests -n auto`（完整 stdout+stderr →
  `logs/pytest_parity_3e82455.txt`）。
- 结果：**1342 passed / 5 skipped / 615 warnings / 272.18s（4:32）**。
- 计数对账：1342 = 1332（PR3 门禁）+ 10（`test_ci_sharding.py` 新契约，
  其中 1 项在 RED 阶段即绿：脚本缺失时 emit 的 returncode≠0 断言）；
  skipped 5 持平；warnings 615 = 614 + 1（PR3 已口径化的 once-per-process
  重复实例，集合级无新增种类）。
- 提速：272.18s vs B1 串行基线 740.36s = **2.72x**。

### D2 检测器的自我验证（诚实记录）

- 对 xdist parity 日志直接跑 `ci_warning_merge.py --check`：**exit 1，
  精确命中两处 PR3 已口径化的已知差异**（`test_evaluation` 组 7→8 的
  once-per-process 重复实例 + 汇总行 615/614）——检测器灵敏度实证。
- CI 形态（每腿串行、无 xdist）不存在该重复；本地 CI 等价校验使用串行
  门禁日志（见下），预期与基线精确一致。

### R3 更正（评审 R3-F1/F2/F3）

- R3-F1：warnings 行集合比较存在环境绑定缺陷（基线含 7 行本机绝对
  路径：miniconda×4、D:\minequant×3；CI linux 路径首跑必红——评审
  内存模拟实证）。修复：`parse_warnings_section` 出口做位置归一化
  （分隔符统一 `/`；第三方锚定 `site-packages/`；in-repo 锚定包/工具
  前缀），基线以同口径重生成（49 行归一化集合，614 条实例），新增
  契约测试（基线无盘符/miniconda 前缀；windows/linux/第三方三形态
  归一化断言）。本地自洽校验：gate_4156de4 归一化对比 = 0 net-new。
- R3-F3a：`discover_test_files` 改递归（未来 tests/ 子目录文件不再
  逃逸守卫与 CI）；R3-F3b：`ci_warning_merge --expect N` 断言分片日志
  份数（CI 传 4），份数不符 fail closed。
- R3-F2：F5 文本行数勾稽更正（50 → 49，汇总行排除后口径）。

### 最终候选门禁

- 于最终候选（本节提交后的 HEAD）运行：串行全量 pytest（= 4 个 CI 分片
  腿的并集的严格等价复现）+ 对其日志的 D2 归并检查（本地等价于
  `test-warning-merge` job）、web job、compileall -j 0、freeze_lock
  --check、git diff --check、干净环境 pip/--check-full；结果记入 merge
  commit 与评审记录。

## 追加运行（2026-08-31，principle-7 候选 9600160 + D2 比较器精化）

- 命令：`python -m pytest -q tests -n auto`（完整 stdout+stderr →
  `logs/pytest_gate_9600160.txt`）。
- 结果：**1347 passed / 5 skipped / 615 warnings / 386.87s（6:26）**
  （机器负载高于 parity 时段；墙钟波动，计数不变量成立）。
- D2 比较器精化（R3-F1 后续）：对 xdist 日志的本地检查按设计命中
  PR3 已口径化的组计数头差异（7→8），暴露比较器与"去重
  (类别, 消息模板, 调用位置) 集合"口径的偏差 → RED
  `test_warning_merge_ignores_group_count_headers` → 修复：组计数头行
  （`tests/...: N warnings`）自两侧排除，位置/消息行（warning 种类
  载体）保留；基线同口径重生成（614 warnings，38 行归一化种类集合，
  provenance 不变）。
- 复核：修复后同一 xdist 日志 D2 检查 = 0 net-new（exit 0）——本地
  `-n auto` 门禁与 warning 基线完全兼容；CI 串行分片形态同理。
