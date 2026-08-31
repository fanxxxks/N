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

- `osqp` `PendingDeprecationWarning`（raise_error 默认值将变）：
  test_bare_factor_backtest 500 / test_portfolio_optimizer 33 /
  test_p3_portfolio_parity 16 / test_golden_parity 2 /
  test_portfolio_constructor 2，合计 553；
- `numpy` `RuntimeWarning`（invalid value in scalar subtract / divide）：
  test_train 20、test_searcher_bench 10、test_core 2、test_evaluation 7、
  test_candidates 1、test_p3_measurement 1、test_universe 1、
  test_diagnostics 5 测试聚类、test_tier_reports 1 聚类；
- `ashare_model/reward.py` `RuntimeWarning`（All-NaN slice / Mean of
  empty slice）：test_evaluation 9；
- `UniverseDevelopmentFallbackWarning`（data_loader 开发兜底显式告警）：
  test_manifest 1。

以上按 pytest warning summary 的（文件/测试聚类）粒度转录，已与完整快照
（`pytest_b1_baseline_cbda238_full.txt` L20-75）逐组核对一致（R1 复核）；
(类别, 消息模板, 调用位置) 多重集的机器可读再基线在 PR4 落地。

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
