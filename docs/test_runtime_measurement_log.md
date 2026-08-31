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
- 结果：**1323 passed, 5 skipped, 614 warnings in 779.26s（0:12:59）**
- 5 skipped 均为登记在册的 CUDA skipif 基线（test_ops/test_train/test_vm），
  与 p7 终态持平。
- 原始产物：`logs/pytest.txt`（本次会话导出），快照留存于
  `logs/pytest_b1_baseline_cbda238.txt`。

### 与 p7 终态基线的对比（诚实记录，不下结论）

| 口径 | p7 终态 | B1 基线 | 变化 |
|---|---|---|---|
| passed | 1212 | 1323 | +111（P8-05 新增合同测试） |
| skipped | 5 | 5 | 持平 |
| warnings | 614 | 614 | 持平 |
| 墙钟 | 562.68s | 779.26s | +216.58s |
| 平均每测试 | 0.464s | 0.589s | +26.9% |

墙钟增长的分解（测试数增加 vs 每测试变慢）未做归因实验；本表只记录
观察。B1 数字自此取代 p7 终态成为本 campaign 的对比基线。

### 最慢测试（--durations=100 的头部，全量墙钟的结构性事实）

| 测试 | 耗时 | 占全量 |
|---|---|---|
| test_evaluation.py::test_run_protocol_rows_and_determinism | 135.69s | 17.4% |
| test_evaluation.py::test_run_protocol_records_universe_policy | 82.90s | 10.6% |
| test_grammar.py::test_random_samples_execute_100_percent | 21.65s | 2.8% |
| test_evaluation.py::test_cli_smoke | 17.42s | 2.2% |
| test_evaluation.py::test_cli_confirmation_smoke | 16.78s | 2.2% |
| test_archive_run.py::test_run_dir_collision_gets_suffix | 13.50s | 1.7% |
| test_data_loader.py::test_industry_codes_all_nan_without_membership | 8.71s | 1.1% |
| test_promotion.py::test_cli_promotion_refuses_legacy_artifact_without_dataset | 8.53s | 1.1% |
| test_archive_run.py::test_protocol_manifest_records_actual_run_scope | 8.33s | 1.1% |
| test_archive_run.py::test_formula_derived_slug_is_sanitized | 7.91s | 1.0% |

top-2 = 218.59s（28.1%）；top-10 = 321.42s（41.2%）。单条 135.69s 的
协议行测试是分片均衡与并行收益的最大杠杆；test_archive_run.py 以数量
取胜（多条约 7–13.5s）。

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

以上按 pytest warning summary 的（文件/测试聚类）粒度转录；(类别, 消息
模板, 调用位置) 多重集的机器可读再基线在 PR4 落地。

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
| `python -m pip check`（共享 miniconda 环境） | **失败（任务前既有状态，非本任务引入）**：`langchain-community 0.3.24`↔`langchain 1.3.4`、`langchain-core 1.4.0`、`httpx2`↔`idna 3.11` 等冲突；全部涉事包（langchain 系、idna）经 grep 证实**不在任何 requirements 契约文件内**（.in/.txt 均 0 匹配） |
| `python -m pip check`（CI 忠实复现：%TEMP% 干净 venv，仅装 requirements.txt + requirements-optional.txt） | **通过**（"No broken requirements found"）——与 CI 的干净安装语义一致 |

环境偏差披露（均为任务前既有，已记为后续独立小任务线索）：

1. 共享 miniconda 环境存在契约外包（langchain 栈、旧 idna），导致本地
   `pip check` 失败；CI 等价复现须在干净 venv 中执行（本条目已这样做）。
2. `python scripts/freeze_lock.py --check-full` 失败：`requirements.lock`
   全量快照相对当前环境漂移（同因：契约外包）。pin 文件（CI 门禁对象）
   是同步的；lock 卫生修复不在本 campaign 范围。

合并前在最终候选 commit 上重跑同套门禁的结果记录于 merge commit
message 与评审记录（测量日志只承载测量与门禁命令矩阵本身）。
