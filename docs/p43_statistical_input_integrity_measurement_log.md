# P43 统计输入、数值与证据绑定收口 — 测量日志（工作包 7）

- 契约：`docs/p43_statistical_input_integrity_draft.md`（APPROVED，用户
  2026-09-20 直接指令工作包 7）。
- 被测实现：本分支同一包的四个提交（契约 `6140f19` → RED-1 `9b0dd5d` →
  GREEN-1 数值 `4e41e7a` → RED-2 `527e0c5` → GREEN-2 身份 `b6b1eac`）。
  本日志为 evidence-only 后置记录，全量验证跑在精确候选树 `b6b1eac` 上。
- 环境：Windows 10 22621 x64，Python 3.13.12（`D:\minequant\.venv`），
  CUDA 可用（torch 报告 `cuda.is_available() == True`，见 §5 skip 差异）。
- 基线：分支 `codex/b04-load-cache-performance`，p42 完成后 `27a5285`，
  工作树干净（仅既有未跟踪 `.mimosa/`，未触碰）。

## 1 反例复现（勘察，RED 之前，探针于基线树）

| 反例 | 基线行为 |
|---|---|
| 四条噪声序列 max-t（矩阵接口，n_perms=500） | p = 0.9581 |
| 同矩阵仅 [0,7] 改 NaN | **p = 0.0020（= 1/501，最小可达 p）** |
| 行接口（`max_t_from_rows`）干净 / 单 NaN daily | 0.2555 → **0.0020** |
| 基准序列单 NaN | **p = 0.0050** |
| `dsr_from_rows` 单 NaN 输入 | dsr = **NaN**，且 `NaN < 0.95` 为假 → G2 通过 |
| G2 复合伪造（p=0.90 且 significant_95=True、run=engineering_run、1999 年 3 天窗口、csi300 基准、试错集 99/99/198，候选/seed 正确） | significance 门 **passed=True, reasons=[]** |

## 2 RED

- RED-1（数值，`9b0dd5d`）：`tests/test_p43_statistical_input_integrity.py`
  21 失败 / 3 过（正向对照）。证据 `reports/p43_red1_evidence.txt`。
- RED-2（身份，`527e0c5`）：追加 B5–B9 后 12 失败 / 25 过。证据
  `reports/p43_red2_evidence.txt`。其中正向对照
  `test_b9_deterministic_candidate_binds_with_seed_none` 在 RED 阶段即失败，
  暴露 p41 既有缺陷：确定性基线（`baseline:RET_1`，构造上 seed=None）的
  **真实**统计证据被 G2 以 "lacks the statistical identity
  (adjudicated_seed)" 拒绝——seed=None 与字段缺失混同，伪造 seed 成为
  过门唯一途径（即用户交付要求第 5 条的实证依据）。

## 3 GREEN 与聚焦/相邻验证

- GREEN-1（`4e41e7a`）：N1 stitch 非有限拒绝、N2 矩阵有限性、N3 参数
  合法性（n_perms/block_size/seed 非 bool 整数、≥1）、N5 G2 数值判据
  （dsr 有限 ∈[0,1] 且 ≥ 阈值、p ∈(0,1]、significant_95 必须 bool 且
  ≡ p ≤ 阈值）；`PROMOTION_RULE_VERSION` 3→4；两个版本 pin 按契约 §5
  （§10.1 规则 2）同步为 "4"。
- GREEN-2（`b6b1eac`）：`_identity_content_reasons`（B5 运行类型 /
  B6 窗口对 top 试验日期轴 / B7 基准定义与 span / B8 试错集内部与
  artifact 自身证据闭合）+ `candidate_seed_class`（B9：确定性族
  benchmark/baseline/selfcheck/control/combo（含 `cell::` 前缀）强制
  seed=None，其余含未知族强制整数 seed）；`adjudicated_seed` 存在性改为
  按键检查，修复 §2 所述 p41 缺陷。
- 聚焦：p43 37P。相邻：promotion+p41+p40+calibration+stitched_oos 85P；
  module_split+schemas+versions+completion_gates+baseline_harness 102P；
  r01/r02/r03/r42 批次 60P；test_evaluation 51P（GREEN-1 时点）。
- `python -m compileall -j 0 -q …` 通过；`python scripts/check_test_shards.py
  --check` 118/118；`git diff --check` 干净。
- 复合反例复核（GREEN-2 后手工探针）：同 §1 末行构造 → significance 门
  passed=False，11 条原因分别指名运行类型、窗口（dsr 相等性 + max_t
  p40 下限 + 包含关系）、基准定义、基准 span、试错集与 artifact 证据
  不符（p 值/布尔矛盾因"绑定先于数值"被身份失败先行拦截，符合设计）。

## 4 全量回归（精确候选树 `b6b1eac`）

- 第 1 次 `python -m pytest -q tests -n auto`（19 min）：2095 passed /
  **1 failed**（`test_b03_campaign_executor.py::test_r4_killed_driver_tree_
  resumes_with_old_evidence_preserved`）/ 1 skipped / 3 xfailed。
- 第 2 次（31 min，与 b03 整文件复跑、基线 worktree 运行部分并发）：
  2095 passed / **1 failed**（`test_task_executor.py::test_supervisor_
  detects_heartbeat_timeout_and_continues`）/ 1 skipped / 3 xfailed。
  完整日志 `reports/p43_full_suite_run2.log`。
- 归因（两次失败为**不同**测试，同属进程树/心跳墙钟家族）：
  1. 单测隔离复跑均通过：b03-R4 单跑 118s 通过；heartbeat 单跑 27s 通过；
     b03 整文件无负载复跑 23P（352s；负载下同文件 472s 且 1F）。
  2. 该两文件不导入本次任何被改模块（b03 仅 campaign_executor/
     task_executor/ledger/run_store；task_executor 同族）。
  3. 基线 worktree（`27a5285`，p43 之前）b03 整文件 23P（464s）。
  4. 结论：`-n auto` 满载下进程/心跳墙钟预算被挤破的时序波动，非 p43
     回归；p43 聚焦与全部相邻批次 0 失败。未做进一步治理（属 b03/p32
     测试基础设施域，超出本包范围，登记为后续小任务候选）。
- 并行 vs 串行对账：本包未 push，按 §10.2 全量为本地并行；push 前需完成
  parallel-vs-serial parity 记账（与 p40/p41 日志同一欠账口径）。

## 5 skip 差异说明

p42 记录 6 skipped；本机本次 1 skipped。差 5 项为
`@pytest.mark.skipif(not torch.cuda.is_available())`（test_ops/test_train/
test_vm 共 5 个）——本机 CUDA 可用故执行并计入 passed；余 1 为
`test_lock_files` 的 `ALPHAGPT_LOCK_AUDIT_STRICT` 环境门 skip。总数对账：
2063（p42 树）+ 37（p43）= 2100 = 2095+1+1+3。✓

## 6 裁决

- 验收达成：坏数据（NaN/Inf、非法参数、矛盾布尔、伪造身份）一律明确
  失败且不产出显著结论；有限合格输入行为逐位不变（正向对照 + 全量
  2095 passed）。
- 研究声明边界：本包为统计完整性工程收口，不产生任何 alpha/晋级证据；
  G2 变严只可能拒绝此前会被误判的 challenger（v4 verdict 记录
  `promotion_rule_version="4"`）。既有 R01/R02/R03 artifact 未重写，
  其 dsr/max_t 块按原版本保留。
- 后续（非本包）：误报率校准的置信区间与留出验收（用户简报 §四 的
  方法学项）；b03/心跳类测试在满载下的时序预算治理。
