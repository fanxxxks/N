# P7 可维护性治理测量日志

本文只记录实际发生的测量与验证（命令、结果、不变量），计划见
`docs/p7_maintainability_plan.md`。环境：Windows / Python 3.13.12 / CPU 路径
（`torch.cuda.is_available()` 视机器而定）；工作目录 `D:\minequant\AlphaGPT`。

## 基线（Phase A0，2026-07-XX，main @ 5a7c2b3 + A0 合同测试）

- 命令：`python -m pytest -q tests`
- 结果：**1107 passed, 5 skipped, 618 warnings, ~614s**
- 该数字是 Phase A/B 每片 PR 的对比不变量：passed 只允许随新增合同测试
  增加，skipped/warnings 不得增长。

## Phase A：evaluation.py 拆分（A1–A5）

每片验证命令相同：`python -m pytest -q tests`；
`python -m compileall -q ashare_data ashare_model ashare_portfolio ashare_trading
scripts webapi`；`git diff --check`。

| PR | 内容 | 全量结果 | 墙钟 | evaluation.py 行数 |
|---|---|---|---|---|
| A0 | AGENTS.md 入库 + 计划 + facade 面契约测试 | 1107 / 5 / 618 | 614s | 2405 |
| A1 | eval_folds.py（Fold/FoldData/resolve_folds/epoch_slice/search_window_id） | 1109 / 5 / 618 | 611s | 2233 |
| A2 | eval_metrics.py（指标/拼接 + METRIC_KEYS） | 1110 / 5 / 618 | 621s | ~1760 |
| A3 | eval_corrections.py（DSR/max-t/self-check） | 1111 / 5 / 618 | 621s | ~1450 |
| A4 | eval_search.py（baselines/run_fold/三搜索后端） | 1112 / 5 / 618 | 616s | ~990 |
| A5 | eval_artifacts.py（build_result/provenance/ledger seam） | 1113 / 5 / 618 | 632s | ~755 |

passed 递增全部来自 `tests/test_eval_module_split.py` 每片新增的身份断言
（facade 与抽出模块为同一对象）。warnings 全程 618 未增长；skipped 全程 5。

monkeypatch 面保持不变量（测试直接验证）：

- `evaluation.CandidateScorer`（类级 patch）——facade 保留 re-export；
- `evaluation.REWARD_VERSION` —— `eval_artifacts.build_result` 运行时经
  facade 惰性查找；
- `evaluation._build_trainer` —— `eval_search.run_fold` 运行时经 facade 惰性查找；
- `evaluation.PROTOCOL_VERSION` —— `_search_evaluator`/`build_result` 运行时经
  facade 惰性查找。

未运行项：无（每片均完成全量 pytest + compileall + diff check）。
研究结论：本阶段为纯工程变更，不产生也不声称任何研究结论。

## Phase B：train.py 拆分（B1）

| PR | 内容 | 全量结果 | 墙钟 | train.py 行数 |
|---|---|---|---|---|
| B1 | train_windows.py（窗口/采样自由函数） | 1115 / 5 / 618 | 603s | 1688 → 1540 |

- B1 前置检查：tests 对 `train_module.batched_basket_rewards` 的 patch 面不动，
  `AshareTrainer` 留 facade；聚焦 111 passed / 2 skipped（test_train 等 6 文件）。
- **B2 放弃**（计划 §4 已记录）：`_training_contract`/`prepare_window`/
  `_policy_update_loss` 被 tests 直接调用（方法面即测试面）；
  `_write_artifact`/`_build_rl_search_result` 与 15+ 实例状态紧耦合，位移
  不产生边界。trainer artifact 写入侧的类型化边界归 Phase C 处理。
- 契约测试新增 `tests/test_train_module_split.py`（面 + 身份 2 个测试）。

## Phase C：类型化产物（C0–C2）

契约：`docs/p7_artifact_schema_contract.md`（预注册，先于代码）。

| PR | 内容 | 全量结果 | 墙钟 |
|---|---|---|---|
| C0 | requirements.in += pydantic（必要性说明见提交）；**requirements.lock 有意不重生成**——它是单机快照（freeze_lock.py 文档），本机重生成会夹带 torch cu128→cpu 等环境漂移进 PR | freeze_lock --check 通过；test_lock_files 7 passed | — |
| C1 | `artifact_schemas.py`（ARTIFACT_SCHEMA_VERSION=1、Strategy/Protocol 模型顶层 forbid、classify_schema_version/apply_schema_matrix 单一入口）；写入侧 `_write_artifact`/`build_result` stamp+validate；读取侧 run_sim/backtest/load_trial_rows 接矩阵 | 1143 / 5 / 618 | 559s |
| C2 | BacktestResultArtifact / PaperStateArtifact(+Position/Equity)；portfolio save stamp+validate、load fail-closed（损坏/未知版本 raise 且不覆盖；legacy 宽容读入、下次 save 迁移） | 1157 / 5 / 618 | 614s |

- C1 RED 过程：28 个契约测试先红（模块不存在）；其中 1 处测试自身笔误
  修正（`SimulationRunner` 类名，非 `AshoreSim`——引用 run_sim.py:85 实现，
  属白名单"测试笔误"）。
- C2 前置检查：grep 确认无既有测试依赖旧的 `except Exception → reset()`
  fail-open 行为（只有显式 `reset()` 调用，保持合法）。
- 产物证据（契约 §7）：`data/best_ashare_strategy.json`（已打 legacy 戳）与
  `data/protocol_result.json`（无版本键）在实现后经
  `classify_schema_version` 判定为 legacy、读取路径不变
  （`test_on_disk_artifacts_classify_legacy`）。
- 运行态变更声明（契约 §5 已预注册）：模拟 resume 语义 fail-closed 化。
  warnings 全程 618 未增长；skipped 全程 5。
- 未运行项：无。研究结论：本阶段无研究语义变化，不产生研究结论。

## Phase D：单一注册表（D1–D3）

轻量契约：主计划 §6.1（预注册，先于代码）。

| PR | 内容 | 全量结果 | 墙钟 |
|---|---|---|---|
| D1 | `feature_metadata.py`（62 特征著录：availability/hypothesis/direction/semantic_type/promotion）+ `FeatureRecord` 九字段 + 派生（horizon←P6 域、cost/depends_on←FACTOR_REGISTRY）；`FEATURE_REGISTRY_VERSION` 2→3；缺著录 fail-closed | 1167 / 5 / 618 | 612s |
| D2 | `operator_registry.py`（39 算子：类别/逐参数输入语义/输出语义/成本/稳定性备注；arity 从 OPS_CONFIG 派生，import 断言对齐） | 1179 / 5 / 618 | 604s |
| D3 | `scripts/generate_registry_docs.py` 生成 `docs/feature_registry.md` + 漂移守卫（生成物≠注册表即红）；onboarding §5.1/§5.2 手工名单退役 | 1181 / 5 / 618 | 603s |

- D1 著录内容核对路径：62 特征逐条对照 `FactorSpec.description`（计算定义）
  与 fundamentals/capital_flow 数据源；语义类型分布
  price 5 / return 19 / volume 11 / fundamental 13 / cross-sectional 8 /
  event 6 = 62；`KURT_20`、`RSQ_60` 如实著录方向 0（无明确预期）。
- 单一权威核对：expected_horizon 全部由 `RESEARCH_DOMAINS` 派生
  （61 个活跃特征恰属一个域，NORTHBOUND_CHG 为 None）；
  `tests/test_feature_metadata.py::test_expected_horizon_derives_from_research_domains`
  同时是 P6 划分的回归守卫。
- D3 生成物：`docs/feature_registry.md`（131 行，`--check` 同步）；
  onboarding 手工算子名单删除由测试防复活。
- `test_registry_version_is_pinned` 2→3：需求变更路径，引用 §6.1。
- 未运行项：无。研究结论：本阶段全部为描述性元数据与文档，无搜索/评分/
  晋级语义变化，不产生研究结论。
