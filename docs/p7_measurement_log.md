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

## Phase E：搜索采样语义类型（E1–E3，研究语义变更）

预注册契约：`docs/p7_semantic_types_contract.md`（先于实现提交）。测量基线为
Phase D3 的 **1181 / 5 / 618**；E1 基线提交为 `df0e340`。本次 E2/E3 在
`codex/p7-e2-typed-sampling-completion` 的待提交工作树测量；用户未跟踪草稿
`docs/p5_implementation_plan.md` 始终排除在修改、暂存和提交之外。

| 片段 | 内容 | 契约测试增量 | 验证结果 |
|---|---|---:|---|
| E1 | `semantic_sampling.py`：六类类型格、注册表驱动签名/输出解析、共享序列校验器 | 17 | E1 聚焦 17 passed（提交 `df0e340`） |
| E2/E3 | RL/Random/TPE 类型栈与掩码；开布尔 d1/d2/d3 逃生界；GP typed pset/受限域确定性生成；无合法 token fail-closed；v3/v2/v25 版本与 legacy 拒绝 | 14 | 最终全量 **1212 passed / 5 skipped / 614 warnings** |

RED 证据与修复裁决：

- 继承工作树聚焦命令（下列 11 个文件）最初为 **52 failed / 160 passed /
  2 skipped / 16 warnings，103.64s**；首因是 `build_action_mask` 的未定义
  `feasible` 与死代码，失败随后扩散到 grammar、Random/TPE、trainer 和
  evaluation。
- 新增的布尔深度/预算边界测试先为 **2 failed / 13 deselected**：事件生产者
  必须能沿 `event, x, y, GATE, EOS` 的最短五-token 路径完成；d1/d2/d3 的
  post-action 紧致预留分别为 `s+2` / `s` / `s-2`。交接草稿中把 d2 重复写成
  d1 的界会在等号边界制造空合法集，因此按“契约不变量 > 测试 > 实现”以
  可终止性属性为仲裁，未照抄该笔误。
- Random 的空合法集测试先红（旧实现退回全词表均匀采样），现改为带 step/row
  诊断的显式 `RuntimeError`；无 skip/xfail/retry、无弱断言、无 warning filter。
- 测试白名单修正 1 处：`py_random` 原本只在前一个测试函数内导入，后一个
  测试直接引用而 `NameError`；提升为模块导入，seed、样本数和断言均不变。

最终命令与结果：

- 聚焦：
  `python -m pytest -q tests/test_semantic_sampling.py tests/test_alphagpt.py
  tests/test_grammar.py tests/test_vocab.py tests/test_gp_search.py
  tests/test_tpe_search.py tests/test_searcher_bench.py tests/test_train.py
  tests/test_artifact_versions.py tests/test_p4_search_contract.py
  tests/test_evaluation.py` → **215 passed / 2 skipped / 40 warnings，330.77s**
  （随后仅增加契约测试并抽取共享状态推进；最终态由下述全量覆盖）。
- 最终全量：`python -m pytest -q tests` → **1212 passed / 5 skipped /
  614 warnings，562.68s**。相对 D3：passed +31（E1 +17，E2/E3 +14），
  skipped 不变；warnings 实际减少 4、未增长，且未增加过滤器。警告类别均为
  既有 OSQP/NumPy/reward/development-universe 警告。
- `python -m compileall -q ashare_data ashare_model ashare_portfolio
  ashare_trading scripts webapi` → exit 0；`git diff --check` → exit 0。
- 资源：全量墙钟如上；未单独采集峰值内存（未声称内存测量通过）。本阶段无
  训练、真实 DuckDB、research 或 promotion 运行；dataset ID / config hash /
  研究 artifact 路径均不适用，测试仅使用固定 seed 的 engineering fixtures。

版本与迁移不变量：`GRAMMAR_VERSION 2→3`、
`SEARCH_CONTRACT_VERSION 1→2`、`PROTOCOL_VERSION "24"→"25"`；Reward、
Model、Feature Registry、Data Tier、Execution、Constructor 均未 bump。旧公式
仍按名解析且 VM 可执行；v24 strategy/protocol 由既有 artifact classifier
明确判 legacy，不能冒充 current。四搜索器的输出均经共享类型规则验证，GP
保持 `len(tree_to_tokens(tree)) == len(tree)` 与统一长度上限。

研究裁决：这是搜索合法空间的预注册收缩，只证明软件不变量与工程可运行性。
类型化前后的搜索结果**不可宣称 matched comparison**；未运行 OOS、成本压力、
统计校正或未来 paper window，因此不产生 alpha、晋级、production 或实盘就绪
结论。
