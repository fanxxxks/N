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
