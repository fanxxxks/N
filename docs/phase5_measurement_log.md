# Phase 5 measurement log (P1-01 .. P1-05: 低成本测量与成本诊断)

> 规则：改动前先 commit 当前状态 → 分支开发 → 测试 → 验证 → 合并回 main。
> 测试计数为仓库根目录 `pytest tests`（排除 `tests/test_webapi.py`：既有环境阻塞，
> starlette 1.3.1 的 TestClient 需要不可离线安装的 `httpx2`）。
> 本阶段为**纯测量与成本诊断**：所有产出是费用矩阵、固定回测、搜索器成本报告与
> selfcheck 空转验收；**不宣称发现 Alpha**。

| Stage | Commit | Tests passed | Δ | Notes |
|---|---|---|---|---|
| Baseline (main, pre-P1) | 2eaba93 | 921 | — | Phase-0 收尾计数（`logs/pytest_phase0_final.log`），本次基线复跑 913（webapi 排除）+ 8（webapi）= 921，`logs/pytest_p1_baseline.log` |
| P1-02 fee matrix | `TBD` | 932 | +11 | 资金×持仓数×换手率费用矩阵（FEE_MATRIX_VERSION 1，`tests/test_cost_matrix.py`） |
| P1-03 bare-factor fixed backtest |  |  |  | 七裸因子仅固定回测（BARE_FACTOR_BACKTEST_VERSION 1） |
| P1-04/05 searcher bench |  |  |  | 四搜索器成本测量 + 300×400 一折一种子小预算 smoke（SEARCHER_BENCH_VERSION 1；train_search 契约扩展 tpe） |
| P1-01 selfcheck run |  |  |  | 真实数据 selfcheck：DSR 不显著、max-t 不显著 |
| P1 runs + docs |  |  |  | 全部测量运行与验收证据 |

## 1. 提交前后不变量

| 不变量 | 改动前（main @ 2eaba93） | 改动后（branch @ TBD） | 验证方式 |
|---|---|---|---|
| 全量 Python 测试 | 921 passed（Phase-0 记录） | TBD | `pytest -q tests`（webapi 排除） |
| 语义版本 | PROTOCOL_VERSION 20 / REWARD_VERSION 13 / EXECUTION_SPEC_VERSION 1 | 不变（本阶段仅新增测量模块版本） | `grep` 各版本常量 |
| 依赖 pin | requirements 不变 | 不变（内存测量用 stdlib，零新依赖） | `freeze_lock.py --check` |
| 遗留产物 | legacy 盖章产物不变 | 不变（本阶段不写策略/协议产物，只新增测量产物） | — |

## 2. 契约（先于实现）

### P1-02 费用矩阵（FEE_MATRIX_VERSION 1）

- 输入：资金 C、持仓数 N、年换手 T（每个持仓每年完整买卖回合数）、费用参数
  （唯一口径：`BacktestConfig` 的 `commission_rate` / `min_commission` /
  `stamp_tax_rate` / `transfer_fee_rate` / `slippage_rate`，与回测/模拟盘同源）。
- 单持仓单回合成本 = 买佣金 + 卖佣金 + 印花税 + 2×过户费 + 2×滑点，
  每笔佣金 = `max(单笔名义 × 佣金率, 最低佣金)`（**最低佣金地板按笔生效**）。
- 年化拖累 = 单回合成本 × N × T；`drag_pct = 年化/资金 × 100`。
- 可接受规则（预注册）：`drag_pct ≤ budget_pct`（默认 1.0%/年，纯成本预算线，
  与 Alpha 无关）。`capacity` = 网格中成本不超预算的**最大持仓数**；
  `recommended` = 默认换手（T=6，约两月一次调仓）下的 capacity 单元；
  `feasible_structures` = 全部满足预算的 (C, N, T) 单元。
- 产物：`data/fee_matrix.json`（版本、费用参数、网格、单元格、capacity、
  recommended、feasible_structures）。

### P1-03 七裸因子固定回测（BARE_FACTOR_BACKTEST_VERSION 1）

- 只对 `protocol.baseline_signals`（默认 REVERSAL_5 / RSQ_60 / ILLIQ_20 /
  OVERNIGHT_RET / MOMENTUM_20 / ROE / TURNOVER）做**固定回测**：无任何搜索器、
  无采样、无试错；因子列直接进回测引擎，方向按训练窗（`backtest.train_end_date`
  前）`signal_direction` 固定推断（与协议 baseline 行同一约定）。
- 每个因子一行：净收益 / Sharpe / Sortino / 最大回撤 / Calmar / 换手 / IC / ICIR
  + direction + 费用与 top_n 等配置溯源。
- 产物：`data/bare_factor_backtest.json`（版本、dataset_id、universe_policy、
  配置快照、逐因子行）。

### P1-04/05 搜索器成本测量（SEARCHER_BENCH_VERSION 1）

- 预算单位 = **唯一语义公式评价**（T2-01 语义缓存口径，与协议一致）。
- 四个搜索器（gp / tpe / random / rl）在**同一 300×400 裁剪窗口**
  （`prepare_window` 的 `window_cap=(300,400)`，fold 0 训练窗头部）、同一 nominal
  budget、同一 seed 下各自完整跑完；RL 以 `steps=4, batch=budget/4` 折算
  （budget ≥ 16 且可被 4 整除）。
- 每行记录：`unique_semantic_evals`（实际）、`wall_seconds`、
  `wall_per_1000_evals`（= wall/evals×1000）、`peak_rss_mb`（轮询采样，stdlib：
  POSIX `resource.ru_maxrss` / Windows `ctypes GetProcessMemoryInfo`）、
  `completed`、`selected_val_reward`、`n_invalid`、`n_semantic_dedups`。
- `train_search` 契约扩展：接受 `"tpe"`（TPE 与 gp/random 同走
  `SemanticBudgetEvaluator` 预算记账）。既有测试
  `test_train_search_respects_budget_and_backend` 中"tpe 被拒绝"的断言随之更新：
  旧断言编码的是"功能缺失"这一状态而非语义保证；新断言验证 tpe 同样在预算内运行
  并留下选择状态，验证强度不降低（白名单：需求变更 + 契约先行）。
- 产物：`data/searcher_bench.json`（版本、溯源、逐搜索器行）。

### P1-01 selfcheck 空转验收（无新代码）

- 协议 selfcheck（v20 拼接试验矩阵路径，既有实现与测试）：纯噪声候选，
  裁决必须 **DSR < 0.95 且 max-t p > 0.05（不显著）**。
- 产物：`data/selfcheck_result.json`。

## 3. 关键测量（运行结果，验证后回填）

TBD

## 4. 版本变更（语义变更才 bump）

| Module | Before | After |
|---|---|---|
| ashare_model.cost_matrix | — (new) | FEE_MATRIX_VERSION 1 |
| ashare_model.bare_factor_backtest | — (new) | BARE_FACTOR_BACKTEST_VERSION 1 |
| ashare_model.searcher_bench | — (new) | SEARCHER_BENCH_VERSION 1 |
| AshareTrainer.train_search | 接受 gp/random | + tpe（预算语义不变，无产物 schema 变化） |
| PROTOCOL_VERSION / REWARD_VERSION | 20 / 13 | 不变 |

## 5. 迁移 / 拒绝策略

- 本阶段全部为**新增测量产物**（`data/fee_matrix.json`、
  `data/bare_factor_backtest.json`、`data/searcher_bench.json`、
  `data/selfcheck_result.json`），不修改任何既有持久化产物 schema；
  无迁移，旧产物不受影响。
- selfcheck 追加写入既有 trial ledger（append-only，hash 链校验失败即拒绝）。
- 旧 selfcheck 产物（protocol v2，`experiments/20260816_selfcheck/`）仅为历史
  存档，不参与本阶段裁决。

## 6. 验收证据（验证后回填）

1. 10万 / 20万 / 50万资金分别适合多少持仓（费用矩阵 capacity）。
2. 至少一种成本可接受的持有/调仓结构（feasible_structures 实例）。
3. 四搜索器在相同小预算下全部完成（smoke 行 `completed=true`）。
4. 不宣称发现 Alpha：本阶段无任何显著性/超额收益结论。
