# Phase 5 measurement log (P1-01 .. P1-05: 低成本测量与成本诊断)

> 规则：改动前先 commit 当前状态 → 分支开发 → 测试 → 验证 → 合并回 main。
> 测试计数为仓库根目录 `pytest tests`（排除 `tests/test_webapi.py`：既有环境阻塞，
> starlette 1.3.1 的 TestClient 需要不可离线安装的 `httpx2`）。
> 本阶段为**纯测量与成本诊断**：所有产出是费用矩阵、固定回测、搜索器成本报告与
> selfcheck 空转验收；**不宣称发现 Alpha**。

| Stage | Commit | Tests passed | Δ | Notes |
|---|---|---|---|---|
| Baseline (main, pre-P1) | 2eaba93 | 921 | — | Phase-0 收尾计数（`logs/pytest_phase0_final.log`），本次基线复跑 913（webapi 排除）+ 8（webapi）= 921，`logs/pytest_p1_baseline.log` |
| P1-02 fee matrix | `9a3a74f` | 932 | +11 | 资金×持仓数×换手率费用矩阵（FEE_MATRIX_VERSION 1，`tests/test_cost_matrix.py`） |
| P1-03 bare-factor fixed backtest | `af6f0e2` | 938 | +6 | 七裸因子仅固定回测（BARE_FACTOR_BACKTEST_VERSION 1，`tests/test_bare_factor_backtest.py`） |
| P1-04/05 searcher bench | `ed9465e` | 951 | +13 | 四搜索器成本测量 + 300×400 一折一种子小预算 smoke（SEARCHER_BENCH_VERSION 1，`tests/test_searcher_bench.py`；train_search 契约扩展 tpe；语义缓存共享修复） |
| fix(capped window) | `e9c2750` | 953 | +2 | train_search 裁剪窗下 tie_break_keys/adv 切片（300×400 实测暴露，`test_train_search_capped_window_slices_tie_breaks_and_adv`） |
| fix(bench failed rows) | `d813810` | 954 | +1 | 失败行记录真实墙钟/JSON 安全值（`test_benchmark_failed_row_is_recorded_not_dropped`） |
| P1-01 selfcheck run | 运行（无代码） | — | — | 真实数据 v20 selfcheck：DSR=0.000、max-t p=1.0000，passed=True |
| P1 runs + docs | `TBD` |  |  | 全部测量运行、README、phase5 日志与验收证据 |

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
  budget、同一 seed 下各自完整跑完；非 RL 搜索器 `(steps=budget, batch=1)`，
  RL 以 `(steps=4, batch=budget/4)` 折算（budget ≥ 16 且可被 4 整除）。
- 每行记录：`unique_semantic_evals`（实际）、`wall_seconds`、
  `wall_per_1000_evals`（= wall/evals×1000）、`peak_rss_mb`（20ms 轮询采样，
  stdlib：POSIX `resource.ru_maxrss` / Windows `ctypes GetProcessMemoryInfo`，
  零新依赖）、`completed`、`selected_val_reward`。墙钟含每个搜索器都支付的
  相同 `prepare_window` 成本（横向公平）；峰值 RSS 为进程级轮询最大值，
  后跑搜索器继承先前分配（保守偏高，仅做进程内相对比较）。
- 崩溃不丢失：搜索器异常记入行（`completed=false` + 错误文本），绝不抛出。
- `train_search` 契约扩展：接受 `"tpe"`（TPE 与 gp/random 同走
  `SemanticBudgetEvaluator` 预算记账）。既有测试
  `test_train_search_respects_budget_and_backend` 中"tpe 被拒绝"的断言随之更新：
  旧断言编码的是"功能缺失"这一状态而非语义保证；新断言验证 tpe 同样在预算内
  运行并留下选择状态，验证强度不降低（白名单：需求变更 + 契约先行）。
- **测量修复（T2-03 遗留）**：`SemanticBudgetEvaluator` 原本自建内部语义缓存，
  `train_search` 走 gp/tpe/random 时 `trainer.semantic_cache.budget_used` 恒为 0，
  协议 trained 行的 `unique_semantic_evals` 记录为 0。修复：evaluator 接受外部
  `cache` 参数，`train_search` 传入 trainer 的语义缓存——现在每个后端都把预算
  记在同一本账上（RL 本来如此），trained 行的真实评价数恢复记录
  （`test_train_search_bills_the_trainers_semantic_cache` 回归钉死；无 schema
  变化，PROTOCOL_VERSION 不变）。
- 产物：`data/searcher_bench.json`（版本、溯源、逐搜索器行）。

### P1-01 selfcheck 空转验收（无新代码）

- 协议 selfcheck（v20 拼接试验矩阵路径，既有实现与测试）：纯噪声候选，
  裁决必须 **DSR < 0.95 且 max-t p > 0.05（不显著）**。
- 产物：`data/selfcheck_result.json`。

## 3. 关键测量（运行结果）

数据：`dataset_id b927074a45…`（11,003,350 行 / 8 表，同 Phase-0）；机器：
Windows / Python 3.13 / torch 2.11.0+cu128 / venv `D:\minequant\.venv`。

### 3.1 P1-02 费用矩阵（`data/fee_matrix.json`，FEE_MATRIX_VERSION 1）

默认费用口径：佣金万 2.5（最低 5 元/笔）、印花税卖出 0.05%、过户费 0.001%、
滑点 0.05%。预注册成本预算 1.5%/年；换手 T = 每持仓每年完整买卖回合数。

| 资金 | 默认换手（T=6，约两月调仓）下 capacity（最大可负担持仓数） | 年化拖累 | 备注 |
|---|---|---|---|
| 10 万 | **5 只** | 1.212% | 单笔名义 2 万 → 佣金地板生效 |
| 20 万 | **15 只** | 1.362% | |
| 50 万 | **30 只** | 1.272% | |

各资金完整 capacity 表（预算 1.5%）：10 万 {T=1:50, T=2:50, T=4:20, T=6:5,
T=12:0, T=26:0}；20 万 {1:50, 2:50, 4:30, 6:15, 12:0, 26:0}；50 万
{1:50, 2:50, 4:50, 6:30, 12:0, 26:0}。

**至少一种成本可接受的持有/调仓结构**（60 个可行单元中的代表，`drag_pct ≤ 1.5`）：
- 10 万：5 只 × T=6 → 1.212%/年；10 只 × T=4 → 1.008%/年；50 只 × T=1 → 0.652%/年。
- 20 万：15 只 × T=6 → 1.356%/年；30 只 × T=4 → 1.204%/年。
- 50 万：30 只 × T=6 → 1.266%/年；50 只 × T=4 → 1.008%/年。

诊断要点：单笔名义 ≥ 2 万时佣金地板（5 元）不再主导；T≥12（月度）时任何
≥5 只的持仓组合都超 1.5% 预算——**换手是成本的第一变量**。

### 3.2 P1-03 七裸因子固定回测（`data/bare_factor_backtest.json`，v1）

窗口 2015-01-05..2026-08-21（1630 只 × 2828 日），top_n=30，`search:"none"`，
方向按训练窗（2023-12-31 前）rank-IC 符号固定推断：

| 因子 | direction | 总收益 | Sharpe | 最大回撤 | 平均换手 |
|---|---|---|---|---|---|
| REVERSAL_5 | +1 | -100.0% | -1.20 | 100% | 0.95 |
| RSQ_60 | +1 | +2.6% | -0.06 | 66% | 0.17 |
| ILLIQ_20 | +1 | -100.0% | -1.26 | 100% | 0.16 |
| OVERNIGHT_RET | +1 | -100.0% | -2.03 | 100% | 1.69 |
| MOMENTUM_20 | -1 | -100.0% | -0.13 | 100% | 0.54 |
| ROE | +1 | -29.9% | -0.18 | 73% | 0.02 |
| TURNOVER | -1 | -100.0% | -2.28 | 100% | 0.50 |

结论（仅测量，不宣称 Alpha）：固定 top_n=30 下 7 个裸因子净成本后 5 个归零、
ROE 大幅亏损、RSQ_60 仅勉强打平——**裸因子不加搜索/不加合成没有可用收益**，
与"本阶段不宣称发现 Alpha"的约束一致。

### 3.3 P1-05 小预算 smoke（`data/searcher_bench_smoke.json`，v1）

300×400 裁剪窗口（fold 0 训练窗头部）、一折一种子（seed 42）、nominal budget
128，四搜索器**全部完成**（P1-05 验收 ✓）：

| searcher | 实际 unique evals | wall (s) | per-1000 evals (s) | 峰值 RSS (MB) | completed |
|---|---|---|---|---|---|
| gp | 74（停滞规则提前收敛） | 34.6 | 467.8 | 2261 | **True** |
| tpe | 128 | 70.9 | 553.6 | 2357 | **True** |
| random | 104（语义去重） | 39.0 | 375.0 | 2553 | **True** |
| rl | 103（语义去重） | 40.3 | 391.1 | 2536 | **True** |

### 3.4 P1-04 每 1000 唯一语义评价的时间与内存（`data/searcher_bench.json`，v1）

同 3.3 窗口/种子，budget 1000，device=cuda（auto）：

| searcher | 实际 unique evals | wall (s) | per-1000 evals (s) | 峰值 RSS (MB) | completed |
|---|---|---|---|---|---|
| gp | 74（停滞规则提前收敛） | 28.1 | 379.5 | 3386 | **True** |
| tpe | 1000 | 594.7 | **594.7** | 3513 | **True** |
| random | 648（池耗尽/语义去重） | 213.3 | **329.1** | 3610 | **True** |
| rl | 746（语义去重） | 266.4 | **357.1** | 4410 | **True** |

要点：TPE 单次评价最贵（≈0.59 s/评价），random 最便宜（≈0.33 s）；GP 在
pop=40 下约 2 代后陷入已评价语义类的停滞（stall 规则正常收敛，74/1000）；
峰值 RSS 为进程级轮询最大值（后跑搜索器继承先前分配），4 个搜索器同进程
合计 ~4.4 GB。全部 four 在相同 nominal budget 下正常完成（P1-05 验收 ✓）。

### 3.5 P1-01 selfcheck 空转验收（`data/selfcheck_result.json`）

5 折纯噪声候选（seed 1234），v20 拼接试验矩阵路径：

| 裁决 | 值 | 要求 | 结果 |
|---|---|---|---|
| Deflated Sharpe（DSR） | **0.000**（n_trials=1，best sr=-1.256） | < 0.95（不显著） | ✓ |
| max-t | observed=**-43.548**，p=**1.0000** | p > 0.05（不显著） | ✓ |

`selfcheck passed=True`；trial ledger 追加记录（hash 链校验通过）。

## 4. 版本变更（语义变更才 bump）

| Module | Before | After |
|---|---|---|
| ashare_model.cost_matrix | — (new) | FEE_MATRIX_VERSION 1 |
| ashare_model.bare_factor_backtest | — (new) | BARE_FACTOR_BACKTEST_VERSION 1 |
| ashare_model.searcher_bench | — (new) | SEARCHER_BENCH_VERSION 1 |
| AshareTrainer.train_search | 接受 gp/random | + tpe（预算语义不变，无产物 schema 变化） |
| SemanticBudgetEvaluator | 内部自建语义缓存 | + `cache` 参数（train_search 共享 trainer 缓存，修复 trained 行 unique_semantic_evals=0 的记账） |
| PROTOCOL_VERSION / REWARD_VERSION | 20 / 13 | 不变 |

## 5. 迁移 / 拒绝策略

- 本阶段全部为**新增测量产物**（`data/fee_matrix.json`、
  `data/bare_factor_backtest.json`、`data/searcher_bench.json`、
  `data/selfcheck_result.json`），不修改任何既有持久化产物 schema；
  无迁移，旧产物不受影响。
- selfcheck 追加写入既有 trial ledger（append-only，hash 链校验失败即拒绝）。
- 旧 selfcheck 产物（protocol v2，`experiments/20260816_selfcheck/`）仅为历史
  存档，不参与本阶段裁决。

## 6. 验收证据

1. **10万 / 20万 / 50万资金分别适合多少持仓**：费用矩阵 `capacity`（预算
   1.5%/年、T=6 默认调仓）：**5 / 15 / 30 只**（完整表见 §3.1）。
2. **至少一种成本可接受的持有/调仓结构**：60 个可行单元，例如
   10万×5只×T=6（1.212%/年）、20万×15只×T=6（1.356%/年）、
   50万×30只×T=6（1.266%/年）——结构选择 = 持仓数 × 调仓频率（换手），
   月度调仓（T=12）在全部资金档位都不可接受。
3. **四搜索器在相同小预算下全部完成**：smoke（budget 128，300×400，一折
   一种子）gp/tpe/random/rl 全部 `completed=true`（§3.3）。
4. **不宣称发现 Alpha**：selfcheck 空转验收通过（§3.5，噪声 DSR/max-t 均
   不显著）；七裸因子固定回测净成本后无可用收益（§3.2）；本阶段全部产物为
   费用/成本/时间/内存测量，无任何显著性或超额收益结论。
