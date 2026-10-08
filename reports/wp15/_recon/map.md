# WP15 侦察地图（只读侦察，2026-09-22）

除本文件外未写任何文件。所有行号基于当前工作区状态核实。

## 1. 现有 priority / proportional 分配算法（constructor.py，精确描述）

入口：`PortfolioConstructor.construct()`（`ashare_portfolio/constructor.py:218-530`）。构造器是**纯权重空间**函数：输入无价格、无股数、无现金（见 §4）。

预算路径判定（`constructor.py:410-413`）：
```python
budget = self.config.turnover_budget
budgeted = budget is not None and not diagnostics["initial_funding"]
```
`initial_funding` 在 `constructor.py:272` 定义：`bool(prev.sum() <= _WEIGHT_EPSILON)` —— **"现在空仓"即"首次建仓"**，与本次运行是否曾用过豁免无关（W4 缺陷，见 §2）。

三分支：

- **unbudgeted**（`constructor.py:412-413`）：`turnover_budget is None` 或 initial_funding → 预算完全不约束，目标原样通过（后续仅 `_cap_increases_to_available_cash` 再封顶一次，line 519）。
- **proportional**（v1 对照，`constructor.py:414-440`）：`turnover_allocation == "proportional"` 时，discretionary（非 mandatory）腿的 L1 周转若超过预算，所有 discretionary 腿统一乘 `scale = budget / discretionary_turnover`（line 423-429，诊断 `turnover_budget_scale`），然后**重放最低金额过滤**（line 431-440）——p44 正是把这种"缩成碎片再全被最低金额清掉"判定为缺陷。
- **priority**（v2 默认，`constructor.py:441-514`）：
  - 腿集合：`leg_mask = discretionary & (|delta| > 0)`（line 455）；mandatory = forced_exit（`held & ~eligible & ~sell_blocked`，line 299）+ legacy wind-down（line 317-321）不进预算分配，**天然豁免预算**（这是现状中唯一的"强制退出独立处理"机制）。
  - 顺序：exits 按 `(-rank[i], stable_keys[i])` 排序（信号排名越差越先卖，line 456-459）；entries 按 `(rank[i], stable_keys[i])`（越想买越先买，line 460-463）。
  - 逐腿消耗：exits `alloc = min(want, remaining)`（line 477-489）；entries `alloc = min(want, remaining, headroom)`，`headroom = max(1 - working.sum(), 0)`（权重空间现金上限，line 490-506）。
  - 跳过规则：`min_trade_amount` 不满足的腿进 `budget_skipped_no_viable`，**不消耗预算**（line 479-485, 494-500）。
  - 诊断键：`turnover_budget_mode / budget_exits_executed / budget_buys_executed / budget_skipped_no_viable / budget_cash_capped / turnover_budget_consumed`（line 508-514），并入 `suppressed_trade_count`（line 525-529）。
  - **关键结构事实（WP15 W1 痛点）**：exits 整类先于 entries，预算逐腿递减 → 预算紧时卖腿耗尽预算、买腿全部 skip；不存在"买卖配组成一个可融资交易组"的概念，也没有费用/整手概念（构造器看不到价格与现金）。场景1（10% 预算换 10 只）在 priority 下每期只能完成 exit 侧一组。

版本：`PORTFOLIO_CONSTRUCTOR_VERSION = 2`（`constructor.py:29`）；p44 契约 `docs/p44_priority_turnover_allocation_draft.md`。

## 2. line 272 豁免缺陷与调用链（谁维护 prev、跨期状态放哪）

缺陷：固定目标探针下 priority 模式连续卖出至空仓后，`prev.sum() == 0` 使下一期 `initial_funding=True` → `budgeted=False` → **清仓后自动重新获得 unbudgeted 全额买入资格**（`constructor.py:272` + `411`）。豁免应由"本次账户/运行是否已用过豁免（或曾有过持仓）"决定，而非瞬时空仓。

三个生产调用方（全部把构造器实例跨期复用，同时自己在循环里维护账本状态）：

1. **回测引擎** `ashare_model/backtest.py`：
   - 构造器实例在 `__init__` 创建一次（`backtest.py:86-88`），整个 run 复用。
   - `prev = canonical_weights_to_output_grid(weights)`，`weights` 是引擎逐日漂移的持仓权重（`backtest.py:238` 初始化为零、`268` 传入、`416-434` 漂移到下一开盘）；`capital=capital`（当期权益，`backtest.py:269`）；引擎自持 `shares`/`cash`（`239-240`），构造器**看不到**它们。
   - 引擎在构造后自己做股数级执行：卖出（`backtest.py:302-324`）、整手买入+可负担性循环（`333-363`）、费用走 `self.cost_model`（`ExecutionCostModel.from_config`，`backtest.py:85`）。
2. **奖励篮子** `ashare_model/reward_baskets.py`：构造器在函数内创建一次（`reward_baskets.py:285`），跨 t 与 b 行复用；每行独立 `prev[row]`/`capital[row]`（`282-283`），`construct` 逐行调用（`307-318`）——**同一实例上并行演进 b 套账本**。
3. **纸面模拟** `ashare_trading/run_sim.py`：`self.portfolio_constructor` 创建一次（`run_sim.py:156`）；`prev_weights, equity = self._portfolio_weights(...)` 来自**持久化组合状态**（`state_path`，`config.py:469`），跨日重启可恢复（`run_sim.py:472-486`）。

跨期状态放哪（结论）：**由调用方显式传入 `construct()` 的新 kwargs（opt-in）最稳**：
- 构造器内部自记状态在 reward_baskets 的 b 套账本下语义不明（一个实例服务多账本），且 run_sim 的状态可从磁盘恢复而构造器状态不可；引擎还允许外部注入构造器实例（`backtest.py:81-88`），实例内状态会随实例共享跨 run 泄漏。
- 引擎/run_sim/reward_baskets 各自的 run 级循环里都已有持续状态（`weights`/`shares`/`cash`/portfolio state），维护一个 `has_ever_held`（或 `initial_funding_used`）布尔并传入是零歧义改动；默认 `None` 时保持 line 272 现行为（红线：默认不变）。
- 下游消费注意：`ashare_model/eval_metrics.py:119-139` 的漏斗统计读取 session 诊断的 `initial_funding`（`funnel_initial_funding_sessions` 等），W4 改语义会影响这些计数的含义（p37 测试 `tests/test_p37_r02_matrix.py:212-230` 钉住了当前语义）。

## 3. ExecutionCostModel 可复用 API（唯一费用权威，`ashare_execution.py`）

- `ExecutionCosts`（frozen dataclass，`ashare_execution.py:26-48`）：`commission/stamp_tax/transfer_fee/slippage/total`，标量或向量化。
- `ExecutionCostModel.__init__(*, commission_rate, min_commission, stamp_tax_rate, transfer_fee_rate, slippage_rate)`（`58-81`，全部非负有限校验）。
- `from_config(config: BacktestConfig) -> ExecutionCostModel`（`83-91`）。
- `trade_cost(side: "buy"|"sell", notional) -> ExecutionCosts`（`103-135`）：commission = `max(rate*amount, min_commission)` **每个非零订单独立下限**；stamp 仅卖出；transfer/slippage 双边。
- `buy_cost(notional)` / `sell_cost(notional)`（`137-141`）。
- `rebalance_cost(buy_weights, sell_weights, capital)`（`143-177`）与 `rebalance_cost_fraction(...)`（`179-194`）：末轴=独立计费订单；**同一腿多个内部子单若分开调用会重复收取最低佣金 → W3 要求同日同股同方向合并后一次送检**。
- `weights_to_notional(weights, capital)`（`93-101`）。
- **整手可负担买入辅助**：`affordable_shares(cash, price, requested=None, *, lot_size=100) -> int`（`196-225`）：二分搜索最大整手数使 `notional + buy_cost(notional).total <= cash`，费用内含。
- 正确复用范例：`ashare_model/cost_matrix.py:58-73`（`round_trip_cost` 全部委托 `buy_cost(n).total + sell_cost(n).total`，自己不推导费用）。

## 4. 构造器当前输入清单（证明确缺价格/股数/现金）

`construct()` 签名（`constructor.py:218-234`）：`signal, prev_weights, *, capital, eligible, buy_blocked, sell_blocked, stable_keys, rebalance_due=True, cov=None, industries=None, beta=None, size=None, adv=None`。
- 有：信号、昨持权重、权益名义 `capital`（仅用于 min_trade_amount 折算与 optimizer 的绝对项，line 274/404/423 等）、eligibility/block 掩码、稳定键、ADV。
- **没有**：价格（无法整手化/算费用）、股数、现金余额。构造器唯一的"现金"概念是权重空间余量：`_cap_increases_to_available_cash`（`57-69`）与 priority 的 `headroom = 1 - working.sum()`（`492`）。
- 结论：W2 的"预算/整手/最低金额/现金/费用"五项检查**不可能在现有构造器输入下完成**；离散可行性必须放在拿到价格与现金的层（引擎/run_sim）或一个新的、显式接收这些输入的纯计划函数。

## 5. 计划函数放哪 + 接线点 + 显式模式键建议

- **建议位置**：新纯模块 `ashare_portfolio/trade_groups.py`（包内目前仅 `__init__/constructor/execution_spec/optimizer/rebalance`；`__init__.py` 是懒导出表，`16-61`，可加可不加——直接 submodule import 是包内首选风格，见其 docstring）。签名应是纯函数：输入 `prev_weights/shares, prices, cash, capital, config(预算/最低金额/整手), ExecutionCostModel, signal rank 排序键`，输出计划（每腿股数/名义 + 原因码：budget/lot/min_amount/cash/fees/no_viable_counterpart）+ 诊断。禁止写盘、禁止内部状态。
- **接线点**：
  1. 引擎执行段 `backtest.py:282-363`（已有整手卖出、整手+可负担买入、费用对账）是最自然的消费方——实验模式下把"先构造目标权重再各自执行"换成/叠加"交易组计划"，构造器权重路径保持默认。
  2. `run_sim.py:495-502`（`_make_orders` 下游 broker 执行）同理。
  3. 构造器内 priority 段（`constructor.py:441-514`）只动权重，不动它（红线：默认行为不变；新枚举值分支才进）。
- **显式模式键建议**：
  - 首选：扩展现有枚举 `turnover_allocation`（`config.py:448`，默认 `"priority"`）新增值如 `"trade_groups"`——`validate_portfolio_config`（`constructor.py:102-107`）与 `execution_spec.portfolio_config_provenance`（`execution_spec.py:81-83`）已透传该字段，provenance 自动携带模式，A/B 对照与 golden 校验链零新键成本。
  - W4 标志：`construct()` 新 kwargs（如 `initial_funding_used: bool | None = None`），默认 None=现行为；由三个调用方在 run 级维护后传入。
  - `make_backtest_config` 用 `BacktestConfig.__dataclass_fields__` 自动透传（`config.py:912-915`），新增 dataclass 字段（带默认值）即自动进 YAML 装载路径，无需改 loader。
  - 参考配置补键（加法式）：`configs/r1_slow_ab.yaml` 的 backtest 段目前**未写** `turnover_allocation`（缺省即 `"priority"`，语义不变）；按红线显式补 `turnover_allocation: priority`。注意其 `turnover_budget` 也不在 yaml 里——由 `protocol.domain: slow_fundamental` 注入 `0.05`（`config.py:939-940` + `ashare_domain.py:271`）。
- 诊断：沿用现有键风格（`constructor.py:252-273` 字典初始化 + `suppressed_trade_count` 并集，`525-529`）；W2 原因码应为每腿可审计枚举，进 diagnostics 而非仅日志。

## 6. W6 配置可行性数字（已在侦察中核实来源）

- slow 参考配置：`configs/r1_slow_ab.yaml` `initial_capital: 100000.0`（line 58）、`min_trade_amount: 5000.0`（line 63）、`top_n: 20`/`buy_rank: 20`/`sell_rank: 30`（59-62）；`turnover_budget` 来自域默认 `slow_fundamental` 的 `0.05`（`ashare_domain.py:271`，经 `ashare_data/config.py:939-940` 注入）。
- 判定（忽略整手与费用）：一次普通换股需 `卖出最低 + 买入最低 = 5000 + 5000 = 10000 > 0.05 × 100000 = 5000` → **单会话无法完成一对普通买卖**。此为 W6 冻结推导的输入，报告措辞按任务要求写入（长期受限收益不代表因子信息质量）。

## 7. 既有测试清单（"默认行为不变"回归基线）与运行命令

直接调用 `construct()` 的测试（构造器语义基线）：
- `tests/test_portfolio_constructor.py`（17 tests；含 `initial_funding` 语义钉扎 line 77、`test_turnover_budget_scales_existing_book_but_not_initial_funding` line 215-235）
- `tests/test_p44_priority_turnover_allocation.py`（16 tests；P1-P11 全景：预算顺序、强制退出豁免 P5、sell-blocked P6、mode 枚举 P10、版本==2 P11；engine e2e 断言 `modes <= {None, "unbudgeted", "priority"}` line 285）
- `tests/test_no_signal_semantics.py`（12 tests）
- `tests/test_universe.py`（4 处 `construct(` 调用）

经由引擎/间接路径依赖构造器行为的测试：
- `tests/test_p3_portfolio_parity.py`（4）、`tests/test_golden_parity.py`（26）、`tests/test_price_semantics.py`（18）、`tests/test_bare_factor_backtest.py`（7）、`tests/test_evaluation.py`（51）、`tests/test_baseline_harness.py`（13）、`tests/test_reward.py`、`tests/test_run_sim.py`、`tests/test_p4_admission_diagnostics.py`、`tests/test_promotion.py`、`tests/test_train.py`、`tests/test_research_doctor.py`、`tests/test_artifact_*.py`、`tests/test_p37_r02_matrix.py`（initial_funding 漏斗计数）、`tests/test_portfolio_optimizer.py` / `test_portfolio_objectives.py`（优化器层）。

基线运行命令（本次侦察已执行，当前全绿）：
```
cd D:/minequant/AlphaGPT && python -m pytest tests/test_portfolio_constructor.py tests/test_p44_priority_turnover_allocation.py tests/test_p3_portfolio_parity.py tests/test_no_signal_semantics.py -q
→ 57 passed in 6.52s
```
全量回归（实现后必跑）：`cd D:/minequant/AlphaGPT && python -m pytest tests/ -q`。

## 8. 其他核实事实

- `reports/wp15/` 与 `FROZEN_EXPECTATIONS.md` 目前尚不存在（WP15 流程将创建；对实现者只读）。`docs/` 下已有 `p44_priority_turnover_allocation_draft.md`(+measurement log) 与 `p50_experiment_integrity_draft.md`(+log)。
- 禁改：`AlphaGPT/data/**`（红线）、`ashare_model/experiment_integrity/`（WP14 已建，勿动）。
- 测试命名：本包用 p51 前缀（`tests/test_p51_wp15_*.py`；p50 已被 WP14 占用）。
- `validate_portfolio_config`（`constructor.py:82-112`）是配置键合法性的单点闸门：新枚举值/新键的校验加在这里；`execution_spec._PORTFOLIO_CONFIG_FIELDS`（`execution_spec.py:36-55`）钉死 provenance 字段集，加新配置键若要进 provenance 需同步加该元组（否则 `validate_portfolio_config_provenance` 会把带新键的记录判为 "unknown fields"，`execution_spec.py:128-135`）——这是一个容易踩的接线细节。
