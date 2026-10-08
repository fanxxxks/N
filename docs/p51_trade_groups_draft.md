# p51 (WP15) 交易组分配契约草案与配置指引

状态：实现落地文档（契约真源为 `reports/wp15/FROZEN_EXPECTATIONS.md`，本文件
是其 §1.1.5 落地方式的裁决后载体，见文末"契约裁决记录"）。

## 1. 交付物

- `ashare_portfolio/trade_groups.py`：纯计划函数 `plan_trade_session`（冻结
  §1.1.3/§1.2/§1.4），M/D/R 三通道（§1.3）、整手/最低金额/现金/费用/预算五项
  检查（W2）、同日同股同方向合并计费（§1.5/W3）、封闭原因码（§1.7）。
- `ashare_portfolio/trade_groups.py`（接线阶段追加）：
  - `execute_trade_session`（§1.1.4 执行层契约）：按账本顺序结算计划的
    `executed_orders`，可注入执行桩 `fill(key, side, shares, notional)`；
    买入失败保留真实卖出与现金（禁止伪造回滚、不做补偿交易），失败名记入
    `execution_failed_buys`（按 stable_key 排序）；卖出桩失败 = 契约违约，
    fail-closed 抛错（规划器只发已验证价格/未阻断的卖单，引擎卖腿无失败路径）；
    费用一律按 (会话, 名, 方向) 合并名义增量送 `ExecutionCostModel`（§1.5
    望远镜恒等式，执行期与规划期逐字节对账）；输出 `SessionExecution`
    （shares/cash/filled_orders/failed_buys/session_fees + §1.8 诊断块——
    即"预算未用满原因码"到 run 级 diagnostics 的透出面）。
  - `ordinary_swap_session_feasibility`（§5.1/W6 纯判据函数）：一对普通
    买卖单会话可完成 ⟺ `2 × min_trade_amount ≤ turnover_budget × E`
    （忽略整手与费用的上界必要条件）；其输出逐字段复现
    `reports/wp15/config_feasibility.json` 的 slow 参考判定。
- `ashare_portfolio/constructor.py`：
  - `turnover_allocation` 新枚举值 `"trade_groups"`（§1.1.1）——实验模式，
    显式 opt-in；构造器在该模式下走与 unbudgeted 完全相同的代码路径，仅将
    `diagnostics["turnover_budget_mode"]` 标为 `"trade_groups"`，预算交由
    规划器（§1.1.2）。
  - `construct(..., initial_funding_used: bool | None = None)`（§6.2/W4）：
    缺省 `None` 与旧行为逐字节一致；显式传入时启用
    `initial_funding := (not b) and (prev.sum() <= eps)` 状态机语义并回显
    `diagnostics["initial_funding_used"]`。状态由调用方在 run 级维护（引擎
    与 weights/shares/cash 并列；run_sim 随组合状态持久化；reward_baskets
    每 (t, b) 一标志）——清仓后不重新获得豁免。
- 测试：`tests/test_p51_wp15_planner.py`（机制层）、
  `tests/test_p51_wp15_funding.py`（冻结场景 1/2/3 + W4 + 模式接线 + W6）、
  `tests/test_p51_wp15_integration.py`（§1.1.4 执行桩/失败语义、构造器→
  规划器→执行器端到端、W4 固定目标探针【默认模式缺陷钉住 + 显式旗标/trade_groups
  模式修复】、W6 判据对 JSON 的逐字段复现）。
- W6 产出：`reports/wp15/config_feasibility.json`（slow 参考配置判定 +
  冻结措辞；数值由纯判据函数复现，测试钉住）。
- RED/GREEN 证据：`reports/p51_red_evidence.txt`（工作区根 reports/，p44/p45
  同目录惯例；含 2026-09-23 接线阶段条目：集成测试 RED 9 failed + 2 pins、
  W4 探针缺陷复现、GREEN 与全量回归）。

## 2. 配置指引（未来生效）：新配置必须显式写出 allocation 模式键

冻结 §1.1.5 的意图是"防止未来默认值漂移"，不是追溯改写历史配置。经裁决，
落地为以下面向未来的规则：

1. **任何新的实验计划/新配置文件**（backtest 段）必须显式写
   `turnover_allocation: priority | proportional | trade_groups`，不得依赖
   dataclass 缺省值。缺省仍为 `"priority"`（`ashare_data/config.py`），语义
   不变——显式化是可审计性要求，不是行为变更。
2. **已被冻结报告锚定的历史配置不得追溯编辑**：WP14（p50）实验完整性会把
   `configs/*.yaml` 的 sha256 快照进 plan 并与真实 run 报告的
   `provenance.config_sha256` 比对；对被锚定文件的任何编辑（即使语义空操
   作）都会触发 `IntegrityRefusal`。`configs/r1_slow_ab.yaml` 属此类
   （r01_slow_baseline 等 8 个冻结工件锚定）。
3. 清单式断言：`tests/test_p51_wp15_funding.py::test_allocation_mode_...`
   钉住（a）缺省值不漂移（`BacktestConfig().turnover_allocation == "priority"`
   且 loader 缺省解析一致）、（b）本指引已文档化（本文件 §2）、（c）
   opt-in 模式键机制存在且经 validate/provenance 透传。新增未被锚定的配置
   文件时应将其加入该断言的显式清单。
4. **`trade_groups` 在引擎类入口是显式拒绝项（审计修复 2026-09-23）**：
   在回测引擎 run、run_sim 纸面模拟或 reward_baskets 评分中开启
   `turnover_allocation: trade_groups` 会被
   `refuse_unwired_trade_groups` 在初始化处以 `ValueError` 拒绝
   （见 §3）——在这些入口开启该模式等于放弃换手预算，仅库层组合
   （构造器 → `plan_trade_session` → `execute_trade_session`）有效。
   引擎类配置只能写 `priority` 或 `proportional`。

## 3. 接线现状与边界

- 已交付（2026-09-23 接线阶段）：`execute_trade_session` 落地了 §1.1.4
  执行层契约（可注入执行桩、买入失败保留真实卖出与现金、
  `execution_failed_buys` 记录、合并计费对账、§1.8 诊断透出），
  `ordinary_swap_session_feasibility` 落地了 §5.1/W6 纯判据；端到端
  （构造器 trade_groups 模式 → 规划器 → 执行器）由
  `tests/test_p51_wp15_integration.py` 以冻结 §3.2 数字钉住。
- **尚未消费 + fail-closed（审计修复 2026-09-23）**：引擎
  （`ashare_model/backtest.py` 执行段）、纸面模拟（run_sim）与
  reward_baskets 评分仍走既有权重空间执行路径——把
  `plan_trade_session`/`execute_trade_session` 接入这些主循环（含 W4 旗标
  在 run 级的初始化与每会话迁移 `used := used or (执行后书非零)`、
  run_sim 随 state_path 持久化、reward_baskets 每 (t, b) 一标志）属后续
  工作。在接线落地之前，这三个入口对 `turnover_allocation="trade_groups"`
  **fail-closed 拒绝运行**（共享守卫
  `ashare_portfolio.trade_groups.refuse_unwired_trade_groups`，分别接入
  `AshareBacktestEngine.__init__`、`SimulationRunner.__init__` 与
  `simulate_basket_daily_returns_batch`；标量入口委托批入口，同被覆盖），
  而不是静默走 unbudgeted——构造器在该模式下不执行自身预算分配
  （§1.1.2），静默执行其目标权重等于丢弃换手预算。守卫行为由
  `tests/test_p51_wp15_integration.py::test_unwired_*_fails_closed_on_
  trade_groups` 三测钉住（RED→GREEN 证据见 `reports/p51_red_evidence.txt`
  审计修复条目）；其余 allocation 模式（含缺省）不受影响。
- 参考配置 `configs/r1_slow_ab.yaml` 保持字节原样（见裁决记录）。

## 4. 契约裁决记录（2026-09-22）

- 冲突：WP15 冻结 §1.1.5/红线 2 要求给 `configs/r1_slow_ab.yaml` 加法式补
  `turnover_allocation: priority`；WP14 实验完整性测试从磁盘 yaml 重新物化
  plan 并与冻结真实 run 报告的 `config_sha256` 比对，任何编辑触发
  `IntegrityRefusal`（实测加键后 10 个 `test_p50_wp14_*` 失败；错误串
  `config anchor report='3D20EFDCB2E36E649477188D0A41C2CAE044C2D709967080883DB6B750D0087C'`
  vs `plan='704212e25e3e27d02ef5d71b17da905bb0ac143d886c9a600a397363ac54ef7a'`；
  临时撤销编辑后 70/70 全绿）。
- 裁决：**撤销补键，恢复原文件**。WP14 的锚点是字节级溯源（P45 教训的直接
  产物）；对被冻结报告锚定的历史配置做任何追溯性编辑违反"不修改旧产物"的
  更深红线。历史不可变性优先；§1.1.5 的防漂移意图改由本文件 §2 的未来生效
  规则承载（代码层 opt-in 机制与默认语义照旧保留）。
- 证据：`reports/p51_red_evidence.txt` 追加条目"WP14 锚点机制按设计工作的
  独立验证"（只追加不删除）。
