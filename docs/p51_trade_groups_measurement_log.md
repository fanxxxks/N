# P51 (WP15) 可融资地接近目标组合 — 测量日志

- 契约真源：`reports/wp15/FROZEN_EXPECTATIONS.md`（2026-09-22 冻结，先于
  实现；§0.4 冻结时点核实：无 trade_groups 模块、无 W4 kwarg、无 p51 测试）。
- 落地载体：`docs/p51_trade_groups_draft.md`（§1.1.5 裁决后载体，含 §4 契约
  裁决记录）。
- 逐字命令与退出码：`reports/p51_red_evidence.txt`（工作区根 reports/，
  p44/p45 同目录惯例；本日志为其测量记录的 docs 侧汇总，只引用不复制推导）。
- 环境：Windows 10 22621 x64，Python 3.13（miniconda），pytest 9.1.1；测试
  一律 `cd D:/minequant/AlphaGPT && python -m pytest ...`。

## 1 冻结后 RED（2026-09-22，实现存在之前）

| 命令 | 退出码 | 结果 |
|---|---|---|
| `python -m pytest tests/test_p51_wp15_planner.py -q` | 2 | collection error：`ModuleNotFoundError: No module named 'ashare_portfolio.trade_groups'` |
| `python -m pytest tests/test_p51_wp15_funding.py -q` | 1 | **11 failed, 2 passed**（场景 1/2/3 六测 + W4 两测 + 模式接线两测 + 参考配置一测失败；2 过 = 置换不变性等既有行为 pin） |

失败模式与冻结预期一致：规划器缺失（ImportError）、`construct()` 无
`initial_funding_used` kwarg（TypeError）、`turnover_allocation` 枚举不含
`trade_groups`（ValueError）、参考配置未写模式键（KeyError）。

## 2 GREEN 与基线（2026-09-23 接线前）

- GREEN（机制层 + 冻结场景）：`tests/test_p51_wp15_planner.py
  tests/test_p51_wp15_funding.py -q` → 退出码 0，**24 passed**。
- 构造器四件套基线（map §7）：`tests/test_portfolio_constructor.py
  tests/test_p44_priority_turnover_allocation.py
  tests/test_p3_portfolio_parity.py tests/test_no_signal_semantics.py -q`
  → **57 passed**（基线对齐）。
- 全量：`python -m pytest tests/ -q` → 退出码 0，**2359 passed, 6 skipped**
  （0:47:24）。

## 3 §1.1.5 参考配置补键的裁决复现（yaml adjudication）

冻结 §1.1.5 要求给 `configs/r1_slow_ab.yaml` 加法式补
`turnover_allocation: priority`。实现期实测（逐字记录于
`reports/p51_red_evidence.txt` ADJUDICATION ENTRY）：

- 加键后全量：退出码 1，**11 failed, 2348 passed, 6 skipped**——其中 10 个
  `test_p50_wp14_*` 抛 `IntegrityRefusal`（WP14 实验完整性把 yaml 的
  sha256 快照进 plan 并与冻结真实 run 报告的 `provenance.config_sha256`
  比对；代表串 report=`3D20EFDC…` vs plan=`704212e2…`）；第 11 个为
  `test_ci_sharding.py` 分片清单未注册新测试文件（注册 `scripts/
  check_test_shards.py` 后 24/24 通过）。
- 撤销补键后 WP14 四文件：退出码 0，**70 passed**。

裁决：**撤销补键、参考配置保持字节原样**——历史不可变性（P45 教训的
字节级溯源）优先；防默认漂移意图改由 `docs/p51_trade_groups_draft.md` §2
的面向未来规则承载（新实验计划/新配置必须显式写 allocation 模式键）。
该裁决已按红线 3 以"勘误附注"追加于冻结文件文末（不删改原文）。

## 4 接线阶段（2026-09-23：§1.1.4 执行层 + W4 探针 + §5.1 判据）

- RED：`python -m pytest tests/test_p51_wp15_integration.py -q` → 退出码 1，
  **9 failed, 2 passed**（九个失败全部是 `execute_trade_session` /
  `ordinary_swap_session_feasibility` 未实现的 ImportError；2 过 = W4 探针
  对已落地构造器行为的 pin——默认模式缺陷钉住 + §6.2 kwarg 修复，与
  funding 文件的 11F+2P RED 模式镜像）。
- W4 缺陷复现（冻结 §6.1，逐字探针输出见红证据）：默认 priority 模式十期
  exits-first 清仓后，第 11 期 `prev_sum=0.00` 翻 `mode=unbudgeted,
  turnover=1.00, initial_funding=True`（豁免重获）；传入
  `initial_funding_used=True` 后第 11 期保持 `mode=priority,
  turnover=0.10, buys=(19,)`。可执行对应物为
  `test_w4_probe_default_mode_*`（红线 2 钉住缺陷为不变的旧行为）与
  `test_w4_probe_run_level_flag_*`。
- GREEN：integration 文件 **11 passed**；三文件合计 **35 passed**；
  `test_ci_sharding.py` 24 passed；构造器四件套 57 passed（§2 基线对齐）；
  WP14 六文件 **106 passed**（锚点完好，yaml 字节原样）。
- 全量：`python -m pytest tests/ -q` → 退出码 0，**2370 passed, 6 skipped**
  （0:39:27；2359 接线前 + 11 新增 integration 测试）。

## 5 审计修复（2026-09-23，major：trade_groups 在真实 run 入口静默弃预算）

- 缺陷（审计发现）：引擎（`ashare_model/backtest.py:266-363`）、run_sim
  （`SimulationRunner._make_orders`）与 reward_baskets（权重空间费用记账）
  均直接执行 `construct().weights`，不消费
  `plan_trade_session`/`execute_trade_session`——在这些入口开启
  `turnover_allocation="trade_groups"`（构造器仅打标签、预算交规划器，
  §1.1.2）等于静默丢弃换手预算，实验模式成为预算旁路。
- 修复选型：审计修复令三选一中的 **(2) 引擎层 fail-closed**（完整接线
  (1)——引擎/run_sim/reward_baskets 主循环消费规划器 + W4 run 级状态 +
  state_path 持久化——仍为后续工作，边界见 draft §3）。共享守卫
  `ashare_portfolio.trade_groups.refuse_unwired_trade_groups` 接入
  `AshareBacktestEngine.__init__`、`SimulationRunner.__init__`（早于任何
  portfolio 状态创建）与 `simulate_basket_daily_returns_batch`（标量入口
  委托批入口）；其余 allocation 模式（含缺省）零影响（红线 2）。
- RED：`python -m pytest tests/test_p51_wp15_integration.py -q -k "unwired"`
  → 退出码 1，**3 failed, 11 deselected**（三个 `DID NOT RAISE`——静默
  旁路在修复前被测试当场复现）。
- GREEN：integration 文件 **14 passed**（退出码 0）；`tests/ -k wp15 -q`
  → **38 passed, 2341 deselected**；相邻面（backtest/run_sim/execution/
  p3_parity/no_signal/p44/constructor/golden_parity/reward/baseline_harness/
  p37_r02_matrix）→ **243 passed**（164.73s）；`compileall` 四改文件通过。
- 配套：draft §2 增补第 4 条（引擎类配置不得写 trade_groups 的显式警告，
  修复令的最低限度项 (3) 随 (2) 一并落地）；draft §3 改述为
  "尚未消费 + fail-closed"。红证据追加审计修复条目（只追加）。

## 6 W6 报告章节：slow 参考配置的会话可行性判定（冻结 §5）

判定公式（§5.1，忽略整手与费用的上界必要条件）：一对普通买卖腿单会话
可完成 ⟺ `2 × min_trade_amount ≤ turnover_budget × E`。

slow 参考配置代入（`configs/r1_slow_ab.yaml`：initial_capital 100000.0、
min_trade_amount 5000.0；turnover_budget 0.05 由 slow_fundamental 域经
config 注入）：

```
2 × 5000 = 10000 > 0.05 × 100000 = 5000
```

**判定：该配置下单会话无法完成一对普通买卖（不可行）。** 边界注记：单笔
最低腿 5,000 恰等于全部会话预算 5,000——每会话至多执行一支最低尺寸的
单腿，永远凑不成一对；等号 `2m = B_r` 只在忽略整手与费用的理想化下恰好
可行。数值与措辞由纯判据函数 `ordinary_swap_session_feasibility` 逐字段
复现并经 `test_w6_pure_criterion_reproduces_the_slow_reference_json` 钉住，
工件为 `reports/wp15/config_feasibility.json`。

**结论措辞（冻结 §5.3 原文逐字；任何引用本判定的报告章节必须采用）：**

> 在该配置下测出的长期受限收益不能直接代表因子的信息质量：收益被执行可行性（换手预算与最低订单金额的结构性冲突）压制，而非由信号的信息含量决定。对此配置产出的任何收益结论，必须附带本可行性判定；要评估因子信息质量，需在满足 §5.1 判据的配置下重测，或明确声明收益为执行受限下界。

## 7 不变量与验收对照（冻结 §7 逐条）

| 要求 | 证据 |
|---|---|
| 断言只引用冻结 §编号（W5） | 三测试文件 docstring 全部标注 §；场景 1/2/3 数字为冻结值（20 期表、991/99,991、983/99,983、分支诊断） |
| 先冻结 → RED → 实现 → GREEN | §1/§2/§4/§5；逐字命令+退出码在 `reports/p51_red_evidence.txt` |
| 默认行为不变（红线 2） | 构造器四件套 57P 基线对齐；W4 缺陷钉为旧行为；`initial_funding_used=None` 逐字节旧语义；全量 2370P/6S；审计修复后相邻面 243P + wp15 38P |
| 费用唯一权威（红线 3） | 一切费用经 `ExecutionCostModel`（合并名义增量，§1.5 望远镜恒等式；规划期与执行期对账测试钉住） |
| 不另建回测系统（红线 4） | 离散可行性 = 纯函数 `plan_trade_session`/`execute_trade_session`/`ordinary_swap_session_feasibility`；引擎类入口对模式 fail-closed 拒绝而非半接线 |
| W6 产出去向（§7.4） | `reports/wp15/config_feasibility.json` + 本日志 §6（采用 §5.3 冻结措辞） |

## 8 边界与后续（非本包）

- 引擎/run_sim/reward_baskets 主循环消费规划器（审计修复令的完整接线
  选项 (1)，含 W4 run 级状态机迁移 `used := used or 执行后书非零`、
  run_sim 随 state_path 持久化、reward_baskets 每 (t, b) 一标志）——
  接线落地后移除对应 fail-closed 守卫与三测的拒绝断言。
- §1.4 反例数值注记（冻结文末"待审核确认的候选勘误"）：650/350 股示例
  与整手规则不一致，按规范算法应为 600/300 股；三个冻结场景数字不受
  影响，实现遵循规范算法。待审核确认后按红线 3 处理。
- 全量（审计修复后终态）：`python -m pytest tests/ -q` → 退出码 0，
  **2373 passed, 6 skipped**（0:40:46；2370 修复前基线 + 3 个守卫测试，
  0 failed）。
