# P44 有限换手预算的优先级分配 — 测量日志（工作包 8）

- 契约：`docs/p44_priority_turnover_allocation_draft.md`（APPROVED，用户
  2026-09-20 直接指令工作包 8，P1）。
- 被测实现：本包四提交（契约 `34b9249` → RED `d7c8fc0` → GREEN
  `79121d6` → 本日志）。全量验证跑在精确候选树 `79121d6` 上。
- 环境：Windows 10 22621 x64，Python 3.13.12（`D:\minequant\.venv`），
  CUDA 可用。基线：p43 完成后 `1ee0e45`，工作树干净。

## 1 规则必要性验证（勘察探针，基线树 vs 修复树）

十持十换场景（资金 10 万、buy_rank=sell_rank=10、每腿 0.10 权重、
budget=0.10；探针 `_tmp_wp8_probe.py`，运行后已删除，脚本逻辑保真于
P1 测试）：

| min_trade | 基线 orders | 基线 turnover | 修复后 orders | 修复后 turnover | 反事实可行性 |
|---|---|---|---|---|---|
| None | 20 | 0.10 | 1 | 0.10 | — |
| 1000 | **0** | 0 | 1 | 0.10 | 一条完整退出腿 1 万元 ≥ 1000，可行 |
| 2500 | **0** | 0 | 1 | 0.10 | 可行 |
| 5000 | **0** | 0 | 1 | 0.10 | 可行 |
| （budget=0.005, min=1000） | 0 | 0 | 0 | 0 | 500 元 < 1000，**确实无可行交易** |

修复后行为：最差排名退出腿（index 9，rank 19）以整腿 0.10 消耗全部
预算并成交，其余 19 条腿记入 `budget_skipped_no_viable`（suppressed=19）。
无预算对照（orders=20, turnover=2.0）不受影响。

## 2 RED / GREEN

- RED（`d7c8fc0`）：`tests/test_p44_priority_turnover_allocation.py`
  17 失败 + 1 过（置换不变性钉住在 v1 下同样成立，属双模式 pin）。
  证据 `reports/p44_red_evidence.txt`。
- GREEN（`79121d6`）：分配器实现 + 两处实现期缺陷被 RED 套件当场捕获并
  修复——(a) 跳过腿未回退 prev 导致 turnover 2.0 泄漏；(b) 进入腿现金
  头寸按完整计划书计算导致首腿 alloc 恒 0（腿先重置再逐腿分配）。
  RED 夹具修正三处（§10.1 规则 1，断言层/合法频率/非调仓 mode=None）。
- 聚焦：p44 18P。相邻：constructor 23P；parity/execution/price_semantics/
  objectives/optimizer 95P；golden_parity/backtest/schemas/versions 120P；
  runspec/promotion/bare_factor/domain/config 134P；r01/r02/r03/r42/
  baseline 73P；evaluation 51P。
- 守卫：compileall 通过；`check_test_shards.py --check` 119/119；
  `git diff --check` 干净。

## 3 全量回归（精确候选树 `79121d6`）

`python -m pytest -q tests -n auto`（22:25）：**2113 passed / 1 failed /
1 skipped / 3 xfailed**。总数对账：2100（p43 树）+ 18（p44）= 2118 ✓。
完整日志 `reports/p44_full_suite.log`。

唯一失败 `test_task_executor.py::test_supervisor_detects_heartbeat_timeout_
and_continues`：与 p43 两轮全量中先后出现的 b03-R4 / 同一 heartbeat
测试属同一进程/心跳墙钟家族——单测隔离复跑通过（24s），不导入任何本包
被改模块，`-n auto` 满载下时序预算被挤破的既有波动（p43 日志已登记为
后续测试基建项）。非 p44 回归。

skip=1 与 p43 轮次一致（本机 CUDA 可用，5 个 CUDA 用例执行计入 passed；
余 1 为 lock-audit 环境门）。

## 4 不变量与验收对照（契约 §5 逐条）

| 要求 | 证据 |
|---|---|
| 反例封口 | P1 三档 min-trade 全部从 0 单变为 1 条整腿成交；proportional 对照模式复现 0 单（版本化 A/B 钉住） |
| 可行→执行 / 无可行→记录原因 | P1 / P3（`budget_skipped_no_viable` 覆盖全部腿、`turnover_budget_consumed=0`） |
| 预算不超限 | P4 随机场景 `consumed ≤ budget+1e-9`；e2e 每个预算会话同验 |
| 费用后现金非负 | P4 e2e `final_cash ≥ 0`（引擎 p35 账本不变量未动，golden parity 22P 复验） |
| 强制退出/卖出受限/整手/缓冲 | P5（预算 0.001 仍全额退出）、P6（钉 prev + 预算让位次差退出）、P8（survivors 权益不变）；整手由引擎 p35 股数层不变量承载（本包未改引擎，golden parity 通过） |
| 输入顺序不敏感/并列稳定 | P7（置换逐名一致 + stable_keys 决出并列入场券） |
| 成交来自既定信号与优先级 | A2 分配只收缩 want；P9（无腿零动作） |
| 三态可区分 | 无信号变化（no_signal/target==prev）／无可行（P3 诊断）／被缩放消掉（proportional 对照复现） |

## 5 裁决与研究声明边界

- 工程成功标准达成：可行性与正确性（不变量全绿）；**未做任何收益
  排名或收益声明**——priority 模式下组合收益可能变差（预算集中于
  少量腿，分散化更少），这是本包的预期代价，留待后续测量包在
  R02/R03 面板上以 proportional/priority 双模式 A/B 量化。
- 版本与迁移：`PORTFOLIO_CONSTRUCTOR_VERSION` 1→2；
  `EXECUTION_SPEC_VERSION` 不变；provenance 增 `turnover_allocation`
  （旧产物经版本失配 + provenance 缺字段双重 fail-closed，不重写）。
- 后续（非本包）：R02/R03 重跑以双模式量化预算分配对漏斗与净超额的
  影响；task_executor/b03 心跳类测试满载时序治理（承 p43 登记）。
- 并行 vs 串行对账：未 push，按 §10.2 全量为本地并行；push 前补记
  （与 p40–p43 同一欠账口径）。
