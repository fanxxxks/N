# P44 有限换手预算的优先级分配预注册契约（工作包 8）

- 状态：**APPROVED**，用户 2026-09-20 直接指令「完成工作包 8：让有限换手
  预算产生可执行订单。P1，研究主线最高优先。范围：组合构造、订单生成及
  必要的诊断字段。保留现有实现作为版本化对照。……建议研究一个简单、
  确定性的预算分配规则：按预先固定的退出/进入优先级，集中完成少量满足
  整手、最低订单、现金和费用约束的交易。首先验证是否需要这种规则，再
  决定具体实现；不必引入大型优化系统。……停止条件：账本或预算不变量
  未通过，不做收益排名。本包的工程成功标准是可行性与正确性，收益允许
  变差。」（批准记录回写于本文件头）。
- 起草：执行模型，分支 `codex/b04-load-cache-performance`（承接 p40–p43
  同一集成线）。
- 勘察基线：p43 完成后 HEAD `1ee0e45`，工作树干净（仅既有未跟踪
  `.mimosa/`）。

## 1 问题陈述（已实证，2026-09-20 探针 `_tmp_wp8_probe.py` @ 基线树）

`PortfolioConstructor.construct` 的预算路径（constructor.py §L388–409）：
全部 discretionary 腿按 `budget / discretionary_turnover` **比例缩放**，
再按最低订单过滤。十持十换场景（资金 10 万、buy_rank=10、每腿 0.10
权重、budget=0.10）：

| min_trade | orders | turnover | suppressed | 反事实可行性 |
|---|---|---|---|---|
| None | 20 | 0.10 | 0 | — |
| 1000 | **0** | 0 | **20** | 一条完整退出腿 = 1 万元 ≥ 1000，**可行** |
| 2500 | **0** | 0 | **20** | 同上，可行 |
| 5000 | **0** | 0 | **20** | 同上，可行 |

预算被摊成 20 条 500 元腿（scale=0.05），任何一档最低订单都全数消灭。
**可行交易被机制消掉**（用户三态中的第三态），规则必要性成立。R03 中间格
~88.8% 应调仓会话被完全抑制的漏斗即此机制的批量表现。对照：budget=0.005
（500 元 < 1000）时**确实无可行交易**（第二态）；target==prev 时无腿
（第一态）。

既有结构性事实（保留，不改）：引擎侧（backtest.py）股数执行已有整手
向下、全额退出、费用后现金 fail-closed、p35 账本恒等式；缺陷仅在构造器
权重层的分配方式。

## 2 冻结语义（实现必须逐条可对照）

**A1 模式、版本与对照**
- 新增 `BacktestConfig.turnover_allocation: str`，合法值
  `"proportional"`（v1 行为，版本化对照，供 A/B 面板显式选择）与
  `"priority"`（**新默认**）。非法值 `ValueError`（合法性并入
  `validate_portfolio_config`）。
- `PORTFOLIO_CONSTRUCTOR_VERSION` 1 → **2**；artifact/诊断记录新版本。
- `execution_spec._PORTFOLIO_CONFIG_FIELDS` 与
  `portfolio_config_provenance` 增录 `turnover_allocation`——旧产物的
  provenance 因此缺失字段而 fail-closed（与构造器版本失配双保险；
  迁移=重测，不重写旧产物）。
- 未设预算（`turnover_budget=None`）或初始建仓（`initial_funding`）：
  两模式行为一致（不预算，v1 语义逐位不变）。mandatory 腿（强制退出 +
  legacy wind-down）不进入预算——v1 既有语义保留。

**A2 优先级分配规则**（`turnover_allocation="priority"` 且预算绑定）：
- **腿**：discretionary 且 `|target − prev| > 0` 的名字（delta 取
  阈值过滤与首次最低订单过滤之后的值；此时每条腿要么无 min 约束、要么
  want × capital ≥ min_trade_amount）。
- **顺序**（预先固定）：先**退出**（delta<0）后**进入**（delta>0）；
  退出按信号排名**降序**（最不需要的先退），进入按信号排名**升序**
  （最想要的先进）；并列以 `stable_keys` 升序决出。
- **逐腿分配**：`alloc = min(want, remaining_budget)`；进入腿另受现金
  头寸约束 `alloc = min(alloc, 1 − 当前目标书权重和)`（退出先执行，先
  释放现金）。
- **可行判据**：未设 min_trade，或 `alloc × capital ≥ min_trade_amount`；
  不满足 → **跳过该腿**（该名回退 prev，不消耗预算），记入
  `budget_skipped_no_viable`，继续下一腿（不稀释其他腿）。
- **无中生有禁止**：分配只收缩既有 want，不创造腿、不放大腿；全部成交
  来自既有信号差与既定优先级。
- 收尾重申既有 `_cap_increases_to_available_cash`（安全网，正常路径
  应为 no-op）。
- 允许部分腿：`alloc < want` 且 ≥ min 时按 alloc 执行（预算内完成一条
  较小的交易优于零交易；残余差进入下一会话，与既有 threshold/min 残差
  语义同构）。

**A3 诊断字段**（additive，既有字段不动）
- `turnover_budget_mode`：`"unbudgeted" | "proportional" | "priority"`；
- `budget_exits_executed` / `budget_buys_executed`：成交腿索引 tuple；
- `budget_skipped_no_viable`：因剩余预算不足一条最低可行订单而跳过的腿；
- `budget_cash_capped`：被现金头寸缩减的进入腿；
- `turnover_budget_consumed`：实际消耗（成交腿 |delta| 合计）；
- `suppressed_trade_count` 并入 budget 跳过腿（受抑制语义一致）；
- proportional 模式保留 `turnover_budget_scale`（v1 对照）。

**A4 不变量**（独立 oracle，RED 直接断言）
1. 预算：成交 discretionary 腿 |delta| 合计 ≤ `turnover_budget` + 1e-9。
2. 自融资：目标书权重和 ≤ 1 + 1e-9；引擎执行后**费用后现金 ≥ 0**
   （p35 账本恒等式 E=C+Σs·p 复核）。
3. 确定性：输入数组置换不变（同名字最终权重相同）；信号并列时
   stable_keys 决出；同输入同输出。
4. 强制退出不受预算影响（预算再小也全额退出）；卖出受限名目标钉在
   prev（零卖出订单）。
5. 排名缓冲不变：survivors（rank < sell_rank）与 buffer 语义 v1 保留。
6. 无腿 → 零订单零换手（不为活动次数硬凑）。

**A5 三态可区分**
- 第一态（无值得交易的信号变化）：`reason="no_signal"` 或 target==prev
  或阈值全滤——order_count=0 且无 budget 跳过腿；
- 第二态（约束下无可行交易）：有腿、全跳过、
  `budget_skipped_no_viable` 非空、order_count=0、原因在诊断中；
- 第三态（可行交易被缩放消掉）：v1/proportional 模式可复现（对照钉住）；
  v2 默认下由 A2 消除。

## 3 范围与非目标

范围：`ashare_portfolio/constructor.py`（分配规则 + 诊断 + 版本）、
`ashare_data/config.py`（新字段）、`ashare_portfolio/execution_spec.py`
（provenance 字段）、新测试 `tests/test_p44_priority_turnover_allocation.py`
（同 commit 注册分片）、§5 所列既有测试的规则 2 更新。引擎
（backtest.py）与 SimBroker 不改——它们消费构造器输出，股数层整手/费用/
现金语义已由 p35 冻结。

非目标：优化器目标权重下的特殊分配逻辑（A2 对任意 delta 通用）、收益
提升或收益排名（工程成功标准=可行性与正确性，收益允许变差）、大型优化
系统、R02/R03 面板重跑（后续测量包）、G 门禁阈值、funnel 口径重构
（新增诊断自然流入既有 funnel 计数）。

## 4 版本影响表

| 版本 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `PORTFOLIO_CONSTRUCTOR_VERSION` | 1 | **2** | 预算分配语义变更（A2）；v1 行为以 `"proportional"` 显式保留为版本化对照 |
| `EXECUTION_SPEC_VERSION` | 4 | 不变 | 股数层执行语义（p19/p35）零变化 |
| `portfolio_config` provenance | 16 字段 | **+`turnover_allocation`** | 分配模式进入身份；旧产物缺失字段 → fail-closed（构造器版本失配之外的第二重独立判别） |
| 旧 artifact | — | 不重写 | v1 产物经版本/provenance 双重失配拒绝；重测走新 run |

## 5 预期 RED（独立 oracle——禁止从实现反推）

`tests/test_p44_priority_turnover_allocation.py`（同 commit 注册分片）：

- **P1 反例封口**：十持十换、budget=0.10、min∈{1000,2500,5000}、
  10 万资金 → priority 模式 orders ≥ 1 且成交腿 = 最差排名退出腿全额、
  turnover ≤ 0.10；同输入 `"proportional"` 模式 orders == 0（v1 对照）。
- **P2 优先级与集中**：预算恰容一条腿 → 最差排名退出先得；预算容
  k 条 → 按序成交 k 条，其余跳过；进入腿在退出之后（退出先释放现金）。
- **P3 第二态记录**：budget=0.005（500 元 < 1000 min）→ orders=0、
  `budget_skipped_no_viable` 覆盖全部腿、`turnover_budget_mode=="priority"`。
- **P4 预算与现金不变量**：多腿随机场景成交合计 ≤ budget+1e-9；目标书
  和 ≤ 1+1e-9；引擎端到端（populated_db）一个调仓会话费用后现金 ≥ 0、
  p35 恒等式成立。
- **P5 强制退出豁免**：held 不再 eligible → 预算极小仍全额退出。
- **P6 卖出受限**：sell_blocked 名目标钉 prev，零卖出订单。
- **P7 顺序不敏感与并列稳定**：腿数组置换（signal/prev/掩码/keys 一致
  置换）→ 同名字最终权重逐项相同；并列信号 → stable_keys 升序决出。
- **P8 缓冲不变**：sell_rank 缓冲下 survivors 权益在预算路径中保持
  v1 语义（survivor 不因预算被逐出书）。
- **P9 无腿零动作**：target==prev → 零订单、零换手、无 budget 字段噪声。
- **P10 配置与 provenance**：`turnover_allocation="x"` → ValueError；
  provenance 含新字段且 canonical。
- **P11 版本 pin**：`PORTFOLIO_CONSTRUCTOR_VERSION == 2`。

既有测试更新（§10.1 规则 2，依据=本契约 §2/§4；断言强度不降）：
`test_turnover_budget_scales_existing_book_but_not_initial_funding` 与
`test_minimum_amount_is_reapplied_after_turnover_scaling` 显式声明
`turnover_allocation="proportional"`（它们钉的就是 v1 对照行为，保留原
断言不动）；`test_portfolio_constructor.py` / `test_price_semantics.py`
的版本 pin 1→2。`test_price_semantics` 的 legacy fixture 本就以旧版本
身份被拒，预期不需改动（如红，按规则 2 举证后最小修）。

## 6 验证矩阵

1. 聚焦：`python -m pytest -q tests/test_p44_priority_turnover_allocation.py`。
2. 相邻：`tests/test_portfolio_constructor.py`、`test_p3_portfolio_parity.py`、
   `test_execution.py`、`test_backtest.py`、`test_golden_parity.py`、
   `test_price_semantics.py`、`test_artifact_schemas.py`、
   `test_artifact_versions.py`、`test_promotion.py`。
3. `python -m compileall -j 0 -q …`、`python scripts/check_test_shards.py
   --check`、`git diff --check`、全量 `python -m pytest -q tests -n auto`
   （并行-串行对账 push 前补记，与 p40–p43 同口径）。

## 7 停止条件

- 账本恒等式（p35）或预算/现金不变量（A4）任一未通过 → 停止，不放宽
  断言、不做收益排名。
- 优先级路径需要改引擎股数层语义才能闭合 → 停止回契约。
- 全量回归出现无法归因于本包的新红/新 warning → 停止诊断。
- 发现"为活动硬凑"通道（无信号差却产生订单）→ 立即停止修复。
