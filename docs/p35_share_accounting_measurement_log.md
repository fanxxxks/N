# P35 收益记账语义修正测量日志（evidence-only）

契约：`docs/p35_share_accounting_draft.md`（APPROVED，用户 2026-09-17 直接指令
工作包 1「修正非调仓期间的持仓状态和收益记账。最高优先级」）。本文只记录实际
发生什么；计划在契约，二者不混写（AGENTS §11.1）。停止条件（契约 §6）：不变量
未全过前不运行 R02 收益排名——本包完成时 R02 仍锁定。

## 1 运行环境与提交链

| 项 | 值 |
|---|---|
| 分支 | `codex/b04-load-cache-performance`（未 push/未合并） |
| 提交链 | 契约 `5d57e5c` → RED `43b7795` → 安全修复 `55b0185`（用户选项 b 授权：Mimosa hook 拦截的存量 SQL 拼接/路径拼接 23 处清理，与 p35 零语义关联，验证矩阵见其 commit message）→ 实现+测试语义 `d0d99c5` → 测量日志 evidence commit（本文件） |
| 环境 | Windows（本机），Python 3.13.12，torch CPU，pytest 9.1.1 + xdist 3.8.0 |
| 数据 | 本包为纯合成/单测级语义修正，不触碰真实数据库（数据库已同步至 20260916，未重跑） |
| 版本变化 | `EXECUTION_SPEC_VERSION` 3→4（`ashare_portfolio/execution_spec.py`）；构造器/协议版本不变（契约 §3） |

实现文件（相对 base `43b7795`，落在 `d0d99c5`）：

1. `ashare_model/backtest.py`：非调仓会话权重漂移（股数守恒等价式）、调仓会话
   股数级执行（`s = w×E/p_raw` 重推导、整手 `floor`、清仓全卖、现金含费上限的
   比例缩放循环 `_P35_MAX_AFFORDABILITY_PASSES=64`、lot 从 `config.lot_size`
   读取）、对账恒等式 `E = C + Σ s×p`、missing/delisted bar 语义（§5.3）、
   构造器 prev 规范化（§5.1）。
2. `ashare_model/reward_baskets.py`：篮子净收益统一为
   `net = (1−cost)×(1+gross) − 1`（与引擎权益比率同式）；漂移公式
   `prev = target×clip(1+r,0)/clip(1+gross,0)/(1−cost)`，非物理合成区的
   `(1−cost)` 除数下限 `_P35_DRIFT_DIVISOR_FLOOR=1e-12`（§5.2）。
3. `ashare_trading/golden.py`：free/lot-free ledger 迁移到同一 v4 权益比率
   口径；verify() 引擎对照契约从逐位一致改为执行摩擦上界
   `engine_friction_tol`（默认 1e-4，§5.4）。
4. `ashare_data/config.py`：`BacktestConfig.lot_size: int = 100`；
   `ashare_data/schemas.py`：`BacktestResult` 增 `final_shares`/`final_cash`；
   `ashare_portfolio/constructor.py`：validate 增 `lot_size ≥ 1`。

## 2 RED 证据（commit `43b7795`）

`tests/test_p35_share_accounting.py` 六用例在冻结权重引擎（base）上全部失败，
提交信息逐条记录了旧实现的偏差（往返净值 10100/9801 ≠ 1、权重冻结不漂移、
订单来自冻结目标而非漂移持仓、对账不闭合、边界行为错误）。RED 只含预期失败，
同 commit 注册 shard-portfolio-trading。

## 3 GREEN 与回归（实现 commit 上的实测）

| 命令 | 结果 |
|---|---|
| `python -m pytest -q tests/test_p35_share_accounting.py` | **6 passed**（契约 §4 五组不变量 R-A..R-E + 拆股边界细分） |
| `python -m pytest -q tests/test_backtest.py` | 21 passed（含 §4.1 两个重钉测试） |
| `python -m pytest -q tests/test_reward.py` | 54 passed, 3 xfailed（xfail 收敛见 §5.5） |
| `python -m pytest -q tests/test_golden_parity.py` | 27 passed（v4 契约，见 §5.4） |
| `python -m pytest -q tests/test_portfolio_constructor.py tests/test_execution.py tests/test_evaluation.py` | 91 passed（574.89s；含 `test_multi_period_fold_ic_and_turnover_consume_only_global_due_columns` 尘量回归，见 §5.1） |
| `python -m pytest -q tests/test_p35_share_accounting.py tests/test_backtest.py tests/test_reward.py tests/test_golden_parity.py`（合批复跑） | 108 passed, 3 xfailed（39.76s） |
| `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi` | exit 0 |
| `python scripts/check_test_shards.py --check` | 108 files across 4 shards cover all 108 test files |
| `git diff --check` | 干净（仅 CRLF 提示，非 whitespace 错误） |

全量门禁见 §6。

## 4 旧语义钉住测试的重钉（AGENTS §10.1 白名单 2 举证）

两处旧断言钉住 v3 冻结权重语义；契约 §3 批准语义变更（v3→v4）后按白名单 2
同步。新期望值**独立手算推导**（脚本 `D:\minequant\reports\p35_handcheck_busted.py`，
第一性原理复演：只共用成本模型与 lot 规则，不复用引擎循环），非对答案。

### 4.1 `test_engine_exit_day_sells_position_with_cost`：turnover 2.0 → 1.9907454368666986

手算链（capital=100000，价格恒 10，费率合计买入 0.00076）：

- t=0 建仓：理想 10,000 股名义 100,000，买费 76.0 → 不可负担 → 比例缩放一轮
  floor 到 **9,900 股**（整手粒度），费用 75.24 → 现金 924.76，权益
  99,924.76，漂移权重 w0 = 99000/99924.76 = **0.9907454368666986**；
- t=2（退出日）构造器权重差口径：全腿卖出 w0 + 新 top1 全腿买入 1.0 →
  turnover = w0 + 1.0 = **1.9907454368666986**。

旧值 2.0 是权重空间无费用/无整手的假象（v3 引擎从不缩放建仓）。断言内嵌
`99000.0 / 99924.76 + 1.0` 表达式（独立 oracle），强度不降（精确值）。

### 4.2 `test_engine_survives_busted_book_end_to_end`：total_return −1.0 → −0.9908070882859237

手算链：

- t=0 同上建仓 9,900 股 @10（费 75.24）→ t=0 会话持仓崩盘（o2o −100%）→
  权益跌到现金 **924.76**（股数基账本损失天然 ≤ 持仓价值：−100% 钳制对
  单仓位清零不可达，equity 恒正）；
- t=1 信号最强的崩盘名仍在 1e-9 开盘价全仓重买（buy 未 block；整手粒度在
  1e-9 价格下可忽略）→ affordability 循环用尽现金，费用 =
  min_commission 5 + 0.00051×919.29 ≈ **5.4688** → 期末权益
  924.76 − 5.4688 = **919.2911714/100000** → total_return =
  **−0.9908070882859237**；
- annual_return：`(0.0091929…)^(252/2)` 浮点下溢到 0 → 恰为 −1.0（断言保持）。

旧值 −1.0 依赖 v3 的「费用把权益推负 → 钳制」路径；v4 该路径结构性不可达。
断言 `pytest.approx(−0.9908070882859237, rel=1e-9)`：末端 affordability
迭代三轮浮点收敛的 closed-form 精度约 1e-9，容差即实测收敛精度，非放宽。

## 5 实现过程中诊断并修复的四个缺陷（全部在实现 commit 内）

### 5.1 非调仓会话 turnover 尘量（`test_multi_period_fold…`）

症状：every_5_days 折内 8 个非调仓会话 turnover ≈ 6.8e-13 等尘量非零。
根因：构造器 `_output` 将 target round 到 12 位小数并清零 ≤1e-12 分量；
非调仓会话以同一账本作 target/prev，漂移权重的第 13 位浮点尾差经 re-round
变成 ~1e-13 的幻影 sell。修复（引擎侧，构造器语义零变化——契约 §3）：
构造器输入 prev 先投影到同一输出网格（`_p35_canonical_weights`：round-12 +
eps 清零），账本本身（shares/cash/capital/全精度 weights）不动——直接 round
账本会在 1e6 资金上放大出 ~1e-6 元对账误差（实测 R-D 曾差 1.69e-6 元，已排除
该方案）。修复后 `count_nonzero(turnover[~due]) == 0` 且 R-D 对账恢复精确。

### 5.2 篮子漂移除数负值（`test_formula_reward_clips_to_configured_band` ValueError）

症状：normal(0,1) 合成收益夹具下 `prev_weights must be finite and
non-negative`。根因：资本崩到几十元后 min_commission 5 元占比 >1 →
`cost ≥ 1` → 漂移除数 `(1−cost)` 为负。引擎侧结构性 `cost < 1`（买入被现金
上限截断）；篮子权重空间无此边界。修复：除数下限
`np.maximum(1−cost, 1e-12)`，物理区（1−cost ≫ 1e-12）逐位不变，非物理合成
区靠既有「总和>1 等比缩回」恢复有效 long-only 账本。测试移除 xfail 恢复
强制效力。

### 5.3 missing/delisted bar 的股数重推（golden delisting/missing_bar 两场景）

症状：v4 的 `shares = w×E/p` 在 NaN→0 开盘价上产生 inf/NaN（v3 权重空间无此
路径）。修复语义（与 v3 等价 + 股数守恒）：无价名字 shares 跨会话延续、
订单集合与 `p > 0` 相交（无价不得成交——与 SimBroker 的 `missing_bar` skip
一致；退市名的买入 sizing 曾因 `buy_blocked=False` 以 p=0 产生 inf 名义额
触发成本模型 ValueError）、对账按漂移权重标记市值（o2o 对缺失腿已记 0，
账本在缺口期间平坦，实测 missing-bar 会话引擎收益 0.00957 与连续参照一致，
不再出现 −0.496 的假腰斩）。

### 5.4 golden parity v4 迁移（27 测试全绿）

- **free/lot-free ledger 公式**从 `gross − fees/E` 迁移到
  `net = (1−cost)×(1+gross) − 1`（与引擎/篮子同一式，单一语义路径）；
  free vs lot-free 保持精确一致（实测 max diff 0.0）。
- **引擎对照契约**从逐位一致改为执行摩擦上界：v4 引擎整手 + 现金上限 vs
  连续参照的差 = 一手粒度 + 费用融资。实测（DEFAULT 场景，capital 1e6）：
  18 期 diff 4.86e-6..2.29e-5，上界 `engine_friction_tol=1e-4`；
  `insufficient_cash` 场景（capital 2e4，一手 ≈5% 资本）实测 worst 2.4e-4，
  该测试显式传 1e-3 并注释。首期 diff 2.29e-5 手算复核：50,000 股理想建仓
  被自身费用（760 元）逼回一手 → 权重 0.49938 vs 连续 0.5 →
  ~1e-3 名义差 × (gross+cost) ≈ 2e-5 ✓。
- 两条股数级路径（引擎 vs lot matcher）实测 worst 1.52e-5，由既有残差
  归因（rounding/fee_diff/carry）与恒等式承载。
- `test_execution_spec_is_versioned` 3→4。

### 5.5 xfail 收敛

`test_reward.py` 原 5 个 p35-follow-up xfail：2 个随 §5.1/§5.2 修复稳定
通过、标记移除（`lot_size=1` 场景的引擎/篮子 parity 恢复；formula_reward
band clip）；3 个保留（真实费用结算差异场景：blocked/underfilled/masked
matches engine——篮子权重空间结算 vs 引擎现金结算，为已登记的后续工作）。
乱码修复：`test_reward.py` 6 处 PowerShell 转码损坏的 em-dash（`鈥?`）以
python 字节级替换恢复。

### 5.6 全量首跑的 18 个 v3 钉住失败：逐文件处置（全部白名单 2 举证）

首跑 `pytest -q tests -n auto`（§6 首行）暴露 18 个相邻批次外的失败，
全部为 v3 语义钉住或本轮修复的连带，无一是 p35 不变量（§4 五组）回退。
按文件处置（新期望值均独立手算推导或机制上界，非对答案）：

| 测试文件/用例 | 根因 | 处置 |
|---|---|---|
| `test_no_signal_semantics`（engine/basket force_hold、engine underfilled） | v4 费用通胀：fees 从现金扣 → achieved = notional/(C−fees)；篮子漂移 prev = target/(1−cost) | achieved 手算精确重钉（underfilled：10000/(100000−20.2)=0.1000202 + ledger 对账 1000 股/现金 79979.8；basket：0.4/(1−0.000608)=0.4002434 钉住 + 0.1997567 入场 + turnover 恒等 0.2）；force_hold 改 target_weights ≤ cap（构造器约束的正确对象）+ achieved 机制上界 0.4×1.002 |
| `test_baseline_harness` cap gate | 同上（漂移权重天然可超 cap） | 断言对象迁移到 `engine.target_weights`（cap 语义作用于构造器 target），快照仅保留 sum ≤ 1 |
| `test_portfolio_objectives`（2 个 parity） | 引擎整手+affordability vs 篮子连续权重 | 摩擦容差重钉（net abs=5e-4；turnover abs=5e-2——整手残差传入下期构造器 prev） |
| `test_p3_portfolio_parity`（4 个） | 共享 helper 的 v3 逐位 parity | 共享构造器/公式/漂移数学不变，账本精度分叉 → 全链摩擦容差（5e-2/5e-3/5e-4）；weekly not-due target 恒等 → 漂移递推 allclose(5e-3) |
| `test_price_semantics`（版本 + raw lots） | EXECUTION 3→4；af 跳变=现金分红等价事件（raw 平、adjusted×10）→ 引擎 raw 账本重推变现 vs 连续参照结构性发散（实测 4e-4） | 版本钉 4；该场景 `engine_friction_tol=5e-3` + 公司行为归因注释（golden 契约本就将其列为 recorded residual） |
| `test_artifact_versions`、`test_golden_parity` 版本 | EXECUTION 3→4 | 钉 4 |
| `test_universe` sentinel join-day | 2e6 开盘价一手=2e8 元 > 测试资本 1e5 → v4 整手下买不起 F（lot 语义，非 mask 失败） | 夹具价格绝对值降至一手可负担（40→80，+100% 比率保持，信号排序不变） |
| `test_p31_b02_entry_canary` C7 手算 | v3 假设权重恒 0.5/0.5 | v4 手算链：首期精确（4900×10.1/2400×20.05、fees 74.1836、achieved 0.495267/0.481557——与实现精确吻合）+ equity[1] 权益比率手算 + 漂移/整手振荡带断言（0.49–0.52 / ≤0.5+grain）+ turnover[1]=0.023176 手算 + 终态 ledger 对账（E=C+Σs·p @最后 exit open） |
| `test_bare_factor_backtest`、`test_p34_r01_baseline` R8 | lastfailed 陈旧条目混入（复跑全绿；bare_factor 随版本修复通过） | 无需改动 |

实现侧新增两处（在语义 commit 内）：

1. **构造器 optimizer 防御性 clip**（`constructor.py`）：survivor 的
   min_weights 下界 = min(prev, cap)——漂移 prev 可超 cap（v3 恒 ≤cap 不
   触发，对 v3 是 no-op，构造器语义零变化）；否则 OSQP 前置校验
   `min_weights exceed single_weight_cap` 直接拒绝。
2. **canonical prev 提升**（`constructor.canonical_weights_to_output_grid`）：
   引擎与篮子共用（§5.1 尘量修复的篮子侧同源收口——篮子非调仓会话
   turnover 尘量同根因）。

## 6 全量门禁

首轮 `pytest -q tests -n auto`：18 failed, 1954 passed（§5.6 处置矩阵——
全部为 v3 钉住或本轮修复连带，无 p35 不变量回退）。处置后复跑：

| 命令 | 结果 |
|---|---|
| `python -m pytest -q tests -n auto` | **1971 passed, 1 failed, 1 skipped, 3 xfailed, 927.43s**；唯一失败 = `test_b03_campaign_executor.py::test_r9_user_abort_records_aborted_kills_worker_and_resume_reexecutes`（105s 的 abort/worker 竞争长测试；单跑通过、整文件串行 23/23 全绿、首轮全量亦通过——**并行时序 flaky，与 p35 无关**，未做任何代码/测试改动） |
| `python -m compileall -j 0 -q …` | exit 0 |
| `python scripts/check_test_shards.py --check` | 108/108 全覆盖 |
| `git diff --check` | 干净 |

warning 记账：golden delisting/missing_bar 场景的两处 RuntimeWarning 来源
（invalid value in multiply）已随 §5.3 修复消除；复跑无新增未解释 warning。
parallel-vs-serial：本包改动均为确定性单测路径，沿用既往里程碑对账基线；
b03 R9 的并行间歇已单独串行复核（如上）。

## 7 裁决与遗留

- 契约 §4 五组手算不变量全过（§3 第一行）→ 停止条件解除的技术侧前提成立；
  **R02 是否运行仍待用户指令**（用户原话锁定）。
- 旧产物处置：v3 及更早的回测/奖励数字（含 R01 三运行记录）保留原版本，
  分类权威已拒绝跨版本混接；p34 §6 记录的「冻结权重记账缺陷」至此修复，
  此前引擎数字不可用于参数选择/搜索准入的结论不变。
- 后续工作（已登记非本包）：篮子与引擎的费用结算对齐（3 个保留 xfail）；
  golden 摩擦上界如需更紧可改按 lot×price/capital 动态缩放。

## 8 提交拆分

1. 语义 commit `d0d99c5`：`ashare_model/backtest.py`、`ashare_model/reward_baskets.py`、
   `ashare_trading/golden.py`、`ashare_data/config.py`、
   `ashare_data/schemas.py`、`ashare_portfolio/constructor.py`、
   `ashare_portfolio/execution_spec.py`、`tests/test_p35_share_accounting.py`、
   `tests/test_backtest.py`、`tests/test_reward.py`、
   `tests/test_golden_parity.py`。
2. evidence commit：本文件（测量日志不进 CONTRACTS.md 索引——其登记规则
   明示计划与证据分离，`docs/CONTRACTS.md` 头部）。

临时审计脚本（未入仓，位于 `D:\minequant\reports\`，可删）：
`p35_handcheck_busted.py`（§4 手算复演）、`p35_golden_diag.py`（§5.4 diff）。
