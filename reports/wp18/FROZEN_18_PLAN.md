# WP18 冻结计划（FROZEN_18_PLAN）——信号诊断与账户执行两轨的预注册真源

- 状态：**运行前冻结**（pre-registered，2026-09-27）。本文件是 WP18 唯一预注册真源；对执行者/修复器**只读**。
- 依据：`D:/minequant/AlphaGPT/reports/wp18/_recon/map.md`（2026-09-27 只读侦察，下称"侦察图"）+ 本包任务规范（用户需求逐字保留段）。本计划引用的代码事实均在冻结会话中重新亲读核验（关键行号见各节）。
- 产物纪律：历史产物（`data/r1/**` 既有文件、`reports/p45`、`reports/p46`、`reports/wp16c` 等）一律只读；本包新产物只写新目录（§9）。红证据只追加；本包不得触发方法修订（WP17 `method_revision` 守卫，episodes=[]）。
- 硬性红线（全文有效）：算法搜索预算为零（不运行 GP/任何搜索器、不拟合权重、不新增因子——slow 新组合 `combo:slow_all5` 是唯一允许的预注册新 ID，等权、只用现有五成员）；方向每折学习一次，阶段产出后冻结，账户执行与成本压力一律消费同一份冻结方向；正式评价行两域合计 ≤300；所有 2021–2025 结果标注开发验证。

---

## 0. 术语与总览

| 术语 | 定义（冻结） |
|---|---|
| 域 | `medium`（r1_medium_a 配置族）、`slow`（r1_slow_ab 配置族） |
| 候选 | 单因子或等权组合的信号 ID；本计划共 10 个（§1） |
| 规则 | 方向规则；A=`reference_return`，B=`label_ic`（§2） |
| 模式 | 账户执行模式：`dynamic`（动态换股）与 `hold`（首次建仓后持有，§3） |
| 正式评价行 | 账户评价行 + 成本压力行（会计口径写死于 §4） |
| 测量行 | 方向证据行、纯信号摘要行——逐行物化与对账，但不计入 300 正式评价行预算 |

两轨纪律（用户规范逐字）：**分两轨：信号诊断 与 账户执行——不得用受预算限制的 0x 回测亏损推导信号没有信息。**

阶段顺序与域门：medium 全部阶段（M1→M2→M3）→ 域门（§10）→ slow 全部阶段（S1→S2→S3）→ 条件性成本压力（C1/C2，各自依 §5 初筛判据）。**先跑 medium 域，检查结果完整性和运行成本，再决定是否进入 slow 域。**

---

## 1. 候选与 ID 清单（冻结）

8 个单因子 = medium 3 + slow 5；10 个候选 = medium 4 + slow 6。**不新增因子，不拟合权重。**

### 1.1 medium 域（4 候选）

| ID | 类型 | 成员 | 依据 |
|---|---|---|---|
| `MOMENTUM_20` | 单因子 | — | `configs/r1_medium_a.yaml:52` |
| `RSQ_60` | 单因子 | — | 同上 |
| `REVERSAL_60` | 单因子 | — | 同上 |
| `combo:all3` | 既有组合 ID | MOMENTUM_20, RSQ_60, REVERSAL_60 等权平均秩 | `ashare_model/r01_baseline.py:88-91`（`R01_COMBO_ARMS` 的 `combo:all3`），构造模式 `ashare_model/r02_matrix.py:335-357` |

### 1.2 slow 域（6 候选）

| ID | 类型 | 成员 | 依据 |
|---|---|---|---|
| `ROE` | 单因子 | — | `configs/r1_slow_ab.yaml:50` |
| `PE_TTM` | 单因子 | — | 同上 |
| `REVENUE_YOY` | 单因子 | — | 同上 |
| `DIVIDEND_YIELD` | 单因子 | — | 同上 |
| `MARKET_CAP` | 单因子 | — | 同上 |
| `combo:slow_all5` | **新预注册组合 ID** | ROE, PE_TTM, REVENUE_YOY, DIVIDEND_YIELD, MARKET_CAP 等权平均秩 | 本节预注册；八因子名均在词表 `ashare_model/vocab.py:52-111` |

### 1.3 `combo:slow_all5` 假设陈述（预注册，逐字冻结）

> **分散化假设**：五个慢速基本面定义（质量 ROE、估值 PE_TTM、成长 REVENUE_YOY、股息 DIVIDEND_YIELD、规模 MARKET_CAP）各自度量经济假象的不同侧面，等权秩空间平均可降低单一定义的度量噪声与定义特定偏差；等权是唯一聚合方式，无任何权重拟合，不引入新信息源。预期效果为降噪与波动下降，不预期产生超出成员的新信息。

实现约束（冻结）：
- 构造按 `r02_matrix.py:335-357` 的 `combo_signal`（等权平均秩）模式直接实现，成员为 slow 五因子与各自冻结方向；**不得走 `R01_COMBO_ARMS`**（其成员是 medium 因子；slow 历史运行组合臂因此全败 `KeyError: 'MOMENTUM_20'`，见侦察图 §1.3 与 `data/r1/r01_slow_baseline/` 报告——该历史缺陷是 WP14 precheck 的 documented case）。
- 组合不重学方向：消费成员在 M1/S1 产出的同规则冻结方向（§2.4）。

---

## 2. 方向政策 A/B 事前固定（冻结）

权威模块 `ashare_model/direction.py`（`DIRECTION_RULE_VERSION = 1`，:43）。两规则 ID 均为 precheck 认可的枚举值（`ashare_model/experiment_integrity/precheck.py:443-471`，权威={reference_return, label_ic}）。

### 2.1 Rule A（id `reference_return`）

- 正式命名（既有，逐字）：**"参考配置下的验证 reward 择向（Rule A）"**（`direction.py:51` `REFERENCE_RETURN_RULE_NAME`）。
- 比较键：`(val_reward, val_icir)` 字典序（`_rule_a_key`，:159-168）；决策委托 `CandidateScorer.score_many`（:392-418）。
- 取证窗口：训练窗尾部的验证子窗（`train_windows.py:65-104` `validation_windows`：validation_fraction 尾巴切成 validation_splits 个不相交窗）。
- 参考配置（冻结）：medium 域 = **参考格 `n20_m5000`**（`r02_matrix.py:96` `R02_REFERENCE_CELL`），其耦合参数（top_n=20/buy_rank=20/sell_rank=30/single_weight_cap=0.05/min_trade=5000）与 `configs/r1_medium_a.yaml:59-65` backtest 块逐字一致；slow 域 = `configs/r1_slow_ab.yaml:54-69` backtest 块逐字（同为 top_n 20/sell_rank 30/cap 0.05/min_trade 5000 的同构耦合形状）。`portfolio_config_provenance` 记录资金/最低订单/格可达决策（`direction.py:353`）。
- 证据字段（既有 WP16-C 双向证据链）：单批评分批交织 ±signal 双侧原值 `val_reward± / val_icir± / train_reward± / train_icir± / objectives± / reward_clip / tie_break_reason / canonical_orientation_locus`（`direction.py:249-355`；产出面 `candidates.py` `score_many(..., oriented_evidence_out=[])` :390、:419、:664-667）。
- 兜底：平局走镜像稳定典范定向（:90-106）。

### 2.2 Rule B（id `label_ic`）

- **正式命名（本计划铸造，预注册）**：**"训练窗验证段之前切片对固定因果标签的截面秩 IC 符号择向（Rule B）"**。
- 决策：训练窗内信号 vs 固定因果标签的逐日截面秩 IC 均值的符号（`label_ic_direction`，`direction.py:109-156`）；接口无任何配置参数（:109-118 签名）。
- 取证窗口（依侦察图钉死的既有唯一形态）：**`(train_signal_start, val_start)`——验证段之前的训练切片**（`D:/minequant/reports/p46/run_measurement.py:103` `train_range = (int(contract.train_signal_start), int(val_start))`；`eval_search.py:147-150` 同一约定）。
- 证据字段：`{"ic_mean", "n_dates"}`（`direction.py:155`）。**落盘缺口（实现要求）**：目前无任何 r0x 报告行携带 Rule B provenance（侦察图 §6.4）；WP18 必须把 `label_ic_direction(...).to_json()` 写进每个方向行与试错账本。
- 兜底：IC 均值 0/非有限走同一典范定向（:141-146）。

### 2.3 B-fulltrain 裁决（预注册决定：**不保留**）

全仓 grep `fulltrain|full_train|全训练窗` 零命中（侦察图 §2；代码+docs）——B 的全训练窗（含验证尾）变体不存在于任何代码或文档，保留它即铸造全新政策身份，且正式评价行预算恰在 300 上限、无余量。**冻结决定：WP18 不包含 B-fulltrain。** 若未来包需要，须以独立政策身份 `B-fulltrain` 另行预注册，不得与验证窗 B 混用。

### 2.4 主比较取证日期纪律（冻结）

主比较（A vs B）在**同一折的训练窗口内部取证**：A 在训练窗尾部验证子窗，B 在验证段之前的训练切片；两者取证日期均不越过该折 `train_end`，不接触测试窗。方向**每折每规则学一次**（在各自域参考配置上），M1/S1 产出后冻结落盘（`direction_frozen.json`）；账户执行（M3/S3）与成本压力（C1/C2）一律消费同一份冻结方向，**不得重学**；跨模式（dynamic/hold）、跨成本档（1x/0x/2x）复用（`r02_matrix.py:263-295` 的 direction_reuse 惯例，reporting_notes :995-1001）。

### 2.5 参考格选择登记（试错记录义务）

任务简报曾提及"n10_m1000 参考格惯例"，与代码不符：`n10_m1000` 只是格表第一格（`r02_matrix.py:85-92`）；代码权威参考格 = `n20_m5000`（:96），且 r1_medium_a.yaml backtest 块与 n20_m5000 逐字一致（:59-65）；p46/wp16c 测量脚本用第三个格 n10_m2500（`D:/minequant/reports/p46/run_measurement.py:61-64`）。**冻结决定：WP18 一律用 `R02_REFERENCE_CELL = "n20_m5000"`（medium）与 r1_slow_ab backtest 块逐字（slow）。** 该选择（含对简报 n10_m1000 提法的否决与理由）按 §8 登记试错。

---

## 3. 执行语义（冻结）

### 3.1 引擎与行路径

- 唯一行引擎：`ashare_model/eval_metrics.py:143` `evaluate_signal(signal, loader, fold, bt_cfg)`（单因子/组合/现金控制全走它）。**禁止使用 `bare_factor_backtest.py` 路径做账户评价**（侦察图 §5）。
- 行级观测：`evaluate_signal` 每行自带引擎指标 + 双 IC（universe-masked 与 tradable，:194-218、:352-357）+ funnel（:295-336）+ lot gap 字段（:82）+ `daily_returns/daily_dates`（拼接与年化分解原料）。

### 3.2 两模式：dynamic 与 hold

**dynamic（动态）**：既有构造器语义，逐到期会话按带向信号重选（survivors/entrants 再选择 + 强制风险退出）。

**hold（首次建仓后持有）**——本包新实现面，语义逐字段冻结如下：

| 字段 | hold 模式语义（冻结） |
|---|---|
| 会话网格 | 与 dynamic 完全相同的 `rebalance_mask`（同一会话、同一日期、同一折） |
| 信号输入 | 与 dynamic 相同的带向信号（消费同一份冻结方向） |
| 首建仓会话 | 账户尚未建仓的第一个到期会话：走与 dynamic 完全相同的正常构造（排名 top_n 选择） |
| 首建仓后的到期会话 | **关闭主动换股**：不执行 survivors/entrants 再选择，target=prev；仅当存在 forced_exit 时清零失资格持仓（见下行） |
| 强制风险退出定义 | `forced_exit = held & ~eligible & ~sell_blocked`（`ashare_portfolio/constructor.py:357` 逐字沿用） |
| 到期会话上的强制退出 | 执行卖出腿 + 费用入账；卖出所得留在现金，**不买入替代**（hold 模式首建仓后永不主动买入） |
| 非到期会话上的强制退出 | **仅执行强制退出腿**：卖出失资格持仓、费用入账、所得留现金；无买单。语义依据：既有 `not_due` 早退（`constructor.py:333-335`）发生在 forced_exit 计算（:357）之前，非到期会话上强制退出从未执行——hold 模式按语义补上，用户规范逐字"保留必要的强制风险退出，仅关闭主动换股" |
| 非到期且无强制退出的会话 | prev 原样结转，零成本（既有引擎 else 分支，`backtest.py:387-394`，不变） |
| 成本口径 | 与 dynamic 同一成本阶梯（1x 主口径）；费用仅记在实际执行的（卖出）腿上 |
| 观测面 | 新增 **additive** 计数键（如 `funnel_hold_forced_exit_sessions`）；既有键名一律不改 |

实现约束（冻结；p56 测试先行）：
1. `constructor.construct`：把 `forced_exit` 计算上提到 not_due 分支之前；新增 opt-in 关键字（如 `not_due_forced_exit: bool = False`），True 且非到期且 `forced_exit.any()` 时返回 `target=prev.copy()`、`target[forced_exit]=0`、新 reason（如 `"not_due_forced_exit"`）；**默认 False 路径字节不变**（红线：既有 p35+ 数字不可动）。
2. `backtest.py:291` 的执行门放宽为 `if rebalance_due or output.reason == "not_due_forced_exit":`（卖腿+费用核算执行，构造上买单为空）。
3. hold 到期会话（首建仓后）由驱动/构造层跳过再选择返回 prev+forced_exit 清零（新 reason，如 `"hold_forced_exit"`），共享同一会话网格与费用口径——dynamic/hold 只在"主动换股腿"上不同。
4. 全部改动默认关闭、additive、p56 测试先行；对既有基线逐字节保持。

### 3.3 成本档

- **1x 为主口径**（配置原样，`r01_baseline.py:113-138` `cost_variant`）；0x/2x 仅条件性压力（§5）。
- 逐行切换唯一权威 `cost_variant`（`r01_baseline.py:113-138`）：0x=五成本参数全零；1x=config 原样；2x=commission/stamp/transfer/slippage ×2 且 **min_commission 不变**；未知档 ValueError。注意 `R02_COST_LEVELS=("0x","1x")`（`r02_matrix.py:99`）——2x 必须直接调 `cost_variant`，不得指望 r02 常量。

### 3.4 折与种子

- 折：五折，测试年 2021–2025（`configs/r1_medium_a.yaml:46-51`；slow 同构），经 `eval_folds.resolve_folds` 解析为 `FoldTimeContract`。
- 种子：无随机臂——种子按协议记录但不驱动任何抽样（`r01_baseline.py:1277-1279` "no stochastic arm runs in round 1; seeds are recorded" 惯例）；引擎确定性。

### 3.5 控制臂（等权 + 现金）

每域每折各测一次，规则无关、模式无关（控制臂既非 dynamic 也非 hold）：
- `benchmark:equal_weight`：无成本参考（cost_level="ref"，格无关；`r02_matrix.py:256-261` 惯例）；
- `control:cash`：全 NaN 信号，1x，参考格测一次（`r02_matrix.py:358-367` 惯例）。

### 3.6 格点

账户评价与成本压力一律只在参考格运行（§2.5），**无格点扫描**（搜索预算为零）。格点选择（仅参考格、否决扫描）按 §8 登记试错。

---

## 4. 每阶段每域预期行集——逐行物化（冻结，与 REQ 预算表精确一致）

行键唯一性：`(domain, stage, candidate, rule, mode, cost_level, fold)`；控制行键 `(domain, stage, control_arm, fold)`。所有行在物化计划（experiment_integrity materialize 的 expected_keys）上机械对账（§10）。

### 4.1 方向证据行：8 单因子 × 2 规则 × 5 折 = **80**（测量行）

- M1（medium）：3 单因子 × 2 规则 × 5 折 = **30**
- S1（slow）：5 单因子 × 2 规则 × 5 折 = **50**
- 每行含：方向决定 + 双侧证据（A：oriented_evidence 双侧原值；B：`label_ic_direction(...).to_json()` 即 ic_mean/n_dates + 窗口登记）。组合不产生方向行（消费成员方向）。

### 4.2 纯信号摘要行：10 候选 × 2 规则 × 5 折 = **100**（测量行）

- M2（medium）：4 候选 × 2 规则 × 5 折 = **40**
- S2（slow）：6 候选 × 2 规则 × 5 折 = **60**
- 每行含：有效日期（n_dates）、IC（ic_mean/icir，universe-masked 与 tradable 双口径）、分位收益（§7.1 新增 additive 助手的 Q 高低差）。

### 4.3 medium 账户评价（M3）：**90 行**（正式评价行，全部 1x、参考格）

- 4 候选 × 2 规则 × 2 模式（dynamic/hold）× 5 折 = 80
- 控制臂 2（equal_weight + cash）× 5 折 = 10
- 80 + 10 = **90**

### 4.4 slow 账户评价（S3，域门后）：**130 行**（正式评价行，全部 1x、参考格）

- 6 候选 × 2 规则 × 2 模式 × 5 折 = 120
- 控制臂 2 × 5 折 = 10
- 120 + 10 = **130**

### 4.5 条件性成本压力（C1/C2）：每域最多 **40** 行（正式评价行，0x 与 2x）

- 行集展开（上限）：1 主组合 × 2 规则 × 2 模式 × 2 档（0x, 2x）× 5 折 = **40/域**
- 仅当 §5 初筛判据通过才运行；按 (规则) 通过粒度 Subset 化（仅 A 通过 ⇒ 20 行；A/B 全通过 ⇒ 40 行；全不通过 ⇒ 0 行）。
- 主组合：medium=`combo:all3`；slow=`combo:slow_all5`。

### 4.6 正式评价行会计口径（写死）

> **正式评价行 = 账户评价行（1x）+ 成本压力行（0x/2x）。**
> 两域全部推进上限：90 + 130 + 40 + 40 = **300 ≤ 300**。
> 方向证据 80 行与纯信号摘要 100 行是诊断测量行，逐行物化、逐行对账，但**不占用** 300 正式评价行预算。
> 算法搜索预算：**零**（不运行 GP/任何搜索器；不拟合权重；不新增因子）。

---

## 5. 1x 初筛判据（预注册，逐字冻结）

判定单位 = （域, 主组合, 规则）。主组合在 **1x、参考格、拼接 5 个测试年（2021–2025）** 的账户评价行上，须同时满足以下三条（即 §6.2 机械筛选门槛，逐字）：

> （i）拼接净超额（相对等权基准，费后）**至少 3/5 个测试年为正**；
> （ii）**年度净超额中位数为正**；
> （iii）**动态相对持有达到固定经济增量：动态−持有的年化净超额差 ≥ 50bp**（同规则、同折配对；年化 = 各腿 (1+累计净超额)^(1/5)−1 后作差）。

- 通过 ⇒ 该 (域, 主组合, 规则) 补跑 0x 与 2x（§4.5 行集内）；不通过 ⇒ 不补。
- 判据性质标注（逐字）：**研发资源分配门槛，非统计显著性证明也非市场规律**。

---

## 6. 停止/继续条件表与机械筛选门槛（逐字冻结）

### 6.1 停止与继续条件表（运行前冻结，逐字）

| # | 条件 | 动作（逐字） |
|---|---|---|
| 1 | 主要臂缺失 / 程序失败 / 日期或价格口径不一致 | 停止经济解释，修测量完整性 |
| 2 | 纯信号有信息但账户执行后消失 | 只处理已定位的执行问题，不扩搜索 |
| 3 | 持有价值、动态没有增量 | 转向静态选股/持有假设，停止宣称动态选股有效 |
| 4 | A/B 对结果影响很大但都无稳健增量 | 保留负结果，停止继续试探方向规则 |
| 5 | 可执行、拼接净超额与 active IR 为正、年度表现较一致、动态优于持有、成本压力下仍有空间 | 才考虑有限 GP/Random 对照 |
| 6 | 两域在有效、可比的测量下均无增量 | 暂停这批假设，新经济假设或新信息源才值得重启 |

### 6.2 机械筛选门槛（预注册，逐字）

> **至少 3/5 年为正、年度中位数为正、动态相对持有达到固定经济增量（年化 50bp）。**
> 标注：**研发资源分配门槛，非统计显著性证明也非市场规律**。
> 另报告去掉贡献最大年份后的结果，避免只看一行平均数。

---

## 7. 报告五问与计算口径（冻结）

五问（用户规范逐字）：1 固定标签上的 IC、分位排序和有效覆盖是否显示信息？2 A/B 差异来自评分规则，还是取证窗口、参考配置变化？3 动态交易相对首次持有是否有净增量？4 超额中有多少来自实际现金暴露而非选股？5 成本变化造成的是直接费用，还是现金、整手和预算引起的后续路径变化？

通用口径（冻结）：
- **现金分解用实际成交后持仓（非目标权重）**。
- **算术累计贡献、几何累计超额、年化分开报告**（三个量各自成列，不合并为单一数字）。
- **固定成交路径的费用核算与重新执行成本压力分开**（两者回答不同问题）。
- **去最大贡献年份**：主指标另报"去掉贡献最大年份"的变体（§6.2）。

### 7.1 Q1（信号信息）

固定因果标签上的：IC（`diagnostics.rank_ic_stats`，diagnostics.py:100-143；双口径 universe-masked/tradable）；有效覆盖（`factor_coverage`，:73-97 + 逐日 N_eff）；分位收益 = **新增 additive 助手**：按日把带向信号分位分桶、对固定因果标签求桶均值、产出 Q 高低差序列与均值——消费 `reward.rank_ic_series`/`diagnostics` 同一秩权威与 `causal_target_returns` 同一标签权威，不触引擎（侦察图 §5：分位收益工具不存在，grep 亲证）。

### 7.2 Q2（A/B 差异归因）

按 (因子, 折) 逐格对比 A/B 方向决定与两侧证据（A 双侧 provenance；B ic_mean/n_dates + 窗口）。因两规则取证窗口（§2.4）与参考配置（§2.5，同格同学折）均已事前固定，任何 A/B 结果差异可归因于评分规则本身；不一致率与 p46 实测（首折 4 成员仅 RSQ_60 一致，`D:/minequant/reports/p46/direction_measurement.json`）对照报告。归因结论登记试错（§8）。

### 7.3 Q3（动态 vs 持有）

同域、同规则、同折、同成本档的配对行差；报告年化差（§5 (iii) 口径）+ 算术/几何/年化三分开 + 去最大贡献年份变体。

### 7.4 Q4（现金暴露 vs 选股）

用**实际成交后持仓**逐日分解净超额：(i) 暴露/现金效应 = 实际持仓权重相对满仓等权基准的暴露差贡献；(ii) 选股效应 = 实际持仓内部相对等权基准的名称选择差贡献。两分量分开报告。

### 7.5 Q5（成本敏感性，两种口径分开）

- **固定成交路径费用核算**：固定 1x 成交路径，仅改费率重算费用（直接费用效应）；
- **重新执行成本压力**：0x/2x 全路径重新执行（现金、整手、预算引起的后续路径变化效应）；
- 两者之差 = 路径效应，单列报告。

---

## 8. 试错登记口径（冻结）

- **方向、规则、格点、组合选择全部登记试错记录，不能只登记最后留下的候选。**
- 入口：`RunHandle.record_trial`（`ledger.py:269`；`trial()` 上下文 :353-361，异常自动记 failed）——追加式哈希链 JSONL 试错账，落 `<data_dir>/runs/<spec_id>/<run_id>/trials.jsonl`（`run_store.py:221-222`）。模块契约："every trial is recorded — no manual omission of failed trials"（ledger.py:1-16）。
- 条目形态：`algorithm="direction_rule:<id>"`（或 `"grid:"` / `"combo:"` / `"mode:"` 前缀），`payload={规则、格、组合、决策与证据}`，附 `candidate / seed / fold_train_end / fold_test_end / metrics`（`LedgerEntry`，ledger.py:74-94）。
- 必登条目（预注册清单）：参考格选择 n20_m5000（含对简报 n10_m1000 的否决理由，§2.5）；Rule B 正式命名铸造；B-fulltrain 不保留决定；combo:slow_all5 预注册与假设；hold 模式实现选择（§3.2 的 1/2/3 路径取舍）；每次方向学习（含被拒/失败/落选者与双侧证据）；1x 初筛每条判定（含未通过者）；域门判定（§10）。
- `experiment_tracking.py`（MLflow 桥）与 `elite_archive.py` 均不是登记处（侦察图 §6 三选一结论）。

---

## 9. 输出目录布局（新目录，绝对路径，冻结）

```
D:/minequant/AlphaGPT/reports/wp18/
  FROZEN_18_PLAN.md          ← 本文件（只读真源）
  _recon/map.md              ← 侦察图（既有，只读）
  medium/
    direction_frozen.json    ← M1：每折每规则每单因子方向 + 双侧证据（含 Rule B to_json）
    signal_summary.json      ← M2：40 行纯信号摘要
    account_report.json      ← M3：90 行账户评价（1x）
    run_meta.json
    cost_stress.json         ← C1：≤40 行（条件性）
    integrity/               ← precheck/reconcile/status 产物
  slow/                      ← 同构（S1/S2/S3/C2：50/60/130/≤40）
  errata/                    ← 红证据只追加
D:/minequant/AlphaGPT/data/r1/wp18_medium/   ← 运行存储（runs/<spec_id>/<run_id>/trials.jsonl）与测量副本（r02_matrix.py:1126-1140 out_root 惯例）
D:/minequant/AlphaGPT/data/r1/wp18_slow/
```

因子缓存纪律（冻结）：medium/slow 既有 `factor_tensor_cache`（`data/r1/medium_a/`、`data/r1/slow_ab/`）必须命中（identity = v2 duckdb 路径 + dataset_id + dates + ts_codes，`load_cache.py:79`；冷重建 medium 实测 ~45 分钟，侦察图 §1.5）。实现二选一并登记试错：(a) 驱动层沿用既有 data_dir 的缓存（只读）+ out_root 写新目录；(b) 只读复制既有缓存目录到新 data_dir 后以新 data_dir 运行。任一路径必须日志验证 `factor cache outcome=hit`；若 miss 冷重建发生，视为预算事件登记并计入域门运行时。

---

## 10. 运行时预算与域门规则（冻结）

### 10.1 运行时预算（依侦察图 §1.5 实测外推）

| 域 | 依据 | 预算（预期） | 上限（2×，硬） |
|---|---|---|---|
| medium（M1+M2+M3 合计 160 行） | 缓存命中：固定装载 ~600 s + 引擎 0.53 s/行×90 ≈ 50 s + 方向腿 2 规则×5 折（3 因子 ± 双向批，每折 ~12-14 s）≈ 140 s + 摘要与落盘对账 ~400 s ≈ **1200 s** | 1800 s | **3600 s** |
| slow（S1+S2+S3 合计 240 行） | 引擎 4.5 s/行×130 ≈ 585 s + 装载 ~600 s + 方向腿（5 因子）≈ 400 s + 摘要与落盘 ~600 s ≈ **2200 s** | 3600 s | **7200 s** |
| C1/C2 | 每域 ≤40 行引擎行（medium ≈ 21 s；slow ≈ 180 s） | 计入所在域上限 | 同上 |

### 10.2 域门（逐字冻结）

> **medium 完整性通过且运行时 ≤ 2×预算（即实际 ≤ 3600 s）才进 slow。**
> 完整性判据 = **experiment_integrity 对账无缺行/重复/程序失败行**。

- 运行前：本计划物化的 RunPlan 必须**过 WP14 precheck**（`scripts/wp14_integrity.py precheck --plan ...`，退出码 0；仓库事实）。precheck 权威 = `direction.py` 枚举 {reference_return, label_ic} + DIRECTION_RULE_VERSION（precheck.py:443-471）、组合成员必须是某单因子臂（`_check_direction_evidence` :415）。
- 运行后：**必须过 reconcile**（missing/duplicate/unexpected/failed + sha 溯源）。status 分类另存。
- 注册要求（实现面，additive）：WP18 须在 experiment_integrity additive 注册新 experiment_id（`types.EXPERIMENT_IDS` 封闭四元组 `("r01_slow_baseline","r02_matrix","r02_periods","r03_combos")`，types.py:161；`plan_from_json` 拒未知 id，plan.py:868-873）：改 `types.EXPERIMENT_IDS` + `plan._EXPERIMENT_SOURCES` 新条目 + 新 `_ARM_EXTRACTORS` + `VERDICT_MAPS`；臂的 `direction_rule` 字段取 `direction.py` 枚举值（precheck 强制）。预算机械核对点：`RunPlan.expected_row_count = Σ len(arm.cost_levels)×len(folds)` ≤ 300。
- 写面卫兵：除 errata 外永不写 `AlphaGPT/data/` 下的既有目录（wp14_integrity.py:58-73）；WP18 plan/out 只放 §9 新目录。
- 域门失败处理：完整性不通过 → 停止经济解释，修测量完整性（§6.1 条件 1）；运行时超 2× → 暂停进入 slow，把超支原因（缓存 miss/装载）登记试错后由运行者裁决。

---

## 11. 开发验证标注与 WP17 声明纪律（冻结）

- **所有 2021–2025 结果标注"开发验证"**（development validation）；不作为样本外/推断性结论发布。
- 任何推断性措辞必须走 WP17 `claim_record.py` 五组字段与资格审定；描述性结果照常保存（不受资格限制）。
- 方向/规则/格点/组合选择全部登记试错（§8），不只登记最终幸存者。
- 本包不得触发方法修订（WP17 `method_revision` 守卫、episodes=[] 保持）。
- 负结果同样是产出：§6.1 条件 4/6 的情形下保留负结果并停止相应方向的继续试探。

---

## 附：实现前检查清单（对执行者只读）

1. p56 测试先行：hold 模式三处改动（§3.2 实现 1/2/3）+ 默认关闭 + 既有基线逐字节不变。
2. additive 注册 WP18 experiment_id（§10.2）；materialize → precheck 退出码 0 后才允许运行。
3. Rule B provenance 落盘接线（§2.2）；分位收益 additive 助手（§7.1）；新增 additive funnel 键（§3.2）。
4. 缓存命中验证（§9）；试错账本接线 `record_trial`（§8）。
5. 运行顺序：M1→M2→M3→域门→S1→S2→S3→（C1/C2 依 §5）；运行后 reconcile + status；全部产物入 §9 新目录。
