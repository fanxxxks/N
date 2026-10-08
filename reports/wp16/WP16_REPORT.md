# WP16 工程报告 —— 16-A 现金费用恒等式 / 16-B P45 可比化 / 16-C 双方向证据（三提交 p52/p53/p54）

报告日期：2026-09-23。本报告只陈述可复核事实；每项给出来源路径（代码 `path:line`、文档 §、命令与退出码）。质量门一节区分**本次报告会话实际复跑**的命令与**红证据中记录、本次未复跑**的命令。

## 0 范围、真源与提交形态

- **三提交** = 三个独立红绿循环的提交包：p52（16-A，`tests/test_p52_wp16a_cash_identity.py`，12 测试）、p53（16-B，`tests/test_p53_wp16b_p45_parity.py`，19 测试）、p54（16-C，`tests/test_p54_wp16c_direction_evidence.py` 13 测试 + `tests/test_p54_wp16c_combo_invariance.py` 展开后 6 例，两文件合计 19 例）。测试计数经 `python -m pytest --collect-only -q` 本次核实（p53=19、p54 两文件=19、p52=12）。
- **期望真源**：`reports/wp16/FROZEN_A.md`、`FROZEN_B.md`、`FROZEN_C.md`（2026-09-23 冻结于任何 WP16 代码之前，冻结时点核实见 FROZEN_A §0.4）。冻结文档对实现者只读；唯一获准的改动是 FROZEN_A 文末追加的 §8 勘误附注（§8.1/§8.2，经 2026-09-23 审核确认，原文未删改——本次报告会话读到的 FROZEN_A 已含 §8，符合"只追加"机制）。
- **测量记录真源**：红证据 `reports/p52_red_evidence.txt`、`reports/p53_red_evidence.txt`、`reports/p54_red_evidence.txt`（只追加，逐字命令+退出码）；docs 侧日志 `docs/p53_wp16b_p45_parity_measurement_log.md`、`docs/p54_wp16c_direction_evidence_measurement_log.md`。**如实记录：p52 没有 docs 侧 measurement log**（docs/ 目录无 `p52_*` 文件，本次 `find` 核实）；p52 的测量记录完全落在 `reports/p52_red_evidence.txt` 与 FROZEN_A 内。
- **git 形态**（本次 `git log`/`git status` 核实）：AlphaGPT 仓库 HEAD = `dbaf729`（p49 docs 提交），WP14/WP15/WP16 全部为工作区未提交改动（110 个修改/未跟踪路径）。因此"三提交"以红证据 + measurement log 为提交记录真源；`git diff` 无法单独切出 WP16（基线中混有 WP14/15 改动，例如 `ashare_model/backtest.py` 的唯一工作区 diff 是 WP15 的 `refuse_unwired_trade_groups` 守卫，`ashare_model/backtest.py:87-94`，非 WP16 所为——与"16-A/16-B 引擎不改"红线一致）。
- **环境**：Windows 10 22621 x64，Python 3.13（miniconda），pytest 9.1.1；测试一律 `cd D:/minequant/AlphaGPT && python -m pytest ...`。

---

## 1 三提交：冻结期望 vs 实测逐数字对照

### 1.1 16-A（p52）——冻结数字 vs 实测

冻结数字全部出自 FROZEN_A §3（Python `fractions` 精确分数演算）；实测 = p52 测试在实现后按 §4.1 通道 1 容差（相对 1e-12，净收益/下期权重/现金三分量同标准）逐值通过。**本次报告会话实际复跑：`python -m pytest tests/test_p52_wp16a_cash_identity.py ... -q`（四文件合跑）→ 50 passed in 9.32s（含 p52 的 12 例），退出码 0。**

| 冻结期望（FROZEN_A §） | 冻结值 | 实测 | 钉住的测试 |
|---|---|---|---|
| §3.1 会话 1（部分投资+买腿费） | g=0.022，R_net=0.017，E₁=101,700.000；w'=(110/339, 190/1017, 34/339)=(0.324483775811, 0.186823992134, 0.100294985251)；w'_cash=395/1017=0.388397246804；Σ=1 精确 | 实现输出与手算在 1e-12 相对容差内逐值一致（通过） | `test_p52_frozen_s31_three_session_hand_table_oracle`（纯分数 oracle）+ `test_p52_s31_partial_investment_two_sided_multisession_identity`（逐会话查 net/w'/w'_cash/台账闭合/(C1)/Σ=1，见 `tests/test_p52_wp16a_cash_identity.py:267-341`） |
| §3.1 会话 2（双侧费） | g=−0.002，R_net=−0.010，E₂=100,683.000；w'=(2/11, 208/495, 101/495)；w'_cash=32/165 | 同上（通过，含双侧腿出现断言） | 同上（§5.3 双侧费用腿断言） |
| §3.1 会话 3 + 累计 | g=0.005，R_net=0.001，E₃=100,783.683；累计 1.017×0.990×1.001=1.00783683 | 同上（通过） | 同上 |
| §3.2 费后满仓 | Σa=0.996=1−c；R_net=0.018410；w'=(206/409, 102/409, 101/409)；**w'_cash=0 精确** | 通过 | `test_p52_frozen_s32_post_fee_full_investment_oracle` + `test_p52_s32_post_fee_full_investment_cash_is_zero`（`tests/test_p52_wp16a_cash_identity.py:343`） |
| §3.3 零费 | R_net=g=0.010；乘法/可加两形式重合（c·g=0）；既有零费锚 `tests/test_reward.py:981-1025` 不改仍绿 | 通过；`python -m pytest tests/test_reward.py -q` 本次复跑（合于 69 passed）→ 57 passed 基线保持 | `test_p52_frozen_s33_zero_fee_oracle` + `test_p52_s33_zero_fee_identity_and_form_coincidence`（`:391`） |
| §3.4 反例（未预留 Σa=1） | 真实 R_net=0.016000、w'_cash=−1/254=−0.003937007874<0（不可行，须拒绝）；旧实现 net=0.015920、drifted 合计 250/249=1.004016064257 被 scale 249/250 洗回 1 | RED 期实测（红证据记录，未复跑）：旧实现 `sum(target)+cost = 0.9999999999990001+0.003999999999996001 = 1.004 > 1+1e-12` 被入账；GREEN 后按声明路线 2（构造器费用预留）使该输入不可达，三查测试通过 | `test_p52_s34_counterexample_three_checks_route_declared`（`:193`，三查=净收益/下期权重/现金，路线显式声明）+ `test_p52_frozen_s34_counterexample_arithmetic_oracle`（`:175`） |
| §4.1 无整手严格一致 | 相对 1e-12 三分量 | 通过（通道 1 全部用该容差） | p52 全部实现类测试 |
| §4.2 整手单独归因 | 通道 1 仍 1e-12；通道 2（basket↔engine）三项归因（Δg/Δc/尾部）不复用 1e-12 | 通过（lot_size=100 默认下通道 1 严格、通道 2 按 p45 界） | `test_p52_s42_channel1_strict_and_channel2_lot_attribution`（`:456`） |
| §5.9 回归基线 | test_reward 57 / p3+p46 12 / p50+p51 套件不破坏 | 本次复跑：69 passed（57+12 合跑）与 144 passed（p50 六文件+p51 三文件），退出码 0 | 见 §6 质量门 |

RED 记录（红证据，未复跑）：实现前 `python -m pytest tests/test_p52_wp16a_cash_identity.py -q` → **5 failed, 5 passed**（退出码 1）；其中 §3.1 类失败实测 `net impl=0.004618099999999847` vs 手算 `g−c=0.00462`，差 `1.900e-06` = 旧乘法形式交叉项 c·g（该测试自身构造的 a/c/r 下的值）。

### 1.2 16-B（p53）——冻结数字 vs 实测

冻结数字出自 FROZEN_B §2（手算经 fractions 复核）；实测 = p53 测试通过 + 面板重跑产物。**本次复跑：p53 文件含于四文件合跑 50 passed；面板数字本次从 `reports/wp16b/comparison.json`、`parity_report_pass2_fold_window.json` 重新解析核对。**

| 冻结期望（FROZEN_B §2） | 冻结值 | 实测 | 钉住的测试 |
|---|---|---|---|
| S1 拆股中性 | o2o(adjusted)=(0, +5%) 两路径逐位一致；负对照 o2o(raw)=(−50%, +5%) 不得出现；shares 1000→2000、名义 20,000 不变；无方向翻转；无虚假费用/换手 | 通过 | `test_p53_s1_split_neutral_both_paths_exact_and_no_flip`（`tests/test_p53_wp16b_p45_parity.py:126`） |
| S2 非调仓价格变化 | 会话1 R_net=0.005、w'=(34/67, 33/67)=(0.507462686567, 0.492537313432)；会话2 R_net=101/3350=0.030149253731、w''=(104/203, 99/203)；累计 1.005×(1+101/3350)=1.035300 精确；非调仓列 cost=0、turnover=0 精确；两路径差 1e-12 | 通过 | `test_p53_s2_non_rebalance_move_books_on_drift_ledger`（`:223`） |
| S3 费用融资 | 可负担缩放 floor(5000×100000/100100)=4995 股/名；a=(0.4995, 0.4995)；c=0.000999；g=0.014985；R_net=0.013986；E'=101,398.60；w'=(0.502462558655, 0.497536455138)；w'_cash≈9.8620691015×10⁻⁷；现金全程 ≥0 | 通过 | `test_p53_s3_fee_financing_affordable_scaling_and_cash_nonneg`（`:303`） |
| S4 折尾+1 天拒绝 | ValueError 四要素（`length mismatch`/`sim={n1}`/`engine={n2}`/候选+窗口标识）；两侧注入都测；等长对照正常完成 | 通过 | `test_p53_s4a_*`×2、`test_p53_s4b_*`、`test_p53_s4c_*`×2、`test_p53_s4_control_*`（`:374-505`） |
| §1 F3–F6 不变量 B1–B6 | B3.1 同一 (s0,s1) 传两路径；B4.1–B4.4 拒绝先于误差计算；B5.1 零不对称；B6.1–B6.3 同 cfg/零费恒 0/误差族用给定方向 | 通过（RED 期 15 项失败，见 §5） | `test_p53_b31/b33/b34/b12/b42/b52/b5/b63`（`:553-773`） |
| §3.3 面板分解 | 三数 resolved/remains/new-regression + 方向/排名/verdict 对照 | 实测：resolved 0 / remains 3 / within_both 13 / **new-regression 4**（详表见 §3.3） | `reports/wp16b/comparison.json`（本次解析） |

### 1.3 16-C（p54）——冻结期望 vs 实测

冻结期望出自 FROZEN_C §3（V1/V2/V3）与 §4.2；**本次复跑：p54 两文件含于四文件合跑 50 passed（19 例）；wp16c 面板数字本次从 `reports/wp16c/combo_pct_measurement.json` 重新解析核对。**

| 冻结期望（FROZEN_C §） | 期望内容 | 实测 | 钉住的测试 |
|---|---|---|---|
| §3 V1 | 两方向分数不同例：`val_reward_plus ≠ val_reward_minus`；direction=较大 key 侧；`tie_break_reason="validation_reward"`；证据数值=批量数组元素（独立 reward_function 对账）；恰一次评分批量；−10.0 clip 地板携 `reward_clip` 披露 | 通过 | `test_v1_positive_slope_plus_side_wins_with_full_field_pattern`、`test_v1_negative_slope_minus_side_wins`、`test_v1_clip_floor_is_disclosed_not_bare`、`test_v1_exactly_one_scoring_batch_per_decision`（`tests/test_p54_wp16c_direction_evidence.py:181-304`） |
| §3 V2 | 取反后两侧证据交换、`direction(−s)==−direction(s)`、施加的经济暴露不变；canonical 分支记录 locus；全零→`all_zero_nonfinite` | 通过 | `test_v2_mirror_exchanges_both_sides_and_keeps_applied_exposure`、`test_v2_mirror_on_the_canonical_branch`、`test_v2_all_zero_canonical_locus`（`:305-394`） |
| §3 V3 | 毒化测试期（×−10⁹ 与 NaN 变体）训练侧全字段不变；既有 N4 不改仍绿 | 通过（p46 套件含于本次 69 passed） | `test_v3_poisoned_test_window_leaves_evidence_unchanged`、`test_v3_poisoning_on_canonical_and_zero_windows`（`:395-428`） |
| §2 tie-break 分类器 | 三分支判定与 `_choose_direction` 逐点一致（6⁴ 网格） | 通过 | `test_rule_a_tie_break_classifier_branches`、`test_rule_a_classifier_agrees_with_choose_direction`（`:429-487`） |
| §1/§5 边缘 | 被拒候选数字留 None 不造数、direction 保持；legacy 四键保留；baseline 行一批出证 | 通过 | `test_rejected_near_constant_signal_keeps_direction_and_jsonable_evidence`、`test_baseline_provenance_carries_both_sides_from_one_batch`（`:488-512`） |
| §4.2.2 恒等式 | 对列内严格单调 T（percentile+仿射），`combo(T(s)) == combo(s)` 容差 0 | 通过 | `tests/test_p54_wp16c_combo_invariance.py` 4 个测试（单成员×参数化/多成员/单调性钉） |
| §4.2.1 面板 | 恒等以**结构性**呈现（非经验发现），含冻结原因串 | 实测：5 折逐列 max_abs_diff==0.0；engine raw==pct 逐折相同（如 2021 折 −0.072613，51/51 单） | `test_wp16c_measurement_output_presents_structural_identity`（gated，产物在即实跑） |

RED 记录（红证据，未复跑）：实现前两文件 → **13 failed, 5 passed, 1 skipped**（退出码 1）；失败三类：`reference_return_direction() got an unexpected keyword argument 'dates'`（10）、`cannot import name 'classify_rule_a_tie_break'`（2）、`baseline_candidates must run exactly ONE scoring batch, got [1, 1]`（1）。GREEN 前 5 通过 = §4.2.2 恒等式钉（绿到港）。

---

## 2 16-A：现金费用恒等式（reward_baskets）

### 2.1 反例独立失败测试（三查）

用户上一答反例（未预留满仓 Σa=1、c=0.004）落成独立测试 `test_p52_s34_counterexample_three_checks_route_declared`（`tests/test_p52_wp16a_cash_identity.py:193`），同输入下三查：①净收益不得是旧乘法值 0.015920，亦不得输出 0.016000 后继续（状态本身不可行）；②下一期权重不得是旧缩放值（drifted×249/250）；③现金 `w'_cash=−0.003937…<0` 必须被识别——实现的处置路线（FROZEN_A §3.4 二选一）须在测试中显式声明。实测声明路线为**路线 2（构造器端费用预留，使该输入不可达）**（红证据 GREEN 节）。FROZEN_A §3.4 的数值警告在测试设计中生效：合规预留版净收益 0.015920 与旧乘法值代数恒等（§2.1 恒等式 (1−c)(1+g₀)−1=(1−c)g₀−c），故三查必须同时含权重分母与现金分量，不得以净收益单独判定。

### 2.2 恒等式落地（当前代码逐点，本次会话读码核实）

`ashare_model/reward_baskets.py`（工作区现状）：

- **§1(2) 可加净收益**：`ashare_model/reward_baskets.py:581-582` `gross = (target*rets).sum(axis=1); net = gross - cost`，注释引 FROZEN §1(2)/§2.2 并说明旧乘法形式的 −c·g 交叉项已被取代。
- **§1(3) 漂移分母**：`reward_baskets.py:632-638` 漂移除以 `max(1.0 + gross - cost, _P35_DRIFT_DIVISOR_FLOOR)`；除数下限仅护非物理合成坍缩（§6），>1 的 scale-back（`:643-645`）同仅限合成极端——(C1) 已在上游执行，二者不能再掩盖费用融资（注释原文引 §3.4/§6）。
- **§1(4) 现金显式化**：`reward_baskets.py:655` `cash_full[:, output_col] = 1.0 - prev.sum(axis=1)`，经 `BatchBasketSimulation.daily_cash_weights`（`:94`、`:669`）输出，与 gridded prev 账本一致。
- **构造器费用预留（路线 2 买侧）**：`ashare_portfolio/constructor.py:57-88` `_cap_increases_to_available_cash` 新增 `fee_reserve`（默认 0.0，引擎/run_sim 等旧调用方字节不变），可融现金 `1 − Σbase − fee_reserve`；`construct` 侧参数校验（`constructor.py:249-273`）与传递（`:595`）。
- **会话端预留不动点**：`reward_baskets.py:406-461` 以 reserve 的牛顿式迭代逼近 Σa+c=1 定点（佣金地板使费用成阶梯函数，保留最紧可行构造；权重空间版引擎可负担循环）；`_P52_C1_TOLERANCE` 容差对齐 round-12 投影。
- **(C1) fail-closed 兜底（不设买入门槛）**：`reward_baskets.py:546-565` 物理账本上剩余 Σa+c>1+容差即 `raise ValueError`（消息含 `sum(target)+cost`、FROZEN 引用），不产出任何 daily 值。
- **审核修复（卖出腿，§5 详述）**：`reward_baskets.py:471-545` 尘额卖腿的地板佣金使无买入会话也可能 Σa+c>1——处置为递延地板支配的卖腿（最小名义优先，仅在确实缩小缺口时保留）直至会话自融资费用，诊断键 `sell_fee_funding_dropped`；对应测试 `test_p52_s1c1_buyless_floor_dominated_dust_exit_funds_its_fees`（`:576`）与无地板对照 `test_p52_s1c1_dust_exit_is_feasible_without_the_commission_floor`（`:619`）。
- **p51 守卫不动**：`refuse_unwired_trade_groups` 调用保留（`reward_baskets.py:234`，红证据称现位于 line 233/234 一带），`test_p52_s59_p51_guard_still_refuses_unwired_trade_groups`（`:527`）钉住。

### 2.3 乘法→可加的语义变化

- 差异恒等式（FROZEN_A §2.1）：`(1−c)(1+g)−1 − (g−c) = −c·g`。旧乘法把费用记成"从整个账本（含股票腿）比例融资、费用切片参与当期收益"；可加口径费用只从零收益现金支付（用户口径）。§3.1 会话 1 输入下差 0.000110 = c·g。
- **引擎事实（FROZEN_A §2.3，勘误后以 §8.2 划界）**：引擎费用从现金扣、股票名义不因费用缩水，其会话净收益代数上精确等于 g−c（`backtest.py:373/380-386/403-404/431-436` 的逐步推导见 §2.3）；§8.2 勘误澄清"现金 ≥ 0 恒成立"只覆盖买入腿——卖出腿在最低佣金地板支配的尘额卖出下引擎会精确记账负现金，basket 侧不得镜像（(C1) 禁止），故走递延路线。**16-A 因此是 basket 向引擎真实记账收敛**：修后 basket↔engine 的 c·g 量级差额从对账中消失（面板尺度 g≈0.01108、c≈1e-3 → ≈1.1e-5）。
- FROZEN_A §8.1（审核后追加的勘误）：§5.9/§7"既有断言数值与容差不改"与 §1(C1)/§3.4 在 7 处满仓部署 pin 上数学互斥（未预留换手 1.0/2.0 只能在现金=−c<0 的不可行状态下满足）。处置 = 7 pin 按表重锚为 (C1) 等式形式（`tests/test_reward.py:332/978`、`tests/test_execution.py:100`、`tests/test_no_signal_semantics.py:260/321/324`、`tests/test_p51_wp15_integration.py:759`，旧→新值逐条在 FROZEN_A §8.1 表与 `reports/p52_red_evidence.txt` "Task item 3"两节）；新 pin 均为旧实现下会失败的形式；**p3 容差与 `tests/test_p3_portfolio_parity.py` 断言零改动**。

### 2.4 对 reward 链的影响面与实测证据

影响面（冻结时点核实，`reports/wp16/_recon/map.md` §1.4）：`simulate_basket_daily_returns_batch` 是唯一语义径——`ashare_model/reward.py:311-315/462/567-643`（训练 reward 与 val_reward 择向键）、`candidates.py`、`scoring_parity.py`（p45 代理路径）、p3 测量同径生效，无第二实现可漏改。实测影响（`reports/wp16b/comparison.json` `continuity`，本次解析）：

- **engine 侧 vs 原始 p45 报告：0 处不一致**（引擎未被 p52/p53 触及的直接证据）；
- **proxy 侧 5 处变化，全部归因 16-A**：MOM +0.3629%→+0.5467%、RSQ +7.4622%→+7.6817%、REV −0.5700%→−0.4240%、IND −10.5158%→−10.2170%、combo −3.5410%→−1.7528%；**方向全部不变**；
- val_reward 数值本身随之变化属预期（FROZEN_C §5 红线 4"16-C 证据层只保证数组里的数=证据里的数，不锁数值大小"）；wp16c 面板实例：2021 折 MOMENTUM_20 val_reward_plus=−0.1066 / minus=−1.0114（`reports/wp16c/combo_pct_measurement.json`）。

### 2.5 措辞边界（冻结，本报告遵守）

- **历史冻结产物不动**：原 `D:/minequant/reports/p45/**`、`reports/p46/**` 只读——本次核实 mtime 仍为 2026-09-20（p45：14:47:53/14:53:07；p46：16:07:42/16:16:43），全程未写入；重跑输出全部落新目录 `AlphaGPT/reports/wp16b/`、`reports/wp16c/`。
- **既往择向不回改**：16-C 红线 2——direction VALUES 不变；wp16c 面板的 member_directions 是**修复后计量口径下的重跑记录**，不回改任何历史冻结产物中的方向。
- **重选属后续工作**：以修复后 reward 重新择向/重新排名候选，不在本包三提交内；三提交均未执行任何重选（16-B 分解"只回答原差异多少消失/仍在/新增，不回答生产择向是否被复现"，FROZEN_B §3.3 判读纪律）。

---

## 3 16-B：P45 可比化（scoring_parity / backtest）

### 3.1 六项冻结清单落实表（FROZEN_B §1 F1–F6；落点本次读码核实）

| 冻结项 | 状态 | 落点（工作区现状） |
|---|---|---|
| F1 收益基础=复权开盘价 | 已有→钉测试 | 标签基 `open_to_open_returns(raw_cache["adjusted_open"])` 保留；新增 harness 侧缺键 fail-closed `_REQUIRED_CACHE_KEYS`（`ashare_model/scoring_parity.py:62`、`:341`、`:350`）；S1 含 −50% 负对照 |
| F2 成交基础=原始价格 | 引擎已有→钉测试 | 引擎 raw open 驱动执行不动（红线 3）；basket 权重空间差异如实记录，未造价格层 |
| F3 local_signal_range 贯穿两径 | **本包实现** | `compare_candidate_parity`（`scoring_parity.py:280`）与 `run_fixed_candidate_parity`（`:634`、`:669`）新增参数；`_run_orientation` 同一元组传 sim 与 engine（B3.1 桩断言）；缺省 `None` 保持旧默认路径（加法式，红线 3） |
| F4 长度/日期不一致拒绝 | **本包实现（删裁剪）** | 删 `n=min(...)`/`[:n]`；三族各自长度相等断言 + 四要素消息（`scoring_parity.py:257-263`）；日期轴长度/重复校验（B4.2）；2-D raw_cache 逐键形状校验；一切拒绝先于误差计算（B4.4） |
| F5 调仓轴/PIT/阻断 | 已有+显性计量 | 调仓轴/universe 同对象传递保留；新增 `CandidateParityRow.blocked_extra_buy_columns`（`scoring_parity.py:184`、`:412-416`，basket_blocked & ~engine_blocked 列计数）；B5.1 零不对称钉测试 |
| F6 同一固定方向+费用/资本 | 已有→钉测试 | 行级误差族用候选给定 direction（B6.3）；同 cfg 对象（B6.1）；零费两路径费用序列恒 0（B6.2） |
| 完整出口/contiguity（B3.3/B3.4/B3.5） | **本包实现** | `scoring_parity.py:359-383` 越界索引/n_dates/t+delay+1 三要素消息预校验；引擎 contiguity 语义直钉 |

引擎 `ashare_model/backtest.py` 未因 16-B 改动（红线 3；工作区该文件唯一 diff 是 WP15 守卫，见 §0）。

### 3.2 四场景结果

S1–S4 冻结值与实测对照见 §1.2 表；四场景共 10 个测试全部通过（本次复跑包含在 50 passed 内）。补充：S4 两种注入方向（engine 侧 dates/cache 多一列、sim 侧多一列）均以 ValueError 拒绝，等长对照输入正常完成（拒绝逻辑不误伤）。

### 3.3 P45 面板重跑：消失/仍在/新增逐项数字

协议（FROZEN_B §3.2 两遍）：脚本 `reports/wp16b/run_measurement.py`（机制复用只读原件；同 r1_medium_a、folds[2] 2023、n10_m2500、5 候选方向全 +1）。三遍：pass0 `lsr=None`、pass1 `lsr=(0,240)`、pass2 `lsr=(0,231)`（折真窗，窗口 242 列）。逐字命令与两轮脚本修正（首轮 exit 1 为脚本自身汇总代码 dict/dataclass 笔误，测量路径已完整；修脚本后 exit 0）记录于 `reports/p53_red_evidence.txt`。数字本次从 `comparison.json` 重新解析：

**连续性控制**：pass1 vs pass0 数值不一致 **0 处**（16-B 显式窗口对默认路径零行为变化）；engine vs 原始报告 **0 处**；proxy vs 原始 **5 处**（全部归因 16-A，数值见 §2.4）。

**分解（原始 → pass2，20 项 = 5 候选 × 4 族；族级计数 resolved 0 / remains 3 / within_both 13 / new-regression 4）**：

| 候选 | 族 | old | new（pass2） | 界 | 判定 |
|---|---|---|---|---|---|
| MOMENTUM_20 | daily_net | 8.6899e-03 | 8.6895e-03 | 2.216e-03 | 仍在 |
| REVERSAL_60 | daily_net | 1.0354e-02 | 1.0362e-02 | 2.216e-03 | 仍在（+7.6e-06，计入新增桶） |
| IND_REL_RET_20 | daily_net | 1.5255e-02 | 1.5255e-02 | 2.216e-03 | 仍在 |
| combo:all3 | daily_net | 1.0784e-02 | 8.2353e-03 | 2.216e-03 | 仍在（数值降约 24%，仍超界） |
| RSQ_60 | turnover | 1.5585826e-02 | 1.5585830e-02 | 0.1108 | 新增（+3.6e-10，界内） |
| REVERSAL_60 | fee | 1.2517e-04 | 1.2529e-04 | 5.54e-04 | 新增（+1.2e-07，界内） |
| REVERSAL_60 | turnover | 4.8922e-02 | 4.8968e-02 | 0.1108 | 新增（+4.6e-05，界内） |

- **消失（resolved）= 0**：统一口径后没有任何族从超界转为界内（RSQ_60 daily_net 1.078e-2→…注意 RSQ daily_net old=2.139e-03 本就界内，new=1.495e-03 仍界内）。
- **仍在**：4 行 daily_net 双侧超界（MOM/REV/IND/combo），族级计数 3（REV 的 +7.6e-06 微增计入新增桶）。
- **新增（new-regression）= 4**，且四项 **pass0==pass1==pass2 逐位相同**——全部源于 16-A（p52）先落地的 basket 语义（费预留使 basket 换手成为"已付费换手"），**无一来自 16-B**（pass1≡pass0 已排除）；绝对量 3.6e-10…4.6e-05，均远在界内。FROZEN_B §3.3 对 new-regression"必须为零"的要求针对 16-B 修复引入的新差异——按该口径 16-B 自身新增为 0；16-A 的四处已在 p52 语义变化中声明并归因（此句为报告判读，依据 = comparison.json 的 bit-identical 证据与 p53 日志 §5.2）。
- **方向/排名/verdict**：direction_flips {MOMENTUM_20, REVERSAL_60} 前后不变；ranking spearman 0.9 → 1.0；material inversion (MOMENTUM_20, REVERSAL_60) 在 pass2 消失；verdict unreliable → unreliable（reason 7 → 6 条，pass2 verdict 本次重读：6 条）。
- **F5 计量**：`blocked_extra_buy_columns = 242/242` 列（5 行全部）——basket 侧 loader 阻断（额外叠加 ~universe_mask 与 adj_factor 堕名禁买）逐列严于引擎自算矩阵，不对称从隐性变显性计量。

### 3.4 两诊断命名契约（FROZEN_B §4）

- 诊断一：**测试窗累计净收益方向对照** `test_window_total_return_orientation (proxy|engine)`（面板内 ± 方向各跑一条路径取测试窗累计净收益高者）；`direction_flips` = **方向翻转诊断**。
- 诊断二：**生产验证 reward 择向（Rule A）**（`candidates.py:652-685` 语义；本面板不测量、不复现）。
- 落地（本次读码/解析核实）：`scoring_parity.py:53` `ORIENTATION_DISCLOSURE`（§4.2 冻结披露串逐字："proxy/engine_direction = 测试窗累计净收益择向（诊断），非生产 Rule A 验证 reward 择向"）；verdict 携带 `orientation_disclosure` 键（`:624`）；flip 措辞改 §4.4 允许形式（`:574`）；重跑 JSON 顶部 `diagnostic_naming` 两分列示；`test_p53_s42_rerun_output_text_honors_the_naming_contract`（`tests/test_p53_wp16b_p45_parity.py:819`）扫描 `reports/wp16b/` 全部输出文本断言披露串存在 + §4.3 禁用表述缺席。审核修复后 `run_console.txt` 每个 pass 块与对比汇总块打印披露串——本次 `grep -c "测试窗累计净收益择向" reports/wp16b/run_console.txt` → **4**。
- 边界重申：本面板**不得**表述为"生产择向路径已复现"（§4.3 禁用清单）。

---

## 4 16-C：一次评分保存两个固定方向的原始证据（direction / candidates）

### 4.1 证据模式（落点本次读码核实）

- `ashare_model/candidates.py:390` `score_many` 新增 `oriented_evidence_out` 出参：一次既有的 ±signal 交错批量（`:469-475` 结构原样）把两侧 `rewards/val_rewards/icir/val_icir/objectives[B,8]` 逐元素带出（plus=2k/minus=2k+1，与 `_choose_direction` 消费同批同源）；被拒候选记 `rejected_reason`、数字留 None 不造数（`:664-667`）。
- `ashare_model/direction.py:249` 新增 `reference_return_provenance` 组装 FROZEN_C §1 字段模式：双侧 train/val reward+ICIR、`objectives_plus/minus`（费用与活动由此出）、`train_signal_range`、`train_dates`/`val_dates`（新增可选 `dates` 参数，`:358`；不传则 None，p46 既有调用面零改动）、`val_windows`、`reward_clip=[-10,+10]` 披露、`direction`、`applied_signal_basis`、`tie_break_reason`、`canonical_orientation_locus`、`selection_rule`（正式命名"参考配置下的验证 reward 择向（Rule A）"）+ legacy 四键（真实 chosen/opposite 语义，向后兼容 p46 N1）。非有限浮点按严格 JSON 惯例存 None。
- `reference_return_direction` 删除两次 `scorer.score(...)`（旧 `direction.py:194-197` 镜像缺陷——"opposite"拿回同一胜者），改为**一次** `score_many`（`:405-417`）；`ashare_model/eval_search.py:140-156` 删除整轮镜像重评分（旧 163-179），baseline 行经 `oriented_evidence_out` + `reference_return_provenance` 一批出证（`:173`）。**红线 1（禁止二次调用自动择优评分器取反方向）与红线 3（Direction VALUES 不变）均落实。**
- 两个真实数值锚（FROZEN_C §1）：落选侧真值不再与胜者同值记两遍——p54 合成例胜者 +10（clip 高）/落选 −10（clip 低，携披露）；实盘面板逐成员记录真值（如 2021 折 MOMENTUM_20：val+ −0.1066 / val− −1.0114；2024 折 RSQ_60：val+ +0.9709 / val− −0.5978，均为 `validation_reward` 分支，本次解析自 `combo_pct_measurement.json`）。

### 4.2 三不变量结果

V1/V2/V3 全部通过（§1.3 表；含独立批量逐元素对账、恰一次评分批量的 monkeypatch 断言、取反交换+施加暴露不变、毒化（×−10⁹ 与 NaN）全字段不变）。既有锚不改动仍绿：p46 N1（四键存在）与 N4（毒化）随本次 69 passed 一并通过。

### 4.3 选择规则未变的 diff 证明

`git diff -- ashare_model/candidates.py`（本次执行）显示**唯一删除行是 `from typing import Callable, Iterable, Sequence` → `from typing import Any, Callable, Iterable, Sequence`**（+83 行新增全部在 `_choose_direction` 之外：`oriented_evidence_out` 出参、被拒行组装等）；`_choose_direction` 比较键、扫描序、默认 +1 零改动。`direction.py` 的删除行全部落在被替换的两次 `scorer.score` 旧路径与旧 docstring（本次 `git diff | grep "^-"` 核实）。新增 `classify_rule_a_tie_break`（`direction.py:171`）为纯分支归类器（NaN→−inf 降级同款），以 6⁴ 网格测试断言与 `_choose_direction` 逐点一致。**证据修复与策略变更未混提交**（红线 2）。

### 4.4 p46 百分位小修与范围栅栏

- 面板：`reports/wp16c/run_measurement.py`（原 `D:/minequant/reports/p46/run_measurement.py` 只读，mtime 本次核实仍 09-20）。5 折逐列 `max_abs_diff == 0.0`，以**结构性恒等**呈现，产物含冻结原因串（"combo_signal 内部对成员再排名，消去任何列内严格单调尺度，包括百分位"）；engine 总收益 raw==pct 逐折相同（2021 折 −0.072613、51/51 单；2023 折 −0.005058、39/39 单；2025 折 +0.022971、53/53 单，本次解析）。
- 对照臂（仅测量脚本本地 helper：对已归一化成员取均值、无内部再排名）：5 折 spearman_min 0.9999981–0.9999993（≈1 的退化在 JSON 内解释：成员同列共享 N_eff 时均值为 combo_raw 的列内仿射映射）；top10_overlap_min = 0.90 于 2022/2024/2025 折、1.00 于 2021/2023 折——分离只在成员有限集不同的单元格显现。
- 栅栏遵守（本次核实）：`ashare_model/r01_baseline.py` 与 `ashare_model/reward.py` **零改动**（`git status -- <两文件>` 为空）；`R01_COMBO_ARMS`、词表、组合定义未动；combo 语义测试（p34/p42）随 p54 回归通过（红证据记录 62 passed，本次未复跑该组）。

---

## 5 审核汇总

三路章程 = 三份冻结规格各自的 §0.3 红线（FROZEN_A §0.3 五条：data/p50 不动、p51 守卫、ExecutionCostModel 唯一权威、引擎不改语义、证据/策略分离；FROZEN_B §0.3 四条：p45/p46 原件只读、基线不破坏、引擎不改、命名契约硬性；FROZEN_C §0.3 四条：禁止二次调用择优评分器、证据修复≠策略变更、p46 修复仅测量脚本+小测试、原件只读与基线不破坏）。**没有独立的审核汇总文件**（本次 `find`/`grep` 全库核实：无 `p52_*measurement_log`、无 "must-fix/章程" 命中的文件；.mimosa 任务评审均为 09-22 及以前）——审核发现与处置全部记录在三份红证据的 AUDIT FIX/勘误节与 FROZEN_A §8 附注内，逐条如下：

| # | 包 | 级别 | 发现 | 处置 | 复验 |
|---|---|---|---|---|---|
| 1 | 16-A | **blocker** | 无买入会话（尘额卖出被最低佣金地板支配）被豁免于 (C1) 拒绝，负现金经 total>1 scale-back 被洗掉；复现脚本 `reports/wp16/_recon/fix_repro_c1_buyless.py`（RED：`sum+c−1=+9.927673e-05`、buy=0、cash=0.0 被入账，exit 1） | 卖腿递延（最小名义优先、仅在实际缩小时保留）直至会话自融资费用 + (C1) fail-closed 保留为无门槛兜底；新增 2 测试 | 红证据 GREEN：p52 12 passed、六文件回归 108 passed、13 文件 194 passed 等（记录，未复跑）；本次报告复跑 p52 12 例通过（含于 50 passed） |
| 1a | 16-A | 路线试错记录 | 纯路线 1（无差别拒绝）先被红绿验证，但使 5 个既有绿测试失败（引擎对同会话精确记账负现金，parity 路径在 basket 侧崩溃）——记录于红证据 | 改路线 2（卖侧）+ 兜底；不镜像引擎负现金（FROZEN_A §1(C1) 禁止） | 同上 |
| 1b | 16-A | 勘误（2 条，经审核确认后**只追加**于 FROZEN_A 文末） | §8.1：§5.9/§7"断言数值不改"与 §1(C1)/§3.4 在 7 处满仓 pin 上数学互斥 → 7 pin 按 (C1) 等式重锚（表+依据）；§8.2：§2.3"现金≥0 恒成立"仅覆盖买入腿 | 文档与测试恢复一致；p3 容差与 test_p3 断言零改动 | 本次读 FROZEN_A 已含 §8（原文未删改）；被重锚文件随质量门通过（63 passed 含 test_execution/test_no_signal_semantics） |
| 2 | 16-B | **major ×2** | ① 重跑控制台输出出现 direction_flips/proxy-engine 方向值但无 §4.2 冻结披露串；② 命名测试只扫 parity JSON 不扫全部输出文本 | 脚本在每 pass 块与对比汇总打印 ORIENTATION_DISCLOSURE；测试改扫 `reports/wp16b/` 全部输出（JSON+txt）；面板整轮重生成，连续性/分解数字不变 | 红证据：重跑 exit 0、GREEN 19 passed（记录）；本次复跑 `grep -c` → 4、p53 19 例通过（含于 50 passed） |
| 3 | 16-C | 诚实更正（非审核条目，红证据 GREEN 节自录） | 两处 RED 期测试期望写错：V2 legacy 键镜像断言把 chosen 物理侧写反；baseline 接线测试未接受被拒分支（populated_db 的 MOMENTUM_20 行被拒：数字 None、tie_break None、携带 rejection_reasons） | 仅改测试期望，未触碰实现行为 | 本次 p54 19 例通过（含于 50 passed） |

**发现计数：blocker 1 + major 2 + 勘误 2 + 测试侧诚实更正 2 = 全部处置完毕，无未解决项**（处置证据见上表"复验"列；§6 为本次独立复跑）。

---

## 6 质量门记录（本次报告会话实际复跑，2026-09-23，cwd=D:/minequant/AlphaGPT）

| # | 命令（逐字） | 结果 | 退出码 |
|---|---|---|---|
| 1 | `python -m pytest tests/test_p52_wp16a_cash_identity.py tests/test_p53_wp16b_p45_parity.py tests/test_p54_wp16c_direction_evidence.py tests/test_p54_wp16c_combo_invariance.py -q` | **50 passed in 9.32s**（p52 12 + p53 19 + p54 19） | 0 |
| 2 | `python -m pytest tests/test_reward.py tests/test_p3_portfolio_parity.py tests/test_p46_direction_provenance.py -q` | **69 passed in 19.09s**（test_reward 57 与 p3+p46 12 基线保持） | 0 |
| 3 | `python -m pytest tests/test_p50_wp14_acceptance.py tests/test_p50_wp14_errata.py tests/test_p50_wp14_plan.py tests/test_p50_wp14_precheck.py tests/test_p50_wp14_reconcile.py tests/test_p50_wp14_status.py tests/test_p51_wp15_funding.py tests/test_p51_wp15_integration.py tests/test_p51_wp15_planner.py -q` | **144 passed in 8.75s** | 0 |
| 4 | `python -m pytest tests/test_p45_scoring_parity.py tests/test_execution.py tests/test_no_signal_semantics.py tests/test_ci_sharding.py -q` | **63 passed in 9.71s**（含被 §8.1 重锚的 test_execution/test_no_signal_semantics） | 0 |
| 5 | `python scripts/check_test_shards.py` | **shard union == full set: 137 files across 4 shards** | 0 |

**红证据记录、本次未复跑**（按本报告纪律如实标注）：`tests/test_evaluation.py tests/test_train.py -q` → 100 passed, 2 skipped in 655.37s（两个 score_many spy 原样通过）；全量套件曾于 p52 后跑出 14 failed（全部为被取代的未预留/乘法算术 pin，修复后）→ pin 重锚后 `tests/test_execution.py tests/test_no_signal_semantics.py -q` 29 passed；面板脚本两轮执行（wp16b 首轮 exit 1 后修复脚本汇总代码、wp16c 中途加退化注释后整轮重跑）详见 `reports/p53_red_evidence.txt`、`reports/p54_red_evidence.txt`。

**三份红证据位置（只追加）**：`reports/p52_red_evidence.txt`、`reports/p53_red_evidence.txt`、`reports/p54_red_evidence.txt`（工作区 AlphaGPT/reports/ 下；本次全文读取核实）。

**只读合规（本次核实）**：`D:/minequant/reports/p45/`（parity_report.json 09-20 14:53、run_measurement.py 09-20 14:47）与 `D:/minequant/reports/p46/`（direction_measurement.json 09-20 16:16、run_measurement.py 09-20 16:07）mtime 未变；`AlphaGPT/data/**` 本次未写入（WP16 各会话自查记录 + 本次无写操作）；FROZEN_A/B/C 原文未删改（A 的 §8 为文末追加）。

---

## 7 遗留与 deferred

1. **16-B 差异残留**：族级消失 0、仍在 3（+REV daily_net 微增计入新增桶）；4 行 daily_net 双侧超界。判读（p53 日志 §5.3）：统一口径后剩余差异属于近似本身（权重空间 vs 整手执行账本）与 F5 计量出的阻断不对称（242/242 列）——**前者是 basket 近似与引擎执行账本的结构差，后者是不对称的显性计量；两者均未在本包消除**（红线：不改引擎、不给 basket 造价格层）。面板 verdict 仍 unreliable（6 条 reason）。
2. **方向翻转诊断仍报 2 例翻转**（MOMENTUM_20、REVERSAL_60，测试窗累计净收益口径）——研究决策项，非本包范围；按命名契约不得解读为择向规则失效。
3. **16-A 语义变化后的重选**：以修复后（可加恒等式+费预留）的 reward/val_reward 重新择向、重排名候选与组合，属后续工作；既往冻结产物与择向不回改（§2.5）。
4. **引擎负现金记账**：引擎在地板佣金支配的尘额卖出上会精确记账负现金（FROZEN_A §8.2）；basket 已不镜像（递延）。若未来要求引擎侧同样现金非负，需在引擎侧另行立项（本包红线 4 禁改引擎语义）。
5. **p52 无 docs 侧 measurement log**（p53/p54 有）——如实记录的惯例缺口，测量记录以 `reports/p52_red_evidence.txt` 为唯一真源。
6. **16-C 边缘语义**（p54 日志 §6）：`dates` 不传则 `train_dates/val_dates=None`（p46 调用面零改动的代价；生产路径均传入）；被拒候选 §1 数字字段为 None、`tie_break_reason=None`（无选择发生，direction 保持旧实现 +1）——FROZEN_C §1 未枚举该边缘，None 为诚实值；对照臂在本面板 rho≈1 的退化已写入产物，成员覆盖差异更大的面板上分离会显现（测量事实，非承诺）。
7. **全量套件**：p52 会话曾记录 55m33s 全量（14 failed → pin 重锚后解决）；本报告未复跑全量（耗时），质量门以 §6 五项聚焦门 + 红证据记录覆盖。

---

*本报告产物路径：`AlphaGPT/reports/wp16/WP16_REPORT.md`（唯一写入物；data/**、reports/p45|p46 原件、FROZEN_*.md 均未触碰）。*
