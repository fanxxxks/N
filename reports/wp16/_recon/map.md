# WP16 侦察地图（只读侦察，2026-09-23）

本文件为唯一写入产物。所有行号基于当前工作区状态逐一核实（文件均全文读过）。
本包三提交测试前缀 p52/p53/p54；不得破坏 p50/p51 回归基线与 refuse_unwired_trade_groups 守卫。
基线验证（本次实际运行，cwd=AlphaGPT）：

- `python -m pytest tests/test_p3_portfolio_parity.py tests/test_p46_direction_provenance.py -q` → **12 passed in 11.12s**
- `python -m pytest tests/test_reward.py -q` → **57 passed in 6.70s**

p45/p46 原始产物在 **D:/minequant/reports/p45、D:/minequant/reports/p46**（工作区根的 reports/，非 AlphaGPT/reports/），本次只读摘要，未做任何写操作。

---

## 1. reward_baskets.py：`simulate_basket_daily_returns_batch` 现行精确实现（16-A 落点）

文件：`ashare_model/reward_baskets.py`（共 431 行）。标量入口 `simulate_basket_daily_returns`（line 81-137）纯委托批量版。

### 1.1 会话循环逐步（line 302-415）

- 状态初始化（line 293-295）：`prev = np.zeros((b, n_stocks))`（权重向量，**无显式现金分量**）；`capital = cfg.initial_capital`；`cost_model = ExecutionCostModel.from_config(cfg)`。
- 每会话（line 302，`t` 遍历 `[signal_start, signal_end)`，`exec_col = t+1`）：
  - 构造器产出（line 319-333）：`constructor.construct(signals[row,:,t], prev[row], capital=capital[row], ...)` → `target/buy/sell`（权重空间，无价格无股数）。
  - 费用（line 334-337）：`cost = cost_model.rebalance_cost_fraction(buy, sell, capital)`。`rebalance_cost_fraction`（`ashare_execution.py:179-194`）= 各腿佣金（带最低佣金地板，`ashare_execution.py:116-120`）+ 卖方印花税 + 双边过户费与滑点，除以交易前 capital → **c 是对交易前权益的比例**（与用户口径一致）。
  - 毛收益（line 346）：`gross = (target * rets).sum(axis=1)`，`rets = target_clean[:, t]` → **g = Σ a_i r_i 成立**（a_i = 交易后目标权重）。
  - 净收益（line 347）：`net = (1.0 - cost) * (1.0 + gross) - 1.0` → **乘法形式**，与恒等式 R_net = g − c 相差交叉项 **−c·g**。
  - 权益滚动（line 381）：`capital *= 1.0 + net`。
  - 漂移（line 396-415）：
    ```python
    drifted = target / max(1.0 - cost, 1e-12) * clip(1+r_i) / clip(1+g)
    total = drifted.sum();  scale = 1/total 当 total > 1
    drifted *= scale;  prev = canonical_weights_to_output_grid(drifted)
    ```
    即 **w'_i = a_i(1+r_i) / [(1−c)(1+g)]**，分母与 16-A 的 (1+g−c) 同差 c·g。
- **现金**：全程不存在显式现金权重。隐含现金 = 1 − Σ w'（仅当 Σ w' ≤ 1）。line 406-408 只在 `total > 1` 时缩回 1 —— **这正是"用最终权重归一化掩盖费用融资不足"的机关**（满仓 Σa=1 时 drifted 合计 = 1/(1−c) > 1，必触发缩放，负现金被洗掉；部分投资 Σa < 1−c 时不触发）。
- **买入不预留费用**：构造器的现金上限 `_cap_increases_to_available_cash`（`ashare_portfolio/constructor.py:57-69`，应用于 line 414 与 560）把增量买到 `available = 1 − Σbase` 为止，即 Σtarget 可达 1，**c 无处可付**。对照组：引擎买入有含费可负担性循环且 fail-closed（`ashare_model/backtest.py:356-368`），保证现金非负——16-A 的"实际买入必须先满足费用后现金非负"在引擎侧已有实现范本。
- 引擎侧同形确认（供差异对照）：引擎把费用从现金扣（backtest.py:331/370-371），`capital = cash + marked`（line 385），`net = capital*(1+gross)/capital_before − 1`（line 404）→ 同为乘法形式 (1−c)(1+g)−1（整手前精确同形）；漂移 `weights * (1+r)/(1+g)`（line 431-443）不除 (1−c)，因为 `weights = marked/capital` 已是费后权益权重（line 386）。

### 1.2 与 16-A 恒等式逐项差异

| 项 | 16-A 要求 | 现状 | 差异 |
|---|---|---|---|
| g = Σ a_i r_i | ✔ | line 346 ✔ | 无 |
| R_net = g − c（可加） | ✔ | line 347 乘法 (1−c)(1+g)−1 | 交叉项 c·g |
| w'_i = a_i(1+r_i)/(1+g−c) | ✔ | line 397-401 除 (1−c)(1+g) | 分母差 c·g |
| w'_cash = (1−Σa−c)/(1+g−c) 显式 | ✔ | **无现金分量**；隐含 = 1−Σw' | 现金被隐式化 |
| 买入先付费用、现金非负 | ✔ | 构造器 cap 不留费（constructor.py:57-69） | **费用融资缺口**：Σa 可=1，c 无人付 |
| 不得用归一化掩盖 | ✔ | line 406-408 total>1 才缩放回 1 | 满仓+费用时恰好掩盖 |
| 分母不得保留旧漂移 | ✔ | 若只改 net 不改 line 397-401 即違反 | 修复必须三处同改（net/漂移分母/现金） |

### 1.3 钉住旧行为的测试（改语义会动的清单）

- `tests/test_p3_portfolio_parity.py`（**最紧**）：line 76-78 注释明言两路径"share the SAME constructor, drift math and (1 − cost) * (1 + gross) − 1 net form"；3 个 parity 测试 + golden free path 以 atol（净 5e-4 / 费 5e-3 / 权重 5e-2）钉 basket↔engine。basket 改可加而引擎不动时，差额 ~c·g（面板 g≈0.011、c≈1e-3 → ~1e-5）大概率仍在容差内，**但这是需要实测确认的点**。
- `tests/test_reward.py`：`test_simulate_basket_matches_backtest_engine_daily_returns`（line ~309，abs=5e-4 含费 lot_size=1）；p45 机制界 parity（line 605-640）；`test_basket_underfilled_keeps_cash_like_backtest_engine`（line ~996：零费手算 (1/3)·0.01+(1/3)·0.02，abs=1e-12 —— **零费下两形式重合，16-A 后应仍过**，且是"部分投资"验收的现成锚）；极值对比（line 931/941）。
- `tests/test_portfolio_objectives.py`（line 154 "daily nets agree to the whole-lot execution friction"）、`tests/test_baseline_harness.py`（cap 绑定）。
- 引擎侧手算链（改 basket 不动，但作 16-A"手算路径"参照）：`tests/test_no_signal_semantics.py:143-176`（费后权重膨胀 1/(1−费) 手算）、`tests/test_p31_b02_entry_canary.py:715`（net1 = (1 − 74.1836/100000)·(1+…) 手算）。
- **回归红线**：`reward_baskets.py:201-203` 的 `refuse_unwired_trade_groups(cfg, ...)`（p51 守卫，标量入口经批量覆盖）与 p50 六文件锚点机制不得触碰。

### 1.4 调用方（reward 链）影响面

- 再出口门面：`ashare_model/reward.py:311-315`（reward.py 只留指标/组装，basket 层整体搬至 reward_baskets.py，语义单径）。
- 直接调用：`reward.py:462`（`formula_reward` 标量）与 `reward.py:567/608`（`batched_basket_rewards` 训练窗 + 逐 val 窗重启动，line 598-639：**val_reward = 各窗中位数的 active_IR − cost_weight·年化平均费用，再 clip**）。
- 上游：`candidates.py:13/281/476`（score_many 的 reward_function）、`train.py:52/185`（训练循环）、`eval_search.py:111-146`（baseline 候选）、`p3_measurement.py:479`、`scoring_parity.py:180`（p45 代理路径）。
- 影响面结论：改 `simulate_basket_daily_returns_batch` 的净收益/漂移/现金即同时改 **训练 reward、val_reward（择向键）、p45 代理路径、p3 测量**——一条语义径，无第二实现可漏改。

---

## 2. scoring_parity.py 与 backtest.py：16-B 冻结清单逐项对照

文件：`ashare_model/scoring_parity.py`（447 行）、`ashare_model/backtest.py`（859 行）。

| 冻结项 | 状态 | 证据 |
|---|---|---|
| 收益基础=复权开盘价 | **已做** | scoring_parity.py:234-241（`target = open_to_open_returns(adjusted_open)`，注释引 p19 §2.3 并说明原始价系统性发散）；backtest.py:198-203+238（引擎 target_ret 同基）；eval_folds.py:175-184（epoch_slice 同基） |
| 成交基础=原始价格 | **已做（引擎侧）** | backtest.py:130-139+183-188（raw `open` 驱动一切执行决策）；line 290 `p_entry = open_[:, entry_day]`；line 164-181 缺 p19 键 fail-closed。basket 为纯权重空间无价格（费用按权重×capital，ashare_execution.py:94-101） |
| local_signal_range 传两条路径 | **缺失** | `_run_orientation`（scoring_parity.py:180-197）对 sim 与 engine **均不传 signal_range**（各自默认 (0, n−2) / range(n−2)）；`compare_candidate_parity`（line 201-212）与 `run_fixed_candidate_parity`（line 414-447）**无该参数**。引擎本身支持并校验 signal_range（backtest.py:107, 222-236：完整 t+delay+1 出口 + contiguous 强制），但 harness 未用 |
| 长度/日期不一致拒绝 vs 裁剪 | **缺失（现状=裁剪）** | scoring_parity.py:286-296：`n = min(sim_daily.size, eng_daily.size)` 后 `[:n]` 比较，费用（line 291-292）与 turnover（line 295-296）同样 min 截断。**正是被禁止的"取两侧最短"**。当前靠 epoch_slice 同刀切片使长度相等（test_reward.py:631 有 `len == len` 断言，但 harness 本身不拒） |
| 同一固定方向 | **部分** | 行级误差族用候选传入 direction（line 283 `by_orientation[int(direction)]`）；p45 面板冻结 +1（run_measurement.py:5-6, 76, 84 并在 log 声明）。但 `proxy_direction`/`engine_direction`（line 271-280）按**测试窗累计净收益**双向择优 —— 与生产择向（验证 reward，candidates.py Rule A）是两种诊断，**命名纪律问题**：`direction_flips`（line 281-282, 378-395）话术"proxy and engine choose opposite directions"未区分 |
| 同一费用配置与资本口径 | **已做** | 两路径同 `backtest_config`：basket `ExecutionCostModel.from_config(cfg)`+`cfg.initial_capital`（reward_baskets.py:294-295）；引擎同（backtest.py:84-86） |
| PIT 掩码与交易阻断对齐 | **部分** | 同一 universe_mask 传两径（scoring_parity.py:186/195）；但 basket 的 `blocked_buy` 来自 `loader.tradability_masks()`（data_loader.py:871-911）——在 exact_tradability_matrix 之上**额外叠加** `blocked_buy |= ~universe_mask`（line 907）与 adj_factor 堕名禁买（line 909-911）；引擎在 harness 内自行由 raw_cache 计算（backtest.py:206-219）**无这两层**（其资格走 constructor `eligible`）。→ basket 侧买入阻断严格更强，非对称 |
| 实际日期轴 | **部分** | 引擎收 `list(dates)`（scoring_parity.py:194）；basket 无日期概念（纯列索引）；比较是位置对位置。日期一致性靠 epoch_slice 的同一 `loader.dates[s0:s1]`（eval_folds.py:191）间接保证，harness 不校验 |
| 全局调仓轴 | **已做** | `rebalance_mask = fold.policy.rebalance_mask(loader.dates)[s0:s1]`（eval_folds.py:174，全局策略切片防 5/10 日节奏在折边界重启）；两径同传（scoring_parity.py:188/196） |

其他要点：
- `run_fixed_candidate_parity`（scoring_parity.py:428-436）把 `blocked_*_full` 切到 `[test_signal_start:test_price_end]`，与 signals 同刀。
- `compare_candidate_parity` 自算 target 用**稠密 o2o**（line 239-241），弃用 epoch_slice 的 `causal_target_returns` 稀疏 target（eval_folds.py:177-182）——引擎同样稠密（backtest.py:238），两径内部一致，但与训练 reward 的稀疏标签基不同（如实记录，非缺陷判定）。
- 判定门：`proxy_reliability_verdict`（scoring_parity.py:341-411），冻结机制界系数 line 37-40（0.2g/0.05g/10g/0.05g·期）。

### 2.1 p45 面板机制与原始差异量（只读摘要）

- 脚本：`D:/minequant/reports/p45/run_measurement.py`。r1_medium_a 配置，2023 折（folds[2]），参考格 n10_m2500：top_n=10, buy_rank=10, sell_rank=15, cap=0.10, min_trade=2500, capital=100k, freq=every_10_days/horizon=10, budget=0.1/priority。5 候选 = MOMENTUM_20/RSQ_60/REVERSAL_60/IND_REL_RET_20 四单因子 + combo:all3（line 74-84），方向全 +1。输出 parity_report.json（line 105-131）。
- `parity_report.json` 摘要（python json 只读解析）：
  - verdict=**unreliable**，7 条 reason；granularity g=0.01108，界 daily_net=2.216e-3 / fee=5.54e-4 / turnover=0.1108 / total_net=0.13296。
  - daily_net_error_max：MOMENTUM 8.69e-3、RSQ 2.139e-3（界内）、REVERSAL 1.035e-2、IND_REL 1.526e-2、combo 1.078e-2。
  - direction_flips：MOMENTUM_20（proxy +1 / engine −1）、REVERSAL_60（proxy −1 / engine +1）。
  - 排名：spearman=0.9，material inversion (MOMENTUM_20, REVERSAL_60)。
  - 累计净收益：proxy/eng = +0.36%/+2.45%（MOM）、+7.46%/+7.58%（RSQ）、−0.57%/+2.59%（REV）、−10.52%/−6.09%（IND）、−3.54%/−0.51%（combo）。
- **16-B 重跑基线即此面板**（新输出目录，原 json 只读）；"差异多少消失"以此为对照列。

---

## 3. direction.py 与 candidates.py：一次评分批量与镜像缺陷（16-C 落点）

### 3.1 一次评分批量返回什么（candidates.py）

- `score_many`（line 374-608）把通过预检的候选堆成 `raw = np.stack(batch_signals)`，然后（line 469-475）：
  ```python
  oriented = np.empty((raw.shape[0] * 2, *raw.shape[1:]))
  oriented[0::2] = raw;  oriented[1::2] = -raw   # +signal/-signal 交错
  ```
  注释明言"makes mirror invariance straightforward to audit"。
- `reward_out = self.reward_function(oriented, ...)`（line 476-490）→ `rewards, val_rewards, icir, val_icir = reward_out[:4]`（line 491）——**两方向的 val_reward/val_icir 都在数组里**（plus=2k, minus=2k+1）。
- 逐候选（line 521-533）：`_choose_direction(raw[k], val_windows, val_rewards[plus], val_icir[plus], val_rewards[minus], val_icir[minus], universe_mask)` → `chosen = plus if direction == 1 else minus`（line 534）→ **只有 chosen 侧的数值进 CandidateScore**（line 544-574：direction/val_reward(−penalty)/val_icir/train_reward/train_icir/复杂度/active_ir/risk_exposure/turnover/capacity/eligible/reasons）。**落选方向的全部数字在 line 534 被丢弃**。

### 3.2 Rule A 精确比较键与 tie-break（candidates.py:652-685）

```python
plus_key  = (plus_reward  非有限→−inf, plus_icir  非有限→−inf)   # 字典序元组
minus_key = (minus_reward …, minus_icir …)
plus_key > minus_key → +1;  minus_key > plus_key → −1
平局 → 镜像稳定规范朝向：按 val_windows 顺序扫 eligible 列（values.T.ravel()），
      第一个有限非零观测的符号；全零/全非有限 → +1
```
注意：择向用**未扣复杂度罚**的 val_rewards（罚对两方向同值，无影响）；比较键在 `_choose_direction` 内部对非有限降级为 −inf。

### 3.3 direction.py 两次调用各拿到什么（缺陷仍在）

`reference_return_direction`（direction.py:144-212）：
- line 194：`chosen = scorer.score(spec, signal, ...)` —— score 内部**已双向择优**，返回胜者。
- line 195-197：`opposite = scorer.score(spec, -signal, ...)` —— 同样内部双向择优，**拿回同一物理胜者**（direction 字段翻号，val_reward 数值相同）。
- 证据块（line 206-211）：`{"val_reward", "val_icir", "val_reward_opposite", "val_icir_opposite"}` —— **val_reward_opposite == val_reward（同值记两遍），落选侧真值缺席**。docstring（line 163-169）只对 direction 字段成立。
- 同构缺陷在 eval_search.py:150-203：`opposite_scores = scorer.score_many(specs, [-s for s in train_signals], ...)`（line 163-179）后 `opposite_scores[i].val_reward` 进 provenance（line 200）——同值；且这是**一整轮重复评分**（非从已有批量取证）。
- r02/r03 只记胜者侧三元组 {direction, val_reward, val_icir}（r02_matrix.py:282-287、r03_combos.py:434-438）。

### 3.4 两个真实验证 reward（−1.8418 与 −10.0）丢失环节

- 两数都在 `batched_basket_rewards` 的 val_rewards 数组中算出（reward.py:591-643：各 val 窗 active_IR − cost_weight·年化费 → clip(reward_clip_low=**−10.0**, reward_clip_high=+10.0)（config.py:222-223）→ 窗中位数；**−10.0 即 clip 地板值**，−1.8418 为胜者侧）。
- 丢失点链：① candidates.py:534/550-551 只留 chosen；② direction.py:195-197 / eval_search.py:163-179 用"会自动择优的评分器"再跑一遍取"反方向"，取回的仍是胜者 → DirectionDecision.evidence 与 baseline 行的 *_opposite 字段均无落选侧真值。16-C 红线"禁止再次调用会自动择优的评分器来取得反方向结果"即针对此。
- 现保存证据缺 16-A/B 要求的：两方向费用与活动（objectives 里有 val_exposure/val_turnover 但仅 chosen 侧，candidates.py:558-573）、训练/验证日期（仅 train_cutoff 字符串）、tie-break 原因（未记录决策走 reward / icir / 规范朝向哪一支）。

### 3.5 相关既有测试（p46）

`tests/test_p46_direction_provenance.py`：N1（证据四键**存在**，line 106-108——不断言 opposite ≠ 本侧，故镜像修复后仍绿）、N1 dispatch 拒绝、N2 label_ic 纯信号、N3 可复现可归因、N4 **毒化测试窗不改训练决策**（line 196-231，16-C 验收已有锚）、N6 percentile 手算、N6 双规则镜像稳定、N5 baseline 行带 provenance（populated_db）。r02/r03 侧 test_p37_r02_matrix.py / test_p42_r03_combos.py。

---

## 4. reports/p46/run_measurement.py：百分位流程与 combo 内部排名（p46 修复落点）

文件：`D:/minequant/reports/p46/run_measurement.py`（215 行，只读）。

- 百分位流水（§3 段，line 151-161）：
  - line 156：`combo_raw = combo_signal(member_signals, fold_data.universe_mask)`
  - line 157-160：`member_pct = [(d, percentile_normalize_signal(s, fold_data.universe_mask)) ...]`
  - line 161：`combo_pct = combo_signal(member_pct, fold_data.universe_mask)`
- `percentile_normalize_signal`（ashare_model/r01_baseline.py:170-191）：`rank_normalize_signal` 的每列秩 ÷ 该列 N_eff → (0,1]。
- `combo_signal`（r01_baseline.py:194-217）：**line 203-209 对每个成员再跑一次 `rank_normalize_signal(direction * signal)`** —— 秩变换吞掉任何严格单调尺度：percentile = rank/N_eff 再排名 ≡ rank 本身（同列 N_eff 为正常数）。
- 实测确证（direction_measurement.json 只读解析）：5 折全部 `rho=1.0、top10_overlap=1.0/1.0、engine_total_raw == engine_total_pct（位相同）`、orders 相同（例：2021 折 raw=pct=−7.261306%，51/51 单）。**百分位尺度被完全消去，用户定性成立。**
- 面板其余：§1 有效截面（line 108-122）、§2 Rule A vs Rule B 方向（line 124-149；json 内逐折 rule_a/rule_b/agreement——2021 折 4 中 3 不合，2024 折全合）、引擎对照 run（line 175-185，传 fold_data.raw + fold_data.rebalance_mask）。
- 修复边界（用户红线）：仅测量脚本 + 小测试；不扩词表、不切换组合定义（percentile helper 本就刻意不进 R01_COMBO_ARMS，r01_baseline.py:179-182）。
- 相关测试：`test_p46_direction_provenance.py::test_n6_percentile_normalize_hand_case`（helper 自身，line 230-244）；combo 语义由 test_p34_r01_baseline.py / test_p42_r03_combos.py 钉住。**无任何测试钉"percentile 经 combo 后被消去"这一事实**（新 p 测试的空位）。

---

## 5. 风险与注意（供实现者）

1. 16-A 若只改 basket 而不动引擎，p3 parity 的容差（净 atol 5e-4）大概率容纳 ~c·g 差额，但**必须实测**而非推断；`test_reward.py:996` 零费手算不区分两形式。满仓+费的手算验收用例会直接撞 line 406-408 的缩放机关——该行为必须按 16-A 重写（现金显式化），并把用户反例做成独立失败测试（净收益、下期权重、现金三查）。
2. 16-B 的"拒绝不一致"要把 scoring_parity.py:286-296 的 min 裁剪改为长度/日期断言；同时 local_signal_range 需新增参数贯穿 compare_candidate_parity → _run_orientation → 两路径。折尾部故意多一天的拒绝测试可直接喂长一天的价格矩阵构造。
3. 16-C 修证据不得改 Rule A 选择键（红线：证据修复与策略变更分离）；正确路径是从一次 score_many 的交错批量里把两侧 val_rewards/val_icir/objectives 直接保存（candidates.py:491-496 数组已在手），而非二次调用评分器。N4 毒化测试已有，可扩展。
4. 命名纪律：p45 的 proxy/engine_direction 是"测试窗累计净收益择向"诊断，不得表述为"生产择向路径已复现"；生产择向 = Rule A 验证 reward（§3.2）。
5. 产物路径：p45/p46 原始产物在 D:/minequant/reports/{p45,p46}（只读）；WP16 重跑面板输出必须写新目录；红证据 reports/pNN_red_evidence.txt 只追加。
