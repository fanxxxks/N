# WP16 FROZEN B —— 16-B P45 可比化（scoring_parity / backtest）冻结规格

冻结日期：2026-09-23（实现存在之前；冻结时点核实见 FROZEN_A.md §0.4，同一次核实覆盖本文件：无 p53 测试、无 WP16 代码）。

## §0 文档地位、真源与使用规则

### §0.1 地位
- 本文件是 16-B 的**唯一期望真源**：`tests/test_p53_wp16b_*.py` 的一切断言（含错误类型与消息要素）只能引用本文件 §编号。
- 对实现者/修复者**只读**；冲突修实现；勘误只可以附注追加于文末。
- 推导依据：用户 16-B 原文、`reports/wp16/_recon/map.md` §2/§2.1、本会话逐行核实的代码（下列）与对 `D:/minequant/reports/p45/parity_report.json` 的只读解析（§3.4 基线数值出自本会话解析，非转抄）。

### §0.2 推导依据（本会话逐行核实）
- `ashare_model/scoring_parity.py`：165-198（`_run_orientation`：sim 与 engine **均不传 signal_range**）、201-212（`compare_candidate_parity` **无 signal_range 参数**）、226-241（open 形状校验；标签基 = `open_to_open_returns(raw_cache["adjusted_open"])`，注释引 p19 §2.3 并说明原始价系统性发散）、249-283（± 双向探测 `proxy_direction`/`engine_direction` 按**测试窗累计净收益** `max(...)` 择优；行级误差族用候选给定 direction）、**286-296（`n = min(sim_daily.size, eng_daily.size)` 后 `[:n]` 比较；费用/换手同样 min 截断——被禁止的"取两侧最短"）**、414-447（`run_fixed_candidate_parity`：epoch_slice 单刀切片 + blocked 矩阵同刀切 `[test_signal_start:test_price_end]`）。
- `ashare_model/backtest.py`：84-86（引擎同 cfg 构建费用模型/资本）、107-108（`execution_delay` 默认 1）、150-152（delay ≥ 1 强制）、164-181（p19 键缺失 fail-closed，侦察 map 核实段）、183-188+198-203+238（raw `open` 驱动执行；target_ret 同 adjusted_open 基）、206-219（引擎内部由 raw_cache 计算 exact_tradability 矩阵）、222-236（signal_range 校验：`t + delay + 1 >= n_dates` 完整出口拒绝 + contiguity 强制）、290（`p_entry = open_[:, entry_day]`）。
- `ashare_model/eval_folds.py:171-193`（epoch_slice：factors/raw/universe_mask 同刀 `[s0:s1]` 切片；**line 174 `rebalance_mask = fold.policy.rebalance_mask(loader.dates)[s0:s1]` 全局策略切片**；dates = `loader.dates[s0:s1]`）。
- `ashare_model/data_loader.py:871-913`（`tradability_masks`：exact p19 §2.4 规则之上**额外** line 910 `blocked_buy |= ~universe_mask` 与 line 913 起 adj_factor 堕名禁买——basket 侧阻断严格更强）。
- `ashare_model/reward_baskets.py:212-215`（sim 侧 signal_range 默认 `(0, max(n_dates−2, 0))`）。
- `ashare_model/ashare_execution.py:179-194` 与 `backtest.py:373`（两路径费用分母同为交易前权益）。
- `D:/minequant/reports/p45/run_measurement.py`（全文 137 行只读：r1_medium_a、folds[2]、cell n10_m2500、5 候选方向全 +1、输出 parity_report.json）；`D:/minequant/reports/p45/parity_report.json`（本会话只读解析，§3.4）。
- `ashare_model/r01_baseline.py`（combo 构造，p45 面板候选来源之一）。

### §0.3 全局红线（16-B 部分）
1. `D:/minequant/reports/p45/**` 与 `D:/minequant/reports/p46/**` 原始产物**只读**；重跑输出写新目录（建议 `AlphaGPT/reports/wp16/p45_rerun/`）。
2. 不破坏 p50/p51 基线与 `refuse_unwired_trade_groups` 守卫；不破坏 `test_p3_portfolio_parity.py` 合集基线（12 passed）与 `test_reward.py`（57 passed）。
3. 不改引擎语义（引擎现状已满足冻结项 F1/F2/F6 的引擎侧要求，见各条"现状"）。
4. 两个诊断必须按 §4 命名契约区分；重跑结论不得越界表述。

---

## §1 六项冻结与机器可检不变量

> 编号 F1–F6 对应用户原文六项。"机器可检不变量"= p53 测试可直接断言的性质；B*.x 为不变量编号，供测试引用。

### F1 收益基础 = 复权开盘价（已做，冻结 + 钉测试）
**冻结**：两条路径的收益标签唯一由 `raw_cache["adjusted_open"]` 的开盘到开盘收益（o2o）推出；禁止原始价收益基。
**不变量**：
- B1.1 harness 的 target 必须**调用同一个** `open_to_open_returns`（scoring_parity.py:239-241 现行实现）对 `adjusted_open` 计算得到——禁止在测试或 harness 里出现第二个 o2o 实现。
- B1.2 引擎 target_ret 同基（backtest.py:238 现行已满足）；`raw_cache` 缺 `adjusted_open` 键 → 两条路径都必须 fail-closed（引擎 p19 键校验 backtest.py:164-181 已满足；harness 侧同要求：键缺失在比较前抛错）。
- B1.3 拆股负对照（§2 S1）：构造 adj_factor ≠ 1 事件后，两路径收益必须等于 o2o(adjusted) 手算值，而原始价口径会得出 −50% 假收益——测试同时断言"正确值出现"与"错误值不出现"。
**现状**：scoring_parity.py:234-241 + backtest.py:198-203/238 已做；p53 只需把 B1.1–B1.3 钉成测试。

### F2 成交基础 = 原始价格（引擎侧已做；basket 为权重空间——如实记录为设计差异）
**冻结**：一切执行决策（入场定价、整手、可负担性、涨跌停/停牌矩阵）用 raw `open`。basket（reward 代理）是纯权重空间、无价格——此不对称**如实记录**，不是缺陷，p53 不得给 basket 造价格层。
**不变量**：
- B2.1 拆股日：引擎以 raw open 重推股数（backtest.py:292-307 语义）→ shares 翻倍、名义不变、净值不跳变（§2 S1 手算）。
- B2.2 缺 bar（open NaN→0）的名不可下单（missing_bar 语义；引擎现行满足），两路径对缺 bar 名的标签为 0。
**现状**：backtest.py:130-139/183-188/290 已做；basket 权重空间费用按权重×capital（ashare_execution.py:179-194）。

### F3 信号窗口与价格尾部：local_signal_range 贯穿两条路径（**缺失，p53 实现**）
**冻结**：`compare_candidate_parity` 与 `run_fixed_candidate_parity` 新增显式参数 `local_signal_range = (s0, s1)`（折内测试窗信号列区间），**同一个值**传给 sim 的 `signal_range` 与 engine 的 `signal_range`。
**现状（缺失）**：`_run_orientation`（scoring_parity.py:180-197）对 sim 与 engine 均不传——sim 默认 `(0, n−2)`（reward_baskets.py:212-213）、engine 默认 `range(n−(delay+1))`（backtest.py:222-227）。delay 默认 1（backtest.py:108）时两者恰好相同，**delay > 1 时默认分叉**——显式传递同时消除这颗地雷与"未验证的窗口"本身。
**不变量**：
- B3.1 两路径收到同一 `(s0, s1)`（harness 单次持有、两处引用同一值；测试可用 mock/桩断言两调用参数相等）。
- B3.2 sim 输出长度 = `s1 − s0` = engine 输出长度。
- B3.3 尾部无完整出口（存在 t ∈ [s0,s1) 使 `t + delay + 1 ≥ n_dates`）→ ValueError（复用引擎语义 backtest.py:228-232；消息要素：越界索引、n_dates、需要 `t+delay+1` 完整出口）。
- B3.4 非连续区间 → ValueError（backtest.py:233-236 语义）。
- B3.5 价格尾部对齐：价格矩阵列数 ≥ `s1 + delay + 1`（给最后一个信号完整出口）。

### F4 实际日期轴 + 长度/日期不一致必须拒绝（**现状 = 裁剪，p53 改拒绝**）
**冻结**：长度或日期不一致 → **拒绝**，不得取两侧最短长度裁剪后继续比较。
**现状（被禁止）**：scoring_parity.py:286-296 对 daily/fee/turnover 三族全部 `min` 截断；当前仅靠 epoch_slice 同刀切片使长度相等（test_reward.py:631 的 `len == len` 断言是外部约束，harness 本身不拒）。
**不变量**：
- B4.1 `sim_daily.size == eng_daily.size`，否则 ValueError；**消息四要素（冻结）**：①语义标识 `"length mismatch"`；②`sim={n_sim}`；③`engine={n_eng}`；④候选名/窗口标识。费用与换手序列同断言（各自的长度相等，不得复用 daily 的 n）。
- B4.2 harness 校验 `len(dates) == n_dates`（universe_mask/open 的列数）且 dates 无重复；不一致 → ValueError（消息含两侧计数）。
- B4.3 引擎回报的 positions 中 entry/exit 日期 ∈ dates 集合。
- B4.4 拒绝必须发生在**任何误差计算之前**：不得产出部分报告/部分行（p53 以"抛错后无文件写入、无行返回"断言）。

### F5 全局调仓轴 + PIT 掩码与交易阻断（部分已做；不对称如实量化）
**冻结**：
- 调仓轴：`rebalance_mask` 由**全局策略日期轴切片**产生（eval_folds.py:174 模式），两条路径同传同一对象（scoring_parity.py:188/196 现行已满足）——禁止在折内重启 5/10 日节奏。
- PIT universe_mask：两条路径同一对象（scoring_parity.py:186/195 现行已满足）。
- 阻断不对称**如实量化**（不改引擎）：basket 的 `blocked_buy` 来自 `loader.tradability_masks()`，在 exact p19 §2.4 之上**额外**叠加 `~universe_mask`（data_loader.py:910）与 adj_factor 堕名禁买（data_loader.py:911-913）；引擎在 harness 内由 raw_cache 自算（backtest.py:206-219），**无这两层**（其资格经 constructor `eligible`）。要求：报告每行新增诊断字段 `blocked_extra_buy_columns`（basket_blocked & ~engine_blocked 的列计数）——把不对称从隐性变成显性计量。
**不变量**：
- B5.1 构造无 universe 退出、无 adj_factor 堕名的面板上，两路径阻断集合恒等（`blocked_extra_buy_columns == 0`）。
- B5.2 rebalance_mask 两路径同一对象且形状 = (n_dates,)（reward_baskets.py:287-291 现行校验保留）。

### F6 同一固定方向 + 同一费用配置与资本口径（已做，冻结 + 钉测试）
**冻结**：
- 行级误差族用候选**给定** direction（scoring_parity.py:283 现行满足）；±1 只是信号乘子。`proxy_direction`/`engine_direction` 的双向探测保留，但其输出是**诊断**而非择向结果（§4 命名契约）。
- 两条路径同一 `backtest_config`：费用模型 `ExecutionCostModel.from_config(cfg)` 与 `cfg.initial_capital`（reward_baskets.py:294-295 与 backtest.py:84-86 同 cfg）；费用分母同为交易前权益（ashare_execution.py:179-194 与 backtest.py:373）。
**不变量**：
- B6.1 同 cfg 断言：两路径构建时引用同一 cfg（id 相等或 provenance 全键相等）。
- B6.2 零费配置下两路径费用序列恒为 0（逐元素）。
- B6.3 误差族不得改用择向结果的方向（冻结行为：`by_orientation[int(direction)]`）。

---

## §2 四个验收场景与期望输出（手算数字经 Python `fractions` 复核）

### S1 拆股中性（split-neutral）
**构造**：单名（或两名对照）窗口 3 天；day1 发生 1:2 拆股：adjusted_open = (20, 20, 21)、raw open = (20, 10, 10.5)、adj_factor = (1, 2, 2)。
**期望输出**：
- o2o(adjusted) = (20/20 − 1, 21/20 − 1) = **(0, +5%)**——两路径逐日收益必须等于此值（零费零整手构造下逐位一致，容差 0/1e-12）；
- 负对照：o2o(raw) = (10/20 − 1, 10.5/10 − 1) = **(−50%, +5%)**——若任一路径出现 −50% 即收益基退化（B1.3 失败）；
- 引擎 shares 在 day1 翻倍、名义不变（1000 股×20 = 2000 股×10 = 20,000 名义），净值路径 = (1.0, 1.0, 1.05)；
- **方向无翻转**：direction_flips 不含该候选（冻结方向 +1 两路径同向同值）；
- 拆股不产生虚假费用/换手：拆股日 cost 与 turnover 仅来自真实调仓（构造为不调仓则皆为 0）。

### S2 非调仓价格变化（non-rebalance price move）
**构造**：day0 调仓买入 a = (1/2, 1/2)，此后 rebalance_mask 全 False；两期收益 r₁ = (+0.02, −0.01)、r₂ = (+0.04, +0.02)；零费零整手。
**期望输出**（手算）：
```
会话1: g₁ = 0.5×0.02 + 0.5×(−0.01) = 0.010 − 0.005 = 0.005 = R_net（c=0）
       漂移 w' = (0.5×1.02/1.005, 0.5×0.99/1.005) = (102/201, 99/201) = (34/67, 33/67)
              = (0.507462686567, 0.492537313432)
会话2: g₂ = (34/67)×0.04 + (33/67)×0.02 = (1.36+0.66)/67 = 2.02/67 = 101/3350
              = 0.030149253731 = R_net
       漂移 w'' = (104/203, 99/203) = (0.512315270936, 0.487684729064)
累计净值 = 1.005 × (1+101/3350) = 1.035300（精确）
```
- 非调仓列：cost = 0、turnover = 0 **精确**（不是近似 0）；
- 两路径逐日 net 之差在零费零整手下 = 0（1e-12）——收益记在漂移账本上（basket 非到期会话 target=prev；engine else 分支 cost_fraction=0，backtest.py:387-394）。

### S3 费用融资（fee financing）
**构造**：E = 100,000；两名、价格恒 10、lot_size = 1、佣金率 0.001、无最低佣金、其余费率 0；意图目标等权（未预留则 Σa = 1）。
**期望输出**（引擎可负担循环一轮收敛，手算；basket 16-A 修复后同不变量）：
```
名义尝试 5000 股 each：名义 50,000 + 费 50 = 50,050/名 → 总 100,100 > 100,000（现金 −100 < 0，不可行）
可负担缩放 scale = 100,000/100,100 → floor(5000×scale) = 4995 股 → 名义 49,950/名、费 49.95/名
总支出 2×49,950 + 2×49.95 = 99,999.90 ≤ 100,000 → 现金 +0.10 ≥ 0 ✓
a = (0.4995, 0.4995)；c = 99.90/100,000 = 999/1,000,000 = 0.000999；现金比例 = 1×10⁻⁶
r = (0.02, 0.01)：g = 0.4995×0.03 = 0.014985
R_net = 0.014985 − 0.000999 = 6993/500,000 = 0.013986；E' = 101,398.60
w' = (0.502462558655, 0.497536455138)；w'_cash = 1/101,398.6 ≈ 9.8620691015×10⁻⁷
```
- **不变量**：现金全程 ≥ 0（两路径）；达成 Σa + c ≤ 1；不可负担时不伪造回滚（引擎 fail-closed 语义 backtest.py:364-368）；
- 误差族：两路径费用差与净收益差在整手容差内（此处 lot=1 无整手残差 → 逐位一致）。

### S4 折尾 + 1 天：拒绝（fold tail + 1 day）
**构造**：给一侧多一天数据——两种注入都必须测：(a) engine 的 raw_cache 列数比 sim 的 target/信号多 1（或 dates 多 1 个）；(b) sim 侧多 1。
**期望输出**：
- **ValueError**（非截断、非警告、非 NaN 传播）；
- 消息四要素（FROZEN B §1 B4.1）：`"length mismatch"` + `sim={n1}` + `engine={n2}` + 候选/窗口标识；日期轴注入时为 §1 B4.2 的日期计数要素；
- 在任何误差计算前抛出；不写 parity_report/不返回部分行；
- 反面断言：等长同轴的对照输入必须正常完成（证明拒绝逻辑不误伤）。

---

## §3 P45 面板重跑对照口径

### §3.1 重跑对象（冻结，与原面板逐项一致）
配置 `configs/r1_medium_a.yaml`；折 `folds[2]`（2023：train_end 2022-12-31、test_end 2023-12-31）；参考格 n10_m2500：top_n=10、buy_rank=10、sell_rank=15、cap=0.10、min_trade=2500、capital=100,000、every_10_days/horizon=10、budget 0.1/priority；候选 = MOMENTUM_20、RSQ_60、REVERSAL_60、IND_REL_RET_20 四单因子 + combo:all3（成员 MOMENTUM_20/RSQ_60/REVERSAL_60）；方向全 +1（p45 run_measurement.py:43-84 逐项对应）。

### §3.2 两遍协议（冻结）
- **第一遍（连续性控制）**：显式 `local_signal_range` 传与原默认等价的窗口。期望：若 16-B 修复未引入其他行为变化，全部数字与原 `parity_report.json` **逐候选逐族一致**。任何不一致 = 引入了预期外行为变化，必须先归因（写 measurement log）再进第二遍。
- **第二遍（折真窗口）**：`local_signal_range` = 折契约的真实测试窗信号列（`contract.test_signal_start` 起）。差异分解在第二遍上做。
- 输出目录：新目录（建议 `AlphaGPT/reports/wp16/p45_rerun/`），原 `D:/minequant/reports/p45/**` 只读（红线 1）。

### §3.3 分解维度（每候选 × 每误差族 daily_net / fee / turnover / total_net + 方向 + 排名）
- **消失（resolved）**：old > granularity bound 且 new ≤ bound（族级转绿），记录 old→new 两值；
- **仍在（remains）**：old > bound 且 new > bound，记录剩余量与比值 new/old；
- **新增（new-regression）**：new > old + 1e-12（相对）——**必须为零**；非零即修复引入了新差异，单独归因前不得宣称完成；
- 方向翻转列表 before/after；ranking spearman before/after；verdict before/after；
- 汇总三数：resolved / remains / new-regression 计数（族级）。
- 判读纪律：以上分解只回答"16-B 修复让原差异多少消失/仍在/新增"；**不回答**生产择向是否被复现（§4）。

### §3.4 原始基线数值（本会话只读解析 `D:/minequant/reports/p45/parity_report.json`，供对照列）
- verdict = **unreliable**（7 条 reason）；granularity g = 0.011080；bounds：daily_net = 2.216×10⁻³、fee = 5.540×10⁻⁴、turnover = 0.11080、total_net = 0.132960。
- 逐候选（dir = proxy/engine，netmax = daily_net_error_max）：
  - MOMENTUM_20：dir +1/−1，netmax 8.69×10⁻³，proxy/eng 累计 +0.36%/+2.45%；
  - RSQ_60：dir +1/+1，netmax 2.139×10⁻³（界内），+7.46%/+7.58%；
  - REVERSAL_60：dir −1/+1，netmax 1.035×10⁻²，−0.57%/+2.59%；
  - IND_REL_RET_20：dir −1/−1，netmax 1.526×10⁻²，−10.52%/−6.09%；
  - combo:all3：dir −1/−1，netmax 1.078×10⁻²，−3.54%/−0.51%。
- direction_flips = {MOMENTUM_20, REVERSAL_60}；ranking_spearman = 0.9；material inversion = (MOMENTUM_20, REVERSAL_60)。

---

## §4 两个诊断的命名契约

### §4.1 两个诊断（都必须保留，但语义互斥）
- **诊断一（P45 面板内）**：`proxy_direction` / `engine_direction` 的语义 = **测试窗累计净收益择向**（scoring_parity.py:271-280 现行实现：对 ± 方向各跑一条路径，取测试窗累计净收益较高者）。建议正式名：`test_window_total_return_orientation (proxy|engine)`；`direction_flips` 建议名 **方向翻转诊断（direction-flip diagnostic）**。
- **诊断二（生产择向）**：Rule A = **参考配置下的验证 reward 择向**（`candidates.py:652-685` 精确语义；字段模式与验收见 FROZEN_C.md §2）。
- 两者对比对象不同（测试窗累计净收益 vs 训练内验证窗 reward 中位数）、时间轴不同（折测试窗 vs 训练内 val 窗）、口径不同（无 clip/罚 vs clip±10 未罚 val reward 字典序）——**任何一处输出混用两者即为命名违规**。

### §4.2 披露串（冻结文本）
凡出现 `proxy_direction`/`engine_direction`/`direction_flips` 的输出（JSON、log、报告），必须携带：
> "proxy/engine_direction = 测试窗累计净收益择向（诊断），非生产 Rule A 验证 reward 择向"

### §4.3 禁用表述清单
- "生产择向路径已复现" / "production direction path reproduced"（任何语言变体）；
- 把 direction_flips 表述为"择向规则不一致/择向规则失效"（它只测两条路径测试窗收益排序的翻转，不涉及 Rule A）；
- "proxy 选择了 +1"——应说"proxy 在测试窗累计净收益上 +1 侧更高"；
- "direction validated / direction verified"；
- 任何把"累计净收益择向"与"验证 reward 择向"并列为同一规则的措辞。

### §4.4 允许的表述
- "两条路径在同一固定方向下的误差族为 X"（行级误差族，F6）；
- "方向翻转诊断在候选 C 上翻转（proxy +1 / engine −1，测试窗累计净收益口径）"。

---

## §5 红线汇总（16-B）
1. 原 p45/p46 产物只读；重跑写新目录；红证据 `reports/p53_red_evidence.txt` 只追加（逐字命令 + 退出码）。
2. p50/p51 基线、`refuse_unwired_trade_groups`、p3 parity（12 passed）与 test_reward（57 passed）不破坏。
3. 引擎语义不改；harness 侧新增参数为加法式（缺省行为向后兼容：`local_signal_range=None` 时保持现行默认路径——但面板重跑必须显式传）。
4. 命名契约（§4）为硬性验收：p53 需含对披露串存在性与禁用表述缺席的断言（对重跑输出文本）。

---
*冻结于 2026-09-23，任何 WP16 代码存在之前。勘误只可追加于文末。*
