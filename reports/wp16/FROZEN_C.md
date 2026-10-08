# WP16 FROZEN C —— 16-C 一次评分保存两个固定方向的原始证据（direction / candidates）冻结规格

冻结日期：2026-09-23（实现存在之前；冻结时点核实见 FROZEN_A.md §0.4，同一次核实覆盖本文件：无 p54 测试、无 WP16 代码）。

## §0 文档地位、真源与使用规则

### §0.1 地位
- 本文件是 16-C 的**唯一期望真源**：`tests/test_p54_wp16c_*.py` 的一切断言只能引用本文件 §编号。
- 对实现者/修复者**只读**；冲突修实现；勘误只可以附注追加于文末。
- 推导依据：用户 16-C 原文、`reports/wp16/_recon/map.md` §3/§4、本会话逐行核实的代码（§0.2）。

### §0.2 推导依据（本会话逐行核实）
- `ashare_model/candidates.py`：**469-475**（`oriented[0::2] = raw; oriented[1::2] = -raw` 交错批量，注释明言"makes mirror invariance straightforward to audit"）、476-491（一次 `reward_function(oriented, ...)` → `rewards, val_rewards, icir, val_icir = reward_out[:4]`）、496（`objectives = reward_out[4]`，[B,8]）、**521-534**（`_choose_direction(raw[k], val_windows, val_rewards[plus], val_icir[plus], val_rewards[minus], val_icir[minus], universe_mask)`；`chosen = plus if direction == 1 else minus`——**落选方向数值在 line 534 被丢弃**）、**652-685**（`_choose_direction` 全文：字典序 key + 非有限降级 −inf + 镜像稳定规范朝向扫描）。
- `ashare_model/direction.py`：144-212（`reference_return_direction`）：**194**（`chosen = scorer.score(spec, signal, ...)`——score 内部已双向择优）、**195-197**（`opposite = scorer.score(spec, -signal, ...)`——同样内部择优，拿回同一物理胜者）、**206-211**（evidence 四键 `val_reward / val_icir / val_reward_opposite / val_icir_opposite`——opposite 与本侧同值记两遍）、162-169（docstring 只对 direction 字段成立）。
- `ashare_model/eval_search.py`：163-179（`opposite_scores = scorer.score_many(specs, [-s for s in train_signals], ...)` 一整轮重复评分）、197-202（provenance `val_reward_opposite = opposite_scores[i].val_reward`——同值；Direction VALUES 不变注释在 150-154）。
- `ashare_model/reward.py`：585-643（rewards = clip(objectives[:,0] − cost_weight·mean_cost·年化, ±10)；**逐 val 窗 win_raw = active_ir − cost_weight·窗均费·年化 → clip → 窗中位数** 得 val_rewards/val_icir/val_objectives）。
- `ashare_data/config.py:222-223`（`reward_clip_low = −10.0`、`reward_clip_high = +10.0`）。
- `ashare_model/r01_baseline.py`：170-191（`percentile_normalize_signal` = rank/N_eff，注释明言刻意不进 R01_COMBO_ARMS）、194-217（`combo_signal` 内部对每成员**再跑一次** `rank_normalize_signal`，line 203-209）。
- `D:/minequant/reports/p46/run_measurement.py`（只读）：151-161（§3 百分位流水：combo_raw / member_pct / combo_pct）、124-149（§2 Rule A vs Rule B）。
- `tests/test_p46_direction_provenance.py`（只读）：68-108（N1 四键存在性——**不断言 opposite ≠ 本侧**）、196-224（N4 毒化测试：`poisoned_signal/poisoned_label[:, TRAIN_END+2:]` 缩放 −10⁹，断言两规则 direction 与 evidence 逐字段不变）、230-244（N6 百分位手算）、246+（N6 镜像稳定）。
- 侦察 map §3.4 记录的两个真实验证 reward：−1.8418（胜者侧）与 −10.0（clip 地板值，config.py:222-223）都在 `batched_basket_rewards` 的 val_rewards 数组中算出后丢失。

### §0.3 全局红线（16-C 部分）
1. **禁止再次调用会自动择优的评分器来取得反方向结果**（现状缺陷：direction.py:195-197 与 eval_search.py:163-179）。正确路径：从**同一次** `score_many` 的交错批量返回数组直接保存两侧（candidates.py:491-496 数组已在手）。
2. **证据修复与策略变更不得混在一个提交**：`_choose_direction` 的比较键、扫描序、默认 +1 一概不动（§2）；本包不改变任何 direction 输出值（eval_search.py:150-154 的"Direction VALUES are unchanged"约束保持）。
3. p46 百分位修复仅限测量脚本 + 小测试：不扩展词表、不切换组合定义（§4 栅栏）。
4. 原 `D:/minequant/reports/p46/**` 产物只读；p50/p51 基线与 p46 既有测试（N1/N4/N5/N6）不破坏。

---

## §1 双向证据的字段模式（每候选一份，全部来自同一次 score_many 批量）

> 来源约束（冻结）：所有字段取自**同一次** `score_many` 调用的返回数组（`rewards / val_rewards / icir / val_icir / objectives[B,8]`，candidates.py:491-496）与调用上下文（日期轴、窗口、配置）。禁止二次调用评分器补数（红线 1）；禁止在证据层重新计算任何 reward 数字。

设批内候选 k 的 `plus = 2k`、`minus = 2k+1`（candidates.py:473-475 交错布局）。字段模式（建议 JSON 键名；p54 按此断言存在性与类型）：

| 字段 | 来源 | 语义 |
|---|---|---|
| `val_reward_plus` / `val_reward_minus` | `val_rewards[plus]` / `val_rewards[minus]` | **未扣复杂度罚**的验证 reward（择向键用未罚值，与现行一致——罚对两方向同值不影响择向，candidates.py:652-685） |
| `val_icir_plus` / `val_icir_minus` | `val_icir[...]` | 验证 ICIR（择向次键） |
| `train_reward_plus` / `train_reward_minus` | `rewards[...]` | 训练窗 reward（两侧） |
| `train_icir_plus` / `train_icir_minus` | `icir[...]` | 训练窗 ICIR（两侧） |
| `objectives_plus` / `objectives_minus` | `objectives[2k]` / `objectives[2k+1]` | [B,8] 行整存：train active_ir / exposure / turnover / capacity + 各 val 窗中位数——**两侧费用与活动**（val_exposure / val_turnover）由此提供（现行仅存 chosen 侧，candidates.py:558-573 之外的全部丢弃） |
| `train_signal_range` | 调用参数 | (s0, s1) 列区间 |
| `train_dates` | 日期轴切片 | `dates[s0:s1]` 字符串列表（**实际训练日期**；现行只有 train_cutoff 字符串） |
| `val_windows` / `val_dates` | 调用参数 + 日期轴 | 每个 val 窗 (start, end) 与 `dates[start:end]` 列表（**实际验证日期**） |
| `train_cutoff` | 调用参数 | 既有字段保留 |
| `reward_clip` | config | `[reward_clip_low, reward_clip_high] = [−10.0, +10.0]`（config.py:222-223）；val_reward 为**逐窗 clip 后的中位数**（reward.py:634-639）——`−10.0` 是地板值，必须带 clip 披露呈现，不得当裸数 |
| `direction` | §2 规则输出 | ∈ {+1, −1}（最终施加方向） |
| `applied_signal_basis` | 派生 | `direction × raw signal`（声明经济暴露 = 方向×原信号） |
| `tie_break_reason` | §2 分支判定 | ∈ {"validation_reward", "validation_icir", "canonical_orientation"} |
| `canonical_orientation_locus` | §2 分支 3 时必填 | 规范朝向的决策定位：哪个 val 窗 (start,end) 的哪个 eligible 观测（或 "all_zero_nonfinite"） |

**两个真实数值锚（冻结）**：−1.8418 与 −10.0 都在 val_rewards 数组中算出（map §3.4）。修复后证据必须能读出**落选侧真值**（例：胜者侧 −1.8418、落选侧 −10.0 = clip 地板），不再出现 `val_reward_opposite == val_reward` 同值记两遍（direction.py:206-211 现状）。

---

## §2 选择规则（Rule A 原文语义，保持不变；正确命名）

**正式命名（冻结）**：**参考配置下的验证 reward 择向（Rule A）**。（与 FROZEN_B §4 的诊断一"测试窗累计净收益择向"互斥命名。）

**精确语义（冻结自 candidates.py:652-685 现行代码，一字不改；此处为逐句转录供测试对照）**：
```
plus_key  = (plus_reward  若非有限则 −inf, plus_icir  若非有限则 −inf)   # 字典序元组
minus_key = (minus_reward 若非有限则 −inf, minus_icir 若非有限则 −inf)
plus_key  > minus_key → +1
minus_key > plus_key  → −1
平局 → 镜像稳定规范朝向：按 val_windows 顺序，对每窗取 eligible（universe_mask 掩蔽后）信号列，
      按 values.T.ravel() 序扫描，第一个有限非零观测的符号；全零/全非有限 → +1
```
`tie_break_reason` 分支判定（冻结）：
- `"validation_reward"`：两 key 第一分量可分（含经 −inf 降级后可分，如一侧 NaN）；
- `"validation_icir"`：第一分量相等（含同为 −inf）但第二分量可分；
- `"canonical_orientation"`：两分量均相等 → 记录 `canonical_orientation_locus`。
注意：择向用**未扣复杂度罚**的 val_rewards（罚两向同值；现行即如此），证据字段与比较键同源。

**红线重申**：证据修复不得触碰上述任何比较行为；`direction.py:194-197` 的两次 `scorer.score(...)` 调用模式与 `eval_search.py:163-179` 的整轮重复评分是**被替换对象**（改为从一次批量取证），不是被保留对象。

---

## §3 三条验收不变量（p54 测试的期望）

### V1 构造两方向分数不同的例子
**构造**：训练/验证窗内信号与标签同向单调（如 target = 严格单调于 signal 的标签，或直接用与标签相关的因子）→ plus 侧 val_reward 显著大于 minus 侧（或反向构造择 −1 的例子）。
**期望**：
- 证据中 `val_reward_plus ≠ val_reward_minus`（且 `val_icir_plus ≠ val_icir_minus`）；
- `direction` = 较大 key 侧；`tie_break_reason == "validation_reward"`；
- 证据数值逐项等于批量数组元素：测试可**独立调用 reward_function**（不经 score_many 的选择层）得到同形状数组，断言 `evidence.val_reward_plus == arr[2k]` 等（防证据层造假或二次评分漂移）；
- 落选侧真值出现在证据中（不再与胜者侧同值）。

### V2 信号取反后两侧证据交换、最终施加的经济方向不变
**构造**：同一 spec 分别以 `s` 与 `−s` 送入同一条评分批量路径。
**期望**（逐字段）：
```
evidence_plus(s)  == evidence_minus(−s)      （train/val reward、icir、objectives、日期等全部字段）
evidence_minus(s) == evidence_plus(−s)
direction(−s)     == −direction(s)
applied = direction(s)·s  与  direction(−s)·(−s)  相同 → 施加的经济暴露不变
```
镜像不变性由交错批量结构保证（candidates.py:469-475 注释明言），p54 把它从注释升为断言。

### V3 毒化测试期不能改变训练决策
**构造**：折测试窗（train_end 之后）数据任意篡改（沿用 N4 手法：`poisoned[:, TRAIN_END+2:] *= −10⁹`，含置 NaN/极端值变体）。
**期望**：
- 训练侧全部证据字段（train_*、val_*、objectives、train/val dates、direction、tie_break_reason、canonical_orientation_locus）**逐字节不变**；
- 既有锚：`tests/test_p46_direction_provenance.py:196-224`（N4 现断言 direction 与 evidence 四键不变）；p54 扩展到 §1 全字段模式。
- 既有 N1（line 68-108，四键存在性）不改动仍绿——它不断言 opposite ≠ 本侧，修复后自然通过。

---

## §4 p46 百分位修复的期望行为与范围栅栏

### §4.1 事实（已定性，冻结为已知事实）
`percentile_normalize_signal`（r01_baseline.py:170-191，每列 rank/N_eff ∈ (0,1]）之后再喂 `combo_signal`（r01_baseline.py:194-217）时，combo 内部对每个成员**再跑一次** `rank_normalize_signal`（line 203-209）→ 秩变换消去任何列内严格单调尺度：
```
rank(rank(s)/N_eff) ≡ rank(s)     （同列 N_eff 为正常数）
```
实测确证（direction_measurement.json 只读，map §4）：5 折全部 rho = 1.0、top10_overlap = 1.0、engine_total_raw == engine_total_pct（位相同）——百分位尺度被完全消去，用户定性成立。

### §4.2 期望行为（修复后）
1. **测量脚本如实呈现等价性由构造保证**：p46 §3 面板（run_measurement.py:151-161 对应逻辑的新版本脚本）将 combo_raw 与 combo_pct 的逐列恒等报告为**结构性恒等**而非经验发现；输出文本必须含原因说明（"combo_signal 内部对成员再排名，消去任何列内严格单调尺度，包括百分位"）。
2. **新小测试钉住恒等式**（现行无任何测试钉"percentile 经 combo 被消去"，map §4 明言此空位）：对任意列内严格单调的成员变换 T（至少覆盖 `percentile_normalize_signal` 与一个任意仿射例子）：
```
combo_signal([(d, T(s))], mask) == combo_signal([(d, s)], mask)      （同 universe_mask；容差 0/1e-12，逐列）
```
多成员变体：成员各自做列内严格单调变换后 combo 不变。
3. **不修 combo_signal 语义**：它是 rank 平均的定义（p34 §3.2 冻结），`percentile_normalize_signal` 的定义与其"刻意不进 R01_COMBO_ARMS"的设计（r01_baseline.py:179-182）原样保留。

### §4.3 范围栅栏（红线）
- 只允许：新测量脚本（原 `D:/minequant/reports/p46/run_measurement.py` 只读；新脚本写新目录，建议 `AlphaGPT/reports/wp16/p46_fix/`）+ `tests/test_p54_wp16c_*.py` 小测试。
- **不**扩展词表；**不**切换组合定义；`R01_COMBO_ARMS` 不动；`r01_baseline.py` 本体不改；combo 语义测试（test_p34_r01_baseline.py / test_p42_r03_combos.py）与 p46 既有测试（N1/N4/N5/N6）不破坏。
- 百分位对照的正确呈现方式是"披露 + 钉恒等式"，**不是**给 combo 加免排名通道（那是切换组合定义，越栅栏）。

---

## §5 红线汇总（16-C）
1. 红线 1（禁止二次调用择优评分器取反方向）与红线 2（证据修复 ≠ 策略变更，不混提交）见 §0.3。
2. `direction.py` 的 `DirectionDecision.evidence` 键扩展为 §1 模式时，`to_json` 输出向后兼容（既有四键保留，新增键叠加）——p46 N1 断言四键存在（line 106-108），不得改名或移除。
3. 原 p45/p46 产物只读；红证据 `reports/p54_red_evidence.txt` 只追加。
4. 16-A 改变 val_reward 数值本身（净收益恒等式修复）属预期——16-C 证据层只保证"数组里的数 = 证据里的数"，不锁数值大小；跨包回归由 p52/p53/p54 各自套件负责。

---
*冻结于 2026-09-23，任何 WP16 代码存在之前。勘误只可追加于文末。*
