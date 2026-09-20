# P46 方向学习与组合定义的明确化预注册契约（工作包 10）

- 状态：**APPROVED**，用户 2026-09-20 直接指令「工作包 10：明确方向学习
  与组合定义。现有方向依赖 val_reward，因此依赖组合配置。这可以是一种
  合法研究设计，但必须明确它优化的是什么。建议保留两个独立、事前固定
  的对照：当前『训练窗内按参考组合收益择向』的规则与『训练窗内按固定
  标签的 IC 择向』的规则。每个方向结果记录训练截止日、标签、缺失处理、
  选择规则和参考配置。执行矩阵内保持方向冻结；禁止逐格或根据测试收益
  重新选方向。另外先测量不同成员的有效截面数量。如果原始秩的尺度差异
  确实影响组合，再单独比较百分位秩；它属于新候选定义，不应静默替换旧
  组合。验收重点是可解释和可复现……」（批准记录回写于本文件头）。
- 起草：执行模型，分支 `codex/b04-load-cache-performance`。
- 勘察基线：p45 完成后 HEAD `01ef617`，工作树干净。

## 1 问题陈述（已实证）

1. **方向学习的优化对象未明确**：择向唯一实现
   `CandidateScorer._choose_direction` 按 `(val_reward, val_icir)` 字典序
   选择——`val_reward` 来自验证窗篮子迷你回测，随组合配置（资金、
   最低订单、预算、参考格）改变。p37 勘误 1（RSQ_60 折 2024 方向随
   参考格翻转）与 p45 实测（代理 vs 引擎方向翻转）都是该依赖的表现。
   该设计可以是合法的（它优化"参考组合下的训练窗净收益"），但其
   优化对象必须显式化且可审计。
2. **方向结果缺五要素记录**：现行 `directions_by_fold` 只记
   `direction` 与 `val_reward`；训练截止日、标签、缺失处理、选择
   规则、参考配置未随行记录，事后无法解释某方向来自哪个配置。
3. **组合成员的秩尺度未测量**：`rank_normalize_signal` 输出**原始秩
   1..N**；`combo_signal` 对各成员原始秩直接平均。若成员有效截面 N
   差异大（如 IND_REL_RET_20 依赖行业数据覆盖可能远小于
   MOMENTUM_20），不同尺度秩的平均有系统倾斜。差异是否存在、多大、
   是否改变组合选股，未测量。

## 2 冻结语义（实现必须逐条可对照）

**B1 方向规则权威**（新模块 `ashare_model/direction.py`，
`DIRECTION_RULE_VERSION = 1`，additive）：
- **规则 A `reference_return`（现行规则的形式化，语义零变化）**：
  方向 = 训练窗内验证子窗上 `(val_reward, val_icir)` 字典序择优
  ——决策**委托** `CandidateScorer`（单一实现，禁止复制第二套）；
  优化对象显式声明为"参考组合配置下的训练窗验证净收益"。输入含
  参考配置（`BacktestConfig`），provenance 记录其
  `portfolio_config_provenance`。
- **规则 B `label_ic`（新对照，纯信号规则）**：方向 = 训练窗内信号
  与**固定标签**（p19 因果前向目标，同域 horizon）逐日截面秩 IC 均
  值的符号。IC 计算复用 `diagnostics.rank_ic_stats`（单一 IC 权威）；
  IC 均值为 0 或非有限时沿用 A 的镜像稳定典范定向（首个有限非零
  观测为正）。**输入不含任何组合配置**——资金/最低订单/预算/参考格
  概念上不可影响 B。
- `decide_direction(rule, ...)` 分派两规则；镜像稳定：`−signal →
  −direction`。

**B2 五要素 provenance**（`DirectionDecision` dataclass，两规则同构）：
`candidate`、`direction`、`rule`、`train_cutoff`（训练截止日）、
`label`（标签定义：名称/基差/horizon/窗口）、`missing_handling`
（"nan-excluded+pit-universe-mask"，两规则同一约定）、
`reference_config`（仅 A，provenance dict）、`evidence`（规则专属数值：
A 记 `val_reward±`/`val_icir±`；B 记 `ic_mean`/`n_dates`）。

**B3 行级接线**（additive）：`eval_search.baseline_candidates` 每行新增
`direction_provenance`（规则 A 的 DirectionDecision 序列化）；r01/
r02/r03 行经同一调用自动继承。方向**数值不变**（决策仍来自同一
scorer 调用）——本包不改任何已测方向。

**B4 冻结与禁止事项**（既有纪律的显式声明 + 钉住）：执行矩阵内方向
逐折学习一次、跨格冻结（p37/p42 既有实现与 W7/N4 测试钉住，本契约
引用不改）；禁止逐格重选方向、禁止以测试期收益参与择向。两规则均
只消费训练窗切片（调用方切片；测试以毒化测试期验证不变性）。

**B5 有效截面与百分位秩（测量先行，条件分支）**：
- 测量：各成员（MOMENTUM_20/RSQ_60/REVERSAL_60/IND_REL_RET_20）在
  r1 medium 各折的逐日有效截面 N（有限 ∩ PIT mask），报告
  mean/median/min 与跨成员比。
- 条件分支：若成员 N 差异使组合倾斜**实际改变选股**（原始秩组合 vs
  百分位组合的逐日截面 Spearman、top_n=10 选股重叠、参考格引擎
  OOS 差异），则百分位秩组合 `combo:all3_pct` 作为**新候选臂**单独
  比较而非替换。
- 实现（additive）：`r01_baseline.percentile_normalize_signal`
  （秩 → `avg_rank / N_eff ∈ (0,1]`，尺度无关；紧邻
  `rank_normalize_signal`，不接入 `R01_COMBO_ARMS` 默认臂）。
- 静默替换禁止：`combo_signal`/`rank_normalize_signal` 语义逐位不变。

## 3 验收 oracle（RED 断言的独立来源）

1. **纯信号规则不随资金/最低订单改变**：规则 B 在（资金 1e5 vs 1e6、
   min_trade 1000 vs 5000）四种配置组合下方向逐成员相同（其接口根本
   不接受配置——测试同时钉住该签名性质：`decide_direction("label_ic",
   ...)` 无配置参数）。
2. **收益择向可解释**：规则 A 的两个决策（配置 C1/C2）各自携带
   `reference_config` 且可复现：同配置重跑同方向；若 C1/C2 方向不同，
   两条记录的 `reference_config` 能区分来源（p37 勘误 1 场景的兜底：
   任何未来翻转可归因到配置差异）。
3. **不得访问测试期**：毒化测试期（标签/信号在 ≥ 训练截止列置反号
   极值）后，A/B 决策与 evidence 逐位不变。
4. **五要素完整**：`baseline_candidates` 行的 `direction_provenance`
   含 B2 全部字段。
5. **结论边界**：本包测量结论只评价两条择向规则与秩口径；p42 的
   ICIR 加权负结果不外推为"学习权重无价值"（契约非目标 + 日志措辞
   纪律）。

## 4 版本影响表

| 版本 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `DIRECTION_RULE_VERSION` | — | **1** | 新增 additive 权威（两规则 + provenance） |
| `REWARD_VERSION` / `PROTOCOL_VERSION` / `EXECUTION_SPEC_VERSION` / `PORTFOLIO_CONSTRUCTOR_VERSION` | — | 不变 | 规则 A 决策逐位不变（委托原实现）；行级字段 additive；percentile 助手不接入默认臂 |
| r01/r02/r03 报告 schema | — | 行 +`direction_provenance`（additive） | 不触碰既有键；报告写入路径对 additive 键已宽容（rows_to_jsonable） |

## 5 范围与非目标

范围：`ashare_model/direction.py`（新）、`ashare_model/eval_search.py`
（行级 provenance）、`ashare_model/r01_baseline.py`
（percentile_normalize_signal 助手）、
`tests/test_p46_direction_provenance.py`（新，同 commit 注册分片）、
真实数据测量（成员有效截面、A/B 方向对照、原始秩 vs 百分位组合
差异）与测量日志。

非目标：搜索循环内改用规则 B 或双规则并行评分（语义变更，属对齐
裁决后续包）；`combo_signal`/`rank_normalize_signal` 语义修改；
`R01_COMBO_ARMS` 默认臂变化；p42 ICIR 负结果的重新审判（结论只评
规则与口径，不外推）；方向冻结机制的改动（既有，引用）。

## 6 预期 RED（独立 oracle——禁止从实现反推）

- **N1** 模块缺失（`ModuleNotFoundError ashare_model.direction`）：
  两规则、provenance、分派不存在。
- **N2** 规则 B 配置不变性（验收 1）。
- **N3** 规则 A 可解释性/可复现（验收 2）：同配置复现；异配置记录
  可区分（合成场景构造方向一致与翻转两种记录形态）。
- **N4** 测试期毒化不变（验收 3，两规则）。
- **N5** 行级五要素（验收 4）。
- **N6** 百分位助手：手算小例（N_eff=4 的秩 → (0.5,]区间百分位）；
  镜像稳定（−signal → −direction，两规则）。

## 7 验证矩阵

1. 聚焦：`python -m pytest -q tests/test_p46_direction_provenance.py`。
2. 相邻：`tests/test_candidates.py`、`test_eval_search 相关`（grep 定位）、
   `test_p34_r01_baseline.py`、`test_p37_r02_matrix.py`、
   `test_p42_r03_combos.py`、`test_reward.py`。
3. `python -m compileall -j 0 -q …`、
   `python scripts/check_test_shards.py --check`、`git diff --check`、
   全量 `python -m pytest -q tests -n auto`（对账口径同 p43–p45）。

## 8 停止条件

- 规则 A 委托无法在不复制 `_choose_direction` 的情况下取得方向与
  evidence → 停止回契约（宁可加只读缝隙也不复制第二实现）。
- 实测发现方向冻结在 r02/r03 有漏洞（逐格重选）→ 停止，冻结是硬
  门禁（p37/p42 契约），本包只引用不放松。
- 全量回归出现无法归因于本包的新红/新 warning → 停止诊断。
