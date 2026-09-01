# P11 Reward v15 区分度重构预注册契约（A 线：reward v14→v15）

- 状态：**APPROVED — 契约门禁双轮评审（t8 首轮 → t39 修复 → t40 复审 pass）**。批准后本文保持前瞻性，禁止用实现或
  测量输出反向修改；测量结果只能执行预注册裁决。
- 起草：contract-a（AgentTeams alphagpt-p0-p1 t5），2026-09-02
- 证据基线：main @ `fde9f8b`（阶段 0 清障后，t4 审核通过）；P10 契约
  （`docs/p10_searcher_fairness_contract.md`，APPROVED）与 P10 测量日志
  （`docs/p10_measurement_log.md`）；P10 综合报告
  （`docs/p10_final_synthesis_report.md`）；原始搜索产物
  `data/p10_searcher_comparison/campaign.json`（git 未跟踪，本契约起草时逐行
  提取复核）；代码版本集合经 `git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"`
  在 main @ fde9f8b 全量检索（§6）。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。任务路由：**研究语义变更**
  （AGENTS §3.2：Reward 与候选过滤）。仲裁顺序：本文/批准 → 契约测试 →
  实现 → 测量日志。§11.1：本文只写计划；实际测量只能运行后被测实现写入
  `docs/p11_measurement_log.md`（evidence-only），禁止预写结果。

## 1. 问题陈述

### 1.1 结构性缺陷：复杂度惩罚与 clip 的乘积封顶使组合公式永不可能胜出裸因子

v13/v14 reward 的主项是年化组合 active IR（对等权基准的 gross basket 超额收益，
effective-n 收缩），减去精确年化执行成本，最后截断到
`[reward_clip_low, reward_clip_high] = [-1.0, +1.0]`
（`ashare_model/reward.py` L792/L897/L943；`ashare_data/config.py` L121-122）。
候选评分器随后对每个候选扣减
`complexity_penalty × complexity_bill(ast)`（`ashare_model/candidates.py`
L525、L540/L542；默认 `complexity_penalty = 0.02`，`ashare_data/config.py`
L133）。复杂度账单 `complexity_bill >= 1.0`，裸因子恒等于恰好 1.0
（`ashare_model/complexity.py` L137-154）。

因为扣减发生在 clip **之后**，任何公式的 reward 上限为：

```
ceiling(bill) = reward_clip_high − complexity_penalty × bill
```

| 公式形态 | bill | v14 默认配置下的 reward 上限 |
|---|---|---|
| 裸因子（单特征） | 1.0 | 1.0 − 0.02 = **0.98** |
| 两层组合（bill≥2，如三层算术栈） | ≥ 2.0 | ≤ 1.0 − 0.04 = **0.96** |
| 典型 12 节点窗口化"怪物" | ~5.05（P10 精英档案实测最大值） | ≤ 1.0 − 0.101 = **0.899** |
| max_complexity 边界 | 25.0 | ≤ 1.0 − 0.50 = **0.50** |

即：**只要存在一个 raw active IR ≥ 1.0 的裸因子，任何 bill>1 的组合公式无论
raw active IR 多高，reward 都严格低于该裸因子**——这是由公式结构（而非数据）
保证的永久性赤字 0.02×(bill−1)。搜索因此在结构上被锁死在"裸因子或
恰好 bill=1 的公式"上：组合公式的额外预测力在奖励维度上不可表达。

### 1.2 缺陷的实测证据（P10 四搜索器公平对比，全部可复核）

1. **0.98 精确平台，12/12 行**：P10 Stage A 全部 12 行（4 搜索器 × seeds
   42/7/2024，B=2000）的 best-so-far 终点 reward 均为精确 0.98
   （`docs/p10_measurement_log.md` L47；`docs/p10_searcher_comparison_20260901/summary.json`
   12 行 `final_best_so_far` 逐行 [·, 0.98]）。0.98 = clip_high −
   0.02×1.0，且只有 bill=1.0 的公式可能达到——即每行的搜索最优都是
   触顶裸因子。
2. **9/9 精英榜首触顶**：campaign.json 中 9 个基线臂精英档案
   （gp/tpe/random × 3 seeds；RL 臂为 random-init 不建档案）的榜首条目
   val_reward 全部 = 0.98 精确值，且榜首全部为裸因子（bill=1.0，如 tpe:seed42
   档案顶部 MARGIN_BALANCE_CHG / MOMENTUM_20 / IND_REL_RET_20 / MACD_DIF，
   complexity_cost=1.0）。精英池的头部被触顶裸因子整体占据。
3. **预算 0.5–35% 即达顶**：逐行提取 best-so-far 曲线首次到达 0.98 的坐标
   （B=2000 的 %）：42:gp 2.0%、42:tpe 3.6%、42:random 14.9%、42:rl 4.8%、
   7:gp 2.8%、7:tpe **0.5%**、7:random 1.0%、7:rl 3.8%、2024:gp 4.2%、
   2024:tpe 16.7%、2024:random **35.0%**、2024:rl 29.9%。区分度被压缩到
   预算的前 0.5–35%，此后曲线全程平坦。
4. **四臂 area 趋同 <0.4%**：四臂 best-so-far area 中位数 1951.6–1959.5，
   极差 <0.4%（`docs/p10_measurement_log.md` L155）——奖励函数无法区分搜索器
   质量，P10 §8 裁决因此无法在臂间产生有效区分。
5. **组合活动门全面近静态**：12 条选中公式在 fold-0 测试窗全部为 2–6 次实际
   调仓/年、21–33 笔订单、5940–9064 笔被抑制交易、日均换手 0.44–0.68%
   （`docs/p10_measurement_log.md` §11 表与 L139）。近静态书在日调仓协议下
   付近零成本，"低成本达标"不可解读为有效（AGENTS §8.5）。

### 1.3 方向背书（预注册级出处）

P10 契约 §8 预注册裁决表行「无任何臂 backtest_admissible」
（`docs/p10_searcher_fairness_contract.md` §8，L237）与测量日志的原文执行记录
（`docs/p10_measurement_log.md` **L153-154**：L153 = 裁决 headline
`negative_no_admissible_formula`；L154 = §8 裁决原文执行——"方向转向评价器/
Reward 与成本-换手结构（引用 t2 成本证据），不归咎某一搜索器"）构成本契约的
预注册级背书；P10 综合报告 §3.3 结构性发现 1
（"0.98 平台饱和表明 reward 缺乏区分度，是首要结构性瓶颈"）与 §4 立项建议 1/2
同向。本文即该改进方向的 A 线落地契约。

## 2. 假设

- **H1（clip 尺度错配）**：年化 active IR 的自然量纲远大于 ±1.0（日频
  shrunk ICIR 0.063 即年化触顶；P10 因子诊断中可用族的日频 ICIR ≥ 0.11，年化
  ≈ 1.75+），默认 clip 带宽 [-1, +1] 对该统计量过窄，是 0.98 平台的第一
  直接原因。扩大 clip 带宽可恢复 reward 对信号质量的连续区分。
  （先例：`tests/test_reward.py` L375-383 为使成本分解可见即用 ±10 宽带——
  仓库自身已承认默认带会饱和。）
- **H2（惩罚恒偏最简）**：从 bill=1 起线性递增的复杂度惩罚，在 clip 触顶域内
  退化为"对一切组合的结构性封顶"；改为非单调（低复杂度区免罚、高复杂度区
  陡峭递增）可保留抗 bloat 功能，同时解除对中等复杂度组合的压制。
- **H3（区分度可恢复）**：H1+H2 同时修正后，best-so-far 曲线将不再早期精确
  平台化，四臂 area 差异恢复到可分辨水平。若测量否定 H3，则按 §9 裁决记录
  负结果并转向评价目标函数本身（非 clip 问题）。
- 明确的非假设：本契约**不**假设 v15 能改变 OOS 回测硬门槛的通过率；搜索期
  区分度与 OOS 表现的关联性不在本契约裁决范围（P10 §8 已将其列为"奖励-未来
  表现错位"疑点，属后续契约）。

## 3. 范围与非目标

### 范围（A 线唯一改动面）

1. `ashare_model/reward.py`：`REWARD_VERSION "14" → "15"`；模块版本历史追加
   v15 条目；clip 语义保持"同一 band 作用于 train/val 全部 reward 路径"，
   仅默认带宽数值变化（§5.1）。
2. `ashare_data/config.py` + `config/ashare_config.yaml`：RewardConfig 默认值
   变更（§5.4 参数表）与注释同步（同一原子提交）。
3. `ashare_model/candidates.py`（或其唯一调用的共享 helper）：复杂度惩罚函数
   形状改为 §5.2 的非单调两段式；**惩罚计算必须保持单一实现路径**
   （现路径即 `CandidateScorer.complexity_penalty`，不得出现第二套）。
4. 测试：§7 RED 清单的新增测试 + §6.3 白名单 pin/期望更新。
5. `docs/p11_reward_v15_contract.md`（本文）与（运行后）
   `docs/p11_measurement_log.md`。

### 非目标

- 不改 active IR / ICIR / IC 的统计定义与 HAC 收缩（signal_quality 不动）；
- 不改成本模型、组合构造、执行、费用、`cost_weight=1.0` 的诚实成本语义；
- 不改 `complexity_bill` 本身（`ashare_model/complexity.py` 四轴账单不动）；
- 不改搜索器实现、词表 v5、grammar、max_formula_len、预算计数、seed pairing
  （P10 的 SEARCH_CONTRACT 3 空间原样沿用）；
- 不改晋级门禁阈值与 `promotion.py`（晋级侧执法属 B 线 p12 辖区；§5.3 只落
  评价端）；
- 不切换生产默认搜索器（gp 不变）；不做 paper/sim、lifecycle 状态转换；
- 不改 `min_val_reward`/`min_val_icir`/质量门/容量门的语义与数值；
- 不动 P10 及更早的历史产物（不重写、不重跑、不拼接）。

## 4. 不变量（实施全程必须成立）

1. **单一语义路径**：clip 只存在于 reward.py 的三条公式化路径
   （formula_reward L792 / batched L897 / val-window L943）；惩罚只在
   CandidateScorer 一处扣减；禁止出现第二套 clip 或惩罚实现（AGENTS §1.3、§9）。
2. **bad_reward < reward_clip_low** 恒成立（无效/常数公式的哨兵值必须仍在
   clip 带之外可分辨；`ashare_data/config.py` L123-124 既有约束），§7 RED-5
   固化。
3. **方向与 PIT 语义零改动**：universe_mask 在信号日与入场日双消费、direction
   学习、tie-break、validation 窗口中位数、rebalance_mask 全局调度、精确成本
   扣除全部原样。
4. **v15 内部一致性**：scalar 与 batched 路径 reward 相等的既有测试继续成立；
   val 窗口仍逐窗独立 clip 后取中位数。
5. **max_complexity 硬拒绝门不变**（25.0，`complexity_above_maximum`）。
6. **语义缓存跨版本隔离**：SemanticCacheKey 已含 reward_version
   （`ashare_model/semantic_cache.py` L72）与 complexity_bill
   （L206-209），v15 评审不得复用任何 v14 缓存条目；SEMANTIC_CACHE_VERSION
   本身不需 bump（键已隔离，§6）。
7. **P11 测量的 §7 公平要素**（§8.2）全要素保持；跨版本（v14 vs v15）结果
   禁止宣称 matched comparison。

## 5. v15 设计（预注册语义与常数）

实现必须原样使用本节数值；任何调整必须先修订本契约并重新批准。

### 5.1 clip 带宽重推（H1）

- 默认带：`reward_clip_low: -1.0 → -10.0`，`reward_clip_high: 1.0 → 10.0`
  （配置键与 clip 机制不变，仅默认值变化；YAML 与 dataclass 默认同 commit
  同步）。
- 理由：±1.0 对年化 active IR 过窄（H1）；±10 = 10 倍带宽，对应日频 shrunk
  ICIR ≈ 0.63 的饱和线，高于可用因子族的合理区间（日频 ICIR 0.11–0.4），
  使"触顶"重新成为罕见事件而非普遍状态；保留上界而非删除，是因为
  `robust_icir` 对完美稳定序列的上界为 1e9（reward.py L288/L652），无 clip
  时单个退化高 IR 公式可产生天文数值并破坏 RL advantage 归一化的数值环境
  （`model.advantage_clip` 的 std 归一化可吸收尺度，但不能吸收无穷）。
- 哨兵值联动：`bad_reward: -2.0 → -20.0`（保持 bad_reward < clip_low 的
  既有约束与"无效公式可分辨"语义，不变量 2）。

### 5.2 复杂度惩罚非单调化（H2）

- 现行（v14）：`penalty(bill) = 0.02 × bill`，自 bill=1 起线性。
- v15（预注册两段式，非单调——低段平坦、高段线性递增）：

```
complexity_free_bill = 3.0   # 新增 RewardConfig 字段
complexity_penalty   = 0.05  # 语义重定义为"超量斜率"（原 0.02 为全量斜率）
penalty(bill) = 0.0                          若 bill <= complexity_free_bill
             = complexity_penalty × (bill − complexity_free_bill)   否则
```

- 理由与效果核对：complexity.py 自身的标定（L84-87）"典型 3 节点窗口化公式
  bill ≈ 1；12 节点 MA60 怪物 bill ≈ 5"，P10 精英档案实测 bill ∈ [1.0, 5.05]。
  免罚区 3.0 覆盖全部典型中小公式（裸因子与 bill≤3 组合同顶，结构性赤字
  归零）；高复杂度区间压力不低于现状且陡峭化：bill ≥ 5 后新惩罚 ≥ 旧值
  （bill=5 时 0.10 与旧值相等，bill=25 时 1.10 对旧 0.50）；**bill ∈ (3,5)
  区间新惩罚低于现状**（如 bill=4：0.05 对 0.08），连同免罚区属**有意放宽**
  ——目的正是解除对中等复杂度组合的结构性压制（§1.1 缺陷）。最大惩罚 1.10
  相对带宽 20 仅 5.5%，组合公式的封顶天花板不再低于裸因子的可达上限。
  历史"裸因子微弱倾向"（0.02×1.0）随之归零，这正是被修复缺陷的组成部分，
  作为有意语义变化在此明示。
- `complexity_penalty` 字段语义重定义必须在 config.py docstring 与
  ashare_config.yaml 注释中同 commit 说明（旧行为/新行为/版本），且必须
  显式披露两处放宽：免罚区 bill ≤ 3.0 与 bill ∈ (3,5) 区间惩罚低于现状
  （含 bill=4 时 0.05 对 0.08 的实例），不得只写"高区陡峭化"而掩盖
  中低区放宽。

### 5.3 组合活动门与成本-换手的评价端落点（方向 3）

- **成本-换手已在评价内**：v13 起 reward 即精确扣除年化执行成本
  （`cost_weight=1.0`），本契约不改。近静态书低成本的"虚假优势"在 reward
  维度已被诚实计价；P10 揭示的残余问题是**披露与裁决层面**的活动维度。
- **评价端落点（本契约范围内）**：P11 测量（§8）沿用 P10 t8 已 additive 落地
  的组合活动门字段（`rebalance_count`/`order_count`/`suppressed_trade_count`/
  `average_turnover`，`ashare_model/backtest.py` L295-319、
  `ashare_model/eval_metrics.py` L182-184，单一语义路径），并把活动门披露写入
  预注册裁决：任何"成功"结论必须同时报告活动门四字段；近静态行（§8.5 操作
  化定义）不得单独支撑成功声明（AGENTS §8.5）。
- **晋级门槛端 = 非目标**：把活动门升级为晋级硬门槛属 promotion 门禁变化，
  归 B 线 `docs/p12_promotion_enforcement_contract.md` 辖区；本文显式不做
  （§12 文件所有权，防止并行线双改 `promotion.py`）。
- **不在 v15 reward 内新增活动惩罚项**：若 H3 被测量否定，"在目标函数内加入
  活动项"属 §9 预注册的 v16 方向，不在本契约内追加（防范围蔓延与事后调参）。

### 5.4 v15 参数汇总（预注册常数）

| 参数 | v14 | v15 | 所在 |
|---|---|---|---|
| REWARD_VERSION | "14" | **"15"** | ashare_model/reward.py L141 |
| reward_clip_low | -1.0 | **-10.0** | RewardConfig + ashare_config.yaml |
| reward_clip_high | 1.0 | **10.0** | RewardConfig + ashare_config.yaml |
| bad_reward | -2.0 | **-20.0** | RewardConfig + ashare_config.yaml |
| complexity_penalty（语义：超量斜率） | 0.02（全量斜率） | **0.05** | RewardConfig + ashare_config.yaml |
| complexity_free_bill（新增） | — | **3.0** | RewardConfig + ashare_config.yaml |
| max_complexity / cost_weight / min_val_reward / min_val_icir / 质量门 | 不变 | 不变 | — |

## 6. 版本影响总表与迁移/拒绝策略

### 6.1 全量版本所有者检索（AGENTS §3.2 要求的 `git grep`，main @ fde9f8b 实测）

`git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"` 共 **33** 处命中 =
**31 个赋值所有者** + 2 处 docstring 引用（`alphagpt.py:8` 与
`tests/test_searcher_bench.py:4`，非赋值所有者，不计入下表）。31 个赋值
所有者逐项旧值/新值/bump 理由：

| 版本所有者（文件:行） | 旧值 | 新值 | bump 理由 |
|---|---|---|---|
| ashare_model/reward.py:141 REWARD_VERSION | "14" | **"15"** | reward 评分语义变化（band + 惩罚形状 + 哨兵联动）；v14/v15 reward 不可比 |
| ashare_data/manifest.py:54 MANIFEST_VERSION | "1" | 不变 | manifest 结构零改动 |
| ashare_model/admission.py:18 ADMISSION_RULE_VERSION | 2 | 不变 | 准入规则零改动（且本契约不产生 admission 决定） |
| ashare_model/alphagpt.py:33 MODEL_VERSION | 3 | 不变 | 模型结构/初始化零改动 |
| ashare_model/artifact_schemas.py:44 ARTIFACT_SCHEMA_VERSION | 2 | 不变 | reward_version 字段已存在（L280/L313，str），仅取值变化，无 schema 变化 |
| ashare_model/artifact_schemas.py:47 LEGACY_SCHEMA_VERSIONS | (1,) | 不变 | legacy 集合不变 |
| ashare_model/bare_factor_backtest.py:56 BARE_FACTOR_BACKTEST_VERSION | 3 | 不变 | bare_factor 产物 stamping 复用 reward_version 字段（L176），自动随 15 分类，无独立 schema 变化 |
| ashare_model/cost_matrix.py:47 FEE_MATRIX_VERSION | 1 | 不变 | 费率矩阵零改动 |
| ashare_model/data_tier.py:37 DATA_TIER_VERSION | 1 | 不变 | 数据分层零改动 |
| ashare_model/elite_archive.py:15 ELITE_ARCHIVE_VERSION | 1 | 不变 | 档案 schema 零改动（条目值为新 reward 产出，属运行时数据） |
| ashare_model/evaluation.py:287 PROTOCOL_VERSION | "25" | 不变 | 评价协议（fold/指标/校正）零改动；protocol 产物经 reward_version 字段自动随 15 分类（artifact_versions L191-196） |
| ashare_model/factors.py:46 FACTOR_COMPUTE_VERSION | 1 | 不变 | 因子计算零改动 |
| ashare_model/feature_registry.py:57 FEATURE_REGISTRY_VERSION | 5 | 不变 | 词表零改动 |
| ashare_model/imitation.py:12 IMITATION_VERSION | 1 | 不变 | 模仿学习零改动 |
| ashare_model/ledger.py:47 LEDGER_SCHEMA_VERSION | 1 | 不变 | ledger schema 零改动 |
| ashare_model/p3_measurement.py:50 P3_MEASUREMENT_VERSION | 3 | 不变 | 测量模块零改动 |
| ashare_model/research_domain.py:39 RESEARCH_DOMAIN_VERSION | 2 | 不变 | 研究域零改动 |
| ashare_model/rl_diagnostics.py:12 RL_DIAGNOSTICS_VERSION | 1 | 不变 | RL 诊断零改动 |
| ashare_model/run_store.py:49 INDEX_SCHEMA_VERSION | 1 | 不变 | run 索引 schema 零改动 |
| ashare_model/runspec.py:40 RUNSPEC_SCHEMA_VERSION | 1 | 不变 | RunSpec schema 零改动；reward_version 身份字段（L57 映射）自动取新值 |
| ashare_model/search_contract.py:24 SEARCH_CONTRACT_VERSION | 3 | 不变 | 搜索空间契约零改动（P10 版 GP node cap 原样） |
| ashare_model/searcher_adjudication.py:38 P10_ADJUDICATION_VERSION | 1 | 不变 | P10 裁决器不动；P11 测量如需扩展按 §8 用新 run 目录与 evidence 记录，不改 P10 产物 |
| ashare_model/searcher_bench.py:72 SEARCHER_BENCH_VERSION | 3 | 不变 | bench 运行器零改动（P11 复刻运行不改其代码；如复刻需要参数化缺口，停下修订本契约而非悄悄改 bench） |
| ashare_model/semantic_cache.py:40 SEMANTIC_CACHE_VERSION | 1 | 不变 | 缓存键已含 reward_version（L72）+ complexity_bill（L206-209），跨版本天然隔离 |
| ashare_model/targets.py:12 TARGET_CONTRACT_VERSION | 1 | 不变 | 标签契约零改动 |
| ashare_model/tier_reports.py:35 TIER_REPORT_VERSION | 1 | 不变 | 报告零改动 |
| ashare_model/vocab.py:226 GRAMMAR_VERSION | 5 | 不变 | 语法零改动 |
| ashare_portfolio/constructor.py:23 PORTFOLIO_CONSTRUCTOR_VERSION | 1 | 不变 | 组合构造零改动 |
| ashare_portfolio/execution_spec.py:22 EXECUTION_SPEC_VERSION | 2 | 不变 | 执行语义零改动 |
| ashare_portfolio/optimizer.py:41 PORTFOLIO_OPTIMIZER_VERSION | 1 | 不变 | 优化器零改动 |
| ashare_portfolio/rebalance.py:12 REBALANCE_POLICY_VERSION | 2 | 不变 | 调仓策略零改动 |

明确不 bump 的关键判定：`PROTOCOL_VERSION`/`ARTIFACT_SCHEMA_VERSION` 不动，
因为二者都是"记录 reward_version"而不是"定义 reward 语义"；既有
artifact-facing 兼容性分类权威 `ashare_model/artifact_versions.py` 已把
`reward_version` 作为分类字段（L138-140/L191-196/L327），本契约不新增任何
artifact-facing 版本，故无需新接入。

### 6.2 v14 产物分类：legacy（迁移/拒绝策略）

- **分类（自动，既有权威）**：REWARD_VERSION 15 生效后，所有记录
  `reward_version="14"`（或缺失）的 strategy / protocol / bare_factor 产物由
  `artifact_versions.classify_*` 判 legacy，reason 如
  `reward_version 14 != current 15`；`stamp_legacy_artifacts` 幂等打标
  （legacy / legacy_reason / legacy_stamped_at）。**零数据迁移**：旧产物不
  转换、不删除、只读保留（AGENTS §4.3 legacy 只读展示与人工审计）。
- **拒绝（fail-closed，既有门禁）**：`promotion.py` L386-389 对
  reward_version 不匹配的 artifact 拒绝晋级；`research_doctor` 报告不一致；
  v14 产物不得恢复到新模拟账户、不得冒充 current champion。
- **运行隔离**：v15 训练/搜索使用新 run ID 与新输出目录
  （建议 `data/p11_searcher_comparison/<run_id>/`）；禁止覆盖或追加任何 P10
  产物；禁止把 v14 与 v15 的 reward/best_reward/area 拼接比较
  （AGENTS §4.3、§7）。
- **语义缓存**：v15 键（reward_version=15）与 v14 键天然隔离（不变量 6），
  无清理动作。

### 6.3 受影响既有测试（§10.1 白名单情形 2 预告，实现 commit 逐项举证）

- `tests/test_artifact_versions.py` L164：`assert REWARD_VERSION == "14"` →
  "15"（版本 pin）；
- `tests/test_artifact_schemas.py` L191 等：示例 reason 中 "current 14" 字面
  量按新值更新；
- `tests/test_train.py` L319/L329-339/L390-397：组合公式封顶与裸因子惩罚期望
  按本契约 §5.2 新形状更新（旧断言"组合上限 = clip_high − 0.02×bill"在 v15
  下不再成立）；
- `tests/test_signal_quality.py` L407-408：`complexity_penalty ==
  complexity_cost × complexity_penalty` 的线性关系断言按 §5.2 更新；
- 上述更新一律引用本契约 §5/§6 作为需求变更依据；断言强度不得降低。

## 7. 预期 RED 测试清单（实现前先红， impl-a 落地）

约定：RED-1..RED-4 在 v14 代码上必须稳定失败且失败原因唯一（结构性缺陷）；
出现无关失败先停下诊断（§2.2(3)）。全部新测试进入
`tests/test_reward.py` 或新文件 `tests/test_reward_v15_contract.py`（同 commit
注册进分片，`scripts/check_test_shards.py` fail-closed）。

1. **RED-0 / 缺陷存在性测试（版本条件化，v14 下 PASS——先固化缺陷）**：
   `test_structural_ceiling_tracks_version_contract`：构造 raw active IR ≥
   reward_clip_high 的裸因子（bill=1.0）与组合候选各一（成本归零构造，
   如 cost_weight=0 或等成本对）。**组合候选的 bill 必须钉住具体值**：取
   三 token 算术组合 `ADD(X, Y)`（nodes=3 / depth=2 / window=0 / opcost=1），
   先断言其 `complexity_bill == 1.70`（钉值，与"bill>1 天花板存在性"标题
   一致；且 1.0 < 1.70 ≤ complexity_free_bill=3.0，保证 v15 分支的零赤字
   断言适定）。断言二者 reward 上限之差等于按 `REWARD_VERSION` 查得的
   预注册值："14" → `0.02×(bill−1) = 0.014`，即组合上限**严格低于**裸因子；
   "15" → **0.0（同顶）——该零赤字断言显式限定于 bill ≤ complexity_free_bill
   的候选，本钉值满足**；bill > complexity_free_bill 区间的惩罚形状由
   RED-3 单独钉住，不在本测试范围。断言来源 = 本契约表格（非实现输出）。
   v14 运行即固化"bill>1 组合公式当前配置下 max reward < 裸因子"的缺陷
   事实；v15 合入后同一测试继续通过并防"语义回退未 bump"。
2. **RED-1 / 区分度恢复断言（v14 失败 → v15 通过）**：同一数据窗内构造组合
   公式 raw active IR 严格高于裸因子（或相等），断言 v15 默认配置下
   `reward(组合) >= reward(裸因子)`。v14 下因 §1.1 结构性赤字必然失败——
   其失败输出（含两组 reward 数值与 ceiling 算术）就是缺陷被修复前的
   固化证据，实现报告必须原样保留。
3. **RED-2 / 带宽重推断言（v14 失败）**：构造 raw active IR ∈ (1.0, 10.0)
   的信号（cost_weight=0 隔离），断言 reward == raw（不被截断）。v14 下
   reward == 1.0（触顶）→ 失败。
4. **RED-3 / 非单调惩罚断言（v14 失败）**：经 CandidateScorer 公共路径（或其
   唯一惩罚函数）断言：penalty(1.0)=0、penalty(3.0)=0、penalty(5.0)=0.10、
   penalty(25.0)=1.10。v14 线性 0.02×bill → 失败。
5. **RED-4 / 配置不变量**：`bad_reward < reward_clip_low`；`penalty(bill)` 在
   `complexity_free_bill` 处连续（右极限 = 0）；bill 单调不减性仅在
   bill > free 区成立（低段平坦属预期，断言形状而非误断单调）。
6. **版本 pin 与既有期望更新**（§6.3 白名单情形 2）：REWARD_VERSION pin 14→15；
   test_train/test_signal_quality 期望更新。实现前 RED 阶段先记录旧断言的
   失败/失效证据，再随实现 commit 更新。

## 8. 测量方案（P11 = P10 四搜索器复刻，v15 裁决）

### 8.1 设计（原样复用 P10 契约 §4 常量，不 CLI 放水）

- B=2000 唯一语义评价/行；paired seeds [42, 7, 2024]；fold 0
  （train_end 2020-12-31 → test_end 2021-12-31）；window cap (300, 400)；
  RL split 8×250（random-init）；非 RL 2000×1；GP node cap 11（P10 §4.3 版）；
  行序 seed 升序 × (gp, tpe, random, rl)。
- 配置单一来源 `config/ashare_config.yaml`（v15 reward 段生效）+
  `runtime_overrides.yaml`；逐行记录 effective config hash。
- 12 行新 run 目录（`data/p11_searcher_comparison/<run_id>/`），campaign JSON +
  append-only ledger（P10 §7 机制原样）。
- Stage B/C：12 条 selected 公式经 `evaluation.evaluate_formula` 单一评价路径
  → fold-0 测试窗 → 引擎全成本指标 + P10 硬门槛四条（年化>10%/回撤<15%/
  Sharpe≥1.0/Calmar≥1.0）+ 组合活动门四字段逐行披露。
- 数据身份：逐行记录 dataset_id（P1-5 daily_bar 同步后 dataset_id 允许不同于
  P10 的 a839ecf2…；**12 行必须同行内一致**，行间漂移 fail-closed 停机）。

### 8.2 §7 公平要素全要素核对表（逐项预注册）

同一数据/fold/domain/候选评分器；同一有效词表与语义规范化（SEMANTIC_CACHE
单一权威）；同一最大有效公式长度（含 EOS 口径）；同一唯一语义评价预算与
计数器（invalid/duplicate 不计费）；预注册 paired seeds；同一
execution/portfolio/Reward 配置（v15 四臂共享）；同一设备。唯一自由变量 =
searcher 身份与 paired seed。跨版本对照（P10 v14 vs P11 v15）仅描述性，
**不宣称 matched**。

### 8.3 每行必录字段

P10 §4.4 全集原样（requested/consumed、proposal/invalid/sem-dup 计数、终止与
停滞原因、best-so-far 曲线（累计唯一评价数坐标）、canonical 长度直方图、
selected 全字段、wall/RSS/device、版本集合、dataset_id、window/fold、seed、
config hash）+ 本契约新增：**首达最大值的预算坐标**（first-reach-max
coordinate，§9 判据输入）与 selected 的活动门四字段。

### 8.4 资源推导

P10 校准（附录 A：gp 0.657 / tpe 1.012 / random 0.562 / rl 0.676 s/eval，
CPU）继续适用（评价器统计与执行语义零改动，band/惩罚不改变单评价耗时量
级）；Stage A ≈ 5.1h、B/C ≤ 1h、全程 ≤ 7h 的投影沿用 P10 §4.5。运行时以
首个 seed 块实测重算投影并执行 2× 偏差熔断（§10）。

## 9. 预注册成功判据与裁决规则（允许负结果）

### 9.1 判据（运行前固定，可证伪）

- **S1（结构性，测试级，实现 commit 上判定）**：RED-0 在 REWARD_VERSION="15"
  下通过（bill≤3 组合与裸因子同顶，0.02×(bill−1) 赤字不存在）。
- **S2（测量级，平台化消失）**：P11 12 行 best-so-far 终点取值**不同值数 ≥ 2**
  （P10 基线：12/12 同 = 0.98），**或**存在至少 1 行首达最大值坐标
  > 35%×B（P10 基线：全部 ≤35%，最小 0.5%）。
- **S3（辅助记录，不作门槛）**：四臂 area 中位数极差对照 P10 的 <0.4% 做描述
  性记录（n=3，跨版本不可比，禁止显著性声明）。
- **负结果合法**：S2 不成立即为合法负结果，不得事后放宽阈值、更换统计口径或
  只展示有利 seed（AGENTS §1.1）。

### 9.2 裁决映射（运行后逐条执行，禁止事后改写）

| 观测 | 裁决 |
|---|---|
| S1 通过 且 S2 成立 | v15 区分度修复按预注册判据判定**有效**；P11 记录为正面结果；四臂 matched 结论仅限本次运行；流程进入团队计划的正式测量阶段（t32/t34） |
| S1 通过 且 S2 不成立（12/12 仍同值平台且首达全部 ≤35%） | **负结果**：clip/复杂度惩罚不是区分度瓶颈 → 转向评价目标函数本身（active IR 构造、风险调整、活动项），另立 v16 契约；v15 代码是否保留由工程评审独立决定（结构性上限消除是缺陷修复，与研究裁决解耦），本契约记录该负结果 |
| 实现期 S1 无法通过 | 实现缺陷：修复后重验；禁止修改 §9.1 判据或常数 |
| 熔断/资源超限（§10） | campaign = FAILED，保留已有行，修订本契约后方可重跑；不得静默截断后宣称 matched |

## 10. 资源上限与停止条件

1. **墙钟**：Stage A ≤ 7h（仅搜索墙钟计入，campaign 级行间核算）；Stage B/C
   ≤ 1h。累计 + 下一行投影超限 → 停止启动新行，未运行行记 `not_run`，
   campaign = FAILED（不伪装 matched）。
2. **偏差熔断**：首个 seed 块 4 行完成后，块均墙钟 > 校准投影 2× → 停止后续
   seed，campaign = FAILED（P10 §4.5 机制原样）。
3. **内存**：逐行记录峰值 RSS；机器资源不足即停，禁止缩小配置偷跑。
4. **fail-closed 触发**（任一命中即停，保留已有行）：行间 dataset_id 漂移；
   `require_production()` 失败；任何行 backend_error（错误行入 ledger，禁止
   丢弃或重试覆盖）。
5. **重跑规则**：任何重跑用新 run ID/新目录；失败与部分行原样保留。
6. **实现期停止条件**：RED 出现无关失败 → 停下诊断（§2.2(3)）；发现需要改动
   非范围文件（如 bench 运行器参数化缺口）→ 停下修订本契约，禁止顺手扩围。
7. **研究停止条件**：§9.2 负结果裁决后，本契约关闭；后续改进（v16 方向）必须
   另立预注册契约，禁止在本契约名义下追加调参轮次。

## 11. 批准与实施边界

- 批准流程：本契约经契约门禁评审（t8）通过后生效；批准记录补记于本节。
- 实施顺序（§2.2 闭环、§12 串行小 PR）：**契约 PR（本文）先行合入 main** →
  impl-a 从最新 main 创建实现分支：RED（§7）→ 最小实现（§5/§6 范围）→
  GREEN/回归（最小相关 → 相邻契约 → 全量并行/串行对账 → compileall →
  `git diff --check` → 分片检查）→ 原子提交 → 独立审核（t21）→ 集成合入
  （t26）→ 测量（measurer 按 §8 执行，运行后写
  `docs/p11_measurement_log.md`，引用被测精确 SHA）。
- 本契约不授权 push/PR（远程）/合并之外的操作由 captain 流程另行授权；全程
  仅本地、禁止 push（团队约束）。
- 工程标定不得升级为研究结论；本契约的全部数值证据（§1.2）为 P10 既有产物
  的引用与逐行复核，不含任何 v15 预测值。
