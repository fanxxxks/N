# P14 搜索层消化率预注册契约（P1-1 + P1-2）—— S 线需求基线

- 状态：**DRAFT — 待独立评审（t17 门禁）。批准前禁止实现动代码（§2.2 门禁顺序不可倒置）；批准后本文保持前瞻性，禁止用实现或测量输出反向修改。**
- 起草：impl-p1（AgentTeams alphagpt-p0-p1，S 线搜索层），2026-09-02
- 证据基线：main @ fde9f8b（阶段 0 清障后）；P10 产物 `docs/p10_searcher_comparison_20260901/summary.json`、
  `docs/p10_measurement_log.md`、`docs/p10_final_synthesis_report.md`、`docs/p10_searcher_fairness_contract.md`（APPROVED）；
  源码现场 `ashare_model/gp_search.py`、`ashare_model/tpe_search.py`、`ashare_model/baseline_harness.py`、
  `ashare_model/train_windows.py`、`ashare_model/search_contract.py`。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。仲裁顺序：本文/批准 → 契约测试 → 实现 → 测量日志。
  测量结果只能执行预注册裁决，不得反向改写本文。
- 分支：`codex/p1-search-digest`（自 fde9f8b 创建，S 线独立 worktree）；本文为该分支首个原子提交。
  全部工作仅本地，禁止 push/建远程 PR；合回 main 由 integrator 串行小 PR 执行（§12）。

## 1. 问题陈述（全部为既有证据，非本文新测量）

### 1.1 GP 中性适应度 = 当前最优 → 种群坍缩

`ashare_model/gp_search.py:408-413`：被跳过的提案（invalid、degenerate、canonical/semantic
duplicate、claimed 类）在 `fitness_of` 中获得 `evaluator.best_reward`（当前最优）作为"中性适应度"
（docstring 366-369 自述该设计）。后果链：

1. 复制类提案零评价成本却获得与当前最优并列的适应度，锦标赛选择（`tournsize=3`）对它们
   与真实最优个体无区分压力；种群迅速漂移进已评价语义类。
2. `best_reward` 单调不减（`baseline_harness.py:266,388-392`），越晚出现的复制类适应度越高，
   而早期真实评价个体的 stored fitness 静止不变——选择压力系统性偏向"产生跳过提案"的个体。
3. P10 实测：三行 GP 全部提前停滞——42:gp 消耗 199/2000（提案 4600 次，仅 4.3% 提案获得计费）、
   7:gp 99/2000、2024:gp 279/2000（消费率 5.0%–14.0%），终止原因一律
   `proposal_stagnation / three_generations_without_new_semantic_class`（`docs/p10_measurement_log.md`
   Stage A 行表）。GP 每行浪费 86%–95% 的正式预算。

### 1.2 TPE 重复提案 tell 当前最优 → 代理模型被投毒

`ashare_model/tpe_search.py:153-162`：`tell` 对被跳过提案（duplicate/claimed/semantic-dup/非有限
reward）报告 `evaluator.best_reward`。Optuna 拒绝 NaN 但接受任意有限值（源码注释 158-161 自认），
于是代理模型持续收到"该区域提案 = 当前最优质量"的假信号，恰好把 TPE 引导向产生重复的区域。
P10 的 reward 平台期（12/12 行 best-so-far 终点一致 0.98，`docs/p10_final_synthesis_report.md` §3.3）
放大了该缺陷：`best_reward` 饱和后，**每一个被跳过提案都被 tell 精确的平台值**，surrogate 的
below/above 分位信息被系统性污染。TPE 三行 proposals 4705–5255 次仅 2000 次获得计费
（重复/跳过提案占 57.5%–61.9%），满额消耗但搜索信息被稀释。

### 1.3 长度先验失衡：91% 提案堆满 12 token 上限，赢家实际 4–11

P10 campaign 12 行被评唯一公式（canonical AST，含 EOS 口径）的长度分布：

- TPE/random/RL 的 9 行：**90.4%–91.4% 的被评唯一公式堆在 12 token（= max_formula_len 上限）**；
  12 行合计 88.0%（14708/16712）。
- GP 三行仅 1.1%–4.5% 在上限（树生成器天然偏短）。
- 12 条 selected 赢家公式的 content 长度仅 **4–11**（GP 4/7/8；TPE/random/RL 11 content + EOS = 12）。

机制根因：TPE（`index % len(legal)` 映射）、random（`train_windows.sample_random_formulas`
对合法 mask 均匀采样）、RL（同 mask 下的随机初始化策略）三个逐 token 采样器在每个位置对
EOS 的命中概率 ≈ 1/|legal|，语法极少强制提前终止 → 提案系统性堆向上限。正式预算被大量
花在 P10 实测无赢家的长度区域。

### 1.4 目标×晋级错配：9/12 精英 max_tier≠A

晋级数据门 G6 默认 `allowed_data_tiers=("A",)`（`docs/p2_data_tier_contract.md` §5：
Champion 候选公式每个特征必须 Tier A；C 永不进入晋级结果）。而搜索目标（val reward）对
tier 完全盲：

- P10 12 条 selected 的 `data_tier.max_tier` 分布 = A:3 / B:2 / **C:7** → **9/12 ≠ A**，
  即 75% 的搜索赢家在默认晋级门下按构造必然被拒。
- 其中 **7/12 赢家使用 IND_REL_\*（C 档，申万当前快照非 PIT）**——恰是 P9 实测最强的因子族
  （IND_REL_RET_60 ΔIC +0.0271 PASS）。搜索把正式预算集中花在"研究上真实、晋级上不可用"
  的区域，而这是**目标函数与晋级约束的结构性错配**，不是任何一臂的搜索质量缺陷。
- 当前可采样词表（grammar v5，73 名 − 12 deprecated = 61）：Tier A 41 / Tier B 13 / Tier C 7
  （代码推导见附录 A）。晋级可用的有效空间（A，41 特征）只占可采样空间的 67%。

## 2. 假设

1. P10 campaign 产物（summary.json / measurement log）如实记录，其行为数字（消费率、长度
   直方图、tier 追溯）可作为诊断基线；本文引用数字全部可在附录 A 口径下复算。
2. Tier 映射（`PitLevel → DataTier`，`data_tier.py` v1）正确且稳定；IND_REL_\* 为 C 档源于
   非 PIT 行业快照（P2/P9 已披露），短期不会升级。
3. `feature_ids`（P6 §4.2）限制机制在四个后端（gp/tpe/random/rl 的 mask 与 pset 构建路径）
   均已实现且有测试覆盖，可作为 track 限制的现成机制。
4. A 线（reward v15）、B 线（promotion 执法）、C 线（fundamental 字段）并行推进且与本文
   无共享文件（§13 所有权表）；本文的机制与判据不依赖任何一线的合并结果。
5. Optuna `TPESampler` 接受任意有限目标值及 `-inf`（实装 4.9.0 实测
   `tell(-inf)` 接受；`tell(NaN)` 不抛异常而是 UserWarning +
   TrialState.FAIL，trial 不进入代理模型——`tpe_search.py:158-161` 的
   既有注释"仅拒绝 NaN"与实装不符，实现时更正该注释；惩罚路径永不发送
   NaN：非有限真实分数同样路由到惩罚值，t17 F2）。若安装版本实测拒绝
   ±inf，触发 §12 停止条件。
6. 测试 toy vocab/evaluator fixtures（`tests/test_gp_search.py`、`tests/test_tpe_search.py` 模式）
   可确定性地构造重复压力场景。

## 3. 范围与非目标

**范围**（P1-1 + P1-2 的机制与验收语义）：

- (A) P1-1a：GP 跳过提案的惩罚性适应度（替换中性 = best）；
- (B) P1-1b：TPE 跳过提案的惩罚性 tell（替换中性 = best）；
- (C) P1-1c：TPE 与 random 的长度先验（预注册长度目标分布，消除上限堆积）；
- (D) P1-2：研究/晋级预算分离（tier-A 晋级 track + 全词表研究 track，预算拆分与 fail-closed 校验）；
- (E) 版本 bump（SEARCH_CONTRACT 3→4、SEARCHER_BENCH 3→4）、RED 清单、bench v4 行字段。

**非目标**（逐条禁止顺手扩张）：

- 不改动 Reward、Protocol、评价协议、组合构造、执行、费用、股票池、PIT mask 语义
  （REWARD/PROTOCOL/EXECUTION/PORTFOLIO 全部不 bump；0.98 平台的区分度问题归 A 线）；
- 不改动词表、grammar、合法性掩码、语义规范化（GRAMMAR/SEMANTIC_CACHE 不 bump）；
  deprecated 特征依旧不可采样（grammar v4+ 既有规则原样保留）；
- 不改动晋级门禁本身：G6/G7 的执法与 fail-closed 归 B 线；本文的 tier-A track 只是搜索层
  预算分离，**不是**晋级裁决，也不是 admission 决定（ADMISSION_RULE 不 bump）；
- 不改动 RL 臂：RL rollout 的长度先验与惩罚性 tell 为明确排除项（理由见 §5.3），RL 的
  steps/batch split、随机初始化语义原样保留（IMITATION/RL_DIAGNOSTICS 不 bump）；
- 不改动 GP 的树生成器与长度分布（P1-1c 不适用 GP，理由见 §5.3）；
- 不切换生产默认搜索器（`gp` 保持不变）；不做 lifecycle 状态转换（P8/L 线另辖）；
- 不改预算计数口径、seed pairing、最大有效公式长度口径（§4 不变量，与 §7 一一对应）；
- 不新增特征/算子；不触碰数据库（无任何 DB 写入）；
- 不授权 push/PR/合并；不产生任何 alpha/晋级/production 声明。

## 4. 公平性不变量（AGENTS §7 对照，实施全程必须成立）

1. **预算单位不变**：唯一语义评价仍是唯一计费单位；invalid/degenerate/canonical-duplicate/
   semantic-duplicate 永不计费（T2-01 语义）；每行 requested/consumed、proposal/invalid/dup
   计数、best-so-far 曲线、长度直方图、终止/停滞原因照旧全量入产物（§7 记录全集）。
   提前停滞不得伪装成耗满预算。
2. **seed pairing 不变**：paired seeds [42, 7, 2024] 与执行顺序规则原样；同一 (seed, backend)
   的总唯一评价预算 2000 不变（拆分为研究 1200 + 晋级 800，见 §5.4）。
3. **长度上限口径不变**：最大有效公式长度 = max_formula_len = 12（含 EOS）在所有后端、所有
   track 不变；`len(tree_to_tokens(tree)) == len(tree)` 与统一 token 上限属性测试原样成立；
   GP node cap = max_len − 1 = 11 不变；GP node-cap 修正（P10 §4.3）不回退。
4. **track 内 matched**：同一 track 内四臂共享同一数据/fold/window、同一有效词表与语义规范化、
   同一候选评分器及全部 eligibility gates、同一 reward/protocol/execution/portfolio 配置与设备；
   唯一自由变量是 searcher 身份、paired seed（P10 §2 不变量 2 原样继承）。**跨 track、跨
   contract 版本的结果一律不得宣称 matched comparison**（SEARCH_CONTRACT v4 注释延续 P10
   先例）。提案长度分布本身不是 §7 共享项——它是被比较的搜索器机制（P10 四臂分布本不相同，
   该契约已获批准并执行）。
5. **停滞/终止规则不变**：GP 3 代无新类、TPE 50 连续无新类、random 8×B 有界池、RL
   steps×batch；`TERMINATION_REASONS` 集合不变（无新增终止原因）。
6. **tier 单一权威**：track 限制只能经 `data_tier.tier_features(("A",))`（registry → PitLevel →
   DataTier 唯一映射）在运行时派生 feature ids；禁止硬编码特征名单、禁止第二套 tier 映射、
   禁止把 track 伪装成 research domain（RESEARCH_DOMAIN 不变）。
7. **记录完备**：v4 行在 §7 全集之上新增 track、track 预算常量、tier 限制溯源（仅 A track：
   特征数与来源版本集合）、长度先验 profile id；缺失字段的 v4 产物按 §7 拒绝读取。

## 5. 方案（复用既有机制，不发明新机制）

### 5.1 P1-1a：GP 惩罚性适应度

- `gp_search.fitness_of`（gp_search.py:408-413）：跳过提案（`score_of` 返回 None，或返回的
  `val_reward` 非有限）的适应度从 `evaluator.best_reward` 改为**惩罚值**：
  `punitive = 该 run 迄今已注册分数中的最小有限 val_reward（worst-so-far）`；尚无任何有限
  分数时取 `-inf`（仅作 DEAP 内部适应度比较，不入任何产物；锦标赛语义下等价于最大惩罚）。
- 同步在 `SemanticBudgetEvaluator` 增加 `worst_reward` 只读属性（min over 已注册有限
  val_reward；空集时 -inf）——单一权威，GP/TPE 共用；`baseline_harness.py:274-284` propose
  docstring 中"(e.g. the current best)"同步更正。
- 语义：被跳过提案严格劣于（至多并列于）全部已评提案 → 选择压力转向"能产生新语义类"的
  个体；不引入任何可调常数；对 run 历史确定性。
- 非有限 reward 的已评分数（NaN）同样按惩罚值处理，消除 DEAP NaN 比较病态。

### 5.2 P1-1b：TPE 惩罚性 tell

- `tpe_search.tell`（tpe_search.py:153-162）：跳过/非有限情形从 `study.tell(trial, best_reward)`
  改为 `study.tell(trial, punitive)`，punitive 定义与 §5.1 完全一致（同一 `worst_reward` 权威；
  无有限分数时 `-inf`；Optuna 4.9.0 接受 ±inf，NaN 会以 UserWarning + FAIL 拒判——惩罚
  路径永不发送 NaN，t17 F2）。真实评价的有限 reward 原样 tell（重复类若命中
  已评缓存、`score_of` 返回真实分数，tell 真实分数——该路径不变，见
  `baseline_harness.py:291-293,349-357`）。
- 语义：surrogate 收到"该区域 = 迄今最差"的一致信号 → 后验回避重复生成区；同一 run 内
  worst-so-far 单调不增，无事后可调参数。
- 启动期时序不变：先 `flush` 后 tell 的既有顺序（tpe_search.py:172-176,183-186）保证首个
  tell 批次前已存在真实分数（除非首批全 invalid——此时按 -inf fallback，测试覆盖）。

### 5.3 P1-1c：长度先验（仅 TPE + random；GP 与 RL 明确排除）

- 预注册 profile：**`p14-uniform-2-11-v1`** —— 目标 content 长度分布 = 离散均匀
  U{2, …, max_formula_len − 1}（生产 max_len=12 → U{2..11}，即上限长度目标占比 10%）；
  新模块 `ashare_model/search_length_prior.py` 单一权威：profile id、逐 step 的 EOS 权重表、
  对 max_len < 4 的退化规则（不注入偏置，保持既有均匀行为）、对 toy vocab 的自适应裁剪。
  profile id 写入每行 diagnostics；任何重标定 = profile id 变更 + SEARCH_CONTRACT 再 bump。
- 接入点（两处，均为采样映射层，无签名变更）：
  - `tpe_search.propose`：`index → token` 的确定性映射在 step ≥ 2 后按权重表提升 EOS 的
    index 空间份额（surrogate 仍建模同一 index 域，ask/tell 机制零改动）；
  - `train_windows.sample_random_formulas`：合法 mask 上的均匀采样改为权重采样
    （非 EOS 合法 token 保持既有均匀份额）。
- **验收界**（机制门，RED 可测，生产词表 114/max_len 12、固定 seed，作用于**提案分布**
  ——机制唯一可控对象）：提案中 content=11（即 12 token 含 EOS）占比 **≤ 25%**、
  content ≤ 8 占比 **≥ 40%**；自然合法公式只会缩短分布，界留有工程余量。正式运行逐行
  记录的**被评唯一**长度分布是结果不是门（短长度唯一类会先耗尽，被评分布允许长于
  提案分布），其判据归 §9 判据 2。
- **GP 排除**：GP 三行上限占比 1.1%–4.5%、赢家 content 4–8，无堆积病灶；树生成深度
  min_=1/max_=2 是 GP 搜索机制的一部分，不动。
- **RL 排除**：RL rollout 位于 train.py（§12 热点文件，并行线禁改）且采样即策略语义
  （MODEL 邻接）；RL 消耗本受 steps×batch 上限约束（64% 消耗、steps_exhausted），长度
  偏置的收益与本契约的风险不成比例。列为后续独立预注册项。
- 无条件生效（非 caller 可选参数）：与 P10 `_max_nodes` 修正同先例——这是经预注册的
  永久语义修正，不是一次性实验配置（§9"默认配置向后兼容"以 SEARCH_CONTRACT 3→4 +
  版本集合记录承担；协议 random/matched baseline 行为随之变化并被版本集合如实记录，
  PROTOCOL 本身不 bump，P10 先例：SEARCH_CONTRACT 2→3 时 PROTOCOL 保持 25）。
  grammar/sampling 属性测试以合法性断言为主，预期不受分布影响；若个别 fixture 断言
  分布性质转红，按 §10.1 白名单情形 2 引用本文 §5.3/§6 修订。

### 5.4 P1-2 方案裁决：**研究/晋级预算分离**（二选一之选择）

**选定：方案乙——研究/晋级预算分离**。机制：

1. track 定义：`research`（全可采样词表，61 特征）与 `promotion_tier_a`
   （`tier_features(("A",))` **交可采样集**（排除 12 个 deprecated——
   deprecated 本就不可采样，grammar v4+ 规则，故行为无差，t17 F1）运行
   时派生，现 41 特征；随 registry 演进自动一致——C 线合并后无需改本文）。
2. campaign 行结构：v4 campaign 模式每 (seed × backend) 拆两行：research 行预算
   **P14_RESEARCH_BUDGET = 1200**，promotion_tier_a 行预算 **P14_PROMOTION_BUDGET = 800**；
   合计 2000 与 P10 每 (seed, backend) 总预算一致 → 最终 7h 正式运行的墙钟投影不被本
   契约改变（§10）。行序固定：seed 升序 → track（research, promotion_tier_a）→
   backend (gp, tpe, random, rl)，共 24 行。
3. 实现载体：track 行在 bench 内以现成 `feature_ids` 传递（bench → trainer → backend 的
   feature_ids 链路已存在并有 P6/P10 测试覆盖），零新采样代码；四臂的 A track 空间完全
   一致，track 内 matched（§4.4）。
4. fail-closed 校验：promotion_tier_a 行完成后逐一核对**全部计费候选**的
   `formula_data_tier_report(tokens)["max_tier"] == "A"`（P2-02 既有 CandidateScore 追溯；
   纯 token 解码，无 IO）；任何违规 → 该行 failed 且 campaign = FAILED（数据/身份漂移同级
   处置），禁止降级继续。
5. 晋级裁决关系：G6/G7 仍是唯一晋级权威；A track 的 selected 天然满足 tier-A 前提，但
   是否晋级仍由 B 线执法的门禁决定。research track 保留全词表信号（含 IND_REL_\*），
   为未来 PIT 化（C 线方向）保留研究证据——两条 track 的结论互不冒充。

**被否：方案甲——tier 感知采样权重**。理由：(i) 需在 GP pset 终端采样、TPE index 映射、
random mask 采样、RL rollout 四处分别注入权重——四处实现同一"tier 权重"概念，正是 §1
原则 3 禁止的第二实现漂移面，且 RL 注入点在 train.py 热点文件（§12 并行线禁改）；
(ii) 权重常数（A/B/C 各多少）无证据可预注册，任意取值即事后调参；(iii) 把"晋级假设"
硬编码进采样分布，同一次运行内再也无法检验 C 档信号的真实价值——P9 实测最强的
IND_REL_\* 族会被系统性压制，研究结论被晋级假设污染；(iv) 方案乙以 41 特征的
A-restricted 空间达到同等的"目标×晋级对齐"，且以既有机制零采样代码实现。

**预算拆分理由**：research 1200（60%）保持研究主线预算（P10 平台期到达坐标远早于 2000，
1200 保有充足余量）；promotion_tier_a 800（40%）≈ 校准预算 128 的 6.25 倍，对 41 特征
受限空间足够；两常数为预注册常量，实现与 CLI 不得放水。

## 6. 版本影响总表（`git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"` 全量检索后逐项判定）

**bump（2 项）**：

| 版本 | 旧→新 | 理由 | 迁移/拒绝 |
|---|---|---|---|
| SEARCH_CONTRACT_VERSION（search_contract.py:24） | 3→4 | 惩罚性适应度/tell 改变搜索选择语义；长度先验改变提案分布；A track 引入 track 域有效空间。跨版本结果永不互为 matched（延续 v2/v3 注释先例） | v≤3 产物只读 legacy |
| SEARCHER_BENCH_VERSION（searcher_bench.py:72） | 3→4 | campaign 增 track 维度（12→24 行）、双预算常量、A track tier 溯源、长度先验 profile id、行级 tier 纯度校验（全部加性） | v3 payload 仍可读；读取方在读新字段前必须 gate 版本==4 |

**明确不 bump（近邻项逐个给理由，其余见下组）**：

| 版本 | 现值 | 不 bump 理由 |
|---|---|---|
| PROTOCOL_VERSION | "25" | 协议 schema 与评价语义零改动；gp/tpe 基线行为变化由版本集合中的 search_contract=4 如实记录（P10 先例：SEARCH_CONTRACT 2→3、PROTOCOL 保持 25） |
| REWARD_VERSION | "14" | S 线零接触 reward.py；区分度问题归 A 线（其 bump 由 A 线契约管辖） |
| GRAMMAR_VERSION | 5 | 词表、合法性掩码、EOS 语义零改动；长度先验只改采样分布不改合法集 |
| FEATURE_REGISTRY_VERSION / DATA_TIER_VERSION / RESEARCH_DOMAIN_VERSION | 5 / 1 / 2 | 注册表与 tier 映射只读；track 不是 research domain（§4.4/§5.4） |
| MODEL_VERSION / IMITATION_VERSION / RL_DIAGNOSTICS_VERSION | 3 / 1 / 1 | alphagpt.py、train.py、RL 臂零改动（§5.3 RL 排除） |
| SEMANTIC_CACHE_VERSION | 1 | 缓存键、计费语义零改动；惩罚值不进入缓存 |
| ADMISSION_RULE_VERSION | 2 | 晋级门禁零改动（B 线所有）；track 不是 admission |
| LEDGER_SCHEMA_VERSION | 1 | campaign trial 数 12→24 是数据量变化，非 schema 变化 |
| ELITE_ARCHIVE_VERSION / P10_ADJUDICATION_VERSION / P3_MEASUREMENT_VERSION / BARE_FACTOR_BACKTEST_VERSION | 1 / 1 / 3 / 3 | 未触碰 |
| MANIFEST / FACTOR_COMPUTE / TARGET_CONTRACT / TIER_REPORT / FEE_MATRIX / RUNSPEC / ARTIFACT_SCHEMA | 1 / 1 / 1 / 1 / 1 / 1 / 2 | 数据与产物面零改动；无 DB 写入 |
| PORTFOLIO_CONSTRUCTOR / EXECUTION_SPEC / PORTFOLIO_OPTIMIZER / REBALANCE_POLICY | 1 / 2 / 1 / 2 | 组合/执行零改动 |
| INDEX_SCHEMA_VERSION（run_store.py） | 1 | L 线所有；S 线零接触 |

**新增 artifact-facing 标识**：长度先验 profile id（字符串常量 `p14-uniform-2-11-v1`，随行
diagnostics 记录）——不设独立 `*_VERSION`：它无独立 schema/reader，兼容性由
SEARCH_CONTRACT v4 统一承载（§3.2"新增版本接入既有兼容性权威"的适用说明）。

## 7. 迁移 / 拒绝策略

1. 无任何数据迁移：搜索产物按 run 隔离，v≤3 产物保持只读 legacy；v≤3 与 v4 结果之间
   一律不得宣称 matched comparison（search_contract.py 版本注释按 P10 文体续写）。
2. v4 读取方：campaign/行新增字段全部加性；读取方在依赖 track/profile 字段前必须校验
   `SEARCHER_BENCH_VERSION == 4`；对 v4 产物中出现未知 profile id 或缺失必录字段（§4.7）
   一律拒绝（fail-closed），禁止静默默认。
3. 请求构造拒绝：research/promotion 预算常量必须满足 1200 + 800 = 2000 且均 > 0；track 名
   必须属于 {research, promotion_tier_a}；违规在 plan 构造时抛错。
4. A track 纯度拒绝：§5.4.4 的行级校验违规 → 行 failed + campaign FAILED（不重试覆盖，
   失败行原样保留入 ledger；重跑用新 run ID/新输出目录）。
5. 回滚：单一 revert S 线 PR 即恢复 v3 行为；无持久状态依赖 v4 语义；无兼容层、无
   feature flag、无双路径（§9.1）。

## 8. 预期 RED 测试清单（实现前先红并留证；新文件 `tests/test_p14_search_digest.py`，
同 commit 注册进 `scripts/check_test_shards.py` 分片，否则 fail-closed 红）

1. **GP 惩罚性适应度单元**：toy evaluator 构造重复提案场景；断言被跳过个体的适应度
   == worst-so-far 且 < 当前 best_reward（现行为 == best → 稳定红）。
2. **GP 坍缩-恢复行为**：确定性 toy 场景（小词表高重复压力、budget 200、固定 seed）：
   断言 consumed/requested ≥ 0.50 且终止为 budget_exhausted（现行为停滞于 ≪50%、
   proposal_stagnation → 稳定红）。fixture 需复用 test_gp_search.py 的 toy 模式并证明
   确定性（同 seed 两次运行一致）。
   【delta note（t24-ruling-obsA / t49，判据非改动）】本条的 toy 坍缩断言经三环境实证
   不可稳健携带：GP 动态被 process hash 顺序 + BLAS 线程 FP 主导，跨环境实测
   (punitive, neutral) = (288, 84) / (115, 83) / (60, 103)（第三组为全钉环境
   PYTHONHASHSEED=0 + threadpool_limits(1)），单线程 2/2 通过但跨 hash seed 仍翻转——
   方向本身被环境决定，toy 可达语义类空间退化所致。实现落地为 FP 稳健确定性护栏
   （threadpool_limits(1) 下两锚点臂各两遍 consumed + best-so-far 逐字节相等，
   tests/test_gp_search.py），跨臂消耗比较断言撤除；坍缩裁决由 §9 正式运行承担
   （GP 消耗 ≥ 50%——判据值与全部冻结数字逐字不变，正式运行机制见 §9 研究测量段）。
   FP 敏感性实证全文见 t49 任务产出与 t31/t32 测量日志。
3. **TPE 惩罚性 tell**：以 spy/monkeypatch 捕获 `study.tell`（或对实现的纯函数权威断言）：
   跳过提案被 tell 的值 == worst-so-far ≠ best_reward；首批全 invalid 时 fallback -inf；
   真实评价路径 tell 真实分数不变（现行为 tell best → 稳定红）。
4. **长度先验诱导分布**：生产词表、max_len=12、固定 seed。random 路径：对
   `sample_random_formulas` 采样 ≥ 2000 提案，断言 content=11 占比 ≤ 25%、content ≤ 8
   占比 ≥ 40%、全部序列 EOS 终止且 ≤ 12 token；TPE 路径：对 propose 映射的发射序列
   断言同一分布界与 ≤ 12 token 上限（未终结序列按发射长度计，另行统计）；max_len=6
   toy 场景不崩溃且自适应（现行为 91% 堆上限 → 稳定红）。
5. **A track 限制与纯度**（单元级，无数据）：track→feature_ids 派生 ==
   `tier_features(("A",))`（且排除 deprecated）；A track 行预算 == 800、research 行 == 1200、
   合计 == 2000；注入一个 max_tier ≠ A 的计费候选 → 行 failed + campaign FAILED
   （现行为无 track 支持 → 稳定红）。
6. **版本 pin**：SEARCH_CONTRACT == 4、SEARCHER_BENCH == 4（新文件内断言）。
7. **既有 pin 修订**（§10.1 白名单情形 2，引用本文 §6）：`tests/test_p4_search_contract.py:34-37`、
   `tests/test_p10_campaign.py:120-121`、`tests/test_semantic_sampling.py:290` 的 ==3 断言
   更新为 ==4；旧行为为何改变（§5.1–5.4 语义修正）与新断言为何不弱（版本+加性字段断言）
   随 commit 证据列出。除此之外不得修改任何既有测试；若有其他既有测试转红，先停下按
   §10.1 举证裁决，禁止对答案。

## 9. 测量方案与判据（允许负结果）

**工程验证（S 线实现任务 t18 的完成门，全部 engineering 语义）**：

- RED 三步先于实现（§2.2.3），仅含预期失败；聚焦测试 → 相邻契约（gp/tpe/bench/grammar/
  sampling/domain）→ 全量 `python -m pytest -q tests -n auto`（并行-串行对账记入日志，未
  对账则串行）→ `python -m compileall -j 0 -q …` → `git diff --check` → 分片检查 exit 0。
- 工程冒烟：campaign 模式 budget 64、seeds {42}、24 行（ledger 登记、engineering 标注、
  原始产物新目录）；验收：24 行 completed；A track 行级纯度校验通过；长度分布在 §5.3
  界内；GP research 行消费率显著高于 P10 停滞形态（描述性，不下研究结论）。

**研究测量（预注册给最终阶段正式运行复用；由正式运行契约与 measurer 执行）**：

- 逐 arm × track 记录：consumed/requested、proposal/invalid/dup、上限堆积占比、
  best-so-far 曲线、selected tier 追溯（§7 记录全集 + §4.7 新字段）。
- 预注册判据（测量后逐条执行，禁止事后改写）：
  1. GP 消费率：research track 三 seed 中位 consumed/requested **≥ 0.50** → P1-1a 判定
     有效；**< 0.50** → 如实记录负结果（"惩罚性适应度未达成预期消费率"），机制去留回
     契约修订裁决，禁止静默调参。
  2. 上限堆积：机制门以 §5.3 提案分布界为准（RED 已盖）；结果判据按**被评唯一**分布：
     tpe/random 全部行 content=11 占比 **≤ 50%**（P10 基线 91.4% 减半）→ P1-1c 判定有效；
     > 50% → 如实记录负结果，并必须同时报告各 content 段唯一类耗尽占比作为归因
     （短长度池耗尽会把被评分布推向长段——允许的机制后果，须与机制失效区分），机制
     去留回契约修订裁决，禁止静默调参。
  3. A track 纯度：计费候选 max_tier==A 占比 **== 100%**（硬门，违规 = fail-closed 缺陷，
     campaign FAILED，不构成研究结果）。
  4. 错配消除的判定以 1–3 为准；**A track 无 arm 通过回测门槛是合法负结果**（tier-A 空间
     可能在该预算下无可晋级 alpha），不构成对预算分离机制本身的否证。
- matched 边界：比较只在同 track、同 contract 版本、同 reward 版本内进行；P10（v3、
  REWARD 14）与 P14 之后的运行（v4、届时集成的 REWARD 版本）之间禁止 matched 宣称。
  本线不产生任何 alpha/晋级/production 结论。

## 10. 资源上限

1. 实现阶段禁止全预算搜索：验证 = 单元/属性测试 + §9 工程冒烟（budget 64 × 24 行，
   按 P10 校准 ~0.7 s/eval 投影 ≈ 18 min）+ 全量回归；任何超出需在测量日志报备理由。
2. 正式运行墙钟：每 (seed, backend) 总唯一评价数维持 2000（1200 + 800），Stage A 墙钟
   投影不被本契约改变（P10 实测 3.91h 对 7h 上限的 56%）；行数 12→24 的行级开销可忽略
   （P10 附录 A 实测行级窗口准备 <1s，仅加载/门禁为单次 5.6 min 不变）；更短公式评价
   更省，投影偏保守。7h 门禁、偏差熔断与放行审核归最终阶段契约与 t33，本文不重复、
   不放宽。
3. 无 DB 写入、无网络依赖、无新生产依赖（新模块仅用 stdlib + 既有 vocab/registry API）。

## 11. 裁决规则（实现与评审依据）

- 实现完成的定义：§8 RED 全绿 + §9 工程验证全绿 + 版本表（§6）落实 + 分片注册 +
  §13 所有权边界未越界；缺一不可。
- 评审（t17/t24）按本文逐节对照：问题陈述证据可复算（附录 A 口径）、不变量（§4）逐条
  有测试或守卫、非目标未被扩张、版本表逐项落实、RED 先行证据完整。
- 任何与本文冲突的实测 → 停下按 §12 处置；禁止以实现反推修改本文。

## 12. 停止条件（任一命中即停，保留证据，回契约修订，禁止静默变通）

1. RED 出现预期之外的无关失败（§2.2.3）。
2. Optuna 实装版本拒绝惩罚值（-inf 或 worst-so-far 有限值）。
3. 长度先验在不违反 §4.3（长度上限/合法集/计费）的前提下无法达到 §5.3 验收界。
4. `feature_ids` 路径无法保证 A track 纯度（存在越权特征泄漏）。
5. 既有测试转红且无法按 §8.7 白名单完成契约举证。
6. 并行线（A/B/C/L）与本文出现共享文件冲突（对照 §13），先停 S 线，交 captain/integrator
   裁决接口冻结。

## 13. 实施顺序与文件所有权（S 线声明）

实施顺序（t18，依据 §2.2）：基线记录 → 本文批准 → RED（§8.1–8.5,8.6 新文件）→ pin 修订
（§8.7 白名单）→ 最小实现（§5）→ GREEN/回归（§9）→ 原子提交（代码+测试+版本+docstring
同 commit）→ 工程冒烟 + 测量日志（evidence-only，引用精确 SHA）。

| 处置 | 文件 | 内容 |
|---|---|---|
| 修改 | `ashare_model/search_contract.py` | 版本 3→4 + 版本注释（v4 语义摘要） |
| 修改 | `ashare_model/gp_search.py` | fitness_of 惩罚值 + docstring |
| 修改 | `ashare_model/tpe_search.py` | tell 惩罚值 + propose 长度先验映射 + docstring |
| 修改 | `ashare_model/baseline_harness.py` | `worst_reward` 属性 + docstring 更正 |
| 修改 | `ashare_model/train_windows.py` | sample_random_formulas 权重采样（EOS 先验） |
| 修改 | `ashare_model/searcher_bench.py` | v4：track 行/预算常量/行字段/纯度校验 |
| **新增** | `ashare_model/search_length_prior.py` | profile 单一权威（id、EOS 权重表、退化规则） |
| **新增** | `tests/test_p14_search_digest.py` | RED 清单（同 commit 注册分片） |
| 修改 | `tests/test_p4_search_contract.py`、`tests/test_p10_campaign.py`、`tests/test_semantic_sampling.py` | 版本 pin 3→4（白名单情形 2） |
| **禁改** | `train.py`、`evaluation.py`、`alphagpt.py`、`reward.py`、`promotion.py`、`vocab.py`、`feature_registry.py`、`data_tier.py`、`run_store.py`、`ashare_portfolio/*`、`ashare_data/*`、`webapi/*` | 热点文件/他线所有权/零语义接触；触碰前必须先向 captain 申报 |

## 附录 A：证据复算口径

1. 消费率/提案数：`docs/p10_measurement_log.md` Stage A 行表（42:gp 199/2000/4600、
   7:gp 99/2000/2160、2024:gp 279/2000/5600；tpe 2000/2000/4705–5255；random 2000/2000/
   3859–3993；rl 1280–1289/2000/2000）。
2. 上限堆积与赢家长度：对 `docs/p10_searcher_comparison_20260901/summary.json` 逐行
   `formula_len_histogram`（去重 canonical AST，含 EOS）求 12-token 占比（tpe/random/rl 行
   90.4–91.4%，12 行合计 14708/16712 = 88.0%，gp 行 1.1–4.5%）；`selected.tokens` 去尾
   EOS=113 得 content 长度（GP 无尾 EOS，原样即 content）：4/7/8/11×9。
3. tier 与 IND_REL：同文件逐行 `selected.data_tier.max_tier`（A:3/B:2/C:7）与
   `formula_text` 含 "IND_REL" 计数（7/12）。
4. 词表 tier 计数（fde9f8b 现场，PYTHONDONTWRITEBYTECODE=1 只读推导）：
   FEATURE_NAMES 73、vocab size 114、EOS=113；deprecated 12；samplable 61 =
   A 41（`tier_features(("A",))` 交可采样集）+ B 13 + C 7。
5. 本文全部数字为对既有产物的复算与源码现场引用，不构成新测量；测量日志（实现后）
   只写"实际发生什么"（§11.1），本文批准后保持前瞻。
