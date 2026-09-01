# P10 四搜索器公平对比预注册契约（gp/tpe/random/rl）

- 状态：**DRAFT — 待批准**。批准前不实施任何语义变更（本契约的实现在批准后
  由 t7/t8 按本文执行）。批准后本文保持前瞻性，禁止用实现或测量输出反向修改。
- 起草：search-runner（AgentTeams alpha-orth-research t6），2026-09-01
- 证据基线：t1 数据/产物 Readiness 审计（G1–G7 PASS，manifest 重建后
  dataset_id=a839ecf2…）；t2 62 因子库存审计；t3/t4/t5 P9 因子族契约与实现
  （分支 codex/p9-factor-families，提交 781f5e1→403ee15，词表 v5）；P4 搜索
  契约（docs/p4_search_transformer_contract.md，ADMISSION_RULE_VERSION=2）。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。仲裁顺序：本文/批准 → 契约测试
  → 实现 → 测量日志。测量结果只能执行预注册裁决，不得反向改写本文。

## 1. 问题陈述与动机

1. **P4 之后的有效空间已改变，必须重测**。P4/T2-03 在 v3 代词表上以
   ADMISSION tier（8×128=1024 唯一语义评价、窗口 300×400）裁决 RL 未获准入，
   生产默认保持 `gp`。此后 P9 完成：GRAMMAR 3→5（12 个 deprecated 特征退出
   采样：8 个窗口变体 + LIMIT_STREAK/LIQ_SHOCK_20/CROWD_TURNOVER_60 二次裁
   决）、11 个新特征入册、稀疏事件表示修复（FACTOR_COMPUTE_VERSION=1）。
   当前有效采样词表为 **61 特征**（73 名 − 12 deprecated，grammar v5，
   feature_version ce3cf72b4af8，词表 size 114，EOS=113）。四个搜索器的有效
   搜索空间已整体改变，P4 的结论不能平移到 v5 空间。
2. **发现一处未披露的匹配性缺口（本契约的核心公平性修正）**：GP 的树大小上限
   `_max_nodes(max_formula_len) = max_formula_len // 2 = 6`，而
   `tree_to_tokens` 每节点恰好发射 1 个 token（既有属性测试
   `len(tree_to_tokens(tree)) == len(tree)` 为权威），故 GP 的有效公式长度上限
   为 **6 content + EOS = 7 token**；TPE/Random/RL 的有效上限为 12 token
   （含 EOS）。这与 P4 契约 §1/AGENTS.md §7"所有后端输出必须满足统一 token
   长度上限"的口径不一致——四个后端并未共享同一最大有效公式长度。本次校准
   运行（附录 A）中 GP 全部候选 ≤6 content token、TPE 选中公式 11 content
   token，实证了该缺口。**不修复则任何"matched comparison"声明都不合法**
   （AGENTS.md §7：不同有效搜索空间的结果不得宣称 matched comparison）。
3. **缺少回测端硬门槛**：P4 只以 best-so-far area 与 OOS active IR 裁决，从未
   把"搜索赢家"放到带完整成本的组合回测硬门槛下检验。t2 已证明 A 股快价格
   信号在高换手下成本硬约束显著（10 日调仓 ~14%/yr）；搜索赢家能否通过净成本
   回测门槛是决定"下一步改进方向"的关键证据，目前缺失。
4. **预算从未被标定到真实墙钟**：正式对比需要"同一唯一语义评价预算"，且该
   预算必须匹配可用算力（本任务约束：预估总算力 ≈ 7h 墙钟）。现有 v1 bench
   产物是 cuda/旧数据集（dataset b927…，legacy）上的产物，不能用于标定当前
   CPU 主机。本契约以本次工程校准（附录 A）为标定依据。

## 2. 范围与非目标

**范围**：(A) 四搜索器匹配对比的预注册设计（词表、长度、预算、paired seeds、
窗口/fold、评测器、记录字段）；(B) GP 有效公式长度对齐修正（唯一一处搜索空间
语义变更）；(C) 回测硬性及格门槛与 OOS 裁决方案；(D) 资源上限与超时熔断；
(E) ledger/测量日志要求与"下一步改进方向"的预注册裁决映射。

**非目标**：

- 不切换生产默认搜索器（`model.searcher: gp` 保持不变；本对比不产生任何
  production admission 或晋级决定，ADMISSION_RULE_VERSION 不变）；
- 不比较 imitation RL 臂（P4 已裁决；本契约 RL 臂固定 random-init，独立于其他
  搜索器输出）；不新建/消费 elite archive 于 RL 初始化；
- 不改动 Reward、Protocol、组合构造、执行、费用、股票池、PIT mask 语义；
- 不新增特征或算子；不改动词表 v5（61 可采样特征）与语义规范化（SEMANTIC_CACHE
  单一权威）；
- 不做 paper/sim 模拟盘操作，不做 lifecycle 状态转换（P8 契约另行管辖）；
- 不修改 t2/t5 基线审计产物；不重跑数据同步（dataset_id 以运行时实测为准，
  见 §6 fail-closed 条款）。

**不变量**（实施全程必须成立）：

1. 唯一语义评价是唯一预算单位；无效、退化、canonical duplicate、semantic
   duplicate 不计费（T2-01 ledger 既有语义）。
2. 四臂共享：同一 dataset_id、同一 fold/window（含 cap）、同一有效词表与语义
   规范化、同一最大有效公式长度（含 EOS 口径）、同一候选评分器
   （SemanticBudgetEvaluator）及其全部 eligibility gates、同一 reward/protocol/
   execution/portfolio 版本与配置、同一设备；唯一自由变量是 searcher 身份与
   paired seed。
3. 每臂每 seed 的 requested/consumed budget、proposal/invalid/semantic-duplicate
   计数、终止/停滞原因、best-so-far 曲线、公式长度分布全部入产物；提前停滞
   不得伪装成耗满预算。
4. Stage B/C 使用搜索时选出的 direction（训练/验证数据上决定），禁止在测试
   窗口上重新拟合方向。
5. 负结果是合法结果：任何臂（含全部臂）通不过回测门槛都如实报告。
6. 所有测量引用 dataset_id、data_end、commit、配置 hash 与设备。

## 3. 机制选择（复用既有机制，不发明新机制）

| 需求 | 采用机制 | 先例 |
|---|---|---|
| 四后端统一运行结果 | `SearchRequest`/`SearchResult`（search_contract） | P4 §2 |
| 匹配运行器 | 扩展 `searcher_bench.benchmark_searchers`（同一 trainer/窗口/评测器构造路径），新增 seed 列表 campaign 模式 | searcher_bench v2 |
| 预算账本 | trainer 级 semantic cache（每行新建 trainer，cache 不跨行）+ run 级 `ExperimentLedger` append-only JSONL | T2-01、AGENTS §4.3 |
| GP 树↔token 映射 | `tree_to_tokens`/`tokens_to_tree` 不变，仅修正 `_max_nodes` | gp_search 既有 |
| 选中公式 OOS 评价 | `evaluation.evaluate_formula`（全历史执行→fold 测试窗切片→`evaluate_signal` 全引擎指标） | P4 §5 OOS active IR |
| 回测基准行 | `eval_metrics.benchmark_row`（等权基准，仅上下文参考，非门槛） | 同上 |
| 资源/时间证据 | bench 行内 wall/RSS/termination 字段 + 附录 A 工程校准 | P1-04 |

## 4. Stage A：匹配搜索运行设计（t7 执行）

### 4.1 预注册常量（实现必须原样使用，不得 CLI 放水）

| 常量 | 值 | 说明 |
|---|---|---|
| P10_COMPARE_BUDGET | 2000 | 每行 requested 唯一语义评价数（B） |
| P10_COMPARE_SEEDS | [42, 7, 2024] | 协议默认 seed 三元组（paired seeds，≥3 满足任务要求） |
| P10_COMPARE_WINDOW | (300, 400) | 与 ADMISSION_WINDOW 相同的 fold 训练窗头部 cap |
| P10_COMPARE_FOLD | 0 | protocol folds[0]：train_end 2020-12-31 → test_end 2021-12-31 |
| RL split | steps=8, batch=250 | 与 P4 ADMISSION_STEPS=8 先例一致（校准行用了 bench v2 的 4×32，仅作计时依据，不作 split 依据） |
| 非 RL split | steps=B=2000, batch=1 | bench 既有惯例 |
| GP node cap | max_formula_len − 1 = 11 | §4.3，唯一语义变更 |
| 行序 | seed 升序 × (gp, tpe, random, rl) | 固定执行顺序，便于中断后核算 |

预算整除约束：B=2000 同时满足 `%4==0` 与 `%8==0`（rl_split 与新 split 均合法）。

### 4.2 固定配置

- 配置单一来源：`config/ashare_config.yaml`（+ `runtime_overrides.yaml` 合并），
  运行时记录 effective config hash。实测有效值：top_n=20、single_weight_cap=0.05、
  initial_capital=100000、daily rebalance、horizon=1、佣金 0.00025（最低 5 元）、
  印花税 0.0005、过户费 0.00001、滑点 0.0005、benchmark 全市场等权、
  max_formula_len=12、validation_fraction=0.35、validation_splits=4（中位数
  validation reward 选出每行 selected）。
- 设备：本机 CPU（`resolve_device` 实测 cuda=False，已核实）；逐行记录 device。
- 版本集合（2026-09-01，HEAD 403ee15 实测）：PROTOCOL 25、REWARD 14、GRAMMAR 5、
  FEATURE_REGISTRY 5、FACTOR_COMPUTE 1、RESEARCH_DOMAIN 2、MODEL 3、
  SEARCH_CONTRACT 2→3（本契约实施后）、SEARCHER_BENCH 2→3（同）、
  SEMANTIC_CACHE 1、DATA_TIER 1、EXECUTION_SPEC 2、PORTFOLIO_CONSTRUCTOR 1。
- 数据：dataset_id 以运行时 `loader.dataset_id` 实测并逐行记录；12 行必须
  携带同一 dataset_id（§6）。

### 4.3 GP 有效公式长度对齐（唯一搜索空间语义变更）

- 修正：`_max_nodes(max_formula_len) = max_formula_len − 1`（对 max_len=12 即
  11 个节点），保证任意合法树 `len(tree_to_tokens(tree, vocab)) + 1(EOS)
  ≤ max_formula_len`，且达到上限的树（11 节点）恰好用满 12 token 口径。
- 同时修正 `_max_nodes` docstring 中"n 节点序列化为 2n−1 token"的错误表述
  （`tree_to_tokens` 实为每节点 1 token；以既有属性测试为权威）。
- 该修正放宽 GP 的有效空间至与其他三臂相同的长度边界，是 matched comparison
  的前提。修正前后结果不可互比，故 `SEARCH_CONTRACT_VERSION 2→3`。

### 4.4 每行必录字段（AGENTS §7 全集）

requested/consumed budget、proposal_count、invalid_proposals、
semantic_duplicates、termination_reason、stagnation_reason（停滞必须带具体
原因）、best_so_far 曲线（含 EOS 口径的坐标 = 累计唯一评价数）、被评唯一公式
的 token 长度分布（直方图）、selected（tokens/text/hash/direction/各分项
reward 与 gate 拒绝原因）、wall_seconds、peak RSS、device、versions 集合、
dataset_id、window/fold/cap、seed。consumed < requested 时差额必须可见。

### 4.5 预算推导（预估总算力 ↔ 7h 墙钟）

工程校准（附录 A：budget 128、seed 42、CPU、当前数据集与 v5 词表、窗口
300×400、fold 0）实测单评价耗时：gp 0.657 s/eval（58 唯一后停滞）、tpe 1.012、
random 0.562、rl 0.676。最坏情况模型（每行都耗满 B）：

```
Stage A 墙钟 ≈ 3 seeds × B × (0.657+1.012+0.562+0.676) = 8.72·B 秒
B=2000 → 17,436s ≈ 4.84h；+ 单次加载/门禁 ≈ 0.2h → Stage A ≈ 5.1h
Stage B/C ≈ 0.3–0.6h → 全程 ≈ 5.4–5.7h ≤ 7h（余量 ≈ 19–23%）
```

B=2000 在 7h 约束下取到"有安全余量的最大预算"；若 B=2400 则全程 ≈ 6.3h，
余量 <10%，不采纳。**偏差熔断**：第一个 seed 的 4 行完成后，若实测行均墙钟
超过校准投影的 2 倍，停止后续 seed，campaign 标记 FAILED（fail-closed），
修订本契约后才可重跑。

## 5. Stage B/C：OOS 评价与回测硬性及格门槛（t8 执行）

### 5.1 单一评价路径

对 12 行 selected 公式逐一调用 `evaluation.evaluate_formula(tokens, loader,
folds[0], backtest_config, direction=selected.direction)`：全历史执行 →
fold 0 测试窗（(2020-12-31, 2021-12-31]）切片 → `evaluate_signal` 全引擎指标。
返回 payload 即唯一证据来源，禁止重算第二套指标。`benchmark_row` 等权基准同窗
记录为上下文参考，不参与门槛。公式在测试窗无效/退化（`evaluate_formula`
返回 None 或 NaN 指标）记为该行失败行，合法。

### 5.2 回测硬性及格门槛（逐公式，四条全过才算过）

指标定义 = `AshareBacktestEngine._metrics` 原文（单一语义路径，252 日年化，
夏普含 2% 无风险补偿）：

| 门槛 | 阈值 | 引擎定义 |
|---|---|---|
| 年化净收益 | **> 0.10** | `(1+total_return)^(252/n) − 1` |
| 最大回撤 | **< 0.15** | `max clip(1 − equity/running_max, 0, 1)` |
| 夏普 | **≥ 1.0** | `(annual_return − 0.02)/(std·√252)` |
| 卡玛 | **≥ 1.0** | `annual_return/(max_drawdown + 1e-9)` |

组合/成本语义 = 共享 `PortfolioConstructor` + `ExecutionCostModel`（T+1、
涨跌停/停牌 mask、佣金下限、整手、单股上限 5%），资金 100,000，top_n=20，
日调仓——与搜索期奖励同一套执行语义（§8 单一路径）。**组合活动门**：逐行记录
rebalance_count/order_count/suppressed_trade_count/average_turnover；接近空仓
或仅建仓一次的低换手"达标"不得宣称有效（AGENTS §8.5）。

### 5.3 臂级汇总（预注册）

- `oos_positive(arm)`：≥2/3 seed 的 selected 公式 active_ir > 0。
- `backtest_admissible(arm)`：≥2/3 seed 的 selected 公式通过 §5.2 全部四条。
- `search_efficient(arm)`：paired-median best-so-far area 最高。area 按请求
  预算 B 积分、停滞行把最后 best 持有到 B（P4 §5 既有口径）。
- n=3 的 paired 样本只支持方向性描述（3/3 或 2/3 一致性），**禁止宣称统计
  显著性**。

## 6. 资源上限与超时熔断

1. **墙钟上限**：Stage A ≤ 7h（campaign 级，行间核算）；Stage B/C ≤ 1h。
   行间检查：累计+下一行投影 超限 → 停止启动新行，未运行行记 `not_run`，
   campaign 标记 FAILED——**不得静默截断后继续宣称 matched comparison**。
2. **每行有界性**（工程事实，写入契约测试）：gp 3 代无新语义类即停；
   tpe 50 连续无新类即停；random 从 `8×B` 的有界 canonical 池提议，止于
   budget_exhausted 或 candidate_pool_exhausted；RL 固定 8 step。
   四个循环都有界，不存在无限挂起；单行墙钟逐行记录并与附录 A 投影对照
   （>3× 投影 → 在测量日志中强制解释）。
3. **内存**：校准实测峰值 RSS 2.95–3.44GB/行；逐行记录峰值；机器内存不足时
   停止而非换小配置偷跑。
4. **fail-closed 触发条件**（任一命中即停，保留已有行，campaign=FAILED）：
   - 行间发现 dataset_id 变化或与 provenance 不一致；
   - `ProductionGateRunner.require_production()` 失败（bench 入口既有）；
   - 校准偏差熔断（§4.5）；
   - 任何行 backend_error（错误行必须入 ledger，不得丢弃或重试覆盖）。
5. **重跑规则**：任何重跑使用新 run ID/新输出目录；失败与部分行原样保留；
   禁止覆盖或拼接。

## 7. Ledger 与测量日志

- 每次 campaign 通过共享 `ExperimentLedger`（append-only JSONL，
  LEDGER_SCHEMA_VERSION=1）登记 12 个 trial（running→closed/failed），Stage
  B/C 的 12 次评价同样入账；崩溃行保留 running 或 failed 原状。
- 产物布局：`data/p10_searcher_comparison/<run_id>/`（bench v3 campaign JSON、
  ledger JSONL）；bench CLI `--output` 指向新目录，禁止写历史文件。
- `docs/p10_measurement_log.md`（运行后写，禁止预写结果）：命令、commit、
  环境（OS/CPU/内存/设备）、dataset_id 与 data_end、effective config hash、
  版本集合、B/seeds、逐行 requested/consumed/终止/停滞/墙钟/RSS、Stage B/C
  逐公式指标与门槛判定、裁决结果、未运行项及原因。

## 8. 预注册裁决规则（决定"下一步改进方向"）

运行前固定，测量后逐条执行，禁止事后改写：

| 观测 | 裁决（下一步改进方向） |
|---|---|
| 恰一臂 backtest_admissible 且 search_efficient | 改进方向 = 该搜索器：预算放大曲线 + 该臂专项改进立项 |
| ≥2 臂 backtest_admissible | 方向 = 对 search_efficient 臂做预算放大研究；如实报告并列，不宣称单赢 |
| 无任何臂 backtest_admissible | **合法负结果**：当前预算/窗口/成本下无搜索器产出可过门槛公式；方向转向评价器/Reward 与成本-换手结构（引用 t2 成本证据），不归咎某一搜索器 |
| 全臂 consumed≪requested 且停滞 | 把"预算瓶颈在提案生成（重复率）而非评价"记录为首要发现，方向 = 采样多样性/去重机制改进 |
| 单臂 search_efficient 但 oos 非正 | 明确记录"搜索效率与 OOS 表现脱钩"疑点（奖励-未来表现错位），方向 = 奖励构造审计，不改门槛 |

补充裁决：(a) GP 停滞本身是测量对象（记录 unique/requested 比例），不因停滞
判负；(b) 四臂全部行、含失败与停滞行，一律完整公布；(c) 本对比是 fold 0 上的
研究测量，**不构成 alpha/晋级/production 证据**；fold 0 测试窗是协议常规使用
窗口，不是新的锁定 holdout。

## 9. 版本影响总表与迁移/拒绝策略

| 版本 | 旧→新 | 理由 | 迁移/拒绝 |
|---|---|---|---|
| SEARCH_CONTRACT_VERSION | 2→3 | GP 有效公式长度对齐改变四臂匹配空间；前后结果不可互比 | v2 行只读 legacy；不得进入 p10 matched 声明 |
| SEARCHER_BENCH_VERSION | 2→3 | campaign seed 列表、RL split 参数化、行级长度分布/campaign 字段（additive） | v1/v2 JSON 只读保留；v1 不得用于任何 matched 声明（其 dataset b927 已 legacy） |
| 其余全部 | 不变 | PROTOCOL 25 / REWARD 14 / GRAMMAR 5 / REGISTRY 5 / FACTOR_COMPUTE 1 / RESEARCH_DOMAIN 2 / MODEL 3 / SEMANTIC_CACHE 1 / DATA_TIER 1 / EXECUTION_SPEC 2 / PORTFOLIO_CONSTRUCTOR 1 / ADMISSION_RULE 2 / LEDGER_SCHEMA 1 | 无数据迁移；单 revert 回滚 |

明确不 bump：GP node cap 修正在 SEARCH_CONTRACT 中版本化，不再单独动
GRAMMAR/PROTOCOL（词表与评价协议零改动）。

## 10. RED 测试清单（实现前先红，t7 落地）

1. **长度对齐属性测试**：`_max_nodes(12) == 11`；对随机 typed 树满足
   `len(tree_to_tokens(tree, vocab)) + 1 ≤ max_formula_len`，且存在 11 节点树
   恰好取满（6 旧上限树仍合法——旧产物公式语义不受影响）。
2. **统一 token 上限属性测试**：四后端采样器产出的完整 token 序列（含 EOS）
   ≤ max_formula_len（P4 §1 口径的直接断言）。
3. **版本 pin 更新**：`test_p4_search_contract.py` 的
   SEARCH_CONTRACT/SEARCHER_BENCH pin 2→3（白名单 §10.1 情形 2，引用本契约 §9）。
4. **campaign 运行器测试**：12 行 = 3 seeds × 4 backends；provenance 中除 seed
   外全部字段逐行一致；RL split=(8,250)；B=2000；行序固定；`not_run` 与
   backend_error 行进 ledger；行间 dataset_id 不一致即 fail-closed。
5. **预算记账**：consumed ≤ requested；duplicate/invalid 不计费（扩展现有
   T2-01 测试至 campaign 边界）；best_so_far 坐标 ≤ consumed。
6. **门槛判定**：§5.2 四条判定只读取引擎 metrics 字段；direction 来自
   selected，不得在测试窗重拟合（负例：注入 refit 路径必须被拒绝）。
7. **既有 GP 行为守卫**：旧上限（6 节点）树在新代码下 tokens_to_tree 往返
   parity 不变（历史公式可解析性）。

## 11. 批准与边界

- 批准后实施顺序：t7 = RED（§10）→ 最小实现（gp_search `_max_nodes`、
  searcher_bench v3 campaign 模式、ledger 接线）→ GREEN/回归 → Stage A 运行
  + 测量日志；t8 = Stage B/C + 裁决报告。两步各自原子提交，不与本契约混合。
- 本契约不授权 push/PR/合并；不授权任何 paper/sim/lifecycle 操作。
- 未运行/未验证项必须如实列出；工程校准（附录 A）不得升级为研究结论。

## 附录 A：工程校准证据（engineering，非研究证据）

- 运行：2026-09-01 09:30–09:47 本机；`python -m ashare_model.searcher_bench
  --output data/p10_calibration_engineering.json --budget 128 --seed 42
  --fold 0 --window-cap 300x400`；HEAD 403ee15；CPU（cuda=False）；
  dataset_id=a839ecf2…；GRAMMAR 5（61 可采样特征）；top_n=20（effective config）；
  SEARCHER_BENCH v2 schema。原始产物：`data/p10_calibration_engineering.json`
  （git 未跟踪，数字全文抄录于此）+ `logs/searcher_bench_20260901_093058.log`。

| searcher | requested | consumed | wall_s | s/eval | 终止 | best val |
|---|---|---|---|---|---|---|
| gp | 128 | 58 | 38.1 | 0.657 | proposal_stagnation（3 代无新语义类） | 0.9689 |
| tpe | 128 | 128 | 129.6 | 1.012 | budget_exhausted | 0.98 |
| random | 128 | 128 | 71.9 | 0.562 | budget_exhausted | 0.923 |
| rl | 128 | 81 | 54.8 | 0.676 | steps_exhausted（4×32，bench v2 固定 split） | 0.891 |

- 峰值 RSS 2.95–3.44 GB/行；加载+门禁+首窗准备 ≈ 5.6 min（单次）；行级窗口
  准备 <1s。
- 用途限定：仅用于 §4.5 预算推导与 §6 投影对照；不产生任何搜索质量结论。
  校准中 GP 消耗 58/128（stagnation）与 TPE 满额消耗的对比为 §1.2 长度缺口
  提供了行为侧佐证（GP 候选全部 ≤6 content token，TPE 选中公式 11 content
  token）。
- 历史 v1 bench（cuda、dataset b927…、grammar v3、top_n=30）仅作"该机器曾用
  GPU"的历史参考，其数值不进入任何推导。
