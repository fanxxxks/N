# P4 四搜索器统一与 Transformer 改造契约

状态：预注册（实现前）
适用版本：`SEARCH_CONTRACT_VERSION = 1`、`ELITE_ARCHIVE_VERSION = 1`、
`RL_DIAGNOSTICS_VERSION = 1`、`IMITATION_VERSION = 1`、
`ADMISSION_RULE_VERSION = 2`、`PROTOCOL_VERSION = 23`、
`MODEL_VERSION = 3`、`SEARCHER_BENCH_VERSION = 2`。

本文是 P4 测试断言的来源。仲裁顺序为本文/需求、测试、实现；测量结果不得
反向改写本契约。

## 1. 范围与不变量

P4 只改公式搜索与 Transformer 实验链路，不重构数据、因子、候选评分、
组合或交易模块。下列不变量必须在改动前后成立：

1. GP 仍是生产默认搜索器；RL 或 imitation 失败不得影响 GP 的导入、配置、
   搜索或产物写入。
2. GP、TPE、Random、RL 的一次配对比较共享同一 dataset id、fold/window、
   vocabulary、候选评分器、reward/protocol/execution 版本和请求预算。
3. 唯一语义公式评价仍是预算单位；无效、退化、canonical duplicate 和
   semantic duplicate 不消耗预算。
4. 搜索器不得超过请求预算。实际消耗可因重复、停滞或固定 RL step 结束而
   小于请求预算，差额必须显式记录，不能伪装成满预算。
5. 主链路不依赖 Transformer：`model.searcher: gp|tpe|random` 不加载 elite、
   不做 imitation、也不创建 policy checkpoint。

## 2. 统一搜索契约

四个后端均实现 `SearchBackend.search(SearchRequest,
SemanticBudgetEvaluator, **backend_context) -> SearchResult`。注册名严格为
`gp`、`tpe`、`random`、`rl`；`model.searcher` 接受且只接受这四个值。

`SearchRequest` 至少包含：

- `seed`：本次配对的独立种子；
- `budget`：请求的唯一语义评价预算，正整数；
- `max_formula_len`：公式最大长度，至少 2；
- 可选的 `steps`、`batch_size` 必须同时给出，且乘积等于 `budget`。

`SearchResult` 是四后端唯一合法的运行结果，至少包含：

- `contract_version`、`backend`、`seed`；
- `requested_budget`、`consumed_budget`；
- `termination_reason` 和可空的 `stagnation_reason`；
- `best_so_far`，元素为 `(累计实际预算, best validation reward)`；
- `scores`、`selected`、按原因聚合的 `rejection_reasons`；
- `proposal_count`、`invalid_proposals`、`semantic_duplicates`；
- `elite_archive` 和后端诊断 `diagnostics`。

合法终止原因只有：`budget_exhausted`、`steps_exhausted`、
`proposal_stagnation`、`candidate_pool_exhausted`、`no_eligible_candidate`、
`backend_error`。`proposal_stagnation` 必须给出非空具体原因，其余原因不得
伪称停滞。曲线的预算坐标严格递增且不超过 `consumed_budget`，reward 单调
不减；`consumed_budget == 0` 时曲线必须为空。

后端异常由比较/benchmark 边界转成 `backend_error` 结果或失败行并记录异常
类型；生产 GP 入口不捕获并吞掉自己的错误。

## 3. 预算、曲线、日志与指标

每次搜索至少写两条结构化日志：`search.start`（backend、seed、requested）
和 `search.stop`（backend、consumed、termination、stagnation、best）。
benchmark v2 逐后端保存完整 `SearchResult` 的预算、停滞和曲线字段，而不是
只保存一个最终 reward。

RL 每一步必须记录以下可序列化指标，运行结果还需给出 run-level 聚合：

- reward 的 count/min/q25/median/q75/max/mean/std；
- 候选拒绝原因计数；
- mean entropy；
- semantic duplicate 数量和比例；
- normalized/clipped advantage 方差；
- update 前的全参数 L2 gradient norm；
- 公式长度 count/min/mean/max；
- 本步和全运行的算子名称覆盖。

这些指标只增加可观测性，不改变 REINFORCE 损失、reward 或候选门槛。

## 4. Elite archive 与 imitation

GP、TPE、Random 的结果各自建立 elite archive。archive 只接收：tokens 存在、
reward 有限且通过既有 eligibility gates 的候选；按 validation reward、training
reward 降序，complexity cost、稳定 token key 升序确定性排序；canonical 重复
只保留排名最高者。默认每后端最多 64 条。

合并 archive 保留来源后端并再次确定性去重/截断。持久化 schema 带
`ELITE_ARCHIVE_VERSION = 1`；未知版本必须明确拒绝，禁止猜测迁移。

Imitation 只允许消费来源集合为 `gp|tpe|random` 的非空 archive。训练采用
teacher forcing next-token cross entropy，输入从现有 PAD/BOS token 开始，公式
补齐到 `max_formula_len`；必须报告初末 loss、token accuracy、epoch、样本和
token 数。进入 RL 前先完成 imitation，并重新创建 RL optimizer，避免把
pretrain optimizer state 隐式带入比较。

随机初始化 RL 仅作为显式实验臂；生产默认仍为 GP。archive 缺失/空/版本不符
时 imitation RL 明确失败，不静默退化成随机初始化 RL。

## 5. 配对种子与预注册晋级

P4 使用至少五个互不相同的 `PAIR_SEEDS`。每个 pair 中，GP、TPE、Random、
随机初始化 RL、imitation RL 使用同一个 pair seed 和相同的第 1 节 provenance；
不同 pair 使用不同 seed。禁止在五行中重复一次固定 GP/TPE/Random 结果。

所有后端都报告请求与实际预算；best-so-far area 按相同的请求预算积分，提前
停滞时把最后 best 持有到请求预算末端。OOS active IR 使用同一 fold test window。

`ADMISSION_RULE_VERSION = 2` 的规则在运行前固定：imitation RL 必须同时相对
随机初始化 RL 和 GP：

1. best-so-far area 的配对中位数严格更高；
2. OOS active IR 的配对中位数严格更高；
3. 两个指标分别至少赢 80% 的 pair（五个 pair 即至少四胜）。

全部条件成立才有 `rl_admitted = true`，并且才有
`advanced_rl_allowed = true`。否则默认搜索器保持 `gp`，且 P4 不实现/启用
PPO、辅助价值预测或 AST-aware embedding。TPE 和 Random 同预算结果必须完整
报告，但不降低上述 GP 晋级门槛。

只有精确使用预注册的 `ADMISSION_STEPS`、`ADMISSION_BATCH` 和
`ADMISSION_WINDOW` 才具备生产晋级资格。CLI 的较小预算或窗口覆盖属于工程层；
即使指标规则碰巧通过，也必须记录 `metric_rule_passed`，同时强制
`rl_admitted = false`、`advanced_rl_allowed = false`、默认 `gp`，并给出
`non_registered_admission_tier` blocker，禁止把 smoke 冒充正式准入。

## 6. 版本与旧产物策略

- 搜索比较、预算曲线与准入语义变化：`PROTOCOL_VERSION 22 -> 23`。
- Transformer 增加可审计 imitation 初始化语义：`MODEL_VERSION 2 -> 3`。
- benchmark schema 增加统一结果：`SEARCHER_BENCH_VERSION 1 -> 2`。
- v22/v2 strategy/checkpoint 继续作为历史产物，可读取公式文本做人工审计，
  但不得冒充 v23/v3 champion；checkpoint 不自动迁移，必须在 v3 下重训。
- T2-03 admission JSON 使用固定 baseline seed，不满足 paired-seed 契约，只能
  作为历史证据，明确拒绝用于 P4 晋级。
- benchmark v1 保留只读，不自动补造缺失的停滞/曲线字段，也不能用于 P4
  准入。

## 7. 验收证据

`pytest` 只证明软件回归。P4 完成还必须提交可复现测量报告，至少列出：

- 改动前/后全量测试计数、skip、warning 和 wall time；
- 同一 provenance 下四搜索器逐 seed 的 requested/consumed budget、终止/停滞、
  best-so-far area、OOS active IR；
- 随机 RL、imitation RL、GP 的配对原始值和 v2 规则裁决；
- imitation 初末 loss/accuracy；
- GP 在 RL/imitation 故障注入下仍能运行的验证。

任何 smoke 或合成数据结果都必须标为工程验证，不能宣称发现 alpha。
