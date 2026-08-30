# P8 研究生命周期统一契约（预注册）

状态：预注册（实现前，P8-01）。适用范围：P8 阶段 A–F 的全部实现 PR
（P8-02 … P8-15）。仲裁顺序：本文 > 测试 > 实现。本文本身不写任何代码、
不 bump 任何既有版本；它为后续阶段固定唯一仲裁来源。AGENTS.md §4.6 的
生命周期硬约束标注"待 lifecycle v1 激活"，在 P8-15 正式激活前不改变任何
现行行为。

术语约定（两个互不相同的 G 编号空间，禁止混用）：

- **数据资格 G1–G7**：`ashare_data/gates.py` 的 `ProductionGateRunner`
  （CLI 入口 `scripts/check_production_gates.py`）。
- **晋级门禁 G1–G6**：`ashare_model/promotion.py` 的 `evaluate_challenger`
  （data/formula P0、统计显著性、超额收益与风险、成本与容量压力、paper
  window、数据可信度 tier）。

## 0. 问题陈述

现状（逐项经代码勘察确认）：

1. 没有 RunSpec/预注册身份：`spec_id`、`artifact_id` 概念不存在；ledger 的
   `run_id` 是 `run-{timestamp}` 时间戳串，只存在于 ledger 与 protocol
   artifact 的嵌套 `ledger` 块；strategy/backtest 产物与 paper state 完全
   不携带 run 身份；四类边界产物（strategy/protocol/backtest/paper state）
   均无 git commit、无内容 hash。
2. 研究状态无处可查：一个 run 处于什么阶段（数据是否合格、搜索是否准入、
   OOS 是否锁定、paper 是否完成）只能靠人工翻 `data/*.json` 和运行时内存
   判断，没有 append-only 事件来源，也没有可重放的状态。
3. 证据不落地：`ProductionGateRunner` 的 G1–G7 结果是内存冻结 dataclass，
   从不持久化、从不 hash（`research_doctor` 仅在显式 `--output` 时写一份
   无 schema 版本的报告）；RL admission verdict 只被
   `scripts/admission_experiment.py` 写入 `experiments/`，没有任何生产代码
   消费它做准入。
4. paper 线最弱：`data/sim_portfolio_state.json` 无 spec/candidate/account/
   dataset/formula/config 任何身份；resume 对 legacy strategy verdict 与
   `portfolio_config` 漂移仅 warning 后继续，`dataset_id` 从不比对；
   `equity_history` 静默截断到最近 10000 条；没有观察数据单调扩展或历史
   分区防改写检查；没有 retire/promote 转换。
5. 缺失机器检查：没有"预期交易日 vs 实际交易日"的缺失交易日门禁；loader
   的 fundamentals/capital frame 降级只写日志，不进入任何类型化产物字段。

后果：跨 run 拼接结论、错误账户 resume、证据事后改写都只能靠人工纪律防
范。本契约把上述每一项都换成机器可查的裁决。

## 1. 范围与非目标

**范围**：统一生命周期状态集、合法转换、证据矩阵、身份层次、资格门禁的
精确机器标准、受保护 OOS 隔离、paper 观察数据单调扩展、旧产物与 paper
account 迁移/拒绝策略、激活边界。P8-02…P8-15 的实现 PR 以本文为仲裁来源。

**非目标**：本 PR 不改任何代码与版本常量；不实现状态机/RunSpec/RunStore；
不改变训练、评估、晋级、paper 的现行行为；不迁移或改写任何既有产物；不
预写任何测量结果；不引入新依赖；不重复定义既有单一权威（版本分类、门禁、
registry、hash）——只引用它们。

## 2. 身份层次

**唯一实现**（P8-02 落地 `ashare_model/identity.py`）：全部内容身份使用同
一个 canonical JSON + SHA-256 实现——UTF-8、键排序、紧凑分隔符、
`ensure_ascii=False`；**内容身份模式**下拒绝 NaN/Infinity/JSON 之外类型
（fail-closed，禁止静默转换）；哈希输入必须带 kind 域分隔
（`runspec`/`candidate`/`artifact`），防止跨类型替换。哈希输出为小写 hex
全长 SHA-256。ledger 现有条目 hash 的历史字节行为（含非有限浮点的既有序
列化行为）作为**同一实现**的显式兼容模式保留并测试钉死——这是参数化，
不是第二套实现。`ledger.py`、`promotion.py` 改为从唯一实现 re-export，
默认输出逐字节不变。

定义（后续阶段按此实现，不得另造）：

| 身份 | 定义 | 派生方式 |
|---|---|---|
| `spec_id` | 冻结 RunSpec 的内容身份 | `H(kind="runspec", RunSpec 全部语义字段)`；显式排除 created_at、hostname、绝对路径、run_id、输出目录等非语义字段。同一语义集合 ⇒ 同一 spec_id；任一语义字段变化 ⇒ 新 spec_id |
| `run_id` | 一次执行的唯一标识 | 随机 UUID4（非内容派生）。同一 spec 的两次 run 必然不同。写入该 run 的一切 ledger/artifact/事件 |
| `candidate_id` | 候选公式在生命周期内的内容身份 | `H(kind="candidate", spec_id + canonical formula tokens + direction)`。同 spec 内公式语义唯一确定；跨 spec 不可复用 |
| `artifact_id` | 产物内容身份 | `H(kind="artifact", 产物 payload 的 canonical JSON)`。登记记录 = {artifact_id, artifact_type, schema 版本, spec_id, run_id, candidate_id?, 路径, git commit, created_at} |
| `account_id` | paper 账户持久标识 | 随机 UUID4，一次创建终身不变；绑定 spec_id/candidate_id/observation lineage（P8-09） |

与既有字段的衔接裁决：

- 既有 ledger `run_id`（`run-{timestamp}`）：P8-04 引入
  `LEDGER_SCHEMA_VERSION` 起，lifecycle-bound ledger 条目的 `run_id` 字段
  携带本表定义的 UUID4；历史条目值保持原样只读。
- 既有 strategy artifact 的 `candidate_id`（`rl:<tokens>` 等搜索器内部
  标签）：自 P8-05 起，正式产物的 `candidate_id` 字段统一携带本表的生命
  周期身份；搜索器内部标签降级为诊断字段。legacy 产物的历史值保留只读。

## 3. 生命周期状态与机器证据矩阵

**状态机实例 = 一次 run（spec_id, run_id）**。事件追加于
`runs/<spec_id>/<run_id>/lifecycle.jsonl`（P8-04 目录布局），append-only、
hash-chained（沿用 `ledger.py` 的链式校验机制语义）。**当前状态只能由
事件重放计算得出**；不存在作为权威的 status JSON，禁止直接改写任何
status 表示；spec 级"campaign 视图"只是跨 run 的只读投影。状态集恰为
以下 11 个规范标识（P8-06+ 的枚举值必须逐字使用）：

`IDEA, SPEC_LOCKED, DATA_QUALIFIED, FACTOR_SET_QUALIFIED, SEARCH_PLAN_ADMITTED, OOS_QUALIFIED, PAPER_OBSERVING, PROMOTED, REJECTED, FAILED, RETIRED`

证据矩阵（每行"必须消费的机器证据"为该状态进入的唯一合法依据；证据必
须是 hash 已登记的类型化 artifact/事件负载，不是 console 输出）：

| 状态 | 语义 | 必须消费的机器证据 | 证据生产者 |
|---|---|---|---|
| `IDEA` | 研究假设已记录 | IdeaRecord（类型化）：假设、范围、经济机制、非目标；hash 登记 | P8-04 RunStore |
| `SPEC_LOCKED` | 预注册冻结 | frozen RunSpec artifact（`RUNSPEC_SCHEMA_VERSION = 1`）+ spec_id + clean Git commit + dependency lock hash | P8-03 RunSpec factory |
| `DATA_QUALIFIED` | 数据资格通过 | DataQualificationReport：dataset ID（`resolve_dataset_id`）、数据截止日（manifest `date_range` 上界）、预期与实际交易日、缺失交易日列表、G1–G7 原始 `GateCheck` 结果 + hash、coverage、degraded 状态与原因 | `ashare_data.gates` + manifest（复用；报告持久化为新增） |
| `FACTOR_SET_QUALIFIED` | 因子集资格通过 | FactorQualificationReport（§5.2 七项检查的原始结果 + hash） | feature_registry / data_tier / research_domain / baseline_harness（复用） |
| `SEARCH_PLAN_ADMITTED` | 搜索计划准入 | SearchAdmissionEvidence（§5.3） | search_contract / admission（复用；RL verdict 封装引用） |
| `OOS_QUALIFIED` | OOS 资格通过 | OOSQualificationEvidence：未 taint 的锁定 OOS 窗口 + lock hash、预注册统计/成本门 resolved 阈值 | P8-08（统计与成本口径复用 eval_corrections/promotion） |
| `PAPER_OBSERVING` | paper 观察中 | spec_id + candidate_id + account_id 绑定事件、observation lineage 初始化、公式/方向/费用/资金/execution/constructor hash 固定 | P8-09 |
| `PROMOTED` | 晋级完成 | 完整 paper window、晋级门禁 G1–G6 全部通过、strategy/protocol/backtest/paper 全链路 lineage 一致 | `promotion.evaluate_challenger`（复用） |
| `REJECTED` | 研究门完整证据但未通过 | 对应门禁的完整证据 artifact + fail verdict + 未过条款标识 | 各资格门禁 |
| `FAILED` | 基础设施、资源或运行故障 | 故障事件：error class、阶段、资源指标、日志引用；**无**完整门禁证据 | operational handler |
| `RETIRED` | 被替代或停止使用 | 退役事件：原因、可选替代 spec_id；既有证据只读保留 | administrative |

## 4. 合法转换与终态判别

转换守卫由 `LIFECYCLE_CONTRACT_VERSION = 1` 的状态机（P8-06
`ashare_model/lifecycle.py`）执行：**只接受下列枚举边，任何其他边、跳级、
身份错配（事件负载的 spec_id/run_id 与 ledger 归属不符）一律 fail-closed
拒绝**。`->` 为机器可读边记法：

```text
IDEA                 -> SPEC_LOCKED
IDEA                 -> FAILED | RETIRED
SPEC_LOCKED          -> DATA_QUALIFIED
SPEC_LOCKED          -> REJECTED | FAILED | RETIRED
DATA_QUALIFIED       -> FACTOR_SET_QUALIFIED
DATA_QUALIFIED       -> REJECTED | FAILED | RETIRED
FACTOR_SET_QUALIFIED -> SEARCH_PLAN_ADMITTED
FACTOR_SET_QUALIFIED -> REJECTED | FAILED | RETIRED
SEARCH_PLAN_ADMITTED -> OOS_QUALIFIED
SEARCH_PLAN_ADMITTED -> REJECTED | FAILED | RETIRED
OOS_QUALIFIED        -> PAPER_OBSERVING
OOS_QUALIFIED        -> REJECTED | FAILED | RETIRED
PAPER_OBSERVING      -> PROMOTED
PAPER_OBSERVING      -> REJECTED | FAILED | RETIRED
PROMOTED             -> RETIRED
```

- `REJECTED`、`FAILED`、`RETIRED` 为终态，无出边（PROMOTED 唯一出边是
  `-> RETIRED`）。`RETIRED` 可从任何非终态进入（管理动作），`FAILED` 同理
  （运行故障）。
- 转入 `REJECTED` 的前提是**该阶段资格门禁的完整证据已生成且判定为
  fail**（`IDEA`/`SPEC_LOCKED` 无研究门禁，不得进入 `REJECTED`，只能
  `FAILED`/`RETIRED`）。

**REJECTED / FAILED / RETIRED 的机器判别（禁止事后人工归类）**：

- 判别器 = 门禁评估是否**完成**。评估过程产出了完整 verdict（该门禁的所
  有检查项完成并给出判定）且判定 fail ⇒ `REJECTED`；评估未完成（异常、
  超时、中断、输入不可用、进程被杀）⇒ `FAILED`。完成性由事件负载中的
  completed 标志 + 证据完整性（各检查项结果齐全且 hash 匹配）机器校验。
- `RETIRED` 是管理/时序裁决（被替代、停用），不是对 alpha 的研究结论；
  与前两者语义不可互换。
- 重试规则：`FAILED` 允许同一 spec_id 用**新 run_id** 重试（新 lifecycle
  链）；`REJECTED` 是研究裁决，同 spec 重跑必然复现，重试必须以修订语义
  后的新 RunSpec（新 spec_id）进行。任何情况下不得在同一 run 的 ledger
  中把 `FAILED` 改写为 `REJECTED`（反之亦然）。

**激活分期**（未激活的边在状态机上 fail-closed，不得静默放行）：

| 阶段 | 激活的边 |
|---|---|
| P8-06 | `IDEA -> SPEC_LOCKED`、`SPEC_LOCKED -> DATA_QUALIFIED`、各态 `-> FAILED`、各态 `-> RETIRED` |
| P8-07 | `DATA_QUALIFIED -> FACTOR_SET_QUALIFIED`、`FACTOR_SET_QUALIFIED -> SEARCH_PLAN_ADMITTED` |
| P8-08 | `SEARCH_PLAN_ADMITTED -> OOS_QUALIFIED`、各门禁态 `-> REJECTED` |
| P8-09 | `OOS_QUALIFIED -> PAPER_OBSERVING`、`PAPER_OBSERVING -> PROMOTED`、`PROMOTED -> RETIRED` |

## 5. 资格门禁的精确机器标准

阈值数字一律来自 RunSpec 的 resolved thresholds（预注册），本契约只固定
检查的结构与数据来源，禁止在实现中硬编码第二份阈值表。

### 5.1 Data Qualified（P8-06/07）

`DataQualificationReport`（类型化 schema）**必须复用
`ProductionGateRunner`**，不得复制 G1–G7 任何判定逻辑。报告至少含：
spec_id、dataset ID、data cutoff、预期交易日（日历推导）与实际交易日、
缺失交易日列表、G1–G7 每项原始 `GateCheck(name, ok, detail)` 结果 + 整体
hash、coverage、degraded 状态与原因（含 loader frame 降级——现状只写日
志，P8-06 起必须进入类型化字段）。

verdict 机器规则：

- pass：formal 模式 G1–G7 全部 ok，且缺失交易日列表为空（每个缺失日都由
  日历证明为非交易日的情形不算缺失），且无未披露 degraded。
- **dev/degraded 不能 Data Qualified**：`--dev` 模式结果（`degraded=True`）
  或任何降级未消除 ⇒ 拒绝转换。
- 任一 G fail（结果完整）⇒ `REJECTED`；G 无法完成运行 ⇒ `FAILED`。

G1–G7 结果持久化（含 mode、min_eligible、`UniverseContractStatus` 摘要、
dataset_id、git commit）是该转换的必要证据；现状"结果只存在于内存/stdout"
必须在 P8-06 修复。

### 5.2 Factor Set Qualified（P8-07）

`FactorQualificationReport`（类型化 schema）至少检查，全部通过才允许转
换：

1. **单一注册表**：特征集合 ⊆ `FEATURE_METADATA`/`FeatureRegistry` 键集，
   每个特征带 registry 元数据（`semantic_type`、`pit_level`、tier、
   availability）；存在任何第二份特征名单即拒绝。
2. **PIT availability 完整**：每个特征 PIT 级别已解析；`SNAPSHOT`/`NEUTRAL`
   级特征仅当 RunSpec 数据 tier 策略显式允许时可用。
3. **promotion permission**：参与可晋级候选的特征 `promotion_allowed` 为
   真（registry 权威），否则该因子集不可用于晋级路径。
4. **research domain/horizon 一致**：每个特征 `domain_of_feature` 与
   RunSpec 研究域一致；spec 的 frequency/horizon 通过
   `ResearchDomain.is_legal_execution`；`unified` 仅用于兼容/基线。
5. **覆盖率与分布**：逐特征 coverage ≥ spec resolved threshold；分布检查
   （退化方差/缺失率/staleness）结果入报告。
6. **简单基线与消融**：报告内嵌 `baseline_harness` 简单基线对照与
   leave-family-out 消融结果（同一评分器），各自带 hash。
7. **数据 tier 允许**：因子集最弱 tier（`formula_data_tier_report` 语义）
   在 RunSpec tier 策略允许范围内；C tier 特征按 `data_tier.py` 既有规则
   不得成为晋级输入。

任一项 fail（证据完整）⇒ `REJECTED`；检查无法运行 ⇒ `FAILED`。

### 5.3 Search Plan Admitted（P8-07）

`SearchAdmissionEvidence`（类型化 schema）至少检查：

1. **backend 合法**：searcher ∈ RunSpec admission policy 的合法集合；未注
   册 backend 拒绝。
2. **domain/预算 tier/版本匹配**：`search_contract_version`、grammar、
   research domain、预算 tier 与 RunSpec 一致。
3. **长度与预算口径一致**：GP/TPE/Random/RL 共享同一 max_formula_len
   （含 EOS 口径，`SearchRequest` 校验语义复用）与同一唯一语义评价预算
   计数（budget = unique semantic evaluations；steps×batch==budget 规则复
   用 `search_contract` 既有校验）；禁止任何后端持私有口径。
4. **seeds 锁定**：seed 集预注册并配对（`paired_seed_plan` 语义），证据内
   记录 plan hash；运行期不得换 seed。
5. **语义合法性不可经配置关闭**：admission 拒绝任何试图关闭语义检查/类型
   合法性/dedup 的配置；不存在此类开关，新增此类开关本身就是契约违反。
6. **RL admission 复用**：RL 的准入证据 = 封装引用既有
   `decide_p4_admission` + `apply_p4_tier_gate` verdict（含
   `rule_version`、`promotion_blockers`），**禁止复制其判定逻辑**；verdict
   缺失或含 blocker ⇒ 拒绝准入。

## 6. 受保护 OOS 与 PR 开发反馈隔离

- OOS 窗口在 RunSpec 中 **write-once 锁定**；锁定后不得查看、筛选或修改
  模型，任何解锁访问永久标记 tainted（AGENTS §4.4 的生命周期化）。
- `OOS_QUALIFIED` 的前提：窗口锁定且 `taint=False`；multiplicity 校正、
  统计显著性与成本门以 RunSpec resolved thresholds 机器评估；受保护 OOS
  窗口 tainted ⇒ 永久拒绝 `OOS_QUALIFIED`。
- **隔离规则**：受保护 OOS 的评估入口要求 lifecycle 状态允许（进入
  `SEARCH_PLAN_ADMITTED` 之后）且运行类型 ∈ {research, promotion}；评估
  输出只能写入 lifecycle-bound 的 research/promotion evidence artifact
  （携带 spec_id/run_id/candidate_id/hash）。**禁止**进入：常规 PR 的开
  发反馈（CI 指标、开发 dashboard、测量日志 dev 节、迭代调参循环）或任
  何非 evidence 遥测。engineering 运行引用受保护 OOS ⇒ 写入时拒绝。
- Reward↔OOS Spearman 只出现在 research/promotion evidence，不作为反复
  可见的开发指标。

## 7. Paper 观察数据单调扩展

- 观察数据 **只允许追加**：observation lineage 为 hash 链，每次追加记录
  {dataset_id, [start, end], manifest hash, previous lineage hash, 预期交
  易日数, 缺失日报告}，形成 `OBSERVATION_LINEAGE_VERSION = 1`（若改为扩
  展 DatasetManifest 则按其归属 bump `MANIFEST_VERSION`，由 P8-09 裁决，
  二者取一）。
- **机器单调证明**（append 前校验，全部满足才允许）：新窗口 == 旧窗口 ∪
  [old_max+1, new_max]，无缝隙；历史分区 hash 逐字节不变；缺失日由日历证
  明为非交易日。任一不满足 ⇒ 拒绝追加并冻结账户（resume 拒绝），违规事
  实写入 lifecycle 事件；**历史分区改写即拒绝**，无 warning-continue 路
  径。
- paper events、orders、trades、equity 全部引用 account_id 与 lineage；
  lifecycle-bound 账户的 equity/events 历史完整保留于 RunStore（现状
  `equity_history` 截断 10000 条的行为仅允许存在于 legacy state 文件）。
- **paper window 完成之前不得生成 promotion evidence**；`PROMOTED` 的
  paper window 完整性 = 预注册窗口与 lineage 链机器比对一致。

## 8. 旧产物与 paper account 策略

- current/legacy/unknown 判定沿用既有单一权威：
  `artifact_schemas.classify_schema_version`（schema 维度）+
  `artifact_versions.classify_*`（研究版本维度）；P8-05 在**同一权威**上
  增加 lineage 维度（spec/run/candidate 引用缺失或不匹配 ⇒ 拒绝组合），
  禁止出现第二套分类实现。
- **v1/v0 artifact**：只读审计；不参与 promotion/resume；禁止自动改写。
- **v1 paper account 默认只读拒绝**（不自动迁移）。仅允许两条显式管理路
  径：(a) **Retire**——标记 `RETIRED`，证据保留；(b) **精确重建迁移**——
  仅当机器证明能重建逐字节相同的 RunSpec（得到相同 spec_id）且
  full-replay == segmented-resume parity 通过时，才可将历史证据迁移为
  current 版本：迁移产出**新** artifact、绝不覆盖 v1 原件、必须由显式操
  作命令触发并写入 lifecycle 事件。无法精确重建 RunSpec 的账户不得
  resume，只能 retire。禁止任何自动/隐式迁移。
- run_sim 现存 warn-and-continue 路径（legacy strategy verdict、
  `portfolio_config` 漂移、`dataset_id` 从不比对）对 lifecycle-bound 账
  户必须改为 fail-closed 拒绝（P8-08/09 范围）；legacy 账户维持现状只读。

## 9. 激活边界与 formal run 行为

- **激活前**（P8-02…P8-14 逐阶段落地）：既有 formal 入口（train/protocol/
  promotion/paper）默认行为不变；新机制只通过新入口生效；每个阶段 PR 自
  行声明其 enforcement 范围（如 P8-05 起 formal reader 拒绝非 current
  schema）。任何阶段不得留下隐藏 bypass 分支。
- **激活点 = P8-15**：AGENTS §4.6"待激活"改强制；`research_doctor` 检查
  RunSpec/lineage/lifecycle/ledger/paper；正式 CLI/Web/API 全部无 bypass；
  生成 `docs/p8_measurement_log.md`。
- **跨激活 run 的处置**：激活前开始、激活后结束的 formal run 无 lifecycle
  事件链，其产物按 artifact 规则归类（相对 lifecycle-bound 读取者为
  legacy），激活后不得用于晋级；需要继续时以新 run_id 重跑。
- 激活后 `RETIRED` 的证据永久只读保留。

## 10. 版本影响

本阶段（P8-01）：**无代码、无任何 `*_VERSION` 变化**。既有版本保持：

| 常量 | 当前值 | 本阶段 |
|---|---|---|
| `PROTOCOL_VERSION` / `REWARD_VERSION` / `MODEL_VERSION` / `GRAMMAR_VERSION` | "25" / "14" / 3 / 3 | 不变 |
| `FEATURE_REGISTRY_VERSION` / `DATA_TIER_VERSION` / `RESEARCH_DOMAIN_VERSION` / `SEARCH_CONTRACT_VERSION` | 3 / 1 / 1 / 2 | 不变 |
| `ARTIFACT_SCHEMA_VERSION` / `EXECUTION_SPEC_VERSION` / `PORTFOLIO_CONSTRUCTOR_VERSION` / `MANIFEST_VERSION` / `ADMISSION_RULE_VERSION` | 1 / 2 / 1 / "1" / 2 | 不变 |

前瞻常量（**由各自阶段 PR 定义与 bump**，本表仅为占位声明，不构成实现）：

| 常量 | 引入阶段 | 初值 |
|---|---|---|
| `RUNSPEC_SCHEMA_VERSION` | P8-03 | 1 |
| `LEDGER_SCHEMA_VERSION` | P8-04（ledger 现状无版本） | 1 |
| `ARTIFACT_SCHEMA_VERSION` 1→2 | P8-05 | 2 |
| `LIFECYCLE_CONTRACT_VERSION` | P8-06 | 1 |
| `OBSERVATION_LINEAGE_VERSION`（或 `MANIFEST_VERSION` bump，P8-09 裁决其一） | P8-09 | 1 |

## 11. 本阶段预期 RED 测试

`tests/test_p8_lifecycle_contract_doc.py`（结构守卫，本契约的可执行投影）：

1. 契约文档存在；
2. 11 个规范状态标识全部出现；
3. §4 机器可读转换块包含全部 15 条合法边（空白归一化后逐条断言）；
4. 五个身份名称（spec_id/run_id/candidate_id/artifact_id/account_id）齐
   备；
5. append-only + 事件重放 + `LIFECYCLE_CONTRACT_VERSION = 1` 声明在文；
6. 八个裁决章节标题在文；
7. AGENTS.md 引用本契约且含"待 lifecycle v1 激活"标记。

RED 证据（实现前实测）：`python -m pytest -q
tests/test_p8_lifecycle_contract_doc.py` → **7 failed**，全部为预期失败
（文档缺失 ×6、AGENTS 待激活块缺失 ×1），无无关失败。

## 12. 验证、资源与停止条件

验证命令（在精确候选 commit 上运行）：

1. 聚焦：`python -m pytest -q tests/test_p8_lifecycle_contract_doc.py
   tests/test_registry_docs.py`（相邻 docs drift guard）；
2. 全量：`python -m pytest -q tests`；
3. `python -m compileall -q ashare_data ashare_model ashare_portfolio
   ashare_trading scripts webapi`；
4. `git diff --check`。

本阶段专项 gate = 契约结构守卫测试 + `git diff --check`。不变量：全量
pytest 相对 base（6099e9f）passed 只增不减；无新增 warning/skip/xfail；
既有测试零修改。资源：不新增任何运行（纯文档 + 守卫测试）。

**停止条件**：

1. 任一必须裁决无法用机器可查标准表达（仍需"人工看 JSON 判断"状态或迁
   移）；
2. 本契约与既有单一权威（`artifact_versions`/`artifact_schemas`、
   `ProductionGateRunner`、`ledger` hash、`admission` verdict）发现不可
   调和冲突；
3. 全量测试出现无法归因于本变更的回归。

## 13. Retirement

本阶段不退役任何代码路径。前瞻退役义务（由对应阶段执行，本文预先登记以
防漂移）：ledger `run-{timestamp}` 命名与无版本 ledger（P8-04）；strategy
artifact 搜索器内部 `candidate_id` 作为正式身份（P8-05）；run_sim 的
warn-and-continue resume 路径与 equity 10000 条截断（P8-09，lifecycle-
bound 账户）；`TARGET_CONTRACT_VERSION`、`REBALANCE_POLICY_VERSION` 等
无消费者常量（记录为后续独立小任务，不在 P8 范围内顺手处理）。
