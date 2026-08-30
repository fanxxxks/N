# AlphaGPT 工程与研究守则

本文件适用于整个仓库。子目录可以用更具体的 `AGENTS.md` 或
`AGENTS.override.md` 收紧规则；若发生冲突，以离目标文件最近且更严格的规则为准。

本文使用“必须 / 禁止 / 应当”表达强制要求。除非用户明确授权例外，否则不得为
赶进度、产出 champion、通过测试或降低运行成本而绕过这些要求。

## 1. 最高原则

1. **研究真实性优先于结果好看**：负结果、无 alpha、搜索器未准入和门禁失败都是
   合法结果；禁止事后放宽阈值、删失败 trial、换口径或只展示有利 seed。
2. **时间因果优先于模型复杂度**：任何数据、特征、标签、股票池和行业信息必须满足
   point-in-time 可用性；无法证明可用时间的数据不得进入正式研究或晋级。
3. **单一语义路径优先于局部便利**：股票池 mask、IC/Spearman、Reward、组合目标、
   成本和模拟成交不得出现互相漂移的第二套实现。
4. **可复现优先于运行速度**：正式结论必须能由固定数据、配置、代码提交、seed、预算
   和产物重新解释。无 provenance 的结果只可作为历史线索。
5. **可维护性优先于堆功能**：优先收缩搜索空间、复用公共契约、拆分职责和增加不变量，
   不以新增因子、算子、模型或分支逻辑替代问题诊断。
6. **测试通过不等于研究有效**：pytest 证明软件行为，真实数据测量、OOS、统计校正、
   成本压力和未来 paper window 才能支持研究或收益结论。

仓库内的仲裁顺序为：用户本次明确要求 / 已批准且仍适用的预注册契约 → 契约测试 →
实现 → 测量日志与说明文档。测量结果只能执行预注册裁决，不能反向改写契约。

未来泄漏、证据篡改、跨版本产物拼接、覆盖用户工作和为实现弱化测试属于不得以流程豁免
绕过的硬门禁。需求可以通过先修订契约而面向未来地改变，但不得据此重写已经发生的证据。
其他流程例外必须由用户明确授权，并记录适用规则、精确范围、原因、风险、补偿验证和失效条件；
沉默、赶进度或“本地看起来正常”不构成授权。

## 2. 开始任何任务前

必须先完成以下检查：

1. 读取本文件、目标目录下更近的指令文件、相关预注册契约和现有测试。
2. 执行 `git status --short --branch`，把已有修改视为用户工作；不得覆盖、清理、回滚、
   格式化或顺手修改不在本任务范围内的文件。
3. 明确任务类型：
   - 只读分析；
   - 工程实现（不改变研究语义）；
   - 研究语义变更；
   - 数据/股票池变更；
   - 运行态、晋级或模拟交易变更。
4. 写清目标、非目标、受影响契约、验证命令和停止条件。涉及多个独立阶段时先给出计划。
5. 确认现有数据、策略、协议、回测和模拟产物是否 current、legacy、缺失或互相不匹配；
   禁止默认把“文件存在”解释成“系统 ready”。

### 写入任务的授权边界与 Git 基线

1. 用户要求“实现 / 修复 / 重构 / 更新 / 落盘”时，视为自动授权在本地创建任务分支并完成
   必要的原子 commit；无需再次询问。该授权不自动包含 push、创建/更新 PR、合并回 `main`、
   发布、部署或其他外部状态变更，这些动作必须由用户明确要求。
2. 禁止直接在 `main` 开发或提交。默认从当前已核对的 `main` 创建 `codex/<task>` 分支；
   若用户指定其他分支名或起点，以用户要求为准。开始时记录 base commit、当前分支、upstream
   ahead/behind 和工作树状态。
3. 干净工作树的当前 `HEAD` 就是基线，禁止制造无意义的空 checkpoint commit。工作树不干净时，
   所有既有修改和未跟踪文件均视为用户工作；禁止自动 commit、stash、删除、回滚或顺手纳入。
   不重叠且风险可控时可以保留并继续；存在路径重叠或误提交风险时，使用独立 clean worktree，
   或停下请求用户裁决。只有用户明确授权精确路径时才可创建 checkpoint commit。
4. 一个任务分支只承载一个可独立审查、解释和回滚的变化原因。若开发期间 `main` 前进，合并前
   必须集成最新 `main`，复核冲突是否改变契约，并重新运行规定验证。

### 受控变更闭环

所有写入任务依次执行，禁止倒置门禁：

1. **基线**：记录目标、非目标、允许修改的文件、任务类型、base commit、既有用户修改、
   受影响契约/版本、预期不变量、验证矩阵和停止条件。
2. **契约**：研究语义变化先提交并批准预注册契约；其他行为变化至少先固定需求、接口契约和
   验收条件。禁止用实现当前输出反推契约。
3. **RED**：新增行为和缺陷修复先写能稳定失败的测试，记录命令、预期失败及其与契约的关系；
   RED 只能包含预期失败，出现无关失败先停下诊断。纯机械重构若无法产生有意义 RED，必须先有
   parity/golden/导入面等守卫，并说明原因。
4. **最小实现**：只修改范围内路径，复用既有公共契约和实现，不顺手重构、扩库或清理无关模块。
5. **GREEN 与回归**：按风险从聚焦测试扩展到相邻契约、全量测试和专项 gate，记录失败、skip、
   warning、耗时和未运行项。
6. **原子提交**：代码、测试、行为文档、版本变化和迁移/拒绝实现组成同一可工作的语义提交；
   在首次 push 前验证精确候选 commit，而不只验证未提交工作树。
7. **推送与合并**：仅在获得相应授权且本地门禁满足后 push/建 PR/合并；合并后在精确 merge
   commit 上重新验证组合状态。失败必须保留证据并通过新 commit/revert 处理，禁止改写历史掩盖。
8. **交付**：报告分支、base/head/merge commit、变更范围、版本、RED/GREEN、验证结果、
   迁移策略、研究声明边界以及 push/PR/merge 状态。

### 只读任务

用户要求“不做任何改动”时，只能执行确定只读的检查。以下命令可能写日志、JSON、缓存、
数据库或前端产物，未经允许不得运行：pytest、训练、协议、同步、回填、doctor、模拟、
前端 build 以及无 `--check` 的 lock 命令。应优先阅读现有日志和产物，并在结束时用
`git status` 确认工作区没有被本任务改变。

## 3. 变更分类与预注册

### 3.1 工程变更

纯重构、日志、错误隔离、类型化、性能优化和 UI 展示不得改变数据、公式、Reward、执行、
统计或晋级语义。必须用 golden/parity 或逐字节兼容测试证明默认路径不变。若无法证明，
按研究语义变更处理。

新增功能或缺陷修复即使不改变研究语义，也必须在实现前固定行为契约和验收条件，并先获得
对应 RED 证据。工程测量只能证明边界、兼容性、性能或可观测性，不得升级为研究结论。

### 3.2 研究语义变更

以下任一项变化都属于研究语义变更：

- 特征或算子定义、可用时间、grammar、公式长度或规范化；
- 标签、horizon、rebalance frequency、Reward 或候选过滤；
- 搜索空间、预算计数、seed pairing、停滞/终止规则；
- 股票池、PIT mask、数据 tier 或降级策略；
- 组合构造、费用、成交、容量、统计检验或晋级阈值。

实现前必须提交预注册契约，至少包含：问题陈述、假设、范围与非目标、不变量、版本变化、
迁移/拒绝策略、预期 RED 测试、测量方案、资源上限、裁决规则和停止条件。先固定契约，
再写失败测试，再实现；禁止看见 OOS 结果后修改通过标准。

语义变化必须提升对应版本，并在代码、artifact schema、测试、契约和测量日志中保持一致。
常见版本来源包括：

- `ashare_model/evaluation.py`：protocol；
- `ashare_model/reward.py`：reward；
- `ashare_model/alphagpt.py`：model；
- `ashare_model/vocab.py` / grammar；
- `ashare_model/feature_registry.py`：feature registry / data tier；
- `ashare_portfolio/constructor.py`：portfolio constructor；
- `ashare_portfolio/execution_spec.py`：execution；
- `ashare_model/search_contract.py`：search contract。

不确定是否需要 bump 时，默认 bump 或停下说明理由，禁止悄悄改变语义。

每个行为变化必须给出版本影响表：逐项列出相关版本的旧值、新值和 bump/不 bump 理由。
实现前应搜索仓库内实际的 `*_VERSION` 所有者和 artifact 兼容性入口，不能只依赖上述示例名单。
新增 artifact-facing 版本必须接入既有兼容性分类权威；确实不适用时必须在契约中说明。

## 4. 正式运行、证据与产物

### 4.1 三种运行不得混用

- **engineering**：验证代码路径、内存、不变量和错误隔离；不能支持 alpha 或晋级结论。
- **research**：固定假设、数据和 OOS 方案，用于判断方法是否有效。
- **promotion**：使用预注册正式预算、锁定窗口和完整门禁，只决定能否进入 paper/champion。

运行类型必须写入 artifact 和日志。低预算、裁剪窗口、合成数据或 dev gate 结果禁止标为
production、champion 或 admitted。

### 4.2 正式产物必须具备完整身份

新正式产物至少记录：

- run/campaign ID、创建时间、Git commit；
- dataset ID、数据截止日、universe policy；
- feature/operator/grammar、protocol、reward、model、search、execution、constructor 版本；
- 完整 formula tokens/text/hash 与 direction；
- frequency、horizon、portfolio config/hash、资金和费用；
- 搜索器、唯一语义预算、seed、fold/window、终止原因；
- 数据 tier / research domain、holdout 状态、失败与降级原因。

边界产物应使用 Pydantic/dataclass 等类型化 schema 校验；禁止在正式边界继续扩散未经验证的
松散 `dict` 和静默 `payload.get(..., default)`。

### 4.3 Fail closed

- dataset、formula、配置或版本不一致时拒绝组合、晋级和 resume；仅 warning 后继续不合格。
- legacy 产物可以只读展示和人工审计，但不能冒充 current champion，不能恢复到新模拟账户。
- strategy、protocol、backtest、paper state 必须来自同一 run/campaign；禁止从 `data/` 中挑选
  多个互不相干的 JSON 拼接结论。
- 所有 trial（成功、失败、崩溃、提前停止）都必须进入 append-only ledger；不得覆盖原始证据。
- 重跑使用新 run ID/输出目录；禁止覆盖失败产物或删除不利行。
- JSON/state 写入使用原子写；数据库迁移要有备份、范围审计和恢复方案。

### 4.4 Holdout 与声明

反复用于开发、调参或选择的日期不能再称为 holdout。锁定窗口后不得查看、筛选或修改模型；
解锁即永久标记 tainted。没有新的锁定 OOS 或未来 paper window 时，只能说“尚无证据”，
不得说“已验证 alpha”。

### 4.5 关键路径可观测性

数据同步/迁移、训练、搜索、评估、晋级、回测、paper simulation 和 artifact 迁移属于关键路径。
必须复用现有日志、ledger 和 artifact 体系，禁止为局部方便新增第二套互相漂移的遥测路径。

关键路径至少记录：run/campaign ID、dataset/config hash、版本集合、开始/结束/失败、停止或降级
原因、requested/consumed budget、成功/失败/重复/抑制计数、墙钟和相关资源指标。异常日志应有
足够定位上下文，但不得泄露密钥或敏感数据。新增关键分支时必须有日志/指标契约测试；console
日志和 dashboard 数字只能帮助诊断，不能替代 append-only ledger 和类型化 artifact 证据。

## 5. 数据与时间因果

1. `AshareDataLoader` 和既有 universe contract 是 PIT eligibility 的唯一正式路径；不得复制
   第二套股票池或 mask 逻辑。
2. 成分、上市/退市、财报、行业、两融和事件数据必须记录来源与 availability timestamp；
   当前快照不得回填历史。
3. 正式路径不得把缺失的 B/C tier 数据静默变成看似可用的中性值。允许降级时必须显式记录
   degraded reason，且默认不可晋级。
4. 数据或 universe 改动必须运行并记录 G1–G7；不得用 `--dev` 结果支持正式结论。
5. 数据 readiness 必须检查最近开市日、缺失交易日、覆盖率和 manifest，而不是只检查表存在。
6. 数据库同步、purge、回填和范围迁移是有状态操作：先只读审计精确范围，再执行，再复核
   row count、日期范围、dataset ID 和 gates；不得在其他正式运行占用数据库时并发写入。

## 6. 研究域、因子与算子

1. 新 campaign 必须显式声明预测域、target horizon、rebalance frequency、成本和换手纪律；
   `unified` 只用于兼容或基线，不能掩盖短/中/慢信号的语义混合。
2. 不同 horizon、Reward 或执行纪律的 `reward` / `best_reward` 不可直接横向比较。
3. 慢速基本面、日频价格量和事件信号不得在缺少预注册依据时共用一个标签与执行周期。
4. 若存在已批准的 research-domain 契约，以其显式特征划分和合法执行点为准。

新增因子前必须记录：经济假设、数据源、PIT 可用时间、tier、预期方向、预测周期、覆盖率、
换手/成本、相关簇和相对已有因子的增量 OOS。每次只增加小批量因子族，并先通过简单基线
与消融；禁止因搜索结果不好而无约束扩库。

新增算子前必须证明现有算子无法表达该经济假设，并给出语义类型、输入/输出、数值稳定性、
等价规范化、复杂度和增量测量。优先增加语义类型、模板限制、去重和剪枝，禁止仅为扩大
表达空间增加通用数学变换。

因子/算子元数据必须进入单一注册表；代码、搜索空间、artifact 和文档从注册表读取，禁止
维护第二份手工名单。

## 7. 搜索器公平性

比较 GP、TPE、Random、RL 或后续搜索器时必须共享：

- 同一数据、fold/window、research domain 和候选评分器；
- 同一有效 feature/operator vocabulary 与 semantic canonicalization；
- 同一最大有效公式长度（含 EOS 的口径一致）；
- 同一唯一语义评价预算及计数器；
- 预注册的独立 paired seeds；
- 同一 execution/portfolio/Reward 配置。

必须用属性测试直接验证真实不变量，不能把实现中的假设复制到测试注释中。例如 GP 树必须
满足 `len(tree_to_tokens(tree)) == len(tree)`，所有后端输出必须满足统一的 token 长度上限。

每个搜索结果必须记录 requested/consumed budget、proposal count、invalid、semantic duplicate、
公式长度分布、停滞原因和 best-so-far 曲线。提前停滞不得伪装成耗满预算；不同有效搜索空间
的结果不得宣称 matched comparison。

## 8. 组合与执行

1. Reward、回测、paper simulation 必须复用 `ashare_portfolio` 和
   `ashare_execution.py` 的同一构造、调仓和费用语义；禁止复制近似实现。
2. 任何相关改动必须验证：T+1、整手、停牌、涨跌停、佣金下限、印花税、过户费、滑点、
   现金上限、单股上限、权重和不超过 1。
3. golden parity 应比较逐日 target、buy/sell weight、turnover、order、cost 和 net return，
   不能只比较最终收益。
4. 资金规模是研究契约的一部分。持仓数、最小交易额、调仓频率和费用必须在目标资金下可行。
5. 同时设置组合活动门：实际调仓次数、持仓暴露、订单数、换手和受抑制交易。接近空仓或
   只建仓一次的低成本策略不得仅凭低费用取得“有效”结论。

## 9. 架构与代码组织

依赖方向应保持：

```text
ashare_data / ashare_portfolio / shared execution
                  ↓
             ashare_model
                  ↓
           ashare_trading
                  ↓
          webapi / dashboard / webui
```

底层模块不得反向依赖 CLI、Web、模拟管理或 workflow。跨层通信使用小型、明确、类型化接口。

- `evaluation.py`、`train.py` 已是高风险巨型模块：不得继续加入新的独立职责。修改相关功能时，
  优先把 folds、metrics、corrections、artifact writer、diagnostics、backend adapter 等抽成模块，
  原文件保留兼容 facade。
- 一个概念只保留一个实现：IC、universe mask、formula canonicalization、成本、组合目标、版本
  判定和 artifact classification 禁止复制。
- 新模块必须职责单一、依赖方向清楚；避免循环 import、全局可变状态和 process-global 隐式配置。
- 正式研究路径不得用宽泛 `except Exception` 静默返回中性结果；UI 可以防御性降级，但必须展示
  unavailable/legacy/degraded，不能显示成健康数据。
- 默认配置保持向后兼容。实验配置使用独立文件或显式参数，不为一次实验改变生产默认值。
- 不做与任务无关的重命名、格式化或“大扫除”；机械重构与语义变化分开提交。
- 添加生产依赖前必须说明必要性。修改 Python 依赖要同步 `.in`、pin/lock 与 CPU/CUDA 契约；
  修改 Web 依赖要同步 `package.json` 和 `package-lock.json`。

### 死代码、冗余与兼容层

- 新增模块、helper、配置项、feature flag、schema 字段或抽象必须服务当前已批准需求，具有真实
  调用方/消费者和测试，或是契约明确要求的公共边界；禁止为假想未来需求预留空壳。
- 新实现前必须搜索既有注册表、公共 helper、类型化 schema 和共享路径。能够扩展单一权威时，
  禁止复制后再以 adapter、fallback 或 feature flag 长期并存。
- 兼容层只有在明确适用版本、迁移/拒绝策略、测试和退役条件时才是有效代码；无消费者、无测试、
  永久 fallback 或永不触发的分支一律视为冗余。
- 当前任务发现的无关死代码应记录为后续小任务，不得借机清理；确需删除时使用独立、可回滚的
  机械 PR，并以引用搜索、契约测试和全量回归证明无消费者。

## 10. 测试与回归

### 10.1 测试原则

测试是需求和接口契约的可执行投影。测试断言的唯一合法来源是已批准需求、契约和独立业务
不变量，绝不是实现的当前输出。测试与实现冲突时必须回到契约仲裁，禁止默认认定测试写错。

- 修复缺陷先写能稳定复现的 RED 测试，再实现 GREEN；新功能至少包含正常、边界、失败、
  legacy/mismatch 和确定性路径。覆盖率数字只是诊断信息，不能替代对真实不变量的直接断言。
- 既有测试默认可信。曾经通过的测试在新改动后变红，第一嫌疑是新实现；认为测试有误的一方
  承担举证责任，必须引用契约原文或独立事实来源，而不是引用实现输出。
- 固定随机 seed 并记录；统计测试要避免只验证合成的“必然成功”样本。真实 DuckDB 测量与
  单元测试分离；PR 测试不得依赖网络或修改正式数据库。
- warning 是回归指标。不得新增未解释 warning 或用全局 filter 隐藏问题；比较时应同时审查
  warning 类别和调用位置，禁止用“总数未增加”掩盖新 warning 替换旧 warning。

允许修改既有测试的情形仅限以下白名单，并且必须举证：

1. 测试本身存在笔误、夹具错误或对需求理解偏差：必须引用契约说明旧断言为何错误，且修正后
   的覆盖范围、失败敏感度和断言强度不得降低。
2. 需求确实变更：必须先修改并批准契约、bump 对应版本、确定迁移/拒绝策略，再同步修改测试。
3. 不改变语义的测试重构：仅可提取 setup、改名、拆分文件或消除重复；测试集合、业务路径和
   断言强度必须保持，必要时用 collection/parity 证据证明。

任何测试 diff 都必须在 commit/PR 证据中列出所属白名单、契约条款、旧行为为何错误以及新测试
为何不更弱。以下模式出现即为阻断问题：

- “对答案”式把 expected 改成实现的 actual，或用新实现生成值作为自身 oracle；
- 将 `==` 改为 `approx`、精确匹配改为 `contains`、具体值改为 `is not None`，或扩大 tolerance；
- 删除失败断言/用例，或新增/扩大 skip、xfail、retry、timeout 来消除红灯；
- 在测试中用 `try/except` 吞掉失败，或 mock 被测对象、核心算法、关键门禁本身；
- 盲更 snapshot/golden。每个 snapshot/golden diff 必须逐项对照契约和独立权威 review；
- 通过 warning filter、随机重试、减少样本/seed/窗口或降低属性测试强度获得绿色。

既有平台/硬件能力型 skip 必须登记为基线，不得未经契约批准净增。新增硬件相关行为应在对应
运行环境建立强制验证，不能用临时 skip 代替；任何基线 skip 的变化都要在测量中逐项解释。

### 10.2 按风险验证

开发迭代可以先运行聚焦测试，但首次 push 前必须在精确候选 commit 上完整复现目标分支配置的
所有 CI-equivalent job，并叠加本节按风险要求的专项门禁；CI 中无条件触发的 job 视为适用，
不能因为本次没有修改该技术栈而跳过。记录实际 OS、Python/Node、CPU/CUDA 和依赖环境；本地
通过不替代 push 后 CI，CI 通过也不替代本地专项 gate。

项目必须显式区分最低兼容运行时、CI 运行时和正式研究参考运行时；三者不一致时，涉及语法、
数值、依赖、序列化、并发或硬件路径的改动必须在受影响运行时分别验证。支持版本变化必须同步
CI、依赖契约和 onboarding 文档，禁止只在开发机的新版本上通过就宣称兼容。

Python 改动通常按以下顺序验证：

1. 最小相关测试文件；
2. 相邻契约/golden/parity 测试；
3. `python -m pytest -q tests`；
4. `python -m compileall -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi`；
5. `git diff --check`。

数据/universe 改动另需：

- `python scripts/check_production_gates.py`；
- `python -m ashare_model.research_doctor`；
- 覆盖率、日期、manifest、dataset ID 的只读审计。

依赖改动另需：

- `python -m pip check`；
- `python scripts/freeze_lock.py --check`。

注意：`python scripts/freeze_lock.py` 不带 `--check` 会改写依赖文件，只能在明确的依赖更新任务中运行。

Web 改动在 `webui/` 中验证：

- `npm ci`；
- `npm ls --depth=0`；
- `npm run build`。

不得用 `npm install` 代替 CI 的干净锁文件验证，除非任务本身就是更新依赖。

文档/配置/生成物改动至少运行对应 drift/check 命令、链接/结构检查和 `git diff --check`；若目标
分支 CI 仍会无条件运行 Python/Web job，push 前仍须本地复现。新增或修改门禁时，AGENTS、CI、
本地验证入口和相关文档必须在同一流程变更中同步，禁止形成互相漂移的命令清单。

若受时间、硬件或环境限制不能运行某项验证，必须明确列出“未运行”及原因。首次 push 的任何
mandatory 项未运行、失败或出现未裁决的新 warning/skip 时，本地原子 commit 可以作为明确标注的
中间交付，但禁止 push、建 PR、合并或宣称“push-ready/已完整验证”。获得合并授权后，还必须在
最新 `main` 的精确 merge commit 上重跑组合验证。

## 11. 文档、测量与版本一致性

1. 预注册契约写“计划做什么”；测量日志只写“实际发生什么”，二者不得混写。
2. 测量日志至少记录：命令、commit/tree 状态、环境、dataset ID、配置 hash、预算、seed、
   passed/skipped/warnings、墙钟/内存、原始产物路径、失败和裁决。
3. 文档中的 current 版本、默认搜索器、数据范围和测试计数必须从代码或最新证据核对；禁止复制
   已过时数字。能生成的表和版本应从注册表自动生成。
4. 预注册契约必须先于实现独立提交并在批准后保持前瞻性；描述当前行为的代码、测试、API/schema
   文档、版本变化和迁移/拒绝说明必须在同一原子语义提交更新。实际测量只能在运行精确实现后
   写入，可使用后置 evidence-only commit，但必须引用被测 implementation/merge SHA，禁止预写结果。
5. 生成文档必须与来源注册表在同一提交更新，并有 drift guard；不得复制一份手工 current 表。
6. 不得把 README、onboarding 或 dashboard 的友好描述当成正式证据来源；原始 artifact/ledger
   和版本化代码是事实来源。

## 12. 串行、小 PR 与并行开发规则

默认使用串行小 PR。一个 commit/PR 只允许一个独立变化原因，而不是只允许一种文件类型；实现、
测试、行为文档、版本 bump 和该行为必需的迁移属于同一原子变化。依赖更新、机械重构、研究语义
变化、测量记录和无关清理具有不同回滚边界，必须拆开。

- 研究语义变化通常先合并预注册契约 PR，再从最新 `main` 创建实现 PR。RED 测试必须先在本地
  实际失败并保留证据，但禁止把故意红的中间提交单独合并到 `main`；最终实现提交必须自洽且绿。
- 一个小 PR 合并并完成组合验证后，下一项工作才从新 `main` 开始。禁止让后续 PR 依赖尚未合并、
  未提交或仅存在于另一个工作树的隐式状态。
- commit/PR 证据至少包含：目标/非目标、base/head、契约引用、允许文件、版本影响表、RED/GREEN、
  测试修改白名单说明、迁移/拒绝策略、完整验证结果、warning/skip、研究声明边界和回滚方式。
- 禁止为了减少 PR 数量把机械重构与语义变化、多个不相关缺陷或多个可独立回滚的功能打包。

并行只用于真正独立的工作包。开始前必须冻结共享接口并声明每条线的文件所有权、输入、输出、
验收标准和集成顺序。

- 不允许两个执行者同时修改 `evaluation.py`、`train.py`、`webapi/service.py`、
  `ashare_trading/run_sim.py`、公共配置或同一个 schema；这些热点文件由单一集成人接线。
- 并行线优先新增独立模块和测试，不提前修改共享 facade。
- 每条线使用独立分支/worktree 和单一职责提交；禁止依赖另一条线未提交的工作树状态。
- 合并前逐线验证，合并后重新运行集成与全量回归；“各分支都绿”不代表组合后绿。
- 机械拆分、schema/registry、workflow、测试基础设施可以并行；最终运行态门禁必须单点集成。
- 发现共享契约需要变化时暂停相关并行线，先更新预注册契约和接口，再继续，禁止各自发明兼容层。

## 13. Code Review Rules

以下任一项应视为阻断问题：

- 未来泄漏、当前快照历史化、PIT mask 绕过或数据降级未披露；
- 改变研究语义但未预注册、未 bump 版本或未说明迁移/拒绝策略；
- 不同 dataset/config/formula/version 的 artifact 被组合或模拟账户被继续使用；
- 工程/小预算/合成结果被表述为 alpha、production、champion 或 admission 证据；
- 搜索器的有效词表、公式长度、预算或 seed 不公平；
- 新增第二套 Reward、IC、组合、成本、mask 或 artifact 判定逻辑；
- 测试修改不属于 §10.1 白名单、缺少契约举证，或通过对答案、删测试、弱断言、skip、扩大
  tolerance、吞异常、mock 核心路径或隐藏 warning 获得绿色；
- 正式路径静默降级、宽泛捕获异常后继续、或缺少 provenance 仍 fail open；
- 无假设、无 PIT 审计、无增量 OOS 的因子/算子扩充；
- 在巨型模块继续堆职责，或一个 PR 同时包含机械重构与语义变化；
- 直接在 `main` 开发/提交，自动吸收用户既有修改，或在 mandatory 本地门禁未满足时 push/建 PR/
  合并；
- 新增无消费者/无测试的死代码、第二套实现、无退役条件兼容层，或关键路径缺少规定日志与指标；
- 文档宣称的版本、数据范围、测试结果或收益与原始证据不一致。

安全替代路径是：停止晋级/运行，保留失败证据，补契约或属性测试，使用独立版本和 run ID 重跑，
并诚实报告未验证项。

## 14. Definition of Done

任务只有在以下条件满足后才可称为完成：

- 已记录 base commit、任务分支、既有用户修改、目标/非目标和允许文件；本地实现已形成单一职责
  的原子 commit，且未直接提交到 `main`；
- 范围内实现完成，非目标未被顺手扩张；
- 相关不变量有测试，缺陷修复有回归测试；
- 当前交付范围所需的目标/golden/gate/build 验证已在精确候选 commit 上运行并记录；若交付范围
  包含 push/PR/merge，则全部 CI-equivalent mandatory 项没有失败、未运行或未裁决的新 warning/skip；
- 研究语义变化已预注册、版本化并有迁移/拒绝策略；
- 新产物 provenance 完整，失败和 legacy 未被覆盖；
- 行为文档、代码、测试、版本和迁移说明一致，无过期 current 声明；后置测量引用精确被测 SHA；
- `git diff`/commit 只包含本任务文件，没有覆盖用户已有修改，也没有死代码、冗余兼容层或第二实现；
- 若用户授权 push/PR/merge，相应状态已明确，CI 已通过且 merge commit 已完成组合复验；未获授权时
  明确报告“仅本地提交、未 push/未合并”，不得暗示已经集成；
- 最终汇报给出 branch、base/head/merge、验证命令与结果，并区分：软件正确性、工程测量、
  研究证据、生产/实盘就绪度。

任何收益研究都必须允许最终结论为“未发现可晋级 alpha”。任何工程任务都不得以“测试很多”
替代可维护的边界、单一语义路径和可复现证据。
