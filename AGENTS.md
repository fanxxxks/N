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

## 10. 测试与回归

### 10.1 测试原则

- 修复缺陷先写能稳定复现的 RED 测试，再实现 GREEN；测试必须断言业务不变量，不得照抄实现。
- 新功能至少包含正常、边界、失败、legacy/mismatch 和确定性路径。
- 不得删除测试、弱化断言、扩大 tolerance、增加 skip/xfail/retry 或吞异常来获得绿色结果。
- 固定随机 seed 并记录；统计测试要避免只验证合成的“必然成功”样本。
- 警告是回归指标：不得新增未解释 warning，不得用全局 filter 隐藏问题。
- 真实 DuckDB 测量与单元测试分离；PR 测试不得依赖网络或修改正式数据库。

### 10.2 按风险验证

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

若受时间、硬件或只读要求限制不能运行某项验证，必须明确列出“未运行”及原因，不能暗示已通过。

## 11. 文档、测量与版本一致性

1. 预注册契约写“计划做什么”；测量日志只写“实际发生什么”，二者不得混写。
2. 测量日志至少记录：命令、commit/tree 状态、环境、dataset ID、配置 hash、预算、seed、
   passed/skipped/warnings、墙钟/内存、原始产物路径、失败和裁决。
3. 文档中的 current 版本、默认搜索器、数据范围和测试计数必须从代码或最新证据核对；禁止复制
   已过时数字。能生成的表和版本应从注册表自动生成。
4. 代码、测试、契约、迁移说明和相关文档应在同一语义提交更新。
5. 不得把 README、onboarding 或 dashboard 的友好描述当成正式证据来源；原始 artifact/ledger
   和版本化代码是事实来源。

## 12. 并行开发规则

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
- 通过删测试、弱断言、skip、扩大 tolerance、吞异常或隐藏 warning 获得绿色；
- 正式路径静默降级、宽泛捕获异常后继续、或缺少 provenance 仍 fail open；
- 无假设、无 PIT 审计、无增量 OOS 的因子/算子扩充；
- 在巨型模块继续堆职责，或一个 PR 同时包含机械重构与语义变化；
- 文档宣称的版本、数据范围、测试结果或收益与原始证据不一致。

安全替代路径是：停止晋级/运行，保留失败证据，补契约或属性测试，使用独立版本和 run ID 重跑，
并诚实报告未验证项。

## 14. Definition of Done

任务只有在以下条件满足后才可称为完成：

- 范围内实现完成，非目标未被顺手扩张；
- 相关不变量有测试，缺陷修复有回归测试；
- 所需目标/全量/golden/gate/build 验证已运行并记录，或明确说明未运行项；
- 研究语义变化已预注册、版本化并有迁移/拒绝策略；
- 新产物 provenance 完整，失败和 legacy 未被覆盖；
- 文档与代码一致，无过期 current 声明；
- `git diff` 只包含本任务文件，没有覆盖用户已有修改；
- 最终汇报区分：软件正确性、工程测量、研究证据、生产/实盘就绪度。

任何收益研究都必须允许最终结论为“未发现可晋级 alpha”。任何工程任务都不得以“测试很多”
替代可维护的边界、单一语义路径和可复现证据。
