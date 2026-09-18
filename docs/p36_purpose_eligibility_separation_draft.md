# P36 历史研究与在线观察的资格分离预注册契约

- 状态：**APPROVED**。批准依据 = 用户 2026-09-17（Asia/Shanghai）直接指令「完成
  工作包 2：落实历史研究与在线观察的资格分离。范围：gates.py、RunSpec/运行
  上下文、相关入口和测试。用途必须显式进入运行身份。同一历史快照在不同运行日
  得到相同的历史研究资格。窗口内缺失、身份漂移、PIT 或交易约束问题仍应拒绝。
  在线观察使用过期数据仍应拒绝。先完成用途正确性。重复全库确认的性能优化可
  后置，不与本包混在一起」（AGENTS §1 仲裁顺序第一位：用户本次明确要求）。
- 起草：执行模型（本会话，用户直接指令「完成工作包 2」），2026-09-17。本契约为
  p35（同日工作包 1）之后的第 2 个直接指令工作包。
- 起草依据：
  - 用户任务指令（本文开头引文，含明确的范围、四条验收语义与后置声明）；
  - `docs/p16_data_freshness_gate_contract.md`（G8；本契约修订其评估日口径，
    见 §9）；
  - `docs/p17_tushare_pit_data_contract.md` §12（D7 滞后口径；本契约修订其
    「最近会话」参考，见 §9）；
  - `docs/p26_a02_dataset_snapshot_identity_draft.md` §4.2（RUNSPEC_SCHEMA_VERSION
    1→2 的 bump/迁移先例）、`docs/p27_a03_lifecycle_runcontext_draft.md` §8
    版本影响表第 2 行（「动作模式若进入 RunSpec 会改变 spec_id」的预期——本契约
    的 purpose 即该预见的实现）；
  - `docs/p30_a05_doctor_eligibility_draft.md` §2 不变量 6（历史/在线分离已在
    只读组合层成立：快照完整性与今日无关、在线新鲜度单独呈现；本契约把同一
    分离落到**执法层**）；
  - AGENTS.md §1（原则 2/4）、§3.2、§4.3、§5、§10.1、§11。
- 勘察基线（docs/CONTRACTS.md「勘察基线惯例」）：起草于分支
  `codex/b04-load-cache-performance` @ `588f378`（= main 上的 p35 证据链头）。
  本文全部代码事实在该 commit 上核对。

## 0. 问题陈述（逐项经代码勘察确认）

1. **资格评估日对一切用途都是墙钟**：`ashare_data/gates.py` G8 的评估日
   （p16 §2.1）是 `Asia/Shanghai` 当次调用墙钟；`ashare_model/data_eligibility.py`
   的 D7 滞后检查默认评估日同样是墙钟（`_shanghai_today()`）。于是对一份**内容
   完全一致**的冻结复合快照（A02 已确认、dataset_id 已冻结），历史研究资格随
   运行日漂移：快照冻结后第 4 个开市日起 G8 必红、D7 各源滞后必超限——同一
   快照在不同运行日得到**不同**的历史研究资格。这违反 AGENTS §1 原则 4
   （可复现）与 §4.2（正式产物完整身份）：资格是身份的函数，不是墙钟的函数。
2. **用途不在运行身份里**：`RunSpec`（`ashare_model/runspec.py`）不含任何用途
   字段；资格评估基准因此是**环境隐式**（墙钟）而非从冻结身份推导。p27 §8
   版本影响表第 2 行已预见「（动作模式）若进入 RunSpec，会改变 spec_id」——
   本契约把**用途**（而非动作模式本身）以预注册方式落入 RunSpec。
3. **组合层已分离、执法层未分离**：p30 §2 不变量 6 已在 readiness 报告层把
   「快照完整性（与今日无关）」与「在线数据新鲜度（今日相关，仅 paper 适用）」
   分开；但执法层（G8/D7 与各正式入口的门禁调用）仍把两种语义混在同一个墙钟
   评估日上：历史研究被在线新鲜度误杀，在线观察又没有任何入口级机制声明自己
   的用途基准。
4. **在线观察的正确性依赖现状偶然性**：`ashare_trading/run_sim.py` 的
   `require_production()`（墙钟）目前恰好是正确语义（过期数据拒绝），但它与
   历史研究入口调用形态完全相同——没有任何机器证据区分两者的评估基准。

## 1. 术语

- **历史研究（historical research）**：对冻结复合快照上历史窗口的正式研究与
  复现——train / evaluation / backtest / campaign / R01 基线，即 p27 动作模式
  `research` / `protected_oos` 下的正式入口动作。其资格问题：「这份冻结输入在
  它自己的时间线上是否完整合格」（p30 §2.6 的 snapshot_integrity 语义在执法层
  的对应）。
- **在线观察（online observation）**：paper 模拟对绑定账户的持续观察，即
  p27 动作模式 `paper`（`run_sim` 正式路径）。其资格问题：「**现在**的库是否
  新鲜到可以继续观察」（p30 §2.6 的 online_data_freshness 语义）。
- **用途（purpose）**：上述二者之一，数据资格评估日的**唯一**决定因素。
- **快照评估日（snapshot as-of day）**：`data_cutoff` 的下一日历日
  （`snapshot_as_of_day(data_cutoff)`，`ashare_data.gates` 唯一实现）。G8 的
  参考会话 = 严格早于该日的最大开市会话；由于 `frozen_data_cutoff` 本身是
  开市会话（data_loader.py：sessions 内 ≤ v2 日线最大日的最后一日），参考会话
  恰为 cutoff 会话，内洽快照滞后恒为 0。

## 2. 不变量（机器裁决，全部 fail-closed）

1. **用途显式进入运行身份**：`RunSpec.purpose ∈ {"historical_research",
   "online_observation"}`（闭集 Literal），必填、冻结（`frozen=True,
   extra="forbid"`）、进入 `to_payload()` → `spec_id`。`build_runspec` /
   `resolve_runtime_runspec` 的 `purpose` 为**必填参数、无默认值**——用途由
   调用方声明，绝不被静默默认（错误默认 = 错误资格基准 = 本包要消灭的缺陷
   类别）。
2. **历史资格 = 身分的纯函数**：用途为 `historical_research` 时，G8 与 D7 的
   评估日 = `snapshot_as_of_day(spec.data_cutoff)`，由冻结身份唯一决定；
   实现路径**不得读取墙钟**（测试以「把墙钟函数替换为 raise 后历史基准仍完整
   运行」证明）。同一快照在同一（或不同）运行日重算，G1–G8 与 D7 结论逐项
   相同。
3. **仍然拒绝**（本包不放松任何既有判定）：
   - G1–G7 原样；G8 在历史基准下衡量「主库日线窗口相对快照 cutoff 的滞后」，
     主库日线停在 cutoff − N 会话之前（N=`FRESHNESS_TOLERANCE_SESSIONS`）⇒
     拒绝（快照窗口末端缺 bar）；
   - 身份漂移：A02 `confirm_composite_snapshot` 的全量重哈希确认、各正式入口
     freeze→load 前后的 dataset_id 漂移检查、RunSpec 精确往返校验原样；
   - PIT：G5 严格宇宙契约、universe mask、p17 §6.1 可用性映射原样；
   - 交易约束：D7 的 `daily_price_limit` 家族滞后/覆盖阈值与 p19 §2.4 执行
     证据原样。
4. **在线观察过期仍拒绝**：`purpose="online_observation"`（含缺省）时评估日
   = `Asia/Shanghai` 墙钟——`run_sim` 正式路径的 `require_production()` 不传
   purpose/snapshot_cutoff（缺省即在线基准，语义与 p16 §2.1 完全一致），过期
   数据 ⇒ 拒绝；以测试钉住。
5. **基准混用拒绝**：`purpose="historical_research"` 而缺 `snapshot_cutoff` ⇒
   `ValueError`；`historical_research` 又传 `today` ⇒ `ValueError`（today 注入
   属在线基准测试缝）；`online_observation` 又传 `snapshot_cutoff` ⇒
   `ValueError`。缺省 purpose = `online_observation`（更严格、会拒绝的方向）：
   忘记声明用途的入口绝不可能悄悄获得历史宽限。
6. **可审计**：G8 的 detail 必须携带评估基准标记——在线：
   `basis=online_wall_clock`；历史：`basis=historical_snapshot` 且含
   `data_cutoff=...`。DATA_QUALIFIED 证据经既有 `gate_result_hash` 自然携带
   （report payload 形状不变，见 §7）。

## 3. gates.py 机器标准（唯一实现：`ashare_data/gates.py`）

1. 新增常量（用途闭集与推导的唯一权威；`ashare_model` 按既有依赖方向引用，
   不得复制第二份）：
   - `ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH = "historical_research"`
   - `ELIGIBILITY_PURPOSE_ONLINE_OBSERVATION = "online_observation"`
   - `ELIGIBILITY_PURPOSES = frozenset({...})`
   - `snapshot_as_of_day(data_cutoff: str) -> str`：校验 YYYYMMDD 真实日期，
     返回下一日历日。
2. `ProductionGateRunner.run(mode, *, today=None, purpose=ONLINE, snapshot_cutoff=None)`
   与 `require_production(...)` 同参透传：
   - 在线基准：评估日 = `today`（测试注入）或墙钟；`snapshot_cutoff` 必须为
     `None`。
   - 历史基准：评估日 = `snapshot_as_of_day(snapshot_cutoff)`；`today` 必须为
     `None`；不得触墙钟。
   - 其余 G1–G7 与既有判定逐字节一致（仅 G8 的评估日来源与 detail 追加
     basis 标记变化）。
3. **G8 的「日历预列视界耗尽」守卫仅属在线基准**：该守卫（p16 §2.2）防御
   「墙钟今天越过预列日历、无法枚举至今会话」；历史基准的参考会话（cutoff
   会话）按构造在日历内部，`reference` 缺失仍 fail-closed，但「日历最大会话
   < 评估日」不构成历史基准的失败理由（否则 cutoff 恰为日历最末会话的冻结
   快照会被误杀）。RED 测试 R3-4 钉住。
4. 不存在关闭 G8 或改写基准来源的配置面/flag（p16 §2.2「无开关」延续）。

## 4. RunSpec/运行上下文机器标准

1. `RUNSPEC_SCHEMA_VERSION` 2→3（p26 先例：必填语义字段加入 ⇒ 身份代际）。
   `RunSpec.purpose: Literal[...]`（取值 = §3.1 常量）位于「research
   semantics」字段组；进入 payload/spec_id。
2. 生产者全部显式声明 `purpose="historical_research"`：
   `train_entry.establish_formal_run_context`、`evaluation`（formal 与非正式
   两条 spec 路径——工程探针的 spec 语义仍是历史窗口研究计划）、
   `campaign_executor._resolve_leg_spec`、`r01_baseline`、
   `train_artifacts._persist_strategy_artifact`（非正式 P8-05 路径）。
   本包不产生 `purpose="online_observation"` 的 RunSpec：paper 观察按 P8/AGENTS
   §4.3「同一 run」原则附着源研究 run（p27 §3/I4），其在线用途经入口门禁基准
   + `action_mode="paper"` 表达；该取值保留在闭集中是因为资格权威按它分支
   （D7 缺省推导），未来若预注册在线观察 spec 无需再改 schema。
3. 迁移/拒绝：v2（无 purpose）存储 spec 过 `RunSpec.model_validate` 即失败
   ⇒ 拒绝作为 lineage 源（`reconstruct_runspec_from_lineage` /
   `RunContext.attach` fail-closed，p26 同款）；旧 run 原始 JSON 保持审计可读，
   不重写。升级边界的在途正式 run 必须以 v3 重新预注册。

## 5. D7 机器标准（`ashare_model/data_eligibility.py`）

`check_runspec_data_eligibility` 的缺省评估日改为从 spec 推导：
`evaluation_day` 显式注入保持（测试缝）；未注入时
`purpose=="historical_research"` ⇒ `snapshot_as_of_day(spec.data_cutoff)`，
`"online_observation"` ⇒ 墙钟。滞后、覆盖、阈值、fail-closed 语义逐项不变；
p17 §12 的「最近会话」按用途解读（见 §9 修订指针）。FACTOR_SET_QUALIFIED
证据因此对历史 spec 成为 (spec, data) 的纯函数。B01 审计报告
（`build_data_quality_report`）的 evaluation_day 语义保持「现在」审计——非目标。

## 6. 入口接线（正式路径）

| 入口 | 基准 | 接线 |
|---|---|---|
| `train_entry`（formal） | 历史 | 门禁移入 `establish_formal_run_context`：confirm → freeze spec（含 purpose）→ **G8(purpose=historical, snapshot_cutoff=spec.data_cutoff)** → attach → load |
| `train_entry`（engineering） | 在线（维持现状） | 原地 `require_production()` 墙钟；无冻结身份 |
| `evaluation`（formal） | 历史 | 门禁移到 freeze spec 之后、`RunContext.attach` 之前，同上参 |
| `evaluation`（engineering/audit） | 在线（维持现状） | 原地墙钟 |
| `backtest`（formal） | 历史 | 先读策略 lineage（廉价 JSON）得 `source_spec`，再以 `snapshot_cutoff=source_spec.data_cutoff` 过门禁，然后 attach |
| `backtest`（非正式） | 在线（维持现状） | 原地墙钟 |
| `r01_baseline` | 历史 | 门禁从「confirm 前」移到「freeze spec 后」，`.run(mode="formal", purpose=historical, snapshot_cutoff=spec.data_cutoff)`；DATA_QUALIFIED 报告嵌入历史基准 G8 结果 |
| `campaign_executor` | — | 仅补 `purpose="historical_research"`（leg spec；门禁由各 leg 入口执行） |
| `train_artifacts`（非正式路径） | — | 仅补 `purpose="historical_research"` |
| `run_sim`（formal paper） | **在线** | **零改动**：`require_production()` 缺省在线基准；测试钉住其不传 purpose/snapshot_cutoff |
| `research_doctor` / readiness（p30） | 在线 | 零改动（`gather_gates` 墙钟 = 在线语义，与 p30 §3.1「G8 回答在线观察数据是否新鲜」一致） |
| `scripts/check_production_gates.py` | 在线 | 零改动（活库新鲜度审计） |
| `diagnostics` / `p3_measurement` / `promotion` / `searcher_bench` / `searcher_adjudication` / `admission_experiment` / `archive_run` | 在线（维持现状） | 零改动；这些分析 CLI 不建立运行身份，其门禁语义的用途化属后续独立工作包（见 §8 非目标 4） |

门禁与昂贵操作的顺序总则：历史基准需要冻结 cutoff，因此 formal 路径的门禁在
「复合身份确认 + spec 冻结」之后、attach/load 之前触发（p27 I6 的「身份先于
重活」次序保持；A02 确认本就在 formal 路径中执行，本包不新增确认次数——
性能收敛见非目标 1）。

## 7. 版本影响表

| 版本 | 旧→新 | 理由 |
|---|---|---|
| `ashare_model/runspec.py` `RUNSPEC_SCHEMA_VERSION` | 2→**3** | 必填语义字段 `purpose` 加入且进入 spec_id：v2/v3 spec 不可互换（p27 §8 预见）。旧 payload 验证即拒 ⇒ lineage 拒绝、审计可读（p26 同款迁移） |
| `ashare_model/artifact_schemas.py` `ARTIFACT_SCHEMA_VERSION` | 3 不变 | 边界产物顶层字段集不变（产物携带 spec_id/run_id，不内嵌 spec payload） |
| `ashare_model/artifact_schemas.py` `LEGACY_SCHEMA_VERSIONS` | (1,2) 不变 | 产物侧无新代际；RunSpec schema 拒绝由 `RunSpec.model_validate` 的 Literal 自身执行 |
| `ashare_model/lifecycle.py` `LIFECYCLE_CONTRACT_VERSION` | 2 不变 | 状态机/动作模式表/证据矩阵零变化；DATA_QUALIFIED `report_schema_version` 维持 1（payload 形状不变，G8 detail 字符串追加 basis 标记，`gate_result_hash` 随之逐次一致） |
| `ashare_model/data_eligibility.py` `DATA_QUALITY_REPORT_VERSION` | 2 不变 | B01 报告不动 |
| 其余 `*_VERSION`（`git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"` 全量检索后逐一核对） | 不变 | G1–G8 判定逻辑、D7 阈值、p19/p17 语义、组合/执行/晋级全部不动 |

## 8. 范围与非目标

**范围（inScope）**：`ashare_data/gates.py`；`ashare_model/runspec.py`；
`ashare_model/data_eligibility.py`（仅缺省评估日推导）；`ashare_model/
train_entry.py`、`evaluation.py`、`backtest.py`、`r01_baseline.py`、
`campaign_executor.py`、`train_artifacts.py` 的上述接线；新测试
`tests/test_p36_purpose_eligibility_separation.py` + 分片注册；既有测试的
契约内适配（§10 白名单举证）；`docs/p16` §2.1 与 `docs/p17` §12 的修订指针
（§9）；`docs/CONTRACTS.md` 登记。

**非目标（本包禁止）**：

1. **重复全库确认的性能优化**（用户明示后置）：A02 确认的全量未缓存重哈希在
   formal 路径中的收敛/缓存/复用属独立工作包；本包不新增、也不削减确认次数，
   只把门禁移动到确认产物（冻结 cutoff）可得之后。
2. 不改变 G1–G7 判定、D7 阈值、p17 §6.1 映射、p19 价格/执行语义。
3. 不为 paper 阶段铸造新 RunSpec / 新 run 身份（P8 §4.3 同一 run 原则）。
4. 不改动不建立运行身份的分析 CLI 门禁（§6 表末行）与 webui/dashboard 展示。
5. 不实现 broker 级盘中新鲜度；不做数据补同步。
6. 不删除/改写任何历史产物、旧 run 证据或数据库内容。

## 9. 既有契约修订（指针式，不改冻结正文）

1. `docs/p16_data_freshness_gate_contract.md` §2.1 评估日定义追加一行修订
   注：自 P36 起评估日按用途推导——在线观察 = 墙钟（本契约 §2.1 原文），
   历史研究 = `snapshot_as_of_day(data_cutoff)`（P36 §3）；「禁止用配置项
   覆盖评估日」不变（purpose 是运行身份的冻结字段，不是配置项）。
2. `docs/p17_tushare_pit_data_contract.md` §12「最近会话」追加一行修订注：
   自 P36 起按用途解读——在线 = 墙钟最近完成会话；历史 = 快照 cutoff 会话
   （P36 §5）。

## 10. 预期 RED 测试（`tests/test_p36_purpose_eligibility_separation.py`）

- **R1 身份**：`RunSpec` 缺 `purpose` ⇒ 校验拒绝；`build_runspec` 不传
  `purpose` ⇒ `TypeError`（必填无默认）；同参数仅 purpose 不同 ⇒ `spec_id`
  不同；`RUNSPEC_SCHEMA_VERSION == 3`。
- **R2 历史确定性（核心）**：同一 fixture 库，`purpose=historical` +
  `snapshot_cutoff` 下把 `ashare_data.gates` 的墙钟函数 monkeypatch 为
  `raise` ⇒ 门禁完整运行且 G8 ok（历史基准可证明不读墙钟）；G8 detail 含
  `basis=historical_snapshot` 与 `data_cutoff=`。
- **R3 历史基准的拒绝面**：
  - R3-1 主库日线停在 cutoff − 8 会话 ⇒ `require_production(
    purpose=historical, snapshot_cutoff=...)` raise（窗口末端缺失仍拒绝）；
  - R3-2 cutoff 非法日期 / 缺失 ⇒ `ValueError`；
  - R3-3 混用（historical+today / online+snapshot_cutoff）⇒ `ValueError`；
  - R3-4 cutoff = 日历最末会话且日线完整 ⇒ 历史基准 pass（「视界耗尽」守卫
    不适用），同 fixture 在线基准 + 未来 today ⇒ fail。
- **R4 在线过期仍拒绝**：陈旧 fixture ⇒ 缺省 `require_production()` raise；
  `run_sim` 源级钉住：其 `require_production(` 调用不携带 purpose/
  snapshot_cutoff（在线基准缺省）。
- **R5 D7 缺省日推导**：historical spec（不注入 evaluation_day）在两个不同
  monkeypatch 墙钟下 D7 结论逐项相同；online spec ⇒ 消费墙钟。
- **R6 入口接线（源级）**：`establish_formal_run_context`、`evaluation`
  formal 路径、`backtest` formal 路径、`r01_baseline` 的门禁调用含
  `purpose=...historical...` 与 `snapshot_cutoff=...spec/source_spec.data_cutoff`；
  全部 `resolve_runtime_runspec`/`build_runspec` 生产调用显式传 purpose。
- **R7 迁移拒绝**：v2 形状 payload（无 purpose）过 `RunSpec.model_validate`
  ⇒ `ValidationError`（lineage 拒绝路径）。
- **R8 G8 detail 审计**：在线基准 detail 含 `basis=online_wall_clock`。
- **R9 不变量 5**：缺省 purpose 为在线（不传 purpose 的 run 在陈旧 fixture 上
  拒绝——宽限必须显式声明）。

## 11. 既有测试适配（AGENTS §10.1 白名单声明）

1. `tests/test_runspec.py:93,345`（`RUNSPEC_SCHEMA_VERSION == 2`）：白名单
   第 2 类，本契约 §7 为批准依据；随 bump 改 3，断言强度不变。
2. `tests/conftest.py::make_test_spec` 增补 `purpose` 缺省
   `"historical_research"`（研究夹具默认）——测试构造面集中收敛，直接调用
   `build_runspec` 的少量测试显式传 purpose；属白名单第 3 类（集合与断言
   强度不变）。
3. 其余因「必填参数」机械补参的测试调用点：白名单第 3 类，逐文件在提交证据
   列出；不改动任何断言语义。

## 12. 测量、验证与停止条件

- 验证（精确候选 commit）：聚焦
  `python -m pytest -q tests/test_p36_purpose_eligibility_separation.py
  tests/test_gates.py tests/test_runspec.py tests/test_run_store.py
  tests/test_data_eligibility*.py tests/test_p34_r01_baseline.py` → 相邻契约
  （p26/p27/p30/p31/p32/p35 相关文件）→ `python -m pytest -q tests -n auto`
  （parallel-vs-serial 对账未做则串行）→ `python -m compileall -j 0 -q
  ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi` →
  `git diff --check`。
- 测量日志：`docs/p36_purpose_eligibility_separation_measurement_log.md`
  （后置 evidence-only，引用被测实现 SHA）。
- **停止条件**：(a) 用途需要第三值才能表达某正式入口语义；(b) 历史基准 G8
  与 G3/`missing_trading_days` 出现不可调和职责重叠；(c) formal 路径无法在
  不破坏 p27 I6 身份先于重活次序的前提下获得冻结 cutoff；(d) 全量回归出现
  无法归因于本包的失败。

## 13. Retirement

本契约不退役任何代码路径。前瞻义务：若未来预注册 `online_observation` 用途
的 RunSpec（如 paper 阶段独立身份），须修订 §4.2 与 p27 动作模式表并另行
bump；分析 CLI 门禁的用途化（§8 非目标 4）立项时以本契约 §6 表为起点。
