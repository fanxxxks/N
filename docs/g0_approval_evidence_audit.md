# G0 批准证据审计：P12 / P13 / P14（t1，governance-auditor）

- 性质：**只读审计报告**（evidence-only）。本文只记录"实际查到什么"，不改变任何
  契约状态；三份契约文件头现仍标注的 DRAFT 原样保留（任务纪律：禁止把 DRAFT
  改成 APPROVED）。
- 审计人：governance-auditor（AgentTeams tushare-pit-migration t1），2026-09-03。
- 证据基线：main @ `8ff21f6`（工作分支 codex/g0-tushare-pit-contract 与 main 同
  点创建）。审计方法：契约文件头与批准条款原文 + git 全历史（含已合并分支的
  实现 commit 正文）+ 代码现场（版本常量/门禁存在性）+ 测量日志与产物索引，
  逐项交叉核对。审计命令均为只读（git log / git show / 文件读取）。
- 结论预告：三份契约均为"**评审门有运行痕迹、实现已合入 main 并生效，但契约
  文件头的批准记录从未补记**"。不属于证据篡改；属于批准记录完整性缺口，且
  批准权归属需要用户追认。问题清单见 §4，由 captain 转达用户裁决。

## 0. 仓库既有批准记录规范（对照基准）

| 契约 | 文件头批准记录 | 记录形式 |
|---|---|---|
| p9 | APPROVED — 2026-09-01 由 AgentTeams captain 批准 | 明注"用户已授权 captain 自主推进，豁免 AGENTS 1.7 质问义务" |
| p10 | APPROVED — 同 p9 模式 | 同上 |
| p11 | APPROVED — 契约门禁双轮评审（t8 首轮 → t39 修复 → t40 复审 pass） | 以团队评审门闭环为批准依据 |
| p16 | 已获用户治理裁决正式批准（2026-09-03） | 补记 commit `2f5e18b`：预注册 `1745773` → 用户批准广播 → 实现 `50d0753` → 合并 `2fb4910`，git 可查时序，实现未先于批准 |
| **p12/p13/p14** | **DRAFT（未变）** | **无批准记录**（本审计对象） |

即：仓库存在三种合法批准形态（用户授权 captain 批准 / 评审门闭环 / 用户治理
裁决），且 p16 确立了"事后补记批准记录 + git 可查时序"的先例。p12/p13/p14
缺的不是证据链，而是**最后一步的批准记录回写**。

## 1. P12 晋级执法契约（p12_promotion_enforcement_contract.md）

### 1.1 文件头状态（原样，未改动）

- `状态：DRAFT — 提交契约门禁评审（t9）`（该 t9 为 alphagpt-p0-p1 团队的任务
  编号，与当前团队 t9 无关）。
- §11 批准与实施边界："批准流程：经契约门禁评审（t9）通过后生效；**批准记录
  补记于本节**。"——本节至今无任何批准记录。

### 1.2 实现与生效证据（git + 代码现场，全部在 main @ 8ff21f6 可复核）

1. 契约 PR：`97a86fe docs: pre-register P12 promotion enforcement contract
   (B-line gate)` 已合入 main。
2. 实现 commit：`bed24d0 feat(promotion): enforce registry status via new
   promotion gate G7 (P12)`，正文明文："Preregistered contract: … (merged to
   main @ 97a86fe, **review t9 pass**)"，并记录 RED 证据（实现前 8 failed /
   34 passed）与实现后验证（test_promotion 42 passed；相邻测试 79 passed；
   compileall 0；分片检查 91 files）。
3. 代码现场：`ashare_model/promotion.py:91 PROMOTION_RULE_VERSION = "2"`；
   G7（feature_registry_status）在生产晋级路径生效——当前团队 t34（R5）的
   晋级门编号说明已按 G1–G7 口径书写，即 G7 已被后续契约当作事实引用。
4. 测量日志：无 p12 measurement log——与契约 §8 预注册一致（"本线不产生研究
   测量"），**非缺口**。

### 1.3 缺口定性

- 批准记录缺失：契约 §11 承诺的批准记录未补记；"review t9 pass"仅存在于实现
  commit 正文，契约文件本身仍自称"提交契约门禁评审"。
- 状态矛盾：文件头（DRAFT 待评审）与代码事实（G7 已在晋级路径执法）不一致。
  按契约自身条款，若评审确已通过，应补记批准；若未通过，实现先于批准落地，
  需要用户裁决处置（G7 门已实际拒绝含非晋级特征的公式）。

## 2. P13 fundamental 字段契约（p13_fundamental_fields_contract.md）

### 2.1 文件头状态（原样，未改动）

- `状态：DRAFT — 提交契约门禁评审（t10）`；§10："批准记录补记于本节"——无
  批准记录。

### 2.2 实现与生效证据

1. 实现链（全部可 `git log` 复核）：
   - `02c30e8 feat(p13): fundamental_pit cash-flow + total-asset fields,
     unlock family 5 (C-line t14)`（主实现）；
   - `e5f63ff` / `d0d41d8` / `e5565a2` / `8d89f6a`（t46 方案 A/B 往返，终局
     "captain final ruling" 方案 A）；
   - `4e0c6cc` + `632b13f` / `5d6dd7c`（t57 F2：按代扩展冻结布局，勘误原文
     回写契约 §6.3/§6.4）。
2. 代码现场：`GRAMMAR_VERSION = 6`（vocab.py:285）、`FEATURE_REGISTRY_VERSION
   = 6`（feature_registry.py:61）、`RESEARCH_DOMAIN_VERSION = 3`（research_
   domain.py:54）——契约 §6.1 版本表的三项 bump 全部兑现；第⑤族 4 特征
   （CASHFLOW_QUALITY/ACCRUALS/ASSET_GROWTH/EARNINGS_ACCEL）已在 grammar-6
   词表中。
3. 数据回填证据：`docs/p13_fundamental_backfill_measurement_log.md`（evidence-
   only）完整记录：回填窗口②（2026-09-02 04:03–05:31）、三次缺陷与修复
   （c2e1ec7）、purge 116,613 行、覆盖率门 PASS（net_operate_cash_flow
   0.9984 / total_assets 0.9706 ≥ 0.9）、G1–G7 formal 全 PASS、research_doctor
   HEALTHY、dataset_id `e15b4fc4… → b7b4dd4b…`。数据面已生效并被后续任务
   （当前团队 t18 引用该 dataset_id）当作事实基线。
4. 遗留（如实记录，非批准缺口）：§8 族级裁决（77 名重测）**未执行**——即当前
   团队 t18；4 特征的 promotion_allowed 族级裁决悬而未决（当前 authored=True
   按 §5.3 预注册生效，族裁决可改为 False）。

### 2.3 缺口定性

- 批准记录缺失：同 P12 模式；实现、回填、版本 bump 全部完成并生效，契约头
  仍是 DRAFT。
- 契约内勘误（§6.3 t46 方案 A、§6.4 t57 追加段）由 contract-a 裁决、captain
  接受后落盘（commit 正文可查），**用户批准痕迹未见**——勘误本身的批准权也
  在问询范围内。

## 3. P14 搜索层消化率契约（p14_search_digest_preregistration.md）

### 3.1 文件头状态（原样，未改动）

- `状态：DRAFT — 待独立评审（t17 门禁）。批准前禁止实现动代码（§2.2 门禁
  顺序不可倒置）`——这是三份中措辞最强的一份：明文禁止批准前动代码。

### 3.2 实现与生效证据

1. 实现 commit：`64acd6e feat(p14): search digest P1-1+P1-2 — punitive
   worst-so-far fitness/tell anchors, per-trial EOS length prior (profile
   p14-uniform-2-11-v1), research/promotion_tier_a campaign tracks with
   tier-purity fail-closed validation (SEARCH_CONTRACT 4, SEARCHER_BENCH 4;
   docs/p14_search_digest_preregistration.md §5/§6)`。
2. 代码现场：`SEARCH_CONTRACT_VERSION = 4`（search_contract.py:31）、
   `SEARCHER_BENCH_VERSION = 4`（searcher_bench.py:83）、`P14_RESEARCH_BUDGET
   = 1200` / `P14_PROMOTION_BUDGET = 800`（searcher_bench.py:96-97）——§6 版本
   表两项 bump 与 §5.4 预算拆分全部兑现并在生产搜索路径生效。
3. 评审门运行痕迹（间接但具体）：契约文本已并入 t17 评审发现——§2 假设 5 注
  "（实装 4.9.0 实测 …，t17 F2）"、§5.4 注"（… 故行为无差，t17 F1）"；另有
  t24-ruling-obsA / t49（RED-2 FP 敏感性实证，`fda4635`/`3a98650`）、t43
  （TPE 分布界恢复，`ba9bf31`）、t51（threadpoolctl pin，`a754ddc`）等实现期
  修订均有 commit。即评审与修复闭环在团队流程内真实发生，但批准记录未按文件
  头要求回写。
4. 遗留（如实记录，非批准缺口）：§9 正式研究测量（GP 消耗率 ≥0.50、上限堆积
  ≤50%、A track 纯度 ==100% 三判据）**未执行**——即当前团队 t23（E4：P14
  正式测量，语义稳定后执行）；无 p14 measurement log（正确：测量未运行，
  禁止预写）。

### 3.3 缺口定性

- P14 与 P12/P13 同模式，但增加一层张力：文件头明文"批准前禁止实现动代码"，
  而实现已合入并生效。若用户追认批准，则"实现未先于批准启动"的时序主张需要
  以用户追认日期为准补记（p16 先例：用户批准后补记时序）；若用户不追认，则
  需要处置已生效的 SEARCH_CONTRACT 4 机制（回退 = 单一 revert，契约 §7.5 自
  述可回滚）。

## 4. 需向用户确认的问题清单（captain 转达，逐项等待裁决）

- **Q1（P12 追认）**：P12 契约（晋级门 G7 执法）实现已合入 main 并生效
  （PROMOTION_RULE_VERSION="2"；实现 commit 自述"review t9 pass"）。是否追认
  该契约的批准？（A) 追认：授权按 p16 先例补记批准记录（注明用户追认日期与
  依据）；(B) 不追认：请明确已生效 G7 门的处置（保留待重新批准 / 回退）。
- **Q2（P13 追认）**：P13 契约（第⑤族解锁 + 数据回填）实现与回填已完成并
  生效（grammar/registry/domain 6/6/3，dataset b7b4dd4b）；族级裁决（§8，
  当前团队 t18）未执行。是否追认该契约批准（含 §6.3 t46 方案 A 勘误与 §6.4
  t57 追加段两处 captain/contract-a 裁决修订）？（A/B 同 Q1。）
- **Q3（P14 追认）**：P14 契约（搜索层消化率机制）实现已合入并生效
  （SEARCH_CONTRACT/SEARCHER_BENCH 4；预算拆分 1200/800），文件头曾明文
  "批准前禁止实现动代码"；正式研究测量待当前团队 t23。是否追认该契约批准？
  （A/B 同 Q1；不追认时的回退路径 = 契约 §7.5 单一 revert。）
- **Q4（补记授权与追认格式）**：若 Q1–Q3 任一为"追认"：批准记录由谁补记？
  建议按 p16 先例（`2f5e18b` 模式）由 captain 或 governance-auditor 以
  evidence-only commit 在各契约批准节补记"用户于 <日期> 追认 + 依据 + 实现
  时序 git 引用"，并同步把文件头状态行改为 APPROVED（该次状态改动本身即用户
  授权的产物）。
- **Q5（流程根因，非契约问题）**：p12/p13/p14 的实现均发生在
  alphagpt-p0-p1 团队流程内且评审门有运行痕迹，批准记录却系统性缺失。若用户
  确认当时存在"授权 captain 自主推进"类授权（p9/p10 先例），建议在补记时一
  并写明授权来源，避免后续团队复现同一缺口。

### 4.1 裁决结果（2026-09-04 补记；t35 执行规格第 1 项）

- **批准原文（逐字引用，唯一证据基准）**：「p17批准，1.1,2.1,3.1,4.1,5.1」。
- **转达渠道与时间**：captain 会话（AgentTeams tushare-pit-migration，
  captain → governance-auditor 团队邮箱消息），2026-09-04。
- **映射（captain 声明的编号清单对应关系，本审计如实记录、不自行添加解释）**：
  - Q1① = P12 追认 **APPROVED**；
  - Q2① = P13 追认 **APPROVED**（含 t46/t57 勘误追认）；
  - Q3① = P14 追认 **APPROVED**；
  - Q4① = 按 p16 先例（commit `2f5e18b`）模式补记批准记录；
  - Q5① = 起草 captain 自主推进授权条款交用户批准（另派 t37，Q5 对应项）。
- **执行（Q4① 落地）**：四份契约文件头按 p16 先例补批准行——p12/p13/p14
  状态 DRAFT→APPROVED（追认，git 时序见各自批准记录：`97a86fe`→`bed24d0`、
  `7f9c1fd`→`02c30e8`+勘误、`dcc228b`→`64acd6e`）；p17 状态 DRAFT→APPROVED
  （**全文按 `00e276b` 版本生效，契约文本零修改**——批准对象即该版本原文）。
  `docs/CONTRACTS.md` 索引按索引规则（不作状态裁决）不动。
- **证据性质说明**：原文为简式批准（"p17批准" + 五个条目号），条目号到
  Q1-Q5 选项的映射来自 captain 发送给用户的编号清单及 captain 的转达声明；
  本节逐字留存原文与映射关系供复核。三份被追认契约的遗留事项（P13 §8 族级
  裁决 = t18、P14 §9 正式测量 = t23）不因追认而预写结论。
- 本补记为 captain 转达用户裁决后的授权执行（AGENTS §1：流程例外须由用户
  明确授权——本补记的授权即用户裁决原文本身 + Q4① 条目），非审计结论变更；
  §1-§3 的审计事实保持原样，§5/§6 中描述"审计时点状态"的条目已就地加注
  更新指引。

## 5. 审计结论

1. 三份契约**保持 DRAFT 原样**（本审计零状态改动；`git status` 复核见 §6）。
   （2026-09-04 更新：此条描述审计时点状态；用户裁决后已按 §4.1 追认为
   APPROVED 并补记批准行。）
2. 批准记录缺失是文档完整性问题 + 批准权问题，不是证据伪造：评审门运行痕迹、
   RED/GREEN 证据、版本 bump、测量日志、代码现场五类证据链完整且互相一致。
3. 在用户裁决（Q1–Q5）之前，当前团队的下游依赖已把这些契约当作事实基线
   （t18 引用 P13 dataset、t23 执行 P14 正式测量、t34 按 G1–G7 口径书写）。
   建议 captain 在转达问题时同步说明：用户裁决只影响"契约状态的合法性记录"，
   不改变已固化证据；任何"不追认"分支都需要单独的处置契约（回退或重新批准），
   禁止以沉默继续。
4. 附带治理观察（与批准无关，供 captain 参考）：当前团队 DAG 中 t2（p17 评审
   + 用户批准门）无任何下游依赖，而 t7/t9/t12 的任务文本均自述"按**已批准的**
   p17 契约"实施。建议将 t2 加入 t7（至少 t9，回填含 DB 写入）的依赖，避免
   复现"实现先于批准"的同类缺口。

## 6. 审计操作边界复核

- 全程只读：git log/show/status、文件读取、版本常量检索；无 DB 连接、无
  pytest、无任何写入口运行。
- 契约文件零改动：p12/p13/p14 三文件在本任务分支的 diff 中不存在。
  （2026-09-04 t35 补记：经用户裁决授权（§4.1），四份契约文件头在本分支
  补批准行——p12/p13/p14 DRAFT→APPROVED、p17 DRAFT→APPROVED（文本零修改）；
  本条描述 t1 审计时点状态。）
- 本报告为任务唯一新增文件之一（另一为 p17 契约，见同分支另一 commit）。
