# P12 晋级执法预注册契约（B 线：promotion_allowed/deprecated 晋级门禁消费）

- 状态：**DRAFT — 提交契约门禁评审（t9）**。批准后本文保持前瞻性，禁止用实现
  或测量输出反向修改；测量结果只能执行预注册裁决。
- 起草：contract-a（AgentTeams alphagpt-p0-p1 t6），2026-09-02
- 证据基线：main @ `fde9f8b`（阶段 0 清障后，t4 审核通过）；代码证据全部在
  main @ fde9f8b 实测（命中计数/行号见 §1）；`ashare_model/promotion.py`、
  `tests/test_promotion.py`、`ashare_model/alphagpt.py:353-368`、
  `ashare_model/gp_search.py:98-115`、`docs/p9_measurement_log.md` §2.4/§3、
  `docs/p10_measurement_log.md` L128（活体案例）逐项复核。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。任务路由：**研究语义变更**
  （AGENTS §3.2"统计检验或晋级阈值"）+ 运行态/晋级（§4.3 fail closed）。
  仲裁顺序：本文/批准 → 契约测试 → 实现 → 验证记录。§11.1：本文只写计划。

## 1. 问题陈述

### 1.1 六门零消费：promotion_allowed / deprecated 从未进入晋级路径

晋级门（`ashare_model/promotion.py` `evaluate_challenger`）现有六门：
G1 data_formula_p0、G2 significance、G3 excess_and_risk、G4
cost_capacity_stress、G5 paper_window、G6 data_tier（L554-577）。**没有任何
一门读取特征注册表的 `promotion_allowed` 或 `deprecated` 状态。**

main @ fde9f8b 全仓 `*.py` 命中计数（本契约起草时实测）：

- `promotion_allowed` ×32，分布于 5 个文件：`feature_metadata.py` 18（ authored
  标志定义）、`feature_registry.py` 8（记录/序列化/一致性守卫）、
  `scripts/generate_registry_docs.py` 1（文档生成）、
  `tests/test_feature_metadata.py` 4、`tests/test_feature_registry.py` 1。
  **运行时执法消费者：0。**
- `deprecated` ×48，分布于 14 个文件。运行时消费者全部在**采样/域侧**：
  `alphagpt.py` 2（动作掩码，L353-368）、`gp_search.py` 4（GP 终端集，
  L98-115）、`vocab.py` 3（`deprecated_names` 定义与接线 L241/L298）、
  `research_domain.py` 3（域特征划分排除 deprecated）；其余为注册表序列化
  （`feature_registry.py` 12）、元数据/脚本/测试。**`promotion.py`、
  `evaluation.py`、artifact 加载（`artifact_writer.py`）、run_store、webapi：
  0 命中。**
- `deprecated_names` ×5（vocab 2 + alphagpt 1 + gp_search 1 + test 1）——
  唯一的跨模块执法通道，且只到采样侧为止。

即：注册表状态在**采样侧有双执法先例**（RL 动作掩码把 deprecated 特征的
logit 置 −inf；GP 终端集直接不收录 deprecated 特征），但在**晋级侧零执法**——
一个公式无论包含多少非晋级/已弃用特征，都能通过全部六门。

### 1.2 与 P9"可晋级 58"预注册裁决的矛盾

P9 测量日志（`docs/p9_measurement_log.md`）：

- §2.2：族③（涨跌停事件条件）预注册增量裁决 = **负结果**，"特征保留计算与
  采样、promotion_allowed=False"；
- §3 裁决执行：`LIMIT_UP_CNT_5`、`LIMIT_DOWN_STREAK`、`LIMIT_BREAK_5` →
  feature_metadata `promotion_allowed=False`（**保留采样、禁止晋级**）；
- §2.4：可采样 61 名；**可晋级（promotion_allowed）= 58**。

"可晋级 58"在代码中的全部支撑是 `feature_metadata.py` 的 authored 标志
（族③三个特征 L630/L637/L644 = False）与注册表一致性守卫
（`feature_registry.py` L362-364：deprecated 特征必须 promotion_allowed=False）。
由于晋级门零消费（§1.1），该裁决对晋级路径**没有任何强制力**：含族③特征的
公式（族③特征为 PIT_DAILY → **Tier A**，`data_tier.py` L76，可通过默认
G6 数据层级门）今天可以通过全部六门。这是裁决与执法之间的结构性缺口，
不是文档瑕疵。

### 1.3 活体案例：P10 random seed-7（测量日志 L128）

P10 Stage B/C 中唯一通过全部四条 OOS 回测硬门槛的公式（年化 22.77%/回撤
7.52%/Sharpe 1.268/Calmar 3.027，`docs/p10_measurement_log.md` **L128**，
"✓ 全过"）：

```
(STD5(TURNOVER) DIV (ATR_14 CORR5 STD20(DECAY(
    GATE(LIMIT_UP_CNT_5, CROWD_AMOUNT_60, MARGIN_CROWD_60)))))
```

该公式**内嵌 `LIMIT_UP_CNT_5`**——P9 预注册裁决明令"禁止晋级"的族③特征。
后果链完整可见：搜索侧按设计保留族③可采样（P9 §3）→ 搜索自然选中它产出
OOS 最优公式 → 晋级门对 promotion_allowed 零消费 → 该公式在晋级路径上
畅通无阻。分层细节（如实记录）：该公式同时含 `MARGIN_CROWD_60`（融资余额，
Tier B），在默认 Tier-A-only 政策下 G6 会因 Tier B 拒绝；但在 P2-03 明文
允许的 `--allow-tier-b` 独立对比路径下 G6 放行——而**两条路径都不消费
promotion_allowed**，缺口与分层政策无关。活体案例证明该缺口有真实后果：
OOS 表现最好的公式恰好踩在裁决禁区上。

## 2. 假设

- **H1（执法缺口假设）**：在晋级门新增对注册表状态的 fail-closed 消费，可使
  "可晋级 58"从文档声明变为机器执法，且不改变任何采样/评价/数据门语义。
- **H2（最小改动假设）**：G1–G6 已有的 artifact 加载与解码路径
  （`_top_rows` → tokens → `formula_data_tier_report` 的 token 解码）足以
  承载新门——新增门复用同一解码路径即可覆盖"新采样的公式"与"从 artifact
  加载的历史公式"两个来源，无需第二套解析。
- **H3（对照假设）**：采样侧双执法先例证明"注册表状态 → 机制排除"是本仓库
  既有模式；晋级侧补齐同类执法是模式一致化，不是新发明。

## 3. 范围与非目标

### 范围（B 线唯一改动面）

1. `ashare_model/promotion.py`：新增版本常量 `PROMOTION_RULE_VERSION = "2"`
   （§6.1）；`evaluate_challenger` 新增第七门 **G7 `feature_registry_status`**
   （§5）；verdict payload 增量字段（`promotion_rule_version`、
   `registry_status_policy`）；模块 docstring 记录 G1–G6 = v1 未版本化时代。
2. `ashare_model/feature_registry.py`：新增单一报告函数
   `formula_registry_status_report(tokens=None, feature_name=None)`（§5.2），
   复用 `data_tier.formula_feature_names` 的既有解码路径。
3. `tests/test_promotion.py`（或新文件，同 commit 注册分片）：§7 RED 清单。
4. `docs/p12_promotion_enforcement_contract.md`（本文）。

### 非目标（显式声明，评审按此驳回越界改动）

1. **不改数据门 G1–G7**（`ashare_data/gates.py`、
   `scripts/check_production_gates.py`）。数据门与晋级门是两个独立编号空间
   （AGENTS §5.4 明文），本契约新增的是**晋级门 G7**，与数据门 G7 无任何
   关系，禁止混用表述与实现。
2. **不改采样侧执法**：`alphagpt.py:353-368` 动作掩码与 `gp_search.py:98-115`
   终端集是**对照先例，不是改动对象**；族③特征**保留可采样**是 P9 预注册
   裁决（"保留计算与采样、promotion_allowed=False"），本契约不得把
   promotion_allowed 泄漏进采样空间（否则改变搜索空间语义，须 GRAMMAR
   bump 与 P9 级预注册，显式非目标）。
3. 不改 G1–G6 的阈值、语义、理由字符串；不改 `PromotionThresholds`；
   不改 `admission.py`（ADMISSION_RULE_VERSION 是搜索准入，另一编号空间）。
4. 不改注册表数据与 authored 元数据（`_DEPRECATION_REASONS`、
   feature_metadata 全部标志、FEATURE_REGISTRY_VERSION 均不变——只新增
   消费方，不改数据）。
5. 不改 legacy 产物的只读展示与人工审计路径（§4.3：legacy 只读展示合法）；
   执法点只有晋级判定。
6. 不切生产默认、不做 paper/sim、不做 P8 lifecycle 状态转换、不动 P10/P9
   历史产物。

## 4. 不变量（实施全程必须成立）

1. **fail-closed**：无法追踪公式（tokens 不可解码）、特征名不在注册表、
   元数据缺失 → G7 一律拒绝并给出理由；禁止 warning 后继续（AGENTS §4.3）。
2. **单一解码路径**：公式 → 特征名只经 `data_tier.formula_feature_names`
   （G6 既有路径）；禁止第二套 token 解码（AGENTS §1.3、§9）。
3. **单一注册表权威**：promotion_allowed/deprecated 只从
   `feature_registry`/`feature_metadata` 读取；禁止在晋级门内复制一份名单。
4. **G1–G6 零扰动**：既有六门的判定、理由、阈值、测试断言全部原样；
   新门为纯增量（verdict 多一门，`promoted` 由六门 AND 变七门 AND——
   这是唯一的全局语义变化，即本契约的目的本身）。
5. **编号空间纪律**：晋级门 G1–G7 与数据门 G1–G7 在代码、文档、测试中始终
   以"晋级门/数据门"前缀区分。
6. **legacy 可读性**：含 deprecated 特征的历史公式保持可解码、可展示、可
   审计（P9 token 保值承诺不回退）；仅晋级消费被拒绝。

## 5. 方案：新增晋级门 G7（feature_registry_status）

### 5.1 选择新增 G7 而非扩展 G6 的理由（预注册决策）

- **权威分离**：G6 的权威是 `data_tier.py`（数据可信层级），注册表状态权威是
  `feature_registry.py`（研究裁决记录）；合并会让一门持有两个权威、理由
  语义混杂。
- **增量最小**：新增门对既有六门测试零扰动（§4.4）；扩展 G6 需要改写其既有
  理由字符串与全部相关断言。
- **审计清晰**：拒绝发生时 verdict 明示是哪一门、哪一权威拒绝。

### 5.2 G7 语义（预注册）

- 输入：与 G6 相同的 top-trial 来源（`_top_rows(artifact)[-1]` 的 `formula`
  tokens 与 `formula_text`）——**artifact 加载路径天然覆盖**：晋级判定的
  输入本来就是从 artifact 读出的公式，无论新采样还是历史存储，走同一检查。
- 报告：`formula_registry_status_report(tokens, feature_name)` 返回
  `{"feature_registry_version", "per_feature": {name: {"promotion_allowed":
  bool, "deprecated": bool}}, "traceable": bool}` 或不可追踪时
  `traceable=False`（复用 `formula_feature_names`；bare-factor 基线行走
  `feature_name` 路由，与 G6 对称）。
- 判定：`per_feature` 中任一特征 `promotion_allowed=False`（deprecated 集合
  因一致性守卫是其子集，自动覆盖）→ 拒绝，理由（预注册字符串）：
  - `feature {name} is not promotion_allowed (feature registry status)`
  - 当该特征同时 deprecated，附加
    `feature {name} is deprecated (feature registry status)`
  - 不可追踪（无 tokens/不可解码/未知特征名）→
    `no traceable formula for registry status; promotion requires a formula
    whose every feature is registry-promotable`
- verdict 增量字段：`promotion_rule_version: "2"` 与
  `registry_status_policy: {"feature_registry_version": 5, "report": ...}`
  （additive；verdict JSON 非类型化 schema 产物，无 schema bump）。

## 6. 版本影响总表与迁移/拒绝策略

### 6.1 版本影响表（promotion 相关所有者；`git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"` 在 main @ fde9f8b 全量检索的基础上取相关子集）

| 版本所有者 | 旧值 | 新值 | bump 理由 |
|---|---|---|---|
| **新增** ashare_model/promotion.py `PROMOTION_RULE_VERSION` | （无——G1–G6 为未版本化的 v1 时代） | **"2"** | 晋级门集合语义变更（六门→七门）；v1 时代的判定不可与 v2 互比；docstring 记录 v1 历史。不在无常量处悄悄改语义（§3.2：默认 bump） |
| ashare_model/feature_registry.py:57 FEATURE_REGISTRY_VERSION | 5 | 不变 | 注册表数据零改动（15 个非晋级特征原样）；只新增消费方。数据未变则 bump 反而制造伪不兼容 |
| ashare_model/data_tier.py:37 DATA_TIER_VERSION | 1 | 不变 | 层级语义零改动；G6 原样 |
| ashare_model/vocab.py:226 GRAMMAR_VERSION | 5 | 不变 | 采样空间零改动（族③保留采样是 P9 裁决；禁止借本契约改采样） |
| ashare_model/admission.py:18 ADMISSION_RULE_VERSION | 2 | 不变 | 搜索准入门是独立编号空间，零改动 |
| ashare_model/runspec.py:40 RUNSPEC_SCHEMA_VERSION | 1 | 不变 | RunSpec 仅引用 PromotionThresholds（L99）；新门不读阈值，schema 零改动 |
| ashare_model/artifact_schemas.py:44 ARTIFACT_SCHEMA_VERSION | 2 | 不变 | verdict JSON 由 promotion.main 直接写出（非类型化 schema 产物），additive 字段无需 bump；被消费的 protocol artifact 侧零改动 |
| ashare_model/evaluation.py:287 PROTOCOL_VERSION | "25" | 不变 | 评价协议零改动 |
| ashare_model/reward.py:141 REWARD_VERSION | "14" | 不变（本契约范围） | B 线不 bump；注：A 线 p11 独立 bump 14→15，经 G1 既有 pin 自动生效，见 §11 集成顺序 |
| 其余全部 `*_VERSION` 所有者（manifest/artifact_schema/legacy/bare_factor/fee_matrix/elite/factor_compute/imitation/ledger/p3/research_domain/rl_diagnostics/run_store/search_contract/p10_adjudication/searcher_bench/semantic_cache/target/tier_report/model/portfolio×3/execution/rebalance） | — | 不变 | 与晋级门语义无交集；全量清单见 docs/p11_reward_v15_contract.md §6.1（同一基线检索，此处不复制第二份） |

### 6.2 legacy 公式重放与历史判定的迁移/拒绝策略

- **历史 verdict**：不含 `promotion_rule_version` 的既有
  `data/promotion_verdict.json` 视为 v1 时代产物——只读保留、人工审计；
  不得与 v2 判定互比或拼接（§4.3）。无重写、无迁移脚本。
- **旧 artifact 重新晋级**：按 v2 规则全量重判；G1 既有版本 pin +
  新 G7 共同 fail-closed。被拒绝是预期执法行为，不是回归。
- **含 deprecated 特征的 legacy 公式**：保持可解码/可展示/可审计（不变量 6）；
  作为晋级证据一律被 G7 拒绝。禁止为其开"豁免通道"——如确需晋级某历史公式，
  唯一合法路径是先修订注册表裁决（feature_metadata + 新预注册），而不是
  绕门。
- **族③特征**：本契约不改变其任何状态（仍 Tier A、仍可采样、仍
  promotion_allowed=False）；G7 生效后"禁止晋级"首次获得强制力。若未来数据
  侧补齐后要恢复族③晋级资格，必须走新的预注册裁决（P9 §7 同级），显式不在
  本契约范围。

## 7. 预期 RED 测试清单（实现前先红；实现后全绿）

新增测试进入 `tests/test_promotion.py`（复用其 `_strong_artifact`/
`_passing_stress`/`_paper_registry` fixture 体系）或新文件（同 commit 注册
分片，fail-closed）。测试断言唯一来源 = 本契约 §5 预注册语义。

1. **RED-1 族③公式晋级拒绝（现状：六门全过 → 断言失败，红证缺陷）**：
   构造通过型 artifact（复用 `_strong_artifact`），top trial 的 formula
   tokens 编码为仅含 Tier A 特征的组合公式、其中内嵌 `LIMIT_UP_CNT_5`
   （保证 G6 通过，使拒绝只能来自 G7）。断言
   `evaluate_challenger` 的 G7 拒绝且理由匹配 §5.2 预注册字符串、
   `promoted=False`。现状代码无 G7 → 断言失败，失败输出即缺陷固化证据。
2. **RED-2 deprecated 公式经 artifact 加载路径拒绝（现状失败）**：top trial
   formula 内嵌已弃用特征（如 `LIMIT_STREAK`，token 保留可解码）。采样侧
   今天已不会产出该公式，但 **artifact 加载路径可以携带它**——这正是必须
   在晋级侧执法的原因。断言 G7 拒绝。
3. **RED-3 裸因子基线行路径（现状失败）**：top trial `formula=None`、
   `formula_text="LIMIT_UP_CNT_5"`（G6 的 bare-factor `feature_name` 路由
   对称场景）。断言 G7 经 `feature_name` 路由拒绝。
4. **RED-4 fail-closed 边界（现状部分缺失）**：(a) tokens 不可解码 →
   G7 拒绝且不弱化 G6 既有"no traceable formula"行为；(b) 构造注册表外
   特征名 → 拒绝（禁止静默跳过）。
5. **RED-5 放行对照（防过杀）**：仅含可晋级特征（如 `RET_1`）的 artifact
   在其余六门通过的 fixture 下，G7 必须通过且
   `registry_status_policy.report` 逐特征记录 `{promotion_allowed: true,
   deprecated: false}`；既有六门断言零变化。
6. **RED-6 活体案例回归（seed-7 公式固化）**：以 p10 产物记录的 seed-7
   选中公式 tokens（`docs/p10_searcher_comparison_20260901/summary.json`
   selected.tokens）构造通过型 artifact（G1–G6 stub 全过），断言 G7 因
   `LIMIT_UP_CNT_5` 拒绝——把 §1.3 活体案例变成永久回归测试。
7. **版本 pin**：新增 `assert PROMOTION_RULE_VERSION == "2"`（新常量首版
   pin；无既有测试需修改——本契约零白名单情形 1/3 消耗，情形 2 不触发，
   因为没有任何既有断言描述新门）。

## 8. 验证与测量方案（B 线为执法门禁线，无研究测量）

- 本线**不产生研究测量**：G7 是确定性执法规则，不存在采样/回测/统计口径；
  p10 seed-7 案例是**既有产物的引用**（§1.3），不是要复刻的测量。§3.2 的
  "测量方案"要素在本线落地为以下验证矩阵：
  1. 最小相关：`python -m pytest -q tests/test_promotion.py`（含新增 RED
     全绿）；
  2. 相邻契约：`tests/test_feature_registry.py tests/test_feature_metadata.py
     tests/test_vocab.py tests/test_identity.py`（注册表/词表/哈希零回归）；
  3. 全量：`python -m pytest -q tests -n auto`（并行/串行 parity 对账记入
     实现报告）；
  4. `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio
     ashare_trading scripts webapi`；
  5. `python scripts/check_test_shards.py`（新测试文件同 commit 注册）；
  6. `git diff --check`。
- 无 DB 写入、无搜索运行、无数据同步；验证在精确候选 commit 上执行并记录
  环境（OS/Python/依赖）。

## 9. 预注册成功判据与裁决规则

- **S1（实现级）**：§7 RED-1..7 全绿于实现 commit；既有 test_promotion 及
  §8.2 相邻测试零失败（G1–G6 无行为回归）。
- **S2（执法有效性，字符级）**：对 §1.3 seed-7 公式与 §7 RED-1 构造公式，
  verdict 的 `promoted=False` 且拒绝理由精确归因 G7（非 G6 Tier 或其他门）
  ——证明缺口由本契约闭合。
- **S3（零泄漏）**：采样侧行为零变化——`test_alphagpt.py`/`test_gp_search.py`
  /`test_vocab.py` 全绿且未修改；族③特征仍在采样词表（可采样 61 不变）。
- **裁决**：S1–S3 全部成立 → B 线实现过门禁，进入独立审核（t22）。
  任一不成立 → 实现缺陷，修复重验；**禁止**通过弱化断言、放宽理由匹配或
  修改本契约判据换取绿色。本线无"负结果"研究空间：执法门要么闭合缺口
  （成功），要么未闭合（实现失败）。

## 10. 资源上限与停止条件

1. 资源：纯代码 + 测试；全量回归一次并行 + 一次串行对账（与仓库其他线同一
   预算量级）；无 DB 写入、无搜索墙钟消耗。
2. 停止条件（实现期）：RED 出现无关失败 → 停下诊断（§2.2(3)）；发现需要
   改采样侧/数据门/注册表数据才能通过 → 停下修订本契约，禁止顺手越界。
3. 停止条件（集成期）：若 A 线 p11（REWARD 15）先行合入 main，B 线实现分支
   合入前必须集成最新 main 并重跑验证矩阵（§12.3）；G1 的 reward pin 随
   导入常量自动更新，fixtures 使用动态 `REWARD_VERSION`（test_promotion.py
   L26 既有写法），禁止手工钉死 "14"。
4. 任何重跑用新 verdict 输出路径；禁止覆盖历史 verdict。

## 11. 批准与实施边界

- 批准流程：经契约门禁评审（t9）通过后生效；批准记录补记于本节。
- 实施顺序（§2.2、§12）：契约 PR（本文）先行合入 main → impl-b 从最新 main
  创建实现分支：RED（§7）→ 最小实现（§5/§6 范围）→ GREEN/回归（§8 矩阵）→
  原子提交 → 独立审核（t22）→ 集成合入（t27，串行于 A 线 t26 之后）。
- 与 A 线的文件所有权边界（§12 并行规则）：B 线唯一代码改动面 =
  `promotion.py` + `feature_registry.py`（新增报告函数）+ 新测试；不触碰
  `reward.py`/`config`/`evaluation.py`/`ashare_data/gates.py`/采样侧模块。
- 本契约不授权 push/远程 PR；合并由 integrator 按串行小 PR 流程执行；
  全程仅本地。
- 未运行/未验证项在实现报告中如实列出；本文不预写任何验证结果（§11.1）。
