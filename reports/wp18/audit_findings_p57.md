# WP18 审核 findings 逐条明细（p57，收口包自审）

- 审核角色：p57 收口执行员（本包内审；按 `docs/p57_audit_findings_persistence.md` 执行——吃自己的狗粮）
- 审核日期：2026-09-27
- 审核对象：p57 收口包全部产出（E8 消息修复、声明审计、核验脚本、流程固位文档）

## Findings

| # | 严重度 | 位置 | 发现 | 证据 | 处置 | 状态 |
|---|--------|------|------|------|------|------|
| F1 | minor | ashare_model/reward_baskets.py:582 | E8 修复仅消息披露：打印阈值由 `1.0 + _P52_C1_TOLERANCE`（1e-12 形式）改为 `1.0 + _P52_C1_REFUSAL_TOLERANCE`（1e-9 形式）；需确认零行为变化 | `grep -n "_P52_C1"` 两常量原值（1e-12/1e-9）；:563 比较逐字未动；`python -m pytest tests/test_p52_wp16a_cash_identity.py -q` → 12 passed | 行为锚定测试 `test_p57_r5_e8_behavior_anchors_unchanged` 钉住双常量与比较行；FROZEN_A §8.4 + errata E10 追加记录 | resolved |
| F2 | minor | reports/wp18/ANALYSIS.md:275 | 声明审计 needs_annotation 命中：标题「WP17 资格状态（推断性措辞纪律）」含"推断"且 ±160 字符窗内无合规标记 | `python scripts/p57_claim_audit.py`（hit: inference/ambient） | 人工复核：该标题本身是措辞纪律章节名（元话语），非推断性声明；正文按 append-only 不改写 | resolved |
| F3 | minor | reports/wp18/WP18_REPORT.md:185 | 声明审计 needs_annotation 命中：「bisect 证明：…失败原样」含"证明"且窗内无合规标记 | 同上（hit: proof/ambient） | 人工复核：软件工程意义上的测试隔离（bisect）记录，非统计显著性/样本外推断声明；不在 FROZEN_17 契约禁止面内；按 append-only 保留原文 | resolved |
| F4 | info | reports/wp18/claim_audit.json | 机械声明审计全局结果：47 文件、213 命中，211 描述性合规 / 2 需标注（F2/F3）/ **0 疑似推断**；强模式（nominal-5%、size supported、误报率上限、p 阈值、定向显著、市场规律）零未标记命中 | `python scripts/p57_claim_audit.py` 输出 + summary 节 | 无疑似推断 ⇒ 按任务书无需 errata 勘误；词表与标记清单在脚本头部可审核声明 | resolved |
| F5 | minor | tests/test_p57_wp18_closure.py `_NaiveFullInvestConstructor` | (C1) 拒绝的测试触发以桩替换 `PortfolioConstructor`（确定性朴素满仓输出）驱动真实会话循环——需披露该桩不经过真实构造器 | 测试 docstring 已声明；拒绝消息实测 `sum(target)+cost=1.00076 > …`，费用出自真实 `ExecutionCostModel`（`_session_cost_fraction` 未打桩） | 已披露；行为面由 F1 的锚定测试独立钉住（双常量 + :563 比较 + p52 回归） | resolved |
| F6 | minor | scripts/p57_claim_audit.py `proof` 模式 | 首跑 15 条 needs_annotation 中 14 条为 `provenance` 字段名内的 "prove" 子串误报（`portfolio_config_provenance` 等） | 首跑 hit 清单（map.md/ANALYSIS/FROZEN_18_PLAN 的 provenance 标识符） | 英文臂加 `\b` 词边界（中文"证明"不受影响）；复跑 213 命中、误报清零；修正依据写入模式 basis 注释 | resolved |

## 汇总

- 总数 6：resolved 6 / standing 0 / deferred 0
- 声明审计 needs_annotation 两条（F2/F3）均已人工复核并记录处置；wp18 语料
  零改写（append-only）。

---

# WP18 收口包独立审核 findings 逐条明细（p57，追加小节）

- 审核角色：p57 收口包独立审核员（审核员A-修复与演练；按
  `docs/p57_audit_findings_persistence.md` 追加小节，不覆盖上方内审）
- 审核日期：2026-09-27
- 审核对象：p57 全部改动——E8 修复（`ashare_model/reward_baskets.py`）、
  `tests/test_p57_wp18_closure.py`、`scripts/p57_{claim_audit,closure_verifications}.py`、
  `reports/wp18/{claim_audit.json,closure_verifications.json,_closure_recon.md,
  P57_CLOSURE_REPORT.md,audit_findings_p57.md(内审节)}`、
  `docs/p57_audit_findings_persistence.md`、FROZEN_A §8.4、errata E10、
  `reports/p57_red_evidence.txt`
- 章程核查结论（均本会话亲跑/亲读）：
  - **E8 无行为变化**：(a) `reward_baskets.py:562-565` 比较块与
    `_closure_recon.md` §2（修复前只读侦察快照，消息当时渲染
    `1.000000000001`）逐字一致；双常量 :35 `1e-12` / :51 `1e-9` 与
    §8.3 时代 `p52_red_evidence.txt:203-232` 摘要一致；(b) 消息与在役门
    一致有测试钉住且亲跑通过（11 passed）；(c) 消息不被任何代码解析——
    repo 内（ashare_model/ashare_portfolio/scripts/tests/webapi/webui/
    dashboard，除 test_p57 外）grep "cannot fund its execution fees" 与
    两个字面形式零命中；(d) `p52/p55/p56` 回归 133 passed 亲跑；(e) RED
    真实失败有独立旁证：历史账本
    `data/r1/wp18_slow/runs/wp18_stage12/.../trials.jsonl` 20 条拒绝行
    原文即 `sum(target)+cost=1.000000000001 > 1.000000000001`（亲
    grep），修复前断言 `"1.000000001" in message`（且旧形式
    `not in`）必然失败。
  - **WP17 演练断言真实**：
    `test_p57_r2_claim_record_descriptive_drill_real_wp18_statistic`
    用盘上真实统计量（`_analysis/derived.json` slow combo B 折 0 active
    IR）组装，断言 `qualified is False`、`reason_codes ==
    [EVIDENCE_NOT_REGISTERED]`（`claim_record.py:78` 冻结枚举）、reason
    等于 `no_reliable_inference_reason` 渲染、五组字段齐备、tmp 保存读回
    不阻断、CLI 两行输出——全部亲跑通过；`build_claim_record` 本身不抛
    （docstring :289-295 与 `_auto_evidence` 的 `except Exception: return
    None` 共同保证非阻断语义）。
  - **FROZEN_A §8.4 只追加**：侦察时该文件 336 行、末节 §8.3（:325）；
    现 344 行，新增恰为 §8.4（:338-344，+8 行），§8.1-§8.3 原位未动。
  - **红线**：零新增测量（p57 脚本均为文本扫描/哈希/读盘；`find data
    -newermt "2026-09-27 21:00"` 零命中——data/** 在本包时段零写入）；
    slow 未追跑（R1 verify 双载体 True、authorized=0）；方法修订零触发
    （episodes=[] + G1_REFUSED 亲跑）；errata E1-E9 未动（E10 尾追加，
    E8 原文行号 :583 未随代码移位更新——append-only 的一致性旁证）。
  - 复跑一致性：`scripts/check_test_shards.py` 140/140；fresh
    `p57_claim_audit.py`（输出到临时文件）与盘上 `claim_audit.json`
    summary/hits/files_scanned 三项全等（47 文件、213 命中、0 疑似、
    2 需标注）；R1/R3 verify 亲跑复核；E3 两份 1,684,390,858B 条目
    sha256 独立 hashlib 重算 = 脚本值 = errata E3 登记值
    （81db3d19…），文件在场未删除。

## Findings（独立审核）

| # | 严重度 | 位置 | 发现 | 证据 | 处置 | 状态 |
|---|--------|------|------|------|------|------|
| A1 | minor | tests/test_p57_wp18_closure.py:107 | 测试 docstring 引用的拒绝块行号 `:577-585` 为 RED 时代行号；修复追加 :577-580 四行注释后 raise 块实际在 :581-589，docstring 未更新 | 当前盘 `reward_baskets.py` :577-580 为 p57 注释、:581 为 raise；`_closure_recon.md` §2 记录修复前 raise 在 :577-585 | 行号引用漂移仅影响可读性，断言不依赖行号（异常消息触发实测）；登记，未来包触碰该文件时顺带对齐 | resolved（登记） |
| A2 | minor | scripts/p57_claim_audit.py:60-63 vs :137-143 | 脚本头部 docstring 声明的排除清单列 4 项（漏 `P57_CLOSURE_REPORT.md`），`EXCLUDED_NAMES` 实为 5 项——"declared for auditability" 的头部声明与代码不一致 | 两处亲读对比 | 收口报告因被排除而未经机械扫描，其措辞合规由页脚自声明 + 本独立审核人工复核兜底（已复核：页脚与正文无推断性声明）；docstring 漂移登记，未来包加尾注更正 | resolved（登记） |
| A3 | minor | scripts/p57_claim_audit.py `directed_sig` 等模式 | 英文推断臂覆盖有限（如 outperform / beats / superior 不在词表），机械扫描理论上可漏英文定向措辞 | 亲跑探针 `grep -rniE "outperform|beat[s ]|superior|alpha of|predict[s ]|edge over|robust gain"` over reports/wp18（*.md/*.json/*.jsonl，排除 claim_audit.json）→ 零命中，本包语料无实际漏检 | 词表 append-only 允许加尾；未来包复用扫描器前先对本语料跑同型英文探针；本包无需动作 | resolved（登记+探针已跑） |
| A4 | minor | reports/wp18/_closure_recon.md 附录 2 ↔ P57_CLOSURE_REPORT.md §四 | 侦察预告"ANALYSIS.md §10 五组字段别名漂移按红线以 **errata 追加**处置"，实际无对应 errata 条目（E1-E10 无此条），收口报告只写"未发现需勘误项（声明审计 0 疑似）"——预案与执行的差异未交代 | 漂移本身亲验为真：ANALYSIS.md:277 写 `statistics/multiplicity/operating-point/structure/qualification`，`claim_record.py:310-342` 冻结键为 `statistics/method/task_actuals/calibration_evidence/qualification`；漂移非推断性声明（不触发声明审计 errata 路径），但侦察已预告处置 | 漂移是描述性、零测量完整性影响，不改写 ANALYSIS 正文（append-only）正确；本 finding 即登记载体——读者引用 ANALYSIS §10 字段名时须对照 claim_record.py 冻结键名；p58+ 如需可加对照勘误 | resolved（登记） |

## 汇总（独立审核）

- 总数 4：resolved 4 / standing 0 / deferred 0
- **无 blocker、无 major**：E8 修复确认零行为变化（:563/:562-565 比较
  与双常量对照修复前快照逐字未动；消息与在役门一致被测试钉住；RED 有
  账本级旁证；p52/p55/p56 133 passed）；WP17 描述性路径演练断言全部真实
  （qualified=False + EVIDENCE_NOT_REGISTERED + 保存不阻断，亲跑）；
  FROZEN_A §8.4 只追加（336→344 行，恰 8 行）；红线全项遵守。
- 本审核员判定：p57 收口包 **passed**。
# WP18 收口包独立审核 findings 逐条明细（p57，追加小节·二）

- 审核角色：p57 收口包独立审核员（审核员B-范围与审计；按
  `docs/p57_audit_findings_persistence.md` 追加小节，不覆盖上方内审与审核员A）
- 审核日期：2026-09-27
- 审核对象：p57 全部改动的**范围与审计面**——零新增评价行证明、声明审计词表
  与处置越权、核验 JSON 与盘面事实、流程固位执行（E8 行为面已由审核员A覆盖，
  本节仅做独立旁证）
- 章程核查结论（均本会话亲跑/亲读）：
  - **零新增评价行**：p57 时段（2026-09-27 21:00 后，红证据时间戳 21:18:56）
    全仓写足迹 `find . -newermt "2026-09-27 21:00"`（排除 logs/缓存/工具目录）
    = 恰 14 个本包申报文件；`find data -newermt` 同口径**零命中**；七本 wp18
    账本行数亲数 18/64/182/46 + 156/262/6 与收口前侦察记录逐一致（零追加行）；
    新脚本/测试 import 面检查无回测/评分/压力入口（R5 测试对 1×3 合成数据跑
    拒绝路径，桩构造器，不写任何账本/缓存）。p57 套件 11 passed、
    p52/p55/p56 133 passed、shard 140/140 均亲跑复核。
  - **E8 零行为变化（独立旁证）**：`reward_baskets.py:563` 比较与
    `_closure_recon.md` §2 修复前快照逐字一致；:35=1e-12/:51=1e-9；旧消息形式
    `1.0 + _P52_C1_TOLERANCE` 源内计数 0、新形式计数 1（python 统计）；消息
    repo 内无解析方（producer 为跨行字面拼接，唯一命中=test_p57 断言）；
    FROZEN_A §8.1-§8.3 头在 :303/:321/:325 原位、§8.4 追加于 :338-344（全文
    344 行）；errata 10 条 = E1-E9 + E10 尾追加（E8 原文含历史行号 :583 未改，
    append-only 旁证）。
  - **核验 JSON 抽核**：`ashare_model/method_revision_registry.json` 亲读
    `{"episodes": []}`；E3 两份 1,684,390,858 B 条目独立 sha256 重算
    = 81db3d19… = errata E3 登记值（entry 内 64 位十六进制唯一候选亲验），
    既有 4ab7587a… 条目在场；slow skip 双载体 4 条 screen 行 payload 与
    cost_stress.json 逐字一致、authorized=0、账本 6 行；fresh 复算
    verify_r1/r3/e3（内存中，不写盘）与盘上 closure_verifications.json 逐字段相等。
  - **声明审计抽核**：对象面完整（52 个 md/json/jsonl = 47 扫描 + 5 个 p57
    自产物排除，无漏扫）；词表对照 FROZEN_17 契约原文（§3.3 串 C、§3.4 两分支
    模板、§4.1 "size supported"、§4.2/§4.3 at-most/误报率不超过）各模式有覆盖；
    fresh 扫描与盘上 claim_audit.json summary+hits 全等；211 条 marker 豁免
    逐条机械复验（记录的 marker 确在记录的 context 内命中，0 错配）。
  - **处置不越权**：语料文件 mtime 全部早于 21:00（最晚 WP18_REPORT 20:47）；
    2 条 needs_annotation 原文（ANALYSIS.md:275 / WP18_REPORT.md:185）逐字
    在场；0 疑似推断 ⇒ 无需 errata、DISPOSITIONS 空、语料零改写。

## Findings（审核员B-范围与审计）

| # | 严重度 | 位置 | 发现 | 证据 | 处置 | 状态 |
|---|--------|------|------|------|------|------|
| B1 | minor | scripts/p57_claim_audit.py `WINDOW=160` 分类器语义 | ±160 字符合规标记窗存在**前置性宽松**：无关标记落在窗内可把强模式定向声明降为 descriptive_compliant——对抗探针（"统计上显著优于基准…这就是市场规律" + 100+ 字符外一个"描述性"）实测 3 条强命中全被豁免 | 本会话 `mod.scan_text` 对抗探针输出 [(directed_sig, compliant)×2, (market_law, compliant)] | **对本包语料零实际影响**：关闭全部 marker/key 豁免的严格重扫显示语料 66 处强模式出现全为冻结否定标注本身（"非…市场规律"/"不可作为名义 5%…发布"），无肯定式定向声明；登记为复用前置条件——未来包复用扫描器须同跑严格无豁免版作交叉核（或收窄 marker 作用域至同从句） | resolved（登记+严格扫已跑） |
| B2 | minor | scripts/p57_claim_audit.py 词表/对象面 | (a) 英文定向动词（outperform/beats/superior/predict 等）不在词表；(b) .py 产出面不在扫描范围（仅 md/json/jsonl）——原则性覆盖缺口 | 独立探针（与 A3 不同词集）：`grep -rniE "outperform\|beat[s ]\|superior\|predict…"` 语料零命中；中文定向探针（跑赢/击败/可以预测/稳定盈利等）零命中；裸 p 值探针唯一命中为 `single_weight_ca`**`p=0.05`** 子串（配置参数）；wp18 py 产物强模式探针仅命中冻结标注串本身 | 与 A3 独立互证：本语料无实际漏检；登记供未来包复用时先跑同型探针（含 .py 面） | resolved（登记+探针已跑） |
| B3 | minor | tests/test_p57_wp18_closure.py `test_p57_r4_…dogfood` | 狗粮断言是弱代理：只查 findings 文件不含某个"未…处置"字面三字串（测试 :477 的否定断言），不逐条校验每条 finding 带处置 | 亲读该测试 :469-477 + 亲读 audit_findings_p57.md（F1-F6/A1-A4 实际全部带处置与证据） | 实际合规由人工核验兜底（本审核员已核：内审+审核员A 两节 10 条全部 resolved、无留白）；**本节即实测踩中该耦合**——B3 初稿引用该三字串导致 test_p57_r4 失败，改写措辞后复跑 11 passed（处置过程留此为证）；未来包可加逐行表格列校验替代字符串否定断言 | resolved（登记+实测） |

## 汇总（审核员B）

- 总数 3：resolved 3 / standing 0 / deferred 0
- **无 blocker、无 major**：零新增评价行（写足迹/账本行数/data mtime 三路证明）；
  声明审计对象面完整、词表覆盖 FROZEN_17 契约面、结论在无豁免严格扫下稳健；
  处置零越权（语料 mtime 与原文保全）；核验 JSON 与盘面事实逐字段一致；
  E8 零行为变化有独立旁证；流程固位文档可用且本包三节审核均按其执行。
- 本审核员判定：p57 收口包 **passed**。
