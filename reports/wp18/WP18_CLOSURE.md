# WP18 收口报告（WP18_CLOSURE.md）——最终汇报五项未覆盖逐项处置

- 日期：2026-09-27。收口包：**p57**（测试 `tests/test_p57_wp18_closure.py`，前缀 p57，11 条）。包性质：**收口（closure only）**——零新增评价行/回测/评分运行，不追跑 slow 压力，不触发方法修订，不重启假设。
- 收口对象：WP18 最终汇报列出的五项未覆盖（slow 成本压力、WP17 资格审定、方法修订、10 条 deferred minor、E8 文案）+ `WP18_REPORT.md` §9 可处置项。
- 同题材在盘文件：执行员逐项处置表 `reports/wp18/P57_CLOSURE_REPORT.md`；本报告为收口最终汇报载体，由报告撰写员独立复核后撰写——**本报告引用的全部关键断言均在撰写会话内重读盘面或重跑命令复核**（命令与结果见附录 A）。
- 描述性框架（与 WP18_REPORT 页脚同一契约）：本报告全部结论为开发验证（2021–2025 测试窗）下的描述性结果；除 R2 演练件（qualified=False）外未生成 ClaimRecord，未运行 WP17 max-t 资格审定，任何语句不可作为名义 5% 显著性或样本外推断发布；机械筛选门槛为研发资源分配门槛，非统计显著性证明也非市场规律。

---

## 一、五项未覆盖逐项处置表

| # | 未覆盖项 | 处置（本包实际动作） | 本会话复核证据 | 归向 |
|---|----------|----------------------|----------------|------|
| 1 | **slow 成本压力 0 行**（1x 初筛未过，拒授权 C2） | **按设计关闭，零追跑**（追跑=对已拒绝初筛的结果选购）：(a) 双载体核验逐字一致——`slow/cost_stress.json` `screen_adjudication.{A,B}.skip_reason` ↔ C2 账本 4 条 `screen:1x_initial` 行 `payload.skip_reason`；(b) `stress_rows_authorized=0`、`row_count=0`（rows 空）；(c) 60 个已物化未运行行键在计划中保全（slow C2 40 + medium C1:A 20），缺行恰为 §5 初筛拒授权的压力臂；(d) `measurement_incomplete` 语义确认=设计内状态（初筛拒授权，非测量失败、非经济负结果）。slow 两规则 1x 初筛均负：A 累计净超额 −0.265384、B −0.214913（`slow/cost_stress.json` `screen_adjudication`，dynamic 腿；medium 参考值 A −0.098043 拒 / B −0.038089 过，`medium/cost_stress.json`）。 | `closure_verifications.json`（本会话重跑再生）`r1_slow_cost_stress_closure`：`dual_carrier_verbatim_identical=true`、`stress_rows_authorized=0`、`ledger_rows_total=6`（4 screen + 2 stage:C2）、保全 60=20+40、`slow_missing_by_stage={"C2":40}`、`medium_missing_by_stage={"C1":20}`、两域 `label=measurement_incomplete`/`verdict_valid=false`；亲读账本 `data/r1/wp18_slow/runs/wp18_c2/wp18-c2-20260927/trials.jsonl`（6 行，skip_reason 逐字同上）；测试 `test_p57_r1_slow_skip_dual_carrier_verbatim`、`test_p57_r1_unrun_row_keys_preserved_in_plans`（slow 计划 8 个 C2 臂、`expected_row_count=280`、`reconcile_full.n_missing=40` 且 note 明载缺行恰为 §5 筛除压力臂） | **按设计关闭** |
| 2 | **WP17 资格审定未启用**（WP18 全部结论为描述性） | (a) **机械声明审计**：`scripts/p57_claim_audit.py`（词表按 FROZEN_17 措辞契约在脚本头部声明）扫 wp18 全部 47 个 MD/JSON/JSONL 产出 → **213 命中：211 描述性合规 / 2 需标注（均人工复核后保留原文）/ 0 疑似推断**，`suspected_inferential_unhandled=0` ⇒ 无需措辞 errata、语料零改写；(b) **描述性路径实弹演练**：对 WP18 真实统计量（`_analysis/derived.json` slow combo B 折 0 active IR）以 `evidence=None` 组装 ClaimRecord → `qualified=False`、`reason_codes=[EVIDENCE_NOT_REGISTERED]`、冻结 no-reliable-inference 句式渲染（含 "no nominal-5% significance or promotion claim is made"）、五组字段齐备、**保存不被阻断**（tmp 写读回环 + CLI 两行输出）。 | `claim_audit.json` summary（47/213/211+2+0/0）+ 本会话 fresh 复跑（`--output` 至临时文件）：summary/hits/files_scanned 三项与盘上全等；2 条需标注定位 `ANALYSIS.md:275`（措辞纪律章节标题，元话语）与 `WP18_REPORT.md:185`（bisect 测试隔离记录），处置见 `audit_findings_p57.md` F2/F3；演练测试 `test_p57_r2_claim_record_descriptive_drill_real_wp18_statistic`（断言体亲读）+ `test_p57_r2_claim_audit_scanner_semantics`、`test_p57_r2_claim_audit_ran_over_wp18_products` | **已处置（审计+演练，无需勘误）** |
| 3 | **方法修订未触发** | 复核登记 `ashare_model/method_revision_registry.json` = `{"episodes": []}`（从未开启触发记录）+ **守卫活性实测**（`require_open_episode(None)` → `G1_REFUSED` 逐字；attested 版本 1 免 episode）+ 正式记录关闭：重开须走 W6 episode 协议。 | 亲读 registry（本会话输出 `{"episodes": []}`）；`closure_verifications.json`（重跑再生）`r3_method_revision_zero_trigger`：`episodes=[]`、`require_open_episode_none="G1_REFUSED"`、`attested_version_needs_no_episode=true`；测试 `test_p57_r3_method_revision_zero_trigger_and_guard_liveness` | **核验关闭（零触发）** |
| 4 | **10 条 deferred minor**（含 WP17/WP18 两轮审核明细未落盘） | (a) §9 盘面可核实各条已逐条处置（见 §二八条表）；(b) **流程固位**：新增 `docs/p57_audit_findings_persistence.md`（规则：审核员同会话把全部 findings 逐条写 `reports/wpNN/audit_findings_pNN.md`，每条含编号/严重度/位置/发现/证据/处置/状态，未处置不得留白，含模板；背景=WP17 §6 与 WP18 §9.5 两轮同型缺口）；**本包自己的审核按其执行（吃狗粮）**。 | 两文件亲读在盘；狗粮执行情况见 §四（内审 F1–F6 + 独立审核员A A1–A4 + 审核员B B1–B3，13 条全部 resolved）；测试 `test_p57_r4_audit_findings_persistence_rule_and_dogfood` | **已处置（规则自本包生效）** |
| 5 | **E8 消息文案不一致**（拒绝消息打印 1e-12 形式而在役门槛 1e-9） | **RED→GREEN 仅消息披露，行为零变化**：先写测试断言消息含 1e-9 字面形式（`1.000000001`）→ RED 失败（实测消息 `sum(target)+cost=1.00076 > 1.000000000001`，见红证据逐字）；修复仅把打印值改为 `1.0 + _P52_C1_REFUSAL_TOLERANCE`（渲染 `1.000000001`）；**:563 比较 `shortfall > _P52_C1_REFUSAL_TOLERANCE` 与双常量语义逐字不动**（:35 `_P52_C1_TOLERANCE=1e-12` / :51 `_P52_C1_REFUSAL_TOLERANCE=1e-9`）；p52 家族回归全绿；FROZEN_A §8.4 追加（追认双常量设计）；errata E10 追加执行记录。 | 亲读 `ashare_model/reward_baskets.py`（:35/:51/:563/:586 现值如上，:577–580 为 p57 注释、raise 块随后）；`reports/p57_red_evidence.txt`（RED 逐字 1 failed in 4.83s → GREEN 2 passed、p52 12 passed、133 passed）；`reports/wp16/FROZEN_A.md` 344 行、§8.4 在 :338–344、§8.1–§8.3 头 :303/:321/:325 原位；`errata/errata.jsonl` 10 条（E1–E9 未动，E10 尾追加，id `E10-wp18-p57-e8-refusal-message-disclosure-executed`）；本会话亲跑 p52/p55/p56 回归 133 passed（附录 A #2）；行为锚定测试 `test_p57_r5_e8_behavior_anchors_unchanged`、消息测试 `test_p57_r5_e8_refusal_message_discloses_operative_tolerance`、追加记录测试 `test_p57_r5_e8_append_only_records_present` | **已修复（无行为变化）** |

---

## 二、WP18_REPORT §9 八条逐条处置（归向：已修复 / 按设计关闭 / standing 重申 / noted）

| §9 条 | 内容（摘要） | 本包处置 | 归向 | 证据 |
|-------|------|----------|------|------|
| §9.1 | E8 已知问题（拒绝消息文案 1e-12 vs 1e-9） | 执行修复（§一 #5） | **已修复** | 红证据、FROZEN_A §8.4、errata E10 |
| §9.2 | 契约裁决可推翻（E2/E9 追认，用户仍可推翻） | 本包不触碰，收口重申（§三.1） | **standing（用户可推翻）** | `closure_verifications.json` `standing_items_restated.s9_2`；errata E9 standing |
| §9.3 | 域门形态偏差已披露未回滚（E4） | 编排层管线注记落盘：未来包复制流水线须按域拆信号诊断阶段或将门定义与阶段划分对齐，使冻结形态可逐字执行；不回滚、不重跑 | **noted（不回滚）** | `closure_verifications.json` `domain_gate_pipeline_note`（source=WP18_REPORT §9.3 + errata E4） |
| §9.4 | status_full=measurement_incomplete（两域，设计内） | 语义确认 + 60 行键保全声明（§一 #1） | **按设计关闭** | `closure_verifications.json` `r1_slow_cost_stress_closure.status_labels`、`unrun_row_keys_preserved` |
| §9.5 | 三路审核 17 条明细未落盘（工作流日志不可达） | 流程固位（`docs/p57_audit_findings_persistence.md`）+ 本包狗粮执行；**wp18 历史 17 条明细仍不可达**（工作流日志缺口，如实保留，不臆造） | **规则生效；历史缺口如实登记** | WP18_REPORT §9.5 自认 + 本包 `audit_findings_p57.md` |
| §9.6 | E3 缓存目录多出内容寻址条目 | 哈希同一性核验：slow_ab 与 medium_a 的 `ed8dc37c….npz`（各 1,684,390,858 B）sha256 相同且等于 errata E3 登记值（`81db3d19…`）；既有 `4ab7587a….npz` 在场未动；**核实过的无操作，不删除**（删除是对既有目录的进一步改写） | **closed（verified no-op）** | `closure_verifications.json` `e3_cache_entry_byte_identity`（本会话重跑脚本对两文件现场重算 sha256：`byte_identical=true`、`matches_errata_registration=true`） |
| §9.7 | 重启条件（停止表行 6：这批假设暂停） | 本包不重启，收口重申（§三.2） | **standing** | `closure_verifications.json` `standing_items_restated.s9_7` |
| §9.8 | 测量限制（DIVIDEND_YIELD 覆盖受限读数；slow 2021 折预算退化） | 收口重申（§三.3） | **standing** | `closure_verifications.json` `standing_items_restated.s9_8` |

---

## 三、standing 项重申（§9.2 / §9.7 / §9.8）

1. **契约裁决可推翻（§9.2）**：E2/E9 追认（FROZEN_A §8.3 双常量拒绝门、19 个被拒 slow Rule A 键重学）是**运行所有者级决定，用户仍可推翻**。若推翻，回退动作清单在 errata E9 standing（WP18_REPORT §1.4 摘要）：恢复 slow 方向书 31 键 partial、slow 账户/压力行按停止表行 1 停止经济解释、E2/§8.3 勘误作废、重走裁决。本包对该裁决零触碰。
2. **重启条件（§9.7）**：这批假设（medium 3 因子 + slow 5 因子及等权组合、A/B 择向、参考格）**暂停**（停止表行 4+行 6 裁决=停止）；仅新经济假设或新信息源值得重启；B-fulltrain 如需保留须以独立政策身份另行预注册（契约 §2.3）。
3. **测量限制（§9.8）**：DIVIDEND_YIELD 覆盖 0.291–0.361 的"无信息"读数受覆盖不足限制（measurement_incomplete 语义，不作经济负结果表述）；slow 2021 折预算约束使 dynamic 退化为 hold（Q3 机制注记）。

---

## 四、双路审核汇总（findings 已落盘）

- **落盘路径（唯一逐条载体）**：`reports/wp18/audit_findings_p57.md`，按 `docs/p57_audit_findings_persistence.md` 规则执行（同文件追加小节、不覆盖），共三节 13 条：
  - 内审（p57 收口执行员）：F1–F6，**6 条全部 resolved**（含声明审计 2 条 needs_annotation 的复核处置 F2/F3、扫描器词边界修正 F6、测试桩披露 F5）；
  - 独立审核员A（修复与演练）：A1–A4，**4 条全部 resolved**（含 E8 零行为变化五路核证、WP17 演练断言真实性、FROZEN_A 只追加 336→344 行、行号/声明漂移登记）；判定 **passed**；
  - 独立审核员B（范围与审计）：B1–B3，**3 条全部 resolved**（零新增评价行三路证明：写足迹/账本行数/data mtime；扫描器无豁免严格扫交叉核；狗粮断言弱代理实测）；判定 **passed**。
- 计数：13 条 = resolved 13 / standing 0 / deferred 0；**无 blocker、无 major**。两条独立审核均判定 p57 收口包 **passed**。
- 与 errata 的关系：涉及测量完整性/契约的处置另按 errata 纪律追加（本包 E10）；findings 文件不替代 errata。

---

## 五、质量门（通过，全部解决；本会话亲跑）

| # | 命令（cwd `D:\minequant\AlphaGPT`） | 结果（本会话实测） |
|---|---|---|
| 1 | `python -m pytest tests/test_p57_wp18_closure.py -q` | **11 passed in 9.39s** |
| 2 | `python -m pytest tests/test_p52_wp16a_cash_identity.py tests/test_p55_wp17_claim_qualification.py tests/test_p56_wp18_round.py -q` | **133 passed in 4.95s**（E8 行为零变化 + p52/p55/p56 基线不破） |
| 3 | `python scripts/check_test_shards.py` | **shard union == full set: 140 files across 4 shards**（p57 已注册） |
| 4 | `python scripts/p57_claim_audit.py --output <临时文件>`（fresh 复跑，不动盘上产物） | **47 files, 213 hits, suspected_inferential_unhandled=0**；与盘上 `claim_audit.json` 的 summary/hits/files_scanned 三项**全等**（211 合规 / 2 需标注 / 0 疑似） |
| 5 | `python scripts/p57_closure_verifications.py`（重跑再生，含对两份 1,684,390,858 B npz 现场重算 sha256） | **R1 dual-carrier identical: True；preserved row keys: 60；E3 byte-identical: True（registered match: True）** |
| 6 | `find data -newermt "2026-09-27 21:00" \| wc -l` | **0**（data/** 自本包时段起零写入，含本报告撰写会话） |

**结论：通过，全部解决**——五项未覆盖各归向已修复/按设计关闭/已处置/核验关闭（§一）；§9 八条全部归向落定（§二）；审核 13 条 findings 全部 resolved、双路独立判定 passed（§四）；上述命令本会话全部亲跑通过（附录 A）。

---

## 六、红线与零新增评价行声明

**本包零新增评价行、零回测、零评分运行。** 具体红线遵守（逐项与本会话核证对应）：

- slow 成本压力未追跑（`stress_rows_authorized=0`、账本 6 行无追加，§一 #1）；
- 方法修订未触发（registry `{"episodes": []}`，守卫反而实测为 fail-closed 活性，§一 #3）；
- 假设未重启（§三.2 standing 重申，零动作）；
- **data/** 既有内容零改动**：E3 条目只哈希核验不删除（§二 §9.6）；`find data -newermt` 零命中（§五 #6）；
- errata 只追加（E10，E1–E9 未动，10 条在案）；FROZEN_A 只追加（§8.4，§8.1–§8.3 原位，全文 344 行）；红证据只追加（`reports/p57_red_evidence.txt`）；
- E2/E9 追认维持 standing，本包不触碰（§三.1）；
- E8 修复仅消息披露：:563 比较与双常量语义逐字不动（§一 #5），p52 家族 133 passed 亲跑佐证；
- ANALYSIS.md 经声明审计 0 疑似推断 ⇒ 无需勘误，零静默改写（§一 #2）。

---

## 附录 A：本报告撰写会话执行的核查（命令与结果，cwd `D:\minequant\AlphaGPT`）

| # | 核查 | 方式/命令 | 结果 |
|---|---|---|---|
| 1 | p57 套件 | `python -m pytest tests/test_p57_wp18_closure.py -q` | 11 passed in 9.39s |
| 2 | 行为零变化回归 | `python -m pytest tests/test_p52_wp16a_cash_identity.py tests/test_p55_wp17_claim_qualification.py tests/test_p56_wp18_round.py -q` | 133 passed in 4.95s |
| 3 | shard 注册 | `python scripts/check_test_shards.py` | 140 files across 4 shards |
| 4 | 声明审计 fresh 复跑 | `python scripts/p57_claim_audit.py --output $TEMP/...` + json 对比 | 47/213/0 未处置；与盘上 claim_audit.json summary+hits+files 全等 |
| 5 | 核验脚本重跑（R1/R3/E3） | `python scripts/p57_closure_verifications.py` + json 读取 | R1 双载体 True / authorized=0 / 保全 60；E3 byte_identical=True 且 sha256=81db3d19…=errata 登记值；R3 episodes=[] / G1_REFUSED |
| 6 | data 零写入 | `find data -newermt "2026-09-27 21:00"` | 0 条 |
| 7 | slow 初筛双载体 | 读 `reports/wp18/slow/cost_stress.json` + `data/r1/wp18_slow/runs/wp18_c2/wp18-c2-20260927/trials.jsonl` | A/B 均 passed=false（−0.265384 / −0.214913）、skip_reason 两侧逐字一致、账本 6 行（4 screen + 2 stage）、row_count=0 |
| 8 | medium 初筛参考值 | 读 `reports/wp18/medium/cost_stress.json` | A 拒（−0.098043）/ B 过（−0.038089），各 20 行 |
| 9 | E8 代码现值 | `grep -n` + `sed -n 555,592p` `ashare_model/reward_baskets.py` | :35=1e-12、:51=1e-9、:563 比较用拒绝常量、消息 :586 打印 `1.0 + _P52_C1_REFUSAL_TOLERANCE` |
| 10 | FROZEN_A 只追加 | `grep -n "### §8" + wc -l` `reports/wp16/FROZEN_A.md` | 344 行；§8.1–§8.3 头 :303/:321/:325、§8.4 头 :338 |
| 11 | errata 在案 | 读 `reports/wp18/errata/errata.jsonl` | 10 条：E1–E9 未动 + E10 尾追加 |
| 12 | registry | 读 `ashare_model/method_revision_registry.json` | `{"episodes": []}` |
| 13 | 红证据 | 读 `reports/p57_red_evidence.txt`（163 行） | RED（1 failed in 4.83s，消息含 1.000000000001）→ GREEN（2 passed / 12 passed / 133 passed）逐字在案 |
| 14 | 审核明细落盘 | 读 `reports/wp18/audit_findings_p57.md`、`docs/p57_audit_findings_persistence.md` | 三节 13 条全 resolved；两名独立审核员均判 passed |
| 15 | 测试清单 | `grep -n "^def test_p57" tests/test_p57_wp18_closure.py` | 11 条，覆盖 R1×2/R2×3/R3×1/R4×1/R5×3/R6×1 |

---

*本报告只陈述可复核事实并给来源路径；全部结论为开发验证（2021–2025 测试窗）下的描述性结果：未运行 WP17 max-t 资格审定（R2 演练件 qualified=False），任何语句不可作为名义 5% 显著性或样本外推断发布；机械筛选门槛为研发资源分配门槛，非统计显著性证明也非市场规律。*
