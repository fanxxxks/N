# P30 A05 双审处置记录（dual-review disposition）

- 评审对象：分支 `codex/a05-doctor-eligibility`，提交链 `a18cac8`（A03 集成
  基线）→ `160eea1`（契约+索引）→ `5c5b2b3`（RED）→ `7a6d924`（实现）→
  `006c8e4`（修复 1）→ `df08c94`（双审修复批次）→ `33f70ac`（release 条件
  元组 bug 修复）。
- 评审方式：两个独立子 agent（技术审 + 合规审），全程只读；主会话逐条处置。
  两轮评审均针对 `006c8e4` 及其前链；修复落在 `df08c94`/`33f70ac`。

## 一、合规审（compliance review）

**裁定：PASS**（0 blocker / 2 medium / 2 low）。五项治理要点全部合规：
批准链条（用户直接指令依据、未虚援转授、契约先于实现且同 commit 登记
索引）、版本影响（`READINESS_REPORT_VERSION` 非 artifact-facing 论证与代码
事实相符、`_VERSION_IMPORTS` 未触）、测试白名单（§10.1 第 2 类举证充分、
强度不降）、证据完整性（引用精确 SHA、无编造数字、时间线自洽）、范围纪律
（diff 恰为 inScope 文件集；`reports/` 用户工作未触碰；doctor 退出码语义未变）。

| # | 级别 | 问题 | 处置 |
|---|---|---|---|
| C1 | medium | 测量日志全量回归行为占位符，§6 却引用为已完成 | **已修复**。全量套件在最终 HEAD `33f70ac` 实跑后把真实 passed/skipped/时长补入日志 §2/§4（见日志 §2 表与 §4.1）；§6 措辞只声明实际跑过的范围。evidence-only commit 在补记之后落盘。 |
| C2 | medium | R3 测试在实现中从「注入 ok gates」重构为 `gates=None`，契约文本未同步、日志未披露 | **已修复（补用例）**。新增 `test_r3_ok_gates_cannot_whitewash_a_failing_runtime`（ok gates 与 config error 共存 → runtime 仍 fail、stale pass 留在 data 维度、blocking 含 config_loaded）；原 `gates=None` 用例保留（对应 config-error 真实路径）。契约 §4 R3 本意（「runtime 不被洗白」）覆盖两例，契约正文不修订，以本记录为准。日志 §2 已披露拆分。 |
| C3 | low | 契约 §3.1 online_data_freshness 行在实现 commit 7a6d924 内被同步修改而非独立契约修订 commit | **登记为流程偏差**。修改方向与 §2.3/§2.4 冻结语义严格一致（N/A 永不聚合为 pass），同 commit 携带对应测试；后续契约文本变更优先独立成 commit。 |
| C4 | low | 契约 §4.1 白名单预声明未列 test_run_store.py | **登记预声明缺口**。实际修改已按 §10.1 第 2 类完整举证（合规审确认强度不降）；后续预声明须列全可能受影响的相邻测试文件。 |

## 二、技术审（technical review）

**裁定：NEEDS_REVISION → 两处 HIGH + 七处 medium + 五处 low 全部处置完毕
（2 修复验证 + 8 修复 + 1 登记后续包 + 2 措辞/断言收紧），复测全绿。**

| # | 级别 | 问题 | 处置 |
|---|---|---|---|
| T1 | HIGH | 本次 diff 向 research_doctor.py 引入 45 处 GBK 误解码乱码（含 error 级 findings 消息字面量），推翻「保序保文案」主张 | **已修复（重建）**。根因 = 主会话曾用 `pwsh Get-Content -Raw \| Set-Content` 做 replace，UTF-8 文件被按 GBK 误读再回写。以 `git show a18cac8` 原始 blob 为基、标记切片程序化搬移冲突循环（逐字节保真）+ 重放全部 p30 编辑，重建整个文件；`ast.parse` 通过；乱码探针 0 命中（鈥/鈶/搂/U+FFFD/锟），em-dash 23、⑥ 11、§ 5 全部为合法字符。新增守卫测试 `test_module_sources_have_no_encoding_mojibake`（四份源文件逐一断言）。修复 commit `df08c94`。 |
| T2 | HIGH | `_bound_paper_account` 对未持锁的只读 RunHandle 调 `close()`，会无条件 `unlink` 活跃 writer 的单写者锁 → 双写者窗口 | **已修复**。删除 `close()` 调用（RunHandle 无持久 OS 资源，ledger 按次 append 打开；docstring 说明禁止自动清锁的契约依据）；`RunHandle(...)` 构造移入 try 并捕获 `LedgerIntegrityError/OSError`（顺带关闭 T5 的崩溃面）。回归测试 `test_high2_gather_paper_never_deletes_an_active_writer_lock`：attach_run 持锁期间跑 `gather_paper`，断言锁文件仍在。修复 commit `df08c94`。 |
| T3 | medium | production_gates 检查只有计数摘要，未嵌逐 G 明细（§3.1 冻结项） | **已修复**。`per_gate`（name/ok/detail × 8）嵌入该检查项；`test_production_gates_check_embeds_per_gate_detail` + 验收 JSON 实测 per_gate len=8。`df08c94`。 |
| T4 | medium | §3.2「非 pass 必须有解除条件」系统性欠账（D7 fail、fairness fail、lifecycle fail、source_run fail、unknown、freshness fail、carried 均缺） | **已修复**。为全部非 pass 路径补 release_condition（含 unknown 的「运行 X 后重评」类）；新增 `test_every_non_pass_check_carries_a_release_condition`（遍历全维度断言非 pass ⇒ 字符串型非空 release，blocking 同查）。`df08c94`。 |
| T5 | medium | gather_paper：重放失败 + bound 为空误判 not_applicable（未证实命题）；scan_errors 不进报告；RunHandle 构造在 try 外，账本损坏会崩 doctor | **已修复**。bound 空 + scan_errors 非空 → unknown（detail 摘要错误）；bound 非空 → pass 且 scan_errors 在 builder detail 中可见；构造入 try。测试：`test_paper_scan_corrupted_chain_is_unknown`（篡改哈希链 → unknown；存在合法 paper run 时 pass+scan_errors 并存——存在性已被证实、不完整性随明细披露）。`df08c94`。 |
| T6 | medium | config-error 最小报告与 `/api/readiness` 传 `artifacts=[]` → N/A 理由失实；store_root 硬编码 fail 指错修复对象 | **已修复**。builder `artifacts=None` → unknown（"inventory was not gathered"）；两处调用方改传 None；store_root 支持显式 status 覆盖，config-error 路径为 unknown。测试断言 API 场景 promotion binding=unknown。`df08c94`。 |
| T7 | medium | O5 provenance 只在成功载荷：404 路径（最需要 provenance 处）缺失；根解析异常会 500 | **已修复**。`get_run_detail` 根解析失败 → `RunStoreError("run root unresolved: … runs_dir_source=unresolved …")`（404）；缺 run 的错误消息携带 `runs_dir/runs_dir_source/runs_dir_exists`。测试：`test_r10_missing_run_error_carries_provenance` + `test_r10_real_root_consistency_without_injection`（不注入 resolver 的同根一致性）。`df08c94`。 |
| T8 | medium | 同一报告两把 dataset 参照系：readiness 用 A02 composite_id，findings 侧仍用主库 latest manifest id；对 A02 代际非 legacy 产物会自相矛盾（基线已有缺陷，当前环境仅 legacy 产物未触发） | **登记后续包**（不改本包语义：统一 findings 侧参照系会改 `healthy`/退出码行为，超出 A05 范围）。`_readiness_section` docstring 已如实声明分歧与依据（`df08c94`），后续以独立小变更统一并配契约修订。 |
| T9 | low | `_shanghai_today` 别名零消费者（§9.1 死代码） | **已删除**。`df08c94`。 |
| T10 | low | 契约称 `assert_paper_account_writable`「保留给真实调用方」，实际生产调用方为零 | **契约措辞已修正**（本记录同批提交）：如实写明「保留为公开 API；生产调用方为零，仅测试/审计语义消费者（P8-15① 仍在测其语义）；若需退役须按 §9.1 独立机械 PR 举证」。 |
| T11 | low | webapi `evaluated_at` 用本地时区，契约 §3.3 冻结 UTC | **已修复**。统一 `datetime.now(timezone.utc)`。`df08c94`。 |
| T12 | low | 恒真断言（`>= 0`）与 R10 by-construction/空洞分支 | **已收紧**。`by_status` 精确断言（pass 11 / N/A 6）；R10 直接断言 runs_dir 含 tmp_path；新增不注入的真实根一致性测试。`df08c94`。 |
| T13 | low | R2 拒绝象限只在纯输入层断言；gatherer 层象限无测试；R8 未测 REJECTED 排除 | **已补**。`test_paper_scan_gatherer_quadrants`（bound+无状态文件→unknown；状态文件+无绑定→fail；REJECTED run 经 §4 判别器真实构造且被扫描排除）。`df08c94`。 |

处置中自查发现并修复（修复批次复审产物）：`production_gates_g1_g8` 的
release_condition 因括号位置成为 1-tuple，人读输出打印成 `('…',)` ——
修正字面量（`33f70ac`）并在解除条件测试中加字符串类型守卫。

## 三、裁定汇总

- 合规审：**PASS**（4 条 findings 全部处置：2 修复 + 2 登记）。
- 技术审：**NEEDS_REVISION → 修复后复测通过**（2 HIGH 全修复并有回归
  守卫；7 medium：6 修复 + 1 登记后续包；5 low 全部处置；另修复自查发现的
  1-tuple bug）。
- 复测证据（`33f70ac`）：`tests/test_p30_a05_doctor_eligibility.py` 28 passed；
  相邻契约（research_doctor/webapi/run_store/p27/runspec）全绿；
  全量回归见测量日志 §2/§4；`compileall`、`git diff --check`、分片守卫通过；
  doctor 验收重跑（JSON 绑定 commit `33f70ac8640d`）结论不变：
  runtime PASS / snapshot PASS / data FAIL(G8 lag 4>3) / research FAIL /
  promotion N/A / paper FAIL——9 pass / 3 fail / 0 unknown / 5 not_applicable。
