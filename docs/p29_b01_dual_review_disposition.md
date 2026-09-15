# P29 B01 双审处置记录（dual-review disposition）

- 审核对象：分支 `codex/b01-pit-data-quality`，实现 commit `5013423`，
  测量日志首版 `ab68ff0`。审核者与实现者（本包执行模型）相互独立。
- 审核授权：用户 2026-09-15 直接指令「开两个子agent来负责审核你的工作，
  除此之外不要开任何多余的子agent」——恰好两个独立审核：合规 +
  技术/验证（p29 §9.7 的双审要求），均只读、未写库、未开子agent。

## 1. 首轮裁决

| 审线 | 裁决 | 摘要 |
|---|---|---|
| 合规（review-compliance） | **PASS** | 批准链、范围纪律、版本影响诚实性（含独立裁决：`DATA_QUALITY_REPORT_VERSION` 不入 `_VERSION_IMPORTS` 符合授权记录裁决 7 与 p29 §4.2）、RED/GREEN 与测试诚信、I9/I12/I13/G1–G8 零改动、I14 只读、测量日志对账、AGENTS §13 六维度——全部 PASS；独立复现 1210P/1S |
| 技术（review-technical） | **NEEDS_REVISION** | 全部关键数字独立复现逐位一致（10,701 / 61,272 / 2,451,953 / 2,390,681 / 8,564 / 272,948÷913,619 / 0 混合格式 / I5 零违例 / 资源上限）；26/26 测试通过；但发现 1 high（F1）+ 5 low（F2–F6） |

## 2. 发现与处置（逐条闭合）

| id | 严重度 | 问题（摘要） | 处置 | 闭合证据 |
|---|---|---|---|---|
| **F1** | **high** | I8 停牌对账 endpoint 常量用 Tushare API 名 `suspend_d`，台账权威命名是 V2 表名 `daily_suspension`（写入侧 `backfill_v2.py:337` 以表名入账、store 以 `V2_TABLES` 校验）→ 真实库 0 批次命中，20141013/20240412（各有 succeeded+0 行批次）被误判漏采并入缺口清单，20260907 才是真漏采；报告 JSON 与日志三处失真。fixture 复用同一错误常量致 R3/R4/R15 全绿而真实路径失效 | 常量改为台账权威表名；测试钉三重（= `daily_suspension`、∈ `V2_TABLES`、∈ 族声明 tables）；fixture 改用真实约定；新增负例钉（错误命名不构成零事件证据，fail-closed 默认保持）；payload 增加 `suspension_endpoint`/`ingest_batches_seen` 诊断字段；日志 §2.1/§4/§5 evidence-only 更正 | 实现者独立复核台账（`endpoint LIKE '%susp%'` → 仅 `daily_suspension` 3,083 批次；两批次 ingest_id 与审核者给出的一致）；修复后重测 zero_event=[20141013, 20240412]（证据引用批次）、missed=[20260907]、缺口清单 −2 |
| F2 | low | daily/price_limit/margin 的 detail 无条件打印 `lag N <= T`，N>T 时为假陈述 | 三处改条件方向（与 industry_index 一致）+ fail 路径测试钉（`lag 4 > 3`，`<=` 不出现） | 新测试通过 |
| F3 | low | `month_end_snapshot_intervals` 不校验 open_sessions 有序/唯一（bisect 前提） | fail-closed 断言 + 乱序/重复输入抛错测试 | 新测试通过 |
| F4 | low | margin 全会话缺行不影响四态（与 daily 族「停牌合法无 bar」豁免不同质且未说明） | 族活跃期（首个采集会话起）缺行 → fail + 缺口；族前期会话出分母（`pre_family_sessions`）；两个测试钉；日志 §2.2 更正 | 新测试通过；真实库钉定日 margin 转判 fail（20260904/20260907 为真实缺陷） |
| F5 | low | 缺口样例无 ORDER BY，证据样例不可复现 | 反连接与 invalid-adj 查询加 `ORDER BY ts_code, trade_date` + 确定性测试 | 新测试通过 |
| F6 | low | 事件族无 I3 声明、四分项不全 | `EVENTS_AUDIT_TIME_SEMANTICS`（unrestorable + I4 Tier 注记）进 shipped-declarations 测试；build_events 输出 revision 分项 + detail 显式四分项豁免边界 | 新测试通过 |
| C-1 | low（合规） | 日志 §7「detail 字符串语义不变」可能被读成逐字节不变 | §7 增补「加法性追加计数、原语义段逐字保留」 | 已改 |

**处置原则**：F1 的修复方向 = 审核者 requiredFix 第 1/2/3 项全做；第 4 项
（可选加固「0 批次判 unknown」）经评估**不采纳原案、以等价更强方式落
地**——0 批次仍按 I8 fail-closed 全判漏采（default 不放宽），但
`ingest_batches_seen` 使「命名漂移」在证据中可见（原案会把「ledger 缺
失」与「命名错位」混为同一 unknown，反而弱化区分）。此偏离在日志 §5
与本表留证，交复核裁决。

## 3. 修复轮验证

- B01 测试：**34 passed**（26 原有 + 8 修复钉；RED 先行：修复钉在实现
  前实测 8 failed）。
- 修复后真实库重测（同探针）：审计模式 67.2s / 2,111 MB（§8.2 限内）；
  头条数字 10,701 / 61,272 / 8,564 / 93.39% / 272,948÷913,619 / 895 /
  0 混合格式全部不变（修复不触及这些口径），唯一变化 = I8 分类更正
  （缺口清单 −2）与 margin 钉定日转 fail。
- 两分片回归：修复 commit 上复跑（见测量日志 §9）。

## 4. 复核轮（技术审核者，同一 agent）

**总裁决：PASS**。首轮 F1（high）与 F2–F6（low）全部闭合（逐项独立验证，
非采信声明）；对 0 批次处置的偏离**专项裁决：接受偏离**（三点理由：
I8 契约字面执行；「ledger 不可读→unknown」与「可读但 0 批次→missed→
fail」区分度更强；命名漂移经 payload/detail/负例测试三重机器可见）。

复核轮新发现 2 条 low，已随后续小提交闭合：

| id | 问题 | 处置 |
|---|---|---|
| F7 | build_daily 方向条件误用整体 status 作键：adj 缺陷触发 fail 而 lag 新鲜时打印反向假陈述 `lag 0 > 3` | 方向改以 lag 为键（与 price_limit/margin 同款）；测试钉：adj-fail 场景 detail 含 `lag 0 <= 3`（先 RED 实测复现该假陈述） |
| F8 | 修复轮新增 payload 结构字段未 bump `DATA_QUALITY_REPORT_VERSION` | **bump 1→2**（加法性字段仍 bump 约定适用于独立版本；版本钉与四态测试同步更新，§10.1 白名单举证：钉值随 owner 常量的合法演进）；最终探针 JSON 证据以 report_version=2 生成 |

复核者实测验证矩阵：B01 34 passed、universe/pit_import/gates 67 passed、
真实库头条数字全部不变、两库 mtime 未变（修复轮零数据库写入）。

## 5. 终态

- 分支 commit 链：`41f771c`（批准）→ `a927d20`（RED）→ `5013423`
  （GREEN）→ `ab68ff0`（测量日志）→ `ffe1ab4`（修复轮 F1–F6）→
  `5beb4e1`（处置+日志更正）→ 复核轮修复 commit（F7/F8 + 版本 bump）。
- 双审终态：合规 PASS；技术 PASS（复核轮）。
