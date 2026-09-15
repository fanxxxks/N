# P29 B01 PIT 与可交易数据质量测量日志（evidence-only）

- 被测实现：分支 `codex/b01-pit-data-quality`，实现 commit **`5013423`**
  （`feat(p29): B01 PIT/tradable data quality (I1-I14, R1-R16 GREEN)`）。
  前置：`41f771c`（p29 契约 DRAFT→APPROVED 批准回写）、`a927d20`（RED）。
  base = `f28f885`（`codex/a05-doctor-eligibility` tip；B01 依赖 A02/A04
  均在该祖先链上，`main @ aac4aad` 尚未合并 A03–A05）。
- 本日志只写「实际发生什么」（AGENTS §11.1/§11.4：所有数值在精确实现
  commit 上实测；不预写、不复制过时数字）。研究声明边界（p29 §7.8）：
  本包是**数据资格工程**，本日志的一切数字只证明边界、覆盖、缺口与
  可观测性，**不构成** alpha、收益、晋级或 paper 证据。

## 1. 命令与环境

| 项 | 值 |
|---|---|
| OS / Shell | Windows（pwsh 7） |
| Python | 3.13.12（项目参考运行时 `D:\minequant\.venv`；AGENTS §10.2 参考运行时） |
| CPU / RAM | Intel i7-9750H（6C/12T）/ 16GB，串行 |
| 主库 | `data/ashare.duckdb` 968.3 MB，mtime 2026-09-11 03:02:01 |
| v2 库 | `data/ashare_v2.duckdb` 7,588.8 MB，mtime 2026-09-08 11:00:58 |
| 探针脚本 | `D:\minequant\reports\AlphaGPT-B01-2026-09-15\run_b01_report.py`（仓库外，p29 §12 先例：探针不入提交；全部 `read_only=True`，I14） |
| 原始产物 | 同目录 `b01_report_audit_realclock.json`、`b01_report_audit_pinned_20260908.json`、`b01_report_spec_ret1_pinned_20260908.json`（机器可复核） |

验证矩阵（全部在 `5013423` 工作树 = 精确候选 commit）：

| 命令 | 结果 |
|---|---|
| `python -m pytest -q tests/test_p29_b01_data_quality.py` | **26 passed**（R1–R16 + 声明/I5/分片注册钉） |
| `python -m pytest -q <shard-data> <shard-model-2>`（参考 venv，465s） | **1210 passed / 1 skipped / 0 failed** |
| 同命令在 `C:\ProgramData\miniconda3` python | 1206P / **2F**——`test_lock_files` 两例，归因：该环境缺 `tushare` 包元数据（`importlib.metadata.PackageNotFoundError`），与被测 diff 无关（venv 中 `tushare==1.4.29` 存在且通过） |
| `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi` | exit 0 |
| `git diff --check` | 干净 |
| `python scripts/check_test_shards.py --check` | `103 files across 4 shards cover all 103 test files`（新测试文件同 commit 注册） |

未运行项（AGENTS §10.2）：`shard-model-1` 与 `shard-portfolio-trading`
两个分片本会话未运行——本次交付仅为本地 commit，未 push/未合并（§2.1.1
的 push 授权不在转授范围），变更面（ashare_data 五文件 + D7）由上述两
分片覆盖；若后续合并，按 §10.2 在精确 merge commit 上补全四分片。

## 2. 分域资格报告（真实两库，I9 四态；§7.2）

热路径资源（p29 §8.2 上限 10min / 4GB）：**实测 wall 73.5s、峰值 RSS
2,109 MB**（审计模式，真实时钟）——合规。

### 2.1 审计模式（真实时钟 evaluation_day=20260915）

| 族 | 状态 | 关键数字（精确） |
|---|---|---|
| `daily_bar_raw` | **fail**（新鲜度） | lag 4 > 3 会话；覆盖 2,390,681/2,451,953 = **97.50%**（分母 = 已上市 ∧ PIT universe ∧ 应当可观察，I6）；缺口 61,272 个 missing_bar 单元（例：000001.SZ 20140715 等——多为停牌日合法无 bar，报告只列缺口不定罪） |
| `daily_price_limit` | **fail**（I7 真实缺口） | **10,701 个 bar 单元缺 `daily_price_limit` 记录**（逐码逐日反连接枚举，按板块分类；例：001914.SZ 20140113 起，szse_main）；停牌族对账：**零事件 2 日（20141013、20240412，各有 succeeded+0 行批次）+ 漏采 1 日（20260907）**（§9 修复轮更正，证据引用批次 ingest_id） |
| `margin_detail_pit` | **fail**（新鲜度+缺口） | lag 6 > 2；2 个全会话无行：**20260904、20260907**（同步停止前的最后两个会话） |
| `industry_membership_pit` | pass | 覆盖 99.96% ≥ 98%，overlap 0，availability 完整 |
| `industry_index` | **fail**（新鲜度） | lag 5 > 3 会话 |
| `financial_version_tables` | pass | 报告期覆盖 **93.39%** ≥ 90%；真实缺口 **8,564** 个（代码,报告期）；修订版本键 **272,948/913,619**（29.9% 业务键有 ≥2 版本行——修订历史真实存在但**不完整**，I4 的 Tier 限制注记随报告输出）；非季末行 **10** 条（R11 质量发现，erratum #4 口径跳过） |
| `dataset_manifest` | pass | 复合身份 fast 校验通过 |
| `index_weight_pit` | pass | **I5 可见性规则 0 违例**（store 的 available_date 与「月末后首个开市会话」逐行一致） |
| `events_audit`（信息性） | **fail** | holder_number_pit lag **895** 会话（历史止于 2022-12-30 前后——p29 §1.7 的 F3 前置数据缺口实证）；forecast/express/repurchase lag 4–13。**信息性，不阻断纯价格基线**（I10） |

### 2.2 审计模式（evaluation_day 钉在 20260908，工程探针）

钉在最后 bar 次日时：daily/industry_index 转 pass，**price_limit 的
10,701 缺口与 events 的 holder 缺口不变**——证明这些是与时钟无关的真
实数据缺陷，不是新鲜度伪影。margin 在修复轮（§9 F4）后于钉定日亦判
**fail**（20260904/20260907 两会话在族活跃期内无行，属缺陷而非新鲜度
伪影；修复前误判 pass，因当时缺口不影响状态）。

### 2.3 RunSpec 模式（RET_1，纯价格因子集，I10 实证）

仅 `daily_bar_raw`、`daily_price_limit`、`dataset_manifest` 被判
（required=true），margin/industry/financial/index_weight/events 全部
`not_applicable`（非 pass，带理由）。**结论：纯价格基线不被
holder_number 缺口阻断**（p29 §1.7/I10 在真实库上的机器证明）。

## 3. 时间格式审计（§7.3，I1）

- 本次探针：主库 4 个消费列（daily_bar/calendar/stocks/constituents 的
  日期列）+ v2 13 个消费列全扫描，**mixed_format_findings = 空**。
- 与契约基线（§1.1：主库 daily_bar 118 行 ISO 混合）的差异归因：该
  118 行已由用户单独授权的 T46 有状态修复规范化（授权记录 §4quater(1)，
  `reports/alphagpt-ab/T46_iso_date_normalization.md`），本包只报告
  不修复的边界（裁决 5）因此以「零混合」通过。修复前基线以 T46 记录
  为准，不在本日志复制第二份（AGENTS §11.5）。

## 4. 可交易约束缺口（§7.4，I7）

- `daily_price_limit` 反连接（bar 单元 LEFT JOIN 约束表）= **10,701**
  个 (股票, 日期) 缺口，JSON 产物含逐码逐日样例与板块分类
  （star/chinext/sse_main/szse_main/bse/unknown）。
- 未使用任何硬编码涨跌幅填补；这些单元在 p19 §2.4 语义下本就
  fail-closed 不可作成交证据——B01 把「不可见缺口」变成机器清单。
- `daily_suspension`（§9 修复轮更正）：零行会话 3 个——
  **20141013、20240412 各有 1 个 succeeded 且 row_count=0 的
  `daily_suspension` 批次（`daily_suspension_20260904T071106Z_c964ca82`、
  `daily_suspension_20260904T090313Z_f27ab964`）→ 判「零事件」，不进
  缺口清单**；**20260907 无任何批次 → 判「漏采」**，进缺口清单。首轮
  报告因 endpoint 常量错用 Tushare API 名（`suspend_d`）把前两日误判
  漏采（F1），修复后与台账逐条对齐。

## 5. 零事件 vs 漏采对账（§7.5，I8；§9 修复轮更正）

- 对账权威 = `source_ingest_runs`（endpoint 匹配 + parameters_json 内
  日期匹配 + status）。**endpoint 权威命名 = V2 表名**（写入侧
  `backfill_v2.py` 把表名传给批次记录器，store 以 `V2_TABLES` 校验），
  修复轮把常量从 API 名 `suspend_d` 改为 `daily_suspension` 并加测试
  钉（F1）。台账中该 endpoint 共 3,083 批次。
- 真实库最终分类：**零事件 2 日**（20141013、20240412，证据批次
  `daily_suspension_20260904T071106Z_c964ca82`、
  `daily_suspension_20260904T090313Z_f27ab964`，均 succeeded +
  row_count=0）+ **漏采 1 日**（20260907，无批次）。payload 另带
  `suspension_endpoint` / `ingest_batches_seen` 诊断字段，命名再漂移
  时 0 批次对账在证据中可见。
- ledger 不可读或 endpoint 无批次时保持 fail-closed：零行一律判漏采
  （机器钉：`test_zero_event_requires_the_ledger_endpoint_name`）。

## 6. 阈值一致性（§7.6，I12）

报告全部阈值在调用时从 `ashare_data.gates` 模块属性读取（消费而非复
制）：`DAILY_ELIGIBILITY_MAX_LAG_SESSIONS`、`PRICE_LIMIT_MAX_LAG_
SESSIONS`、`MARGIN_MAX_LAG_SESSIONS`、`INDUSTRY_INDEX_MAX_LAG_SESSIONS`
（gates.py:73-76）、`INDUSTRY_MEMBERSHIP_MIN_COVERAGE=0.98`、
`FINANCIAL_REPORT_COVERAGE_MIN=0.90`（:77-78）、`FRESHNESS_TOLERANCE_
SESSIONS=3`（:63）。机器证据：`test_report_margin_threshold_is_consumed_
not_copied_i12`（monkeypatch gates 常量 → 报告 tolerance 随动）；
报告 payload 的 `threshold_sources` 逐项登记权威名字。

## 7. 版本与判定边界

- 版本影响表见实现 commit `5013423` message（p29 §4.1 重检：41 个
  `*_VERSION` owner；新增 `DATA_QUALITY_REPORT_VERSION=1` 独立承载报
  告结构，**不入** `_VERSION_IMPORTS`——报告是审计产物而非运行输入，
  先例 = A05 `READINESS_REPORT_VERSION`；曾试注册 owner 索引导致
  RunSpec extra=forbid 112 例构建失败，已回退并留证）。`DATA_TIER_
  VERSION`/`TIER_REPORT_VERSION` 不变（Tier 映射零改动，§4.0）。
- G1–G8 判定逻辑与阈值零改动（p29 §2.3-2）；主库 daily_bar 的格式
  发现标 G-space 归属注记，不改门禁 ok 值。既有 D7 verdict 语义未改
  （`check_runspec_data_eligibility` 仅内部重构为复用 `_financial_
  coverage` 富结果：**ratio、阈值判定与既有语义段逐字保留，detail 串
  为加法性追加计数**（gaps/versioned/non-quarter-end），非逐字节不变
  ——双审 C-1 澄清），相关 1210 例回归通过。

## 8. 剩余风险与移交项

1. 61,272 个 missing_bar 与 10,701 个 missing_price_limit 单元的**处置**
   （回填/修复）属有状态写操作，按 AGENTS §5.6 另立任务并单独请批
   （p29 裁决 5：B01 只报告）。
2. margin 20260904/20260907 两会话缺口、industry_index/事件族新鲜度
   随数据同步恢复自然消解，无需代码处置。
3. `index_weight_pit` 未纳入 A02 复合身份（p26 裁决 4）；若未来准入，
   走 I3-EXT 扩展契约 + `SNAPSHOT_PUBLICATION_VERSION` bump。
4. R01 前若做 F3 扩张，holder_number_pit 的 2023+ 缺口必须先单独检查
   （p29 §1.7；本报告的 events_audit 族已在真实库上给出缺口清单）。

## 9. 双审修复轮（evidence-only 更正，修复 commit 见 disposition 文档）

双审（合规 PASS + 技术 NEEDS_REVISION）发现 1 high + 6 low，全部处置：

| 发现 | 处置 | 修复后实测 |
|---|---|---|
| **F1 (high)** I8 对账 endpoint 常量 `suspend_d` ≠ 台账权威命名 `daily_suspension`（写入侧以表名入账），真实库 0 批次命中，20141013/20240412 两个有 succeeded+0 行批次的会话被误判漏采 | 常量改为台账权威表名 + 测试钉（`V2_TABLES` 成员 + 族声明 tables 成员）+ fixture 改用真实约定 + 负例钉（错误命名不构成零事件证据）+ `suspension_endpoint`/`ingest_batches_seen` 诊断字段 | 重测：zero_event = [20141013, 20240412]（证据引用批次）、missed = [20260907]、batches_seen = 3083；缺口清单 −2 |
| F2 三处 detail 无条件打印 `lag N <= T` | 条件方向（与 industry_index 一致）+ fail 路径测试钉 | fail 路径输出 `lag 4 > 3` |
| F3 `month_end_snapshot_intervals` 不校验 sessions 有序/唯一 | fail-closed 断言 + 乱序/重复测试 | 乱序输入抛 ValueError |
| F4 margin 全会话缺行不影响四态 | 活跃期（族首会话起）缺行 → fail + 缺口；族前期会话出分母；两测试钉 | 钉定日 margin 由误判 pass 转 fail（20260904/20260907 为真实缺陷） |
| F5 缺口样例无 ORDER BY 不可复现 | 反连接与 adj 查询加 `ORDER BY ts_code, trade_date` + 确定性测试 | 样例跨运行可复现 |
| F6 事件族无 I3 声明、四分项不全 | `EVENTS_AUDIT_TIME_SEMANTICS`（unrestorable + I4 注记）+ revision 分项 + detail 显式豁免边界 | 声明进 shipped-declarations 测试 |
| C-1 (合规，low) 日志 §7「detail 字符串语义不变」易误读 | §7 增补「加法性追加计数、原语义段逐字保留」 | 已改 |

修复后复测（同一探针，修复 commit 工作树）：审计模式 **67.2s /
2,111 MB**（§8.2 限内）；B01 测试 **34 passed**（26 原有 + 8 修复钉）；
`daily_price_limit` 缺口样例 21 = 20 缺约束 + 1 漏采（修复前 23 = 20 + 3
误判）。10,701 / 61,272 / 8,564 / 93.39% / 272,948÷913,619 / 895 等头条
数字不变（修复不触及这些口径）。

## 10. 复核轮与终态

- 技术复核：**PASS**（F1–F6 闭合确认 + 偏离接受；详见 disposition §4）。
- 复核轮新发现 F7（daily 方向条件误用 status 作键）/ F8（payload 结构
  增加未 bump 版本）→ 已闭合：方向改以 lag 为键；`DATA_QUALITY_REPORT_
  VERSION` 1→2（加法性字段仍 bump；版本钉与四态测试同步更新）。
- 终态探针（复核轮修复 commit 工作树）：审计模式 **69.2s / 2,108 MB**；
  三个 JSON 证据以 **report_version=2** 重新生成（`D:\minequant\reports\
  AlphaGPT-B01-2026-09-15\`），头条数字全部不变。
- 回归：shard-data + shard-model-2 于修复轮 commit **1218P/1S**；
  shard-data 于终态 commit **543P/1S**；B01 全套 **34 passed**。
- 分支 commit 链（终态）：`41f771c` → `a927d20` → `5013423` →
  `ab68ff0` → `ffe1ab4` → `5beb4e1` → `1efd8df`。未 push、未合并
  （push 不在转授范围）。
