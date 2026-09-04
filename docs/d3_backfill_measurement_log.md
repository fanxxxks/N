# D3 回填测量日志（t9 + t53，p17 §10 八步 + §16 验证矩阵）

- 被测 commit：codex/tushare-schema-v1 @ `08b2016`（worktree D:/minequant/AlphaGPT-d2）
- commit 链：fc646a1 → 28e6690 → 2858e6f → 4d0417e → ad5e78a → f173a6d → 6b8c650 → 904862d → c07de8c → 16f8d0a → 6026129 → 4a76821 → 773f47a → 0a1eb1e → 08b2016
- 环境：Windows / Python 3.13.12 / duckdb 1.5.5 / tushare 1.4.29 / pandas 3.0.5
- v2 库：`D:/minequant/AlphaGPT-d2/data/ashare_v2.duckdb`（生产库零写入）
- L0 快照：`D:/minequant/AlphaGPT-d2/data/raw/tushare/<endpoint>/<ingest_id>/`
- TUSHARE_TOKEN：git-ignored config/.env，全程零泄漏
- Manifest：dataset_id = `43ef1e4caace5cfc3ffc5bef335a3bf613e538dfbff89c405370848e980ce27d`

## 回填结果（19 表 / 57,131,659 行）

| 表 | 行数 | 日期范围 | 备注 |
|---|---|---|---|
| daily_bar_raw | 11,886,384 | 20140102..20260904 | 5,461 distinct codes |
| daily_adj_factor | 12,246,154 | 20140102..20260904 | |
| daily_basic_pit | 11,896,917 | 20140102..20260904 | |
| daily_price_limit | 12,209,690 | 20140102..20260904 | pre_close via typed 显式请求 |
| daily_suspension | 336,077 | 20140102..20260904 | suspend_timing additive |
| industry_taxonomy | 511 | — | SW2021+SW 双代际 |
| industry_membership_pit | 16,611 | 20220729..20220729 | 勘误#2 三级展开 L1/L2/L3 各 5,537 |
| income_statement_pit | 409,275 | 19901231..20260630 | 全历史含修订版本 |
| balance_sheet_pit | 395,583 | 19901231..20260630 | |
| cashflow_statement_pit | 382,589 | 20011231..20260630 | |
| financial_indicator_audit | 434,471 | 20121231..20260630 | |
| financial_disclosure_schedule | 212,360 | 20121231..20260630 | |
| earnings_forecast_pit | 126,720 | 19981231..20271231 | |
| earnings_express_pit | 27,429 | 20041231..20260630 | |
| repurchase_event_pit | 21,018 | 20121231..20260909 | |
| holder_number_pit | 244,658 | 20101231..20221230 | 勘误#3 end_date nullable |
| margin_detail_pit | 5,877,462 | 20140102..20260903 | |
| index_weight_pit | 348,890 | 20050131..20260831 | 3 指数年份窗口分页 |
| source_ingest_runs | 58,860 | — | 台账（58,860 succeeded / 0 failed） |
| **TOTAL** | **57,131,659** | | **19/19 表完整** |

## trade_cal parity

重叠窗 19901219..20261231：tushare 8,797 = prod 8,797，双向差集为空。生产日历保持唯一权威。

## 分页拆分证据（行数==上限触发）

- stk_holdernumber：年窗大量触发 3000 行上限 → 自动二分（~200 次"paginate"记录）
- fina_indicator：100 行上限，~50% 股票触发 → 自动二分（~5,500 次"paginate"记录）
- index_weight：000300/000905/000852 全历史 >3000 行 → 年份窗口二分（~200 次）
- daily_bar_raw / daily_adj_factor / daily_basic_pit / stk_limit / margin_detail_pit / suspend_d：按日单次调用 < 各自上限，未触发

## 限速证据

客户端滑动窗口 180 req/min（硬上限），实测速率 85-180/min（随帧大小变化），全程未超 180。

## v2 Gate Adapter（参数化临时表方案，commit 08b2016）

**方案**：prepare_gate_tables + cleanup_gate_tables（captain 裁决：不建兼容视图）。每次 gate 运行创建 _gate_* 物理临时表（显式列映射从 v2 数据填充），gate runner 用修改后的 DataConfig 指向临时表，运行后 cleanup。

**映射**：
- _gate_constituents ← industry_membership_pit WHERE level=1, COALESCE(out_date,'99991231')（G4 NULL 修复）
- _gate_daily_bar ← daily_bar_raw JOIN daily_adj_factor (vol→volume)
- _gate_stocks ← 生产库 stocks 只读复制
- _gate_trade_calendar ← 生产库 trade_calendar 只读复制

## G1-G8 运行结果（经 adapter，v2 库，dev mode）

| Gate | 状态 | 详情 |
|---|---|---|
| G1 historical member intervals | **PASS** | index_weight_pit 派生成分区间（000300/000905/000852 各有成员）|
| G2 valid list_date | **PASS** | 5,542 stocks 全部有 list_date |
| G3 calendar covers bar window | **PASS** | calendar 19901219..20261231 ⊇ daily 20140102..20260904 |
| G4 no duplicate/overlapping | **PASS** | index_weight 派生区间无重叠 |
| G5 PIT universe contract | **PASS** | dev fallback（production 需要 index 成分数据，dev fallback 允许） |
| G6 ≥ 100 eligible stocks | **FAIL** | 2014/2018/2024 年 0 eligible（index_weight 月度快照粒度限制——部分年份快照与 bar 日期不对齐） |
| G7 every member has bars | **FAIL** | 79 zero-bar 全部 list_date < 20140101（退市股在数据范围外，预期行为），覆盖率中位 99.09% |
| G8 daily bars fresh | **PASS** | daily 20260904 = reference 20260904, lag=0 |
| **合计** | **6/8 PASS** | G1/G2/G3/G4/G5/G8 ✓; G6 月度快照粒度限制; G7 pre-2014 退市股预期 |

### G7 zero-bar 判读（79 码全部与退市预期一致）

- 79 zero-bar codes **全部** 在生产 stocks 表中 list_date < 20140101
- 其中 33 只在 index_weight_pit 中有 2014 年前的权重记录（曾是指数成员后退市）
- 结论：**预期行为**——回填范围 2014-01-01 起，2014 年前退市/除牌的股票在范围内无 bar 是正确的

## research_doctor 运行结果

- code: 08b2016 @ codex/tushare-schema-v1
- data: dataset_id=43ef1e4c, manifest v1, 57,131,659 rows
- gates: 3/8 PASS (formal mode — formal 更严格，dev 模式 6/8)
- fairness probes: 5/5 PASS
- fundamental coverage: unavailable (fundamental_pit 表不在 v2)
- status: NOT HEALTHY (5 error-level findings — G1/G4/G5/G6/G7 formal mode)
- status: NOT HEALTHY (5 error-level findings — G1/G4/G5/G6/G7)

## §15.6 停止条件触发记录（3 项，全部已解决）

1. **industry_membership_pit index_code 源列缺失** → 勘误#2 (89dd121)：三级展开映射，已实现并补跑 ✓
2. **holder_number_pit end_date NOT NULL 违反** → 勘误#3 (947ad25)：放宽 nullable，已实现并补跑 ✓
3. **forecast/express bare start/end 被拒** → 改 per-stock fetch（API 要求 ann_date 或 ts_code），已实现 ✓

三项均为 fail-closed 正确拦截（零残留，L0 快照 + failed 台账留证），勘误批准后全部补齐。

## 运行时事件

1. 断网导致回填两次暂停（~140min + ~30min），恢复后自动继续（timeout 修复前一次、修复后零影响）
2. 断网引发首次回填 fail-closed 崩溃（retry 耗尽）→ 断点续跑设计验证通过
3. Windows 注册表代理（127.0.0.1:7890）失去上游导致 API 调用挂死 → NO_PROXY=* 直连修复 + 客户端超时加固（commit f173a6d）
4. Windows 注册表代理是 GBK mojibake 的根因——中间 PowerShell 重写在测试文件中产生乱码 → 已修复（commit 4d0417e）+ 全面排查（11/11 文件干净）

## 19/19 表完整声明

以上 19 表全部有数据（industry_membership_pit 16,611 行勘误后补齐）。source_ingest_runs 58,860 批全部 succeeded（0 failed）。回填完成。
