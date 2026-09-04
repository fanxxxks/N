# D3 回填测量日志（t9，p17 §10 八步 + §16 验证矩阵）

- 被测 commit：codex/tushare-schema-v1 @ `2858e6f`→`904862d`（worktree D:/minequant/AlphaGPT-d2）
- 环境：Windows / Python 3.13.12 / duckdb 1.5.5 / tushare 1.4.29 / pandas 3.0.5
- v2 库：`D:/minequant/AlphaGPT-d2/data/ashare_v2.duckdb`（生产库零写入）
- L0 快照：`D:/minequant/AlphaGPT-d2/data/raw/tushare/<endpoint>/<ingest_id>/`
- TUSHARE_TOKEN：git-ignored config/.env，全程零泄漏

## 回填结果（19 表 / 57,109,667 行）

| 表 | 行数 | 日期范围 | 备注 |
|---|---|---|---|
| daily_bar_raw | 11,886,384 | 20140102..20260904 | 5,461 distinct codes |
| daily_adj_factor | 12,246,154 | 20140102..20260904 | |
| daily_basic_pit | 11,896,917 | 20140102..20260904 | |
| daily_price_limit | 12,209,690 | 20140102..20260904 | pre_close via typed 显式请求 |
| daily_suspension | 336,077 | 20140102..20260904 | suspend_timing additive |
| industry_taxonomy | 511 | — | SW2021+SW 双代际 |
| industry_membership_pit | 0 | — | **§15.6 停止：index_code 源列缺失** |
| income_statement_pit | 409,275 | 19901231..20260630 | 全历史含修订版本 |
| balance_sheet_pit | 395,583 | 19901231..20260630 | |
| cashflow_statement_pit | 382,589 | 20011231..20260630 | |
| financial_indicator_audit | 434,471 | 20121231..20260630 | |
| financial_disclosure_schedule | 212,360 | 20121231..20260630 | |
| earnings_forecast_pit | 126,720 | 19981231..20271231 | |
| earnings_express_pit | 27,429 | 20041231..20260630 | |
| repurchase_event_pit | 21,018 | 20121231..20260909 | |
| holder_number_pit | 244,658 | 20101231..20221230 | **部分完成：end_date NULL §15.6 停止** |
| margin_detail_pit | 5,877,462 | 20140102..20260903 | |
| index_weight_pit | 348,890 | 20050131..20260831 | 3 指数年份窗口分页 |
| source_ingest_runs | 53,479 | — | 台账（53,475 succeeded / 4 failed） |
| **TOTAL** | **57,109,667** | | |

## trade_cal parity

重叠窗 19901219..20261231：tushare 8,797 = prod 8,797，双向差集为空。生产日历保持唯一权威。

## 分页拆分证据（行数==上限触发）

- stk_holdernumber：年窗大量触发 3000 行上限 → 自动二分（~130 次"paginate"记录）
- fina_indicator：100 行上限，~50% 股票触发 → 自动二分（~5,500 次"paginate"记录）
- index_weight：000300/000905/000852 全历史 >3000 行 → 年份窗口二分（~130 次）
- daily_bar_raw / daily_adj_factor / daily_basic_pit / stk_limit / margin_detail_pit / suspend_d：按日单次调用 < 各自上限，未触发

## 限速证据

客户端滑动窗口 180 req/min（硬上限），实测速率 85-178/min（随帧大小变化），全程未超 180。

## G1-G8 + research_doctor 状态

**未运行**：G1-G8 gate runner 与 research_doctor 期望生产库 schema 表（constituents/daily_bar 等），v2 库使用 p17 §5 新表名（industry_membership_pit/daily_bar_raw 等），表名/schema 不同导致 gate runner 直接报 Catalog Error。这是 p17 契约的设计缺口（§10 步骤 9 假定 gates 可直接跑 v2，但未预见表名映射需求）。需 v2 gate adapter（待后续任务）。数据完整性已通过独立验证脚本确认。

## §15.6 停止条件触发记录（2 项）

1. **industry_membership_pit index_code 源列缺失**：index_member_all API 返回 l1/l2/l3_code 而非单一 index_code。建议映射：每行按三级展开。已报 captain 转 governance 勘误。
2. **holder_number_pit end_date NOT NULL 违反**：部分行 end_date 为 NULL（"数据不定期公布"）。建议放宽为 nullable。已报 captain 转 governance 勘误。

两项均为 fail-closed 正确拦截（零残留，L0 快照 + failed 台账留证），不影响其他 17 表的完整性。

## 运行时事件

1. 断网导致回填两次暂停（~140min + ~30min），恢复后自动继续（timeout 修复前一次、修复后零影响）
2. 断网引发首次回填 fail-closed 崩溃（retry 耗尽）→ 断点续跑设计验证通过
3. Windows 注册表代理（127.0.0.1:7890）失去上游导致 API 调用挂死 → NO_PROXY=* 直连修复 + 客户端超时加固（commit f173a6d）
