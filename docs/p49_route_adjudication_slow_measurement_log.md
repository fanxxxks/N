# P49 研究路线裁决与 slow 域基线轮 — 测量日志（工作包 13）

- 契约：`docs/p49_route_adjudication_slow_draft.md`（APPROVED，用户
  2026-09-20 直接指令 + 冻结决策表；勘误 1 于运行间记入）。
- 被测实现：契约 `b393053` → RED `7a107f9` → GREEN `342ce52` →
  本日志。数据侧操作与运行记录见 §2。
- 产物：`data/r1/r01_slow_baseline/r01_baseline_report.json`
  （schema v3，completed=true，五折全臂全成本档）。

## 1 路线裁决（决策表逐行，证据引用）

| 行 | 裁决 | 依据 |
|---|---|---|
| 1 小预算搜索对照 | 未触发 | P47：动态劣于首建仓持有（4 臂中 3）；P45：评分 vs 引擎 unreliable |
| 2 等权参照 | 搭载生效 | p42 ICIR 负结果维持；无新组合假设 |
| 3 slow 域基线 | **选定并执行** | P47 裁决 1/3 双触发（medium 无可靠净空间） |
| 4 配置/持有问题 | 未触发 | 无优势可归因 |
| 5 暂停算法投入 | **本轮后生效** | slow 轮 worth_search=false（§3）→ 双域负结果在册 |

## 2 执行史（四次运行，两处诚实拒绝 + 两项解封操作）

1. **run1 拒绝**（exit 2，零 OOS 行）：p34 §5.2 check-6 的
   leave-family-out 消融对单族集（slow 五信号全属 fundamental 族）
   结构不可能。**勘误 1**（前瞻性，未见任何经济结果）：单族集回退
   族内 leave-one-signal-out，`R01_REPORT_SCHEMA_VERSION` 2→3
   （RED `7a107f9` → GREEN `342ce52`，多族行为逐位不变）。
2. **run2/3 拒绝**：D7 dataset_manifest 门——v2 库**自身**清单过期
   （`source_ingest_runs` ALL、`daily_price_limit` 2014–2026 分区哈希
   漂移；同步发生在上次 v2 发布之后）。复合发布（generation 7，仅写
   主库账本）不解此门——资格检查读 v2 自身清单。
3. **解封**：按 §5.6 纪律（审计=拒绝消息精确定位漂移表；无并发正式
   运行；发布为追加式身份行、零数据行变更）以摄取管线同一权威
   `publish_snapshot(TUSHARE_V2_CONFIG)` 发布 v2 清单：**新 v2
   dataset_id b1237612eae21c02**（57,349,882 行，generation 2）。
   过程障碍：`ashare_model`（torch）+ DuckDB 同进程段错误（隔离复现
   定位；发布脚本留在 ashare_data 层绕开；**登记为环境已知问题**）。
4. **run4 成功**（exit 0；因子缓存命中无需重建）。

## 3 slow 轮结果（五折 2021–2025，every_20_days/20，n20 配置）

**裁决：worth_search = FALSE；qualified_arms = []。**

| 臂（1x） | 年均净超额 | 2x | 正折 | active IR | 投入 | 持仓 |
|---|---|---|---|---|---|---|
| ROE | **−8.63%** | −8.68% | 0/5 | −0.98 | 0.55 | 13.4 |
| PE_TTM | −3.80% | −3.85% | 1/5 | −0.41 | 0.64 | 15.5 |
| REVENUE_YOY | −5.67% | −5.71% | 1/5 | −0.69 | 0.48 | 12.3 |
| DIVIDEND_YIELD | −3.25% | −3.30% | 2/5 | −0.52 | 0.69 | 16.2 |
| MARKET_CAP | −5.72% | −5.76% | 1/5 | −0.82 | 0.53 | 13.3 |
| combo:all3 / momentum_reversal | — | — | — | unreliable/低活动/薄书 标记 | | |
| control:cash | −4.61% | — | 2/5 | — | 0 | 0 |

拼接参照：基准等权五年 +31.3%；ROE 书 −17.8%（maxDD 47.2%）。
费用 1x→2x 差 <0.06pp/年——成本不是主因；投入比例 0.48–0.69（慢域
预算纪律下同样受限）。

## 4 最终裁决（§4 映射执行）

- **决策表第 5 行生效：暂停扩大算法投入（GP/TPE/RL 预算冻结维持），
  保留研究平台与全部负结果。** medium（R01/R02/R03/P47）与 slow
  （本轮）双域简单基线均无可折现净空间；p42 ICIR 加权负结果在册；
  WP12 统计冻结在先（AR(1) 邻域与短窗不可作 5% 声明）。
- 任何重启需新预注册假设 + WP12 方法修订重校准 + p44 执行病理修订
  三者至少其一先行（登记，不在本包实施）。

## 5 归档与验证

- 运行记录：`reports/p49/r01_slow_run{,2,3,4}.log`、
  `publish_v2_manifest*.py`（含段错误绕开）、`probe_coverage.py`、
  RED 证据 `reports/p49_red_evidence.txt`。
- 工程验证：p49 聚焦 5P；r01/r02 回归 37P（两 pin 规则 2 更新后 27P
  复验）；compileall/分片 123/123/diff-check 干净。全量（精确候选树
  `342ce52`，20:06）：**2148 passed / 1 failed / 1 skipped**——总数对账
  2145（p48 树）+ 5（p49）= 2150 ✓；唯一失败为 b03 R4（自 p43 归名的
  进程树满载时序波动家族；单测复跑 121s 通过，非回归）。
- 零数据行变更（两处发布均为追加式身份代际）；主库 generation 7、
  v2 generation 2 为当前身份。

## 6 结论边界

- 本轮覆盖 r1_slow_ab 冻结帧（Tier AB、五因子、n20、0x/1x/2x、
  2021–2025）；不外推其他慢因子/参数；组合臂的 unreliable 标记来自
  其自身执行行为（低活动/薄书），非数据失败。
- 无显著性/晋级声明（WP12 冻结）。
