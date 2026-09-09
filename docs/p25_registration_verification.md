# P25 登记核验记录（F3 预期与事件四因子——批准链、内容核验、event 新域、p24 索引缺口发现）

- 性质：治理登记核验（evidence-only）。核验对象 = factor-engineer 的
  F3 预注册草案 `docs/p25_f3_expectation_event_draft.md`
  （codex/factor-semantic-fixes @ `e1cb087`，t31 任务书四候选集）。
- 核验人：governance-auditor（AgentTeams tushare-pit-migration），2026-09-08。
- 编号核验：**p25 ✓**（p24 = F2 已实施终局版；下一空闲号 = p25，无冲突）。

## 1. 内容核验（通过项）

| 核验项 | 结果 |
|---|---|
| 候选集 | ✓ 与 t31 任务书逐字一致：FORECAST_GROWTH_MIDPOINT（+1，预告中点）/ FORECAST_UNCERTAINTY（−1，区间宽度/abs(last_parent_net) 规模化 + abs>0 守卫）/ EXPRESS_VS_FORECAST（+1，快报增长−最新已知预告中点）/ BUYBACK_INTENSITY（+1，已实施回购金额/公告日前最近 circ_mv） |
| event 新域声明 | ✓ EXPRESS_VS_FORECAST 为 **event 域首个成员**（公告日事件注入）——RESEARCH_DOMAIN 5→6（event 域为新增域能力），语义变化幅度如实预注册 |
| 数据可用性预检 | ✓ 三张事件表已在 v2 库（forecast 126,720 / express 27,429 / repurchase 21,018 行，全部带 ann_date/available_date——factor-engineer 报告值）；asof_winner_intervals 单一路径；**公告日后下一交易日可见（PIT 测试直接断言公告日当日不可见——t31 验收明示）** |
| 零 schema/零 DB | ✓ 三张事件表已在 v2 库，本批零 schema 变更、零 DB 写入 |
| 版本表 | ✓ 以真实常量正确重算：GRAMMAR 10→11、FEATURE_REGISTRY 15→16、RESEARCH_DOMAIN 5→6（该分支实测 G9/R14/D5——F2 实施已按 p24 核验 §2 更正后基线落地，本草案重算与实测吻合）|
| RED/族门/资源上限 | ✓ RED 清单、族门（|w_b| 修正基线 per p9 附录 B）、资源上限齐备 |

## 2. 版本表基线核验（通过——F2 教训已采纳）

p24 核验 §2 发现的"版本表基线差一"问题在本草案**未被重复**：GRAMMAR
10→11 / FEATURE_REGISTRY 15→16 / RESEARCH_DOMAIN 5→6 与该分支实测
（G9/R14/D5 @ e1cb087）逐项吻合——实施时点真实常量重算的 standing
约束已被执行方采纳。

## 3. **p24 索引行缺口发现（需 factor-engineer 补登记）**

**实测**：`git log -S'p24_f2_margin_crowding' -- docs/CONTRACTS.md`
全史检索**零命中**——p24 的 CONTRACTS.md 索引行**从未登记**（F2 实施
commit 0c1f0bb/d1c651d 未携带索引行，违反 same-commit 登记规则
——p20/p23 先例：实施 commit 携带索引行）。

**更正要求**：factor-engineer 在 codex/factor-semantic-fixes 补一笔
docs commit——CONTRACTS.md 追加 p24 行（F2 两因子、族门通过、候选池
准入）+ 缺口披露说明；或随 p23/p24 线集成时由 integrator 一并补登
（captain 定）。p23 行 ✓（eab1061 已登记）、p22 行 ✓（315bce4）不受
影响。

## 4. sequencing 约束（F2 教训——不重演）

**F3 实施仅在用户批准门之后启动**：F2 线的实施/重测/族门裁决先于
用户批准门的顺序倒置已升级 captain（p24 核验记录时序治理标注）——
F3 线不得重演：p25 用户批准落地前，禁止实施 commit（RED 起草可先
行）。批准链：用户批准（p18 §3.2 硬边界 #2 + t31 任务书流程；代行
不覆盖——p23 先例），captain 转达。

## 5. 登记安排

- CONTRACTS.md 索引行随 p25 实施 commit 同 commit 登记（p23 先例）；
  落点 = codex/factor-semantic-fixes。
- 实施人 = factor-engineer（含 events.py 新增事件帧构建器与 event
  注入路径——t31 范围）。

## 6. 未生效声明

p25 在用户批准与实施落地前为 DRAFT 草案（不索引、不生效、不授权任何
registry/词表/域变更）；本记录仅为治理侧核验，含 p24 索引缺口发现、
sequencing 约束与登记安排。
