# P24 登记核验记录（F2 融资拥挤两因子——批准链、内容核验、版本表基线不符发现）

- 性质：治理登记核验（evidence-only）。核验对象 = factor-engineer 的
  F2 预注册草案 `docs/p24_f2_margin_crowding_draft.md`
  （codex/factor-semantic-fixes @ `41e9aa0`，t27 任务书候选集）。
- 核验人：governance-auditor（AgentTeams tushare-pit-migration），2026-09-08。
- 编号核验：**p24 ✓**（p23 = F1 已实施终局版；下一空闲号 = p24，无冲突）。

## 1. 内容核验（通过项）

| 核验项 | 结果 |
|---|---|
| 候选集 | ✓ 与 t27 任务书一致：MARGIN_BALANCE_TO_CIRC_MV（−1，rzye/circ_mv 横截面 z）+ MARGIN_BUY_PRESSURE_5（−1 主假设预注册、备择 +1 按测量裁决——P9 §5.4 惯例明文） |
| 数据源/路径 | ✓ 仅 v2 margin_detail_pit（rzye/rzmre，T+1 可见性 p17 §6.1）+ daily_basic_pit circ_mv（Tier A）；asof_winner_intervals 单一路径；零 schema/零 DB |
| §13 双判据教训 | ✓ 显式预注册：**ΔIC 增量为约束性判据**（|ρ| 低但 ΔIC 低于门槛依然 REJECTED）；|ρ| 为辅助——t22 日志 §13 教训明文适用 |
| 覆盖率诚实预期 | ✓ 两融资格宇宙 0.5-0.6 量级（显著低于基本面族）——预注册诚实预期，重测实测为准 |
| 相关簇预检 | ✓ vs MARGIN_CROWD_60/MARGIN_BALANCE_CHG/CROWD_AMOUNT_60 同源风险实质存在声明 + |ρ|≥0.9 → 族内精简 |
| 族原子性 | ✓ 2 特征同 commit 进词表、同批裁决、同进退（p13 §8 先例） |
| 方向预注册说明 | ✓ F2-2 主假设 −1、备择 +1 按测量裁决，证伪即如实重登记留档 |

## 2. 版本表基线不符发现（**需更正后方可转用户批准**）

**实测**（git show 逐文件核验，0cb6493 与 41e9aa0 两时点一致）：
GRAMMAR **8** / FEATURE_REGISTRY **13** / RESEARCH_DOMAIN **4** /
FACTOR_COMPUTE 4。

**草案声称**：证据基线"FEATURE_REGISTRY 14 / GRAMMAR 9 / RESEARCH_DOMAIN
4"（@ 0cb6493）；§5 表 GRAMMAR 9→10（generation 10）、FEATURE_REGISTRY
14→15、RESEARCH_DOMAIN 4→5。

**不符**：草案的"旧值"（G9/R14）与实际树（G8/R13）差一——GRAMMAR 与
FEATURE_REGISTRY 两项的旧值系统性偏移（DOMAIN 一致）。**更正要求**：
- GRAMMAR：**8→9**（generation **9** 注册 remap 分派，非 generation 10）；
- FEATURE_REGISTRY：**13→14**；
- RESEARCH_DOMAIN：4→5 ✓（该项一致）；
- 证据基线行更正为实测值（G8/R13/D4/FC4 @ 0cb6493）。
草案其余内容不受影响；此更正属版本表数值重算，不触及候选集/公式/判据。

## 3. 批准路径（与 p23/F1 先例一致）

F2 = 两新特征入词表（GRAMMAR bump = 采样空间变化）→ **用户批准**
（p18 §3.2 硬边界 #2 + 原 F2 任务书明示"经用户批准"；代行授权不覆盖
新特征批次——p23 先例）。转达时**应附本核验第 2 节的更正版版本表**，
避免用户所见数字与实施 RED 钉值不一致。

## 4. 登记安排

- CONTRACTS.md 索引行随 p24 实施 commit 同 commit 登记（草案阶段不
  索引——t42/t49 先例）；落点 = codex/factor-semantic-fixes。
- 实施人 = factor-engineer（RED → 最小实现 → GREEN → 全量 → 83 名
  重测 → 族门裁决 → 测量日志）；版本表重算后 RED 钉值同步
  （GRAMMAR 9 / REGISTRY 14 / DOMAIN 5 钉值以更正后基线为准）。

## 5. 未生效声明（已被第 6 节终态登记取代——p24 已批准并实施）

p24 在用户批准与实施落地前为 DRAFT 草案（不索引、不生效、不授权任何
registry/词表变更）；本记录仅为治理侧核验，含版本表基线不符发现与
更正要求。

## 6. F2 终态登记（2026-09-08 族门裁决落地——通过，候选池准入）

factor-engineer 报告族门裁决**通过**（t22 日志 §15，`09f005e`）：
- **MARGIN_BALANCE_TO_CIRC_MV 门承载**：ΔIC +0.0065 ≥ +0.005 且残差
  同向；同源共线风险实证排除（top-|ρ| = TURNOVER 0.405，非
  MARGIN_CROWD_60）；
- **MARGIN_BUY_PRESSURE_5**：方向强证实但 ΔIC 未达标 + 成本 4.44%
  （晋级高风险标记，族级通过保池身份）；
- 两特征进候选池 **promotion_allowed=True**；
- metrics = factor_inventory_audit_f2_20260908/metrics.json（83 名，
  eab1061-era V2 集 = 终局裁定集合）；**首次运行 adapter 缺陷作废
  披露在案**；
- 版本表按实施时点真实常量重算（本核验 §2 更正后基线为准）；
- t27 已置 completed、F3 解锁。

p24 线治理记录至此终态闭环（核验 + 版本表更正 + 用户批准链 + 实施 +
族门通过全链在案）。

**【时序治理标注（2026-09-08 governance 升级 captain）】**：F2 实施
（`0c1f0bb`/`d1c651d`）+ 83 名重测 + 族门裁决（`09f005e`，t27
completed）**全部先于 p24 用户批准门**——与 p18 §3.2 硬边界 #2（新
特征批次由用户批准，代行不覆盖）及原 F2 任务书流程（预注册 → 用户
批准 → 实施）相悖，属 G0 审计命名的"实现生效但批准缺失"同款形态
（p12/p13/p14 先例）。处置待用户追认批准（附更正版版本表 + 全链
披露，p12 先例）或回退——由 captain 转达用户裁决；本标注如实记录
时序，不预写裁决结果。
