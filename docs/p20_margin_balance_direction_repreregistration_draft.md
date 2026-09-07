# P20 草案：MARGIN_BALANCE_CHG 方向重预注册（authored +1 证伪 → −1）

> **[2026-09-07 路由更新，captain 裁决]**：本草案**不再单独立勘误**——方向重注册**并入 F2 批次**（融资拥挤）预注册：authored +1 两轮证伪是合法负结果（promotion_allowed=False 维持+证据保留），"换方向重试"是新假设，须按 F2 批次流程（经济假设重述 + 两轮证伪历史披露 + 新鲜数据裁决）与 MARGIN_BALANCE_TO_CIRC_MV/MARGIN_BUY_PRESSURE_5 同批预注册。本文件保留为证据/历史草案，**不按本文实施**。

- 状态：**APPROVED & EXECUTED — t70 @ `6e11b76` + 索引登记 `57f6f78`（captain 代行批准，双签齐备；p16 `2f5e18b` 先例的 APPROVED 状态与实施完成态并列）**（起草：factor-engineer，AgentTeams tushare-pit-migration，2026-09-06；依据 captain t22 路由指令第 3 条"方向重预注册草案（authored +1 被证伪 → 新方向假设的预注册文本）"）。批准后本文保持前瞻性，禁止以实现输出反向修改。
- 批准记录（p16 `2f5e18b` 先例，回写义务模式）：**批准人 = captain（代行，p18 §4 + 89dd121 授权扩展；批准链核验 = `docs/p20_approval_chain_verification.md` @ `6995ede`，governance-auditor 入册）**。代行批准声明逐字引用（t70 任务描述原文）：「批准 MARGIN_BALANCE_CHG 方向重预注册 −1——拥挤经济学假设（融资余额变化与拥挤同向，与 MARGIN_CROWD_60 语义一致）；authored +1 四轮证伪（含 t22 修复后 v2 T+1 源 −0.0306/−0.366）完整保留为负结果证据；范围 = 方向元数据 + FEATURE_REGISTRY 7→8 + 文档再生，公式与其他元数据零改动；RED + 禁再改纪律照草案。依据：用户授权扩展记录 89dd121 附录 C。」实施 = `6e11b76`（expected_direction −1 + FEATURE_REGISTRY 8→9 + registry doc 再生 + RED 先行）+ 索引登记 `57f6f78`（CONTRACTS.md）+ 本批准行补记（首记 `ff7280d`，本行按 captain 微项指令补全 t70 任务描述逐字声明）。
- 路由沿革：本草案曾按 captain t22 验收 item 1 路由"并入 F2 批次、不按本文实施"（下段路由更新）；其后 captain keep-list 终裁明确 t70 先行执行（REGISTRY 7→8 按计划序，实际因 Part A 在途重叠消费为 8→9），**t70 按双签批准实施**——下段"不按本文实施"的 F2 路由由 t70 执行所取代，F2 批次内的方向表述以本文件 t70 实施后的注册表为准。
- 证据基线：**t18/t22 两轮 v2 重测**（`docs/t18_factor_retest_measurement_log.md`、`docs/t22_semantic_fix_measurement_log.md`；dataset `43ef1e4c…` → `157e0cf5…`）+ t2/v4 冻结审计（`docs/factor_inventory_audit_20260831/`、`docs/factor_inventory_audit_v4_20260901/`，dataset `a839ecf2…`）。勘察基线 commit：codex/factor-semantic-fixes @ `0e4c1fd`。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。任务路由：**研究语义变更**（AGENTS §3.2：预期方向为预注册元数据）。

## 1. 问题陈述

`MARGIN_BALANCE_CHG`（20 日融资余额变化率，external 族，Tier B，v2 `margin_detail_pit` T+1 源）的 authored `expected_direction = +1`（"融资流入追涨抬价"）被**四轮独立测量一致证伪**：

| 轮 | dataset | 源/语义 | IC10 | ICIR10 | 覆盖 | 年度同向占比 |
|---|---|---|---|---|---|---|
| t2（2026-08-31） | a839ecf2（legacy qfq） | legacy margin trade_date 透视 | −0.0335 | −0.414 | 0.883 | （逐年表见 metrics.json） |
| v4（2026-09-01） | a839ecf2（legacy qfq） | 同上 | −0.0335 | −0.414 | 0.883 | |
| t18（2026-09-06） | 43ef1e4c（v2 修复前） | 同上 | −0.0263 | −0.237 | 0.816 | 0.08 |
| t22（2026-09-06） | 157e0cf5（v2 勘误后） | **v2 margin_detail_pit T+1（t22 修复后）** | −0.0306 | −0.366 | 0.875 | 0.00-0.08 |

方向证伪在 legacy→v2 数据底座、t22 修复前后、不同源透视语义下全部成立（h=10 原始 IC 恒为负；t22 残差 OOS −0.0308 同号）。同族的 `MARGIN_CROWD_60`（authored −1，P9 §5.4 预注册"按测量裁决"）四轮全部符号一致（IC10 −0.0225…−0.0322）并通过族门——负方向的融资拥挤经济学在本数据上成立，+1 方向的"流入抬价"假设不成立。

## 2. 新方向假设（预注册）

- **新 `expected_direction = −1`**：融资余额 20 日加速上升 = 杠杆资金拥挤/后期追涨 → 后续收益转负。经济学上与 P9 §5.4 族④（拥挤 → 负）及本库实测最强的 `MARGIN_CROWD_60`（authored −1，四轮符号一致）同向。
- 按 P9 §5.4 惯例"方向按测量裁决、预注册 expected_direction"——本草案即该裁决的预注册落盘；**禁止在批准后按未来测量反向再改**（再改须新预注册）。

## 3. 范围与非目标

- **范围**：仅 `feature_metadata.py` 中 `MARGIN_BALANCE_CHG` 的 `expected_direction`（+1 → −1）；`FEATURE_REGISTRY_VERSION 7 → 8`（元数据变更）+ `docs/feature_registry.md` 同 commit 再生。
- **非目标**：不改公式/数据源/可用时间（t22 已落地且四轮证据一致）；不改 FACTOR_COMPUTE（方向元数据不改变因子值）；不改 GRAMMAR/词表/研究域/tier；不改晋级阈值；不新增特征；不动 `MARGIN_CROWD_60`（其 −1 已四轮验证）。

## 4. 不变量

1. 公式、数据源、PIT 可见性（t22 的 v2 T+1 语义）零改动——方向是解释元数据，不是计算语义。
2. registry 单一来源：方向只存于 `feature_metadata.py`，生成文档随动（AGENTS §11.5）。
3. 旧 artifact 不迁移、不拒绝：artifact 携带自身的 direction 载荷，元数据变更只影响未来的治理/搜索解读（无兼容面破坏）。
4. 负结果纪律：若未来重测显示 −1 方向亦被证伪 → 该特征进入弃用/负结果流程（新预注册），不得再次单方改向。

## 5. 预期 RED 测试（实现前先红）

1. `tests/test_feature_metadata.py`（或新增断言）：`MARGIN_BALANCE_CHG` authored `expected_direction == −1`——当前 +1 → RED。
2. `docs/feature_registry.md` drift guard：再生后"预期方向"列 = −1（同 commit）。
3. 版本钉值：`FEATURE_REGISTRY_VERSION == 8`。

## 6. 测量方案与裁决

- 方向翻转不消耗测量预算、不需要即时重跑（因子值零变化）；下一次全量重测（F1-F4 新增前的例行口径）把 `annual_direction_stability_h10` 的对比基线更新为 −1 方向，逐轮留档。
- 裁决规则：−1 方向经下轮重测若同向占比 ≥ 0.5 且 IC 符号稳定 → 维持；若证伪 → 触发 §4.4 负结果流程。

## 7. 资源上限与停止条件

- 实现量：1 行元数据 + 版本常量 + 文档再生 + 断言更新，单 commit。
- 停止条件：governance/captain 任一否决 → 草案作废留档；批准后 RED 无法稳定失败 → 回到本文修订。

## 8. 批准与实施边界

- 批准流程：governance-auditor 登记（CONTRACTS.md 同 commit）+ captain 代行批准（p18 §4 框架）→ 实施由 factor-engineer 在 codex/factor-semantic-fixes 上执行（单 commit）。
- 本草案不授权 push/PR/merge。
