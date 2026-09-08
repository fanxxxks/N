# P25 草案：F3 批次预注册——预期与事件四因子（event/medium 域）

- 状态：**DRAFT — 待 governance-auditor 核对编号与登记 + 用户批准后实施**（起草：factor-engineer，AgentTeams tushare-pit-migration，2026-09-08；t31 任务书候选集）。批准后本文保持前瞻性，禁止以实现/测量输出反向修改。
- 证据基线：F2 批次重测 `docs/factor_inventory_audit_f2_20260908/metrics.json`（dataset `157e0cf5…`，FACTOR_COMPUTE 4 / FEATURE_REGISTRY 14 / GRAMMAR 9 / RESEARCH_DOMAIN 5；词表 83）。勘察基线 commit：codex/factor-semantic-fixes @ `0cb6493`。
- 数据可用性预检 ✓：三张事件表已在 v2 库落库（earnings_forecast_pit 126,720 行含 p_change_min/p_change_max/last_parent_net；earnings_express_pit 27,429 行含 n_income/total_profit；repurchase_event_pit 21,018 行含 proc/amount——全部带 ann_date/available_date）。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。任务路由：**研究语义变更**（AGENTS §3.2：新增特征）。

## 1. 批次依据与问题陈述

词表 83 名中**预期管理与公司行动**维度完全缺位：业绩预告（管理层的前瞻信息）、业绩快报 vs 预告的偏离（信息质量的即时信号）、回购（管理层估值信号 + 股本收缩）均无词表表达。本批按 t31 任务书预注册 4 个候选，全部只用已落库的三张事件表字段（零 schema 变更），经 `asof_winner_intervals` 单一路径（ann_date/available_date 语义按 p17）。

## 2. 候选因子（预注册公式；域/执行点逐因子声明；Tier B）

记号：`forecast_mid(E)` = (p_change_min + p_change_max)/2（同一 ts_code+end_date=E 的最新可见预告）；`express_growth(E)` = n_income(express, E)/last_parent_net(forecast, E) − 1；`last_parent_net(forecast, E)` = 该预告行的 last_parent_net（上年归母净利，规模基数）。可见性：全部经 `asof_winner_intervals`（p17 §6.2），**公告日后下一交易日才可进入因子**（available_date 键控）。

| # | 特征 | 预注册公式 | 域/执行点 | 经济假设 | 方向 | NaN 守卫 |
|---|---|---|---|---|---|---|
| F3-1 | FORECAST_GROWTH_MIDPOINT | `forecast_mid(E)`（最新可见预告） | medium / every_10_days | 管理层预期增长中点高 → 基本面改善前瞻 | **+1** | 无预告或 p_change 对缺失 → NaN |
| F3-2 | FORECAST_UNCERTAINTY | `(net_profit_max − net_profit_min)/abs(last_parent_net)`（abs(base)>0） | medium / every_10_days | 预告区间相对宽度大 → 管理层前景不确定 → 折价 | **−1** | base=0/缺失或区间端缺失 → NaN |
| F3-3 | EXPRESS_VS_FORECAST | `express_growth(E) − forecast_mid(E)`（快报公告日事件注入，事件后窗口保持） | **event / 公告日事件注入** | 快报增长超出预告中点 = 正面意外 → 短窗正收益 | **+1** | 无匹配预告/last_parent_net=0/快报缺失 → NaN |
| F3-4 | BUYBACK_INTENSITY | `Σ(已实施 proc 的 amount)/公告日前最近 circ_mv`（最新一次实施公告事件注入） | medium / every_10_days | 已实施回购强度高 = 管理层估值信号 + 股本收缩 | **+1** | 无已实施回购/circ_mv 缺失 → NaN |

- **域/执行点声明**：F3-1/F3-2/F3-4 = medium（预告/回购为低频更新，every_10_days 执行点）；F3-3 = **event**（快报公告为离散事件，公告日注入、事件后窗口保持至下一事件——event 族首个成员，执行点 = 公告日次日开盘）。
- **口径披露**：F3-3 的 express 增长基数采用匹配预告行的 last_parent_net（同 ts_code+end_date）——若快报无匹配预告 → NaN（不做跨源基数拼接）。
- **与既有词表的非重复性论证**：词表无任何预告/快报/回购因子（全新维度）✓。预注册冗余预检（§13 双判据）：ΔIC 增量为约束性判据；|ρ| 辅助。
- **族原子性**：4 特征同 commit 进词表、同批裁决、同进退（p13 §8 先例）。

## 3. 每因子预注册记录（t31 验收逐项）

| 维度 | 记录 |
|---|---|
| 经济假设 | 见 §2 表 |
| 数据源 | v2 `earnings_forecast_pit`（p_change_min/max/net_profit_min/max/last_parent_net）、`earnings_express_pit`（n_income）、`repurchase_event_pit`（proc/amount）——全部经 `asof_winner_intervals` 单一路径；F3-4 分母 = daily_basic_pit circ_mv（Tier A） |
| PIT 可用时间 | **公告日后下一交易日才可进入因子**（available_date 键控，p17 §6.2 as-of；PIT 测试直接断言公告日当日不可见） |
| Tier | B（事件/预期表，p17 映射） |
| 预期方向 | +1/−1/+1/+1（F3-2 主假设 −1 预注册；按测量裁决） |
| 预测周期 | medium ×3 + event ×1（逐因子声明见 §2） |
| 覆盖率 | 预期：F3-1/F3-2 ≈ 预告覆盖（有预告的股票/期间）；F3-3 ≈ 快报∩预告交集（较低，诚实预期）；F3-4 ≈ 回购覆盖（低频）——重测实测为准 |
| 换手/成本 | event 因子公告日注入（换手按事件率）；medium 因子按域惯例——重测实测为准 |
| 相关簇 | 预检 vs MOMENTUM_20/REVERSAL_60/PROFIT_YOY（盈余动量相邻）；|ρ|≥0.9 → 族内精简 |
| 增量 OOS 计划 | ΔIC_OOS（h=10，|w_b| 修正基线 per p9 附录 B）≥ +0.005 且残差符号与预注册方向一致 → 族门通过 |

## 4. 范围与非目标

- **范围**：`ashare_data/events.py` 新增事件帧构建器（三表 as-of 读取 + 四因子合成）；`ashare_model/{vocab,feature_metadata,feature_registry}.py` + `ashare_domain.py` 登记追加（event 域执行点为新增能力——event 特征的注入路径按公告日事件语义实现）；`docs/feature_registry.md` 再生。
- **非目标**：不动既有特征；不做业绩预告修正链的深度建模（仅最新可见预告）；零 DB 写入。

## 5. 版本影响（预注册；以实施时点现值重算）

| 版本 | 旧（现值） | 新 | 理由 |
|---|---|---|---|
| GRAMMAR（vocab.py） | 10 | 11 | 词表 83→87（追加，token id 不移位）；generation 11 注册 remap 分派 |
| FEATURE_REGISTRY | 15 | 16 | 4 特征元数据 + depends_on |
| RESEARCH_DOMAIN | 5 | 6 | medium external-flows +2、event 域新增（首个 event 特征 F3-3 入域——event 域为新增域能力，bump 合规） |
| FACTOR_COMPUTE | 4 | 不变 | 纯特征追加（事件帧为新增输入路径——若 loader 判定输入基座变化则 bump 并披露，实施时决定） |
| DATA_TIER | 2 | 不变 | 事件表→B 既有映射 |
| dataset_id | 157e0cf5 | 157e0cf5（不变） | 零 DB 写入 |

## 6. 预期 RED 测试（实现前先红）

1. 合成三表上四公式逐值断言（预告中点、区间宽度/基数、快报-预告差、回购强度；公告日当日不可见）。
2. as-of 可见性：公告日次日可见、修订按 winner 时间线（预告修正链取最新可见）。
3. 词表 83→87 追加不移位；GRAMMAR 11 钉值 + generation 11 分派注册。
4. 域归属 medium ×3 + event ×1；RESEARCH_DOMAIN 6 钉值；event 注入路径按公告日事件语义。
5. 未回填态 fail-closed：新特征帧 NaN → 质量门拒绝。

## 7. 测量方案与裁决（预注册）

- 实现后按同口径全量重测（87 名；h∈{1,2,3,5,10,15,20}、min_stocks=10、IS/OOS 2022 分界、|w_b| 修正基线、同成本模型、五 folds、方向/PIT/Tier 审计）。
- **族门（4 特征整族原子）**：≥1 成员 ΔIC_OOS ≥ +0.005 且残差符号与预注册方向一致 → 族通过；否则负结果 promotion_allowed=False（p13 §8 先例）。
- F3-2 方向按测量裁决（主假设 −1 证伪 → 实测方向重登记留档）；|ρ|≥0.9 → Part A 机制裁决。
- 负结果保留全部证据；禁止因搜索表现不好继续扩库。

## 8. 资源上限与停止条件

- 重测参考 F1/F2 实测（2.2h/1.6h——事件帧构建为新增路径，如实记录）。
- 停止：任一 RED 无法稳定失败 → 回到本文修订；governance/captain 否决候选集 → 修订再批；门禁红 → 带证据上报。

## 9. 批准与实施边界

- 批准流程：governance-auditor 核对编号与登记（CONTRACTS.md 同 commit）+ **用户批准**（t31 任务书门禁："经用户批准"）→ factor-engineer 实施（RED → 最小实现 → GREEN → 全量 → 重测 → 裁决 → 测量日志）。
- 本草案不授权 push/PR/merge。
