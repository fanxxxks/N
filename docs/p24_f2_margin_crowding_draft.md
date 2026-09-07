# P24 草案：F2 批次预注册——融资拥挤两因子（medium 域）

- 状态：**APPROVED — 用户原文「批准」+ captain 代行签署（89dd121 授权扩展）**——本版 = t27 任务书两候选。批准后本文保持前瞻性，禁止以实现/测量输出反向修改。
- **实施期澄清（circ_mv 可见性，2026-09-07）**：生产 `daily_basic_pit` 无 `available_date` 列（trade_date 键控）——circ_mv 的可见性 = **trade_date 同会话约定**（会话 t 的 daily_basic 值于 t 收盘时点已知，与 loader 既有 turnover_rate 路径同一约定，D4 评审通过的 PIT 语义），**非**本文初稿措辞的"同 T+1"。该澄清不改变公式结构（分母仍为 t 会话 circ_mv），仅明确其可见性来源；margin 明细维持 T+1（t−1 数据 t 日可用）——两源可见性节奏不同，组合在 t 会话收盘时点无未来泄漏。
- 证据基线：F1 批次重测 `docs/factor_inventory_audit_f1_20260907/metrics.json`（dataset `157e0cf5…`，FACTOR_COMPUTE 4 / FEATURE_REGISTRY 14 / GRAMMAR 9 / RESEARCH_DOMAIN 4；词表 81）。勘察基线 commit：codex/factor-semantic-fixes @ `0cb6493`。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。任务路由：**研究语义变更**（AGENTS §3.2：新增特征）。

## 1. 批次依据与问题陈述

词表 81 名中两融拥挤维度现有 MARGIN_BALANCE_CHG（20 日融资余额变化率）与 MARGIN_CROWD_60（融资余额对其自身基线的偏离）——两者度量**时间序列维度**（变化率/基线偏离），缺少**横截面强度维度**：融资余额相对流通市值的水平强度（杠杆持仓对自由流通盘的占比）与融资买入流量的短窗压力。本批按 t27 任务书预注册 2 个候选，全部只用已落库的 v2 `margin_detail_pit`（rzye/rzmre）+ `daily_basic_pit`（circ_mv）字段（零 schema 变更），复用 t22 已修复的单一路径（`asof_winner_intervals` + margin T+1 可见性）与 D4 已切源的 basic 帧（circ_mv，Tier A）。

## 2. 候选因子（预注册公式；全部 medium 域；Tier B）

记号：`rzye_t` = t 会话融资余额（margin_detail_pit，T+1 可见：会话 t−1 的明细于 t 日约 8:30 更新后可用，p17 §6.1）；`circ_mv_t` = t 会话流通市值（daily_basic_pit，同 T+1 可见性）；`xcs_z(·)_t` = t 会话横截面 z 标准化（有效截面内）；`Σ5(rzmre)` = t−4..t 五会话融资买入额之和。

| # | 特征 | 预注册公式 | 经济假设 | 方向 | NaN 守卫 |
|---|---|---|---|---|---|
| F2-1 | MARGIN_BALANCE_TO_CIRC_MV | `xcs_z(rzye_t / circ_mv_t)` | 融资余额占流通市值比高 = 杠杆持仓对自由流通盘强度高 → 拥挤 → 去杠杆时跑输 | **−1** | rzye 或 circ_mv 缺失/≤0 → NaN |
| F2-2 | MARGIN_BUY_PRESSURE_5 | `xcs_z(Σ5(rzmre)_t / circ_mv_t)` | 5 日融资买入额占流通市值比高 = 杠杆追涨强度高 → 后续回吐 | **预注册 −1，按测量裁决**（备择 +1：追涨延续） | 任一会话 rzmre 缺失或 circ_mv ≤0 → NaN |

- **方向预注册说明（F2-2）**：主假设 −1（杠杆追涨 → 回吐）；备择 +1（追涨动量延续）同为合法经济假设——按测量裁决（P9 §5.4 惯例），证伪即如实按实测方向重登记并留档。
- **横截面标准化口径**：xcs_z = 每会话有效截面（margin 资格股 ∩ circ_mv 有效）内 (x − mean)/std；零第二套标准化实现（复用既有横截面 z 路径）。
- **与既有词表的非重复性论证 + 冗余预检（双判据，t22 日志 §13 教训适用）**：MARGIN_CROWD_60 度量余额对自身 60 日基线的偏离（时间序列维度），F2-1 度量余额对流通市值的水平强度（横截面维度）——概念相邻但度量维度不同；MARGIN_BALANCE_CHG 为 20 日变化率。**预注册风险声明**：F2-1 与 MARGIN_CROWD_60/MARGIN_BALANCE_CHG 同源（rzye），|ρ| 共线风险实质存在；**约束性判据 = ΔIC 增量**（相对 7 信号复合基线的增量 ≥ +0.005）——若 |ρ| 低但 ΔIC 低于门槛，依然 REJECTED（§13 教训）；|ρ| 通过不治愈 ΔIC 失败。实施后重测若与 MARGIN_CROWD_60/MARGIN_BALANCE_CHG |ρ|≥0.9 → 按族内精简裁决。
- **族原子性**：2 特征同 commit 进词表、同批裁决、同进退（p13 §8 先例）。

## 3. 每因子预注册记录（t27 验收逐项）

| 维度 | 记录 |
|---|---|
| 经济假设 | 见 §2 表（杠杆拥挤 − / 追涨压力 − 主假设） |
| 数据源 | v2 `margin_detail_pit`（rzye/rzmre，T+1 可见性）+ `daily_basic_pit`（circ_mv，Tier A）——全部经 `asof_winner_intervals` 单一路径；两融数据消费 margin_detail_pit 完整字段中本批所需列（rzye/rzmre），单一路径纪律不变 |
| PIT 可用时间 | **T+1 语义**：会话 t−1 的 margin 明细于 t 日约 8:30 更新后方可用于 t（p17 §6.1）；available_date 对齐；circ_mv 同 T+1 口径 |
| Tier | B（pit_capital，margin 线既有映射） |
| 预期方向 | −1 / −1（F2-2 主假设预注册，备择 +1 按测量裁决） |
| 预测周期 | medium（every_10_days，medium_cross_section 执行点） |
| 覆盖率 | 预期显著低于基本面族（两融资格宇宙 ≈ 3,000/5,400，0.5-0.6 量级）——预注册诚实预期，重测实测为准 |
| 换手/成本 | medium 域中换手（MARGIN_CROWD_60 先例）；成本后空间按 ΔIC 与成本拖累联合评估 |
| 相关簇 | 预检 vs MARGIN_CROWD_60/MARGIN_BALANCE_CHG/CROWD_AMOUNT_60（同源风险实质存在）；|ρ|≥0.9 → 族内精简；**ΔIC 增量为约束性判据** |
| 增量 OOS 计划 | ΔIC_OOS（h=10，|w_b| 修正基线 per p9 附录 B）≥ +0.005 且残差符号与预注册方向一致 → 族门通过 |

## 4. 范围与非目标

- **范围**：`ashare_data/capital_flow.py` 新增 2 帧（margin T+1 帧构建器扩展 + circ_mv 分母接入）；`ashare_model/{vocab,feature_metadata,feature_registry}.py` + `ashare_domain.py` 登记追加；`docs/feature_registry.md` 再生。
- **非目标**：不改既有特征公式/源；不动 MARGIN_BALANCE_CHG/MARGIN_CROWD_60（t70 线与 P9 线既有成员）；不做 paper/lifecycle 转换；零 DB 写入（字段已落库，dataset 不变）。

## 5. 版本影响（预注册；以实施时点现值重算）

| 版本 | 旧（现值） | 新 | 理由 |
|---|---|---|---|
| GRAMMAR（vocab.py） | 9 | 10 | 词表 81→83（追加，token id 不移位）；generation 10 注册 remap 分派 |
| FEATURE_REGISTRY | 14 | 15 | 2 特征元数据 + depends_on |
| RESEARCH_DOMAIN | 4 | 5 | medium external-flows 18→20 |
| FACTOR_COMPUTE | 4 | 不变 | 纯特征追加，既有行计算语义零改动（p13 先例） |
| DATA_TIER | 2 | 不变 | pit_capital→B 既有映射 |
| dataset_id | 157e0cf5 | 157e0cf5（不变） | 零 DB 写入 |

- 旧 artifact：GRAMMAR 10 注册进 remap 分派（布局 = 9 布局追加，by-name 解码不变）；deprecated 集（13 名）维持退出采样。

## 6. 预期 RED 测试（实现前先红）

1. 合成 margin/basic 表上两公式逐值断言（T+1 可见性、xcs_z 截面标准化、Σ5 窗口、分母守卫）。
2. as-of 可见性：t−1 明细 t 日可见、公告日当日不可见、修订按 winner 时间线。
3. 词表 81→83 追加不移位（既有 token 往返 parity）；GRAMMAR 10 钉值 + generation 10 分派注册。
4. 域归属 medium ×2；RESEARCH_DOMAIN 5 钉值。
5. 未回填态 fail-closed：新特征帧 NaN → 质量门拒绝。

## 7. 测量方案与裁决（预注册）

- 实现后按 t18/t22/F1 同口径全量重测（83 名；h∈{1,2,3,5,10,15,20} 非重叠 IC、min_stocks=10、IS/OOS 2022 分界、|w_b| 修正基线 per p9 附录 B、同成本模型、五 folds、方向/PIT/Tier 审计）。
- **族门（2 特征整族原子）**：7 信号等权复合基线（|w_b| 修正语义），族内 ≥1 成员 ΔIC_OOS ≥ +0.005 且残差符号与预注册（或实测重登记）方向一致 → 族通过（成员维持 promotion_allowed=True）；否则负结果：保留计算与采样、promotion_allowed=False（p13 §8 先例）。
- 相关性：|ρ|≥0.9 → Part A 机制裁决；F2-2 方向按测量裁决（主假设 −1 证伪 → 实测方向重登记留档，§13 双判据纪律）。
- 负结果保留全部证据；禁止因搜索表现不好继续扩库。

## 8. 资源上限与停止条件

- 重测单次 ≤30 分钟（参考口径；F1 实测 2.2 小时——如实记录，超上限需说明）。
- 停止：任一 RED 无法稳定失败 → 回到本文修订；governance/captain 否决候选集 → 修订再批；门禁红 → 带证据上报。

## 9. 批准与实施边界

- 批准流程：governance-auditor 核对编号与登记（CONTRACTS.md 同 commit）+ **用户批准**（t27 任务书门禁："经用户批准"）→ factor-engineer 实施（RED → 最小实现 → GREEN → 全量 → 重测 → 裁决 → 测量日志）。
- 本草案不授权 push/PR/merge。
