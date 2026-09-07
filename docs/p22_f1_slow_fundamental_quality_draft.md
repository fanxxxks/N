# P22 草案 v2：F1 批次预注册——慢基本面资产收益率与杠杆趋势四因子

- 状态：**DRAFT v2 — 待 governance-auditor 核对编号与登记 + 用户批准后实施**（v1 草案的四个自提候选被 t24 任务书的预注册候选集取代；起草：factor-engineer，2026-09-07）。批准后本文保持前瞻性，禁止以实现/测量输出反向修改。
- 证据基线：t22 修复后重算 `docs/factor_inventory_audit_v2_fix2_20260906/metrics.json`（dataset `157e0cf5…`，FACTOR_COMPUTE 4 / FEATURE_REGISTRY 9 / GRAMMAR 7；全宇宙有效横截面 0.995）。勘察基线 commit：codex/factor-semantic-fixes @ `6e11b76`。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。任务路由：**研究语义变更**（AGENTS §3.2：新增特征）。

## 1. 批次依据与问题陈述

词表 77 名中慢基本面（slow_fundamental 域，16 名）的质量经济学缺少**资产收益率（Novy-Marx 系）**维度：既有 ROE/ROA（legacy 静态水平读）、family-⑤（现金流确认/应计/资产存量增长/盈利加速度）之外，**经营现金流对资产的比率、毛利对资产的比率、利润总额对资产的比率**（资产调整后的盈利能力三截面）与**杠杆的期间变化**（DEBT_RATIO 只存水平，方向缺位）均无词表表达。本批按 t24 任务书预注册 4 个候选，全部只用 p17 §5.3 已落库的 v2 三表字段（零 schema 变更），复用 t22 已修复的单一路径（`asof_winner_intervals` + `_ttm`/单期转换）。

## 2. 候选因子（预注册公式；全部 slow_fundamental 域；Tier B）

记号：`TTM(x)_t` = 截至 t 报告期的滚动四季和（`_ttm` 单一路径）；`avg_assets_t` = mean(total_assets_t, total_assets_{t−4 报告期})（两端均 winner-resolved as-of t，经典平均资产口径）；`lev_t` = total_liab_t / total_assets_t（同批 winner-resolved）；可见性一律经 `asof_winner_intervals`（p17 §6.2 修订追踪），分母/分子各取 winner-resolved 值、较晚可见。

| # | 特征 | 预注册公式 | 经济假设 | 方向 | NaN 守卫 |
|---|---|---|---|---|---|
| F1-1 | CFO_TO_ASSETS | `TTM(n_cashflow_act)_t / avg_assets_t` | 经营现金流对资产的转化率（应计质量之外的现金创造强度）高 → 盈利质量与财务弹性 | **+1** | TTM 断档或 avg_assets ≤0/缺任一端 → NaN |
| F1-2 | GROSS_PROFITABILITY | `(TTM(revenue)_t − TTM(oper_cost)_t) / avg_assets_t` | 毛利/资产比（Novy-Marx 毛利盈利能力）：定价权与成本结构的质量截面 | **+1** | 同上；revenue 或 oper_cost 缺失 → NaN |
| F1-3 | OPERATING_PROFITABILITY | `TTM(total_profit)_t / avg_assets_t` | 利润总额/资产比（综合盈利能力） | **+1** | 同上 |
| F1-4 | LEVERAGE_CHANGE | `lev_t − lev_{t−4 报告期}` | 杠杆 4 报告期恶化（被动加杠杆）→ 困境风险 | **−1** | 任一期 lev 缺失或 assets ≤0 → NaN |

- **口径披露（p17 schema 缺口预注册）**：`income_statement_pit` 无 `operate_profit` 列（p17 §5.3 DDL）——F1-3 的"营业利润"以 `total_profit`（利润总额，含营业外收支）为预注册代理；若未来 p17 增补 `operate_profit` 列，须经勘误后重算。此为本批唯一的口径近似，明文预注册而非静默替代。
- **与既有词表的非重复性论证**：ROA（legacy，净利/总资产静态）≠ 本批三比率的现金流/毛利/利润总额分子与平均资产分母口径；DEBT_RATIO（legacy，静态水平）≠ F1-4 的方向变化量；family-⑤ 的现金流确认（cfo/profit）≠ cfo/assets。预注册冗余预检：实施后重测若与 ROA/DEBT_RATIO/family-⑤ 任何成员 |ρ|≥0.9 → 按族内精简裁决（p9 §7.2 机制）。
- **族原子性**：4 特征同 commit 进词表、同批裁决、同进退（p13 §8 先例）。
- **CAPEX_INTENSITY 处置补充（captain 裁决 2b）**：v2 候选集 = 路线图四因子逐字采纳，v1 自提的 CAPEX_INTENSITY 已在 v2 重写中剔除（上表末行说明）——不存在无说明替换。

- **路线图映射表（captain 裁决 2a：v1 自提候选 → 路线图 F1 指名单逐个对应）**：

| v1 草案候选 | 路线图 F1 指名单 | 对应裁决 | 理由 |
|---|---|---|---|
| CFO_TO_DEBT（cfo/total_liab） | **CFO_TO_ASSETS**（cfo/平均总资产） | 改分母采纳路线图 | 分母选总资产（非总负债）理由：(1) 与 GROSS/OPERATING 两比率同分母（avg_assets），族内截面可比；(2) 资产口径覆盖全部资本结构（负债+权益），负债分母只覆盖债权人视角；(3) 文献标准（Novy-Marx 系均以资产为分母）。负债口径的未来提案走新预注册 |
| （无 v1 对应——v1 以 NET_MARGIN_TREND_4Q 提名） | **GROSS_PROFITABILITY**（毛利/平均总资产） | 采纳路线图 | 毛利/资产比（Novy-Marx 2013）是文献最强截面之一；v1 的净利率趋势概念由 F1-3 的利润总额口径部分承载，趋势变体可未来提案 |
| （无 v1 对应） | **OPERATING_PROFITABILITY**（利润总额/平均总资产） | 采纳路线图 | p17 DDL 无 operate_profit 列 → 预注册代理 = total_profit（利润总额，含营业外收支）——口径缺口已在 §2 明文披露 |
| LEVERAGE_TREND_4Q（负债率 4 报告期变化） | **LEVERAGE_CHANGE**（负债率_t − 负债率_{t−4q}） | 同一概念，公式逐字一致 | ✓ 直接映射 |
| CAPEX_INTENSITY（投资现金流/总资产） | （路线图之外） | **v2 已剔除** | 路线图指名单为四因子，v1 自提的第五概念无位； captain 裁决 2b：不許无说明替换——本行即说明：v2 按路线图集合执行，CAPEX_INTENSITY 保留为未来批次候选（需新预注册） |

## 3. 每因子预注册记录（t24 验收逐项）

| 维度 | 记录 |
|---|---|
| 经济假设 | 见 §2 表（现金流创造强度 + / 毛利盈利能力 + / 综合盈利能力 + / 杠杆恶化 −） |
| 数据源 | v2 `income_statement_pit`（revenue/oper_cost/total_profit）、`balance_sheet_pit`（total_assets/total_liab）、`cashflow_statement_pit`（n_cashflow_act）——全部经 `asof_winner_intervals` 单一路径 |
| PIT 可用时间 | p17 §6.2：候选 = available_date ≤ s；胜者字典序；公告日次日可见；修订按 winner 时间线精确反映 |
| Tier | B（PIT_FUNDAMENTAL，既有映射） |
| 预期方向 | +1/+1/+1/−1（预注册，按测量裁决；证伪即如实降级） |
| 预测周期 | slow（every_20_days，slow_fundamental 域） |
| 覆盖率 | 预期 ≈ family-⑤ 量级（0.9+；TTM 断档按 NaN）；重测实测为准 |
| 换手/成本 | slow 域低换手（family-⑤ 先例 tau_h10 ≈0.02-0.09）；成本后空间按 ΔIC 与成本拖累联合评估 |
| 相关簇 | 预检 vs ROA/DEBT_RATIO/NET_MARGIN/family-⑤；|ρ|≥0.9 → 族内精简 |
| 增量 OOS 计划 | ΔIC_OOS（h=10，|w_b| 修正基线 per p9 附录 B）≥ +0.005 且残差符号同向 → 族门通过 |

## 4. 范围与非目标

- **范围**：`ashare_data/fundamentals.py` 新增 4 帧（`_family5_asof_frames` 同构路径扩展）；`ashare_model/{vocab,feature_metadata,feature_registry,research_domain}.py` 登记追加；`docs/feature_registry.md` 再生。
- **非目标**：不改既有特征公式/源；不动 `financial_indicator_audit`（p17 §5.4 禁入因子路径）；不动 family-⑤ 与 11 legacy 特征；不做 paper/lifecycle 转换；零 DB 写入（三表已含全部字段，dataset 不变）。

## 5. 版本影响（预注册）

| 版本 | 旧 | 新 | 理由 |
|---|---|---|---|
| GRAMMAR（vocab.py） | 7 | 8 | 词表 77→81（追加，token id 不移位）；generation 8 注册 remap 分派（布局 = 7 布局追加）。（captain 裁决 2c 原文为 8→9——其 Part A 基线假设 7；实际 Part A 基线为 6，落地 6→7，故 F1 为 7→8；语义内容一致） |
| FEATURE_REGISTRY | 10 | 11 | 4 特征元数据 + depends_on。（captain 裁决 2c 原文为 9→10——未计入 INDUSTRY_MOMENTUM 负结果翻转的 9→10；F1 按其实际基线 10 顺延为 10→11） |
| RESEARCH_DOMAIN | 3 | 4 | slow_fundamental 16→20 |
| FACTOR_COMPUTE | 4 | 不变 | 纯特征追加，既有行计算语义零改动（p13 先例） |
| DATA_TIER | 2 | 不变 | PIT_FUNDAMENTAL→B 既有映射 |
| dataset_id | 157e0cf5 | 157e0cf5（不变） | 零 DB 写入 |

- 旧 artifact：GRAMMAR 8 注册进 remap 分派（布局 = 7 布局追加，by-name 解码不变）；deprecated 集（13 名，含 TURNOVER_MA20）维持退出采样。

## 6. 预期 RED 测试（实现前先红）

1. 合成三表上四公式逐值断言（TTM 断档 NaN、avg_assets 口径、分母守卫、Δ4q 对齐、winner-revision 反映）。
2. as-of 可见性：公告日前不可见、修订按 winner 时间线。
3. 词表 77→81 追加不移位（既有 token 往返 parity）；GRAMMAR 8 钉值 + generation 8 分派注册。
4. 域归属 slow_fundamental ×4；RESEARCH_DOMAIN 4 钉值。
5. 未回填态 fail-closed：新特征帧 NaN → 质量门拒绝。

## 7. 测量方案与裁决（预注册）

- 实现后按 t18/t22 同口径全量重测（81 名；h∈{1,2,3,5,10,15,20} 非重叠 IC、min_stocks=10、IS/OOS 2022 分界、|w_b| 修正基线 per p9 附录 B、同成本模型、五 folds、方向/PIT/Tier 审计）。
- **族门（4 特征整族原子）**：7 信号等权复合基线（|w_b| 修正语义），族内 ≥1 成员 ΔIC_OOS ≥ +0.005 且残差符号与预注册方向一致 → 族通过（成员维持 promotion_allowed=True）；否则负结果：保留计算与采样、promotion_allowed=False（p13 §8 先例）。
- 相关性：|ρ|≥0.9 → Part A 机制裁决；coverage=0 因子的 ΔIC 腿按附录 B 记 not-evaluable。
- 方向按测量裁决（P9 §5.4 惯例）；证伪即如实降级，禁止单方改向。
- 负结果保留全部证据；禁止因搜索表现不好继续扩库。

## 8. 资源上限与停止条件

- 重测单次 ≤30 分钟（参考口径；同 p13 §9.1）。
- 停止：任一 RED 无法稳定失败 → 回到本文修订；governance/captain 否决候选集 → 修订再批；门禁红 → 带证据上报。

## 9. 批准与实施边界

- 批准流程：governance-auditor 核对编号与登记（CONTRACTS.md 同 commit）+ **用户批准**（t24 任务书门禁）→ factor-engineer 实施（RED → 最小实现 → GREEN → 全量 → 重测 → 裁决 → 测量日志）。
- 本草案不授权 push/PR/merge。

## 10. 草案沿革

- v1（2026-09-07，已废弃）：四个自提候选（NET_MARGIN_TREND_4Q / LEVERAGE_TREND_4Q / CAPEX_INTENSITY / CFO_TO_DEBT）——被 t24 任务书的预注册候选集取代（其中 LEVERAGE_TREND 概念演化为 F1-4 LEVERAGE_CHANGE，其余三个资产收益率候选按 Novy-Marx 系口径重定义）。
- v2（本版）：t24 任务书候选集。
