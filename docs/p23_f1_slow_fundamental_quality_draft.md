# P23 草案 v3：F1 批次预注册——慢基本面质量四因子（用户批准版）

- 状态：**APPROVED — 用户原文「批准」（2026-09-07，经 captain 转达；governance 核验 @ 176241e 先行）**——本版 = 用户批准的候选集（v1 四因子），取代 v2 的 t24 建议集。批准后本文保持前瞻性，禁止以实现/测量输出反向修改。
- 批准记录：captain 转达「F1 批次批准送达……用户原文『批准』——p23 四候选（NET_MARGIN_TREND_4Q +1 / LEVERAGE_TREND_4Q −1 / CAPEX_INTENSITY −1 / CFO_TO_DEBT +1）全部批准（governance 核验通过的新特征批次，p18 §3.2 保留项经用户亲批）」；governance 核验 = 提交 7dd4b6b（docs/p21_p22_f_batch_registration_verification.md）。
- 证据基线：t22 修复后重算 `docs/factor_inventory_audit_v2_fix2_20260906/metrics.json`（dataset `157e0cf5…`，FACTOR_COMPUTE 4 / FEATURE_REGISTRY 11 / GRAMMAR 7 / RESEARCH_DOMAIN 3；全宇宙有效横截面 0.995）。勘察基线 commit：codex/factor-semantic-fixes @ `a09c02f`。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。任务路由：**研究语义变更**（AGENTS §3.2：新增特征）。

## 1. 批次依据与问题陈述

词表 77 名中慢基本面（slow_fundamental 域，16 名）的质量经济学缺口：盈利能力的**期间趋势**（净利率 4 报告期变化）、**杠杆的期间变化**（负债率水平已在词表，方向缺位）、**投资强度**（流量面资本开支）、**偿债质量**（经营现金流对负债的覆盖）均无词表表达。本批按用户批准的 v1 候选集预注册 4 个新特征，全部只用 p17 §5.3 已落库的 v2 三表字段（零 schema 变更），复用 t22 已修复的单一路径（`asof_winner_intervals` + `_ttm`/单期转换）。

## 2. 候选因子（预注册公式；全部 slow_fundamental 域；Tier B；用户批准集）

记号：`TTM(x)_t` = 截至 t 报告期的滚动四季和（`_ttm` 单一路径）；`lev_t` = total_liab_t / total_assets_t（winner-resolved）；`Δ4q(x)_t` = x_t − x_{t−4 报告期}；可见性一律经 `asof_winner_intervals`（p17 §6.2 修订追踪），分母/分子各取 winner-resolved 值、较晚可见。

| # | 特征 | 预注册公式 | 经济假设 | 方向 | NaN 守卫 |
|---|---|---|---|---|---|
| F1-1 | NET_MARGIN_TREND_4Q | `(TTM(n_income_attr_p)/TTM(revenue))_t − (同式)_{t−4 报告期}` | 净利率 4 报告期改善（定价权/成本控制）延续为正收益；恶化惩罚 | **+1** | 任一 TTM 断档或 revenue ≤0 → NaN |
| F1-2 | LEVERAGE_TREND_4Q | `lev_t − lev_{t−4 报告期}` | 杠杆 4 报告期恶化（被动加杠杆）→ 困境风险上升 | **−1** | 任一期 lev 缺失或 total_assets ≤0 → NaN |
| F1-3 | CAPEX_INTENSITY | `TTM(−n_cashflow_inv_act)_t / total_assets_t` | 投资/资本开支强度高 → 过度投资与代理成本 → 后续收益转负（投资因子流量面，与 ASSET_GROWTH 的存量面互补） | **−1** | total_assets ≤0 或投资现金流缺失 → NaN（n_cashflow_inv_act 通常 ≤0，取负） |
| F1-4 | CFO_TO_DEBT | `TTM(n_cashflow_act)_t / total_liab_t` | 经营现金流对总负债覆盖高 → 偿债质量与财务弹性 | **+1** | total_liab ≤0 或 TTM 断档 → NaN |

- **与既有词表的非重复性论证**：NET_MARGIN（legacy，静态水平）≠ F1-1 的 4 报告期趋势；DEBT_RATIO（legacy，静态水平）≠ F1-2 的方向变化量；ASSET_GROWTH（family-⑤，存量变化）≠ F1-3 的流量面强度；CASHFLOW_QUALITY（family-⑤，cfo/profit）≠ F1-4 的 cfo/负债。预注册冗余预检：实施后重测若与上述任何成员 |ρ|≥0.9 → 按族内精简裁决（p9 §7.2 机制）。
- **v2 建议集处置记录**：t24 任务书的建议公式（CFO_TO_ASSETS/GROSS_PROFITABILITY/OPERATING_PROFITABILITY/LEVERAGE_CHANGE）与本批准集并存于 t24 任务描述，用户亲批的是本版四因子——建议集保留为未来批次候选（需新预注册）。
- **族原子性**：4 特征同 commit 进词表、同批裁决、同进退（p13 §8 先例）。

## 3. 每因子预注册记录（t24 验收逐项）

| 维度 | 记录 |
|---|---|
| 经济假设 | 见 §2 表（净利率趋势 + / 杠杆恶化 − / 投资强度 − / 偿债质量 +） |
| 数据源 | v2 `income_statement_pit`（n_income_attr_p/revenue）、`balance_sheet_pit`（total_assets/total_liab）、`cashflow_statement_pit`（n_cashflow_act/n_cashflow_inv_act）——全部经 `asof_winner_intervals` 单一路径 |
| PIT 可用时间 | p17 §6.2：候选 = available_date ≤ s；胜者字典序；公告日次日可见；修订按 winner 时间线精确反映 |
| Tier | B（PIT_FUNDAMENTAL，既有映射） |
| 预期方向 | +1/−1/−1/+1（用户批准集预注册，按测量裁决；证伪即如实降级） |
| 预测周期 | slow（every_20_days，slow_fundamental 域） |
| 覆盖率 | 预期 ≈ family-⑤ 量级（0.9+；TTM 断档按 NaN）；重测实测为准 |
| 换手/成本 | slow 域低换手（family-⑤ 先例 tau_h10 ≈0.02-0.09）；成本后空间按 ΔIC 与成本拖累联合评估 |
| 相关簇 | 预检 vs NET_MARGIN/DEBT_RATIO/ASSET_GROWTH/CASHFLOW_QUALITY/family-⑤；|ρ|≥0.9 → 族内精简 |
| 增量 OOS 计划 | ΔIC_OOS（h=10，|w_b| 修正基线 per p9 附录 B）≥ +0.005 且残差符号同向 → 族门通过 |

## 4. 范围与非目标

- **范围**：`ashare_data/fundamentals.py` 新增 4 帧（`_family5_asof_frames` 同构路径扩展）；`ashare_model/{vocab,feature_metadata,feature_registry,research_domain}.py` 登记追加；`docs/feature_registry.md` 再生。
- **非目标**：不改既有特征公式/源；不动 `financial_indicator_audit`（p17 §5.4 禁入因子路径）；不动 family-⑤ 与 11 legacy 特征；不做 paper/lifecycle 转换；零 DB 写入（三表已含全部字段，dataset 不变）。

## 5. 版本影响（预注册；以实施时点现值重算）

| 版本 | 旧（现值） | 新 | 理由 |
|---|---|---|---|
| GRAMMAR（vocab.py） | 7 | 8 | 词表 77→81（追加，token id 不移位）；generation 8 注册 remap 分派（布局 = 7 布局追加） |
| FEATURE_REGISTRY | 11 | 12 | 4 特征元数据 + depends_on |
| RESEARCH_DOMAIN | 3 | 4 | slow_fundamental 16→20 |
| FACTOR_COMPUTE | 4 | 不变 | 纯特征追加，既有行计算语义零改动（p13 先例） |
| DATA_TIER | 2 | 不变 | PIT_FUNDAMENTAL→B 既有映射 |
| dataset_id | 157e0cf5 | 157e0cf5（不变） | 零 DB 写入 |

- 旧 artifact：GRAMMAR 8 注册进 remap 分派（布局 = 7 布局追加，by-name 解码不变）；deprecated 集（13 名，含 TURNOVER_MA20）维持退出采样。
- 版本沿革披露：captain 批准文中 F1 版本算术（GRAMMAR 8→9 / FEATURE_REGISTRY 9→10）以 Part A 基线 7 为前提；实际 Part A 基线为 6（GRAMMAR 6→7），且 FEATURE_REGISTRY 因 t70（8→9）与 t24 路由负结果翻转（9→10）先行消费——本表按实施时点现值重算（GRAMMAR 7→8、FEATURE_REGISTRY 11→12），单调无跳号。

## 6. 预期 RED 测试（实现前先红）

1. 合成三表上四公式逐值断言（TTM 断档 NaN、分母守卫、Δ4q 对齐、winner-revision 反映）。
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

- 重测单次 ≤30 分钟（参考口径；同 p13 §9.1；44× as-of 优化后实际预计分钟级）。
- 停止：任一 RED 无法稳定失败 → 回到本文修订；governance/captain 否决候选集 → 修订再批；门禁红 → 带证据上报。

## 9. 批准与实施边界

- 批准流程：**用户原文批准已到达**（见批准记录）+ governance 核验在册（7dd4b6b）→ factor-engineer 实施（RED → 最小实现 → GREEN → 全量 → 重测 → 裁决 → 测量日志）；CONTRACTS.md 索引行同 commit 登记。
- 本草案不授权 push/PR/merge。

## 10. 草案沿革

- v1（2026-09-07）：四个自提候选（NET_MARGIN_TREND_4Q / LEVERAGE_TREND_4Q / CAPEX_INTENSITY / CFO_TO_DEBT）——governance 核验通过（7dd4b6b）并经用户亲批。
- v2（2026-09-07，已废弃）：按 t24 任务书建议集重写（CFO_TO_ASSETS / GROSS_PROFITABILITY / OPERATING_PROFITABILITY / LEVERAGE_CHANGE）——用户批准先于该重写到达，重写未被批准链覆盖。
- v3（本版）：**回滚至用户批准的 v1 候选集**，补记批准记录与版本沿革；v2 的建议集保留为未来批次候选（需新预注册）。
