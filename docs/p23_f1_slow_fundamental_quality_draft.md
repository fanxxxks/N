# P23 草案 v5：F1 批次预注册——慢基本面质量四因子（用户亲批集，captain 纠正确认版）

- 状态：**APPROVED & IMPLEMENTED — 用户原文「批准」（效力范围 = v1 表格四因子，用户未见过 v2 重写）+ captain 纠正确认（2026-09-07：「你的纠正成立，我的『v2 覆盖』裁定错误」）**——本版 = 用户亲批四因子（v1 集）。批准后本文保持前瞻性，禁止以实现/测量输出反向修改。
- **captain 纠正确认（逐字）**：「你的纠正成立，我的『v2 覆盖』裁定错误。用户「批准」的效力范围 = 我呈递的 v1 表格（NET_MARGIN_TREND_4Q/LEVERAGE_TREND_4Q/CAPEX_INTENSITY/CFO_TO_DEBT）；v2 重写发生在批准之后、用户未见过。你按 AGENTS §1 回滚至用户亲批集合并实施——处置正确。我的裁定错误正式记录（这是你第二次抓住我的裁决/摘要缺陷：keep-list B→A、本次批准范围——你的纪律一直在校准我）。」
- 证据基线：t22 修复后重算 `docs/factor_inventory_audit_v2_fix2_20260906/metrics.json`（dataset `157e0cf5…`，FACTOR_COMPUTE 4 / FEATURE_REGISTRY 9 / GRAMMAR 7；全宇宙有效横截面 0.995）。勘察基线 commit：codex/factor-semantic-fixes @ `6e11b76`。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。任务路由：**研究语义变更**（AGENTS §3.2：新增特征）。

## 1. 批次依据与问题陈述

词表 77 名中慢基本面（slow_fundamental 域，16 名）的质量经济学缺少**资产收益率（Novy-Marx 系）**维度：既有 ROE/ROA（legacy 静态水平读）、family-⑤（现金流确认/应计/资产存量增长/盈利加速度）之外，**经营现金流对资产的比率、毛利对资产的比率、利润总额对资产的比率**（资产调整后的盈利能力三截面）与**杠杆的期间变化**（DEBT_RATIO 只存水平，方向缺位）均无词表表达。本批按 t24 任务书预注册 4 个候选，全部只用 p17 §5.3 已落库的 v2 三表字段（零 schema 变更），复用 t22 已修复的单一路径（`asof_winner_intervals` + `_ttm`/单期转换）。

## 2. 候选因子（预注册公式；全部 slow_fundamental 域；Tier B）

记号：`TTM(x)_t` = 截至 t 报告期的滚动四季和（`_ttm` 单一路径）；`avg_assets_t` = mean(total_assets_t, total_assets_{t−4 报告期})（两端均 winner-resolved as-of t，经典平均资产口径）；`lev_t` = total_liab_t / total_assets_t（同批 winner-resolved）；可见性一律经 `asof_winner_intervals`（p17 §6.2 修订追踪），分母/分子各取 winner-resolved 值、较晚可见。

| # | 特征 | 预注册公式 | 经济假设 | 方向 | NaN 守卫 |
|---|---|---|---|---|---|
| F1-1 | NET_MARGIN_TREND_4Q | `(TTM(n_income_attr_p)/TTM(revenue))_t − (同式)_{t−4 报告期}` | 净利率 4 报告期改善（定价权/成本控制）延续为正收益；恶化惩罚 | **+1** | 任一 TTM 断档或 revenue ≤0 → NaN |
| F1-2 | LEVERAGE_TREND_4Q | `lev_t − lev_{t−4 报告期}` | 杠杆 4 报告期恶化（被动加杠杆）→ 困境风险上升 | **−1** | 任一期 lev 缺失或 total_assets ≤0 → NaN |
| F1-3 | CAPEX_INTENSITY | `TTM(−n_cashflow_inv_act)_t / total_assets_t` | 投资/资本开支强度高 → 过度投资与代理成本 → 后续收益转负（投资因子流量面，与 ASSET_GROWTH 的存量面互补） | **−1** | total_assets ≤0 或投资现金流缺失 → NaN（n_cashflow_inv_act 通常 ≤0，取负） |
| F1-4 | CFO_TO_DEBT | `TTM(n_cashflow_act)_t / total_liab_t` | 经营现金流对总负债覆盖高 → 偿债质量与财务弹性 | **+1** | total_liab ≤0 或 TTM 断档 → NaN |

- **与既有词表的非重复性论证（v5）**：NET_MARGIN（legacy，静态水平）≠ F1-1 的 4 报告期趋势；DEBT_RATIO（legacy，静态水平）≠ F1-2 的方向变化量；ASSET_GROWTH（family-⑤，存量变化）≠ F1-3 的流量面强度；CASHFLOW_QUALITY（family-⑤，cfo/profit）≠ F1-4 的 cfo/负债。预注册冗余预检：实施后重测若与上述任何成员 |ρ|≥0.9 → 按族内精简裁决（p9 §7.2 机制）。
- **族原子性**：4 特征同 commit 进词表、同批裁决、同进退（p13 §8 先例）。
- **路线图映射记录（v2 时代的映射表保留为历史）**：v2 曾按 t24 任务书建议集采纳路线图四因子（CFO_TO_ASSETS 等）——v5 恢复用户亲批集（v1 四因子）；路线图指名集（CFO_TO_ASSETS/GROSS_PROFITABILITY/OPERATING_PROFITABILITY/LEVERAGE_CHANGE）保留为未来批次首选候选（需独立预注册提请用户批准）。

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

## 5. 版本影响（预注册；以实施时点现值重算——v5 实际链）

| 版本 | 旧（现值） | 新（实际） | 理由 |
|---|---|---|---|
| GRAMMAR（vocab.py） | 8 | 9 | F1 名称集按用户亲批集调换（数量 81 不变，id 78-81 名称变更 = 布局变更 → bump）；generation 9 注册 remap 分派。（历史：V1 实施 c132ae7 曾 7→8，revert 后 V2 实施 eab1061 再 7→8，本次 v5 调换 8→9——每次名称集变更各 bump 一次） |
| FEATURE_REGISTRY | 13 | 14 | 4 特征元数据调换。（历史链：t70 8→9 → t24 路由 9→10 → p22 翻转 10→11 → V1 实施 11→12 → 翻转回退 12→13 → 本 v5 调换 13→14——单调无跳号） |
| RESEARCH_DOMAIN | 3 | 4 | slow_fundamental 16→20 |
| FACTOR_COMPUTE | 4 | 不变 | 纯特征追加，既有行计算语义零改动（p13 先例） |
| DATA_TIER | 2 | 不变 | PIT_FUNDAMENTAL→B 既有映射 |
| dataset_id | 157e0cf5 | 157e0cf5（不变） | 零 DB 写入 |

- 旧 artifact：GRAMMAR 9 注册进 remap 分派（数量布局同 8，by-name 解码不变）；deprecated 集（13 名，含 TURNOVER_MA20）维持退出采样。

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

- v1（2026-09-07）：四个自提候选（NET_MARGIN_TREND_4Q / LEVERAGE_TREND_4Q / CAPEX_INTENSITY / CFO_TO_DEBT）——governance 核验通过（7dd4b6b）并经用户亲批（captain 呈递的表格 = 本集）。
- v2（2026-09-07）：按 t24 任务书建议集重写（CFO_TO_ASSETS / GROSS_PROFITABILITY / OPERATING_PROFITABILITY / LEVERAGE_CHANGE）——**用户未见过此重写**。
- v3（2026-09-07）：按"用户亲批 v1 集"回滚并实施（c132ae7）——其后 captain"v2 覆盖"裁定（后自认错误）导致 revert（24520cb）并按 v2 集实施（eab1061）。
- v4（2026-09-07，已废弃）：v2 集实施态（eab1061）。
- v5（本版）：**captain 纠正确认**（"你的纠正成立，我的『v2 覆盖』裁定错误。用户「批准」的效力范围 = 我呈递的 v1 表格……你按 AGENTS §1 回滚至用户亲批集合并实施——处置正确"）——**恢复用户亲批集（v1 四因子）**，v2 路线图指名集保留为未来批次首选候选（需独立预注册提请用户批准）。
